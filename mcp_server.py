#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async/Concurrent optimized MCP server for CodeMem.

Optimizations:
- Async I/O with asyncio
- Concurrent request handling
- Parallel index building with multiprocessing
- Non-blocking database operations with aiosqlite
- Connection pooling
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import aiosqlite
from rank_bm25 import BM25Okapi
import tiktoken

sys.path.append(str(Path(__file__).parent))

from unified_history import collect_files, load_records, to_df
from export_sessions_md import export_sessions
from intent_recognition import parse_intent, QueryIntent, expand_synonyms
from nl_formatter import (
    format_search_results,
    format_activity_summary,
    format_error_response,
    format_context_response
)
from context_manager import (
    ContextManager,
    ConversationContext,
    SearchResult,
    resolve_reference,
    is_followup_query
)


# Initialize tiktoken encoder
try:
    _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken_encoder = None


MD_SESSIONS_DIR = Path.home() / ".codemem" / "md_sessions"

# Global state
_db_ready = asyncio.Event()
_db_build_error: Optional[str] = None
_db_path: Optional[Path] = None

# BM25 indexes (shared across async tasks)
_bm25_index = None
_bm25_docs = []
_bm25_metadata = []
_bm25_md_index = None
_bm25_md_docs = []
_bm25_md_metadata = []
_bm25_lock = asyncio.Lock()

# Query cache with async lock
_query_cache: Dict[str, Tuple[Any, float]] = {}
_cache_lock = asyncio.Lock()
_cache_max_size = 100
_cache_ttl = 3600  # 1 hour

# Connection pool
_db_pool: Optional[aiosqlite.Connection] = None
_pool_lock = asyncio.Lock()

# Context manager (Phase 2)
_context_manager: Optional[ContextManager] = None


def smart_tokenize(text: str) -> List[str]:
    """Smart tokenization using tiktoken."""
    if not text:
        return []

    if _tiktoken_encoder is not None:
        try:
            token_ids = _tiktoken_encoder.encode(text.lower())
            return [str(tid) for tid in token_ids]
        except Exception:
            pass

    return text.lower().split()


async def get_db_connection() -> aiosqlite.Connection:
    """Get database connection from pool."""
    global _db_pool

    async with _pool_lock:
        if _db_pool is None:
            _db_pool = await aiosqlite.connect(str(_db_path))
            _db_pool.row_factory = aiosqlite.Row
        return _db_pool


async def get_from_cache(key: str) -> Optional[Any]:
    """Get value from cache (async)."""
    async with _cache_lock:
        if key in _query_cache:
            result, timestamp = _query_cache[key]
            if time.time() - timestamp < _cache_ttl:
                return result
            else:
                del _query_cache[key]
    return None


async def put_to_cache(key: str, value: Any):
    """Put value to cache (async)."""
    async with _cache_lock:
        if len(_query_cache) >= _cache_max_size:
            # LRU eviction
            oldest_key = min(_query_cache.keys(), key=lambda k: _query_cache[k][1])
            del _query_cache[oldest_key]
        _query_cache[key] = (value, time.time())


def cache_key(tool: str, *args, **kwargs) -> str:
    """Generate cache key."""
    key_str = f"{tool}:{json.dumps(args)}:{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()


def build_bm25_index_sync(db_path: Path) -> Tuple[Any, List, List]:
    """Build BM25 index (runs in process pool)."""
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_id, timestamp, role, text, session_id, platform
            FROM events
            WHERE text IS NOT NULL AND text != ''
            ORDER BY timestamp DESC
            LIMIT 10000
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None, [], []

        docs = []
        metadata = []

        for event_id, timestamp, role, text, session_id, platform in rows:
            tokens = smart_tokenize(text)
            docs.append(tokens)
            metadata.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "role": role,
                "text": text,
                "session_id": session_id,
                "platform": platform
            })

        index = BM25Okapi(docs)
        return index, docs, metadata

    except Exception:
        return None, [], []


def build_bm25_md_index_sync(md_sessions_dir: Path) -> Tuple[Any, List, List]:
    """Build Markdown BM25 index (runs in process pool)."""
    try:
        if not md_sessions_dir.exists():
            return None, [], []

        md_files = list(md_sessions_dir.glob("*.md"))
        if not md_files:
            return None, [], []

        docs = []
        metadata = []

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                session_id = md_file.stem

                import re
                messages = re.split(r'\n### \[(.*?)\] (.*?)\n', content)

                for i in range(1, len(messages), 3):
                    if i + 2 < len(messages):
                        role = messages[i]
                        timestamp = messages[i + 1]
                        text = messages[i + 2].strip()

                        if text:
                            tokens = smart_tokenize(text)
                            docs.append(tokens)
                            metadata.append({
                                "session_id": session_id,
                                "timestamp": timestamp,
                                "role": role,
                                "text": text,
                                "source": "markdown",
                                "file": md_file.name
                            })
            except Exception:
                continue

        if docs:
            index = BM25Okapi(docs)
            return index, docs, metadata

        return None, [], []

    except Exception:
        return None, [], []


async def build_bm25_indexes_parallel():
    """Build both BM25 indexes in parallel using process pool."""
    global _bm25_index, _bm25_docs, _bm25_metadata
    global _bm25_md_index, _bm25_md_docs, _bm25_md_metadata

    loop = asyncio.get_event_loop()

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Build both indexes in parallel
        sql_future = loop.run_in_executor(executor, build_bm25_index_sync, _db_path)
        md_future = loop.run_in_executor(executor, build_bm25_md_index_sync, MD_SESSIONS_DIR)

        # Wait for both to complete
        sql_result, md_result = await asyncio.gather(sql_future, md_future)

        # Update global state
        async with _bm25_lock:
            _bm25_index, _bm25_docs, _bm25_metadata = sql_result
            _bm25_md_index, _bm25_md_docs, _bm25_md_metadata = md_result


async def bm25_search_async(query: str, limit: int = 20, source: str = "sql") -> Dict[str, Any]:
    """Async BM25 search."""
    # Check cache
    key = cache_key("bm25", query, limit=limit, source=source)
    cached = await get_from_cache(key)
    if cached is not None:
        return cached

    # Wait for DB ready
    await asyncio.wait_for(_db_ready.wait(), timeout=120.0)

    # Build indexes if needed
    async with _bm25_lock:
        if _bm25_index is None and _bm25_md_index is None:
            pass  # Will be built in background

    # Enforce limit
    limit = min(limit, 50)

    # Tokenize query
    query_tokens = smart_tokenize(query)

    results = []

    # Search SQL index
    if source in ("sql", "both") and _bm25_index is not None:
        async with _bm25_lock:
            scores = _bm25_index.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

            for idx in top_indices:
                if scores[idx] > 0:
                    meta = _bm25_metadata[idx]
                    results.append({
                        "event_id": meta.get("event_id"),
                        "timestamp": meta["timestamp"],
                        "role": meta["role"],
                        "text": meta["text"][:200] + "..." if len(meta["text"]) > 200 else meta["text"],
                        "session_id": meta["session_id"],
                        "platform": meta["platform"],
                        "score": float(scores[idx]),
                        "source": "sql"
                    })

    # Search Markdown index
    if source in ("markdown", "both") and _bm25_md_index is not None:
        async with _bm25_lock:
            md_scores = _bm25_md_index.get_scores(query_tokens)
            md_top_indices = sorted(range(len(md_scores)), key=lambda i: md_scores[i], reverse=True)[:limit]

            for idx in md_top_indices:
                if md_scores[idx] > 0:
                    meta = _bm25_md_metadata[idx]
                    results.append({
                        "timestamp": meta["timestamp"],
                        "role": meta["role"],
                        "text": meta["text"][:200] + "..." if len(meta["text"]) > 200 else meta["text"],
                        "session_id": meta["session_id"],
                        "file": meta["file"],
                        "score": float(md_scores[idx]),
                        "source": "markdown"
                    })

    # Sort combined results
    if source == "both":
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

    result = {
        "query": query,
        "count": len(results),
        "results": results,
        "source": source
    }

    # Cache result
    await put_to_cache(key, result)

    return result


async def get_recent_activity_async(days: int = 7) -> Dict[str, Any]:
    """Get recent activity (async)."""
    try:
        conn = await get_db_connection()
        cursor = await conn.execute("""
            SELECT session_id, platform, COUNT(*) as event_count,
                   MIN(timestamp) as first_seen, MAX(timestamp) as last_seen
            FROM events_raw
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY session_id, platform
            ORDER BY last_seen DESC
            LIMIT 20
        """, (days,))

        rows = await cursor.fetchall()

        sessions = []
        for row in rows:
            session_id = row[0]

            # Get sample messages
            sample_cursor = await conn.execute("""
                SELECT text FROM events
                WHERE session_id = ? AND role = 'user'
                LIMIT 3
            """, (session_id,))

            sample_rows = await sample_cursor.fetchall()
            sample_messages = [r[0][:100] for r in sample_rows if r[0]]

            sessions.append({
                "session_id": session_id,
                "platforms": row[1],
                "event_count": row[2],
                "first_seen": row[3],
                "last_seen": row[4],
                "sample_messages": sample_messages
            })

        return {
            "days": days,
            "session_count": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        return {"error": str(e)}


async def memory_query_async(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Intelligent memory query interface with natural language support.

    Phase 2 enhancements:
    - Conversation context management
    - Follow-up query support
    - Reference resolution (指代词解析)

    Args:
        query: Natural language query (e.g., "我之前讨论过 Python 异步吗？")
        context: Optional conversation context ID for follow-up queries

    Returns:
        Natural language response with summary, insights, and suggestions
        Also includes context_id for follow-up queries
    """
    global _context_manager

    try:
        # Wait for DB ready
        await asyncio.wait_for(_db_ready.wait(), timeout=120.0)

        if _db_build_error:
            return format_error_response(_db_build_error, query)

        # Initialize context manager if needed
        if _context_manager is None:
            _context_manager = ContextManager(max_contexts=100, ttl_seconds=1800)
            await _context_manager.start_cleanup_task()

        # Get or create conversation context
        ctx = await _context_manager.get_or_create(context)

        # Check if this is a follow-up query
        is_followup = is_followup_query(query)

        # Try to resolve references if follow-up
        resolved_ref = None
        if is_followup and ctx.query_history:
            resolved_ref = resolve_reference(query, ctx)

        # If reference resolved, handle accordingly
        if resolved_ref:
            ref_type = resolved_ref.get("type")

            if ref_type in ("rank", "code", "previous"):
                # Return detailed info about the referenced result
                result = resolved_ref.get("result")

                response = {
                    "summary": f"这是{resolved_ref.get('rank', '上一个')}结果的详细信息。",
                    "insights": [
                        f"会话 ID: {result.session_id}",
                        f"时间: {result.timestamp}",
                        f"角色: {result.role}",
                        f"相关度: {result.score:.2f}"
                    ],
                    "key_findings": [{
                        "text": result.text,
                        "session_id": result.session_id,
                        "item_index": result.item_index,
                        "source": result.source
                    }],
                    "suggestions": [
                        "需要查看完整上下文吗？",
                        "导出这次对话？"
                    ],
                    "metadata": {
                        "query": query,
                        "context_id": ctx.context_id,
                        "reference_resolved": True,
                        "reference_type": ref_type
                    }
                }

                # Add to context history
                ctx.add_query(query, [result])

                return response

            elif ref_type == "session":
                # Search within the focused session
                session_id = resolved_ref.get("session_id")
                search_results = await bm25_search_async(query, limit=20, source="both")

                # Filter results to focused session
                filtered_results = [
                    r for r in search_results.get("results", [])
                    if r.get("session_id") == session_id
                ]

                if filtered_results:
                    formatted = format_search_results(
                        query=query,
                        results=filtered_results,
                        source="both"
                    )
                    formatted["metadata"]["context_id"] = ctx.context_id
                    formatted["metadata"]["filtered_to_session"] = session_id

                    # Convert to SearchResult objects
                    search_result_objs = [
                        SearchResult(
                            session_id=r.get("session_id", ""),
                            timestamp=r.get("timestamp", ""),
                            role=r.get("role", ""),
                            text=r.get("text", ""),
                            score=r.get("score", 0.0),
                            source=r.get("source", ""),
                            item_index=r.get("item_index"),
                            event_id=r.get("event_id")
                        )
                        for r in filtered_results
                    ]
                    ctx.add_query(query, search_result_objs)

                    return formatted
                else:
                    return {
                        "summary": f"在会话 {session_id} 中没有找到相关内容。",
                        "insights": [],
                        "key_findings": [],
                        "suggestions": ["尝试搜索其他会话？"],
                        "metadata": {
                            "query": query,
                            "context_id": ctx.context_id,
                            "filtered_to_session": session_id
                        }
                    }

        # Parse intent (normal query flow)
        parsed = parse_intent(query)

        # Route to appropriate handler based on intent
        if parsed.intent == QueryIntent.SEARCH_CONTENT:
            # Expand keywords with synonyms
            expanded_keywords = expand_synonyms(parsed.keywords)
            search_query = " ".join(expanded_keywords) if expanded_keywords else query

            # Perform BM25 search
            search_results = await bm25_search_async(search_query, limit=20, source="both")

            # Format as natural language
            formatted = format_search_results(
                query=query,
                results=search_results.get("results", []),
                source=search_results.get("source", "both")
            )

            # Add context ID to metadata
            formatted["metadata"]["context_id"] = ctx.context_id

            # Convert to SearchResult objects and add to context
            search_result_objs = [
                SearchResult(
                    session_id=r.get("session_id", ""),
                    timestamp=r.get("timestamp", ""),
                    role=r.get("role", ""),
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    source=r.get("source", ""),
                    item_index=r.get("item_index"),
                    event_id=r.get("event_id")
                )
                for r in search_results.get("results", [])
            ]
            ctx.add_query(query, search_result_objs)

            return formatted

        elif parsed.intent == QueryIntent.ACTIVITY_SUMMARY:
            # Determine days from time range
            days = 7  # default
            if parsed.time_range:
                start_time, end_time = parsed.time_range
                days = (end_time - start_time).days or 1

            # Get activity data
            activity_data = await get_recent_activity_async(days=days)

            # Format as natural language
            formatted = format_activity_summary(activity_data)
            formatted["metadata"]["context_id"] = ctx.context_id

            # Add to context (no results for activity summary)
            ctx.add_query(query, [])

            return formatted

        elif parsed.intent == QueryIntent.GET_CONTEXT:
            # Try to extract session_id and item_index from keywords
            # For now, perform a search and return detailed context
            search_results = await bm25_search_async(query, limit=5, source="both")

            if search_results.get("results"):
                # Return first result with more context
                formatted = format_search_results(
                    query=query,
                    results=search_results.get("results", []),
                    source=search_results.get("source", "both")
                )
                formatted["metadata"]["context_id"] = ctx.context_id

                # Convert and add to context
                search_result_objs = [
                    SearchResult(
                        session_id=r.get("session_id", ""),
                        timestamp=r.get("timestamp", ""),
                        role=r.get("role", ""),
                        text=r.get("text", ""),
                        score=r.get("score", 0.0),
                        source=r.get("source", ""),
                        item_index=r.get("item_index"),
                        event_id=r.get("event_id")
                    )
                    for r in search_results.get("results", [])
                ]
                ctx.add_query(query, search_result_objs)

                return formatted
            else:
                response = format_error_response("未找到相关上下文", query)
                response["metadata"]["context_id"] = ctx.context_id
                ctx.add_query(query, [])
                return response

        elif parsed.intent == QueryIntent.FIND_SESSION:
            # Search for sessions with time filter
            search_query = " ".join(parsed.keywords) if parsed.keywords else query
            search_results = await bm25_search_async(search_query, limit=20, source="both")

            # Format with session grouping
            formatted = format_search_results(
                query=query,
                results=search_results.get("results", []),
                source=search_results.get("source", "both")
            )
            formatted["metadata"]["context_id"] = ctx.context_id

            # Convert and add to context
            search_result_objs = [
                SearchResult(
                    session_id=r.get("session_id", ""),
                    timestamp=r.get("timestamp", ""),
                    role=r.get("role", ""),
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    source=r.get("source", ""),
                    item_index=r.get("item_index"),
                    event_id=r.get("event_id")
                )
                for r in search_results.get("results", [])
            ]
            ctx.add_query(query, search_result_objs)

            return formatted

        elif parsed.intent == QueryIntent.EXPORT:
            # For now, guide user to use session.export tool
            response = {
                "summary": "导出功能需要指定会话 ID。",
                "insights": [
                    "请先搜索找到要导出的会话",
                    "然后使用 session.export 工具导出"
                ],
                "key_findings": [],
                "suggestions": [
                    "搜索相关对话以获取会话 ID",
                    "使用 session.list 查看最近的会话"
                ],
                "metadata": {
                    "query": query,
                    "intent": "export",
                    "context_id": ctx.context_id
                }
            }
            ctx.add_query(query, [])
            return response

        elif parsed.intent == QueryIntent.PATTERN_DISCOVERY:
            # Pattern discovery - future implementation
            # For now, return activity summary as a proxy
            activity_data = await get_recent_activity_async(days=30)
            response = format_activity_summary(activity_data)
            response["insights"].insert(0, "模式发现功能正在开发中，这里显示最近 30 天的活动摘要")
            response["metadata"]["context_id"] = ctx.context_id
            ctx.add_query(query, [])
            return response

        else:
            # UNKNOWN intent - try search as fallback
            search_results = await bm25_search_async(query, limit=20, source="both")

            if search_results.get("results"):
                formatted = format_search_results(
                    query=query,
                    results=search_results.get("results", []),
                    source=search_results.get("source", "both")
                )
                formatted["metadata"]["context_id"] = ctx.context_id

                # Convert and add to context
                search_result_objs = [
                    SearchResult(
                        session_id=r.get("session_id", ""),
                        timestamp=r.get("timestamp", ""),
                        role=r.get("role", ""),
                        text=r.get("text", ""),
                        score=r.get("score", 0.0),
                        source=r.get("source", ""),
                        item_index=r.get("item_index"),
                        event_id=r.get("event_id")
                    )
                    for r in search_results.get("results", [])
                ]
                ctx.add_query(query, search_result_objs)

                return formatted
            else:
                response = {
                    "summary": f"无法理解查询「{query}」。",
                    "insights": [
                        "尝试使用更具体的关键词",
                        "或者描述你想查找的内容"
                    ],
                    "key_findings": [],
                    "suggestions": [
                        "查看最近的活动：「最近在做什么？」",
                        "搜索特定话题：「讨论过 Python 吗？」",
                        "查找某个时间的对话：「上周的对话」"
                    ],
                    "metadata": {
                        "query": query,
                        "intent": "unknown",
                        "context_id": ctx.context_id
                    }
                }
                ctx.add_query(query, [])
                return response

    except asyncio.TimeoutError:
        return format_error_response("数据库初始化超时", query)
    except Exception as e:
        return format_error_response(str(e), query)


async def build_db_async(db_path: Path, include_history: bool, extra_roots: List[Path]):
    """Build database asynchronously."""
    global _db_build_error

    try:
        # Run CPU-intensive work in thread pool
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Collect files in parallel
            files_future = loop.run_in_executor(executor, collect_files, extra_roots)
            files = await files_future

            # Load records in parallel
            records_future = loop.run_in_executor(executor, load_records, files)
            records = await records_future

            # Convert to rows
            rows_future = loop.run_in_executor(executor, to_df, records)
            rows = await rows_future

        # Serialize JSON fields
        for row in rows:
            for col in ("tool_args", "tool_result", "raw_json"):
                if col in row:
                    v = row[col]
                    if isinstance(v, (dict, list)):
                        row[col] = json.dumps(v, ensure_ascii=False, default=str)
                    elif v is None:
                        row[col] = ""

        # Insert into database (async)
        conn = await aiosqlite.connect(str(db_path))

        if rows:
            columns = list(rows[0].keys())
            placeholders = ",".join(["?"] * len(columns))

            await conn.execute("DROP TABLE IF EXISTS events_raw")
            await conn.executemany(
                f"INSERT INTO events_raw ({','.join(columns)}) VALUES ({placeholders})",
                [tuple(row[col] for col in columns) for row in rows]
            )

        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_raw_time ON events_raw(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_raw_indexable ON events_raw(is_indexable)")

        # Create events view
        await conn.execute("DROP TABLE IF EXISTS events")
        await conn.execute("""
            CREATE TABLE events AS
            SELECT
              rowid as event_id,
              timestamp,
              role,
              index_text as text,
              session_id,
              turn_id,
              item_index,
              line_number,
              source_file,
              platform
            FROM events_raw
            WHERE is_indexable = 1 AND index_text IS NOT NULL AND index_text != ''
        """)

        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_role ON events(role)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")

        await conn.commit()
        await conn.close()

        # Build BM25 indexes in parallel
        await build_bm25_indexes_parallel()

        # Export markdown sessions
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                export_sessions,
                db_path,
                MD_SESSIONS_DIR,
                False
            )

        _db_ready.set()

    except Exception as e:
        _db_build_error = str(e)
        _db_ready.set()


async def main_async():
    """Async main function with MCP server."""
    global _db_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".codemem" / "codemem.sqlite"))
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    _db_path = Path(args.db)
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if rebuild needed
    needs_build = not _db_path.exists() or args.rebuild or _db_path.stat().st_size == 0

    if needs_build:
        if _db_path.exists():
            _db_path.unlink()

        # Start background build
        asyncio.create_task(build_db_async(_db_path, args.include_history, [Path(p) for p in args.root]))
    else:
        _db_ready.set()
        # Build indexes in background
        asyncio.create_task(build_bm25_indexes_parallel())

    # Start MCP server
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    server = Server("codemem")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="memory.query",
                description=(
                    "🌟 智能记忆查询 (推荐) - 使用自然语言查询对话历史。\n\n"
                    "支持的查询类型：\n"
                    "- 搜索内容：「我之前讨论过 Python 异步吗？」\n"
                    "- 查找会话：「上周关于数据库的对话」\n"
                    "- 活动摘要：「最近在做什么？」\n"
                    "- 获取上下文：「那段代码的完整上下文」\n\n"
                    "返回自然语言响应，包含摘要、洞察和建议。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "自然语言查询，例如：「我之前讨论过 Python 异步吗？」"
                        },
                        "context": {
                            "type": "string",
                            "description": "可选的对话上下文 ID，用于 follow-up 查询 (Phase 2 功能)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="semantic.search",
                description=(
                    "[Legacy] BM25 语义搜索 - 在对话历史中搜索相关内容。\n"
                    "推荐使用 memory.query 代替。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询"},
                        "top_k": {"type": "integer", "description": "返回结果数量", "default": 20},
                        "source": {
                            "type": "string",
                            "enum": ["sql", "markdown", "both"],
                            "description": "搜索来源",
                            "default": "both"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="activity.recent",
                description="[Legacy] 获取最近的活动记录。推荐使用 memory.query 代替。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "description": "天数", "default": 7}
                    }
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "memory.query":
                query = arguments.get("query", "")
                context = arguments.get("context")

                result = await memory_query_async(query, context)

                # Format response as readable text
                response_text = f"# {result.get('summary', '')}\n\n"

                if result.get("insights"):
                    response_text += "## 💡 洞察\n"
                    for insight in result["insights"]:
                        response_text += f"- {insight}\n"
                    response_text += "\n"

                if result.get("key_findings"):
                    response_text += "## 🔍 关键发现\n"
                    for finding in result["key_findings"]:
                        if isinstance(finding, dict):
                            rank = finding.get("rank", "")
                            session = finding.get("session", "")
                            text = finding.get("text", "")
                            score = finding.get("score", "")

                            if rank:
                                response_text += f"\n### {rank}. {session} (相关度: {score})\n"
                            response_text += f"{text}\n"
                    response_text += "\n"

                if result.get("suggestions"):
                    response_text += "## 💭 建议\n"
                    for suggestion in result["suggestions"]:
                        response_text += f"- {suggestion}\n"

                return [TextContent(type="text", text=response_text)]

            elif name == "semantic.search":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 20)
                source = arguments.get("source", "both")

                result = await bm25_search_async(query, limit=top_k, source=source)
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

            elif name == "activity.recent":
                days = arguments.get("days", 7)
                result = await get_recent_activity_async(days=days)
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # Run MCP server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
