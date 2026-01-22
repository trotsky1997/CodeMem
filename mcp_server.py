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

# BM25 index (markdown only)
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
                # Updated pattern to match actual format: ### timestamp | role | type
                messages = re.split(r'\n### (\d+) \| (\w+) \| (\w+)\n', content)

                # messages[0] is the header, then groups of 4: timestamp, role, type, text
                for i in range(1, len(messages), 4):
                    if i + 3 < len(messages):
                        timestamp = messages[i]
                        role = messages[i + 1]
                        msg_type = messages[i + 2]
                        text = messages[i + 3].strip()

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


async def build_bm25_index_async():
    """Build markdown BM25 index."""
    global _bm25_md_index, _bm25_md_docs, _bm25_md_metadata

    loop = asyncio.get_event_loop()

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor for Windows compatibility
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Build markdown index
        md_future = loop.run_in_executor(executor, build_bm25_md_index_sync, MD_SESSIONS_DIR)
        md_result = await md_future

        # Update global state
        async with _bm25_lock:
            _bm25_md_index, _bm25_md_docs, _bm25_md_metadata = md_result


async def bm25_search_async(query: str, limit: int = 20, mode: str = "refs") -> Dict[str, Any]:
    """Async BM25 search (markdown only).

    Args:
        query: Search query
        limit: Max results to return
        mode: Return mode - "refs" (references only), "preview" (with preview), "full" (complete text)
    """
    # Check cache
    key = cache_key("bm25", query, limit=limit, mode=mode)
    cached = await get_from_cache(key)
    if cached is not None:
        return cached

    # Wait for DB ready
    await asyncio.wait_for(_db_ready.wait(), timeout=120.0)

    # Check if markdown directory exists
    if not MD_SESSIONS_DIR.exists():
        return {
            "query": query,
            "count": 0,
            "results": [],
            "error": "Markdown sessions directory not found. Database may be empty or not initialized.",
            "hint": "Restart the MCP server to rebuild the database and export markdown files."
        }

    # Build index if needed
    async with _bm25_lock:
        if _bm25_md_index is None:
            return {
                "query": query,
                "count": 0,
                "results": [],
                "error": "BM25 index not built yet. Please wait for initialization to complete.",
                "hint": "The index is being built in the background. Try again in a few seconds."
            }

    # Enforce limit
    limit = min(limit, 50)

    # Tokenize query
    query_tokens = smart_tokenize(query)

    results = []

    # Search Markdown index
    if _bm25_md_index is not None:
        async with _bm25_lock:
            md_scores = _bm25_md_index.get_scores(query_tokens)
            md_top_indices = sorted(range(len(md_scores)), key=lambda i: md_scores[i], reverse=True)[:limit]

            for idx in md_top_indices:
                if md_scores[idx] > 0:
                    meta = _bm25_md_metadata[idx]

                    # Build result based on mode
                    if mode == "refs":
                        # Minimal reference mode - only IDs and metadata
                        result_item = {
                            "ref_id": f"{meta['session_id']}:{meta['timestamp']}",
                            "session_id": meta["session_id"],
                            "timestamp": meta["timestamp"],
                            "role": meta["role"],
                            "score": float(md_scores[idx]),
                            "file": f"~/.codemem/md_sessions/{meta['file']}",
                            "source": "markdown"
                        }
                    elif mode == "preview":
                        # Preview mode - include first 100 chars
                        text = meta["text"]
                        preview = text[:100] + "..." if len(text) > 100 else text
                        result_item = {
                            "ref_id": f"{meta['session_id']}:{meta['timestamp']}",
                            "session_id": meta["session_id"],
                            "timestamp": meta["timestamp"],
                            "role": meta["role"],
                            "preview": preview,
                            "score": float(md_scores[idx]),
                            "file": f"~/.codemem/md_sessions/{meta['file']}",
                            "source": "markdown"
                        }
                    else:  # mode == "full"
                        # Full mode - include complete text
                        result_item = {
                            "ref_id": f"{meta['session_id']}:{meta['timestamp']}",
                            "session_id": meta["session_id"],
                            "timestamp": meta["timestamp"],
                            "role": meta["role"],
                            "text": meta["text"],
                            "score": float(md_scores[idx]),
                            "file": f"~/.codemem/md_sessions/{meta['file']}",
                            "source": "markdown"
                        }

                    results.append(result_item)

    # Format result based on mode
    if mode == "summary":
        # Summary mode - return stats and sample
        result = {
            "query": query,
            "mode": "summary",
            "total_results": len(results),
            "sample_results": results[:3],  # First 3 results as sample
            "hint": "使用 mode='refs' 获取所有引用，或 mode='full' 获取完整内容"
        }
    else:
        # Other modes - return all results
        result = {
            "query": query,
            "mode": mode,
            "count": len(results),
            "results": results
        }

        # Add hint for refs mode
        if mode == "refs" and results:
            result["hint"] = "使用 get_message 工具获取完整消息内容，或使用 Read 工具读取文件"

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


async def build_db_async(db_path: Path, include_history: bool, extra_roots: List[Path]):
    """Build database asynchronously."""
    global _db_build_error

    try:
        # Add default platform paths if no extra_roots provided
        if not extra_roots:
            home = Path.home()
            if sys.platform == "win32":
                cursor_path = home / "AppData" / "Roaming" / "Cursor" / "User"
                opencode_path = home / "AppData" / "Local" / "opencode" / "project"
                claude_base = home / "AppData" / "Roaming" / ".claude"
                codex_base = home / "AppData" / "Roaming" / ".codex"
            elif sys.platform == "darwin":
                cursor_path = home / "Library" / "Application Support" / "Cursor" / "User"
                opencode_path = home / "Library" / "Application Support" / "opencode" / "project"
                claude_base = home / ".claude"
                codex_base = home / ".codex"
            else:  # Linux
                cursor_path = home / ".config" / "Cursor" / "User"
                opencode_path = home / ".config" / "opencode" / "project"
                claude_base = home / ".claude"
                codex_base = home / ".codex"

            # Add paths that exist
            default_roots = []
            for path in [claude_base, codex_base, cursor_path, opencode_path]:
                if path.exists():
                    default_roots.append(path)

            extra_roots = default_roots
            print(f"Using default paths: {[str(p) for p in extra_roots]}", file=sys.stderr)

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

        # Create events_raw table
        await conn.execute("DROP TABLE IF EXISTS events_raw")
        await conn.execute("""
            CREATE TABLE events_raw (
                platform TEXT,
                session_id TEXT,
                message_id TEXT,
                turn_id TEXT,
                item_index INTEGER,
                line_number INTEGER,
                timestamp TEXT,
                role TEXT,
                is_meta INTEGER,
                agent_id TEXT,
                is_indexable INTEGER,
                item_type TEXT,
                text TEXT,
                index_text TEXT,
                tool_name TEXT,
                tool_args TEXT,
                tool_result TEXT,
                tool_result_summary TEXT,
                source_file TEXT,
                raw_json TEXT
            )
        """)

        if rows:
            columns = list(rows[0].keys())
            placeholders = ",".join(["?"] * len(columns))

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

        # Export markdown sessions FIRST (BM25 needs these files)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                export_sessions,
                db_path,
                MD_SESSIONS_DIR,
                False
            )

        # Build BM25 indexes AFTER markdown export
        await build_bm25_index_async()

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

    # Find latest versioned database if symlink doesn't exist or is broken
    def find_latest_db() -> Optional[Path]:
        """Find the latest versioned database file."""
        db_dir = _db_path.parent
        db_stem = _db_path.stem
        # Find all versioned databases: codemem-{timestamp}.db
        versioned_dbs = list(db_dir.glob(f"{db_stem}-*.db"))
        if not versioned_dbs:
            return None
        # Sort by timestamp (extracted from filename)
        versioned_dbs.sort(key=lambda p: int(p.stem.split('-')[-1]), reverse=True)
        return versioned_dbs[0]

    # Check if rebuild needed
    needs_build = args.rebuild

    if not needs_build:
        # Try to use existing database
        if _db_path.exists() and _db_path.is_symlink():
            # Resolve symlink
            try:
                _db_path = _db_path.resolve()
                if _db_path.exists() and _db_path.stat().st_size > 0:
                    needs_build = False
                else:
                    needs_build = True
            except Exception:
                needs_build = True
        elif _db_path.exists() and not _db_path.is_symlink():
            # Direct file (not symlink)
            if _db_path.stat().st_size > 0:
                needs_build = False
            else:
                needs_build = True
        else:
            # Try to find latest versioned database
            latest_db = find_latest_db()
            if latest_db and latest_db.stat().st_size > 0:
                _db_path = latest_db
                needs_build = False
            else:
                needs_build = True

    if needs_build:
        # Use versioned database files with symlink
        import time
        timestamp = int(time.time())
        versioned_db = _db_path.parent / f"{_db_path.stem}-{timestamp}.db"

        # Clean up old versioned databases (keep last 3)
        db_dir = _db_path.parent
        db_stem = _db_path.stem
        old_dbs = list(db_dir.glob(f"{db_stem}-*.db"))
        if old_dbs:
            # Sort by timestamp
            old_dbs.sort(key=lambda p: int(p.stem.split('-')[-1]), reverse=True)
            # Keep last 3, delete the rest
            for old_db in old_dbs[3:]:
                try:
                    old_db.unlink()
                    print(f"Deleted old database: {old_db.name}", file=sys.stderr)
                except Exception as e:
                    print(f"Could not delete {old_db.name}: {e}", file=sys.stderr)

        # Build new database with versioned name
        asyncio.create_task(build_db_async(versioned_db, args.include_history, [Path(p) for p in args.root]))

        # Wait for build to complete, then update symlink
        async def update_symlink_after_build():
            await _db_ready.wait()
            if not _db_build_error:
                try:
                    # Remove old symlink if exists
                    if _db_path.exists() or _db_path.is_symlink():
                        _db_path.unlink()
                    # Create new symlink pointing to versioned db
                    _db_path.symlink_to(versioned_db.name)
                    print(f"Updated symlink: {_db_path} -> {versioned_db.name}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not create symlink: {e}", file=sys.stderr)
                    print(f"Using versioned database directly: {versioned_db}", file=sys.stderr)
                    # Update global path to point to versioned db
                    globals()['_db_path'] = versioned_db

        asyncio.create_task(update_symlink_after_build())
    else:
        _db_ready.set()
        # Build indexes in background
        asyncio.create_task(build_bm25_index_async())

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
                name="semantic.search",
                description=(
                    "BM25 语义搜索 - 搜索对话历史（仅 Markdown）。\n\n"
                    "使用 BM25 算法进行语义搜索，支持中英文分词。\n"
                    "搜索来源：Markdown 导出文件（按行匹配）。\n\n"
                    "参数：\n"
                    "- query: 搜索查询\n"
                    "- top_k: 返回结果数量（默认 20）\n"
                    "- mode: 返回模式（默认 refs）\n"
                    "  * summary: 返回统计信息和前3条样本（最节省上下文）\n"
                    "  * refs: 只返回引用ID和元数据（推荐，节省上下文）\n"
                    "  * preview: 返回前100字预览\n"
                    "  * full: 返回完整内容（慎用，占用大量上下文）\n\n"
                    "返回：匹配的对话记录，按相关性排序。\n"
                    "提示：使用 refs 模式后，可用 get_message 获取完整内容。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询"},
                        "top_k": {"type": "integer", "description": "返回结果数量", "default": 20},
                        "mode": {
                            "type": "string",
                            "enum": ["summary", "refs", "preview", "full"],
                            "default": "refs",
                            "description": "返回模式：summary=统计+样本, refs=引用, preview=预览, full=完整内容"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="sql.query",
                description=(
                    "SQL 查询 - 直接执行 SQL 查询。\n\n"
                    "适用场景：\n"
                    "- 复杂的数据分析\n"
                    "- 自定义统计查询\n"
                    "- 数据导出\n\n"
                    "常用模板：\n"
                    "- SELECT * FROM events WHERE text LIKE '%keyword%' LIMIT 10\n"
                    "- SELECT COUNT(*) FROM events WHERE role='user'\n"
                    "- SELECT session_id, COUNT(*) FROM events GROUP BY session_id\n\n"
                    "参数：\n"
                    "- query: SQL SELECT 查询语句\n"
                    "- limit: 结果数量限制（默认 100）\n"
                    "- mode: 返回模式（默认 summary）\n"
                    "  * summary: 返回统计信息和前3行示例（推荐）\n"
                    "  * full: 返回所有行（慎用）\n\n"
                    "⚠️ 只支持 SELECT 查询（只读）。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT 查询语句"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "结果数量限制",
                            "default": 100
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["summary", "full"],
                            "default": "summary",
                            "description": "返回模式：summary=摘要, full=完整结果"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="regex.search",
                description=(
                    "正则表达式搜索 - 使用正则表达式搜索文本。\n\n"
                    "适用场景：\n"
                    "- 精确的模式匹配\n"
                    "- 代码片段搜索\n"
                    "- 特定格式查找\n\n"
                    "示例：\n"
                    "- async def \\w+\\(.*\\): 查找异步函数定义\n"
                    "- \\d{3}-\\d{4}: 查找电话号码格式\n"
                    "- https?://\\S+: 查找 URL\n\n"
                    "参数：\n"
                    "- mode: 返回模式（默认 summary）\n"
                    "  * summary: 返回统计信息和前3条匹配（推荐）\n"
                    "  * refs: 返回引用ID和预览\n"
                    "  * full: 返回完整匹配内容（慎用）\n\n"
                    "支持 Python re 模块的正则语法。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "正则表达式模式"
                        },
                        "flags": {
                            "type": "string",
                            "description": "正则标志 (i=忽略大小写, m=多行, s=点匹配换行)",
                            "default": ""
                        },
                        "limit": {
                            "type": "integer",
                            "description": "结果数量限制",
                            "default": 50
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["summary", "refs", "full"],
                            "default": "summary",
                            "description": "返回模式：summary=摘要, refs=引用, full=完整内容"
                        }
                    },
                    "required": ["pattern"]
                }
            ),
            Tool(
                name="get_message",
                description=(
                    "获取完整消息 - 通过引用ID获取完整消息内容。\n\n"
                    "适用场景：\n"
                    "- 在使用 refs 模式搜索后获取完整内容\n"
                    "- 查看特定消息的详细信息\n\n"
                    "参数：\n"
                    "- ref_id: 引用ID，格式为 'session_id:timestamp'\n"
                    "  例如：'20250122_143022:1737529822.123'\n\n"
                    "返回：完整的消息内容，包括时间戳、角色、文本等。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "引用ID (格式: session_id:timestamp)"
                        }
                    },
                    "required": ["ref_id"]
                }
            ),
            Tool(
                name="list_sessions",
                description=(
                    "列出会话 - 列出所有对话会话的元数据。\n\n"
                    "适用场景：\n"
                    "- 浏览所有历史会话\n"
                    "- 查找特定时间段的会话\n"
                    "- 了解会话的基本信息（消息数量、时间范围等）\n\n"
                    "参数：\n"
                    "- limit: 返回会话数量限制（默认 50）\n"
                    "- platform: 过滤平台（可选，如 'claude.ai', 'api'）\n\n"
                    "返回：会话列表，包含会话ID、平台、消息数量、时间范围等。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "返回会话数量限制",
                            "default": 50
                        },
                        "platform": {
                            "type": "string",
                            "description": "过滤平台（可选）"
                        }
                    }
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "semantic.search":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 20)
                mode = arguments.get("mode", "refs")

                result = await bm25_search_async(query, limit=top_k, mode=mode)
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

            elif name == "sql.query":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 100)
                mode = arguments.get("mode", "summary")

                # Security: only allow SELECT queries
                if not query.strip().upper().startswith("SELECT"):
                    return [TextContent(type="text", text="Error: Only SELECT queries are allowed")]

                conn = await get_db_connection()
                cursor = await conn.execute(query)
                rows = await cursor.fetchall()

                # Limit results
                rows = rows[:limit]

                # Format as JSON
                columns = [desc[0] for desc in cursor.description] if cursor.description else []

                if mode == "summary":
                    # Summary mode - return stats and sample
                    result = {
                        "query": query,
                        "mode": "summary",
                        "total_rows": len(rows),
                        "columns": columns,
                        "sample_rows": [dict(zip(columns, row)) for row in rows[:3]],
                        "hint": "使用 mode='full' 获取所有行，或调整 LIMIT 子句"
                    }
                else:  # mode == "full"
                    # Full mode - return all rows
                    result = {
                        "query": query,
                        "mode": "full",
                        "columns": columns,
                        "rows": [dict(zip(columns, row)) for row in rows],
                        "count": len(rows)
                    }

                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

            elif name == "regex.search":
                pattern = arguments.get("pattern", "")
                flags_str = arguments.get("flags", "")
                limit = arguments.get("limit", 50)
                mode = arguments.get("mode", "summary")

                # Parse flags
                import re
                flags = 0
                if 'i' in flags_str:
                    flags |= re.IGNORECASE
                if 'm' in flags_str:
                    flags |= re.MULTILINE
                if 's' in flags_str:
                    flags |= re.DOTALL

                # Search in database
                conn = await get_db_connection()
                cursor = await conn.execute("""
                    SELECT timestamp, role, text, session_id, platform
                    FROM events
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)

                rows = await cursor.fetchall()

                # Apply regex
                matches = []
                regex = re.compile(pattern, flags)

                for row in rows:
                    text = row[2]
                    match_obj = regex.search(text)
                    if match_obj:
                        timestamp = row[0]
                        role = row[1]
                        session_id = row[3]
                        platform = row[4]

                        if mode == "summary":
                            # Summary mode - just count, show samples later
                            matches.append({
                                "timestamp": timestamp,
                                "role": role,
                                "text": text,
                                "session_id": session_id,
                                "platform": platform
                            })
                        elif mode == "refs":
                            # Refs mode - reference ID and preview
                            preview = text[:100] + "..." if len(text) > 100 else text
                            matches.append({
                                "ref_id": f"{session_id}:{timestamp}",
                                "timestamp": timestamp,
                                "role": role,
                                "preview": preview,
                                "session_id": session_id,
                                "platform": platform
                            })
                        else:  # mode == "full"
                            # Full mode - complete text
                            matches.append({
                                "timestamp": timestamp,
                                "role": role,
                                "text": text,
                                "session_id": session_id,
                                "platform": platform
                            })

                        if len(matches) >= limit:
                            break

                # Format result based on mode
                if mode == "summary":
                    result = {
                        "pattern": pattern,
                        "mode": "summary",
                        "total_matches": len(matches),
                        "sample_matches": matches[:3],
                        "hint": "使用 mode='refs' 或 mode='full' 获取更多详情"
                    }
                elif mode == "refs":
                    result = {
                        "pattern": pattern,
                        "mode": "refs",
                        "count": len(matches),
                        "matches": matches,
                        "hint": "使用 get_message 工具获取完整消息内容"
                    }
                else:  # mode == "full"
                    result = {
                        "pattern": pattern,
                        "mode": "full",
                        "count": len(matches),
                        "matches": matches
                    }

                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

            elif name == "get_message":
                ref_id = arguments.get("ref_id", "")

                # Parse ref_id (format: session_id:timestamp)
                if ":" not in ref_id:
                    return [TextContent(type="text", text="Error: Invalid ref_id format. Expected 'session_id:timestamp'")]

                session_id, timestamp = ref_id.split(":", 1)

                # Query database (timestamp is stored as ISO string)
                conn = await get_db_connection()
                cursor = await conn.execute("""
                    SELECT timestamp, role, text, session_id, platform
                    FROM events
                    WHERE session_id = ? AND timestamp = ?
                    LIMIT 1
                """, (session_id, timestamp))

                row = await cursor.fetchone()

                if not row:
                    return [TextContent(type="text", text=f"Error: Message not found for ref_id: {ref_id}")]

                result = {
                    "ref_id": ref_id,
                    "timestamp": row[0],
                    "role": row[1],
                    "text": row[2],
                    "session_id": row[3],
                    "platform": row[4]
                }

                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

            elif name == "list_sessions":
                limit = arguments.get("limit", 50)
                platform = arguments.get("platform")

                # Query database for session metadata
                conn = await get_db_connection()

                # Build query based on platform filter
                if platform:
                    query = """
                        SELECT
                            session_id,
                            platform,
                            COUNT(*) as message_count,
                            MIN(timestamp) as first_message,
                            MAX(timestamp) as last_message
                        FROM events
                        WHERE platform = ?
                        GROUP BY session_id
                        ORDER BY last_message DESC
                        LIMIT ?
                    """
                    cursor = await conn.execute(query, (platform, limit))
                else:
                    query = """
                        SELECT
                            session_id,
                            platform,
                            COUNT(*) as message_count,
                            MIN(timestamp) as first_message,
                            MAX(timestamp) as last_message
                        FROM events
                        GROUP BY session_id
                        ORDER BY last_message DESC
                        LIMIT ?
                    """
                    cursor = await conn.execute(query, (limit,))

                rows = await cursor.fetchall()

                # Format sessions
                sessions = []
                for row in rows:
                    sessions.append({
                        "session_id": row[0],
                        "platform": row[1],
                        "message_count": row[2],
                        "first_message": row[3],
                        "last_message": row[4]
                    })

                result = {
                    "total_sessions": len(sessions),
                    "sessions": sessions
                }

                if platform:
                    result["filtered_by_platform"] = platform

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
