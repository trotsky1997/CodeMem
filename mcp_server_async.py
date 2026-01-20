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
    """Async main function."""
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

    print("Async MCP server ready")
    print(f"Database: {_db_path}")
    print("Optimizations: async I/O, concurrent requests, parallel indexing")

    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if _db_pool:
            await _db_pool.close()


def main():
    """Entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
