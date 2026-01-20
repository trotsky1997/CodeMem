#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MCP stdio server for CodeMem.
Provides one tool: sql.query
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi
import tiktoken

sys.path.append(str(Path(__file__).parent))

from unified_history import collect_files, load_records, to_df
from export_sessions_md import export_sessions


# Initialize tiktoken encoder (using cl100k_base for GPT-4/GPT-3.5-turbo)
try:
    _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken_encoder = None


MD_SESSIONS_DIR = Path.home() / ".codemem" / "md_sessions"
SESSIONS_INDEX_URI = "codemem://sessions/index"
SESSIONS_URI_PREFIX = "codemem://sessions/"

# Global state for background database build
_db_build_lock = threading.Lock()
_db_ready = threading.Event()
_db_build_error: str | None = None

# Global BM25 index (SQL-based)
_bm25_index = None
_bm25_docs = []
_bm25_metadata = []

# Global BM25 index (Markdown-based)
_bm25_md_index = None
_bm25_md_docs = []
_bm25_md_metadata = []

# Query cache with LRU and TTL
_query_cache: Dict[str, Tuple[Any, float]] = {}
_cache_hits = 0
_cache_misses = 0
_cache_max_size = 100
_cache_ttl = 3600  # 1 hour

RESOURCES = [
    {
        "uri": "codemem://schema/events",
        "name": "events schema",
        "description": "Schema for events",
        "mimeType": "text/markdown",
    },
    {
        "uri": "codemem://schema/events_raw",
        "name": "events_raw schema",
        "description": "Schema for events_raw",
        "mimeType": "text/markdown",
    },
    {
        "uri": "codemem://query/templates",
        "name": "query templates",
        "description": "Common query patterns and examples",
        "mimeType": "text/markdown",
    },
    {
        "uri": "codemem://stats/summary",
        "name": "database statistics",
        "description": "Pre-computed statistics and aggregations (saves tokens)",
        "mimeType": "text/markdown",
    },
]


def smart_tokenize(text: str) -> List[str]:
    """Smart tokenization using tiktoken (supports Chinese and English)."""
    if not text:
        return []

    # Use tiktoken if available
    if _tiktoken_encoder is not None:
        try:
            # Encode to token IDs, then decode each token back to text
            token_ids = _tiktoken_encoder.encode(text.lower())
            # Convert token IDs to strings for BM25
            tokens = [str(tid) for tid in token_ids]
            return tokens
        except Exception:
            pass

    # Fallback to simple split
    return text.lower().split()


def cache_key(prefix: str, query: str, **kwargs) -> str:
    """Generate cache key from query and parameters."""
    key_data = f"{prefix}:{query}:{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_from_cache(key: str) -> Any | None:
    """Get result from cache if not expired."""
    global _cache_hits, _cache_misses

    if key in _query_cache:
        result, timestamp = _query_cache[key]
        if time.time() - timestamp < _cache_ttl:
            _cache_hits += 1
            return result
        else:
            # Expired, remove it
            del _query_cache[key]

    _cache_misses += 1
    return None


def put_in_cache(key: str, result: Any) -> None:
    """Put result in cache with LRU eviction."""
    global _query_cache

    # LRU eviction if cache is full
    if len(_query_cache) >= _cache_max_size:
        # Remove oldest entry
        oldest_key = min(_query_cache.keys(), key=lambda k: _query_cache[k][1])
        del _query_cache[oldest_key]

    _query_cache[key] = (result, time.time())


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0

    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total_requests": total,
        "hit_rate": f"{hit_rate:.1f}%",
        "cache_size": len(_query_cache),
        "max_size": _cache_max_size,
        "ttl_seconds": _cache_ttl
    }


def get_session_details(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    """Get full conversation history for a session."""
    try:
        wait_for_db(timeout=5.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    try:
        messages = conn.execute("""
            SELECT event_id, timestamp, role, text, platform
            FROM events
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT 200
        """, (session_id,)).fetchall()

        if not messages:
            return {"error": f"Session {session_id} not found"}

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [
                {
                    "event_id": msg[0],
                    "timestamp": msg[1],
                    "role": msg[2],
                    "text": msg[3],
                    "platform": msg[4]
                }
                for msg in messages
            ]
        }
    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}


def get_tool_usage(conn: sqlite3.Connection, days: int = 30) -> Dict[str, Any]:
    """Get tool usage statistics."""
    try:
        wait_for_db(timeout=5.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    try:
        tools = conn.execute(f"""
            SELECT
                tool_name,
                COUNT(*) as usage_count,
                COUNT(DISTINCT session_id) as session_count,
                MAX(timestamp) as last_used
            FROM events_raw
            WHERE tool_name IS NOT NULL
            AND datetime(timestamp) >= datetime('now', '-{days} days')
            GROUP BY tool_name
            ORDER BY usage_count DESC
            LIMIT 20
        """).fetchall()

        return {
            "days": days,
            "tool_count": len(tools),
            "tools": [
                {
                    "name": t[0],
                    "usage_count": t[1],
                    "session_count": t[2],
                    "last_used": t[3]
                }
                for t in tools
            ]
        }
    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}


def get_platform_stats(conn: sqlite3.Connection, days: int = 30) -> Dict[str, Any]:
    """Get platform usage statistics."""
    try:
        wait_for_db(timeout=5.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    try:
        platforms = conn.execute(f"""
            SELECT
                platform,
                COUNT(*) as event_count,
                COUNT(DISTINCT session_id) as session_count,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM events
            WHERE datetime(timestamp) >= datetime('now', '-{days} days')
            GROUP BY platform
            ORDER BY event_count DESC
        """).fetchall()

        return {
            "days": days,
            "platforms": [
                {
                    "name": p[0],
                    "event_count": p[1],
                    "session_count": p[2],
                    "first_seen": p[3],
                    "last_seen": p[4]
                }
                for p in platforms
            ]
        }
    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}


def get_recent_activity(conn: sqlite3.Connection, days: int = 7) -> Dict[str, Any]:
    """Get structured summary of recent activity."""
    try:
        wait_for_db(timeout=5.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    try:
        # Get sessions from last N days
        sessions = conn.execute(f"""
            SELECT
                session_id,
                COUNT(*) as event_count,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen,
                GROUP_CONCAT(DISTINCT platform) as platforms
            FROM events
            WHERE datetime(timestamp) >= datetime('now', '-{days} days')
            AND session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY last_seen DESC
            LIMIT 10
        """).fetchall()

        # Get sample messages from each session
        activity = []
        for session_id, event_count, first_seen, last_seen, platforms in sessions:
            # Get user messages from this session
            messages = conn.execute("""
                SELECT text
                FROM events
                WHERE session_id = ? AND role = 'user'
                ORDER BY timestamp DESC
                LIMIT 5
            """, (session_id,)).fetchall()

            activity.append({
                "session_id": session_id,
                "event_count": event_count,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "platforms": platforms,
                "sample_messages": [msg[0][:100] for msg in messages if msg[0]]
            })

        return {
            "days": days,
            "session_count": len(activity),
            "sessions": activity
        }
    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}


def generate_stats_summary(conn: sqlite3.Connection) -> str:
    """Generate pre-computed statistics to save agent tokens."""
    try:
        wait_for_db(timeout=5.0)
    except (TimeoutError, RuntimeError):
        return "# Database Statistics\n\nDatabase is still building. Please try again in a moment."

    lines = ["# Database Statistics", "", "**Quick overview to help you understand what's available without querying.**", ""]

    # Total counts
    try:
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        total_raw = conn.execute("SELECT COUNT(*) FROM events_raw").fetchone()[0]
        lines.extend([
            "## Overview",
            f"- Total searchable events: **{total_events:,}**",
            f"- Total raw events: **{total_raw:,}**",
            ""
        ])
    except sqlite3.Error:
        pass

    # Platform breakdown
    try:
        platforms = conn.execute("""
            SELECT platform, COUNT(*) as count
            FROM events
            GROUP BY platform
            ORDER BY count DESC
        """).fetchall()
        if platforms:
            lines.extend(["## By Platform", ""])
            for platform, count in platforms:
                lines.append(f"- **{platform}**: {count:,} events")
            lines.append("")
    except sqlite3.Error:
        pass

    # Role breakdown
    try:
        roles = conn.execute("""
            SELECT role, COUNT(*) as count
            FROM events
            GROUP BY role
            ORDER BY count DESC
        """).fetchall()
        if roles:
            lines.extend(["## By Role", ""])
            for role, count in roles:
                lines.append(f"- **{role}**: {count:,} events")
            lines.append("")
    except sqlite3.Error:
        pass

    # Top sessions
    try:
        sessions = conn.execute("""
            SELECT session_id, COUNT(*) as count
            FROM events
            WHERE session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()
        if sessions:
            lines.extend(["## Top 10 Sessions (by message count)", ""])
            for session_id, count in sessions:
                lines.append(f"- `{session_id}`: {count} messages")
            lines.append("")
    except sqlite3.Error:
        pass

    # Recent activity
    try:
        recent = conn.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM events
            WHERE timestamp IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        """).fetchall()
        if recent:
            lines.extend(["## Recent Activity (last 7 days)", ""])
            for date, count in recent:
                lines.append(f"- **{date}**: {count} events")
            lines.append("")
    except sqlite3.Error:
        pass

    # Tool usage
    try:
        tools = conn.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM events_raw
            WHERE tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()
        if tools:
            lines.extend(["## Top 10 Tools Used", ""])
            for tool, count in tools:
                lines.append(f"- **{tool}**: {count} times")
            lines.append("")
    except sqlite3.Error:
        pass

    lines.extend([
        "---",
        "",
        "## Cache Statistics",
        "",
    ])

    cache_stats = get_cache_stats()
    lines.extend([
        f"- **Hit rate**: {cache_stats['hit_rate']} ({cache_stats['hits']} hits / {cache_stats['total_requests']} total)",
        f"- **Cache size**: {cache_stats['cache_size']} / {cache_stats['max_size']} entries",
        f"- **TTL**: {cache_stats['ttl_seconds']} seconds",
        "",
        "---",
        "",
        "**Tip**: Use this summary to understand your data before querying.",
        "Check `codemem://query/templates` for query examples."
    ])

    return "\n".join(lines)


def query_templates_markdown() -> str:
    return """# Common Query Templates

## Basic Queries

### 1. Recent conversations (most common)
```sql
SELECT event_id, timestamp, role, text, session_id
FROM events
ORDER BY timestamp DESC
LIMIT 20;
```

### 2. Search by keyword
```sql
SELECT event_id, timestamp, role, text, session_id
FROM events
WHERE text LIKE '%keyword%'
ORDER BY timestamp DESC
LIMIT 20;
```

### 3. Get specific session
```sql
SELECT event_id, timestamp, role, text
FROM events
WHERE session_id = 'abc12345'
ORDER BY timestamp ASC
LIMIT 50;
```

### 4. Count by platform
```sql
SELECT platform, COUNT(*) as count
FROM events
GROUP BY platform;
```

### 5. Recent user messages
```sql
SELECT event_id, timestamp, text, session_id
FROM events
WHERE role = 'user'
ORDER BY timestamp DESC
LIMIT 20;
```

### 6. Recent assistant responses
```sql
SELECT event_id, timestamp, text, session_id
FROM events
WHERE role = 'assistant'
ORDER BY timestamp DESC
LIMIT 20;
```

## Advanced Queries

### 7. Sessions with most messages
```sql
SELECT session_id, COUNT(*) as msg_count
FROM events
WHERE session_id IS NOT NULL
GROUP BY session_id
ORDER BY msg_count DESC
LIMIT 20;
```

### 8. Search across specific platform
```sql
SELECT event_id, timestamp, role, text, session_id
FROM events
WHERE platform = 'claude' AND text LIKE '%search term%'
ORDER BY timestamp DESC
LIMIT 20;
```

### 9. Messages from specific date
```sql
SELECT event_id, timestamp, role, text, session_id
FROM events
WHERE timestamp LIKE '2026-01-20%'
ORDER BY timestamp DESC
LIMIT 50;
```

### 10. Get tool usage (from events_raw)
```sql
SELECT timestamp, tool_name, tool_result_summary, session_id
FROM events_raw
WHERE tool_name IS NOT NULL
ORDER BY timestamp DESC
LIMIT 20;
```

## Tips

- **Always use LIMIT**: Maximum 50 rows allowed
- **Use WHERE**: Filter by session_id, platform, role, or timestamp
- **Search text**: Use `text LIKE '%keyword%'` for full-text search
- **Order by timestamp**: Usually `ORDER BY timestamp DESC` for recent first
- **Platform values**: 'claude', 'codex', 'opencode', 'cursor'
- **Role values**: 'user', 'assistant', 'system'

## Performance Tips

- Add `WHERE session_id = 'xxx'` to limit scope
- Use `timestamp LIKE '2026-01%'` for date filtering
- Avoid `SELECT *` - specify needed columns
- Use `events` table for text search (faster, indexed)
- Use `events_raw` table only when you need tool details or raw_json
"""


def sessions_index_markdown(md_dir: Path) -> str:
    missing_dir = not md_dir.exists()
    files = sorted(p.name for p in md_dir.glob("*.md")) if not missing_dir else []
    lines = [
        "# sessions",
        "",
        f"count: {len(files)}",
        "",
        "files:",
    ]
    if files:
        lines.extend(f"- {name}" for name in files)
    else:
        lines.append("- (none)")
    if missing_dir:
        lines.append("")
        lines.append("note: md_sessions directory not found; export may not have run yet.")
    lines.append("")
    lines.append("Usage:")
    lines.append(f"- read: {SESSIONS_URI_PREFIX}<filename>")
    return "\n".join(lines)


def resolve_md_path(md_dir: Path, path_str: str) -> Path:
    md_root = md_dir.resolve()
    if not path_str or path_str == ".":
        return md_root

    raw_path = Path(path_str)
    if raw_path.is_absolute():
        target = raw_path.resolve()
        if target == md_root.parent:
            return md_root
    else:
        parts = raw_path.parts
        if parts and parts[0] == md_root.name:
            raw_path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        target = (md_dir / raw_path).resolve()

    if target != md_root and md_root not in target.parents:
        raise ValueError(f"path outside md_sessions: {path_str}; use md_sessions or omit paths")
    return target


def run_text_shell(cmd: str, args: Any, paths: Any) -> Dict[str, Any]:
    allowed = {"rg", "sed", "awk"}
    if cmd not in allowed:
        return {"error": f"unsupported cmd: {cmd}"}
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        return {"error": "args must be a list of strings"}
    if paths is None:
        paths = []
    if not isinstance(paths, list) or not all(isinstance(x, str) for x in paths):
        return {"error": "paths must be a list of strings"}

    if cmd in ("sed", "awk") and not paths:
        return {"error": f"{cmd} requires at least one path"}
    if cmd == "rg" and not args:
        return {"error": "rg requires a pattern in args"}

    md_root = MD_SESSIONS_DIR.resolve()
    try:
        resolved_paths = [resolve_md_path(md_root, p) for p in paths] if paths else []
    except ValueError as exc:
        return {"error": str(exc), "md_root": str(md_root)}
    if cmd == "rg" and not resolved_paths:
        resolved_paths = [md_root]

    cmdline = [cmd] + args + [str(p) for p in resolved_paths]
    try:
        proc = subprocess.run(
            cmdline,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        return {"error": f"{cmd} not found", "md_root": str(md_root)}
    except subprocess.TimeoutExpired:
        return {"error": "command timed out", "md_root": str(md_root)}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    truncated = False
    if len(stdout) > 20000:
        stdout = stdout[:20000] + "\n...truncated..."
        truncated = True
    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "truncated": truncated,
        "md_root": str(md_root),
        "resolved_paths": [str(p) for p in resolved_paths],
    }


def build_bm25_index(db_path: Path) -> None:
    """Build BM25 index from events table."""
    global _bm25_index, _bm25_docs, _bm25_metadata

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Fetch all searchable events
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
            return

        # Prepare documents for BM25
        _bm25_docs = []
        _bm25_metadata = []

        for event_id, timestamp, role, text, session_id, platform in rows:
            # Smart tokenization (supports Chinese and English)
            tokens = smart_tokenize(text)
            _bm25_docs.append(tokens)
            _bm25_metadata.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "role": role,
                "text": text,
                "session_id": session_id,
                "platform": platform
            })

        # Build BM25 index
        _bm25_index = BM25Okapi(_bm25_docs)

    except Exception:
        # Silently fail if index building fails
        pass


def build_bm25_md_index(md_sessions_dir: Path) -> None:
    """Build BM25 index from Markdown session files."""
    global _bm25_md_index, _bm25_md_docs, _bm25_md_metadata

    try:
        if not md_sessions_dir.exists():
            return

        md_files = list(md_sessions_dir.glob("*.md"))
        if not md_files:
            return

        # Prepare documents for BM25
        _bm25_md_docs = []
        _bm25_md_metadata = []

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")

                # Extract session_id from filename
                session_id = md_file.stem

                # Parse markdown to extract messages
                # Split by message headers (### [role] timestamp)
                import re
                messages = re.split(r'\n### \[(.*?)\] (.*?)\n', content)

                # Process messages (skip header, process in groups of 3: role, timestamp, content)
                for i in range(1, len(messages), 3):
                    if i + 2 < len(messages):
                        role = messages[i]
                        timestamp = messages[i + 1]
                        text = messages[i + 2].strip()

                        if text:
                            # Smart tokenization
                            tokens = smart_tokenize(text)
                            _bm25_md_docs.append(tokens)
                            _bm25_md_metadata.append({
                                "session_id": session_id,
                                "timestamp": timestamp,
                                "role": role,
                                "text": text,
                                "source": "markdown",
                                "file": md_file.name
                            })
            except Exception:
                # Skip files that fail to parse
                continue

        if _bm25_md_docs:
            # Build BM25 index
            _bm25_md_index = BM25Okapi(_bm25_md_docs)

    except Exception:
        # Silently fail if index building fails
        pass


def bm25_search(query: str, limit: int = 20, source: str = "sql") -> Dict[str, Any]:
    """Search using BM25 ranking.

    Args:
        query: Search query
        limit: Maximum number of results
        source: Data source - "sql" (default), "markdown", or "both"
    """
    global _bm25_index, _bm25_docs, _bm25_metadata
    global _bm25_md_index, _bm25_md_docs, _bm25_md_metadata

    # Check cache first
    key = cache_key("bm25", query, limit=limit, source=source)
    cached = get_from_cache(key)
    if cached is not None:
        return cached

    # Wait for database to be ready
    try:
        wait_for_db(timeout=120.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    # Enforce maximum limit
    MAX_LIMIT = 50
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT

    # Smart tokenization for query (supports Chinese and English)
    query_tokens = smart_tokenize(query)

    results = []

    # Search SQL index
    if source in ("sql", "both"):
        # Build SQL index if not exists
        if _bm25_index is None:
            db_path = Path.home() / ".codemem" / "codemem.sqlite"
            build_bm25_index(db_path)

        if _bm25_index is not None:
            # Get BM25 scores
            scores = _bm25_index.get_scores(query_tokens)

            # Get top results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
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
    if source in ("markdown", "both"):
        # Build Markdown index if not exists
        if _bm25_md_index is None:
            build_bm25_md_index(MD_SESSIONS_DIR)

        if _bm25_md_index is not None:
            # Get BM25 scores
            md_scores = _bm25_md_index.get_scores(query_tokens)

            # Get top results
            md_top_indices = sorted(range(len(md_scores)), key=lambda i: md_scores[i], reverse=True)[:limit]

            for idx in md_top_indices:
                if md_scores[idx] > 0:  # Only include results with positive scores
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

    # Sort combined results by score
    if source == "both":
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

    result = {
        "query": query,
        "count": len(results),
        "results": results,
        "source": source
    }

    # Cache the result
    put_to_cache(key, result)

    return result

    # Get BM25 scores
    scores = _bm25_index.get_scores(query_tokens)

    # Get top results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include results with positive scores
            meta = _bm25_metadata[idx]
            results.append({
                "event_id": meta["event_id"],
                "timestamp": meta["timestamp"],
                "role": meta["role"],
                "text": meta["text"][:200] + "..." if len(meta["text"]) > 200 else meta["text"],
                "session_id": meta["session_id"],
                "platform": meta["platform"],
                "score": float(scores[idx])
            })

    result = {
        "query": query,
        "results": results,
        "count": len(results)
    }

    # Cache the result
    put_in_cache(key, result)
    return result


def build_db(db_path: Path, include_history: bool, extra_roots: List[Path]) -> None:
    home = Path.home()

    # Platform-specific paths
    if sys.platform == "win32":
        # Windows paths
        cursor_path = home / "AppData" / "Roaming" / "Cursor" / "User"
        opencode_path = home / "AppData" / "Local" / "opencode" / "project"
        claude_base = home / "AppData" / "Roaming" / ".claude"
        codex_base = home / "AppData" / "Roaming" / ".codex"
    elif sys.platform == "darwin":
        # macOS paths
        cursor_path = home / "Library" / "Application Support" / "Cursor" / "User"
        opencode_path = home / "Library" / "Application Support" / "opencode" / "project"
        claude_base = home / ".claude"
        codex_base = home / ".codex"
    else:
        # Linux paths
        cursor_path = home / ".config" / "Cursor" / "User"
        opencode_path = home / ".local" / "share" / "opencode" / "project"
        claude_base = home / ".claude"
        codex_base = home / ".codex"

    roots = [
        claude_base / "projects",
        claude_base / "transcripts",
        codex_base / "sessions",
        opencode_path,
        cursor_path / "workspaceStorage",
        cursor_path / "globalStorage",
    ]
    if include_history:
        roots.append(claude_base / "history.jsonl")
    roots.extend(extra_roots)

    files = collect_files(roots)
    records = load_records(files)
    df = to_df(records)

    df = df.copy()
    for col in ("tool_args", "tool_result", "raw_json"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False, default=str)
                if isinstance(v, (dict, list))
                else ("" if v is None else v)
            )

    with sqlite3.connect(str(db_path)) as conn:
        df.to_sql("events_raw", conn, if_exists="replace", index=False)
        conn.execute("create index if not exists idx_events_raw_time on events_raw(timestamp)")
        conn.execute("create index if not exists idx_events_raw_indexable on events_raw(is_indexable)")
        conn.execute("drop table if exists events")
        conn.execute(
            """
            create table events as
            select
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
            from events_raw
            where is_indexable = 1 and index_text is not null and index_text != ''
            """
        )
        conn.execute("create index if not exists idx_events_time on events(timestamp)")
        conn.execute("create index if not exists idx_events_role on events(role)")
        conn.execute("create index if not exists idx_events_session on events(session_id)")
        conn.execute("create index if not exists idx_events_line on events(session_id, item_index, line_number)")


def build_db_background(db_path: Path, include_history: bool, extra_roots: List[Path], md_sessions_dir: Path, export_md: bool) -> None:
    """Build database in background thread."""
    global _db_build_error
    try:
        build_db(db_path, include_history, extra_roots)
        if export_md:
            try:
                export_sessions(db_path, md_sessions_dir, include_meta=False)
            except OSError:
                pass
        # Build BM25 index after database is ready
        build_bm25_index(db_path)
        _db_ready.set()
    except Exception as exc:
        _db_build_error = str(exc)
        _db_ready.set()


def wait_for_db(timeout: float = 60.0) -> None:
    """Wait for database to be ready, raise error if build failed."""
    if not _db_ready.wait(timeout=timeout):
        raise TimeoutError("Database build timed out")
    if _db_build_error:
        raise RuntimeError(f"Database build failed: {_db_build_error}")


def sql_query(conn: sqlite3.Connection, query: str, limit: int) -> Dict[str, Any]:
    # Check cache first
    key = cache_key("sql", query, limit=limit)
    cached = get_from_cache(key)
    if cached is not None:
        return cached

    # Wait for database to be ready before executing queries
    try:
        wait_for_db(timeout=120.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    # Enforce maximum limit of 50 rows to prevent token waste
    MAX_LIMIT = 50
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT

    q = query.strip()
    q_norm = _strip_sql_leading_comments(q)
    if not q_norm:
        return {"error": "空查询：只允许只读 SELECT/CTE/PRAGMA"}
    if _is_readonly_pragma(q_norm):
        exec_q = q_norm
    else:
        if not _is_readonly_select(q_norm):
            return {"error": "只允许只读 SELECT/CTE/PRAGMA"}
        exec_q = q.rstrip().rstrip(";")
        if not re.search(r"\blimit\b", q_norm, re.IGNORECASE):
            exec_q = f"{exec_q} limit {limit}"
        else:
            # Check if user's LIMIT exceeds MAX_LIMIT
            limit_match = re.search(r"\blimit\s+(\d+)", q_norm, re.IGNORECASE)
            if limit_match:
                user_limit = int(limit_match.group(1))
                if user_limit > MAX_LIMIT:
                    return {"error": f"LIMIT exceeds maximum allowed ({MAX_LIMIT}). Please use a smaller LIMIT or add more specific WHERE conditions."}
    try:
        cur = conn.execute(exec_q)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []

        # Additional safety check
        if len(rows) > MAX_LIMIT:
            rows = rows[:MAX_LIMIT]
            result = {
                "columns": cols,
                "rows": rows,
                "warning": f"Results truncated to {MAX_LIMIT} rows. Use more specific WHERE conditions to narrow your search."
            }
        else:
            result = {"columns": cols, "rows": rows}

        # Cache the result
        put_in_cache(key, result)
        return result
    except sqlite3.Error as exc:
        return {"error": f"sqlite error: {exc}; query={exec_q}"}


READONLY_PRAGMAS = {
    "collation_list",
    "compile_options",
    "database_list",
    "foreign_key_list",
    "function_list",
    "index_info",
    "index_list",
    "index_xinfo",
    "module_list",
    "pragma_list",
    "table_info",
    "table_list",
    "table_xinfo",
}

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|replace|alter|drop|create|attach|detach|vacuum|reindex)\b",
    re.IGNORECASE,
)


def _strip_sql_leading_comments(query: str) -> str:
    s = query.lstrip()
    while True:
        if s.startswith("--"):
            nl = s.find("\n")
            if nl == -1:
                return ""
            s = s[nl + 1 :].lstrip()
            continue
        if s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                return ""
            s = s[end + 2 :].lstrip()
            continue
        return s


def _is_readonly_pragma(query: str) -> bool:
    if not query.lower().startswith("pragma"):
        return False
    if "=" in query:
        return False
    match = re.match(r"(?is)^pragma\s+([a-z0-9_]+(?:\.[a-z0-9_]+)?)", query)
    if not match:
        return False
    name = match.group(1).split(".")[-1].lower()
    return name in READONLY_PRAGMAS


def _is_readonly_select(query: str) -> bool:
    q_lower = query.lower()
    if not (q_lower.startswith("select") or q_lower.startswith("with")):
        return False
    if FORBIDDEN_SQL.search(query):
        return False
    return True


def table_schema_markdown(conn: sqlite3.Connection, table: str) -> str:
    rows = conn.execute(f"pragma table_info({table})").fetchall()
    if not rows:
        return f"# {table}\n\nNo schema found."
    lines = [
        f"# {table}",
        "",
        "| name | type | notnull | pk | default |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, name, col_type, notnull, dflt_value, pk in rows:
        dflt = "" if dflt_value is None else str(dflt_value)
        lines.append(f"| {name} | {col_type} | {notnull} | {pk} | {dflt} |")
    lines.extend(
        [
            "",
            "Example:",
            "```sql",
            f"select * from {table} limit 5;",
            "```",
        ]
    )
    return "\n".join(lines)


def table_schema_data(conn: sqlite3.Connection, table: str) -> Dict[str, Any]:
    rows = conn.execute(f"pragma table_info({table})").fetchall()
    columns = []
    for _, name, col_type, notnull, dflt_value, pk in rows:
        columns.append(
            {
                "name": name,
                "type": col_type,
                "notnull": int(notnull),
                "pk": int(pk),
                "default": dflt_value,
            }
        )
    return {"table": table, "columns": columns}


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select name from sqlite_master where type in ('table','view') and name = ?",
        (table,),
    ).fetchone()
    return bool(row)



def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def sql_preview_text(result: Dict[str, Any], max_rows: int = 5, max_cell: int = 80) -> str:
    if not isinstance(result, dict) or "rows" not in result or "columns" not in result:
        return "ok"
    cols = result.get("columns") or []
    rows = result.get("rows") or []
    lines = [
        f"columns: {', '.join(str(c) for c in cols)}" if cols else "columns: (none)",
        f"rows: {len(rows)} (preview {min(len(rows), max_rows)})",
    ]
    if cols and rows:
        header = " | ".join(str(c) for c in cols)
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows[:max_rows]:
            cells = [_truncate_text("" if v is None else str(v), max_cell) for v in row]
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def tool_text_summary(
    name: str,
    result: Dict[str, Any],
    preview: bool = False,
    preview_rows: int = 5,
    preview_cell_len: int = 80,
) -> str:
    if not isinstance(result, dict):
        return ""
    if "error" in result:
        return str(result["error"])
    if name == "sql.query" and preview:
        return sql_preview_text(result, max_rows=preview_rows, max_cell=preview_cell_len)
    if name == "text.shell":
        stdout = result.get("stdout", "") or ""
        stderr = result.get("stderr", "") or ""
        if stdout:
            return stdout
        if stderr:
            return stderr
        return f"(no output; returncode={result.get('returncode')})"
    return "ok"


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def parse_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def respond(msg_id: Any, result: Dict[str, Any]) -> None:
    out = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def respond_error(msg_id: Any, code: int, message: str) -> None:
    out = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    global _db_build_error
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".codemem" / "codemem.sqlite"))
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--no-export-md-sessions", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild database even if it exists")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if database needs to be built
    needs_build = not db_path.exists() or args.rebuild or db_path.stat().st_size == 0

    if needs_build:
        if db_path.exists():
            db_path.unlink()
        # Start background build thread
        build_thread = threading.Thread(
            target=build_db_background,
            args=(db_path, args.include_history, [Path(p) for p in args.root], MD_SESSIONS_DIR, not args.no_export_md_sessions),
            daemon=True
        )
        build_thread.start()
    else:
        # Database exists, mark as ready immediately
        _db_ready.set()

    # Lazy connection - will be created when first query arrives
    conn = None
    def get_conn():
        nonlocal conn
        if conn is None:
            conn = sqlite3.connect(str(db_path))
        return conn
    tools = [
        {
            "name": "activity.recent",
            "description": "Get structured summary of recent activity (FASTEST way to see what you've been working on). Returns sessions with sample messages from last N days. Use this FIRST before sql.query for activity summaries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "Number of days to look back (default 7)"},
                },
            },
        },
        {
            "name": "semantic.search",
            "description": "Search conversations using natural language (BM25). No SQL needed! Just describe what you're looking for. SECOND BEST option after activity.recent for finding content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query (e.g., 'Python debugging tips')"},
                    "limit": {"type": "integer", "default": 20, "description": "Max results to return (max 50)"},
                    "source": {
                        "type": "string",
                        "enum": ["sql", "markdown", "both"],
                        "default": "sql",
                        "description": "Data source: 'sql' (structured DB, default), 'markdown' (session files), or 'both' (combined search)"
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "session.get",
            "description": "Get full conversation history for a specific session. Use when you need complete context of a session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID (8-char hash, e.g., '73133d96')"},
                },
                "required": ["session_id"],
            },
        },
        {
            "name": "tools.usage",
            "description": "Get tool usage statistics. See which tools were used most frequently.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 30, "description": "Number of days to analyze (default 30)"},
                },
            },
        },
        {
            "name": "platform.stats",
            "description": "Get platform usage breakdown (Claude/Codex/Cursor/OpenCode). See activity distribution across platforms.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 30, "description": "Number of days to analyze (default 30)"},
                },
            },
        },
        {
            "name": "sql.query",
            "description": "Run read-only SELECT/CTE/PRAGMA queries against the events table. Use ONLY when activity.recent and semantic.search don't work. Check codemem://query/templates for examples.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Read-only SQL (SELECT/CTE/PRAGMA)"},
                    "limit": {"type": "integer", "default": 100, "description": "Row limit if query has no LIMIT"},
                    "preview": {
                        "type": "boolean",
                        "default": False,
                        "description": "When true, content.text includes a short preview of rows.",
                    },
                    "preview_rows": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max rows shown in content.text when preview is true (clamped 1-50).",
                    },
                    "preview_cell_len": {
                        "type": "integer",
                        "default": 80,
                        "description": "Max cell length in content.text when preview is true (clamped 10-200).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "text.context",
            "description": "Get context lines around a specific line in a session. Useful for viewing code snippets or conversation context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID (8-char hash)"},
                    "item_index": {"type": "integer", "description": "Item index within the message"},
                    "line_number": {"type": "integer", "description": "Line number within the item (0-indexed)"},
                    "context_lines": {"type": "integer", "default": 3, "description": "Number of lines before and after (default 3)"},
                },
                "required": ["session_id", "item_index", "line_number"],
            },
        },
        {
            "name": "text.shell",
            "description": "Run rg/sed/awk against md_sessions with a restricted allowlist.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "enum": ["rg", "sed", "awk"]},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Command args; rg needs a pattern"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relative to md_sessions; accepts md_sessions or ./md_sessions as alias",
                    },
                },
                "required": ["cmd"],
            },
        },
    ]
    resource_map = {item["uri"]: item for item in RESOURCES}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # JSON-RPC notifications have no "id"; must not respond.
        if "id" not in msg or msg["id"] is None:
            continue

        msg_id = msg["id"]
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "initialize":
            respond(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "codemem", "version": "0.1.0"},
                },
            )
        elif method == "tools/list":
            respond(msg_id, {"tools": tools})
        elif method == "resources/list":
            resources = list(RESOURCES)
            resources.append(
                {
                    "uri": SESSIONS_INDEX_URI,
                    "name": "sessions index",
                    "description": "List exported session markdown files",
                    "mimeType": "text/markdown",
                }
            )
            respond(msg_id, {"resources": resources})
        elif method == "resources/read":
            uri = params.get("uri")
            if uri == SESSIONS_INDEX_URI:
                if not MD_SESSIONS_DIR.exists() or not list(MD_SESSIONS_DIR.glob("*.md")):
                    try:
                        export_sessions(db_path, MD_SESSIONS_DIR, include_meta=False)
                    except OSError:
                        pass
                text = sessions_index_markdown(MD_SESSIONS_DIR)
                respond(
                    msg_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ]
                    },
                )
                continue
            if uri == "codemem://query/templates":
                text = query_templates_markdown()
                respond(
                    msg_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ]
                    },
                )
                continue
            if uri == "codemem://stats/summary":
                text = generate_stats_summary(get_conn())
                respond(
                    msg_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ]
                    },
                )
                continue
            if isinstance(uri, str) and uri.startswith(SESSIONS_URI_PREFIX):
                name = uri[len(SESSIONS_URI_PREFIX) :]
                name = Path(name).name
                if not name.endswith(".md"):
                    name = f"{name}.md"
                target = (MD_SESSIONS_DIR / name).resolve()
                if not target.exists() or target.parent.resolve() != MD_SESSIONS_DIR.resolve():
                    try:
                        export_sessions(db_path, MD_SESSIONS_DIR, include_meta=False)
                    except OSError:
                        pass
                    target = (MD_SESSIONS_DIR / name).resolve()
                    if not target.exists() or target.parent.resolve() != MD_SESSIONS_DIR.resolve():
                        respond_error(msg_id, -32602, "unknown session markdown")
                        continue
                text = target.read_text(encoding="utf-8")
                respond(
                    msg_id,
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ],
                        "data": {"uri": uri, "text": text},
                    },
                )
                continue
            resource = resource_map.get(uri)
            if not resource:
                respond_error(msg_id, -32602, "unknown resource")
                continue
            table = uri.split("/")[-1]
            text = table_schema_markdown(get_conn(), table)
            data = table_schema_data(get_conn(), table)
            respond(
                msg_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": resource.get("mimeType", "text/markdown"),
                            "text": text,
                        }
                    ],
                    "data": data,
                },
            )
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "activity.recent":
                days = parse_int(args.get("days", 7), 7)
                if days <= 0:
                    days = 7
                result = get_recent_activity(get_conn(), days)

                # Format results as text
                if "error" in result:
                    text = result["error"]
                else:
                    lines = [f"Recent Activity (last {result['days']} days)", ""]
                    lines.append(f"Found {result['session_count']} active sessions:", "")
                    for i, session in enumerate(result.get("sessions", []), 1):
                        lines.append(f"{i}. Session {session['session_id']}")
                        lines.append(f"   Platform: {session['platforms']} | Events: {session['event_count']}")
                        lines.append(f"   Active: {session['first_seen']} to {session['last_seen']}")
                        if session.get('sample_messages'):
                            lines.append(f"   Sample tasks:")
                            for msg in session['sample_messages'][:3]:
                                lines.append(f"   - {msg}")
                        lines.append("")
                    text = "\n".join(lines)

                payload = {
                    "content": [{"type": "text", "text": text}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "ACTIVITY_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"session_count": result["session_count"], "days": result["days"]}
                respond(msg_id, payload)
                continue
            if name == "session.get":
                session_id = args.get("session_id", "")
                result = get_session_details(get_conn(), session_id)

                # Format results as text
                if "error" in result:
                    text = result["error"]
                else:
                    lines = [f"Session {result['session_id']} ({result['message_count']} messages)", ""]
                    for i, msg in enumerate(result.get("messages", [])[:50], 1):
                        lines.append(f"{i}. [{msg['role']}] {msg['text'][:150]}")
                        lines.append(f"   Time: {msg['timestamp']}")
                        lines.append("")
                    if result['message_count'] > 50:
                        lines.append(f"... and {result['message_count'] - 50} more messages")
                    text = "\n".join(lines)

                payload = {
                    "content": [{"type": "text", "text": text}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "SESSION_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"message_count": result["message_count"]}
                respond(msg_id, payload)
                continue
            if name == "tools.usage":
                days = parse_int(args.get("days", 30), 30)
                if days <= 0:
                    days = 30
                result = get_tool_usage(get_conn(), days)

                # Format results as text
                if "error" in result:
                    text = result["error"]
                else:
                    lines = [f"Tool Usage (last {result['days']} days)", ""]
                    lines.append(f"Found {result['tool_count']} tools:", "")
                    for i, tool in enumerate(result.get("tools", []), 1):
                        lines.append(f"{i}. {tool['name']}")
                        lines.append(f"   Used {tool['usage_count']} times in {tool['session_count']} sessions")
                        lines.append(f"   Last used: {tool['last_used']}")
                        lines.append("")
                    text = "\n".join(lines)

                payload = {
                    "content": [{"type": "text", "text": text}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "TOOLS_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"tool_count": result["tool_count"], "days": result["days"]}
                respond(msg_id, payload)
                continue
            if name == "platform.stats":
                days = parse_int(args.get("days", 30), 30)
                if days <= 0:
                    days = 30
                result = get_platform_stats(get_conn(), days)

                # Format results as text
                if "error" in result:
                    text = result["error"]
                else:
                    lines = [f"Platform Statistics (last {result['days']} days)", ""]
                    for i, platform in enumerate(result.get("platforms", []), 1):
                        lines.append(f"{i}. {platform['name']}")
                        lines.append(f"   Events: {platform['event_count']} | Sessions: {platform['session_count']}")
                        lines.append(f"   Active: {platform['first_seen']} to {platform['last_seen']}")
                        lines.append("")
                    text = "\n".join(lines)

                payload = {
                    "content": [{"type": "text", "text": text}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "PLATFORM_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"platform_count": len(result.get("platforms", [])), "days": result["days"]}
                respond(msg_id, payload)
                continue
            if name == "sql.query":
                query = args.get("query", "")
                limit = parse_int(args.get("limit", 100), 100)
                if limit <= 0:
                    limit = 100
                preview = bool(args.get("preview", False))
                preview_rows = clamp_int(parse_int(args.get("preview_rows", 5), 5), 1, 50)
                preview_cell_len = clamp_int(parse_int(args.get("preview_cell_len", 80), 80), 10, 200)
                result = sql_query(get_conn(), query, limit)
                payload: Dict[str, Any] = {
                    "content": [
                        {
                            "type": "text",
                            "text": tool_text_summary(
                                name,
                                result,
                                preview=preview,
                                preview_rows=preview_rows,
                                preview_cell_len=preview_cell_len,
                            ),
                        }
                    ],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "SQL_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {
                        "limit_applied": limit,
                        "row_count": len(result.get("rows", [])),
                    }
                respond(msg_id, payload)
                continue
            if name == "semantic.search":
                query = args.get("query", "")
                limit = parse_int(args.get("limit", 20), 20)
                source = args.get("source", "sql")

                # Validate source parameter
                if source not in ("sql", "markdown", "both"):
                    source = "sql"

                result = bm25_search(query, limit, source)

                # Format results as text
                if "error" in result:
                    text = result["error"]
                else:
                    lines = [f"Found {result['count']} results for: '{result['query']}' (source: {result['source']})", ""]
                    for i, r in enumerate(result.get("results", [])[:10], 1):
                        lines.append(f"{i}. [{r['role']}] {r['text']}")

                        # Format metadata based on source
                        if r.get('source') == 'markdown':
                            lines.append(f"   Session: {r['session_id']} | File: {r.get('file', 'N/A')} | Score: {r['score']:.2f} | Source: markdown")
                        else:
                            lines.append(f"   Session: {r['session_id']} | Platform: {r.get('platform', 'N/A')} | Score: {r['score']:.2f} | Source: sql")
                        lines.append("")
                    text = "\n".join(lines)

                payload = {
                    "content": [{"type": "text", "text": text}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "SEARCH_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"result_count": result["count"], "source": result["source"]}
                respond(msg_id, payload)
                continue
            if name == "text.context":
                session_id = args.get("session_id", "")
                item_index = parse_int(args.get("item_index", 0), 0)
                line_number = parse_int(args.get("line_number", 0), 0)
                context_lines = parse_int(args.get("context_lines", 3), 3)

                # Clamp context_lines
                context_lines = max(0, min(context_lines, 20))

                try:
                    # Query for context lines
                    cursor = get_conn().cursor()
                    cursor.execute("""
                        SELECT line_number, text, role, timestamp
                        FROM events_raw
                        WHERE session_id = ?
                          AND item_index = ?
                          AND line_number >= ?
                          AND line_number <= ?
                        ORDER BY line_number
                    """, (
                        session_id,
                        item_index,
                        max(0, line_number - context_lines),
                        line_number + context_lines
                    ))

                    rows = cursor.fetchall()

                    if not rows:
                        result = {"error": f"No lines found for session {session_id}, item {item_index}, line {line_number}"}
                    else:
                        lines = []
                        for ln, text, role, ts in rows:
                            marker = ">>>" if ln == line_number else "   "
                            lines.append({
                                "line_number": ln,
                                "text": text,
                                "is_target": ln == line_number,
                                "role": role,
                                "timestamp": ts
                            })

                        result = {
                            "session_id": session_id,
                            "item_index": item_index,
                            "target_line": line_number,
                            "context_lines": context_lines,
                            "lines": lines
                        }

                        # Format text output
                        text_lines = [f"Context for session {session_id}, item {item_index}, line {line_number}:", ""]
                        for line in lines:
                            marker = ">>>" if line["is_target"] else "   "
                            text_lines.append(f"{marker} {line['line_number']:3d} | {line['text']}")
                        text = "\n".join(text_lines)

                except Exception as e:
                    result = {"error": str(e)}
                    text = str(e)

                payload = {
                    "content": [{"type": "text", "text": text if "error" not in result else result["error"]}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "CONTEXT_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"line_count": len(result["lines"])}
                respond(msg_id, payload)
                continue
            if name == "text.shell":
                cmd = args.get("cmd", "")
                cmd_args = args.get("args")
                cmd_paths = args.get("paths")
                result = run_text_shell(cmd, cmd_args, cmd_paths)
                payload = {
                    "content": [{"type": "text", "text": tool_text_summary(name, result)}],
                    "data": result,
                    "ok": "error" not in result,
                }
                if "error" in result:
                    payload["isError"] = True
                    payload["error"] = {"code": "TEXT_SHELL_ERROR", "message": result["error"]}
                else:
                    payload["meta"] = {"truncated": bool(result.get("truncated", False))}
                respond(msg_id, payload)
                continue
            respond_error(msg_id, -32601, "unknown tool")
        elif method in ("shutdown", "exit"):
            respond(msg_id, {"ok": True})
            break
        else:
            respond_error(msg_id, -32601, "unknown method")

    if conn:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

