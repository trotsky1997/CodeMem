#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MCP stdio server for CodeMem.
Provides one tool: sql.query
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.append(str(Path(__file__).parent))

from unified_history import collect_files, load_records, to_df
from export_sessions_md import export_sessions


MD_SESSIONS_DIR = Path.home() / ".codemem" / "md_sessions"
SESSIONS_INDEX_URI = "codemem://sessions/index"
SESSIONS_URI_PREFIX = "codemem://sessions/"

# Global state for background database build
_db_build_lock = threading.Lock()
_db_ready = threading.Event()
_db_build_error: str | None = None

# Global BM25 index
_bm25_index = None
_bm25_docs = []
_bm25_metadata = []

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
            # Simple tokenization (split by whitespace and lowercase)
            tokens = text.lower().split()
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


def bm25_search(query: str, limit: int = 20) -> Dict[str, Any]:
    """Search using BM25 ranking."""
    global _bm25_index, _bm25_docs, _bm25_metadata

    # Wait for database to be ready
    try:
        wait_for_db(timeout=120.0)
    except (TimeoutError, RuntimeError) as exc:
        return {"error": f"Database not ready: {exc}"}

    # Build index if not exists
    if _bm25_index is None:
        db_path = Path.home() / ".codemem" / "codemem.sqlite"
        build_bm25_index(db_path)

        if _bm25_index is None:
            return {"error": "BM25 index not available. Database may be empty."}

    # Enforce maximum limit
    MAX_LIMIT = 50
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT

    # Tokenize query
    query_tokens = query.lower().split()

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

    return {
        "query": query,
        "results": results,
        "count": len(results)
    }


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
              source_file,
              platform
            from events_raw
            where is_indexable = 1 and index_text is not null and index_text != ''
            """
        )
        conn.execute("create index if not exists idx_events_time on events(timestamp)")
        conn.execute("create index if not exists idx_events_role on events(role)")


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
            return {
                "columns": cols,
                "rows": rows,
                "warning": f"Results truncated to {MAX_LIMIT} rows. Use more specific WHERE conditions to narrow your search."
            }

        return {"columns": cols, "rows": rows}
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
            "name": "sql.query",
            "description": "Run read-only SELECT/CTE/PRAGMA queries against the events table.",
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
            "name": "semantic.search",
            "description": "Search conversations using natural language (BM25). No SQL needed! Just describe what you're looking for.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query (e.g., 'Python debugging tips')"},
                    "limit": {"type": "integer", "default": 20, "description": "Max results to return (max 50)"},
                },
                "required": ["query"],
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
                result = bm25_search(query, limit)

                # Format results as text
                if "error" in result:
                    text = result["error"]
                else:
                    lines = [f"Found {result['count']} results for: '{result['query']}'", ""]
                    for i, r in enumerate(result.get("results", [])[:10], 1):
                        lines.append(f"{i}. [{r['role']}] {r['text']}")
                        lines.append(f"   Session: {r['session_id']} | Platform: {r['platform']} | Score: {r['score']:.2f}")
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
                    payload["meta"] = {"result_count": result["count"]}
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

