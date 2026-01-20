#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export each session into a templated Markdown file.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return slug or "session"


def try_json_load(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if not (text.startswith("{") or text.startswith("[")):
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_payload(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return try_json_load(value)
    return str(value)


def format_payload(value: Any) -> str:
    payload = normalize_payload(value)
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return str(payload)


def fetch_sessions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        select
          session_id,
          min(timestamp) as start_time,
          max(timestamp) as end_time,
          count(*) as rows,
          group_concat(distinct platform) as platforms,
          group_concat(distinct source_file) as source_files
        from events_raw
        where session_id is not null and session_id != ''
        group by session_id
        order by start_time asc
        """
    ).fetchall()
    sessions = []
    for row in rows:
        sessions.append(
            {
                "session_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "rows": row[3],
                "platforms": row[4] or "",
                "source_files": row[5] or "",
            }
        )
    return sessions


def fetch_events(
    conn: sqlite3.Connection, session_id: str, include_meta: bool
) -> Iterable[Tuple[Any, ...]]:
    rows = conn.execute(
        """
        select
          timestamp,
          role,
          item_type,
          text,
          tool_name,
          tool_args,
          tool_result,
          is_meta
        from events_raw
        where session_id = ?
        order by timestamp asc, item_index asc
        """,
        (session_id,),
    ).fetchall()
    for row in rows:
        if not include_meta and row[7]:
            continue
        yield row


def render_session_md(
    session: Dict[str, Any],
    events: Iterable[Tuple[Any, ...]],
    include_meta: bool,
) -> str:
    session_id = session["session_id"]
    start_time = session["start_time"] or ""
    end_time = session["end_time"] or ""
    platforms = session["platforms"]
    source_files = session["source_files"]
    lines: List[str] = []
    lines.append("---")
    lines.append(f"title: \"会话 {session_id}\"")
    lines.append(f"date: {start_time[:10] if start_time else ''}")
    lines.append(f"session_id: {session_id}")
    lines.append(f"platforms: {platforms}")
    lines.append(f"source_files: {source_files}")
    lines.append(f"range: {start_time} ~ {end_time}")
    lines.append("---")
    lines.append("")
    lines.append(f"# 会话：{session_id}")
    lines.append("")
    lines.append("## 元数据")
    lines.append(f"- 起始时间: {start_time}")
    lines.append(f"- 结束时间: {end_time}")
    lines.append(f"- 平台: {platforms}")
    lines.append(f"- 来源文件: {source_files}")
    lines.append(f"- 是否包含 meta: {'是' if include_meta else '否'}")
    lines.append("")
    lines.append("## 对话")
    for ts, role, item_type, text, tool_name, tool_args, tool_result, is_meta in events:
        meta_flag = " [meta]" if is_meta else ""
        header = f"### {ts} | {role} | {item_type}{meta_flag}"
        lines.append(header)
        if item_type == "text":
            lines.append(text or "")
        elif item_type == "tool_use":
            lines.append(f"tool_name: {tool_name or ''}")
            payload = format_payload(tool_args)
            lines.append("```json")
            lines.append(payload)
            lines.append("```")
        elif item_type == "tool_result":
            lines.append("tool_result:")
            payload = format_payload(tool_result)
            lines.append("```text")
            lines.append(payload)
            lines.append("```")
        else:
            content = text if text else format_payload(tool_result or tool_args)
            if content:
                lines.append(content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_sessions(db_path: Path, out_dir: Path, include_meta: bool) -> int:
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        sessions = fetch_sessions(conn)
        for session in sessions:
            session_id = session["session_id"]
            filename = safe_slug(session_id) + ".md"
            out_path = out_dir / filename
            events = fetch_events(conn, session_id, include_meta)
            out_path.write_text(
                render_session_md(session, events, include_meta), encoding="utf-8"
            )
    return len(sessions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sessions to Markdown files.")
    parser.add_argument(
        "--db",
        default=str(Path.home() / ".codemem" / "codemem.sqlite"),
        help="path to codemem sqlite db",
    )
    parser.add_argument(
        "--out",
        default=str(Path.home() / ".codemem" / "md_sessions"),
        help="output directory for markdown files",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="include meta events in output",
    )
    args = parser.parse_args()

    count = export_sessions(Path(args.db), Path(args.out), args.include_meta)
    print(f"exported {count} sessions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

