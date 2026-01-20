#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeMem unified chat history loader.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from models import UnifiedEventRow

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
_HEX32_RE = re.compile(r"^[0-9a-fA-F]{32}$")


@dataclass
class LoadStats:
    files: int = 0
    records: int = 0
    rows: int = 0


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj["_source"] = str(path)
                yield obj
            except json.JSONDecodeError:
                yield {
                    "_parse_error": True,
                    "_line_no": line_no,
                    "_raw": line,
                    "_source": str(path),
                }


def normalize_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        items = []
        for item in content:
            if isinstance(item, dict) and "type" in item:
                items.append(item)
            else:
                items.append({"type": "text", "text": str(item)})
        return items
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def content_key(items: List[Dict[str, Any]]) -> str:
    parts = []
    for it in items:
        t = it.get("type", "unknown")
        if t == "text":
            parts.append(f"text:{it.get('text','')}")
        elif t in ("tool_use", "tool_result"):
            parts.append(f"{t}:{it.get('name','')}")
        else:
            parts.append(t)
    return "|".join(parts)


def shorten_uuid(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    s = str(value)
    if _UUID_RE.match(s) or _HEX32_RE.match(s):
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def session_id_from_source(source_file: Optional[str]) -> Optional[str]:
    if not source_file:
        return None
    name = Path(source_file).name
    if name.startswith("rollout-") and name.endswith(".jsonl"):
        base = name[len("rollout-") : -len(".jsonl")]
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        if "T" in base:
            date_part = base.split("T", 1)[0]
            return f"{date_part}-{digest}"
        return digest
    return None


def detect_platform(source_file: Optional[str]) -> str:
    if not source_file:
        return "unknown"
    normalized = str(source_file).replace("\\", "/")
    if "/.claude/" in normalized:
        return "claude"
    if "/.codex/" in normalized:
        return "codex"
    return "unknown"


def extract_codex_items(
    record: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[bool]]:
    rtype = record.get("type")
    payload = record.get("payload") or {}
    role = None
    items: List[Dict[str, Any]] = []
    is_meta = None

    if rtype == "response_item":
        ptype = payload.get("type")
        if ptype == "message":
            role = payload.get("role")
            for c in payload.get("content") or []:
                ctype = c.get("type")
                if ctype in ("input_text", "output_text", "summary_text"):
                    items.append({"type": "text", "text": c.get("text")})
                elif ctype == "input_image":
                    items.append({"type": "image", "text": c.get("image_url")})
                elif ctype in ("tool_call", "tool_use"):
                    items.append(
                        {
                            "type": "tool_use",
                            "name": c.get("name"),
                            "input": c.get("arguments") or c.get("input"),
                        }
                    )
                elif ctype == "tool_result":
                    items.append({"type": "tool_result", "content": c.get("content")})
                else:
                    items.append({"type": ctype or "unknown", "text": c.get("text")})
        elif ptype == "reasoning":
            role = "assistant"
            for s in payload.get("summary") or []:
                if s.get("type") == "summary_text":
                    items.append({"type": "thinking", "text": s.get("text")})
        elif ptype == "function_call":
            role = "assistant"
            args = payload.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                pass
            items.append(
                {
                    "type": "tool_use",
                    "name": payload.get("name"),
                    "input": args,
                }
            )
        elif ptype == "function_call_output":
            role = "user"
            items.append({"type": "tool_result", "content": payload.get("output")})
        elif ptype == "custom_tool_call":
            role = "assistant"
            items.append(
                {
                    "type": "tool_use",
                    "name": payload.get("name"),
                    "input": payload.get("input"),
                }
            )
        elif ptype == "custom_tool_call_output":
            role = "user"
            items.append({"type": "tool_result", "content": payload.get("output")})
    elif rtype == "event_msg":
        ptype = payload.get("type")
        if ptype == "user_message":
            role = "user"
            items.append({"type": "text", "text": payload.get("message")})
        elif ptype == "agent_message":
            role = "assistant"
            items.append({"type": "text", "text": payload.get("message")})
    elif rtype in ("session_meta", "turn_context", "token_count"):
        role = "system"
        is_meta = True

    return role, items, is_meta


def summarize_tool_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except TypeError:
            value = str(value)
    if not isinstance(value, str):
        value = str(value)
    lines = value.splitlines()
    head = "\n".join(lines[:2])
    return head[:1000]


def load_agent_messages(agent_id: str, base_dir: Path) -> List[Dict[str, Any]]:
    agent_path = base_dir / f"agent-{agent_id}.jsonl"
    if not agent_path.exists():
        return []
    return list(iter_jsonl(agent_path))


def collect_files(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".jsonl":
            files.append(root)
            continue
        if root.is_dir():
            files.extend(
                p for p in root.rglob("*.jsonl") if not p.name.startswith("agent-")
            )
    return sorted(set(files))


def load_records(files: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in files:
        records.extend(iter_jsonl(p))

    first_agent_idx: Dict[str, int] = {}
    for i, r in enumerate(records):
        agent_id = r.get("agentId")
        if agent_id and agent_id not in first_agent_idx:
            first_agent_idx[agent_id] = i
    if first_agent_idx:
        offset = 0
        for agent_id, idx in sorted(first_agent_idx.items(), key=lambda x: x[1]):
            source = records[idx].get("_source")
            base_dir = Path(source).parent if source else (files[0].parent if files else Path("."))
            agent_msgs = load_agent_messages(agent_id, base_dir)
            insert_at = idx + 1 + offset
            records[insert_at:insert_at] = agent_msgs
            offset += len(agent_msgs)

    def ts_key(r: Dict[str, Any]) -> int:
        ts = r.get("timestamp", 0)
        if isinstance(ts, (int, float)):
            return int(ts)
        try:
            return int(str(ts))
        except (TypeError, ValueError):
            return 0

    records.sort(key=ts_key)

    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    deduped: List[Dict[str, Any]] = []
    for r in records:
        msg = r.get("message") or {}
        items = normalize_content(msg.get("content"))
        key = (
            r.get("type"),
            r.get("timestamp"),
            r.get("isMeta"),
            r.get("sessionId"),
            content_key(items),
        )
        prev = seen.get(key)
        if prev is None:
            seen[key] = r
            deduped.append(r)
        else:
            prev_items = normalize_content((prev.get("message") or {}).get("content"))
            if len(items) > len(prev_items):
                seen[key] = r
                deduped[-1] = r

    return deduped


def to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    turn_counters: Dict[str, int] = {}
    current_turn: Dict[str, str] = {}
    for r in records:
        source_file = r.get("_source")
        platform = detect_platform(source_file)

        if platform == "codex":
            role, items, is_meta = extract_codex_items(r)
            msg = r.get("payload") or {}
            session_id = (
                r.get("sessionId")
                or msg.get("id")
                or session_id_from_source(source_file)
            )
            session_id = shorten_uuid(session_id)
            message_id = shorten_uuid(msg.get("id") or r.get("uuid"))
        else:
            msg = r.get("message") or {}
            items = normalize_content(msg.get("content"))
            display_text = r.get("display")
            if not items:
                if display_text:
                    items = [{"type": "text", "text": display_text}]
                else:
                    items = [{"type": "unknown"}]
            role = msg.get("role") or r.get("type")
            is_meta = r.get("isMeta")
            session_id = shorten_uuid(r.get("sessionId"))
            message_id = shorten_uuid(msg.get("id") or r.get("uuid"))

        if not items:
            items = [{"type": "unknown"}]

        if session_id and role == "user":
            turn_counters[session_id] = turn_counters.get(session_id, 0) + 1
            current_turn[session_id] = f"{session_id}:{turn_counters[session_id]}"
        turn_id = current_turn.get(session_id) if session_id else None
        if role == "system":
            turn_id = None

        for idx, it in enumerate(items):
            item_type = it.get("type", "unknown")
            summary = ""
            if item_type == "tool_result":
                summary = summarize_tool_result(it.get("content"))
            text = it.get("text")
            index_text = text if item_type == "text" else summary
            row = UnifiedEventRow(
                platform=platform,
                session_id=session_id,
                message_id=message_id,
                turn_id=turn_id,
                item_index=idx,
                timestamp=r.get("timestamp"),
                role=role,
                is_meta=is_meta,
                agent_id=r.get("agentId"),
                is_indexable=(
                    role in ("user", "assistant")
                    and item_type in ("text", "tool_result")
                ),
                item_type=item_type,
                text=text,
                index_text=index_text,
                tool_name=it.get("name"),
                tool_args=it.get("input"),
                tool_result=it.get("content"),
                tool_result_summary=summary,
                source_file=source_file,
                raw_json=r,
            )
            rows.append(row.model_dump())
    if not rows:
        if hasattr(UnifiedEventRow, "model_fields"):
            cols = list(UnifiedEventRow.model_fields.keys())
        else:
            cols = list(UnifiedEventRow.__fields__.keys())
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="extra root path")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--out", default="", help="output parquet/csv path")
    args = parser.parse_args()

    home = Path.home()
    roots = [
        home / ".claude" / "projects",
        home / ".claude" / "transcripts",
        home / ".codex" / "sessions",
    ]
    if args.include_history:
        roots.append(home / ".claude" / "history.jsonl")
    roots.extend(Path(p) for p in args.root)

    files = collect_files(roots)
    records = load_records(files)
    df = to_df(records)

    print(f"files: {len(files)}")
    print(f"records: {len(records)}")
    print(f"rows: {len(df)}")
    print(df.head(10).to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix == ".csv":
            df.to_csv(out_path, index=False)
        else:
            df = df.copy()
            for col in ("tool_args", "tool_result", "raw_json"):
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False, default=str)
                    if isinstance(v, (dict, list))
                    else ("" if v is None else v)
                )
            df.to_parquet(out_path, index=False)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
