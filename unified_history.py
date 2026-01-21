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
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Optional

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
    """Load JSON or JSONL files. Single JSON files are yielded as-is."""
    with path.open("r", encoding="utf-8") as f:
        # Check if it's a single JSON file (OpenCode msg_*.json)
        if path.suffix == ".json":
            try:
                obj = json.load(f)
                if isinstance(obj, dict):
                    obj["_source"] = str(path)
                    yield obj
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            item["_source"] = str(path)
                        yield item
            except json.JSONDecodeError:
                yield {
                    "_parse_error": True,
                    "_raw": f.read(),
                    "_source": str(path),
                }
            return

        # JSONL format (one JSON per line)
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
    if "/opencode/" in normalized or "\\opencode\\" in str(source_file):
        return "opencode"
    if "/cursor/" in normalized.lower() or "\\cursor\\" in str(source_file).lower():
        return "cursor"
    return "unknown"


def extract_claude_items(
    record: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[bool]]:
    """Extract items from Claude Code format."""
    rtype = record.get("type")
    role = None
    items: List[Dict[str, Any]] = []
    is_meta = None

    if rtype in ("user", "assistant"):
        role = rtype
        message = record.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        ctype = c.get("type")
                        if ctype == "text":
                            items.append({"type": "text", "text": c.get("text")})
                        elif ctype == "thinking":
                            # Store thinking text in 'text' field for consistency
                            items.append({"type": "thinking", "text": c.get("thinking")})
                        elif ctype == "tool_use":
                            items.append({
                                "type": "tool_use",
                                "name": c.get("name"),
                                "input": c.get("input"),
                            })
                        elif ctype == "tool_result":
                            items.append({"type": "tool_result", "content": c.get("content")})
                        else:
                            items.append({"type": ctype or "unknown", "text": str(c)})
            elif isinstance(content, str):
                items.append({"type": "text", "text": content})
    elif rtype == "summary":
        role = "system"
        is_meta = True
        summary_text = record.get("summary", "")
        if summary_text:
            items.append({"type": "text", "text": summary_text})
    elif rtype == "system":
        # System messages - mark as meta, no content to index
        role = "system"
        is_meta = True
        # Don't add any items - these are metadata records

    return role, items, is_meta


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


def extract_opencode_items(
    record: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[bool]]:
    """Extract items from OpenCode message format."""
    role = record.get("role")
    content = record.get("content")
    items: List[Dict[str, Any]] = []
    is_meta = None

    # OpenCode message format: {role, content: [{type, text/...}]}
    if isinstance(content, str):
        items.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "text")

            if item_type == "text":
                items.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "tool_use":
                items.append({
                    "type": "tool_use",
                    "name": item.get("name"),
                    "input": item.get("input"),
                })
            elif item_type == "tool_result":
                items.append({
                    "type": "tool_result",
                    "content": item.get("content"),
                })
            elif item_type == "image":
                items.append({"type": "image", "text": item.get("source", {}).get("data", "")})
            elif item_type == "reasoning":
                # OpenCode reasoning blocks have text
                items.append({"type": "reasoning", "text": item.get("text", "")})
            elif item_type in ("step-start", "step-finish"):
                # Skip step markers, they're metadata
                continue
            else:
                # For other types, try to extract text or convert to string
                text = item.get("text", "")
                if not text:
                    text = str(item)
                items.append({"type": item_type, "text": text})

    return role, items, is_meta


def extract_cursor_items(
    record: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[bool]]:
    """Extract items from Cursor chat format.

    New format: record has 'message' dict with 'role' and 'content'
    """
    items: List[Dict[str, Any]] = []
    is_meta = None

    # Extract from message dict
    message = record.get("message")
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
    else:
        role = None
        content = None

    # Parse content
    if isinstance(content, str):
        items.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "text")
                if item_type == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                else:
                    items.append({"type": item_type, "text": str(item)})
            elif isinstance(item, str):
                items.append({"type": "text", "text": item})
    elif isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if text:
            items.append({"type": "text", "text": text})

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
    # Limit to 1000 characters
    if len(value) > 1000:
        return value[:1000] + "...[truncated]"
    return value


def summarize_tool_args(value: Any) -> str:
    """Summarize tool arguments to save space."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except TypeError:
            value = str(value)
    if not isinstance(value, str):
        value = str(value)
    # Limit to 500 characters for args
    if len(value) > 500:
        return value[:500] + "...[truncated]"
    return value


def load_agent_messages(agent_id: str, base_dir: Path) -> List[Dict[str, Any]]:
    agent_path = base_dir / f"agent-{agent_id}.jsonl"
    if not agent_path.exists():
        return []
    return list(iter_jsonl(agent_path))


def load_cursor_chats(db_path: Path) -> List[Dict[str, Any]]:
    """Load chat history from Cursor's state.vscdb SQLite database.

    New Cursor format:
    - Workspace DB: composer.composerData in ItemTable
    - Global DB: composerData and bubbleId in cursorDiskKV table
    """
    records = []

    try:
        # Determine if this is a workspace or global database
        is_workspace_db = "workspaceStorage" in str(db_path)
        is_global_db = "globalStorage" in str(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        if is_workspace_db:
            # Workspace DB: Get composer IDs from composer.composerData
            cursor.execute("""
                SELECT value FROM ItemTable
                WHERE key = 'composer.composerData'
            """)

            result = cursor.fetchone()
            if result and result[0]:
                try:
                    composer_data = json.loads(result[0])
                    all_composers = composer_data.get('allComposers', [])

                    # Get global DB path (sibling to workspace storage)
                    workspace_storage_dir = db_path.parent.parent
                    global_db_path = workspace_storage_dir.parent / 'globalStorage' / 'state.vscdb'

                    if global_db_path.exists():
                        # Load conversations from global DB
                        records.extend(load_cursor_global_conversations(
                            global_db_path,
                            [c.get('composerId') for c in all_composers if c.get('composerId')],
                            str(db_path)
                        ))
                except json.JSONDecodeError:
                    pass

        elif is_global_db:
            # Global DB: Load all conversations from cursorDiskKV
            records.extend(load_cursor_global_all_conversations(db_path))

        conn.close()

    except sqlite3.Error:
        pass

    return records


def load_cursor_global_conversations(
    global_db_path: Path,
    composer_ids: List[str],
    source_path: str
) -> List[Dict[str, Any]]:
    """Load conversations for specific composer IDs from global database."""
    records = []

    try:
        conn = sqlite3.connect(str(global_db_path))
        cursor = conn.cursor()

        for composer_id in composer_ids:
            # Get composer metadata
            cursor.execute("""
                SELECT value FROM cursorDiskKV
                WHERE key = ?
            """, (f"composerData:{composer_id}",))

            result = cursor.fetchone()
            if not result or not result[0]:
                continue

            try:
                composer_data = json.loads(result[0])
                conversation_headers = composer_data.get('fullConversationHeadersOnly', [])
                created_at = composer_data.get('createdAt', 0)

                # Load each bubble in the conversation
                for header in conversation_headers:
                    bubble_id = header.get('bubbleId')
                    if not bubble_id:
                        continue

                    cursor.execute("""
                        SELECT value FROM cursorDiskKV
                        WHERE key = ?
                    """, (f"bubbleId:{composer_id}:{bubble_id}",))

                    bubble_result = cursor.fetchone()
                    if bubble_result and bubble_result[0]:
                        try:
                            bubble_data = json.loads(bubble_result[0])

                            # Convert to unified format
                            record = {
                                "sessionId": composer_id[:8],  # Shortened for consistency
                                "timestamp": str(created_at) if created_at else None,  # Convert to string
                                "type": "message",
                                "message": {
                                    "role": "user" if bubble_data.get('type') == 1 else "assistant",
                                    "content": bubble_data.get('text', '')
                                },
                                "_source": source_path,
                                "_cursor_composer_id": composer_id,
                                "_cursor_bubble_id": bubble_id,
                                "_cursor_type": bubble_data.get('type')
                            }
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
            except json.JSONDecodeError:
                continue

        conn.close()
    except sqlite3.Error:
        pass

    return records


def load_cursor_global_all_conversations(db_path: Path) -> List[Dict[str, Any]]:
    """Load all conversations from global database (when scanning global DB directly)."""
    records = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get all composer IDs
        cursor.execute("""
            SELECT key FROM cursorDiskKV
            WHERE key LIKE 'composerData:%'
        """)

        composer_ids = [row[0].split(':', 1)[1] for row in cursor.fetchall()]
        conn.close()

        # Load conversations for all composers
        records.extend(load_cursor_global_conversations(
            db_path,
            composer_ids,
            str(db_path)
        ))

    except sqlite3.Error:
        pass

    return records


def load_opencode_message_with_parts(msg_path: Path) -> Dict[str, Any]:
    """Load OpenCode message and merge with its parts.

    OpenCode stores messages in two places:
    - Message metadata: storage/message/<session-id>/msg_*.json
    - Message content: storage/part/msg_<message-id>/prt_*.json
    """
    # Load message metadata
    with msg_path.open("r", encoding="utf-8") as f:
        message = json.load(f)

    message["_source"] = str(msg_path)

    # Find storage root (go up to storage directory)
    storage_dir = msg_path.parent.parent.parent  # message/<session>/<file> -> storage
    part_dir = storage_dir / "part" / msg_path.stem  # msg_<id>

    # Load parts if they exist
    content_parts = []
    if part_dir.exists():
        part_files = sorted(part_dir.glob("prt_*.json"))
        for part_file in part_files:
            try:
                with part_file.open("r", encoding="utf-8") as f:
                    part = json.load(f)
                    content_parts.append(part)
            except (json.JSONDecodeError, IOError):
                continue

    # Add content to message
    if content_parts:
        message["content"] = content_parts

    # Extract timestamp from time.created
    if "time" in message and "created" in message["time"]:
        message["timestamp"] = message["time"]["created"]

    return message


def collect_files(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix in (".jsonl", ".json"):
            files.append(root)
            continue
        if root.is_dir():
            # Collect JSONL files (Claude, Codex)
            files.extend(
                p for p in root.rglob("*.jsonl") if not p.name.startswith("agent-")
            )
            # Collect JSON files (OpenCode messages)
            # OpenCode stores messages as msg_*.json in storage/message/<session-id>/
            if "opencode" in str(root).lower():
                files.extend(
                    p for p in root.rglob("msg_*.json")
                )
            # Collect Cursor SQLite databases
            # Cursor stores chats in state.vscdb in workspaceStorage/<hash>/ and globalStorage/
            if "cursor" in str(root).lower():
                # Collect workspace databases
                files.extend(
                    p for p in root.rglob("state.vscdb")
                    if "workspaceStorage" in str(p)
                )
                # Also collect global database if it exists
                global_db = root / "User" / "globalStorage" / "state.vscdb"
                if global_db.exists():
                    files.append(global_db)
    return sorted(set(files))


def load_records(files: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    # Parallel file loading with ThreadPoolExecutor
    def load_file(p: Path) -> List[Dict[str, Any]]:
        # Handle Cursor SQLite databases separately
        if p.name == "state.vscdb":
            return load_cursor_chats(p)
        # Handle OpenCode messages (need to load with parts)
        if p.name.startswith("msg_") and "opencode" in str(p).lower():
            return [load_opencode_message_with_parts(p)]
        return list(iter_jsonl(p))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_file, p): p for p in files}
        for future in as_completed(futures):
            try:
                records.extend(future.result())
            except Exception as exc:
                # Log error but continue processing other files
                print(f"Error loading {futures[future]}: {exc}", file=sys.stderr)

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
        # Get content for deduplication
        # OpenCode has content at top level, others have it in message
        msg = r.get("message") or {}
        content = msg.get("content") or r.get("content")
        items = normalize_content(content)

        key = (
            r.get("type"),
            r.get("timestamp"),
            r.get("isMeta"),
            r.get("sessionId") or r.get("sessionID"),  # OpenCode uses sessionID
            content_key(items),
        )
        prev = seen.get(key)
        if prev is None:
            seen[key] = r
            deduped.append(r)
        else:
            prev_msg = prev.get("message") or {}
            prev_content = prev_msg.get("content") or prev.get("content")
            prev_items = normalize_content(prev_content)
            if len(items) > len(prev_items):
                seen[key] = r
                deduped[-1] = r

    return deduped


def to_df(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert records to list of dicts (replaces pandas DataFrame)."""
    rows = []
    turn_counters: Dict[str, int] = {}
    current_turn: Dict[str, str] = {}
    for r in records:
        source_file = r.get("_source")
        platform = detect_platform(source_file)

        if platform == "claude":
            role, items, is_meta = extract_claude_items(r)
            session_id = shorten_uuid(r.get("sessionId"))
            message_id = shorten_uuid(r.get("uuid"))
        elif platform == "codex":
            role, items, is_meta = extract_codex_items(r)
            msg = r.get("payload") or {}
            session_id = (
                r.get("sessionId")
                or msg.get("id")
                or session_id_from_source(source_file)
            )
            session_id = shorten_uuid(session_id)
            message_id = shorten_uuid(msg.get("id") or r.get("uuid"))
        elif platform == "opencode":
            role, items, is_meta = extract_opencode_items(r)
            # OpenCode: extract session_id from file path (storage/message/<session-id>/msg_*.json)
            session_id = None
            if source_file:
                parts = Path(source_file).parts
                if "message" in parts:
                    msg_idx = parts.index("message")
                    if msg_idx + 1 < len(parts):
                        session_id = shorten_uuid(parts[msg_idx + 1])
            message_id = shorten_uuid(r.get("id"))
            # Convert timestamp to string for OpenCode
            timestamp = r.get("timestamp")
            if timestamp and isinstance(timestamp, (int, float)):
                timestamp = str(timestamp)
        elif platform == "cursor":
            role, items, is_meta = extract_cursor_items(r)
            # Use sessionId from record (new format)
            session_id = r.get("sessionId")
            message_id = shorten_uuid(r.get("_cursor_bubble_id") or r.get("id") or r.get("messageId"))
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

        # Ensure timestamp is string for all platforms
        timestamp = r.get("timestamp")
        if timestamp and isinstance(timestamp, (int, float)):
            timestamp = str(timestamp)

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
            index_text = text if item_type in ("text", "thinking") else summary

            # Split text into lines if it's a text or thinking item
            if text and item_type in ("text", "thinking") and "\n" in text:
                lines = text.split("\n")
                for line_no, line in enumerate(lines):
                    # Create a row for each line
                    try:
                        row = UnifiedEventRow(
                            platform=platform,
                            session_id=session_id,
                            message_id=message_id,
                            turn_id=turn_id,
                            item_index=idx,
                            line_number=line_no,
                            timestamp=timestamp,
                            role=role,
                            is_meta=is_meta,
                            agent_id=r.get("agentId"),
                            is_indexable=(
                                role in ("user", "assistant")
                                and item_type in ("text", "tool_result", "thinking")
                                and line.strip()  # Only index non-empty lines
                            ),
                            item_type=item_type,
                            text=line,
                            index_text=line if line.strip() else None,
                            tool_name=it.get("name"),
                            tool_args=None,  # Only store on first line
                            tool_result=None,  # Only store on first line
                            tool_result_summary=None,  # Only store on first line
                            source_file=source_file,
                            raw_json=r if line_no == 0 else {},  # Only store raw_json on first line
                        )
                        rows.append(row.model_dump())
                    except Exception:
                        # Skip problematic records
                        pass
            else:
                # Single line or non-text item
                try:
                    row = UnifiedEventRow(
                        platform=platform,
                        session_id=session_id,
                        message_id=message_id,
                        turn_id=turn_id,
                        item_index=idx,
                        line_number=0,  # Single line items have line_number=0
                        timestamp=timestamp,
                        role=role,
                        is_meta=is_meta,
                        agent_id=r.get("agentId"),
                        is_indexable=(
                            role in ("user", "assistant")
                            and item_type in ("text", "tool_result", "thinking")
                        ),
                        item_type=item_type,
                        text=text,
                        index_text=index_text,
                        tool_name=it.get("name"),
                        tool_args=summarize_tool_args(it.get("input")),
                        tool_result=summarize_tool_result(it.get("content")),
                        tool_result_summary=summary,
                        source_file=source_file,
                        raw_json=r
                    )
                    rows.append(row.model_dump())
                except Exception:
                    # Skip problematic records
                    pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="extra root path")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--out", default="", help="output parquet/csv path")
    args = parser.parse_args()

    home = Path.home()

    # Platform-specific paths
    if sys.platform == "win32":
        # Windows paths
        cursor_path = home / "AppData" / "Roaming" / "Cursor" / "User"
        opencode_path = home / ".local" / "share" / "opencode" / "storage" / "message"
        claude_base = home / "AppData" / "Roaming" / ".claude"
        codex_base = home / "AppData" / "Roaming" / ".codex"
    elif sys.platform == "darwin":
        # macOS paths
        cursor_path = home / "Library" / "Application Support" / "Cursor" / "User"
        opencode_path = home / ".local" / "share" / "opencode" / "storage" / "message"
        claude_base = home / ".claude"
        codex_base = home / ".codex"
    else:
        # Linux paths
        cursor_path = home / ".config" / "Cursor" / "User"
        opencode_path = home / ".local" / "share" / "opencode" / "storage" / "message"
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
    if args.include_history:
        roots.append(claude_base / "history.jsonl")
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
