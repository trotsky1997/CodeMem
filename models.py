#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeMem unified events models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UnifiedEvent(BaseModel):
    platform: str = Field(..., description="claude or codex")
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    turn_id: Optional[str] = None
    item_index: Optional[int] = None

    timestamp: Optional[datetime] = None
    role: Optional[str] = None
    is_meta: Optional[bool] = None
    agent_id: Optional[str] = None

    item_type: str = Field("unknown", description="text/tool_use/tool_result/thinking/image/unknown")
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_result: Optional[Any] = None

    source_file: Optional[str] = None
    raw_json: Dict[str, Any] = Field(default_factory=dict)


class UnifiedEventRow(BaseModel):
    platform: str
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    turn_id: Optional[str] = None
    item_index: Optional[int] = None
    timestamp: Optional[str] = None
    role: Optional[str] = None
    is_meta: Optional[bool] = None
    agent_id: Optional[str] = None
    is_indexable: Optional[bool] = None
    item_type: str = "unknown"
    text: Optional[str] = None
    index_text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_result: Optional[Any] = None
    tool_result_summary: Optional[str] = None
    source_file: Optional[str] = None
    raw_json: Dict[str, Any] = Field(default_factory=dict)
