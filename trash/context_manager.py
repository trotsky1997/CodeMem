#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation context management for Phase 2.

Supports:
- Context state storage
- Query history tracking
- Result caching
- Reference resolution
- Context expiration
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class SearchResult:
    """Search result with metadata."""
    session_id: str
    timestamp: str
    role: str
    text: str
    score: float
    source: str
    item_index: Optional[int] = None
    line_number: Optional[int] = None
    event_id: Optional[str] = None


@dataclass
class ConversationContext:
    """
    Conversation context for follow-up queries.

    Tracks query history, results, and focused items for reference resolution.
    """
    context_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Query history
    query_history: List[str] = field(default_factory=list)

    # Result history (last N queries)
    result_history: List[List[SearchResult]] = field(default_factory=list)

    # Currently focused items
    focused_session: Optional[str] = None
    focused_item_index: Optional[int] = None
    focused_results: List[SearchResult] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_query(self, query: str, results: List[SearchResult]):
        """Add a query and its results to history."""
        self.query_history.append(query)
        self.result_history.append(results)
        self.focused_results = results

        # Update focused session if results exist
        if results:
            self.focused_session = results[0].session_id
            self.focused_item_index = results[0].item_index

        # Keep only last 10 queries
        if len(self.query_history) > 10:
            self.query_history = self.query_history[-10:]
            self.result_history = self.result_history[-10:]

        self.last_accessed = time.time()

    def get_last_results(self) -> List[SearchResult]:
        """Get results from last query."""
        if self.result_history:
            return self.result_history[-1]
        return []

    def get_result_by_rank(self, rank: int) -> Optional[SearchResult]:
        """Get result by rank (1-indexed)."""
        last_results = self.get_last_results()
        if 0 < rank <= len(last_results):
            return last_results[rank - 1]
        return None

    def is_expired(self, ttl_seconds: int = 1800) -> bool:
        """Check if context has expired (default 30 minutes)."""
        return (time.time() - self.last_accessed) > ttl_seconds

    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()


class ContextManager:
    """
    Manages conversation contexts.

    Features:
    - Context creation and retrieval
    - Automatic expiration
    - LRU eviction
    """

    def __init__(self, max_contexts: int = 100, ttl_seconds: int = 1800):
        """
        Initialize context manager.

        Args:
            max_contexts: Maximum number of active contexts
            ttl_seconds: Time-to-live for contexts (default 30 minutes)
        """
        self._contexts: Dict[str, ConversationContext] = {}
        self._lock = asyncio.Lock()
        self._max_contexts = max_contexts
        self._ttl_seconds = ttl_seconds

        # Start cleanup task
        self._cleanup_task = None

    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background task to clean up expired contexts."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def cleanup_expired(self):
        """Remove expired contexts."""
        async with self._lock:
            expired_ids = [
                ctx_id for ctx_id, ctx in self._contexts.items()
                if ctx.is_expired(self._ttl_seconds)
            ]

            for ctx_id in expired_ids:
                del self._contexts[ctx_id]

            return len(expired_ids)

    async def get_or_create(self, context_id: Optional[str] = None) -> ConversationContext:
        """
        Get existing context or create new one.

        Args:
            context_id: Optional context ID. If None, creates new context.

        Returns:
            ConversationContext instance
        """
        async with self._lock:
            # Create new context if no ID provided
            if context_id is None:
                context_id = self._generate_context_id()
                context = ConversationContext(context_id=context_id)
                self._contexts[context_id] = context

                # Evict oldest if at capacity
                if len(self._contexts) > self._max_contexts:
                    await self._evict_oldest()

                return context

            # Get existing context
            if context_id in self._contexts:
                context = self._contexts[context_id]
                context.touch()
                return context

            # Context not found, create new one with same ID
            context = ConversationContext(context_id=context_id)
            self._contexts[context_id] = context
            return context

    async def _evict_oldest(self):
        """Evict oldest context (LRU)."""
        if not self._contexts:
            return

        oldest_id = min(
            self._contexts.keys(),
            key=lambda k: self._contexts[k].last_accessed
        )
        del self._contexts[oldest_id]

    def _generate_context_id(self) -> str:
        """Generate unique context ID."""
        return f"ctx_{uuid.uuid4().hex[:12]}"

    async def get_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        async with self._lock:
            total_contexts = len(self._contexts)
            total_queries = sum(len(ctx.query_history) for ctx in self._contexts.values())

            # Calculate age distribution
            now = time.time()
            ages = [(now - ctx.last_accessed) / 60 for ctx in self._contexts.values()]

            return {
                "total_contexts": total_contexts,
                "total_queries": total_queries,
                "avg_queries_per_context": total_queries / total_contexts if total_contexts > 0 else 0,
                "oldest_context_minutes": max(ages) if ages else 0,
                "newest_context_minutes": min(ages) if ages else 0,
            }


def resolve_reference(query: str, context: ConversationContext) -> Optional[Dict[str, Any]]:
    """
    Resolve references in query to specific results.

    Supports:
    - "第一个" / "first one" → rank 1
    - "第二个" / "second one" → rank 2
    - "那段代码" / "that code" → last result with code
    - "那次对话" / "that conversation" → focused session
    - "上一个" / "previous one" → last result

    Args:
        query: Query string with potential references
        context: Conversation context

    Returns:
        Dict with resolved reference info, or None if no reference found
    """
    query_lower = query.lower()

    # Rank references
    rank_patterns = [
        (r'第一个|first one|first result|1st', 1),
        (r'第二个|second one|second result|2nd', 2),
        (r'第三个|third one|third result|3rd', 3),
        (r'第四个|fourth one|4th', 4),
        (r'第五个|fifth one|5th', 5),
    ]

    import re
    for pattern, rank in rank_patterns:
        if re.search(pattern, query_lower):
            result = context.get_result_by_rank(rank)
            if result:
                return {
                    "type": "rank",
                    "rank": rank,
                    "result": result,
                    "session_id": result.session_id,
                    "item_index": result.item_index
                }

    # "那段代码" / "that code"
    if any(kw in query_lower for kw in ["那段代码", "that code", "the code", "那个代码"]):
        # Find last result with code
        for result in reversed(context.get_last_results()):
            if any(marker in result.text for marker in ["```", "def ", "class ", "function ", "async def"]):
                return {
                    "type": "code",
                    "result": result,
                    "session_id": result.session_id,
                    "item_index": result.item_index
                }

    # "那次对话" / "that conversation"
    if any(kw in query_lower for kw in ["那次对话", "that conversation", "that session", "那个会话"]):
        if context.focused_session:
            return {
                "type": "session",
                "session_id": context.focused_session,
                "item_index": context.focused_item_index
            }

    # "上一个" / "previous one" / "last one"
    if any(kw in query_lower for kw in ["上一个", "previous one", "last one", "上个"]):
        last_results = context.get_last_results()
        if last_results:
            return {
                "type": "previous",
                "result": last_results[0],
                "session_id": last_results[0].session_id,
                "item_index": last_results[0].item_index
            }

    return None


def is_followup_query(query: str) -> bool:
    """
    Check if query is a follow-up query.

    Follow-up indicators:
    - References: "第一个", "那段代码", "那次对话"
    - Short queries: < 10 characters
    - Question words without context: "详细", "完整", "更多"

    Args:
        query: Query string

    Returns:
        True if likely a follow-up query
    """
    query_lower = query.lower()

    # Reference indicators
    reference_keywords = [
        "第一个", "第二个", "第三个",
        "那段代码", "那次对话", "那个",
        "上一个", "上个",
        "first", "second", "third",
        "that code", "that conversation",
        "previous", "last one"
    ]

    if any(kw in query_lower for kw in reference_keywords):
        return True

    # Short queries (likely follow-up)
    if len(query) < 10:
        followup_words = ["详细", "完整", "更多", "全部", "导出", "保存"]
        if any(word in query_lower for word in followup_words):
            return True

    return False
