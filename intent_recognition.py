#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent recognition module for natural language queries.

Supports:
- Search content queries
- Session finding
- Context retrieval
- Activity summaries
- Pattern discovery
- Export requests
"""

from enum import Enum
from typing import Optional, Dict, Any
import re
from datetime import datetime, timedelta


class QueryIntent(Enum):
    """Query intent types."""
    SEARCH_CONTENT = "search"      # "我之前讨论过 Python 异步吗？"
    FIND_SESSION = "session"       # "上周关于数据库的对话"
    GET_CONTEXT = "context"        # "那段代码的完整上下文"
    ACTIVITY_SUMMARY = "activity"  # "最近在做什么？"
    PATTERN_DISCOVERY = "pattern"  # "我经常问什么问题？"
    EXPORT = "export"              # "导出那次对话"
    UNKNOWN = "unknown"            # Fallback


class ParsedQuery:
    """Parsed query with intent and parameters."""

    def __init__(
        self,
        intent: QueryIntent,
        original_query: str,
        keywords: Optional[list] = None,
        time_range: Optional[tuple] = None,
        limit: int = 20,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.intent = intent
        self.original_query = original_query
        self.keywords = keywords or []
        self.time_range = time_range
        self.limit = limit
        self.metadata = metadata or {}


def parse_temporal_expression(query: str) -> Optional[tuple]:
    """
    Parse temporal expressions in query.

    Returns:
        Tuple of (start_datetime, end_datetime) or None
    """
    now = datetime.now()

    # 昨天
    if any(kw in query for kw in ["昨天", "yesterday"]):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        return (start, end)

    # 今天
    if any(kw in query for kw in ["今天", "today"]):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        return (start, end)

    # 上周
    if any(kw in query for kw in ["上周", "last week"]):
        start = now - timedelta(days=7)
        return (start, now)

    # 最近 N 天
    match = re.search(r'最近(\d+)天|past (\d+) days?', query)
    if match:
        days = int(match.group(1) or match.group(2))
        start = now - timedelta(days=days)
        return (start, now)

    # 最近 (默认 7 天)
    if any(kw in query for kw in ["最近", "recent", "lately"]):
        start = now - timedelta(days=7)
        return (start, now)

    # N 天前
    match = re.search(r'(\d+)天前|(\d+) days? ago', query)
    if match:
        days = int(match.group(1) or match.group(2))
        target = now - timedelta(days=days)
        start = target.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        return (start, end)

    # 本周
    if any(kw in query for kw in ["本周", "this week"]):
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start, now)

    # 本月
    if any(kw in query for kw in ["本月", "this month"]):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (start, now)

    return None


def extract_keywords(query: str) -> list:
    """
    Extract keywords from query.

    Removes common stop words and temporal expressions.
    """
    # Remove temporal expressions
    temporal_patterns = [
        r'昨天|今天|上周|本周|本月|最近\d*天?',
        r'yesterday|today|last week|this week|this month|past \d+ days?|recently'
    ]

    cleaned = query
    for pattern in temporal_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove common question words
    question_words = [
        '吗', '呢', '么', '什么', '怎么', '如何', '为什么', '哪里', '谁',
        'what', 'how', 'why', 'where', 'who', 'when', 'which'
    ]

    for word in question_words:
        cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)

    # Remove punctuation and extra spaces
    cleaned = re.sub(r'[？?！!。.，,、；;：:""\'\'（）()]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Split into keywords
    keywords = [kw for kw in cleaned.split() if len(kw) > 1]

    return keywords


def parse_intent(query: str) -> ParsedQuery:
    """
    Parse query intent using rule-based matching.

    Args:
        query: Natural language query

    Returns:
        ParsedQuery object with intent and parameters
    """
    query_lower = query.lower()

    # 1. EXPORT intent
    export_keywords = ['导出', 'export', '下载', 'download', '保存', 'save']
    if any(kw in query_lower for kw in export_keywords):
        return ParsedQuery(
            intent=QueryIntent.EXPORT,
            original_query=query,
            keywords=extract_keywords(query)
        )

    # 2. ACTIVITY_SUMMARY intent
    activity_keywords = [
        '最近在做', '最近做了', '最近的活动', '活动摘要', '做了什么',
        'recent activity', 'what have i been doing', 'activity summary'
    ]
    if any(kw in query_lower for kw in activity_keywords):
        time_range = parse_temporal_expression(query)
        return ParsedQuery(
            intent=QueryIntent.ACTIVITY_SUMMARY,
            original_query=query,
            time_range=time_range,
            keywords=extract_keywords(query)
        )

    # 3. PATTERN_DISCOVERY intent
    pattern_keywords = [
        '经常', '频繁', '模式', '习惯', '趋势', '统计',
        'frequently', 'often', 'pattern', 'habit', 'trend', 'statistics'
    ]
    if any(kw in query_lower for kw in pattern_keywords):
        return ParsedQuery(
            intent=QueryIntent.PATTERN_DISCOVERY,
            original_query=query,
            keywords=extract_keywords(query)
        )

    # 4. GET_CONTEXT intent
    context_keywords = [
        '完整', '上下文', '全部', '详细', '整个',
        'full', 'complete', 'context', 'entire', 'whole', 'detail'
    ]
    if any(kw in query_lower for kw in context_keywords):
        return ParsedQuery(
            intent=QueryIntent.GET_CONTEXT,
            original_query=query,
            keywords=extract_keywords(query)
        )

    # 5. FIND_SESSION intent
    session_keywords = [
        '对话', '会话', '那次', '那个',
        'conversation', 'session', 'that time', 'that discussion'
    ]
    time_range = parse_temporal_expression(query)
    if any(kw in query_lower for kw in session_keywords) or time_range:
        return ParsedQuery(
            intent=QueryIntent.FIND_SESSION,
            original_query=query,
            time_range=time_range,
            keywords=extract_keywords(query)
        )

    # 6. SEARCH_CONTENT intent (default for most queries)
    search_keywords = [
        '讨论', '说过', '提到', '问过', '聊过', '之前',
        'discuss', 'mentioned', 'talked about', 'asked', 'said', 'before'
    ]

    # If has search keywords or has meaningful keywords, treat as search
    keywords = extract_keywords(query)
    if any(kw in query_lower for kw in search_keywords) or len(keywords) > 0:
        time_range = parse_temporal_expression(query)
        return ParsedQuery(
            intent=QueryIntent.SEARCH_CONTENT,
            original_query=query,
            time_range=time_range,
            keywords=keywords
        )

    # 7. UNKNOWN (fallback)
    return ParsedQuery(
        intent=QueryIntent.UNKNOWN,
        original_query=query,
        keywords=extract_keywords(query)
    )


def expand_synonyms(keywords: list) -> list:
    """
    Expand keywords with synonyms.

    Args:
        keywords: List of keywords

    Returns:
        Expanded list with synonyms
    """
    # Domain-specific synonym dictionary
    SYNONYMS = {
        "异步": ["async", "asyncio", "协程", "coroutine", "concurrent", "并发"],
        "async": ["异步", "asyncio", "协程", "coroutine", "concurrent"],
        "数据库": ["database", "db", "sql", "sqlite", "postgresql", "mysql"],
        "database": ["数据库", "db", "sql", "sqlite", "postgresql"],
        "性能": ["performance", "optimization", "速度", "效率", "优化", "快"],
        "performance": ["性能", "optimization", "速度", "效率", "优化"],
        "优化": ["optimization", "improve", "enhance", "性能", "提升"],
        "optimization": ["优化", "improve", "enhance", "性能"],
        "索引": ["index", "indexing", "bm25", "搜索"],
        "index": ["索引", "indexing", "bm25", "搜索"],
        "python": ["py", "python3", "python2"],
        "javascript": ["js", "node", "nodejs", "typescript", "ts"],
        "错误": ["error", "bug", "问题", "issue", "exception"],
        "error": ["错误", "bug", "问题", "issue", "exception"],
    }

    expanded = set(keywords)

    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in SYNONYMS:
            expanded.update(SYNONYMS[keyword_lower])

    return list(expanded)
