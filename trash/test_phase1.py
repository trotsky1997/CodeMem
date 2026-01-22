#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for memory.query interface (Phase 1).

Tests:
- Intent recognition
- Temporal expression parsing
- Synonym expansion
- Natural language formatting
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from intent_recognition import parse_intent, QueryIntent, parse_temporal_expression, expand_synonyms
from nl_formatter import format_search_results, format_activity_summary


def test_intent_recognition():
    """Test intent recognition."""
    print("=" * 60)
    print("Testing Intent Recognition")
    print("=" * 60)

    test_cases = [
        ("我之前讨论过 Python 异步吗？", QueryIntent.SEARCH_CONTENT),
        ("上周关于数据库的对话", QueryIntent.FIND_SESSION),
        ("最近在做什么？", QueryIntent.ACTIVITY_SUMMARY),
        ("那段代码的完整上下文", QueryIntent.GET_CONTEXT),
        ("导出那次对话", QueryIntent.EXPORT),
        ("我经常问什么问题？", QueryIntent.PATTERN_DISCOVERY),
    ]

    for query, expected_intent in test_cases:
        parsed = parse_intent(query)
        status = "✅" if parsed.intent == expected_intent else "❌"
        print(f"{status} Query: {query}")
        print(f"   Intent: {parsed.intent.value} (expected: {expected_intent.value})")
        print(f"   Keywords: {parsed.keywords}")
        print(f"   Time Range: {parsed.time_range}")
        print()


def test_temporal_parsing():
    """Test temporal expression parsing."""
    print("=" * 60)
    print("Testing Temporal Expression Parsing")
    print("=" * 60)

    test_cases = [
        "昨天的对话",
        "上周关于数据库的讨论",
        "最近在做什么",
        "最近7天的活动",
        "3天前的对话",
        "本周的工作",
    ]

    for query in test_cases:
        time_range = parse_temporal_expression(query)
        print(f"Query: {query}")
        if time_range:
            start, end = time_range
            print(f"  Time Range: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  Time Range: None")
        print()


def test_synonym_expansion():
    """Test synonym expansion."""
    print("=" * 60)
    print("Testing Synonym Expansion")
    print("=" * 60)

    test_cases = [
        ["Python", "异步"],
        ["数据库", "优化"],
        ["性能", "问题"],
    ]

    for keywords in test_cases:
        expanded = expand_synonyms(keywords)
        print(f"Keywords: {keywords}")
        print(f"Expanded: {expanded}")
        print()


def test_nl_formatting():
    """Test natural language formatting."""
    print("=" * 60)
    print("Testing Natural Language Formatting")
    print("=" * 60)

    # Mock search results
    mock_results = [
        {
            "session_id": "20260119_153045_abc123",
            "timestamp": "2026-01-19T15:30:45",
            "role": "assistant",
            "text": "async def build_bm25_indexes_parallel(): 使用 ProcessPoolExecutor 并行构建两个索引",
            "score": 0.85,
            "source": "sql",
            "item_index": 5
        },
        {
            "session_id": "20260118_140000_def456",
            "timestamp": "2026-01-18T14:00:00",
            "role": "user",
            "text": "如何实现 Python 异步编程？",
            "score": 0.72,
            "source": "markdown"
        }
    ]

    formatted = format_search_results(
        query="Python 异步",
        results=mock_results,
        source="both"
    )

    print("Query: Python 异步")
    print(f"\nSummary:\n{formatted['summary']}")
    print(f"\nInsights:")
    for insight in formatted['insights']:
        print(f"  - {insight}")
    print(f"\nKey Findings: {len(formatted['key_findings'])} results")
    print(f"\nSuggestions:")
    for suggestion in formatted['suggestions']:
        print(f"  - {suggestion}")
    print()

    # Mock activity data
    mock_activity = {
        "days": 7,
        "sessions": [
            {
                "session_id": "20260119_153045_abc123",
                "platforms": "claude",
                "event_count": 25,
                "first_seen": "2026-01-19T15:30:00",
                "last_seen": "2026-01-19T16:45:00",
                "sample_messages": ["讨论 Python 异步优化", "实现并行索引构建"]
            },
            {
                "session_id": "20260118_140000_def456",
                "platforms": "codex",
                "event_count": 15,
                "first_seen": "2026-01-18T14:00:00",
                "last_seen": "2026-01-18T15:30:00",
                "sample_messages": ["学习 asyncio 基础"]
            }
        ]
    }

    formatted_activity = format_activity_summary(mock_activity)

    print("\n" + "=" * 60)
    print("Activity Summary Formatting")
    print("=" * 60)
    print(f"\nSummary:\n{formatted_activity['summary']}")
    print(f"\nInsights:")
    for insight in formatted_activity['insights']:
        print(f"  - {insight}")
    print()


if __name__ == "__main__":
    print("\n🧪 CodeMem Phase 1 Test Suite\n")

    test_intent_recognition()
    test_temporal_parsing()
    test_synonym_expansion()
    test_nl_formatting()

    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
