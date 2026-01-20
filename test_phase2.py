#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 2: Conversation Context Management.

Tests:
- Context creation and retrieval
- Query history tracking
- Reference resolution
- Follow-up query detection
- Context expiration
"""

import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from context_manager import (
    ContextManager,
    ConversationContext,
    SearchResult,
    resolve_reference,
    is_followup_query
)


def test_followup_detection():
    """Test follow-up query detection."""
    print("=" * 60)
    print("Testing Follow-up Query Detection")
    print("=" * 60)

    test_cases = [
        ("第一个", True),
        ("那段代码", True),
        ("那次对话", True),
        ("详细", True),
        ("我之前讨论过 Python 异步吗？", False),
        ("最近在做什么？", False),
    ]

    for query, expected in test_cases:
        result = is_followup_query(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} Query: {query}")
        print(f"   Is Follow-up: {result} (expected: {expected})")
        print()


def test_reference_resolution():
    """Test reference resolution."""
    print("=" * 60)
    print("Testing Reference Resolution")
    print("=" * 60)

    # Create mock context with results
    ctx = ConversationContext(context_id="test_ctx")

    # Add mock results
    mock_results = [
        SearchResult(
            session_id="session1",
            timestamp="2026-01-20T10:00:00",
            role="assistant",
            text="async def build_bm25_indexes_parallel(): ...",
            score=0.95,
            source="sql",
            item_index=5
        ),
        SearchResult(
            session_id="session2",
            timestamp="2026-01-19T14:00:00",
            role="user",
            text="如何实现 Python 异步？",
            score=0.85,
            source="markdown"
        ),
    ]

    ctx.add_query("Python 异步", mock_results)

    # Test cases
    test_cases = [
        ("第一个", "rank", 1),
        ("第二个", "rank", 2),
        ("那段代码", "code", None),
        ("那次对话", "session", None),
        ("上一个", "previous", None),
    ]

    for query, expected_type, expected_rank in test_cases:
        resolved = resolve_reference(query, ctx)
        if resolved:
            status = "✅" if resolved["type"] == expected_type else "❌"
            print(f"{status} Query: {query}")
            print(f"   Type: {resolved['type']} (expected: {expected_type})")
            if expected_rank:
                print(f"   Rank: {resolved.get('rank', 'N/A')} (expected: {expected_rank})")
            print(f"   Session: {resolved.get('session_id', 'N/A')}")
        else:
            print(f"❌ Query: {query}")
            print(f"   No reference resolved (expected: {expected_type})")
        print()


async def test_context_manager():
    """Test context manager."""
    print("=" * 60)
    print("Testing Context Manager")
    print("=" * 60)

    manager = ContextManager(max_contexts=5, ttl_seconds=10)

    # Test 1: Create new context
    ctx1 = await manager.get_or_create()
    print(f"✅ Created context: {ctx1.context_id}")

    # Test 2: Retrieve existing context
    ctx1_retrieved = await manager.get_or_create(ctx1.context_id)
    status = "✅" if ctx1_retrieved.context_id == ctx1.context_id else "❌"
    print(f"{status} Retrieved same context: {ctx1_retrieved.context_id}")

    # Test 3: Add query to context
    mock_result = SearchResult(
        session_id="session1",
        timestamp="2026-01-20T10:00:00",
        role="assistant",
        text="Test result",
        score=0.9,
        source="sql"
    )
    ctx1.add_query("test query", [mock_result])
    print(f"✅ Added query to context. History length: {len(ctx1.query_history)}")

    # Test 4: Get last results
    last_results = ctx1.get_last_results()
    status = "✅" if len(last_results) == 1 else "❌"
    print(f"{status} Retrieved last results: {len(last_results)} results")

    # Test 5: Get result by rank
    result_rank1 = ctx1.get_result_by_rank(1)
    status = "✅" if result_rank1 is not None else "❌"
    print(f"{status} Retrieved result by rank 1: {result_rank1.text if result_rank1 else 'None'}")

    # Test 6: Context expiration
    import time
    time.sleep(0.1)
    is_expired = ctx1.is_expired(ttl_seconds=0.05)
    status = "✅" if is_expired else "❌"
    print(f"{status} Context expired after TTL: {is_expired}")

    # Test 7: Stats
    stats = await manager.get_stats()
    print(f"✅ Context manager stats:")
    print(f"   Total contexts: {stats['total_contexts']}")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Avg queries per context: {stats['avg_queries_per_context']:.2f}")

    print()


async def test_context_history():
    """Test context query history."""
    print("=" * 60)
    print("Testing Context Query History")
    print("=" * 60)

    ctx = ConversationContext(context_id="history_test")

    # Add multiple queries
    for i in range(12):
        mock_result = SearchResult(
            session_id=f"session{i}",
            timestamp=f"2026-01-20T{10+i}:00:00",
            role="assistant",
            text=f"Result {i}",
            score=0.9 - i * 0.05,
            source="sql"
        )
        ctx.add_query(f"query {i}", [mock_result])

    # Check history limit (should keep only last 10)
    status = "✅" if len(ctx.query_history) == 10 else "❌"
    print(f"{status} Query history limited to 10: {len(ctx.query_history)} queries")

    # Check focused session (should be from last query)
    status = "✅" if ctx.focused_session == "session11" else "❌"
    print(f"{status} Focused session updated: {ctx.focused_session}")

    # Check last results
    last_results = ctx.get_last_results()
    status = "✅" if len(last_results) == 1 and last_results[0].text == "Result 11" else "❌"
    print(f"{status} Last results correct: {last_results[0].text if last_results else 'None'}")

    print()


async def test_context_cleanup():
    """Test context cleanup."""
    print("=" * 60)
    print("Testing Context Cleanup")
    print("=" * 60)

    manager = ContextManager(max_contexts=3, ttl_seconds=1)

    # Create 5 contexts (should evict 2 oldest)
    contexts = []
    for i in range(5):
        ctx = await manager.get_or_create()
        contexts.append(ctx.context_id)
        await asyncio.sleep(0.1)

    stats = await manager.get_stats()
    status = "✅" if stats['total_contexts'] == 3 else "❌"
    print(f"{status} LRU eviction: {stats['total_contexts']} contexts (max 3)")

    # Wait for expiration
    await asyncio.sleep(1.5)

    # Cleanup expired
    expired_count = await manager.cleanup_expired()
    print(f"✅ Cleaned up {expired_count} expired contexts")

    stats = await manager.get_stats()
    print(f"✅ Remaining contexts: {stats['total_contexts']}")

    print()


async def main():
    """Run all tests."""
    print("\n🧪 CodeMem Phase 2 Test Suite\n")

    test_followup_detection()
    test_reference_resolution()
    await test_context_manager()
    await test_context_history()
    await test_context_cleanup()

    print("=" * 60)
    print("✅ All Phase 2 tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
