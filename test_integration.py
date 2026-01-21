#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end integration test with actual database initialization.
Tests the complete workflow: build DB -> search -> query.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from mcp_server import (
    build_db_async,
    bm25_search_async,
    get_recent_activity_async,
    _db_ready,
    _db_path,
)


async def test_full_workflow():
    """Test complete workflow with database initialization."""
    print("="*60)
    print("CodeMem MCP Server - Full Integration Test")
    print("="*60)

    # Step 1: Build database
    print("\n[1/5] Building database...")
    db_path = Path.home() / ".codemem" / "test_codemem.db"

    try:
        await build_db_async(
            db_path=db_path,
            include_history=True,
            extra_roots=[]
        )
        print(f"✓ Database built: {db_path}")

        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            print(f"✓ Database size: {size_kb:.2f} KB")
    except Exception as e:
        print(f"⚠ Database build: {e}")
        return False

    # Step 2: Wait for database readiness
    print("\n[2/5] Waiting for database readiness...")
    try:
        await asyncio.wait_for(_db_ready.wait(), timeout=30)
        print("✓ Database ready")
    except asyncio.TimeoutError:
        print("⚠ Database readiness timeout")
        # Continue anyway

    # Step 3: Test semantic search
    print("\n[3/5] Testing semantic search...")
    test_queries = [
        "test",
        "error",
        "function",
        "测试",  # Chinese
    ]

    for query in test_queries:
        try:
            result = await bm25_search_async(query, limit=5)
            if isinstance(result, dict):
                count = result.get('count', 0)
                print(f"✓ Query '{query}': {count} results")

                # Show first result if available
                if 'results' in result and result['results']:
                    first = result['results'][0]
                    score = first.get('score', 0)
                    print(f"  Top score: {score:.4f}")
            else:
                print(f"⚠ Query '{query}': unexpected result type {type(result)}")
        except Exception as e:
            print(f"⚠ Query '{query}': {e}")

    # Step 4: Test SQL queries
    print("\n[4/5] Testing SQL queries...")

    if _db_path and _db_path.exists():
        import aiosqlite

        sql_tests = [
            ("Total events", "SELECT COUNT(*) as count FROM events_raw"),
            ("Platforms", "SELECT DISTINCT platform FROM events_raw"),
            ("Sessions", "SELECT COUNT(DISTINCT session_id) as count FROM events_raw"),
            ("User messages", "SELECT COUNT(*) as count FROM events_raw WHERE role='user'"),
            ("Assistant messages", "SELECT COUNT(*) as count FROM events_raw WHERE role='assistant'"),
        ]

        try:
            async with aiosqlite.connect(_db_path) as db:
                for name, query in sql_tests:
                    try:
                        cursor = await db.execute(query)
                        result = await cursor.fetchall()
                        print(f"✓ {name}: {result}")
                    except Exception as e:
                        print(f"⚠ {name}: {e}")
        except Exception as e:
            print(f"⚠ SQL tests: {e}")
    else:
        print("⚠ Database not available for SQL tests")

    # Step 5: Test recent activity
    print("\n[5/5] Testing recent activity...")
    try:
        activity = await get_recent_activity_async(days=7)
        if isinstance(activity, dict):
            sessions = activity.get('sessions', [])
            print(f"✓ Recent activity: {len(sessions)} sessions")

            # Show summary
            if sessions:
                total_events = sum(s.get('event_count', 0) for s in sessions)
                print(f"  Total events: {total_events}")
                print(f"  Platforms: {set(s.get('platform') for s in sessions)}")
        else:
            print(f"✓ Recent activity returned: {type(activity)}")
    except Exception as e:
        print(f"⚠ Recent activity: {e}")

    print("\n" + "="*60)
    print("Integration Test Complete")
    print("="*60)
    return True


async def test_concurrent_operations():
    """Test concurrent operations."""
    print("\n" + "="*60)
    print("Testing Concurrent Operations")
    print("="*60)

    queries = [
        "test query 1",
        "test query 2",
        "test query 3",
        "测试查询",
        "error handling",
    ]

    print(f"\nRunning {len(queries)} concurrent searches...")

    import time
    start = time.time()

    try:
        results = await asyncio.gather(
            *[bm25_search_async(q, limit=5) for q in queries],
            return_exceptions=True
        )

        elapsed = time.time() - start

        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        print(f"✓ Completed in {elapsed:.4f}s")
        print(f"✓ Successful: {successful}/{len(queries)}")
        print(f"✓ Failed: {failed}/{len(queries)}")
        print(f"✓ Average per query: {elapsed/len(queries):.4f}s")

        # Show any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠ Query {i+1} error: {result}")

    except Exception as e:
        print(f"⚠ Concurrent test failed: {e}")


async def test_cache_performance():
    """Test cache performance."""
    print("\n" + "="*60)
    print("Testing Cache Performance")
    print("="*60)

    query = "cache performance test query"

    import time

    # First query (cache miss)
    print("\nFirst query (cache miss)...")
    start = time.time()
    try:
        result1 = await bm25_search_async(query, limit=10)
        time1 = time.time() - start
        print(f"✓ Time: {time1:.4f}s")
    except Exception as e:
        print(f"⚠ First query: {e}")
        return

    # Second query (cache hit)
    print("\nSecond query (cache hit)...")
    start = time.time()
    try:
        result2 = await bm25_search_async(query, limit=10)
        time2 = time.time() - start
        print(f"✓ Time: {time2:.4f}s")
    except Exception as e:
        print(f"⚠ Second query: {e}")
        return

    # Compare
    if time1 > 0 and time2 > 0:
        speedup = time1 / time2
        print(f"\n✓ Cache speedup: {speedup:.2f}x")
        print(f"✓ Time saved: {(time1-time2)*1000:.2f}ms")

    # Verify results match
    if result1 == result2:
        print("✓ Cached results match original")
    else:
        print("⚠ Cached results differ from original")


async def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("CODEMEM MCP SERVER - COMPREHENSIVE INTEGRATION TEST")
    print("="*60)

    # Test 1: Full workflow
    success = await test_full_workflow()

    if success:
        # Test 2: Concurrent operations
        await test_concurrent_operations()

        # Test 3: Cache performance
        await test_cache_performance()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
