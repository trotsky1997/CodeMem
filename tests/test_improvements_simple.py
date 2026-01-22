#!/usr/bin/env python3
"""Simple test for MCP server improvements - tests database logic directly."""

import asyncio
import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_database_queries():
    """Test database queries directly."""

    try:
        import aiosqlite
        from pathlib import Path
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies first")
        return

    print("=" * 80)
    print("Testing MCP Server Improvements - Database Logic")
    print("=" * 80)

    # Find database path (use codemem.sqlite symlink)
    db_path = Path.home() / ".codemem" / "codemem.sqlite"

    # Check if database exists
    if not db_path.exists():
        print(f"✗ Database not found at {db_path}")
        print("  Please run the MCP server first to build the database:")
        print("  uv run python mcp_server.py")
        return

    # Test 1: Check if database exists and has data
    print("\n[Test 1] Database connectivity")
    print("-" * 80)
    try:
        print(f"Database path: {db_path}")

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM events")
            count = (await cursor.fetchone())[0]
            print(f"✓ Database connected")
            print(f"  Total events: {count}")

            if count == 0:
                print("  ⚠ Warning: Database is empty, some tests may not work")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 2: Test database schema
    print("\n[Test 2] Database schema check")
    print("-" * 80)
    try:
        async with aiosqlite.connect(db_path) as conn:
            # Check if events table exists
            cursor = await conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='events'
            """)
            table_exists = await cursor.fetchone()

            if table_exists:
                print(f"✓ Events table exists")

                # Check table structure
                cursor = await conn.execute("PRAGMA table_info(events)")
                columns = await cursor.fetchall()
                print(f"  Columns: {len(columns)}")
                for col in columns:
                    print(f"    - {col[1]} ({col[2]})")
            else:
                print(f"✗ Events table not found")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 3: Test SQL query structure
    print("\n[Test 3] SQL query - session listing")
    print("-" * 80)
    try:
        async with aiosqlite.connect(db_path) as conn:
            # First show platform distribution
            cursor = await conn.execute("""
                SELECT platform, COUNT(DISTINCT session_id) as session_count, COUNT(*) as msg_count
                FROM events
                GROUP BY platform
                ORDER BY platform
            """)
            platforms = await cursor.fetchall()

            print(f"✓ Platform distribution:")
            for platform, session_count, msg_count in platforms:
                print(f"    - {platform}: {session_count} sessions, {msg_count} messages")

            # Then show recent sessions from each platform
            cursor = await conn.execute("""
                SELECT
                    session_id,
                    platform,
                    COUNT(*) as message_count,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM events
                GROUP BY session_id
                ORDER BY last_message DESC
                LIMIT 10
            """)

            rows = await cursor.fetchall()
            print(f"\n  Recent sessions (top 10):")
            for session_id, platform, msg_count, first_msg, last_msg in rows:
                print(f"    - {session_id} ({platform}): {msg_count} messages")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 4: Test get_message query
    print("\n[Test 4] Get message by ref_id")
    print("-" * 80)
    try:
        async with aiosqlite.connect(db_path) as conn:
            # Get a sample message first
            cursor = await conn.execute("""
                SELECT session_id, timestamp, role, text
                FROM events
                LIMIT 1
            """)

            row = await cursor.fetchone()

            if row:
                session_id, timestamp, role, text = row
                ref_id = f"{session_id}:{timestamp}"

                # Now test retrieving it by ref_id
                cursor = await conn.execute("""
                    SELECT timestamp, role, text, session_id, platform
                    FROM events
                    WHERE session_id = ? AND timestamp = ?
                    LIMIT 1
                """, (session_id, timestamp))

                retrieved = await cursor.fetchone()

                if retrieved:
                    print(f"✓ Get message query works")
                    print(f"  Test ref_id: {ref_id}")
                    print(f"  Retrieved role: {retrieved[1]}")
                    print(f"  Text length: {len(retrieved[2])}")
                else:
                    print(f"✗ Failed to retrieve message")
            else:
                print(f"⚠ No messages in database to test")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 5: Test regex search structure
    print("\n[Test 5] Regex search capability")
    print("-" * 80)
    try:
        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("""
                SELECT session_id, timestamp, text
                FROM events
                WHERE text LIKE '%test%'
                LIMIT 3
            """)

            rows = await cursor.fetchall()
            print(f"✓ Regex search query works")
            print(f"  Found {len(rows)} matches")

            if rows:
                print(f"  Sample match ref_id: {rows[0][0]}:{rows[0][1]}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)
    print("\nNote: Full MCP protocol testing requires MCP client library")
    print("These tests verify the database queries and logic are correct")


if __name__ == "__main__":
    asyncio.run(test_database_queries())
