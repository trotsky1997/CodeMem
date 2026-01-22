#!/usr/bin/env python3
"""Test script for MCP server improvements."""

import asyncio
import json
from mcp_server import main_async
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_tools():
    """Test all improved MCP tools."""

    # Start MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 80)
            print("Testing MCP Server Improvements")
            print("=" * 80)

            # Test 1: semantic.search with mode=summary
            print("\n[Test 1] semantic.search with mode=summary")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "semantic.search",
                    arguments={"query": "database", "mode": "summary", "limit": 5}
                )
                print("✓ Success")
                # Debug: print raw response
                raw_text = result.content[0].text
                print(f"  Raw response length: {len(raw_text)}")
                if len(raw_text) < 200:
                    print(f"  Raw response: {raw_text}")
                data = json.loads(raw_text)
                print(f"  Total results: {data.get('total_results', 0)}")
                print(f"  Sample results: {len(data.get('sample_results', []))}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 2: semantic.search with mode=refs
            print("\n[Test 2] semantic.search with mode=refs")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "semantic.search",
                    arguments={"query": "search", "mode": "refs", "limit": 3}
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Count: {data.get('count', 0)}")
                if data.get('results'):
                    print(f"  First ref_id: {data['results'][0].get('ref_id', 'N/A')}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 3: semantic.search with mode=full
            print("\n[Test 3] semantic.search with mode=full")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "semantic.search",
                    arguments={"query": "test", "mode": "full", "limit": 2}
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Count: {data.get('count', 0)}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 4: sql.query with mode=summary
            print("\n[Test 4] sql.query with mode=summary")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "sql.query",
                    arguments={
                        "query": "SELECT session_id, COUNT(*) as count FROM events GROUP BY session_id LIMIT 10",
                        "mode": "summary"
                    }
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Total rows: {data.get('total_rows', 0)}")
                print(f"  Sample rows: {len(data.get('sample_rows', []))}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 5: sql.query with mode=full
            print("\n[Test 5] sql.query with mode=full")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "sql.query",
                    arguments={
                        "query": "SELECT * FROM events LIMIT 5",
                        "mode": "full"
                    }
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Count: {data.get('count', 0)}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 6: regex.search with mode=summary
            print("\n[Test 6] regex.search with mode=summary")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "regex.search",
                    arguments={
                        "pattern": r"\btest\b",
                        "flags": "i",
                        "mode": "summary",
                        "limit": 10
                    }
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Total matches: {data.get('total_matches', 0)}")
                print(f"  Sample matches: {len(data.get('sample_matches', []))}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 7: regex.search with mode=refs
            print("\n[Test 7] regex.search with mode=refs")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "regex.search",
                    arguments={
                        "pattern": r"\bsearch\b",
                        "mode": "refs",
                        "limit": 5
                    }
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Count: {data.get('count', 0)}")
                ref_id = None
                if data.get('matches'):
                    ref_id = data['matches'][0].get('ref_id')
                    print(f"  First ref_id: {ref_id}")

                # Test 8: get_message with ref_id from previous test
                if ref_id:
                    print("\n[Test 8] get_message with ref_id")
                    print("-" * 80)
                    try:
                        result = await session.call_tool(
                            "get_message",
                            arguments={"ref_id": ref_id}
                        )
                        print("✓ Success")
                        data = json.loads(result.content[0].text)
                        print(f"  Retrieved message for: {data.get('ref_id', 'N/A')}")
                        print(f"  Role: {data.get('role', 'N/A')}")
                        print(f"  Text length: {len(data.get('text', ''))}")
                    except Exception as e:
                        print(f"✗ Failed: {e}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 9: list_sessions
            print("\n[Test 9] list_sessions")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "list_sessions",
                    arguments={"limit": 10}
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Total sessions: {data.get('total_sessions', 0)}")
                if data.get('sessions'):
                    first_session = data['sessions'][0]
                    print(f"  First session ID: {first_session.get('session_id', 'N/A')}")
                    print(f"  Platform: {first_session.get('platform', 'N/A')}")
                    print(f"  Message count: {first_session.get('message_count', 0)}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            # Test 10: list_sessions with platform filter
            print("\n[Test 10] list_sessions with platform filter")
            print("-" * 80)
            try:
                result = await session.call_tool(
                    "list_sessions",
                    arguments={"limit": 5, "platform": "claude"}
                )
                print("✓ Success")
                data = json.loads(result.content[0].text)
                print(f"  Total sessions: {data.get('total_sessions', 0)}")
                print(f"  Filtered by: {data.get('filtered_by_platform', 'N/A')}")
            except Exception as e:
                print(f"✗ Failed: {e}")

            print("\n" + "=" * 80)
            print("Testing Complete")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_tools())
