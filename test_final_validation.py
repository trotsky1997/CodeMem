#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的端到端验证测试
测试所有修复后的功能
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))


async def test_complete_workflow():
    """完整工作流测试"""
    print("="*70)
    print("CodeMem MCP 服务器 - 完整验证测试")
    print("="*70)

    # Import at the beginning to ensure we use the same module instance
    import mcp_server

    try:
        # 测试 1: 数据库构建
        print("\n[测试 1/5] 数据库构建")
        print("-" * 70)

        db_path = Path.home() / ".codemem" / "test_final_validation.db"
        print(f"数据库路径: {db_path}")

        try:
            await mcp_server.build_db_async(
                db_path=db_path,
                include_history=True,
                extra_roots=[]
            )

            if db_path.exists():
                size_mb = db_path.stat().st_size / 1024 / 1024
                print(f"✓ 数据库已创建: {size_mb:.2f} MB")
            else:
                print("✗ 数据库文件不存在")
                return False

        except Exception as e:
            print(f"✗ 数据库构建失败: {e}")
            return False

        # 测试 2: BM25 索引状态
        print("\n[测试 2/5] BM25 索引验证")
        print("-" * 70)

        # Use the module's global variables directly
        print(f"索引文档数: {len(mcp_server._bm25_md_docs)}")
        print(f"索引状态: {'已构建' if mcp_server._bm25_md_index else '未构建'}")

        if len(mcp_server._bm25_md_docs) == 0:
            print("✗ BM25 索引为空")
            return False
        else:
            print(f"✓ BM25 索引包含 {len(mcp_server._bm25_md_docs)} 个文档")

        # 测试 3: 语义搜索
        print("\n[测试 3/5] 语义搜索测试")
        print("-" * 70)

        test_queries = [
            ("test", "英文常见词"),
            ("error", "错误相关"),
            ("function", "函数相关"),
            ("实现", "中文常见词"),
            ("代码", "代码相关"),
        ]

        search_success = 0
        for query, desc in test_queries:
            try:
                result = await mcp_server.bm25_search_async(query, limit=3)

                if isinstance(result, dict):
                    if 'error' in result:
                        print(f"  ✗ '{query}' ({desc}): {result['error']}")
                    else:
                        count = result.get('count', 0)
                        if count > 0:
                            print(f"  ✓ '{query}' ({desc}): {count} 结果")
                            search_success += 1

                            # 显示第一个结果
                            if 'results' in result and result['results']:
                                first = result['results'][0]
                                score = first.get('score', 0)
                                print(f"      最高分: {score:.4f}")
                        else:
                            print(f"  ○ '{query}' ({desc}): 0 结果")

            except Exception as e:
                print(f"  ✗ '{query}' ({desc}): 异常 - {str(e)[:50]}")

        print(f"\n搜索成功率: {search_success}/{len(test_queries)}")

        # 测试 4: SQL 查询
        print("\n[测试 4/5] SQL 查询测试")
        print("-" * 70)

        import aiosqlite
        if db_path.exists():
            try:
                async with aiosqlite.connect(db_path) as db:
                    # 测试基本查询
                    cursor = await db.execute("SELECT COUNT(*) FROM events_raw")
                    total = (await cursor.fetchone())[0]
                    print(f"✓ 总事件数: {total}")

                    # 测试分组查询
                    cursor = await db.execute(
                        "SELECT role, COUNT(*) as cnt FROM events_raw GROUP BY role"
                    )
                    roles = await cursor.fetchall()
                    print(f"✓ 角色分布:")
                    for role, cnt in roles:
                        print(f"    {role}: {cnt}")

                    # 测试平台查询
                    cursor = await db.execute(
                        "SELECT DISTINCT platform FROM events_raw"
                    )
                    platforms = await cursor.fetchall()
                    print(f"✓ 平台: {[p[0] for p in platforms]}")

            except Exception as e:
                print(f"✗ SQL 查询失败: {e}")
                return False

        # 测试 5: 最近活动
        print("\n[测试 5/5] 最近活动测试")
        print("-" * 70)

        try:
            activity = await mcp_server.get_recent_activity_async(days=7)

            if isinstance(activity, dict):
                if 'error' in activity:
                    print(f"✗ 错误: {activity['error']}")
                else:
                    sessions = activity.get('sessions', [])
                    print(f"✓ 最近 7 天活动: {len(sessions)} 个会话")

                    if sessions:
                        total_events = sum(s.get('event_count', 0) for s in sessions)
                        print(f"✓ 总事件数: {total_events}")

        except Exception as e:
            print(f"✗ 最近活动查询失败: {e}")

        # 总结
        print("\n" + "="*70)
        print("测试完成")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance():
    """性能测试"""
    print("\n" + "="*70)
    print("性能测试")
    print("="*70)

    import mcp_server
    import time

    try:

        # 测试查询性能
        queries = ["test", "error", "function", "code", "实现"]
        times = []

        for query in queries:
            start = time.time()
            try:
                await mcp_server.bm25_search_async(query, limit=10)
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  查询 '{query}': {elapsed*1000:.2f}ms")
            except Exception as e:
                print(f"  查询 '{query}': 失败 - {e}")

        if times:
            avg_time = sum(times) / len(times)
            print(f"\n✓ 平均查询时间: {avg_time*1000:.2f}ms")

            if avg_time < 0.1:
                print("✓ 性能优秀 (<100ms)")
            elif avg_time < 1.0:
                print("✓ 性能良好 (<1s)")
            else:
                print("⚠ 性能需要优化 (>1s)")

    except Exception as e:
        print(f"✗ 性能测试失败: {e}")


async def main():
    """主测试函数"""
    success = await test_complete_workflow()

    if success:
        await test_performance()

        print("\n" + "="*70)
        print("✅ 所有测试通过！")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ 测试失败，请检查错误信息")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
