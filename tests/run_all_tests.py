#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试套件 - 运行所有测试
"""

import subprocess
import sys
from pathlib import Path


def run_test(test_file, description):
    """运行单个测试文件"""
    print(f"\n{'='*70}")
    print(f"运行: {description}")
    print(f"文件: {test_file}")
    print('='*70)

    try:
        if test_file.endswith('.py'):
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=120
            )

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            return result.returncode == 0
        else:
            # pytest
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_file, '-v', '-s', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=120
            )

            print(result.stdout)
            return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"✗ 测试超时 (>120秒)")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  CodeMem MCP 服务器 - 完整测试套件".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    tests = [
        ('test_mcp_server.py', '自动化测试套件 (18 测试)', True),
        ('test_mcp_protocol.py', '协议合规性测试 (8 测试)', False),
        ('test_final_validation.py', '端到端验证测试', False),
    ]

    results = {}

    for test_file, description, use_pytest in tests:
        if not Path(test_file).exists():
            print(f"\n⚠ 跳过 {test_file} (文件不存在)")
            results[description] = None
            continue

        if use_pytest:
            success = run_test(test_file, description)
        else:
            success = run_test(test_file, description)

        results[description] = success

    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    total = len([r for r in results.values() if r is not None])
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\n总计: {total} 个测试套件")
    print(f"通过: {passed} ✓")
    print(f"失败: {failed} ✗")
    print(f"跳过: {skipped} ○")

    if failed == 0 and passed > 0:
        print(f"\n成功率: {passed/total*100:.1f}%")
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n成功率: {passed/total*100:.1f}%")
        print("\n⚠ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
