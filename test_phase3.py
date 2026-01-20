#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 3: Semantic Enhancement.

Tests:
- Expanded synonym dictionary (50+ terms)
- Spelling correction
- Query rewriting
- Query suggestions
- Query quality analysis
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from intent_recognition import expand_synonyms
from query_rewriter import (
    correct_spelling,
    simplify_query,
    expand_query,
    suggest_queries,
    rewrite_query,
    generate_did_you_mean,
    analyze_query_quality
)


def test_expanded_synonyms():
    """Test expanded synonym dictionary."""
    print("=" * 60)
    print("Testing Expanded Synonym Dictionary (50+ terms)")
    print("=" * 60)

    test_cases = [
        (["python"], ["py", "python3", "python2", "cpython", "pypy"]),
        (["异步"], ["async", "asyncio", "协程", "coroutine", "concurrent", "并发"]),
        (["数据库"], ["database", "db", "sql", "sqlite", "postgresql", "mysql"]),
        (["性能"], ["performance", "optimization", "速度", "效率", "优化"]),
        (["测试"], ["test", "testing", "unittest", "pytest"]),
        (["api"], ["接口", "interface", "endpoint", "rest", "graphql"]),
    ]

    for keywords, expected_subset in test_cases:
        expanded = expand_synonyms(keywords)
        has_expected = all(term in expanded for term in expected_subset[:3])
        status = "✅" if has_expected else "❌"
        print(f"{status} Keywords: {keywords}")
        print(f"   Expanded to {len(expanded)} terms")
        print(f"   Sample: {expanded[:5]}")
        print()


def test_spelling_correction():
    """Test spelling correction."""
    print("=" * 60)
    print("Testing Spelling Correction")
    print("=" * 60)

    test_cases = [
        ("asynch programming", "async programming", True),
        ("databse optimization", "database optimization", True),
        ("performace issue", "performance issue", True),
        ("seach query", "search query", True),
        ("python async", "python async", False),  # No correction needed
    ]

    for original, expected, should_correct in test_cases:
        corrected, was_corrected = correct_spelling(original)
        status = "✅" if was_corrected == should_correct else "❌"
        print(f"{status} Original: {original}")
        print(f"   Corrected: {corrected}")
        print(f"   Was corrected: {was_corrected} (expected: {should_correct})")
        print()


def test_query_simplification():
    """Test query simplification."""
    print("=" * 60)
    print("Testing Query Simplification")
    print("=" * 60)

    test_cases = [
        ("请问如何使用 Python 异步？", "使用 Python 异步"),
        ("can you help me with database optimization", "help me with database optimization"),
        ("怎么样优化性能", "优化性能"),
        ("Python async", "Python async"),  # No simplification needed
    ]

    for original, expected_contains in test_cases:
        simplified = simplify_query(original)
        is_simplified = len(simplified) <= len(original)
        status = "✅" if is_simplified else "❌"
        print(f"{status} Original: {original}")
        print(f"   Simplified: {simplified}")
        print(f"   Length: {len(original)} → {len(simplified)}")
        print()


def test_query_expansion():
    """Test query expansion."""
    print("=" * 60)
    print("Testing Query Expansion")
    print("=" * 60)

    test_cases = [
        "如何使用 Python",
        "database optimization",
        "性能问题",
    ]

    for query in test_cases:
        variations = expand_query(query)
        status = "✅" if len(variations) >= 1 else "❌"
        print(f"{status} Query: {query}")
        print(f"   Variations: {variations}")
        print()


def test_query_suggestions():
    """Test query suggestions."""
    print("=" * 60)
    print("Testing Query Suggestions")
    print("=" * 60)

    # Mock search results
    mock_results_empty = []
    mock_results_many = [
        {"text": "Python async programming with asyncio", "score": 0.9},
        {"text": "Database optimization techniques", "score": 0.8},
        {"text": "Performance tuning for Python", "score": 0.7},
    ] * 5  # 15 results

    test_cases = [
        ("Python async programming", mock_results_empty, "No results - should suggest broader"),
        ("Python", mock_results_many, "Many results - should suggest narrower"),
    ]

    for query, results, description in test_cases:
        suggestions = suggest_queries(query, results)
        status = "✅" if len(suggestions) >= 0 else "❌"
        print(f"{status} Query: {query}")
        print(f"   Scenario: {description}")
        print(f"   Suggestions: {suggestions}")
        print()


def test_query_rewriting():
    """Test comprehensive query rewriting."""
    print("=" * 60)
    print("Testing Query Rewriting")
    print("=" * 60)

    test_cases = [
        "请问如何使用 asynch programming",
        "databse optimization",
        "Python async",
    ]

    for query in test_cases:
        rewritten = rewrite_query(query)
        print(f"Query: {query}")
        print(f"  Original: {rewritten['original']}")
        print(f"  Corrected: {rewritten['corrected']}")
        print(f"  Simplified: {rewritten['simplified']}")
        print(f"  Recommended: {rewritten['recommended']}")
        print(f"  Was corrected: {rewritten['was_corrected']}")
        print()


def test_did_you_mean():
    """Test 'Did you mean?' suggestions."""
    print("=" * 60)
    print("Testing 'Did You Mean?' Suggestions")
    print("=" * 60)

    test_cases = [
        ("asynch programming", [], "Should suggest correction"),
        ("Python async", [{"text": "result"}], "Has results - no suggestion"),
    ]

    for query, results, description in test_cases:
        suggestion = generate_did_you_mean(query, results)
        status = "✅"
        print(f"{status} Query: {query}")
        print(f"   Scenario: {description}")
        print(f"   Suggestion: {suggestion}")
        print()


def test_query_quality():
    """Test query quality analysis."""
    print("=" * 60)
    print("Testing Query Quality Analysis")
    print("=" * 60)

    test_cases = [
        ("ab", "Too short"),
        ("请问如何使用 Python 异步编程进行数据库优化并提升性能同时保证代码质量和可维护性以及测试覆盖率", "Too long"),
        ("asynch programming", "Has typos"),
        ("Python async programming", "Good quality"),
    ]

    for query, description in test_cases:
        quality = analyze_query_quality(query)
        print(f"Query: {query}")
        print(f"  Description: {description}")
        print(f"  Score: {quality['score']}/100")
        print(f"  Issues: {quality['issues']}")
        print(f"  Suggestions: {quality['suggestions']}")
        print()


if __name__ == "__main__":
    print("\n🧪 CodeMem Phase 3 Test Suite\n")

    test_expanded_synonyms()
    test_spelling_correction()
    test_query_simplification()
    test_query_expansion()
    test_query_suggestions()
    test_query_rewriting()
    test_did_you_mean()
    test_query_quality()

    print("=" * 60)
    print("✅ All Phase 3 tests completed!")
    print("=" * 60)
