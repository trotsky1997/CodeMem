#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 4.5: Pattern Clustering.

Tests:
- Query clustering (similar queries)
- Topic aggregation (hierarchical topics)
- Session clustering (conversation types)
- Problem pattern recognition (recurring issues)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from pattern_clusterer import PatternClusterer, format_aggregation_report


def create_mock_events_for_clustering():
    """Create mock events for clustering tests."""
    now = datetime.now()
    events = []

    # Similar queries (should cluster together)
    similar_queries = [
        ("How to use Python async?", "user"),
        ("如何使用 Python 异步？", "user"),
        ("Python async programming tutorial", "user"),
        ("How do I use asyncio in Python?", "user"),
    ]

    for i, (text, role) in enumerate(similar_queries):
        events.append({
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "role": role,
            "text": text,
            "session_id": f"session_similar_{i}",
            "platform": "claude"
        })

    # Different topics for aggregation
    topic_queries = [
        ("Python optimization tips", "编程语言"),
        ("JavaScript async await", "编程语言"),
        ("Database performance tuning", "数据存储"),
        ("Redis caching strategy", "数据存储"),
        ("API design best practices", "API开发"),
        ("REST API authentication", "API开发"),
        ("Unit testing in Python", "测试调试"),
        ("Debug memory leak", "测试调试"),
    ]

    for i, (text, topic) in enumerate(topic_queries):
        events.append({
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "role": "user",
            "text": text,
            "session_id": f"session_topic_{i}",
            "platform": "claude"
        })

    # Different session types
    # Learning session
    learning_session = [
        ("What is Python asyncio?", "user"),
        ("Asyncio is a library for async programming", "assistant"),
        ("How to use asyncio?", "user"),
        ("Here's a tutorial...", "assistant"),
    ]

    for i, (text, role) in enumerate(learning_session):
        events.append({
            "timestamp": (now - timedelta(days=10, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_learning",
            "platform": "claude"
        })

    # Problem-solving session
    problem_session = [
        ("I'm getting an error in my async code", "user"),
        ("What's the error message?", "assistant"),
        ("TypeError: object is not callable", "user"),
        ("This error occurs when...", "assistant"),
    ]

    for i, (text, role) in enumerate(problem_session):
        events.append({
            "timestamp": (now - timedelta(days=11, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_problem",
            "platform": "claude"
        })

    # Recurring problems
    recurring_problems = [
        "My database query is too slow",
        "Database performance is slow",
        "Slow database queries",
        "API endpoint returns 500 error",
        "Getting 500 error from API",
        "500 internal server error",
    ]

    for i, text in enumerate(recurring_problems):
        events.append({
            "timestamp": (now - timedelta(days=15+i)).isoformat(),
            "role": "user",
            "text": text,
            "session_id": f"session_problem_{i}",
            "platform": "claude"
        })

    return events


def test_query_clustering():
    """Test query clustering."""
    print("=" * 60)
    print("Testing Query Clustering")
    print("=" * 60)

    events = create_mock_events_for_clustering()
    clusterer = PatternClusterer(events)

    clusters = clusterer.cluster_queries(similarity_threshold=0.5)

    print(f"✅ Found {len(clusters)} query clusters")
    for i, cluster in enumerate(clusters[:3], 1):
        print(f"\n{i}. Representative: {cluster['representative'][:60]}")
        print(f"   Count: {cluster['count']}")
        print(f"   Similar queries: {len(cluster['queries'])}")
        if len(cluster['queries']) > 1:
            print(f"   Example: {cluster['queries'][1][:60]}")
    print()


def test_topic_aggregation():
    """Test topic aggregation."""
    print("=" * 60)
    print("Testing Topic Aggregation")
    print("=" * 60)

    events = create_mock_events_for_clustering()
    clusterer = PatternClusterer(events)

    aggregation = clusterer.aggregate_topics()

    print(f"✅ Total topics: {aggregation['total_topics']}")
    print("\nTopic hierarchy:")
    for topic_name, topic_info in list(aggregation['hierarchy'].items())[:5]:
        if topic_info.get('count', 0) > 0:
            print(f"  - {topic_name}: {topic_info['count']} occurrences")
    print()


def test_session_clustering():
    """Test session clustering."""
    print("=" * 60)
    print("Testing Session Clustering")
    print("=" * 60)

    events = create_mock_events_for_clustering()
    clusterer = PatternClusterer(events)

    session_clusters = clusterer.cluster_sessions()

    print(f"✅ Found {len(session_clusters)} session types")
    for cluster in session_clusters:
        print(f"\n  Type: {cluster['type']}")
        print(f"  Count: {cluster['count']} sessions")
        if cluster['sessions']:
            print(f"  Example: {cluster['sessions'][0]['sample'][:60]}")
    print()


def test_problem_pattern_recognition():
    """Test problem pattern recognition."""
    print("=" * 60)
    print("Testing Problem Pattern Recognition")
    print("=" * 60)

    events = create_mock_events_for_clustering()
    clusterer = PatternClusterer(events)

    patterns = clusterer.recognize_problem_patterns()

    print(f"✅ Found {len(patterns)} recurring problem patterns")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. Pattern: {pattern['pattern']}")
        print(f"   Occurrences: {pattern['count']}")
        print(f"   Example: {pattern['occurrences'][0][:60]}")
    print()


def test_aggregation_report():
    """Test full aggregation report."""
    print("=" * 60)
    print("Testing Aggregation Report Generation")
    print("=" * 60)

    events = create_mock_events_for_clustering()
    clusterer = PatternClusterer(events)

    report = clusterer.generate_aggregation_report()

    print(f"✅ Summary: {report['summary'][:100]}...")
    print(f"✅ Query clusters: {len(report['query_clusters'])}")
    print(f"✅ Topic aggregation: {report['topic_aggregation']['total_topics']} topics")
    print(f"✅ Session clusters: {len(report['session_clusters'])}")
    print(f"✅ Problem patterns: {len(report['problem_patterns'])}")
    print()


def test_report_formatting():
    """Test report formatting."""
    print("=" * 60)
    print("Testing Report Formatting")
    print("=" * 60)

    events = create_mock_events_for_clustering()
    clusterer = PatternClusterer(events)

    report = clusterer.generate_aggregation_report()
    formatted = format_aggregation_report(report)

    print("✅ Formatted report generated")
    print(f"✅ Report length: {len(formatted)} characters")
    print("\nReport preview:")
    print(formatted[:600])
    print("...")
    print()


def test_empty_events():
    """Test with empty events."""
    print("=" * 60)
    print("Testing with Empty Events")
    print("=" * 60)

    clusterer = PatternClusterer([])

    clusters = clusterer.cluster_queries()
    aggregation = clusterer.aggregate_topics()
    sessions = clusterer.cluster_sessions()
    patterns = clusterer.recognize_problem_patterns()

    status = "✅" if len(clusters) == 0 else "❌"
    print(f"{status} Empty query clusters: {len(clusters)}")

    status = "✅" if aggregation['total_topics'] == 0 else "❌"
    print(f"{status} Empty topic aggregation: {aggregation['total_topics']}")

    status = "✅" if len(sessions) == 0 else "❌"
    print(f"{status} Empty session clusters: {len(sessions)}")

    status = "✅" if len(patterns) == 0 else "❌"
    print(f"{status} Empty problem patterns: {len(patterns)}")
    print()


if __name__ == "__main__":
    print("\n🧪 CodeMem Phase 4.5 Test Suite\n")

    test_query_clustering()
    test_topic_aggregation()
    test_session_clustering()
    test_problem_pattern_recognition()
    test_aggregation_report()
    test_report_formatting()
    test_empty_events()

    print("=" * 60)
    print("✅ All Phase 4.5 tests completed!")
    print("=" * 60)
