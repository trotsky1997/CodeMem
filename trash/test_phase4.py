#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 4: Pattern Discovery.

Tests:
- Frequent topics analysis
- Activity time patterns
- Knowledge evolution tracking
- Unresolved questions detection
- Insights report generation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from pattern_analyzer import PatternAnalyzer, format_insights_report


def create_mock_events():
    """Create mock events for testing."""
    now = datetime.now()
    events = []

    # Create events over 30 days
    for day in range(30):
        date = now - timedelta(days=day)

        # Morning events (9-11 AM)
        for hour in [9, 10, 11]:
            events.append({
                "timestamp": date.replace(hour=hour, minute=0).isoformat(),
                "role": "user",
                "text": "How to use Python async programming?",
                "session_id": f"session_{day}",
                "platform": "claude"
            })

            events.append({
                "timestamp": date.replace(hour=hour, minute=5).isoformat(),
                "role": "assistant",
                "text": "Python async programming uses asyncio for concurrent operations.",
                "session_id": f"session_{day}",
                "platform": "claude"
            })

    # Add topic-specific events
    topics = ["python", "async", "database", "performance", "test"]
    for i, topic in enumerate(topics):
        for j in range(10 - i * 2):  # Decreasing frequency
            events.append({
                "timestamp": (now - timedelta(days=j)).isoformat(),
                "role": "user",
                "text": f"Question about {topic} optimization",
                "session_id": f"session_topic_{i}_{j}",
                "platform": "claude"
            })

    # Add knowledge evolution events for "python"
    evolution_stages = [
        ("什么是 Python？", "基础概念"),
        ("如何使用 Python 异步？", "实践应用"),
        ("Python 性能优化技巧", "深入优化"),
        ("Python async 有问题", "问题解决"),
    ]

    for i, (text, stage) in enumerate(evolution_stages):
        events.append({
            "timestamp": (now - timedelta(days=25 - i * 5)).isoformat(),
            "role": "user",
            "text": text,
            "session_id": f"session_evolution_{i}",
            "platform": "claude"
        })

    # Add unresolved questions
    events.append({
        "timestamp": (now - timedelta(days=2)).isoformat(),
        "role": "user",
        "text": "How to optimize database queries? I'm not sure about the best approach.",
        "session_id": "session_unresolved_1",
        "platform": "claude"
    })

    events.append({
        "timestamp": (now - timedelta(days=2, minutes=10)).isoformat(),
        "role": "user",
        "text": "But what about caching strategies?",
        "session_id": "session_unresolved_1",
        "platform": "claude"
    })

    return events


def test_frequent_topics():
    """Test frequent topics analysis."""
    print("=" * 60)
    print("Testing Frequent Topics Analysis")
    print("=" * 60)

    events = create_mock_events()
    analyzer = PatternAnalyzer(events)

    topics = analyzer.analyze_frequent_topics(top_n=5)

    print(f"✅ Found {len(topics)} frequent topics")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic['topic']}: {topic['count']} times ({topic['percentage']:.1f}%)")
    print()


def test_activity_time():
    """Test activity time analysis."""
    print("=" * 60)
    print("Testing Activity Time Analysis")
    print("=" * 60)

    events = create_mock_events()
    analyzer = PatternAnalyzer(events)

    activity = analyzer.analyze_activity_time()

    print(f"✅ Peak hours: {activity['peak_hours']}")
    print(f"✅ Peak weekdays: {activity['peak_weekdays']}")
    print(f"✅ Total days: {activity['total_days']}")
    print(f"✅ Max streak: {activity['max_streak']}")
    print(f"✅ Avg events/day: {activity['avg_events_per_day']:.1f}")
    print()


def test_knowledge_evolution():
    """Test knowledge evolution tracking."""
    print("=" * 60)
    print("Testing Knowledge Evolution Tracking")
    print("=" * 60)

    events = create_mock_events()
    analyzer = PatternAnalyzer(events)

    # Test Python evolution
    evolution = analyzer.analyze_knowledge_evolution("python")

    print(f"✅ Topic: {evolution['topic']}")
    print(f"✅ Total discussions: {evolution['total_discussions']}")
    print(f"✅ Progression: {evolution['progression']}")
    print(f"✅ Stages found: {len(evolution['stages'])}")

    if evolution['stages']:
        print("\nStages:")
        for stage in evolution['stages'][:3]:
            print(f"  - {stage['stage']}: {stage['text_preview'][:50]}...")
    print()


def test_unresolved_questions():
    """Test unresolved questions detection."""
    print("=" * 60)
    print("Testing Unresolved Questions Detection")
    print("=" * 60)

    events = create_mock_events()
    analyzer = PatternAnalyzer(events)

    unresolved = analyzer.find_unresolved_questions()

    print(f"✅ Found {len(unresolved)} potentially unresolved questions")
    for i, q in enumerate(unresolved[:3], 1):
        print(f"{i}. {q['question'][:80]}...")
        print(f"   Reason: {q['reason']}")
    print()


def test_insights_report():
    """Test insights report generation."""
    print("=" * 60)
    print("Testing Insights Report Generation")
    print("=" * 60)

    events = create_mock_events()
    analyzer = PatternAnalyzer(events)

    report = analyzer.generate_insights_report(days=30)

    print(f"✅ Period: {report['period']}")
    print(f"✅ Summary: {report['summary'][:100]}...")
    print(f"✅ Total events: {report['total_events']}")
    print(f"✅ Total sessions: {report['total_sessions']}")
    print(f"✅ Frequent topics: {len(report['frequent_topics'])}")
    print(f"✅ Unresolved questions: {len(report['unresolved_questions'])}")
    print()


def test_report_formatting():
    """Test report formatting."""
    print("=" * 60)
    print("Testing Report Formatting")
    print("=" * 60)

    events = create_mock_events()
    analyzer = PatternAnalyzer(events)

    report = analyzer.generate_insights_report(days=30)
    formatted = format_insights_report(report)

    print("✅ Formatted report generated")
    print(f"✅ Report length: {len(formatted)} characters")
    print("\nReport preview:")
    print(formatted[:500])
    print("...")
    print()


def test_empty_events():
    """Test with empty events."""
    print("=" * 60)
    print("Testing with Empty Events")
    print("=" * 60)

    analyzer = PatternAnalyzer([])

    topics = analyzer.analyze_frequent_topics()
    activity = analyzer.analyze_activity_time()
    unresolved = analyzer.find_unresolved_questions()
    report = analyzer.generate_insights_report(days=30)

    status = "✅" if len(topics) == 0 else "❌"
    print(f"{status} Empty topics: {len(topics)}")

    status = "✅" if activity['total_days'] == 0 else "❌"
    print(f"{status} Empty activity: {activity['total_days']} days")

    status = "✅" if len(unresolved) == 0 else "❌"
    print(f"{status} Empty unresolved: {len(unresolved)}")

    status = "✅" if report['total_events'] == 0 else "❌"
    print(f"{status} Empty report: {report['total_events']} events")
    print()


if __name__ == "__main__":
    print("\n🧪 CodeMem Phase 4 Test Suite\n")

    test_frequent_topics()
    test_activity_time()
    test_knowledge_evolution()
    test_unresolved_questions()
    test_insights_report()
    test_report_formatting()
    test_empty_events()

    print("=" * 60)
    print("✅ All Phase 4 tests completed!")
    print("=" * 60)
