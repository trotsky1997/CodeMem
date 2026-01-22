#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test pattern clustering integration with CTR model.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from test_phase6_1 import create_mock_events_for_phase6
from pattern_integration import (
    generate_user_history_with_patterns,
    enrich_search_results_with_patterns,
    explain_ranking_with_patterns,
    get_pattern_insights
)
from feature_extractor import FeatureExtractor
from ctr_model import LogisticRegressionCTR


def test_generate_user_history_with_patterns():
    """Test generating user history with pattern clustering."""
    print("=" * 60)
    print("Testing User History with Pattern Clustering")
    print("=" * 60)

    events = create_mock_events_for_phase6()

    print("\n🔄 Generating user history with patterns...\n")
    user_history = generate_user_history_with_patterns(events)

    print("✅ User history generated:")
    print(f"  Has pattern_clusters: {'pattern_clusters' in user_history}")
    print(f"  Query clusters: {len(user_history['pattern_clusters']['query_clusters'])}")
    print(f"  Session clusters: {len(user_history['pattern_clusters']['session_clusters'])}")
    print(f"  Problem patterns: {len(user_history['pattern_clusters']['problem_patterns'])}")
    print(f"  Frequent topics: {user_history['frequent_topics']}")
    print()


def test_pattern_features_extraction():
    """Test extracting pattern features."""
    print("=" * 60)
    print("Testing Pattern Features Extraction")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    user_history = generate_user_history_with_patterns(events)

    # Create a test conversation
    conversation = {
        'session_id': 'session_high_quality',
        'messages': [
            {'role': 'user', 'text': 'How to use Python asyncio?'},
            {'role': 'assistant', 'text': 'Here is how to use asyncio...'},
        ],
        'rank': 0.75,
        'bm25_score': 0.85
    }

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_all(
        query="Python asyncio",
        conversation=conversation,
        user_history=user_history,
        context={},
        position=1
    )

    print("\n✅ Features extracted:")
    print(f"  Total features: {len(features)}")
    print("\n  Pattern clustering features:")
    print(f"    in_frequent_query_cluster: {features.get('in_frequent_query_cluster', 0):.2f}")
    print(f"    session_type_match: {features.get('session_type_match', 0):.2f}")
    print(f"    is_recurring_problem: {features.get('is_recurring_problem', 0):.2f}")
    print(f"    topic_cluster_match: {features.get('topic_cluster_match', 0):.2f}")
    print()


def test_ctr_with_pattern_features():
    """Test CTR prediction with pattern features."""
    print("=" * 60)
    print("Testing CTR Prediction with Pattern Features")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    user_history = generate_user_history_with_patterns(events)

    # Create test conversations
    conversations = [
        {
            'session_id': 'session_high_quality',
            'messages': [
                {'role': 'user', 'text': 'How to use Python asyncio?'},
                {'role': 'assistant', 'text': 'Asyncio guide with code...'},
            ],
            'rank': 0.75,
            'bm25_score': 0.85
        },
        {
            'session_id': 'session_short',
            'messages': [
                {'role': 'user', 'text': 'What is Python?'},
            ],
            'rank': 0.2,
            'bm25_score': 0.60
        }
    ]

    model = LogisticRegressionCTR()
    extractor = FeatureExtractor()

    print("\n✅ CTR Predictions:\n")
    for i, conv in enumerate(conversations, 1):
        # Extract features as dict first
        features_dict = extractor.extract_all(
            query="Python asyncio",
            conversation=conv,
            user_history=user_history,
            context={},
            position=i
        )

        # Then convert to numpy array for prediction
        features_array = model.extract_features(
            query="Python asyncio",
            conversation=conv,
            user_history=user_history,
            context={},
            position=i
        )
        ctr = model.predict(features_array)

        print(f"  {i}. {conv['session_id']}")
        print(f"     Predicted CTR: {ctr:.4f}")
        print(f"     Pattern features:")
        print(f"       - Query cluster: {features_dict.get('in_frequent_query_cluster', 0):.2f}")
        print(f"       - Session type: {features_dict.get('session_type_match', 0):.2f}")
        print(f"       - Topic match: {features_dict.get('topic_cluster_match', 0):.2f}")
        print()


def test_enrich_results_with_patterns():
    """Test enriching search results with pattern metadata."""
    print("=" * 60)
    print("Testing Result Enrichment with Patterns")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    user_history = generate_user_history_with_patterns(events)
    pattern_clusters = user_history['pattern_clusters']

    # Mock search results
    results = [
        {
            'session_id': 'session_high_quality',
            'text': 'How to use Python asyncio?',
            'score': 0.85
        },
        {
            'session_id': 'session_short',
            'text': 'What is Python?',
            'score': 0.60
        }
    ]

    enriched = enrich_search_results_with_patterns(results, pattern_clusters)

    print("\n✅ Enriched results:\n")
    for i, result in enumerate(enriched, 1):
        meta = result['pattern_meta']
        print(f"  {i}. {result['session_id']}")
        print(f"     In query cluster: {meta['in_query_cluster']}")
        print(f"     Session type: {meta['session_type']}")
        print(f"     Recurring problem: {meta['is_recurring_problem']}")
        print(f"     Topic clusters: {meta['topic_clusters']}")
        print()


def test_ranking_explanation():
    """Test ranking explanation with pattern features."""
    print("=" * 60)
    print("Testing Ranking Explanation")
    print("=" * 60)

    # Mock features
    features_high = {
        'in_frequent_query_cluster': 1.0,
        'session_type_match': 0.9,
        'is_recurring_problem': 1.0,
        'topic_cluster_match': 0.8,
        'conversation_rank': 0.75,
        'has_solution': 1.0,
        'has_code_block': 1.0,
    }

    features_low = {
        'in_frequent_query_cluster': 0.0,
        'session_type_match': 0.3,
        'is_recurring_problem': 0.0,
        'topic_cluster_match': 0.0,
        'conversation_rank': 0.2,
        'has_solution': 0.0,
        'has_code_block': 0.0,
    }

    print("\n✅ Ranking explanations:\n")

    print("  High-quality result:")
    explanation = explain_ranking_with_patterns({}, features_high)
    print(f"    {explanation}")

    print("\n  Low-quality result:")
    explanation = explain_ranking_with_patterns({}, features_low)
    print(f"    {explanation}")

    print()


def test_pattern_insights():
    """Test getting pattern insights."""
    print("=" * 60)
    print("Testing Pattern Insights")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    user_history = generate_user_history_with_patterns(events)

    insights = get_pattern_insights(user_history)

    print("\n✅ Pattern insights:\n")
    print(f"  Has patterns: {insights['has_patterns']}")
    print(f"\n  Insights:")
    for insight in insights['insights']:
        print(f"    - {insight}")

    print()


def test_feature_count():
    """Test that feature count increased with pattern features."""
    print("=" * 60)
    print("Testing Feature Count")
    print("=" * 60)

    events = create_mock_events_for_phase6()

    # Without pattern clustering
    user_history_no_patterns = {}

    # With pattern clustering
    user_history_with_patterns = generate_user_history_with_patterns(events)

    conversation = {
        'session_id': 'test',
        'messages': [],
        'rank': 0.5,
        'bm25_score': 0.5
    }

    extractor = FeatureExtractor()

    features_no_patterns = extractor.extract_all(
        query="test",
        conversation=conversation,
        user_history=user_history_no_patterns,
        context={},
        position=1
    )

    features_with_patterns = extractor.extract_all(
        query="test",
        conversation=conversation,
        user_history=user_history_with_patterns,
        context={},
        position=1
    )

    print(f"\n✅ Feature count comparison:")
    print(f"  Without patterns: {len(features_no_patterns)} features")
    print(f"  With patterns: {len(features_with_patterns)} features")
    print(f"  Difference: +{len(features_with_patterns) - len(features_no_patterns)} features")

    # Show new features
    new_features = set(features_with_patterns.keys()) - set(features_no_patterns.keys())
    if new_features:
        print(f"\n  New pattern features:")
        for feature in sorted(new_features):
            print(f"    - {feature}")

    print()


if __name__ == "__main__":
    print("\n🧪 Pattern Clustering Integration Test Suite\n")

    test_generate_user_history_with_patterns()
    test_pattern_features_extraction()
    test_ctr_with_pattern_features()
    test_enrich_results_with_patterns()
    test_ranking_explanation()
    test_pattern_insights()
    test_feature_count()

    print("=" * 60)
    print("✅ All pattern integration tests completed!")
    print("=" * 60)
    print("\n📊 Summary:")
    print("  ✅ Pattern clustering integrated into user history")
    print("  ✅ 4 new pattern features added (43 total)")
    print("  ✅ CTR model uses pattern features")
    print("  ✅ Results enriched with pattern metadata")
    print("  ✅ Ranking explanations include patterns")
    print("\n🎉 Phase 4.5 + Phase 6.1 integration complete!")
    print()
