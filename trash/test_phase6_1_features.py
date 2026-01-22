#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 6.1: Feature Extraction and CTR Model.

Tests:
- Feature extraction (6 classes)
- CTR model training
- CTR-based ranking
- Integration test
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from feature_extractor import (
    QueryFeatures, ConversationFeatures, MatchFeatures,
    UserHistoryFeatures, ContextFeatures, PositionFeatures,
    FeatureExtractor
)
from ctr_model import LogisticRegressionCTR, CTRRanker, create_ctr_model_from_distance_data
from test_phase6_1 import create_mock_events_for_phase6


def test_query_features():
    """Test query feature extraction."""
    print("=" * 60)
    print("Testing Query Features")
    print("=" * 60)

    extractor = QueryFeatures()

    # Test different query types
    queries = [
        "How to use Python asyncio?",
        "为什么数据库查询很慢？",
        "What is REST API?",
        "```python\nimport asyncio\n```",
    ]

    for query in queries:
        features = extractor.extract(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"  Length: {features['query_length']:.0f}")
        print(f"  Word count: {features['query_word_count']:.0f}")
        print(f"  Has code: {features['has_code']:.0f}")
        print(f"  Tech keywords: {features['tech_keyword_count']:.0f}")
        print(f"  Is how question: {features['is_how_question']:.0f}")

    print("\n✅ Query features extracted successfully\n")


def test_conversation_features():
    """Test conversation feature extraction."""
    print("=" * 60)
    print("Testing Conversation Features")
    print("=" * 60)

    extractor = ConversationFeatures()

    # Create test conversation
    conversation = {
        'session_id': 'test_session',
        'timestamp': datetime.now().isoformat(),
        'rank': 0.75,
        'messages': [
            {'role': 'user', 'text': 'How to use asyncio?'},
            {'role': 'assistant', 'text': 'Here is how:\n```python\nimport asyncio\n```'},
            {'role': 'user', 'text': 'Thanks, it works!'},
        ]
    }

    features = extractor.extract(conversation)

    print("\n✅ Conversation features:")
    print(f"  Message count: {features['message_count']:.0f}")
    print(f"  Has code: {features['has_code_block']:.0f}")
    print(f"  Has solution: {features['has_solution']:.0f}")
    print(f"  Has confirmation: {features['has_confirmation']:.0f}")
    print(f"  ConversationRank: {features['conversation_rank']:.3f}")
    print(f"  Turn count: {features['turn_count']:.0f}")
    print()


def test_match_features():
    """Test match feature extraction."""
    print("=" * 60)
    print("Testing Match Features")
    print("=" * 60)

    extractor = MatchFeatures()

    query = "Python asyncio tutorial"
    conversation = {
        'messages': [
            {'role': 'user', 'text': 'How to use Python asyncio?'},
            {'role': 'assistant', 'text': 'Asyncio is a Python library for async programming...'},
        ],
        'bm25_score': 0.85
    }

    features = extractor.extract(query, conversation)

    print("\n✅ Match features:")
    print(f"  BM25 score: {features['bm25_score']:.3f}")
    print(f"  Keyword overlap: {features['keyword_overlap']:.3f}")
    print(f"  Tech keyword overlap: {features['tech_keyword_overlap']:.3f}")
    print(f"  Query in title: {features['query_in_title']:.0f}")
    print()


def test_position_features():
    """Test position feature extraction."""
    print("=" * 60)
    print("Testing Position Features")
    print("=" * 60)

    extractor = PositionFeatures()

    for position in [1, 2, 3, 5, 10]:
        features = extractor.extract(position)
        print(f"\nPosition {position}:")
        print(f"  Position bias: {features['position_bias']:.3f}")
        print(f"  Is top 1: {features['is_top_1']:.0f}")
        print(f"  Is top 3: {features['is_top_3']:.0f}")

    print("\n✅ Position features extracted successfully\n")


def test_feature_extractor():
    """Test full feature extraction."""
    print("=" * 60)
    print("Testing Full Feature Extraction")
    print("=" * 60)

    extractor = FeatureExtractor()

    query = "How to use Python asyncio?"
    conversation = {
        'session_id': 'test_session',
        'timestamp': datetime.now().isoformat(),
        'rank': 0.75,
        'messages': [
            {'role': 'user', 'text': 'How to use asyncio?'},
            {'role': 'assistant', 'text': 'Here is how:\n```python\nimport asyncio\n```'},
        ],
        'bm25_score': 0.85
    }
    user_history = {'frequent_topics': ['python', 'async']}
    context = {'current_session': 'test_session', 'is_follow_up': True}
    position = 1

    features = extractor.extract_all(query, conversation, user_history, context, position)

    print(f"\n✅ Extracted {len(features)} features")
    print("\nTop 10 features:")
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, value in sorted_features[:10]:
        print(f"  {name:30s}: {value:.3f}")

    print()


def test_ctr_model_initialization():
    """Test CTR model initialization."""
    print("=" * 60)
    print("Testing CTR Model Initialization")
    print("=" * 60)

    model = LogisticRegressionCTR()

    print(f"\n✅ Model initialized")
    print(f"  Feature count: {len(model.feature_names)}")
    print(f"  Weights initialized: {model.weights is not None}")

    # Get feature importance
    importance = model.get_feature_importance()

    print("\nTop 10 most important features:")
    for name, weight in importance[:10]:
        print(f"  {name:30s}: {weight:.3f}")

    print()


def test_ctr_prediction():
    """Test CTR prediction."""
    print("=" * 60)
    print("Testing CTR Prediction")
    print("=" * 60)

    model = LogisticRegressionCTR()

    # Create test data
    query = "How to use Python asyncio?"
    conversations = [
        {
            'session_id': 'high_quality',
            'timestamp': datetime.now().isoformat(),
            'rank': 0.9,
            'messages': [
                {'role': 'user', 'text': 'How to use asyncio?'},
                {'role': 'assistant', 'text': 'Here is a complete guide:\n```python\nimport asyncio\n```'},
                {'role': 'user', 'text': 'Thanks, it works!'},
            ],
            'bm25_score': 0.95
        },
        {
            'session_id': 'low_quality',
            'timestamp': (datetime.now() - timedelta(days=100)).isoformat(),
            'rank': 0.2,
            'messages': [
                {'role': 'user', 'text': 'Hello'},
                {'role': 'assistant', 'text': 'Hi'},
            ],
            'bm25_score': 0.1
        }
    ]

    user_history = {}
    context = {}

    print("\n✅ Predicted CTRs:\n")
    for i, conv in enumerate(conversations, 1):
        features = model.extract_features(query, conv, user_history, context, i)
        ctr = model.predict(features)
        print(f"  {i}. {conv['session_id']:20s}: {ctr:.4f}")

    print("\n📊 Expected: high_quality > low_quality")
    print()


def test_ctr_training():
    """Test CTR model training."""
    print("=" * 60)
    print("Testing CTR Model Training")
    print("=" * 60)

    # Create training data
    training_data = []

    for i in range(10):
        training_data.append({
            'query': f'Query {i}',
            'conversation': {
                'session_id': f'session_{i}',
                'messages': [],
                'rank': 0.5,
                'bm25_score': 0.5
            },
            'user_history': {},
            'context': {},
            'position': 1,
            'label': 1.0 if i < 5 else 0.0,  # First 5 are positive
            'weight': 1.0
        })

    model = LogisticRegressionCTR()
    print(f"\n🔄 Training with {len(training_data)} samples...\n")
    model.train(training_data, epochs=50, learning_rate=0.01)

    print("\n✅ Training completed\n")


def test_ctr_ranker():
    """Test CTR-based ranking."""
    print("=" * 60)
    print("Testing CTR Ranker")
    print("=" * 60)

    model = LogisticRegressionCTR()
    ranker = CTRRanker(model)

    query = "Python asyncio"
    candidates = [
        {
            'session_id': 'session_1',
            'timestamp': datetime.now().isoformat(),
            'rank': 0.3,
            'messages': [{'role': 'user', 'text': 'Hello'}],
            'bm25_score': 0.9  # High BM25 but low quality
        },
        {
            'session_id': 'session_2',
            'timestamp': datetime.now().isoformat(),
            'rank': 0.8,
            'messages': [
                {'role': 'user', 'text': 'How to use asyncio?'},
                {'role': 'assistant', 'text': '```python\nimport asyncio\n```'},
                {'role': 'user', 'text': 'Thanks!'},
            ],
            'bm25_score': 0.7  # Lower BM25 but high quality
        }
    ]

    user_history = {}
    context = {}

    ranked = ranker.rank(query, candidates, user_history, context)

    print("\n✅ Ranked results:\n")
    for i, result in enumerate(ranked, 1):
        print(f"  {i}. {result['session_id']:20s}")
        print(f"     Predicted CTR: {result['predicted_ctr']:.4f}")
        print(f"     Original position: {result['original_position']}")
        print(f"     BM25: {result['bm25_score']:.2f}, Rank: {result['rank']:.2f}")
        print()

    print("📊 Expected: session_2 (high quality) should rank higher\n")


def test_integration_with_distance_data():
    """Test integration with distance-based training data."""
    print("=" * 60)
    print("Testing Integration with Distance Data")
    print("=" * 60)

    events = create_mock_events_for_phase6()

    print("\n🔄 Creating CTR model from distance data...\n")
    model = create_ctr_model_from_distance_data(events, method='session_based', epochs=50)

    print("\n✅ Model created and trained")
    print(f"  Feature count: {len(model.feature_names)}")

    # Test prediction
    query = "Python asyncio"
    conversation = {
        'session_id': 'test',
        'timestamp': datetime.now().isoformat(),
        'rank': 0.75,
        'messages': [{'role': 'user', 'text': 'How to use asyncio?'}],
        'bm25_score': 0.8
    }

    features = model.extract_features(query, conversation, {}, {}, 1)
    ctr = model.predict(features)

    print(f"\n✅ Test prediction: {ctr:.4f}")
    print()


if __name__ == "__main__":
    print("\n🧪 CodeMem Phase 6.1 Feature & CTR Test Suite\n")

    test_query_features()
    test_conversation_features()
    test_match_features()
    test_position_features()
    test_feature_extractor()
    test_ctr_model_initialization()
    test_ctr_prediction()
    test_ctr_training()
    test_ctr_ranker()
    test_integration_with_distance_data()

    print("=" * 60)
    print("✅ All Feature & CTR tests completed!")
    print("=" * 60)
