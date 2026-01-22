#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 6.1: ConversationRank and Distance-based Training.

Tests:
- ConversationRank calculation
- Distance-based training data generation
- Integration with existing system
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from conversation_rank import ConversationRank, calculate_conversation_ranks, get_top_conversations
from distance_trainer import DistanceBasedTrainer, generate_distance_training_data


def create_mock_events_for_phase6():
    """Create mock events for Phase 6 testing."""
    now = datetime.now()
    events = []

    # Session 1: High-quality technical discussion (should rank high)
    session1_messages = [
        ("How to use Python asyncio?", "user"),
        ("Asyncio is a library for asynchronous programming in Python. Here's an example:\n```python\nimport asyncio\n\nasync def main():\n    print('Hello')\n    await asyncio.sleep(1)\n    print('World')\n\nasyncio.run(main())\n```", "assistant"),
        ("Thanks! How do I use asyncio.gather?", "user"),
        ("asyncio.gather() runs multiple coroutines concurrently:\n```python\nimport asyncio\n\nasync def task1():\n    await asyncio.sleep(1)\n    return 'Task 1'\n\nasync def task2():\n    await asyncio.sleep(2)\n    return 'Task 2'\n\nasync def main():\n    results = await asyncio.gather(task1(), task2())\n    print(results)\n\nasyncio.run(main())\n```", "assistant"),
        ("Got it, thanks!", "user"),
    ]

    for i, (text, role) in enumerate(session1_messages):
        events.append({
            "timestamp": (now - timedelta(days=5, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_high_quality",
            "platform": "claude"
        })

    # Session 2: Short question (should rank lower)
    session2_messages = [
        ("What is Python?", "user"),
        ("Python is a programming language.", "assistant"),
    ]

    for i, (text, role) in enumerate(session2_messages):
        events.append({
            "timestamp": (now - timedelta(days=10, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_short",
            "platform": "claude"
        })

    # Session 3: References session 1 (should boost session 1's rank)
    session3_messages = [
        ("I remember we discussed asyncio before. Can you explain asyncio.create_task?", "user"),
        ("Sure! asyncio.create_task() schedules a coroutine to run...", "assistant"),
    ]

    for i, (text, role) in enumerate(session3_messages):
        events.append({
            "timestamp": (now - timedelta(days=3, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_reference",
            "platform": "claude"
        })

    # Session 4: Recent but low quality (should rank medium)
    session4_messages = [
        ("Hello", "user"),
        ("Hi! How can I help?", "assistant"),
        ("Just testing", "user"),
    ]

    for i, (text, role) in enumerate(session4_messages):
        events.append({
            "timestamp": (now - timedelta(days=1, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_recent_low_quality",
            "platform": "claude"
        })

    # Session 5: Old but high quality (should rank medium-high)
    session5_messages = [
        ("How to optimize database queries?", "user"),
        ("Here are some database optimization techniques:\n1. Use indexes\n2. Avoid N+1 queries\n3. Use connection pooling\n```sql\nCREATE INDEX idx_user_email ON users(email);\n```", "assistant"),
        ("Thanks, that solved my performance issue!", "user"),
    ]

    for i, (text, role) in enumerate(session5_messages):
        events.append({
            "timestamp": (now - timedelta(days=30, minutes=i*5)).isoformat(),
            "role": role,
            "text": text,
            "session_id": "session_old_quality",
            "platform": "claude"
        })

    return events


def test_conversation_rank():
    """Test ConversationRank calculation."""
    print("=" * 60)
    print("Testing ConversationRank Calculation")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    ranker = ConversationRank(events)

    # Calculate ranks for all sessions
    ranks = ranker.calculate_all_ranks()

    print(f"\n✅ Calculated ranks for {len(ranks)} sessions\n")

    # Sort by rank
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    for session_id, rank in sorted_ranks:
        print(f"  {session_id:30s}: {rank:.3f}")

    # Verify expected ranking
    print("\n📊 Expected ranking:")
    print("  1. session_high_quality (has code, solution, references)")
    print("  2. session_old_quality (has code, solution)")
    print("  3. session_reference (references other session)")
    print("  4. session_recent_low_quality (recent but low quality)")
    print("  5. session_short (very short)")

    # Check if high quality session ranks highest
    top_session = sorted_ranks[0][0]
    if top_session == "session_high_quality":
        print("\n✅ Ranking is correct!")
    else:
        print(f"\n⚠️ Expected session_high_quality to rank highest, got {top_session}")

    print()


def test_top_conversations():
    """Test get_top_conversations function."""
    print("=" * 60)
    print("Testing Top Conversations")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    top_convs = get_top_conversations(events, top_n=3)

    print(f"\n✅ Top 3 conversations:\n")
    for i, conv in enumerate(top_convs, 1):
        print(f"  {i}. {conv['session_id']:30s}: {conv['rank']:.3f}")

    print()


def test_session_based_training():
    """Test session-based training data generation."""
    print("=" * 60)
    print("Testing Session-Based Training Data")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    trainer = DistanceBasedTrainer(events)

    training_data = trainer.generate_session_based_data()

    print(f"\n✅ Generated {len(training_data)} training samples\n")

    # Count positive and negative samples
    positive = sum(1 for d in training_data if d['label'] == 1.0)
    negative = sum(1 for d in training_data if d['label'] == 0.0)

    print(f"  Positive samples (same session): {positive}")
    print(f"  Negative samples (different session): {negative}")

    # Show a few examples
    print("\n📝 Sample training data:\n")
    for i, sample in enumerate(training_data[:3], 1):
        print(f"  {i}. Query: {sample['query'][:50]}...")
        print(f"     Candidate: {sample['candidate_text'][:50]}...")
        print(f"     Label: {sample['label']}")
        print(f"     Method: {sample['method']}")
        print()


def test_time_decay_training():
    """Test time-decay training data generation."""
    print("=" * 60)
    print("Testing Time-Decay Training Data")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    trainer = DistanceBasedTrainer(events)

    training_data = trainer.generate_time_decay_data()

    print(f"\n✅ Generated {len(training_data)} training samples\n")

    # Show label distribution
    labels = [d['label'] for d in training_data]
    print(f"  Average label: {sum(labels) / len(labels):.3f}")
    print(f"  Max label: {max(labels):.3f}")
    print(f"  Min label: {min(labels):.3f}")

    # Show a few examples with time differences
    print("\n📝 Sample training data with time decay:\n")
    for i, sample in enumerate(training_data[:3], 1):
        print(f"  {i}. Query: {sample['query'][:50]}...")
        print(f"     Time diff: {sample.get('time_diff', 0):.1f} days")
        print(f"     Label: {sample['label']:.3f}")
        print()


def test_sliding_window_training():
    """Test sliding window training data generation."""
    print("=" * 60)
    print("Testing Sliding Window Training Data")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    trainer = DistanceBasedTrainer(events)

    training_data = trainer.generate_sliding_window_data(window_size=5)

    print(f"\n✅ Generated {len(training_data)} training samples\n")

    # Count within/outside window
    within_window = sum(1 for d in training_data if d.get('distance', 999) <= 5)
    outside_window = sum(1 for d in training_data if d.get('distance', 999) > 5)

    print(f"  Within window (distance ≤ 5): {within_window}")
    print(f"  Outside window (distance > 5): {outside_window}")

    # Show a few examples
    print("\n📝 Sample training data:\n")
    for i, sample in enumerate(training_data[:3], 1):
        print(f"  {i}. Distance: {sample.get('distance', 0)}")
        print(f"     Label: {sample['label']:.3f}")
        print()


def test_hybrid_training():
    """Test hybrid training data generation."""
    print("=" * 60)
    print("Testing Hybrid Training Data")
    print("=" * 60)

    events = create_mock_events_for_phase6()
    trainer = DistanceBasedTrainer(events)

    training_data = trainer.generate_hybrid_data()

    print(f"\n✅ Generated {len(training_data)} training samples\n")

    # Show label distribution
    labels = [d['label'] for d in training_data]
    print(f"  Average label: {sum(labels) / len(labels):.3f}")
    print(f"  Max label: {max(labels):.3f}")
    print(f"  Min label: {min(labels):.3f}")

    # Show high/medium/low relevance counts
    high_rel = sum(1 for l in labels if l > 0.7)
    medium_rel = sum(1 for l in labels if 0.3 <= l <= 0.7)
    low_rel = sum(1 for l in labels if l < 0.3)

    print(f"\n  High relevance (>0.7): {high_rel}")
    print(f"  Medium relevance (0.3-0.7): {medium_rel}")
    print(f"  Low relevance (<0.3): {low_rel}")

    print()


def test_integration():
    """Test integration of ConversationRank and distance training."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)

    events = create_mock_events_for_phase6()

    # Calculate ConversationRanks
    ranks = calculate_conversation_ranks(events)

    # Generate training data
    training_data = generate_distance_training_data(events, method='hybrid')

    print(f"\n✅ Calculated {len(ranks)} ConversationRanks")
    print(f"✅ Generated {len(training_data)} training samples")

    # Verify we can attach ranks to training data
    for sample in training_data[:3]:
        session_id = sample['candidate_session_id']
        rank = ranks.get(session_id, 0.0)
        sample['conversation_rank'] = rank

    print(f"\n✅ Successfully attached ConversationRank to training samples")

    # Show example
    print("\n📝 Sample with ConversationRank:\n")
    sample = training_data[0]
    print(f"  Query: {sample['query'][:50]}...")
    print(f"  Session: {sample['candidate_session_id']}")
    print(f"  ConversationRank: {sample.get('conversation_rank', 0):.3f}")
    print(f"  Distance label: {sample['label']:.3f}")

    print()


if __name__ == "__main__":
    print("\n🧪 CodeMem Phase 6.1 Test Suite\n")

    test_conversation_rank()
    test_top_conversations()
    test_session_based_training()
    test_time_decay_training()
    test_sliding_window_training()
    test_hybrid_training()
    test_integration()

    print("=" * 60)
    print("✅ All Phase 6.1 tests completed!")
    print("=" * 60)
