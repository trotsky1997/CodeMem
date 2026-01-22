#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance-based training data generation for Phase 6.1.

Generates pseudo-labeled training data from conversation history
based on message distance (session, time, sequence).
"""

import math
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict


class DistanceBasedTrainer:
    """
    Generate training data based on conversation distance.

    Core idea: Messages that are close in conversation history
    are more likely to be relevant to each other.
    """

    def __init__(self, events: List[Dict[str, Any]]):
        """
        Initialize trainer.

        Args:
            events: List of event dictionaries
        """
        self.events = events
        self.user_messages = [e for e in events if e.get('role') == 'user']
        self.sessions = self._group_by_session()

    def _group_by_session(self) -> Dict[str, List[Dict]]:
        """Group events by session."""
        sessions = defaultdict(list)
        for event in self.events:
            session_id = event.get('session_id')
            if session_id:
                sessions[session_id].append(event)
        return dict(sessions)

    def generate_session_based_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data: session-based (simplest).

        Strategy:
        - Same session = positive sample (label=1.0)
        - Different session = negative sample (label=0.0)

        Returns:
            List of training samples
        """
        training_data = []
        session_list = list(self.sessions.values())

        for session in session_list:
            user_msgs = [m for m in session if m.get('role') == 'user']

            for query_msg in user_msgs:
                query_text = query_msg.get('text', '')
                if not query_text:
                    continue

                # Positive samples: same session
                for other_msg in session:
                    if other_msg.get('text') == query_text:
                        continue

                    training_data.append({
                        'query': query_text,
                        'candidate_session_id': other_msg.get('session_id'),
                        'candidate_text': other_msg.get('text', ''),
                        'label': 1.0,
                        'method': 'session_based',
                        'query_timestamp': query_msg.get('timestamp'),
                        'candidate_timestamp': other_msg.get('timestamp')
                    })

                # Negative samples: different sessions (sample)
                other_sessions = [s for s in session_list if s != session]
                if other_sessions:
                    negative_session = random.choice(other_sessions)
                    negative_msg = random.choice(negative_session)

                    training_data.append({
                        'query': query_text,
                        'candidate_session_id': negative_msg.get('session_id'),
                        'candidate_text': negative_msg.get('text', ''),
                        'label': 0.0,
                        'method': 'session_based',
                        'query_timestamp': query_msg.get('timestamp'),
                        'candidate_timestamp': negative_msg.get('timestamp')
                    })

        return training_data

    def generate_time_decay_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data: time-based decay.

        Strategy:
        - Label = exp(-time_distance / 7 days)
        - Same session gets 2x boost

        Returns:
            List of training samples
        """
        training_data = []

        for query_msg in self.user_messages:
            query_text = query_msg.get('text', '')
            if not query_text:
                continue

            query_time = self._parse_timestamp(query_msg.get('timestamp'))
            query_session = query_msg.get('session_id')

            # Sample candidates (not all, too many)
            candidates = random.sample(self.events, min(50, len(self.events)))

            for candidate_msg in candidates:
                if candidate_msg.get('text') == query_text:
                    continue

                candidate_time = self._parse_timestamp(candidate_msg.get('timestamp'))
                candidate_session = candidate_msg.get('session_id')

                # Calculate time distance (days)
                time_diff = abs((candidate_time - query_time).total_seconds() / 86400)

                # Time decay label (7-day half-life)
                label = math.exp(-time_diff / 7)

                # Session boost
                if query_session == candidate_session:
                    label = min(label * 2, 1.0)

                training_data.append({
                    'query': query_text,
                    'candidate_session_id': candidate_session,
                    'candidate_text': candidate_msg.get('text', ''),
                    'label': label,
                    'method': 'time_decay',
                    'time_diff': time_diff,
                    'query_timestamp': query_msg.get('timestamp'),
                    'candidate_timestamp': candidate_msg.get('timestamp')
                })

        return training_data

    def generate_sliding_window_data(self, window_size: int = 20) -> List[Dict[str, Any]]:
        """
        Generate training data: sliding window.

        Strategy:
        - Within window = positive (label decays with distance)
        - Outside window = negative (label=0)

        Args:
            window_size: Window size (number of messages)

        Returns:
            List of training samples
        """
        training_data = []

        for i, query_msg in enumerate(self.user_messages):
            query_text = query_msg.get('text', '')
            if not query_text:
                continue

            # Positive samples: within window
            for j in range(max(0, i - window_size), min(len(self.user_messages), i + window_size + 1)):
                if i == j:
                    continue

                candidate_msg = self.user_messages[j]
                distance = abs(j - i)

                # Distance-based label (decays every 5 messages)
                label = 1.0 / (1.0 + distance / 5)

                training_data.append({
                    'query': query_text,
                    'candidate_session_id': candidate_msg.get('session_id'),
                    'candidate_text': candidate_msg.get('text', ''),
                    'label': label,
                    'method': 'sliding_window',
                    'distance': distance,
                    'query_timestamp': query_msg.get('timestamp'),
                    'candidate_timestamp': candidate_msg.get('timestamp')
                })

            # Negative samples: outside window (sample a few)
            far_indices = [j for j in range(len(self.user_messages)) if abs(j - i) > window_size]

            if far_indices:
                negative_samples = random.sample(far_indices, min(5, len(far_indices)))

                for j in negative_samples:
                    candidate_msg = self.user_messages[j]

                    training_data.append({
                        'query': query_text,
                        'candidate_session_id': candidate_msg.get('session_id'),
                        'candidate_text': candidate_msg.get('text', ''),
                        'label': 0.0,
                        'method': 'sliding_window',
                        'distance': abs(j - i),
                        'query_timestamp': query_msg.get('timestamp'),
                        'candidate_timestamp': candidate_msg.get('timestamp')
                    })

        return training_data

    def generate_hybrid_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data: hybrid strategy (best).

        Strategy:
        - Combine session distance, time distance, message distance
        - Relevance = session(50%) + time(30%) + message(20%)

        Returns:
            List of training samples
        """
        training_data = []

        for i, query_msg in enumerate(self.user_messages):
            query_text = query_msg.get('text', '')
            if not query_text:
                continue

            # Calculate relevance for all candidates
            candidates = []
            for j, candidate_msg in enumerate(self.user_messages):
                if i == j:
                    continue

                relevance = self._calculate_relevance(query_msg, candidate_msg, i, j)

                candidates.append({
                    'message': candidate_msg,
                    'relevance': relevance,
                    'index': j
                })

            # Sort by relevance
            candidates.sort(key=lambda x: x['relevance'], reverse=True)

            # Sampling strategy:
            # - Top 10 high relevance (positive)
            # - Bottom 5 low relevance (negative)
            # - Middle 5 random (medium)

            for candidate in candidates[:10]:  # High relevance
                training_data.append({
                    'query': query_text,
                    'candidate_session_id': candidate['message'].get('session_id'),
                    'candidate_text': candidate['message'].get('text', ''),
                    'label': candidate['relevance'],
                    'method': 'hybrid',
                    'query_timestamp': query_msg.get('timestamp'),
                    'candidate_timestamp': candidate['message'].get('timestamp')
                })

            for candidate in candidates[-5:]:  # Low relevance
                training_data.append({
                    'query': query_text,
                    'candidate_session_id': candidate['message'].get('session_id'),
                    'candidate_text': candidate['message'].get('text', ''),
                    'label': candidate['relevance'],
                    'method': 'hybrid',
                    'query_timestamp': query_msg.get('timestamp'),
                    'candidate_timestamp': candidate['message'].get('timestamp')
                })

            # Middle samples
            if len(candidates) > 15:
                middle_samples = random.sample(candidates[10:-5], min(5, len(candidates) - 15))
                for candidate in middle_samples:
                    training_data.append({
                        'query': query_text,
                        'candidate_session_id': candidate['message'].get('session_id'),
                        'candidate_text': candidate['message'].get('text', ''),
                        'label': candidate['relevance'],
                        'method': 'hybrid',
                        'query_timestamp': query_msg.get('timestamp'),
                        'candidate_timestamp': candidate['message'].get('timestamp')
                    })

        return training_data

    def _calculate_relevance(self, msg1: Dict, msg2: Dict, idx1: int, idx2: int) -> float:
        """
        Calculate relevance score between two messages.

        Factors:
        - Session distance (50%)
        - Time distance (30%)
        - Message distance (20%)

        Returns:
            Relevance score (0-1)
        """
        # 1. Session distance
        session1 = msg1.get('session_id')
        session2 = msg2.get('session_id')

        if session1 == session2:
            session_score = 1.0  # Same session
        else:
            # Different sessions - calculate session distance
            session_list = list(self.sessions.keys())
            if session1 in session_list and session2 in session_list:
                idx_s1 = session_list.index(session1)
                idx_s2 = session_list.index(session2)
                session_dist = abs(idx_s2 - idx_s1)

                if session_dist == 1:
                    session_score = 0.7  # Adjacent sessions
                else:
                    session_score = 0.3 / session_dist  # Farther sessions
            else:
                session_score = 0.3

        # 2. Time distance
        time1 = self._parse_timestamp(msg1.get('timestamp'))
        time2 = self._parse_timestamp(msg2.get('timestamp'))
        time_diff = abs((time2 - time1).total_seconds() / 86400)  # Days
        time_score = math.exp(-time_diff / 7)  # 7-day half-life

        # 3. Message distance
        msg_dist = abs(idx2 - idx1)
        msg_score = 1.0 / (1.0 + msg_dist / 10)  # Decay every 10 messages

        # Weighted average
        relevance = (
            session_score * 0.5 +
            time_score * 0.3 +
            msg_score * 0.2
        )

        return relevance

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string."""
        if not timestamp_str:
            return datetime.min

        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.min


def generate_distance_training_data(
    events: List[Dict[str, Any]],
    method: str = 'session_based'
) -> List[Dict[str, Any]]:
    """
    Generate distance-based training data.

    Args:
        events: List of event dictionaries
        method: 'session_based', 'time_decay', 'sliding_window', or 'hybrid'

    Returns:
        List of training samples
    """
    trainer = DistanceBasedTrainer(events)

    if method == 'session_based':
        return trainer.generate_session_based_data()
    elif method == 'time_decay':
        return trainer.generate_time_decay_data()
    elif method == 'sliding_window':
        return trainer.generate_sliding_window_data()
    elif method == 'hybrid':
        return trainer.generate_hybrid_data()
    else:
        raise ValueError(f"Unknown method: {method}")
