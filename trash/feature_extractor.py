#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction module for Phase 6.1.

Extracts 6 classes of features for CTR prediction:
1. Query features
2. Conversation features
3. Match features
4. User history features
5. Context features
6. Position features
"""

import re
import math
from typing import List, Dict, Any, Set
from datetime import datetime
from collections import Counter


class QueryFeatures:
    """Extract query-related features."""

    def __init__(self):
        self.tech_keywords = self._load_tech_keywords()

    def _load_tech_keywords(self) -> Set[str]:
        """Load technical keywords."""
        return {
            'python', 'javascript', 'java', 'go', 'rust', 'typescript',
            'async', 'asyncio', '异步', '协程', 'concurrent', '并发',
            'database', '数据库', 'sql', 'nosql', 'redis', 'cache',
            'api', 'rest', 'http', 'performance', '性能', 'test', '测试'
        }

    def extract(self, query: str) -> Dict[str, float]:
        """
        Extract query features.

        Args:
            query: Query string

        Returns:
            Dict of features
        """
        query_lower = query.lower()

        return {
            # Basic features
            'query_length': float(len(query)),
            'query_word_count': float(len(query.split())),
            'has_code': 1.0 if '```' in query else 0.0,

            # Query type
            'is_how_question': 1.0 if any(w in query_lower for w in ['如何', 'how']) else 0.0,
            'is_why_question': 1.0 if any(w in query_lower for w in ['为什么', 'why']) else 0.0,
            'is_what_question': 1.0 if any(w in query_lower for w in ['什么', 'what']) else 0.0,

            # Technical depth
            'tech_keyword_count': float(sum(1 for kw in self.tech_keywords if kw in query_lower)),
            'has_version_number': 1.0 if re.search(r'\d+\.\d+', query) else 0.0,

            # Time-related
            'has_time_expression': 1.0 if any(w in query_lower for w in ['昨天', '上周', '最近', 'yesterday', 'last week', 'recent']) else 0.0,
        }


class ConversationFeatures:
    """Extract conversation-related features."""

    def __init__(self):
        self.tech_keywords = self._load_tech_keywords()

    def _load_tech_keywords(self) -> Set[str]:
        """Load technical keywords."""
        return {
            'python', 'javascript', 'java', 'go', 'rust', 'typescript',
            'async', 'asyncio', '异步', '协程', 'concurrent', '并发',
            'database', '数据库', 'sql', 'nosql', 'redis', 'cache',
            'api', 'rest', 'http', 'performance', '性能', 'test', '测试'
        }

    def extract(self, conversation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract conversation features.

        Args:
            conversation: Conversation dict with messages

        Returns:
            Dict of features
        """
        messages = conversation.get('messages', [])
        if not messages:
            return self._empty_features()

        # Calculate features
        message_count = len(messages)
        avg_length = sum(len(m.get('text', '')) for m in messages) / max(message_count, 1)

        # Code blocks
        code_blocks = sum(1 for m in messages if '```' in m.get('text', ''))
        has_code = 1.0 if code_blocks > 0 else 0.0

        # Conversation structure
        turn_count = self._count_turns(messages)
        has_solution = 1.0 if self._detect_solution(messages) else 0.0
        has_confirmation = 1.0 if self._has_confirmation(messages) else 0.0

        # Technical density
        tech_density = self._calculate_tech_density(messages)
        unique_tech_keywords = self._count_unique_tech_keywords(messages)

        # Time features
        timestamp = conversation.get('timestamp')
        days_ago = self._days_since(timestamp) if timestamp else 999999
        is_recent = 1.0 if days_ago <= 7 else 0.0

        # ConversationRank
        conversation_rank = conversation.get('rank', 0.5)

        return {
            # Quality
            'message_count': float(message_count),
            'avg_message_length': avg_length,
            'has_code_block': has_code,
            'code_block_count': float(code_blocks),

            # Structure
            'turn_count': float(turn_count),
            'has_solution': has_solution,
            'has_confirmation': has_confirmation,

            # Technical density
            'tech_keyword_density': tech_density,
            'unique_tech_keywords': float(unique_tech_keywords),

            # Time
            'days_ago': days_ago,
            'is_recent': is_recent,

            # ConversationRank
            'conversation_rank': conversation_rank,
        }

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features."""
        return {
            'message_count': 0.0,
            'avg_message_length': 0.0,
            'has_code_block': 0.0,
            'code_block_count': 0.0,
            'turn_count': 0.0,
            'has_solution': 0.0,
            'has_confirmation': 0.0,
            'tech_keyword_density': 0.0,
            'unique_tech_keywords': 0.0,
            'days_ago': 999999.0,
            'is_recent': 0.0,
            'conversation_rank': 0.0,
        }

    def _count_turns(self, messages: List[Dict]) -> int:
        """Count conversation turns (role changes)."""
        if not messages:
            return 0

        turns = 1
        prev_role = messages[0].get('role')

        for msg in messages[1:]:
            if msg.get('role') != prev_role:
                turns += 1
                prev_role = msg.get('role')

        return turns

    def _detect_solution(self, messages: List[Dict]) -> bool:
        """Detect if conversation has solution pattern."""
        if len(messages) < 3:
            return False

        # Has question
        problem_keywords = ['如何', '怎么', '为什么', '错误', 'error', 'how', 'why', 'issue']
        has_question = any(
            any(kw in m.get('text', '').lower() for kw in problem_keywords)
            for m in messages if m.get('role') == 'user'
        )

        # Has code or detailed answer
        has_code = any('```' in m.get('text', '') for m in messages if m.get('role') == 'assistant')
        has_detailed = any(len(m.get('text', '')) > 200 for m in messages if m.get('role') == 'assistant')

        # Has confirmation
        confirmation_keywords = ['谢谢', '明白', '懂了', 'thanks', 'got it', 'works', 'solved']
        has_confirmation = any(
            any(kw in m.get('text', '').lower() for kw in confirmation_keywords)
            for m in messages if m.get('role') == 'user'
        )

        return has_question and (has_code or has_detailed) and has_confirmation

    def _has_confirmation(self, messages: List[Dict]) -> bool:
        """Check if user confirmed/thanked."""
        confirmation_keywords = ['谢谢', '明白', '懂了', 'thanks', 'got it', 'works', 'solved']
        return any(
            any(kw in m.get('text', '').lower() for kw in confirmation_keywords)
            for m in messages if m.get('role') == 'user'
        )

    def _calculate_tech_density(self, messages: List[Dict]) -> float:
        """Calculate technical keyword density."""
        text = ' '.join([m.get('text', '').lower() for m in messages])
        if not text:
            return 0.0

        word_count = len(text.split())
        tech_count = sum(1 for kw in self.tech_keywords if kw in text)

        return tech_count / max(word_count, 1)

    def _count_unique_tech_keywords(self, messages: List[Dict]) -> int:
        """Count unique technical keywords."""
        text = ' '.join([m.get('text', '').lower() for m in messages])
        found = set()

        for kw in self.tech_keywords:
            if kw in text:
                found.add(kw)

        return len(found)

    def _days_since(self, timestamp_str: str) -> float:
        """Calculate days since timestamp."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            delta = datetime.now(timestamp.tzinfo) - timestamp
            return delta.total_seconds() / 86400
        except (ValueError, AttributeError):
            return 999999.0


class MatchFeatures:
    """Extract query-conversation match features."""

    def extract(self, query: str, conversation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract match features.

        Args:
            query: Query string
            conversation: Conversation dict

        Returns:
            Dict of features
        """
        messages = conversation.get('messages', [])
        if not messages:
            return self._empty_features()

        # Get conversation text
        conv_text = ' '.join([m.get('text', '') for m in messages])
        first_message = messages[0].get('text', '')

        # BM25 score (placeholder - will be filled by caller)
        bm25_score = conversation.get('bm25_score', 0.0)

        # Exact matches
        exact_matches = self._count_exact_matches(query, conv_text)

        # Keyword overlap
        keyword_overlap = self._calculate_keyword_overlap(query, conv_text)
        tech_keyword_overlap = self._calculate_tech_keyword_overlap(query, conv_text)

        # Semantic matches
        query_in_title = 1.0 if query.lower() in first_message.lower() else 0.0

        return {
            'bm25_score': bm25_score,
            'exact_match_count': float(exact_matches),
            'keyword_overlap': keyword_overlap,
            'tech_keyword_overlap': tech_keyword_overlap,
            'query_in_title': query_in_title,
        }

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features."""
        return {
            'bm25_score': 0.0,
            'exact_match_count': 0.0,
            'keyword_overlap': 0.0,
            'tech_keyword_overlap': 0.0,
            'query_in_title': 0.0,
        }

    def _count_exact_matches(self, query: str, text: str) -> int:
        """Count exact phrase matches."""
        query_lower = query.lower()
        text_lower = text.lower()
        return text_lower.count(query_lower)

    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """Calculate keyword overlap ratio."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        if not query_words:
            return 0.0

        intersection = len(query_words & text_words)
        return intersection / len(query_words)

    def _calculate_tech_keyword_overlap(self, query: str, text: str) -> float:
        """Calculate technical keyword overlap."""
        tech_keywords = {
            'python', 'javascript', 'async', 'database', 'api', 'performance', 'test'
        }

        query_tech = set(kw for kw in tech_keywords if kw in query.lower())
        text_tech = set(kw for kw in tech_keywords if kw in text.lower())

        if not query_tech:
            return 0.0

        intersection = len(query_tech & text_tech)
        return intersection / len(query_tech)


class UserHistoryFeatures:
    """Extract user history features."""

    def extract(self, user_history: Dict[str, Any], conversation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract user history features.

        Args:
            user_history: User history dict (can include pattern_clusters)
            conversation: Conversation dict

        Returns:
            Dict of features
        """
        if not user_history:
            return self._empty_features()

        # Topic preference
        topic_preference = self._calculate_topic_preference(user_history, conversation)

        # Time preference
        prefers_recent = self._prefers_recent(user_history)

        # Conversation type preference
        prefers_long = self._prefers_long_conversations(user_history)
        prefers_code = self._prefers_code_heavy(user_history)

        # Reference history
        has_referenced = self._has_referenced_before(user_history, conversation)

        # Pattern clustering features (Phase 4.5 integration)
        pattern_features = self._extract_pattern_features(user_history, conversation)

        # Combine all features
        features = {
            'user_topic_preference': topic_preference,
            'user_prefers_recent': prefers_recent,
            'prefers_long_conversations': prefers_long,
            'prefers_code_heavy': prefers_code,
            'has_referenced_before': has_referenced,
        }
        features.update(pattern_features)

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features."""
        return {
            'user_topic_preference': 0.5,
            'user_prefers_recent': 0.5,
            'prefers_long_conversations': 0.5,
            'prefers_code_heavy': 0.5,
            'has_referenced_before': 0.0,
            # Pattern clustering features
            'in_frequent_query_cluster': 0.0,
            'session_type_match': 0.0,
            'is_recurring_problem': 0.0,
            'topic_cluster_match': 0.0,
        }

    def _calculate_topic_preference(self, user_history: Dict, conversation: Dict) -> float:
        """Calculate user's preference for this conversation's topic."""
        # Simplified: check if conversation topic is in user's frequent topics
        frequent_topics = user_history.get('frequent_topics', [])
        conv_topic = conversation.get('topic', '')

        if conv_topic in frequent_topics:
            return 1.0
        return 0.5

    def _prefers_recent(self, user_history: Dict) -> float:
        """Check if user prefers recent conversations."""
        # Simplified: check average age of selected results
        avg_age = user_history.get('avg_selected_age_days', 30)
        return 1.0 if avg_age < 14 else 0.5

    def _prefers_long_conversations(self, user_history: Dict) -> float:
        """Check if user prefers long conversations."""
        avg_length = user_history.get('avg_selected_length', 10)
        return 1.0 if avg_length > 20 else 0.5

    def _prefers_code_heavy(self, user_history: Dict) -> float:
        """Check if user prefers code-heavy conversations."""
        code_ratio = user_history.get('selected_code_ratio', 0.5)
        return code_ratio

    def _has_referenced_before(self, user_history: Dict, conversation: Dict) -> float:
        """Check if user has referenced this conversation before."""
        referenced_sessions = user_history.get('referenced_sessions', [])
        session_id = conversation.get('session_id', '')
        return 1.0 if session_id in referenced_sessions else 0.0

    def _extract_pattern_features(self, user_history: Dict, conversation: Dict) -> Dict[str, float]:
        """
        Extract features from pattern clustering (Phase 4.5 integration).

        Uses pattern_clusters from user_history:
        - query_clusters: Similar query groups
        - topic_aggregation: Hierarchical topics
        - session_clusters: Session type classification
        - problem_patterns: Recurring problems

        Args:
            user_history: User history with optional pattern_clusters
            conversation: Conversation dict

        Returns:
            Dict of pattern-based features
        """
        pattern_clusters = user_history.get('pattern_clusters', {})

        if not pattern_clusters:
            return {
                'in_frequent_query_cluster': 0.0,
                'session_type_match': 0.0,
                'is_recurring_problem': 0.0,
                'topic_cluster_match': 0.0,
            }

        features = {}

        # 1. Query cluster match
        # Check if conversation is in a frequent query cluster
        query_clusters = pattern_clusters.get('query_clusters', [])
        in_cluster = self._check_query_cluster_match(conversation, query_clusters)
        features['in_frequent_query_cluster'] = 1.0 if in_cluster else 0.0

        # 2. Session type match
        # Check if conversation type matches user's preferred session type
        session_clusters = pattern_clusters.get('session_clusters', [])
        type_match = self._check_session_type_match(conversation, session_clusters)
        features['session_type_match'] = type_match

        # 3. Recurring problem
        # Check if conversation addresses a recurring problem
        problem_patterns = pattern_clusters.get('problem_patterns', [])
        is_recurring = self._check_recurring_problem(conversation, problem_patterns)
        features['is_recurring_problem'] = 1.0 if is_recurring else 0.0

        # 4. Topic cluster match
        # Check if conversation topic is in user's frequent topic clusters
        topic_aggregation = pattern_clusters.get('topic_aggregation', {})
        topic_match = self._check_topic_cluster_match(conversation, topic_aggregation)
        features['topic_cluster_match'] = topic_match

        return features

    def _check_query_cluster_match(self, conversation: Dict, query_clusters: List[Dict]) -> bool:
        """Check if conversation is in a frequent query cluster."""
        if not query_clusters:
            return False

        session_id = conversation.get('session_id', '')

        # Check if this session appears in any query cluster
        for cluster in query_clusters:
            if session_id in cluster.get('sessions', []):
                return True

        return False

    def _check_session_type_match(self, conversation: Dict, session_clusters: List[Dict]) -> float:
        """
        Check if conversation type matches user's preferred session type.

        Returns:
            1.0 if matches top session type
            0.5 if matches second/third type
            0.0 otherwise
        """
        if not session_clusters:
            return 0.5

        session_id = conversation.get('session_id', '')

        # Find which cluster this session belongs to
        for i, cluster in enumerate(session_clusters):
            sessions = cluster.get('sessions', [])
            if any(s.get('session_id') == session_id for s in sessions):
                # Top cluster = 1.0, second = 0.7, third = 0.5, rest = 0.3
                if i == 0:
                    return 1.0
                elif i == 1:
                    return 0.7
                elif i == 2:
                    return 0.5
                else:
                    return 0.3

        return 0.5  # Default

    def _check_recurring_problem(self, conversation: Dict, problem_patterns: List[Dict]) -> bool:
        """Check if conversation addresses a recurring problem."""
        if not problem_patterns:
            return False

        session_id = conversation.get('session_id', '')

        # Check if this session is in any problem pattern
        for pattern in problem_patterns:
            if session_id in pattern.get('sessions', []):
                return True

        return False

    def _check_topic_cluster_match(self, conversation: Dict, topic_aggregation: Dict) -> float:
        """
        Check if conversation topic matches user's frequent topics.

        Returns:
            1.0 if in top topic
            0.7 if in top 3 topics
            0.5 if in top 5 topics
            0.0 otherwise
        """
        if not topic_aggregation:
            return 0.5

        hierarchy = topic_aggregation.get('hierarchy', {})
        if not hierarchy:
            return 0.5

        # Get conversation text
        messages = conversation.get('messages', [])
        if not messages:
            return 0.5

        conv_text = ' '.join([m.get('text', '').lower() for m in messages])

        # Check which topics match
        topic_matches = []
        for topic_name, topic_info in hierarchy.items():
            keywords = topic_info.get('keywords', [])
            if any(kw in conv_text for kw in keywords):
                count = topic_info.get('count', 0)
                topic_matches.append((topic_name, count))

        if not topic_matches:
            return 0.0

        # Sort by count
        topic_matches.sort(key=lambda x: x[1], reverse=True)

        # Get all topics sorted by count
        all_topics = sorted(hierarchy.items(), key=lambda x: x[1].get('count', 0), reverse=True)

        # Find rank of matched topic
        matched_topic = topic_matches[0][0]
        for i, (topic_name, _) in enumerate(all_topics):
            if topic_name == matched_topic:
                if i == 0:
                    return 1.0  # Top topic
                elif i < 3:
                    return 0.7  # Top 3
                elif i < 5:
                    return 0.5  # Top 5
                else:
                    return 0.3

        return 0.0


class ContextFeatures:
    """Extract context features."""

    def extract(self, context: Dict[str, Any], conversation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract context features.

        Args:
            context: Current context dict
            conversation: Conversation dict

        Returns:
            Dict of features
        """
        if not context:
            return self._empty_features()

        # Session continuity
        current_session = context.get('current_session')
        conv_session = conversation.get('session_id')
        is_same_session = 1.0 if current_session == conv_session else 0.0

        recent_sessions = context.get('recent_sessions', [])
        is_recent_session = 1.0 if conv_session in recent_sessions else 0.0

        # Follow-up
        is_follow_up = 1.0 if context.get('is_follow_up') else 0.0

        return {
            'is_same_session': is_same_session,
            'is_recent_session': is_recent_session,
            'is_follow_up': is_follow_up,
        }

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features."""
        return {
            'is_same_session': 0.0,
            'is_recent_session': 0.0,
            'is_follow_up': 0.0,
        }


class PositionFeatures:
    """Extract position features."""

    def extract(self, position: int) -> Dict[str, float]:
        """
        Extract position features.

        Args:
            position: Result position (1-indexed)

        Returns:
            Dict of features
        """
        return {
            'position': float(position),
            'position_bias': 1.0 / math.log2(position + 1),  # DCG-style
            'is_top_1': 1.0 if position == 1 else 0.0,
            'is_top_3': 1.0 if position <= 3 else 0.0,
            'is_top_5': 1.0 if position <= 5 else 0.0,
        }


class FeatureExtractor:
    """Main feature extractor combining all feature classes."""

    def __init__(self):
        self.query_features = QueryFeatures()
        self.conversation_features = ConversationFeatures()
        self.match_features = MatchFeatures()
        self.user_history_features = UserHistoryFeatures()
        self.context_features = ContextFeatures()
        self.position_features = PositionFeatures()

    def extract_all(
        self,
        query: str,
        conversation: Dict[str, Any],
        user_history: Dict[str, Any],
        context: Dict[str, Any],
        position: int
    ) -> Dict[str, float]:
        """
        Extract all features.

        Args:
            query: Query string
            conversation: Conversation dict
            user_history: User history dict
            context: Context dict
            position: Result position

        Returns:
            Dict of all features
        """
        features = {}

        # Extract from each feature class
        features.update(self.query_features.extract(query))
        features.update(self.conversation_features.extract(conversation))
        features.update(self.match_features.extract(query, conversation))
        features.update(self.user_history_features.extract(user_history, conversation))
        features.update(self.context_features.extract(context, conversation))
        features.update(self.position_features.extract(position))

        return features

    def get_feature_names(self) -> List[str]:
        """Get all feature names in order."""
        # Extract features from a dummy example to get names
        dummy_features = self.extract_all(
            query="test",
            conversation={'messages': [], 'rank': 0.5},
            user_history={},
            context={},
            position=1
        )
        return sorted(dummy_features.keys())
