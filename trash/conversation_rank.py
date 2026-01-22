#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConversationRank module for Phase 6.1.

Implements PageRank-like algorithm for conversation importance scoring.
"""

import re
import math
from typing import List, Dict, Any, Set
from datetime import datetime
from collections import defaultdict, Counter


class ConversationRank:
    """
    Calculate conversation importance scores (similar to PageRank).

    Factors:
    - Conversation depth (message count, technical keywords)
    - Reference count (how many later conversations reference this one)
    - Solution quality (has complete problem-solution pattern)
    - Recency (time decay)
    """

    # Technical keywords for depth scoring
    TECHNICAL_KEYWORDS = {
        # Programming languages
        'python', 'javascript', 'java', 'go', 'rust', 'typescript', 'c++', 'c#',

        # Technical concepts
        'async', 'asyncio', '异步', '协程', 'coroutine', 'concurrent', '并发',
        'database', '数据库', 'sql', 'nosql', 'redis', 'cache', '缓存',
        'api', 'rest', 'graphql', 'http', 'websocket', 'grpc',
        'performance', '性能', 'optimization', '优化', 'benchmark',
        'test', '测试', 'debug', '调试', 'error', '错误', 'exception', '异常',

        # Tech stack
        'django', 'flask', 'fastapi', 'react', 'vue', 'node', 'express',
        'docker', 'kubernetes', 'k8s', 'aws', 'azure', 'gcp',
        'postgres', 'mysql', 'mongodb', 'elasticsearch',

        # Design patterns
        'singleton', 'factory', 'observer', 'mvc', 'mvvm',
        'microservice', '微服务', 'architecture', '架构', 'design pattern',

        # Algorithms
        'algorithm', '算法', 'sort', '排序', 'search', '搜索',
        'tree', '树', 'graph', '图', 'hash', '哈希',
    }

    def __init__(self, events: List[Dict[str, Any]]):
        """
        Initialize ConversationRank calculator.

        Args:
            events: List of event dictionaries
        """
        self.events = events
        self.sessions = self._group_by_session()
        self.session_list = list(self.sessions.keys())

    def _group_by_session(self) -> Dict[str, List[Dict]]:
        """Group events by session."""
        sessions = defaultdict(list)
        for event in self.events:
            session_id = event.get('session_id')
            if session_id:
                sessions[session_id].append(event)
        return dict(sessions)

    def calculate_rank(self, session_id: str) -> float:
        """
        Calculate ConversationRank for a session.

        Returns:
            Score between 0 and 1
        """
        if session_id not in self.sessions:
            return 0.0

        session = self.sessions[session_id]
        score = 0.0

        # 1. Base score: conversation length (20%)
        message_count = len(session)
        base_score = min(message_count / 50, 1.0)  # 50 messages = max
        score += base_score * 0.2

        # 2. Depth score: technical keyword density (20%)
        tech_keywords = self._count_technical_keywords(session)
        depth_score = min(tech_keywords / 20, 1.0)  # 20 keywords = max
        score += depth_score * 0.2

        # 3. Reference score: how many later sessions reference this (30%)
        reference_count = self._count_references(session_id)
        reference_score = min(reference_count / 5, 1.0)  # 5 references = max
        score += reference_score * 0.3

        # 4. Quality score: has complete problem-solution pattern (20%)
        has_solution = self._detect_solution_pattern(session)
        quality_score = 1.0 if has_solution else 0.5
        score += quality_score * 0.2

        # 5. Recency score: time decay (10%)
        if session:
            days_ago = self._days_since(session[0].get('timestamp'))
            recency_score = math.exp(-days_ago / 90)  # 90-day half-life
            score += recency_score * 0.1

        return score

    def calculate_all_ranks(self) -> Dict[str, float]:
        """
        Calculate ConversationRank for all sessions.

        Returns:
            Dict mapping session_id to rank score
        """
        ranks = {}
        for session_id in self.sessions:
            ranks[session_id] = self.calculate_rank(session_id)
        return ranks

    def _count_technical_keywords(self, session: List[Dict]) -> int:
        """Count unique technical keywords in session."""
        text = ' '.join([msg.get('text', '').lower() for msg in session])
        found_keywords = set()

        for keyword in self.TECHNICAL_KEYWORDS:
            if keyword in text:
                found_keywords.add(keyword)

        return len(found_keywords)

    def _count_references(self, target_session_id: str) -> int:
        """
        Count how many later sessions reference this session.

        Reference signals:
        - Explicit: "之前讨论过", "上次提到", "那段代码"
        - Implicit: High keyword overlap (>60%)
        """
        if target_session_id not in self.sessions:
            return 0

        target_session = self.sessions[target_session_id]
        target_timestamp = self._get_session_timestamp(target_session)
        target_keywords = self._extract_key_phrases(target_session)

        reference_count = 0

        for other_session_id, other_session in self.sessions.items():
            if other_session_id == target_session_id:
                continue

            other_timestamp = self._get_session_timestamp(other_session)

            # Only count later sessions
            if other_timestamp <= target_timestamp:
                continue

            # Check explicit references
            if self._has_explicit_reference(other_session):
                reference_count += 1
                continue

            # Check implicit references (keyword overlap)
            other_keywords = self._extract_key_phrases(other_session)
            overlap = self._calculate_keyword_overlap(target_keywords, other_keywords)

            if overlap > 0.6:  # 60% overlap threshold
                reference_count += 0.5  # Implicit reference counts as 0.5

        return reference_count

    def _detect_solution_pattern(self, session: List[Dict]) -> bool:
        """
        Detect if session has complete problem-solution pattern.

        Pattern:
        1. User asks question (with problem keywords)
        2. Assistant provides detailed answer (code or long explanation)
        3. User confirms or follows up (indicates helpfulness)
        """
        if len(session) < 3:
            return False

        # Check for problem keywords
        problem_keywords = [
            '如何', '怎么', '为什么', '错误', 'error', 'how', 'why', 'issue',
            'problem', '问题', 'bug', 'help', '帮助'
        ]

        has_question = any(
            any(kw in msg.get('text', '').lower() for kw in problem_keywords)
            for msg in session if msg.get('role') == 'user'
        )

        # Check for code blocks or detailed answers
        has_code = any('```' in msg.get('text', '') for msg in session if msg.get('role') == 'assistant')
        has_detailed_answer = any(
            len(msg.get('text', '')) > 200
            for msg in session if msg.get('role') == 'assistant'
        )

        # Check for user confirmation
        confirmation_keywords = [
            '谢谢', '明白', '懂了', 'thanks', 'got it', 'works', 'solved',
            '解决', '成功', 'success'
        ]

        has_confirmation = any(
            any(kw in msg.get('text', '').lower() for kw in confirmation_keywords)
            for msg in session if msg.get('role') == 'user'
        )

        return has_question and (has_code or has_detailed_answer) and has_confirmation

    def _has_explicit_reference(self, session: List[Dict]) -> bool:
        """Check if session has explicit reference phrases."""
        reference_phrases = [
            '之前讨论', '上次提到', '那段代码', '那次对话',
            'previously discussed', 'mentioned before', 'that code', 'that conversation'
        ]

        text = ' '.join([msg.get('text', '').lower() for msg in session])

        return any(phrase in text for phrase in reference_phrases)

    def _extract_key_phrases(self, session: List[Dict]) -> Set[str]:
        """Extract key technical phrases from session."""
        text = ' '.join([msg.get('text', '').lower() for msg in session])

        # Extract technical keywords
        keywords = set()
        for keyword in self.TECHNICAL_KEYWORDS:
            if keyword in text:
                keywords.add(keyword)

        return keywords

    def _calculate_keyword_overlap(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """Calculate keyword overlap ratio."""
        if not keywords1 or not keywords2:
            return 0.0

        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)

        return intersection / union if union > 0 else 0.0

    def _get_session_timestamp(self, session: List[Dict]) -> datetime:
        """Get timestamp of first message in session."""
        if not session:
            return datetime.min

        timestamp_str = session[0].get('timestamp')
        if not timestamp_str:
            return datetime.min

        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.min

    def _days_since(self, timestamp_str: str) -> float:
        """Calculate days since timestamp."""
        if not timestamp_str:
            return 999999

        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            delta = datetime.now(timestamp.tzinfo) - timestamp
            return delta.total_seconds() / 86400
        except (ValueError, AttributeError):
            return 999999


def calculate_conversation_ranks(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate ConversationRank for all sessions.

    Args:
        events: List of event dictionaries

    Returns:
        Dict mapping session_id to rank score (0-1)
    """
    ranker = ConversationRank(events)
    return ranker.calculate_all_ranks()


def get_top_conversations(events: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top N conversations by ConversationRank.

    Args:
        events: List of event dictionaries
        top_n: Number of top conversations to return

    Returns:
        List of dicts with session_id and rank
    """
    ranks = calculate_conversation_ranks(events)

    # Sort by rank
    sorted_sessions = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    return [
        {'session_id': session_id, 'rank': rank}
        for session_id, rank in sorted_sessions[:top_n]
    ]
