#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering and aggregation module for Phase 4.5.

Features:
- Query clustering (similar queries)
- Topic aggregation (hierarchical topics)
- Session clustering (conversation types)
- Problem pattern recognition (recurring issues)
"""

import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
from difflib import SequenceMatcher


class PatternClusterer:
    """
    Clusters and aggregates patterns from user behavior.
    """

    def __init__(self, events: List[Dict[str, Any]]):
        """
        Initialize pattern clusterer.

        Args:
            events: List of event dictionaries
        """
        self.events = events
        self.user_messages = [e for e in events if e.get("role") == "user"]

    def cluster_queries(self, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Cluster similar queries together.

        Args:
            similarity_threshold: Minimum similarity to group queries (0-1)

        Returns:
            List of query clusters
        """
        if not self.user_messages:
            return []

        # Extract query texts
        queries = [msg.get("text", "") for msg in self.user_messages]

        # Build clusters
        clusters = []
        used_indices = set()

        for i, query1 in enumerate(queries):
            if i in used_indices:
                continue

            # Start new cluster
            cluster = {
                "representative": query1,
                "queries": [query1],
                "count": 1,
                "sessions": [self.user_messages[i].get("session_id")]
            }

            # Find similar queries
            for j, query2 in enumerate(queries):
                if j <= i or j in used_indices:
                    continue

                similarity = self._calculate_similarity(query1, query2)
                if similarity >= similarity_threshold:
                    cluster["queries"].append(query2)
                    cluster["count"] += 1
                    cluster["sessions"].append(self.user_messages[j].get("session_id"))
                    used_indices.add(j)

            used_indices.add(i)
            clusters.append(cluster)

        # Sort by count
        clusters.sort(key=lambda x: x["count"], reverse=True)

        # Only return clusters with more than 1 query
        return [c for c in clusters if c["count"] > 1]

    def aggregate_topics(self) -> Dict[str, Any]:
        """
        Aggregate fine-grained topics into hierarchical structure.

        Returns:
            Dict with topic hierarchy
        """
        # Define topic hierarchy
        topic_hierarchy = {
            "编程语言": {
                "keywords": ["python", "javascript", "java", "go", "rust", "ruby"],
                "subtopics": {}
            },
            "异步编程": {
                "keywords": ["async", "asyncio", "异步", "协程", "concurrent", "并发"],
                "subtopics": {}
            },
            "数据存储": {
                "keywords": ["database", "数据库", "sql", "nosql", "cache", "缓存", "redis"],
                "subtopics": {}
            },
            "性能优化": {
                "keywords": ["performance", "性能", "optimization", "优化", "速度", "效率"],
                "subtopics": {}
            },
            "测试调试": {
                "keywords": ["test", "测试", "debug", "调试", "error", "错误", "bug"],
                "subtopics": {}
            },
            "API开发": {
                "keywords": ["api", "rest", "http", "接口", "endpoint"],
                "subtopics": {}
            },
            "架构设计": {
                "keywords": ["architecture", "架构", "design", "设计", "pattern", "模式"],
                "subtopics": {}
            }
        }

        # Count occurrences for each topic
        for topic_name, topic_info in topic_hierarchy.items():
            count = 0
            examples = []

            for event in self.events:
                text = event.get("text", "").lower()
                if any(kw in text for kw in topic_info["keywords"]):
                    count += 1
                    if len(examples) < 3:
                        examples.append(text[:100])

            topic_info["count"] = count
            topic_info["examples"] = examples

        # Sort by count
        sorted_topics = sorted(
            topic_hierarchy.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )

        return {
            "hierarchy": dict(sorted_topics),
            "total_topics": len([t for t in topic_hierarchy.values() if t.get("count", 0) > 0])
        }

    def cluster_sessions(self) -> List[Dict[str, Any]]:
        """
        Cluster sessions by conversation type.

        Returns:
            List of session clusters
        """
        # Group events by session
        sessions = defaultdict(list)
        for event in self.events:
            session_id = event.get("session_id")
            if session_id:
                sessions[session_id].append(event)

        # Classify each session
        session_types = {
            "学习型": [],
            "问题解决型": [],
            "探索型": [],
            "实践型": []
        }

        for session_id, events in sessions.items():
            session_type = self._classify_session(events)
            session_types[session_type].append({
                "session_id": session_id,
                "event_count": len(events),
                "sample": events[0].get("text", "")[:100] if events else ""
            })

        # Format results
        clusters = []
        for type_name, sessions_list in session_types.items():
            if sessions_list:
                clusters.append({
                    "type": type_name,
                    "count": len(sessions_list),
                    "sessions": sessions_list[:5]  # Top 5 examples
                })

        return sorted(clusters, key=lambda x: x["count"], reverse=True)

    def recognize_problem_patterns(self) -> List[Dict[str, Any]]:
        """
        Recognize recurring problem patterns.

        Returns:
            List of problem patterns
        """
        # Extract problem-related messages
        problem_keywords = [
            "错误", "error", "bug", "问题", "issue",
            "不工作", "doesn't work", "失败", "failed",
            "报错", "exception", "异常"
        ]

        problem_messages = []
        for event in self.user_messages:
            text = event.get("text", "").lower()
            if any(kw in text for kw in problem_keywords):
                problem_messages.append(event)

        if not problem_messages:
            return []

        # Cluster similar problems
        problem_clusters = []
        used_indices = set()

        for i, msg1 in enumerate(problem_messages):
            if i in used_indices:
                continue

            text1 = msg1.get("text", "")
            cluster = {
                "pattern": self._extract_problem_pattern(text1),
                "occurrences": [text1],
                "count": 1,
                "sessions": [msg1.get("session_id")]
            }

            # Find similar problems
            for j, msg2 in enumerate(problem_messages):
                if j <= i or j in used_indices:
                    continue

                text2 = msg2.get("text", "")
                if self._are_similar_problems(text1, text2):
                    cluster["occurrences"].append(text2)
                    cluster["count"] += 1
                    cluster["sessions"].append(msg2.get("session_id"))
                    used_indices.add(j)

            used_indices.add(i)
            if cluster["count"] > 1:  # Only recurring patterns
                problem_clusters.append(cluster)

        return sorted(problem_clusters, key=lambda x: x["count"], reverse=True)

    def generate_aggregation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive aggregation report.

        Returns:
            Dict with all aggregation results
        """
        query_clusters = self.cluster_queries(similarity_threshold=0.6)
        topic_aggregation = self.aggregate_topics()
        session_clusters = self.cluster_sessions()
        problem_patterns = self.recognize_problem_patterns()

        # Generate summary
        summary_parts = []

        if query_clusters:
            summary_parts.append(f"发现 {len(query_clusters)} 组相似查询")

        if topic_aggregation["total_topics"] > 0:
            summary_parts.append(f"涵盖 {topic_aggregation['total_topics']} 个主题领域")

        if session_clusters:
            top_session_type = session_clusters[0]
            summary_parts.append(f"主要是{top_session_type['type']}会话 ({top_session_type['count']}次)")

        if problem_patterns:
            summary_parts.append(f"识别出 {len(problem_patterns)} 种重复问题模式")

        summary = "。".join(summary_parts) + "。" if summary_parts else "暂无足够数据进行聚类分析。"

        return {
            "summary": summary,
            "query_clusters": query_clusters[:10],  # Top 10
            "topic_aggregation": topic_aggregation,
            "session_clusters": session_clusters,
            "problem_patterns": problem_patterns[:5]  # Top 5
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, text1, text2).ratio()

    def _classify_session(self, events: List[Dict[str, Any]]) -> str:
        """Classify session type based on content."""
        user_texts = [e.get("text", "").lower() for e in events if e.get("role") == "user"]
        combined_text = " ".join(user_texts)

        # Learning indicators
        learning_keywords = ["什么是", "如何", "怎么", "介绍", "教程", "学习", "what is", "how to", "tutorial"]
        if any(kw in combined_text for kw in learning_keywords):
            return "学习型"

        # Problem-solving indicators
        problem_keywords = ["错误", "bug", "问题", "不工作", "报错", "error", "issue", "doesn't work"]
        if any(kw in combined_text for kw in problem_keywords):
            return "问题解决型"

        # Practice indicators
        practice_keywords = ["实现", "写", "创建", "开发", "implement", "create", "develop", "build"]
        if any(kw in combined_text for kw in practice_keywords):
            return "实践型"

        # Default to exploration
        return "探索型"

    def _extract_problem_pattern(self, text: str) -> str:
        """Extract problem pattern from text."""
        # Remove specific details, keep general pattern
        text = text.lower()

        # Replace specific values with placeholders
        text = re.sub(r'\d+', 'N', text)  # Numbers
        text = re.sub(r'["\'].*?["\']', 'STRING', text)  # Strings

        # Extract key phrases
        if "error" in text or "错误" in text:
            return "错误/异常问题"
        elif "slow" in text or "慢" in text or "performance" in text or "性能" in text:
            return "性能问题"
        elif "not work" in text or "不工作" in text or "failed" in text or "失败" in text:
            return "功能失效问题"
        elif "how to" in text or "如何" in text or "怎么" in text:
            return "使用方法问题"
        else:
            return "一般问题"

    def _are_similar_problems(self, text1: str, text2: str) -> bool:
        """Check if two problems are similar."""
        pattern1 = self._extract_problem_pattern(text1)
        pattern2 = self._extract_problem_pattern(text2)

        # Same pattern type
        if pattern1 == pattern2:
            # Also check text similarity
            similarity = self._calculate_similarity(text1, text2)
            return similarity > 0.4

        return False


def format_aggregation_report(report: Dict[str, Any]) -> str:
    """
    Format aggregation report as readable text.

    Args:
        report: Aggregation report dict

    Returns:
        Formatted text report
    """
    lines = []

    lines.append("# 🔍 模式聚类分析报告\n")
    lines.append(f"## 📝 摘要\n")
    lines.append(f"{report['summary']}\n")

    # Query clusters
    if report["query_clusters"]:
        lines.append(f"## 🔗 相似查询聚类\n")
        for i, cluster in enumerate(report["query_clusters"][:5], 1):
            lines.append(f"{i}. **{cluster['representative'][:80]}** ({cluster['count']} 次)")
            if len(cluster['queries']) > 1:
                lines.append(f"   相似查询: {cluster['queries'][1][:60]}...")
        lines.append("")

    # Topic aggregation
    if report["topic_aggregation"]["total_topics"] > 0:
        lines.append(f"## 📚 话题聚合\n")
        hierarchy = report["topic_aggregation"]["hierarchy"]
        for topic_name, topic_info in list(hierarchy.items())[:5]:
            if topic_info.get("count", 0) > 0:
                lines.append(f"- **{topic_name}**: {topic_info['count']} 次")
        lines.append("")

    # Session clusters
    if report["session_clusters"]:
        lines.append(f"## 💬 会话类型分布\n")
        for cluster in report["session_clusters"]:
            lines.append(f"- **{cluster['type']}**: {cluster['count']} 次会话")
        lines.append("")

    # Problem patterns
    if report["problem_patterns"]:
        lines.append(f"## ⚠️ 重复问题模式\n")
        for i, pattern in enumerate(report["problem_patterns"], 1):
            lines.append(f"{i}. **{pattern['pattern']}** (出现 {pattern['count']} 次)")
            lines.append(f"   示例: {pattern['occurrences'][0][:80]}...")
        lines.append("")

    return "\n".join(lines)
