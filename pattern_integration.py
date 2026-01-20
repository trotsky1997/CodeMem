#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern clustering integration for CTR model.

Provides utilities to integrate Phase 4.5 pattern clustering
with Phase 6.1 CTR-based search ranking.
"""

from typing import List, Dict, Any
from pattern_clusterer import PatternClusterer


def generate_user_history_with_patterns(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate user history dict with pattern clustering results.

    This integrates Phase 4.5 (pattern clustering) with Phase 6.1 (CTR model).

    Args:
        events: List of event dictionaries

    Returns:
        User history dict with pattern_clusters field
    """
    # Generate pattern clusters
    clusterer = PatternClusterer(events)
    aggregation_report = clusterer.generate_aggregation_report()

    # Build user history
    user_history = {
        'pattern_clusters': {
            'query_clusters': aggregation_report['query_clusters'],
            'topic_aggregation': aggregation_report['topic_aggregation'],
            'session_clusters': aggregation_report['session_clusters'],
            'problem_patterns': aggregation_report['problem_patterns'],
        },
        # Additional user history fields can be added here
        'frequent_topics': [
            topic_name for topic_name, topic_info in
            aggregation_report['topic_aggregation']['hierarchy'].items()
            if topic_info.get('count', 0) > 0
        ][:5],  # Top 5 topics
    }

    return user_history


def enrich_search_results_with_patterns(
    results: List[Dict[str, Any]],
    pattern_clusters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Enrich search results with pattern clustering information.

    Adds metadata about:
    - Which query cluster the result belongs to
    - Session type classification
    - Whether it's a recurring problem
    - Topic cluster membership

    Args:
        results: Search results
        pattern_clusters: Pattern clustering results

    Returns:
        Enriched results with pattern metadata
    """
    enriched = []

    for result in results:
        session_id = result.get('session_id', '')

        # Add pattern metadata
        pattern_meta = {
            'in_query_cluster': False,
            'session_type': None,
            'is_recurring_problem': False,
            'topic_clusters': [],
        }

        # Check query clusters
        for cluster in pattern_clusters.get('query_clusters', []):
            if session_id in cluster.get('sessions', []):
                pattern_meta['in_query_cluster'] = True
                break

        # Check session type
        for cluster in pattern_clusters.get('session_clusters', []):
            sessions = cluster.get('sessions', [])
            if any(s.get('session_id') == session_id for s in sessions):
                pattern_meta['session_type'] = cluster.get('type')
                break

        # Check recurring problems
        for pattern in pattern_clusters.get('problem_patterns', []):
            if session_id in pattern.get('sessions', []):
                pattern_meta['is_recurring_problem'] = True
                break

        # Check topic clusters
        topic_hierarchy = pattern_clusters.get('topic_aggregation', {}).get('hierarchy', {})
        result_text = result.get('text', '').lower()
        for topic_name, topic_info in topic_hierarchy.items():
            keywords = topic_info.get('keywords', [])
            if any(kw in result_text for kw in keywords):
                pattern_meta['topic_clusters'].append(topic_name)

        # Add to result
        result_with_meta = result.copy()
        result_with_meta['pattern_meta'] = pattern_meta
        enriched.append(result_with_meta)

    return enriched


def explain_ranking_with_patterns(
    result: Dict[str, Any],
    features: Dict[str, float]
) -> str:
    """
    Generate human-readable explanation of ranking with pattern features.

    Args:
        result: Search result
        features: Extracted features

    Returns:
        Explanation string
    """
    explanations = []

    # Pattern clustering explanations
    if features.get('in_frequent_query_cluster', 0) > 0:
        explanations.append("✓ 属于你经常搜索的查询类型")

    if features.get('session_type_match', 0) > 0.7:
        explanations.append("✓ 匹配你偏好的会话类型")

    if features.get('is_recurring_problem', 0) > 0:
        explanations.append("⚠️ 这是一个反复出现的问题")

    if features.get('topic_cluster_match', 0) > 0.7:
        explanations.append("✓ 属于你常讨论的话题")

    # Other important features
    if features.get('conversation_rank', 0) > 0.7:
        explanations.append("✓ 高质量对话（被多次引用）")

    if features.get('has_solution', 0) > 0:
        explanations.append("✓ 包含完整的解决方案")

    if features.get('has_code_block', 0) > 0:
        explanations.append("✓ 包含代码示例")

    if features.get('is_same_session', 0) > 0:
        explanations.append("✓ 来自当前会话")

    if not explanations:
        explanations.append("相关性匹配")

    return " | ".join(explanations)


def get_pattern_insights(user_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get insights from pattern clustering for display.

    Args:
        user_history: User history with pattern_clusters

    Returns:
        Dict with insights
    """
    pattern_clusters = user_history.get('pattern_clusters', {})

    if not pattern_clusters:
        return {
            'has_patterns': False,
            'insights': []
        }

    insights = []

    # Query clusters
    query_clusters = pattern_clusters.get('query_clusters', [])
    if query_clusters:
        top_cluster = query_clusters[0]
        insights.append(f"你经常搜索：{top_cluster['representative'][:50]}... ({top_cluster['count']} 次)")

    # Session types
    session_clusters = pattern_clusters.get('session_clusters', [])
    if session_clusters:
        top_type = session_clusters[0]
        insights.append(f"主要会话类型：{top_type['type']} ({top_type['count']} 次)")

    # Recurring problems
    problem_patterns = pattern_clusters.get('problem_patterns', [])
    if problem_patterns:
        top_problem = problem_patterns[0]
        insights.append(f"反复出现的问题：{top_problem['pattern']} ({top_problem['count']} 次)")

    # Topic clusters
    topic_agg = pattern_clusters.get('topic_aggregation', {})
    hierarchy = topic_agg.get('hierarchy', {})
    if hierarchy:
        top_topics = sorted(hierarchy.items(), key=lambda x: x[1].get('count', 0), reverse=True)[:3]
        topic_names = [name for name, _ in top_topics if _['count'] > 0]
        if topic_names:
            insights.append(f"常讨论话题：{', '.join(topic_names)}")

    return {
        'has_patterns': len(insights) > 0,
        'insights': insights
    }
