#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern analysis and discovery module for Phase 4.

Features:
- Frequent topics analysis
- Activity time patterns
- Knowledge evolution tracking
- Unresolved questions detection
- Insights report generation
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict


class PatternAnalyzer:
    """
    Analyzes user behavior patterns and generates insights.
    """

    def __init__(self, events: List[Dict[str, Any]]):
        """
        Initialize pattern analyzer.

        Args:
            events: List of event dictionaries from database
        """
        self.events = events
        self.user_messages = [e for e in events if e.get("role") == "user"]
        self.assistant_messages = [e for e in events if e.get("role") == "assistant"]

    def analyze_frequent_topics(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze most frequently discussed topics.

        Args:
            top_n: Number of top topics to return

        Returns:
            List of topics with counts and examples
        """
        # Extract keywords from messages
        keyword_counts = Counter()
        keyword_examples = defaultdict(list)

        # Technical keywords to track
        tech_keywords = {
            "python", "javascript", "java", "go", "rust",
            "async", "asyncio", "异步", "协程",
            "database", "数据库", "sql", "nosql",
            "performance", "性能", "optimization", "优化",
            "test", "测试", "unittest",
            "api", "rest", "http",
            "error", "bug", "错误", "问题",
            "cache", "缓存",
            "index", "索引", "search", "搜索",
        }

        for event in self.events:
            text = event.get("text", "").lower()
            words = re.findall(r'\b\w+\b', text)

            for word in words:
                if word in tech_keywords:
                    keyword_counts[word] += 1
                    if len(keyword_examples[word]) < 3:
                        keyword_examples[word].append(text[:100])

        # Get top topics
        top_topics = []
        for keyword, count in keyword_counts.most_common(top_n):
            top_topics.append({
                "topic": keyword,
                "count": count,
                "examples": keyword_examples[keyword],
                "percentage": (count / len(self.events)) * 100 if self.events else 0
            })

        return top_topics

    def analyze_activity_time(self) -> Dict[str, Any]:
        """
        Analyze activity time patterns.

        Returns:
            Dict with time pattern analysis
        """
        if not self.events:
            return {
                "peak_hours": [],
                "peak_weekdays": [],
                "total_days": 0,
                "max_streak": 0,
                "avg_events_per_day": 0
            }

        # Parse timestamps
        hour_counts = Counter()
        day_counts = Counter()
        weekday_counts = Counter()

        for event in self.events:
            timestamp = event.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour_counts[dt.hour] += 1
                day_counts[dt.date()] += 1
                weekday_counts[dt.strftime("%A")] += 1
            except Exception:
                continue

        # Find peak hours (top 3)
        peak_hours = [hour for hour, _ in hour_counts.most_common(3)]

        # Find peak weekdays
        peak_weekdays = [day for day, _ in weekday_counts.most_common(3)]

        # Calculate activity streak
        sorted_days = sorted(day_counts.keys())
        current_streak = 0
        max_streak = 0
        last_day = None

        for day in sorted_days:
            if last_day is None or (day - last_day).days == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
            last_day = day

        return {
            "peak_hours": peak_hours,
            "peak_weekdays": peak_weekdays,
            "total_days": len(day_counts),
            "max_streak": max_streak,
            "avg_events_per_day": len(self.events) / len(day_counts) if day_counts else 0
        }

    def analyze_knowledge_evolution(self, topic: str) -> Dict[str, Any]:
        """
        Track knowledge evolution for a specific topic.

        Args:
            topic: Topic to track (e.g., "python", "async")

        Returns:
            Dict with evolution analysis
        """
        topic_lower = topic.lower()

        # Find all messages mentioning the topic
        topic_messages = []
        for event in self.events:
            text = event.get("text", "").lower()
            if topic_lower in text:
                topic_messages.append(event)

        if not topic_messages:
            return {
                "topic": topic,
                "stages": [],
                "progression": "未讨论过此话题"
            }

        # Categorize by complexity indicators
        stages = []
        for event in topic_messages:
            text = event.get("text", "").lower()
            timestamp = event.get("timestamp", "")

            # Determine stage based on keywords
            if any(kw in text for kw in ["什么是", "介绍", "基础", "入门", "what is", "introduction", "basic"]):
                stage = "基础概念"
            elif any(kw in text for kw in ["如何", "怎么", "实现", "使用", "how to", "implement", "use"]):
                stage = "实践应用"
            elif any(kw in text for kw in ["优化", "性能", "最佳实践", "高级", "optimization", "performance", "advanced"]):
                stage = "深入优化"
            elif any(kw in text for kw in ["问题", "错误", "调试", "bug", "error", "debug"]):
                stage = "问题解决"
            else:
                stage = "一般讨论"

            stages.append({
                "stage": stage,
                "timestamp": timestamp,
                "text_preview": text[:100]
            })

        # Determine progression
        stage_sequence = [s["stage"] for s in stages]
        if "基础概念" in stage_sequence and "深入优化" in stage_sequence:
            progression = "从基础到深入"
        elif "基础概念" in stage_sequence:
            progression = "正在学习基础"
        elif "深入优化" in stage_sequence:
            progression = "已达到高级水平"
        else:
            progression = "持续探索中"

        return {
            "topic": topic,
            "total_discussions": len(topic_messages),
            "stages": stages,
            "progression": progression,
            "first_mentioned": stages[0]["timestamp"] if stages else None,
            "last_mentioned": stages[-1]["timestamp"] if stages else None
        }

    def find_unresolved_questions(self) -> List[Dict[str, Any]]:
        """
        Find questions that may not have been fully answered.

        Returns:
            List of potentially unresolved questions
        """
        unresolved = []

        # Look for user questions
        for i, event in enumerate(self.user_messages):
            text = event.get("text", "")

            # Check if it's a question
            is_question = any(marker in text for marker in ["?", "？", "吗", "呢", "如何", "怎么", "为什么"])

            if not is_question:
                continue

            # Check if there's a follow-up question soon after
            # (might indicate the answer wasn't satisfactory)
            has_followup = False
            if i + 1 < len(self.user_messages):
                next_event = self.user_messages[i + 1]
                next_text = next_event.get("text", "")

                # Check for follow-up indicators
                followup_indicators = [
                    "还是", "但是", "不过", "那", "另外",
                    "still", "but", "however", "also", "what about"
                ]

                if any(ind in next_text for ind in followup_indicators):
                    has_followup = True

            # Check for uncertainty indicators
            has_uncertainty = any(ind in text for ind in [
                "不确定", "不太清楚", "不明白", "confused", "unclear", "not sure"
            ])

            if has_followup or has_uncertainty:
                unresolved.append({
                    "question": text[:200],
                    "timestamp": event.get("timestamp", ""),
                    "session_id": event.get("session_id", ""),
                    "reason": "有后续问题" if has_followup else "表达不确定"
                })

        return unresolved[:10]  # Return top 10

    def generate_insights_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive insights report.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with comprehensive insights
        """
        # Filter events by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = []

        for event in self.events:
            timestamp = event.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if dt.replace(tzinfo=None) >= cutoff_date:
                    recent_events.append(event)
            except Exception:
                continue

        # Create analyzer for recent events
        recent_analyzer = PatternAnalyzer(recent_events)

        # Gather all insights
        frequent_topics = recent_analyzer.analyze_frequent_topics(top_n=5)
        activity_time = recent_analyzer.analyze_activity_time()
        unresolved = recent_analyzer.find_unresolved_questions()

        # Generate summary
        summary_parts = []

        if frequent_topics:
            top_topic = frequent_topics[0]
            summary_parts.append(f"最常讨论的话题是「{top_topic['topic']}」(出现 {top_topic['count']} 次)")

        if activity_time["peak_hours"]:
            hours_str = "、".join(f"{h}:00" for h in activity_time["peak_hours"])
            summary_parts.append(f"通常在 {hours_str} 最活跃")

        if activity_time["max_streak"] > 1:
            summary_parts.append(f"最长连续活跃 {activity_time['max_streak']} 天")

        summary = "。".join(summary_parts) + "。" if summary_parts else "暂无足够数据生成摘要。"

        return {
            "period": f"最近 {days} 天",
            "summary": summary,
            "frequent_topics": frequent_topics,
            "activity_patterns": activity_time,
            "unresolved_questions": unresolved,
            "total_events": len(recent_events),
            "total_sessions": len(set(e.get("session_id") for e in recent_events if e.get("session_id")))
        }


def format_insights_report(report: Dict[str, Any]) -> str:
    """
    Format insights report as readable text.

    Args:
        report: Insights report dict

    Returns:
        Formatted text report
    """
    lines = []

    lines.append(f"# 📊 {report['period']}洞察报告\n")
    lines.append(f"## 📝 摘要\n")
    lines.append(f"{report['summary']}\n")

    # Frequent topics
    if report["frequent_topics"]:
        lines.append(f"## 🔥 高频话题\n")
        for i, topic in enumerate(report["frequent_topics"], 1):
            lines.append(f"{i}. **{topic['topic']}** - {topic['count']} 次 ({topic['percentage']:.1f}%)")
            if topic["examples"]:
                lines.append(f"   示例: {topic['examples'][0][:80]}...")
        lines.append("")

    # Activity patterns
    if report["activity_patterns"]:
        patterns = report["activity_patterns"]
        lines.append(f"## ⏰ 活动模式\n")
        if patterns["peak_hours"]:
            hours_str = "、".join(f"{h}:00" for h in patterns["peak_hours"])
            lines.append(f"- 活跃时段: {hours_str}")
        if patterns["peak_weekdays"]:
            days_str = "、".join(patterns["peak_weekdays"])
            lines.append(f"- 活跃日期: {days_str}")
        lines.append(f"- 活跃天数: {patterns['total_days']} 天")
        lines.append(f"- 最长连续: {patterns['max_streak']} 天")
        lines.append(f"- 日均消息: {patterns['avg_events_per_day']:.1f} 条\n")

    # Unresolved questions
    if report["unresolved_questions"]:
        lines.append(f"## ❓ 可能未解决的问题\n")
        for i, q in enumerate(report["unresolved_questions"][:5], 1):
            lines.append(f"{i}. {q['question'][:100]}")
            lines.append(f"   原因: {q['reason']}\n")

    # Statistics
    lines.append(f"## 📈 统计数据\n")
    lines.append(f"- 总消息数: {report['total_events']}")
    lines.append(f"- 总会话数: {report['total_sessions']}")

    return "\n".join(lines)
