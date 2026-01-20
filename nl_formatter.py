#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural language response formatter.

Converts structured search results into natural language insights.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def format_timestamp(ts_str: str) -> str:
    """
    Format timestamp to human-readable format.

    Args:
        ts_str: ISO format timestamp string

    Returns:
        Human-readable timestamp (e.g., "1月19日 下午3:30")
    """
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

        # Chinese format
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute

        period = "上午" if hour < 12 else "下午"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12

        return f"{month}月{day}日 {period}{display_hour}:{minute:02d}"

    except Exception:
        return ts_str


def truncate_text(text: str, max_length: int = 150) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_search_results(
    query: str,
    results: List[Dict[str, Any]],
    source: str = "both"
) -> Dict[str, Any]:
    """
    Format search results into natural language response.

    Args:
        query: Original query
        results: List of search results
        source: Search source (sql/markdown/both)

    Returns:
        Formatted response with summary, insights, and suggestions
    """
    if not results:
        return {
            "summary": f"没有找到关于「{query}」的相关记录。",
            "insights": [
                "可能是查询词不够准确，试试换个说法？",
                "或者这个话题还没有讨论过。"
            ],
            "key_findings": [],
            "suggestions": [
                "尝试使用更通用的关键词",
                "查看最近的活动记录"
            ],
            "metadata": {
                "total_results": 0,
                "unique_sessions": 0,
                "source": source,
                "query": query
            }
        }

    # Generate summary
    count = len(results)
    first_result = results[0]

    # Extract session info
    session_id = first_result.get("session_id", "unknown")
    timestamp = first_result.get("timestamp", "")
    formatted_time = format_timestamp(timestamp)

    # Count unique sessions
    unique_sessions = len(set(r.get("session_id") for r in results if r.get("session_id")))

    # Generate summary text
    if unique_sessions == 1:
        summary = f"在 {formatted_time} 的对话中找到了 {count} 条相关记录。"
    else:
        summary = f"在 {unique_sessions} 次对话中找到了 {count} 条相关记录。"

    # Add topic context
    first_text = first_result.get("text", "")
    if first_text:
        preview = truncate_text(first_text, 100)
        summary += f"\n\n最相关的内容：「{preview}」"

    # Generate insights
    insights = []

    # Source insight
    sql_count = sum(1 for r in results if r.get("source") == "sql")
    md_count = sum(1 for r in results if r.get("source") == "markdown")

    if sql_count > 0 and md_count > 0:
        insights.append(f"结果来自 SQL 索引 ({sql_count} 条) 和 Markdown 索引 ({md_count} 条)")
    elif sql_count > 0:
        insights.append(f"结果来自最近 10,000 条记录 (SQL 索引)")
    elif md_count > 0:
        insights.append(f"结果来自完整历史记录 (Markdown 索引)")

    # Recency insight
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days_ago = (datetime.now() - dt.replace(tzinfo=None)).days

            if days_ago == 0:
                insights.append("这是今天的对话")
            elif days_ago == 1:
                insights.append("这是昨天的对话")
            elif days_ago <= 7:
                insights.append(f"这是 {days_ago} 天前的对话")
            elif days_ago <= 30:
                insights.append(f"这是约 {days_ago // 7} 周前的对话")
            else:
                insights.append(f"这是约 {days_ago // 30} 个月前的对话")
        except Exception:
            pass

    # Role distribution insight
    user_count = sum(1 for r in results if r.get("role") == "user")
    assistant_count = sum(1 for r in results if r.get("role") == "assistant")

    if user_count > 0 and assistant_count > 0:
        insights.append(f"包含 {user_count} 条用户消息和 {assistant_count} 条助手回复")

    # Generate key findings
    key_findings = []

    for i, result in enumerate(results[:3]):  # Top 3 results
        finding = {
            "rank": i + 1,
            "session": format_timestamp(result.get("timestamp", "")),
            "session_id": result.get("session_id", ""),
            "role": result.get("role", ""),
            "text": truncate_text(result.get("text", ""), 200),
            "score": round(result.get("score", 0), 2),
            "source": result.get("source", "unknown")
        }

        # Add context link
        if result.get("session_id"):
            item_index = result.get("item_index")
            if item_index is not None:
                finding["context_link"] = f"codemem://session/{result['session_id']}#item{item_index}"

        key_findings.append(finding)

    # Generate suggestions
    suggestions = []

    if count > 3:
        suggestions.append(f"查看完整的 {count} 条结果？")

    if unique_sessions > 1:
        suggestions.append(f"浏览这 {unique_sessions} 次相关对话？")

    # Check if contains code
    has_code = any(
        "```" in r.get("text", "") or
        "def " in r.get("text", "") or
        "class " in r.get("text", "") or
        "function " in r.get("text", "")
        for r in results[:5]
    )

    if has_code:
        suggestions.append("发现代码片段，需要查看完整上下文吗？")

    # Suggest export
    if session_id and session_id != "unknown":
        suggestions.append(f"导出这次对话？")

    # Suggest related searches
    if count < 3:
        suggestions.append("结果较少，试试更通用的关键词？")

    return {
        "summary": summary,
        "insights": insights,
        "key_findings": key_findings,
        "suggestions": suggestions,
        "metadata": {
            "total_results": count,
            "unique_sessions": unique_sessions,
            "source": source,
            "query": query
        }
    }


def format_activity_summary(
    activity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format activity summary into natural language.

    Args:
        activity_data: Raw activity data from database

    Returns:
        Formatted activity summary
    """
    days = activity_data.get("days", 7)
    sessions = activity_data.get("sessions", [])
    session_count = len(sessions)

    if session_count == 0:
        return {
            "summary": f"最近 {days} 天没有活动记录。",
            "insights": [],
            "key_findings": [],
            "suggestions": ["尝试查看更长时间范围的活动？"]
        }

    # Generate summary
    total_events = sum(s.get("event_count", 0) for s in sessions)
    summary = f"最近 {days} 天有 {session_count} 次对话，共 {total_events} 条消息。"

    # Most recent session
    if sessions:
        recent = sessions[0]
        recent_time = format_timestamp(recent.get("last_seen", ""))
        summary += f"\n\n最近一次活动：{recent_time}"

        # Add sample messages
        sample_messages = recent.get("sample_messages", [])
        if sample_messages:
            summary += f"\n讨论内容：「{sample_messages[0]}」"

    # Generate insights
    insights = []

    # Activity frequency
    avg_events_per_session = total_events / session_count if session_count > 0 else 0
    insights.append(f"平均每次对话 {avg_events_per_session:.1f} 条消息")

    # Platform distribution
    platforms = {}
    for session in sessions:
        platform = session.get("platforms", "unknown")
        platforms[platform] = platforms.get(platform, 0) + 1

    if len(platforms) > 1:
        platform_str = ", ".join(f"{p} ({c}次)" for p, c in platforms.items())
        insights.append(f"使用平台：{platform_str}")

    # Time span
    if sessions:
        first_seen = sessions[-1].get("first_seen", "")
        last_seen = sessions[0].get("last_seen", "")

        if first_seen and last_seen:
            try:
                first_dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                last_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                span_days = (last_dt - first_dt).days

                if span_days > 0:
                    insights.append(f"活动跨度 {span_days} 天")
            except Exception:
                pass

    # Generate key findings (top sessions)
    key_findings = []

    for i, session in enumerate(sessions[:5]):
        finding = {
            "rank": i + 1,
            "session_id": session.get("session_id", ""),
            "time": format_timestamp(session.get("last_seen", "")),
            "event_count": session.get("event_count", 0),
            "platform": session.get("platforms", "unknown"),
            "sample_messages": session.get("sample_messages", [])[:2]
        }
        key_findings.append(finding)

    # Generate suggestions
    suggestions = [
        "查看某次对话的详细内容？",
        "搜索特定话题的讨论？"
    ]

    if session_count > 5:
        suggestions.append(f"浏览全部 {session_count} 次对话？")

    return {
        "summary": summary,
        "insights": insights,
        "key_findings": key_findings,
        "suggestions": suggestions,
        "metadata": {
            "days": days,
            "session_count": session_count,
            "total_events": total_events
        }
    }


def format_error_response(error_msg: str, query: str) -> Dict[str, Any]:
    """
    Format error response.

    Args:
        error_msg: Error message
        query: Original query

    Returns:
        Formatted error response
    """
    return {
        "summary": f"查询「{query}」时出现错误。",
        "insights": [
            f"错误信息：{error_msg}"
        ],
        "key_findings": [],
        "suggestions": [
            "检查数据库连接是否正常",
            "尝试重新构建索引",
            "简化查询内容"
        ],
        "metadata": {
            "error": error_msg,
            "query": query
        }
    }


def format_context_response(
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format context retrieval response.

    Args:
        context_data: Context data with surrounding lines

    Returns:
        Formatted context response
    """
    session_id = context_data.get("session_id", "")
    item_index = context_data.get("item_index", 0)
    lines = context_data.get("lines", [])

    if not lines:
        return {
            "summary": "未找到上下文信息。",
            "insights": [],
            "key_findings": [],
            "suggestions": ["检查 session_id 和 item_index 是否正确"]
        }

    # Generate summary
    line_count = len(lines)
    summary = f"找到 {line_count} 行上下文。"

    # Format lines
    formatted_lines = []
    for line in lines:
        line_num = line.get("line_number", 0)
        text = line.get("text", "")
        formatted_lines.append(f"{line_num:3d} | {text}")

    context_text = "\n".join(formatted_lines)

    # Generate insights
    insights = [
        f"会话 ID: {session_id}",
        f"项目索引: {item_index}",
        f"共 {line_count} 行"
    ]

    return {
        "summary": summary,
        "insights": insights,
        "key_findings": [{
            "context": context_text,
            "session_id": session_id,
            "item_index": item_index
        }],
        "suggestions": [
            "需要更多上下文？",
            "导出完整对话？"
        ],
        "metadata": {
            "session_id": session_id,
            "item_index": item_index,
            "line_count": line_count
        }
    }
