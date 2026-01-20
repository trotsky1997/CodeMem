#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query rewriting and suggestion module for Phase 3.

Features:
- Spelling correction
- Query expansion
- Query simplification
- Query suggestions
"""

import re
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher


# Common typos and corrections
COMMON_TYPOS = {
    # Chinese
    "异歩": "异步",
    "数据苦": "数据库",
    "优画": "优化",
    "索应": "索引",
    "错误": "错误",

    # English
    "asynch": "async",
    "databse": "database",
    "databas": "database",
    "optimze": "optimize",
    "optimiztion": "optimization",
    "performace": "performance",
    "performence": "performance",
    "seach": "search",
    "serach": "search",
    "querry": "query",
    "querys": "query",
    "eror": "error",
    "erro": "error",
    "exeption": "exception",
    "exceptoin": "exception",
}


# Common programming terms for spell checking
PROGRAMMING_TERMS = {
    "python", "javascript", "java", "c++", "go", "rust", "ruby",
    "async", "await", "asyncio", "coroutine", "concurrent",
    "database", "sql", "nosql", "mongodb", "redis", "postgresql",
    "performance", "optimization", "optimize",
    "index", "indexing", "search", "query",
    "error", "exception", "bug", "issue",
    "test", "testing", "unittest", "pytest",
    "api", "rest", "http", "https",
    "cache", "caching",
    "thread", "process", "multiprocessing",
}


def correct_spelling(query: str) -> tuple[str, bool]:
    """
    Correct common spelling mistakes in query.

    Args:
        query: Original query

    Returns:
        Tuple of (corrected_query, was_corrected)
    """
    corrected = query
    was_corrected = False

    # Check for common typos
    for typo, correction in COMMON_TYPOS.items():
        if typo in corrected:
            corrected = corrected.replace(typo, correction)
            was_corrected = True

    # Check for similar words (fuzzy matching)
    words = corrected.split()
    corrected_words = []

    for word in words:
        word_lower = word.lower()

        # Skip if already a known term
        if word_lower in PROGRAMMING_TERMS:
            corrected_words.append(word)
            continue

        # Find closest match
        best_match = None
        best_ratio = 0.0

        for term in PROGRAMMING_TERMS:
            ratio = SequenceMatcher(None, word_lower, term).ratio()
            if ratio > best_ratio and ratio > 0.8:  # 80% similarity threshold
                best_ratio = ratio
                best_match = term

        if best_match:
            corrected_words.append(best_match)
            was_corrected = True
        else:
            corrected_words.append(word)

    if was_corrected:
        corrected = " ".join(corrected_words)

    return corrected, was_corrected


def simplify_query(query: str) -> str:
    """
    Simplify query by removing redundant words.

    Args:
        query: Original query

    Returns:
        Simplified query
    """
    # Remove redundant question words
    redundant_patterns = [
        r'\b(请问|请|能不能|可以吗|怎么样|如何|怎样)\b',
        r'\b(please|can you|could you|would you)\b',
    ]

    simplified = query
    for pattern in redundant_patterns:
        simplified = re.sub(pattern, '', simplified, flags=re.IGNORECASE)

    # Remove extra whitespace
    simplified = re.sub(r'\s+', ' ', simplified).strip()

    return simplified


def expand_query(query: str) -> List[str]:
    """
    Generate query variations for better search coverage.

    Args:
        query: Original query

    Returns:
        List of query variations
    """
    variations = [query]

    # Add variations with common prefixes/suffixes
    prefixes = ["如何", "怎么", "什么是", "how to", "what is"]
    suffixes = ["问题", "错误", "优化", "实现", "issue", "error", "optimization"]

    # Remove existing prefixes/suffixes
    cleaned = query
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
            break

    # Add cleaned version if different
    if cleaned != query and len(cleaned) > 2:
        variations.append(cleaned)

    return variations


def suggest_queries(query: str, search_results: List[Dict[str, Any]]) -> List[str]:
    """
    Generate query suggestions based on current query and results.

    Args:
        query: Current query
        search_results: Current search results

    Returns:
        List of suggested queries
    """
    suggestions = []

    # If no results, suggest broader queries
    if not search_results:
        # Remove specific terms
        words = query.split()
        if len(words) > 2:
            # Suggest removing last word
            suggestions.append(" ".join(words[:-1]))
            # Suggest removing first word
            suggestions.append(" ".join(words[1:]))

        # Suggest related terms
        query_lower = query.lower()
        if "python" in query_lower:
            suggestions.append("Python 教程")
            suggestions.append("Python 示例")
        elif "异步" in query_lower or "async" in query_lower:
            suggestions.append("异步编程")
            suggestions.append("asyncio 使用")
        elif "数据库" in query_lower or "database" in query_lower:
            suggestions.append("数据库设计")
            suggestions.append("SQL 查询")

    # If many results, suggest narrower queries
    elif len(search_results) > 10:
        # Extract common terms from results
        term_counts: Dict[str, int] = {}
        for result in search_results[:5]:
            text = result.get("text", "").lower()
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if len(word) > 3:  # Skip short words
                    term_counts[word] = term_counts.get(word, 0) + 1

        # Suggest adding common terms
        common_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for term, count in common_terms:
            if term not in query.lower():
                suggestions.append(f"{query} {term}")

    # Remove duplicates and limit
    suggestions = list(dict.fromkeys(suggestions))[:5]

    return suggestions


def rewrite_query(query: str) -> Dict[str, Any]:
    """
    Comprehensive query rewriting.

    Args:
        query: Original query

    Returns:
        Dict with rewritten query and metadata
    """
    # Step 1: Spelling correction
    corrected, was_corrected = correct_spelling(query)

    # Step 2: Simplification
    simplified = simplify_query(corrected)

    # Step 3: Expansion
    variations = expand_query(simplified)

    return {
        "original": query,
        "corrected": corrected if was_corrected else None,
        "simplified": simplified if simplified != query else None,
        "variations": variations if len(variations) > 1 else [],
        "was_corrected": was_corrected,
        "recommended": simplified if simplified != query else corrected
    }


def generate_did_you_mean(query: str, search_results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Generate "Did you mean?" suggestion.

    Args:
        query: Original query
        search_results: Search results

    Returns:
        Suggested query or None
    """
    # If no results and spelling was corrected, suggest correction
    corrected, was_corrected = correct_spelling(query)

    if was_corrected and not search_results:
        return corrected

    # If few results, suggest simplified query
    if len(search_results) < 3:
        simplified = simplify_query(query)
        if simplified != query:
            return simplified

    return None


def analyze_query_quality(query: str) -> Dict[str, Any]:
    """
    Analyze query quality and provide feedback.

    Args:
        query: Query to analyze

    Returns:
        Dict with quality metrics and suggestions
    """
    issues = []
    suggestions = []

    # Check length
    if len(query) < 3:
        issues.append("查询太短")
        suggestions.append("尝试使用更具体的关键词")

    if len(query) > 100:
        issues.append("查询太长")
        suggestions.append("尝试简化查询")

    # Check for typos
    _, has_typos = correct_spelling(query)
    if has_typos:
        issues.append("可能包含拼写错误")
        suggestions.append("检查拼写")

    # Check for redundant words
    simplified = simplify_query(query)
    if len(simplified) < len(query) * 0.7:
        issues.append("包含冗余词汇")
        suggestions.append("简化查询")

    # Check for programming terms
    query_lower = query.lower()
    has_programming_term = any(term in query_lower for term in PROGRAMMING_TERMS)

    if not has_programming_term and len(query.split()) > 2:
        issues.append("缺少具体的技术关键词")
        suggestions.append("添加编程语言或技术名称")

    quality_score = 100 - (len(issues) * 20)
    quality_score = max(0, min(100, quality_score))

    return {
        "score": quality_score,
        "issues": issues,
        "suggestions": suggestions,
        "has_typos": has_typos,
        "is_too_short": len(query) < 3,
        "is_too_long": len(query) > 100,
    }
