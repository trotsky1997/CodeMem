# Search Relevance Philosophy for CodeMem

## 概述

借鉴 Google PageRank 和电商搜索的核心理念，优先提升搜索相关性，而非个性化。

**核心原则：相关性 > 个性化**

## 设计哲学

### Google PageRank 的启示

PageRank 的核心思想：
- 不只看内容匹配（关键词）
- 更看"权威性"/"重要性"（链接关系）
- 重要的页面应该排在前面

### CodeMem 的应用

在对话历史中，什么是"重要的对话"？
- 深入的技术讨论（对话长度）
- 被后续引用的对话（引用关系）
- 完整解决问题的对话（解决方案质量）
- 知识密度高的对话（技术关键词数量）
- 最新的信息（时间新近度）

## 核心理念

### 1. 个性化 (Personalization)

**电商做法：**
- 基于用户浏览历史推荐商品
- "猜你喜欢"
- 个性化排序

**CodeMem 应用：**
```python
# 个性化查询建议
def get_personalized_suggestions(user_patterns: Dict) -> List[str]:
    """基于用户模式生成个性化建议"""
    frequent_topics = user_patterns['frequent_topics'][:3]
    recent_queries = user_patterns['recent_queries'][:5]

    suggestions = []
    for topic in frequent_topics:
        suggestions.append(f"回顾 {topic} 的相关讨论")

    return suggestions
```

**实现优先级：** ⭐⭐⭐⭐⭐

### 2. 协同过滤 (Collaborative Filtering)

**电商做法：**
- "买了这个的人还买了..."
- "浏览了这个的人还浏览了..."

**CodeMem 应用：**
```python
# 协同模式发现
def find_collaborative_patterns(query: str, all_users_data: List) -> List[str]:
    """
    发现：搜索过类似问题的用户还搜索了什么

    注意：需要多用户数据支持，当前 CodeMem 是单用户系统
    可以改为：同一用户在相似上下文中还搜索了什么
    """
    similar_sessions = find_similar_query_sessions(query)
    related_queries = extract_related_queries(similar_sessions)
    return related_queries[:5]
```

**实现优先级：** ⭐⭐⭐ (需要适配单用户场景)

### 3. 内容相似推荐 (Content-Based Filtering)

**电商做法：**
- "相似商品推荐"
- 基于商品属性匹配

**CodeMem 应用：**
```python
# 相似对话推荐
def recommend_similar_conversations(current_query: str, history: List) -> List[Dict]:
    """基于内容相似度推荐相关对话"""
    # 使用 Phase 4.5 的 query clustering
    clusterer = PatternClusterer(history)
    clusters = clusterer.cluster_queries(similarity_threshold=0.6)

    # 找到当前查询所属的 cluster
    current_cluster = find_query_cluster(current_query, clusters)

    # 推荐同 cluster 的其他查询
    return current_cluster['queries'][:5]
```

**实现优先级：** ⭐⭐⭐⭐ (可以直接利用现有 clustering)

### 4. 实时排序优化 (Real-time Ranking)

**电商做法：**
- 综合相关性、销量、评分、价格等多维度排序
- 动态调整权重

**CodeMem 应用：**
```python
# 智能排序
def smart_ranking(results: List[Dict], context: Dict) -> List[Dict]:
    """
    多维度排序：
    - BM25 相关性分数 (40%)
    - 时间新近度 (20%)
    - 个人兴趣匹配度 (20%)
    - 会话上下文相关性 (20%)
    """
    for result in results:
        score = 0.0

        # 1. BM25 相关性
        score += result['bm25_score'] * 0.4

        # 2. 时间新近度 (越新越高)
        days_ago = (datetime.now() - result['timestamp']).days
        recency_score = 1.0 / (1.0 + days_ago / 30)  # 30天衰减
        score += recency_score * 0.2

        # 3. 个人兴趣 (是否属于常讨论话题)
        if result['topic'] in context['frequent_topics']:
            score += 1.0 * 0.2

        # 4. 会话上下文 (是否与当前会话相关)
        if result['session_id'] == context.get('current_session'):
            score += 1.0 * 0.2

        result['final_score'] = score

    return sorted(results, key=lambda x: x['final_score'], reverse=True)
```

**实现优先级：** ⭐⭐⭐⭐⭐

### 5. 查询补全 (Query Auto-completion)

**电商做法：**
- 搜索框自动补全
- 基于热门搜索 + 个人历史

**CodeMem 应用：**
```python
# 查询补全
def autocomplete_query(partial_query: str, user_history: List) -> List[str]:
    """
    基于用户历史的查询补全

    示例：
    输入: "Python as"
    输出:
    - "Python async 最佳实践" (你搜索过 3 次)
    - "Python asyncio 教程" (相关搜索)
    - "Python async vs threading" (热门搜索)
    """
    # 1. 从用户历史中匹配
    history_matches = [
        q for q in user_history
        if q.lower().startswith(partial_query.lower())
    ]

    # 2. 从 query clusters 中匹配
    cluster_matches = find_cluster_matches(partial_query)

    # 3. 合并去重，按频率排序
    suggestions = merge_and_rank(history_matches, cluster_matches)

    return suggestions[:5]
```

**实现优先级：** ⭐⭐⭐

### 6. 结果多样化 (Result Diversification)

**电商做法：**
- 不只展示最相关的，还展示不同品牌、价格段、类型
- 避免结果过于单一

**CodeMem 应用：**
```python
# 结果多样化
def diversify_results(results: List[Dict], top_n: int = 10) -> List[Dict]:
    """
    确保结果的多样性：
    - 不同时间段
    - 不同话题
    - 不同会话类型
    """
    diversified = []

    # 1. 最相关的 3 个
    diversified.extend(results[:3])

    # 2. 不同时间段各 1 个
    time_buckets = group_by_time_period(results[3:])
    for bucket in ['recent', 'last_week', 'last_month', 'older']:
        if time_buckets.get(bucket):
            diversified.append(time_buckets[bucket][0])

    # 3. 不同话题各 1 个
    topic_buckets = group_by_topic(results)
    for topic in get_diverse_topics(topic_buckets):
        if len(diversified) < top_n:
            diversified.append(topic_buckets[topic][0])

    return diversified[:top_n]
```

**实现优先级：** ⭐⭐⭐⭐

### 7. 主动推荐 (Proactive Recommendations)

**电商做法：**
- "您可能感兴趣的商品"
- 基于浏览但未购买的商品推荐

**CodeMem 应用：**
```python
# 主动洞察推荐
def generate_proactive_insights(user_patterns: Dict) -> List[str]:
    """
    主动发现并推荐：
    - 重复出现的问题 → 建议回顾解决方案
    - 学习路径 → 建议下一步学习内容
    - 未解决的问题 → 提醒跟进
    """
    insights = []

    # 1. 重复问题模式
    if user_patterns['problem_patterns']:
        top_problem = user_patterns['problem_patterns'][0]
        insights.append(
            f"💡 发现：你在 {top_problem['count']} 次对话中都提到了 "
            f"{top_problem['pattern']}，建议回顾相关解决方案"
        )

    # 2. 学习路径建议
    if user_patterns['knowledge_evolution']:
        current_stage = user_patterns['knowledge_evolution']['current_stage']
        if current_stage == '实践应用':
            insights.append(
                "📚 建议：你已经掌握了基础，可以尝试更高级的优化技巧"
            )

    # 3. 未解决问题提醒
    if user_patterns['unresolved_questions']:
        insights.append(
            f"⚠️ 提醒：你有 {len(user_patterns['unresolved_questions'])} "
            f"个问题可能还未完全解决"
        )

    return insights
```

**实现优先级：** ⭐⭐⭐⭐⭐

### 8. A/B 测试与优化 (A/B Testing)

**电商做法：**
- 测试不同排序算法
- 测试不同推荐策略
- 基于转化率优化

**CodeMem 应用：**
```python
# 查询策略 A/B 测试
class QueryStrategy:
    """不同的查询策略"""

    @staticmethod
    def strategy_a_pure_relevance(query: str) -> List[Dict]:
        """策略 A：纯相关性排序"""
        return bm25_search(query)

    @staticmethod
    def strategy_b_smart_ranking(query: str, context: Dict) -> List[Dict]:
        """策略 B：智能多维度排序"""
        results = bm25_search(query)
        return smart_ranking(results, context)

    @staticmethod
    def strategy_c_personalized(query: str, user_patterns: Dict) -> List[Dict]:
        """策略 C：个性化排序"""
        results = bm25_search(query)
        return personalized_ranking(results, user_patterns)

# 跟踪哪个策略效果更好
# 指标：用户是否点击了结果、是否进行了 follow-up 查询等
```

**实现优先级：** ⭐⭐ (需要用户反馈数据)

## 实现路线图

### Phase 6.1: 智能排序 (v1.1.0)
- [ ] 实现多维度排序算法
- [ ] 集成时间新近度
- [ ] 集成个人兴趣匹配
- [ ] 集成会话上下文

### Phase 6.2: 个性化推荐 (v1.2.0)
- [ ] 基于用户模式的查询建议
- [ ] 主动洞察推荐
- [ ] 相似对话推荐

### Phase 6.3: 查询增强 (v1.3.0)
- [ ] 查询自动补全
- [ ] 相关搜索建议
- [ ] 结果多样化

### Phase 6.4: 协同模式 (v1.4.0)
- [ ] 同一用户的协同模式发现
- [ ] "在相似上下文中你还搜索了..."

## 技术挑战

### 1. 单用户系统
- 电商的协同过滤需要多用户数据
- CodeMem 是单用户系统
- **解决方案**：改为"同一用户在相似上下文中的模式"

### 2. 反馈数据缺失
- 电商有点击率、转化率等反馈数据
- CodeMem 目前没有用户反馈机制
- **解决方案**：
  - 跟踪 follow-up 查询（说明结果不够好）
  - 跟踪 context 引用（说明结果有用）
  - 跟踪查询重复（说明没找到答案）

### 3. 冷启动问题
- 新用户没有历史数据
- **解决方案**：
  - 前期使用纯相关性排序
  - 积累一定数据后启用个性化

## 性能考虑

### 1. 实时计算 vs 预计算
- **实时**：个性化排序、上下文匹配
- **预计算**：用户模式分析、query clustering

### 2. 缓存策略
- 缓存用户模式分析结果（TTL: 1小时）
- 缓存 query clusters（TTL: 24小时）
- 缓存个性化建议（TTL: 30分钟）

### 3. 增量更新
- 不是每次查询都重新分析所有历史
- 增量更新用户模式
- 定期重建 clusters

## 总结

电商搜推的核心是**理解用户意图 + 个性化 + 持续优化**。CodeMem 已经有了很好的基础（Phase 1-5），现在可以：

1. **短期（v1.1.0）**：实现智能排序，立即提升搜索质量
2. **中期（v1.2.0-1.3.0）**：添加个性化推荐和查询增强
3. **长期（v1.4.0+）**：建立完整的反馈循环和持续优化机制

**最大价值**：从"被动搜索工具"变成"主动记忆助手"。
