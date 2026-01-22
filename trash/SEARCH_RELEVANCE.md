# Search Relevance Philosophy for CodeMem

## 概述

借鉴 Google PageRank 和电商搜索的核心理念：**优先提升搜索相关性**。

**核心原则：相关性 > 个性化**

## 设计哲学

### Google PageRank 的启示

PageRank 的核心思想：
```
PageRank(A) = (1-d) + d * Σ(PageRank(Ti) / C(Ti))
```

- **不只看内容匹配**（关键词）
- **更看"权威性"**（链接关系）
- **重要的页面排在前面**

### CodeMem 的类比

在对话历史中，什么是"重要的对话"？

| Google 网页 | CodeMem 对话 |
|------------|-------------|
| 被链接次数 | 被后续引用次数 |
| 页面质量 | 对话深度、技术密度 |
| 域名权威性 | 解决方案完整性 |
| 内容新鲜度 | 时间新近度 |

## ConversationRank 算法

### 1. 对话重要性评分

```python
def calculate_conversation_rank(session: Dict, all_sessions: List[Dict]) -> float:
    """
    计算对话的重要性分数（类似 PageRank）

    ConversationRank = 基础分 + 深度分 + 引用分 + 质量分 + 时间分
    """
    score = 0.0

    # 1. 基础分：对话长度（归一化到 0-1）
    message_count = len(session['messages'])
    base_score = min(message_count / 50, 1.0)  # 50条消息为满分
    score += base_score * 0.2

    # 2. 深度分：技术关键词密度
    tech_keywords = count_technical_keywords(session['messages'])
    depth_score = min(tech_keywords / 20, 1.0)  # 20个关键词为满分
    score += depth_score * 0.2

    # 3. 引用分：被后续对话引用的次数（类似 PageRank）
    reference_count = count_references_to_session(session['id'], all_sessions)
    reference_score = min(reference_count / 5, 1.0)  # 5次引用为满分
    score += reference_score * 0.3  # 最重要的因素

    # 4. 质量分：是否包含完整的问题-解决方案
    has_solution = detect_solution_pattern(session['messages'])
    quality_score = 1.0 if has_solution else 0.5
    score += quality_score * 0.2

    # 5. 时间分：新近度（指数衰减）
    days_ago = (datetime.now() - session['timestamp']).days
    recency_score = math.exp(-days_ago / 90)  # 90天半衰期
    score += recency_score * 0.1

    return score
```

### 2. 引用关系检测

```python
def count_references_to_session(session_id: str, all_sessions: List[Dict]) -> int:
    """
    检测有多少后续对话引用了这个会话

    引用信号：
    - "之前讨论过的..."
    - "上次提到的..."
    - "那段代码..."
    - 相似的技术关键词组合
    """
    reference_count = 0
    target_keywords = extract_key_phrases(session_id)

    for other_session in all_sessions:
        if other_session['timestamp'] <= session_id['timestamp']:
            continue  # 只看后续对话

        # 检测显式引用
        if has_explicit_reference(other_session, session_id):
            reference_count += 1
            continue

        # 检测隐式引用（关键词重叠）
        other_keywords = extract_key_phrases(other_session)
        overlap = calculate_keyword_overlap(target_keywords, other_keywords)
        if overlap > 0.6:  # 60% 关键词重叠
            reference_count += 0.5  # 隐式引用权重减半

    return reference_count
```

### 3. 技术关键词识别

```python
TECHNICAL_KEYWORDS = {
    # 编程语言
    'python', 'javascript', 'java', 'go', 'rust', 'typescript',

    # 技术概念
    'async', 'asyncio', '异步', '协程', 'concurrent', '并发',
    'database', '数据库', 'sql', 'nosql', 'redis', 'cache',
    'api', 'rest', 'graphql', 'http', 'websocket',
    'performance', '性能', 'optimization', '优化',
    'test', '测试', 'debug', '调试', 'error', '错误',

    # 技术栈
    'django', 'flask', 'fastapi', 'react', 'vue', 'node',
    'docker', 'kubernetes', 'aws', 'azure', 'gcp',

    # 设计模式
    'singleton', 'factory', 'observer', 'mvc', 'mvvm',
    'microservice', '微服务', 'architecture', '架构',
}

def count_technical_keywords(messages: List[Dict]) -> int:
    """统计技术关键词出现次数（去重）"""
    text = ' '.join([m['text'].lower() for m in messages])
    found_keywords = set()

    for keyword in TECHNICAL_KEYWORDS:
        if keyword in text:
            found_keywords.add(keyword)

    return len(found_keywords)
```

### 4. 解决方案模式检测

```python
def detect_solution_pattern(messages: List[Dict]) -> bool:
    """
    检测对话是否包含完整的问题-解决方案模式

    模式：
    1. 用户提问（包含问题关键词）
    2. 助手回答（包含代码或详细解释）
    3. 用户确认或后续问题（说明有帮助）
    """
    if len(messages) < 3:
        return False

    # 检测问题关键词
    problem_keywords = ['如何', '怎么', '为什么', '错误', 'error', 'how', 'why', 'issue']
    has_question = any(
        any(kw in msg['text'].lower() for kw in problem_keywords)
        for msg in messages if msg['role'] == 'user'
    )

    # 检测代码块或详细解释
    has_code = any('```' in msg['text'] for msg in messages if msg['role'] == 'assistant')
    has_detailed_answer = any(
        len(msg['text']) > 200
        for msg in messages if msg['role'] == 'assistant'
    )

    # 检测用户确认
    confirmation_keywords = ['谢谢', '明白', '懂了', 'thanks', 'got it', 'works']
    has_confirmation = any(
        any(kw in msg['text'].lower() for kw in confirmation_keywords)
        for msg in messages if msg['role'] == 'user'
    )

    return has_question and (has_code or has_detailed_answer) and has_confirmation
```

## 最终排序算法

### 综合评分公式

```python
def calculate_final_score(result: Dict, query: str, context: Dict) -> float:
    """
    最终排序分数 = BM25分 × ConversationRank × 时间衰减 × 上下文加成
    """
    # 1. BM25 基础相关性分数（0-1 归一化）
    bm25_score = result['bm25_score'] / max_bm25_score

    # 2. ConversationRank（0-1）
    conversation_rank = result['conversation_rank']

    # 3. 时间衰减（0-1）
    days_ago = (datetime.now() - result['timestamp']).days
    time_decay = math.exp(-days_ago / 90)  # 90天半衰期

    # 4. 上下文加成（1.0-2.0）
    context_boost = 1.0
    if context.get('current_session') == result['session_id']:
        context_boost = 2.0  # 当前会话加倍
    elif result['session_id'] in context.get('recent_sessions', []):
        context_boost = 1.5  # 最近会话加50%

    # 综合评分（乘法模型）
    final_score = bm25_score * conversation_rank * time_decay * context_boost

    return final_score
```

### 权重说明

| 因素 | 权重 | 说明 |
|-----|------|------|
| BM25 相关性 | 基础分 | 内容匹配度 |
| ConversationRank | 0-1 倍增 | 对话重要性 |
| 时间衰减 | 0-1 倍增 | 新近度 |
| 上下文加成 | 1-2 倍增 | 会话连续性 |

**为什么用乘法而不是加法？**
- 任何一个维度为 0，结果就应该排后面
- 例如：BM25=0（完全不相关）→ 即使 ConversationRank 很高也不应该返回

## 结果多样化

### 避免结果过于单一

```python
def diversify_results(results: List[Dict], top_n: int = 10) -> List[Dict]:
    """
    确保结果的多样性（类似 Google 不会只返回同一个网站的结果）

    策略：
    1. 最相关的 3 个（不管来源）
    2. 不同时间段各 1-2 个
    3. 不同话题各 1 个
    4. 填充剩余位置
    """
    diversified = []
    used_sessions = set()

    # 1. Top 3 最相关
    for result in results[:3]:
        diversified.append(result)
        used_sessions.add(result['session_id'])

    # 2. 时间多样性
    time_buckets = {
        'today': [],
        'this_week': [],
        'this_month': [],
        'older': []
    }

    for result in results[3:]:
        if result['session_id'] in used_sessions:
            continue

        days_ago = (datetime.now() - result['timestamp']).days
        if days_ago == 0:
            time_buckets['today'].append(result)
        elif days_ago <= 7:
            time_buckets['this_week'].append(result)
        elif days_ago <= 30:
            time_buckets['this_month'].append(result)
        else:
            time_buckets['older'].append(result)

    # 每个时间段取 1 个
    for bucket in ['today', 'this_week', 'this_month', 'older']:
        if time_buckets[bucket] and len(diversified) < top_n:
            result = time_buckets[bucket][0]
            diversified.append(result)
            used_sessions.add(result['session_id'])

    # 3. 话题多样性
    topic_buckets = defaultdict(list)
    for result in results:
        if result['session_id'] in used_sessions:
            continue
        topic = result.get('primary_topic', 'other')
        topic_buckets[topic].append(result)

    # 每个话题取 1 个
    for topic, topic_results in topic_buckets.items():
        if len(diversified) >= top_n:
            break
        if topic_results:
            result = topic_results[0]
            diversified.append(result)
            used_sessions.add(result['session_id'])

    # 4. 填充剩余
    for result in results:
        if len(diversified) >= top_n:
            break
        if result['session_id'] not in used_sessions:
            diversified.append(result)

    return diversified[:top_n]
```

## 查询理解增强

### 1. 查询扩展（已有）

```python
# Phase 3 已实现
- 同义词扩展：77+ 技术术语
- 拼写纠错：常见错误 + 模糊匹配
- 查询简化：移除冗余词汇
```

### 2. 查询意图识别（已有）

```python
# Phase 1 已实现
- 6 种意图类型
- 时间表达式解析
- 关键词提取
```

### 3. 新增：查询重要性判断

```python
def analyze_query_importance(query: str) -> Dict[str, Any]:
    """
    判断查询的重要性，决定搜索策略

    - 高重要性：深度搜索，返回更多结果
    - 低重要性：快速搜索，返回 top 结果
    """
    importance_score = 0.0

    # 1. 查询长度（长查询通常更具体）
    word_count = len(query.split())
    if word_count > 5:
        importance_score += 0.3

    # 2. 技术关键词数量
    tech_keyword_count = sum(1 for kw in TECHNICAL_KEYWORDS if kw in query.lower())
    importance_score += min(tech_keyword_count * 0.1, 0.3)

    # 3. 问题类型
    if any(kw in query.lower() for kw in ['如何', '为什么', 'how', 'why']):
        importance_score += 0.2  # 深度问题

    # 4. 是否包含代码片段
    if '```' in query or any(op in query for op in ['()', '[]', '{}']):
        importance_score += 0.2

    return {
        'importance_score': min(importance_score, 1.0),
        'search_depth': 'deep' if importance_score > 0.6 else 'normal'
    }
```

## 实现路线图

### Phase 6.1: ConversationRank (v1.1.0) ⭐⭐⭐⭐⭐

**目标**：实现对话重要性评分

- [ ] 实现 `calculate_conversation_rank()` 函数
- [ ] 实现引用关系检测
- [ ] 实现技术关键词统计
- [ ] 实现解决方案模式检测
- [ ] 预计算所有会话的 ConversationRank
- [ ] 存储到数据库（新增 `conversation_rank` 字段）

**预期效果**：
- 重要的技术讨论排在前面
- 被多次引用的对话权重更高
- 完整的问题-解决方案对话优先

### Phase 6.2: 综合排序 (v1.2.0) ⭐⭐⭐⭐⭐

**目标**：实现多维度综合排序

- [ ] 实现 `calculate_final_score()` 函数
- [ ] 集成 BM25 + ConversationRank + 时间衰减
- [ ] 添加上下文加成
- [ ] 替换当前的纯 BM25 排序

**预期效果**：
- 搜索结果更相关
- 重要对话优先展示
- 考虑时间新近度

### Phase 6.3: 结果多样化 (v1.3.0) ⭐⭐⭐⭐

**目标**：避免结果过于单一

- [ ] 实现 `diversify_results()` 函数
- [ ] 时间多样性（不同时间段）
- [ ] 话题多样性（不同技术领域）
- [ ] 应用到 `memory.query` 工具

**预期效果**：
- 结果覆盖不同时间段
- 结果覆盖不同技术话题
- 避免只返回最近的对话

### Phase 6.4: 查询优化 (v1.4.0) ⭐⭐⭐

**目标**：根据查询重要性调整搜索策略

- [ ] 实现 `analyze_query_importance()` 函数
- [ ] 高重要性查询：深度搜索
- [ ] 低重要性查询：快速搜索
- [ ] 动态调整返回结果数量

**预期效果**：
- 重要查询返回更多结果
- 简单查询快速响应
- 资源使用更高效

## 性能考虑

### 1. 预计算 ConversationRank

```python
# 不是每次查询都计算，而是：
# - 数据库构建时计算一次
# - 新增会话时增量更新
# - 定期重新计算（每天一次）

async def rebuild_conversation_ranks(db_path: str):
    """重建所有会话的 ConversationRank"""
    sessions = await load_all_sessions(db_path)

    for session in sessions:
        rank = calculate_conversation_rank(session, sessions)
        await update_session_rank(db_path, session['id'], rank)
```

### 2. 缓存策略

```python
# 缓存 ConversationRank 计算结果
conversation_rank_cache = TTLCache(maxsize=1000, ttl=86400)  # 24小时

# 缓存最终排序结果
search_result_cache = TTLCache(maxsize=100, ttl=3600)  # 1小时
```

### 3. 增量更新

```python
# 新增会话时，只更新相关的 ConversationRank
async def on_new_session(session: Dict):
    # 1. 计算新会话的 rank
    rank = calculate_conversation_rank(session, all_sessions)

    # 2. 检测是否引用了旧会话
    referenced_sessions = detect_references(session)

    # 3. 更新被引用会话的 rank（引用分增加）
    for ref_session_id in referenced_sessions:
        await increment_reference_count(ref_session_id)
```

## 对比：当前 vs 优化后

### 当前（v1.0.0-rc1）

```python
# 纯 BM25 排序
results = bm25.get_top_n(query, documents, n=10)
```

**问题**：
- 只看关键词匹配
- 不考虑对话重要性
- 不考虑引用关系
- 结果可能过于单一

### 优化后（v1.1.0+）

```python
# 1. BM25 初筛
candidates = bm25.get_top_n(query, documents, n=50)

# 2. 计算综合分数
for result in candidates:
    result['final_score'] = calculate_final_score(result, query, context)

# 3. 重新排序
candidates.sort(key=lambda x: x['final_score'], reverse=True)

# 4. 结果多样化
final_results = diversify_results(candidates, top_n=10)
```

**改进**：
- ✅ 考虑对话重要性（ConversationRank）
- ✅ 考虑引用关系（类似 PageRank）
- ✅ 考虑时间新近度
- ✅ 结果多样化

## 总结

### 核心改进

1. **ConversationRank**：类似 PageRank，通过引用关系判断对话重要性
2. **综合排序**：BM25 × ConversationRank × 时间衰减 × 上下文加成
3. **结果多样化**：避免结果过于单一
4. **查询优化**：根据查询重要性调整搜索策略

### 优先级

1. **Phase 6.1 (v1.1.0)** - ConversationRank ⭐⭐⭐⭐⭐
2. **Phase 6.2 (v1.2.0)** - 综合排序 ⭐⭐⭐⭐⭐
3. **Phase 6.3 (v1.3.0)** - 结果多样化 ⭐⭐⭐⭐
4. **Phase 6.4 (v1.4.0)** - 查询优化 ⭐⭐⭐

### 预期效果

- 搜索相关性提升 50%+
- 重要对话优先展示
- 结果更加多样化
- 用户体验显著改善

**最大价值**：从"关键词匹配"升级到"智能相关性排序"。
