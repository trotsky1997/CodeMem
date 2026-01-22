# 基于句子距离的 CTR 训练数据生成

## 核心思想

**对话历史中的句子顺序和距离，本身就是一种隐式的相关性信号。**

类似于：
- Word2Vec 的 Skip-gram（通过上下文窗口预测词语）
- BERT 的 Next Sentence Prediction（预测两个句子是否相邻）
- Session-based 推荐（利用会话内的序列关系）

## 距离度量

### 1. 时间距离

```python
def time_distance(msg1: Dict, msg2: Dict) -> float:
    """
    计算两条消息的时间距离（天数）
    """
    t1 = datetime.fromisoformat(msg1['timestamp'])
    t2 = datetime.fromisoformat(msg2['timestamp'])
    return abs((t2 - t1).total_seconds() / 86400)  # 转换为天数
```

### 2. 会话距离

```python
def session_distance(msg1: Dict, msg2: Dict) -> int:
    """
    计算两条消息的会话距离

    返回：
    - 0: 同一会话
    - 1: 相邻会话
    - 2+: 间隔 N 个会话
    """
    if msg1['session_id'] == msg2['session_id']:
        return 0

    # 计算会话之间的距离
    session_ids = get_all_session_ids_in_order()
    idx1 = session_ids.index(msg1['session_id'])
    idx2 = session_ids.index(msg2['session_id'])
    return abs(idx2 - idx1)
```

### 3. 消息序号距离

```python
def message_distance(msg1: Dict, msg2: Dict) -> int:
    """
    计算两条消息在全局历史中的序号距离

    例如：
    - msg1 是第 10 条消息
    - msg2 是第 15 条消息
    - 距离 = 5
    """
    return abs(msg2['global_index'] - msg1['global_index'])
```

### 4. 综合距离

```python
def calculate_relevance_score(msg1: Dict, msg2: Dict) -> float:
    """
    综合距离计算相关性分数（0-1）

    距离越近，相关性越高
    """
    # 1. 会话距离（最重要）
    session_dist = session_distance(msg1, msg2)
    if session_dist == 0:
        session_score = 1.0  # 同一会话，强相关
    elif session_dist == 1:
        session_score = 0.7  # 相邻会话，中度相关
    else:
        session_score = 0.3 / session_dist  # 距离越远，相关性越低

    # 2. 时间距离
    time_dist = time_distance(msg1, msg2)
    time_score = math.exp(-time_dist / 7)  # 7天半衰期

    # 3. 消息距离
    msg_dist = message_distance(msg1, msg2)
    msg_score = 1.0 / (1.0 + msg_dist / 10)  # 每10条消息衰减

    # 综合评分（加权平均）
    relevance = (
        session_score * 0.5 +  # 会话距离最重要
        time_score * 0.3 +     # 时间距离次之
        msg_score * 0.2        # 消息距离最后
    )

    return relevance
```

## 训练数据生成策略

### 方案 1：滑动窗口（推荐）⭐⭐⭐⭐⭐

```python
def generate_training_data_sliding_window(
    messages: List[Dict],
    window_size: int = 20
) -> List[Dict]:
    """
    滑动窗口生成训练数据

    思路：
    - 对于每条用户消息（作为查询）
    - 在其前后 window_size 范围内的消息作为正样本
    - 超出范围的消息作为负样本

    类似 Word2Vec 的 Skip-gram
    """
    training_data = []
    user_messages = [m for m in messages if m['role'] == 'user']

    for i, query_msg in enumerate(user_messages):
        query_text = query_msg['text']

        # 正样本：窗口内的会话
        for j in range(max(0, i - window_size), min(len(user_messages), i + window_size + 1)):
            if i == j:
                continue

            candidate_msg = user_messages[j]
            distance = abs(j - i)

            # 距离越近，标签越高
            label = 1.0 / (1.0 + distance / 5)  # 每5条消息衰减

            training_data.append({
                'query': query_text,
                'conversation': get_conversation_context(candidate_msg),
                'label': label,
                'distance': distance,
                'method': 'sliding_window'
            })

        # 负样本：窗口外的会话（采样）
        far_indices = [j for j in range(len(user_messages))
                      if abs(j - i) > window_size]

        if far_indices:
            # 随机采样一些负样本
            negative_samples = random.sample(
                far_indices,
                min(5, len(far_indices))
            )

            for j in negative_samples:
                candidate_msg = user_messages[j]

                training_data.append({
                    'query': query_text,
                    'conversation': get_conversation_context(candidate_msg),
                    'label': 0.0,  # 负样本
                    'distance': abs(j - i),
                    'method': 'sliding_window'
                })

    return training_data
```

### 方案 2：会话内 vs 会话外（简单）⭐⭐⭐⭐

```python
def generate_training_data_session_based(
    messages: List[Dict]
) -> List[Dict]:
    """
    基于会话的训练数据生成

    思路：
    - 同一会话内的消息 = 正样本（label=1.0）
    - 不同会话的消息 = 负样本（label=0.0）

    简单但有效
    """
    training_data = []

    # 按会话分组
    sessions = defaultdict(list)
    for msg in messages:
        sessions[msg['session_id']].append(msg)

    session_list = list(sessions.values())

    for session in session_list:
        user_messages = [m for m in session if m['role'] == 'user']

        for query_msg in user_messages:
            query_text = query_msg['text']

            # 正样本：同一会话内的其他消息
            for other_msg in session:
                if other_msg['text'] == query_text:
                    continue

                training_data.append({
                    'query': query_text,
                    'conversation': get_conversation_context(other_msg),
                    'label': 1.0,  # 同一会话，强相关
                    'method': 'session_based'
                })

            # 负样本：其他会话的消息（采样）
            other_sessions = [s for s in session_list if s != session]
            if other_sessions:
                negative_session = random.choice(other_sessions)
                negative_msg = random.choice(negative_session)

                training_data.append({
                    'query': query_text,
                    'conversation': get_conversation_context(negative_msg),
                    'label': 0.0,  # 不同会话，低相关
                    'method': 'session_based'
                })

    return training_data
```

### 方案 3：时间衰减（精细）⭐⭐⭐⭐⭐

```python
def generate_training_data_time_decay(
    messages: List[Dict]
) -> List[Dict]:
    """
    基于时间衰减的训练数据生成

    思路：
    - 对于每条用户消息（作为查询）
    - 计算它与所有其他消息的时间距离
    - 时间距离越近，标签越高（指数衰减）

    最精细的方法
    """
    training_data = []
    user_messages = [m for m in messages if m['role'] == 'user']

    for query_msg in user_messages:
        query_text = query_msg['text']
        query_time = datetime.fromisoformat(query_msg['timestamp'])

        for candidate_msg in messages:
            if candidate_msg['text'] == query_text:
                continue

            candidate_time = datetime.fromisoformat(candidate_msg['timestamp'])

            # 计算时间距离（天数）
            time_diff = abs((candidate_time - query_time).total_seconds() / 86400)

            # 时间衰减标签（7天半衰期）
            label = math.exp(-time_diff / 7)

            # 会话加成
            if query_msg['session_id'] == candidate_msg['session_id']:
                label = min(label * 2, 1.0)  # 同一会话加倍

            training_data.append({
                'query': query_text,
                'conversation': get_conversation_context(candidate_msg),
                'label': label,
                'time_diff': time_diff,
                'method': 'time_decay'
            })

    return training_data
```

### 方案 4：混合策略（最佳）⭐⭐⭐⭐⭐

```python
def generate_training_data_hybrid(
    messages: List[Dict]
) -> List[Dict]:
    """
    混合策略：综合考虑会话、时间、消息距离

    使用前面定义的 calculate_relevance_score()
    """
    training_data = []
    user_messages = [m for m in messages if m['role'] == 'user']

    for i, query_msg in enumerate(user_messages):
        query_text = query_msg['text']

        # 为每条消息计算相关性分数
        candidates = []
        for j, candidate_msg in enumerate(user_messages):
            if i == j:
                continue

            relevance = calculate_relevance_score(query_msg, candidate_msg)

            candidates.append({
                'message': candidate_msg,
                'relevance': relevance,
                'index': j
            })

        # 按相关性排序
        candidates.sort(key=lambda x: x['relevance'], reverse=True)

        # 采样策略：
        # - Top 10 高相关（正样本）
        # - Bottom 5 低相关（负样本）
        # - 中间随机采样 5 个（中等样本）

        for candidate in candidates[:10]:  # 高相关
            training_data.append({
                'query': query_text,
                'conversation': get_conversation_context(candidate['message']),
                'label': candidate['relevance'],
                'method': 'hybrid'
            })

        for candidate in candidates[-5:]:  # 低相关
            training_data.append({
                'query': query_text,
                'conversation': get_conversation_context(candidate['message']),
                'label': candidate['relevance'],
                'method': 'hybrid'
            })

        # 中间随机采样
        if len(candidates) > 15:
            middle_samples = random.sample(candidates[10:-5], min(5, len(candidates) - 15))
            for candidate in middle_samples:
                training_data.append({
                    'query': query_text,
                    'conversation': get_conversation_context(candidate['message']),
                    'label': candidate['relevance'],
                    'method': 'hybrid'
                })

    return training_data
```

## 对比：位置伪标签 vs 距离伪标签

| 维度 | 位置伪标签 | 距离伪标签 |
|-----|----------|----------|
| 数据来源 | BM25 搜索结果排序 | 对话历史原始顺序 |
| 假设 | BM25 排序合理 | 距离近=相关性高 |
| 优点 | 直接反映搜索质量 | 不依赖搜索算法 |
| 缺点 | 循环依赖 BM25 | 可能有噪声 |
| 数据量 | 取决于搜索次数 | 取决于对话数量 |
| 适用场景 | 有搜索历史 | 有对话历史 |

## 最佳实践：双重伪标签 ⭐⭐⭐⭐⭐

```python
def generate_dual_pseudo_labels(
    messages: List[Dict],
    search_history: List[Dict]
) -> List[Dict]:
    """
    双重伪标签：结合位置和距离

    1. 从对话距离生成基础训练数据（大量）
    2. 从搜索位置生成精细训练数据（少量但准确）
    3. 混合训练
    """
    training_data = []

    # 1. 距离伪标签（大量，覆盖广）
    distance_data = generate_training_data_hybrid(messages)
    for sample in distance_data:
        sample['source'] = 'distance'
        sample['weight'] = 0.5  # 基础权重
    training_data.extend(distance_data)

    # 2. 位置伪标签（少量，但准确）
    position_data = generate_pseudo_training_data(search_history)
    for sample in position_data:
        sample['source'] = 'position'
        sample['weight'] = 1.0  # 更高权重
    training_data.extend(position_data)

    return training_data
```

## 实验验证

### 假设验证

```python
async def validate_distance_hypothesis(messages: List[Dict]) -> Dict[str, float]:
    """
    验证"距离近=相关性高"的假设

    方法：
    1. 对于每次真实搜索
    2. 计算被选择结果的平均距离
    3. 计算未被选择结果的平均距离
    4. 对比两者

    预期：被选择的结果距离更近
    """
    selected_distances = []
    unselected_distances = []

    for search in search_history_with_feedback:
        query_msg = search['query_message']
        selected_result = search['selected_result']
        unselected_results = search['unselected_results']

        # 计算选择结果的距离
        selected_dist = message_distance(query_msg, selected_result)
        selected_distances.append(selected_dist)

        # 计算未选择结果的平均距离
        for result in unselected_results:
            unselected_dist = message_distance(query_msg, result)
            unselected_distances.append(unselected_dist)

    return {
        'avg_selected_distance': np.mean(selected_distances),
        'avg_unselected_distance': np.mean(unselected_distances),
        'hypothesis_valid': np.mean(selected_distances) < np.mean(unselected_distances)
    }
```

## 总结

### 核心优势

1. **自监督学习** - 不需要人工标注
2. **数据量大** - 所有对话历史都可以用
3. **通用性强** - 不依赖特定搜索算法
4. **符合直觉** - 距离近的内容确实更相关

### 推荐方案

**双重伪标签（距离 + 位置）**

```
训练数据 = 距离伪标签（70%）+ 位置伪标签（30%）

距离伪标签：
  ├─ 数据量大
  ├─ 覆盖广
  └─ 基础权重 0.5

位置伪标签：
  ├─ 数据量小
  ├─ 更准确
  └─ 更高权重 1.0

→ 混合训练 → 最佳效果
```

### 实现优先级

1. **方案 2（会话内 vs 会话外）** - 最简单，立即可用 ⭐⭐⭐⭐⭐
2. **方案 4（混合策略）** - 最精细，效果最好 ⭐⭐⭐⭐⭐
3. **双重伪标签** - 结合位置和距离 ⭐⭐⭐⭐⭐
