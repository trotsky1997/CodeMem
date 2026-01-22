# CTR 预估模型设计 for CodeMem

## 概述

借鉴电商搜索的 CTR（Click-Through Rate）预估模型，预测用户会选择哪个搜索结果。

**核心思想**：不是简单的相关性排序，而是预测"用户最可能需要哪个结果"。

## 问题定义

### 电商场景

```
输入：用户 + 查询 + 商品
输出：用户点击该商品的概率 P(click | user, query, item)
目标：最大化点击率
```

### CodeMem 场景

```
输入：用户历史 + 查询 + 对话结果
输出：用户选择该结果的概率 P(select | history, query, conversation)
目标：最大化结果有用性
```

### 关键差异

| 维度 | 电商 | CodeMem |
|-----|------|---------|
| 用户行为 | 点击、购买 | Follow-up 查询、引用 |
| 数据量 | 海量用户 | 单用户 |
| 反馈信号 | 显式点击 | 隐式行为 |
| 实时性 | 毫秒级 | 秒级可接受 |

## 定义"点击"的等价行为

在 CodeMem 中，什么行为表示"这个结果有用"？

### 1. 显式引用（强信号）

```python
# 用户在 follow-up 查询中引用了某个结果
"第一个"  # 引用排名第1的结果
"那段代码"  # 引用包含代码的结果
"那次对话"  # 引用某个会话
```

**权重：1.0**（最强信号）

### 2. 话题延续（中等信号）

```python
# 用户在后续查询中继续讨论相关话题
查询1: "Python 异步编程"
结果: [关于 asyncio 的对话]
查询2: "asyncio.gather 怎么用"  # 话题延续

# 说明第一次搜索的结果有用
```

**权重：0.7**

### 3. 会话延续（弱信号）

```python
# 用户在同一会话中继续提问
# 说明当前上下文有价值
```

**权重：0.3**

### 4. 负反馈信号

```python
# 用户重复搜索相同或相似的查询
查询1: "Python 异步"
查询2: "Python 异步"  # 5分钟内重复

# 说明第一次搜索没找到答案
```

**权重：-0.5**

## 特征工程

### 1. 查询特征（Query Features）

```python
class QueryFeatures:
    """查询相关特征"""

    def extract(self, query: str) -> Dict[str, float]:
        return {
            # 基础特征
            'query_length': len(query),  # 查询长度
            'query_word_count': len(query.split()),  # 词数
            'has_code': 1.0 if '```' in query else 0.0,  # 是否包含代码

            # 查询类型
            'is_how_question': 1.0 if any(w in query.lower() for w in ['如何', 'how']) else 0.0,
            'is_why_question': 1.0 if any(w in query.lower() for w in ['为什么', 'why']) else 0.0,
            'is_what_question': 1.0 if any(w in query.lower() for w in ['什么', 'what']) else 0.0,

            # 技术深度
            'tech_keyword_count': self._count_tech_keywords(query),
            'has_version_number': 1.0 if re.search(r'\d+\.\d+', query) else 0.0,

            # 时间相关
            'has_time_expression': 1.0 if self._has_time_expr(query) else 0.0,
        }
```

### 2. 对话特征（Conversation Features）

```python
class ConversationFeatures:
    """对话结果相关特征"""

    def extract(self, conversation: Dict) -> Dict[str, float]:
        return {
            # 对话质量
            'message_count': len(conversation['messages']),  # 对话长度
            'avg_message_length': self._avg_length(conversation['messages']),
            'has_code_block': 1.0 if self._has_code(conversation) else 0.0,
            'code_block_count': self._count_code_blocks(conversation),

            # 对话结构
            'turn_count': self._count_turns(conversation),  # 轮次
            'has_solution': 1.0 if self._detect_solution(conversation) else 0.0,
            'has_confirmation': 1.0 if self._has_confirmation(conversation) else 0.0,

            # 技术密度
            'tech_keyword_density': self._tech_density(conversation),
            'unique_tech_keywords': self._unique_tech_keywords(conversation),

            # 时间特征
            'days_ago': (datetime.now() - conversation['timestamp']).days,
            'is_recent': 1.0 if self._is_recent(conversation, days=7) else 0.0,

            # ConversationRank（Phase 6.1）
            'conversation_rank': conversation.get('rank', 0.5),
        }
```

### 3. 匹配特征（Match Features）

```python
class MatchFeatures:
    """查询与对话的匹配特征"""

    def extract(self, query: str, conversation: Dict) -> Dict[str, float]:
        return {
            # 文本匹配
            'bm25_score': self._bm25_score(query, conversation),
            'exact_match_count': self._exact_matches(query, conversation),
            'fuzzy_match_score': self._fuzzy_match(query, conversation),

            # 关键词匹配
            'keyword_overlap': self._keyword_overlap(query, conversation),
            'tech_keyword_overlap': self._tech_keyword_overlap(query, conversation),

            # 语义匹配
            'query_in_title': 1.0 if query.lower() in conversation['first_message'].lower() else 0.0,
            'topic_match': 1.0 if self._topic_match(query, conversation) else 0.0,
        }
```

### 4. 用户历史特征（User History Features）

```python
class UserHistoryFeatures:
    """用户历史行为特征"""

    def extract(self, user_history: Dict, conversation: Dict) -> Dict[str, float]:
        return {
            # 话题偏好
            'user_topic_preference': self._topic_preference(user_history, conversation),
            'topic_frequency': self._topic_frequency(user_history, conversation['topic']),

            # 时间偏好
            'user_prefers_recent': self._prefers_recent(user_history),
            'user_avg_result_age': self._avg_result_age(user_history),

            # 对话类型偏好
            'prefers_long_conversations': self._prefers_long(user_history),
            'prefers_code_heavy': self._prefers_code(user_history),

            # 引用历史
            'has_referenced_before': 1.0 if self._has_referenced(user_history, conversation) else 0.0,
            'reference_count': self._reference_count(user_history, conversation),
        }
```

### 5. 上下文特征（Context Features）

```python
class ContextFeatures:
    """当前会话上下文特征"""

    def extract(self, context: Dict, conversation: Dict) -> Dict[str, float]:
        return {
            # 会话连续性
            'is_same_session': 1.0 if context.get('current_session') == conversation['session_id'] else 0.0,
            'is_recent_session': 1.0 if conversation['session_id'] in context.get('recent_sessions', []) else 0.0,

            # 话题连续性
            'topic_continuity': self._topic_continuity(context, conversation),
            'keyword_continuity': self._keyword_continuity(context, conversation),

            # 查询历史
            'query_similarity_to_last': self._query_similarity(context, conversation),
            'is_follow_up': 1.0 if context.get('is_follow_up') else 0.0,
        }
```

### 6. 位置特征（Position Features）

```python
class PositionFeatures:
    """结果位置特征（重要！）"""

    def extract(self, position: int) -> Dict[str, float]:
        return {
            # 位置偏差（用户倾向于点击前面的结果）
            'position': position,  # 1, 2, 3, ...
            'position_bias': 1.0 / math.log2(position + 1),  # DCG 风格的位置偏差

            # 位置分组
            'is_top_1': 1.0 if position == 1 else 0.0,
            'is_top_3': 1.0 if position <= 3 else 0.0,
            'is_top_5': 1.0 if position <= 5 else 0.0,
        }
```

## CTR 预估模型

### 方案 1：Logistic Regression（简单快速）⭐⭐⭐⭐⭐

```python
import numpy as np
from typing import List, Dict

class LogisticRegressionCTR:
    """
    逻辑回归 CTR 预估模型

    优点：
    - 简单、快速、可解释
    - 适合小数据量
    - 特征权重清晰

    缺点：
    - 无法捕捉特征交叉
    """

    def __init__(self):
        self.weights = None
        self.feature_names = []

    def extract_features(self, query: str, conversation: Dict,
                        user_history: Dict, context: Dict, position: int) -> np.ndarray:
        """提取所有特征"""
        features = {}

        # 1. 查询特征
        features.update(QueryFeatures().extract(query))

        # 2. 对话特征
        features.update(ConversationFeatures().extract(conversation))

        # 3. 匹配特征
        features.update(MatchFeatures().extract(query, conversation))

        # 4. 用户历史特征
        features.update(UserHistoryFeatures().extract(user_history, conversation))

        # 5. 上下文特征
        features.update(ContextFeatures().extract(context, conversation))

        # 6. 位置特征
        features.update(PositionFeatures().extract(position))

        # 转换为向量
        self.feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in self.feature_names])

        return feature_vector

    def predict(self, features: np.ndarray) -> float:
        """预测 CTR"""
        if self.weights is None:
            # 初始权重（启发式）
            self.weights = self._initialize_weights()

        # Logistic function: P = 1 / (1 + exp(-w·x))
        logit = np.dot(self.weights, features)
        ctr = 1.0 / (1.0 + np.exp(-logit))

        return ctr

    def _initialize_weights(self) -> np.ndarray:
        """
        初始化权重（基于领域知识）

        在没有训练数据前，使用启发式权重
        """
        weights = {}

        # 匹配特征（最重要）
        weights['bm25_score'] = 2.0
        weights['keyword_overlap'] = 1.5
        weights['tech_keyword_overlap'] = 1.5

        # ConversationRank
        weights['conversation_rank'] = 1.5

        # 对话质量
        weights['has_solution'] = 1.0
        weights['has_code_block'] = 0.8
        weights['message_count'] = 0.3

        # 时间特征
        weights['is_recent'] = 0.5
        weights['days_ago'] = -0.01  # 负权重：越久越不相关

        # 上下文
        weights['is_same_session'] = 1.2
        weights['topic_continuity'] = 0.8
        weights['is_follow_up'] = 0.5

        # 位置偏差
        weights['position_bias'] = 0.5
        weights['is_top_1'] = 0.3

        # 用户偏好
        weights['user_topic_preference'] = 0.8
        weights['has_referenced_before'] = 1.0

        # 默认权重
        for name in self.feature_names:
            if name not in weights:
                weights[name] = 0.1

        return np.array([weights.get(name, 0.1) for name in self.feature_names])

    def train(self, training_data: List[Dict]):
        """
        训练模型（梯度下降）

        training_data: [
            {
                'query': '...',
                'conversation': {...},
                'user_history': {...},
                'context': {...},
                'position': 1,
                'label': 1.0  # 1=选择了这个结果, 0=没选择
            },
            ...
        ]
        """
        if not training_data:
            return

        # 提取特征和标签
        X = []
        y = []
        for sample in training_data:
            features = self.extract_features(
                sample['query'],
                sample['conversation'],
                sample['user_history'],
                sample['context'],
                sample['position']
            )
            X.append(features)
            y.append(sample['label'])

        X = np.array(X)
        y = np.array(y)

        # 梯度下降
        learning_rate = 0.01
        epochs = 100

        if self.weights is None:
            self.weights = self._initialize_weights()

        for epoch in range(epochs):
            # 预测
            predictions = 1.0 / (1.0 + np.exp(-np.dot(X, self.weights)))

            # 计算梯度
            gradient = np.dot(X.T, (predictions - y)) / len(y)

            # 更新权重
            self.weights -= learning_rate * gradient

            # 计算损失（可选）
            if epoch % 10 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-10) +
                               (1 - y) * np.log(1 - predictions + 1e-10))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### 方案 2：Factorization Machines（特征交叉）⭐⭐⭐⭐

```python
class FactorizationMachineCTR:
    """
    因子分解机 CTR 预估模型

    优点：
    - 自动学习特征交叉
    - 适合稀疏特征
    - 比 LR 更强大

    缺点：
    - 计算复杂度更高
    - 需要更多训练数据
    """

    def __init__(self, n_factors: int = 10):
        self.n_factors = n_factors
        self.w0 = 0.0  # 全局偏置
        self.w = None  # 一阶权重
        self.V = None  # 二阶交叉矩阵

    def predict(self, features: np.ndarray) -> float:
        """
        FM 预测公式：
        y = w0 + Σ(wi·xi) + Σ(Σ(<vi,vj>·xi·xj))
        """
        if self.w is None:
            self._initialize(len(features))

        # 一阶项
        linear = self.w0 + np.dot(self.w, features)

        # 二阶交叉项（优化计算）
        interaction = 0.0
        for f in range(self.n_factors):
            sum_square = np.sum(self.V[:, f] * features) ** 2
            square_sum = np.sum((self.V[:, f] ** 2) * (features ** 2))
            interaction += sum_square - square_sum

        interaction *= 0.5

        # Sigmoid
        logit = linear + interaction
        ctr = 1.0 / (1.0 + np.exp(-logit))

        return ctr

    def _initialize(self, n_features: int):
        """初始化参数"""
        self.w = np.random.randn(n_features) * 0.01
        self.V = np.random.randn(n_features, self.n_factors) * 0.01
```

### 方案 3：轻量级神经网络（深度学习）⭐⭐⭐

```python
class DeepCTR:
    """
    轻量级深度 CTR 模型（类似 Wide & Deep）

    优点：
    - 强大的非线性拟合能力
    - 可以学习复杂模式

    缺点：
    - 需要大量训练数据
    - 计算开销大
    - 可解释性差

    注意：CodeMem 数据量有限，不推荐使用
    """
    pass  # 暂不实现
```

## 收集训练数据

### 1. 隐式反馈收集

```python
class FeedbackCollector:
    """收集用户隐式反馈"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.feedback_log = []

    async def log_search(self, query: str, results: List[Dict], context: Dict):
        """记录搜索行为"""
        search_id = str(uuid.uuid4())

        await self._save_search_log({
            'search_id': search_id,
            'query': query,
            'results': [r['session_id'] for r in results],
            'timestamp': datetime.now().isoformat(),
            'context': context
        })

        return search_id

    async def log_selection(self, search_id: str, selected_position: int,
                           selection_type: str, confidence: float):
        """
        记录用户选择行为

        selection_type:
        - 'explicit_reference': 显式引用（"第一个"）
        - 'topic_continuation': 话题延续
        - 'session_continuation': 会话延续
        """
        await self._save_feedback({
            'search_id': search_id,
            'selected_position': selected_position,
            'selection_type': selection_type,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

    async def log_negative_feedback(self, search_id: str, reason: str):
        """
        记录负反馈

        reason:
        - 'repeated_query': 重复查询
        - 'no_follow_up': 没有后续行为
        """
        await self._save_feedback({
            'search_id': search_id,
            'selected_position': -1,
            'selection_type': 'negative',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
```

### 2. 自动标注训练数据

```python
async def generate_training_data(db_path: str, days: int = 30) -> List[Dict]:
    """
    从历史搜索日志中生成训练数据

    标注规则：
    1. 如果用户引用了某个结果 → label=1.0
    2. 如果用户继续讨论相关话题 → label=0.7
    3. 如果用户在同一会话继续 → label=0.3
    4. 如果用户重复搜索 → 所有结果 label=0.0
    5. 其他未选择的结果 → label=0.0
    """
    search_logs = await load_search_logs(db_path, days)
    feedback_logs = await load_feedback_logs(db_path, days)

    training_data = []

    for search in search_logs:
        # 查找对应的反馈
        feedback = find_feedback(search['search_id'], feedback_logs)

        for position, result in enumerate(search['results'], 1):
            label = 0.0

            if feedback:
                if feedback['selected_position'] == position:
                    # 用户选择了这个结果
                    if feedback['selection_type'] == 'explicit_reference':
                        label = 1.0
                    elif feedback['selection_type'] == 'topic_continuation':
                        label = 0.7
                    elif feedback['selection_type'] == 'session_continuation':
                        label = 0.3

            training_data.append({
                'query': search['query'],
                'conversation': result,
                'user_history': search['context'].get('user_history', {}),
                'context': search['context'],
                'position': position,
                'label': label
            })

    return training_data
```

## 在线学习（Online Learning）

```python
class OnlineCTRModel:
    """
    在线学习 CTR 模型

    特点：
    - 实时更新模型
    - 适应用户行为变化
    - 无需离线训练
    """

    def __init__(self):
        self.model = LogisticRegressionCTR()
        self.feedback_buffer = []
        self.update_threshold = 10  # 累积10个样本后更新

    async def predict_and_rank(self, query: str, candidates: List[Dict],
                               user_history: Dict, context: Dict) -> List[Dict]:
        """预测 CTR 并重新排序"""
        for position, candidate in enumerate(candidates, 1):
            features = self.model.extract_features(
                query, candidate, user_history, context, position
            )
            candidate['predicted_ctr'] = self.model.predict(features)

        # 按预测 CTR 排序
        candidates.sort(key=lambda x: x['predicted_ctr'], reverse=True)

        return candidates

    async def update_with_feedback(self, feedback: Dict):
        """根据反馈更新模型"""
        self.feedback_buffer.append(feedback)

        # 累积足够样本后更新
        if len(self.feedback_buffer) >= self.update_threshold:
            self.model.train(self.feedback_buffer)
            self.feedback_buffer = []  # 清空缓冲区
```

## 评估指标

### 1. 离线指标

```python
def evaluate_offline(model, test_data: List[Dict]) -> Dict[str, float]:
    """离线评估"""
    predictions = []
    labels = []

    for sample in test_data:
        features = model.extract_features(
            sample['query'],
            sample['conversation'],
            sample['user_history'],
            sample['context'],
            sample['position']
        )
        pred = model.predict(features)
        predictions.append(pred)
        labels.append(sample['label'])

    return {
        'auc': calculate_auc(labels, predictions),  # AUC
        'logloss': calculate_logloss(labels, predictions),  # Log Loss
        'accuracy': calculate_accuracy(labels, predictions),  # 准确率
    }
```

### 2. 在线指标

```python
def evaluate_online(search_logs: List[Dict]) -> Dict[str, float]:
    """在线评估（更重要）"""
    return {
        # 点击率（选择率）
        'ctr': calculate_ctr(search_logs),

        # 平均选择位置（越小越好）
        'mean_selected_position': calculate_mean_position(search_logs),

        # Top-3 命中率
        'hit_rate_at_3': calculate_hit_rate(search_logs, k=3),

        # MRR (Mean Reciprocal Rank)
        'mrr': calculate_mrr(search_logs),

        # NDCG (Normalized Discounted Cumulative Gain)
        'ndcg': calculate_ndcg(search_logs),
    }
```

## 实现路线图

### Phase 6.1: 特征工程 (v1.1.0) ⭐⭐⭐⭐⭐

- [ ] 实现 6 类特征提取器
- [ ] 实现反馈收集系统
- [ ] 实现训练数据自动标注

### Phase 6.2: LR 模型 (v1.2.0) ⭐⭐⭐⭐⭐

- [ ] 实现 Logistic Regression CTR 模型
- [ ] 启发式权重初始化
- [ ] 集成到 `memory.query` 工具

### Phase 6.3: 在线学习 (v1.3.0) ⭐⭐⭐⭐

- [ ] 实现在线学习机制
- [ ] 实时模型更新
- [ ] A/B 测试框架

### Phase 6.4: 高级模型 (v1.4.0) ⭐⭐⭐

- [ ] 实现 FM 模型（可选）
- [ ] 特征交叉优化
- [ ] 模型集成

## 冷启动策略：从位置反推 CTR ⭐⭐⭐⭐⭐

### 核心思想

**问题**：没有真实用户反馈，无法训练模型。

**解决方案**：从 BM25 的原始排序反推 CTR，作为初始训练数据。

**假设**：BM25 排序有一定合理性，排在前面的结果更可能被用户选择。

### 位置偏差模型

```python
def position_to_pseudo_ctr(position: int) -> float:
    """
    从位置反推伪 CTR

    基于经验公式（类似 Google 的点击模型）：
    CTR(position) = 1 / log2(position + 1)

    位置 1: 1.0 / log2(2) = 1.0
    位置 2: 1.0 / log2(3) = 0.63
    位置 3: 1.0 / log2(4) = 0.50
    位置 4: 1.0 / log2(5) = 0.43
    位置 5: 1.0 / log2(6) = 0.39
    ...
    """
    return 1.0 / math.log2(position + 1)


def generate_pseudo_training_data(search_history: List[Dict]) -> List[Dict]:
    """
    从历史搜索结果生成伪训练数据

    输入：历史上所有的搜索记录（带 BM25 排序）
    输出：带伪标签的训练数据
    """
    training_data = []

    for search in search_history:
        query = search['query']
        results = search['results']  # BM25 排序的结果
        context = search['context']

        for position, result in enumerate(results, 1):
            # 从位置反推伪 CTR
            pseudo_ctr = position_to_pseudo_ctr(position)

            # 转换为二分类标签（可选）
            # 方案1：直接用 pseudo_ctr 作为软标签
            label = pseudo_ctr

            # 方案2：转换为硬标签（0/1）
            # label = 1.0 if position <= 3 else 0.0

            training_data.append({
                'query': query,
                'conversation': result,
                'user_history': context.get('user_history', {}),
                'context': context,
                'position': position,
                'label': label,
                'is_pseudo': True  # 标记为伪标签
            })

    return training_data
```

### 优点与缺点

#### ✅ 优点

1. **立即可用** - 不需要等待真实反馈
2. **数据量大** - 历史上所有搜索都可以用
3. **冷启动友好** - 新用户也能有初始模型
4. **快速迭代** - 可以立即开始训练和测试

#### ⚠️ 缺点

1. **假设 BM25 是对的** - 但实际上 BM25 可能不准确
2. **循环依赖** - 用 BM25 训练模型，模型学到的就是 BM25 的模式
3. **位置偏差** - 用户倾向点击前面的结果，不代表前面的结果就一定更好

### 解决方案：混合训练策略 ⭐⭐⭐⭐⭐

```python
class HybridTrainingStrategy:
    """
    混合训练策略：伪标签 + 真实反馈

    阶段1：用伪标签训练初始模型
    阶段2：部署模型，收集真实反馈
    阶段3：用真实反馈逐步替换伪标签
    阶段4：持续在线学习
    """

    def __init__(self):
        self.pseudo_data = []  # 伪标签数据
        self.real_data = []    # 真实反馈数据
        self.model = LogisticRegressionCTR()

    async def initialize(self, search_history: List[Dict]):
        """阶段1：用伪标签训练初始模型"""
        print("🔄 生成伪训练数据...")
        self.pseudo_data = generate_pseudo_training_data(search_history)

        print(f"✅ 生成 {len(self.pseudo_data)} 条伪训练数据")
        print("🔄 训练初始模型...")

        self.model.train(self.pseudo_data)
        print("✅ 初始模型训练完成")

    async def collect_real_feedback(self, feedback: Dict):
        """阶段2：收集真实反馈"""
        self.real_data.append(feedback)

        # 真实数据达到一定量后，开始混合训练
        if len(self.real_data) >= 10:
            await self.hybrid_train()

    async def hybrid_train(self):
        """阶段3：混合训练（伪标签 + 真实反馈）"""

        # 计算真实数据的权重（随着真实数据增多，权重增加）
        real_data_ratio = len(self.real_data) / (len(self.real_data) + len(self.pseudo_data))
        real_weight = min(real_data_ratio * 2, 1.0)  # 最多到1.0
        pseudo_weight = 1.0 - real_weight

        print(f"🔄 混合训练：真实数据权重={real_weight:.2f}, 伪数据权重={pseudo_weight:.2f}")

        # 加权混合训练数据
        training_data = []

        # 添加真实数据（高权重）
        for sample in self.real_data:
            sample['weight'] = real_weight
            training_data.append(sample)

        # 添加伪数据（低权重，且随着真实数据增多而降低）
        sample_size = min(len(self.pseudo_data), len(self.real_data) * 5)  # 最多5倍
        sampled_pseudo = random.sample(self.pseudo_data, sample_size)

        for sample in sampled_pseudo:
            sample['weight'] = pseudo_weight
            training_data.append(sample)

        # 训练模型
        self.model.train_weighted(training_data)

        print(f"✅ 混合训练完成：{len(self.real_data)} 真实 + {sample_size} 伪标签")

    async def phase_out_pseudo_data(self):
        """阶段4：逐步淘汰伪数据"""

        # 当真实数据足够多时（例如 > 100），完全停用伪数据
        if len(self.real_data) > 100:
            print("✅ 真实数据充足，停用伪标签数据")
            self.pseudo_data = []
            self.model.train(self.real_data)
```

### 去偏技术（高级）

```python
def inverse_propensity_scoring(position: int) -> float:
    """
    逆倾向得分（Inverse Propensity Scoring）

    用于去除位置偏差：
    - 位置靠前的结果，即使质量一般，也容易被点击
    - 位置靠后的结果，即使质量很好，也不容易被点击

    解决方案：给位置靠后的样本更高的权重
    """
    # 位置偏差（用户点击该位置的倾向）
    propensity = 1.0 / math.log2(position + 1)

    # 逆倾向得分（位置越靠后，权重越高）
    ips_weight = 1.0 / propensity

    return ips_weight


def generate_unbiased_training_data(search_history: List[Dict]) -> List[Dict]:
    """
    生成去偏的训练数据

    对位置靠后但质量高的结果给予更高权重
    """
    training_data = []

    for search in search_history:
        for position, result in enumerate(search['results'], 1):
            pseudo_ctr = position_to_pseudo_ctr(position)

            # 计算 IPS 权重
            ips_weight = inverse_propensity_scoring(position)

            training_data.append({
                'query': search['query'],
                'conversation': result,
                'user_history': search['context'].get('user_history', {}),
                'context': search['context'],
                'position': position,
                'label': pseudo_ctr,
                'weight': ips_weight,  # 去偏权重
                'is_pseudo': True
            })

    return training_data
```

### 实际应用策略

```python
class ColdStartStrategy:
    """冷启动策略（改进版）"""

    def get_model(self, user_history: Dict, search_history: List[Dict]):
        """根据用户数据量选择策略"""

        real_feedback_count = len(user_history.get('feedbacks', []))

        if real_feedback_count == 0:
            # 阶段1：纯伪标签训练
            print("📊 阶段1：使用伪标签训练初始模型")
            model = LogisticRegressionCTR()
            pseudo_data = generate_pseudo_training_data(search_history)
            model.train(pseudo_data)
            return model

        elif real_feedback_count < 50:
            # 阶段2：混合训练（伪标签 + 真实反馈）
            print(f"📊 阶段2：混合训练 ({real_feedback_count} 真实反馈)")
            strategy = HybridTrainingStrategy()
            strategy.initialize(search_history)
            strategy.collect_real_feedback(user_history['feedbacks'])
            return strategy.model

        else:
            # 阶段3：纯真实反馈训练
            print(f"📊 阶段3：纯真实反馈训练 ({real_feedback_count} 条)")
            model = LogisticRegressionCTR()
            model.train(user_history['feedbacks'])
            return model
```

### 评估：伪标签 vs 真实标签

```python
async def evaluate_pseudo_labels(search_history: List[Dict],
                                 real_feedback: List[Dict]) -> Dict[str, float]:
    """
    评估伪标签的质量

    对比：
    - 伪标签预测的 CTR
    - 真实用户行为的 CTR
    """
    results = {
        'pseudo_accuracy': 0.0,
        'position_correlation': 0.0,
        'top3_agreement': 0.0
    }

    # 1. 准确率：伪标签预测的 top-3 和真实 top-3 的重叠度
    pseudo_top3 = get_pseudo_top3(search_history)
    real_top3 = get_real_top3(real_feedback)
    results['top3_agreement'] = calculate_overlap(pseudo_top3, real_top3)

    # 2. 相关性：位置和真实 CTR 的相关性
    results['position_correlation'] = calculate_correlation(
        [f['position'] for f in real_feedback],
        [f['label'] for f in real_feedback]
    )

    return results
```

### 总结：推荐方案

**最佳实践：混合训练策略**

```
阶段1（0 真实反馈）：
  ├─ 从历史搜索生成伪标签
  ├─ 训练初始 LR 模型
  └─ 立即可用

阶段2（1-50 真实反馈）：
  ├─ 收集真实用户反馈
  ├─ 混合训练（真实权重逐步增加）
  └─ 模型逐步改进

阶段3（50+ 真实反馈）：
  ├─ 停用伪标签
  ├─ 纯真实反馈训练
  └─ 持续在线学习
```

**优势**：
- ✅ 立即可用（不需要等待数据）
- ✅ 平滑过渡（从伪标签到真实反馈）
- ✅ 持续改进（随着使用越来越准确）
- ✅ 避免冷启动问题

## 总结

### 核心设计

1. **特征工程**：6 类特征，50+ 维度
2. **模型选择**：Logistic Regression（简单、快速、可解释）
3. **训练数据**：隐式反馈自动标注
4. **在线学习**：实时更新模型
5. **冷启动**：启发式权重 → 在线学习

### 优先级

1. **Phase 6.1** - 特征工程 + 反馈收集 ⭐⭐⭐⭐⭐
2. **Phase 6.2** - LR 模型 + 启发式权重 ⭐⭐⭐⭐⭐
3. **Phase 6.3** - 在线学习 ⭐⭐⭐⭐
4. **Phase 6.4** - 高级模型（FM）⭐⭐⭐

### 预期效果

- 搜索结果排序更准确
- 用户需要的结果排在前面
- 随着使用越来越智能
- 平均选择位置从 3-4 降到 1-2

**最大价值**：从"静态排序"升级到"预测式排序"。
