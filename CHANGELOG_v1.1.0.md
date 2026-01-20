# CodeMem v1.1.0 - Phase 6.1: CTR-Based Search Ranking

## 发布日期：2026-01-20

## 🎉 重大更新

**CTR 预估模型 + ConversationRank 算法**

Phase 6.1 实现了基于 CTR（点击率）预估的智能搜索排序系统，借鉴 Google PageRank 和电商搜索的核心理念。

---

## 核心功能

### 1. ConversationRank 算法 ⭐⭐⭐⭐⭐

类似 PageRank 的对话重要性评分系统。

**评分维度（总分 1.0）**：
- **基础分 (20%)** - 对话长度（50 条消息为满分）
- **深度分 (20%)** - 技术关键词密度（80+ 关键词库）
- **引用分 (30%)** - 被后续对话引用次数 ⭐ 最重要
- **质量分 (20%)** - 完整的问题-解决方案模式
- **时间分 (10%)** - 新近度（90 天半衰期）

**引用关系检测**：
- 显式引用："之前讨论过"、"上次提到"、"那段代码"
- 隐式引用：关键词重叠度 > 60%

**示例**：
```python
from conversation_rank import calculate_conversation_ranks

ranks = calculate_conversation_ranks(events)
# {'session_high_quality': 0.375, 'session_old_quality': 0.324, ...}
```

### 2. 距离伪标签训练数据生成 ⭐⭐⭐⭐⭐

从对话历史自动生成训练数据，无需人工标注。

**核心思想**：对话中距离近的句子更可能相关。

**4 种生成策略**：

1. **会话内 vs 会话外**（最简单）
   - 同一会话 = label 1.0
   - 不同会话 = label 0.0

2. **时间衰减**
   - label = exp(-时间距离 / 7天)
   - 同一会话 × 2 加成

3. **滑动窗口**
   - 窗口内（±20条）= 正样本
   - 窗口外 = 负样本

4. **混合策略**（最佳）
   - 综合评分 = 会话距离(50%) + 时间距离(30%) + 消息距离(20%)

**示例**：
```python
from distance_trainer import generate_distance_training_data

training_data = generate_distance_training_data(events, method='hybrid')
# 生成 117 条训练样本（14 高相关 + 63 中等 + 40 低相关）
```

### 3. 特征提取系统 ⭐⭐⭐⭐⭐

**6 大类特征，39 个维度**：

**查询特征 (9)**：
- 长度、词数、是否包含代码
- 查询类型（how/why/what）
- 技术关键词数量
- 版本号、时间表达式

**对话特征 (12)**：
- 消息数量、平均长度
- 代码块数量、是否有解决方案
- 技术关键词密度
- ConversationRank ⭐
- 时间新近度

**匹配特征 (5)**：
- BM25 分数
- 关键词重叠度
- 技术关键词重叠度
- 查询是否在标题中

**用户历史特征 (5)**：
- 话题偏好
- 时间偏好（喜欢新/旧内容）
- 对话类型偏好（长/短、代码密集）
- 引用历史

**上下文特征 (3)**：
- 是否同一会话
- 是否最近会话
- 是否 follow-up 查询

**位置特征 (5)**：
- 位置编号
- 位置偏差（DCG 风格）
- 是否 Top 1/3/5

**示例**：
```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_all(query, conversation, user_history, context, position)
# 返回 39 个特征的字典
```

### 4. CTR 预估模型 ⭐⭐⭐⭐⭐

**Logistic Regression 实现**：

**优点**：
- 简单、快速、可解释
- 适合小数据量（单用户场景）
- 特征权重清晰
- 启发式权重冷启动

**预测公式**：
```
P(选择) = 1 / (1 + exp(-Σ(wi × xi)))
```

**特征权重（Top 10）**：
1. bm25_score: 2.0
2. conversation_rank: 1.5
3. keyword_overlap: 1.5
4. tech_keyword_overlap: 1.5
5. is_same_session: 1.2
6. has_solution: 1.0
7. has_referenced_before: 1.0
8. has_code_block: 0.8
9. user_topic_preference: 0.8
10. is_recent: 0.5

**示例**：
```python
from ctr_model import LogisticRegressionCTR

model = LogisticRegressionCTR()
model.train(training_data, epochs=100)

features = model.extract_features(query, conversation, user_history, context, position)
ctr = model.predict(features)  # 0.0 - 1.0
```

### 5. 搜索排序集成 ⭐⭐⭐⭐⭐

**完整工作流**：
1. BM25 初始排序（现有）
2. 计算 ConversationRank
3. 提取特征（43 维）
4. 预测 CTR
5. 重新排序

**示例**：
```python
from search_ranker import search_with_ctr_ranking

results = await search_with_ctr_ranking(
    query="Python asyncio",
    bm25_search_func=bm25_search_async,
    limit=10,
    use_ctr=True
)
# 返回 CTR 排序的结果
```

### 6. Pattern Clustering 集成 ⭐⭐⭐⭐⭐

**Phase 4.5 + Phase 6.1 集成**

将 Pattern Clustering（模式聚类）的结果集成到 CTR 特征中，利用用户行为模式改进搜索排序。

**新增 4 个 Pattern 特征**：

1. **in_frequent_query_cluster** (权重 0.9)
   - 对话是否属于用户经常搜索的查询类型
   - 来自 query_clusters（相似查询聚类）

2. **session_type_match** (权重 0.7)
   - 对话类型是否匹配用户偏好
   - 来自 session_clusters（学习型、问题解决型等）
   - Top 1 = 1.0, Top 2 = 0.7, Top 3 = 0.5

3. **is_recurring_problem** (权重 1.2) ⭐ 最高权重
   - 对话是否解决了反复出现的问题
   - 来自 problem_patterns（重复问题识别）

4. **topic_cluster_match** (权重 0.8)
   - 对话话题是否属于用户常讨论的领域
   - 来自 topic_aggregation（7 大主题层级）
   - Top 1 = 1.0, Top 3 = 0.7, Top 5 = 0.5

**特征总数**：39 → **43 个特征**

**示例**：
```python
from pattern_integration import generate_user_history_with_patterns

# 生成包含 pattern clustering 的用户历史
user_history = generate_user_history_with_patterns(events)

# user_history 包含：
# - pattern_clusters: 4 类聚类结果
# - frequent_topics: Top 5 话题

# CTR 模型自动使用这些 pattern 特征
results = await search_with_ctr_ranking(
    query="Python asyncio",
    bm25_search_func=bm25_search_async,
    user_history=user_history,  # 包含 pattern_clusters
    use_ctr=True
)
```

**排序解释**：
```python
from pattern_integration import explain_ranking_with_patterns

explanation = explain_ranking_with_patterns(result, features)
# "✓ 属于你经常搜索的查询类型 | ✓ 匹配你偏好的会话类型 |
#  ⚠️ 这是一个反复出现的问题 | ✓ 属于你常讨论的话题"
```

---

## 新增模块

### conversation_rank.py
- `ConversationRank` 类 - 对话重要性评分
- `calculate_conversation_ranks()` - 批量计算
- `get_top_conversations()` - 获取 Top N 对话

### distance_trainer.py
- `DistanceBasedTrainer` 类 - 训练数据生成
- 4 种生成策略实现
- `generate_distance_training_data()` - 统一接口

### feature_extractor.py
- 6 个特征提取器类
- `FeatureExtractor` - 统一特征提取
- 43 个特征维度（含 4 个 pattern 特征）

### ctr_model.py
- `LogisticRegressionCTR` - LR 模型实现
- `CTRRanker` - CTR 排序器
- `create_ctr_model_from_distance_data()` - 快速创建

### search_ranker.py
- `SearchRanker` - 搜索排序器
- `initialize_search_ranker()` - 全局初始化
- `search_with_ctr_ranking()` - 统一搜索接口

### pattern_integration.py ⭐ 新增
- `generate_user_history_with_patterns()` - 生成包含 pattern clustering 的用户历史
- `enrich_search_results_with_patterns()` - 为搜索结果添加 pattern 元数据
- `explain_ranking_with_patterns()` - 生成排序解释
- `get_pattern_insights()` - 获取 pattern 洞察

---

## 测试

### 新增测试文件

**test_phase6_1.py** - ConversationRank 和距离训练数据
- 7 个测试用例
- 所有测试通过 ✅

**test_phase6_1_features.py** - 特征提取和 CTR 模型
- 9 个测试用例
- 所有测试通过 ✅

**test_phase6_1_e2e.py** - 端到端集成测试
- 5 个测试用例
- 所有测试通过 ✅

**test_pattern_integration.py** - Pattern clustering 集成测试 ⭐ 新增
- 7 个测试用例
- 所有测试通过 ✅
- 验证 Phase 4.5 + Phase 6.1 集成

### 测试结果

**ConversationRank 排序**：
```
session_high_quality:          0.375 ✅ (最高)
session_old_quality:           0.324
session_reference:             0.235
session_recent_low_quality:    0.221
session_short:                 0.207 (最低)
```

**CTR 预测**：
```
high_quality:  1.0000 ✅ (高质量对话)
low_quality:   0.8041 (低质量对话)
```

**特征提取**：
- 39 个特征成功提取
- Top 10 特征权重合理

---

## 性能

### 训练性能
- 训练数据生成：< 1 秒（100 条事件）
- ConversationRank 计算：< 0.5 秒（5 个会话）
- CTR 模型训练：< 2 秒（31 个样本，50 epochs）

### 预测性能
- 特征提取：< 1ms / 样本
- CTR 预测：< 0.1ms / 样本
- 重排序：< 10ms（10 个结果）

### 内存占用
- ConversationRank 缓存：~1KB / 会话
- CTR 模型：~50KB（39 个特征权重）
- 特征提取器：~100KB（关键词库）

---

## 向后兼容

✅ **完全兼容 v0.6.1**
- 所有现有工具保持不变
- CTR 排序是可选的（`use_ctr=True/False`）
- 默认仍使用 BM25 排序
- 不影响现有功能

---

## 使用示例

### 基础使用

```python
import asyncio
from search_ranker import initialize_search_ranker, search_with_ctr_ranking

async def main():
    # 1. 加载历史事件
    events = load_events_from_db()

    # 2. 初始化搜索排序器
    await initialize_search_ranker(events, method='hybrid')

    # 3. 执行搜索（自动使用 CTR 排序）
    results = await search_with_ctr_ranking(
        query="Python asyncio tutorial",
        bm25_search_func=bm25_search_async,
        limit=10,
        use_ctr=True
    )

    # 4. 查看结果
    for i, result in enumerate(results['results'], 1):
        print(f"{i}. {result['session_id']}")
        print(f"   CTR: {result['predicted_ctr']:.4f}")
        print(f"   BM25: {result['score']:.3f}")

asyncio.run(main())
```

### 高级使用

```python
from conversation_rank import calculate_conversation_ranks
from distance_trainer import generate_distance_training_data
from ctr_model import LogisticRegressionCTR

# 1. 计算 ConversationRank
ranks = calculate_conversation_ranks(events)

# 2. 生成训练数据
training_data = generate_distance_training_data(events, method='hybrid')

# 3. 训练 CTR 模型
model = LogisticRegressionCTR()
model.train(training_data, epochs=100)

# 4. 查看特征重要性
importance = model.get_feature_importance()
for name, weight in importance[:10]:
    print(f"{name}: {weight:.3f}")
```

---

## 设计文档

新增 3 个设计文档：

1. **SEARCH_RELEVANCE.md** - ConversationRank 算法设计
2. **CTR_PREDICTION.md** - CTR 预估模型设计
3. **DISTANCE_BASED_CTR.md** - 距离伪标签设计

---

## 已知限制

### 1. 训练数据量
- 当前实现适合单用户场景
- 训练数据量较小（< 1000 条）
- CTR 预测可能不够精确

**解决方案**：
- 使用启发式权重冷启动
- 随着使用逐步积累真实反馈
- 在线学习持续改进

### 2. 特征工程
- 当前 39 个特征可能不够全面
- 缺少向量化语义特征
- 缺少用户行为序列特征

**未来改进**：
- 添加 embedding 特征
- 添加会话序列特征
- 添加时间序列特征

### 3. 模型复杂度
- 当前只实现了 LR 模型
- 无法捕捉特征交叉
- 无法学习复杂非线性模式

**未来改进**：
- 实现 FM（Factorization Machines）
- 实现轻量级神经网络
- 模型集成

---

## 下一步计划

### Phase 6.2: 在线学习 (v1.2.0)
- [ ] 实时反馈收集
- [ ] 在线模型更新
- [ ] A/B 测试框架

### Phase 6.3: 高级模型 (v1.3.0)
- [ ] Factorization Machines
- [ ] 特征交叉优化
- [ ] 模型集成

### Phase 6.4: 结果多样化 (v1.4.0)
- [ ] 时间多样性
- [ ] 话题多样性
- [ ] 避免结果过于单一

---

## 贡献者

感谢所有为 CodeMem 做出贡献的开发者！

---

## 反馈

如有问题或建议，请在 GitHub Issues 中提出。

---

## 总结

Phase 6.1 实现了完整的 CTR 预估排序系统：

✅ **ConversationRank** - 对话重要性评分（类似 PageRank）
✅ **距离伪标签** - 自监督训练数据生成
✅ **特征提取** - 6 大类 39 个特征
✅ **CTR 模型** - Logistic Regression 实现
✅ **搜索集成** - 端到端 CTR 排序

**核心价值**：从"关键词匹配"升级到"智能相关性排序"。

**预期效果**：
- 搜索相关性提升 50%+
- 重要对话优先展示
- 用户需要的结果排在前面
- 平均选择位置从 3-4 降到 1-2
