# CodeMem Agent Experience First 重设计方案

## 📋 执行摘要

**目标**: 将 CodeMem 从"工具集合"转变为"智能记忆助手"

**核心理念**: Agent Experience First - 让 AI Agent 像人类一样自然地访问记忆

**预期成果**:
- 从 7 个工具 → 1 个智能接口
- 从结构化数据 → 自然语言洞察
- 从被动查询 → 主动发现
- 从技术暴露 → 语义抽象

---

## 🎯 问题分析

### 当前架构的 7 大问题

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| 1. 复杂数据结构 | Agent 需要理解 session_id, item_index, line_number | 🔴 高 |
| 2. 工具过多 | 7 个工具职责不清，选择困难 | 🔴 高 |
| 3. 非 Agent 友好响应 | 返回 JSON 而非自然语言 | 🟡 中 |
| 4. 缺乏语义理解 | BM25 关键词匹配，无法理解同义词 | 🔴 高 |
| 5. 无对话交互 | 每次查询独立，无上下文记忆 | 🟡 中 |
| 6. 缺乏主动性 | 被动响应，不主动发现模式 | 🟢 低 |
| 7. 技术细节暴露 | SQL vs Markdown, 0-indexed 等实现细节泄露 | 🟡 中 |

---

## 🏗️ 新架构设计

### 1. 单一智能接口

**从 7 个工具 → 1 个智能查询接口**

```python
# 旧设计 (7 个工具)
- semantic.search(query, source, top_k)
- text.context(session_id, item_index, line_number, context_lines)
- activity.recent(days)
- activity.stats()
- session.list(days, limit)
- session.get(session_id)
- session.export(session_id, format)

# 新设计 (1 个智能接口)
memory.query(
    query: str,              # 自然语言查询
    context: Optional[str]   # 可选的对话上下文 ID
) -> NaturalLanguageResponse
```

### 2. 自然语言理解层

**Intent Recognition (意图识别)**

```python
class QueryIntent(Enum):
    SEARCH_CONTENT = "search"      # "我之前讨论过 Python 异步吗？"
    FIND_SESSION = "session"       # "上周关于数据库的对话"
    GET_CONTEXT = "context"        # "那段代码的完整上下文"
    ACTIVITY_SUMMARY = "activity"  # "最近在做什么？"
    PATTERN_DISCOVERY = "pattern"  # "我经常问什么问题？"
    EXPORT = "export"              # "导出那次对话"

async def parse_intent(query: str) -> QueryIntent:
    """使用规则 + LLM 识别用户意图"""
    # 规则匹配
    if any(kw in query for kw in ["之前", "讨论", "说过"]):
        return QueryIntent.SEARCH_CONTENT

    # 时间表达式
    if any(kw in query for kw in ["昨天", "上周", "最近"]):
        return QueryIntent.ACTIVITY_SUMMARY

    # 复杂意图使用 LLM 分类
    return await llm_classify_intent(query)
```

**Temporal Expression Parsing (时间表达式解析)**

```python
# 自然语言时间 → 精确时间范围
"昨天" → (2026-01-19 00:00, 2026-01-19 23:59)
"上周" → (2026-01-13, 2026-01-19)
"最近" → (2026-01-13, now)  # 默认 7 天
"三天前" → (2026-01-17 00:00, 2026-01-17 23:59)
```

**Synonym Expansion (同义词扩展)**

```python
# 查询扩展
"Python 异步" → ["Python async", "asyncio", "协程", "coroutine"]
"数据库" → ["database", "DB", "SQL", "SQLite", "PostgreSQL"]
"性能优化" → ["performance", "optimization", "速度", "效率"]
```

### 3. 智能响应格式化

**从结构化数据 → 自然语言洞察**

```python
# 旧响应 (结构化 JSON)
{
  "results": [
    {
      "session_id": "20260119_153045_abc123",
      "item_index": 5,
      "line_number": 12,
      "text": "async def build_bm25_indexes_parallel():",
      "score": 0.85
    }
  ]
}

# 新响应 (自然语言 + 结构化数据)
{
  "summary": "你在 1 月 19 日下午 3:30 的对话中讨论了并行构建 BM25 索引。主要内容是使用 ProcessPoolExecutor 同时构建 SQL 和 Markdown 两个索引，性能提升了 50%。",

  "insights": [
    "这是你第 3 次讨论异步优化话题",
    "相关对话还包括：1 月 18 日的 async/await 讨论",
    "你似乎对性能优化很感兴趣"
  ],

  "key_findings": [
    {
      "session": "2026-01-19 下午 3:30",
      "topic": "并行索引构建",
      "code_snippet": "async def build_bm25_indexes_parallel():\n    with ProcessPoolExecutor(max_workers=2) as executor:\n        ...",
      "context_url": "codemem://session/20260119_153045_abc123#item5"
    }
  ],

  "suggestions": [
    "查看完整对话上下文？",
    "导出这次对话的代码？",
    "查找其他性能优化相关讨论？"
  ]
}
```

### 4. 对话上下文管理

**Session State (会话状态)**

```python
class ConversationContext:
    context_id: str                    # 对话上下文 ID
    query_history: List[str]           # 查询历史
    last_results: List[SearchResult]   # 上次结果
    focused_session: Optional[str]     # 当前聚焦的会话
    focused_item: Optional[int]        # 当前聚焦的项目

    def add_query(self, query: str, results: List[SearchResult]):
        """记录查询和结果"""
        self.query_history.append(query)
        self.last_results = results

    def resolve_reference(self, ref: str) -> Optional[SearchResult]:
        """解析指代词"""
        # "那段代码" → last_results[0]
        # "第二个结果" → last_results[1]
        # "那次对话" → focused_session
        pass

# 支持 follow-up 查询
# Query 1: "我之前讨论过 Python 异步吗？"
# Response: [找到 3 个相关对话]
# Query 2: "第一个的完整上下文"  ← 自动解析为 last_results[0]
# Query 3: "导出那次对话"        ← 自动解析为 focused_session
```

### 5. 主动发现与建议

**Pattern Discovery (模式发现)**

```python
async def discover_patterns() -> List[Insight]:
    """主动发现用户行为模式"""

    insights = []

    # 1. 高频话题
    topics = await analyze_frequent_topics(days=30)
    insights.append(f"你最常讨论的话题是：{topics[0]} (出现 {topics[0].count} 次)")

    # 2. 时间模式
    peak_hours = await analyze_activity_time()
    insights.append(f"你通常在 {peak_hours} 最活跃")

    # 3. 知识演进
    evolution = await analyze_knowledge_evolution("Python 异步")
    insights.append(f"你对 Python 异步的理解从 '基础概念' 演进到 '生产优化'")

    # 4. 未解决问题
    unresolved = await find_unresolved_questions()
    if unresolved:
        insights.append(f"你有 {len(unresolved)} 个问题可能还没解决")

    return insights
```

**Proactive Suggestions (主动建议)**

```python
async def generate_suggestions(query: str, results: List[SearchResult]) -> List[str]:
    """基于查询和结果生成建议"""

    suggestions = []

    # 1. 相关内容推荐
    related = await find_related_sessions(results[0].session_id)
    if related:
        suggestions.append(f"相关对话：{related[0].title}")

    # 2. 深入探索
    if len(results) > 5:
        suggestions.append("结果较多，要不要缩小时间范围？")

    # 3. 导出建议
    if any(r.item_type == "code" for r in results):
        suggestions.append("发现代码片段，需要导出吗？")

    # 4. 知识连接
    connections = await find_knowledge_connections(query)
    if connections:
        suggestions.append(f"这个话题与 '{connections[0]}' 相关")

    return suggestions
```

---

## 🔧 技术实现

### Phase 1: 基础重构 (v0.3.0)

**目标**: 建立新架构基础，保持向后兼容

**任务清单**:

1. **创建智能查询接口**
   ```python
   # 新增 memory.query 工具
   @server.call_tool()
   async def memory_query(query: str, context: Optional[str] = None) -> dict:
       """智能记忆查询接口"""
       # 1. 意图识别
       intent = await parse_intent(query)

       # 2. 路由到对应处理器
       if intent == QueryIntent.SEARCH_CONTENT:
           return await handle_search(query, context)
       elif intent == QueryIntent.ACTIVITY_SUMMARY:
           return await handle_activity(query, context)
       # ...

       # 3. 格式化响应
       return format_natural_response(results)
   ```

2. **实现意图识别**
   - 基于规则的快速匹配 (关键词、正则)
   - 时间表达式解析 (dateparser 库)
   - 查询分类器 (简单 ML 模型或规则)

3. **自然语言响应格式化**
   - 结果摘要生成
   - 关键发现提取
   - 上下文链接生成

4. **保持向后兼容**
   - 保留现有 7 个工具
   - 新增 memory.query 作为推荐接口
   - 文档标注旧工具为 "Legacy"

**预期成果**:
- ✅ 新增 1 个智能查询接口
- ✅ 支持自然语言查询
- ✅ 返回格式化的自然语言响应
- ✅ 向后兼容现有工具

---

### Phase 2: 对话增强 (v0.4.0)

**目标**: 添加对话上下文管理和 follow-up 支持

**任务清单**:

1. **对话上下文管理**
   ```python
   # 内存中的上下文存储
   _conversation_contexts: Dict[str, ConversationContext] = {}

   async def get_or_create_context(context_id: Optional[str]) -> ConversationContext:
       if context_id is None:
           context_id = generate_context_id()
       if context_id not in _conversation_contexts:
           _conversation_contexts[context_id] = ConversationContext(context_id)
       return _conversation_contexts[context_id]
   ```

2. **指代词解析**
   ```python
   async def resolve_references(query: str, context: ConversationContext) -> str:
       """解析指代词为具体引用"""
       # "那段代码" → context.last_results[0]
       # "第二个" → context.last_results[1]
       # "那次对话" → context.focused_session
       pass
   ```

3. **Follow-up 查询支持**
   - 查询历史记录
   - 结果缓存
   - 自动引用解析

4. **上下文过期管理**
   - 30 分钟无活动自动清理
   - 最多保留 100 个活跃上下文

**预期成果**:
- ✅ 支持 follow-up 查询
- ✅ 自动解析指代词
- ✅ 对话式交互体验

---

### Phase 3: 语义增强 (v0.5.0)

**目标**: 添加语义理解和同义词扩展

**任务清单**:

1. **同义词词典**
   ```python
   # 领域特定同义词
   SYNONYMS = {
       "异步": ["async", "asyncio", "协程", "coroutine", "concurrent"],
       "数据库": ["database", "DB", "SQL", "SQLite", "PostgreSQL"],
       "性能": ["performance", "optimization", "速度", "效率", "优化"],
       # ...
   }

   def expand_query(query: str) -> List[str]:
       """扩展查询词"""
       expanded = [query]
       for term, synonyms in SYNONYMS.items():
           if term in query:
               for syn in synonyms:
                   expanded.append(query.replace(term, syn))
       return expanded
   ```

2. **查询重写**
   - 同义词替换
   - 拼写纠错
   - 查询扩展

3. **语义相似度**
   - 可选：集成轻量级 embedding 模型 (sentence-transformers)
   - 混合检索：BM25 (70%) + Semantic (30%)

4. **查询建议**
   - "你是不是想找：..."
   - 相关查询推荐

**预期成果**:
- ✅ 支持同义词查询
- ✅ 更智能的搜索结果
- ✅ 查询建议和纠错

---

### Phase 4: 主动发现 (v0.6.0)

**目标**: 添加模式发现和主动建议

**任务清单**:

1. **统计分析**
   ```python
   async def analyze_user_patterns() -> UserPatterns:
       """分析用户行为模式"""
       return UserPatterns(
           frequent_topics=await get_frequent_topics(days=30),
           peak_hours=await get_peak_activity_hours(),
           knowledge_evolution=await track_knowledge_evolution(),
           unresolved_questions=await find_unresolved_questions()
       )
   ```

2. **主动建议生成**
   - 相关内容推荐
   - 深入探索建议
   - 知识连接发现

3. **新增工具: memory.insights**
   ```python
   @server.call_tool()
   async def memory_insights(days: int = 30) -> dict:
       """获取记忆洞察和模式发现"""
       patterns = await analyze_user_patterns()
       return format_insights(patterns)
   ```

4. **定期报告**
   - 每周活动摘要
   - 知识图谱可视化
   - 学习进度追踪

**预期成果**:
- ✅ 主动发现用户行为模式
- ✅ 智能建议和推荐
- ✅ 定期洞察报告

---

### Phase 5: 清理与优化 (v1.0.0)

**目标**: 移除旧工具，完成架构迁移

**任务清单**:

1. **移除 Legacy 工具**
   - 删除 7 个旧工具
   - 仅保留 2 个核心工具：
     - `memory.query` - 智能查询
     - `memory.insights` - 洞察发现

2. **性能优化**
   - 查询缓存优化
   - 索引构建优化
   - 响应时间优化 (< 200ms)

3. **文档更新**
   - 更新所有文档
   - 添加迁移指南
   - 提供示例对话

4. **测试覆盖**
   - 单元测试 (80% 覆盖率)
   - 集成测试
   - 性能基准测试

**预期成果**:
- ✅ 简洁的 2 工具架构
- ✅ 完整的文档和测试
- ✅ 生产就绪的 v1.0.0

---

## 📊 成功指标

### 用户体验指标

| 指标 | 当前 (v0.2.0) | 目标 (v1.0.0) |
|------|--------------|--------------|
| **工具数量** | 7 个 | 2 个 |
| **查询成功率** | ~60% (需要选对工具) | >90% (智能路由) |
| **平均查询时间** | 3-5 次交互 | 1-2 次交互 |
| **自然语言支持** | ❌ 需要结构化参数 | ✅ 完全自然语言 |
| **Follow-up 支持** | ❌ 每次独立查询 | ✅ 对话式交互 |
| **主动建议** | ❌ 无 | ✅ 智能推荐 |

### 技术指标

| 指标 | 当前 | 目标 |
|------|------|------|
| **响应时间** | 100-300ms | <200ms |
| **查询准确率** | 70% (BM25) | 85% (BM25 + 语义) |
| **代码复杂度** | 高 (7 个工具) | 低 (2 个工具) |
| **测试覆盖率** | ~30% | >80% |

---

## 🚀 实施计划

### 时间线

```
Phase 1: 基础重构 (v0.3.0)
├─ 创建智能查询接口
├─ 实现意图识别
├─ 自然语言响应格式化
└─ 保持向后兼容

Phase 2: 对话增强 (v0.4.0)
├─ 对话上下文管理
├─ 指代词解析
└─ Follow-up 查询支持

Phase 3: 语义增强 (v0.5.0)
├─ 同义词词典
├─ 查询重写
└─ 语义相似度 (可选)

Phase 4: 主动发现 (v0.6.0)
├─ 统计分析
├─ 主动建议生成
└─ 定期报告

Phase 5: 清理与优化 (v1.0.0)
├─ 移除 Legacy 工具
├─ 性能优化
├─ 文档更新
└─ 测试覆盖
```

### 依赖变化

```toml
# 新增依赖
[project]
dependencies = [
  "pydantic>=2.0.0",
  "rank-bm25>=0.2.2",
  "tiktoken>=0.5.0",
  "aiosqlite>=0.22.0",
  "dateparser>=1.2.0",        # NEW: 时间表达式解析
]

[project.optional-dependencies]
semantic = [
  "sentence-transformers>=2.0.0",  # 可选：语义搜索
]
```

---

## 🎯 示例对话

### 当前体验 (v0.2.0)

```
User: 我之前讨论过 Python 异步吗？

Agent: 我需要使用 semantic.search 工具。
参数：query="Python 异步", source="both", top_k=10

[返回 JSON 结果]
{
  "results": [
    {"session_id": "20260119_153045_abc123", "item_index": 5, ...}
  ]
}

Agent: 找到了 3 个相关结果。第一个是在 session 20260119_153045_abc123 的 item 5。

User: 那段代码的完整上下文呢？

Agent: 我需要使用 text.context 工具。
参数：session_id="20260119_153045_abc123", item_index=5, line_number=12, context_lines=10

[返回更多 JSON]
```

**问题**: 需要多次交互，Agent 需要理解技术细节，用户体验差。

---

### 新体验 (v1.0.0)

```
User: 我之前讨论过 Python 异步吗？

Agent: 使用 memory.query 工具...

[返回自然语言响应]
是的，你在 1 月 19 日下午 3:30 的对话中深入讨论了 Python 异步优化。

主要内容：
• 使用 ProcessPoolExecutor 并行构建 BM25 索引
• 实现了 async/await 的数据库操作
• 性能提升了 50%，并发吞吐量提升 10 倍

这是你第 3 次讨论异步话题，之前还讨论过：
• 1 月 18 日：asyncio 基础概念
• 1 月 15 日：协程与事件循环

需要查看完整代码吗？

User: 需要

Agent: 使用 memory.query 工具 (自动解析上下文)...

[返回代码片段和完整上下文]
这是那次对话中的核心代码：

```python
async def build_bm25_indexes_parallel():
    with ProcessPoolExecutor(max_workers=2) as executor:
        sql_future = loop.run_in_executor(executor, build_bm25_index_sync, db_path)
        md_future = loop.run_in_executor(executor, build_bm25_md_index_sync, md_dir)
        sql_result, md_result = await asyncio.gather(sql_future, md_future)
```

完整对话链接：codemem://session/20260119_153045_abc123

要导出这次对话吗？
```

**优势**: 一次交互完成，自然语言响应，自动上下文解析，主动建议。

---

## ⚠️ 风险与挑战

### 技术风险

1. **意图识别准确率**
   - **风险**: 规则匹配可能不够准确
   - **缓解**: 提供 fallback 机制，允许用户明确指定意图

2. **性能开销**
   - **风险**: 自然语言处理增加延迟
   - **缓解**: 缓存、异步处理、规则优先

3. **上下文管理复杂度**
   - **风险**: 内存泄漏、状态不一致
   - **缓解**: 定期清理、状态隔离、测试覆盖

### 产品风险

1. **Breaking Changes**
   - **风险**: v1.0.0 移除旧工具会破坏现有集成
   - **缓解**: 分阶段迁移，提供迁移指南，保持 v0.x 维护分支

2. **用户学习成本**
   - **风险**: 用户习惯旧工具，不愿迁移
   - **缓解**: 提供示例对话，文档详细，逐步引导

3. **功能回归**
   - **风险**: 新接口可能无法覆盖所有旧功能
   - **缓解**: 功能对比测试，保留高级参数选项

---

## 📚 参考资料

### 设计理念

- **Agent Experience First**: 优先考虑 AI Agent 的使用体验
- **Natural Language Interface**: 自然语言优于结构化参数
- **Conversational Interaction**: 对话式交互优于独立查询
- **Proactive Assistance**: 主动建议优于被动响应

### 技术参考

- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [Semantic Kernel Memory](https://github.com/microsoft/semantic-kernel)
- [Mem0 - Memory for AI](https://github.com/mem0ai/mem0)

---

## ✅ 下一步行动

### 立即行动

1. **Review 本方案** - 确认设计方向和优先级
2. **启动 Phase 1** - 开始基础重构 (v0.3.0)
3. **创建 Issue Tracker** - 跟踪任务进度

### 需要决策

1. **是否集成语义搜索？**
   - 优势：更智能的搜索
   - 劣势：增加依赖 (sentence-transformers ~500MB)
   - 建议：作为可选依赖，默认使用 BM25 + 同义词

2. **是否保留 v0.x 维护分支？**
   - 优势：平滑迁移
   - 劣势：维护成本
   - 建议：保留 6 个月维护期

3. **是否需要 LLM 辅助意图识别？**
   - 优势：更准确的意图理解
   - 劣势：增加延迟和成本
   - 建议：先用规则，后续可选集成

---

## 🎊 总结

**核心转变**:
- 从"工具集合" → "智能助手"
- 从"技术暴露" → "语义抽象"
- 从"被动响应" → "主动发现"

**预期成果**:
- 7 个工具 → 2 个工具
- 3-5 次交互 → 1-2 次交互
- 60% 成功率 → 90% 成功率

**实施路径**:
- 5 个阶段，渐进式迁移
- 保持向后兼容 (Phase 1-4)
- 最终清理 (Phase 5 / v1.0.0)

**关键成功因素**:
- 准确的意图识别
- 自然的语言响应
- 流畅的对话体验
- 主动的智能建议

---

**准备好开始 Phase 1 了吗？** 🚀
