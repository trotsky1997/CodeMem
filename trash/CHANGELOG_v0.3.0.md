# CodeMem v0.3.0 - Phase 1: Agent Experience First

## 🎉 新功能

### 智能记忆查询接口 `memory.query`

全新的自然语言查询接口，让 AI Agent 像人类一样访问记忆。

**特性**:
- ✅ 自然语言查询支持
- ✅ 自动意图识别（6 种意图类型）
- ✅ 时间表达式解析（昨天、上周、最近等）
- ✅ 同义词扩展（异步 → async, asyncio, 协程等）
- ✅ 自然语言响应（摘要、洞察、建议）
- ✅ 向后兼容（保留旧工具）

### 支持的查询类型

1. **搜索内容** - `QueryIntent.SEARCH_CONTENT`
   ```
   "我之前讨论过 Python 异步吗？"
   "有没有提到数据库优化？"
   ```

2. **查找会话** - `QueryIntent.FIND_SESSION`
   ```
   "上周关于数据库的对话"
   "昨天的讨论"
   ```

3. **活动摘要** - `QueryIntent.ACTIVITY_SUMMARY`
   ```
   "最近在做什么？"
   "最近7天的活动"
   ```

4. **获取上下文** - `QueryIntent.GET_CONTEXT`
   ```
   "那段代码的完整上下文"
   "详细内容是什么？"
   ```

5. **导出** - `QueryIntent.EXPORT`
   ```
   "导出那次对话"
   "保存这个会话"
   ```

6. **模式发现** - `QueryIntent.PATTERN_DISCOVERY`
   ```
   "我经常问什么问题？"
   "最常讨论的话题"
   ```

### 响应格式

```json
{
  "summary": "在 1月19日 下午3:30 的对话中找到了 3 条相关记录...",
  "insights": [
    "结果来自 SQL 索引 (2 条) 和 Markdown 索引 (1 条)",
    "这是今天的对话",
    "包含 1 条用户消息和 2 条助手回复"
  ],
  "key_findings": [
    {
      "rank": 1,
      "session": "1月19日 下午3:30",
      "text": "async def build_bm25_indexes_parallel()...",
      "score": 0.85
    }
  ],
  "suggestions": [
    "查看完整的 3 条结果？",
    "发现代码片段，需要查看完整上下文吗？"
  ]
}
```

## 🔧 技术实现

### 新增模块

1. **intent_recognition.py** - 意图识别
   - `parse_intent()` - 解析查询意图
   - `parse_temporal_expression()` - 时间表达式解析
   - `expand_synonyms()` - 同义词扩展

2. **nl_formatter.py** - 自然语言格式化
   - `format_search_results()` - 格式化搜索结果
   - `format_activity_summary()` - 格式化活动摘要
   - `format_error_response()` - 格式化错误响应

3. **test_phase1.py** - 测试套件
   - 意图识别测试
   - 时间解析测试
   - 同义词扩展测试
   - 格式化测试

### 架构变化

```
旧架构 (v0.2.0):
User Query → Tool Selection → BM25 Search → JSON Response

新架构 (v0.3.0):
User Query → Intent Recognition → Query Expansion → BM25 Search → NL Formatting → Natural Response
```

## 📊 对比

| 特性 | v0.2.0 | v0.3.0 |
|------|--------|--------|
| **工具数量** | 7 个 | 3 个 (1 新 + 2 旧) |
| **自然语言查询** | ❌ | ✅ |
| **意图识别** | ❌ | ✅ (6 种) |
| **时间解析** | ❌ | ✅ |
| **同义词扩展** | ❌ | ✅ |
| **自然语言响应** | ❌ | ✅ |
| **向后兼容** | N/A | ✅ |

## 🚀 使用示例

### 使用新接口 (推荐)

```python
# 自然语言查询
result = await memory_query_async("我之前讨论过 Python 异步吗？")

# 输出
{
  "summary": "在 1月19日 下午3:30 的对话中找到了 3 条相关记录。\n\n最相关的内容：「async def build_bm25_indexes_parallel(): 使用 ProcessPoolExecutor 并行构建两个索引」",
  "insights": [
    "结果来自 SQL 索引 (2 条) 和 Markdown 索引 (1 条)",
    "这是今天的对话"
  ],
  "suggestions": [
    "查看完整的 3 条结果？",
    "发现代码片段，需要查看完整上下文吗？"
  ]
}
```

### 使用旧接口 (Legacy)

```python
# 仍然可用，但标记为 [Legacy]
result = await bm25_search_async("Python async", limit=20, source="both")
```

## 📦 安装

```bash
# 安装依赖
pip install pydantic rank-bm25 tiktoken aiosqlite mcp

# 或使用 pip install -e .
pip install -e .
```

## 🧪 测试

```bash
# 运行 Phase 1 测试套件
python test_phase1.py
```

测试覆盖：
- ✅ 意图识别（6 种意图）
- ✅ 时间表达式解析（昨天、上周、最近等）
- ✅ 同义词扩展（10+ 领域词汇）
- ✅ 自然语言格式化

## 🔄 迁移指南

### 从 v0.2.0 迁移

**推荐方式**：使用新的 `memory.query` 工具

```python
# 旧方式 (v0.2.0)
result = await bm25_search_async("Python async", limit=20, source="both")
# 返回: {"query": "...", "results": [...]}

# 新方式 (v0.3.0)
result = await memory_query_async("我之前讨论过 Python 异步吗？")
# 返回: {"summary": "...", "insights": [...], "suggestions": [...]}
```

**向后兼容**：旧工具仍然可用

```python
# 这些工具仍然可用，但标记为 [Legacy]
- semantic.search
- activity.recent
```

## 🎯 Phase 1 目标达成

- ✅ 创建智能查询接口 `memory.query`
- ✅ 实现意图识别（6 种意图）
- ✅ 自然语言响应格式化
- ✅ 保持向后兼容
- ✅ 测试覆盖

## 🔮 下一步：Phase 2

Phase 2 将添加：
- 对话上下文管理
- Follow-up 查询支持
- 指代词解析（"那段代码"、"第二个结果"）
- 会话状态管理

## 📝 变更日志

### v0.3.0 (2026-01-20)

**新增**:
- 新增 `memory.query` 智能查询接口
- 新增 `intent_recognition.py` 模块
- 新增 `nl_formatter.py` 模块
- 新增 `test_phase1.py` 测试套件
- 支持 6 种查询意图识别
- 支持时间表达式解析（昨天、上周、最近等）
- 支持同义词扩展（10+ 领域词汇）
- 自然语言响应格式化

**标记为 Legacy**:
- `semantic.search` - 推荐使用 `memory.query`
- `activity.recent` - 推荐使用 `memory.query`

**依赖**:
- 新增 `mcp>=0.9.0` 依赖

**文档**:
- 新增 `CHANGELOG_v0.3.0.md`
- 更新 `AGENT_EXPERIENCE_REDESIGN.md`

## 🙏 致谢

感谢 Agent Experience First 设计理念的指导，让 CodeMem 从"工具集合"转变为"智能记忆助手"。

---

**准备好体验 Phase 2 了吗？** 🚀
