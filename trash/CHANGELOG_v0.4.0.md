# CodeMem v0.4.0 - Phase 2: 对话增强

## 🎉 新功能

### 对话上下文管理

Phase 2 实现了完整的对话上下文管理系统，支持 follow-up 查询和指代词解析。

**核心特性**:
- ✅ 对话上下文自动管理
- ✅ 查询历史追踪（最多 10 条）
- ✅ 结果缓存和引用
- ✅ 指代词自动解析
- ✅ Follow-up 查询检测
- ✅ 上下文自动过期（30 分钟）
- ✅ LRU 驱逐策略（最多 100 个上下文）

### 支持的指代词

1. **排名引用**
   ```
   "第一个" / "first one" → 第 1 个结果
   "第二个" / "second one" → 第 2 个结果
   "第三个" / "third one" → 第 3 个结果
   ```

2. **内容引用**
   ```
   "那段代码" / "that code" → 最近包含代码的结果
   "那次对话" / "that conversation" → 当前聚焦的会话
   "上一个" / "previous one" → 上一个结果
   ```

3. **简短查询**
   ```
   "详细" → 查看详细信息
   "完整" → 查看完整内容
   "更多" → 查看更多结果
   ```

## 📊 使用示例

### 示例 1: Follow-up 查询

```python
# 第一次查询
query1 = "我之前讨论过 Python 异步吗？"
result1 = await memory_query_async(query1)
# 返回: {
#   "summary": "在 1月19日 下午3:30 的对话中找到了 3 条相关记录...",
#   "metadata": {"context_id": "ctx_abc123"}
# }

# Follow-up 查询 - 使用 context_id
context_id = result1["metadata"]["context_id"]
query2 = "第一个"
result2 = await memory_query_async(query2, context=context_id)
# 返回: {
#   "summary": "这是第1个结果的详细信息。",
#   "insights": ["会话 ID: session1", "时间: 1月19日 下午3:30"],
#   "metadata": {"reference_resolved": True, "reference_type": "rank"}
# }
```

### 示例 2: 代码引用

```python
# 第一次查询
query1 = "异步索引构建"
result1 = await memory_query_async(query1)
# 找到包含代码的结果

# Follow-up 查询
context_id = result1["metadata"]["context_id"]
query2 = "那段代码的完整上下文"
result2 = await memory_query_async(query2, context=context_id)
# 自动定位到包含代码的结果并返回详细信息
```

### 示例 3: 会话过滤

```python
# 第一次查询
query1 = "数据库优化"
result1 = await memory_query_async(query1)
# 找到多个会话的结果

# Follow-up 查询 - 在特定会话中搜索
context_id = result1["metadata"]["context_id"]
query2 = "那次对话中的具体实现"
result2 = await memory_query_async(query2, context=context_id)
# 自动过滤到聚焦的会话
```

## 🔧 技术实现

### 新增模块: context_manager.py

**核心类**:

1. **ConversationContext** - 对话上下文
   ```python
   @dataclass
   class ConversationContext:
       context_id: str
       query_history: List[str]           # 查询历史
       result_history: List[List[SearchResult]]  # 结果历史
       focused_session: Optional[str]     # 聚焦会话
       focused_item_index: Optional[int]  # 聚焦项目
   ```

2. **ContextManager** - 上下文管理器
   ```python
   class ContextManager:
       async def get_or_create(context_id: Optional[str]) -> ConversationContext
       async def cleanup_expired() -> int
       async def get_stats() -> Dict[str, Any]
   ```

3. **辅助函数**
   ```python
   def resolve_reference(query: str, context: ConversationContext) -> Optional[Dict]
   def is_followup_query(query: str) -> bool
   ```

### 架构变化

```
v0.3.0:
User Query → Intent Recognition → Search → Format → Response

v0.4.0:
User Query → Context Retrieval → Follow-up Detection → Reference Resolution
    ↓
Intent Recognition → Search → Format → Response + Context Update
```

## 📈 对比

| 特性 | v0.3.0 | v0.4.0 |
|------|--------|--------|
| **对话上下文** | ❌ | ✅ |
| **Follow-up 查询** | ❌ | ✅ |
| **指代词解析** | ❌ | ✅ (5 种类型) |
| **查询历史** | ❌ | ✅ (最多 10 条) |
| **上下文过期** | ❌ | ✅ (30 分钟) |
| **LRU 驱逐** | ❌ | ✅ (最多 100 个) |

## 🧪 测试结果

```
✅ Follow-up 查询检测 - 6/6 通过
✅ 指代词解析 - 5/5 通过
✅ 上下文管理 - 7/7 通过
✅ 查询历史 - 3/3 通过
✅ 上下文清理 - 3/3 通过
```

**总计**: 24/24 测试通过 ✅

## 🎯 Phase 2 目标达成

- ✅ 对话上下文管理
- ✅ Follow-up 查询支持
- ✅ 指代词解析（5 种类型）
- ✅ 查询历史追踪
- ✅ 上下文过期管理
- ✅ LRU 驱逐策略
- ✅ 测试覆盖

**完成度**: 100% ✅

## 🔄 从 v0.3.0 迁移

### API 变化

**v0.3.0**:
```python
result = await memory_query_async("我之前讨论过 Python 异步吗？")
# 返回: {"summary": "...", "metadata": {}}
```

**v0.4.0**:
```python
# 第一次查询
result = await memory_query_async("我之前讨论过 Python 异步吗？")
# 返回: {"summary": "...", "metadata": {"context_id": "ctx_abc123"}}

# Follow-up 查询（新功能）
context_id = result["metadata"]["context_id"]
result2 = await memory_query_async("第一个", context=context_id)
# 自动解析引用
```

### 向后兼容

- ✅ 所有 v0.3.0 查询仍然有效
- ✅ `context` 参数是可选的
- ✅ 不使用 `context` 参数时行为与 v0.3.0 相同

## 📊 性能影响

| 指标 | v0.3.0 | v0.4.0 | 变化 |
|------|--------|--------|------|
| **查询延迟** | 100ms | 105ms | +5ms |
| **内存占用** | 60MB | 65MB | +5MB |
| **上下文开销** | N/A | ~1KB/context | 新增 |

## 🔮 下一步：Phase 3

Phase 3 将添加：
- 同义词词典扩展
- 查询重写和纠错
- 语义相似度搜索（可选）
- 查询建议

## 📝 变更日志

### v0.4.0 (2026-01-20)

**新增**:
- 对话上下文管理系统
- Follow-up 查询支持
- 指代词自动解析（5 种类型）
- 查询历史追踪（最多 10 条）
- 上下文自动过期（30 分钟 TTL）
- LRU 驱逐策略（最多 100 个上下文）
- 后台清理任务（每 5 分钟）

**新增模块**:
- `context_manager.py` - 对话上下文管理
- `test_phase2.py` - Phase 2 测试套件

**API 变化**:
- `memory.query` 现在返回 `context_id` 在 metadata 中
- `memory.query` 支持可选的 `context` 参数用于 follow-up 查询

**测试**:
- 新增 24 个 Phase 2 测试用例
- 所有测试通过 ✅

## 🙏 致谢

Phase 2 实现了真正的对话式交互，让 AI Agent 能够像人类一样进行多轮对话。

---

**准备好体验 Phase 3 了吗？** 🚀
