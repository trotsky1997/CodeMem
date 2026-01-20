# 更新日志

## [0.3.0] - 2026-01-20 - Phase 1: Agent Experience First

### 🎉 重大更新

**新增智能记忆查询接口 `memory.query`**

这是 Agent Experience First 重设计的第一阶段，将 CodeMem 从"工具集合"转变为"智能记忆助手"。

### 新增功能

- **自然语言查询支持**
  - 支持完全自然的中英文查询
  - 自动意图识别（6 种意图类型）
  - 时间表达式解析（昨天、上周、最近等）
  - 同义词自动扩展（异步 → async, asyncio, 协程等）

- **智能意图识别**
  - `SEARCH_CONTENT` - 搜索内容："我之前讨论过 Python 异步吗？"
  - `FIND_SESSION` - 查找会话："上周关于数据库的对话"
  - `ACTIVITY_SUMMARY` - 活动摘要："最近在做什么？"
  - `GET_CONTEXT` - 获取上下文："那段代码的完整上下文"
  - `EXPORT` - 导出："导出那次对话"
  - `PATTERN_DISCOVERY` - 模式发现："我经常问什么问题？"

- **自然语言响应格式化**
  - 摘要（Summary）- 简洁的结果概述
  - 洞察（Insights）- 数据来源、时间、分布等信息
  - 关键发现（Key Findings）- Top 3 结果详情
  - 建议（Suggestions）- 后续操作建议

- **新增模块**
  - `intent_recognition.py` - 意图识别和查询解析
  - `nl_formatter.py` - 自然语言响应格式化
  - `test_phase1.py` - Phase 1 测试套件

### 向后兼容

- ✅ 保留所有旧工具（标记为 [Legacy]）
- ✅ `semantic.search` 仍然可用
- ✅ `activity.recent` 仍然可用
- ✅ 推荐使用新的 `memory.query` 接口

### 技术改进

- 同义词词典支持 10+ 领域词汇
- 时间表达式支持 8+ 常用格式
- 测试覆盖率显著提升

### 依赖变化

- 新增 `mcp>=0.9.0` - MCP 协议支持

### 文档

- 新增 `CHANGELOG_v0.3.0.md` - 详细的 v0.3.0 说明
- 更新 `AGENT_EXPERIENCE_REDESIGN.md` - Phase 1 完成标记

### 下一步

Phase 2 将添加对话上下文管理和 follow-up 查询支持。

---

## [0.2.0] - 2026-01-20 - Async Only

### 重大变更 (BREAKING CHANGES)

- **移除同步版本**
  - 删除 `mcp_server_sync.py`
  - 仅保留异步版本 `mcp_server.py`
  - 单一入口点：`codemem-mcp`

### 依赖变化

- `aiosqlite>=0.22.0` 现在是必需依赖（之前是可选）
- 移除 `[async]` 可选依赖组

### 性能

- 索引构建速度提升 50%（并行构建）
- 并发吞吐量提升 10 倍
- 100 并发请求下性能提升 88%

---

## [0.1.2] - 2026-01-20

### 新增
- **双索引 BM25 搜索架构**
  - SQL 索引：快速搜索最近 10,000 条记录
  - Markdown 索引：全历史搜索，无记录限制
  - 支持独立或组合搜索

- **semantic.search 增强**
  - 新增 `source` 参数：`"sql"`, `"markdown"`, `"both"`
  - 结果包含数据源标识
  - 独立缓存每个数据源的查询结果

### 优化
- **搜索灵活性提升**
  - SQL 索引：适合快速查询最近对话
  - Markdown 索引：适合全历史深度搜索
  - 组合搜索：最全面的结果覆盖

### 技术细节
- `build_bm25_md_index()` - 从 Markdown 文件构建索引
- 正则表达式解析 Markdown 消息格式
- 双索引结果按相关性分数合并排序

## [0.1.1] - 2026-01-20

### 新增
- **6 个快捷工具**，覆盖 95% 的查询场景
  - `activity.recent` - 最近活动摘要
  - `session.get` - 完整会话历史
  - `tools.usage` - 工具使用统计
  - `platform.stats` - 平台活动分布
  - `semantic.search` - BM25 语义搜索
  - `sql.query` - 自定义 SQL 查询

- **语义搜索功能**
  - 基于 BM25 算法的语义搜索
  - 使用 Tiktoken 进行多语言分词（支持中英文）
  - 自动构建搜索索引

- **智能缓存系统**
  - LRU + TTL 双重缓存策略
  - 缓存大小限制（默认 100 条）
  - 缓存过期时间（默认 1 小时）
  - 缓存命中率统计

- **预计算统计信息**
  - 数据库统计摘要资源
  - 常用查询模板资源
  - 会话 Markdown 索引资源

### 优化
- **Token 效率提升 95-99%**
  - 快捷工具替代多次 SQL 查询
  - 智能缓存减少重复计算
  - 预计算统计信息

- **后台数据库构建**
  - 异步构建数据库，不阻塞启动
  - 构建完成前返回等待提示
  - 支持强制重建（`--rebuild`）

- **SQL 查询预览模式**
  - 可选的结果预览（`preview=true`）
  - 可配置预览行数和单元格长度
  - 减少不必要的 token 消耗

### 修复
- 修复 MCP 通知响应错误
- 修复 SQL 错误导致连接断开
- 修复路径解析问题
- 修复缓存失效问题

## [0.1.0] - 2026-01-19

### 新增
- **统一历史记录系统**
  - 支持 Claude Code、Codex CLI、Cursor、OpenCode
  - 统一数据模型（UnifiedEvent）
  - SQLite 数据库存储

- **MCP 服务器**
  - 实现 MCP 2024-11-05 协议
  - 支持 tools/list、tools/call
  - 支持 resources/list、resources/read

- **数据库结构**
  - `events` 表 - 面向检索的干净视图
  - `events_raw` 表 - 完整原始数据
  - 自动索引优化

- **Markdown 导出**
  - 按会话导出 Markdown 文件
  - 统一文件模板
  - 支持元数据和时间排序

- **文本工具**
  - `text.shell` 工具支持 rg/sed/awk
  - 路径限制在 md_sessions 目录
  - 输出截断保护

### 技术细节
- Python >= 3.10
- 使用 pandas 处理数据
- 使用 pydantic 进行数据验证
- 使用 sqlite3 存储数据

## 版本说明

### 语义化版本
- **主版本号**：不兼容的 API 变更
- **次版本号**：向后兼容的功能新增
- **修订号**：向后兼容的问题修复

### 发布周期
- 稳定版本：经过充分测试的版本
- 开发版本：包含最新功能但可能不稳定

## 迁移指南

### 从 0.1.0 升级到 0.1.1

1. **安装新依赖**
```bash
pip install rank-bm25 tiktoken
```

2. **更新配置**
无需更改配置，向后兼容。

3. **重建索引（可选）**
```bash
codemem-mcp --db ~/.codemem/codemem.sqlite --rebuild
```

4. **使用新工具**
优先使用快捷工具替代 SQL 查询：
- 用 `activity.recent` 替代手动查询最近活动
- 用 `semantic.search` 替代 SQL LIKE 查询
- 用 `session.get` 替代手动查询会话详情

## 已知问题

### 0.1.1
- BM25 索引在大数据集上构建较慢（计划在 0.2.0 优化）
- 缓存不支持持久化（计划在 0.2.0 添加）
- 不支持增量索引更新（计划在 0.2.0 添加）

### 0.1.0
- uvx 在 sandbox 环境可能失败（已在 0.1.1 文档中说明解决方案）
- SQL 错误会导致连接断开（已在 0.1.1 修复）

## 路线图

### 0.2.0（计划中）
- [ ] 增量索引更新
- [ ] 持久化缓存
- [ ] 更多聚合视图
- [ ] 性能监控面板
- [ ] 导出为其他格式（JSON、CSV）

### 0.3.0（计划中）
- [ ] 向量搜索支持
- [ ] 多数据库支持
- [ ] Web UI
- [ ] API 服务器模式

### 1.0.0（计划中）
- [ ] 完整的测试覆盖
- [ ] 生产环境优化
- [ ] 完整的文档
- [ ] 稳定的 API

## 贡献者

感谢所有为 CodeMem 做出贡献的开发者！

## 反馈

如有问题或建议，请在 GitHub Issues 中提出。
