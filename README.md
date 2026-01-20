# CodeMem

CodeMem 是一个高效的 AI 对话历史管理系统，通过 MCP (Model Context Protocol) 为 AI Agent 提供长期记忆能力。

## 📚 文档导航

- **[快速开始](QUICKSTART.md)** - 5 分钟快速上手
- **[架构文档](ARCHITECTURE.md)** - 深入了解设计和实现
- **[更新日志](CHANGELOG.md)** - 版本历史和路线图
- **[文档索引](DOCS.md)** - 完整文档导航

## 特性

### 核心功能
- **统一历史记录** - 整合 Claude Code、Codex CLI、Cursor、OpenCode 等平台的对话历史
- **语义搜索** - 基于 BM25 + Tiktoken 的多语言语义搜索（支持中英文）
- **智能缓存** - LRU + TTL 缓存机制，重复查询零成本
- **快捷工具** - 6 个高效工具覆盖 95% 的查询场景
- **结构化存储** - SQLite 数据库 + Markdown 文档库双重存储

### 性能优化
- **Token 效率提升 95-99%** - 通过快捷工具替代多次 SQL 查询
- **后台构建** - 数据库异步构建，不阻塞启动
- **增量更新** - 仅处理新增的对话记录

## 安装

### 依赖要求
- Python >= 3.10
- pandas >= 2.0.0
- pydantic >= 2.0.0
- rank-bm25 >= 0.2.2
- tiktoken >= 0.5.0

### 安装方法

#### 方法 1: 使用 uvx（推荐）
```bash
uvx --from /path/to/codemem codemem-mcp --db ~/.codemem/codemem.sqlite
```

#### 方法 2: 使用 venv
```bash
# 创建虚拟环境
python3 -m venv ~/.venv-codemem

# 安装依赖
~/.venv-codemem/bin/pip install -e /path/to/codemem

# 启动服务
~/.venv-codemem/bin/python /path/to/codemem/mcp_server.py --db ~/.codemem/codemem.sqlite
```

#### 方法 3: 直接安装
```bash
pip install -e /path/to/codemem
codemem-mcp --db ~/.codemem/codemem.sqlite
```

## 配置 MCP

### Claude Code
在 `~/.claude/config.json` 中添加：
```json
{
  "mcpServers": {
    "codemem": {
      "command": "python",
      "args": ["/path/to/codemem/mcp_server.py", "--db", "/path/to/.codemem/codemem.sqlite"]
    }
  }
}
```

### Codex CLI
```bash
codex mcp add codemem -- python /path/to/codemem/mcp_server.py --db /path/to/.codemem/codemem.sqlite
```

## 工具列表

### Tier 1: 快捷工具（最高效）

#### 1. activity.recent
获取最近活动摘要，一次调用获取所有信息。

**参数:**
- `days` (int, 默认 7) - 查询最近 N 天的活动

**使用场景:**
- "我最近在做什么？"
- "过去一周的工作总结"

**示例:**
```json
{"name": "activity.recent", "arguments": {"days": 7}}
```

#### 2. session.get
获取特定会话的完整对话历史。

**参数:**
- `session_id` (string, 必需) - 会话 ID（8 字符哈希）

**使用场景:**
- "查看某个项目的完整对话"
- "回顾之前的讨论内容"

**示例:**
```json
{"name": "session.get", "arguments": {"session_id": "73133d96"}}
```

#### 3. tools.usage
查看工具使用统计。

**参数:**
- `days` (int, 默认 30) - 统计最近 N 天

**使用场景:**
- "我最常用哪些工具？"
- "工具使用频率分析"

**示例:**
```json
{"name": "tools.usage", "arguments": {"days": 30}}
```

#### 4. platform.stats
查看各平台活动分布。

**参数:**
- `days` (int, 默认 30) - 统计最近 N 天

**使用场景:**
- "我在哪个平台上最活跃？"
- "平台使用情况分析"

**示例:**
```json
{"name": "platform.stats", "arguments": {"days": 30}}
```

### Tier 2: 搜索工具

#### 5. semantic.search
自然语言语义搜索，支持中英文。

**参数:**
- `query` (string, 必需) - 搜索查询
- `limit` (int, 默认 20) - 最大结果数（最大 50）

**使用场景:**
- "找到关于 Python 调试的对话"
- "搜索数据库优化相关内容"

**示例:**
```json
{"name": "semantic.search", "arguments": {"query": "Python debugging", "limit": 20}}
```

### Tier 3: 高级工具

#### 6. sql.query
执行自定义 SQL 查询（仅限只读）。

**参数:**
- `query` (string, 必需) - SQL 查询语句
- `limit` (int, 默认 100) - 行数限制（最大 50）
- `preview` (bool, 默认 false) - 是否在 content.text 中显示预览
- `preview_rows` (int, 默认 5) - 预览行数（1-50）
- `preview_cell_len` (int, 默认 80) - 单元格最大长度（10-200）

**使用场景:**
- 复杂自定义查询
- 仅在前面工具无法满足时使用

**示例:**
```json
{
  "name": "sql.query",
  "arguments": {
    "query": "SELECT timestamp, role, text FROM events WHERE text LIKE '%Python%' ORDER BY timestamp DESC",
    "preview": true,
    "preview_rows": 10
  }
}
```

## 资源列表

### 1. codemem://schema/events
`events` 表结构（面向检索的干净视图）

### 2. codemem://schema/events_raw
`events_raw` 表结构（底表，包含原始数据）

### 3. codemem://query/templates
常用 SQL 查询模板

### 4. codemem://stats/summary
预计算的统计信息

### 5. codemem://sessions/index
会话 Markdown 文件列表

### 6. codemem://sessions/<filename>
单个会话的 Markdown 文档

## 数据库结构

### events 表（检索视图）
面向检索的干净视图，仅包含可索引内容。

| 字段 | 类型 | 说明 |
|------|------|------|
| platform | TEXT | 平台名称（claude/codex/cursor/opencode） |
| session_id | TEXT | 会话 ID |
| message_id | TEXT | 消息 ID |
| timestamp | TEXT | 时间戳 |
| role | TEXT | 角色（user/assistant） |
| item_type | TEXT | 内容类型（text/tool_use/tool_result/thinking） |
| text | TEXT | 文本内容 |
| tool_name | TEXT | 工具名称 |
| source_file | TEXT | 源文件路径 |

### events_raw 表（完整数据）
包含所有原始字段，适合追溯和排错。

额外字段：
- `is_indexable` - 是否可索引
- `index_text` - 索引文本（优化后的检索字段）
- `tool_args` - 工具参数
- `tool_result` - 工具结果
- `tool_result_summary` - 工具结果摘要
- `raw_json` - 原始 JSON 数据

## 常用查询模板

### 关键词搜索
```sql
SELECT timestamp, role, text, source_file
FROM events
WHERE text LIKE '%关键词%'
ORDER BY timestamp DESC
LIMIT 50;
```

### 时间范围查询
```sql
SELECT timestamp, role, text, source_file
FROM events
WHERE timestamp >= '2026-01-01'
  AND text LIKE '%关键词%'
ORDER BY timestamp DESC
LIMIT 50;
```

### 会话统计
```sql
SELECT session_id, COUNT(*) as message_count
FROM events
GROUP BY session_id
ORDER BY message_count DESC
LIMIT 20;
```

### 角色过滤
```sql
SELECT timestamp, text, source_file
FROM events
WHERE role = 'assistant'
ORDER BY timestamp DESC
LIMIT 50;
```

### 工具使用统计
```sql
SELECT tool_name, COUNT(*) as usage_count
FROM events
WHERE tool_name IS NOT NULL
GROUP BY tool_name
ORDER BY usage_count DESC;
```

## 使用建议

### Agent 决策树
```
用户请求
    ↓
需要最近活动？ → activity.recent
    ↓
需要特定session？ → session.get
    ↓
需要工具统计？ → tools.usage
    ↓
需要平台分析？ → platform.stats
    ↓
需要搜索内容？ → semantic.search
    ↓
需要复杂查询？ → sql.query
```

### 最佳实践
1. **优先使用快捷工具** - 95% 的场景可以用前 4 个工具解决
2. **语义搜索优于 SQL** - 自然语言查询更直观，避免 SQL 试错
3. **仅在必要时使用 sql.query** - 复杂查询才需要自定义 SQL
4. **利用缓存** - 重复查询会自动命中缓存，零成本返回
5. **使用预览模式** - sql.query 开启 preview 可以快速查看结果

## 命令行参数

```bash
codemem-mcp [OPTIONS]

选项:
  --db PATH                    数据库路径（默认: ~/.codemem/codemem.sqlite）
  --include-history            包含历史记录
  --root PATH                  额外的根目录（可多次指定）
  --no-export-md-sessions      不导出 Markdown 会话文件
  --rebuild                    强制重建数据库
```

## 开发

### 项目结构
```
codemem/
├── mcp_server.py           # MCP 服务器主文件
├── unified_history.py      # 历史记录统一处理
├── export_sessions_md.py   # Markdown 导出
├── models.py               # 数据模型
├── test_query.py           # 查询测试
└── pyproject.toml          # 项目配置
```

### 本地开发
```bash
# 安装开发依赖
pip install -e .

# 运行测试
python test_query.py

# 导出 Markdown 会话
python export_sessions_md.py --db ~/.codemem/codemem.sqlite --out ~/.codemem/md_sessions
```

### 调试
```bash
# 启用详细日志
python mcp_server.py --db ~/.codemem/codemem.sqlite 2>&1 | tee mcp.log

# 查看缓存统计
# 缓存命中率会在日志中显示
```

## 性能指标

| 场景 | 之前 | 现在 | 工具 | 节省 |
|------|------|------|------|------|
| 最近活动 | 6次调用，41秒 | 1次调用，2秒 | activity.recent | 95% |
| Session详情 | 多次SQL | 1次调用 | session.get | 90% |
| 工具统计 | 手写SQL | 1次调用 | tools.usage | 85% |
| 平台分析 | 多次查询 | 1次调用 | platform.stats | 85% |
| 主题搜索 | SQL试错 | 自然语言 | semantic.search | 90% |
| 重复查询 | 每次执行 | 缓存返回 | 智能缓存 | 99% |

**总体 Token 节省：95-99%**

## 故障排除

### MCP 连接失败
1. 检查数据库路径是否正确
2. 确认 Python 版本 >= 3.10
3. 验证所有依赖已安装：`pip list | grep -E "(rank-bm25|tiktoken|pandas|pydantic)"`

### 查询返回空结果
1. 检查数据库是否已构建：`ls -lh ~/.codemem/codemem.sqlite`
2. 使用 `--rebuild` 强制重建数据库
3. 检查时间范围是否正确

### uvx 缓存问题
```bash
# 清理缓存
uv cache clean --force

# 使用 --refresh 强制更新
uvx --refresh --from /path/to/codemem codemem-mcp
```

### 性能问题
1. 检查数据库大小：`du -h ~/.codemem/codemem.sqlite`
2. 查看缓存命中率（日志中显示）
3. 考虑定期清理旧数据

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

详细的变更历史请参考 [CHANGELOG.md](CHANGELOG.md)
