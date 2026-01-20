# CodeMem 快速开始

5 分钟快速上手 CodeMem！

## 1. 安装（2 分钟）

### 安装依赖
```bash
pip install pandas pydantic rank-bm25 tiktoken
```

### 安装 CodeMem
```bash
cd /path/to/codemem
pip install -e .
```

## 2. 配置 MCP（1 分钟）

### Claude Code
编辑 `~/.claude/config.json`：
```json
{
  "mcpServers": {
    "codemem": {
      "command": "python",
      "args": ["/path/to/codemem/mcp_server.py", "--db", "~/.codemem/codemem.sqlite"]
    }
  }
}
```

### Codex CLI
```bash
codex mcp add codemem -- python /path/to/codemem/mcp_server.py --db ~/.codemem/codemem.sqlite
```

## 3. 首次使用（2 分钟）

### 启动服务
服务会在首次调用时自动构建数据库（可能需要几分钟）。

### 测试连接
在 Claude Code 或 Codex CLI 中：

```
查看我最近 7 天的活动
```

CodeMem 会自动调用 `activity.recent` 工具返回结果。

## 4. 常用场景

### 场景 1：查看最近活动
```
我最近在做什么？
```
→ 使用 `activity.recent` 工具

### 场景 2：搜索特定内容
```
找到关于 Python 调试的对话
```
→ 使用 `semantic.search` 工具

### 场景 3：查看完整会话
```
查看 session 73133d96 的完整对话
```
→ 使用 `session.get` 工具

### 场景 4：工具使用统计
```
我最常用哪些工具？
```
→ 使用 `tools.usage` 工具

### 场景 5：平台分析
```
我在哪个平台上最活跃？
```
→ 使用 `platform.stats` 工具

## 5. 高级用法

### 自定义 SQL 查询
```
查询最近 30 天内包含 "bug" 的所有消息
```

CodeMem 会使用 `sql.query` 工具执行：
```sql
SELECT timestamp, role, text, source_file
FROM events
WHERE text LIKE '%bug%'
  AND timestamp >= date('now', '-30 days')
ORDER BY timestamp DESC
LIMIT 50;
```

### 使用预览模式
在需要快速查看结果时，可以启用预览：
```json
{
  "name": "sql.query",
  "arguments": {
    "query": "SELECT * FROM events LIMIT 100",
    "preview": true,
    "preview_rows": 10
  }
}
```

## 6. 工具选择指南

```
需要最近活动？
  → activity.recent（最快）

需要特定会话？
  → session.get

需要工具统计？
  → tools.usage

需要平台分析？
  → platform.stats

需要搜索内容？
  → semantic.search（自然语言）

需要复杂查询？
  → sql.query（最后选择）
```

## 7. 性能优化建议

### 优先使用快捷工具
快捷工具比 SQL 查询快 10-20 倍：
- ✅ `activity.recent` - 1 次调用，2 秒
- ❌ 手动 SQL 查询 - 6 次调用，41 秒

### 利用缓存
重复查询会自动命中缓存：
- 第一次查询：2 秒
- 后续查询：< 0.1 秒（99% 节省）

### 使用语义搜索
自然语言搜索比 SQL LIKE 更准确：
- ✅ `semantic.search("Python debugging")` - 智能匹配
- ❌ `WHERE text LIKE '%Python%' AND text LIKE '%debug%'` - 简单匹配

## 8. 故障排除

### 问题：MCP 连接失败
```bash
# 检查依赖
pip list | grep -E "(rank-bm25|tiktoken|pandas|pydantic)"

# 检查 Python 版本
python --version  # 需要 >= 3.10
```

### 问题：查询返回空结果
```bash
# 检查数据库
ls -lh ~/.codemem/codemem.sqlite

# 强制重建
python mcp_server.py --db ~/.codemem/codemem.sqlite --rebuild
```

### 问题：uvx 缓存问题
```bash
# 清理缓存
uv cache clean --force

# 强制刷新
uvx --refresh --from /path/to/codemem codemem-mcp
```

## 9. 下一步

### 阅读完整文档
- [README.md](README.md) - 完整功能说明
- [CHANGELOG.md](CHANGELOG.md) - 版本历史

### 探索高级功能
- 自定义查询模板
- Markdown 文档导出
- 文本工具（rg/sed/awk）

### 参与贡献
- 提交 Issue 报告问题
- 提交 Pull Request 贡献代码
- 分享使用经验

## 10. 获取帮助

### 文档资源
- 查看 MCP 资源：`codemem://query/templates`
- 查看表结构：`codemem://schema/events`
- 查看统计信息：`codemem://stats/summary`

### 社区支持
- GitHub Issues：报告问题和建议
- 讨论区：分享经验和最佳实践

---

**恭喜！** 你已经完成了 CodeMem 的快速入门。开始享受高效的 AI 对话历史管理吧！
