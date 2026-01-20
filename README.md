日期：2025-09-07
标题：初始化 CodeMem 聊天历史统一抽取

背景：
需要将 Claude Code 与 Codex CLI 的本机聊天历史统一为一张表，便于 SQL 查询与长期记忆。

过程：
1) 统一最小颗粒度为 content item（text/tool_use/tool_result/thinking/image）。
2) 使用单表字段承载跨平台事件，并保留 `raw_json` 兜底。
3) 默认查询只关注 `is_indexable=true`；工具结果仅索引 `tool_result_summary`，原始内容保留在 `raw_json`。
4) 提供 Python 脚本可直接扫描本机默认路径并导出 parquet/csv。

结果：
形成 CodeMem 独立实现。

复盘：
单表方案对查询友好，但需要更多字段过滤；建议在后续引入 view 或派生表。

示例：
```sql
-- 面表：面向 agent 的干净检索表
select timestamp, role, text, source_file
from events
where text like '%关键词%'
order by timestamp desc
limit 20;
```

```sql
-- 仅检索可索引文本，避免工具输出污染
select timestamp, role, text, source_file
from events_raw
where is_indexable = true
  and index_text like '%关键词%'
order by timestamp desc
limit 20;
```

MCP 启动：
```bash
uvx --from ./codemem codemem-mcp
```

参考来源：
- 本项目 AGENTS.md：解析流程摘要。

更新原因：新建 CodeMem 的独立实现与文档入口。

日期：2025-09-07
标题：修复 MCP 本地安装失败

背景：
使用 `uvx --from /Users/zdzd/CodeMem/codemem codemem-mcp` 启动时，setuptools 报 "Multiple top-level modules discovered"。

过程：
1) 在 `pyproject.toml` 明确声明 `py-modules`，避免自动发现。
2) 将 entrypoint 改为顶层模块 `mcp_server:main`。
3) 补齐 `build-system` 配置，固定构建后端。

结果：
`uvx --from /Users/zdzd/CodeMem/codemem codemem-mcp` 可正常构建并启动。

复盘：
单文件模块布局适合 `py-modules`；若后续需要扩展，可迁移为 `src/` 布局。

参考来源：
- 本次 MCP 启动错误输出。

更新原因：记录 MCP 本地安装失败的打包修复方案。

日期：2026-01-19
标题：修复 MCP 启动时通知报错

背景：
MCP client 启动报 “tools/list failed: Transport closed”，手工 JSON-RPC 调用里出现对 `notifications/initialized` 的 error 响应。

过程：
1) MCP 服务器收到无 `id` 的通知时直接忽略，避免返回 error。
2) 若仍报错，说明 uvx 复用了旧构建缓存，需要刷新构建结果。
3) 若刷新后仍命中旧缓存，清理 uv cache 后再启动。

结果：
通知不再触发 error，`tools/list` 可正常返回。

复盘：
本地路径包修改后，需显式刷新 uvx 构建缓存以避免旧行为残留。

命令：
```bash
uvx --refresh --from /Users/zdzd/CodeMem/codemem codemem-mcp --db /Users/zdzd/CodeMem/.codemem/codemem.sqlite
```
```bash
uv cache clean --force
```

参考来源：
- 本次 MCP 启动报错输出。

更新原因：记录通知报错的原因与 uvx 缓存刷新方式。

更新原因：补充 uv cache 清理兜底步骤。

日期：2026-01-19
标题：规避 uvx 缓存导致的旧版本运行

背景：
本地改动 MCP 代码后，uvx 仍可能命中旧缓存导致行为未更新。

过程：
1) 通过 bump `pyproject.toml` 的版本号触发 uvx 重新构建。
2) 或使用 `--with-editable` 直接指向本地源码，绕过缓存。

结果：
本地改动可稳定生效，不再反复清理缓存。

复盘：
版本号递增是最稳妥的长期方案；临时调试更适合 editable。

命令：
```bash
uvx --from /Users/zdzd/CodeMem/codemem codemem-mcp --db /Users/zdzd/CodeMem/.codemem/codemem.sqlite
```
```bash
uvx --from /Users/zdzd/CodeMem/codemem --with-editable /Users/zdzd/CodeMem/codemem codemem-mcp --db /Users/zdzd/CodeMem/.codemem/codemem.sqlite
```

参考来源：
- 本次 MCP 启动缓存问题排查。

更新原因：补充 uvx 缓存规避的长期方案。

日期：2026-01-19
标题：补充 MCP resources/list 支持

背景：
Codex 调用 `resources/list` 获取资源列表时返回 “unknown method”，导致 MCP 资源不可见。

过程：
1) 在 MCP 服务器内实现 `resources/list` 与 `resources/read`。
2) 提供 `events` 与 `events_raw` 的 schema 资源，返回 Markdown 格式内容。
3) 在 `initialize` 能力中声明 `resources`。

结果：
`resources/list` 可用，`resources/read` 可读取表结构，`list_mcp_resources` 不再报错。

复盘：
对 SQL-only MCP 服务器，最小资源集即可满足客户端探测需求。

参考来源：
- 本次 MCP `resources/list` 报错输出。

更新原因：记录 `resources/list` 支持与 schema 资源说明。

日期：2026-01-19
标题：避免 SQL 错误导致 MCP 断连

背景：
MCP 调用 `tools/call` 时若 SQL 出错（如字段不存在），服务器抛异常退出，客户端报 “Transport closed”。

过程：
1) 为 `sql_query` 增加 sqlite 异常捕获。
2) SQL 执行失败时返回 `error` 字段而不是退出进程。
3) `tools/call` 若命中 `error`，追加 `isError=true` 让 agent 直接感知失败。

结果：
SQL 错误不会中断 MCP 连接，客户端可收到明确错误信息并感知失败。

复盘：
工具层应对用户输入错误做兜底，并显式上报失败，避免影响通道可用性。

重启说明：
修改 MCP 代码后需重启 Codex CLI 会话，确保新进程加载最新实现。

参考来源：
- 本次 MCP SQL 报错与 “Transport closed” 输出。
- 本次 `sql_query` 本地复现日志（2026-01-19）。

更新原因：记录 MCP 在 SQL 失败时的稳定性改进与错误上报；补充参考来源与重启说明（本次补记）。

日期：2026-01-19
标题：在 sandbox 下改用本地 venv 启动 MCP

背景：
Codex MCP 在 sandbox 中启动 `uvx` 会触发 `system-configuration` 的 panic，导致 “Transport closed”。

过程：
1) 在工作区创建独立 venv，并安装 `codemem-mcp` 依赖。
2) 将 MCP 命令改为 venv 的 `python` 直接执行 `mcp_server.py`，绕开 `uvx`。
3) 修改 MCP 配置后需重启 Codex CLI 会话，旧会话不会热更新。
4) 若仍要尝试 `uvx`，在 `codemem/uv.toml` 固定 uv 行为，并用 `--directory` 读取配置（仍可能受 sandbox 限制）。

结果：
sandbox 环境下可稳定启动 MCP，`tools/call` 不再断连。

复盘：
uvx 在受限环境可能依赖系统配置服务；用 venv 直跑更可靠。

重启说明：
修改 MCP 配置后需重启 Codex CLI 会话，旧会话不会热更新。

日期：2026-01-19
标题：平滑 text.shell 的 md_sessions 路径输入

背景：
text.shell 对路径限制为 `md_sessions`，但用户传入 `md_sessions` 或项目根目录时会被错误解析或直接拒绝，体验不顺。

过程：
1) 兼容 `md_sessions`、`./md_sessions` 与空路径，统一映射到 `MD_SESSIONS_DIR`。
2) 若传入 `md_sessions` 的父目录（项目根），自动回退到 `md_sessions`。
3) 保持安全边界不变，仅调整解析与错误提示，提示“use md_sessions or omit paths”。

结果：
text.shell 的路径输入更宽容，常见用法不再触发 “path outside md_sessions” 或重复拼接。

复盘：
工具型 MCP 的路径约束需要配合易用的别名与提示，避免阻塞基础验证流程。

参考来源：
- 本次 text.shell 路径报错与修复。

更新原因：记录 text.shell 路径容错改进，优化使用体验。

日期：2026-01-19
标题：统一 MCP 工具返回为机器友好结构

背景：
当前 tools/call 的 `content.text` 是 JSON 字符串，agent 需二次解析；路径与元信息也不够显式。

过程：
1) `content` 置空，避免任何自然语言提示，输出仅保留结构化 `data`。
2) `data` 保持结构化结果，并补充 `md_root` 与 `resolved_paths` 以消除路径歧义。
3) 在 inputSchema 中补充说明，降低调用端猜测成本。

结果：
agent 只消费 `data` 字段，无需额外 JSON 解析，路径解析结果可被日志/调试直接复用。

复盘：
MCP 工具输出应以机器消费为默认，必要时由调用端自行渲染提示。

参考来源：
- 本次 MCP 工具返回结构调整。

更新原因：降低 agent 解析成本，明确路径与元数据输出。

命令：
```bash
python3 -m venv /Users/zdzd/CodeMem/.venv-codemem
/Users/zdzd/CodeMem/.venv-codemem/bin/pip install -e /Users/zdzd/CodeMem/codemem
```
```bash
codex mcp remove codemem
codex mcp add codemem -- /Users/zdzd/CodeMem/.venv-codemem/bin/python /Users/zdzd/CodeMem/codemem/mcp_server.py --db /Users/zdzd/CodeMem/.codemem/codemem.sqlite
```
```bash
cat <<'EOF' > /Users/zdzd/CodeMem/codemem/uv.toml
[tool.uv]
cache-dir = "/Users/zdzd/CodeMem/.uv-cache"
no-env-file = true
EOF
```
```bash
codex mcp remove codemem
codex mcp add codemem -- uvx --directory /Users/zdzd/CodeMem/codemem --from /Users/zdzd/CodeMem/codemem codemem-mcp --db /Users/zdzd/CodeMem/.codemem/codemem.sqlite
```

参考来源：
- 本次 sandbox 环境 uvx panic 输出。
- 本次 Codex CLI MCP 配置重启验证记录（2026-01-19）。
- 本次 uv.toml 配置尝试记录（2026-01-19）。

更新原因：记录 sandbox 下 uvx 崩溃的规避方案；补充参考来源、重启说明与 uv.toml 配置（本次补记）。

日期：2026-01-19
标题：新增会话 Markdown 文档库导出入口

背景：
需要将每个会话模板化为单独 Markdown 文件，并汇总到一个目录中，便于 grep/sed/awk 快速检索。

过程：
1) 提供导出脚本，按 session_id 生成 Markdown 文件。
2) 统一文件模板，包含元数据与按时间排序的对话内容。
3) 给出常用命令入口，直接面向文本检索工具。

结果：
导出命令：
```bash
python /Users/zdzd/CodeMem/codemem/export_sessions_md.py \
  --db /Users/zdzd/CodeMem/.codemem/codemem.sqlite \
  --out /Users/zdzd/CodeMem/md_sessions
```

grep/sed/awk 入口：
```bash
rg -n "关键词" /Users/zdzd/CodeMem/md_sessions
```
```bash
sed -n '1,120p' /Users/zdzd/CodeMem/md_sessions/<session_id>.md
```
```bash
awk '/^### /{print $0}' /Users/zdzd/CodeMem/md_sessions/<session_id>.md
```

复盘：
Markdown 模板化把 SQL 的“结构性”落到文件系统里，能用最简单的文本工具快速定位内容。

参考来源：
- `codemem/export_sessions_md.py`：会话导出实现。
- `codemem/mcp_server.py`：events_raw 数据生成逻辑。

更新原因：补齐会话 Markdown 文档库的导出入口与检索命令。

日期：2026-01-19
标题：暴露会话 Markdown 资源到 MCP

背景：
需要通过 MCP 直接访问会话 Markdown 文档库，便于在工具侧读取与检索。

过程：
1) 在 `resources/list` 中增加 sessions 索引资源。
2) 在 `resources/read` 中支持读取 sessions 索引与单个 Markdown 文件。
3) 防止路径穿越，仅允许读取 `md_sessions` 目录下文件。

结果：
新增资源：
- `codemem://sessions/index`：列出 Markdown 会话文件列表。
- `codemem://sessions/<filename>`：读取单个会话 Markdown。

复盘：
资源侧暴露索引即可串起“列出 → 读取”的工作流，同时避免一次性返回大量文件。

参考来源：
- `codemem/mcp_server.py`：resources/list 与 resources/read 扩展实现。
- `codemem/export_sessions_md.py`：会话 Markdown 生成脚本。

更新原因：让会话 Markdown 文档库可通过 MCP 资源接口读取。

日期：2026-01-19
标题：新增 MCP 文本工具入口（rg/sed/awk）

背景：
需要让 agent 直接通过 MCP 使用 grep/sed/awk 与会话 Markdown 文档库交互。

过程：
1) 增加 `text.shell` 工具，仅允许 `rg/sed/awk`。
2) 强制路径限制在 `md_sessions` 目录内，避免路径穿越。
3) 返回 stdout/stderr/returncode，并对输出做截断保护。

结果：
工具用法示例：
```json
{"cmd": "rg", "args": ["-n", "关键词"], "paths": []}
```
```json
{"cmd": "sed", "args": ["-n", "1,120p"], "paths": ["<session_id>.md"]}
```
```json
{"cmd": "awk", "args": ["/^### /{print $0}"], "paths": ["<session_id>.md"]}
```

复盘：
受限命令 + 目录约束可以覆盖大部分检索需求，同时降低安全风险。

参考来源：
- `codemem/mcp_server.py`：text.shell 工具实现。

更新原因：补齐 MCP 侧 grep/sed/awk 入口。

日期：2026-01-19
标题：补充 MCP 查询模板与表语义说明

背景：
MCP 可用，但查询模板与 `events`/`events_raw` 的语义说明不足，`index_text` 的使用也不够直观。

过程：
1) 明确 `events` 面向检索、`events_raw` 面向追溯的定位。
2) 补充 `index_text` 的索引策略与适用场景。
3) 添加常用 SQL 模板，覆盖关键词、时间窗口、会话聚合与角色过滤。

结果：
表语义说明：
- `events`：面向检索的干净视图，仅保留可索引内容。
- `events_raw`：底表，保留原始字段与工具结果，适合追溯与排错。
- `index_text`：优先用于检索，避免工具输出污染；仅当 `is_indexable = 1` 才可靠。

常用查询模板：
```sql
-- 关键词 + 时间窗口
select timestamp, role, text, source_file
from events
where text like '%关键词%'
  and timestamp >= '2026-01-01'
order by timestamp desc
limit 50;
```

```sql
-- 会话内聚合：按 session_id 统计消息量
select session_id, count(*) as cnt
from events
group by session_id
order by cnt desc
limit 20;
```

```sql
-- 只查可索引文本，规避工具输出噪声
select timestamp, role, index_text, source_file
from events_raw
where is_indexable = 1
  and index_text like '%关键词%'
order by timestamp desc
limit 50;
```

```sql
-- 角色过滤：只看 assistant 回复
select timestamp, text, source_file
from events
where role = 'assistant'
order by timestamp desc
limit 50;
```

复盘：
查询入口越明确越省沟通成本，保持 `events`/`index_text` 的语义清晰是关键。

参考来源：
- `codemem/mcp_server.py`：`events` 生成逻辑与 `index_text` 策略。
- `codemem/unified_history.py`：`index_text` 生成逻辑。
- 本次 MCP 查询验证（2026-01-19）。

更新原因：补齐查询模板与表语义说明，降低 MCP 使用门槛。

日期：2026-01-19
标题：验证 MCP 资源与文本工具

背景：
此前 Codex CLI 会话内 MCP 资源为空、`sql.query` 报 “Transport closed”，需要在新会话内复核资源与工具是否正常。

过程：
1) 调用 `resources/list` 检查资源列表。
2) 调用 `sql.query` 执行简单计数查询。
3) 调用 `text.shell` 在 `md_sessions` 内执行 `rg` 验证命令通路。

结果：
- 资源列表返回 `events`、`events_raw` 与 `sessions index`。
- `sql.query` 正常返回结果。
- `text.shell` 正常返回 `rg` 输出。

复盘：
当前会话 MCP 功能正常；若 CLI 仍出现空资源或断连，应优先核对 MCP 启动命令与会话重启状态。

参考来源：
- 本次 MCP 调用结果（`resources/list`、`sql.query`、`text.shell`）。

更新原因：记录 MCP 资源与文本工具的验证结果。

日期：2026-01-19
标题：补充 MCP 使用体验说明

背景：
MCP 已可用，但 `sql.query` 的返回是 JSON 文本，初次使用容易误以为是结构化对象。

过程：
1) 补充 `sql.query` 的返回示例，明确需要再次解析 JSON。
2) 结合 `resources/read` 示例，强化 schema 发现路径。
3) 标注 `tools/call` 同步返回结构化 `data`、`ok`、`error`、`meta` 字段，便于 agent 直接消费。

结果：
`sql.query` 返回示例：
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"columns\":[\"n\"],\"rows\":[[7764]]}"
    }
  ],
  "data": {
    "columns": ["n"],
    "rows": [[7764]]
  },
  "ok": true,
  "meta": {
    "limit_applied": 100,
    "row_count": 1
  }
}
```

`resources/read` 示例：
```json
{"uri": "codemem://schema/events"}
```

复盘：
清楚标注“返回为 JSON 文本”并提供结构化 `data` 与 `ok/error/meta`，可以降低误解与解析成本。

参考来源：
- `codemem/mcp_server.py`：`tools/call` 与 `resources/read` 的响应格式。
- 本次 MCP 交互验证（2026-01-19）。

更新原因：补充 MCP 使用体验说明与返回示例；增加结构化 `data/ok/error/meta` 字段说明。

日期：2026-01-19
标题：记忆恢复摘要（MCP 结构化响应）

背景：
需要在 MCP 层面提升 agent 消费体验，并记录本次验证结论与现状。

过程：
1) `tools/call` 响应统一增加 `data/ok/error/meta`，避免额外 JSON 解析。
2) `resources/read` 为表结构返回 `data`（字段结构化），便于 agent 直接使用。
3) 本会话已验证 `resources/list`、`resources/read`、`sql.query`、`text.shell` 正常。

结果：
- MCP 资源与工具可用，`sql.query` 返回正常。
- agent 可直接读取结构化 `data`，错误可用 `ok/error/isError` 识别。

复盘：
明确结构化响应与错误语义，可显著降低 agent 解析成本与歧义。

参考来源：
- `codemem/mcp_server.py`：`tools/call` 与 `resources/read` 的响应结构。
- 本次 MCP 调用验证（2026-01-19）。

更新原因：补充记忆恢复摘要，记录结构化响应与验证状态。

日期：2026-01-19
标题：修复 MCP 工具响应类型

背景：
Codex 调用 `tools/call` 报 “Unexpected response type”，`sql.query` 与 `text.shell` 无法返回结果。

过程：
1) 保持结构化 `data` 输出不变。
2) 在 `content` 中补充最小可读文本，替代原有 JSON 字符串，确保响应满足 MCP 解析要求。

结果：
`sql.query` 与 `text.shell` 可正常返回，`content.text` 变为简要摘要，Codex 不再报响应类型错误。

复盘：
MCP 工具响应应兼顾结构化数据与最低限度的可读内容，避免客户端拒收。

参考来源：
- `codemem/mcp_server.py`：tools/call 响应修复点。

更新原因：记录 MCP 工具响应类型修复与结果。

日期：2026-01-19
标题：保留仅机器消费并兼容 MCP 响应

背景：
`tools/call` 需要 `content` 字段，否则 Codex 报 “Unexpected response type”；但 `content` 里的人类可读摘要违背“仅机器消费”目标。

过程：
1) 保留结构化 `data/ok/error/meta` 作为唯一有效载体。
2) `content` 仅返回空字符串占位，满足 MCP 响应格式要求。

结果：
Codex 可正常解析工具响应，同时 `content` 不再包含人类摘要。

复盘：
在必须字段上用最小占位，避免影响机器消费语义。

参考来源：
- `codemem/mcp_server.py`：tools/call `content` 占位实现。

更新原因：确保 MCP 响应兼容，同时保留仅机器消费原则。

日期：2026-01-19
标题：恢复 MCP 工具输出可读摘要

背景：
Codex CLI 只展示 `content.text`，`data` 不可见，导致 `sql.query`/`text.shell` 看起来无返回。

过程：
1) `content.text` 输出简短 JSON 摘要（`sql.query` 行数与前 20 行，`text.shell` stdout/stderr 前 2000 字符）。
2) `data` 结构化结果保持不变。

结果：
CLI 可直接看到可读输出，结构化 `data` 仍用于机器消费。

复盘：
在不破坏结构化语义的前提下提供最小可读信息，避免“空返回”误判。

参考来源：
- `codemem/mcp_server.py`：`tool_text_summary` 实现。

更新原因：修复 Codex CLI 工具输出为空的问题。

日期：2026-01-19
标题：简化 MCP 工具输出文本

背景：
需要保持 `content.text` 仅作占位，但希望明确成功状态。

过程：
1) 成功时固定返回 `ok`；失败时返回错误文本。
2) `data` 结构化结果保持不变。

结果：
CLI 可感知成功与失败，结构化输出不受影响。

复盘：
最小可读文本能减少误解，同时避免输出过多内容。

参考来源：
- `codemem/mcp_server.py`：`tool_text_summary` 实现。

更新原因：按需将 `content.text` 简化为 `ok`。

日期：2026-01-19
标题：增加 sql.query 预览开关

背景：
`content.text` 固定为 `ok` 省 token，但会导致查询时看不到任何结果预览。

过程：
1) 为 `sql.query` 增加 `preview` 参数，默认 `false`。
2) 当 `preview=true` 时，`content.text` 输出列名与少量行预览。
3) 支持 `preview_rows` 与 `preview_cell_len` 控制预览行数与单元格长度（范围限制：`preview_rows` 1–50，`preview_cell_len` 10–200）。
3) `data` 结构化结果保持不变，仍是唯一可机器消费的主体。

结果：
需要时可显式请求简短预览，不需要时维持最小输出。

复盘：
通过显式开关兼顾可读性与 token 成本。

参考来源：
- `codemem/mcp_server.py`：`sql.query` `preview` 与预览文本实现。

更新原因：为 `sql.query` 提供可控预览，避免盲查且不默认耗 token。

更新原因：增加预览大小参数，便于精细控制 token。

更新原因：为预览参数增加范围限制，避免异常或过大输出。

示例：
```json
{"name":"sql.query","arguments":{"query":"select timestamp, role, text from events order by timestamp desc","preview":true,"preview_rows":8,"preview_cell_len":60}}
```

日期：2026-01-19
标题：放宽只读 SQL 规则

背景：
仅允许 `SELECT` 会拦截 CTE 与部分只读 `PRAGMA`，但仍需保持只读边界。

过程：
1) 放行以 `WITH` 开头的 CTE 查询。
2) 增加只读 `PRAGMA` 白名单（schema/索引/列表类）。
3) 增加写操作关键字拦截，并兼容前置注释。

结果：
只读查询更灵活，仍避免写操作进入 MCP。

复盘：
用白名单与写关键字拦截折中安全与可用性。

参考来源：
- 本次对话需求：放开 CTE 与只读 PRAGMA。

更新原因：支持只读 CTE/PRAGMA，同时保持只读约束。

# CodeMem
