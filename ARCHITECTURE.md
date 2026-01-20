# CodeMem 架构文档

## 概述

CodeMem 是一个高效的 AI 对话历史管理系统，通过 MCP (Model Context Protocol) 为 AI Agent 提供长期记忆能力。本文档详细说明了 CodeMem 的架构设计、实现细节和优化策略。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent                              │
│                  (Claude Code / Codex CLI)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    MCP Server                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Tools      │  │  Resources   │  │   Cache      │      │
│  │  Handler     │  │   Handler    │  │   Manager    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│   SQLite DB    │  │  BM25 Index     │  │  MD Sessions   │
│   (events)     │  │  (semantic)     │  │  (markdown)    │
└────────────────┘  └─────────────────┘  └────────────────┘
```

## 核心组件

### 1. MCP Server (mcp_server.py)

MCP 服务器是 CodeMem 的核心，负责处理所有 MCP 协议请求。

#### 主要功能
- **协议处理** - 实现 MCP 2024-11-05 协议
- **工具调度** - 路由工具调用到对应的处理函数
- **资源管理** - 提供数据库 schema、查询模板等资源
- **缓存管理** - LRU + TTL 双重缓存策略
- **后台构建** - 异步构建数据库，不阻塞启动

#### 关键代码结构
```python
# 全局状态
_db_ready = threading.Event()          # 数据库就绪标志
_bm25_index = None                     # BM25 搜索索引
_query_cache = {}                      # 查询缓存

# 主循环
def main():
    # 1. 解析参数
    # 2. 启动后台构建（如需要）
    # 3. 进入 JSON-RPC 消息循环
    # 4. 处理 initialize/tools/list/tools/call/resources/*
```

### 2. 统一历史处理 (unified_history.py)

负责从不同平台收集和统一对话历史。

#### 支持的平台
- **Claude Code** - `~/.claude/projects/*/conversations/*.jsonl`
- **Codex CLI** - `~/.codex/history/*.jsonl`
- **Cursor** - `~/.cursor/history/*.jsonl`
- **OpenCode** - `~/.opencode/history/*.jsonl`

#### 数据流程
```
原始 JSONL 文件
    ↓
collect_files()  # 扫描文件
    ↓
load_records()   # 解析 JSON
    ↓
UnifiedEvent     # 统一模型
    ↓
to_df()          # 转换为 DataFrame
    ↓
SQLite 数据库
```

#### 关键函数
```python
def collect_files(roots: List[Path]) -> List[Path]:
    """扫描所有平台的历史文件"""

def load_records(files: List[Path]) -> List[UnifiedEvent]:
    """加载并解析 JSONL 文件"""

def to_df(events: List[UnifiedEvent]) -> pd.DataFrame:
    """转换为 DataFrame 用于数据库插入"""
```

### 3. 数据模型 (models.py)

定义统一的事件模型。

#### UnifiedEvent
```python
class UnifiedEvent(BaseModel):
    platform: str              # 平台名称
    session_id: Optional[str]  # 会话 ID
    message_id: Optional[str]  # 消息 ID
    timestamp: Optional[datetime]  # 时间戳
    role: Optional[str]        # 角色 (user/assistant)
    item_type: str             # 类型 (text/tool_use/tool_result)
    text: Optional[str]        # 文本内容
    tool_name: Optional[str]   # 工具名称
    tool_args: Optional[Any]   # 工具参数
    tool_result: Optional[Any] # 工具结果
    raw_json: Dict[str, Any]   # 原始 JSON
```

#### UnifiedEventRow
扩展版本，包含索引优化字段：
- `is_indexable` - 是否可索引
- `index_text` - 优化的索引文本
- `tool_result_summary` - 工具结果摘要

### 4. Markdown 导出 (export_sessions_md.py)

将会话导出为 Markdown 文档。

#### 导出格式
```markdown
# Session: <session_id>

**Platform:** claude
**First Seen:** 2026-01-19 10:00:00
**Last Seen:** 2026-01-19 12:00:00
**Message Count:** 42

---

## Messages

### [user] 2026-01-19 10:00:00
用户消息内容...

### [assistant] 2026-01-19 10:00:05
助手回复内容...
```

## 数据库设计

### events 表（检索视图）

面向检索的干净视图，仅包含可索引内容。

```sql
CREATE VIEW events AS
SELECT
    platform, session_id, message_id, turn_id, item_index,
    timestamp, role, is_meta, agent_id,
    item_type, text, tool_name, source_file
FROM events_raw
WHERE is_indexable = 1;
```

**设计理念：**
- 过滤掉不可索引的内容（如大型工具结果）
- 简化字段，减少查询复杂度
- 优化检索性能

### events_raw 表（完整数据）

```sql
CREATE TABLE events_raw (
    platform TEXT,
    session_id TEXT,
    message_id TEXT,
    turn_id TEXT,
    item_index INTEGER,
    timestamp TEXT,
    role TEXT,
    is_meta INTEGER,
    agent_id TEXT,
    is_indexable INTEGER,
    item_type TEXT,
    text TEXT,
    index_text TEXT,
    tool_name TEXT,
    tool_args TEXT,
    tool_result TEXT,
    tool_result_summary TEXT,
    source_file TEXT,
    raw_json TEXT
);
```

**索引策略：**
```sql
CREATE INDEX idx_timestamp ON events_raw(timestamp);
CREATE INDEX idx_session ON events_raw(session_id);
CREATE INDEX idx_platform ON events_raw(platform);
CREATE INDEX idx_tool ON events_raw(tool_name);
CREATE INDEX idx_indexable ON events_raw(is_indexable);
```

## 工具实现

### 快捷工具设计

快捷工具通过预定义的 SQL 查询和聚合逻辑，避免 Agent 多次试错。

#### activity.recent
```python
def get_recent_activity(conn: sqlite3.Connection, days: int = 7):
    """获取最近活动摘要"""
    # 1. 查询最近 N 天的会话
    # 2. 按会话聚合统计
    # 3. 提取示例消息
    # 4. 返回结构化结果
```

**优化点：**
- 单次查询获取所有信息
- 预聚合统计数据
- 智能采样示例消息

#### session.get
```python
def get_session_details(conn: sqlite3.Connection, session_id: str):
    """获取完整会话历史"""
    # 1. 查询指定会话的所有消息
    # 2. 按时间排序
    # 3. 返回结构化结果
```

**优化点：**
- 直接查询，无需多次调用
- 按时间排序，保持对话连贯性

#### tools.usage
```python
def get_tool_usage(conn: sqlite3.Connection, days: int = 30):
    """获取工具使用统计"""
    # 1. 统计工具使用次数
    # 2. 统计涉及的会话数
    # 3. 获取最后使用时间
    # 4. 按使用次数排序
```

**优化点：**
- 预聚合统计
- 多维度分析（次数、会话数、时间）

#### platform.stats
```python
def get_platform_stats(conn: sqlite3.Connection, days: int = 30):
    """获取平台统计"""
    # 1. 按平台聚合事件数
    # 2. 统计会话数
    # 3. 获取活跃时间范围
```

**优化点：**
- 跨平台对比
- 时间范围分析

### 语义搜索实现

#### BM25 算法

BM25 (Best Matching 25) 是一种概率检索模型，用于文档排序。

**核心公式：**
```
score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
```

其中：
- `D` - 文档
- `Q` - 查询
- `qi` - 查询中的第 i 个词
- `f(qi,D)` - qi 在文档 D 中的频率
- `|D|` - 文档长度
- `avgdl` - 平均文档长度
- `k1`, `b` - 调优参数

#### Tiktoken 分词

使用 OpenAI 的 tiktoken 库进行分词，支持多语言。

```python
def smart_tokenize(text: str) -> List[str]:
    """智能分词"""
    if _tiktoken_encoder is not None:
        # 使用 tiktoken 编码
        token_ids = _tiktoken_encoder.encode(text.lower())
        return [str(tid) for tid in token_ids]
    else:
        # 降级到简单分词
        return text.lower().split()
```

**优势：**
- 支持中英文混合
- 与 GPT 模型一致的分词
- 更准确的语义匹配

#### 索引构建

```python
def build_bm25_index(db_path: Path):
    """构建 BM25 索引"""
    # 1. 从数据库加载可索引文本
    # 2. 使用 smart_tokenize 分词
    # 3. 构建 BM25Okapi 索引
    # 4. 保存元数据（session_id, role, platform 等）
```

**优化点：**
- 仅索引 `is_indexable=1` 的内容
- 使用 `index_text` 字段（优化后的文本）
- 后台异步构建，不阻塞启动

#### 搜索执行

```python
def bm25_search(query: str, limit: int = 20):
    """执行 BM25 搜索"""
    # 1. 等待索引构建完成
    # 2. 对查询分词
    # 3. 计算 BM25 分数
    # 4. 返回 Top-K 结果
```

## 缓存系统

### 设计目标
- 减少重复查询的计算成本
- 支持 LRU 淘汰策略
- 支持 TTL 过期机制

### 实现细节

```python
_query_cache: Dict[str, Tuple[Any, float]] = {}
_cache_max_size = 100
_cache_ttl = 3600  # 1 小时

def get_from_cache(key: str) -> Optional[Any]:
    """从缓存获取"""
    if key in _query_cache:
        result, timestamp = _query_cache[key]
        if time.time() - timestamp < _cache_ttl:
            return result
        else:
            del _query_cache[key]  # 过期删除
    return None

def put_to_cache(key: str, value: Any):
    """写入缓存"""
    if len(_query_cache) >= _cache_max_size:
        # LRU 淘汰：删除最旧的条目
        oldest_key = min(_query_cache.keys(),
                        key=lambda k: _query_cache[k][1])
        del _query_cache[oldest_key]
    _query_cache[key] = (value, time.time())
```

### 缓存键生成

```python
def cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    """生成缓存键"""
    # 使用工具名 + 参数哈希
    args_str = json.dumps(args, sort_keys=True)
    return f"{tool_name}:{hashlib.md5(args_str.encode()).hexdigest()}"
```

## 性能优化

### 1. 后台构建

数据库构建在后台线程中进行，不阻塞 MCP 服务器启动。

```python
def build_db_background(db_path, ...):
    """后台构建数据库"""
    try:
        # 1. 收集文件
        # 2. 加载记录
        # 3. 写入数据库
        # 4. 构建索引
        # 5. 导出 Markdown
        _db_ready.set()  # 标记完成
    except Exception as e:
        _db_build_error = str(e)
```

### 2. 查询限制

所有查询都有行数限制，避免返回过大结果。

```python
# sql.query 默认限制 100 行
limit = min(limit, 50)  # 最大 50 行

# semantic.search 默认限制 20 条
limit = min(limit, 50)  # 最大 50 条
```

### 3. 预览模式

sql.query 支持预览模式，减少不必要的 token 消耗。

```python
if preview:
    # 仅返回前 N 行的简短摘要
    text = sql_preview_text(result, max_rows=preview_rows)
else:
    # 返回 "ok"，节省 token
    text = "ok"
```

### 4. 索引优化

数据库使用多个索引加速查询。

```sql
-- 时间范围查询
CREATE INDEX idx_timestamp ON events_raw(timestamp);

-- 会话查询
CREATE INDEX idx_session ON events_raw(session_id);

-- 平台过滤
CREATE INDEX idx_platform ON events_raw(platform);

-- 工具统计
CREATE INDEX idx_tool ON events_raw(tool_name);

-- 可索引过滤
CREATE INDEX idx_indexable ON events_raw(is_indexable);
```

## 安全性

### 1. SQL 注入防护

使用参数化查询，避免 SQL 注入。

```python
# ✅ 安全
cursor.execute("SELECT * FROM events WHERE session_id = ?", (session_id,))

# ❌ 不安全
cursor.execute(f"SELECT * FROM events WHERE session_id = '{session_id}'")
```

### 2. 只读查询

sql.query 工具仅允许只读查询。

```python
def is_readonly_query(query: str) -> bool:
    """检查是否为只读查询"""
    # 允许 SELECT、WITH (CTE)、PRAGMA
    # 拒绝 INSERT、UPDATE、DELETE、DROP 等
```

### 3. 路径限制

text.shell 工具限制路径在 md_sessions 目录内。

```python
def resolve_path(path: str) -> Optional[Path]:
    """解析并验证路径"""
    resolved = (MD_SESSIONS_DIR / path).resolve()
    if not resolved.is_relative_to(MD_SESSIONS_DIR):
        return None  # 拒绝路径穿越
    return resolved
```

### 4. 命令白名单

text.shell 仅允许 rg/sed/awk 命令。

```python
ALLOWED_COMMANDS = ["rg", "sed", "awk"]

if cmd not in ALLOWED_COMMANDS:
    return {"error": "Command not allowed"}
```

## 扩展性

### 添加新平台

1. 在 `unified_history.py` 中添加文件扫描逻辑
2. 实现平台特定的解析函数
3. 映射到 `UnifiedEvent` 模型

```python
def parse_newplatform(file_path: Path) -> List[UnifiedEvent]:
    """解析新平台的历史文件"""
    events = []
    # 解析逻辑
    return events
```

### 添加新工具

1. 在 `mcp_server.py` 中定义工具 schema
2. 实现工具处理函数
3. 在 `tools/call` 中添加路由

```python
# 1. 定义 schema
{
    "name": "new.tool",
    "description": "新工具描述",
    "inputSchema": {...}
}

# 2. 实现函数
def handle_new_tool(conn, args):
    """处理新工具"""
    return {"result": ...}

# 3. 添加路由
if name == "new.tool":
    result = handle_new_tool(get_conn(), args)
    respond(msg_id, {"content": [...], "data": result})
```

### 添加新资源

1. 在 `RESOURCES` 列表中添加资源定义
2. 在 `resources/read` 中实现读取逻辑

```python
# 1. 定义资源
RESOURCES.append({
    "uri": "codemem://new/resource",
    "name": "新资源",
    "description": "资源描述",
    "mimeType": "text/markdown"
})

# 2. 实现读取
if uri == "codemem://new/resource":
    text = generate_new_resource()
    respond(msg_id, {"contents": [{"uri": uri, "text": text}]})
```

## 测试

### 单元测试

```bash
# 测试查询功能
python test_query.py
```

### 集成测试

```bash
# 启动服务器并测试
python mcp_server.py --db test.sqlite &
# 发送 JSON-RPC 请求测试
```

### 性能测试

```python
import time

# 测试查询性能
start = time.time()
result = sql_query(conn, "SELECT * FROM events LIMIT 1000")
print(f"Query time: {time.time() - start:.2f}s")

# 测试缓存命中率
print(f"Cache hits: {_cache_hits}")
print(f"Cache misses: {_cache_misses}")
print(f"Hit rate: {_cache_hits / (_cache_hits + _cache_misses) * 100:.1f}%")
```

## 部署

### 开发环境

```bash
pip install -e .
python mcp_server.py --db ~/.codemem/codemem.sqlite
```

### 生产环境

```bash
# 使用 uvx
uvx --from /path/to/codemem codemem-mcp --db /path/to/codemem.sqlite

# 或使用 venv
python3 -m venv /opt/codemem-venv
/opt/codemem-venv/bin/pip install /path/to/codemem
/opt/codemem-venv/bin/codemem-mcp --db /var/lib/codemem/codemem.sqlite
```

### 监控

```bash
# 查看日志
tail -f mcp.log

# 查看数据库大小
du -h ~/.codemem/codemem.sqlite

# 查看缓存统计
# 在日志中搜索 "Cache hit" 和 "Cache miss"
```

## 故障排除

### 常见问题

1. **数据库构建失败**
   - 检查文件权限
   - 检查磁盘空间
   - 查看错误日志

2. **查询性能慢**
   - 检查索引是否存在
   - 减少查询范围
   - 使用快捷工具

3. **缓存不生效**
   - 检查缓存配置
   - 查看缓存命中率
   - 考虑增加缓存大小

## 未来优化

### 短期（0.2.0）
- 增量索引更新
- 持久化缓存
- 更多聚合视图

### 中期（0.3.0）
- 向量搜索支持
- 多数据库支持
- Web UI

### 长期（1.0.0）
- 分布式部署
- 实时同步
- 高级分析功能

## 参考资料

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Tiktoken Documentation](https://github.com/openai/tiktoken)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
