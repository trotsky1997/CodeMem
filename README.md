# CodeMem

**CodeMem** is an MCP (Model Context Protocol) server that provides semantic search and querying capabilities over your AI chat conversation history. It enables AI assistants like Claude to search through past conversations, retrieve relevant context, and answer questions about your interaction history.

## Features

- 🔍 **Semantic Search**: BM25-based semantic search with smart tokenization (supports English and Chinese)
- 💾 **SQL Queries**: Direct SQL access for complex data analysis and custom queries
- 🎯 **Regex Search**: Pattern-based search for precise matching
- ⚡ **High Performance**: Async I/O, connection pooling, and query caching
- 📝 **Markdown Export**: Exports conversations to readable Markdown files
- 🔄 **Multi-format Support**: Handles JSON, JSONL, and various chat history formats

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Install from Source

1. Clone the repository:
```bash
git clone <repository-url>
cd CodeMem
```

2. Install dependencies using uv (recommended):
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -e .
```

### Dependencies

The following packages will be installed automatically:
- `pydantic>=2.0.0` - Data validation
- `rank-bm25>=0.2.2` - BM25 search algorithm
- `tiktoken>=0.5.0` - Smart tokenization
- `aiosqlite>=0.22.0` - Async SQLite operations
- `mcp>=0.9.0` - Model Context Protocol

## Usage

### 1. Configure MCP Client

Add CodeMem to your MCP client configuration (e.g., Claude Desktop):

**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "codemem": {
      "command": "python",
      "args": [
        "/path/to/CodeMem/mcp_server.py",
        "--db-path", "/path/to/your/chat_history.sqlite"
      ]
    }
  }
}
```

**Command-line options:**
- `--db-path`: Path to SQLite database (default: `~/.codemem/chat_history.sqlite`)
- `--data-dir`: Directory containing chat history files to import

### 2. Prepare Your Chat History

CodeMem can import chat history from various sources:

```bash
# Import from a directory containing JSON/JSONL files
python mcp_server.py --data-dir /path/to/chat/logs --db-path ~/.codemem/chat_history.sqlite
```

The server will:
1. Scan the data directory for chat history files
2. Build a SQLite database with indexed conversations
3. Export sessions to Markdown files in `~/.codemem/md_sessions/`
4. Build BM25 search indexes

### 3. Use the Tools

Once configured, your AI assistant can use three main tools:

#### **semantic.search** - Semantic Search
Search conversation history using natural language queries:

```python
# Example query from Claude
semantic.search(
    query="How do I implement authentication?",
    top_k=10,
    mode="refs"  # Options: summary, refs, preview, full
)
```

**Modes:**
- `summary`: Returns statistics and top 3 samples (most context-efficient)
- `refs`: Returns reference IDs and metadata only (recommended)
- `preview`: Returns first 100 characters preview
- `full`: Returns complete content (use sparingly)

#### **sql.query** - SQL Queries
Execute SQL queries for complex analysis:

```python
# Example queries
sql.query(
    query="SELECT * FROM events WHERE text LIKE '%authentication%' LIMIT 10",
    mode="summary"
)

sql.query(
    query="SELECT session_id, COUNT(*) as msg_count FROM events GROUP BY session_id ORDER BY msg_count DESC",
    mode="full"
)
```

**Common queries:**
- `SELECT * FROM events WHERE role='user' LIMIT 10` - Get user messages
- `SELECT COUNT(*) FROM events WHERE text LIKE '%keyword%'` - Count matches
- `SELECT session_id, MIN(timestamp) as start_time FROM events GROUP BY session_id` - Session stats

#### **regex.search** - Pattern Matching
Search using regular expressions:

```python
# Example patterns
regex.search(
    pattern=r"async def \w+\(.*\):",  # Find async function definitions
    mode="summary"
)

regex.search(
    pattern=r"https?://\S+",  # Find URLs
    mode="preview"
)
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client (Claude)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ MCP Protocol
┌───────────────────────────▼─────────────────────────────────┐
│                    CodeMem MCP Server                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Semantic   │  │  SQL Query   │  │    Regex     │     │
│  │    Search    │  │   Engine     │  │   Search     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐    │
│  │           SQLite Database + BM25 Index             │    │
│  │  • events table (session_id, role, text, etc.)    │    │
│  │  • BM25 index for semantic search                 │    │
│  │  • Query cache for performance                    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Data Sources & Exports                          │
│  • JSON/JSONL chat logs                                     │
│  • Markdown exports (~/.codemem/md_sessions/)              │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

1. **Initialization**:
   - Server starts and loads configuration
   - Scans data directory for chat history files
   - Parses JSON/JSONL files and normalizes data

2. **Database Building**:
   - Creates SQLite database with `events` table
   - Stores: session_id, timestamp, role, content, metadata
   - Builds indexes for fast querying

3. **Markdown Export**:
   - Exports each session to a formatted Markdown file
   - Stored in `~/.codemem/md_sessions/`
   - Used for BM25 search indexing

4. **BM25 Index Building**:
   - Tokenizes Markdown content using tiktoken
   - Builds BM25Okapi index for semantic search
   - Supports multilingual tokenization

5. **Query Processing**:
   - Receives tool calls from MCP client
   - Checks query cache for recent results
   - Executes search/query with appropriate strategy
   - Returns results in requested format

### Data Model

The SQLite database uses the following schema:

```sql
CREATE TABLE events (
    session_id TEXT,      -- Unique session identifier
    timestamp INTEGER,    -- Unix timestamp
    role TEXT,           -- 'user' or 'assistant'
    text TEXT,           -- Message content
    content_json TEXT,   -- Full content structure (JSON)
    source TEXT,         -- Source file path
    content_hash TEXT    -- Deduplication hash
);
```

### Performance Optimizations

- **Async I/O**: Non-blocking database operations with aiosqlite
- **Connection Pooling**: Reuses database connections
- **Query Caching**: 1-hour TTL cache for repeated queries (max 100 entries)
- **Parallel Processing**: Multi-threaded index building
- **Smart Tokenization**: tiktoken-based tokenization for better search quality

## Configuration

### Environment Variables

- `CODEMEM_DB_PATH`: Default database path (default: `~/.codemem/chat_history.sqlite`)
- `CODEMEM_CACHE_SIZE`: Query cache size (default: 100)
- `CODEMEM_CACHE_TTL`: Cache TTL in seconds (default: 3600)

### Directory Structure

```
~/.codemem/
├── chat_history.sqlite       # Main database
├── chat_history.sqlite-journal  # SQLite journal
└── md_sessions/              # Exported Markdown files
    ├── session_abc123.md
    ├── session_def456.md
    └── ...
```

## Troubleshooting

### Database not found
```
Error: Markdown sessions directory not found
```
**Solution**: Restart the MCP server to rebuild the database and export markdown files.

### BM25 index not ready
```
Error: BM25 index not built yet
```
**Solution**: Wait a few seconds for initialization to complete. The index is built in the background.

### No results found
- Check that your data directory contains valid JSON/JSONL files
- Verify the database path is correct
- Try using SQL queries to inspect the database directly

## Development

### Running Tests

```bash
# Run all tests (if available)
python -m pytest tests/

# Test database building
python unified_history.py --data-dir /path/to/data --db-path test.sqlite

# Test markdown export
python export_sessions_md.py --db-path test.sqlite --output-dir ./test_md
```

### Project Structure

```
CodeMem/
├── mcp_server.py           # Main MCP server
├── unified_history.py      # Chat history loader
├── export_sessions_md.py   # Markdown exporter
├── models.py              # Data models
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [MCP (Model Context Protocol)](https://modelcontextprotocol.io/)
- Uses [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) for semantic search
- Powered by [tiktoken](https://github.com/openai/tiktoken) for tokenization
