# CodeMem MCP Server Test Plan

## Overview
This test plan covers comprehensive testing of the CodeMem MCP server implementation, which provides unified conversation history management through the Model Context Protocol.

## Test Environment
- **Python Version**: >= 3.10
- **Database**: SQLite (aiosqlite)
- **MCP Protocol**: 2024-11-05
- **Transport**: stdio

## Test Categories

### 1. Database Building & Initialization Tests

#### 1.1 Database Creation
- [ ] Test database creation with default path (`~/.codemem/codemem.sqlite`)
- [ ] Test database creation with custom path
- [ ] Verify versioned database naming (`codemem-{timestamp}.db`)
- [ ] Verify symlink creation to latest version
- [ ] Test database rebuild with `--rebuild` flag

#### 1.2 Platform Data Collection
- [ ] Test auto-discovery of Claude Code history
- [ ] Test auto-discovery of Codex CLI history
- [ ] Test auto-discovery of Cursor history
- [ ] Test auto-discovery of OpenCode history
- [ ] Test custom root directory with `--root` flag
- [ ] Verify handling of missing platform directories

#### 1.3 Schema Verification
- [ ] Verify `events_raw` table creation
- [ ] Verify `events` view creation
- [ ] Verify all indexes are created correctly
- [ ] Test data integrity after insertion

### 2. MCP Tool Tests

#### 2.1 semantic.search Tool
- [ ] Test basic search with simple query
- [ ] Test search with Chinese characters
- [ ] Test search with English text
- [ ] Test `top_k` parameter (default 20)
- [ ] Test `top_k` parameter with custom values
- [ ] Verify BM25 scoring and ranking
- [ ] Test empty query handling
- [ ] Test query with no results
- [ ] Verify result format (JSON with scores)

#### 2.2 sql.query Tool
- [ ] Test basic SELECT query
- [ ] Test SELECT with WHERE clause
- [ ] Test SELECT with JOIN operations
- [ ] Test SELECT with ORDER BY
- [ ] Test SELECT with LIMIT
- [ ] Test `limit` parameter enforcement (max 50)
- [ ] Verify SQL injection prevention (reject non-SELECT)
- [ ] Test invalid SQL syntax handling
- [ ] Verify result format (columns + rows)

#### 2.3 regex.search Tool
- [ ] Test basic regex pattern
- [ ] Test case-insensitive flag (`i`)
- [ ] Test multiline flag (`m`)
- [ ] Test dotall flag (`s`)
- [ ] Test combined flags
- [ ] Test `limit` parameter (default 50)
- [ ] Test invalid regex pattern handling
- [ ] Verify result format with context

### 3. Caching Tests

#### 3.1 Cache Functionality
- [ ] Verify cache hit on repeated queries
- [ ] Verify cache miss on new queries
- [ ] Test LRU eviction (max 100 entries)
- [ ] Test TTL expiration (1 hour)
- [ ] Verify cache key generation (MD5 hash)
- [ ] Test cache thread safety with concurrent requests

#### 3.2 Cache Performance
- [ ] Measure response time for cached vs uncached queries
- [ ] Verify <0.1s response for cached queries

### 4. BM25 Index Tests

#### 4.1 Index Building
- [ ] Verify markdown export to `~/.codemem/md_sessions/`
- [ ] Test BM25 index construction
- [ ] Verify Tiktoken tokenization
- [ ] Test index with empty database
- [ ] Test index rebuild

#### 4.2 Index Performance
- [ ] Test search performance with small dataset (<100 records)
- [ ] Test search performance with medium dataset (100-1000 records)
- [ ] Test search performance with large dataset (>1000 records)

### 5. Error Handling Tests

#### 5.1 Database Errors
- [ ] Test behavior when database file is locked
- [ ] Test behavior when database is corrupted
- [ ] Test behavior when disk is full
- [ ] Verify error messages are informative

#### 5.2 Tool Errors
- [ ] Test unknown tool name
- [ ] Test missing required parameters
- [ ] Test invalid parameter types
- [ ] Test parameter validation errors

#### 5.3 Timeout Handling
- [ ] Test database readiness timeout (120 seconds)
- [ ] Test long-running queries

### 6. MCP Server Tests

#### 6.1 Server Startup
- [ ] Test server initialization
- [ ] Verify stdio transport setup
- [ ] Test server with default configuration
- [ ] Test server with custom configuration

#### 6.2 Protocol Compliance
- [ ] Verify MCP protocol version (2024-11-05)
- [ ] Test `list_tools()` response format
- [ ] Test `call_tool()` request/response format
- [ ] Verify TextContent wrapping

#### 6.3 Concurrent Operations
- [ ] Test multiple simultaneous tool calls
- [ ] Test concurrent database access
- [ ] Verify async operation correctness

### 7. Integration Tests

#### 7.1 End-to-End Workflows
- [ ] Test complete workflow: build DB → search → query results
- [ ] Test workflow with multiple platforms
- [ ] Test workflow with incremental updates

#### 7.2 Client Integration
- [ ] Test with Claude Code client
- [ ] Test with Codex CLI client
- [ ] Verify configuration examples work

### 8. Performance Tests

#### 8.1 Token Efficiency
- [ ] Measure token usage for semantic search
- [ ] Measure token usage for SQL queries
- [ ] Verify 95-99% token savings claim

#### 8.2 Query Performance
- [ ] Benchmark semantic search latency
- [ ] Benchmark SQL query latency
- [ ] Benchmark regex search latency

### 9. Security Tests

#### 9.1 SQL Injection Prevention
- [ ] Test SQL injection attempts (DROP, DELETE, UPDATE)
- [ ] Test parameterized query safety
- [ ] Verify only SELECT queries allowed

#### 9.2 Path Traversal Prevention
- [ ] Test markdown directory restrictions
- [ ] Test file access outside allowed paths

### 10. Data Validation Tests

#### 10.1 Input Validation
- [ ] Test UnifiedEvent model validation
- [ ] Test parameter type validation
- [ ] Test required field enforcement

#### 10.2 Output Validation
- [ ] Verify JSON output format
- [ ] Test result limit enforcement
- [ ] Verify data serialization correctness

## Test Execution Priority

### Phase 1: Core Functionality (Critical)
1. Database building and initialization
2. MCP server startup
3. All three tool basic operations

### Phase 2: Robustness (High Priority)
4. Error handling
5. Caching functionality
6. Security tests

### Phase 3: Performance (Medium Priority)
7. Performance benchmarks
8. Concurrent operations
9. Large dataset handling

### Phase 4: Integration (Low Priority)
10. Client integration tests
11. End-to-end workflows

## Success Criteria

- All critical tests pass (Phase 1)
- No security vulnerabilities found
- Performance meets documented claims
- Error handling is robust and informative
- MCP protocol compliance verified

## Test Automation

Tests should be automated using:
- `pytest` for unit and integration tests
- `pytest-asyncio` for async test support
- `pytest-benchmark` for performance tests
- Mock MCP clients for protocol testing
