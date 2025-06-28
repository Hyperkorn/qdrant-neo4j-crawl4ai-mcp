# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic RAG MCP Server** that combines vector search (Qdrant), knowledge graphs (Neo4j), and web intelligence (Crawl4AI) through a unified Model Context Protocol (MCP) interface. The system provides autonomous query routing and result fusion using FastMCP 2.0.

## Key Architecture

- **FastMCP 2.0**: Service composition pattern with mounted endpoints
- **Async-First**: All operations use async/await for high concurrency
- **Service Layer**: Three main services (Vector, Graph, Web) with unified interfaces
- **MCP Tools**: Individual tool registrations for each service type
- **Production-Ready**: JWT auth, rate limiting, monitoring, security headers

## Build System & Dependencies

### Primary Tools

- **uv**: Package management (replaces pip) - use `uv sync` for installation
- **ruff**: Code formatting and linting - run `ruff check . --fix` then `ruff format .`
- **mypy**: Type checking - run `mypy .`
- **pytest**: Testing framework with async support

### Development Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Run the server locally
uv run python -m qdrant_neo4j_crawl4ai_mcp

# Run with Docker
docker-compose up -d

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=qdrant_neo4j_crawl4ai_mcp --cov-report=html

# Run specific test suites
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/security/

# Code quality checks
uv run ruff check . --fix
uv run ruff format .
uv run mypy .

# Security scanning
uv run bandit -r src/
```

### Test Execution

- **Unit tests**: `tests/unit/` - Fast, isolated component tests
- **Integration tests**: `tests/integration/` - Service integration tests  
- **Security tests**: `tests/security/` - Authentication and authorization tests
- **Performance tests**: `tests/performance/` - Load and benchmark tests
- **Property tests**: `tests/property/` - MCP protocol compliance tests

## Architecture Components

### Core Services (src/qdrant_neo4j_crawl4ai_mcp/services/)

1. **VectorService** (`vector_service.py`): Qdrant integration for semantic search
2. **GraphService** (`graph_service.py`): Neo4j integration for knowledge graphs  
3. **WebService** (`web_service.py`): Crawl4AI integration for web intelligence

### MCP Tools (src/qdrant_neo4j_crawl4ai_mcp/tools/)

1. **vector_tools.py**: MCP tool registrations for vector operations
2. **graph_tools.py**: MCP tool registrations for graph operations
3. **web_tools.py**: MCP tool registrations for web operations

### Data Models (src/qdrant_neo4j_crawl4ai_mcp/models/)

- **vector_models.py**: Pydantic models for vector operations
- **graph_models.py**: Pydantic models for graph operations
- **web_models.py**: Pydantic models for web operations

### Core Application Files

- **main.py**: FastAPI application with FastMCP integration and service lifecycle
- **config.py**: Comprehensive settings with Pydantic validation and secrets handling
- **auth.py**: JWT authentication with scope-based authorization
- **middleware.py**: Security middleware stack (CORS, rate limiting, security headers)

## Key Development Patterns

### Service Pattern

Each service follows this structure:

- **Config class**: Pydantic settings with validation
- **Service class**: Business logic with async methods
- **Client integration**: Database/API client management
- **Error handling**: Structured exceptions with proper logging
- **Health checks**: Individual service health endpoints

### MCP Tool Registration

Tools are registered in dedicated modules:

```python
def register_vector_tools(mcp: FastMCP, service: VectorService) -> None:
    @mcp.tool()
    async def vector_search(query: str) -> VectorSearchResponse:
        # Tool implementation
```

### Configuration Management

All configuration uses Pydantic with environment variable support:

```python
class Settings(BaseSettings):
    qdrant_url: str = Field(env="QDRANT_URL", default="http://localhost:6333")
    
    class Config:
        env_file = ".env"
```

## Security Considerations

- **JWT Authentication**: All endpoints require valid JWT tokens with appropriate scopes
- **Rate Limiting**: Redis-backed distributed rate limiting
- **Input Validation**: Comprehensive Pydantic validation on all inputs
- **OWASP Compliance**: Security headers and API security best practices
- **Secrets Management**: SecretStr for sensitive configuration values

## Environment Setup

Required environment variables:

```env
# Database URLs
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-secret-key
ADMIN_API_KEY=admin-key

# External APIs (optional)
OPENAI_API_KEY=sk-...
```

## Service Dependencies

- **Qdrant**: Vector database (port 6333)
- **Neo4j**: Graph database (port 7687)  
- **Redis**: Cache and rate limiting (port 6379)
- **External APIs**: OpenAI (for GraphRAG), various web APIs

## Monitoring & Health Checks

- **Health endpoint**: `/health` - Comprehensive service status
- **Readiness endpoint**: `/ready` - Kubernetes-style readiness
- **Metrics endpoint**: `/metrics` - Prometheus metrics
- **Admin stats**: `/api/v1/admin/stats` - Administrative monitoring

## Production Deployment

The application supports multiple deployment methods:

- **Docker**: Single container with docker-compose for local development
- **Kubernetes**: Production manifests in `k8s/manifests/`
- **Cloud providers**: Railway, Fly.io configurations in `deployment/`

## Error Handling Patterns

- **Structured exceptions**: Custom exception classes with proper HTTP status codes
- **Async error handling**: Proper exception propagation in async contexts
- **Circuit breakers**: Fail-fast patterns for external service dependencies
- **Graceful degradation**: Service availability checks before operations

## Testing Patterns

- **Async testing**: All tests use `pytest-asyncio` for async support
- **Mocking**: Comprehensive mocking of external services
- **Test containers**: Integration tests use testcontainers for real database testing
- **Property testing**: Hypothesis for MCP protocol compliance testing

---

## Serena MCP Server Integration

**CRITICAL: Serena is a specialized code analysis and editing system. Always follow this workflow for optimal code understanding and modification.**

### Essential Workflow (ALWAYS Follow This Order)

**Startup Sequence:**
1. `mcp__serena__initial_instructions` - Get project instructions (ALWAYS call first)
2. `mcp__serena__check_onboarding_performed` - Check if onboarding completed
3. `mcp__serena__onboarding` - Perform onboarding if needed
4. `mcp__serena__list_memories` - Check existing project knowledge

### Project Analysis & Navigation

**Overview Tools:**
- `mcp__serena__get_current_config` - Current configuration, projects, tools, modes
- `mcp__serena__list_dir` - List files/directories (use `recursive: true` for deep exploration)
- `mcp__serena__find_file` - Find files by name/pattern mask
- `mcp__serena__get_symbols_overview` - Get code symbol overview (classes, functions, etc.)

**Code Discovery:**
- `mcp__serena__find_symbol` - Find symbols by name path pattern
  - Use patterns like `"ClassName"`, `"method"`, or `"/ClassName/method"` for absolute paths
  - Add `depth: 1` to get children (methods of a class)
  - Use `substring_matching: true` for partial matches
- `mcp__serena__find_referencing_symbols` - Find what references a symbol
- `mcp__serena__search_for_pattern` - Regex search across project files

**Analysis Checkpoints:**
- `mcp__serena__think_about_collected_information` - ALWAYS call after search sequences
- `mcp__serena__think_about_task_adherence` - ALWAYS call before making code changes

### File Operations

**Reading:**
- `mcp__serena__read_file` - Read file content or chunks
  - Prefer symbolic operations when you know the symbol name
  - Use `start_line` and `end_line` for large files

**Creating:**
- `mcp__serena__create_text_file` - Create new files
  - **Only use for new files**; prefer symbolic operations for existing files

### Code Editing (Prefer Symbolic Operations)

**Symbol-Level Editing (PREFERRED):**
- `mcp__serena__replace_symbol_body` - Replace entire function/class body
- `mcp__serena__insert_after_symbol` - Insert after symbol definition
- `mcp__serena__insert_before_symbol` - Insert before symbol definition

**Pattern-Based Editing:**
- `mcp__serena__replace_regex` - Use with wildcards for complex replacements
  - **CRITICAL: Use wildcards (`.*?`) to avoid specifying exact content**
  - Example: `"function.*?{.*?}"` instead of full function text

**Line-Based Editing (Last Resort):**
- `mcp__serena__delete_lines` - Delete line ranges (requires prior read_file)
- `mcp__serena__replace_lines` - Replace line ranges (requires prior read_file)
- `mcp__serena__insert_at_line` - Insert at specific line

### System Operations

**Shell Commands:**
- `mcp__serena__execute_shell_command` - Execute shell commands
  - **ALWAYS check memories first** for suggested commands
  - Read "suggested shell commands" memory before using

**Language Server:**
- `mcp__serena__restart_language_server` - Restart on errors/inconsistencies

### Memory Management

**Project Knowledge:**
- `mcp__serena__write_memory` - Store important project insights
  - Use during onboarding and when discovering key patterns
  - Short, focused memories are better than large ones
- `mcp__serena__read_memory` - Read stored memory by filename
- `mcp__serena__list_memories` - List all available memories
- `mcp__serena__delete_memory` - Delete memory (only on explicit user request)

### Quality Assurance Workflow

**Before Code Changes:**
1. `mcp__serena__think_about_task_adherence` - Verify alignment with task
2. Make changes using symbolic operations
3. `mcp__serena__think_about_whether_you_are_done` - Evaluate completion
4. `mcp__serena__summarize_changes` - Document what was changed

**Common Patterns:**

```javascript
// Code Discovery Pattern
get_symbols_overview → find_symbol → find_referencing_symbols → think_about_collected_information

// Editing Pattern
think_about_task_adherence → replace_symbol_body → think_about_whether_you_are_done → summarize_changes

// Search Pattern
search_for_pattern → find_symbol → read_file → think_about_collected_information
```

**Key Rules:**
- **Always prefer symbolic operations** over direct file editing
- **Use wildcards in regex** to avoid brittle exact matches
- **Call think_about_* tools** at workflow checkpoints
- **Store project insights** in memories during discovery
- **Check memories before shell commands** for project-specific commands
