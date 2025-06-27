# Qdrant Neo4j Crawl4AI MCP Server API Documentation

The Unified MCP Intelligence Server provides a comprehensive API for agentic Retrieval-Augmented Generation (RAG) applications, combining vector search, knowledge graphs, and web intelligence into a single Model Context Protocol (MCP) server.

## 🚀 API Overview

The server exposes three primary intelligence services through MCP tools:

- **Vector Intelligence**: Semantic search and embedding storage via Qdrant
- **Graph Intelligence**: Knowledge graphs and memory systems via Neo4j  
- **Web Intelligence**: Smart web crawling and content extraction via Crawl4AI

All services are unified under FastMCP 2.0 with enterprise-grade authentication, rate limiting, and monitoring.

## 🔗 Quick Navigation

### Core APIs

- [MCP Tools](./tools/) - All available MCP tools for AI assistants
- [MCP Resources](./resources/) - Data resources and collections
- [Data Schemas](./schemas/) - Request/response models and validation

### Security & Operations

- [Authentication](./authentication.md) - JWT tokens and API keys
- [Rate Limiting](./rate-limiting.md) - Request throttling and quotas
- [Webhooks](./webhooks.md) - Event notifications and integrations
- [Client Libraries](./client-libraries.md) - SDKs and client examples

### Service-Specific Guides

- [Vector Search API](./tools/vector-tools.md) - Qdrant vector operations
- [Knowledge Graph API](./tools/graph-tools.md) - Neo4j graph operations
- [Web Intelligence API](./tools/web-tools.md) - Crawl4AI web operations

## 🏗️ Architecture

```mermaid
graph TB
    Client[AI Assistant Client] --> Gateway[FastMCP Gateway]
    
    subgraph "Unified MCP Server"
        Gateway --> Router[Request Router]
        Router --> Vector[Vector Service]
        Router --> Graph[Graph Service] 
        Router --> Web[Web Intelligence Service]
        
        Vector --> |MCP Tools| QdrantOps[Qdrant Operations]
        Graph --> |MCP Tools| Neo4jOps[Neo4j Operations]
        Web --> |MCP Tools| Crawl4AIOps[Crawl4AI Operations]
    end
    
    subgraph "Data Layer"
        QdrantOps --> QdrantDB[(Qdrant Vector DB)]
        Neo4jOps --> Neo4jDB[(Neo4j Graph DB)]
        Crawl4AIOps --> WebSources[Web Data Sources]
    end
    
    subgraph "Security & Monitoring"
        Gateway --> Auth[JWT/API Key Auth]
        Gateway --> RateLimit[Rate Limiting]
        Gateway --> Audit[Audit Logging]
    end
```

## 🔐 Authentication

The API supports two authentication methods:

### 1. JWT Bearer Tokens

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 2. API Keys

```http
X-API-Key: your-admin-api-key
```

See [Authentication Guide](./authentication.md) for detailed setup.

## 📊 Usage Examples

### Vector Search

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def semantic_search():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "qdrant_neo4j_crawl4ai_mcp"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the client
            await session.initialize()
            
            # Perform semantic search
            result = await session.call_tool(
                "semantic_vector_search",
                {
                    "query": "artificial intelligence machine learning",
                    "collection_name": "knowledge_base",
                    "limit": 5,
                    "score_threshold": 0.7
                }
            )
            
            print(f"Found {len(result.content['results'])} results")
            for item in result.content['results']:
                print(f"Score: {item['score']:.3f} - {item['content'][:100]}...")

asyncio.run(semantic_search())
```

### Knowledge Graph Query

```python
async def graph_analysis():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Search knowledge graph
            result = await session.call_tool(
                "search_graph",
                {
                    "query": "machine learning algorithms",
                    "node_types": ["Concept", "Algorithm"],
                    "max_depth": 3,
                    "use_embeddings": True
                }
            )
            
            print(f"Found {result.content['total_results']} graph results")
            for node in result.content['nodes']:
                print(f"Node: {node['name']} ({node['node_type']})")

asyncio.run(graph_analysis())
```

### Web Content Extraction

```python
async def web_crawl():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Crawl and extract web content
            result = await session.call_tool(
                "crawl_web_page",
                {
                    "url": "https://example.com/article",
                    "max_depth": 2,
                    "output_format": "markdown",
                    "extract_content": True
                }
            )
            
            print(f"Crawled {result.content['total_pages']} pages")
            for content in result.content['content']:
                print(f"Page: {content['title']}")
                print(f"Word count: {content['word_count']}")

asyncio.run(web_crawl())
```

## 📈 Performance Characteristics

| Operation | Typical Response Time | Throughput |
|-----------|----------------------|------------|
| Vector Search | 15-50ms | 1000+ RPS |
| Graph Query | 20-100ms | 500+ RPS |
| Web Crawl | 2-10s | 10-50 concurrent |
| Embedding Generation | 50-200ms | 200+ RPS |

## 🚦 Rate Limits

Default rate limits (configurable):

| Authentication | Requests/Minute | Burst Limit |
|----------------|-----------------|-------------|
| API Key | 1000 | 100 |
| JWT Token | 100 | 20 |
| Unauthenticated | 10 | 5 |

## 🏥 Health Monitoring

### Health Check Endpoints

- `GET /health` - Service health status
- `GET /ready` - Kubernetes readiness check
- `GET /metrics` - Prometheus metrics

### Service-Specific Health

- `GET /api/v1/vector/health` - Vector service health
- Use MCP tool `get_graph_health` - Graph service health
- Use MCP tool `web_service_health` - Web service health

## 🔧 Configuration

### Environment Variables

```env
# Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
JWT_SECRET_KEY=your-secure-secret-key

# Database Configuration  
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Security
RATE_LIMIT_PER_MINUTE=100
CORS_ORIGINS=https://your-domain.com
```

See [Configuration Guide](../configuration.md) for complete settings.

## 📚 API Reference

### MCP Tools

#### Vector Operations

- [`store_vector_document`](./tools/vector-tools.md#store_vector_document) - Store document with embedding
- [`semantic_vector_search`](./tools/vector-tools.md#semantic_vector_search) - Semantic similarity search
- [`create_vector_collection`](./tools/vector-tools.md#create_vector_collection) - Create collection
- [`list_vector_collections`](./tools/vector-tools.md#list_vector_collections) - List collections
- [`generate_text_embeddings`](./tools/vector-tools.md#generate_text_embeddings) - Generate embeddings
- [`delete_vector_collection`](./tools/vector-tools.md#delete_vector_collection) - Delete collection
- [`get_vector_service_stats`](./tools/vector-tools.md#get_vector_service_stats) - Service statistics

#### Graph Operations

- [`create_graph_node`](./tools/graph-tools.md#create_graph_node) - Create knowledge graph node
- [`create_graph_relationship`](./tools/graph-tools.md#create_graph_relationship) - Create relationships
- [`search_graph`](./tools/graph-tools.md#search_graph) - Search knowledge graph
- [`extract_knowledge_from_text`](./tools/graph-tools.md#extract_knowledge_from_text) - AI knowledge extraction
- [`create_memory_node`](./tools/graph-tools.md#create_memory_node) - Create memory node
- [`execute_cypher_query`](./tools/graph-tools.md#execute_cypher_query) - Raw Cypher queries
- [`analyze_graph_structure`](./tools/graph-tools.md#analyze_graph_structure) - Network analysis
- [`get_graph_health`](./tools/graph-tools.md#get_graph_health) - Service health

#### Web Operations

- [`crawl_web_page`](./tools/web-tools.md#crawl_web_page) - Web crawling and extraction
- [`extract_web_content`](./tools/web-tools.md#extract_web_content) - Targeted content extraction
- [`capture_web_screenshot`](./tools/web-tools.md#capture_web_screenshot) - Screenshot capture
- [`monitor_web_content`](./tools/web-tools.md#monitor_web_content) - Content monitoring
- [`get_web_monitor_status`](./tools/web-tools.md#get_web_monitor_status) - Monitor status
- [`stop_web_monitor`](./tools/web-tools.md#stop_web_monitor) - Stop monitoring
- [`list_web_monitors`](./tools/web-tools.md#list_web_monitors) - List monitors
- [`web_service_health`](./tools/web-tools.md#web_service_health) - Service health

### REST API Endpoints

#### Authentication

- `POST /auth/token` - Create JWT access token
- `GET /api/v1/profile` - Get user profile

#### Unified Intelligence

- `POST /api/v1/intelligence/query` - Unified multi-service query

#### Vector Service (REST)

- `POST /api/v1/vector/search` - Vector search
- `POST /api/v1/vector/store` - Store vectors
- `GET /api/v1/vector/collections` - List collections
- `GET /api/v1/vector/health` - Health check

#### Admin

- `GET /api/v1/admin/stats` - System statistics

## 🔍 Error Handling

All APIs return consistent error responses:

```json
{
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "uuid-string"
}
```

Common HTTP status codes:

- `200` - Success
- `400` - Bad Request (validation error)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error
- `503` - Service Unavailable

## 📞 Support

- **Documentation**: [Full Documentation](../../README.md)
- **GitHub Issues**: [GitHub Repository](https://github.com/your-org/qdrant-neo4j-crawl4ai-mcp)
- **Community**: [Discord Channel](#)
- **Email**: <support@your-domain.com>

## 🔄 API Versioning

Current API version: `v1`

Version strategy:

- **Major versions** (`v1` → `v2`): Breaking changes
- **Minor updates**: New features, backward compatible
- **Patch updates**: Bug fixes, security patches

## 🚀 Getting Started

1. **Set up authentication** - See [Authentication Guide](./authentication.md)
2. **Choose your client** - See [Client Libraries](./client-libraries.md)
3. **Explore MCP tools** - Start with [Vector Search](./tools/vector-tools.md)
4. **Build your application** - Check out [Examples](../examples/)

---

*Generated with Claude Code - Last Updated: 2024-01-15*
