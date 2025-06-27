# First Queries Guide

Learn how to use the Qdrant Neo4j Crawl4AI MCP Server with practical examples and real-world use cases.

## 🎯 What You'll Learn

- 🔑 Authentication and token management
- 🔍 Vector search for semantic retrieval
- 🕸️ Graph queries for relationship analysis
- 🌐 Web intelligence for real-time data
- 🤖 MCP client integration
- 🔧 Advanced query patterns

## 🔑 Authentication Setup

All API endpoints require authentication. Start by getting an access token.

### Get Access Token

```bash
# Request a token with read/write permissions
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "scopes": ["read", "write", "admin"]
  }'
```

Response:

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "scopes": ["read", "write", "admin"]
}
```

### Set Up Environment

```bash
# Save your token for subsequent requests
export TOKEN="your_access_token_here"

# Verify authentication
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/profile"
```

## 🔍 Vector Search Queries

Vector search enables semantic search across your document collections.

### Store Your First Document

```bash
# Store a sample document
curl -X POST "http://localhost:8000/api/v1/vector/store" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence and machine learning are revolutionizing how we analyze data, automate processes, and make intelligent decisions. These technologies enable computers to learn from data without explicit programming.",
    "content_type": "text",
    "source": "ai_overview_doc",
    "tags": ["ai", "ml", "technology"],
    "metadata": {
      "category": "technology",
      "difficulty": "beginner",
      "language": "en"
    }
  }'
```

Response:

```json
{
  "status": "success",
  "id": "doc_001",
  "collection_name": "mcp_intelligence",
  "vector_dimensions": 384,
  "embedding_time_ms": 45,
  "storage_time_ms": 12,
  "timestamp": "2024-01-15T10:30:00Z",
  "user_id": "demo_user"
}
```

### Store Multiple Documents

```bash
# Store additional documents for richer search results
curl -X POST "http://localhost:8000/api/v1/vector/store" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Cloud computing provides on-demand access to computing resources including servers, storage, databases, networking, software, and analytics. Major providers include AWS, Azure, and Google Cloud Platform.",
    "source": "cloud_computing_guide",
    "tags": ["cloud", "infrastructure", "aws", "azure"],
    "metadata": {"category": "infrastructure", "level": "intermediate"}
  }'

curl -X POST "http://localhost:8000/api/v1/vector/store" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "DevOps practices combine software development and IT operations to shorten development lifecycles and deliver high-quality software continuously. Key practices include CI/CD, infrastructure as code, and monitoring.",
    "source": "devops_practices",
    "tags": ["devops", "cicd", "automation"],
    "metadata": {"category": "development", "level": "advanced"}
  }'
```

### Perform Semantic Search

```bash
# Search for AI-related content
curl -X POST "http://localhost:8000/api/v1/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning automation",
    "limit": 3,
    "filters": {}
  }'
```

### Advanced Vector Search

```bash
# Search with filters and metadata constraints
curl -X POST "http://localhost:8000/api/v1/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cloud infrastructure deployment",
    "limit": 5,
    "filters": {
      "category": "infrastructure",
      "tags": {"$in": ["cloud", "aws"]}
    }
  }'
```

### Check Collection Status

```bash
# List all vector collections
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/vector/collections"
```

Response includes collection statistics:

```json
{
  "status": "success",
  "collections": [
    {
      "name": "mcp_intelligence",
      "status": "green",
      "vector_size": 384,
      "distance_metric": "cosine",
      "points_count": 3,
      "indexed_vectors": 3,
      "segments_count": 1,
      "disk_usage_mb": 0.5,
      "ram_usage_mb": 1.2
    }
  ],
  "total_collections": 1,
  "total_vectors": 3
}
```

## 🕸️ Graph Queries

Graph queries help you understand relationships and build knowledge systems.

### Basic Graph Query

```bash
# Query the graph database
curl -X POST "http://localhost:8000/api/v1/graph/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence concepts",
    "mode": "graph",
    "limit": 10
  }'
```

### Store Graph Knowledge

For demonstration, let's imagine we're storing knowledge relationships:

```python
# Example Python script for graph operations
import asyncio
import httpx
import json

async def store_graph_knowledge():
    token = "your_token_here"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # Example of storing interconnected knowledge
    knowledge_nodes = [
        {
            "content": "Machine Learning is a subset of Artificial Intelligence",
            "type": "relationship",
            "entities": ["Machine Learning", "Artificial Intelligence"],
            "relation": "subset_of"
        },
        {
            "content": "Neural Networks are a type of Machine Learning algorithm",
            "type": "relationship", 
            "entities": ["Neural Networks", "Machine Learning"],
            "relation": "type_of"
        },
        {
            "content": "Deep Learning uses multi-layer Neural Networks",
            "type": "relationship",
            "entities": ["Deep Learning", "Neural Networks"],
            "relation": "uses"
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for knowledge in knowledge_nodes:
            response = await client.post(
                "http://localhost:8000/api/v1/vector/store",
                headers=headers,
                json={
                    "content": knowledge["content"],
                    "content_type": "knowledge_graph",
                    "metadata": {
                        "type": knowledge["type"],
                        "entities": knowledge["entities"],
                        "relation": knowledge["relation"]
                    },
                    "tags": ["knowledge", "relationship"]
                }
            )
            print(f"Stored: {knowledge['content']}")
            print(f"Response: {response.status_code}")

# Run the knowledge storage
# asyncio.run(store_graph_knowledge())
```

### Query Relationships

```bash
# Search for relationship patterns
curl -X POST "http://localhost:8000/api/v1/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks deep learning relationship",
    "limit": 5,
    "filters": {
      "content_type": "knowledge_graph"
    }
  }'
```

## 🌐 Web Intelligence Queries

Web intelligence allows you to crawl and analyze web content in real-time.

### Crawl Web Content

```bash
# Crawl a webpage for content
curl -X POST "http://localhost:8000/api/v1/web/crawl" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "https://docs.python.org/3/library/asyncio.html",
    "mode": "web",
    "filters": {
      "extract_code": true,
      "extract_headers": true
    }
  }'
```

### Intelligent Content Extraction

```bash
# Extract specific content types
curl -X POST "http://localhost:8000/api/v1/web/crawl" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "https://github.com/microsoft/pyright",
    "mode": "web",
    "filters": {
      "content_types": ["documentation", "code", "readme"],
      "max_depth": 2,
      "follow_links": true
    }
  }'
```

## 🔮 Unified Intelligence Queries

The unified interface automatically routes queries to the most appropriate service.

### Auto-Mode Query

```bash
# Let the system decide the best approach
curl -X POST "http://localhost:8000/api/v1/intelligence/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in large language models?",
    "mode": "auto",
    "limit": 5
  }'
```

The system will:

1. Analyze the query intent
2. Route to vector search for stored knowledge
3. Supplement with graph relationships
4. Optionally fetch recent web content

### Multi-Modal Query

```bash
# Specify multiple processing modes
curl -X POST "http://localhost:8000/api/v1/intelligence/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning deployment best practices",
    "mode": "vector+graph+web",
    "limit": 10,
    "filters": {
      "include_sources": true,
      "confidence_threshold": 0.7
    }
  }'
```

## 🤖 MCP Client Integration

Integrate with AI assistants and applications using the Model Context Protocol.

### Python MCP Client

```python
# Example MCP client implementation
import asyncio
import httpx
from typing import Dict, List, Any

class QdrantNeo4jCrawl4AIMCPClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.token = None
        
    async def authenticate(self, username: str = "mcp_client"):
        """Get authentication token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/auth/token",
                json={
                    "username": username,
                    "scopes": ["read", "write"]
                }
            )
            data = response.json()
            self.token = data["access_token"]
            return self.token
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def vector_search(self, query: str, limit: int = 10, 
                           filters: Dict = None) -> Dict[str, Any]:
        """Perform semantic vector search"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/vector/search",
                headers=self._get_headers(),
                json={
                    "query": query,
                    "limit": limit,
                    "filters": filters or {}
                }
            )
            return response.json()
    
    async def store_document(self, content: str, metadata: Dict = None,
                           tags: List[str] = None) -> Dict[str, Any]:
        """Store document with vector embedding"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/vector/store",
                headers=self._get_headers(),
                json={
                    "content": content,
                    "content_type": "text",
                    "metadata": metadata or {},
                    "tags": tags or []
                }
            )
            return response.json()
    
    async def graph_query(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Query knowledge graph"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/graph/query",
                headers=self._get_headers(),
                json={
                    "query": query,
                    "mode": "graph",
                    "limit": limit
                }
            )
            return response.json()
    
    async def web_crawl(self, url: str, extract_options: Dict = None) -> Dict[str, Any]:
        """Crawl and extract web content"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/web/crawl",
                headers=self._get_headers(),
                json={
                    "query": url,
                    "mode": "web",
                    "filters": extract_options or {}
                }
            )
            return response.json()
    
    async def unified_query(self, query: str, mode: str = "auto",
                           limit: int = 10) -> Dict[str, Any]:
        """Unified intelligence query across all services"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/intelligence/query",
                headers=self._get_headers(),
                json={
                    "query": query,
                    "mode": mode,
                    "limit": limit
                }
            )
            return response.json()

# Usage example
async def main():
    # Initialize client
    client = QdrantNeo4jCrawl4AIMCPClient("http://localhost:8000")
    
    # Authenticate
    await client.authenticate("ai_assistant")
    
    # Store knowledge
    doc_result = await client.store_document(
        content="Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
        metadata={"category": "devops", "technology": "kubernetes"},
        tags=["containers", "orchestration", "k8s"]
    )
    print(f"Stored document: {doc_result['id']}")
    
    # Search for information
    search_results = await client.vector_search(
        query="container orchestration platform",
        limit=5
    )
    print(f"Found {len(search_results.get('results', []))} results")
    
    # Unified intelligence query
    intelligence_result = await client.unified_query(
        query="How do I deploy a microservice architecture?",
        mode="auto"
    )
    print(f"Intelligence result: {intelligence_result['content']}")

# Run the example
# asyncio.run(main())
```

### JavaScript/TypeScript Client

```typescript
// TypeScript MCP client
interface QueryOptions {
  limit?: number;
  filters?: Record<string, any>;
  mode?: 'auto' | 'vector' | 'graph' | 'web';
}

class QdrantNeo4jCrawl4AIMCPClient {
  private baseUrl: string;
  private token: string | null = null;
  
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }
  
  async authenticate(username: string = 'js_client'): Promise<string> {
    const response = await fetch(`${this.baseUrl}/auth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username,
        scopes: ['read', 'write']
      })
    });
    
    const data = await response.json();
    this.token = data.access_token;
    return this.token;
  }
  
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }
    
    return headers;
  }
  
  async vectorSearch(query: string, options: QueryOptions = {}): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/vector/search`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        query,
        limit: options.limit || 10,
        filters: options.filters || {}
      })
    });
    
    return response.json();
  }
  
  async storeDocument(content: string, metadata: Record<string, any> = {}): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/vector/store`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        content,
        content_type: 'text',
        metadata
      })
    });
    
    return response.json();
  }
  
  async unifiedQuery(query: string, options: QueryOptions = {}): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/intelligence/query`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        query,
        mode: options.mode || 'auto',
        limit: options.limit || 10,
        filters: options.filters || {}
      })
    });
    
    return response.json();
  }
}

// Usage example
async function example() {
  const client = new QdrantNeo4jCrawl4AIMCPClient('http://localhost:8000');
  
  // Authenticate
  await client.authenticate('web_app');
  
  // Store document
  await client.storeDocument(
    'React is a JavaScript library for building user interfaces.',
    { category: 'frontend', framework: 'react' }
  );
  
  // Search
  const results = await client.vectorSearch('JavaScript UI framework');
  console.log('Search results:', results);
  
  // Unified query
  const intelligence = await client.unifiedQuery(
    'Best practices for React development'
  );
  console.log('Intelligence:', intelligence);
}
```

## 🔧 Advanced Query Patterns

### Batch Operations

```bash
# Store multiple documents efficiently
curl -X POST "http://localhost:8000/api/v1/vector/batch-store" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Docker containerization basics...",
        "metadata": {"category": "devops", "level": "beginner"}
      },
      {
        "content": "Kubernetes networking concepts...",
        "metadata": {"category": "devops", "level": "intermediate"}
      },
      {
        "content": "Service mesh architecture patterns...",
        "metadata": {"category": "devops", "level": "advanced"}
      }
    ]
  }'
```

### Complex Filtering

```bash
# Advanced search with multiple criteria
curl -X POST "http://localhost:8000/api/v1/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "microservices architecture patterns",
    "limit": 10,
    "filters": {
      "$and": [
        {"category": {"$eq": "architecture"}},
        {"level": {"$in": ["intermediate", "advanced"]}},
        {"tags": {"$contains": "microservices"}},
        {"created_at": {"$gte": "2024-01-01T00:00:00Z"}}
      ]
    },
    "score_threshold": 0.75,
    "include_payload": true,
    "include_vectors": false
  }'
```

### Hybrid Search

```bash
# Combine vector search with metadata filtering
curl -X POST "http://localhost:8000/api/v1/vector/hybrid-search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning deployment strategies",
    "vector_weight": 0.7,
    "metadata_weight": 0.3,
    "filters": {
      "category": "ml",
      "tags": {"$contains": "deployment"}
    },
    "rerank": true,
    "limit": 15
  }'
```

### Real-Time Updates

```bash
# Set up webhook for real-time updates
curl -X POST "http://localhost:8000/api/v1/webhooks/register" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/webhooks/mcp",
    "events": ["document.stored", "search.performed", "graph.updated"],
    "filters": {
      "collection": "mcp_intelligence"
    }
  }'
```

## 📊 Performance Monitoring

### Query Performance

```bash
# Get performance metrics
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/metrics/performance"
```

### Health Monitoring

```bash
# Comprehensive health check
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/health" | jq '.'
```

### Service Status

```bash
# Check individual service health
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/vector/health"

curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/graph/health"

curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/web/health"
```

## 🎯 Real-World Use Cases

### 1. Building a Knowledge Base

```python
async def build_knowledge_base():
    """Build a comprehensive knowledge base"""
    client = QdrantNeo4jCrawl4AIMCPClient("http://localhost:8000")
    await client.authenticate("knowledge_curator")
    
    # Store various types of knowledge
    knowledge_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
            "metadata": {"type": "definition", "subject": "python", "domain": "programming"}
        },
        {
            "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It features automatic API documentation, type hints, and async support.",
            "metadata": {"type": "technology", "subject": "fastapi", "domain": "web_development"}
        },
        {
            "content": "Vector databases store high-dimensional vectors and enable similarity search. They're essential for AI applications like semantic search and recommendation systems.",
            "metadata": {"type": "concept", "subject": "vector_databases", "domain": "ai"}
        }
    ]
    
    for doc in knowledge_docs:
        result = await client.store_document(
            content=doc["content"],
            metadata=doc["metadata"],
            tags=[doc["metadata"]["subject"], doc["metadata"]["domain"]]
        )
        print(f"Stored: {doc['metadata']['subject']}")
    
    # Query the knowledge base
    search_result = await client.vector_search(
        "What is a modern Python web framework?",
        filters={"domain": "web_development"}
    )
    
    return search_result
```

### 2. Content Research Assistant

```python
async def research_assistant(topic: str):
    """Research assistant that combines stored knowledge with web data"""
    client = QdrantNeo4jCrawl4AIMCPClient("http://localhost:8000")
    await client.authenticate("researcher")
    
    # Search existing knowledge
    existing_knowledge = await client.vector_search(
        topic,
        limit=5
    )
    
    # If insufficient results, crawl web for more information
    if len(existing_knowledge.get('results', [])) < 3:
        # Simulate web crawling for more information
        web_sources = [
            f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            f"https://stackoverflow.com/search?q={topic.replace(' ', '+')}"
        ]
        
        for url in web_sources:
            try:
                web_content = await client.web_crawl(url)
                # Store the crawled content
                if web_content.get('content'):
                    await client.store_document(
                        content=web_content['content'][:1000],  # Truncate for demo
                        metadata={
                            "source": "web_crawl",
                            "url": url,
                            "topic": topic
                        },
                        tags=["research", "web_content"]
                    )
            except Exception as e:
                print(f"Failed to crawl {url}: {e}")
    
    # Final comprehensive search
    final_results = await client.unified_query(
        f"Comprehensive information about {topic}",
        mode="auto",
        limit=10
    )
    
    return final_results
```

### 3. Intelligent Document Management

```python
async def intelligent_document_manager():
    """Manage documents with automatic categorization and relationships"""
    client = QdrantNeo4jCrawl4AIMCPClient("http://localhost:8000")
    await client.authenticate("doc_manager")
    
    # Store documents with rich metadata
    documents = [
        {
            "content": "Project requirements document for the new customer portal...",
            "metadata": {
                "type": "requirements",
                "project": "customer_portal",
                "phase": "planning",
                "stakeholders": ["product", "engineering", "design"]
            }
        },
        {
            "content": "Technical architecture design for microservices implementation...",
            "metadata": {
                "type": "architecture",
                "project": "customer_portal", 
                "phase": "design",
                "stakeholders": ["engineering", "devops"]
            }
        }
    ]
    
    stored_docs = []
    for doc in documents:
        result = await client.store_document(
            content=doc["content"],
            metadata=doc["metadata"],
            tags=[doc["metadata"]["type"], doc["metadata"]["project"]]
        )
        stored_docs.append(result)
    
    # Find related documents
    related_docs = await client.vector_search(
        "customer portal project documents",
        filters={"project": "customer_portal"}
    )
    
    # Build knowledge graph of relationships
    for doc in stored_docs:
        # Store relationship information
        await client.store_document(
            content=f"Document {doc['id']} is part of {doc['metadata']['project']} project",
            metadata={
                "type": "relationship",
                "source_doc": doc["id"],
                "relation": "part_of",
                "target": doc["metadata"]["project"]
            },
            tags=["relationship", "project_structure"]
        )
    
    return related_docs
```

## 🚨 Error Handling

### Common Error Responses

```bash
# Authentication error (401)
{
  "error": "Could not validate credentials",
  "status_code": 401,
  "timestamp": "2024-01-15T10:30:00Z"
}

# Rate limit error (429)
{
  "error": "Rate limit exceeded",
  "status_code": 429,
  "retry_after": 60,
  "timestamp": "2024-01-15T10:30:00Z"
}

# Service unavailable (503)
{
  "error": "Vector service is not available",
  "status_code": 503,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Robust Error Handling

```python
import httpx
import asyncio
from typing import Optional

async def robust_query(client: QdrantNeo4jCrawl4AIMCPClient, 
                      query: str, retries: int = 3) -> Optional[dict]:
    """Query with robust error handling and retries"""
    
    for attempt in range(retries):
        try:
            result = await client.vector_search(query)
            return result
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                # Re-authenticate and retry
                await client.authenticate()
                continue
            elif e.response.status_code == 429:
                # Rate limited, wait and retry
                await asyncio.sleep(2 ** attempt)
                continue
            elif e.response.status_code >= 500:
                # Server error, retry with backoff
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                # Client error, don't retry
                raise
                
        except httpx.ConnectError:
            # Connection error, retry with backoff
            await asyncio.sleep(2 ** attempt)
            continue
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1)
    
    return None
```

## 🎉 Next Steps

Now that you've learned the basics:

1. **[Explore advanced configuration](./configuration.md)** - Customize the system for your needs
2. **[Set up production deployment](./installation.md#kubernetes-deployment)** - Deploy at scale
3. **[Monitor and troubleshoot](./troubleshooting.md)** - Keep your system healthy
4. **[Check the API reference](../API_REFERENCE.md)** - Learn all available endpoints
5. **[Review examples](../examples/)** - See real-world implementations

## 📚 Additional Resources

- **Interactive API Docs**: <http://localhost:8000/docs>
- **Alternative Docs**: <http://localhost:8000/redoc>  
- **Neo4j Browser**: <http://localhost:7474>
- **Monitoring Dashboard**: <http://localhost:3000>
- **Technical Documentation**: [../TECHNICAL_DOCUMENTATION.md](../TECHNICAL_DOCUMENTATION.md)

---

**🔗 Quick Links:**

- [Configuration Guide](./configuration.md) - Customize your setup
- [Troubleshooting](./troubleshooting.md) - Solve problems
- [Installation Guide](./installation.md) - Advanced deployment options
