# Vector Database Tools

The Vector Intelligence service provides semantic search, embedding generation, and vector storage capabilities through Qdrant integration. These MCP tools enable AI assistants to work with dense vector representations for similarity search and content retrieval.

## Overview

Vector tools leverage state-of-the-art embedding models to convert text into numerical vectors that capture semantic meaning. This enables:

- **Semantic Search**: Find similar content based on meaning, not just keywords
- **Content Discovery**: Explore related documents and information
- **Knowledge Retrieval**: Extract relevant context for AI responses
- **Similarity Analysis**: Compare and cluster similar content

## Available Tools

### store_vector_document

Store a document with vector embedding in Qdrant for later semantic search.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | `string` | ✅ | - | Text content to embed and store (1-100,000 chars) |
| `collection_name` | `string` | ❌ | "documents" | Target collection name |
| `content_type` | `string` | ❌ | "text" | Content type classification |
| `source` | `string` | ❌ | `null` | Optional source identifier |
| `tags` | `array[string]` | ❌ | `[]` | Tags for categorization |
| `metadata` | `object` | ❌ | `{}` | Additional metadata dictionary |
| `embedding_model` | `string` | ❌ | `null` | Override default embedding model |

**Response:**

```json
{
  "status": "success",
  "id": "abc123def456",
  "collection_name": "knowledge_base",
  "vector_dimensions": 384,
  "embedding_time_ms": 45.2,
  "storage_time_ms": 12.8,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Store a research paper abstract
result = await session.call_tool(
    "store_vector_document",
    {
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn patterns and make predictions.",
        "collection_name": "research_papers",
        "content_type": "abstract",
        "tags": ["machine-learning", "artificial-intelligence", "algorithms"],
        "metadata": {
            "author": "Smith et al.",
            "year": 2024,
            "journal": "AI Research Quarterly",
            "doi": "10.1000/ai.2024.001"
        },
        "source": "research_paper_abstract_001"
    }
)
```

**Error Handling:**

```json
{
  "status": "error",
  "error": "Collection 'invalid_collection' does not exist",
  "error_type": "VectorServiceError"
}
```

---

### semantic_vector_search

Perform semantic similarity search across vector embeddings to find relevant content.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `string` | ✅ | - | Natural language search query (1-1,000 chars) |
| `collection_name` | `string` | ❌ | "documents" | Collection to search |
| `limit` | `integer` | ❌ | 10 | Maximum results (1-100) |
| `score_threshold` | `float` | ❌ | 0.0 | Minimum similarity score (0.0-1.0) |
| `mode` | `string` | ❌ | "semantic" | Search mode: semantic/hybrid/exact |
| `content_type` | `string` | ❌ | `null` | Filter by content type |
| `tags` | `array[string]` | ❌ | `null` | Filter by tags (must contain all) |
| `include_vectors` | `boolean` | ❌ | `false` | Include vector embeddings in response |
| `include_content` | `boolean` | ❌ | `true` | Include original content in response |

**Response:**

```json
{
  "status": "success",
  "query": "artificial intelligence machine learning",
  "collection_name": "research_papers",
  "results": [
    {
      "id": "abc123",
      "score": 0.89,
      "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience...",
      "content_type": "abstract",
      "source": "research_paper_abstract_001",
      "tags": ["machine-learning", "artificial-intelligence"],
      "metadata": {
        "author": "Smith et al.",
        "year": 2024
      }
    }
  ],
  "total_found": 1,
  "search_time_ms": 23.4,
  "mode": "semantic",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Search for machine learning content
result = await session.call_tool(
    "semantic_vector_search",
    {
        "query": "deep learning neural networks computer vision",
        "collection_name": "research_papers",
        "limit": 5,
        "score_threshold": 0.75,
        "tags": ["deep-learning"],
        "include_content": True
    }
)

# Process results
for item in result.content['results']:
    print(f"Relevance: {item['score']:.2f}")
    print(f"Title: {item['metadata'].get('title', 'Untitled')}")
    print(f"Content: {item['content'][:200]}...")
    print("---")
```

**Advanced Filtering:**

```python
# Search with complex metadata filters
result = await session.call_tool(
    "semantic_vector_search",
    {
        "query": "transformer architecture attention mechanisms",
        "collection_name": "research_papers",
        "limit": 10,
        "score_threshold": 0.8,
        "filters": {
            "year": {"gte": 2020},
            "journal": {"in": ["NeurIPS", "ICML", "ICLR"]},
            "citation_count": {"gt": 100}
        }
    }
)
```

---

### create_vector_collection

Create a new vector collection with specified configuration for storing embeddings.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `string` | ✅ | - | Collection name (1-64 chars, alphanumeric + hyphens/underscores) |
| `vector_size` | `integer` | ✅ | - | Vector dimensions (1-2048, must match embedding model) |
| `distance_metric` | `string` | ❌ | "Cosine" | Distance metric: Cosine/Dot/Euclidean/Manhattan |
| `description` | `string` | ❌ | `null` | Optional collection description |

**Response:**

```json
{
  "status": "success",
  "name": "research_papers",
  "vector_size": 768,
  "distance_metric": "Cosine",
  "created": true,
  "description": "Academic research papers collection"
}
```

**Example Usage:**

```python
# Create collection for BERT embeddings
result = await session.call_tool(
    "create_vector_collection",
    {
        "name": "bert_embeddings",
        "vector_size": 768,
        "distance_metric": "Cosine",
        "description": "Collection for BERT-base-uncased embeddings"
    }
)

# Create collection for OpenAI embeddings
result = await session.call_tool(
    "create_vector_collection",
    {
        "name": "openai_embeddings",
        "vector_size": 1536,
        "distance_metric": "Cosine",
        "description": "Collection for OpenAI text-embedding-ada-002"
    }
)
```

**Distance Metrics Guide:**

| Metric | Best For | Range | Description |
|--------|----------|-------|-------------|
| `Cosine` | Text similarity | 0-2 | Measures angle between vectors |
| `Dot` | Normalized vectors | -∞ to +∞ | Dot product similarity |
| `Euclidean` | Spatial data | 0 to +∞ | Geometric distance |
| `Manhattan` | High dimensions | 0 to +∞ | Sum of absolute differences |

---

### list_vector_collections

List all available vector collections with their statistics and configuration.

**Parameters:** None

**Response:**

```json
{
  "status": "success",
  "collections": [
    {
      "name": "research_papers",
      "status": "active",
      "vector_size": 768,
      "distance_metric": "Cosine",
      "points_count": 15420,
      "indexed_vectors": 15420,
      "segments_count": 3,
      "disk_usage_bytes": 157286400,
      "disk_usage_mb": 150.0,
      "ram_usage_bytes": 52428800,
      "ram_usage_mb": 50.0,
      "created_at": "2024-01-10T09:00:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_collections": 1,
  "total_vectors": 15420,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# List all collections
result = await session.call_tool("list_vector_collections")

# Display collection information
for collection in result.content['collections']:
    print(f"Collection: {collection['name']}")
    print(f"  Status: {collection['status']}")
    print(f"  Vectors: {collection['points_count']:,}")
    print(f"  Dimensions: {collection['vector_size']}")
    print(f"  Storage: {collection['disk_usage_mb']:.1f} MB")
    print(f"  Memory: {collection['ram_usage_mb']:.1f} MB")
    print()
```

---

### generate_text_embeddings

Generate vector embeddings for text inputs using pre-trained models.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `texts` | `array[string]` | ✅ | - | List of texts to embed (1-100 items, each 1-10,000 chars) |
| `model` | `string` | ❌ | `null` | Override default embedding model |
| `normalize` | `boolean` | ❌ | `true` | Whether to normalize vectors to unit length |

**Response:**

```json
{
  "status": "success",
  "embeddings": [
    [0.1, 0.2, -0.3, 0.4, ...],
    [0.4, -0.1, 0.8, 0.2, ...]
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimensions": 384,
  "processing_time_ms": 42.1,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Generate embeddings for similarity analysis
texts = [
    "Machine learning algorithms for classification",
    "Deep neural networks for image recognition",
    "Natural language processing with transformers"
]

result = await session.call_tool(
    "generate_text_embeddings",
    {
        "texts": texts,
        "normalize": True
    }
)

# Use embeddings for similarity calculation
embeddings = result.content['embeddings']
from scipy.spatial.distance import cosine

similarity = 1 - cosine(embeddings[0], embeddings[1])
print(f"Similarity between first two texts: {similarity:.3f}")
```

**Supported Models:**

| Model | Dimensions | Max Length | Best For |
|-------|------------|------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 tokens | General text similarity |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 384 tokens | High-quality embeddings |
| `text-embedding-ada-002` | 1536 | 8191 tokens | OpenAI embeddings |
| `text-embedding-3-small` | 1536 | 8191 tokens | Latest OpenAI model |

---

### delete_vector_collection

Permanently delete a vector collection and all its data.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection_name` | `string` | ✅ | - | Name of collection to delete |
| `confirm` | `boolean` | ✅ | `false` | Must be `true` to confirm deletion |

**Response:**

```json
{
  "status": "success",
  "collection_name": "test_collection",
  "deleted": true,
  "warning": "Collection and all data permanently deleted"
}
```

**Example Usage:**

```python
# Delete a test collection
result = await session.call_tool(
    "delete_vector_collection",
    {
        "collection_name": "test_collection",
        "confirm": True
    }
)

# Safety check - confirmation required
try:
    result = await session.call_tool(
        "delete_vector_collection",
        {
            "collection_name": "important_data",
            "confirm": False  # Will fail
        }
    )
except Exception as e:
    print("Deletion prevented - confirmation required")
```

**⚠️ Warning:** This operation is irreversible. All vectors and metadata in the collection will be permanently deleted.

---

### get_vector_service_stats

Get comprehensive statistics about the vector service for monitoring and optimization.

**Parameters:** None

**Response:**

```json
{
  "status": "success",
  "stats": {
    "total_collections": 3,
    "total_vectors": 15420,
    "total_disk_usage_bytes": 314572800,
    "total_disk_usage_mb": 300.0,
    "total_ram_usage_bytes": 104857600,
    "total_ram_usage_mb": 100.0,
    "average_search_time_ms": 18.2,
    "embeddings_generated": 15420,
    "last_updated": "2024-01-15T10:30:00Z"
  },
  "health": {
    "service": "vector",
    "status": "healthy",
    "response_time_ms": 15.3,
    "details": {
      "qdrant_version": "1.7.0",
      "collections_healthy": 3,
      "total_points": 15420
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Example Usage:**

```python
# Get service statistics
result = await session.call_tool("get_vector_service_stats")

stats = result.content['stats']
health = result.content['health']

print(f"Vector Service Status: {health['status']}")
print(f"Collections: {stats['total_collections']}")
print(f"Total Vectors: {stats['total_vectors']:,}")
print(f"Storage Usage: {stats['total_disk_usage_mb']:.1f} MB")
print(f"Memory Usage: {stats['total_ram_usage_mb']:.1f} MB")
print(f"Avg Search Time: {stats['average_search_time_ms']:.1f}ms")

# Performance monitoring
if stats['average_search_time_ms'] > 50:
    print("⚠️ Search performance degraded")
if health['status'] != 'healthy':
    print("🚨 Service health issue detected")
```

## Best Practices

### Content Preparation

1. **Text Preprocessing:**
   ```python
   import re
   
   def clean_text(text):
       # Remove extra whitespace
       text = re.sub(r'\s+', ' ', text.strip())
       # Remove special characters if needed
       text = re.sub(r'[^\w\s\-\.]', '', text)
       return text
   
   content = clean_text(raw_content)
   ```

2. **Chunking Strategy:**
   ```python
   def chunk_text(text, max_length=1000, overlap=100):
       chunks = []
       start = 0
       while start < len(text):
           end = start + max_length
           chunk = text[start:end]
           chunks.append(chunk)
           start = end - overlap
       return chunks
   ```

### Collection Management

1. **Choose Appropriate Vector Dimensions:**
   - Small models (384d): Fast, good for simple similarity
   - Medium models (768d): Balanced performance and quality
   - Large models (1536d): High quality, more storage

2. **Distance Metric Selection:**
   - Use `Cosine` for most text applications
   - Use `Dot` for pre-normalized vectors
   - Use `Euclidean` for spatial/numerical data

3. **Collection Naming:**
   ```python
   # Good collection names
   "documents_2024"
   "research-papers"
   "user_content"
   
   # Avoid
   "Collection 1"  # Contains spaces
   "TEMP"          # Too generic
   ```

### Search Optimization

1. **Query Formulation:**
   ```python
   # Good queries
   "machine learning classification algorithms"
   "neural networks computer vision applications"
   
   # Less effective
   "ML"           # Too short
   "the and or"   # Only stop words
   ```

2. **Filter Usage:**
   ```python
   # Efficient filtering
   filters = {
       "content_type": "research_paper",
       "year": {"gte": 2020},
       "tags": {"in": ["AI", "ML"]}
   }
   ```

3. **Threshold Selection:**
   - `0.7+`: High similarity, fewer results
   - `0.5-0.7`: Moderate similarity, balanced
   - `0.0-0.5`: Low similarity, more results

### Performance Monitoring

```python
async def monitor_vector_service():
    stats = await session.call_tool("get_vector_service_stats")
    
    # Check performance metrics
    avg_time = stats.content['stats']['average_search_time_ms']
    if avg_time > 100:
        print(f"⚠️ Slow searches: {avg_time:.1f}ms")
    
    # Check storage usage
    disk_mb = stats.content['stats']['total_disk_usage_mb']
    if disk_mb > 1000:  # 1GB threshold
        print(f"⚠️ High storage usage: {disk_mb:.1f}MB")
    
    # Check health status
    health = stats.content['health']['status']
    if health != 'healthy':
        print(f"🚨 Service unhealthy: {health}")
```

## Error Handling

### Common Error Types

```python
# Handle collection errors
try:
    result = await session.call_tool("semantic_vector_search", {
        "query": "test query",
        "collection_name": "nonexistent"
    })
except Exception as e:
    if "not exist" in str(e):
        print("Collection doesn't exist")
        # Create collection first
        await session.call_tool("create_vector_collection", {
            "name": "nonexistent",
            "vector_size": 384
        })

# Handle embedding errors
try:
    result = await session.call_tool("generate_text_embeddings", {
        "texts": [""],  # Empty text
    })
except Exception as e:
    print(f"Embedding error: {e}")

# Handle quota/rate limiting
try:
    result = await session.call_tool("store_vector_document", {
        "content": "test" * 10000  # Too long
    })
except Exception as e:
    if "rate limit" in str(e):
        print("Rate limited - wait before retrying")
    elif "too long" in str(e):
        print("Content too long - chunk the text")
```

### Retry Logic

```python
import asyncio
import random

async def retry_with_backoff(tool_name, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await session.call_tool(tool_name, params)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
            print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
```

## Integration Examples

### RAG Pipeline

```python
async def rag_search(query, context_limit=5):
    """Retrieval-Augmented Generation search pipeline."""
    
    # 1. Semantic search for relevant context
    search_result = await session.call_tool(
        "semantic_vector_search",
        {
            "query": query,
            "collection_name": "knowledge_base",
            "limit": context_limit,
            "score_threshold": 0.7
        }
    )
    
    # 2. Extract relevant content
    contexts = []
    for result in search_result.content['results']:
        contexts.append({
            "content": result['content'],
            "score": result['score'],
            "source": result.get('source', 'unknown')
        })
    
    # 3. Rank by relevance and recency
    contexts.sort(key=lambda x: (x['score'], x.get('timestamp', 0)), reverse=True)
    
    return {
        "query": query,
        "contexts": contexts[:context_limit],
        "total_found": search_result.content['total_found']
    }
```

### Document Indexing

```python
async def index_document(content, metadata=None):
    """Index a document with proper chunking and metadata."""
    
    # 1. Clean and chunk content
    clean_content = clean_text(content)
    chunks = chunk_text(clean_content, max_length=800, overlap=100)
    
    # 2. Store each chunk
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        result = await session.call_tool(
            "store_vector_document",
            {
                "content": chunk,
                "collection_name": "documents",
                "content_type": "text_chunk",
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk)
                }
            }
        )
        chunk_ids.append(result.content['id'])
    
    return {
        "document_id": metadata.get('id') if metadata else None,
        "chunks_stored": len(chunks),
        "chunk_ids": chunk_ids
    }
```

---

*For more information, see the [API Overview](../README.md) or explore [Graph Tools](./graph-tools.md) and [Web Tools](./web-tools.md).*