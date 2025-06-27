# Knowledge Graph Tools

The Graph Intelligence service provides knowledge graph operations, relationship analysis, and memory management capabilities through Neo4j integration. These MCP tools enable AI assistants to build, query, and analyze complex knowledge structures and relationships.

## Overview

Graph tools enable sophisticated knowledge representation through:

- **Knowledge Graphs**: Store entities, concepts, and their relationships
- **Memory Systems**: Create persistent memory for AI assistants
- **Relationship Analysis**: Discover patterns and connections
- **Network Analytics**: Analyze graph structure and influence
- **Knowledge Extraction**: AI-powered entity and relationship extraction

## Available Tools

### create_graph_node

Create a new node in the knowledge graph to represent entities, concepts, or memories.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `string` | ✅ | - | Node name or title |
| `node_type` | `string` | ✅ | - | Type: Entity, Concept, Person, Event, Memory, etc. |
| `description` | `string` | ❌ | `null` | Optional description |
| `properties` | `object` | ❌ | `{}` | Additional node properties |
| `tags` | `array[string]` | ❌ | `[]` | Classification tags |
| `emotional_valence` | `float` | ❌ | `null` | Emotional sentiment (-1 to 1) |
| `abstraction_level` | `integer` | ❌ | `null` | Abstraction level (1-10) |
| `confidence_score` | `float` | ❌ | 0.5 | Confidence in node data (0-1) |
| `source` | `string` | ❌ | `null` | Data source or origin |

**Response:**

```json
{
  "success": true,
  "node_id": "node_abc123",
  "name": "Machine Learning",
  "node_type": "Concept",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Create a concept node
result = await session.call_tool(
    "create_graph_node",
    {
        "name": "Machine Learning",
        "node_type": "Concept",
        "description": "A subset of artificial intelligence focusing on algorithms that learn from data",
        "properties": {
            "field": "Computer Science",
            "difficulty": "Intermediate",
            "applications": ["Classification", "Regression", "Clustering"]
        },
        "tags": ["AI", "Data Science", "Technology"],
        "abstraction_level": 7,
        "confidence_score": 0.9,
        "source": "academic_definition"
    }
)

# Create a person node
person_result = await session.call_tool(
    "create_graph_node",
    {
        "name": "Geoffrey Hinton",
        "node_type": "Person",
        "description": "Pioneer in deep learning and neural networks",
        "properties": {
            "profession": "Computer Scientist",
            "affiliation": "University of Toronto",
            "notable_work": "Backpropagation algorithm",
            "awards": ["Turing Award"]
        },
        "tags": ["AI Pioneer", "Deep Learning", "Academic"],
        "confidence_score": 1.0,
        "source": "biographical_data"
    }
)
```

**Node Types:**

| Type | Description | Use Cases |
|------|-------------|-----------|
| `Entity` | Concrete objects, things | Organizations, products, locations |
| `Concept` | Abstract ideas, topics | Theories, methodologies, fields |
| `Person` | Individual people | Authors, researchers, historical figures |
| `Event` | Temporal occurrences | Conferences, discoveries, milestones |
| `Memory` | AI assistant memories | Conversations, preferences, observations |
| `Document` | Information sources | Papers, books, articles |
| `Skill` | Abilities or competencies | Programming languages, techniques |

---

### create_graph_relationship

Create a relationship between two nodes in the knowledge graph.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | `string` | ✅ | - | Source node ID |
| `target_id` | `string` | ✅ | - | Target node ID |
| `relationship_type` | `string` | ✅ | - | Type of relationship |
| `properties` | `object` | ❌ | `{}` | Relationship properties |
| `weight` | `float` | ❌ | 1.0 | Relationship strength (0-1) |
| `confidence` | `float` | ❌ | 0.5 | Confidence in relationship (0-1) |
| `evidence` | `array[string]` | ❌ | `[]` | Evidence supporting relationship |

**Response:**

```json
{
  "success": true,
  "relationship_id": "rel_xyz789",
  "relationship_type": "DEVELOPED_BY",
  "source_id": "node_abc123",
  "target_id": "node_def456",
  "weight": 0.9,
  "confidence": 0.85,
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Create a "developed by" relationship
result = await session.call_tool(
    "create_graph_relationship",
    {
        "source_id": "concept_neural_networks",
        "target_id": "person_geoffrey_hinton",
        "relationship_type": "DEVELOPED_BY",
        "properties": {
            "year": 1986,
            "contribution": "Backpropagation algorithm",
            "impact": "Revolutionary"
        },
        "weight": 0.95,
        "confidence": 0.9,
        "evidence": [
            "Learning representations by back-propagating errors (1986)",
            "Turing Award citation",
            "Academic consensus"
        ]
    }
)

# Create a semantic relationship
concept_result = await session.call_tool(
    "create_graph_relationship",
    {
        "source_id": "concept_machine_learning",
        "target_id": "concept_artificial_intelligence",
        "relationship_type": "SUBSET_OF",
        "properties": {
            "specificity": "More specific",
            "overlap": "High"
        },
        "weight": 0.8,
        "confidence": 0.95
    }
)
```

**Relationship Types:**

| Type | Description | Example |
|------|-------------|---------|
| `SUBSET_OF` | Hierarchical inclusion | ML → AI |
| `RELATED_TO` | General association | AI ↔ Robotics |
| `DEVELOPED_BY` | Creation attribution | Algorithm → Person |
| `INFLUENCES` | Impact relationship | Theory → Practice |
| `COLLABORATED_WITH` | Professional relationship | Person ↔ Person |
| `PUBLISHED_IN` | Publication venue | Paper → Journal |
| `APPLIES_TO` | Application domain | Algorithm → Problem |
| `PRECEDED_BY` | Temporal sequence | Event → Event |
| `REMEMBERS` | Memory association | Memory → Entity |

---

### search_graph

Search the knowledge graph for nodes and relationships using various criteria.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `string` | ✅ | - | Search query |
| `node_types` | `array[string]` | ❌ | `null` | Node types to search |
| `relationship_types` | `array[string]` | ❌ | `null` | Relationship types to include |
| `max_depth` | `integer` | ❌ | 2 | Maximum traversal depth |
| `limit` | `integer` | ❌ | 10 | Maximum results |
| `confidence_threshold` | `float` | ❌ | 0.0 | Minimum confidence threshold |
| `use_embeddings` | `boolean` | ❌ | `false` | Use vector embeddings for search |
| `embedding_similarity_threshold` | `float` | ❌ | 0.7 | Embedding similarity threshold |

**Response:**

```json
{
  "success": true,
  "total_results": 5,
  "search_time_ms": 45.2,
  "query_type": "semantic_search",
  "nodes": [
    {
      "id": "node_abc123",
      "name": "Machine Learning",
      "node_type": "Concept",
      "description": "A subset of artificial intelligence...",
      "confidence_score": 0.9,
      "tags": ["AI", "Data Science"]
    }
  ],
  "relationships": [
    {
      "id": "rel_xyz789",
      "source_id": "node_abc123",
      "target_id": "node_def456",
      "relationship_type": "SUBSET_OF",
      "weight": 0.8,
      "confidence": 0.9
    }
  ],
  "paths": [],
  "confidence_scores": [0.9, 0.85, 0.8],
  "filters_applied": ["node_types", "confidence_threshold"]
}
```

**Example Usage:**

```python
# Search for AI-related concepts
result = await session.call_tool(
    "search_graph",
    {
        "query": "artificial intelligence neural networks",
        "node_types": ["Concept", "Person"],
        "max_depth": 3,
        "limit": 10,
        "confidence_threshold": 0.7,
        "use_embeddings": True
    }
)

# Process search results
for node in result.content['nodes']:
    print(f"Found: {node['name']} ({node['node_type']})")
    print(f"  Confidence: {node['confidence_score']:.2f}")
    print(f"  Tags: {', '.join(node['tags'])}")

# Find relationship patterns
for rel in result.content['relationships']:
    print(f"Relationship: {rel['relationship_type']}")
    print(f"  Weight: {rel['weight']:.2f}")

# Advanced search with specific relationships
advanced_result = await session.call_tool(
    "search_graph",
    {
        "query": "deep learning pioneers",
        "node_types": ["Person"],
        "relationship_types": ["DEVELOPED_BY", "COLLABORATED_WITH"],
        "max_depth": 2,
        "limit": 15
    }
)
```

---

### extract_knowledge_from_text

Extract knowledge from text using AI-powered analysis to create nodes and relationships.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | `string` | ✅ | - | Text to extract knowledge from |
| `extract_entities` | `boolean` | ❌ | `true` | Extract entities |
| `extract_relationships` | `boolean` | ❌ | `true` | Extract relationships |
| `extract_concepts` | `boolean` | ❌ | `true` | Extract concepts |
| `merge_similar_entities` | `boolean` | ❌ | `true` | Merge similar entities |
| `confidence_threshold` | `float` | ❌ | 0.5 | Extraction confidence threshold |
| `max_entities` | `integer` | ❌ | 50 | Maximum entities to extract |
| `source_url` | `string` | ❌ | `null` | Source URL |
| `source_type` | `string` | ❌ | `null` | Source type |
| `document_id` | `string` | ❌ | `null` | Document identifier |

**Response:**

```json
{
  "success": true,
  "total_entities": 8,
  "total_relationships": 12,
  "processing_time_ms": 1250.5,
  "average_confidence": 0.78,
  "low_confidence_items": 2,
  "extracted_nodes": [
    {
      "id": "extracted_node_1",
      "name": "Transformer Architecture",
      "node_type": "Concept",
      "description": "A neural network architecture based on self-attention mechanisms",
      "confidence_score": 0.85,
      "emotional_valence": 0.1,
      "abstraction_level": 7,
      "tags": ["AI", "Deep Learning", "NLP"],
      "source": "research_paper_extract"
    }
  ],
  "extracted_relationships": [
    {
      "id": "extracted_rel_1",
      "source_id": "extracted_node_1",
      "target_id": "extracted_node_2",
      "relationship_type": "USES",
      "weight": 0.8,
      "confidence": 0.75,
      "evidence": ["attention mechanism", "encoder-decoder structure"]
    }
  ],
  "source_metadata": {
    "source_url": "https://arxiv.org/abs/1706.03762",
    "source_type": "research_paper",
    "document_id": "transformer_paper"
  }
}
```

**Example Usage:**

```python
# Extract knowledge from research paper abstract
text = """
The Transformer architecture, introduced by Vaswani et al. in 2017, 
revolutionized natural language processing. Unlike recurrent neural networks, 
Transformers rely entirely on attention mechanisms to process sequences. 
The architecture consists of an encoder-decoder structure with multi-head 
self-attention layers. This approach has led to breakthroughs in machine 
translation, text summarization, and language modeling, forming the foundation 
for models like BERT and GPT.
"""

result = await session.call_tool(
    "extract_knowledge_from_text",
    {
        "text": text,
        "extract_entities": True,
        "extract_relationships": True,
        "extract_concepts": True,
        "confidence_threshold": 0.6,
        "max_entities": 20,
        "source_url": "https://arxiv.org/abs/1706.03762",
        "source_type": "research_paper",
        "document_id": "transformer_paper_2017"
    }
)

# Process extracted knowledge
print(f"Extracted {result.content['total_entities']} entities")
print(f"Extracted {result.content['total_relationships']} relationships")

for node in result.content['extracted_nodes']:
    print(f"Entity: {node['name']} ({node['node_type']})")
    print(f"  Confidence: {node['confidence_score']:.2f}")
    print(f"  Description: {node['description'][:100]}...")

# Batch process multiple documents
documents = [
    {"text": abstract1, "source": "paper1"},
    {"text": abstract2, "source": "paper2"},
    {"text": abstract3, "source": "paper3"}
]

extracted_knowledge = []
for doc in documents:
    result = await session.call_tool(
        "extract_knowledge_from_text",
        {
            "text": doc["text"],
            "source_type": "research_abstract",
            "document_id": doc["source"]
        }
    )
    extracted_knowledge.append(result.content)
```

---

### create_memory_node

Create a memory node for AI assistant memory systems.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | `string` | ✅ | - | Memory subject name |
| `memory_type` | `string` | ❌ | "general" | Type: general, person, concept, etc. |
| `observations` | `array[string]` | ❌ | `null` | Observed attributes |
| `context` | `string` | ❌ | `null` | Memory context |
| `psychological_profile` | `string` | ❌ | `null` | Psychological characteristics |
| `values` | `array[string]` | ❌ | `null` | Core values |

**Response:**

```json
{
  "success": true,
  "memory_id": "memory_abc123",
  "name": "John Smith",
  "memory_type": "person",
  "observations": ["Prefers email communication", "Expert in machine learning"],
  "insights": ["Technical expertise", "Communication preferences"],
  "context": "Professional colleague",
  "psychological_profile": "Detail-oriented, prefers structured approaches",
  "values": ["Accuracy", "Efficiency"],
  "created_at": "2024-01-15T10:30:00Z",
  "access_count": 0
}
```

**Example Usage:**

```python
# Create memory for a person
result = await session.call_tool(
    "create_memory_node",
    {
        "name": "Dr. Sarah Chen",
        "memory_type": "person",
        "observations": [
            "Lead researcher in computer vision",
            "Published 50+ papers in top venues",
            "Prefers collaborative work environments",
            "Uses Python and PyTorch primarily"
        ],
        "context": "Research collaboration",
        "psychological_profile": "Analytical, collaborative, innovation-focused",
        "values": ["Scientific rigor", "Open science", "Mentorship"]
    }
)

# Create memory for a project
project_memory = await session.call_tool(
    "create_memory_node",
    {
        "name": "NLP Pipeline Project",
        "memory_type": "project",
        "observations": [
            "Uses transformer-based models",
            "Handles multilingual text",
            "Performance target: 95% accuracy",
            "Timeline: 6 months"
        ],
        "context": "Current active project"
    }
)

# Create conversation memory
conversation_memory = await session.call_tool(
    "create_memory_node",
    {
        "name": "ML Discussion with Team",
        "memory_type": "conversation",
        "observations": [
            "Discussed model architecture choices",
            "Team prefers ensemble methods",
            "Budget constraints limit computational resources",
            "Next meeting scheduled for Friday"
        ],
        "context": "Weekly team meeting"
    }
)
```

---

### execute_cypher_query

Execute raw Cypher queries against the Neo4j database for advanced operations.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `string` | ✅ | - | Cypher query string |
| `parameters` | `object` | ❌ | `{}` | Query parameters |
| `read_only` | `boolean` | ❌ | `true` | Read-only query flag |
| `timeout` | `integer` | ❌ | `null` | Query timeout in seconds |
| `limit` | `integer` | ❌ | `null` | Result limit |
| `include_stats` | `boolean` | ❌ | `false` | Include execution statistics |

**Response:**

```json
{
  "success": true,
  "records": [
    {
      "n.name": "Machine Learning",
      "n.node_type": "Concept",
      "count": 15
    }
  ],
  "execution_time_ms": 23.4,
  "records_available": 1,
  "stats": {
    "nodes_created": 0,
    "relationships_created": 0,
    "properties_set": 0
  },
  "error": null,
  "error_code": null
}
```

**Example Usage:**

```python
# Find most connected nodes
result = await session.call_tool(
    "execute_cypher_query",
    {
        "query": """
        MATCH (n)-[r]-()
        RETURN n.name AS name, n.node_type AS type, count(r) AS connections
        ORDER BY connections DESC
        LIMIT 10
        """,
        "read_only": True,
        "include_stats": True
    }
)

# Parameterized query for security
search_result = await session.call_tool(
    "execute_cypher_query",
    {
        "query": """
        MATCH (n:Concept)
        WHERE n.name CONTAINS $search_term
        RETURN n.name, n.description, n.confidence_score
        ORDER BY n.confidence_score DESC
        """,
        "parameters": {"search_term": "machine learning"},
        "read_only": True,
        "limit": 20
    }
)

# Complex path analysis
path_result = await session.call_tool(
    "execute_cypher_query",
    {
        "query": """
        MATCH path = (start:Person {name: $person_name})-[*1..3]-(end:Concept)
        WHERE end.name CONTAINS $concept_term
        RETURN path, length(path) as path_length
        ORDER BY path_length
        """,
        "parameters": {
            "person_name": "Geoffrey Hinton",
            "concept_term": "neural"
        },
        "read_only": True
    }
)
```

**Common Cypher Patterns:**

```cypher
-- Find all concepts related to AI
MATCH (ai:Concept {name: "Artificial Intelligence"})-[r]-(related)
RETURN related.name, type(r), r.weight

-- Get node degree centrality
MATCH (n)-[r]-()
RETURN n.name, count(r) AS degree
ORDER BY degree DESC

-- Find shortest path between nodes
MATCH path = shortestPath((a)-[*]-(b))
WHERE a.name = "Machine Learning" AND b.name = "Computer Vision"
RETURN path

-- Community detection
CALL gds.louvain.stream('graph-projection')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId

-- Temporal analysis
MATCH (n)-[r]->(m)
WHERE r.created_at > datetime('2024-01-01')
RETURN n.name, m.name, r.created_at
ORDER BY r.created_at DESC
```

---

### analyze_graph_structure

Analyze graph structure and compute network metrics for insights.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `focus_node_id` | `string` | ❌ | `null` | Node to focus analysis on |
| `analysis_type` | `string` | ❌ | "centrality" | Type: centrality, community, paths |
| `depth` | `integer` | ❌ | 3 | Analysis depth |
| `include_metrics` | `boolean` | ❌ | `true` | Include graph metrics |
| `include_communities` | `boolean` | ❌ | `false` | Include community detection |
| `include_paths` | `boolean` | ❌ | `false` | Include shortest paths |
| `node_types` | `array[string]` | ❌ | `null` | Node types to include |
| `relationship_types` | `array[string]` | ❌ | `null` | Relationship types to include |

**Response:**

```json
{
  "success": true,
  "analysis_type": "centrality",
  "focus_node": {
    "id": "node_abc123",
    "name": "Machine Learning",
    "node_type": "Concept"
  },
  "node_count": 150,
  "relationship_count": 320,
  "density": 0.028,
  "average_clustering": 0.45,
  "centrality_measures": {
    "degree_centrality": {
      "node_abc123": 0.15,
      "node_def456": 0.12
    },
    "betweenness_centrality": {
      "node_abc123": 0.08,
      "node_def456": 0.06
    },
    "closeness_centrality": {
      "node_abc123": 0.22,
      "node_def456": 0.18
    }
  },
  "influential_nodes": [
    {"id": "node_abc123", "name": "Machine Learning", "influence_score": 0.85},
    {"id": "node_def456", "name": "Neural Networks", "influence_score": 0.78}
  ],
  "communities": [],
  "modularity": 0.0,
  "shortest_paths": [],
  "temporal_patterns": {},
  "analysis_time_ms": 456.7,
  "confidence": 0.9
}
```

**Example Usage:**

```python
# Analyze centrality around a specific concept
result = await session.call_tool(
    "analyze_graph_structure",
    {
        "focus_node_id": "concept_machine_learning",
        "analysis_type": "centrality",
        "depth": 3,
        "include_metrics": True
    }
)

# Find most influential nodes
influential = result.content['influential_nodes']
for node in influential[:5]:
    print(f"{node['name']}: {node['influence_score']:.2f}")

# Community detection analysis
community_result = await session.call_tool(
    "analyze_graph_structure",
    {
        "analysis_type": "community",
        "include_communities": True,
        "node_types": ["Concept", "Person"],
        "depth": 4
    }
)

# Path analysis between concepts
path_result = await session.call_tool(
    "analyze_graph_structure",
    {
        "analysis_type": "paths",
        "include_paths": True,
        "relationship_types": ["RELATED_TO", "SUBSET_OF"],
        "depth": 3
    }
)

# Network overview
overview = await session.call_tool(
    "analyze_graph_structure",
    {
        "analysis_type": "centrality",
        "include_metrics": True,
        "include_communities": True
    }
)

print(f"Graph Overview:")
print(f"  Nodes: {overview.content['node_count']}")
print(f"  Relationships: {overview.content['relationship_count']}")
print(f"  Density: {overview.content['density']:.3f}")
print(f"  Clustering: {overview.content['average_clustering']:.3f}")
```

**Analysis Types:**

| Type | Description | Metrics Included |
|------|-------------|------------------|
| `centrality` | Node importance analysis | Degree, betweenness, closeness centrality |
| `community` | Community detection | Modularity, community assignments |
| `paths` | Path analysis | Shortest paths, path distributions |
| `temporal` | Time-based patterns | Evolution, growth patterns |

---

### get_graph_health

Get Neo4j graph service health status and statistics.

**Parameters:** None

**Response:**

```json
{
  "success": true,
  "status": "healthy",
  "database_connected": true,
  "response_time_ms": 15.3,
  "total_nodes": 1520,
  "total_relationships": 3240,
  "node_types_count": 8,
  "relationship_types_count": 12,
  "memory_usage_mb": 256.5,
  "disk_usage_mb": 1024.2,
  "average_query_time_ms": 25.4,
  "neo4j_version": "5.15.0",
  "driver_version": "5.15.0",
  "errors": [],
  "warnings": [],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Check service health
health = await session.call_tool("get_graph_health")

print(f"Graph Service: {health.content['status']}")
print(f"Database Connected: {health.content['database_connected']}")
print(f"Response Time: {health.content['response_time_ms']:.1f}ms")
print(f"Total Nodes: {health.content['total_nodes']:,}")
print(f"Total Relationships: {health.content['total_relationships']:,}")
print(f"Memory Usage: {health.content['memory_usage_mb']:.1f} MB")

# Monitor performance
if health.content['average_query_time_ms'] > 100:
    print("⚠️ Slow query performance detected")

if health.content['errors']:
    print("🚨 Errors detected:")
    for error in health.content['errors']:
        print(f"  - {error}")

# Health check automation
async def monitor_graph_health():
    health = await session.call_tool("get_graph_health")
    
    metrics = {
        "status": health.content['status'],
        "response_time": health.content['response_time_ms'],
        "nodes": health.content['total_nodes'],
        "relationships": health.content['total_relationships'],
        "memory_mb": health.content['memory_usage_mb']
    }
    
    # Alert conditions
    if metrics['status'] != 'healthy':
        send_alert(f"Graph service unhealthy: {metrics['status']}")
    
    if metrics['response_time'] > 50:
        send_alert(f"Slow response time: {metrics['response_time']:.1f}ms")
    
    return metrics
```

## Best Practices

### Node Design

1. **Consistent Naming:**

   ```python
   # Good naming patterns
   "Geoffrey Hinton"  # Person names: First Last
   "Machine Learning"  # Concepts: Title Case
   "NeurIPS 2023"     # Events: Name + Year
   
   # Avoid
   "geoffrey hinton"   # Inconsistent case
   "ML"               # Abbreviations without context
   ```

2. **Meaningful Properties:**

   ```python
   # Rich node properties
   {
       "name": "Transformer Architecture",
       "node_type": "Concept",
       "properties": {
           "introduced_year": 2017,
           "primary_application": "Natural Language Processing",
           "key_innovation": "Self-attention mechanism",
           "complexity": "High",
           "impact_score": 9.5
       }
   }
   ```

3. **Appropriate Abstraction Levels:**
   - Level 1-3: Concrete, specific instances
   - Level 4-6: General concepts, categories
   - Level 7-10: Abstract theories, principles

### Relationship Modeling

1. **Directional Relationships:**

   ```python
   # Clear direction and semantics
   "Neural Networks" --SUBSET_OF--> "Machine Learning"
   "Geoffrey Hinton" --DEVELOPED--> "Backpropagation"
   "BERT" --BASED_ON--> "Transformer Architecture"
   ```

2. **Weighted Relationships:**

   ```python
   # Use weights to indicate strength
   {
       "relationship_type": "INFLUENCES",
       "weight": 0.9,  # Strong influence
       "confidence": 0.85,
       "evidence": ["Multiple citations", "Acknowledged by authors"]
   }
   ```

### Query Optimization

1. **Use Indexes:**

   ```cypher
   -- Create indexes for frequent queries
   CREATE INDEX node_name_index FOR (n:Node) ON (n.name)
   CREATE INDEX concept_type_index FOR (n:Concept) ON (n.node_type)
   ```

2. **Limit Results:**

   ```python
   # Always use LIMIT in queries
   query = """
   MATCH (n:Concept)-[r:RELATED_TO]-(m:Concept)
   WHERE n.confidence_score > 0.8
   RETURN n, r, m
   LIMIT 100
   """
   ```

3. **Use Parameters:**

   ```python
   # Parameterized queries for security and performance
   await session.call_tool(
       "execute_cypher_query",
       {
           "query": "MATCH (n:Person {name: $name}) RETURN n",
           "parameters": {"name": user_input}
       }
   )
   ```

### Memory Management

1. **Contextual Memories:**

   ```python
   # Associate memories with context
   memory = {
       "name": "Project Alpha Discussion",
       "memory_type": "meeting",
       "context": "Weekly standup - 2024-01-15",
       "observations": [
           "Team prefers React for frontend",
           "API performance needs optimization",
           "Deadline moved to March 1st"
       ]
   }
   ```

2. **Memory Updates:**

   ```python
   # Update existing memories
   await session.call_tool(
       "update_node_properties",
       {
           "node_id": "memory_meeting_123",
           "properties": {
               "observations": updated_observations,
               "last_updated": datetime.now().isoformat(),
               "access_count": access_count + 1
           }
       }
   )
   ```

## Integration Patterns

### Knowledge Graph RAG

```python
async def graph_rag_search(query, depth=2):
    """Enhanced RAG using knowledge graph relationships."""
    
    # 1. Find relevant nodes
    search_result = await session.call_tool(
        "search_graph",
        {
            "query": query,
            "max_depth": depth,
            "use_embeddings": True,
            "limit": 10
        }
    )
    
    # 2. Expand with relationships
    expanded_context = []
    for node in search_result.content['nodes']:
        # Get connected nodes for richer context
        connections = await session.call_tool(
            "execute_cypher_query",
            {
                "query": """
                MATCH (n)-[r]-(connected)
                WHERE n.id = $node_id
                RETURN connected.name, type(r), r.weight
                ORDER BY r.weight DESC
                LIMIT 5
                """,
                "parameters": {"node_id": node['id']}
            }
        )
        
        context = {
            "primary": node,
            "connections": connections.content['records']
        }
        expanded_context.append(context)
    
    return expanded_context
```

### Incremental Knowledge Building

```python
async def build_knowledge_incrementally(documents):
    """Build knowledge graph incrementally from documents."""
    
    for doc in documents:
        # 1. Extract knowledge
        extraction = await session.call_tool(
            "extract_knowledge_from_text",
            {
                "text": doc['content'],
                "source_type": doc['type'],
                "document_id": doc['id']
            }
        )
        
        # 2. Merge with existing knowledge
        for node in extraction.content['extracted_nodes']:
            # Check for existing similar nodes
            existing = await session.call_tool(
                "search_graph",
                {
                    "query": node['name'],
                    "node_types": [node['node_type']],
                    "limit": 5,
                    "use_embeddings": True,
                    "embedding_similarity_threshold": 0.9
                }
            )
            
            if existing.content['nodes']:
                # Update existing node
                await session.call_tool(
                    "update_node_properties",
                    {
                        "node_id": existing.content['nodes'][0]['id'],
                        "properties": {
                            "sources": [doc['id']],
                            "confidence_score": max(
                                node['confidence_score'],
                                existing.content['nodes'][0]['confidence_score']
                            )
                        },
                        "merge_mode": True
                    }
                )
            else:
                # Create new node
                await session.call_tool("create_graph_node", node)
```

### Memory-Driven Conversations

```python
async def memory_enhanced_response(user_input, user_id):
    """Generate responses enhanced by memory graph."""
    
    # 1. Find relevant memories
    memories = await session.call_tool(
        "search_graph",
        {
            "query": user_input,
            "node_types": ["Memory"],
            "max_depth": 2,
            "limit": 5
        }
    )
    
    # 2. Get user-specific context
    user_memories = await session.call_tool(
        "execute_cypher_query",
        {
            "query": """
            MATCH (u:Person {id: $user_id})-[r:REMEMBERS]-(m:Memory)
            RETURN m.name, m.observations, m.context
            ORDER BY r.last_accessed DESC
            LIMIT 10
            """,
            "parameters": {"user_id": user_id}
        }
    )
    
    # 3. Create new memory for this interaction
    await session.call_tool(
        "create_memory_node",
        {
            "name": f"Conversation {datetime.now().strftime('%Y%m%d_%H%M')}",
            "memory_type": "conversation",
            "observations": [user_input],
            "context": f"User {user_id} interaction"
        }
    )
    
    return {
        "relevant_memories": memories.content['nodes'],
        "user_context": user_memories.content['records']
    }
```

---

*For more information, see the [API Overview](../README.md) or explore [Vector Tools](./vector-tools.md) and [Web Tools](./web-tools.md).*
