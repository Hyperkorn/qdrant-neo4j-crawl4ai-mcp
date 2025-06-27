"""
Example: Vector Operations with Qdrant
Description: Demonstrates semantic document storage and search using vector embeddings
Services: Vector (Qdrant)
Complexity: Beginner

This example shows how to:
1. Store documents with vector embeddings
2. Perform semantic similarity search
3. Manage vector collections
4. Generate text embeddings
"""

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPVectorClient:
    """Simple MCP client focused on vector operations."""
    
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via HTTP API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/tools/{tool_name}",
                json=params,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

async def demonstrate_vector_storage():
    """Demonstrate storing documents with vector embeddings."""
    print("🔄 Demonstrating Vector Document Storage")
    print("=" * 50)
    
    client = MCPVectorClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Sample documents to store
    documents = [
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "content_type": "definition",
            "source": "AI textbook",
            "tags": ["ai", "machine-learning", "technology"],
            "metadata": {"category": "computer_science", "difficulty": "intermediate"}
        },
        {
            "content": "Natural language processing (NLP) allows computers to understand, interpret, and generate human language in a meaningful way.",
            "content_type": "definition", 
            "source": "NLP guide",
            "tags": ["nlp", "ai", "language"],
            "metadata": {"category": "computer_science", "difficulty": "intermediate"}
        },
        {
            "content": "Vector databases store data as high-dimensional vectors, enabling semantic search and similarity matching for AI applications.",
            "content_type": "definition",
            "source": "Database guide", 
            "tags": ["vectors", "database", "search"],
            "metadata": {"category": "data_engineering", "difficulty": "advanced"}
        }
    ]
    
    # Store each document
    stored_ids = []
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Storing document {i}: {doc['content'][:50]}...")
        
        result = await client.call_tool("store_vector_document", {
            "content": doc["content"],
            "collection_name": "demo_collection",
            "content_type": doc["content_type"],
            "source": doc["source"],
            "tags": doc["tags"],
            "metadata": doc["metadata"]
        })
        
        if result.get("status") == "success":
            doc_id = result["id"]
            stored_ids.append(doc_id)
            print(f"✅ Stored with ID: {doc_id}")
            print(f"   Vector dimensions: {result['vector_dimensions']}")
            print(f"   Embedding time: {result['embedding_time_ms']:.1f}ms")
            print(f"   Storage time: {result['storage_time_ms']:.1f}ms")
        else:
            print(f"❌ Failed to store: {result.get('error')}")
    
    print(f"\n✅ Successfully stored {len(stored_ids)} documents")
    return stored_ids

async def demonstrate_semantic_search():
    """Demonstrate semantic similarity search."""
    print("\n🔍 Demonstrating Semantic Search")
    print("=" * 50)
    
    client = MCPVectorClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Search queries with different similarity levels
    queries = [
        "What is artificial intelligence?",
        "How do computers understand human speech?", 
        "What are vector databases used for?",
        "Programming languages for web development"  # Should have lower relevance
    ]
    
    for query in queries:
        print(f"\n🔎 Query: '{query}'")
        
        result = await client.call_tool("semantic_vector_search", {
            "query": query,
            "collection_name": "demo_collection",
            "limit": 3,
            "score_threshold": 0.0,
            "include_content": True
        })
        
        if result.get("status") == "success":
            results = result["results"]
            print(f"   Found {len(results)} results (search time: {result['search_time_ms']:.1f}ms)")
            
            for i, res in enumerate(results, 1):
                print(f"   {i}. Score: {res['score']:.3f}")
                print(f"      Content: {res['content'][:80]}...")
                print(f"      Tags: {res.get('tags', [])}")
                print(f"      Source: {res.get('source', 'Unknown')}")
        else:
            print(f"   ❌ Search failed: {result.get('error')}")

async def demonstrate_collection_management():
    """Demonstrate vector collection operations."""
    print("\n📊 Demonstrating Collection Management")
    print("=" * 50)
    
    client = MCPVectorClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # List existing collections
    print("📋 Listing current collections:")
    result = await client.call_tool("list_vector_collections", {})
    
    if result.get("status") == "success":
        collections = result["collections"]
        print(f"   Found {len(collections)} collections:")
        
        for coll in collections:
            print(f"   • {coll['name']}")
            print(f"     - Vectors: {coll['points_count']:,}")
            print(f"     - Size: {coll['vector_size']} dimensions")
            print(f"     - Distance: {coll['distance_metric']}")
            print(f"     - Disk usage: {coll['disk_usage_mb']:.1f} MB")
            print(f"     - Status: {coll['status']}")
    else:
        print(f"   ❌ Failed to list collections: {result.get('error')}")
    
    # Create a new test collection
    print(f"\n🆕 Creating a new test collection:")
    result = await client.call_tool("create_vector_collection", {
        "name": "test_collection",
        "vector_size": 384,  # Common size for sentence transformers
        "distance_metric": "Cosine",
        "description": "Test collection for demonstration"
    })
    
    if result.get("status") == "success":
        print(f"   ✅ Created collection: {result['name']}")
        print(f"      Vector size: {result['vector_size']}")
        print(f"      Distance metric: {result['distance_metric']}")
    else:
        print(f"   ❌ Failed to create collection: {result.get('error')}")

async def demonstrate_embeddings():
    """Demonstrate text embedding generation."""
    print("\n🧮 Demonstrating Text Embeddings")
    print("=" * 50)
    
    client = MCPVectorClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Generate embeddings for sample texts
    texts = [
        "Hello, world!",
        "Machine learning is fascinating",
        "I love programming in Python"
    ]
    
    print(f"🔄 Generating embeddings for {len(texts)} texts:")
    for text in texts:
        print(f"   • {text}")
    
    result = await client.call_tool("generate_text_embeddings", {
        "texts": texts,
        "normalize": True
    })
    
    if result.get("status") == "success":
        embeddings = result["embeddings"]
        print(f"\n✅ Generated {len(embeddings)} embeddings:")
        print(f"   Model: {result['model']}")
        print(f"   Dimensions: {result['dimensions']}")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
        
        # Show first few values of each embedding
        for i, embedding in enumerate(embeddings):
            print(f"   Embedding {i+1}: [{embedding[0]:.3f}, {embedding[1]:.3f}, {embedding[2]:.3f}, ...]")
    else:
        print(f"   ❌ Failed to generate embeddings: {result.get('error')}")

async def demonstrate_service_stats():
    """Demonstrate vector service statistics."""
    print("\n📈 Demonstrating Service Statistics")
    print("=" * 50)
    
    client = MCPVectorClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    result = await client.call_tool("get_vector_service_stats", {})
    
    if result.get("status") == "success":
        stats = result["stats"]
        health = result["health"]
        
        print("📊 Service Statistics:")
        print(f"   Total collections: {stats['total_collections']}")
        print(f"   Total vectors: {stats['total_vectors']:,}")
        print(f"   Disk usage: {stats['total_disk_usage_mb']:.1f} MB")
        print(f"   RAM usage: {stats['total_ram_usage_mb']:.1f} MB")
        print(f"   Avg search time: {stats['average_search_time_ms']:.1f}ms")
        print(f"   Embeddings generated: {stats['embeddings_generated']:,}")
        
        print(f"\n🏥 Service Health:")
        print(f"   Status: {health['status']}")
        print(f"   Response time: {health['response_time_ms']:.1f}ms")
        print(f"   Service: {health['service']}")
    else:
        print(f"   ❌ Failed to get stats: {result.get('error')}")

async def cleanup_demo_data():
    """Clean up demo collections (optional)."""
    print("\n🧹 Cleanup (Optional)")
    print("=" * 50)
    
    response = input("Delete test collection? (y/N): ")
    if response.lower() == 'y':
        client = MCPVectorClient(
            base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        )
        
        result = await client.call_tool("delete_vector_collection", {
            "collection_name": "test_collection",
            "confirm": True
        })
        
        if result.get("status") == "success":
            print("   ✅ Test collection deleted")
        else:
            print(f"   ❌ Failed to delete: {result.get('error')}")
    else:
        print("   ⏭️  Skipping cleanup")

async def main():
    """Run all vector operation demonstrations."""
    print("🚀 Vector Operations Demo")
    print("=" * 50)
    print("This demo shows the core vector database capabilities:")
    print("• Document storage with embeddings")
    print("• Semantic similarity search")
    print("• Collection management")
    print("• Embedding generation")
    print("• Service monitoring")
    
    try:
        # Run demonstrations in sequence
        await demonstrate_vector_storage()
        await demonstrate_semantic_search()
        await demonstrate_collection_management()
        await demonstrate_embeddings()
        await demonstrate_service_stats()
        await cleanup_demo_data()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("• Try graph-operations.py for knowledge graphs")
        print("• Check web-intelligence.py for content extraction")
        print("• Build a complete app with document-qa-system/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("• Ensure MCP server is running")
        print("• Check environment variables")
        print("• Verify Qdrant connection")

if __name__ == "__main__":
    asyncio.run(main())