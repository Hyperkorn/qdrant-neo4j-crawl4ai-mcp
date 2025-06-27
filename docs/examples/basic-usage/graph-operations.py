"""
Example: Graph Operations with Neo4j
Description: Demonstrates knowledge graph construction and querying using Neo4j
Services: Graph (Neo4j)
Complexity: Beginner

This example shows how to:
1. Create nodes and relationships
2. Query the knowledge graph
3. Extract knowledge from text
4. Analyze graph structure
"""

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPGraphClient:
    """Simple MCP client focused on graph operations."""
    
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

async def demonstrate_node_creation():
    """Demonstrate creating nodes in the knowledge graph."""
    print("🔄 Demonstrating Node Creation")
    print("=" * 50)
    
    client = MCPGraphClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Sample entities to create
    entities = [
        {
            "name": "Machine Learning",
            "node_type": "Concept",
            "description": "A subset of artificial intelligence that enables computers to learn from data",
            "properties": {
                "field": "Computer Science",
                "complexity": "High",
                "popularity": "Very High"
            },
            "tags": ["AI", "Technology", "Data Science"],
            "abstraction_level": 3,
            "confidence_score": 0.95
        },
        {
            "name": "Geoffrey Hinton", 
            "node_type": "Person",
            "description": "British-Canadian cognitive psychologist and computer scientist, known as the 'Godfather of AI'",
            "properties": {
                "profession": "Computer Scientist",
                "nationality": "British-Canadian",
                "birth_year": 1947,
                "notable_for": "Deep Learning"
            },
            "tags": ["AI Pioneer", "Researcher", "Academic"],
            "abstraction_level": 1,
            "confidence_score": 0.99
        },
        {
            "name": "Neural Networks",
            "node_type": "Concept", 
            "description": "Computing systems inspired by biological neural networks",
            "properties": {
                "type": "Algorithm",
                "inspired_by": "Brain Structure",
                "applications": ["Image Recognition", "NLP", "Game Playing"]
            },
            "tags": ["AI", "Deep Learning", "Algorithm"],
            "abstraction_level": 4,
            "confidence_score": 0.92
        },
        {
            "name": "Turing Award",
            "node_type": "Award",
            "description": "Annual prize given by the Association for Computing Machinery for contributions to computer science",
            "properties": {
                "first_awarded": 1966,
                "organization": "ACM",
                "prestige": "Highest"
            },
            "tags": ["Award", "Computer Science", "Recognition"],
            "abstraction_level": 2,
            "confidence_score": 1.0
        }
    ]
    
    # Create each entity
    created_nodes = []
    for i, entity in enumerate(entities, 1):
        print(f"\n📍 Creating node {i}: {entity['name']}")
        
        result = await client.call_tool("create_graph_node", entity)
        
        if result.get("success"):
            node_id = result["node_id"]
            created_nodes.append({"id": node_id, "name": entity["name"], "type": entity["node_type"]})
            print(f"✅ Created node: {entity['name']} (ID: {node_id})")
            print(f"   Type: {result['node_type']}")
            print(f"   Created: {result['created_at']}")
        else:
            print(f"❌ Failed to create node: {result.get('error')}")
    
    print(f"\n✅ Successfully created {len(created_nodes)} nodes")
    return created_nodes

async def demonstrate_relationship_creation(nodes: list[dict]):
    """Demonstrate creating relationships between nodes."""
    print("\n🔗 Demonstrating Relationship Creation")
    print("=" * 50)
    
    client = MCPGraphClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    if len(nodes) < 4:
        print("⚠️  Not enough nodes to create relationships")
        return []
    
    # Find nodes by name for relationship creation
    node_map = {node["name"]: node for node in nodes}
    
    # Define relationships to create
    relationships = [
        {
            "source": "Geoffrey Hinton",
            "target": "Machine Learning", 
            "relationship_type": "CONTRIBUTED_TO",
            "description": "Pioneer in machine learning research",
            "properties": {"role": "Pioneer", "years_active": "1970-present"},
            "confidence_score": 0.98
        },
        {
            "source": "Geoffrey Hinton",
            "target": "Neural Networks",
            "relationship_type": "DEVELOPED", 
            "description": "Key contributor to neural network development",
            "properties": {"contribution": "Backpropagation", "impact": "Revolutionary"},
            "confidence_score": 0.95
        },
        {
            "source": "Geoffrey Hinton",
            "target": "Turing Award",
            "relationship_type": "RECEIVED",
            "description": "Awarded Turing Award in 2018",
            "properties": {"year": 2018, "shared_with": ["Yann LeCun", "Yoshua Bengio"]},
            "confidence_score": 1.0
        },
        {
            "source": "Neural Networks", 
            "target": "Machine Learning",
            "relationship_type": "PART_OF",
            "description": "Neural networks are a key component of machine learning",
            "properties": {"relationship_nature": "subset", "importance": "High"},
            "confidence_score": 0.96
        }
    ]
    
    # Create each relationship
    created_relationships = []
    for i, rel in enumerate(relationships, 1):
        source_node = node_map.get(rel["source"])
        target_node = node_map.get(rel["target"])
        
        if not source_node or not target_node:
            print(f"⚠️  Skipping relationship {i}: Missing nodes")
            continue
            
        print(f"\n🔗 Creating relationship {i}: {rel['source']} → {rel['target']}")
        
        result = await client.call_tool("create_graph_relationship", {
            "source_id": source_node["id"],
            "target_id": target_node["id"],
            "relationship_type": rel["relationship_type"],
            "description": rel["description"],
            "properties": rel["properties"],
            "confidence_score": rel["confidence_score"]
        })
        
        if result.get("success"):
            rel_id = result["relationship_id"] 
            created_relationships.append(rel_id)
            print(f"✅ Created relationship: {rel['relationship_type']}")
            print(f"   ID: {rel_id}")
            print(f"   Strength: {result['strength']}")
        else:
            print(f"❌ Failed to create relationship: {result.get('error')}")
    
    print(f"\n✅ Successfully created {len(created_relationships)} relationships")
    return created_relationships

async def demonstrate_graph_search():
    """Demonstrate searching the knowledge graph."""
    print("\n🔍 Demonstrating Graph Search")
    print("=" * 50)
    
    client = MCPGraphClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Search queries
    search_queries = [
        {
            "query": "machine learning pioneers",
            "search_type": "semantic",
            "limit": 5
        },
        {
            "query": "artificial intelligence researchers",
            "search_type": "semantic", 
            "limit": 3
        },
        {
            "query": "awards in computer science",
            "search_type": "semantic",
            "limit": 5
        }
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n🔎 Search {i}: '{query['query']}'")
        
        result = await client.call_tool("search_knowledge_graph", {
            "query": query["query"],
            "search_type": query["search_type"],
            "limit": query["limit"],
            "include_relationships": True
        })
        
        if result.get("success"):
            results = result["results"]
            print(f"   Found {len(results)} results (search time: {result.get('search_time_ms', 0):.1f}ms)")
            
            for j, res in enumerate(results, 1):
                print(f"   {j}. {res['name']} ({res['node_type']})")
                print(f"      Score: {res['score']:.3f}")
                print(f"      Description: {res.get('description', 'N/A')[:80]}...")
                if res.get('relationships'):
                    print(f"      Connections: {len(res['relationships'])}")
        else:
            print(f"   ❌ Search failed: {result.get('error')}")

async def demonstrate_cypher_queries():
    """Demonstrate custom Cypher queries."""
    print("\n📊 Demonstrating Cypher Queries")
    print("=" * 50)
    
    client = MCPGraphClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Sample Cypher queries
    cypher_queries = [
        {
            "description": "Find all people and their relationships",
            "query": """
                MATCH (p:Person)-[r]->(n)
                RETURN p.name as person, type(r) as relationship, n.name as target
                LIMIT 10
            """
        },
        {
            "description": "Find award recipients",
            "query": """
                MATCH (p:Person)-[:RECEIVED]->(a:Award)
                RETURN p.name as recipient, a.name as award
            """
        },
        {
            "description": "Find concepts and their connections",
            "query": """
                MATCH (c1:Concept)-[r]-(c2:Concept)
                RETURN c1.name as concept1, type(r) as relationship, c2.name as concept2
                LIMIT 5
            """
        },
        {
            "description": "Get graph statistics",
            "query": """
                MATCH (n)
                RETURN labels(n)[0] as node_type, count(n) as count
                ORDER BY count DESC
            """
        }
    ]
    
    for i, query_info in enumerate(cypher_queries, 1):
        print(f"\n📋 Query {i}: {query_info['description']}")
        print(f"   Cypher: {query_info['query'].strip()}")
        
        result = await client.call_tool("execute_cypher_query", {
            "query": query_info["query"],
            "parameters": {}
        })
        
        if result.get("success"):
            records = result["records"]
            print(f"   ✅ Returned {len(records)} records:")
            
            for j, record in enumerate(records[:5], 1):  # Show first 5 results
                print(f"      {j}. {record}")
                
            if len(records) > 5:
                print(f"      ... and {len(records) - 5} more")
        else:
            print(f"   ❌ Query failed: {result.get('error')}")

async def demonstrate_knowledge_extraction():
    """Demonstrate extracting knowledge from text."""
    print("\n🧠 Demonstrating Knowledge Extraction")
    print("=" * 50)
    
    client = MCPGraphClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Sample text for knowledge extraction
    sample_text = """
    Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. 
    He created the first web browser, called WorldWideWeb, and the first web server.
    The first website was launched in 1991. Berners-Lee founded the World Wide Web 
    Consortium (W3C) in 1994 to develop web standards. He received a knighthood 
    from Queen Elizabeth II in 2004 for his services to the development of the Internet.
    """
    
    print(f"📄 Extracting knowledge from text:")
    print(f"   {sample_text[:100]}...")
    
    result = await client.call_tool("extract_knowledge_from_text", {
        "text": sample_text,
        "domain": "technology",
        "extract_entities": True,
        "extract_relationships": True,
        "confidence_threshold": 0.7
    })
    
    if result.get("success"):
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        
        print(f"\n✅ Extracted {len(entities)} entities and {len(relationships)} relationships:")
        
        print(f"\n   📍 Entities:")
        for entity in entities[:5]:  # Show first 5
            print(f"      • {entity['name']} ({entity['type']})")
            print(f"        Confidence: {entity['confidence']:.2f}")
            
        print(f"\n   🔗 Relationships:")
        for rel in relationships[:5]:  # Show first 5
            print(f"      • {rel['source']} → {rel['relationship']} → {rel['target']}")
            print(f"        Confidence: {rel['confidence']:.2f}")
    else:
        print(f"   ❌ Knowledge extraction failed: {result.get('error')}")

async def demonstrate_graph_analysis():
    """Demonstrate graph analysis and statistics.""" 
    print("\n📊 Demonstrating Graph Analysis")
    print("=" * 50)
    
    client = MCPGraphClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    result = await client.call_tool("analyze_graph_structure", {
        "include_centrality": True,
        "include_communities": True,
        "include_paths": True
    })
    
    if result.get("success"):
        analysis = result["analysis"]
        
        print("📈 Graph Structure Analysis:")
        print(f"   Total nodes: {analysis.get('total_nodes', 0)}")
        print(f"   Total relationships: {analysis.get('total_relationships', 0)}")
        print(f"   Graph density: {analysis.get('density', 0):.3f}")
        print(f"   Connected components: {analysis.get('connected_components', 0)}")
        
        # Node centrality
        if "centrality" in analysis:
            print(f"\n   🎯 Most central nodes:")
            for node in analysis["centrality"][:3]:
                print(f"      • {node['name']}: {node['centrality']:.3f}")
        
        # Communities
        if "communities" in analysis:
            print(f"\n   👥 Communities found: {len(analysis['communities'])}")
            for i, community in enumerate(analysis["communities"][:2], 1):
                print(f"      Community {i}: {len(community)} nodes")
                
        # Average path length
        if "average_path_length" in analysis:
            print(f"\n   🛤️  Average path length: {analysis['average_path_length']:.2f}")
            
    else:
        print(f"   ❌ Analysis failed: {result.get('error')}")

async def cleanup_demo_data():
    """Clean up demo graph data (optional)."""
    print("\n🧹 Cleanup (Optional)")
    print("=" * 50)
    
    response = input("Delete demo graph data? (y/N): ")
    if response.lower() == 'y':
        client = MCPGraphClient(
            base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        )
        
        # Delete all demo nodes (this will cascade to relationships)
        result = await client.call_tool("execute_cypher_query", {
            "query": """
                MATCH (n)
                WHERE n.source = 'demo' OR n.tags CONTAINS 'demo'
                DETACH DELETE n
                RETURN count(n) as deleted_count
            """,
            "parameters": {}
        })
        
        if result.get("success"):
            deleted_count = result["records"][0].get("deleted_count", 0)
            print(f"   ✅ Deleted {deleted_count} demo nodes and their relationships")
        else:
            print(f"   ❌ Cleanup failed: {result.get('error')}")
    else:
        print("   ⏭️  Skipping cleanup")

async def main():
    """Run all graph operation demonstrations."""
    print("🚀 Graph Operations Demo")
    print("=" * 50)
    print("This demo shows the core knowledge graph capabilities:")
    print("• Node creation and management")
    print("• Relationship building")
    print("• Graph search and querying")
    print("• Knowledge extraction from text")
    print("• Graph analysis and statistics")
    
    try:
        # Run demonstrations in sequence
        nodes = await demonstrate_node_creation()
        await demonstrate_relationship_creation(nodes)
        await demonstrate_graph_search()
        await demonstrate_cypher_queries()
        await demonstrate_knowledge_extraction()
        await demonstrate_graph_analysis()
        await cleanup_demo_data()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("• Try vector-operations.py for semantic search")
        print("• Check web-intelligence.py for content extraction")
        print("• Combine services in advanced-workflows/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("• Ensure MCP server is running")
        print("• Check environment variables")
        print("• Verify Neo4j connection")

if __name__ == "__main__":
    asyncio.run(main())