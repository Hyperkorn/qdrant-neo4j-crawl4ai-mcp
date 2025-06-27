"""
Advanced Workflow: Hybrid Search System
Description: Sophisticated search combining vector similarity, graph relationships, and web intelligence
Services: Vector (Qdrant) + Graph (Neo4j) + Web (Crawl4AI)
Complexity: Advanced

This workflow demonstrates:
1. Multi-modal query understanding and expansion
2. Parallel search across vector, graph, and web sources
3. Intelligent result fusion and ranking
4. Context-aware answer generation
5. Real-time learning from user feedback
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import httpx
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"           # Specific facts and definitions
    ANALYTICAL = "analytical"     # Analysis and relationships
    EXPLORATORY = "exploratory"   # Broad topic exploration
    COMPARATIVE = "comparative"   # Comparing entities or concepts
    TEMPORAL = "temporal"         # Time-based queries

class SourceType(Enum):
    """Types of information sources."""
    VECTOR = "vector"             # Document embeddings
    GRAPH = "graph"               # Knowledge graph
    WEB = "web"                   # Real-time web content
    CACHED = "cached"             # Previously computed results

@dataclass
class SearchResult:
    """Unified search result structure."""
    content: str
    score: float
    source_type: SourceType
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_factors: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QueryContext:
    """Context information for query processing."""
    query: str
    query_type: QueryType
    user_intent: str
    expanded_queries: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    search_depth: int = 1
    max_results: int = 20
    min_confidence: float = 0.6

@dataclass
class HybridSearchResponse:
    """Response from hybrid search system."""
    query: str
    answer: str
    confidence: float
    sources: List[SearchResult]
    processing_time_ms: float
    search_stats: Dict[str, Any]
    suggestions: List[str] = field(default_factory=list)

class MCPClient:
    """MCP client with connection pooling and error handling."""
    
    def __init__(self, base_url: str, token: str = None, max_connections: int = 10):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.semaphore = asyncio.Semaphore(max_connections)
        self.circuit_breaker = CircuitBreaker()
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with circuit breaker and retries."""
        async with self.semaphore:
            return await self.circuit_breaker.call(self._make_request, tool_name, params)
    
    async def _make_request(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to MCP server."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/tools/{tool_name}",
                json=params,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

class CircuitBreaker:
    """Circuit breaker for handling service failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if datetime.utcnow().timestamp() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow().timestamp()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

class QueryAnalyzer:
    """Analyzes queries to understand intent and expand search terms."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.query_patterns = self._load_query_patterns()
    
    def _load_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Load query patterns for classification."""
        return {
            QueryType.FACTUAL: [
                "what is", "define", "definition of", "explain", "describe"
            ],
            QueryType.ANALYTICAL: [
                "analyze", "compare", "relationship between", "how does", "why"
            ],
            QueryType.EXPLORATORY: [
                "overview of", "about", "tell me about", "information on"
            ],
            QueryType.COMPARATIVE: [
                "difference between", "compare", "versus", "vs", "better than"
            ],
            QueryType.TEMPORAL: [
                "when", "history of", "timeline", "before", "after", "recent"
            ]
        }
    
    async def analyze_query(self, query: str) -> QueryContext:
        """Analyze query to extract intent and context."""
        query_lower = query.lower()
        
        # Classify query type
        query_type = self._classify_query_type(query_lower)
        
        # Extract user intent
        user_intent = await self._extract_intent(query)
        
        # Generate expanded queries
        expanded_queries = await self._expand_query(query, query_type)
        
        # Determine search parameters
        search_depth = self._determine_search_depth(query_type)
        max_results = self._determine_max_results(query_type)
        
        return QueryContext(
            query=query,
            query_type=query_type,
            user_intent=user_intent,
            expanded_queries=expanded_queries,
            search_depth=search_depth,
            max_results=max_results
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query based on patterns."""
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query)
            scores[query_type] = score
        
        # Return type with highest score, default to exploratory
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return QueryType.EXPLORATORY
    
    async def _extract_intent(self, query: str) -> str:
        """Extract user intent using LLM or rule-based approach."""
        # In a real implementation, you'd use an LLM for intent extraction
        # For this example, we'll use a simplified approach
        
        intent_keywords = {
            "learn": ["understand", "learn", "know about", "explain"],
            "find": ["find", "search", "locate", "discover"],
            "compare": ["compare", "difference", "versus", "better"],
            "analyze": ["analyze", "examine", "study", "investigate"]
        }
        
        query_lower = query.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "explore"
    
    async def _expand_query(self, query: str, query_type: QueryType) -> List[str]:
        """Generate expanded query variations."""
        expanded = []
        
        # Add synonyms and related terms
        if query_type == QueryType.FACTUAL:
            expanded.extend([
                f"definition of {query}",
                f"what is {query}",
                f"{query} explained"
            ])
        elif query_type == QueryType.ANALYTICAL:
            expanded.extend([
                f"analysis of {query}",
                f"how {query} works",
                f"{query} relationships"
            ])
        
        # Add domain-specific expansions
        try:
            # Use the graph to find related entities
            graph_result = await self.mcp_client.call_tool("search_knowledge_graph", {
                "query": query,
                "limit": 3,
                "search_type": "semantic"
            })
            
            if graph_result.get("success"):
                for result in graph_result.get("results", []):
                    related_name = result.get("name", "")
                    if related_name and related_name not in expanded:
                        expanded.append(f"{query} {related_name}")
        
        except Exception as e:
            logger.warning(f"Failed to expand query using graph: {e}")
        
        return expanded[:5]  # Limit to 5 expansions
    
    def _determine_search_depth(self, query_type: QueryType) -> int:
        """Determine how deep to search based on query type."""
        depth_mapping = {
            QueryType.FACTUAL: 1,
            QueryType.ANALYTICAL: 2,
            QueryType.EXPLORATORY: 3,
            QueryType.COMPARATIVE: 2,
            QueryType.TEMPORAL: 2
        }
        return depth_mapping.get(query_type, 1)
    
    def _determine_max_results(self, query_type: QueryType) -> int:
        """Determine maximum results based on query type."""
        result_mapping = {
            QueryType.FACTUAL: 10,
            QueryType.ANALYTICAL: 15,
            QueryType.EXPLORATORY: 25,
            QueryType.COMPARATIVE: 20,
            QueryType.TEMPORAL: 15
        }
        return result_mapping.get(query_type, 15)

class VectorSearchEngine:
    """Vector similarity search using Qdrant."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
    
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """Perform vector similarity search."""
        results = []
        
        # Search with original query
        vector_results = await self._search_vectors(
            context.query, 
            limit=context.max_results // 3
        )
        results.extend(vector_results)
        
        # Search with expanded queries if depth > 1
        if context.search_depth > 1 and context.expanded_queries:
            for expanded_query in context.expanded_queries[:2]:
                expanded_results = await self._search_vectors(
                    expanded_query,
                    limit=min(5, context.max_results // 6)
                )
                results.extend(expanded_results)
        
        return results
    
    async def _search_vectors(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search vector database."""
        try:
            result = await self.mcp_client.call_tool("semantic_vector_search", {
                "query": query,
                "collection_name": "documents",
                "limit": limit,
                "score_threshold": 0.7,
                "include_content": True
            })
            
            if result.get("status") != "success":
                logger.warning(f"Vector search failed: {result.get('error')}")
                return []
            
            search_results = []
            for item in result.get("results", []):
                search_result = SearchResult(
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    source_type=SourceType.VECTOR,
                    source_id=item.get("id", ""),
                    metadata=item.get("metadata", {}),
                    relevance_factors={
                        "semantic_similarity": item.get("score", 0.0),
                        "content_length": len(item.get("content", "")),
                        "source_quality": item.get("metadata", {}).get("quality_score", 0.5)
                    }
                )
                search_results.append(search_result)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

class GraphSearchEngine:
    """Knowledge graph search using Neo4j."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
    
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """Perform graph knowledge search."""
        results = []
        
        # Semantic search in graph
        graph_results = await self._search_graph_semantic(
            context.query,
            limit=context.max_results // 3
        )
        results.extend(graph_results)
        
        # Relationship-based search for analytical queries
        if context.query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            relationship_results = await self._search_graph_relationships(
                context.query,
                limit=min(10, context.max_results // 4)
            )
            results.extend(relationship_results)
        
        return results
    
    async def _search_graph_semantic(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Semantic search in knowledge graph."""
        try:
            result = await self.mcp_client.call_tool("search_knowledge_graph", {
                "query": query,
                "search_type": "semantic",
                "limit": limit,
                "include_relationships": True
            })
            
            if not result.get("success"):
                logger.warning(f"Graph search failed: {result.get('error')}")
                return []
            
            search_results = []
            for item in result.get("results", []):
                content = item.get("description", item.get("name", ""))
                if item.get("relationships"):
                    relationships_text = "; ".join([
                        f"{rel.get('type', 'related to')} {rel.get('target_name', '')}"
                        for rel in item.get("relationships", [])[:3]
                    ])
                    content += f" Related: {relationships_text}"
                
                search_result = SearchResult(
                    content=content,
                    score=item.get("score", 0.0),
                    source_type=SourceType.GRAPH,
                    source_id=item.get("id", ""),
                    metadata=item,
                    relevance_factors={
                        "semantic_score": item.get("score", 0.0),
                        "relationship_count": len(item.get("relationships", [])),
                        "node_importance": item.get("importance_score", 0.5)
                    }
                )
                search_results.append(search_result)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []
    
    async def _search_graph_relationships(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for relationship patterns in the graph."""
        try:
            # Use Cypher to find relationship patterns
            cypher_query = """
                MATCH (n1)-[r]-(n2)
                WHERE n1.name CONTAINS $query OR n2.name CONTAINS $query
                RETURN n1.name as source, type(r) as relationship, n2.name as target,
                       n1.description as source_desc, n2.description as target_desc
                LIMIT $limit
            """
            
            result = await self.mcp_client.call_tool("execute_cypher_query", {
                "query": cypher_query,
                "parameters": {"query": query, "limit": limit}
            })
            
            if not result.get("success"):
                return []
            
            search_results = []
            for record in result.get("records", []):
                content = (
                    f"{record.get('source', '')} {record.get('relationship', 'relates to')} "
                    f"{record.get('target', '')}. "
                    f"{record.get('source_desc', '')} {record.get('target_desc', '')}"
                )
                
                search_result = SearchResult(
                    content=content,
                    score=0.8,  # Default score for relationship matches
                    source_type=SourceType.GRAPH,
                    source_id=f"rel_{record.get('source', '')}_{record.get('target', '')}",
                    metadata=record,
                    relevance_factors={
                        "relationship_relevance": 0.8,
                        "entity_match": 1.0 if query.lower() in content.lower() else 0.5
                    }
                )
                search_results.append(search_result)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Graph relationship search error: {e}")
            return []

class WebSearchEngine:
    """Real-time web content search using Crawl4AI."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.search_engines = [
            "https://www.google.com/search?q=",
            "https://duckduckgo.com/?q=",
            "https://www.bing.com/search?q="
        ]
    
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """Perform web content search."""
        if context.query_type == QueryType.TEMPORAL:
            # For temporal queries, prioritize recent content
            return await self._search_recent_content(context)
        else:
            # For other queries, search authoritative sources
            return await self._search_authoritative_sources(context)
    
    async def _search_recent_content(self, context: QueryContext) -> List[SearchResult]:
        """Search for recent web content."""
        results = []
        
        # Search news and recent articles
        search_urls = [
            f"https://news.google.com/search?q={context.query}",
            f"https://www.reddit.com/search/?q={context.query}"
        ]
        
        for url in search_urls[:2]:  # Limit to 2 sources for performance
            try:
                web_results = await self._crawl_and_extract(url, context.query)
                results.extend(web_results[:3])  # Limit results per source
            except Exception as e:
                logger.warning(f"Web search failed for {url}: {e}")
        
        return results
    
    async def _search_authoritative_sources(self, context: QueryContext) -> List[SearchResult]:
        """Search authoritative web sources."""
        results = []
        
        # Target specific authoritative domains based on query type
        if context.query_type == QueryType.FACTUAL:
            target_domains = ["wikipedia.org", "britannica.com", "edu"]
        elif context.query_type == QueryType.ANALYTICAL:
            target_domains = ["arxiv.org", "scholar.google.com", "researchgate.net"]
        else:
            target_domains = ["wikipedia.org", "edu"]
        
        # Search each domain
        for domain in target_domains[:2]:
            try:
                search_url = f"site:{domain} {context.query}"
                web_results = await self._web_search(search_url)
                results.extend(web_results[:2])
            except Exception as e:
                logger.warning(f"Web search failed for domain {domain}: {e}")
        
        return results
    
    async def _web_search(self, query: str) -> List[SearchResult]:
        """Perform web search and content extraction."""
        try:
            # In a real implementation, you'd use a proper search API
            # For this example, we'll simulate web search results
            
            result = await self.mcp_client.call_tool("crawl_website", {
                "url": f"https://www.google.com/search?q={query}",
                "max_depth": 1,
                "max_pages": 3,
                "extract_content": True,
                "output_format": "markdown"
            })
            
            if not result.get("success"):
                return []
            
            search_results = []
            pages = result.get("crawl_result", {}).get("pages", [])
            
            for page in pages[:3]:
                content = page.get("content", "")
                if len(content) > 100:  # Filter out very short content
                    search_result = SearchResult(
                        content=content[:1000],  # Limit content length
                        score=0.7,  # Default web content score
                        source_type=SourceType.WEB,
                        source_id=page.get("url", ""),
                        metadata={
                            "url": page.get("url", ""),
                            "title": page.get("title", ""),
                            "crawl_timestamp": datetime.utcnow().isoformat()
                        },
                        relevance_factors={
                            "freshness": 1.0,  # Recent content
                            "authority": 0.7,   # Estimated authority
                            "content_quality": min(len(content) / 1000, 1.0)
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    async def _crawl_and_extract(self, url: str, query: str) -> List[SearchResult]:
        """Crawl specific URL and extract relevant content."""
        try:
            result = await self.mcp_client.call_tool("extract_webpage_content", {
                "url": url,
                "strategy": "llm",
                "instruction": f"Extract information relevant to: {query}",
                "output_format": "markdown"
            })
            
            if result.get("success"):
                content = result.get("extracted_content", "")
                if content:
                    return [SearchResult(
                        content=content[:800],
                        score=0.8,
                        source_type=SourceType.WEB,
                        source_id=url,
                        metadata={"url": url, "extraction_method": "llm"},
                        relevance_factors={"targeted_extraction": 1.0}
                    )]
            
            return []
        
        except Exception as e:
            logger.error(f"Web crawl error for {url}: {e}")
            return []

class ResultFusion:
    """Fuses and ranks results from multiple search engines."""
    
    def __init__(self):
        self.weights = {
            SourceType.VECTOR: 0.4,
            SourceType.GRAPH: 0.3,
            SourceType.WEB: 0.3
        }
    
    def fuse_results(self, 
                    vector_results: List[SearchResult],
                    graph_results: List[SearchResult],
                    web_results: List[SearchResult],
                    context: QueryContext) -> List[SearchResult]:
        """Fuse and rank results from different sources."""
        
        all_results = vector_results + graph_results + web_results
        
        # Remove duplicates based on content similarity
        unique_results = self._remove_duplicates(all_results)
        
        # Calculate fusion scores
        for result in unique_results:
            result.score = self._calculate_fusion_score(result, context)
        
        # Sort by fusion score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:context.max_results]
    
    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        
        for result in results:
            is_duplicate = False
            for unique_result in unique_results:
                similarity = self._calculate_content_similarity(
                    result.content, 
                    unique_result.content
                )
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    # Keep the result with higher score
                    if result.score > unique_result.score:
                        unique_results.remove(unique_result)
                        unique_results.append(result)
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity (simplified implementation)."""
        # In a real implementation, you'd use more sophisticated similarity measures
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_fusion_score(self, result: SearchResult, context: QueryContext) -> float:
        """Calculate fusion score combining multiple factors."""
        base_score = result.score
        source_weight = self.weights.get(result.source_type, 0.3)
        
        # Apply source type weighting
        weighted_score = base_score * source_weight
        
        # Apply relevance factors
        relevance_bonus = 0.0
        for factor, value in result.relevance_factors.items():
            if factor == "semantic_similarity":
                relevance_bonus += value * 0.3
            elif factor == "relationship_count":
                relevance_bonus += min(value / 10, 0.2)  # Cap at 0.2
            elif factor == "freshness":
                relevance_bonus += value * 0.2
            elif factor == "content_quality":
                relevance_bonus += value * 0.1
        
        # Apply query type bonuses
        if context.query_type == QueryType.FACTUAL and result.source_type == SourceType.VECTOR:
            weighted_score *= 1.2
        elif context.query_type == QueryType.ANALYTICAL and result.source_type == SourceType.GRAPH:
            weighted_score *= 1.3
        elif context.query_type == QueryType.TEMPORAL and result.source_type == SourceType.WEB:
            weighted_score *= 1.4
        
        final_score = min(weighted_score + relevance_bonus, 1.0)
        return final_score

class AnswerGenerator:
    """Generates coherent answers from search results."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
    
    async def generate_answer(self, 
                            query: str,
                            results: List[SearchResult],
                            context: QueryContext) -> Tuple[str, float, List[str]]:
        """Generate answer from search results."""
        
        if not results:
            return "I couldn't find relevant information to answer your question.", 0.0, []
        
        # Extract the most relevant content
        top_results = results[:5]  # Use top 5 results
        
        # Create context for answer generation
        context_text = self._create_context_text(top_results)
        
        # Generate answer based on query type
        if context.query_type == QueryType.FACTUAL:
            answer = await self._generate_factual_answer(query, context_text, top_results)
        elif context.query_type == QueryType.COMPARATIVE:
            answer = await self._generate_comparative_answer(query, context_text, top_results)
        else:
            answer = await self._generate_general_answer(query, context_text, top_results)
        
        # Calculate confidence
        confidence = self._calculate_answer_confidence(top_results, context)
        
        # Generate suggestions for follow-up queries
        suggestions = await self._generate_suggestions(query, top_results, context)
        
        return answer, confidence, suggestions
    
    def _create_context_text(self, results: List[SearchResult]) -> str:
        """Create context text from search results."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source_prefix = f"[Source {i} - {result.source_type.value}]"
            content = result.content[:500]  # Limit content length
            context_parts.append(f"{source_prefix}: {content}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_factual_answer(self, 
                                     query: str, 
                                     context: str,
                                     results: List[SearchResult]) -> str:
        """Generate factual answer."""
        # For factual queries, prioritize the highest-scored result
        best_result = results[0]
        
        # Create a structured answer
        answer = f"Based on the available information: {best_result.content[:300]}"
        
        # Add supporting information from other sources
        if len(results) > 1:
            supporting_info = []
            for result in results[1:3]:  # Add up to 2 supporting sources
                if result.source_type != best_result.source_type:
                    supporting_info.append(result.content[:150])
            
            if supporting_info:
                answer += f"\n\nAdditional context: {' '.join(supporting_info)}"
        
        return answer
    
    async def _generate_comparative_answer(self,
                                         query: str,
                                         context: str,
                                         results: List[SearchResult]) -> str:
        """Generate comparative answer."""
        # For comparative queries, try to identify the entities being compared
        entities = self._extract_entities_from_query(query)
        
        answer = f"Comparing the available information:\n\n"
        
        # Group results by relevance to each entity
        entity_info = {}
        for result in results[:4]:
            for entity in entities:
                if entity.lower() in result.content.lower():
                    if entity not in entity_info:
                        entity_info[entity] = []
                    entity_info[entity].append(result.content[:200])
        
        # Create comparative structure
        for entity, info_list in entity_info.items():
            answer += f"**{entity}**: {' '.join(info_list[:2])}\n\n"
        
        if not entity_info:
            # Fallback to general comparison
            answer = f"Based on the available information: {results[0].content[:400]}"
        
        return answer
    
    async def _generate_general_answer(self,
                                     query: str,
                                     context: str,
                                     results: List[SearchResult]) -> str:
        """Generate general answer."""
        # Combine information from multiple sources
        answer_parts = []
        
        # Start with the best result
        answer_parts.append(results[0].content[:300])
        
        # Add complementary information from different source types
        seen_sources = {results[0].source_type}
        
        for result in results[1:4]:
            if result.source_type not in seen_sources:
                answer_parts.append(result.content[:200])
                seen_sources.add(result.source_type)
        
        return " ".join(answer_parts)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from comparative queries."""
        # Simple entity extraction for comparative queries
        comparative_words = ["vs", "versus", "compare", "difference between"]
        
        for word in comparative_words:
            if word in query.lower():
                parts = query.lower().split(word)
                if len(parts) == 2:
                    entity1 = parts[0].strip().replace("the", "").strip()
                    entity2 = parts[1].strip().replace("the", "").strip()
                    return [entity1, entity2]
        
        # Fallback: extract nouns (simplified)
        words = query.split()
        entities = [word for word in words if len(word) > 3 and word.isalpha()]
        return entities[:2]
    
    def _calculate_answer_confidence(self, 
                                   results: List[SearchResult],
                                   context: QueryContext) -> float:
        """Calculate confidence in the generated answer."""
        if not results:
            return 0.0
        
        # Base confidence from result scores
        avg_score = sum(result.score for result in results[:3]) / min(len(results), 3)
        
        # Bonus for multiple source types
        source_types = set(result.source_type for result in results[:5])
        source_diversity_bonus = min(len(source_types) * 0.1, 0.3)
        
        # Bonus for high-quality sources
        quality_bonus = 0.0
        for result in results[:3]:
            if result.source_type == SourceType.GRAPH:
                quality_bonus += 0.05  # Graph sources are generally high quality
            elif "edu" in result.source_id or "wikipedia" in result.source_id:
                quality_bonus += 0.05  # Educational sources
        
        # Penalty for low result count
        count_penalty = 0.0 if len(results) >= 3 else 0.1 * (3 - len(results))
        
        confidence = min(avg_score + source_diversity_bonus + quality_bonus - count_penalty, 1.0)
        return max(confidence, 0.0)
    
    async def _generate_suggestions(self,
                                  query: str,
                                  results: List[SearchResult],
                                  context: QueryContext) -> List[str]:
        """Generate follow-up query suggestions."""
        suggestions = []
        
        # Extract entities and concepts from results
        entities = set()
        for result in results[:3]:
            # Simple entity extraction from content
            words = result.content.split()
            for i, word in enumerate(words):
                if word.istitle() and len(word) > 3:
                    entities.add(word)
                    if i < len(words) - 1 and words[i + 1].istitle():
                        entities.add(f"{word} {words[i + 1]}")
        
        # Generate suggestions based on entities
        for entity in list(entities)[:3]:
            if entity.lower() not in query.lower():
                if context.query_type == QueryType.FACTUAL:
                    suggestions.append(f"What is {entity}?")
                elif context.query_type == QueryType.ANALYTICAL:
                    suggestions.append(f"How does {entity} relate to {query}?")
                else:
                    suggestions.append(f"Tell me more about {entity}")
        
        # Add query type specific suggestions
        if context.query_type == QueryType.FACTUAL:
            suggestions.append(f"Applications of {query}")
            suggestions.append(f"History of {query}")
        elif context.query_type == QueryType.COMPARATIVE:
            suggestions.append(f"Pros and cons of {query}")
        
        return suggestions[:3]

class HybridSearchSystem:
    """Main hybrid search system orchestrating all components."""
    
    def __init__(self, mcp_server_url: str, auth_token: str = None):
        self.mcp_client = MCPClient(mcp_server_url, auth_token)
        self.query_analyzer = QueryAnalyzer(self.mcp_client)
        self.vector_engine = VectorSearchEngine(self.mcp_client)
        self.graph_engine = GraphSearchEngine(self.mcp_client)
        self.web_engine = WebSearchEngine(self.mcp_client)
        self.result_fusion = ResultFusion()
        self.answer_generator = AnswerGenerator(self.mcp_client)
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def search(self, query: str, **kwargs) -> HybridSearchResponse:
        """Perform hybrid search across all sources."""
        start_time = datetime.utcnow()
        
        try:
            # Analyze query
            context = await self.query_analyzer.analyze_query(query)
            
            # Apply any user-provided filters
            context.filters.update(kwargs.get("filters", {}))
            context.max_results = kwargs.get("max_results", context.max_results)
            context.min_confidence = kwargs.get("min_confidence", context.min_confidence)
            
            logger.info(f"Processing {context.query_type.value} query: {query}")
            
            # Parallel search across all engines
            vector_task = self.vector_engine.search(context)
            graph_task = self.graph_engine.search(context)
            web_task = self.web_engine.search(context)
            
            vector_results, graph_results, web_results = await asyncio.gather(
                vector_task, graph_task, web_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            
            if isinstance(graph_results, Exception):
                logger.error(f"Graph search failed: {graph_results}")
                graph_results = []
            
            if isinstance(web_results, Exception):
                logger.error(f"Web search failed: {web_results}")
                web_results = []
            
            # Fuse results
            fused_results = self.result_fusion.fuse_results(
                vector_results, graph_results, web_results, context
            )
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in fused_results 
                if result.score >= context.min_confidence
            ]
            
            # Generate answer
            answer, confidence, suggestions = await self.answer_generator.generate_answer(
                query, filtered_results, context
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update stats
            self.stats["total_queries"] += 1
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_queries"] - 1) + processing_time)
                / self.stats["total_queries"]
            )
            
            # Create response
            response = HybridSearchResponse(
                query=query,
                answer=answer,
                confidence=confidence,
                sources=filtered_results,
                processing_time_ms=processing_time,
                search_stats={
                    "vector_results": len(vector_results),
                    "graph_results": len(graph_results),
                    "web_results": len(web_results),
                    "fused_results": len(fused_results),
                    "filtered_results": len(filtered_results),
                    "query_type": context.query_type.value,
                    "search_depth": context.search_depth
                },
                suggestions=suggestions
            )
            
            logger.info(f"Search completed in {processing_time:.1f}ms with {len(filtered_results)} results")
            return response
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HybridSearchResponse(
                query=query,
                answer=f"I encountered an error while searching: {str(e)}",
                confidence=0.0,
                sources=[],
                processing_time_ms=processing_time,
                search_stats={"error": str(e)},
                suggestions=[]
            )
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            **self.stats,
            "circuit_breaker_state": self.mcp_client.circuit_breaker.state,
            "failure_count": self.mcp_client.circuit_breaker.failure_count
        }

# Demo and Testing Functions

async def demo_hybrid_search():
    """Demonstrate the hybrid search system."""
    print("🚀 Hybrid Search System Demo")
    print("=" * 60)
    
    # Initialize system
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    auth_token = os.getenv("MCP_AUTH_TOKEN", "")
    
    search_system = HybridSearchSystem(mcp_server_url, auth_token)
    
    # Sample queries of different types
    test_queries = [
        {
            "query": "What is machine learning?",
            "description": "Factual query about ML definition"
        },
        {
            "query": "Compare neural networks and decision trees",
            "description": "Comparative analysis query"
        },
        {
            "query": "Recent developments in artificial intelligence",
            "description": "Temporal query for recent information"
        },
        {
            "query": "How do transformers work in natural language processing?",
            "description": "Analytical technical query"
        },
        {
            "query": "Overview of computer vision applications",
            "description": "Exploratory domain overview"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {test_case['description']}")
        print(f"   Question: {test_case['query']}")
        print("   " + "-" * 50)
        
        try:
            response = await search_system.search(test_case["query"])
            
            print(f"   ✅ Answer (Confidence: {response.confidence:.2f}):")
            print(f"      {response.answer[:200]}...")
            
            print(f"\n   📊 Search Statistics:")
            stats = response.search_stats
            print(f"      • Vector results: {stats.get('vector_results', 0)}")
            print(f"      • Graph results: {stats.get('graph_results', 0)}")
            print(f"      • Web results: {stats.get('web_results', 0)}")
            print(f"      • Final results: {stats.get('filtered_results', 0)}")
            print(f"      • Processing time: {response.processing_time_ms:.1f}ms")
            print(f"      • Query type: {stats.get('query_type', 'unknown')}")
            
            print(f"\n   📚 Top Sources ({len(response.sources[:3])}):")
            for j, source in enumerate(response.sources[:3], 1):
                print(f"      {j}. {source.source_type.value} (score: {source.score:.2f})")
                print(f"         {source.content[:100]}...")
            
            if response.suggestions:
                print(f"\n   💡 Suggestions:")
                for suggestion in response.suggestions:
                    print(f"      • {suggestion}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
    
    # Show system statistics
    print(f"\n📈 System Performance:")
    stats = await search_system.get_system_stats()
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Average response time: {stats['avg_response_time']:.1f}ms")
    print(f"   Circuit breaker state: {stats['circuit_breaker_state']}")

async def interactive_search():
    """Interactive search session."""
    print("🤖 Interactive Hybrid Search")
    print("=" * 40)
    print("Ask questions combining vector, graph, and web search!")
    print("Type 'quit' to exit.\n")
    
    # Initialize system
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    auth_token = os.getenv("MCP_AUTH_TOKEN", "")
    
    search_system = HybridSearchSystem(mcp_server_url, auth_token)
    
    while True:
        try:
            query = input("❓ Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("🔍 Searching across all sources...")
            
            response = await search_system.search(query)
            
            print(f"\n💡 Answer (Confidence: {response.confidence:.2f}):")
            print(f"   {response.answer}")
            
            print(f"\n⚡ Processed in {response.processing_time_ms:.1f}ms")
            
            # Show source breakdown
            source_counts = {}
            for source in response.sources:
                source_type = source.source_type.value
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            if source_counts:
                print(f"📊 Sources: {', '.join([f'{k}: {v}' for k, v in source_counts.items()])}")
            
            # Show suggestions
            if response.suggestions:
                print(f"\n💭 Related questions:")
                for suggestion in response.suggestions:
                    print(f"   • {suggestion}")
            
            print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("👋 Thanks for using the Hybrid Search System!")

async def main():
    """Main function to run the hybrid search demo."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await interactive_search()
    else:
        await demo_hybrid_search()

if __name__ == "__main__":
    asyncio.run(main())