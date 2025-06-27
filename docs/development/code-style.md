# Code Style Guide

This guide establishes coding standards and best practices for the Unified MCP Intelligence Server to ensure consistent, maintainable, and professional code.

## 🎯 Core Principles

### Code Philosophy

1. **Readability First**: Code is read more often than written
2. **Explicit is Better**: Clear intentions over clever shortcuts
3. **Consistency**: Follow established patterns throughout the codebase
4. **Simplicity**: Prefer simple solutions over complex ones
5. **Maintainability**: Write code for the next developer

### Quality Standards

- **Type Safety**: Comprehensive type hints
- **Error Handling**: Explicit error handling with proper logging
- **Documentation**: Clear docstrings and comments
- **Testing**: High test coverage with meaningful tests
- **Performance**: Efficient and scalable implementations

## 🐍 Python Style Guide

### Base Standards

We follow **PEP 8** with specific extensions and modifications configured in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 88
```

### Import Organization

```python
# Standard library imports
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Third-party imports
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import structlog

# Local application imports
from qdrant_neo4j_crawl4ai_mcp.config import Settings
from qdrant_neo4j_crawl4ai_mcp.services.vector_service import VectorService
from qdrant_neo4j_crawl4ai_mcp.models.vector_models import VectorSearchRequest
```

**Import Order (enforced by ruff):**
1. Standard library
2. Third-party packages
3. Local application modules

### Type Hints

**Required for all public APIs:**

```python
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Function signatures
async def search_vectors(
    query: str,
    collection_name: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> VectorSearchResponse:
    """Search vectors with type safety."""
    pass

# Class attributes
class VectorService:
    """Vector service with typed attributes."""
    
    def __init__(self, config: VectorServiceConfig) -> None:
        self.config: VectorServiceConfig = config
        self.client: Optional[QdrantClient] = None
        self.is_initialized: bool = False
        self._embedding_cache: Dict[str, List[float]] = {}

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class ServiceResponse(Generic[T]):
    """Generic service response."""
    
    def __init__(self, data: T, status: str = "success") -> None:
        self.data: T = data
        self.status: str = status
```

### Function and Method Design

#### Function Naming

```python
# Good: Clear, descriptive names
async def generate_text_embedding(text: str) -> List[float]:
    """Generate embedding vector for text."""
    pass

async def store_document_vector(document: Document) -> StorageResult:
    """Store document with generated vector."""
    pass

# Avoid: Unclear or abbreviated names
async def gen_emb(text: str) -> List[float]:  # Too abbreviated
    pass

async def process_data(data: Any) -> Any:  # Too generic
    pass
```

#### Function Structure

```python
async def search_similar_documents(
    query: str,
    collection_name: str,
    limit: int = 10,
    score_threshold: float = 0.0,
    filters: Optional[Dict[str, Any]] = None
) -> VectorSearchResponse:
    """
    Search for documents similar to the query text.
    
    This function performs semantic search by:
    1. Converting query text to embeddings
    2. Performing vector similarity search
    3. Filtering and ranking results
    4. Returning formatted response
    
    Args:
        query: Text to search for
        collection_name: Target collection name
        limit: Maximum number of results (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        filters: Optional metadata filters
        
    Returns:
        VectorSearchResponse with ranked results
        
    Raises:
        ValueError: If query is empty or limit is invalid
        ServiceError: If search operation fails
        
    Example:
        ```python
        response = await search_similar_documents(
            query="machine learning",
            collection_name="research_papers",
            limit=5,
            score_threshold=0.7
        )
        
        for result in response.results:
            print(f"Score: {result.score}, Title: {result.metadata['title']}")
        ```
    """
    # 1. Input validation
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100")
    
    # 2. Generate embedding
    try:
        embedding = await self._generate_embedding(query)
    except Exception as e:
        logger.exception("Failed to generate embedding", query=query)
        raise ServiceError(f"Embedding generation failed: {e}") from e
    
    # 3. Perform search
    try:
        search_results = await self._client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=self._build_filter(filters) if filters else None
        )
    except Exception as e:
        logger.exception("Vector search failed", collection=collection_name)
        raise ServiceError(f"Search operation failed: {e}") from e
    
    # 4. Format response
    return VectorSearchResponse(
        results=[
            VectorSearchResult(
                id=result.id,
                score=result.score,
                content=result.payload.get("content", ""),
                metadata=result.payload
            )
            for result in search_results
        ],
        total_count=len(search_results),
        search_time_ms=search_results.search_time_ms,
        timestamp=datetime.utcnow()
    )
```

### Class Design

#### Class Structure

```python
class VectorService:
    """
    Service for vector database operations.
    
    This service provides high-level operations for vector storage,
    search, and management using Qdrant as the backend.
    
    Attributes:
        config: Service configuration
        is_initialized: Whether service is ready for operations
        
    Example:
        ```python
        config = VectorServiceConfig(qdrant_url="http://localhost:6333")
        service = VectorService(config)
        await service.initialize()
        ```
    """
    
    def __init__(self, config: VectorServiceConfig) -> None:
        """
        Initialize vector service.
        
        Args:
            config: Service configuration object
        """
        self.config: VectorServiceConfig = config
        self.is_initialized: bool = False
        
        # Private attributes
        self._client: Optional[QdrantClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize service resources."""
        if self.is_initialized:
            return
        
        try:
            # Initialize Qdrant client
            self._client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.connection_timeout
            )
            
            # Initialize embedding model
            self._embedding_model = SentenceTransformer(
                self.config.default_embedding_model
            )
            
            # Verify connection
            await self._client.get_collections()
            
            self.is_initialized = True
            logger.info("Vector service initialized successfully")
            
        except Exception as e:
            logger.exception("Vector service initialization failed")
            raise ServiceInitializationError(f"Initialization failed: {e}") from e
    
    async def shutdown(self) -> None:
        """Clean up service resources."""
        if not self.is_initialized:
            return
        
        try:
            if self._client:
                await self._client.close()
            
            self._cache.clear()
            self.is_initialized = False
            
            logger.info("Vector service shutdown completed")
            
        except Exception as e:
            logger.warning("Error during service shutdown", error=str(e))
    
    # Public interface methods
    async def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """Public search interface."""
        self._ensure_initialized()
        return await self._perform_search(request)
    
    # Private helper methods
    def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self.is_initialized:
            raise ServiceError("Service not initialized")
    
    async def _perform_search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """Internal search implementation."""
        # Implementation details
        pass
```

### Error Handling

#### Exception Hierarchy

```python
# Base exceptions
class MCPServerError(Exception):
    """Base exception for MCP server errors."""
    pass

class ServiceError(MCPServerError):
    """Service-level errors."""
    pass

class ValidationError(MCPServerError):
    """Input validation errors."""
    pass

# Specific exceptions
class VectorServiceError(ServiceError):
    """Vector service specific errors."""
    pass

class GraphServiceError(ServiceError):
    """Graph service specific errors."""
    pass

class AuthenticationError(MCPServerError):
    """Authentication and authorization errors."""
    pass
```

#### Error Handling Pattern

```python
async def store_vector(self, request: VectorStoreRequest) -> VectorStoreResponse:
    """Store vector with comprehensive error handling."""
    try:
        # 1. Input validation
        self._validate_store_request(request)
        
        # 2. Generate embedding
        embedding = await self._generate_embedding(request.content)
        
        # 3. Store in database
        result = await self._client.upsert(
            collection_name=request.collection_name,
            points=[{
                "id": request.id or str(uuid.uuid4()),
                "vector": embedding,
                "payload": request.metadata
            }]
        )
        
        # 4. Return response
        return VectorStoreResponse(
            id=result.point_id,
            status="stored",
            timestamp=datetime.utcnow()
        )
        
    except ValidationError:
        # Re-raise validation errors
        raise
    except ConnectionError as e:
        logger.exception("Database connection failed", request_id=request.id)
        raise VectorServiceError(f"Storage failed: connection error") from e
    except Exception as e:
        logger.exception("Unexpected error during vector storage", request_id=request.id)
        raise VectorServiceError(f"Storage failed: {e}") from e

def _validate_store_request(self, request: VectorStoreRequest) -> None:
    """Validate store request parameters."""
    if not request.content.strip():
        raise ValidationError("Content cannot be empty")
    
    if len(request.content) > self.config.max_content_length:
        raise ValidationError(f"Content exceeds maximum length: {len(request.content)}")
    
    if not request.collection_name:
        raise ValidationError("Collection name is required")
```

### Logging Standards

#### Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

async def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
    """Search with structured logging."""
    # Log request start
    logger.info(
        "Vector search started",
        collection=request.collection_name,
        query_length=len(request.query),
        limit=request.limit,
        has_filters=bool(request.filters)
    )
    
    start_time = time.time()
    
    try:
        # Perform search
        response = await self._perform_search(request)
        
        # Log success
        logger.info(
            "Vector search completed",
            collection=request.collection_name,
            results_count=len(response.results),
            search_time_ms=response.search_time_ms,
            total_time_ms=int((time.time() - start_time) * 1000)
        )
        
        return response
        
    except Exception as e:
        # Log error with context
        logger.exception(
            "Vector search failed",
            collection=request.collection_name,
            query_length=len(request.query),
            error_type=type(e).__name__,
            total_time_ms=int((time.time() - start_time) * 1000)
        )
        raise
```

#### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about operations
- **WARNING**: Something unexpected but not critical
- **ERROR**: Error that caused operation to fail
- **CRITICAL**: Serious error that may cause system to abort

### Documentation Standards

#### Docstring Format

We use **Google-style docstrings**:

```python
def calculate_similarity_score(
    vector_a: List[float], 
    vector_b: List[float],
    metric: str = "cosine"
) -> float:
    """
    Calculate similarity score between two vectors.
    
    This function computes the similarity between two embedding vectors
    using the specified distance metric. Supports cosine, euclidean,
    and dot product similarity measures.
    
    Args:
        vector_a: First vector for comparison
        vector_b: Second vector for comparison  
        metric: Similarity metric ('cosine', 'euclidean', 'dot_product')
    
    Returns:
        Similarity score between 0.0 and 1.0, where 1.0 indicates
        perfect similarity
    
    Raises:
        ValueError: If vectors have different dimensions
        ValueError: If metric is not supported
    
    Example:
        ```python
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.15, 0.25, 0.35]
        
        score = calculate_similarity_score(vec1, vec2, metric="cosine")
        print(f"Similarity: {score:.3f}")  # Output: Similarity: 0.999
        ```
        
    Note:
        Cosine similarity is recommended for normalized embeddings.
        Use euclidean distance for geometric similarity comparison.
    """
    pass
```

## 🏗️ Architecture Patterns

### Service Layer Pattern

```python
# Service interface
from abc import ABC, abstractmethod

class VectorServiceInterface(ABC):
    """Interface for vector service implementations."""
    
    @abstractmethod
    async def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def store_vector(self, request: VectorStoreRequest) -> VectorStoreResponse:
        """Store a vector."""
        pass

# Service implementation
class VectorService(VectorServiceInterface):
    """Qdrant-based vector service implementation."""
    
    async def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """Implementation using Qdrant."""
        pass
```

### Dependency Injection

```python
# Service dependencies
class ServiceContainer:
    """Dependency injection container."""
    
    def __init__(self) -> None:
        self._services: Dict[type, Any] = {}
    
    def register(self, service_type: type, instance: Any) -> None:
        """Register service instance."""
        self._services[service_type] = instance
    
    def get(self, service_type: type) -> Any:
        """Get service instance."""
        return self._services.get(service_type)

# Usage in FastMCP tools
def register_vector_tools(mcp: FastMCP, container: ServiceContainer) -> None:
    """Register vector tools with dependency injection."""
    
    @mcp.tool()
    async def search_vectors(request: VectorSearchRequest) -> VectorSearchResponse:
        """Vector search tool."""
        service = container.get(VectorService)
        return await service.search_vectors(request)
```

### Configuration Management

```python
# Configuration with validation
from pydantic import BaseSettings, validator
from typing import List

class VectorServiceConfig(BaseSettings):
    """Vector service configuration."""
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    default_collection: str = "documents"
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_content_length: int = 10000
    connection_timeout: int = 30
    max_retries: int = 3
    
    @validator('max_content_length')
    def validate_content_length(cls, v: int) -> int:
        """Validate content length limit."""
        if v <= 0:
            raise ValueError("Content length must be positive")
        return v
    
    @validator('connection_timeout')
    def validate_timeout(cls, v: int) -> int:
        """Validate connection timeout."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    class Config:
        env_prefix = "VECTOR_"
        case_sensitive = False
```

## 🔧 FastMCP Integration

### Tool Registration Pattern

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Tool request/response models
class VectorSearchToolRequest(BaseModel):
    """Vector search tool request."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    collection_name: str = Field(..., description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Result limit")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

class VectorSearchToolResponse(BaseModel):
    """Vector search tool response."""
    
    results: List[VectorSearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total results found")
    search_time_ms: float = Field(..., description="Search duration")

# Tool registration
def register_vector_tools(mcp: FastMCP, service: VectorService) -> None:
    """Register vector-related MCP tools."""
    
    @mcp.tool()
    async def vector_search(request: VectorSearchToolRequest) -> VectorSearchToolResponse:
        """
        Semantic vector search tool.
        
        Performs similarity search across vector collections using embedding-based
        retrieval. Supports metadata filtering and relevance scoring.
        """
        try:
            # Convert tool request to service request
            service_request = VectorSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                filters=request.filters
            )
            
            # Execute search
            service_response = await service.search_vectors(service_request)
            
            # Convert service response to tool response
            return VectorSearchToolResponse(
                results=service_response.results,
                total_count=service_response.total_count,
                search_time_ms=service_response.search_time_ms
            )
            
        except Exception as e:
            logger.exception("Vector search tool failed", error=str(e))
            raise ToolExecutionError(f"Search failed: {e}") from e
```

### Resource Management

```python
# MCP resource implementation
@mcp.resource("vector://collections/{collection_name}")
async def get_collection_info(collection_name: str) -> Dict[str, Any]:
    """Get vector collection information."""
    try:
        service = get_vector_service()
        collection_info = await service.get_collection_info(collection_name)
        
        return {
            "name": collection_info.name,
            "vector_size": collection_info.vector_size,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
            "created_at": collection_info.created_at.isoformat()
        }
        
    except Exception as e:
        logger.exception("Failed to get collection info", collection=collection_name)
        raise ResourceError(f"Collection info failed: {e}") from e
```

## 📊 Performance Guidelines

### Async/Await Best Practices

```python
# Good: Proper async usage
async def process_multiple_documents(documents: List[Document]) -> List[ProcessingResult]:
    """Process documents concurrently."""
    tasks = [process_single_document(doc) for doc in documents]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def process_single_document(document: Document) -> ProcessingResult:
    """Process individual document."""
    # CPU-bound work should be in thread pool
    embedding = await asyncio.get_event_loop().run_in_executor(
        None, generate_embedding, document.content
    )
    
    # I/O-bound work can be awaited directly
    result = await store_embedding(embedding)
    return result

# Avoid: Blocking operations in async functions
async def bad_example(documents: List[Document]) -> List[ProcessingResult]:
    """Don't do this - blocks event loop."""
    results = []
    for doc in documents:
        # This blocks the event loop
        embedding = generate_embedding_sync(doc.content)  # Bad!
        results.append(embedding)
    return results
```

### Memory Management

```python
# Good: Efficient memory usage
async def process_large_dataset(dataset_path: str) -> ProcessingStats:
    """Process large dataset with memory efficiency."""
    stats = ProcessingStats()
    
    # Process in chunks to avoid memory issues
    async for chunk in read_dataset_chunks(dataset_path, chunk_size=1000):
        chunk_results = await process_chunk(chunk)
        stats.update(chunk_results)
        
        # Clear chunk from memory
        del chunk
        del chunk_results
    
    return stats

# Use generators for large sequences
async def read_dataset_chunks(path: str, chunk_size: int) -> AsyncGenerator[List[Document], None]:
    """Read dataset in memory-efficient chunks."""
    current_chunk = []
    
    async with aiofiles.open(path, 'r') as file:
        async for line in file:
            document = parse_document(line)
            current_chunk.append(document)
            
            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []
        
        # Yield remaining documents
        if current_chunk:
            yield current_chunk
```

## 🔒 Security Practices

### Input Validation

```python
from pydantic import BaseModel, Field, validator
import re

class SecureRequest(BaseModel):
    """Request model with security validation."""
    
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    filters: Optional[Dict[str, Any]] = Field(default=None)
    
    @validator('query')
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', v.strip())
        
        if not sanitized:
            raise ValueError("Query cannot be empty after sanitization")
        
        return sanitized
    
    @validator('filters')
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filter structure."""
        if v is None:
            return v
        
        # Limit filter complexity
        if len(v) > 10:
            raise ValueError("Too many filters")
        
        # Validate filter values
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 100:
                raise ValueError(f"Invalid filter key: {key}")
        
        return v
```

### Secure Logging

```python
import structlog
from typing import Any, Dict

def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive information from log data."""
    sanitized = {}
    sensitive_keys = {'password', 'token', 'api_key', 'secret'}
    
    for key, value in data.items():
        if key.lower() in sensitive_keys:
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value
    
    return sanitized

# Usage in service methods
async def authenticate_user(credentials: UserCredentials) -> AuthResult:
    """Authenticate user with secure logging."""
    logger.info(
        "Authentication attempt",
        username=credentials.username,
        # Don't log password
        **sanitize_log_data({"ip": request.client.host})
    )
    
    # Authentication logic
    pass
```

## 🧪 Testing Standards

### Test Structure

```python
# tests/unit/test_vector_service.py
import pytest
from unittest.mock import AsyncMock, Mock
from qdrant_neo4j_crawl4ai_mcp.services.vector_service import VectorService

class TestVectorService:
    """Test suite for VectorService."""
    
    @pytest.fixture
    def vector_config(self) -> VectorServiceConfig:
        """Vector service configuration fixture."""
        return VectorServiceConfig(
            qdrant_url="http://localhost:6333",
            default_collection="test_collection"
        )
    
    @pytest.fixture
    def mock_qdrant_client(self) -> Mock:
        """Mock Qdrant client fixture."""
        client = Mock()
        client.search = AsyncMock(return_value=[])
        client.upsert = AsyncMock(return_value={"status": "success"})
        return client
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_vectors_success(
        self, 
        vector_config: VectorServiceConfig,
        mock_qdrant_client: Mock
    ) -> None:
        """Test successful vector search."""
        # Arrange
        service = VectorService(vector_config)
        service._client = mock_qdrant_client
        service.is_initialized = True
        
        request = VectorSearchRequest(
            query="test query",
            collection_name="test_collection",
            limit=5
        )
        
        # Act
        response = await service.search_vectors(request)
        
        # Assert
        assert response is not None
        assert isinstance(response, VectorSearchResponse)
        mock_qdrant_client.search.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_vectors_empty_query_raises_error(
        self,
        vector_config: VectorServiceConfig
    ) -> None:
        """Test that empty query raises ValueError."""
        # Arrange
        service = VectorService(vector_config)
        service.is_initialized = True
        
        request = VectorSearchRequest(
            query="",  # Empty query
            collection_name="test_collection"
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.search_vectors(request)
```

## 📝 Code Review Checklist

### Pre-Review Checklist

- [ ] **Formatting**: Code is formatted with `ruff format`
- [ ] **Linting**: No `ruff check` violations
- [ ] **Type Checking**: Passes `mypy` checks
- [ ] **Tests**: New code has test coverage ≥90%
- [ ] **Documentation**: Public APIs have docstrings
- [ ] **Security**: No obvious security vulnerabilities

### Review Guidelines

```python
# Good: Clear variable names and structure
async def calculate_document_similarity(
    document_a: Document,
    document_b: Document,
    similarity_metric: str = "cosine"
) -> float:
    """Calculate similarity between two documents."""
    embedding_a = await generate_embedding(document_a.content)
    embedding_b = await generate_embedding(document_b.content)
    
    return compute_similarity(embedding_a, embedding_b, similarity_metric)

# Avoid: Unclear names and structure
async def calc_sim(doc1, doc2, metric="cosine"):  # Unclear names
    emb1 = await gen_emb(doc1.content)  # Abbreviated names
    emb2 = await gen_emb(doc2.content)
    return comp_sim(emb1, emb2, metric)  # No type hints or docstring
```

## 🔧 Development Tools

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
```

### IDE Configuration

#### VS Code Settings

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 📚 Additional Resources

- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **MyPy Documentation**: https://mypy.readthedocs.io/
- **FastMCP Documentation**: https://fastmcp.readthedocs.io/
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Python Type Hints**: https://docs.python.org/3/library/typing.html

---

*Following these standards ensures our codebase remains clean, maintainable, and professional!* 🚀