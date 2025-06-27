# Contributing Guidelines

Welcome to the Unified MCP Intelligence Server project! We're excited to have you contribute to this production-ready agentic RAG system.

## 🤝 Getting Started

### Prerequisites

Before contributing, please:

1. **Read the Documentation**
   - [Local Setup Guide](local-setup.md)
   - [Testing Guidelines](testing.md)
   - [Code Style Guide](code-style.md)
   - [Architecture Documentation](../ARCHITECTURE.md)

2. **Set Up Development Environment**

   ```bash
   # Clone the repository
   git clone <repository-url>
   cd qdrant-neo4j-crawl4ai-mcp
   
   # Follow local setup guide
   uv install --extra dev
   pre-commit install
   docker-compose up -d
   ```

3. **Understand the Codebase**
   - Review the [API Reference](../API_REFERENCE.md)
   - Explore [Architecture Decision Records](../adrs/)
   - Run the test suite to ensure everything works

## 🔄 Contribution Workflow

### 1. Find or Create an Issue

- **Bug Reports**: Use the bug report template
- **Feature Requests**: Use the feature request template
- **Documentation**: Label with `documentation`
- **Questions**: Use GitHub Discussions

### 2. Fork and Branch

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/qdrant-neo4j-crawl4ai-mcp.git

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-fix-name
```

### 3. Make Changes

```bash
# Make your changes
# Write tests for new functionality
# Update documentation if needed

# Run quality checks
ruff format .
ruff check . --fix
mypy src/
pytest

# Commit changes
git add .
git commit -m "feat: add vector search optimization"
```

### 4. Submit Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Use the pull request template
```

## 📋 Issue Templates

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Configure with '...'
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.6]
- Version: [e.g., 1.0.0]

**Additional Context**
- Error logs
- Configuration files
- Screenshots (if applicable)
```

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
How you envision this feature working.

**Alternatives Considered**
Other approaches you've considered.

**Implementation Notes**
Any technical considerations or constraints.
```

## 🏗️ Development Guidelines

### Code Organization

```
src/qdrant_neo4j_crawl4ai_mcp/
├── main.py                 # Application entry point
├── config.py               # Configuration management
├── auth.py                 # Authentication & authorization
├── middleware.py           # FastAPI middleware
├── services/               # Core business logic
│   ├── vector_service.py   # Vector operations
│   ├── graph_service.py    # Graph operations
│   └── web_service.py      # Web intelligence
├── tools/                  # MCP tool implementations
│   ├── vector_tools.py     # Vector MCP tools
│   ├── graph_tools.py      # Graph MCP tools
│   └── web_tools.py        # Web MCP tools
└── models/                 # Pydantic data models
    ├── vector_models.py    # Vector-related models
    ├── graph_models.py     # Graph-related models
    └── web_models.py       # Web-related models
```

### Adding New Features

#### 1. New MCP Tools

```python
# Example: Adding a new vector analysis tool
from fastmcp import FastMCP
from pydantic import BaseModel, Field

class VectorAnalysisRequest(BaseModel):
    collection_name: str = Field(..., description="Collection to analyze")
    metric: str = Field(default="diversity", description="Analysis metric")

@mcp.tool()
async def analyze_vector_collection(request: VectorAnalysisRequest) -> dict:
    """
    Analyze vector collection for insights.
    
    Args:
        request: Analysis parameters
        
    Returns:
        Analysis results with metrics and insights
    """
    # Implementation here
    pass

# Register in tools/vector_tools.py
def register_vector_tools(mcp: FastMCP, service: VectorService) -> None:
    """Register all vector-related MCP tools."""
    # ... existing tools ...
    
    @mcp.tool()
    async def analyze_vector_collection(request: VectorAnalysisRequest) -> dict:
        return await service.analyze_collection(request)
```

#### 2. New Service Methods

```python
# In services/vector_service.py
class VectorService:
    async def analyze_collection(self, request: VectorAnalysisRequest) -> dict:
        """Analyze vector collection for patterns and insights."""
        try:
            # Input validation
            if not request.collection_name:
                raise ValueError("Collection name is required")
            
            # Business logic
            analysis_result = await self._perform_analysis(request)
            
            # Return structured result
            return VectorAnalysisResponse(
                collection_name=request.collection_name,
                metrics=analysis_result,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.exception("Collection analysis failed", error=str(e))
            raise ServiceError(f"Analysis failed: {e}") from e
```

#### 3. New Data Models

```python
# In models/vector_models.py
class VectorAnalysisResponse(BaseModel):
    """Response model for vector collection analysis."""
    
    collection_name: str = Field(..., description="Analyzed collection")
    metrics: dict[str, Any] = Field(..., description="Analysis metrics")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Collection diversity")
    cluster_count: int = Field(..., ge=0, description="Number of detected clusters")
    outlier_count: int = Field(..., ge=0, description="Number of outliers")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### Database Integrations

#### Adding New Vector Operations

```python
# In services/vector_service.py
async def bulk_upsert_vectors(
    self, 
    requests: list[VectorStoreRequest]
) -> BulkUpsertResponse:
    """Efficiently store multiple vectors in batch."""
    try:
        # Validate batch size
        if len(requests) > self.config.max_batch_size:
            raise ValueError(f"Batch size exceeds limit: {len(requests)}")
        
        # Generate embeddings in parallel
        embedding_tasks = [
            self._generate_embedding(req.content) 
            for req in requests
        ]
        embeddings = await asyncio.gather(*embedding_tasks)
        
        # Prepare points for bulk upsert
        points = [
            PointStruct(
                id=req.id or str(uuid.uuid4()),
                vector=embedding,
                payload=req.metadata
            )
            for req, embedding in zip(requests, embeddings)
        ]
        
        # Perform bulk upsert
        result = await self._client.upsert(
            collection_name=requests[0].collection_name,
            points=points
        )
        
        return BulkUpsertResponse(
            successful_count=len(points),
            failed_count=0,
            operation_time_ms=result.operation_time_ms
        )
        
    except Exception as e:
        logger.exception("Bulk upsert failed", error=str(e))
        raise
```

#### Adding New Graph Operations

```python
# In services/graph_service.py
async def create_knowledge_graph(
    self, 
    documents: list[Document]
) -> KnowledgeGraphResponse:
    """Create knowledge graph from document collection."""
    try:
        # Extract entities and relationships
        entities = await self._extract_entities(documents)
        relationships = await self._extract_relationships(entities)
        
        # Create graph structure
        async with self._driver.session() as session:
            # Create nodes
            for entity in entities:
                await session.run(
                    "MERGE (e:Entity {id: $id, name: $name, type: $type})",
                    id=entity.id,
                    name=entity.name,
                    type=entity.type
                )
            
            # Create relationships
            for rel in relationships:
                await session.run(
                    """
                    MATCH (a:Entity {id: $from_id})
                    MATCH (b:Entity {id: $to_id})
                    MERGE (a)-[r:RELATES_TO {type: $rel_type, confidence: $confidence}]->(b)
                    """,
                    from_id=rel.from_id,
                    to_id=rel.to_id,
                    rel_type=rel.type,
                    confidence=rel.confidence
                )
        
        return KnowledgeGraphResponse(
            node_count=len(entities),
            relationship_count=len(relationships),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception("Knowledge graph creation failed", error=str(e))
        raise
```

## ✅ Quality Standards

### Code Quality Requirements

1. **Type Hints**: All functions must have type hints
2. **Documentation**: Public APIs must have docstrings
3. **Error Handling**: Proper exception handling with logging
4. **Testing**: 90%+ test coverage for new code
5. **Performance**: No significant performance regressions

### Pre-commit Checks

All commits must pass:

```bash
# Formatting
ruff format .

# Linting
ruff check . --fix

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Tests
pytest --cov=qdrant_neo4j_crawl4ai_mcp --cov-fail-under=90
```

### Code Review Process

1. **Automated Checks**: All CI checks must pass
2. **Peer Review**: At least one approving review required
3. **Documentation Review**: Documentation changes reviewed
4. **Security Review**: Security-sensitive changes get extra scrutiny

## 🧪 Testing Requirements

### Test Coverage

- **New Features**: 95%+ test coverage
- **Bug Fixes**: Tests that reproduce the bug
- **Integration Tests**: For cross-service functionality
- **Performance Tests**: For performance-critical features

### Test Categories

```python
# Unit tests (fast, isolated)
@pytest.mark.unit
def test_vector_embedding_generation():
    """Test vector embedding generation."""
    pass

# Integration tests (require services)
@pytest.mark.integration
async def test_vector_search_integration():
    """Test vector search with real database."""
    pass

# Performance tests
@pytest.mark.performance
def test_bulk_operation_performance(benchmark):
    """Benchmark bulk operations."""
    pass

# Security tests
@pytest.mark.security
def test_input_validation():
    """Test input validation and sanitization."""
    pass
```

## 📝 Documentation Standards

### Code Documentation

```python
class VectorService:
    """
    Vector database service for semantic search operations.
    
    This service provides a high-level interface for vector operations
    including embedding generation, storage, and similarity search.
    
    Attributes:
        config: Service configuration
        is_initialized: Whether service is ready for operations
        
    Example:
        ```python
        config = VectorServiceConfig(qdrant_url="http://localhost:6333")
        service = VectorService(config)
        await service.initialize()
        
        # Store document
        response = await service.store_vector(VectorStoreRequest(
            content="Sample document",
            collection_name="documents"
        ))
        ```
    """
    
    async def search_vectors(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """
        Perform semantic vector search.
        
        Args:
            request: Search parameters including query, filters, and limits
            
        Returns:
            Search results with ranked matches and metadata
            
        Raises:
            ValueError: If request parameters are invalid
            ServiceError: If search operation fails
            
        Example:
            ```python
            request = VectorSearchRequest(
                query="artificial intelligence",
                collection_name="documents",
                limit=10
            )
            response = await service.search_vectors(request)
            
            for result in response.results:
                print(f"Score: {result.score}, Content: {result.content}")
            ```
        """
        pass
```

### API Documentation

- **OpenAPI/Swagger**: Auto-generated from Pydantic models
- **Examples**: Include request/response examples
- **Error Codes**: Document all possible error responses

### Architecture Documentation

- **ADRs**: Document significant architectural decisions
- **Diagrams**: Use Mermaid for system diagrams
- **Integration Guides**: How to integrate with external systems

## 🔒 Security Guidelines

### Secure Coding Practices

1. **Input Validation**: Validate all inputs using Pydantic
2. **Authentication**: Proper JWT handling and scope checking
3. **Rate Limiting**: Implement appropriate rate limiting
4. **Logging**: Don't log sensitive information
5. **Dependencies**: Keep dependencies updated

### Security Review Process

Security-sensitive changes require:

1. **Security Review**: Additional review from security-focused team member
2. **Threat Modeling**: Consider potential attack vectors
3. **Penetration Testing**: For authentication/authorization changes

## 🚀 Performance Guidelines

### Performance Considerations

1. **Async/Await**: Use async operations for I/O
2. **Connection Pooling**: Reuse database connections
3. **Caching**: Cache expensive operations when appropriate
4. **Batching**: Use bulk operations for multiple items
5. **Monitoring**: Add metrics for performance tracking

### Performance Testing

```python
@pytest.mark.performance
def test_vector_search_performance(benchmark):
    """Test vector search performance."""
    def search_operation():
        return asyncio.run(vector_service.search_vectors(request))
    
    result = benchmark(search_operation)
    
    # Performance assertions
    assert benchmark.stats.stats.mean < 0.5  # < 500ms
    assert result is not None
```

## 📦 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update Version**: Update version in `pyproject.toml`
2. **Update Changelog**: Document changes in `CHANGELOG.md`
3. **Run Tests**: Ensure all tests pass
4. **Security Scan**: Run security audits
5. **Documentation**: Update documentation if needed
6. **Create Tag**: Create git tag for release
7. **Deploy**: Deploy to staging first, then production

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: Feature development branches
- **hotfix/***: Critical bug fixes
- **release/***: Release preparation branches

## 🏷️ Commit Messages

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(vector): add bulk vector upsert capability

Add batch processing for vector storage to improve performance
when inserting large numbers of documents.

Closes #123

fix(auth): handle expired JWT tokens properly

Previously expired tokens would cause 500 errors instead of
returning proper 401 unauthorized responses.

docs(api): update vector search examples

Add more comprehensive examples showing filter usage
and response handling.
```

## 🤔 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration

### Development Support

- **Documentation**: Check existing documentation first
- **Search Issues**: Look for similar issues or questions
- **Minimal Reproduction**: Provide minimal code to reproduce problems
- **Environment Details**: Include OS, Python version, dependencies

### Code Review Support

- **Be Respectful**: Constructive feedback only
- **Be Specific**: Point to specific lines and suggest improvements
- **Be Helpful**: Explain the reasoning behind suggestions
- **Be Responsive**: Address review feedback promptly

## 🏆 Recognition

### Contributors

We recognize contributions in multiple ways:

- **Contributors List**: Listed in README.md
- **Release Notes**: Significant contributions mentioned
- **GitHub Contributors**: Automatic GitHub recognition

### Types of Contributions

All contributions are valuable:

- **Code**: Features, bug fixes, performance improvements
- **Documentation**: Guides, examples, API documentation
- **Testing**: New tests, test improvements
- **Issues**: Well-researched bug reports and feature requests
- **Reviews**: Code review feedback and suggestions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Thank you for contributing to the Unified MCP Intelligence Server! Your contributions help make this project better for everyone.** 🚀
