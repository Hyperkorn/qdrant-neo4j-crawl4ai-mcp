# Qdrant Neo4j Crawl4AI MCP Server

[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp)
[![FastMCP 2.0](https://img.shields.io/badge/FastMCP-2.0-blue)](https://github.com/jlowin/fastmcp)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-ready agentic RAG MCP server combining Qdrant vector search, Neo4j knowledge graphs, and Crawl4AI web intelligence with autonomous orchestration capabilities**

## 🎯 What is This?

This is an **Agentic RAG (Retrieval-Augmented Generation) MCP Server** that provides intelligent, autonomous coordination of multiple AI services through a single Model Context Protocol interface. It combines:

- **Vector Intelligence**: Semantic search and embedding storage via Qdrant
- **Graph Intelligence**: Knowledge graphs and memory systems via Neo4j  
- **Web Intelligence**: Smart web crawling and content extraction via Crawl4AI
- **Agentic Orchestration**: Autonomous query routing and result fusion
- **Production-Ready**: Enterprise security, monitoring, and deployment patterns

## 🏗️ Architecture

```mermaid
graph TB
    Client[AI Assistant Client] --> Gateway[FastMCP Gateway]
    
    subgraph "Qdrant Neo4j Crawl4AI MCP Server"
        Gateway --> Router[Request Router]
        Router --> Vector[Vector Service]
        Router --> Graph[Graph Service] 
        Router --> Web[Web Intelligence Service]
        
        Vector --> |mount: /vector| QdrantMCP[Qdrant MCP Server]
        Graph --> |mount: /graph| Neo4jMCP[Neo4j Memory MCP]
        Web --> |mount: /web| Crawl4AIMCP[Crawl4AI MCP Server]
    end
    
    subgraph "Data Layer"
        QdrantMCP --> QdrantDB[(Qdrant Vector DB)]
        Neo4jMCP --> Neo4jDB[(Neo4j Graph DB)]
        Crawl4AIMCP --> WebSources[Web Data Sources]
    end
```

## ⚡ Technology Stack

- **FastMCP 2.0**: Server composition and MCP protocol handling
- **Python 3.11+**: Modern async patterns and type safety
- **Qdrant**: Vector database for semantic search
- **Neo4j**: Graph database for knowledge representation
- **Crawl4AI**: Web intelligence and content extraction
- **Docker**: Containerized deployment with health checks

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip
- Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run with Docker
docker-compose up -d

# Or run locally
uv run python -m qdrant_neo4j_crawl4ai_mcp
```

### Configuration

Key environment variables:

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

## 💻 Development

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=qdrant_neo4j_crawl4ai_mcp --cov-report=html

# Run specific test suite
uv run pytest tests/integration/
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Type checking
uv run mypy .
```

## 📚 API Documentation

Once running, access the interactive API documentation at:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>

### Example Usage

```python
import asyncio
from qdrant_neo4j_crawl4ai_mcp.client import QdrantNeo4jCrawl4AIMCPClient

async def main():
    client = QdrantNeo4jCrawl4AIMCPClient("http://localhost:8000")
    
    # Vector search
    results = await client.vector_search("artificial intelligence")
    
    # Graph query
    memories = await client.graph_query("MATCH (n:Memory) RETURN n LIMIT 10")
    
    # Web crawling
    content = await client.web_crawl("https://example.com")

asyncio.run(main())
```

## 📦 Deployment

### Docker Deployment

```bash
# Production build
docker build -t qdrant-neo4j-crawl4ai-mcp .
docker run -p 8000:8000 qdrant-neo4j-crawl4ai-mcp
```

### Cloud Deployment

- **Railway**: One-click deployment via railway.app
- **Fly.io**: Global edge deployment
- **AWS**: ECS/Lambda deployment with CDK

## 📚 Complete Documentation

### 🚀 Getting Started
- **[📖 Documentation Hub](docs/README.md)** - Complete navigation guide  
- **[⚡ Quick Start](docs/getting-started/quick-start.md)** - 5-minute setup  
- **[🔧 Installation Guide](docs/getting-started/installation.md)** - Detailed setup  
- **[⚙️ Configuration](docs/getting-started/configuration.md)** - Environment setup  
- **[🎯 First Queries](docs/getting-started/first-queries.md)** - Learn the system  

### 📖 User Guides
- **[🔍 Vector Search Guide](docs/guides/semantic-search.md)** - Semantic similarity search  
- **[🕸️ Knowledge Graph Guide](docs/guides/knowledge-graphs.md)** - Graph reasoning  
- **[🌐 Web Intelligence Guide](docs/guides/web-intelligence.md)** - Real-time web data  
- **[🤖 Agentic Workflows](docs/guides/agentic-workflows.md)** - Multi-modal intelligence  

### 🔧 Technical Reference
- **[📋 API Reference](docs/API_REFERENCE.md)** - Complete REST API docs  
- **[🏗️ Architecture](docs/ARCHITECTURE.md)** - System design overview  
- **[🔒 Security Guide](docs/guides/security-hardening.md)** - Enterprise security  
- **[📊 Monitoring Setup](docs/guides/monitoring-observability.md)** - Production monitoring  

### 🚢 Deployment & Operations
- **[🚀 Deployment Operations](docs/DEPLOYMENT_OPERATIONS.md)** - Production deployment  
- **[☸️ Kubernetes Guide](docs/deployment/kubernetes.md)** - Container orchestration  
- **[🐳 Docker Guide](docs/deployment/docker.md)** - Containerized deployment  
- **[☁️ Cloud Platforms](docs/deployment/cloud-platforms.md)** - Railway, Fly.io, etc.  

### 💻 Development & Contributing
- **[👨‍💻 Developer Guide](docs/DEVELOPER_GUIDE.md)** - Complete dev workflow  
- **[🧪 Testing Framework](docs/development/testing.md)** - Unit & integration tests  
- **[🎨 Contributing Guidelines](docs/development/contributing.md)** - How to contribute  
- **[🔧 Local Development](docs/development/local-setup.md)** - Dev environment setup  

### 📝 Examples & Tutorials
- **[📚 Examples Hub](docs/examples/README.md)** - Code examples & tutorials  
- **[🔰 Basic Usage](docs/examples/basic-usage/README.md)** - Simple queries  
- **[🚀 Advanced Workflows](docs/examples/advanced-workflows/README.md)** - Complex patterns  
- **[📱 Client SDKs](docs/examples/client-implementations/README.md)** - Multiple languages  

For detailed deployment guides, see **[🚢 Deployment Operations](docs/DEPLOYMENT_OPERATIONS.md)**.

## 🔒 Security & Compliance

- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Rate Limiting**: Redis-backed distributed request throttling
- **OWASP Compliance**: Following API security best practices and security headers
- **Input Validation**: Comprehensive Pydantic-based request sanitization
- **Audit Logging**: Security event tracking with structured logging
- **Enterprise Security**: [Complete security hardening guide](docs/guides/security-hardening.md)

## 📊 Monitoring & Observability

- **Health Checks**: Multi-layer `/health` endpoints with dependency validation
- **Structured Logging**: JSON logs with correlation IDs and context
- **Prometheus Metrics**: Custom business and infrastructure metrics
- **Grafana Dashboards**: Pre-built dashboards for monitoring
- **Error Tracking**: Sentry integration for error reporting
- **Distributed Tracing**: Request flow visualization across services

**Setup Guide**: [📊 Monitoring & Observability](docs/guides/monitoring-observability.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/development/contributing.md) for details.

### Quick Start for Contributors

```bash
# 1. Fork and clone the repository
git clone https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# 2. Set up development environment
uv sync --dev
uv run pre-commit install

# 3. Run tests to verify setup
uv run pytest

# 4. Start development server
docker-compose up -d
uv run python -m qdrant_neo4j_crawl4ai_mcp
```

**Detailed Setup**: [💻 Developer Guide](docs/DEVELOPER_GUIDE.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Project Goals

This project demonstrates:

- **Modern Python Patterns**: Async programming, type safety, and current ecosystem tools
- **AI/ML Integration**: Vector databases, knowledge graphs, and web intelligence
- **Production Engineering**: Security, monitoring, testing, and deployment automation
- **Clean Architecture**: Composable services with clear abstractions
- **DevOps Excellence**: Container orchestration, CI/CD, and infrastructure as code

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [linkedin.com/in/yourprofile]
- **Portfolio**: [yourportfolio.com]

---

> Built with ☕ using FastMCP 2.0, Qdrant, Neo4j, and Web Intelligence
