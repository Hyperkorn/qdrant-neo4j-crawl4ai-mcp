# Examples and Tutorials

Welcome to the comprehensive examples and tutorials for the Agentic RAG MCP server! This directory contains practical code examples, tutorials, and real-world use cases that demonstrate how to leverage the power of Qdrant vector search, Neo4j knowledge graphs, and Crawl4AI web intelligence through a unified Model Context Protocol (MCP) interface.

## 🚀 Quick Start

If you're new to the Agentic RAG MCP server, start here:

1. **[Basic Usage](./basic-usage/)** - Simple examples to get you started
2. **[First Tutorial: Document Q&A System](./basic-usage/document-qa-system/)** - Build your first application
3. **[Client Setup Guide](./client-implementations/)** - Connect your applications

## 📚 Example Categories

### 🌟 [Basic Usage](./basic-usage/)
Simple examples and getting started guides for each service:
- **Vector Operations** - Store and search documents with semantic similarity
- **Graph Operations** - Build and query knowledge graphs
- **Web Intelligence** - Extract and analyze web content
- **Authentication** - Secure your MCP connections

### 🔧 [Advanced Workflows](./advanced-workflows/)
Complex multi-service scenarios and advanced patterns:
- **Multi-Agent Research** - Coordinate multiple AI agents for research tasks
- **Real-time Knowledge Graphs** - Build dynamic knowledge graphs from live data
- **Hybrid Search Systems** - Combine vector and graph search for better results
- **Content Pipeline** - Automated content extraction and knowledge building

### 💻 [Client Implementations](./client-implementations/)
Complete application examples in different languages:
- **Python Applications** - FastAPI, Django, and Jupyter notebook examples
- **TypeScript/JavaScript** - Next.js, Express.js, and Node.js examples
- **CLI Tools** - Command-line interfaces for automation
- **Integration Patterns** - Common architectural patterns

### 🎯 [Use Cases](./use-cases/)
Real-world application scenarios:
- **Research Assistant** - Academic and business research automation
- **Content Management** - Document analysis and organization
- **Customer Support** - Intelligent knowledge base systems
- **Business Intelligence** - Data analysis and insight generation

### 🔗 [Integration Patterns](./integration-patterns/)
Common integration approaches and patterns:
- **LangChain Integration** - Connect with LangChain pipelines
- **AutoGen Integration** - Multi-agent frameworks
- **Workflow Orchestration** - Temporal, Prefect, and Airflow
- **API Gateway Patterns** - Production deployment patterns

## 🛠️ Prerequisites

Before running the examples, ensure you have:

### Required Services
- **Qdrant** - Vector database (local or cloud)
- **Neo4j** - Graph database (local or cloud)
- **MCP Server** - Running instance of the agentic RAG server

### Development Environment
```bash
# Install dependencies
uv install qdrant-neo4j-crawl4ai-mcp[dev]

# Or using pip
pip install qdrant-neo4j-crawl4ai-mcp[dev]
```

### Environment Variables
```bash
# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key

# Graph Database  
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Web Intelligence
CRAWL4AI_MAX_CONCURRENT=5

# Authentication (for examples)
JWT_SECRET_KEY=your-secret-key
```

## 🎓 Learning Path

### Beginners
1. Start with [Basic Vector Operations](./basic-usage/vector-operations.py)
2. Try [Simple Graph Queries](./basic-usage/graph-operations.py)
3. Explore [Web Content Extraction](./basic-usage/web-intelligence.py)
4. Build your [First Q&A System](./basic-usage/document-qa-system/)

### Intermediate
1. [Multi-Service Workflows](./advanced-workflows/hybrid-search.py)
2. [Custom MCP Clients](./client-implementations/python-client/)
3. [Real-time Data Pipelines](./advanced-workflows/content-pipeline/)
4. [Graph-Enhanced RAG](./advanced-workflows/graph-rag.py)

### Advanced
1. [Multi-Agent Research Systems](./advanced-workflows/multi-agent-research/)
2. [Production Deployment](./integration-patterns/production-deployment/)
3. [Custom Tool Development](./advanced-workflows/custom-tools/)
4. [Performance Optimization](./integration-patterns/performance-optimization/)

## 📖 Example Structure

Each example follows a consistent structure:

```
example-name/
├── README.md              # Overview and instructions
├── requirements.txt       # Additional dependencies
├── .env.example          # Environment variables template
├── main.py               # Primary example code
├── config.py             # Configuration management
├── models.py             # Data models (if needed)
├── tests/                # Example tests
│   └── test_example.py
└── docs/                 # Additional documentation
    └── architecture.md
```

## 🔍 Finding Examples

### By Technology
- **Vector Search**: `grep -r "semantic_vector_search" .`
- **Graph Queries**: `grep -r "create_graph_node" .`  
- **Web Crawling**: `grep -r "crawl_website" .`
- **Multi-Service**: `grep -r "vector.*graph" .`

### By Use Case
- **Document Processing**: `./use-cases/content-management/`
- **Research Automation**: `./use-cases/research-assistant/`
- **Customer Support**: `./use-cases/customer-support/`
- **Business Intelligence**: `./use-cases/business-intelligence/`

### By Complexity
- **Simple** (< 50 lines): `./basic-usage/`
- **Medium** (50-200 lines): `./advanced-workflows/`
- **Complex** (200+ lines): `./client-implementations/`

## 🤝 Contributing Examples

We welcome contributions! To add a new example:

1. **Fork the repository**
2. **Create your example** following the structure above
3. **Add comprehensive documentation**
4. **Include tests** and example data
5. **Update this README** with links to your example
6. **Submit a pull request**

### Example Guidelines
- **Clear documentation** with step-by-step instructions
- **Realistic use cases** that solve real problems
- **Error handling** and edge case coverage
- **Performance considerations** and optimization tips
- **Security best practices** for production use

## 📊 Example Metrics

| Category | Examples | Languages | Difficulty |
|----------|----------|-----------|------------|
| Basic Usage | 12 | Python, TypeScript | Beginner |
| Advanced Workflows | 8 | Python, TypeScript | Intermediate |
| Client Implementations | 6 | Python, TypeScript, JavaScript | Advanced |
| Use Cases | 10 | Python, TypeScript | Mixed |
| Integration Patterns | 4 | Python, TypeScript | Advanced |

## 🆘 Getting Help

- **Documentation**: [Main docs](../README.md)
- **API Reference**: [API docs](../API_REFERENCE.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/qdrant-neo4j-crawl4ai-mcp/issues)
- **Community**: [Discussions](https://github.com/your-username/qdrant-neo4j-crawl4ai-mcp/discussions)

## 📝 License

All examples are provided under the same MIT license as the main project. See [LICENSE](../../LICENSE) for details.

---

**Ready to build something amazing?** Start with our [Quick Start Guide](./basic-usage/README.md) or jump into a specific [use case](./use-cases/) that interests you!