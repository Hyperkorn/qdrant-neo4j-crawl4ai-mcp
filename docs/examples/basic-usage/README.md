# Basic Usage Examples

This directory contains simple, focused examples that demonstrate the core functionality of each service in the Agentic RAG MCP server. These examples are perfect for getting started and understanding the fundamental capabilities.

## 📋 Examples Overview

### Core Service Examples
- **[vector-operations.py](./vector-operations.py)** - Store and search documents with Qdrant
- **[graph-operations.py](./graph-operations.py)** - Build knowledge graphs with Neo4j
- **[web-intelligence.py](./web-intelligence.py)** - Extract web content with Crawl4AI
- **[authentication.py](./authentication.py)** - Secure your MCP connections

### Simple Applications
- **[document-qa-system/](./document-qa-system/)** - Complete Q&A system tutorial
- **[knowledge-base-builder/](./knowledge-base-builder/)** - Build a knowledge base from web content
- **[semantic-search-app/](./semantic-search-app/)** - Semantic search application

### Getting Started Utilities
- **[mcp-client-setup.py](./mcp-client-setup.py)** - MCP client connection helper
- **[health-check.py](./health-check.py)** - Verify service connectivity
- **[data-setup.py](./data-setup.py)** - Load sample data for testing

## 🚀 Quick Start

### 1. Verify Your Setup

First, ensure all services are running and accessible:

```bash
python health-check.py
```

### 2. Load Sample Data

Load some sample data to work with:

```bash
python data-setup.py
```

### 3. Try Basic Operations

Run the core examples to understand each service:

```bash
# Vector operations
python vector-operations.py

# Graph operations  
python graph-operations.py

# Web intelligence
python web-intelligence.py
```

### 4. Build Your First Application

Follow the complete tutorial to build a document Q&A system:

```bash
cd document-qa-system/
python main.py
```

## 📚 Learning Progression

### Step 1: Individual Services
Start with understanding each service independently:

1. **Vector Operations** - Learn semantic search and document storage
2. **Graph Operations** - Understand knowledge graph construction
3. **Web Intelligence** - Master content extraction and analysis

### Step 2: Service Integration
Combine services for more powerful applications:

1. **Hybrid Search** - Vector + Graph search
2. **Web to Knowledge** - Web content → Graph + Vectors
3. **Multi-Modal Intelligence** - Text + Structure + Web data

### Step 3: Complete Applications
Build full applications that showcase real-world usage:

1. **Document Q&A System** - RAG with vector search
2. **Knowledge Base Builder** - Automated knowledge extraction
3. **Research Assistant** - Multi-service intelligence

## 🛠️ Example Requirements

### Environment Setup
```bash
# Install dependencies
pip install mcp-client httpx asyncio python-dotenv

# Set environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Required Services
- **MCP Server**: Running on `http://localhost:8000` (or configured URL)
- **Qdrant**: Available and accessible
- **Neo4j**: Available and accessible
- **Internet**: For web intelligence examples

### Authentication
Most examples include optional authentication. For testing:

```python
# Get a demo token
import requests

response = requests.post("http://localhost:8000/auth/token", 
                        json={"username": "demo", "scopes": ["read", "write"]})
token = response.json()["access_token"]
```

## 📝 Example Format

Each Python example follows this pattern:

```python
"""
Example: [Name]
Description: [What this example demonstrates]
Services: [Which services are used]
Complexity: [Beginner/Intermediate/Advanced]
"""

import asyncio
from mcp_client import MCPClient

async def main():
    """Main example function."""
    # 1. Setup and connection
    client = MCPClient("http://localhost:8000")
    
    # 2. Example operations
    result = await client.call_tool("tool_name", {"param": "value"})
    
    # 3. Display results
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Success Metrics

After completing these examples, you should be able to:

- ✅ Connect to the MCP server and authenticate
- ✅ Store and search documents using vector similarity
- ✅ Create and query knowledge graphs
- ✅ Extract and analyze web content
- ✅ Combine multiple services in a single workflow
- ✅ Build a complete document Q&A application

## 🔧 Troubleshooting

### Common Issues

**Connection Errors**
```bash
# Check if server is running
curl http://localhost:8000/health

# Verify environment variables
python -c "import os; print(os.getenv('MCP_SERVER_URL'))"
```

**Authentication Errors**
```bash
# Test token generation
python authentication.py
```

**Service Unavailable**
```bash
# Check individual service health
python health-check.py --verbose
```

### Debug Mode

Run examples with debug output:

```bash
python vector-operations.py --debug
```

## 📖 Code Style

All examples follow these conventions:

- **Type hints** for all function parameters and returns
- **Docstrings** explaining purpose and usage
- **Error handling** with meaningful messages
- **Async/await** for all MCP operations
- **Configuration** via environment variables
- **Logging** for debugging and monitoring

## ➡️ Next Steps

Once you're comfortable with these basic examples:

1. **[Advanced Workflows](../advanced-workflows/)** - Multi-service patterns
2. **[Client Implementations](../client-implementations/)** - Full applications
3. **[Use Cases](../use-cases/)** - Real-world scenarios
4. **[Integration Patterns](../integration-patterns/)** - Production patterns

## 🤝 Contributing

Found an issue or want to improve an example?

1. **Report bugs** in the GitHub issues
2. **Suggest improvements** via pull requests
3. **Add new examples** following the established patterns
4. **Update documentation** to help other users

---

**Ready to start?** Begin with [vector-operations.py](./vector-operations.py) to see semantic search in action!