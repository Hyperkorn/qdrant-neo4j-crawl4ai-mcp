# Quick Start Guide

Get the Qdrant Neo4j Crawl4AI MCP Server running in **5 minutes** with Docker Compose.

## 🎯 What You'll Achieve

- ✅ Full agentic RAG system running locally
- ✅ Vector search with Qdrant
- ✅ Knowledge graphs with Neo4j
- ✅ Web intelligence with Crawl4AI
- ✅ Interactive API documentation
- ✅ Monitoring dashboards

## 🚀 Quick Setup

### Step 1: Clone and Navigate

```bash
# Clone the repository
git clone https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp
```

### Step 2: Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Quick configuration (optional - defaults work for development)
cat << 'EOF' > .env
# Quick Start Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database Configuration (using Docker service names)
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=development
REDIS_URL=redis://redis:6379/0

# Security (development only - change for production)
JWT_SECRET_KEY=dev-secret-key-change-in-production
JWT_ALGORITHM=HS256

# Service Configuration
DEFAULT_COLLECTION=quick_start_intelligence
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CRAWL4AI_MAX_CONCURRENT=5

# Enable Documentation
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true
EOF
```

### Step 3: Launch the System

```bash
# Start all services with Docker Compose
docker-compose up -d

# Check services are starting
docker-compose ps
```

Expected output:

```
NAME                         COMMAND                  SERVICE                      STATUS
grafana-dev                  "/run.sh"                grafana                      Up
jaeger-dev                   "/go/bin/all-in-one-…"   jaeger                       Up
loki-dev                     "/usr/bin/loki -conf…"   loki                         Up
neo4j-dev                    "tini -g -- /startup…"   neo4j                        Up (healthy)
nginx-dev                    "/docker-entrypoint.…"   nginx                        Up
prometheus-dev               "/bin/prometheus --c…"   prometheus                   Up
promtail-dev                 "/usr/bin/promtail -…"   promtail                     Up
qdrant-dev                   "./entrypoint.sh qdra…"  qdrant                       Up (healthy)
qdrant-neo4j-crawl4ai-mcp-dev "/app/entrypoint.sh …"   qdrant-neo4j-crawl4ai-mcp    Up (healthy)
redis-dev                    "docker-entrypoint.s…"   redis                        Up (healthy)
```

### Step 4: Verify the Installation

```bash
# Check main application health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-01-15T10:30:00Z",
#   "version": "1.0.0",
#   "environment": "development",
#   "services": {
#     "vector": {"status": "ready", "last_check": "..."},
#     "graph": {"status": "ready", "last_check": "..."},
#     "web": {"status": "ready", "last_check": "..."}
#   }
# }
```

## 🎉 You're Ready

Your agentic RAG system is now running! Here's what's available:

### 🌐 Web Interfaces

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | <http://localhost:8000/docs> | Interactive Swagger UI |
| **Alternative Docs** | <http://localhost:8000/redoc> | ReDoc documentation |
| **Health Status** | <http://localhost:8000/health> | System health check |
| **Neo4j Browser** | <http://localhost:7474> | Graph database interface |
| **Grafana Dashboards** | <http://localhost:3000> | Monitoring (admin/development) |
| **Prometheus Metrics** | <http://localhost:9090> | Metrics collection |
| **Jaeger Tracing** | <http://localhost:16686> | Distributed tracing |

### 🔑 Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| Neo4j | `neo4j` | `development` |
| Grafana | `admin` | `development` |

## 🚀 First Test

Let's make your first API call:

### 1. Get an Access Token

```bash
# Create a demo token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "scopes": ["read", "write"]
  }'
```

Response:

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "scopes": ["read", "write"]
}
```

### 2. Test Vector Search

```bash
# Save your token
TOKEN="your_access_token_here"

# Test vector search
curl -X POST "http://localhost:8000/api/v1/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence machine learning",
    "limit": 5
  }'
```

### 3. Store Your First Document

```bash
# Store a document in the vector database
curl -X POST "http://localhost:8000/api/v1/vector/store" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence and machine learning are transforming how we process information and make decisions.",
    "content_type": "text",
    "source": "quick_start_demo",
    "tags": ["ai", "ml", "demo"],
    "metadata": {
      "category": "technology",
      "author": "quick_start_user"
    }
  }'
```

## 🎯 What's Working

After the quick start, you have:

### ✅ Vector Intelligence

- **Qdrant** vector database running on port 6333
- **Semantic search** capabilities ready
- **Document storage** and retrieval working
- **Embedding generation** with sentence-transformers

### ✅ Graph Intelligence  

- **Neo4j** graph database running on port 7687
- **Knowledge graph** construction ready
- **Cypher query** capabilities
- **Memory systems** for conversational AI

### ✅ Web Intelligence

- **Crawl4AI** web crawling service
- **Content extraction** capabilities
- **Real-time data** processing
- **Respectful crawling** with rate limits

### ✅ Production Features

- **JWT authentication** with token-based access
- **Rate limiting** protection
- **Health monitoring** endpoints
- **Structured logging** with JSON output
- **Metrics collection** via Prometheus
- **Distributed tracing** with Jaeger

## 🔧 Quick Configuration

The system is pre-configured for development, but you can quickly customize:

### Change Default Collection

```bash
# Edit .env file
echo "DEFAULT_COLLECTION=my_custom_collection" >> .env

# Restart the service
docker-compose restart qdrant-neo4j-crawl4ai-mcp
```

### Enable Production Mode

```bash
# Update .env for production-like settings
cat << 'EOF' >> .env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ENABLE_SWAGGER_UI=false
ENABLE_REDOC=false
JWT_EXPIRE_MINUTES=60
EOF

# Restart services
docker-compose restart
```

### Add Custom Embedding Model

```bash
# Use a different embedding model
echo "DEFAULT_EMBEDDING_MODEL=all-mpnet-base-v2" >> .env
docker-compose restart qdrant-neo4j-crawl4ai-mcp
```

## 🚀 Next Steps

Now that your system is running:

1. **[Learn the APIs](./first-queries.md)** - Detailed examples of all capabilities
2. **[Configure for your needs](./configuration.md)** - Customize settings and security
3. **[Deploy to production](./installation.md#production-deployment)** - Kubernetes and cloud deployment
4. **[Monitor and troubleshoot](./troubleshooting.md)** - Keep your system healthy

## 🛑 Stopping the System

When you're done:

```bash
# Stop all services
docker-compose down

# Stop and remove all data (WARNING: This deletes all stored data)
docker-compose down -v
```

## 🆘 Quick Troubleshooting

### Services Not Starting?

```bash
# Check logs for issues
docker-compose logs qdrant-neo4j-crawl4ai-mcp

# Check individual service health
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis
```

### Port Conflicts?

If ports 8000, 6333, 7474, or 7687 are in use:

```bash
# Edit docker-compose.yml to use different ports
# For example, change 8000:8000 to 8080:8000
```

### Memory Issues?

```bash
# Check Docker resource allocation
docker stats

# Reduce services for lower memory usage
docker-compose up -d qdrant neo4j redis qdrant-neo4j-crawl4ai-mcp
```

### API Returning Errors?

```bash
# Check service health
curl http://localhost:8000/health

# Check logs for specific errors
docker-compose logs -f qdrant-neo4j-crawl4ai-mcp
```

## 🎉 Success

You now have a fully functional agentic RAG system! The quick start gives you:

- **Vector database** for semantic search
- **Graph database** for relationship analysis  
- **Web intelligence** for real-time information
- **Production-ready** authentication and monitoring
- **Interactive documentation** for easy API exploration

**Ready for more?** Continue with the [First Queries Guide](./first-queries.md) to explore the full capabilities of your new intelligent system.

---

**🔗 Quick Links:**

- [First Queries Guide](./first-queries.md) - Learn all the capabilities
- [Configuration Guide](./configuration.md) - Customize your setup
- [Troubleshooting](./troubleshooting.md) - Get help when needed
