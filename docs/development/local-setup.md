# Local Development Setup

This guide will help you set up a complete local development environment for the Unified MCP Intelligence Server.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Node.js**: 18+ (for pre-commit hooks and tooling)
- **Docker**: 20.10+ and Docker Compose v2
- **Git**: 2.30+
- **uv**: Fast Python package manager

### Operating System

- **Linux**: Ubuntu 20.04+, Debian 11+, or equivalent
- **macOS**: 11.0+ (Big Sur)
- **Windows**: WSL2 with Ubuntu 20.04+

### Hardware Recommendations

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **CPU**: 4+ cores recommended for parallel testing

## 🛠️ Installation Steps

### 1. Install Python and uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
python3.11 --version
```

### 2. Clone Repository

```bash
git clone <repository-url>
cd qdrant-neo4j-crawl4ai-mcp

# Verify you're on the main branch
git branch -a
```

### 3. Environment Setup

```bash
# Create Python virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies including development tools
uv install --extra dev --extra test --extra security

# Verify installation
uv pip list | grep -E "(fastmcp|qdrant|neo4j|crawl4ai)"
```

### 4. Environment Configuration

Create environment files for local development:

```bash
# Copy example environment file
cp .env.example .env.local

# Edit with your configuration
vim .env.local
```

**Example `.env.local`:**

```bash
# Server Configuration
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Security
JWT_SECRET_KEY=your-development-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
DEFAULT_COLLECTION=dev_documents
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=devpassword
NEO4J_DATABASE=neo4j

# Crawl4AI Configuration
CRAWL4AI_MAX_CONCURRENT=5
CRAWL4AI_REQUEST_TIMEOUT=30
CRAWL4AI_ENABLE_STEALTH=false

# Development Settings
LOG_LEVEL=DEBUG
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true
```

### 5. Docker Services Setup

Start the required databases and services:

```bash
# Start development services in background
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs if needed
docker-compose logs qdrant
docker-compose logs neo4j
```

**Services Started:**
- **Qdrant**: Vector database on port 6333
- **Neo4j**: Graph database on ports 7474 (HTTP) and 7687 (Bolt)
- **Prometheus**: Metrics collection on port 9090
- **Grafana**: Dashboards on port 3000

### 6. Database Initialization

```bash
# Initialize Qdrant collections
uv run python -c "
from qdrant_neo4j_crawl4ai_mcp.services.vector_service import VectorService
from qdrant_neo4j_crawl4ai_mcp.models.vector_models import VectorServiceConfig
import asyncio

async def init_qdrant():
    config = VectorServiceConfig(
        qdrant_url='http://localhost:6333',
        default_collection='dev_documents'
    )
    service = VectorService(config)
    await service.initialize()
    print('Qdrant initialized successfully')

asyncio.run(init_qdrant())
"

# Initialize Neo4j constraints and indexes
uv run python -c "
from qdrant_neo4j_crawl4ai_mcp.services.graph_service import GraphService
from qdrant_neo4j_crawl4ai_mcp.models.graph_models import Neo4jServiceConfig
import asyncio

async def init_neo4j():
    config = Neo4jServiceConfig(
        uri='bolt://localhost:7687',
        username='neo4j',
        password='devpassword'
    )
    service = GraphService(config)
    await service.initialize()
    print('Neo4j initialized successfully')

asyncio.run(init_neo4j())
"
```

### 7. Development Tools Setup

```bash
# Install pre-commit hooks
pre-commit install

# Verify pre-commit setup
pre-commit run --all-files

# Install development shell completions (optional)
uv generate-shell-completion bash >> ~/.bashrc
# or for zsh:
# uv generate-shell-completion zsh >> ~/.zshrc
```

## 🧪 Verify Installation

### 1. Run Test Suite

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest -m "not slow" -v

# Run with coverage
pytest --cov=qdrant_neo4j_crawl4ai_mcp --cov-report=html
```

### 2. Start Development Server

```bash
# Start the MCP server
uv run qdrant-neo4j-crawl4ai-mcp

# Or with auto-reload for development
uvicorn qdrant_neo4j_crawl4ai_mcp.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 3. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get demo token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "developer", "scopes": ["read", "write"]}' | \
  jq -r '.access_token')

# Test authenticated endpoint
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/profile
```

### 4. Access Web Interfaces

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Neo4j Browser**: http://localhost:7474 (neo4j/devpassword)
- **Qdrant Web UI**: http://localhost:6333/dashboard
- **Grafana**: http://localhost:3000 (admin/admin)

## 🔧 Development Workflow

### Code Quality Tools

```bash
# Format code with ruff
ruff format .

# Lint and fix issues
ruff check . --fix

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Run all quality checks
make quality  # or manually run each tool
```

### Database Management

```bash
# Reset Qdrant collections
docker-compose exec qdrant sh -c "rm -rf /qdrant/storage/*"
docker-compose restart qdrant

# Reset Neo4j database
docker-compose exec neo4j cypher-shell -u neo4j -p devpassword "MATCH (n) DETACH DELETE n"

# View database logs
docker-compose logs -f qdrant
docker-compose logs -f neo4j
```

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|-----------|
| `ENVIRONMENT` | Deployment environment | `development` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `HOST` | Server host | `0.0.0.0` | No |
| `PORT` | Server port | `8000` | No |
| `QDRANT_URL` | Qdrant connection URL | `http://localhost:6333` | Yes |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` | Yes |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes |
| `JWT_SECRET_KEY` | JWT signing key | - | Yes |

## 🐛 Troubleshooting

### Common Issues

#### Docker Services Won't Start

```bash
# Check Docker daemon
sudo systemctl status docker

# Check port conflicts
sudo netstat -tlnp | grep -E "(6333|7687|7474)"

# Reset Docker environment
docker-compose down --volumes
docker-compose up -d
```

#### Python Dependencies Issues

```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv/
uv venv
uv install --extra dev

# Check for conflicting packages
uv pip check
```

#### Database Connection Issues

```bash
# Test Qdrant connection
curl http://localhost:6333/collections

# Test Neo4j connection
docker-compose exec neo4j cypher-shell -u neo4j -p devpassword "RETURN 1 as test"

# Check firewall settings
sudo ufw status
```

#### Permission Issues

```bash
# Fix file permissions
chmod +x scripts/*.sh

# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Monitor database performance
docker-compose exec qdrant cat /qdrant/config/production.yaml
docker-compose exec neo4j neo4j-admin memrec
```

### Logs and Debugging

```bash
# Application logs
tail -f logs/app.log

# Docker service logs
docker-compose logs -f --tail=100

# System resource monitoring
htop
iotop
```

## 🔄 Regular Maintenance

### Weekly Tasks

```bash
# Update dependencies
uv sync --upgrade

# Update pre-commit hooks
pre-commit autoupdate

# Clean up Docker
docker system prune -f
```

### Monthly Tasks

```bash
# Backup development data
docker-compose exec qdrant tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage/
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j-backup.dump

# Security audit
safety check
bandit -r src/ --format json -o security-report.json
```

## 📚 Next Steps

After completing the local setup:

1. Read the [Contributing Guidelines](contributing.md)
2. Review the [Testing Guidelines](testing.md)
3. Explore the [Code Style Guide](code-style.md)
4. Check out the [API Reference](../API_REFERENCE.md)

---

**Need help?** Check the [troubleshooting section](#-troubleshooting) or create an issue on GitHub.