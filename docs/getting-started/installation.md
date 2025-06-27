# Installation Guide

Comprehensive installation instructions for all deployment scenarios of the Qdrant Neo4j Crawl4AI MCP Server.

## 🎯 Choose Your Installation Method

| Method | Best For | Time | Complexity |
|--------|----------|------|------------|
| [Docker Compose](#docker-compose-installation) | Development, Testing | 5 minutes | ⭐ Easy |
| [Local Development](#local-development-installation) | Contribution, Debugging | 10 minutes | ⭐⭐ Medium |
| [Kubernetes](#kubernetes-deployment) | Production, Scaling | 15 minutes | ⭐⭐⭐ Advanced |
| [Cloud Deployment](#cloud-deployment) | Managed Hosting | 20 minutes | ⭐⭐⭐ Advanced |

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **CPU**: 2 cores (4+ recommended)
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 10GB free space (50GB+ for production)
- **Network**: Internet connection for downloading dependencies

### Software Prerequisites
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Python**: 3.11+ (for local development)
- **Git**: Latest version
- **uv**: Package manager (recommended) or pip

### Optional Requirements
- **Kubernetes**: 1.25+ (for K8s deployment)
- **kubectl**: Latest version
- **Helm**: 3.0+ (for easier K8s deployment)

## 🐳 Docker Compose Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# Verify the structure
ls -la
```

### Step 2: Environment Configuration

```bash
# Create environment file from template
cp .env.example .env

# Edit configuration (see Configuration Guide for details)
nano .env
```

**Minimal .env configuration:**
```env
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database URLs (Docker service names)
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password_123
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY=your-super-secure-secret-key-minimum-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Service Configuration
DEFAULT_COLLECTION=mcp_intelligence
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CRAWL4AI_MAX_CONCURRENT=5
CRAWL4AI_REQUEST_TIMEOUT=30

# Feature Flags
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true
ENABLE_PROMETHEUS=true
```

### Step 3: Launch Services

```bash
# Start all services in background
docker-compose up -d

# Monitor startup progress
docker-compose logs -f
```

### Step 4: Verify Installation

```bash
# Check service health
curl http://localhost:8000/health

# Check individual services
curl http://localhost:6333/  # Qdrant
curl http://localhost:7474/  # Neo4j
redis-cli -h localhost ping  # Redis
```

Expected health response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "environment": "development",
  "services": {
    "vector": {"status": "ready"},
    "graph": {"status": "ready"},
    "web": {"status": "ready"}
  }
}
```

### Step 5: Access Web Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | None |
| Neo4j Browser | http://localhost:7474 | neo4j / secure_password_123 |
| Grafana | http://localhost:3000 | admin / development |

## 💻 Local Development Installation

For contributing or debugging the codebase.

### Step 1: System Dependencies

**Ubuntu/Debian:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and system dependencies
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential curl

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for Docker permissions
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 docker docker-compose git

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (WSL2):**
```bash
# Inside WSL2 Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential curl

# Install Docker Desktop for Windows and enable WSL2 integration
```

### Step 2: Clone and Setup

```bash
# Clone repository
git clone https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# Create virtual environment and install dependencies
uv sync --dev

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

### Step 3: Database Setup

```bash
# Start only the databases with Docker
docker-compose up -d qdrant neo4j redis

# Wait for services to be ready
sleep 30

# Verify databases are running
curl http://localhost:6333/  # Qdrant should return version info
curl http://localhost:7474/  # Neo4j browser should load
redis-cli ping               # Should return PONG
```

### Step 4: Environment Configuration

```bash
# Create local environment file
cat << 'EOF' > .env
# Local Development Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Local database connections
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379/0

# Development settings
JWT_SECRET_KEY=dev-local-secret-key-change-in-production
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true

# Service configuration
DEFAULT_COLLECTION=local_development
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EOF
```

### Step 5: Run the Application

```bash
# Run with uv
uv run python -m qdrant_neo4j_crawl4ai_mcp

# Or run directly with Python
python -m qdrant_neo4j_crawl4ai_mcp

# Or use the main module
python src/qdrant_neo4j_crawl4ai_mcp/main.py
```

### Step 6: Development Tools

```bash
# Code formatting
uv run ruff format .

# Linting
uv run ruff check . --fix

# Type checking
uv run mypy .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=qdrant_neo4j_crawl4ai_mcp --cov-report=html
```

## ☸️ Kubernetes Deployment

For production environments with scaling and high availability.

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm (optional but recommended)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
```

### Step 1: Prepare Configuration

```bash
# Clone repository
git clone https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# Navigate to Kubernetes manifests
cd k8s/manifests
```

### Step 2: Create Secrets

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Create production secrets
kubectl create secret generic qdrant-neo4j-crawl4ai-mcp-secrets \
  --namespace=qdrant-neo4j-crawl4ai-mcp \
  --from-literal=jwt-secret-key="$(openssl rand -base64 32)" \
  --from-literal=neo4j-password="$(openssl rand -base64 16)" \
  --from-literal=admin-api-key="$(openssl rand -base64 32)"

# Verify secrets
kubectl get secrets -n qdrant-neo4j-crawl4ai-mcp
```

### Step 3: Deploy Infrastructure

```bash
# Deploy databases first
kubectl apply -f qdrant.yaml

# Wait for Qdrant to be ready
kubectl wait --for=condition=ready pod -l app=qdrant \
  --namespace=qdrant-neo4j-crawl4ai-mcp --timeout=300s

# Deploy Neo4j
kubectl apply -f neo4j.yaml

# Wait for Neo4j to be ready
kubectl wait --for=condition=ready pod -l app=neo4j \
  --namespace=qdrant-neo4j-crawl4ai-mcp --timeout=300s
```

### Step 4: Deploy Application

```bash
# Apply configuration
kubectl apply -f configmap.yaml

# Deploy the main application
kubectl apply -f qdrant-neo4j-crawl4ai-mcp.yaml

# Apply ingress (modify for your domain)
kubectl apply -f ingress.yaml
```

### Step 5: Verify Deployment

```bash
# Check pod status
kubectl get pods -n qdrant-neo4j-crawl4ai-mcp

# Check services
kubectl get services -n qdrant-neo4j-crawl4ai-mcp

# Check ingress
kubectl get ingress -n qdrant-neo4j-crawl4ai-mcp

# View logs
kubectl logs -f deployment/qdrant-neo4j-crawl4ai-mcp \
  -n qdrant-neo4j-crawl4ai-mcp
```

### Step 6: Access the Application

```bash
# Port forward for testing (if no ingress)
kubectl port-forward svc/qdrant-neo4j-crawl4ai-mcp-service 8000:8000 \
  -n qdrant-neo4j-crawl4ai-mcp

# Test health endpoint
curl http://localhost:8000/health
```

## ☁️ Cloud Deployment

### Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy from repository root
cd qdrant-neo4j-crawl4ai-mcp
railway up

# Configure environment variables in Railway dashboard
# Add your production secrets and configurations
```

### Fly.io Deployment

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login and deploy
cd qdrant-neo4j-crawl4ai-mcp
fly auth login
fly launch

# Configure secrets
fly secrets set JWT_SECRET_KEY="$(openssl rand -base64 32)"
fly secrets set NEO4J_PASSWORD="$(openssl rand -base64 16)"

# Deploy
fly deploy
```

### AWS ECS with CDK

```bash
# Prerequisites: AWS CLI and CDK installed
npm install -g aws-cdk

# Deploy infrastructure
cd deployment/aws-cdk
npm install
cdk bootstrap
cdk deploy
```

## 🔧 Post-Installation Configuration

### SSL/TLS Setup

For production deployments, configure SSL certificates:

```bash
# Using Let's Encrypt with cert-manager (Kubernetes)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Configure certificate issuer
cat << 'EOF' | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Monitoring Setup

```bash
# Deploy monitoring stack (if not using Docker Compose)
cd monitoring/

# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Jaeger: http://localhost:16686"
```

### Backup Configuration

```bash
# Create backup scripts directory
mkdir -p scripts/backup

# Qdrant backup script
cat << 'EOF' > scripts/backup/backup-qdrant.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant/$DATE"
mkdir -p "$BACKUP_DIR"

# Create Qdrant snapshot
curl -X POST "http://qdrant:6333/collections/snapshot" \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "your_collection"}'

# Copy snapshot to backup directory
# Implementation depends on your storage setup
EOF

chmod +x scripts/backup/backup-qdrant.sh
```

## ✅ Installation Verification

### Complete System Test

```bash
# Run the installation verification script
cat << 'EOF' > verify-installation.sh
#!/bin/bash
set -e

echo "🔍 Verifying Qdrant Neo4j Crawl4AI MCP Server Installation..."

# Check main application
echo "Testing main application..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$response" = "200" ]; then
    echo "✅ Main application is healthy"
else
    echo "❌ Main application failed (HTTP $response)"
    exit 1
fi

# Check Qdrant
echo "Testing Qdrant..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/)
if [ "$response" = "200" ]; then
    echo "✅ Qdrant is responding"
else
    echo "❌ Qdrant failed (HTTP $response)"
    exit 1
fi

# Check Neo4j
echo "Testing Neo4j..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7474/)
if [ "$response" = "200" ]; then
    echo "✅ Neo4j is responding"
else
    echo "❌ Neo4j failed (HTTP $response)"
    exit 1
fi

# Check Redis
echo "Testing Redis..."
if redis-cli -h localhost ping | grep -q PONG; then
    echo "✅ Redis is responding"
else
    echo "❌ Redis failed"
    exit 1
fi

# Test API functionality
echo "Testing API functionality..."
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "test_user", "scopes": ["read"]}' | \
  python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

if [ -n "$TOKEN" ]; then
    echo "✅ Token generation working"
    
    # Test vector search
    response=$(curl -s -X POST "http://localhost:8000/api/v1/vector/search" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"query": "test", "limit": 1}' | \
      python3 -c "import sys, json; print(json.load(sys.stdin).get('source', ''))")
    
    if [ "$response" = "vector" ]; then
        echo "✅ Vector search working"
    else
        echo "⚠️  Vector search may need initialization"
    fi
else
    echo "❌ Token generation failed"
    exit 1
fi

echo "🎉 Installation verification completed successfully!"
echo ""
echo "Next steps:"
echo "1. Visit http://localhost:8000/docs for API documentation"
echo "2. Visit http://localhost:7474 for Neo4j browser (neo4j/password)"
echo "3. Visit http://localhost:3000 for Grafana dashboard (admin/development)"
echo "4. Read the First Queries guide: docs/getting-started/first-queries.md"
EOF

chmod +x verify-installation.sh
./verify-installation.sh
```

## 🚨 Troubleshooting Installation

### Common Issues

#### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in

# Or run with sudo temporarily
sudo docker-compose up -d
```

#### Port Conflicts
```bash
# Check what's using ports
sudo netstat -tulpn | grep :8000
sudo netstat -tulpn | grep :6333
sudo netstat -tulpn | grep :7474
sudo netstat -tulpn | grep :7687

# Kill conflicting processes or change ports in docker-compose.yml
```

#### Memory Issues
```bash
# Check Docker memory usage
docker stats

# Increase Docker memory allocation (Docker Desktop)
# Or reduce services for testing:
docker-compose up -d qdrant neo4j redis qdrant-neo4j-crawl4ai-mcp
```

#### Database Connection Issues
```bash
# Check container logs
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis

# Restart databases
docker-compose restart qdrant neo4j redis
```

### Getting Help

If you encounter issues:

1. **Check [Troubleshooting Guide](./troubleshooting.md)** for detailed solutions
2. **Review logs**: `docker-compose logs -f`
3. **Verify prerequisites**: Ensure Docker, Python, and Git are properly installed
4. **Check system resources**: Ensure adequate memory and disk space
5. **Test network connectivity**: Ensure no firewall blocking required ports

## 🎯 Next Steps

After successful installation:

1. **[Configure your system](./configuration.md)** - Customize settings for your environment
2. **[Try your first queries](./first-queries.md)** - Learn how to use the system
3. **[Set up monitoring](./troubleshooting.md#monitoring-and-observability)** - Keep your system healthy
4. **[Deploy to production](../DEPLOYMENT_OPERATIONS.md)** - Scale for production use

---

**🔗 Quick Links:**
- [Configuration Guide](./configuration.md) - Customize your installation
- [First Queries](./first-queries.md) - Start using the system
- [Troubleshooting](./troubleshooting.md) - Solve common problems