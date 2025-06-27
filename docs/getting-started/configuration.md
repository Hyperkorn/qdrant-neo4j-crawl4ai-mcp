# Configuration Guide

Comprehensive configuration reference for the Qdrant Neo4j Crawl4AI MCP Server.

## 🎯 Configuration Overview

The system uses environment variables for configuration, following the [12-factor app methodology](https://12factor.net/config). All settings can be configured via:

- **Environment variables** (recommended for production)
- **.env files** (convenient for development)
- **Docker Compose environment** (for containerized deployments)
- **Kubernetes ConfigMaps and Secrets** (for K8s deployments)

## 📁 Configuration Files

### Primary Configuration

| File | Purpose | Environment |
|------|---------|-------------|
| `.env` | Main environment configuration | Development |
| `.env.example` | Template with all options | Reference |
| `docker-compose.yml` | Docker environment settings | Development |
| `k8s/manifests/configmap.yaml` | Kubernetes configuration | Production |
| `k8s/manifests/secrets.yaml` | Kubernetes secrets | Production |

## 🔧 Core Configuration

### Application Settings

```env
# Application Environment
ENVIRONMENT=development  # development, staging, production
DEBUG=true              # Enable debug mode (development only)
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json         # json, text

# Server Configuration
HOST=0.0.0.0           # Server bind address
PORT=8000              # Server port
WORKERS=1              # Number of worker processes (production: 4-8)

# Application Metadata
APP_NAME="Qdrant Neo4j Crawl4AI MCP Server"
APP_VERSION=1.0.0
```

### Security Configuration

```env
# JWT Authentication
JWT_SECRET_KEY=your-super-secure-secret-key-minimum-32-characters
JWT_ALGORITHM=HS256     # HS256, RS256
JWT_EXPIRE_MINUTES=1440 # Token expiration (24 hours)

# API Security
API_KEY_HEADER=X-API-Key
ADMIN_API_KEY=your-admin-api-key-for-admin-operations

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100  # Requests per minute per IP
RATE_LIMIT_BURST=20        # Burst allowance

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=Authorization,Content-Type,X-API-Key

# Security Headers
HSTS_MAX_AGE=31536000  # 1 year in seconds
CSP_POLICY="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
```

### Database Configuration

```env
# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                      # Optional API key
QDRANT_GRPC_PORT=6334               # gRPC port (optional)

# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password_123
NEO4J_DATABASE=neo4j
NEO4J_MAX_POOL_SIZE=50
NEO4J_CONNECTION_TIMEOUT=30

# Redis Cache
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=                      # Optional password
REDIS_MAX_CONNECTIONS=20
REDIS_CONNECTION_TIMEOUT=5
```

### Service Configuration

```env
# Vector Service
DEFAULT_COLLECTION=mcp_intelligence
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384  # Depends on embedding model
DISTANCE_METRIC=cosine  # cosine, euclidean, dot

# Graph Service (GraphRAG)
NEO4J_ENABLE_GRAPHRAG=true
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o

# Web Intelligence Service
CRAWL4AI_MAX_CONCURRENT=5
CRAWL4AI_REQUEST_TIMEOUT=30
CRAWL4AI_MAX_RETRIES=3
CRAWL4AI_RETRY_DELAY=1.0
CRAWL4AI_USER_AGENT="QdrantNeo4jCrawl4AIMCP/1.0 (Crawl4AI; +https://github.com/your-repo)"
CRAWL4AI_CHECK_ROBOTS_TXT=true
CRAWL4AI_ENABLE_STEALTH=false
CRAWL4AI_ENABLE_CACHING=true
CRAWL4AI_CACHE_TTL=3600
```

## 🌍 Environment-Specific Configuration

### Development Environment

```env
# .env for development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# Enable development features
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true
ENABLE_PROMETHEUS=true

# Relaxed security for development
JWT_EXPIRE_MINUTES=1440
ALLOWED_ORIGINS=*
RATE_LIMIT_PER_MINUTE=1000

# Local database connections
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=development
REDIS_URL=redis://localhost:6379/0

# Development secrets (not secure!)
JWT_SECRET_KEY=dev-secret-key-not-for-production
ADMIN_API_KEY=dev-admin-key-not-for-production
```

### Production Environment

```env
# .env for production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json

# Disable development features
ENABLE_SWAGGER_UI=false
ENABLE_REDOC=false
ENABLE_PROMETHEUS=true

# Enhanced security
JWT_EXPIRE_MINUTES=60
ALLOWED_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
RATE_LIMIT_PER_MINUTE=100

# Production database connections
QDRANT_URL=http://qdrant-service:6333
NEO4J_URI=bolt://neo4j-service:7687
REDIS_URL=redis://redis-service:6379/0

# Production secrets (use secure generation!)
JWT_SECRET_KEY=<generated-with-openssl-rand-base64-32>
ADMIN_API_KEY=<generated-with-openssl-rand-base64-32>
NEO4J_PASSWORD=<secure-generated-password>
```

### Staging Environment

```env
# .env for staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json

# Limited development features
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true
ENABLE_PROMETHEUS=true

# Moderate security
JWT_EXPIRE_MINUTES=120
ALLOWED_ORIGINS=https://staging.yourdomain.com
RATE_LIMIT_PER_MINUTE=200
```

## 🔒 Security Configuration

### JWT Token Security

```env
# Strong JWT configuration
JWT_SECRET_KEY=$(openssl rand -base64 32)
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60  # Short expiration for security

# For RS256 (asymmetric keys)
JWT_ALGORITHM=RS256
JWT_PRIVATE_KEY_PATH=/path/to/private.key
JWT_PUBLIC_KEY_PATH=/path/to/public.key
```

### API Security

```env
# Rate limiting configuration
RATE_LIMIT_PER_MINUTE=100    # Base rate limit
RATE_LIMIT_BURST=20          # Burst allowance
RATE_LIMIT_WINDOW=60         # Window in seconds

# CORS security
ALLOWED_ORIGINS=https://yourdomain.com  # Specific domains only
ALLOWED_METHODS=GET,POST,PUT,DELETE     # No OPTIONS in production
ALLOWED_HEADERS=Authorization,Content-Type,X-API-Key
ALLOW_CREDENTIALS=true
```

### Database Security

```env
# Qdrant security
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_TLS_ENABLED=true
QDRANT_TLS_VERIFY=true

# Neo4j security
NEO4J_ENCRYPTION=true
NEO4J_TRUST_STRATEGY=TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
NEO4J_USER=neo4j
NEO4J_PASSWORD=$(openssl rand -base64 16)

# Redis security
REDIS_PASSWORD=$(openssl rand -base64 16)
REDIS_TLS_ENABLED=true
```

## 🎛️ Feature Configuration

### Monitoring and Observability

```env
# Prometheus metrics
ENABLE_METRICS=true
METRICS_PORT=9090
METRICS_PATH=/metrics

# Health checks
HEALTH_CHECK_TIMEOUT=5
ENABLE_HEALTH_ENDPOINT=true

# Logging
LOG_FILE=/app/logs/application.log
LOG_ROTATION=true
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5

# Tracing
JAEGER_AGENT_HOST=jaeger-agent
JAEGER_AGENT_PORT=6831
JAEGER_SAMPLER_TYPE=const
JAEGER_SAMPLER_PARAM=0.1  # 10% sampling
```

### Performance Tuning

```env
# Connection pools
NEO4J_MAX_POOL_SIZE=50
NEO4J_CONNECTION_TIMEOUT=30
REDIS_MAX_CONNECTIONS=20
QDRANT_CONNECTION_TIMEOUT=30

# Async configuration
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
CONNECTION_TIMEOUT=30
READ_TIMEOUT=30

# Caching
ENABLE_CACHING=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Web crawling limits
CRAWL4AI_MAX_CONCURRENT=10
CRAWL4AI_REQUEST_TIMEOUT=60
CRAWL4AI_MAX_RETRIES=3
CRAWL4AI_BACKOFF_FACTOR=2.0
```

## 🐳 Docker Configuration

### Docker Compose Environment

```yaml
# docker-compose.override.yml for custom configuration
version: '3.8'

services:
  qdrant-neo4j-crawl4ai-mcp:
    environment:
      # Override any environment variables
      - LOG_LEVEL=DEBUG
      - CRAWL4AI_MAX_CONCURRENT=10
      - JWT_EXPIRE_MINUTES=120
    volumes:
      # Mount custom configuration
      - ./config/custom.env:/app/.env
      - ./logs:/app/logs
    ports:
      # Expose additional ports if needed
      - "9090:9090"  # Metrics port
```

### Production Docker Configuration

```env
# Production environment variables for Docker
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4

# Use Docker internal networking
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret
NEO4J_PASSWORD_FILE=/run/secrets/neo4j_password
ADMIN_API_KEY_FILE=/run/secrets/admin_api_key
```

## ☸️ Kubernetes Configuration

### ConfigMap Example

```yaml
# k8s/manifests/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-config
  namespace: qdrant-neo4j-crawl4ai-mcp
data:
  # Application configuration
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  
  # Service URLs (using Kubernetes DNS)
  QDRANT_URL: "http://qdrant-service:6333"
  NEO4J_URI: "bolt://neo4j-service:7687"
  NEO4J_USER: "neo4j"
  NEO4J_DATABASE: "neo4j"
  REDIS_URL: "redis://redis-service:6379/0"
  
  # Feature flags
  ENABLE_SWAGGER_UI: "false"
  ENABLE_REDOC: "false"
  ENABLE_PROMETHEUS: "true"
  
  # Performance settings
  WORKERS: "4"
  CRAWL4AI_MAX_CONCURRENT: "10"
  NEO4J_MAX_POOL_SIZE: "50"
```

### Secrets Example

```yaml
# k8s/manifests/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-secrets
  namespace: qdrant-neo4j-crawl4ai-mcp
type: Opaque
data:
  # Base64 encoded secrets
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  NEO4J_PASSWORD: <base64-encoded-neo4j-password>
  ADMIN_API_KEY: <base64-encoded-admin-key>
  OPENAI_API_KEY: <base64-encoded-openai-key>
```

Create secrets:
```bash
# Generate and create secrets
kubectl create secret generic qdrant-neo4j-crawl4ai-mcp-secrets \
  --namespace=qdrant-neo4j-crawl4ai-mcp \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -base64 32)" \
  --from-literal=NEO4J_PASSWORD="$(openssl rand -base64 16)" \
  --from-literal=ADMIN_API_KEY="$(openssl rand -base64 32)" \
  --from-literal=OPENAI_API_KEY="your-openai-api-key"
```

## 🔧 Advanced Configuration

### Custom Embedding Models

```env
# Sentence Transformers models
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Fast, 384 dim
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # Better quality, 768 dim
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-distilroberta-v1  # Balanced, 768 dim

# OpenAI models (requires API key)
DEFAULT_EMBEDDING_MODEL=text-embedding-ada-002  # 1536 dim
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small  # 1536 dim
DEFAULT_EMBEDDING_MODEL=text-embedding-3-large  # 3072 dim

# Custom model configuration
EMBEDDING_MODEL_PATH=/app/models/custom-model
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_LENGTH=512
```

### GraphRAG Configuration

```env
# Enable GraphRAG features
NEO4J_ENABLE_GRAPHRAG=true

# OpenAI configuration for GraphRAG
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1

# Graph construction settings
GRAPHRAG_CHUNK_SIZE=1000
GRAPHRAG_CHUNK_OVERLAP=200
GRAPHRAG_MAX_ENTITIES=100
GRAPHRAG_ENTITY_EXTRACTION_PROMPT="Extract entities and relationships..."
```

### Web Crawling Configuration

```env
# Crawl4AI configuration
CRAWL4AI_MAX_CONCURRENT=5
CRAWL4AI_REQUEST_TIMEOUT=30
CRAWL4AI_MAX_RETRIES=3
CRAWL4AI_RETRY_DELAY=1.0
CRAWL4AI_BACKOFF_FACTOR=2.0

# User agent configuration
CRAWL4AI_USER_AGENT="QdrantNeo4jCrawl4AIMCP/1.0 (Educational; +https://github.com/your-repo)"

# Crawling behavior
CRAWL4AI_CHECK_ROBOTS_TXT=true
CRAWL4AI_RESPECT_DELAY=true
CRAWL4AI_MIN_DELAY=1.0
CRAWL4AI_MAX_DELAY=5.0

# Stealth mode (use carefully)
CRAWL4AI_ENABLE_STEALTH=false
CRAWL4AI_STEALTH_CONFIG='{"viewport": {"width": 1920, "height": 1080}}'

# Content filtering
CRAWL4AI_EXCLUDE_SELECTORS=.advertisement,.popup,.cookie-banner
CRAWL4AI_INCLUDE_SELECTORS=article,main,.content
CRAWL4AI_MAX_CONTENT_LENGTH=1000000
```

## 📊 Configuration Validation

### Validation Script

```bash
#!/bin/bash
# validate-config.sh - Validate configuration

set -e

echo "🔍 Validating Qdrant Neo4j Crawl4AI MCP Server configuration..."

# Check required environment variables
REQUIRED_VARS=(
    "ENVIRONMENT"
    "JWT_SECRET_KEY"
    "QDRANT_URL"
    "NEO4J_URI"
    "NEO4J_USER"
    "NEO4J_PASSWORD"
    "REDIS_URL"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Missing required environment variable: $var"
        exit 1
    else
        echo "✅ $var is set"
    fi
done

# Validate JWT secret length
if [ ${#JWT_SECRET_KEY} -lt 32 ]; then
    echo "❌ JWT_SECRET_KEY must be at least 32 characters"
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    echo "❌ ENVIRONMENT must be one of: development, staging, production"
    exit 1
fi

# Test database connections
echo "Testing database connections..."

# Test Qdrant
if curl -s "$QDRANT_URL" > /dev/null; then
    echo "✅ Qdrant connection successful"
else
    echo "⚠️  Qdrant connection failed - check QDRANT_URL"
fi

# Test Redis
REDIS_HOST=$(echo "$REDIS_URL" | sed 's|redis://||' | cut -d':' -f1)
REDIS_PORT=$(echo "$REDIS_URL" | sed 's|redis://||' | cut -d':' -f2 | cut -d'/' -f1)
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
    echo "✅ Redis connection successful"
else
    echo "⚠️  Redis connection failed - check REDIS_URL"
fi

echo "🎉 Configuration validation completed!"
```

Make it executable and run:
```bash
chmod +x validate-config.sh
./validate-config.sh
```

## 🔄 Configuration Management

### Environment-Specific Files

```bash
# Directory structure for multiple environments
config/
├── .env.development
├── .env.staging
├── .env.production
└── .env.local          # Local overrides (git-ignored)

# Load environment-specific configuration
ENV_FILE=".env.${ENVIRONMENT:-development}"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Load local overrides
if [ -f ".env.local" ]; then
    source ".env.local"
fi
```

### Configuration Templates

```bash
# Generate configuration from template
envsubst < config/template.env > .env
```

Template example (`config/template.env`):
```env
# Generated configuration for ${ENVIRONMENT}
ENVIRONMENT=${ENVIRONMENT}
DEBUG=${DEBUG:-false}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
```

## 🚨 Configuration Troubleshooting

### Common Issues

#### Missing Environment Variables
```bash
# Check all environment variables
printenv | grep -E "(QDRANT|NEO4J|REDIS|JWT)"

# Validate .env file loading
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('JWT_SECRET_KEY:', os.getenv('JWT_SECRET_KEY', 'NOT_SET'))
"
```

#### Database Connection Issues
```bash
# Test individual connections
curl -v $QDRANT_URL
redis-cli -u $REDIS_URL ping

# Test Neo4j with cypher-shell
echo "RETURN 1 as test" | cypher-shell -a $NEO4J_URI -u $NEO4J_USER -p $NEO4J_PASSWORD
```

#### Permission Issues
```bash
# Check file permissions
ls -la .env
chmod 600 .env  # Secure permissions for .env file

# Check Docker permissions
id -nG | grep docker
```

### Configuration Debugging

```python
# Debug configuration loading
from qdrant_neo4j_crawl4ai_mcp.config import get_settings

settings = get_settings()
print(f"Environment: {settings.environment}")
print(f"Debug mode: {settings.debug}")
print(f"Qdrant URL: {settings.qdrant_url}")
print(f"Neo4j URI: {settings.neo4j_uri}")
```

## 🎯 Next Steps

After configuring your system:

1. **[Test with first queries](./first-queries.md)** - Verify your configuration works
2. **[Set up monitoring](./troubleshooting.md#monitoring-and-observability)** - Monitor your configured system
3. **[Deploy to production](../DEPLOYMENT_OPERATIONS.md)** - Use your configuration in production
4. **[Backup configuration](./troubleshooting.md#backup-and-recovery)** - Protect your configuration

---

**🔗 Quick Links:**
- [Installation Guide](./installation.md) - Set up the system
- [First Queries](./first-queries.md) - Test your configuration
- [Troubleshooting](./troubleshooting.md) - Solve configuration issues