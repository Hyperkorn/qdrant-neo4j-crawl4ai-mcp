# Docker Deployment Guide

This guide covers Docker and Docker Compose deployment for the Qdrant Neo4j Crawl4AI MCP Server, suitable for development, testing, and small-scale production deployments.

## Overview

Docker deployment provides a containerized environment with all dependencies managed and isolated. This approach is ideal for:

- **Development**: Local development and testing
- **CI/CD**: Automated testing pipelines
- **Small Production**: Single-node production deployments
- **Edge Deployment**: Resource-constrained environments

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Docker | 24.0+ | 26.0+ |
| Docker Compose | 2.20+ | 2.24+ |
| CPU | 4 cores | 8 cores |
| Memory | 8 GB | 16 GB |
| Storage | 50 GB | 100 GB |

### Installation

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp
```

### 2. Environment Configuration

```bash
# Create environment file
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Start Services

```bash
# Development environment
docker compose up -d

# Production environment
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose logs -f
```

### 4. Verify Deployment

```bash
# Check service health
curl http://localhost:8000/health

# View web interface
open http://localhost:8000/docs
```

## Configuration

### Environment Variables

Create `.env` file with required configuration:

```bash
# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security Configuration
JWT_SECRET_KEY=your-super-secure-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
ADMIN_API_KEY=your-admin-api-key-here

# Database Configuration
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=production-password
NEO4J_DATABASE=neo4j
REDIS_URL=redis://redis:6379/0

# Service Configuration
DEFAULT_COLLECTION=qdrant_neo4j_crawl4ai_intelligence
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CRAWL4AI_MAX_CONCURRENT=10
CRAWL4AI_REQUEST_TIMEOUT=60

# Monitoring
ENABLE_PROMETHEUS=true
ENABLE_SWAGGER_UI=false
ENABLE_REDOC=false

# CORS Configuration
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=Authorization,Content-Type,X-API-Key

# Performance Tuning
NEO4J_MAX_POOL_SIZE=50
CONNECTION_TIMEOUT=30
MAX_RETRIES=3
ENABLE_CACHING=true
```

### Production Docker Compose

The production configuration includes enhanced security, monitoring, and performance optimizations:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Main Application
  qdrant-neo4j-crawl4ai-mcp:
    build:
      context: .
      dockerfile: Dockerfile.prod
      args:
        BUILD_DATE: ${BUILD_DATE}
        VERSION: ${VERSION}
        VCS_REF: ${VCS_REF}
    image: qdrant-neo4j-crawl4ai-mcp:${VERSION:-latest}
    container_name: qdrant-neo4j-crawl4ai-mcp-prod
    restart: unless-stopped
    
    ports:
      - "${PORT:-8000}:8000"
    
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - DEBUG=false
      - HOST=0.0.0.0
      - PORT=8000
      - WORKERS=${WORKERS:-4}
    
    env_file:
      - .env
    
    depends_on:
      qdrant:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      redis:
        condition: service_healthy
    
    volumes:
      - ./logs:/app/logs:rw
      - ./data:/app/data:rw
      - /tmp:/tmp:rw
    
    networks:
      - mcp-network
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    security_opt:
      - no-new-privileges:true
    
    cap_drop:
      - ALL
    
    cap_add:
      - NET_BIND_SERVICE
    
    user: "1000:1000"
    
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: qdrant-prod
    restart: unless-stopped
    
    ports:
      - "6333:6333"
      - "6334:6334"
    
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
      - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32
      - QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
      - QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots
      - QDRANT__STORAGE__MEMORY_THRESHOLD_MB=1024
    
    volumes:
      - qdrant_data:/qdrant/storage
      - qdrant_snapshots:/qdrant/snapshots
    
    networks:
      - mcp-network
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M

  # Neo4j Graph Database
  neo4j:
    image: neo4j:5.15-enterprise
    container_name: neo4j-prod
    restart: unless-stopped
    
    ports:
      - "7474:7474"
      - "7687:7687"
    
    environment:
      - NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_import_file_use__neo4j__config=true
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*
      - NEO4J_dbms_memory_heap_initial__size=1g
      - NEO4J_dbms_memory_heap_max__size=4g
      - NEO4J_dbms_memory_pagecache_size=2g
      - NEO4J_dbms_security_auth_enabled=true
      - NEO4J_dbms_logs_http_enabled=true
      - NEO4J_dbms_logs_query_enabled=INFO
      - NEO4J_metrics_prometheus_enabled=true
      - NEO4J_metrics_prometheus_endpoint=0.0.0.0:2004
      - NEO4J_dbms_connector_http_listen__address=0.0.0.0:7474
      - NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687
    
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_conf:/conf
      - neo4j_plugins:/plugins
      - neo4j_import:/import
    
    networks:
      - mcp-network
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474/"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 6G
        reservations:
          cpus: '0.5'
          memory: 1G

  # Redis Cache
  redis:
    image: redis:7.2-alpine
    container_name: redis-prod
    restart: unless-stopped
    
    ports:
      - "6379:6379"
    
    command: >
      redis-server
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
      --requirepass ${REDIS_PASSWORD:-}
    
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/etc/redis/redis.conf:ro
    
    networks:
      - mcp-network
    
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.1'
          memory: 256M

  # Monitoring Services

  # Prometheus
  prometheus:
    image: prom/prometheus:v2.48.1
    container_name: prometheus-prod
    restart: unless-stopped
    
    ports:
      - "9090:9090"
    
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
      - '--web.enable-admin-api'
    
    volumes:
      - ./monitoring/prometheus/prometheus.prod.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    
    networks:
      - mcp-network
    
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M

  # Grafana
  grafana:
    image: grafana/grafana:10.2.3
    container_name: grafana-prod
    restart: unless-stopped
    
    ports:
      - "3000:3000"
    
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=redis-datasource,grafana-clock-panel
      - GF_SECURITY_DISABLE_GRAVATAR=true
      - GF_ANALYTICS_REPORTING_ENABLED=false
      - GF_ANALYTICS_CHECK_FOR_UPDATES=false
      - GF_SERVER_ROOT_URL=https://grafana.yourdomain.com
    
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - ./monitoring/grafana/grafana.ini:/etc/grafana/grafana.ini:ro
    
    networks:
      - mcp-network
    
    depends_on:
      - prometheus
    
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.1'
          memory: 128M

  # Loki Log Aggregation
  loki:
    image: grafana/loki:2.9.4
    container_name: loki-prod
    restart: unless-stopped
    
    ports:
      - "3100:3100"
    
    command: -config.file=/etc/loki/loki.yml
    
    volumes:
      - ./monitoring/loki/loki.prod.yml:/etc/loki/loki.yml:ro
      - loki_data:/loki
    
    networks:
      - mcp-network
    
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.1'
          memory: 256M

  # Promtail Log Shipper
  promtail:
    image: grafana/promtail:2.9.4
    container_name: promtail-prod
    restart: unless-stopped
    
    volumes:
      - ./monitoring/promtail/promtail.prod.yml:/etc/promtail/config.yml:ro
      - ./logs:/var/log/app:ro
      - /var/log:/var/log/host:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    
    command: -config.file=/etc/promtail/config.yml
    
    networks:
      - mcp-network
    
    depends_on:
      - loki

  # Jaeger Tracing
  jaeger:
    image: jaegertracing/all-in-one:1.51
    container_name: jaeger-prod
    restart: unless-stopped
    
    ports:
      - "16686:16686"
      - "14268:14268"
      - "6831:6831/udp"
    
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=memory
      - MEMORY_MAX_TRACES=50000
    
    networks:
      - mcp-network
    
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.1'
          memory: 256M

  # Reverse Proxy
  nginx:
    image: nginx:1.25-alpine
    container_name: nginx-prod
    restart: unless-stopped
    
    ports:
      - "80:80"
      - "443:443"
    
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
    
    networks:
      - mcp-network
    
    depends_on:
      - qdrant-neo4j-crawl4ai-mcp
    
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
        reservations:
          cpus: '0.1'
          memory: 64M

networks:
  mcp-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  qdrant_data:
    driver: local
  qdrant_snapshots:
    driver: local
  neo4j_data:
    driver: local
  neo4j_logs:
    driver: local
  neo4j_conf:
    driver: local
  neo4j_plugins:
    driver: local
  neo4j_import:
    driver: local
  redis_data:
    driver: local
  grafana_data:
    driver: local
  prometheus_data:
    driver: local
  loki_data:
    driver: local
```

## Production Dockerfile

Create an optimized production Dockerfile:

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

# Build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set build environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast Python package management
RUN pip install uv

# Copy dependency files
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies with UV
RUN uv pip install --system -r pyproject.toml

# Production stage
FROM python:3.11-slim as production

# Build arguments for labels
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add labels for metadata
LABEL org.opencontainers.image.title="Qdrant Neo4j Crawl4AI MCP Server" \
      org.opencontainers.image.description="Production-ready MCP server with vector search, graph database, and web intelligence" \
      org.opencontainers.image.version=${VERSION} \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.revision=${VCS_REF} \
      org.opencontainers.image.source="https://github.com/your-username/qdrant-neo4j-crawl4ai-mcp" \
      org.opencontainers.image.licenses="MIT"

# Set production environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set up application directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/cache /tmp \
    && chown -R appuser:appuser /app /tmp

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "qdrant_neo4j_crawl4ai_mcp.main"]
```

## Management Commands

### Service Management

```bash
# Start all services
docker compose -f docker-compose.prod.yml up -d

# Stop all services
docker compose -f docker-compose.prod.yml down

# Restart specific service
docker compose -f docker-compose.prod.yml restart qdrant-neo4j-crawl4ai-mcp

# View service status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f qdrant-neo4j-crawl4ai-mcp

# Scale application
docker compose -f docker-compose.prod.yml up -d --scale qdrant-neo4j-crawl4ai-mcp=3
```

### Data Management

```bash
# Backup Qdrant data
docker exec qdrant-prod curl -X POST "http://localhost:6333/collections/qdrant_neo4j_crawl4ai_intelligence/snapshots"

# Backup Neo4j data
docker exec neo4j-prod neo4j-admin database dump --to-path=/backups neo4j

# Backup Redis data
docker exec redis-prod redis-cli BGSAVE

# Create full backup
./scripts/backup.sh production
```

### Updates and Maintenance

```bash
# Update images
docker compose -f docker-compose.prod.yml pull

# Rolling update
docker compose -f docker-compose.prod.yml up -d --no-deps qdrant-neo4j-crawl4ai-mcp

# Clean up unused resources
docker system prune -f
docker volume prune -f
```

## Security Hardening

### Docker Security

```yaml
# Additional security configurations
security_opt:
  - no-new-privileges:true
  - seccomp:unconfined

cap_drop:
  - ALL

cap_add:
  - NET_BIND_SERVICE

read_only: true

tmpfs:
  - /tmp
  - /var/tmp

ulimits:
  nofile:
    soft: 65536
    hard: 65536
  memlock:
    soft: -1
    hard: -1
```

### Network Security

```bash
# Create custom network with encryption
docker network create \
  --driver overlay \
  --opt encrypted \
  --subnet 10.0.0.0/16 \
  mcp-secure-network
```

### Secrets Management

```yaml
# Using Docker secrets
services:
  qdrant-neo4j-crawl4ai-mcp:
    secrets:
      - jwt_secret
      - neo4j_password
      - admin_api_key

secrets:
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  neo4j_password:
    file: ./secrets/neo4j_password.txt
  admin_api_key:
    file: ./secrets/admin_api_key.txt
```

## Monitoring Setup

### Grafana Dashboards

Access pre-configured dashboards at `http://localhost:3000`:

- **System Overview**: Resource utilization and health
- **Application Metrics**: Request rates, latency, errors
- **Database Performance**: Qdrant and Neo4j metrics
- **Business Intelligence**: Knowledge graph insights

### Alerting Rules

```yaml
# prometheus-rules.yml
groups:
  - name: qdrant_neo4j_crawl4ai_mcp
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          
      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: High memory usage detected
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   sudo netstat -tlnp | grep :8000
   
   # Change ports in .env
   PORT=8001
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory
   # Add to .env
   NEO4J_dbms_memory_heap_max__size=2g
   ```

3. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data
   sudo chown -R 1000:1000 ./logs
   ```

### Log Analysis

```bash
# View all logs
docker compose logs -f

# Filter by service
docker compose logs -f qdrant-neo4j-crawl4ai-mcp

# Search for errors
docker compose logs | grep ERROR

# Real-time monitoring
watch 'docker compose ps'
```

## Performance Optimization

### Resource Tuning

```bash
# Set Docker resource limits
docker update --memory="2g" --cpus="1.0" qdrant-neo4j-crawl4ai-mcp-prod

# Optimize for production workload
docker compose -f docker-compose.prod.yml up -d --scale qdrant-neo4j-crawl4ai-mcp=3
```

### Database Optimization

```bash
# Qdrant optimization
QDRANT__STORAGE__MEMORY_THRESHOLD_MB=2048
QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=64

# Neo4j optimization
NEO4J_dbms_memory_heap_max__size=4g
NEO4J_dbms_memory_pagecache_size=2g
```

## Next Steps

- **[Kubernetes Deployment](./kubernetes.md)** - Scale to production with Kubernetes
- **[Cloud Deployment](./cloud-providers.md)** - Deploy on AWS, GCP, or Azure
- **[Monitoring Setup](./monitoring.md)** - Advanced observability
- **[Security Hardening](./security-hardening.md)** - Production security

---

This Docker deployment provides a solid foundation for running the Qdrant Neo4j Crawl4AI MCP Server in containerized environments with comprehensive monitoring and security features.