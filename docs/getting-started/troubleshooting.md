# Troubleshooting Guide

Comprehensive troubleshooting guide for the Qdrant Neo4j Crawl4AI MCP Server.

## 🎯 Quick Diagnosis

### System Health Check

```bash
#!/bin/bash
# quick-health-check.sh - Run this first for immediate diagnosis

echo "🔍 Running Quick Health Check..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running or accessible"
    echo "💡 Solution: Start Docker or check Docker permissions"
    exit 1
fi

# Check main application
echo "Testing main application..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
case $response in
    200) echo "✅ Main application is healthy" ;;
    000) echo "❌ Main application is not responding (not started?)" ;;
    *) echo "⚠️  Main application returned HTTP $response" ;;
esac

# Check databases
echo "Testing Qdrant..."
qdrant_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/ 2>/dev/null)
case $qdrant_response in
    200) echo "✅ Qdrant is responding" ;;
    000) echo "❌ Qdrant is not responding" ;;
    *) echo "⚠️  Qdrant returned HTTP $qdrant_response" ;;
esac

echo "Testing Neo4j..."
neo4j_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7474/ 2>/dev/null)
case $neo4j_response in
    200) echo "✅ Neo4j is responding" ;;
    000) echo "❌ Neo4j is not responding" ;;
    *) echo "⚠️  Neo4j returned HTTP $neo4j_response" ;;
esac

echo "Testing Redis..."
if command -v redis-cli > /dev/null && redis-cli -h localhost ping > /dev/null 2>&1; then
    echo "✅ Redis is responding"
else
    echo "❌ Redis is not responding"
fi

# Check Docker containers
echo -e "\n📦 Docker Container Status:"
docker-compose ps 2>/dev/null || echo "❌ Docker Compose not found in current directory"

echo -e "\n🔗 Service URLs:"
echo "Main App: http://localhost:8000/health"
echo "API Docs: http://localhost:8000/docs"
echo "Neo4j: http://localhost:7474"
echo "Grafana: http://localhost:3000"
```

Make it executable and run:

```bash
chmod +x quick-health-check.sh
./quick-health-check.sh
```

## 🚨 Common Issues & Solutions

### 1. Services Not Starting

#### Issue: Docker Compose fails to start services

**Symptoms:**

```bash
docker-compose up -d
# Returns errors or services show as "unhealthy"
```

**Diagnosis:**

```bash
# Check logs for specific errors
docker-compose logs

# Check individual service logs
docker-compose logs qdrant-neo4j-crawl4ai-mcp
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis
```

**Solutions:**

**Port Conflicts:**

```bash
# Check what's using the ports
sudo netstat -tulpn | grep :8000
sudo netstat -tulpn | grep :6333
sudo netstat -tulpn | grep :7474

# Solution 1: Kill conflicting processes
sudo kill -9 $(sudo lsof -t -i:8000)

# Solution 2: Change ports in docker-compose.yml
# Edit docker-compose.yml and change port mappings
# Example: "8080:8000" instead of "8000:8000"
```

**Memory Issues:**

```bash
# Check Docker memory usage
docker stats

# Check system memory
free -h

# Solution: Increase Docker memory or reduce services
# Start only essential services:
docker-compose up -d qdrant neo4j redis qdrant-neo4j-crawl4ai-mcp
```

**Permission Issues:**

```bash
# Docker permission denied
sudo usermod -aG docker $USER
# Log out and back in, or use newgrp docker

# Temporary solution:
sudo docker-compose up -d
```

### 2. Authentication Issues

#### Issue: 401 Unauthorized errors

**Symptoms:**

```bash
curl http://localhost:8000/api/v1/profile
# {"error": "Could not validate credentials", "status_code": 401}
```

**Solutions:**

**Get a new token:**

```bash
# Generate new token
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "test_user", "scopes": ["read", "write"]}' | \
  jq -r '.access_token')

echo "Token: $TOKEN"

# Test with token
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/profile"
```

**Check JWT configuration:**

```bash
# Verify JWT secret is set
docker-compose exec qdrant-neo4j-crawl4ai-mcp env | grep JWT_SECRET_KEY

# If missing, add to .env file:
echo "JWT_SECRET_KEY=$(openssl rand -base64 32)" >> .env
docker-compose restart qdrant-neo4j-crawl4ai-mcp
```

### 3. Database Connection Issues

#### Issue: Vector search returns "service unavailable"

**Symptoms:**

```bash
curl -X POST "http://localhost:8000/api/v1/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "test"}'
# {"error": "Vector service is not available", "status_code": 503}
```

**Diagnosis:**

```bash
# Check Qdrant directly
curl http://localhost:6333/
curl http://localhost:6333/collections

# Check connection from app container
docker-compose exec qdrant-neo4j-crawl4ai-mcp curl http://qdrant:6333/
```

**Solutions:**

**Qdrant not ready:**

```bash
# Wait for Qdrant to fully start
docker-compose logs qdrant | grep "Qdrant HTTP listening"

# Restart Qdrant if needed
docker-compose restart qdrant

# Check Qdrant health
curl http://localhost:6333/health
```

**Network issues:**

```bash
# Check Docker network
docker network ls
docker network inspect qdrant-neo4j-crawl4ai-mcp_mcp-network

# Verify service names resolve
docker-compose exec qdrant-neo4j-crawl4ai-mcp nslookup qdrant
```

#### Issue: Neo4j connection failures

**Diagnosis:**

```bash
# Test Neo4j connectivity
docker-compose exec qdrant-neo4j-crawl4ai-mcp nc -zv neo4j 7687

# Check Neo4j logs
docker-compose logs neo4j | grep -i error

# Test with cypher-shell
echo "RETURN 1 as test" | \
  docker-compose exec -T neo4j cypher-shell -u neo4j -p development
```

**Solutions:**

**Authentication issues:**

```bash
# Reset Neo4j password
docker-compose stop neo4j
docker-compose run --rm neo4j neo4j-admin set-initial-password newpassword
docker-compose start neo4j

# Update .env file
sed -i 's/NEO4J_PASSWORD=.*/NEO4J_PASSWORD=newpassword/' .env
docker-compose restart qdrant-neo4j-crawl4ai-mcp
```

**Memory issues:**

```bash
# Check Neo4j memory settings in docker-compose.yml
# Reduce memory if needed:
# NEO4J_dbms_memory_heap_max__size=1g
# NEO4J_dbms_memory_pagecache_size=512m
```

### 4. Performance Issues

#### Issue: Slow response times or timeouts

**Diagnosis:**

```bash
# Check response times
time curl http://localhost:8000/health

# Monitor system resources
htop
docker stats

# Check application logs for slow queries
docker-compose logs qdrant-neo4j-crawl4ai-mcp | grep -i "slow\|timeout"
```

**Solutions:**

**Resource optimization:**

```bash
# Update .env for better performance
cat << 'EOF' >> .env
# Performance tuning
WORKERS=4
CRAWL4AI_MAX_CONCURRENT=3
NEO4J_MAX_POOL_SIZE=20
RATE_LIMIT_PER_MINUTE=50
EOF

docker-compose restart qdrant-neo4j-crawl4ai-mcp
```

**Database optimization:**

```bash
# Optimize Qdrant
curl -X POST "http://localhost:6333/collections/your_collection/index" \
  -H "Content-Type: application/json" \
  -d '{"wait": true}'

# Optimize Neo4j (run in Neo4j browser)
# CREATE CONSTRAINT ON (n:Node) ASSERT n.id IS UNIQUE;
# CREATE INDEX ON :Node(category);
```

### 5. Web Crawling Issues

#### Issue: Crawl4AI timeouts or failures

**Symptoms:**

```bash
# Crawling requests timeout or return errors
curl -X POST "http://localhost:8000/api/v1/web/crawl" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "https://example.com"}'
# Timeout or 500 error
```

**Solutions:**

**Increase timeouts:**

```bash
# Update .env
cat << 'EOF' >> .env
CRAWL4AI_REQUEST_TIMEOUT=60
CRAWL4AI_MAX_RETRIES=5
CRAWL4AI_RETRY_DELAY=2.0
EOF

docker-compose restart qdrant-neo4j-crawl4ai-mcp
```

**Network issues:**

```bash
# Test network connectivity from container
docker-compose exec qdrant-neo4j-crawl4ai-mcp curl -I https://example.com

# Check DNS resolution
docker-compose exec qdrant-neo4j-crawl4ai-mcp nslookup google.com
```

## 🔍 Diagnostic Tools

### Log Analysis

```bash
# Comprehensive log analysis script
#!/bin/bash
# analyze-logs.sh

echo "📊 Analyzing Qdrant Neo4j Crawl4AI MCP Server logs..."

# Get recent application logs
echo -e "\n🔍 Recent Application Logs (last 50 lines):"
docker-compose logs --tail=50 qdrant-neo4j-crawl4ai-mcp

# Check for errors
echo -e "\n❌ Error Patterns:"
docker-compose logs qdrant-neo4j-crawl4ai-mcp | grep -i "error\|exception\|failed" | tail -10

# Check authentication issues
echo -e "\n🔐 Authentication Issues:"
docker-compose logs qdrant-neo4j-crawl4ai-mcp | grep -i "401\|unauthorized\|authentication" | tail -5

# Check database connections
echo -e "\n🗄️ Database Connection Issues:"
docker-compose logs qdrant-neo4j-crawl4ai-mcp | grep -i "connection.*failed\|unable to connect" | tail -5

# Performance issues
echo -e "\n⚡ Performance Issues:"
docker-compose logs qdrant-neo4j-crawl4ai-mcp | grep -i "timeout\|slow\|performance" | tail -5

# Check specific service logs
echo -e "\n📦 Service-Specific Logs:"
echo "Qdrant errors:"
docker-compose logs qdrant | grep -i error | tail -3

echo "Neo4j errors:"
docker-compose logs neo4j | grep -i error | tail -3

echo "Redis errors:"
docker-compose logs redis | grep -i error | tail -3
```

### Performance Monitoring

```bash
# performance-monitor.sh
#!/bin/bash

echo "📈 Performance Monitoring Dashboard"

# System resources
echo -e "\n💻 System Resources:"
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo "Memory Usage:"
free -h | awk 'NR==2{printf "%.1f%%\n", $3*100/$2 }'

echo "Disk Usage:"
df -h / | awk 'NR==2{print $5}'

# Docker resources
echo -e "\n🐳 Docker Container Resources:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Application metrics
echo -e "\n📊 Application Metrics:"
if curl -s http://localhost:8000/metrics > /dev/null; then
    echo "✅ Metrics endpoint accessible"
    curl -s http://localhost:8000/metrics | grep -E "http_requests_total|response_time" | head -5
else
    echo "❌ Metrics endpoint not accessible"
fi

# Database performance
echo -e "\n🗄️ Database Performance:"
# Qdrant collections info
if curl -s http://localhost:6333/collections > /dev/null; then
    echo "Qdrant collections:"
    curl -s http://localhost:6333/collections | jq -r '.result.collections[].name' 2>/dev/null || echo "Unable to parse collections"
fi

# Response time test
echo -e "\n⏱️ Response Time Test:"
echo "Health endpoint:"
time curl -s http://localhost:8000/health > /dev/null

echo "API endpoint (with auth):"
if [ -n "$TOKEN" ]; then
    time curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/profile > /dev/null
else
    echo "No token available - run: export TOKEN=your_token_here"
fi
```

### Network Connectivity Testing

```bash
# network-test.sh
#!/bin/bash

echo "🌐 Network Connectivity Testing"

# Test internal Docker networking
echo -e "\n🐳 Docker Network Tests:"
docker-compose exec qdrant-neo4j-crawl4ai-mcp nc -zv qdrant 6333
docker-compose exec qdrant-neo4j-crawl4ai-mcp nc -zv neo4j 7687
docker-compose exec qdrant-neo4j-crawl4ai-mcp nc -zv redis 6379

# Test external connectivity
echo -e "\n🌍 External Connectivity:"
docker-compose exec qdrant-neo4j-crawl4ai-mcp curl -s -I https://httpbin.org/get | head -1

# DNS resolution
echo -e "\n🔍 DNS Resolution:"
docker-compose exec qdrant-neo4j-crawl4ai-mcp nslookup google.com | grep "Name\|Address"

# Port accessibility from host
echo -e "\n🚪 Port Accessibility from Host:"
for port in 8000 6333 7474 7687 6379; do
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ Port $port is accessible"
    else
        echo "❌ Port $port is not accessible"
    fi
done
```

## 🖥️ Monitoring and Observability

### Setting Up Monitoring

The system includes comprehensive monitoring out of the box:

```bash
# Start monitoring stack
docker-compose up -d prometheus grafana loki promtail jaeger

# Wait for services to start
sleep 30

# Import dashboards
curl -X POST \
  http://admin:development@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/qdrant-neo4j-crawl4ai-mcp-overview.json
```

### Accessing Monitoring Tools

| Tool | URL | Credentials | Purpose |
|------|-----|-------------|---------|
| **Grafana** | <http://localhost:3000> | admin/development | Dashboards and alerts |
| **Prometheus** | <http://localhost:9090> | None | Metrics collection |
| **Jaeger** | <http://localhost:16686> | None | Distributed tracing |
| **Loki** | <http://localhost:3100> | None | Log aggregation |

### Key Metrics to Monitor

```bash
# Application metrics
curl -s http://localhost:8000/metrics | grep -E "http_requests_total|response_time_seconds|vector_search_duration"

# Qdrant metrics
curl -s http://localhost:6333/metrics

# System metrics via Prometheus
curl -s http://localhost:9090/api/v1/query?query=up
```

### Setting Up Alerts

```yaml
# alerts.yml for Prometheus
groups:
  - name: qdrant-neo4j-crawl4ai-mcp
    rules:
      - alert: ServiceDown
        expr: up{job="qdrant-neo4j-crawl4ai-mcp"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MCP Server is down"
          
      - alert: HighResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

## 🔧 Advanced Troubleshooting

### Debug Mode

```bash
# Enable debug mode
echo "DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose restart qdrant-neo4j-crawl4ai-mcp

# Watch debug logs
docker-compose logs -f qdrant-neo4j-crawl4ai-mcp
```

### Memory Debugging

```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check for memory leaks
docker-compose exec qdrant-neo4j-crawl4ai-mcp ps aux | sort -k4 -nr | head -10

# Adjust memory limits if needed
# Edit docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 1G
```

### Database Debugging

```bash
# Qdrant debugging
# Check collection status
curl http://localhost:6333/collections/your_collection

# Check cluster info
curl http://localhost:6333/cluster

# Neo4j debugging
# Connect to Neo4j browser: http://localhost:7474
# Run diagnostic queries:
# CALL db.info() YIELD *;
# CALL dbms.components() YIELD *;
```

### Application State Debugging

```python
# debug_state.py - Debug application state
import asyncio
import httpx

async def debug_application_state():
    """Debug application internal state"""
    
    # Test authentication
    print("🔐 Testing Authentication...")
    auth_response = await httpx.AsyncClient().post(
        "http://localhost:8000/auth/token",
        json={"username": "debug_user", "scopes": ["admin"]}
    )
    
    if auth_response.status_code == 200:
        token = auth_response.json()["access_token"]
        print("✅ Authentication working")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test admin stats
        print("\n📊 Checking Admin Stats...")
        stats_response = await httpx.AsyncClient().get(
            "http://localhost:8000/api/v1/admin/stats",
            headers=headers
        )
        
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"Uptime: {stats.get('uptime_seconds', 'unknown')} seconds")
            print(f"Environment: {stats.get('environment', 'unknown')}")
            print("Services:", stats.get('services', {}))
        else:
            print(f"❌ Admin stats failed: {stats_response.status_code}")
            
        # Test vector service
        print("\n🔍 Testing Vector Service...")
        vector_health = await httpx.AsyncClient().get(
            "http://localhost:8000/api/v1/vector/health",
            headers=headers
        )
        print(f"Vector service status: {vector_health.status_code}")
        if vector_health.status_code == 200:
            print("Vector health:", vector_health.json().get('status'))
            
    else:
        print(f"❌ Authentication failed: {auth_response.status_code}")
        print("Response:", auth_response.text)

# Run: python debug_state.py
if __name__ == "__main__":
    asyncio.run(debug_application_state())
```

## 🔄 Backup and Recovery

### Backup Procedures

```bash
# backup.sh - Comprehensive backup script
#!/bin/bash

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🗄️ Starting backup to $BACKUP_DIR..."

# Backup Qdrant data
echo "Backing up Qdrant..."
docker-compose exec qdrant qdrant-backup create /qdrant/storage/backup
docker cp $(docker-compose ps -q qdrant):/qdrant/storage/backup "$BACKUP_DIR/qdrant"

# Backup Neo4j data
echo "Backing up Neo4j..."
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j-backup.dump
docker cp $(docker-compose ps -q neo4j):/tmp/neo4j-backup.dump "$BACKUP_DIR/"

# Backup Redis data
echo "Backing up Redis..."
docker-compose exec redis redis-cli --rdb /tmp/redis-backup.rdb
docker cp $(docker-compose ps -q redis):/tmp/redis-backup.rdb "$BACKUP_DIR/"

# Backup configuration
echo "Backing up configuration..."
cp .env "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"

echo "✅ Backup completed: $BACKUP_DIR"
```

### Recovery Procedures

```bash
# restore.sh - Recovery script
#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

BACKUP_DIR="$1"

echo "🔄 Restoring from $BACKUP_DIR..."

# Stop services
docker-compose down

# Restore Qdrant
if [ -d "$BACKUP_DIR/qdrant" ]; then
    echo "Restoring Qdrant..."
    docker volume rm qdrant-neo4j-crawl4ai-mcp_qdrant_data
    docker volume create qdrant-neo4j-crawl4ai-mcp_qdrant_data
    # Copy backup data to volume
fi

# Restore Neo4j
if [ -f "$BACKUP_DIR/neo4j-backup.dump" ]; then
    echo "Restoring Neo4j..."
    docker-compose up -d neo4j
    sleep 30
    docker cp "$BACKUP_DIR/neo4j-backup.dump" $(docker-compose ps -q neo4j):/tmp/
    docker-compose exec neo4j neo4j-admin load --from=/tmp/neo4j-backup.dump --database=neo4j --force
fi

# Restore configuration
if [ -f "$BACKUP_DIR/.env" ]; then
    echo "Restoring configuration..."
    cp "$BACKUP_DIR/.env" .
fi

# Start all services
docker-compose up -d

echo "✅ Recovery completed"
```

## 📞 Getting Help

### Diagnostic Information Collection

```bash
# collect-diagnostics.sh - Collect system info for support
#!/bin/bash

DIAG_DIR="diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAG_DIR"

echo "🔍 Collecting diagnostic information..."

# System information
uname -a > "$DIAG_DIR/system_info.txt"
docker version > "$DIAG_DIR/docker_version.txt"
docker-compose version > "$DIAG_DIR/docker_compose_version.txt"

# Configuration
cp .env "$DIAG_DIR/" 2>/dev/null || echo "No .env file found"
cp docker-compose.yml "$DIAG_DIR/"

# Service status
docker-compose ps > "$DIAG_DIR/service_status.txt"
docker stats --no-stream > "$DIAG_DIR/resource_usage.txt"

# Logs (last 100 lines each)
docker-compose logs --tail=100 qdrant-neo4j-crawl4ai-mcp > "$DIAG_DIR/app_logs.txt"
docker-compose logs --tail=100 qdrant > "$DIAG_DIR/qdrant_logs.txt"
docker-compose logs --tail=100 neo4j > "$DIAG_DIR/neo4j_logs.txt"
docker-compose logs --tail=100 redis > "$DIAG_DIR/redis_logs.txt"

# Network information
docker network ls > "$DIAG_DIR/networks.txt"
docker network inspect qdrant-neo4j-crawl4ai-mcp_mcp-network > "$DIAG_DIR/network_details.txt" 2>/dev/null

# Health checks
curl -s http://localhost:8000/health > "$DIAG_DIR/health_check.json" 2>/dev/null || echo "Health check failed" > "$DIAG_DIR/health_check.json"

echo "📦 Diagnostic information collected in: $DIAG_DIR"
echo "📧 You can share this directory for support"

# Create archive
tar -czf "$DIAG_DIR.tar.gz" "$DIAG_DIR"
echo "📦 Archive created: $DIAG_DIR.tar.gz"
```

### Support Checklist

Before seeking help, ensure you have:

- [ ] **Run the quick health check** script
- [ ] **Checked the logs** for error messages
- [ ] **Verified system requirements** are met
- [ ] **Tested with default configuration**
- [ ] **Collected diagnostic information**
- [ ] **Documented the exact error** messages and steps to reproduce

### Common Solutions Summary

| Issue | Quick Solution |
|-------|----------------|
| **Services won't start** | `docker-compose down && docker-compose up -d` |
| **Authentication fails** | Generate new token with `/auth/token` endpoint |
| **Port conflicts** | Check with `netstat -tulpn \| grep :8000` |
| **Memory issues** | Reduce services or increase Docker memory |
| **Database connection** | Restart individual services: `docker-compose restart qdrant` |
| **Slow performance** | Check resource usage with `docker stats` |
| **Permission errors** | Add user to docker group: `sudo usermod -aG docker $USER` |

## 🎯 Prevention

### Regular Maintenance

```bash
# maintenance.sh - Regular maintenance tasks
#!/bin/bash

echo "🔧 Running maintenance tasks..."

# Update Docker images
docker-compose pull

# Clean up unused Docker resources
docker system prune -f

# Restart services for fresh state
docker-compose restart

# Check disk usage
df -h

# Verify all services are healthy
./quick-health-check.sh

echo "✅ Maintenance completed"
```

### Monitoring Setup

Set up regular monitoring:

```bash
# Add to crontab (crontab -e)
# Check health every 5 minutes
*/5 * * * * /path/to/quick-health-check.sh >> /var/log/mcp-health.log 2>&1

# Daily maintenance
0 2 * * * /path/to/maintenance.sh >> /var/log/mcp-maintenance.log 2>&1

# Weekly backup
0 3 * * 0 /path/to/backup.sh >> /var/log/mcp-backup.log 2>&1
```

---

## 🎉 Success

You now have comprehensive troubleshooting tools and knowledge to keep your Qdrant Neo4j Crawl4AI MCP Server running smoothly.

**🔗 Quick Links:**

- [Configuration Guide](./configuration.md) - Optimize your setup
- [First Queries Guide](./first-queries.md) - Test functionality
- [Installation Guide](./installation.md) - Reinstall if needed

**📞 Still need help?**

- Check the [examples directory](../examples/) for working configurations
- Review the [API documentation](../API_REFERENCE.md) for endpoint details
- Consult the [technical documentation](../TECHNICAL_DOCUMENTATION.md) for architecture details
