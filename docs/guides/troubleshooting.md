# Advanced Troubleshooting Guide

[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp)
[![Support Available](https://img.shields.io/badge/support-available-blue)](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/issues)

> **Comprehensive troubleshooting, diagnostic techniques, and problem resolution for production agentic RAG MCP server deployments**

## 📖 Table of Contents

- [Quick Diagnostic Steps](#-quick-diagnostic-steps)
- [Common Issues & Solutions](#-common-issues--solutions)
- [Performance Issues](#-performance-issues)
- [Database Connection Problems](#-database-connection-problems)
- [Authentication & Security Issues](#-authentication--security-issues)
- [Monitoring & Observability](#-monitoring--observability)
- [Advanced Diagnostics](#-advanced-diagnostics)
- [Recovery Procedures](#-recovery-procedures)
- [Getting Help](#-getting-help)

## 🩺 Quick Diagnostic Steps

### 1. System Health Check

```bash
# Check overall system health
curl -s http://localhost:8000/health | jq .

# Expected healthy response:
{
  "status": "healthy",
  "timestamp": "2025-06-27T12:00:00Z",
  "checks": {
    "qdrant": {"status": "healthy", "response_time": 0.05},
    "neo4j": {"status": "healthy", "response_time": 0.02},
    "redis": {"status": "healthy", "response_time": 0.01}
  }
}
```

### 2. Check Service Status

```bash
# Docker Compose
docker-compose ps

# Kubernetes
kubectl get pods -n mcp-server
kubectl get services -n mcp-server

# Check logs
docker-compose logs mcp-server
kubectl logs -n mcp-server deployment/qdrant-neo4j-crawl4ai-mcp
```

### 3. Verify Connectivity

```bash
# Test individual services
curl -s http://localhost:6333/health  # Qdrant
curl -s http://localhost:7474/        # Neo4j Browser
redis-cli ping                        # Redis

# Test MCP endpoints
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -X POST http://localhost:8000/vector/search \
     -d '{"query": "test", "limit": 5}'
```

## 🔧 Common Issues & Solutions

### Issue: Server Won't Start

**Symptoms:**
- Container exits immediately
- "Connection refused" errors
- Port binding failures

**Diagnostic Steps:**

```bash
# Check port conflicts
netstat -tulpn | grep :8000
lsof -i :8000

# Check Docker logs
docker-compose logs mcp-server

# Common error patterns to look for:
# - "Port already in use"
# - "Permission denied"
# - "Environment variable not set"
```

**Solutions:**

```yaml
# 1. Port conflicts - Update docker-compose.yml
services:
  mcp-server:
    ports:
      - "8001:8000"  # Use different external port

# 2. Permission issues - Fix user permissions
services:
  mcp-server:
    user: "${UID}:${GID}"
    
# 3. Environment variables - Check .env file
JWT_SECRET_KEY=your-secret-key-here
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
```

### Issue: Database Connection Failures

**Symptoms:**
- 500 errors on API calls
- "Connection timeout" in logs
- Health check failures

**Diagnostic Steps:**

```bash
# Test direct database connections
# Qdrant
curl -s http://localhost:6333/collections

# Neo4j
echo "RETURN 1" | cypher-shell -u neo4j -p password

# Check network connectivity in Docker
docker network ls
docker network inspect qdrant-neo4j-crawl4ai-mcp_default
```

**Solutions:**

```python
# 1. Connection pool configuration
from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://qdrant:6333",
    timeout=30,           # Increase timeout
    pool_size=20,         # Increase pool size
    retries=3             # Add retries
)

# 2. Health check implementation
async def check_database_health():
    try:
        # Test Qdrant
        await qdrant_client.get_collections()
        
        # Test Neo4j
        async with neo4j_driver.session() as session:
            await session.run("RETURN 1")
            
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
```

### Issue: Authentication Problems

**Symptoms:**
- 401 Unauthorized errors
- "Invalid token" messages
- JWT decode failures

**Diagnostic Steps:**

```bash
# Verify JWT token
export TOKEN="your-jwt-token"
echo $TOKEN | cut -d. -f2 | base64 -d | jq .

# Check token expiration
python3 -c "
import jwt
import json
token = 'your-jwt-token'
payload = jwt.decode(token, options={'verify_signature': False})
print(json.dumps(payload, indent=2))
"
```

**Solutions:**

```python
# 1. Token validation fix
from jose import JWTError, jwt
from datetime import datetime

def verify_token(token: str):
    try:
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=["HS256"],
            options={"verify_exp": True}  # Ensure expiration check
        )
        
        # Additional validation
        if payload.get("exp", 0) < datetime.utcnow().timestamp():
            raise JWTError("Token expired")
            
        return payload
    except JWTError as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

# 2. Token refresh mechanism
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
        
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
```

## 🚀 Performance Issues

### Issue: Slow Query Response Times

**Symptoms:**
- API responses > 2 seconds
- High CPU usage
- Memory consumption growing

**Diagnostic Steps:**

```bash
# Monitor resource usage
docker stats

# Check query performance
curl -w "@curl-format.txt" \
     -H "Authorization: Bearer $TOKEN" \
     -X POST http://localhost:8000/vector/search \
     -d '{"query": "test", "limit": 10}'

# curl-format.txt content:
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
```

**Solutions:**

```python
# 1. Implement caching
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute and cache
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator

# 2. Connection pooling optimization
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="http://qdrant:6333",
    timeout=30,
    pool_size=50,          # Increase pool size
    retries=3,
    retry_delay=1.0
)

# 3. Query optimization
async def optimized_vector_search(query: str, limit: int = 10):
    """Optimized vector search with proper indexing"""
    
    # Use query filters to reduce search space
    search_params = {
        "vector": await get_query_embedding(query),
        "limit": limit,
        "params": {
            "hnsw_ef": 128,    # Increase search accuracy
            "exact": False      # Use approximate search for speed
        }
    }
    
    return await qdrant_client.search(**search_params)
```

### Issue: Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of Memory (OOM) kills
- Container restarts

**Diagnostic Steps:**

```bash
# Monitor memory usage over time
docker stats --no-stream | grep mcp-server

# Check for memory leaks in Python
pip install memory-profiler
python -m memory_profiler your_script.py

# Kubernetes memory monitoring
kubectl top pods -n mcp-server
kubectl describe pod -n mcp-server <pod-name>
```

**Solutions:**

```python
# 1. Proper connection management
import asyncio
from contextlib import asynccontextmanager

class ConnectionManager:
    def __init__(self):
        self._connections = {}
        
    @asynccontextmanager
    async def get_connection(self, service: str):
        """Ensure connections are properly closed"""
        connection = None
        try:
            connection = await self._get_or_create_connection(service)
            yield connection
        finally:
            if connection:
                await self._return_connection(service, connection)
    
    async def cleanup(self):
        """Cleanup all connections"""
        for service, connections in self._connections.items():
            for conn in connections:
                await conn.close()

# 2. Memory monitoring
import psutil
import asyncio

class MemoryMonitor:
    def __init__(self):
        self.initial_memory = psutil.Process().memory_info().rss
        
    async def monitor_memory(self):
        """Monitor memory usage and alert on leaks"""
        while True:
            current_memory = psutil.Process().memory_info().rss
            memory_growth = current_memory - self.initial_memory
            
            if memory_growth > 500 * 1024 * 1024:  # 500MB growth
                logger.warning(f"Memory growth detected: {memory_growth / 1024 / 1024:.2f}MB")
                
            await asyncio.sleep(60)  # Check every minute

# 3. Garbage collection tuning
import gc

# Tune garbage collection for better memory management
gc.set_threshold(700, 10, 10)  # More aggressive GC

# Periodic manual cleanup
async def periodic_cleanup():
    while True:
        gc.collect()
        await asyncio.sleep(300)  # Every 5 minutes
```

## 💾 Database Connection Problems

### Issue: Qdrant Connection Issues

**Symptoms:**
- Vector search failures
- Collection creation errors
- Timeout exceptions

**Diagnostic Steps:**

```bash
# Test Qdrant directly
curl http://localhost:6333/health
curl http://localhost:6333/collections

# Check Qdrant logs
docker-compose logs qdrant

# Test from within container
docker exec -it qdrant-container /bin/bash
curl localhost:6333/health
```

**Solutions:**

```python
# 1. Robust Qdrant client configuration
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import asyncio

class RobustQdrantClient:
    def __init__(self, url: str):
        self.url = url
        self.client = None
        self.max_retries = 3
        
    async def get_client(self):
        """Get or create Qdrant client with retry logic"""
        if not self.client:
            await self._initialize_client()
        return self.client
    
    async def _initialize_client(self):
        """Initialize client with retry and health check"""
        for attempt in range(self.max_retries):
            try:
                self.client = QdrantClient(
                    url=self.url,
                    timeout=30,
                    pool_size=20,
                    retries=3
                )
                
                # Verify connection
                await self.client.get_collections()
                logger.info("Qdrant client initialized successfully")
                return
                
            except Exception as e:
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

# 2. Collection management with error handling
async def ensure_collection_exists(collection_name: str, vector_size: int = 384):
    """Ensure collection exists with proper error handling"""
    client = await qdrant_client.get_client()
    
    try:
        # Check if collection exists
        collections = await client.get_collections()
        existing_names = [c.name for c in collections.collections]
        
        if collection_name not in existing_names:
            # Create collection
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
        
    except UnexpectedResponse as e:
        if "already exists" in str(e):
            logger.info(f"Collection {collection_name} already exists")
        else:
            raise
```

### Issue: Neo4j Connection Issues

**Symptoms:**
- Graph query failures
- Transaction timeouts
- Authentication errors

**Diagnostic Steps:**

```bash
# Test Neo4j connectivity
echo "RETURN 1" | cypher-shell -u neo4j -p password

# Check Neo4j browser
open http://localhost:7474

# Test from container
docker exec -it neo4j-container cypher-shell -u neo4j -p password
```

**Solutions:**

```python
# 1. Robust Neo4j driver configuration
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
import asyncio

class RobustNeo4jDriver:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.max_retries = 3
        
    async def get_driver(self):
        """Get or create Neo4j driver with retry logic"""
        if not self.driver:
            await self._initialize_driver()
        return self.driver
    
    async def _initialize_driver(self):
        """Initialize driver with proper configuration"""
        for attempt in range(self.max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60,
                    connection_timeout=30,
                    max_retry_time=30
                )
                
                # Verify connection
                await self.driver.verify_connectivity()
                logger.info("Neo4j driver initialized successfully")
                return
                
            except ServiceUnavailable as e:
                logger.warning(f"Neo4j connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

# 2. Transaction retry logic
async def execute_query_with_retry(query: str, parameters: dict = None):
    """Execute Neo4j query with retry logic for transient errors"""
    driver = await neo4j_driver.get_driver()
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            async with driver.session() as session:
                result = await session.run(query, parameters or {})
                return await result.data()
                
        except TransientError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Transient error, retrying: {e}")
                await asyncio.sleep(1)
            else:
                raise
        except Exception as e:
            logger.error(f"Non-retryable error: {e}")
            raise
```

## 🔒 Authentication & Security Issues

### Issue: JWT Token Problems

**Symptoms:**
- Constant re-authentication required
- "Token expired" errors
- Invalid signature errors

**Solutions:**

```python
# 1. Token refresh implementation
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

class TokenManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        
    def create_tokens(self, user_data: dict):
        """Create access and refresh tokens"""
        # Short-lived access token (15 minutes)
        access_token = self._create_token(
            user_data, 
            timedelta(minutes=15),
            token_type="access"
        )
        
        # Long-lived refresh token (7 days)
        refresh_token = self._create_token(
            user_data,
            timedelta(days=7),
            token_type="refresh"
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def refresh_access_token(self, refresh_token: str):
        """Generate new access token from refresh token"""
        try:
            payload = jwt.decode(
                refresh_token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            if payload.get("token_type") != "refresh":
                raise HTTPException(status_code=401, detail="Invalid token type")
                
            # Create new access token
            user_data = {k: v for k, v in payload.items() 
                        if k not in ["exp", "iat", "token_type"]}
            
            return self._create_token(
                user_data,
                timedelta(minutes=15),
                token_type="access"
            )
            
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

# 2. Middleware for automatic token refresh
@app.middleware("http")
async def token_refresh_middleware(request: Request, call_next):
    """Automatically handle token refresh"""
    
    # Skip auth for public endpoints
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        return await call_next(request)
    
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return await call_next(request)
    
    try:
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        
        # Check if token expires soon (within 5 minutes)
        exp = payload.get("exp", 0)
        if exp - datetime.utcnow().timestamp() < 300:
            logger.info("Token expiring soon, suggesting refresh")
            
    except JWTError:
        pass  # Let the endpoint handle the invalid token
    
    return await call_next(request)
```

## 📊 Monitoring & Observability

### Issue: Missing or Inaccurate Metrics

**Symptoms:**
- Prometheus metrics not updating
- Grafana dashboards empty
- Alert rules not firing

**Diagnostic Steps:**

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Prometheus configuration
docker exec prometheus cat /etc/prometheus/prometheus.yml
```

**Solutions:**

```python
# 1. Comprehensive metrics implementation
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client.exposition import make_asgi_app
import time

# Application info
app_info = Info('mcp_server_info', 'MCP Server information')
app_info.info({
    'version': '1.0.0',
    'build_date': '2025-06-27',
    'git_commit': 'abc123'
})

# Request metrics
request_count = Counter(
    'mcp_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'mcp_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Database metrics
db_connections = Gauge(
    'mcp_database_connections',
    'Active database connections',
    ['database']
)

db_query_duration = Histogram(
    'mcp_database_query_duration_seconds',
    'Database query duration',
    ['database', 'operation']
)

# Business metrics
vector_searches = Counter(
    'mcp_vector_searches_total',
    'Total vector searches performed'
)

graph_queries = Counter(
    'mcp_graph_queries_total',
    'Total graph queries performed'
)

# Middleware to collect metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)
    
    return response

# 2. Health metrics
import psutil

async def update_system_metrics():
    """Periodically update system metrics"""
    while True:
        # System metrics
        cpu_usage = Gauge('mcp_cpu_usage_percent', 'CPU usage percentage')
        memory_usage = Gauge('mcp_memory_usage_bytes', 'Memory usage in bytes')
        
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().used)
        
        # Database connection counts
        db_connections.labels(database='qdrant').set(qdrant_pool.active_connections)
        db_connections.labels(database='neo4j').set(neo4j_pool.active_connections)
        
        await asyncio.sleep(10)  # Update every 10 seconds
```

### Issue: Log Analysis Problems

**Symptoms:**
- Logs not structured
- Difficult to correlate events
- Missing context information

**Solutions:**

```python
# 1. Structured logging with correlation
import structlog
import uuid
from contextvars import ContextVar

# Request correlation ID
request_id_var: ContextVar[str] = ContextVar('request_id')

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Middleware to add correlation ID
@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    request_id_var.set(correlation_id)
    
    # Add to response headers for client tracing
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response

# 2. Enhanced logging with context
async def vector_search_with_logging(query: str, user_id: str):
    """Vector search with comprehensive logging"""
    
    # Bind context to logger
    log = logger.bind(
        operation="vector_search",
        user_id=user_id,
        query_length=len(query),
        correlation_id=request_id_var.get()
    )
    
    log.info("Starting vector search")
    start_time = time.time()
    
    try:
        # Perform search
        results = await qdrant_service.search(query)
        
        duration = time.time() - start_time
        log.info(
            "Vector search completed successfully",
            duration=duration,
            result_count=len(results),
            performance_threshold_met=duration < 1.0
        )
        
        return results
        
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "Vector search failed",
            error=str(e),
            error_type=type(e).__name__,
            duration=duration
        )
        raise

# 3. Log aggregation configuration
version: '3.8'
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    depends_on:
      - loki
```

## 🔧 Advanced Diagnostics

### Performance Profiling

```python
# 1. Application profiling
import cProfile
import pstats
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            pr.disable()
            
            # Save profile results
            stats = pstats.Stats(pr)
            stats.sort_stats('cumulative')
            stats.dump_stats(f'/tmp/{func.__name__}_profile.prof')
            
            # Log top functions
            top_functions = stats.get_stats_profile().func_profiles
            logger.info(f"Performance profile for {func.__name__}", 
                       top_functions=list(top_functions.keys())[:5])
    
    return wrapper

# 2. Database query analysis
class QueryAnalyzer:
    def __init__(self):
        self.slow_queries = []
        self.query_stats = {}
    
    async def analyze_query(self, query: str, execution_time: float):
        """Analyze query performance"""
        
        # Track slow queries
        if execution_time > 1.0:  # Queries slower than 1 second
            self.slow_queries.append({
                'query': query,
                'execution_time': execution_time,
                'timestamp': datetime.utcnow()
            })
            
            logger.warning(
                "Slow query detected",
                query=query[:100],  # First 100 chars
                execution_time=execution_time
            )
        
        # Update statistics
        query_hash = hash(query)
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'max_time': 0
            }
        
        stats = self.query_stats[query_hash]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], execution_time)

# 3. Memory leak detection
import tracemalloc
import linecache

class MemoryLeakDetector:
    def __init__(self):
        self.snapshots = []
        
    def start_monitoring(self):
        """Start memory monitoring"""
        tracemalloc.start()
        
    def take_snapshot(self):
        """Take memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'snapshot': snapshot,
            'timestamp': datetime.utcnow()
        })
        
        # Keep only last 10 snapshots
        if len(self.snapshots) > 10:
            self.snapshots.pop(0)
    
    def analyze_memory_growth(self):
        """Analyze memory growth between snapshots"""
        if len(self.snapshots) < 2:
            return
        
        current = self.snapshots[-1]['snapshot']
        previous = self.snapshots[-2]['snapshot']
        
        top_stats = current.compare_to(previous, 'lineno')
        
        logger.info("Top memory growth:")
        for stat in top_stats[:10]:
            logger.info(f"{stat}")
```

## 🛠️ Recovery Procedures

### Database Recovery

```bash
#!/bin/bash
# Database recovery script

set -euo pipefail

RECOVERY_TYPE=${1:-"auto"}
BACKUP_DATE=${2:-"latest"}

case $RECOVERY_TYPE in
  "qdrant")
    echo "Recovering Qdrant from backup..."
    kubectl exec -n mcp-server qdrant-0 -- \
      qdrant-cli backup restore --backup-dir /backups/$BACKUP_DATE/qdrant
    ;;
    
  "neo4j")
    echo "Recovering Neo4j from backup..."
    kubectl exec -n mcp-server neo4j-0 -- \
      neo4j-admin restore --from=/backups/$BACKUP_DATE/neo4j \
      --database=neo4j --force
    ;;
    
  "auto")
    echo "Full system recovery..."
    # Stop services
    kubectl scale deployment qdrant-neo4j-crawl4ai-mcp --replicas=0
    
    # Restore databases
    $0 qdrant $BACKUP_DATE
    $0 neo4j $BACKUP_DATE
    
    # Restart services
    kubectl scale deployment qdrant-neo4j-crawl4ai-mcp --replicas=3
    ;;
    
  *)
    echo "Usage: $0 {qdrant|neo4j|auto} [backup_date]"
    exit 1
    ;;
esac
```

### Service Recovery

```python
# Automatic service recovery
import asyncio
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILED = "failed"

class ServiceRecovery:
    def __init__(self):
        self.recovery_attempts = {}
        self.max_attempts = 3
        
    async def monitor_and_recover(self):
        """Monitor services and attempt recovery"""
        while True:
            try:
                # Check service health
                services = {
                    'qdrant': await self.check_qdrant_health(),
                    'neo4j': await self.check_neo4j_health(),
                    'redis': await self.check_redis_health()
                }
                
                # Attempt recovery for failed services
                for service, status in services.items():
                    if status == ServiceStatus.FAILED:
                        await self.attempt_recovery(service)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def attempt_recovery(self, service: str):
        """Attempt to recover a failed service"""
        attempt_count = self.recovery_attempts.get(service, 0)
        
        if attempt_count >= self.max_attempts:
            logger.error(f"Max recovery attempts reached for {service}")
            return
        
        logger.info(f"Attempting recovery for {service}, attempt {attempt_count + 1}")
        
        try:
            if service == 'qdrant':
                await self.recover_qdrant()
            elif service == 'neo4j':
                await self.recover_neo4j()
            elif service == 'redis':
                await self.recover_redis()
                
            # Reset attempt counter on success
            self.recovery_attempts[service] = 0
            logger.info(f"Successfully recovered {service}")
            
        except Exception as e:
            self.recovery_attempts[service] = attempt_count + 1
            logger.error(f"Recovery attempt failed for {service}: {e}")
    
    async def recover_qdrant(self):
        """Attempt to recover Qdrant connection"""
        # Recreate client with fresh connection
        global qdrant_client
        qdrant_client = QdrantClient(
            url=config.qdrant_url,
            timeout=30,
            pool_size=20
        )
        
        # Verify connection
        await qdrant_client.get_collections()
```

## 🆘 Getting Help

### Self-Service Resources

1. **Health Check**: `curl http://localhost:8000/health`
2. **API Documentation**: `http://localhost:8000/docs`
3. **Metrics**: `http://localhost:8000/metrics`
4. **Logs**: `docker-compose logs` or `kubectl logs`

### Community Support

- **GitHub Issues**: [Report bugs](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/discussions)
- **Discord Community**: [Real-time help](https://discord.gg/mcp-community)

### Professional Support

- **Enterprise Support**: `enterprise@yourproject.com`
- **Security Issues**: `security@yourproject.com`
- **Performance Consulting**: Available for production deployments

### Escalation Paths

| Severity | Response Time | Contact Method |
|----------|---------------|----------------|
| **Critical** (Production down) | 1 hour | Phone + Email |
| **High** (Major functionality impacted) | 4 hours | Email |
| **Medium** (Minor issues) | 24 hours | GitHub Issues |
| **Low** (Questions/Enhancements) | 72 hours | GitHub Discussions |

---

**Next**: [Performance Optimization →](performance-optimization.md)  
**Previous**: [← Best Practices](best-practices.md)

**Last Updated**: June 27, 2025 | **Version**: 1.0.0 | **Status**: Production Ready