# Kubernetes Deployment Guide

This guide covers production-ready Kubernetes deployment for the Qdrant Neo4j Crawl4AI MCP Server, including auto-scaling, high availability, and comprehensive observability.

## Overview

Kubernetes deployment provides enterprise-grade features for production workloads:

- **High Availability**: Multi-replica deployment with automatic failover
- **Auto-scaling**: Horizontal and vertical scaling based on metrics
- **Rolling Updates**: Zero-downtime deployments
- **Service Discovery**: Internal service communication
- **Resource Management**: CPU and memory limits/requests
- **Security**: RBAC, network policies, and secrets management

## Prerequisites

### Cluster Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Kubernetes | 1.28+ | 1.29+ |
| Nodes | 3 | 5+ |
| CPU (total) | 8 cores | 16+ cores |
| Memory (total) | 16 GB | 32+ GB |
| Storage | 200 GB | 500+ GB |

### Required Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

# Verify installation
kubectl version --client
helm version
kustomize version
```

## Quick Start

### 1. Deploy Infrastructure

```bash
# Clone repository
git clone https://github.com/your-username/qdrant-neo4j-crawl4ai-mcp.git
cd qdrant-neo4j-crawl4ai-mcp

# Deploy namespace and RBAC
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/rbac.yaml

# Deploy databases
kubectl apply -f k8s/manifests/qdrant.yaml
kubectl apply -f k8s/manifests/neo4j.yaml
kubectl apply -f k8s/manifests/redis.yaml

# Deploy application
kubectl apply -f k8s/manifests/qdrant-neo4j-crawl4ai-mcp.yaml

# Deploy ingress
kubectl apply -f k8s/manifests/ingress.yaml
```

### 2. Verify Deployment

```bash
# Check pod status
kubectl get pods -n qdrant-neo4j-crawl4ai-mcp

# Check services
kubectl get svc -n qdrant-neo4j-crawl4ai-mcp

# View logs
kubectl logs -f deployment/qdrant-neo4j-crawl4ai-mcp -n qdrant-neo4j-crawl4ai-mcp
```

## Helm Deployment

### Helm Chart Structure

```
helm/
├── Chart.yaml
├── values.yaml
├── values-production.yaml
├── values-staging.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── hpa.yaml
    ├── pdb.yaml
    ├── servicemonitor.yaml
    └── networkpolicy.yaml
```

### Install with Helm

```bash
# Add custom Helm repository (if needed)
helm repo add qdrant-neo4j-crawl4ai-mcp https://your-username.github.io/qdrant-neo4j-crawl4ai-mcp

# Install for production
helm install qdrant-neo4j-crawl4ai-mcp \
  --namespace qdrant-neo4j-crawl4ai-mcp \
  --create-namespace \
  --values helm/values-production.yaml \
  ./helm

# Upgrade deployment
helm upgrade qdrant-neo4j-crawl4ai-mcp \
  --namespace qdrant-neo4j-crawl4ai-mcp \
  --values helm/values-production.yaml \
  ./helm
```

### Production Values

```yaml
# helm/values-production.yaml
global:
  environment: production
  imageRegistry: ghcr.io
  storageClass: fast-ssd

replicaCount: 3

image:
  repository: your-username/qdrant-neo4j-crawl4ai-mcp
  tag: "1.0.0"
  pullPolicy: IfNotPresent

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000
  capabilities:
    drop:
    - ALL
    add:
    - NET_BIND_SERVICE

service:
  type: ClusterIP
  port: 8000
  targetPort: http
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: qdrant-neo4j-crawl4ai-mcp-tls
      hosts:
        - api.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60

nodeSelector: {}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - qdrant-neo4j-crawl4ai-mcp
        topologyKey: kubernetes.io/hostname

# Database configurations
qdrant:
  enabled: true
  persistence:
    enabled: true
    storageClass: fast-ssd
    size: 100Gi
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 500m
      memory: 1Gi

neo4j:
  enabled: true
  auth:
    neo4j:
      password: "production-password"
  persistence:
    enabled: true
    storageClass: fast-ssd
    size: 200Gi
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 1000m
      memory: 2Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "redis-password"
  persistence:
    enabled: true
    storageClass: fast-ssd
    size: 50Gi
  resources:
    limits:
      cpu: 1000m
      memory: 2Gi
    requests:
      cpu: 200m
      memory: 512Mi

# Monitoring
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true

# Network policies
networkPolicy:
  enabled: true
  ingress:
    enabled: true
  egress:
    enabled: true

# Pod disruption budget
podDisruptionBudget:
  enabled: true
  minAvailable: 2
```

## Infrastructure Components

### Namespace and RBAC

```yaml
# k8s/manifests/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: qdrant-neo4j-crawl4ai-mcp
  labels:
    name: qdrant-neo4j-crawl4ai-mcp
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    app.kubernetes.io/part-of: qdrant-neo4j-crawl4ai-mcp-platform
---
# Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: qdrant-neo4j-crawl4ai-mcp
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    app.kubernetes.io/component: service-account
automountServiceAccountToken: true
---
# Role for application
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: qdrant-neo4j-crawl4ai-mcp
  name: qdrant-neo4j-crawl4ai-mcp
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create"]
---
# Role Binding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: qdrant-neo4j-crawl4ai-mcp
  namespace: qdrant-neo4j-crawl4ai-mcp
subjects:
- kind: ServiceAccount
  name: qdrant-neo4j-crawl4ai-mcp
  namespace: qdrant-neo4j-crawl4ai-mcp
roleRef:
  kind: Role
  name: qdrant-neo4j-crawl4ai-mcp
  apiGroup: rbac.authorization.k8s.io
```

### Secrets Management

```yaml
# k8s/manifests/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-secrets
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    app.kubernetes.io/component: secrets
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-super-secure-jwt-secret-key-generate-this"
  ADMIN_API_KEY: "your-admin-api-key-generate-this"
  NEO4J_PASSWORD: "your-neo4j-password"
  REDIS_PASSWORD: "your-redis-password"
  OPENAI_API_KEY: "your-openai-api-key-if-needed"
---
# External Secrets Operator example (recommended for production)
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: qdrant-neo4j-crawl4ai-mcp
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-secrets
  namespace: qdrant-neo4j-crawl4ai-mcp
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: qdrant-neo4j-crawl4ai-mcp-secrets
    creationPolicy: Owner
  data:
  - secretKey: JWT_SECRET_KEY
    remoteRef:
      key: qdrant-neo4j-crawl4ai-mcp/jwt-secret
  - secretKey: ADMIN_API_KEY
    remoteRef:
      key: qdrant-neo4j-crawl4ai-mcp/admin-api-key
  - secretKey: NEO4J_PASSWORD
    remoteRef:
      key: qdrant-neo4j-crawl4ai-mcp/neo4j-password
```

### ConfigMap

```yaml
# k8s/manifests/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-config
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    app.kubernetes.io/component: config
data:
  # Application configuration
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  
  # Performance tuning
  WORKERS: "4"
  CRAWL4AI_MAX_CONCURRENT: "10"
  CRAWL4AI_REQUEST_TIMEOUT: "60"
  CONNECTION_TIMEOUT: "30"
  MAX_RETRIES: "3"
  
  # Service endpoints
  QDRANT_URL: "http://qdrant-service:6333"
  NEO4J_URI: "bolt://neo4j-service:7687"
  NEO4J_USER: "neo4j"
  NEO4J_DATABASE: "neo4j"
  REDIS_URL: "redis://redis-service:6379/0"
  
  # Feature flags
  ENABLE_PROMETHEUS: "true"
  ENABLE_SWAGGER_UI: "false"
  ENABLE_REDOC: "false"
  ENABLE_CORS: "true"
  
  # Security
  JWT_ALGORITHM: "HS256"
  JWT_EXPIRE_MINUTES: "60"
  
  # Default configurations
  DEFAULT_COLLECTION: "qdrant_neo4j_crawl4ai_intelligence"
  DEFAULT_EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
  
  # CORS settings
  ALLOWED_ORIGINS: "https://yourdomain.com,https://app.yourdomain.com"
  ALLOWED_METHODS: "GET,POST,PUT,DELETE,OPTIONS"
  ALLOWED_HEADERS: "Authorization,Content-Type,X-API-Key"
  
  # Crawl4AI settings
  CRAWL4AI_USER_AGENT: "QdrantNeo4jCrawl4AIMCP/1.0 (Crawl4AI; +https://github.com/qdrant-neo4j-crawl4ai-mcp)"
  CRAWL4AI_CHECK_ROBOTS_TXT: "true"
  CRAWL4AI_ENABLE_CACHING: "true"
  CRAWL4AI_CACHE_TTL: "3600"
  CRAWL4AI_MAX_RETRIES: "3"
  CRAWL4AI_RETRY_DELAY: "1.0"
```

### Ingress Configuration

```yaml
# k8s/manifests/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-ingress
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    app.kubernetes.io/component: ingress
  annotations:
    # NGINX Ingress Controller annotations
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/proxy-body-size: "32m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    
    # Rate limiting
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/rate-limit-burst-multiplier: "5"
    
    # Security headers
    nginx.ingress.kubernetes.io/custom-http-errors: "404,503"
    nginx.ingress.kubernetes.io/server-snippet: |
      add_header X-Frame-Options "SAMEORIGIN" always;
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-XSS-Protection "1; mode=block" always;
      add_header Referrer-Policy "strict-origin-when-cross-origin" always;
      add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Certificate management
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    cert-manager.io/acme-challenge-type: "http01"
    
    # WAF (if using external WAF)
    # external-dns.alpha.kubernetes.io/hostname: "api.yourdomain.com"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: qdrant-neo4j-crawl4ai-mcp-tls
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: qdrant-neo4j-crawl4ai-mcp-service
                port:
                  number: 8000
---
# Additional ingress for monitoring services (optional)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-monitoring-ingress
  namespace: qdrant-neo4j-crawl4ai-mcp
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: monitoring-auth
    nginx.ingress.kubernetes.io/auth-realm: "Monitoring Access"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - monitoring.yourdomain.com
      secretName: qdrant-neo4j-crawl4ai-mcp-monitoring-tls
  rules:
    - host: monitoring.yourdomain.com
      http:
        paths:
          - path: /grafana
            pathType: Prefix
            backend:
              service:
                name: grafana-service
                port:
                  number: 3000
          - path: /prometheus
            pathType: Prefix
            backend:
              service:
                name: prometheus-service
                port:
                  number: 9090
```

## Database Deployments

### Qdrant StatefulSet

```yaml
# k8s/manifests/qdrant.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant
    app.kubernetes.io/component: vector-database
spec:
  serviceName: qdrant-headless
  replicas: 3
  selector:
    matchLabels:
      app.kubernetes.io/name: qdrant
  template:
    metadata:
      labels:
        app.kubernetes.io/name: qdrant
        app.kubernetes.io/component: vector-database
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "6333"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: qdrant-neo4j-crawl4ai-mcp
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      
      containers:
        - name: qdrant
          image: qdrant/qdrant:v1.7.4
          imagePullPolicy: IfNotPresent
          
          ports:
            - name: http
              containerPort: 6333
              protocol: TCP
            - name: grpc
              containerPort: 6334
              protocol: TCP
          
          env:
            - name: QDRANT__SERVICE__HTTP_PORT
              value: "6333"
            - name: QDRANT__SERVICE__GRPC_PORT
              value: "6334"
            - name: QDRANT__LOG_LEVEL
              value: "INFO"
            - name: QDRANT__STORAGE__STORAGE_PATH
              value: "/qdrant/storage"
            - name: QDRANT__STORAGE__SNAPSHOTS_PATH
              value: "/qdrant/snapshots"
            - name: QDRANT__STORAGE__MEMORY_THRESHOLD_MB
              value: "2048"
            - name: QDRANT__SERVICE__MAX_REQUEST_SIZE_MB
              value: "64"
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
            - name: QDRANT__CLUSTER__P2P__PORT
              value: "6335"
          
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2
              memory: 4Gi
          
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          
          volumeMounts:
            - name: data
              mountPath: /qdrant/storage
            - name: snapshots
              mountPath: /qdrant/snapshots
          
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
  
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi
    - metadata:
        name: snapshots
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 50Gi
---
# Qdrant Service
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant
    app.kubernetes.io/component: vector-database
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "6333"
    prometheus.io/path: "/metrics"
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: qdrant
  ports:
    - name: http
      port: 6333
      targetPort: http
      protocol: TCP
    - name: grpc
      port: 6334
      targetPort: grpc
      protocol: TCP
---
# Headless service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: qdrant-headless
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant
    app.kubernetes.io/component: vector-database
spec:
  type: ClusterIP
  clusterIP: None
  selector:
    app.kubernetes.io/name: qdrant
  ports:
    - name: http
      port: 6333
      targetPort: http
    - name: grpc
      port: 6334
      targetPort: grpc
    - name: p2p
      port: 6335
      targetPort: 6335
```

### Neo4j StatefulSet

```yaml
# k8s/manifests/neo4j.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: neo4j
    app.kubernetes.io/component: graph-database
spec:
  serviceName: neo4j-headless
  replicas: 1  # Scale to 3 for production cluster
  selector:
    matchLabels:
      app.kubernetes.io/name: neo4j
  template:
    metadata:
      labels:
        app.kubernetes.io/name: neo4j
        app.kubernetes.io/component: graph-database
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "2004"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: qdrant-neo4j-crawl4ai-mcp
      securityContext:
        runAsNonRoot: true
        runAsUser: 7474
        runAsGroup: 7474
        fsGroup: 7474
      
      initContainers:
        - name: init-permissions
          image: busybox:1.36
          command: ['sh', '-c', 'chown -R 7474:7474 /data /logs /conf']
          volumeMounts:
            - name: data
              mountPath: /data
            - name: logs
              mountPath: /logs
            - name: conf
              mountPath: /conf
          securityContext:
            runAsUser: 0
      
      containers:
        - name: neo4j
          image: neo4j:5.15-enterprise
          imagePullPolicy: IfNotPresent
          
          ports:
            - name: http
              containerPort: 7474
              protocol: TCP
            - name: bolt
              containerPort: 7687
              protocol: TCP
            - name: metrics
              containerPort: 2004
              protocol: TCP
          
          env:
            - name: NEO4J_AUTH
              valueFrom:
                secretKeyRef:
                  name: qdrant-neo4j-crawl4ai-mcp-secrets
                  key: NEO4J_PASSWORD
            - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
              value: "yes"
            - name: NEO4J_PLUGINS
              value: '["apoc", "graph-data-science"]'
            - name: NEO4J_apoc_export_file_enabled
              value: "true"
            - name: NEO4J_apoc_import_file_enabled
              value: "true"
            - name: NEO4J_dbms_security_procedures_unrestricted
              value: "apoc.*,gds.*"
            - name: NEO4J_dbms_memory_heap_initial__size
              value: "2g"
            - name: NEO4J_dbms_memory_heap_max__size
              value: "4g"
            - name: NEO4J_dbms_memory_pagecache_size
              value: "2g"
            - name: NEO4J_metrics_prometheus_enabled
              value: "true"
            - name: NEO4J_metrics_prometheus_endpoint
              value: "0.0.0.0:2004"
            - name: NEO4J_dbms_connector_http_listen__address
              value: "0.0.0.0:7474"
            - name: NEO4J_dbms_connector_bolt_listen__address
              value: "0.0.0.0:7687"
            - name: NEO4J_dbms_logs_query_enabled
              value: "INFO"
            - name: NEO4J_dbms_logs_http_enabled
              value: "true"
            
            # Clustering configuration (for multi-replica)
            # - name: NEO4J_causal__clustering_initial__discovery__members
            #   value: "neo4j-0.neo4j-headless:5000,neo4j-1.neo4j-headless:5000,neo4j-2.neo4j-headless:5000"
            # - name: NEO4J_causal__clustering_discovery__listen__address
            #   value: "0.0.0.0:5000"
            # - name: NEO4J_causal__clustering_transaction__listen__address
            #   value: "0.0.0.0:6000"
            # - name: NEO4J_causal__clustering_raft__listen__address
            #   value: "0.0.0.0:7000"
          
          resources:
            requests:
              cpu: 1
              memory: 2Gi
            limits:
              cpu: 4
              memory: 8Gi
          
          livenessProbe:
            httpGet:
              path: /
              port: http
            initialDelaySeconds: 60
            periodSeconds: 20
            timeoutSeconds: 10
            failureThreshold: 3
          
          readinessProbe:
            httpGet:
              path: /
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          volumeMounts:
            - name: data
              mountPath: /data
            - name: logs
              mountPath: /logs
            - name: conf
              mountPath: /conf
            - name: plugins
              mountPath: /plugins
          
          securityContext:
            runAsNonRoot: true
            runAsUser: 7474
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
  
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 200Gi
    - metadata:
        name: logs
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 50Gi
    - metadata:
        name: conf
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 10Gi
    - metadata:
        name: plugins
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 10Gi
---
# Neo4j Service
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: neo4j
    app.kubernetes.io/component: graph-database
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: neo4j
  ports:
    - name: http
      port: 7474
      targetPort: http
      protocol: TCP
    - name: bolt
      port: 7687
      targetPort: bolt
      protocol: TCP
    - name: metrics
      port: 2004
      targetPort: metrics
      protocol: TCP
---
# Headless service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: neo4j-headless
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: neo4j
    app.kubernetes.io/component: graph-database
spec:
  type: ClusterIP
  clusterIP: None
  selector:
    app.kubernetes.io/name: neo4j
  ports:
    - name: http
      port: 7474
      targetPort: http
    - name: bolt
      port: 7687
      targetPort: bolt
    - name: discovery
      port: 5000
      targetPort: 5000
    - name: transaction
      port: 6000
      targetPort: 6000
    - name: raft
      port: 7000
      targetPort: 7000
```

## Auto-scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
# Advanced HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-hpa
  namespace: qdrant-neo4j-crawl4ai-mcp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qdrant-neo4j-crawl4ai-mcp
  minReplicas: 3
  maxReplicas: 20
  metrics:
    # CPU-based scaling
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    
    # Memory-based scaling
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    
    # Custom metric: HTTP requests per second
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
    
    # Custom metric: Query latency
    - type: Pods
      pods:
        metric:
          name: query_duration_seconds
        target:
          type: AverageValue
          averageValue: "0.5"
    
    # External metric: Queue depth
    - type: External
      external:
        metric:
          name: queue_depth
          selector:
            matchLabels:
              queue: processing
        target:
          type: AverageValue
          averageValue: "100"
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
        - type: Percent
          value: 10  # Scale down by 10% of current replicas
          periodSeconds: 60
        - type: Pods
          value: 2  # Or scale down by 2 pods
          periodSeconds: 60
      selectPolicy: Min  # Use the most conservative policy
    
    scaleUp:
      stabilizationWindowSeconds: 60  # 1 minute
      policies:
        - type: Percent
          value: 50  # Scale up by 50% of current replicas
          periodSeconds: 60
        - type: Pods
          value: 4  # Or scale up by 4 pods
          periodSeconds: 60
      selectPolicy: Max  # Use the most aggressive policy
```

### Vertical Pod Autoscaler (VPA)

```yaml
# VPA for automatic resource adjustment
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-vpa
  namespace: qdrant-neo4j-crawl4ai-mcp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qdrant-neo4j-crawl4ai-mcp
  updatePolicy:
    updateMode: "Auto"  # "Off", "Initial", or "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: qdrant-neo4j-crawl4ai-mcp
      minAllowed:
        cpu: 200m
        memory: 512Mi
      maxAllowed:
        cpu: 4
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

## Monitoring and Observability

### ServiceMonitor for Prometheus

```yaml
# Prometheus ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: qdrant-neo4j-crawl4ai-mcp
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    prometheus: kube-prometheus
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
    metricRelabelings:
    - sourceLabels: [__name__]
      regex: 'go_.*'
      action: drop  # Drop Go runtime metrics to reduce cardinality
  - port: metrics
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
---
# PrometheusRule for alerting
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-alerts
  namespace: qdrant-neo4j-crawl4ai-mcp
  labels:
    app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
    prometheus: kube-prometheus
spec:
  groups:
  - name: qdrant-neo4j-crawl4ai-mcp.rules
    interval: 30s
    rules:
    # High error rate alert
    - alert: HighErrorRate
      expr: |
        (
          rate(http_requests_total{status=~"5.."}[5m]) /
          rate(http_requests_total[5m])
        ) > 0.05
      for: 5m
      labels:
        severity: critical
        service: qdrant-neo4j-crawl4ai-mcp
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
    
    # High latency alert
    - alert: HighLatency
      expr: |
        histogram_quantile(0.95, 
          rate(http_request_duration_seconds_bucket[5m])
        ) > 1.0
      for: 10m
      labels:
        severity: warning
        service: qdrant-neo4j-crawl4ai-mcp
      annotations:
        summary: "High latency detected"
        description: "95th percentile latency is {{ $value }}s for the last 10 minutes"
    
    # Pod restart alert
    - alert: PodRestartTooOften
      expr: |
        rate(kube_pod_container_status_restarts_total[1h]) > 1/3600
      for: 0m
      labels:
        severity: warning
        service: qdrant-neo4j-crawl4ai-mcp
      annotations:
        summary: "Pod restarting too often"
        description: "Pod {{ $labels.pod }} has restarted {{ $value }} times in the last hour"
    
    # Memory usage alert
    - alert: HighMemoryUsage
      expr: |
        (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
      for: 5m
      labels:
        severity: warning
        service: qdrant-neo4j-crawl4ai-mcp
      annotations:
        summary: "High memory usage"
        description: "Memory usage is {{ $value | humanizePercentage }} for pod {{ $labels.pod }}"
    
    # Database connection alert
    - alert: DatabaseConnectionFailed
      expr: |
        up{job="qdrant-neo4j-crawl4ai-mcp"} == 0
      for: 1m
      labels:
        severity: critical
        service: qdrant-neo4j-crawl4ai-mcp
      annotations:
        summary: "Database connection failed"
        description: "Cannot connect to database services"
```

## Network Policies

```yaml
# Network security policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qdrant-neo4j-crawl4ai-mcp-netpol
  namespace: qdrant-neo4j-crawl4ai-mcp
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: qdrant-neo4j-crawl4ai-mcp
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  # Allow ingress from nginx ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  
  # Allow ingress from monitoring namespace
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 9090
  
  egress:
  # Allow egress to databases
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: qdrant
    ports:
    - protocol: TCP
      port: 6333
    - protocol: TCP
      port: 6334
  
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: neo4j
    ports:
    - protocol: TCP
      port: 7687
  
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: redis
    ports:
    - protocol: TCP
      port: 6379
  
  # Allow egress to DNS
  - to: []
    ports:
    - protocol: UDP
      port: 53
  
  # Allow egress to external web services
  - to: []
    ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
```

## Operations

### Deployment Commands

```bash
# Deploy everything
kubectl apply -k k8s/manifests/

# Rolling update
kubectl rollout restart deployment/qdrant-neo4j-crawl4ai-mcp -n qdrant-neo4j-crawl4ai-mcp

# Check rollout status
kubectl rollout status deployment/qdrant-neo4j-crawl4ai-mcp -n qdrant-neo4j-crawl4ai-mcp

# Scale manually
kubectl scale deployment qdrant-neo4j-crawl4ai-mcp --replicas=5 -n qdrant-neo4j-crawl4ai-mcp

# View events
kubectl get events -n qdrant-neo4j-crawl4ai-mcp --sort-by=.metadata.creationTimestamp
```

### Debugging

```bash
# Get pod logs
kubectl logs -f deployment/qdrant-neo4j-crawl4ai-mcp -n qdrant-neo4j-crawl4ai-mcp

# Exec into pod
kubectl exec -it deployment/qdrant-neo4j-crawl4ai-mcp -n qdrant-neo4j-crawl4ai-mcp -- /bin/bash

# Port forward for debugging
kubectl port-forward svc/qdrant-neo4j-crawl4ai-mcp-service 8000:8000 -n qdrant-neo4j-crawl4ai-mcp

# Check resource usage
kubectl top pods -n qdrant-neo4j-crawl4ai-mcp
kubectl top nodes
```

### Backup Operations

```bash
# Backup Qdrant
kubectl exec qdrant-0 -n qdrant-neo4j-crawl4ai-mcp -- \
  curl -X POST "http://localhost:6333/collections/qdrant_neo4j_crawl4ai_intelligence/snapshots"

# Backup Neo4j
kubectl exec neo4j-0 -n qdrant-neo4j-crawl4ai-mcp -- \
  neo4j-admin database dump --to-path=/backups neo4j

# Backup Redis
kubectl exec redis-0 -n qdrant-neo4j-crawl4ai-mcp -- redis-cli BGSAVE
```

## Next Steps

- **[Cloud Providers](./cloud-providers.md)** - Cloud-specific Kubernetes deployments
- **[Monitoring](./monitoring.md)** - Advanced observability setup
- **[Security Hardening](./security-hardening.md)** - Production security
- **[Backup & Recovery](./backup-recovery.md)** - Data protection strategies

---

This Kubernetes deployment provides enterprise-grade scalability, security, and observability for production workloads.
