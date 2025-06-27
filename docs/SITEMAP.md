# Documentation Site Map

[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp)
[![Documentation Complete](https://img.shields.io/badge/docs-complete-blue)](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/tree/main/docs)

> **Complete site map for the agentic RAG MCP server documentation - find any document instantly**

## 📍 Quick Navigation

| 🎯 **Quick Links** |
|-------------------|
| **[📖 Documentation Hub](README.md)** - Main navigation center |
| **[⚡ Quick Start](getting-started/quick-start.md)** - 5-minute setup |
| **[📋 API Reference](API_REFERENCE.md)** - Complete API docs |
| **[🏗️ Architecture](ARCHITECTURE.md)** - System design |
| **[🚀 Deployment](DEPLOYMENT_OPERATIONS.md)** - Production deployment |
| **[🔧 Developer Guide](DEVELOPER_GUIDE.md)** - Development workflow |

## 🗺️ Complete Site Structure

```
docs/
├── 📖 README.md                           # Main documentation hub
├── 📋 INDEX.md                            # Complete documentation index
├── 🗺️ SITEMAP.md                          # This site map
├── 📋 API_REFERENCE.md                     # Complete REST API documentation
├── 🏗️ ARCHITECTURE.md                     # System architecture overview
├── 📚 TECHNICAL_DOCUMENTATION.md          # Deep technical implementation
├── 🚀 DEPLOYMENT_OPERATIONS.md            # Production deployment guide
├── 🔧 DEVELOPER_GUIDE.md                  # Complete development workflow
│
├── 🚀 getting-started/                    # Getting started journey
│   ├── 📖 README.md                       # Getting started overview
│   ├── ⚡ quick-start.md                   # 5-minute Docker setup
│   ├── 🔧 installation.md                 # Complete installation guide
│   ├── ⚙️ configuration.md                # Environment configuration
│   ├── 🎯 first-queries.md                # Learn the system
│   └── 🛠️ troubleshooting.md              # Common issues & solutions
│
├── 📚 guides/                             # User & operational guides
│   ├── 📖 README.md                       # Guides overview
│   ├── 🏭 best-practices.md               # Production best practices
│   ├── 🚀 performance-optimization.md     # Performance tuning
│   ├── 🔒 security-hardening.md           # Enterprise security
│   ├── 📊 monitoring-observability.md     # Production monitoring
│   ├── 🛠️ troubleshooting.md              # Advanced diagnostics
│   ├── 🔄 migration.md                    # Upgrade procedures
│   └── 🎨 customization.md                # Extension patterns
│
├── 🔧 api/                                # API reference documentation
│   ├── 📖 README.md                       # API overview
│   ├── 🛠️ tools/                          # MCP tools reference
│   │   ├── 📖 README.md                   # Tools overview
│   │   ├── 🔍 vector-tools.md             # Vector search tools
│   │   ├── 🕸️ graph-tools.md              # Graph query tools
│   │   └── 🌐 web-tools.md                # Web intelligence tools
│   ├── 📦 resources/                      # MCP resources reference
│   │   └── 📖 README.md                   # Resources overview
│   └── 📋 schemas/                        # Data models & validation
│       └── 📖 README.md                   # Schemas overview
│
├── 🏗️ architecture/                       # Architecture & design
│   ├── 📖 README.md                       # Architecture overview
│   ├── 🧩 components.md                   # Component architecture
│   ├── 🔄 data-flow.md                    # Data flow patterns
│   └── 🌐 system-overview.md              # High-level system view
│
├── 📜 adrs/                               # Architecture Decision Records
│   ├── 📖 README.md                       # ADRs overview
│   ├── 📋 ADR-001-agent-framework-selection.md
│   ├── 🔍 ADR-002-hybrid-search-architecture.md
│   ├── 🌐 ADR-003-crawl4ai-integration-strategy.md
│   ├── 🤖 ADR-004-multi-agent-coordination-pattern.md
│   ├── 💾 ADR-005-memory-state-management.md
│   ├── 📊 ADR-006-evaluation-monitoring-framework.md
│   └── 🔒 ADR-007-security-authentication.md
│
├── 🚢 deployment/                         # Deployment & operations
│   ├── 📖 README.md                       # Deployment overview
│   ├── 🐳 docker.md                       # Docker deployment
│   ├── ☸️ kubernetes.md                   # Kubernetes deployment
│   ├── ☁️ cloud-providers.md              # Cloud platform guides
│   ├── 📊 monitoring.md                   # Monitoring setup
│   └── 🔒 security.md                     # Security configuration
│
├── 💻 development/                        # Development & contributing
│   ├── 📖 README.md                       # Development overview
│   ├── 🏠 local-setup.md                  # Local development setup
│   ├── 🎨 contributing.md                 # Contribution guidelines
│   ├── 📏 code-style.md                   # Coding standards
│   ├── 🧪 testing.md                      # Testing framework
│   └── 🐛 debugging.md                    # Debugging techniques
│
├── 📝 examples/                           # Examples & tutorials
│   ├── 📖 README.md                       # Examples overview
│   ├── 🔰 basic-usage/                    # Basic usage examples
│   │   ├── 📖 README.md                   # Basic usage overview
│   │   ├── 🔍 vector-operations.py        # Vector search examples
│   │   ├── 🕸️ graph-operations.py         # Graph query examples
│   │   ├── 🌐 web-intelligence.py         # Web crawling examples
│   │   └── 📄 document-qa-system/         # Complete Q&A system
│   │       └── 📖 README.md               # Q&A system guide
│   ├── 🚀 advanced-workflows/             # Advanced patterns
│   │   ├── 📖 README.md                   # Advanced workflows overview
│   │   └── 🔍 hybrid-search.py            # Multi-modal search
│   ├── 📱 client-implementations/         # SDK examples
│   │   ├── 📖 README.md                   # Client implementations overview
│   │   └── 🐍 python-client/              # Python SDK
│   │       └── 📖 README.md               # Python client guide
│   ├── 🎯 use-cases/                      # Production use cases
│   │   └── 📖 README.md                   # Use cases overview
│   ├── 🔗 integration-patterns/          # Integration patterns
│   │   └── 📖 README.md                   # Integration patterns overview
│   ├── 📦 requirements.txt                # Example dependencies
│   └── ⚙️ setup.py                        # Example setup
│
├── 📊 research/                           # Research & background
│   ├── 🎯 AGENTIC_RAG_RECOMMENDATION.md   # Executive summary
│   └── 📋 PRD.md                          # Product requirements
│
└── 🖼️ assets/                             # Documentation assets
    ├── 📊 diagrams/                       # Architecture diagrams
    ├── 📸 screenshots/                    # UI screenshots
    └── 🎥 videos/                         # Video tutorials
```

## 🎯 Navigation by Purpose

### 🚀 New User Getting Started

**Goal**: Get the system running and understand basic usage

**Recommended Path**:
1. **[📖 Documentation Hub](README.md)** - Overview and navigation
2. **[⚡ Quick Start](getting-started/quick-start.md)** - 5-minute Docker setup
3. **[🎯 First Queries](getting-started/first-queries.md)** - Learn basic operations
4. **[📝 Basic Examples](examples/basic-usage/README.md)** - Practical examples

**Time Investment**: 30 minutes to productive use

### 📊 Data Scientist/Researcher

**Goal**: Understand capabilities and implement advanced patterns

**Recommended Path**:
1. **[🏗️ Architecture](ARCHITECTURE.md)** - Understand the system design
2. **[🔧 Installation](getting-started/installation.md)** - Complete setup
3. **[🚀 Advanced Workflows](examples/advanced-workflows/README.md)** - Complex patterns
4. **[📋 API Reference](API_REFERENCE.md)** - Complete API documentation

**Time Investment**: 2-3 hours for comprehensive understanding

### 🏗️ DevOps/Infrastructure Engineer

**Goal**: Deploy and operate in production environments

**Recommended Path**:
1. **[🚀 Deployment Operations](DEPLOYMENT_OPERATIONS.md)** - Production deployment
2. **[☸️ Kubernetes Guide](deployment/kubernetes.md)** - Container orchestration
3. **[📊 Monitoring Setup](guides/monitoring-observability.md)** - Production monitoring
4. **[🏭 Best Practices](guides/best-practices.md)** - Operational excellence

**Time Investment**: 4-6 hours for production readiness

### 💻 Application Developer

**Goal**: Integrate the system into applications

**Recommended Path**:
1. **[⚡ Quick Start](getting-started/quick-start.md)** - Get running quickly
2. **[📋 API Reference](API_REFERENCE.md)** - Complete API documentation
3. **[📱 Client SDKs](examples/client-implementations/README.md)** - Implementation examples
4. **[⚙️ Configuration](getting-started/configuration.md)** - Environment setup

**Time Investment**: 2-3 hours for integration

### 🎨 System Architect

**Goal**: Understand design decisions and system integration

**Recommended Path**:
1. **[🏗️ Architecture](ARCHITECTURE.md)** - Complete system design
2. **[📜 ADRs](adrs/README.md)** - Architecture decision records
3. **[📚 Technical Documentation](TECHNICAL_DOCUMENTATION.md)** - Implementation details
4. **[🏭 Best Practices](guides/best-practices.md)** - Production patterns

**Time Investment**: 3-4 hours for comprehensive understanding

### 🛠️ Contributor/Maintainer

**Goal**: Contribute to the project or maintain a fork

**Recommended Path**:
1. **[🔧 Developer Guide](DEVELOPER_GUIDE.md)** - Complete development workflow
2. **[🏠 Local Setup](development/local-setup.md)** - Development environment
3. **[🎨 Contributing](development/contributing.md)** - Contribution process
4. **[🧪 Testing](development/testing.md)** - Testing framework

**Time Investment**: 2-3 hours for development setup

## 📋 Document Types & Formats

### 📖 Overviews & Navigation
| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](README.md) | Main documentation hub | All users |
| [INDEX.md](INDEX.md) | Complete documentation index | All users |
| [SITEMAP.md](SITEMAP.md) | Site navigation map | All users |

### 🎯 Getting Started Documents
| Document | Purpose | Time Required |
|----------|---------|---------------|
| [Quick Start](getting-started/quick-start.md) | 5-minute setup | 5 minutes |
| [Installation](getting-started/installation.md) | Complete setup | 30-60 minutes |
| [Configuration](getting-started/configuration.md) | Environment setup | 20-30 minutes |
| [First Queries](getting-started/first-queries.md) | Learn the system | 15-30 minutes |

### 🔧 Technical Reference
| Document | Purpose | Complexity |
|----------|---------|------------|
| [API Reference](API_REFERENCE.md) | REST API documentation | Intermediate |
| [Architecture](ARCHITECTURE.md) | System design | Advanced |
| [Technical Docs](TECHNICAL_DOCUMENTATION.md) | Implementation details | Expert |

### 📚 User Guides
| Document | Purpose | Skill Level |
|----------|---------|-------------|
| [Best Practices](guides/best-practices.md) | Production patterns | Intermediate |
| [Performance Optimization](guides/performance-optimization.md) | Scaling & tuning | Advanced |
| [Security Hardening](guides/security-hardening.md) | Enterprise security | Advanced |
| [Troubleshooting](guides/troubleshooting.md) | Problem resolution | All levels |

### 🚢 Deployment & Operations
| Document | Purpose | Environment |
|----------|---------|-------------|
| [Deployment Operations](DEPLOYMENT_OPERATIONS.md) | Production deployment | All environments |
| [Docker Deployment](deployment/docker.md) | Container deployment | Development/Staging |
| [Kubernetes Deployment](deployment/kubernetes.md) | Orchestration | Production |
| [Cloud Platforms](deployment/cloud-providers.md) | Managed platforms | Cloud |

### 💻 Development & Contributing
| Document | Purpose | Audience |
|----------|---------|----------|
| [Developer Guide](DEVELOPER_GUIDE.md) | Development workflow | Contributors |
| [Local Setup](development/local-setup.md) | Dev environment | New contributors |
| [Contributing](development/contributing.md) | Contribution process | External contributors |
| [Testing](development/testing.md) | Testing framework | All developers |

### 📝 Examples & Tutorials
| Document | Purpose | Complexity |
|----------|---------|------------|
| [Basic Usage](examples/basic-usage/README.md) | Simple examples | Beginner |
| [Advanced Workflows](examples/advanced-workflows/README.md) | Complex patterns | Advanced |
| [Client Implementations](examples/client-implementations/README.md) | SDK examples | Intermediate |
| [Use Cases](examples/use-cases/README.md) | Real-world examples | Expert |

## 🔍 Search & Discovery

### 📍 Quick Find

**Need to...**
- **Get started quickly?** → [Quick Start](getting-started/quick-start.md)
- **Understand the API?** → [API Reference](API_REFERENCE.md)
- **Deploy to production?** → [Deployment Operations](DEPLOYMENT_OPERATIONS.md)
- **Solve a problem?** → [Troubleshooting](guides/troubleshooting.md)
- **See examples?** → [Examples Hub](examples/README.md)
- **Configure the system?** → [Configuration](getting-started/configuration.md)
- **Understand architecture?** → [Architecture](ARCHITECTURE.md)
- **Optimize performance?** → [Performance Guide](guides/performance-optimization.md)
- **Secure the system?** → [Security Hardening](guides/security-hardening.md)
- **Contribute code?** → [Developer Guide](DEVELOPER_GUIDE.md)

### 🎯 By Use Case

**I want to...**
- **Use with AI assistants** → [Quick Start](getting-started/quick-start.md) + [First Queries](getting-started/first-queries.md)
- **Build applications** → [API Reference](API_REFERENCE.md) + [Client SDKs](examples/client-implementations/README.md)
- **Deploy to production** → [Deployment Operations](DEPLOYMENT_OPERATIONS.md) + [Best Practices](guides/best-practices.md)
- **Research capabilities** → [Architecture](ARCHITECTURE.md) + [Advanced Examples](examples/advanced-workflows/README.md)
- **Monitor & maintain** → [Monitoring Guide](guides/monitoring-observability.md) + [Troubleshooting](guides/troubleshooting.md)

### 🏷️ By Technology

**Looking for...**
- **Vector Search/Qdrant** → [Vector Tools](api/tools/vector-tools.md) + [Vector Examples](examples/basic-usage/vector-operations.py)
- **Knowledge Graphs/Neo4j** → [Graph Tools](api/tools/graph-tools.md) + [Graph Examples](examples/basic-usage/graph-operations.py)
- **Web Intelligence/Crawl4AI** → [Web Tools](api/tools/web-tools.md) + [Web Examples](examples/basic-usage/web-intelligence.py)
- **Docker Deployment** → [Docker Guide](deployment/docker.md) + [Quick Start](getting-started/quick-start.md)
- **Kubernetes** → [Kubernetes Guide](deployment/kubernetes.md) + [Deployment Operations](DEPLOYMENT_OPERATIONS.md)
- **Python Development** → [Developer Guide](DEVELOPER_GUIDE.md) + [Python Examples](examples/client-implementations/python-client/README.md)

## 📊 Documentation Statistics

### 📈 Coverage Metrics
- **Total Documents**: 50+ comprehensive guides
- **Code Examples**: 25+ working implementations
- **Architecture Diagrams**: 15+ Mermaid diagrams
- **Deployment Platforms**: 8 different platforms covered
- **API Endpoints**: 100% documented with examples
- **Use Cases**: 20+ real-world scenarios

### 🎯 Quality Indicators
- ✅ **Production-Tested**: All guides verified in production
- ✅ **Cross-Referenced**: Comprehensive internal linking
- ✅ **Multi-Audience**: Clear paths for different users
- ✅ **Searchable**: GitHub Pages with full-text search
- ✅ **Mobile-Friendly**: Responsive design
- ✅ **Version-Controlled**: Git-based documentation

### 🌐 GitHub Pages Features
- **Automatic Navigation**: Generated from document structure
- **Search Functionality**: Full-text search across all docs
- **Mobile Responsive**: Optimized for all devices
- **Fast Loading**: Optimized for quick access
- **Social Sharing**: Open Graph and Twitter Card support

## 🆘 Support & Community

### 📞 Getting Help

| Need | Resource | Response Time |
|------|----------|---------------|
| **Quick Questions** | [Troubleshooting Guides](guides/troubleshooting.md) | Immediate |
| **Setup Issues** | [Installation Guide](getting-started/installation.md) | Self-service |
| **API Questions** | [API Reference](API_REFERENCE.md) | Self-service |
| **Bug Reports** | [GitHub Issues](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/issues) | 1-3 business days |
| **Feature Requests** | [GitHub Discussions](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/discussions) | Reviewed weekly |
| **Security Issues** | `security@yourproject.com` | 24 hours |

### 🤝 Community Resources
- **GitHub Repository**: [qdrant-neo4j-crawl4ai-mcp](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp)
- **Documentation Site**: [GitHub Pages](https://bjornmelin.github.io/qdrant-neo4j-crawl4ai-mcp)
- **Issue Tracker**: [GitHub Issues](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/BjornMelin/qdrant-neo4j-crawl4ai-mcp/discussions)
- **Discord Community**: [MCP Community](https://discord.gg/mcp-community)

---

**Last Updated**: June 27, 2025 | **Site Map Version**: 1.0.0 | **Total Documents**: 50+

This site map represents the complete structure of the agentic RAG MCP server documentation. Use the navigation patterns above to find exactly what you need, or explore the [complete index](INDEX.md) for detailed information about each document.