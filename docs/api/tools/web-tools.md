# Web Intelligence Tools

The Web Intelligence service provides smart web crawling, content extraction, and monitoring capabilities through Crawl4AI integration. These MCP tools enable AI assistants to gather, process, and monitor web content with advanced extraction strategies.

## Overview

Web intelligence tools provide:

- **Smart Web Crawling**: Multi-page crawling with configurable strategies
- **Content Extraction**: AI-powered content extraction with multiple formats
- **Screenshot Capture**: Full-page and viewport screenshots  
- **Content Monitoring**: Automated change detection and notifications
- **Rate-Limited Access**: Respectful crawling with robots.txt compliance

## Available Tools

### crawl_web_page

Crawl and extract content from a web page or website with comprehensive configuration options.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `string` | ✅ | - | Target URL to crawl |
| `max_depth` | `integer` | ❌ | 1 | Maximum crawling depth (1-3) |
| `max_pages` | `integer` | ❌ | 10 | Maximum pages to crawl (1-50) |
| `output_format` | `string` | ❌ | "markdown" | Output format: markdown, html, json |
| `strategy` | `string` | ❌ | "bfs" | Crawling strategy: bfs, dfs, best_first |
| `include_external` | `boolean` | ❌ | `false` | Include external domains |
| `extract_content` | `boolean` | ❌ | `true` | Extract main content |
| `capture_screenshots` | `boolean` | ❌ | `false` | Capture screenshots |
| `wait_time` | `float` | ❌ | 0.0 | Wait time before extraction (0-30s) |

**Response:**

```json
{
  "status": "completed",
  "url": "https://example.com",
  "total_pages": 5,
  "successful_pages": 5,
  "failed_pages": 0,
  "crawl_time_ms": 3250.5,
  "content": [
    {
      "url": "https://example.com",
      "title": "Example Article",
      "content": "This is the main content of the article...",
      "content_type": "article",
      "format": "markdown",
      "word_count": 450,
      "links_count": 12,
      "images_count": 3,
      "status_code": 200
    }
  ],
  "metadata": {
    "urls_crawled": ["https://example.com", "https://example.com/about"],
    "urls_failed": [],
    "robots_txt_allowed": true,
    "rate_limited": false,
    "errors": [],
    "warnings": []
  }
}
```

**Example Usage:**

```python
# Basic single-page crawl
result = await session.call_tool(
    "crawl_web_page",
    {
        "url": "https://example.com/article",
        "output_format": "markdown",
        "extract_content": True
    }
)

# Multi-page crawling with depth
site_crawl = await session.call_tool(
    "crawl_web_page",
    {
        "url": "https://docs.example.com",
        "max_depth": 2,
        "max_pages": 20,
        "strategy": "bfs",
        "include_external": False,
        "output_format": "markdown"
    }
)

# Process crawled content
for page in site_crawl.content['content']:
    print(f"Page: {page['title']}")
    print(f"URL: {page['url']}")
    print(f"Words: {page['word_count']}")
    print(f"Links: {page['links_count']}")
    print("---")

# Comprehensive crawl with screenshots
full_crawl = await session.call_tool(
    "crawl_web_page",
    {
        "url": "https://example.com",
        "max_depth": 2,
        "max_pages": 15,
        "output_format": "html",
        "capture_screenshots": True,
        "wait_time": 2.0
    }
)
```

**Crawling Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `bfs` | Breadth-first search | Site overviews, navigation |
| `dfs` | Depth-first search | Deep content exploration |
| `best_first` | Priority-based crawling | Important content first |

**Output Formats:**

| Format | Description | Use Cases |
|--------|-------------|-----------|
| `markdown` | Clean markdown text | Documentation, articles |
| `html` | Structured HTML | Preserving formatting |
| `json` | Structured data | API integration, analysis |

---

### extract_web_content

Extract specific content from a web page using targeted strategies.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `string` | ✅ | - | Target URL for extraction |
| `strategy` | `string` | ❌ | "llm" | Extraction strategy: llm, css_selector, regex |
| `css_selector` | `string` | ❌ | `null` | CSS selector for extraction |
| `regex_pattern` | `string` | ❌ | `null` | Regex pattern for extraction |
| `instruction` | `string` | ❌ | `null` | LLM extraction instruction |
| `schema` | `object` | ❌ | `null` | JSON schema for structured extraction |
| `output_format` | `string` | ❌ | "markdown" | Output format |

**Response:**

```json
{
  "status": "success",
  "url": "https://example.com/article",
  "title": "Example Article Title",
  "content": "Extracted content based on strategy...",
  "content_type": "article",
  "word_count": 320,
  "character_count": 1850,
  "extraction_time_ms": 850.2,
  "metadata": {
    "strategy_used": "llm",
    "confidence": 0.92,
    "elements_found": 1
  }
}
```

**Example Usage:**

```python
# LLM-based content extraction
result = await session.call_tool(
    "extract_web_content",
    {
        "url": "https://news.example.com/article-123",
        "strategy": "llm",
        "instruction": "Extract the main article content, including title, author, publish date, and body text. Ignore navigation, ads, and comments.",
        "output_format": "markdown"
    }
)

# CSS selector extraction
specific_content = await session.call_tool(
    "extract_web_content",
    {
        "url": "https://example.com/product",
        "strategy": "css_selector",
        "css_selector": ".product-description, .price, .reviews",
        "output_format": "json"
    }
)

# Regex pattern extraction
pattern_result = await session.call_tool(
    "extract_web_content",
    {
        "url": "https://example.com/data",
        "strategy": "regex",
        "regex_pattern": r"Price: \$(\d+\.?\d*)",
        "output_format": "json"
    }
)

# Structured data extraction with schema
structured_extraction = await session.call_tool(
    "extract_web_content",
    {
        "url": "https://example.com/research-paper",
        "strategy": "llm",
        "instruction": "Extract paper information according to the provided schema",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "abstract": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "publication_date": {"type": "string", "format": "date"},
                "doi": {"type": "string"}
            },
            "required": ["title", "authors", "abstract"]
        },
        "output_format": "json"
    }
)

# Process structured result
if structured_extraction.content['status'] == 'success':
    import json
    paper_data = json.loads(structured_extraction.content['content'])
    print(f"Title: {paper_data['title']}")
    print(f"Authors: {', '.join(paper_data['authors'])}")
    print(f"Keywords: {', '.join(paper_data.get('keywords', []))}")
```

**Extraction Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `llm` | AI-powered intelligent extraction | Complex layouts, semantic understanding |
| `css_selector` | Precise element selection | Known site structures, specific elements |
| `regex` | Pattern-based text extraction | Consistent patterns, structured data |

---

### capture_web_screenshot

Capture screenshots of web pages with configurable options.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `string` | ✅ | - | Target URL for screenshot |
| `full_page` | `boolean` | ❌ | `true` | Capture full page |
| `width` | `integer` | ❌ | 1280 | Screenshot width (320-3840) |
| `height` | `integer` | ❌ | 720 | Screenshot height (240-2160) |
| `format` | `string` | ❌ | "png" | Image format: png, jpeg, webp |
| `wait_for_selector` | `string` | ❌ | `null` | Wait for CSS selector |
| `wait_time` | `float` | ❌ | 2.0 | Wait time before capture (0-30s) |

**Response:**

```json
{
  "status": "success",
  "url": "https://example.com",
  "image_data": "iVBORw0KGgoAAAANSUhEUgAA...",
  "format": "png",
  "width": 1280,
  "height": 1024,
  "file_size_bytes": 245760,
  "capture_time_ms": 1250.3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Basic full-page screenshot
result = await session.call_tool(
    "capture_web_screenshot",
    {
        "url": "https://example.com",
        "full_page": True,
        "format": "png"
    }
)

# Save screenshot to file
import base64
image_data = base64.b64decode(result.content['image_data'])
with open("screenshot.png", "wb") as f:
    f.write(image_data)

# Mobile viewport screenshot
mobile_screenshot = await session.call_tool(
    "capture_web_screenshot",
    {
        "url": "https://example.com",
        "width": 375,
        "height": 667,
        "full_page": False,
        "format": "jpeg"
    }
)

# Wait for dynamic content
dynamic_content = await session.call_tool(
    "capture_web_screenshot",
    {
        "url": "https://spa-app.example.com",
        "wait_for_selector": ".content-loaded",
        "wait_time": 5.0,
        "full_page": True
    }
)

# High-quality screenshot for documentation
doc_screenshot = await session.call_tool(
    "capture_web_screenshot",
    {
        "url": "https://dashboard.example.com",
        "width": 1920,
        "height": 1080,
        "format": "png",
        "wait_time": 3.0
    }
)

# Multiple screenshots for comparison
urls = [
    "https://example.com",
    "https://competitor1.com", 
    "https://competitor2.com"
]

screenshots = []
for url in urls:
    result = await session.call_tool(
        "capture_web_screenshot",
        {
            "url": url,
            "width": 1280,
            "height": 800,
            "format": "png"
        }
    )
    screenshots.append({
        "url": url,
        "image_data": result.content['image_data'],
        "file_size": result.content['file_size_bytes']
    })
```

**Image Formats:**

| Format | Description | Best For |
|--------|-------------|----------|
| `png` | Lossless, high quality | Screenshots, documentation |
| `jpeg` | Compressed, smaller files | Web sharing, thumbnails |
| `webp` | Modern format, good compression | Web optimization |

---

### monitor_web_content

Start monitoring a web page for content changes with configurable notifications.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `string` | ✅ | - | URL to monitor |
| `check_interval` | `integer` | ❌ | 300 | Check interval in seconds (60-86400) |
| `css_selector` | `string` | ❌ | `null` | CSS selector to monitor |
| `webhook_url` | `string` | ❌ | `null` | Webhook for notifications |
| `max_checks` | `integer` | ❌ | 100 | Maximum number of checks (1-1000) |

**Response:**

```json
{
  "status": "success",
  "monitor_id": "monitor_abc123",
  "url": "https://example.com",
  "monitor_status": "active",
  "check_interval": 300,
  "max_checks": 100,
  "last_check": "2024-01-15T10:30:00Z",
  "next_check": "2024-01-15T10:35:00Z",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Monitor entire page for changes
result = await session.call_tool(
    "monitor_web_content",
    {
        "url": "https://news.example.com",
        "check_interval": 600,  # 10 minutes
        "max_checks": 1440      # 10 days at 10-minute intervals
    }
)

monitor_id = result.content['monitor_id']
print(f"Started monitoring: {monitor_id}")

# Monitor specific element with webhook
specific_monitor = await session.call_tool(
    "monitor_web_content",
    {
        "url": "https://shop.example.com/product/123",
        "css_selector": ".price",
        "check_interval": 300,  # 5 minutes
        "webhook_url": "https://your-app.com/webhooks/price-change",
        "max_checks": 500
    }
)

# Monitor multiple pages
pages_to_monitor = [
    {
        "url": "https://competitor1.com/pricing",
        "selector": ".pricing-table",
        "interval": 3600  # 1 hour
    },
    {
        "url": "https://competitor2.com/features", 
        "selector": ".feature-list",
        "interval": 7200  # 2 hours
    }
]

active_monitors = []
for page in pages_to_monitor:
    monitor = await session.call_tool(
        "monitor_web_content",
        {
            "url": page["url"],
            "css_selector": page["selector"],
            "check_interval": page["interval"],
            "webhook_url": "https://your-app.com/webhooks/competitor-change"
        }
    )
    active_monitors.append(monitor.content['monitor_id'])
```

---

### get_web_monitor_status

Get the current status and statistics for a web content monitor.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `monitor_id` | `string` | ✅ | - | Unique identifier of the monitor |

**Response:**

```json
{
  "status": "success",
  "monitor_id": "monitor_abc123",
  "url": "https://example.com",
  "monitor_status": "active",
  "last_check": "2024-01-15T10:30:00Z",
  "next_check": "2024-01-15T10:35:00Z",
  "change_detected": false,
  "check_count": 25,
  "error_count": 1,
  "last_error": "Connection timeout",
  "created_at": "2024-01-14T15:20:00Z"
}
```

**Example Usage:**

```python
# Check single monitor status
status = await session.call_tool(
    "get_web_monitor_status",
    {"monitor_id": "monitor_abc123"}
)

print(f"Monitor Status: {status.content['monitor_status']}")
print(f"Checks Performed: {status.content['check_count']}")
print(f"Changes Detected: {status.content['change_detected']}")

if status.content['error_count'] > 0:
    print(f"Last Error: {status.content['last_error']}")

# Monitor health check
async def check_monitor_health(monitor_ids):
    unhealthy_monitors = []
    
    for monitor_id in monitor_ids:
        status = await session.call_tool(
            "get_web_monitor_status", 
            {"monitor_id": monitor_id}
        )
        
        # Check for issues
        if status.content['monitor_status'] != 'active':
            unhealthy_monitors.append({
                "id": monitor_id,
                "status": status.content['monitor_status'],
                "issue": "Inactive"
            })
        elif status.content['error_count'] > 5:
            unhealthy_monitors.append({
                "id": monitor_id,
                "status": "errors",
                "issue": f"High error count: {status.content['error_count']}"
            })
    
    return unhealthy_monitors

# Automated status reporting
async def generate_monitor_report(monitor_ids):
    report = {"active": 0, "inactive": 0, "errors": 0, "changes": 0}
    
    for monitor_id in monitor_ids:
        status = await session.call_tool(
            "get_web_monitor_status",
            {"monitor_id": monitor_id}
        )
        
        if status.content['monitor_status'] == 'active':
            report['active'] += 1
        else:
            report['inactive'] += 1
            
        if status.content['error_count'] > 0:
            report['errors'] += 1
            
        if status.content['change_detected']:
            report['changes'] += 1
    
    return report
```

---

### stop_web_monitor

Stop an active web content monitor and clean up resources.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `monitor_id` | `string` | ✅ | - | Unique identifier of the monitor to stop |

**Response:**

```json
{
  "status": "success",
  "message": "Monitor stopped successfully",
  "monitor_id": "monitor_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Usage:**

```python
# Stop a specific monitor
result = await session.call_tool(
    "stop_web_monitor",
    {"monitor_id": "monitor_abc123"}
)

if result.content['status'] == 'success':
    print(f"Monitor {result.content['monitor_id']} stopped")

# Stop multiple monitors
monitor_ids = ["monitor_abc123", "monitor_def456", "monitor_ghi789"]

for monitor_id in monitor_ids:
    try:
        result = await session.call_tool(
            "stop_web_monitor",
            {"monitor_id": monitor_id}
        )
        print(f"Stopped: {monitor_id}")
    except Exception as e:
        print(f"Failed to stop {monitor_id}: {e}")

# Conditional monitor cleanup
async def cleanup_old_monitors():
    # Get all monitors
    monitors = await session.call_tool("list_web_monitors")
    
    stopped_count = 0
    for monitor in monitors.content['monitors']:
        # Stop monitors older than 30 days
        created_at = datetime.fromisoformat(monitor['created_at'].replace('Z', '+00:00'))
        if (datetime.now(timezone.utc) - created_at).days > 30:
            await session.call_tool(
                "stop_web_monitor",
                {"monitor_id": monitor['monitor_id']}
            )
            stopped_count += 1
    
    return stopped_count
```

---

### list_web_monitors

List all active web content monitors with their status information.

**Parameters:** None

**Response:**

```json
{
  "status": "success",
  "total_monitors": 3,
  "monitors": [
    {
      "monitor_id": "monitor_abc123",
      "url": "https://example.com",
      "status": "active",
      "last_check": "2024-01-15T10:30:00Z",
      "next_check": "2024-01-15T10:35:00Z",
      "change_detected": false,
      "check_count": 25,
      "error_count": 1,
      "created_at": "2024-01-14T15:20:00Z"
    }
  ]
}
```

**Example Usage:**

```python
# List all monitors
result = await session.call_tool("list_web_monitors")

print(f"Total Monitors: {result.content['total_monitors']}")

for monitor in result.content['monitors']:
    print(f"Monitor: {monitor['monitor_id']}")
    print(f"  URL: {monitor['url']}")
    print(f"  Status: {monitor['status']}")
    print(f"  Checks: {monitor['check_count']}")
    print(f"  Errors: {monitor['error_count']}")
    print(f"  Changes: {monitor['change_detected']}")
    print("---")

# Filter and analyze monitors
def analyze_monitors(monitors):
    analysis = {
        "total": len(monitors),
        "active": 0,
        "inactive": 0,
        "with_changes": 0,
        "with_errors": 0,
        "avg_checks": 0
    }
    
    total_checks = 0
    for monitor in monitors:
        if monitor['status'] == 'active':
            analysis['active'] += 1
        else:
            analysis['inactive'] += 1
            
        if monitor['change_detected']:
            analysis['with_changes'] += 1
            
        if monitor['error_count'] > 0:
            analysis['with_errors'] += 1
            
        total_checks += monitor['check_count']
    
    if monitors:
        analysis['avg_checks'] = total_checks / len(monitors)
    
    return analysis

# Usage
all_monitors = await session.call_tool("list_web_monitors")
analysis = analyze_monitors(all_monitors.content['monitors'])

print(f"Monitor Analysis:")
print(f"  Active: {analysis['active']}/{analysis['total']}")
print(f"  With Changes: {analysis['with_changes']}")
print(f"  With Errors: {analysis['with_errors']}")
print(f"  Avg Checks: {analysis['avg_checks']:.1f}")
```

---

### web_service_health

Get comprehensive health information about the web intelligence service.

**Parameters:** None

**Response:**

```json
{
  "status": "healthy",
  "service": "web",
  "browser_available": true,
  "memory_usage_mb": 156.8,
  "active_crawls": 2,
  "total_crawls": 45,
  "success_rate": 0.89,
  "average_response_time_ms": 2340.5,
  "response_time_ms": 15.2,
  "timestamp": "2024-01-15T10:30:00Z",
  "statistics": {
    "pages_crawled": 1250,
    "pages_extracted": 1180,
    "screenshots_captured": 85,
    "monitors_active": 8,
    "total_data_mb": 450.2
  },
  "details": {
    "browser_version": "Chrome 121.0.6167.85",
    "crawl4ai_version": "0.2.77",
    "max_concurrent_crawls": 10,
    "queue_size": 3
  }
}
```

**Example Usage:**

```python
# Check service health
health = await session.call_tool("web_service_health")

print(f"Web Service: {health.content['status']}")
print(f"Browser Available: {health.content['browser_available']}")
print(f"Memory Usage: {health.content['memory_usage_mb']:.1f} MB")
print(f"Active Crawls: {health.content['active_crawls']}")
print(f"Success Rate: {health.content['success_rate']:.1%}")

# Service statistics
stats = health.content['statistics']
print(f"\nStatistics:")
print(f"  Pages Crawled: {stats['pages_crawled']:,}")
print(f"  Pages Extracted: {stats['pages_extracted']:,}")
print(f"  Screenshots: {stats['screenshots_captured']:,}")
print(f"  Active Monitors: {stats['monitors_active']}")
print(f"  Total Data: {stats['total_data_mb']:.1f} MB")

# Health monitoring automation
async def monitor_web_service():
    health = await session.call_tool("web_service_health")
    
    alerts = []
    
    # Check critical metrics
    if health.content['status'] != 'healthy':
        alerts.append(f"Service unhealthy: {health.content['status']}")
    
    if not health.content['browser_available']:
        alerts.append("Browser not available")
    
    if health.content['memory_usage_mb'] > 500:
        alerts.append(f"High memory usage: {health.content['memory_usage_mb']:.1f} MB")
    
    if health.content['success_rate'] < 0.8:
        alerts.append(f"Low success rate: {health.content['success_rate']:.1%}")
    
    if health.content['active_crawls'] > 8:
        alerts.append(f"High concurrent crawls: {health.content['active_crawls']}")
    
    return {
        "healthy": len(alerts) == 0,
        "alerts": alerts,
        "metrics": health.content
    }

# Performance trend analysis
async def analyze_performance_trends():
    health = await session.call_tool("web_service_health")
    
    current_metrics = {
        "timestamp": health.content['timestamp'],
        "success_rate": health.content['success_rate'],
        "avg_response_time": health.content['average_response_time_ms'],
        "memory_usage": health.content['memory_usage_mb'],
        "active_crawls": health.content['active_crawls']
    }
    
    # Store metrics for trend analysis
    # In a real implementation, you'd store these in a database
    return current_metrics
```

## Best Practices

### Respectful Crawling

1. **Check robots.txt:**
   ```python
   # The service automatically checks robots.txt
   # You can verify compliance in the response
   if not result.content['metadata']['robots_txt_allowed']:
       print("⚠️ robots.txt disallows crawling")
   ```

2. **Rate Limiting:**
   ```python
   # Use appropriate delays between requests
   await session.call_tool(
       "crawl_web_page",
       {
           "url": "https://example.com",
           "wait_time": 1.0,  # Respectful delay
           "max_pages": 10    # Reasonable limit
       }
   )
   ```

3. **Concurrent Limits:**
   ```python
   # Monitor active crawls
   health = await session.call_tool("web_service_health")
   if health.content['active_crawls'] > 5:
       print("High concurrent activity - consider waiting")
   ```

### Content Extraction Optimization

1. **Choose Appropriate Strategy:**
   ```python
   # For known site structures
   structured_content = await session.call_tool(
       "extract_web_content",
       {
           "url": url,
           "strategy": "css_selector",
           "css_selector": "article.main-content"
       }
   )
   
   # For unknown or complex layouts
   smart_content = await session.call_tool(
       "extract_web_content",
       {
           "url": url,
           "strategy": "llm",
           "instruction": "Extract the main article content, ignoring navigation and ads"
       }
   )
   ```

2. **Structured Data Extraction:**
   ```python
   # Define clear schemas for consistent results
   product_schema = {
       "type": "object",
       "properties": {
           "name": {"type": "string"},
           "price": {"type": "number"},
           "description": {"type": "string"},
           "availability": {"type": "string"},
           "rating": {"type": "number", "minimum": 0, "maximum": 5}
       },
       "required": ["name", "price"]
   }
   ```

### Screenshot Best Practices

1. **Viewport Considerations:**
   ```python
   # Desktop viewport
   desktop = {"width": 1920, "height": 1080}
   
   # Tablet viewport  
   tablet = {"width": 768, "height": 1024}
   
   # Mobile viewport
   mobile = {"width": 375, "height": 667}
   ```

2. **Wait for Dynamic Content:**
   ```python
   # Wait for specific elements
   await session.call_tool(
       "capture_web_screenshot",
       {
           "url": "https://spa-app.com",
           "wait_for_selector": ".content-loaded",
           "wait_time": 5.0
       }
   )
   ```

### Monitoring Setup

1. **Strategic Monitor Placement:**
   ```python
   # Monitor critical pages
   critical_monitors = [
       {
           "url": "https://api.example.com/status",
           "interval": 300,    # 5 minutes
           "selector": ".status"
       },
       {
           "url": "https://example.com/pricing",
           "interval": 3600,   # 1 hour
           "selector": ".price-table"
       }
   ]
   ```

2. **Webhook Integration:**
   ```python
   # Set up webhook for notifications
   webhook_url = "https://your-app.com/webhooks/change-detected"
   
   monitor = await session.call_tool(
       "monitor_web_content",
       {
           "url": target_url,
           "webhook_url": webhook_url,
           "check_interval": 1800  # 30 minutes
       }
   )
   ```

## Integration Patterns

### Content Pipeline

```python
async def web_content_pipeline(urls, extraction_config):
    """Process multiple URLs through extraction pipeline."""
    
    results = []
    
    for url in urls:
        try:
            # 1. Extract content
            extraction = await session.call_tool(
                "extract_web_content",
                {
                    "url": url,
                    "strategy": extraction_config.get("strategy", "llm"),
                    "instruction": extraction_config.get("instruction"),
                    "output_format": "markdown"
                }
            )
            
            # 2. Capture screenshot for reference
            screenshot = await session.call_tool(
                "capture_web_screenshot",
                {
                    "url": url,
                    "width": 1280,
                    "height": 800
                }
            )
            
            # 3. Store in vector database (if available)
            if extraction.content['status'] == 'success':
                vector_result = await session.call_tool(
                    "store_vector_document",
                    {
                        "content": extraction.content['content'],
                        "source": url,
                        "metadata": {
                            "url": url,
                            "title": extraction.content['title'],
                            "word_count": extraction.content['word_count'],
                            "extracted_at": extraction.content['timestamp']
                        }
                    }
                )
            
            results.append({
                "url": url,
                "extraction": extraction.content,
                "screenshot": screenshot.content,
                "indexed": vector_result.content if 'vector_result' in locals() else None
            })
            
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e),
                "extraction": None,
                "screenshot": None
            })
    
    return results
```

### Competitive Intelligence

```python
async def setup_competitive_monitoring(competitors):
    """Set up monitoring for competitor websites."""
    
    monitors = []
    
    for competitor in competitors:
        # Monitor pricing pages
        pricing_monitor = await session.call_tool(
            "monitor_web_content",
            {
                "url": f"{competitor['website']}/pricing",
                "css_selector": ".pricing-table, .price",
                "check_interval": 3600,  # 1 hour
                "webhook_url": f"https://your-app.com/webhooks/competitor/{competitor['id']}/pricing"
            }
        )
        
        # Monitor feature pages
        feature_monitor = await session.call_tool(
            "monitor_web_content",
            {
                "url": f"{competitor['website']}/features",
                "css_selector": ".feature-list, .features",
                "check_interval": 7200,  # 2 hours
                "webhook_url": f"https://your-app.com/webhooks/competitor/{competitor['id']}/features"
            }
        )
        
        # Monitor blog/news for announcements
        news_monitor = await session.call_tool(
            "monitor_web_content",
            {
                "url": f"{competitor['website']}/blog",
                "css_selector": ".blog-posts, .news",
                "check_interval": 14400,  # 4 hours
                "webhook_url": f"https://your-app.com/webhooks/competitor/{competitor['id']}/news"
            }
        )
        
        monitors.extend([pricing_monitor, feature_monitor, news_monitor])
    
    return monitors
```

### Documentation Scraping

```python
async def scrape_documentation_site(base_url, max_pages=50):
    """Scrape technical documentation with smart navigation."""
    
    # 1. Initial crawl to discover structure
    site_crawl = await session.call_tool(
        "crawl_web_page",
        {
            "url": base_url,
            "max_depth": 3,
            "max_pages": max_pages,
            "strategy": "bfs",
            "output_format": "markdown",
            "include_external": False
        }
    )
    
    # 2. Extract and structure content
    structured_docs = []
    
    for page in site_crawl.content['content']:
        if page['content_type'] in ['article', 'documentation']:
            # Extract structured information
            doc_extraction = await session.call_tool(
                "extract_web_content",
                {
                    "url": page['url'],
                    "strategy": "llm",
                    "instruction": """
                    Extract documentation content with this structure:
                    - Title
                    - Summary/Overview  
                    - Main content sections
                    - Code examples
                    - Related links
                    """,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "heading": {"type": "string"},
                                        "content": {"type": "string"}
                                    }
                                }
                            },
                            "code_examples": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "related_links": {
                                "type": "array", 
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "output_format": "json"
                }
            )
            
            structured_docs.append({
                "url": page['url'],
                "raw_content": page['content'],
                "structured_content": doc_extraction.content
            })
    
    return {
        "total_pages": len(structured_docs),
        "documentation": structured_docs,
        "crawl_metadata": site_crawl.content['metadata']
    }
```

---

*For more information, see the [API Overview](../README.md) or explore [Vector Tools](./vector-tools.md) and [Graph Tools](./graph-tools.md).*