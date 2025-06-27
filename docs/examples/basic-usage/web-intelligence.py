"""
Example: Web Intelligence with Crawl4AI
Description: Demonstrates web content crawling, extraction, and analysis
Services: Web (Crawl4AI)
Complexity: Beginner

This example shows how to:
1. Crawl websites and extract content
2. Capture screenshots for visual analysis
3. Monitor web pages for changes
4. Extract structured data from web pages
"""

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPWebClient:
    """Simple MCP client focused on web intelligence operations."""
    
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via HTTP API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/mcp/tools/{tool_name}",
                json=params,
                headers=self.headers,
                timeout=60.0  # Longer timeout for web operations
            )
            response.raise_for_status()
            return response.json()

async def demonstrate_basic_crawling():
    """Demonstrate basic website crawling and content extraction."""
    print("🔄 Demonstrating Basic Web Crawling")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Sample URLs to crawl (using publicly accessible sites)
    urls = [
        {
            "url": "https://example.com",
            "description": "Simple test page",
            "max_depth": 1
        },
        {
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "description": "AI Wikipedia page",
            "max_depth": 1
        },
        {
            "url": "https://docs.python.org/3/",
            "description": "Python documentation",
            "max_depth": 2
        }
    ]
    
    for i, site in enumerate(urls, 1):
        print(f"\n🌐 Crawling site {i}: {site['description']}")
        print(f"   URL: {site['url']}")
        
        result = await client.call_tool("crawl_website", {
            "url": site["url"],
            "max_depth": site["max_depth"],
            "max_pages": 3,
            "output_format": "markdown",
            "strategy": "bfs",
            "extract_content": True,
            "wait_time": 1.0
        })
        
        if result.get("success"):
            crawl_data = result["crawl_result"]
            print(f"   ✅ Successfully crawled {len(crawl_data.get('pages', []))} pages")
            print(f"   Crawl time: {result.get('crawl_time_ms', 0):.1f}ms")
            
            # Show first page content preview
            pages = crawl_data.get("pages", [])
            if pages:
                first_page = pages[0]
                content = first_page.get("content", "")
                print(f"   Preview: {content[:150]}...")
                print(f"   Page title: {first_page.get('title', 'No title')}")
                print(f"   Content length: {len(content)} characters")
        else:
            print(f"   ❌ Failed to crawl: {result.get('error')}")

async def demonstrate_content_extraction():
    """Demonstrate advanced content extraction with different strategies."""
    print("\n🔍 Demonstrating Content Extraction")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Different extraction strategies
    extraction_examples = [
        {
            "url": "https://news.ycombinator.com",
            "strategy": "llm",
            "instruction": "Extract the top 5 news headlines and their scores",
            "description": "LLM-based extraction from Hacker News"
        },
        {
            "url": "https://example.com",
            "strategy": "css_selector", 
            "css_selector": "h1, p",
            "description": "CSS selector extraction"
        },
        {
            "url": "https://httpbin.org/json",
            "strategy": "json",
            "description": "JSON content extraction"
        }
    ]
    
    for i, example in enumerate(extraction_examples, 1):
        print(f"\n📄 Extraction {i}: {example['description']}")
        print(f"   URL: {example['url']}")
        print(f"   Strategy: {example['strategy']}")
        
        params = {
            "url": example["url"],
            "strategy": example["strategy"],
            "output_format": "markdown"
        }
        
        # Add strategy-specific parameters
        if example["strategy"] == "llm" and "instruction" in example:
            params["instruction"] = example["instruction"]
        elif example["strategy"] == "css_selector" and "css_selector" in example:
            params["css_selector"] = example["css_selector"]
        
        result = await client.call_tool("extract_webpage_content", params)
        
        if result.get("success"):
            extracted_content = result["extracted_content"]
            print(f"   ✅ Extraction successful")
            print(f"   Content length: {len(extracted_content)} characters")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            print(f"   Preview: {extracted_content[:200]}...")
        else:
            print(f"   ❌ Extraction failed: {result.get('error')}")

async def demonstrate_structured_extraction():
    """Demonstrate structured data extraction with schemas."""
    print("\n📊 Demonstrating Structured Data Extraction")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Define a schema for structured extraction
    extraction_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Page title"},
            "description": {"type": "string", "description": "Page description or summary"},
            "main_topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main topics or themes discussed"
            },
            "key_facts": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "Important facts or information"
            },
            "publication_date": {"type": "string", "description": "Publication date if available"},
            "author": {"type": "string", "description": "Author or organization"}
        },
        "required": ["title", "description"]
    }
    
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://arxiv.org/abs/1706.03762",  # Attention is All You Need paper
        "https://blog.openai.com"
    ]
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n📋 Structured extraction {i}")
        print(f"   URL: {url}")
        
        result = await client.call_tool("extract_structured_data", {
            "url": url,
            "schema": extraction_schema,
            "strategy": "llm",
            "instruction": "Extract structured information according to the provided schema"
        })
        
        if result.get("success"):
            structured_data = result["structured_data"]
            print(f"   ✅ Structured extraction successful")
            print(f"   Title: {structured_data.get('title', 'N/A')}")
            print(f"   Description: {structured_data.get('description', 'N/A')[:100]}...")
            
            topics = structured_data.get('main_topics', [])
            if topics:
                print(f"   Topics: {', '.join(topics[:3])}")
                
            facts = structured_data.get('key_facts', [])
            if facts:
                print(f"   Key facts: {len(facts)} extracted")
        else:
            print(f"   ❌ Structured extraction failed: {result.get('error')}")

async def demonstrate_screenshot_capture():
    """Demonstrate website screenshot capabilities."""
    print("\n📸 Demonstrating Screenshot Capture")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Screenshot configurations
    screenshot_configs = [
        {
            "url": "https://example.com",
            "description": "Full page screenshot",
            "full_page": True,
            "width": 1280,
            "height": 720,
            "format": "png"
        },
        {
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "description": "Viewport screenshot with wait",
            "full_page": False,
            "width": 1920,
            "height": 1080,
            "format": "jpeg",
            "wait_time": 3.0
        }
    ]
    
    for i, config in enumerate(screenshot_configs, 1):
        print(f"\n📷 Screenshot {i}: {config['description']}")
        print(f"   URL: {config['url']}")
        print(f"   Size: {config['width']}x{config['height']}")
        
        result = await client.call_tool("capture_webpage_screenshot", {
            "url": config["url"],
            "full_page": config["full_page"],
            "width": config["width"],
            "height": config["height"],
            "format": config["format"],
            "wait_time": config.get("wait_time", 2.0)
        })
        
        if result.get("success"):
            screenshot_data = result["screenshot"]
            print(f"   ✅ Screenshot captured")
            print(f"   Size: {len(screenshot_data)} bytes")
            print(f"   Format: {config['format']}")
            print(f"   Capture time: {result.get('capture_time_ms', 0):.1f}ms")
            
            # Save screenshot to file (optional)
            filename = f"screenshot_{i}.{config['format']}"
            with open(filename, "wb") as f:
                # Note: In real implementation, screenshot_data would be base64 decoded
                print(f"   📁 Would save to: {filename}")
        else:
            print(f"   ❌ Screenshot failed: {result.get('error')}")

async def demonstrate_web_monitoring():
    """Demonstrate web page monitoring for changes."""
    print("\n👁️  Demonstrating Web Monitoring")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # Monitoring configurations
    monitoring_configs = [
        {
            "url": "https://httpbin.org/json",
            "description": "Monitor JSON endpoint for changes",
            "check_interval": 300,  # 5 minutes
            "max_checks": 3
        },
        {
            "url": "https://example.com",
            "description": "Monitor page title changes",
            "css_selector": "title",
            "check_interval": 600,  # 10 minutes  
            "max_checks": 2
        }
    ]
    
    for i, config in enumerate(monitoring_configs, 1):
        print(f"\n👀 Monitor {i}: {config['description']}")
        print(f"   URL: {config['url']}")
        print(f"   Check interval: {config['check_interval']} seconds")
        
        params = {
            "url": config["url"],
            "check_interval": config["check_interval"],
            "max_checks": config["max_checks"]
        }
        
        if "css_selector" in config:
            params["css_selector"] = config["css_selector"]
        
        result = await client.call_tool("start_webpage_monitoring", params)
        
        if result.get("success"):
            monitor_id = result["monitor_id"]
            print(f"   ✅ Monitoring started (ID: {monitor_id})")
            print(f"   Status: {result.get('status', 'Unknown')}")
            print(f"   Next check: {result.get('next_check_time', 'Unknown')}")
        else:
            print(f"   ❌ Monitoring failed: {result.get('error')}")

async def demonstrate_batch_processing():
    """Demonstrate batch processing multiple URLs."""
    print("\n⚡ Demonstrating Batch Processing")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    # URLs for batch processing
    batch_urls = [
        "https://example.com",
        "https://httpbin.org/json",
        "https://httpbin.org/html",
        "https://httpbin.org/xml"
    ]
    
    print(f"🔄 Processing {len(batch_urls)} URLs in batch")
    
    result = await client.call_tool("batch_process_urls", {
        "urls": batch_urls,
        "operation": "extract_content",
        "output_format": "markdown",
        "max_concurrent": 3,
        "timeout_per_url": 30
    })
    
    if result.get("success"):
        batch_results = result["batch_results"]
        print(f"   ✅ Batch processing completed")
        print(f"   Total URLs: {len(batch_urls)}")
        print(f"   Successful: {batch_results.get('successful', 0)}")
        print(f"   Failed: {batch_results.get('failed', 0)}")
        print(f"   Processing time: {result.get('total_time_ms', 0):.1f}ms")
        
        # Show results summary
        results = batch_results.get("results", [])
        for i, url_result in enumerate(results, 1):
            status = "✅" if url_result["success"] else "❌"
            print(f"   {status} URL {i}: {url_result['url']}")
            if url_result["success"]:
                content_length = len(url_result.get("content", ""))
                print(f"      Content: {content_length} chars")
    else:
        print(f"   ❌ Batch processing failed: {result.get('error')}")

async def demonstrate_web_analytics():
    """Demonstrate web analytics and insights."""
    print("\n📈 Demonstrating Web Analytics")
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    test_url = "https://en.wikipedia.org/wiki/Web_scraping"
    
    print(f"📊 Analyzing webpage: {test_url}")
    
    result = await client.call_tool("analyze_webpage", {
        "url": test_url,
        "include_performance": True,
        "include_seo": True,
        "include_accessibility": True,
        "include_content_analysis": True
    })
    
    if result.get("success"):
        analysis = result["analysis"]
        
        print(f"   ✅ Analysis completed")
        
        # Performance metrics
        if "performance" in analysis:
            perf = analysis["performance"]
            print(f"\n   ⚡ Performance:")
            print(f"      Load time: {perf.get('load_time_ms', 0):.1f}ms")
            print(f"      Page size: {perf.get('page_size_kb', 0):.1f} KB")
            print(f"      Resources: {perf.get('resource_count', 0)}")
        
        # SEO analysis
        if "seo" in analysis:
            seo = analysis["seo"]
            print(f"\n   🔍 SEO:")
            print(f"      Title: {seo.get('title_length', 0)} chars")
            print(f"      Description: {seo.get('meta_description_length', 0)} chars")
            print(f"      Headings: {seo.get('heading_count', 0)}")
            print(f"      Internal links: {seo.get('internal_links', 0)}")
        
        # Content analysis
        if "content" in analysis:
            content = analysis["content"]
            print(f"\n   📝 Content:")
            print(f"      Word count: {content.get('word_count', 0):,}")
            print(f"      Reading time: {content.get('reading_time_minutes', 0):.1f} min")
            print(f"      Language: {content.get('language', 'Unknown')}")
            
            topics = content.get("main_topics", [])
            if topics:
                print(f"      Topics: {', '.join(topics[:3])}")
    else:
        print(f"   ❌ Analysis failed: {result.get('error')}")

async def demonstrate_web_service_stats():
    """Demonstrate web service statistics and health."""
    print("\n📊 Demonstrating Web Service Statistics")  
    print("=" * 50)
    
    client = MCPWebClient(
        base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    )
    
    result = await client.call_tool("get_web_service_stats", {})
    
    if result.get("success"):
        stats = result["stats"]
        health = result["health"]
        
        print("📈 Service Statistics:")
        print(f"   Total requests: {stats.get('total_requests', 0):,}")
        print(f"   Successful crawls: {stats.get('successful_crawls', 0):,}")
        print(f"   Failed crawls: {stats.get('failed_crawls', 0):,}")
        print(f"   Average response time: {stats.get('avg_response_time_ms', 0):.1f}ms")
        print(f"   Total pages crawled: {stats.get('total_pages_crawled', 0):,}")
        print(f"   Total content extracted: {stats.get('total_content_mb', 0):.1f} MB")
        
        print(f"\n🏥 Service Health:")
        print(f"   Status: {health.get('status', 'Unknown')}")
        print(f"   Response time: {health.get('response_time_ms', 0):.1f}ms")
        print(f"   Active monitors: {health.get('active_monitors', 0)}")
        print(f"   Queue size: {health.get('queue_size', 0)}")
    else:
        print(f"   ❌ Failed to get stats: {result.get('error')}")

async def main():
    """Run all web intelligence demonstrations."""
    print("🚀 Web Intelligence Demo")
    print("=" * 50)
    print("This demo shows the core web intelligence capabilities:")
    print("• Website crawling and content extraction")
    print("• Screenshot capture and visual analysis")
    print("• Structured data extraction")
    print("• Web monitoring and change detection")
    print("• Batch processing and analytics")
    
    try:
        # Run demonstrations in sequence
        await demonstrate_basic_crawling()
        await demonstrate_content_extraction()
        await demonstrate_structured_extraction()
        await demonstrate_screenshot_capture()
        await demonstrate_web_monitoring()
        await demonstrate_batch_processing()
        await demonstrate_web_analytics()
        await demonstrate_web_service_stats()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("• Try vector-operations.py for semantic search")
        print("• Check graph-operations.py for knowledge graphs")
        print("• Combine all services in advanced-workflows/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("• Ensure MCP server is running")
        print("• Check internet connectivity")
        print("• Verify Crawl4AI configuration")
        print("• Check website accessibility and robots.txt")

if __name__ == "__main__":
    asyncio.run(main())