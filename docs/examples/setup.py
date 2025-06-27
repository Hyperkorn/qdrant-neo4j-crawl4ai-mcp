#!/usr/bin/env python3
"""
Setup script for MCP Examples
=============================

This script helps you set up and configure the examples environment
for the Agentic RAG MCP server. It handles dependency installation,
environment configuration, and service verification.

Usage:
    python setup.py [options]

Options:
    --basic         Install basic dependencies only
    --full          Install all dependencies including optional ones
    --dev           Install development dependencies
    --check         Check service connectivity
    --demo-data     Create sample demo data
    --help          Show this help message
"""

import argparse
import asyncio
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class MCPExamplesSetup:
    """Main setup class for MCP examples."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.env_file = self.base_dir / ".env"
        self.requirements_file = self.base_dir / "requirements.txt"
        self.sample_data_dir = self.base_dir / "sample_data"
        
    def print_header(self):
        """Print setup header."""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 60)
        print("🚀 MCP Examples Setup")
        print("   Agentic RAG with Qdrant + Neo4j + Crawl4AI")
        print("=" * 60)
        print(f"{Colors.END}")
    
    def print_step(self, step: str, status: str = "info"):
        """Print setup step with status."""
        color = {
            "info": Colors.BLUE,
            "success": Colors.GREEN,
            "warning": Colors.YELLOW,
            "error": Colors.RED
        }.get(status, Colors.WHITE)
        
        emoji = {
            "info": "🔄",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }.get(status, "ℹ️")
        
        print(f"{color}{emoji} {step}{Colors.END}")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with error handling."""
        try:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            self.print_step(f"Command failed: {' '.join(command)}", "error")
            self.print_step(f"Error: {e.stderr}", "error")
            if check:
                sys.exit(1)
            return e
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.print_step("Checking Python version...")
        
        if sys.version_info < (3, 11):
            self.print_step(
                f"Python 3.11+ required, found {sys.version}", "error"
            )
            sys.exit(1)
        
        self.print_step(
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓", 
            "success"
        )
    
    def create_virtual_environment(self):
        """Create virtual environment if it doesn't exist."""
        venv_path = self.base_dir / "venv"
        
        if venv_path.exists():
            self.print_step("Virtual environment already exists", "info")
            return
        
        self.print_step("Creating virtual environment...")
        self.run_command([sys.executable, "-m", "venv", str(venv_path)])
        self.print_step("Virtual environment created", "success")
        
        # Provide activation instructions
        if os.name == 'nt':  # Windows
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:  # Unix/Linux/macOS
            activate_cmd = f"source {venv_path}/bin/activate"
        
        print(f"\n{Colors.YELLOW}To activate the virtual environment, run:{Colors.END}")
        print(f"{Colors.CYAN}{activate_cmd}{Colors.END}\n")
    
    def install_dependencies(self, install_type: str = "basic"):
        """Install Python dependencies."""
        requirements_map = {
            "basic": [
                "httpx>=0.25.0",
                "asyncio-compat>=0.1.0", 
                "python-dotenv>=1.0.0",
                "pydantic>=2.5.0",
                "structlog>=23.2.0",
                "fastapi>=0.104.0",
                "uvicorn[standard]>=0.24.0"
            ],
            "full": None,  # Use requirements.txt
            "dev": None   # Use requirements.txt + dev tools
        }
        
        if install_type == "basic":
            self.print_step("Installing basic dependencies...")
            for package in requirements_map["basic"]:
                self.print_step(f"Installing {package}...")
                self.run_command([sys.executable, "-m", "pip", "install", package])
        else:
            if not self.requirements_file.exists():
                self.print_step("requirements.txt not found", "error")
                sys.exit(1)
            
            self.print_step(f"Installing {install_type} dependencies from requirements.txt...")
            self.run_command([
                sys.executable, "-m", "pip", "install", 
                "-r", str(self.requirements_file)
            ])
            
            if install_type == "dev":
                dev_packages = [
                    "black>=23.11.0",
                    "isort>=5.12.0", 
                    "flake8>=6.1.0",
                    "mypy>=1.7.0",
                    "pre-commit>=3.5.0",
                    "pytest>=7.4.0",
                    "pytest-asyncio>=0.21.0"
                ]
                
                for package in dev_packages:
                    self.print_step(f"Installing dev package: {package}")
                    self.run_command([sys.executable, "-m", "pip", "install", package])
        
        self.print_step("Dependencies installed successfully", "success")
    
    def setup_environment_file(self):
        """Set up environment configuration file."""
        self.print_step("Setting up environment configuration...")
        
        env_example = self.base_dir / ".env.example"
        
        if not env_example.exists():
            self.print_step(".env.example not found", "error")
            return
        
        if self.env_file.exists():
            response = input(f"\n{Colors.YELLOW}.env file already exists. Overwrite? (y/N): {Colors.END}")
            if response.lower() != 'y':
                self.print_step("Keeping existing .env file", "info")
                return
        
        # Copy example to .env
        with open(env_example, 'r') as src, open(self.env_file, 'w') as dst:
            content = src.read()
            dst.write(content)
        
        self.print_step(".env file created from template", "success")
        
        # Prompt for critical configuration
        self.configure_environment()
    
    def configure_environment(self):
        """Interactive environment configuration."""
        print(f"\n{Colors.CYAN}🔧 Environment Configuration{Colors.END}")
        print("Please provide the following configuration values:")
        print("(Press Enter to use default values)\n")
        
        config_prompts = [
            ("MCP_SERVER_URL", "MCP Server URL", "http://localhost:8000"),
            ("QDRANT_URL", "Qdrant URL", "http://localhost:6333"),
            ("NEO4J_URI", "Neo4j URI", "bolt://localhost:7687"),
            ("NEO4J_USER", "Neo4j Username", "neo4j"),
            ("NEO4J_PASSWORD", "Neo4j Password", ""),
            ("LOG_LEVEL", "Log Level", "INFO")
        ]
        
        env_updates = {}
        
        for env_var, prompt, default in config_prompts:
            if env_var == "NEO4J_PASSWORD":
                import getpass
                value = getpass.getpass(f"{prompt} [{default or 'required'}]: ")
            else:
                value = input(f"{prompt} [{default}]: ").strip()
            
            if value:
                env_updates[env_var] = value
            elif default:
                env_updates[env_var] = default
        
        # Update .env file
        if env_updates:
            self.update_env_file(env_updates)
            self.print_step("Environment configuration updated", "success")
    
    def update_env_file(self, updates: Dict[str, str]):
        """Update environment file with new values."""
        if not self.env_file.exists():
            return
        
        # Read current content
        with open(self.env_file, 'r') as f:
            lines = f.readlines()
        
        # Update values
        for i, line in enumerate(lines):
            for key, value in updates.items():
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    break
        
        # Write back
        with open(self.env_file, 'w') as f:
            f.writelines(lines)
    
    async def check_service_connectivity(self):
        """Check connectivity to required services."""
        self.print_step("Checking service connectivity...")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv(self.env_file)
        
        services = [
            {
                "name": "MCP Server",
                "url": os.getenv("MCP_SERVER_URL", "http://localhost:8000") + "/health",
                "critical": True
            },
            {
                "name": "Qdrant",
                "url": os.getenv("QDRANT_URL", "http://localhost:6333") + "/collections",
                "critical": True
            },
            {
                "name": "Neo4j",
                "url": os.getenv("NEO4J_URI", "bolt://localhost:7687").replace("bolt://", "http://") + "/browser",
                "critical": True
            }
        ]
        
        async with httpx.AsyncClient() as client:
            for service in services:
                try:
                    response = await client.get(service["url"], timeout=5.0)
                    if response.status_code < 400:
                        self.print_step(f"{service['name']} is accessible", "success")
                    else:
                        self.print_step(
                            f"{service['name']} returned {response.status_code}", 
                            "warning" if not service["critical"] else "error"
                        )
                except Exception as e:
                    self.print_step(
                        f"{service['name']} is not accessible: {e}", 
                        "warning" if not service["critical"] else "error"
                    )
    
    def create_sample_data(self):
        """Create sample data for examples."""
        self.print_step("Creating sample data...")
        
        # Create sample data directory
        self.sample_data_dir.mkdir(exist_ok=True)
        
        # Sample documents
        sample_documents = {
            "ai_overview.txt": """
Artificial Intelligence (AI) is a broad field of computer science concerned with 
building smart machines capable of performing tasks that typically require human 
intelligence. Machine learning is a subset of AI that enables computers to learn 
and make decisions from data without being explicitly programmed.

Key pioneers in AI include Alan Turing, who proposed the Turing Test, and Geoffrey 
Hinton, known as the "Godfather of Deep Learning." These researchers laid the 
foundation for modern AI systems.

Applications of AI span across many industries including healthcare, finance, 
transportation, and entertainment. In healthcare, AI helps with medical diagnosis, 
drug discovery, and personalized treatment plans.
            """.strip(),
            
            "machine_learning.txt": """
Machine Learning (ML) is a method of data analysis that automates analytical 
model building. It is a branch of artificial intelligence based on the idea 
that systems can learn from data, identify patterns and make decisions with 
minimal human intervention.

Types of Machine Learning:
1. Supervised Learning - Learning with labeled examples
2. Unsupervised Learning - Finding patterns in data without labels  
3. Reinforcement Learning - Learning through interaction and feedback

Popular algorithms include linear regression, decision trees, neural networks,
support vector machines, and ensemble methods like random forests and gradient boosting.
            """.strip(),
            
            "neural_networks.txt": """
Neural Networks are computing systems inspired by biological neural networks. 
They consist of interconnected nodes (neurons) organized in layers that process 
information using a connectionist approach to computation.

Deep Learning uses neural networks with multiple hidden layers to model and 
understand complex patterns in data. This has led to breakthroughs in computer 
vision, natural language processing, and speech recognition.

Key architectures include:
- Feedforward Networks - Information flows in one direction
- Convolutional Networks (CNNs) - Excellent for image processing
- Recurrent Networks (RNNs) - Good for sequential data
- Transformers - State-of-the-art for language tasks
            """.strip(),
            
            "research_trends.txt": """
Current trends in AI and ML research include:

1. Large Language Models (LLMs) - GPT, BERT, and transformer architectures
2. Computer Vision - Object detection, image segmentation, and generation
3. Reinforcement Learning - Game playing, robotics, and optimization
4. Federated Learning - Privacy-preserving distributed learning
5. Explainable AI - Making AI decisions interpretable and transparent
6. Edge AI - Running AI models on mobile and embedded devices
7. Quantum Machine Learning - Leveraging quantum computing for ML

These areas are driving innovation in autonomous vehicles, drug discovery,
climate modeling, and many other applications.
            """.strip()
        }
        
        # Write sample documents
        for filename, content in sample_documents.items():
            file_path = self.sample_data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create sample configuration file
        sample_config = {
            "documents": list(sample_documents.keys()),
            "created_at": datetime.utcnow().isoformat(),
            "description": "Sample data for MCP examples",
            "total_documents": len(sample_documents),
            "total_characters": sum(len(content) for content in sample_documents.values())
        }
        
        config_path = self.sample_data_dir / "metadata.json"
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        self.print_step(f"Created {len(sample_documents)} sample documents", "success")
        self.print_step(f"Sample data available in: {self.sample_data_dir}", "info")
    
    def create_example_scripts(self):
        """Create convenience scripts for running examples."""
        self.print_step("Creating example scripts...")
        
        scripts_dir = self.base_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Demo script
        demo_script = """#!/bin/bash
# Demo script for MCP examples

echo "🚀 Running MCP Examples Demo"
echo "================================="

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run basic examples
echo ""
echo "1. Vector Operations Demo"
python basic-usage/vector-operations.py

echo ""
echo "2. Graph Operations Demo"
python basic-usage/graph-operations.py

echo ""
echo "3. Web Intelligence Demo"
python basic-usage/web-intelligence.py

echo ""
echo "4. Hybrid Search Demo"
python advanced-workflows/hybrid-search.py

echo ""
echo "✅ Demo completed!"
        """.strip()
        
        demo_path = scripts_dir / "run-demo.sh"
        with open(demo_path, 'w') as f:
            f.write(demo_script)
        demo_path.chmod(0o755)
        
        # Test connectivity script
        test_script = """#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup import MCPExamplesSetup

async def main():
    setup = MCPExamplesSetup()
    setup.print_header()
    await setup.check_service_connectivity()

if __name__ == "__main__":
    asyncio.run(main())
        """.strip()
        
        test_path = scripts_dir / "test-connectivity.py"
        with open(test_path, 'w') as f:
            f.write(test_script)
        test_path.chmod(0o755)
        
        self.print_step("Example scripts created", "success")
        self.print_step(f"Scripts available in: {scripts_dir}", "info")
    
    def print_next_steps(self):
        """Print next steps after setup."""
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Setup completed successfully!{Colors.END}\n")
        
        print(f"{Colors.CYAN}Next steps:{Colors.END}")
        print(f"1. {Colors.YELLOW}Start required services:{Colors.END}")
        print("   • MCP Server: Follow the main project deployment guide")
        print("   • Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("   • Neo4j: docker run -p 7474:7474 -p 7687:7687 neo4j")
        
        print(f"\n2. {Colors.YELLOW}Test connectivity:{Colors.END}")
        print("   python scripts/test-connectivity.py")
        
        print(f"\n3. {Colors.YELLOW}Run examples:{Colors.END}")
        print("   • Basic usage: python basic-usage/vector-operations.py")
        print("   • Advanced: python advanced-workflows/hybrid-search.py")
        print("   • Demo script: ./scripts/run-demo.sh")
        
        print(f"\n4. {Colors.YELLOW}Explore use cases:{Colors.END}")
        print("   • Document Q&A: cd basic-usage/document-qa-system/")
        print("   • Research Assistant: cd use-cases/research-assistant/")
        print("   • Customer Support: cd use-cases/customer-support/")
        
        print(f"\n{Colors.PURPLE}Documentation:{Colors.END}")
        print("• Examples overview: docs/examples/README.md")
        print("• API reference: docs/API_REFERENCE.md")
        print("• Architecture guide: docs/ARCHITECTURE.md")
        
        print(f"\n{Colors.BLUE}Need help? Check:{Colors.END}")
        print("• GitHub Issues: https://github.com/your-repo/issues")
        print("• Documentation: docs/")
        print("• Community: discussions/")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup MCP Examples environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--basic", action="store_true",
        help="Install basic dependencies only"
    )
    parser.add_argument(
        "--full", action="store_true", 
        help="Install all dependencies including optional ones"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Install development dependencies"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check service connectivity"
    )
    parser.add_argument(
        "--demo-data", action="store_true",
        help="Create sample demo data"
    )
    parser.add_argument(
        "--venv", action="store_true",
        help="Create virtual environment"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, run full setup
    if not any(vars(args).values()):
        args.basic = True
        args.demo_data = True
        args.venv = True
    
    setup = MCPExamplesSetup()
    setup.print_header()
    
    try:
        # Check Python version
        setup.check_python_version()
        
        # Create virtual environment
        if args.venv:
            setup.create_virtual_environment()
        
        # Install dependencies
        if args.basic:
            setup.install_dependencies("basic")
        elif args.full:
            setup.install_dependencies("full")
        elif args.dev:
            setup.install_dependencies("dev")
        
        # Setup environment
        if args.basic or args.full or args.dev:
            setup.setup_environment_file()
        
        # Check connectivity
        if args.check:
            asyncio.run(setup.check_service_connectivity())
        
        # Create demo data
        if args.demo_data:
            setup.create_sample_data()
            setup.create_example_scripts()
        
        # Show next steps
        if args.basic or args.full or args.dev:
            setup.print_next_steps()
            
    except KeyboardInterrupt:
        setup.print_step("Setup interrupted by user", "warning")
        sys.exit(1)
    except Exception as e:
        setup.print_step(f"Setup failed: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()