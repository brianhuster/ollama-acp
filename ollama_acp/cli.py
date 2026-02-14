#!/usr/bin/env python3
"""
Ollama Agent CLI - Command line interface for Ollama Agent
"""

import asyncio
import logging
import sys

try:
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    print("Error: ollama package not installed")
    print("Please install: pip install ollama")
    sys.exit(1)

from .agent import OllamaAgent, run_agent


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def list_models_command(host: str) -> int:
    """List available models and exit"""
    client = OllamaAsyncClient(host=host)
    
    try:
        result = await client.list()
        models = result.models if hasattr(result, 'models') else []
        
        if not models:
            print("No models found. Please pull a model first:")
            print("  ollama pull llama3.2")
            return 1
        
        print(f"\nAvailable models on {host}:")
        print("-" * 70)
        
        for m in models:
            if hasattr(m, 'model'):
                name = m.model
                size_gb = m.size / (1024**3) if hasattr(m, 'size') else 0
                modified = m.modified_at.strftime('%Y-%m-%d %H:%M') if hasattr(m, 'modified_at') else 'Unknown'
                print(f"  • {name:<35} {size_gb:>7.2f} GB    {modified}")
        
        print("-" * 70)
        print(f"Total: {len(models)} models\n")
        return 0
        
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        print(f"\nError: Could not connect to Ollama at {host}")
        print("Make sure Ollama is running: ollama serve")
        return 1


async def run_agent_command(model: str, host: str) -> int:
    """Run the agent"""
    
    agent = OllamaAgent(model=model, ollama_host=host)
    
    # Verify connection
    if not await agent.verify_connection():
        logging.error("Failed to connect to Ollama. Exiting.")
        return 1
    
    try:
        await run_agent(agent)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logging.exception("Agent error")
        return 1


def parse_args():
    """Parse command line arguments"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        prog='ollama-agent',
        description='Ollama Agent - Agent Client Protocol adapter for Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  ollama-agent --model llama3.2
  ollama-agent --model gemma3:1b --host http://192.168.1.100:11434
  ollama-agent --list-models
  ollama-agent --debug
  
Environment Variables:
  OLLAMA_MODEL    Default model name (default: llama3.2)
  OLLAMA_HOST     Ollama server URL (default: http://localhost:11434)
        '''
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help='Ollama model to use (default: llama3.2 or OLLAMA_MODEL env var)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help='Ollama server URL (default: http://localhost:11434 or OLLAMA_HOST env var)'
    )
    
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point"""
    args = parse_args()
    setup_logging(args.debug)
    
    # Handle list-models command
    if args.list_models:
        return asyncio.run(list_models_command(args.host))
    
    # Run agent
    return asyncio.run(run_agent_command(args.model, args.host))


if __name__ == "__main__":
    sys.exit(main())
