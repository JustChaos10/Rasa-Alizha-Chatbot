#!/usr/bin/env python3
"""
Secure RAG MCP Server

A simplified MCP server that exposes the Secure RAG pipeline for:
- Secure document querying with RBAC
- User permission checking
- System statistics

Uses the consolidated secure_rag.py module.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging - use stderr to avoid corrupting MCP JSON protocol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("secure_rag_server")

# Import the consolidated secure_rag module
try:
    from secure_rag_system.secure_rag import SecureRAGPipeline, PipelineResult
    SECURE_RAG_AVAILABLE = True
    logger.info("✅ Secure RAG module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Secure RAG module not available: {e}")
    SECURE_RAG_AVAILABLE = False

# Initialize MCP server
server = Server("secure-rag")

# Global pipeline instance (lazy initialization)
_pipeline: Optional[SecureRAGPipeline] = None


def get_pipeline() -> SecureRAGPipeline:
    """Get or create the SecureRAGPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SecureRAGPipeline(verbose=False)
        logger.info("🚀 SecureRAGPipeline initialized")
    return _pipeline


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Secure RAG tools."""
    tools = [
        Tool(
            name="secure_query",
            description="""Query the secure knowledge base for employee info, policies, or company data.

Use this to FIND information like "Who is Akash?", "What are the policies?", etc.
Access is filtered based on the user's role (admin sees everything, employees see limited data).

IMPORTANT: 
- 'query' = the question being asked
- 'user_id' = ID of the person ASKING (not the subject of the question)
- 'user_role' = database role ('admin' or 'user')""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to answer (e.g., 'Who is Akash?')"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "ID of the logged-in user making the request"
                    },
                    "user_role": {
                        "type": "string",
                        "description": "Database role of the user ('admin' or 'user')",
                        "default": "user"
                    }
                },
                "required": ["query", "user_id"]
            }
        ),
        Tool(
            name="check_permissions",
            description="""Check what permissions a user has based on their role.
Returns their access level and allowed document categories.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID to check permissions for"
                    },
                    "user_role": {
                        "type": "string",
                        "description": "Database role ('admin' or 'user')",
                        "default": "user"
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="list_users",
            description="""List all predefined users and their roles in the RBAC system.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_stats",
            description="""Get statistics about the Secure RAG system.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for Secure RAG operations."""
    
    if not SECURE_RAG_AVAILABLE:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "Secure RAG module not available",
                "message": "Please check the installation"
            }, indent=2)
        )]
    
    try:
        if name == "secure_query":
            result = await handle_secure_query(arguments)
        elif name == "check_permissions":
            result = await handle_check_permissions(arguments)
        elif name == "list_users":
            result = await handle_list_users(arguments)
        elif name == "get_stats":
            result = await handle_get_stats(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name
            }, indent=2)
        )]


async def handle_secure_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle secure_query tool - full pipeline execution."""
    query = arguments.get("query", "")
    user_id = arguments.get("user_id", "")
    user_role = arguments.get("user_role", "user")
    
    if not query:
        return {"error": "Query is required"}
    if not user_id:
        return {"error": "user_id is required"}
    
    pipeline = get_pipeline()
    
    # Run pipeline in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.process_query(query, user_id, user_role)
    )
    
    # Check if this is a personal query (not for knowledge base)
    if result.result_type.value == "error" and "not applicable to the knowledge base" in result.response:
        return {
            "response": None,  # Signal to router: not applicable
            "success": False,
            "blocked": False,
            "not_applicable": True,
            "reason": "Personal query - not for knowledge base"
        }
    
    # Return ONLY the response text for the frontend
    # The router should display this directly, not as JSON
    return {
        "response": result.response,
        "success": result.success,
        "blocked": result.result_type.value != "success"
    }


async def handle_check_permissions(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle check_permissions tool."""
    user_id = arguments.get("user_id", "")
    user_role = arguments.get("user_role", "user")
    
    if not user_id:
        return {"error": "user_id is required"}
    
    pipeline = get_pipeline()
    return pipeline.get_user_permissions(user_id, user_role)


async def handle_list_users(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle list_users tool."""
    pipeline = get_pipeline()
    return pipeline.list_users()


async def handle_get_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get_stats tool."""
    pipeline = get_pipeline()
    return pipeline.get_stats()


async def main():
    """Main entry point for the MCP server."""
    logger.info("🚀 Starting Secure RAG MCP Server...")
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("📡 Server ready")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
