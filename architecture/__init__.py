"""
Architecture Module - Core MCP Infrastructure

This module contains all the core architectural components for the MCP-based
chatbot system including:

- MCP Client/Host: Connection management for MCP servers
- Router: LLM-based intelligent routing
- HybridRouter: Combined routing with form handling
- Registry: Plugin discovery and management
- Code Executor: Safe code execution sandbox
- Tool Logger: Structured logging for tool calls
- Conversation Memory: Context management
- Form Registry: Adaptive card form handling
"""

from .base_tool import BaseTool, ToolSchema
from .registry import ToolRegistry, get_registry
from .mcp_client import MCPClient, MCPServerConfig, MCPTool
from .mcp_host import MCPHost, get_mcp_host, initialize_mcp_host
from .router import LLMRouter, RouterConfig, get_router
from .hybrid_router import HybridRouter, HybridRouterConfig
from .code_executor import CodeExecutor
from .tool_logger import ToolLogger, get_tool_logger, ToolType
from .conversation_memory import ConversationMemory, get_conversation_memory
from .form_registry import FormHandlerRegistry, FormSubmission

# NOTE: LLM calls should use GlobalLLMService from shared_utils
# Use: from shared_utils import get_global_llm_service

__all__ = [
    # Base classes
    "BaseTool",
    "ToolSchema",
    
    # Registry
    "ToolRegistry",
    "get_registry",
    
    # MCP
    "MCPClient",
    "MCPServerConfig", 
    "MCPTool",
    "MCPHost",
    "get_mcp_host",
    "initialize_mcp_host",
    
    # Routing
    "LLMRouter",
    "RouterConfig",
    "get_router",
    "HybridRouter",
    "HybridRouterConfig",
    
    # Execution
    "CodeExecutor",
    
    # Logging
    "ToolLogger",
    "get_tool_logger",
    "ToolType",
    
    # Memory
    "ConversationMemory",
    "get_conversation_memory",
    
    # Forms
    "FormHandlerRegistry",
    "FormSubmission",
]
