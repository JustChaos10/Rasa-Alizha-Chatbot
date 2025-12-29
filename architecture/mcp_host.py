"""
MCP Host - Central orchestrator for MCP clients and tool execution.

Implements the Anthropic code execution pattern:
- Manages multiple MCP client connections
- Progressive disclosure (load tool definitions on-demand)
- Tool search across all connected servers
- Code execution integration
- Skill persistence

Based on: https://www.anthropic.com/engineering/code-execution-with-mcp
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from architecture.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPTool,
    TransportType
)

logger = logging.getLogger(__name__)


@dataclass
class ToolSearchResult:
    """Result from tool search."""
    tool: MCPTool
    server_name: str
    relevance_score: float = 0.0


@dataclass
class ExecutionResult:
    """Result from code/tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class MCPHost:
    """
    MCP Host - Central orchestrator for MCP ecosystem.
    
    Features:
    1. Multi-server management: Connect to multiple MCP servers
    2. Progressive disclosure: Search tools on-demand, load definitions lazily
    3. Unified tool interface: call_tool("server.tool_name", params)
    4. Code execution: Run LLM-generated code with tool access
    5. Skill persistence: Save and reuse successful tool patterns
    
    Usage:
        host = MCPHost()
        await host.load_config("mcp_servers.json")
        await host.connect_all()
        
        # Search for tools
        tools = await host.search_tools("weather")
        
        # Call a tool
        result = await host.call_tool("weather.get_current", {"city": "London"})
        
        # Execute code
        result = await host.execute_code('''
            weather = await call_tool("weather.get_current", {"city": "London"})
            print(f"Temperature: {weather['temp']}°C")
        ''')
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._clients: Dict[str, MCPClient] = {}
        self._server_configs: Dict[str, MCPServerConfig] = {}
        self._all_tools: Dict[str, MCPTool] = {}  # "server.tool" -> MCPTool
        self._tool_descriptions: Dict[str, str] = {}  # For search
        self._skills_dir = Path("skills")
        self._code_executor = None  # Lazy init
        
        if config_path:
            asyncio.create_task(self.load_config(config_path))
    
    async def load_config(self, config_path: str) -> bool:
        """
        Load MCP server configurations from JSON file.
        
        Config format:
        {
            "servers": {
                "weather": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["servers/weather_server.py"],
                    "description": "Weather information service"
                },
                "github": {
                    "transport": "http",
                    "url": "http://localhost:8080",
                    "description": "GitHub integration"
                }
            }
        }
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"MCP config file not found: {config_path}")
                return False
            
            with open(path, 'r') as f:
                config = json.load(f)
            
            servers = config.get("servers", {})
            for name, server_config in servers.items():
                transport_str = server_config.get("transport", "stdio")
                transport = TransportType(transport_str)
                
                self._server_configs[name] = MCPServerConfig(
                    name=name,
                    transport=transport,
                    command=server_config.get("command"),
                    args=server_config.get("args"),
                    env=server_config.get("env"),
                    cwd=server_config.get("cwd"),
                    url=server_config.get("url"),
                    headers=server_config.get("headers"),
                    description=server_config.get("description", ""),
                    enabled=server_config.get("enabled", True),
                    tool_metadata=server_config.get("tool_metadata"),  # Distributed prompting
                    routing_keywords=server_config.get("routing_keywords", [])  # For improved routing
                )
            
            logger.info(f"📋 Loaded {len(self._server_configs)} MCP server configs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return False
    
    def add_server(self, config: MCPServerConfig) -> None:
        """Add a server configuration programmatically."""
        self._server_configs[config.name] = config
        logger.info(f"Added MCP server config: {config.name}")
    
    async def connect(self, server_name: str) -> bool:
        """Connect to a specific MCP server."""
        if server_name in self._clients and self._clients[server_name].is_connected:
            return True
        
        config = self._server_configs.get(server_name)
        if not config:
            logger.error(f"No config found for server: {server_name}")
            return False
        
        if not config.enabled:
            logger.info(f"Server '{server_name}' is disabled, skipping")
            return False
        
        client = MCPClient(config)
        if await client.connect():
            self._clients[server_name] = client
            
            # Index tools for search
            for tool in client.tools:
                full_name = f"{server_name}.{tool.name}"
                self._all_tools[full_name] = tool
                self._tool_descriptions[full_name] = f"{tool.description} (from {server_name})"
            
            return True
        
        return False
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all configured servers."""
        results = {}
        for name in self._server_configs:
            results[name] = await self.connect(name)
        
        connected = sum(1 for v in results.values() if v)
        logger.info(f"✅ Connected to {connected}/{len(results)} MCP servers")
        return results
    
    async def disconnect(self, server_name: str) -> None:
        """Disconnect from a specific server."""
        if server_name in self._clients:
            await self._clients[server_name].disconnect()
            del self._clients[server_name]
            
            # Remove tools from index
            to_remove = [k for k in self._all_tools if k.startswith(f"{server_name}.")]
            for key in to_remove:
                del self._all_tools[key]
                del self._tool_descriptions[key]
    
    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._clients.keys()):
            await self.disconnect(name)
    
    # =========================================================================
    # Tool Discovery (Progressive Disclosure)
    # =========================================================================
    
    def list_servers(self) -> List[str]:
        """List all connected server names."""
        return list(self._clients.keys())
    
    def list_all_tools(self, detail_level: str = "name") -> List[Dict[str, Any]]:
        """
        List all available tools with configurable detail level.
        
        Args:
            detail_level: "name" | "brief" | "full"
            
        Returns:
            List of tool info dicts
        """
        tools = []
        for full_name, tool in self._all_tools.items():
            if detail_level == "name":
                tools.append({"name": full_name})
            elif detail_level == "brief":
                tools.append({
                    "name": full_name,
                    "description": tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
                })
            else:  # full
                tools.append({
                    "name": full_name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "server": tool.server_name
                })
        return tools
    
    async def search_tools(
        self,
        query: str,
        max_results: int = 5,
        detail_level: str = "brief"
    ) -> List[ToolSearchResult]:
        """
        Search for tools matching a query.
        
        This enables progressive disclosure - the model can search for
        relevant tools instead of loading all definitions upfront.
        
        Enhanced with routing_keywords support for better tool selection.
        Priority:
        1. Exact keyword match from routing_keywords (highest)
        2. Server description match
        3. Tool name/description match
        
        Args:
            query: Search query (matches name/description/keywords)
            max_results: Maximum number of results
            detail_level: How much detail to include
            
        Returns:
            List of matching tools with relevance scores
        """
        query_lower = query.lower()
        # Split query into individual keywords for flexible matching
        query_words = set(query_lower.split())
        results = []
        
        # Build a map of server -> routing_keywords and description
        server_keywords = {}
        server_descriptions = {}
        for name, config in self._server_configs.items():
            if config.routing_keywords:
                server_keywords[name] = [kw.lower() for kw in config.routing_keywords]
            else:
                server_keywords[name] = []
            server_descriptions[name] = config.description.lower()
        
        for full_name, tool in self._all_tools.items():
            name_lower = full_name.lower()
            desc_lower = tool.description.lower()
            server_name = tool.server_name
            
            # Get server-level keywords and description
            srv_keywords = server_keywords.get(server_name, [])
            srv_desc = server_descriptions.get(server_name, "")
            
            score = 0.0
            
            # =====================================================================
            # PRIORITY 1: Check routing_keywords (highest priority)
            # These are explicit keywords defined in mcp_servers.json
            # =====================================================================
            keyword_matches = 0
            for kw in srv_keywords:
                if kw in query_lower:
                    keyword_matches += 1
                    # Multi-word keywords get higher scores
                    if " " in kw:
                        keyword_matches += 1
            
            if keyword_matches > 0:
                # High score for keyword matches (0.85 - 1.0)
                score = min(0.85 + (keyword_matches * 0.05), 1.0)
            
            # =====================================================================
            # PRIORITY 2: Check server description match
            # =====================================================================
            elif query_lower in srv_desc:
                score = 0.75
            else:
                # Check if any query word appears in server description
                srv_desc_matches = sum(1 for w in query_words if w in srv_desc and len(w) > 2)
                if srv_desc_matches > 0:
                    score = 0.6 + (srv_desc_matches / len(query_words)) * 0.15
            
            # =====================================================================
            # PRIORITY 3: Check tool name and description (fallback)
            # =====================================================================
            if score == 0:
                exact_name_match = query_lower in name_lower
                exact_desc_match = query_lower in desc_lower
                
                keyword_name_matches = sum(1 for w in query_words if w in name_lower)
                keyword_desc_matches = sum(1 for w in query_words if w in desc_lower)
                
                if exact_name_match:
                    score = 0.5
                elif exact_desc_match:
                    score = 0.4
                elif keyword_name_matches > 0:
                    score = 0.3 + (keyword_name_matches / len(query_words)) * 0.1
                elif keyword_desc_matches > 0:
                    score = 0.2 + (keyword_desc_matches / len(query_words)) * 0.1
            
            if score > 0:
                results.append(ToolSearchResult(
                    tool=tool,
                    server_name=server_name,
                    relevance_score=score
                ))
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    def get_tool_definition(self, full_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full definition for a specific tool.
        
        Format: "server_name.tool_name"
        """
        tool = self._all_tools.get(full_name)
        if not tool:
            return None
        
        return {
            "name": full_name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "server": tool.server_name
        }
    
    def get_tool_metadata(self, full_name: str) -> Dict[str, str]:
        """
        Get distributed prompting metadata for a tool.
        
        Retrieves system_instruction and code_example from server config.
        Used for injecting routing hints into LLM prompts dynamically.
        
        Args:
            full_name: "server_name.tool_name" format
            
        Returns:
            Dict with system_instruction and/or code_example, or empty dict
        """
        if "." not in full_name:
            return {}
        
        server_name, tool_name = full_name.split(".", 1)
        config = self._server_configs.get(server_name)
        
        if not config or not config.tool_metadata:
            return {}
        
        return config.tool_metadata.get(tool_name, {})
    
    def get_all_tool_instructions(self) -> List[Dict[str, str]]:
        """
        Get all system_instruction entries from connected MCP servers.
        
        Returns list of dicts with:
        - tool_name: Full tool name (server.tool)
        - system_instruction: The routing instruction
        - code_example: Optional code example
        
        Used for building dynamic LLM prompts.
        """
        instructions = []
        
        for server_name, config in self._server_configs.items():
            if not config.enabled or not config.tool_metadata:
                continue
            
            for tool_name, metadata in config.tool_metadata.items():
                full_name = f"{server_name}.{tool_name}"
                if "system_instruction" in metadata:
                    instructions.append({
                        "tool_name": full_name,
                        "system_instruction": metadata.get("system_instruction", ""),
                        "code_example": metadata.get("code_example", "")
                    })
        
        return instructions
    
    # =========================================================================
    # Tool Execution
    # =========================================================================
    
    async def call_tool(
        self,
        full_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a tool by its full name (server.tool_name).
        
        Args:
            full_name: "server_name.tool_name" format
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Parse server and tool name
        if "." not in full_name:
            return {"error": f"Invalid tool name format: {full_name}. Use 'server.tool_name'"}
        
        server_name, tool_name = full_name.split(".", 1)
        
        # Get client
        client = self._clients.get(server_name)
        if not client:
            # Try to connect if config exists
            if server_name in self._server_configs:
                if not await self.connect(server_name):
                    return {"error": f"Failed to connect to server: {server_name}"}
                client = self._clients.get(server_name)
            else:
                return {"error": f"Unknown server: {server_name}"}
        
        if not client or not client.is_connected:
            return {"error": f"Server not connected: {server_name}"}
        
        # Call the tool
        return await client.call_tool(tool_name, arguments)
    
    async def call_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls in parallel across different servers.
        
        This enables scenarios where multiple MCP servers need to be queried
        simultaneously (e.g., KB search + SQL query for comprehensive answers).
        
        Args:
            tool_calls: List of dicts with 'tool' (server.tool_name) and 'arguments'
            
        Returns:
            List of results in the same order as input tool_calls
            
        Example:
            results = await host.call_tools_parallel([
                {"tool": "knowledgebase.knowledgebase_query", "arguments": {"query": "budget info"}},
                {"tool": "sql.sql_query", "arguments": {"query": "show sales by region"}}
            ])
        """
        if not tool_calls:
            return []
        
        # Create tasks for each tool call
        async def call_with_index(index: int, call: Dict) -> Tuple[int, Dict]:
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            try:
                result = await self.call_tool(tool_name, arguments)
                return (index, {"success": True, "tool": tool_name, "result": result})
            except Exception as e:
                logger.error(f"Parallel tool call failed for {tool_name}: {e}")
                return (index, {"success": False, "tool": tool_name, "error": str(e)})
        
        # Execute all calls concurrently
        tasks = [call_with_index(i, call) for i, call in enumerate(tool_calls)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort by original index and extract results
        results = [None] * len(tool_calls)
        for item in completed:
            if isinstance(item, Exception):
                # Handle any gather exceptions
                logger.error(f"Parallel execution exception: {item}")
                continue
            index, result = item
            results[index] = result
        
        # Fill any None entries with error placeholders
        for i, r in enumerate(results):
            if r is None:
                results[i] = {"success": False, "tool": tool_calls[i].get("tool"), "error": "Execution failed"}
        
        logger.info(f"✅ Parallel tool execution: {len([r for r in results if r.get('success')])} succeeded, {len([r for r in results if not r.get('success')])} failed")
        return results
    
    # =========================================================================
    # Code Execution (Anthropic Pattern)
    # =========================================================================
    
    async def execute_code(
        self,
        code: str,
        timeout: float = 30.0,
        allowed_servers: Optional[List[str]] = None
    ) -> ExecutionResult:
        """
        Execute LLM-generated code with MCP tool access.
        
        The code can use:
        - call_tool(name, args) - Call an MCP tool
        - search_tools(query) - Search for tools
        - get_tool_definition(name) - Get tool schema
        - print() - Log output (returned to model)
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            allowed_servers: Optional whitelist of servers
            
        Returns:
            ExecutionResult with output and logs
        """
        # Lazy import code executor
        if self._code_executor is None:
            from architecture.code_executor import CodeExecutor
            self._code_executor = CodeExecutor(self)
        
        return await self._code_executor.execute(
            code,
            timeout=timeout,
            allowed_servers=allowed_servers
        )
    
    # =========================================================================
    # Skill Persistence
    # =========================================================================
    
    async def save_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Save a successful code pattern as a reusable skill.
        
        Args:
            name: Skill name (used as filename)
            code: The Python code
            description: What the skill does
            tags: Optional tags for discovery
        """
        try:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            
            skill_path = self._skills_dir / f"{name}.py"
            skill_meta_path = self._skills_dir / f"{name}.json"
            
            # Save code
            with open(skill_path, 'w') as f:
                f.write(code)
            
            # Save metadata
            metadata = {
                "name": name,
                "description": description,
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "code_file": str(skill_path)
            }
            with open(skill_meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved skill: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save skill: {e}")
            return False
    
    async def load_skill(self, name: str) -> Optional[str]:
        """Load a saved skill's code."""
        try:
            skill_path = self._skills_dir / f"{name}.py"
            if not skill_path.exists():
                return None
            
            with open(skill_path, 'r') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Failed to load skill: {e}")
            return None
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all saved skills."""
        skills = []
        
        if not self._skills_dir.exists():
            return skills
        
        for meta_path in self._skills_dir.glob("*.json"):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    skills.append(metadata)
            except Exception:
                continue
        
        return skills
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def generate_tool_tree(self) -> str:
        """
        Generate a filesystem-like tree of available tools.
        
        This is what the model sees for progressive disclosure:
        
        servers/
        ├── weather/
        │   ├── get_current.py
        │   └── get_forecast.py
        ├── github/
        │   ├── create_issue.py
        │   └── list_repos.py
        """
        lines = ["servers/"]
        
        servers = {}
        for full_name, tool in self._all_tools.items():
            server, tool_name = full_name.split(".", 1)
            if server not in servers:
                servers[server] = []
            servers[server].append(tool_name)
        
        server_names = sorted(servers.keys())
        for i, server in enumerate(server_names):
            is_last_server = (i == len(server_names) - 1)
            prefix = "└── " if is_last_server else "├── "
            lines.append(f"{prefix}{server}/")
            
            tools = sorted(servers[server])
            for j, tool in enumerate(tools):
                is_last_tool = (j == len(tools) - 1)
                inner_prefix = "    " if is_last_server else "│   "
                tool_prefix = "└── " if is_last_tool else "├── "
                lines.append(f"{inner_prefix}{tool_prefix}{tool}.py")
        
        return "\n".join(lines)
    
    @property
    def connected_servers(self) -> int:
        """Number of connected servers."""
        return len(self._clients)
    
    @property
    def total_tools(self) -> int:
        """Total number of available tools."""
        return len(self._all_tools)


# Global instance
_mcp_host: Optional[MCPHost] = None


def get_mcp_host() -> MCPHost:
    """Get or create the global MCP Host instance."""
    global _mcp_host
    if _mcp_host is None:
        _mcp_host = MCPHost()
    return _mcp_host


async def initialize_mcp_host(config_path: Optional[str] = None) -> MCPHost:
    """Initialize and connect the MCP Host."""
    host = get_mcp_host()
    
    if config_path:
        await host.load_config(config_path)
    
    await host.connect_all()
    return host
