"""
LLM Router - Intelligent tool selection using Groq LLM.

This module provides a lightweight LLM-based router that:
1. Uses Tool Search to find relevant tools (token efficient)
2. Routes queries to the appropriate tool (local plugins + MCP servers)
3. Extracts parameters from natural language
4. Falls back to chat for general conversation
5. Supports code execution for complex multi-step operations

Based on Anthropic's advanced tool use patterns.
See: https://www.anthropic.com/engineering/code-execution-with-mcp
"""

import json
import logging
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import httpx

from architecture.registry import get_registry
from architecture.mcp_host import MCPHost, get_mcp_host
from architecture.code_executor import CodeExecutor
from architecture.telemetry import trace_llm_call, log_llm_event, is_telemetry_enabled

# Import GlobalLLMService for all LLM calls
import importlib.util
_shared_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared_utils.py')
_spec = importlib.util.spec_from_file_location('shared_utils', _shared_utils_path)
_shared_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared_utils)
get_global_llm_service = _shared_utils.get_global_llm_service

# Try to import Global Input Guard from consolidated module
try:
    from mcp_servers.secure_rag_system.secure_rag import InputGuard
    INPUT_GUARD_AVAILABLE = True
except ImportError:
    INPUT_GUARD_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Result from an LLM API call including usage data."""
    content: str
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    provider: str = "unknown"
    model: str = "unknown"


@dataclass
class RouterConfig:
    """Configuration for the LLM Router."""
    # Groq API settings
    groq_api_key: str = ""
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    
    # Gemini API settings (fallback/load balancing)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"  # Upgraded model
    
    # Model settings - use fast model for routing
    routing_model: str = "llama-3.3-70b-versatile"
    chat_model: str = "llama-3.3-70b-versatile"
    code_model: str = "llama-3.3-70b-versatile"  # Model for code generation
    
    # Routing settings
    routing_temperature: float = 0.1  # Low temp for consistent routing
    routing_max_tokens: int = 500
    routing_timeout: float = 10.0
    
    # Code execution settings
    code_execution_enabled: bool = True
    code_execution_timeout: float = 300.0
    code_first_mode: bool = True  # NEW: Use code execution as primary path
    
    # Tool search settings
    max_tools_per_search: int = 5
    min_search_score: float = 0.1
    
    # MCP settings
    mcp_config_path: str = "mcp_servers.json"
    enable_mcp: bool = True
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour


class PromptCache:
    """
    Simple in-memory cache for prompt responses.
    Reduces API calls for repeated queries.
    """
    
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl
    
    def _hash_key(self, prompt: str, tools: List[str]) -> str:
        """Generate cache key from prompt and tools."""
        content = f"{prompt}:{','.join(sorted(tools))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, tools: List[str]) -> Optional[Any]:
        """Get cached response if available and not expired."""
        import time
        key = self._hash_key(prompt, tools)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, prompt: str, tools: List[str], value: Any) -> None:
        """Cache a response."""
        import time
        key = self._hash_key(prompt, tools)
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


class LLMRouter:
    """
    LLM-based router for intelligent tool selection and execution.
    
    PRIMARY MODE (code_first_mode=True):
    - LLM receives minimal tool list (just names + descriptions)
    - LLM generates Python code to call tools via call_tool()
    - Code is executed in sandbox with access to both local plugins and MCP tools
    - This avoids clogging the context window with full tool schemas
    
    LEGACY MODE (code_first_mode=False):
    - Traditional routing: LLM selects tool and extracts params
    - Tool schemas are loaded into context
    - Direct tool execution
    
    Features:
    - Unified call_tool() for both local plugins and MCP servers
    - Tool Search: Discovers relevant tools on-demand
    - Code Execution: Primary path for tool calls
    - MCP Integration: Connects to external MCP servers
    - Fallback: Routes to chat for general conversation
    
    Usage:
        router = LLMRouter()
        await router.initialize()  # Connect to MCP servers
        response = await router.route_and_execute("What's the weather in Mumbai?")
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        
        # Load API keys from environment if not provided
        if not self.config.groq_api_key:
            self.config.groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        if not self.config.gemini_api_key:
            self.config.gemini_api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        
        if not self.config.groq_api_key and not self.config.gemini_api_key:
            logger.warning("⚠️ No LLM API keys set - LLM features will be limited")
        
        # Initialize components
        self.registry = get_registry()
        self.cache = PromptCache(ttl=self.config.cache_ttl) if self.config.enable_cache else None
        self._client: Optional[httpx.AsyncClient] = None
        
        # MCP components (lazy init)
        self._mcp_host: Optional[MCPHost] = None
        self._code_executor: Optional[CodeExecutor] = None
        self._mcp_initialized = False
        
        # Use GlobalLLMService for all LLM calls
        self._global_llm = get_global_llm_service()
        
        mode = "code-first" if self.config.code_first_mode else "legacy"
        providers = []
        if self.config.groq_api_key:
            providers.append("GROQ")
        if self.config.gemini_api_key:
            providers.append("Gemini")
            
        # Initialize Global Input Guard
        self.input_guard = None
        if INPUT_GUARD_AVAILABLE:
            try:
                self.input_guard = InputGuard()
                logger.info("🛡️ Global Input Guard initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Global Input Guard: {e}")
                
        logger.info(f"✅ LLMRouter initialized (mode: {mode}, providers: {providers})")
    
    async def initialize(self) -> None:
        """
        Initialize MCP host and connect to configured servers.
        Call this before using MCP features.
        """
        if self._mcp_initialized:
            return
        
        if self.config.enable_mcp:
            try:
                self._mcp_host = get_mcp_host()
                
                # Load config if exists
                if os.path.exists(self.config.mcp_config_path):
                    await self._mcp_host.load_config(self.config.mcp_config_path)
                    await self._mcp_host.connect_all()
                    logger.info(f"🔌 MCP Host connected: {self._mcp_host.connected_servers} servers, {self._mcp_host.total_tools} tools")
                
                # Initialize code executor (now handles both local and MCP tools)
                if self.config.code_execution_enabled:
                    self._code_executor = CodeExecutor(self._mcp_host, self.registry)
                    logger.info("🐍 Code executor initialized (unified local + MCP)")
                
                self._mcp_initialized = True
                
            except Exception as e:
                logger.warning(f"⚠️ MCP initialization failed: {e}")
    
    @property
    def mcp_host(self) -> Optional[MCPHost]:
        """Get the MCP host instance."""
        return self._mcp_host
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        prefer_provider: Optional[str] = None,
        is_small_request: bool = False,
        trace_name: str = "router-llm-call"
    ) -> str:
        """
        LLM call using GlobalLLMService.
        
        All routing, rate limiting, and failover is handled by GlobalLLMService.
        """
        # Use json_mode if response_format requests JSON
        json_mode = response_format and response_format.get("type") == "json_object"
        
        result = await self._global_llm.call_with_messages_async(
            messages=messages,
            max_tokens=max_tokens or self.config.routing_max_tokens,
            temperature=temperature if temperature is not None else self.config.routing_temperature,
            timeout=self.config.routing_timeout,
            json_mode=json_mode,
            trace_name=trace_name
        )
        
        return result
    
    def _build_routing_prompt(self, query: str, available_tools: List[str], mcp_tools: List[Dict] = None) -> str:
        """
        Build the tool routing prompt.
        
        Includes parameter schemas for accurate extraction.
        Now includes both local plugins and MCP tools.
        """
        tools_desc = []
        
        # Local plugin tools
        for name in available_tools:
            schema = self.registry.get_schema(name)
            if schema:
                # Include parameter info for better extraction
                params = schema.parameters.get("properties", {})
                param_names = list(params.keys())
                param_str = f" (params: {', '.join(param_names)})" if param_names else ""
                tools_desc.append(f"- {name}: {schema.description}{param_str}")
        
        # MCP tools (external servers)
        if mcp_tools:
            tools_desc.append("\n[External MCP Tools]:")
            for tool in mcp_tools:
                name = tool.get("name", "")
                desc = tool.get("description", "")[:100]
                # Include parameter names from input_schema
                input_schema = tool.get("input_schema", {})
                params = input_schema.get("properties", {})
                param_names = list(params.keys())
                param_str = f" (params: {', '.join(param_names)})" if param_names else ""
                tools_desc.append(f"- {name}: {desc}{param_str}")
        
        tools_text = "\n".join(tools_desc) if tools_desc else "No specific tools available."
        
        # Add code execution mode hint
        code_hint = ""
        if self.config.code_execution_enabled and self._code_executor:
            code_hint = """

Special modes:
- "code_execution": For complex multi-step operations that require combining multiple tools or data transformations.
  When this mode is selected, you'll generate Python code to orchestrate the operation."""
        
        return f"""You are a tool router. Given a user query, decide which tool to use and extract parameters.

Available tools:
{tools_text}{code_hint}

Rules:
1. Choose the most appropriate tool for the query
2. If no tool matches well, use "chat" for general conversation
3. Extract relevant parameters from the query using the EXACT parameter names shown above
4. For MCP tools, use the full name format "server.tool_name"
5. Use "code_execution" mode for complex operations requiring multiple steps or data transformations
6. For weather queries:
   - Extract the city name into the "city" parameter
   - Extract time references into the "when" parameter (e.g., "today", "tomorrow", "next week", "in 3 days")
   - If no time is mentioned, don't include "when" (defaults to current weather)
7. For search queries, extract the search terms into the "query" parameter
8. Return valid JSON only

User query: "{query}"

Respond with JSON:
{{"tool": "tool_name", "params": {{"param_name": "extracted_value"}}, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

For code execution, respond with:
{{"tool": "code_execution", "params": {{"task": "description of what code should do"}}, "confidence": 0.0-1.0, "reasoning": "needs multi-step processing"}}

Examples:
- "What's the weather in Tokyo?" -> {{"tool": "weather", "params": {{"city": "Tokyo"}}, "confidence": 0.95, "reasoning": "weather query for Tokyo, no time specified"}}
- "Weather in London tomorrow" -> {{"tool": "weather", "params": {{"city": "London", "when": "tomorrow"}}, "confidence": 0.95, "reasoning": "weather forecast for London tomorrow"}}
- "What will the weather be in Mumbai next week?" -> {{"tool": "weather", "params": {{"city": "Mumbai", "when": "next week"}}, "confidence": 0.95, "reasoning": "weather forecast for Mumbai next week"}}
- "Is it going to rain in Delhi today?" -> {{"tool": "weather", "params": {{"city": "Delhi", "when": "today"}}, "confidence": 0.95, "reasoning": "weather query for Delhi today"}}
- "Temperature in Paris in 3 days" -> {{"tool": "weather", "params": {{"city": "Paris", "when": "in 3 days"}}, "confidence": 0.95, "reasoning": "weather forecast for Paris in 3 days"}}
- "Search for Python tutorials" -> {{"tool": "chat", "params": {{"message": "Search for Python tutorials"}}, "confidence": 0.9, "reasoning": "search request - chat has integrated web search"}}
- "Hello there" -> {{"tool": "chat", "params": {{"message": "Hello there"}}, "confidence": 1.0, "reasoning": "greeting - general conversation"}}"""""
    
    async def route(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate tool.
        
        This is the main routing logic:
        1. Search for relevant tools (local + MCP)
        2. Ask LLM to select tool and extract params
        3. Return routing decision
        
        Args:
            query: The user's query
            context: Optional context (user info, conversation history)
            
        Returns:
            Dict with 'tool', 'params', 'confidence', 'reasoning'
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(query, self.registry.list_tools())
            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached
        
        # Step 1: Tool Search - find relevant local tools
        always_loaded = [t.schema.name for t in self.registry.get_always_loaded_tools()]
        searched_tools = self.registry.search_tools(query, max_results=self.config.max_tools_per_search)
        
        # Combine always-loaded and searched tools (deduplicated)
        available_tools = list(dict.fromkeys(always_loaded + searched_tools))
        
        # Step 2: Search MCP tools (if enabled and initialized)
        mcp_tools = []
        if self._mcp_host and self._mcp_host.connected_servers > 0:
            try:
                mcp_results = await self._mcp_host.search_tools(query, max_results=3)
                mcp_tools = [
                    {
                        "name": f"{r.server_name}.{r.tool.name}",
                        "description": r.tool.description,
                        "input_schema": r.tool.input_schema,
                        "score": r.relevance_score
                    }
                    for r in mcp_results
                ]
            except Exception as e:
                logger.warning(f"MCP tool search failed: {e}")
        
        if not available_tools and not mcp_tools:
            # No tools found - fall back to chat
            return {
                "tool": "chat",
                "params": {"message": query},
                "confidence": 1.0,
                "reasoning": "No tools matched the query"
            }
        
        logger.debug(f"Available tools: local={available_tools}, mcp={[t['name'] for t in mcp_tools]}")
        
        # Step 3: LLM routing decision
        try:
            prompt = self._build_routing_prompt(query, available_tools, mcp_tools)
            
            response = await self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                is_small_request=True,  # Routing is a small request
                trace_name="router-tool-selection"
            )
            
            decision = json.loads(response)
            
            # Validate the decision
            tool_name = decision.get("tool", "chat")
            
            # Check if it's a special mode
            if tool_name == "code_execution":
                # Valid - handled by execute_with_code
                pass
            elif tool_name == "chat":
                # Valid - fallback
                pass
            elif "." in tool_name:
                # MCP tool - validate server exists
                server_name = tool_name.split(".")[0]
                if self._mcp_host and server_name in self._mcp_host.list_servers():
                    pass  # Valid MCP tool
                else:
                    logger.warning(f"LLM selected unknown MCP tool '{tool_name}', falling back to chat")
                    decision = {
                        "tool": "chat",
                        "params": {"message": query},
                        "confidence": 0.5,
                        "reasoning": f"MCP server '{server_name}' not connected"
                    }
            elif tool_name not in available_tools:
                logger.warning(f"LLM selected unknown tool '{tool_name}', falling back to chat")
                decision = {
                    "tool": "chat",
                    "params": {"message": query},
                    "confidence": 0.5,
                    "reasoning": f"Selected tool '{tool_name}' not available"
                }
            
            # Cache the result
            if self.cache:
                self.cache.set(query, self.registry.list_tools(), decision)
            
            logger.info(f"Routing decision: {tool_name} (confidence: {decision.get('confidence', 'N/A')})")
            return decision
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM routing response: {e}")
            return {
                "tool": "chat",
                "params": {"message": query},
                "confidence": 0.5,
                "reasoning": "Failed to parse routing decision"
            }
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return {
                "tool": "chat",
                "params": {"message": query},
                "confidence": 0.5,
                "reasoning": f"Routing error: {str(e)}"
            }
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Supports both local plugins and MCP tools.
        
        Args:
            tool_name: Name of the tool to execute (or "server.tool" for MCP)
            params: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        from architecture.tool_logger import get_tool_logger, ToolType
        tool_logger = get_tool_logger()
        
        # Check if it's an MCP tool (format: "server.tool_name")
        if "." in tool_name and self._mcp_host:
            server_name = tool_name.split(".")[0]
            
            with tool_logger.tool_call(
                tool_name,
                ToolType.MCP_SERVER,
                params,
                server_name=server_name
            ) as log:
                try:
                    result = await self._mcp_host.call_tool(tool_name, params)
                    log.success = "error" not in result
                    
                    # Check result type for logging
                    data = result.get("data", result.get("content", result))
                    if isinstance(data, dict):
                        if data.get("type") == "adaptive_card":
                            log.result_type = "adaptive_card"
                            tool_logger.log_adaptive_card("leave_form" if "leave" in tool_name else "generic")
                        else:
                            log.result_type = data.get("type", "dict")
                    else:
                        log.result_type = "text"
                    
                    return {
                        "success": "error" not in result,
                        "data": data,
                        "error": result.get("error"),
                        "source": "mcp"
                    }
                except Exception as e:
                    logger.error(f"MCP tool execution error: {e}")
                    log.success = False
                    log.error = str(e)
                    return {
                        "success": False,
                        "error": str(e),
                        "source": "mcp"
                    }
        
        # Local plugin tool
        tool = self.registry.get_tool(tool_name)
        
        if tool is None:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        with tool_logger.tool_call(
            tool_name,
            ToolType.PLUGIN,
            params
        ) as log:
            try:
                result = await tool.safe_execute(**params)
                log.success = result.get("success", False)
                log.result_type = result.get("type", "unknown")
                return result
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                log.success = False
                log.error = str(e)
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def execute_with_code(
        self,
        task: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate and execute code for complex multi-step operations.
        
        This implements Anthropic's code execution pattern:
        1. LLM analyzes the task and available tools
        2. LLM generates Python code to accomplish the task
        3. Code is executed in sandbox with tool access
        4. Results are returned
        
        Args:
            task: Description of what to accomplish
            context: Optional context
            
        Returns:
            Execution result with output and logs
        """
        if not self._code_executor:
            return {
                "success": False,
                "error": "Code execution not enabled"
            }
        
        # Get available tools for context
        local_tools = self.registry.list_tools()
        mcp_tools = []
        if self._mcp_host:
            mcp_tools = self._mcp_host.list_all_tools(detail_level="brief")
        
        # Build code generation prompt
        prompt = self._build_code_generation_prompt(task, local_tools, mcp_tools)
        
        try:
            # Generate code - larger request, prefer Gemini
            code = await self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.code_model,
                temperature=0.3,
                max_tokens=2000,
                is_small_request=False,  # Code gen is a large request
                trace_name="router-code-generation"
            )
            
            # Extract code from response (handle markdown code blocks)
            code = self._extract_code(code)
            
            logger.info(f"Generated code for task: {task[:50]}...")
            logger.debug(f"Code:\n{code}")
            
            # Execute in sandbox
            result = await self._code_executor.execute(
                code,
                timeout=self.config.code_execution_timeout
            )
            
            return {
                "success": result.success,
                "output": result.output,
                "logs": result.logs,
                "error": result.error,
                "code": code,
                "execution_time": result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_code_generation_prompt(
        self,
        task: str,
        local_tools: List[str],
        mcp_tools: List[Dict]
    ) -> str:
        """Build prompt for code generation."""
        tools_info = []
        
        # Local tools
        for name in local_tools[:10]:  # Limit to avoid token overflow
            schema = self.registry.get_schema(name)
            if schema:
                tools_info.append(f"- {name}: {schema.description}")
        
        # MCP tools
        for tool in mcp_tools[:10]:
            tools_info.append(f"- {tool['name']}: {tool.get('description', '')[:100]}")
        
        tools_text = "\n".join(tools_info)
        
        return f"""You are a code generator. Generate Python code to accomplish the following task.

Task: {task}

Available tools (call with `await call_tool("name", {{"param": "value"}})`):
{tools_text}

Available functions in the execution environment:
- call_tool(name, arguments) -> dict: Call a tool and get result
- search_tools(query) -> list: Search for tools
- print(message): Log output (will be returned to user)
- json, datetime, math, re modules are available

Rules:
1. Use await for call_tool() calls
2. Handle errors gracefully
3. Print meaningful output for the user
4. Keep code simple and focused
5. Return results using print()

Generate only the Python code, no explanations:
```python
# Your code here
```"""
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Handle markdown code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Return as-is if no code blocks
        return response.strip()
    
    async def route_and_execute(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Route a query and execute the selected tool.
        
        This is the main entry point for processing user queries.
        Supports local tools, MCP tools, and code execution.
        
        Args:
            query: The user's query
            context: Optional context (user info, conversation history)
            
        Returns:
            Dict with:
            - 'tool': The tool that was used
            - 'params': Parameters that were extracted
            - 'result': The tool's result
            - 'response': Formatted response for the user
        """
        # GLOBAL SECURITY CHECK
        if self.input_guard:
            try:
                # Run scan in thread pool to avoid blocking async loop
                loop = asyncio.get_running_loop()
                scan_result = await loop.run_in_executor(None, self.input_guard.scan, query)
                
                if not scan_result.is_valid:
                    logger.warning(f"🚨 Global Input Guard blocked query: {query}")
                    return {
                        "tool": "blocked",
                        "params": {},
                        "result": {"success": False, "error": "Query blocked by security policy."},
                        "response": "I cannot answer that query as it violates our security policies.",
                        "mode": "security_block"
                    }
            except Exception as e:
                logger.error(f"Error in global input guard: {e}")
                # Fail open or closed? Fail open for now to avoid blocking valid queries on error
                pass

        # CODE-FIRST MODE: Use code execution as primary path
        if self.config.code_first_mode and self._code_executor:
            return await self._route_and_execute_code_first(query, context)
        
        # LEGACY MODE: Traditional routing
        return await self._route_and_execute_legacy(query, context)
    
    async def _route_and_execute_code_first(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Code-first execution mode.
        
        Instead of loading full tool schemas into context, we:
        1. Give LLM a minimal list of available tools
        2. LLM generates Python code to accomplish the task
        3. Code calls tools via call_tool() in sandbox
        
        This dramatically reduces context window usage.
        """
        # Check if this is a chat query (no tools needed)
        is_chat = await self._is_chat_query(query)
        if is_chat:
            # Pass conversation history string for context
            history_str = context.get("_conversation_history", "") if context else ""
            chat_result = await self.chat(query, history_str)
            response_text = chat_result.get("data", "") if isinstance(chat_result, dict) else str(chat_result)
            return {
                "tool": "chat",
                "params": {"message": query},
                "result": chat_result if isinstance(chat_result, dict) else {"success": True, "data": chat_result},
                "response": response_text,
                "mode": "code_first"
            }
        
        # Get minimal tool list (just names + descriptions)
        tool_list = self._get_minimal_tool_list()
        
        # Generate code
        code = await self._generate_tool_code(query, tool_list, context)
        
        if not code:
            # Fallback to chat
            history_str = context.get("_conversation_history", "") if context else ""
            chat_result = await self.chat(query, history_str)
            response_text = chat_result.get("data", "") if isinstance(chat_result, dict) else str(chat_result)
            return {
                "tool": "chat",
                "params": {"message": query},
                "result": chat_result if isinstance(chat_result, dict) else {"success": True, "data": chat_result},
                "response": response_text,
                "mode": "code_first_fallback"
            }
        
        # Execute the generated code with context (for tools that need conversation history)
        result = await self._code_executor.execute(
            code,
            timeout=self.config.code_execution_timeout,
            context=context  # Pass context for tools like survey that need conversation history
        )
        
        # Check if any tool returned an adaptive card - need to pass it to frontend
        adaptive_card_result = None
        tool_data_response = None  # Fallback response from tool data
        formatted_tool_response = None  # Response from tool's format_response method
        chat_image_data = None  # Image data from chat tool
        chat_related_questions = None  # Related questions from chat tool
        
        if result.tool_calls:
            for tc in result.tool_calls:
                tc_result = tc.get("result", {})
                data = tc_result.get("data", {})
                # Fix: code_executor uses "name" key, not "tool"
                tool_name = tc.get("name", tc.get("tool", ""))
                
                if isinstance(data, dict) and data.get("type") == "adaptive_card":
                    adaptive_card_result = data
                    break
                
                # Handle MCP tool responses - extract the "response" field directly
                if "." in tool_name and isinstance(data, dict) and "response" in data:
                    # This is an MCP tool (like secure_rag.secure_query)
                    # Use the response text directly, not the JSON
                    if not tool_data_response:
                        tool_data_response = data["response"]
                    continue
                
                # Extract chat tool extras (image_data, related_questions)
                if tool_name == "chat" and tc_result.get("success"):
                    chat_image_data = tc_result.get("image_data")
                    chat_related_questions = tc_result.get("related_questions")
                    if chat_image_data:
                        logger.info(f"📷 Chat tool returned image_data for: {chat_image_data.get('title', 'unknown')}")
                    if chat_related_questions:
                        logger.info(f"❓ Chat tool returned {len(chat_related_questions)} related questions")
                
                # Try to use the tool's format_response method for proper formatting
                if not formatted_tool_response and tc_result.get("success"):
                    # Get the tool from registry and use its formatter
                    if tool_name and "." not in tool_name:  # Local tool (not MCP)
                        tool = self.registry.get_tool(tool_name)
                        if tool:
                            if hasattr(tool, 'format_response'):
                                try:
                                    logger.info(f"Formatting response for tool: {tool_name}")
                                    formatted_tool_response = tool.format_response(tc_result)
                                except Exception as e:
                                    logger.warning(f"Tool format_response failed for {tool_name}: {e}")
                            else:
                                logger.debug(f"Tool {tool_name} has no format_response method")
                
                # Capture tool response data as fallback
                if not tool_data_response:
                    if isinstance(data, str) and data:
                        tool_data_response = data
                    elif tc_result.get("success") and isinstance(data, str):
                        tool_data_response = data
        
        # Format response - prefer tool's formatted response over LLM-generated print statements
        if result.success:
            logs = result.logs
            output = result.output
            # Priority: 1. Tool's format_response, 2. Logs, 3. Output, 4. Tool data, 5. Default
            if formatted_tool_response:
                response = formatted_tool_response
            else:
                response = "\n".join(logs) if logs else str(output) if output else tool_data_response or "✅ Task completed"
        else:
            response = f"❌ {result.error or 'Execution failed'}"
        
        # Build the return result
        return_result = {
            "tool": "code_execution",
            "params": {"query": query},
            "result": {
                "success": result.success,
                "output": result.output,
                "logs": result.logs,
                "error": result.error,
                "tool_calls": result.tool_calls,
                "execution_time": result.execution_time
            },
            "code": code,
            "response": response,
            "mode": "code_first"
        }
        
        # If we got an adaptive card, include it so the frontend can render it
        if adaptive_card_result:
            return_result["result"]["type"] = "adaptive_card"
            return_result["result"]["card"] = adaptive_card_result.get("card")
            return_result["result"]["message"] = adaptive_card_result.get("message", response)
            return_result["result"]["entities"] = adaptive_card_result.get("entities")
            return_result["result"]["metadata"] = adaptive_card_result.get("metadata", {})  # Include template info
            return_result["response"] = adaptive_card_result.get("message", response)
        
        # If we got chat extras (image_data, related_questions), include them
        if chat_image_data:
            return_result["result"]["image_data"] = chat_image_data
        if chat_related_questions:
            return_result["result"]["related_questions"] = chat_related_questions
        
        return return_result
    
    def _get_local_plugin_keywords(self) -> List[str]:
        """
        Get routing keywords for local plugins only.
        
        MCP servers are auto-routed through tool search, so they don't need
        explicit keywords here. Only local plugins need keyword bypass to
        prevent misrouting to chat.
        
        Returns a flat list of keywords for quick matching.
        """
        # Local plugin keywords (these tools are not MCP servers)
        local_plugin_keywords = {
            "news": [
                "news", "headlines", "breaking", "latest news", "current events",
                "happening in", "happening today", "news today", "news from",
                "news in", "news about"
            ],
            "weather": [
                "weather", "temperature", "forecast", "rain", "sunny", "cloudy",
                "humid", "wind", "storm", "snow", "climate"
            ],
            "survey": [
                "survey", "feedback", "questionnaire", "poll"
            ]
        }
        
        all_keywords = []
        for keywords in local_plugin_keywords.values():
            all_keywords.extend(keywords)
        
        return all_keywords
    
    async def _is_chat_query(self, query: str) -> bool:
        """
        Quick check if query is just chat (no tools needed).
        
        Detects greetings and knowledge questions that should use ChatTool
        for rich responses (images, related questions).
        
        IMPORTANT: First checks for tool-specific keywords to avoid
        misrouting queries like "what is the news" to chat.
        
        Keywords are loaded dynamically from:
        1. MCP server configs (routing_keywords in mcp_servers.json)
        2. Local plugin keywords (hardcoded for non-MCP tools)
        """
        query_lower = query.lower().strip()
        
        # =====================================================================
        # STEP 1: Check for local plugin keywords FIRST (exclusions from chat)
        # MCP servers are auto-routed, only local plugins need keyword bypass
        # =====================================================================
        plugin_keywords = self._get_local_plugin_keywords()
        
        for kw in plugin_keywords:
            if kw in query_lower:
                return False  # Route to LLM for tool selection
        
        # =====================================================================
        # STEP 2: Check for obvious chat patterns (greetings/farewells)
        # =====================================================================
        obvious_chat_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "thanks", "thank you", "bye", "goodbye", "see you",
            "what's up", "howdy", "greetings"
        ]
        
        for pattern in obvious_chat_patterns:
            if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.startswith(pattern + ","):
                return True
        
        # =====================================================================
        # STEP 3: REMOVED - Knowledge patterns check
        # We want "what is...", "who is..." questions to go to the LLM router
        # so it can intelligently decide between Knowledge Base, SQL, or Chat.
        # =====================================================================
        
        # Everything else goes to LLM for intelligent routing
        return False
    
    def _get_minimal_tool_list(self) -> str:
        """Get minimal tool descriptions for code generation prompt."""
        lines = ["[Local Plugins]"]
        
        # Local plugins with required params shown
        for name in self.registry.list_tools():
            schema = self.registry.get_schema(name)
            if schema:
                desc = schema.description.split('.')[0][:80]
                # Show required parameters for clarity
                required = schema.parameters.get("required", [])
                if required:
                    params_hint = ", ".join(required[:2])
                    lines.append(f"  - {name}({{{params_hint}}}): {desc}")
                else:
                    lines.append(f"  - {name}: {desc}")
        
        # MCP tools (if available)
        if self._mcp_host and self._mcp_host.total_tools > 0:
            lines.append("\n[MCP Servers]")
            for tool in self._mcp_host.list_all_tools(detail_level="brief"):
                name = tool["name"]
                desc = tool.get("description", "")[:80]
                lines.append(f"  - {name}: {desc}")
        
        return "\n".join(lines)
    
    def _get_tool_metadata_for_tools(self, tool_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get distributed prompting metadata for a list of tools.
        
        This implements the "Search-Then-Teach" pattern:
        - Only load heavy instructions for tools that matched the query
        - Each tool can have system_instruction and code_example in mcp_servers.json
        
        Args:
            tool_names: List of tool names (can be local or MCP format "server.tool")
            
        Returns:
            Tuple of (system_instructions, code_examples) lists
        """
        system_instructions = []
        code_examples = []
        
        for tool_name in tool_names:
            # Try to get metadata from MCP host
            if self._mcp_host and "." in tool_name:
                metadata = self._mcp_host.get_tool_metadata(tool_name)
                if metadata:
                    if metadata.get("system_instruction"):
                        system_instructions.append(f"# {tool_name}:\n{metadata['system_instruction']}")
                    if metadata.get("code_example"):
                        code_examples.append(f"# {tool_name}:\n{metadata['code_example']}")
            
            # Local plugins could also have metadata in the future
            # For now, we only support MCP tools with distributed prompting
        
        return system_instructions, code_examples
    
    async def _generate_tool_code(
        self,
        query: str,
        tool_list: str,
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """Generate Python code to accomplish the task using Search-Then-Teach pattern."""
        
        # =====================================================================
        # STEP 1: Search for relevant tools first (Search phase)
        # =====================================================================
        matched_mcp_tools = []
        if self._mcp_host and self._mcp_host.connected_servers > 0:
            try:
                mcp_results = await self._mcp_host.search_tools(query, max_results=5)
                matched_mcp_tools = [f"{r.server_name}.{r.tool.name}" for r in mcp_results]
            except Exception as e:
                logger.warning(f"MCP tool search for metadata failed: {e}")
        
        # =====================================================================
        # STEP 2: Get metadata only for matched tools (Teach phase)
        # =====================================================================
        dynamic_instructions, dynamic_examples = self._get_tool_metadata_for_tools(matched_mcp_tools)
        
        # Build dynamic sections
        specific_tool_rules = ""
        if dynamic_instructions:
            specific_tool_rules = "\nSPECIFIC TOOL RULES (for matched tools):\n" + "\n".join(dynamic_instructions)
        
        dynamic_examples_section = ""
        if dynamic_examples:
            dynamic_examples_section = "\nDYNAMIC EXAMPLES (from tool metadata):\n" + "\n".join(dynamic_examples)
        
        # Include conversation history if available
        conversation_history = ""
        if context:
            history = context.get("_conversation_history", "")
            if history:
                conversation_history = f"""
CONVERSATION HISTORY (use this to understand context of follow-up questions):
{history}

"""
        
        # Extract user info from context for secure_rag
        user_id = "guest"
        user_role = "user"
        if context:
            user_id = context.get("employee_id", "guest")
            user_role = context.get("role", "user")
        
        # =====================================================================
        # STEP 3: Build prompt with dynamic metadata injection
        # =====================================================================
        prompt = f"""You are a code generator. Generate Python code to answer the user's query.
{conversation_history}
USER QUERY: {query}

CURRENT USER INFO (use these values for secure_rag.secure_query):
- user_id: "{user_id}"
- user_role: "{user_role}"

AVAILABLE TOOLS:
{tool_list}

HOW TO CALL TOOLS:
- Local plugin: result = await call_tool("tool_name", {{"param": "value"}})
- MCP server: result = await call_tool("server.tool_name", {{"param": "value"}})

IMPORTANT: For 'secure_rag.secure_query', you MUST use the CURRENT USER INFO above.
Example: result = await call_tool("secure_rag.secure_query", {{"query": "...", "user_id": "{user_id}", "user_role": "{user_role}"}})

The result is a dict with:
- result["success"]: True/False  
- result["data"]: The tool's output (already a dict or string)
- result["error"]: Error message if failed

PRE-IMPORTED MODULES (DO NOT ADD IMPORT STATEMENTS):
- json (use json.loads(), json.dumps())
- datetime, math, re, collections
{specific_tool_rules}

GENERAL TOOL SELECTION RULES:
1. For leave/vacation/time-off/PTO requests: Use "leave.analyze_leave_request" with the user's query
2. For weather queries: Use "weather" tool with city parameter
3. For BUDGET SPEECH, TAX STRUCTURE, TAX SLABS, GOVERNMENT POLICIES, FINANCE MINISTER ANNOUNCEMENTS: Use "knowledgebase.knowledgebase_query" - NOT chat, NOT sql
4. For DATABASE queries (sales, products, customers, employees, orders, inventory, revenue): Use "sql.sql_query"
5. For news: Use "news" tool
6. For surveys/feedback/questionnaires: Use "survey" tool with action="start"
7. For collecting user info (name, phone, address): Use "contact_form" with action="start"
8. For showing stored user info: Use "contact_form" with action="show"
9. For general greetings, small talk, general questions NOT about budget/database: Use "chat" tool
10. If result["data"] has type="adaptive_card", just print the message field - the card will be shown automatically
11. If this is a follow-up to a previous leave request (check conversation history), include relevant context in the query
12. IMPORTANT: Questions about "tax structure", "tax brackets", "tax slabs", "budget allocations", "revised tax" -> knowledgebase.knowledgebase_query
13. IMPORTANT: Questions about "sales data", "products", "customers", "employees", "orders", "top selling" -> sql.sql_query
14. IMPORTANT: For secure_rag.secure_query ONLY use for: security threats, malicious content, weapons, bombs, hacking, confidential data access. Do NOT use for general questions about uploaded files.
15. IMPORTANT: For questions about uploaded files, documents, images, PDFs, summaries of uploaded content: Use "file" tool with action="followup" and question parameter
16. IMPORTANT: If the user asks "where does she work", "what is her role", "who is this person" after uploading a file, use "file" tool NOT secure_rag

CODE EXAMPLES:
{dynamic_examples_section}

# Fallback examples for common tools:
```python
# Weather example
result = await call_tool("weather", {{"city": "Mumbai"}})
if result["success"]:
    # The system will automatically format the rich output (emojis, details)
    # Just print a confirmation or the raw data
    print(result["data"])
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# File tool example - for uploaded files, documents, images
# Use this when user asks about uploaded content, NOT secure_rag
result = await call_tool("file", {{"action": "followup", "question": "Where does she work?"}})
if result["success"]:
    print(result["data"])
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# Chat example - for questions, greetings, current events, web searches
# IMPORTANT: The chat tool requires 'message' parameter, NOT 'query'
result = await call_tool("chat", {{"message": "Who won last week's F1 race?"}})
if result["success"]:
    print(result["data"])
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# Adaptive Card example - for dashboards, forms, profiles, notifications
# IMPORTANT: ALWAYS pass the 'description' parameter - it's REQUIRED!
result = await call_tool("adaptive_card", {{
    "description": "Create a quarterly business performance dashboard with sales metrics, revenue charts, and KPIs",
    "tone": "professional",
    "card_type": "dashboard"
}})
if result["success"]:
    data = result["data"]
    print(data.get("message", "Here is your adaptive card."))
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# News example - supports regions/countries via 'query' or 'country' parameter
# For specific regions like "Riyadh", "India", "Saudi Arabia" - use query parameter
# For country codes like "us", "in", "sa", "gb" - use country parameter
result = await call_tool("news", {{"query": "Saudi Arabia", "max_results": 5}})
if result["success"]:
    data = result["data"]
    articles = data.get("articles", [])
    if articles:
        messages = []
        for i, article in enumerate(articles[:5], 1):
            title = article.get('title', 'No title')
            url = article.get('url', '#')
            source = article.get('source', 'Unknown')
            description = article.get('description', '')
            # Build article message with clickable blue title
            msg = f"{{i}}. <a href='{{url}}' target='_blank' style='color: #29b6f6; font-weight: bold; text-decoration: none;'>{{title}}</a>"
            if description:
                desc_short = description[:120] + "..." if len(description) > 120 else description
                msg += f"\\n{{desc_short}}"
            msg += f"\\nSource: {{source}}"
            messages.append(msg)
        # Print each article separated by MESSAGE_SPLIT for separate bubbles
        print("\\n---MESSAGE_SPLIT---\\n".join(messages))
    else:
        print("No articles found for this query.")
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# Survey example - starts a conversational survey based on previous conversation
# The survey tool generates questions dynamically and collects responses one at a time
result = await call_tool("survey", {{"action": "start", "topic": "feedback", "num_questions": 3}})
if result["success"]:
    # The tool returns "data" field with the response message
    print(result.get("data", "Survey started!"))
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# Contact form example - collect user info through back-and-forth conversation
# Use action="start" to begin collecting name, phone, address
# Use action="show" to display stored info as an adaptive card
result = await call_tool("contact_form", {{"action": "start"}})
if result["success"]:
    print(result.get("data", "Let's collect your info!"))
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# Knowledge Base example - for BUDGET SPEECH, TAX STRUCTURE, TAX SLABS, GOVERNMENT POLICIES
# Use for: "What is the revised tax structure?", "tax brackets", "budget allocation for education"
# NEVER use chat or sql for these questions!
result = await call_tool("knowledgebase.knowledgebase_query", {{"query": "What is the revised tax structure?", "state_id": "session123"}})
if result["success"]:
    data = json.loads(result["data"]) if isinstance(result["data"], str) else result["data"]
    answer = data.get("answer", "No answer found")
    # Chart is automatically displayed by frontend - DO NOT print path
    print(answer)
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# SQL Database example - for SALES, PRODUCTS, CUSTOMERS, EMPLOYEES, ORDERS queries
# Use for: "top selling products", "customer list", "sales by month", "employee count"
# NEVER use for budget/tax questions - those go to knowledgebase!
result = await call_tool("sql.sql_query", {{"query": "Show me top 5 products by sales", "state_id": "session123"}})
if result["success"]:
    data = json.loads(result["data"]) if isinstance(result["data"], str) else result["data"]
    answer = data.get("answer", "Query completed")
    # Chart is automatically displayed by frontend - DO NOT print path
    print(answer)
else:
    print(f"Error: {{result.get('error')}}")
```

```python
# Secure RAG example - for employee info queries, security checks, sensitive data
# Access is restricted by user role - admins see everything, employees see limited info
# USE THE user_id AND user_role VALUES FROM "CURRENT USER INFO" SECTION ABOVE
result = await call_tool("secure_rag.secure_query", {{"query": "What is Akash's role?", "user_id": "{user_id}", "user_role": "{user_role}"}})
# IMPORTANT: Print ONLY result["data"]["response"] - NOT the full dict!
data = result["data"]
if isinstance(data, dict):
    print(data.get("response", "No response"))
else:
    print(data)
```

RULES:
1. DO NOT include any import statements - modules are already available
2. Use `await call_tool()` to call tools
3. Handle errors with if/else on result["success"]
4. result["data"] is ALREADY a dict - do NOT use json.loads() on it
5. Use print() to output the final answer for the user
6. Keep code simple - usually just 1-2 tool calls
7. Format output nicely for humans
8. For ANY leave/vacation/time-off/PTO request, use leave.analyze_leave_request with the user's query
9. For chat tool: ALWAYS use {{"message": "..."}} NOT {{"query": "..."}}
10. For adaptive_card tool: ALWAYS include "description" parameter with the user's request - it's REQUIRED
11. For contact info collection: use contact_form with action="start", for viewing: action="show"
12. For TAX/BUDGET questions: ALWAYS use knowledgebase.knowledgebase_query, NEVER chat or sql
13. For DATABASE questions (sales, products, customers): ALWAYS use sql.sql_query, NEVER knowledgebase

Generate ONLY the Python code (no imports, no explanations):
```python
"""

        try:
            response = await self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.code_model,
                temperature=0.2,
                max_tokens=1000,
                is_small_request=True,  # Code snippets are small
                trace_name="router-code-snippet"
            )
            
            code = self._extract_code(response)
            
            # Remove any import statements (safety net)
            lines = code.split('\n')
            lines = [line for line in lines if not line.strip().startswith('import ') and not line.strip().startswith('from ')]
            code = '\n'.join(lines)
            
            logger.debug(f"Generated code for: {query[:50]}...\n{code}")
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    async def _route_and_execute_legacy(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Legacy routing mode (traditional tool selection).
        """
        # Route the query
        decision = await self.route(query, context)
        
        tool_name = decision.get("tool", "chat")
        params = decision.get("params", {})
        
        # Handle special modes
        if tool_name == "code_execution":
            # Execute with code generation
            task = params.get("task", query)
            result = await self.execute_with_code(task, context)
            
            # Format response
            if result.get("success"):
                logs = result.get("logs", [])
                output = result.get("output")
                response = "\n".join(logs) if logs else str(output) if output else "✅ Task completed"
            else:
                response = f"❌ {result.get('error', 'Code execution failed')}"
            
            return {
                "tool": "code_execution",
                "params": params,
                "result": result,
                "response": response,
                "mode": "legacy",
                "routing": {
                    "confidence": decision.get("confidence"),
                    "reasoning": decision.get("reasoning")
                }
            }
        
        # Execute the tool (local or MCP)
        result = await self.execute_tool(tool_name, params)
        
        # Format the response
        if "." in tool_name:
            # MCP tool - format result
            if result.get("success"):
                data = result.get("data", {})
                # Special handling for secure_rag - return just the response text
                if isinstance(data, dict) and "response" in data:
                    response = data["response"]
                elif isinstance(data, dict):
                    response = json.dumps(data, indent=2)
                else:
                    response = str(data)
            else:
                response = f"❌ {result.get('error', 'An error occurred')}"
        else:
            # Local tool
            tool = self.registry.get_tool(tool_name)
            if tool and result.get("success", False):
                response = tool.format_response(result)
            elif not result.get("success", False):
                response = f"❌ {result.get('error', 'An error occurred')}"
            else:
                response = str(result.get("data", ""))
        
        return {
            "tool": tool_name,
            "params": params,
            "result": result,
            "response": response,
            "mode": "legacy",
            "routing": {
                "confidence": decision.get("confidence"),
                "reasoning": decision.get("reasoning")
            }
        }
    
    async def chat(self, message: str, conversation_history: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle general chat using ChatTool plugin for rich responses.
        
        Args:
            message: The user's message
            conversation_history: Optional conversation history string from Redis
            
        Returns:
            Dict with 'success', 'data', 'image_data', 'related_questions'
        """
        # Use ChatTool from registry for full functionality (images, related questions)
        chat_tool = self.registry.get_tool("chat")
        if chat_tool:
            try:
                result = await chat_tool.safe_execute(
                    message=message,
                    conversation_history=conversation_history
                )
                return result
            except Exception as e:
                logger.error(f"ChatTool error: {e}")
        
        # Fallback to simple LLM if ChatTool unavailable
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Be concise and helpful."
            }
        ]
        
        # Add conversation history as context if available
        if conversation_history:
            messages.append({
                "role": "system",
                "content": f"Previous conversation context:\n{conversation_history}"
            })
        
        messages.append({"role": "user", "content": message})
        
        try:
            response = await self._call_llm(
                messages=messages,
                model=self.config.chat_model,
                temperature=0.7,
                max_tokens=1000,
                is_small_request=False,  # Chat can have longer context
                trace_name="router-chat-fallback"
            )
            return {"success": True, "data": response}
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"success": False, "data": "I'm sorry, I'm having trouble responding right now."}
    
    async def execute_confirmed_action(
        self,
        pending_action: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute a previously confirmed action.
        
        This is called when a user confirms a pending permission request.
        The pending_action contains the tool and params that were awaiting confirmation.
        
        Args:
            pending_action: Dict with 'tool' and 'params' from the pending request
            context: Additional context
            
        Returns:
            Execution result
        """
        tool_name = pending_action.get("tool", "")
        params = pending_action.get("params", {})
        
        logger.info(f"✅ Executing confirmed action: {tool_name}")
        
        return await self.execute_tool(tool_name, params)
    
    async def close(self) -> None:
        """Close the HTTP client and MCP connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        if self._mcp_host:
            await self._mcp_host.disconnect_all()
            self._mcp_host = None
            self._mcp_initialized = False
        
        logger.info("🔌 Router closed")


# Synchronous wrapper for non-async contexts
class SyncLLMRouter:
    """
    Synchronous wrapper for LLMRouter.
    
    Use this when you need to call the router from synchronous code
    (e.g., Flask routes without async support).
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self._async_router = LLMRouter(config)
    
    def _run_async(self, coro):
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # We're in an async context, use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    
    def initialize(self) -> None:
        """Initialize MCP connections."""
        return self._run_async(self._async_router.initialize())
    
    def route(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous route method."""
        return self._run_async(self._async_router.route(query, context))
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous execute_tool method."""
        return self._run_async(self._async_router.execute_tool(tool_name, params))
    
    def route_and_execute(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous route_and_execute method."""
        return self._run_async(self._async_router.route_and_execute(query, context))
    
    def execute_with_code(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous execute_with_code method."""
        return self._run_async(self._async_router.execute_with_code(task, context))
    
    def chat(self, message: str, conversation_history: Optional[str] = None) -> str:
        """Synchronous chat method."""
        return self._run_async(self._async_router.chat(message, conversation_history))
    
    def close(self) -> None:
        """Close the router."""
        self._run_async(self._async_router.close())
    
    @property
    def mcp_host(self) -> Optional[MCPHost]:
        """Get the MCP host instance."""
        return self._async_router.mcp_host


# Global convenience functions
_router_instance: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    """Get the global LLMRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


def get_sync_router() -> SyncLLMRouter:
    """Get a synchronous router instance."""
    return SyncLLMRouter()