"""
Code Executor - Sandboxed execution environment for LLM-generated code.

Provides a safe execution environment where LLM-generated code can:
- Call MCP tools via call_tool("server.tool", args)
- Call local plugins via call_tool("weather", args)  
- Search for tools via search_tools()
- Print output (captured and returned)
- Access filesystem in a sandboxed workspace

This is the PRIMARY execution path for all tool calls - both MCP and local plugins.
By using code execution, we avoid clogging the context window with tool schemas.
The LLM only sees minimal tool descriptions and generates code to orchestrate calls.

Security features:
- Timeout enforcement
- Restricted imports
- Sandboxed filesystem access
- Resource limits

Based on: https://www.anthropic.com/engineering/code-execution-with-mcp
See also: https://www.anthropic.com/engineering/claude-code-sandboxing
"""

import asyncio
import ast
import io
import logging
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import time

from architecture.telemetry import log_llm_event, trace_llm_call

if TYPE_CHECKING:
    from architecture.mcp_host import MCPHost

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    variables: Dict[str, Any] = field(default_factory=dict)
    # Track which tools were called
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


# Allowed built-in modules (safe subset)
ALLOWED_MODULES = {
    'json', 'math', 'datetime', 'time', 're', 'collections',
    'itertools', 'functools', 'operator', 'string', 'random',
    'hashlib', 'base64', 'urllib.parse', 'pathlib', 'typing',
    'dataclasses', 'enum', 'uuid', 'decimal', 'fractions',
    'statistics', 'copy', 'pprint', 'textwrap', 'difflib',
}

# Blocked patterns in code (security)
BLOCKED_PATTERNS = [
    'import os',
    'import sys', 
    'import subprocess',
    'import socket',
    'import shutil',
    '__import__',
    'eval(',
    'exec(',
    'compile(',
    'open(',  # Use sandboxed file access instead
    'globals(',
    'locals(',
    'getattr(',
    'setattr(',
    'delattr(',
    '__builtins__',
    '__class__',
    '__bases__',
    '__subclasses__',
    '__code__',
    '__globals__',
]


class SandboxedPrint:
    """Capture print output."""
    
    def __init__(self):
        self.logs: List[str] = []
    
    def __call__(self, *args, **kwargs):
        output = ' '.join(str(arg) for arg in args)
        self.logs.append(output)
        # Also print to actual stdout for debugging
        # print(f"[SANDBOX] {output}")


class SandboxedFileSystem:
    """Sandboxed filesystem access within workspace."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace = workspace_dir
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path within sandbox."""
        resolved = (self.workspace / path).resolve()
        # Ensure path is within workspace
        if not str(resolved).startswith(str(self.workspace.resolve())):
            raise PermissionError(f"Access denied: {path} is outside workspace")
        return resolved
    
    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        full_path = self._resolve_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_text()
    
    def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    def list_dir(self, path: str = ".") -> List[str]:
        """List directory contents."""
        full_path = self._resolve_path(path)
        if not full_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return [p.name for p in full_path.iterdir()]
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        try:
            return self._resolve_path(path).exists()
        except PermissionError:
            return False


class AsyncRunner:
    """Helper to run async functions in the executor context."""
    
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.results: Dict[str, Any] = {}
    
    def run(self, coro, name: str = "result") -> Any:
        """Run a coroutine and store result."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        result = future.result(timeout=30.0)
        self.results[name] = result
        return result


class CodeExecutor:
    """
    Sandboxed code execution environment.
    
    This is the PRIMARY execution engine for all tools - both MCP servers and local plugins.
    By using code execution, we avoid clogging the context window with tool schemas.
    
    Features:
    - Execute Python code safely
    - Call local plugins via call_tool("weather", args)
    - Call MCP tools via call_tool("server.tool", args)
    - Unified interface - the code doesn't need to know if it's MCP or local
    - Capture stdout/stderr
    - Enforce timeouts
    - Restrict dangerous operations
    
    Usage:
        executor = CodeExecutor(mcp_host, registry)
        result = await executor.execute('''
            # Call local plugin
            weather = await call_tool("weather", {"city": "London"})
            print(f"Temperature: {weather['data']['temp']}°C")
            
            # Call MCP server tool
            forecast = await call_tool("weather_server.get_forecast", {"city": "London", "days": 5})
            print(f"Forecast: {forecast}")
        ''')
    """
    
    def __init__(
        self,
        mcp_host: "MCPHost",
        registry: Optional[Any] = None,
        workspace_dir: Optional[Path] = None
    ):
        self.mcp_host = mcp_host
        self._registry = registry  # Local plugin registry
        self.workspace = workspace_dir or Path("workspace")
        self.fs = SandboxedFileSystem(self.workspace)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tool_calls: List[Dict[str, Any]] = []  # Track tool calls for debugging
    
    @property
    def registry(self):
        """Lazy load registry to avoid circular imports."""
        if self._registry is None:
            from architecture.registry import get_registry
            self._registry = get_registry()
        return self._registry
    
    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for security issues.
        
        Returns error message if code is unsafe, None if OK.
        """
        # Check for blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if pattern in code:
                return f"Blocked pattern detected: {pattern}"
        
        # Try to parse as valid Python
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"
        
        return None
    
    def _create_safe_globals(
        self,
        printer: SandboxedPrint,
        allowed_servers: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a restricted globals dict for execution."""
        
        # Track tool calls for debugging
        tool_call_log = self._tool_calls
        
        # Context to inject into tool calls (conversation history, sender_id, etc.)
        injected_context = context or {}
        
        # Create unified async tool caller (handles both local plugins AND MCP tools)
        async def call_tool(name: str, arguments: Optional[Dict] = None) -> Dict[str, Any]:
            """
            Call a tool - works for both local plugins and MCP servers.
            
            For local plugins: call_tool("weather", {"city": "London"})
            For MCP servers: call_tool("weather_server.get_current", {"city": "London"})
            
            The function automatically detects which type based on whether
            there's a "." in the name.
            
            Context (conversation history, sender_id) is automatically injected.
            """
            arguments = arguments or {}
            
            # Inject context into arguments for tools that need it
            # This provides conversation history, sender_id, etc. to tools like survey
            augmented_args = {**arguments}
            if injected_context:
                # Add context with underscore prefix to avoid conflicts
                if "_conversation_history" not in augmented_args and injected_context.get("_conversation_history"):
                    augmented_args["_conversation_history"] = injected_context["_conversation_history"]
                if "_sender_id" not in augmented_args and injected_context.get("_sender_id"):
                    augmented_args["_sender_id"] = injected_context["_sender_id"]
                if "sender_id" not in augmented_args and injected_context.get("_sender_id"):
                    augmented_args["sender_id"] = injected_context["_sender_id"]
                # Inject user_id if available (for secure_rag)
                if "user_id" not in augmented_args and injected_context.get("user_id"):
                    augmented_args["user_id"] = injected_context["user_id"]
                # Inject language info (for response formatting / bilingual behavior)
                if "original_language" not in augmented_args and injected_context.get("original_language"):
                    augmented_args["original_language"] = injected_context["original_language"]
                if "text_direction" not in augmented_args and injected_context.get("text_direction"):
                    augmented_args["text_direction"] = injected_context["text_direction"]
            
            call_record = {"name": name, "arguments": arguments, "timestamp": time.time()}
            
            # Use trace_llm_call for tool call telemetry with I/O
            with trace_llm_call(
                name="tool-call",
                model="tool-executor",
                input_data={"tool": name, "arguments": arguments},
                metadata={"source": "CodeExecutor.call_tool", "has_context": bool(injected_context)}
            ) as tool_trace:
                try:
                    # Check if it's an MCP tool (has "." separator) or local plugin
                    if "." in name:
                        # MCP server tool
                        if allowed_servers:
                            server_name = name.split(".")[0]
                            if server_name not in allowed_servers:
                                result = {"success": False, "error": f"Server '{server_name}' not allowed"}
                                call_record["result"] = result
                                tool_call_log.append(call_record)
                                tool_trace.update(output=str(result), metadata={"success": False, "reason": "server_not_allowed"})
                                return result
                        
                        result = await self.mcp_host.call_tool(name, augmented_args)
                        call_record["result"] = result
                        call_record["source"] = "mcp"
                        tool_call_log.append(call_record)
                        tool_trace.update(output=str(result), metadata={"success": result.get("success", True), "source": "mcp"})
                        return result
                    else:
                        # Local plugin tool
                        tool = self.registry.get_tool(name)
                        if tool is None:
                            # Maybe it's an MCP tool without server prefix - search for it
                            mcp_tools = self.mcp_host.list_all_tools(detail_level="name")
                            matching = [t["name"] for t in mcp_tools if t["name"].endswith(f".{name}")]
                            if matching:
                                # Found matching MCP tool, call it
                                result = await self.mcp_host.call_tool(matching[0], augmented_args)
                                call_record["result"] = result
                                call_record["source"] = "mcp"
                                call_record["resolved_name"] = matching[0]
                                tool_call_log.append(call_record)
                                tool_trace.update(output=str(result), metadata={"success": result.get("success", True), "source": "mcp", "resolved": matching[0]})
                                return result
                            
                            result = {"success": False, "error": f"Tool '{name}' not found"}
                            call_record["result"] = result
                            tool_call_log.append(call_record)
                            tool_trace.update(output=str(result), metadata={"success": False, "reason": "not_found"})
                            return result
                        
                        # Execute local plugin with injected context
                        result = await tool.safe_execute(**augmented_args)
                        call_record["result"] = result
                        call_record["source"] = "local"
                        tool_call_log.append(call_record)
                        tool_trace.update(output=str(result), metadata={"success": result.get("success", True), "source": "local"})
                        return result
                        
                except Exception as e:
                    result = {"success": False, "error": str(e)}
                    call_record["result"] = result
                    call_record["exception"] = str(e)
                    tool_call_log.append(call_record)
                    tool_trace.update(output=str(result), error=str(e), metadata={"success": False})
                    return result
        
        async def search_tools(query: str, max_results: int = 5) -> List[Dict]:
            """
            Search for tools across both local plugins and MCP servers.
            Returns a unified list with source information.
            """
            results = []
            
            # Search local plugins
            local_matches = self.registry.search_tools(query, max_results=max_results)
            for name in local_matches:
                schema = self.registry.get_schema(name)
                if schema:
                    results.append({
                        "name": name,
                        "description": schema.description,
                        "source": "local",
                        "score": 1.0  # Local matches are prioritized
                    })
            
            # Search MCP tools
            mcp_results = await self.mcp_host.search_tools(query, max_results=max_results)
            for r in mcp_results:
                results.append({
                    "name": f"{r.server_name}.{r.tool.name}",
                    "description": r.tool.description,
                    "source": "mcp",
                    "score": r.relevance_score
                })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:max_results]
        
        def get_tool_definition(name: str) -> Optional[Dict]:
            """Get full definition for a tool (local or MCP)."""
            if "." in name:
                # MCP tool
                return self.mcp_host.get_tool_definition(name)
            else:
                # Local plugin
                schema = self.registry.get_schema(name)
                if schema:
                    return schema.to_full_schema()
                return None
        
        def list_servers() -> List[str]:
            """List connected MCP servers."""
            return self.mcp_host.list_servers()
        
        def list_local_tools() -> List[Dict]:
            """List all local plugin tools."""
            return [
                {"name": name, "description": self.registry.get_schema(name).description}
                for name in self.registry.list_tools()
            ]
        
        def list_mcp_tools(server: Optional[str] = None) -> List[Dict]:
            """List MCP server tools."""
            tools = self.mcp_host.list_all_tools(detail_level="brief")
            if server:
                tools = [t for t in tools if t["name"].startswith(f"{server}.")]
            return tools
        
        async def call_tools_parallel(tool_calls: List[Dict]) -> List[Dict]:
            """
            Call multiple tools in parallel.
            
            This is useful when you need to query multiple MCP servers simultaneously,
            such as searching both KB and SQL databases for comprehensive answers.
            
            Args:
                tool_calls: List of dicts with 'tool' and 'arguments' keys
                
            Returns:
                List of results in the same order as input
                
            Example:
                results = await call_tools_parallel([
                    {"tool": "knowledgebase.knowledgebase_query", "arguments": {"query": "budget info"}},
                    {"tool": "sql.sql_query", "arguments": {"query": "show sales data"}}
                ])
                kb_result, sql_result = results
            """
            return await self.mcp_host.call_tools_parallel(tool_calls)
        
        def list_all_tools() -> List[Dict]:
            """List all available tools (both local and MCP)."""
            tools = []
            
            # Local plugins
            for name in self.registry.list_tools():
                schema = self.registry.get_schema(name)
                tools.append({
                    "name": name,
                    "description": schema.description if schema else "",
                    "source": "local"
                })
            
            # MCP tools
            for tool in self.mcp_host.list_all_tools(detail_level="brief"):
                tool["source"] = "mcp"
                tools.append(tool)
            
            return tools
        
        # Safe builtins
        safe_builtins = {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'bytes': bytes,
            'callable': callable,
            'chr': chr,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hash': hash,
            'hex': hex,
            'id': id,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': printer,  # Sandboxed print
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
            # Exception types
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'RuntimeError': RuntimeError,
        }
        
        # Import allowed modules
        allowed_imports = {}
        for mod_name in ALLOWED_MODULES:
            try:
                if '.' in mod_name:
                    parts = mod_name.split('.')
                    mod = __import__(mod_name)
                    for part in parts[1:]:
                        mod = getattr(mod, part)
                    allowed_imports[parts[-1]] = mod
                else:
                    allowed_imports[mod_name] = __import__(mod_name)
            except ImportError:
                pass
        
        return {
            '__builtins__': safe_builtins,
            # Unified tool functions (work for both local and MCP)
            'call_tool': call_tool,
            'call_tools_parallel': call_tools_parallel,  # Parallel execution across servers
            'search_tools': search_tools,
            'get_tool_definition': get_tool_definition,
            # Listing functions
            'list_servers': list_servers,
            'list_local_tools': list_local_tools,
            'list_mcp_tools': list_mcp_tools,
            'list_all_tools': list_all_tools,
            # Filesystem (sandboxed)
            'fs': self.fs,
            'read_file': self.fs.read_file,
            'write_file': self.fs.write_file,
            'list_dir': self.fs.list_dir,
            'file_exists': self.fs.exists,
            # Provide asyncio for async code
            'asyncio': asyncio,
            # Allowed modules
            **allowed_imports,
        }
    
    async def execute(
        self,
        code: str,
        timeout: float = 30.0,
        allowed_servers: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.
        
        This is the PRIMARY execution path for all tool calls.
        The code can call both local plugins and MCP tools via call_tool().
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time
            allowed_servers: Optional whitelist of allowed MCP servers
            context: Optional context dict with _conversation_history, _sender_id, etc.
                     This context is automatically injected into tool calls.
            
        Returns:
            ExecutionResult with output, logs, tool_calls, and any errors
        """
        start_time = time.time()
        
        # Clear tool call log for this execution
        self._tool_calls = []
        
        # Validate code
        error = self._validate_code(code)
        if error:
            return ExecutionResult(
                success=False,
                error=f"Code validation failed: {error}",
                execution_time=time.time() - start_time
            )
        
        # Create execution environment with context for tool calls
        printer = SandboxedPrint()
        globals_dict = self._create_safe_globals(printer, allowed_servers, context)
        locals_dict: Dict[str, Any] = {}
        
        # Use trace_llm_call for code execution telemetry with I/O
        with trace_llm_call(
            name="code-execution",
            model="code-executor",
            input_data={"code": code, "timeout": timeout, "has_context": bool(context)},
            metadata={"source": "CodeExecutor.execute"}
        ) as trace:
            try:
                # Wrap code in async function to support await
                # Note: We create the async function and execute it directly
                wrapped_code = f"""
async def __executor_main__():
{self._indent_code(code)}
    return None
"""
                
                # Execute the wrapper definition
                exec(wrapped_code, globals_dict, locals_dict)
                
                # Get the async function and run it with timeout
                main_func = locals_dict['__executor_main__']
                
                result = await asyncio.wait_for(
                    main_func(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                # Build output summary
                output_summary = {
                    "success": True,
                    "result": str(result) if result else None,
                    "logs": printer.logs,
                    "tool_calls_count": len(self._tool_calls),
                    "execution_time": execution_time
                }
                
                trace.update(
                    output=str(output_summary),
                    metadata={"success": True, "tool_calls_count": len(self._tool_calls), "logs_count": len(printer.logs)}
                )
                
                return ExecutionResult(
                    success=True,
                    output=result,
                    logs=printer.logs,
                    execution_time=execution_time,
                    variables={k: v for k, v in locals_dict.items() 
                              if not k.startswith('_')},
                    tool_calls=self._tool_calls.copy()
                )
                
            except asyncio.TimeoutError:
                trace.update(
                    output=f"Timeout after {timeout}s",
                    error=f"Execution timed out after {timeout}s",
                    metadata={"success": False, "timeout": True, "tool_calls_count": len(self._tool_calls)}
                )
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout}s",
                    logs=printer.logs,
                    execution_time=timeout,
                    tool_calls=self._tool_calls.copy()
                )
            except Exception as e:
                trace.update(
                    output=f"Error: {type(e).__name__}: {str(e)}",
                    error=str(e),
                    metadata={"success": False, "error_type": type(e).__name__, "tool_calls_count": len(self._tool_calls)}
                )
                return ExecutionResult(
                    success=False,
                    error=f"{type(e).__name__}: {str(e)}",
                    logs=printer.logs,
                    execution_time=time.time() - start_time,
                    tool_calls=self._tool_calls.copy()
                )
    
    def _execute_sync(
        self,
        code: str,
        globals_dict: Dict,
        locals_dict: Dict
    ) -> None:
        """Execute code synchronously (called from executor)."""
        exec(code, globals_dict, locals_dict)
    
    def _indent_code(self, code: str, spaces: int = 4) -> str:
        """Indent code block."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line for line in lines)
    
    async def execute_simple(
        self,
        code: str,
        timeout: float = 30.0
    ) -> ExecutionResult:
        """
        Execute simple synchronous code (no async/await).
        
        Useful for quick data transformations.
        """
        start_time = time.time()
        
        # Validate
        error = self._validate_code(code)
        if error:
            return ExecutionResult(
                success=False,
                error=f"Validation failed: {error}",
                execution_time=time.time() - start_time
            )
        
        printer = SandboxedPrint()
        globals_dict = self._create_safe_globals(printer)
        locals_dict: Dict[str, Any] = {}
        
        try:
            # Capture stdout
            stdout_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture):
                exec(code, globals_dict, locals_dict)
            
            # Add captured stdout to logs
            stdout_output = stdout_capture.getvalue()
            if stdout_output:
                printer.logs.extend(stdout_output.strip().split('\n'))
            
            return ExecutionResult(
                success=True,
                output=locals_dict.get('result'),
                logs=printer.logs,
                execution_time=time.time() - start_time,
                variables={k: v for k, v in locals_dict.items() 
                          if not k.startswith('_')}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                logs=printer.logs,
                execution_time=time.time() - start_time
            )


class CodeGenerator:
    """
    Generate code for common MCP operations.
    
    Provides templates for common patterns that the LLM can use.
    """
    
    @staticmethod
    def generate_tool_wrapper(server: str, tool_name: str, schema: Dict) -> str:
        """Generate a typed wrapper function for a tool."""
        params = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        # Build function signature
        param_strs = []
        for name, info in params.items():
            param_type = info.get("type", "Any")
            type_map = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
            py_type = type_map.get(param_type, "Any")
            
            if name in required:
                param_strs.append(f"{name}: {py_type}")
            else:
                param_strs.append(f"{name}: {py_type} = None")
        
        params_str = ", ".join(param_strs)
        
        return f'''
async def {tool_name}({params_str}):
    """Auto-generated wrapper for {server}.{tool_name}"""
    args = {{k: v for k, v in locals().items() if v is not None}}
    return await call_tool("{server}.{tool_name}", args)
'''
    
    @staticmethod
    def generate_data_pipeline(
        source_tool: str,
        transform_code: str,
        sink_tool: str
    ) -> str:
        """Generate a data pipeline: fetch -> transform -> store."""
        return f'''
# Fetch data
data = await call_tool("{source_tool}")
if "error" in data:
    print(f"Error fetching data: {{data['error']}}")
else:
    # Transform
    result = data.get("data", data)
    {transform_code}
    
    # Store
    await call_tool("{sink_tool}", {{"data": result}})
    print("Pipeline completed successfully")
'''
