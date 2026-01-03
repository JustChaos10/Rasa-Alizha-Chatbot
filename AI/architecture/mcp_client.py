"""
MCP Client - Connects to MCP servers via stdio or HTTP transport.

Implements the Model Context Protocol (MCP) client-side:
- JSON-RPC 2.0 message format
- stdio transport (subprocess communication)
- HTTP/SSE transport (remote servers)
- Tool discovery and invocation

Based on: https://modelcontextprotocol.io/docs/concepts/transports
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

def _env_timeout(name: str, default: float) -> float:
    """Read a timeout value from env, falling back safely."""
    try:
        raw = (os.getenv(name) or "").strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


class TransportType(Enum):
    """Supported transport types for MCP connections."""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server.
    
    Supports distributed prompting via tool_metadata field:
    - Allows specifying system_instruction and code_example for external MCP tools
    - Metadata is injected into LLM prompts at runtime
    """
    name: str
    transport: TransportType
    # For stdio transport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    # For HTTP transport
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    # Metadata
    description: str = ""
    enabled: bool = True
    # Distributed prompting: per-tool metadata for LLM routing
    # Format: {"tool_name": {"system_instruction": "...", "code_example": "..."}}
    tool_metadata: Optional[Dict[str, Dict[str, str]]] = None
    # Routing keywords for improved tool selection
    routing_keywords: Optional[List[str]] = None


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class Transport(ABC):
    """Abstract base class for MCP transports."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the server."""
        pass
    
    @abstractmethod
    async def send(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message."""
        pass
    
    @abstractmethod
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive a JSON-RPC message."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        pass


class StdioTransport(Transport):
    """
    Stdio transport for MCP servers running as subprocesses.
    
    Communicates via stdin/stdout using newline-delimited JSON.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._read_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._connected = False
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # Store the loop we're connected on
    
    async def connect(self) -> bool:
        """Start the subprocess and establish communication."""
        if self._connected:
            return True
        
        if not self.config.command:
            logger.error(f"No command specified for stdio server {self.config.name}")
            return False
        
        try:
            # Store the event loop we're connecting on
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.get_event_loop()
            
            # Build command (normalize known venv paths cross-platform)
            command = self.config.command
            try:
                cmd_path = Path(command)
                if not cmd_path.is_absolute():
                    candidate = Path(self.config.cwd or os.getcwd()) / cmd_path
                else:
                    candidate = cmd_path

                if not candidate.exists():
                    normalized = command.replace("\\", "/")
                    if normalized.endswith(".venv/bin/python") or normalized.endswith(".venv/bin/python3"):
                        win_cmd = (
                            normalized
                            .replace(".venv/bin/python3", ".venv/Scripts/python.exe")
                            .replace(".venv/bin/python", ".venv/Scripts/python.exe")
                        )
                        command = win_cmd.replace("/", "\\")
            except Exception:
                pass

            cmd = [command]
            if self.config.args:
                cmd.extend(self.config.args)
            
            # Prepare environment
            env = os.environ.copy()
            if self.config.env:
                env.update(self.config.env)
            
            # Start process
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.cwd
            )
            
            self._connected = True
            
            # Start reader task
            self._read_task = asyncio.create_task(self._reader_loop())
            
            # Start stderr reader for debugging
            self._stderr_task = asyncio.create_task(self._stderr_loop())
            
            logger.info(f"✅ Connected to MCP server '{self.config.name}' via stdio (PID: {self._process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.config.name}': {e}")
            self._connected = False
            return False
    
    async def _reader_loop(self) -> None:
        """Background task to read messages from stdout."""
        try:
            logger.debug(f"🔄 Reader loop started for {self.config.name}")
            while self._connected and self._process:
                line = await self._process.stdout.readline()
                if not line:
                    logger.debug("📭 Reader loop: empty line, process may have ended")
                    break
                
                try:
                    message = json.loads(line.decode('utf-8').strip())
                    logger.debug(f"📨 Reader loop received message: {str(message)[:100]}...")
                    
                    # Check if this is a response to a pending request
                    msg_id = message.get("id")
                    if msg_id and msg_id in self._pending_requests:
                        future = self._pending_requests.pop(msg_id)
                        if not future.done():
                            logger.debug(f"✅ Setting result for request {msg_id[:8]}...")
                            future.set_result(message)
                        else:
                            logger.warning(f"⚠️ Future already done for {msg_id[:8]}...")
                    else:
                        # Queue for general consumption (notifications, etc.)
                        logger.debug("📥 Queuing message (no pending request found)")
                        await self._message_queue.put(message)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from server: {e}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Reader loop error: {e}")
        finally:
            self._connected = False
    
    async def _stderr_loop(self) -> None:
        """Background task to read and log stderr from the MCP server."""
        try:
            while self._connected and self._process and self._process.stderr:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode('utf-8').strip()
                if text:
                    logger.info(f"[MCP-{self.config.name}] {text}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Stderr loop error: {e}")
    
    async def send(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message via stdin."""
        if not self._connected or not self._process or not self._process.stdin:
            raise ConnectionError("Transport not connected")
        
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode('utf-8'))
        await self._process.stdin.drain()
    
    async def send_request(self, method: str, params: Optional[Dict] = None, timeout: float = 300.0) -> Dict[str, Any]:
        """Send a request and wait for response."""
        request_id = str(uuid.uuid4())
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            message["params"] = params
        
        # Use the stored loop from connect(), or get current running loop
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
        
        # Create future on the correct loop
        future: asyncio.Future = loop.create_future()
        self._pending_requests[request_id] = future
        
        logger.debug(f"📤 Sending MCP request {method} (id: {request_id[:8]}...)")
        
        try:
            await self.send(message)
            response = await asyncio.wait_for(future, timeout=timeout)
            logger.debug(f"📥 Received MCP response for {method} (id: {request_id[:8]}...)")
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            logger.error(f"⏱️ Request {method} timed out after {timeout}s (id: {request_id[:8]}...)")
            raise TimeoutError(f"Request {method} timed out after {timeout}s")
    
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive a message from the queue."""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def close(self) -> None:
        """Terminate the subprocess."""
        self._connected = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, '_stderr_task') and self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None
        
        logger.info(f"Closed connection to MCP server '{self.config.name}'")
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None


class HTTPTransport(Transport):
    """
    HTTP transport for remote MCP servers.
    
    Uses HTTP POST for requests and optionally SSE for streaming.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._connected = False
        self._client = None
    
    async def connect(self) -> bool:
        """Initialize HTTP client."""
        if not self.config.url:
            logger.error(f"No URL specified for HTTP server {self.config.name}")
            return False
        
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.config.url,
                headers=self.config.headers or {},
                timeout=300.0
            )
            self._connected = True
            logger.info(f"✅ Connected to MCP server '{self.config.name}' via HTTP ({self.config.url})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to HTTP server '{self.config.name}': {e}")
            return False
    
    async def send(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message via HTTP POST."""
        if not self._connected or not self._client:
            raise ConnectionError("Transport not connected")
        
        response = await self._client.post("/", json=message)
        response.raise_for_status()
    
    async def send_request(self, method: str, params: Optional[Dict] = None, timeout: float = 300.0) -> Dict[str, Any]:
        """Send request and get response."""
        if not self._connected or not self._client:
            raise ConnectionError("Transport not connected")
        
        request_id = str(uuid.uuid4())
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            message["params"] = params
        
        response = await self._client.post("/", json=message, timeout=timeout)
        response.raise_for_status()
        return response.json()
    
    async def receive(self) -> Optional[Dict[str, Any]]:
        """HTTP transport doesn't support push messages in basic mode."""
        return None
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info(f"Closed connection to MCP server '{self.config.name}'")
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None


class MCPClient:
    """
    MCP Client - Manages connection to a single MCP server.
    
    Provides:
    - Connection management
    - Tool discovery (list_tools)
    - Tool invocation (call_tool)
    - Resource access (list_resources, read_resource)
    
    Usage:
        config = MCPServerConfig(
            name="weather",
            transport=TransportType.STDIO,
            command="python",
            args=["weather_server.py"]
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("get_weather", {"city": "London"})
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._transport: Optional[Transport] = None
        self._tools: List[MCPTool] = []
        self._resources: List[MCPResource] = []
        self._initialized = False
    
    async def connect(self) -> bool:
        """Connect to the MCP server and initialize."""
        # Create appropriate transport
        if self.config.transport == TransportType.STDIO:
            self._transport = StdioTransport(self.config)
        elif self.config.transport in (TransportType.HTTP, TransportType.SSE):
            self._transport = HTTPTransport(self.config)
        else:
            logger.error(f"Unsupported transport type: {self.config.transport}")
            return False
        
        # Connect
        if not await self._transport.connect():
            return False
        
        # Initialize MCP session
        try:
            init_timeout = _env_timeout("MCP_INIT_TIMEOUT", 10.0)
            response = await self._transport.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "clientInfo": {
                    "name": "rasa-chatbot-mcp-client",
                    "version": "1.0.0"
                }
            }, timeout=init_timeout)
            
            if "error" in response:
                logger.error(f"MCP initialization error: {response['error']}")
                return False
            
            # Send initialized notification
            await self._transport.send({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })
            
            self._initialized = True
            logger.info(f"✅ MCP session initialized with '{self.config.name}'")
            
            # Discover tools
            await self._discover_tools()
            
            return True
            
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            await self.disconnect()
            return False
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        try:
            list_timeout = _env_timeout("MCP_LIST_TOOLS_TIMEOUT", _env_timeout("MCP_INIT_TIMEOUT", 10.0))
            response = await self._transport.send_request("tools/list", timeout=list_timeout)
            
            if "error" in response:
                logger.warning(f"Failed to list tools: {response['error']}")
                return
            
            tools_data = response.get("result", {}).get("tools", [])
            self._tools = [
                MCPTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    server_name=self.config.name
                )
                for t in tools_data
            ]
            
            logger.info(f"📦 Discovered {len(self._tools)} tools from '{self.config.name}': {[t.name for t in self._tools]}")
            
        except Exception as e:
            logger.warning(f"Tool discovery failed for '{self.config.name}': {e}")
    
    async def list_tools(self) -> List[MCPTool]:
        """Return the list of available tools."""
        return self._tools
    
    async def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
    
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self._initialized or not self._transport:
            return {"success": False, "error": "Client not connected"}
        
        try:
            response = await self._transport.send_request("tools/call", {
                "name": name,
                "arguments": arguments or {}
            })
            
            if "error" in response:
                return {"success": False, "error": response["error"]}
            
            result = response.get("result", {})
            
            # Extract content from MCP response format
            content = result.get("content", [])
            if content and isinstance(content, list):
                # Combine text content
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                if texts:
                    combined_text = "\n".join(texts)
                    # Try to parse as JSON (tools may return JSON as text)
                    try:
                        import json
                        parsed = json.loads(combined_text)
                        return {"success": True, "data": parsed}
                    except (json.JSONDecodeError, TypeError):
                        return {"success": True, "data": combined_text}
            
            return {"success": True, "data": result}
            
        except TimeoutError as e:
            return {"success": False, "error": f"Tool call timed out: {e}"}
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_resources(self) -> List[MCPResource]:
        """List available resources from the server."""
        if not self._initialized or not self._transport:
            return []
        
        try:
            response = await self._transport.send_request("resources/list")
            
            if "error" in response:
                logger.warning(f"Failed to list resources: {response['error']}")
                return []
            
            resources_data = response.get("result", {}).get("resources", [])
            self._resources = [
                MCPResource(
                    uri=r["uri"],
                    name=r.get("name", r["uri"]),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType", "text/plain")
                )
                for r in resources_data
            ]
            
            return self._resources
            
        except Exception as e:
            logger.warning(f"Resource listing failed: {e}")
            return []
    
    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource from the server."""
        if not self._initialized or not self._transport:
            return None
        
        try:
            response = await self._transport.send_request("resources/read", {
                "uri": uri
            })
            
            if "error" in response:
                logger.warning(f"Failed to read resource: {response['error']}")
                return None
            
            contents = response.get("result", {}).get("contents", [])
            if contents:
                return contents[0].get("text", "")
            return None
            
        except Exception as e:
            logger.warning(f"Resource read failed: {e}")
            return None
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._transport:
            await self._transport.close()
            self._transport = None
        self._initialized = False
        self._tools = []
        self._resources = []
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected and initialized."""
        return self._initialized and self._transport is not None and self._transport.is_connected
    
    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name
    
    @property
    def tools(self) -> List[MCPTool]:
        """Get cached tools list."""
        return self._tools
