"""
Base Tool Module - Foundation for all tool plugins.

This module provides the abstract base class and schema definitions
that all tool plugins must implement. Based on Anthropic's advanced
tool use patterns for token-efficient tool discovery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class ToolSchema:
    """
    Schema that describes a tool to the LLM for routing decisions.
    
    Based on Anthropic's Tool Search Tool pattern:
    - Minimal tokens for discovery (name, description, examples)
    - Full schema loaded only when tool is selected (defer_loading)
    
    Distributed Prompting Support:
    - system_instruction: LLM routing hints injected dynamically into prompts
    - code_example: Code-first execution example for code generation mode
    
    Attributes:
        name: Unique identifier for the tool (e.g., "weather", "web_search")
        description: Clear description of what the tool does (used for search matching)
        parameters: JSON Schema defining the tool's input parameters
        examples: Example queries this tool handles (improves LLM routing accuracy)
        input_examples: Concrete example inputs showing correct parameter usage
        defer_loading: If True, full schema only loaded when tool is discovered
        always_loaded: If True, tool is always available without search (high-frequency tools)
        system_instruction: Instructions for LLM on when/how to use this tool.
                           Injected into prompts dynamically - no hardcoding in router.
        code_example: Example code snippet showing how to call this tool.
                      Used for code-first execution mode.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": []
    })
    examples: List[str] = field(default_factory=list)
    input_examples: List[Dict[str, Any]] = field(default_factory=list)
    defer_loading: bool = True
    always_loaded: bool = False
    system_instruction: Optional[str] = None
    code_example: Optional[str] = None
    
    def to_search_entry(self) -> Dict[str, Any]:
        """
        Generate minimal representation for Tool Search Tool.
        Only name, description, and examples - saves tokens during discovery.
        Includes system_instruction if present for routing hints.
        """
        entry = {
            "name": self.name,
            "description": self.description,
            "examples": self.examples[:3]  # Limit examples for token efficiency
        }
        # Include system_instruction if available (for distributed prompting)
        if self.system_instruction:
            entry["system_instruction"] = self.system_instruction
        return entry
    
    def to_full_schema(self) -> Dict[str, Any]:
        """
        Generate complete tool schema for LLM invocation.
        Only loaded after tool is selected via search.
        Includes code_example for code-first execution mode.
        """
        schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
        if self.input_examples:
            schema["input_examples"] = self.input_examples
        # Include code_example for code-first execution
        if self.code_example:
            schema["code_example"] = self.code_example
        # Include system_instruction for routing context
        if self.system_instruction:
            schema["system_instruction"] = self.system_instruction
        return schema
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "examples": self.examples,
            "input_examples": self.input_examples,
            "defer_loading": self.defer_loading,
            "always_loaded": self.always_loaded
        }


class BaseTool(ABC):
    """
    Abstract base class for all MCP tool plugins.
    
    To create a new tool, simply:
    1. Create a new file in plugins/ (e.g., my_tool.py)
    2. Subclass BaseTool and implement schema property and execute method
    3. The tool will be auto-discovered by the registry
    
    Example:
        class MyTool(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="my_tool",
                    description="Does something useful",
                    parameters={...},
                    examples=["example query 1", "example query 2"]
                )
            
            async def execute(self, **params) -> Dict[str, Any]:
                # Tool logic here
                return {"result": "..."}
    """
    
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """
        Return the tool's schema for LLM routing.
        
        This defines:
        - Tool name and description (used for search)
        - Parameter schema (JSON Schema format)
        - Example queries (improves routing accuracy)
        - Loading behavior (defer_loading, always_loaded)
        """
        pass
    
    @abstractmethod
    async def execute(self, **params) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            **params: Parameters extracted by the LLM router
            
        Returns:
            Dict containing the tool's result. Should include:
            - 'success': bool indicating if execution succeeded
            - 'data': The actual result data
            - 'error': Error message if success is False
        """
        pass
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format the tool result for display to the user.
        
        Override this method to customize how results are presented.
        Default implementation returns a simple string representation.
        
        Args:
            result: The result from execute()
            
        Returns:
            Human-readable string representation of the result
        """
        if not result.get("success", True):
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        data = result.get("data")
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return json.dumps(data, indent=2, ensure_ascii=False)
        return str(data)
    
    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against the tool's schema.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = self.schema.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required parameters
        for param in required:
            if param not in params or params[param] is None:
                return False, f"Missing required parameter: {param}"
        
        # Coerce and validate types (be lenient)
        for param, value in list(params.items()):
            if param in properties:
                expected_type = properties[param].get("type")
                
                # Try to coerce types
                if expected_type == "string":
                    if not isinstance(value, str):
                        params[param] = str(value) if value is not None else ""
                elif expected_type == "integer":
                    if not isinstance(value, int):
                        try:
                            params[param] = int(value) if value is not None else 0
                        except (ValueError, TypeError):
                            pass  # Keep original, tool will handle
                elif expected_type == "number":
                    if not isinstance(value, (int, float)):
                        try:
                            params[param] = float(value) if value is not None else 0.0
                        except (ValueError, TypeError):
                            pass
                elif expected_type == "boolean":
                    if not isinstance(value, bool):
                        params[param] = str(value).lower() in ("true", "1", "yes")
        
        return True, None
    
    async def safe_execute(self, **params) -> Dict[str, Any]:
        """
        Execute with validation and error handling.
        
        This is the recommended way to call tools - it validates
        parameters and catches exceptions.
        """
        # Validate parameters
        is_valid, error = self.validate_params(params)
        if not is_valid:
            return {"success": False, "error": error}
        
        try:
            result = await self.execute(**params)
            if "success" not in result:
                result["success"] = True
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.schema.name}')>"
