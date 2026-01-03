"""
Tool Registry - Auto-discovery and management of MCP tool plugins.

This module provides automatic discovery of tool plugins from the
plugins/ directory, eliminating the need to manually register tools.

Based on Anthropic's Tool Search Tool pattern:
- Tools are auto-discovered at startup
- Schemas are cached for efficient lookup
- Supports deferred loading for token efficiency
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Singleton registry for MCP tool plugins with auto-discovery.
    
    Features:
    - Automatically discovers tools in plugins/ directory
    - Caches tool instances and schemas
    - Supports tool search for token-efficient discovery
    - Thread-safe singleton pattern
    
    Usage:
        registry = ToolRegistry()
        tool = registry.get_tool("weather")
        result = await tool.execute(city="Mumbai")
    """
    
    _instance: Optional["ToolRegistry"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once (singleton)
        if ToolRegistry._initialized:
            return
        
        self._tools: Dict[str, BaseTool] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._search_index: List[Dict] = []  # Minimal entries for search
        
        # Discover and load plugins
        self._discover_plugins()
        ToolRegistry._initialized = True
        
        logger.info(f"✅ ToolRegistry initialized with {len(self._tools)} tools: {list(self._tools.keys())}")
    
    def _discover_plugins(self) -> None:
        """
        Auto-discover and load all tool plugins from plugins/ directory.
        
        Scans for *_tool.py files and loads any BaseTool subclasses found.
        """
        plugins_path = Path(__file__).parent.parent / "plugins"
        
        if not plugins_path.exists():
            logger.warning(f"⚠️ Plugins directory not found: {plugins_path}")
            return
        
        # Iterate through all Python files in plugins/
        for file_path in plugins_path.glob("*_tool.py"):
            module_name = file_path.stem  # e.g., "weather_tool"
            
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{module_name}",
                    file_path
                )
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find BaseTool subclasses in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if it's a class that's a subclass of BaseTool
                    if (isinstance(attr, type) 
                        and issubclass(attr, BaseTool) 
                        and attr is not BaseTool):
                        
                        try:
                            # Instantiate the tool
                            tool_instance = attr()
                            schema = tool_instance.schema
                            
                            # Register the tool
                            self._tools[schema.name] = tool_instance
                            self._schemas[schema.name] = schema
                            
                            # Add to search index (minimal entry)
                            self._search_index.append(schema.to_search_entry())
                            
                            logger.debug(f"  📦 Loaded tool: {schema.name} from {module_name}.py")
                            
                        except Exception as e:
                            logger.error(f"  ❌ Failed to instantiate {attr_name}: {e}")
                            
            except Exception as e:
                logger.error(f"❌ Failed to load plugin {module_name}: {e}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: The tool's name (e.g., "weather", "web_search")
            
        Returns:
            The tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """
        Get a tool's schema by name.
        
        Args:
            name: The tool's name
            
        Returns:
            The tool's schema or None if not found
        """
        return self._schemas.get(name)
    
    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())
    
    def get_all_tools(self) -> List[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())
    
    def get_always_loaded_tools(self) -> List[BaseTool]:
        """
        Get tools marked as always_loaded=True.
        These are high-frequency tools that should always be in context.
        """
        return [
            tool for tool in self._tools.values()
            if tool.schema.always_loaded
        ]
    
    def get_deferred_tools(self) -> List[BaseTool]:
        """
        Get tools marked as defer_loading=True.
        These are discovered on-demand via Tool Search.
        """
        return [
            tool for tool in self._tools.values()
            if tool.schema.defer_loading and not tool.schema.always_loaded
        ]
    
    def get_search_index(self) -> List[Dict]:
        """
        Get minimal search entries for all tools.
        Used by Tool Search Tool for token-efficient discovery.
        """
        return self._search_index.copy()
    
    def get_schemas_for_llm(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate tool schemas formatted for LLM consumption.
        
        Args:
            tool_names: Optional list of specific tools to include.
                       If None, returns schemas for always_loaded tools only.
        
        Returns:
            List of tool schemas in LLM-friendly format
        """
        if tool_names is not None:
            # Return specific tools
            schemas = []
            for name in tool_names:
                schema = self._schemas.get(name)
                if schema:
                    schemas.append(schema.to_full_schema())
            return schemas
        
        # Return always-loaded tools only (for initial context)
        return [
            tool.schema.to_full_schema()
            for tool in self.get_always_loaded_tools()
        ]
    
    def search_tools(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search for relevant tools based on query.
        
        Implements a simple BM25-like search over tool names,
        descriptions, and examples.
        
        Args:
            query: The user's query
            max_results: Maximum number of tools to return
            
        Returns:
            List of tool names sorted by relevance
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))
        
        scores: List[tuple[str, float]] = []
        
        for entry in self._search_index:
            name = entry["name"]
            description = entry["description"].lower()
            examples = [ex.lower() for ex in entry.get("examples", [])]
            
            score = 0.0
            
            # Exact name match (highest weight)
            if name.lower() in query_lower or query_lower in name.lower():
                score += 10.0
            
            # Term matches in name
            name_terms = set(re.findall(r'\w+', name.lower()))
            name_overlap = len(query_terms & name_terms)
            score += name_overlap * 5.0
            
            # Term matches in description
            desc_terms = set(re.findall(r'\w+', description))
            desc_overlap = len(query_terms & desc_terms)
            score += desc_overlap * 2.0
            
            # Matches in examples
            for example in examples:
                example_terms = set(re.findall(r'\w+', example))
                example_overlap = len(query_terms & example_terms)
                score += example_overlap * 3.0
                
                # Bonus for substring match in example
                if query_lower in example:
                    score += 5.0
            
            if score > 0:
                scores.append((name, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N tool names
        return [name for name, _ in scores[:max_results]]
    
    def reload(self) -> None:
        """
        Reload all plugins (useful for development hot-reloading).
        """
        self._tools.clear()
        self._schemas.clear()
        self._search_index.clear()
        self._discover_plugins()
        logger.info(f"🔄 ToolRegistry reloaded with {len(self._tools)} tools")
    
    def get_tool_summary(self) -> str:
        """
        Generate a human-readable summary of all registered tools.
        Useful for debugging and system prompts.
        """
        lines = ["Available Tools:"]
        for name, schema in self._schemas.items():
            status = "🔵 always" if schema.always_loaded else "🟡 deferred"
            lines.append(f"  {status} {name}: {schema.description[:60]}...")
        return "\n".join(lines)


# Global convenience function
def get_registry() -> ToolRegistry:
    """Get the global ToolRegistry instance."""
    return ToolRegistry()
