"""
Plugins Package - Tool Plugins for the MCP Architecture.

This directory contains all tool plugins that are automatically
discovered and loaded by the ToolRegistry.

To create a new tool:
1. Create a new file named *_tool.py (e.g., my_tool.py)
2. Subclass BaseTool from architecture.base_tool
3. Implement the schema property and execute method
4. The tool will be auto-discovered on startup

Example:
    # my_tool.py
    from architecture.base_tool import BaseTool, ToolSchema
    
    class MyTool(BaseTool):
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(
                name="my_tool",
                description="Does something useful",
                parameters={...},
                examples=["example query"]
            )
        
        async def execute(self, **params):
            return {"success": True, "data": "result"}
"""

# This file intentionally left mostly empty.
# Plugin discovery happens in architecture/registry.py by scanning this directory.
