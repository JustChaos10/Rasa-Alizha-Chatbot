"""
Web Search Tool (Tavily)

Standalone web search tool that always calls Tavily.
Use for: "search the web", "google", "latest news", "recent", "sources", "links"

Separate from KB/Vector search which handles internal documents.
"""

import os
import logging
import httpx
from typing import Any, Dict, List, Optional

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search tool using Tavily API.
    
    Routes to this tool when user wants to search the internet,
    not internal documents/knowledge base.
    """
    
    def __init__(self):
        self._api_key = os.getenv("TAVILY_API_KEY", "")
        self._base_url = "https://api.tavily.com/search"
        self._timeout = float(os.getenv("TAVILY_TIMEOUT", "10"))
        
        if self._api_key:
            logger.info("✅ WebSearchTool: Tavily API configured")
        else:
            logger.warning("⚠️ WebSearchTool: No TAVILY_API_KEY - web search disabled")
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="""Search the web for current information, news, and external sources.
Use this for: "search the web for...", "google...", "find online...", 
"latest news about...", "recent...", "what's happening with...",
"sources for...", "links about..."

Do NOT use for internal documents, knowledge base, PDFs, or company policies.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (1-10, default 5)"
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Include AI-generated answer summary (default true)"
                    }
                },
                "required": ["query"]
            },
            examples=[
                "Search the web for latest AI news",
                "Google Python best practices",
                "Find recent articles about climate change",
                "What's happening with tech stocks today",
                "search the web",
                "google this",
                "find links about",
                "get sources for"
            ],
            system_instruction="""ROUTING PRIORITY: Use web_search when user says:
- "search the web", "google", "search online", "find links", "get sources"
- Wants EXTERNAL web results with URLs/citations
- Asks about current events, recent news, or live information

Do NOT use web_search when:
- User asks specifically for "news headlines" without "search" (use news tool)
- User asks about internal documents/policies (use vector_search)""",
            always_loaded=True  # High-frequency tool, always available
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute web search using Tavily."""
        query = kwargs.get("query", "")
        max_results = min(max(kwargs.get("max_results", 5), 1), 10)
        include_answer = kwargs.get("include_answer", True)
        
        if not query:
            return {
                "success": False,
                "error": "No search query provided",
                "data": None
            }
        
        if not self._api_key:
            return {
                "success": False,
                "error": "Web search not configured (missing TAVILY_API_KEY)",
                "data": None
            }
        
        try:
            payload = {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": include_answer,
                "search_depth": "basic"
            }
            
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._base_url, json=payload)
                response.raise_for_status()
                data = response.json()
            
            # Format results
            results = data.get("results", [])
            answer = data.get("answer", "")
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:300],
                    "score": r.get("score", 0)
                })
            
            return {
                "success": True,
                "data": {
                    "query": query,
                    "answer": answer,
                    "results": formatted_results,
                    "total": len(formatted_results)
                }
            }
            
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Search timed out after {self._timeout}s",
                "data": None
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"API error: {e.response.status_code}",
                "data": None
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format search results for display."""
        if not result.get("success"):
            return f"❌ Search failed: {result.get('error', 'Unknown error')}"
        
        data = result.get("data", {})
        if not data:
            return "No results found."
        
        output_parts = []
        
        # Include AI answer if available
        answer = data.get("answer", "")
        if answer:
            output_parts.append(f"**Summary:** {answer}")
            output_parts.append("")
        
        # Format results
        results = data.get("results", [])
        if results:
            output_parts.append("**Sources:**")
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                
                output_parts.append(f"{i}. [{title}]({url})")
                if snippet:
                    output_parts.append(f"   {snippet[:150]}...")
                output_parts.append("")
        else:
            output_parts.append("No results found for this query.")
        
        return "\n".join(output_parts)


# Note: Tool is auto-discovered by ToolRegistry - no manual registration needed

