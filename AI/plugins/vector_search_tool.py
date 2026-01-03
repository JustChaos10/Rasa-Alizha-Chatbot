"""
Vector Search Tool Plugin - Semantic document retrieval.

Connects to an external vector search service for RAG-style
document retrieval and knowledge base queries.
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


class VectorSearchTool(BaseTool):
    """
    Tool for semantic vector search across document collections.
    
    Connects to an external retrieval service for RAG-style
    document lookup and knowledge base queries.
    """
    
    def __init__(self):
        self._base_url = os.getenv("VECTOR_BASE_URL", "http://localhost:8001").rstrip("/")
        self._api_key = os.getenv("VECTOR_API_KEY", "")
        self._timeout = float(os.getenv("VECTOR_TIMEOUT", "10.0"))
        self._default_namespace = os.getenv("VECTOR_NAMESPACE", "default").strip() or "default"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="vector_search",
            description="Search through documents and knowledge bases using semantic similarity. Retrieves relevant information from stored documents, policies, FAQs, and other text content. Use for questions about specific topics in the knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to find relevant documents for"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace/collection to search within",
                        "default": "default"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            },
            examples=[
                "Search for information about company policies",
                "Find documents about security guidelines",
                "Look up HR policies",
                "Search the knowledge base for vacation policy",
                "Find information about Qdrant vector database",
                "Search for onboarding procedures",
                "Look up compliance requirements"
            ],
            input_examples=[
                {"query": "What is the vacation policy?"},
                {"query": "security guidelines", "top_k": 3},
                {"query": "employee benefits", "namespace": "hr_docs"}
            ],
            defer_loading=True,
            always_loaded=False
        )
    
    async def execute(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform semantic vector search.
        
        Args:
            query: Search query or question
            namespace: Optional namespace to search
            top_k: Number of results to return
            
        Returns:
            Search results with relevant documents
        """
        try:
            # Use default namespace if not specified
            if not namespace:
                namespace = self._default_namespace
            
            # Clamp top_k
            top_k = max(1, min(20, top_k))
            
            # Generate trace ID for debugging
            trace_id = uuid.uuid4().hex
            
            # Call retrieval service
            results = await self._call_service(query, namespace, top_k, trace_id)
            
            if not results or not results.get("hits"):
                return {
                    "type": "text",
                    "text": f"🔍 No relevant documents found for: \"{query}\"\n\nTry rephrasing your question or searching in a different namespace."
                }
            
            # Format results
            formatted = self._format_results(results.get("hits", []), query)
            
            return {
                "type": "text",
                "text": formatted,
                "metadata": {
                    "namespace": namespace,
                    "num_results": len(results.get("hits", [])),
                    "trace_id": trace_id
                }
            }
            
        except httpx.ConnectError:
            logger.error("Cannot connect to vector search service")
            return {
                "type": "text",
                "text": "🔍 Vector search service is currently unavailable. Please try again later."
            }
        except httpx.TimeoutException:
            logger.error("Vector search timeout")
            return {
                "type": "text",
                "text": "⏱️ Search is taking too long. Please try a more specific query."
            }
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {
                "type": "text",
                "text": f"❌ Search error: {str(e)}"
            }
    
    async def _call_service(
        self, 
        query: str, 
        namespace: str, 
        top_k: int, 
        trace_id: str
    ) -> Dict[str, Any]:
        """Call the vector retrieval service."""
        client = await self._get_client()
        
        headers = {
            "Content-Type": "application/json",
            "X-Trace-Id": trace_id
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        
        payload = {
            "namespace": namespace,
            "query": query,
            "top_k": top_k
        }
        
        response = await client.post(
            f"{self._base_url}/retrieve",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    def _format_results(self, hits: List[Dict[str, Any]], query: str) -> str:
        """Format search results for display."""
        lines = [f"🔍 **Search Results for:** \"{query}\"\n"]
        
        for i, hit in enumerate(hits[:5], 1):
            text = (hit.get("text") or "").strip()
            score = hit.get("score", 0)
            metadata = hit.get("metadata", {})
            
            # Truncate long text
            if len(text) > 300:
                text = text[:297] + "..."
            
            # Clean up whitespace
            text = " ".join(text.split())
            
            # Format entry
            lines.append(f"**{i}.** {text}")
            
            # Add metadata if available
            meta_parts = []
            if metadata.get("topic"):
                meta_parts.append(f"Topic: {metadata['topic']}")
            if metadata.get("source"):
                meta_parts.append(f"Source: {metadata['source']}")
            if score:
                meta_parts.append(f"Relevance: {score:.0%}")
            
            if meta_parts:
                lines.append(f"   _({', '.join(meta_parts)})_")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def upsert_documents(
        self, 
        documents: List[Dict[str, Any]], 
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Upsert documents into the vector store.
        
        Args:
            documents: List of documents with id, text, and optional metadata
            namespace: Namespace to store documents in
            
        Returns:
            Upsert result
        """
        try:
            if not namespace:
                namespace = self._default_namespace
            
            client = await self._get_client()
            
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            
            payload = {
                "namespace": namespace,
                "items": documents
            }
            
            response = await client.post(
                f"{self._base_url}/upsert",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            return {
                "success": True,
                "message": f"Upserted {len(documents)} documents to namespace '{namespace}'"
            }
            
        except Exception as e:
            logger.error(f"Upsert error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
