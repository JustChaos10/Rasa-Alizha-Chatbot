"""
Chat Tool Plugin - General conversation with integrated web search.

Handles all chat queries with:
- LLM-powered conversational responses via Groq
- Automatic web search fallback for current events/real-time data
- Image fetching for knowledge questions
- Related questions generation for engagement
"""

import os
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from architecture.base_tool import BaseTool, ToolSchema
from architecture.telemetry import trace_llm_call, log_llm_event

# Import the shared services - GlobalLLMService is the ONLY way to call LLMs
try:
    from shared_utils import get_service_manager, get_global_rate_limiter, get_global_llm_service
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    get_global_rate_limiter = None
    get_global_llm_service = None
    get_service_manager = None

logger = logging.getLogger(__name__)


class ChatTool(BaseTool):
    """
    Primary chat tool with integrated web search fallback.
    
    Features:
    - LLM-powered conversational responses via GlobalLLMService (GROQ → Gemini failover)
    - Automatic web search for current events/real-time queries
    - Image fetching via ImageSearchService (shared_utils.py)
    - Related questions generation for follow-up engagement
    
    Response format supports:
    - Plain text (data field)
    - Visual panel (image_data field) 
    - Related links (related_questions field)
    """
    
    def __init__(self):
        # Use GlobalLLMService for ALL LLM calls (centralized rate limiting + failover)
        self._global_llm = None
        self._model = "llama-3.3-70b-versatile"
        self._client: Optional[httpx.AsyncClient] = None
        
        # Tavily API for web search fallback
        self._tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self._tavily_url = "https://api.tavily.com/search"
        
        # Use ImageSearchService from shared_utils
        self._image_service = None
        if SHARED_UTILS_AVAILABLE and get_service_manager:
            try:
                self._image_service = get_service_manager().get_image_service()
                if self._image_service and self._image_service.is_enabled():
                    logger.info("✅ ChatTool: ImageSearchService enabled")
                else:
                    logger.info("ℹ️ ChatTool: ImageSearchService not configured")
            except Exception as e:
                logger.warning(f"⚠️ ChatTool: Could not initialize ImageSearchService: {e}")
        
        if self._tavily_api_key:
            logger.info("✅ ChatTool: Web search fallback enabled (Tavily)")
        else:
            logger.info("ℹ️ ChatTool: Web search fallback disabled (no TAVILY_API_KEY)")
    
    def _get_global_llm(self):
        """Get GlobalLLMService (centralized rate limiting + failover)."""
        if self._global_llm is None:
            if SHARED_UTILS_AVAILABLE and get_global_llm_service:
                self._global_llm = get_global_llm_service()
        return self._global_llm
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="chat",
            description="General conversation, knowledge questions, and explanations. Use for greetings, general world knowledge (NOT company-specific), science, technology, history, coding help, creative writing, math, explanations of concepts, and any query that is NOT about internal company data/employees/policies.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The user's message or question"
                    },
                    "conversation_history": {
                        "type": "string",
                        "description": "Optional conversation history string for context"
                    }
                },
                "required": ["message"]
            },
            examples=[
                # Greetings
                "Hello",
                "Hi there",
                "Good morning",
                # General knowledge - What is X?
                "What is Python?",
                "What is machine learning?",
                "What is artificial intelligence?",
                "What is quantum computing?",
                "What is blockchain?",
                "What is cloud computing?",
                "What are neural networks?",
                "What is deep learning?",
                # Explanations - Explain X
                "Explain quantum computing",
                "Explain machine learning",
                "Explain how databases work",
                "Explain the internet",
                "Explain neural networks",
                # How does X work?
                "How does Python work?",
                "How does machine learning work?",
                "How do computers work?",
                "How does the internet work?",
                "How do databases store data?",
                "How does AI work?",
                # Tell me about/more about
                "Tell me about Python",
                "Tell me more about machine learning",
                "Tell me about neural networks",
                "Tell me more about databases",
                "Tell me about cloud computing",
                # Benefits/advantages
                "What are the benefits of exercise?",
                "What are the advantages of Python?",
                "What are the benefits of machine learning?",
                # Coding help
                "How do I sort a list in Python?",
                "Write a Python function",
                "Help me with JavaScript",
                "Debug this code",
                # Creative and misc
                "Write a poem about the ocean",
                "What's 2 + 2?",
                "Tell me a joke",
                "Who wrote Romeo and Juliet?",
                # Follow-up style questions (related questions)
                "Can you explain that further?",
                "Tell me more",
                "What else should I know?",
                "How is this used in practice?",
                "What are the applications?",
                "Give me an example",
                "Why is this important?"
            ],
            input_examples=[
                {"message": "Hello, how are you?"},
                {"message": "Explain machine learning in simple terms"},
                {"message": "Write a haiku about coding"},
                {"message": "What is Python and how does it work?"},
                {"message": "Tell me more about neural networks"}
            ],
            defer_loading=False,  # Always available
            always_loaded=True    # This is the fallback
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def _is_knowledge_question(self, message: str) -> bool:
        """
        Check if the message is a knowledge/GK question that warrants
        an image and related questions.
        
        Returns True for questions like:
        - "What is X?"
        - "Who is Y?"
        - "Explain Z"
        - "Tell me about W"
        """
        message_lower = message.lower().strip()
        
        # Patterns that indicate a knowledge question
        knowledge_patterns = [
            r"^what\s+is\s+",
            r"^what\s+are\s+",
            r"^who\s+is\s+",
            r"^who\s+was\s+",
            r"^who\s+are\s+",
            r"^where\s+is\s+",
            r"^when\s+was\s+",
            r"^when\s+did\s+",
            r"^how\s+does\s+",
            r"^how\s+do\s+",
            r"^why\s+is\s+",
            r"^why\s+do\s+",
            r"^explain\s+",
            r"^describe\s+",
            r"^tell\s+me\s+about\s+",
            r"^define\s+",
        ]
        
        for pattern in knowledge_patterns:
            if re.match(pattern, message_lower):
                return True
        
        return False
    
    def _needs_web_search(self, message: str) -> bool:
        """
        Detect if the query needs real-time/current information from web search.
        
        Returns True for queries about:
        - Current events, news, recent happenings
        - Real-time data (prices, weather, scores)
        - Information likely beyond LLM training cutoff (2023-2024)
        - Fact verification requests
        """
        message_lower = message.lower().strip()
        
        # Patterns indicating need for current/real-time info
        web_search_indicators = [
            # Current events and news
            r"latest\s+",
            r"recent\s+",
            r"current\s+",
            r"today['\"]?s?\s+",
            r"this\s+week",
            r"this\s+month",
            r"this\s+year",
            r"last\s+week",
            r"last\s+month",
            r"yesterday",
            r"news\s+about",
            r"what['\"]?s\s+happening",
            r"breaking\s+",
            
            # Time-sensitive data
            r"price\s+of",
            r"stock\s+price",
            r"weather\s+in",
            r"score\s+of",
            r"results\s+of",
            r"standings",
            
            # Verification/fact-checking
            r"is\s+it\s+true",
            r"fact\s+check",
            r"verify\s+",
            r"confirm\s+",
            
            # Explicit year references (recent years beyond training)
            r"\b202[4-9]\b",
            r"\b203\d\b",
            
            # Questions about recent events/winners/results
            r"who\s+won",
            r"who\s+is\s+the\s+current",
            r"what\s+happened\s+",
            r"did\s+.+\s+happen",
            r"winner\s+of",
            r"won\s+the\s+",
        ]
        
        for pattern in web_search_indicators:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    async def _web_search(self, query: str, max_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Perform web search using Tavily API.
        
        Returns dict with:
        - answer: AI-generated summary from Tavily
        - results: List of search results with title, url, content
        - query: Original query
        
        Returns None if API not configured or search fails.
        """
        if not self._tavily_api_key:
            logger.debug("Tavily API key not configured - skipping web search")
            return None
        
        try:
            client = await self._get_client()
            
            payload = {
                "api_key": self._tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False
            }
            
            response = await client.post(
                self._tavily_url,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "query": query,
                "answer": data.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:300]  # Truncate for context
                    }
                    for r in data.get("results", [])[:max_results]
                ]
            }
            
        except Exception as e:
            logger.warning(f"Web search failed for '{query}': {e}")
            return None
    
    def _extract_topic(self, message: str) -> str:
        """
        Extract the main topic from a knowledge question.
        
        E.g., "What is a dog?" -> "dog"
              "Tell me about quantum computing" -> "quantum computing"
        """
        message_lower = message.lower().strip()
        
        # Remove question marks and trailing punctuation
        message_clean = re.sub(r'[?!.,]+$', '', message_lower)
        
        # Common patterns to extract the subject
        patterns = [
            (r"^what\s+(?:is|are)\s+(?:a|an|the)?\s*(.+)", 1),
            (r"^who\s+(?:is|was|are)\s+(.+)", 1),
            (r"^where\s+is\s+(.+)", 1),
            (r"^when\s+(?:was|did)\s+(.+)", 1),
            (r"^how\s+(?:does|do)\s+(.+)\s+work", 1),
            (r"^why\s+(?:is|do|are)\s+(.+)", 1),
            (r"^explain\s+(.+)", 1),
            (r"^describe\s+(.+)", 1),
            (r"^tell\s+me\s+about\s+(.+)", 1),
            (r"^define\s+(.+)", 1),
        ]
        
        for pattern, group in patterns:
            match = re.match(pattern, message_clean)
            if match:
                return match.group(group).strip()
        
        # Fallback: return the whole message without common words
        return message_clean
    
    def _fetch_image(self, topic: str) -> Optional[Dict[str, str]]:
        """
        Fetch an image for the topic using ImageSearchService from shared_utils.
        
        Returns dict with image_url, title, source_url, attribution
        or None if not available/configured.
        """
        if not self._image_service or not self._image_service.is_enabled():
            logger.debug("ImageSearchService not available - skipping image fetch")
            return None
        
        try:
            # Use the shared ImageSearchService (synchronous)
            result = self._image_service.fetch_image(topic)
            
            if not result:
                return None
            
            # Map the result to our expected format
            return {
                "image_url": result.get("image_url", ""),
                "thumbnail_url": result.get("thumbnail_url", ""),
                "title": result.get("alt", topic.title()),
                "source_url": result.get("context_url", ""),
                "attribution": result.get("attribution", ""),
                "alt": result.get("alt", f"Image of {topic}")
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch image for '{topic}': {e}")
            return None
    
    async def _generate_related_questions(self, message: str, response_text: str) -> List[Dict[str, str]]:
        """
        Generate related follow-up questions using GlobalLLMService.
        
        Returns list of dicts with 'title' and 'prompt' keys.
        """
        try:
            llm = self._get_global_llm()
            if not llm:
                logger.warning("GlobalLLMService not available - skipping related questions")
                return []
            
            prompt = f"""Based on this Q&A, generate exactly 3 related follow-up questions the user might want to ask.

User Question: {message}
Assistant Answer: {response_text[:500]}...

Return ONLY a JSON array with 3 objects, each having "title" (short display text, max 50 chars) and "prompt" (the full question).
Example: [{{"title": "How does it work?", "prompt": "How does photosynthesis work in plants?"}}]

JSON array:"""

            messages = [{"role": "user", "content": prompt}]
            
            # Use GlobalLLMService (handles rate limiting + failover automatically)
            content = await llm.call_with_messages_async(
                messages=messages,
                max_tokens=300,
                temperature=0.5,
                timeout=10.0,
                trace_name="chat-related-questions"
            )
            
            # Try to parse JSON from response
            # Handle cases where LLM adds extra text
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                import json
                questions = json.loads(json_match.group())
                # Validate structure
                valid_questions = []
                for q in questions[:3]:
                    if isinstance(q, dict) and "title" in q and "prompt" in q:
                        valid_questions.append({
                            "title": str(q["title"])[:50],
                            "prompt": str(q["prompt"])[:200]
                        })
                return valid_questions
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to generate related questions: {e}")
            log_llm_event("related-questions-failed", {"error": str(e)}, level="WARNING")
            return []
    
    async def execute(
        self,
        message: str,
        conversation_history: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle general conversation with rich responses and web search fallback.
        
        For knowledge questions (What is X?), also fetches:
        - Relevant image via Google Custom Search
        - Related follow-up questions
        
        For real-time/current events queries, uses web search to augment response.
        
        Args:
            message: The user's message
            conversation_history: Optional previous conversation context string from Redis
            
        Returns:
            Dict with:
            - success: bool
            - data: The text response
            - image_data: Optional image info for visual panel
            - related_questions: Optional list of follow-up questions
            - sources: Optional list of web sources used
        """
        llm = self._get_global_llm()
        if not llm:
            return {
                "success": False,
                "error": "GlobalLLMService not available - check API keys"
            }
        
        # Check if this is a knowledge question
        is_knowledge_q = self._is_knowledge_question(message)
        topic = self._extract_topic(message) if is_knowledge_q else ""
        
        # Check if web search is needed for current/real-time info
        needs_web = self._needs_web_search(message)
        web_context = ""
        web_sources = []
        
        if needs_web:
            logger.info(f"🔍 Web search triggered for: {message[:50]}...")
            search_result = await self._web_search(message)
            if search_result:
                # Build context from web search results
                web_context = "\n\n[Web Search Results for context]\n"
                if search_result.get("answer"):
                    web_context += f"Summary: {search_result['answer']}\n\n"
                
                for i, r in enumerate(search_result.get("results", [])[:3], 1):
                    web_context += f"{i}. {r['title']}\n{r['content']}\n\n"
                    web_sources.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", "")
                    })
                
                logger.info(f"✅ Web search found {len(search_result.get('results', []))} results")
        
        # Build system prompt based on question type
        if needs_web and web_context:
            system_content = (
                "You are a helpful AI assistant with access to real-time web search results. "
                "Use the provided web search context to give accurate, up-to-date information. "
                "Cite your sources naturally in the response when relevant. "
                "Be concise but informative. If the web results don't fully answer the question, "
                "combine them with your knowledge while being clear about what's from web vs your training."
            )
        elif is_knowledge_q:
            system_content = (
                "You are a helpful AI assistant. For knowledge questions, provide a BRIEF, "
                "concise answer in 2-3 short sentences maximum. Focus on the key facts only. "
                "Do NOT use bullet points, numbered lists, or headers. "
                "Keep it simple and conversational like a quick explanation to a friend."
            )
        else:
            system_content = (
                "You are a helpful, friendly AI assistant. "
                "Be concise but thorough. Use markdown formatting when helpful. "
                "If asked about real-time data (weather, news, stocks), note that you can search the web for current info."
            )
        
        # Build messages for main response
        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]
        
        # Add conversation history as context if provided (string from Redis)
        if conversation_history and isinstance(conversation_history, str) and conversation_history.strip():
            messages.append({
                "role": "system",
                "content": f"Previous conversation context (use this to understand follow-ups):\n{conversation_history}"
            })
        
        # Add web context to user message if available
        user_message = message
        if web_context:
            user_message = f"{message}\n{web_context}"
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Use GlobalLLMService for the main chat response
            # Handles rate limiting, failover (GROQ → Gemini), and telemetry automatically
            assistant_message = await llm.call_with_messages_async(
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                timeout=30.0,
                trace_name="chat-main-response"
            )
            
            result = {
                "success": True,
                "data": assistant_message
            }
            
            # Add web sources if used
            if web_sources:
                result["sources"] = web_sources
            
            # For knowledge questions, fetch image and related questions
            if is_knowledge_q and topic:
                # Fetch image using ImageSearchService (synchronous)
                image_data = self._fetch_image(topic)
                if image_data:
                    result["image_data"] = image_data
                
                # Generate related questions
                related = await self._generate_related_questions(message, assistant_message)
                if related:
                    result["related_questions"] = related
            
            return result
            
        except Exception as e:
            logger.error(f"Chat LLM error: {e}")
            return {
                "success": False,
                "error": f"Failed to get response: {str(e)}"
            }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format chat response for display."""
        if not result.get("success"):
            return f"❌ {result.get('error', 'Unknown error')}"
        
        return result.get("data", "I'm not sure how to respond to that.")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
