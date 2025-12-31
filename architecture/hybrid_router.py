"""
Hybrid Router - Modular MCP Architecture

This module provides a clean, modular routing layer that:
1. Uses Rasa for fast small-talk intent classification
2. Routes everything else to the LLM for tool selection
3. All tools are discovered via MCP - NO hardcoded tool logic

Architecture follows Copilot/Cursor/Claude Desktop pattern:
- Tools self-describe via MCP schema
- LLM decides which tool to use
- Router is completely tool-agnostic
- New tools just need to be added to mcp_servers.json

Benefits:
- Zero code changes needed to add new tools
- Fully modular and extensible
- Tools can request permissions before execution
- Consistent interface for all tools
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import httpx

from architecture.router import LLMRouter
from architecture.tool_logger import get_tool_logger, ToolType
from architecture.form_registry import FormHandlerRegistry, FormSubmission
from architecture.conversation_memory import get_conversation_memory

logger = logging.getLogger(__name__)
tool_logger = get_tool_logger()


# ============================================================================
# Configuration
# ============================================================================

# Intents that Rasa handles (small talk only)
RASA_HANDLED_INTENTS = {
    "greet",
    "goodbye", 
    "affirm",
    "deny",
    "bot_challenge",
    "bot_capabilities",
}

# Minimum confidence for Rasa to handle the intent
RASA_CONFIDENCE_THRESHOLD = 0.7

# Keywords that indicate user wants to cancel current operation
EXIT_KEYWORDS = {
    "cancel", "stop", "exit", "quit", "nevermind", "never mind",
    "forget it", "start over", "go back", "new question", "something else"
}

# Session timeout in seconds (5 minutes)
SESSION_TIMEOUT = 300

# Sticky MCP tools that should maintain conversation context
STICKY_MCP_TOOLS: Set[str] = {
    "leave.analyze_leave_request",
    "leave.get_leave_request_card",
    "leave.validate_leave",
}

# MCP server names for sticky sessions
STICKY_MCP_SERVERS: Set[str] = {
    "leave",  # Leave management server
}

# Local plugins that support sticky context (file uploads, surveys, etc.)
STICKY_LOCAL_PLUGINS: Set[str] = {
    "file",  # File tool for uploaded documents/images
    "survey",
    "contact_form",
}

# Keywords that indicate user is asking about uploaded files
FILE_CONTEXT_KEYWORDS: Set[str] = {
    "file", "document", "pdf", "image", "picture", "photo", "upload",
    "uploaded", "attachment", "summary", "summarize", "describe",
    "tell me about", "explain", "analyze",
    "the file", "the document", "the image", "it says",
    "content", "main point", "about this", "in the file", "in this",
    "الملف", "المستند", "الصورة", "ملخص", "لخص",  # Arabic keywords
}

# Keywords that indicate user is switching to a DIFFERENT tool (exit file context)
FILE_CONTEXT_EXIT_KEYWORDS: Set[str] = {
    "weather", "forecast", "temperature", "rain", "sunny", "cloudy",
    "news", "headlines", "latest news",
    "leave", "vacation", "pto", "time off", "holiday",
    "survey", "feedback", "questionnaire",
    "contact", "phone", "address", "email",
    "sales", "revenue", "products", "customers", "orders", "database",
    "tax", "budget", "finance", "government",
    "hello", "hi", "hey", "good morning", "good evening",
    "bye", "goodbye", "thanks", "thank you",
    "الطقس", "أخبار", "إجازة",  # Arabic: weather, news, leave
}

@dataclass  
class HybridRouterConfig:
    """Configuration for the Hybrid Router."""
    # Rasa settings
    rasa_url: str = "http://localhost:5005"
    rasa_timeout: float = 5.0
    rasa_confidence_threshold: float = RASA_CONFIDENCE_THRESHOLD
    
    # Enable/disable Rasa layer
    enable_rasa_layer: bool = True
    
    # Use predefined responses instead of calling Rasa actions
    use_predefined_responses: bool = True
    
    # Session timeout
    session_timeout: float = SESSION_TIMEOUT


@dataclass
class MCPToolSession:
    """Tracks a sticky MCP tool session."""
    tool_name: str  # Full tool name like "leave.analyze_leave_request"
    server_name: str  # MCP server name like "leave"
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)  # Tool-specific context
    
    def is_expired(self, timeout: float = SESSION_TIMEOUT) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > timeout
    
    def touch(self) -> None:
        """Update activity timestamp."""
        self.last_activity = time.time()
        self.turn_count += 1


@dataclass
class ConversationSession:
    """Tracks conversation context for a user session."""
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    pending_confirmation: Optional[Dict[str, Any]] = None  # For permission requests
    
    # MCP tool session (for stickiness)
    active_mcp_session: Optional[MCPToolSession] = None
    
    # Sticky context for local plugins (survey, contact_form, etc.)
    sticky_context: Optional[Dict[str, Any]] = None
    
    def is_expired(self, timeout: float = SESSION_TIMEOUT) -> bool:
        """Check if session has expired due to inactivity."""
        return (time.time() - self.last_activity) > timeout
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
        self.turn_count += 1
    
    def set_sticky_context(self, context: Dict[str, Any]) -> None:
        """Set sticky context for conversational chain tools."""
        self.sticky_context = context
        logger.debug(f"Set sticky context: {context.get('tool', 'unknown')}")
    
    def clear_sticky_context(self) -> None:
        """Clear sticky context."""
        if self.sticky_context:
            logger.debug(f"Cleared sticky context: {self.sticky_context.get('tool', 'unknown')}")
        self.sticky_context = None
    
    def has_sticky_context(self) -> bool:
        """Check if there's an active sticky context."""
        return self.sticky_context is not None
    
    # Aliases for consistent naming with the new pattern
    def has_active_sticky_context(self) -> bool:
        """Check if there's an active sticky context (alias)."""
        return self.has_sticky_context()
    
    def start_sticky_context(self, tool_name: str, initial_state: Dict[str, Any], language: str = None) -> None:
        """Start a sticky context session for a local plugin."""
        self.sticky_context = {
            "tool_name": tool_name,
            "state": initial_state,
            "language": language,  # Store language from the start
            "started_at": time.time(),
            "last_activity": time.time()
        }
        logger.info(f"📌 Started sticky context for: {tool_name} (language: {language})")
    
    def update_sticky_context(self, new_context: Dict[str, Any]) -> None:
        """Update the sticky context with new state."""
        if new_context:
            tool_name = new_context.get("tool_name") or new_context.get("tool")
            state = new_context.get("state") or new_context
            
            # PRESERVE LANGUAGE: Get language from new context or keep existing
            language = new_context.get("language")
            if not language and self.sticky_context:
                language = self.sticky_context.get("language")
            
            self.sticky_context = {
                "tool_name": tool_name,
                "state": state,
                "language": language,  # Preserve language across updates
                "started_at": self.sticky_context.get("started_at", time.time()) if self.sticky_context else time.time(),
                "last_activity": time.time()
            }
            logger.debug(f"Updated sticky context for: {tool_name} (language: {language})")
    
    def end_sticky_context(self, reason: str = "") -> None:
        """End the current sticky context session."""
        if self.sticky_context:
            tool_name = self.sticky_context.get("tool_name", "unknown")
            logger.info(f"📌 Ended sticky context for: {tool_name} ({reason})")
            self.sticky_context = None
    
    def start_mcp_session(self, tool_name: str, server_name: str) -> MCPToolSession:
        """Start a sticky MCP tool session."""
        self.active_mcp_session = MCPToolSession(
            tool_name=tool_name,
            server_name=server_name
        )
        tool_logger.log_session_start(f"{server_name}_session", tool_name)
        return self.active_mcp_session
    
    def end_mcp_session(self, reason: str = "") -> None:
        """End the current MCP tool session."""
        if self.active_mcp_session:
            tool_logger.log_session_end(
                f"{self.active_mcp_session.server_name}_session", 
                reason
            )
            self.active_mcp_session = None
    
    def has_active_mcp_session(self) -> bool:
        """Check if there's an active MCP session."""
        if self.active_mcp_session is None:
            return False
        if self.active_mcp_session.is_expired():
            self.end_mcp_session("expired")
            return False
        return True


class HybridRouter:
    """
    Modular Hybrid Router - Tool-Agnostic Design
    
    This router does NOT contain any tool-specific code.
    All tools are discovered and executed via MCP.
    
    Flow:
    1. Parse with Rasa NLU
    2. If small-talk intent with high confidence → predefined response
    3. Otherwise → LLM Router for tool selection and execution
    4. LLM router uses MCP to discover and call tools
    
    Adding New Tools:
    1. Create an MCP server (in mcp_servers/ directory)
    2. Add it to mcp_servers.json with enabled: true
    3. Done! The LLM will discover and use it automatically
    """
    
    def __init__(self, config: Optional[HybridRouterConfig] = None):
        self.config = config or HybridRouterConfig()
        
        # Override from environment
        self.config.rasa_url = os.getenv("RASA_URL", self.config.rasa_url)
        
        # Initialize LLM Router (lazy)
        self._llm_router: Optional[LLMRouter] = None
        self._client: Optional[httpx.AsyncClient] = None
        
        # Session tracking (minimal - just for context, not tool-specific)
        self._sessions: Dict[str, ConversationSession] = {}
        
        # Response rotation index
        self._response_index: Dict[str, int] = {}
        
        logger.info(f"✅ HybridRouter initialized (Rasa: {'enabled' if self.config.enable_rasa_layer else 'disabled'})")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def _get_llm_router(self) -> LLMRouter:
        """Get or create LLM Router instance."""
        if self._llm_router is None:
            self._llm_router = LLMRouter()
        return self._llm_router
    
    async def _get_initialized_llm_router(self) -> LLMRouter:
        """
        Get LLM Router with MCP servers initialized.
        
        This is the key method - it ensures MCP servers are connected
        before routing, so the LLM knows about all available tools.
        """
        router = self._get_llm_router()
        
        # Initialize MCP connections if not done yet
        if not router._mcp_initialized:
            await router.initialize()
            if router._mcp_host:
                logger.info(f"🔌 MCP initialized: {router._mcp_host.connected_servers} servers, {router._mcp_host.total_tools} tools")
        
        return router
    
    # ========== Session Management ==========
    
    def get_session(self, sender_id: str) -> ConversationSession:
        """Get or create a conversation session."""
        session = self._sessions.get(sender_id)
        if session is None or session.is_expired(self.config.session_timeout):
            session = ConversationSession()
            self._sessions[sender_id] = session
        return session
    
    def clear_session(self, sender_id: str) -> None:
        """Clear session for a sender."""
        if sender_id in self._sessions:
            del self._sessions[sender_id]
    
    def _should_cancel(self, message: str) -> bool:
        """Check if user wants to cancel current operation."""
        msg_lower = message.lower().strip()
        return any(keyword in msg_lower for keyword in EXIT_KEYWORDS)
    
    # ========== Rasa Integration ==========
    
    async def parse_with_rasa(self, message: str) -> Tuple[str, float, Dict]:
        """Parse a message using Rasa NLU."""
        if not self.config.enable_rasa_layer:
            return self._keyword_match(message)
        
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.config.rasa_url}/model/parse",
                json={"text": message},
                timeout=self.config.rasa_timeout
            )
            
            if response.status_code != 200:
                return self._keyword_match(message)
            
            data = response.json()
            intent = data.get("intent", {})
            return (
                intent.get("name", "nlu_fallback"),
                intent.get("confidence", 0.0),
                data
            )
            
        except Exception as e:
            logger.warning(f"Rasa parse error: {e}")
            return self._keyword_match(message)
    
    def _keyword_match(self, message: str) -> Tuple[str, float, Dict]:
        """Simple keyword-based intent matching (fallback)."""
        msg_lower = message.lower().strip()
        
        # Greeting patterns
        if any(msg_lower.startswith(p) or msg_lower == p for p in 
               ["hi", "hello", "hey", "good morning", "good evening", "howdy"]):
            return ("greet", 0.95, {})
        
        # Goodbye patterns
        if any(p in msg_lower for p in ["bye", "goodbye", "see you", "farewell"]):
            return ("goodbye", 0.95, {})
        
        # Bot challenge patterns
        if any(p in msg_lower for p in ["are you a bot", "are you human", "what are you"]):
            return ("bot_challenge", 0.95, {})
        
        # Capabilities patterns
        if any(p in msg_lower for p in ["what can you do", "your capabilities", "help me"]):
            return ("bot_capabilities", 0.95, {})
        
        # Affirmation/Denial
        if msg_lower in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
            return ("affirm", 0.90, {})
        if msg_lower in ["no", "nope", "nah"]:
            return ("deny", 0.90, {})
        
        return ("nlu_fallback", 0.0, {})
    
    def should_use_rasa(self, intent: str, confidence: float) -> bool:
        """Determine if Rasa should handle this intent."""
        return (
            self.config.enable_rasa_layer
            and intent in RASA_HANDLED_INTENTS
            and confidence >= self.config.rasa_confidence_threshold
        )
    
    async def handle_rasa_intent(self, intent: str, parse_data: Dict) -> Dict[str, Any]:
        """
        Handle an intent by calling Rasa to get the response.
        
        This calls Rasa's webhook endpoint to get the trained response
        from domain.yml instead of using hardcoded responses.
        """
        # Try to get response from Rasa first
        response_text = await self._get_rasa_response(intent, parse_data)
        
        # Fallback to predefined if Rasa fails
        if not response_text:
            response_text = self._get_predefined_response(intent)
            logger.debug(f"Using fallback response for intent: {intent}")
        
        return {
            "tool": None,
            "params": {},
            "result": {"success": True},
            "response": response_text,
            "routing": {
                "handler": "rasa",
                "intent": intent,
                "confidence": parse_data.get("intent", {}).get("confidence", 1.0)
            }
        }
    
    async def _get_rasa_response(self, intent: str, parse_data: Dict) -> Optional[str]:
        """
        Get response from Rasa's trained model.
        
        Calls the Rasa webhook endpoint to get responses defined in domain.yml.
        """
        try:
            client = await self._get_client()
            
            # Map intent to utter action
            utter_action = f"utter_{intent}" if intent != "bot_challenge" else "utter_iamabot"
            if intent == "bot_capabilities":
                utter_action = "utter_capabilities"
            elif intent == "greet":
                utter_action = "utter_greet"
            elif intent == "goodbye":
                utter_action = "utter_goodbye"
            
            # Call Rasa's action endpoint to trigger the response
            response = await client.post(
                f"{self.config.rasa_url}/webhooks/rest/webhook",
                json={
                    "sender": "hybrid_router",
                    "message": parse_data.get("text", "")
                },
                timeout=self.config.rasa_timeout
            )
            
            if response.status_code == 200:
                messages = response.json()
                if messages and len(messages) > 0:
                    # Get the text from the first message
                    return messages[0].get("text", "")
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get Rasa response: {e}")
            return None
    
    # ========== Form Submission Handling (Modular) ==========
    
    async def _handle_form_submission(
        self,
        message: str,
        context: Dict,
        sender_id: str
    ) -> Dict[str, Any]:
        """
        Handle form submissions from Adaptive Cards using the form registry.
        
        This is now fully modular - form handlers register themselves
        and we just look them up. No hardcoding needed!
        """
        logger.info(f"📝 Handling form submission: {message}")
        
        # Get the form handler registry
        registry = FormHandlerRegistry.get_instance()
        
        # Look up handler for this action
        handler = registry.get_handler(message)
        
        if handler:
            # Create submission object
            submission = FormSubmission(
                action=message,
                data=context.get("data", {}),
                context=context,
                sender_id=sender_id
            )
            
            # Get LLM router for executing tools
            llm_router = await self._get_initialized_llm_router()
            
            # Define the tool executor callback
            async def execute_tool(tool_name: str, params: Dict) -> Dict:
                return await llm_router.execute_tool(tool_name, params)
            
            # Handle the form
            result = await handler.handle(submission, execute_tool)
            
            # Convert result to dict
            return result.to_dict()
        else:
            logger.warning(f"No handler registered for form action: {message}")
            return {
                "tool": None,
                "params": {},
                "result": {"success": False, "error": "Unknown form type"},
                "response": "I couldn't process that form submission. Please try again.",
                "routing": {"handler": "form_submission", "form_type": message}
            }
    
    async def _handle_sticky_context_followup(
        self,
        message: str,
        session: ConversationSession,
        merged_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Handle follow-up messages in an active sticky context (survey, contact form, etc.).
        
        NOTE: For data collection (contact_form, survey), we use the ORIGINAL message
        (not translated) to preserve user input like names and addresses in their 
        original language/script.
        
        LANGUAGE PRESERVATION: The sticky_context stores the language from when the
        conversation started, so numeric inputs (phone numbers) don't reset the language.
        """
        if not session.sticky_context:
            return None
        
        sticky = session.sticky_context
        tool_name = sticky.get("tool_name")
        state = sticky.get("state", {})
        
        # LANGUAGE PRESERVATION: Use language from sticky_context if available
        # This ensures that entering a phone number (digits only) doesn't switch to English
        sticky_language = sticky.get("language")
        if sticky_language:
            merged_context["original_language"] = sticky_language
            logger.debug(f"📌 Using sticky context language: {sticky_language}")
        
        logger.info(f"📌 Sticky context follow-up: {tool_name}")
        
        try:
            # Get the tool from registry
            llm_router = await self._get_initialized_llm_router()
            tool = llm_router.registry.get_tool(tool_name)
            
            if not tool:
                logger.warning(f"Sticky context tool not found: {tool_name}")
                session.end_sticky_context("tool not found")
                return None
            
            # For data collection tools, use the ORIGINAL message to preserve
            # user input in its original language/script (e.g., Arabic names)
            # The translated message is used for routing, but actual user data
            # should be stored as provided.
            user_response = message  # Default to the (possibly translated) message
            if tool_name in ("contact_form", "survey"):
                # Use original message for data collection to preserve Arabic/non-English input
                original_message = merged_context.get("original_message")
                if original_message:
                    user_response = original_message
                    logger.debug(f"Using original message for {tool_name}: {user_response[:50]}...")
            
            # Prepare parameters for collect action
            # Include the user's response and the current state
            params = {
                "action": "collect",
                "user_response": user_response,
                "sender_id": merged_context.get("_sender_id", "anonymous"),
                **state  # Include state fields (current_index, questions, answers, etc.)
            }
            
            # Merge with context (but don't override explicit params)
            for key, value in merged_context.items():
                if key not in params:
                    params[key] = value
            
            # Execute the tool
            result = await tool.execute(**params)
            
            # Check if result has sticky_context to continue
            # Tools may return response in "response" or "data" field
            response = result.get("response") or result.get("data", "")
            new_sticky = result.get("sticky_context")
            
            return {
                "tool": tool_name,
                "params": params,
                "result": result,
                "response": response,
                "routing": {"handler": "sticky_context", "tool": tool_name},
                "sticky_context": new_sticky
            }
            
        except Exception as e:
            logger.error(f"Error in sticky context follow-up: {e}", exc_info=True)
            session.end_sticky_context(f"error: {e}")
            return {
                "tool": tool_name,
                "params": {},
                "result": {"success": False, "error": str(e)},
                "response": f"Sorry, something went wrong. Let me know if you'd like to try again.",
                "routing": {"handler": "sticky_context_error", "tool": tool_name}
            }
    
    async def _check_file_context(
        self,
        message: str,
        session: ConversationSession,
        merged_context: Dict[str, Any],
        sender_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if there's an active file context and if the message is about the uploaded file.
        Routes to file tool instead of secure_rag for file-related queries.
        """
        try:
            # Get file service to check for active file context
            from shared_utils import get_service_manager
            file_service = get_service_manager().get_file_service()
            file_context = file_service.get_last_file_context()
            
            # No active file context
            if not file_context.get("file_path"):
                return None
            
            message_lower = message.lower()
            
            # FIRST: Check if user is switching to a different topic/tool
            # If they mention weather, news, leave, etc. - exit file context
            is_different_topic = any(kw in message_lower for kw in FILE_CONTEXT_EXIT_KEYWORDS)
            if is_different_topic:
                logger.info(f"📄 File context: detected different topic, letting other tools handle")
                return None
            
            # Check if message is specifically about the file
            is_file_query = any(kw in message_lower for kw in FILE_CONTEXT_KEYWORDS)
            
            if not is_file_query:
                return None
            
            logger.info(f"📄 File context detected, routing to file tool")
            tool_logger.log_routing_decision(
                message=message,
                intent="file_followup",
                confidence=1.0,
                selected_tool="file",
                tool_type=ToolType.PLUGIN,
                reason="Active file context - bypassing secure_rag"
            )
            
            # Get the file tool and execute followup
            llm_router = await self._get_initialized_llm_router()
            file_tool = llm_router.registry.get_tool("file")
            
            if not file_tool:
                logger.warning("File tool not found in registry")
                return None
            
            # Execute file tool with followup action
            result = await file_tool.execute(
                action="followup",
                question=message,
                language=""  # Auto-detect
            )
            
            response = result.get("data", "")
            
            return {
                "tool": "file",
                "params": {"action": "followup", "question": message},
                "result": result,
                "response": response,
                "routing": {"handler": "file_context", "tool": "file"}
            }
            
        except Exception as e:
            logger.error(f"Error checking file context: {e}", exc_info=True)
            return None
    
    # ========== Main Processing ==========
    
    async def _record_exchange(
        self,
        sender_id: str,
        message: str,
        result: Dict[str, Any]
    ) -> None:
        """Record the exchange in conversation memory."""
        try:
            memory = await get_conversation_memory()
            response = result.get("response", "")
            tool_name = result.get("tool")
            tool_params = result.get("params")
            tool_result = result.get("result")
            
            # Determine topic from tool name
            topic = None
            if tool_name and "." in tool_name:
                topic = tool_name.split(".")[0]  # Server name
            
            # Check if result contains an adaptive card
            # If so, we need to store it in the message metadata so it can be re-rendered
            metadata = {}
            if tool_result and isinstance(tool_result, dict):
                # Check for adaptive card in result
                if tool_result.get("type") == "adaptive_card":
                    metadata["type"] = "adaptive_card"
                    metadata["card"] = tool_result.get("card")
                    metadata["metadata"] = tool_result.get("metadata")
                # Check for adaptive card in data field
                elif tool_result.get("data") and isinstance(tool_result["data"], dict) and tool_result["data"].get("type") == "adaptive_card":
                    card_data = tool_result["data"]
                    metadata["type"] = "adaptive_card"
                    metadata["card"] = card_data.get("card")
                    metadata["metadata"] = card_data.get("metadata")
            
            # Add metadata to the exchange
            # We need to modify add_exchange to accept metadata or update the message afterwards
            # Since add_exchange doesn't accept metadata directly, we'll use entities for now
            # or update the message object if possible.
            # Actually, let's update conversation_memory.py to accept metadata in add_exchange/add_session_exchange
            
            # For now, let's pass it via entities if we can't change the signature easily
            # But wait, we can change the signature! I'll update conversation_memory.py first.
            
            await memory.add_exchange(
                sender_id=sender_id,
                user_message=message,
                assistant_response=response,
                tool_name=tool_name,
                tool_params=tool_params,
                tool_result=tool_result,
                topic=topic,
                metadata=metadata if metadata else None
            )
        except Exception as e:
            logger.warning(f"Failed to record conversation: {e}")
    
    async def process_message(
        self,
        message: str,
        context: Optional[Dict] = None,
        sender_id: str = "user"
    ) -> Dict[str, Any]:
        """
        Process a message through the hybrid routing pipeline.
        
        This is the main entry point. The router is completely tool-agnostic.
        All tool selection and execution is delegated to the LLM Router.
        
        Args:
            message: The user's message
            context: Optional context (user info, form data, etc.)
            sender_id: User/session identifier
            
        Returns:
            Dict with 'response', 'tool', 'result', 'routing'
        """
        session = self.get_session(sender_id)
        session.touch()
        
        # Get conversation history for LLM context
        memory = await get_conversation_memory()
        conversation_context = await memory.get_context_for_llm(sender_id)
        
        # Merge session context with provided context
        merged_context = {**session.context, **(context or {})}
        merged_context["_session_turn"] = session.turn_count
        merged_context["_conversation_history"] = conversation_context
        merged_context["_sender_id"] = sender_id
        
        # Check for cancel/exit keywords
        if self._should_cancel(message):
            session.pending_confirmation = None
            # End any active MCP session
            if session.has_active_mcp_session():
                session.end_mcp_session("user cancelled")
            # End any active sticky context (survey, contact form, etc.)
            if session.has_active_sticky_context():
                session.end_sticky_context("user cancelled")
            return {
                "tool": None,
                "params": {},
                "result": {"success": True},
                "response": "No problem! What else can I help you with?",
                "routing": {"handler": "cancel"}
            }
        
        # Check for pending confirmation (permission system)
        if session.pending_confirmation:
            result = await self._handle_confirmation(message, session, merged_context)
            if result:
                return result
        
        # Handle form submissions (from Adaptive Cards)
        if message.startswith("/submit_"):
            return await self._handle_form_submission(message, merged_context, sender_id)
        
        # ===== Check for active sticky context (local plugins like survey/contact) =====
        if session.has_active_sticky_context():
            result = await self._handle_sticky_context_followup(message, session, merged_context)
            if result:
                # Check if sticky context should continue or end
                if result.get("sticky_context"):
                    session.update_sticky_context(result["sticky_context"])
                else:
                    session.end_sticky_context("completed")
                # Record the exchange
                await self._record_exchange(sender_id, message, result)
                return result
        
        # ===== Check for active MCP sticky session =====
        if session.has_active_mcp_session():
            mcp_session = session.active_mcp_session
            
            # Check if this is a new topic (should exit session)
            if self._is_new_topic(message, mcp_session.server_name):
                session.end_mcp_session("new topic detected")
            else:
                # Continue in sticky session
                tool_logger.log_session_continue(
                    f"{mcp_session.server_name}_session",
                    mcp_session.turn_count + 1,
                    message[:50]
                )
                mcp_session.touch()
                
                # Handle follow-up in MCP session
                result = await self._handle_mcp_session_followup(
                    message, session, mcp_session, merged_context
                )
                if result:
                    return result
        
        # ===== Check for file context (uploaded files get priority) =====
        file_context_result = await self._check_file_context(message, session, merged_context, sender_id)
        if file_context_result:
            logger.info("📄 Routed to file tool (active file context)")
            await self._record_exchange(sender_id, message, file_context_result)
            return file_context_result
        
        # Try Rasa NLU first (for small talk)
        intent, confidence, parse_data = await self.parse_with_rasa(message)
        
        logger.info(f"🔍 Rasa: {intent} ({confidence:.2f})")
        
        if self.should_use_rasa(intent, confidence):
            logger.info(f"✅ Handled by Rasa: {intent}")
            tool_logger.log_routing_decision(
                message=message,
                intent=intent,
                confidence=confidence,
                selected_tool=intent,
                tool_type=ToolType.RASA,
                reason="Small talk handled by Rasa"
            )
            return await self.handle_rasa_intent(intent, parse_data)
        
        # Route to LLM Router for tool selection
        logger.info("🤖 Routing to LLM Router")
        
        # IMPORTANT: Use the async version that initializes MCP connections
        llm_router = await self._get_initialized_llm_router()
        result = await llm_router.route_and_execute(message, context=merged_context)
        
        # Determine the actual tool(s) used
        tool_name = result.get("tool", "")
        actual_mcp_tool = None
        sticky_context_from_code = None
        
        # For code_execution mode, check what MCP tools were actually called
        if tool_name == "code_execution":
            tool_calls = result.get("result", {}).get("tool_calls", [])
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_result = tc.get("result", {})
                
                if "." in tc_name:  # MCP tool
                    if not actual_mcp_tool:
                        actual_mcp_tool = tc_name
                
                # Check for sticky_context in tool call results
                if tc_result.get("sticky_context"):
                    sticky_context_from_code = tc_result["sticky_context"]
        
        # Check if an MCP tool was called and if it should start a sticky session
        mcp_tool_to_check = actual_mcp_tool or (tool_name if "." in tool_name else None)
        if mcp_tool_to_check:
            server_name = mcp_tool_to_check.split(".")[0]
            if server_name in STICKY_MCP_SERVERS:
                # Start sticky session for this MCP server
                session.start_mcp_session(mcp_tool_to_check, server_name)
                # Store any context from the result
                if result.get("result", {}).get("entities"):
                    session.active_mcp_session.context["entities"] = result["result"]["entities"]
        
        # Check if a local plugin wants to start a sticky context (survey, contact form, etc.)
        # Look in both direct result and code_execution tool calls
        sticky_to_use = sticky_context_from_code or result.get("sticky_context")
        if sticky_to_use:
            if sticky_to_use.get("tool_name") and sticky_to_use.get("state"):
                logger.info(f"📌 Starting sticky context for: {sticky_to_use.get('tool_name')}")
                session.start_sticky_context(
                    tool_name=sticky_to_use["tool_name"],
                    initial_state=sticky_to_use["state"]
                )
        
        # Log the final routing decision - only once, with the actual tool that was called
        log_tool_name = actual_mcp_tool or tool_name
        if log_tool_name:
            if "." in log_tool_name:
                server_name = log_tool_name.split(".")[0]
                tool_logger.log_routing_decision(
                    message=message,
                    intent=intent,
                    confidence=confidence,
                    selected_tool=log_tool_name,
                    tool_type=ToolType.MCP_SERVER,
                    reason=f"MCP server: {server_name}"
                )
            elif log_tool_name == "code_execution":
                # Code execution - show what tools it actually called
                tool_calls = result.get("result", {}).get("tool_calls", [])
                called_tools = [tc.get("name") for tc in tool_calls if tc.get("name")]
                if called_tools:
                    tool_logger.log_routing_decision(
                        message=message,
                        intent=intent,
                        confidence=confidence,
                        selected_tool=", ".join(called_tools),
                        tool_type=ToolType.CODE_EXECUTION,
                        reason="LLM generated code to call tools"
                    )
            else:
                tool_logger.log_routing_decision(
                    message=message,
                    intent=intent,
                    confidence=confidence,
                    selected_tool=log_tool_name,
                    tool_type=ToolType.PLUGIN,
                    reason="Local plugin"
                )
        
        # Update session context with any returned context
        if result.get("result", {}).get("context"):
            session.context.update(result["result"]["context"])
        
        # Check if tool requested confirmation (permission system)
        if result.get("requires_confirmation"):
            session.pending_confirmation = result
            confirmation_result = {
                "tool": result.get("tool"),
                "params": result.get("params", {}),
                "result": {"pending": True},
                "response": result.get("confirmation_message", "Do you want me to proceed?"),
                "routing": {
                    "handler": "llm",
                    "awaiting_confirmation": True
                }
            }
            await self._record_exchange(sender_id, message, confirmation_result)
            return confirmation_result
        
        # Add routing info
        result["routing"] = result.get("routing", {})
        result["routing"]["handler"] = "llm"
        result["routing"]["rasa_intent"] = intent
        result["routing"]["rasa_confidence"] = confidence
        
        # Record the exchange in conversation memory
        await self._record_exchange(sender_id, message, result)
        
        return result
    
    def _is_new_topic(self, message: str, current_server: str) -> bool:
        """
        Check if the message indicates a completely new topic.
        This helps break out of sticky sessions when user changes subject.
        """
        msg_lower = message.lower()
        
        # Keywords that indicate new topics (not related to current server)
        if current_server == "leave":
            new_topic_keywords = [
                "weather", "temperature", "forecast", "rain", "sunny",
                "search", "find", "look up", "google",
                "database", "sql", "query", "data", "report",
                "brochure", "document", "file",
                "contact", "email", "call", "phone"
            ]
            if any(keyword in msg_lower for keyword in new_topic_keywords):
                return True
        
        return False
    
    async def _handle_mcp_session_followup(
        self,
        message: str,
        session: ConversationSession,
        mcp_session: MCPToolSession,
        context: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Handle follow-up messages in an active MCP sticky session.
        
        This is now generic - uses the form registry's follow-up patterns
        and conversation memory instead of hardcoded per-server logic.
        """
        server_name = mcp_session.server_name
        msg_lower = message.lower()
        
        # Get conversation memory for context
        memory = await get_conversation_memory()
        sender_id = context.get("sender_id", "user")
        conv_context = await memory.get_context_for_llm(sender_id)
        
        # Check for affirmative responses (confirming action)
        affirmative = ["yes", "confirm", "submit", "approve", "ok", "okay", "sure", "proceed"]
        if any(word in msg_lower for word in affirmative) and len(msg_lower.split()) <= 5:
            session.end_mcp_session("user confirmed")
            return {
                "tool": f"{server_name}.confirmation",
                "params": {},
                "result": {"success": True},
                "response": "✅ Your request has been noted. Is there anything else I can help you with?",
                "routing": {"handler": "mcp_session_followup", "action": "confirm"}
            }
        
        # Check if this matches a follow-up pattern from the registry
        registry = FormHandlerRegistry.get_instance()
        pattern = registry.get_followup_pattern(server_name)
        
        if pattern and pattern.matches(message):
            logger.info(f"🔄 {server_name} session followup: re-routing with conversation context")
            
            # Re-route through LLM with conversation history
            # The LLM will understand this is a follow-up because of the context
            llm_router = await self._get_initialized_llm_router()
            
            # Add conversation context to help LLM understand follow-up
            enhanced_context = {**context}
            enhanced_context["conversation_history"] = conv_context
            enhanced_context["active_topic"] = server_name
            enhanced_context["is_followup"] = True
            
            # Let the LLM route this naturally
            result = await llm_router.route_and_execute(message, context=enhanced_context)
            
            if result.get("tool"):
                result["routing"] = {"handler": "mcp_session_followup", "action": "modify"}
                return result
        
        # Not a recognizable followup - let it re-route normally
        return None

    async def _handle_confirmation(
        self,
        message: str,
        session: ConversationSession,
        context: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Handle user response to a pending confirmation.
        
        This implements the permission system like Copilot/Cursor.
        """
        pending = session.pending_confirmation
        msg_lower = message.lower().strip()
        
        # Check for affirmative response
        affirm_words = ["yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "proceed", "do it", "confirm"]
        if any(word in msg_lower for word in affirm_words):
            # User approved - execute the pending action
            session.pending_confirmation = None
            
            llm_router = await self._get_initialized_llm_router()
            result = await llm_router.execute_confirmed_action(pending, context)
            
            result["routing"] = {
                "handler": "confirmation",
                "action": "approved"
            }
            return result
        
        # Check for negative response
        deny_words = ["no", "nope", "nah", "don't", "stop", "cancel"]
        if any(word in msg_lower for word in deny_words):
            session.pending_confirmation = None
            return {
                "tool": pending.get("tool"),
                "params": {},
                "result": {"success": False, "cancelled": True},
                "response": "Okay, I won't do that. What else can I help with?",
                "routing": {
                    "handler": "confirmation",
                    "action": "denied"
                }
            }
        
        # Not a clear yes/no - ask again or continue with new query
        return None
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._llm_router:
            await self._llm_router.close()
            self._llm_router = None


# ============================================================================
# Synchronous Wrapper for Flask
# ============================================================================

class SyncHybridRouter:
    """Synchronous wrapper for HybridRouter (use in Flask routes)."""
    
    def __init__(self, config: Optional[HybridRouterConfig] = None):
        self._async_router = HybridRouter(config)
    
    def _run_async(self, coro):
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    
    def process_message(
        self,
        message: str,
        context: Optional[Dict] = None,
        sender_id: str = "user"
    ) -> Dict[str, Any]:
        """Synchronous process_message."""
        return self._run_async(
            self._async_router.process_message(message, context, sender_id)
        )
    
    def close(self) -> None:
        """Close the router."""
        self._run_async(self._async_router.close())


# ============================================================================
# Global Instance
# ============================================================================

_hybrid_router: Optional[HybridRouter] = None


def get_hybrid_router() -> HybridRouter:
    """Get the global HybridRouter instance."""
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridRouter()
    return _hybrid_router


def get_sync_hybrid_router() -> SyncHybridRouter:
    """Get a synchronous hybrid router instance."""
    return SyncHybridRouter()