"""
Contact Form Tool Plugin - Collect and manage user contact information.

Collects user information through a conversational back-and-forth flow:
1. Ask for name
2. Ask for phone number  
3. Ask for address
Then saves to Redis and can display as an Adaptive Card.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)

_MEMORY_SESSIONS: Dict[str, Dict[str, Any]] = {}
_MEMORY_USER_INFO: Dict[str, Dict[str, Any]] = {}


class ContactFormTool(BaseTool):
    """
    Tool for collecting user contact information conversationally.
    
    Flow:
    1. User triggers collection -> Bot asks for name
    2. User responds -> Bot asks for phone
    3. User responds -> Bot asks for address
    4. User responds -> Bot saves and confirms
    
    Uses sticky_context to maintain state across turns.
    """
    
    # Collection states
    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_COMPLETE = "complete"
    
    # Questions to ask (in order)
    QUESTIONS = [
        {"id": "name", "text": "What is your **name**?", "emoji": "👤"},
        {"id": "phone", "text": "What is your **phone number**?", "emoji": "📞"},
        {"id": "address", "text": "What is your **address**?", "emoji": "🏠"},
    ]
    
    def __init__(self):
        self._redis_client = None
    
    async def _get_redis(self):
        """Get Redis client for storage."""
        if self._redis_client is False:
            return None
        if self._redis_client is None:
            try:
                import redis.asyncio as redis
                from config.config import ConfigManager
                config = ConfigManager()
                client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    password=config.redis_password,
                    decode_responses=True
                )
                await client.ping()
                self._redis_client = client
            except Exception as exc:
                logger.warning("⚠️ Redis unavailable for contact_form; using in-memory store (%s)", exc)
                self._redis_client = False
                return None
        return self._redis_client
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="contact_form",
            description="Collect user contact information through conversation. Use 'start' to begin collecting info (name, phone, address). Use 'collect' to process responses. Use 'show' to display stored information as a card.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'start' to begin collection, 'collect' to process response, 'show' to display stored info",
                        "enum": ["start", "collect", "show"]
                    },
                    "user_response": {
                        "type": "string",
                        "description": "User's response to current question (used with 'collect' action)"
                    }
                },
                "required": ["action"]
            },
            examples=[
                "اجمع معلوماتي",
                "اعرض معلوماتي",
                "اعرض بطاقة الاتصال الخاصة بي",
                "Collect my info",
                "Collect my contact info",
                "Take my information",
                "I want to provide my details",
                "Show my info",
                "Show my contact card",
                "View my contact card",
                "Display my contact information",
                "What contact info do you have about me?",
                "Show my saved contact details"
            ],
            input_examples=[
                {"action": "start"},
                {"action": "collect", "user_response": "John Doe"},
                {"action": "show"}
            ],
            system_instruction=(
                "Use contact_form to collect or show the CURRENT user's contact information. "
                "If the user says 'collect my info' or wants to provide details, call contact_form with action='start'. "
                "If the user says 'show my info'/'show my contact card', call contact_form with action='show'. "
                "Do NOT use secure_rag for collecting user-provided contact info."
            ),
            code_example=(
                "result = await call_tool('contact_form', {'action': 'start'})\n"
                "if result['success']:\n"
                "    print(result['data'])\n"
                "else:\n"
                "    print(f\"Error: {result.get('error')}\")"
            ),
            defer_loading=True,
            always_loaded=False
        )
    
    async def execute(
        self,
        action: str,
        user_response: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle contact form operations.
        
        Args:
            action: 'start', 'collect', or 'show'
            user_response: User's answer to current question
            
        Returns:
            Dict with success, data, and optional sticky_context
        """
        # Get sender_id from kwargs
        sender_id = kwargs.get("sender_id", kwargs.get("_sender_id", "anonymous"))
        
        try:
            if action == "start":
                return await self._start_collection(sender_id)
            
            elif action == "collect":
                return await self._collect_response(
                    sender_id,
                    user_response,
                    kwargs.get("current_step"),
                    kwargs.get("collected_data", {})
                )
            
            elif action == "show":
                return await self._show_info(sender_id)
            
            else:
                return {
                    "success": False,
                    "data": "Unknown action. Use 'start', 'collect', or 'show'."
                }
                
        except Exception as e:
            logger.error(f"Contact form error: {e}", exc_info=True)
            return {
                "success": False,
                "data": f"❌ Error: {str(e)}"
            }
    
    async def _start_collection(self, sender_id: str) -> Dict[str, Any]:
        """Start collecting user information."""
        logger.info(f"📋 Starting contact info collection for {sender_id}")
        
        # Initialize collection data
        collected_data = {
            "sender_id": sender_id,
            "started_at": time.time(),
            "responses": {}
        }
        
        # Ask first question
        first_q = self.QUESTIONS[0]
        
        return {
            "success": True,
            "data": f"📋 **Let's collect your contact information!**\n\n"
                   f"{first_q['emoji']} {first_q['text']}",
            "sticky_context": {
                "tool_name": "contact_form",
                "state": {
                    "action": "collect",
                    "current_step": 0,
                    "collected_data": collected_data
                }
            }
        }
    
    async def _collect_response(
        self,
        sender_id: str,
        user_response: str,
        current_step: Optional[int],
        collected_data: Dict
    ) -> Dict[str, Any]:
        """Collect user's response and ask next question."""
        
        # Load from Redis if not provided
        if not collected_data or not collected_data.get("responses"):
            collected_data = await self._load_session(sender_id)
            if not collected_data:
                collected_data = {
                    "sender_id": sender_id,
                    "started_at": time.time(),
                    "responses": {}
                }
        
        current_step = current_step if current_step is not None else len(collected_data.get("responses", {}))
        
        # Store the response
        if current_step < len(self.QUESTIONS):
            question = self.QUESTIONS[current_step]
            
            # Validate response
            validation_error = self._validate_response(question["id"], user_response)
            if validation_error:
                # Ask same question again with error
                return {
                    "success": True,
                    "data": f"❌ {validation_error}\n\n{question['emoji']} {question['text']}",
                    "sticky_context": {
                        "tool_name": "contact_form",
                        "state": {
                            "action": "collect",
                            "current_step": current_step,
                            "collected_data": collected_data
                        }
                    }
                }
            
            # Store valid response
            collected_data["responses"][question["id"]] = user_response.strip()
        
        # Move to next step
        next_step = current_step + 1
        
        # Check if collection is complete
        if next_step >= len(self.QUESTIONS):
            return await self._complete_collection(sender_id, collected_data)
        
        # Save session
        await self._save_session(sender_id, collected_data)
        
        # Ask next question
        next_q = self.QUESTIONS[next_step]
        
        return {
            "success": True,
            "data": f"✅ Got it!\n\n{next_q['emoji']} {next_q['text']}",
            "sticky_context": {
                "tool_name": "contact_form",
                "state": {
                    "action": "collect",
                    "current_step": next_step,
                    "collected_data": collected_data
                }
            }
        }
    
    def _validate_response(self, field_id: str, response: str) -> Optional[str]:
        """Validate a response. Returns error message or None if valid."""
        response = response.strip() if response else ""
        
        if not response:
            return "Please provide a response."
        
        if field_id == "name":
            if len(response) < 2:
                return "Name must be at least 2 characters."
        
        elif field_id == "phone":
            digits = re.sub(r'\D', '', response)
            if len(digits) < 10:
                return "Phone number must have at least 10 digits."
        
        elif field_id == "address":
            if len(response) < 5:
                return "Please provide a more complete address."
        
        return None
    
    async def _complete_collection(self, sender_id: str, collected_data: Dict) -> Dict[str, Any]:
        """Complete the collection and save to permanent storage."""
        collected_data["completed_at"] = time.time()
        
        # Save permanently
        await self._save_user_info(sender_id, collected_data["responses"])
        
        # Clear session
        await self._clear_session(sender_id)
        
        # Build confirmation
        responses = collected_data.get("responses", {})
        
        return {
            "success": True,
            "data": f"🎉 **Information Saved Successfully!**\n\n"
                   f"👤 **Name:** {responses.get('name', 'N/A')}\n"
                   f"📞 **Phone:** {responses.get('phone', 'N/A')}\n"
                   f"🏠 **Address:** {responses.get('address', 'N/A')}\n\n"
                   f"✅ Your contact information has been stored. Say **'show my contact card'** anytime to see it!",
            "sticky_context": None  # Clear sticky context
        }
    
    async def _show_info(self, sender_id: str) -> Dict[str, Any]:
        """Show stored user information as a simple Adaptive Card."""
        user_info = await self._load_user_info(sender_id)
        
        if not user_info:
            return {
                "success": True,
                "data": "📋 No stored information found. Say 'collect my info' to provide your details!"
            }
        
        # Simple Adaptive Card with static avatar
        card = {
            "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [
                {
                    "type": "ColumnSet",
                    "columns": [
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [
                                {
                                    "type": "Image",
                                    "url": "/static/images/userAvatar.jpg",
                                    "size": "Medium",
                                    "style": "Person"
                                }
                            ]
                        },
                        {
                            "type": "Column",
                            "width": "stretch",
                            "verticalContentAlignment": "Center",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": user_info.get("name", "Unknown"),
                                    "size": "Large",
                                    "weight": "Bolder",
                                    "wrap": True
                                }
                            ]
                        }
                    ]
                },
                {
                    "type": "FactSet",
                    "spacing": "Medium",
                    "facts": [
                        {
                            "title": "📞 Phone",
                            "value": user_info.get("phone", "Not provided")
                        },
                        {
                            "title": "🏠 Address",
                            "value": user_info.get("address", "Not provided")
                        }
                    ]
                }
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": "✏️ Update Information",
                    "data": {
                        "action": "start_contact_collection",
                        "summary": "collect my info"
                    }
                }
            ]
        }
        
        return {
            "success": True,
            "data": {
                "type": "adaptive_card",
                "card": card,
                "message": f"📇 Here's your contact information, {user_info.get('name', 'user')}!",
                "metadata": {
                    "template": "contact_card"
                }
            }
        }
    
    # ==================== Redis Storage Methods ====================
    
    async def _save_session(self, sender_id: str, data: Dict) -> None:
        """Save active collection session to Redis."""
        redis_client = await self._get_redis()
        key = f"contact_session:{sender_id}"
        if not redis_client:
            _MEMORY_SESSIONS[sender_id] = dict(data)
            return
        try:
            await redis_client.set(key, json.dumps(data), ex=3600)  # 1 hour TTL
        except Exception as exc:
            logger.warning("⚠️ Redis write failed; using in-memory session store (%s)", exc)
            _MEMORY_SESSIONS[sender_id] = dict(data)
    
    async def _load_session(self, sender_id: str) -> Optional[Dict]:
        """Load active collection session from Redis."""
        redis_client = await self._get_redis()
        key = f"contact_session:{sender_id}"
        if not redis_client:
            return _MEMORY_SESSIONS.get(sender_id)
        try:
            data = await redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as exc:
            logger.warning("⚠️ Redis read failed; using in-memory session store (%s)", exc)
            return _MEMORY_SESSIONS.get(sender_id)
        return None
    
    async def _clear_session(self, sender_id: str) -> None:
        """Clear active collection session."""
        redis_client = await self._get_redis()
        key = f"contact_session:{sender_id}"
        _MEMORY_SESSIONS.pop(sender_id, None)
        if not redis_client:
            return
        try:
            await redis_client.delete(key)
        except Exception:
            return
    
    async def _save_user_info(self, sender_id: str, info: Dict) -> None:
        """Save user info permanently to Redis."""
        redis_client = await self._get_redis()
        key = f"contact_info:{sender_id}"
        info["updated_at"] = time.time()
        _MEMORY_USER_INFO[sender_id] = dict(info)
        if not redis_client:
            return
        try:
            await redis_client.set(key, json.dumps(info), ex=86400 * 365)  # 1 year TTL
        except Exception as exc:
            logger.warning("⚠️ Redis write failed; kept contact info in-memory (%s)", exc)
        logger.info(f"✅ Saved contact info for {sender_id}")
    
    async def _load_user_info(self, sender_id: str) -> Optional[Dict]:
        """Load user info from Redis."""
        redis_client = await self._get_redis()
        key = f"contact_info:{sender_id}"
        if not redis_client:
            return _MEMORY_USER_INFO.get(sender_id)
        try:
            data = await redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as exc:
            logger.warning("⚠️ Redis read failed; using in-memory contact info (%s)", exc)
        return _MEMORY_USER_INFO.get(sender_id)
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format the response for display."""
        if not result.get("success", False):
            return result.get("data", "An error occurred.")
        
        data = result.get("data", "")
        if isinstance(data, str):
            return data
        elif isinstance(data, dict) and data.get("message"):
            return data["message"]
        return str(data)
