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


class ContactFormTool(BaseTool):
    """
    Tool for collecting user contact information conversationally.
    
    Flow:
    1. User triggers collection -> Bot asks for name
    2. User responds -> Bot asks for phone
    3. User responds -> Bot asks for address
    4. User responds -> Bot saves and confirms
    
    Uses sticky_context to maintain state across turns.
    Supports both English and Arabic languages.
    """
    
    # Collection states
    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_COMPLETE = "complete"
    
    # Questions to ask (in order) - with Arabic translations
    QUESTIONS = [
        {
            "id": "name", 
            "text": "What is your **name**?",
            "text_ar": "ما هو **اسمك**؟",
            "emoji": "👤"
        },
        {
            "id": "phone", 
            "text": "What is your **phone number**?",
            "text_ar": "ما هو **رقم هاتفك**؟",
            "emoji": "📞"
        },
        {
            "id": "address", 
            "text": "What is your **address**?",
            "text_ar": "ما هو **عنوانك**؟",
            "emoji": "🏠"
        },
    ]
    
    # Bilingual messages
    MESSAGES = {
        "start_collection": {
            "en": "📋 **Let's collect your contact information!**",
            "ar": "📋 **هيا نجمع معلومات الاتصال الخاصة بك!**"
        },
        "got_it": {
            "en": "✅ Got it!",
            "ar": "✅ تم!"
        },
        "saved_success": {
            "en": "🎉 **Information Saved Successfully!**",
            "ar": "🎉 **تم حفظ المعلومات بنجاح!**"
        },
        "name_label": {
            "en": "Name",
            "ar": "الاسم"
        },
        "phone_label": {
            "en": "Phone",
            "ar": "الهاتف"
        },
        "address_label": {
            "en": "Address",
            "ar": "العنوان"
        },
        "show_card_hint": {
            "en": "Your contact information has been stored. Say **'show my contact card'** anytime to see it!",
            "ar": "تم تخزين معلومات الاتصال الخاصة بك. قل **'أظهر بطاقة الاتصال الخاصة بي'** في أي وقت لرؤيتها!"
        },
        "no_info_found": {
            "en": "📋 No stored information found. Say 'collect my info' to provide your details!",
            "ar": "📋 لا توجد معلومات مخزنة. قل 'اجمع معلوماتي' لتقديم بياناتك!"
        },
        "validation_error": {
            "en": "Please provide a response.",
            "ar": "يرجى تقديم استجابة."
        },
        "name_too_short": {
            "en": "Name must be at least 2 characters.",
            "ar": "يجب أن يتكون الاسم من حرفين على الأقل."
        },
        "phone_invalid": {
            "en": "Phone number must have at least 10 digits.",
            "ar": "يجب أن يحتوي رقم الهاتف على 10 أرقام على الأقل."
        },
        "address_too_short": {
            "en": "Please provide a more complete address.",
            "ar": "يرجى تقديم عنوان أكثر اكتمالاً."
        }
    }
    
    def __init__(self):
        self._redis_client = None
    
    def _get_text(self, key: str, lang: str = "en") -> str:
        """Get text in the specified language."""
        messages = self.MESSAGES.get(key, {})
        return messages.get(lang, messages.get("en", key))
    
    def _get_question_text(self, question: dict, lang: str = "en") -> str:
        """Get question text in the specified language."""
        if lang == "ar":
            return question.get("text_ar", question["text"])
        return question["text"]
    
    async def _get_redis(self):
        """Get Redis client for storage."""
        if self._redis_client is None:
            import redis.asyncio as redis
            from config.config import ConfigManager
            config = ConfigManager()
            self._redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                password=config.redis_password,
                decode_responses=True
            )
        return self._redis_client
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="contact_form",
            description="Collect user's OWN PERSONAL contact details (name, phone number, address) through a conversational form. Use when user says: collect my info, take my info, gather my information, save my details, store my contact info. NOT for querying knowledge bases or finding info about others. Use 'start' to begin collecting personal details. Use 'collect' to process responses. Use 'show' to display the user's stored contact card.",
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
                # English - Starting collection
                "Collect my info",
                "Collect my contact info",
                "Take my information",
                "I want to provide my details",
                "I want to give you my contact details",
                "Save my personal information",
                "Store my contact details",
                "Register my details",
                "Add my contact info",
                "Enter my personal details",
                "Fill in my contact form",
                "Start contact form",
                "Begin collecting my info",
                "Gather my information",
                "Gather my details",
                "Record my contact details",
                # Arabic-translated variations (common translations from Arabic)
                "Collect my personal information",
                "Save my personal data",
                "Take my personal details",
                "I want to register my information",
                "Register my personal data",
                "Store my data",
                "Keep my contact information",
                "Enter my data",
                # English - Showing contact card
                "Show my contact card",
                "View my contact card",
                "Display my contact information",
                "What contact info do you have about me?",
                "Show my saved contact details",
                "Show my details",
                "View my saved info",
                "Display my personal card",
                "See my contact information",
                "My contact card",
                "Get my contact details"
            ],
            input_examples=[
                {"action": "start"},
                {"action": "collect", "user_response": "John Doe"},
                {"action": "show"}
            ],
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
        
        # Detect language - PRIORITY ORDER:
        # 1. From collected_data.lang (preserved across collection steps)
        # 2. From original_language in context (set by sticky context handler)
        # 3. Default to "en"
        lang = "en"
        
        # First check collected_data (for ongoing collection)
        collected_data = kwargs.get("collected_data", {})
        if collected_data and collected_data.get("lang"):
            lang = collected_data.get("lang")
            logger.debug(f"📌 Using language from collected_data: {lang}")
        else:
            # Fall back to original_language from context
            original_language = kwargs.get("original_language", "")
            if original_language == "ar":
                lang = "ar"
                logger.debug(f"📌 Using original_language: {lang}")
        
        try:
            if action == "start":
                return await self._start_collection(sender_id, lang)
            
            elif action == "collect":
                # Language already determined above from collected_data or context
                return await self._collect_response(
                    sender_id,
                    user_response,
                    kwargs.get("current_step"),
                    collected_data,
                    lang
                )
            
            elif action == "show":
                return await self._show_info(sender_id, lang)
            
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
    
    async def _start_collection(self, sender_id: str, lang: str = "en") -> Dict[str, Any]:
        """Start collecting user information."""
        logger.info(f"📋 Starting contact info collection for {sender_id} (lang={lang})")
        
        # Initialize collection data
        collected_data = {
            "sender_id": sender_id,
            "started_at": time.time(),
            "responses": {},
            "lang": lang  # Store language for subsequent questions
        }
        
        # Ask first question in the appropriate language
        first_q = self.QUESTIONS[0]
        question_text = self._get_question_text(first_q, lang)
        start_msg = self._get_text("start_collection", lang)
        
        return {
            "success": True,
            "data": f"{start_msg}\n\n{first_q['emoji']} {question_text}",
            "sticky_context": {
                "tool_name": "contact_form",
                "language": lang,  # Store language at top level for sticky context
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
        collected_data: Dict,
        lang: str = "en"
    ) -> Dict[str, Any]:
        """Collect user's response and ask next question."""
        
        # Load from Redis if not provided
        if not collected_data or not collected_data.get("responses"):
            collected_data = await self._load_session(sender_id)
            if not collected_data:
                collected_data = {
                    "sender_id": sender_id,
                    "started_at": time.time(),
                    "responses": {},
                    "lang": lang
                }
        
        # Get language from collected_data (preserves across steps)
        lang = collected_data.get("lang", lang)
        # Always update collected_data['lang'] to current lang
        collected_data["lang"] = lang
        
        current_step = current_step if current_step is not None else len(collected_data.get("responses", {}))
        
        # Store the response
        if current_step < len(self.QUESTIONS):
            question = self.QUESTIONS[current_step]
            # Validate response
            validation_error = self._validate_response(question["id"], user_response, lang)
            if validation_error:
                # Always update collected_data['lang']
                collected_data["lang"] = lang
                # Ask same question again with error (in appropriate language)
                question_text = self._get_question_text(question, lang)
                return {
                    "success": True,
                    "data": f"❌ {validation_error}\n\n{question['emoji']} {question_text}",
                    "sticky_context": {
                        "tool_name": "contact_form",
                        "language": lang,  # Preserve language across validation retries
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
        # Always update collected_data['lang']
        collected_data["lang"] = lang
        # Check if collection is complete
        if next_step >= len(self.QUESTIONS):
            return await self._complete_collection(sender_id, collected_data, lang)
        # Save session
        await self._save_session(sender_id, collected_data)
        # Ask next question in appropriate language
        next_q = self.QUESTIONS[next_step]
        question_text = self._get_question_text(next_q, lang)
        got_it_msg = self._get_text("got_it", lang)
        return {
            "success": True,
            "data": f"{got_it_msg}\n\n{next_q['emoji']} {question_text}",
            "sticky_context": {
                "tool_name": "contact_form",
                "language": lang,  # Preserve language across collection steps
                "state": {
                    "action": "collect",
                    "current_step": next_step,
                    "collected_data": collected_data
                }
            }
        }
    
    def _validate_response(self, field_id: str, response: str, lang: str = "en") -> Optional[str]:
        """Validate a response. Returns error message or None if valid."""
        response = response.strip() if response else ""
        
        if not response:
            return self._get_text("validation_error", lang)
        
        if field_id == "name":
            if len(response) < 2:
                return self._get_text("name_too_short", lang)
        
        elif field_id == "phone":
            digits = re.sub(r'\D', '', response)
            if len(digits) < 10:
                return self._get_text("phone_invalid", lang)
        
        elif field_id == "address":
            if len(response) < 5:
                return self._get_text("address_too_short", lang)
        
        return None
    
    async def _complete_collection(self, sender_id: str, collected_data: Dict, lang: str = "en") -> Dict[str, Any]:
        """Complete the collection and save to permanent storage."""
        collected_data["completed_at"] = time.time()
        
        # Save permanently
        await self._save_user_info(sender_id, collected_data["responses"])
        
        # Clear session
        await self._clear_session(sender_id)
        
        # Build confirmation in appropriate language
        responses = collected_data.get("responses", {})
        saved_msg = self._get_text("saved_success", lang)
        name_label = self._get_text("name_label", lang)
        phone_label = self._get_text("phone_label", lang)
        address_label = self._get_text("address_label", lang)
        hint_msg = self._get_text("show_card_hint", lang)
        
        return {
            "success": True,
            "data": f"{saved_msg}\n\n"
                   f"👤 **{name_label}:** {responses.get('name', 'N/A')}\n"
                   f"📞 **{phone_label}:** {responses.get('phone', 'N/A')}\n"
                   f"🏠 **{address_label}:** {responses.get('address', 'N/A')}\n\n"
                   f"✅ {hint_msg}",
            "sticky_context": None  # Clear sticky context
        }
    
    async def _show_info(self, sender_id: str, lang: str = "en") -> Dict[str, Any]:
        """Show stored user information as a simple Adaptive Card."""
        user_info = await self._load_user_info(sender_id)
        
        if not user_info:
            return {
                "success": True,
                "data": self._get_text("no_info_found", lang)
            }
        
        # Get labels in appropriate language
        phone_label = self._get_text("phone_label", lang)
        address_label = self._get_text("address_label", lang)
        
        # Button text based on language
        update_btn = "✏️ تحديث المعلومات" if lang == "ar" else "✏️ Update Information"
        not_provided = "غير متوفر" if lang == "ar" else "Not provided"
        
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
                            "title": f"📞 {phone_label}",
                            "value": user_info.get("phone", not_provided)
                        },
                        {
                            "title": f"🏠 {address_label}",
                            "value": user_info.get("address", not_provided)
                        }
                    ]
                }
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": update_btn,
                    "data": {
                        "action": "start_contact_collection",
                        "summary": "collect my info"
                    }
                }
            ]
        }
        
        # Message in appropriate language
        if lang == "ar":
            msg = f"📇 إليك معلومات الاتصال الخاصة بك، {user_info.get('name', 'المستخدم')}!"
        else:
            msg = f"📇 Here's your contact information, {user_info.get('name', 'user')}!"
        
        return {
            "success": True,
            "data": {
                "type": "adaptive_card",
                "card": card,
                "message": msg,
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
        await redis_client.set(key, json.dumps(data), ex=3600)  # 1 hour TTL
    
    async def _load_session(self, sender_id: str) -> Optional[Dict]:
        """Load active collection session from Redis."""
        redis_client = await self._get_redis()
        key = f"contact_session:{sender_id}"
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def _clear_session(self, sender_id: str) -> None:
        """Clear active collection session."""
        redis_client = await self._get_redis()
        key = f"contact_session:{sender_id}"
        await redis_client.delete(key)
    
    async def _save_user_info(self, sender_id: str, info: Dict) -> None:
        """Save user info permanently to Redis."""
        redis_client = await self._get_redis()
        key = f"contact_info:{sender_id}"
        info["updated_at"] = time.time()
        await redis_client.set(key, json.dumps(info), ex=86400 * 365)  # 1 year TTL
        logger.info(f"✅ Saved contact info for {sender_id}")
    
    async def _load_user_info(self, sender_id: str) -> Optional[Dict]:
        """Load user info from Redis."""
        redis_client = await self._get_redis()
        key = f"contact_info:{sender_id}"
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
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
