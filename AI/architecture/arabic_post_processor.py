"""
Arabic Post-Processor

Centralizes all Arabic translation for responses:
- Plain text
- Multi-message (---MESSAGE_SPLIT---)  
- Adaptive cards (TextBlock, Actions, labels)
- Related questions

Preserves message splitting by translating each bubble independently.
"""

import logging
import re
import json
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Common English→Arabic translations for UI labels
LABEL_TRANSLATIONS = {
    # Card actions
    "submit": "إرسال",
    "cancel": "إلغاء",
    "close": "إغلاق",
    "open": "فتح",
    "view": "عرض",
    "view details": "عرض التفاصيل",
    "learn more": "اعرف المزيد",
    "read more": "اقرأ المزيد",
    "confirm": "تأكيد",
    "next": "التالي",
    "back": "رجوع",
    "save": "حفظ",
    "delete": "حذف",
    "edit": "تعديل",
    "update": "تحديث",
    "yes": "نعم",
    "no": "لا",
    "ok": "حسناً",
    
    # Form labels
    "name": "الاسم",
    "email": "البريد الإلكتروني",
    "phone": "الهاتف",
    "address": "العنوان",
    "message": "الرسالة",
    "date": "التاريخ",
    "time": "الوقت",
    "notes": "ملاحظات",
    
    # Contact info
    "📞 phone": "📞 الهاتف",
    "🏠 address": "🏠 العنوان",
    "📧 email": "📧 البريد",
    "👤 name": "👤 الاسم",
    "📱 mobile": "📱 الجوال",
    
    # Leave calculator
    "available leave": "الإجازات المتاحة",
    "leave balance": "رصيد الإجازات",
    "request leave": "طلب إجازة",
    "approve": "موافقة",
    "reject": "رفض",
    "pending": "قيد الانتظار",
    "approved": "موافق عليه",
    "rejected": "مرفوض",
    "annual leave": "إجازة سنوية",
    "sick leave": "إجازة مرضية",
    "days": "أيام",
    "start date": "تاريخ البدء",
    "end date": "تاريخ الانتهاء",
    
    # Common
    "your information": "معلوماتك",
    "contact information": "معلومات الاتصال",
    "personal information": "المعلومات الشخصية",
    "related questions": "أسئلة ذات صلة",
    "here are some related questions": "إليك بعض الأسئلة ذات الصلة",
}


class ArabicPostProcessor:
    """
    Post-processor for Arabic responses.
    
    Usage:
        processor = ArabicPostProcessor(translation_service)
        result = processor.process_response(response_dict, detected_language="ar")
    """
    
    def __init__(self, translation_service=None):
        """
        Initialize with translation service.
        
        Args:
            translation_service: LLMTranslationService instance (optional, will get from ServiceManager)
        """
        self._translation_service = translation_service
        self._label_cache = LABEL_TRANSLATIONS.copy()
    
    def _get_translator(self):
        """Lazy-load translation service."""
        if self._translation_service is None:
            from shared_utils import get_service_manager
            self._translation_service = get_service_manager().get_translation_service()
        return self._translation_service
    
    def process_response(
        self,
        response: Union[str, Dict[str, Any]],
        detected_language: str = "en"
    ) -> Union[str, Dict[str, Any]]:
        """
        Process a response and translate to Arabic if needed.
        
        Args:
            response: String or dict response from tool/LLM
            detected_language: Detected input language ("ar" or "en")
            
        Returns:
            Translated response (same type as input)
        """
        if detected_language != "ar":
            return response
        
        if isinstance(response, str):
            return self._process_text_response(response)
        elif isinstance(response, dict):
            return self._process_dict_response(response)
        
        return response
    
    def _process_text_response(self, text: str) -> str:
        """
        Process plain text or multi-bubble text.
        
        Preserves ---MESSAGE_SPLIT--- by:
        1. Splitting into bubbles
        2. Translating each independently
        3. Rejoining with same delimiter
        """
        if not text:
            return text
        
        # Check for message split delimiter
        if "---MESSAGE_SPLIT---" in text:
            parts = text.split("---MESSAGE_SPLIT---")
            translated_parts = []
            
            for part in parts:
                part = part.strip()
                if part:
                    translated_parts.append(self._translate_text(part))
            
            return "---MESSAGE_SPLIT---".join(translated_parts)
        
        # Single message
        return self._translate_text(text)
    
    def _translate_text(self, text: str) -> str:
        """Translate text to Arabic."""
        if not text or not text.strip():
            return text
        
        try:
            translator = self._get_translator()
            return translator.translate_english_to_arabic(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    
    def _process_dict_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process dictionary response.
        
        Handles:
        - response.response (text)
        - response.result.card (Adaptive Card)
        - response.related_questions
        - response.data (various formats)
        """
        result = response.copy()
        
        # Translate main response text
        if "response" in result and isinstance(result["response"], str):
            result["response"] = self._process_text_response(result["response"])
        
        # Translate data if it's text
        if "data" in result:
            if isinstance(result["data"], str):
                result["data"] = self._process_text_response(result["data"])
            elif isinstance(result["data"], dict):
                result["data"] = self._translate_data_dict(result["data"])
        
        # Translate adaptive card
        if "result" in result and isinstance(result["result"], dict):
            if "card" in result["result"]:
                result["result"]["card"] = self._translate_adaptive_card(result["result"]["card"])
            if "type" in result["result"] and result["result"].get("type") == "adaptive_card":
                if "card" in result["result"]:
                    result["result"]["card"] = self._translate_adaptive_card(result["result"]["card"])
        
        # Translate related questions
        if "related_questions" in result:
            result["related_questions"] = self._translate_related_questions(result["related_questions"])
        
        return result
    
    def _translate_data_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate data dictionary fields."""
        result = data.copy()
        
        # Translate message field
        if "message" in result and isinstance(result["message"], str):
            result["message"] = self._translate_text(result["message"])
        
        # Translate text field
        if "text" in result and isinstance(result["text"], str):
            result["text"] = self._translate_text(result["text"])
        
        # Translate card if present
        if "card" in result and isinstance(result["card"], dict):
            result["card"] = self._translate_adaptive_card(result["card"])
        
        return result
    
    def _translate_adaptive_card(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Adaptive Card content.
        
        Translates:
        - TextBlock.text
        - Action.title
        - Input.placeholder
        - FactSet facts
        """
        if not card or not isinstance(card, dict):
            return card
        
        result = card.copy()
        
        # Translate body elements
        if "body" in result:
            result["body"] = self._translate_card_elements(result["body"])
        
        # Translate actions
        if "actions" in result:
            result["actions"] = self._translate_card_actions(result["actions"])
        
        return result
    
    def _translate_card_elements(self, elements: List[Any]) -> List[Any]:
        """Translate card body elements recursively."""
        if not elements:
            return elements
        
        translated = []
        for elem in elements:
            if not isinstance(elem, dict):
                translated.append(elem)
                continue
            
            elem = elem.copy()
            elem_type = elem.get("type", "")
            
            # TextBlock
            if elem_type == "TextBlock" and "text" in elem:
                elem["text"] = self._translate_label_or_text(elem["text"])
            
            # Input.Text
            elif elem_type == "Input.Text":
                if "placeholder" in elem:
                    elem["placeholder"] = self._translate_label_or_text(elem["placeholder"])
                if "label" in elem:
                    elem["label"] = self._translate_label_or_text(elem["label"])
            
            # FactSet
            elif elem_type == "FactSet" and "facts" in elem:
                elem["facts"] = [
                    {
                        "title": self._translate_label_or_text(f.get("title", "")),
                        "value": f.get("value", "")  # Keep values as-is (user data)
                    }
                    for f in elem.get("facts", [])
                ]
            
            # Container/ColumnSet - recurse
            elif elem_type in ("Container", "ColumnSet"):
                if "items" in elem:
                    elem["items"] = self._translate_card_elements(elem["items"])
                if "columns" in elem:
                    for col in elem.get("columns", []):
                        if "items" in col:
                            col["items"] = self._translate_card_elements(col["items"])
            
            # Column - recurse
            elif elem_type == "Column" and "items" in elem:
                elem["items"] = self._translate_card_elements(elem["items"])
            
            translated.append(elem)
        
        return translated
    
    def _translate_card_actions(self, actions: List[Any]) -> List[Any]:
        """Translate card action buttons."""
        if not actions:
            return actions
        
        translated = []
        for action in actions:
            if not isinstance(action, dict):
                translated.append(action)
                continue
            
            action = action.copy()
            
            if "title" in action:
                action["title"] = self._translate_label_or_text(action["title"])
            
            translated.append(action)
        
        return translated
    
    def _translate_label_or_text(self, text: str) -> str:
        """
        Translate text, using cached labels for common terms.
        Falls back to LLM translation for unknown text.
        """
        if not text:
            return text
        
        text_lower = text.lower().strip()
        
        # Check cache first (fast path)
        if text_lower in self._label_cache:
            return self._label_cache[text_lower]
        
        # Check if it's a simple known label with emoji prefix
        for en, ar in self._label_cache.items():
            if text_lower.startswith(en) or text_lower.endswith(en):
                return text.lower().replace(en, ar)
        
        # Fall back to LLM translation
        return self._translate_text(text)
    
    def _translate_related_questions(self, questions: List[Any]) -> List[Any]:
        """Translate related questions list."""
        if not questions:
            return questions
        
        translated = []
        for q in questions:
            if isinstance(q, str):
                translated.append(self._translate_text(q))
            elif isinstance(q, dict):
                q = q.copy()
                if "title" in q:
                    q["title"] = self._translate_text(q["title"])
                if "text" in q:
                    q["text"] = self._translate_text(q["text"])
                if "prompt" in q:
                    q["prompt"] = self._translate_text(q["prompt"])
                translated.append(q)
            else:
                translated.append(q)
        
        return translated


# Singleton accessor
_processor_instance: Optional[ArabicPostProcessor] = None

def get_arabic_post_processor() -> ArabicPostProcessor:
    """Get the Arabic post-processor singleton."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ArabicPostProcessor()
    return _processor_instance
