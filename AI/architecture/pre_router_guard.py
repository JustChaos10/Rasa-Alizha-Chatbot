"""
Pre-Router Guard

Language-agnostic dangerous content detection.
Runs BEFORE translation/routing to block harmful requests early.
Returns refusal in the input language (English or Arabic).
"""

import sys
import re
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

# Fix Windows console encoding for emoji/unicode
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result of guard check."""
    allowed: bool
    reason: str
    refusal_message: str  # In original language
    category: str = ""  # e.g., "violence", "weapons", "illegal"


# Dangerous patterns (language-agnostic keywords)
DANGEROUS_PATTERNS = {
    "weapons": [
        r"\b(make|build|create|construct|assemble)\b.*\b(bomb|explosive|weapon|gun|firearm)\b",
        r"\b(bomb|explosive|weapon)\b.*\b(make|build|create|how to)\b",
        r"\bhow to make.*\b(bomb|weapon|explosive|poison|drug)\b",
        r"\brecipe for.*\b(bomb|explosive|meth|cocaine)\b",
        r"\binstructions for.*\b(bomb|weapon|poison)\b",
    ],
    "violence": [
        r"\bhow to (kill|murder|harm|hurt|attack)\b",
        r"\b(kill|murder|assassinate)\b.*\b(person|someone|people)\b",
        r"\bhurt (someone|people|person)\b",
    ],
    "illegal": [
        r"\bhow to hack\b.*\b(account|password|bank|system)\b",
        r"\bsteal (money|identity|credit card)\b",
        r"\bhow to (launder|counterfeit)\b",
        r"\bhow to (break into|burglary)\b",
    ],
    "self_harm": [
        r"\bhow to (commit suicide|kill myself|end my life)\b",
        r"\bways to (die|hurt myself)\b",
    ],
}

# Arabic patterns (common dangerous queries in Arabic)
ARABIC_DANGEROUS_PATTERNS = {
    "weapons": [
        r"كيف.*صنع.*(قنبلة|متفجرات|سلاح)",
        r"طريقة.*صنع.*(قنبلة|متفجرات)",
        r"صناعة.*(قنبلة|سلاح|متفجرات)",
    ],
    "violence": [
        r"كيف.*(أقتل|اقتل|نقتل|قتل)",
        r"طريقة.*قتل",
    ],
}

# Unicode Arabic patterns (preferred). These are kept separate so we don't
# depend on any mojibake/encoding artifacts in older pattern sets.
ARABIC_DANGEROUS_PATTERNS_UNICODE = {
    "weapons": [
        r"(?:كيف(?:ية)?|طريقة)\s+(?:صنع|عمل|تحضير|تجهيز)\s+(?:قنبلة|متفجرات|عبوة\s+ناسفة)\b",
        r"\b(?:اصنع|أصنع|صنع)\s+(?:قنبلة|متفجرات|عبوة\s+ناسفة)\b",
        r"\b(?:مكونات|مواد)\s+(?:قنبلة|متفجرات|عبوة\s+ناسفة)\b",
    ],
    "violence": [
        r"\bكيف\s+(?:أقتل|اقتل|أؤذي|اؤذي|أضر|اضر|أهاجم|اهاجم)\b",
        r"(?:كيف|طريقة)\s+(?:قتل|إيذاء|اذية|إضرار|اضرار)\s+(?:شخص|إنسان|أحد|احد)\b",
    ],
    "illegal": [
        r"\bكيف\s+(?:أخترق|اخترق|أهكر|اهكر)\b",
        r"(?:طريقة|كيف)\s+(?:اختراق|تهكير)\s+(?:حساب|بريد|نظام|شبكة)\b",
    ],
}

# Refusal messages
REFUSAL_EN = """I can't help with that request. This type of content could cause harm.

If you're experiencing difficult thoughts, please reach out to:
• Emergency: 911
• Crisis helpline: 988 (US)

Is there something else I can help you with?"""

REFUSAL_AR = """لا أستطيع المساعدة في هذا الطلب. هذا النوع من المحتوى قد يسبب ضررًا.

إذا كنت تمر بأفكار صعبة، يرجى التواصل مع:
• الطوارئ: 911
• خط المساعدة النفسية

هل يمكنني مساعدتك في شيء آخر؟"""


REFUSAL_AR_UNICODE = """لا أستطيع المساعدة في هذا الطلب لأنه قد يسبب ضررًا.

إذا كنت في خطر أو تمرّ بأزمة، تواصل مع خدمات الطوارئ المحلية أو خط المساعدة في بلدك.

هل يمكنني مساعدتك في شيء آخر؟"""


class PreRouterGuard:
    """
    Pre-routing guard for dangerous content.
    
    Usage:
        guard = PreRouterGuard()
        result = guard.check(message, detected_language)
        if not result.allowed:
            return result.refusal_message
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize guard.
        
        Args:
            strict_mode: If True, be more aggressive in blocking
        """
        self.strict_mode = strict_mode
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching."""
        self._compiled_en = {}
        for category, patterns in DANGEROUS_PATTERNS.items():
            self._compiled_en[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._compiled_ar = {}
        for category, patterns in ARABIC_DANGEROUS_PATTERNS.items():
            self._compiled_ar[category] = [
                re.compile(p, re.UNICODE) for p in patterns
            ]

        # Preferred Unicode Arabic patterns
        for category, patterns in ARABIC_DANGEROUS_PATTERNS_UNICODE.items():
            compiled = [re.compile(p, re.UNICODE) for p in patterns]
            if category in self._compiled_ar:
                self._compiled_ar[category].extend(compiled)
            else:
                self._compiled_ar[category] = compiled
    
    def check(self, message: str, detected_language: str = "en") -> GuardResult:
        """
        Check if message contains dangerous content.
        
        Args:
            message: User's message
            detected_language: Detected language ("en" or "ar")
            
        Returns:
            GuardResult with allowed status and refusal if blocked
        """
        if not message:
            return GuardResult(allowed=True, reason="", refusal_message="")
        
        message_lower = message.lower().strip()
        
        # Check English patterns
        for category, patterns in self._compiled_en.items():
            for pattern in patterns:
                if pattern.search(message_lower):
                    logger.warning(f"🛡️ Guard blocked: category={category}, pattern matched")
                    return GuardResult(
                        allowed=False,
                        reason=f"Dangerous content detected: {category}",
                        refusal_message=REFUSAL_AR_UNICODE if detected_language == "ar" else REFUSAL_EN,
                        category=category
                    )
        
        # Check Arabic patterns (if text contains Arabic)
        if any('\u0600' <= c <= '\u06FF' for c in message):
            for category, patterns in self._compiled_ar.items():
                for pattern in patterns:
                    if pattern.search(message):
                        logger.warning(f"🛡️ Guard blocked (AR): category={category}")
                        return GuardResult(
                            allowed=False,
                            reason=f"Dangerous content detected: {category}",
                            refusal_message=REFUSAL_AR_UNICODE,
                            category=category
                        )
        
        return GuardResult(allowed=True, reason="", refusal_message="")
    
    def is_safe(self, message: str, detected_language: str = "en") -> bool:
        """Quick check if message is safe to process."""
        return self.check(message, detected_language).allowed


# Singleton accessor
_guard_instance: Optional[PreRouterGuard] = None

def get_pre_router_guard() -> PreRouterGuard:
    """Get the pre-router guard singleton."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = PreRouterGuard()
    return _guard_instance
