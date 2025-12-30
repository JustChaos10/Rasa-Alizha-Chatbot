import asyncio
import base64
import io
import json
import logging
import os
import re
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import PyPDF2
from docx import Document

# Import telemetry
from architecture.telemetry import trace_llm_call, log_llm_event
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Import langdetect for language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Language detection will use LLM only.")
 
class Config:
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TIMEOUT = 30
    VISION_TIMEOUT = 60
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_DIMENSION = 2048
    JPEG_QUALITY = 85
    MAX_STORAGE_SIZE = 1000
    UPLOAD_FOLDER = "uploads"
    MIN_PHONE_LENGTH = 10
    MAX_PHONE_LENGTH = 15
    MIN_NAME_LENGTH = 2
    MIN_ADDRESS_LENGTH = 3
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    DEFAULT_TEXT_MODEL = "llama-3.3-70b-versatile"
    AVAILABLE_MODELS = [
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'meta-llama/llama-guard-4-12b',
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
        'moonshotai/kimi-k2-instruct',
        'qwen/qwen3-32b',
    ]
    
    # Gemini Configuration (fallback provider)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
   
    # Translation Configuration
    TRANSLATION_MODEL = "llama-3.3-70b-versatile"
    TRANSLATION_TIMEOUT = 30
    TRANSLATION_MAX_TOKENS = 1000
    TRANSLATION_TEMPERATURE = 0.3
    SUPPORTED_LANGUAGES = ["en", "ar"]
    LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.7

    # Image search / visual enrichment
    IMAGE_SEARCH_ENABLED = os.getenv("ENABLE_IMAGE_SEARCH", "1").strip() not in {"0", "false", "False"}
    IMAGE_SEARCH_PROVIDER = os.getenv("IMAGE_SEARCH_PROVIDER", "google_cse").strip().lower()
    IMAGE_SEARCH_MAX_RESULTS = int(os.getenv("IMAGE_SEARCH_MAX_RESULTS", "3") or 3)
    IMAGE_SEARCH_TIMEOUT = float(os.getenv("IMAGE_SEARCH_TIMEOUT", "4.0") or 4.0)
    IMAGE_SEARCH_CACHE_TTL = int(os.getenv("IMAGE_SEARCH_CACHE_TTL", "900") or 900)

# =============================================================================
# GLOBAL RATE LIMITER - Import from standalone module
# =============================================================================
from rate_limiter import GlobalRateLimiter, get_global_rate_limiter


# =============================================================================
# GLOBAL LLM SERVICE - THE ONLY WAY TO CALL LLMs IN THIS PROJECT
# =============================================================================
# All servers, plugins, routers MUST use this service for LLM calls.
# This ensures:
# 1. Centralized rate limiting (sliding window)
# 2. Automatic provider failover (GROQ → Gemini)
# 3. NO duplicate retry logic in individual servers
# 4. All calls are counted and traced
# =============================================================================

class GlobalLLMService:
    """
    Centralized LLM Service - THE ONLY WAY TO CALL LLMs.
    
    Features:
    - Works both SYNC and ASYNC
    - Automatic failover: GROQ → Gemini
    - Centralized rate limiting (sliding window)
    - NO retry logic - fails fast, tries next provider
    - All calls counted against global rate limit
    - Telemetry integration
    
    Usage (Sync):
        llm = get_global_llm_service()
        response = llm.call("Your prompt here")
        
    Usage (Async):
        llm = get_global_llm_service()
        response = await llm.call_async("Your prompt here")
        
    Usage with messages:
        response = llm.call_with_messages([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ])
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def _ensure_initialized(self):
        """Lazy initialization."""
        if self._initialized:
            return
            
        self._rate_limiter = get_global_rate_limiter()
        
        # Provider configuration
        self._groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self._gemini_api_key = (
            os.getenv("GEMINI_API_KEY") or 
            os.getenv("GOOGLE_API_KEY") or 
            os.getenv("GOOGLE_GEMINI_API_KEY") or ""
        ).strip()
        
        # Models
        self._groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # API URLs
        self._groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self._gemini_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # Track provider health (reset after 3 consecutive failures)
        self._groq_failures = 0
        self._gemini_failures = 0
        self._max_failures = 3
        
        # Default settings
        self._default_temperature = 0.7
        self._default_max_tokens = 1000
        self._default_timeout = 30.0
        
        self._initialized = True
        
        # Log initialization
        groq_status = "✅" if self._groq_api_key else "❌"
        gemini_status = "✅" if self._gemini_api_key else "❌"
        logger.info(f"🤖 GlobalLLMService initialized: GROQ {groq_status}, Gemini {gemini_status}")
    
    def _get_provider_order(self) -> list:
        """Get providers in order of preference, skipping ones that are failing."""
        self._ensure_initialized()
        providers = []
        
        # GROQ first (primary) if available and not failing too much
        if self._groq_api_key and self._groq_failures < self._max_failures:
            providers.append("groq")
        
        # Gemini as fallback
        if self._gemini_api_key and self._gemini_failures < self._max_failures:
            providers.append("gemini")
        
        # If all failing, reset and try again
        if not providers:
            logger.warning("🔄 All LLM providers failing, resetting failure counts...")
            self._groq_failures = 0
            self._gemini_failures = 0
            if self._groq_api_key:
                providers.append("groq")
            if self._gemini_api_key:
                providers.append("gemini")
        
        return providers
    
    def _call_groq_sync(
        self, 
        messages: list, 
        max_tokens: int, 
        temperature: float,
        timeout: float,
        json_mode: bool = False
    ) -> str:
        """Synchronous GROQ API call."""
        payload = {
            "model": self._groq_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        response = requests.post(
            self._groq_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._groq_api_key}",
                "Content-Type": "application/json"
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        self._groq_failures = max(0, self._groq_failures - 1)  # Reduce on success
        return data["choices"][0]["message"]["content"].strip()
    
    async def _call_groq_async(
        self, 
        messages: list, 
        max_tokens: int, 
        temperature: float,
        timeout: float,
        json_mode: bool = False
    ) -> str:
        """Asynchronous GROQ API call."""
        import httpx
        
        payload = {
            "model": self._groq_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self._groq_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._groq_api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            data = response.json()
            self._groq_failures = max(0, self._groq_failures - 1)
            return data["choices"][0]["message"]["content"].strip()
    
    def _call_gemini_sync(
        self, 
        messages: list, 
        max_tokens: int, 
        temperature: float,
        timeout: float
    ) -> str:
        """Synchronous Gemini API call."""
        # Convert messages to single prompt
        prompt = self._messages_to_prompt(messages)
        
        url = f"{self._gemini_url_template.format(model=self._gemini_model)}?key={self._gemini_api_key}"
        
        # Gemini 2.5 uses "thinking" tokens internally, add buffer
        effective_max_tokens = max(max_tokens + 100, 200)
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": effective_max_tokens,
            }
        }
        
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        self._gemini_failures = max(0, self._gemini_failures - 1)
        return self._extract_gemini_text(data)
    
    async def _call_gemini_async(
        self, 
        messages: list, 
        max_tokens: int, 
        temperature: float,
        timeout: float
    ) -> str:
        """Asynchronous Gemini API call."""
        import httpx
        
        prompt = self._messages_to_prompt(messages)
        url = f"{self._gemini_url_template.format(model=self._gemini_model)}?key={self._gemini_api_key}"
        
        effective_max_tokens = max(max_tokens + 100, 200)
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": effective_max_tokens,
            }
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            self._gemini_failures = max(0, self._gemini_failures - 1)
            return self._extract_gemini_text(data)
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to single prompt for Gemini."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"Instructions: {content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)
    
    def _extract_gemini_text(self, data: dict) -> str:
        """Extract text from Gemini response, handling various formats."""
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            finish_reason = candidates[0].get("finishReason", "UNKNOWN")
            if finish_reason == "MAX_TOKENS":
                raise ValueError("Gemini hit MAX_TOKENS before generating content")
            raise ValueError(f"No parts in Gemini response (finish_reason: {finish_reason})")
        
        return parts[0].get("text", "").strip()
    
    # =========================================================================
    # PUBLIC API - Use these methods for all LLM calls
    # =========================================================================
    
    def call(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        timeout: float = None,
        json_mode: bool = False,
        trace_name: str = "global-llm"
    ) -> str:
        """
        SYNC: Call LLM with automatic failover.
        
        Args:
            prompt: Simple text prompt
            max_tokens: Max response tokens (default: 1000)
            temperature: Sampling temperature (default: 0.7)
            timeout: Request timeout seconds (default: 30)
            json_mode: Request JSON response format (GROQ only)
            trace_name: Name for telemetry trace
            
        Returns:
            LLM response text
            
        Raises:
            Exception if all providers fail
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call_with_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            json_mode=json_mode,
            trace_name=trace_name
        )
    
    def call_with_messages(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
        timeout: float = None,
        json_mode: bool = False,
        trace_name: str = "global-llm"
    ) -> str:
        """
        SYNC: Call LLM with message list and automatic failover.
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Max response tokens
            temperature: Sampling temperature
            timeout: Request timeout
            json_mode: Request JSON format
            trace_name: Telemetry trace name
            
        Returns:
            LLM response text
        """
        self._ensure_initialized()
        
        max_tokens = max_tokens or self._default_max_tokens
        temperature = temperature if temperature is not None else self._default_temperature
        timeout = timeout or self._default_timeout
        
        providers = self._get_provider_order()
        if not providers:
            raise ValueError("No LLM API keys configured")
        
        last_error = None
        
        for provider in providers:
            # Rate limit check BEFORE each attempt
            self._rate_limiter.wait_if_needed()
            
            logger.info(f"🤖 [{trace_name}] Calling {provider.upper()}...")
            
            try:
                with trace_llm_call(
                    name=trace_name,
                    model=f"{provider}/{self._groq_model if provider == 'groq' else self._gemini_model}",
                    input_data={"messages": messages},
                    model_parameters={"temperature": temperature, "max_tokens": max_tokens},
                    metadata={"source": "GlobalLLMService", "provider": provider}
                ) as trace:
                    if provider == "groq":
                        result = self._call_groq_sync(messages, max_tokens, temperature, timeout, json_mode)
                    else:
                        result = self._call_gemini_sync(messages, max_tokens, temperature, timeout)
                    
                    # Record successful call
                    self._rate_limiter.record_call()
                    
                    logger.info(f"✅ [{trace_name}] {provider.upper()} succeeded")
                    trace.update(output=result[:500], metadata={"success": True, "provider": provider})
                    return result
                    
            except requests.HTTPError as e:
                self._rate_limiter.record_call()  # Count failed attempts too
                
                if e.response.status_code == 429:
                    logger.warning(f"⚠️ [{trace_name}] {provider.upper()} rate limited (429)")
                    if provider == "groq":
                        self._groq_failures += 1
                    else:
                        self._gemini_failures += 1
                    last_error = e
                    continue
                raise
                
            except Exception as e:
                self._rate_limiter.record_call()
                logger.warning(f"⚠️ [{trace_name}] {provider.upper()} failed: {e}")
                if provider == "groq":
                    self._groq_failures += 1
                else:
                    self._gemini_failures += 1
                last_error = e
                continue
        
        raise Exception(f"All LLM providers failed. Last error: {last_error}")
    
    async def call_async(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        timeout: float = None,
        json_mode: bool = False,
        trace_name: str = "global-llm"
    ) -> str:
        """
        ASYNC: Call LLM with automatic failover.
        
        Same as call() but for async contexts.
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.call_with_messages_async(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            json_mode=json_mode,
            trace_name=trace_name
        )
    
    async def call_with_messages_async(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
        timeout: float = None,
        json_mode: bool = False,
        trace_name: str = "global-llm"
    ) -> str:
        """
        ASYNC: Call LLM with message list and automatic failover.
        
        Same as call_with_messages() but for async contexts.
        """
        self._ensure_initialized()
        
        max_tokens = max_tokens or self._default_max_tokens
        temperature = temperature if temperature is not None else self._default_temperature
        timeout = timeout or self._default_timeout
        
        providers = self._get_provider_order()
        if not providers:
            raise ValueError("No LLM API keys configured")
        
        last_error = None
        
        for provider in providers:
            # Rate limit check BEFORE each attempt
            await self._rate_limiter.wait_for_slot_async()
            
            logger.info(f"🤖 [{trace_name}] Calling {provider.upper()}...")
            
            try:
                with trace_llm_call(
                    name=trace_name,
                    model=f"{provider}/{self._groq_model if provider == 'groq' else self._gemini_model}",
                    input_data={"messages": messages},
                    model_parameters={"temperature": temperature, "max_tokens": max_tokens},
                    metadata={"source": "GlobalLLMService", "provider": provider}
                ) as trace:
                    if provider == "groq":
                        result = await self._call_groq_async(messages, max_tokens, temperature, timeout, json_mode)
                    else:
                        result = await self._call_gemini_async(messages, max_tokens, temperature, timeout)
                    
                    # Record successful call
                    self._rate_limiter.record_call()
                    
                    logger.info(f"✅ [{trace_name}] {provider.upper()} succeeded")
                    trace.update(output=result[:500], metadata={"success": True, "provider": provider})
                    return result
                    
            except Exception as e:
                self._rate_limiter.record_call()
                
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str
                
                if is_rate_limit:
                    logger.warning(f"⚠️ [{trace_name}] {provider.upper()} rate limited")
                else:
                    logger.warning(f"⚠️ [{trace_name}] {provider.upper()} failed: {e}")
                
                if provider == "groq":
                    self._groq_failures += 1
                else:
                    self._gemini_failures += 1
                last_error = e
                continue
        
        raise Exception(f"All LLM providers failed. Last error: {last_error}")
    
    def get_status(self) -> dict:
        """Get current status of all providers."""
        self._ensure_initialized()
        return {
            "groq": {
                "available": bool(self._groq_api_key),
                "model": self._groq_model,
                "failures": self._groq_failures
            },
            "gemini": {
                "available": bool(self._gemini_api_key),
                "model": self._gemini_model,
                "failures": self._gemini_failures
            },
            "rate_limiter": self._rate_limiter.get_stats()
        }


# Global singleton instance
_global_llm_service = None


def get_global_llm_service() -> GlobalLLMService:
    """
    Get the global LLM service singleton.
    
    USE THIS FOR ALL LLM CALLS IN THE PROJECT.
    """
    global _global_llm_service
    if _global_llm_service is None:
        _global_llm_service = GlobalLLMService()
    return _global_llm_service


# Secure RAG JSONL logger helper
def _sr_log(entry: Dict[str, Any]) -> None:
    try:
        path = Path("secure_rag.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
 
class ServiceManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._storage = []
        return cls._instance
    def get_llm_service(self):
        if 'llm' not in self._services:
            self._services['llm'] = LLMService()
        return self._services['llm']
    def get_storage_service(self):
        if 'storage' not in self._services:
            self._services['storage'] = StorageService(self._storage)
        return self._services['storage']
    def get_file_service(self):
        if 'file' not in self._services:
            self._services['file'] = FileService()
        return self._services['file']
    def get_validation_service(self):
        if 'validation' not in self._services:
            self._services['validation'] = ValidationService()
        return self._services['validation']
    def get_translation_service(self):
        if 'translation' not in self._services:
            self._services['translation'] = LLMTranslationService()
        return self._services['translation']
    def get_image_service(self):
        if 'image' not in self._services:
            self._services['image'] = ImageSearchService()
        return self._services['image']
 
def get_service_manager():
    return ServiceManager()
 
class LLMService:
    """
    Legacy LLM Service - Now delegates to GlobalLLMService.
    
    This class is kept for backward compatibility. All calls are
    routed through the centralized GlobalLLMService.
    """
    def __init__(self):
        # Use the global LLM service for all calls
        self._global_llm = get_global_llm_service()
        
        # Lazy init for Secure RAG components
        self._input_guard = None
        self._output_guard = None
        self._rbac = None
        
    def _get_api_key(self):
        """Legacy method - kept for compatibility."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or len(api_key.strip()) < 20:
            raise ValueError("GROQ_API_KEY not configured properly")
        return api_key.strip()
    
    def generate_text(self, messages, model=None, trace_name: str = "shared-llm-call", **kwargs):
        """
        Generate text using GlobalLLMService.
        
        All calls are routed through the centralized service which handles:
        - Rate limiting
        - Provider failover (GROQ → Gemini)
        - Telemetry
        """
        # Delegate to GlobalLLMService
        json_mode = kwargs.get('response_format', {}).get('type') == 'json_object' if kwargs.get('response_format') else False
        
        return self._global_llm.call_with_messages(
            messages=messages,
            max_tokens=kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
            temperature=kwargs.get('temperature', Config.DEFAULT_TEMPERATURE),
            timeout=kwargs.get('timeout', Config.DEFAULT_TIMEOUT),
            json_mode=json_mode,
            trace_name=trace_name
        )
 
    # --- Secure RAG integration wrappers ---
    def _init_secure_services(self):
        if self._input_guard is None or self._output_guard is None or self._rbac is None:
            try:
                from mcp_servers.secure_rag_system.secure_rag import InputGuard, OutputGuard, RBACService, AccessLevel
                self._InputGuardCls = InputGuard
                self._OutputGuardCls = OutputGuard
                self._RBACServiceCls = RBACService
                self._AccessLevelCls = AccessLevel
                # Create instances
                self._input_guard = self._InputGuardCls()
                self._output_guard = self._OutputGuardCls()
                self._rbac = self._RBACServiceCls()
 
                # Seed RBAC with a minimal document set if empty, so context building is useful
                try:
                    if hasattr(self._rbac, 'documents') and not self._rbac.documents:
                        sample_docs = [
                            {
                                "owner_ids": ["hr_admin", "hr_executive"],
                                "shared": True,
                                "category": "employee_data",
                                "sensitivity_level": "medium",
                                "created_by": "hr_admin",
                                "department": "Engineering",
                                "text": "Employee Details: Akash works at ITC Infotech as IS2 level engineer in Engineering department"
                            },
                            {
                                "owner_ids": ["hr_admin", "emp2"],
                                "shared": True,
                                "category": "employee_data",
                                "sensitivity_level": "medium",
                                "created_by": "hr_admin",
                                "department": "Data Science",
                                "text": "Employee Details: Arpan works at ITC Infotech as IS1 level Data Scientist in MOC Innovation Team"
                            },
                            {
                                "owner_ids": [],
                                "shared": True,
                                "category": "public",
                                "sensitivity_level": "low",
                                "created_by": "system",
                                "department": "General",
                                "text": "Company Policies: Security guidelines, work hours, and professional conduct standards"
                            },
                            {
                                "owner_ids": ["hr_admin"],
                                "shared": True,
                                "category": "general",
                                "sensitivity_level": "low",
                                "created_by": "hr_admin",
                                "department": "General",
                                "text": "ITC Infotech provides digital transformation and IT services to clients worldwide"
                            }
                        ]
                        try:
                            self._rbac.ingest_documents(sample_docs)
                        except Exception as ie:
                            logger.warning(f"RBAC document seeding failed: {ie}")
                except Exception as seed_err:
                    logger.warning(f"RBAC initialization warning: {seed_err}")
            except Exception as e:
                logger.error(f"Secure RAG init failed: {e}")
                self._input_guard = None
                self._output_guard = None
                self._rbac = None
 
    def _map_role_to_user_id(self, user_role: Optional[str]) -> str:
        role = (user_role or "user").strip().lower()
        return "hr_admin" if role == "admin" else "hr_common"
 
    def _should_include_rbac_context(self, prompt: str) -> bool:
        """Heuristic to include RBAC document context only for document-relevant queries.
        Reduces leakage of internal names into unrelated answers (e.g., jokes).
        """
        text = (prompt or "").lower()
        triggers = [
            "employee", "employees", "hr", "confidential", "policy", "policies",
            "document", "documents", "docs", "company policy", "company policies",
            "details for", "itc infotech", "akash", "arpan"
        ]
        return any(t in text for t in triggers)
 
    def generate_text_secure(
        self,
        user_text: str,
        user_role: Optional[str] = None,
        model: Optional[str] = None,
        context_tail: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Secure RAG wrapper around generate_text with input/output guards and RBAC.
 
        - user_role: "admin" or "user" (defaults to user)
        - model: keep your adapter's chosen model (auto/llama/qwen/gptoss)
        - context_tail: optional recent conversation to include for tone/continuity
        """
        # Initialize Secure RAG services (lazy)
        self._init_secure_services()
 
        # If Secure RAG unavailable, fall back to vanilla generation
        if not self._input_guard or not self._output_guard or not self._rbac:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_text},
            ]
            return self.generate_text(messages, model=model, **kwargs)
 
        # 1) Input guard
        scan = self._input_guard.scan(user_text or "")
        sanitized_prompt = scan.sanitized_prompt if scan else user_text
        sanitized_changed = sanitized_prompt != user_text
 
        # Admin override for false positives
        admin_override = False
        if not scan.is_valid:
            if (user_role or "").strip().lower() == "admin":
                admin_override = True
            else:
                try:
                    _sr_log({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "input_guard",
                        "result": "blocked",
                        "warnings": scan.warnings
                    })
                except Exception:
                    pass
                return "Your input was blocked by security policy. Please rephrase."
 
        if admin_override:
            sanitized_prompt = user_text
            sanitized_changed = False
 
        if (user_role or "").strip().lower() == "admin" and sanitized_changed:
            # Preserve full context for administrators to enable RBAC evaluation.
            sanitized_prompt = user_text
            sanitized_changed = False
        try:
            _sr_log({
                "timestamp": datetime.now().isoformat(),
                "stage": "input_guard",
                "result": "ok",
                "warnings": scan.warnings,
                "sanitized": sanitized_changed
            })
        except Exception:
            pass
 
        # 2) RBAC context (admin gets FULL, user LIMITED)
        user_id_for_rbac = self._map_role_to_user_id(user_role)
        rbac_user = self._rbac.get_user(user_id_for_rbac, db_role=(user_role or "user").strip().lower())
        access = self._rbac.check_access(rbac_user, sanitized_prompt)
        # Build context from accessible docs (only when relevant)
        include_docs = self._should_include_rbac_context(sanitized_prompt)
        rbac_context = ""
        if include_docs and access and access.filtered_documents:
            ctx_parts = []
            for doc in access.filtered_documents:
                # Documents are dicts with 'text' and 'category' keys
                text = doc.get('text', '')
                category = doc.get('category', 'general')
                part = f"Document: {text} (Category: {category})"
                ctx_parts.append(part)
            rbac_context = "\n\n".join(ctx_parts)
 
        # Optionally include conversation tail
        composed_context = rbac_context
        if context_tail and isinstance(context_tail, str) and len(context_tail.strip()) > 0:
            # Only append conversation tail; do not inject default doc text when none
            composed_context = (composed_context + ("\n\nRecent conversation:\n" + context_tail.strip() if composed_context else f"Recent conversation:\n{context_tail.strip()}"))
 
        # 3) Call existing adapter generate_text (preserve your model selection)
        # Add explicit role-based policy instructions to avoid contradictory answers
        AccessLevel = self._AccessLevelCls  # type: ignore[attr-defined]
        level = AccessLevel.FULL if (user_role or "").strip().lower() == "admin" else AccessLevel.LIMITED
        base_system = "You are a helpful AI assistant."
        if level.name == "LIMITED":
            policy = (
                " Do not disclose any sensitive or employee-specific personal details. "
                "If asked for confidential information, politely refuse and offer only general, public information."
            )
        else:
            policy = (
                " You may answer using the provided document context when the question is about policies, employees, or company documents. "
                "Do not claim you are unauthorized when the context allows answering. Do not introduce internal names in unrelated topics."
            )
        if composed_context:
            sys_content = f"{base_system}{policy} Use this context if relevant: {composed_context}"
        else:
            sys_content = f"{base_system}{policy}"
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": sanitized_prompt},
        ]
        raw = self.generate_text(messages, model=model, **kwargs)
        try:
            _sr_log({
                "timestamp": datetime.now().isoformat(),
                "stage": "llm_generation",
                "model": model or Config.DEFAULT_TEXT_MODEL,
                "prompt_len": len(sanitized_prompt or ""),
                "context_len": len(composed_context or "")
            })
        except Exception:
            pass
 
        # 4) Output guard (filters sensitive data like SSN, credit cards, passwords)
        try:
            out = self._output_guard.scan(raw)
            if not out.is_valid:
                try:
                    _sr_log({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "output_guard",
                        "result": "blocked",
                        "warnings": out.warnings
                    })
                except Exception:
                    pass
                return "The response was blocked by security policy. Please try a different question."
            final = out.sanitized_response or raw
            try:
                _sr_log({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "output_guard",
                    "result": "ok",
                    "warnings": out.warnings,
                    "final_len": len(final or "")
                })
            except Exception:
                pass
            return MessageFormatter.clean_markdown_text(final)
        except Exception as e:
            logger.error(f"Output guard failed, returning raw: {e}")
            return MessageFormatter.clean_markdown_text(raw)
    def analyze_vision(self, image_base64, query, **kwargs):
        for model in Config.VISION_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                    "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
                }
                response = requests.post(Config.GROQ_API_URL, json=payload, headers=self._headers, timeout=Config.VISION_TIMEOUT)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception:
                continue
        raise Exception("All vision models failed")
 
class ModelSelector:
    @staticmethod
    def choose_model(user_text: Optional[str], preferred_model: Optional[str], available_models: Optional[List[str]] = None, default_model: Optional[str] = None) -> str:
        # Honor explicit user preference when provided
        if preferred_model and isinstance(preferred_model, str) and preferred_model.strip():
            return preferred_model.strip()
 
        text = (user_text or "").lower()
        available_models = available_models or Config.AVAILABLE_MODELS
        default_model = default_model or Config.DEFAULT_TEXT_MODEL
 
        # Helper to pick best available option among candidates, else default
        def pick(*candidates: str) -> str:
            for cand in candidates:
                if cand in available_models:
                    return cand
            return default_model
 
        reasoning_triggers = [
            "why", "explain", "step by step", "think", "reason", "justify", "proof", "derive",
            "analyze", "compare", "evaluate", "plan", "strategy", "trade-off", "pros and cons",
            "eli5", "explain like i'm", "explain like im"
        ]
        coding_triggers = [
            "code", "python", "javascript", "typescript", "bug", "error", "stack trace", "function", "class", "regex",
            "refactor", "optimize", "snippet"
        ]
        math_triggers = [
            "calculate", "compute", "solve", "equation", "integral", "derivative", "proof"
        ]
        json_triggers = ["json", "return json", "json_object", "valid json", "response_format"]
        creative_triggers = ["essay", "story", "poem", "creative", "blog", "rewrite", "paraphrase"]
        quick_triggers = ["hi", "hello", "hey", "thanks", "thank you", "help", "who are you", "what can you do"]
 
        long_query = len(text.split()) >= 60 or len(text) >= 300
 
        # Heuristics
        if any(k in text for k in reasoning_triggers) or "chain of thought" in text or "cot" in text:
            # Heavy reasoning → prefer Qwen first, then 120B, then 70B
            return pick("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
 
        if any(k in text for k in coding_triggers) or "explain code" in text or "refactor" in text:
            # Coding → prefer Qwen first for code tasks
            return pick("qwen/qwen3-32b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if any(k in text for k in math_triggers):
            return pick("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
 
        if any(k in text for k in json_triggers):
            return pick("openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if any(k in text for k in creative_triggers):
            return pick("moonshotai/kimi-k2-instruct", "llama-3.3-70b-versatile", "openai/gpt-oss-20b")
 
        if any(text.strip().startswith(t) or t == text.strip() for t in quick_triggers):
            return pick("llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if long_query or "summarize" in text or "analyze" in text or "write" in text:
            # Long/summary → prefer 70B or Qwen before 120B to reduce cost/noise
            return pick("llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-120b")
 
        # Default balanced choice
        return pick("llama-3.3-70b-versatile", "openai/gpt-oss-20b", default_model)
 
class StorageService:
    def __init__(self, storage_list):
        self._storage = storage_list
    def store_user_info(self, user_id, data):
        data.update({"user_id": user_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        existing = next((item for item in self._storage if item["user_id"] == user_id), None)
        if existing:
            existing.update(data)
        else:
            self._storage.append(data)
        self._cleanup()
    def get_user_info(self, user_id):
        return next((item for item in self._storage if item["user_id"] == user_id), None)
    def _cleanup(self):
        if len(self._storage) >= Config.MAX_STORAGE_SIZE:
            keep_count = int(Config.MAX_STORAGE_SIZE * 0.8)
            self._storage[:] = self._storage[-keep_count:]
 
class ValidationService:
    def validate_name(self, name):
        return bool(name and len(name.strip()) >= Config.MIN_NAME_LENGTH)
    def normalize_phone(self, phone):
        """Return a normalized representation for phone numbers."""
        if phone is None:
            return ""
        text = str(phone).strip()
        if not text:
            return ""
        digits = ''.join(ch for ch in text if ch.isdigit())
        if not digits:
            return ""
        return digits
 
    def validate_phone(self, phone):
        normalized = self.normalize_phone(phone)
        if not normalized:
            return False
        digits = ''.join(ch for ch in normalized if ch.isdigit())
        return Config.MIN_PHONE_LENGTH <= len(digits) <= Config.MAX_PHONE_LENGTH
    def validate_address(self, address):
        return bool(address and len(address.strip()) >= Config.MIN_ADDRESS_LENGTH)

class ImageSearchService:
    """Fetch illustrative images using the configured provider (Google CSE by default)."""

    GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
        self.cx = os.getenv("GOOGLE_CSE_CX", "").strip()
        self.enabled = (
            Config.IMAGE_SEARCH_ENABLED
            and Config.IMAGE_SEARCH_PROVIDER == "google_cse"
            and bool(self.api_key and self.cx)
        )
        self.cache_ttl = Config.IMAGE_SEARCH_CACHE_TTL
        self._cache: Dict[str, Dict[str, Any]] = {}

    def is_enabled(self) -> bool:
        return self.enabled

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(key)
        if not cached:
            return None
        if time.time() - cached["ts"] > self.cache_ttl:
            self._cache.pop(key, None)
            return None
        return cached["value"]

    def _cache_set(self, key: str, value: Optional[Dict[str, Any]]) -> None:
        self._cache[key] = {"ts": time.time(), "value": value}

    def fetch_image(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a representative image dict for the given query."""
        if not self.is_enabled():
            return None
        normalized = (query or "").strip()
        if len(normalized) < 2:
            return None
        cache_key = normalized.lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached or None

        params = {
            "q": normalized,
            "searchType": "image",
            "num": max(1, min(Config.IMAGE_SEARCH_MAX_RESULTS, 5)),
            "safe": "active",
            "key": self.api_key,
            "cx": self.cx,
        }
        try:
            response = requests.get(
                self.GOOGLE_ENDPOINT,
                params=params,
                timeout=Config.IMAGE_SEARCH_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items") or []
            for item in items:
                link = item.get("link")
                if not link:
                    continue
                image_meta = item.get("image") or {}
                result = {
                    "image_url": link,
                    "thumbnail_url": image_meta.get("thumbnailLink") or link,
                    "width": image_meta.get("width"),
                    "height": image_meta.get("height"),
                    "context_url": image_meta.get("contextLink") or item.get("link"),
                    "attribution": item.get("displayLink") or item.get("title") or "",
                    "alt": (item.get("snippet") or item.get("title") or normalized).strip(),
                }
                self._cache_set(cache_key, result)
                return result
        except Exception as exc:
            logger.warning("Image search failed for '%s': %s", normalized, exc)
        self._cache_set(cache_key, None)
        return None

    def clear_cache(self) -> None:
        self._cache.clear()

    def analyze_vision(self, image_base64, query, **kwargs):
        for model in Config.VISION_MODELS:
            try:
                headers = {"Content-Type": "application/json"}
                api_key = os.getenv("GROQ_API_KEY", "").strip()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                    "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
                }
                response = requests.post(Config.GROQ_API_URL, json=payload, headers=headers, timeout=Config.VISION_TIMEOUT)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception:
                continue
        raise Exception("All vision models failed")
 
class FileService:
    def __init__(self):
        self.upload_dir = Path(Config.UPLOAD_FOLDER)
        self.upload_dir.mkdir(exist_ok=True)
        # Track last processed file for sticky context
        self._last_processed_file = None
        self._last_file_text = None
        self._last_file_language = None
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        if not text:
            return "en"
        # Count Arabic characters (Unicode range for Arabic)
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return "en"
        arabic_ratio = arabic_chars / total_alpha
        return "ar" if arabic_ratio > 0.3 else "en"
    
    def process_image(self, image_path, target_language: str = None):
        try:
            if not self._is_valid_image(image_path):
                return ""
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                return ""
            
            # Store for sticky context
            self._last_processed_file = str(image_path)
            
            llm_service = ServiceManager().get_llm_service()
            
            # If no target language specified, first detect if image contains Arabic text
            if not target_language:
                # Ask the model to detect language in the image
                detect_prompt = "Look at this image. Does it contain Arabic text/writing? Reply with ONLY 'ar' if it contains Arabic text, or 'en' if it contains English or no text."
                lang_response = llm_service.analyze_vision(base64_image, detect_prompt)
                detected_lang = "ar" if lang_response and "ar" in lang_response.lower()[:10] else "en"
                target_language = detected_lang
                logger.info(f"🖼️ Image language detection: {detected_lang}")
            
            self._last_file_language = target_language
            
            # Generate description in the detected/target language
            if target_language == "ar":
                prompt = "صف هذه الصورة في 30-50 كلمة بالعربية. كن موجزاً ومباشراً. ابدأ بـ 'الملف الذي رفعته هو' ثم أضف المزيد من المعلومات."
            else:
                prompt = "Describe this image in 30-50 words in English. Be concise. Give the summary directly, don't add extra information. Start with 'The file you uploaded is' and add more info about it."
            
            result = llm_service.analyze_vision(base64_image, prompt)
            self._last_file_text = result or ""  # Store for follow-up questions
            return result
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return ""
    
    def process_document(self, doc_path, target_language: str = None):
        try:
            logger.info(f"📄 Processing document: {doc_path}")
            text = self._extract_document_text(doc_path)
            logger.info(f"📄 Extracted text length: {len(text) if text else 0} chars")
            
            # Log Arabic character count for debugging
            if text:
                arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
                logger.info(f"📄 Arabic characters in extracted text: {arabic_chars}")
            
            if not text or len(text.strip()) < 10:
                logger.warning(f"⚠️ Document text too short or empty: {doc_path}")
                return ""
            
            # Detect language if not specified
            detected_lang = self._detect_language(text)
            use_lang = target_language or detected_lang
            logger.info(f"📄 Detected language: {detected_lang}, using: {use_lang}")
            
            # Store for sticky context
            self._last_processed_file = str(doc_path)
            self._last_file_text = text[:8000]  # Store more for follow-up questions
            self._last_file_language = use_lang
            
            # Language-specific prompts
            if use_lang == "ar":
                system_prompt = "أنت مساعد ذكي. قم بتلخيص المستند في 30-50 كلمة بالعربية. كن موجزاً ومباشراً."
                user_prompt = f"المستند:\n\n{text[:4000]}"
            else:
                system_prompt = "You are a helpful assistant. Summarize the document in 30-50 words. Be concise and direct. dont add extra information."
                user_prompt = f"Document:\n\n{text[:4000]}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            llm_service = ServiceManager().get_llm_service()
            result = llm_service.generate_text(messages, max_tokens=150, temperature=0.3)
            logger.info(f"✅ Document analysis complete: {len(result) if result else 0} chars")
            return MessageFormatter.clean_markdown_text(result)
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return ""
    
    def get_last_file_context(self) -> dict:
        """Get context of the last processed file for sticky follow-up questions."""
        return {
            "file_path": self._last_processed_file,
            "text": self._last_file_text,
            "language": self._last_file_language
        }
    
    def answer_followup(self, question: str, target_language: str = None) -> str:
        """Answer a follow-up question about the last processed file."""
        if not self._last_processed_file or not self._last_file_text:
            return "No file context available. Please upload a file first."
        
        use_lang = target_language or self._last_file_language or "en"
        
        if use_lang == "ar":
            system_prompt = "أنت مساعد ذكي. أجب على السؤال بناءً على محتوى المستند. أجب بالعربية في 30-50 كلمة.  dont add extra information."
        else:
            system_prompt = "You are a helpful assistant. Answer the question based on the document content. Respond in English in 30-50 words. dont add extra information."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document content:\n{self._last_file_text[:4000]}\n\nQuestion: {question}"}
        ]
        
        llm_service = ServiceManager().get_llm_service()
        return llm_service.generate_text(messages, max_tokens=150, temperature=0.3)
    
    def find_recent_files(self, file_type):
        extensions = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp'},
            'document': {'.pdf', '.docx', '.txt'}
        }
        if file_type not in extensions:
            return []
        files = []
        if self.upload_dir.exists():
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions[file_type]:
                    files.append(file_path)
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    def _encode_image_to_base64(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            if len(image_bytes) > Config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                if image.width > Config.MAX_IMAGE_DIMENSION or image.height > Config.MAX_IMAGE_DIMENSION:
                    image.thumbnail((Config.MAX_IMAGE_DIMENSION, Config.MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=Config.JPEG_QUALITY)
                image_bytes = img_buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    def _extract_document_text(self, doc_path):
        if doc_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(doc_path)
        elif doc_path.suffix.lower() == '.docx':
            return self._extract_from_docx(doc_path)
        elif doc_path.suffix.lower() == '.txt':
            return self._extract_from_txt(doc_path)
        else:
            raise Exception(f"Unsupported file type: {doc_path.suffix}")
    def _extract_from_pdf(self, pdf_path):
        """Extract text from PDF with Arabic/Unicode support using PyMuPDF."""
        text = ""
        
        # Try PyMuPDF first (better Arabic/Unicode support)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            logger.info(f"📄 PDF has {page_count} pages (using PyMuPDF)")
            
            for i, page in enumerate(doc):
                try:
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_err:
                    logger.warning(f"⚠️ PyMuPDF failed on page {i+1}: {page_err}")
                    continue
            
            doc.close()
            
            # Check if we got meaningful Arabic text
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
            logger.info(f"📄 PyMuPDF extracted: {len(text)} chars, {arabic_chars} Arabic chars")
            
            if text.strip():
                return text.strip()
                
        except ImportError:
            logger.warning("⚠️ PyMuPDF not installed, falling back to PyPDF2")
        except Exception as e:
            logger.warning(f"⚠️ PyMuPDF extraction failed: {e}, trying PyPDF2")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                page_count = len(pdf_reader.pages)
                logger.info(f"📄 PDF has {page_count} pages (using PyPDF2)")
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_err:
                        logger.warning(f"⚠️ Failed to extract page {i+1}: {page_err}")
                        continue
                
                logger.info(f"📄 PyPDF2 extracted: {len(text)} chars from {page_count} pages")
                return text.strip()
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            return ""
    def _extract_from_docx(self, docx_path):
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    def _extract_from_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
 
class MessageFormatter:
    @staticmethod
    def clean_markdown_text(text):
        if not text:
            return text
        # Strip chain-of-thought style tags and obvious reasoning lead-ins
        try:
            import re
            # Remove <think>...</think> blocks if model leaked internal reasoning
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            # Remove stray HTML-like tags that may break frontend rendering
            text = re.sub(r"<\/?[a-zA-Z][^>]*>", "", text)
            # Trim common meta lead-ins like "Okay," "Let's" when leaking planning
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pruned = []
            dropping = True
            for ln in lines:
                if re.match(r"^[A-Za-z0-9].*", ln) and not re.match(r"^(Okay|Let\'s|Let us|I will|I cannot|The user|We need to|Reasoning|Analysis|Plan)\b", ln, flags=re.IGNORECASE):
                    dropping = False
                if not dropping:
                    pruned.append(ln)
            text = (" ".join(pruned) if pruned else text).strip()
        except Exception:
            # Best-effort only; continue with other cleanups
            pass
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        text = re.sub(r'\*\*([^*]*?)\*\*', r'\1', text)
        text = re.sub(r'__([^_]*?)__', r'\1', text)
        text = re.sub(r'\*([^*]*?)\*', r'\1', text)
        text = re.sub(r'_([^_]*?)_', r'\1', text)
        text = re.sub(r'^#{1,6}\s*(.*)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', '', text)
        text = text.replace('**', '').replace('__', '')
        return ' '.join(text.split()).strip()
 
    @staticmethod
    def truncate_sentences(text: str, max_sentences: int = 2) -> str:
        """Return only the first `max_sentences` sentences from text."""
        if not text or max_sentences <= 0:
            return text
        try:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if not sentences:
                return text.strip()
            truncated = ' '.join(sentences[:max_sentences]).strip()
            return truncated or text.strip()
        except Exception:
            return text.strip()
    @staticmethod
    def get_current_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
 
class LLMTranslationService:
    """Translation service using GROQ Qwen3.3 32B for Arabic ↔ English translation"""
   
    def __init__(self):
        self.llm_service = None  # Will be initialized lazily to avoid circular imports
        self.cache = {}  # In-memory cache as fallback
        self.redis_cache = None  # Redis cache (will be initialized if available)
        self._init_redis_cache()
        self.translation_prompts = {
            "ar_to_en": """You are a professional Arabic to English translator. Translate the following Arabic text to clear, natural English while preserving the exact meaning, tone, and context. Only return the translation, no explanations.
 
Arabic: {text}
 
English:""",
           
            "en_to_ar": """You are a professional English to Arabic translator. Translate the following English text to clear, natural Arabic while preserving the exact meaning, tone, and context. Only return the translation, no explanations.
 
English: {text}
 
Arabic:"""
        }
        self._markdown_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        self.COMMON_PHRASES = {
        "learn more": "اعرف المزيد",
        "read more": "اقرأ المزيد",
        "view details": "عرض التفاصيل",
        "submit": "إرسال",
        "cancel": "إلغاء",
        "close": "إغلاق",
        "open": "فتح",
        "search": "بحث",
        "menu": "القائمة",
        "home": "الرئيسية",
        "back": "رجوع",
        "next": "التالي",
        "previous": "السابق",
        "yes": "نعم",
        "no": "لا",
        "ok": "موافق",
        "confirm": "تأكيد",
        "apply": "تطبيق",
        "reset": "إعادة تعيين",
        "save": "حفظ",
        "delete": "حذف",
        "edit": "تعديل",
        "update": "تحديث",
        "create": "إنشاء",
        "new": "جديد",
        "add": "إضافة",
        "remove": "إزالة",
        "select": "تحديد",
        "choose": "اختر",
        "upload": "رفع",
        "download": "تحميل",
        "share": "مشاركة",
        "send": "إرسال",
        "reply": "رد",
        "forward": "إعادة توجيه",
        "copy": "نسخ",
        "paste": "لصق",
        "cut": "قص",
        "undo": "تراجع",
        "redo": "إعادة",
        "print": "طباعة",
        "export": "تصدير",
        "import": "استيراد",
        "filter": "تصفية",
        "sort": "فرز",
        "group": "تجميع",
        "list": "قائمة",
        "grid": "شبكة",
        "table": "جدول",
        "chart": "رسم بياني",
        "graph": "رسم بياني",
        "map": "خريطة",
        "calendar": "تقويم",
        "timeline": "خط زمن",
        "form": "نموذج",
        "report": "تقرير",
        "dashboard": "لوحة القيادة",
        "brochure": "كتيب",
        "click here": "اضغط هنا",
        "error": "خطأ",
        "success": "نجاح",
        "warning": "تحذير",
        "info": "معلومات",
        "help": "مساعدة",
        "settings": "الإعدادات",
        "profile": "الملف الشخصي",
        "logout": "تسجيل الخروج",
        "login": "تسجيل الدخول",
        "welcome": "مرحباً"
    }

    def _translate_plain_en_to_ar(self, english_text: str) -> str:
        """Translate text without preserving Markdown (used internally)."""
        if not english_text or not english_text.strip():
            return english_text

        # Check common phrases first (case-insensitive)
        lower_text = english_text.lower().strip().strip(".,!?:;")
        if lower_text in self.COMMON_PHRASES:
            return self.COMMON_PHRASES[lower_text]

        cached = self._get_cached_translation(english_text, "en_to_ar_plain")
        if cached:
            return cached

        # Use a more robust prompt to prevent hallucination/garbage output
        prompt = (
            "You are a professional translator. Translate the following English text to Arabic.\n"
            "Rules:\n"
            "1. Provide ONLY the Arabic translation.\n"
            "2. Do NOT add any explanations, notes, or extra text.\n"
            "3. Do NOT repeat the input text.\n"
            "4. If the text is a proper noun or technical term that should not be translated, keep it as is.\n"
            "5. Ensure the translation is natural and grammatically correct.\n"
            "6. If the text is a short UI label (like a button), translate it concisely (1-3 words).\n\n"
            f"English: {english_text}\n\n"
            "Arabic:"
        )
        messages = [{"role": "user", "content": prompt}]

        # Use a lower temperature for more deterministic output
        translation = self._get_llm_service().generate_text(
            messages=messages,
            model=Config.TRANSLATION_MODEL,
            max_tokens=Config.TRANSLATION_MAX_TOKENS,
            temperature=0.1,  # Lower temperature to reduce hallucinations
            timeout=Config.TRANSLATION_TIMEOUT
        )

        translation = translation.strip()
        translation = self._strip_reasoning(translation, target_lang="ar")
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]

        self._cache_translation(english_text, translation, "en_to_ar_plain")
        return translation

    def _segment_text_with_links(self, text: str) -> List[Dict[str, str]]:
        segments: List[Dict[str, str]] = []
        if not text:
            return segments
        last_index = 0
        for match in self._markdown_link_pattern.finditer(text):
            if match.start() > last_index:
                segments.append({"type": "text", "value": text[last_index:match.start()]})
            segments.append({"type": "link", "label": match.group(1), "url": match.group(2)})
            last_index = match.end()
        if last_index < len(text):
            segments.append({"type": "text", "value": text[last_index:]})
        return segments

    def _sanitize_markdown_label(self, translated_label: str, fallback: str) -> str:
        """Ensure translated link labels cannot break markdown syntax."""
        clean = (translated_label or "").strip()
        if not clean:
            return fallback
        clean = re.sub(r"\s+", " ", clean)
        clean = clean.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        clean = clean.strip()
        return clean if clean else fallback

    def _translate_text_segment_en_to_ar(self, segment: str) -> str:
        if not segment:
            return segment
        if segment.strip() == "":
            return segment
        prefix_match = re.match(r"^\s*", segment)
        prefix = prefix_match.group(0) if prefix_match else ""
        suffix_match = re.search(r"\s*$", segment)
        suffix = suffix_match.group(0) if suffix_match else ""
        core_start = len(prefix)
        core_end = len(segment) - len(suffix) if suffix else len(segment)
        core = segment[core_start:core_end]
        translated_core = self._translate_plain_en_to_ar(core) if core else core
        return f"{prefix}{translated_core}{suffix}"

    def _normalize_arabic_digits(self, text: str) -> str:
        """Convert Eastern Arabic digits to ASCII digits so Markdown numbering stays intact."""
        if not text:
            return text
        digit_map = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        return text.translate(digit_map)
   
    def _init_redis_cache(self):
        """Initialize Redis cache if available"""
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_db = int(os.getenv('REDIS_DB', '0'))
           
            self.redis_cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_cache.ping()
            logger.info("✅ Redis cache initialized successfully")
        except ImportError:
            logger.warning("⚠️ Redis not installed. Using in-memory cache only.")
            self.redis_cache = None
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Using in-memory cache only.")
            self.redis_cache = None
   
    def _get_llm_service(self):
        """Lazy initialization of LLM service to avoid circular imports"""
        if self.llm_service is None:
            self.llm_service = get_service_manager().get_llm_service()
        return self.llm_service
   
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        Returns: (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 2:
            return "en", 0.5  # Default to English for very short text
       
        text = text.strip()
       
        # Check for Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
       
        if total_chars == 0:
            return "en", 0.5  # Default for non-alphabetic text
       
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
       
        # If more than 30% Arabic characters, likely Arabic
        if arabic_ratio > 0.3:
            confidence = min(0.9, 0.5 + arabic_ratio)
            return "ar", confidence
       
        # Use langdetect if available for better detection
        if LANGDETECT_AVAILABLE:
            try:
                detected_langs = detect_langs(text)
                for lang_obj in detected_langs:
                    if lang_obj.lang == "ar" and lang_obj.prob > Config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
                        return "ar", lang_obj.prob
                    elif lang_obj.lang == "en" and lang_obj.prob > Config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
                        return "en", lang_obj.prob
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
       
        # Default to English if uncertain
        return "en", 0.6
   
    def _get_cached_translation(self, text: str, direction: str) -> Optional[str]:
        """Get cached translation from Redis or in-memory cache"""
        cache_key = f"translation:{direction}:{hash(text)}"
       
        # Try Redis first
        if self.redis_cache:
            try:
                cached = self.redis_cache.get(cache_key)
                if cached:
                    logger.info(f"Redis cache hit ({direction.upper()})")
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
       
        # Fallback to in-memory cache
        if cache_key in self.cache:
            logger.info(f"Memory cache hit ({direction.upper()})")
            return self.cache[cache_key]
       
        return None
   
    def _cache_translation(self, text: str, translation: str, direction: str):
        """Cache translation in Redis and in-memory cache"""
        cache_key = f"translation:{direction}:{hash(text)}"
        ttl = int(os.getenv('TRANSLATION_CACHE_TTL', '3600'))  # 1 hour default
       
        # Cache in Redis
        if self.redis_cache:
            try:
                self.redis_cache.setex(cache_key, ttl, translation)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
       
        # Cache in memory (with size limit)
        self.cache[cache_key] = translation
        if len(self.cache) > 1000:  # Limit memory cache size
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]
 
    def translate_arabic_to_english(self, arabic_text: str) -> str:
        """Translate Arabic text to English using GROQ LLM"""
        if not arabic_text or not arabic_text.strip():
            return arabic_text
       
        # Check cache first
        cached_translation = self._get_cached_translation(arabic_text, "ar_to_en")
        if cached_translation:
            return cached_translation
       
        try:
            prompt = self.translation_prompts["ar_to_en"].format(text=arabic_text)
            messages = [{"role": "user", "content": prompt}]
           
            translation = self._get_llm_service().generate_text(
                messages=messages,
                model=Config.TRANSLATION_MODEL,
                max_tokens=Config.TRANSLATION_MAX_TOKENS,
                temperature=Config.TRANSLATION_TEMPERATURE,
                timeout=Config.TRANSLATION_TIMEOUT
            )
           
            # Clean up the translation
            translation = translation.strip()
            translation = self._strip_reasoning(translation, target_lang="en")
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
           
            # Cache the result
            self._cache_translation(arabic_text, translation, "ar_to_en")
            logger.info(f"Translated AR→EN: {arabic_text[:50]}... → {translation[:50]}...")
           
            return translation
           
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {e}")
            return f"[Translation Error] {arabic_text}"
   
    def translate_english_to_arabic(self, english_text: str) -> str:
        """
        Translate English text to Arabic while preserving links (Markdown and HTML).
        
        OPTIMIZED: Uses batched translation to reduce API calls.
        """
        if not english_text or not english_text.strip():
            return english_text

        cached_translation = self._get_cached_translation(english_text, "en_to_ar")
        if cached_translation:
            return cached_translation

        try:
            # Pattern for HTML links: <a href='url'>text</a>
            html_link_pattern = re.compile(r"<a\s+href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>", re.IGNORECASE)
            
            lines_en = english_text.splitlines()
            has_html_links = bool(html_link_pattern.search(english_text))

            if has_html_links:
                # OPTIMIZED: Collect all texts to translate, then batch translate
                texts_to_translate: List[str] = []
                line_mapping: List[Tuple[int, str, Any]] = []  # (line_idx, type, extra_data)
                
                for idx, line in enumerate(lines_en):
                    stripped = line.strip()
                    if not stripped:
                        line_mapping.append((idx, "empty", None))
                        continue
                    
                    if stripped == "---MESSAGE_SPLIT---":
                        line_mapping.append((idx, "split", None))
                        continue
                    
                    # Extract HTML links and surrounding text
                    html_match = html_link_pattern.search(stripped)
                    if html_match:
                        # Get link text to translate
                        link_text = html_match.group(2)
                        url = html_match.group(1)
                        texts_to_translate.append(link_text)
                        line_mapping.append((idx, "html_link", {"url": url, "original": stripped, "text_idx": len(texts_to_translate) - 1}))
                    else:
                        texts_to_translate.append(stripped)
                        line_mapping.append((idx, "text", {"text_idx": len(texts_to_translate) - 1}))
                
                # Batch translate all collected texts
                if texts_to_translate:
                    translations = self._batch_translate_en_to_ar(texts_to_translate)
                else:
                    translations = []
                
                # Reconstruct lines
                translated_lines: List[str] = [""] * len(lines_en)
                for idx, line_type, data in line_mapping:
                    if line_type == "empty":
                        translated_lines[idx] = ""
                    elif line_type == "split":
                        translated_lines[idx] = "---MESSAGE_SPLIT---"
                    elif line_type == "html_link":
                        url = data["url"]
                        text_idx = data["text_idx"]
                        translated_text = translations[text_idx] if text_idx < len(translations) else data.get("original", "")
                        # Preserve number prefix if present
                        prefix_match = re.match(r'^(\d+)\.\s*', data["original"])
                        prefix = prefix_match.group(0) if prefix_match else ""
                        translated_lines[idx] = f"{prefix}<a href='{url}' target='_blank' style='color: #29b6f6; font-weight: bold; text-decoration: none;'>{translated_text}</a>"
                    elif line_type == "text":
                        text_idx = data["text_idx"]
                        translated_lines[idx] = translations[text_idx] if text_idx < len(translations) else lines_en[idx]
                
                translation = "\n".join(translated_lines)
            else:
                # For simple text without links, translate as single block
                translation = self._translate_plain_en_to_ar(english_text)

            translation = self._normalize_arabic_digits(translation)
            self._cache_translation(english_text, translation, "en_to_ar")
            logger.info(f"Translated EN->AR: {english_text[:50]}... -> {translation[:50]}...")

            return translation
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {e}")
            return english_text  # Return original English if translation fails
   
    def translate_with_fallback(self, text: str, direction: str) -> str:
        """Translate with fallback handling"""
        try:
            if direction == "ar_to_en":
                return self.translate_arabic_to_english(text)
            elif direction == "en_to_ar":
                return self.translate_english_to_arabic(text)
            else:
                logger.error(f"Unknown translation direction: {direction}")
                return text
        except Exception as e:
            logger.error(f"Translation failed ({direction}): {e}")
            if direction == "ar_to_en":
                return f"[Translation Error] {text}"
            else:
                return text  # Return English if Arabic translation fails
   
    def is_arabic(self, text: str) -> bool:
        """Quick check if text contains Arabic characters"""
        if not text:
            return False
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return arabic_chars > 0

    def translate_adaptive_card(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate all text content in an Adaptive Card from English to Arabic.
        
        OPTIMIZED: Uses BATCHED translation (ONE LLM call) instead of per-field calls.
        This reduces API calls from 10-20 down to 1-2.
        
        Returns a new card dict with translated content.
        """
        if not card or not isinstance(card, dict):
            return card
        
        from copy import deepcopy
        translated_card = deepcopy(card)
        
        # Text fields that should be translated
        TEXT_FIELDS = {"text", "title", "label", "placeholder", "altText", "fallbackText"}
        # Fields that should NOT be translated (URLs, IDs, types, etc.)
        SKIP_FIELDS = {"type", "url", "id", "$schema", "version", "style", "size", "weight", 
                       "color", "horizontalAlignment", "verticalContentAlignment", "spacing",
                       "separator", "wrap", "isSubtle", "width", "height", "data", "verb",
                       "iconUrl", "backgroundImage", "bleed", "minHeight", "fallback",
                       "requires", "lang", "speak", "selectAction", "refresh", "authentication"}
        
        def should_translate(text: str) -> bool:
            """Check if text should be translated"""
            if not text or not isinstance(text, str):
                return False
            text = text.strip()
            if len(text) < 2:
                return False
            if text.startswith(("http://", "https://", "data:", "mailto:")):
                return False
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if arabic_chars > len(text) * 0.3:
                return False
            if re.match(r'^[\d\s\-\/\:\.\,\$\%]+$', text):
                return False
            return True
        
        # STEP 1: Extract all translatable texts with their paths
        texts_to_translate: List[Tuple[str, List]] = []  # (text, path)
        
        def extract_texts(node: Any, path: List) -> None:
            """Recursively extract texts and their paths"""
            if isinstance(node, dict):
                for key, val in node.items():
                    if key in SKIP_FIELDS:
                        continue
                    elif key in TEXT_FIELDS and isinstance(val, str) and should_translate(val):
                        texts_to_translate.append((val, path + [key]))
                    elif key == "facts" and isinstance(val, list):
                        for i, fact in enumerate(val):
                            if isinstance(fact, dict):
                                if isinstance(fact.get("title"), str) and should_translate(fact["title"]):
                                    texts_to_translate.append((fact["title"], path + [key, i, "title"]))
                                if isinstance(fact.get("value"), str) and should_translate(fact["value"]):
                                    texts_to_translate.append((fact["value"], path + [key, i, "value"]))
                    elif key in ("body", "items", "columns", "actions", "card", "inlines"):
                        extract_texts(val, path + [key])
                    else:
                        extract_texts(val, path + [key])
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    extract_texts(item, path + [i])
        
        extract_texts(translated_card, [])
        
        if not texts_to_translate:
            logger.info("✅ No texts to translate in Adaptive Card")
            return translated_card
        
        # STEP 2: Batch translate ALL texts in ONE LLM call
        logger.info(f"🔄 Batched translation: {len(texts_to_translate)} texts in 1 LLM call")
        
        # Build batch prompt
        texts_only = [t[0] for t in texts_to_translate]
        translations = self._batch_translate_en_to_ar(texts_only)
        
        # STEP 3: Apply translations back to card
        def set_nested(obj, path: List, value):
            """Set a value in a nested dict/list using a path"""
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = value
        
        for (original_text, path), translated in zip(texts_to_translate, translations):
            try:
                set_nested(translated_card, path, translated)
            except (KeyError, IndexError) as e:
                logger.warning(f"Failed to set translation at path {path}: {e}")
        
        # Add RTL lang attribute
        if "lang" not in translated_card:
            translated_card["lang"] = "ar"
        
        logger.info("✅ Adaptive Card translated to Arabic (batched)")
        return translated_card
    
    def _batch_translate_en_to_ar(self, texts: List[str]) -> List[str]:
        """
        Translate multiple texts in ONE LLM call to save API quota.
        Returns list of translations in same order as input.
        """
        if not texts:
            return []
        
        # Check cache first - return cached ones, only translate uncached
        uncached_indices = []
        results = [""] * len(texts)
        
        for i, text in enumerate(texts):
            # Check common phrases
            lower_text = text.lower().strip().strip(".,!?:;")
            if lower_text in self.COMMON_PHRASES:
                results[i] = self.COMMON_PHRASES[lower_text]
                continue
            
            # Check cache
            cached = self._get_cached_translation(text, "en_to_ar_plain")
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)
        
        if not uncached_indices:
            logger.info(f"✅ All {len(texts)} texts found in cache")
            return results
        
        # Build batch prompt for uncached texts
        uncached_texts = [texts[i] for i in uncached_indices]
        
        # Format: numbered list for easy parsing
        numbered_input = "\n".join(f"{i+1}. {text}" for i, text in enumerate(uncached_texts))
        
        prompt = f"""You are a professional English to Arabic translator.
Translate each numbered English line to Arabic. Return ONLY the Arabic translations, one per line, keeping the same numbering.

English texts:
{numbered_input}

Arabic translations (same numbering):"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self._get_llm_service().generate_text(
                messages=messages,
                model=Config.TRANSLATION_MODEL,
                max_tokens=min(1500, len(uncached_texts) * 100),  # Reasonable limit
                temperature=0.1,
                timeout=Config.TRANSLATION_TIMEOUT + 10
            )
            
            # Parse response - extract numbered lines
            lines = response.strip().split("\n")
            translations = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Match "1. translation" or "1- translation" or "1) translation"
                match = re.match(r'^(\d+)[.\-\)]\s*(.+)$', line)
                if match:
                    idx = int(match.group(1)) - 1  # 0-based
                    trans = match.group(2).strip()
                    if 0 <= idx < len(uncached_texts):
                        translations[idx] = trans
            
            # Apply translations and cache them
            for local_idx, original_idx in enumerate(uncached_indices):
                if local_idx in translations:
                    trans = translations[local_idx]
                    trans = self._strip_reasoning(trans, "ar")
                    results[original_idx] = trans
                    # Cache for future use
                    self._cache_translation(texts[original_idx], trans, "en_to_ar_plain")
                else:
                    # Fallback: keep original if parsing failed
                    results[original_idx] = texts[original_idx]
                    logger.warning(f"Batch translation missing index {local_idx}")
            
            logger.info(f"✅ Batch translated {len(uncached_indices)} texts in 1 LLM call")
            return results
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Fallback: return originals
            for i in uncached_indices:
                results[i] = texts[i]
            return results
   
    def clear_cache(self):
        """Clear translation cache"""
        self.cache.clear()
        logger.info("Translation cache cleared")
 
    def _strip_reasoning(self, text: str, target_lang: str) -> str:
        """Remove common chain-of-thought artifacts and keep the likely translation body.
        For EN->AR, keep content from the first Arabic character onward.
        For AR->EN, remove <think> blocks and leading meta lines like 'Okay,' or 'Let's'.
        """
        if not text:
            return text
        try:
            # Remove <think>...</think> blocks
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            # Trim whitespace
            text = text.strip()
            if target_lang == "ar":
                # Keep from first Arabic char onward and drop stray non-Arabic symbols
                for idx, ch in enumerate(text):
                    if '\u0600' <= ch <= '\u06FF':
                        ar = text[idx:]
                        # Remove any characters outside Arabic block, digits, whitespace and common punctuation
                        ar = re.sub(r"[^\u0600-\u06FF0-9\s\.,!\?;:\-\[\]\(\)]+", "", ar)
                        # Normalize spaces while preserving line breaks for markdown structure
                        ar = re.sub(r"[ \t]+", " ", ar)
                        ar = re.sub(r"\r\n", "\n", ar)
                        ar = re.sub(r"\n{3,}", "\n\n", ar)
                        return ar.strip()
                return text.strip()
            else:
                # Remove obvious reasoning lead-ins in English
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                pruned = []
                dropping = True
                for ln in lines:
                    # stop dropping when the line looks like a sentence that could be a translation
                    if re.match(r"^[A-Za-z0-9].*", ln) and not re.match(r"^(Okay|Let\'s|Let us|I need to|I will|The user|We need to|Reasoning|Analysis)\b", ln, flags=re.IGNORECASE):
                        dropping = False
                    if not dropping:
                        pruned.append(ln)
                return (" ".join(pruned) if pruned else text).strip()
        except Exception:
            return text