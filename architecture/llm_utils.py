"""
Multi-Provider LLM Client - NOW DELEGATES TO GlobalLLMService

This module provides backward-compatible interfaces (MultiProviderLLM, get_llm)
that now delegate to the centralized GlobalLLMService in shared_utils.py.

All LLM calls in the project MUST go through GlobalLLMService, which handles:
- Centralized rate limiting (sliding window)
- Provider failover (GROQ → Gemini)
- NO duplicate retry logic
- Telemetry
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    GROQ = "groq"
    GOOGLE = "google"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: LLMProvider
    model: str
    success: bool
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # Token usage stats


# =============================================================================
# Import GlobalLLMService - THE SOURCE OF TRUTH FOR ALL LLM CALLS
# =============================================================================

# Import from shared_utils using importlib to avoid circular imports
import importlib.util
_shared_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared_utils.py')
_spec = importlib.util.spec_from_file_location('shared_utils', _shared_utils_path)
_shared_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared_utils)
get_global_llm_service = _shared_utils.get_global_llm_service


class MultiProviderLLM:
    """
    Multi-provider LLM client - NOW DELEGATES TO GlobalLLMService.
    
    This class is kept for backward compatibility. All calls are
    routed through the centralized GlobalLLMService.
    
    Usage:
        llm = MultiProviderLLM()
        response = await llm.invoke("Your prompt here")
        print(response.content)
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        groq_model: str = "llama-3.1-8b-instant",
        google_model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0
    ):
        # Store settings for backward compatibility
        self.groq_model = groq_model
        self.google_model = google_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Get the global LLM service
        self._global_llm = get_global_llm_service()
        
        logger.info(f"✅ GROQ client initialized (model: {groq_model})")
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            logger.info(f"✅ Google Gemini client initialized (model: {google_model})")
    
    async def invoke(self, prompt: str, trace_name: str = "mcp-server-llm") -> LLMResponse:
        """
        Invoke LLM with automatic failover - DELEGATES TO GlobalLLMService.
        
        Args:
            prompt: The prompt to send to the LLM
            trace_name: Name for telemetry trace
            
        Returns:
            LLMResponse with content and metadata
        """
        try:
            # Use GlobalLLMService for the actual call
            result = await self._global_llm.call_async(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                trace_name=trace_name
            )
            
            # Get status to determine which provider was used
            status = self._global_llm.get_status()
            
            # Determine provider from status (approximate - GlobalLLMService doesn't track last provider)
            provider = LLMProvider.GROQ if status["groq"]["failures"] == 0 else LLMProvider.GOOGLE
            model = self.groq_model if provider == LLMProvider.GROQ else self.google_model
            
            return LLMResponse(
                content=result,
                provider=provider,
                model=model,
                success=True,
                error=None,
                usage=None  # Token usage not tracked in GlobalLLMService
            )
            
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            return LLMResponse(
                content=f"Error: {e}",
                provider=LLMProvider.GROQ,
                model=self.groq_model,
                success=False,
                error=str(e)
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all providers - delegates to GlobalLLMService."""
        return self._global_llm.get_status()


# ============================================================================
# Singleton instance for shared use across servers
# ============================================================================

_llm_instance: Optional[MultiProviderLLM] = None


def get_llm(
    groq_model: str = "llama-3.1-8b-instant",
    google_model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 1024
) -> MultiProviderLLM:
    """
    Get or create the shared MultiProviderLLM instance.
    
    NOTE: This now delegates to GlobalLLMService internally.
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = MultiProviderLLM(
            groq_model=groq_model,
            google_model=google_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    return _llm_instance
