"""
Multi-Provider LLM Client with Automatic Failover

Provides a unified interface for LLM calls with automatic rotation between providers:
- Primary: GROQ (Llama 3.1, fast and free)
- Fallback: Google Gemini (when GROQ fails)

Features:
- Automatic failover on 429/503 errors
- Simple round-robin or fallback mechanism
- Unified invoke() interface
- Rate limit awareness
- Langfuse telemetry integration
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Import telemetry using importlib to avoid triggering full package load
import importlib.util
_telemetry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telemetry.py')
_telemetry_spec = importlib.util.spec_from_file_location('telemetry', _telemetry_path)
_telemetry = importlib.util.module_from_spec(_telemetry_spec)
_telemetry_spec.loader.exec_module(_telemetry)
trace_llm_call = _telemetry.trace_llm_call
log_llm_event = _telemetry.log_llm_event

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


class MultiProviderLLM:
    """
    Multi-provider LLM client with automatic failover.
    
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
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.groq_model = groq_model
        self.google_model = google_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize clients
        self.groq_client = None
        self.google_client = None
        
        # Track provider health
        self._provider_failures: Dict[LLMProvider, int] = {
            LLMProvider.GROQ: 0,
            LLMProvider.GOOGLE: 0
        }
        self._max_failures = 3  # Reset after this many successes
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available LLM clients."""
        # Initialize GROQ
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info(f"✅ GROQ client initialized (model: {self.groq_model})")
            except ImportError:
                logger.warning("⚠️ GROQ SDK not installed")
            except Exception as e:
                logger.warning(f"⚠️ GROQ initialization failed: {e}")
        
        # Initialize Google Gemini
        if self.google_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                self.google_client = genai.GenerativeModel(self.google_model)
                logger.info(f"✅ Google Gemini client initialized (model: {self.google_model})")
            except ImportError:
                logger.warning("⚠️ Google Generative AI SDK not installed")
            except Exception as e:
                logger.warning(f"⚠️ Google Gemini initialization failed: {e}")
        
        if not self.groq_client and not self.google_client:
            logger.error("❌ No LLM providers available!")
    
    def _get_provider_order(self) -> List[LLMProvider]:
        """
        Get providers in order of preference.
        Primary: GROQ (fast, free)
        Fallback: Google Gemini
        
        Deprioritize providers with recent failures.
        """
        providers = []
        
        # Add GROQ first (primary) if available and not failing
        if self.groq_client and self._provider_failures[LLMProvider.GROQ] < self._max_failures:
            providers.append(LLMProvider.GROQ)
        
        # Add Google as fallback if available
        if self.google_client and self._provider_failures[LLMProvider.GOOGLE] < self._max_failures:
            providers.append(LLMProvider.GOOGLE)
        
        # If all providers are failing, reset and try again
        if not providers:
            logger.warning("🔄 All providers failing, resetting failure counts")
            self._provider_failures = {p: 0 for p in LLMProvider}
            if self.groq_client:
                providers.append(LLMProvider.GROQ)
            if self.google_client:
                providers.append(LLMProvider.GOOGLE)
        
        return providers
    
    async def _call_groq(self, prompt: str) -> LLMResponse:
        """Call GROQ API."""
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                ),
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content.strip()
            self._provider_failures[LLMProvider.GROQ] = 0  # Reset on success
            
            # Extract token usage from GROQ response
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GROQ,
                model=self.groq_model,
                success=True,
                usage=usage
            )
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate" in error_str or "limit" in error_str
            is_server_error = "503" in error_str or "500" in error_str
            
            if is_rate_limit or is_server_error:
                self._provider_failures[LLMProvider.GROQ] += 1
                logger.warning(f"⚠️ GROQ rate limited/error: {e}")
            else:
                logger.error(f"❌ GROQ call failed: {e}")
            
            return LLMResponse(
                content="",
                provider=LLMProvider.GROQ,
                model=self.groq_model,
                success=False,
                error=str(e)
            )
    
    async def _call_google(self, prompt: str) -> LLMResponse:
        """Call Google Gemini API."""
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.google_client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens
                        }
                    )
                ),
                timeout=self.timeout
            )
            
            content = response.text.strip()
            self._provider_failures[LLMProvider.GOOGLE] = 0  # Reset on success
            
            # Extract token usage from Google response
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GOOGLE,
                model=self.google_model,
                success=True,
                usage=usage
            )
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate" in error_str or "quota" in error_str
            is_server_error = "503" in error_str or "500" in error_str
            
            if is_rate_limit or is_server_error:
                self._provider_failures[LLMProvider.GOOGLE] += 1
                logger.warning(f"⚠️ Google rate limited/error: {e}")
            else:
                logger.error(f"❌ Google call failed: {e}")
            
            return LLMResponse(
                content="",
                provider=LLMProvider.GOOGLE,
                model=self.google_model,
                success=False,
                error=str(e)
            )
    
    async def invoke(self, prompt: str, trace_name: str = "mcp-server-llm") -> LLMResponse:
        """
        Invoke LLM with automatic failover.
        
        Tries providers in order until one succeeds.
        
        Args:
            prompt: The prompt to send to the LLM
            trace_name: Name for telemetry trace
            
        Returns:
            LLMResponse with content and metadata
        """
        providers = self._get_provider_order()
        
        if not providers:
            return LLMResponse(
                content="No LLM providers available",
                provider=LLMProvider.GROQ,
                model="none",
                success=False,
                error="No LLM providers configured"
            )
        
        last_error = None
        
        for provider in providers:
            logger.debug(f"🔄 Trying provider: {provider.value}")
            
            # Determine model for telemetry
            model = self.groq_model if provider == LLMProvider.GROQ else self.google_model
            
            # Telemetry: trace this LLM call
            with trace_llm_call(
                name=trace_name,
                model=f"{provider.value}/{model}",
                input_data={"messages": [{"role": "user", "content": prompt}]},
                model_parameters={"temperature": self.temperature, "max_tokens": self.max_tokens},
                metadata={"provider": provider.value, "prompt_length": len(prompt)}
            ) as trace:
                if provider == LLMProvider.GROQ:
                    response = await self._call_groq(prompt)
                elif provider == LLMProvider.GOOGLE:
                    response = await self._call_google(prompt)
                else:
                    continue
                
                if response.success:
                    logger.info(f"✅ LLM call succeeded via {provider.value}")
                    trace.update(
                        output=response.content,
                        usage=response.usage,  # Pass token usage
                        metadata={"provider_used": provider.value, "success": True}
                    )
                    return response
                
                last_error = response.error
                trace.update(
                    output=f"Failed: {last_error}",
                    metadata={"provider_used": provider.value, "success": False, "error": last_error}
                )
                log_llm_event("provider-fallback", {"provider": provider.value, "error": last_error}, level="WARNING")
                logger.warning(f"⚠️ Provider {provider.value} failed, trying next...")
        
        # All providers failed
        logger.error(f"❌ All LLM providers failed. Last error: {last_error}")
        log_llm_event("all-providers-failed", {"last_error": last_error}, level="ERROR")
        return LLMResponse(
            content=f"All LLM providers failed. Error: {last_error}",
            provider=providers[0] if providers else LLMProvider.GROQ,
            model="none",
            success=False,
            error=last_error
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all providers."""
        return {
            "groq": {
                "available": self.groq_client is not None,
                "model": self.groq_model,
                "failures": self._provider_failures[LLMProvider.GROQ]
            },
            "google": {
                "available": self.google_client is not None,
                "model": self.google_model,
                "failures": self._provider_failures[LLMProvider.GOOGLE]
            }
        }


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
    
    First call configures the instance, subsequent calls return the same instance.
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
