"""
Langfuse Telemetry Module for LLM Observability

Provides centralized telemetry tracking for all LLM calls across the application:
- Automatic tracing of LLM generations
- Token usage tracking
- Error logging
- Performance metrics

Usage:
    from architecture.telemetry import get_langfuse, trace_llm_call
    
    # Context manager for tracing
    with trace_llm_call("chat-completion", model="llama-3.1-8b") as trace:
        response = await llm.invoke(prompt)
        trace.update(output=response, usage={"tokens": 100})
    
    # Or use decorator
    @trace_generation("sql-generation")
    async def generate_sql(query: str):
        ...
"""

import os
import logging
import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Langfuse
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("⚠️ Langfuse not installed. Telemetry disabled. Install with: pip install langfuse")

# Global Langfuse client instance
_langfuse_client: Optional["Langfuse"] = None
_telemetry_enabled: bool = False


@dataclass
class GenerationTrace:
    """Wrapper for a Langfuse generation trace."""
    name: str
    model: str
    start_time: float
    _generation: Any = None  # Langfuse generation context manager
    
    def update(
        self,
        output: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        level: str = "DEFAULT"
    ):
        """Update the generation with output and metrics."""
        if not _telemetry_enabled or self._generation is None:
            return
        
        try:
            # Build update kwargs
            update_kwargs = {}
            
            if output is not None:
                update_kwargs["output"] = output
            
            if usage:
                # Map to Langfuse expected format - uses 'usage_details' not 'usage'
                # Langfuse expects: input, output, total (or prompt_tokens, completion_tokens, total_tokens)
                update_kwargs["usage_details"] = {
                    "input": usage.get("prompt_tokens", usage.get("input_tokens", usage.get("input", 0))),
                    "output": usage.get("completion_tokens", usage.get("output_tokens", usage.get("output", 0))),
                    "total": usage.get("total_tokens", usage.get("total", 0))
                }
            
            if metadata:
                update_kwargs["metadata"] = metadata
            
            if error:
                update_kwargs["level"] = "ERROR"
                update_kwargs["status_message"] = error
            
            # Update the generation
            if update_kwargs:
                self._generation.update(**update_kwargs)
            
        except Exception as e:
            logger.debug(f"Telemetry update failed: {e}")
    
    def score(self, name: str, value: float, comment: Optional[str] = None):
        """Add a score to this generation."""
        if not _telemetry_enabled:
            return
        
        try:
            langfuse = get_langfuse()
            if langfuse:
                langfuse.score_current_trace(name=name, value=value, comment=comment)
        except Exception as e:
            logger.debug(f"Telemetry score failed: {e}")


def init_langfuse() -> Optional["Langfuse"]:
    """
    Initialize and return the Langfuse client.
    
    Reads configuration from environment variables:
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_BASE_URL (optional, defaults to cloud)
    
    Returns:
        Langfuse client or None if not available/configured
    """
    global _langfuse_client, _telemetry_enabled
    
    if _langfuse_client is not None:
        return _langfuse_client
    
    if not LANGFUSE_AVAILABLE:
        logger.info("ℹ️ Langfuse not available - telemetry disabled")
        return None
    
    # Check for required environment variables
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    
    if not secret_key or not public_key:
        logger.info("ℹ️ Langfuse credentials not configured - telemetry disabled")
        return None
    
    try:
        base_url = os.getenv("LANGFUSE_BASE_URL", "").strip() or None
        
        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=base_url,
            flush_at=10,  # Flush after 10 events
            flush_interval=5  # Or every 5 seconds
        )
        
        _telemetry_enabled = True
        logger.info(f"✅ Langfuse telemetry initialized (host: {base_url or 'cloud'})")
        return _langfuse_client
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Langfuse: {e}")
        return None


def get_langfuse() -> Optional["Langfuse"]:
    """Get the global Langfuse client, initializing if needed."""
    global _langfuse_client
    
    if _langfuse_client is None:
        init_langfuse()
    
    return _langfuse_client


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled and working."""
    return _telemetry_enabled and _langfuse_client is not None


@contextmanager
def trace_llm_call(
    name: str,
    model: str = "unknown",
    input_data: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Context manager for tracing LLM calls.
    
    Args:
        name: Name of the generation (e.g., "sql-generation", "chat-response")
        model: Model identifier being used
        input_data: Input prompt/messages (optional)
        model_parameters: Temperature, max_tokens, etc.
        metadata: Additional metadata to attach
        trace_id: Optional trace ID to link to
        session_id: Session identifier
        user_id: User identifier
        
    Yields:
        GenerationTrace object with update() method
        
    Example:
        with trace_llm_call("chat", model="llama-3.1-8b", input_data={"query": "hello"}) as trace:
            response = await llm.invoke(prompt)
            trace.update(output=response.content, usage={"total_tokens": 150})
    """
    trace = GenerationTrace(
        name=name,
        model=model,
        start_time=time.time()
    )
    
    if not _telemetry_enabled:
        yield trace
        return
    
    langfuse = get_langfuse()
    if langfuse is None:
        yield trace
        return
    
    try:
        # Use Langfuse v3 API: start_as_current_observation with as_type='generation'
        # This creates a trace automatically and starts a generation within it
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_data,
            model_parameters=model_parameters or {},
            metadata=metadata,
        ) as generation:
            trace._generation = generation
            yield trace
            
    except Exception as e:
        logger.debug(f"Telemetry trace creation failed: {e}")
        yield trace
    
    finally:
        # Ensure flush happens periodically (Langfuse handles batching)
        pass


def trace_generation(
    name: str,
    model_param: str = "model",
    extract_usage: bool = True
):
    """
    Decorator for tracing LLM generation functions.
    
    Args:
        name: Name for the generation trace
        model_param: Name of the model parameter in the function
        extract_usage: Whether to try extracting usage from response
        
    Example:
        @trace_generation("sql-query-gen")
        async def generate_sql(query: str, model: str = "llama-3.1-8b"):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            model = kwargs.get(model_param, "unknown")
            
            with trace_llm_call(name, model=model, input_data=kwargs) as trace:
                result = await func(*args, **kwargs)
                
                # Try to extract usage if result has it
                if extract_usage and hasattr(result, "usage"):
                    trace.update(output=str(result), usage=vars(result.usage) if hasattr(result.usage, "__dict__") else {})
                elif isinstance(result, str):
                    trace.update(output=result)
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            model = kwargs.get(model_param, "unknown")
            
            with trace_llm_call(name, model=model, input_data=kwargs) as trace:
                result = func(*args, **kwargs)
                
                if extract_usage and hasattr(result, "usage"):
                    trace.update(output=str(result), usage=vars(result.usage) if hasattr(result.usage, "__dict__") else {})
                elif isinstance(result, str):
                    trace.update(output=result)
                
                return result
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_llm_event(
    event_type: str,
    data: Dict[str, Any],
    level: str = "DEFAULT"
):
    """
    Log a standalone LLM event to Langfuse.
    
    Useful for logging events outside of generation context.
    
    Args:
        event_type: Type of event (e.g., "fallback-used", "rate-limit-hit")
        data: Event data
        level: Log level (DEFAULT, DEBUG, WARNING, ERROR)
    """
    if not _telemetry_enabled:
        return
    
    langfuse = get_langfuse()
    if langfuse is None:
        return
    
    try:
        # Langfuse v3 uses create_event instead of event
        langfuse.create_event(
            name=event_type,
            metadata=data,
            level=level
        )
    except Exception as e:
        logger.debug(f"Telemetry event logging failed: {e}")


def flush_telemetry():
    """Flush all pending telemetry data."""
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception as e:
            logger.debug(f"Telemetry flush failed: {e}")


def shutdown_telemetry():
    """Shutdown telemetry and flush remaining data."""
    global _langfuse_client, _telemetry_enabled
    
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            _langfuse_client.shutdown()
            logger.info("✅ Langfuse telemetry shutdown complete")
        except Exception as e:
            logger.debug(f"Telemetry shutdown failed: {e}")
        finally:
            _langfuse_client = None
            _telemetry_enabled = False


# Auto-initialize on module import
init_langfuse()
