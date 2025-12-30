"""
Global Rate Limiter Module - Standalone with NO dependencies

This module is intentionally kept dependency-free to avoid circular imports.
It provides a singleton rate limiter that can be used across the entire application.

Algorithm: Sliding Window Log
- Tracks timestamps of all LLM calls in the last N seconds
- Blocks/waits if limit exceeded
- More accurate than fixed window counters
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    Global rate limiter to prevent hitting API rate limits.
    
    Uses Sliding Window Log algorithm:
    - Tracks all call timestamps in the window
    - More accurate than fixed bucket counters
    - Prevents burst at window boundaries
    
    Configuration via environment variables:
    - RATE_LIMIT_MAX_CALLS: Max calls per window (default: 25)
    - RATE_LIMIT_WINDOW_SECONDS: Window size in seconds (default: 60)
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._calls = []  # List of timestamps
            cls._instance._initialized = False
        return cls._instance
    
    def _ensure_initialized(self):
        """Lazy initialization to read env vars at runtime."""
        if not self._initialized:
            self._max_calls = int(os.getenv("RATE_LIMIT_MAX_CALLS", "25"))
            self._window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
            self._initialized = True
            logger.info(f"🚦 Rate Limiter initialized: {self._max_calls} calls / {self._window_seconds}s")
    
    @property
    def max_calls(self) -> int:
        self._ensure_initialized()
        return self._max_calls
    
    @property
    def window_seconds(self) -> int:
        self._ensure_initialized()
        return self._window_seconds
    
    def _cleanup_old_calls(self):
        """Remove calls outside the current window."""
        cutoff = time.time() - self.window_seconds
        self._calls = [t for t in self._calls if t > cutoff]
    
    def can_make_call(self) -> bool:
        """Check if we can make another LLM call."""
        self._cleanup_old_calls()
        return len(self._calls) < self.max_calls
    
    def record_call(self):
        """Record that an LLM call was made."""
        self._calls.append(time.time())
        self._cleanup_old_calls()
        logger.debug(f"📊 Rate limiter: {len(self._calls)}/{self.max_calls} calls in window")
    
    def calls_remaining(self) -> int:
        """Get number of calls remaining in current window."""
        self._cleanup_old_calls()
        return max(0, self.max_calls - len(self._calls))
    
    def time_until_reset(self) -> float:
        """Get seconds until oldest call expires from window."""
        self._cleanup_old_calls()
        if not self._calls:
            return 0
        oldest = min(self._calls)
        return max(0, (oldest + self.window_seconds) - time.time())
    
    def wait_if_needed(self) -> float:
        """
        SYNC version: Block until we can make a call. Returns wait time.
        Use this before making an LLM call in synchronous code.
        """
        self._cleanup_old_calls()
        if len(self._calls) < self.max_calls:
            return 0
        
        # Calculate wait time
        wait_time = self.time_until_reset() + 0.5  # Add buffer
        if wait_time > 0:
            logger.warning(f"🚦 Rate limit reached ({len(self._calls)}/{self.max_calls}). Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            self._cleanup_old_calls()
        return wait_time
    
    async def wait_for_slot_async(self) -> float:
        """
        ASYNC version: Wait until we can make a call. Returns wait time.
        Use this in async contexts (MCP servers, etc.)
        """
        self._cleanup_old_calls()
        if len(self._calls) < self.max_calls:
            return 0
        
        # Calculate wait time
        wait_time = self.time_until_reset() + 0.5  # Add buffer
        if wait_time > 0:
            logger.warning(f"🚦 Rate limit reached ({len(self._calls)}/{self.max_calls}). Async waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            self._cleanup_old_calls()
        return wait_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter stats."""
        self._cleanup_old_calls()
        return {
            "calls_in_window": len(self._calls),
            "max_calls": self.max_calls,
            "window_seconds": self.window_seconds,
            "calls_remaining": self.calls_remaining(),
            "time_until_reset": round(self.time_until_reset(), 1)
        }


# Global singleton instance
_global_rate_limiter = None


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get the global rate limiter singleton."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter
