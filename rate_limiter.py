"""
Global Rate Limiter Module - SHARED ACROSS PROCESSES

This module provides a cross-process rate limiter using file-based state.
This is critical because MCP servers run as separate subprocesses, so 
in-memory singletons don't share state.

Algorithm: Sliding Window Log with File Locking
- Stores call timestamps in a shared JSON file
- Uses file locking to prevent race conditions
- All processes (main app + MCP servers) share the same limit
"""

import asyncio
import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Shared state file location
RATE_LIMIT_STATE_FILE = Path(__file__).parent / ".rate_limit_state.json"


class GlobalRateLimiter:
    """
    Cross-process rate limiter using file-based shared state.
    
    Uses Sliding Window Log algorithm with file locking:
    - All processes read/write to the same state file
    - File locking prevents race conditions
    - Timestamps are shared across main app and MCP subprocesses
    
    Configuration via environment variables:
    - RATE_LIMIT_MAX_CALLS: Max calls per window (default: 25)
    - RATE_LIMIT_WINDOW_SECONDS: Window size in seconds (default: 60)
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def _ensure_initialized(self):
        """Lazy initialization to read env vars at runtime."""
        if not self._initialized:
            # Conservative defaults - GROQ free tier is ~30/min but bursts trigger 429
            self._max_calls = int(os.getenv("RATE_LIMIT_MAX_CALLS", "20"))
            self._window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
            self._state_file = RATE_LIMIT_STATE_FILE
            self._initialized = True
            logger.info(f"🚦 Rate Limiter initialized: {self._max_calls} calls / {self._window_seconds}s (shared file: {self._state_file.name})")
    
    @property
    def max_calls(self) -> int:
        self._ensure_initialized()
        return self._max_calls
    
    @property
    def window_seconds(self) -> int:
        self._ensure_initialized()
        return self._window_seconds
    
    def _read_state(self) -> List[float]:
        """Read call timestamps from shared file."""
        self._ensure_initialized()
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                    try:
                        data = json.load(f)
                        return data.get("calls", [])
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Rate limiter state read error: {e}")
        return []
    
    def _write_state(self, calls: List[float]):
        """Write call timestamps to shared file with exclusive lock."""
        self._ensure_initialized()
        try:
            # Open for read+write, create if doesn't exist
            mode = 'r+' if self._state_file.exists() else 'w'
            with open(self._state_file, mode) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                try:
                    f.seek(0)
                    f.truncate()
                    json.dump({"calls": calls, "updated": time.time()}, f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError as e:
            logger.error(f"Rate limiter state write error: {e}")
    
    def _get_calls_in_window(self) -> List[float]:
        """Get all calls within the current window."""
        cutoff = time.time() - self.window_seconds
        all_calls = self._read_state()
        return [t for t in all_calls if t > cutoff]
    
    def can_make_call(self) -> bool:
        """Check if we can make another LLM call."""
        calls = self._get_calls_in_window()
        return len(calls) < self.max_calls
    
    def record_call(self):
        """Record that an LLM call was made (atomic read-modify-write)."""
        self._ensure_initialized()
        try:
            # Atomic read-modify-write with exclusive lock
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file if doesn't exist
            if not self._state_file.exists():
                self._state_file.write_text('{"calls": []}')
            
            with open(self._state_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                try:
                    f.seek(0)
                    try:
                        data = json.load(f)
                        calls = data.get("calls", [])
                    except json.JSONDecodeError:
                        calls = []
                    
                    # Clean old calls and add new one
                    cutoff = time.time() - self.window_seconds
                    calls = [t for t in calls if t > cutoff]
                    calls.append(time.time())
                    
                    # Write back
                    f.seek(0)
                    f.truncate()
                    json.dump({"calls": calls, "updated": time.time()}, f)
                    
                    logger.debug(f"📊 Rate limiter: {len(calls)}/{self.max_calls} calls in window")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError as e:
            logger.error(f"Rate limiter record error: {e}")
    
    def calls_remaining(self) -> int:
        """Get number of calls remaining in current window."""
        calls = self._get_calls_in_window()
        return max(0, self.max_calls - len(calls))
    
    def time_until_reset(self) -> float:
        """Get seconds until oldest call expires from window."""
        calls = self._get_calls_in_window()
        if not calls:
            return 0
        oldest = min(calls)
        return max(0, (oldest + self.window_seconds) - time.time())
    
    def wait_if_needed(self) -> float:
        """
        SYNC version: Block until we can make a call. Returns wait time.
        Use this before making an LLM call in synchronous code.
        """
        calls = self._get_calls_in_window()
        if len(calls) < self.max_calls:
            return 0
        
        # Calculate wait time
        oldest = min(calls)
        wait_time = (oldest + self.window_seconds) - time.time() + 0.5  # Add buffer
        if wait_time > 0:
            logger.warning(f"🚦 Rate limit reached ({len(calls)}/{self.max_calls}). Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        return wait_time
    
    async def wait_for_slot_async(self) -> float:
        """
        ASYNC version: Wait until we can make a call. Returns wait time.
        Use this in async contexts (MCP servers, etc.)
        
        Also enforces minimum delay between calls to prevent API bursts.
        """
        # Minimum 1 second between calls to prevent burst
        calls = self._get_calls_in_window()
        if calls:
            last_call = max(calls)
            since_last = time.time() - last_call
            if since_last < 1.0:
                wait_burst = 1.0 - since_last
                logger.debug(f"🚦 Anti-burst delay: {wait_burst:.1f}s")
                await asyncio.sleep(wait_burst)
        
        # Check if we're at the limit
        calls = self._get_calls_in_window()  # Re-read after potential sleep
        if len(calls) < self.max_calls:
            return 0
        
        # Calculate wait time
        oldest = min(calls)
        wait_time = (oldest + self.window_seconds) - time.time() + 0.5  # Add buffer
        if wait_time > 0:
            logger.warning(f"🚦 Rate limit reached ({len(calls)}/{self.max_calls}). Async waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
        return wait_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter stats."""
        calls = self._get_calls_in_window()
        return {
            "calls_in_window": len(calls),
            "max_calls": self.max_calls,
            "window_seconds": self.window_seconds,
            "calls_remaining": self.calls_remaining(),
            "time_until_reset": round(self.time_until_reset(), 1)
        }
    
    def reset(self):
        """Clear all recorded calls (for testing)."""
        self._ensure_initialized()
        try:
            self._state_file.write_text('{"calls": []}')
            logger.info("🔄 Rate limiter state reset")
        except IOError as e:
            logger.error(f"Rate limiter reset error: {e}")


# Global singleton instance
_global_rate_limiter = None


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get the global rate limiter singleton."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter
