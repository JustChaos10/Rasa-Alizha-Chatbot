"""
Centralized Redis Client for RASA V2 Chatbot.

Provides:
- Connection pooling
- Graceful error handling (no stack traces to user)
- Configurable TTL with 7-day default
- Consistent interface for all state storage

Usage:
    from architecture.redis_client import get_redis, RedisState
    
    # Simple operations
    redis = get_redis()
    if redis.is_available():
        redis.set("key", "value")
        value = redis.get("key")
    
    # With explicit TTL
    redis.set("key", "value", ttl=3600)  # 1 hour
"""

import os
import json
import logging
from typing import Optional, Any, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


class RedisConfig:
    """Redis configuration from environment variables."""
    
    def __init__(self):
        # Connection settings
        self.url = os.getenv("REDIS_URL", "")
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.db = int(os.getenv("REDIS_DB", 0))
        self.password = os.getenv("REDIS_PASSWORD", None)
        
        # Timeouts
        self.socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", 5.0))
        self.connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT", 5.0))
        
        # Default TTL: 7 days in seconds
        self.default_ttl = int(os.getenv("STATE_TTL_SECONDS", 604800))
        
        # Feature-specific TTLs (override default if set)
        self.chat_memory_ttl = int(os.getenv("CHAT_MEMORY_TTL", self.default_ttl))
        self.contact_form_ttl = int(os.getenv("CONTACT_FORM_TTL", self.default_ttl))
        self.survey_session_ttl = int(os.getenv("SURVEY_SESSION_TTL", self.default_ttl))
        self.stored_info_ttl = int(os.getenv("STORED_INFO_TTL", 2592000))  # 30 days default


class RedisState:
    """
    Centralized Redis client with graceful error handling.
    
    Never raises exceptions to callers - returns None/False instead with logging.
    This ensures Redis downtime doesn't crash the application.
    """
    
    def __init__(self, config: RedisConfig = None):
        self.config = config or RedisConfig()
        self._client = None
        self._available = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis
            
            if self.config.url:
                # Use URL if provided (e.g., redis://localhost:6379/0)
                self._client = redis.from_url(
                    self.config.url,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.connect_timeout,
                    decode_responses=True
                )
            else:
                # Use individual settings
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.connect_timeout,
                    decode_responses=True
                )
            
            # Test connection
            self._client.ping()
            self._available = True
            logger.info(f"Redis connected: {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("Redis package not installed")
            self._available = False
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self._available or not self._client:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._available = False
            return False
    
    def get_unavailable_message(self) -> str:
        """User-friendly message when Redis is down."""
        return "State storage is temporarily unavailable. Please try again later."
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value with optional TTL.
        
        Args:
            key: Redis key
            value: Value (will be JSON-encoded if not string)
            ttl: TTL in seconds (defaults to config.default_ttl)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
            
        try:
            ttl = ttl or self.config.default_ttl
            
            # JSON-encode non-string values
            if not isinstance(value, str):
                value = json.dumps(value)
            
            self._client.set(key, value, ex=ttl)
            return True
            
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value, optionally JSON-decoding it.
        
        Returns:
            The value, or default if not found/error
        """
        if not self.is_available():
            return default
            
        try:
            value = self._client.get(key)
            if value is None:
                return default
            
            # Try to JSON-decode
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.warning(f"Redis GET failed for {key}: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        if not self.is_available():
            return False
            
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE failed for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.is_available():
            return False
            
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set/update TTL on existing key."""
        if not self.is_available():
            return False
            
        try:
            return bool(self._client.expire(key, ttl))
        except Exception as e:
            logger.warning(f"Redis EXPIRE failed for {key}: {e}")
            return False
    
    def hset(self, name: str, key: str, value: Any, ttl: int = None) -> bool:
        """Set hash field."""
        if not self.is_available():
            return False
            
        try:
            if not isinstance(value, str):
                value = json.dumps(value)
            self._client.hset(name, key, value)
            if ttl:
                self._client.expire(name, ttl)
            return True
        except Exception as e:
            logger.warning(f"Redis HSET failed for {name}.{key}: {e}")
            return False
    
    def hget(self, name: str, key: str, default: Any = None) -> Any:
        """Get hash field."""
        if not self.is_available():
            return default
            
        try:
            value = self._client.hget(name, key)
            if value is None:
                return default
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.warning(f"Redis HGET failed for {name}.{key}: {e}")
            return default
    
    def hgetall(self, name: str) -> dict:
        """Get all hash fields."""
        if not self.is_available():
            return {}
            
        try:
            data = self._client.hgetall(name)
            # Try to JSON-decode values
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            return result
        except Exception as e:
            logger.warning(f"Redis HGETALL failed for {name}: {e}")
            return {}
    
    def lpush(self, key: str, *values) -> bool:
        """Push to list (left)."""
        if not self.is_available():
            return False
            
        try:
            encoded = [json.dumps(v) if not isinstance(v, str) else v for v in values]
            self._client.lpush(key, *encoded)
            return True
        except Exception as e:
            logger.warning(f"Redis LPUSH failed for {key}: {e}")
            return False
    
    def lrange(self, key: str, start: int, end: int) -> list:
        """Get list range."""
        if not self.is_available():
            return []
            
        try:
            values = self._client.lrange(key, start, end)
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.append(v)
            return result
        except Exception as e:
            logger.warning(f"Redis LRANGE failed for {key}: {e}")
            return []


# Singleton instance
_redis_instance: Optional[RedisState] = None


def get_redis() -> RedisState:
    """Get the global Redis client instance."""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = RedisState()
    return _redis_instance


def reset_redis() -> None:
    """Reset the global Redis instance (for testing)."""
    global _redis_instance
    _redis_instance = None
