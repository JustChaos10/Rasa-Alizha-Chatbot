"""
Unified Configuration Loader

Single source of truth for all service configurations.
Handles DB_* vs PG_* fallbacks, Redis, Vector, STT, TTS.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedisConfig:
    """Redis configuration with defaults."""
    host: str
    port: int
    db: int
    password: Optional[str]
    url: Optional[str]
    
    # TTLs
    state_ttl: int  # General state (7 days default)
    session_ttl: int  # Active sessions
    stored_info_ttl: int  # Stored user data (30 days)
    conversation_ttl: int  # Conversation memory


@dataclass
class PostgresConfig:
    """PostgreSQL configuration with DB_*/PG_* fallback."""
    host: str
    port: int
    database: str
    user: str
    password: str
    
    def connection_string(self) -> str:
        """Return psycopg2-compatible connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass  
class VectorConfig:
    """Vector service configuration."""
    base_url: str
    api_key: Optional[str]


@dataclass
class VoiceConfig:
    """STT/TTS configuration."""
    stt_base_url: Optional[str]
    tts_base_url: Optional[str]
    stt_api_key: Optional[str]
    tts_api_key: Optional[str]


class UnifiedConfig:
    """
    Unified configuration loader.
    
    Usage:
        config = UnifiedConfig()
        pg = config.postgres
        redis = config.redis
    """
    
    _instance: Optional['UnifiedConfig'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_all()
    
    def _load_all(self):
        """Load all configurations."""
        self._load_redis()
        self._load_postgres()
        self._load_vector()
        self._load_voice()
        self._load_api_keys()
    
    def _get_env(self, *keys: str, default: str = "") -> str:
        """Get first available env var from list of keys."""
        for key in keys:
            val = os.getenv(key)
            if val:
                return val
        return default
    
    def _load_redis(self):
        """Load Redis configuration."""
        state_ttl = int(self._get_env("STATE_TTL_SECONDS", default="604800"))
        
        self.redis = RedisConfig(
            host=self._get_env("REDIS_HOST", default="localhost"),
            port=int(self._get_env("REDIS_PORT", default="6379")),
            db=int(self._get_env("REDIS_DB", default="0")),
            password=self._get_env("REDIS_PASSWORD") or None,
            url=self._get_env("REDIS_URL") or None,
            state_ttl=state_ttl,
            session_ttl=int(self._get_env("SESSION_TTL_SECONDS", default=str(state_ttl))),
            stored_info_ttl=int(self._get_env("STORED_INFO_TTL_SECONDS", default="2592000")),
            conversation_ttl=int(self._get_env("REDIS_CONVERSATION_TTL", default=str(state_ttl))),
        )
    
    def _load_postgres(self):
        """Load PostgreSQL config with DB_*/PG_* fallback support."""
        self.postgres = PostgresConfig(
            host=self._get_env("PGHOST", "PG_HOST", "DB_HOST", default="localhost"),
            port=int(self._get_env("PGPORT", "PG_PORT", "DB_PORT", default="5432")),
            database=self._get_env("PGDATABASE", "PG_DATABASE", "DB_NAME", default="adventureworks"),
            user=self._get_env("PGUSER", "PG_USER", "DB_USER", default="postgres"),
            password=self._get_env("PGPASSWORD", "PG_PASSWORD", "DB_PASSWORD", default=""),
        )
    
    def _load_vector(self):
        """Load vector service config."""
        self.vector = VectorConfig(
            base_url=self._get_env("VECTOR_BASE_URL", default="http://localhost:8001"),
            api_key=self._get_env("VECTOR_API_KEY") or None,
        )
    
    def _load_voice(self):
        """Load STT/TTS config."""
        self.voice = VoiceConfig(
            stt_base_url=self._get_env("STT_BASE_URL") or None,
            tts_base_url=self._get_env("TTS_BASE_URL") or None,
            stt_api_key=self._get_env("STT_API_KEY") or None,
            tts_api_key=self._get_env("TTS_API_KEY") or None,
        )
    
    def _load_api_keys(self):
        """Load API keys."""
        self.groq_api_key = self._get_env("GROQ_API_KEY")
        self.tavily_api_key = self._get_env("TAVILY_API_KEY")
        self.weather_api_key = self._get_env("WEATHER_API_KEY")
        self.news_api_key = self._get_env("NEWS_API_KEY")
        self.google_cse_api_key = self._get_env("GOOGLE_CSE_API_KEY")
        self.google_cse_cx = self._get_env("GOOGLE_CSE_CX")
    
    def check_redis(self) -> tuple[bool, str]:
        """
        Check Redis connectivity.
        Returns: (is_connected, message)
        """
        try:
            import redis
            r = redis.Redis(
                host=self.redis.host,
                port=self.redis.port,
                db=self.redis.db,
                password=self.redis.password,
                socket_timeout=3
            )
            r.ping()
            return True, f"Redis connected at {self.redis.host}:{self.redis.port}"
        except Exception as e:
            return False, f"Redis connection failed: {e}"
    
    def check_postgres(self) -> tuple[bool, str]:
        """
        Check PostgreSQL connectivity.
        Returns: (is_connected, message)
        """
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.postgres.host,
                port=self.postgres.port,
                database=self.postgres.database,
                user=self.postgres.user,
                password=self.postgres.password,
                connect_timeout=3
            )
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            return True, f"PostgreSQL connected at {self.postgres.host}:{self.postgres.port}/{self.postgres.database}"
        except ImportError:
            return False, "psycopg2 not installed"
        except Exception as e:
            return False, f"PostgreSQL connection failed: {e}"
    
    def check_vector(self) -> tuple[bool, str]:
        """
        Check Vector service connectivity.
        Returns: (is_connected, message)
        """
        try:
            import httpx
            headers = {}
            if self.vector.api_key:
                headers["Authorization"] = f"Bearer {self.vector.api_key}"
            
            with httpx.Client(timeout=3) as client:
                resp = client.get(f"{self.vector.base_url}/health", headers=headers)
                if resp.status_code == 200:
                    return True, f"Vector service connected at {self.vector.base_url}"
                elif resp.status_code == 401:
                    return False, f"Vector service auth failed (401) - check VECTOR_API_KEY"
                else:
                    return False, f"Vector service returned {resp.status_code}"
        except Exception as e:
            return False, f"Vector service unreachable: {e}"


# Singleton accessor
def get_config() -> UnifiedConfig:
    """Get the unified configuration singleton."""
    return UnifiedConfig()
