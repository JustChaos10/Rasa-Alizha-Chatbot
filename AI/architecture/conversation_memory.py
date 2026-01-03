"""
Conversation Memory for LLM Context.

This module manages conversation history so the LLM has full context
of what was discussed. This eliminates the need for hardcoded session
handling - the LLM can naturally understand follow-ups.

Key features:
- Per-user conversation history
- Multiple chat sessions per user (ChatGPT-like)
- Automatic summarization for long conversations
- Tool call history (so LLM knows what tools were used)
- Configurable context window
"""

import json
import redis.asyncio as redis
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
from config.config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def to_context_string(self) -> str:
        """Convert to a context string for the LLM."""
        if self.role == "user":
            return f"User: {self.content}"
        elif self.role == "assistant":
            return f"Assistant: {self.content}"
        elif self.role == "tool":
            tool_name = self.metadata.get("tool_name", "unknown")
            return f"[Tool {tool_name} returned: {self.content[:200]}...]" if len(self.content) > 200 else f"[Tool {tool_name} returned: {self.content}]"
        else:
            return self.content


@dataclass
class ToolCall:
    """Record of a tool that was called."""
    tool_name: str
    params: Dict[str, Any]
    result: Any
    success: bool
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "tool_name": self.tool_name,
            "params": self.params,
            "result": self.result,
            "success": self.success,
            "timestamp": self.timestamp
        }
    
    def to_context_string(self) -> str:
        """Convert to context string."""
        status = "✓" if self.success else "✗"
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"[{status} Called {self.tool_name}({params_str})]"


@dataclass
class Session:
    """
    Represents a single chat session.
    A user can have multiple sessions, like ChatGPT conversations.
    """
    session_id: str
    name: str  # Display name (can be auto-generated or renamed)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0
    preview: str = ""  # First message or summary for display
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "preview": self.preview
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create Session from dict."""
        return cls(
            session_id=data["session_id"],
            name=data["name"],
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            message_count=data.get("message_count", 0),
            preview=data.get("preview", "")
        )


@dataclass 
class ConversationContext:
    """
    Full conversation context for a user session.
    """
    messages: List[Message] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    active_topic: Optional[str] = None  # e.g., "leave_request"
    active_entities: Dict[str, Any] = field(default_factory=dict)  # Extracted entities
    last_tool: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ConversationContext to a dictionary for serialization."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "active_topic": self.active_topic,
            "active_entities": self.active_entities,
            "last_tool": self.last_tool,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create ConversationContext from a dictionary."""
        messages = [Message(**msg_data) for msg_data in data.get("messages", [])]
        tool_calls = [ToolCall(**tc_data) for tc_data in data.get("tool_calls", [])]
        return cls(
            messages=messages,
            tool_calls=tool_calls,
            active_topic=data.get("active_topic"),
            active_entities=data.get("active_entities", {}),
            last_tool=data.get("last_tool"),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
        )
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a user message."""
        self.messages.append(Message(
            role="user",
            content=content,
            metadata=metadata or {}
        ))
        self.last_activity = time.time()
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add an assistant message."""
        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata=metadata or {}
        ))
        self.last_activity = time.time()
    
    def add_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        success: bool
    ) -> None:
        """Record a tool call."""
        self.tool_calls.append(ToolCall(
            tool_name=tool_name,
            params=params,
            result=result,
            success=success
        ))
        self.last_tool = tool_name
        self.last_activity = time.time()
        
        # Also add as a message for context
        result_str = str(result)[:500]  # Truncate for context
        self.messages.append(Message(
            role="tool",
            content=result_str,
            metadata={"tool_name": tool_name, "params": params, "success": success}
        ))
    
    def update_entities(self, entities: Dict[str, Any]) -> None:
        """Update extracted entities (merge with existing)."""
        self.active_entities.update(entities)
        self.last_activity = time.time()
    
    def set_topic(self, topic: str) -> None:
        """Set the active topic."""
        self.active_topic = topic
        self.last_activity = time.time()
    
    def clear_topic(self) -> None:
        """Clear the active topic."""
        self.active_topic = None
        self.active_entities = {}
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:]
    
    def get_context_string(self, max_messages: int = 10) -> str:
        """
        Build a context string for the LLM.
        
        This gives the LLM the conversation history so it can
        understand follow-up questions naturally.
        """
        lines = []
        
        # Add active topic/entities if any
        if self.active_topic:
            lines.append(f"[Active topic: {self.active_topic}]")
        if self.active_entities:
            entities_str = ", ".join(f"{k}: {v}" for k, v in self.active_entities.items())
            lines.append(f"[Known context: {entities_str}]")
        
        # Add recent messages
        recent = self.get_recent_messages(max_messages)
        for msg in recent:
            lines.append(msg.to_context_string())
        
        return "\n".join(lines)
    
    def get_messages_for_llm(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get messages in LLM API format."""
        recent = self.get_recent_messages(max_messages)
        return [msg.to_dict() for msg in recent if msg.role in ("user", "assistant")]
    
    def is_stale(self, max_age_seconds: int = 1800) -> bool:
        """Check if conversation is stale (default 30 min)."""
        return time.time() - self.last_activity > max_age_seconds
    
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)


class ConversationMemory:
    """
    Manages conversation memory for all users.
    
    This is a singleton that tracks conversations by sender_id.
    """
    
    _instance: Optional['ConversationMemory'] = None
    
    def __init__(self, max_conversations: int = 1000, max_messages_per_conversation: int = 50):
        # Use an in-memory cache for the current request cycle to avoid multiple Redis lookups
        self._conversations_cache: Dict[str, ConversationContext] = {}
        self._max_messages = max_messages_per_conversation
        self._max_conversations = max_conversations # Retained for compatibility but Redis handles primary eviction
        
        # Initialize Redis client
        config = ConfigManager()
        self.redis_host = config.redis_host
        self.redis_port = config.redis_port
        self.redis_db = config.redis_db
        self.redis_password = config.redis_password
        self.conversation_ttl = config.redis_conversation_ttl  # TTL in seconds
        
        self.redis_client: Optional[redis.Redis] = None
        self._redis_loop_id: Optional[int] = None  # Track which event loop created the client
        self._redis_available = False  # Track if Redis is available
        
        # In-memory fallback storage (used when Redis is unavailable)
        self._memory_sessions: Dict[str, List[Session]] = {}  # sender_id -> list of sessions
        self._memory_contexts: Dict[str, ConversationContext] = {}  # key -> context
        self._memory_active_session: Dict[str, str] = {}  # sender_id -> active session_id
        
        # Session settings
        self.max_sessions_per_user = 50  # Maximum sessions per user
        
        logger.info(f"ConversationMemory initialized. Redis: {self.redis_host}:{self.redis_port}/{self.redis_db}")
    
    async def _init_redis_client(self):
        """Initialize the async Redis client if not already initialized or if event loop changed."""
        import asyncio
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = None
        
        # Reinitialize if no client, or if the event loop has changed
        need_reinit = (
            self.redis_client is None or 
            self._redis_loop_id != current_loop_id
        )
        
        if need_reinit:
            # Close old client if it exists
            if self.redis_client is not None:
                try:
                    await self.redis_client.close()
                except Exception:
                    pass  # Ignore errors closing old client
                self.redis_client = None
            
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    decode_responses=True
                )
                await self.redis_client.ping()
                self._redis_loop_id = current_loop_id
                self._redis_available = True
                logger.debug(f"Redis client initialized for loop {current_loop_id}")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
                self.redis_client = None
                self._redis_loop_id = None
                self._redis_available = False
    
    @classmethod
    def get_instance(cls) -> 'ConversationMemory':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = ConversationMemory()
        return cls._instance
    
    async def _save_context_to_redis(self, sender_id: str, context: ConversationContext) -> None:
        """Saves a conversation context to Redis."""
        await self._init_redis_client()
        if self.redis_client:
            try:
                # Update last_activity before saving
                context.last_activity = time.time()
                serialized_context = json.dumps(context.to_dict())
                await self.redis_client.set(
                    f"conversation:{sender_id}",
                    serialized_context,
                    ex=self.conversation_ttl # Set expiration
                )
            except Exception as e:
                logger.error(f"Error saving context to Redis for {sender_id}: {e}")
    
    async def _load_context_from_redis(self, sender_id: str) -> Optional[ConversationContext]:
        """Loads a conversation context from Redis."""
        await self._init_redis_client()
        if self.redis_client:
            try:
                serialized_context = await self.redis_client.get(f"conversation:{sender_id}")
                if serialized_context:
                    data = json.loads(serialized_context)
                    context = ConversationContext.from_dict(data)
                    # Check for staleness based on last_activity stored in Redis
                    # We can use the Redis TTL to handle actual expiry,
                    # but this check ensures in-memory cache is fresh.
                    if context.is_stale(self.conversation_ttl): # Use Redis TTL for in-memory staleness check too
                        logger.info(f"Conversation {sender_id} loaded from Redis was stale, resetting.")
                        # Optionally delete from Redis if we decide staleness check here is definitive for expiry
                        # await self.redis_client.delete(f"conversation:{sender_id}")
                        return None
                    return context
            except Exception as e:
                logger.error(f"Error loading context from Redis for {sender_id}: {e}")
        return None
    
    async def get_context(self, sender_id: str) -> ConversationContext:
        """
        Get or create conversation context for a user.
        Now session-aware: automatically uses the active session.
        """
        # Use session-aware method
        return await self.get_session_context(sender_id, session_id=None)
    
    async def clear_context(self, sender_id: str) -> None:
        """Clear conversation context for a user from cache and Redis."""
        if sender_id in self._conversations_cache:
            del self._conversations_cache[sender_id]
        
        await self._init_redis_client()
        if self.redis_client:
            try:
                await self.redis_client.delete(f"conversation:{sender_id}")
                logger.info(f"Cleared conversation context for {sender_id} from Redis.")
            except Exception as e:
                logger.error(f"Error clearing context from Redis for {sender_id}: {e}")
        
    async def add_exchange(
        self,
        sender_id: str,
        user_message: str,
        assistant_response: str,
        tool_name: Optional[str] = None,
        tool_params: Optional[Dict] = None,
        tool_result: Optional[Any] = None,
        entities: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a complete exchange (user message + assistant response).
        Now session-aware: automatically uses the active session.
        """
        # Use session-aware method
        await self.add_session_exchange(
            sender_id=sender_id,
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=None,  # Use active session
            tool_name=tool_name,
            tool_params=tool_params,
            tool_result=tool_result,
            entities=entities,
            topic=topic,
            metadata=metadata
        )
    
    def _trim_if_needed(self, sender_id: str, context: ConversationContext) -> None:
        """Trim conversation if it's too long."""
        if context.message_count() > self._max_messages:
            context.messages = context.messages[-self._max_messages:]
            logger.debug(f"Trimmed conversation for {sender_id} to {self._max_messages} messages")
    
    async def get_context_for_llm(self, sender_id: str, include_system_context: bool = True) -> str:
        """
        Get conversation context formatted for LLM.
        """
        context = await self.get_context(sender_id)
        
        if context.message_count() == 0:
            return ""
        
        parts = []
        
        if include_system_context:
            parts.append("=== Conversation History ===")
        
        parts.append(context.get_context_string())
        
        if include_system_context:
            parts.append("=== Current Message ===")
        
        return "\n".join(parts)
    
    async def get_active_topic(self, sender_id: str) -> Optional[str]:
        """Get the active topic for a user."""
        context = await self.get_context(sender_id)
        return context.active_topic
    
    async def get_active_entities(self, sender_id: str) -> Dict[str, Any]:
        """Get active entities for a user."""
        context = await self.get_context(sender_id)
        return context.active_entities.copy()
    
    async def get_last_tool(self, sender_id: str) -> Optional[str]:
        """Get the last tool that was called."""
        context = await self.get_context(sender_id)
        return context.last_tool

    # =============================================
    # SESSION MANAGEMENT METHODS
    # =============================================
    
    def _session_key(self, sender_id: str, session_id: str) -> str:
        """Generate Redis key for a specific session's conversation."""
        return f"conversation:{sender_id}:{session_id}"
    
    def _sessions_index_key(self, sender_id: str) -> str:
        """Generate Redis key for user's session index."""
        return f"sessions:{sender_id}"
    
    def _active_session_key(self, sender_id: str) -> str:
        """Generate Redis key for user's active session pointer."""
        return f"active_session:{sender_id}"
    
    async def _get_active_session_id(self, sender_id: str) -> Optional[str]:
        """Get the currently active session ID for a user."""
        await self._init_redis_client()
        
        # Use in-memory fallback if Redis unavailable
        if not self._redis_available:
            return self._memory_active_session.get(sender_id)
        
        if self.redis_client:
            try:
                session_id = await self.redis_client.get(self._active_session_key(sender_id))
                return session_id
            except Exception as e:
                logger.error(f"Error getting active session for {sender_id}: {e}")
        return None
    
    async def _set_active_session_id(self, sender_id: str, session_id: str) -> None:
        """Set the currently active session ID for a user."""
        await self._init_redis_client()
        
        # Use in-memory fallback if Redis unavailable
        if not self._redis_available:
            self._memory_active_session[sender_id] = session_id
            return
        
        if self.redis_client:
            try:
                await self.redis_client.set(
                    self._active_session_key(sender_id),
                    session_id,
                    ex=self.conversation_ttl
                )
            except Exception as e:
                logger.error(f"Error setting active session for {sender_id}: {e}")
    
    async def _generate_session_name(self, first_message: str) -> str:
        """Generate a session name from the first message."""
        # Take first 40 chars, clean up
        name = first_message.strip()[:40]
        if len(first_message) > 40:
            name += "..."
        return name if name else "New Chat"
    
    async def create_session(self, sender_id: str, name: Optional[str] = None) -> Session:
        """
        Create a new chat session for a user.
        Returns the created Session.
        """
        await self._init_redis_client()
        
        session_id = str(uuid.uuid4())[:8]  # Short unique ID
        session = Session(
            session_id=session_id,
            name=name or "New Chat",
            created_at=time.time(),
            last_activity=time.time(),
            message_count=0,
            preview=""
        )
        
        # Add to session index
        await self._add_session_to_index(sender_id, session)
        
        # Set as active session
        await self._set_active_session_id(sender_id, session_id)
        
        # Create empty conversation context for this session
        context = ConversationContext()
        await self._save_session_context(sender_id, session_id, context)
        
        logger.info(f"Created new session {session_id} for user {sender_id}")
        return session
    
    async def _add_session_to_index(self, sender_id: str, session: Session) -> None:
        """Add a session to the user's session index."""
        await self._init_redis_client()
        
        # Use in-memory fallback if Redis unavailable
        if not self._redis_available:
            if sender_id not in self._memory_sessions:
                self._memory_sessions[sender_id] = []
            self._memory_sessions[sender_id].insert(0, session)
            # Trim if too many
            if len(self._memory_sessions[sender_id]) > self.max_sessions_per_user:
                self._memory_sessions[sender_id] = self._memory_sessions[sender_id][:self.max_sessions_per_user]
            return
        
        if not self.redis_client:
            return
            
        try:
            # Get existing sessions
            sessions = await self.list_sessions(sender_id)
            
            # Add new session at the beginning (most recent first)
            sessions.insert(0, session)
            
            # Trim if too many sessions
            if len(sessions) > self.max_sessions_per_user:
                old_sessions = sessions[self.max_sessions_per_user:]
                sessions = sessions[:self.max_sessions_per_user]
                # Delete old session data
                for old in old_sessions:
                    await self._delete_session_data(sender_id, old.session_id)
            
            # Save updated index
            sessions_data = [s.to_dict() for s in sessions]
            await self.redis_client.set(
                self._sessions_index_key(sender_id),
                json.dumps(sessions_data),
                ex=self.conversation_ttl * 2  # Longer TTL for index
            )
        except Exception as e:
            logger.error(f"Error adding session to index for {sender_id}: {e}")
    
    async def _update_session_in_index(self, sender_id: str, session: Session) -> None:
        """Update a session's metadata in the index."""
        await self._init_redis_client()
        if not self.redis_client:
            return
            
        try:
            sessions = await self.list_sessions(sender_id)
            for i, s in enumerate(sessions):
                if s.session_id == session.session_id:
                    sessions[i] = session
                    break
            
            sessions_data = [s.to_dict() for s in sessions]
            await self.redis_client.set(
                self._sessions_index_key(sender_id),
                json.dumps(sessions_data),
                ex=self.conversation_ttl * 2
            )
        except Exception as e:
            logger.error(f"Error updating session in index: {e}")
    
    async def list_sessions(self, sender_id: str) -> List[Session]:
        """
        List all chat sessions for a user, sorted by last activity (most recent first).
        """
        await self._init_redis_client()
        
        # Use in-memory fallback if Redis unavailable
        if not self._redis_available:
            sessions = self._memory_sessions.get(sender_id, [])
            sessions.sort(key=lambda s: s.last_activity, reverse=True)
            return sessions
            
        try:
            data = await self.redis_client.get(self._sessions_index_key(sender_id))
            if data:
                sessions_data = json.loads(data)
                sessions = [Session.from_dict(s) for s in sessions_data]
                # Sort by last activity, most recent first
                sessions.sort(key=lambda s: s.last_activity, reverse=True)
                return sessions
        except Exception as e:
            logger.error(f"Error listing sessions for {sender_id}: {e}")
        return []
    
    async def get_session(self, sender_id: str, session_id: str) -> Optional[Session]:
        """Get a specific session by ID."""
        sessions = await self.list_sessions(sender_id)
        for session in sessions:
            if session.session_id == session_id:
                return session
        return None
    
    async def rename_session(self, sender_id: str, session_id: str, new_name: str) -> bool:
        """Rename a chat session."""
        session = await self.get_session(sender_id, session_id)
        if session:
            session.name = new_name
            await self._update_session_in_index(sender_id, session)
            logger.info(f"Renamed session {session_id} to '{new_name}'")
            return True
        return False
    
    async def delete_session(self, sender_id: str, session_id: str) -> bool:
        """
        Delete a chat session and its conversation data.
        If it's the active session, switches to another session or creates new one.
        """
        await self._init_redis_client()
        if not self.redis_client:
            return False
            
        try:
            # Remove from index
            sessions = await self.list_sessions(sender_id)
            sessions = [s for s in sessions if s.session_id != session_id]
            
            sessions_data = [s.to_dict() for s in sessions]
            await self.redis_client.set(
                self._sessions_index_key(sender_id),
                json.dumps(sessions_data),
                ex=self.conversation_ttl * 2
            )
            
            # Delete conversation data
            await self._delete_session_data(sender_id, session_id)
            
            # If this was the active session, switch to another or create new
            active_id = await self._get_active_session_id(sender_id)
            if active_id == session_id:
                if sessions:
                    await self._set_active_session_id(sender_id, sessions[0].session_id)
                else:
                    # Create a new session
                    await self.create_session(sender_id)
            
            logger.info(f"Deleted session {session_id} for user {sender_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def _delete_session_data(self, sender_id: str, session_id: str) -> None:
        """Delete the conversation data for a session."""
        if self.redis_client:
            try:
                await self.redis_client.delete(self._session_key(sender_id, session_id))
            except Exception as e:
                logger.error(f"Error deleting session data: {e}")
    
    async def switch_session(self, sender_id: str, session_id: str) -> bool:
        """
        Switch to a different chat session.
        Returns True if successful.
        """
        # Verify session exists
        session = await self.get_session(sender_id, session_id)
        if session:
            await self._set_active_session_id(sender_id, session_id)
            # Clear in-memory cache to force reload from Redis
            if sender_id in self._conversations_cache:
                del self._conversations_cache[sender_id]
            logger.info(f"Switched user {sender_id} to session {session_id}")
            return True
        return False
    
    async def get_or_create_active_session(self, sender_id: str) -> Session:
        """
        Get the active session for a user, or create one if none exists.
        This is the main entry point for session-aware conversation.
        """
        session_id = await self._get_active_session_id(sender_id)
        
        if session_id:
            session = await self.get_session(sender_id, session_id)
            if session:
                return session
        
        # No active session, check if user has any sessions
        sessions = await self.list_sessions(sender_id)
        if sessions:
            # Use the most recent session
            session = sessions[0]
            await self._set_active_session_id(sender_id, session.session_id)
            return session
        
        # No sessions at all, create first one
        return await self.create_session(sender_id)
    
    async def _save_session_context(self, sender_id: str, session_id: str, context: ConversationContext) -> None:
        """Save conversation context for a specific session."""
        await self._init_redis_client()
        cache_key = f"{sender_id}:{session_id}"
        context.last_activity = time.time()
        
        # Always save to in-memory cache
        self._memory_contexts[cache_key] = context
        
        # Try to save to Redis if available
        if self._redis_available and self.redis_client:
            try:
                serialized_context = json.dumps(context.to_dict())
                await self.redis_client.set(
                    self._session_key(sender_id, session_id),
                    serialized_context,
                    ex=self.conversation_ttl
                )
            except Exception as e:
                logger.error(f"Error saving session context: {e}")
    
    async def _load_session_context(self, sender_id: str, session_id: str) -> Optional[ConversationContext]:
        """Load conversation context for a specific session."""
        cache_key = f"{sender_id}:{session_id}"
        
        # Check in-memory cache first
        if cache_key in self._memory_contexts:
            return self._memory_contexts[cache_key]
        
        await self._init_redis_client()
        
        # Try Redis if available
        if self._redis_available and self.redis_client:
            try:
                data = await self.redis_client.get(self._session_key(sender_id, session_id))
                if data:
                    return ConversationContext.from_dict(json.loads(data))
            except Exception as e:
                logger.error(f"Error loading session context: {e}")
        return None
    
    async def get_session_context(self, sender_id: str, session_id: Optional[str] = None) -> ConversationContext:
        """
        Get conversation context for a specific session.
        If no session_id provided, uses the active session.
        """
        if not session_id:
            session = await self.get_or_create_active_session(sender_id)
            session_id = session.session_id
        
        # Check in-memory cache
        cache_key = f"{sender_id}:{session_id}"
        if cache_key in self._conversations_cache:
            return self._conversations_cache[cache_key]
        
        context = await self._load_session_context(sender_id, session_id)
        if context is None:
            context = ConversationContext()
            await self._save_session_context(sender_id, session_id, context)
        
        self._conversations_cache[cache_key] = context
        return context
    
    async def add_session_exchange(
        self,
        sender_id: str,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_params: Optional[Dict] = None,
        tool_result: Optional[Any] = None,
        entities: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a complete exchange in a specific session.
        Updates session metadata (message count, preview, last activity).
        """
        # Get or create active session
        if not session_id:
            session = await self.get_or_create_active_session(sender_id)
            session_id = session.session_id
        else:
            session = await self.get_session(sender_id, session_id)
        
        if not session:
            logger.error(f"Session {session_id} not found for {sender_id}")
            return
        
        # Get conversation context
        context = await self.get_session_context(sender_id, session_id)
        
        context.add_user_message(user_message)
        
        if tool_name:
            context.add_tool_call(
                tool_name=tool_name,
                params=tool_params or {},
                result=tool_result,
                success=True
            )
        
        context.add_assistant_message(assistant_response, metadata=metadata)
        
        if entities:
            context.update_entities(entities)
        
        if topic:
            context.set_topic(topic)
        
        # Trim if needed
        cache_key = f"{sender_id}:{session_id}"
        self._trim_if_needed(cache_key, context)
        
        # Save context
        await self._save_session_context(sender_id, session_id, context)
        
        # Update session metadata
        session.message_count = context.message_count()
        session.last_activity = time.time()
        
        # Update session name from first message if still "New Chat"
        if session.name == "New Chat" and user_message:
            session.name = await self._generate_session_name(user_message)
        
        # Update preview with last user message
        session.preview = user_message[:50] + "..." if len(user_message) > 50 else user_message
        
        await self._update_session_in_index(sender_id, session)


# Convenience function

async def get_conversation_memory() -> ConversationMemory:
    """
    Get a new instance of ConversationMemory for the current request cycle.
    This ensures a clean in-memory cache for each request.
    """
    memory = ConversationMemory()
    await memory._init_redis_client() # Ensure Redis client is initialized
    return memory
