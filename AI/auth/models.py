from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Upload quota tracking
    total_upload_bytes = db.Column(db.BigInteger, default=0)
    last_upload_reset = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to chat sessions
    chat_sessions = db.relationship('ChatSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def has_role(self, role):
        """Check if user has specific role"""
        return self.role == role
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    
    def __repr__(self):
        return f'<User {self.email}>'


class ChatSession(db.Model):
    """
    Represents a chat session/conversation.
    Similar to ChatGPT conversations - users can have multiple sessions.
    """
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    name = db.Column(db.String(255), default='New Chat')
    preview = db.Column(db.String(500), default='')  # First message preview
    is_active = db.Column(db.Boolean, default=True)  # Currently active session
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to messages
    messages = db.relationship('ChatMessage', backref='session', lazy='dynamic', 
                               cascade='all, delete-orphan', order_by='ChatMessage.created_at')
    
    @property
    def message_count(self):
        """Get total message count for this session."""
        return self.messages.count()
    
    @property
    def last_activity(self):
        """Get timestamp of last activity."""
        return self.updated_at.timestamp() if self.updated_at else self.created_at.timestamp()
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'session_id': self.session_id,
            'name': self.name,
            'preview': self.preview or '',
            'message_count': self.message_count,
            'is_active': self.is_active,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'last_activity': self.last_activity,
            'updated_at': self.updated_at.timestamp() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<ChatSession {self.session_id}>'


class ChatMessage(db.Model):
    """
    Individual chat message within a session.
    Stores the full conversation history for replay.
    """
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # 'user', 'assistant', 'system', 'tool'
    content = db.Column(db.Text, nullable=False)
    
    # Store additional metadata as JSON
    metadata_json = db.Column(db.Text, default='{}')
    
    # Tool call information (if this is a tool response)
    tool_name = db.Column(db.String(100), nullable=True)
    tool_params_json = db.Column(db.Text, nullable=True)
    tool_success = db.Column(db.Boolean, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    @property
    def msg_metadata(self):
        """Parse metadata JSON."""
        try:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        except:
            return {}
    
    @msg_metadata.setter
    def msg_metadata(self, value):
        """Set metadata as JSON."""
        self.metadata_json = json.dumps(value) if value else '{}'
    
    @property
    def tool_params(self):
        """Parse tool params JSON."""
        try:
            return json.loads(self.tool_params_json) if self.tool_params_json else None
        except:
            return None
    
    @tool_params.setter
    def tool_params(self, value):
        """Set tool params as JSON."""
        self.tool_params_json = json.dumps(value) if value else None
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'metadata': self.msg_metadata,
            'tool_name': self.tool_name,
            'tool_params': self.tool_params,
            'tool_success': self.tool_success,
            'timestamp': self.created_at.timestamp() if self.created_at else None
        }
    
    def to_llm_format(self):
        """Convert to LLM API format (role + content only)."""
        # Include metadata for frontend rendering (e.g. adaptive cards)
        return {
            'role': self.role,
            'content': self.content,
            'metadata': self.msg_metadata
        }
    
    def __repr__(self):
        return f'<ChatMessage {self.id} ({self.role})>'