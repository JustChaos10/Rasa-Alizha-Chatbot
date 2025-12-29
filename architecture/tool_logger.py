"""
Tool Logger - Comprehensive logging for MCP servers and plugins.

This module provides a centralized logging system that clearly shows:
- Which tool/plugin/MCP server was called
- The input parameters
- The execution time
- The result (success/failure)
- Session stickiness state

Usage:
    from architecture.tool_logger import ToolLogger
    
    logger = ToolLogger.get_logger()
    
    with logger.tool_call("leave.get_leave_request_card", params):
        result = await execute_tool(...)
        logger.log_result(result)
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Tool types
    MCP_SERVER = "\033[38;5;39m"      # Blue
    PLUGIN = "\033[38;5;208m"          # Orange
    RASA = "\033[38;5;40m"             # Green
    LLM = "\033[38;5;141m"             # Purple
    
    # Status
    SUCCESS = "\033[38;5;40m"          # Green
    ERROR = "\033[38;5;196m"           # Red
    WARNING = "\033[38;5;220m"         # Yellow
    INFO = "\033[38;5;45m"             # Cyan
    
    # Session
    SESSION_START = "\033[38;5;51m"    # Light cyan
    SESSION_CONTINUE = "\033[38;5;219m" # Pink
    SESSION_END = "\033[38;5;245m"     # Gray


class ToolType(Enum):
    """Type of tool being called."""
    MCP_SERVER = "mcp_server"
    PLUGIN = "plugin"
    RASA = "rasa"
    CODE_EXECUTION = "code_execution"
    FORM_SUBMISSION = "form_submission"
    LLM = "llm"


@dataclass
class ToolCallLog:
    """Record of a single tool call."""
    tool_name: str
    tool_type: ToolType
    server_name: Optional[str]  # For MCP servers
    params: Dict[str, Any]
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    error: Optional[str] = None
    result_type: Optional[str] = None  # e.g., "adaptive_card", "text", "error"
    session_id: Optional[str] = None
    is_sticky: bool = False
    turn_number: int = 0


@dataclass
class SessionLog:
    """Record of a sticky session."""
    session_id: str
    tool_name: str
    started_at: datetime
    last_activity: datetime
    turn_count: int = 0
    is_active: bool = True


class ToolLogger:
    """
    Comprehensive logger for tool/plugin/MCP server calls.
    
    Features:
    - Color-coded output for different tool types
    - Timing information
    - Session tracking for sticky tools
    - Structured logging to file
    - Clear indication of MCP server vs plugin calls
    """
    
    _instance: Optional['ToolLogger'] = None
    
    def __init__(self, log_file: str = "tool_calls.log"):
        self.log_file = log_file
        self._setup_loggers()
        
        # Call history
        self.call_history: List[ToolCallLog] = []
        self.max_history = 1000
        
        # Active sessions
        self.active_sessions: Dict[str, SessionLog] = {}
        
        # Current call context (for nested logging)
        self._current_call: Optional[ToolCallLog] = None
        
    def _setup_loggers(self):
        """Set up console and file loggers."""
        # Console logger - only shows ROUTING (uses WARNING level for routing only)
        self.console_logger = logging.getLogger("tool_console")
        self.console_logger.setLevel(logging.WARNING)  # Only WARNING and above
        self.console_logger.propagate = False  # Don't propagate to root logger
        self.console_logger.handlers = []
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.console_logger.addHandler(console_handler)
        
        # File logger for structured logs (logs everything)
        self.file_logger = logging.getLogger("tool_file")
        self.file_logger.setLevel(logging.DEBUG)
        self.file_logger.propagate = False  # Don't propagate to root logger
        self.file_logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.file_logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls) -> 'ToolLogger':
        """Get the singleton logger instance."""
        if cls._instance is None:
            cls._instance = ToolLogger()
        return cls._instance
    
    def _get_color_for_type(self, tool_type: ToolType) -> str:
        """Get the color code for a tool type."""
        colors = {
            ToolType.MCP_SERVER: Colors.MCP_SERVER,
            ToolType.PLUGIN: Colors.PLUGIN,
            ToolType.RASA: Colors.RASA,
            ToolType.LLM: Colors.LLM,
            ToolType.FORM_SUBMISSION: Colors.INFO,
            ToolType.CODE_EXECUTION: Colors.LLM,  # Purple - same as LLM
        }
        return colors.get(tool_type, Colors.RESET)
    
    def _get_icon_for_type(self, tool_type: ToolType) -> str:
        """Get the icon for a tool type."""
        icons = {
            ToolType.MCP_SERVER: "🔌",
            ToolType.PLUGIN: "🔧",
            ToolType.RASA: "🤖",
            ToolType.LLM: "🧠",
            ToolType.FORM_SUBMISSION: "📝",
            ToolType.CODE_EXECUTION: "🐍",  # Python snake
        }
        return icons.get(tool_type, "❓")
    
    def _format_params(self, params: Dict[str, Any], max_length: int = 100) -> str:
        """Format parameters for display."""
        if not params:
            return "{}"
        
        try:
            formatted = json.dumps(params, default=str)
            if len(formatted) > max_length:
                return formatted[:max_length] + "..."
            return formatted
        except Exception:
            return str(params)[:max_length]
    
    def log_routing_decision(
        self,
        message: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        selected_tool: Optional[str] = None,
        tool_type: Optional[ToolType] = None,
        reason: str = ""
    ):
        """Log a routing decision."""
        icon = "🎯"
        color = Colors.INFO
        
        parts = [f"{color}{icon} ROUTING{Colors.RESET}"]
        
        # Message preview
        msg_preview = message[:50] + "..." if len(message) > 50 else message
        parts.append(f"  Message: \"{msg_preview}\"")
        
        # Intent info
        if intent:
            conf_str = f" ({confidence:.2f})" if confidence else ""
            parts.append(f"  Intent: {intent}{conf_str}")
        
        # Tool selection
        if selected_tool and tool_type:
            tool_color = self._get_color_for_type(tool_type)
            tool_icon = self._get_icon_for_type(tool_type)
            parts.append(f"  → {tool_color}{tool_icon} {tool_type.value.upper()}: {selected_tool}{Colors.RESET}")
        
        if reason:
            parts.append(f"  Reason: {reason}")
        
        # Use WARNING level so routing shows on console (console only shows WARNING+)
        self.console_logger.warning("\n".join(parts))
        self.file_logger.info(f"ROUTING | message=\"{msg_preview}\" | intent={intent} | confidence={confidence} | tool={selected_tool} | type={tool_type}")
    
    @contextmanager
    def tool_call(
        self,
        tool_name: str,
        tool_type: ToolType,
        params: Dict[str, Any] = None,
        server_name: Optional[str] = None,
        session_id: Optional[str] = None,
        is_sticky: bool = False,
        turn_number: int = 0
    ):
        """
        Context manager for logging a tool call.
        
        Usage:
            with logger.tool_call("leave.get_leave_request_card", ToolType.MCP_SERVER, params) as log:
                result = await execute_tool(...)
                log.success = True
                log.result_type = "adaptive_card"
        """
        params = params or {}
        
        # Create log entry
        log = ToolCallLog(
            tool_name=tool_name,
            tool_type=tool_type,
            server_name=server_name,
            params=params,
            started_at=datetime.now(),
            session_id=session_id,
            is_sticky=is_sticky,
            turn_number=turn_number
        )
        
        self._current_call = log
        
        # Log the call start
        color = self._get_color_for_type(tool_type)
        icon = self._get_icon_for_type(tool_type)
        
        type_label = tool_type.value.upper()
        if server_name:
            type_label = f"{type_label} ({server_name})"
        
        session_info = ""
        if is_sticky and session_id:
            session_info = f" {Colors.SESSION_CONTINUE}[Session: turn {turn_number}]{Colors.RESET}"
        
        self.console_logger.info(
            f"\n{color}{'─' * 60}{Colors.RESET}\n"
            f"{color}{icon} {type_label}: {tool_name}{Colors.RESET}{session_info}\n"
            f"   Params: {self._format_params(params)}"
        )
        
        try:
            yield log
        except Exception as e:
            log.error = str(e)
            log.success = False
            raise
        finally:
            log.ended_at = datetime.now()
            log.duration_ms = (log.ended_at - log.started_at).total_seconds() * 1000
            
            # Log the result
            if log.success:
                status_color = Colors.SUCCESS
                status_icon = "✅"
            elif log.error:
                status_color = Colors.ERROR
                status_icon = "❌"
            else:
                status_color = Colors.WARNING
                status_icon = "⚠️"
            
            result_info = ""
            if log.result_type:
                result_info = f" → {log.result_type}"
            
            self.console_logger.info(
                f"   {status_color}{status_icon} Result: {'Success' if log.success else 'Failed'}{result_info} ({log.duration_ms:.1f}ms){Colors.RESET}"
            )
            
            if log.error:
                self.console_logger.info(f"   Error: {log.error}")
            
            self.console_logger.info(f"{color}{'─' * 60}{Colors.RESET}")
            
            # File log
            self.file_logger.info(
                f"TOOL_CALL | tool={tool_name} | type={tool_type.value} | server={server_name} | "
                f"success={log.success} | duration_ms={log.duration_ms:.1f} | "
                f"result_type={log.result_type} | error={log.error}"
            )
            
            # Add to history
            self.call_history.append(log)
            if len(self.call_history) > self.max_history:
                self.call_history.pop(0)
            
            self._current_call = None
    
    def log_session_start(self, session_id: str, tool_name: str):
        """Log the start of a sticky session."""
        session = SessionLog(
            session_id=session_id,
            tool_name=tool_name,
            started_at=datetime.now(),
            last_activity=datetime.now()
        )
        self.active_sessions[session_id] = session
        
        self.console_logger.info(
            f"\n{Colors.SESSION_START}🔗 SESSION START: {tool_name}{Colors.RESET}\n"
            f"   Session ID: {session_id}\n"
            f"   This tool will handle follow-up messages"
        )
        
        self.file_logger.info(f"SESSION_START | session_id={session_id} | tool={tool_name}")
    
    def log_session_continue(self, session_id: str, turn_count: int, message_preview: str):
        """Log a continuation in a sticky session."""
        session = self.active_sessions.get(session_id)
        if session:
            session.turn_count = turn_count
            session.last_activity = datetime.now()
        
        self.console_logger.info(
            f"\n{Colors.SESSION_CONTINUE}🔄 SESSION CONTINUE: Turn {turn_count}{Colors.RESET}\n"
            f"   Message: \"{message_preview[:50]}...\""
        )
        
        self.file_logger.info(f"SESSION_CONTINUE | session_id={session_id} | turn={turn_count}")
    
    def log_session_end(self, session_id: str, reason: str = ""):
        """Log the end of a sticky session."""
        session = self.active_sessions.pop(session_id, None)
        
        duration_info = ""
        if session:
            duration = (datetime.now() - session.started_at).total_seconds()
            duration_info = f" | Duration: {duration:.1f}s | Turns: {session.turn_count}"
        
        self.console_logger.info(
            f"\n{Colors.SESSION_END}🔓 SESSION END{duration_info}{Colors.RESET}\n"
            f"   Reason: {reason or 'completed'}"
        )
        
        self.file_logger.info(f"SESSION_END | session_id={session_id} | reason={reason}")
    
    def log_entity_extraction(self, query: str, entities: Dict[str, Any]):
        """Log LLM entity extraction results."""
        self.console_logger.info(
            f"\n{Colors.LLM}🧠 ENTITY EXTRACTION{Colors.RESET}\n"
            f"   Query: \"{query[:80]}...\"\n"
            f"   Entities: {json.dumps(entities, default=str, indent=6)}"
        )
        
        self.file_logger.info(f"ENTITY_EXTRACTION | query=\"{query[:50]}\" | entities={json.dumps(entities, default=str)}")
    
    def log_llm_call(self, purpose: str, model: str, prompt_preview: str = ""):
        """Log an LLM API call."""
        self.console_logger.info(
            f"\n{Colors.LLM}🧠 LLM CALL: {purpose}{Colors.RESET}\n"
            f"   Model: {model}"
        )
        
        self.file_logger.info(f"LLM_CALL | purpose={purpose} | model={model}")
    
    def log_mcp_connection(self, server_name: str, tool_count: int, success: bool):
        """Log MCP server connection."""
        if success:
            self.console_logger.info(
                f"\n{Colors.MCP_SERVER}🔌 MCP CONNECTED: {server_name}{Colors.RESET}\n"
                f"   Tools available: {tool_count}"
            )
        else:
            self.console_logger.info(
                f"\n{Colors.ERROR}❌ MCP CONNECTION FAILED: {server_name}{Colors.RESET}"
            )
        
        self.file_logger.info(f"MCP_CONNECTION | server={server_name} | tools={tool_count} | success={success}")
    
    def log_adaptive_card(self, card_type: str, template: str = ""):
        """Log adaptive card generation."""
        self.console_logger.info(
            f"   {Colors.INFO}🎴 Adaptive Card: {card_type}{Colors.RESET}"
        )
        
        self.file_logger.info(f"ADAPTIVE_CARD | type={card_type} | template={template}")
    
    def get_recent_calls(self, count: int = 10) -> List[ToolCallLog]:
        """Get the most recent tool calls."""
        return self.call_history[-count:]
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get statistics about tool calls."""
        if not self.call_history:
            return {"total_calls": 0}
        
        by_type = {}
        by_tool = {}
        total_duration = 0
        success_count = 0
        
        for call in self.call_history:
            # By type
            type_key = call.tool_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # By tool
            by_tool[call.tool_name] = by_tool.get(call.tool_name, 0) + 1
            
            # Stats
            total_duration += call.duration_ms
            if call.success:
                success_count += 1
        
        return {
            "total_calls": len(self.call_history),
            "success_rate": success_count / len(self.call_history) * 100,
            "avg_duration_ms": total_duration / len(self.call_history),
            "by_type": by_type,
            "by_tool": by_tool,
            "active_sessions": len(self.active_sessions)
        }


# Global instance
_tool_logger: Optional[ToolLogger] = None


def get_tool_logger() -> ToolLogger:
    """Get the global tool logger instance."""
    global _tool_logger
    if _tool_logger is None:
        _tool_logger = ToolLogger()
    return _tool_logger
