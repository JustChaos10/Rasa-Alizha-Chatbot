"""
Form Handler Registry for MCP Servers.

This module provides a scalable way for MCP servers to register their own form handlers.
Instead of hardcoding form handling logic in hybrid_router.py, each MCP server can:
1. Register form submission handlers
2. Register follow-up pattern matchers
3. Define their own session context requirements

This keeps the hybrid_router clean and makes adding new MCP servers simple.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class FormSubmission:
    """Data class for form submission parameters."""
    action: str  # e.g., "/submit_leave_form"
    data: Dict[str, Any]
    context: Dict[str, Any]
    sender_id: str


@dataclass
class FormHandlerResult:
    """Result from a form handler."""
    success: bool
    tool_name: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    response: str
    card: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for routing."""
        output = {
            "tool": self.tool_name,
            "params": self.params,
            "result": self.result,
            "response": self.response,
            "routing": {"handler": "form_submission"}
        }
        if self.card:
            output["type"] = "card"
            output["payload"] = self.card
            output["result"]["type"] = "adaptive_card"
            output["result"]["card"] = self.card
        return output


class FormHandler(ABC):
    """
    Abstract base class for form handlers.
    
    MCP servers implement this to handle their specific form submissions.
    """
    
    @property
    @abstractmethod
    def action(self) -> str:
        """The form action this handler responds to (e.g., '/submit_leave_form')."""
        pass
    
    @property
    @abstractmethod
    def server_name(self) -> str:
        """The MCP server name this handler belongs to."""
        pass
    
    @abstractmethod
    async def handle(
        self,
        submission: FormSubmission,
        execute_tool: Callable
    ) -> FormHandlerResult:
        """
        Handle the form submission.
        
        Args:
            submission: The form submission data
            execute_tool: Callback to execute MCP tools - signature: async (tool_name, params) -> result
        
        Returns:
            FormHandlerResult with the response
        """
        pass


@dataclass
class FollowUpPattern:
    """Pattern for detecting follow-up messages in a conversation."""
    server_name: str
    patterns: List[Pattern] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def matches(self, message: str) -> bool:
        """Check if message matches any pattern or keyword."""
        msg_lower = message.lower()
        
        # Check regex patterns
        for pattern in self.patterns:
            if pattern.search(msg_lower):
                return True
        
        # Check keywords
        return any(kw in msg_lower for kw in self.keywords)


class FormHandlerRegistry:
    """
    Global registry for form handlers.
    
    MCP servers register their handlers here. The hybrid router queries
    this registry instead of having hardcoded handlers.
    """
    
    _instance: Optional['FormHandlerRegistry'] = None
    
    def __init__(self):
        self._handlers: Dict[str, FormHandler] = {}  # action -> handler
        self._followup_patterns: Dict[str, FollowUpPattern] = {}  # server_name -> pattern
        self._server_handlers: Dict[str, List[str]] = {}  # server_name -> [actions]
    
    @classmethod
    def get_instance(cls) -> 'FormHandlerRegistry':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = FormHandlerRegistry()
        return cls._instance
    
    def register_handler(self, handler: FormHandler) -> None:
        """Register a form handler."""
        self._handlers[handler.action] = handler
        
        # Track which handlers belong to which server
        if handler.server_name not in self._server_handlers:
            self._server_handlers[handler.server_name] = []
        self._server_handlers[handler.server_name].append(handler.action)
        
        logger.info(f"📋 Registered form handler: {handler.action} -> {handler.server_name}")
    
    def register_followup_pattern(self, pattern: FollowUpPattern) -> None:
        """Register follow-up detection patterns for a server."""
        self._followup_patterns[pattern.server_name] = pattern
        logger.info(f"🔄 Registered follow-up patterns for: {pattern.server_name}")
    
    def get_handler(self, action: str) -> Optional[FormHandler]:
        """Get the handler for a form action."""
        return self._handlers.get(action)
    
    def has_handler(self, action: str) -> bool:
        """Check if a handler exists for an action."""
        return action in self._handlers
    
    def get_followup_pattern(self, server_name: str) -> Optional[FollowUpPattern]:
        """Get follow-up patterns for a server."""
        return self._followup_patterns.get(server_name)
    
    def list_handlers(self) -> List[str]:
        """List all registered form actions."""
        return list(self._handlers.keys())
    
    def list_servers(self) -> List[str]:
        """List all servers with registered handlers."""
        return list(self._server_handlers.keys())


# ==================== Built-in Handlers ====================
# These can be moved to their respective MCP server files

class LeaveFormHandler(FormHandler):
    """Handler for leave form submissions."""
    
    @property
    def action(self) -> str:
        return "/submit_leave_form"
    
    @property
    def server_name(self) -> str:
        return "leave"
    
    async def handle(
        self,
        submission: FormSubmission,
        execute_tool: Callable
    ) -> FormHandlerResult:
        """Handle leave form submission by calling validate_leave."""
        
        # Extract form data - check both top-level and nested 'data'
        context = submission.context
        data = context.get("data", context)
        
        employee_id = data.get("employee_id", context.get("employee_id", "1"))
        start_date = data.get("start_date", context.get("start_date", ""))
        end_date = data.get("end_date", context.get("end_date", ""))
        leave_type = data.get("leave_type", context.get("leave_type", ""))
        
        logger.info(f"📅 Leave form: {leave_type} from {start_date} to {end_date} for employee {employee_id}")
        
        # Validate required fields
        if not all([start_date, end_date, leave_type]):
            return FormHandlerResult(
                success=False,
                tool_name="leave.validate_leave",
                params={},
                result={"success": False, "error": "Missing required fields"},
                response="Please fill in all required fields: start date, end date, and leave type."
            )
        
        # Parse employee ID
        try:
            emp_id = int(employee_id) if employee_id else 1
        except (ValueError, TypeError):
            emp_id = 1
        
        params = {
            "employee_id": emp_id,
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type
        }
        
        # Call the MCP tool
        result = await execute_tool("leave.validate_leave", params)
        
        if result.get("success"):
            data = result.get("data", {})
            
            # Handle string data
            if isinstance(data, str):
                import json
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass
            
            if isinstance(data, dict) and data.get("type") == "adaptive_card":
                return FormHandlerResult(
                    success=True,
                    tool_name="leave.validate_leave",
                    params=params,
                    result=data,
                    response=data.get("message", "Leave request processed."),
                    card=data.get("card")
                )
            else:
                return FormHandlerResult(
                    success=True,
                    tool_name="leave.validate_leave",
                    params=params,
                    result=result,
                    response=data.get("message", str(data)) if isinstance(data, dict) else str(data)
                )
        else:
            return FormHandlerResult(
                success=False,
                tool_name="leave.validate_leave",
                params=params,
                result=result,
                response=f"❌ Error processing leave request: {result.get('error', 'Unknown error')}"
            )


class ContactFormHandler(FormHandler):
    """Handler for contact info form submissions."""
    
    @property
    def action(self) -> str:
        return "/submit_contact_info"
    
    @property
    def server_name(self) -> str:
        return "contact"
    
    async def handle(
        self,
        submission: FormSubmission,
        execute_tool: Callable
    ) -> FormHandlerResult:
        """Handle contact form submission."""
        data = submission.context.get("data", submission.context)
        person_name = data.get("person_name", "")
        phone_number = data.get("phone_number", "")
        address = data.get("address", "")
        
        logger.info(f"📞 Contact form: {person_name}, {phone_number}")
        
        if person_name:
            return FormHandlerResult(
                success=True,
                tool_name="contact_form",
                params={"person_name": person_name, "phone_number": phone_number, "address": address},
                result={"success": True},
                response=f"Thank you, {person_name}! I've recorded your contact information."
            )
        else:
            return FormHandlerResult(
                success=False,
                tool_name="contact_form",
                params={},
                result={"success": False},
                response="Please provide at least your name."
            )


def register_default_handlers() -> None:
    """Register the default form handlers."""
    registry = FormHandlerRegistry.get_instance()
    
    # Register handlers
    registry.register_handler(LeaveFormHandler())
    registry.register_handler(ContactFormHandler())
    
    # Register follow-up patterns for leave server
    registry.register_followup_pattern(FollowUpPattern(
        server_name="leave",
        patterns=[
            re.compile(r'\d{1,2}[/\-]\d{1,2}'),  # Date patterns like 12/25, 12-25
            re.compile(r'\d{1,2}(?:st|nd|rd|th)?\s*(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.I),
        ],
        keywords=[
            "change", "modify", "update", "different", "instead",
            "actually", "make that", "make it", "rather", "switch"
        ]
    ))
    
    logger.info("✅ Default form handlers registered")


# Auto-register on module import
register_default_handlers()
