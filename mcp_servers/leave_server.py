"""
Leave Management MCP Server with Adaptive Card Support

This server provides leave-related tools that return Adaptive Cards for UI rendering.
It connects to PostgreSQL to check employee leave balances and validate leave requests.

Tools:
- analyze_leave_request: Uses LLM to extract dates/leave type from natural language
- get_leave_request_card: Returns an Adaptive Card form for leave requests
- validate_leave: Validates leave request and returns response Adaptive Card
- get_leave_balance: Gets current leave balance for an employee
- get_leave_types: Returns available leave types

Usage:
    This server is spawned by the Flask app via stdio transport.
    Configure in mcp_servers.json and it will be automatically managed.
"""

import json
import os
import logging
import re
import httpx
from datetime import datetime, timedelta
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
import psycopg2

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("leave-server")

# Initialize MCP server
mcp = FastMCP("leave-server")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "adventureworks"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "dikshith")
}

# Groq API configuration for entity extraction
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Import telemetry
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.telemetry import trace_llm_call, log_llm_event
from rate_limiter import get_global_rate_limiter

ENTITY_MODEL = "llama-3.3-70b-versatile"  # Fast model for extraction
_rate_limiter = get_global_rate_limiter()


# ============================================================================
# LLM ENTITY EXTRACTION
# ============================================================================

async def extract_leave_entities(query: str) -> Dict[str, Any]:
    """
    Use LLM to extract leave request entities from natural language.
    
    Extracts:
    - start_date: YYYY-MM-DD format
    - end_date: YYYY-MM-DD format
    - leave_type: sick leave, vacation leave, annual leave, or flexi leave
    
    Args:
        query: Natural language leave request (e.g., "I want to take sick leave from Dec 4 to Dec 6")
    
    Returns:
        Dict with extracted entities and confidence
    """
    if not GROQ_API_KEY:
        logger.warning("⚠️ GROQ_API_KEY not set - using basic extraction")
        return _basic_entity_extraction(query)
    
    today = datetime.now()
    current_year = today.year
    
    prompt = f"""Extract leave request details from this message. Today is {today.strftime('%Y-%m-%d')} ({today.strftime('%A')}).

User message: "{query}"

Extract and return ONLY valid JSON (no explanation):
{{
    "start_date": "YYYY-MM-DD or null if not specified",
    "end_date": "YYYY-MM-DD or null if not specified",
    "leave_type": "sick leave|vacation leave|annual leave|flexi leave or null if not specified",
    "duration_days": number or null,
    "confidence": 0.0-1.0
}}

Rules:
1. Use {current_year} for dates if year not specified
2. For relative dates like "next Monday", calculate the actual date
3. If only duration given (e.g., "3 days"), set start_date to today and calculate end_date
4. If single date given, use it for both start and end
5. ONLY set leave_type if the user explicitly mentions a type (sick, vacation, annual, flexi). Return null otherwise.
6. Return ONLY the JSON object, nothing else"""

    # Telemetry: trace this LLM call
    with trace_llm_call(
        name="leave-entity-extraction",
        model=f"groq/{ENTITY_MODEL}",
        input_data={"query_preview": query[:100]},
        model_parameters={"temperature": 0.1, "max_tokens": 200},
        metadata={"source": "leave_server"}
    ) as trace:
        try:
            # Rate Limit Check
            await _rate_limiter.wait_for_slot_async()
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    GROQ_API_URL,
                    json={
                        "model": ENTITY_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200
                    },
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                
                # Record successful call
                _rate_limiter.record_call()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Parse JSON response
                # Handle markdown code blocks if present
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\n?', '', content)
                    content = re.sub(r'\n?```$', '', content)
                
                entities = json.loads(content)
                
                trace.update(
                    output=content[:200],
                    metadata={"success": True, "entities_extracted": list(entities.keys())}
                )
                logger.info(f"🧠 LLM extracted entities: {entities}")
                return entities
                
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            log_llm_event("leave-extraction-error", {"error": str(e)}, level="ERROR")
            trace.update(output=f"Error: {e}", metadata={"success": False})
            return _basic_entity_extraction(query)


def _basic_entity_extraction(query: str) -> Dict[str, Any]:
    """
    Basic regex-based entity extraction fallback.
    """
    query_lower = query.lower()
    today = datetime.now()
    
    result = {
        "start_date": None,
        "end_date": None,
        "leave_type": None,
        "duration_days": None,
        "confidence": 0.5
    }
    
    # Extract leave type - only if explicitly mentioned
    if "sick" in query_lower:
        result["leave_type"] = "sick leave"
    elif "vacation" in query_lower:
        result["leave_type"] = "vacation leave"
    elif "annual" in query_lower:
        result["leave_type"] = "annual leave"
    elif "flexi" in query_lower:
        result["leave_type"] = "flexi leave"
    # Don't default - leave as None so user must select
    
    # Extract dates (simple patterns)
    # Pattern: "from X to Y" or "X to Y"
    date_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?(\w+)(?:\s+(\d{4}))?'
    matches = list(re.finditer(date_pattern, query_lower))
    
    months = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    dates = []
    for match in matches:
        day = int(match.group(1))
        month_str = match.group(2).lower()
        year = int(match.group(3)) if match.group(3) else today.year
        
        if month_str in months:
            try:
                date = datetime(year, months[month_str], day)
                dates.append(date.strftime('%Y-%m-%d'))
            except ValueError:
                pass
    
    if len(dates) >= 2:
        result["start_date"] = dates[0]
        result["end_date"] = dates[1]
    elif len(dates) == 1:
        result["start_date"] = dates[0]
        result["end_date"] = dates[0]
    
    # Check for "tomorrow", "next week" etc.
    if "tomorrow" in query_lower:
        tomorrow = today + timedelta(days=1)
        result["start_date"] = tomorrow.strftime('%Y-%m-%d')
        result["end_date"] = tomorrow.strftime('%Y-%m-%d')
    
    return result


def get_db_connection():
    """Create a database connection with timeout."""
    try:
        conn = psycopg2.connect(
            **DB_CONFIG,
            connect_timeout=5  # 5 second connection timeout
        )
        logger.info("✅ Database connection established")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        return None


def get_employee_leave_balance(employee_id: int) -> dict:
    """
    Fetch leave balance from the database for a given employee.
    
    Returns dict with leave types and their available hours.
    """
    logger.info(f"📊 Fetching leave balance for employee {employee_id}")
    
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed"}
    
    try:
        cursor = conn.cursor()
        
        # Query the humanresources.employee table for vacation and sick leave
        query = """
            SELECT 
                e.businessentityid,
                e.vacationhours,
                e.sickleavehours,
                p.firstname,
                p.lastname
            FROM humanresources.employee e
            JOIN person.person p ON e.businessentityid = p.businessentityid
            WHERE e.businessentityid = %s
        """
        
        cursor.execute(query, (employee_id,))
        result = cursor.fetchone()
        
        if result:
            # Convert hours to days (assuming 8 hours per day)
            vacation_days = result[1] / 8 if result[1] else 0
            sick_days = result[2] / 8 if result[2] else 0
            
            balance = {
                "employee_id": result[0],
                "employee_name": f"{result[3]} {result[4]}",
                "vacation_leave": round(vacation_days, 1),
                "sick_leave": round(sick_days, 1),
                "annual_leave": round(vacation_days, 1),  # Annual = Vacation in this schema
                "flexi_leave": 5.0,  # Default flexi leave
                "vacation_hours": result[1] or 0,
                "sick_leave_hours": result[2] or 0
            }
            logger.info(f"✅ Found balance for {balance['employee_name']}: vacation={vacation_days}d, sick={sick_days}d")
            return balance
        else:
            logger.warning(f"⚠️ Employee {employee_id} not found")
            return {"error": f"Employee {employee_id} not found"}
            
    except Exception as e:
        logger.error(f"❌ Error fetching leave balance: {e}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()


def calculate_business_days(start_date: str, end_date: str) -> int:
    """
    Calculate number of business days (excluding weekends) between two dates.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end < start:
            return 0
            
        business_days = 0
        current = start
        
        while current <= end:
            # 0 = Monday, 6 = Sunday
            if current.weekday() < 5:  # Monday to Friday
                business_days += 1
            current += timedelta(days=1)
            
        return business_days
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        return 0


def build_request_card(start_date: str = "", end_date: str = "", leave_type: str = "") -> dict:
    """
    Build the leave request Adaptive Card form.
    """
    return {
        "type": "AdaptiveCard",
        "version": "1.5",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "body": [
            {
                "type": "TextBlock",
                "text": "Leave Calculator",
                "weight": "Bolder",
                "size": "Large",
                "spacing": "None"
            },
            {
                "type": "TextBlock",
                "text": "View your available leave balance. Your associated employee ID will be used.",
                "wrap": True,
                "size": "Small",
                "isSubtle": True,
                "spacing": "Small"
            },
            {
                "type": "TextBlock",
                "text": "Select the leave window you want to evaluate",
                "weight": "Bolder",
                "size": "Default",
                "wrap": True,
                "spacing": "Medium"
            },
            {
                "type": "ColumnSet",
                "spacing": "Small",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "Input.Date",
                                "id": "start_date",
                                "placeholder": "Start Date",
                                "value": start_date
                            }
                        ]
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "Input.Date",
                                "id": "end_date",
                                "placeholder": "End Date",
                                "value": end_date
                            }
                        ]
                    }
                ]
            },
            {
                "type": "Input.ChoiceSet",
                "id": "leave_type",
                "style": "compact",
                "placeholder": "Leave Type",
                "value": leave_type,
                "spacing": "Small",
                "choices": [
                    {"title": "Sick Leave", "value": "sick leave"},
                    {"title": "Vacation Leave", "value": "vacation leave"},
                    {"title": "Annual Leave", "value": "annual leave"},
                    {"title": "Flexi Leave", "value": "flexi leave"}
                ]
            }
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "Calculate Leave Impact",
                "style": "positive",
                "data": {
                    "submit": "leave_calculator_submit",
                    "intent": "submit_leave_form"
                }
            }
        ]
    }


def build_response_card(
    status_message: str,
    status_style: str,
    leave_type: str,
    requested_days: int,
    start_date: str,
    end_date: str,
    remaining_days: float,
    sick_leave_days: float,
    vacation_days: float
) -> dict:
    """
    Build the leave validation response Adaptive Card.
    
    Args:
        status_message: "Approved" or "Insufficient Balance"
        status_style: "Good" for approved, "Attention" for denied
        leave_type: Type of leave requested
        requested_days: Number of days requested
        start_date: Start date of leave
        end_date: End date of leave
        remaining_days: Balance remaining after this leave
        sick_leave_days: Current sick leave balance
        vacation_days: Current vacation leave balance
    """
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "Container",
                "id": "statusContainer",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"{status_message} for {requested_days} Day {leave_type}.",
                        "color": status_style,
                        "weight": "Bolder",
                        "size": "Medium",
                        "wrap": True,
                        "spacing": "Small"
                    }
                ]
            },
            {
                "type": "Container",
                "id": "dateRangeContainer",
                "spacing": "Small",
                "items": [
                    {
                        "type": "RichTextBlock",
                        "inlines": [
                            {"type": "TextRun", "text": "Leave Days : "},
                            {"type": "TextRun", "text": f"{start_date} to {end_date}"}
                        ]
                    }
                ]
            },
            {
                "type": "Container",
                "id": "balanceInfoContainer",
                "spacing": "Small",
                "items": [
                    {
                        "type": "RichTextBlock",
                        "inlines": [
                            {"type": "TextRun", "text": f"{leave_type} Balance after Leave: "},
                            {"type": "TextRun", "text": f"{remaining_days} Days"}
                        ]
                    }
                ]
            },
            {
                "type": "Container",
                "id": "currentBalanceContainer",
                "spacing": "Medium",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "Current Balance",
                        "weight": "Bolder",
                        "size": "Medium",
                        "spacing": "Small"
                    },
                    {
                        "type": "ColumnSet",
                        "spacing": "Small",
                        "columns": [
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "Container",
                                        "style": "emphasis",
                                        "bleed": True,
                                        "items": [
                                            {
                                                "type": "RichTextBlock",
                                                "horizontalAlignment": "Center",
                                                "inlines": [
                                                    {"type": "TextRun", "text": "Sick Leave : "},
                                                    {"type": "TextRun", "text": f"{sick_leave_days} Days"}
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            },
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "Container",
                                        "style": "emphasis",
                                        "bleed": True,
                                        "items": [
                                            {
                                                "type": "RichTextBlock",
                                                "horizontalAlignment": "Center",
                                                "inlines": [
                                                    {"type": "TextRun", "text": "Vacation Leave : "},
                                                    {"type": "TextRun", "text": f"{vacation_days} Days"}
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ],
        "actions": []
    }


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
async def analyze_leave_request(query: str, employee_id: int = 1) -> dict:
    """
    Analyze a natural language leave request and intelligently respond.
    
    This is the PRIMARY entry point for leave requests. It:
    1. Uses LLM to extract dates and leave type from the user's message
    2. If ALL required info is provided (dates + leave type) → DIRECTLY validates and returns result
    3. If ANY info is missing → Returns pre-filled form for user to complete
    
    Call this when a user mentions anything about leave, time off, vacation, sick days, etc.
    
    Examples that will DIRECTLY validate:
    - "I want to take sick leave from Dec 4 to Dec 6" → has dates + leave type
    - "Need vacation leave from 2025-01-10 to 2025-01-15" → has dates + leave type
    
    Examples that will show form:
    - "I want to take leave" → missing dates and type
    - "Can I get sick leave?" → missing dates
    - "I need time off from Dec 4 to Dec 6" → missing leave type
    
    Args:
        query: Natural language leave request from the user
        employee_id: The employee's ID (defaults to 1)
    
    Returns:
        dict: Either validation result card (if all info provided) or form card (if info missing)
    """
    logger.info("🔌 MCP TOOL: analyze_leave_request")
    logger.info(f"   Query: {query}")
    logger.info(f"   Employee ID: {employee_id}")
    
    # Extract entities using LLM
    entities = await extract_leave_entities(query)
    
    logger.info(f"   Extracted entities: {entities}")
    
    start_date = entities.get("start_date") or ""
    end_date = entities.get("end_date") or ""
    leave_type = entities.get("leave_type") or ""
    
    # SMART ROUTING: If all required info is provided, validate directly
    if start_date and end_date and leave_type:
        logger.info("   ✅ All parameters provided - validating directly!")
        
        # Call validate_leave directly and return the result
        validation_result = validate_leave(
            employee_id=employee_id,
            start_date=start_date,
            end_date=end_date,
            leave_type=leave_type
        )
        
        # Add context about entity extraction
        validation_result["entities"] = entities
        validation_result["auto_validated"] = True
        validation_result["session_type"] = "leave_calculator"
        
        logger.info("   Result type: adaptive_card (direct validation result)")
        return validation_result
    
    # Otherwise, show the form with pre-filled values
    logger.info("   ⚠️ Missing parameters - showing form")
    missing = []
    if not start_date:
        missing.append("start date")
    if not end_date:
        missing.append("end date")
    if not leave_type:
        missing.append("leave type")
    logger.info(f"   Missing: {', '.join(missing)}")
    
    # Build the request card with pre-filled values
    card = build_request_card(start_date, end_date, leave_type)
    
    # Create a helpful message based on what was extracted
    if start_date or end_date or leave_type:
        filled_parts = []
        if leave_type:
            filled_parts.append(leave_type.title())
        if start_date:
            filled_parts.append(f"from {start_date}")
        if end_date:
            filled_parts.append(f"to {end_date}")
        message = f"I've filled in: {' '.join(filled_parts)}. Please complete the remaining fields ({', '.join(missing)}) and click 'Calculate Leave Impact'."
    else:
        message = "Please fill out the leave request form below to check your eligibility."
    
    result = {
        "type": "adaptive_card",
        "card": card,
        "message": message,
        "entities": entities,
        "missing_fields": missing,
        "session_type": "leave_calculator",  # For stickiness
        "is_sticky": True,
        "metadata": {
            "template": "leave_request"  # For CSS styling
        }
    }
    
    logger.info("   Result type: adaptive_card (pre-filled form)")
    return result


@mcp.tool()
def get_leave_request_card(
    start_date: str = "",
    end_date: str = "",
    leave_type: str = ""
) -> dict:
    """
    Get the leave request form as an Adaptive Card.
    
    This returns an Adaptive Card that the user can fill out to request leave.
    Call this when a user wants to apply for leave, check leave balance impact,
    or calculate leave days.
    
    Args:
        start_date: Optional pre-filled start date (YYYY-MM-DD format)
        end_date: Optional pre-filled end date (YYYY-MM-DD format)
        leave_type: Optional pre-filled leave type (sick leave, vacation leave, annual leave, flexi leave)
    
    Returns:
        dict: Adaptive Card JSON with the leave request form
    """
    logger.info("🔌 MCP TOOL: get_leave_request_card")
    logger.info(f"   start_date={start_date}, end_date={end_date}, leave_type={leave_type}")
    
    card = build_request_card(start_date, end_date, leave_type)
    
    result = {
        "type": "adaptive_card",
        "card": card,
        "message": "Please fill out the leave request form below to check your eligibility.",
        "metadata": {
            "template": "leave_request"  # For CSS styling
        }
    }
    
    logger.info("   Result type: adaptive_card")
    return result


@mcp.tool()
def validate_leave(
    employee_id: int,
    start_date: str,
    end_date: str,
    leave_type: str
) -> dict:
    """
    Validate a leave request and return the result as an Adaptive Card.
    
    Checks if the employee has sufficient leave balance for the requested dates
    and returns an Adaptive Card showing approval status and remaining balance.
    
    Args:
        employee_id: The employee's ID in the system
        start_date: Leave start date in YYYY-MM-DD format
        end_date: Leave end date in YYYY-MM-DD format
        leave_type: Type of leave (sick leave, vacation leave, annual leave, flexi leave)
    
    Returns:
        dict: Adaptive Card JSON with validation result and balance info
    """
    import sys
    logger.info("🔌 MCP TOOL: validate_leave")
    logger.info(f"   employee_id={employee_id}, start_date={start_date}, end_date={end_date}, leave_type={leave_type}")
    sys.stderr.write(f"validate_leave called: emp={employee_id}, dates={start_date} to {end_date}\n")
    sys.stderr.flush()
    
    try:
        # Get employee's current leave balance
        logger.info("   Fetching leave balance...")
        sys.stderr.write("Fetching leave balance...\n")
        sys.stderr.flush()
        
        balance = get_employee_leave_balance(employee_id)
        
        logger.info(f"   Balance result: {balance}")
        sys.stderr.write(f"Balance result: {balance}\n")
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"   Exception getting balance: {e}")
        sys.stderr.write(f"Exception: {e}\n")
        sys.stderr.flush()
        return {
            "type": "error",
            "message": f"Database error: {str(e)}",
            "card": None
        }
    
    if "error" in balance:
        logger.error(f"   Error: {balance['error']}")
        return {
            "type": "error",
            "message": balance["error"],
            "card": None
        }
    
    # Calculate requested days
    requested_days = calculate_business_days(start_date, end_date)
    
    if requested_days == 0:
        return {
            "type": "error",
            "message": "Invalid date range. Please ensure end date is after start date.",
            "card": None
        }
    
    # Normalize leave type
    leave_type_lower = leave_type.lower().strip()
    
    # Map leave type to balance field
    leave_type_mapping = {
        "sick leave": "sick_leave",
        "vacation leave": "vacation_leave",
        "annual leave": "annual_leave",
        "flexi leave": "flexi_leave"
    }
    
    balance_key = leave_type_mapping.get(leave_type_lower)
    
    if not balance_key:
        logger.warning(f"   Unknown leave type: {leave_type}")
        return {
            "type": "error",
            "message": f"Unknown leave type: {leave_type}. Valid types are: sick leave, vacation leave, annual leave, flexi leave",
            "card": None
        }
    
    available_days = balance.get(balance_key, 0)
    
    # Check eligibility
    is_eligible = available_days >= requested_days
    remaining_days = max(0, available_days - requested_days)
    
    if is_eligible:
        status_message = "Eligible"
        status_style = "Good"
        logger.info(f"   ✅ Leave APPROVED: {requested_days} days, {remaining_days} remaining")
    else:
        status_message = "Insufficient Balance"
        status_style = "Attention"
        logger.info(f"   ❌ Leave DENIED: requested {requested_days} days, only {available_days} available")
    
    # Build response card
    card = build_response_card(
        status_message=status_message,
        status_style=status_style,
        leave_type=leave_type.title(),
        requested_days=requested_days,
        start_date=start_date,
        end_date=end_date,
        remaining_days=round(remaining_days, 1),
        sick_leave_days=balance.get("sick_leave", 0),
        vacation_days=balance.get("vacation_leave", 0)
    )
    
    result = {
        "type": "adaptive_card",
        "card": card,
        "is_eligible": is_eligible,
        "employee_name": balance.get("employee_name", "Unknown"),
        "requested_days": requested_days,
        "available_days": available_days,
        "remaining_days": remaining_days,
        "message": f"{'Leave request approved!' if is_eligible else 'Insufficient leave balance.'} You requested {requested_days} days of {leave_type}. Available: {available_days} days.",
        "metadata": {
            "template": "leave_response"  # For CSS styling
        }
    }
    
    logger.info("   Result type: adaptive_card (validation result)")
    return result


@mcp.tool()
def get_leave_balance(employee_id: int) -> dict:
    """
    Get the current leave balance for an employee.
    
    Retrieves all leave type balances for the specified employee from the database.
    
    Args:
        employee_id: The employee's ID in the system
    
    Returns:
        dict: Employee's leave balance information including all leave types
    """
    logger.info("🔌 MCP TOOL: get_leave_balance")
    logger.info(f"   employee_id={employee_id}")
    
    balance = get_employee_leave_balance(employee_id)
    
    if "error" in balance:
        logger.error(f"   Error: {balance['error']}")
        return {
            "type": "error",
            "message": balance["error"]
        }
    
    logger.info(f"   Balance retrieved for {balance['employee_name']}")
    return {
        "type": "balance",
        "employee_id": balance["employee_id"],
        "employee_name": balance["employee_name"],
        "balances": {
            "sick_leave": balance["sick_leave"],
            "vacation_leave": balance["vacation_leave"],
            "annual_leave": balance["annual_leave"],
            "flexi_leave": balance["flexi_leave"]
        },
        "message": f"Leave balance for {balance['employee_name']}: Sick Leave: {balance['sick_leave']} days, Vacation: {balance['vacation_leave']} days, Annual: {balance['annual_leave']} days, Flexi: {balance['flexi_leave']} days"
    }


@mcp.tool()
def get_leave_types() -> dict:
    """
    Get the list of available leave types.
    
    Returns all supported leave types that employees can request.
    
    Returns:
        dict: List of available leave types with descriptions
    """
    logger.info("🔌 MCP TOOL: get_leave_types")
    
    return {
        "type": "leave_types",
        "leave_types": [
            {
                "id": "sick_leave",
                "name": "Sick Leave",
                "description": "Leave for medical reasons or illness"
            },
            {
                "id": "vacation_leave",
                "name": "Vacation Leave",
                "description": "Planned time off for personal activities"
            },
            {
                "id": "annual_leave",
                "name": "Annual Leave",
                "description": "Yearly entitled leave allocation"
            },
            {
                "id": "flexi_leave",
                "name": "Flexi Leave",
                "description": "Flexible leave for personal matters"
            }
        ],
        "message": "Available leave types: Sick Leave, Vacation Leave, Annual Leave, and Flexi Leave"
    }


@mcp.tool()
def calculate_leave_days(start_date: str, end_date: str) -> dict:
    """
    Calculate the number of business days between two dates.
    
    Calculates working days (excluding weekends) for leave planning purposes.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        dict: Number of business days and date details
    """
    logger.info("🔌 MCP TOOL: calculate_leave_days")
    logger.info(f"   start_date={start_date}, end_date={end_date}")
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end < start:
            logger.warning("   End date before start date")
            return {
                "type": "error",
                "message": "End date must be after start date"
            }
        
        business_days = calculate_business_days(start_date, end_date)
        total_days = (end - start).days + 1
        weekend_days = total_days - business_days
        
        logger.info(f"   Calculated: {business_days} business days")
        return {
            "type": "calculation",
            "start_date": start_date,
            "end_date": end_date,
            "business_days": business_days,
            "total_days": total_days,
            "weekend_days": weekend_days,
            "message": f"From {start_date} to {end_date}: {business_days} business days ({weekend_days} weekend days excluded)"
        }
    except ValueError as e:
        logger.error(f"   Date parsing error: {e}")
        return {
            "type": "error",
            "message": f"Invalid date format. Please use YYYY-MM-DD. Error: {str(e)}"
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the MCP server using stdio transport
    mcp.run()