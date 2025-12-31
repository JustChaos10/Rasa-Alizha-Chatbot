"""
Flask Application - LLM MCP Architecture

This is the main web server that:
1. Serves the chat UI
2. Routes messages through the LLM Router
3. Handles file uploads
4. Manages user authentication
"""

# Load environment variables FIRST, before any other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Will warn later if needed

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
from flask_migrate import Migrate
from werkzeug.utils import secure_filename

from shared_utils import get_service_manager, MessageFormatter, logger
from rate_limiter import get_global_rate_limiter
from auth.models import db, User
from auth.routes import auth_bp
from auth.rbac import role_required

# Import the Hybrid Router (Rasa + LLM)
from architecture.hybrid_router import HybridRouter, HybridRouterConfig
from architecture.registry import ToolRegistry
from architecture.conversation_memory import ConversationMemory
from architecture.document_manager import get_document_manager

# Initialize Langfuse telemetry for LLM observability
from architecture.telemetry import init_langfuse, is_telemetry_enabled, flush_telemetry
init_langfuse()
if is_telemetry_enabled():
    logger.info("✅ Langfuse telemetry enabled for LLM observability")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Create Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this'))
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Database configuration
if os.getenv('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
else:
    # Use absolute path to avoid issues with working directory
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{data_dir}/app.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Flask-Login configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

CORS(app)

# Register blueprints
app.register_blueprint(auth_bp)

# Upload configuration
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MAX_UPLOAD_AGE_HOURS = int(os.getenv('UPLOAD_MAX_AGE_HOURS', '24'))
MAX_UPLOAD_FILES = int(os.getenv('UPLOAD_MAX_FILES', '200'))

# Initialize Hybrid Router (Rasa + LLM)
_router: Optional[HybridRouter] = None

def get_router() -> HybridRouter:
    """Get or create the Hybrid Router instance."""
    global _router
    if _router is None:
        # Configure hybrid routing
        config = HybridRouterConfig(
            enable_rasa_layer=True,  # Enable Rasa for small talk
            use_predefined_responses=True,  # Fast responses for greetings
            rasa_confidence_threshold=0.7,  # High confidence required
        )
        _router = HybridRouter(config)
        logger.info("✅ Hybrid Router initialized (Rasa + LLM)")
        
        # Log available tools
        registry = ToolRegistry()
        tools = registry.list_tools()
        logger.info(f"📦 Loaded {len(tools)} tools: {', '.join(tools)}")
    return _router


def cleanup_upload_folder(max_age_hours: int = MAX_UPLOAD_AGE_HOURS, max_files: int = MAX_UPLOAD_FILES) -> None:
    """Remove old or excess files from the upload directory."""
    try:
        if not UPLOAD_FOLDER.exists():
            return
        files = [p for p in UPLOAD_FOLDER.iterdir() if p.is_file()]
        if max_age_hours > 0:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            for file_path in files:
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                    file_path.unlink(missing_ok=True)
        if max_files > 0:
            files = [p for p in UPLOAD_FOLDER.iterdir() if p.is_file()]
            if len(files) > max_files:
                for file_path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[max_files:]:
                    file_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Upload cleanup warning: {exc}")


ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'docx', 'txt'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def _append_json_log(entry: Dict[str, Any], log_file: Path = Path("chatbot.log")) -> None:
    """Append a JSON log entry to the chatbot log file."""
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to chatbot.log: {e}")


# Thread-safe async execution for Flask's threaded mode
import threading

_async_loop: Optional[asyncio.AbstractEventLoop] = None
_async_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _start_async_loop():
    """Start the async event loop in a background thread."""
    global _async_loop
    _async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_async_loop)
    _async_loop.run_forever()


async def _generate_smart_session_name(user_message: str, bot_response: str) -> str:
    """
    Generate a ChatGPT-style smart session name using the LLM.
    
    Examples:
    - "Weather in Bangalore" 
    - "Leave Request for December"
    - "Dashboard Creation"
    - "News Headlines"
    """
    import httpx
    
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        # Fallback to simple name
        return (user_message[:40] + '...') if len(user_message) > 40 else user_message
    
    prompt = f"""Generate a short, descriptive title (2-5 words) for this chat conversation. 
The title should capture the main topic or intent. Don't use quotes or punctuation.

User's first message: "{user_message[:200]}"
Bot's response summary: "{bot_response[:100] if bot_response else 'N/A'}"

Examples of good titles:
- Weather in Mumbai
- Leave Request Help
- Company Dashboard
- News from Saudi Arabia
- Contact Form Setup

Title:"""

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 20
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                title = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                # Clean up the title
                title = title.strip('"\'').strip()
                # Ensure it's not too long
                if len(title) > 50:
                    title = title[:47] + "..."
                if title:
                    return title
    except Exception as e:
        logger.warning(f"LLM name generation failed: {e}")
    
    # Fallback
    return (user_message[:40] + '...') if len(user_message) > 40 else user_message


def _get_async_loop() -> asyncio.AbstractEventLoop:
    """Get or create the shared async event loop."""
    global _async_loop, _async_thread
    
    with _loop_lock:
        if _async_loop is None or not _async_loop.is_running():
            _async_thread = threading.Thread(target=_start_async_loop, daemon=True)
            _async_thread.start()
            # Wait for loop to start
            import time
            for _ in range(50):  # Wait up to 5 seconds
                if _async_loop is not None and _async_loop.is_running():
                    break
                time.sleep(0.1)
            if _async_loop is None or not _async_loop.is_running():
                raise RuntimeError("Failed to start async event loop")
    
    return _async_loop


def run_async(coro):
    """Run async coroutine in sync context using a shared event loop.
    
    This is thread-safe and works with Flask's threaded=True mode.
    All async operations run on a single dedicated event loop thread.
    """
    loop = _get_async_loop()
    
    # Submit coroutine to the async thread and wait for result
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=120)  # 2 minute timeout
    except Exception as e:
        logger.error(f"Async execution error: {e}")
        raise


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    """
    Main chat endpoint - Routes messages through the LLM Router.
    
    This replaces the old Rasa proxy with direct LLM-based tool routing.
    """
    try:
        data = request.get_json(force=True) or {}
        message: str = data.get("message", "").strip()
        
        if not message:
            return jsonify([{"text": "Please enter a message."}])
        
        # Get user context
        try:
            user_id = getattr(current_user, 'id', None)
            sender_id = f"user-{int(user_id)}" if user_id is not None else "flask_user"
        except Exception:
            sender_id = "flask_user"
        
        metadata: Dict[str, Any] = data.get("metadata", {})
        
        # Inject user role and employee_id (using user.id) for RAG/RBAC if available
        try:
            user_role = getattr(current_user, 'role', None)
            if user_role:
                metadata["role"] = str(user_role)
            # Use user.id as employee_id for MCP tools
            if user_id:
                metadata["employee_id"] = str(user_id)
        except Exception:
            pass
        
        # Language detection and translation
        translation_service = get_service_manager().get_translation_service()
        
        # DEBUG: Log incoming metadata for language debugging
        logger.info(f"📨 Incoming message: {message}")
        logger.info(f"📨 Metadata keys: {list(metadata.keys())}")
        if "language" in metadata:
            logger.info(f"📨 Metadata language: {metadata['language']}")
        
        # Check if language is explicitly provided in metadata (e.g. from adaptive card submission)
        provided_language = metadata.get("language")
        if provided_language in ["ar", "en"]:
             detected_language = provided_language
             confidence = 1.0
             logger.info(f"🌍 Using provided language from metadata: {detected_language}")
        else:
             detected_language, confidence = translation_service.detect_language(message)
        
        metadata["original_language"] = detected_language
        metadata["original_message"] = message
        metadata["language_confidence"] = confidence
        direction = 'rtl' if detected_language == 'ar' else 'ltr'
        metadata["text_direction"] = direction
        
        # Translate Arabic input to English for processing
        # BUT: Do NOT translate slash commands (form submissions like /submit_leave_form)
        processed_message = message
        if detected_language == "ar" and not message.startswith("/"):
            processed_message = translation_service.translate_arabic_to_english(message)
            logger.info(f"🔄 Translated AR→EN: {message[:50]}... → {processed_message[:50]}...")
        elif message.startswith("/"):
            logger.info(f"📝 Slash command detected, skipping translation: {message}")
        
        # Route through Hybrid Router (Rasa for small talk, LLM for tools)
        router = get_router()
        result = run_async(router.process_message(processed_message, context=metadata, sender_id=sender_id))
        
        # Format response
        response_text = result.get("response", "I couldn't process your request.")
        
        # Extract adaptive card (move logic up)
        adaptive_card = None
        result_data = result.get("result", {})
        if not isinstance(result_data, dict):
            result_data = {}
        
        # DEBUG: Log result structure for card extraction
        logger.info(f"🔍 Result keys: {list(result.keys())}")
        logger.info(f"🔍 Result type field: {result.get('type')}")
        logger.info(f"🔍 Result_data keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")
        logger.info(f"🔍 Result_data type field: {result_data.get('type') if isinstance(result_data, dict) else 'N/A'}")
        
        # Check for adaptive card in different result formats
        if result.get("type") == "card":
            # Legacy format (from form handler)
            adaptive_card = result.get("payload", {})
            logger.info(f"✅ Found card via result.type=card, payload present: {bool(adaptive_card)}")
        elif result_data.get("type") == "adaptive_card":
            # MCP tool format - direct response
            adaptive_card = result_data.get("card")
        else:
            # Check in tool_calls from code execution
            tool_calls = result_data.get("tool_calls", [])
            for call in tool_calls:
                call_result = call.get("result", {})
                # Check if the result data contains adaptive card
                data = call_result.get("data", call_result)
                if isinstance(data, dict):
                    if data.get("type") == "adaptive_card":
                        adaptive_card = data.get("card")
                        break
                elif isinstance(data, str):
                    # Try to parse as JSON
                    try:
                        import json
                        parsed = json.loads(data)
                        if isinstance(parsed, dict) and parsed.get("type") == "adaptive_card":
                            adaptive_card = parsed.get("card")
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Extract related questions (move logic up)
        related_questions = result_data.get("related_questions", [])
        
        # ============================================
        # TRANSLATE CONTENT IF ARABIC REQUEST
        # ============================================
        if detected_language == "ar":
            # Translate text response
            if response_text:
                try:
                    original_text = response_text
                    response_text = translation_service.translate_english_to_arabic(original_text)
                    logger.info(f"🔄 Translated EN→AR: {original_text[:50]}... → {response_text[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to translate response text: {e}")
            
            # Translate adaptive card
            if adaptive_card:
                try:
                    logger.info("🔄 Translating Adaptive Card to Arabic...")
                    logger.info(f"   Card before translation (preview): {str(adaptive_card)[:200]}")
                    adaptive_card = translation_service.translate_adaptive_card(adaptive_card)
                    logger.info(f"   Card after translation (preview): {str(adaptive_card)[:200]}")
                except Exception as trans_err:
                    logger.warning(f"Failed to translate adaptive card: {trans_err}")
            
            # Translate related questions
            if related_questions and isinstance(related_questions, list):
                try:
                    logger.info("🔄 Translating Related Questions to Arabic...")
                    translated_related = []
                    for q in related_questions:
                        if q.get("title") and q.get("prompt"):
                            translated_related.append({
                                "title": translation_service.translate_english_to_arabic(q["title"]),
                                "prompt": translation_service.translate_english_to_arabic(q["prompt"])
                            })
                    related_questions = translated_related
                except Exception as e:
                    logger.warning(f"Failed to translate related questions: {e}")

        # ============================================
        # SAVE MESSAGES TO POSTGRESQL
        # ============================================
        try:
            from auth.models import ChatSession, ChatMessage
            import uuid
            
            if user_id:
                # Find or create active session for user
                active_session = ChatSession.query.filter_by(user_id=user_id, is_active=True).first()
                
                if not active_session:
                    active_session = ChatSession(
                        session_id=str(uuid.uuid4())[:8],
                        user_id=user_id,
                        name='New Chat',
                        is_active=True
                    )
                    db.session.add(active_session)
                    db.session.commit()
                
                # Save user message
                user_msg = ChatMessage(
                    session_id=active_session.id,
                    role='user',
                    content=message,
                    metadata_json='{"language": "' + detected_language + '"}'
                )
                user_msg.msg_metadata = {'language': detected_language, 'processed': processed_message}
                db.session.add(user_msg)
                
                # Build assistant message metadata - include adaptive card if present
                assistant_metadata = {'routing': result.get("routing", {})}
                
                if adaptive_card:
                    assistant_metadata["type"] = "adaptive_card"
                    assistant_metadata["card"] = adaptive_card
                    # Extract template from result metadata
                    result_metadata = {}
                    if result_data.get("metadata"):
                        result_metadata = result_data["metadata"]
                    else:
                        for call in result_data.get("tool_calls", []):
                            call_result = call.get("result", {})
                            data = call_result.get("data", call_result)
                            if isinstance(data, dict) and data.get("metadata"):
                                result_metadata = data["metadata"]
                                break
                    assistant_metadata["card_metadata"] = result_metadata
                
                # Check for image_data (Chat Tool visual panel)
                if isinstance(result_data, dict) and result_data.get("image_data"):
                    assistant_metadata["image_data"] = result_data.get("image_data")
                
                # Extract and store chart_path for persistence
                saved_chart_path = None
                tool_calls = result_data.get("tool_calls", [])
                for call in tool_calls:
                    call_result = call.get("result", {})
                    data = call_result.get("data", call_result)
                    if isinstance(data, dict) and data.get("chart"):
                        saved_chart_path = data.get("chart")
                        break
                    if isinstance(data, dict) and data.get("chart_path"):
                        saved_chart_path = data.get("chart_path")
                        break
                    elif isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                            if isinstance(parsed, dict) and parsed.get("chart"):
                                saved_chart_path = parsed.get("chart")
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
                if not saved_chart_path and isinstance(result_data, dict):
                    saved_chart_path = result_data.get("chart")
                
                if saved_chart_path:
                    assistant_metadata["chart_path"] = saved_chart_path
                
                # Save assistant message with full metadata
                # If card exists, we might want to suppress text in history too, or keep it.
                # Keeping it ensures context is preserved.
                assistant_msg = ChatMessage(
                    session_id=active_session.id,
                    role='assistant',
                    content=response_text,
                    tool_name=result.get("tool")
                )
                assistant_msg.msg_metadata = assistant_metadata
                db.session.add(assistant_msg)
                
                # Update session preview if this is the first message
                if not active_session.preview:
                    active_session.preview = message[:100] if len(message) > 100 else message
                    # Auto-name the session with LLM-generated smart name
                    if active_session.name == 'New Chat':
                        try:
                            smart_name = run_async(_generate_smart_session_name(message, response_text))
                            active_session.name = smart_name
                        except Exception as name_err:
                            logger.warning(f"Failed to generate smart name, using fallback: {name_err}")
                            # Fallback to first 40 chars
                            active_session.name = (message[:40] + '...') if len(message) > 40 else message
                
                db.session.commit()
        except Exception as db_err:
            logger.warning(f"Failed to save messages to DB: {db_err}")
            db.session.rollback()
        
        # Handle adaptive cards - check multiple possible locations
        responses = []
        
        if adaptive_card:
            # Extract template from result metadata (from MCP server response)
            result_metadata = {}
            if result_data.get("metadata"):
                result_metadata = result_data["metadata"]
            else:
                # Check in tool_calls for metadata
                for call in result_data.get("tool_calls", []):
                    call_result = call.get("result", {})
                    data = call_result.get("data", call_result)
                    if isinstance(data, dict) and data.get("metadata"):
                        result_metadata = data["metadata"]
                        break
            
            # Return card payload for frontend rendering
            # Frontend expects: custom.payload === 'adaptiveCard' and custom.data contains the card
            # NO TEXT - just the card (user requested: card speaks for itself)
            responses.append({
                "text": "",  # Empty text - card is the response
                "custom": {
                    "payload": "adaptiveCard",  # This triggers the AdaptiveCard component
                    "data": adaptive_card,  # The actual card JSON
                    "metadata": {
                        "employee_id": metadata.get("employee_id", "1"),
                        "template": result_metadata.get("template", result.get("template", ""))
                    }
                },
                "metadata": {
                    "language": detected_language,
                    "direction": direction,
                    "tool": result.get("tool", "unknown")
                }
            })
        
        # Only show text response if there is NO adaptive card
        # This prevents redundant "Booklet created successfully" messages when the card is shown
        if not adaptive_card:
            # Standard text response - check for multiple messages
            if "---MESSAGE_SPLIT---" in response_text:
                parts = response_text.split("---MESSAGE_SPLIT---")
                for part in parts:
                    if part.strip():
                        responses.append({
                            "text": part.strip(),
                            "metadata": {
                                "language": detected_language,
                                "direction": direction,
                                "tool": result.get("tool", "unknown")
                            }
                        })
            else:
                responses.append({
                    "text": response_text,
                    "metadata": {
                        "language": detected_language,
                        "direction": direction,
                        "tool": result.get("tool", "unknown")
                    }
                })
        
        # Check for image_data (visual panel) - from chat tool for GK questions
        # Also check in tool_calls for MCP tools (like SQL) that return charts
        image_data = result_data.get("image_data")
        
        # Check tool_calls for image_data from MCP tools (like SQL charts)
        if not image_data:
            tool_calls = result_data.get("tool_calls", [])
            for call in tool_calls:
                call_result = call.get("result", {})
                data = call_result.get("data", call_result)
                if isinstance(data, dict) and data.get("image_data"):
                    image_data = data["image_data"]
                    break
                # Also check for chart_path directly (SQL tool format)
                if isinstance(data, dict) and data.get("chart_path"):
                    chart_path = data["chart_path"]
                    image_data = {
                        "image_url": chart_path,
                        "title": data.get("answer", "SQL Query Result")[:100],
                        "alt": "SQL Query Visualization"
                    }
                    break
        
        if image_data and isinstance(image_data, dict) and image_data.get("image_url"):
            responses.append({
                "custom": {
                    "payload": "visual_panel",
                    "data": {
                        "image_url": image_data.get("image_url"),
                        "thumbnail_url": image_data.get("image_url"),  # Same for now
                        "title": image_data.get("title", ""),
                        "alt": image_data.get("alt", ""),
                        "summary": "",  # Could add summary if available
                        "attribution": image_data.get("attribution", ""),
                        "source_url": image_data.get("source_url", "")
                    }
                }
            })
        
        # Check for chart paths from KB/SQL MCP servers
        chart_path = None
        # Look in tool_calls for chart data from MCP servers
        tool_calls = result_data.get("tool_calls", [])
        for call in tool_calls:
            call_result = call.get("result", {})
            data = call_result.get("data", call_result)
            if isinstance(data, dict) and data.get("chart"):
                chart_path = data.get("chart")
                break
            elif isinstance(data, str):
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict) and parsed.get("chart"):
                        chart_path = parsed.get("chart")
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Also check in direct result data (e.g., if MCP returns directly)
        if not chart_path and isinstance(result_data, dict):
            chart_path = result_data.get("chart")
        
        # Serve chart as inline_chart (displayed within chat, not visual_panel)
        if chart_path and isinstance(chart_path, str) and chart_path.strip():
            # Convert file path to URL
            # e.g., "charts/kb/bar_test_20251212_202534.png" -> "/charts/kb/bar_test_20251212_202534.png"
            if chart_path.startswith("charts/"):
                chart_url = "/" + chart_path
            else:
                chart_url = "/charts/" + chart_path
            
            logger.info(f"📊 Sending inline chart to frontend: {chart_url}")
            responses.append({
                "custom": {
                    "payload": "inline_chart",
                    "data": {
                        "image_url": chart_url,
                        "alt": "Data visualization chart",
                        "caption": ""
                    }
                }
            })
        
        # Check for related_questions (related links) - from chat tool
        if related_questions and isinstance(related_questions, list) and len(related_questions) > 0:
            responses.append({
                "custom": {
                    "payload": "related_links",
                    "data": {
                        "items": [
                            {"title": q.get("title", ""), "prompt": q.get("prompt", "")}
                            for q in related_questions
                            if q.get("title") and q.get("prompt")
                        ]
                    }
                }
            })
        
        # Log the interaction
        timestamp = datetime.now().isoformat()
        for item in responses:
            _append_json_log({
                "timestamp": timestamp,
                "event": "bot_uttered",
                "source": "llm_router",
                "sender_id": sender_id,
                "request": message,
                "processed_request": processed_message,
                "original_language": detected_language,
                "language_confidence": confidence,
                "tool_used": result.get("tool", "unknown"),
                "metadata": metadata,
                "response": item
            })
        
        return jsonify(responses)
        
    except Exception as e:
        logger.error(f"/chat error: {e}", exc_info=True)
        return jsonify([{"text": f"❌ Error: {str(e)}"}]), 500


@app.route("/api/tools", methods=["GET"])
@login_required
def list_tools():
    """List all available tools."""
    try:
        registry = ToolRegistry()
        tools = registry.list_tools()
        schemas = []
        for tool_name in tools:
            tool = registry.get_tool(tool_name)
            if tool:
                schema = tool.schema
                schemas.append({
                    "name": schema.name,
                    "description": schema.description,
                    "examples": schema.examples[:3] if schema.examples else []
                })
        return jsonify({
            "success": True,
            "tools": schemas
        })
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================
# SESSION MANAGEMENT API ENDPOINTS (PostgreSQL)
# =============================================

@app.route("/api/sessions", methods=["GET"])
@login_required
def list_sessions():
    """List all chat sessions for the current user."""
    try:
        from auth.models import ChatSession
        import uuid
        
        user_id = current_user.id
        
        # Get all sessions for user, ordered by last activity
        sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).all()
        
        # Find or create active session
        active_session = ChatSession.query.filter_by(user_id=user_id, is_active=True).first()
        
        if not active_session:
            # Create a new active session
            active_session = ChatSession(
                session_id=str(uuid.uuid4())[:8],
                user_id=user_id,
                name='New Chat',
                is_active=True
            )
            db.session.add(active_session)
            db.session.commit()
            sessions = [active_session] + sessions
        
        return jsonify({
            "success": True,
            "sessions": [s.to_dict() for s in sessions],
            "active_session_id": active_session.session_id
        })
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions", methods=["POST"])
@login_required
def create_session():
    """Create a new chat session."""
    try:
        from auth.models import ChatSession
        import uuid
        
        user_id = current_user.id
        data = request.get_json() or {}
        name = data.get("name") or "New Chat"
        
        # Deactivate all other sessions
        ChatSession.query.filter_by(user_id=user_id, is_active=True).update({'is_active': False})
        
        # Create new session
        new_session = ChatSession(
            session_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            name=name,
            is_active=True
        )
        db.session.add(new_session)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "session": new_session.to_dict()
        })
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["PUT"])
@login_required
def rename_session(session_id):
    """Rename a chat session."""
    try:
        from auth.models import ChatSession
        
        user_id = current_user.id
        data = request.get_json() or {}
        new_name = data.get("name", "").strip()
        
        if not new_name:
            return jsonify({"success": False, "error": "Name is required"}), 400
        
        session_obj = ChatSession.query.filter_by(session_id=session_id, user_id=user_id).first()
        
        if not session_obj:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        session_obj.name = new_name
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error renaming session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@login_required
def delete_session_api(session_id):
    """Delete a chat session."""
    try:
        from auth.models import ChatSession
        import uuid
        
        user_id = current_user.id
        
        session_obj = ChatSession.query.filter_by(session_id=session_id, user_id=user_id).first()
        
        if not session_obj:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        was_active = session_obj.is_active
        db.session.delete(session_obj)
        db.session.commit()
        
        # If deleted session was active, create a new one or activate another
        if was_active:
            remaining = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).first()
            if remaining:
                remaining.is_active = True
            else:
                # Create new session
                new_session = ChatSession(
                    session_id=str(uuid.uuid4())[:8],
                    user_id=user_id,
                    name='New Chat',
                    is_active=True
                )
                db.session.add(new_session)
            db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>/switch", methods=["POST"])
@login_required
def switch_session_api(session_id):
    """Switch to a different chat session and load its messages."""
    try:
        from auth.models import ChatSession, ChatMessage
        
        user_id = current_user.id
        
        # Find the target session
        target_session = ChatSession.query.filter_by(session_id=session_id, user_id=user_id).first()
        
        if not target_session:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        # Deactivate all sessions and activate target
        ChatSession.query.filter_by(user_id=user_id, is_active=True).update({'is_active': False})
        target_session.is_active = True
        db.session.commit()
        
        # Get session messages
        messages = ChatMessage.query.filter_by(session_id=target_session.id)\
            .order_by(ChatMessage.created_at.asc())\
            .all()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "messages": [m.to_llm_format() for m in messages]
        })
    except Exception as e:
        logger.error(f"Error switching session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/current", methods=["GET"])
@login_required
def get_current_session():
    """Get the current active session and its messages."""
    try:
        from auth.models import ChatSession, ChatMessage
        import uuid
        
        user_id = current_user.id
        
        # Find active session
        active_session = ChatSession.query.filter_by(user_id=user_id, is_active=True).first()
        
        if not active_session:
            # Create a new active session
            active_session = ChatSession(
                session_id=str(uuid.uuid4())[:8],
                user_id=user_id,
                name='New Chat',
                is_active=True
            )
            db.session.add(active_session)
            db.session.commit()
        
        # Get session messages
        messages = ChatMessage.query.filter_by(session_id=active_session.id)\
            .order_by(ChatMessage.created_at.asc())\
            .all()
        
        return jsonify({
            "success": True,
            "session": active_session.to_dict(),
            "messages": [m.to_llm_format() for m in messages]
        })
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


def initialize_session():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'processed_files' not in session:
        session['processed_files'] = []


@app.route('/')
@login_required
def index():
    initialize_session()
    return render_template('components/index.html', chat_history=session.get('chat_history', []))


@app.route('/login')
def login_redirect():
    return redirect(url_for('auth.login'))


@app.route('/logout')
def logout_redirect():
    return redirect(url_for('auth.logout'))


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file uploads for image/document processing."""
    initialize_session()
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        logger.info(f"📤 File uploaded: {file.filename} -> {filepath}")
        
        file_type = request.form.get('file_type', 'auto')
        question = request.form.get('question', '')
        add_to_kb = request.form.get('add_to_kb', 'false').lower() == 'true'
        
        if file_type == 'auto':
            file_type = 'image' if is_image_file(file.filename) else 'document'
        
        if not question:
            question = 'Describe this image in detail' if file_type == 'image' else 'Summarize the main points'
        
        # Process file
        logger.info(f"🔄 Processing {file_type}: {file.filename}")
        services = get_service_manager()
        if file_type == 'image':
            result = services.get_file_service().process_image(filepath)
        else:
            result = services.get_file_service().process_document(filepath)
        
        # Handle empty result
        if not result or not result.strip():
            logger.warning(f"⚠️ Empty result for file: {file.filename}")
            result = f"📄 File '{file.filename}' was uploaded but could not be analyzed. This may happen with:\n- Non-English documents (limited support)\n- Scanned PDFs without OCR\n- Empty or corrupted files\n\nThe file has been saved and can be accessed later."
        
        result = MessageFormatter.clean_markdown_text(result)
        logger.info(f"✅ File processed: {file.filename} (result length: {len(result)})")
        
        # Optionally add document to knowledge base for future querying
        kb_status = None
        if add_to_kb and file_type == 'document':
            try:
                from architecture.document_manager import get_document_manager
                doc_manager = get_document_manager()
                # Move to documents folder for KB indexing
                move_result = doc_manager.move_to_documents(filename, source_dir=app.config['UPLOAD_FOLDER'])
                kb_status = "added_to_kb" if move_result['success'] else "kb_failed"
                logger.info(f"📚 KB status for {file.filename}: {kb_status}")
            except Exception as kb_err:
                logger.error(f"KB error: {kb_err}")
                kb_status = "kb_error"
        
        file_info = {
            'type': file_type,
            'name': file.filename,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'result': result[:500] + '...' if len(result) > 500 else result,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        session['processed_files'].append(file_info)
        if len(session['processed_files']) > 3:
            session['processed_files'] = session['processed_files'][-3:]
        session.modified = True
        
        cleanup_upload_folder()
        
        _append_json_log({
            "timestamp": datetime.now().isoformat(),
            "event": "bot_uttered",
            "sender_id": session.get('session_id', 'flask_user'),
            "request": f"/upload {file_type}",
            "metadata": {"filename": file.filename},
            "response": {"text": result, "file_type": file_type},
            "source": "flask_upload"
        })
        
        return jsonify({
            'success': True,
            'result': result,
            'file_type': file_type,
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'File processing failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        services = get_service_manager()
        services.get_llm_service()._get_api_key()
        api_key_status = True
    except ValueError:
        api_key_status = False
    
    registry = ToolRegistry()
    tools = registry.list_tools()
    
    return jsonify({
        'status': 'healthy',
        'architecture': 'LLM MCP Router',
        'groq_api': 'configured' if api_key_status else 'missing',
        'tools_loaded': len(tools),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/healthz')
def healthz():
    return jsonify({'ok': True}), 200


@app.route('/api/rate-limit-stats')
def rate_limit_stats():
    """Get current rate limiter statistics for debugging."""
    try:
        rate_limiter = get_global_rate_limiter()
        stats = rate_limiter.get_stats()
        return jsonify({
            'success': True,
            'rate_limiter': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    session['chat_history'] = []
    session['processed_files'] = []
    session.modified = True
    return jsonify({'success': True})


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.route('/api/documents/status', methods=['GET'])
@login_required
def get_documents_status():
    """Get status of documents and uploads."""
    try:
        doc_manager = get_document_manager()
        status = doc_manager.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/move-uploads', methods=['POST'])
@login_required
def move_uploads_to_documents():
    """Move all uploads to documents directory."""
    try:
        doc_manager = get_document_manager()
        result = doc_manager.move_all_uploads()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error moving uploads: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/index', methods=['POST'])
@login_required
async def trigger_document_indexing():
    """Trigger vector database rebuild."""
    try:
        doc_manager = get_document_manager()
        result = await doc_manager.trigger_indexing()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error triggering indexing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/process', methods=['POST'])
@login_required
async def process_and_index_documents():
    """Move uploads to documents and trigger indexing."""
    try:
        doc_manager = get_document_manager()
        result = await doc_manager.process_and_index()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/list', methods=['GET'])
@login_required
def list_documents():
    """List all documents in the knowledge base."""
    try:
        doc_manager = get_document_manager()
        documents = doc_manager.list_documents()
        return jsonify({'documents': documents, 'count': len(documents)})
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Chat Session Management
# ============================================================================
    return jsonify({'success': True, 'message': 'Chat history cleared'})


@app.route('/admin')
@role_required('admin')
def admin_dashboard():
    """Admin dashboard."""
    users = User.query.all()
    return render_template('admin/dashboard.html', users=users)


@app.route('/admin/api/users', methods=['GET'])
@role_required('admin')
def admin_list_users():
    """API endpoint to list all users for admin."""
    users = User.query.all()
    users_data = [
        {
            'id': user.id,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }
        for user in users
    ]
    return jsonify({'users': users_data})


@app.route('/api/models', methods=['GET'])
@login_required
def get_available_models():
    try:
        from shared_utils import Config
        return jsonify({
            'success': True,
            'models': Config.AVAILABLE_MODELS
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to retrieve models'}), 500


@app.route('/api/model_preference', methods=['POST'])
@login_required
def set_model_preference():
    try:
        data = request.get_json()
        preferred_model = data.get('model')
        
        if not preferred_model:
            return jsonify({'error': 'Model not specified'}), 400
        
        from shared_utils import Config
        if preferred_model not in Config.AVAILABLE_MODELS and preferred_model != 'auto':
            return jsonify({'error': 'Invalid model'}), 400
        
        if 'model_preferences' not in session:
            session['model_preferences'] = {}
        
        session['model_preferences']['preferred_model'] = preferred_model
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f'Model preference set to: {preferred_model}',
            'preferred_model': preferred_model
        })
        
    except Exception as e:
        logger.error(f"Error setting model preference: {e}")
        return jsonify({'error': 'Failed to set model preference'}), 500


@app.route('/api/model_preference', methods=['GET'])
@login_required
def get_model_preference():
    try:
        preferences = session.get('model_preferences', {})
        preferred_model = preferences.get('preferred_model', 'auto')
        return jsonify({
            'success': True,
            'preferred_model': preferred_model
        })
    except Exception as e:
        logger.error(f"Error getting model preference: {e}")
        return jsonify({'error': 'Failed to get model preference'}), 500


@app.route('/api/test_model', methods=['POST'])
@login_required
def test_model():
    """Test a specific model with a query."""
    try:
        data = request.get_json()
        model = data.get('model', 'llama-3.3-70b-versatile')
        test_query = data.get('query', 'Hello! Please respond with a short greeting.')
        
        from shared_utils import Config
        if model not in Config.AVAILABLE_MODELS:
            return jsonify({'error': 'Invalid model'}), 400
        
        services = get_service_manager()
        messages = [{"role": "user", "content": test_query}]
        
        start_time = datetime.now()
        response = services.get_llm_service().generate_text(
            messages=messages,
            model=model,
            max_tokens=100,
            temperature=0.7,
            timeout=10
        )
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        _append_json_log({
            "timestamp": datetime.now().isoformat(),
            "event": "bot_uttered",
            "sender_id": session.get('session_id', 'flask_user'),
            "request": f"/api/test_model: {test_query}",
            "metadata": {"model": model},
            "response": {"text": response},
            "source": "llm_test"
        })
        
        return jsonify({
            'success': True,
            'model': model,
            'response': response,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template('components/index.html'), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/favicon.ico')
def favicon():
    try:
        return send_from_directory('static/images', 'aliza-icon.jpg')
    except Exception:
        return ('', 204)


@app.route('/charts/<path:filename>')
def serve_chart(filename):
    """Serve chart images generated by KB and SQL servers."""
    try:
        # Charts are stored in charts/kb/ or charts/sql/ directories
        charts_base = Path('charts')
        
        # Handle both flat paths (bar_xxx.png) and nested paths (kb/bar_xxx.png)
        file_path = charts_base / filename
        
        if file_path.exists() and file_path.is_file():
            return send_from_directory(file_path.parent, file_path.name)
        
        # Try kb subdirectory
        kb_path = charts_base / 'kb' / filename
        if kb_path.exists():
            return send_from_directory(kb_path.parent, kb_path.name)
        
        # Try sql subdirectory
        sql_path = charts_base / 'sql' / filename
        if sql_path.exists():
            return send_from_directory(sql_path.parent, sql_path.name)
        
        logger.warning(f"Chart not found: {filename}")
        return ('', 404)
    except Exception as e:
        logger.error(f"Error serving chart {filename}: {e}")
        return ('', 404)


if __name__ == '__main__':
    import atexit
    from architecture.telemetry import shutdown_telemetry
    
    # Register shutdown hook to flush telemetry
    atexit.register(shutdown_telemetry)
    
    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("✅ Database tables created")
    
    # Check API key
    try:
        services = get_service_manager()
        services.get_llm_service()._get_api_key()
        api_key_status = "✅ Set"
    except ValueError:
        api_key_status = "❌ Missing"
    
    # Initialize router and log status
    router = get_router()
    registry = ToolRegistry()
    tools = registry.list_tools()
    
    logger.info("=" * 50)
    logger.info("🚀 LLM MCP Router - Flask Application")
    logger.info("=" * 50)
    logger.info(f"GROQ API Key: {api_key_status}")
    logger.info(f"Tools loaded: {len(tools)}")
    for tool_name in tools:
        tool = registry.get_tool(tool_name)
        if tool:
            logger.info(f"  📦 {tool_name}: {tool.schema.description[:50]}...")
    logger.info("=" * 50)
    
    if api_key_status == "❌ Missing":
        logger.warning("🔑 Set GROQ_API_KEY in .env for AI features")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
