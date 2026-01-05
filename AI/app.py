"""
AI Orchestration Server

Owns the heavy /chat orchestration, file processing, and safety gates.
This server is intended to be called via the Web gateway (reverse-proxied),
but routes and handler function names are kept identical to the monolith.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, current_user, login_required
from flask_migrate import Migrate
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

AI_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AI_ROOT.parent

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=AI_ROOT / ".env", override=False)
except Exception:
    pass

# Keep relative paths scoped under AI/ (prevents top-level artifacts).
try:
    os.chdir(AI_ROOT)
except Exception:
    pass

from architecture.conversation_memory import get_conversation_memory
from architecture.document_manager import get_document_manager
from architecture.hybrid_router import HybridRouter, HybridRouterConfig
from architecture.registry import ToolRegistry
from auth import auth_bp
from auth.models import User, db
from auth.routes import init_auth_limiter
from core.error_handlers import register_error_handlers
from core.llm_security import get_input_sanitizer, get_output_validator
from architecture.pre_router_guard import get_pre_router_guard
from shared_utils import MessageFormatter, get_service_manager, logger


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _normalize_database_url(database_url: str, base_dir: Path) -> str:
    database_url = (database_url or "").strip()
    if not database_url:
        return ""

    # Flask-SQLAlchemy resolves relative SQLite paths under app.instance_path, which breaks
    # our tier-relative layout (e.g. `sqlite:///../API/data/app.db`). Normalize to absolute.
    if not database_url.startswith("sqlite:///"):
        return database_url

    url_no_prefix = database_url[len("sqlite:///") :]
    path_part, sep, query = url_no_prefix.partition("?")

    # Handle :memory: and other special sqlite targets.
    if path_part in {":memory:", ""}:
        return database_url

    # Absolute Windows path (e.g. C:/... or C:\...)
    if len(path_part) >= 2 and path_part[1] == ":":
        return database_url

    # Absolute POSIX path (sqlite:////...); keep as-is.
    if database_url.startswith("sqlite:////"):
        return database_url

    abs_path = (base_dir / Path(path_part)).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = f"sqlite:///{abs_path.as_posix()}"
    if sep:
        normalized += f"?{query}"
    return normalized


AI_PORT = _get_env_int("AI_PORT", 5003)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = Flask(
    __name__,
    template_folder=str(REPO_ROOT / "Web" / "templates"),
    static_folder=str(REPO_ROOT / "Web" / "static"),
)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-this"))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

database_url = (os.getenv("DATABASE_URL") or "").strip()
if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = _normalize_database_url(database_url, AI_ROOT)
else:
    data_dir = REPO_ROOT / "API" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{(data_dir / 'app.db').as_posix()}"

db.init_app(app)
Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access this page."


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


CORS(app)
app.register_blueprint(auth_bp)
register_error_handlers(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"],
    strategy="fixed-window",
)
init_auth_limiter(limiter)

csrf = CSRFProtect(app)

force_https = os.getenv("FLASK_ENV", "development") != "development"
Talisman(
    app,
    force_https=force_https,
    strict_transport_security=True,
    content_security_policy={
        "default-src": "'self'",
        "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "ajax.googleapis.com", "cdnjs.cloudflare.com"],
        "style-src": ["'self'", "'unsafe-inline'", "fonts.googleapis.com", "cdnjs.cloudflare.com"],
        "img-src": ["'self'", "data:", "https:"],
        "font-src": ["'self'", "data:", "fonts.gstatic.com"],
        "connect-src": ["'self'"],
    },
    content_security_policy_nonce_in=[],
)


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Upload configuration (shared state under API/state/uploads by default)
default_uploads = REPO_ROOT / "API" / "state" / "uploads"
upload_folder_env = (os.getenv("UPLOAD_FOLDER") or os.getenv("UPLOADS_DIR") or "").strip()
UPLOAD_FOLDER = Path(upload_folder_env) if upload_folder_env else default_uploads
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MAX_UPLOAD_AGE_HOURS = _get_env_int("UPLOAD_MAX_AGE_HOURS", 24)
MAX_UPLOAD_FILES = _get_env_int("UPLOAD_MAX_FILES", 200)


def cleanup_upload_folder(max_age_hours: int = MAX_UPLOAD_AGE_HOURS, max_files: int = MAX_UPLOAD_FILES) -> None:
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


ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
ALLOWED_DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def _append_json_log(entry: Dict[str, Any], log_file: Path = AI_ROOT / "logs" / "chatbot.log") -> None:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def initialize_session():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "processed_files" not in session:
        session["processed_files"] = []


# =========================
# Async loop bridge (shared)
# =========================

_async_loop: Optional[asyncio.AbstractEventLoop] = None
_async_thread: Optional[threading.Thread] = None
_async_lock = threading.Lock()


def _get_async_loop() -> asyncio.AbstractEventLoop:
    global _async_loop, _async_thread
    with _async_lock:
        if _async_loop is not None and _async_loop.is_running():
            return _async_loop

        loop = asyncio.new_event_loop()

        def _run(loop_to_run: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop_to_run)
            loop_to_run.run_forever()

        thread = threading.Thread(target=_run, args=(loop,), daemon=True)
        thread.start()

        start = time.time()
        while time.time() - start < 5:
            if loop.is_running():
                break
            time.sleep(0.05)

        _async_loop = loop
        _async_thread = thread
        return _async_loop


def run_async(coro):
    loop = _get_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


# =========================
# Hybrid Router (Rasa + LLM)
# =========================

_router: Optional[HybridRouter] = None


def get_router() -> HybridRouter:
    global _router
    if _router is None:
        config = HybridRouterConfig(
            enable_rasa_layer=True,
            use_predefined_responses=True,
            rasa_confidence_threshold=0.7,
        )
        _router = HybridRouter(config)
        logger.info("✅ Hybrid Router initialized (Rasa + LLM)")
    return _router


# =========================
# Routes owned by AI tier
# =========================


@app.route("/chat", methods=["POST"])
@csrf.exempt  # keep behavior (no CSRF required)
@limiter.limit("20 per minute")
@login_required
def chat():
    """
    Main chat endpoint - Routes messages through the LLM Router.
    Kept compatible with existing frontend response shapes.
    """
    try:
        data = request.get_json(force=True) or {}
        message: str = data.get("message", "").strip()

        if not message:
            return jsonify([{"text": "Please enter a message."}])

        sanitizer = get_input_sanitizer()
        sanitization_result = sanitizer.sanitize(message)
        if not sanitization_result.is_safe:
            logger.warning(f"Blocked unsafe input: {sanitization_result.blocked_reason}")
            return (
                jsonify(
                    [
                        {
                            "text": "⚠️ Your message contains potentially unsafe content and cannot be processed. Please rephrase your question."
                        }
                    ]
                ),
                400,
            )

        message = sanitization_result.sanitized_input

        # User context
        try:
            user_id = getattr(current_user, "id", None)
            sender_id = f"user-{int(user_id)}" if user_id is not None else "flask_user"
        except Exception:
            user_id = None
            sender_id = "flask_user"

        metadata: Dict[str, Any] = data.get("metadata", {}) or {}

        # Inject role + employee_id for tool RBAC
        try:
            user_role = getattr(current_user, "role", None)
            if user_role:
                metadata["role"] = str(user_role)
            if user_id:
                metadata["employee_id"] = str(user_id)
        except Exception:
            pass

        # Language detection/translation
        translation_service = get_service_manager().get_translation_service()
        detected_language, confidence = translation_service.detect_language(message)
        direction = "rtl" if detected_language == "ar" else "ltr"

        metadata["original_language"] = detected_language
        metadata["original_message"] = message
        metadata["language_confidence"] = confidence
        metadata["text_direction"] = direction

        # SECURITY: Pre-router guard BEFORE translation
        guard = get_pre_router_guard()
        guard_result = guard.check(message, detected_language)
        if not guard_result.allowed:
            logger.warning(f"Blocked dangerous input (pre-router guard): category={guard_result.category}")
            return jsonify([{"text": guard_result.refusal_message, "metadata": {"tool": "blocked", "language": detected_language, "direction": direction}}]), 200

        processed_message = message
        if detected_language == "ar":
            processed_message = translation_service.translate_arabic_to_english(message)
            logger.info(f"🔄 Translated AR→EN: {message[:50]}... → {processed_message[:50]}...")

        router = get_router()
        result = run_async(router.process_message(processed_message, context=metadata, sender_id=sender_id))

        response_text = result.get("response", "I couldn't process your request.")

        output_validator = get_output_validator()
        is_safe_output, filtered_output = output_validator.validate_output(response_text)
        if not is_safe_output:
            logger.warning(f"Filtered unsafe LLM output: {response_text[:100]}")
            response_text = filtered_output

        adaptive_card = None
        result_data = result.get("result", {})
        if not isinstance(result_data, dict):
            result_data = {}

        if result.get("type") == "card":
            adaptive_card = result.get("payload", {})
        elif result_data.get("type") == "adaptive_card":
            adaptive_card = result_data.get("card")
        elif isinstance(result_data.get("data"), dict) and result_data["data"].get("type") == "adaptive_card":
            adaptive_card = result_data["data"].get("card")
        else:
            tool_calls = result_data.get("tool_calls", [])
            for call in tool_calls:
                call_result = call.get("result", {})
                data_block = call_result.get("data", call_result)
                if isinstance(data_block, dict) and data_block.get("type") == "adaptive_card":
                    adaptive_card = data_block.get("card")
                    break
                if isinstance(data_block, str):
                    try:
                        parsed = json.loads(data_block)
                        if isinstance(parsed, dict) and parsed.get("type") == "adaptive_card":
                            adaptive_card = parsed.get("card")
                            break
                    except Exception:
                        pass

        related_questions = result_data.get("related_questions", [])

        # Translate output for Arabic requests
        if detected_language == "ar":
            if response_text:
                try:
                    original_text = response_text
                    response_text = translation_service.translate_english_to_arabic(original_text)
                    logger.info(f"🔄 Translated EN→AR: {original_text[:50]}... → {response_text[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to translate response text: {e}")

            if adaptive_card:
                try:
                    adaptive_card = translation_service.translate_adaptive_card(adaptive_card)
                except Exception as trans_err:
                    logger.warning(f"Failed to translate adaptive card: {trans_err}")

            if related_questions and isinstance(related_questions, list):
                try:
                    translated_related = []
                    for q in related_questions:
                        if q.get("title") and q.get("prompt"):
                            translated_related.append(
                                {
                                    "title": translation_service.translate_english_to_arabic(q["title"]),
                                    "prompt": translation_service.translate_english_to_arabic(q["prompt"]),
                                }
                            )
                    related_questions = translated_related
                except Exception as e:
                    logger.warning(f"Failed to translate related questions: {e}")

        # Persist messages (same behavior as monolith)
        try:
            from auth.models import ChatMessage, ChatSession
            import uuid

            if user_id:
                active_session = ChatSession.query.filter_by(user_id=user_id, is_active=True).first()
                if not active_session:
                    active_session = ChatSession(session_id=str(uuid.uuid4())[:8], user_id=user_id, name="New Chat", is_active=True)
                    db.session.add(active_session)
                    db.session.commit()

                user_msg = ChatMessage(session_id=active_session.id, role="user", content=message, metadata_json='{"language": "' + detected_language + '"}')
                user_msg.msg_metadata = {"language": detected_language, "processed": processed_message}
                db.session.add(user_msg)

                assistant_metadata = {"routing": result.get("routing", {})}
                if adaptive_card:
                    assistant_metadata["type"] = "adaptive_card"
                    assistant_metadata["card"] = adaptive_card
                    result_metadata = {}
                    if result_data.get("metadata"):
                        result_metadata = result_data["metadata"]
                    else:
                        for call in result_data.get("tool_calls", []):
                            call_result = call.get("result", {})
                            data_block = call_result.get("data", call_result)
                            if isinstance(data_block, dict) and data_block.get("metadata"):
                                result_metadata = data_block["metadata"]
                                break
                    assistant_metadata["card_metadata"] = result_metadata

                if isinstance(result_data, dict) and result_data.get("image_data"):
                    assistant_metadata["image_data"] = result_data.get("image_data")

                saved_chart_path = None
                for call in result_data.get("tool_calls", []):
                    call_result = call.get("result", {})
                    data_block = call_result.get("data", call_result)
                    if isinstance(data_block, dict) and data_block.get("chart"):
                        saved_chart_path = data_block.get("chart")
                        break
                    if isinstance(data_block, dict) and data_block.get("chart_path"):
                        saved_chart_path = data_block.get("chart_path")
                        break
                    if isinstance(data_block, str):
                        try:
                            parsed = json.loads(data_block)
                            if isinstance(parsed, dict) and parsed.get("chart"):
                                saved_chart_path = parsed.get("chart")
                                break
                        except Exception:
                            pass
                if not saved_chart_path and isinstance(result_data, dict):
                    saved_chart_path = result_data.get("chart")
                if saved_chart_path:
                    assistant_metadata["chart_path"] = saved_chart_path

                assistant_msg = ChatMessage(session_id=active_session.id, role="assistant", content=response_text, tool_name=result.get("tool"))
                assistant_msg.msg_metadata = assistant_metadata
                db.session.add(assistant_msg)

                if not active_session.preview:
                    active_session.preview = message[:100] if len(message) > 100 else message
                    if active_session.name == "New Chat":
                        try:
                            smart_name = run_async(_generate_smart_session_name(message, response_text))
                            active_session.name = smart_name
                        except Exception:
                            active_session.name = (message[:40] + "...") if len(message) > 40 else message
                db.session.commit()
        except Exception as db_err:
            logger.warning(f"Failed to save messages to DB: {db_err}")
            db.session.rollback()

        responses = []
        if adaptive_card:
            result_metadata = {}
            if result_data.get("metadata"):
                result_metadata = result_data["metadata"]
            elif isinstance(result_data.get("data"), dict) and isinstance(result_data["data"].get("metadata"), dict):
                result_metadata = result_data["data"]["metadata"]
            else:
                for call in result_data.get("tool_calls", []):
                    call_result = call.get("result", {})
                    data_block = call_result.get("data", call_result)
                    if isinstance(data_block, dict) and isinstance(data_block.get("metadata"), dict):
                        result_metadata = data_block["metadata"]
                        break
                    if isinstance(data_block, dict) and isinstance(data_block.get("data"), dict) and isinstance(data_block["data"].get("metadata"), dict):
                        result_metadata = data_block["data"]["metadata"]
                        break

            responses.append(
                {
                    "text": None,
                    "custom": {
                        "payload": "adaptiveCard",
                        "data": adaptive_card,
                        "metadata": {
                            "employee_id": metadata.get("employee_id", "1"),
                            "template": result_metadata.get("template", result.get("template", "")),
                            "language": detected_language,
                            "direction": direction,
                        },
                    },
                    "metadata": {"language": detected_language, "direction": direction, "tool": result.get("tool", "unknown")},
                }
            )

        if not adaptive_card:
            if "---MESSAGE_SPLIT---" in response_text:
                parts = response_text.split("---MESSAGE_SPLIT---")
                for part in parts:
                    if part.strip():
                        responses.append({"text": part.strip(), "metadata": {"language": detected_language, "direction": direction, "tool": result.get("tool", "unknown")}})
            else:
                responses.append({"text": response_text, "metadata": {"language": detected_language, "direction": direction, "tool": result.get("tool", "unknown")}})

        image_data = result_data.get("image_data")
        if not image_data:
            for call in result_data.get("tool_calls", []):
                call_result = call.get("result", {})
                data_block = call_result.get("data", call_result)
                if isinstance(data_block, dict) and data_block.get("image_data"):
                    image_data = data_block["image_data"]
                    break
                if isinstance(data_block, dict) and data_block.get("chart_path"):
                    chart_path = data_block["chart_path"]
                    image_data = {"image_url": chart_path, "title": (data_block.get("answer", "SQL Query Result") or "")[:100], "alt": "SQL Query Visualization"}
                    break

        if image_data and isinstance(image_data, dict) and image_data.get("image_url"):
            responses.append(
                {
                    "custom": {"payload": "visual_panel", "data": image_data},
                    "metadata": {"language": detected_language, "direction": direction, "tool": result.get("tool", "unknown")},
                }
            )

        if related_questions:
            responses.append(
                {
                    "custom": {"payload": "related_links", "data": {"items": [{"title": q.get("title", ""), "prompt": q.get("prompt", "")} for q in related_questions if q.get("title") and q.get("prompt")]}}
                }
            )

        timestamp = datetime.now().isoformat()
        for item in responses:
            _append_json_log(
                {
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
                    "response": item,
                }
            )

        # Eval metadata (opt-in) for DeepEval benchmarks.
        # Added after logging to avoid dumping large eval payloads into chat logs.
        if metadata.get("eval_mode") is True:
            tools_called: List[str] = []
            tool_calls = result_data.get("tool_calls", []) if isinstance(result_data, dict) else []
            if (not tool_calls) and isinstance(result_data, dict):
                top_tool_name = result.get("tool")
                if isinstance(top_tool_name, str) and "." in top_tool_name and top_tool_name != "code_execution":
                    tool_calls = [{"name": top_tool_name, "result": result_data}]
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tool_name = tc.get("name") or tc.get("tool") or tc.get("tool_name") or ""
                    if tool_name:
                        tools_called.append(str(tool_name))

            top_tool = result.get("tool")
            if top_tool and top_tool not in tools_called:
                tools_called.insert(0, str(top_tool))

            retrieval_context: List[str] = []
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_result = tc.get("result", {})
                    if not isinstance(tc_result, dict):
                        continue

                    data_block = tc_result.get("data", tc_result)
                    payload: Any = None
                    if isinstance(data_block, dict):
                        payload = data_block
                    elif isinstance(data_block, str):
                        try:
                            payload = json.loads(data_block)
                        except Exception:
                            payload = None

                    if not isinstance(payload, dict):
                        continue

                    rc = payload.get("retrieval_context")
                    if isinstance(rc, list):
                        retrieval_context.extend([str(item) for item in rc if item])

                    inner = payload.get("data")
                    if isinstance(inner, dict):
                        inner_rc = inner.get("retrieval_context")
                        if isinstance(inner_rc, list):
                            retrieval_context.extend([str(item) for item in inner_rc if item])

                    nested_response = payload.get("response")
                    if isinstance(nested_response, str):
                        try:
                            nested_payload = json.loads(nested_response)
                            if isinstance(nested_payload, dict):
                                nested_rc = nested_payload.get("retrieval_context")
                                if isinstance(nested_rc, list):
                                    retrieval_context.extend([str(item) for item in nested_rc if item])
                        except Exception:
                            pass

            eval_metadata = {
                "tools_called": tools_called,
                "routing": result.get("routing", {}),
                "retrieval_context": retrieval_context,
                "safety_checks": {
                    "input_blocked": not sanitization_result.is_safe,
                    "output_filtered": not is_safe_output,
                },
            }

            for item in responses:
                if not isinstance(item, dict):
                    continue
                item_metadata = item.get("metadata")
                if not isinstance(item_metadata, dict):
                    item["metadata"] = {}
                    item_metadata = item["metadata"]
                item_metadata["eval"] = eval_metadata

        return jsonify(responses)
    except Exception as e:
        logger.error(f"/chat error: {e}", exc_info=True)
        return jsonify([{"text": f"❌ Error: {str(e)}"}]), 500


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    """Handle file uploads for image/document processing."""
    initialize_session()
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = app.config["UPLOAD_FOLDER"] / filename
        file.save(filepath)
        logger.info(f"📎 File uploaded: {file.filename} → {filepath}")

        file_type = request.form.get("file_type", "auto")
        question = request.form.get("question", "")
        add_to_kb = request.form.get("add_to_kb", "false").lower() == "true"

        if file_type == "auto":
            file_type = "image" if is_image_file(file.filename) else "document"
        if not question:
            question = "Describe this image in detail" if file_type == "image" else "Summarize the main points"

        services = get_service_manager()
        if file_type == "image":
            result_text = services.get_file_service().process_image(filepath)
        else:
            result_text = services.get_file_service().process_document(filepath)

        if not result_text or not result_text.strip():
            result_text = (
                f"File '{file.filename}' was uploaded but could not be analyzed.\n"
                "- Non-English documents (limited support)\n"
                "- Scanned PDFs without OCR\n"
                "- Empty or corrupted files\n"
            )

        result_text = MessageFormatter.clean_markdown_text(result_text)

        kb_status = None
        if add_to_kb and file_type == "document":
            try:
                doc_manager = get_document_manager()
                move_result = doc_manager.move_to_documents(filename, source_dir=app.config["UPLOAD_FOLDER"])
                kb_status = "added_to_kb" if move_result.get("success") else "kb_failed"
            except Exception as kb_err:
                logger.error(f"KB error: {kb_err}")
                kb_status = "kb_error"

        file_info = {
            "type": file_type,
            "name": file.filename,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "result": result_text[:500] + "..." if len(result_text) > 500 else result_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        session["processed_files"].append(file_info)
        if len(session["processed_files"]) > 3:
            session["processed_files"] = session["processed_files"][-3:]
        session.modified = True

        cleanup_upload_folder()

        _append_json_log(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "bot_uttered",
                "sender_id": session.get("session_id", "flask_user"),
                "request": f"/upload {file_type}",
                "metadata": {"filename": file.filename, "kb_status": kb_status},
                "response": {"text": result_text, "file_type": file_type},
                "source": "flask_upload",
            }
        )

        response_data = {"success": True, "result": result_text, "file_type": file_type, "filename": file.filename}
        if request.form.get("eval_mode") == "true":
            try:
                file_context = services.get_file_service().get_last_file_context()
                response_data["source_text"] = file_context.get("text", "") if isinstance(file_context, dict) else ""
            except Exception:
                response_data["source_text"] = ""

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500


@app.route("/clear_chat", methods=["POST"])
@login_required
def clear_chat():
    session["chat_history"] = []
    session["processed_files"] = []
    session.modified = True
    try:
        user_id = getattr(current_user, "id", None)
        sender_id = f"user-{int(user_id)}" if user_id is not None else "flask_user"
        try:
            get_router().clear_session(sender_id)
        except Exception:
            pass
        try:
            memory = run_async(get_conversation_memory())
            run_async(memory.clear_context(sender_id))
        except Exception:
            pass
    except Exception:
        pass
    return jsonify({"success": True})


@app.route("/api/test_model", methods=["POST"])
@login_required
def test_model():
    """Test a specific model with a query."""
    try:
        data = request.get_json() or {}
        model = data.get("model", "llama-3.3-70b-versatile")
        test_query = data.get("query", "Hello! Please respond with a short greeting.")

        from shared_utils import Config

        if model not in Config.AVAILABLE_MODELS:
            return jsonify({"error": "Invalid model"}), 400

        services = get_service_manager()
        messages = [{"role": "user", "content": test_query}]
        start_time = datetime.now()
        response = services.get_llm_service().generate_text(messages=messages, model=model, max_tokens=100, temperature=0.7, timeout=10)
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()
        _append_json_log(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "bot_uttered",
                "sender_id": session.get("session_id", "flask_user"),
                "request": f"/api/test_model: {test_query}",
                "metadata": {"model": model},
                "response": {"text": response},
                "source": "llm_test",
            }
        )

        return jsonify({"success": True, "model": model, "response": response, "response_time": response_time, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _generate_smart_session_name(user_message: str, bot_response: str) -> str:
    services = get_service_manager()
    prompt = (
        "Generate a short, descriptive name (2-5 words) for this conversation based on the user's first message.\n\n"
        f"User: {user_message}\nAssistant: {bot_response}\n\n"
        "Return ONLY the name, no quotes."
    )
    try:
        name = services.get_llm_service().generate_text(messages=[{"role": "user", "content": prompt}], max_tokens=20, temperature=0.3, timeout=10)
        return (name or "New Chat").strip().strip('"').strip("'")[:50] or "New Chat"
    except Exception:
        return "New Chat"


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logger.info("AI server: database tables ensured")
    # Disable the reloader: we `chdir()` into AI/, and the reloader re-executes using
    # the original relative script path (e.g. `AI/app.py`), which becomes `AI/AI/app.py`.
    app.run(host="0.0.0.0", port=AI_PORT, debug=True, threaded=True, use_reloader=False)
