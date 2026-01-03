"""
API Server (non-LLM business APIs)

Owns JSON endpoints that power the UI (sessions, models, documents, admin lists, health).
This server is intended to be called via the Web gateway (reverse-proxied), but can also
run standalone for debugging.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, current_user, login_required
from flask_migrate import Migrate
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect

API_ROOT = Path(__file__).resolve().parent
REPO_ROOT = API_ROOT.parent
AI_ROOT = REPO_ROOT / "AI"

# Ensure shared code (moved under AI/) is importable.
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=API_ROOT / ".env", override=False)
except Exception:
    pass

# Keep relative paths scoped under API/ (prevents top-level artifacts).
try:
    os.chdir(API_ROOT)
except Exception:
    pass

from architecture.document_manager import get_document_manager
from architecture.registry import ToolRegistry
from auth import auth_bp
from auth.models import User, db
from auth.rbac import role_required
from auth.routes import init_auth_limiter
from core.error_handlers import register_error_handlers
from shared_utils import get_service_manager, logger


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
    # our tier-relative layout (e.g. `sqlite:///data/app.db`). Normalize to absolute.
    if not database_url.startswith("sqlite:///"):
        return database_url

    url_no_prefix = database_url[len("sqlite:///") :]
    path_part, sep, query = url_no_prefix.partition("?")

    if path_part in {":memory:", ""}:
        return database_url

    if len(path_part) >= 2 and path_part[1] == ":":
        return database_url

    if database_url.startswith("sqlite:////"):
        return database_url

    abs_path = (base_dir / Path(path_part)).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = f"sqlite:///{abs_path.as_posix()}"
    if sep:
        normalized += f"?{query}"
    return normalized


API_PORT = _get_env_int("API_PORT", 5002)

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
    app.config["SQLALCHEMY_DATABASE_URI"] = _normalize_database_url(database_url, API_ROOT)
else:
    data_dir = API_ROOT / "data"
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
Talisman(app, force_https=force_https, strict_transport_security=True)


@app.route("/health")
def health():
    """Health check endpoint (kept compatible with existing response shape)."""
    try:
        services = get_service_manager()
        services.get_llm_service()._get_api_key()
        api_key_status = True
    except ValueError:
        api_key_status = False

    registry = ToolRegistry()
    tools = registry.list_tools()

    return jsonify(
        {
            "status": "healthy",
            "architecture": "LLM MCP Router",
            "groq_api": "configured" if api_key_status else "missing",
            "tools_loaded": len(tools),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
    )


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True}), 200


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
                schemas.append(
                    {"name": schema.name, "description": schema.description, "examples": schema.examples[:3] if schema.examples else []}
                )
        return jsonify({"success": True, "tools": schemas})
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================
# SESSION MANAGEMENT API ENDPOINTS (DB-backed)
# =============================================


@app.route("/api/sessions", methods=["GET"])
@login_required
def list_sessions():
    """List all chat sessions for the current user."""
    try:
        from auth.models import ChatSession
        import uuid

        user_id = current_user.id
        sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).all()
        active_session = ChatSession.query.filter_by(user_id=user_id, is_active=True).first()

        if not active_session:
            active_session = ChatSession(session_id=str(uuid.uuid4())[:8], user_id=user_id, name="New Chat", is_active=True)
            db.session.add(active_session)
            db.session.commit()
            sessions = [active_session] + sessions

        return jsonify({"success": True, "sessions": [s.to_dict() for s in sessions], "active_session_id": active_session.session_id})
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

        ChatSession.query.filter_by(user_id=user_id, is_active=True).update({"is_active": False})
        new_session = ChatSession(session_id=str(uuid.uuid4())[:8], user_id=user_id, name=name, is_active=True)
        db.session.add(new_session)
        db.session.commit()

        return jsonify({"success": True, "session": new_session.to_dict()})
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["PUT"])
@login_required
def rename_session(session_id: str):
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
        return jsonify({"success": True, "session": session_obj.to_dict()})
    except Exception as e:
        logger.error(f"Error renaming session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@login_required
def delete_session_api(session_id: str):
    """Delete a chat session."""
    try:
        from auth.models import ChatMessage, ChatSession
        import uuid

        user_id = current_user.id
        session_obj = ChatSession.query.filter_by(session_id=session_id, user_id=user_id).first()
        if not session_obj:
            return jsonify({"success": False, "error": "Session not found"}), 404

        was_active = session_obj.is_active
        ChatMessage.query.filter_by(session_id=session_obj.id).delete()
        db.session.delete(session_obj)
        db.session.commit()

        if was_active:
            ChatSession.query.filter_by(user_id=user_id, is_active=True).update({"is_active": False})
            remaining = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).first()
            if remaining:
                remaining.is_active = True
            else:
                new_session = ChatSession(session_id=str(uuid.uuid4())[:8], user_id=user_id, name="New Chat", is_active=True)
                db.session.add(new_session)
            db.session.commit()

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>/switch", methods=["POST"])
@login_required
def switch_session_api(session_id: str):
    """Switch to a different chat session and load its messages."""
    try:
        from auth.models import ChatMessage, ChatSession

        user_id = current_user.id
        target_session = ChatSession.query.filter_by(session_id=session_id, user_id=user_id).first()
        if not target_session:
            return jsonify({"success": False, "error": "Session not found"}), 404

        ChatSession.query.filter_by(user_id=user_id, is_active=True).update({"is_active": False})
        target_session.is_active = True
        db.session.commit()

        messages = ChatMessage.query.filter_by(session_id=target_session.id).order_by(ChatMessage.created_at.asc()).all()
        return jsonify({"success": True, "session_id": session_id, "messages": [m.to_llm_format() for m in messages]})
    except Exception as e:
        logger.error(f"Error switching session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/current", methods=["GET"])
@login_required
def get_current_session():
    """Get the current active session and its messages."""
    try:
        from auth.models import ChatMessage, ChatSession
        import uuid

        user_id = current_user.id
        active_session = ChatSession.query.filter_by(user_id=user_id, is_active=True).first()
        if not active_session:
            active_session = ChatSession(session_id=str(uuid.uuid4())[:8], user_id=user_id, name="New Chat", is_active=True)
            db.session.add(active_session)
            db.session.commit()

        messages = ChatMessage.query.filter_by(session_id=active_session.id).order_by(ChatMessage.created_at.asc()).all()
        return jsonify({"success": True, "session": active_session.to_dict(), "messages": [m.to_llm_format() for m in messages]})
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# Document Management Endpoints
# ============================================================================


@app.route("/api/documents/status", methods=["GET"])
@login_required
def get_documents_status():
    try:
        doc_manager = get_document_manager()
        status = doc_manager.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/move-uploads", methods=["POST"])
@login_required
def move_uploads_to_documents():
    try:
        doc_manager = get_document_manager()
        result = doc_manager.move_all_uploads()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error moving uploads: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/index", methods=["POST"])
@login_required
async def trigger_document_indexing():
    try:
        doc_manager = get_document_manager()
        result = await doc_manager.trigger_indexing()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error triggering indexing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/process", methods=["POST"])
@login_required
async def process_and_index_documents():
    try:
        doc_manager = get_document_manager()
        result = await doc_manager.process_and_index()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/list", methods=["GET"])
@login_required
def list_documents():
    try:
        doc_manager = get_document_manager()
        documents = doc_manager.list_documents()
        return jsonify({"documents": documents, "count": len(documents)})
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Model preference (session-backed)
# ============================================================================


@app.route("/api/models", methods=["GET"])
@login_required
def get_available_models():
    try:
        from shared_utils import Config

        return jsonify({"success": True, "models": Config.AVAILABLE_MODELS})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({"error": "Failed to retrieve models"}), 500


@app.route("/api/model_preference", methods=["POST"])
@login_required
def set_model_preference():
    try:
        data = request.get_json() or {}
        preferred_model = data.get("model")
        if not preferred_model:
            return jsonify({"error": "Model not specified"}), 400

        from shared_utils import Config

        if preferred_model not in Config.AVAILABLE_MODELS and preferred_model != "auto":
            return jsonify({"error": "Invalid model"}), 400

        from flask import session

        if "model_preferences" not in session:
            session["model_preferences"] = {}
        session["model_preferences"]["preferred_model"] = preferred_model
        session.modified = True

        return jsonify({"success": True, "message": f"Model preference set to: {preferred_model}", "preferred_model": preferred_model})
    except Exception as e:
        logger.error(f"Error setting model preference: {e}")
        return jsonify({"error": "Failed to set model preference"}), 500


@app.route("/api/model_preference", methods=["GET"])
@login_required
def get_model_preference():
    try:
        from flask import session

        preferences = session.get("model_preferences", {})
        preferred_model = preferences.get("preferred_model", "auto")
        return jsonify({"success": True, "preferred_model": preferred_model})
    except Exception as e:
        logger.error(f"Error getting model preference: {e}")
        return jsonify({"error": "Failed to get model preference"}), 500


# ============================================================================
# Admin JSON endpoints
# ============================================================================


@app.route("/admin/api/users", methods=["GET"])
@role_required("admin")
def admin_list_users():
    users = User.query.all()
    users_data = [{"id": u.id, "email": u.email, "role": u.role, "created_at": u.created_at.isoformat()} for u in users]
    return jsonify({"users": users_data})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logger.info("API server: database tables ensured")
    # Disable the reloader: we `chdir()` into API/, and the reloader re-executes using
    # the original relative script path (e.g. `API/app.py`), which becomes `API/API/app.py`.
    app.run(host="0.0.0.0", port=API_PORT, debug=True, threaded=True, use_reloader=False)
