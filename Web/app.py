"""
Web Server (Gateway + UI)

- Serves templates/static
- Handles browser-facing auth/UI routes
- Proxies expensive and API routes to downstream tier servers while preserving:
  - route paths
  - handler function names
  - response bodies/status codes
  - Set-Cookie headers (session continuity)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, login_required
from flask_migrate import Migrate
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect

# Paths
WEB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEB_ROOT.parent
AI_ROOT = REPO_ROOT / "AI"

# Ensure shared code (moved under AI/) is importable.
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

# Load environment variables (Web/.env only; no repo-root .env).
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=WEB_ROOT / ".env", override=False)
except Exception:
    pass

# Keep relative paths scoped under Web/ (prevents top-level artifacts).
try:
    os.chdir(WEB_ROOT)
except Exception:
    pass

from auth import auth_bp
from auth.models import User, db
from auth.rbac import role_required
from auth.routes import init_auth_limiter
from core.error_handlers import register_error_handlers
from shared_utils import logger


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


WEB_PORT = _get_env_int("WEB_PORT", _get_env_int("FLASK_PORT", 5001))
API_SERVER_URL = (os.getenv("API_SERVER_URL") or "http://127.0.0.1:5002").rstrip("/")
AI_SERVER_URL = (os.getenv("AI_SERVER_URL") or "http://127.0.0.1:5003").rstrip("/")


app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)

# Configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-this"))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Database configuration (single shared DB file owned under API/data/)
database_url = (os.getenv("DATABASE_URL") or "").strip()
if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = _normalize_database_url(database_url, WEB_ROOT)
else:
    data_dir = REPO_ROOT / "API" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{(data_dir / 'app.db').as_posix()}"

# Initialize extensions
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
        "script-src": [
            "'self'",
            "'unsafe-inline'",
            "'unsafe-eval'",
            "ajax.googleapis.com",
            "cdnjs.cloudflare.com",
        ],
        "style-src": ["'self'", "'unsafe-inline'", "fonts.googleapis.com", "cdnjs.cloudflare.com"],
        "img-src": ["'self'", "data:", "https:"],
        "font-src": ["'self'", "data:", "fonts.gstatic.com"],
        "connect-src": ["'self'"],
    },
    content_security_policy_nonce_in=[],
)


@app.after_request
def set_security_headers(response: Response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def _hop_by_hop_headers() -> set[str]:
    return {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }


def _iter_request_headers() -> Iterable[Tuple[str, str]]:
    for key, value in request.headers.items():
        lk = key.lower()
        if lk in _hop_by_hop_headers():
            continue
        if lk == "host":
            continue
        # requests computes content-length from body
        if lk == "content-length":
            continue
        yield key, value


def _drop_header(headers: Dict[str, str], header_name: str) -> None:
    # Flask gives canonical casing; requests header dict is case-insensitive at send-time,
    # but we store as normal dict, so remove by lowercasing keys.
    target = header_name.lower()
    for key in list(headers.keys()):
        if key.lower() == target:
            headers.pop(key, None)


def _form_tuples() -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    try:
        for key, values in request.form.lists():
            for value in values:
                items.append((key, value))
    except Exception:
        for key in request.form.keys():
            items.append((key, request.form.get(key, "")))
    return items


def _proxy(target_base: str) -> Response:
    target_url = f"{target_base}{request.path}"
    if request.query_string:
        target_url = f"{target_url}?{request.query_string.decode('utf-8', errors='replace')}"

    headers: Dict[str, str] = {k: v for (k, v) in _iter_request_headers()}

    # Special handling for multipart/form-data: CSRF validation and Flask parsing consume
    # the raw stream, so request.get_data(cache=False) becomes empty. Rebuild multipart
    # from parsed form/files so uploads proxy correctly.
    if request.files:
        _drop_header(headers, "Content-Type")
        files = []
        for field_name, storage in request.files.items(multi=True):
            try:
                storage.stream.seek(0)
            except Exception:
                pass
            files.append((field_name, (storage.filename, storage.stream, storage.mimetype)))

        upstream = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=_form_tuples(),
            files=files,
            allow_redirects=False,
            timeout=300,
        )
    else:
        body = request.get_data(cache=False)
        upstream = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body,
            allow_redirects=False,
            timeout=300,
        )

    resp = Response(upstream.content, status=upstream.status_code)

    # Copy headers (preserve Set-Cookie and Location)
    excluded = _hop_by_hop_headers() | {"content-encoding", "content-length"}
    for key, value in upstream.headers.items():
        if key.lower() in excluded:
            continue
        resp.headers[key] = value

    # Preserve multiple Set-Cookie headers if present
    try:
        set_cookies = upstream.raw.headers.get_all("Set-Cookie")  # type: ignore[attr-defined]
    except Exception:
        set_cookies = None
    if set_cookies:
        resp.headers.pop("Set-Cookie", None)
        for cookie in set_cookies:
            resp.headers.add("Set-Cookie", cookie)

    return resp


# =========================
# UI Routes (owned by Web)
# =========================


def initialize_session():
    from flask import session

    if "chat_history" not in session:
        session["chat_history"] = []
    if "processed_files" not in session:
        session["processed_files"] = []


@app.route("/")
@login_required
def index():
    initialize_session()
    from flask import session

    return render_template("components/index.html", chat_history=session.get("chat_history", []))


@app.route("/login")
def login_redirect():
    return redirect(url_for("auth.login"))


@app.route("/logout")
def logout_redirect():
    return redirect(url_for("auth.logout"))


@app.route("/admin")
@role_required("admin")
def admin_dashboard():
    users = User.query.all()
    return render_template("admin/dashboard.html", users=users)


@app.route("/favicon.ico")
def favicon():
    try:
        return send_from_directory(str(WEB_ROOT / "static" / "images"), "aliza-icon.jpg")
    except Exception:
        return ("", 204)


@app.route("/charts/<path:filename>")
def serve_chart(filename: str):
    charts_base = WEB_ROOT / "charts"
    file_path = charts_base / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(file_path.parent), file_path.name)
    kb_path = charts_base / "kb" / filename
    if kb_path.exists():
        return send_from_directory(str(kb_path.parent), kb_path.name)
    sql_path = charts_base / "sql" / filename
    if sql_path.exists():
        return send_from_directory(str(sql_path.parent), sql_path.name)
    return ("", 404)


# =========================
# Proxy Routes (gateway)
# =========================


@app.route("/chat", methods=["POST"])
@csrf.exempt  # keep path+behavior (no CSRF required)
@limiter.limit("20 per minute")
@login_required
def chat():
    return _proxy(AI_SERVER_URL)


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    return _proxy(AI_SERVER_URL)


@app.route("/clear_chat", methods=["POST"])
@login_required
def clear_chat():
    return _proxy(AI_SERVER_URL)


@app.route("/health")
def health():
    return _proxy(API_SERVER_URL)


@app.route("/healthz")
def healthz():
    return _proxy(API_SERVER_URL)


@app.route("/api/tools", methods=["GET"])
@login_required
def list_tools():
    return _proxy(API_SERVER_URL)


@app.route("/api/models", methods=["GET"])
@login_required
def get_available_models():
    return _proxy(API_SERVER_URL)


@app.route("/api/model_preference", methods=["POST"])
@login_required
def set_model_preference():
    return _proxy(API_SERVER_URL)


@app.route("/api/model_preference", methods=["GET"])
@login_required
def get_model_preference():
    return _proxy(API_SERVER_URL)


@app.route("/api/test_model", methods=["POST"])
@login_required
def test_model():
    return _proxy(AI_SERVER_URL)


@app.route("/api/sessions", methods=["GET"])
@login_required
def list_sessions():
    return _proxy(API_SERVER_URL)


@app.route("/api/sessions", methods=["POST"])
@login_required
def create_session():
    return _proxy(API_SERVER_URL)


@app.route("/api/sessions/<session_id>", methods=["PUT"])
@login_required
def rename_session(session_id: str):
    return _proxy(API_SERVER_URL)


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@login_required
def delete_session_api(session_id: str):
    return _proxy(API_SERVER_URL)


@app.route("/api/sessions/<session_id>/switch", methods=["POST"])
@login_required
def switch_session_api(session_id: str):
    return _proxy(API_SERVER_URL)


@app.route("/api/sessions/current", methods=["GET"])
@login_required
def get_current_session():
    return _proxy(API_SERVER_URL)


@app.route("/api/documents/status", methods=["GET"])
@login_required
def get_documents_status():
    return _proxy(API_SERVER_URL)


@app.route("/api/documents/move-uploads", methods=["POST"])
@login_required
def move_uploads_to_documents():
    return _proxy(API_SERVER_URL)


@app.route("/api/documents/index", methods=["POST"])
@login_required
def trigger_document_indexing():
    return _proxy(API_SERVER_URL)


@app.route("/api/documents/process", methods=["POST"])
@login_required
def process_and_index_documents():
    return _proxy(API_SERVER_URL)


@app.route("/api/documents/list", methods=["GET"])
@login_required
def list_documents():
    return _proxy(API_SERVER_URL)


@app.route("/admin/api/users", methods=["GET"])
@role_required("admin")
def admin_list_users():
    return _proxy(API_SERVER_URL)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logger.info("Web server: database tables ensured")
    # Disable the reloader: we `chdir()` into Web/, and the reloader re-executes using
    # the original relative script path (e.g. `Web/app.py`), which becomes `Web/Web/app.py`.
    app.run(host="0.0.0.0", port=WEB_PORT, debug=True, threaded=True, use_reloader=False)
