"""FastAPI retrieval service exposing FAISS and Qdrant under one contract."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .embeddings import EMBEDDING_DIM
from .faiss_store import FaissVectorStore
from .qdrant_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------
load_dotenv()


@dataclass
class Settings:
    vector_api_key: str = ""
    default_backend: str = "auto"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    vector_base_url: str = "http://localhost:8001"
    request_body_limit: int = 32 * 1024
    qdrant_timeout: float = 2.0
    service_timeout: float = 2.0

    def __post_init__(self) -> None:
        self.vector_api_key = os.getenv("VECTOR_API_KEY", "")
        self.default_backend = os.getenv("DEFAULT_BACKEND", "auto").lower() or "auto"
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.vector_base_url = os.getenv("VECTOR_BASE_URL", "http://localhost:8001")
        self.request_body_limit = int(os.getenv("VECTOR_REQUEST_LIMIT", str(32 * 1024)))
        self.qdrant_timeout = float(os.getenv("QDRANT_TIMEOUT", "2.0"))
        self.service_timeout = float(os.getenv("SERVICE_TIMEOUT", "2.0"))


settings = Settings()

LOG_FILE = (Path(__file__).resolve().parent.parent / "secure_rag.log").resolve()
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("secure_rag")
if not logger.handlers:
    LOG_FILE.touch(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

if not settings.vector_api_key:
    logger.info("VECTOR_API_KEY is not set; proceeding without auth.")

# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
FAISS_STORE = FaissVectorStore(dim=EMBEDDING_DIM)
try:
    QDRANT_STORE: Optional[QdrantVectorStore] = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
        dim=EMBEDDING_DIM,
    )
except Exception as exc:  # pragma: no cover - defensive, logs for ops
    logger.warning("Failed to initialise Qdrant client: %s", exc)
    QDRANT_STORE = None

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Secure Retrieval Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_payload(request: Request, call_next):
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Trace-Id") or request.headers.get("X-Request-Id")
    if not request_id:
        request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover
        duration = (time.time() - start) * 1000
        logger.exception(
            "request error | request_id=%s path=%s duration_ms=%.2f error=%s",
            request_id,
            request.url.path,
            duration,
            exc,
        )
        raise
    duration = (time.time() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "request complete | request_id=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ---------------------------------------------------------------------------
# Models & dependencies
# ---------------------------------------------------------------------------
BackendOption = Literal["auto", "faiss", "qdrant"]


class UpsertItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=8192)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("metadata", pre=True)
    def ensure_metadata_dict(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be an object")
        return value


class UpsertRequest(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    backend: BackendOption = Field("auto")
    items: list[UpsertItem] = Field(..., min_items=1, max_items=32)


class RetrieveRequest(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(5, ge=1, le=20)
    backend: BackendOption = Field("auto")
    filters: Dict[str, Any] = Field(default_factory=dict)


class DeleteRequest(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    ids: list[str] = Field(..., min_items=1, max_items=64)
    backend: BackendOption = Field("auto")


class AuthContext(BaseModel):
    request_id: str


async def require_auth(
    request: Request,
    authorization: str = Header(default=""),
) -> AuthContext:
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token or token != settings.vector_api_key:
        logger.warning(
            "unauthorised request | request_id=%s path=%s",
            getattr(request.state, "request_id", "unknown"),
            request.url.path,
        )
        try:
            await request.body()
        except Exception:  # pragma: no cover - defensive cleanup
            pass
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return AuthContext(request_id=getattr(request.state, "request_id", uuid.uuid4().hex))


# ---------------------------------------------------------------------------
# Backend resolution helpers
# ---------------------------------------------------------------------------
def _qdrant_ready() -> bool:
    return QDRANT_STORE is not None and QDRANT_STORE.is_available()


def resolve_backend(requested: BackendOption) -> str:
    requested = requested or "auto"
    if requested == "qdrant":
        if not _qdrant_ready():
            raise HTTPException(status_code=503, detail="Qdrant backend unavailable")
        return "qdrant"
    if requested == "faiss":
        return "faiss"
    # auto selection
    if _qdrant_ready():
        return "qdrant"
    return "faiss"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upsert")
async def upsert_vectors(payload: UpsertRequest, ctx: AuthContext = Depends(require_auth)) -> Dict[str, Any]:
    backend = resolve_backend(payload.backend)
    namespace = payload.namespace
    items = [item.dict() for item in payload.items]
    try:
        if backend == "qdrant":
            assert QDRANT_STORE is not None
            count = QDRANT_STORE.upsert(namespace, items)
        else:
            count = FAISS_STORE.upsert(namespace, items)
    except Exception as exc:  # fallback for auto
        logger.exception(
            "upsert failed | request_id=%s backend=%s namespace=%s error=%s",
            ctx.request_id,
            backend,
            namespace,
            exc,
        )
        if payload.backend == "auto" and backend == "qdrant":
            count = FAISS_STORE.upsert(namespace, items)
            backend = "faiss"
        else:
            raise HTTPException(status_code=500, detail="Failed to upsert items")
    logger.info(
        "upsert ok | request_id=%s backend=%s namespace=%s count=%s",
        ctx.request_id,
        backend,
        namespace,
        count,
    )
    return {"ok": True, "count": count, "backend": backend}


@app.post("/retrieve")
async def retrieve_vectors(payload: RetrieveRequest, ctx: AuthContext = Depends(require_auth)) -> Dict[str, Any]:
    backend = resolve_backend(payload.backend)
    namespace = payload.namespace
    start = time.time()
    try:
        if backend == "qdrant":
            assert QDRANT_STORE is not None
            hits = QDRANT_STORE.retrieve(
                namespace,
                payload.query,
                payload.top_k,
                filters=payload.filters,
            )
        else:
            hits = FAISS_STORE.retrieve(namespace, payload.query, payload.top_k)
        group_filters = payload.filters.get("security_groups") if payload.filters else None
        if group_filters:
            allow_groups = {str(g).strip().lower() for g in group_filters if str(g).strip()}
            if allow_groups:
                filtered_hits = []
                for item in hits:
                    metadata = item.get("metadata") or {}
                    doc_groups = metadata.get("security_groups") or []
                    if isinstance(doc_groups, str):
                        doc_groups = [doc_groups]
                    doc_set = {str(g).strip().lower() for g in doc_groups if str(g).strip()}
                    if doc_set.intersection(allow_groups):
                        filtered_hits.append(item)
                hits = filtered_hits
    except Exception as exc:
        logger.exception(
            "retrieve failed | request_id=%s backend=%s namespace=%s error=%s",
            ctx.request_id,
            backend,
            namespace,
            exc,
        )
        if payload.backend == "auto" and backend == "qdrant":
            hits = FAISS_STORE.retrieve(namespace, payload.query, payload.top_k)
            backend = "faiss"
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve vectors")
    duration = (time.time() - start) * 1000
    logger.info(
        "retrieve ok | request_id=%s backend=%s namespace=%s hits=%s duration_ms=%.2f",
        ctx.request_id,
        backend,
        namespace,
        len(hits),
        duration,
    )
    return {"hits": hits, "backend": backend, "took_ms": round(duration, 2)}


@app.post("/delete")
async def delete_vectors(payload: DeleteRequest, ctx: AuthContext = Depends(require_auth)) -> Dict[str, Any]:
    backend = resolve_backend(payload.backend)
    namespace = payload.namespace
    ids = payload.ids
    try:
        if backend == "qdrant":
            assert QDRANT_STORE is not None
            deleted = QDRANT_STORE.delete(namespace, ids)
        else:
            deleted = FAISS_STORE.delete(namespace, ids)
    except Exception as exc:
        logger.exception(
            "delete failed | request_id=%s backend=%s namespace=%s error=%s",
            ctx.request_id,
            backend,
            namespace,
            exc,
        )
        if payload.backend == "auto" and backend == "qdrant":
            deleted = FAISS_STORE.delete(namespace, ids)
            backend = "faiss"
        else:
            raise HTTPException(status_code=500, detail="Failed to delete items")
    logger.info(
        "delete ok | request_id=%s backend=%s namespace=%s count=%s",
        ctx.request_id,
        backend,
        namespace,
        deleted,
    )
    return {"ok": True, "deleted": deleted, "backend": backend}


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):  # pragma: no cover
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    logger.info(
        "http error | request_id=%s path=%s status=%s detail=%s",
        request_id,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # pragma: no cover
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    logger.exception(
        "unhandled error | request_id=%s path=%s error=%s",
        request_id,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})



