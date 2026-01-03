"""
Vector Server (RAG / Vector storage)

This repo already provides the vector service as a FastAPI app at:
  `secure_rag.main:app`

This entrypoint runs that service as the Vector tier.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VECTOR_ROOT = Path(__file__).resolve().parent

# Ensure Vector/ is on sys.path so `secure_rag` (moved under Vector/) is importable.
if str(VECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VECTOR_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=VECTOR_ROOT / ".env", override=False)
except Exception:
    pass

import uvicorn


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


VECTOR_PORT = _get_env_int("VECTOR_PORT", 8001)
VECTOR_HOST = os.getenv("VECTOR_HOST", "0.0.0.0")


if __name__ == "__main__":
    try:
        os.chdir(VECTOR_ROOT)
    except Exception:
        pass
    uvicorn.run("secure_rag.main:app", host=VECTOR_HOST, port=VECTOR_PORT, reload=False)
