"""Utility functions for deterministic embeddings used by the retrieval service."""
from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np

EMBEDDING_DIM = 128


def _generate_bytes(source: bytes, target_len: int) -> bytes:
    """Generate a byte sequence of at least ``target_len`` bytes using SHA-256."""
    buffer = bytearray()
    seed = hashlib.sha256(source).digest()
    buffer.extend(seed)
    while len(buffer) < target_len:
        seed = hashlib.sha256(seed + source).digest()
        buffer.extend(seed)
    return bytes(buffer[:target_len])


def embed_text(text: str, *, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Create a deterministic unit-length embedding vector for ``text``.

    The embedding is generated via repeated SHA-256 hashing so equal inputs
    always map to equal vectors without external model calls.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    payload = text.strip().encode("utf-8")
    if not payload:
        payload = b"empty"

    raw = _generate_bytes(payload, dim * 4)
    vector = np.frombuffer(raw, dtype=np.uint32).astype(np.float32)
    vector /= np.float32(2**32)
    vector -= float(vector.mean())

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        # For degenerate cases, distribute values evenly across dimensions.
        vector = np.full(dim, 1.0 / dim, dtype=np.float32)
    else:
        vector /= norm
    return vector.astype(np.float32)


def embed_texts(texts: Iterable[str], *, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Vectorise an iterable of texts into a 2-D float32 numpy array."""
    vectors: List[np.ndarray] = [embed_text(text, dim=dim) for text in texts]
    if not vectors:
        return np.empty((0, dim), dtype=np.float32)
    return np.stack(vectors).astype(np.float32)
