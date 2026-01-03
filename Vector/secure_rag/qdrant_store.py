"""Qdrant client wrapper used by the retrieval service."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient, models

from .embeddings import EMBEDDING_DIM, embed_text, embed_texts

_LOGGER = logging.getLogger(__name__)


class QdrantVectorStore:
    """Lightweight wrapper around :mod:`qdrant_client`."""

    def __init__(
        self,
        *,
        url: str,
        api_key: Optional[str],
        timeout: float = 2.0,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self.url = url
        self.api_key = api_key or None
        self.timeout = timeout
        self.dim = dim
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
            prefer_grpc=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_collection(self, namespace: str) -> None:
        try:
            self.client.get_collection(collection_name=namespace)
        except Exception:
            _LOGGER.info("creating collection '%s' in Qdrant", namespace)
            self.client.create_collection(
                collection_name=namespace,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filters:
            return None
        conditions: List[models.FieldCondition] = []
        for key, value in filters.items():
            conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )
        if not conditions:
            return None
        return models.Filter(must=conditions)

    def is_available(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            _LOGGER.debug("qdrant availability check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API used by FastAPI handlers
    # ------------------------------------------------------------------
    def upsert(self, namespace: str, items: Iterable[Dict[str, Any]]) -> int:
        payload = list(items)
        if not payload:
            return 0
        self._ensure_collection(namespace)
        vectors = embed_texts((item["text"] for item in payload), dim=self.dim)
        payloads: List[Dict[str, Any]] = []
        ids: List[str] = []
        for item in payload:
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be a dict")
            payloads.append({
                "text": item["text"],
                "metadata": metadata,
            })
            ids.append(str(item["id"]))
        self.client.upsert(
            collection_name=namespace,
            points=models.Batch(ids=ids, vectors=vectors.tolist(), payloads=payloads),
        )
        return len(payload)

    def retrieve(
        self,
        namespace: str,
        query: str,
        top_k: int,
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("qdrant is not reachable")
        self._ensure_collection(namespace)
        query_vector = embed_text(query, dim=self.dim).tolist()
        filter_expr = self._build_filter(filters)
        search_result = self.client.search(
            collection_name=namespace,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_expr,
            with_payload=True,
            with_vectors=False,
        )
        hits: List[Dict[str, Any]] = []
        for point in search_result:
            payload = point.payload or {}
            hits.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return hits

    def delete(self, namespace: str, ids: Iterable[str]) -> int:
        ids = [str(_id) for _id in ids]
        if not ids:
            return 0
        if not self.is_available():
            raise RuntimeError("qdrant is not reachable")
        self._ensure_collection(namespace)
        self.client.delete(
            collection_name=namespace,
            points_selector=models.PointIdsList(points=ids),
        )
        return len(ids)

    def get_vector_count(self, namespace: str) -> int:
        if not self.is_available():
            return 0
        try:
            collection_info = self.client.get_collection(collection_name=namespace)
        except Exception:
            return 0
        return int(collection_info.vectors_count or 0)
