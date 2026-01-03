"""Lightweight in-memory vector store without external FAISS dependency."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .embeddings import EMBEDDING_DIM, embed_text


@dataclass
class _NamespaceStore:
    """State container for a single namespace."""

    dim: int = EMBEDDING_DIM
    items: Dict[str, Tuple[np.ndarray, str, Dict[str, Any]]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:  # initialise empty matrix
        self.matrix = np.empty((0, self.dim), dtype=np.float32)

    def _rebuild_matrix(self) -> None:
        if not self.items:
            self.order = []
            self.matrix = np.empty((0, self.dim), dtype=np.float32)
            return
        self.order = sorted(self.items.keys())
        self.matrix = np.stack([self.items[item_id][0] for item_id in self.order]).astype(np.float32)

    def upsert(self, payloads: Iterable[Tuple[str, str, Dict[str, Any]]]) -> int:
        count = 0
        for item_id, text, metadata in payloads:
            vector = embed_text(text, dim=self.dim)
            self.items[item_id] = (vector, text, metadata or {})
            count += 1
        if count:
            self._rebuild_matrix()
        return count

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.items or self.matrix.size == 0:
            return []
        query_vec = embed_text(query, dim=self.dim).astype(np.float32)
        scores = self.matrix @ query_vec  # cosine similarity because embeddings are unit length
        top_k = max(1, min(top_k, len(self.order)))
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            item_id = self.order[int(idx)]
            score = float(scores[int(idx)])
            _, text, metadata = self.items[item_id]
            results.append({
                "id": item_id,
                "score": score,
                "text": text,
                "metadata": metadata,
            })
        return results

    def delete(self, ids: Iterable[str]) -> int:
        removed = 0
        for item_id in list(ids):
            if item_id in self.items:
                del self.items[item_id]
                removed += 1
        if removed:
            self._rebuild_matrix()
        return removed

    def vector_count(self) -> int:
        return len(self.items)


class FaissVectorStore:
    """Simple namespace-indexed vector store using pure NumPy operations."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self._namespaces: Dict[str, _NamespaceStore] = {}

    def _get_namespace(self, namespace: str) -> _NamespaceStore:
        if namespace not in self._namespaces:
            self._namespaces[namespace] = _NamespaceStore(dim=self.dim)
        return self._namespaces[namespace]

    def upsert(self, namespace: str, items: Iterable[Dict[str, Any]]) -> int:
        store = self._get_namespace(namespace)
        payloads = ((item["id"], item["text"], item.get("metadata", {})) for item in items)
        return store.upsert(payloads)

    def retrieve(self, namespace: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        store = self._get_namespace(namespace)
        return store.search(query, top_k)

    def delete(self, namespace: str, ids: Iterable[str]) -> int:
        store = self._get_namespace(namespace)
        return store.delete(ids)

    def get_vector_count(self, namespace: str) -> int:
        store = self._namespaces.get(namespace)
        if store is None:
            return 0
        return store.vector_count()

    def has_namespace(self, namespace: str) -> bool:
        return namespace in self._namespaces and self._namespaces[namespace].vector_count() > 0
