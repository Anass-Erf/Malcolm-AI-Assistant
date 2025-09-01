import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import faiss
import numpy as np


@dataclass
class VectorHit:
    id: int
    score: float
    payload: Dict[str, Any]


class FAISSVectorStore:
    """FAISS wrapper (cosine via inner product on L2-normalized vectors) with payload persistence."""
    def __init__(self, dim: int, index_path: str, payload_path: str) -> None:
        self.dim = dim
        self.index_path = index_path
        self.payload_path = payload_path

        self.index = None  # type: ignore
        self.id_map: Dict[int, Dict[str, Any]] = {}

        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            self.index = faiss.read_index(self.index_path)
            if self.index.d != self.dim:
                raise ValueError(
                    f"FAISS index dim ({self.index.d}) != expected dim ({self.dim}). "
                    f"Delete {self.index_path} and rebuild."
                )
            if os.path.exists(self.payload_path):
                with open(self.payload_path, "rb") as f:
                    self.id_map = pickle.load(f)
        else:
            base = faiss.IndexFlatIP(self.dim)  # inner product
            self.index = faiss.IndexIDMap(base)

    @staticmethod
    def _ensure_float32(X: np.ndarray) -> np.ndarray:
        return X.astype("float32", copy=False)

    @staticmethod
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        X = X.astype("float32", copy=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    def add(self, vectors: np.ndarray, ids: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        vectors = self._l2_normalize(self._ensure_float32(vectors))
        assert len(ids) == len(payloads) == vectors.shape[0], "mismatched lengths"
        ids = np.asarray(ids, dtype=np.int64)
        # ensure uniqueness within batch
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Duplicate IDs in add() batch")
        self.index.add_with_ids(vectors, ids)
        for i, p in zip(ids.tolist(), payloads):
            self.id_map[int(i)] = p

    def upsert(self, vectors: np.ndarray, ids: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.delete(ids)
        self.add(vectors, ids, payloads)

    def delete(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0 or self.index.ntotal == 0:
            return
        sel = faiss.IDSelectorBatch(ids)
        self.index.remove_ids(sel)
        for i in ids.tolist():
            self.id_map.pop(int(i), None)

    def save(self) -> None:
        dir_ = os.path.dirname(self.index_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.payload_path, "wb") as f:
            pickle.dump(self.id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> List[List[VectorHit]]:
        query_vectors = self._l2_normalize(self._ensure_float32(query_vectors))
        D, I = self.index.search(query_vectors, top_k)
        results: List[List[VectorHit]] = []
        for row_scores, row_ids in zip(D, I):
            hits: List[VectorHit] = []
            for score, _id in zip(row_scores, row_ids):
                if _id == -1:
                    continue
                payload = self.id_map.get(int(_id), {})
                hits.append(VectorHit(id=int(_id), score=float(score), payload=payload))
            results.append(hits)
        return results

    # Optional helpers
    def count(self) -> int:
        return int(self.index.ntotal)

    def clear(self) -> None:
        base = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base)
        self.id_map.clear()
