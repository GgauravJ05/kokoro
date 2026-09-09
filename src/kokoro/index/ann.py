# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Approximate nearest neighbour search over the item embeddings.

Exact search is fine at 20k titles and the honest thing to say is that HNSW is
not strictly necessary at this catalog size. It is here because the
recall-versus-latency trade-off it exposes is a real engineering result worth
plotting, and because ``ef_search`` is the one production knob that visibly
moves both numbers at once.

Every index written to disk gets a provenance sidecar; see
:func:`kokoro.provenance.write_sidecar`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from kokoro import provenance

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["ExactIndex", "HNSWIndex", "VectorIndex"]


class VectorIndex(Protocol):
    """Anything that can be built over vectors and queried for neighbours."""

    def build(self, vectors: npt.NDArray[np.float32]) -> VectorIndex:
        """Index ``vectors`` of shape ``(n, dim)`` and return ``self``."""
        ...

    def search(
        self, queries: npt.NDArray[np.float32], k: int = 10
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Return ``(ids, scores)``, each of shape ``(n_queries, k)``."""
        ...


class ExactIndex:
    """Brute-force cosine search. The ground truth ANN recall is measured against."""

    name = "exact"

    def __init__(self) -> None:
        self._vectors: npt.NDArray[np.float32] | None = None

    def build(self, vectors: npt.NDArray[np.float32]) -> ExactIndex:
        """Store L2-normalised copies of ``vectors``."""
        if vectors.ndim != 2:
            raise ValueError(f"expected (n, dim), got {vectors.shape}")
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        self._vectors = (vectors / norms).astype(np.float32)
        return self

    def search(
        self, queries: npt.NDArray[np.float32], k: int = 10
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Return the exact top ``k`` by cosine similarity.

        Raises:
            RuntimeError: If called before :meth:`build`.
        """
        if self._vectors is None:
            raise RuntimeError("call build() before searching")
        q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        sims = q.astype(np.float32) @ self._vectors.T
        idx = np.argpartition(-sims, kth=min(k, sims.shape[1]) - 1, axis=1)[:, :k]
        order = np.take_along_axis(sims, idx, axis=1).argsort(axis=1)[:, ::-1]
        ids = np.take_along_axis(idx, order, axis=1)
        return ids.astype(np.int64), np.take_along_axis(sims, ids, axis=1)


class HNSWIndex:
    """Hierarchical navigable small-world index.

    Args:
        m: Graph out-degree. Higher means better recall and a larger index.
        ef_construction: Build-time candidate list size.
        ef_search: Query-time candidate list size — the knob to sweep for the
            recall/latency curve. Adjustable after building.
        seed: Deterministic graph construction.
    """

    name = "hnsw"

    def __init__(
        self, m: int = 32, ef_construction: int = 200, ef_search: int = 64, seed: int = 1337
    ) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._index: object | None = None

    def build(self, vectors: npt.NDArray[np.float32]) -> HNSWIndex:
        """Build the graph over L2-normalised ``vectors``."""
        import hnswlib

        if vectors.ndim != 2:
            raise ValueError(f"expected (n, dim), got {vectors.shape}")
        n, dim = vectors.shape
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        normed = (vectors / norms).astype(np.float32)

        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=n, ef_construction=self.ef_construction, M=self.m, random_seed=self.seed
        )
        index.add_items(normed, np.arange(n))
        index.set_ef(self.ef_search)
        self._index = index
        return self

    def search(
        self, queries: npt.NDArray[np.float32], k: int = 10
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Return approximate top ``k`` neighbours.

        Raises:
            RuntimeError: If called before :meth:`build`.
        """
        if self._index is None:
            raise RuntimeError("call build() before searching")
        q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        ids, distances = self._index.knn_query(q.astype(np.float32), k=k)  # type: ignore[attr-defined]
        return ids.astype(np.int64), (1.0 - distances).astype(np.float32)

    def save(self, path: str) -> None:
        """Write the index plus its provenance sidecar.

        Raises:
            RuntimeError: If called before :meth:`build`.
        """
        if self._index is None:
            raise RuntimeError("call build() before saving")
        self._index.save_index(path)  # type: ignore[attr-defined]
        provenance.write_sidecar(
            path,
            {
                "index": self.name,
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
            },
        )
