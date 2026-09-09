# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Non-negotiable baselines.

Every neural claim in this project is reported against these. They are cheap,
and they are surprisingly hard to beat on a long-tailed catalog — which is the
point. :class:`PopularityRecommender` in particular doubles as the reference
ranking for :func:`kokoro.eval.metrics.serendipity`, because "would the
popularity ranker have said this anyway?" is the sharpest available definition
of an obvious recommendation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kokoro.eval.splits import Interactions

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["ItemKNNRecommender", "PopularityRecommender", "RandomRecommender"]


def _top_k(scores: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.int64]:
    """Return the indices of the top ``k`` scores per row, best-first."""
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, ordered, axis=1).astype(np.int64)


class PopularityRecommender:
    """Recommends the globally most-interacted items to everyone.

    Args:
        name: Override the reporting label.
    """

    def __init__(self, name: str = "popularity") -> None:
        self.name = name
        self.popularity: npt.NDArray[np.float64] | None = None
        self._seen: dict[int, set[int]] = {}
        self._n_items = 0

    def fit(self, data: Interactions) -> PopularityRecommender:
        """Count interactions per item.

        Args:
            data: Training interactions.

        Returns:
            ``self``.
        """
        self._n_items = data.n_items
        self.popularity = np.bincount(data.item, minlength=self._n_items).astype(np.float64)
        self._seen = {}
        for u, i in zip(data.user.tolist(), data.item.tolist(), strict=True):
            self._seen.setdefault(u, set()).add(i)
        return self

    def recommend(
        self, users: npt.NDArray[np.int64], k: int = 10, *, exclude_seen: bool = True
    ) -> npt.NDArray[np.int64]:
        """Return the top ``k`` most popular unseen items per user.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.popularity is None:
            raise RuntimeError("call fit() before recommending")
        scores = np.tile(self.popularity, (users.shape[0], 1))
        if exclude_seen:
            for row, u in enumerate(users.tolist()):
                seen = self._seen.get(int(u))
                if seen:
                    scores[row, list(seen)] = -np.inf
        return _top_k(scores, k)


class RandomRecommender:
    """Uniformly random rankings — the floor, and a sanity check on the harness.

    If any model scores at or below this, the bug is in the evaluation code, not
    the model.

    Args:
        seed: RNG seed.
    """

    name = "random"

    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed
        self._n_items = 0

    def fit(self, data: Interactions) -> RandomRecommender:
        """Record the catalog size; there is nothing else to learn."""
        self._n_items = data.n_items
        return self

    def recommend(
        self, users: npt.NDArray[np.int64], k: int = 10, *, exclude_seen: bool = True
    ) -> npt.NDArray[np.int64]:
        """Return ``k`` distinct random item ids per user."""
        del exclude_seen  # a random ranker has nothing to exclude against
        rng = np.random.default_rng(self.seed)
        return np.stack(
            [rng.choice(self._n_items, size=k, replace=False) for _ in range(users.shape[0])]
        ).astype(np.int64)


class ItemKNNRecommender:
    r"""Item-item collaborative filtering with shrunk cosine similarity.

    Similarity between items is cosine over their user-interaction columns,
    shrunk toward zero by co-occurrence count:

    .. math:: s'_{ij} = s_{ij} \\cdot \\frac{c_{ij}}{c_{ij} + \\lambda}

    Without the shrinkage two items sharing a single user score a perfect 1.0,
    and the long tail of an anime catalog is full of such pairs — they would
    otherwise dominate every neighbourhood.

    Args:
        k_neighbors: Neighbours retained per item; the rest are zeroed.
        shrinkage: :math:`\\lambda` above. Larger means more distrust of
            low-support pairs.
        name: Override the reporting label.
    """

    def __init__(
        self, k_neighbors: int = 50, shrinkage: float = 20.0, name: str = "item-knn"
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.name = name
        self.similarity: npt.NDArray[np.float64] | None = None
        self._matrix: npt.NDArray[np.float64] | None = None
        self._n_items = 0

    def fit(self, data: Interactions) -> ItemKNNRecommender:
        """Build the shrunk, top-``k`` truncated item-item similarity matrix.

        Args:
            data: Training interactions.

        Returns:
            ``self``.
        """
        n_users, n_items = data.n_users, data.n_items
        self._n_items = n_items

        mat = np.zeros((n_users, n_items), dtype=np.float64)
        mat[data.user, data.item] = 1.0
        self._matrix = mat

        norms = np.maximum(np.linalg.norm(mat, axis=0), 1e-12)
        cooccur = mat.T @ mat
        sim = cooccur / np.outer(norms, norms)
        # np.where alone would still evaluate 0/0 for never-co-occurring pairs;
        # divide(where=...) leaves those entries at the `out` value instead.
        shrink = np.divide(
            cooccur,
            cooccur + self.shrinkage,
            out=np.zeros_like(cooccur),
            where=cooccur > 0,
        )
        sim *= shrink
        np.fill_diagonal(sim, 0.0)

        if self.k_neighbors < n_items:
            cut = np.partition(sim, kth=n_items - self.k_neighbors, axis=1)[
                :, n_items - self.k_neighbors
            ]
            sim[sim < cut[:, None]] = 0.0

        self.similarity = sim
        return self

    def recommend(
        self, users: npt.NDArray[np.int64], k: int = 10, *, exclude_seen: bool = True
    ) -> npt.NDArray[np.int64]:
        """Score items by similarity to everything the user has interacted with.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.similarity is None or self._matrix is None:
            raise RuntimeError("call fit() before recommending")
        profiles = self._matrix[users]
        scores = profiles @ self.similarity
        if exclude_seen:
            scores[profiles > 0] = -np.inf
        return _top_k(scores, k)
