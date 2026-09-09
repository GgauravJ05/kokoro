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
from scipy import sparse

from kokoro.eval.splits import Interactions

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["ItemKNNRecommender", "PopularityRecommender", "RandomRecommender"]


def mask_to_candidates(
    scores: npt.NDArray[np.float64], candidates: npt.NDArray[np.int64] | None
) -> npt.NDArray[np.float64]:
    """Restrict a score matrix to an eligible candidate set, in place.

    Used for the cold-only cold-start protocol: ranking a brand-new title
    against the entire back catalog measures mostly how well a model avoids
    warm distractors, which is a different question from whether it can order
    new titles sensibly among themselves. Both protocols are reported.

    Args:
        scores: Score matrix of shape ``(n_users, n_items)``. Mutated.
        candidates: Eligible item ids, or ``None`` to leave scores untouched.

    Returns:
        The same array, with ineligible columns set to ``-inf``.
    """
    if candidates is None:
        return scores
    keep = np.zeros(scores.shape[1], dtype=bool)
    keep[candidates] = True
    scores[:, ~keep] = -np.inf
    return scores


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
        self,
        users: npt.NDArray[np.int64],
        k: int = 10,
        *,
        exclude_seen: bool = True,
        candidates: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Return the top ``k`` most popular unseen items per user.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.popularity is None:
            raise RuntimeError("call fit() before recommending")
        scores = np.tile(self.popularity, (users.shape[0], 1))
        scores = mask_to_candidates(scores, candidates)
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
        self,
        users: npt.NDArray[np.int64],
        k: int = 10,
        *,
        exclude_seen: bool = True,
        candidates: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Return ``k`` distinct random item ids per user."""
        del exclude_seen  # a random ranker has nothing to exclude against
        rng = np.random.default_rng(self.seed)
        pool = np.arange(self._n_items) if candidates is None else np.asarray(candidates)
        return np.stack(
            [rng.choice(pool, size=k, replace=False) for _ in range(users.shape[0])]
        ).astype(np.int64)


class ItemKNNRecommender:
    r"""Item-item collaborative filtering with shrunk cosine similarity.

    Similarity between items is cosine over their user-interaction columns,
    shrunk toward zero by co-occurrence count:

    .. math:: s'_{ij} = s_{ij} \cdot \frac{c_{ij}}{c_{ij} + \lambda}

    Without the shrinkage two items sharing a single user score a perfect 1.0,
    and the long tail of an anime catalog is full of such pairs — they would
    otherwise dominate every neighbourhood.

    Everything is held sparse. On the real corpus (69,599 users x 9,864 titles)
    a dense user-item matrix is 5.5 GB and the item-item matrix another 0.8 GB;
    the similarity is therefore accumulated in row blocks and truncated to
    ``k_neighbors`` before anything dense is materialised.

    Args:
        k_neighbors: Neighbours retained per item; the rest are zeroed.
        shrinkage: :math:`\lambda` above. Larger means more distrust of
            low-support pairs.
        block: Items per similarity block. Trades peak memory against speed;
            the default keeps each block under ~50 MB at this catalog size.
        name: Override the reporting label.
    """

    def __init__(
        self,
        k_neighbors: int = 50,
        shrinkage: float = 20.0,
        block: int = 512,
        name: str = "item-knn",
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if block < 1:
            raise ValueError(f"block must be >= 1, got {block}")
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.block = block
        self.name = name
        self.similarity: sparse.csr_matrix | None = None
        self._matrix: sparse.csr_matrix | None = None
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

        mat = sparse.csr_matrix(
            (np.ones(len(data), dtype=np.float64), (data.user, data.item)),
            shape=(n_users, n_items),
        )
        mat.data[:] = 1.0  # collapse duplicate interactions to a single implicit 1
        self._matrix = mat

        norms = np.asarray(np.sqrt(mat.multiply(mat).sum(axis=0))).ravel()
        norms = np.maximum(norms, 1e-12)

        csc = mat.tocsc()
        rows: list[sparse.csr_matrix] = []

        for start in range(0, n_items, self.block):
            stop = min(start + self.block, n_items)
            # (block, n_items) dense: the only dense allocation, bounded by `block`.
            cooccur = (csc[:, start:stop].T @ mat).toarray()

            sim = cooccur / np.outer(norms[start:stop], norms)
            shrink = np.divide(
                cooccur,
                cooccur + self.shrinkage,
                out=np.zeros_like(cooccur),
                where=cooccur > 0,
            )
            sim *= shrink
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0

            if self.k_neighbors < n_items:
                kth = n_items - self.k_neighbors
                cut = np.partition(sim, kth=kth, axis=1)[:, kth]
                sim[sim < cut[:, None]] = 0.0

            rows.append(sparse.csr_matrix(sim))

        self.similarity = sparse.vstack(rows, format="csr")
        return self

    def recommend(
        self,
        users: npt.NDArray[np.int64],
        k: int = 10,
        *,
        exclude_seen: bool = True,
        candidates: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Score items by similarity to everything the user has interacted with.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.similarity is None or self._matrix is None:
            raise RuntimeError("call fit() before recommending")
        profiles = self._matrix[users]
        scores = np.asarray((profiles @ self.similarity).todense())
        scores = mask_to_candidates(scores, candidates)
        if exclude_seen:
            scores[np.asarray(profiles.todense()) > 0] = -np.inf
        return _top_k(scores, k)
