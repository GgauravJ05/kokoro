# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Offline retrieval and beyond-accuracy metrics.

Two families live here, and the project is evaluated on both:

*Accuracy* — did we retrieve the held-out items? ``recall@k``, ``precision@k``,
``ndcg@k``, ``mrr``, ``hit_rate``.

*Beyond-accuracy* — is the system worth deploying? A recommender can top the
accuracy table by recommending the same fifty popular titles to everyone, which
is the known failure mode in anime recommendation because the rating
distribution is brutally long-tailed. ``catalog_coverage``, ``gini``,
``intra_list_diversity``, ``serendipity`` and ``popularity_lift`` are what catch
that, and every result table in this project reports them alongside NDCG.

Conventions used throughout:

* ``ranked``: an integer array of item ids, most-relevant first, shape ``(k,)``
  for a single query or ``(n_queries, k)`` for a batch.
* ``relevant``: the ground-truth positive item ids for a query.
* Metrics ignore queries with no ground truth rather than scoring them zero;
  the count of scored queries is returned by :func:`evaluate` for transparency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = [
    "candidate_share",
    "catalog_coverage",
    "evaluate",
    "gini",
    "hit_rate_at_k",
    "intra_list_diversity",
    "mrr_at_k",
    "ndcg_at_k",
    "popularity_lift",
    "precision_at_k",
    "recall_at_k",
    "serendipity",
]


def _as_2d(ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Coerce a single ranking or a batch of rankings to a 2-D int array."""
    arr = np.asarray(ranked)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected rankings of shape (k,) or (n, k), got {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _hits(
    ranked: npt.NDArray[np.int_], relevant: Sequence[set[int]], k: int
) -> npt.NDArray[np.bool_]:
    """Return a ``(n_queries, k)`` boolean hit matrix for the top ``k``."""
    top = ranked[:, :k]
    return np.array(
        [[item in rel for item in row] for row, rel in zip(top, relevant, strict=True)],
        dtype=bool,
    ).reshape(top.shape)


def _normalise_relevant(
    relevant: Iterable[Iterable[int]] | Iterable[int],
) -> list[set[int]]:
    """Coerce ground truth into one set of positive ids per query.

    Accepts either a flat iterable of ids (a single query) or an iterable of
    per-query iterables, so callers evaluating one query need not wrap it.
    """
    items = list(relevant)
    if items and isinstance(items[0], (int, np.integer)):
        flat = cast("list[int]", items)
        return [{int(i) for i in flat}]
    nested = cast("list[Iterable[int]]", items)
    return [{int(i) for i in row} for row in nested]


def recall_at_k(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    k: int = 10,
) -> float:
    """Mean fraction of each query's relevant items that appear in the top ``k``.

    Args:
        ranked: Ranked item ids, shape ``(k,)`` or ``(n_queries, k)``.
        relevant: Ground-truth positive ids per query.
        k: Cut-off. Values larger than the ranking length are clamped.

    Returns:
        Mean recall over queries that have at least one relevant item; ``0.0``
        when no query does.
    """
    r = _as_2d(ranked)
    rel = _normalise_relevant(relevant)
    scored = [i for i, s in enumerate(rel) if s]
    if not scored:
        return 0.0
    hits = _hits(r, rel, k)
    return float(np.mean([hits[i].sum() / len(rel[i]) for i in scored]))


def precision_at_k(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    k: int = 10,
) -> float:
    """Mean fraction of the top ``k`` that is relevant."""
    r = _as_2d(ranked)
    rel = _normalise_relevant(relevant)
    scored = [i for i, s in enumerate(rel) if s]
    if not scored:
        return 0.0
    hits = _hits(r, rel, k)
    denom = min(k, r.shape[1])
    return float(np.mean([hits[i].sum() / denom for i in scored]))


def hit_rate_at_k(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    k: int = 10,
) -> float:
    """Fraction of queries with at least one relevant item in the top ``k``."""
    r = _as_2d(ranked)
    rel = _normalise_relevant(relevant)
    scored = [i for i, s in enumerate(rel) if s]
    if not scored:
        return 0.0
    hits = _hits(r, rel, k)
    return float(np.mean([hits[i].any() for i in scored]))


def mrr_at_k(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    k: int = 10,
) -> float:
    """Mean reciprocal rank of the first relevant item within the top ``k``."""
    r = _as_2d(ranked)
    rel = _normalise_relevant(relevant)
    scored = [i for i, s in enumerate(rel) if s]
    if not scored:
        return 0.0
    hits = _hits(r, rel, k)
    out = []
    for i in scored:
        pos = np.flatnonzero(hits[i])
        out.append(1.0 / (pos[0] + 1) if pos.size else 0.0)
    return float(np.mean(out))


def ndcg_at_k(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    k: int = 10,
    gains: Mapping[int, float] | None = None,
) -> float:
    """Normalised discounted cumulative gain at ``k``.

    Uses binary gains by default. Pass ``gains`` to weight items unequally —
    for example a user's 10/10 ratings above their 7/10 ones.

    Args:
        ranked: Ranked item ids, shape ``(k,)`` or ``(n_queries, k)``.
        relevant: Ground-truth positive ids per query.
        k: Cut-off.
        gains: Optional item id -> gain map. Missing ids default to ``1.0``.

    Returns:
        Mean NDCG over queries with at least one relevant item.
    """
    r = _as_2d(ranked)
    rel = _normalise_relevant(relevant)
    scored = [i for i, s in enumerate(rel) if s]
    if not scored:
        return 0.0

    cut = min(k, r.shape[1])
    discounts = 1.0 / np.log2(np.arange(2, cut + 2))

    out = []
    for i in scored:
        row, truth = r[i, :cut], rel[i]
        g = np.array(
            [(gains.get(int(it), 1.0) if gains else 1.0) if it in truth else 0.0 for it in row]
        )
        dcg = float(g @ discounts)

        ideal = sorted((gains.get(it, 1.0) if gains else 1.0 for it in truth), reverse=True)[:cut]
        idcg = float(np.asarray(ideal) @ discounts[: len(ideal)])
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(out))


# --------------------------------------------------------------------------------------
# Beyond-accuracy
# --------------------------------------------------------------------------------------


def catalog_coverage(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    n_items: int,
    k: int = 10,
) -> float:
    """Fraction of the catalog that appears in at least one top-``k`` list.

    Args:
        ranked: Rankings for every evaluation query.
        n_items: Size of the full catalog.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``. A system that only ever surfaces blockbusters
        scores near zero no matter how good its NDCG looks.

    Raises:
        ValueError: If ``n_items`` is not positive.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    r = _as_2d(ranked)[:, :k]
    return float(len(np.unique(r)) / n_items)


def gini(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    n_items: int,
    k: int = 10,
) -> float:
    """Gini coefficient of the recommendation-frequency distribution.

    ``0.0`` means every item is recommended equally often; values approaching
    ``1.0`` mean a handful of items absorb all the exposure. Reported together
    with :func:`catalog_coverage` because coverage alone hides *how* skewed the
    surfaced tail is.

    Args:
        ranked: Rankings for every evaluation query.
        n_items: Size of the full catalog; unrecommended items count as zeros.
        k: Cut-off.

    Returns:
        The Gini coefficient in ``[0, 1]``.

    Raises:
        ValueError: If ``n_items`` is not positive.
    """
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    r = _as_2d(ranked)[:, :k]
    counts = np.bincount(r.ravel(), minlength=n_items).astype(np.float64)
    if counts.sum() == 0:
        return 0.0
    x = np.sort(counts)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def intra_list_diversity(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    embeddings: npt.NDArray[np.float64],
    k: int = 10,
) -> float:
    """Mean pairwise cosine *distance* within each recommendation list.

    Higher is more diverse. Computed in the item embedding space, so it
    measures the diversity the model actually believes in rather than genre-tag
    overlap.

    Args:
        ranked: Rankings, shape ``(k,)`` or ``(n_queries, k)``.
        embeddings: Item embedding matrix, shape ``(n_items, dim)``, indexed by
            item id.
        k: Cut-off; lists shorter than 2 contribute nothing.

    Returns:
        Mean intra-list distance in ``[0, 2]``, or ``0.0`` if no list has two
        or more items.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {embeddings.shape}")
    r = _as_2d(ranked)[:, :k]
    if r.shape[1] < 2:
        return 0.0

    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    out = []
    iu = np.triu_indices(r.shape[1], k=1)
    for row in r:
        sims = unit[row] @ unit[row].T
        out.append(float(np.mean(1.0 - sims[iu])))
    return float(np.mean(out)) if out else 0.0


def popularity_lift(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    popularity: npt.NDArray[np.float64],
    k: int = 10,
) -> float:
    """Ratio of mean recommended-item popularity to mean catalog popularity.

    ``1.0`` means the system is popularity-neutral; ``3.0`` means it recommends
    items three times as popular as the average title, i.e. it is coasting on
    the head of the distribution.

    Args:
        ranked: Rankings for every evaluation query.
        popularity: Per-item popularity (interaction counts), shape
            ``(n_items,)``.
        k: Cut-off.

    Returns:
        The lift ratio; ``0.0`` when the catalog has no interactions.
    """
    r = _as_2d(ranked)[:, :k]
    base = float(np.mean(popularity))
    if base <= 0:
        return 0.0
    return float(np.mean(popularity[r.ravel()]) / base)


def serendipity(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    baseline_ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    k: int = 10,
) -> float:
    """Fraction of the top ``k`` that is both relevant and *not* obvious.

    "Obvious" means the primitive baseline — usually a popularity ranker —
    would have recommended it anyway. This is the metric that separates a model
    that learned something from one that memorised the leaderboard.

    Args:
        ranked: The system's rankings.
        relevant: Ground-truth positive ids per query.
        baseline_ranked: The baseline's rankings, same shape as ``ranked``.
        k: Cut-off.

    Returns:
        Mean serendipity over queries with at least one relevant item.

    Raises:
        ValueError: If the two ranking arrays disagree on query count.
    """
    r = _as_2d(ranked)
    b = _as_2d(baseline_ranked)
    if r.shape[0] != b.shape[0]:
        raise ValueError(f"query count mismatch: {r.shape[0]} vs {b.shape[0]}")

    rel = _normalise_relevant(relevant)
    scored = [i for i, s in enumerate(rel) if s]
    if not scored:
        return 0.0

    cut = min(k, r.shape[1])
    out = []
    for i in scored:
        obvious = set(b[i, :k].tolist())
        novel_hits = sum(1 for it in r[i, :cut] if it in rel[i] and it not in obvious)
        out.append(novel_hits / cut)
    return float(np.mean(out))


def candidate_share(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    candidates: Iterable[int],
    k: int = 10,
) -> float:
    """Fraction of the top ``k`` drawn from the set of answerable items.

    Exists because of a trap in cold-start evaluation. On a cold-start split
    every held-out positive is, by construction, an item with zero training
    interactions — so the *oracle* popularity lift is ``0.0``, and a model
    scores a **high** lift precisely by filling its slots with warm items that
    cannot possibly be correct. Popularity lift therefore does not mean
    "biased toward blockbusters" on that split; it means "wasting the ranking".

    This metric says the same thing directly and without the misreading: what
    share of what the model returned was even eligible.

    Args:
        ranked: Rankings, shape ``(k,)`` or ``(n_queries, k)``.
        candidates: Item ids that could legitimately be correct.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``; ``1.0`` means every slot was spent on an
        answerable item.
    """
    r = _as_2d(ranked)[:, :k]
    eligible = {int(i) for i in candidates}
    if not eligible:
        return 0.0
    return float(np.mean([[int(i) in eligible for i in row] for row in r]))


def evaluate(
    ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_],
    relevant: Iterable[Iterable[int]] | Iterable[int],
    *,
    n_items: int,
    k: int = 10,
    embeddings: npt.NDArray[np.float64] | None = None,
    popularity: npt.NDArray[np.float64] | None = None,
    baseline_ranked: Sequence[Sequence[int]] | npt.NDArray[np.int_] | None = None,
    gains: Mapping[int, float] | None = None,
    candidates: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute the full metric suite in one pass.

    Optional arguments unlock the corresponding beyond-accuracy metrics; when
    omitted, those keys are simply absent from the result rather than reported
    as zero.

    Args:
        ranked: Rankings for every evaluation query.
        relevant: Ground-truth positive ids per query.
        n_items: Catalog size.
        k: Cut-off.
        embeddings: Enables ``intra_list_diversity``.
        popularity: Enables ``popularity_lift``.
        baseline_ranked: Enables ``serendipity``.
        gains: Optional graded gains for NDCG.
        candidates: Enables ``candidate_share``. On a cold-start split, pass the
            cold item ids — see :func:`candidate_share` for why popularity lift
            alone is misleading there.

    Returns:
        A metric-name to value mapping, including ``n_queries_scored``.
    """
    rel = _normalise_relevant(relevant)
    results: dict[str, float] = {
        f"recall@{k}": recall_at_k(ranked, rel, k),
        f"precision@{k}": precision_at_k(ranked, rel, k),
        f"ndcg@{k}": ndcg_at_k(ranked, rel, k, gains=gains),
        f"mrr@{k}": mrr_at_k(ranked, rel, k),
        f"hit_rate@{k}": hit_rate_at_k(ranked, rel, k),
        f"coverage@{k}": catalog_coverage(ranked, n_items, k),
        f"gini@{k}": gini(ranked, n_items, k),
        "n_queries_scored": float(sum(1 for s in rel if s)),
    }
    if embeddings is not None:
        results[f"intra_list_diversity@{k}"] = intra_list_diversity(ranked, embeddings, k)
    if popularity is not None:
        results[f"popularity_lift@{k}"] = popularity_lift(ranked, popularity, k)
    if baseline_ranked is not None:
        results[f"serendipity@{k}"] = serendipity(ranked, rel, baseline_ranked, k)
    if candidates is not None:
        results[f"candidate_share@{k}"] = candidate_share(ranked, candidates, k)
    return results
