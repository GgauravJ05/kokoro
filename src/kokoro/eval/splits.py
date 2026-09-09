# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Dataset splitting strategies, and the leakage traps they avoid.

Three strategies, used for different claims:

:func:`random_split`
    The weakest. Splits interactions at random, so the same user appears in
    train and test and the model can memorise them. Included only as the
    optimistic upper bound in the ablation table, never as a headline number.

:func:`leave_one_out_split`
    The recommender-systems convention: hold out each user's single most recent
    interaction. Comparable to published baselines, but still lets the model see
    the future of *other* users.

:func:`temporal_split`
    The honest one. A global timestamp cut — train strictly before it, test
    strictly after. This is the number to quote, because it is the only split
    that answers the deployed question: *given everything known today, can we
    predict what people watch tomorrow?*

:func:`user_holdout_split`
    Holds out a random sample of each user's interactions. The honest fallback
    when the source has **no timestamps at all** — which is the case for the
    rating matrix this project ingests. Weaker than a temporal split because it
    still lets the model see a user's later behaviour, but unlike
    :func:`random_split` it guarantees every evaluated user has held-out items,
    so per-user metrics are computed over the population you claim to serve.

:func:`cold_start_split`
    Holds out entire items released after the cut, so no interaction for them
    exists at training time. Measures whether the content tower alone can place
    a brand-new title, which is what a real catalog needs every season.

    **This is the headline split for this project.** The ingested rating matrix
    has no interaction timestamps, so :func:`temporal_split` cannot be computed
    honestly on it; debut dates come from the catalog instead, which makes cold
    start both possible and the more interesting question. Every collaborative
    baseline scores exactly 0.0 here by construction — an item nobody has
    interacted with has no collaborative representation — so this split is where
    a content-based model has to justify itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = [
    "Interactions",
    "Split",
    "cold_start_split",
    "leave_one_out_split",
    "random_split",
    "temporal_split",
    "user_holdout_split",
]


@dataclass(frozen=True, slots=True)
class Interactions:
    """A user-item interaction log stored as parallel arrays.

    Attributes:
        user: Contiguous user ids, shape ``(n,)``.
        item: Contiguous item ids, shape ``(n,)``.
        rating: Explicit ratings, shape ``(n,)``. Use ones for implicit
            feedback.
        timestamp: Unix seconds, shape ``(n,)``. Required by the temporal and
            leave-one-out splits.
        catalog_size: Total items in the catalog, which is **not** the same as
            the number of items appearing in this log. A cold-start split
            removes every cold item from the training side, so a model sizing
            itself from ``item.max() + 1`` would build a score matrix with no
            column for the very items it is about to be evaluated on — and would
            then raise, or worse, silently rank a different item space. Splits
            propagate this so the catalog stays fixed.
    """

    user: npt.NDArray[np.int64]
    item: npt.NDArray[np.int64]
    rating: npt.NDArray[np.float32]
    timestamp: npt.NDArray[np.int64]
    catalog_size: int | None = None

    def __post_init__(self) -> None:
        """Validate that all four arrays are 1-D and the same length."""
        if self.catalog_size is not None and self.catalog_size < 0:
            raise ValueError(f"catalog_size must be non-negative, got {self.catalog_size}")
        lengths = {a.shape for a in (self.user, self.item, self.rating, self.timestamp)}
        if len(lengths) != 1:
            raise ValueError(f"all arrays must share one shape, got {lengths}")
        if self.user.ndim != 1:
            raise ValueError(f"expected 1-D arrays, got {self.user.ndim}-D")

    def __len__(self) -> int:
        """Number of interactions."""
        return int(self.user.shape[0])

    @property
    def n_users(self) -> int:
        """One past the largest user id."""
        return int(self.user.max()) + 1 if len(self) else 0

    @property
    def n_items(self) -> int:
        """Catalog size: the explicit one when set, else one past the largest id."""
        if self.catalog_size is not None:
            return self.catalog_size
        return int(self.item.max()) + 1 if len(self) else 0

    def take(self, idx: npt.NDArray[np.int64]) -> Interactions:
        """Return the subset of interactions at ``idx``."""
        return Interactions(
            user=self.user[idx],
            item=self.item[idx],
            rating=self.rating[idx],
            timestamp=self.timestamp[idx],
            catalog_size=self.n_items,
        )

    def positives_by_user(self, min_rating: float = 7.0) -> dict[int, set[int]]:
        """Group item ids by user, keeping only ratings at or above ``min_rating``.

        Args:
            min_rating: Threshold on the 1-10 MAL/AniList scale. Defaults to 7,
                the conventional "the user liked it" cut for this domain.

        Returns:
            A user id to positive item id set mapping.
        """
        out: dict[int, set[int]] = {}
        mask = self.rating >= min_rating
        for u, i in zip(self.user[mask].tolist(), self.item[mask].tolist(), strict=True):
            out.setdefault(u, set()).add(i)
        return out


@dataclass(frozen=True, slots=True)
class Split:
    """A train/test partition plus a description of how it was made."""

    train: Interactions
    test: Interactions
    strategy: str
    #: Free-form details worth recording in the results table, e.g. the cut
    #: timestamp or the RNG seed.
    meta: dict[str, float | int | str]


def random_split(data: Interactions, *, test_frac: float = 0.2, seed: int = 1337) -> Split:
    """Split interactions uniformly at random.

    Optimistic by construction — see the module docstring. Report it only as an
    upper bound.

    Args:
        data: The interaction log.
        test_frac: Fraction of interactions held out.
        seed: RNG seed.

    Returns:
        The resulting :class:`Split`.

    Raises:
        ValueError: If ``test_frac`` is not strictly between 0 and 1.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    n_test = max(1, round(len(data) * test_frac))
    return Split(
        train=data.take(perm[n_test:]),
        test=data.take(perm[:n_test]),
        strategy="random",
        meta={"test_frac": test_frac, "seed": seed},
    )


def leave_one_out_split(data: Interactions, *, n_holdout: int = 1) -> Split:
    """Hold out each user's ``n_holdout`` most recent interactions.

    Users with too few interactions to give up ``n_holdout`` and still leave at
    least one for training are kept entirely in train.

    Args:
        data: The interaction log.
        n_holdout: Interactions to hold out per user.

    Returns:
        The resulting :class:`Split`.

    Raises:
        ValueError: If ``n_holdout`` is not positive.
    """
    if n_holdout < 1:
        raise ValueError(f"n_holdout must be >= 1, got {n_holdout}")

    # Sort by (user, timestamp) so each user's block ends with their latest.
    order = np.lexsort((data.timestamp, data.user))
    users_sorted = data.user[order]
    _, starts, counts = np.unique(users_sorted, return_index=True, return_counts=True)

    test_pos: list[int] = []
    for start, count in zip(starts.tolist(), counts.tolist(), strict=True):
        if count <= n_holdout:
            continue  # too short to hold out from
        test_pos.extend(range(start + count - n_holdout, start + count))

    test_idx = order[np.asarray(test_pos, dtype=np.int64)] if test_pos else np.empty(0, np.int64)
    mask = np.ones(len(data), dtype=bool)
    mask[test_idx] = False
    return Split(
        train=data.take(np.flatnonzero(mask)),
        test=data.take(test_idx),
        strategy="leave_one_out",
        meta={"n_holdout": n_holdout, "n_test_users": len(test_pos) // n_holdout},
    )


def temporal_split(data: Interactions, *, cut: int | None = None, test_frac: float = 0.2) -> Split:
    """Split on a global timestamp: train strictly before, test at or after.

    Args:
        data: The interaction log.
        cut: Unix-seconds boundary. When ``None``, chosen as the quantile that
            puts ``test_frac`` of interactions on the test side.
        test_frac: Only used when ``cut`` is ``None``.

    Returns:
        The resulting :class:`Split`.

    Raises:
        ValueError: If ``test_frac`` is not strictly between 0 and 1.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if cut is None:
        cut = round(float(np.quantile(data.timestamp, 1.0 - test_frac)))
    test_idx = np.flatnonzero(data.timestamp >= cut)
    train_idx = np.flatnonzero(data.timestamp < cut)
    return Split(
        train=data.take(train_idx),
        test=data.take(test_idx),
        strategy="temporal",
        meta={"cut": int(cut), "n_train": int(train_idx.size), "n_test": int(test_idx.size)},
    )


def cold_start_split(data: Interactions, item_debut: npt.NDArray[np.int64], *, cut: int) -> Split:
    """Hold out every interaction with an item that debuted at or after ``cut``.

    Unlike :func:`temporal_split`, this guarantees the held-out items have
    *zero* training interactions, which is the only way to honestly measure
    cold-start performance.

    Args:
        data: The interaction log.
        item_debut: Per-item release timestamp in unix seconds, shape
            ``(n_items,)``, indexed by item id.
        cut: Debut boundary in unix seconds.

    Returns:
        The resulting :class:`Split`, with ``meta["n_cold_items"]`` recording
        how many items were withheld.

    Raises:
        ValueError: If ``item_debut`` is too short for the observed item ids.
    """
    if item_debut.shape[0] < data.n_items:
        raise ValueError(
            f"item_debut has {item_debut.shape[0]} entries but ids go up to {data.n_items - 1}"
        )
    cold = item_debut >= cut
    is_cold = cold[data.item]
    return Split(
        train=data.take(np.flatnonzero(~is_cold)),
        test=data.take(np.flatnonzero(is_cold)),
        strategy="cold_start",
        meta={"cut": int(cut), "n_cold_items": int(cold.sum())},
    )


def user_holdout_split(
    data: Interactions,
    *,
    holdout_frac: float = 0.2,
    min_interactions: int = 5,
    seed: int = 1337,
) -> Split:
    """Hold out a random fraction of each user's interactions.

    Use this when the source carries no timestamps, so neither
    :func:`temporal_split` nor :func:`leave_one_out_split` can be computed
    honestly. Report it as such: it is not a substitute for a temporal split,
    because a model can still learn from a user's later behaviour.

    Users with fewer than ``min_interactions`` are kept wholly in train — a user
    with two ratings cannot support both a profile and an evaluation.

    Args:
        data: The interaction log.
        holdout_frac: Fraction of each eligible user's interactions to hold out.
        min_interactions: Minimum interactions for a user to be evaluated.
        seed: RNG seed.

    Returns:
        The resulting :class:`Split`, with ``meta["n_eval_users"]`` recording how
        many users are actually scored.

    Raises:
        ValueError: If ``holdout_frac`` is not strictly between 0 and 1.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")

    rng = np.random.default_rng(seed)
    order = np.argsort(data.user, kind="stable")
    users_sorted = data.user[order]
    _, starts, counts = np.unique(users_sorted, return_index=True, return_counts=True)

    test_pos: list[int] = []
    n_eval_users = 0
    for start, count in zip(starts.tolist(), counts.tolist(), strict=True):
        if count < min_interactions:
            continue
        n_hold = max(1, round(count * holdout_frac))
        if n_hold >= count:  # always leave the user something to train on
            n_hold = count - 1
        picked = rng.choice(count, size=n_hold, replace=False) + start
        test_pos.extend(picked.tolist())
        n_eval_users += 1

    test_idx = order[np.asarray(test_pos, dtype=np.int64)] if test_pos else np.empty(0, np.int64)
    mask = np.ones(len(data), dtype=bool)
    mask[test_idx] = False
    return Split(
        train=data.take(np.flatnonzero(mask)),
        test=data.take(test_idx),
        strategy="user_holdout",
        meta={
            "holdout_frac": holdout_frac,
            "min_interactions": min_interactions,
            "seed": seed,
            "n_eval_users": n_eval_users,
        },
    )
