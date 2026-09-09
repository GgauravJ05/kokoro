# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""The one interface every retriever in this project implements.

Baselines, the from-scratch matrix factoriser, the two-tower model and the
trajectory matcher all satisfy :class:`Retriever`, which is what lets
``kokoro eval`` run the identical harness over every one of them and produce a
directly comparable ablation table. Adding a model means implementing two
methods, not touching the evaluation code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import numpy.typing as npt

    from kokoro.eval.splits import Interactions

__all__ = ["Retriever"]


@runtime_checkable
class Retriever(Protocol):
    """A model that can be fitted on interactions and asked for rankings."""

    name: str

    def fit(self, data: Interactions) -> Retriever:
        """Train on an interaction log.

        Args:
            data: Training interactions. Implementations must not peek at
                anything outside this object.

        Returns:
            ``self``, so calls can be chained.
        """
        ...

    def recommend(
        self,
        users: npt.NDArray[np.int64],
        k: int = 10,
        *,
        exclude_seen: bool = True,
        candidates: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Rank the catalog for each user.

        Args:
            users: User ids to score, shape ``(n_users,)``.
            k: Number of items to return per user.
            exclude_seen: Drop items the user already interacted with in the
                training data. Leaving this on is what makes the numbers
                comparable to published baselines.
            candidates: Restrict the ranking to these item ids. Used for the
                cold-only cold-start protocol, where ranking against the whole
                back catalog measures distractor avoidance rather than the
                ability to order new titles among themselves.

        Returns:
            Item ids, shape ``(n_users, k)``, best-first.
        """
        ...
