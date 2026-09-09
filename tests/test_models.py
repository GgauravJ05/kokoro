# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Baselines and the from-scratch matrix factoriser."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.eval.splits import Interactions, temporal_split
from kokoro.models.base import Retriever
from kokoro.models.baselines import ItemKNNRecommender, PopularityRecommender, RandomRecommender
from kokoro.models.mf import BPRMatrixFactorization


def _models() -> list[Retriever]:
    return [
        RandomRecommender(seed=1),
        PopularityRecommender(),
        ItemKNNRecommender(k_neighbors=10),
        BPRMatrixFactorization(n_factors=8, n_epochs=2, seed=1),
    ]


@pytest.mark.parametrize("model", _models(), ids=lambda m: m.name)
def test_every_model_satisfies_the_protocol(model: Retriever) -> None:
    assert isinstance(model, Retriever)


@pytest.mark.parametrize("model", _models(), ids=lambda m: m.name)
def test_recommend_shape_and_distinctness(model: Retriever, interactions: Interactions) -> None:
    model.fit(interactions)
    users = np.arange(10, dtype=np.int64)
    ranked = model.recommend(users, k=5)

    assert ranked.shape == (10, 5)
    assert ranked.dtype == np.int64
    for row in ranked:
        assert len(set(row.tolist())) == 5, "a ranking must not repeat an item"
        assert (row < interactions.n_items).all()


def test_popularity_ranks_by_frequency() -> None:
    data = Interactions(
        user=np.array([0, 1, 2, 3, 4], np.int64),
        item=np.array([5, 5, 5, 2, 9], np.int64),
        rating=np.full(5, 8.0, np.float32),
        timestamp=np.arange(5, dtype=np.int64),
    )
    ranked = (
        PopularityRecommender()
        .fit(data)
        .recommend(np.array([9], np.int64), k=1, exclude_seen=False)
    )
    assert ranked[0, 0] == 5


def test_exclude_seen_removes_training_items(interactions: Interactions) -> None:
    model = PopularityRecommender().fit(interactions)
    user = int(interactions.user[0])
    seen = set(interactions.item[interactions.user == user].tolist())
    ranked = model.recommend(np.array([user], np.int64), k=10, exclude_seen=True)
    assert not (set(ranked[0].tolist()) & seen)


def test_bpr_requires_fit_before_use(interactions: Interactions) -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        BPRMatrixFactorization().recommend(np.array([0], np.int64))


def test_bpr_rejects_empty_data() -> None:
    empty = Interactions(
        user=np.empty(0, np.int64),
        item=np.empty(0, np.int64),
        rating=np.empty(0, np.float32),
        timestamp=np.empty(0, np.int64),
    )
    with pytest.raises(ValueError, match="empty interaction log"):
        BPRMatrixFactorization().fit(empty)


def test_bpr_loss_decreases(interactions: Interactions) -> None:
    model = BPRMatrixFactorization(n_factors=16, n_epochs=8, lr=0.1, seed=1).fit(interactions)
    assert model.loss_history[-1] < model.loss_history[0], (
        f"BPR loss did not decrease: {model.loss_history[0]:.4f} -> {model.loss_history[-1]:.4f}"
    )


def test_bpr_beats_random_on_held_out_data(interactions: Interactions) -> None:
    """The floor check. If this fails the training loop is broken."""
    from kokoro.eval.metrics import recall_at_k

    split = temporal_split(interactions, test_frac=0.25)
    truth = split.test.positives_by_user(min_rating=7.0)
    users = np.array(sorted(truth), dtype=np.int64)
    relevant = [truth[int(u)] for u in users]

    bpr = BPRMatrixFactorization(n_factors=16, n_epochs=10, lr=0.1, seed=1).fit(split.train)
    rnd = RandomRecommender(seed=1).fit(split.train)

    assert recall_at_k(bpr.recommend(users, k=20), relevant, 20) > recall_at_k(
        rnd.recommend(users, k=20), relevant, 20
    )


def test_bpr_rejects_k_larger_than_catalog(interactions: Interactions) -> None:
    model = BPRMatrixFactorization(n_epochs=1).fit(interactions)
    with pytest.raises(ValueError, match="exceeds catalog size"):
        model.recommend(np.array([0], np.int64), k=interactions.n_items + 1)


def test_itemknn_shrinkage_suppresses_low_support_pairs() -> None:
    """Two items sharing one user must not look identical."""
    data = Interactions(
        user=np.array([0, 0, 1, 1, 2, 2], np.int64),
        item=np.array([0, 1, 0, 1, 0, 2], np.int64),
        rating=np.full(6, 8.0, np.float32),
        timestamp=np.arange(6, dtype=np.int64),
    )
    strong = ItemKNNRecommender(k_neighbors=3, shrinkage=0.0).fit(data)
    shrunk = ItemKNNRecommender(k_neighbors=3, shrinkage=20.0).fit(data)
    assert strong.similarity is not None and shrunk.similarity is not None
    # Items 0 and 2 co-occur exactly once; shrinkage must damp that pair hard.
    assert shrunk.similarity[0, 2] < strong.similarity[0, 2]
