# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Splitting strategies, especially the leakage they are supposed to prevent."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.eval.splits import (
    Interactions,
    cold_start_split,
    leave_one_out_split,
    random_split,
    temporal_split,
)


def test_interactions_validates_shapes() -> None:
    with pytest.raises(ValueError, match="must share one shape"):
        Interactions(
            user=np.array([0, 1], np.int64),
            item=np.array([0], np.int64),
            rating=np.array([1.0], np.float32),
            timestamp=np.array([1], np.int64),
        )


def test_random_split_preserves_every_interaction(interactions: Interactions) -> None:
    split = random_split(interactions, test_frac=0.25)
    assert len(split.train) + len(split.test) == len(interactions)
    assert len(split.test) == pytest.approx(len(interactions) * 0.25, rel=0.02)


def test_random_split_is_reproducible(interactions: Interactions) -> None:
    a = random_split(interactions, seed=7).test
    b = random_split(interactions, seed=7).test
    np.testing.assert_array_equal(a.item, b.item)


def test_temporal_split_has_no_leakage(interactions: Interactions) -> None:
    split = temporal_split(interactions, test_frac=0.2)
    assert split.train.timestamp.max() < split.test.timestamp.min(), (
        "a training interaction at or after the cut is future leakage"
    )


def test_temporal_split_honours_an_explicit_cut(interactions: Interactions) -> None:
    cut = 1_600_000_000
    split = temporal_split(interactions, cut=cut)
    assert (split.train.timestamp < cut).all()
    assert (split.test.timestamp >= cut).all()


def test_leave_one_out_holds_out_the_latest_per_user() -> None:
    data = Interactions(
        user=np.array([0, 0, 0, 1, 1, 2], np.int64),
        item=np.array([10, 11, 12, 20, 21, 30], np.int64),
        rating=np.full(6, 8.0, np.float32),
        timestamp=np.array([1, 2, 3, 1, 2, 1], np.int64),
    )
    split = leave_one_out_split(data)
    # User 2 has a single interaction and must stay entirely in train.
    assert sorted(split.test.item.tolist()) == [12, 21]
    assert 30 in split.train.item.tolist()


def test_leave_one_out_rejects_zero_holdout(interactions: Interactions) -> None:
    with pytest.raises(ValueError, match="n_holdout must be"):
        leave_one_out_split(interactions, n_holdout=0)


def test_cold_start_items_are_absent_from_train() -> None:
    data = Interactions(
        user=np.array([0, 0, 1, 1], np.int64),
        item=np.array([0, 1, 1, 2], np.int64),
        rating=np.full(4, 9.0, np.float32),
        timestamp=np.array([1, 2, 3, 4], np.int64),
    )
    debut = np.array([100, 100, 500], np.int64)
    split = cold_start_split(data, debut, cut=300)

    cold = set(split.test.item.tolist())
    assert cold == {2}
    assert not (cold & set(split.train.item.tolist())), (
        "a cold-start item with training interactions is not cold"
    )


def test_cold_start_validates_debut_length(interactions: Interactions) -> None:
    with pytest.raises(ValueError, match="item_debut has"):
        cold_start_split(interactions, np.array([1], np.int64), cut=0)


def test_positives_by_user_thresholds(interactions: Interactions) -> None:
    positives = interactions.positives_by_user(min_rating=9.0)
    for items in positives.values():
        assert items
    strict = interactions.positives_by_user(min_rating=10.0)
    assert sum(map(len, strict.values())) <= sum(map(len, positives.values()))
