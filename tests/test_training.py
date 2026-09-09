# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Pair construction and contrastive training of the projection heads."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.data.pairs import PairSet, is_boilerplate, iter_batches

torch = pytest.importorskip("torch", reason="optional [train] extra not installed")

from kokoro.train.contrastive import (
    ProjectionTower,
    TrainConfig,
    train_projections,
)


def _pairs(n_titles: int = 20, per_title: int = 10) -> PairSet:
    """A pair set with a known title structure."""
    queries = [f"segment {t}-{i}" for t in range(n_titles) for i in range(per_title)]
    item_pos = np.repeat(np.arange(n_titles, dtype=np.int64), per_title)
    return PairSet(queries=queries, item_pos=item_pos, anime_ids=item_pos + 1000)


# ---------------------------------------------------------------------------
# Boilerplate filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "thanks for reading my review",
        "feel free to leave a comment",
        "check my profile on my profile page",
        "sorry for my english",
        "story: 8 art: 9",
        "am i being overly harsh?",
    ],
)
def test_boilerplate_is_dropped(text: str) -> None:
    assert is_boilerplate(text)


@pytest.mark.parametrize(
    "text",
    [
        "the pacing is slow and the ending devastated me",
        "a quiet meditation on grief and memory",
        "it completely changes after episode 12",
        "the first cour drags but the payoff lands",
    ],
)
def test_real_mood_commentary_is_kept(text: str) -> None:
    """Over-filtering would delete exactly the signal the model trains on."""
    assert not is_boilerplate(text)


# ---------------------------------------------------------------------------
# Batching — the false-negative problem
# ---------------------------------------------------------------------------


def test_batches_never_repeat_a_title() -> None:
    """A repeated title inside a batch is a false negative: InfoNCE would push
    apart two segments describing the same work."""
    pairs = _pairs(n_titles=30, per_title=8)
    for batch in iter_batches(pairs, batch_size=16, seed=7):
        titles = pairs.item_pos[batch]
        assert len(set(titles.tolist())) == len(titles), "duplicate title in batch"


def test_batch_size_is_clamped_to_available_titles() -> None:
    """A batch larger than the title count cannot be collision-free."""
    pairs = _pairs(n_titles=5, per_title=20)
    for batch in iter_batches(pairs, batch_size=100, seed=1):
        assert batch.size <= 5


def test_batches_are_reproducible() -> None:
    pairs = _pairs()
    a = [b.tolist() for b in iter_batches(pairs, batch_size=8, seed=3)]
    b = [b.tolist() for b in iter_batches(pairs, batch_size=8, seed=3)]
    assert a == b


def test_batches_index_within_range() -> None:
    pairs = _pairs()
    for batch in iter_batches(pairs, batch_size=8, seed=1):
        assert batch.min() >= 0
        assert batch.max() < len(pairs)


def test_iter_batches_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be"):
        next(iter_batches(_pairs(), batch_size=0))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_split_is_disjoint_by_title() -> None:
    """Splitting by pair would put segments of one review on both sides, so
    validation would measure memorisation rather than transfer."""
    pairs = _pairs(n_titles=20, per_title=10)
    train, val = pairs.split(val_frac=0.2, seed=5)

    assert not set(train.item_pos.tolist()) & set(val.item_pos.tolist())
    assert len(train) + len(val) == len(pairs)
    assert val.n_titles == 4


def test_split_always_holds_out_at_least_one_title() -> None:
    train, val = _pairs(n_titles=3, per_title=2).split(val_frac=0.01, seed=1)
    assert val.n_titles >= 1
    assert len(train) > 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_projection_tower_normalises_output() -> None:
    tower = ProjectionTower(16, 8).eval()
    out = tower(torch.randn(5, 16))
    assert out.shape == (5, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_training_learns_a_separable_signal() -> None:
    """The one test that would catch a broken objective: with queries that
    genuinely cluster by title, validation retrieval must beat chance."""
    rng = np.random.default_rng(0)
    n_titles, per_title, dim = 24, 12, 32

    # Each title gets a distinct centroid; its queries are noisy copies of it.
    centroids = rng.standard_normal((n_titles, dim)).astype(np.float32)
    item_emb = centroids.copy()
    query_emb = np.repeat(centroids, per_title, axis=0) + 0.25 * rng.standard_normal(
        (n_titles * per_title, dim)
    ).astype(np.float32)

    pairs = _pairs(n_titles=n_titles, per_title=per_title)
    train, val = pairs.split(val_frac=0.25, seed=1)

    train_mask = np.isin(pairs.item_pos, np.unique(train.item_pos))
    result = train_projections(
        train,
        query_emb[train_mask],
        item_emb,
        val_pairs=val,
        val_query_embeddings=query_emb[~train_mask],
        config=TrainConfig(output_dim=16, epochs=25, batch_size=8, patience=0),
        verbose=False,
    )

    assert result.train_loss[-1] < result.train_loss[0], "loss must decrease"
    chance = 1.0 / val.n_titles
    assert max(result.val_recall) > 3 * chance, (
        f"val recall@1 {max(result.val_recall):.3f} is not meaningfully above chance {chance:.3f}"
    )


def test_project_items_covers_every_item_including_cold() -> None:
    """Cold titles have no interactions, but they do have metadata — so the item
    tower must project the whole catalog, not just the trained subset."""
    rng = np.random.default_rng(1)
    pairs = _pairs(n_titles=6, per_title=5)
    item_emb = rng.standard_normal((200, 12)).astype(np.float32)  # 200 items, 6 trained

    result = train_projections(
        pairs,
        rng.standard_normal((len(pairs), 12)).astype(np.float32),
        item_emb,
        config=TrainConfig(output_dim=8, epochs=2, batch_size=4, patience=0),
        verbose=False,
    )
    projected = result.project_items(item_emb)

    assert projected.shape == (200, 8), "every catalog item must be projected"
    assert np.isfinite(projected).all()
    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-4)


def test_training_rejects_misaligned_embeddings() -> None:
    pairs = _pairs(n_titles=4, per_title=3)
    with pytest.raises(ValueError, match="query embeddings for"):
        train_projections(
            pairs,
            np.zeros((5, 8), np.float32),
            np.zeros((4, 8), np.float32),
            verbose=False,
        )


def test_training_rejects_item_positions_outside_the_bank() -> None:
    pairs = _pairs(n_titles=10, per_title=2)
    with pytest.raises(ValueError, match="outside item_embeddings"):
        train_projections(
            pairs,
            np.zeros((len(pairs), 8), np.float32),
            np.zeros((3, 8), np.float32),
            verbose=False,
        )


def test_training_rejects_validation_without_embeddings() -> None:
    pairs = _pairs(n_titles=6, per_title=3)
    train, val = pairs.split(val_frac=0.3, seed=1)
    with pytest.raises(ValueError, match="without val_query_embeddings"):
        train_projections(
            train,
            np.zeros((len(train), 8), np.float32),
            np.zeros((6, 8), np.float32),
            val_pairs=val,
            verbose=False,
        )
