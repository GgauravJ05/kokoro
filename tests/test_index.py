# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Vector index behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.index.ann import ExactIndex

hnswlib = pytest.importorskip("hnswlib", reason="optional [index] extra not installed")


def test_exact_index_finds_the_planted_neighbour(rng: np.random.Generator) -> None:
    vectors = rng.standard_normal((200, 32)).astype(np.float32)
    index = ExactIndex().build(vectors)
    ids, scores = index.search(vectors[:5], k=1)
    np.testing.assert_array_equal(ids.ravel(), np.arange(5))
    assert np.allclose(scores.ravel(), 1.0, atol=1e-5)


def test_exact_index_requires_build() -> None:
    with pytest.raises(RuntimeError, match="call build"):
        ExactIndex().search(np.zeros((1, 4), np.float32))


def test_hnsw_recall_against_exact(rng: np.random.Generator) -> None:
    """The claim the README makes about ANN recall is checked here."""
    from kokoro.index.ann import HNSWIndex

    vectors = rng.standard_normal((2000, 64)).astype(np.float32)
    queries = rng.standard_normal((100, 64)).astype(np.float32)

    exact_ids, _ = ExactIndex().build(vectors).search(queries, k=10)
    approx_ids, _ = HNSWIndex(ef_search=128).build(vectors).search(queries, k=10)

    recall = np.mean(
        [
            len(set(a.tolist()) & set(b.tolist())) / 10
            for a, b in zip(exact_ids, approx_ids, strict=True)
        ]
    )
    assert recall >= 0.95, f"HNSW recall@10 was {recall:.3f}, below the 0.95 target"
