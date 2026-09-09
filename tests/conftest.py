# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.eval.splits import Interactions


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test is deterministic."""
    return np.random.default_rng(1337)


@pytest.fixture
def interactions(rng: np.random.Generator) -> Interactions:
    """A small synthetic interaction log with a realistic long tail."""
    n, n_users, n_items = 2000, 80, 120
    # Zipf-ish item popularity: the head absorbs most interactions, which is
    # what makes the beyond-accuracy metrics meaningful.
    weights = 1.0 / np.arange(1, n_items + 1)
    weights /= weights.sum()
    return Interactions(
        user=rng.integers(0, n_users, n).astype(np.int64),
        item=rng.choice(n_items, size=n, p=weights).astype(np.int64),
        rating=rng.integers(1, 11, n).astype(np.float32),
        timestamp=np.sort(rng.integers(1_500_000_000, 1_700_000_000, n)).astype(np.int64),
    )
