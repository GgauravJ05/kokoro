# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Building mood curves from position-bearing review segments."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.features.trajectories import (
    POSITION_WEIGHTS,
    build_trajectories,
    curve_statistics,
)


def _seg(text: str, value: float, n_axes: int = 2) -> tuple[str, np.ndarray]:
    """One segment whose mood is constant across axes."""
    return text, np.full(n_axes, value, dtype=np.float64)


def _spread(value_by_region: dict[str, float], repeats: int = 4) -> list:
    """Segments spanning the runtime, so a title clears the coverage thresholds."""
    out = []
    for region, value in value_by_region.items():
        for i in range(repeats):
            out.append(_seg(f"the {region} was memorable in way {i}", value))
    return out


# ---------------------------------------------------------------------------
# Position weighting
# ---------------------------------------------------------------------------


def test_season_mentions_are_excluded() -> None:
    """Mapping "season 2" to a position needs a season count nobody has, and
    reviewers use the phrase to compare other works as often as to locate a
    moment in this one."""
    assert POSITION_WEIGHTS["season"] == 0.0


def test_precise_kinds_outrank_coarse_ones() -> None:
    assert POSITION_WEIGHTS["episode"] > POSITION_WEIGHTS["part"]
    assert POSITION_WEIGHTS["part"] > POSITION_WEIGHTS["region"]


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------


def test_builds_a_curve_that_follows_the_evidence() -> None:
    """A title described as warm at the start and bleak at the end must produce
    a falling curve — this is the whole premise in one assertion."""
    segments = _spread({"beginning": 0.9, "midpoint": 0.0, "finale": -0.9})
    traj = build_trajectories({0: segments}, n_arcs=3, min_located=6, min_arcs_covered=3)

    assert len(traj) == 1
    curve = traj.curves[0, :, 0]
    assert curve[0] > curve[1] > curve[2], f"expected a falling curve, got {curve}"


def test_rising_and_falling_curves_are_distinguishable() -> None:
    rising = _spread({"beginning": -0.9, "midpoint": 0.0, "finale": 0.9})
    falling = _spread({"beginning": 0.9, "midpoint": 0.0, "finale": -0.9})
    traj = build_trajectories({0: rising, 1: falling}, n_arcs=3, min_located=6, min_arcs_covered=3)

    a = traj.curves[traj.item_pos.tolist().index(0), :, 0]
    b = traj.curves[traj.item_pos.tolist().index(1), :, 0]
    # Identical means, opposite shapes: exactly the case a point model cannot see.
    assert a.mean() == pytest.approx(b.mean(), abs=1e-6)
    assert not np.allclose(a, b)


def test_mean_and_shape_decompose_the_curve() -> None:
    traj = build_trajectories(
        {0: _spread({"beginning": 0.8, "midpoint": 0.2, "finale": -0.4})},
        n_arcs=3,
        min_located=6,
        min_arcs_covered=3,
    )
    reconstructed = traj.shape() + traj.mean_curve()[:, None, :]
    np.testing.assert_allclose(reconstructed, traj.curves, atol=1e-12)
    assert np.allclose(traj.shape().mean(axis=1), 0.0, atol=1e-12)


def test_titles_without_enough_evidence_are_dropped() -> None:
    """A curve observed at one point is a point, not a curve."""
    only_endings = [_seg("the finale was devastating", -0.9) for _ in range(20)]
    with pytest.raises(ValueError, match="no title met the evidence thresholds"):
        build_trajectories({0: only_endings}, n_arcs=5, min_located=5, min_arcs_covered=3)


def test_segments_without_positions_contribute_nothing() -> None:
    positionless = [_seg("a beautiful and quietly sad show", 0.5) for _ in range(30)]
    with pytest.raises(ValueError, match="no title met the evidence thresholds"):
        build_trajectories({0: positionless}, n_arcs=3, min_located=1, min_arcs_covered=1)


def test_unobserved_arcs_are_interpolated_not_zeroed() -> None:
    """Leaving a gap at zero would read as 'neutral mood' and put a false dip in
    every curve that happens to lack midpoint commentary."""
    segments = _spread({"beginning": 0.8, "finale": 0.6})
    traj = build_trajectories({0: segments}, n_arcs=5, min_located=4, min_arcs_covered=2)

    curve = traj.curves[0, :, 0]
    assert curve.min() > 0.5, f"interpolated arcs dipped toward zero: {curve}"
    assert (traj.support == 0).any(), "this fixture is meant to leave arcs unobserved"


def test_support_records_which_arcs_were_observed() -> None:
    traj = build_trajectories(
        {0: _spread({"beginning": 0.5, "midpoint": 0.5, "finale": 0.5})},
        n_arcs=3,
        min_located=6,
        min_arcs_covered=3,
    )
    assert traj.support.shape == (1, 3)
    assert (traj.support > 0).all()


def test_rejects_bad_arc_count() -> None:
    with pytest.raises(ValueError, match="n_arcs must be"):
        build_trajectories({0: _spread({"beginning": 0.1})}, n_arcs=0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_curve_statistics_detect_flat_curves() -> None:
    """If within-title variation vanished, a point estimate would lose nothing
    and this module would be unjustified."""
    flat = build_trajectories(
        {0: _spread({"beginning": 0.4, "midpoint": 0.4, "finale": 0.4})},
        n_arcs=3,
        min_located=6,
        min_arcs_covered=3,
    )
    stats = curve_statistics(flat)
    assert stats["within_title_std"] == pytest.approx(0.0, abs=1e-9)
    assert stats["mean_abs_arc_step"] == pytest.approx(0.0, abs=1e-9)


def test_curve_statistics_detect_movement() -> None:
    moving = build_trajectories(
        {0: _spread({"beginning": 0.9, "midpoint": 0.0, "finale": -0.9})},
        n_arcs=3,
        min_located=6,
        min_arcs_covered=3,
    )
    stats = curve_statistics(moving)
    assert stats["within_title_std"] > 0.5
    assert stats["mean_abs_arc_step"] > 0.5
    assert stats["n_arcs"] == 3.0
