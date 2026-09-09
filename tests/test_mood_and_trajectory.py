# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""The mood bottleneck and the trajectory encoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="optional [train] extra not installed")

from kokoro.models.mood_axes import (
    MOOD_AXES,
    MoodBottleneck,
    anchor_alignment_loss,
)
from kokoro.models.trajectory import (
    TrajectoryEncoder,
    resample_trajectory,
    shape_distance,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(1337)


def test_axes_are_named_and_bipolar() -> None:
    for name, neg, pos in MOOD_AXES:
        assert name.isidentifier(), f"{name!r} must be usable as a field name"
        assert neg and pos and neg != pos


def test_bottleneck_shapes_and_bounds() -> None:
    block = MoodBottleneck(input_dim=64, n_axes=8, output_dim=128)
    recon, axes = block(torch.randn(5, 64))
    assert recon.shape == (5, 128)
    assert axes.shape == (5, 8)
    assert axes.abs().max() <= 1.0, "bounded axes must stay in [-1, 1]"


def test_unbounded_bottleneck_is_not_squashed() -> None:
    block = MoodBottleneck(input_dim=64, bounded=False)
    _, axes = block(torch.randn(64, 64) * 50)
    assert axes.abs().max() > 1.0


def test_bottleneck_rejects_too_many_axes() -> None:
    with pytest.raises(ValueError, match="axes are named"):
        MoodBottleneck(input_dim=16, n_axes=len(MOOD_AXES) + 1)


def test_explain_returns_named_axes_ordered_by_magnitude() -> None:
    block = MoodBottleneck(input_dim=32, n_axes=8)
    out = block.explain(torch.randn(3, 32), top_n=3)
    assert len(out) == 3
    for row in out:
        assert len(row) == 3
        assert all(name in block.axis_names for name, _ in row)
        assert [abs(v) for _, v in row] == sorted((abs(v) for _, v in row), reverse=True)


def test_anchor_loss_rewards_correctly_poled_anchors() -> None:
    axis = torch.tensor([0, 1])
    pole = torch.tensor([1.0, -1.0])
    good = torch.tensor([[0.9, 0.0], [0.0, -0.9]])
    bad = torch.tensor([[-0.9, 0.0], [0.0, 0.9]])
    assert anchor_alignment_loss(good, axis, pole) < anchor_alignment_loss(bad, axis, pole)
    assert anchor_alignment_loss(good, axis, pole).item() == pytest.approx(0.0)


def test_anchor_loss_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="share a batch dimension"):
        anchor_alignment_loss(torch.zeros(2, 4), torch.tensor([0]), torch.tensor([1.0]))


def test_resample_preserves_endpoints() -> None:
    traj = torch.linspace(0, 1, 5).view(1, 5, 1)
    out = resample_trajectory(traj, n_points=16)
    assert out.shape == (1, 16, 1)
    assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[0, -1, 0].item() == pytest.approx(1.0, abs=1e-5)


def test_resample_is_a_noop_at_matching_length() -> None:
    traj = torch.randn(2, 16, 4)
    assert resample_trajectory(traj, 16) is traj


def test_resample_rejects_empty_and_wrong_rank() -> None:
    with pytest.raises(ValueError, match="no arcs"):
        resample_trajectory(torch.zeros(1, 0, 3))
    with pytest.raises(ValueError, match=r"expected \(B, T, A\)"):
        resample_trajectory(torch.zeros(4, 3))


def test_trajectory_encoder_output_shape() -> None:
    encoder = TrajectoryEncoder(n_axes=8, output_dim=64, n_points=16).eval()
    assert encoder(torch.randn(4, 11, 8)).shape == (4, 64)


def test_trajectory_encoder_handles_variable_lengths() -> None:
    encoder = TrajectoryEncoder(n_axes=8, output_dim=64).eval()
    short = encoder(torch.randn(2, 5, 8))
    long = encoder(torch.randn(2, 40, 8))
    assert short.shape == long.shape


def test_shape_distance_separates_curves_with_equal_means() -> None:
    """The whole point of the trajectory model, as a single assertion."""
    t = torch.linspace(-1, 1, 16).view(1, 16, 1)
    rising = t.clone()  # light -> devastating
    falling = -t.clone()  # devastating -> light
    flat = torch.zeros_like(t)  # uniformly neutral
    # All three have mean zero, so any point-based metric calls them identical.
    assert rising.mean().abs() < 1e-6
    assert shape_distance(rising, falling) > shape_distance(rising, flat)


def test_shape_distance_is_zero_for_identical_curves() -> None:
    traj = torch.randn(3, 16, 8)
    assert shape_distance(traj, traj.clone()).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_shape_distance_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        shape_distance(torch.zeros(1, 16, 4), torch.zeros(1, 8, 4))
