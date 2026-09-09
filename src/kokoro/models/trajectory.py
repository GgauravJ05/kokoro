# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Mood as a curve over the runtime — the contribution this project is built on.

Every deployed anime recommender treats a title as one point in taste space.
That is wrong in a way viewers feel constantly: *Vinland Saga* season 1 and
season 2 are different emotional objects; *Steins;Gate* inverts at episode 12;
*Oshi no Ko* spends one episode establishing a register and the rest destroying
it. A single mood vector averages those into something describing neither half.

Kokoro binds review sentences to the arcs they discuss (see
:mod:`kokoro.features.arcs`), producing a sequence of mood vectors per title —
a *mood trajectory* of shape ``(n_arcs, n_axes)``. Two things follow that
point-based systems cannot do:

* **Shape queries.** "Starts light, gets devastating" is a constraint on the
  derivative of the comfort axis, not on its mean. :class:`TrajectoryEncoder`
  embeds the curve so such queries are retrievable.
* **Honest warnings.** "Comfort for eight episodes, then it is not" is
  computable from the curve, and is the single most-requested thing viewers
  cannot get from a genre tag.

Trajectories vary in length, so they are resampled to a fixed grid before
encoding. A dilated 1-D CNN then reads local shape (turns, inversions) while
the residual pooling head keeps the global register.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["TrajectoryEncoder", "resample_trajectory", "shape_distance"]


def resample_trajectory(traj: Tensor, n_points: int = 16) -> Tensor:
    """Resample a variable-length mood trajectory onto a fixed grid.

    Linear interpolation along the arc axis. A 12-episode single-cour show and
    a 64-episode epic both become ``n_points`` samples of *narrative position*,
    which is the right invariance: "turns bleak two-thirds of the way through"
    should match regardless of runtime.

    Args:
        traj: Shape ``(B, T, A)`` — batch, arcs, axes. ``T`` may vary between
            calls but not within a batch.
        n_points: Length of the output grid.

    Returns:
        Shape ``(B, n_points, A)``.

    Raises:
        ValueError: If ``traj`` is not 3-D or has no arcs.
    """
    if traj.dim() != 3:
        raise ValueError(f"expected (B, T, A), got {tuple(traj.shape)}")
    if traj.shape[1] == 0:
        raise ValueError("trajectory has no arcs")
    if traj.shape[1] == n_points:
        return traj
    # interpolate expects (B, C, T), so move the axis dimension into channels.
    return nn.functional.interpolate(
        traj.transpose(1, 2), size=n_points, mode="linear", align_corners=True
    ).transpose(1, 2)


class TrajectoryEncoder(nn.Module):
    """Encode a mood trajectory into a single retrievable vector.

    Args:
        n_axes: Width of each mood vector.
        hidden: Channel width of the convolutional stack.
        output_dim: Width of the emitted embedding.
        n_points: Grid length trajectories are resampled to.
        dilations: Dilation per convolutional block. The defaults give a
            receptive field spanning the whole resampled grid, so the top block
            sees act-length structure rather than episode-length noise.

    Raises:
        ValueError: If ``dilations`` is empty.
    """

    def __init__(
        self,
        n_axes: int = 8,
        hidden: int = 128,
        output_dim: int = 256,
        n_points: int = 16,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations must not be empty")
        self.n_points = n_points

        blocks: list[nn.Module] = []
        in_ch = n_axes
        for d in dilations:
            blocks += [
                nn.Conv1d(in_ch, hidden, kernel_size=3, padding=d, dilation=d),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
            ]
            in_ch = hidden
        self.conv = nn.Sequential(*blocks)

        # Mean pooling keeps the overall register; max pooling keeps the single
        # most extreme moment, which is what a content warning needs. Both are
        # concatenated because averaging them loses the second.
        self.head = nn.Sequential(nn.Linear(2 * hidden, output_dim), nn.GELU())

    def forward(self, traj: Tensor) -> Tensor:
        """Embed a batch of trajectories.

        Args:
            traj: Shape ``(B, T, A)``.

        Returns:
            Shape ``(B, output_dim)``.
        """
        x = resample_trajectory(traj, self.n_points).transpose(1, 2)  # (B, A, P)
        h = self.conv(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        out: Tensor = self.head(pooled)
        return out


def shape_distance(a: Tensor, b: Tensor, *, derivative_weight: float = 0.5) -> Tensor:
    """Distance between two mood trajectories, weighting *change* as well as level.

    Plain pointwise distance says two shows are similar when they sit at the
    same average mood, which is exactly the failure this module exists to fix.
    Adding the first difference makes "light then devastating" close to other
    light-then-devastating curves and far from uniformly-bleak ones, even when
    the two have identical means.

    Args:
        a: Shape ``(B, T, A)``.
        b: Shape ``(B, T, A)``, resampled to the same ``T`` as ``a``.
        derivative_weight: Relative weight of the first-difference term.

    Returns:
        Per-pair distance, shape ``(B,)``.

    Raises:
        ValueError: If the two trajectories differ in shape.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    level = (a - b).pow(2).mean(dim=(1, 2))
    if a.shape[1] < 2:
        return level.sqrt()
    slope = (a.diff(dim=1) - b.diff(dim=1)).pow(2).mean(dim=(1, 2))
    return (level + derivative_weight * slope).sqrt()
