# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""The interpretable mood bottleneck.

A dense retriever's 768-dimensional embedding is a black box: it can rank well
and still not tell a user *why*. Kokoro forces the item representation through a
low-dimensional bottleneck whose axes are anchored to named affective poles, so
every recommendation comes with a readable explanation and a radar chart.

The axes are bipolar and deliberately few. Each is defined by two natural
language anchor phrases, and an :func:`anchor_alignment_loss` term pulls the
bottleneck so that the projection of an anchor's encoding lands at that axis's
pole. Without that term the bottleneck is still a bottleneck, but the axes are
an arbitrary rotation and the interpretability claim is false — so the ablation
that removes it is the one that proves the claim is real.

Validation is external and non-negotiable: :mod:`kokoro.eval.harness` scores
learned axis values against human mood judgements and reports Spearman
correlation per axis. An axis that does not correlate is reported as such, not
quietly dropped.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["MOOD_AXES", "MoodBottleneck", "anchor_alignment_loss"]

#: The bipolar axes, as ``(name, negative_pole, positive_pole)``. Chosen because
#: each is (a) something viewers actually say in reviews, (b) not recoverable
#: from a genre tag, and (c) actionable — a user can state a preference on it.
MOOD_AXES: tuple[tuple[str, str, str], ...] = (
    ("comfort", "devastating, leaves you hollow", "warm, safe, easy to return to"),
    ("pace", "slow, contemplative, drifting", "relentless, frenetic, breathless"),
    ("hope", "bleak, nihilistic, no way out", "hopeful, earnest, worth believing in"),
    ("intimacy", "lonely, isolating, alienated", "close, tender, deeply connected"),
    ("cognition", "visceral, felt in the body", "cerebral, puzzle-like, demands thought"),
    ("levity", "heavy, serious, no relief", "funny, light, playful"),
    ("catharsis", "unresolved, withholding", "cathartic, earns its ending"),
    ("intensity", "gentle, low-stakes", "overwhelming, operatic"),
)

#: Axis names in bottleneck order.
AXIS_NAMES: tuple[str, ...] = tuple(name for name, _, _ in MOOD_AXES)


class MoodBottleneck(nn.Module):
    """Project a dense encoding down to named mood axes and back up.

    The ``down`` projection is what a user sees; ``up`` exists so retrieval can
    still happen in the wide space, letting the ablation table separate "the
    bottleneck cost us accuracy" from "the bottleneck bought us explanation".

    Args:
        input_dim: Width of the incoming encoder representation.
        n_axes: Number of mood axes. Defaults to ``len(MOOD_AXES)``.
        output_dim: Width of the reconstructed retrieval space. Defaults to
            ``input_dim``.
        bounded: Squash axis values through ``tanh`` into ``[-1, 1]``, so a
            value is directly readable as a position between the two poles.

    Raises:
        ValueError: If ``n_axes`` exceeds the number of named axes.
    """

    def __init__(
        self,
        input_dim: int,
        n_axes: int = len(MOOD_AXES),
        output_dim: int | None = None,
        bounded: bool = True,
    ) -> None:
        super().__init__()
        if n_axes > len(MOOD_AXES):
            raise ValueError(f"only {len(MOOD_AXES)} axes are named, got n_axes={n_axes}")
        self.n_axes = n_axes
        self.bounded = bounded
        self.axis_names = AXIS_NAMES[:n_axes]
        self.down = nn.Linear(input_dim, n_axes)
        self.up = nn.Linear(n_axes, output_dim or input_dim)

    def axes(self, x: Tensor) -> Tensor:
        """Return the mood-axis values for ``x``, shape ``(B, n_axes)``."""
        z = self.down(x)
        return torch.tanh(z) if self.bounded else z

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(reconstructed_embedding, axis_values)``.

        Args:
            x: Encoder output, shape ``(B, input_dim)``.

        Returns:
            The reconstruction of shape ``(B, output_dim)`` and the axis values
            of shape ``(B, n_axes)``.
        """
        z = self.axes(x)
        return self.up(z), z

    def explain(self, x: Tensor, top_n: int = 3) -> list[list[tuple[str, float]]]:
        """Return the ``top_n`` most extreme axes per row, for user-facing text.

        Args:
            x: Encoder output, shape ``(B, input_dim)``.
            top_n: Axes to report per row.

        Returns:
            One list of ``(axis_name, value)`` pairs per row, ordered by
            absolute magnitude. A value near ``-1`` means the negative pole.
        """
        z = self.axes(x).detach()
        idx = z.abs().argsort(dim=-1, descending=True)[:, :top_n]
        return [
            [(self.axis_names[int(j)], float(z[i, j])) for j in row] for i, row in enumerate(idx)
        ]


def anchor_alignment_loss(axis_values: Tensor, anchor_axis: Tensor, anchor_pole: Tensor) -> Tensor:
    """Pull anchor encodings toward their declared pole on their declared axis.

    This is the term that makes the axes mean what their names say. Encode the
    anchor phrases from :data:`MOOD_AXES` through the item tower, pass their
    bottleneck values here, and the loss penalises any anchor whose value on its
    own axis falls short of its pole.

    Args:
        axis_values: Bottleneck values for the anchors, shape ``(B, n_axes)``.
        anchor_axis: Which axis each anchor belongs to, shape ``(B,)``.
        anchor_pole: Target pole per anchor, ``-1.0`` or ``+1.0``, shape ``(B,)``.

    Returns:
        A scalar loss tensor.

    Raises:
        ValueError: If the batch dimensions disagree.
    """
    if not axis_values.shape[0] == anchor_axis.shape[0] == anchor_pole.shape[0]:
        raise ValueError("axis_values, anchor_axis and anchor_pole must share a batch dimension")
    own = axis_values.gather(1, anchor_axis.view(-1, 1)).squeeze(1)
    # Hinge rather than MSE: we want the anchor pushed *past* 0.8 toward its
    # pole, not pinned exactly at 1.0, which would over-constrain the space.
    return torch.relu(0.8 - anchor_pole * own).mean()
