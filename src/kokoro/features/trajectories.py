# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Assemble per-title mood curves from position-bearing review segments.

This is the data layer under the project's central claim: that a title is a
*curve* over narrative position rather than a point. Every segment that says
where in the runtime it is talking about contributes its mood to that position.

**What the corpus actually supports.** Measured on the real reviews, 8.0% of
segments carry a resolvable position, which is a median of 194 located segments
per title — enough for a curve. The composition is the important part:

===================  ======  ===================================================
kind                 share   note
===================  ======  ===================================================
``region``           62%     "the opening", "the finale" — coarse but abundant
``season``           20%     excluded below; see :data:`POSITION_WEIGHTS`
``part``             10%     "the first cour", "the final act"
``episode``           8%     an explicit number, the only precise signal
``named_arc``        0.03%   dead: the source text is lowercased
===================  ======  ===================================================

So the resolution available is roughly *beginning / middle / end*, not
per-episode. :func:`build_trajectories` therefore defaults to a small number of
arcs; asking for twenty would manufacture detail the evidence cannot support.

``season`` mentions are excluded outright. Mapping "season 2" to a position
needs a season count nobody has, and reviewers use the phrase to compare *other*
works as often as to locate a moment in this one — it is noise wearing the
costume of signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kokoro.features.arcs import extract_arc_mentions, to_arc_index

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

__all__ = ["TrajectorySet", "build_trajectories", "curve_statistics"]

#: Per-kind trust, multiplied into each segment's contribution. ``season`` is
#: zero rather than absent so that the exclusion is visible and revertible.
POSITION_WEIGHTS: dict[str, float] = {
    "episode_range": 1.0,
    "episode": 1.0,
    "chapter_range": 1.0,
    "chapter": 1.0,
    "part": 0.8,
    "named_arc": 0.8,
    "region": 0.6,
    "season": 0.0,
}


@dataclass(slots=True)
class TrajectorySet:
    """Mood curves for the titles that had enough located evidence.

    Attributes:
        item_pos: Contiguous item position per curve, shape ``(n_titles,)``.
        curves: Mood curves, shape ``(n_titles, n_arcs, n_axes)``.
        support: Weighted segment count per arc, shape ``(n_titles, n_arcs)``.
            An arc with low support is interpolated, not observed, and this is
            what says which.
        axis_names: Name of each axis, in order.
    """

    item_pos: npt.NDArray[np.int64]
    curves: npt.NDArray[np.float64]
    support: npt.NDArray[np.float64]
    axis_names: tuple[str, ...]

    def __len__(self) -> int:
        """Number of titles with a curve."""
        return int(self.item_pos.shape[0])

    @property
    def n_arcs(self) -> int:
        """Arcs per curve."""
        return int(self.curves.shape[1])

    def mean_curve(self) -> npt.NDArray[np.float64]:
        """Return the per-title mean over arcs, shape ``(n_titles, n_axes)``.

        This is the point estimate a conventional recommender would use, and the
        baseline the trajectory has to beat to justify itself.
        """
        return self.curves.mean(axis=1)

    def shape(self) -> npt.NDArray[np.float64]:
        """Return curves with each title's mean removed.

        What is left is purely *how the mood moves*, with the overall register
        subtracted — the signal a point-based model cannot represent at all.
        """
        return self.curves - self.curves.mean(axis=1, keepdims=True)


def build_trajectories(
    segments_by_title: Mapping[int, Sequence[tuple[str, npt.NDArray[np.float64]]]],
    *,
    episodes_by_title: Mapping[int, int | None] | None = None,
    n_arcs: int = 5,
    min_located: int = 10,
    min_arcs_covered: int = 3,
    axis_names: Sequence[str] = (),
) -> TrajectorySet:
    """Bucket located segments into arcs and average their mood per arc.

    Args:
        segments_by_title: Item position to a sequence of
            ``(segment_text, axis_values)`` pairs. ``axis_values`` is that
            segment's mood vector.
        episodes_by_title: Item position to episode count, used to turn an
            absolute episode number into a narrative position.
        n_arcs: Buckets per curve. Kept small on purpose: the evidence resolves
            to roughly beginning/middle/end, so a finer grid would invent
            detail.
        min_located: Titles with fewer located segments are dropped.
        min_arcs_covered: Titles whose evidence touches fewer arcs than this are
            dropped — a "curve" observed at one point is a point.
        axis_names: Names of the axis columns, for reporting.

    Returns:
        The assembled :class:`TrajectorySet`.

    Raises:
        ValueError: If ``n_arcs`` is not positive, or no title qualifies.
    """
    if n_arcs < 1:
        raise ValueError(f"n_arcs must be >= 1, got {n_arcs}")

    kept_pos: list[int] = []
    kept_curves: list[npt.NDArray[np.float64]] = []
    kept_support: list[npt.NDArray[np.float64]] = []

    for pos, segments in segments_by_title.items():
        if not segments:
            continue
        n_axes = int(np.asarray(segments[0][1]).shape[0])
        total = np.zeros((n_arcs, n_axes), dtype=np.float64)
        weight = np.zeros(n_arcs, dtype=np.float64)
        located = 0

        episodes = (episodes_by_title or {}).get(pos)
        for text, values in segments:
            mentions = extract_arc_mentions(text, total_episodes=episodes)
            best = _best_mention(mentions)
            if best is None:
                continue
            position, trust = best
            arc = to_arc_index(position, n_arcs)
            total[arc] += np.asarray(values, dtype=np.float64) * trust
            weight[arc] += trust
            located += 1

        if located < min_located or int((weight > 0).sum()) < min_arcs_covered:
            continue

        curve = _fill_gaps(total, weight)
        kept_pos.append(pos)
        kept_curves.append(curve)
        kept_support.append(weight)

    if not kept_pos:
        raise ValueError(
            "no title met the evidence thresholds — lower min_located/min_arcs_covered "
            "or check that segments carry position references"
        )

    return TrajectorySet(
        item_pos=np.asarray(kept_pos, dtype=np.int64),
        curves=np.stack(kept_curves),
        support=np.stack(kept_support),
        axis_names=tuple(axis_names),
    )


def _best_mention(mentions: Sequence[object]) -> tuple[float, float] | None:
    """Pick the most trustworthy positioned mention in a segment.

    Args:
        mentions: Output of :func:`~kokoro.features.arcs.extract_arc_mentions`.

    Returns:
        ``(position, trust)``, or ``None`` when nothing usable was found.
    """
    best: tuple[float, float] | None = None
    for m in mentions:
        position = getattr(m, "position", None)
        if position is None:
            continue
        trust = POSITION_WEIGHTS.get(getattr(m, "kind", ""), 0.0)
        trust *= float(getattr(m, "confidence", 0.0))
        if trust <= 0.0:
            continue
        if best is None or trust > best[1]:
            best = (float(position), trust)
    return best


def _fill_gaps(
    total: npt.NDArray[np.float64], weight: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Average observed arcs and interpolate the unobserved ones.

    An arc nobody wrote about is filled by linear interpolation between its
    observed neighbours rather than left at zero, which would read as "neutral
    mood" and put a false dip in every curve.
    """
    n_arcs, n_axes = total.shape
    curve = np.zeros_like(total)
    seen = weight > 0
    curve[seen] = total[seen] / weight[seen, None]

    if seen.all():
        return curve
    idx = np.arange(n_arcs)
    for axis in range(n_axes):
        curve[:, axis] = np.interp(idx, idx[seen], curve[seen, axis])
    return curve


def curve_statistics(trajectories: TrajectorySet) -> dict[str, float]:
    """Summarise whether the curves carry shape at all.

    The premise of this whole module is that mood *moves* within a title. If
    within-title variation were negligible next to between-title variation, a
    single point per title would lose nothing and the trajectory model would be
    unjustified. This is the number that says which world we are in.

    Args:
        trajectories: The set to summarise.

    Returns:
        Within- and between-title standard deviations, their ratio, and the mean
        absolute arc-to-arc change.
    """
    curves = trajectories.curves
    within = float(curves.std(axis=1).mean())
    between = float(curves.mean(axis=1).std(axis=0).mean())
    steps = np.abs(np.diff(curves, axis=1))
    return {
        "n_titles": float(len(trajectories)),
        "n_arcs": float(trajectories.n_arcs),
        "within_title_std": round(within, 5),
        "between_title_std": round(between, 5),
        "within_over_between": round(within / between, 4) if between > 1e-12 else float("nan"),
        "mean_abs_arc_step": round(float(steps.mean()), 5),
        "mean_arc_support": round(float(trajectories.support.mean()), 3),
        "interpolated_arc_fraction": round(float((trajectories.support == 0).mean()), 4),
    }
