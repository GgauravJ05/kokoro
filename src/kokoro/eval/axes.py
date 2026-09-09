# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Validating that the mood axes mean what their names say.

A low-dimensional bottleneck with axis labels attached is not an interpretable
model; it is a low-dimensional bottleneck with labels attached. The claim that
axis 0 is *comfort* is an empirical claim, and this module is what tests it.

**The probe.** Each axis is given two sets of AniList tags that a human would
place at opposite poles — ``Iyashikei`` and ``Cute Girls Doing Cute Things``
against ``Tragedy``, ``Suicide`` and ``Gore`` for *comfort*. If the axis means
what it says, titles carrying the positive tags should sit higher on it than
titles carrying the negative ones. That is measured with ROC-AUC, which is
rank-based and unaffected by the axis's arbitrary scale, plus Cohen's *d* for
effect size.

**The probe tags are held out of training.** :func:`probe_tags` returns every
tag used here, and the item text builder excludes them, so the model never sees
``Tragedy`` written on a title whose *comfort* score it is being graded on.
Without that exclusion the test would be circular and would pass trivially.

**Not every axis is testable.** ``pace`` and ``catharsis`` have no tag proxy in
this vocabulary: nothing in 348 AniList tags distinguishes a slow show from a
frenetic one, or an ending that resolves from one that withholds. They are
reported as *unvalidated* rather than quietly dropped or assumed to work. That
is the whole point of doing this — an axis nobody checked is a decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

__all__ = ["AXIS_PROBES", "AxisReport", "evaluate_axes", "probe_tags"]

#: Tag probes per axis, as ``(positive_pole_tags, negative_pole_tags)``.
#:
#: Every tag here was checked to exist in the corpus with at least ~30 titles;
#: inventing plausible-sounding tag names would make this validation fictional.
#: Axes absent from this mapping have no usable proxy and are reported as
#: unvalidated.
AXIS_PROBES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "comfort": (
        ("Iyashikei", "Cute Girls Doing Cute Things", "Educational"),
        ("Tragedy", "Suicide", "Gore", "Body Horror"),
    ),
    "hope": (
        ("Iyashikei", "Cute Girls Doing Cute Things", "Found Family"),
        ("Dystopian", "Post-Apocalyptic", "Tragedy", "Suicide"),
    ),
    "levity": (
        ("Parody", "Slapstick", "Surreal Comedy", "Satire"),
        ("Tragedy", "War", "Gore", "Suicide"),
    ),
    "cognition": (
        ("Philosophy", "Time Manipulation", "Memory Manipulation", "Politics"),
        ("Slapstick", "Gore", "Body Horror"),
    ),
    "intensity": (
        ("Gore", "War", "Body Horror", "Cosmic Horror"),
        ("Iyashikei", "Cute Girls Doing Cute Things", "Educational"),
    ),
    "intimacy": (
        ("Found Family", "Ensemble Cast"),
        ("Cosmic Horror", "Dystopian", "Post-Apocalyptic"),
    ),
}

#: Axes with no tag proxy in this vocabulary. Listed explicitly so a reader can
#: see what was *not* tested rather than inferring it from absence.
UNVALIDATED_AXES: tuple[str, ...] = ("pace", "catharsis")

#: Minimum titles per pole for a probe to be reported as conclusive.
MIN_SUPPORT = 15


def probe_tags() -> frozenset[str]:
    """Return every tag used by any probe.

    These must be excluded from the text the item tower encodes, or the
    validation is circular: the model would simply be reading the answer.

    Returns:
        The full set of probe tags.
    """
    out: set[str] = set()
    for positive, negative in AXIS_PROBES.values():
        out.update(positive)
        out.update(negative)
    return frozenset(out)


@dataclass(frozen=True, slots=True)
class AxisReport:
    """The outcome of probing one axis.

    Attributes:
        axis: Axis name.
        auc: ROC-AUC of the axis value separating positive-pole from
            negative-pole titles. ``0.5`` is chance; below ``0.5`` means the
            axis runs backwards relative to its name.
        cohens_d: Standardised mean difference between the two poles.
        n_positive: Titles carrying a positive-pole tag.
        n_negative: Titles carrying a negative-pole tag.
        conclusive: Whether both poles cleared :data:`MIN_SUPPORT`.
    """

    axis: str
    auc: float
    cohens_d: float
    n_positive: int
    n_negative: int
    conclusive: bool

    @property
    def verdict(self) -> str:
        """A short human-readable judgement of this axis."""
        if not self.conclusive:
            return "inconclusive (too few titles)"
        if self.auc >= 0.70:
            return "validated"
        if self.auc >= 0.60:
            return "weak"
        if self.auc <= 0.40:
            return "INVERTED — the axis runs opposite to its name"
        return "not validated"


def _auc(positive: npt.NDArray[np.float64], negative: npt.NDArray[np.float64]) -> float:
    """Return ROC-AUC via the rank-sum identity.

    Equivalent to the probability that a randomly chosen positive scores above a
    randomly chosen negative, with ties counted as half.
    """
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)

    # Average ranks within ties so a constant axis scores exactly 0.5 rather
    # than depending on sort order.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(counts.size)
    np.add.at(tie_sum, inverse, ranks)
    ranks = (tie_sum / counts)[inverse]

    n_pos = positive.size
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * negative.size))


def _cohens_d(positive: npt.NDArray[np.float64], negative: npt.NDArray[np.float64]) -> float:
    """Return the pooled-standard-deviation effect size."""
    if positive.size < 2 or negative.size < 2:
        return float("nan")
    var_p, var_n = positive.var(ddof=1), negative.var(ddof=1)
    pooled = np.sqrt(
        ((positive.size - 1) * var_p + (negative.size - 1) * var_n)
        / (positive.size + negative.size - 2)
    )
    if pooled < 1e-12:
        return 0.0
    return float((positive.mean() - negative.mean()) / pooled)


def evaluate_axes(
    axis_values: npt.NDArray[np.float64],
    axis_names: Sequence[str],
    item_tags: Mapping[int, Iterable[str]],
) -> list[AxisReport]:
    """Probe every axis against its tag poles.

    Args:
        axis_values: Learned axis values, shape ``(n_items, n_axes)``, indexed
            by contiguous item position.
        axis_names: Name of each column, in order.
        item_tags: Item position to its tag list. Positions absent from this
            mapping are skipped.

    Returns:
        One :class:`AxisReport` per named axis, in the order given. Axes with no
        probe are reported as inconclusive rather than omitted.

    Raises:
        ValueError: If ``axis_values`` does not match ``axis_names``.
    """
    if axis_values.ndim != 2:
        raise ValueError(f"expected (n_items, n_axes), got {axis_values.shape}")
    if axis_values.shape[1] != len(axis_names):
        raise ValueError(f"{axis_values.shape[1]} axis columns but {len(axis_names)} names")

    tag_sets = {pos: set(tags) for pos, tags in item_tags.items()}
    reports: list[AxisReport] = []

    for col, name in enumerate(axis_names):
        probe = AXIS_PROBES.get(name)
        if probe is None:
            reports.append(
                AxisReport(
                    axis=name,
                    auc=float("nan"),
                    cohens_d=float("nan"),
                    n_positive=0,
                    n_negative=0,
                    conclusive=False,
                )
            )
            continue

        positive_tags, negative_tags = set(probe[0]), set(probe[1])
        pos_vals: list[float] = []
        neg_vals: list[float] = []

        for pos, tags in tag_sets.items():
            if pos >= axis_values.shape[0]:
                continue
            in_pos = bool(tags & positive_tags)
            in_neg = bool(tags & negative_tags)
            # A title carrying both poles is genuinely ambiguous on this axis
            # (a comedy about suicide), so it informs neither side.
            if in_pos and not in_neg:
                pos_vals.append(float(axis_values[pos, col]))
            elif in_neg and not in_pos:
                neg_vals.append(float(axis_values[pos, col]))

        p = np.asarray(pos_vals, dtype=np.float64)
        n = np.asarray(neg_vals, dtype=np.float64)
        reports.append(
            AxisReport(
                axis=name,
                auc=_auc(p, n),
                cohens_d=_cohens_d(p, n),
                n_positive=p.size,
                n_negative=n.size,
                conclusive=p.size >= MIN_SUPPORT and n.size >= MIN_SUPPORT,
            )
        )
    return reports


def summarise(reports: Sequence[AxisReport]) -> dict[str, float | int]:
    """Aggregate axis reports into headline numbers.

    Args:
        reports: Output of :func:`evaluate_axes`.

    Returns:
        Counts of validated, weak, failed and inconclusive axes, plus the mean
        AUC over conclusive ones.
    """
    conclusive = [r for r in reports if r.conclusive]
    return {
        "n_axes": len(reports),
        "n_conclusive": len(conclusive),
        "n_validated": sum(1 for r in conclusive if r.auc >= 0.70),
        "n_weak": sum(1 for r in conclusive if 0.60 <= r.auc < 0.70),
        "n_inverted": sum(1 for r in conclusive if r.auc <= 0.40),
        "mean_auc": float(np.mean([r.auc for r in conclusive])) if conclusive else float("nan"),
    }
