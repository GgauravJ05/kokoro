# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Mood-axis probing, and the anchor term that is supposed to shape the axes."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.eval.axes import (
    AXIS_PROBES,
    UNVALIDATED_AXES,
    AxisReport,
    evaluate_axes,
    probe_tags,
    summarise,
)

torch = pytest.importorskip("torch", reason="optional [train] extra not installed")

from kokoro.data.pairs import PairSet
from kokoro.models.mood_axes import AXIS_NAMES
from kokoro.train.contrastive import TrainConfig, train_projections

# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------


def test_probe_tags_exist_in_the_corpus_vocabulary() -> None:
    """Probes built from invented tag names would validate nothing."""
    import pandas as pd

    path = "data/processed/titles.parquet"
    try:
        titles = pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        pytest.skip("corpus not built")

    vocabulary = {t for tags in titles["tags"] for t in tags}
    missing = probe_tags() - vocabulary
    assert not missing, f"probe tags absent from the corpus: {sorted(missing)}"


def test_probe_poles_are_disjoint() -> None:
    """A tag on both poles of one axis would make that probe meaningless."""
    for axis, (positive, negative) in AXIS_PROBES.items():
        assert not set(positive) & set(negative), f"{axis} has a tag on both poles"


def test_unvalidated_axes_are_declared_not_silently_absent() -> None:
    """An axis nobody checked must be visible as such."""
    for axis in UNVALIDATED_AXES:
        assert axis in AXIS_NAMES
        assert axis not in AXIS_PROBES


def test_every_probed_axis_is_a_real_axis() -> None:
    assert set(AXIS_PROBES) <= set(AXIS_NAMES)


# ---------------------------------------------------------------------------
# AUC behaviour
# ---------------------------------------------------------------------------


def _values(n_items: int, comfort: np.ndarray) -> np.ndarray:
    """Build an axis matrix whose `comfort` column is supplied."""
    out = np.zeros((n_items, len(AXIS_NAMES)))
    out[:, AXIS_NAMES.index("comfort")] = comfort
    return out


def test_perfect_separation_scores_auc_one() -> None:
    tags = {0: ["Iyashikei"], 1: ["Iyashikei"], 2: ["Tragedy"], 3: ["Gore"]}
    values = _values(4, np.array([1.0, 0.9, -0.9, -1.0]))
    report = next(r for r in evaluate_axes(values, list(AXIS_NAMES), tags) if r.axis == "comfort")
    assert report.auc == pytest.approx(1.0)
    assert report.cohens_d > 0


def test_inverted_axis_scores_below_half() -> None:
    """The failure mode worth naming: the axis works but points the wrong way."""
    tags = {0: ["Iyashikei"], 1: ["Iyashikei"], 2: ["Tragedy"], 3: ["Gore"]}
    values = _values(4, np.array([-1.0, -0.9, 0.9, 1.0]))
    report = next(r for r in evaluate_axes(values, list(AXIS_NAMES), tags) if r.axis == "comfort")
    assert report.auc == pytest.approx(0.0)


def test_constant_axis_scores_exactly_chance() -> None:
    """A collapsed axis must read as 0.5, not as an artefact of sort order."""
    tags = {0: ["Iyashikei"], 1: ["Tragedy"]}
    values = _values(2, np.array([0.3, 0.3]))
    report = next(r for r in evaluate_axes(values, list(AXIS_NAMES), tags) if r.axis == "comfort")
    assert report.auc == pytest.approx(0.5)


def test_titles_carrying_both_poles_are_excluded() -> None:
    """A comedy about suicide is genuinely ambiguous and must inform neither side."""
    tags = {0: ["Iyashikei", "Tragedy"], 1: ["Iyashikei"], 2: ["Tragedy"]}
    values = _values(3, np.array([0.0, 1.0, -1.0]))
    report = next(r for r in evaluate_axes(values, list(AXIS_NAMES), tags) if r.axis == "comfort")
    assert report.n_positive == 1
    assert report.n_negative == 1


def test_unprobed_axes_are_reported_as_inconclusive() -> None:
    tags = {0: ["Iyashikei"], 1: ["Tragedy"]}
    reports = evaluate_axes(_values(2, np.array([1.0, -1.0])), list(AXIS_NAMES), tags)
    by_name = {r.axis: r for r in reports}
    for axis in UNVALIDATED_AXES:
        assert not by_name[axis].conclusive
        assert "inconclusive" in by_name[axis].verdict
    assert len(reports) == len(AXIS_NAMES), "no axis may be silently dropped"


def test_low_support_is_inconclusive_not_a_pass() -> None:
    tags = {0: ["Iyashikei"], 1: ["Tragedy"]}
    report = next(
        r
        for r in evaluate_axes(_values(2, np.array([1.0, -1.0])), list(AXIS_NAMES), tags)
        if r.axis == "comfort"
    )
    assert report.auc == pytest.approx(1.0)
    assert not report.conclusive, "two titles cannot validate an axis"


def test_evaluate_axes_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="axis columns but"):
        evaluate_axes(np.zeros((4, 3)), ["a", "b"], {})


@pytest.mark.parametrize(
    ("auc", "expected"),
    [(0.85, "validated"), (0.65, "weak"), (0.50, "not validated"), (0.30, "INVERTED")],
)
def test_verdicts(auc: float, expected: str) -> None:
    report = AxisReport("x", auc, 0.0, 100, 100, conclusive=True)
    assert expected in report.verdict


def test_summarise_counts_categories() -> None:
    reports = [
        AxisReport("a", 0.80, 1.0, 50, 50, True),
        AxisReport("b", 0.65, 0.5, 50, 50, True),
        AxisReport("c", 0.30, -1.0, 50, 50, True),
        AxisReport("d", float("nan"), float("nan"), 0, 0, False),
    ]
    s = summarise(reports)
    assert s["n_axes"] == 4
    assert s["n_conclusive"] == 3
    assert s["n_validated"] == 1
    assert s["n_weak"] == 1
    assert s["n_inverted"] == 1


# ---------------------------------------------------------------------------
# The anchor term
# ---------------------------------------------------------------------------


def _toy_training(anchor_weight: float, anchors: object | None) -> np.ndarray:
    """Train a tiny bottleneck model and return its axis values."""
    rng = np.random.default_rng(0)
    n_titles, per_title, dim = 12, 8, 32
    pairs = PairSet(
        queries=[f"q{t}-{i}" for t in range(n_titles) for i in range(per_title)],
        item_pos=np.repeat(np.arange(n_titles, dtype=np.int64), per_title),
        anime_ids=np.repeat(np.arange(n_titles, dtype=np.int64), per_title),
    )
    result = train_projections(
        pairs,
        rng.standard_normal((len(pairs), dim)).astype(np.float32),
        rng.standard_normal((n_titles, dim)).astype(np.float32),
        config=TrainConfig(
            output_dim=16,
            epochs=15,
            batch_size=6,
            seed=1,
            use_bottleneck=True,
            n_axes=8,
            anchor_weight=anchor_weight,
            patience=0,
        ),
        anchors=anchors,  # type: ignore[arg-type]
        verbose=False,
    )
    return result.axis_values(rng.standard_normal((n_titles, dim)).astype(np.float32))


def test_anchor_term_actually_reaches_the_optimiser() -> None:
    """Regression test for a silent no-op.

    The anchor argument was once dropped between the trainer and its epoch
    helper, so every anchor weight — including 20.0 — produced byte-identical
    axis values. Nothing raised; the ablation simply reported four copies of the
    same run. This asserts the term has an effect at all.
    """
    dim = 32
    rng = np.random.default_rng(1)
    anchors = (
        torch.as_tensor(rng.standard_normal((16, dim)), dtype=torch.float32),
        torch.arange(8).repeat_interleave(2),
        torch.tensor([-1.0, 1.0] * 8),
    )
    without = _toy_training(0.0, None)
    with_anchor = _toy_training(50.0, anchors)

    assert not np.allclose(without, with_anchor), (
        "the anchor weight changed nothing — the term is not reaching the loss"
    )
    assert np.abs(with_anchor).max() > np.abs(without).max(), (
        "the anchor term should push axis values toward their poles"
    )


def test_bottleneck_run_exposes_axis_values() -> None:
    values = _toy_training(0.0, None)
    assert values.shape[1] == 8
    assert np.isfinite(values).all()
    assert np.abs(values).max() <= 1.0, "bounded axes must stay within [-1, 1]"


def test_axis_values_require_a_bottleneck() -> None:
    rng = np.random.default_rng(2)
    pairs = PairSet(
        queries=["a", "b", "c", "d"],
        item_pos=np.array([0, 1, 2, 3], np.int64),
        anime_ids=np.array([0, 1, 2, 3], np.int64),
    )
    result = train_projections(
        pairs,
        rng.standard_normal((4, 16)).astype(np.float32),
        rng.standard_normal((4, 16)).astype(np.float32),
        config=TrainConfig(output_dim=8, epochs=1, batch_size=2, use_bottleneck=False),
        verbose=False,
    )
    with pytest.raises(TypeError, match="without a mood bottleneck"):
        result.axis_values(rng.standard_normal((4, 16)).astype(np.float32))
