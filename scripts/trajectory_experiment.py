#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Does a mood trajectory carry information that its own mean does not?

This is the experiment the project's central claim rests on. Everything else
here can be true — the curves can be built, they can look plausible, they can
even improve retrieval — and the claim still fails if the *shape* of a curve is
just noise around its average. A point-based recommender would then lose
nothing, and modelling trajectories would be decoration.

The test is deliberately unforgiving and needs no human labels. For each
withheld AniList tag, predict whether a title carries it from:

    mean-only   the per-title average mood vector          (what a point model has)
    shape-only  the curve with its own mean subtracted     (pure movement)
    full        both                                       (what the curve has)

If ``shape-only`` beats chance, movement carries signal. If ``full`` beats
``mean-only``, that signal is not already in the average. Scoring is 5-fold
cross-validated ROC-AUC with logistic regression, so a richer feature vector
does not win by capacity alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kokoro.data.corpus import load_corpus
from kokoro.data.pairs import build_pairs, is_boilerplate
from kokoro.eval.axes import probe_tags
from kokoro.features.arcs import segment_review
from kokoro.features.trajectories import build_trajectories, curve_statistics
from kokoro.models.content import encode_texts
from kokoro.models.mood_axes import AXIS_NAMES
from kokoro.train.contrastive import (
    TrainConfig,
    build_anchor_batch,
    train_projections,
)

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("artifacts/trajectory")
N_ARCS = 5
MIN_TAG_SUPPORT = 25


def main() -> int:
    """Build trajectories and test shape against mean."""
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus("data/processed", seed=1337)
    held = probe_tags()

    # --- train the mood bottleneck so segments have axis values at all -------
    pairs = build_pairs(corpus, max_per_title=150, seed=1337)
    train_pairs, val_pairs = pairs.split(val_frac=0.1, seed=1337)
    item_emb = encode_texts(corpus.item_texts(exclude_tags=held), model_name=ENCODER)
    all_q = encode_texts(pairs.queries, model_name=ENCODER)
    val_titles = set(val_pairs.item_pos.tolist())
    is_val = np.array([int(p) in val_titles for p in pairs.item_pos])

    anchors = build_anchor_batch(lambda p: encode_texts(p, model_name=ENCODER), 8, "cpu")
    result = train_projections(
        train_pairs,
        all_q[~is_val],
        item_emb,
        val_pairs=val_pairs,
        val_query_embeddings=all_q[is_val],
        config=TrainConfig(
            output_dim=256,
            epochs=40,
            batch_size=128,
            seed=1337,
            use_bottleneck=True,
            n_axes=8,
            anchor_weight=20.0,
            patience=8,
        ),
        anchors=anchors,
        verbose=False,
    )
    print(f"bottleneck trained; best val recall@1 {max(result.val_recall):.4f}")

    # --- locate segments and give each one a mood vector ---------------------
    episodes = corpus.titles.set_index("anime_id")["episodes"].to_dict()
    texts: list[str] = []
    owners: list[int] = []
    for aid, body in zip(
        corpus.reviews["anime_id"].to_numpy(), corpus.reviews["text"].to_numpy(), strict=True
    ):
        pos = corpus.item_index.get(int(aid))
        if pos is None:
            continue
        for seg in segment_review(str(body), min_chars=60):
            if not is_boilerplate(seg):
                texts.append(seg)
                owners.append(pos)

    print(f"encoding {len(texts):,} segments…")
    seg_emb = encode_texts(texts, model_name=ENCODER, batch_size=256)
    tower = result.item_tower
    tower.eval()
    with torch.no_grad():
        seg_axes = tower.axes(torch.as_tensor(seg_emb)).cpu().numpy().astype(np.float64)

    np.savez_compressed(
        OUT / "segment_axes.npz",
        axes=seg_axes,
        owners=np.asarray(owners),
        texts=np.asarray(texts, dtype=object),
    )

    by_title: dict[int, list[tuple[str, np.ndarray]]] = {}
    for text, pos, vec in zip(texts, owners, seg_axes, strict=True):
        by_title.setdefault(pos, []).append((text, vec))

    eps_by_pos = {
        pos: (int(episodes[raw]) if raw in episodes and not pd.isna(episodes[raw]) else None)
        for raw, pos in corpus.item_index.items()
    }

    trajectories = build_trajectories(
        by_title,
        episodes_by_title=eps_by_pos,
        n_arcs=N_ARCS,
        axis_names=AXIS_NAMES[:8],
    )
    stats = curve_statistics(trajectories)
    print("\ncurve statistics:")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")

    # --- control: is the curve movement real, or noise in buckets? -----------
    # Shuffling each title's mood vectors across its own segments destroys any
    # true relationship between position and mood while preserving the marginal
    # distribution, the bucket sizes and the sample size. If the shuffled curves
    # move as much as the real ones, the "trajectory" is an artefact of
    # averaging noise, not a signal.
    rng = np.random.default_rng(1337)
    shuffled: dict[int, list[tuple[str, np.ndarray]]] = {}
    for pos, items in by_title.items():
        vectors = [v for _, v in items]
        order = rng.permutation(len(vectors))
        shuffled[pos] = [(items[i][0], vectors[order[i]]) for i in range(len(items))]

    control = build_trajectories(
        shuffled, episodes_by_title=eps_by_pos, n_arcs=N_ARCS, axis_names=AXIS_NAMES[:8]
    )
    control_stats = curve_statistics(control)
    print("\ncontrol (mood shuffled within each title):")
    for key in ("within_title_std", "mean_abs_arc_step", "within_over_between"):
        real, fake = stats[key], control_stats[key]
        ratio = real / fake if fake else float("nan")
        print(f"  {key:<24} real={real:<10} shuffled={fake:<10} ratio={ratio:.3f}")

    # --- the test: does shape add anything over the mean? --------------------
    tag_map = corpus.item_tag_map()
    labels = {pos: set(tag_map.get(int(pos), [])) for pos in trajectories.item_pos.tolist()}
    counts: dict[str, int] = {}
    for tags in labels.values():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    usable = sorted(
        t for t, n in counts.items() if MIN_TAG_SUPPORT <= n <= len(labels) - MIN_TAG_SUPPORT
    )
    print(
        f"\n{len(usable)} tags with support in [{MIN_TAG_SUPPORT}, {len(labels) - MIN_TAG_SUPPORT}]"
    )

    mean_feats = trajectories.mean_curve()
    shape_feats = trajectories.shape().reshape(len(trajectories), -1)
    full_feats = np.hstack([mean_feats, shape_feats])

    rows = []
    for tag in usable:
        y = np.array([1 if tag in labels[p] else 0 for p in trajectories.item_pos.tolist()])
        if y.sum() < MIN_TAG_SUPPORT:
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
        scores = {}
        for name, X in (("mean", mean_feats), ("shape", shape_feats), ("full", full_feats)):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            scores[name] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))
        rows.append({"tag": tag, "n": int(y.sum()), **{k: round(v, 4) for k, v in scores.items()}})

    rows.sort(key=lambda r: r["full"] - r["mean"], reverse=True)
    print(f"\n{'tag':<26}{'n':>5}{'mean':>8}{'shape':>8}{'full':>8}{'gain':>8}")
    for r in rows:
        print(
            f"  {r['tag']:<24}{r['n']:>5}{r['mean']:>8.3f}{r['shape']:>8.3f}"
            f"{r['full']:>8.3f}{r['full'] - r['mean']:>+8.3f}"
        )

    summary = {
        "control_curve_statistics": control_stats,
        "n_tags": len(rows),
        "mean_auc_mean_only": round(float(np.mean([r["mean"] for r in rows])), 4),
        "mean_auc_shape_only": round(float(np.mean([r["shape"] for r in rows])), 4),
        "mean_auc_full": round(float(np.mean([r["full"] for r in rows])), 4),
        "tags_where_full_beats_mean": sum(1 for r in rows if r["full"] > r["mean"]),
    }
    print("\nsummary:", summary)

    (OUT / "trajectory.json").write_text(
        json.dumps({"curve_statistics": stats, "summary": summary, "per_tag": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    np.save(OUT / "curves.npy", trajectories.curves)
    np.save(OUT / "curve_item_pos.npy", trajectories.item_pos)
    print(f"\nwrote {OUT}/trajectory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
