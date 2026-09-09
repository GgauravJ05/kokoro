#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Sweep the anchor alignment weight and report what it buys and costs.

The first bottleneck run validated 0 of 6 axes at a mean AUC of 0.465. Axis
values sat at a standard deviation of 0.08 against a possible range of +/-1,
so the anchor hinge (which asks for 0.8) was never satisfied and contributed
about 4% of the total loss. This sweep tests the obvious hypothesis: the anchor
term was simply too weak.

Encodings are computed once and reused across configurations, because encoding
is the slow part and the comparison must hold the inputs fixed anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kokoro.data.corpus import load_corpus
from kokoro.data.pairs import build_pairs
from kokoro.eval.axes import evaluate_axes, probe_tags, summarise
from kokoro.models.content import encode_texts
from kokoro.models.mood_axes import AXIS_NAMES
from kokoro.train.contrastive import TrainConfig, build_anchor_batch, train_projections

WEIGHTS = [0.0, 1.0, 5.0, 20.0]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("artifacts/anchor_sweep")


def main() -> int:
    """Run the sweep and write a JSON report."""
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus("data/processed", seed=1337)
    pairs = build_pairs(corpus, max_per_title=150, seed=1337)
    train_pairs, val_pairs = pairs.split(val_frac=0.1, seed=1337)

    held = probe_tags()
    print(f"holding out {len(held)} probe tags; {len(pairs):,} pairs over {pairs.n_titles} titles")

    item_emb = encode_texts(corpus.item_texts(exclude_tags=held), model_name=ENCODER)
    all_q = encode_texts(pairs.queries, model_name=ENCODER)

    val_titles = set(val_pairs.item_pos.tolist())
    is_val = np.array([int(p) in val_titles for p in pairs.item_pos])
    train_q, val_q = all_q[~is_val], all_q[is_val]

    anchors = build_anchor_batch(
        lambda phrases: encode_texts(phrases, model_name=ENCODER), 8, "cpu"
    )
    tag_map = corpus.item_tag_map()
    names = list(AXIS_NAMES[:8])

    rows = []
    for weight in WEIGHTS:
        cfg = TrainConfig(
            output_dim=256,
            epochs=40,
            batch_size=128,
            seed=1337,
            use_bottleneck=True,
            n_axes=8,
            anchor_weight=weight,
            patience=8,
        )
        result = train_projections(
            train_pairs,
            train_q,
            item_emb,
            val_pairs=val_pairs,
            val_query_embeddings=val_q,
            config=cfg,
            anchors=anchors if weight > 0 else None,
            verbose=False,
        )
        values = result.axis_values(item_emb)
        reports = evaluate_axes(values, names, tag_map)
        summary = summarise(reports)

        row = {
            "anchor_weight": weight,
            "val_recall@1": round(max(result.val_recall), 4),
            "axis_std": round(float(values.std()), 4),
            "axis_absmax": round(float(np.abs(values).max()), 4),
            "mean_auc": round(summary["mean_auc"], 4),
            "n_validated": summary["n_validated"],
            "n_inverted": summary["n_inverted"],
            "per_axis": {r.axis: (None if r.auc != r.auc else round(r.auc, 3)) for r in reports},
        }
        rows.append(row)
        print(
            f"weight={weight:>5}  val_recall@1={row['val_recall@1']:.4f}  "
            f"axis_std={row['axis_std']:.3f}  |max|={row['axis_absmax']:.3f}  "
            f"mean_auc={row['mean_auc']:.3f}  validated={row['n_validated']}  "
            f"inverted={row['n_inverted']}"
        )
        np.save(OUT / f"axis_values_w{weight}.npy", values)

    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}/sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
