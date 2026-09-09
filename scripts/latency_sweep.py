#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Measure the recall/latency trade-off of the serving path.

Two things are reported, and the second is the one that matters for a claim.

**ANN recall.** HNSW is approximate, so its top-k can differ from brute force.
Recall is measured against :class:`~kokoro.index.ann.ExactIndex` on the same
vectors: a fast index that returns the wrong neighbours is not a speedup.

**Where the time actually goes.** The encoder forward pass and the index lookup
are timed separately, because they scale differently and only one of them is
worth optimising at this catalog size. Reporting a single end-to-end number
would hide that.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from kokoro.index.ann import ExactIndex, HNSWIndex
from kokoro.serve.engine import Engine

QUERIES = [
    "something melancholy but hopeful, slow, that won't wreck me",
    "a funny lighthearted show to relax with",
    "dark psychological thriller that messes with your head",
    "comfort watch for a bad day",
    "epic space opera with political intrigue",
    "quiet slice of life about growing up",
    "devastating drama that earns its ending",
    "weird surreal comedy that makes no sense",
    "tense survival horror with real stakes",
    "warm found family story",
]
EF_VALUES = [16, 32, 64, 128, 256]
OUT = Path("artifacts/serving")


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile in milliseconds."""
    return float(np.percentile(values, p))


def main() -> int:
    """Run the sweep and write a report."""
    OUT.mkdir(parents=True, exist_ok=True)
    engine = Engine("artifacts/two_tower", "data/processed", index="exact")
    print(f"catalog: {engine.vectors.shape[0]:,} items x {engine.vectors.shape[1]} dims")

    # Warm the encoder so its one-off load does not land in the timings.
    engine.embed_query("warmup")
    embedded = np.vstack([engine.embed_query(q) for q in QUERIES])

    exact = ExactIndex().build(engine.vectors)
    gold, _ = exact.search(embedded, k=10)

    encode_ms: list[float] = []
    for q in QUERIES * 3:
        t0 = time.perf_counter()
        engine.embed_query(q)
        encode_ms.append((time.perf_counter() - t0) * 1000)

    rows = []
    for ef in EF_VALUES:
        index = HNSWIndex(ef_search=ef).build(engine.vectors)
        approx, _ = index.search(embedded, k=10)
        recall = float(
            np.mean(
                [
                    len(set(a.tolist()) & set(b.tolist())) / 10
                    for a, b in zip(gold, approx, strict=True)
                ]
            )
        )

        search_ms: list[float] = []
        for _ in range(20):
            for vec in embedded:
                t0 = time.perf_counter()
                index.search(vec[None, :], k=10)
                search_ms.append((time.perf_counter() - t0) * 1000)

        row = {
            "ef_search": ef,
            "ann_recall@10": round(recall, 4),
            "search_p50_ms": round(statistics.median(search_ms), 4),
            "search_p95_ms": round(percentile(search_ms, 95), 4),
        }
        rows.append(row)
        print(
            f"  ef={ef:>4}  recall@10={recall:.4f}  "
            f"search p50={row['search_p50_ms']:.3f}ms  p95={row['search_p95_ms']:.3f}ms"
        )

    exact_ms: list[float] = []
    for _ in range(20):
        for vec in embedded:
            t0 = time.perf_counter()
            exact.search(vec[None, :], k=10)
            exact_ms.append((time.perf_counter() - t0) * 1000)

    summary = {
        "catalog_size": int(engine.vectors.shape[0]),
        "encode_p50_ms": round(statistics.median(encode_ms), 3),
        "encode_p95_ms": round(percentile(encode_ms, 95), 3),
        "exact_search_p50_ms": round(statistics.median(exact_ms), 4),
        "exact_search_p95_ms": round(percentile(exact_ms, 95), 4),
        "sweep": rows,
    }

    best = min(
        (r for r in rows if r["ann_recall@10"] >= 0.99),
        key=lambda r: r["search_p95_ms"],
        default=rows[-1],
    )
    end_to_end_p95 = summary["encode_p95_ms"] + best["search_p95_ms"]
    summary["recommended_ef"] = best["ef_search"]
    summary["end_to_end_p95_ms"] = round(end_to_end_p95, 3)

    print(f"\nencoder  p50={summary['encode_p50_ms']:.2f}ms  p95={summary['encode_p95_ms']:.2f}ms")
    print(
        f"exact    p50={summary['exact_search_p50_ms']:.3f}ms "
        f"p95={summary['exact_search_p95_ms']:.3f}ms"
    )
    print(
        f"\nrecommended ef_search={best['ef_search']} "
        f"(recall {best['ann_recall@10']:.4f}) -> end-to-end p95 {end_to_end_p95:.1f}ms"
    )

    (OUT / "latency.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}/latency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
