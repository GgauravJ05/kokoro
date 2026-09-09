# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""The single evaluation path every model in this project goes through.

Written before the models, on purpose. An evaluation harness authored after the
model it evaluates tends to grow whatever affordance makes that model look good;
one written first is a fixed target. Every number in the README comes from
:func:`run_benchmark`, every result carries a provenance stamp, and no model
gets to bring its own metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kokoro import provenance
from kokoro.eval.metrics import evaluate

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from kokoro.eval.splits import Split
    from kokoro.models.base import Retriever

__all__ = ["BenchmarkResult", "run_benchmark", "to_markdown_table"]


class BenchmarkResult(dict[str, Any]):
    """A model's metrics plus the context needed to reproduce them."""

    @property
    def model(self) -> str:
        """Reporting name of the evaluated model."""
        return str(self["model"])


def run_benchmark(
    models: list[Retriever],
    split: Split,
    *,
    k: int = 10,
    min_rating: float = 7.0,
    embeddings: npt.NDArray[np.float64] | None = None,
    baseline_name: str = "popularity",
) -> list[BenchmarkResult]:
    """Fit and score every model on one split.

    Args:
        models: Models to evaluate. Each is fitted on ``split.train`` only.
        split: The train/test partition.
        k: Cut-off for every metric.
        min_rating: Threshold above which a test interaction counts as a
            positive.
        embeddings: Optional item embeddings, enabling intra-list diversity.
        baseline_name: Which model's rankings serve as the "obvious"
            reference for serendipity. Absent, serendipity is skipped.

    Returns:
        One :class:`BenchmarkResult` per model, in input order.

    Raises:
        ValueError: If the test split has no positives above ``min_rating``.
    """
    truth = split.test.positives_by_user(min_rating=min_rating)
    if not truth:
        raise ValueError(
            f"test split contains no interactions rated >= {min_rating}; "
            "lower min_rating or check the split"
        )

    users = np.array(sorted(truth), dtype=np.int64)
    relevant = [truth[int(u)] for u in users]
    n_items = max(split.train.n_items, split.test.n_items)
    popularity = np.bincount(split.train.item, minlength=n_items).astype(np.float64)

    rankings: dict[str, npt.NDArray[np.int64]] = {}
    timings: dict[str, float] = {}

    for model in models:
        t0 = time.perf_counter()
        model.fit(split.train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        rankings[model.name] = model.recommend(users, k=k)
        query_s = (time.perf_counter() - t0) / max(users.size, 1)
        timings[model.name] = fit_s
        timings[f"{model.name}:query"] = query_s

    baseline = rankings.get(baseline_name)
    results: list[BenchmarkResult] = []
    for model in models:
        ranked = rankings[model.name]
        metrics = evaluate(
            ranked,
            relevant,
            n_items=n_items,
            k=k,
            embeddings=embeddings,
            popularity=popularity,
            baseline_ranked=baseline if model.name != baseline_name else None,
        )
        results.append(
            BenchmarkResult(
                model=model.name,
                split=split.strategy,
                split_meta=split.meta,
                k=k,
                min_rating=min_rating,
                n_users_evaluated=int(users.size),
                fit_seconds=round(timings[model.name], 3),
                query_ms=round(timings[f"{model.name}:query"] * 1000, 3),
                **{m: round(v, 5) for m, v in metrics.items()},
            )
        )
    return results


def to_markdown_table(results: list[BenchmarkResult], *, columns: list[str] | None = None) -> str:
    """Render results as the Markdown table that goes in the README.

    Args:
        results: Output of :func:`run_benchmark`.
        columns: Column order. Defaults to the headline set.

    Returns:
        A GitHub-flavoured Markdown table, or a placeholder when empty.
    """
    if not results:
        return "_no results_"
    k = results[0]["k"]
    columns = columns or [
        "model",
        f"recall@{k}",
        f"ndcg@{k}",
        f"mrr@{k}",
        f"coverage@{k}",
        f"gini@{k}",
        f"popularity_lift@{k}",
        "query_ms",
    ]
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = ["| " + " | ".join(str(r.get(c, "—")) for c in columns) + " |" for r in results]
    return "\n".join([header, rule, *rows])


def save_results(results: list[BenchmarkResult], path: str | Path) -> Path:
    """Write a stamped JSON report.

    Args:
        results: Output of :func:`run_benchmark`.
        path: Destination ``.json`` file; parent directories are created.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = provenance.stamp({"results": list(results)})
    target.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    return target
