# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Command line entry point: ``kokoro <command>``."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from kokoro import __version__, provenance

app = typer.Typer(
    name="kokoro",
    help="Mood-conditioned anime & manga retrieval.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"kokoro {__version__}")


@app.command("provenance")
def provenance_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit the record as JSON."),
) -> None:
    """Show the authorship and provenance record for this build."""
    record = provenance.current()
    if json_out:
        console.print_json(json.dumps(record.to_dict()))
        return
    console.print(f"[bold]{record.banner()}[/bold]")


@app.command()
def config() -> None:
    """Show the resolved runtime configuration."""
    from kokoro.config import get_settings

    settings = get_settings()
    table = Table("setting", "value", title="kokoro configuration")
    for key, value in settings.model_dump().items():
        table.add_row(key, str(value))
    table.add_row("resolved_device", settings.resolve_device())
    console.print(table)


@app.command()
def ingest(
    media_type: str = typer.Option("ANIME", help="ANIME or MANGA."),
    max_pages: int = typer.Option(5, min=1, help="Pages of 50 titles to fetch."),
    out: str = typer.Option("data/raw/titles.jsonl", help="Output JSONL path."),
) -> None:
    """Fetch titles from AniList into a JSONL file.

    Rate-limited to the value in ``KOKORO_ANILIST_RPS``; do not raise it.
    """
    import asyncio
    from pathlib import Path

    from kokoro.data.anilist import AniListClient

    async def _run() -> int:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as fh:
            async with AniListClient() as client:
                async for title in client.iter_titles(media_type, max_pages=max_pages):
                    fh.write(title.model_dump_json() + "\n")
                    count += 1
                    if count % 50 == 0:
                        console.print(f"  fetched {count} titles…")
        return count

    total = asyncio.run(_run())
    console.print(f"[green]wrote {total} titles to {out}[/green]")


@app.command()
def benchmark(
    results_out: str = typer.Option("artifacts/benchmark.json", help="Report path."),
    k: int = typer.Option(10, min=1, help="Metric cut-off."),
    seed: int = typer.Option(1337, help="RNG seed."),
) -> None:
    """Run the baseline suite on synthetic data as a harness smoke test.

    Replace the synthetic log with the real corpus once ingestion has run; the
    harness code is identical either way.
    """
    import numpy as np

    from kokoro.eval.harness import run_benchmark, save_results, to_markdown_table
    from kokoro.eval.splits import Interactions, temporal_split
    from kokoro.models.baselines import ItemKNNRecommender, PopularityRecommender, RandomRecommender
    from kokoro.models.mf import BPRMatrixFactorization

    rng = np.random.default_rng(seed)
    n = 4000
    data = Interactions(
        user=rng.integers(0, 200, n).astype(np.int64),
        item=rng.integers(0, 300, n).astype(np.int64),
        rating=rng.integers(1, 11, n).astype(np.float32),
        timestamp=np.sort(rng.integers(1_500_000_000, 1_700_000_000, n)).astype(np.int64),
    )
    split = temporal_split(data, test_frac=0.2)
    results = run_benchmark(
        [
            RandomRecommender(seed=seed),
            PopularityRecommender(),
            ItemKNNRecommender(),
            BPRMatrixFactorization(n_epochs=5, seed=seed),
        ],
        split,
        k=k,
    )
    console.print(to_markdown_table(results))
    console.print(f"[green]report written to {save_results(results, results_out)}[/green]")


@app.command()
def watermark(
    path: str = typer.Argument(..., help="Path to a .npy embedding matrix."),
    salt: str = typer.Option("", help="Salt used when the watermark was applied."),
) -> None:
    """Test whether an embedding matrix carries this project's watermark."""
    import numpy as np

    vectors = np.load(path)
    mean, z = provenance.detect_embedding_watermark(vectors, salt=salt)
    verdict = "DETECTED" if z > 5.0 else "not detected"
    colour = "green" if z > 5.0 else "yellow"
    console.print(f"mean projection: {mean:.6f}   z-score: {z:.2f}")
    console.print(f"[{colour}]watermark {verdict}[/{colour}] (salt={salt!r})")


if __name__ == "__main__":  # pragma: no cover
    app()
