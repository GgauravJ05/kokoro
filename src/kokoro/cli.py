# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Command line entry point: ``kokoro <command>``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from kokoro import __version__, provenance

if TYPE_CHECKING:  # pragma: no cover
    from kokoro.data.corpus import Corpus
    from kokoro.models.base import Retriever

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
    corpus: str | None = typer.Option(
        None, help="Path to a built corpus. Omit to run on synthetic data."
    ),
    max_users: int = typer.Option(
        3000, min=10, help="User subsample size when running on a real corpus."
    ),
    epochs: int = typer.Option(5, min=1, help="BPR training epochs."),
    strategy: str = typer.Option(
        "user_holdout", "--split", help="Split strategy: user_holdout or cold_start."
    ),
    cold_cut_year: int = typer.Option(2014, help="Debut year cut for the cold_start split."),
    cold_only: bool = typer.Option(
        False, "--cold-only", help="Rank against cold candidates only (standard protocol)."
    ),
    content: bool = typer.Option(
        False, "--content", help="Add the off-the-shelf content retriever (downloads an encoder)."
    ),
    encoder: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", help="Sentence encoder for --content."
    ),
    trained: str | None = typer.Option(
        None, help="Path to trained item embeddings (.npy) from `kokoro train`."
    ),
) -> None:
    """Run the baseline suite, on a real corpus when one is given.

    Without ``--corpus`` this is a harness smoke test on synthetic data, where
    a random ranker legitimately wins because the synthetic log has no
    structure to learn.
    """
    if corpus is not None:
        # Keywords, not positions: a silently-dropped positional argument here
        # made --content a no-op that reported nothing.
        _benchmark_corpus(
            corpus_path=corpus,
            results_out=results_out,
            k=k,
            seed=seed,
            max_users=max_users,
            epochs=epochs,
            strategy=strategy,
            cold_cut_year=cold_cut_year,
            cold_only=cold_only,
            content=content,
            encoder=encoder,
            trained=trained,
        )
        return
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


def _add_content_models(
    models: list[Retriever],
    corpus: Corpus,
    *,
    content: bool,
    encoder: str,
    trained: str | None,
) -> None:
    """Append the content retrievers the flags asked for."""
    import numpy as np

    from kokoro.models.content import ContentRetriever, encode_texts

    if content:
        texts = corpus.item_texts()
        console.print(f"encoding {len(texts):,} item texts with {encoder}…")
        console.print(f"  example: [dim]{texts[0][:150]}[/dim]")
        models.append(
            ContentRetriever(
                encode_texts(texts, model_name=encoder, show_progress=True),
                name="content-offshelf",
            )
        )

    if trained is not None:
        vectors = np.load(trained)
        console.print(f"loaded trained item embeddings {vectors.shape} from {trained}")
        models.append(ContentRetriever(vectors.astype(np.float32), name="content-trained"))


def _benchmark_corpus(
    corpus_path: str,
    results_out: str,
    k: int,
    seed: int,
    max_users: int,
    epochs: int,
    strategy: str = "user_holdout",
    cold_cut_year: int = 2014,
    cold_only: bool = False,
    content: bool = False,
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    trained: str | None = None,
) -> None:
    """Run the baselines against a built corpus."""
    from datetime import datetime, timezone

    import numpy as np

    from kokoro.data.corpus import load_corpus
    from kokoro.eval.harness import run_benchmark, save_results, to_markdown_table
    from kokoro.eval.splits import cold_start_split, user_holdout_split
    from kokoro.models.baselines import (
        ItemKNNRecommender,
        PopularityRecommender,
        RandomRecommender,
    )
    from kokoro.models.mf import BPRMatrixFactorization

    with console.status("loading corpus…"):
        c = load_corpus(corpus_path, max_users=max_users, seed=seed)

    data = c.interactions
    console.print(
        f"corpus [bold]{c.version}[/bold]  "
        f"{len(data):,} interactions  {data.n_users:,} users  {data.n_items:,} items"
    )
    candidates = None
    if strategy == "cold_start":
        cut = int(datetime(cold_cut_year, 1, 1, tzinfo=timezone.utc).timestamp())
        split = cold_start_split(data, c.item_debut, cut=cut)
        n_cold = split.meta["n_cold_items"]
        cold_items = np.flatnonzero(c.item_debut[: data.n_items] >= cut).astype(np.int64)
        answerable = len(set(split.test.item.tolist()))
        if cold_only:
            candidates = cold_items
            console.print(
                f"[yellow]protocol: cold-only — ranking restricted to the "
                f"{len(cold_items):,} cold titles, of which {answerable:,} carry "
                f"held-out ratings and can actually be correct.[/yellow]"
            )
        console.print(
            f"[yellow]split: cold_start at {cold_cut_year} — {n_cold:,} titles have "
            f"zero training interactions. Collaborative models cannot represent them "
            f"at all; that is the point of the split.[/yellow]"
        )
    elif strategy == "user_holdout":
        split = user_holdout_split(data, holdout_frac=0.2, seed=seed)
        console.print(
            "[yellow]split: user_holdout — this source carries no interaction "
            "timestamps, so a temporal split is not computable on it.[/yellow]"
        )
    else:
        console.print(f"[red]unknown split {strategy!r}: use user_holdout or cold_start[/red]")
        raise typer.Exit(1)

    models: list[Retriever] = [
        RandomRecommender(seed=seed),
        PopularityRecommender(),
        ItemKNNRecommender(k_neighbors=50),
        BPRMatrixFactorization(n_factors=64, n_epochs=epochs, lr=0.05, seed=seed),
    ]
    _add_content_models(models, c, content=content, encoder=encoder, trained=trained)

    with console.status("fitting and scoring…"):
        results = run_benchmark(
            models,
            split,
            k=k,
            # candidates -> what is reported; restrict_to -> what may be ranked.
            candidates=cold_items if strategy == "cold_start" else None,
            restrict_to=candidates,
        )

    for r in results:
        r["corpus_version"] = c.version
        r["max_users_subsample"] = max_users
        r["n_interactions"] = len(data)
        r["split_strategy"] = strategy
        r["protocol"] = "cold_only" if cold_only else "full_catalog"

    console.print()
    console.print(to_markdown_table(results))
    console.print(f"\n[green]report written to {save_results(results, results_out)}[/green]")


@app.command()
def train(
    corpus_path: str = typer.Option("data/processed", "--corpus", help="Built corpus."),
    out: str = typer.Option("artifacts/two_tower", help="Directory for the trained artifacts."),
    encoder: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", help="Frozen sentence encoder."
    ),
    max_per_title: int = typer.Option(150, min=1, help="Cap on review segments per title."),
    epochs: int = typer.Option(30, min=1, help="Training epochs."),
    batch_size: int = typer.Option(128, min=2, help="Distinct titles per batch."),
    output_dim: int = typer.Option(256, min=8, help="Shared retrieval space width."),
    temperature: float = typer.Option(0.05, help="InfoNCE temperature."),
    seed: int = typer.Option(1337, help="RNG seed."),
) -> None:
    """Train the two-tower projections on review text and evaluate cold start."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    import numpy as np

    from kokoro import provenance
    from kokoro.data.corpus import load_corpus
    from kokoro.data.pairs import build_pairs
    from kokoro.models.content import encode_texts
    from kokoro.train.contrastive import TrainConfig, train_projections

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with console.status("loading corpus…"):
        # No user subsample here: training reads reviews and metadata, never
        # interactions, and the item space is subsample-independent anyway.
        c = load_corpus(corpus_path, seed=seed)
    console.print(f"corpus [bold]{c.version}[/bold]  {len(c.item_index):,} items")

    with console.status("building training pairs…"):
        pairs = build_pairs(c, max_per_title=max_per_title, seed=seed)
    console.print(
        f"pairs: {len(pairs):,} segments over {pairs.n_titles} titles "
        f"(capped at {max_per_title}/title)"
    )
    console.print(f"  example query: [dim]{pairs.queries[0][:140]}[/dim]")

    train_pairs, val_pairs = pairs.split(val_frac=0.1, seed=seed)
    console.print(
        f"  split by title: {len(train_pairs):,} train / {len(val_pairs):,} val "
        f"({val_pairs.n_titles} held-out titles)"
    )

    item_texts = c.item_texts()
    console.print(f"encoding {len(item_texts):,} item texts…")
    item_emb = encode_texts(item_texts, model_name=encoder, show_progress=True)

    console.print(f"encoding {len(pairs):,} review segments…")
    all_q = encode_texts(pairs.queries, model_name=encoder, show_progress=True)

    # Re-encode per split rather than slicing, so alignment cannot drift.
    train_idx = np.array(
        [i for i, p in enumerate(pairs.item_pos) if p in set(train_pairs.item_pos.tolist())]
    )
    val_idx = np.array(
        [i for i, p in enumerate(pairs.item_pos) if p in set(val_pairs.item_pos.tolist())]
    )
    train_q, val_q = all_q[train_idx], all_q[val_idx]

    cfg = TrainConfig(
        output_dim=output_dim,
        epochs=epochs,
        batch_size=batch_size,
        temperature=temperature,
        seed=seed,
    )
    console.print("\n[bold]training[/bold]")
    result = train_projections(
        train_pairs,
        train_q,
        item_emb,
        val_pairs=val_pairs,
        val_query_embeddings=val_q,
        config=cfg,
    )

    projected = result.project_items(item_emb)
    np.save(out_dir / "item_embeddings_trained.npy", projected)
    np.save(out_dir / "item_embeddings_offshelf.npy", item_emb)

    report = provenance.stamp(
        {
            "corpus_version": c.version,
            "encoder": encoder,
            "config": asdict(cfg),
            "n_pairs": len(pairs),
            "n_titles": pairs.n_titles,
            "best_epoch": result.best_epoch,
            "train_loss": result.train_loss,
            "val_recall_at_1": result.val_recall,
        }
    )
    (out_dir / "training.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    console.print(f"\n[green]artifacts written to {out_dir}/[/green]")
    console.print(
        f"best epoch {result.best_epoch}  "
        f"val recall@1 {result.val_recall[result.best_epoch - 1]:.4f}"
        if result.val_recall
        else ""
    )
    console.print(
        "\nEvaluate with:\n"
        f"  kokoro benchmark --corpus {corpus_path} --split cold_start "
        f"--content --trained {out_dir}/item_embeddings_trained.npy"
    )


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


corpus_app = typer.Typer(
    name="corpus", help="Build and inspect the frozen training corpus.", no_args_is_help=True
)
app.add_typer(corpus_app)


@corpus_app.command("sources")
def corpus_sources() -> None:
    """List the configured data sources with their licences and known biases."""
    from kokoro.data.sources import SOURCES

    for src in SOURCES.values():
        console.print(f"\n[bold]{src.key}[/bold]  ({src.role})  —  {src.repo}")
        console.print(f"  licence : {src.licence}   rows ~{src.rows:,}")
        console.print(f"  card    : {src.card_url}")
        console.print(f"  {src.notes}")
        for b in src.biases:
            console.print(f"  [yellow]bias[/yellow]  {b}")


@corpus_app.command("build")
def corpus_build(
    out: str = typer.Option("data/processed", help="Directory for the frozen corpus."),
    raw: str = typer.Option("data/raw", help="Directory for downloaded source files."),
    min_reviews: int = typer.Option(5, min=1, help="Minimum reviews for a title to be supervised."),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached."),
) -> None:
    """Download the sources and build the frozen, versioned corpus."""
    from pathlib import Path

    from rich.progress import BarColumn, DownloadColumn, Progress, TaskID, TextColumn

    from kokoro.data.build import build_corpus
    from kokoro.data.sources import SOURCES, download

    raw_dir, out_dir = Path(raw), Path(out)
    downloads = {}

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        console=console,
    ) as progress:
        for key, src in SOURCES.items():
            task = progress.add_task(key, total=None)

            def _tick(seen: int, total: int | None, _t: TaskID = task) -> None:
                progress.update(_t, completed=seen, total=total)

            result = download(src, raw_dir, force=force, on_progress=_tick)
            progress.update(task, description=f"{key} {'(cached)' if result.cached else ''}")
            downloads[key] = result

    manifest = build_corpus(downloads, out_dir, min_reviews_per_title=min_reviews)
    stats = manifest["statistics"]

    table = Table("statistic", "value", title=f"corpus {manifest.version}")
    for k, v in stats.items():
        table.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    console.print(table)
    console.print(f"[green]corpus written to {out_dir}/[/green]")

    console.print("\n[yellow]Known biases recorded in the manifest:[/yellow]")
    for key, src in SOURCES.items():
        for b in src.biases:
            console.print(f"  [dim]{key}[/dim]  {b}")


@corpus_app.command("info")
def corpus_info(
    path: str = typer.Option("data/processed", help="Directory holding the built corpus."),
) -> None:
    """Show the manifest of an already-built corpus."""
    import json
    from pathlib import Path

    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]no manifest at {manifest_path} — run `kokoro corpus build`[/red]")
        raise typer.Exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console.print(f"[bold]corpus {manifest['corpus_version']}[/bold]  built {manifest['built_at']}")

    table = Table("statistic", "value")
    for k, v in manifest["statistics"].items():
        table.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    console.print(table)

    src_table = Table("source", "licence", "sha256", title="inputs")
    for key, src in manifest["sources"].items():
        src_table.add_row(key, src["licence"], src["sha256"][:16] + "…")
    console.print(src_table)


if __name__ == "__main__":  # pragma: no cover
    app()
