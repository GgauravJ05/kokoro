# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Declarative registry of the static corpora Kokoro ingests.

The live APIs this project was designed around are, as of 2026-09-09, both
unavailable: AniList returns ``403 The AniList API has been temporarily disabled
due to severe stability issues``, and Jikan returns ``504`` because it cannot
reach MyAnimeList. Rather than block on someone else's outage, ingestion is
source-agnostic and the default route is a set of static Hugging Face dumps —
which has the side benefit of being exactly reproducible for anyone who clones
this repo, with no account, no API key and no rate limit.

Each source declares its licence and its **known biases**. That second field is
not documentation politeness: the review corpus is scraped from MyAnimeList's
top 500 series only and drops ``mixed`` verdicts, which is a selection bias that
shapes what the mood axes can possibly learn. It belongs in the manifest next to
the data, not in a footnote discovered at evaluation time.

Nothing here is redistributed. Files are downloaded to ``data/raw/`` at
ingestion time, and ``data/`` is gitignored in its entirety.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

__all__ = ["SOURCES", "Source", "download", "resolve_url"]

Role = Literal["catalog", "reviews", "ratings"]


@dataclass(frozen=True, slots=True)
class Source:
    """One downloadable file from a Hugging Face dataset repository.

    Attributes:
        key: Short name used on the command line.
        repo: Hugging Face dataset repository id.
        filename: Path of the file within that repository.
        role: What the file contributes to the corpus.
        licence: Licence as declared on the dataset card. ``"other"`` and
            ``"unknown"`` are recorded verbatim rather than guessed at.
        rows: Approximate row count, for progress reporting and sanity checks.
        notes: What the file is and why it was chosen.
        biases: Known sampling or coverage biases. Copied into the corpus
            manifest so that every downstream result carries them.
        redistributable: Whether the file may be re-published. Everything here
            is ``False``; ingestion downloads, it never vendors.
    """

    key: str
    repo: str
    filename: str
    role: Role
    licence: str
    rows: int
    notes: str
    biases: tuple[str, ...] = ()
    redistributable: bool = False

    @property
    def url(self) -> str:
        """Direct download URL for the file."""
        return resolve_url(self.repo, self.filename)

    @property
    def card_url(self) -> str:
        """Human-readable dataset card, for licence review."""
        return f"https://huggingface.co/datasets/{self.repo}"


def resolve_url(repo: str, filename: str) -> str:
    """Build the Hugging Face ``resolve`` URL for a file in a dataset repo."""
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


#: The corpus, as validated on 2026-09-09. Coverage figures in ``notes`` were
#: measured, not assumed — see ``tests/test_sources.py`` for the assertions that
#: keep them honest.
SOURCES: dict[str, Source] = {
    "catalog": Source(
        key="catalog",
        repo="lyfesan/myanimelist-top-anime-dataset",
        filename="mal_top_anime_dataset.csv",
        role="catalog",
        licence="mit",
        rows=28_880,
        notes=(
            "Title metadata including synopsis, genres, themes, studios and air dates. "
            "Carries no id column, but every row's URL embeds the MyAnimeList id "
            "(/anime/<id>/...), which is what joins it to the review and rating sets. "
            "Recovering that id lifts review coverage from 46.3% to 99.8%."
        ),
        biases=(
            "Ranked subset of MyAnimeList, not the full catalog — the unranked "
            "long tail is absent, so catalog-coverage metrics are computed over "
            "a catalog that is already filtered toward the popular.",
        ),
    ),
    "reviews": Source(
        key="reviews",
        repo="Raiser1/anime-review-mal",
        filename="anime_sentiment_limpio.csv",
        role="reviews",
        licence="other",
        rows=52_723,
        notes=(
            "Long-form review text with a MyAnimeList anime_id on every row "
            "(median 1,488 characters). This is the contrastive training signal. "
            "Column names are Spanish: texto_resena is the review body."
        ),
        biases=(
            "Scraped from MyAnimeList's top 500 series only: 490 distinct titles "
            "against a 28,880-title catalog. The model is supervised on 1.7% of "
            "the catalog and must generalise to the rest through the content "
            "tower — which makes cold-start the headline evaluation, not a footnote.",
            "91.3% of reviews are positive. Mood axes learned here will be better "
            "resolved on the positive pole; report per-axis coverage before "
            "claiming any axis is calibrated.",
            "'mixed' verdicts were dropped by the dataset author. That is exactly "
            "the ambivalent middle a mood model most needs, and its absence caps "
            "what the catharsis and hope axes can learn.",
            "Text is lowercased and cleaned, which destroys the capitalisation "
            "that features.arcs uses to detect named arcs (the 'Marley arc' "
            "pattern). Expect degraded named-arc recall on this source.",
        ),
    ),
    "ratings": Source(
        key="ratings",
        repo="jason1966/CooperUnion_anime-recommendations-database",
        filename="rating.csv",
        role="ratings",
        licence="unknown",
        rows=7_813_737,
        notes=(
            "User-item rating matrix, ~73k users. Supplies the collaborative "
            "signal for the item-kNN and BPR baselines, which until now have only "
            "ever run on synthetic data. A rating of -1 means 'watched, unrated'."
        ),
        biases=(
            "2017 snapshot: contains no title released after it, so a temporal "
            "split on this matrix cannot test on recent seasons.",
            "Licence is undeclared on the dataset card. Ingest only; do not "
            "redistribute, and re-check before the repository goes public.",
        ),
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in chunks.

    Args:
        path: File to hash.
        chunk: Read size in bytes.

    Returns:
        Lowercase hex digest. This is what pins a corpus version: a manifest
        records it, so a training run can prove which bytes it saw.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(slots=True)
class DownloadResult:
    """Outcome of a single download."""

    source: Source
    path: Path
    sha256: str
    bytes: int
    cached: bool
    extra: dict[str, str] = field(default_factory=dict)


def download(
    source: Source,
    dest_dir: Path,
    *,
    force: bool = False,
    timeout: float = 120.0,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> DownloadResult:
    """Fetch one source file, skipping the download if it is already present.

    Args:
        source: The source to fetch.
        dest_dir: Directory to write into; created if absent.
        force: Re-download even when the file already exists.
        timeout: Per-request timeout in seconds.
        on_progress: Called with ``(bytes_so_far, total_or_None)`` as the body
            streams.

    Returns:
        A :class:`DownloadResult` carrying the path and content hash.

    Raises:
        httpx.HTTPStatusError: If the server returns an error status.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{source.key}__{Path(source.filename).name}"

    if target.exists() and not force:
        return DownloadResult(
            source=source,
            path=target,
            sha256=sha256_file(target),
            bytes=target.stat().st_size,
            cached=True,
        )

    partial = target.with_suffix(target.suffix + ".part")
    with httpx.stream("GET", source.url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers["content-length"]) if "content-length" in resp.headers else None
        seen = 0
        with partial.open("wb") as fh:
            for block in resp.iter_bytes(1 << 20):
                fh.write(block)
                seen += len(block)
                if on_progress:
                    on_progress(seen, total)

    # Rename only after a complete body, so an interrupted download never leaves
    # a truncated file that the cache check would happily accept next run.
    partial.replace(target)
    return DownloadResult(
        source=source,
        path=target,
        sha256=sha256_file(target),
        bytes=target.stat().st_size,
        cached=False,
    )
