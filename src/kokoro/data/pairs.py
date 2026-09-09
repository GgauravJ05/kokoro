# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Build the ``(review segment -> title)`` pairs the contrastive objective trains on.

Three decisions here are load-bearing, and getting any of them wrong produces a
model that scores well in training and fails at cold start.

**The item side is metadata, never reviews.** A cold title has no reviews by
definition. If the item tower learned to encode review text, it would have
nothing to encode for the titles the headline split is about. So reviews supply
the *query* side only; the item side is always the same metadata string a
brand-new title would have.

**Titles are masked out of the query.** Handled in
:func:`kokoro.features.text.query_text` — a review naming its own show lets the
model match on the title string rather than on mood.

**Batches contain each title at most once.** This corpus has 449 supervised
titles, so a batch of 256 drawn uniformly would repeat titles constantly, and
InfoNCE would then push apart two segments that describe the *same* work while
labelling one of them a negative. That is not a small amount of noise: it is a
gradient pointing the wrong way, on a large fraction of the batch.
:func:`iter_batches` samples distinct titles instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kokoro.features.arcs import segment_review
from kokoro.features.text import query_text

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    import numpy.typing as npt

    from kokoro.data.corpus import Corpus

__all__ = ["PairSet", "build_pairs", "iter_batches"]

#: Reviewer meta-commentary, which is fluent English about the *review* rather
#: than about the work. Left in, it trains the model to associate "thanks for
#: reading" with whichever titles happen to attract chatty reviewers — a real
#: signal, and an entirely useless one.
_BOILERPLATE = re.compile(
    r"""
    thanks?\s+for\s+reading | my\s+(first|second)\s+review | feel\s+free\s+to
  | leave\s+(a\s+)?(comment|feedback) | on\s+my\s+profile | sorry\s+for\s+my\s+english
  | (helpful|found\s+this)\s+review | click\s+here | read\s+more
  | ^\s*(story|art|sound|character|enjoyment|overall)\s*[:\-]?\s*\d
  | \d\s*/\s*10\s*$ | this\s+review\s+(contains|may\s+contain)
  | am\s+i\s+being\s+(overly|unfair) | in\s+conclusion\s*,?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_boilerplate(segment: str) -> bool:
    """Return whether a segment is reviewer meta-commentary rather than content.

    Args:
        segment: A candidate training segment.

    Returns:
        ``True`` when the segment should be dropped.
    """
    return bool(_BOILERPLATE.search(segment))


@dataclass(slots=True)
class PairSet:
    """Training pairs, already masked and aligned to item positions.

    Attributes:
        queries: Review segments, one per pair.
        item_pos: Contiguous item position of the title each query describes.
        anime_ids: Raw ``anime_id`` per pair, for grouping and diagnostics.
    """

    queries: list[str]
    item_pos: npt.NDArray[np.int64]
    anime_ids: npt.NDArray[np.int64]

    def __len__(self) -> int:
        """Number of pairs."""
        return len(self.queries)

    @property
    def n_titles(self) -> int:
        """Distinct titles represented."""
        return int(np.unique(self.item_pos).size)

    def split(self, *, val_frac: float = 0.05, seed: int = 1337) -> tuple[PairSet, PairSet]:
        """Hold out a validation slice, split by *title* rather than by pair.

        Splitting by pair would put segments of the same review on both sides,
        so validation loss would measure memorisation rather than transfer.

        Args:
            val_frac: Fraction of titles held out.
            seed: RNG seed.

        Returns:
            ``(train, validation)``.
        """
        rng = np.random.default_rng(seed)
        titles = np.unique(self.item_pos)
        n_val = max(1, round(len(titles) * val_frac))
        val_titles = set(rng.choice(titles, size=n_val, replace=False).tolist())

        mask = np.array([int(p) in val_titles for p in self.item_pos], dtype=bool)
        return self._take(~mask), self._take(mask)

    def _take(self, mask: npt.NDArray[np.bool_]) -> PairSet:
        """Return the subset selected by a boolean mask."""
        idx = np.flatnonzero(mask)
        return PairSet(
            queries=[self.queries[i] for i in idx],
            item_pos=self.item_pos[idx],
            anime_ids=self.anime_ids[idx],
        )


def build_pairs(
    corpus: Corpus,
    *,
    max_per_title: int = 150,
    min_chars: int = 60,
    max_chars: int = 400,
    seed: int = 1337,
) -> PairSet:
    """Segment reviews into masked training pairs.

    Args:
        corpus: A loaded corpus.
        max_per_title: Cap on segments kept per title. The review distribution
            is heavily skewed — some titles have 580 reviews and others five —
            and without a cap the loss is dominated by a handful of works.
        min_chars: Drop segments shorter than this; a six-word fragment carries
            no mood judgement.
        max_chars: Truncation limit passed to
            :func:`~kokoro.features.text.query_text`.
        seed: RNG seed for the per-title subsample.

    Returns:
        The assembled :class:`PairSet`.

    Raises:
        ValueError: If the corpus yields no usable pairs.
    """
    rng = np.random.default_rng(seed)
    variants = corpus.title_variants()

    by_title: dict[int, list[str]] = {}
    for anime_id, text in zip(
        corpus.reviews["anime_id"].to_numpy(), corpus.reviews["text"].to_numpy(), strict=True
    ):
        pos = corpus.item_index.get(int(anime_id))
        if pos is None:
            continue  # rated-catalog subsample does not cover this title
        titles = variants.get(int(anime_id), ())
        for raw_segment in segment_review(str(text), min_chars=min_chars):
            if is_boilerplate(raw_segment):
                continue
            cleaned = query_text(raw_segment, *titles, max_chars=max_chars)
            if len(cleaned) >= min_chars:
                by_title.setdefault(pos, []).append(cleaned)

    queries: list[str] = []
    item_pos: list[int] = []
    reverse = {pos: raw for raw, pos in corpus.item_index.items()}

    for pos, segments in by_title.items():
        keep = segments
        if len(segments) > max_per_title:
            picked = rng.choice(len(segments), size=max_per_title, replace=False)
            keep = [segments[i] for i in picked]
        queries.extend(keep)
        item_pos.extend([pos] * len(keep))

    if not queries:
        raise ValueError(
            "no training pairs were produced — check that the corpus reviews join "
            "to the loaded item index"
        )

    positions = np.asarray(item_pos, dtype=np.int64)
    return PairSet(
        queries=queries,
        item_pos=positions,
        anime_ids=np.array([reverse[int(p)] for p in positions], dtype=np.int64),
    )


def iter_batches(
    pairs: PairSet,
    *,
    batch_size: int = 128,
    seed: int = 1337,
    drop_last: bool = True,
) -> Iterator[npt.NDArray[np.int64]]:
    """Yield batches of pair indices in which every title appears at most once.

    With 449 titles and a batch of 128, uniform sampling would put a duplicate
    title in essentially every batch. InfoNCE treats every non-diagonal entry as
    a negative, so a duplicate becomes a *false* negative: the loss actively
    pushes apart two segments describing the same work. This groups by title and
    draws one segment from each of ``batch_size`` distinct titles.

    Args:
        pairs: The pair set to iterate.
        batch_size: Titles per batch. Silently clamped to the number of
            available titles, since a larger batch cannot be collision-free.
        seed: RNG seed.
        drop_last: Skip a trailing short batch.

    Yields:
        Index arrays into ``pairs``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    rng = np.random.default_rng(seed)
    order = np.argsort(pairs.item_pos, kind="stable")
    sorted_pos = pairs.item_pos[order]
    titles, starts, counts = np.unique(sorted_pos, return_index=True, return_counts=True)

    effective = min(batch_size, len(titles))
    # One pass over the data is defined by the busiest title, so every title is
    # sampled about as often as the rarest one allows without starving it.
    n_batches = max(1, len(pairs) // effective)

    for _ in range(n_batches):
        chosen = rng.choice(len(titles), size=effective, replace=False)
        picks = starts[chosen] + rng.integers(0, counts[chosen])
        batch = order[picks]
        if drop_last and batch.size < effective:
            continue
        yield batch.astype(np.int64)
