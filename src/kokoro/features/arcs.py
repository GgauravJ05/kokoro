# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Binding review sentences to the arc they discuss.

This is the unglamorous step the trajectory model depends on. Reviews mention
position constantly — "the first cour", "after episode 12", "the Marley arc",
"by the halfway point" — and each phrase localises a sentence somewhere in the
runtime. Extracting that turns an unordered bag of opinions into a time series.

The extractor is deliberately a rule set rather than a model. Every rule is
inspectable, each returns a confidence, and low-confidence segments fall back to
whole-work commentary instead of inventing a position. A learned span-labeller
is the obvious upgrade, but only once this establishes the ceiling worth beating
— and only against a hand-annotated set, which is why
``scripts/annotate_arcs.py`` exists.
"""

from __future__ import annotations

import re

__all__ = ["ArcMention", "extract_arc_mentions", "segment_review", "to_arc_index"]

#: Ordered by specificity: an explicit episode number beats "the first half",
#: which beats a bare ordinal. First match wins.
#: RUF001 is suppressed below because the en and em dashes are deliberate:
#: reviewers really do type "episodes 3–7", and a plain hyphen would miss it.
_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (r"\bepisodes?\s+(\d{1,3})\s*[-–—to]+\s*(\d{1,3})\b", "episode_range", 0.95),  # noqa: RUF001
    (r"\bep(?:isode)?\.?\s*(\d{1,3})\b", "episode", 0.9),
    (r"\bchapters?\s+(\d{1,4})\s*[-–—to]+\s*(\d{1,4})\b", "chapter_range", 0.95),  # noqa: RUF001
    (r"\bch(?:apter)?\.?\s*(\d{1,4})\b", "chapter", 0.9),
    (r"\bseasons?\s*(\d)\b|\bs(\d)\b", "season", 0.85),
    (r"\b(?:the\s+)?(first|second|third|final|last)\s+(?:cour|arc|half|third|act)\b", "part", 0.7),
    (r"\b(?:the\s+)([A-Z][A-Za-z']{2,20})\s+arc\b", "named_arc", 0.8),
    (r"\b(?:the\s+)?(opening|beginning|start|ending|finale|climax|midpoint)\b", "region", 0.55),
)

_ORDINAL_POSITION: dict[str, float] = {
    "first": 0.0,
    "opening": 0.0,
    "beginning": 0.0,
    "start": 0.0,
    "second": 0.33,
    "midpoint": 0.5,
    "third": 0.66,
    "final": 1.0,
    "last": 1.0,
    "ending": 1.0,
    "finale": 1.0,
    "climax": 0.9,
}


class ArcMention:
    """A positional reference found in a review segment.

    Attributes:
        kind: Which pattern matched, e.g. ``"episode"`` or ``"part"``.
        position: Narrative position in ``[0, 1]``, or ``None`` when the
            mention is an absolute episode number and total length is unknown.
        episode: The absolute episode/chapter number, when one was stated.
        confidence: Confidence in ``[0, 1]``.
        span: ``(start, end)`` character offsets into the segment.
    """

    __slots__ = ("confidence", "episode", "kind", "position", "span")

    def __init__(
        self,
        kind: str,
        position: float | None,
        episode: int | None,
        confidence: float,
        span: tuple[int, int],
    ) -> None:
        self.kind = kind
        self.position = position
        self.episode = episode
        self.confidence = confidence
        self.span = span

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return (
            f"ArcMention(kind={self.kind!r}, position={self.position}, "
            f"episode={self.episode}, confidence={self.confidence})"
        )


def segment_review(body: str, *, min_chars: int = 40) -> list[str]:
    """Split a review body into candidate training segments.

    Sentence-level is too short to carry a mood judgement and paragraph-level
    mixes several, so this splits on sentence boundaries then re-joins until
    each segment clears ``min_chars``.

    Args:
        body: Raw review text.
        min_chars: Minimum characters per emitted segment.

    Returns:
        Segments in document order; may be empty for very short reviews.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    out: list[str] = []
    buf = ""
    for s in sentences:
        buf = f"{buf} {s}".strip()
        if len(buf) >= min_chars:
            out.append(buf)
            buf = ""
    if buf and out:
        out[-1] = f"{out[-1]} {buf}"
    elif buf:
        out.append(buf)
    return out


def extract_arc_mentions(segment: str, *, total_episodes: int | None = None) -> list[ArcMention]:
    """Find every positional reference in one segment.

    Args:
        segment: A single review segment.
        total_episodes: Runtime length, used to convert an absolute episode
            number into a narrative position. Without it, ``position`` stays
            ``None`` and only ``episode`` is populated.

    Returns:
        Mentions in the order the patterns are tried, most specific first.
    """
    found: list[ArcMention] = []
    for pattern, kind, conf in _PATTERNS:
        for m in re.finditer(pattern, segment, flags=re.IGNORECASE):
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            token = groups[0]
            episode: int | None = None
            position: float | None = None

            if token.isdigit():
                episode = int(token)
                if kind == "season":
                    # A season number is a coarse position; without a season
                    # map, assume equal-length seasons up to a plausible five.
                    position = min((episode - 1) / 4.0, 1.0)
                elif total_episodes:
                    position = min(episode / max(total_episodes, 1), 1.0)
            else:
                position = _ORDINAL_POSITION.get(token.lower())

            found.append(ArcMention(kind, position, episode, conf, m.span()))
    return found


def to_arc_index(position: float, n_arcs: int) -> int:
    """Bucket a narrative position in ``[0, 1]`` into one of ``n_arcs`` slots.

    Args:
        position: Narrative position.
        n_arcs: Number of trajectory buckets.

    Returns:
        A zero-based arc index.

    Raises:
        ValueError: If ``n_arcs`` is not positive or ``position`` is outside
            ``[0, 1]``.
    """
    if n_arcs < 1:
        raise ValueError(f"n_arcs must be >= 1, got {n_arcs}")
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0, 1], got {position}")
    return min(int(position * n_arcs), n_arcs - 1)
