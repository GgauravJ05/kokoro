# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Turn structured metadata and review prose into the strings the towers encode.

Two builders, one per tower.

:func:`item_text` renders a title as a sentence. This is the only thing a
cold-start title has — no interactions, and this catalog carries no synopsis at
all — so the whole cold-start claim rests on how much signal survives here.
AniList tags do most of the work: ``Tragedy``, ``Philosophy``, ``Found Family``
carry mood in a way ``Drama`` does not.

:func:`query_text` prepares a review segment. Its one non-obvious job is
:func:`mask_title_mentions`: a review that names its own show lets the model
match on the title string instead of on mood, which would inflate every training
metric while learning nothing transferable — and would collapse completely at
cold start, where the title has never been seen.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

__all__ = ["item_text", "mask_title_mentions", "query_text"]

#: Replacement for a masked title mention. A neutral placeholder rather than
#: deletion, so the sentence keeps its grammatical shape.
TITLE_MASK = "this show"

#: Tokens too generic to be worth masking on their own — masking them would
#: shred unrelated sentences ("the movie was slow" becoming "the this show was
#: slow"). Only multi-word or distinctive title fragments are masked.
_STOP_TITLE_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "no",
        "wa",
        "ga",
        "to",
        "ni",
        "wo",
        "de",
        "season",
        "movie",
        "film",
        "part",
        "ova",
        "special",
        "tv",
        "series",
        "first",
        "second",
        "third",
        "final",
        "not",
        "is",
        "it",
        "in",
        "on",
    }
)


def _title_variants(*titles: str | None) -> list[str]:
    """Return distinctive title strings worth masking, longest first.

    Args:
        *titles: Romaji, English, native and synonym forms; ``None`` is skipped.

    Returns:
        Deduplicated variants, longest first so the fullest match is replaced
        before any of its fragments.
    """
    out: set[str] = set()
    for raw in titles:
        if not raw or not isinstance(raw, str):
            continue
        for piece in re.split(r"[;,]", raw):
            cleaned = piece.strip()
            if len(cleaned) < 4:
                continue
            out.add(cleaned)
            # Both halves around a colon leak. Reviewers say "Fullmetal
            # Alchemist" and "Brotherhood" interchangeably, and the subtitle is
            # very often the half they actually use.
            for half in cleaned.split(":"):
                part = half.strip()
                if len(part) >= 4 and part.lower() not in _STOP_TITLE_WORDS:
                    out.add(part)
    return sorted(out, key=len, reverse=True)


def mask_title_mentions(text: str, *titles: str | None, mask: str = TITLE_MASK) -> str:
    """Replace mentions of a work's own title with a neutral placeholder.

    Without this the contrastive objective has a shortcut: the review of
    *Steins;Gate* contains the string "steins;gate", so the model can score the
    pair by string overlap and never learn anything about mood. The shortcut is
    invisible in training metrics and fatal at cold start.

    Matching is case-insensitive and whole-word. Single common words are never
    masked; see :data:`_STOP_TITLE_WORDS`.

    Args:
        text: The review segment.
        *titles: Known title variants for the work under review.
        mask: Replacement string.

    Returns:
        The segment with title mentions replaced.
    """
    out = text
    for variant in _title_variants(*titles):
        if variant.lower() in _STOP_TITLE_WORDS:
            continue
        out = re.sub(rf"\b{re.escape(variant)}\b", mask, out, flags=re.IGNORECASE)
    return out


def _as_int(value: Any) -> int | None:
    """Coerce a possibly-missing numeric cell to ``int``.

    Guards two NaN traps that both surface only on real data: ``NaT.year`` is
    ``nan`` rather than ``None``, and ``if value:`` is *true* for ``nan``, so a
    naive truthiness check walks straight into ``int(nan)``.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    import math

    if math.isnan(f) or f <= 0:
        return None
    return int(f)


def _as_list(value: Any) -> list[str]:
    """Coerce a possibly-array cell into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(v).strip() for v in value if str(v).strip()]
    except TypeError:
        return []


def item_text(
    *,
    title: str,
    genres: Iterable[str] | None = None,
    themes: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    studios: Iterable[str] | None = None,
    media_type: str | None = None,
    episodes: int | None = None,
    year: int | None = None,
    demographic: str | None = None,
    max_tags: int = 15,
) -> str:
    """Render a title's metadata as one natural-language string.

    Written as prose rather than a delimited field dump because the encoders
    used here were pretrained on sentences; a bag of comma-separated tokens sits
    off-distribution and embeds worse.

    Args:
        title: Display title.
        genres: Broad genre labels.
        themes: MyAnimeList theme labels.
        tags: AniList descriptive tags — the strongest mood signal available,
            so they are placed first among the descriptors.
        studios: Producing studios.
        media_type: TV, Movie, OVA and so on.
        episodes: Episode count.
        year: Debut year.
        demographic: Target demographic.
        max_tags: Cap on tags, to keep the string inside the encoder's window.

    Returns:
        A single string suitable for the item tower.
    """
    parts: list[str] = [title.strip()]

    kind = []
    if media_type and str(media_type).lower() not in {"nan", "none", "unknown"}:
        kind.append(str(media_type))
    if (n_eps := _as_int(episodes)) is not None:
        kind.append(f"{n_eps} episode{'s' if n_eps != 1 else ''}")
    if (n_year := _as_int(year)) is not None:
        kind.append(f"from {n_year}")
    if kind:
        parts.append(", ".join(kind))

    tag_list = _as_list(tags)[:max_tags]
    if tag_list:
        parts.append("Themes and tags: " + ", ".join(tag_list))

    genre_list = _as_list(genres)
    if genre_list:
        parts.append("Genres: " + ", ".join(genre_list))

    theme_list = [t for t in _as_list(themes) if t not in set(tag_list)]
    if theme_list:
        parts.append("Also: " + ", ".join(theme_list))

    if demographic and str(demographic).lower() not in {"nan", "none"}:
        parts.append(f"Aimed at a {demographic} audience")

    studio_list = _as_list(studios)
    if studio_list:
        parts.append("Animated by " + ", ".join(studio_list[:3]))

    return ". ".join(p.rstrip(".") for p in parts if p) + "."


def query_text(segment: str, *titles: str | None, max_chars: int = 512) -> str:
    """Prepare a review segment for the query tower.

    Args:
        segment: One review segment.
        *titles: Title variants of the work under review, masked out.
        max_chars: Truncation limit, applied at a word boundary.

    Returns:
        The cleaned segment.
    """
    text = mask_title_mentions(" ".join(segment.split()), *titles)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def build_item_texts(titles: Sequence[dict[str, Any]]) -> list[str]:
    """Render many titles at once.

    Args:
        titles: Row mappings carrying the fields :func:`item_text` accepts.

    Returns:
        One rendered string per input row, in order.
    """
    out: list[str] = []
    for row in titles:
        aired = row.get("aired_start")
        year = getattr(aired, "year", None) if aired is not None else None
        out.append(
            item_text(
                title=str(row.get("title", "")),
                genres=row.get("genres"),
                themes=row.get("themes"),
                tags=row.get("tags"),
                studios=row.get("studios"),
                media_type=row.get("media_type"),
                episodes=row.get("episodes"),
                year=year,
                demographic=row.get("demographic"),
            )
        )
    return out
