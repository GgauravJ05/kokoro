# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Validated record types for everything that enters the corpus.

Third-party APIs return inconsistent payloads — missing synopses, null episode
counts, HTML in description fields. Validating at the boundary means the rest of
the pipeline can assume its inputs, and a schema change upstream fails loudly at
ingestion rather than silently poisoning a training run.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["Review", "ReviewSegment", "Title"]

MediaType = Literal["ANIME", "MANGA"]


class Title(BaseModel):
    """One anime or manga entry."""

    model_config = ConfigDict(frozen=True)

    id: int
    source: Literal["anilist", "mal"]
    media_type: MediaType
    romaji: str
    english: str | None = None
    native: str | None = None
    synopsis: str | None = None
    genres: tuple[str, ...] = ()
    #: AniList descriptive tags with their crowd-confidence percentage. Far
    #: richer than genres, and the closest thing to a mood label that exists
    #: upstream — used as weak supervision, never as ground truth.
    tags: tuple[tuple[str, int], ...] = ()
    episodes: int | None = Field(default=None, ge=0)
    chapters: int | None = Field(default=None, ge=0)
    mean_score: float | None = Field(default=None, ge=0, le=100)
    popularity: int | None = Field(default=None, ge=0)
    start_date: date | None = None
    studios: tuple[str, ...] = ()
    #: Set when this entry is a sequel/adaptation of another, so the corpus can
    #: avoid leaking a season's own sequel into its evaluation set.
    relations: tuple[int, ...] = ()

    @field_validator("synopsis")
    @classmethod
    def _strip_markup(cls, v: str | None) -> str | None:
        """Remove the ``<br>`` and ``<i>`` markup AniList embeds in descriptions."""
        if v is None:
            return None
        import re

        return re.sub(r"<[^>]+>", " ", v).strip() or None


class Review(BaseModel):
    """A user review of a title."""

    model_config = ConfigDict(frozen=True)

    id: int
    title_id: int
    source: Literal["anilist", "mal"]
    body: str
    score: int | None = Field(default=None, ge=0, le=100)
    #: Community upvotes. Used to weight the contrastive training signal:
    #: a review nobody found helpful is noisier supervision.
    helpful_votes: int = 0
    created_at: date | None = None
    #: Whether the reviewer flagged spoilers. Spoiler-marked segments are
    #: excluded from user-facing explanations but kept for training.
    has_spoilers: bool = False


class ReviewSegment(BaseModel):
    """A single sentence or paragraph of a review, bound to a narrative arc.

    This is the actual training unit: the contrastive objective pairs one
    segment with the title it discusses. Binding a segment to an arc is what
    makes trajectory modelling possible.
    """

    model_config = ConfigDict(frozen=True)

    review_id: int
    title_id: int
    text: str
    #: Zero-based index of the arc this segment discusses, or ``None`` when the
    #: segment is about the work as a whole.
    arc: int | None = Field(default=None, ge=0)
    #: Confidence of the arc assignment in ``[0, 1]``; segments below the
    #: pipeline threshold are treated as whole-work commentary.
    arc_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Position in the review, so an opening thesis can be weighted above an
    #: aside.
    position: int = Field(default=0, ge=0)
