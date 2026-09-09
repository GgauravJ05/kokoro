# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""FastAPI service exposing mood-conditioned retrieval.

Every response carries an ``X-Kokoro-Provenance`` header and every
recommendation carries its mood-axis explanation, because an unexplained
affective recommendation is not actionable — the user cannot tell whether the
system understood "not too heavy" or just matched the word "heavy".

AGPL-3.0 section 13: deploying this service over a network obliges you to offer
its complete corresponding source to its users. ``GET /source`` exists to
satisfy that; do not remove it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, Query, Request, Response
from pydantic import BaseModel, Field

from kokoro import __version__, provenance

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Awaitable, Callable

__all__ = ["Recommendation", "app"]


class MoodAxis(BaseModel):
    """One named axis of the explanation."""

    axis: str
    value: float = Field(ge=-1.0, le=1.0, description="-1 is the negative pole, +1 the positive.")
    pole: str = Field(description="Human-readable label for the pole this value leans toward.")


class Recommendation(BaseModel):
    """A single recommended title with its explanation."""

    title_id: int
    romaji: str
    english: str | None = None
    score: float = Field(description="Cosine similarity to the query in the shared space.")
    axes: list[MoodAxis] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list,
        description="Review sentences that drove the match, for user-visible justification.",
    )
    trajectory_warning: str | None = Field(
        default=None,
        description="Set when the mood curve inverts, e.g. 'comfortable until episode 8'.",
    )


class RecommendResponse(BaseModel):
    """The full response body."""

    query: str
    results: list[Recommendation]
    latency_ms: float
    model_version: str


app = FastAPI(
    title="Kokoro",
    description=(
        "Mood-conditioned anime & manga retrieval. "
        "AGPL-3.0 — source at https://github.com/GgauravJ05/kokoro"
    ),
    version=__version__,
    contact={"name": provenance.AUTHOR, "url": provenance.REPOSITORY},
    license_info={"name": "AGPL-3.0-or-later", "url": "https://www.gnu.org/licenses/agpl-3.0"},
)


@app.middleware("http")
async def attach_provenance(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Stamp every response with the build fingerprint."""
    response = await call_next(request)
    response.headers["X-Kokoro-Provenance"] = provenance.fingerprint()
    response.headers["X-Kokoro-License"] = provenance.LICENSE
    response.headers["X-Kokoro-Source"] = provenance.REPOSITORY
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@app.get("/source")
async def source() -> dict[str, str]:
    """Point users at the corresponding source, as AGPL-3.0 section 13 requires."""
    record = provenance.current()
    return {
        "repository": record.repository,
        "commit": record.git_commit or "unknown",
        "license": record.license,
        "notice": (
            "This service is AGPL-3.0. You are entitled to the complete "
            "corresponding source of the version running here."
        ),
    }


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    q: Annotated[str, Query(min_length=3, max_length=500, description="Free-text mood query.")],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    media_type: Annotated[str, Query(pattern="^(ANIME|MANGA)$")] = "ANIME",
) -> Any:
    """Retrieve titles matching a free-text mood query.

    Raises:
        NotImplementedError: Until a trained checkpoint is wired in. The route
            is defined now so the response contract — explanation axes and
            trajectory warnings included — is fixed before the model exists.
    """
    raise NotImplementedError("load a trained checkpoint into the app state; see docs/serving.md")
