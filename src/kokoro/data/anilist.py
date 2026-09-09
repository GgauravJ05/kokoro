# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""AniList GraphQL client.

AniList publishes a documented rate limit and this client stays well under it:
a token-bucket limiter paces requests, and 429 responses are honoured by
sleeping for the server's ``Retry-After`` rather than retrying immediately. That
is not politeness theatre — a scraper that ignores published limits gets the
project's data source revoked, and the ingestion is not reproducible without it.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from kokoro.config import Settings, get_settings
from kokoro.data.schema import Title

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncIterator

__all__ = ["AniListClient", "RateLimiter"]

MEDIA_QUERY = """
query ($page: Int, $perPage: Int, $type: MediaType) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage currentPage lastPage }
    media(type: $type, sort: POPULARITY_DESC, isAdult: false) {
      id
      type
      title { romaji english native }
      description
      genres
      tags { name rank isGeneralSpoiler }
      episodes
      chapters
      meanScore
      popularity
      startDate { year month day }
      studios(isMain: true) { nodes { name } }
      relations { edges { node { id } } }
    }
  }
}
"""


class RateLimiter:
    """A single-slot token bucket enforcing a maximum request rate.

    Args:
        rps: Requests per second.

    Raises:
        ValueError: If ``rps`` is not positive.
    """

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError(f"rps must be positive, got {rps}")
        self._interval = 1.0 / rps
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until the next request is allowed."""
        async with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class AniListClient:
    """Async client for the AniList GraphQL API.

    Args:
        settings: Configuration; defaults to the process settings.

    Example:
        >>> async with AniListClient() as client:  # doctest: +SKIP
        ...     async for title in client.iter_titles("ANIME", max_pages=2):
        ...         print(title.romaji)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._limiter = RateLimiter(self.settings.anilist_rps)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AniListClient:
        """Open the underlying HTTP connection pool."""
        self._client = httpx.AsyncClient(
            timeout=self.settings.request_timeout_s,
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Execute one GraphQL request, respecting rate limits and backoff.

        Raises:
            RuntimeError: If the client is used outside its context manager, or
                if the response carries GraphQL-level errors.
        """
        if self._client is None:
            raise RuntimeError("use AniListClient as an async context manager")

        await self._limiter.acquire()
        resp = await self._client.post(
            self.settings.anilist_endpoint, json={"query": query, "variables": variables}
        )
        if resp.status_code == 429:
            # Honour the server's own backoff instead of guessing.
            await asyncio.sleep(float(resp.headers.get("Retry-After", "60")))
            resp.raise_for_status()
        resp.raise_for_status()

        payload: dict[str, Any] = resp.json()
        if payload.get("errors"):
            raise RuntimeError(f"AniList GraphQL errors: {payload['errors']}")
        data: dict[str, Any] = payload["data"]
        return data

    async def iter_titles(
        self, media_type: str = "ANIME", *, per_page: int = 50, max_pages: int | None = None
    ) -> AsyncIterator[Title]:
        """Yield validated titles in descending popularity order.

        Args:
            media_type: ``"ANIME"`` or ``"MANGA"``.
            per_page: Page size; AniList caps this at 50.
            max_pages: Stop after this many pages. ``None`` walks to the end.

        Yields:
            Validated :class:`~kokoro.data.schema.Title` records.
        """
        page = 1
        while max_pages is None or page <= max_pages:
            data = await self._post(
                MEDIA_QUERY, {"page": page, "perPage": min(per_page, 50), "type": media_type}
            )
            block = data["Page"]
            for node in block["media"]:
                yield self._to_title(node)
            if not block["pageInfo"]["hasNextPage"]:
                return
            page += 1

    @staticmethod
    def _to_title(node: dict[str, Any]) -> Title:
        """Map one GraphQL media node onto the internal schema."""
        from datetime import date

        d = node.get("startDate") or {}
        start = date(d["year"], d.get("month") or 1, d.get("day") or 1) if d.get("year") else None

        return Title(
            id=node["id"],
            source="anilist",
            media_type=node["type"],
            romaji=node["title"]["romaji"],
            english=node["title"].get("english"),
            native=node["title"].get("native"),
            synopsis=node.get("description"),
            genres=tuple(node.get("genres") or ()),
            # Spoiler tags are dropped: they leak plot into what is meant to be
            # a spoiler-safe user-facing representation.
            tags=tuple(
                (t["name"], t["rank"])
                for t in (node.get("tags") or [])
                if not t.get("isGeneralSpoiler")
            ),
            episodes=node.get("episodes"),
            chapters=node.get("chapters"),
            mean_score=node.get("meanScore"),
            popularity=node.get("popularity"),
            start_date=start,
            studios=tuple(s["name"] for s in (node.get("studios") or {}).get("nodes", [])),
            relations=tuple(
                e["node"]["id"] for e in (node.get("relations") or {}).get("edges", [])
            ),
        )
