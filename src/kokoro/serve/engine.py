# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""The retrieval engine behind the API.

Loads a training run's artifacts and answers free-text mood queries. Kept
separate from the FastAPI layer so it can be benchmarked, tested and profiled
without an HTTP server in the way — the latency numbers this project reports
come from here, not from a route handler.

Serving is deliberately the cheap half of the design. All catalog embedding
happens offline at training time; a query costs one small encoder forward pass
plus one index lookup. That is what makes sub-100 ms feasible on a laptop, and
it is the concrete advantage over asking a frontier model per request.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["Engine", "Hit"]


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved title."""

    item_pos: int
    anime_id: int
    title: str
    score: float
    genres: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    year: int | None = None
    episodes: int | None = None
    members: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "anime_id": self.anime_id,
            "title": self.title,
            "score": round(self.score, 5),
            "genres": list(self.genres),
            "tags": list(self.tags),
            "year": self.year,
            "episodes": self.episodes,
            "members": self.members,
        }


class Engine:
    """Free-text mood retrieval over a trained catalog.

    Args:
        artifact_dir: Directory written by ``kokoro train``.
        corpus_dir: Directory written by ``kokoro corpus build``, for titles.
        index: ``"hnsw"`` for the approximate index, ``"exact"`` for brute
            force. Exact is the ground truth ANN recall is measured against.
        ef_search: HNSW query-time breadth. The one production knob that moves
            recall and latency together.

    Raises:
        FileNotFoundError: If the artifacts are missing.
    """

    def __init__(
        self,
        artifact_dir: str | Path = "artifacts/two_tower",
        corpus_dir: str | Path = "data/processed",
        *,
        index: str = "hnsw",
        ef_search: int = 64,
    ) -> None:
        import torch

        from kokoro.train.contrastive import ProjectionTower

        self.artifact_dir = Path(artifact_dir)
        bundle_path = self.artifact_dir / "query_tower.pt"
        vectors_path = self.artifact_dir / "item_embeddings_trained.npy"
        if not bundle_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                f"no serving artifacts in {self.artifact_dir} — run `kokoro train` first"
            )

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.encoder_name: str = bundle["encoder"]
        self.query_tower = ProjectionTower(
            bundle["input_dim"], bundle["output_dim"], bundle["dropout"]
        )
        self.query_tower.load_state_dict(bundle["query_tower"])
        self.query_tower.eval()

        self.vectors: npt.NDArray[np.float32] = np.load(vectors_path).astype(np.float32)
        self.item_ids: npt.NDArray[np.int64] = np.load(self.artifact_dir / "item_ids.npy")

        self._encoder: Any | None = None  # loaded lazily; it is the heavy part
        self._meta = self._load_metadata(Path(corpus_dir))
        #: Audience size per contiguous position, for the serving-time
        #: popularity floor. See :meth:`search`.
        self.members = np.array(
            [self._meta.get(int(raw), {}).get("members", 0) for raw in self.item_ids],
            dtype=np.int64,
        )
        self.index_kind = index
        self.index = self._build_index(index, ef_search)

    # -- construction helpers ------------------------------------------------

    def _load_metadata(self, corpus_dir: Path) -> dict[int, dict[str, Any]]:
        """Load display metadata keyed by raw ``anime_id``."""
        import pandas as pd

        titles = pd.read_parquet(corpus_dir / "titles.parquet")
        wanted = set(self.item_ids.tolist())
        titles = titles[titles["anime_id"].isin(wanted)]

        def _positive_int(value: object) -> int | None:
            """Coerce to a positive int, or None.

            NaT.year is NaN rather than None, and float(NaN) is truthy, so both
            traps have to be closed explicitly — this is the same failure that
            once crashed item_text on real rows.
            """
            try:
                n = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return int(n) if not math.isnan(n) and n > 0 else None

        def _year(value: object) -> int | None:
            return _positive_int(getattr(value, "year", None))

        def _strings(value: object) -> tuple[str, ...]:
            if value is None or isinstance(value, (str, bytes)):
                return ()
            if not isinstance(value, Iterable):
                return ()
            return tuple(str(v) for v in value)

        out: dict[int, dict[str, Any]] = {}
        for raw, title, genres, tags, aired, episodes, members in zip(
            titles["anime_id"].tolist(),
            titles["title"].tolist(),
            titles["genres"].tolist(),
            titles["tags"].tolist(),
            titles["aired_start"].tolist(),
            titles["episodes"].tolist(),
            titles["members"].tolist(),
            strict=True,
        ):
            out[int(raw)] = {
                "title": str(title),
                "members": _positive_int(members) or 0,
                "genres": _strings(genres),
                "tags": _strings(tags),
                "year": _year(aired),
                "episodes": _positive_int(episodes),
            }
        return out

    def _build_index(self, kind: str, ef_search: int) -> Any:
        """Build the vector index over the catalog."""
        from kokoro.index.ann import ExactIndex, HNSWIndex

        if kind == "exact":
            return ExactIndex().build(self.vectors)
        if kind == "hnsw":
            return HNSWIndex(ef_search=ef_search).build(self.vectors)
        raise ValueError(f"unknown index {kind!r}: use 'hnsw' or 'exact'")

    @property
    def encoder(self) -> Any:
        """The sentence encoder, loaded on first use."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name, device="cpu")
        return self._encoder

    # -- retrieval -----------------------------------------------------------

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed one free-text query into the shared retrieval space.

        Args:
            text: The user's query.

        Returns:
            A single L2-normalised vector of shape ``(1, output_dim)``.

        Raises:
            ValueError: If ``text`` is blank.
        """
        import torch

        if not text.strip():
            raise ValueError("query text must not be empty")

        raw = self.encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        with torch.no_grad():
            projected = self.query_tower(torch.as_tensor(np.asarray(raw), dtype=torch.float32))
        vector: npt.NDArray[np.float32] = projected.cpu().numpy().astype(np.float32)
        return vector

    def search(self, text: str, k: int = 10, *, min_members: int = 0) -> list[Hit]:
        """Retrieve the ``k`` titles closest to a free-text mood query.

        Args:
            text: The query.
            k: Number of results.
            min_members: Drop titles with fewer than this many MyAnimeList
                members. This is a **presentation** filter and is deliberately
                absent from every evaluated path: the model carries no
                popularity prior, which is good for catalog coverage (its
                popularity lift is 1.7x against the collaborative baselines'
                17-25x) and bad for a demo, where 40% of unfiltered results are
                titles below 10k members that a visitor will not recognise.
                Applying it during evaluation would inflate the metrics by
                smuggling in exactly the popularity bias those metrics exist to
                detect.

        Returns:
            Hits, best first.

        Raises:
            ValueError: If ``k`` is not positive.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        query = self.embed_query(text)
        if min_members > 0:
            # Over-fetch, then filter: asking the index for k and discarding
            # would return fewer than k results.
            eligible = np.flatnonzero(self.members >= min_members)
            if eligible.size:
                sims = (query @ self.vectors[eligible].T)[0]
                take = min(k, eligible.size)
                top = np.argpartition(-sims, kth=take - 1)[:take]
                top = top[np.argsort(-sims[top])]
                return self._to_hits(eligible[top], sims[top])

        ids, scores = self.index.search(query, k=min(k, self.vectors.shape[0]))

        return self._to_hits(ids[0], scores[0])

    def _to_hits(
        self, positions: npt.NDArray[np.int64], scores: npt.NDArray[np.float32]
    ) -> list[Hit]:
        """Attach display metadata to ranked positions."""
        hits: list[Hit] = []
        for pos, score in zip(positions.tolist(), scores.tolist(), strict=True):
            raw = int(self.item_ids[pos])
            meta = self._meta.get(raw, {})
            hits.append(
                Hit(
                    item_pos=int(pos),
                    anime_id=raw,
                    title=str(meta.get("title", f"unknown {raw}")),
                    score=float(score),
                    genres=meta.get("genres", ()),
                    tags=meta.get("tags", ()),
                    year=meta.get("year"),
                    episodes=meta.get("episodes"),
                    members=int(meta.get("members", 0)),
                )
            )
        return hits

    def stats(self) -> dict[str, Any]:
        """Return a description of what is loaded, for the health endpoint."""
        return {
            "catalog_size": int(self.vectors.shape[0]),
            "embedding_dim": int(self.vectors.shape[1]),
            "index": self.index_kind,
            "encoder": self.encoder_name,
            "artifact_dir": str(self.artifact_dir),
            "titles_with_metadata": len(self._meta),
        }

    def save_manifest(self, path: str | Path) -> Path:
        """Write a stamped description of the served model."""
        from kokoro import provenance

        target = Path(path)
        target.write_text(
            json.dumps(provenance.stamp(self.stats()), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return target
