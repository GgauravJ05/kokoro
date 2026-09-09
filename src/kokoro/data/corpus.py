# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Load a built corpus into the structures the models consume.

Raw MyAnimeList ids are sparse and unbounded; every model here indexes arrays by
id, so ids are remapped to contiguous positions on load. The mappings travel
with the data so a prediction can be turned back into a title.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from kokoro.eval.splits import Interactions

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["Corpus", "load_corpus"]


@dataclass(slots=True)
class Corpus:
    """A loaded corpus with contiguous ids.

    Attributes:
        interactions: Ratings as contiguous ``(user, item)`` pairs.
        titles: Title metadata indexed by raw ``anime_id``.
        reviews: Review rows carrying raw ``anime_id``.
        item_index: Raw ``anime_id`` to contiguous item position.
        item_debut: Debut timestamp per contiguous item position, in unix
            seconds, for :func:`~kokoro.eval.splits.cold_start_split`. Titles
            with no known air date get ``0`` so they are never treated as cold.
        manifest: The corpus manifest, including its version and known biases.
    """

    interactions: Interactions
    titles: pd.DataFrame
    reviews: pd.DataFrame
    item_index: dict[int, int]
    item_debut: npt.NDArray[np.int64]
    manifest: dict[str, Any]

    @property
    def version(self) -> str:
        """Corpus content hash."""
        return str(self.manifest["corpus_version"])

    def item_texts(self) -> list[str]:
        """Render every catalog item as the string the item tower encodes.

        Ordered by contiguous item position, so the result lines up with the
        embedding matrix a content model is built from.
        """
        from kokoro.features.text import build_item_texts

        by_id = self.titles.set_index("anime_id")
        rows: list[dict[str, Any]] = []
        for raw, _ in sorted(self.item_index.items(), key=lambda kv: kv[1]):
            if raw in by_id.index:
                row = {str(key): value for key, value in by_id.loc[raw].to_dict().items()}
                rows.append(row)
            else:
                rows.append({"title": f"unknown title {raw}"})
        return build_item_texts(rows)

    def title_variants(self) -> dict[int, tuple[str, ...]]:
        """Return every known title string per raw ``anime_id``, for masking."""
        out: dict[int, tuple[str, ...]] = {}
        cols = [
            c
            for c in ("title", "title_english", "title_japanese", "synonyms")
            if c in self.titles.columns
        ]
        frame = self.titles[["anime_id", *cols]]
        ids = frame["anime_id"].to_numpy()
        rest = frame[cols].to_numpy(dtype=object)
        for anime_id, values in zip(ids, rest, strict=True):
            out[int(anime_id)] = tuple(str(v) for v in values if isinstance(v, str) and v.strip())
        return out

    def item_titles(self) -> npt.NDArray[np.str_]:
        """Return display titles ordered by contiguous item position."""
        lookup = self.titles.set_index("anime_id")["title"]
        out = np.empty(len(self.item_index), dtype=object)
        for raw, pos in self.item_index.items():
            out[pos] = lookup.get(raw, f"<unknown {raw}>")
        return out.astype(str)


def load_corpus(
    path: str | Path = "data/processed",
    *,
    min_rating: int = 1,
    max_users: int | None = None,
    min_user_interactions: int = 5,
    seed: int = 1337,
) -> Corpus:
    """Load a built corpus, optionally subsampling users.

    Args:
        path: Directory containing the corpus and its manifest.
        min_rating: Drop ratings below this. The source uses ``-1`` for
            "watched, unrated", already removed at build time.
        max_users: Keep at most this many users, sampled at random. The full
            matrix is 6.3M interactions, which the pure-Python BPR loop cannot
            traverse in reasonable time; subsampling is how the baseline table
            stays runnable. It is recorded in the results so the number is never
            mistaken for a full-corpus one.
        min_user_interactions: Drop users with fewer interactions than this.
        seed: RNG seed for the user subsample.

    Returns:
        The loaded :class:`Corpus`.

    Raises:
        FileNotFoundError: If the corpus has not been built.
    """
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no corpus at {root} — run `kokoro corpus build` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    titles = pd.read_parquet(root / "titles.parquet")
    reviews = pd.read_parquet(root / "reviews.parquet")
    ratings = pd.read_parquet(root / "ratings.parquet")

    ratings = ratings[ratings["rating"] >= min_rating]

    counts = ratings.groupby("user_id").size()
    eligible = counts[counts >= min_user_interactions].index
    ratings = ratings[ratings["user_id"].isin(eligible)]

    if max_users is not None and ratings["user_id"].nunique() > max_users:
        rng = np.random.default_rng(seed)
        keep = rng.choice(ratings["user_id"].unique(), size=max_users, replace=False)
        ratings = ratings[ratings["user_id"].isin(keep)]

    user_codes, _ = pd.factorize(ratings["user_id"], sort=True)
    item_codes, item_uniques = pd.factorize(ratings["anime_id"], sort=True)
    item_index = {int(raw): pos for pos, raw in enumerate(item_uniques)}

    # Debut dates come from the catalog, not the rating matrix — which is the
    # only reason a cold-start split is possible on a source that carries no
    # interaction timestamps at all.
    debut_lookup = titles.dropna(subset=["aired_start"]).set_index("anime_id")["aired_start"]
    item_debut = np.zeros(len(item_index), dtype=np.int64)
    for raw, pos in item_index.items():
        ts = debut_lookup.get(raw)
        if ts is not None and not pd.isna(ts):
            item_debut[pos] = int(pd.Timestamp(ts).timestamp())

    interactions = Interactions(
        user=user_codes.astype(np.int64),
        item=item_codes.astype(np.int64),
        rating=ratings["rating"].to_numpy(dtype=np.float32),
        # The source carries no timestamps. Zeros make that explicit and cause a
        # temporal split to be obviously degenerate rather than quietly wrong;
        # use user_holdout_split on this corpus.
        timestamp=np.zeros(len(ratings), dtype=np.int64),
    )

    return Corpus(
        interactions=interactions,
        titles=titles,
        reviews=reviews,
        item_index=item_index,
        item_debut=item_debut,
        manifest=manifest,
    )
