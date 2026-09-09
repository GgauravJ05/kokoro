# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Turn the raw source files into one frozen, versioned corpus.

The output is three Parquet tables and a manifest. The manifest is the point:
it records the SHA-256 of every input, the row counts, the measured join rates,
the licence of each source and its known biases, plus a provenance stamp. A
training run cites a corpus version; anyone can then verify which bytes produced
which number.

Freezing matters more here than usual. The review corpus covers 490 titles while
the catalog holds 28,880, so the supervised fraction is 1.7% — small enough that
a silent change in the join would move every downstream metric.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from kokoro import provenance
from kokoro.data.sources import SOURCES, DownloadResult

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = ["CorpusManifest", "build_corpus", "normalise_catalog", "normalise_reviews"]

#: Rating sentinel used by the source matrix for "watched, did not rate".
UNRATED = -1

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}


def _parse_aired_start(value: object) -> pd.Timestamp | None:
    """Extract the start date from MyAnimeList's ``Aired`` string.

    The field looks like ``"Sep 29, 2023 to Mar 22, 2024"``, ``"Apr 3, 2016"``
    or ``"2021 to ?"``. Only the start is needed — it is what the cold-start
    split cuts on.

    Args:
        value: Raw ``Aired`` cell.

    Returns:
        The start date, or ``None`` when no year can be recovered.
    """
    if not isinstance(value, str):
        return None
    if m := re.match(r"([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})", value.strip()):
        month, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        return pd.Timestamp(year=year, month=_MONTHS.get(month, 1), day=day)
    if m := re.search(r"\b(\d{4})\b", value):
        return pd.Timestamp(year=int(m.group(1)), month=1, day=1)
    return None


def _split_list(value: object) -> list[str]:
    """Split a comma-separated cell into a clean list of strings."""
    if not isinstance(value, str) or value.strip().lower() in {"", "nan", "none"}:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def normalise_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise the title metadata table.

    The source carries no id column; the MyAnimeList id is embedded in each
    row's URL and is what joins this table to reviews and ratings.

    Args:
        raw: The catalog CSV as read.

    Returns:
        A frame keyed by ``anime_id``.

    Raises:
        ValueError: If no ids could be recovered, which would silently produce
            an empty join rather than an obvious failure.
    """
    df = pd.DataFrame()
    df["anime_id"] = pd.to_numeric(
        raw["URL"].astype(str).str.extract(r"/anime/(\d+)")[0], errors="coerce"
    ).astype("Int64")

    if df["anime_id"].notna().sum() == 0:
        raise ValueError("no anime ids could be extracted from the URL column")

    df["title"] = raw["Title"].astype("string")
    df["title_english"] = raw.get("English", pd.Series(dtype="string")).astype("string")
    df["title_japanese"] = raw.get("Japanese", pd.Series(dtype="string")).astype("string")

    synopsis_col = next((c for c in raw.columns if "syn" in c.lower()), None)
    df["synopsis"] = (
        raw[synopsis_col].astype("string") if synopsis_col else pd.Series(pd.NA, dtype="string")
    )

    df["media_type"] = raw["Type"].astype("string")
    df["episodes"] = pd.to_numeric(raw["Episodes"], errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(raw["Score"], errors="coerce")
    df["members"] = pd.to_numeric(
        raw["Members"].astype(str).str.replace(",", ""), errors="coerce"
    ).astype("Int64")
    df["favorites"] = pd.to_numeric(
        raw["Favorites"].astype(str).str.replace(",", ""), errors="coerce"
    ).astype("Int64")

    df["genres"] = raw["Genres"].map(_split_list)
    df["themes"] = raw["Themes"].map(_split_list)
    df["studios"] = raw["Studios"].map(_split_list)
    df["source_material"] = raw["Source"].astype("string")
    df["demographic"] = raw.get("Demographic", pd.Series(dtype="string")).astype("string")
    df["aired_start"] = raw["Aired"].map(_parse_aired_start)

    df = df.dropna(subset=["anime_id"]).drop_duplicates(subset=["anime_id"], keep="first")
    return df.reset_index(drop=True)


def normalise_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise the review table into the schema the trainer consumes.

    Args:
        raw: The reviews CSV as read. Columns are Spanish; ``texto_resena`` is
            the review body.

    Returns:
        A frame with ``review_id``, ``anime_id``, ``text``, ``recommended`` and
        ``n_chars``.

    Raises:
        KeyError: If the expected review-text column is absent.
    """
    if "texto_resena" not in raw.columns:
        raise KeyError(f"expected a 'texto_resena' column, got {list(raw.columns)}")

    df = pd.DataFrame()
    df["anime_id"] = pd.to_numeric(raw["anime_id"], errors="coerce").astype("Int64")
    df["text"] = raw["texto_resena"].astype("string")
    df["recommended"] = pd.to_numeric(raw["sentimiento"], errors="coerce").astype("Int8")
    df["verdict"] = raw["recomendacion_original"].astype("string")

    df = df.dropna(subset=["anime_id", "text"])
    df = df[df["text"].str.len() > 0]

    # Reviews carry no stable id upstream, so mint a deterministic one from the
    # content: re-running ingestion gives the same ids, and an exact duplicate
    # collapses to one row instead of being counted twice as training signal.
    df["review_id"] = pd.util.hash_pandas_object(
        df["anime_id"].astype(str) + "|" + df["text"], index=False
    ).astype("uint64")
    df = df.drop_duplicates(subset=["review_id"], keep="first")

    df["n_chars"] = df["text"].str.len().astype("int32")
    return df[["review_id", "anime_id", "text", "recommended", "verdict", "n_chars"]].reset_index(
        drop=True
    )


def normalise_ratings(raw: pd.DataFrame, *, drop_unrated: bool = True) -> pd.DataFrame:
    """Normalise the user-item rating matrix.

    Args:
        raw: The ratings CSV as read.
        drop_unrated: Drop the ``-1`` sentinel rows meaning "watched, unrated".
            Kept out by default: BPR treats every row as a positive, and a
            watched-but-unrated title is not evidence the user liked it.

    Returns:
        A frame with ``user_id``, ``anime_id`` and ``rating``.
    """
    df = pd.DataFrame(
        {
            "user_id": pd.to_numeric(raw["user_id"], errors="coerce").astype("Int64"),
            "anime_id": pd.to_numeric(raw["anime_id"], errors="coerce").astype("Int64"),
            "rating": pd.to_numeric(raw["rating"], errors="coerce").astype("Int8"),
        }
    ).dropna()
    if drop_unrated:
        df = df[df["rating"] != UNRATED]
    return df.reset_index(drop=True)


class CorpusManifest(dict[str, Any]):
    """The frozen description of one built corpus."""

    @property
    def version(self) -> str:
        """Content hash identifying this corpus."""
        return str(self["corpus_version"])


def build_corpus(
    downloads: Mapping[str, DownloadResult],
    out_dir: Path,
    *,
    min_reviews_per_title: int = 5,
) -> CorpusManifest:
    """Build the frozen corpus from downloaded source files.

    Args:
        downloads: Mapping of source key to its :class:`DownloadResult`.
        out_dir: Directory to write the Parquet tables and manifest into.
        min_reviews_per_title: Titles below this are excluded from the
            supervised set. A title with two reviews cannot support a stable
            mood estimate, and including it inflates the apparent catalog.

    Returns:
        The manifest, also written to ``out_dir/manifest.json``.

    Raises:
        KeyError: If a required source is missing from ``downloads``.
    """
    for required in ("catalog", "reviews", "ratings"):
        if required not in downloads:
            raise KeyError(f"missing required source {required!r}")

    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = normalise_catalog(pd.read_csv(downloads["catalog"].path))
    reviews = normalise_reviews(pd.read_csv(downloads["reviews"].path))
    ratings = normalise_ratings(
        pd.read_csv(downloads["ratings"].path, names=["user_id", "anime_id", "rating"], header=0)
    )

    catalog_ids = set(catalog["anime_id"].dropna().astype(int))

    # Join rates are recorded rather than assumed: they are the number that
    # silently breaks everything downstream if a source is updated upstream.
    review_ids = set(reviews["anime_id"].dropna().astype(int))
    reviews_joined = reviews[reviews["anime_id"].isin(catalog_ids)].copy()
    rating_ids = set(ratings["anime_id"].dropna().astype(int))
    ratings_joined = ratings[ratings["anime_id"].isin(catalog_ids)].copy()

    counts = reviews_joined.groupby("anime_id").size()
    supervised = set(counts[counts >= min_reviews_per_title].index.astype(int))
    reviews_joined = reviews_joined[reviews_joined["anime_id"].isin(supervised)]

    catalog.to_parquet(out_dir / "titles.parquet", index=False)
    reviews_joined.to_parquet(out_dir / "reviews.parquet", index=False)
    ratings_joined.to_parquet(out_dir / "ratings.parquet", index=False)

    stats = {
        "catalog_titles": len(catalog),
        "reviews_total": len(reviews),
        "reviews_joined": len(reviews_joined),
        "review_join_rate": round(len(reviews_joined) / max(len(reviews), 1), 4),
        "review_titles_total": len(review_ids),
        "review_titles_in_catalog": len(review_ids & catalog_ids),
        "supervised_titles": len(supervised),
        "supervised_fraction_of_catalog": round(len(supervised) / max(len(catalog), 1), 5),
        "ratings_total": len(ratings),
        "ratings_joined": len(ratings_joined),
        "rating_join_rate": round(len(ratings_joined) / max(len(ratings), 1), 4),
        "rating_titles_in_catalog": len(rating_ids & catalog_ids),
        "rating_users": int(ratings_joined["user_id"].nunique()),
        "median_reviews_per_supervised_title": float(
            counts[counts >= min_reviews_per_title].median()
        ),
        "median_review_chars": float(reviews_joined["n_chars"].median()),
    }

    # The corpus version is the hash of the inputs plus the build parameters, so
    # changing either produces a different version and no result can be
    # mistakenly attributed to the wrong data.
    import hashlib

    fingerprint = hashlib.sha256(
        "|".join(
            [downloads[k].sha256 for k in sorted(downloads)] + [str(min_reviews_per_title)]
        ).encode()
    ).hexdigest()[:16]

    manifest = CorpusManifest(
        provenance.stamp(
            {
                "corpus_version": fingerprint,
                "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "build_params": {"min_reviews_per_title": min_reviews_per_title},
                "statistics": stats,
                "tables": {
                    "titles": "titles.parquet",
                    "reviews": "reviews.parquet",
                    "ratings": "ratings.parquet",
                },
                "sources": {
                    key: {
                        **{k: v for k, v in asdict(SOURCES[key]).items() if k not in {"rows"}},
                        "sha256": dl.sha256,
                        "bytes": dl.bytes,
                        "url": SOURCES[key].url,
                    }
                    for key, dl in downloads.items()
                },
                "redistribution": (
                    "No source file is redistributed by this repository. data/ is "
                    "gitignored. The reviews source declares licence 'other' and the "
                    "ratings source declares none; both are ingested for "
                    "non-commercial research only."
                ),
            }
        )
    )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest
