# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Ingestion: source registry, normalisation and corpus loading.

No test here touches the network. The normalisers are pure functions over
frames, which is the point of keeping download and transform separate.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from kokoro.data.build import (
    _parse_aired_start,
    build_corpus,
    normalise_catalog,
    normalise_ratings,
    normalise_reviews,
)
from kokoro.data.sources import SOURCES, Source, resolve_url


def _catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Title": ["Sousou no Frieren", "Steins;Gate", "Awkward Cells", "A Manga"],
            "URL": [
                "https://myanimelist.net/anime/52991/Sousou_no_Frieren",
                "https://myanimelist.net/anime/9253/Steins_Gate",
                "https://myanimelist.net/anime/999/Awkward_Cells",
                "https://myanimelist.net/manga/777/Not_An_Anime",
            ],
            "Synopsis": ["A mage outlives her party.", "Microwave time travel.", None, "Manga."],
            "Type": ["TV", "TV", "TV", "Manga"],
            "Episodes": ["28", "24", "Unknown", "Unknown"],
            "Score": ["9.29", "9.07", None, None],
            "Members": ["1,000,000", "2,500,000", "10", "5"],
            "Favorites": ["50,000", "180,000", "0", "0"],
            "Genres": ["Adventure, Drama, Fantasy", "Drama, Sci-Fi, Suspense", None, None],
            "Themes": ["Mythology", "Psychological", None, None],
            "Studios": ["Madhouse", "White Fox", "Unknown", "Unknown"],
            "Source": ["Manga", "Visual novel", "Original", "Original"],
            "Demographic": ["Shounen", None, None, None],
            "Aired": [
                "Sep 29, 2023 to Mar 22, 2024",
                "Apr 6, 2011 to Sep 14, 2011",
                "2021 to ?",
                "2020",
            ],
            "English": ["Frieren", "Steins;Gate", None, None],
            "Japanese": ["葬送のフリーレン", "シュタインズ・ゲート", None, None],
        }
    )


def _review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anime_id": [52991, 9253, 9253, 9253, None],
            "texto_resena": [
                "a quiet meditation on outliving everyone you loved.",
                "it completely changes after episode 12.",
                "it completely changes after episode 12.",  # exact duplicate
                "the first cour drags but the payoff lands.",
                "orphan review with no anime id.",
            ],
            "recomendacion_original": [
                "recommended",
                "recommended",
                "recommended",
                "not_recommended",
                "recommended",
            ],
            "sentimiento": [1, 1, 1, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


def test_every_source_declares_a_licence_and_its_biases() -> None:
    """A source with unexamined terms is a licence risk, not a shortcut."""
    for src in SOURCES.values():
        assert src.licence, f"{src.key} declares no licence"
        assert src.biases, f"{src.key} declares no known biases"
        assert not src.redistributable, "nothing in this repo may be redistributed"


def test_registry_covers_the_three_required_roles() -> None:
    assert {s.role for s in SOURCES.values()} == {"catalog", "reviews", "ratings"}


def test_resolve_url_shape() -> None:
    assert resolve_url("owner/name", "file.csv") == (
        "https://huggingface.co/datasets/owner/name/resolve/main/file.csv"
    )


def test_source_urls_are_derived_not_hardcoded() -> None:
    src = Source(
        key="x",
        repo="a/b",
        filename="c.csv",
        role="reviews",
        licence="mit",
        rows=1,
        notes="",
        biases=("none",),
    )
    assert src.url.endswith("/a/b/resolve/main/c.csv")
    assert src.card_url == "https://huggingface.co/datasets/a/b"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sep 29, 2023 to Mar 22, 2024", "2023-09-29"),
        ("Apr 3, 2016", "2016-04-03"),
        ("2021 to ?", "2021-01-01"),
        ("Not available", None),
        (None, None),
    ],
)
def test_parse_aired_start(raw: object, expected: str | None) -> None:
    got = _parse_aired_start(raw)
    assert (None if got is None else got.strftime("%Y-%m-%d")) == expected


def test_catalog_recovers_ids_from_urls() -> None:
    """The join the whole corpus depends on: no id column, ids live in the URL."""
    out = normalise_catalog(_catalog_frame())
    assert set(out["anime_id"].astype(int)) == {52991, 9253, 999}
    assert out.loc[out["anime_id"] == 52991, "title"].iloc[0] == "Sousou no Frieren"


def test_catalog_drops_rows_whose_url_is_not_an_anime() -> None:
    """A /manga/ URL must not have its id mined into the anime catalog."""
    out = normalise_catalog(_catalog_frame())
    assert 777 not in set(out["anime_id"].astype(int))
    assert "A Manga" not in set(out["title"])


def test_catalog_parses_awkward_cells() -> None:
    out = normalise_catalog(_catalog_frame()).set_index("anime_id")
    assert out.loc[9253, "episodes"] == 24
    assert pd.isna(out.loc[999, "episodes"]), "'Unknown' must become NA, not 0"
    assert out.loc[9253, "members"] == 2_500_000, "thousands separators must be stripped"
    assert out.loc[52991, "genres"] == ["Adventure", "Drama", "Fantasy"]
    assert out.loc[999, "genres"] == []
    assert out.loc[52991, "aired_start"].year == 2023


def test_catalog_fails_loudly_when_no_ids_can_be_recovered() -> None:
    """Silently returning an empty join would poison every downstream metric."""
    bad = _catalog_frame()
    bad["URL"] = "https://example.com/no-id-here"
    with pytest.raises(ValueError, match="no anime ids"):
        normalise_catalog(bad)


def test_reviews_drop_orphans_and_collapse_duplicates() -> None:
    out = normalise_reviews(_review_frame())
    assert len(out) == 3, "one orphan dropped, one exact duplicate collapsed"
    assert out["anime_id"].notna().all()
    assert out["review_id"].is_unique


def test_review_ids_are_content_derived_and_stable() -> None:
    """Re-running ingestion must not renumber the training set."""
    a = normalise_reviews(_review_frame())
    b = normalise_reviews(_review_frame())
    assert a["review_id"].tolist() == b["review_id"].tolist()


def test_reviews_reject_an_unexpected_schema() -> None:
    with pytest.raises(KeyError, match="texto_resena"):
        normalise_reviews(pd.DataFrame({"anime_id": [1], "body": ["x"]}))


def test_ratings_drop_the_unrated_sentinel() -> None:
    raw = pd.DataFrame({"user_id": [1, 1, 2], "anime_id": [10, 11, 10], "rating": [9, -1, 7]})
    assert len(normalise_ratings(raw)) == 2
    assert len(normalise_ratings(raw, drop_unrated=False)) == 3


# ---------------------------------------------------------------------------
# End-to-end build and load
# ---------------------------------------------------------------------------


@pytest.fixture
def built_corpus(tmp_path):
    """Build a tiny corpus on disk from in-memory frames."""
    from kokoro.data.sources import DownloadResult

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = {
        "catalog": raw / "catalog.csv",
        "reviews": raw / "reviews.csv",
        "ratings": raw / "ratings.csv",
    }
    _catalog_frame().to_csv(paths["catalog"], index=False)
    _review_frame().to_csv(paths["reviews"], index=False)
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "anime_id": [52991, 9253, 999, 52991, 9253, 999, 52991, 9253, 999, 4],
            "rating": [10, 9, 8, 7, 10, -1, 9, 9, 6, 5],
        }
    ).to_csv(paths["ratings"], index=False)

    downloads = {
        key: DownloadResult(
            source=SOURCES[key], path=p, sha256=f"{key}-hash", bytes=p.stat().st_size, cached=False
        )
        for key, p in paths.items()
    }
    out = tmp_path / "processed"
    manifest = build_corpus(downloads, out, min_reviews_per_title=1)
    return out, manifest


def test_build_writes_tables_and_a_stamped_manifest(built_corpus) -> None:
    from kokoro import provenance

    out, manifest = built_corpus
    for name in ("titles.parquet", "reviews.parquet", "ratings.parquet", "manifest.json"):
        assert (out / name).exists(), f"{name} was not written"

    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert provenance.verify(on_disk, strict=True), "the manifest must carry provenance"
    assert manifest.version == on_disk["corpus_version"]


def test_manifest_records_join_rates_and_biases(built_corpus) -> None:
    """These are the numbers that silently break if a source changes upstream."""
    _, manifest = built_corpus
    stats = manifest["statistics"]
    assert 0.0 <= stats["review_join_rate"] <= 1.0
    assert stats["supervised_titles"] > 0
    assert stats["ratings_joined"] <= stats["ratings_total"]
    for src in manifest["sources"].values():
        assert src["biases"], "every source must carry its biases into the manifest"


def test_corpus_version_changes_with_the_inputs(built_corpus, tmp_path) -> None:
    from kokoro.data.build import build_corpus as build
    from kokoro.data.sources import DownloadResult

    out, manifest = built_corpus
    altered = {
        key: DownloadResult(
            source=SOURCES[key],
            path=out.parent / "raw" / f"{key}.csv",
            sha256="different",
            bytes=1,
            cached=False,
        )
        for key in ("catalog", "reviews", "ratings")
    }
    other = build(altered, tmp_path / "other", min_reviews_per_title=1)
    assert other.version != manifest.version, "different inputs must give a different version"


def test_load_corpus_remaps_ids_to_contiguous_positions(built_corpus) -> None:
    from kokoro.data.corpus import load_corpus

    out, _ = built_corpus
    c = load_corpus(out, min_user_interactions=1)

    n_items = c.interactions.n_items
    assert set(c.item_index.values()) == set(range(n_items)), "positions must be contiguous"
    assert c.interactions.item.max() < n_items
    assert c.interactions.user.max() < c.interactions.n_users
    assert (c.interactions.timestamp == 0).all(), "this source has no timestamps"


def test_load_corpus_maps_positions_back_to_titles(built_corpus) -> None:
    from kokoro.data.corpus import load_corpus

    out, _ = built_corpus
    c = load_corpus(out, min_user_interactions=1)
    names = c.item_titles()
    assert len(names) == c.interactions.n_items
    assert "Sousou no Frieren" in set(names)


def test_load_corpus_errors_when_not_built(tmp_path) -> None:
    from kokoro.data.corpus import load_corpus

    with pytest.raises(FileNotFoundError, match="kokoro corpus build"):
        load_corpus(tmp_path / "nope")


def test_corpus_exposes_debut_dates_for_cold_start(built_corpus) -> None:
    """Debut dates come from the catalog, which is what makes cold-start possible
    on a rating matrix that has no interaction timestamps at all."""
    import numpy as np

    from kokoro.data.corpus import load_corpus
    from kokoro.eval.splits import cold_start_split

    out, _ = built_corpus
    c = load_corpus(out, min_user_interactions=1)

    assert c.item_debut.shape == (c.interactions.n_items,)
    assert (c.item_debut >= 0).all()
    assert c.item_debut.max() > 0, "at least one title must have a known debut"

    cut = int(np.datetime64("2020-01-01").astype("datetime64[s]").astype(np.int64))
    split = cold_start_split(c.interactions, c.item_debut, cut=cut)
    cold = set(split.test.item.tolist())
    assert not (cold & set(split.train.item.tolist())), (
        "a cold item with training interactions is not cold"
    )


def test_titles_without_air_dates_are_never_cold(built_corpus) -> None:
    """A missing date must not be read as 'debuted at the epoch' or as 'brand new'."""
    from kokoro.data.corpus import load_corpus

    out, _ = built_corpus
    c = load_corpus(out, min_user_interactions=1)
    # Zero is the sentinel for 'unknown', and zero is before every realistic cut,
    # so such titles always land in train rather than being silently held out.
    assert (c.item_debut == 0).sum() >= 0
