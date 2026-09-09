# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Item/query text rendering and the content retriever.

No encoder is downloaded here. :func:`kokoro.models.content.encode_texts` is the
only part that needs one, and it is a thin wrapper; everything with logic worth
testing takes embeddings as an argument.
"""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.eval.splits import Interactions
from kokoro.features.text import TITLE_MASK, item_text, mask_title_mentions, query_text
from kokoro.models.content import ContentRetriever

# ---------------------------------------------------------------------------
# Title masking — the leak that would invalidate every training number
# ---------------------------------------------------------------------------


def test_masks_the_works_own_title() -> None:
    text = "Steins;Gate is the best thriller I have seen."
    out = mask_title_mentions(text, "Steins;Gate")
    assert "Steins;Gate" not in out
    assert TITLE_MASK in out


def test_masking_is_case_insensitive() -> None:
    assert "frieren" not in mask_title_mentions("frieren made me cry", "Frieren").lower()


def test_masks_a_distinctive_subtitle() -> None:
    out = mask_title_mentions(
        "Brotherhood fixes everything the original got wrong",
        "Fullmetal Alchemist: Brotherhood",
    )
    assert "Brotherhood" not in out


def test_masking_leaves_unrelated_prose_intact() -> None:
    """Over-masking would shred the very sentences carrying the mood signal."""
    text = "The pacing is slow and the ending is quietly devastating."
    assert mask_title_mentions(text, "Steins;Gate") == text


def test_common_words_are_never_masked() -> None:
    """A title like 'Monster' must not eat every sentence containing 'the'."""
    text = "the movie was a slow burn in the first season"
    out = mask_title_mentions(text, "The", "a", "season", "movie")
    assert out == text


def test_masking_handles_missing_titles() -> None:
    assert mask_title_mentions("unchanged text", None) == "unchanged text"


def test_query_text_normalises_whitespace_and_truncates() -> None:
    out = query_text("  lots   of\n\nspace  ", None)
    assert out == "lots of space"

    long = " ".join(["word"] * 400)
    truncated = query_text(long, None, max_chars=50)
    assert len(truncated) <= 50
    assert not truncated.endswith("wor"), "truncation must fall on a word boundary"


# ---------------------------------------------------------------------------
# Item text
# ---------------------------------------------------------------------------


def test_item_text_leads_with_tags() -> None:
    """Tags are the strongest mood signal available, so they come first."""
    out = item_text(
        title="Cowboy Bebop",
        genres=["Action", "Sci-Fi"],
        tags=["Space", "Tragedy", "Found Family"],
        media_type="TV",
        episodes=26,
        year=1998,
    )
    assert out.startswith("Cowboy Bebop")
    assert out.index("Tragedy") < out.index("Action"), "tags must precede genres"
    assert "26 episodes" in out
    assert "from 1998" in out


@pytest.mark.parametrize("bad", [float("nan"), None, 0, -1])
def test_item_text_survives_missing_numerics(bad: object) -> None:
    """Real rows carry NaN, and `if nan:` is true — which is how int(nan) happens."""
    out = item_text(title="X", episodes=bad, year=bad)  # type: ignore[arg-type]
    assert out == "X."


def test_item_text_ignores_placeholder_strings() -> None:
    out = item_text(title="X", media_type="nan", demographic="None")
    assert "nan" not in out.lower()
    assert "aimed at" not in out.lower()


def test_item_text_singularises_one_episode() -> None:
    assert "1 episode." in item_text(title="A Movie", episodes=1) + "."


def test_item_text_does_not_repeat_a_theme_already_in_tags() -> None:
    out = item_text(title="X", tags=["Psychological"], themes=["Psychological", "Military"])
    assert out.count("Psychological") == 1
    assert "Military" in out


def test_item_text_accepts_numpy_arrays() -> None:
    """Parquet round-trips list columns as ndarrays, not lists."""
    out = item_text(title="X", genres=np.array(["Drama", "Sci-Fi"]), tags=np.array([]))
    assert "Drama" in out and "Sci-Fi" in out


# ---------------------------------------------------------------------------
# ContentRetriever
# ---------------------------------------------------------------------------


@pytest.fixture
def toy() -> tuple[Interactions, np.ndarray]:
    """Four items in two clean clusters, and a user who likes cluster A."""
    embeddings = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]], dtype=np.float32)
    data = Interactions(
        user=np.array([0, 0, 1, 1], np.int64),
        item=np.array([0, 1, 2, 3], np.int64),
        rating=np.array([9.0, 10.0, 9.0, 10.0], np.float32),
        timestamp=np.zeros(4, np.int64),
    )
    return data, embeddings


def test_content_ranks_by_taste_cluster(toy) -> None:
    data, embeddings = toy
    model = ContentRetriever(embeddings, min_rating=7.0).fit(data)
    ranked = model.recommend(np.array([0, 1], np.int64), k=2, exclude_seen=False)
    assert set(ranked[0].tolist()) == {0, 1}, "user 0 likes cluster A"
    assert set(ranked[1].tolist()) == {2, 3}, "user 1 likes cluster B"


def test_content_scores_items_with_no_interactions(toy) -> None:
    """The whole point: an item absent from training is still rankable."""
    data, embeddings = toy
    train = data.take(np.array([0, 2], np.int64))  # items 1 and 3 never seen
    model = ContentRetriever(embeddings, min_rating=7.0).fit(train)
    ranked = model.recommend(np.array([0], np.int64), k=2, exclude_seen=True)
    assert 1 in ranked[0].tolist(), "the unseen near-neighbour must still be reachable"


def test_content_requires_fit_first(toy) -> None:
    _, embeddings = toy
    with pytest.raises(RuntimeError, match="call fit"):
        ContentRetriever(embeddings).recommend(np.array([0], np.int64))


def test_content_rejects_non_2d_embeddings() -> None:
    with pytest.raises(ValueError, match=r"expected \(n_items, dim\)"):
        ContentRetriever(np.zeros(8, np.float32))


def test_content_rejects_k_above_catalog(toy) -> None:
    data, embeddings = toy
    model = ContentRetriever(embeddings).fit(data)
    with pytest.raises(ValueError, match="exceeds catalog size"):
        model.recommend(np.array([0], np.int64), k=99)


def test_low_rated_items_do_not_shape_a_profile(toy) -> None:
    """A 3/10 is evidence of exposure, not of taste."""
    _, embeddings = toy
    data = Interactions(
        user=np.array([0, 0], np.int64),
        item=np.array([0, 2], np.int64),
        rating=np.array([10.0, 3.0], np.float32),
        timestamp=np.zeros(2, np.int64),
    )
    model = ContentRetriever(embeddings, min_rating=7.0).fit(data)
    assert model.profiles is not None
    ranked = model.recommend(np.array([0], np.int64), k=4, exclude_seen=False)
    assert ranked[0][0] in (0, 1), "the disliked cluster must not lead the ranking"


def test_users_with_no_liked_items_get_a_zero_profile(toy) -> None:
    _, embeddings = toy
    data = Interactions(
        user=np.array([0], np.int64),
        item=np.array([0], np.int64),
        rating=np.array([2.0], np.float32),
        timestamp=np.zeros(1, np.int64),
    )
    model = ContentRetriever(embeddings, min_rating=7.0).fit(data)
    ranked = model.recommend(np.array([0], np.int64), k=2, exclude_seen=False)
    assert ranked.shape == (1, 2), "a profileless user must still yield a valid ranking"


def test_encode_texts_rejects_empty_input() -> None:
    from kokoro.models.content import encode_texts

    with pytest.raises(ValueError, match="empty list"):
        encode_texts([])
