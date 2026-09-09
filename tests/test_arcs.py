# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Arc mention extraction from review text."""

from __future__ import annotations

import pytest

from kokoro.features.arcs import extract_arc_mentions, segment_review, to_arc_index


def test_extracts_an_explicit_episode_number() -> None:
    mentions = extract_arc_mentions("It completely changes after episode 12.", total_episodes=24)
    assert mentions
    assert mentions[0].episode == 12
    assert mentions[0].position == pytest.approx(0.5)


def test_position_is_none_without_a_known_runtime() -> None:
    mentions = extract_arc_mentions("Watch until ep. 8.")
    assert mentions[0].episode == 8
    assert mentions[0].position is None


def test_extracts_ordinal_regions() -> None:
    mentions = extract_arc_mentions("The final act is devastating.")
    assert any(m.position == 1.0 for m in mentions)


def test_returns_nothing_for_positionless_text() -> None:
    assert extract_arc_mentions("A beautiful, quietly sad show about grief.") == []


def test_more_specific_patterns_come_first() -> None:
    mentions = extract_arc_mentions("Episodes 3-7 drag, but the ending lands.", total_episodes=12)
    assert mentions[0].kind == "episode_range"
    assert mentions[0].confidence > mentions[-1].confidence


def test_segment_review_merges_short_sentences() -> None:
    body = "Short. Also short. " + "A considerably longer sentence about the tone. " * 2
    segments = segment_review(body, min_chars=40)
    assert segments
    assert all(len(s) >= 20 for s in segments)
    assert segments[0].startswith("Short. Also short."), "short sentences must merge forward"
    assert sum(len(s) for s in segments) >= len(body.strip()) - len(segments), (
        "no text may be dropped"
    )


def test_segment_review_handles_text_below_the_threshold() -> None:
    assert segment_review("Tiny.", min_chars=40) == ["Tiny."]


@pytest.mark.parametrize(
    ("position", "n_arcs", "expected"),
    [(0.0, 4, 0), (0.5, 4, 2), (0.99, 4, 3), (1.0, 4, 3), (0.3, 1, 0)],
)
def test_to_arc_index(position: float, n_arcs: int, expected: int) -> None:
    assert to_arc_index(position, n_arcs) == expected


def test_to_arc_index_validates_inputs() -> None:
    with pytest.raises(ValueError, match="position must be in"):
        to_arc_index(1.5, 4)
    with pytest.raises(ValueError, match="n_arcs must be"):
        to_arc_index(0.5, 0)
