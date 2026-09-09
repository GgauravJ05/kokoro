# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Metrics, checked against hand-computed values rather than themselves."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro.eval import metrics


def test_recall_hand_computed() -> None:
    # Relevant {1, 5, 9}; top-3 contains only item 1 -> 1/3.
    assert metrics.recall_at_k([[1, 2, 3, 4, 5]], [{1, 5, 9}], k=3) == pytest.approx(1 / 3)


def test_precision_hand_computed() -> None:
    # Two of the top 4 are relevant.
    assert metrics.precision_at_k([[1, 2, 3, 4]], [{1, 3}], k=4) == pytest.approx(0.5)


def test_mrr_uses_first_hit() -> None:
    assert metrics.mrr_at_k([[7, 3, 1]], [{3, 1}], k=3) == pytest.approx(0.5)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert metrics.ndcg_at_k([[1, 2, 3]], [{1, 2, 3}], k=3) == pytest.approx(1.0)


def test_ndcg_hand_computed() -> None:
    # Relevant item sits at rank 2 -> DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    expected = (1 / np.log2(3)) / 1.0
    assert metrics.ndcg_at_k([[9, 1, 8]], [{1}], k=3) == pytest.approx(expected)


def test_ndcg_respects_graded_gains() -> None:
    graded = metrics.ndcg_at_k([[1, 2]], [{1, 2}], k=2, gains={1: 3.0, 2: 1.0})
    inverted = metrics.ndcg_at_k([[2, 1]], [{1, 2}], k=2, gains={1: 3.0, 2: 1.0})
    assert graded == pytest.approx(1.0)
    assert inverted < graded, "putting the low-gain item first must score worse"


def test_queries_without_ground_truth_are_skipped_not_zeroed() -> None:
    scored = metrics.recall_at_k([[1, 2], [3, 4]], [{1}, set()], k=2)
    assert scored == pytest.approx(1.0), "an empty-truth query must not drag the mean to 0.5"


def test_coverage_and_gini_on_a_degenerate_recommender() -> None:
    # Everyone gets the same two items out of a catalog of 100.
    ranked = np.tile(np.array([0, 1]), (50, 1))
    assert metrics.catalog_coverage(ranked, n_items=100, k=2) == pytest.approx(0.02)
    assert metrics.gini(ranked, n_items=100, k=2) > 0.95


def test_gini_is_zero_for_uniform_exposure() -> None:
    ranked = np.array([[0, 1, 2, 3]])
    assert metrics.gini(ranked, n_items=4, k=4) == pytest.approx(0.0, abs=0.3)


def test_popularity_lift_detects_head_bias() -> None:
    popularity = np.array([100.0, 100.0, 1.0, 1.0])
    head = np.array([[0, 1]])
    tail = np.array([[2, 3]])
    assert metrics.popularity_lift(head, popularity, k=2) > 1.0
    assert metrics.popularity_lift(tail, popularity, k=2) < 1.0


def test_intra_list_diversity_ranges() -> None:
    identical = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    orthogonal = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert metrics.intra_list_diversity([[0, 1, 2]], identical, k=3) == pytest.approx(0.0, abs=1e-6)
    assert metrics.intra_list_diversity([[0, 1, 2]], orthogonal, k=3) > 0.9


def test_serendipity_discounts_obvious_hits() -> None:
    ranked = [[1, 2]]
    relevant = [{1, 2}]
    # Baseline already surfaced item 1, so only item 2 counts as serendipitous.
    assert metrics.serendipity(ranked, relevant, [[1, 99]], k=2) == pytest.approx(0.5)
    assert metrics.serendipity(ranked, relevant, [[97, 99]], k=2) == pytest.approx(1.0)


def test_serendipity_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="query count mismatch"):
        metrics.serendipity([[1, 2], [3, 4]], [{1}, {3}], [[1, 2]], k=2)


def test_evaluate_includes_optionals_only_when_given() -> None:
    base = metrics.evaluate([[1, 2, 3]], [{1}], n_items=10, k=3)
    assert "intra_list_diversity@3" not in base
    assert "serendipity@3" not in base
    assert base["n_queries_scored"] == 1.0

    full = metrics.evaluate(
        [[1, 2, 3]],
        [{1}],
        n_items=10,
        k=3,
        embeddings=np.eye(10),
        popularity=np.ones(10),
        baseline_ranked=[[4, 5, 6]],
    )
    assert {"intra_list_diversity@3", "popularity_lift@3", "serendipity@3"} <= full.keys()


@pytest.mark.parametrize("n_items", [0, -1])
def test_coverage_rejects_bad_catalog_size(n_items: int) -> None:
    with pytest.raises(ValueError, match="n_items must be positive"):
        metrics.catalog_coverage([[1]], n_items=n_items, k=1)


def test_candidate_share_counts_eligible_slots() -> None:
    """On a cold-start split this is what popularity lift was being misread as."""
    assert metrics.candidate_share([[1, 2, 3, 4]], candidates=[1, 2], k=4) == pytest.approx(0.5)
    assert metrics.candidate_share([[1, 2]], candidates=[1, 2], k=2) == pytest.approx(1.0)
    assert metrics.candidate_share([[9, 8]], candidates=[1, 2], k=2) == pytest.approx(0.0)


def test_candidate_share_with_no_candidates_is_zero() -> None:
    assert metrics.candidate_share([[1, 2]], candidates=[], k=2) == 0.0


def test_candidate_share_averages_over_queries() -> None:
    ranked = [[1, 9], [1, 2]]
    assert metrics.candidate_share(ranked, candidates=[1, 2], k=2) == pytest.approx(0.75)


def test_evaluate_adds_candidate_share_only_when_asked() -> None:
    base = metrics.evaluate([[1, 2, 3]], [{1}], n_items=10, k=3)
    assert "candidate_share@3" not in base
    with_c = metrics.evaluate([[1, 2, 3]], [{1}], n_items=10, k=3, candidates=[1, 2])
    assert with_c["candidate_share@3"] == pytest.approx(2 / 3)


def test_popularity_lift_is_degenerate_when_truth_has_zero_popularity() -> None:
    """The trap candidate_share exists to expose: on a cold-start split every
    correct answer has zero training popularity, so a HIGH lift means the model
    spent its slots on warm items that could not possibly be right."""
    popularity = np.array([100.0, 100.0, 0.0, 0.0])  # items 2,3 are cold
    warm_ranking = np.array([[0, 1]])
    cold_ranking = np.array([[2, 3]])

    assert metrics.popularity_lift(warm_ranking, popularity, k=2) > 1.0
    assert metrics.popularity_lift(cold_ranking, popularity, k=2) == 0.0
    # candidate_share says the same thing without the misreading.
    assert metrics.candidate_share(warm_ranking, candidates=[2, 3], k=2) == 0.0
    assert metrics.candidate_share(cold_ranking, candidates=[2, 3], k=2) == 1.0
