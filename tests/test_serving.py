# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""The serving path: engine loading, the API contract, and its probes.

Tests that need trained artifacts skip when none are present, so CI stays green
on a fresh clone. The contract tests — what the API promises when the model is
*not* loaded — run everywhere, because that is the state a misconfigured
deployment is actually in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="optional [serve] extra not installed")
pytest.importorskip("torch", reason="optional [train] extra not installed")

from fastapi.testclient import TestClient

from kokoro.serve.app import app

ARTIFACTS = Path("artifacts/two_tower")
CORPUS = Path("data/processed")
HAS_MODEL = (ARTIFACTS / "query_tower.pt").exists() and (CORPUS / "titles.parquet").exists()
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="no trained artifacts on this machine")


@pytest.fixture
def client() -> TestClient:
    """A test client over the real app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Contract that holds with or without a model
# ---------------------------------------------------------------------------


def test_health_never_loads_the_model(client: TestClient) -> None:
    """Liveness must stay cheap: a probe that loads a model is a probe that
    times out and gets the pod killed."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_source_endpoint_satisfies_agpl_section_13(client: TestClient) -> None:
    body = client.get("/source").json()
    assert "github.com" in body["repository"]
    assert "AGPL" in body["license"]


def test_every_response_carries_provenance_headers(client: TestClient) -> None:
    from kokoro import provenance

    headers = client.get("/health").headers
    assert headers["X-Kokoro-Provenance"] == provenance.fingerprint()
    assert headers["X-Kokoro-License"] == provenance.LICENSE
    assert "github.com" in headers["X-Kokoro-Source"]


def test_query_validation_rejects_bad_input(client: TestClient) -> None:
    assert client.get("/recommend", params={"q": "ab"}).status_code == 422
    assert client.get("/recommend", params={"q": "valid query", "k": 0}).status_code == 422
    assert client.get("/recommend", params={"q": "valid query", "k": 999}).status_code == 422


def test_missing_artifacts_yield_503_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """A service with no model is unavailable, not broken. 503 tells a load
    balancer to route elsewhere; 500 reads as a bug and pages someone."""
    import kokoro.serve.app as app_module

    monkeypatch.setattr(app_module, "_engine", None)
    monkeypatch.setenv("KOKORO_ARTIFACT_DIR", "/nonexistent/artifacts")

    response = TestClient(app, raise_server_exceptions=False).get(
        "/recommend", params={"q": "a query with enough characters"}
    )
    assert response.status_code == 503
    assert "kokoro train" in response.json()["detail"]


# ---------------------------------------------------------------------------
# With a trained model
# ---------------------------------------------------------------------------


@needs_model
def test_engine_returns_ranked_hits() -> None:
    from kokoro.serve.engine import Engine

    engine = Engine(ARTIFACTS, CORPUS, index="exact")
    hits = engine.search("a funny lighthearted show to relax with", k=5)

    assert len(hits) == 5
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "hits must be ordered best-first"
    assert all(h.title for h in hits)
    assert len({h.anime_id for h in hits}) == 5, "no duplicate titles in one ranking"


@needs_model
def test_engine_rejects_empty_and_bad_k() -> None:
    from kokoro.serve.engine import Engine

    engine = Engine(ARTIFACTS, CORPUS, index="exact")
    with pytest.raises(ValueError, match="must not be empty"):
        engine.embed_query("   ")
    with pytest.raises(ValueError, match="k must be"):
        engine.search("a valid query", k=0)


@needs_model
def test_hnsw_agrees_with_exact_search() -> None:
    """A fast index that returns different neighbours is not a speedup."""
    from kokoro.serve.engine import Engine

    exact = Engine(ARTIFACTS, CORPUS, index="exact")
    approx = Engine(ARTIFACTS, CORPUS, index="hnsw", ef_search=64)

    query = "dark psychological thriller that messes with your head"
    a = [h.anime_id for h in exact.search(query, k=10)]
    b = [h.anime_id for h in approx.search(query, k=10)]
    overlap = len(set(a) & set(b)) / 10
    assert overlap >= 0.9, f"HNSW recall@10 was {overlap:.2f} against exact search"


@needs_model
def test_unknown_index_is_rejected() -> None:
    from kokoro.serve.engine import Engine

    with pytest.raises(ValueError, match="unknown index"):
        Engine(ARTIFACTS, CORPUS, index="quantum")


@needs_model
def test_engine_reports_a_consistent_catalog() -> None:
    from kokoro.serve.engine import Engine

    engine = Engine(ARTIFACTS, CORPUS, index="exact")
    stats = engine.stats()
    assert stats["catalog_size"] == engine.vectors.shape[0]
    assert stats["catalog_size"] == engine.item_ids.shape[0], (
        "one id per vector, or results map to the wrong titles"
    )


@needs_model
def test_ready_actually_warms_the_encoder(client: TestClient) -> None:
    """Regression test: /ready once returned 200 while the first real query
    still paid a multi-second model load. A readiness probe that lies is worse
    than none, because traffic gets routed to it."""
    import kokoro.serve.app as app_module

    app_module._engine = None
    assert client.get("/ready").json()["status"] == "ready"

    first = client.get("/recommend", params={"q": "a comfort watch for a bad day", "k": 5}).json()
    assert first["latency_ms"] < 1000, (
        f"first query after /ready took {first['latency_ms']}ms — the probe did not warm"
    )


@needs_model
def test_recommend_returns_the_documented_shape(client: TestClient) -> None:
    client.get("/ready")
    body = client.get("/recommend", params={"q": "epic space opera", "k": 7}).json()

    assert body["query"] == "epic space opera"
    assert len(body["results"]) == 7
    assert body["catalog_size"] > 0
    assert body["latency_ms"] >= 0
    first = body["results"][0]
    assert {"title_id", "romaji", "score"} <= set(first)


def test_demo_page_is_served_and_self_contained(client: TestClient) -> None:
    """The page must render from the same origin with no external JS host, or it
    breaks behind any CSP a deployment target imposes."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "<title>Kokoro" in body
    assert "/recommend" in body, "the page must call the API on the same origin"
    assert "<script src=" not in body, "no external script hosts"


def test_demo_page_ships_inside_the_package() -> None:
    """Packaged as data, so `pip install kokoro-retrieval` can serve the demo."""
    from kokoro.serve import app as app_module

    assert (app_module._STATIC / "index.html").is_file()


@needs_model
def test_popularity_floor_raises_the_audience_of_results() -> None:
    """A serving-side presentation filter, deliberately absent from evaluation."""
    from kokoro.serve.engine import Engine

    engine = Engine(ARTIFACTS, CORPUS, index="exact")
    query = "comfort watch for a bad day"

    unfiltered = engine.search(query, k=10)
    filtered = engine.search(query, k=10, min_members=50_000)

    assert all(h.members >= 50_000 for h in filtered), "the floor must actually hold"
    assert len(filtered) == 10, "over-fetching must still return a full page"

    median = lambda hits: sorted(h.members for h in hits)[len(hits) // 2]  # noqa: E731
    assert median(filtered) > median(unfiltered)


@needs_model
def test_popularity_floor_still_returns_ranked_results() -> None:
    from kokoro.serve.engine import Engine

    engine = Engine(ARTIFACTS, CORPUS, index="exact")
    hits = engine.search("epic space opera", k=5, min_members=100_000)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({h.anime_id for h in hits}) == len(hits)


@needs_model
def test_an_impossible_floor_degrades_gracefully() -> None:
    """No catalog title has 50M members; the engine must not crash or hang."""
    from kokoro.serve.engine import Engine

    engine = Engine(ARTIFACTS, CORPUS, index="exact")
    hits = engine.search("anything at all", k=5, min_members=50_000_000)
    assert isinstance(hits, list), "an unsatisfiable filter falls back, it does not raise"


@needs_model
def test_recommend_exposes_min_members(client: TestClient) -> None:
    client.get("/ready")
    body = client.get(
        "/recommend", params={"q": "a comfort watch", "k": 5, "min_members": 100_000}
    ).json()
    assert all(r["members"] >= 100_000 for r in body["results"])
