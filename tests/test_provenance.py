# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Provenance stamping and embedding watermarking."""

from __future__ import annotations

import numpy as np
import pytest

from kokoro import provenance


def test_fingerprint_is_deterministic() -> None:
    assert provenance.fingerprint() == provenance.fingerprint()
    assert len(provenance.fingerprint()) == 32


def test_fingerprint_is_version_scoped_when_asked() -> None:
    assert provenance.fingerprint("1.0") != provenance.fingerprint("2.0")
    assert provenance.fingerprint("1.0") != provenance.fingerprint()


def test_stamp_round_trips_and_does_not_mutate() -> None:
    payload = {"metric": 0.42}
    stamped = provenance.stamp(payload)
    assert provenance.STAMP_KEY not in payload, "stamp() must not mutate its input"
    assert stamped["metric"] == 0.42
    assert provenance.verify(stamped, strict=True)


def test_verify_rejects_unstamped_and_tampered_payloads() -> None:
    assert not provenance.verify({"metric": 0.42})
    tampered = provenance.stamp({})
    tampered[provenance.STAMP_KEY]["fingerprint"] = "0" * 32
    assert provenance.verify(tampered) is True, "non-strict only checks presence"
    assert provenance.verify(tampered, strict=True) is False


def test_write_sidecar(tmp_path) -> None:
    import json
    from pathlib import Path

    artifact = tmp_path / "index.faiss"
    artifact.write_bytes(b"not-a-real-index")
    sidecar = provenance.write_sidecar(str(artifact), {"m": 32})
    body = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    assert body["artifact"] == "index.faiss"
    assert body["m"] == 32
    assert provenance.verify(body, strict=True)


def test_watermark_vector_is_deterministic_and_unit_norm() -> None:
    a = provenance.watermark_vector(256)
    b = provenance.watermark_vector(256)
    np.testing.assert_array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-6)


def test_watermark_salts_are_independent() -> None:
    item = provenance.watermark_vector(256, salt="item")
    query = provenance.watermark_vector(256, salt="query")
    # Two random unit vectors in 256-D are near-orthogonal; anything above 0.2
    # would mean the salt is not actually separating the directions.
    assert abs(float(item @ query)) < 0.2


@pytest.mark.parametrize("dim", [0, -8])
def test_watermark_rejects_bad_dim(dim: int) -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        provenance.watermark_vector(dim)


def test_watermark_is_detectable(rng: np.random.Generator) -> None:
    clean = rng.standard_normal((4000, 256)).astype(np.float32)
    marked = provenance.apply_embedding_watermark(clean)

    _, z_clean = provenance.detect_embedding_watermark(clean)
    _, z_marked = provenance.detect_embedding_watermark(marked)

    assert abs(z_clean) < 5.0, "unwatermarked vectors must not trip the detector"
    assert z_marked > 5.0, "watermarked vectors must be detectable"


def test_watermark_is_retrieval_neutral(rng: np.random.Generator) -> None:
    """The whole design rests on this: the mark must not move the rankings."""
    clean = rng.standard_normal((500, 256)).astype(np.float32)
    clean /= np.linalg.norm(clean, axis=1, keepdims=True)
    marked = provenance.apply_embedding_watermark(clean)

    queries = rng.standard_normal((50, 256)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    top_clean = (queries @ clean.T).argsort(axis=1)[:, ::-1][:, :10]
    top_marked = (queries @ marked.T).argsort(axis=1)[:, ::-1][:, :10]

    overlap = np.mean(
        [
            len(set(a.tolist()) & set(b.tolist())) / 10
            for a, b in zip(top_clean, top_marked, strict=True)
        ]
    )
    assert overlap >= 0.99, f"watermark perturbed top-10 rankings (overlap={overlap:.3f})"


def test_watermark_rejects_bad_strength(rng: np.random.Generator) -> None:
    emb = rng.standard_normal((10, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="strength must be in"):
        provenance.apply_embedding_watermark(emb, strength=1.5)


def test_detect_requires_two_rows(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="at least 2 rows"):
        provenance.detect_embedding_watermark(rng.standard_normal((1, 8)).astype(np.float32))
