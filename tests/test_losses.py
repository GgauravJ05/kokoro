# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav
"""Contrastive objectives."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="optional [train] extra not installed")

from kokoro.losses.infonce import HardNegativeInfoNCELoss, InfoNCELoss


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(1337)


def test_infonce_is_lower_for_aligned_pairs() -> None:
    emb = torch.randn(16, 32)
    aligned = InfoNCELoss()(emb, emb.clone())
    shuffled = InfoNCELoss()(emb, emb[torch.randperm(16)])
    assert aligned < shuffled, "perfectly aligned pairs must score better than shuffled ones"


def test_infonce_approaches_zero_for_perfect_separation() -> None:
    # Orthogonal rows: each query matches exactly one item and nothing else.
    emb = torch.eye(8) * 10.0
    assert InfoNCELoss(temperature=0.05)(emb, emb.clone()).item() < 1e-4


def test_infonce_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        InfoNCELoss()(torch.randn(4, 8), torch.randn(5, 8))


def test_infonce_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        InfoNCELoss(temperature=0.0)


def test_symmetric_and_asymmetric_both_produce_finite_gradients() -> None:
    for symmetric in (True, False):
        q = torch.randn(8, 16, requires_grad=True)
        InfoNCELoss(symmetric=symmetric)(q, torch.randn(8, 16)).backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()


def test_learnable_temperature_is_a_parameter() -> None:
    assert any(p.requires_grad for p in InfoNCELoss(learnable_temperature=True).parameters())
    assert not list(InfoNCELoss(learnable_temperature=False).parameters())


def test_hard_negatives_penalise_a_close_negative() -> None:
    query = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    positive = query.clone()
    far = torch.nn.functional.normalize(torch.randn(4, 3, 32), dim=-1)
    # A negative nearly identical to the query is the hardest possible case.
    near = query.unsqueeze(1).repeat(1, 3, 1) + 0.01 * torch.randn(4, 3, 32)

    loss = HardNegativeInfoNCELoss(use_in_batch=False)
    assert loss(query, positive, near) > loss(query, positive, far)


def test_hard_negative_loss_rejects_bad_shapes() -> None:
    q = torch.randn(4, 8)
    with pytest.raises(ValueError, match=r"negatives must be \(B, N, D\)"):
        HardNegativeInfoNCELoss()(q, q, torch.randn(4, 8))
    with pytest.raises(ValueError, match="share a batch dimension"):
        HardNegativeInfoNCELoss()(q, torch.randn(5, 8), torch.randn(4, 2, 8))
