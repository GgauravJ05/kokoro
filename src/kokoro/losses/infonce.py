# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""InfoNCE contrastive objective, with the negative-mining variants ablated.

The training signal for the two-tower retriever: a sentence taken from a real
review of anime *X* should embed near *X* and far from everything else.

**In-batch negatives** (:class:`InfoNCELoss`) treat the other ``B - 1`` items in
the batch as negatives. Cheap, and the effective negative count scales with
batch size — which is why batch size is a *modelling* hyperparameter here, not
just a memory knob, and why it appears in the ablation table.

**Hard negatives** (:class:`HardNegativeInfoNCELoss`) additionally take explicit
negatives mined from the same genre or studio. This is the part that matters for
mood: a random negative for *Violet Evergarden* is some sports comedy, which the
model separates trivially. The useful negative is another beautiful, slow,
melancholy drama that nevertheless leaves you feeling different — and only
explicit mining produces those.

**Bidirectional** (``symmetric=True``) averages the query→item and item→query
directions, the CLIP formulation. It costs one extra softmax and consistently
helps when the two towers do not share weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["HardNegativeInfoNCELoss", "InfoNCELoss"]


class InfoNCELoss(nn.Module):
    r"""InfoNCE with in-batch negatives.

    Args:
        temperature: Softmax temperature :math:`\\tau`. Lower values sharpen the
            distribution and penalise near-misses harder; ``0.05`` is the usual
            starting point for sentence-level retrieval.
        symmetric: Average the query→item and item→query losses.
        learnable_temperature: Optimise :math:`\\log(1/\\tau)` jointly, as CLIP
            does. Removes a hyperparameter at the cost of a small stability
            risk, so the logit scale is clamped at 100.

    Raises:
        ValueError: If ``temperature`` is not positive.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        symmetric: bool = True,
        learnable_temperature: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.symmetric = symmetric
        init_scale = torch.log(torch.tensor(1.0 / temperature))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("logit_scale", init_scale)

    def forward(self, query: Tensor, item: Tensor) -> Tensor:
        """Compute the loss for a batch of aligned pairs.

        Args:
            query: Query embeddings, shape ``(B, D)``.
            item: Item embeddings, shape ``(B, D)``. Row ``i`` is the positive
                for query row ``i``.

        Returns:
            A scalar loss tensor.

        Raises:
            ValueError: If the two tensors disagree in shape.
        """
        if query.shape != item.shape:
            raise ValueError(f"shape mismatch: query {query.shape} vs item {item.shape}")

        q = nn.functional.normalize(query, dim=-1)
        it = nn.functional.normalize(item, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)

        logits = scale * (q @ it.t())  # (B, B)
        targets = torch.arange(q.size(0), device=q.device)

        loss = nn.functional.cross_entropy(logits, targets)
        if self.symmetric:
            loss = 0.5 * (loss + nn.functional.cross_entropy(logits.t(), targets))
        return loss


class HardNegativeInfoNCELoss(nn.Module):
    """InfoNCE over in-batch negatives *plus* explicitly mined hard negatives.

    Args:
        temperature: Softmax temperature.
        use_in_batch: Keep the in-batch negatives alongside the mined ones.
            Turning this off isolates the contribution of mining, which is one
            row of the ablation table.

    Raises:
        ValueError: If ``temperature`` is not positive.
    """

    def __init__(self, temperature: float = 0.05, use_in_batch: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.use_in_batch = use_in_batch

    def forward(self, query: Tensor, positive: Tensor, negatives: Tensor) -> Tensor:
        """Compute the loss.

        Args:
            query: Shape ``(B, D)``.
            positive: Shape ``(B, D)``, aligned with ``query``.
            negatives: Shape ``(B, N, D)`` — ``N`` mined hard negatives per
                query.

        Returns:
            A scalar loss tensor.

        Raises:
            ValueError: If ``negatives`` is not 3-D or the batch dims disagree.
        """
        if negatives.dim() != 3:
            raise ValueError(f"negatives must be (B, N, D), got {tuple(negatives.shape)}")
        if not query.shape[0] == positive.shape[0] == negatives.shape[0]:
            raise ValueError("query, positive and negatives must share a batch dimension")

        q = nn.functional.normalize(query, dim=-1)
        p = nn.functional.normalize(positive, dim=-1)
        n = nn.functional.normalize(negatives, dim=-1)

        pos_logits = (q * p).sum(-1, keepdim=True)  # (B, 1)
        hard_logits = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1)  # (B, N)

        parts = [pos_logits, hard_logits]
        if self.use_in_batch:
            in_batch = q @ p.t()
            # Mask the diagonal: it is the positive, already in pos_logits.
            eye = torch.eye(q.size(0), dtype=torch.bool, device=q.device)
            parts.append(in_batch.masked_fill(eye, float("-inf")))

        logits = torch.cat(parts, dim=1) / self.temperature
        targets = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return nn.functional.cross_entropy(logits, targets)
