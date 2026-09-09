# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""The two-tower retriever: free-text mood queries in, titles out.

A *query tower* encodes what someone types ("something to watch after a
breakup, not too heavy"). An *item tower* encodes a title from its reviews,
synopsis and tags. Both land in one shared space, so retrieval is a nearest
neighbour lookup and the whole catalog can be indexed offline — the property
that makes this servable at tens of milliseconds where a cross-encoder or an
LLM call is not.

The towers do **not** share weights. Queries and reviews are different registers
(imperative and second-person versus past-tense and discursive), and tying the
encoders measurably hurts; the untied-versus-tied comparison is a row in the
ablation table.

Optionally the item tower routes through :class:`~kokoro.models.mood_axes.
MoodBottleneck`, and optionally its output is fused with a trajectory embedding
from :class:`~kokoro.models.trajectory.TrajectoryEncoder`. Both are switchable
precisely so their contribution can be measured rather than asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from kokoro.models.mood_axes import MoodBottleneck

if TYPE_CHECKING:  # pragma: no cover
    from kokoro.models.trajectory import TrajectoryEncoder

__all__ = ["TextEncoder", "TwoTowerRetriever"]

Pooling = Literal["mean", "cls"]


class TextEncoder(nn.Module):
    """A transformer encoder plus a projection into the shared space.

    Args:
        model_name: Any HuggingFace encoder checkpoint. The default is a strong
            small retrieval encoder; the ablation sweeps it against larger ones
            to show where the accuracy-per-millisecond curve bends.
        output_dim: Width of the shared retrieval space.
        pooling: ``"mean"`` masked-average pooling, or ``"cls"``.
        freeze_backbone: Train only the projection head. Useful as a cheap
            baseline and as the first stage of a two-stage schedule.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        output_dim: int = 256,
        pooling: Pooling = "mean",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.project = nn.Linear(self.backbone.config.hidden_size, output_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a batch of tokenised text.

        Args:
            input_ids: Shape ``(B, L)``.
            attention_mask: Shape ``(B, L)``, 1 for real tokens.

        Returns:
            L2-normalised embeddings, shape ``(B, output_dim)``.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return nn.functional.normalize(self.project(pooled), dim=-1)


class TwoTowerRetriever(nn.Module):
    """Untied query and item towers sharing one embedding space.

    Args:
        query_encoder: Tower for user queries.
        item_encoder: Tower for titles.
        bottleneck: Optional mood bottleneck applied to the item tower. Pass
            ``None`` to run the no-bottleneck ablation.
        trajectory_encoder: Optional trajectory tower. When given, its output is
            gated into the item embedding by a learned scalar, so the model can
            decide how much narrative shape matters rather than having it
            imposed.

    Raises:
        ValueError: If the two towers project to different widths.
    """

    def __init__(
        self,
        query_encoder: TextEncoder,
        item_encoder: TextEncoder,
        bottleneck: MoodBottleneck | None = None,
        trajectory_encoder: TrajectoryEncoder | None = None,
    ) -> None:
        super().__init__()
        if query_encoder.project.out_features != item_encoder.project.out_features:
            raise ValueError(
                "towers must project to the same width: "
                f"{query_encoder.project.out_features} vs {item_encoder.project.out_features}"
            )
        self.query_encoder = query_encoder
        self.item_encoder = item_encoder
        self.bottleneck = bottleneck
        self.trajectory_encoder = trajectory_encoder
        # Starts at zero so training begins as a pure text model and only
        # admits trajectory signal if it earns loss reduction.
        self.trajectory_gate = nn.Parameter(torch.zeros(1))

    def encode_query(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Embed user queries, shape ``(B, D)``."""
        out: Tensor = self.query_encoder(input_ids, attention_mask)
        return out

    def encode_item(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        trajectory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Embed titles.

        Args:
            input_ids: Shape ``(B, L)``.
            attention_mask: Shape ``(B, L)``.
            trajectory: Optional mood trajectories, shape ``(B, T, A)``.

        Returns:
            ``(embedding, axis_values)``. ``axis_values`` is ``None`` when no
            bottleneck is configured.
        """
        emb: Tensor = self.item_encoder(input_ids, attention_mask)
        axes: Tensor | None = None
        if self.bottleneck is not None:
            emb, axes = self.bottleneck(emb)
        if trajectory is not None and self.trajectory_encoder is not None:
            emb = emb + torch.tanh(self.trajectory_gate) * self.trajectory_encoder(trajectory)
        return nn.functional.normalize(emb, dim=-1), axes

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        item_ids: Tensor,
        item_mask: Tensor,
        trajectory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Encode an aligned batch of (query, item) pairs.

        Returns:
            ``(query_embedding, item_embedding, axis_values)``, ready to hand
            to :class:`~kokoro.losses.infonce.InfoNCELoss`.
        """
        q = self.encode_query(query_ids, query_mask)
        it, axes = self.encode_item(item_ids, item_mask, trajectory)
        return q, it, axes
