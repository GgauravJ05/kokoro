# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Contrastive training of the two projection heads.

The backbone stays frozen and both towers are reduced to a learned linear map
over precomputed sentence embeddings. That is a deliberate first configuration,
not a shortcut:

* It isolates the contribution of the **objective** from the contribution of the
  encoder. The off-the-shelf baseline is exactly this model with identity
  projections, so any difference is attributable to the contrastive training and
  nothing else.
* Embeddings are computed once, so an epoch is a few seconds of matrix
  multiplication and the whole ablation table is runnable on a laptop. Unfreezing
  the backbone is the next configuration, and it needs a GPU to be worth doing.

The item tower sees **metadata text only** — the same string a title released
tomorrow would have — which is what keeps the cold-start evaluation meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from kokoro.data.pairs import iter_batches
from kokoro.losses.infonce import InfoNCELoss
from kokoro.models.mood_axes import MOOD_AXES, MoodBottleneck, anchor_alignment_loss

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from kokoro.data.pairs import PairSet

__all__ = [
    "MoodItemTower",
    "ProjectionTower",
    "TrainConfig",
    "TrainResult",
    "train_projections",
]


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for one training run.

    Attributes:
        output_dim: Width of the shared retrieval space.
        epochs: Passes over the pair set.
        batch_size: Distinct titles per batch. Doubles as the in-batch negative
            count, so it is a modelling parameter here, not just a memory knob.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        temperature: InfoNCE temperature.
        symmetric: Average the query->item and item->query losses.
        dropout: Dropout on the projection input.
        seed: Seed for initialisation and batching.
        patience: Stop after this many epochs without validation improvement.
            ``0`` disables early stopping.
        use_bottleneck: Route the item tower through the named mood axes. This
            is the interpretability configuration; switching it off is the
            ablation that says what interpretability cost.
        n_axes: Bottleneck width when ``use_bottleneck`` is set.
        anchor_weight: Weight of the anchor alignment loss. Without it the axes
            are an arbitrary rotation and the axis *names* mean nothing, so the
            ablation at ``0.0`` is what proves the naming is real.
    """

    output_dim: int = 256
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.05
    symmetric: bool = True
    dropout: float = 0.1
    seed: int = 1337
    patience: int = 5
    use_bottleneck: bool = False
    n_axes: int = 8
    anchor_weight: float = 0.2


class ProjectionTower(nn.Module):
    """A learned linear map from frozen encoder space into the shared space.

    Args:
        input_dim: Width of the precomputed embeddings.
        output_dim: Width of the shared space.
        dropout: Dropout applied to the input.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalise, shape ``(B, output_dim)``."""
        out: Tensor = self.net(x)
        return nn.functional.normalize(out, dim=-1)


class MoodItemTower(nn.Module):
    """Item tower whose representation passes through named mood axes.

    The encoder output is compressed to ``n_axes`` interpretable values and then
    expanded back to the retrieval width. Retrieval still happens in the wide
    space, so the ablation can separate "the bottleneck cost us accuracy" from
    "the bottleneck bought us an explanation".

    Args:
        input_dim: Width of the frozen encoder output.
        output_dim: Width of the shared retrieval space.
        n_axes: Number of mood axes.
        dropout: Dropout applied to the input.
    """

    def __init__(
        self, input_dim: int, output_dim: int, n_axes: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = MoodBottleneck(input_dim, n_axes=n_axes, output_dim=output_dim)

    def axes(self, x: Tensor) -> Tensor:
        """Return the mood-axis values for ``x``, shape ``(B, n_axes)``."""
        return self.bottleneck.axes(self.dropout(x))

    def forward(self, x: Tensor) -> Tensor:
        """Project through the bottleneck and L2-normalise."""
        reconstructed, _ = self.bottleneck(self.dropout(x))
        return nn.functional.normalize(reconstructed, dim=-1)


@dataclass(slots=True)
class TrainResult:
    """Outcome of a training run.

    Attributes:
        query_tower: The trained query projection.
        item_tower: The trained item projection.
        train_loss: Mean InfoNCE loss per epoch.
        val_loss: Mean validation loss per epoch, if a validation set was given.
        val_recall: Validation retrieval recall@1 over held-out titles, which is
            the metric early stopping actually watches — loss can drift while
            ranking quality holds.
        best_epoch: Epoch whose weights were kept.
        config: The configuration used.
    """

    query_tower: ProjectionTower
    item_tower: nn.Module
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_recall: list[float] = field(default_factory=list)
    best_epoch: int = 0
    config: TrainConfig = field(default_factory=TrainConfig)

    @torch.no_grad()
    def axis_values(self, item_embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Return per-item mood axis values, shape ``(n_items, n_axes)``.

        Raises:
            TypeError: If this run did not use a mood bottleneck.
        """
        if not isinstance(self.item_tower, MoodItemTower):
            raise TypeError("this run was trained without a mood bottleneck")
        self.item_tower.eval()
        tensor = torch.as_tensor(item_embeddings, dtype=torch.float32)
        values: npt.NDArray[np.float64] = (
            self.item_tower.axes(tensor).cpu().numpy().astype(np.float64)
        )
        return values

    @torch.no_grad()
    def project_items(self, item_embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Map catalog item embeddings into the trained space.

        Applied to **every** item, cold ones included — the item tower only ever
        saw metadata text, so a title with no interactions projects exactly like
        any other.

        Args:
            item_embeddings: Frozen encoder outputs, shape ``(n_items, dim)``.

        Returns:
            Projected, L2-normalised embeddings of shape ``(n_items, output_dim)``.
        """
        self.item_tower.eval()
        tensor = torch.as_tensor(item_embeddings, dtype=torch.float32)
        projected: npt.NDArray[np.float32] = (
            self.item_tower(tensor).cpu().numpy().astype(np.float32)
        )
        return projected


@torch.no_grad()
def _validation_recall(
    query_tower: ProjectionTower,
    item_tower: nn.Module,
    queries: Tensor,
    items: Tensor,
    targets: Tensor,
) -> float:
    """Fraction of validation queries whose top-1 item is the correct title."""
    query_tower.eval()
    item_tower.eval()
    q = query_tower(queries)
    it = item_tower(items)
    predicted = (q @ it.t()).argmax(dim=1)
    return float((predicted == targets).float().mean())


def _prepare_validation(
    val_pairs: PairSet,
    val_query_embeddings: npt.NDArray[np.float32],
    item_bank: Tensor,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Assemble the validation tensors for :func:`_validation_recall`.

    Validation queries are ranked against **only the held-out titles**, so the
    number reads as "among works it has never been trained on, did it find the
    right one" rather than being diluted by the whole catalog.
    """
    v_q = torch.as_tensor(val_query_embeddings, dtype=torch.float32, device=device)
    v_titles = torch.as_tensor(np.unique(val_pairs.item_pos), dtype=torch.long, device=device)
    remap = {int(t): j for j, t in enumerate(v_titles.tolist())}
    v_targets = torch.as_tensor(
        [remap[int(p)] for p in val_pairs.item_pos], dtype=torch.long, device=device
    )
    return v_q, item_bank[v_titles], v_targets


def _validate_inputs(
    train_pairs: PairSet,
    query_embeddings: npt.NDArray[np.float32],
    item_embeddings: npt.NDArray[np.float32],
    val_pairs: PairSet | None,
    val_query_embeddings: npt.NDArray[np.float32] | None,
    cfg: TrainConfig,
    anchors: tuple[Tensor, Tensor, Tensor] | None,
) -> None:
    """Reject misaligned inputs before a long run starts.

    Raises:
        ValueError: If the embeddings do not line up with the pair set, or if
            validation pairs arrive without their embeddings.
    """
    if query_embeddings.shape[0] != len(train_pairs):
        raise ValueError(
            f"{query_embeddings.shape[0]} query embeddings for {len(train_pairs)} pairs"
        )
    if int(train_pairs.item_pos.max()) >= item_embeddings.shape[0]:
        raise ValueError("train_pairs reference an item position outside item_embeddings")
    if val_pairs is not None and val_query_embeddings is None:
        raise ValueError("val_pairs given without val_query_embeddings")
    # A config that asks for anchor alignment while no anchors are supplied is a
    # contradiction. Training would silently proceed *without* the term and
    # report a different model than the one requested — which is exactly how a
    # dropped argument produced an entire ablation table of identical runs.
    if cfg.use_bottleneck and cfg.anchor_weight > 0 and anchors is None:
        raise ValueError(
            f"anchor_weight={cfg.anchor_weight} requires anchors, but none were given; "
            "pass build_anchor_batch(...) or set anchor_weight=0 to run the ablation"
        )


def build_anchor_batch(encode: object, n_axes: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
    """Encode the axis pole phrases into an anchor batch.

    Each axis contributes two anchors, one per pole. The alignment loss then
    requires the negative-pole phrase to land low on that axis and the
    positive-pole phrase to land high, which is what ties an axis to its name.

    Args:
        encode: Callable mapping ``list[str]`` to an embedding array.
        n_axes: Number of axes in use; anchors beyond it are ignored.
        device: Torch device.

    Returns:
        ``(embeddings, axis_index, pole)`` where ``pole`` is -1.0 or +1.0.
    """
    phrases: list[str] = []
    axis_idx: list[int] = []
    poles: list[float] = []

    for i, (_name, negative, positive) in enumerate(MOOD_AXES[:n_axes]):
        phrases.append(negative)
        axis_idx.append(i)
        poles.append(-1.0)
        phrases.append(positive)
        axis_idx.append(i)
        poles.append(1.0)

    vectors = encode(phrases)  # type: ignore[operator]
    return (
        torch.as_tensor(np.asarray(vectors), dtype=torch.float32, device=device),
        torch.as_tensor(axis_idx, dtype=torch.long, device=device),
        torch.as_tensor(poles, dtype=torch.float32, device=device),
    )


def _run_epoch(
    query_tower: ProjectionTower,
    item_tower: nn.Module,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    train_pairs: PairSet,
    q_all: Tensor,
    i_all: Tensor,
    pos_all: Tensor,
    cfg: TrainConfig,
    epoch: int,
    device: str,
    anchors: tuple[Tensor, Tensor, Tensor] | None = None,
) -> float:
    """Run one training epoch and return its mean loss."""
    query_tower.train()
    item_tower.train()
    losses: list[float] = []

    for batch in iter_batches(train_pairs, batch_size=cfg.batch_size, seed=cfg.seed + epoch):
        idx = torch.as_tensor(batch, dtype=torch.long, device=device)
        q = query_tower(q_all[idx])
        it = item_tower(i_all[pos_all[idx]])

        loss = criterion(q, it)

        # The anchor term is what makes axis names claims rather than labels.
        if anchors is not None and isinstance(item_tower, MoodItemTower):
            anchor_emb, anchor_axis, anchor_pole = anchors
            loss = loss + cfg.anchor_weight * anchor_alignment_loss(
                item_tower.axes(anchor_emb), anchor_axis, anchor_pole
            )

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    return float(np.mean(losses)) if losses else float("nan")


def train_projections(
    train_pairs: PairSet,
    query_embeddings: npt.NDArray[np.float32],
    item_embeddings: npt.NDArray[np.float32],
    *,
    val_pairs: PairSet | None = None,
    val_query_embeddings: npt.NDArray[np.float32] | None = None,
    config: TrainConfig | None = None,
    anchors: tuple[Tensor, Tensor, Tensor] | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> TrainResult:
    """Train both projection heads with InfoNCE over in-batch negatives.

    Args:
        train_pairs: Pairs to train on.
        query_embeddings: Frozen embeddings of ``train_pairs.queries``, aligned
            by index, shape ``(n_pairs, dim)``.
        item_embeddings: Frozen embeddings of every catalog item's metadata
            text, indexed by contiguous item position.
        val_pairs: Optional held-out pairs, split by title.
        val_query_embeddings: Frozen embeddings of ``val_pairs.queries``.
        config: Hyperparameters; defaults to :class:`TrainConfig`.
        anchors: Output of :func:`build_anchor_batch`. Required for the anchor
            alignment term; without it a bottleneck is still a bottleneck but
            its axis names are unjustified.
        device: Torch device.
        verbose: Print per-epoch progress.

    Returns:
        The :class:`TrainResult`, carrying the best-scoring weights.

    Raises:
        ValueError: If the embedding matrices disagree with the pair set, or if
            validation pairs are supplied without their embeddings.
    """
    cfg = config or TrainConfig()

    _validate_inputs(
        train_pairs,
        query_embeddings,
        item_embeddings,
        val_pairs,
        val_query_embeddings,
        cfg,
        anchors,
    )

    torch.manual_seed(cfg.seed)

    q_dim = int(query_embeddings.shape[1])
    i_dim = int(item_embeddings.shape[1])
    query_tower = ProjectionTower(q_dim, cfg.output_dim, cfg.dropout).to(device)
    item_tower: nn.Module = (
        MoodItemTower(i_dim, cfg.output_dim, cfg.n_axes, cfg.dropout).to(device)
        if cfg.use_bottleneck
        else ProjectionTower(i_dim, cfg.output_dim, cfg.dropout).to(device)
    )

    criterion = InfoNCELoss(temperature=cfg.temperature, symmetric=cfg.symmetric).to(device)
    optimiser = torch.optim.AdamW(
        [*query_tower.parameters(), *item_tower.parameters()],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    q_all = torch.as_tensor(query_embeddings, dtype=torch.float32, device=device)
    i_all = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    pos_all = torch.as_tensor(train_pairs.item_pos, dtype=torch.long, device=device)

    val_ready: tuple[Tensor, Tensor, Tensor] | None = None
    if val_pairs is not None and val_query_embeddings is not None:
        v_q = torch.as_tensor(val_query_embeddings, dtype=torch.float32, device=device)
        # Rank each validation query against only the held-out titles, so the
        # number reads as "did it find the right one among unseen works".
        v_titles = torch.as_tensor(np.unique(val_pairs.item_pos), dtype=torch.long, device=device)
        remap = {int(t): j for j, t in enumerate(v_titles.tolist())}
        v_targets = torch.as_tensor(
            [remap[int(p)] for p in val_pairs.item_pos], dtype=torch.long, device=device
        )
        val_ready = (v_q, i_all[v_titles], v_targets)

    result = TrainResult(query_tower=query_tower, item_tower=item_tower, config=cfg)
    best_score = -1.0
    best_state: dict[str, dict[str, Tensor]] | None = None
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        mean_loss = _run_epoch(
            query_tower,
            item_tower,
            criterion,
            optimiser,
            train_pairs,
            q_all,
            i_all,
            pos_all,
            cfg,
            epoch,
            device,
            anchors=anchors,
        )
        result.train_loss.append(mean_loss)

        line = f"epoch {epoch:3d}/{cfg.epochs}  loss={mean_loss:.4f}"
        score = -mean_loss

        if val_ready is not None:
            recall = _validation_recall(query_tower, item_tower, *val_ready)
            result.val_recall.append(recall)
            score = recall
            line += f"  val_recall@1={recall:.4f}"

        if score > best_score:
            best_score = score
            best_state = {
                "query": {k: v.detach().clone() for k, v in query_tower.state_dict().items()},
                "item": {k: v.detach().clone() for k, v in item_tower.state_dict().items()},
            }
            result.best_epoch = epoch
            stale = 0
            line += "  *"
        else:
            stale += 1

        if verbose:
            print(line)

        if cfg.patience and stale >= cfg.patience:
            if verbose:
                print(f"early stop: no improvement for {cfg.patience} epochs")
            break

    if best_state is not None:
        query_tower.load_state_dict(best_state["query"])
        item_tower.load_state_dict(best_state["item"])

    return result
