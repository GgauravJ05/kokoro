# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""A content-based retriever that can score items nobody has interacted with.

This is the model class that makes the headline split answerable. Every
collaborative baseline in :mod:`kokoro.models.baselines` scores exactly ``0.0``
on cold start, and not because it is bad — a model whose only representation of
an item is *who interacted with it* has no representation at all of an item
nobody has interacted with. :class:`ContentRetriever` represents an item by its
text, so a title released after the training cut is embeddable on day one.

The user side is deliberately the simplest thing that works: a user is the mean
of the item embeddings they rated highly. That keeps the comparison honest —
any cold-start lift is attributable to the item representation, which is what is
being tested, rather than to a learned user model.

Two configurations share this class, and the ablation depends on the difference:

* **Off-the-shelf.** Pass a pretrained sentence encoder and embed the metadata
  as-is. This is the "would a generic embedding model have done just as well?"
  baseline, and it must be beaten before any trained model is interesting.
* **Trained.** Pass embeddings produced by :mod:`kokoro.models.two_tower` after
  contrastive training on review text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kokoro.eval.splits import Interactions

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["ContentRetriever", "encode_texts"]


def encode_texts(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str | None = None,
    show_progress: bool = False,
) -> npt.NDArray[np.float32]:
    """Embed strings with a pretrained sentence encoder.

    Args:
        texts: Strings to embed.
        model_name: Any sentence-transformers checkpoint. The default is small
            enough to run on CPU in minutes, which keeps the baseline
            reproducible without a GPU.
        batch_size: Encoder batch size.
        device: Torch device string; ``None`` lets the library choose.
        show_progress: Display the encoder's progress bar.

    Returns:
        L2-normalised embeddings of shape ``(len(texts), dim)``.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("cannot encode an empty list of texts")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


class ContentRetriever:
    """Rank items by similarity between a user profile and item text embeddings.

    Args:
        embeddings: Item embeddings of shape ``(n_items, dim)``, indexed by
            contiguous item position. Must cover every item in the catalog,
            cold ones included — that is the entire point.
        min_rating: Ratings at or above this contribute to a user's profile.
            Below it, an interaction is evidence of exposure, not of taste.
        weighted: Weight each item's contribution to the profile by how far its
            rating exceeds ``min_rating``.
        name: Reporting label.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """

    def __init__(
        self,
        embeddings: npt.NDArray[np.float32],
        *,
        min_rating: float = 7.0,
        weighted: bool = True,
        name: str = "content",
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"expected (n_items, dim) embeddings, got {embeddings.shape}")
        norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.min_rating = min_rating
        self.weighted = weighted
        self.name = name

        self.profiles: npt.NDArray[np.float32] | None = None
        self._seen: dict[int, set[int]] = {}
        self._n_items = int(embeddings.shape[0])

    def fit(self, data: Interactions) -> ContentRetriever:
        """Build one profile vector per user from their liked items.

        Args:
            data: Training interactions. Items outside the embedding matrix are
                ignored rather than raising, so a corpus can be subsampled
                without breaking the model.

        Returns:
            ``self``.
        """
        dim = self.embeddings.shape[1]
        profiles = np.zeros((data.n_users, dim), dtype=np.float32)

        liked = (data.rating >= self.min_rating) & (data.item < self._n_items)
        users = data.user[liked]
        items = data.item[liked]
        weights = (
            (data.rating[liked] - self.min_rating + 1.0).astype(np.float32)
            if self.weighted
            else np.ones(int(liked.sum()), dtype=np.float32)
        )

        # Scatter-add: one pass over interactions rather than a Python loop per user.
        np.add.at(profiles, users, self.embeddings[items] * weights[:, None])

        norms = np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
        self.profiles = (profiles / norms).astype(np.float32)

        self._seen = {}
        for u, i in zip(data.user.tolist(), data.item.tolist(), strict=True):
            self._seen.setdefault(u, set()).add(i)
        return self

    def recommend(
        self, users: npt.NDArray[np.int64], k: int = 10, *, exclude_seen: bool = True
    ) -> npt.NDArray[np.int64]:
        """Rank the catalog for each user by profile-item cosine similarity.

        Args:
            users: User ids, shape ``(n_users,)``.
            k: Items per user.
            exclude_seen: Mask out the user's training interactions.

        Returns:
            Item ids, shape ``(n_users, k)``, best-first.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``k`` exceeds the catalog size.
        """
        if self.profiles is None:
            raise RuntimeError("call fit() before recommending")
        if k > self._n_items:
            raise ValueError(f"k={k} exceeds catalog size {self._n_items}")

        scores = self.profiles[users] @ self.embeddings.T

        # A user with no liked items has a zero profile and therefore a flat
        # score row; leaving it flat would make argpartition return an arbitrary
        # but *identical* list for every such user, quietly inflating apparent
        # popularity concentration. Push them to the bottom instead.
        cold_users = np.linalg.norm(self.profiles[users], axis=1) < 1e-6
        scores[cold_users] = -np.inf

        if exclude_seen:
            for row, u in enumerate(users.tolist()):
                seen = self._seen.get(int(u))
                if seen:
                    scores[row, list(seen)] = -np.inf

        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(part, ordered, axis=1).astype(np.int64)
