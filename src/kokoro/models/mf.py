# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
r"""Bayesian Personalised Ranking matrix factorisation, written from scratch.

No autograd, no framework: the gradients below are derived by hand from the BPR
objective and applied with plain NumPy. It exists for three reasons.

1. It is the collaborative-filtering half of the hybrid, and a strong one — BPR
   is still competitive on implicit feedback.
2. It is the reference point every neural result in this project is measured
   against. A two-tower model that cannot beat 200 lines of NumPy has not
   earned its complexity.
3. Deriving the update by hand is the part that survives an interview
   whiteboard.

**The objective.** BPR maximises the posterior probability that a user prefers
an observed item ``i`` over an unobserved item ``j``:

.. math::

    \\mathcal{L} = \\sum_{(u,i,j)} \\ln \\sigma(\\hat{x}_{uij})
                   - \\lambda \\lVert \\Theta \\rVert^2

where :math:`\\hat{x}_{uij} = \\hat{x}_{ui} - \\hat{x}_{uj}` and
:math:`\\hat{x}_{ui} = b_i + p_u \\cdot q_i`.

**The gradients.** With :math:`s = \\sigma(-\\hat{x}_{uij})` (the sigmoid of the
*negated* score, which is the factor that vanishes once a triple is already
ranked correctly):

.. math::

    \\partial_{p_u} = s\\,(q_i - q_j) - \\lambda p_u \\\\
    \\partial_{q_i} = s\\,p_u - \\lambda q_i \\\\
    \\partial_{q_j} = -s\\,p_u - \\lambda q_j \\\\
    \\partial_{b_i} = s - \\lambda b_i, \\quad \\partial_{b_j} = -s - \\lambda b_j

Note there is no user bias: it cancels in the pairwise difference, which is
exactly why BPR is a ranking rather than a rating model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kokoro.eval.splits import Interactions

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["BPRMatrixFactorization"]


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Numerically stable logistic sigmoid.

    The naive ``1 / (1 + exp(-x))`` overflows for large negative ``x``, which
    happens constantly in early BPR epochs when scores are unconstrained.
    """
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class BPRMatrixFactorization:
    """Implicit-feedback matrix factorisation trained with BPR.

    Args:
        n_factors: Latent dimensionality.
        n_epochs: Passes over the interaction log.
        lr: SGD learning rate.
        reg: L2 coefficient, applied to every parameter touched in a step.
        init_std: Standard deviation of the Gaussian factor initialisation.
        seed: RNG seed for initialisation, shuffling and negative sampling.
        verbose: Print mean BPR loss per epoch.

    Attributes:
        user_factors: Shape ``(n_users, n_factors)`` after :meth:`fit`.
        item_factors: Shape ``(n_items, n_factors)`` after :meth:`fit`.
        item_bias: Shape ``(n_items,)`` after :meth:`fit`.
        loss_history: Mean loss per epoch, useful for the convergence plot.
    """

    name = "bpr-mf"

    def __init__(
        self,
        n_factors: int = 64,
        n_epochs: int = 30,
        lr: float = 0.05,
        reg: float = 0.01,
        init_std: float = 0.1,
        seed: int = 1337,
        verbose: bool = False,
    ) -> None:
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.init_std = init_std
        self.seed = seed
        self.verbose = verbose

        self.user_factors: npt.NDArray[np.float64] | None = None
        self.item_factors: npt.NDArray[np.float64] | None = None
        self.item_bias: npt.NDArray[np.float64] | None = None
        self.loss_history: list[float] = []
        self._seen: dict[int, set[int]] = {}
        self._n_items = 0

    def fit(self, data: Interactions) -> BPRMatrixFactorization:
        """Train with one uniformly sampled negative per observed interaction.

        Args:
            data: Training interactions. Every row is treated as a positive;
                threshold ratings beforehand if you want explicit feedback
                semantics.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``data`` is empty.
        """
        n = len(data)
        if n == 0:
            raise ValueError("cannot fit on an empty interaction log")

        rng = np.random.default_rng(self.seed)
        n_users, n_items = data.n_users, data.n_items
        self._n_items = n_items

        self.user_factors = rng.normal(0, self.init_std, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, self.init_std, (n_items, self.n_factors))
        self.item_bias = np.zeros(n_items)

        self._seen = {}
        for u, i in zip(data.user.tolist(), data.item.tolist(), strict=True):
            self._seen.setdefault(u, set()).add(i)

        users, items = data.user, data.item
        self.loss_history = []

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0

            for idx in order:
                u = int(users[idx])
                i = int(items[idx])
                j = self._sample_negative(rng, u, n_items)

                pu = self.user_factors[u]
                qi, qj = self.item_factors[i], self.item_factors[j]
                x_uij = float(self.item_bias[i] - self.item_bias[j] + pu @ (qi - qj))

                # sigma(-x): the "how wrong is this triple" weight. Near zero
                # once the pair is confidently ordered, so correctly ranked
                # triples stop contributing — BPR's built-in hard-example focus.
                s = float(_sigmoid(np.array([-x_uij]))[0])
                epoch_loss += -np.log(max(_sigmoid(np.array([x_uij]))[0], 1e-12))

                grad_pu = s * (qi - qj) - self.reg * pu
                grad_qi = s * pu - self.reg * qi
                grad_qj = -s * pu - self.reg * qj

                self.user_factors[u] = pu + self.lr * grad_pu
                self.item_factors[i] = qi + self.lr * grad_qi
                self.item_factors[j] = qj + self.lr * grad_qj
                self.item_bias[i] += self.lr * (s - self.reg * self.item_bias[i])
                self.item_bias[j] += self.lr * (-s - self.reg * self.item_bias[j])

            mean_loss = epoch_loss / n
            self.loss_history.append(float(mean_loss))
            if self.verbose:
                print(f"epoch {epoch + 1:3d}/{self.n_epochs}  bpr_loss={mean_loss:.4f}")

        return self

    def _sample_negative(self, rng: np.random.Generator, user: int, n_items: int) -> int:
        """Sample an item the user has not interacted with.

        Rejection sampling with a bounded retry count: with a sparse matrix a
        hit is overwhelmingly likely on the first draw, and the fallback keeps
        a pathologically dense user from spinning forever.
        """
        seen = self._seen.get(user, set())
        for _ in range(32):
            j = int(rng.integers(n_items))
            if j not in seen:
                return j
        return int(rng.integers(n_items))

    def score(self, users: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Return the full score matrix for ``users``, shape ``(n_users, n_items)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.user_factors is None or self.item_factors is None or self.item_bias is None:
            raise RuntimeError("call fit() before scoring")
        return self.user_factors[users] @ self.item_factors.T + self.item_bias

    def recommend(
        self,
        users: npt.NDArray[np.int64],
        k: int = 10,
        *,
        exclude_seen: bool = True,
    ) -> npt.NDArray[np.int64]:
        """Rank the catalog for each user.

        Args:
            users: User ids, shape ``(n_users,)``.
            k: Items per user.
            exclude_seen: Mask out training interactions before ranking.

        Returns:
            Item ids, shape ``(n_users, k)``, best-first.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``k`` exceeds the catalog size.
        """
        # Order matters: an unfitted model has _n_items == 0, which would
        # otherwise surface as a confusing ValueError about catalog size.
        scores = self.score(users)
        if k > self._n_items:
            raise ValueError(f"k={k} exceeds catalog size {self._n_items}")
        if exclude_seen:
            for row, u in enumerate(users.tolist()):
                seen = self._seen.get(int(u))
                if seen:
                    scores[row, list(seen)] = -np.inf
        # argpartition finds the top k in O(n_items); only those k get sorted.
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(part, ordered, axis=1).astype(np.int64)
