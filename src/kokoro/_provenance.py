# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
# Canonical source: https://github.com/GgauravJ05/kokoro
"""Authorship provenance for every artifact this project produces.

Kokoro stamps its own outputs. A checkpoint, an evaluation report, an exported
embedding matrix or an API response all carry a block identifying who wrote the
code that produced them, which commit produced them, and under what license they
may be redistributed.

Three independent layers, in increasing order of tamper-resistance:

1. **Source headers.** Every ``.py`` file starts with an SPDX identifier and a
   copyright line. Enforced in CI by ``scripts/check_headers.py``.
2. **Artifact stamps.** :func:`stamp` embeds a JSON provenance block into any
   serialisable payload; :func:`verify` checks it later. Easy to strip, but
   stripping it is a deliberate, documented act rather than an accident.
3. **Embedding watermark.** :func:`watermark_vector` derives a deterministic
   unit vector from the author fingerprint. Blending it into released
   embeddings at a magnitude far below the noise floor (see
   :func:`apply_embedding_watermark`) leaves retrieval quality intact while
   making provenance statistically detectable in any derived model — a
   downstream copy that distils or fine-tunes on these vectors inherits a
   correlation that :func:`detect_embedding_watermark` can recover.

None of this replaces the licence. It makes the licence *enforceable*: AGPL-3.0
section 7(b) lets the author require that these attribution notices survive
redistribution, and these layers are how a violation becomes provable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import numpy.typing as npt

# --------------------------------------------------------------------------------------
# Canonical identity. Changing these values changes the fingerprint by design.
# --------------------------------------------------------------------------------------

AUTHOR: Final[str] = "Gaurav Jadhav"
AUTHOR_EMAIL: Final[str] = "gauravmakarandjadhav@gmail.com"
AUTHOR_GITHUB: Final[str] = "JGaurav26"
PROJECT: Final[str] = "kokoro"
REPOSITORY: Final[str] = "https://github.com/GgauravJ05/kokoro"
LICENSE: Final[str] = "AGPL-3.0-or-later"
COPYRIGHT: Final[str] = f"Copyright (C) 2026 {AUTHOR} <{AUTHOR_EMAIL}>"

#: Bytes hashed to produce the provenance fingerprint. Stable across releases so
#: that artifacts from different versions still trace to the same author.
_IDENTITY: Final[str] = "|".join((PROJECT, AUTHOR, AUTHOR_EMAIL, AUTHOR_GITHUB, REPOSITORY))

#: Domain-separation tag, so the fingerprint can never collide with a hash
#: computed for some other purpose over the same identity string.
_DOMAIN: Final[bytes] = b"kokoro/provenance/v1"


@dataclass(frozen=True, slots=True)
class Provenance:
    """An immutable description of who and what produced an artifact."""

    project: str
    author: str
    author_email: str
    author_github: str
    repository: str
    license: str
    copyright: str
    version: str
    fingerprint: str
    git_commit: str | None
    git_dirty: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serialisable mapping."""
        return {
            "project": self.project,
            "author": self.author,
            "author_email": self.author_email,
            "author_github": self.author_github,
            "repository": self.repository,
            "license": self.license,
            "copyright": self.copyright,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "created_at": self.created_at,
        }

    def banner(self) -> str:
        """Return a short human-readable attribution line."""
        commit = (self.git_commit or "unknown")[:8]
        dirty = "+dirty" if self.git_dirty else ""
        return (
            f"{self.project} v{self.version} ({commit}{dirty}) — {self.copyright}\n"
            f"Licensed {self.license}. Source: {self.repository}\n"
            f"Provenance fingerprint: {self.fingerprint}"
        )


def fingerprint(version: str | None = None) -> str:
    """Return the deterministic author fingerprint.

    The value depends only on the canonical identity constants in this module,
    never on the machine, clock or environment, so two people running the same
    commit get the same fingerprint and can compare artifacts.

    Args:
        version: Optional version to bind into the hash. When ``None`` the
            fingerprint is version-independent, which is what artifact stamps
            use so that provenance survives version bumps.

    Returns:
        A 32-character lowercase hex digest (128 bits of SHA-256).
    """
    payload = _IDENTITY if version is None else f"{_IDENTITY}|{version}"
    digest = hashlib.sha256(_DOMAIN + b"\x00" + payload.encode("utf-8")).hexdigest()
    return digest[:32]


def _git(*args: str) -> str | None:
    """Run a git command in the package directory, returning ``None`` on failure."""
    try:
        # S603/S607: the argument list is fixed and never interpolated from
        # user input, and resolving git on PATH is the portable behaviour.
        out = subprocess.run(  # noqa: S603
            ("git", *args),  # noqa: S607
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover — no git binary
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def current(version: str | None = None) -> Provenance:
    """Build a :class:`Provenance` record for the running code."""
    if version is None:
        from kokoro import __version__

        version = __version__
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return Provenance(
        project=PROJECT,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        author_github=AUTHOR_GITHUB,
        repository=REPOSITORY,
        license=LICENSE,
        copyright=COPYRIGHT,
        version=version,
        fingerprint=fingerprint(),
        git_commit=commit,
        git_dirty=bool(status),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


#: Key under which :func:`stamp` writes its block.
STAMP_KEY: Final[str] = "_kokoro_provenance"


def stamp(payload: dict[str, Any], *, version: str | None = None) -> dict[str, Any]:
    """Return ``payload`` with a provenance block attached.

    The input is not mutated. Call this on every checkpoint metadata dict,
    every evaluation report and every exported artifact.

    Args:
        payload: Any JSON-serialisable mapping.
        version: Override the recorded version; defaults to the installed one.

    Returns:
        A shallow copy of ``payload`` carrying :data:`STAMP_KEY`.
    """
    return {**payload, STAMP_KEY: current(version).to_dict()}


def verify(payload: dict[str, Any], *, strict: bool = False) -> bool:
    """Check that ``payload`` carries an intact provenance stamp.

    Args:
        payload: A mapping previously passed through :func:`stamp`.
        strict: When true, also require that the recorded fingerprint matches
            this build's fingerprint — i.e. that the artifact came from *this*
            author's code and not a rebranded fork.

    Returns:
        ``True`` when the stamp is present and (if ``strict``) matches.
    """
    block = payload.get(STAMP_KEY)
    if not isinstance(block, dict):
        return False
    if block.get("fingerprint") is None:
        return False
    return block["fingerprint"] == fingerprint() if strict else True


def write_sidecar(path: str, payload: dict[str, Any] | None = None) -> str:
    """Write a ``<path>.provenance.json`` sidecar next to a binary artifact.

    Used for formats that cannot carry a JSON block themselves (``.faiss``
    indexes, raw ``.npy`` matrices).

    Args:
        path: The artifact the sidecar describes.
        payload: Extra fields to record alongside the provenance block.

    Returns:
        The sidecar path that was written.
    """
    target = Path(f"{path}.provenance.json")
    body = stamp(payload or {})
    body["artifact"] = Path(path).name
    target.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(target)


# --------------------------------------------------------------------------------------
# Embedding watermark
# --------------------------------------------------------------------------------------

#: Default blend magnitude, chosen by sweeping detectability against retrieval
#: damage. At 0.01 the mark scores z > 9 over 4k vectors while preserving more
#: than 99% of top-10 rankings; at 0.002 it is retrieval-perfect but only
#: reaches z ≈ 1.7, which is not evidence of anything. Both claims are
#: regression-tested in ``tests/test_provenance.py``.
DEFAULT_WATERMARK_STRENGTH: Final[float] = 0.01


def watermark_vector(dim: int, *, salt: str = "") -> npt.NDArray[np.float32]:
    """Derive a deterministic unit vector from the author fingerprint.

    The vector is pseudo-random but reproducible: the same ``dim`` and ``salt``
    always yield the same direction, and no one without the identity constants
    in this module can regenerate it.

    Args:
        dim: Dimensionality of the embedding space.
        salt: Optional domain separator, e.g. ``"item"`` vs ``"query"``, so the
            two towers carry independent watermarks.

    Returns:
        A unit-norm ``float32`` vector of shape ``(dim,)``.

    Raises:
        ValueError: If ``dim`` is not positive.
    """
    import numpy as np

    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    seed_bytes = hashlib.sha256(_DOMAIN + b"\x01" + f"{_IDENTITY}|{salt}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def apply_embedding_watermark(
    embeddings: npt.NDArray[np.float32],
    *,
    strength: float = DEFAULT_WATERMARK_STRENGTH,
    salt: str = "",
) -> npt.NDArray[np.float32]:
    """Blend the author watermark into an embedding matrix and re-normalise.

    Args:
        embeddings: Array of shape ``(n, dim)``. Not mutated.
        strength: Blend magnitude in ``[0, 1)``. See
            :data:`DEFAULT_WATERMARK_STRENGTH` for why the default is safe.
        salt: Passed through to :func:`watermark_vector`.

    Returns:
        A new ``float32`` array of the same shape, row-normalised.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or ``strength`` is out of range.
    """
    import numpy as np

    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2-D (n, dim) array, got shape {embeddings.shape}")
    if not 0.0 <= strength < 1.0:
        raise ValueError(f"strength must be in [0, 1), got {strength}")

    mark = watermark_vector(embeddings.shape[1], salt=salt)
    out = embeddings.astype(np.float32, copy=True)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    out += strength * mark
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def detect_embedding_watermark(
    embeddings: npt.NDArray[np.float32], *, salt: str = ""
) -> tuple[float, float]:
    """Test whether an embedding matrix carries this author's watermark.

    Projects every row onto the watermark direction and runs a one-sample
    z-test against the null hypothesis that the projections are centred at
    zero — which is what unwatermarked vectors, isotropic in expectation,
    give you.

    Args:
        embeddings: Array of shape ``(n, dim)``.
        salt: Must match the salt used at watermarking time.

    Returns:
        ``(mean_projection, z_score)``. A ``z_score`` above ~5 over a few
        thousand rows is strong evidence the vectors derive from this project.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or has fewer than two rows.
    """
    import numpy as np

    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2-D (n, dim) array, got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 rows to compute a z-score, got {n}")

    mark = watermark_vector(embeddings.shape[1], salt=salt)
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    proj = unit @ mark
    mean = float(proj.mean())
    std = float(proj.std(ddof=1))
    z = 0.0 if std < 1e-12 else mean / (std / np.sqrt(n))
    return mean, float(z)
