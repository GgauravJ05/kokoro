# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Typed, environment-driven configuration.

Every knob is settable three ways, in increasing precedence: a default here, a
``KOKORO_*`` environment variable, or an explicit keyword argument. The ``.env``
file in the repository root is read automatically when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cpu", "cuda", "mps"]


class Settings(BaseSettings):
    """Runtime settings for ingestion, training and serving."""

    model_config = SettingsConfigDict(
        env_prefix="KOKORO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Data sources ---------------------------------------------------------
    anilist_endpoint: str = "https://graphql.anilist.co"
    jikan_endpoint: str = "https://api.jikan.moe/v4"

    #: Requests per second against each public API. These are deliberately
    #: conservative: AniList publishes a 90 req/min budget and Jikan a 60
    #: req/min one, and this project must stay a well-behaved client.
    anilist_rps: float = Field(default=1.0, gt=0, le=1.5)
    jikan_rps: float = Field(default=0.5, gt=0, le=1.0)

    user_agent: str = "kokoro-research/0.1 (+https://github.com/GgauravJ05/kokoro)"
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=5, ge=0, le=10)

    # --- Paths ----------------------------------------------------------------
    data_dir: Path = Path("./data")
    artifact_dir: Path = Path("./artifacts")

    # --- Training -------------------------------------------------------------
    device: Device = "auto"
    seed: int = 1337

    @field_validator("data_dir", "artifact_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        """Expand ``~`` and resolve to an absolute path."""
        return v.expanduser().resolve()

    @property
    def raw_dir(self) -> Path:
        """Directory for untouched API payloads."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Directory for partially cleaned intermediates."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Directory for the frozen, versioned training corpus."""
        return self.data_dir / "processed"

    def resolve_device(self) -> str:
        """Turn ``device="auto"`` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def ensure_dirs(self) -> None:
        """Create the data and artifact directories if they do not exist."""
        for d in (self.raw_dir, self.interim_dir, self.processed_dir, self.artifact_dir):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings(**overrides: object) -> Settings:
    """Return the process-wide settings singleton.

    Args:
        **overrides: When given, build a fresh (uncached) ``Settings`` with
            these values applied on top of the environment. Useful in tests.

    Returns:
        The shared :class:`Settings` instance, or a one-off when overridden.
    """
    global _settings  # noqa: PLW0603
    if overrides:
        return Settings(**overrides)  # type: ignore[arg-type]
    if _settings is None:
        _settings = Settings()
    return _settings
