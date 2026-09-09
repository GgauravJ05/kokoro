# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
# Canonical source: https://github.com/GgauravJ05/kokoro
"""Kokoro — mood-conditioned anime & manga retrieval.

Public surface:

    >>> from kokoro import __version__, provenance
    >>> provenance.fingerprint()[:8]
    ...

Heavy optional dependencies (``torch``, ``hnswlib``, ``fastapi``) are imported
lazily by the submodules that need them, so ``import kokoro`` stays cheap and
works in a metadata-only install.
"""

from __future__ import annotations

from kokoro import _provenance as provenance

__version__ = "0.1.0"
__author__ = provenance.AUTHOR
__license__ = provenance.LICENSE
__copyright__ = provenance.COPYRIGHT

__all__ = ["__author__", "__copyright__", "__license__", "__version__", "provenance"]
