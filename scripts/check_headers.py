#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
"""Fail CI if any source file has lost its SPDX or copyright header.

This is layer one of the provenance scheme described in
``src/kokoro/_provenance.py``. It runs on every push so attribution cannot be
dropped by accident — or quietly, by anyone else.

Usage:
    python scripts/check_headers.py            # check, exit 1 on failure
    python scripts/check_headers.py --fix      # insert missing headers
"""

from __future__ import annotations

import sys
from pathlib import Path

SPDX = "# SPDX-License-Identifier: AGPL-3.0-or-later"
COPYRIGHT = "# Copyright (C) 2026 Gaurav Jadhav"
HEADER = f"{SPDX}\n{COPYRIGHT} <gauravmakarandjadhav@gmail.com>\n"

ROOTS = ("src", "tests", "scripts")
SKIP_DIRS = {".venv", "build", "dist", "__pycache__", ".git", "node_modules"}


def iter_sources(root: Path) -> list[Path]:
    """Return every ``.py`` file under the checked roots."""
    files: list[Path] = []
    for r in ROOTS:
        base = root / r
        if not base.exists():
            continue
        files += [
            p
            for p in base.rglob("*.py")
            if not SKIP_DIRS & set(p.parts) and p.stat().st_size > 0
        ]
    return sorted(files)


def check(path: Path) -> str | None:
    """Return a human-readable problem, or ``None`` when the header is intact."""
    head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
    if SPDX not in head:
        return "missing SPDX-License-Identifier"
    if COPYRIGHT.rsplit(" ", 2)[0] not in head:
        return "missing copyright line"
    return None


def fix(path: Path) -> None:
    """Insert the header, preserving any shebang on line one."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("#!"):
        shebang, _, rest = text.partition("\n")
        path.write_text(f"{shebang}\n{HEADER}{rest}", encoding="utf-8")
    else:
        path.write_text(HEADER + text, encoding="utf-8")


def main() -> int:
    """Check every source file, optionally repairing it."""
    repair = "--fix" in sys.argv
    root = Path(__file__).resolve().parent.parent

    failures: list[tuple[Path, str]] = []
    for path in iter_sources(root):
        problem = check(path)
        if problem is None:
            continue
        if repair:
            fix(path)
            print(f"fixed  {path.relative_to(root)}")
        else:
            failures.append((path, problem))

    if failures:
        print(f"\n{len(failures)} file(s) are missing their attribution header:\n")
        for path, problem in failures:
            print(f"  {path.relative_to(root)}: {problem}")
        print("\nRun `python scripts/check_headers.py --fix` to insert them.")
        print("Removing these headers violates NOTICE and AGPL-3.0 §7(b).")
        return 1

    print(f"all {len(iter_sources(root))} source files carry attribution headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
