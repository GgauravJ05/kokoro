# Contributing to Kokoro

Thanks for your interest. This is a research project with a single maintainer,
so please read the two sections that are unusual — [licensing](#licensing-and-attribution)
and [what gets rejected](#what-will-be-rejected) — before opening a PR.

## Licensing and attribution

Kokoro is **AGPL-3.0-or-later**. By contributing you agree that:

1. Your contribution is licensed under AGPL-3.0-or-later.
2. You retain copyright in your own contribution; **Gaurav Jadhav** retains
   copyright in the existing work.
3. You will not remove, alter or obscure the SPDX headers, the `NOTICE` file,
   `CITATION.cff`, or anything in `src/kokoro/_provenance.py`. These are
   attribution notices protected under §7(b) of the licence, and CI enforces
   them.

Every new `.py` file must begin with:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
```

`python scripts/check_headers.py --fix` inserts these for you.

## Setup

```bash
git clone https://github.com/GgauravJ05/kokoro.git
cd kokoro
make install      # venv + all extras + pre-commit hooks
make check        # lint, types, headers, tests — the exact CI gate
```

Requires Python 3.10–3.13.

## Branch model

`main` is protected and always releasable. Work on a branch, open a PR, let CI
pass, get a review.

| Prefix | For |
|---|---|
| `feat/` | new capability |
| `fix/` | bug fix |
| `exp/` | an experiment that may not merge |
| `docs/` | documentation only |
| `chore/` | tooling, CI, dependencies |

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat(models): add trajectory gating`, `fix(eval): correct NDCG at k > list length`.

## The bar for a PR

Non-negotiable:

- [ ] `make check` passes locally
- [ ] New code has tests. Not "a test exists" — a test that **fails without the
      change**
- [ ] Public functions have Google-style docstrings with `Args`/`Returns`/`Raises`
- [ ] Type annotations everywhere; `mypy --strict` is not optional
- [ ] The SPDX header is present

For anything touching a model or a metric:

- [ ] **A number, from `kokoro benchmark`.** A claim that something improves
      retrieval needs the before and after on the same split with the same seed
- [ ] If it changes evaluation semantics, say so loudly in the PR description.
      Silently making the numbers better is the one unforgivable thing here

## What will be rejected

- **Metric changes bundled with model changes.** They must be separate PRs, or
  the result is uninterpretable.
- **Evaluating on the random split and reporting it as the headline.** The
  temporal split is the headline. See `src/kokoro/eval/splits.py` for why.
- **New dependencies without justification.** Especially anything that pulls a
  second deep-learning framework in.
- **Scraping changes that raise the rate limits.** `KOKORO_ANILIST_RPS` is
  capped in code deliberately. A PR that raises it will be closed.
- **Removing or weakening the provenance layer.**
- **LLM-generated code you have not read.** Use whatever tools you like, but you
  are answerable for every line.

## Reporting bugs

Use the issue templates. For a model or metric bug, include the seed, the split,
and the `artifacts/benchmark.json` that reproduces it — otherwise it cannot be
triaged.

## Security

Do not open a public issue. See [SECURITY.md](SECURITY.md).
