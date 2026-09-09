## What and why

<!-- What changes, and what problem it solves. Link the issue if there is one. -->

Closes #

## Type

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `exp` — experiment
- [ ] `docs` / `chore`

## Evidence

<!-- Required for anything touching a model, a metric, or the pipeline.
     Paste the before/after from `kokoro benchmark` on the SAME split and seed. -->

| | recall@10 | ndcg@10 | coverage@10 | gini@10 | query_ms |
|---|---|---|---|---|---|
| before | | | | | |
| after | | | | | |

Split: `temporal` / seed: `1337`

## Checklist

- [ ] `make check` passes
- [ ] Tests added that **fail without this change**
- [ ] Docstrings with `Args` / `Returns` / `Raises`
- [ ] SPDX header on every new file
- [ ] No evaluation semantics changed — or, if they were, called out in bold above
- [ ] No new dependency — or, if there is one, justified here
- [ ] Rate limits unchanged
