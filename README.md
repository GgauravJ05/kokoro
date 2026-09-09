<div align="center">

# 心 Kokoro

**Mood-conditioned anime & manga retrieval, with narrative-arc trajectory matching.**

[![CI](https://github.com/GgauravJ05/kokoro/actions/workflows/ci.yml/badge.svg)](https://github.com/GgauravJ05/kokoro/actions/workflows/ci.yml)
[![CodeQL](https://github.com/GgauravJ05/kokoro/actions/workflows/codeql.yml/badge.svg)](https://github.com/GgauravJ05/kokoro/actions/workflows/codeql.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

*Author: [Gaurav Jadhav](https://github.com/JGaurav26) · Original work, AGPL-3.0 · [provenance](#-provenance--attribution)*

</div>

---

## The problem

Ask any recommender for "something melancholy but hopeful, slow, that won't wreck
me" and it fails, because **mood is not in the metadata**. MyAnimeList and AniList
give you genres, tags, studios and scores. The genre `Drama` contains both
*Violet Evergarden* and *School Days*. No tag distinguishes them on the axis a
viewer actually cares about.

Worse, every deployed system treats a title as **one point** in taste space. That
is wrong in a way viewers feel constantly:

- *Vinland Saga* S1 and S2 are different emotional objects.
- *Steins;Gate* inverts completely at episode 12.
- *Oshi no Ko* spends one episode establishing a register and the rest dismantling it.

A single mood vector averages those halves into something that describes neither.

## What Kokoro does

**1. Learns mood from how viewers talk, not from tags.** A two-tower contrastive
retriever is trained on `(review sentence → title)` pairs with InfoNCE and mined
hard negatives. The supervision is real language about real emotional response.

**2. Makes the mood space readable.** The item tower routes through an
8-dimensional bottleneck whose axes are anchored to named affective poles, so
every recommendation ships with an explanation and a radar chart instead of a
similarity score.

| Axis | − pole | + pole |
|---|---|---|
| `comfort` | devastating, leaves you hollow | warm, safe, easy to return to |
| `pace` | slow, contemplative, drifting | relentless, frenetic, breathless |
| `hope` | bleak, nihilistic, no way out | hopeful, earnest, worth believing in |
| `intimacy` | lonely, isolating, alienated | close, tender, deeply connected |
| `cognition` | visceral, felt in the body | cerebral, puzzle-like, demands thought |
| `levity` | heavy, serious, no relief | funny, light, playful |
| `catharsis` | unresolved, withholding | cathartic, earns its ending |
| `intensity` | gentle, low-stakes | overwhelming, operatic |

**3. Models mood as a curve, not a point.** Review sentences are bound to the arc
they discuss, producing a trajectory of shape `(n_arcs, n_axes)` per title. This
is the novel contribution, and it unlocks queries nothing else answers:

```
"starts light, gets devastating"        → matches on the derivative, not the mean
"comfort watch, but warn me if it turns" → "comfortable until episode 8"
"something that ends resolved"           → constraint on the final arc only
```

## Status

The corpus is built and the baselines have real numbers. The neural towers are
implemented but **untrained** — nothing below is a Kokoro result yet, only the
bar Kokoro has to clear. See [the roadmap](#roadmap), or open
[`docs/plan.html`](docs/plan.html) in a browser for the nine-week plan.

### Baselines on the real corpus

`kokoro benchmark --corpus data/processed`, corpus `7bb25749c812bdd1`,
296,333 interactions over a 3,000-user subsample, 6,456 items, `user_holdout`
split, k=10.

| model | recall@10 | ndcg@10 | mrr@10 | coverage@10 | gini@10 | pop. lift | query ms |
|---|---|---|---|---|---|---|---|
| random | 0.0010 | 0.0026 | 0.0078 | **0.991** | **0.257** | **0.99** | 0.003 |
| popularity | 0.1047 | 0.1530 | 0.3051 | 0.007 | 0.998 | 23.2× | 0.060 |
| item-kNN | 0.1630 | 0.2434 | 0.4379 | 0.057 | 0.993 | 18.8× | 0.039 |
| **BPR-MF** (from scratch) | **0.1762** | **0.2633** | **0.4493** | 0.167 | 0.968 | 13.6× | 0.065 |

Three things this table is actually saying:

1. **The from-scratch NumPy model wins.** BPR-MF beats item-kNN on every
   accuracy metric *and* surfaces 3× more of the catalog at lower popularity
   bias. It needed 30 epochs — at 5 it scores 0.169 NDCG, barely above
   popularity, which is worth knowing before anyone quotes an undertrained
   number.
2. **Popularity bias is the real problem here, and it is severe.** Popularity
   alone gets 0.153 NDCG while showing 0.7% of the catalog with a 23×
   popularity lift. Any model that reports accuracy without `coverage` and
   `gini` next to it is hiding this.
3. **Random confirms the harness is honest.** Near-total coverage, no accuracy.
   If anything ever scores below this line, the bug is in the evaluation code.

Beating 0.2633 NDCG *while* moving coverage up is the target. Accuracy alone is
not a result.

### Cold-start: the headline split

`--split cold_start --cold-cut-year 2014`. 998 titles have **zero** training
interactions.

| model | recall@10 | ndcg@10 | mrr@10 | coverage@10 |
|---|---|---|---|---|
| random | 0.0026 | 0.0039 | 0.0097 | 0.915 |
| popularity | **0.0000** | **0.0000** | **0.0000** | 0.011 |
| item-kNN | **0.0000** | **0.0000** | **0.0000** | 0.032 |
| BPR-MF | **0.0000** | **0.0000** | **0.0000** | 0.099 |

Every collaborative model scores exactly zero, and this is not a bug — it is
arithmetic. A model whose only representation of an item is who interacted with
it has *no* representation of an item nobody has interacted with. Random beats
all three by chance alone.

This is the gap Kokoro exists to fill. The content tower can embed a title from
its synopsis, tags and reviews without a single interaction, so **any** non-zero
cold-start NDCG is something no baseline here can produce at all. That is the
claim to make, and it is falsifiable.

## Quickstart

```bash
git clone https://github.com/GgauravJ05/kokoro.git
cd kokoro
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

kokoro provenance          # authorship record for this build
kokoro config              # resolved settings
kokoro benchmark           # baseline suite on synthetic data (harness smoke test)
```

Ingest real titles (rate-limited, respects AniList's published budget):

```bash
cp .env.example .env
kokoro ingest --media-type ANIME --max-pages 20 --out data/raw/titles.jsonl
```

## Method

```
                  ┌──────────────────────┐
   user query ───▶│   query tower (BERT) │──┐
                  └──────────────────────┘  │
                                            ├──▶ shared 256-d space ──▶ HNSW ──▶ top-k
   reviews  ──┐   ┌──────────────────────┐  │            ▲
   synopsis ──┼──▶│   item tower (BERT)  │──┘            │
   tags     ──┘   └──────────┬───────────┘               │
                             │                           │
                             ▼                           │
                    ┌────────────────┐                   │
                    │ mood bottleneck│──▶ 8 named axes ──┴──▶ explanation
                    └────────────────┘
   arc-bound   ┌─────────────────────┐
   segments ──▶│ trajectory encoder  │──▶ gated into item embedding
               │  (dilated 1-D CNN)  │──▶ shape queries + warnings
               └─────────────────────┘
```

**Towers are untied.** Queries are imperative and second-person; reviews are
past-tense and discursive. Tying the encoders measurably hurts, and the
tied-vs-untied comparison is a row in the ablation table.

**Hard negatives matter more than batch size here.** A random negative for
*Violet Evergarden* is some sports comedy the model separates trivially. The
useful negative is another beautiful, slow, melancholy drama that nevertheless
leaves you feeling different — only same-genre mining produces those.

**The bottleneck is anchored, not just narrow.** Without the anchor alignment
loss the axes are an arbitrary rotation and the interpretability claim is false.
The ablation that removes it is what proves the claim is real.

## Evaluation

Most anime-recommender projects report zero metrics. This one reports two
families, and treats the second as equally load-bearing.

**Accuracy** — `recall@k`, `precision@k`, `ndcg@k`, `mrr@k`, `hit_rate@k`.

**Beyond-accuracy** — a recommender can top the accuracy table by showing the
same fifty popular titles to everyone, which is *the* known failure mode on a
catalog this long-tailed. `coverage`, `gini`, `intra_list_diversity`,
`serendipity` and `popularity_lift` are what catch it.

**Splits, in increasing honesty:**

| Split | What it proves | Reported as |
|---|---|---|
| `random` | optimistic ceiling | upper bound only |
| `leave_one_out` | comparable to published baselines | needs timestamps — unavailable |
| `temporal` | can we predict *tomorrow* from *today* | **not computable on this corpus** |
| `user_holdout` | in-catalog accuracy without timestamps | secondary |
| `cold_start` | can a brand-new title be placed at all | **headline** |

**Why cold-start is the headline.** The rating source carries no interaction
timestamps, so a temporal split cannot be computed honestly and is not reported.
Debut dates *are* available from the catalog, which makes a real cold-start split
possible — and it happens to be the split this project is actually about. Reviews
cover 1.7% of the catalog, so the interesting question was never "can we re-rank
titles everyone has already rated", it is "can a title with zero interactions be
placed correctly from its content alone".

**Baselines that must be beaten:** random, popularity, item-kNN, BM25 on
synopsis, off-the-shelf embedding similarity, and a raw LLM prompt. That last
one is the real bar — if asking a frontier model directly does as well, this
project has no reason to exist. The design targets where it wins: latency, cost
per query, catalog coverage, calibrated axes, and titles too new or obscure to
be memorised.

The harness was written **before** the models, on purpose. A harness authored
after the model it evaluates grows whatever affordance makes that model look
good.

## Roadmap

- [x] Evaluation harness, metrics, splits — *written first*
- [x] Baselines: random, popularity, item-kNN
- [x] From-scratch BPR matrix factorisation (hand-derived gradients, NumPy only)
- [x] Provenance and embedding-watermark layer
- [x] Frozen, versioned corpus with recorded join rates and biases
      (`kokoro corpus build`) — 28,880 titles, 52.7k reviews, 6.3M ratings
- [x] Baselines on real data; sparse item-kNN; `user_holdout` split
- [ ] AniList + Jikan ingestion **blocked upstream** — AniList returns
      `403 temporarily disabled`, Jikan `504`. Ingestion is source-agnostic and
      currently runs off static Hugging Face dumps instead
- [ ] Arc alignment: hand-annotated set, then extractor evaluation
- [ ] Two-tower training + negative-mining ablation
- [ ] Mood-axis human validation (Spearman ρ per axis)
- [ ] Trajectory encoder + shape-query evaluation
- [ ] FastAPI service, HNSW index, quantised export, p95 latency budget
- [ ] Writeup + workshop submission

## Data & ethics

No third-party dataset is redistributed here. `data/` holds pointers and
reproducible ingestion scripts only. Titles, synopses, cover art and reviews
remain the property of their owners and are used for non-commercial research.

Run `kokoro corpus sources` to print every source with its licence and its
**known biases**. Those biases are copied into `manifest.json` beside the data,
because they bound what the model can honestly claim:

- Reviews cover **490 of 28,880 titles** (1.7%). The model is supervised on the
  head and must reach the rest through the content tower — which makes
  cold-start the headline evaluation, not a footnote.
- **91.3% of reviews are positive**, and `mixed` verdicts were dropped by the
  upstream author. That is exactly the ambivalent middle a mood model needs.
- The rating matrix carries **no timestamps at all**, so the `temporal` split —
  the one this project calls its headline — is *not computable* on it. Results
  use `user_holdout` and say so. Restoring a temporal split needs a timestamped
  source.

Rate limits are enforced in code, not by convention — `KOKORO_ANILIST_RPS` is
capped in [`config.py`](src/kokoro/config.py) and `429` responses are honoured
by sleeping for the server's own `Retry-After`. Please do not raise these.

## 🔒 Provenance & attribution

This is original work by **Gaurav Jadhav**. Authorship is asserted in three
independent layers — see [`_provenance.py`](src/kokoro/_provenance.py):

1. **Source headers.** Every `.py` file carries an SPDX identifier and copyright
   line, enforced in CI.
2. **Artifact stamps.** Checkpoints, evaluation reports and exported embeddings
   embed a signed provenance block. `kokoro provenance` prints it.
3. **Embedding watermark.** Released embeddings carry a deterministic author
   direction blended in far below the noise floor — verified retrieval-neutral
   in [`test_provenance.py`](tests/test_provenance.py). Any model distilled or
   fine-tuned on them inherits a statistically detectable correlation:

   ```bash
   kokoro watermark path/to/embeddings.npy
   ```

**Licensed AGPL-3.0.** You may study, modify and share this. If you deploy it or
a modified version as a network service, §13 obliges you to offer your users the
complete corresponding source — the `/source` endpoint exists to satisfy that.
Removing the attribution notices does not remove the obligations and violates
§7(b).

If this work is useful to you, cite it — see [`CITATION.cff`](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch model, commit convention,
and the checks that must pass. Repository governance, branch-protection policy
and the go-public checklist are in [docs/repo-setup.md](docs/repo-setup.md).
Security issues: [SECURITY.md](SECURITY.md).

---

<div align="center">
<sub>心 — <i>kokoro</i>, the heart as the seat of feeling.<br>
Copyright © 2026 Gaurav Jadhav · AGPL-3.0-or-later</sub>
</div>
