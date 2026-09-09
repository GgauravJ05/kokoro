<div align="center">

# 心 Kokoro

**Mood-conditioned cold-start retrieval for anime.**

[![CI](https://github.com/GgauravJ05/kokoro/actions/workflows/ci.yml/badge.svg)](https://github.com/GgauravJ05/kokoro/actions/workflows/ci.yml)
[![CodeQL](https://github.com/GgauravJ05/kokoro/actions/workflows/codeql.yml/badge.svg)](https://github.com/GgauravJ05/kokoro/actions/workflows/codeql.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

*[Gaurav Jadhav](https://github.com/JGaurav26) · AGPL-3.0 · [paper](paper/kokoro.tex) · [plan](docs/plan.html)*

</div>

---

## The result in one table

Cold start: **1,491 titles with zero training interactions**, ranked under the
standard cold-only protocol.

| model | NDCG@10 | vs. random |
|---|---|---|
| popularity | 0.0032 | 0.19× |
| item-kNN | 0.0032 | 0.19× |
| BPR-MF | 0.0147 | 0.87× |
| random | 0.0170 | 1.00× |
| content, off-the-shelf encoder | 0.0597 | 3.51× |
| **content, contrastively trained** | **0.0845** | **4.97×** |

Under the full-catalog protocol every collaborative model scores **exactly
0.0000**. That is not a tuning failure — a model whose only representation of an
item is *who interacted with it* has no representation of an item nobody has
interacted with. This is the gap Kokoro fills, served at **4.4 ms p95**.

## Why this problem

Genre tags do not express mood. `Drama` contains both *Violet Evergarden* and
*School Days*, and nothing in the metadata separates them on the axis a viewer
actually cares about. But viewers write about exactly that, at length, in
reviews — so mood is learnable from language even though it is absent from the
catalog.

And a seasonal catalog adds titles continuously. A recommender that cannot rank
a show released last week is not solving the problem people have.

## How it works

```
   free text ──▶ query tower ──┐
                               ├──▶ shared 256-d space ──▶ HNSW ──▶ top-k
   metadata  ──▶ item tower  ──┘
                     │
                     └──▶ 8 named mood axes ──▶ explanation
```

A **frozen MiniLM encoder** feeds two learned projections trained with InfoNCE
on `(review segment → title metadata)` pairs. Three decisions carry the result:

- **The item tower never sees reviews.** A cold title has none. An item tower
  trained on review text would have nothing to encode for exactly the titles the
  evaluation is about.
- **Titles are masked out of queries.** A review naming its own show lets the
  model match on the title string instead of on mood — invisible in training
  metrics, fatal at cold start.
- **Each batch holds a title at most once.** With 194 trainable titles, uniform
  sampling repeats them constantly, and InfoNCE treats every off-diagonal entry
  as a negative — so a repeat is a gradient *actively separating* two segments
  about the same work.

Because the backbone is frozen and identical across both content rows above, the
+41% is attributable to the **objective**, not to encoder capacity.

## Two negative results

These are reported because a system evaluated only where it succeeds has not
been evaluated.

### The mood axes work as a mechanism, but support only a weak claim

The item tower can route through eight named bipolar axes with an
anchor-alignment loss. Validation uses AniList tag probes, and **the 21 probe
tags are withheld from the item text during training** — the model must infer
*comfort* without ever reading the word *Tragedy*.

| anchor weight | axis spread σ | mean AUC | inverted axes |
|---|---|---|---|
| 0.0 (ablation) | 0.085 | 0.465 | **2** |
| 20.0 | 0.238 | **0.565** | **0** |

The mechanism triples axis spread, corrects both inverted axes, and costs
nothing measurable in retrieval. But only `levity` (0.686), `comfort` (0.663)
and `hope` (0.630) show signal; `cognition`, `intensity` and `intimacy` show
none; and `pace` and `catharsis` have no tag proxy anywhere in 348 AniList tags,
so they are reported as **untested** rather than assumed.

No axis clears the 0.70 bar this project set. The supportable claim is *"three of
eight axes are weakly but measurably aligned with their names."*

### The trajectory hypothesis is falsified on this data

This project was motivated by the argument that a title is a **curve** over
narrative position — *Steins;Gate* inverts at episode 12, and one mood vector
averages its halves into something describing neither. That argument was tested
and did not survive.

Shuffling each title's mood vectors across its own segments destroys any
position–mood relationship while preserving sample size and bucket sizes:

| statistic | real | shuffled | ratio |
|---|---|---|---|
| within-title σ | 0.0827 | 0.0720 | **1.15×** |
| arc-to-arc step | 0.1075 | 0.0967 | **1.11×** |

A real signal exists — but **~87% of the apparent movement survives shuffling.**
And the shape carries nothing the mean does not:

| features | mean AUC |
|---|---|
| mean only | **0.547** |
| shape only | 0.485 — *chance* |
| mean + shape | 0.513 — *worse than the mean alone* |

**The neural trajectory encoder was deliberately not trained.** Fitting a
higher-capacity model to a signal that fails its permutation control is fitting
noise.

Three constraints bound this, offered as conditions rather than excuses: 140
titles against 40 shape features; three-region position resolution that cannot
express "inverts at episode 12"; and tag prediction possibly being an unfair
target. A fair test needs case-sensitive, episode-level discussion text, which
was not obtainable — AniList returns `403`, MyAnimeList `504`, and the one
promising Reddit dump is access-gated.

## In-catalog retrieval

| model | recall@10 | NDCG@10 | coverage@10 | pop. lift |
|---|---|---|---|---|
| random | 0.0010 | 0.0026 | 0.991 | 0.99× |
| popularity | 0.1047 | 0.1530 | 0.007 | 23.2× |
| item-kNN | 0.1630 | 0.2434 | 0.057 | 18.8× |
| **BPR-MF** (from scratch, NumPy) | **0.1762** | **0.2633** | **0.167** | **13.6×** |

The from-scratch matrix factoriser — hand-derived BPR gradients, no autograd —
wins on accuracy *and* surfaces 2.9× more of the catalog at lower popularity
bias.

## Evaluation discipline

The harness was written **before** any model. One authored afterwards tends to
grow whatever affordance flatters the model it was built for.

**Splits, in increasing honesty:**

| split | what it proves | reported as |
|---|---|---|
| `random` | optimistic ceiling | upper bound only |
| `user_holdout` | in-catalog accuracy without timestamps | secondary |
| `temporal` | predict tomorrow from today | **not computable — no timestamps** |
| `cold_start` | can a brand-new title be placed | **headline** |

**Beyond-accuracy metrics are load-bearing.** A recommender can top the accuracy
table by showing the same fifty popular titles to everyone, which is *the* known
failure mode on a long-tailed catalog. `coverage`, `gini`,
`intra_list_diversity`, `serendipity` and `popularity_lift` catch it.

**One metric needed correcting.** Popularity lift is *uninterpretable* on a
cold-start split: every correct answer there has zero training popularity, so the
oracle lift is `0.00×` and a high lift means the model wasted its ranking on
warm items. An earlier version of this README misread that as a regression.
`candidate_share@k` — the fraction of the top-k that is eligible to be correct
at all — states it directly. Collaborative models sit at **0.000**.

## Quickstart

```bash
git clone https://github.com/GgauravJ05/kokoro.git
cd kokoro
make install                  # venv + extras + pre-commit

make demo                     # http://localhost:8000 — works straight away
```

The trained model and catalog metadata are committed, so the demo runs from a
clone with no data build. To reproduce the corpus and the numbers:

```bash
kokoro corpus sources         # every source with licence and known biases
kokoro corpus build           # 28,880 titles / 52.5k reviews / 6.3M ratings
kokoro train --bottleneck --anchor-weight 20
kokoro benchmark --corpus data/processed --split cold_start --cold-only \
    --content --trained artifacts/two_tower/item_embeddings_trained.npy
```

```bash
$ curl 'localhost:8000/recommend?q=a+funny+lighthearted+show+to+relax+with&k=3'
{"results":[{"romaji":"Danshi Koukousei no Nichijou","score":0.507},
            {"romaji":"Nichijou","score":0.498}, ...],
 "latency_ms":3.2}
```

`make docker` runs the same thing in a container. The image ships code, not
data — the corpus is derived from third-party dumps this project does not
redistribute, so `data/processed` and `artifacts/two_tower` are mounted in.

### A note on the demo

The model carries **no popularity prior**, which is good for catalog coverage
and hard on a demo: unfiltered, 40% of results are titles below 10k members that
a visitor will not recognise. The web UI therefore has a *"well-known titles
only"* toggle that applies an audience floor **at serving time only**. It is
absent from every reported metric — applying it during evaluation would inflate
the numbers by smuggling back exactly the popularity bias the beyond-accuracy
metrics exist to detect.

It is also wrong sometimes. *"comfort watch for a bad day"* returns
*Evangelion 3.0*, which is among the least comforting things ever animated. The
page says so rather than hiding it.

## Reproducing the results

| result | command |
|---|---|
| Baselines, in-catalog | `kokoro benchmark --corpus data/processed` |
| Cold start, both protocols | `kokoro benchmark --split cold_start [--cold-only]` |
| Anchor-weight ablation | `python scripts/anchor_sweep.py` |
| Trajectory test + control | `python scripts/trajectory_experiment.py` |
| Recall/latency sweep | `python scripts/latency_sweep.py` |

The corpus is pinned by content hash (`a11aac2a2c226689`) and every artifact
carries a provenance stamp, so a number can always be traced to the bytes that
produced it.

## Data & ethics

**What ships in this repo.** `data/processed/titles.parquet` (catalog metadata)
and `artifacts/two_tower/` (the trained model) are committed, so a clone runs
`make demo` immediately with no corpus build. The metadata derives solely from
an MIT-licensed catalog and a CC0 tag dump, and contains no review text or user
ratings.

**What does not.** Review text (licence `other`) and the rating matrix (no
declared licence) are never redistributed — `kokoro corpus build` fetches them
from their published locations when you want to retrain or re-evaluate.

`data/` also holds ingestion scripts and a manifest recording each input's
SHA-256, measured join rate, licence and **known biases**. Run `kokoro corpus sources` to print them. The ones that bound the
science:

- Reviews cover **449 of 28,880 titles (1.7%)** — supervision is on the head.
- **91.3% of reviews are positive**, and `mixed` verdicts were dropped upstream.
- Review text is **lowercased**, which is what killed named-arc extraction.
- The rating matrix has **no timestamps**, so no temporal split is reported.

## 🔒 Provenance

Original work by **Gaurav Jadhav**, asserted redundantly so no single deletion
removes it: CI-enforced SPDX headers, artifact stamps, `NOTICE`, `CITATION.cff`,
`X-Kokoro-Provenance` response headers, and a retrieval-neutral **embedding
watermark** (detectable at z > 9 over 4k vectors while preserving >99% of top-10
rankings — both claims regression-tested):

```bash
kokoro provenance
kokoro watermark artifacts/two_tower/item_embeddings_trained.npy
```

**AGPL-3.0.** Deploying this or a modified version as a network service obliges
you to offer users the corresponding source; the `/source` endpoint exists to
satisfy §13. Removing the attribution notices does not remove the obligations
and violates §7(b).

## Project layout

```
src/kokoro/
  data/        ingestion, corpus build, training pairs
  features/    text rendering, arc extraction, trajectories
  models/      baselines, from-scratch BPR, two-tower, content, mood axes
  train/       contrastive training
  eval/        metrics, splits, harness, axis probes
  index/       exact + HNSW
  serve/       engine + FastAPI
paper/         kokoro.tex — built to PDF by CI
docs/          plan.html, repo-setup.md
scripts/       reproducible experiments
```

**223 tests · ruff clean · mypy strict clean · Python 3.10–3.13 · Linux + macOS**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Governance and the go-public checklist
are in [docs/repo-setup.md](docs/repo-setup.md). Security:
[SECURITY.md](SECURITY.md).

---

<div align="center">
<sub>心 — <i>kokoro</i>, the heart as the seat of feeling.<br>
Copyright © 2026 Gaurav Jadhav · AGPL-3.0-or-later</sub>
</div>
