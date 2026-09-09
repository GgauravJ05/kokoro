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

`--split cold_start --cold-cut-year 2014`. 1,491 titles have **zero** training
interactions; 998 of them carry held-out ratings and can actually be scored.

Two protocols, because they answer different questions and only reporting one
is how cold-start results get overstated.

**Cold-only** — rank the new titles against each other. The standard protocol,
and the one that isolates "can this model order titles it has never seen".

| model | recall@10 | ndcg@10 | mrr@10 | vs. random |
|---|---|---|---|---|
| popularity | 0.0016 | 0.0032 | 0.0084 | 0.19× |
| item-kNN | 0.0016 | 0.0032 | 0.0084 | 0.19× |
| BPR-MF | 0.0062 | 0.0147 | 0.0355 | 0.87× |
| random | 0.0078 | 0.0170 | 0.0413 | 1.00× |
| content, off-the-shelf | 0.0274 | 0.0597 | 0.1430 | 3.51× |
| **content, contrastively trained** | **0.0487** | **0.0845** | **0.1688** | **4.97×** |

Every collaborative model is **at or below random** here. Popularity and
item-kNN land at 0.19× because they assign identical scores to all cold items
and therefore return one constant list to every user — `coverage@10` of 0.001
and `gini` of 0.999 say the same thing. BPR-MF is indistinguishable from random.

**Full catalog** — rank new titles against the entire back catalog. Harsher and
more realistic; dominated by warm distractors.

| model | ndcg@10 | candidate_share@10 | popularity_lift@10 |
|---|---|---|---|
| popularity | 0.0000 | **0.000** | 38.6× |
| item-kNN | 0.0000 | **0.000** | 31.9× |
| BPR-MF | 0.0000 | **0.000** | 28.2× |
| content, off-the-shelf | 0.0112 | 0.064 | 2.4× |
| **content, contrastively trained** | **0.0165** | **0.109** | 3.3× |

#### A metric that was being misread

An earlier version of this README called the trained model's higher popularity
lift (2.4× → 3.3×) a regression. That was wrong, and the correction is worth
stating plainly because it is a trap this whole split invites.

On a cold-start split **every correct answer has zero training popularity, by
construction**. So the oracle popularity lift is `0.00×`, and a model scores a
*high* lift precisely by filling its slots with warm items that cannot possibly
be right. Popularity lift there does not mean "biased toward blockbusters"; it
means "wasted the ranking".

`candidate_share@10` says it directly: what fraction of the top 10 was even
eligible. The collaborative models sit at **0.000** — not one answerable item in
any list, which is exactly why their NDCG is zero and their lift is 28–39×. The
trained content model reaches 0.109 against off-the-shelf's 0.064, **69% more
eligible items surfaced**. Training improved this; it did not regress it.

The lesson generalises: a beyond-accuracy metric is only interpretable against
its oracle value on the split you are running. `popularity_lift` is meaningful
on `user_holdout`, where the truth is popularity-skewed, and misleading on
`cold_start`, where it is not.

### Mood axes: what the interpretability claim actually survives

The item tower can be routed through eight named bipolar axes
(`kokoro train --bottleneck`). Whether those axes *mean* their names is an
empirical question, so it gets an experiment rather than an assertion.

**The probe.** Each axis is given AniList tags a human would place at opposite
poles — `Iyashikei` and `Cute Girls Doing Cute Things` against `Tragedy`,
`Suicide`, `Gore` for *comfort*. Agreement is ROC-AUC, which is rank-based and
immune to the axis's arbitrary scale. **The 21 probe tags are withheld from the
item text during training**, so the model must infer *comfort* without ever
reading the word "Tragedy". Without that exclusion the test is circular and
passes for free.

**Anchor-weight ablation.** The anchor alignment loss pulls each axis's pole
phrases toward that pole. Removing it is what shows whether the axis names are
claims or decoration.

| anchor weight | val recall@1 | axis spread (σ) | mean AUC | inverted axes |
|---|---|---|---|---|
| 0.0 (ablation) | 0.175 | 0.085 | 0.465 | **2** |
| 1.0 | 0.183 | 0.186 | 0.537 | 0 |
| 5.0 | 0.176 | 0.218 | 0.561 | 0 |
| 20.0 | 0.179 | 0.238 | **0.565** | 0 |

Per axis, at weight 20 against the ablation:

| axis | AUC @ w=0 | AUC @ w=20 | verdict |
|---|---|---|---|
| `levity` | 0.471 | **0.686** | weak, responds strongly |
| `comfort` | **0.372** (inverted) | **0.663** | weak, inversion corrected |
| `hope` | **0.394** (inverted) | **0.630** | weak, inversion corrected |
| `cognition` | 0.540 | 0.509 | no signal |
| `intensity` | 0.493 | 0.471 | no signal |
| `intimacy` | 0.517 | 0.433 | degraded |
| `pace` | — | — | no tag proxy exists |
| `catharsis` | — | — | no tag proxy exists |

**What this supports, and what it does not.** The anchor mechanism works: it
roughly triples axis spread, corrects two inverted axes, and costs essentially
nothing in retrieval (recall@1 moves within noise). Three axes — `levity`,
`comfort`, `hope` — reach AUC 0.63–0.69, meaningfully above chance.

But **no axis clears the 0.70 bar this project set for "validated"**, and three
show no signal at all. So the honest claim is *"three of eight axes are weakly
but measurably aligned with their names"*, not "the model has interpretable mood
axes". The README will say the stronger thing when the numbers do.

One confound worth stating: a failing probe is not proof of a failing axis. The
`intimacy` probe (`Found Family`/`Ensemble Cast` against `Cosmic Horror`/
`Dystopian`) is really testing cast structure, not loneliness. `pace` and
`catharsis` have no proxy anywhere in 348 AniList tags. Distinguishing "the axis
is meaningless" from "the probe is bad" needs human judgements, which is the
part of Week 6 that remains genuinely undone.

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
- [x] Content tower + off-the-shelf cold-start baseline (first non-zero result)
- [x] Two-tower contrastive training (frozen backbone, learned projections)
- [ ] Hard-negative mining ablation; unfreeze the backbone
- [x] Mood bottleneck + anchor alignment + tag-probe validation
- [ ] **Human** mood judgements — the tag probes are a proxy, not a substitute
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
