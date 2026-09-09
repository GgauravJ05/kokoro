# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-09

Weeks 1–9 of the build. Headline: cold-start retrieval at 0.0845 NDCG@10
(4.97× random, +41% over an off-the-shelf encoder) where every collaborative
baseline scores exactly 0.0000, served at 4.4 ms p95.

### Reported negative results
- **Trajectory hypothesis falsified.** Curve shape carries no information beyond
  its own mean (shape-only AUC 0.485 = chance; mean+shape 0.513 < mean 0.547),
  despite a permutation control confirming a real 1.15× position signal. The
  neural trajectory encoder was deliberately not trained.
- **Mood axes only weakly validated.** The anchor mechanism triples axis spread
  and corrects two inverted axes at no retrieval cost, but 3 of 8 axes show
  signal and none clears the stated 0.70 bar.

### Added
- Evaluation harness, written before the models: accuracy metrics
  (`recall`/`precision`/`ndcg`/`mrr`/`hit_rate`) and beyond-accuracy metrics
  (`coverage`, `gini`, `intra_list_diversity`, `serendipity`, `popularity_lift`).
- Four splitting strategies — `random`, `leave_one_out`, `temporal`,
  `cold_start` — with leakage regression tests.
- Baselines: random, popularity, item-kNN with shrunk cosine similarity.
- `BPRMatrixFactorization`: from-scratch BPR-MF with hand-derived gradients,
  NumPy only.
- `InfoNCELoss` and `HardNegativeInfoNCELoss` with the symmetric/asymmetric and
  in-batch/mined ablation switches.
- `MoodBottleneck`: eight named bipolar mood axes with an anchor alignment loss.
- `TrajectoryEncoder`, `resample_trajectory` and `shape_distance` for
  narrative-arc mood curves.
- `TwoTowerRetriever` with untied towers and a learned trajectory gate.
- Rule-based arc mention extraction from review text.
- AniList GraphQL ingestion with a token-bucket rate limiter.
- `ExactIndex` and `HNSWIndex` with a recall regression test against exact search.
- FastAPI service skeleton with an AGPL §13 `/source` endpoint.
- Three-layer provenance: source headers, artifact stamps, and a
  retrieval-neutral embedding watermark.
- `kokoro` CLI: `provenance`, `config`, `ingest`, `benchmark`, `watermark`.

[Unreleased]: https://github.com/GgauravJ05/kokoro/compare/main...HEAD
