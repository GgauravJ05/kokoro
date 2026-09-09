# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
