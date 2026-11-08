# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) and [`continuation.md`](./continuation.md) for the complete, high-density context, current progress, and architectural map across Phases 1–3.5.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`) vs FAISS.
- **Current State**: Milestone M0 (`T0.1`–`T0.6`, Gates `G0`/`G1`) closed; Phase 4 `T2.1` (100k pilot + `results/phase4/PROPOSAL.md`) and `T2.2` (`phase4` sweep runner + CLI) implemented and verified (84 `cargo test`, 27 `pytest`). Phase 4 100k pilot ($M=16, \text{efC}=100$, commit `26a9807`): build 26.066 s (3,836.41 vec/s), graph-only 145.57 B/vec (13.88 MiB), Peak RSS 77.61 MiB, Recall@10 from 0.82103 (ef=10) to 0.99702 (ef=600), 12-point sweep 173.49 s (2.89 min).
- **Next Task**: Dev reviews Phase 4 proposal (`results/phase4/PROPOSAL.md`) and decides **Gate G2** (approves 23-config 100k grid and time budget), then proceed to `T2.3` (100k sweep execution).
- **Needs**: Rust >= 1.88 (`rust-version` in `Cargo.toml`). `data/*.fvecs` is gitignored; siftsmall-dependent tests skip when it is missing.
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
- **Run Tests**: `cargo test --release` and `pytest`.
- **Run Phase 2**: `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`.
- **Run Phase 3**: `cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json`.
- **Run Phase 4**: `cargo run --release --bin phase4 -- --scale 100000 --output-dir results/phase4`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
