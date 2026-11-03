# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) for the complete, high-density context, current progress, architectural map, and detailed specifications for Phase 2.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`) vs FAISS.
- **Current State**: Phases 1, 2 and 3 are implemented and committed (55 `cargo test`, 27 `pytest`). Phase 3 on SIFT1M (M=16, efC=100): build 728 s, ~855 B/vector, Recall@10 94.7% at efSearch=50 and 98.1% at efSearch=100. Phase 3 is awaiting Dev's review; do not start Phase 4 before that.
- **Next Task**: Dev reviews Phase 3, then Phase 4 (parameter sweep). Likely first optimizations, only if profiled: `u32` ids, flat adjacency, replace the `HashSet` in the neighbor heuristic.
- **Needs**: Rust >= 1.88 (`rust-version` in `Cargo.toml`). `data/*.fvecs` is gitignored; siftsmall-dependent tests skip when it is missing.
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
- **Run Tests**: `cargo test --release` and `pytest`.
- **Run Phase 2**: `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`.
- **Run Phase 3**: `cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
