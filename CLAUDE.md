# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) for the complete, high-density context, current progress, architectural map, and detailed specifications for Phase 2.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`) vs FAISS.
- **Current State**: Phases 1 and 2 are complete (51 `cargo test` incl. HNSW unit/adversarial suites, 27 `pytest`). Phase 2 passes its spec: Recall@10 >= 0.98 at < 0.05 ms/query (efSearch=20 on `siftsmall`).
- **Next Task**: **Phase 3: scale to `SIFT1M`, profile memory and build time** (profile first; likely `u32` ids + flatter adjacency storage).
- **Needs**: Rust >= 1.88 (`rust-version` in `Cargo.toml`). `data/*.fvecs` is gitignored; siftsmall-dependent tests skip when it is missing.
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
- **Run Tests**: `cargo test --release` and `pytest`.
- **Run Phase 2**: `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
