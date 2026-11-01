# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) for the complete, high-density context, current progress, architectural map, and detailed specifications for Phase 2.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`) vs FAISS.
- **Current State**: Phase 1 is 100% complete and verified (16/16 `cargo test`, 27/27 `pytest`).
- **Next Task**: **Phase 2: HNSW Insert + Search on 10k vectors**.
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
- **Run Tests**: `cargo test` and `pytest`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
