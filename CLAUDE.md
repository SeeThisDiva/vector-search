# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) and [`continuation.md`](./continuation.md) for the complete, high-density context, current progress, and architectural map across Phases 1–4.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`, `std` only) vs FAISS.
- **Current State**: Milestones M0, M1 (Phase 3.5), and M2 (Phase 4, `T2.1`–`T2.6`) are complete. Verified on 84 `cargo test` and 27 `pytest`. Phase 4 swept 23 configurations at 100k vectors and evaluated 4 empirical Pareto points at 1M vectors on authentic SIFT1M with 5 timed repetitions per point pinned to P-core `0x0004` ($M \in \{8, 16, 32, 48\}$). All 4 publication-grade Pareto charts are generated in `results/phase4/`, the Phase 4 write-up is in `README.md`, and all metrics are logged in `results/CLAIMS.md` Section 10.
- **Next Task**: Dev reviews Phase 4 evaluation and decides **Gate G3**. Upon Gate G3 sign-off, begin Milestone M3 (Phase 5: Filtered Search) by authoring `results/phase5/DESIGN.md` (strict stop at Gate G4 before any code).
- **Needs**: Rust >= 1.88 (`rust-version` in `Cargo.toml`). `data/*.fvecs` is gitignored; siftsmall-dependent tests skip when it is missing.
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
- **Run Tests**: `cargo test --release` (84 tests) and `python -m pytest tests -q -rs` (27 tests, 0 skipped).
- **Lint/Check**: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `ruff check .`, `mypy src tests scripts`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
- **Run Phase 2**: `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`.
- **Run Phase 3**: `cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json`.
- **Run Phase 4**: `cargo run --release --bin phase4 -- --scale 100000 --output-dir results/phase4`.
