# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) and [`continuation.md`](./continuation.md) for the complete, high-density context, current progress, and architectural map across Phases 1–3.5.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`) vs FAISS.
- **Current State**: Phases 1, 2, 3, and 3.5 (Milestone M1) are implemented and verified (84 `cargo test`, 27 `pytest`). Phase 3 / 3.5 on SIFT1M (M=16, efC=100, compact `u32` flat adjacency): build 493.8 s (2,025.2 vec/s; 415.1 s / 2,409.2 vec/s with `--no-keep-pruned-shrink`; 100k 3-run median 27.157 s vs 30.299 s baseline), 664.91 B/vector total (512.00 B raw + 152.91 B graph incl. visited tags / 148.91 B graph-only, down 56.0% from 338.67 B), Peak RSS 646.05 MiB (11.94 MiB at 10k, 69.21 MiB at 100k in isolated child processes), Recall@10 94.703% at efSearch=50 (0.299 ms median mean) and 98.061% at efSearch=100 (0.490 ms median mean) across 5 sweep repetitions. Phase 3 / Gate G1 is awaiting Dev's sign-off; do not start Phase 4 before that.
- **Next Task**: Dev reviews Phase 3.5 (`results/phase3_5/LOG.md`) and decides Gate G1 (`keep_pruned_shrink = Some(false)` default vs flag), then Phase 4 (parameter sweep).
- **Needs**: Rust >= 1.88 (`rust-version` in `Cargo.toml`). `data/*.fvecs` is gitignored; siftsmall-dependent tests skip when it is missing.
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
- **Run Tests**: `cargo test --release` and `pytest`.
- **Run Phase 2**: `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`.
- **Run Phase 3**: `cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json`.
- **Run Phase 4**: `cargo run --release --bin phase4 -- --scale 100000 --output-dir results/phase4`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
