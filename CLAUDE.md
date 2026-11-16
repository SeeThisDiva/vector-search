# CLAUDE.md

See [`CONTEXT_AND_MEMORY.md`](./CONTEXT_AND_MEMORY.md) and [`continuation.md`](./continuation.md) for the complete, high-density context, current progress, and architectural map across Phases 1–5.

## Quick Reference
- **Mission**: HNSW Approximate Nearest Neighbor Search from scratch in Rust (`#![forbid(unsafe_code)]`, `std` only) vs FAISS.
- **Current State**: Milestones M0–M3 (Phases 1–5) are complete and verified (`184` Rust tests + `33` Python `pytest` tests, 0 skipped; `11` `--features instrument` tests pass). Phase 5 evaluated `100k` and `1M` authentic SIFT1M filtered search across `S1`, `S2`, and `S3` (`Option C`, `reps=5`, `48` cells + `70`-arm sensitivity + `18` opaque `eval_limit=16_384` probes per scale), authored [`results/phase5/FINDINGS.md`](./results/phase5/FINDINGS.md), and logged claims in [`results/CLAIMS.md`](./results/CLAIMS.md) §11. Draft design specifications for **Phase 6 ([`results/phase6/DESIGN.md`](./results/phase6/DESIGN.md))** and **Phase 7 ([`results/phase7/DESIGN.md`](./results/phase7/DESIGN.md))** are ready.
- **Next Task**: Dev reviews Phase 5 results (plus formal Gate G3 tick if desired) and decides **Gate G5** (`SQ8 only` vs `SQ8 + PQ` in `results/phase6/DESIGN.md`) and **Gate G6** (`FAISS HNSWFlat` protocol in `results/phase7/DESIGN.md`) before any M4/M5 library code or timed runs begin.
- **Needs**: Rust >= 1.88 (`rust-version` in `Cargo.toml`). `data/*.fvecs` and `data/phase5/` caches are gitignored; siftsmall-dependent tests skip when missing (`--offline` / `--synthetic` flags available on `phase5`).
- **Specs**: See `docs/HNSW_ALGORITHM.md` (Algorithms 1–5), `results/phase5/DESIGN.md`, `results/phase6/DESIGN.md`, and `results/phase7/DESIGN.md`.
- **Run Tests**: `cargo test --release` (`184` tests), `cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_filtered_counters_tests` (`11` tests), and `python -m pytest tests -q -rs` (`33` tests, 0 skipped).
- **Lint/Check**: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo clippy --all-targets --features instrument -- -D warnings`, `ruff check .`, `mypy src tests scripts`.
- **Run Phase 1**: `cargo run --release --bin phase1 -- --dataset siftsmall --k 10`.
- **Run Phase 2**: `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`.
- **Run Phase 3**: `cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json`.
- **Run Phase 4**: `cargo run --release --bin phase4 -- --scale 100000 --output-dir results/phase4`.
- **Run Phase 5**: `cargo run --release --bin phase5 -- --matrix --scale 100000 --reps 5 --include-sensitivity --output-dir results/phase5`.
