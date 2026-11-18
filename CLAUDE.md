# CLAUDE.md

From-scratch HNSW ANN engine in safe, `std`-only Rust (`#![forbid(unsafe_code)]`, Rust `>= 1.88`) with exact brute-force parity, filtered search, `SQ8`/`PQ` quantization, and FAISS benchmarks (Phases 1–7 complete).

## Agent & Architecture Docs
- **Read first (handoff & protocol)**: [`docs/dev/continuation.md`](./docs/dev/continuation.md), [`docs/dev/AGENT_PLAN.md`](./docs/dev/AGENT_PLAN.md), [`docs/dev/CONTEXT_AND_MEMORY.md`](./docs/dev/CONTEXT_AND_MEMORY.md)
- **Specs & results**: [`docs/HNSW_ALGORITHM.md`](./docs/HNSW_ALGORITHM.md) (Algorithms 1–5), [`results/README.md`](./results/README.md), [`results/CLAIMS.md`](./results/CLAIMS.md)

## Conventions & Verification Gate
- **Rules**: Classify changes (`PRESERVING` / `GRAPH-CHANGING` / `MEASUREMENT-ONLY`), never claim unmeasured numbers, keep `data/` gitignored (`--synthetic` / `--offline` available for smoke tests).
- **CI Gate** ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)):
  - Rust: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings` (and `--features instrument`), `cargo test --release`, `cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_phase3_tests --test rust_filtered_counters_tests`
  - Python: `cargo build --release --bin phase1 --bin filtered_exact_knn --bin phase5 --bin phase6`, `python -m pytest tests/ -rs --cov=src`, `ruff check .`, `mypy --config-file mypy.ini src tests scripts`
