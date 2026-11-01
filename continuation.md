# continuation.md: handoff for coding agents

Read this first, then `CLAUDE.md` and `CONTEXT_AND_MEMORY.md`. This file is the most recent state; if they disagree, this wins.

## Project in one line
From-scratch HNSW ANN search in Rust (`#![forbid(unsafe_code)]`, std only), benchmarked against brute force and FAISS. Phases 1 and 2 are done. **Next: Phase 3 (SIFT1M, profile memory and build time).**

## Working agreement
- Dev designs, agents code. Do not start a later phase until the current one is reviewed. Stop after Phase 3 and report.
- Free tools and datasets only.
- Never claim a number you did not measure. Every benchmark claim must come from a run whose JSON records hardware and dataset source.
- Do not weaken a pass condition to make it green. If a target is missed, report the miss.

## Current verified state
- `cargo test --release`: 51 pass (12 adversarial + 16 phase1 + 23 phase2). Siftsmall tests skip when `data/siftsmall/*.fvecs` is missing (helper: `tests/common/mod.rs`).
- `pytest`: 27 pass (needs `faiss-cpu`, now in `requirements.txt`). `ruff`, `mypy`, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` all clean.
- Needs Rust >= 1.88 (`rust-version` in `Cargo.toml`; `as_chunks` and lockfile v4).
- **Nothing from Phase 2 is committed.** Untracked: `src/hnsw.rs`, `src/bin/`, `tests/rust_phase2_tests.rs`, `tests/adversarial_hnsw_tests.rs`, `tests/common/`. First task: review the diff, then commit in small logical commits.

## Phase 2 results (siftsmall, M=16, efC=100, k=10)
Recall is deterministic; latency is hardware-dependent.

| efSearch | Recall@10 |
|---------:|----------:|
| 10 | 95.1% |
| 20 | 98.6% |
| 50 | 100% |
| 100 | 100% |
| 200 | 100% |

Spec target (recall >= 0.98 at < 0.05 ms mean) is met at ef=20. The README latency table was measured on a 1-core Linux sandbox, not Dev's i7-12650H. **Regenerate on Dev's machine** before it is quoted anywhere:
`cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json`

## Design decisions already made (do not undo)
- `HnswConfig.m_max`, `m_max0`, `m_l` are `Option` overrides, derived from `m` when `None` (`effective_m_max()`, `effective_m_max0()`, `effective_m_l()`). No sentinel-value guessing.
- `insert(id)` requires ids in order `0..n`; anything else returns `InvalidArgument`.
- Phase 2 `overall_phase2_passed` = monotonic recall AND reproducibility AND performance target (recall >= 0.98 and mean latency < 0.05 ms at one ef). Speedup vs brute force is reported, not gated.
- Sweep points with `ef < k` collapse to `ef == k` (the index clamps them).
- Distance ties break on `(distance, id)` everywhere. Keep it, since parity tests depend on it.

## Known limitations
- Inner-product mode is only parity-tested. The diversity heuristic assumes distance-like (non-negative) values, which negative dot products violate. Do not report IP recall as validated.
- `siftsmall` saturates at 100% recall by ef=50, so it says little about graph quality. On a synthetic clustered set the same code needed ef=100 for ~96%. SIFT1M is the real test.
- `HnswIndex` borrows the matrix (`&'a RowMatrix<f32>`); there is no deletion or update.

## Phase 3 task: scale to SIFT1M
Goal: build on 1M x 128-d vectors, measure build time, memory, recall and latency.

1. **Profile before optimizing.** Record at 10k first (build is ~1.0-1.6 s), then at 100k, then 1M. Report build time, peak RSS, bytes per vector.
2. Suspected hot spots (verify with a profiler, do not assume):
   - the neighbor-selection heuristic allocates a `HashSet` and `Vec`s on every call;
   - graph storage is `Vec<Vec<Vec<usize>>>` (per-node, per-layer heap allocations, 8-byte ids);
   - `VisitedTracker` and candidate heaps are per-query structures.
3. Likely changes, only if profiling supports them: `u32` ids, flat or fixed-capacity adjacency, reusable scratch buffers for the heuristic.
4. Every optimization must keep bit-exact reproducibility (`--repro-check`) and all existing tests green. Add a test if behavior could change.
5. Data: `cargo run --release --bin phase1 -- --dataset sift1m` downloads the Texmex corpus (`data/` is gitignored). Check disk and RAM first; 1M x 128 x 4 B is ~512 MB for vectors alone.
6. Acceptance for Phase 3 (propose and get Dev to confirm before coding if unsure): Recall@10 vs ef on SIFT1M against official ground truth, build time, memory per vector, all in a JSON report with hardware and dataset source recorded.

## Commands
```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json
python -m pytest tests -q && ruff check . && mypy src tests
```

## Housekeeping checklist
- [ ] Commit the Phase 2 work (see above).
- [ ] Regenerate the README Phase 2 table on Dev's machine and commit it (`data/*.json` is gitignored, so paste the table into the README).
- [ ] CI: confirm green on a fresh clone after pushing.
- [ ] Update `CLAUDE.md`, `CONTEXT_AND_MEMORY.md` and README test counts whenever tests are added.
