# continuation.md: handoff for coding agents

Read this first, then `CLAUDE.md` and `CONTEXT_AND_MEMORY.md`. This file is the most recent state; if they disagree, this wins.

## Project in one line
From-scratch HNSW ANN search in Rust (`#![forbid(unsafe_code)]`, std only), benchmarked against brute force and FAISS. Phases 1, 2 and 3 are implemented. **Phase 3 has been code-reviewed (see "Review log"), but Dev has not signed it off. Do not start Phase 4 until he confirms.**

## Working agreement
- Dev designs, agents code. Do not start a later phase until the current one is reviewed.
- Free tools and datasets only.
- Never claim a number you did not measure. Every benchmark claim must come from a run whose JSON records hardware and dataset source.
- Do not weaken a pass condition to make it green. If a target is missed, report the miss.
- A check that did not run must never be reported as passed. (This rule exists because of review finding 1 below.)

## Current verified state
- `cargo test --release`: 55 pass (12 adversarial + 16 phase1 + 23 phase2 + 4 phase3). Siftsmall tests skip when `data/siftsmall/*.fvecs` is missing (helper: `tests/common/mod.rs`). Phase 3 tests use small synthetic data and need no download.
- `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` clean.
- `pytest`: 27 pass (needs `faiss-cpu`, in `requirements.txt`); `ruff` and `mypy` were clean at the last Python change. The review did not touch Python and did not re-run it.
- Needs Rust >= 1.88 (`rust-version` in `Cargo.toml`; `as_chunks` and lockfile v4). The review used rustc 1.90.
- Git: Phase 1 to 3 and the Phase 3 review fixes (`review_fixes.patch`, commit `d4b6fed` + follow-up edge-case test/validation hardening) are applied and committed (`git log`).

## Review log (Phase 3 review, what happened)
After Phase 3 was committed, Dev asked for a review of the whole repo and fixes for any flaws. The review read `hnsw.rs` and the Phase 2/3 benchmark code in full and checked them against `docs/HNSW_ALGORITHM.md`, then ran the full test suite before and after the changes. No Windows machine or SIFT1M run was available to the reviewer, so **no benchmark was re-run and none of the numbers in `data/*.json` changed**.

Conclusion: the HNSW algorithm implementation is correct. Insert, layer sampling, search-layer, the diversity heuristic, neighbor shrinking and entry-point updates all match the paper, and the SIFT1M recall and top layer (5, matching ln(1e6)/ln(16)) are what a correct implementation should give. The flaws were in how results were reported and measured, plus one panic and stale docs.

### Flaws found and fixed (do not undo)
| # | Flaw | Where | Fix |
|---|------|-------|-----|
| 1 | A skipped reproducibility check (`--no-repro-check`) was reported as `reproducibility_verified = true`, and in Phase 2 that fed `overall_phase2_passed`. | `benchmark.rs`, Phase 2 and Phase 3 runners | Skipped now yields `false`. New field `reproducibility_checked` (struct + JSON) says whether it ran. Console prints PASSED / FAILED / SKIPPED. |
| 2 | `phase3_completed` was hardcoded `true` in the JSON. | `Phase3Report::to_json_string` | Now `monotonic_scaling_verified && reproducibility_verified`. (`forbid_unsafe_code_enforced` and `std_only_enforced` stay `true`; they are compile-time facts.) |
| 3 | The 10k and 100k rows of `multi_scale_baseline_summary` are constants pasted into `load_or_default_baseline_profiles`, but were presented like measurements of the current run. | `BaselineScaleProfile` | New field `measured_this_run` (also in the JSON). Hardcoded rows are `false`; the live 1M row and rows from `run_baseline_multi_scale_profile` are `true`. |
| 4 | `search_batch_parallel` validated k and dimensions but not NaN/Inf, so a bad query panicked inside a worker thread, while the sequential path returned `Err`. | `hnsw.rs` | All queries are checked up front and return `InvalidArgument`. |
| 5 | Phase 3 had no tests at all. | `tests/rust_phase3_tests.rs` (new, 4 tests) | Covers the three report-semantics fixes and the parallel NaN case. The NaN test was confirmed to fail without the fix. |
| 6 | Docs were stale: they said Phase 2 was uncommitted, Phase 3 was next, and listed 51 tests; the README had no Phase 3 results or `phase3` binary. | `README.md`, `CLAUDE.md`, `CONTEXT_AND_MEMORY.md`, this file | Refreshed. README now has a Phase 3 results section with caveats. |

### Flaws found and NOT fixed (open, need Dev's decision)
- **Per-scale peak RSS for 10k and 100k is meaningless.** `get_peak_rss_mb` reads a process-wide high-water mark, and the multi-scale profile runs inside one process after loading the full 488 MiB base file. That is why the 10k row shows ~510 MB. Only the 1M row is valid. A proper fix is to run each scale in its own process (or add a `--max-base` run per scale) and record RSS after build.
- **`official_ground_truth_matched` is a tautology.** It calls `verify_self_recall(gt, gt)`. It does not show the official file agrees with brute force on SIFT1M. A real check: run `ExactKnn` on ~100 queries against the full 1M base and compare to the first 100 rows of `sift_groundtruth.ivecs`.
- **Latency is one noisy run.** The same index at efSearch=50 had 0.35 ms mean in the baseline run and 0.42 ms in the Phase 3 run. Latency claims need several repetitions (report the median and spread) with nothing else running.
- **`graph_bytes_per_vector` includes the visited-tag array** (4 B/vector, ~1%), and the earlier baseline JSON (339.6 B/vec) and the Phase 3 JSON (342.7 B/vec) define "graph bytes" slightly differently. Pick one definition (graph only, excluding scratch and visited tags) when memory work starts.
- **Allocator overhead is not counted** in `memory_bytes()` (about 2M small `Vec` allocations); peak RSS is the honest total.
- **`search_knn` takes `&mut self`** only to borrow the internal tracker. `search_knn_with_tracker(&self, ...)` already exists; prefer it in new code.
- `HnswConfig::new` silently clamps `m` and `ef_construction`, while `validate()` rejects the same values. Harmless, but inconsistent.

## Measured results (Dev's machine: i7-12650H, 16 logical cores, 15.8 GB RAM, Windows, single-threaded queries)
Phase 2, siftsmall, M=16, efC=100, k=10 (recall is deterministic with seed 42):

| efSearch | Recall@10 |
|---------:|----------:|
| 10 | 95.1% |
| 20 | 98.6% |
| 50 | 100% |
| 100 | 100% |
| 200 | 100% |

Spec target (recall >= 0.98 at < 0.05 ms mean) is met at ef=20. Latency tables are in the README.

Phase 3, SIFT1M, M=16, efC=100, k=10, official ground truth, 10,000 queries, one run:

| efSearch | Recall@10 | Mean (ms) | p99 (ms) |
|---------:|----------:|----------:|---------:|
| 10 | 73.3% | 0.124 | 0.326 |
| 20 | 84.9% | 0.194 | 0.542 |
| 50 | 94.7% | 0.419 | 0.961 |
| 100 | 98.1% | 0.688 | 1.461 |
| 200 | 99.4% | 1.192 | 2.146 |

Build 728 s (1,373 vectors/s). Peak working set after build 834 MiB. About 855 B/vector total: 512 raw + ~343 graph (layer-0 edge lists 264 B, node structs 40 B, upper layers 9 B, `Vec` headers 25.6 B, visited tags 4 B). Recall at fixed efSearch=50 falls with scale (99.8% at 10k, 98.1% at 100k, 94.7% at 1M), so efSearch must grow with N.

## Design decisions already made (do not undo)
- `HnswConfig.m_max`, `m_max0`, `m_l` are `Option` overrides, derived from `m` when `None` (`effective_m_max()`, `effective_m_max0()`, `effective_m_l()`). No sentinel-value guessing.
- `insert(id)` requires ids in order `0..n`; anything else returns `InvalidArgument`.
- Phase 2 `overall_phase2_passed` = monotonic recall AND reproducibility (actually run) AND performance target (recall >= 0.98 and mean latency < 0.05 ms at one ef). Speedup vs brute force is reported, not gated.
- Sweep points with `ef < k` collapse to `ef == k` (the index clamps them).
- Distance ties break on `(distance, id)` everywhere. Keep it, since parity tests depend on it.
- Report flags: `reproducibility_checked`, `reproducibility_verified`, `measured_this_run` and the computed `phase3_completed` are part of the JSON contract now. Do not turn them back into constants.

## Known limitations
- Inner-product mode is only parity-tested. The diversity heuristic assumes distance-like (non-negative) values, which negative dot products violate. Do not report IP recall as validated.
- `siftsmall` saturates at 100% recall by ef=50, so it says little about graph quality. SIFT1M is the real test.
- `HnswIndex` borrows the matrix (`&'a RowMatrix<f32>`); there is no deletion or update.
- Graph storage is `Vec<Vec<Vec<usize>>>` with 8-byte ids and layer-0 capacity `m_max0 + 1` (33 slots), so about 343 B/vector of graph for 128-d vectors.

## How to move forward
Order matters. Do not skip ahead, and stop for Dev's review after each numbered step.

1. **Dev applies and commits the review patch**, runs `cargo test --release`, and pushes so CI runs on a fresh clone.
2. **Small Phase 3 follow-ups (propose to Dev first, then do them as one reviewable change):**
   - real ground-truth check on SIFT1M (open flaw 2);
   - per-scale RSS in separate processes (open flaw 1);
   - repeat the efSearch sweep N times and report median and spread (open flaw 3);
   - regenerate `data/phase3_report.json` on Dev's machine and update the README Phase 3 table. Do not edit numbers by hand.
3. **Profile the 1M build before optimizing.** Build is 728 s. Verify with a profiler, do not assume. Candidates, in rough order of expected payoff:
   - the neighbor heuristic's `HashSet<usize>` (SipHash). Dedup only matters when `extend_candidates` is true; a generation-tagged `Vec<u32>` gives identical output without hashing;
   - the shrink step: with `keep_pruned = true`, a saturated node is re-pruned back to full degree, so nearly every new reverse edge triggers a full heuristic run. Try `keep_pruned = false` for shrinking only, and measure both build time and recall (this changes the graph, so it is an experiment, not a refactor);
   - `u32` ids and flat fixed-capacity adjacency (also roughly halves graph memory);
   - per-query heap allocations in `search_layer`.
4. **Every optimization must keep `--repro-check` passing and all tests green.** If a change alters the graph, say so and re-record recall; if it does not, add a test that proves identical results before and after.
5. **Phase 4 (only after Dev confirms Phase 3):** sweep M, efConstruction and efSearch and plot Recall@10 vs p99 latency. Use the same JSON conventions (hardware, dataset source, `measured_this_run`). Sweeps at 1M are slow (~12 min per build at M=16, efC=100), so propose the grid and a time budget to Dev before running anything, and consider 100k for the wide sweep and 1M for the final few points.

## Commands
```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json
cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json
python -m pytest tests -q && ruff check . && mypy src tests
```
Tip for quick iteration: release builds use `lto = "thin"` and `codegen-units = 1`, which is slow to compile. For test runs only, `CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo test --release` compiles in about 10 s. Never use those overrides for numbers you report.

## Housekeeping checklist
- [x] Apply and commit the review patch.
- [ ] CI: confirm green on a fresh clone after pushing.
- [ ] Dev signs off Phase 3.
- [ ] Update `CLAUDE.md`, `CONTEXT_AND_MEMORY.md` and README test counts whenever tests are added.
