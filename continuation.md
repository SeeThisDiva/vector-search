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
- `cargo test --release`: 84 pass (12 `adversarial_hnsw` + 10 `adversarial_graph_fingerprint` + 8 `challenge_m1_p35` + 16 `phase1` + 23 `phase2` + 11 `phase3` + 4 `phase4`). Siftsmall tests skip when `data/siftsmall/*.fvecs` is missing (helper: `tests/common/mod.rs`). Phase 3 and Phase 3.5 tests use small synthetic data and need no download.
- `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` clean.
- `pytest`: 27 pass (needs `faiss-cpu`, in `requirements.txt`); `ruff check .` and `mypy src tests` clean. The 4 Rust-parity tests only run when `target/release/phase1` exists, so build it first (`cargo build --release --bin phase1`) or they skip silently and show as 23 passed, 4 skipped.
- Needs Rust >= 1.88 (`rust-version` in `Cargo.toml`; `as_chunks` and lockfile v4).
- Git: Phase 1 to 3, the Phase 3 review fixes (`review_fixes.patch`, commit `d4b6fed` + `9e3c1a2`), the Phase 3 follow-up fixes, and **Phase 3.5 (Milestone M1)** are implemented and verified:
  - `T1.1` / `T1.4`: Zero-cost `--features instrument` phase timing and distance evaluation counters (`search_layer` 78.83%, `shrink` 14.19%, `heuristic` 4.80% on 1M).
  - `T1.2` / `T1.3`: Logical `HnswIndex::graph_fingerprint() -> u64` (including `enter_point`) and golden fixtures (`results/phase3_5/golden_10k.json`, `results/phase3_5/golden_100k.json`) with bit-exact distance/ID verification.
  - `T1.5` (`seen_tags` in `HeuristicScratch`) and `T1.6` (`search_layer` binary heap scratch reuse): tested at 100k, reverted and logged `"no gain"` in `results/phase3_5/LOG.md`.
  - `T1.7` (`u32` flat adjacency storage in `layer0` and `upper_arena`, 8-byte `HnswNodeMeta`, `HnswIndex::neighbors(layer, node) -> &[u32]`): **KEPT** (`d6d36ff`). Cuts graph-only memory by **56.0%** at 1M (`338.67 -> 148.91 B/vector`, `-180.97 MiB`), cuts isolated 1M Peak RSS from `845.75 MiB` to **`646.05 MiB`** (`-199.70 MiB`), and speeds up 100k 3-run median build time by **10.37%** (`30.299 s -> 27.157 s`) while keeping 100% bit-exact graph fingerprints and query results.
  - `T1.9` (`HnswConfig::keep_pruned_shrink: Option<bool>`, default `None`, CLI `--no-keep-pruned-shrink`): evaluated at 100k (`21.351 s`, `-29.53%`) and 1M (`415.076 s`, `-18.81%` vs `511.271 s` baseline; `-11.87%` vs `470.966 s` historical Phase 3) and recorded in `results/phase3_5/LOG.md` + `results/phase3_5/exp_no_keep_pruned_shrink_1m.json` for Dev's Gate G1 decision.
  - `T1.10`: `data/phase3_report.json` (`results/phase3_5/final_1m_report.json`) and `data/baseline_profile_report.json` regenerated from a clean commit on the final code (`366f185`).

## Review log (Phase 3 review, what happened)
After Phase 3 was committed, Dev asked for a review of the whole repo and fixes for any flaws. The review read `hnsw.rs` and the Phase 2/3 benchmark code in full and checked them against `docs/HNSW_ALGORITHM.md`, then ran the full test suite before and after the changes.

Conclusion: the HNSW algorithm implementation is correct. Insert, layer sampling, search-layer, the diversity heuristic, neighbor shrinking and entry-point updates all match the paper, and the SIFT1M recall and top layer (5, matching ln(1e6)/ln(16)) are what a correct implementation should give. The flaws were in how results were reported and measured, plus one panic and stale docs.

### Flaws found and fixed (do not undo)
| # | Flaw | Where | Fix |
|---|------|-------|-----|
| 1 | A skipped reproducibility check (`--no-repro-check`) was reported as `reproducibility_verified = true`, and in Phase 2 that fed `overall_phase2_passed`. | `benchmark.rs`, Phase 2 and Phase 3 runners | Skipped now yields `false`. New field `reproducibility_checked` (struct + JSON) says whether it ran. Console prints PASSED / FAILED / SKIPPED. |
| 2 | `phase3_completed` was hardcoded `true` in the JSON. | `Phase3Report::to_json_string` | Now `monotonic_scaling_verified && reproducibility_verified`. (`forbid_unsafe_code_enforced` and `std_only_enforced` stay `true`; they are compile-time facts.) |
| 3 | The 10k and 100k rows of `multi_scale_baseline_summary` are constants pasted into `load_or_default_baseline_profiles`, but were presented like measurements of the current run. | `BaselineScaleProfile` | New field `measured_this_run` (also in the JSON). Hardcoded fallback rows are `false`; live rows profiled in isolated child processes (or `run_baseline_multi_scale_profile`) are `true`. |
| 4 | `search_batch_parallel` validated k and dimensions but not NaN/Inf, so a bad query panicked inside a worker thread, while the sequential path returned `Err`. | `hnsw.rs` | All queries are checked up front and return `InvalidArgument`. |
| 5 | Phase 3 had no tests at all. | `tests/rust_phase3_tests.rs` (7 tests) | Covers report-semantics fixes, parallel NaN/Inf/empty-index guards, brute-force official ground-truth verification (including single bad neighbor rejection and $k$-th boundary distance ties), multi-run sweep median & spread, prefix `.fvecs` loading, `graph_only_bytes_per_vector`, and `HnswConfig::new` validation. |
| 6 | Docs were stale: they said Phase 2 was uncommitted, Phase 3 was next, and listed 51 tests; the README had no Phase 3 results or `phase3` binary. | `README.md`, `CLAUDE.md`, `CONTEXT_AND_MEMORY.md`, this file | Refreshed with 58 Rust + 27 Python tests and regenerated Phase 3 SIFT1M results. |
| 7 | Per-scale peak RSS for 10k and 100k was polluted by loading the full 488 MiB 1M base file in one process (~510 MB at 10k). | `dataset.rs`, `benchmark.rs` | Added prefix `.fvecs`/`.ivecs` loading (`read_fvecs_max_rows`, `read_ivecs_max_rows`) and isolated child-process profiling (`--internal-profile-scale`). Isolated Peak RSS is now **13.70 MiB** at 10k, **88.67 MiB** at 100k, and **845.84 MiB** at 1M. |
| 8 | `official_ground_truth_matched` was a tautology (`verify_self_recall(gt, gt)`). | `exact_knn.rs`, `benchmark.rs` | Added `ExactKnn::search_parallel_with_base(&loaded.base, ...)` and verified the first 100 queries against `sift_groundtruth.ivecs` (requiring $\ge 99\%$ ID overlap and 100% of official top-$k$ neighbors to have true distance $\le$ the brute-force $k$-th distance, accounting for the 1 integer $L_2^2$ boundary tie on SIFT1M: `999/1000` ID recall, `991/1000` exact rank matches). |
| 9 | Latency was a single noisy run. | `benchmark.rs` | `efSearch` sweep now runs `--sweep-runs 5` times by default and reports median latency (`mean`, `p50`, `p95`, `p99`, `QPS`) plus min, max, spread, and per-run arrays in JSON. |
| 10 | `graph_bytes_per_vector` included the 4 B/vector visited-tag array without a separate graph-only metric, and old 10k/100k baseline rows came from an earlier code version (`339.6` vs `342.7` B/vec). | `hnsw.rs`, `benchmark.rs` | Added `graph_only_bytes_per_vector` / `graph_only_memory_bytes` (`338.98` / `338.66` / `338.67` B/vec) alongside `graph_bytes_per_vector` (`342.98` / `342.66` / `342.67` B/vec), all measured on the same code version. |
| 11 | `search_batch_timed` required `&mut self`, and `HnswConfig::new` silently clamped `m` and `ef_construction`. | `hnsw.rs` | Added `search_batch_timed_with_tracker(&self, ...)` and removed silent clamping in `HnswConfig::new` so `validate()` rejects invalid values consistently. |

### Remaining notes (not bugs, architectural context)
- **Allocator overhead is not counted** in `memory_bytes()` (about 2M small `Vec` allocations); peak RSS is the honest total.
- **`search_knn` takes `&mut self`** only to borrow the internal tracker. `search_knn_with_tracker(&self, ...)` and `search_batch_timed_with_tracker(&self, ...)` exist; prefer them in new code.

### Review round 2 (commit `9e3c1a2` and the state after it)
Dev committed the round 1 patch (`d4b6fed`) and then `9e3c1a2`. The round 2 review diffed both against the patch, re-ran the whole gate, and found no new defects.
- `9e3c1a2` moves the non-finite and dimension checks in `search_batch_parallel` ahead of the empty-index shortcut, and the dimension check ahead of it in `search_knn`. Result: a NaN or wrong-size query is rejected on an empty index too, matching the sequential path. Correct, and covered by the extended `parallel_search_rejects_non_finite_query_instead_of_panicking` test (NaN, +Inf, -Inf, empty index, wrong dimension).
- The live 1M row now also sets `bytes_per_vector`, `raw_vector_bytes_per_vector` and `k`. Correct (`bytes_per_vector` is defined as graph bytes per vector in that struct).
- `--no-repro-check` is now in the Phase 1/2 help text. The Phase 2 repro tests were added to `rust_phase3_tests.rs` and pass.
- `data/phase2_report.json` was regenerated on Dev's machine after the fixes: `reproducibility_checked` and `reproducibility_verified` true, `overall_phase2_passed` true, target met at ef=20 (recall 98.6%, mean 0.019 ms).
- `data/phase3_report.json` and `data/baseline_profile_report.json` have now been regenerated on Dev's machine with all new fields (`reproducibility_checked: true`, `reproducibility_verified: true`, `measured_this_run: true` on all 3 scales, `official_ground_truth_matched: true`, `graph_only_bytes_per_vector`, and 5-run `ef_search_sweep` medians & spreads).
- Working-tree note: if CRLF vs LF line endings show up on Windows, set `core.autocrlf` or add a `.gitattributes`; do not commit a whole-file line-ending rewrite.

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

Phase 3 (after Phase 3.5 `T1.7` `u32` flat adjacency), SIFT1M, M=16, efC=100, k=10, official ground truth (verified against brute force on 100 queries), 10,000 queries, median of 5 runs:

| efSearch | Recall@10 | Median Mean (ms) | Mean Spread (ms) | Median p99 (ms) | p99 Spread (ms) | Median QPS |
|---------:|----------:|-----------------:|-----------------:|----------------:|----------------:|-----------:|
| 10 | 73.315% | 0.100 | 0.007 | 0.175 | 0.020 | 9,942.1 |
| 20 | 84.949% | 0.153 | 0.003 | 0.255 | 0.013 | 6,523.0 |
| 50 | 94.703% | 0.299 | 0.030 | 0.477 | 0.270 | 3,344.9 |
| 100 | 98.061% | 0.490 | 0.033 | 0.723 | 0.096 | 2,040.4 |
| 200 | 99.376% | 0.828 | 0.035 | 1.177 | 0.098 | 1,208.0 |

Build 493.8 s (2,025.2 vectors/s; 415.1 s / 2,409.2 vectors/s with `--no-keep-pruned-shrink`; 100k 3-run median dropped from 30.299 s to 27.157 s default / 21.351 s with `--no-keep-pruned-shrink`). Isolated Peak RSS after build: **11.94 MiB** at 10k, **69.21 MiB** at 100k, **646.05 MiB** at 1M (down 199.7 MiB from 845.75 MiB). 664.91 B/vector total: 512.00 B raw + 152.91 B graph total / **148.91 B graph-only** (layer-0 flat `u32` edges 132.00 B, `HnswNodeMeta` structs 8.00 B, upper-layer arena 8.91 B, outer headers 0.00 B, visited tags 4.00 B). Recall at fixed efSearch=50 falls with scale (99.75% at 10k, 98.14% at 100k, 94.70% at 1M), so efSearch must grow with N.

## Design decisions already made (do not undo)
- `HnswConfig.m_max`, `m_max0`, `m_l` are `Option` overrides, derived from `m` when `None` (`effective_m_max()`, `effective_m_max0()`, `effective_m_l()`). `HnswConfig.keep_pruned_shrink` is an `Option<bool>` override defaulting to `None` (which delegates to `keep_pruned` via `effective_keep_pruned_shrink()`). No sentinel-value guessing.
- `insert(id)` requires ids in order `0..n` and `matrix.rows() <= u32::MAX as usize`; anything else returns `InvalidArgument`.
- `HnswIndex::neighbors(layer, node)` returns `&[u32]` backed by contiguous flat storage (`layer0: Vec<u32>` with stride `m_max0 + 1` and `upper_arena: Vec<u32>` with stride `m_max + 1`).
- Phase 2 `overall_phase2_passed` = monotonic recall AND reproducibility (actually run) AND performance target (recall >= 0.98 and mean latency < 0.05 ms at one ef). Speedup vs brute force is reported, not gated.
- Sweep points with `ef < k` collapse to `ef == k` (the index clamps them).
- Distance ties break on `(distance, id)` everywhere. Keep it, since parity tests depend on it.
- Report flags: `reproducibility_checked`, `reproducibility_verified`, `measured_this_run` and the computed `phase3_completed` are part of the JSON contract now. Do not turn them back into constants.

## Known limitations
- Inner-product mode is only parity-tested. The diversity heuristic assumes distance-like (non-negative) values, which negative dot products violate. Do not report IP recall as validated.
- `siftsmall` saturates at 100% recall by ef=50, so it says little about graph quality. SIFT1M is the real test.
- `HnswIndex` borrows the matrix (`&'a RowMatrix<f32>`); there is no deletion or update.
- Graph storage uses flat `u32` adjacency (`layer0` stride `m_max0 + 1 = 33` slots, `upper_arena` stride `m_max + 1 = 17` slots, 8-byte `HnswNodeMeta`), costing 148.91 B/vector graph-only (152.91 B/vector incl. visited tags) on SIFT1M.

## How to move forward
Order matters. Do not skip ahead, and stop for Dev's review after each numbered step.

1. **Dev applies and commits the review patch**, runs `cargo test --release`, and pushes so CI runs on a fresh clone. *(Done)*
2. **Small Phase 3 follow-ups:** *(Completed & verified)*
   - real ground-truth check on SIFT1M (open flaw 2);
   - per-scale RSS in separate processes (open flaw 1);
   - repeat the efSearch sweep N times and report median and spread (open flaw 3);
   - regenerate `data/phase3_report.json` on Dev's machine and update the README Phase 3 table.
3. **Profile the 1M build before optimizing (Milestone M1 / Phase 3.5):** *(Completed & verified; see `results/phase3_5/LOG.md`)*
   - Profiled 100k and 1M builds using `--features instrument` (`search_layer` 78.83%, `shrink` 14.19%, `heuristic` 4.80% on 1M);
   - Tested `T1.5` (`seen_tags` in `HeuristicScratch`) and `T1.6` (`search_layer` heap scratch reuse) at 100k — both showed no improvement beyond baseline noise and were reverted (`"no gain"` logged);
   - Implemented and kept `T1.7` (`u32` ids and flat fixed-capacity adjacency in `layer0` + `upper_arena`), cutting graph-only memory by 56.0% (`338.67 -> 148.91 B/vector`), 1M Peak RSS by 199.7 MiB (`845.75 -> 646.05 MiB`), and 100k 3-run median build time by 10.37% (`30.299 s -> 27.157 s`);
   - Implemented and evaluated `T1.9` (`keep_pruned_shrink = Some(false)` behind config flag) at 100k (`21.351 s`, `-29.53%`) and 1M (`415.076 s`, `-18.81%`), recording the full `efSearch` sweep and matched-latency comparison in `results/phase3_5/LOG.md` for Dev's **Gate G1** decision.
4. **Every optimization keeps `--repro-check` passing and all tests green.** *(Verified: 84 Rust tests + 27 Python tests pass; golden 10k/100k fixtures and 1M `--repro-check` match bit-for-bit.)*
5. **Phase 4 (only after Dev signs off Phase 3 / Gate G1):** sweep M, efConstruction and efSearch and plot Recall@10 vs p99 latency. Use the same JSON conventions (hardware, dataset source, `measured_this_run`). Sweeps at 1M are slow (~8 min per build at M=16, efC=100), so propose the grid and a time budget to Dev before running anything, and consider 100k for the wide sweep and 1M for the final few points.

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
- [x] Regenerate `data/phase3_report.json` with the fixed binary and confirm it has `reproducibility_checked`, `measured_this_run` and a computed `phase3_completed`.
- [ ] CI: confirm green on a fresh clone after pushing.
- [ ] Dev signs off Phase 3.
- [x] Update `CLAUDE.md`, `CONTEXT_AND_MEMORY.md` and README test counts whenever tests are added.
