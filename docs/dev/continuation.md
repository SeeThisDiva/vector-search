# continuation.md: handoff for coding agents

Read this first, then `CLAUDE.md` and `CONTEXT_AND_MEMORY.md`. This file is the most recent state; if they disagree, this wins.

## Project in one line
From-scratch HNSW ANN search in Rust (`#![forbid(unsafe_code)]`, std only), benchmarked against brute force and FAISS. **Milestones M0–M5 (Phases 1–7) are complete (`T0.1`–`T5.7`, except optional `T1.8` intentionally skipped):** Phase 6 quantization (`SQ8` & `PQ` in [`results/phase6/FINDINGS.md`](../../results/phase6/FINDINGS.md) & [`results/phase6/PROPOSAL.md`](../../results/phase6/PROPOSAL.md)) and Phase 7 head-to-head vs FAISS `IndexHNSWFlat` ([`results/phase7/FINDINGS.md`](../../results/phase7/FINDINGS.md) & [`results/phase7/matched_recall_summary.json`](../../results/phase7/matched_recall_summary.json)) are implemented, benchmarked at `100k`/`1M` on commit `9f79165`, plotted, and reconciled.

## Working agreement
- Dev designs, agents code. Do not start a later phase until the current one is reviewed.
- Free tools and datasets only.
- Never claim a number you did not measure. Every benchmark claim must come from a run whose JSON records hardware and dataset source.
- Do not weaken a pass condition to make it green. If a target is missed, report the miss.
- A check that did not run must never be reported as passed. (This rule exists because of review finding 1 below.)

## Current verified state
- `cargo test --release`: **231 pass** (`43` lib unit + `188` across `18` `tests/*.rs`: `16` `rust_phase1` + `23` `rust_phase2` + `12` `adversarial_hnsw` + `10` `adversarial_graph_fingerprint` + `8` `challenge_m1_p35` + `11` `rust_phase3` + `4` `rust_phase4` + `10` `rust_metadata` + `7` `rust_filtered_knn` + `14` `rust_filtered_strategy1` + `13` `rust_filtered_strategy2` + `14` `rust_filtered_strategy3` + `3` `rust_filtered_counters` + `4` `rust_phase5` + `16` `rust_phase6` + `12` `rust_phase6_pq` + `7` `rust_phase6_cli` + `4` `rust_phase7`).
- `cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_phase3_tests --test rust_filtered_counters_tests`: **22 pass** (`8 + 11 + 3`).
- `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` (both default and `--features instrument`) clean.
- `pytest`: **56 pass** (`23` `test_phase1` + `4` `test_rust_parity` + `5` `test_filtered_parity` + `1` `test_phase5_pipeline` + `16` `test_phase6_parity` + `7` `test_phase7_harness`, 0 skipped); `ruff check .` and `mypy src tests scripts` clean.
- Needs Rust >= 1.88 (`rust-version` in `Cargo.toml`; `as_chunks` and lockfile v4).
- Git: Phase 1 to 3, the Phase 3 review fixes (`review_fixes.patch`, commit `d4b6fed` + `9e3c1a2`), the Phase 3 follow-up fixes, and **Phase 3.5 (Milestone M1)** are implemented and verified:
  - `T1.1` / `T1.4`: Zero-cost `--features instrument` phase timing and distance evaluation counters (`search_layer` 78.83%, `shrink` 14.19%, `heuristic` 4.80% on 1M).
  - `T1.2` / `T1.3`: Logical `HnswIndex::graph_fingerprint() -> u64` (including `enter_point`) and golden fixtures (`results/phase3_5/golden_10k.json`, `results/phase3_5/golden_100k.json`) with bit-exact distance/ID verification.
  - `T1.5` (`seen_tags` in `HeuristicScratch`) and `T1.6` (`search_layer` binary heap scratch reuse): tested at 100k, reverted and logged `"no gain"` in `results/phase3_5/LOG.md`.
  - `T1.7` (`u32` flat adjacency storage in `layer0` and `upper_arena`, 8-byte `HnswNodeMeta`, `HnswIndex::neighbors(layer, node) -> &[u32]`): **KEPT** (`d6d36ff`). Cuts graph-only memory by **56.0%** at 1M (`338.67 -> 148.91 B/vector`, `-180.97 MiB`), cuts isolated 1M Peak RSS from `845.75 MiB` to **`646.05 MiB`** (`-199.70 MiB`), and speeds up 100k 3-run median build time by **10.37%** (`30.299 s -> 27.157 s`) while keeping 100% bit-exact graph fingerprints and query results.
  - `T1.9` (`HnswConfig::keep_pruned_shrink: Option<bool>`, default `None`, CLI `--no-keep-pruned-shrink`): evaluated at 100k (`21.351 s`, `-29.53%`) and 1M (`415.076 s`, `-18.81%` vs `511.271 s` baseline; `-11.87%` vs `470.966 s` historical Phase 3) and recorded in `results/phase3_5/LOG.md` + `results/phase3_5/exp_no_keep_pruned_shrink_1m.json` for Dev's Gate G1 decision.
  - `T1.10`: `data/phase3_report.json` (`results/phase3_5/final_1m_report.json`) and `data/baseline_profile_report.json` regenerated from a clean commit on the final code (`366f185`).
  - `T2.1` / `T2.2`: Measured 100k pilot ($M=16, \text{efC}=100$, commit `26a9807`: build 26.066 s, 12-point sweep 173.49 s, Recall@10 0.82103 to 0.99702), wrote `results/phase4/PROPOSAL.md` (23-config grid, ~95 min expected 100k budget, ~93 min 1M finals budget), implemented resumable `phase4` runner & CLI (84 Rust tests, 27 Python tests).

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

Phase 6 (`100k` 11 arms solo `0x0004`; `1M` 9 Tier 1+2 arms 2-P-core sharded `0x0004,0x0400`, `9f79165`, `power_mode = "balanced"`):
- Kernel microbench (`pilot/microbench_kernels.json`): Way A `sq8_query_shifted` (`47.1285 ns`, `4.6115e-5` adversarial abs error) adopted as default over Way B `sq8_norm_expanded` (`36.9335 ns`, `63.9910` adversarial abs error); `pq16_adc` `9.8310 ns` (`4.03x` vs `f32` `39.6600 ns`), `pq32_adc` `20.1700 ns` (`1.97x`).
- Streaming `sq8_only_qbuild` (`QuantizedDirect`) cuts `build_peak_rss_mb` to `48.90 MiB` (`100k`, `-44.2%`) and `277.29 MiB` (`1M`, `-64.4%` vs `778.37 MiB`), matching `sq8_only_f32graph` recall within `-0.00012` (`100k` `ef=100`) / `+0.00024` (`1M` `ef=100`). Isolated query-serving `PeakWorkingSet64` drops from `642.71 MiB` (`1M f32_m16`) to `276.02 MiB` (`sq8_only_f32graph`), `184.67 MiB` (`pq32_only_f32graph`), and `169.55 MiB` (`pq16_only_f32graph`). `SQ8+Rerank R=20` recovers 100% of `FullBeam`/`f32` recall (`1M` `0.98060` @ `ef=100`, `0.99804` @ `ef=600`); `pq16_rerank_f32graph` beats `f32_m16` at `>=0.99` recall (`+40.9%` QPS at `100k`, `+27.0%` QPS / `-40.2%` `p99` at `1M`), and `1M pq32_rerank_f32graph` (`ef=300, FB`) hits `0.99561` at `994.0 QPS` (`+49.8%` vs Phase 6 `f32_m16`, `+9.1%` vs solo Phase 4 `f32_m16`).

Phase 7 (`100k` & `1M` Rust vs `faiss-cpu 1.15.1` `IndexHNSWFlat`, `22` reports on `9f79165`, `SIMDLevel_NONE` baseline SSE2 parity, `power_mode = "balanced"`):
- Single-thread matched recall (`results/phase7/matched_recall_summary.json`): Rust `C1` (`M=8, efC=100`) overtakes FAISS at `R*=0.98+` (`1.0392x..1.2582x` Mode 1 QPS at `100k`; `1.2546x` QPS / `1.3793x` lower `p99` at `1M R*=0.98`). Default `C2` (`0.6241x–0.7987x`) and `C3` (`0.5023x–0.8255x`) trail FAISS due to reverse-shrink backfill; `C2-nkp` (`keep_pruned_shrink=false`) cuts `100k`/`1M` build by `23.56%`/`31.46%` (`22.876 s` / `428.827 s`), reaches `0.999890`/`0.999140` recall, closes `100k` to `0.9772x` (`R*=0.995`), and overtakes FAISS `C2` at `1M R*=0.995` (`1.0371x` Mode 1 QPS, `1.5661x` lower Mode 2 `p99`).
- Asymmetry & scaling: FAISS OpenMP `add()` builds `8.96x` (`100k T=16`: `3.339 s`) and `8.19x` (`1M T=16`: `76.360 s` vs concurrent `0x0004` `625.665 s`) faster; Rust saves `~92 MiB` (`100k`) / `~524–526 MiB` (`1M`) isolated Peak RSS (`1.68x–2.21x`); multi-thread `C2` (`T=6, 16`) reaches `87.3%–93.9%` (`T=6`) and `83.5%–90.5%` (`T=16`) of FAISS `1M` Mode 1 QPS.

## Design decisions already made (do not undo)
- `HnswConfig.m_max`, `m_max0`, `m_l` are `Option` overrides, derived from `m` when `None` (`effective_m_max()`, `effective_m_max0()`, `effective_m_l()`). `HnswConfig.keep_pruned_shrink` is an `Option<bool>` override defaulting to `None` (which delegates to `keep_pruned` via `effective_keep_pruned_shrink()`). No sentinel-value guessing.
- `insert(id)` requires ids in order `0..n` and `matrix.rows() <= u32::MAX as usize`; anything else returns `InvalidArgument`.
- `HnswIndex::neighbors(layer, node)` returns `&[u32]` backed by contiguous flat storage (`layer0: Vec<u32>` with stride `m_max0 + 1` and `upper_arena: Vec<u32>` with stride `m_max + 1`).
- Phase 2 `overall_phase2_passed` = monotonic recall AND reproducibility (actually run) AND performance target (recall >= 0.98 and mean latency < 0.05 ms at one ef). Speedup vs brute force is reported, not gated.
- Sweep points with `ef < k` collapse to `ef == k` (the index clamps them).
- Distance ties break on `(distance, id)` everywhere. Keep it, since parity tests depend on it.
- Report flags: `reproducibility_checked`, `reproducibility_verified`, `measured_this_run` and the computed `phase3_completed` are part of the JSON contract now. Do not turn them back into constants.
- `CompactHnswGraph` (`src/quantized/graph.rs`) reproduces the `T1.7` flat `u32` adjacency layout in a lifetime-free owned struct: `from_hnsw_index` clones topology via read-only public accessors to keep `src/hnsw.rs` `PRESERVING`-untouched (`graph_fingerprint()` bit-exact), while `build_with_source` constructs graphs directly over any `QuantizedDistanceSource`.
- Way A `Sq8KernelKind::QueryShifted` (`128 B/vec`, `0` aux bytes) is the default SQ8 kernel over Way B `NormExpanded` (`+4 B/vec`) to avoid `f32` catastrophic cancellation (`4.6115e-5` vs `63.9910` adversarial max abs error).
- PQ training defaults (`PqConfig`): `K = 256` (`PQ_K_CENTROIDS`), `PQ_DEFAULT_MAX_TRAIN_VECTORS = 65_536` cap (`N_train = min(N, 65_536)`), `max_lloyd_iters = 25`, deterministic `Xoshiro256**` `seed = 42`, `M_pq in {16, 32}`; temporary `PqWithSdc` lookup tables are dropped via `into_pq_vectors()` right after `QuantizedDirect` graph construction so steady-state `PQ16`/`PQ32` never retain them.
- Quantized top-$R$ rerank enforces `effective_ef = ef_search.max(R).max(k)` (`CompactHnswGraph::effective_ef_search`) so low-`efSearch` rerank curves never collapse below `R` or `k`.
- Phase 7 Windows process affinity pinning (`scripts/benchmark_faiss.py`) uses explicit 64-bit `ctypes.wintypes` `SetProcessAffinityMask` / `GetProcessAffinityMask` signatures with post-set verification (`masked == requested`), and isolates Mode 1 pure batch (`idx.search` / `search_batch_parallel`) from Mode 2 per-query (`idx.search_c` / `search_batch_timed_with_tracker`) with explicit `ffi_delta_us` (`instant_overhead_us` on Rust).

## Known limitations
- Inner-product mode is only parity-tested. The diversity heuristic assumes distance-like (non-negative) values, which negative dot products violate. Do not report IP recall as validated.
- `siftsmall` saturates at 100% recall by ef=50, so it says little about graph quality. SIFT1M is the real test.
- `HnswIndex` borrows the matrix (`&'a RowMatrix<f32>`); there is no deletion or update.
- Graph storage uses flat `u32` adjacency (`layer0` stride `m_max0 + 1 = 33` slots, `upper_arena` stride `m_max + 1 = 17` slots, 8-byte `HnswNodeMeta`), costing 148.91 B/vector graph-only (152.91 B/vector incl. visited tags) on `HnswIndex` SIFT1M (`144.53 B/vec` on `CompactHnswGraph` without upper slack).
- Rust `HnswIndex::build` is strictly single-threaded (`1`-thread deterministic) vs FAISS OpenMP `add()` (`8.96x` faster at `100k T=16`, `8.19x` faster at `1M T=16` vs `0x0004`).
- Phase 7 benchmarks compare stock PyPI `faiss-cpu 1.15.1` (`SIMDLevel_NONE`, `avx2_available == false`) against default-target Rust SSE2 (`opt-level = 3, lto = "thin"`, no `.cargo/config.toml`); an explicit AVX2 addendum (`-C target-cpu=native` vs AVX2 FAISS) remains unmeasured optional work.
- All Phase 6 and Phase 7 runs (`9f79165`) ran under Windows `power_mode = "balanced"` (not `Best Performance`), and `1M` Phase 6 (`0x0004` + `0x0400` 2-P-core sharding) and single-thread Phase 7 `100k C2/C3` + `1M C1/C2/C3` (`0x0004` Rust + `0x0400` FAISS concurrently) shared L3/DRAM bandwidth, causing higher `1M C2` concurrent build time (`625.665 s` vs `450.301 s` Phase 4 solo) and occasional `p99` tail spikes (e.g., `sq8_only_m32` `1M ef >= 70`).

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
4. **Every optimization keeps `--repro-check` passing and all tests green.** *(Verified at M1 gate: 84 Rust + 27 Python tests; currently 223 default Rust + 11 instrument + 56 Python tests pass, golden 10k/100k fixtures and 1M `--repro-check` match bit-for-bit.)*
5. **Phase 4:** T2.1–T2.6 complete (100k 23-config grid + 1M 4-point confirmations, charts, CLAIMS.md §10; see `results/phase4/G3_CHECKLIST.md`).
6. **Phase 5 (Milestone M3, `T3.1`–`T3.11`):** Complete (`b9fc648`).
   - Added `DenseBitset`, Mode A/B deterministic metadata generators, filtered `ExactKnn`, `S1`/`S2`/`S3` filtered HNSW strategies, `DEFAULT_OPAQUE_EVAL_LIMIT = 16_384` + `with_opaque_eval_limit()` (keeping `FilteredSearchParams::default().eval_limit == None`), zero-overhead `FilteredSearchCounters`, versioned `.bin`/`.ivecs`/`.fvecs` caches (`data/phase5/`), Option C benchmark suite (`100k` solo `0x0004` + `1M` sharded across P-cores `0x0004/0x0010/0x0040/0x0100`, `48` cells + `100k`/`1M` sensitivity), charts (`scripts/plot_phase5.py`), [`results/phase5/FINDINGS.md`](../../results/phase5/FINDINGS.md), and [`results/CLAIMS.md`](../../results/CLAIMS.md#L292) §11.
   - Observed `0.95` recall crossovers $\sigma^*$: `100k` Mode A `0.079149` -> `1M` `0.023799` ($3.33\times$ shift); `100k` Mode B `0.252372` -> `1M` `0.120868` ($2.09\times$ shift).
7. **Phase 6 (Milestone M4, `T4.1`–`T4.9`):** Complete (`416c8f9`, `1074e1d`, `9f79165`).
   - Added `src/quantized/` (`CompactHnswGraph`, `Sq8Quantizer`/`HnswSq8Index`, `PqTrainer`/`HnswPqIndex`), `src/phase6.rs`, `src/bin/phase6.rs`, and `scripts/plot_phase6.py` while keeping `src/hnsw.rs` `f32` paths untouched.
   - Executed `pilot/microbench_kernels.json`, `11` `100k` arms (solo `0x0004`), and `9` Tier 1+2 `1M` arms (sharded `0x0004,0x0400`, `244.06 min` core-minutes within `252.89 min` conservative bound in [`results/phase6/PROPOSAL.md`](../../results/phase6/PROPOSAL.md)), authoring [`results/phase6/FINDINGS.md`](../../results/phase6/FINDINGS.md) and [`results/CLAIMS.md`](../../results/CLAIMS.md) §12.
8. **Phase 7 (Milestone M5, `T5.1`–`T5.7`):** Complete (`388e601`, `75faa5e`, `9f79165`).
   - Added `src/bin/phase7.rs`, [`scripts/benchmark_faiss.py`](../../scripts/benchmark_faiss.py), and [`scripts/plot_phase7.py`](../../scripts/plot_phase7.py); executed `22` isolated-process reports (`11` at `100k`, `11` at `1M`) under stock `SIMDLevel_NONE` baseline SSE2 parity, generated [`results/phase7/matched_recall_summary.json`](../../results/phase7/matched_recall_summary.json), authored [`results/phase7/FINDINGS.md`](../../results/phase7/FINDINGS.md), and logged [`results/CLAIMS.md`](../../results/CLAIMS.md) §13.
9. **Next — Final CI Push & Optional Future Work:** Push M4/M5 commits + reconciled docs/claims and confirm `.github/workflows/ci.yml` is green on a fresh clone; Dev decides open questions in [`AGENT_PLAN.md`](./AGENT_PLAN.md) §7 (optional Phase 7 AVX2 addendum `-C target-cpu=native` vs AVX2 FAISS, `T1.8`, parallel build, handoff docs visibility).

## Commands
```bash
cargo test --release
cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_filtered_counters_tests
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features instrument -- -D warnings
cargo fmt --all -- --check
cargo run --release --bin phase2 -- --dataset siftsmall --json-out results/phase3/phase2_report.json
cargo run --release --bin phase3 -- --dataset sift1m --json-out results/phase3/phase3_report.json
cargo run --release --bin phase5 -- --matrix --scale 100000 --reps 5 --include-sensitivity --output-dir results/phase5
cargo run --release --bin phase6 -- --microbench --output-dir results/phase6/pilot
cargo run --release --bin phase6 -- --matrix --scale 100000 --reps 3 --affinity 0x0004 --output-dir results/phase6
cargo run --release --bin phase6 -- --matrix --scale 1000000 --reps 5 --rerank-list 20,100,full_beam --affinity 0x0004 --output-dir results/phase6
python scripts/plot_phase6.py --results-dir results/phase6 --scale 1000000 --phase4-dir results/phase4 --output-dir results/phase6/charts_1m
cargo run --release --bin phase7 -- --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --affinity 0x0004 --profile-rss-isolated --json-out results/phase7/rust_n100000_C2.json
python scripts/benchmark_faiss.py --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --gt-path data/gt_100000_k100.ivecs --affinity 0x0400 --profile-rss-isolated --json-out results/phase7/faiss_n100000_C2.json
python scripts/plot_phase7.py --results-dir results/phase7 --output-dir results/phase7/charts_1m --scales 1000000
python -m pytest tests -q -rs && ruff check . && mypy src tests scripts
```
Tip for quick iteration: release builds use `lto = "thin"` and `codegen-units = 1`, which is slow to compile. For test runs only, `CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo test --release` compiles in about 10 s. Never use those overrides for numbers you report.

## Housekeeping checklist
- [x] Apply and commit the review patch.
- [x] Regenerate `data/phase3_report.json` with the fixed binary and confirm it has `reproducibility_checked`, `measured_this_run` and a computed `phase3_completed`.
- [x] Dev signs off Phase 3 (Gate G0).
- [x] Dev signs off Phase 3.5 optimization decisions (Gate G1).
- [x] Dev approves Phase 4 grid & time budget (Gate G2).
- [x] Dev signs off Phase 4 evaluation & Pareto write-up (Gate G3).
- [x] Dev approves Phase 5 design & Option C matrix (`Gate G4` + `Gate G4.5`).
- [x] Dev approves Phase 6 Quantization design (`Gate G5`, [`results/phase6/G5_DECISIONS.md`](../../results/phase6/G5_DECISIONS.md)).
- [x] Dev approves Phase 7 FAISS protocol (`Gate G6`, [`results/phase7/G6_DECISIONS.md`](../../results/phase7/G6_DECISIONS.md)).
- [x] Complete Phase 6 Quantization (`M4 T4.1`–`T4.9`: `SQ8`/`PQ` kernels, `100k`/`1M` matrices, [`results/phase6/PROPOSAL.md`](../../results/phase6/PROPOSAL.md), [`results/phase6/FINDINGS.md`](../../results/phase6/FINDINGS.md), `plot_phase6.py` charts).
- [x] Complete Phase 7 vs FAISS (`M5 T5.1`–`T5.7`: `phase7` + `benchmark_faiss.py` `100k`/`1M` runs, [`results/phase7/matched_recall_summary.json`](../../results/phase7/matched_recall_summary.json), [`results/phase7/FINDINGS.md`](../../results/phase7/FINDINGS.md), `plot_phase7.py` charts).
- [x] Reconcile documentation (`README.md`, `CLAUDE.md`, `AGENT_PLAN.md`, `CONTEXT_AND_MEMORY.md`, `continuation.md`) and [`results/CLAIMS.md`](../../results/CLAIMS.md) (§12–§13).
- [ ] CI: confirm green on a fresh clone after pushing.
- [ ] Optional AVX2 addendum (Phase 7: `-C target-cpu=native` vs AVX2 FAISS build).
- [x] Update `CLAUDE.md`, `CONTEXT_AND_MEMORY.md` and README test counts whenever tests are added.
