# Context & Memory: HNSW From Scratch in Rust

> **Purpose**: This file provides complete, token-efficient memory and architectural context for Claude (or any AI assistant/engineer) to resume work on this repository immediately without scanning the entire history.

---

## 1. Project Mission & Build Roadmap

Build a production-grade **Hierarchical Navigable Small World (HNSW)** Approximate Nearest Neighbor (ANN) search engine from scratch in **Rust** (`#![forbid(unsafe_code)]`), measure recall-vs-latency curves, and benchmark head-to-head against **FAISS**.

### Build Roadmap (7 Phases — Stop After Each)
- [x] **Phase 1: Brute-Force Exact $k$-NN Ground Truth & Recall Harness** *(COMPLETED & VERIFIED)*
- [x] **Phase 2: HNSW Insert + Search, Correctness-Checked on 10k Vectors** *(COMPLETED & VERIFIED)*
- [x] **Phase 3: Scale to 1M Vectors (`SIFT1M`), Profile Memory & Build Time** *(COMPLETED & VERIFIED — Gate G0)*
- [x] **Phase 3.5: Memory Optimization & Adjacency Flattening** *(COMPLETED & VERIFIED — Gate G1)*
- [x] **Phase 4: Parameter Sweep ($M$, $efConstruction$, $efSearch$), Plot Recall@10 vs. p99 Latency** *(COMPLETED & VERIFIED — Gate G3 signed off in [`AGENT_PLAN.md`](./AGENT_PLAN.md))*
- [x] **Phase 5: Filtered Search with Metadata Predicates (Including Selective Filters)** *(Milestone M3 `T3.1`–`T3.11` COMPLETED & VERIFIED)*
- [x] **Phase 6: Scalar (SQ8) or Product Quantization (PQ), Measure Recall Cost** *(Milestone M4 `T4.1`–`T4.9` & Gate G5 COMPLETED & VERIFIED)*
- [x] **Phase 7: Head-to-Head Benchmark vs. FAISS on Identical Hardware** *(Milestone M5 `T5.1`–`T5.7` & Gate G6 COMPLETED & VERIFIED)*

---

## 2. Environment & Machine Profile

- **Operating System**: Windows 11 / x86_64 (PowerShell)
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores)
- **RAM**: ~16 GB (15.8 GB usable)
- **Rust Toolchain**: `rustc` / `cargo` 1.99.1 (`#![forbid(unsafe_code)]`)
- **Python Environment**: Python 3.13.5 (`numpy`, `faiss-cpu`, `pytest`, `pytest-cov`, `ruff`, `mypy`)
---

## 3. Current Repository Architecture & File Map

```text
├── Cargo.toml                  # Rust crate manifest: 'vector-search' v0.1.0, binaries phase1..7, generate_metadata, filtered_exact_knn
├── docs/
│   └── HNSW_ALGORITHM.md       # Authoritative math & pseudocode spec (Algorithms 1-5)
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)])
│   ├── main.rs                 # CLI entry point for Phase 1 runner (`phase1`)
│   ├── hnsw.rs                 # HNSW multi-layer graph index (Algorithms 1-5, filtered search, compact u32 flat adjacency)
│   ├── bin/phase2.rs           # CLI entry point for Phase 2 runner (`phase2`)
│   ├── bin/phase3.rs           # CLI entry point for Phase 3 SIFT1M runner (`phase3`)
│   ├── bin/phase4.rs           # CLI entry point for Phase 4 parameter sweep runner (`phase4`)
│   ├── bin/phase5.rs           # CLI entry point for Phase 5 filtered benchmark CLI (`phase5`)
│   ├── bin/phase6.rs           # CLI entry point for Phase 6 SQ8/PQ quantization & parity CLI (`phase6`)
│   ├── bin/phase7.rs           # CLI entry point for Phase 7 Rust-vs-FAISS head-to-head CLI (`phase7`)
│   ├── bin/generate_metadata.rs # Deterministic Mode A/B predicate bitset CLI
│   ├── bin/filtered_exact_knn.rs # Multi-threaded exact filtered k-NN CLI
│   ├── matrix.rs               # Contiguous row-major RowMatrix<f32>
│   ├── distance.rs             # 8-way unrolled L2^2, Euclidean, Inner Product kernels
│   ├── rng.rs                  # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs              # Texmex .fvecs/.ivecs I/O, SIFT downloader, Gaussian & Mode A/B generators
│   ├── exact_knn.rs            # ExactKnn engine (single-query, parallel, filtered, timed)
│   ├── filter.rs               # DenseBitset, metadata predicates & FilteredSearchParams
│   ├── recall.rs               # Recall@k, verify_self_recall, latency percentiles
│   ├── hardware.rs             # System CPU / RAM / OS hardware detection
│   ├── benchmark.rs            # End-to-end benchmark orchestrator & JSON reporter
│   ├── phase4.rs               # Phase 4 resumable parameter sweep runner & prefix GT caching
│   ├── phase5.rs               # Phase 5 Option C matrix, versioned GT/bitset cache & sensitivity
│   ├── phase6.rs               # Phase 6 resumable matrix, microbench, streaming & isolated query RSS runner
│   ├── quantized/
│   │   ├── mod.rs              # Quantized module exports & metric distance/serialization helpers
│   │   ├── graph.rs            # Lifetime-free CompactHnswGraph, QuantizedDistanceSource, QuantizedSearchScratch
│   │   ├── sq8.rs              # SQ8 calibrator, Way A/B & symmetric kernels, streaming .fvecs builder, HnswSq8Index
│   │   └── pq.rs               # Deterministic PQ k-means++/Lloyd trainer, ADC/SDC kernels, HnswPqIndex
│   └── hnsw_phase1/            # Python reference implementation & BLAS GEMM baseline
├── scripts/
│   ├── benchmark_faiss.py      # Phase 7 FAISS IndexHNSWFlat benchmark & matched-recall interpolation harness
│   ├── plot_phase4.py          # Phase 4 Pareto & scaling chart generator
│   ├── plot_phase5.py          # Phase 5 filtered recall, latency & crossover chart generator
│   ├── plot_phase6.py          # Phase 6 quantization recall-vs-latency/QPS, memory & rerank chart generator
│   └── plot_phase7.py          # Phase 7 Rust-vs-FAISS Pareto, matched-recall ratio & scaling chart generator
├── tests/
│   ├── rust_phase1_tests.rs    # 16 Rust Phase 1 integration & edge-case tests
│   ├── rust_phase2_tests.rs    # 23 Rust Phase 2 HNSW correctness & recall tests
│   ├── adversarial_hnsw_tests.rs # 12 Rust HNSW stress & boundary tests
│   ├── adversarial_graph_fingerprint_tests.rs # 10 Rust graph fingerprint & mutation tests
│   ├── challenge_m1_p35_tests.rs # 8 Rust Phase 3.5 instrumentation, golden & layout tests
│   ├── rust_phase3_tests.rs    # 11 Rust Phase 3 report semantics, GT check & sweep tests
│   ├── rust_phase4_tests.rs    # 4 Rust Phase 4 grid parsing, GT caching & sweep tests
│   ├── rust_metadata_tests.rs  # 10 Rust Mode A/B predicate tests
│   ├── rust_filtered_knn_tests.rs # 7 Rust filtered exact k-NN tests
│   ├── rust_filtered_strategy1_tests.rs # 14 Rust S1 post-filter tests
│   ├── rust_filtered_strategy2_tests.rs # 13 Rust S2 in-traversal & eval_limit tests
│   ├── rust_filtered_strategy3_tests.rs # 14 Rust S3 pre-filter bitset tests
│   ├── rust_filtered_counters_tests.rs # 3 Rust instrumented filtered counter tests
│   ├── rust_phase5_tests.rs    # 4 Rust Phase 5 cache, crossover & matrix CLI tests
│   ├── rust_phase6_tests.rs    # 16 Rust Phase 6 CompactHnswGraph & SQ8 unit/integration tests
│   ├── rust_phase6_pq_tests.rs # 12 Rust Phase 6 PQ training, ADC/SDC & HnswPqIndex tests
│   ├── rust_phase6_cli_tests.rs # 7 Rust Phase 6 CLI, resume & parity-check tests
│   ├── rust_phase7_tests.rs    # 4 Rust Phase 7 Mode 1/2, -nkp & GT cache CLI tests
│   ├── test_phase1.py          # 23 Python reference tests
│   ├── test_rust_parity.py     # 4 Rust <-> Python <-> FAISS parity tests
│   ├── test_filtered_parity.py # 5 Filtered search & FAISS parity tests
│   ├── test_phase5_pipeline.py # 1 End-to-end Phase 5 CLI + resume + plot test
│   ├── test_phase6_parity.py   # 16 Python SQ8/PQ parity, rerank & plot_phase6 tests
│   └── test_phase7_harness.py  # 7 Python FAISS harness, affinity, FFI delta & plot_phase7 tests
├── results/
│   ├── phase3_5/               # Phase 3.5 golden fixtures, baseline/final reports & LOG.md
│   ├── phase4/                 # Phase 4 JSON reports, Pareto charts, PROPOSAL.md, G3_CHECKLIST.md
│   ├── phase5/                 # Phase 5 100k/1M JSONs, sensitivity, charts, DESIGN.md, PROPOSAL.md, FINDINGS.md
│   ├── phase6/                 # Phase 6 microbench, 11 100k + 9 1M arm/summary JSONs, charts_100k/charts_1m PNG+SVG, DESIGN.md, G5_DECISIONS.md, PROPOSAL.md, FINDINGS.md
│   └── phase7/                 # Phase 7 22 Rust/FAISS JSONs, matched_recall_summary.json, charts_100k/charts_1m PNG+SVG, DESIGN.md, G6_DECISIONS.md, FINDINGS.md
├── data/                       # SIFT corpora & gitignored Phase 4/5/6/7 GT/bitset caches (incl. gt_100000_k100.ivecs)
└── run_phase1.py               # Python reference runner
```

---

## 4. Phase 1 Accomplishments & Verified Invariants

1. **Algorithm Specification**: Written in `docs/HNSW_ALGORITHM.md` covering:
   - Skip-list hierarchy in metric spaces.
   - Probabilistic layer decay: $m_L = 1 / \ln M$, $l_{max} = \lfloor -\ln(U) \cdot m_L \rfloor$.
   - Degree bounds: $M$ for upper layers, $M_{max0} = 2M$ for layer 0.
   - Top-down 1-greedy zoom-in ($ef=1$) and layer 0 beam search with priority queues.
   - Small-world properties and $\mathcal{O}(\log N)$ logarithmic scaling proof.
   - Heuristic neighbor diversity selection (triangular inequality pruning).
2. **Deterministic Tie-Breaking**: Canonical `(distance, id)` ordering across all engines to prevent platform-dependent ground truth drift.
3. **Distance Precision**: Clamping to $\ge 0.0$ to eliminate floating-point non-associativity rounding negatives.
4. **Texmex I/O**: Bit-exact `.fvecs` (float32) and `.ivecs` (int32) readers/writers.
5. **SIFT Dataset Support**: Auto-downloads authentic INRIA Texmex `siftsmall` (10k vectors) / `SIFT1M` (1M vectors), with seeded Gaussian fallback (`seed=42`).
6. **Self-Verification Invariant**: `Recall@k(GT, GT) == 1.000000` (100.0%) strictly verified.
7. **FAISS Parity Alignment**: `1000/1000` (100.0%) exact neighbor ID match against `faiss.IndexFlatL2`.
8. **Test Suite Status (Phase 4 Milestone Snapshot; see §8 for current M0–M5 totals)**:
   - `cargo test`: **84/84 PASS** at Phase 4 (`16` Phase 1 + `23` Phase 2 + `12` adversarial_hnsw + `10` adversarial_graph_fingerprint + `8` challenge_m1_p35 + `11` Phase 3 + `4` Phase 4; siftsmall tests skip if `data/*.fvecs` is absent)
   - `pytest`: **27/27 PASS** at Phase 4 (with `faiss-cpu` installed; current suite is **223** default Rust + **11** `--features instrument` + **56** pytest in §8)

---

## 5. Phase 2 Verified Acceptance Results

- [x] Build HNSW index on 10,000 vectors (`M=16`, `efConstruction=100`, `m_L=1/ln(16)`).
- [x] Query 100 test vectors with $k=10$.
- [x] Monotonic recall scaling: Demonstrated that as $efSearch$ increases ($10 \to 50 \to 100 \to 200$), `Recall@10` scales monotonically from $\sim 0.80$ to $\ge 0.98+$.
- [x] Latency: mean query latency $< 0.05$ ms at Recall@10 $\ge 0.98$ (met at efSearch=20, 98.6% recall).
- [x] All tests pass under `#![forbid(unsafe_code)]` with zero warnings.

---

## 6. Phase 3 & 3.5 (Milestone M1) Verified Results

- [x] **Scale to 1M Vectors (`SIFT1M`)**: Built index on 1,000,000 128-d vectors in 450.3 s (2,220.7 vec/s).
- [x] **Bit-Exact Ground Truth Alignment**: Official Texmex `sift_groundtruth.ivecs` verified with `999/1000` ID overlap (1 distance boundary tie) and 100% boundary distance compliance.
- [x] **Adjacency Flattening & Memory Breakthrough (`T1.7`)**: Replaced pointer-heavy nested structures with flat contiguous `u32` slice storage (`layer0` stride `m_max0 + 1 = 33`, `upper_arena` stride `m_max + 1 = 17`).
  - Graph-only memory dropped **56.0%** from `338.67 B/vec` to **`148.91 B/vec`** (`-180.97 MiB` at 1M).
  - Isolated 1M peak RSS dropped from `845.75 MiB` to **`646.05 MiB`** (`-199.70 MiB`).
  - 100k build time dropped **10.37%** (`30.3 s -> 27.2 s`).
  - Preserved **100% bit-exact** graph fingerprints and query results.
- [x] **Gate G0 and G1 Signed Off**: Zero-cost instrumentation (`--features instrument`), reproducible 64-bit graph fingerprinting, and strict provenance tracking merged.

---

## 7. Phase 4 (Milestone M2) Verified Results & Current Status

- [x] **100k Wide Parameter Sweep (`T2.3`)**: Swept 23 configurations ($M \in \{8, 12, 16, 24, 32, 48\}$, $\text{efC} \in \{40, 100, 200, 400\}$ with $\text{efC} \ge M$) across a 12-point `efSearch` ladder $\{10..600\}$ on cached prefix ground truth.
- [x] **Publication-Grade Pareto Visualization (`T2.4`)**: Implemented `scripts/plot_phase4.py` (`matplotlib` only, 100% typed, 0 lints) producing 4 clean charts in PNG and SVG under `results/phase4/`:
  - `chart1_recall_vs_p99`: Recall@10 vs p99 query latency (log scale) with empirical Pareto frontier.
  - `chart2_recall_vs_qps`: Recall@10 vs single-thread QPS.
  - `chart3_build_time_vs_recall`: Construction throughput vs achievable recall at fixed latency.
  - `chart4_memory_vs_m`: Graph memory bytes per vector vs $M$ demonstrating strict scale invariance.
- [x] **1M Vector Confirmation (`T2.5`)**: Benchmarked 4 empirical Pareto points on full SIFT1M with 5 timed passes pinned to P-core `0x0004`:
  - $M=8, \text{efC}=100$: 266.5 s build, 85.4 B/vec graph, 20,150.7 QPS at ef=10, 0.9947 Recall@10 at ef=600.
  - $M=16, \text{efC}=100$ (Phase 3 baseline): 450.3 s build, 148.9 B/vec graph, 0.9806 Recall@10 at ef=100 (2,383.9 QPS, 0.55 ms p99). Bit-exact graph fingerprint `0x44044886c62bca5e`.
  - $M=32, \text{efC}=200$: 1,288.9 s build, 272.3 B/vec graph, 0.9828 Recall@10 at ef=50 (2,579.7 QPS), 0.9988 at ef=200.
  - $M=48, \text{efC}=400$: 2,828.0 s build, 402.4 B/vec graph, 0.9917 Recall@10 at ef=50 (1,918.8 QPS), 0.9992 at ef=200.
- [x] **Documentation & Claims Ledger (`T2.6`)**: Authored Phase 4 section in `README.md` and registered numbers in `results/CLAIMS.md` Section 10.

---

## 7.5. Phase 5 (Milestone M3) Verified Results

- [x] **Filtered Search & Provenance (`T3.1`–`T3.8`)**: Implemented `DenseBitset`, Mode A/B deterministic generators, `S1`/`S2`/`S3` filtered strategies, zero-cost `FilteredSearchCounters`, `DEFAULT_OPAQUE_EVAL_LIMIT = 16_384` (`FilteredSearchParams::default().eval_limit == None`), and versioned disk caches under `data/phase5/`.
- [x] **Option C Matrix & Sensitivity (`T3.9`)**: Executed all 24 `100k` cells + sensitivity (`a4621bc`, solo `0x0004`) and 24 `1M` cells + sensitivity (`c39c92b`, `0x0004/0x0010/0x0040/0x0100`). Mode A $\sigma^*_{0.95}$ shifts `0.079149` (`100k`) $\to$ `0.023799` (`1M`); Mode B shifts `0.252372` (`100k`) $\to$ `0.120868` (`1M`); `16_384` bounds `1M` Mode B `0.01` opaque `p99` from `53.729 ms` to `4.797 ms` (`11.20x`).
- [x] **Visualization & Claims Synthesis (`T3.10`–`T3.11`)**: Authored [`results/phase5/FINDINGS.md`](../../results/phase5/FINDINGS.md), generated PNG/SVG charts via `scripts/plot_phase5.py`, updated `README.md`, and logged Section 11 in `results/CLAIMS.md`.

---

## 7.6. Phase 6 — Milestone M4 Verified Results

- [x] **Isolated Quantized Module & Kernel Selection (`T4.1`–`T4.5`)**: Added `src/quantized/` (`CompactHnswGraph`, `HnswSq8Index`, `HnswPqIndex`) leaving `src/hnsw.rs` `f32` paths untouched (`PRESERVING`). On [`results/phase6/pilot/microbench_kernels.json`](../../results/phase6/pilot/microbench_kernels.json) (`dim=128`, `4,096` L2/L3-hot vectors, `0x0004`): Way B `sq8_norm_expanded` (`36.9335 ns`) runs `1.2760x` faster than Way A `sq8_query_shifted` (`47.1285 ns`), but suffers **`63.9910`** adversarial max abs error vs **`4.6115e-5`** (`1.39e6x` lower) and `23.0x` higher standard rel error (`3.9406e-6` vs `1.7156e-7`), so **Way A (`query_shifted`, `128 B/vec`)** is default. `pq16_adc` (`9.8310 ns`, `16 B/vec`, `4.03x`) and `pq32_adc` (`20.1700 ns`, `32 B/vec`, `1.97x`) beat raw `f32_l2_squared` (`39.6600 ns`, `512 B/vec`).
- [x] **100k Pilot & 1M Tier 1+2 Matrix (`T4.6`–`T4.8`, commit `9f79165`, `power_mode = "balanced"`)**: Executed all `11` `100k` arms (`reps=3`, solo `0x0004`, `620.80 s` build + `4,025.75 s` sweep = `77.44 min`) and `9` Tier 1+2 `1M` arms (`reps=5`, 2-P-core sharded `0x0004,0x0400`, `101.20 m` build + `142.86 m` sweep = `244.06 min` core-minutes, inside `252.89 min` conservative bound in [`results/phase6/PROPOSAL.md`](../../results/phase6/PROPOSAL.md)). All `Float32GraphThenQuantize` arms reproduced Phase 4 golden fingerprints (`0x80af03bd7aebaade` / `0xa57cab3db2d6766f` at `100k`; `0x44044886c62bca5e` / `0xb8ddb30d73d248c5` at `1M`).
- [x] **Streaming RAM, Payload Reduction & `QuantizedDirect` Parity**: Two-pass streaming `.fvecs` build (`sq8_only_qbuild`) cuts `build_peak_rss_mb` by **`44.2%`** at `100k` (`87.70 -> 48.90 MiB`) and **`64.4%`** at `1M` (`778.37 -> 277.29 MiB`; `-56.9%` vs `f32_m16` `642.71 MiB`), matching `sq8_only_f32graph` recall within `-0.00012` at `100k` `ef=100` (`0.98464` vs `0.98476`) and `+0.00024` at `1M` `ef=100` (`0.97220` vs `0.97196`). Isolated query-serving `PeakWorkingSet64` falls from `74.95` (`100k`) / `642.71 MiB` (`1M`) (`f32_m16`) to **`38.61 / 276.02 MiB`** (`sq8_only_f32graph`, `-48.5% / -57.1%`), **`29.22 / 184.67 MiB`** (`pq32_only_f32graph`, `-61.0% / -71.3%`), and **`28.13 / 169.55 MiB`** (`pq16_only_f32graph`, `-62.5% / -73.6%`); steady-state payload drops from `657.57 / 660.91 B/vec` (`f32_m16`) to `272.55 / 272.54 B/vec` (`SQ8-Only`), `177.84 / 176.67 B/vec` (`PQ32-Only`), and `161.84 / 160.67 B/vec` (`PQ16-Only`).
- [x] **Recall Ceilings, Top-$R$ Reranking & Visualization (`T4.9`)**:
  - `SQ8-Only M=16` peaks at `0.98778` (`100k`) / `0.98764` (`1M`); widening to `sq8_only_m32` (`400.22 / 400.28 B/vec`, `51.06 / 398.04 MiB` query RSS) reaches `0.99060` (`100k`) and `0.98898` (`1M`, with `ef >= 70` `balanced`/concurrent tail jitter disclosed).
  - `SQ8+Rerank` (`R=20`) restores `100%` of `FullBeam` and `f32` recall across `efSearch in [20..600]` (`100k`: `0.99370` @ `ef=100`, `0.99712` @ `ef=600` for `M=16`; `0.99938 -> 0.99999` for `M=32`; `1M`: `0.98060` @ `ef=100`, `0.99804` @ `ef=600`).
  - Unreranked `PQ16-Only` / `PQ32-Only` saturate at `0.61519 / 0.76461` (`100k`) and `0.54520 / 0.72305` (`1M`). With exact reranking (`+1.8%..+4.9%` payload), `pq16_rerank_f32graph` at `>= 0.99` recall beats `f32_m16` by **`+40.9%` QPS / `-30.7%` `p99`** at `100k` (`5,269.0` vs `3,738.9 QPS`, `ef=150, FB`) and **`+27.0%` QPS / `-40.2%` `p99`** at `1M` (`1,553.9` vs `1,223.5 QPS`, `0.8536` vs `1.4282 ms`, `ef=300, FB`); at `1M >= 0.995` recall, `pq32_rerank_f32graph` (`ef=300, FB`) reaches `0.99561` at **`994.0 QPS` (`p99 = 1.4300 ms`)** — **`+49.8%` faster** than Phase 6 `f32_m16` (`663.4 QPS`, `2.4524 ms`) and **`+9.1%` faster** than solo Phase 4 `f32_m16` (`911.0 QPS`, `1.4942 ms`).
  - Authored [`results/phase6/FINDINGS.md`](../../results/phase6/FINDINGS.md) and rendered PNG/SVG charts via `scripts/plot_phase6.py` under `results/phase6/`.

---

## 7.7. Phase 7 — Milestone M5 Verified Results

- [x] **Head-to-Head Protocol & SIMD Provenance (`T5.1`–`T5.5`, commit `9f79165`)**: Benchmarked Rust `HnswIndex` vs `faiss-cpu 1.15.1` `IndexHNSWFlat(128, M, METRIC_L2)` across `22` isolated-process reports (`11` at `100k` with `3` builds / `5` query reps using `data/gt_100000_k100.ivecs`, `11` at `1M` with `1` build / `5` query reps using official Texmex GT) covering `C1 (8, 100)`, `C2 (16, 100)`, `C3 (32, 200)`, `C2-nkp (16, 100, keep_pruned_shrink=false)`, and `C2` multi-thread (`T in {1, 6, 16}`). Stock PyPI `faiss-cpu 1.15.1` ships in **`SIMDLevel_NONE`** (`avx2_available == false`), matching default-target Rust SSE2 (`opt-level = 3, lto = "thin"`). All `14` single-thread runs verify `max |Mode 1 R@10 - Mode 2 R@10| == 0.0` and `100%` continuity with Phase 4 (`C1`/`C2`/`C3`) and Phase 3.5 (`C2-nkp` `1M`). Runs executed under `power_mode = "balanced"`, with single-thread `100k C2/C3` and `1M C1/C2/C3` paired concurrently across P-cores (`0x0004` Rust, `0x0400` FAISS) and `C2-nkp` / `T=6` / `T=16` solo.
- [x] **Single-Thread Matched-Recall (`results/phase7/matched_recall_summary.json`, zero extrapolation)**:
  - **`C1` (`M=8, efC=100`) Crossover**: Rust trails at `R*=0.90/0.95` (`0.8655x / 0.9058x` Mode 1 QPS at `100k`, `0.8121x / 0.9224x` at `1M`), then **overtakes FAISS at `R*=0.98+`**: **`1.0392x`** (`100k` `R*=0.98`) up to **`1.2582x` QPS / `1.1671x` lower `p99`** (`100k` `R*=0.995`), and **`1.2546x` Mode 1 QPS / `1.3793x` lower Mode 2 `p99`** (`1M` `R*=0.98`: `1,356.90` vs `1,081.52 QPS`, `1.0779` vs `1.4868 ms`). At `1M ef=600`, Rust `C1` reaches `0.994700` while FAISS `C1` saturates at `0.989550`.
  - **`C2` / `C3` Deficit & `C2-nkp` Root Cause**: Default Rust `C2` (`0.7039x–0.7987x` QPS at `100k`; `0.6241x–0.6781x` at `1M`) and `C3` (`0.7045x–0.8255x` at `100k`; `0.5023x–0.7223x` at `1M`) trail FAISS because `keep_pruned_shrink = None -> true` backfills reverse-shrink edges. Disabling reverse-shrink backfill (`C2-nkp`) cuts build time by `23.56%` at `100k` (`29.926 -> 22.876 s`) and `31.46%` at `1M` (`625.665 -> 428.827 s`), raises `ef=600` recall to `0.999890` (`100k`) and `0.999140` (`1M`, exceeding FAISS `C2`'s `0.998720`), lifts Rust `C2` Mode 1 QPS by `+5.7%..+38.8%` (`100k`) / `+17.0%..+53.0%` (`1M`), closes `100k` vs FAISS to `0.9208x` (`R*=0.99`) / `0.9772x` (`R*=0.995`), and **overtakes FAISS `C2` at `1M R*=0.995`** (**`1.0371x` QPS**: `1,107.94` vs `1,068.30`; **`1.5661x` lower `p99`**: `1.3535` vs `2.1197 ms`).
- [x] **OpenMP Build Loss, Peak RSS Savings, FFI Audit & Multi-Thread Scaling (`T5.6`–`T5.7`)**:
  - **Build Time (FAISS OpenMP Wins)**: Single-thread Rust `HnswIndex::build` (`18.163–79.651 s` at `100k`; `313.285–1,562.147 s` at `1M`) trails FAISS OpenMP `add()` (`3.339 s` at `100k T=16`, `76.360 s` at `1M T=16`) by **`8.96x` (`100k`) and `8.19x` (`1M` vs concurrent `0x0004` `625.665 s`)**.
  - **Isolated Peak RSS (Rust Wins)**: Graph-only memory matches within `+0.08%..+1.68%` (`1M C1` `+6.04%`: `85.44` vs `80.57 B/vec`), while Rust saves **`~92 MiB` (`1.99x–2.21x`)** at `100k` (`75.96–93.68` vs `167.89–186.09 MiB`) and **`~524–526 MiB` (`1.68x–1.88x`)** at `1M` (`594.31–776.42` vs `1,118.32–1,301.15 MiB`) by borrowing `&RowMatrix<f32>`.
  - **FFI Tax vs `p99` Margin**: FAISS `ffi_delta_us` median is `6.3050 us/q` (`100k`) / `7.9195 us/q` (`1M`; `7.0058 us/q` combined; `plot_phase7.py` `6.60 / 8.09 / 7.23 us/q`), while claimed Mode 2 `p99` margins (`95.6–766.2 us`) exceed FFI overhead by `12x–100x`.
  - **Multi-Thread Batch Search (`C2`, `T in {1, 6, 16}`)**: Rust `search_batch_parallel` scales `4.87x–6.35x` (`T=6`) / `6.74x–8.89x` (`T=16`) at `100k` and `4.83x–6.78x` (`T=6`) / `6.10x–8.41x` (`T=16`) at `1M`, delivering **`87.3%–93.9%` (`T=6`) and `83.5%–90.5%` (`T=16`) of FAISS OpenMP matched-recall Mode 1 QPS at `1M`** (`8,086.6–27,707.9 QPS` across `R* in {0.90..0.99}` at `T=16`, `5,386.5` at `R*=0.995`).
  - Authored [`results/phase7/FINDINGS.md`](../../results/phase7/FINDINGS.md) and rendered PNG/SVG charts via `scripts/plot_phase7.py` under `results/phase7/`.

---

## 8. Key Commands for Quick Verification

```powershell
# Run all default Rust unit & integration tests (223 tests: 35 in src/ + 188 across 18 tests/*.rs) and instrumented checks (11 tests: 8 + 3)
cargo test --release
cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_filtered_counters_tests

# Run Rust in release mode with clippy checks
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features instrument -- -D warnings

# Check code formatting
cargo fmt --all -- --check

# Run all Python reference, parity & harness tests (56 tests, 0 skipped)
python -m pytest tests -q -rs

# Run Python lint & type checks
ruff check . ; mypy src tests scripts

# Execute Phase 5 filtered benchmark matrix
cargo run --release --bin phase5 -- --matrix --scale 100000 --reps 5 --include-sensitivity --output-dir results/phase5

# Execute Phase 6 kernel microbenchmark, 100k/1M quantization matrices, and plot charts
cargo run --release --bin phase6 -- --microbench --output-dir results/phase6/pilot
cargo run --release --bin phase6 -- --matrix --scale 100000 --reps 3 --affinity 0x0004 --output-dir results/phase6
cargo run --release --bin phase6 -- --matrix --scale 1000000 --reps 5 --rerank-list 20,100,full_beam --affinity 0x0004 --output-dir results/phase6
python scripts/plot_phase6.py --results-dir results/phase6 --scale 1000000 --phase4-dir results/phase4 --output-dir results/phase6/charts_1m

# Execute Phase 7 Rust vs FAISS head-to-head runners and plot charts
cargo run --release --bin phase7 -- --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --affinity 0x0004 --profile-rss-isolated --json-out results/phase7/rust_n100000_C2.json
python scripts/benchmark_faiss.py --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --gt-path data/gt_100000_k100.ivecs --affinity 0x0400 --profile-rss-isolated --json-out results/phase7/faiss_n100000_C2.json
python scripts/plot_phase7.py --results-dir results/phase7 --output-dir results/phase7/charts_1m --scales 1000000
```

---

## 9. Critical Design Rules & Gotchas

1. **No Unsafe Code**: The Rust codebase enforces `#![forbid(unsafe_code)]`. Do not use `unsafe`.
2. **Canonical Ordering**: Whenever sorting top-$k$ results, use `(ordered_float(distance), node_id)` to guarantee bit-exact determinism across platforms.
3. **Cache-Friendly Visited Sets**: For high QPS, avoid allocating a `HashSet<u32>` per query. Use a reusable `VisitedTracker` with a query counter (`Vec<u32>` of generation tags) or a pre-allocated `BitVec`.
4. **Reference Specifications**: Always check `docs/HNSW_ALGORITHM.md` for pseudocode algorithms (Algorithms 1–5).
