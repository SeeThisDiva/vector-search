# Milestone M5 / Phase 7: Head-to-Head Benchmark Design Specification (`vector-search` Rust vs FAISS `IndexHNSWFlat`)

> **Document Status**: DRAFT protocol & architecture specification for **Gate G6** review.  
> **Scope**: Milestone M5 (Phase 7, Gate G6). Strictly `IndexHNSWFlat` (float32 L2 squared) head-to-head comparison; quantized HNSW (`IndexHNSWSQ` / Milestone M4 Phase 6) is out of scope here.  
> **Ground Rules**: Per the benchmark ground rules, no unmeasured `100k`/`1M` FAISS numbers are claimed or assumed here. Any empirical FAISS metrics shown in Section 2 are explicitly labeled **Environment Verification Sanity Probes (`siftsmall` 10k / synthetic 100k)**, not published benchmark results.

---

## 1. Objectives & Fairness Principles

Phase 7 answers one clear engineering question: **How does our from-scratch, `#![forbid(unsafe_code)]`, `std`-only Rust HNSW implementation compare against Meta's industry-standard C++ FAISS `IndexHNSWFlat` on the exact same machine, dataset, ground truth, and metric?**

Per the Phase 7 benchmark protocol, this benchmark is governed by five non-negotiable fairness principles:

1. **Same Machine, Dataset, Query Set, $k$, and Ground Truth**: Authentic INRIA Texmex `SIFT1M` ($100\text{k}$ prefix and full $1\text{M}$ base, $128$-d `float32`, $10,000$ queries, $k = 10$, squared Euclidean $L_2^2$).
2. **Matched-Recall Evaluation (Never Equal-`efSearch`)**: Because FAISS and Rust build structurally different graphs (`keep_pruned` backfill, shrink headroom, RNG level sequences), identical `efSearch` values yield different `Recall@10`. Every latency, QPS, and p99 comparison is evaluated **at matched `Recall@10`** along full Pareto curves and fixed recall targets.
3. **FFI Overhead Isolation**: Python calls FAISS through SWIG C++ bindings. We separate true C++ batch execution (`idx.search(Q, k)`) from per-query Python/SWIG call timing (`idx.search_c(1, ...)`) and explicitly measure and report the per-query FFI call overhead so Rust never "wins" on Python wrapper tax.
4. **Thread-Count Symmetry & Asymmetry Disclosure**: Primary evaluation pins both Rust and FAISS (`faiss.omp_set_num_threads(1)`) to a single physical P-core (`0x0004`). Secondary multi-thread evaluation benchmarks parallel batch query throughput (`HnswIndex::search_batch_parallel` via `std::thread::scope` vs FAISS OpenMP) and discloses honestly that Rust index construction is currently single-threaded while FAISS `add()` uses OpenMP by default.
5. **No Hidden Tuning or Weakened Losses**: Report where Rust loses (e.g., multi-thread OpenMP build wall-clock, C++ memory/distance-kernel gaps if any) as plainly as where Rust matches or wins.

---

## 2. Verified Local FAISS Environment & Windows Hardware Topology

All environment facts below were directly observed via read-only probes in the workspace before authoring this design.

### 2.1 Host Hardware & Windows Hybrid Core Mapping

Observed via `src/hardware.rs`, Phase 4 JSON reports, and Windows `GetLogicalProcessorInformationEx(RelationProcessorCore)`:

| Property | Observed Value | Notes & Relevance |
|---|---|---|
| **CPU Model** | `12th Gen Intel(R) Core(TM) i7-12650H` | Hybrid Alder Lake: 10 physical cores / 16 logical threads |
| **P-Cores (`EfficiencyClass = 1`)** | 6 physical cores (`Flags = 1`, SMT) | CPUs `0..11`, combined mask `0x0FFF`; per-core masks: `0x0003`, `0x000c`, `0x0030`, `0x00c0`, `0x0300`, `0x0c00` |
| **E-Cores (`EfficiencyClass = 0`)** | 4 physical cores (`Flags = 0`, no SMT) | CPUs `12..15`, combined mask `0xF000`; per-core masks: `0x1000`, `0x2000`, `0x4000`, `0x8000` |
| **Primary Single-Core Mask** | **`0x0004`** (CPU 2, P-core 1 SMT0) | Matches Phase 3.5, Phase 4, and Phase 5; avoids CPU 0 (`0x0001`) OS DPC/interrupt noise |
| **6-P-Core Non-SMT Mask** | **`0x0555`** (CPUs `0, 2, 4, 6, 8, 10`) | 1 thread per physical P-core for clean multi-thread scaling without E-core stragglers or SMT contention |
| **System RAM** | `15.78 GiB` (`16 GB` physical) | Safe margin for single-process `1M` build/search (`< 1 GiB` per process) |
| **Power Scheme** | Currently observed `Balanced` (`381b4222-...`) | Protocol requires **AC plugged in + Best/High Performance** before timed runs (`powercfg /getactivescheme` stamped in JSON) |
| **Repo Location / OneDrive** | Local working copy | Pause OneDrive sync (or run from a clean non-synced clone) and quiesce background apps during timed runs |

### 2.2 Windows Process Affinity Pinning in Python (`ctypes` 64-Bit Trap & Fix)

A read-only probe caught a critical 64-bit Windows `ctypes` failure mode: calling `kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 0x0004)` **without** explicit `restype` and `argtypes` passes a truncated 32-bit pseudo-handle and silently fails with `WinError 6` (`ERROR_INVALID_HANDLE`).

The Phase 7 Python runner **must** declare explicit 64-bit `ctypes.wintypes` signatures and assert post-set verification:

```python
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetCurrentProcess.argtypes = []
kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
kernel32.GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
```

Verified probe output with these signatures: `Affinity before: 0xffff -> SetProcessAffinityMask(0x0004) ok=1 -> GetProcessAffinityMask: 0x4 (err=0)`. The exact verified hex mask is stamped into every JSON `environment.processor_affinity_mask`.

### 2.3 Installed FAISS Binary & SIMD Provenance

Inspected directly from the local Python environment (`Python 3.13.5`, `MSC v.1943 64 bit (AMD64)`):

| Field | Observed Value | Provenance / Diagnostic Meaning |
|---|---|---|
| **Distribution & Version** | `faiss-cpu 1.15.1` (`faiss.__version__ == "1.15.1"`) | PyPI wheel `cp313-cp313-win_amd64` (`scikit-build-core 1.0.3`, `delvewheel 1.13.1`) |
| **Shipped Binaries** | `_swigfaiss.pyd`, `faiss.dll`, `libopenblas.dll`, `msvcp140.dll`, `vcomp140.dll` | **No** `swigfaiss_avx2.py` or `swigfaiss_avx512.py` in wheel |
| **`faiss.get_compile_options()`** | `""` (empty string) | Wheel does not populate compile-option string |
| **`faiss.loader` Dispatch Log** | `ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")` -> falls back to `faiss.swigfaiss` | Host CPU `supported_instruction_sets()` reports `AVX2`, `FMA3`, `BMI2`, `X86_V3`, **not** AVX-512 (Alder Lake consumer silicon) |
| **`faiss.SIMDConfig`** | `has_dynamic_dispatch() == False`, `get_level() == 0`, `get_level_name() == "NONE"`, `compiled_simd_levels() == 1` (`0b1`), `is_simd_level_available(SIMDLevel_AVX2) == False` | **Crucial finding:** the stock PyPI `faiss-cpu 1.15.1` Windows wheel is compiled in static **`SIMDLevel_NONE`** mode (baseline x86-64 SSE2), **without** FAISS's explicit AVX2 kernels! |
| **OpenMP Default vs Pinned** | `faiss.omp_get_max_threads()` defaults to `16`; `faiss.omp_set_num_threads(1)` reduces it to `1` | Must explicitly call `faiss.omp_set_num_threads(1)` **before** index build and single-thread search |
| **Rust Codegen Comparison** | `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, no `.cargo/config.toml` | Default Windows x86_64 target (host triple was not recorded in the result JSONs; current builds use `x86_64-pc-windows-gnu`; baseline SSE2 + LLVM auto-vectorization over 8-way unrolled `as_chunks::<8>()` in `src/distance.rs`, `#![forbid(unsafe_code)]`) |

**Implication for Gate G6:** Stock `faiss-cpu 1.15.1` (`SIMDLevel_NONE`) vs default-target Rust (host triple was not recorded in the result JSONs; current builds use `x86_64-pc-windows-gnu`) is actually a **fair baseline-SIMD comparison** (neither binary uses AVX2). Measuring hand-written C++ AVX2 vs safe Rust auto-vectorization (`-C target-cpu=native`) is listed as an optional **AVX2 Addendum Arm** in Section 8 for Gate G6 sign-off (requiring either a source/conda-forge AVX2 FAISS build or confirming stock PyPI `faiss-cpu 1.15.1` is sufficient).

### 2.4 `faiss.IndexHNSWFlat` Structural & Sanity Probe (`siftsmall` 10k & Synthetic 100k)

Observed directly on `faiss.IndexHNSWFlat(128, M)` with `idx.hnsw.efConstruction = 100`, `omp_set_num_threads(1)`, pinned to `0x0004`:

1. **Parameter & Storage Alignment (`M` -> `M_max0 = 2*M`)**:
   - `metric_type == faiss.METRIC_L2 (1)` -> squared Euclidean $L_2^2$, matching Rust `Metric::L2Squared` (`D[0, :3] = [76608., 77004., 79702.]` on `siftsmall` query 0, IDs `[2176, 3752, 882]` matching `siftsmall_groundtruth.ivecs`).
   - `cum_nneighbor_per_level` for $M=16$: `[0, 32, 48, 64, ...]` -> `nb_neighbors(0) = 2*M = 32`, `nb_neighbors(l >= 1) = M = 16`. Verified across `M in {8, 16, 32}`.
   - `assign_probas` for $M=16$: `(1 - 1/M) * (1/M)^l = [0.9375, 0.05859375, 0.00366211, ...]`, mathematically identical to Rust's $\lfloor -\ln(U)/\ln(M) \rfloor$ geometric layer distribution.
   - `hnsw.levels` stores `max_level + 1` as `int32` (`4 B/vec`); `hnsw.offsets` is `uint64` (`8 B/vec`, len `N+1`); `hnsw.neighbors` is `int32` (`4 B/slot`, unused slots `-1`). On `siftsmall` 10k `M=16`, FAISS graph-only arrays consume `144.02 B/vec` (`mean_L0_deg = 15.65`), compared to Rust's `146.96 B/vec` at 10k (`148.91 B/vec` at 1M).
2. **Graph Quality Divergence at Equal `efSearch` (`siftsmall` 10k sanity check, NOT published numbers)**:
   - `M=16, efC=100, k=10` on `siftsmall`:
     - `efSearch = 10`: FAISS `Recall@10 = 0.9070` vs Rust Phase 2 ([`results/phase3/phase2_report.json`](../phase3/phase2_report.json)) `0.9510`.
     - `efSearch = 20`: FAISS `Recall@10 = 0.9640` vs Rust Phase 2 `0.9860`.
     - `efSearch = 50`: FAISS `Recall@10 = 0.9940` vs Rust Phase 2 `1.0000`.
     - `efSearch = 100`: FAISS `Recall@10 = 1.0000` vs Rust Phase 2 `1.0000`.
   - Note on `efSearch < k`: At `efSearch = 5, k = 10` on `siftsmall`, FAISS returned `1000/1000` valid IDs (`Recall@10 = 0.7910`), whereas Rust `search_knn` clamps `effective_ef = ef_search.max(k)` (`src/hnsw.rs:1502`). Because our evaluation ladder starts at `efSearch = 10 == k`, both engines operate in `efSearch >= k` everywhere.
3. **Batch vs Per-Query FFI Call Overhead & Warmup Sensitivity**:
   - On `100k` synthetic vectors (`efC = 16`, `efSearch = 50`, `500` queries, `omp = 1`, pinned to `0x0004`, `1 warmup + 5 timed reps`):
     - Batch `idx.search(Q, 10)`: median **`0.1489 ms/q`** (`74.44 ms` total; runs `[76.06, 74.53, 74.30, 74.29, 74.44]`).
     - Per-query `idx.search_c(1, q_ptr, 10, d_ptr, i_ptr)` loop (preallocated `D`/`I` & `swig_ptr`): median **`0.1554 ms/q`** (`77.71 ms` total; runs `[77.71, 78.03, 79.01, 77.40, 77.11]`).
     - **Isolated per-query FFI + boundary delta**: **`6.54 µs/q`** (`0.0065 ms/q`), confirming `VisitedTable::get_reusable()` in FAISS 1.15.1 avoids per-call `ntotal` table reallocations once warmed up and pinned! (By contrast, an un-warmed/unpinned probe showed `145.99 µs/q` delta, proving why pinning `0x0004` + full `Q`-query warmup is mandatory).
   - Pure empty FFI baseline (`1`-vector `IndexFlatL2`, `1,000` queries): `0.11 µs/q` batch vs `3.69 µs/q` single-row slice loop (`3.58 µs/q` Python+SWIG baseline).

---

## 3. Benchmark Alignment Matrix & Configuration Scope

### 3.1 Dataset & Ground Truth Parity

| Parameter | `100k` Scale | `1M` Scale |
|---|---|---|
| **Base Vectors** | First `100,000` rows of `data/sift1m/sift_base.fvecs` (`128`-d `float32`) | Full `1,000,000` rows of `data/sift1m/sift_base.fvecs` (`128`-d `float32`) |
| **Query Vectors** | Full `10,000` rows of `data/sift1m/sift_query.fvecs` | Full `10,000` rows of `data/sift1m/sift_query.fvecs` |
| **Metric & $k$** | `Metric::L2Squared` / `faiss.METRIC_L2`, $k = 10$ | `Metric::L2Squared` / `faiss.METRIC_L2`, $k = 10$ |
| **Ground Truth** | Cached brute-force `data/gt_100000_k100.ivecs` (`ExactKnn::search_parallel_with_base`, first `10` cols; verified against `IndexFlatL2`) | Official Texmex `data/sift1m/sift_groundtruth.ivecs` (first `10` cols; verified in Phase 3 against `ExactKnn`) |
| **Recall Definition** | Exact shared set-overlap `Recall@10` (`src/hnsw_phase1/recall.py:compute_recall_at_k` == `src/recall.rs:compute_recall_at_k`): `(1/Q) * sum_q |pred[:10] ∩ gt[:10]| / 10` | Same |
| **Repetitions** | `1` untimed warmup over all `10,000` queries + **`5` timed repetitions** (report median, min, max, spread) | `1` untimed warmup over all `10,000` queries + **`5` timed repetitions** (report median, min, max, spread) |
| **Build Repetitions** | `3` builds at `100k` (report median, min, max) | `1` build at `1M` |

*(Note: `data/gt_100000_k100.ivecs` is currently absent from `data/` and will be regenerated once via `ExactKnn::search_parallel_with_base` and cross-checked 100% against `faiss.IndexFlatL2` before running Phase 7.)*

### 3.2 Target Index Configurations (`M`, `efConstruction`)

We align directly with the **4 Pareto-frontier configurations** established in Phase 4 ([`results/phase4/grid_1m.grid`](../phase4/grid_1m.grid) & [`results/phase4/summary.json`](../phase4/summary.json)):

| Config ID | $M$ | $M_{\max0}$ | $\text{efConstruction}$ | Role on Pareto Frontier | Phase 4 Rust `1M` Build (`s`) | Phase 4 Rust `1M` Graph-Only (`B/vec`) |
|---|---|---|---|---|---:|---:|
| **C1** | `8` | `16` | `100` | Low-memory / max-throughput | `266.541` | `85.44` |
| **C2** | `16` | `32` | `100` | General standard / Phase 3 reference | `450.301` | `148.91` |
| **C3** | `32` | `64` | `200` | Balanced high-recall frontier | `1,288.897` | `272.33` |
| **C4** *(Gate G6 opt)* | `48` | `96` | `400` | Accuracy-saturated dense graph | `2,827.980` | `402.42` |
| **C2-nkp** *(Sensitivity)* | `16` | `32` | `100` | Rust `--no-keep-pruned-shrink` (`keep_pruned_shrink = false`) to isolate shrink-backfill effect vs FAISS | `415.076` *(Phase 3.5)* | `148.91` |

- **Gate G6 Scope Choice:** Benchmarking `{C1, C2, C3, C2-nkp}` at `100k` and `1M` avoids the ~47-minute Rust `C4` (`M=48, efC=400`) 1M rebuild (or reuses Phase 4's clean `1M` numbers if `src/hnsw.rs` is untouched), while adding `C4` gives full 4-point parity with Phase 4; `{C1, C2, C3}` vs `{C1, C2, C3, C4}` is decided at Gate G6.
- **`efSearch` Ladder (both engines, both scales)**: `[10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600]` (identical 12-point Phase 4 ladder).
- **Seed & RNG**: Rust uses deterministic `seed = 42` (`Xoshiro256**`). FAISS uses explicit `idx.hnsw.rng = faiss.RandomGenerator(42)` for reproducible FAISS builds across runs (not comparable edge-for-edge with Rust's PRNG, see Section 6).

---

## 4. Isolating Python FFI Overhead vs C++ Batch Throughput

A central methodological risk when comparing a native Rust binary against Python-wrapped FAISS is asymmetric timing boundaries:
- **Rust (`search_batch_timed_with_tracker`)** runs `Instant::now()` around **every single query** inside Rust (`10,000` timer calls per pass), reporting `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, and `QPS` (`Q / total_s`).
- **Rust (`search_batch_parallel(..., num_threads=1)`)** in `src/hnsw.rs:1666` runs a tight single-thread loop over all `10,000` queries **without** per-query `Instant::now()` calls — only one outer timer around the whole batch.
- **FAISS (`idx.search(Q, k)`)** runs a single SWIG C++ call over all `10,000` queries (zero per-query Python overhead, but returns no per-query latency distribution for `p50`/`p95`/`p99`).
- **FAISS (`idx.search_c(1, q_ptr, k, d_ptr, i_ptr)`)** measures individual query times for `p50`/`p95`/`p99`, but pays one Python+SWIG roundtrip + C++ `search()` entry per query (`~6.5 µs/q` observed on `100k`).

To guarantee complete honesty in both throughput and tail latency, Phase 7 executes **paired measurement modes** at every `(config, efSearch)` point:

| Measurement Mode | Rust Path | FAISS Path | Primary Metric & Purpose |
|---|---|---|---|
| **1. Pure Batch Throughput (Zero Per-Query Overhead)** | `search_batch_parallel(&queries, k, ef, 1)` timed once around all `Q` queries (`1 warmup + 5 reps`) | `idx.search(Q, k, D=D_batch, I=I_batch)` timed once around all `Q` queries (`1 warmup + 5 reps`) | **Headline single-thread `QPS` and `mean_ms`** — eliminates both Python FFI loop tax and Rust per-query `Instant::now()` cost |
| **2. Per-Query Distribution (`p50`, `p95`, `p99`)** | `search_batch_timed_with_tracker(&queries, k, ef, &mut tracker)` (`1 warmup + 5 reps`) | Preallocated `idx.search_c(1, q_ptrs[i], k, d_ptr, i_ptr)` loop timed per query with `time.perf_counter_ns()` (`1 warmup + 5 reps`) | **Tail latency (`p50`, `p95`, `p99`, `min`, `max`)** |
| **3. Explicit FFI Tax Disclosure** | Outer-minus-sum `Instant::now()` overhead check | `ffi_delta_us = (loop_total_ms - batch_total_ms) * 1000 / Q` + `1`-vec `IndexFlatL2` floor probe | Quantifies exact per-query Python/SWIG overhead per point; annotated on every tail-latency table and chart |

**Hard Rule:** Headline QPS / mean-latency speedup ratios between Rust and FAISS **must** come from **Mode 1 (Pure Batch Throughput)**. Per-query `p99` comparisons come from **Mode 2** with `ffi_delta_us` explicitly disclosed (and checked so `ffi_delta_us` never accounts for the margin of a claimed Rust win).

---

## 5. Single-Thread Primary vs Multi-Thread Comparison Protocol

### 5.1 Primary Comparison: Single-Thread Pinned (`0x0004`)

Both engines execute in **separate, isolated processes** pinned to `0x0004` (P-core 1 SMT0) so `PeakWorkingSet64` (`peak_rss_mb`) and memory bandwidth are uncontaminated:
- **Rust**: `VECTOR_SEARCH_AFFINITY_MASK=0x0004`, launched via PowerShell `Start-Process` / `ProcessorAffinity = 0x0004` (or `SetProcessAffinityMask`), single-threaded `HnswIndex::build` + single-threaded Mode 1 & Mode 2 search.
- **FAISS**: `faiss.omp_set_num_threads(1)` verified (`assert faiss.omp_get_max_threads() == 1`) + `kernel32.SetProcessAffinityMask(GetCurrentProcess(), 0x0004)` verified (`== 0x0004`), `IndexHNSWFlat.add(base)` + Mode 1 & Mode 2 search.

### 5.2 Secondary Comparison: Multi-Thread Search & Build Disclosure

Because parallel Rust index construction remains out of scope for Phase 7, we separate **Search** (both have parallel implementations!) from **Build** (only FAISS is parallel):

1. **Multi-Thread Batch Query Throughput (`T = 6` P-cores `0x0555` and `T = 16` all cores `0xFFFF`)**:
   - **Rust ALREADY implements parallel batch search**: `HnswIndex::search_batch_parallel(&queries, k, ef_search, num_threads)` (`src/hnsw.rs:1612`) partitions `queries` across `num_threads` `std::thread::scope` workers, each reusing one `VisitedTracker`.
   - **FAISS**: `faiss.omp_set_num_threads(T)` + `idx.search(Q, k, D=D_batch, I=I_batch)` (OpenMP parallelizes across queries in C++).
   - Compare Mode 1 batch `QPS` across `T in {1, 6 (mask 0x0555), 16 (mask 0xFFFF)}` for `C2 (M=16, efC=100)` across the `efSearch` ladder at `100k` and `1M`.
   - Note: `search_batch_parallel` allocates `c0`, `w`, and `results` per query on each worker thread, whereas FAISS reuses per-OpenMP-thread buffers; any multi-thread scaling gap is reported plainly.
2. **Multi-Thread Build Wall-Clock Asymmetry Disclosure**:
   - Rust `HnswIndex::build` is **strictly single-threaded** (sequential `0..N` insert for bit-exact determinism).
   - FAISS `IndexHNSWFlat.add(base)` uses OpenMP parallel insertion (`omp_num_threads = 6` on `0x0555` and `16` on `0xFFFF`).
   - We record FAISS multi-thread build time (`T=6` and `T=16`) alongside single-thread (`T=1`) build time in a separate **Practical Wall-Clock Build** table, explicitly stating that Rust lacks a parallel builder and loses on multi-core wall-clock indexing speed.

---

## 6. Algorithmic Discrepancy Disclosure & Matched-Recall Evaluation Rule

Comparing Rust `HnswIndex` vs FAISS `IndexHNSWFlat` requires explicitly documenting **why** their graphs and raw timings differ even when `(M, efConstruction, efSearch)` match:

| Dimension | Rust `HnswIndex` (Default) | FAISS `IndexHNSWFlat` (`1.15.1`) | Benchmark Impact & Control |
|---|---|---|---|
| **Shrink Backfill (`keep_pruned`)** | `keep_pruned = true` in forward insert **and** reverse-edge shrink (`keep_pruned_shrink = None`) | No `keep_pruned` backfill in `shrink_neighbor_list`; also uses `hnsw.prune_headroom = 0.2` in 1.15.1 | Rust produces denser Layer-0 neighborhoods and higher recall at equal `efSearch`, at the cost of slower build & slightly more distance evals per hop. **Control:** Include `C2-nkp` (`keep_pruned_shrink = false`) to isolate the shrink backfill effect! |
| **Layer Assignment RNG** | Deterministic `SplitMix64 + Xoshiro256**` (`seed = 42`), `m_L = 1 / ln(M)` | C++ `faiss::RandomGenerator(42)` (`mt19937`-based), `assign_probas` `(1 - 1/M)(1/M)^l` | Different node heights and entry points (`top_layer` differs), so per-`efSearch` recall differs slightly; matched-recall evaluation neutralizes this. |
| **Distance Ties & Ordering** | Canonical total order `(distance, id)` via `f32::total_cmp` | Heap ordering by `float` distance (non-deterministic ID ties across OpenMP build threads when `T > 1`) | Negligible on `SIFT1M` (`999/1000` GT agreement in Phase 3); `omp_set_num_threads(1)` makes FAISS build deterministic for a given `rng` seed. |
| **Search Termination** | Standard Algorithm 2 (`d_c > d_f`) + `effective_ef = efSearch.max(k)` | `search_bounded_queue = true`, `check_relative_distance = true` (`search_method = 0`) | Keep FAISS defaults (`search_bounded_queue = True`, `check_relative_distance = True`) as stock behavior; keep `efSearch >= k = 10`. |
| **Memory Layout** | Flat `layer0` (`u32`, stride `2*M+1`, `slot 0 = degree`) + dense `upper_arena` (`u32`) + `8 B` `HnswNodeMeta` | Flat `neighbors` (`int32`, `-1` sentinel padding) + `offsets` (`uint64`, `8 B/vec`) + `levels` (`int32`, `4 B/vec`) | Report both `graph_only_bytes_per_vector` (`neighbors + offsets + levels` for FAISS) and process OS `PeakWorkingSet64` (`peak_rss_mb`) in isolated processes. |
| **SIMD / Safety** | `#![forbid(unsafe_code)]`, `std` only; 8-way unrolled `l2_squared` relying on LLVM auto-vectorization (SSE2 baseline) | Stock PyPI Windows wheel `1.15.1` statically built at `SIMDLevel_NONE` (no `swigfaiss_avx2`, baseline SSE2 + MSVC optimization) | Both default artifacts run at baseline x86-64 SSE2! Disclose explicitly that neither uses AVX2 unless an optional AVX2 arm is approved in Gate G6. |

### Matched-Recall Evaluation Rule

For every configuration and scale, primary comparison evaluates **latency (`mean_ms`, `p99_ms`) and `QPS` as functions of `Recall@10`**, never at fixed `efSearch`:

1. **Continuous Pareto Curves**: Plot `Recall@10` ($x$-axis) vs `p99_ms` (log $y$-axis) and vs `Mode 1 QPS` (log $y$-axis) across all 12 `efSearch` points (`10..600`).
2. **Interpolated Fixed-Recall Targets**: Evaluate both engines at standardized recall targets:
   $$R^* \in \{0.90,\; 0.95,\; 0.98,\; 0.99,\; 0.995\}$$
   For each target $R^*$ achieved within the observed `[min_recall, max_recall]` range (never extrapolate beyond the curve's endpoints!), locate the tightest bracketing pair $(ef_{\text{lo}}, R_{\text{lo}})$ and $(ef_{\text{hi}}, R_{\text{hi}})$ with $R_{\text{lo}} \le R^* \le R_{\text{hi}}$ and interpolate log-latency / log-QPS linearly in recall:
   $$\alpha = \frac{R^* - R_{\text{lo}}}{R_{\text{hi}} - R_{\text{lo}}}, \qquad \log \hat{y}(R^*) = (1 - \alpha)\log y_{\text{lo}} + \alpha \log y_{\text{hi}}$$
   Report $\hat{y}(R^*)$ alongside the exact discrete ceiling operating point $(ef_{\text{hi}}, R_{\text{hi}}, y_{\text{hi}})$ and speed ratio $\text{QPS}_{\text{Rust}} / \text{QPS}_{\text{FAISS}}$.
3. **Graph Distance-Eval Efficiency Diagnostic**: Also inspect FAISS `faiss.cvar.hnsw_stats.ndis / Q` vs Rust `--features instrument` distance-evaluation counts (in a separate un-timed diagnostic pass so instrumentation adds zero overhead to timed runs) to explain whether any QPS difference comes from **graph quality** (fewer distance evaluations at fixed recall) vs **per-evaluation kernel speed** (`ns / dist_eval`).

---

## 7. Implementation & Execution Roadmap (`T5.1`..`T5.7`)

No implementation or benchmark execution begins before **Gate G6** sign-off.

| Task ID | Deliverable | Class | Allowed Files | Acceptance Criteria |
|---|---|---|---|---|
| **T5.1** | Rust Mode 1 pure-batch & multi-thread Phase 7 runner (or `phase7` CLI / flag) | `MEASUREMENT-ONLY` | `src/benchmark.rs` or `src/bin/phase7.rs`, `Cargo.toml`, `tests/` | Reuses `HnswIndex`, runs Mode 1 (`search_batch_parallel(..., 1)` + multi-thread `T in {6, 16}`), Mode 2 (`search_batch_timed_with_tracker`), optionally diagnostic `ndis` count, stamps full JSON provenance (`git_commit`, `git_dirty`, `processor_affinity_mask`, `power_mode`); `cargo test --release` passes with no `PRESERVING` regression. |
| **T5.2** | FAISS Python benchmark harness & parity/affinity self-tests | `MEASUREMENT-ONLY` | `scripts/benchmark_faiss.py` (or `src/hnsw_phase1/faiss_bench.py`), `tests/test_phase7_harness.py` | Enforces 64-bit `SetProcessAffinityMask`, `omp_set_num_threads`, `1 warmup + 5 timed reps`, Mode 1 batch + Mode 2 `search_c` loop, FFI delta calculation, exact shared `compute_recall_at_k`, isolated child process per scale/config for clean PowerShell `PeakWorkingSet64`, `ruff` + `mypy` + `pytest` pass (`0` skipped). |
| **T5.3** | `100k` Pilot & Time-Budget Check (`C2`: `M=16, efC=100` on `100k`) | `MEASUREMENT-ONLY` | `results/phase7/pilot_100k.json` | Single clean-commit pilot comparing Rust vs FAISS across 12 `efSearch` points (`1 warmup + 5 reps`), measures true `100k` runtime + FFI delta + `1M` wall-clock projection before launching `1M` suite. |
| **T5.4** | Full `100k` & `1M` Single-Thread Comparison Runs (`0x0004`, clean commit) | `MEASUREMENT-ONLY` | `results/phase7/*.json` | Runs approved config set (`{C1, C2, C3, C2-nkp}` and optionally `C4`) at `100k` (3 builds, 5 query reps) and `1M` (1 build, 5 query reps) in isolated pinned processes; `git_dirty: false`, `power_mode` recorded, OneDrive paused. |
| **T5.5** | Multi-Thread Search & OpenMP Build Scaling Runs (`0x0555`, `0xFFFF`) | `MEASUREMENT-ONLY` | `results/phase7/multithread_*.json` | Runs `C2 (M=16, efC=100)` across `T in {1, 6, 16}` for Rust `search_batch_parallel` vs FAISS OpenMP batch search, plus FAISS `T in {6, 16}` build times (disclosing Rust single-thread build). |
| **T5.6** | Deterministic Chart Suite & Matched-Recall Synthesis | `MEASUREMENT-ONLY` | `scripts/plot_phase7.py`, `results/phase7/*.png`, `results/phase7/*.svg` | Matplotlib-only deterministic charts: (1) Recall@10 vs p99 latency (`100k` & `1M`), (2) Recall@10 vs Mode 1 single-thread QPS, (3) Matched-recall latency/QPS ratio bars at `R* in {0.90, 0.95, 0.98, 0.99, 0.995}`, (4) Multi-thread search scaling & build time/memory comparison. |
| **T5.7** | Final Write-Up & Claims Ledger Audit | `MEASUREMENT-ONLY` | `results/phase7/FINDINGS.md`, `results/CLAIMS.md`, `README.md` | Every reported number traces to a clean-commit JSON path in `CLAIMS.md`; explicitly reports where Rust loses alongside where it wins; passes full verification gate. |

---

## 8. Gate G6 Decision Checklist

> **Reviewed and decided prior to M5 / Phase 7 code and benchmark runs (outcomes recorded in [`G6_DECISIONS.md`](./G6_DECISIONS.md)).**

1. **Configuration Scope at `100k` and `1M`**:
   - [ ] **Option A (Recommended — Core Frontier + `keep_pruned` Control):** `{ (M=8, efC=100), (M=16, efC=100), (M=32, efC=200), (M=16, efC=100, keep_pruned_shrink=false) }`. Covers low-memory, standard baseline, high-recall frontier, and shrink-backfill isolation without a ~47-minute `M=48, efC=400` 1M rebuild.
   - [ ] **Option B (Full 4-Point Phase 4 Parity):** Add `(M=48, efC=400)` at both `100k` and `1M` (+~47 min Rust 1M build + FAISS 1M build/sweep).
2. **Stock PyPI `faiss-cpu 1.15.1` (`SIMDLevel_NONE`) vs Optional AVX2 Comparison**:
   - [ ] **Option A (Recommended Primary):** Benchmark stock PyPI `faiss-cpu 1.15.1` (`SIMDLevel_NONE`, matching `requirements.txt`) against default-target Rust (`opt-level = 3, lto = "thin"`), both operating at baseline x86-64 SSE2.
   - [ ] **Option B (Add AVX2 Addendum Arm):** Also test an AVX2-enabled FAISS build (e.g., conda-forge/source build if installed separately) vs Rust compiled with `RUSTFLAGS="-C target-cpu=native"`, reported separately from the default-build headline numbers.
3. **Multi-Thread Evaluation Scope**:
   - [ ] Approve benchmarking multi-thread search scaling at `C2 (M=16, efC=100)` across `T = 1 (0x0004)`, `T = 6` physical P-cores (`0x0555`, no SMT), and `T = 16` all logical threads (`0xFFFF`), plus recording FAISS `T = 6` and `T = 16` build times while plainly disclosing Rust build is single-threaded.
4. **Matched-Recall Targets**:
   - [ ] Confirm standardized matched-recall checkpoints $R^* \in \{0.90, 0.95, 0.98, 0.99, 0.995\}$ (evaluated via log-latency/log-QPS interpolation without extrapolation, plus discrete upper-bracket points).
5. **Re-Run Rust on Clean Phase 7 Commit vs Reuse Phase 4 JSON**:
   - [ ] **Recommended:** Re-run Rust Mode 1 (`search_batch_parallel(..., 1)`) + Mode 2 (`search_batch_timed_with_tracker`) on the exact same clean Phase 7 commit and power/thermal session as FAISS (`Best Performance` power scheme, OneDrive paused), and cross-check that graph fingerprints and recalls remain bit-for-bit identical to Phase 4 (`0x44044886c62bca5e` for `1M M=16, efC=100`).
6. **Pre-Flight Environment Controls**:
   - [ ] Switch Windows power plan from `Balanced` to **Best Performance** (AC plugged in) and pause OneDrive sync (or execute from a non-OneDrive local path) prior to `T5.4`/`T5.5` final measurements.
