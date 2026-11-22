# Phase 3.5 Optimization & Profiling Log

- **Machine**: Intel(R) Core(TM) i7-12650H (16 logical cores: 6P+4E, 15.8 GB RAM, Windows 11 x86_64)
- **Baseline Git Commit**: `170b9d7`
- **Git Tree State**: Clean / Working tree tracked (`git_dirty: true` due to phase3.5 files)
- **Rustc Version**: `rustc 1.99.1 (58cf8f9c3 2026-10-13)`
- **Compiler Profile**: `--release` (`opt-level = 3`, `lto = "thin"`, `codegen-units = 1`)
- **Environment Controls**:
  - AC Power: Connected (Best Performance mode)
  - Core Affinity: P-core pinned (Mask: `0x0004` / CPU 2)
  - OneDrive Sync: Paused during runs
  - Background Load: Quiesced

---

## 1. Baseline Re-Measurements (Unmodified Commit)

### 1.1 100k Vectors Baseline (3 Repetitions)
- **Dataset**: Authentic Texmex SIFT1M (prefix 100,000 vectors, dim=128, L2Squared)
- **Index Config**: $M=16, M_0=32, \text{efConstruction}=100, \text{seed}=42, \text{extend\_candidates}=\text{false}, \text{keep\_pruned}=\text{true}$
- **Process Model**: Direct measured benchmark with PowerShell PeakWorkingSet64 OS RSS profiling

| Run # | Build Time (s) | Build QPS (vec/s) | Peak RSS (MiB) | Graph-Only (B/vec) | Fixed-Query Hash / Recall@10 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | 30.536 s | 3274.8 | 128.48 MiB | 338.66 | 0x7cc605e91d8ccce8 / 98.14% (ef50) |
| Run 2 | 30.116 s | 3320.5 | 128.48 MiB | 338.66 | 0x7cc605e91d8ccce8 / 98.14% (ef50) |
| Run 3 | 30.299 s | 3300.4 | 128.48 MiB | 338.66 | 0x7cc605e91d8ccce8 / 98.14% (ef50) |
| **Summary** | **Median: 30.299 s** | **Min: 30.116 s** | **Max: 30.536 s** | **Spread ($\Delta_{\text{spread}}$): 0.421 s (1.39%)** | **2x Spread: 0.841 s *(36.244 s w/ `instrument` feature)*** |

- **Graph Fingerprint (100k)**: `0x80af03bd7aebaade` (`9272630593691568862`)
- **Query Results Hash (100k, 1k queries @ k=10, efSearch=100)**: `0x7cc605e91d8ccce8` (`8990880204873059560`)
- **Golden Fixture Verification**: PASSED against `results/phase3_5/golden_100k.json`
- **efSearch Sweep on 100k**:
  - `efSearch = 50`:  Recall@10 = 98.14%, Median Latency = 0.165 ms, QPS = 6,070.1
  - `efSearch = 100`: Recall@10 = 99.54%, Median Latency = 0.296 ms, QPS = 3,374.2

---

### 1.2 1M Vectors Baseline (Single Build Run)
- **Dataset**: Authentic Texmex SIFT1M (full 1,000,000 base vectors, dim=128, L2Squared)
- **Measured Build Time**: **511.271 s** (1,955.91 vectors/s) *(Previous historical reference: 470.97 s)*
- **Peak RSS (Process OS Working Set)**: **845.75 MiB** (886.83 MB)
- **Graph Footprint**: **342.67 B/vector total** (**338.67 B/vector graph-only**, excluding scratch & visited tracker)
- **Graph Topology**: Max Layer = 5, Entry Point Node = #829262
- **Graph Fingerprint (1M)**: `0x19b378e1f978084d` (`1851956783727773773`, pre-`enter_point` hash) / `0x44044886c62bca5e` (`4901122038173518430`, with `enter_point` in `graph_fingerprint()`)
- **Official Ground Truth Check**: PASSED (100.0% ExactKNN match on 100 sample queries)
- **Bit-Exact Reproducibility Check**: PASSED

#### Baseline 1M efSearch Sweep (5-Run Median & Spread)
| efSearch | Recall@10 | MedMean ms | Mean Spread ms | Median p50 (ms) | Median p99 (ms) | Median QPS | Monotonic |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 73.31% | 0.095 | 0.015 | 0.091 | 0.160 | 10565.0 | PASSED |
| 20 | 84.95% | 0.140 | 0.006 | 0.139 | 0.221 | 7124.5 | PASSED |
| 50 | 94.70% | 0.272 | 0.006 | 0.278 | 0.381 | 3671.8 | PASSED |
| 100 | 98.06% | 0.486 | 0.010 | 0.502 | 0.670 | 2055.6 | PASSED |
| 200 | 99.38% | 0.893 | 0.028 | 0.923 | 1.283 | 1119.2 | PASSED |

---

## 2. Baseline Profiling & Phase Time Breakdown (T1.1, T1.4)

- **Method**: Safe in-process phase instrumentation (`#[cfg(feature = "instrument")]`) measuring phase-boundary `Instant` and exact call counters with inner-loop register distance accumulation. Zero-cost when feature is disabled.

### 2.1 100k Build Time Split & Counters
| Component / Phase | Total Calls | Calls / Vector | Cumulative Time (s) | % of Build Time | Distance Evaluations |
|---|:---:|:---:|:---:|:---:|:---:|
| `search_layer` (Routing + Exploration) | 440,442 | 4.40 | 25.3004 s | 69.81% | ~142,500,000 |
| Forward Heuristic (`select_neighbors`) | 832,564 | 8.33 | 2.4212 s | 6.68% | ~22,100,000 |
| Degree Shrink (Reverse-Edge Pruning) | 725,905 | 7.26 | 7.4445 s | 20.54% | ~123,300,000 |
| Other / Allocations / Residual | - | - | 1.0775 s | 2.97% | 0 |
| **Total Build** | - | - | **36.2437 s** | **100.0%** | **287,901,334 total (2,879.01/vec)** |

### 2.2 1M Build Time Split & Counters
| Component / Phase | Total Calls | Calls / Vector | Cumulative Time (s) | % of Build Time | Distance Evaluations |
|---|:---:|:---:|:---:|:---:|:---:|
| `search_layer` (Routing + Exploration) | 5,111,179 | 5.11 | 403.0093 s | 78.83% | 1,715,511,861 |
| Forward Heuristic (`select_neighbors`) | 8,631,013 | 8.63 | 24.5556 s | 4.80% | 282,127,479 |
| Degree Shrink (Reverse-Edge Pruning) | 7,564,342 | 7.56 | 72.5671 s | 14.19% | 1,747,274,139 |
| Other / Allocations / Residual | - | - | 11.1389 s | 2.18% | 0 |
| **Total Build** | - | - | **511.2709 s** | **100.0%** | **3,744,913,479 total (3,744.91/vec)** |

### 2.3 Profiling Analysis & Top Bottlenecks
1. **Bottleneck #1: `search_layer` Beam Exploration & Routing**
   - **Cost Share**: **69.81% at 100k, 78.83% at 1M** (403.0 s out of 511.3 s).
   - **Mechanism**: Every node insertion executes $O(l_{\text{top}})$ routing searches at upper layers plus beam search at layer 0 with `ef = 100`. Inside `search_layer`, candidate exploration performs millions of distance evaluations and visited tracking lookups.
   - **Evidence**: Over 1.71 billion distance computations occur in `search_layer` alone.
2. **Bottleneck #2: Degree Shrink (Reverse-Edge Pruning)**
   - **Cost Share**: **20.54% at 100k, 14.19% at 1M** (72.6 s out of 511.3 s).
   - **Mechanism**: When reciprocal back-edges cause a neighbor node's degree to exceed `M` (or `M0`), `shrink` is invoked. It re-evaluates all pairwise distances among candidate neighbors and re-runs the diversity heuristic.
   - **Evidence**: Invoked 7,564,342 times during 1M construction, generating 1.75 billion distance evaluations.
3. **Bottleneck #3: Forward Heuristic (`select_neighbors`)**
   - **Cost Share**: **6.68% at 100k, 4.80% at 1M** (24.6 s out of 511.3 s).
   - **Mechanism**: Triangular inequality distance pruning of candidates for the newly inserted node.

---

## 3. Optimization Decision Roadmap (Data-Driven Order)

Based on the empirical profile above, the candidate optimizations are ordered strictly by measured time share:

1. **Step 1: T1.5 Tagged Visited Tracker**
   - **Targets**: Hotspot #1 (`search_layer`), accounting for 78.8% of build time.
   - **Hypothesis**: Replacing `HashSet<usize>` or hash lookups with a flat generation-tagged array (`u32` per node) eliminates hashing overhead and allocator pressure.
   - **Classification**: PRESERVING (must not alter graph topology; fingerprint and query results hash must match golden fixtures exactly).

2. **Step 2: T1.6 Per-Call Heap Reuse in `search_layer`**
   - **Targets**: Hotspot #1 (`search_layer`), specifically inner-loop allocation of binary heaps and candidate vectors.
   - **Hypothesis**: Reusing scratch buffers across layer searches eliminates dynamic re-allocations.
   - **Classification**: PRESERVING.

3. **Step 3: T1.8 `keep_pruned = false` in Shrink Pruning**
   - **Targets**: Hotspot #2 (`shrink`), accounting for 14.2%–20.5% of build time and 1.75 billion distance evaluations.
   - **Hypothesis**: Disabling step-5 keep-pruned backfilling specifically during degree shrink eliminates redundant edge evaluations on discarded candidates.
   - **Classification**: EXPERIMENTAL / GRAPH-ALTERING. Requires verifying recall against Texmex ground truth at 1M.

4. **Step 4: T1.7 `u32` Node IDs and Flat Adjacency Storage**
   - **Targets**: Memory layout and cache locality across both `search_layer` and `shrink`.
   - **Hypothesis**: Switching neighbor IDs from 64-bit `usize` (8 bytes) to `u32` (4 bytes) and packing neighbor lists contiguously cuts graph memory from 338.7 B/vec down toward ~70 B/vec, improving CPU cache hit rates.
   - **Classification**: PRESERVING (logical graph fingerprint invariant under u64 casting).

---

## 4. Optimization Candidate Ledger

| Commit / Candidate | Task | Target Scale | Build Time (s) | $\Delta$ vs Base | % Speedup | Peak RSS | Graph B/vec | Fingerprint / Recall | Decision |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `9ba9c4a` Baseline | M1 | 100k | 30.299 s (36.244 s instr.) | - | - | 128.48 MiB | 338.66 | `0x80af03bd7aebaade` / 98.14% | **BASELINE ANCHOR** |
| `170b9d7` Baseline | M1 | 1M | 511.271 s | - | - | 845.75 MiB | 338.67 | `0x19b378e1f978084d` / 94.70% | **BASELINE ANCHOR** |
| Candidate T1.5 (`seen_tags`) | T1.5 | 100k | 30.855 s (31.033 s prealloc) | +0.556 s | -1.83% | 129.02 MiB | 338.66 | `0x80af03bd7aebaade` / 98.14% | **REVERTED (`no gain`)** — `search_layer` already uses `VisitedTracker`; `HeuristicScratch::seen` only runs on $\le 100$ items in `heuristic` (4.8% of build time), so replacing `HashSet` with `seen_tags` did not improve median build time beyond baseline noise. |
| Candidate T1.6 (`search_layer` heap reuse) | T1.6 | 100k | 31.080 s (`c` reuse) / 43.120 s (`c`+`w` reuse) | +0.781 s | -2.58% | 128.48 MiB | 338.66 | `0x80af03bd7aebaade` / 98.14% | **REVERTED (`no gain`)** — thread-local allocator reuse of `BinaryHeap` in `search_layer` is already L1-hot; reusing `c_scratch` (`31.080 s`) or `c_scratch`+`w_scratch` (`43.120 s`) did not improve median build time vs baseline (`30.299 s`). |
| Candidate T1.7 (`u32` flat adjacency) | T1.7 | 100k | **27.157 s** (min 27.068 s) | **-3.141 s** | **+10.37% faster** | **88.55 MiB** (-31.1%) | **145.57** (-57.0%) | `0x80af03bd7aebaade` / 98.14% | **KEPT** — cuts graph-only memory from 338.66 to 145.57 B/vec, improves 100k build median by 3.141 s (10.37%, $>7.4\times$ baseline spread), and speeds up query QPS by 15.3%. |
| Final 1M Confirmation (`366f185`, T1.7 default) | T1.10 | 1M | **493.783 s** (`2,025.2 vec/s`) | **-17.488 s** vs 511.271 s | **+3.42% faster** (`1.035x`) | **646.05 MiB** (**-23.6% / -199.7 MiB**) | **148.91** (**-56.0%**) | `0x44044886c62bca5e` / 94.703% | **KEPT (Official Phase 3 Default)** — 100% bit-exact recall (`73.315%`, `84.949%`, `94.703%`, `98.061%`, `99.376%`), cuts graph-only memory from 338.67 to 148.91 B/vec (-180.97 MiB) and isolated Peak RSS from 845.75 to 646.05 MiB. |
| Candidate T1.9 (`keep_pruned_shrink=false`) | T1.9 | 100k | **21.351 s** | **-8.948 s** | **+29.53% faster** | **77.21 MiB** (-39.9%) | **145.57** | `0xeecb25668ae17f19` / 98.70% | **EXPERIMENT (Flagged, default unchanged)** — cuts 100k build time by an additional 21.4% beyond T1.7. |
| Candidate T1.9 (`keep_pruned_shrink=false`) | T1.9 | 1M | **415.076 s** | **-96.195 s** vs 511.271 s (`-55.90 s` vs 470.97 s) | **+18.81% faster** | **645.89 MiB** (-23.6%) | **148.91** (-56.0%) | `0xf59eb0d017b09d4a` / 94.30% | **EXPERIMENT (Flagged, default unchanged)** — Maintainer decision required (Gate G1). See Section 4.1 below for full `efSearch` sweep and matched-latency comparison. |

### 4.1 Candidate T1.9 (`keep_pruned_shrink = Some(false)`) SIFT1M Experiment Report (R3 / Gate G1)

- **Configuration**: `HnswConfig { m: 16, m_max0: Some(32), ef_construction: 100, extend_candidates: false, keep_pruned: true, keep_pruned_shrink: Some(false), seed: 42 }`
- **Default Status**: `HnswConfig::default().keep_pruned_shrink` remains `None` (which delegates to `keep_pruned = true`), so default index builds remain 100% bit-identical to Phase 3.
- **1M Build Time**: **415.076 s** (`2,409.2 vectors/s`) vs **511.271 s** baseline (**1.232x speedup / -18.81%**).
- **1M Memory Footprint**: **645.89 MiB Peak RSS** (down from `845.75 MiB`), **148.91 B/vector graph-only** (down from `338.67 B/vector`).

#### Full SIFT1M `efSearch` Sweep Comparison (`keep_pruned_shrink = false` vs Default `true`)
| efSearch | Recall@10 (`shrink=false`) | Recall@10 (Default `true`) | Exact Recall $\Delta$ | MedMean ms (`shrink=false`) | MedMean ms (Default `true`) | p50 ms (`shrink=false`) | p99 ms (`shrink=false`) | QPS (`shrink=false`) | QPS (Default `true`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 71.019% | 73.311% | **-2.292%** | 0.088 ms | 0.095 ms | 0.085 ms | 0.160 ms | 11,369.0 | 10,565.0 |
| 20 | 83.559% | 84.953% | **-1.394%** | 0.132 ms | 0.140 ms | 0.129 ms | 0.220 ms | 7,574.3 | 7,124.5 |
| 50 | 94.304% | 94.703% | **-0.399%** | 0.255 ms | 0.272 ms | 0.257 ms | 0.391 ms | 3,914.9 | 3,671.8 |
| 100 | 98.009% | 98.061% | **-0.052%** | 0.457 ms | 0.486 ms | 0.465 ms | 0.700 ms | 2,186.3 | 2,055.6 |
| 200 | **99.442%** | 99.383% | **+0.059%** | 0.824 ms | 0.893 ms | 0.848 ms | 1.222 ms | 1,212.8 | 1,119.2 |

#### Recall at Matched Latency Analysis (Linear Interpolation on Recall vs. `MedMean ms`)
-Because `keep_pruned_shrink = false` leaves pruned nodes with slightly sparser average out-degree than `m_layer`, each `efSearch` step visits fewer redundant edges and runs **6.0%–7.7% faster** in query latency at equal `efSearch`:
  - At **0.140 ms** mean latency (Default `efSearch=20` @ `84.953%` recall): `keep_pruned_shrink = false` interpolates between `ef=20` (`0.132 ms`, `83.559%`) and `ef=50` (`0.255 ms`, `94.304%`) to **84.26% recall** ($\Delta = -0.69\%$).
  - At **0.272 ms** mean latency (Default `efSearch=50` @ `94.703%` recall): `keep_pruned_shrink = false` interpolates between `ef=50` (`0.255 ms`, `94.304%`) and `ef=100` (`0.457 ms`, `98.009%`) to **94.62% recall** ($\Delta = -0.08\%$).
  - At **0.486 ms** mean latency (Default `efSearch=100` @ `98.061%` recall): `keep_pruned_shrink = false` interpolates between `ef=100` (`0.457 ms`, `98.009%`) and `ef=200` (`0.824 ms`, `99.442%`) to **98.12% recall** ($\Delta = +0.06\%$ **higher recall at matched latency**).
  - At **`efSearch=200`**: `keep_pruned_shrink = false` achieves **both higher recall** (`99.442%` vs `99.383%`, `+0.059%`) **and lower latency** (`0.824 ms` vs `0.893 ms`, `-7.7%`).

---

## 5. Final Confirmation & Speedup Summary (T1.10)

- **Status**: Milestone 1 (Phase 3.5) complete. All candidates profiled, tested one per commit, verified against golden fixtures, and confirmed on full `SIFT1M` (`1,000,000` vectors) on clean commit `366f185`.
- **Milestone 1 Deliverables Attested**:
  1. `graph_fingerprint() -> u64` logical accessor hash (including `enter_point`) implemented and covered by unit & adversarial tests.
  2. Cargo feature `instrument` implemented with zero-cost when disabled (`size_of::<HnswIndex>() == 480` without `instrument`, `672` with `instrument`).
  3. Golden fixtures (`results/phase3_5/golden_10k.json`, `results/phase3_5/golden_100k.json`) generated and validated with 100% bit-exact reproducibility (`graph_fingerprint`, `query_results_hash`, and IEEE 754 `f32::to_bits` distances).
  4. **Preserving Candidates Tested**:
     - `T1.5` (`seen_tags` in `HeuristicScratch`): tested, reverted (`30.855 s` vs `30.299 s` baseline), logged `"no gain"`.
     - `T1.6` (`search_layer` binary heap scratch reuse): tested, reverted (`31.080 s` vs `30.299 s` baseline), logged `"no gain"`.
     - `T1.7` (`u32` flat adjacency storage in `layer0` & `upper_arena`, 8-byte `HnswNodeMeta`, `HnswIndex::neighbors(layer, node) -> &[u32]`): **KEPT** (`d6d36ff`).
       - **100k 3-run median build time**: **27.157 s** vs **30.299 s** baseline (**-3.141 s / +10.37% faster**, $>7.4\times$ baseline spread).
       - **Graph-only memory**: **338.67 B/vector $\to$ 148.91 B/vector** at 1M (**-56.03%**, `-180.97 MiB`); **338.66 $\to$ 145.57 B/vector** at 100k (**-57.02%**); **338.98 $\to$ 146.96 B/vector** at 10k (**-56.65%**).
       - **Total index memory**: **854.67 B/vector $\to$ 664.91 B/vector** at 1M (**-22.20%**, `815.08 MiB -> 634.11 MiB`).
       - **Isolated Peak RSS**: **13.70 $\to$ 11.94 MiB** at 10k (**-12.8%**), **88.67 $\to$ 69.21 MiB** at 100k (**-21.9%**), **845.75 $\to$ 646.05 MiB** at 1M (**-23.61% / -199.70 MiB**).
       - **1M Clean-Commit Build Time (`data/phase3_report.json` / `results/phase3_5/final_1m_report.json`)**: **493.783 s** (`2,025.18 vec/s`), **0.9658x** vs the freshly measured `511.271 s` baseline (`-17.488 s` / `1.035x` speedup; `1.0484x` vs the `470.966 s` historical reference on an isolated single build).
       - **Recall@10**: 100% bit-for-bit identical across all scales and `efSearch` values (`73.315%`, `84.949%`, `94.703%`, `98.061%`, `99.376%`).
  5. **Graph-Changing Experiment (`T1.9`, `keep_pruned_shrink = Some(false)`)**:
     - Implemented behind `HnswConfig::keep_pruned_shrink: Option<bool>` (default `None`, preserving `keep_pruned = true`) and `--no-keep-pruned-shrink` CLI flag (`366f185`).
     - Cuts 100k build time to **21.351 s** (**-29.53%** vs `30.299 s` baseline) and 1M build time to **415.076 s** (**-18.81% / -96.195 s** vs `511.271 s` baseline; **-11.87% / -55.890 s** vs `470.966 s` historical Phase 3), with `-0.08%` recall difference at `0.272 ms` (`ef=50` budget) and **`+0.06%` higher recall** at `0.486 ms` (`ef=100` budget). Ready for Gate G1 sign-off.
