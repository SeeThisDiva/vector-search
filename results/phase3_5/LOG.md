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
| Run 1 | 41.353 s | 2418.2 | 128.52 MiB | 338.66 | 0x7cc605e91d8ccce8 / 98.14% (ef50) |
| Run 2 | 31.142 s | 3211.1 | 128.52 MiB | 338.66 | 0x7cc605e91d8ccce8 / 98.14% (ef50) |
| Run 3 | 36.244 s | 2759.1 | 128.52 MiB | 338.66 | 0x7cc605e91d8ccce8 / 98.14% (ef50) |
| **Summary** | **Median: 36.244 s** | **Min: 31.142 s** | **Max: 41.353 s** | **Spread ($\Delta_{\text{spread}}$): 10.211 s (28.17%)** | **Noise Hurdle ($>2\times\Delta$, $\ge 5\%$): 20.421 s** |

- **Graph Fingerprint (100k)**: `0x8d1cce443935778f` (`10168228851108640655`)
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
- **Graph Fingerprint (1M)**: `0x19b378e1f978084d` (`1851956783727773773`)
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
| `170b9d7` Baseline | M1 | 100k | 36.244 s | - | - | 128.52 MiB | 338.66 | `0x8d1cce443935778f` / 98.14% | **BASELINE ANCHOR** |
| `170b9d7` Baseline | M1 | 1M | 511.271 s | - | - | 845.75 MiB | 338.67 | `0x19b378e1f978084d` / 94.70% | **BASELINE ANCHOR** |
| Candidate T1.5 | T1.5 | 100k | *Pending M2* | - | - | - | - | - | *Pending* |
| Candidate T1.6 | T1.6 | 100k | *Pending M3* | - | - | - | - | - | *Pending* |
| Candidate T1.8 | T1.8 | 100k | *Pending M3* | - | - | - | - | - | *Pending* |
| Candidate T1.7 | T1.7 | 100k | *Pending M4* | - | - | - | - | - | *Pending* |

---

## 5. Final Confirmation & Speedup Summary

- **Status**: Milestone 1 complete. Baseline measurements, golden fixtures (`golden_10k.json`, `golden_100k.json`), and instrumentation ledger frozen.
- **Milestone 1 Deliverables Attested**:
  1. `graph_fingerprint() -> u64` logical accessor hash implemented and covered by unit tests.
  2. Cargo feature `instrument` implemented with zero-cost when disabled.
  3. Golden fixtures generated and validated with bit-exact reproducibility.
  4. 100k baseline measured (median 36.244 s across 3 runs, spread 10.211 s) and 1M single baseline measured (511.271 s, 845.75 MiB peak RSS, 94.70% recall at efSearch=50).
  5. `results/phase3_5/baseline_10k.json`, `results/phase3_5/baseline_100k.json`, and `results/phase3_5/LOG.md` fully populated.
