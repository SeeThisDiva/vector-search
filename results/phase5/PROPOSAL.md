# Phase 5 Filtered Search Benchmark & Time-Budget Proposal

**Task:** Milestone M3 Task `T3.7` (Gate **G4.5** Review Proposal)  
**Date:** 2026-11-13  
**Anchored Evidence:** [`results/phase5/pilot/pilot_100k.json`](./pilot/pilot_100k.json) and [`results/phase5/pilot/microbench_cost.json`](./pilot/microbench_cost.json)  
**Clean Git Commit:** `84e3c81` (`git_dirty: false`, `measured_this_run: true`)  
**Benchmarking Environment:** 12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.78 GB RAM, Windows x86_64), power mode `balanced`, single-core P-core affinity mask `0x0004`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`  
**Document Status:** Submitted for Dev's **Gate G4.5** review. Per [`results/phase5/G4_DECISIONS.md`](./G4_DECISIONS.md), **no `T3.8`/`T3.9` full-matrix runs have started**, and all checklist boxes remain unticked for Dev.

---

## 1. Executive Summary & Pilot Verification

Per [`results/phase5/DESIGN.md`](./DESIGN.md) §4.2, §7.1 and [`results/phase5/G4_DECISIONS.md`](./G4_DECISIONS.md), before launching the full Phase 5 benchmark matrix (`T3.9`) across both scales ($N \in \{100\text{k}, 1\text{M}\}$), both metadata modes (Independent Mode A, Cluster-Correlated Mode B), the 12-point selectivity ladder, and all strategies on the single reference index configuration ($M=16, \text{efConstruction}=100, \text{seed}=42$), we executed:

1. **Cost Model Microbenchmark (`T3.6`)** (`phase5 --microbench`, `dim=128`, `k=10`, `reps=5` after 1 warmup).
2. **100k Empirical Pilot (`T3.7`)** (`phase5 --pilot`, authentic Texmex SIFT1M 100k prefix, 5 representative selectivities $\sigma \in \{0.05\%, 0.1\%, 1\%, 10\%, 50\%\}$ $\times$ both modes, total wall-clock **$3,443.63\text{ s}$ ($57.39\text{ min}$)** pinned to P-core `0x0004`, `Peak RSS = 131.52 MiB`).
3. **Multi-Core Ground Truth Harness Check (`filtered_exact_knn --threads 0`)**: Unpinned 16-core exact filtered KNN timing on `100k` Mode B $\sigma = 0.50$ ($Q=10,000, k=10$) to measure real parallel speedup for 1M dataset preparation.

### 1.1 Pilot Integrity & Graph Baseline Check

| Baseline Metric | Pilot Measured Value | Phase 4 Reference (`100k M=16 efC=100`) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Dataset & Query Set** | $N=100,000$, $Q=10,000$, $D=128$, $k=10$ | Authentic `data/sift1m` prefix (`load_time_s = 0.067 s`) | Verified (`is_authentic_sift: true`) |
| **Build Time ($M=16, \text{efC}=100$)** | **$29.181\text{ s}$** ($3,426.94\text{ vec/s}$) | $26.066\text{ s}$ | Clean single-core `0x0004` |
| **Graph Fingerprint** | **`0x80af03bd7aebaade`** (`9272634270125107934`) | `0x80af03bd7aebaade` | **Bit-for-bit identical** |
| **Graph-Only Memory** | **$145.57\text{ B/vec}$** ($14,557,056\text{ B} = 13.88\text{ MiB}$) | $145.57\text{ B/vec}$ | Exact match |
| **Unfiltered Sanity (`efSearch=100`)** | **Recall@10 = `0.993670`**, `QPS = 3,316.59` | Recall@10 = `0.993670`, QPS = `3,728.92` | Identical recall |
| **S3 Pre-Filter Recall@10 (All 10 Cells)** | **`1.000000`** ($10/10$ cells, $Q=10,000$, $0.000\%$ fewer-than-$k$) | Filtered Exact KNN ground truth parity | **100% exact parity** |

> **Sub-Query Probing Disclosure in [`pilot_100k.json`](./pilot/pilot_100k.json):** Running the full $Q=10,000$ query set $\times$ 5 repetitions (6 total passes including warmup) across slow low-selectivity graph arms in the pilot would alone have exceeded $6.9\text{ hours}$. To gather clean empirical per-query timings within $\approx 1\text{ hour}$ while adhering strictly to 1 warmup + 5 timed repetitions (`reps = 5`), `pilot_100k.json` ran:
> - **Full $Q=10,000$ (`is_subquery_probe: false`)**: all 10 `S3_pre_filter` arms, all $\sigma=50\%$ HNSW arms, unfiltered `ef=100`, and all 10 filtered ground-truth sets.
> - **Tagged Sub-Query Probes (`is_subquery_probe: true`)**: $Q=1,000$ at $\sigma=10\%$ (`S2`, `S1_cap`, `S1_uncap`), $Q=500$ at $\sigma \le 1\%$ (`S2`, `S1_cap`), and $Q=200$ at $\sigma \le 1\%$ (`S1_uncap`). Every JSON arm records its exact `num_queries` and `is_subquery_probe` flag.

---

## 2. Measured Cost Model (`T3.6`) vs Observed 100k Strategy 3 Scaling

Measured single-core (`0x0004`) microbenchmark constants in [`microbench_cost.json`](./pilot/microbench_cost.json) (`dim=128, k=10, reps=5`, safe `std` Rust with compiler auto-vectorization):

| Parameter | Formula Role | Measured Value |
| :--- | :--- | :---: |
| **`t_word`** | 64-bit bitset word scan cost | **`0.3042 ns/word`** |
| **`t_dist(128)`** | 128-dim L2 squared distance in hot buffer | **`21.0420 ns/dist`** |
| **`t_heap(10)`** | Bound-checked `k=10` max-heap insert/replace | **`2.8360 ns/op`** |
| **`t_cand` ($t_{\text{dist}} + t_{\text{heap}}$)** | Combined per-matching-vector microbench cost | **`23.8780 ns/matching_vector`** |

### 2.1 Theoretical $C_{\text{brute}}$ Prediction vs Empirical 100k `S3_pre_filter` Reality

Per [`DESIGN.md`](./DESIGN.md) §4.2, we compare $C_{\text{brute}}(\sigma, 100\text{k}, 128) = \frac{N}{64} t_{\text{word}} + \sigma N (t_{\text{dist}} + t_{\text{heap}})$ against measured 100k `S3` query times ($Q=10,000$, `reps=5`):

| Mode | $\sigma$ | Matches ($N_\sigma$) | Microbench Predicted (ms) | Pilot Measured Mean (ms) | Pilot Measured p50 (ms) | Empirical / Predicted Ratio | Effective $t_{\text{cand}}$ (ns/vec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Independent | 0.05% | 50 | 0.0017 | 0.0044 | 0.0043 | **2.65x** | 78.93 |
| Independent | 0.1% | 100 | 0.0029 | 0.0066 | 0.0064 | **2.29x** | 60.76 |
| Independent | 1.0% | 1,000 | 0.0244 | 0.0420 | 0.0399 | **1.73x** | 41.55 |
| Independent | 10.0% | 10,000 | 0.2393 | 0.7423 | 0.6892 | **3.10x** | 74.18 |
| Independent | 50.0% | 50,000 | 1.1944 | 3.8405 | 3.6641 | **3.22x** | 76.80 |
| Correlated | 0.05% | 50 | 0.0017 | 0.0044 | 0.0043 | **2.62x** | 77.93 |
| Correlated | 0.1% | 100 | 0.0029 | 0.0077 | 0.0065 | **2.70x** | 72.69 |
| Correlated | 1.0% | 1,000 | 0.0244 | 0.0446 | 0.0397 | **1.83x** | 44.17 |
| Correlated | 10.0% | 10,000 | 0.2393 | 0.9822 | 0.7594 | **4.11x** | 98.18 |
| Correlated | 50.0% | 50,000 | 1.1944 | 3.6733 | 3.5737 | **3.08x** | 73.46 |

**Key Diagnostic Finding:** The L1-hot microbenchmark underestimates real 100k `S3` query latency by **`1.73x` to `4.11x`** (typical ~`3.1x` at high $\sigma$). The working set of `100,000` `f32[128]` base vectors is **`48.83 MiB` (`51,200,000 bytes`)**, exceeding the CPU's $24\text{ MiB}$ L3 cache, and non-contiguous bitset ID jumps defeat hardware sequential prefetching compared to the microbenchmark's tiny L1-resident scratch buffer. At **1M** (`488.3 MiB` base matrix, `~20x` larger than L3 cache), DRAM bandwidth pressure makes `t_cand` at least as large (`~75–100+ ns/vec`). We therefore extrapolate **1M `S3`** directly from the measured 100k `S3` latencies (`10x` expected, `13x` conservative worst-case for DRAM/TLB effects), rather than from unadjusted microbenchmark predictions.

---

## 3. 100k Pilot Quantitative Results & Crossover Analysis

### 3.1 Mode A (Independent Uniform Metadata) at 100k

| $\sigma$ | `S3` Mean (ms) [Recall=1.0] | `S2` @ `ef=20` Recall / Mean (ms) / Fewer% | `S2` @ `ef=100` Recall / Mean (ms) / Fewer% | `S2` @ `ef=600` Recall / Mean (ms) / Fewer% | `S1_cap(4096)` @ `ef=100` Recall / Mean / Cap% | `S1_uncap(N)` @ `ef=100` Recall / Mean / Cap% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05%** | **0.0044** | 0.0568 / 0.2316 / 100.0% | 0.2816 / 2.2286 / 95.6% | 0.8562 / 17.6817 / 35.2% | 0.2094 / 5.2913 / 100.0% | 0.9955 / 29.3847 / 0.0% |
| **0.1%** | **0.0066** | 0.1328 / 0.2931 / 99.8% | 0.6080 / 8.5470 / 66.6% | 0.9942 / 22.2689 / 0.6% | 0.4016 / 5.1985 / 99.6% | 0.9970 / 17.2280 / 0.0% |
| **1.0%** | **0.0420** | 0.9770 / 2.5937 / 4.4% | 0.9996 / 8.3772 / 0.0% | 0.9996 / 23.6991 / 0.0% | 0.9996 / 3.0014 / 0.0% | 1.0000 / 3.3919 / 0.0% |
| **10.0%** | 0.7423 | **0.9932 / 0.4704 / 0.0%** | 0.9988 / 1.6075 / 0.0% | 0.9989 / 5.8098 / 0.0% | 0.9920 / 0.4955 / 0.0% | 0.9920 / 0.4887 / 0.0% |
| **50.0%** | 3.8405 | **0.9574 / 0.1326 / 0.0%** | 0.9961 / 0.4870 / 0.0% | 0.9971 / 1.9519 / 0.0% | 0.9912 / 0.2797 / 0.0% | 0.9912 / 0.2918 / 0.0% |

### 3.2 Mode B (Cluster-Correlated Metadata) at 100k

| $\sigma$ | `S3` Mean (ms) [Recall=1.0] | `S2` @ `ef=20` Recall / Mean (ms) / Fewer% | `S2` @ `ef=100` Recall / Mean (ms) / Fewer% | `S2` @ `ef=600` Recall / Mean (ms) / Fewer% | `S1_cap(4096)` @ `ef=100` Recall / Mean / Cap% | `S1_uncap(N)` @ `ef=100` Recall / Mean / Cap% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05%** | **0.0044** | 0.0110 / 0.2019 / 99.0% | 0.0286 / 1.1493 / 97.8% | 0.0914 / 4.8482 / 91.2% | 0.0136 / 5.3851 / 99.0% | 0.4300 / 83.9713 / 57.0% |
| **0.1%** | **0.0077** | 0.1204 / 0.7325 / 90.6% | 0.3156 / 7.1310 / 71.2% | 0.4962 / 12.6820 / 50.4% | 0.2476 / 5.6913 / 85.6% | 0.4300 / 61.9248 / 57.0% |
| **1.0%** | **0.0446** | 0.2152 / 0.7387 / 79.6% | 0.3430 / 3.4394 / 67.0% | 0.4996 / 10.1585 / 54.6% | 0.2852 / 7.5569 / 75.6% | 0.7510 / 60.9836 / 43.0% |
| **10.0%** | **0.9822** | 0.7810 / 1.2019 / 21.6% | 0.9119 / 3.5817 / 8.8% | 0.9827 / 8.5957 / 1.6% | 0.8903 / 4.0740 / 13.9% | 0.9940 / 5.8876 / 0.0% |
| **50.0%** | 3.6733 | 0.9485 / 0.1658 / 0.0% | **0.9955 / 0.5126 / 0.0%** | 0.9972 / 1.8363 / 0.0% | 0.9889 / 0.3537 / 0.0% | 0.9889 / 0.3601 / 0.0% |

### 3.3 Matched-Recall Crossover Analysis at 100k (`S2` vs `S3`)

Using the [`DESIGN.md`](./DESIGN.md) §4.3 matched-recall rule (smallest `efSearch` on `[20, 50, 100, 200, 400, 600]` meeting target recall vs `S3` exact brute force):

| Metadata Mode | Recall Target | 0.05% Winner | 0.1% Winner | 1.0% Winner | 10.0% Winner | 50.0% Winner | Empirical Crossover Bracket | Coarse Log-Log Proxy $\sigma^*$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A (Independent)** | **0.95** (Primary) | `S3` (no hit, max 0.856) | `S3` (`3,341x` faster vs `ef=400`) | `S3` (`61.8x` faster vs `ef=20`) | **`S2 ef=20`** (`1.58x` vs `S3`) | **`S2 ef=20`** (`28.96x` vs `S3`) | **`[1.0%, 10.0%]`** | $\approx 7.95\%$ |
| **Mode A (Independent)** | **0.99** (Secondary) | `S3` (no hit, max 0.856) | `S3` (`3,374x` faster vs `ef=600`) | `S3` (`137.8x` faster vs `ef=50`) | **`S2 ef=20`** (`1.58x` vs `S3`) | **`S2 ef=50`** (`13.63x` vs `S3`) | **`[1.0%, 10.0%]`** | $\approx 8.23\%$ |
| **Mode B (Correlated)** | **0.95** (Primary) | `S3` (no hit, max 0.091) | `S3` (no hit, max 0.496) | `S3` (no hit, max 0.500) | **`S3`** (`6.90x` faster vs `ef=400`) | **`S2 ef=50`** (`11.73x` vs `S3`) | **`[10.0%, 50.0%]`** | $\approx 20.29\%$ |
| **Mode B (Correlated)** | **0.99** (Secondary) | `S3` (no hit) | `S3` (no hit) | `S3` (no hit) | **`S3`** (no hit, max 0.983) | **`S2 ef=100`** (`7.17x` vs `S3`) | **`[10.0%, 50.0%]`** | Between 10% and 50% (`ef=100`) |

> **Pilot Scientific Insights Driving the Budget Design:**
> 1. **Why the Full 12-Point Ladder Matters inside Crossover Brackets:** Mode A's crossover lies tightly inside `[1%, 10%]` (where the full ladder places `{2.0%, 3.5%, 5.0%, 7.5%}`), and Mode B's crossover lies inside `[10%, 50%]` (where the full ladder places `{20.0%}`). Keeping all 12 selectivity points in the full run pins down both $\sigma^*$ brackets precisely.
> 2. **Structural Reachability Ceiling on Unmodified Layer 0 (`PRESERVING` Consequence):** At correlated $\sigma \in \{0.05\%, 0.1\%, 1\%\}$, even `S1_post_uncapped` (`overfetch_cap = N = 100,000`) failed to return 10 matches on **$43.0\%–57.0\%$** of queries (mean latency `61.0–84.0 ms/query`). In [`opaque_eval_probes`](./pilot/pilot_100k.json), worst-case `S2` traversals stalled identically at **`45,963–54,037`** distinct visited nodes (`~46%–54%` of Layer 0), confirming that asymmetric heuristic neighbor shrinking (`M=16`) leaves a portion of Layer 0 unreachable along directed out-edges from a single entry point. When matching nodes are concentrated in 2–4 correlated clusters, `S3` is the **only** strategy achieving `100%` Recall@10 and `0%` fewer-than-$k$ failure.
> 3. **Why Low-$\sigma$ `S2` Dominates Uncut Runtime:** At $\sigma = 1\%$, once $k=10$ matches are admitted early, `h > h_max` no longer terminates `'traversal` (`DESIGN.md` Algorithm 3 line 12: `H > H_max and W_match.len() < k`), so setting `efSearch = 600` tries to collect `600` of only `1,000` total matching vectors—walking `~50%` of the entire graph (`23.7 ms/query`, `564x` slower than `S3`!). At `1M` and $Q=10,000$, running uncapped/high-$ef$ graph sweeps at $\sigma \le 1\%$ for $10,000$ queries per point consumes days of CPU solely to verify results where `S3` is already orders of magnitude faster.

---

## 4. Metadata Generation & Filtered Ground Truth Preparation Costs

### 4.1 Measured 100k Costs & Exact Scaling Laws

From [`pilot_100k.json`](./pilot/pilot_100k.json) and our unpinned 16-core `filtered_exact_knn` check:

| Selectivity $\sigma$ | Mode | Centroids $C$ | 100k Metadata Gen (s) | 100k GT Gen 1-Core (`0x0004`) (s) | 100k GT Gen 16-Core (`threads=0`) (s) | Raw Bitset Size (B) | GT `.ivecs` Size (B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05%** | Ind / Corr | 0 / 4,000 | 0.001 / **30.146** | 0.057 / 0.048 | — | 12,520 | 440,000 |
| **0.1%** | Ind / Corr | 0 / 2,000 | 0.001 / **15.039** | 0.067 / 0.089 | — | 12,520 | 440,000 |
| **1.0%** | Ind / Corr | 0 / 200 | 0.001 / **1.580** | 0.429 / 0.485 | — | 12,520 | 440,000 |
| **10.0%** | Ind / Corr | 0 / 64 | 0.001 / **0.392** | 8.896 / 8.190 | — | 12,520 | 440,000 |
| **50.0%** | Ind / Corr | 0 / 64 | 0.001 / **0.421** | 35.859 / 35.090 | **28.294** *(CLI wall 29.722 s)* | 12,520 | 440,000 |

- **Mode B Centroid Assignment Law ($\mathcal{O}(C \cdot N \cdot d)$):** Empirically linear at **$7.54\text{ ms}$ per `(centroid × 100k vectors)`** (`30.146 s / 4000 = 7.536 ms/C`; `15.039 s / 2000 = 7.520 ms/C`). Across all 12 tiers of the full ladder (`C = max(64, floor(2 / sigma))` $\implies \sum C = 8,084$ centroids), deterministic Mode B bitset generation takes **$\approx 60.9\text{ s}$ ($1.02\text{ min}$) at 100k** and **$\approx 609.3\text{ s}$ ($10.2\text{ min}$) at 1M** (plus $< 15\text{ s}$ for CLI loading if generated in 12 calls). Mode A takes $< 1.5\text{ s}$ total across both scales.
- **Multi-Core Ground Truth Scaling Finding:** `filtered_exact_knn --threads 0` on all 16 logical cores completed `100k, sigma=0.50, Q=10,000` in **$28.294\text{ s}$** vs **$35.090\text{ s}$** on 1 core—a modest **`1.24x` multi-core speedup** because each thread streams scattered vectors from a `48.8 MiB` array exceeding shared L3 cache (`24 MiB`), saturating memory bandwidth. Extrapolating linearly with $N \cdot \sum \sigma$ (`sum(12-point ladder) = 0.9985`, $\times 2\text{ modes} = 1.9970$, $= 3.994 \times$ the $\sigma=0.5$ workload):
  - **All 24 Filtered GT files (`100k`, 16-core):** $\approx 3.994 \times 28.294\text{ s} \approx \mathbf{113\text{ s}\ (1.88\text{ min})}$.
  - **All 24 Filtered GT files (`1M`, 16-core):** $\approx 10 \times 113\text{ s} \approx \mathbf{1,130\text{ s}\ (18.8\text{ min})}$ expected (**$\approx 25\text{ min}$** worst-case with full 1M L3-miss DRAM pressure).

### 4.2 Disk Space & Memory Budget

| Artifact Class | Count & Formula | 100k Footprint | 1M Footprint | Combined Total | Version Control |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dense Bitsets (`.bin`)** | 24 files/scale ($12 \sigma \times 2$ modes) | $24 \times 12,520\text{ B} = 293.4\text{ KiB}$ | $24 \times 125,020\text{ B} = 2.86\text{ MiB}$ | **$3.15\text{ MiB}$** | Gitignored under `data/` |
| **Metadata Provenance (`.json`)** | 24 files/scale ($\approx 1.5\text{ KiB}$ ea.) | $\approx 36\text{ KiB}$ | $\approx 48\text{ KiB}$ | **$< 0.1\text{ MiB}$** | Gitignored under `data/` |
| **Filtered GT Indices (`.ivecs`)** | 24 files/scale ($10,000 \times 44\text{ B} = 440,000\text{ B}$) | $10.07\text{ MiB}$ | $10.07\text{ MiB}$ | **$20.14\text{ MiB}$** | Gitignored under `data/` |
| **Filtered GT Distances (`.fvecs`)** *(optional)* | 24 files/scale ($440,000\text{ B}$ ea.) | $10.07\text{ MiB}$ | $10.07\text{ MiB}$ | **$20.14\text{ MiB}$** | Gitignored (omit if disk-tight) |
| **Result JSONs & Charts** | `results/phase5/*.json, png, svg` | $< 1\text{ MiB}$ | $< 1\text{ MiB}$ | **$\approx 2–4\text{ MiB}$** | Tracked in Git |
| **Total Disk Budget** | Full cached prep + tracked results | **$20.5\text{ MiB}$** *(10.4 MiB w/o `.fvecs`)* | **$23.1\text{ MiB}$** *(13.0 MiB w/o `.fvecs`)* | **$\approx 46\text{ MiB}$** | Trivial ($< 50\text{ MiB}$) |

- **System RAM Safety:** `100k` pilot measured **`131.52 MiB`** Peak RSS (including `S1_uncapped` `FilteredSearchScratch::new(N, N, N)`). Adding `1M` HNSW graph Peak RSS from Phase 3.5 (**`646.05 MiB`**) plus 1 bitset (`0.12 MiB`) and largest `S1_cap` / `S2` scratch (`< 25 MiB`) keeps `1M` Peak RSS around **`~680–750 MiB`** (**$< 5\%$** of $15.78\text{ GB}$ physical RAM; zero OOM risk).

---

## 5. Full-Matrix Wall-Clock Extrapolation & Pruning Options (Gate G4.5)

Using piecewise log-$\sigma$ interpolation on the 5 pilot $\sigma$ points across the 12-point [`DESIGN.md`](./DESIGN.md) §4.3 ladder ($\{0.05\%, 0.1\%, 0.2\%, 0.5\%, 1.0\%, 2.0\%, 3.5\%, 5.0\%, 7.5\%, 10.0\%, 20.0\%, 50.0\%\}$ $\times$ 2 modes $\times$ 6 timed passes = `1 warmup + 5 reps`), we compare **three candidate execution plans**:

### 5.1 Scaling Assumptions from 100k Pilot to 1M

1. **`S3_pre_filter`:** $10\times$ expected (`13\times` worst-case DRAM pressure at $488\text{ MiB}$ base matrix).
2. **`S2` & `S1_cap` at $\sigma \ge 3.5\%$:** Phase 4 empirically measured **`1.85x`** HNSW latency scaling from `100k` $\to$ `1M` (`2.2x` worst-case).
3. **`S2` at $\sigma \le 2.0\%$:** `3.0x` expected / `6.0x` worst-case (at $\sigma \in \{0.05\%, 0.1\%\}$, `S2 ef=600` attempts to collect up to $600$ of only $500$–$1,000$ total 1M matching vectors, traversing far more of the $10\times$ larger Layer 0 than at higher $\sigma$).
4. **`S1_uncapped` ($K'_{\max}=1,000,000$) at $\sigma \le 2.0\%$:** `6.0x` expected / `10.0x` worst-case (correlated queries triggering full widening walk $\approx 10\times$ more Layer-0 nodes in a `646 MiB` working set).

### 5.2 Budget Comparison Across Execution Options (`100k` + `1M`)

| Option | Matrix & Sampling Policy | 100k Query Suite | 1M Query Suite (Expected / Worst-Case) | Build + Meta + GT + Sensitivity | Combined Total (Expected / Worst-Case) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Option A: Uncut Naive Matrix** | `Q=10,000`, `reps=5` (6 passes) on every arm at both scales, including `S1_uncap` down to $\sigma=0.05\%$ | **`25.33 h`** *(S3: 0.25h, S2: 14.31h, S1c: 1.96h, S1u: 8.81h)* | **`91.98 h` / `154.40 h`** *(S2: 37.8h–68.9h, S1u: 47.2h–77.5h, S3: 2.58h, S1c: 4.38h)* | `0.25 h` (100k) + `0.62 h` (1M) | **`118.2 h` / `180.6 h`** *(4.9–7.5 days)* |
| **Option B: Balanced Hybrid Probes** | Keep all 12 $\sigma$ & all 2 modes; `S3` `Q=10k` everywhere; HNSW `Q=500` ($\sigma \le 1\%$), `Q=1,000` ($2\%–10\%$), `Q=10k@100k / 2.5k@1M` ($\sigma \ge 20\%$); **`100k` `S1_uncap`** `200q` at $\le 1\%$; **omit `1M` `S1_uncap` at $\sigma \le 1\%$**; `reps=5` | **`2.64 h`** *(S3: 0.26h, S2: 1.60h, S1c: 0.37h, S1u: 0.42h)* | **`6.42 h` / `9.48 h`** *(S3: 2.58h, S2: 2.71h–4.59h, S1c: 0.39h, S1u: 0.74h)* | `0.33 h` (100k*) + `0.62 h` (1M) | **`10.0 h` / `13.1 h`** |
| **Option C: Recommended Lean Crossover Matrix (`reps=5`)** | Keep all 12 $\sigma$ & all 2 modes; **`S3`** `Q=10k` @ `100k` & `1M` $\sigma \le 5\%$, **`Q=2,000`** @ `1M` $\sigma \ge 7.5\%$; HNSW **`Q=300`** ($\sigma \le 1\%$), **`Q=1,000`** ($2\%–10\%$), **`Q=2,000`** ($\ge 20\%$); **`S1_uncap`** evaluated only for $\sigma \ge 5\%$ (using 100k pilot proof at $\le 1\%$) | **`1.41 h`** *(S3: 0.26h, S2: 0.91h, S1c: 0.16h, S1u: 0.09h)* | **`3.38 h` / `5.05 h`** *(S3: 0.76h–0.98h, S2: 2.14h–3.52h, S1c: 0.32h–0.36h, S1u: 0.16h–0.20h)* | `0.33 h` (100k*) + `0.62 h` (1M) | **`5.74 h` / `7.41 h`** |
| **Option C-Fast (`reps=3`, Phase 4 parity)** | Same as Option C, using Phase 4's `1 warmup + 3 timed reps` (4 passes vs 6 passes) | **`0.94 h`** | **`2.25 h` / `3.37 h`** | `0.25 h` (100k*) + `0.62 h` (1M) | **`4.06 h` / `5.18 h`** |

*\*Note on Sensitivity Breakdown included above (`100k` `~17 min` at `Q=500, reps=5`): `H_max in {1, 10}*ef` at $\sigma \in \{0.1\%, 1\%, 10\%\}$, `ef in {20, 100, 400}` ($\approx 14.2\text{ min}$, `H_max=3*ef` reused from main matrix); `overfetch_factor` $\gamma \in \{1.0, 2.0, 5.0\}$ at $\sigma \in \{1\%, 10\%\}$, `ef=100` ($\approx 2.3\text{ min}$); opaque `eval_limit` check ($\approx 1\text{ min}$).*

### 5.3 Why Option C Is Statistically Sound for Phase 5 Claims

1. **Zero Loss of Crossover Precision:** Every selectivity point in both crossover brackets (`2%, 3.5%, 5%, 7.5%, 10%` and `20%`) still executes $Q=1,000$ to $2,000$ queries $\times$ `reps`, limiting binomial Recall@10 standard error to $\le 0.0069$ (at Recall $= 0.95$, $Q=1,000$).
2. **`S3` Sub-Sampling Safety:** Because `S3` is an exact linear scan with zero graph-path variance (`Recall@10 = 1.000000` everywhere), $Q=2,000$ at high 1M $\sigma \ge 7.5\%$ saves `1.82 hours` while delivering `10,000–12,000` timed query samples across 5 reps.
3. **Worst-Case Ceiling Controlled:** Under Option C (`reps=5`), even full worst-case DRAM + low-$\sigma$ `S2` pressure completes inside **$7.4\text{ hours}$** (or **$5.2\text{ hours}$** with `reps=3`), and all artifacts remain checkpointable per `(scale, mode, sigma)` JSON just like Phase 4.

---

## 6. Proposed Default `eval_limit` for Opaque `LambdaPredicate` Closures

[`DESIGN.md`](./DESIGN.md) §1.4 & §6.1 leave `FilteredSearchParams::default().eval_limit` as `None` pending a T3.7 empirical proposal grounded in opaque closure invocation counts. Notice in [`src/hnsw.rs`](../../src/hnsw.rs) that `eval_limit` is checked **exclusively when `predicate.bitset().is_none()`**—so fast bitset predicates are never truncated by `eval_limit`.

### 6.1 Empirical Evaluation-Count Distribution (`pilot_100k.json`, $Q=1,000$ queries per row)

| Mode | $\sigma$ | Arm | `mean` Evals | `p95` Evals | `p99` Evals | `max` Evals | `max / ef` Ratio | Role in Phase 5 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Ind | 50.0% | `S2 ef=20` / `ef=100` / `ef=600` | 688 / 2,408 / 8,439 | 964 / 3,295 / 11,309 | 1,084 / 3,554 / 12,155 | 1,266 / 3,867 / 13,302 | 63.3x / 38.7x / 22.2x | Active (`S2` wins) |
| Ind | 10.0% | `S2 ef=20` / `ef=100` / `ef=600` | 2,402 / 7,446 / 22,107 | 3,558 / 10,216 / 27,313 | 4,146 / 11,094 / 29,169 | 5,082 / 12,565 / 32,498 | 254.1x / 125.7x / 54.2x | Active (`S2 ef=20` wins) |
| Corr | 50.0% | `S2 ef=20` / `ef=100` / `ef=600` | 777 / 2,380 / 7,721 | 1,898 / 4,718 / 12,770 | 2,823 / 6,060 / 15,726 | 4,307 / 7,983 / 19,036 | 215.4x / 79.8x / 31.7x | Active (`S2` wins) |
| Corr | 10.0% | `S2 ef=20` / `ef=100` / `ef=600` | 4,726 / 12,661 / 26,038 | 15,195 / 33,109 / 45,051 | 22,184 / 38,697 / 47,658 | 30,744 / 43,035 / 48,989 | 1,537x / 430x / 81.6x | Tail / `S3` wins |
| Ind | $\le 1.0\%$ | `S2 ef=20`..`600` | 1,133–50,218 | 2,038–**54,037** | 2,530–**54,037** | 3,333–**54,037** | up to **1,949.4x** | Runaway (`S3` wins) |
| Corr | $\le 1.0\%$ | `S2 ef=20`..`600` | 1,070–27,711 | 1,259–53,450 | 5,439–53,843 | 15,359–**54,021** | up to **1,451.2x** | Runaway (`S3` wins) |
| All | All | `S1_post_capped (4096) ef=100` | 34–4,096 | 34–4,096 | 34–4,096 | 85–**4,096** | up to 40.96x | Bounded by `overfetch_cap` |

### 6.2 Recommended `eval_limit` Candidate Analysis

- **Recommended Value: `DEFAULT_OPAQUE_EVAL_LIMIT = Some(16_384)`** (exactly $4 \times K'_{\max} = 4 \times 4,096$):
  - **0.0% impact on Strategy 1 capped** (`max = 4,096 < 16,384` everywhere).
  - **0.0% truncation on ALL primary `S2` winning operating points**: preserves $100\%$ of queries for Mode A $\sigma=50\%$ (`ef <= 600`, max $13,302$), Mode A $\sigma=10\%$ (`ef <= 100`, max $12,565$), and Mode B $\sigma=50\%$ (`ef <= 100`, max $7,983$), while letting $>99\%$ of Mode B $\sigma=50\%, \text{ef}=600$ (`p99 = 15,726`) finish cleanly.
  - **Caps worst-case closure DoS**: instead of evaluating `54,037` vectors on `100k` (or `500k+` on `1M`), any runaway low-selectivity or disconnected correlated traversal stops cleanly at `16,384` evaluations and raises `EvalLimitReached`.
- **Tighter Alternative (`Some(10_000)`):** Saves another ~39% worst-case closure work (`0%` hit on Mode A/B $\sigma=50\%$ `ef <= 100` and Mode A $\sigma=10\%$ `ef=20`), while truncating the $\approx 5\%$ tail at Mode A $\sigma=10\%, \text{ef}=100$ (`p95 = 10,216`).
- **API Struct Contract Note:** Currently [`src/filter.rs`](../../src/filter.rs#L476) sets `FilteredSearchParams::default().eval_limit = None`, and its unit test asserts `None`. Note also that `S1_post_uncapped` with an opaque closure can call the closure up to $N$ times if `eval_limit` is `None`. At Gate **G4.5**, Dev can either:
  1. Keep `FilteredSearchParams::default().eval_limit = None` (zero impact on existing tests) and define `pub const DEFAULT_OPAQUE_EVAL_LIMIT: usize = 16_384;` plus `FilteredSearchParams::with_opaque_eval_limit()`, **or**
  2. Update `FilteredSearchParams::default().eval_limit = Some(16_384)` (safest default against opaque closure hangs; only touches `predicate.bitset().is_none()` callers).

---

## 7. Gate G4.5 Decision Checklist for Dev (Unticked)

Please review and indicate your sign-off decisions on the items below before `T3.8` / `T3.9` begin:

1. [ ] **Execution Budget Tier (`T3.9`)**: Approve **Option C (`~5.7 h` expected / `~7.4 h` worst-case)**, **Option C-Fast with `reps=3` (`~4.1 h` / `~5.2 h`)**, **Option B (`~10.0 h` / `~13.1 h`)**, or another custom matrix.
2. [ ] **Opaque `eval_limit` Default Policy**: Approve **`16,384`** (vs `10,000` or keeping `None`), and confirm whether `FilteredSearchParams::default().eval_limit` should become `Some(...)` or stay `None` with a named `DEFAULT_OPAQUE_EVAL_LIMIT` / helper constructor.
3. [ ] **Pre-Computation Disk Caching**: Confirm caching both `DenseBitset` (`.bin`) and filtered GT indices (`.ivecs`, skipping `.fvecs` to save $20.14\text{ MiB}$) under gitignored `data/phase5/` across all 24 `(scale, mode, sigma)` predicates.
4. [ ] **Authorize `T3.8` & `T3.9` Execution**: Proceed to build the resumable `phase5` matrix runner + NumPy/Python filtered parity & plotting scripts (`T3.8`, `T3.10`), and execute the approved Phase 5 matrix (`T3.9`).
