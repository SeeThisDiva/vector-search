# Phase 6 Vector Quantization (`SQ8` & `PQ`) Pilot Audit & `1M` Time-Budget Proposal

**Task:** Milestone M4 Task `T4.6` (Phase 6 `1M` Execution Proposal)  
**Date:** 2026-11-17  
**Anchored Evidence:** [`results/phase6/pilot/microbench_kernels.json`](./pilot/microbench_kernels.json), [`results/phase6/summary_n100000.json`](./summary_n100000.json), and [`results/phase6/arm_n100000_*.json`](./summary_n100000.json)  
**Clean Git Commit:** `9f79165` (`git_dirty: false`, `measured_this_run: true`, `warmup_performed: true`)  
**Benchmarking Environment:** `12th Gen Intel(R) Core(TM) i7-12650H` (16 logical cores, `15.78 GB` RAM, Windows `x86_64`), power mode `balanced`, single P-core affinity mask `0x0004`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`  
**Document Status:** Gate `T4.6` timing audit & `1M` Tier 1+2 execution plan grounded in the completed `100k` matrix; executed `1M` outcomes are reported in [`results/phase6/FINDINGS.md`](./FINDINGS.md).

---

## 1. Executive Summary & `100k` Pilot Verification

Per [`results/phase6/DESIGN.md`](./DESIGN.md) §5.2 (`T4.6`) and [`results/phase6/G5_DECISIONS.md`](./G5_DECISIONS.md), we benchmarked the safe-Rust distance kernel microbenchmark (`T4.2`) and the full `11`-arm `100k` `SIFT1M` prefix matrix (`N = 100,000`, `Q = 10,000`, `D = 128`, `k = 10`, `reps = 3` after `1` untimed warmup, `seed = 42`, `L2Squared`) pinned to P-core `0x0004` on clean commit `9f79165`:

1. **Kernel Selection (`T4.2` — `pilot/microbench_kernels.json`):** Although `sq8_norm_expanded` (**Way B**, `36.9335 ns/eval`) runs **`1.2760x`** faster than `sq8_query_shifted` (**Way A**, `47.1285 ns/eval`) on L2/L3-hot `4,096`-vector buffers, Way B suffers **`63.9910` max absolute error** under adversarial cancellation offsets vs **`4.6115e-5`** for Way A (`1.39e6x` lower error), alongside `23.0x` higher standard-input relative error (`3.9406e-6` vs `1.7156e-7`). Per the `DESIGN.md` §3.2 safety rule, **`query_shifted` (Way A)** (`128.00 B/vec`) is selected as the default SQ8 kernel across all Phase 6 runs. Meanwhile, `pq16_adc` (`9.8310 ns/eval`, `16 B/vec`) and `pq32_adc` (`20.1700 ns/eval`, `32 B/vec`) run **`4.03x`** and **`1.97x`** faster than raw `f32` (`39.6600 ns/eval`, `512 B/vec`).
2. **Observed `100k` Wall-Clock Audit (`11` Arms, `324` Sweep Operating Points):**
   - **Total Build Time (`11` arms):** **`620.80 s` (`10.35 min`)** (`27.30–30.04 s` for `M=16` `f32`/SQ8, `72.65–73.90 s` for `M=32` SQ8, `59.22–87.70 s` for `M=16` PQ).
   - **Total 4-Pass Sweep Time (`1` warmup + `3` timed reps):** **`4,025.75 s` (`67.10 min`)** (`3,021.22 s` timed across `3` reps + `1,004.53 s` median-pass warmup estimate).
   - **Combined Active Compute:** **`4,646.55 s` (`77.44 min` = `1.29 hours`)**, excluding brief per-arm isolated child-process RSS checks and JSON serialization.
3. **Graph Fingerprint & Memory Verification (`100k`):**
   - All `M=16` `Float32GraphThenQuantize` arms reproduced the exact Phase 3.5/4 `100k` golden fingerprint `0x80af03bd7aebaade`; both `M=32` arms reproduced `0xa57cab3db2d6766f`.
   - Isolated query-serving `PeakWorkingSet64` dropped from `74.95 MiB` (`f32_m16`) to **`38.61 MiB`** (`sq8_only_f32graph`, **`-48.5%`**), **`29.22 MiB`** (`pq32_only_f32graph`, **`-61.0%`**), and **`28.13 MiB`** (`pq16_only_f32graph`, **`-62.5%`**).
   - Streaming `.fvecs` `QuantizedDirect` builds (`sq8_only_qbuild`, `pq16_only_qbuild`) cut `build_peak_rss_mb` by **`44.2%`** (`87.70 -> 48.90 MiB`) while `sq8_only_qbuild` held recall within `-0.00012` (`ef=100`) to `-0.00002` (`ef=600`).
4. **Recommended `1M` Plan (`Tier 1 + Tier 2`, `9` Arms, 2-Core Sharded):**
   - Execute all **`8` Tier 1 `M=16` arms** plus **`sq8_only_m32` (Tier 2)** (`9` arms total), narrow `sq8_rerank_f32graph` to **`--rerank-list 20,100,full_beam`** (`36` points instead of `60`, since `100k` proves `R=20` already recovers `100%` of `f32` recall on SQ8), omit/defer Tier 3 (`pq16_only_qbuild` and `sq8_rerank_m32`), and run balanced across two physical P-cores (`0x0004` and `0x0400`).
   - **Projected `1M` Compute (`Tier 1 + Tier 2`, `9` arms):** **`195.95 min` (`3.27 hours`) empirical** | **`252.89 min` (`4.21 hours`) conservative**, bounded in wall-clock by Shard B (`108.38 min` / `1.81 h` empirical, `140.86 min` / `2.35 h` conservative).

---

## 2. `T4.2` Kernel Microbenchmark & Way A vs Way B Safety Selection

Measured in [`results/phase6/pilot/microbench_kernels.json`](./pilot/microbench_kernels.json) (`dim = 128`, `num_vectors = 4,096` L2/L3-hot buffer, `num_evals = 200,000`, `reps = 5` after warmup, `seed = 42`, affinity `0x0004`):

| `kernel_id` | Description | Bytes/Vec | Median (ns/eval) | Min (ns) | Max (ns) | Spread (ns) | Throughput (evals/s) | Speedup vs `f32` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f32_l2_squared` | Raw `f32` L2Squared (8-way unrolled) | `512` | `39.6600` | `38.8085` | `47.4620` | `8.6535` | `25,214,321.73` | `1.00x` |
| `sq8_query_shifted` | **SQ8 Way A:** Query-Shifted Asymmetric $L_2^2$ | `128` | `47.1285` | `44.3215` | `48.3125` | `3.9910` | `21,218,583.24` | `0.84x` |
| `sq8_norm_expanded` | **SQ8 Way B:** Norm-Expanded Asymmetric $L_2^2$ (+4 B/vec) | `132` | `36.9335` | `34.5910` | `37.9405` | `3.3495` | `27,075,690.09` | `1.07x` |
| `pq16_adc` | PQ16 ADC Lookup Table Distance (`16 B/vec`) | `16` | **`9.8310`** | `9.8110` | `11.9375` | `2.1265` | `101,719,051.98` | **`4.03x`** |
| `pq32_adc` | PQ32 ADC Lookup Table Distance (`32 B/vec`) | `32` | **`20.1700`** | `18.9965` | `23.2495` | `4.2530` | `49,578,582.05` | **`1.97x`** |

### 2.1 `way_a_vs_way_b` Numerical Stability Audit & Why Way A Was Adopted

| Metric (`way_a_vs_way_b`) | Way A (`sq8_query_shifted`) | Way B (`sq8_norm_expanded`) | Engineering Decision |
| :--- | :---: | :---: | :--- |
| Speedup (`speedup_way_b_over_way_a`) | `1.0000x` | **`1.2760x`** | Way B is `27.6%` faster on L2-hot vectors (`36.93` vs `47.13 ns`, difference `10.20 ns` $> 2\times$ Way A spread `3.99 ns`), **but fails stability** |
| Standard Max Relative Error (`standard_max_rel_err`) | **`1.7156e-7`** | `3.9406e-6` | Way B shows `23.0x` larger `f32` error even on non-adversarial inputs |
| Adversarial Cancellation Error (`adversarial_offset_max_abs_err`) | **`4.6115e-5`** | **`63.9910`** | Way B suffers catastrophic `f32` cancellation (`63.991` abs error, `1.39e6x` worse) |
| Extra Per-Vector RAM | **`0 B/vec`** | `+4.00 B/vec` | Way A saves `3.81 MiB` at `1M` (`base_norms_sq`) |
| `recommended_default_kernel` | **`"query_shifted"`** | — | **Way A adopted as default** across all `SQ8` arms |

**Why Way B Fails Under Cancellation:** Expanding $\sum_j (\tilde{q}_j - s_j c_{i,j})^2 = \|\tilde{\mathbf{q}}\|^2 - 2\langle \hat{\mathbf{q}}, \mathbf{c}_i\rangle + \|\mathbf{s}\odot\mathbf{c}_i\|^2$ subtracts two large positive quantities ($\|\tilde{\mathbf{q}}\|^2 + \|\mathbf{s}\odot\mathbf{c}_i\|^2$) from the cross-term $2\langle \hat{\mathbf{q}}, \mathbf{c}_i\rangle$. Whenever $\|\mathbf{q} - \mathbf{v}_i\| \ll \|\mathbf{q} - \mathbf{min}\|$, `f32` mantissa bits cancel, producing errors up to `63.9910` (vs `4.6115e-5` for Way A, which sums non-negative squares $d_j^2 \ge 0$ using an L1d-resident `512 B` shifted query buffer). Furthermore, as confirmed in `100k` and `1M` HNSW graph searches, DRAM cache misses dominate `f32` (`512 B/vec`) far more than arithmetic latency, allowing Way A (`128 B/vec`) to achieve higher end-to-end QPS than `f32_m16` anyway.

---

## 3. `100k` Observed Timing Breakdown & `100k -> 1M` Extrapolation Model (`--reps 5`)

At `1M`, `--reps 5` executes **`6` passes** (`1` untimed warmup + `5` timed passes) per operating point, a **`1.5x` pass multiplier** relative to `100k` (`4` passes = `1` warmup + `3` timed). Each build and search phase scales distinctly:

- **PQ Training (`N_train = min(N, 65_536)`):** Identical subsample size (`65,536` vectors, `25` max Lloyd iterations) at `100k` and `1M` -> **`1.0x`** scaling (`~37–50 s`, **not** `10x`). **SQ8 Calibration** scans all $N$ vectors -> **`10.0x`** (`< 0.6 s`).
- **Vector Encoding:** Linear in $N$ -> **`10.0x`** (`~2.4–2.7 s` SQ8, `~30–33 s` PQ).
- **Graph Build:** Anchored directly to verified Phase 4 single-threaded builds (`450.301 s` at `1M` vs `27.297 s` at `100k` for `M=16, efC=100` = **`16.497x`**; `1,288.897 s` at `1M` vs `71.941 s` at `100k` for `M=32, efC=200` = **`17.916x`**).
- **Query Sweep Latency:** Anchored to Phase 4 1-pass `1M / 100k` ratios: **`1.647x`** for `M=16` (`72.98 s` vs `44.32 s`) and **`1.818x`** for `M=32` (`112.06 s` vs `61.63 s`), multiplied by `1.5x` (`6 / 4` passes).
- **Conservative Upper Bound:** Applies `1.25x` margin on empirical `1M` build (for thermal/OS/dual-core contention) and `2.20x` per-pass query scaling (`1.5 * 2.20 = 3.30x` of `100k` sweep duration).

### 3.1 Observed `100k` (`11` Arms) & Projected `1M` Per-Arm Timing Table

*(Note: `100k` 4-pass sweep equals `3` exact timed passes from `run_mean_latencies_ms * 10.0` plus `1` warmup pass estimated as `median mean_ms * 10.0`. `sq8_rerank_f32graph` also shows its approved trimmed `--rerank-list 20,100,full_beam` `36`-point subset directly below the full `60`-point row.)*

| `arm_id` | Pts | `100k` Train / Enc / Graph (s) | `100k` Build (s) | `100k` 4p Sweep (s) | `100k` Total (s / m) | `1M` Build Emp. | `1M` Sweep Emp. (`6p`) | `1M` Total Emp. | `1M` Cons. Bound | `G5` Priority Tier |
| :--- | :---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| `f32_m16` | `12` | `0.00 / 0.00 / 27.30` | `27.30` | `177.17` | `204.46 s` (`3.41 m`) | `7.51 m` | `7.29 m` | **`14.80 m`** | `19.13 m` | **Tier 1 (#1)** |
| `sq8_only_f32graph` | `12` | `0.03 / 0.24 / 27.81` | `28.09` | `150.31` | `178.39 s` (`2.97 m`) | `7.69 m` | `6.19 m` | **`13.88 m`** | `17.88 m` | **Tier 1 (#2)** |
| `sq8_only_qbuild` *(stream)* | `12` | `0.05 / 0.27 / 29.71` | `30.04` | `151.86` | `181.90 s` (`3.03 m`) | `8.22 m` | `6.25 m` | **`14.48 m`** | `18.63 m` | **Tier 1 (#3)** |
| `sq8_rerank_f32graph` *(60 pts)* | `60` | `0.03 / 0.24 / 27.56` | `27.83` | `806.14` | `833.98 s` (`13.90 m`) | `7.62 m` | `33.19 m` | `40.82 m` | `53.87 m` | **Tier 1 (#4, Full)** |
| ↳ *Approved `--rerank-list 20,100,full_beam`* | `36` | *(same build)* | `27.83` | `502.75` *(sub)* | `530.58 s` (`8.84 m`) | `7.62 m` | `20.70 m` | **`28.32 m`** | **`37.18 m`** | **Tier 1 (#4, Exec)** |
| `pq16_only_f32graph` | `12` | `38.44 / 3.24 / 27.16` | `68.84` | `88.30` | `157.14 s` (`2.62 m`) | `8.65 m` | `3.64 m` | **`12.28 m`** | `15.67 m` | **Tier 1 (#5)** |
| `pq32_only_f32graph` | `12` | `49.82 / 3.31 / 34.56` | `87.70` | `103.92` | `191.62 s` (`3.19 m`) | `10.89 m` | `4.28 m` | **`15.16 m`** | `19.32 m` | **Tier 1 (#6)** |
| `pq16_rerank_f32graph` | `60` | `37.13 / 2.96 / 27.52` | `67.62` | `511.92` | `579.54 s` (`9.66 m`) | `8.68 m` | `21.08 m` | **`29.76 m`** | `39.00 m` | **Tier 1 (#7)** |
| `pq32_rerank_f32graph` | `60` | `45.57 / 3.35 / 28.69` | `77.62` | `667.72` | `745.34 s` (`12.42 m`) | `9.21 m` | `27.49 m` | **`36.70 m`** | `48.23 m` | **Tier 1 (#8)** |
| `sq8_only_m32` | `12` | `0.03 / 0.24 / 72.38` | `72.65` | `195.84` | `268.49 s` (`4.47 m`) | `21.66 m` | `8.90 m` | **`30.56 m`** | `37.84 m` | **Tier 2 (#9)** |
| `pq16_only_qbuild` *(stream)* | `12` | `37.29 / 2.98 / 18.94` | `59.22` | `79.07` | `138.29 s` (`2.30 m`) | `6.33 m` | `3.26 m` | `9.58 m` | `12.26 m` | **Tier 3 (Omit 1M)** |
| `sq8_rerank_m32` | `60` | `0.03 / 0.24 / 73.62` | `73.90` | `1,093.51` | `1,167.41 s` (`19.46 m`) | `22.03 m` | `49.70 m` | `71.73 m` | `87.68 m` | **Tier 3 (Defer)** |
| **Total (`11` Arms Observed @ `100k`)** | `324` | — | **`620.80 s`** (`10.35 m`) | **`4,025.75 s`** (`67.10 m`) | **`4,646.55 s` (`77.44 m`)** | `118.47 m` | `171.15 m` | `289.62 m` (`4.83 h`) | `369.42 m` (`6.16 h`) | *Uncut 11-Arm Ref* |

---

## 4. Approved `1M` Tier 1+2 Execution Plan (`9` Arms, 2-P-Core Sharded)

### 4.1 Scope Pruning Rationale (`R`-Trim & Tier 3 Deferral)

1. **Tier 1 + Tier 2 Selection (`9` Arms):** All `8` mandatory `M=16` arms (`f32_m16`, `sq8_only_f32graph`, `sq8_only_qbuild`, `sq8_rerank_f32graph`, `pq16_only_f32graph`, `pq32_only_f32graph`, `pq16_rerank_f32graph`, `pq32_rerank_f32graph`) plus **`sq8_only_m32`** (Tier 2). At `100k`, `sq8_only_m32` pushed `SQ8-Only` past the `0.990` barrier (`0.99008` at `ef=100`, `0.99060` ceiling) at `51.06 MiB` query RSS (`-31.9%` vs `f32_m16`), making `1M` verification of dense-graph `SQ8-Only` high-value (`+30.56 m` empirical).
2. **`sq8_rerank_f32graph` `--rerank-list 20,100,full_beam` (`36` vs `60` points):** Across all `100k` `SQ8` rerank points (`M=16` and `M=32`), **`R = 20` already recovered `100.0%` of `FullBeam` and `f32` recall across every `efSearch in [20..600]`** (`0.99370` at `ef=100`, `0.99712` at `ef=600` for `M=16`), while `R = 30, 50, 100` delivered zero additional recall whenever `efSearch >= R`. Dropping redundant intermediate depths `R in {30, 50}` while retaining `R = 20` (minimum lossless threshold), `R = 100` (`Chart 4` / high-$R$ sanity anchor), and `FullBeam` reduces `sq8_rerank_f32graph` from `60` to `36` sweep points, saving **`12.50 min` empirical (`16.69 min` conservative)** with zero loss of scientific coverage.
3. **Tier 3 Deferrals:**
   - **`pq16_only_qbuild` omitted at `1M`:** Explicitly designated `100k`-only in [`DESIGN.md`](./DESIGN.md) §5.3 (`0.61613` ceiling at `100k`); `sq8_only_qbuild` already proves `< 300 MiB` streaming `.fvecs` build RSS at `1M`.
   - **`sq8_rerank_m32` deferred at `1M`:** Would consume `71.73 m` empirical (`87.68 m` conservative) alone while requiring `912+ B/vec` (`~1 GiB` peak RSS), making it strictly heavier in memory than `f32_m32` (`784 B/vec`) and redundant once `R=20` SQ8 parity is verified on `M=16`.

### 4.2 Balanced 2-P-Core Sharding Schedule (`0x0004` / `0x0400`)

To complete the `9`-arm `1M` matrix comfortably inside `~2–2.5 hours` wall-clock without exhausting RAM or thermal headroom, arms are partitioned across two independent physical P-cores:

| Shard (`processor_affinity_mask`) | Assigned `1M` Arms (`9` Total) | Total Sweep Pts | `1M` Build Emp. | `1M` Sweep Emp. | `1M` Shard Total Emp. | `1M` Shard Cons. Bound |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: |
| **Shard A (`0x0004`)** | `f32_m16`, `sq8_only_f32graph`, `sq8_rerank_f32graph` (`20,100,full_beam`), `sq8_only_m32` | `72` | `44.48 m` | `43.09 m` | **`87.56 m` (`1.46 h`)** | **`112.03 m` (`1.87 h`)** |
| **Shard B (`0x0400`)** | `sq8_only_qbuild`, `pq16_only_f32graph`, `pq32_only_f32graph`, `pq16_rerank_f32graph`, `pq32_rerank_f32graph` | `156` | `45.64 m` | `62.74 m` | **`108.38 m` (`1.81 h`)** | **`140.86 m` (`2.35 h`)** |
| **Aggregate (`9` Arms)** | **Tier 1 (`8`) + Tier 2 (`1`)** | **`228`** | **`90.12 m`** | **`105.82 m`** | **`195.95 m` (`3.27 h`)** | **`252.89 m` (`4.21 h`)** |

---

## 5. `1M` RAM, Disk & Execution Gate Summary

- **Concurrent Peak RAM Budget (`15.78 GB` Host):** Even under the worst-case overlap (`sq8_only_m32` at `~1.0 GiB` build RSS on `0x0004` running concurrently with a `~780 MiB` `f32_graph` PQ/SQ8 arm on `0x0400`), combined peak RSS stays around **`~1.8 GiB` (`~11.4%` of system RAM)**, leaving `> 13 GiB` free headroom. Streaming `sq8_only_qbuild` stays under **`300 MiB`** end-to-end.
- **Disk Budget:** Temporary snapshot binaries created per `Only` arm for isolated query-serving RSS profiling (`~160–390 MiB` max) are overwritten/cleaned per arm; persisted outputs consist solely of `9` arm JSONs, `summary_n1000000.json`, and `8` PNG/SVG chart files (`< 5 MiB` total).
- **Gate Validation Note (Post-Execution):** Executing this exact `9`-arm Tier 1+2 plan under `0x0004` / `0x0400` consumed **`244.06 min` (`4.07 h` total core-minutes: `109.64 min` on `0x0004` + `134.42 min` on `0x0400`, `~2.24 h` sharded wall-clock)** — **inside the `252.89 min` conservative budget** (full analysis in [`FINDINGS.md`](./FINDINGS.md)).
