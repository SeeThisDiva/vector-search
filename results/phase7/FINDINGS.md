# Phase 7 Head-to-Head Benchmark Findings (`vector-search` Rust vs FAISS `IndexHNSWFlat`)

> **Scope & Protocol:** Milestone M5 (`T5.1`–`T5.7`, Gate **G6** approved protocol in [`results/phase7/G6_DECISIONS.md`](./G6_DECISIONS.md)). Head-to-head evaluation of `#![forbid(unsafe_code)]`, `std`-only Rust `HnswIndex` against C++ Meta `faiss-cpu 1.15.1` `IndexHNSWFlat(128, M, METRIC_L2)` across `100k` (`3` builds, `5` query reps) and `1M` (`1` build, `5` query reps) authentic INRIA Texmex `SIFT1M` vectors (`Q = 10,000`, `k = 10`, `seed = 42`).  
> **Verification Anchor:** All `22` JSON reports (`11` at `100k` + `11` at `1M`) and [`results/phase7/matched_recall_summary.json`](./matched_recall_summary.json) in `results/phase7/` were verified on clean commit `9f79165` (`git_dirty == false`, `isolated_process == true`, `warmup_performed == true`, `monotonic_recall == true`).

---

## Executive Summary

1. **Baseline-SSE2 Parity (`SIMDLevel_NONE` vs Default Rust Target):** Stock PyPI `faiss-cpu 1.15.1` on Windows ships compiled statically in `SIMDLevel_NONE` (no `swigfaiss_avx2` module, `avx2_available == false`), while default-target Rust (`rustc 1.99.1`, `opt-level = 3, lto = "thin"`, no `.cargo/config.toml`; host triple was not recorded in the result JSONs, current builds use `x86_64-pc-windows-gnu`) uses baseline SSE2 + LLVM auto-vectorization over 8-way unrolled `as_chunks::<8>()` ([`src/distance.rs`](../../src/distance.rs)). Neither binary uses AVX2 (`AVX2` / `-C target-cpu=native` remains an unmeasured addendum).
2. **Sparse-Graph High-Recall Crossover (`C1`, `M=8, efC=100`):** Rust trails FAISS `C1` at `R* in {0.90, 0.95}` (`0.8655x`/`0.9058x` Mode 1 QPS at `100k`, `0.8121x`/`0.9224x` at `1M`), then **overtakes FAISS at `R* = 0.98`** at both scales: **`1.0392x`** (`100k`) and **`1.2546x` Mode 1 QPS / `1.3793x` lower Mode 2 `p99`** (`1M`: `1,356.90` vs `1,081.52 QPS`, `1.0779` vs `1.4868 ms`). At `100k`, Rust widens its lead to `1.1562x` (`R*=0.99`) and **`1.2582x` QPS / `1.1671x` lower `p99`** (`R*=0.995`); at `1M`, Rust `C1` continues to `0.994700` recall (`ef=600`), while FAISS `C1` caps at `0.989550`.
3. **Baseline & Dense-Graph Gap (`C2`, `C3`) and `keep_pruned_shrink` Root Cause (`C2-nkp`):** Under default settings (`keep_pruned = true`, `keep_pruned_shrink = None -> true`), Rust `C2` (`16, 100`) trails FAISS `C2` (`0.7039x–0.7987x` QPS at `100k`; `0.6241x–0.6781x` at `1M`) and `C3` (`32, 200`) trails FAISS `C3` (`0.7045x–0.8255x` at `100k`; `0.5023x–0.7223x` at `1M`). Disabling reverse-shrink backfill (`C2-nkp`, matching FAISS's non-backfilled shrink) cuts `100k`/`1M` build time by `23.56%`/`31.46%` (`22.876 s` / `428.827 s`), raises `ef=600` recall ceilings (`0.999890` at `100k`, `0.999140` at `1M` — exceeding FAISS `C2`'s `0.998720`), boosts Rust `C2` Mode 1 QPS by `+5.7%..+38.8%` (`100k`) and `+17.0%..+53.0%` (`1M`), and closes the gap vs FAISS `C2` to `0.9208x–0.9772x` (`100k` `R*=0.99..0.995`) and **`0.9279x` (`R*=0.99`) to `1.0371x` QPS / `1.5661x` lower `p99` (`R*=0.995` at `1M`: `1,107.94` vs `1,068.30 QPS`, `1.3535` vs `2.1197 ms`)**.
4. **Isolated Peak RSS Savings (`1.68x–2.21x`) vs OpenMP Parallel Build Loss (`6.31x–8.96x`):** Graph-only memory matches FAISS within `+0.08%..+1.68%` (except `1M C1` at `+6.04%`, `85.44` vs `80.57 B/vec`), while Rust's static `std`-only binary cuts isolated Peak RSS by `~92 MiB` at `100k` (`75.96–93.68` vs `167.89–186.09 MiB`) and **`~524–526 MiB` at `1M`** (`594.31–776.42` vs `1,118.32–1,301.15 MiB`) because `HnswIndex` borrows `&RowMatrix<f32>` without duplicating the `488.28 MiB` base matrix. Conversely, Rust `HnswIndex::build` is strictly single-threaded (`18.163–79.651 s` at `100k`; `313.285–1,562.147 s` at `1M`), whereas FAISS OpenMP `add()` reaches `3.339 s` (`100k`, `T=16`) and `76.360 s` (`1M`, `T=16`) — an **`8.96x` (`100k`) and `8.19x` (`1M`, vs `0x0004` `625.665 s`) practical indexing win for FAISS**.
5. **Multi-Thread Batch Search Scaling (`C2`, `T in {1, 6, 16}`):** Despite per-query `BinaryHeap` allocations in `search_batch_parallel`, Rust achieves `4.87x–6.35x` (`T=6`) / `6.74x–8.89x` (`T=16`) at `100k` and `4.83x–6.78x` (`T=6`) / `6.10x–8.41x` (`T=16`) at `1M`, delivering **`87.3%–93.9%` (`T=6`) and `83.5%–90.5%` (`T=16`) of FAISS OpenMP matched-recall Mode 1 QPS at `1M`** (`8.6k–27.7k` QPS).

---

## 1. Verification, Provenance, SSE2 Caveat & Execution Disclosure

### 1.1 Environment, SIMD Provenance & Fairness Controls

- **Host CPU & OS:** `12th Gen Intel(R) Core(TM) i7-12650H` (`10` physical cores: `6` P-cores + `4` E-cores, `16` logical threads, `15.78 GiB` RAM, `windows/x86_64`).
- **Toolchain & Libraries:** `rustc 1.99.1 (58cf8f9c3 2026-10-13)` (`opt-level = 3, lto = "thin", codegen-units = 1`, no `.cargo/config.toml`, `#![forbid(unsafe_code)]`); `Python 3.13.5`, `faiss-cpu 1.15.1` (`IndexHNSWFlat`, `METRIC_L2`).
- **SSE2 Baseline Caveat (Mandatory Disclosure):** Although the Alder Lake CPU supports `AVX2`, `FMA3`, `BMI2`, and `X86_V3`, every FAISS JSON verifies `simd_level_name == "NONE"` (`simd_level == 0`, `has_dynamic_dispatch == false`, `compiled_simd_levels == 1`, `avx2_available == false`). Stock PyPI `faiss-cpu 1.15.1` on Windows runs baseline SSE2 without AVX2 kernels, matching default-target Rust (SSE2 + LLVM auto-vectorization; host triple was not recorded in the result JSONs, current builds use `x86_64-pc-windows-gnu`). Results characterize **default-artifact SSE2 parity**; neither engine was evaluated with explicit AVX2 (`-C target-cpu=native` vs source/conda-forge AVX2 FAISS).
- **Ground Truth & Local Cache Cross-Check (`100k` & `1M`):** At `1M`, both engines evaluate against official INRIA Texmex `data/sift1m/sift_groundtruth.ivecs` (`10,000` queries, first `10` columns). At `100k`, all `11` reports stamp `ground_truth_source = "custom_ivecs (data/gt_100000_k100.ivecs)"`, a local gitignored cache under `data/` (excluded via [`.gitignore`](../../.gitignore) `data/*.ivecs`) generated via [`ExactKnn::search_parallel_with_base`](../../src/exact_knn.rs#L362) per [`results/phase7/DESIGN.md`](./DESIGN.md) §3.1. Independent verification of `data/gt_100000_k100.ivecs` against `faiss.IndexFlatL2(128)` on the first `100` queries confirmed **`1.000000` top-10 and top-100 recall** and **`10,000/10,000` exact ID match** (`1,000/1,000` top-10 ID match).
- **Power Plan & Concurrency Disclosure:** All `22` reports executed under Windows `balanced` power plan (`Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e`), rather than the `Best Performance` plan recommended in [`results/phase7/DESIGN.md`](./DESIGN.md) §8. Furthermore, single-thread Rust (`0x0004`, P-core 1 SMT0) and single-thread FAISS (`0x0400`, P-core 5 SMT0) ran on distinct physical P-cores, and timestamps (`generated_at`) show paired concurrent execution during `100k` (`C2`, `C3`) and `1M` (`C1`, `C2`, `C3`), whereas `C2-nkp` and all `T=6`/`T=16` runs executed sequentially/solo. Sharing L3 cache (`24 MiB`) and DRAM bandwidth across two `~0.6–1.3 GiB` `1M` processes explains why `1M C2` Rust build time rose to `625.665 s` (vs `450.301 s` solo in Phase 4 and `520.115–526.070 s` solo in `T=6`/`T=16`) and why `1M` `C1`/`C2`/`C3` Mode 2 sweeps exhibit higher `run_p99_ms` variance (`up to 350%–401%` spread) than solo `1M C2-nkp` (`63.4%` max `p99` spread) or `100k` (`10.5%–30.6%`).

### 1.2 Full 22-Report Verification & Phase 4/3.5 Continuity Ledger

All `22/22` reports pass: clean JSON parse, `git_commit == "9f79165"`, `git_dirty == false`, `isolated_process == true`, `peak_rss_scope == "isolated_child_process_high_water_mark"`, `warmup_performed == true`, `monotonic_recall == true`, `12`-point `efSearch` ladder `[10..600]`, `max |Mode 1 Recall@10 - Mode 2 Recall@10| == 0.0` (`14/14` single-thread runs), `recall_at_k_mode2 == null` (`8/8` multi-thread runs), and `100%` fingerprint, `entry_point`, `max_layer`, `graph_only_bytes_per_vector`, and `Recall@10` continuity against Phase 4 (`C1`/`C2`/`C3`) and Phase 3.5 (`C2-nkp` `1M`).

| File | Scale | Engine | Config | Mode (`T`) | Affinity (`omp_max`) | `git_commit` / `dirty` / `iso` | `M` / `efC` / `kps` | Rust Fingerprint (`hex`) | Continuity & Parity Check |
|---|---:|---|---|---|---:|---|---|---|---|
| `rust_n100000_C1.json` | `100k` | rust | `C1` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 8/100/`None` | `0xbf5cb3d639e2ad61` | Exact Phase 4 match (`scale_100000_m8_efc100.json`, fp+12pt R) |
| `rust_n100000_C2.json` | `100k` | rust | `C2` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`None` | `0x80af03bd7aebaade` | Exact Phase 4 match (`scale_100000_m16_efc100.json`, fp+12pt R) |
| `rust_n100000_C3.json` | `100k` | rust | `C3` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 32/200/`None` | `0xa57cab3db2d6766f` | Exact Phase 4 match (`scale_100000_m32_efc200.json`, fp+12pt R) |
| `rust_n100000_C2-nkp.json` | `100k` | rust | `C2-nkp` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`False` | `0xeecb25668ae17f19` | Exact Phase 3.5 `100k` fp match (`M1==M2` R) |
| `rust_n100000_C2_t6.json` | `100k` | rust | `C2` | `multithread` (6) | `0x0555` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`None` | `0x80af03bd7aebaade` | Exact Phase 4 match (`scale_100000_m16_efc100.json`, fp+12pt R) |
| `rust_n100000_C2_t16.json` | `100k` | rust | `C2` | `multithread` (16) | `0xffff` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`None` | `0x80af03bd7aebaade` | Exact Phase 4 match (`scale_100000_m16_efc100.json`, fp+12pt R) |
| `faiss_n100000_C1.json` | `100k` | faiss | `C1` | `single` (1) | `0x0400` (`1`) | `9f79165` / `False` / `True` | 8/100/`False` | n/a | `SIMD=NONE`; `M1==M2` R |
| `faiss_n100000_C2.json` | `100k` | faiss | `C2` | `single` (1) | `0x0400` (`1`) | `9f79165` / `False` / `True` | 16/100/`False` | n/a | `SIMD=NONE`; `M1==M2` R |
| `faiss_n100000_C3.json` | `100k` | faiss | `C3` | `single` (1) | `0x0400` (`1`) | `9f79165` / `False` / `True` | 32/200/`False` | n/a | `SIMD=NONE`; `M1==M2` R |
| `faiss_n100000_C2_t6.json` | `100k` | faiss | `C2` | `multithread` (6) | `0x0555` (`6`) | `9f79165` / `False` / `True` | 16/100/`False` | n/a | `SIMD=NONE`; `R==T1` parity |
| `faiss_n100000_C2_t16.json` | `100k` | faiss | `C2` | `multithread` (16) | `0xffff` (`16`) | `9f79165` / `False` / `True` | 16/100/`False` | n/a | `SIMD=NONE`; `R==T1` parity |
| `rust_n1000000_C1.json` | `1M` | rust | `C1` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 8/100/`None` | `0x2a4259464f4c1acd` | Exact Phase 4 match (`scale_1000000_m8_efc100.json`, fp+12pt R) |
| `rust_n1000000_C2.json` | `1M` | rust | `C2` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`None` | `0x44044886c62bca5e` | Exact Phase 4 match (`scale_1000000_m16_efc100.json`, fp+12pt R) |
| `rust_n1000000_C3.json` | `1M` | rust | `C3` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 32/200/`None` | `0xb8ddb30d73d248c5` | Exact Phase 4 match (`scale_1000000_m32_efc200.json`, fp+12pt R) |
| `rust_n1000000_C2-nkp.json` | `1M` | rust | `C2-nkp` | `single` (1) | `0x0004` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`False` | `0x3f201e949db2694b` | Exact Phase 3.5 `1M` fp+5pt R match (`M1==M2` R) |
| `rust_n1000000_C2_t6.json` | `1M` | rust | `C2` | `multithread` (6) | `0x0555` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`None` | `0x44044886c62bca5e` | Exact Phase 4 match (`scale_1000000_m16_efc100.json`, fp+12pt R) |
| `rust_n1000000_C2_t16.json` | `1M` | rust | `C2` | `multithread` (16) | `0xffff` (`n/a`) | `9f79165` / `False` / `True` | 16/100/`None` | `0x44044886c62bca5e` | Exact Phase 4 match (`scale_1000000_m16_efc100.json`, fp+12pt R) |
| `faiss_n1000000_C1.json` | `1M` | faiss | `C1` | `single` (1) | `0x0400` (`1`) | `9f79165` / `False` / `True` | 8/100/`False` | n/a | `SIMD=NONE`; `M1==M2` R |
| `faiss_n1000000_C2.json` | `1M` | faiss | `C2` | `single` (1) | `0x0400` (`1`) | `9f79165` / `False` / `True` | 16/100/`False` | n/a | `SIMD=NONE`; `M1==M2` R |
| `faiss_n1000000_C3.json` | `1M` | faiss | `C3` | `single` (1) | `0x0400` (`1`) | `9f79165` / `False` / `True` | 32/200/`False` | n/a | `SIMD=NONE`; `M1==M2` R |
| `faiss_n1000000_C2_t6.json` | `1M` | faiss | `C2` | `multithread` (6) | `0x0555` (`6`) | `9f79165` / `False` / `True` | 16/100/`False` | n/a | `SIMD=NONE`; `R==T1` parity |
| `faiss_n1000000_C2_t16.json` | `1M` | faiss | `C2` | `multithread` (16) | `0xffff` (`16`) | `9f79165` / `False` / `True` | 16/100/`False` | n/a | `SIMD=NONE`; `R==T1` parity |

---

## 2. Build Time, OpenMP Asymmetry, Graph Memory & Isolated Peak RSS

### 2.1 Complete `100k` and `1M` Construction & Memory Table

| Scale | Config | Engine | `T` (Affinity) | Build Median (s) | Build Range `[min, max]` (s) | Build QPS (`vec/s`) | Graph-Only (`B/vec`) | Graph Total (`MiB`) | Total w/ Vec (`B/vec`) | Isolated Peak RSS (`MiB`) | `max_layer` / `ep` |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `100k` | `C1` | rust | 1 (`0x0004`) | 18.163 | `[17.872, 18.807]` | 5,505.72 | 81.90 | 7.8104 | 593.90 | 75.96 | 6 / 66038 |
| `100k` | `C1` | faiss | 1 (`0x0400`) | 18.040 | `[17.668, 18.111]` | 5,543.15 | 80.55 | 7.6821 | 592.55 | 167.89 | 5 / 87352 |
| `100k` | `C2` | rust | 1 (`0x0004`) | 29.926 | `[29.829, 30.356]` | 3,341.57 | 145.57 | 13.8827 | 657.57 | 82.11 | 4 / 56204 |
| `100k` | `C2-nkp` | rust | 1 (`0x0004`) | 22.876 | `[22.731, 23.274]` | 4,371.33 | 145.57 | 13.8827 | 657.57 | 81.99 | 4 / 56204 |
| `100k` | `C2` | faiss | 1 (`0x0400`) | 20.368 | `[20.311, 20.528]` | 4,909.67 | 144.25 | 13.7567 | 656.25 | 174.00 | 3 / 51759 |
| `100k` | `C2` | rust | 6 (`0x0555`) *(1T build)* | 27.618 | `[27.358, 29.164]` | 3,620.84 | 145.57 | 13.8827 | 657.57 | 85.61 | 4 / 56204 |
| `100k` | `C2` | faiss | 6 (`0x0555`) *(OMP)* | 4.116 | `[3.926, 4.207]` | 24,298.18 | 144.25 | 13.7567 | 656.25 | 171.84 | 3 / 51759 |
| `100k` | `C2` | rust | 16 (`0xffff`) *(1T build)* | 26.977 | `[26.963, 27.391]` | 3,706.92 | 145.57 | 13.8827 | 657.57 | 95.23 | 4 / 56204 |
| `100k` | `C2` | faiss | 16 (`0xffff`) *(OMP)* | 3.339 | `[3.298, 3.351]` | 29,952.65 | 144.25 | 13.7567 | 656.25 | 174.34 | 3 / 51759 |
| `100k` | `C3` | rust | 1 (`0x0004`) | 79.651 | `[78.806, 80.376]` | 1,255.48 | 273.41 | 26.0741 | 785.41 | 93.68 | 3 / 56204 |
| `100k` | `C3` | faiss | 1 (`0x0400`) | 41.807 | `[41.529, 42.261]` | 2,391.94 | 272.09 | 25.9482 | 784.09 | 186.09 | 3 / 87352 |
| `1M` | `C1` | rust | 1 (`0x0004`) | 313.285 | `[313.285, 313.285]` | 3,191.98 | 85.44 | 81.4792 | 597.44 | 594.31 | 6 / 66038 |
| `1M` | `C1` | faiss | 1 (`0x0400`) | 277.695 | `[277.695, 277.695]` | 3,601.07 | 80.57 | 76.8383 | 592.57 | 1,118.32 | 6 / 966036 |
| `1M` | `C2` | rust | 1 (`0x0004`) | 625.665 | `[625.665, 625.665]` | 1,598.30 | 148.91 | 142.0144 | 660.91 | 654.53 | 5 / 829262 |
| `1M` | `C2-nkp` | rust | 1 (`0x0004`) | 428.827 | `[428.827, 428.827]` | 2,331.94 | 148.91 | 142.0144 | 660.91 | 654.32 | 5 / 829262 |
| `1M` | `C2` | faiss | 1 (`0x0400`) | 404.120 | `[404.120, 404.120]` | 2,474.52 | 144.26 | 137.5728 | 656.26 | 1,180.04 | 5 / 575007 |
| `1M` | `C2` | rust | 6 (`0x0555`) *(1T build)* | 526.070 | `[526.070, 526.070]` | 1,900.89 | 148.91 | 142.0144 | 660.91 | 673.78 | 5 / 829262 |
| `1M` | `C2` | faiss | 6 (`0x0555`) *(OMP)* | 99.225 | `[99.225, 99.225]` | 10,078.07 | 144.26 | 137.5728 | 656.26 | 1,185.52 | 5 / 575007 |
| `1M` | `C2` | rust | 16 (`0xffff`) *(1T build)* | 520.115 | `[520.115, 520.115]` | 1,922.65 | 148.91 | 142.0144 | 660.91 | 719.87 | 5 / 829262 |
| `1M` | `C2` | faiss | 16 (`0xffff`) *(OMP)* | 76.360 | `[76.360, 76.360]` | 13,095.93 | 144.26 | 137.5728 | 656.26 | 1,196.62 | 5 / 575007 |
| `1M` | `C3` | rust | 1 (`0x0004`) | 1,562.147 | `[1562.147, 1562.147]` | 640.14 | 272.33 | 259.7097 | 784.33 | 776.42 | 4 / 829262 |
| `1M` | `C3` | faiss | 1 (`0x0400`) | 782.080 | `[782.080, 782.080]` | 1,278.64 | 272.11 | 259.5067 | 784.11 | 1,301.15 | 4 / 575007 |

### 2.2 Key Build, OpenMP Asymmetry & Memory Findings

- **Single-Thread Build (`T=1`):** At `100k`, Rust ties FAISS on `C1` (`18.163` vs `18.040 s`, `1.007x`), trails on `C2` (`29.926` vs `20.368 s`, `1.469x`), and trails on `C3` (`79.651` vs `41.807 s`, `1.905x`). At `1M`, Rust trails by `1.128x` on `C1` (`313.285` vs `277.695 s`), `1.548x` on `C2` (`625.665` vs `404.120 s` under concurrent execution; Phase 4 solo `450.301 s` = `1.114x`), and `1.997x` on `C3` (`1,562.147` vs `782.080 s`). Disabling reverse-shrink backfill (`C2-nkp`) narrows the `T=1` build gap to **`1.123x` at `100k` (`22.876 s`) and `1.061x` at `1M` (`428.827 s`)**.
- **Multi-Thread OpenMP Build Asymmetry (Where Rust Loses):** Rust `HnswIndex::build` is strictly sequential (`1`-thread) for bit-exact determinism. FAISS `IndexHNSWFlat::add()` uses OpenMP across threads: on `C2`, FAISS drops from `20.368 s` (`T=1`) to `4.116 s` (`T=6`, `4.95x`) and `3.339 s` (`T=16`, `6.10x`) at `100k`, and from `404.120 s` (`T=1`) to `99.225 s` (`T=6`, `4.07x`) and `76.360 s` (`T=16`, `5.29x`) at `1M`. Consequently, **FAISS indexes `8.96x` faster at `100k` (`T=16` vs Rust `0x0004`) and `8.19x` faster at `1M` (`T=16` vs Rust `0x0004`; `6.81x` vs Rust `0xffff` `520.115 s`, or `5.90x` vs Phase 4 solo `450.301 s`)**.
- **Graph-Only Bytes/Vector:** Rust and FAISS are nearly identical in topology footprint. At `100k`: `81.90` vs `80.55 B/vec` (`C1`, `+1.68%`), `145.57` vs `144.25` (`C2`, `+0.92%`), `273.41` vs `272.09` (`C3`, `+0.49%`). At `1M`: `85.44` vs `80.57` (`C1`, `+6.04%`), `148.91` vs `144.26` (`C2`, `+3.22%`), `272.33` vs `272.11` (`C3`, `+0.08%`). Rust stores `node_meta` (`8 B/vec`) + fixed-stride `layer0` (`68`/`132`/`260 B/vec`) + `upper_arena` (`9.44`/`8.91`/`4.33 B/vec` at `1M`), plus `4.00 B/vec` for `VisitedTracker` (`bytes_per_vector` = `89.44`/`152.91`/`276.33 B/vec`). FAISS stores `levels` (`4 B/vec`) + `offsets` (`8 B/vec`) + packed `neighbors` (`68.57`/`132.26`/`260.11 B/vec` at `1M`).
- **Isolated Process Peak RSS (`PeakWorkingSet64`) — Why Rust Saves `~525 MiB` at `1M`:** At `100k`, FAISS uses **`+91.89–92.41 MiB` (`1.99x–2.21x`)** more RSS (`167.89–186.09` vs `75.96–93.68 MiB`). At `1M`, FAISS uses **`+524.01–525.51 MiB` (`1.68x–1.88x`)** more RSS (`1,118.32` vs `594.31 MiB` on `C1`, `1,180.04` vs `654.53 MiB` on `C2`, `1,301.15` vs `776.42 MiB` on `C3`). Root cause directly visible in code: Rust `HnswIndex<'a>` ([`src/hnsw/mod.rs:391`](../../src/hnsw/mod.rs#L391)) borrows `&'a RowMatrix<f32>`, retaining a **single** `488.28 MiB` (`1M x 128 x 4 B`) base matrix in memory (`594.31 MiB` = `488.28 MiB` base + `81.48 MiB` graph + `~24.5 MiB` queries/GT/tracker/binary). By contrast, [`scripts/benchmark_faiss.py`](../../scripts/benchmark_faiss.py) (`lines 488–493, 796`) holds the `488.28 MiB` `base_arr` NumPy array alive in Python while `IndexHNSWFlat.add(base_arr)` copies a **second** `488.28 MiB` matrix into C++ `IndexFlatL2` storage, on top of CPython/NumPy/FAISS/OpenBLAS DLL overhead.

---

## 3. Single-Thread Matched-Recall Head-to-Head (`100k` and `1M`)

Evaluated via [`scripts/benchmark_faiss.py`](../../scripts/benchmark_faiss.py#L744) (`compute_matched_recall_comparison`) and exported to [`results/phase7/matched_recall_summary.json`](./matched_recall_summary.json) (linear interpolation of `log(QPS)`, `log(p99_ms)`, and `log(mean_ms)` vs `Recall@10` strictly inside each curve's observed `[min_recall, max_recall]`, **zero extrapolation**). Mode 1 QPS is pure batch throughput (zero per-query FFI or timer overhead); Mode 2 `p99` includes per-query timing.

### 3.1 Scale `N = 100,000` (`100k`)

#### `100k` C1 vs FAISS C1

- Observed recall bounds: **Rust C1** `[0.690090, 0.998540]`, **FAISS C1** `[0.687260, 0.999020]`.

| Target $R^*$ | Status | Rust C1 M1 QPS | FAISS C1 M1 QPS | M1 QPS Ratio (Rust C1/FAISS C1) | Rust C1 M2 p99 (ms) | FAISS C1 M2 p99 (ms) | p99 Speedup (FAISS C1/Rust C1) | Rust C1 Bracket (`ef@R`) | FAISS C1 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 12,598.36 | 14,555.60 | **0.8655x** | 0.1247 | 0.1128 | **0.9050x** | `ef=30@0.883460`..`ef=40@0.917660` ($\alpha=0.4836$) | `ef=30@0.884970`..`ef=40@0.918210` ($\alpha=0.4522$) |
| `0.95` | `matched` | 8,454.29 | 9,333.42 | **0.9058x** | 0.1798 | 0.1650 | **0.9173x** | `ef=50@0.938630`..`ef=70@0.963100` ($\alpha=0.4647$) | `ef=50@0.939360`..`ef=70@0.961620` ($\alpha=0.4780$) |
| `0.98` | `matched` | 5,204.07 | 5,007.83 | **1.0392x** | 0.2974 | 0.2816 | **0.9467x** | `ef=100@0.979110`..`ef=150@0.990030` ($\alpha=0.0815$) | `ef=100@0.978310`..`ef=150@0.989110` ($\alpha=0.1565$) |
| `0.99` | `matched` | 3,799.50 | 3,286.28 | **1.1562x** | 0.3828 | 0.4292 | **1.1213x** | `ef=100@0.979110`..`ef=150@0.990030` ($\alpha=0.9973$) | `ef=150@0.989110`..`ef=200@0.993350` ($\alpha=0.2099$) |
| `0.995` | `matched` | 2,568.21 | 2,041.18 | **1.2582x** | 0.5717 | 0.6673 | **1.1671x** | `ef=200@0.993800`..`ef=300@0.996950` ($\alpha=0.3810$) | `ef=200@0.993350`..`ef=300@0.996840` ($\alpha=0.4728$) |

#### `100k` C2 vs FAISS C2

- Observed recall bounds: **Rust C2** `[0.821030, 0.997020]`, **FAISS C2** `[0.778610, 0.999900]`.

| Target $R^*$ | Status | Rust C2 M1 QPS | FAISS C2 M1 QPS | M1 QPS Ratio (Rust C2/FAISS C2) | Rust C2 M2 p99 (ms) | FAISS C2 M2 p99 (ms) | p99 Speedup (FAISS C2/Rust C2) | Rust C2 Bracket (`ef@R`) | FAISS C2 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 12,532.03 | 17,505.18 | **0.7159x** | 0.1282 | 0.0986 | **0.7685x** | `ef=10@0.821030`..`ef=20@0.916780` ($\alpha=0.8248$) | `ef=20@0.893560`..`ef=30@0.938930` ($\alpha=0.1419$) |
| `0.95` | `matched` | 8,879.06 | 11,757.67 | **0.7552x** | 0.1670 | 0.1368 | **0.8189x** | `ef=20@0.916780`..`ef=30@0.953980` ($\alpha=0.8930$) | `ef=30@0.938930`..`ef=40@0.962150` ($\alpha=0.4767$) |
| `0.98` | `matched` | 5,892.35 | 7,377.77 | **0.7987x** | 0.2504 | 0.2153 | **0.8599x** | `ef=40@0.971300`..`ef=50@0.980190` ($\alpha=0.9786$) | `ef=50@0.974950`..`ef=70@0.987370` ($\alpha=0.4066$) |
| `0.99` | `matched` | 4,272.86 | 5,371.42 | **0.7955x** | 0.3438 | 0.2798 | **0.8138x** | `ef=70@0.988850`..`ef=100@0.993670` ($\alpha=0.2386$) | `ef=70@0.987370`..`ef=100@0.994020` ($\alpha=0.3955$) |
| `0.995` | `matched` | 2,794.41 | 3,970.03 | **0.7039x** | 0.5263 | 0.3668 | **0.6970x** | `ef=100@0.993670`..`ef=150@0.995770` ($\alpha=0.6333$) | `ef=100@0.994020`..`ef=150@0.997800` ($\alpha=0.2593$) |

#### `100k` C3 vs FAISS C3

- Observed recall bounds: **Rust C3** `[0.909740, 0.999990]`, **FAISS C3** `[0.836150, 0.999970]`.

| Target $R^*$ | Status | Rust C3 M1 QPS | FAISS C3 M1 QPS | M1 QPS Ratio (Rust C3/FAISS C3) | Rust C3 M2 p99 (ms) | FAISS C3 M2 p99 (ms) | p99 Speedup (FAISS C3/Rust C3) | Rust C3 Bracket (`ef@R`) | FAISS C3 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `out_of_range` *(no extrap.)* | — | — | — | — | — | — | `out_of_range` | `ef=10@0.836150`..`ef=20@0.934290` ($\alpha=0.6506$) |
| `0.95` | `matched` | 8,469.99 | 12,023.15 | **0.7045x** | 0.1945 | 0.1420 | **0.7299x** | `ef=10@0.909740`..`ef=20@0.968280` ($\alpha=0.6877$) | `ef=20@0.934290`..`ef=30@0.967480` ($\alpha=0.4733$) |
| `0.98` | `matched` | 6,075.90 | 8,246.72 | **0.7368x** | 0.2618 | 0.2009 | **0.7674x** | `ef=20@0.968280`..`ef=30@0.985560` ($\alpha=0.6782$) | `ef=30@0.967480`..`ef=40@0.981410` ($\alpha=0.8988$) |
| `0.99` | `matched` | 4,844.15 | 6,170.63 | **0.7850x** | 0.3189 | 0.2650 | **0.8308x** | `ef=30@0.985560`..`ef=40@0.992600` ($\alpha=0.6307$) | `ef=50@0.988450`..`ef=70@0.994840` ($\alpha=0.2426$) |
| `0.995` | `matched` | 3,968.18 | 4,806.76 | **0.8255x** | 0.3884 | 0.3404 | **0.8766x** | `ef=40@0.992600`..`ef=50@0.995480` ($\alpha=0.8333$) | `ef=70@0.994840`..`ef=100@0.997850` ($\alpha=0.0532$) |

#### `100k` C2-nkp vs FAISS C2

- Observed recall bounds: **Rust C2-nkp** `[0.799120, 0.999890]`, **FAISS C2** `[0.778610, 0.999900]`.

| Target $R^*$ | Status | Rust C2-nkp M1 QPS | FAISS C2 M1 QPS | M1 QPS Ratio (Rust C2-nkp/FAISS C2) | Rust C2-nkp M2 p99 (ms) | FAISS C2 M2 p99 (ms) | p99 Speedup (FAISS C2/Rust C2-nkp) | Rust C2-nkp Bracket (`ef@R`) | FAISS C2 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 13,248.45 | 17,505.18 | **0.7568x** | 0.1220 | 0.0986 | **0.8082x** | `ef=10@0.799120`..`ef=20@0.907240` ($\alpha=0.9330$) | `ef=20@0.893560`..`ef=30@0.938930` ($\alpha=0.1419$) |
| `0.95` | `matched` | 9,424.99 | 11,757.67 | **0.8016x** | 0.1622 | 0.1368 | **0.8434x** | `ef=30@0.949190`..`ef=40@0.969370` ($\alpha=0.0401$) | `ef=30@0.938930`..`ef=40@0.962150` ($\alpha=0.4767$) |
| `0.98` | `matched` | 6,309.39 | 7,377.77 | **0.8552x** | 0.2282 | 0.2153 | **0.9436x** | `ef=40@0.969370`..`ef=50@0.980150` ($\alpha=0.9861$) | `ef=50@0.974950`..`ef=70@0.987370` ($\alpha=0.4066$) |
| `0.99` | `matched` | 4,946.13 | 5,371.42 | **0.9208x** | 0.2962 | 0.2798 | **0.9447x** | `ef=50@0.980150`..`ef=70@0.990490` ($\alpha=0.9526$) | `ef=70@0.987370`..`ef=100@0.994020` ($\alpha=0.3955$) |
| `0.995` | `matched` | 3,879.40 | 3,970.03 | **0.9772x** | 0.3702 | 0.3668 | **0.9908x** | `ef=70@0.990490`..`ef=100@0.995800` ($\alpha=0.8493$) | `ef=100@0.994020`..`ef=150@0.997800` ($\alpha=0.2593$) |

### 3.2 Scale `N = 1,000,000` (`1M`)

#### `1M` C1 vs FAISS C1

- Observed recall bounds: **Rust C1** `[0.572220, 0.994700]`, **FAISS C1** `[0.570260, 0.989550]`.

| Target $R^*$ | Status | Rust C1 M1 QPS | FAISS C1 M1 QPS | M1 QPS Ratio (Rust C1/FAISS C1) | Rust C1 M2 p99 (ms) | FAISS C1 M2 p99 (ms) | p99 Speedup (FAISS C1/Rust C1) | Rust C1 Bracket (`ef@R`) | FAISS C1 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 4,044.44 | 4,980.19 | **0.8121x** | 0.3667 | 0.4384 | **1.1956x** | `ef=50@0.860720`..`ef=70@0.900410` ($\alpha=0.9897$) | `ef=70@0.894910`..`ef=100@0.926080` ($\alpha=0.1633$) |
| `0.95` | `matched` | 2,445.95 | 2,651.72 | **0.9224x** | 0.5963 | 0.5953 | **0.9984x** | `ef=100@0.933040`..`ef=150@0.960080` ($\alpha=0.6272$) | `ef=100@0.926080`..`ef=150@0.952130` ($\alpha=0.9182$) |
| `0.98` | `matched` | 1,356.90 | 1,081.52 | **1.2546x** | 1.0779 | 1.4868 | **1.3793x** | `ef=200@0.973300`..`ef=300@0.985300` ($\alpha=0.5583$) | `ef=300@0.978240`..`ef=400@0.984040` ($\alpha=0.3034$) |
| `0.99` | `out_of_range` *(no extrap.)* | — | — | — | — | — | — | `ef=300@0.985300`..`ef=400@0.990360` ($\alpha=0.9289$) | `out_of_range` |
| `0.995` | `out_of_range` *(no extrap.)* | — | — | — | — | — | — | `out_of_range` | `out_of_range` |

#### `1M` C2 vs FAISS C2

- Observed recall bounds: **Rust C2** `[0.733150, 0.998040]`, **FAISS C2** `[0.695540, 0.998720]`.

| Target $R^*$ | Status | Rust C2 M1 QPS | FAISS C2 M1 QPS | M1 QPS Ratio (Rust C2/FAISS C2) | Rust C2 M2 p99 (ms) | FAISS C2 M2 p99 (ms) | p99 Speedup (FAISS C2/Rust C2) | Rust C2 Bracket (`ef@R`) | FAISS C2 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 4,544.90 | 6,950.89 | **0.6539x** | 0.3476 | 0.3275 | **0.9421x** | `ef=20@0.849490`..`ef=30@0.900130` ($\alpha=0.9974$) | `ef=30@0.884850`..`ef=40@0.918570` ($\alpha=0.4493$) |
| `0.95` | `matched` | 2,755.84 | 4,350.54 | **0.6334x** | 1.0876 | 0.7316 | **0.6727x** | `ef=50@0.947030`..`ef=70@0.967590` ($\alpha=0.1445$) | `ef=50@0.938650`..`ef=70@0.962440` ($\alpha=0.4771$) |
| `0.98` | `matched` | 1,516.71 | 2,430.23 | **0.6241x** | 1.2385 | 0.5978 | **0.4827x** | `ef=70@0.967590`..`ef=100@0.980610` ($\alpha=0.9531$) | `ef=100@0.978210`..`ef=150@0.988910` ($\alpha=0.1673$) |
| `0.99` | `matched` | 1,130.81 | 1,708.65 | **0.6618x** | 3.4777 | 1.2068 | **0.3470x** | `ef=100@0.980610`..`ef=150@0.990040` ($\alpha=0.9958$) | `ef=150@0.988910`..`ef=200@0.993460` ($\alpha=0.2396$) |
| `0.995` | `matched` | 724.36 | 1,068.30 | **0.6781x** | 1.7264 | 2.1197 | **1.2278x** | `ef=200@0.993760`..`ef=300@0.996540` ($\alpha=0.4460$) | `ef=200@0.993460`..`ef=300@0.996730` ($\alpha=0.4709$) |

#### `1M` C3 vs FAISS C3

- Observed recall bounds: **Rust C3** `[0.846140, 0.999440]`, **FAISS C3** `[0.775200, 0.999390]`.

| Target $R^*$ | Status | Rust C3 M1 QPS | FAISS C3 M1 QPS | M1 QPS Ratio (Rust C3/FAISS C3) | Rust C3 M2 p99 (ms) | FAISS C3 M2 p99 (ms) | p99 Speedup (FAISS C3/Rust C3) | Rust C3 Bracket (`ef@R`) | FAISS C3 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 3,826.10 | 7,617.68 | **0.5023x** | 1.1506 | 0.2276 | **0.1978x** | `ef=10@0.846140`..`ef=20@0.929340` ($\alpha=0.6474$) | `ef=20@0.888180`..`ef=30@0.933900` ($\alpha=0.2585$) |
| `0.95` | `matched` | 3,065.92 | 5,223.07 | **0.5870x** | 0.7204 | 0.3310 | **0.4595x** | `ef=20@0.929340`..`ef=30@0.959890` ($\alpha=0.6763$) | `ef=30@0.933900`..`ef=40@0.956790` ($\alpha=0.7034$) |
| `0.98` | `matched` | 2,356.28 | 3,338.47 | **0.7058x** | 0.6178 | 0.5059 | **0.8189x** | `ef=40@0.974600`..`ef=50@0.982780` ($\alpha=0.6601$) | `ef=50@0.969840`..`ef=70@0.983740` ($\alpha=0.7309$) |
| `0.99` | `matched` | 1,676.36 | 2,422.23 | **0.6921x** | 0.8849 | 0.6988 | **0.7897x** | `ef=50@0.982780`..`ef=70@0.990870` ($\alpha=0.8925$) | `ef=70@0.983740`..`ef=100@0.992040` ($\alpha=0.7542$) |
| `0.995` | `matched` | 1,265.53 | 1,752.07 | **0.7223x** | 1.1771 | 0.9626 | **0.8178x** | `ef=70@0.990870`..`ef=100@0.995720` ($\alpha=0.8515$) | `ef=100@0.992040`..`ef=150@0.996550` ($\alpha=0.6563$) |

#### `1M` C2-nkp vs FAISS C2

- Observed recall bounds: **Rust C2-nkp** `[0.710170, 0.999140]`, **FAISS C2** `[0.695540, 0.998720]`.

| Target $R^*$ | Status | Rust C2-nkp M1 QPS | FAISS C2 M1 QPS | M1 QPS Ratio (Rust C2-nkp/FAISS C2) | Rust C2-nkp M2 p99 (ms) | FAISS C2 M2 p99 (ms) | p99 Speedup (FAISS C2/Rust C2-nkp) | Rust C2-nkp Bracket (`ef@R`) | FAISS C2 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 5,315.73 | 6,950.89 | **0.7648x** | 0.2841 | 0.3275 | **1.1524x** | `ef=30@0.891700`..`ef=40@0.924010` ($\alpha=0.2569$) | `ef=30@0.884850`..`ef=40@0.918570` ($\alpha=0.4493$) |
| `0.95` | `matched` | 3,413.07 | 4,350.54 | **0.7845x** | 0.4500 | 0.7316 | **1.6258x** | `ef=50@0.943010`..`ef=70@0.965470` ($\alpha=0.3112$) | `ef=50@0.938650`..`ef=70@0.962440` ($\alpha=0.4771$) |
| `0.98` | `matched` | 2,075.28 | 2,430.23 | **0.8539x** | 0.6354 | 0.5978 | **0.9408x** | `ef=70@0.965470`..`ef=100@0.980130` ($\alpha=0.9911$) | `ef=100@0.978210`..`ef=150@0.988910` ($\alpha=0.1673$) |
| `0.99` | `matched` | 1,585.42 | 1,708.65 | **0.9279x** | 0.8551 | 1.2068 | **1.4113x** | `ef=100@0.980130`..`ef=150@0.990180` ($\alpha=0.9821$) | `ef=150@0.988910`..`ef=200@0.993460` ($\alpha=0.2396$) |
| `0.995` | `matched` | 1,107.94 | 1,068.30 | **1.0371x** | 1.3535 | 2.1197 | **1.5661x** | `ef=200@0.994410`..`ef=300@0.997610` ($\alpha=0.1844$) | `ef=200@0.993460`..`ef=300@0.996730` ($\alpha=0.4709$) |

### 3.3 Cross-Scale Single-Thread Takeaways (`100k` -> `1M`)

1. **`C1` (`M=8, efC=100`) High-Recall Crossover Holds & Strengthens at `1M`:** On sparse `M=8` graphs, Rust crosses ahead of FAISS in Mode 1 pure-batch QPS between `R* = 0.95` and `R* = 0.98` at both scales (`0.9058x -> 1.0392x` at `100k`; `0.9224x -> 1.2546x` at `1M`). At `1M R* = 0.98`, Rust delivers **`1,356.90` vs `1,081.52` Mode 1 QPS (`+25.46%`)** and **`1.0779` vs `1.4868 ms` Mode 2 `p99` (`1.3793x` lower tail)** — and at `ef=400..600`, Rust reaches `0.990360..0.994700` recall whereas FAISS `C1` saturates at `0.989550` (`ef=600`).
2. **`C2` (`M=16, efC=100`) and `C3` (`M=32, efC=200`) Default Deficit:** On `100k` and `1M`, default Rust `C2` (`0.7039x–0.7987x` at `100k`, `0.6241x–0.6781x` at `1M`) and `C3` (`0.7045x–0.8255x` at `100k`, `0.5023x–0.7223x` at `1M`) trail FAISS across all matched targets. Note two important caveats on `1M`: (a) `1M C2` Mode 2 `p99` shows non-monotonic jitter from concurrent `0x0004`/`0x0400` runs (`0.3470x` at `R*=0.99` due to Rust `ef=150` `3.4934 ms` spike vs Phase 4 solo `0.7716 ms`, flipping to `1.2278x` at `R*=0.995` due to FAISS `ef=200..300` `1.8754..2.4322 ms` vs Rust `1.5165..2.0281 ms`), and (b) on `C3` (`M=32, efC=200`), Rust's gap shrinks cleanly as recall rises (`0.5023x` at `R*=0.90` -> `0.7223x` at `R*=0.995` on `1M`; `0.7045x` at `R*=0.95` -> `0.8255x` at `R*=0.995` on `100k`), because Rust reaches high recall at much lower `efSearch` (`ef=100` reaches `0.995720` in Rust vs `ef=150` `0.996550` in FAISS).

---

## 4. `C2` vs `C2-nkp`: Isolating the `keep_pruned_shrink` Root Cause

Why does default Rust `C2` (`M=16, efC=100`) trail FAISS `C2` at mid-to-high recall and plateau earlier (`0.997020` at `100k`, `0.998040` at `1M`) while `C1` (`M=8`) beats FAISS at high recall? In [`src/hnsw/mod.rs:48-62`](../../src/hnsw/mod.rs#L48-L62), `HnswConfig::default()` sets `keep_pruned = true` and `keep_pruned_shrink = None`, so `effective_keep_pruned_shrink()` (`src/hnsw/mod.rs:99`) evaluates to `true`. Consequently, Rust backfills pruned candidates up to degree `M` / `M_max0` in **both** forward insertion (`src/hnsw/mod.rs:1218`) and reverse-edge shrink (`src/hnsw/mod.rs:1272`), whereas FAISS `IndexHNSWFlat` disables `keep_pruned` backfill (`keep_pruned = false`, `keep_pruned_shrink = false`, plus `prune_headroom = 0.2`). On `M=16`, backfilling reverse-edge shrink saturates Layer-0 slots early with non-diverse short-range edges, increasing per-hop distance evaluations at fixed `efSearch` while paradoxically trapping high-`ef` beams in redundant local neighborhoods (a slight recall boost at `ef=10..30`, followed by earlier saturation above `0.99`). On sparse `M=8` (`C1`), by contrast, keeping pruned edges prevents disconnection at `ef >= 200`, helping Rust overtake FAISS!

### Internal Ablation: `Rust C2-nkp` (`keep_pruned_shrink = false`) vs `Rust C2` (Default)

#### `100k` Rust `C2-nkp` vs Rust `C2`

- Observed recall bounds: **Rust C2-nkp** `[0.799120, 0.999890]`, **Rust C2** `[0.821030, 0.997020]`.

| Target $R^*$ | Status | Rust C2-nkp M1 QPS | Rust C2 M1 QPS | M1 QPS Ratio (Rust C2-nkp/Rust C2) | Rust C2-nkp M2 p99 (ms) | Rust C2 M2 p99 (ms) | p99 Speedup (Rust C2/Rust C2-nkp) | Rust C2-nkp Bracket (`ef@R`) | Rust C2 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 13,248.45 | 12,532.03 | **1.0572x** | 0.1220 | 0.1282 | **1.0516x** | `ef=10@0.799120`..`ef=20@0.907240` ($\alpha=0.9330$) | `ef=10@0.821030`..`ef=20@0.916780` ($\alpha=0.8248$) |
| `0.95` | `matched` | 9,424.99 | 8,879.06 | **1.0615x** | 0.1622 | 0.1670 | **1.0299x** | `ef=30@0.949190`..`ef=40@0.969370` ($\alpha=0.0401$) | `ef=20@0.916780`..`ef=30@0.953980` ($\alpha=0.8930$) |
| `0.98` | `matched` | 6,309.39 | 5,892.35 | **1.0708x** | 0.2282 | 0.2504 | **1.0973x** | `ef=40@0.969370`..`ef=50@0.980150` ($\alpha=0.9861$) | `ef=40@0.971300`..`ef=50@0.980190` ($\alpha=0.9786$) |
| `0.99` | `matched` | 4,946.13 | 4,272.86 | **1.1576x** | 0.2962 | 0.3438 | **1.1608x** | `ef=50@0.980150`..`ef=70@0.990490` ($\alpha=0.9526$) | `ef=70@0.988850`..`ef=100@0.993670` ($\alpha=0.2386$) |
| `0.995` | `matched` | 3,879.40 | 2,794.41 | **1.3883x** | 0.3702 | 0.5263 | **1.4215x** | `ef=70@0.990490`..`ef=100@0.995800` ($\alpha=0.8493$) | `ef=100@0.993670`..`ef=150@0.995770` ($\alpha=0.6333$) |

#### `1M` Rust `C2-nkp` vs Rust `C2`

- Observed recall bounds: **Rust C2-nkp** `[0.710170, 0.999140]`, **Rust C2** `[0.733150, 0.998040]`.

| Target $R^*$ | Status | Rust C2-nkp M1 QPS | Rust C2 M1 QPS | M1 QPS Ratio (Rust C2-nkp/Rust C2) | Rust C2-nkp M2 p99 (ms) | Rust C2 M2 p99 (ms) | p99 Speedup (Rust C2/Rust C2-nkp) | Rust C2-nkp Bracket (`ef@R`) | Rust C2 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 5,315.73 | 4,544.90 | **1.1696x** | 0.2841 | 0.3476 | **1.2233x** | `ef=30@0.891700`..`ef=40@0.924010` ($\alpha=0.2569$) | `ef=20@0.849490`..`ef=30@0.900130` ($\alpha=0.9974$) |
| `0.95` | `matched` | 3,413.07 | 2,755.84 | **1.2385x** | 0.4500 | 1.0876 | **2.4168x** | `ef=50@0.943010`..`ef=70@0.965470` ($\alpha=0.3112$) | `ef=50@0.947030`..`ef=70@0.967590` ($\alpha=0.1445$) |
| `0.98` | `matched` | 2,075.28 | 1,516.71 | **1.3683x** | 0.6354 | 1.2385 | **1.9491x** | `ef=70@0.965470`..`ef=100@0.980130` ($\alpha=0.9911$) | `ef=70@0.967590`..`ef=100@0.980610` ($\alpha=0.9531$) |
| `0.99` | `matched` | 1,585.42 | 1,130.81 | **1.4020x** | 0.8551 | 3.4777 | **4.0672x** | `ef=100@0.980130`..`ef=150@0.990180` ($\alpha=0.9821$) | `ef=100@0.980610`..`ef=150@0.990040` ($\alpha=0.9958$) |
| `0.995` | `matched` | 1,107.94 | 724.36 | **1.5295x** | 1.3535 | 1.7264 | **1.2756x** | `ef=200@0.994410`..`ef=300@0.997610` ($\alpha=0.1844$) | `ef=200@0.993760`..`ef=300@0.996540` ($\alpha=0.4460$) |

- **At `100k`:** `C2-nkp` cuts build time by **`23.56%`** (`29.926 -> 22.876 s`), raises `ef=600` recall from `0.997020` to **`0.999890`** (matching FAISS `0.999900`), speeds up Mode 1 QPS over default `C2` by **`+5.7%` (`R*=0.90`) to `+38.8%` (`R*=0.995`: `3,879.40` vs `2,794.41 QPS`)**, and narrows the gap vs FAISS `C2` to `0.9208x` (`R*=0.99`) and **`0.9772x` QPS / `0.9908x` `p99` at `R*=0.995`**.
- **At `1M`:** `C2-nkp` cuts build time by **`31.46%`** (`625.665 -> 428.827 s`, only `6.1%` slower than FAISS `404.120 s`), raises `ef=600` recall from `0.998040` to **`0.999140`** (exceeding FAISS `C2`'s `0.998720`), boosts Rust `C2` Mode 1 QPS by **`+17.0%` (`R*=0.90`) to `+53.0%` (`R*=0.995`: `1,107.94` vs `724.36 QPS`)**, and **overtakes FAISS `C2` at `R*=0.995`** with **`1.0371x` Mode 1 QPS (`1,107.94` vs `1,068.30`) and `1.5661x` lower Mode 2 `p99` (`1.3535` vs `2.1197 ms`)**. *(Note: the `4.0672x` Mode 2 `p99` speedup vs `Rust C2` at `1M R*=0.99` partly reflects `Rust C2`'s concurrent `3.4934 ms` jitter spike at `ef=150`; against Phase 4 solo `C2` `0.7716 ms`, `C2-nkp`'s `0.8597 ms` at `ef=150` reflects parity with higher recall `0.990180` vs `0.990040`, while Mode 1 QPS `1,577.72` vs `1,129.48` is `+39.7%` faster).*

---

## 5. FFI vs Timer Overhead (`ffi_delta_us` vs `instant_overhead_us`) & `ndis` Scaling

### 5.1 Timing Boundary Audit (`14` Single-Thread Reports)

| Scale | Report | Metric | Min (`us/q`) | Median (`us/q`) | Mean (`us/q`) | Max (`us/q`) | Diagnostic Counters (`ndis/q`, `nhops/q`, `ns/ndis`) |
|---:|---|---|---:|---:|---:|---:|---|
| `100k` | `rust_n100000_C1.json` | `instant_overhead_us` (`ffi=0.0`) | -4.6349 | -0.4272 | -0.6766 | 1.6528 | `null` (built without `--features instrument`) |
| `100k` | `rust_n100000_C2.json` | `instant_overhead_us` (`ffi=0.0`) | -6.1326 | -1.9467 | -1.3985 | 7.5142 | `null` (built without `--features instrument`) |
| `100k` | `rust_n100000_C3.json` | `instant_overhead_us` (`ffi=0.0`) | -31.0516 | -0.3376 | 1.8445 | 35.5299 | `null` (built without `--features instrument`) |
| `100k` | `rust_n100000_C2-nkp.json` | `instant_overhead_us` (`ffi=0.0`) | -8.5009 | -0.4481 | -1.3499 | 0.8958 | `null` (built without `--features instrument`) |
| `100k` | `faiss_n100000_C1.json` | `ffi_delta_us` | 2.9714 | 7.0579 | 8.2927 | 15.2084 | `ndis`: `[200.0, 3539.1]`; `nhops`: `[23.4, 610.1]`; `ns/ndis`: `[139.5, 437.4]` |
| `100k` | `faiss_n100000_C2.json` | `ffi_delta_us` | 3.0643 | 5.5838 | 6.8822 | 17.1302 | `ndis`: `[293.3, 4773.2]`; `nhops`: `[20.6, 607.8]`; `ns/ndis`: `[117.6, 355.9]` |
| `100k` | `faiss_n100000_C3.json` | `ffi_delta_us` | -8.4680 | 7.5328 | 8.1544 | 24.3771 | `ndis`: `[393.1, 6126.1]`; `nhops`: `[19.6, 607.2]`; `ns/ndis`: `[113.7, 311.3]` |
| `1M` | `rust_n1000000_C1.json` | `instant_overhead_us` (`ffi=0.0`) | -11.3149 | -0.6841 | -0.2686 | 6.7830 | `null` (built without `--features instrument`) |
| `1M` | `rust_n1000000_C2.json` | `instant_overhead_us` (`ffi=0.0`) | -365.6130 | 6.2684 | -6.3175 | 246.0576 | `null` (built without `--features instrument`) |
| `1M` | `rust_n1000000_C3.json` | `instant_overhead_us` (`ffi=0.0`) | -27.6339 | 0.8341 | 3.2722 | 60.7407 | `null` (built without `--features instrument`) |
| `1M` | `rust_n1000000_C2-nkp.json` | `instant_overhead_us` (`ffi=0.0`) | -41.7723 | -2.3100 | -7.3491 | 11.0998 | `null` (built without `--features instrument`) |
| `1M` | `faiss_n1000000_C1.json` | `ffi_delta_us` | -4.4958 | 8.1429 | 12.1605 | 68.8127 | `ndis`: `[236.6, 4573.9]`; `nhops`: `[26.7, 612.7]`; `ns/ndis`: `[189.3, 428.4]` |
| `1M` | `faiss_n1000000_C2.json` | `ffi_delta_us` | -74.2626 | 5.3109 | -1.7313 | 31.5554 | `ndis`: `[365.9, 6991.6]`; `nhops`: `[23.9, 610.6]`; `ns/ndis`: `[161.5, 333.9]` |
| `1M` | `faiss_n1000000_C3.json` | `ffi_delta_us` | 2.0115 | 10.8136 | 12.4810 | 33.4295 | `ndis`: `[539.6, 9918.8]`; `nhops`: `[21.4, 608.6]`; `ns/ndis`: `[145.8, 279.0]` |

- **Aggregate FAISS Python/SWIG FFI Tax (`ffi_delta_us`):** Across `36` points at `100k`, median **`6.3050 us/q`** (mean `7.7764`, range `[-8.4680, 24.3771]`; [`scripts/plot_phase7.py`](../../scripts/plot_phase7.py) `sorted[18]` = `6.60 us/q`). Across `36` points at `1M`, median **`7.9195 us/q`** (mean `7.6367`, range `[-74.2626, 68.8127]`; `plot_phase7.py` `sorted[18]` = `8.09 us/q`). Combined (`72` points): median **`7.0058 us/q`** (mean `7.7066`; `plot_phase7.py` `sorted[36]` = `7.23 us/q`).
- **Aggregate Rust Per-Query `Instant::now()` Overhead (`instant_overhead_us`):** `ffi_delta_us == 0.0000` everywhere. Median `instant_overhead_us` is **`-0.4528 us/q`** at `100k` (mean `-0.3951`, range `[-31.0516, 35.5299]`) and **`0.0272 us/q`** at `1M` (mean `-2.6657`, range `[-365.6130, 246.0576]`). Large negative/positive extrema on `1M C2` (`-365.61 / +246.06 us/q` in Rust, `-74.26 / +31.56 us/q` in FAISS) reflect run-to-run jitter between separate 5-rep Mode 1 and Mode 2 sweeps under concurrent `1M` execution, exceeding the `~7–8 us/q` microsecond boundary cost.
- **Headline Robustness Check:** Every claimed Rust win (`C1` at `100k` `R*=0.98..0.995` and `1M` `R*=0.98`, `C2-nkp` at `1M` `R*=0.995`) comes from **Mode 1 Pure Batch** (`zero` per-query FFI or timer calls), and the accompanying Mode 2 `p99` margins (`95.6 us` on `100k C1` `R*=0.995`; `408.9 us` on `1M C1` `R*=0.98`; `766.2 us` on `1M C2-nkp` `R*=0.995`) exceed the `~6–8 us/q` FFI tax by `12x–100x`.

### 5.2 Distance Evaluation (`ndis`) & Memory-Hierarchy Scaling (`100k` -> `1M`, FAISS)

*(Note: Rust `dist_evals_mean` is `null` in all `12` Phase 7 Rust JSONs because the release binary ran without `--features instrument`; FAISS reports exact `faiss.cvar.hnsw_stats.ndis / Q` and `nhops / Q` in an un-timed diagnostic pass.)*

| Config | `efSearch` | `100k` `ndis/q` | `1M` `ndis/q` (`1M/100k`) | `100k` `nhops/q` | `1M` `nhops/q` (`1M/100k`) | `100k` `ns/ndis` (M1) | `1M` `ns/ndis` (M1) (`1M/100k`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C1` | 10 | 200.0 | 236.6 (`1.18x`) | 23.4 | 26.7 (`1.14x`) | 139.5 | 189.3 (`1.36x`) |
| `C1` | 50 | 532.0 | 620.7 (`1.17x`) | 61.7 | 65.8 (`1.07x`) | 177.2 | 270.3 (`1.53x`) |
| `C1` | 100 | 895.0 | 1,049.2 (`1.17x`) | 110.9 | 114.4 (`1.03x`) | 208.9 | 263.1 (`1.26x`) |
| `C1` | 200 | 1,530.1 | 1,842.2 (`1.20x`) | 210.4 | 213.5 (`1.01x`) | 255.6 | 287.5 (`1.12x`) |
| `C1` | 600 | 3,539.1 | 4,573.9 (`1.29x`) | 610.1 | 612.7 (`1.00x`) | 437.4 | 428.4 (`0.98x`) |
| `C2` | 10 | 293.3 | 365.9 (`1.25x`) | 20.6 | 23.9 (`1.16x`) | 117.6 | 161.5 (`1.37x`) |
| `C2` | 50 | 776.2 | 975.0 (`1.26x`) | 58.9 | 62.3 (`1.06x`) | 153.6 | 199.3 (`1.30x`) |
| `C2` | 100 | 1,290.4 | 1,662.2 (`1.29x`) | 108.3 | 111.5 (`1.03x`) | 175.9 | 234.3 (`1.33x`) |
| `C2` | 200 | 2,159.5 | 2,897.6 (`1.34x`) | 208.1 | 211.0 (`1.01x`) | 215.1 | 259.1 (`1.20x`) |
| `C2` | 600 | 4,773.2 | 6,991.6 (`1.46x`) | 607.8 | 610.6 (`1.00x`) | 355.9 | 333.9 (`0.94x`) |
| `C3` | 10 | 393.1 | 539.6 (`1.37x`) | 19.6 | 21.4 (`1.10x`) | 113.7 | 145.8 (`1.28x`) |
| `C3` | 50 | 1,064.1 | 1,448.7 (`1.36x`) | 58.0 | 59.8 (`1.03x`) | 141.3 | 168.3 (`1.19x`) |
| `C3` | 100 | 1,756.0 | 2,459.0 (`1.40x`) | 107.7 | 109.2 (`1.01x`) | 162.1 | 181.9 (`1.12x`) |
| `C3` | 200 | 2,888.9 | 4,234.9 (`1.47x`) | 207.4 | 208.9 (`1.01x`) | 192.0 | 204.5 (`1.07x`) |
| `C3` | 600 | 6,126.1 | 9,918.8 (`1.62x`) | 607.2 | 608.6 (`1.00x`) | 311.3 | 279.0 (`0.90x`) |

- **Sublinear `ndis` vs DRAM `ns/ndis` Growth:** Growing the corpus `10x` (`100k -> 1M`) leaves Layer-0 hop counts `nhops` virtually flat (`1.00x–1.16x`) and grows `ndis` by only **`1.17x–1.62x`**. However, the `1M` working set (`488.28 MiB` base + `77–260 MiB` graph) exceeds the `24 MiB` shared L3 cache by `>23x` (vs `~2.6x` at `100k`), increasing Mode 1 `ns/ndis` at low/mid `ef` (`ef <= 100`) by **`1.26x–1.53x`** (`113.7–208.9 ns` -> `145.8–270.3 ns`). Across all `efSearch`, `ns/ndis` rises with `ef` (`117.6 -> 355.9 ns` on `100k C2`) because `VisitedTable` deduplication lowers new evaluations per hop (`14.2` at `ef=10` -> `7.85` at `ef=600`), amortizing per-hop heap/table overhead over fewer new distances.

---

## 6. Multi-Thread Search Scaling (`C2`, `T in {1, 6, 16}` at `100k` and `1M`)

### 6.1 Matched-Recall Mode 1 Batch QPS (`T = 6` on `0x0555`, `T = 16` on `0xffff`)

#### `100k` C2 T=6 vs FAISS C2 T=6 (`0x0555`)

- Observed recall bounds: **Rust C2 T=6** `[0.821030, 0.997020]`, **FAISS C2 T=6** `[0.778610, 0.999900]`.

| Target $R^*$ | Status | Rust C2 T=6 M1 QPS | FAISS C2 T=6 M1 QPS | M1 QPS Ratio (Rust C2 T=6/FAISS C2 T=6) | Rust C2 T=6 M1 Mean (ms) | FAISS C2 T=6 M1 Mean (ms) | Rust C2 T=6 Bracket (`ef@R`) | FAISS C2 T=6 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 78,110.69 | 86,880.01 | **0.8991x** | 0.0128 | 0.0115 | `ef=10@0.821030`..`ef=20@0.916780` ($\alpha=0.8248$) | `ef=20@0.893560`..`ef=30@0.938930` ($\alpha=0.1419$) |
| `0.95` | `matched` | 44,974.99 | 57,005.44 | **0.7890x** | 0.0222 | 0.0175 | `ef=20@0.916780`..`ef=30@0.953980` ($\alpha=0.8930$) | `ef=30@0.938930`..`ef=40@0.962150` ($\alpha=0.4767$) |
| `0.98` | `matched` | 29,731.59 | 38,168.92 | **0.7789x** | 0.0337 | 0.0262 | `ef=40@0.971300`..`ef=50@0.980190` ($\alpha=0.9786$) | `ef=50@0.974950`..`ef=70@0.987370` ($\alpha=0.4066$) |
| `0.99` | `matched` | 21,336.87 | 29,542.47 | **0.7222x** | 0.0469 | 0.0339 | `ef=70@0.988850`..`ef=100@0.993670` ($\alpha=0.2386$) | `ef=70@0.987370`..`ef=100@0.994020` ($\alpha=0.3955$) |
| `0.995` | `matched` | 14,727.37 | 21,572.58 | **0.6827x** | 0.0679 | 0.0464 | `ef=100@0.993670`..`ef=150@0.995770` ($\alpha=0.6333$) | `ef=100@0.994020`..`ef=150@0.997800` ($\alpha=0.2593$) |

#### `100k` C2 T=16 vs FAISS C2 T=16 (`0xffff`)

- Observed recall bounds: **Rust C2 T=16** `[0.821030, 0.997020]`, **FAISS C2 T=16** `[0.778610, 0.999900]`.

| Target $R^*$ | Status | Rust C2 T=16 M1 QPS | FAISS C2 T=16 M1 QPS | M1 QPS Ratio (Rust C2 T=16/FAISS C2 T=16) | Rust C2 T=16 M1 Mean (ms) | FAISS C2 T=16 M1 Mean (ms) | Rust C2 T=16 Bracket (`ef@R`) | FAISS C2 T=16 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 86,168.96 | 131,611.89 | **0.6547x** | 0.0116 | 0.0076 | `ef=10@0.821030`..`ef=20@0.916780` ($\alpha=0.8248$) | `ef=20@0.893560`..`ef=30@0.938930` ($\alpha=0.1419$) |
| `0.95` | `matched` | 76,651.73 | 78,732.69 | **0.9736x** | 0.0131 | 0.0127 | `ef=20@0.916780`..`ef=30@0.953980` ($\alpha=0.8930$) | `ef=30@0.938930`..`ef=40@0.962150` ($\alpha=0.4767$) |
| `0.98` | `matched` | 46,706.30 | 55,665.12 | **0.8391x** | 0.0214 | 0.0179 | `ef=40@0.971300`..`ef=50@0.980190` ($\alpha=0.9786$) | `ef=50@0.974950`..`ef=70@0.987370` ($\alpha=0.4066$) |
| `0.99` | `matched` | 31,781.80 | 38,827.10 | **0.8185x** | 0.0315 | 0.0257 | `ef=70@0.988850`..`ef=100@0.993670` ($\alpha=0.2386$) | `ef=70@0.987370`..`ef=100@0.994020` ($\alpha=0.3955$) |
| `0.995` | `matched` | 23,267.20 | 28,597.87 | **0.8136x** | 0.0430 | 0.0350 | `ef=100@0.993670`..`ef=150@0.995770` ($\alpha=0.6333$) | `ef=100@0.994020`..`ef=150@0.997800` ($\alpha=0.2593$) |

#### `1M` C2 T=6 vs FAISS C2 T=6 (`0x0555`)

- Observed recall bounds: **Rust C2 T=6** `[0.733150, 0.998040]`, **FAISS C2 T=6** `[0.695540, 0.998720]`.

| Target $R^*$ | Status | Rust C2 T=6 M1 QPS | FAISS C2 T=6 M1 QPS | M1 QPS Ratio (Rust C2 T=6/FAISS C2 T=6) | Rust C2 T=6 M1 Mean (ms) | FAISS C2 T=6 M1 Mean (ms) | Rust C2 T=6 Bracket (`ef@R`) | FAISS C2 T=6 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 23,210.04 | 26,126.05 | **0.8884x** | 0.0431 | 0.0382 | `ef=20@0.849490`..`ef=30@0.900130` ($\alpha=0.9974$) | `ef=30@0.884850`..`ef=40@0.918570` ($\alpha=0.4493$) |
| `0.95` | `matched` | 15,064.44 | 17,254.83 | **0.8731x** | 0.0664 | 0.0580 | `ef=50@0.947030`..`ef=70@0.967590` ($\alpha=0.1445$) | `ef=50@0.938650`..`ef=70@0.962440` ($\alpha=0.4771$) |
| `0.98` | `matched` | 8,764.92 | 9,331.47 | **0.9393x** | 0.1141 | 0.1072 | `ef=70@0.967590`..`ef=100@0.980610` ($\alpha=0.9531$) | `ef=100@0.978210`..`ef=150@0.988910` ($\alpha=0.1673$) |
| `0.99` | `matched` | 6,202.30 | 7,091.58 | **0.8746x** | 0.1613 | 0.1410 | `ef=100@0.980610`..`ef=150@0.990040` ($\alpha=0.9958$) | `ef=150@0.988910`..`ef=200@0.993460` ($\alpha=0.2396$) |
| `0.995` | `matched` | 4,286.21 | 4,677.96 | **0.9163x** | 0.2333 | 0.2138 | `ef=200@0.993760`..`ef=300@0.996540` ($\alpha=0.4460$) | `ef=200@0.993460`..`ef=300@0.996730` ($\alpha=0.4709$) |

#### `1M` C2 T=16 vs FAISS C2 T=16 (`0xffff`)

- Observed recall bounds: **Rust C2 T=16** `[0.733150, 0.998040]`, **FAISS C2 T=16** `[0.695540, 0.998720]`.

| Target $R^*$ | Status | Rust C2 T=16 M1 QPS | FAISS C2 T=16 M1 QPS | M1 QPS Ratio (Rust C2 T=16/FAISS C2 T=16) | Rust C2 T=16 M1 Mean (ms) | FAISS C2 T=16 M1 Mean (ms) | Rust C2 T=16 Bracket (`ef@R`) | FAISS C2 T=16 Bracket (`ef@R`) |
|---:|---|---:|---:|---:|---:|---:|---|---|
| `0.9` | `matched` | 27,707.89 | 33,180.22 | **0.8351x** | 0.0361 | 0.0301 | `ef=20@0.849490`..`ef=30@0.900130` ($\alpha=0.9974$) | `ef=30@0.884850`..`ef=40@0.918570` ($\alpha=0.4493$) |
| `0.95` | `matched` | 18,608.51 | 21,054.00 | **0.8838x** | 0.0538 | 0.0475 | `ef=50@0.947030`..`ef=70@0.967590` ($\alpha=0.1445$) | `ef=50@0.938650`..`ef=70@0.962440` ($\alpha=0.4771$) |
| `0.98` | `matched` | 11,026.70 | 12,450.71 | **0.8856x** | 0.0906 | 0.0803 | `ef=70@0.967590`..`ef=100@0.980610` ($\alpha=0.9531$) | `ef=100@0.978210`..`ef=150@0.988910` ($\alpha=0.1673$) |
| `0.99` | `matched` | 8,086.64 | 8,931.53 | **0.9054x** | 0.1236 | 0.1120 | `ef=100@0.980610`..`ef=150@0.990040` ($\alpha=0.9958$) | `ef=150@0.988910`..`ef=200@0.993460` ($\alpha=0.2396$) |
| `0.995` | `matched` | 5,386.49 | 6,164.29 | **0.8738x** | 0.1857 | 0.1623 | `ef=200@0.993760`..`ef=300@0.996540` ($\alpha=0.4460$) | `ef=200@0.993460`..`ef=300@0.996730` ($\alpha=0.4709$) |

### 6.2 Full `efSearch` Multi-Thread Scaling Table (`100k` & `1M` `C2`)

#### `100k` `C2` (`M=16, efC=100`) Multi-Thread Sweep

| `efSearch` | Rust `R@10` | Rust `T=1` QPS | Rust `T=6` QPS (Speedup) | Rust `T=16` QPS (Speedup) | FAISS `R@10` | FAISS `T=1` QPS | FAISS `T=6` QPS (Speedup) | FAISS `T=16` QPS (Speedup) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.821030 | 17,893.65 | 113,581.38 (`6.35x`) | 135,310.49 (`7.56x`) | 0.778610 | 28,943.81 | 168,754.16 (`5.83x`) | 210,431.07 (`7.27x`) |
| 20 | 0.916780 | 11,618.62 | 72,137.48 (`6.21x`) | 78,290.27 (`6.74x`) | 0.893560 | 18,337.11 | 92,518.41 (`5.05x`) | 140,362.30 (`7.65x`) |
| 30 | 0.953980 | 8,597.56 | 42,499.89 (`4.94x`) | 76,457.74 (`8.89x`) | 0.938930 | 13,221.23 | 59,407.36 (`4.49x`) | 89,187.53 (`6.75x`) |
| 40 | 0.971300 | 6,822.88 | 41,043.81 (`6.02x`) | 49,535.85 (`7.26x`) | 0.962150 | 10,337.13 | 54,480.83 (`5.27x`) | 68,663.07 (`6.64x`) |
| 50 | 0.980190 | 5,873.51 | 29,522.97 (`5.03x`) | 46,646.34 (`7.94x`) | 0.974950 | 8,390.09 | 41,276.57 (`4.92x`) | 65,176.60 (`7.77x`) |
| 70 | 0.988850 | 4,572.13 | 22,246.08 (`4.87x`) | 33,056.83 (`7.23x`) | 0.987370 | 6,115.46 | 34,048.59 (`5.57x`) | 44,218.24 (`7.23x`) |
| 100 | 0.993670 | 3,442.68 | 18,676.51 (`5.42x`) | 28,032.53 (`8.14x`) | 0.994020 | 4,405.27 | 23,779.91 (`5.40x`) | 31,829.14 (`7.23x`) |
| 150 | 0.995770 | 2,476.47 | 12,834.97 (`5.18x`) | 20,887.99 (`8.43x`) | 0.997800 | 2,949.25 | 16,331.33 (`5.54x`) | 21,062.09 (`7.14x`) |
| 200 | 0.996450 | 1,972.57 | 9,873.25 (`5.01x`) | 15,436.53 (`7.83x`) | 0.998850 | 2,152.24 | 12,060.21 (`5.60x`) | 17,546.88 (`8.15x`) |
| 300 | 0.996900 | 1,422.46 | 7,792.55 (`5.48x`) | 10,337.98 (`7.27x`) | 0.999550 | 1,392.71 | 7,861.36 (`5.64x`) | 11,237.41 (`8.07x`) |
| 400 | 0.997000 | 1,140.12 | 6,079.87 (`5.33x`) | 9,141.62 (`8.02x`) | 0.999780 | 993.51 | 5,673.39 (`5.71x`) | 8,122.26 (`8.18x`) |
| 600 | 0.997020 | 816.12 | 4,420.94 (`5.42x`) | 6,704.73 (`8.22x`) | 0.999900 | 588.64 | 3,344.17 (`5.68x`) | 5,004.41 (`8.50x`) |

#### `1M` `C2` (`M=16, efC=100`) Multi-Thread Sweep

| `efSearch` | Rust `R@10` | Rust `T=1` QPS | Rust `T=6` QPS (Speedup) | Rust `T=16` QPS (Speedup) | FAISS `R@10` | FAISS `T=1` QPS | FAISS `T=6` QPS (Speedup) | FAISS `T=16` QPS (Speedup) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.733150 | 9,121.80 | 52,530.33 (`5.76x`) | 57,800.33 (`6.34x`) | 0.695540 | 16,919.75 | 57,773.28 (`3.41x`) | 73,911.14 (`4.37x`) |
| 20 | 0.849490 | 5,936.97 | 30,963.25 (`5.22x`) | 36,745.87 (`6.19x`) | 0.825820 | 10,911.78 | 40,693.65 (`3.73x`) | 47,305.84 (`4.34x`) |
| 30 | 0.900130 | 4,541.78 | 23,192.83 (`5.11x`) | 27,687.77 (`6.10x`) | 0.884850 | 7,711.07 | 30,258.15 (`3.92x`) | 35,668.25 (`4.63x`) |
| 40 | 0.929140 | 3,618.95 | 19,261.88 (`5.32x`) | 23,370.64 (`6.46x`) | 0.918570 | 6,120.56 | 21,822.76 (`3.57x`) | 30,366.02 (`4.96x`) |
| 50 | 0.947030 | 2,875.88 | 15,605.16 (`5.43x`) | 19,246.87 (`6.69x`) | 0.938650 | 5,145.46 | 20,042.63 (`3.90x`) | 23,721.05 (`4.61x`) |
| 70 | 0.967590 | 2,140.86 | 12,224.97 (`5.71x`) | 15,238.93 (`7.12x`) | 0.962440 | 3,619.62 | 14,642.67 (`4.05x`) | 18,474.00 (`5.10x`) |
| 100 | 0.980610 | 1,491.23 | 8,622.74 (`5.78x`) | 10,852.73 (`7.28x`) | 0.978210 | 2,567.67 | 9,700.74 (`3.78x`) | 13,141.95 (`5.12x`) |
| 150 | 0.990040 | 1,129.48 | 6,193.60 (`5.48x`) | 8,076.51 (`7.15x`) | 0.988910 | 1,848.07 | 7,692.22 (`4.16x`) | 9,514.61 (`5.15x`) |
| 200 | 0.993760 | 725.04 | 4,916.47 (`6.78x`) | 6,095.59 (`8.41x`) | 0.993460 | 1,332.06 | 5,478.51 (`4.11x`) | 7,307.08 (`5.49x`) |
| 300 | 0.996540 | 723.52 | 3,614.76 (`5.00x`) | 4,619.58 (`6.38x`) | 0.996730 | 833.75 | 3,917.30 (`4.70x`) | 5,092.25 (`6.11x`) |
| 400 | 0.997500 | 568.43 | 2,743.85 (`4.83x`) | 3,578.35 (`6.30x`) | 0.997910 | 601.76 | 3,079.94 (`5.12x`) | 3,922.27 (`6.52x`) |
| 600 | 0.998040 | 389.74 | 2,004.43 (`5.14x`) | 2,590.72 (`6.65x`) | 0.998720 | 428.31 | 2,023.76 (`4.72x`) | 2,580.75 (`6.03x`) |

- **Scaling Analysis (`100k` vs `1M`):** At `100k`, Rust `search_batch_parallel` (`std::thread::scope`) scales by `4.87x–6.35x` (median `5.37x`) at `T=6` (`0x0555`) and `6.74x–8.89x` (median `7.88x`) at `T=16` (`0xffff`), closely matching FAISS OpenMP (`4.49x–5.83x` at `T=6`, `6.64x–8.50x` at `T=16`) and achieving `0.6827x–0.8991x` (`T=6`) and `0.6547x–0.9736x` (`T=16`) at matched recall. At `1M` — where both engines ran `T=6` and `T=16` **solo** — Rust narrows its matched-recall deficit vs FAISS from `0.6241x–0.6781x` (`T=1` concurrent) to **`0.8731x–0.9393x` (`T=6`) and `0.8351x–0.9054x` (`T=16`)**, reaching `18,608.51` (`R*=0.95`) and `11,026.70 QPS` (`R*=0.98`) at `T=16` (and edging out FAISS in raw `ef=600` QPS: `2,590.72` vs `2,580.75`). Note that Rust's `search_knn_with_tracker` ([`src/hnsw/mod.rs:1350-1379`](../../src/hnsw/mod.rs#L1350-L1379)) allocates `BinaryHeap`s (`c0`, `w`) and `Vec` (`results`) per query whereas FAISS reuses per-OpenMP-thread scratch buffers; additionally, `1M` `T=6`/`T=16` speedups over `T=1` appear slightly steeper in Rust (`4.83x–6.78x` / `6.10x–8.41x`) than FAISS (`3.41x–5.12x` / `4.34x–6.52x`) partly because `1M C2 T=1` ran concurrently while `T=6`/`T=16` ran solo.
- **FAISS OpenMP Build Determinism Observation:** Across `T in {1, 6, 16}`, FAISS `C2` reports produce identical `max_layer`, `entry_point`, `graph_only_memory_bytes`, and all `12` `Recall@10` values (`0.778610..0.999900` at `100k`, `0.695540..0.998720` at `1M`).

---

## 7. Visual Evidence & Chart Index (`T5.6`, [`scripts/plot_phase7.py`](../../scripts/plot_phase7.py))

All `24` final chart files (`4` charts in both `.png` and `.svg` across root `results/phase7/`, [`./charts_100k/`](./charts_100k/), and [`./charts_1m/`](./charts_1m/)) were generated cleanly from the committed `results/phase7/` JSON reports (`git_commit = 9f79165`, `git_dirty = false`) and [`results/phase7/matched_recall_summary.json`](./matched_recall_summary.json), and visually verified for label/legend legibility:

### 7.1 Combined & Headline Root Charts (`results/phase7/`)

- [`./chart1_recall_vs_p99.png`](./chart1_recall_vs_p99.png) ([SVG](./chart1_recall_vs_p99.svg)) — Combined `N = 100k` and `N = 1M` side-by-side Mode 2 `Recall@10` vs `p99` tail latency (log y-axis), shared 7-curve legend (`Rust C1/C2/C3/C2-nkp` + `FAISS C1/C2/C3`), and provenance footer noting `FAISS Mode 2 search_c includes median FFI delta 7.23 us/q`.
- [`./chart2_recall_vs_qps.png`](./chart2_recall_vs_qps.png) ([SVG](./chart2_recall_vs_qps.svg)) — Combined `N = 100k` and `N = 1M` Mode 1 pure-batch `Recall@10` vs throughput (`QPS`, log y-axis; zero per-query FFI or timer overhead).
- [`./chart3_matched_recall_ratios.png`](./chart3_matched_recall_ratios.png) ([SVG](./chart3_matched_recall_ratios.svg)) — Combined `100k` and `1M` matched-recall speedup bars (`Rust / FAISS` Mode 1 QPS ratio and `FAISS p99 / Rust p99` Mode 2 latency ratio) across $R^* \in \{0.90, 0.95, 0.98, 0.99, 0.995\}$ without extrapolation (`100k C3` omitted at `R*=0.9`; `1M C1` omitted at `R*=0.99, 0.995`). Bar labels use an offset white-halo `bbox` (`alpha=0.88`, `zorder=6`) so vertical labels crossing the dashed `y = 1.0` parity line mask the line cleanly and remain fully legible.
- [`./chart4_multithread_and_build.png`](./chart4_multithread_and_build.png) ([SVG](./chart4_multithread_and_build.svg)) — Mirrors the clean single-scale `1M` Chart 4 (`2x2` grid: `1M C2` multi-thread Mode 1 batch QPS for `T in {1, 6, 16}`, `11`-bar `Index Build Time` with explicit `Asymmetry note`, `7`-bar `Graph-Only Memory / Vector`, and `7`-bar `Process Peak RSS`), keeping build-time and memory bar comparisons clean per scale rather than squeezing `22` bars onto one axis.

### 7.2 Per-Scale `N = 100,000` Charts (`./charts_100k/`)

- [`./charts_100k/chart1_recall_vs_p99.png`](./charts_100k/chart1_recall_vs_p99.png) ([SVG](./charts_100k/chart1_recall_vs_p99.svg)) — `100k` Mode 2 `Recall@10` vs `p99` tail latency (footer median FFI delta `6.60 us/q`).
- [`./charts_100k/chart2_recall_vs_qps.png`](./charts_100k/chart2_recall_vs_qps.png) ([SVG](./charts_100k/chart2_recall_vs_qps.svg)) — `100k` Mode 1 pure-batch `Recall@10` vs QPS.
- [`./charts_100k/chart3_matched_recall_ratios.png`](./charts_100k/chart3_matched_recall_ratios.png) ([SVG](./charts_100k/chart3_matched_recall_ratios.svg)) — `100k` matched-recall Mode 1 QPS and Mode 2 `p99` ratios with white-halo bar labels (`0.95x` and `0.90x`–`0.92x` cleanly separated from `y = 1.0`).
- [`./charts_100k/chart4_multithread_and_build.png`](./charts_100k/chart4_multithread_and_build.png) ([SVG](./charts_100k/chart4_multithread_and_build.svg)) — `100k C2` multi-thread batch QPS (`T in {1, 6, 16}`), `11`-bar `Index Build Time`, `7`-bar `Graph-Only Memory / Vector`, and `7`-bar `Process Peak RSS`.

### 7.3 Per-Scale `N = 1,000,000` Charts (`./charts_1m/`)

- [`./charts_1m/chart1_recall_vs_p99.png`](./charts_1m/chart1_recall_vs_p99.png) ([SVG](./charts_1m/chart1_recall_vs_p99.svg)) — `1M` Mode 2 `Recall@10` vs `p99` tail latency (footer median FFI delta `8.09 us/q`; transient concurrent-run `p99` spikes on `1M` `C1`/`C2`/`C3` visible alongside smooth solo `C2-nkp`).
- [`./charts_1m/chart2_recall_vs_qps.png`](./charts_1m/chart2_recall_vs_qps.png) ([SVG](./charts_1m/chart2_recall_vs_qps.svg)) — `1M` Mode 1 pure-batch `Recall@10` vs QPS (`C1` and `C2-nkp` crossovers clearly visible at high recall).
- [`./charts_1m/chart3_matched_recall_ratios.png`](./charts_1m/chart3_matched_recall_ratios.png) ([SVG](./charts_1m/chart3_matched_recall_ratios.svg)) — `1M` matched-recall Mode 1 QPS and Mode 2 `p99` ratios with white-halo bar labels (`0.92x`, `0.94x`, `1.00x` cleanly masking `y = 1.0`).
- [`./charts_1m/chart4_multithread_and_build.png`](./charts_1m/chart4_multithread_and_build.png) ([SVG](./charts_1m/chart4_multithread_and_build.svg)) — `1M C2` multi-thread batch QPS (`T in {1, 6, 16}`), `11`-bar `Index Build Time`, `7`-bar `Graph-Only Memory / Vector`, and `7`-bar `Process Peak RSS`.

> **Chart 3 Key-Pairing Note (`C2-nkp`):** `Rust C2-nkp` (`keep_pruned_shrink = false`) appears in **Charts 1, 2, and 4** (full Pareto curves plus build-time, graph-memory, and Peak RSS bars) and in the quantitative ablation tables in **§3.1, §3.2, and §4**. It is omitted from **Chart 3** because [`plot_chart3_matched_recall_ratios`](../../scripts/plot_phase7.py#L704) pairs only identical `(scale, config_id)` keys (`C1`, `C2`, `C3`).

---

## 8. Full Single-Thread `efSearch` Sweep Appendices (`14` Reports: `100k` & `1M`)

### 8.1 Scale `N = 100,000` (`100k`)

### `rust_n100000_C1.json` (`rust` `C1`, `N=100,000`, affinity `0x0004`, `generated_at=2026-11-17T10:19:00Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.690090 | 30,128.95 | 0.0332 | 0.0329 | 0.0314 | 0.0477 | 0.0582 | -0.2542 | `null` |
| 20 | 0.822450 | 19,068.23 | 0.0524 | 0.0518 | 0.0504 | 0.0704 | 0.0867 | -0.6008 | `null` |
| 30 | 0.883460 | 13,782.89 | 0.0726 | 0.0702 | 0.0688 | 0.0925 | 0.1152 | -2.3826 | `null` |
| 40 | 0.917660 | 11,445.78 | 0.0874 | 0.0882 | 0.0875 | 0.1127 | 0.1357 | 0.8570 | `null` |
| 50 | 0.938630 | 9,574.48 | 0.1044 | 0.1041 | 0.1037 | 0.1317 | 0.1577 | -0.3282 | `null` |
| 70 | 0.963100 | 7,325.16 | 0.1365 | 0.1372 | 0.1367 | 0.1699 | 0.2092 | 0.6819 | `null` |
| 100 | 0.979110 | 5,351.83 | 0.1869 | 0.1863 | 0.1881 | 0.2315 | 0.2908 | -0.5262 | `null` |
| 150 | 0.990030 | 3,795.92 | 0.2634 | 0.2613 | 0.2637 | 0.3225 | 0.3831 | -2.1706 | `null` |
| 200 | 0.993800 | 2,911.48 | 0.3435 | 0.3388 | 0.3412 | 0.4211 | 0.5031 | -4.6349 | `null` |
| 300 | 0.996950 | 2,094.57 | 0.4774 | 0.4791 | 0.4835 | 0.5949 | 0.7038 | 1.6528 | `null` |
| 400 | 0.997770 | 1,625.21 | 0.6153 | 0.6167 | 0.6235 | 0.7673 | 0.8964 | 1.3621 | `null` |
| 600 | 0.998540 | 1,145.68 | 0.8728 | 0.8711 | 0.8789 | 1.0870 | 1.2560 | -1.7753 | `null` |

### `faiss_n100000_C1.json` (`faiss` `C1`, `N=100,000`, affinity `0x0400`, `generated_at=2026-11-17T12:36:20Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `ffi_delta_us` | `ndis` / `nhops` (`ns/ndis`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.687260 | 35,834.84 | 0.0279 | 0.0306 | 0.0292 | 0.0438 | 0.0530 | 3.0573 | 200.0 / 23.4 (`139.5`) |
| 20 | 0.825440 | 22,014.19 | 0.0454 | 0.0480 | 0.0461 | 0.0655 | 0.0793 | 2.9714 | 291.3 / 33.1 (`155.9`) |
| 30 | 0.884970 | 16,182.47 | 0.0618 | 0.0652 | 0.0635 | 0.0864 | 0.1029 | 3.9119 | 374.8 / 42.5 (`164.9`) |
| 40 | 0.918210 | 12,802.06 | 0.0781 | 0.0832 | 0.0813 | 0.1079 | 0.1262 | 5.5903 | 454.3 / 52.0 (`171.9`) |
| 50 | 0.939360 | 10,606.85 | 0.0943 | 0.0982 | 0.0963 | 0.1232 | 0.1437 | 4.4574 | 532.0 / 61.7 (`177.2`) |
| 70 | 0.961620 | 8,116.69 | 0.1232 | 0.1348 | 0.1345 | 0.1645 | 0.1918 | 12.3153 | 681.7 / 81.2 (`180.7`) |
| 100 | 0.978310 | 5,348.27 | 0.1870 | 0.1903 | 0.1908 | 0.2292 | 0.2635 | 4.2052 | 895.0 / 110.9 (`208.9`) |
| 150 | 0.989110 | 3,512.96 | 0.2847 | 0.2918 | 0.2931 | 0.3449 | 0.4025 | 8.5254 | 1224.6 / 160.6 (`232.5`) |
| 200 | 0.993350 | 2,556.62 | 0.3911 | 0.4014 | 0.4027 | 0.4715 | 0.5468 | 11.8066 | 1530.1 / 210.4 (`255.6`) |
| 300 | 0.996840 | 1,587.95 | 0.6297 | 0.6414 | 0.6455 | 0.7377 | 0.8332 | 13.5177 | 2091.7 / 310.3 (`301.0`) |
| 400 | 0.998160 | 1,110.19 | 0.9007 | 0.9126 | 0.9175 | 1.0354 | 1.1580 | 13.9449 | 2605.8 / 410.2 (`345.7`) |
| 600 | 0.999020 | 646.05 | 1.5479 | 1.5602 | 1.5669 | 1.7307 | 1.9079 | 15.2084 | 3539.1 / 610.1 (`437.4`) |

### `rust_n100000_C2.json` (`rust` `C2`, `N=100,000`, affinity `0x0004`, `generated_at=2026-11-17T12:09:41Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.821030 | 17,893.65 | 0.0559 | 0.0542 | 0.0527 | 0.0776 | 0.0927 | -1.7263 | `null` |
| 20 | 0.916780 | 11,618.62 | 0.0861 | 0.0858 | 0.0853 | 0.1148 | 0.1374 | -0.2947 | `null` |
| 30 | 0.953980 | 8,597.56 | 0.1163 | 0.1140 | 0.1149 | 0.1480 | 0.1710 | -2.2986 | `null` |
| 40 | 0.971300 | 6,822.88 | 0.1466 | 0.1433 | 0.1453 | 0.1853 | 0.2115 | -3.3027 | `null` |
| 50 | 0.980190 | 5,873.51 | 0.1703 | 0.1681 | 0.1711 | 0.2157 | 0.2513 | -2.1671 | `null` |
| 70 | 0.988850 | 4,572.13 | 0.2187 | 0.2184 | 0.2237 | 0.2797 | 0.3159 | -0.2694 | `null` |
| 100 | 0.993670 | 3,442.68 | 0.2905 | 0.2980 | 0.3034 | 0.3859 | 0.4503 | 7.5142 | `null` |
| 150 | 0.995770 | 2,476.47 | 0.4038 | 0.4037 | 0.4133 | 0.5201 | 0.5760 | -0.0884 | `null` |
| 200 | 0.996450 | 1,972.57 | 0.5070 | 0.5068 | 0.5185 | 0.6606 | 0.7412 | -0.1702 | `null` |
| 300 | 0.996900 | 1,422.46 | 0.7030 | 0.6969 | 0.7133 | 0.9088 | 0.9995 | -6.1326 | `null` |
| 400 | 0.997000 | 1,140.12 | 0.8771 | 0.8737 | 0.8921 | 1.1397 | 1.2437 | -3.3717 | `null` |
| 600 | 0.997020 | 816.12 | 1.2253 | 1.2208 | 1.2438 | 1.5813 | 1.7279 | -4.4745 | `null` |

### `rust_n100000_C2-nkp.json` (`rust` `C2-nkp`, `N=100,000`, affinity `0x0004`, `generated_at=2026-11-17T12:37:44Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.799120 | 20,105.25 | 0.0497 | 0.0489 | 0.0475 | 0.0692 | 0.0827 | -0.7941 | `null` |
| 20 | 0.907240 | 12,857.74 | 0.0778 | 0.0785 | 0.0779 | 0.1057 | 0.1254 | 0.7406 | `null` |
| 30 | 0.949190 | 9,514.21 | 0.1051 | 0.1054 | 0.1058 | 0.1370 | 0.1608 | 0.3023 | `null` |
| 40 | 0.969370 | 7,523.78 | 0.1329 | 0.1332 | 0.1338 | 0.1724 | 0.1995 | 0.2792 | `null` |
| 50 | 0.980150 | 6,293.74 | 0.1589 | 0.1564 | 0.1585 | 0.2000 | 0.2286 | -2.4559 | `null` |
| 70 | 0.990490 | 4,887.20 | 0.2046 | 0.2055 | 0.2087 | 0.2626 | 0.3000 | 0.8958 | `null` |
| 100 | 0.995800 | 3,723.69 | 0.2686 | 0.2684 | 0.2734 | 0.3424 | 0.3843 | -0.1022 | `null` |
| 150 | 0.998340 | 2,673.06 | 0.3741 | 0.3729 | 0.3802 | 0.4781 | 0.5342 | -1.2473 | `null` |
| 200 | 0.999250 | 2,111.79 | 0.4735 | 0.4741 | 0.4819 | 0.6081 | 0.6900 | 0.5746 | `null` |
| 300 | 0.999750 | 1,509.87 | 0.6623 | 0.6597 | 0.6704 | 0.8508 | 0.9502 | -2.6445 | `null` |
| 400 | 0.999880 | 1,195.97 | 0.8361 | 0.8329 | 0.8476 | 1.0739 | 1.1886 | -3.2466 | `null` |
| 600 | 0.999890 | 865.71 | 1.1551 | 1.1466 | 1.1633 | 1.4709 | 1.6370 | -8.5009 | `null` |

### `faiss_n100000_C2.json` (`faiss` `C2`, `N=100,000`, affinity `0x0400`, `generated_at=2026-11-17T12:10:23Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `ffi_delta_us` | `ndis` / `nhops` (`ns/ndis`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.778610 | 28,943.81 | 0.0345 | 0.0375 | 0.0362 | 0.0532 | 0.0653 | 3.3199 | 293.3 / 20.6 (`117.6`) |
| 20 | 0.893560 | 18,337.11 | 0.0545 | 0.0584 | 0.0573 | 0.0783 | 0.0951 | 4.3616 | 424.0 / 29.9 (`128.6`) |
| 30 | 0.938930 | 13,221.23 | 0.0756 | 0.0788 | 0.0782 | 0.1017 | 0.1223 | 3.6922 | 545.9 / 39.4 (`138.5`) |
| 40 | 0.962150 | 10,337.13 | 0.0967 | 0.1007 | 0.1003 | 0.1295 | 0.1547 | 4.6082 | 662.9 / 49.1 (`145.9`) |
| 50 | 0.974950 | 8,390.09 | 0.1192 | 0.1239 | 0.1231 | 0.1614 | 0.1942 | 5.4659 | 776.2 / 58.9 (`153.6`) |
| 70 | 0.987370 | 6,115.46 | 0.1635 | 0.1683 | 0.1678 | 0.2117 | 0.2503 | 5.7018 | 990.9 / 78.6 (`165.0`) |
| 100 | 0.994020 | 4,405.27 | 0.2270 | 0.2324 | 0.2349 | 0.2840 | 0.3317 | 6.5997 | 1290.4 / 108.3 (`175.9`) |
| 150 | 0.997800 | 2,949.25 | 0.3391 | 0.3466 | 0.3494 | 0.4171 | 0.4891 | 9.0074 | 1745.2 / 158.2 (`194.3`) |
| 200 | 0.998850 | 2,152.24 | 0.4646 | 0.4661 | 0.4713 | 0.5586 | 0.6295 | 3.0643 | 2159.5 / 208.1 (`215.1`) |
| 300 | 0.999550 | 1,392.71 | 0.7180 | 0.7250 | 0.7336 | 0.8522 | 0.9627 | 8.8092 | 2907.9 / 308.0 (`246.9`) |
| 400 | 0.999780 | 993.51 | 1.0065 | 1.0153 | 1.0266 | 1.1751 | 1.3037 | 10.8256 | 3579.5 / 407.9 (`281.2`) |
| 600 | 0.999900 | 588.64 | 1.6988 | 1.7130 | 1.7255 | 1.9517 | 2.1156 | 17.1302 | 4773.2 / 607.8 (`355.9`) |

### `rust_n100000_C3.json` (`rust` `C3`, `N=100,000`, affinity `0x0004`, `generated_at=2026-11-17T12:27:23Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.909740 | 11,499.38 | 0.0870 | 0.0857 | 0.0846 | 0.1198 | 0.1429 | -1.2575 | `null` |
| 20 | 0.968280 | 7,372.06 | 0.1356 | 0.1384 | 0.1380 | 0.1885 | 0.2238 | 2.7418 | `null` |
| 30 | 0.985560 | 5,543.34 | 0.1804 | 0.1819 | 0.1834 | 0.2435 | 0.2821 | 1.5185 | `null` |
| 40 | 0.992600 | 4,476.40 | 0.2234 | 0.2231 | 0.2275 | 0.2966 | 0.3427 | -0.2957 | `null` |
| 50 | 0.995480 | 3,873.68 | 0.2582 | 0.2578 | 0.2625 | 0.3446 | 0.3982 | -0.3794 | `null` |
| 70 | 0.998230 | 3,033.30 | 0.3297 | 0.3290 | 0.3359 | 0.4434 | 0.5088 | -0.6292 | `null` |
| 100 | 0.999390 | 2,320.40 | 0.4310 | 0.4295 | 0.4383 | 0.5815 | 0.6565 | -1.5026 | `null` |
| 150 | 0.999800 | 1,717.65 | 0.5822 | 0.5789 | 0.5888 | 0.7914 | 0.8962 | -3.3008 | `null` |
| 200 | 0.999910 | 1,386.07 | 0.7215 | 0.7424 | 0.7499 | 1.0271 | 1.1860 | 20.9558 | `null` |
| 300 | 0.999960 | 997.54 | 1.0025 | 0.9714 | 0.9870 | 1.3303 | 1.4839 | -31.0516 | `null` |
| 400 | 0.999960 | 839.56 | 1.1911 | 1.1909 | 1.2091 | 1.6322 | 1.8381 | -0.1956 | `null` |
| 600 | 0.999990 | 622.89 | 1.6054 | 1.6410 | 1.6667 | 2.2203 | 2.4826 | 35.5299 | `null` |

### `faiss_n100000_C3.json` (`faiss` `C3`, `N=100,000`, affinity `0x0400`, `generated_at=2026-11-17T12:25:36Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `ffi_delta_us` | `ndis` / `nhops` (`ns/ndis`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.836150 | 22,357.39 | 0.0447 | 0.0483 | 0.0467 | 0.0688 | 0.0846 | 4.0102 | 393.1 / 19.6 (`113.7`) |
| 20 | 0.934290 | 13,908.34 | 0.0719 | 0.0744 | 0.0736 | 0.1004 | 0.1224 | 3.0205 | 575.0 / 28.9 (`125.0`) |
| 30 | 0.967480 | 10,224.31 | 0.0978 | 0.1020 | 0.1017 | 0.1363 | 0.1675 | 4.8609 | 745.5 / 38.5 (`131.2`) |
| 40 | 0.981410 | 8,049.48 | 0.1242 | 0.1295 | 0.1300 | 0.1703 | 0.2051 | 6.0103 | 908.5 / 48.2 (`136.7`) |
| 50 | 0.988450 | 6,646.89 | 0.1504 | 0.1563 | 0.1570 | 0.2024 | 0.2456 | 6.7770 | 1064.1 / 58.0 (`141.3`) |
| 70 | 0.994840 | 4,892.21 | 0.2044 | 0.2134 | 0.2132 | 0.2785 | 0.3359 | 10.1553 | 1355.9 / 77.8 (`150.7`) |
| 100 | 0.997850 | 3,511.92 | 0.2847 | 0.2916 | 0.2951 | 0.3731 | 0.4324 | 8.2885 | 1756.0 / 107.7 (`162.1`) |
| 150 | 0.999240 | 2,386.90 | 0.4190 | 0.4274 | 0.4319 | 0.5420 | 0.6284 | 9.9933 | 2353.7 / 157.5 (`178.0`) |
| 200 | 0.999570 | 1,803.08 | 0.5546 | 0.5662 | 0.5720 | 0.7091 | 0.8205 | 13.3239 | 2888.9 / 207.4 (`192.0`) |
| 300 | 0.999890 | 1,178.74 | 0.8484 | 0.8618 | 0.8700 | 1.0593 | 1.2072 | 15.5040 | 3836.3 / 307.3 (`221.2`) |
| 400 | 0.999950 | 827.29 | 1.2088 | 1.1977 | 1.2047 | 1.4527 | 1.6383 | -8.4680 | 4672.3 / 407.3 (`258.7`) |
| 600 | 0.999970 | 524.38 | 1.9070 | 1.9274 | 1.9425 | 2.2540 | 2.4744 | 24.3771 | 6126.1 / 607.2 (`311.3`) |

### 8.2 Scale `N = 1,000,000` (`1M`)

### `rust_n1000000_C1.json` (`rust` `C1`, `N=1,000,000`, affinity `0x0004`, `generated_at=2026-11-17T13:36:10Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.572220 | 15,192.42 | 0.0658 | 0.0643 | 0.0613 | 0.0914 | 0.1115 | -1.5657 | `null` |
| 20 | 0.713990 | 10,057.51 | 0.0994 | 0.0983 | 0.0952 | 0.1353 | 0.1643 | -1.0958 | `null` |
| 30 | 0.784490 | 7,596.29 | 0.1316 | 0.1301 | 0.1272 | 0.1737 | 0.2087 | -1.5489 | `null` |
| 40 | 0.829950 | 6,180.03 | 0.1618 | 0.1614 | 0.1586 | 0.2104 | 0.2549 | -0.4551 | `null` |
| 50 | 0.860720 | 5,216.22 | 0.1917 | 0.1907 | 0.1883 | 0.2447 | 0.2955 | -1.0585 | `null` |
| 70 | 0.900410 | 4,033.71 | 0.2479 | 0.2470 | 0.2469 | 0.3070 | 0.3675 | -0.9131 | `null` |
| 100 | 0.933040 | 3,021.81 | 0.3309 | 0.3310 | 0.3331 | 0.4056 | 0.4897 | 0.1069 | `null` |
| 150 | 0.960080 | 2,157.12 | 0.4636 | 0.4650 | 0.4732 | 0.5678 | 0.6703 | 1.4582 | `null` |
| 200 | 0.973300 | 1,672.37 | 0.5980 | 0.6031 | 0.6135 | 0.7377 | 0.9109 | 5.1172 | `null` |
| 300 | 0.985300 | 1,150.10 | 0.8695 | 0.8582 | 0.8750 | 1.0479 | 1.2315 | -11.3149 | `null` |
| 400 | 0.990360 | 847.11 | 1.1805 | 1.1873 | 1.2195 | 1.4452 | 1.6261 | 6.7830 | `null` |
| 600 | 0.994700 | 590.10 | 1.6946 | 1.6959 | 1.7436 | 2.0774 | 2.3777 | 1.2639 | `null` |

### `faiss_n1000000_C1.json` (`faiss` `C1`, `N=1,000,000`, affinity `0x0400`, `generated_at=2026-11-17T13:29:59Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `ffi_delta_us` | `ndis` / `nhops` (`ns/ndis`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.570260 | 22,327.55 | 0.0448 | 0.0476 | 0.0433 | 0.0685 | 0.1025 | 3.2991 | 236.6 / 26.7 (`189.3`) |
| 20 | 0.713330 | 14,089.85 | 0.0710 | 0.0735 | 0.0673 | 0.1032 | 0.1462 | 3.0098 | 342.9 / 36.9 (`207.1`) |
| 30 | 0.783770 | 10,653.77 | 0.0939 | 0.0958 | 0.0897 | 0.1303 | 0.1663 | 2.5384 | 439.0 / 46.6 (`213.9`) |
| 40 | 0.827330 | 8,585.74 | 0.1165 | 0.1202 | 0.1139 | 0.1589 | 0.2168 | 4.3753 | 530.9 / 56.2 (`219.5`) |
| 50 | 0.857520 | 5,958.75 | 0.1678 | 0.1744 | 0.1525 | 0.2564 | 0.6345 | 7.7462 | 620.7 / 65.8 (`270.3`) |
| 70 | 0.894910 | 5,299.08 | 0.1887 | 0.1962 | 0.1856 | 0.2486 | 0.4408 | 8.5397 | 795.1 / 85.1 (`237.3`) |
| 100 | 0.926080 | 3,623.55 | 0.2760 | 0.2701 | 0.2659 | 0.3272 | 0.4263 | -4.4958 | 1049.2 / 114.4 (`263.1`) |
| 150 | 0.952130 | 2,579.01 | 0.3877 | 0.3977 | 0.3955 | 0.4710 | 0.6133 | 11.5372 | 1455.2 / 163.9 (`266.4`) |
| 200 | 0.965270 | 1,887.78 | 0.5297 | 0.5387 | 0.5359 | 0.6302 | 0.8715 | 10.9099 | 1842.2 / 213.5 (`287.5`) |
| 300 | 0.978240 | 1,198.73 | 0.8342 | 0.8453 | 0.8455 | 0.9782 | 1.3486 | 13.5442 | 2576.9 / 313.1 (`323.7`) |
| 400 | 0.984040 | 853.99 | 1.1710 | 1.1843 | 1.1839 | 1.3678 | 1.8598 | 16.1088 | 3271.5 / 413.0 (`357.9`) |
| 600 | 0.989550 | 510.34 | 1.9595 | 2.0235 | 2.0030 | 2.3986 | 3.0530 | 68.8127 | 4573.9 / 612.7 (`428.4`) |

### `rust_n1000000_C2.json` (`rust` `C2`, `N=1,000,000`, affinity `0x0004`, `generated_at=2026-11-17T13:18:36Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.733150 | 9,121.80 | 0.1096 | 0.1152 | 0.1064 | 0.1628 | 0.2376 | 5.5747 | `null` |
| 20 | 0.849490 | 5,936.97 | 0.1684 | 0.1674 | 0.1637 | 0.2280 | 0.3057 | -1.0728 | `null` |
| 30 | 0.900130 | 4,541.78 | 0.2202 | 0.2206 | 0.2202 | 0.2875 | 0.3477 | 0.4375 | `null` |
| 40 | 0.929140 | 3,618.95 | 0.2763 | 0.2927 | 0.2782 | 0.3873 | 0.8179 | 16.3858 | `null` |
| 50 | 0.947030 | 2,875.88 | 0.3477 | 0.3547 | 0.3378 | 0.4784 | 0.9864 | 6.9621 | `null` |
| 70 | 0.967590 | 2,140.86 | 0.4671 | 0.4745 | 0.4428 | 0.6512 | 1.9394 | 7.3964 | `null` |
| 100 | 0.980610 | 1,491.23 | 0.6706 | 0.5769 | 0.5854 | 0.7375 | 1.2115 | -93.6650 | `null` |
| 150 | 0.990040 | 1,129.48 | 0.8854 | 1.1314 | 0.8807 | 3.0912 | 3.4934 | 246.0576 | `null` |
| 200 | 0.993760 | 725.04 | 1.3792 | 1.0136 | 1.0450 | 1.2836 | 1.5165 | -365.6130 | `null` |
| 300 | 0.996540 | 723.52 | 1.3821 | 1.3757 | 1.4274 | 1.7668 | 2.0281 | -6.3824 | `null` |
| 400 | 0.997500 | 568.43 | 1.7592 | 1.8086 | 1.8765 | 2.3456 | 2.6447 | 49.4276 | `null` |
| 600 | 0.998040 | 389.74 | 2.5658 | 2.6245 | 2.6890 | 3.4665 | 3.9926 | 58.6814 | `null` |

### `rust_n1000000_C2-nkp.json` (`rust` `C2-nkp`, `N=1,000,000`, affinity `0x0004`, `generated_at=2026-11-17T14:55:15Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.710170 | 10,686.78 | 0.0936 | 0.0877 | 0.0851 | 0.1270 | 0.1589 | -5.9195 | `null` |
| 20 | 0.835650 | 7,563.33 | 0.1322 | 0.1344 | 0.1332 | 0.1821 | 0.2121 | 2.1464 | `null` |
| 30 | 0.891700 | 5,607.11 | 0.1783 | 0.1766 | 0.1771 | 0.2322 | 0.2684 | -1.7074 | `null` |
| 40 | 0.924010 | 4,555.34 | 0.2195 | 0.2202 | 0.2220 | 0.2867 | 0.3351 | 0.6555 | `null` |
| 50 | 0.943010 | 3,787.97 | 0.2640 | 0.2751 | 0.2793 | 0.3546 | 0.4061 | 11.0998 | `null` |
| 70 | 0.965470 | 2,710.03 | 0.3690 | 0.3741 | 0.3787 | 0.4861 | 0.5648 | 5.1014 | `null` |
| 100 | 0.980130 | 2,070.33 | 0.4830 | 0.4516 | 0.4658 | 0.5629 | 0.6361 | -31.4200 | `null` |
| 150 | 0.990180 | 1,577.72 | 0.6338 | 0.6309 | 0.6544 | 0.7859 | 0.8597 | -2.9156 | `null` |
| 200 | 0.994410 | 1,234.01 | 0.8104 | 0.8103 | 0.8394 | 1.0182 | 1.1311 | -0.0525 | `null` |
| 300 | 0.997610 | 687.81 | 1.4539 | 1.4334 | 1.3754 | 2.3009 | 2.9939 | -20.4918 | `null` |
| 400 | 0.998490 | 536.42 | 1.8642 | 1.8224 | 1.7683 | 2.8192 | 3.5590 | -41.7723 | `null` |
| 600 | 0.999140 | 393.06 | 2.5441 | 2.5412 | 2.5252 | 3.6734 | 4.3721 | -2.9126 | `null` |

### `faiss_n1000000_C2.json` (`faiss` `C2`, `N=1,000,000`, affinity `0x0400`, `generated_at=2026-11-17T13:12:10Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `ffi_delta_us` | `ndis` / `nhops` (`ns/ndis`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.695540 | 16,919.75 | 0.0591 | 0.0647 | 0.0600 | 0.0948 | 0.1267 | 6.1268 | 365.9 / 23.9 (`161.5`) |
| 20 | 0.825820 | 10,911.78 | 0.0916 | 0.0965 | 0.0919 | 0.1320 | 0.1647 | 5.5442 | 530.0 / 33.6 (`172.8`) |
| 30 | 0.884850 | 7,711.07 | 0.1297 | 0.1339 | 0.1271 | 0.1814 | 0.3313 | 5.0776 | 682.4 / 43.0 (`190.1`) |
| 40 | 0.918570 | 6,120.56 | 0.1634 | 0.1635 | 0.1580 | 0.2147 | 0.3228 | 1.2346 | 830.1 / 52.6 (`196.8`) |
| 50 | 0.938650 | 5,145.46 | 0.1943 | 0.2245 | 0.1990 | 0.3258 | 0.9163 | 31.5554 | 975.0 / 62.3 (`199.3`) |
| 70 | 0.962440 | 3,619.62 | 0.2763 | 0.2642 | 0.2569 | 0.3448 | 0.5717 | -10.5664 | 1257.2 / 81.9 (`219.8`) |
| 100 | 0.978210 | 2,567.67 | 0.3895 | 0.3480 | 0.3509 | 0.4285 | 0.5338 | -39.8261 | 1662.2 / 111.5 (`234.3`) |
| 150 | 0.988910 | 1,848.07 | 0.5411 | 0.5297 | 0.5254 | 0.6688 | 1.0503 | -9.3931 | 2297.7 / 161.2 (`235.5`) |
| 200 | 0.993460 | 1,332.06 | 0.7507 | 0.7703 | 0.7336 | 1.1732 | 1.8754 | 22.4516 | 2897.6 / 211.0 (`259.1`) |
| 300 | 0.996730 | 833.75 | 1.1994 | 1.1213 | 1.0903 | 1.5089 | 2.4322 | -74.2626 | 4018.0 / 310.9 (`298.5`) |
| 400 | 0.997910 | 601.76 | 1.6618 | 1.6734 | 1.6116 | 2.4276 | 3.4035 | 18.0113 | 5061.1 / 410.8 (`328.3`) |
| 600 | 0.998720 | 428.31 | 2.3348 | 2.3524 | 2.3788 | 2.7906 | 3.3884 | 23.2716 | 6991.6 / 610.6 (`333.9`) |

### `rust_n1000000_C3.json` (`rust` `C3`, `N=1,000,000`, affinity `0x0004`, `generated_at=2026-11-17T14:29:49Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `instant_overhead_us` | `dist_evals_mean` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.846140 | 4,650.24 | 0.2150 | 0.1964 | 0.1727 | 0.2772 | 0.6097 | -18.6569 | `null` |
| 20 | 0.929340 | 3,440.38 | 0.2907 | 0.3514 | 0.2824 | 0.5443 | 1.6262 | 60.7407 | `null` |
| 30 | 0.959890 | 2,901.37 | 0.3447 | 0.3170 | 0.3208 | 0.4169 | 0.4879 | -27.6339 | `null` |
| 40 | 0.974600 | 2,662.89 | 0.3755 | 0.3762 | 0.3877 | 0.4842 | 0.5433 | 0.6513 | `null` |
| 50 | 0.982780 | 2,212.47 | 0.4520 | 0.4531 | 0.4657 | 0.5873 | 0.6601 | 1.0801 | `null` |
| 70 | 0.990870 | 1,621.24 | 0.6168 | 0.6214 | 0.6458 | 0.8142 | 0.9167 | 4.5941 | `null` |
| 100 | 0.995720 | 1,212.04 | 0.8251 | 0.8261 | 0.8597 | 1.0942 | 1.2296 | 1.0169 | `null` |
| 150 | 0.998020 | 873.75 | 1.1445 | 1.1353 | 1.1805 | 1.5186 | 1.7116 | -9.1933 | `null` |
| 200 | 0.998810 | 698.87 | 1.4309 | 1.4365 | 1.4906 | 1.9571 | 2.1791 | 5.6499 | `null` |
| 300 | 0.999240 | 502.90 | 1.9884 | 2.0187 | 2.0903 | 2.7911 | 3.0903 | 30.2493 | `null` |
| 400 | 0.999350 | 394.92 | 2.5321 | 2.5313 | 2.6197 | 3.5151 | 3.8575 | -0.8505 | `null` |
| 600 | 0.999440 | 293.30 | 3.4095 | 3.4011 | 3.5164 | 4.7340 | 5.1914 | -8.3807 | `null` |

### `faiss_n1000000_C3.json` (`faiss` `C3`, `N=1,000,000`, affinity `0x0400`, `generated_at=2026-11-17T14:03:02Z`)

| `efSearch` | `Recall@10` | Mode 1 QPS | Mode 1 Mean (ms) | Mode 2 Mean (ms) | Mode 2 p50 (ms) | Mode 2 p95 (ms) | Mode 2 p99 (ms) | `ffi_delta_us` | `ndis` / `nhops` (`ns/ndis`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.775200 | 12,702.16 | 0.0787 | 0.0812 | 0.0781 | 0.1179 | 0.1472 | 3.1044 | 539.6 / 21.4 (`145.8`) |
| 20 | 0.888180 | 8,229.43 | 0.1215 | 0.1250 | 0.1215 | 0.1718 | 0.2143 | 4.3672 | 779.9 / 30.8 (`155.8`) |
| 30 | 0.933900 | 6,103.92 | 0.1638 | 0.1648 | 0.1644 | 0.2209 | 0.2704 | 2.0115 | 1008.4 / 40.3 (`162.4`) |
| 40 | 0.956790 | 4,890.81 | 0.2045 | 0.2113 | 0.2094 | 0.2774 | 0.3605 | 8.0927 | 1231.4 / 50.0 (`166.1`) |
| 50 | 0.969840 | 4,101.28 | 0.2438 | 0.2517 | 0.2511 | 0.3294 | 0.4190 | 9.3825 | 1448.7 / 59.8 (`168.3`) |
| 70 | 0.983740 | 3,094.91 | 0.3231 | 0.3337 | 0.3352 | 0.4343 | 0.5423 | 12.2447 | 1866.9 / 79.5 (`173.1`) |
| 100 | 0.992040 | 2,236.31 | 0.4472 | 0.4595 | 0.4635 | 0.5933 | 0.7590 | 14.0403 | 2459.0 / 109.2 (`181.9`) |
| 150 | 0.996550 | 1,541.90 | 0.6486 | 0.6634 | 0.6746 | 0.8591 | 1.0902 | 16.9402 | 3378.3 / 159.0 (`192.0`) |
| 200 | 0.998160 | 1,154.76 | 0.8660 | 0.8854 | 0.8971 | 1.1464 | 1.4943 | 21.7050 | 4234.9 / 208.9 (`204.5`) |
| 300 | 0.999060 | 771.49 | 1.2962 | 1.3101 | 1.3334 | 1.6704 | 2.0726 | 17.2198 | 5814.9 / 308.7 (`222.9`) |
| 400 | 0.999280 | 563.01 | 1.7762 | 1.7790 | 1.8072 | 2.2529 | 2.8021 | 7.2346 | 7267.5 / 408.6 (`244.4`) |
| 600 | 0.999390 | 361.33 | 2.7676 | 2.7948 | 2.8361 | 3.4775 | 4.0580 | 33.4295 | 9918.8 / 608.6 (`279.0`) |
