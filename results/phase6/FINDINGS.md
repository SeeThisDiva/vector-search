# Phase 6 Vector Quantization (`SQ8` & `PQ`): Complete `100k` & `1M` Empirical Findings

**Task:** Milestone M4 Task `T4.9` (Phase 6 Synthesis & Trade-Off Analysis)  
**Date:** 2026-11-18  
**Clean Git Commit:** `9f79165` (`git_dirty: false`, `measured_this_run: true`, `warmup_performed: true` across all 23 Phase 6 JSON artifacts)  
**Dataset:** Authentic INRIA Texmex `SIFT1M` (`D = 128`, `L2Squared`, `k = 10`, `Q = 10,000`, `seed = 42`) — `100k` prefix (`11` arms, `reps = 3`, cached exact GT `data/gt_100000_k100.ivecs`) and full `1M` corpus (`9` Tier 1+2 arms, `reps = 5`, official `data/sift1m/sift_groundtruth.ivecs`)  
**Host Hardware & Toolchain:** `12th Gen Intel(R) Core(TM) i7-12650H` (16 logical cores, `15.78 GiB` RAM, Windows `x86_64`), `power_mode = "balanced"`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)` (`#![forbid(unsafe_code)]`, `std`-only)  
**Source Artifacts:** [`results/phase6/pilot/microbench_kernels.json`](./pilot/microbench_kernels.json), [`results/phase6/summary_n100000.json`](./summary_n100000.json), [`results/phase6/summary_n1000000.json`](./summary_n1000000.json), and all `20` per-arm JSON reports (`arm_n100000_*.json`, `arm_n1000000_*.json`)

---

## 1. Executive Summary (`100k` & `1M` Headline Results)

1. **Streaming `SQ8-Only` Breaks the `300 MiB` Barrier at `1M` (`-58.8%` Payload, `-57.1%` Query RSS):**
   - Two-pass streaming `.fvecs` construction (`sq8_only_qbuild`, `QuantizedDirect`) eliminates `RowMatrix<f32>` from RAM entirely, cutting `build_peak_rss_mb` to **`48.90 MiB`** at `100k` (`-44.2%` vs `87.70 MiB` `f32`-graph `SQ8-Only`) and **`277.29 MiB`** at `1M` (**`-64.4%`** vs `778.37 MiB` `f32`-graph `SQ8-Only`, **`-56.9%`** vs `642.71 MiB` `f32_m16`).
   - Isolated query-serving `PeakWorkingSet64` drops from `74.95 MiB` (`100k`) / `642.71 MiB` (`1M`) on `f32_m16` down to **`38.61 MiB` / `276.02 MiB`** on `sq8_only_f32graph` (`-48.5%` / **`-57.1%`**), **`29.22 MiB` / `184.67 MiB`** on `pq32_only_f32graph` (`-61.0%` / **`-71.3%`**), and **`28.13 MiB` / `169.55 MiB`** on `pq16_only_f32graph` (`-62.5%` / **`-73.6%`**).
   - Steady-state payload falls from `657.57 / 660.91 B/vec` (`f32_m16`) to **`272.55 / 272.54 B/vec`** (`SQ8-Only`, `-58.6%` / `-58.8%`), **`177.84 / 176.67 B/vec`** (`PQ32-Only`, `-73.0%` / `-73.3%`), and **`161.84 / 160.67 B/vec`** (`PQ16-Only`, `-75.4%` / `-75.7%`) — leaving `CompactHnswGraph` adjacency (`144.53 B/vec`) as **`89.3%–90.0%`** of `PQ16-Only` index memory.
2. **`QuantizedDirect` Matches `Float32GraphThenQuantize` Within `< 0.0003` at High Recall:**
   - Despite constructing the entire HNSW graph purely from symmetric `u8`-vs-`u8` SQ8 distances, `sq8_only_qbuild` stays within **`-0.00012` at `ef=100`** (`0.98464` vs `0.98476`) and **`-0.00002` at `ef=600`** (`0.98776` vs `0.98778`) at `100k`, and **matches or edges out** `f32`-graph SQ8 at `1M`: **`+0.00024` at `ef=100`** (`0.97220` vs `0.97196`) and **`+0.00003` at `ef=600`** (`0.98767` vs `0.98764`).
3. **`SQ8+Rerank` (`R = 20`) Recovers `100%` of `f32` Recall; `SQ8-Only` Ceilings Scale Cleanly:**
   - Rescoring just **`R = 20`** top SQ8 candidates using exact `f32` vectors restores **100% of `FullBeam` and `f32` recall across every `efSearch in [20..600]`** at both `100k` (`0.99370` @ `ef=100`, `0.99712` @ `ef=600` for `M=16`; `0.99938` -> `0.99999` for `M=32`) and `1M` (`0.98060` @ `ef=100`, `0.99804` @ `ef=600` for `M=16`).
   - Without rerank, `SQ8-Only M=16` peaks at **`0.98778`** (`100k`) / **`0.98764`** (`1M`); widening to `M=32, efC=200` (`sq8_only_m32`, `400.22 / 400.28 B/vec`, `51.06 / 398.04 MiB` query RSS) reaches **`0.99060`** (`100k`, breaking `0.99`) and **`0.98898`** (`1M`, `+3.38 pp` at `ef=50` and `+1.37 pp` at `ef=100` over `M=16`).
4. **`PQ+Rerank` Beats `f32_m16` QPS and `p99` at Both Scales for `+1.8%`–`+4.9%` Payload:**
   - Unreranked `PQ16-Only` and `PQ32-Only` saturate at **`0.61519` / `0.76461`** (`100k`) and **`0.54520` / `0.72305`** (`1M`).
   - Pairing `16 B/vec` L3-resident PQ traversal (`9.83 ns` ADC microbench kernel) with top-$R$ `f32` reranking transforms the speed frontier: at `>= 0.99` recall, `pq16_rerank_f32graph` outperforms `f32_m16` by **`+40.9%` QPS / `-30.7%` `p99`** at `100k` (`5,269.0` vs `3,738.9 QPS`) and **`+27.0%` QPS / `-40.2%` `p99`** at `1M` (`1,553.9` vs `1,223.5 QPS`, `0.8536` vs `1.4282 ms`). At `>= 0.995` recall (`1M`), `pq32_rerank_f32graph` (`ef=300, FB`) reaches `0.99561` at **`994.0 QPS` (`p99 = 1.4300 ms`)** — **`+49.8%` faster** than Phase 6 `f32_m16` (`663.4 QPS`, `2.4524 ms`) and even **`+9.1%` faster** than solo Phase 4 `f32_m16` (`911.0 QPS`, `1.4942 ms`).

---

## 2. Provenance, Sharding & Honest Disclosures

Every Phase 6 artifact (`pilot/microbench_kernels.json`, `summary_n100000.json`, `summary_n1000000.json`, `11` `100k` arm JSONs, and `9` `1M` arm JSONs — **23 JSON files total**) was verified to record **`git_commit == "9f79165"`**, **`git_dirty == false`**, **`measured_this_run == true`**, **`warmup_performed == true`**, and **`is_authentic_sift == true`** (for all `20` arm reports). Summary aggregates match every individual arm JSON within `< 1e-6`.

### Honest Execution Disclosures

1. **Windows `balanced` Power Plan (`power_mode = "balanced"`):** Both `100k` and `1M` sweeps executed under Windows `balanced` power management rather than `best_performance`. Occasional OS frequency-scaling or background scheduling spikes appear in `p99` tails and are preserved unedited:
   - **`100k`:** `sq8_rerank_m32 (R=20, ef=400)` hit `p99 = 2.3522 ms` (`run_p99 = [6.0640, 2.3522, 1.4658]`, `spread = 4.5982 ms`, exceeding `ef=600`'s `1.8752 ms`); `pq16_only_f32graph (ef=600)` hit `p99 = 2.7616 ms` (`runs = [3.6450, 2.7616, 2.6345]`, vs `0.6661 ms` on `pq16_only_qbuild`); `pq16_only_f32graph (ef=20)` hit `0.1408 ms` vs `0.0943 ms` at `ef=30`.
   - **`1M`:** `sq8_only_m32` exhibited severe tail jitter for `ef >= 70` (`ef=70`: `1.6285 ms`, `spread = 7.0129 ms`; `ef=100`: `6.3654 ms`, `runs = [3.5887, 9.0077, 3.9136, 6.3654, 8.4021]`), `pq32_only_f32graph (ef=50)` spiked to `2.0461 ms` (vs `1.4153 ms` at `ef=70`), `sq8_rerank_f32graph (ef=100, R=20)` hit `1.2882 ms` (vs `0.6256 ms` at `ef=100, R=100`), and `sq8_rerank_f32graph (ef=300, R=20)` hit `2.3690 ms` (vs `1.9517 ms` at `ef=400, R=20`).
2. **`100k` Solo Pinning vs `1M` Two-Core Sharding (`0x0004` / `0x0400`):**
   - **`100k` (`11` arms):** Executed sequentially **solo** on P-core `0x0004` (`620.80 s` build + `4,025.75 s` 4-pass sweep = **`77.44 min`** total compute).
   - **`1M` (`9` Tier 1+2 arms, `228` points, `6` passes):** Executed under balanced **2-P-core sharding** (`summary_n1000000.json` stamps `"processor_affinity_mask": "0x0004,0x0400"`):
     - **Shard A (`0x0004`, `4` arms, `72` pts):** `f32_m16`, `sq8_only_f32graph`, `sq8_rerank_f32graph` (`--rerank-list 20,100,full_beam`), `sq8_only_m32` -> `51.63 m` build + `58.02 m` sweep = **`109.64 min` (`1.83 h`)**.
     - **Shard B (`0x0400`, `5` arms, `156` pts):** `sq8_only_qbuild`, `pq16_only_f32graph`, `pq32_only_f32graph`, `pq16_rerank_f32graph`, `pq32_rerank_f32graph` -> `49.57 m` build + `84.84 m` sweep = **`134.42 min` (`2.24 h`)**.
     - **Combined `1M` Active Core-Minutes:** `101.20 m` build + `142.86 m` 6-pass sweep = **`244.06 min` (`4.07 h`)**, completing inside the `252.89 min` `T4.6` conservative bound (`PROPOSAL.md`).
   - **Shared L3 / DRAM Contention Disclosure:** Concurrent execution on `0x0004` + `0x0400` under `balanced` power management slowed DRAM-bandwidth-bound `f32_m16` (`512 B/vec`, `642.71 MiB`) relative to solo Phase 4 `f32_m16` (`cd48e7f`, `450.30 s` build; `ef=100`: `2,383.9 QPS`, `p99 = 0.5515 ms`) vs Phase 6 `f32_m16` (`574.78 s` build; `ef=100`: `1,676.2 QPS`, `p99 = 1.1951 ms`). Similarly, `sq8_only_m32` (`09:37:58Z`, `0x0004`) overlapped `pq32_rerank_f32graph` (`09:13:27Z`, `0x0400`), amplifying `sq8_only_m32`'s `ef >= 70` tail latency. Both Phase 6 same-commit `f32_m16` and Phase 4 solo `f32` baselines are cited explicitly below.
3. **Isolated vs Process Lifetime `PeakWorkingSet64` (`Rerank` vs `Only`):**
   - For all `Only` arms (`query_rss_profiled_isolated: true`), `query_serving_peak_rss_mb` is measured by serializing the quantized index (`graph + codes + cal/cb`) and reloading it inside a clean child process (`isolated_query_subprocess_high_water_mark`), ensuring `build_peak_rss_mb` from temporary `f32` graph construction does not pollute serving RSS.
   - For `f32_m16` and all `rerank` arms (`query_rss_profiled_isolated: false`), raw `512 B/vec` `f32` vectors (`48.83 MiB` @ `100k`, `488.28 MiB` @ `1M`) **genuinely remain resident in memory** during query serving to perform exact rescoring, so `query_serving_peak_rss_mb == build_peak_rss_mb` (`process_lifetime_high_water_mark`).

---

## 3. Build Time Decomposition & Graph Fingerprints (`100k` & `1M`)

### 3.1 `100k` Build Time Breakdown (`11` Arms, Solo `0x0004`)

| `arm_id` | $M$ / $\text{efC}$ | Build Mode | Streaming | Train / Cal (s) | Encode (s) | Graph Build (s) | Total Build (s) | Fingerprint (`hex`) |
| :--- | :---: | :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `f32_m16` | `16 / 100` | `f32_graph` | `false` | `0.0000` | `0.0000` | `27.2966` | `27.2966` | `0x80af03bd7aebaade` |
| `sq8_only_f32graph` | `16 / 100` | `f32_graph` | `false` | `0.0264` | `0.2414` | `27.8108` | `28.0855` | `0x80af03bd7aebaade` |
| `sq8_only_qbuild` | `16 / 100` | `quantized_direct` | **`true`** | `0.0540` | `0.2705` | `29.7108` | `30.0359` | `0x3ba635aefd6d2449` |
| `sq8_rerank_f32graph` | `16 / 100` | `f32_graph` | `false` | `0.0261` | `0.2389` | `27.5629` | `27.8345` | `0x80af03bd7aebaade` |
| `sq8_only_m32` | `32 / 200` | `f32_graph` | `false` | `0.0262` | `0.2388` | `72.3798` | `72.6515` | `0xa57cab3db2d6766f` |
| `sq8_rerank_m32` | `32 / 200` | `f32_graph` | `false` | `0.0265` | `0.2406` | `73.6230` | `73.8967` | `0xa57cab3db2d6766f` |
| `pq16_only_f32graph` | `16 / 100` | `f32_graph` | `false` | `38.4384` | `3.2385` | `27.1605` | `68.8440` | `0x80af03bd7aebaade` |
| `pq32_only_f32graph` | `16 / 100` | `f32_graph` | `false` | `49.8186` | `3.3109` | `34.5550` | `87.7001` | `0x80af03bd7aebaade` |
| `pq16_only_qbuild` | `16 / 100` | `quantized_direct` | **`true`** | `37.2900` | `2.9833` | `18.9438` | `59.2195` | `0x7328beaf5527e6d4` |
| `pq16_rerank_f32graph` | `16 / 100` | `f32_graph` | `false` | `37.1334` | `2.9594` | `27.5197` | `67.6189` | `0x80af03bd7aebaade` |
| `pq32_rerank_f32graph` | `16 / 100` | `f32_graph` | `false` | `45.5704` | `3.3521` | `28.6877` | `77.6170` | `0x80af03bd7aebaade` |

### 3.2 `1M` Build Time Breakdown (`9` Arms, Sharded `0x0004` / `0x0400`)

| `arm_id` | $M$ / $\text{efC}$ | Aff. | Build Mode | Streaming | Train / Cal (s) | Encode (s) | Graph Build (s) | Total Build (s) | Fingerprint (`hex`) |
| :--- | :---: | :---: | :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `f32_m16` | `16 / 100` | `0x0004` | `f32_graph` | `false` | `0.0000` | `0.0000` | `574.7792` | `574.7792` | `0x44044886c62bca5e` |
| `sq8_only_f32graph` | `16 / 100` | `0x0004` | `f32_graph` | `false` | `0.2647` | `2.4338` | `487.7146` | `490.4803` | `0x44044886c62bca5e` |
| `sq8_only_qbuild` | `16 / 100` | `0x0400` | `quantized_direct` | **`true`** | `0.6190` | `2.8077` | `543.1089` | `546.5362` | `0xb936e37bcf6011c8` |
| `sq8_rerank_f32graph` | `16 / 100` | `0x0004` | `f32_graph` | `false` | `0.2938` | `2.7250` | `569.2038` | `572.2921` | `0x44044886c62bca5e` |
| `sq8_only_m32` | `32 / 200` | `0x0004` | `f32_graph` | `false` | `0.3178` | `2.4820` | `1,457.1204` | `1,459.9871` | `0xb8ddb30d73d248c5` |
| `pq16_only_f32graph` | `16 / 100` | `0x0400` | `f32_graph` | `false` | `37.0146` | `29.5245` | `516.0504` | `582.6592` | `0x44044886c62bca5e` |
| `pq32_only_f32graph` | `16 / 100` | `0x0400` | `f32_graph` | `false` | `54.2631` | `38.6188` | `550.2742` | `643.2247` | `0x44044886c62bca5e` |
| `pq16_rerank_f32graph` | `16 / 100` | `0x0400` | `f32_graph` | `false` | `36.9532` | `29.6209` | `492.1764` | `558.8162` | `0x44044886c62bca5e` |
| `pq32_rerank_f32graph` | `16 / 100` | `0x0400` | `f32_graph` | `false` | `51.7656` | `34.8899` | `556.4302` | `643.1560` | `0x44044886c62bca5e` |

**Build Takeaways:**
- Every `Float32GraphThenQuantize` arm reproduced Phase 4's exact golden fingerprints (`0x80af03bd7aebaade` / `0xa57cab3db2d6766f` at `100k`; `0x44044886c62bca5e` / `0xb8ddb30d73d248c5` at `1M`), confirming `PRESERVING` parity in `src/hnsw.rs`.
- Because `N_train = min(N, 65_536)` caps PQ $k$-means++ / Lloyd training, PQ training remains **`37.0–54.3 s`** at `1M` (essentially unchanged from `37.1–49.8 s` at `100k`), adding only **`~11%–16%`** overhead atop `1M` `f32` graph build while PQ encoding takes **`29.5–38.6 s`** (`~10x` of `3.0–3.4 s` at `100k`). SQ8 calibration + encoding takes just **`~2.7–3.4 s` total** (`0.5%` of `1M` build).
- Conversion/copying overhead `total - (train + encode + graph_build)` is `0.0006–0.0156 s` at `100k` and `0.0006–0.0704 s` at `1M`.

---

## 4. Steady-State Memory Breakdown & Peak Working Set (`100k` & `1M`)

### 4.1 `100k` Component Memory & Peak RSS (`11` Arms)

| `arm_id` | `codes` (B/v) | `graph` (B/v) | `cal/cb` (B/v) | `raw` (B/v) | Total `steady_state` (B/v) | Payload (MiB) | Payload vs `f32_m16` | `build_peak_rss_mb` | `query_peak_rss_mb` | Isolated RSS | Query RSS vs `f32_m16` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `f32_m16` | `0.00` | `145.5706` | `0.0000` | `512.00` | **`657.5706`** | `62.71` | `0.0%` | `74.95` | `74.95` | `false` | `0.0%` |
| `sq8_only_f32graph` | `128.00` | `144.5315` | `0.0205` | `0.00` | **`272.5520`** | `25.99` | **`-58.6%`** | `87.70` | **`38.61`** | `true` | **`-48.5%`** |
| `sq8_only_qbuild` *(stream)* | `128.00` | `144.5315` | `0.0205` | `0.00` | **`272.5520`** | `25.99` | **`-58.6%`** | **`48.90`** | **`38.86`** | `true` | **`-48.2%`** |
| `sq8_rerank_f32graph` | `128.00` | `144.5315` | `0.0205` | `512.00` | **`784.5520`** | `74.82` | `+19.3%` | `87.75` | `87.75` | `false` | `+17.1%` |
| `sq8_only_m32` | `128.00` | `272.2002` | `0.0205` | `0.00` | **`400.2207`** | `38.17` | **`-39.1%`** | `113.72` | **`51.06`** | `true` | **`-31.9%`** |
| `sq8_rerank_m32` | `128.00` | `272.2002` | `0.0205` | `512.00` | **`912.2207`** | `87.00` | `+38.7%` | `113.63` | `113.63` | `false` | `+51.6%` |
| `pq16_only_f32graph` | `16.00` | `144.5315` | `1.3107` | `0.00` | **`161.8422`** | `15.43` | **`-75.4%`** | `87.68` | **`28.13`** | `true` | **`-62.5%`** |
| `pq16_only_qbuild` *(stream)* | `16.00` | `144.5315` | `1.3107` | `0.00` | **`161.8422`** | `15.43` | **`-75.4%`** | **`48.90`** | **`28.32`** | `true` | **`-62.2%`** |
| `pq16_rerank_f32graph` | `16.00` | `144.5315` | `1.3107` | `512.00` | **`673.8422`** | `64.26` | `+2.5%` | `87.20` | `87.20` | `false` | `+16.3%` |
| `pq32_only_f32graph` | `32.00` | `144.5315` | `1.3107` | `0.00` | **`177.8422`** | `16.96` | **`-73.0%`** | `87.75` | **`29.22`** | `true` | **`-61.0%`** |
| `pq32_rerank_f32graph` | `32.00` | `144.5315` | `1.3107` | `512.00` | **`689.8422`** | `65.79` | `+4.9%` | `88.42` | `88.42` | `false` | `+18.0%` |
| **Phase 4 `f32_m32`** *(ref)* | `0.00` | `273.4067` | `0.0000` | `512.00` | `785.4067` | `74.90` | `+19.4%` | `92.59` | `92.59` | `false` | `+23.5%` |

### 4.2 `1M` Component Memory & Peak RSS (`9` Arms)

| `arm_id` | `codes` (B/v) | `graph` (B/v) | `cal/cb` (B/v) | `raw` (B/v) | Total `steady_state` (B/v) | Payload (MiB) | Payload vs `f32_m16` | `build_peak_rss_mb` | `query_peak_rss_mb` | Isolated RSS | Query RSS vs `f32_m16` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `f32_m16` | `0.00` | `148.9129` | `0.0000` | `512.00` | **`660.9129`** | `630.30` | `0.0%` | `642.71` | `642.71` | `false` | `0.0%` |
| `sq8_only_f32graph` | `128.00` | `144.5340` | `0.0020` | `0.00` | **`272.5361`** | `259.91` | **`-58.8%`** | `778.37` | **`276.02`** | `true` | **`-57.1%`** |
| `sq8_only_qbuild` *(stream)* | `128.00` | `144.5340` | `0.0020` | `0.00` | **`272.5361`** | `259.91` | **`-58.8%`** | **`277.29`** | **`276.18`** | `true` | **`-57.0%`** |
| `sq8_rerank_f32graph` | `128.00` | `144.5340` | `0.0020` | `512.00` | **`784.5361`** | `748.19` | `+18.7%` | `778.41` | `778.41` | `false` | `+21.1%` |
| `sq8_only_m32` | `128.00` | `272.2816` | `0.0020` | `0.00` | **`400.2836`** | `381.74` | **`-39.4%`** | `1,022.15` | **`398.04`** | `true` | **`-38.1%`** |
| `pq16_only_f32graph` | `16.00` | `144.5340` | `0.1311` | `0.00` | **`160.6651`** | `153.22` | **`-75.7%`** | `779.20` | **`169.55`** | `true` | **`-73.6%`** |
| `pq32_only_f32graph` | `32.00` | `144.5340` | `0.1311` | `0.00` | **`176.6651`** | `168.48` | **`-73.3%`** | `768.55` | **`184.67`** | `true` | **`-71.3%`** |
| `pq16_rerank_f32graph` | `16.00` | `144.5340` | `0.1311` | `512.00` | **`672.6651`** | `641.50` | **`+1.8%`** | `778.37` | `778.37` | `false` | `+21.1%` |
| `pq32_rerank_f32graph` | `32.00` | `144.5340` | `0.1311` | `512.00` | **`688.6651`** | `656.76` | **`+4.2%`** | `778.41` | `778.41` | `false` | `+21.1%` |
| **Phase 4 `f32_m32`** *(ref)* | `0.00` | `272.3254` | `0.0000` | `512.00` | `784.3254` | `748.00` | `+18.7%` | `767.40` | `767.40` | `false` | `+19.4%` |

*(Graph anatomy at `1M` `M=16`: `CompactHnswGraph` uses `8.00 B/v` `node_meta` + `132.00 B/v` `layer0` + `4.5340 B/v` `upper` = `144.5340 B/vec`, trimming `4.3789 B/vec` of upper-layer slack vs `HnswIndex`'s `148.9129 B/vec`. Calibration overhead is fixed at `2,048 B` [`2.0 KiB`] for SQ8 [`mins/scales/inv_scales/scales_sq`] and `131,072 B` [`128.0 KiB`] for PQ centroids [`256 * 128 * 4 B`], with `aux_vector_bytes = 0` because Way A is the default kernel.)*

---

## 5. Recall Ceilings, Fixed-`efSearch` & Matched-Recall Performance

### 5.1 `100k` Recall Ceiling & Fixed-`efSearch` (`50 / 100 / 200 / 600`)

| `arm_id` (Curve) | Ceiling (`ef=600` / Peak) | `ef=50` R@10 / QPS / p99 | `ef=100` R@10 / QPS / p99 | `ef=200` R@10 / QPS / p99 | `ef=600` R@10 / QPS / p99 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `f32_m16` | `0.997020` | `0.98019` / `6,523.3` / `0.2200 ms` | `0.99367` / `3,738.9` / `0.3802 ms` | `0.99645` / `2,129.8` / `0.6649 ms` | `0.99702` / `872.5` / `1.6169 ms` |
| `sq8_only_f32graph` | `0.987780` | `0.97256` / `7,431.3` / `0.2157 ms` | `0.98476` / `4,234.9` / `0.3583 ms` | `0.98726` / `2,506.4` / `0.5999 ms` | `0.98778` / `1,066.7` / `1.3900 ms` |
| `sq8_only_qbuild` | `0.987760` | `0.97251` / `7,731.6` / `0.1906 ms` | `0.98464` / `4,177.7` / `0.3495 ms` | `0.98724` / `2,428.8` / `0.6020 ms` | `0.98776` / `1,046.6` / `1.3757 ms` |
| `sq8_rerank_f32graph` (`R=20`) | `0.997120` | `0.98022` / `8,007.5` / `0.1795 ms` | `0.99370` / `4,581.4` / `0.3152 ms` | `0.99654` / `2,636.6` / `0.5637 ms` | `0.99712` / `1,105.2` / `1.2952 ms` |
| `sq8_rerank_f32graph` (`FullBeam`) | `0.997120` | `0.98022` / `7,161.6` / `0.2000 ms` | `0.99370` / `4,002.0` / `0.3586 ms` | `0.99654` / `2,265.9` / `0.6532 ms` | `0.99712` / `879.9` / `1.6404 ms` |
| `sq8_only_m32` | `0.990600` | `0.98671` / `5,266.5` / `0.2858 ms` | `0.99008` / `3,087.0` / `0.4931 ms` | `0.99053` / `1,901.0` / `0.8131 ms` | `0.99060` / `851.1` / `1.7424 ms` |
| `sq8_rerank_m32` (`R=20`) | `0.999990` | `0.99554` / `5,302.4` / `0.2801 ms` | `0.99938` / `3,097.4` / `0.4969 ms` | `0.99990` / `1,885.6` / `0.8048 ms` | `0.99999` / `833.6` / `1.8752 ms` |
| `sq8_rerank_m32` (`FullBeam`) | `0.999990` | `0.99554` / `4,934.8` / `0.3190 ms` | `0.99938` / `2,931.9` / `0.5352 ms` | `0.99990` / `1,713.6` / `0.9270 ms` | `0.99999` / `744.8` / `2.0628 ms` |
| `pq16_only_f32graph` | `0.614680` *(peak `0.61519` @100)* | `0.61397` / `13,540.5` / `0.1122 ms` | `0.61519` / `8,261.9` / `0.1879 ms` | `0.61484` / `4,839.1` / `0.3195 ms` | `0.61468` / `1,520.7` / `2.7616 ms` |
| `pq16_only_qbuild` | `0.616130` | `0.61305` / `13,750.4` / `0.1000 ms` | `0.61524` / `8,494.5` / `0.1526 ms` | `0.61581` / `4,941.3` / `0.2662 ms` | `0.61613` / `1,982.5` / `0.6661 ms` |
| `pq16_rerank_f32graph` (`R=100`) | `0.989160` | `0.98441` / `7,260.3` / `0.1754 ms` *(eff=100)* | `0.98441` / `7,018.9` / `0.2091 ms` | `0.98847` / `4,360.7` / `0.3358 ms` | `0.98916` / `1,857.3` / `0.7750 ms` |
| `pq16_rerank_f32graph` (`R=200`) | `0.996140` | `0.99520` / `4,112.1` / `0.3347 ms` *(eff=200)* | `0.99520` / `4,109.4` / `0.3403 ms` *(eff=200)* | `0.99520` / `4,096.7` / `0.3349 ms` | `0.99614` / `1,823.6` / `0.7410 ms` |
| `pq16_rerank_f32graph` (`FullBeam`) | **`0.997260`** | `0.94316` / `12,119.3` / `0.1247 ms` | `0.98441` / `7,252.8` / `0.1985 ms` | `0.99520` / `4,119.0` / `0.3402 ms` | `0.99726` / `1,556.5` / `0.9398 ms` |
| `pq32_only_f32graph` | `0.764610` | `0.75922` / `9,827.6` / `0.1436 ms` | `0.76396` / `6,337.5` / `0.2025 ms` | `0.76461` / `3,829.2` / `0.3435 ms` | `0.76461` / `1,561.5` / `0.8689 ms` |
| `pq32_rerank_f32graph` (`R=100`) | `0.996530` | `0.99178` / `5,303.1` / `0.2669 ms` *(eff=100)* | `0.99178` / `5,449.6` / `0.2585 ms` | `0.99580` / `3,362.3` / `0.4336 ms` | `0.99653` / `1,482.6` / `0.9449 ms` |
| `pq32_rerank_f32graph` (`R=200`) | `0.996620` | `0.99587` / `3,222.7` / `0.4352 ms` *(eff=200)* | `0.99587` / `3,144.9` / `0.4602 ms` *(eff=200)* | `0.99587` / `3,213.9` / `0.4121 ms` | `0.99662` / `1,442.1` / `0.9521 ms` |
| `pq32_rerank_f32graph` (`FullBeam`) | `0.996620` | `0.97433` / `8,786.2` / `0.1537 ms` | `0.99178` / `5,468.1` / `0.2581 ms` | `0.99587` / `3,157.9` / `0.4485 ms` | `0.99662` / `1,286.4` / `1.1444 ms` |
| **Phase 4 `f32_m32`** *(ref)* | `0.999990` | `0.99548` / `4,279.1` / `0.3412 ms` | `0.99939` / `2,541.9` / `0.5803 ms` | `0.99991` / `1,512.6` / `0.9935 ms` | `0.99999` / `674.5` / `2.1927 ms` |

### 5.2 `1M` Recall Ceiling & Fixed-`efSearch` (`50 / 100 / 200 / 600`)

| `arm_id` (Curve) | Ceiling (`ef=600` / Peak) | `ef=50` R@10 / QPS / p99 | `ef=100` R@10 / QPS / p99 | `ef=200` R@10 / QPS / p99 | `ef=600` R@10 / QPS / p99 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `f32_m16` *(Phase 6, sharded)* | `0.998040` | `0.94703` / `2,914.0` / `0.8210 ms` | `0.98061` / `1,676.2` / `1.1951 ms` | `0.99376` / `939.4` / `1.8406 ms` | `0.99804` / `370.3` / `4.0915 ms` |
| **Phase 4 `f32_m16`** *(solo ref)* | `0.998040` | `0.94703` / `4,209.0` / `0.3230 ms` | `0.98061` / `2,383.9` / `0.5515 ms` | `0.99376` / `1,296.0` / `1.0673 ms` | `0.99804` / `515.5` / `2.6370 ms` |
| `sq8_only_f32graph` | `0.987640` | `0.94028` / `3,816.2` / `0.3981 ms` | `0.97196` / `2,256.4` / `0.6402 ms` | `0.98396` / `1,268.5` / `1.1343 ms` | `0.98764` / `498.2` / `2.8813 ms` |
| `sq8_only_qbuild` | `0.987670` | `0.94052` / `3,725.2` / `0.3975 ms` | `0.97220` / `2,145.2` / `0.6740 ms` | `0.98397` / `1,199.9` / `1.1993 ms` | `0.98767` / `472.0` / `3.0369 ms` |
| `sq8_rerank_f32graph` (`R=20`) | `0.998040` | `0.94704` / `3,396.5` / `0.6797 ms` | `0.98060` / `2,005.4` / `1.2882 ms` | `0.99381` / `1,081.4` / `1.8181 ms` | `0.99804` / `510.2` / `2.7620 ms` |
| `sq8_rerank_f32graph` (`R=100`) | `0.998040` | `0.98060` / `2,183.2` / `0.6141 ms` *(eff=100)* | `0.98060` / `2,172.2` / `0.6256 ms` | `0.99381` / `1,242.8` / `1.1004 ms` | `0.99804` / `485.0` / `2.9574 ms` |
| `sq8_rerank_f32graph` (`FullBeam`) | `0.998040` | `0.94704` / `3,647.1` / `0.3881 ms` | `0.98060` / `2,067.8` / `0.6690 ms` | `0.99381` / `1,118.5` / `1.3588 ms` | `0.99804` / `446.9` / `3.2100 ms` |
| `sq8_only_m32` | `0.988980` | `0.97409` / `2,424.2` / `0.6234 ms` | `0.98567` / `939.7` / `6.3654 ms` | `0.98846` / `674.9` / `4.3937 ms` | `0.98898` / `274.1` / `9.2786 ms` |
| `pq16_only_f32graph` | `0.544780` *(peak `0.54520` @300)* | `0.53950` / `6,991.7` / `0.2167 ms` | `0.54429` / `4,181.3` / `0.3587 ms` | `0.54516` / `2,393.7` / `0.6031 ms` | `0.54478` / `925.2` / `1.5459 ms` |
| `pq16_rerank_f32graph` (`R=100`) | `0.969260` | `0.95032` / `3,893.8` / `0.3500 ms` *(eff=100)* | `0.95032` / `3,805.5` / `0.3791 ms` | `0.96477` / `2,363.5` / `0.5792 ms` | `0.96926` / `963.4` / `1.4041 ms` |
| `pq16_rerank_f32graph` (`R=200`) | `0.990340` | `0.98354` / `2,200.5` / `0.6039 ms` *(eff=200)* | `0.98354` / `2,228.2` / `0.5908 ms` *(eff=200)* | `0.98354` / `2,216.4` / `0.5989 ms` | `0.99034` / `933.0` / `1.4493 ms` |
| `pq16_rerank_f32graph` (`FullBeam`) | `0.996750` | `0.87045` / `6,837.1` / `0.1993 ms` | `0.95032` / `3,978.2` / `0.3356 ms` | `0.98354` / `2,211.3` / `0.5998 ms` | `0.99675` / `838.0` / `1.5768 ms` |
| `pq32_only_f32graph` | `0.723050` | `0.70823` / `3,546.0` / `2.0461 ms` | `0.71898` / `2,207.7` / `1.2880 ms` | `0.72231` / `1,268.6` / `2.6528 ms` | `0.72305` / `521.2` / `4.2303 ms` |
| `pq32_rerank_f32graph` (`R=100`) | `0.996940` | `0.97643` / `2,349.0` / `0.6021 ms` *(eff=100)* | `0.97643` / `2,342.1` / `0.6198 ms` | `0.99184` / `1,402.3` / `1.0307 ms` | `0.99694` / `569.3` / `2.7556 ms` |
| `pq32_rerank_f32graph` (`R=200`) | `0.997570` | `0.99230` / `1,326.1` / `1.2011 ms` *(eff=200)* | `0.99230` / `1,320.9` / `1.2189 ms` *(eff=200)* | `0.99230` / `1,334.3` / `1.1177 ms` | `0.99757` / `502.6` / `3.8811 ms` |
| `pq32_rerank_f32graph` (`FullBeam`) | **`0.997620`** | `0.93549` / `3,948.6` / `0.3600 ms` | `0.97643` / `2,338.8` / `0.5911 ms` | `0.99230` / `1,330.8` / `1.1139 ms` | `0.99762` / `477.7` / `4.2164 ms` |
| **Phase 4 `f32_m32`** *(solo ref)* | `0.999440` | `0.98278` / `2,579.7` / `0.5304 ms` | `0.99572` / `1,476.8` / `0.9404 ms` | `0.99881` / `842.2` / `1.6854 ms` | `0.99944` / `348.6` / `4.2463 ms` |

**Quantization-Inversion Saturation Dips (Without Sufficient Rerank):** On unreranked PQ, widening `efSearch` past saturation slightly lowers recall because extra beam depth admits vectors with marginally smaller PQ-ADC distance but larger true `f32` `L2Squared` distance (`100k` `pq16_only_f32graph`: `0.61519` @100 -> `0.61468` @600; `100k` `pq16_rerank R=30`: `0.89847` @300 -> `0.89840` @600; `100k` `pq32_only_f32graph`: `0.76461` @200 -> `0.76459` @300 -> `0.76461` @600; `1M` `pq16_only_f32graph`: `0.54520` @300 -> `0.54478` @600; `1M` `pq16_rerank R=30`: `0.82576` @400 -> `0.82559` @600). All `SQ8` arms (`M=16`, `M=32`, `QuantizedDirect`, `Rerank`) and all `PQ+Rerank` curves for `R >= 50` and `FullBeam` are strictly non-decreasing across all `12` `efSearch` points at both scales.

### 5.3 Matched-Recall Comparison (`0.95`, `0.98`, `0.99`, `0.995`) — `100k` & `1M`

Each cell reports the **lowest-`p99` observed grid point** meeting `Recall@10 >= target` (`(ef, R) | Recall@10 | QPS | p99`), followed by `[interp: QPS / p99]` from log-linear interpolation along the arm's `(p99, recall)` Pareto frontier (no extrapolation).

#### `100k` Matched-Recall Table (`11` Arms + Phase 4 `f32_m32`)

| `arm_id` | Target `>= 0.95` | Target `>= 0.98` | Target `>= 0.99` | Target `>= 0.995` |
| :--- | :--- | :--- | :--- | :--- |
| `f32_m16` | `ef=30` \| `0.95398` \| `9,742.1` \| `0.1511 ms`<br>`[interp: 10,070.3 QPS / 0.1470 ms]` | `ef=50` \| `0.98019` \| `6,523.3` \| `0.2200 ms`<br>`[interp: 6,548.2 QPS / 0.2191 ms]` | `ef=100` \| `0.99367` \| `3,738.9` \| `0.3802 ms`<br>`[interp: 4,684.3 QPS / 0.3010 ms]` | `ef=150` \| `0.99577` \| `2,681.0` \| `0.5259 ms`<br>`[interp: 3,028.7 QPS / 0.4669 ms]` |
| `sq8_only_f32graph` | `ef=40` \| `0.96434` \| `9,279.1` \| `0.1592 ms`<br>`[interp: 11,137.9 QPS / 0.1373 ms]` | `ef=70` \| `0.98056` \| `5,709.1` \| `0.2620 ms`<br>`[interp: 5,815.5 QPS / 0.2585 ms]` | *Not reached* (`max = 0.98778`) | *Not reached* (`max = 0.98778`) |
| `sq8_only_qbuild` | `ef=40` \| `0.96447` \| `9,226.2` \| `0.1598 ms`<br>`[interp: 10,973.3 QPS / 0.1407 ms]` | `ef=70` \| `0.98055` \| `5,910.1` \| `0.2487 ms`<br>`[interp: 6,019.7 QPS / 0.2442 ms]` | *Not reached* (`max = 0.98776`) | *Not reached* (`max = 0.98776`) |
| `sq8_rerank_f32graph` | `ef=30,R=20` \| `0.95410` \| `11,702.0` \| `0.1236 ms`<br>`[interp: 12,041.5 QPS / 0.1208 ms]` | `ef=50,R=20` \| `0.98022` \| `8,007.5` \| `0.1795 ms`<br>`[interp: 8,042.7 QPS / 0.1787 ms]` | `ef=100,R=20` \| `0.99370` \| `4,581.4` \| `0.3152 ms`<br>`[interp: 5,776.1 QPS / 0.2480 ms]` | `ef=150,R=20` \| `0.99586` \| `3,352.5` \| `0.4289 ms`<br>`[interp: 3,796.4 QPS / 0.3794 ms]` |
| `sq8_only_m32` | `ef=20` \| `0.96161` \| `10,366.7` \| `0.1485 ms`<br>`[interp: 11,253.7 QPS / 0.1392 ms]` | `ef=40` \| `0.98400` \| `6,280.9` \| `0.2396 ms`<br>`[interp: 7,244.9 QPS / 0.2066 ms]` | `ef=100` \| `0.99008` \| `3,087.0` \| `0.4931 ms`<br>`[interp: 3,161.3 QPS / 0.4814 ms]` | *Not reached* (`max = 0.99060`) |
| `sq8_rerank_m32` | `ef=10,R=20` *(eff=20)* \| `0.96819` \| `10,261.2` \| `0.1458 ms`<br>`[interp: 11,614.1 QPS / 0.1323 ms]` | `ef=30,R=20` \| `0.98563` \| `7,650.2` \| `0.1956 ms`<br>`[interp: 8,410.8 QPS / 0.1779 ms]` | `ef=40,R=20` \| `0.99247` \| `6,289.3` \| `0.2334 ms`<br>`[interp: 6,750.3 QPS / 0.2190 ms]` | `ef=50,R=20` \| `0.99554` \| `5,302.4` \| `0.2801 ms`<br>`[interp: 5,464.0 QPS / 0.2713 ms]` |
| `pq16_only_f32graph` | *Not reached* (`max = 0.61519`) | *Not reached* (`max = 0.61519`) | *Not reached* (`max = 0.61519`) | *Not reached* (`max = 0.61519`) |
| `pq16_only_qbuild` | *Not reached* (`max = 0.61613`) | *Not reached* (`max = 0.61613`) | *Not reached* (`max = 0.61613`) | *Not reached* (`max = 0.61613`) |
| `pq16_rerank_f32graph` | `ef=70,R=50` \| `0.95136` \| `10,024.4` \| `0.1277 ms`<br>`[interp: 10,356.7 QPS / 0.1235 ms]` | `ef=40,R=100` *(eff=100)* \| `0.98441` \| `7,291.7` \| `0.1729 ms`<br>`[interp: 7,886.6 QPS / 0.1616 ms]` | `ef=150,FB` \| `0.99274` \| `5,269.0` \| `0.2633 ms`<br>`[interp: 5,863.3 QPS / 0.2293 ms]` | `ef=30,R=200` *(eff=200)* \| `0.99520` \| `4,119.7` \| `0.3250 ms`<br>`[interp: 4,202.9 QPS / 0.3195 ms]` |
| `pq32_only_f32graph` | *Not reached* (`max = 0.76461`) | *Not reached* (`max = 0.76461`) | *Not reached* (`max = 0.76461`) | *Not reached* (`max = 0.76461`) |
| `pq32_rerank_f32graph` | `ef=40,FB` \| `0.96221` \| `9,899.0` \| `0.1424 ms`<br>`[interp: 10,894.4 QPS / 0.1296 ms]` | `ef=70,R=50` \| `0.98461` \| `7,106.6` \| `0.2013 ms`<br>`[interp: 7,816.0 QPS / 0.1784 ms]` | `ef=20,R=100` *(eff=100)* \| `0.99178` \| `5,433.5` \| `0.2451 ms`<br>`[interp: 5,755.6 QPS / 0.2353 ms]` | `ef=200,R=200` \| `0.99587` \| `3,213.9` \| `0.4121 ms`<br>`[interp: 3,908.0 QPS / 0.3675 ms]` |
| **Phase 4 `f32_m32`** *(ref)* | `ef=20` \| `0.96828` \| `8,249.4` \| `0.1851 ms`<br>`[interp: 9,612.5 QPS / 0.1622 ms]` | `ef=30` \| `0.98556` \| `6,139.6` \| `0.2542 ms`<br>`[interp: 6,751.8 QPS / 0.2295 ms]` | `ef=40` \| `0.99260` \| `5,010.0` \| `0.2974 ms`<br>`[interp: 5,400.7 QPS / 0.2807 ms]` | `ef=50` \| `0.99548` \| `4,279.1` \| `0.3412 ms`<br>`[interp: 4,393.1 QPS / 0.3335 ms]` |

#### `1M` Matched-Recall Table (`9` Arms + Phase 4 `f32_m32`)

| `arm_id` | Target `>= 0.95` | Target `>= 0.98` | Target `>= 0.99` | Target `>= 0.995` |
| :--- | :--- | :--- | :--- | :--- |
| `f32_m16` *(Phase 6)* | `ef=70` \| `0.96759` \| `2,249.2` \| `0.9941 ms`<br>`[interp: 2,807.0 QPS / 0.8440 ms]` | `ef=100` \| `0.98061` \| `1,676.2` \| `1.1951 ms`<br>`[interp: 1,699.4 QPS / 1.1848 ms]` | `ef=150` \| `0.99004` \| `1,223.5` \| `1.4282 ms`<br>`[interp: 1,225.1 QPS / 1.4272 ms]` | `ef=300` \| `0.99654` \| `663.4` \| `2.4524 ms`<br>`[interp: 804.4 QPS / 2.0920 ms]` |
| `sq8_only_f32graph` | `ef=70` \| `0.95967` \| `3,011.5` \| `0.4806 ms`<br>`[interp: 3,389.0 QPS / 0.4375 ms]` | `ef=150` \| `0.98058` \| `1,619.8` \| `0.8758 ms`<br>`[interp: 1,656.4 QPS / 0.8575 ms]` | *Not reached* (`max = 0.98764`) | *Not reached* (`max = 0.98764`) |
| `sq8_only_qbuild` | `ef=70` \| `0.95988` \| `2,883.6` \| `0.5038 ms`<br>`[interp: 3,286.2 QPS / 0.4464 ms]` | `ef=150` \| `0.98045` \| `1,534.5` \| `0.9358 ms`<br>`[interp: 1,562.8 QPS / 0.9192 ms]` | *Not reached* (`max = 0.98767`) | *Not reached* (`max = 0.98767`) |
| `sq8_rerank_f32graph` | `ef=70,FB` \| `0.96757` \| `2,788.5` \| `0.4959 ms`<br>`[interp: 3,508.7 QPS / 0.4021 ms]` | `ef=50,R=100` *(eff=100)* \| `0.98060` \| `2,183.2` \| `0.6141 ms`<br>`[interp: 2,208.0 QPS / 0.6081 ms]` | `ef=150,R=100` \| `0.99009` \| `1,579.3` \| `0.8742 ms`<br>`[interp: 1,584.1 QPS / 0.8713 ms]` | `ef=300,R=100` \| `0.99650` \| `895.0` \| `1.5476 ms`<br>`[interp: 1,074.8 QPS / 1.2796 ms]` |
| `sq8_only_m32` | `ef=30` \| `0.95294` \| `3,525.5` \| `0.4597 ms`<br>`[interp: 3,619.3 QPS / 0.4517 ms]` | `ef=70` \| `0.98135` \| `1,682.2` \| `1.6285 ms`<br>`[interp: 1,800.4 QPS / 1.3622 ms]` | *Not reached* (`max = 0.98898`) | *Not reached* (`max = 0.98898`) |
| `pq16_only_f32graph` | *Not reached* (`max = 0.54520`) | *Not reached* (`max = 0.54520`) | *Not reached* (`max = 0.54520`) | *Not reached* (`max = 0.54520`) |
| `pq16_rerank_f32graph` | `ef=100,FB` \| `0.95032` \| `3,978.2` \| `0.3356 ms`<br>`[interp: 3,989.1 QPS / 0.3347 ms]` | `ef=100,R=200` *(eff=200)* \| `0.98354` \| `2,228.2` \| `0.5908 ms`<br>`[interp: 2,435.4 QPS / 0.5415 ms]` | `ef=300,FB` \| `0.99180` \| `1,553.9` \| `0.8536 ms`<br>`[interp: 1,593.8 QPS / 0.8334 ms]` | `ef=600,FB` \| `0.99675` \| `838.0` \| `1.5768 ms`<br>`[interp: 1,152.1 QPS / 1.1495 ms]` |
| `pq32_only_f32graph` | *Not reached* (`max = 0.72305`) | *Not reached* (`max = 0.72305`) | *Not reached* (`max = 0.72305`) | *Not reached* (`max = 0.72305`) |
| `pq32_rerank_f32graph` | `ef=70,R=50` \| `0.95665` \| `3,392.3` \| `0.3954 ms`<br>`[interp: 3,631.0 QPS / 0.3710 ms]` | `ef=150,R=50` \| `0.98137` \| `1,936.4` \| `0.6929 ms`<br>`[interp: 2,040.5 QPS / 0.6630 ms]` | `ef=200,R=100` \| `0.99184` \| `1,402.3` \| `1.0307 ms`<br>`[interp: 1,522.6 QPS / 0.9508 ms]` | `ef=300,FB` \| `0.99561` \| `994.0` \| `1.4300 ms`<br>`[interp: 1,048.9 QPS / 1.3657 ms]` |
| **Phase 4 `f32_m32`** *(solo ref)* | `ef=30` \| `0.95989` \| `3,816.7` \| `0.3708 ms`<br>`[interp: 4,194.8 QPS / 0.3424 ms]` | `ef=50` \| `0.98278` \| `2,579.7` \| `0.5304 ms`<br>`[interp: 2,734.6 QPS / 0.5028 ms]` | `ef=70` \| `0.99087` \| `1,974.6` \| `0.6956 ms`<br>`[interp: 2,032.2 QPS / 0.6756 ms]` | `ef=100` \| `0.99572` \| `1,476.8` \| `0.9404 ms`<br>`[interp: 1,541.9 QPS / 0.8992 ms]` |

*(Note on `1M` `sq8_rerank_f32graph`: `R=20` and `R=100` yield identical recall for `ef >= 100`; the lowest-`p99` selector chose `R=100` at `>= 0.98`, `0.99`, and `0.995` because `R=20` experienced Windows `balanced`-plan tail jitter at `ef=100` [`1.2882 ms`] and `ef=300` [`2.3690 ms`].)*

---

## 6. Top-$R$ Rerank Recovery Ladders (`efSearch = 100` & `efSearch = 600`)

### 6.1 `100k` Top-$R$ Recovery (`f32_m16 ef=100`: `0.993670` @ `3,738.9 QPS`, `0.2672 ms` mean; `f32_m32 ef=100`: `0.999390` @ `2,541.9 QPS`)

| Quantizer Family | `R = None` (Only) | `R = 20` | `R = 30` | `R = 50` | `R = 100` | `R = 200` *(eff=200)* | `FullBeam (R=100)` | Ceiling `ef=600, FB` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`SQ8 (M=16)`** | `0.984760`<br>`4,234.9 QPS` (`0.2360 ms`) | **`0.993700`** (`+0.89 pp`)<br>`4,581.4 QPS` (`0.2182 ms`) | `0.993700`<br>`4,498.7 QPS` (`0.2222 ms`) | `0.993700`<br>`4,432.1 QPS` (`0.2255 ms`) | `0.993700`<br>`4,245.9 QPS` (`0.2354 ms`) | — | `0.993700`<br>`4,002.0 QPS` (`0.2498 ms`) | `0.997120` (`R=20` == `FB`) |
| **`SQ8 (M=32)`** | `0.990080`<br>`3,087.0 QPS` (`0.3238 ms`) | **`0.999380`** (`+0.93 pp`)<br>`3,097.4 QPS` (`0.3227 ms`) | `0.999380`<br>`3,118.5 QPS` (`0.3206 ms`) | `0.999380`<br>`3,098.1 QPS` (`0.3227 ms`) | `0.999380`<br>`2,906.0 QPS` (`0.3440 ms`) | — | `0.999380`<br>`2,931.9 QPS` (`0.3409 ms`) | `0.999990` (`R=20` == `FB`) |
| **`PQ16 (M=16)`** | `0.615190`<br>`8,261.9 QPS` (`0.1209 ms`) | — | `0.897350` (`+28.22 pp`)<br>`8,012.9 QPS` (`0.1247 ms`) | `0.955750` (`+34.06 pp`)<br>`7,913.0 QPS` (`0.1263 ms`) | `0.984410` (`+36.92 pp`)<br>`7,018.9 QPS` (`0.1424 ms`) | **`0.995200`** (`+38.00 pp`)<br>`4,109.4 QPS` (`0.2432 ms`) | `0.984410` (`+36.92 pp`)<br>`7,252.8 QPS` (`0.1378 ms`) | **`0.997260`** (`+38.26 pp` vs Only `0.61468`) |
| **`PQ32 (M=16)`** | `0.763960`<br>`6,337.5 QPS` (`0.1577 ms`) | — | `0.978050` (`+21.41 pp`)<br>`5,806.5 QPS` (`0.1721 ms`) | `0.989820` (`+22.59 pp`)<br>`5,789.2 QPS` (`0.1726 ms`) | `0.991780` (`+22.78 pp`)<br>`5,449.6 QPS` (`0.1834 ms`) | **`0.995870`** (`+23.19 pp`)<br>`3,144.9 QPS` (`0.3179 ms`) | `0.991780` (`+22.78 pp`)<br>`5,468.1 QPS` (`0.1828 ms`) | `0.996620` (`R=200` == `FB`) |

### 6.2 `1M` Top-$R$ Recovery (`f32_m16 ef=100`: `0.980610` @ `1,676.2 QPS`, `0.5962 ms` mean; Phase 4 solo `2,383.9 QPS`, `0.4193 ms`)

| Quantizer Family | `R = None` (Only) | `R = 20` | `R = 30` | `R = 50` | `R = 100` | `R = 200` *(eff=200)* | `FullBeam (R=100)` | Ceiling `ef=600, FB` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`SQ8 (M=16)`** | `0.971960`<br>`2,256.4 QPS` (`0.4431 ms`) | **`0.980600`** (`+0.86 pp`)<br>`2,005.4 QPS` (`0.4985 ms`) | — | — | `0.980600` (`+0.86 pp`)<br>`2,172.2 QPS` (`0.4603 ms`) | — | `0.980600` (`+0.86 pp`)<br>`2,067.8 QPS` (`0.4835 ms`) | `0.998040` (`R=20` == `FB`) |
| **`PQ16 (M=16)`** | `0.544290`<br>`4,181.3 QPS` (`0.2391 ms`) | — | `0.820270` (`+27.60 pp`)<br>`4,229.1 QPS` (`0.2363 ms`) | `0.898550` (`+35.43 pp`)<br>`4,155.4 QPS` (`0.2406 ms`) | `0.950320` (`+40.60 pp`)<br>`3,805.5 QPS` (`0.2627 ms`) | **`0.983540`** (`+43.92 pp`)<br>`2,228.2 QPS` (`0.4487 ms`) | `0.950320` (`+40.60 pp`)<br>`3,978.2 QPS` (`0.2513 ms`) | `0.996750` (`+45.20 pp` vs Only `0.54478`) |
| **`PQ32 (M=16)`** | `0.718980`<br>`2,207.7 QPS` (`0.4528 ms`) | — | `0.950370` (`+23.14 pp`)<br>`2,292.7 QPS` (`0.4361 ms`) | `0.971410` (`+25.24 pp`)<br>`2,632.4 QPS` (`0.3798 ms`) | `0.976430` (`+25.75 pp`)<br>`2,342.1 QPS` (`0.4269 ms`) | **`0.992300`** (`+27.33 pp`)<br>`1,320.9 QPS` (`0.7569 ms`) | `0.976430` (`+25.75 pp`)<br>`2,338.8 QPS` (`0.4274 ms`) | **`0.997620`** (`+27.46 pp` vs Only `0.72305`) |

**Key Rerank Insights:**
- **`R = k` Invariant Verified (`DESIGN.md` §3.3.2):** At `1M` `ef=10`, `sq8_rerank_f32graph (FullBeam, R=10)` records **`0.731130`** — identical to `sq8_only_f32graph` (`0.731130`) because reordering the same `10` IDs cannot change `Recall@10` set overlap; raising `R = 20` (`effective_ef = 20`) immediately lifts recall to `0.849110`.
- **Why `PQ16+Rerank` & `PQ32+Rerank` Beat `f32_m16`:** During graph traversal, `PQ16` reads `16 B/vec` (vs `512 B/vec` for `f32`, **32x less cache-line traffic**) with `9.83 ns` ADC lookups, touching raw `f32` vectors solely for the top $R$ survivors. Thus at `1M` `ef=100`, `pq16_rerank (R=200, eff=200)` achieves higher recall than `f32_m16 (ef=100)` (`0.98354` vs `0.98061`) while delivering **`+32.9%` higher QPS** (`2,228.2` vs `1,676.2`) and **`-50.6%` lower `p99`** (`0.5908` vs `1.1951 ms`).

---

## 7. `QuantizedDirect` vs `Float32GraphThenQuantize` Parity (`100k` & `1M`)

| Scale & Quantizer | `ef = 10` (f32g vs qbuild) | `ef = 50` (f32g vs qbuild) | `ef = 100` (f32g vs qbuild) | `ef = 200` (f32g vs qbuild) | `ef = 600` (f32g vs qbuild) | Full-Sweep $\Delta R@10$ Range |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100k` `SQ8` (`M=16`)** | `0.81899` vs `0.81919`<br>(`+0.00020`) | `0.97256` vs `0.97251`<br>(`-0.00005`) | `0.98476` vs `0.98464`<br>**(`-0.00012`)** | `0.98726` vs `0.98724`<br>(`-0.00002`) | `0.98778` vs `0.98776`<br>**(`-0.00002`)** | `-0.00012` .. `+0.00066` |
| **`1M` `SQ8` (`M=16`)** | `0.73113` vs `0.73123`<br>(`+0.00010`) | `0.94028` vs `0.94052`<br>(`+0.00024`) | `0.97196` vs `0.97220`<br>**(`+0.00024`)** | `0.98396` vs `0.98397`<br>(`+0.00001`) | `0.98764` vs `0.98767`<br>**(`+0.00003`)** | `-0.00013` .. `+0.00156` |
| **`100k` `PQ16` (`M=16`)** | `0.56952` vs `0.56645`<br>(`-0.00307`) | `0.61397` vs `0.61305`<br>(`-0.00092`) | `0.61519` vs `0.61524`<br>(`+0.00005`) | `0.61484` vs `0.61581`<br>(`+0.00097`) | `0.61468` vs `0.61613`<br>(`+0.00145`) | `-0.00307` .. `+0.00145` |

Building the HNSW graph directly on `SQ8` symmetric `u8`-vs-`u8` distances preserves recall across all operating points while cutting `build_peak_rss_mb` by `44.2%` (`100k`) and **`64.4%`** (`1M`: `778.37 -> 277.29 MiB`). Notice at `100k` `PQ16`, `pq16_only_qbuild` (`0x7328beaf5527e6d4`) builds a faster/shallower graph (`18.94 s` vs `27.16 s`) and monotonically climbs to `0.61613` at `ef=600` without the quantization-inversion dip (`0.61519 -> 0.61468`), because building and searching in the same PQ distance space aligns graph edge diversity with query-time ordering.

---

## 8. Memory-Recall Pareto Frontier (`p99 <= 1.0 ms`) & Visualization Index

### 8.1 `p99 <= 1.0 ms` Pareto Frontier Summary (`Chart 3`)

- **`100k` Payload Frontier (`B/vec`):** `pq16_only_qbuild` (`161.84 B`, `0.61613`) $\to$ `pq32_only_f32graph` (`177.84 B`, `0.76461`) $\to$ `sq8_only_f32graph` (`272.55 B`, `0.98767`) $\to$ `sq8_only_m32` (`400.22 B`, `0.99053`) $\to$ `f32_m16` (`657.57 B`, `0.99690`) $\to$ `pq16_rerank_f32graph` (`673.84 B`, `0.99726`) $\to$ Phase 4 `f32_m32` (`785.41 B`, `0.99991`). *(On Query RSS: `pq16_only_f32graph` `28.13 MiB`, `0.61519` starts the frontier just ahead of `pq16_only_qbuild` `28.32 MiB`, `0.61613`, ending at `f32_m32` `92.59 MiB`, `0.99991`.)*
- **`1M` Payload Frontier (`B/vec`):** `pq16_only_f32graph` (`160.67 B`, `0.54520`) $\to$ `pq32_only_f32graph` (`176.67 B`, `0.70138`) $\to$ `sq8_only_f32graph` (`272.54 B`, `0.98058`) $\to$ `pq16_rerank_f32graph` (`672.67 B`, `0.99180`) $\to$ Phase 4 `f32_m32` (`784.33 B`, `0.99572`). *(On Query RSS: `pq16_only` `169.55 MiB` $\to$ `pq32_only` `184.67 MiB` $\to$ `sq8_only` `276.02 MiB` $\to$ Phase 4 `f32_m32` `767.40 MiB`, `0.99572`, while `sq8_only_m32` dropped off the `p99 <= 1.0 ms` `1M` frontier due to its `ef >= 70` `balanced`-mode tail spike capping its eligible point at `ef=50`, `0.97409`.)*

### 8.2 Charts (`scripts/plot_phase6.py`)

All 4 charts exist in 300 DPI PNG and deterministic SVG across [`charts_100k/`](./charts_100k/) (`100k`), [`charts_1m/`](./charts_1m/) (`1M`), and root [`results/phase6/`](./) (identical copies of `charts_1m/`):

| Chart | `100k` (`charts_100k/`) | `1M` (`charts_1m/` & Root `results/phase6/`) | Description |
| :--- | :--- | :--- | :--- |
| **Chart 1** | [`charts_100k/chart1_recall_vs_p99.png`](./charts_100k/chart1_recall_vs_p99.png) ([SVG](./charts_100k/chart1_recall_vs_p99.svg)) | [`charts_1m/chart1_recall_vs_p99.png`](./charts_1m/chart1_recall_vs_p99.png) ([root PNG](./chart1_recall_vs_p99.png), [SVG](./charts_1m/chart1_recall_vs_p99.svg)) | `Recall@10` vs `p99` query latency (ms, log y-axis) with `1 ms` SLA line |
| **Chart 2** | [`charts_100k/chart2_recall_vs_qps.png`](./charts_100k/chart2_recall_vs_qps.png) ([SVG](./charts_100k/chart2_recall_vs_qps.svg)) | [`charts_1m/chart2_recall_vs_qps.png`](./charts_1m/chart2_recall_vs_qps.png) ([root PNG](./chart2_recall_vs_qps.png), [SVG](./charts_1m/chart2_recall_vs_qps.svg)) | `Recall@10` vs single-thread QPS (log y-axis) |
| **Chart 3** | [`charts_100k/chart3_memory_vs_recall.png`](./charts_100k/chart3_memory_vs_recall.png) ([SVG](./charts_100k/chart3_memory_vs_recall.svg)) | [`charts_1m/chart3_memory_vs_recall.png`](./charts_1m/chart3_memory_vs_recall.png) ([root PNG](./chart3_memory_vs_recall.png), [SVG](./charts_1m/chart3_memory_vs_recall.svg)) | Analytic payload (`B/vec`) & OS query-serving Peak RSS (`MiB`) vs Recall (`p99 <= 1 ms`) + Pareto frontier |
| **Chart 4** | [`charts_100k/chart4_rerank_ablation.png`](./charts_100k/chart4_rerank_ablation.png) ([SVG](./charts_100k/chart4_rerank_ablation.svg)) | [`charts_1m/chart4_rerank_ablation.png`](./charts_1m/chart4_rerank_ablation.png) ([root PNG](./chart4_rerank_ablation.png), [SVG](./charts_1m/chart4_rerank_ablation.svg)) | Top-$R$ rerank recovery & mean latency overhead vs $R$ at `efSearch = 100` |
