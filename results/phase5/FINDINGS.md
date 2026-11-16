# Phase 5 Filtered Vector Search: Empirical Findings & Crossover Synthesis

**Tasks:** Milestone M3 (`T3.6`–`T3.11`, Gate **G4.5** Authorized Execution)  
**Index Under Test:** Deterministic `HnswIndex` (`M = 16`, `efConstruction = 100`, `seed = 42`, `dim = 128`, `L2Squared`, `k = 10`)  
**Dataset:** Authentic INRIA Texmex `SIFT1M` (`100k` prefix and full `1M` base, `10,000` queries)  
**Host Hardware:** `12th Gen Intel(R) Core(TM) i7-12650H` (10 physical cores / 6 P-cores + 4 E-cores, 16 logical threads, `15.78 GiB` RAM, Windows `x86_64`, `balanced` power plan)  
**Compiler:** `rustc 1.99.1 (58cf8f9c3 2026-10-13)`  
**Clean Anchored Commits:**
- `T3.6` microbenchmark & `T3.7` pilot: `84e3c81` (`git_dirty: false`)
- `100k` Option C matrix (`24` cells) + `sensitivity_n100000.json`: `a4621bc` (`git_dirty: false`, pinned solo to P-core `0x0004`)
- `1M` Option C matrix (`24` cells) + `sensitivity_n1000000.json`: `c39c92b` (`git_dirty: false`, all `is_authentic_sift: true`)

---

## 1. Execution Protocol & Provenance Disclosure

All runs obey the approved **Gate G4.5** parameters:
1. **Option C Schedule (`reps = 5` after 1 untimed warmup):** Every cell and arm explicitly records `num_queries` and `is_subquery_probe` (`S3_pre_filter`: `10,000` queries at `100k` & `1M` $\sigma \le 0.05$, `2,000` at `1M` $\sigma \ge 0.075$; HNSW `S2`/`S1` arms: `300` at $\sigma \le 0.01$, `1,000` at $\sigma \in [0.02, 0.10]$, `2,000` at $\sigma \ge 0.20$).
2. **`S1_post_uncapped` (`S1_uncap_ef100`) Scope:** Evaluated **only** for $\sigma \ge 0.05$ in the full matrix (its low-selectivity $\le 1\%$ starvation and runaway traversal latency were measured in [`pilot_100k.json`](./pilot/pilot_100k.json)).
3. **Opaque `eval_limit` Guardrail:** `FilteredSearchParams::default().eval_limit` stays `None` for backwards API parity; `DEFAULT_OPAQUE_EVAL_LIMIT = 16_384` (derived from the `100k` pilot opaque evaluation distribution: $4 \times K'_{\max} = 4 \times 4,096$) is exposed via `FilteredSearchParams::with_opaque_eval_limit()`, and tested across `{None, 16_384, 10_000}` directly at both `100k` and `1M`. Note bitset predicates never hit `eval_limit`.
4. **Concurrency & Pinning Transparency:**
   - The `100k` matrix (`24` cells + sensitivity) ran **solo** pinned to physical P-core 1 (`processor_affinity_mask = "0x0004"`). Peak RSS: `127.32 MiB`; build time: `27.365 s` (`3,654.3 vec/s`, fingerprint `0x80af03bd7aebaade`).
   - At `1M`, `independent` $\sigma \in \{0.0005, 0.001, 0.002\}$ (fingerprint `0x44044886c62bca5e`, `606.488 s` build / `1,648.84 vec/s`, `1,015.69 MiB` peak RSS) and `sensitivity_n1000000.json` *(omits `processor_affinity_mask`/`peak_rss_mb` keys in its header, though launched pinned to `0x0004`)* ran solo on `0x0004`. To complete the remaining `21` `1M` cells within schedule, those cells executed in sharded batches across `4` separate physical P-cores (`0x0004`, `0x0010`, `0x0040`, `0x0100`, up to `4` processes / `~4.1 GiB` peak RSS leaving `~1.4 GiB` free headroom out of `5.36 GiB` available RAM; note `independent` `0.005..0.035` carry `0x0004` as one shard of that 4-core run). Exact P-core affinity is stamped in every cell JSON (`provenance.processor_affinity_mask`). Parallel `1M` DRAM/L3 bus sharing increases `S3` high-$\sigma$ linear-scan latency slightly relative to a fully idle memory bus.

---

## 2. Recall-Matched Crossover Results (`S2_in_traversal` vs `S3_pre_filter`)

Under the recall-matched crossover rule ([`DESIGN.md`](./DESIGN.md) §4.3), at each $\sigma$ we pick the lowest-latency `S2` depth (`efSearch` $\in \{20, 50, 100, 200, 400, 600\}$) achieving `Recall@10 >= target` and compare its mean latency against exact brute-force pre-filtering (`S3_pre_filter`, guaranteed `Recall@10 = 1.000000` and `0.000%` under-fill rate everywhere):

| Scale ($N$) | Mode | Recall Target | Empirical Bracket $[\sigma_{\text{lo}}, \sigma_{\text{hi}}]$ | Log-Log Crossover $\sigma^*$ | `S2` Depth at Bracket `[lo -> hi]` | Source Summary |
|---:|---|---:|---|---|---:|---|
| **`100,000`** | Independent (Mode A) | `0.95` | `[0.075000, 0.100000]` | **`0.079149`** | `ef=20 -> ef=20` | [`summary_n100000.json`](./summary_n100000.json) |
| **`100,000`** | Independent (Mode A) | `0.99` | `[0.075000, 0.100000]` | **`0.079149`** | `ef=20 -> ef=20` | [`summary_n100000.json`](./summary_n100000.json) |
| **`100,000`** | Correlated (Mode B) | `0.95` | `[0.200000, 0.500000]` | **`0.252372`** | `ef=400 -> ef=50` | [`summary_n100000.json`](./summary_n100000.json) |
| **`100,000`** | Correlated (Mode B) | `0.99` | `[0.200000, 0.500000]` | `null` *(not reached at `0.20`)* | `null -> ef=100` | [`summary_n100000.json`](./summary_n100000.json) |
| **`1,000,000`** | Independent (Mode A) | `0.95` | `[0.020000, 0.035000]` | **`0.023799`** | `ef=20 -> ef=20` | [`summary_n1000000.json`](./summary_n1000000.json) |
| **`1,000,000`** | Independent (Mode A) | `0.99` | `[0.020000, 0.035000]` | **`0.023799`** | `ef=20 -> ef=20` | [`summary_n1000000.json`](./summary_n1000000.json) |
| **`1,000,000`** | Correlated (Mode B) | `0.95` | `[0.100000, 0.200000]` | **`0.120868`** | `ef=400 -> ef=100` | [`summary_n1000000.json`](./summary_n1000000.json) |
| **`1,000,000`** | Correlated (Mode B) | `0.99` | `[0.100000, 0.200000]` | `null` *(not reached at `0.10`)* | `null -> ef=400` | [`summary_n1000000.json`](./summary_n1000000.json) |

### Key Scaling Law (`100k` $\to$ `1M` Leftward Shift of $\sigma^*$)

As corpus size grows $10\times$ (`100k -> 1M`), exact pre-filtering (`S3`) scales linearly in matching count $\mathcal{O}(N\sigma)$ while out-of-L3 DRAM pressure compounds per-vector scan latency, whereas HNSW filtered traversal grows sublinearly:
- **Mode A (`independent`) crossover shifts left by $3.33\times$:** $\sigma^*_{0.95} = 0.079149 \to \mathbf{0.023799}$. At `1M` $\sigma = 0.020$, `S3` (`272.97 QPS`, `3.663 ms`) still edges out `S2_ef20` (`216.77 QPS`, `4.612 ms`); by $\sigma = 0.035$, `S2_ef20` (`92.37 QPS`, `10.824 ms`, `R@10 = 0.9907`) overtakes `S3` (`55.38 QPS`, `18.053 ms`) and by $\sigma = 0.50$, `S2_ef50` reaches `1,328.34 QPS` (`R@10 = 0.9690`) vs `15.84 QPS` for `S3` ($83.9\times$ speedup).
- **Mode B (`correlated`) crossover shifts left by $2.09\times$:** $\sigma^*_{0.95} = 0.252372 \to \mathbf{0.120868}$. At `1M` $\sigma = 0.100$, `S3` (`26.79 QPS`, `37.330 ms`) still beats `S2_ef600` (`16.09 QPS`, `62.139 ms`, `R@10 = 0.9676`); by $\sigma = 0.200$, `S2_ef100` (`132.70 QPS`, `7.536 ms`, `R@10 = 0.9598`) decisively beats `S3` (`31.15 QPS`, `32.101 ms`).

---

## 3. Theoretical Cost Model (`T3.6`) vs Observed `S3_pre_filter` (`T3.9`)

The `T3.6` microbenchmark ([`microbench_cost.json`](./pilot/microbench_cost.json), `84e3c81`) measured L1/cache-hot single-core primitives:
- $t_{\text{word}} = 0.3042\text{ ns/word}$
- $t_{\text{dist}}(128) = 21.0420\text{ ns/dist}$
- $t_{\text{heap}}(10) = 2.8360\text{ ns/op} \implies t_{\text{cand}} = 23.8780\text{ ns/match}$

Prediction formula:
$$C_{\text{brute}}(\sigma, N, 128) = \frac{N}{64}\, t_{\text{word}} + \sigma N \cdot t_{\text{cand}}$$

Comparing predicted vs measured `S3_pre_filter` mean latency at representative selectivities reveals why real-world crossover occurs *earlier* (lower $\sigma^*$) than naive L1-hot extrapolation predicts:

| Scale & Mode | $\sigma$ | Matches | T3.6 Predicted (ms) | Measured `S3` Mean (ms) | Empirical / Predicted | Physical Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| `100k` Ind | `0.001` | `100` | `0.0029` | `0.0066` | `2.31x` | Non-contiguous bitset-ID jumps miss `48.8 MiB` L3-overflow buffer |
| `100k` Ind | `0.010` | `1,000` | `0.0244` | `0.0422` | `1.73x` | Strided access partially prefetched |
| `100k` Ind | `0.100` | `10,000` | `0.2393` | `0.6763` | `2.83x` | Base array (`48.8 MiB`) > `24 MiB` shared L3 cache |
| `100k` Ind | `0.500` | `50,000` | `1.1944` | `3.5570` | `2.98x` | Sustained DRAM read bandwidth |
| `1M` Ind | `0.001` | `1,000` | `0.0286` | `0.0710` | `2.48x` | TLB & DRAM row-buffer misses in `488.3 MiB` base matrix (`0x0004` solo) |
| `1M` Ind | `0.010` | `10,000` | `0.2435` | `1.1697` | `4.80x` | Full `488.3 MiB` working set (`~20x` L3); concurrent `w1` run |
| `1M` Ind | `0.100` | `100,000` | `2.3926` | `18.6933` | `7.81x` | Full DRAM saturation under `4`-way concurrent P-core runs (`0x0010`) |
| `1M` Ind | `0.500` | `500,000` | `11.9438` | `63.1197` | `5.28x` | Every second vector streamed from DRAM (`0x0010`) |

### 3.1 Graph Traversal Work Scaling ($N_{\text{eval}} \propto \sigma^{-\beta}$, `S2_ef100`)

Fitting `log(dist_evals_mean) = log(A) - beta * log(sigma)` over the active beam-search regime ([`DESIGN.md`](./DESIGN.md) §4.2, $\sigma \ge 0.005$ in Mode A, $\sigma \ge 0.020$ in Mode B, `ef_search = 100`, `M = 16`, reference `ef * M = 1,600`):

| Scale ($N$) | Mode | Fitted Prefactor $A$ | Exponent $\beta$ | $R^2$ | Interpretation |
|---:|---|---:|---:|---:|---|
| `100,000` | Mode A (`independent`) | `1,836.9` | **`0.6000`** | **`0.9934`** | Tight power law near $1.15 \cdot (\text{ef} \cdot M)$; sub-linear $\sigma^{-0.60}$ penalty from bridge filtering |
| `1,000,000` | Mode A (`independent`) | `2,099.3` | **`0.7403`** | **`0.9991`** | Steeper exponent on `1M` graph ($\beta$ climbs from `0.60 -> 0.74`) |
| `100,000` | Mode B (`correlated`) | `3,553.1` | `0.4119` | `0.5025` | Weak power-law fit due to Voronoi boundary stalling / `<k` early termination |
| `1,000,000` | Mode B (`correlated`) | `6,030.7` | `0.5157` | `0.5191` | Non-monotonic cluster isolation breaks clean $\sigma^{-\beta}$ scaling |

---

## 4. Full Cell Tables (`100k` and `1M`, Mode A & Mode B)

### 4.1 Scale $N = 100,000$ — Mode A (`independent`, Solo P-Core `0x0004`)

| $\sigma$ | Matches | `S3` QPS / Mean (ms) | `S2 ef=100` R@10 / QPS / `<k`% | Best `S2 >= 0.95` R@10 / QPS (`ef`) | `S1_cap4096 ef=100` R@10 / QPS / Cap% | `S1_uncap ef=100` R@10 / QPS |
|---:|---:|---|---|---|---|:---|
| `0.0005` | `50` | `224,986.6` / `0.0044` | `0.2677` / `514.1` / `96.3%` | *No hit* (max `0.8430`, `ef600`) | `0.2100` / `193.2` / `100.0%` | *Skipped (`<0.05`)* |
| `0.0010` | `100` | `148,853.2` / `0.0066` | `0.5757` / `128.9` / `70.0%` | `0.9733` / `46.7` (`ef400`) | `0.4030` / `194.0` / `99.3%` | *Skipped (`<0.05`)* |
| `0.0020` | `200` | `95,882.9` / `0.0103` | `0.8883` / `55.5` / `17.3%` | `0.9677` / `46.8` (`ef200`) | `0.7463` / `193.2` / `76.7%` | *Skipped (`<0.05`)* |
| `0.0050` | `500` | `45,820.3` / `0.0217` | `0.9990` / `78.6` / `0.3%` | `0.9680` / `132.1` (`ef50`) | `0.9990` / `197.9` / `0.7%` | *Skipped (`<0.05`)* |
| `0.0100` | `1,000` | `23,644.2` / `0.0422` | `0.9997` / `122.0` / `0.0%` | `0.9717` / `396.4` (`ef20`) | `0.9997` / `337.2` / `0.0%` | *Skipped (`<0.05`)* |
| `0.0200` | `2,000` | `8,102.3` / `0.1233` | `0.9990` / `200.2` / `0.0%` | `0.9954` / `649.0` (`ef20`) | `0.9979` / `585.1` / `0.0%` | *Skipped (`<0.05`)* |
| `0.0350` | `3,500` | `4,026.0` / `0.2482` | `0.9988` / `304.2` / `0.0%` | `0.9978` / `1,007.4` (`ef20`) | `0.9976` / `902.3` / `0.0%` | *Skipped (`<0.05`)* |
| `0.0500` | `5,000` | `2,904.7` / `0.3441` | `0.9986` / `393.0` / `0.0%` | `0.9965` / `1,334.3` (`ef20`) | `0.9964` / `1,190.0` / `0.0%` | `0.9964` / `1,200.8` |
| `0.0750` | `7,500` | **`1,988.3`** / `0.5028` | `0.9986` / `530.7` / `0.0%` | `0.9948` / **`1,803.2`** (`ef20`) | `0.9939` / `1,673.8` / `0.0%` | `0.9939` / `1,657.5` |
| `0.1000` | `10,000` | `1,478.2` / `0.6763` | `0.9988` / `648.5` / `0.0%` | `0.9932` / **`2,259.2`** (`ef20`) | `0.9920` / `2,096.7` / `0.0%` | `0.9920` / `2,105.0` |
| `0.2000` | `20,000` | `534.4` / `1.8709` | `0.9972` / `1,071.2` / `0.0%` | `0.9819` / `3,808.1` (`ef20`) | `0.9826` / `3,569.8` / `0.0%` | `0.9826` / `3,504.6` |
| `0.5000` | `50,000` | `281.1` / `3.5570` | `0.9964` / `2,060.4` / `0.0%` | `0.9557` / `7,536.3` (`ef20`) | `0.9910` / `3,610.7` / `0.0%` | `0.9910` / `3,539.5` |

### 4.2 Scale $N = 100,000$ — Mode B (`correlated`, Solo P-Core `0x0004`)

| $\sigma$ | Matches ($C$) | `S3` QPS / Mean (ms) | `S2 ef=100` R@10 / QPS / `<k`% | Best `S2 >= 0.95` R@10 / QPS (`ef`) | `S1_cap4096 ef=100` R@10 / QPS / Cap% | `S1_uncap ef=100` R@10 / QPS / `<k`% |
|---:|---:|---|---|---|---|---|
| `0.0005` | `50` (`4000`) | `224,825.7` / `0.0044` | `0.0310` / `847.1` / `97.7%` | *No hit* (max `0.0870`, `ef600`) | `0.0117` / `193.4` / `99.3%` | *Skipped (`<0.05`)* |
| `0.0010` | `100` (`2000`) | `145,230.6` / `0.0068` | `0.2653` / `204.9` / `76.7%` | *No hit* (max `0.4703`, `ef600`) | `0.2123` / `194.1` / `88.3%` | *Skipped (`<0.05`)* |
| `0.0020` | `200` (`1000`) | `91,131.0` / `0.0109` | `0.2493` / `367.5` / `76.0%` | *No hit* (max `0.3990`, `ef600`) | `0.1850` / `193.9` / `85.7%` | *Skipped (`<0.05`)* |
| `0.0050` | `500` (`400`) | `46,112.5` / `0.0216` | `0.3740` / `419.8` / `63.0%` | *No hit* (max `0.4567`, `ef600`) | `0.3540` / `120.0` / `66.0%` | *Skipped (`<0.05`)* |
| `0.0100` | `1,000` (`200`) | `24,239.1` / `0.0411` | `0.3437` / `315.8` / `66.3%` | *No hit* (max `0.5093`, `ef600`) | `0.2930` / `145.2` / `74.7%` | *Skipped (`<0.05`)* |
| `0.0200` | `2,000` (`100`) | `8,248.1` / `0.1211` | `0.5053` / `375.8` / `53.0%` | *No hit* (max `0.7555`, `ef600`) | `0.4738` / `147.5` / `56.0%` | *Skipped (`<0.05`)* |
| `0.0350` | `3,500` (`64`) | `3,945.3` / `0.2533` | `0.4413` / `222.8` / `58.2%` | *No hit* (max `0.6815`, `ef600`) | `0.3459` / `95.7` / `70.1%` | *Skipped (`<0.05`)* |
| `0.0500` | `5,000` (`64`) | `2,750.7` / `0.3634` | `0.4683` / `286.4` / `54.9%` | *No hit* (max `0.5588`, `ef600`) | `0.3983` / `90.8` / `65.1%` | `0.7234` / `16.1` / `47.3%` |
| `0.0750` | `7,500` (`64`) | `1,957.9` / `0.5105` | `0.7431` / `170.4` / `26.9%` | `0.9786` / `67.7` (`ef600`) | `0.5912` / `114.0` / `49.4%` | `0.9939` / `62.3` / `0.0%` |
| `0.1000` | `10,000` (`64`) | `1,312.1` / `0.7619` | `0.9119` / `313.0` / `8.8%` | `0.9669` / `143.0` (`ef400`) | `0.8903` / `252.7` / `13.9%` | `0.9940` / `174.2` / `0.0%` |
| `0.2000` | `20,000` (`64`) | **`546.4`** / `1.8299` | `0.8962` / `384.7` / `10.1%` | `0.9645` / **`172.7`** (`ef400`) | `0.8683` / `268.7` / `15.1%` | `0.9839` / `153.5` / `0.0%` |
| `0.5000` | `50,000` (`64`) | `107.5` / `9.3029` | `0.9951` / `1,951.7` / `0.0%` | `0.9872` / **`3,174.1`** (`ef50`) | `0.9881` / `2,869.3` / `0.0%` | `0.9881` / `2,675.5` / `0.0%` |

### 4.3 Scale $N = 1,000,000$ — Mode A (`independent`, P-Cores `0x0004` / `0x0010`)

| $\sigma$ | Matches | Aff. | `S3` QPS / Mean (ms) | `S2 ef=100` R@10 / QPS / `<k`% | Best `S2 >= 0.95` R@10 / QPS (`ef`) | `S1_cap4096 ef=100` R@10 / QPS / Cap% | `S1_uncap ef=100` R@10 / QPS |
|---:|---:|---|---|---|---|:---|:---|
| `0.0005` | `500` | `0x0004` | `26,924.3` / `0.0370` | `0.3930` / `24.7` / `82.7%` | `0.9670` / `2.7` (`ef600`) | `0.2110` / `65.8` / `100.0%` | *Skipped (`<0.05`)* |
| `0.0010` | `1,000` | `0x0004` | `14,049.9` / `0.0710` | `0.7753` / `11.0` / `34.0%` | `0.9567` / `3.5` (`ef200`) | `0.4220` / `50.3` / `98.7%` | *Skipped (`<0.05`)* |
| `0.0020` | `2,000` | `0x0004` | `3,964.2` / `0.2518` | `0.9653` / `4.7` / `6.0%` | `0.9653` / `4.7` (`ef100`) | `0.7823` / `98.2` / `66.3%` | *Skipped (`<0.05`)* |
| `0.0050` | `5,000` | `0x0004` | `2,109.7` / `0.4738` | `0.9987` / `24.3` / `0.3%` | `0.9963` / `41.5` (`ef50`) | `0.9987` / `82.6` / `0.3%` | *Skipped (`<0.05`)* |
| `0.0100` | `10,000` | `0x0004` | `854.7` / `1.1697` | `1.0000` / `42.0` / `0.0%` | `0.9953` / `157.1` (`ef20`) | `0.9977` / `139.0` / `0.0%` | *Skipped (`<0.05`)* |
| `0.0200` | `20,000` | `0x0004` | **`273.0`** / `3.6625` | `0.9999` / `50.9` / `0.0%` | `0.9950` / **`216.8`** (`ef20`) | `0.9940` / `185.0` / `0.0%` | *Skipped (`<0.05`)* |
| `0.0350` | `35,000` | `0x0004` | **`55.4`** / `18.0527` | `0.9996` / `39.7` / `0.0%` | `0.9907` / **`92.4`** (`ef20`) | `0.9904` / `314.5` / `0.0%` | *Skipped (`<0.05`)* |
| `0.0500` | `50,000` | `0x0010` | `117.4` / `8.5209` | `0.9998` / `154.2` / `0.0%` | `0.9880` / `561.7` (`ef20`) | `0.9877` / `525.1` / `0.0%` | `0.9877` / `520.5` |
| `0.0750` | `75,000` | `0x0010` | `76.3` / `13.1012` | `0.9992` / `188.4` / `0.0%` | `0.9813` / `783.8` (`ef20`) | `0.9798` / `739.1` / `0.0%` | `0.9798` / `760.2` |
| `0.1000` | `100,000` | `0x0010` | `53.5` / `18.6933` | `0.9989` / `217.2` / `0.0%` | `0.9727` / `851.0` (`ef20`) | `0.9694` / `247.6` / `0.0%` | `0.9694` / `254.4` |
| `0.2000` | `200,000` | `0x0010` | `12.8` / `77.8529` | `0.9971` / `248.6` / `0.0%` | `0.9898` / `503.3` (`ef50`) | `0.9465` / `797.3` / `0.0%` | `0.9465` / `593.8` |
| `0.5000` | `500,000` | `0x0010` | `15.8` / `63.1197` | `0.9906` / `749.0` / `0.0%` | `0.9690` / `1,328.3` (`ef50`) | `0.9694` / `1,192.9` / `0.0%` | `0.9694` / `1,630.4` |

### 4.4 Scale $N = 1,000,000$ — Mode B (`correlated`, P-Cores `0x0040` / `0x0100` / `0x0010`)

| $\sigma$ | Matches ($C$) | Aff. | `S3` QPS / Mean (ms) | `S2 ef=100` R@10 / QPS / `<k`% | Best `S2 >= 0.95` R@10 / QPS (`ef`) | `S1_cap4096 ef=100` R@10 / QPS / Cap% | `S1_uncap ef=100` R@10 / QPS / `<k`% |
|---:|---:|---|---:|---|---|---|---|
| `0.0005` | `500` (`4000`) | `0x0040` | `26,721.5` / `0.0373` | `0.0847` / `161.5` / `92.3%` | *No hit* (max `0.1643`, `ef600`) | `0.0483` / `79.1` / `96.3%` | *Skipped (`<0.05`)* |
| `0.0010` | `1,000` (`2000`) | `0x0040` | `15,354.3` / `0.0650` | `0.1540` / `107.7` / `85.3%` | *No hit* (max `0.2707`, `ef600`) | `0.1023` / `79.0` / `92.0%` | *Skipped (`<0.05`)* |
| `0.0020` | `2,000` (`1000`) | `0x0040` | `5,646.8` / `0.1769` | `0.3153` / `68.5` / `69.7%` | *No hit* (max `0.4553`, `ef600`) | `0.2270` / `79.4` / `83.3%` | *Skipped (`<0.05`)* |
| `0.0050` | `5,000` (`400`) | `0x0040` | `2,197.9` / `0.4548` | `0.1277` / `167.1` / `88.0%` | *No hit* (max `0.2470`, `ef600`) | `0.0870` / `38.7` / `93.0%` | *Skipped (`<0.05`)* |
| `0.0100` | `10,000` (`200`) | `0x0040` | `756.3` / `1.3218` | `0.2923` / `103.1` / `71.3%` | *No hit* (max `0.3873`, `ef600`) | `0.2197` / `54.9` / `79.7%` | *Skipped (`<0.05`)* |
| `0.0200` | `20,000` (`100`) | `0x0040` | `234.1` / `4.2717` | `0.4062` / `84.3` / `59.6%` | *No hit* (max `0.4434`, `ef600`) | `0.3441` / `21.8` / `68.2%` | *Skipped (`<0.05`)* |
| `0.0350` | `35,000` (`64`) | `0x0040` | `73.1` / `13.6712` | `0.3589` / `26.0` / `64.4%` | *No hit* (max `0.4427`, `ef600`) | `0.2723` / `28.5` / `74.9%` | *Skipped (`<0.05`)* |
| `0.0500` | `50,000` (`64`) | `0x0040` | `131.5` / `7.6039` | `0.5863` / `54.0` / `42.7%` | *No hit* (max `0.7860`, `ef600`) | `0.4468` / `59.5` / `60.5%` | `0.9938` / `5.5` / `0.0%` |
| `0.0750` | `75,000` (`64`) | `0x0100` | `88.5` / `11.3033` | `0.6643` / `67.9` / `34.2%` | *No hit* (max `0.8456`, `ef600`) | `0.5539` / `52.9` / `47.3%` | `0.9880` / `3.2` / `0.0%` |
| `0.1000` | `100,000` (`64`) | `0x0100` | **`26.8`** / `37.3296` | `0.9131` / `50.8` / `8.7%` | `0.9676` / **`16.1`** (`ef600`) | `0.8379` / `91.0` / `17.8%` | `0.9809` / `36.5` / `0.0%` |
| `0.2000` | `200,000` (`64`) | `0x0100` | **`31.2`** / `32.1009` | `0.9598` / `132.7` / `3.5%` | `0.9598` / **`132.7`** (`ef100`) | `0.8749` / `141.7` / `12.4%` | `0.9664` / `103.2` / `0.0%` |
| `0.5000` | `500,000` (`64`) | `0x0010` | `20.6` / `48.6461` | `0.9882` / `735.3` / `0.0%` | `0.9615` / `1,208.0` (`ef50`) | `0.9635` / `1,047.8` / `0.0%` | `0.9635` / `1,030.9` / `0.0%` |

---

## 5. Failure Attribution, Graph Navigation Purity, and Sensitivity Sweeps

1. **Graph Navigation Purity ($\eta_{\text{graph}} = \text{visited matching} / \text{visited total}$):**
   - In Mode A (`independent`), $\eta_{\text{graph}}$ tracks selectivity almost identically across both scales (`100k`: `0.0004` at $\sigma=0.0005$ to `0.4995` at $\sigma=0.50$; `1M`: `0.0004` at $\sigma=0.0005$ to `0.5002` at $\sigma=0.50$). At `1M` $\sigma=0.001$ `S2_ef100`, an average query computes `179,478.0` distances, of which **`179,298.3` (`99.90%`) are bridge evaluations** over non-matching nodes, yet still under-fills `k = 10` on `34.0%` of queries (`Recall@10 = 0.7753`).
   - In Mode B (`correlated`), matching nodes are assigned by greedy spatial clustering (`C = max(64, floor(2 / sigma))` centroids; clusters are marked passing sequentially until `floor(sigma * N)` capacity plus a partial shortfall cluster — i.e., ~2 clusters out of `C` at $\sigma \le 0.035$, rising to ~half of the 64 clusters only at $\sigma = 0.50$). Notice the non-monotonic dip between $\sigma = 0.002$ (`C = 1000`, `1M S2_ef100 R@10 = 0.3153`) and $\sigma = 0.005$ (`C = 400`, `R@10 = 0.1277`): as `C` shrinks faster than $\sigma$ doubles here, matching vectors concentrate into larger separated Voronoi pockets where out-edges of `M=16` HNSW do not easily cross non-matching Voronoi boundaries. Even when $\eta_{\text{graph}}$ climbs above $\sigma$ once a matching cluster is entered (`0.1622` vs `0.10` at `1M` `0.10`), queries whose spatial nearest cluster fails the predicate still suffer `34.2%–92.3%` `<k` starvation for $\sigma \le 0.075$.
   - **`S1_cap4096` vs `S1_uncap`:** Capped post-filtering hits the `4096` fetch limit (`post_filter_cap_reached_rate = 66%–100%` at Mode A $\sigma \le 0.002$ and `60%–96%` at Mode B $\sigma \le 0.05$ on `1M`), yielding poor recall unless overfetching is uncapped. Uncapped `S1` reaches `0.988–0.994` recall on Mode B $\sigma \in \{0.05, 0.075\}$, but pays **`183.4–315.0 ms`** (`3.2–5.5 QPS`) vs **`7.6–11.3 ms`** (`88.5–131.5 QPS`) for `S3`.
2. **Bridge-Hop Budget Sensitivity (`H_max` $\in \{1\times, 3\times, 10\times\} \cdot \text{efSearch}$, [`sensitivity_n1000000.json`](./sensitivity_n1000000.json)):**
   - **Mode A $\sigma=0.001, \text{ef}=100$ (`1M`):** Expanding $H_{\max}$ from `1x -> 3x -> 10x` rescues recall (`0.3000 -> 0.8046 -> 0.9920`) and drops under-fill rate (`91.4% -> 29.8% -> 1.4%`), at the cost of $10.4\times$ higher mean latency (`11.752 -> 87.195 -> 121.816 ms` — over `1,700x` slower than `S3`).
   - **Mode B $\sigma=0.001, \text{ef}=100$ (`1M`):** Even at $H_{\max} = 10\times$, recall peaks at only `0.2548` (`75.0%` hop exhaustion / `<k` starvation) because directed Layer-0 graph connectivity from the entry cluster fails to reach isolated matching clusters regardless of hop depth.
3. **Overfetch Factor Sensitivity ($\gamma \in \{\text{formula}, 1.0, 2.0, 5.0\}$, `S1_cap4096_ef100`):**
   - At `1M` Mode A $\sigma = 0.10$ (`cap` never reached), changing $\gamma = 1.0 \to 5.0$ increases `Recall@10` from `0.9516` to `0.9958` while latency rises from `0.859` to `1.839 ms`.
   - At `1M` Mode B $\sigma = 0.01$, all $\gamma$ variants stall at `Recall@10 = 0.2288–0.2304` (`79.4%` hitting `overfetch_cap = 4096`), showing that post-filter multiplier tuning cannot cure cap exhaustion in correlated low-selectivity workloads.

---

## 6. Opaque Closure `eval_limit` Guardrail (`DEFAULT_OPAQUE_EVAL_LIMIT = 16_384`)

Evaluated across `eval_limit in {None, 16_384, 10_000}` (`200` queries/arm at `1M` in [`sensitivity_n1000000.json`](./sensitivity_n1000000.json), `300` at `100k` in [`sensitivity_n100000.json`](./sensitivity_n100000.json)):

| Scale | Mode & $\sigma$ | Strategy | `limit = None` (R@10 / mean / p99 / max ev.) | `limit = 16_384` (R@10 / mean / p99 / hit%) | `limit = 10_000` (R@10 / mean / p99 / hit%) | Guardrail Takeaway |
|---:|---|---|---:|---:|---:|---|
| `100k` | Ind `0.10` | `S2_ef100` | `1.0000` / `1.563` / `2.308 ms` / `11,738` | `1.0000` / `1.528` / `2.203 ms` / `0.0%` | `1.0000` / `1.772` / `3.985 ms` / `7.0%` | Zero truncation at `16,384` (`7.0%` clipped at `10,000`) |
| `100k` | Corr `0.01` | `S2_ef100` | `0.3437` / `3.291` / `17.612 ms` / `51,915` | `0.3313` / `1.751` / `4.618 ms` / `27.3%` | `0.3107` / `1.235` / `2.778 ms` / `31.0%` | Cuts `100k` `p99` $3.81\times$ |
| `1M` | Ind `0.10` | `S2_ef100` | **`0.9985`** / `2.918` / `4.165 ms` / `16,503` | **`0.9985`** / `2.896` / `4.084 ms` / **`1.0%`** | `0.9985` / `2.428` / `2.904 ms` / **`69.0%`** | **`16,384` preserves `0.9985` recall (`1%` hit vs `69%` at `10k`)** |
| `1M` | Corr `0.01` | `S2_ef100` | `0.2510` / `5.954` / **`53.729 ms`** / **`196,017`** | `0.2050` / `1.844` / **`4.797 ms`** / `17.0%` | `0.1890` / `1.486` / `2.901 ms` / `20.5%` | **Prevents runaway opaque closure DoS: caps `p99` by $11.20\times$** |
| `1M` | Corr `0.10` | `S2_ef100` | `0.8970` / `7.195` / **`29.891 ms`** / `100,604` | `0.8740` / `2.937` / **`5.430 ms`** / `46.0%` | `0.8405` / `1.994` / `3.148 ms` / `55.5%` | **Caps `p99` by $5.50\times$** |
| `1M` | All above | `S1_cap4096` | Unchanged (`max <= 4096`) | `0.0%` hit | `0.0%` hit | `S1_cap4096` sits safely below `16,384` everywhere |

---

## 7. Visualization Index (`T3.10`, [`scripts/plot_phase5.py`](../../scripts/plot_phase5.py))

All 5 charts exist in both PNG and SVG across [`charts_100k/`](./charts_100k/), [`charts_1m/`](./charts_1m/), and root [`results/phase5/`](./) (showing `1M` scale):
1. `chart1_recall_vs_selectivity.{png,svg}` — `Recall@10` across all 11 arms vs selectivity (with `S1_uncap` restricted strictly to $\sigma \ge 0.05$).
2. `chart2_latency_vs_selectivity.{png,svg}` — Mean latency curves with vertical stems to `p99` tails on log-log axes.
3. `chart3_failure_rate_vs_selectivity.{png,svg}` — Fewer-than-$k$ starvation rate vs selectivity.
4. `chart4_matched_recall_crossover.{png,svg}` — Matched-recall latency curves (`>= 0.95` & `>= 0.99`) with shaded crossover bracket and log-log $\sigma^*$ markers.
5. `chart5_distance_evals_and_sensitivity.{png,svg}` — Mean distance evaluations, graph navigation purity $\eta_{\text{graph}}$, $H_{\max}$ bridge sensitivity, and opaque `eval_limit` invocation bars.
