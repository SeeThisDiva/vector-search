# Vector Search: HNSW from Scratch in Safe Rust

A from-scratch **Rust** (`#![forbid(unsafe_code)]`, `std`-only) Approximate Nearest Neighbor (ANN) vector search library and reproducible benchmark suite, cross-verified against **FAISS** (`IndexFlatL2`, `IndexHNSWFlat`) and official **INRIA Texmex SIFT** ground truth, grounded in:
> *Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE TPAMI, 42(4), 824–836.*

## Overview & Key Properties

- **100% Safe, Zero-Dependency Core**: `#![forbid(unsafe_code)]` with only Rust `std`; compact flat `u32` graph arenas (`148.91 B/vec` graph-only on `SIFT1M`) and 8-way unrolled `f32` distance kernels (`L2Squared`, `Euclidean`, `InnerProduct`).
- **Deterministic & Reproducible**: Seeded `SplitMix64 + Xoshiro256**` PRNG (`seed = 42`), canonical `(distance, id)` total-order tie-breaking, and 64-bit FNV-1a `graph_fingerprint()` verification across runs.
- **Full Feature Scope**: Exact brute-force `ExactKnn`, Texmex `.fvecs`/`.ivecs` I/O (`siftsmall`/`SIFT1M` auto-download + offline synthetic fallback), HNSW (`HnswIndex`), filtered search (`S1` post-filter, `S2` in-traversal, `S3` exact pre-filter over `DenseBitset`, opaque `eval_limit` guardrail), and isolated `SQ8` / `PQ16` / `PQ32` quantization (`HnswSq8Index`, `HnswPqIndex`) with optional `f32` reranking.
- **Specs & Audit Trail**: Math & pseudocode in [`docs/HNSW_ALGORITHM.md`](docs/HNSW_ALGORITHM.md); artifact guide in [`results/README.md`](results/README.md); JSON-backed claim ledger in [`results/CLAIMS.md`](results/CLAIMS.md). *(Peak RSS is queried via PowerShell `PeakWorkingSet64` in isolated child processes because `#![forbid(unsafe_code)]` disallows Win32 FFI `GetProcessMemoryInfo`.)*

---

## Quickstart

**Prerequisites:** Rust `>= 1.88`, Python `>= 3.11`, and `pip install -r requirements.txt` (needed only for Python tests, FAISS benchmarks, and chart generation).

### Rust Library Usage

```rust
use vector_search::distance::Metric;
use vector_search::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use vector_search::matrix::RowMatrix;

fn main() -> vector_search::error::Result<()> {
    let base = RowMatrix::new(3, 4, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])?;
    let mut index = HnswIndex::build(&base, Metric::L2Squared, HnswConfig::new(16, 100))?;

    let query = [0.9_f32, 0.0, 0.0, 0.0];
    let hits = index.search_knn(&query, 2, 50)?; // Vec<(distance, id)>

    // Read-only / thread-safe search with caller-owned VisitedTracker:
    let mut tracker = VisitedTracker::new(index.num_vectors());
    let ro_hits = index.search_knn_with_tracker(&query, 2, 50, &mut tracker)?;
    assert_eq!(hits, ro_hits);
    Ok(())
}
```

- **Filtered & Parallel Search**: `HnswIndex::search_batch_parallel` (scoped multi-threading) and `HnswIndex::search_knn_filtered` / `search_knn_filtered_with_scratch` (`DenseBitset`, `LambdaPredicate`, `FilterStrategy`, `FilteredSearchParams::with_opaque_eval_limit()`).
- **Quantized Search**: `vector_search::quantized::{HnswSq8Index, HnswPqIndex}` with `QuantizedBuildMode::{Float32GraphThenQuantize, QuantizedDirect}` (plus `build_streaming_from_fvecs`) and `QuantizedStorageMode::{QuantizedOnly, WithExactRerank}` (`search_knn_rerank_with_scratch`).

### CLI Binaries & Scripts

| Tool | Purpose | Copy-Paste Command |
|---|---|---|
| `phase1` | Exact `ExactKnn` baseline (`siftsmall`, `synthetic`, custom `.fvecs`) | `cargo run --release --bin phase1 -- --dataset siftsmall --k 10 --json-out data/phase1_report.json` |
| `phase2` | `siftsmall` HNSW `efSearch` sweep & repro check | `cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json` |
| `phase3` | `SIFT1M` 1M build, GT parity, multi-scale RSS (`--no-keep-pruned-shrink`) | `cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json` |
| `phase4` | Resumable `M x efConstruction` grid sweep (`.grid` or CSV lists) | `cargo run --release --bin phase4 -- --grid results/phase4/grid_100k.grid --output-dir results/phase4` |
| `phase5` | Filtered search `--microbench`, `--pilot`, `--smoke`, `--matrix` | `cargo run --release --bin phase5 -- --matrix --scale 100000 --reps 5 --include-sensitivity --output-dir results/phase5` |
| `phase6` | Quantization `--microbench`, `--matrix`, `--pilot`, `--parity-check` | `cargo run --release --bin phase6 -- --matrix --dataset sift1m --scale 100000 --reps 3 --affinity 0x0004 --output-dir results/phase6` |
| `phase7` | Rust vs FAISS runner (`--compare-mode single\|multithread`, `--prepare-100k-gt`) | `cargo run --release --bin phase7 -- --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --compare-mode single --threads 1 --affinity 0x0004 --profile-rss-isolated --json-out results/phase7/rust_n100000_C2.json` |
| `generate_metadata` | Deterministic Mode A (`independent`) / Mode B (`correlated`) bitset generator | `cargo run --release --bin generate_metadata -- --mode independent --num-base 10000 --selectivity 0.1 --output data/filter.bin` |
| `filtered_exact_knn` | Multi-threaded exact filtered $k$-NN ground truth | `cargo run --release --bin filtered_exact_knn -- --num-base 10000 --num-query 100 --selectivity 0.1 --output-dir data/filtered_gt` |
| `run_phase1.py` | Python Phase 1 reference & FAISS `IndexFlatL2` parity | `python run_phase1.py --num-base 10000 --num-query 100 --dim 128 --k 10` |
| `scripts/benchmark_faiss.py` | Phase 7 FAISS `IndexHNSWFlat` benchmark & matched-recall harness | `python scripts/benchmark_faiss.py --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --compare-mode single --threads 1 --affinity 0x0400 --profile-rss-isolated --json-out results/phase7/faiss_n100000_C2.json` |
| `scripts/plot_phase{4..7}.py` | Deterministic matplotlib PNG + SVG chart generators | `python scripts/plot_phase6.py --results-dir results/phase6 --scale 1000000 --output-dir results/phase6/charts_1m --phase4-dir results/phase4` |

*(Pass `--dataset synthetic` or `--synthetic --offline` on CLI binaries to run without downloading SIFT corpora.)*

### Verification & CI Gate

Mirrors [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (`231` default Rust tests + `22` `--features instrument` tests + `56` Python tests, `0` skipped; `siftsmall` Rust tests skip if `data/siftsmall/` has not been downloaded via `phase1`):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features instrument -- -D warnings
cargo test --release && cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_phase3_tests --test rust_filtered_counters_tests
cargo build --release --bin phase1 --bin filtered_exact_knn --bin phase5 --bin phase6
python -m pytest tests/ -rs --cov=src && ruff check . && mypy --config-file mypy.ini src tests scripts
```

---

## Verified Benchmark Highlights (Phases 1–7)

All timed benchmarks ran in `--release` (`opt-level = 3, lto = "thin", codegen-units = 1`) on `12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)` with `git_dirty: false`. Every value below maps to a source JSON path in [`results/CLAIMS.md`](results/CLAIMS.md).

### Phases 1–2: Exact Baseline & 10k HNSW (`siftsmall`)

- **Phase 1 Exact Brute-Force `ExactKnn`** (`10,000` base $\times$ `128`-d `float32`, `100` queries, $k=10$): **`~3,640+ QPS`**, `p50 = 0.250 ms`, `p95 = 0.379 ms`, `p99 = 0.520 ms`; `Recall@10(GT, GT) = 100.00% (1.000000)` self-recall; **`100.00%`** (`1000/1000`) official Texmex alignment; **`100.00%`** (`100%` ID alignment) vs `faiss.IndexFlatL2`.
- **Phase 2 `HnswIndex`** (`M=16`, `efConstruction=100`, $k=10$, `seed = 42`, acceptance target `Recall@10 >= 0.98` at `< 0.05 ms` mean latency met at `efSearch=20`; `results/phase3/phase2_report.json`):

| `efSearch` | `10` | `20` | `50` | `100` | `200` |
|---|---:|---:|---:|---:|---:|
| **Recall@10** | `95.1%` | `98.6%` | `100.0%` | `100.0%` | `100.0%` |
| **Mean / p99 (ms)** | `0.018` / `0.027` | `0.027` / `0.037` | `0.057` / `0.082` | `0.096` / `0.135` | `0.165` / `0.242` |
| **Speedup vs Exact** | `13.5x` | `9.1x` | `4.4x` | `2.6x` | `1.5x` |

### Phase 3 & 3.5: Scale to 1M Vectors (`SIFT1M`)

Single-threaded search (`10,000` queries, $k=10$, **5 repetitions**, official Texmex GT cross-checked on 100 queries via brute-force `ExactKnn`: `999/1000` ID alignment + `1` integer $L_2^2$ boundary tie, 100% distance compliance, monotonic recall verified; [`results/phase3/phase3_report.json`](results/phase3/phase3_report.json), [`results/phase3/baseline_profile_report.json`](results/phase3/baseline_profile_report.json), [`results/phase3_5/LOG.md`](results/phase3_5/LOG.md)):

| `efSearch` | Recall@10 | Median Mean (ms) | Mean Spread (ms) | Median p50 (ms) | Median p99 (ms) | p99 Spread (ms) | Median QPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `10` | `73.315%` | `0.100` | `0.007` | `0.097` | `0.175` | `0.020` | `9,942.1` |
| `20` | `84.949%` | `0.153` | `0.003` | `0.150` | `0.255` | `0.013` | `6,523.0` |
| `50` | `94.703%` | `0.299` | `0.030` | `0.299` | `0.477` | `0.270` | `3,344.9` |
| `100` | `98.061%` | `0.490` | `0.033` | `0.501` | `0.723` | `0.096` | `2,040.4` |
| `200` | `99.376%` | `0.828` | `0.035` | `0.861` | `1.177` | `0.098` | `1,208.0` |

- **1M Build & Footprint (`T1.7` Flat `u32` Adjacency):** `493.8 s` (`2,025.2 vectors/s`), top layer `5`, entry point `#829262`, fingerprint `0x44044886c62bca5e` (`415.1 s` / `2,409.2 vectors/s` with `--no-keep-pruned-shrink`). Raw vectors `512.00 B/vector` (`488.28 MiB`); graph-only `148.91 B/vector` (`142.01 MiB`, down `56.0%` from `338.67 B/vector`); graph total (incl. `4 B/vector` visited tags) `152.91 B/vector` (`145.83 MiB`); total index footprint `664.91 B/vector` (`634.11 MiB`); peak working set `646.05 MiB` (down `199.7 MiB` from `845.75 MiB`).
- **Isolated Multi-Scale Profile (`measured_this_run = true`, prefix-only child processes via PowerShell `PeakWorkingSet64`):**

| Scale ($N$) | Build Time (s) | Build QPS | Isolated Peak RSS (MiB) | Graph-Only (B/vec) | Graph Total (B/vec) | Total (B/vec) | Recall@10 (`ef=50`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `10,000` | `1.494` | `6,693.9` | `11.94` | `146.96` | `150.96` | `662.96` | `99.750%` |
| `100,000` | `28.564` | `3,500.9` | `69.21` | `145.57` | `149.57` | `661.57` | `98.140%` |
| `1,000,000` | `493.783` | `2,025.2` | `646.05` | `148.91` | `152.91` | `664.91` | `94.703%` |

### Phase 4: Parameter Sweep, Pareto Frontier & 1M Finals

Swept **`23 configurations`** ($M \in \{8, 12, 16, 24, 32, 48\}$, $\text{efConstruction} \in \{40, 100, 200, 400\}$, $\text{efConstruction} \ge M$) on `100,000` SIFT1M prefix, then confirmed 4 Pareto points on **`1,000,000`** vectors (`10,000` queries, $k=10$, `5 timed passes`, official Texmex 100-NN GT, pinned P-core `0x0004`; [`results/phase4/G3_CHECKLIST.md`](results/phase4/G3_CHECKLIST.md)):

| Operating Point | Build (s / vec/s) | Graph Memory | Peak RSS | `efSearch` (`10 / 50 / 100 / 200 / 600`) Recall@10 | Single-Thread QPS | `p99` Latency (ms) |
|---|---:|---:|---:|---|---|---|
| **$M=8, \text{efC}=100$** *(Low RAM / Max QPS)* | `266.541 s` / `3,751.8` | `85.44 B/vec` (`81.5 MiB`) | `583.42 MiB` | `0.57222` / `0.86072` / `0.93304` / `0.97330` / `0.99470` | `20,150.7` / `6,743.3` / `3,861.6` / `2,071.0` / `781.7` | `0.0830` / `0.2109` / `0.3390` / `0.6434` / `1.6119` |
| **$M=16, \text{efC}=100$** *(Production Standard)* | `450.301 s` / `2,220.7` | `148.91 B/vec` (`142.0 MiB`) | `647.26 MiB` | `0.73315` / `0.94703` / `0.98061` / `0.99376` / `0.99804` | `12,711.7` / `4,209.0` / `2,383.9` / `1,296.0` / `515.5` | `0.1284` / `0.3230` / `0.5515` / `1.0673` / `2.6370` |
| **$M=32, \text{efC}=200$** *(High-Precision Pareto)* | `1,288.897 s` / `775.9` | `272.33 B/vec` (`259.7 MiB`) | `767.40 MiB` | `0.84614` / `0.98278` / `0.99572` / `0.99881` / `0.99944` | `7,953.7` / `2,579.7` / `1,476.8` / `842.2` / `348.6` | `0.2041` / `0.5304` / `0.9404` / `1.6854` / `4.2463` |
| **$M=48, \text{efC}=400$** *(Accuracy-Saturated)* | `2,827.980 s` / `353.6` | `402.42 B/vec` (`383.8 MiB`) | `889.69 MiB` | `0.89372` / `0.99165` / `0.99803` / `0.99922` / `0.99944` | `5,905.2` / `1,918.8` / `1,110.1` / `643.9` / `273.0` | `0.2685` / `0.7254` / `1.2947` / `2.3104` / `5.7290` |

- **Scale Dynamics (`100k -> 1M`):** Fixed `efSearch=50` recall on `M=16, efC=100` drops from `98.02%` to `94.70%` (requiring `ef=100..150` for $\ge 98\%$ at `1M`); graph bytes/vec stays scale-invariant (`81.9 -> 85.4`, `145.6 -> 148.9`, `273.4 -> 272.3`, `404.0 -> 402.4`); build time scales as $O(N \log N)$ (`26.1 s -> 450.3 s`, `17.3x` for $10\times$ vectors).
- **Charts ([`scripts/plot_phase4.py`](scripts/plot_phase4.py)):** [`chart1_recall_vs_p99`](results/phase4/chart1_recall_vs_p99.png) ([SVG](results/phase4/chart1_recall_vs_p99.svg)), [`chart2_recall_vs_qps`](results/phase4/chart2_recall_vs_qps.png) ([SVG](results/phase4/chart2_recall_vs_qps.svg)), [`chart3_build_time_vs_recall`](results/phase4/chart3_build_time_vs_recall.png) ([SVG](results/phase4/chart3_build_time_vs_recall.svg)), [`chart4_memory_vs_m`](results/phase4/chart4_memory_vs_m.png) ([SVG](results/phase4/chart4_memory_vs_m.svg)).

### Phase 5: Filtered Search, Crossovers & Opaque Guardrail

Evaluates `S1_post_filter` (`K'_max=4,096`, uncapped $\sigma \ge 0.05$), `S2_in_traversal`, and `S3_pre_filter` (`DenseBitset`, `T3.6` L1-hot primitives `t_word=0.3042 ns`, `t_cand=23.878 ns`) across Mode A (`independent`) & Mode B (`correlated`) predicates and 12 selectivities $\sigma \in [0.0005, 0.50]$ (`M=16, efC=100, seed=42, reps=5`; [`results/phase5/FINDINGS.md`](results/phase5/FINDINGS.md)):

| Scale ($N$) | Metadata Mode | Crossover $\sigma^*$ (`R@10 >= 0.95`) | Bracket | `1M` Transition Highlight |
|---:|---|---:|---|---|
| `100,000` | Mode A (`independent`) | **`0.079149`** | `[0.075, 0.100]` | Baseline (`S2 ef=20`) |
| `1,000,000` | Mode A (`independent`) | **`0.023799`** | `[0.020, 0.035]` | **$3.33\times$ leftward shift** (`S2 ef=20`: `92.37` vs `S3` `55.38 QPS` at `0.035`; `1,328.34` vs `15.84 QPS` at `0.50`) |
| `100,000` | Mode B (`correlated`) | **`0.252372`** | `[0.200, 0.500]` | Baseline (`S2 ef=400 -> 50`) |
| `1,000,000` | Mode B (`correlated`) | **`0.120868`** | `[0.100, 0.200]` | **$2.09\times$ leftward shift** (`S3` wins for $\sigma \le 0.10$ with `1.0000` recall & `0%` under-fill; `R>=0.99` bracket `[0.10, 0.20]`, `ef=400` at `0.20`; `S1_uncap_ef100` `36.46 QPS`, `399.162 ms p99` at `0.10`) |

- **Opaque `eval_limit` Guardrail & Charts:** `DEFAULT_OPAQUE_EVAL_LIMIT = 16_384` (`FilteredSearchParams::with_opaque_eval_limit()`, keeping `default().eval_limit == None`) preserves `Recall@10 = 0.998500` on `1M` `independent` $\sigma=0.10$ (`1.0%` hit vs `69.0%` at `10_000`) and bounds pathological `1M` `correlated` $\sigma=0.01$ (`196,017 -> 16,384` evals, `p99` `53.729 ms -> 4.797 ms`, **`11.20x` cap**). Charts in [`results/phase5/`](results/phase5/) (`charts_100k/` & `charts_1m/`): `chart1_recall_vs_selectivity`..`chart5_distance_evals_and_sensitivity`.

### Phase 6: Vector Quantization (`SQ8` & `PQ16`/`PQ32`)

Isolated [`src/quantized/`](src/quantized/mod.rs) keeps [`src/hnsw/`](src/hnsw/mod.rs) `f32` paths bit-identical (`0x80af03bd7aebaade` @ `100k`, `0x44044886c62bca5e` @ `1M`). Benchmarked across `11` `100k` arms (`reps=3`, solo `0x0004`) and `9` Tier 1+2 `1M` arms (`reps=5`, sharded `0x0004`/`0x0400`, `power_mode = "balanced"`; [`results/phase6/PROPOSAL.md`](results/phase6/PROPOSAL.md), [`results/phase6/FINDINGS.md`](results/phase6/FINDINGS.md)):

| `T4.2` Microbench Kernel (`4,096` x `128`-d) | `f32_l2_squared` | `sq8_query_shifted` **(Way A, Default)** | `sq8_norm_expanded` *(Way B, Rejected)* | `pq16_adc` *(16x256)* | `pq32_adc` *(32x256)* |
|---|---:|---:|---:|---:|---:|
| **Bytes/Vec / Median (`ns/eval`)** | `512` / `39.6600` (`1.00x`) | `128` / `47.1285` (`0.84x`) | `132` / `36.9335` (`1.07x`) | `16` / **`9.8310`** (**`4.03x`**) | `32` / **`20.1700`** (**`1.97x`**) |
| **Std Rel / Adversarial Abs Err** | exact ref | **`1.7156e-7`** / **`4.6115e-5`** | `3.9406e-6` / **`63.9910`** | — | — |

| `1M` Tier 1+2 Arm (`M=16, efC=100` unless noted) | Build (s) | Payload (`B/vec`) | Build / Query RSS (`MiB`) | Ceiling (`ef=600`) | `ef=100` R@10 / QPS / `p99` | Highlight (`1M`) |
|---|---:|---:|---:|---:|---|---|
| `f32_m16` *(Phase 6, sharded)* | `574.78` | `660.91` | `642.71` / `642.71` | `0.99804` | `0.98061` / `1,676.2` / `1.1951 ms` | Phase 4 solo ref: `450.30 s`, `2,383.9 QPS`, `0.5515 ms` |
| `sq8_only_f32graph` | `490.48` | **`272.54`** (`-58.8%`) | `778.37` / **`276.02`** (`-57.1%`) | `0.98764` | `0.97196` / `2,256.4` / `0.6402 ms` | `R>=0.98`: `1,619.8 QPS` / `0.8758 ms` (`ef=150`) |
| `sq8_only_qbuild` *(streaming)* | `546.54` | **`272.54`** (`-58.8%`) | **`277.29`** (`-64.4%`) / **`276.18`** | `0.98767` | `0.97220` / `2,145.2` / `0.6740 ms` | `< 300 MiB` build & serving RSS; matches `f32graph` |
| `sq8_rerank_f32graph` (`R=100`/`20`) | `572.29` | `784.54` (`+18.7%`) | `778.41` / `778.41` | **`0.99804`** | `0.98060` / `2,172.2` / `0.6256 ms` | `R=20` restores `100%` of `f32` & `FullBeam` recall |
| `sq8_only_m32` (`M=32, efC=200`) | `1,459.99` | `400.28` (`-39.4%`) | `1,022.15` / `398.04` (`-38.1%`) | `0.98898` | `0.98567` / `939.7` / `6.3654 ms`\* | `ef=50`: `0.97409` / `2,424.2 QPS` (`0.99060` @ `100k`) |
| `pq16_only_f32graph` | `582.66` | **`160.67`** (`-75.7%`) | `779.20` / **`169.55`** (`-73.6%`) | `0.54520` *(pk)* | `0.54429` / `4,181.3` / `0.3587 ms` | Extreme RAM compression (`89.9%` graph, `16 B/vec` codes) |
| `pq32_only_f32graph` | `643.22` | `176.67` (`-73.3%`) | `768.55` / `184.67` (`-71.3%`) | `0.72305` | `0.71898` / `2,207.7` / `1.2880 ms` | Unreranked PQ hits recall saturation (`0.76461` @ `100k`) |
| `pq16_rerank_f32graph` | `558.82` | `672.67` (`+1.8%`) | `778.37` / `778.37` | `0.99675` (`FB`) | `0.95032` / `3,978.2` / `0.3356 ms` | `R>=0.99` (`ef=300, FB`): **`1,553.9 QPS` (`+27.0%`) / `0.8536 ms` (`-40.2%`)** |
| `pq32_rerank_f32graph` | `643.16` | `688.67` (`+4.2%`) | `778.41` / `778.41` | **`0.99762`** (`FB`) | `0.97643` / `2,338.8` / `0.5911 ms` | `R>=0.995` (`ef=300, FB`): **`994.0 QPS` (`+49.8%` vs Ph6 `f32_m16`)** |

- **100k Highlights & Caveats:** Streaming cuts `100k` build RSS `-44.2%` (`87.70 -> 48.90 MiB`); isolated query RSS drops to `38.61 MiB` (`SQ8-Only`) & `28.13 MiB` (`PQ16-Only`); `pq16_rerank` at `>=0.99` delivers `+40.9%` QPS (`5,269.0` vs `3,738.9 QPS`) / `-30.7%` `p99`. Windows `balanced` power mode + 2-P-core sharding caused DRAM contention on `f32_m16` and `*` `p99` jitter on `sq8_only_m32`; unreranked PQ has slight post-saturation ranking dips (`R >= 50`/`FB` and `SQ8` stay strictly monotonic). Charts in [`results/phase6/`](results/phase6/) (`charts_100k/` & `charts_1m/`).

### Phase 7: Head-to-Head vs FAISS `IndexHNSWFlat`

Compares safe Rust `HnswIndex` against `faiss-cpu 1.15.1` `IndexHNSWFlat(128, M, METRIC_L2)` across `C1 (8,100)`, `C2 (16,100)`, `C3 (32,200)`, and `C2-nkp (16,100, keep_pruned_shrink=false)` at `100k` (`3` builds, `5` query reps) and `1M` (`1` build, `5` query reps) in [`results/phase7/FINDINGS.md`](results/phase7/FINDINGS.md) & [`matched_recall_summary.json`](results/phase7/matched_recall_summary.json). **SIMD Baseline Disclosure:** Stock PyPI `faiss-cpu 1.15.1` on Windows ships in **`SIMDLevel_NONE`** (`avx2_available == false`), matching default-target Rust SSE2 (`x86_64-pc-windows-msvc`); AVX2 (`-C target-cpu=native` vs AVX2 FAISS) is unmeasured.

| Scale | Config (`T=1` unless noted) | Build Time (Rust vs FAISS `s`) | Graph-Only (`B/vec`) | Isolated Peak RSS (Rust vs FAISS `MiB`) |
|---:|---|---|---|---|
| `100k` | `C1` / `C2` / `C2-nkp` / `C3` | `18.163`/`29.926`/`22.876`/`79.651` vs `18.040`/`20.368`/`—`/`41.807` | `81.90`/`145.57`/`145.57`/`273.41` vs `80.55`/`144.25`/`—`/`272.09` | **`75.96`/`82.11`/`81.99`/`93.68`** vs `167.89`/`174.00`/`—`/`186.09` (`~92 MiB` lower) |
| `100k` | `C2` (`T=16` `0xffff`) | `26.977` *(1T)* vs **`3.339`** *(OMP `8.96x` vs `0x0004`)* | `145.57` vs `144.25` | **`95.23`** vs `174.34` |
| `1M` | `C1` / `C2` / `C2-nkp` / `C3` | `313.285`/`625.665`\* /`428.827`/`1,562.147` vs `277.695`/`404.120`/`—`/`782.080` | `85.44`/`148.91`/`148.91`/`272.33` vs `80.57`/`144.26`/`—`/`272.11` | **`594.31`/`654.53`/`654.32`/`776.42`** vs `1,118.32`/`1,180.04`/`—`/`1,301.15` (`~524–526 MiB` lower) |
| `1M` | `C2` (`T=16` `0xffff`) | `520.115` *(1T)* vs **`76.360`** *(OMP `8.19x` vs `0x0004`)* | `148.91` vs `144.26` | **`719.87`** vs `1,196.62` |

| Matched-Recall (`Mode 1` QPS Ratio / `Mode 2` `p99` Ratio) | `R* = 0.90` | `R* = 0.95` | `R* = 0.98` | `R* = 0.99` | `R* = 0.995` |
|---|---:|---:|---:|---:|---:|
| `100k` `C1` / `C2` / `C2-nkp` / `C3` | `0.8655x`/`0.7159x`/`0.7568x`/— | `0.9058x`/`0.7552x`/`0.8016x`/`0.7045x` | **`1.0392x`**/`0.7987x`/`0.8552x`/`0.7368x` | **`1.1562x`** (`1.1213x`)/`0.7955x`/`0.9208x`/`0.7850x` | **`1.2582x`** (`1.1671x`)/`0.7039x`/`0.9772x`/`0.8255x` |
| `1M` `C1` vs FAISS `C1` | `0.8121x` (**`1.1956x`** `p99`) | `0.9224x` (`0.9984x`) | **`1.2546x`** (**`1.3793x`** `p99`) | *FAISS caps `0.98955`* | *FAISS caps `0.98955`* |
| `1M` `C2` / `C2-nkp` / `C3` | `0.6539x` / `0.7648x` (`1.1524x`) / `0.5023x` | `0.6334x` / `0.7845x` (`1.6258x`) / `0.5870x` | `0.6241x` / `0.8539x` / `0.7058x` | `0.6618x` / `0.9279x` (`1.4113x`) / `0.6921x` | `0.6781x` (**`1.2278x`**) / **`1.0371x`** (**`1.5661x`**) / `0.7223x` |

- **Takeaways & Caveats:** Rust wins **`C1` at `R* >= 0.98`** (`1,356.90` vs `1,081.52 QPS` at `1M R*=0.98`, reaching `0.994700` vs FAISS's `0.989550` cap) and saves **`~524–526 MiB` Peak RSS at `1M`** (`1.68x–1.88x`, borrowing `&RowMatrix<f32>`). Default `C2`/`C3` trail due to `keep_pruned_shrink = true` backfilling reverse-shrink edges; disabling it (`C2-nkp`) cuts `1M` build `31.46%` (`428.827 s`), hits `0.999140` recall, and overtakes FAISS at **`1M R*=0.995` (`1.0371x` QPS, `1.5661x` lower `p99`)**. Rust's single-thread builder loses to FAISS OpenMP `add()` (`8.96x` / `8.19x`); in multi-thread search (`C2`, solo `1M`), `search_batch_parallel` reaches **`87.3%–93.9%` (`T=6`) and `83.5%–90.5%` (`T=16`)** of FAISS Mode 1 QPS (`18,608.51` at `R*=0.95`). Paired `100k C2/C3` & `1M C1/C2/C3` ran concurrently across P-cores (`0x0004`/`0x0400`, raising `*` `1M C2` build vs Phase 4 solo `450.301 s`) under `balanced` power; Mode 2 FAISS `ffi_delta_us` medians (`6.3050`/`7.9195`/`7.0058 us/q`) are `12x–100x` smaller than claimed `p99` margins (`95.6–766.2 us`). Charts in [`results/phase7/`](results/phase7/) (`charts_100k/` & `charts_1m/`).

---

## Repository Layout

```text
├── Cargo.toml                 # Crate `vector-search` (`vector_search` lib + 9 Rust CLIs)
├── src/
│   ├── lib.rs, matrix.rs, distance.rs, rng.rs, exact_knn.rs, recall.rs, hardware.rs, cli_util.rs, json_util.rs
│   ├── dataset/               # Texmex `.fvecs`/`.ivecs` I/O, SIFT loader & Mode A/B metadata bitsets
│   ├── hnsw/                  # `HnswIndex` (Algorithms 1–5, compact u32 arenas, S1/S2/S3 filtered search)
│   ├── filter.rs              # `DenseBitset`, `LambdaPredicate`, `FilteredSearchParams`
│   ├── quantized/             # Isolated `CompactHnswGraph`, `HnswSq8Index` (Way A/B), `HnswPqIndex` (ADC/SDC)
│   ├── benchmark/, phase4/, phase5/, phase6/, main.rs, bin/  # Modularized phase runners & CLI binaries
│   └── hnsw_phase1/           # Python reference implementation & FAISS parity helpers
├── scripts/                   # `benchmark_faiss.py`, `plot_phase{4..7}.py` & shared `plot_common.py`/`_sys_env.py`/`_metrics.py`
├── tests/                     # 18 Rust integration suites (`common/`) + 6 Python pytest suites (`helpers.py`)
├── docs/                      # `HNSW_ALGORITHM.md` (math & pseudocode) + `dev/` (handoff & plan docs)
├── results/                   # `README.md` index, `CLAIMS.md` ledger, `phase3/`..`phase7/` JSONs, findings & charts
├── data/                      # Gitignored SIFT corpora, cached prefix GT & bitsets
└── run_phase1.py              # Python Phase 1 benchmark entry point
```
