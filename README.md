# Vector Search: Hierarchical Navigable Small World (HNSW) from Scratch in Rust

A from-scratch **Rust** (`#![forbid(unsafe_code)]`) Approximate Nearest Neighbor (ANN) vector search engine and reproducible benchmark suite, cross-verified against **FAISS (`IndexFlatL2`)** and official **INRIA Texmex SIFT** ground truth, grounded in:
> *Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE TPAMI, 42(4), 824–836.*

---

## Overview

High-dimensional vector search powers dense semantic retrieval, recommendation systems, and retrieval-augmented generation (RAG). While brute-force exact $k$-NN scales linearly as $\mathcal{O}(N \cdot D)$, HNSW organizes vectors into a multi-layer navigable small-world skip-list graph hierarchy ($m_L = 1 / \ln M$) that achieves $\mathcal{O}(\log N)$ search scaling with high recall.

Before implementing graph construction and greedy/beam routing, **Phase 1** establishes an exact brute-force $k$-NN ground-truth engine, Texmex `.fvecs` / `.ivecs` binary serialization, public SIFT corpus loader (`siftsmall` 10k and `SIFT1M` 1M with deterministic offline fallback), latency percentile instrumentation (`p50`, `p95`, `p99`, `QPS`), and an automated `Recall@k` harness cross-checked against both official Texmex ground truth and `faiss.IndexFlatL2`.

For the full mathematical derivation of HNSW's layer distribution, scale-separated greedy routing, heuristic neighbor diversity, and pseudocode specifications, see **[docs/HNSW_ALGORITHM.md](docs/HNSW_ALGORITHM.md)**.

---

## Build Roadmap (7 Phases — Stop After Each)

```text
Layer L_max :   [Node A] --------------------> [Node B]            (Long-range coarse routing, ef=1)
                     \                              \
Layer L_1   :   [Node A] -----> [Node C] -----> [Node B]           (Intermediate spatial scale, ef=1)
                     \             \                \
Layer L_0   :   [Node A] -> [D] -> [C] -> [E] -> [Node B] -> ...   (All N vectors, beam search ef=efSearch)
```

- [x] **Phase 1: Brute-Force Exact $k$-NN Ground Truth & Recall Measurement Harness**
  - From-scratch Rust `ExactKnn` engine (`L2` squared, Euclidean, Inner Product) with canonical `(distance, id)` deterministic tie-breaking
  - Standard Texmex `.fvecs` / `.ivecs` little-endian readers & writers with full per-record header integrity checks
  - Public SIFT downloader/loader (`siftsmall` 10k $\times$ 128-d and `SIFT1M` 1M $\times$ 128-d) + deterministic seeded `Xoshiro256**` synthetic offline fallback + custom `.fvecs` support
  - Automated `Recall@k` (`Recall@1`, `Recall@5`, `Recall@10`, `Recall@100`) and `Recall@k(GT, GT) == 1.0` self-verification invariant
  - Per-query latency percentile profiler (`min`, `mean`, `p50`, `p95`, `p99`, `max`, `QPS`) & host hardware detector
  - Python + `faiss.IndexFlatL2` cross-language parity verification suite
- [x] **Phase 2: HNSW Insert + Search, Correctness-Checked on 10k Vectors**
  - From-scratch `HnswIndex` (Algorithms 1-5: insert, search-layer, heuristic neighbor selection, k-NN search) with deterministic seeded layer sampling, `Recall@10` vs `efSearch` sweep and bit-exact reproducibility check
  - `m_max`, `m_max0` and `m_l` are derived from `M` unless explicitly overridden; vector ids must be inserted in order `0..n`
- [x] **Phase 3: Scale to 1M Vectors (`SIFT1M`), Profile Memory & Build Time**
  - `phase3` CLI: SIFT1M build, official-ground-truth recall vs efSearch, per-component memory breakdown, peak RSS, bit-exact reproducibility check (see results below)
- [x] **Phase 4: Parameter Sweep (`M`, `efConstruction`, `efSearch`), Plot `Recall@10` vs `p99` Latency**
- [x] **Phase 5: Filtered Search with Metadata Predicates (Including High-Selectivity Filters)**
- [x] **Phase 6: Scalar (`SQ8`) & Product Quantization (`PQ16`/`PQ32`) & Recall Cost Measurement**
  - Isolated [`src/quantized/`](src/quantized/mod.rs) (`CompactHnswGraph`, `HnswSq8Index`, `HnswPqIndex`, `#![forbid(unsafe_code)]`, `std`-only) keeping `f32` `HnswIndex` bit-identical (`0x80af03bd7aebaade` at `100k`, `0x44044886c62bca5e` at `1M`); Way A `sq8_query_shifted` selected over Way B (`4.6115e-5` vs `63.9910` adversarial cancellation error)
  - `100k` (`11` arms) and `1M` (`9` Tier 1+2 arms) `SIFT1M` evaluations ([`results/phase6/FINDINGS.md`](results/phase6/FINDINGS.md)): streaming `sq8_only_qbuild` cuts `1M` build RSS to `277.29 MiB` (`-56.9%` vs `f32_m16`), `SQ8+Rerank` (`R=20`) restores `100%` of `f32` recall, and `PQ16+Rerank` (`1M`) delivers `+27.0%` QPS / `-40.2%` `p99` at `>= 0.99` recall
- [x] **Phase 7: Head-to-Head Benchmark vs FAISS `IndexHNSWFlat` on Identical Data & Hardware**
  - Head-to-head Rust `HnswIndex` vs `faiss-cpu 1.15.1` (`IndexHNSWFlat`, `SIMDLevel_NONE` baseline SSE2 parity) across `C1`/`C2`/`C3`/`C2-nkp` at `100k` and `1M` `SIFT1M` ([`results/phase7/FINDINGS.md`](results/phase7/FINDINGS.md))
  - Matched-recall Mode 1 pure-batch + Mode 2 `p99` (`ffi_delta_us` isolated), `~524–526 MiB` `1M` isolated Peak RSS advantage (`1.68x–1.88x`), `keep_pruned_shrink=false` (`C2-nkp`) root-cause isolation (`1.0371x` QPS / `1.5661x` lower `p99` at `1M R*=0.995`), OpenMP build asymmetry (`8.96x`/`8.19x` FAISS win), and `T in {1, 6, 16}` batch search scaling

---

## Phase 1 Baseline Results (`siftsmall` 10,000 $\times$ 128-d)

- **Hardware**: `12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)`
- **Dataset**: Authentic INRIA Texmex `siftsmall` (`10,000` base 128-d `float32` vectors, `100` query 128-d `float32` vectors)
- **Rust Brute-Force `ExactKnn` (Single-Threaded Sequential Latency)**:
  - **Throughput**: `~3,640+ QPS`
  - **Latency**: `p50 = 0.250 ms`, `p95 = 0.379 ms`, `p99 = 0.520 ms`
  - **Self-Recall**: `Recall@10(GT, GT) = 100.00% (1.000000)`
  - **Official Texmex Parity**: `Recall@10 = 100.00%` (`1000/1000` exact ordered neighbor ID alignment)
  - **FAISS `IndexFlatL2` Parity**: `Recall@10 = 100.00%` (`100%` ID alignment, squared $L_2$ distances within `float32` tolerance)

---

## Phase 2 Results (`siftsmall` 10,000 $\times$ 128-d, `M=16`, `efConstruction=100`, $k=10$)

Recall is deterministic (seed 42). Latencies are single-threaded and measured on `12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)`.

| efSearch | Recall@10 | Mean latency (ms) | p99 (ms) | Speedup vs brute force |
|---------:|----------:|------------------:|---------:|-----------------------:|
| 10  | 95.1%  | 0.018 | 0.027 | 13.5x |
| 20  | 98.6%  | 0.027 | 0.037 | 9.1x  |
| 50  | 100.0% | 0.057 | 0.082 | 4.4x  |
| 100 | 100.0% | 0.096 | 0.135 | 2.6x  |
| 200 | 100.0% | 0.165 | 0.242 | 1.5x  |

Acceptance target: some efSearch reaches Recall@10 >= 0.98 at < 0.05 ms mean latency (met at efSearch=20). `siftsmall` is easy (recall saturates by efSearch=50), so treat these as a correctness check; Phase 3 on `SIFT1M` is the real test.

```bash
cargo run --release --bin phase2 -- --dataset siftsmall --json-out results/phase3/phase2_report.json
```

---

## Phase 3 Results (`SIFT1M` 1,000,000 $\times$ 128-d, `M=16`, `efConstruction=100`, $k=10$)

Measured on `12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)`, official Texmex ground truth (verified against brute-force `ExactKnn` on 100 queries), 10,000 queries, single-threaded search, **5 repetitions per `efSearch`** (reporting median and min–max spread). Source: `results/phase3/phase3_report.json` and `results/phase3/baseline_profile_report.json` (committed; see `results/CLAIMS.md`).

| efSearch | Recall@10 | Median Mean (ms) | Mean Spread (ms) | Median p50 (ms) | Median p99 (ms) | p99 Spread (ms) | Median QPS |
|---------:|----------:|-----------------:|-----------------:|----------------:|----------------:|----------------:|-----------:|
| 10  | 73.315% | 0.100 | 0.007 | 0.097 | 0.175 | 0.020 | 9,942.1 |
| 20  | 84.949% | 0.153 | 0.003 | 0.150 | 0.255 | 0.013 | 6,523.0 |
| 50  | 94.703% | 0.299 | 0.030 | 0.299 | 0.477 | 0.270 | 3,344.9 |
| 100 | 98.061% | 0.490 | 0.033 | 0.501 | 0.723 | 0.096 | 2,040.4 |
| 200 | 99.376% | 0.828 | 0.035 | 0.861 | 1.177 | 0.098 | 1,208.0 |

| Metric | Value |
|--------|-------|
| Build time | 493.8 s (2,025.2 vectors/s), top layer 5, entry point `#829262`, fingerprint `0x44044886c62bca5e` (415.1 s / 2,409.2 vectors/s with `--no-keep-pruned-shrink`) |
| Raw vectors | 512.00 B/vector (488.28 MiB) |
| Graph-only (excl. visited tags & scratch) | 148.91 B/vector (142.01 MiB, down 56.0% from 338.67 B/vector via compact `u32` flat adjacency) |
| Graph total (incl. 4 B/vector visited tags) | 152.91 B/vector (145.83 MiB) |
| Total index footprint | 664.91 B/vector (634.11 MiB) |
| Peak working set after build (1M) | 646.05 MiB (down 199.7 MiB from 845.75 MiB) |

### Multi-Scale Baseline Profile (Isolated Process per Scale, `measured_this_run = true`)

| Scale ($N$) | Build Time (s) | Build QPS | Isolated Peak RSS (MiB) | Graph-Only (B/vec) | Graph Total (B/vec) | Total (B/vec) | Recall@10 (`ef=50`) |
|------------:|---------------:|----------:|------------------------:|-------------------:|--------------------:|--------------:|--------------------:|
| 10,000 | 1.494 | 6,693.9 | 11.94 | 146.96 | 150.96 | 662.96 | 99.750% |
| 100,000 | 28.564 | 3,500.9 | 69.21 | 145.57 | 149.57 | 661.57 | 98.140% |
| 1,000,000 | 493.783 | 2,025.2 | 646.05 | 148.91 | 152.91 | 664.91 | 94.703% |

Notes:
- Each scale (`10k`, `100k`, `1M`) is profiled in its own isolated OS process loading only `scale` prefix rows from `sift_base.fvecs`, so `PeakWorkingSet64` reflects only that scale's true memory footprint.
- `official_ground_truth_matched` runs brute-force `ExactKnn` on the first 100 queries against the full 1M base (`999/1000` identical IDs and `1` exact integer $L_2^2$ distance tie at the $k=10$ boundary, with 100% of official top-10 neighbors verified to have distance $\le$ the true 10th nearest-neighbor distance).
- Recall at fixed `efSearch=50` drops with scale (`99.75%` at 10k, `98.14%` at 100k, `94.70%` at 1M), so `efSearch` must grow with $N$.

```bash
cargo run --release --bin phase3 -- --dataset sift1m --json-out results/phase3/phase3_report.json
```

---

## Repository Structure

```text
├── Cargo.toml                 # Rust crate manifest (vector_search lib + phase1..7, generate_metadata, filtered_exact_knn)
├── docs/
│   └── HNSW_ALGORITHM.md      # HNSW mathematical derivations, layer theory & pseudocode
├── src/
│   ├── lib.rs                 # Rust crate root (#![forbid(unsafe_code)])
│   ├── main.rs                # Rust Phase 1 CLI runner (`phase1`)
│   ├── hnsw.rs                # HNSW index (insert, search-layer, heuristic selection, k-NN, filtered search)
│   ├── bin/phase2.rs          # Rust Phase 2 CLI runner (`phase2`)
│   ├── bin/phase3.rs          # Rust Phase 3 CLI runner (`phase3`, SIFT1M)
│   ├── bin/phase4.rs          # Rust Phase 4 CLI runner (`phase4`, parameter sweep)
│   ├── bin/phase5.rs          # Rust Phase 5 CLI runner (`phase5`, filtered matrix & sensitivity)
│   ├── bin/phase6.rs          # Rust Phase 6 CLI runner (`phase6`, SQ8 & PQ matrix, microbench & parity)
│   ├── bin/phase7.rs          # Rust Phase 7 CLI runner (`phase7`, head-to-head vs FAISS)
│   ├── bin/generate_metadata.rs # Deterministic Mode A/B bitset & metadata generator CLI
│   ├── bin/filtered_exact_knn.rs # Multi-threaded filtered exact k-NN ground-truth CLI
│   ├── matrix.rs              # Contiguous row-major RowMatrix<T> storage
│   ├── distance.rs            # 8-way unrolled L2^2, Euclidean, and Inner Product kernels
│   ├── rng.rs                 # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs             # Texmex .fvecs/.ivecs I/O, SIFT downloader, seeded & Mode A/B generators
│   ├── exact_knn.rs           # Brute-force ExactKnn (single-query, timed, filtered & multi-threaded)
│   ├── filter.rs              # DenseBitset, metadata predicates & FilteredSearchParams
│   ├── recall.rs              # Recall@k harness, self-verification & latency percentiles
│   ├── hardware.rs            # Cross-platform CPU / RAM / OS hardware probe
│   ├── benchmark.rs           # End-to-end Phase 1-3 benchmark pipeline & JSON reporter
│   ├── phase4.rs              # Phase 4 resumable parameter sweep runner & prefix GT caching
│   ├── phase5.rs              # Phase 5 Option C matrix, versioned GT/bitset cache & sensitivity
│   ├── phase6.rs              # Phase 6 SQ8/PQ matrix, kernel microbench, isolated RSS & parity runner
│   ├── quantized/             # Isolated lifetime-free quantized HNSW module (`SQ8` & `PQ`)
│   │   ├── mod.rs             # Public traits, distance formatters & binary snapshot helpers
│   │   ├── graph.rs           # `CompactHnswGraph` (flat `u32` arenas), `QuantizedDistanceSource` & scratch
│   │   ├── sq8.rs             # `Sq8Quantizer`, Way A/B kernels, streaming `.fvecs` builder & `HnswSq8Index`
│   │   └── pq.rs              # Seeded `Xoshiro256**` PQ trainer, ADC/SDC kernels & `HnswPqIndex`
│   └── hnsw_phase1/           # Python reference implementation & FAISS comparison helpers
├── scripts/
│   ├── benchmark_faiss.py     # Phase 7 FAISS `IndexHNSWFlat` benchmark harness & matched-recall helper
│   ├── plot_phase4.py         # Phase 4 Pareto & scaling chart generator
│   ├── plot_phase5.py         # Phase 5 filtered recall, latency & crossover chart generator
│   ├── plot_phase6.py         # Phase 6 quantization recall/latency, memory Pareto & rerank chart generator
│   └── plot_phase7.py         # Phase 7 Rust vs FAISS parity, matched-recall & scaling chart generator
├── tests/                     # 188 Rust integration tests (+ 35 unit tests in src/ = 223 default) + 56 Python tests
│   ├── rust_phase1_tests.rs   # Rust integration & edge-case test suite (16 tests)
│   ├── rust_phase2_tests.rs   # HNSW correctness, topology & recall tests (23 tests)
│   ├── adversarial_hnsw_tests.rs # HNSW stress / boundary / reproducibility tests (12 tests)
│   ├── adversarial_graph_fingerprint_tests.rs # Logical graph fingerprint & mutation tests (10 tests)
│   ├── challenge_m1_p35_tests.rs # Phase 3.5 instrumentation, golden & layout tests (8 tests)
│   ├── rust_phase3_tests.rs   # Phase 3 report semantics + parallel-search input validation (11 tests)
│   ├── rust_phase4_tests.rs   # Phase 4 grid parsing, GT caching & resumable sweep tests (4 tests)
│   ├── rust_metadata_tests.rs # Mode A/B predicate determinism & popcount tests (10 tests)
│   ├── rust_filtered_knn_tests.rs # Exact filtered k-NN & canonical tie-breaking tests (7 tests)
│   ├── rust_filtered_strategy1_tests.rs # S1 post-filter & overfetch cap tests (14 tests)
│   ├── rust_filtered_strategy2_tests.rs # S2 in-traversal, bridge-hop & eval_limit tests (13 tests)
│   ├── rust_filtered_strategy3_tests.rs # S3 pre-filter bitset scan tests (14 tests)
│   ├── rust_filtered_counters_tests.rs # Instrumented filtered distance & bridge-hop counter tests (3 tests)
│   ├── rust_phase5_tests.rs   # Phase 5 cache verification, crossover & CLI matrix tests (4 tests)
│   ├── rust_phase6_tests.rs   # CompactHnswGraph, SQ8 Way A/B, streaming & rerank tests (16 tests)
│   ├── rust_phase6_pq_tests.rs # PQ training determinism, ADC/SDC parity & HnswPqIndex tests (12 tests)
│   ├── rust_phase6_cli_tests.rs # Phase 6 CLI microbench, matrix/resume, parity & isolated RSS tests (7 tests)
│   ├── rust_phase7_tests.rs   # Phase 7 CLI single/multithread, C2-nkp & prefix GT tests (4 tests)
│   ├── test_phase1.py         # Python reference unit tests (23 tests)
│   ├── test_rust_parity.py    # Cross-language Rust <-> Python <-> FAISS parity tests (4 tests)
│   ├── test_filtered_parity.py # Filtered strategy & FAISS parity verification tests (5 tests)
│   ├── test_phase5_pipeline.py # End-to-end Phase 5 CLI + resume + plotting pipeline tests (1 test)
│   ├── test_phase6_parity.py  # Rust <-> NumPy SQ8/PQ parity, plot_phase6 & matrix pipeline tests (16 tests)
│   └── test_phase7_harness.py # Phase 7 affinity, OMP, FAISS parity, matched-recall & plot_phase7 tests (7 tests)
├── results/
│   ├── CLAIMS.md              # Claim ledger tracing documented metrics to source JSON reports
│   ├── phase3/ .. phase5/     # Phases 3–5 JSON reports, proposals, findings & charts
│   ├── phase6/                # Phase 6 DESIGN/PROPOSAL/FINDINGS, 23 JSONs & 100k/1M charts
│   └── phase7/                # Phase 7 DESIGN/FINDINGS, matched_recall_summary.json, 22 run JSONs & charts
└── run_phase1.py              # Python reference runner
```

---

## Quickstart

### 1. Run Rust Phase 1 Benchmark (Authentic `siftsmall` Download + Ground Truth)

```bash
cargo run --release --bin phase1 -- --dataset siftsmall --k 10 --json-out data/phase1_report.json
```

Run offline with deterministic seeded synthetic 128-d vectors (or custom `.fvecs` embeddings):

```bash
# Deterministic synthetic mode (no network required)
cargo run --release --bin phase1 -- --dataset synthetic --num-base 10000 --num-query 100 --dim 128 --k 10 --seed 42

# Custom user embeddings (.fvecs)
cargo run --release --bin phase1 -- --base-fvecs path/to/base.fvecs --query-fvecs path/to/query.fvecs --k 10
```

### 2. Run Rust Tests & Lints

Requires Rust >= 1.88 (`223` default tests: `35` lib unit + `188` across `tests/*.rs`; plus `11` `--features instrument` tests). Tests that need the real `siftsmall` corpus skip themselves when `data/siftsmall/*.fvecs` is missing (run the Phase 1 command above once to download it).

```bash
cargo test --release
cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_filtered_counters_tests
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features instrument -- -D warnings
cargo fmt --all -- --check
```

### 3. Run Cross-Language Python & FAISS Parity Verification

Runs `56` Python unit, parity, harness, and plotting tests (`0 skipped` enforced in [`.github/workflows/ci.yml`](.github/workflows/ci.yml); CI checks `mypy src tests`, and `scripts/` is verified identically locally):

```bash
pip install -r requirements.txt
python -m pytest tests/ -v -rs --cov=src
ruff check .
mypy src tests scripts
```

### 4. Run Phase 6 & Phase 7 Benchmarks and Chart Generators

```bash
# Phase 6: kernel microbenchmark (Way A vs Way B vs PQ16/PQ32), 100k/1M matrix & offline smoke
cargo run --release --bin phase6 -- --microbench --output-dir results/phase6/pilot
cargo run --release --bin phase6 -- --matrix --dataset sift1m --scale 100000 --reps 3 --affinity 0x0004 --output-dir results/phase6
cargo run --release --bin phase6 -- --pilot --synthetic --scale 2000 --queries 100 --reps 1 --output-dir data/phase6_smoke
python scripts/plot_phase6.py --results-dir results/phase6 --scale 100000 --output-dir results/phase6/charts_100k --phase4-dir results/phase4
python scripts/plot_phase6.py --results-dir results/phase6 --scale 1000000 --output-dir results/phase6/charts_1m --phase4-dir results/phase4

# Phase 7: Rust vs FAISS IndexHNSWFlat head-to-head (example C2 = M=16, efC=100 at 100k) & charts
cargo run --release --bin phase7 -- --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --compare-mode single --threads 1 --affinity 0x0004 --profile-rss-isolated --json-out results/phase7/rust_n100000_C2.json
python scripts/benchmark_faiss.py --dataset sift1m --scale 100000 --m 16 --ef-construction 100 --compare-mode single --threads 1 --affinity 0x0400 --profile-rss-isolated --json-out results/phase7/faiss_n100000_C2.json
python scripts/plot_phase7.py --results-dir results/phase7 --scales 100000 --output-dir results/phase7/charts_100k
python scripts/plot_phase7.py --results-dir results/phase7 --scales 1000000 --output-dir results/phase7/charts_1m
```

---

## Phase 4: Parameter Sweep, Pareto Frontier & 1M Finals

Phase 4 executes an empirical grid exploration across degree bounds $M \in \{8, 12, 16, 24, 32, 48\}$ and construction depths $\text{efConstruction} \in \{40, 100, 200, 400\}$ subject to $\text{efConstruction} \ge M$ (23 configurations) on a 100,000-vector prefix of Texmex SIFT1M, followed by 1,000,000-vector confirmation sweeps across 4 selected Pareto-optimal points.

All benchmarks were executed single-threaded, pinned to Performance core affinity mask `0x0004` on a 12th Gen Intel(R) Core(TM) i7-12650H under clean Git commits.

### Publication-Grade Benchmark Charts

Generated deterministically via [`scripts/plot_phase4.py`](scripts/plot_phase4.py) (matplotlib only, zero external styling dependencies) into [`results/phase4/`](results/phase4/):

1. **Recall@10 vs p99 Latency (Log Scale):** [`results/phase4/chart1_recall_vs_p99.png`](results/phase4/chart1_recall_vs_p99.png) ([SVG](results/phase4/chart1_recall_vs_p99.svg))
   *Plots all 23 100k curves, the 100k Pareto frontier, and overlay markers for the 4 1M final confirmations.*
2. **Recall@10 vs Single-Thread QPS:** [`results/phase4/chart2_recall_vs_qps.png`](results/phase4/chart2_recall_vs_qps.png) ([SVG](results/phase4/chart2_recall_vs_qps.svg))
3. **Construction Efficiency (Build Time vs Recall):** [`results/phase4/chart3_build_time_vs_recall.png`](results/phase4/chart3_build_time_vs_recall.png) ([SVG](results/phase4/chart3_build_time_vs_recall.svg))
4. **Graph Memory Footprint vs Degree Bound $M$:** [`results/phase4/chart4_memory_vs_m.png`](results/phase4/chart4_memory_vs_m.png) ([SVG](results/phase4/chart4_memory_vs_m.svg))

### 1,000,000 Vector Final Confirmation (Authentic SIFT1M, 5 Timed Repetitions)

Evaluated against the official Texmex 100-NN ground truth ([`data/sift1m/sift_groundtruth.ivecs`](data/sift1m/sift_groundtruth.ivecs)) across 10,000 queries ($k=10$, 1 warmup + 5 timed passes):

| Operating Point | Build Time (s) | Build QPS | Graph Memory | Peak RSS | efSearch | Recall@10 | Single-Thread QPS | p99 Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M=8, \text{efC}=100$**<br>*(Throughput / Low Memory)* | **266.541 s** | 3,751.8 vec/s | **85.44 B/vec**<br>(81.5 MiB) | 583.42 MiB | 10<br>50<br>100<br>200<br>600 | 0.57222<br>0.86072<br>0.93304<br>0.97330<br>0.99470 | **20,150.7**<br>6,743.3<br>3,861.6<br>2,071.0<br>781.7 | **0.0830 ms**<br>0.2109 ms<br>0.3390 ms<br>0.6434 ms<br>1.6119 ms | 0.0495 ms<br>0.1481 ms<br>0.2588 ms<br>0.4826 ms<br>1.2790 ms |
| **$M=16, \text{efC}=100$**<br>*(Phase 3 Reference)* | **450.301 s** | 2,220.7 vec/s | **148.91 B/vec**<br>(142.0 MiB) | 647.26 MiB | 10<br>50<br>100<br>200<br>600 | 0.73315<br>0.94703<br>0.98061<br>0.99376<br>0.99804 | 12,711.7<br>4,209.0<br>2,383.9<br>1,296.0<br>515.5 | 0.1284 ms<br>0.3230 ms<br>0.5515 ms<br>1.0673 ms<br>2.6370 ms | 0.0785 ms<br>0.2374 ms<br>0.4193 ms<br>0.7714 ms<br>1.9397 ms |
| **$M=32, \text{efC}=200$**<br>*(Balanced Pareto Point)* | **1,288.897 s** | 775.9 vec/s | **272.33 B/vec**<br>(259.7 MiB) | 767.40 MiB | 10<br>50<br>100<br>200<br>600 | 0.84614<br>**0.98278**<br>**0.99572**<br>**0.99881**<br>0.99944 | 7,953.7<br>2,579.7<br>1,476.8<br>842.2<br>348.6 | 0.2041 ms<br>0.5304 ms<br>0.9404 ms<br>1.6854 ms<br>4.2463 ms | 0.1255 ms<br>0.3874 ms<br>0.6769 ms<br>1.1871 ms<br>2.8686 ms |
| **$M=48, \text{efC}=400$**<br>*(High-Accuracy / Saturated)* | **2,827.980 s** | 353.6 vec/s | **402.42 B/vec**<br>(383.8 MiB) | 889.69 MiB | 10<br>50<br>100<br>200<br>600 | 0.89372<br>**0.99165**<br>**0.99803**<br>**0.99922**<br>**0.99944** | 5,905.2<br>1,918.8<br>1,110.1<br>643.9<br>273.0 | 0.2685 ms<br>0.7254 ms<br>1.2947 ms<br>2.3104 ms<br>5.7290 ms | 0.1692 ms<br>0.5209 ms<br>0.9006 ms<br>1.5529 ms<br>3.6630 ms |

### Recommended Operating Points

1. **Ultra-Low Memory / Throughput-Critical:** **$M=8, \text{efConstruction}=100$**
   - Memory: only **$85.44\text{ B/vector}$** graph storage ($583\text{ MiB}$ peak RSS for 1M vectors).
   - Throughput: **$20,150\text{ QPS}$** ($0.083\text{ ms}$ p99) at $\text{ef}=10$.
   - Use case: High-volume serving, embedded environments, or large-scale indexes under strict RAM constraints.
2. **General Production Standard:** **$M=16, \text{efConstruction}=100$**
   - Memory: **$148.91\text{ B/vector}$** ($647\text{ MiB}$ peak RSS for 1M vectors).
   - Performance: $98.06\%$ recall at $\text{ef}=100$ ($2,384\text{ QPS}$), reaching $99.38\%$ recall at $\text{ef}=200$.
   - Build cost: fast $7.5\text{ min}$ build on a single laptop P-core ($2,221\text{ vec/s}$).
3. **High-Precision Production:** **$M=32, \text{efConstruction}=200$**
   - Performance: **$98.28\%$ recall** at only $\text{ef}=50$ ($2,580\text{ QPS}$), surpassing $99.57\%$ recall at $\text{ef}=100$ and $99.88\%$ recall at $\text{ef}=200$.
   - Build cost: $21.5\text{ min}$ build time.
   - Use case: Production retrieval systems requiring $\ge 99.5\%$ recall with sub-millisecond p99 latencies.
4. **Accuracy-Saturated Index:** **$M=48, \text{efConstruction}=400$**
   - Performance: **$99.16\%$ recall** at $\text{ef}=50$, saturating at $99.94\%$ recall.
   - Build cost: $47.1\text{ min}$ build time.
   - Use case: Offline evaluation ground-truthing or quality-first semantic ranking where missing any neighbor is unacceptable.

### Scale Dynamics: 100k vs 1M Transitions

- **Recall Degradation at Fixed $\text{efSearch}$:** For reference point $M=16, \text{efC}=100$, recall at $\text{efSearch}=50$ decreases from $98.02\%$ at 100k to $94.70\%$ at 1M. As the vector volume expands by $10\times$, search depth $\text{efSearch}$ must be scaled from $\approx 50$ to $\approx 100\text{–}150$ to preserve $\ge 98\%$ recall.
- **Graph Adjacency Memory Invariance:** Because of the contiguous flat $u32$ edge storage with fixed upper-layer geometry, graph memory per vector remains strictly bounded and scale-invariant ($M=8$: $81.9 \to 85.4\text{ B/vec}$; $M=16$: $145.6 \to 148.9\text{ B/vec}$; $M=32$: $273.4 \to 272.3\text{ B/vec}$; $M=48$: $404.0 \to 402.4\text{ B/vec}$).
- **Construction Scaling:** Index build time scales cleanly at $O(N \log N)$ (e.g. $M=16, \text{efC}=100$ builds in $26.1\text{ s}$ at 100k and $450.3\text{ s}$ at 1M, a scaling factor of $17.3\times$ for $10\times$ vectors).

---

## Phase 5: Filtered Search, Crossovers & Opaque Predicate Guardrail

Phase 5 evaluates three filtered k-NN strategies (`S1_post_filter` capped at $K'_{\max}=4,096$ and uncapped for $\sigma \ge 0.05$, `S2_in_traversal` across `efSearch` $\in \{20, 50, 100, 200, 400, 600\}$, and `S3_pre_filter` exact brute-force scan over `DenseBitset`) across two metadata modes (Mode A: independent Bernoulli; Mode B: cluster-correlated), 12 selectivity tiers $\sigma \in [0.0005, 0.50]$, and two scales ($100\text{k}$ and $1\text{M}$) on `M=16, efConstruction=100, seed=42` (`Option C`, `reps=5`). See [`results/phase5/FINDINGS.md`](results/phase5/FINDINGS.md) and [`results/CLAIMS.md`](results/CLAIMS.md#L292) §11 for full per-cell evidence.

### Recall-Matched Strategy Crossover (`S2` vs `S3`)

| Scale ($N$) | Metadata Mode | Primary Crossover $\sigma^*$ (`R@10 >= 0.95`) | Empirical Bracket | `1M` Transition Highlight |
|---:|---|---:|---|---|
| `100,000` | Mode A (`independent`) | **`0.079149`** | `[0.075, 0.100]` | Baseline (`S2 ef=20`) |
| `1,000,000` | Mode A (`independent`) | **`0.023799`** | `[0.020, 0.035]` | **$3.33\times$ leftward shift** (`S2 ef=20`: `92.37` vs `S3` `55.38 QPS` at `0.035`; `1,328.34` vs `15.84 QPS` at `0.50`) |
| `100,000` | Mode B (`correlated`) | **`0.252372`** | `[0.200, 0.500]` | Baseline (`S2 ef=400 -> 50`) |
| `1,000,000` | Mode B (`correlated`) | **`0.120868`** | `[0.100, 0.200]` | **$2.09\times$ leftward shift** (`S3` beats all `S2` and `S1_cap4096` arms for $\sigma \le 0.10$ with `1.0000` recall and `0%` under-fill; `S1_uncap_ef100` reaches higher mean QPS `36.46` at $\sigma=0.10$ with `399.162 ms p99`) |

### Opaque Closure `eval_limit` Guardrail & Charts

- `FilteredSearchParams::default().eval_limit` remains `None`; `DEFAULT_OPAQUE_EVAL_LIMIT = 16_384` (accessible via `FilteredSearchParams::with_opaque_eval_limit()`) caps runaway opaque closure evaluations (`predicate.bitset().is_none()`) without truncating typical workloads. On `1M` `independent` $\sigma = 0.10$ (`S2_ef100`), `16_384` preserves `Recall@10 = 0.998500` (`1.0%` hit rate vs `69.0%` at `10_000`); on pathological `1M` `correlated` $\sigma = 0.01$, it bounds worst-case `eval_count_max` from `196,017` (`p99 = 53.729 ms`) down to `16,384` (`p99 = 4.797 ms`, an **`11.20x` tail-latency cap**).
- Charts in [`results/phase5/charts_1m/`](results/phase5/charts_1m/) (`charts_100k/` for `100k`, and root [`results/phase5/`](results/phase5/)): `chart1_recall_vs_selectivity`, `chart2_latency_vs_selectivity`, `chart3_failure_rate_vs_selectivity`, `chart4_matched_recall_crossover`, and `chart5_distance_evals_and_sensitivity` (PNG + SVG).

---

## Phase 6 — Vector Quantization (`SQ8` & `PQ`, `100k` & `1M`)

Phase 6 introduces an isolated `#![forbid(unsafe_code)]`, `std`-only quantization module ([`src/quantized/`](src/quantized/mod.rs): `CompactHnswGraph`, `HnswSq8Index`, `HnswPqIndex`) while leaving [`src/hnsw.rs`](src/hnsw.rs) (`f32`) untouched and bit-exact (`0x80af03bd7aebaade` at `100k`, `0x44044886c62bca5e` at `1M`). Evaluated on authentic INRIA Texmex `SIFT1M` (`D=128`, `L2Squared`, `k=10`, `Q=10,000`, `seed=42`) across **`11` arms at `100k`** (`reps=3`, solo P-core `0x0004`) and **`9` Tier 1+2 arms at `1M`** (`reps=5`, 2-P-core sharded `0x0004`/`0x0400`). See [`results/phase6/PROPOSAL.md`](results/phase6/PROPOSAL.md), [`results/phase6/FINDINGS.md`](results/phase6/FINDINGS.md), and [`results/CLAIMS.md`](results/CLAIMS.md) (§12).

### Kernel Safety Audit (`T4.2` — Way A vs Way B & PQ ADC)

Measured in [`results/phase6/pilot/microbench_kernels.json`](results/phase6/pilot/microbench_kernels.json) (`4,096` L2/L3-hot 128-d vectors, `200,000` evals $\times 5$ reps, `0x0004`):

| Kernel | Bytes/Vec | Median (`ns/eval`) | Speedup vs `f32` | Std Max Rel Error | Adversarial Cancellation Max Abs Error | Default? |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| `f32_l2_squared` *(8-way unrolled)* | `512` | `39.6600` | `1.00x` | exact ref | exact ref | `f32` ref |
| `sq8_query_shifted` **(Way A)** | `128` | `47.1285` | `0.84x` | **`1.7156e-7`** | **`4.6115e-5`** | **Default** |
| `sq8_norm_expanded` *(Way B)* | `132` | `36.9335` | `1.07x` (`1.2760x` vs A) | `3.9406e-6` | **`63.9910`** | Rejected |
| `pq16_adc` *(16 subspaces $\times$ 256)* | `16` | **`9.8310`** | **`4.03x`** | — | — | Default `PQ16` |
| `pq32_adc` *(32 subspaces $\times$ 256)* | `32` | **`20.1700`** | **`1.97x`** | — | — | Default `PQ32` |

- **Why Way A Is Mandatory:** Norm-expanded Way B ($\|\tilde{\mathbf{q}}\|^2 - 2\langle \hat{\mathbf{q}}, \mathbf{c}_i\rangle + \|\mathbf{s}\odot\mathbf{c}_i\|^2$) suffers catastrophic `f32` cancellation whenever $\|\mathbf{q}-\mathbf{v}_i\| \ll \|\mathbf{q}-\mathbf{min}\|$, yielding `63.9910` absolute error (`1.39e6x` worse than Way A's `4.6115e-5`) and requiring `+4 B/vec` auxiliary norms (`3.81 MiB` at `1M`). In HNSW traversal, DRAM cache-line traffic dominates `f32` (`512 B/vec`), letting `128 B/vec` Way A outperform `f32_m16` end-to-end anyway.

### `1M` Tier 1+2 Headline Results (`SIFT1M`, `reps=5`) & `100k` Highlights

| Arm (`M=16, efC=100` unless noted) | Build (s) | Payload (`B/vec`) | Build RSS (`MiB`) | Query RSS (`MiB`) | Ceiling (`ef=600`) | `ef=100` R@10 / QPS / `p99` | Matched-Recall Highlight (`1M`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| `f32_m16` *(Phase 6, sharded)* | `574.78` | `660.91` | `642.71` | `642.71` | `0.99804` | `0.98061` / `1,676.2` / `1.1951 ms` | Phase 4 solo ref: `450.30 s`, `2,383.9 QPS`, `0.5515 ms` |
| `sq8_only_f32graph` | `490.48` | **`272.54`** (`-58.8%`) | `778.37` | **`276.02`** (`-57.1%`) | `0.98764` | `0.97196` / `2,256.4` / `0.6402 ms` | `R>=0.98`: `1,619.8 QPS` / `0.8758 ms` (`ef=150`) |
| `sq8_only_qbuild` *(streaming)* | `546.54` | **`272.54`** (`-58.8%`) | **`277.29`** (`-64.4%`) | **`276.18`** (`-57.0%`) | `0.98767` | `0.97220` / `2,145.2` / `0.6740 ms` | `< 300 MiB` build & serving RSS; matches `f32graph` (`+0.00024` @100) |
| `sq8_rerank_f32graph` (`R=100`/`20`) | `572.29` | `784.54` (`+18.7%`) | `778.41` | `778.41` | **`0.99804`** | `0.98060` / `2,172.2` / `0.6256 ms` | `R=20` restores `100%` of `f32` & `FullBeam` recall |
| `sq8_only_m32` (`M=32, efC=200`) | `1,459.99` | `400.28` (`-39.4%`) | `1,022.15` | `398.04` (`-38.1%`) | `0.98898` | `0.98567` / `939.7` / `6.3654 ms`\* | `ef=50`: `0.97409` / `2,424.2 QPS`; hits `0.99060` at `100k` |
| `pq16_only_f32graph` | `582.66` | **`160.67`** (`-75.7%`) | `779.20` | **`169.55`** (`-73.6%`) | `0.54520` *(pk)* | `0.54429` / `4,181.3` / `0.3587 ms` | Extreme RAM compression (`89.9%` graph, `16 B/vec` codes) |
| `pq32_only_f32graph` | `643.22` | `176.67` (`-73.3%`) | `768.55` | `184.67` (`-71.3%`) | `0.72305` | `0.71898` / `2,207.7` / `1.2880 ms` | Unreranked PQ hits recall saturation (`0.76461` at `100k`) |
| `pq16_rerank_f32graph` | `558.82` | `672.67` (`+1.8%`) | `778.37` | `778.37` | `0.99675` (`FB`) | `0.95032` / `3,978.2` / `0.3356 ms` | `R>=0.99` (`ef=300, FB`): **`1,553.9 QPS` (`+27.0%`) / `0.8536 ms` (`-40.2%`)** |
| `pq32_rerank_f32graph` | `643.16` | `688.67` (`+4.2%`) | `778.41` | `778.41` | **`0.99762`** (`FB`) | `0.97643` / `2,338.8` / `0.5911 ms` | `R>=0.995` (`ef=300, FB`): **`994.0 QPS` (`+49.8%` vs Ph6 `f32_m16`)** |

- **`100k` Highlights (`11` arms):** Streaming builds (`sq8_only_qbuild`, `pq16_only_qbuild`) cut `build_peak_rss_mb` by `-44.2%` (`87.70 -> 48.90 MiB`); isolated query RSS drops to `38.61 MiB` (`SQ8-Only`, `-48.5%`) and `28.13 MiB` (`PQ16-Only`, `-62.5%`). `sq8_only_m32` breaks `0.99` (`0.99060` ceiling), `SQ8+Rerank` (`R=20`) matches `100%` of `f32` recall across `ef in [20..600]`, and `pq16_rerank_f32graph` beats `f32_m16` at `>= 0.99` by `+40.9%` QPS (`5,269.0` vs `3,738.9 QPS`) / `-30.7%` `p99`.
- **Honest Caveats:**
  - *Windows `balanced` power plan & 2-core sharding:* Both `100k` (`0x0004`) and `1M` (`0x0004` / `0x0400`, `244.06` core-min inside `PROPOSAL.md`'s `252.89 min` conservative budget) ran under `power_mode = "balanced"`. Concurrent `1M` sharding caused shared L3/DRAM contention on `f32_m16` (slower than solo Phase 4 `f32_m16`) and isolated `p99` spikes (`*` `sq8_only_m32` `ef=100` hit `6.3654 ms` while overlapping `pq32_rerank_f32graph`).
  - *Isolated vs Lifetime RSS:* `Only` arms profile serving RSS in an isolated child process (`query_rss_profiled_isolated: true`); `f32_m16` and `rerank` arms genuinely retain raw `512 B/vec` vectors (`488.28 MiB` at `1M`) during query serving (`query_serving_peak_rss_mb == build_peak_rss_mb`).
  - *PQ Saturation Dips:* On unreranked PQ (and `pq16_rerank R=30`), deeper `efSearch` past saturation slightly drops recall (`0.54520 -> 0.54478` at `1M`) due to ADC ranking inversions; `R >= 50`/`FullBeam` and all `SQ8` arms are strictly monotonic.
- **Charts ([`scripts/plot_phase6.py`](scripts/plot_phase6.py)):** [`results/phase6/`](results/phase6/) (`1M` headline copy), [`results/phase6/charts_100k/`](results/phase6/charts_100k/), and [`results/phase6/charts_1m/`](results/phase6/charts_1m/):
  1. [`chart1_recall_vs_p99.png`](results/phase6/chart1_recall_vs_p99.png) ([SVG](results/phase6/chart1_recall_vs_p99.svg)) — `Recall@10` vs `p99` latency (log y-axis, `1 ms` SLA line)
  2. [`chart2_recall_vs_qps.png`](results/phase6/chart2_recall_vs_qps.png) ([SVG](results/phase6/chart2_recall_vs_qps.svg)) — `Recall@10` vs single-thread QPS (log y-axis)
  3. [`chart3_memory_vs_recall.png`](results/phase6/chart3_memory_vs_recall.png) ([SVG](results/phase6/chart3_memory_vs_recall.svg)) — Payload (`B/vec`) & query-serving Peak RSS (`MiB`) vs Recall (`p99 <= 1 ms`) + Pareto frontier
  4. [`chart4_rerank_ablation.png`](results/phase6/chart4_rerank_ablation.png) ([SVG](results/phase6/chart4_rerank_ablation.svg)) — Top-$R$ rerank recovery & latency overhead vs $R$ (`efSearch = 100`)

---

## Phase 7 — Head-to-Head vs FAISS `IndexHNSWFlat` (`100k` & `1M`)

Phase 7 compares `#![forbid(unsafe_code)]`, `std`-only Rust `HnswIndex` head-to-head against Meta C++ `faiss-cpu 1.15.1` `IndexHNSWFlat(128, M, METRIC_L2)` across `C1` (`M=8, efC=100`), `C2` (`M=16, efC=100`), `C3` (`M=32, efC=200`), and `C2-nkp` (`M=16, efC=100, keep_pruned_shrink=false`) at `100k` (`3` builds, `5` query reps, cached exact GT `data/gt_100000_k100.ivecs`) and `1M` (`1` build, `5` query reps, official `sift_groundtruth.ivecs`) across `22` run JSONs and [`results/phase7/matched_recall_summary.json`](results/phase7/matched_recall_summary.json). See [`results/phase7/FINDINGS.md`](results/phase7/FINDINGS.md) and [`results/CLAIMS.md`](results/CLAIMS.md) (§13).

> **Mandatory `SIMDLevel_NONE` Baseline Parity Disclosure:** Stock PyPI `faiss-cpu 1.15.1` on Windows ships compiled statically in **`SIMDLevel_NONE`** (`avx2_available == false`, no `swigfaiss_avx2`), matching default-target Rust (`x86_64-pc-windows-msvc` baseline SSE2 + LLVM auto-vectorization over 8-way unrolled `as_chunks::<8>()`). Results measure **baseline SSE2 parity**; explicit AVX2 (`-C target-cpu=native` vs source/conda-forge AVX2 FAISS) remains an unmeasured addendum.

### Build Time, OpenMP Asymmetry, Graph Memory & Isolated Peak RSS

| Scale | Config (`T=1` unless noted) | Rust Build (s) | FAISS Build (s) | Graph-Only (Rust vs FAISS `B/vec`) | Isolated Peak RSS (Rust vs FAISS `MiB`) |
|---:|---|---:|---:|---|---|
| `100k` | `C1` (`8, 100`) | `18.163` | `18.040` | `81.90` vs `80.55` (`+1.68%`) | **`75.96`** vs `167.89` (`2.21x` lower) |
| `100k` | `C2` (`16, 100`) | `29.926` | `20.368` | `145.57` vs `144.25` (`+0.92%`) | **`82.11`** vs `174.00` (`2.12x` lower) |
| `100k` | `C2-nkp` (`16, 100, nkp`) | `22.876` (`-23.56%`) | *(vs `C2`: `20.368`)* | `145.57` vs `144.25` | **`81.99`** vs `174.00` |
| `100k` | `C2` (`T=16` `0xffff`) | `26.977` *(1T build)* | **`3.339`** *(OMP `8.96x` vs `0x0004`)* | `145.57` vs `144.25` | **`95.23`** vs `174.34` |
| `100k` | `C3` (`32, 200`) | `79.651` | `41.807` | `273.41` vs `272.09` (`+0.49%`) | **`93.68`** vs `186.09` (`1.99x` lower) |
| `1M` | `C1` (`8, 100`) | `313.285` | `277.695` | `85.44` vs `80.57` (`+6.04%`) | **`594.31`** vs `1,118.32` (`1.88x` lower) |
| `1M` | `C2` (`16, 100`) | `625.665` *(Ph4 solo `450.301`)* | `404.120` | `148.91` vs `144.26` (`+3.22%`) | **`654.53`** vs `1,180.04` (`1.80x` lower) |
| `1M` | `C2-nkp` (`16, 100, nkp`) | `428.827` (`-31.46%`) | *(vs `C2`: `404.120`)* | `148.91` vs `144.26` | **`654.32`** vs `1,180.04` |
| `1M` | `C2` (`T=16` `0xffff`) | `520.115` *(1T build)* | **`76.360`** *(OMP `8.19x` vs `0x0004`)* | `148.91` vs `144.26` | **`719.87`** vs `1,196.62` |
| `1M` | `C3` (`32, 200`) | `1,562.147` | `782.080` | `272.33` vs `272.11` (`+0.08%`) | **`776.42`** vs `1,301.15` (`1.68x` lower) |

- **Where Rust Saves RAM vs Where FAISS Wins Builds:** Rust `HnswIndex` borrows `&RowMatrix<f32>` rather than copying a second `488.28 MiB` base matrix alongside Python/FAISS DLLs, cutting isolated Peak RSS by `~92 MiB` (`100k`) and **`~524–526 MiB` (`1M`)**. Conversely, Rust's builder is strictly single-threaded, whereas FAISS OpenMP `add()` (`T=16`) builds **`8.96x` faster at `100k`** and **`8.19x` faster at `1M`** (`6.81x` vs `0xffff` Rust `520.115 s`).

### Single-Thread Matched-Recall (`Mode 1` Pure Batch QPS & `Mode 2` `p99`, Zero Extrapolation)

| Scale | Comparison | `R* = 0.90` QPS (`p99`) | `R* = 0.95` QPS (`p99`) | `R* = 0.98` QPS (`p99`) | `R* = 0.99` QPS (`p99`) | `R* = 0.995` QPS (`p99`) |
|---:|---|---:|---:|---:|---:|---:|
| `100k` | `C1` vs FAISS `C1` | `0.8655x` (`0.9050x`) | `0.9058x` (`0.9173x`) | **`1.0392x`** (`0.9467x`) | **`1.1562x`** (**`1.1213x`**) | **`1.2582x`** (**`1.1671x`**) |
| `100k` | `C2` vs FAISS `C2` | `0.7159x` (`0.7685x`) | `0.7552x` (`0.8189x`) | `0.7987x` (`0.8599x`) | `0.7955x` (`0.8138x`) | `0.7039x` (`0.6970x`) |
| `100k` | `C2-nkp` vs FAISS `C2` | `0.7568x` (`0.8082x`) | `0.8016x` (`0.8434x`) | `0.8552x` (`0.9436x`) | `0.9208x` (`0.9447x`) | `0.9772x` (`0.9908x`) |
| `100k` | `C3` vs FAISS `C3` | *out of range* (`>0.9097`) | `0.7045x` (`0.7299x`) | `0.7368x` (`0.7674x`) | `0.7850x` (`0.8308x`) | `0.8255x` (`0.8766x`) |
| `1M` | `C1` vs FAISS `C1` | `0.8121x` (**`1.1956x`**) | `0.9224x` (`0.9984x`) | **`1.2546x`** (**`1.3793x`**) | *FAISS caps `0.98955`* | *FAISS caps `0.98955`* |
| `1M` | `C2` vs FAISS `C2` | `0.6539x` (`0.9421x`) | `0.6334x` (`0.6727x`) | `0.6241x` (`0.4827x`) | `0.6618x` (`0.3470x`) | `0.6781x` (**`1.2278x`**) |
| `1M` | `C2-nkp` vs FAISS `C2` | `0.7648x` (**`1.1524x`**) | `0.7845x` (**`1.6258x`**) | `0.8539x` (`0.9408x`) | `0.9279x` (**`1.4113x`**) | **`1.0371x`** (**`1.5661x`**) |
| `1M` | `C3` vs FAISS `C3` | `0.5023x` (`0.1978x`) | `0.5870x` (`0.4595x`) | `0.7058x` (`0.8189x`) | `0.6921x` (`0.7897x`) | `0.7223x` (`0.8178x`) |

- **Wins & Losses Summarized:**
  - **`C1` (`M=8`) Crossover:** Rust trails at `R* <= 0.95`, then overtakes FAISS at **`R* >= 0.98`** (`1,356.90` vs `1,081.52 QPS` / `1.0779` vs `1.4868 ms` `p99` at `1M R*=0.98`, reaching `0.994700` recall while FAISS caps at `0.989550`).
  - **Root Cause of Default `C2`/`C3` Gap (`C2-nkp`):** Default `HnswConfig` sets `keep_pruned_shrink -> true` (backfilling reverse-edge shrink), which saturates Layer-0 with short-range edges on `M >= 16`. Disabling it (`C2-nkp`) boosts `1M C2` QPS by `+17.0%..+53.0%`, lifts `ef=600` recall to `0.999140` (`> FAISS 0.998720`), and overtakes FAISS at **`1M R*=0.995` (`1.0371x` QPS, `1.5661x` lower `p99`)**. Default `C2` and `C3` still trail FAISS across all matched recall points (`0.5023x–0.8255x` QPS).
  - **Multi-Thread Search (`C2`, `T in {1, 6, 16}`):** On solo `1M` batch runs, Rust `search_batch_parallel` scales `4.83x–6.78x` (`T=6`, `0x0555`) and `6.10x–8.41x` (`T=16`, `0xffff`), achieving **`87.3%–93.9%` (`T=6`) and `83.5%–90.5%` (`T=16`)** of FAISS OpenMP matched-recall Mode 1 QPS (`18,608.51` at `R*=0.95`, `11,026.70` at `R*=0.98` for `T=16`).
- **Honest Caveats:**
  - *Windows `balanced` power plan & concurrent `0x0004`/`0x0400` runs:* All `22` reports ran under Windows `balanced` power mode; single-thread `100k` (`C2`/`C3`) and `1M` (`C1`/`C2`/`C3`) paired Rust (`0x0004`) and FAISS (`0x0400`) concurrently on distinct P-cores sharing L3/DRAM (raising `1M C2` build from `450.301 s` Phase 4 solo to `625.665 s` and widening `Mode 2` `p99` variance), while `C2-nkp` and `T=6`/`T=16` ran solo.
  - *FFI & Instrumentation:* Headline throughput ratios use Mode 1 pure batch (zero per-query FFI/timer overhead); FAISS Mode 2 `search_c` adds median `ffi_delta_us` of `6.3050 us/q` (`100k`) / `7.9195 us/q` (`1M`; `7.0058 us/q` combined), `12x–100x` smaller than claimed Rust `p99` margins (`95.6–766.2 us`). Rust `dist_evals_mean` is `null` (built without `--features instrument`), while FAISS `ndis`/`nhops`/`ns_per_ndis` are detailed in [`results/phase7/FINDINGS.md`](results/phase7/FINDINGS.md).
- **Charts ([`scripts/plot_phase7.py`](scripts/plot_phase7.py)):** [`results/phase7/`](results/phase7/) (combined `100k`+`1M` Charts 1–3 + `1M` Chart 4), [`results/phase7/charts_100k/`](results/phase7/charts_100k/), and [`results/phase7/charts_1m/`](results/phase7/charts_1m/):
  1. [`chart1_recall_vs_p99.png`](results/phase7/chart1_recall_vs_p99.png) ([SVG](results/phase7/chart1_recall_vs_p99.svg)) — Mode 2 `Recall@10` vs `p99` latency (log y-axis; FAISS `search_c` includes `7.23 us/q` combined median FFI delta)
  2. [`chart2_recall_vs_qps.png`](results/phase7/chart2_recall_vs_qps.png) ([SVG](results/phase7/chart2_recall_vs_qps.svg)) — Mode 1 pure-batch `Recall@10` vs single-thread QPS (zero per-query FFI/timer overhead)
  3. [`chart3_matched_recall_ratios.png`](results/phase7/chart3_matched_recall_ratios.png) ([SVG](results/phase7/chart3_matched_recall_ratios.svg)) — Matched-recall Mode 1 QPS & Mode 2 `p99` speedup ratios ($R^* \in \{0.90, 0.95, 0.98, 0.99, 0.995\}$)
  4. [`chart4_multithread_and_build.png`](results/phase7/chart4_multithread_and_build.png) ([SVG](results/phase7/chart4_multithread_and_build.svg)) — `1M C2` multi-thread batch QPS (`T in {1, 6, 16}`), build time (`Asymmetry note`), graph memory & Peak RSS


