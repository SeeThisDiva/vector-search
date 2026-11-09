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
- [ ] **Phase 4: Parameter Sweep (`M`, `efConstruction`, `efSearch`), Plot `Recall@10` vs `p99` Latency**
- [ ] **Phase 5: Filtered Search with Metadata Predicates (Including High-Selectivity Filters)**
- [ ] **Phase 6: Scalar or Product Quantization & Recall Cost Measurement**
- [ ] **Phase 7: Head-to-Head Benchmark vs FAISS on Identical Data & Hardware**

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
├── Cargo.toml                 # Rust crate manifest (vector_search lib + phase1/phase2/phase3/phase4 CLIs)
├── docs/
│   └── HNSW_ALGORITHM.md      # HNSW mathematical derivations, layer theory & pseudocode
├── src/
│   ├── lib.rs                 # Rust crate root (#![forbid(unsafe_code)])
│   ├── main.rs                # Rust Phase 1 CLI runner (`phase1`)
│   ├── hnsw.rs                # HNSW index (insert, search-layer, heuristic selection, k-NN)
│   ├── bin/phase2.rs          # Rust Phase 2 CLI runner (`phase2`)
│   ├── bin/phase3.rs          # Rust Phase 3 CLI runner (`phase3`, SIFT1M)
│   ├── bin/phase4.rs          # Rust Phase 4 CLI runner (`phase4`, parameter sweep)
│   ├── matrix.rs              # Contiguous row-major RowMatrix<T> storage
│   ├── distance.rs            # 8-way unrolled L2^2, Euclidean, and Inner Product kernels
│   ├── rng.rs                 # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs             # Texmex .fvecs/.ivecs I/O, SIFT downloader & seeded generator
│   ├── exact_knn.rs           # Brute-force ExactKnn (single-query, timed, & multi-threaded)
│   ├── recall.rs              # Recall@k harness, self-verification & latency percentiles
│   ├── hardware.rs            # Cross-platform CPU / RAM / OS hardware probe
│   ├── benchmark.rs           # End-to-end Phase 1 benchmark pipeline & JSON reporter
│   ├── phase4.rs              # Phase 4 resumable parameter sweep runner & prefix GT caching
│   └── hnsw_phase1/           # Python reference implementation & FAISS comparison helpers
├── tests/
│   ├── rust_phase1_tests.rs   # Rust integration & edge-case test suite (16 tests)
│   ├── rust_phase2_tests.rs   # HNSW correctness, topology & recall tests (23 tests)
│   ├── adversarial_hnsw_tests.rs # HNSW stress / boundary / reproducibility tests (12 tests)
│   ├── adversarial_graph_fingerprint_tests.rs # Logical graph fingerprint & mutation tests (10 tests)
│   ├── challenge_m1_p35_tests.rs # Phase 3.5 instrumentation, golden & layout tests (8 tests)
│   ├── rust_phase3_tests.rs   # Phase 3 report semantics + parallel-search input validation (11 tests)
│   ├── rust_phase4_tests.rs   # Phase 4 grid parsing, GT caching & resumable sweep tests (4 tests)
│   ├── test_phase1.py         # Python reference unit tests (23 tests)
│   └── test_rust_parity.py    # Cross-language Rust <-> Python <-> FAISS parity tests (4 tests)
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

Requires Rust >= 1.88. Tests that need the real `siftsmall` corpus skip themselves when `data/siftsmall/*.fvecs` is missing (run the Phase 1 command above once to download it).

```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

### 3. Run Cross-Language Python & FAISS Parity Verification

```bash
pip install -r requirements.txt
python -m pytest tests/ -v --cov=src
ruff check .
mypy src tests
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

