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
| 10  | 95.1%  | 0.024 | 0.035 | 10.3x |
| 20  | 98.6%  | 0.042 | 0.060 | 5.9x  |
| 50  | 100.0% | 0.066 | 0.100 | 3.7x  |
| 100 | 100.0% | 0.108 | 0.147 | 2.3x  |
| 200 | 100.0% | 0.186 | 0.254 | 1.3x  |

Acceptance target: some efSearch reaches Recall@10 >= 0.98 at < 0.05 ms mean latency (met at efSearch=20). `siftsmall` is easy (recall saturates by efSearch=50), so treat these as a correctness check; Phase 3 on `SIFT1M` is the real test.

```bash
cargo run --release --bin phase2 -- --dataset siftsmall --json-out data/phase2_report.json
```

---

## Phase 3 Results (`SIFT1M` 1,000,000 $\times$ 128-d, `M=16`, `efConstruction=100`, $k=10$)

Measured on `12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)`, official Texmex ground truth (verified against brute-force `ExactKnn` on 100 queries), 10,000 queries, single-threaded search, **5 repetitions per `efSearch`** (reporting median and min–max spread). Source: `data/phase3_report.json` and `data/baseline_profile_report.json` (gitignored, regenerate with the command below).

| efSearch | Recall@10 | Median Mean (ms) | Mean Spread (ms) | Median p50 (ms) | Median p99 (ms) | p99 Spread (ms) | Median QPS |
|---------:|----------:|-----------------:|-----------------:|----------------:|----------------:|----------------:|-----------:|
| 10  | 73.315% | 0.084 | 0.002 | 0.082 | 0.141 | 0.008 | 11,927.2 |
| 20  | 84.949% | 0.126 | 0.004 | 0.126 | 0.195 | 0.012 | 7,904.3 |
| 50  | 94.703% | 0.248 | 0.002 | 0.254 | 0.347 | 0.010 | 4,024.2 |
| 100 | 98.061% | 0.440 | 0.011 | 0.456 | 0.588 | 0.045 | 2,270.0 |
| 200 | 99.376% | 0.810 | 0.015 | 0.841 | 1.084 | 0.036 | 1,234.5 |

| Metric | Value |
|--------|-------|
| Build time | 471.0 s (2,123.3 vectors/s), top layer 5, entry point `#829262` |
| Raw vectors | 512.00 B/vector (488.28 MiB) |
| Graph-only (excl. visited tags & scratch) | 338.67 B/vector (322.98 MiB) |
| Graph total (incl. 4 B/vector visited tags) | 342.67 B/vector (326.79 MiB) |
| Total index footprint | 854.67 B/vector (815.08 MiB) |
| Peak working set after build (1M) | 845.84 MiB |

### Multi-Scale Baseline Profile (Isolated Process per Scale, `measured_this_run = true`)

| Scale ($N$) | Build Time (s) | Build QPS | Isolated Peak RSS (MiB) | Graph-Only (B/vec) | Graph Total (B/vec) | Total (B/vec) | Recall@10 (`ef=50`) |
|------------:|---------------:|----------:|------------------------:|-------------------:|--------------------:|--------------:|--------------------:|
| 10,000 | 1.440 | 6,942.5 | 13.70 | 338.98 | 342.98 | 854.98 | 99.750% |
| 100,000 | 28.611 | 3,495.2 | 88.67 | 338.66 | 342.66 | 854.66 | 98.140% |
| 1,000,000 | 470.966 | 2,123.3 | 845.84 | 338.67 | 342.67 | 854.67 | 94.703% |

Notes:
- Each scale (`10k`, `100k`, `1M`) is profiled in its own isolated OS process loading only `scale` prefix rows from `sift_base.fvecs`, so `PeakWorkingSet64` reflects only that scale's true memory footprint.
- `official_ground_truth_matched` runs brute-force `ExactKnn` on the first 100 queries against the full 1M base (`999/1000` identical IDs and `1` exact integer $L_2^2$ distance tie at the $k=10$ boundary, with 100% of official top-10 neighbors verified to have distance $\le$ the true 10th nearest-neighbor distance).
- Recall at fixed `efSearch=50` drops with scale (`99.75%` at 10k, `98.14%` at 100k, `94.70%` at 1M), so `efSearch` must grow with $N$.

```bash
cargo run --release --bin phase3 -- --dataset sift1m --json-out data/phase3_report.json
```

---

## Repository Structure

```text
├── Cargo.toml                 # Rust crate manifest (vector_search lib + phase1/phase2/phase3 CLIs)
├── docs/
│   └── HNSW_ALGORITHM.md      # HNSW mathematical derivations, layer theory & pseudocode
├── src/
│   ├── lib.rs                 # Rust crate root (#![forbid(unsafe_code)])
│   ├── main.rs                # Rust Phase 1 CLI runner (`phase1`)
│   ├── hnsw.rs                # HNSW index (insert, search-layer, heuristic selection, k-NN)
│   ├── bin/phase2.rs          # Rust Phase 2 CLI runner (`phase2`)
│   ├── bin/phase3.rs          # Rust Phase 3 CLI runner (`phase3`, SIFT1M)
│   ├── matrix.rs              # Contiguous row-major RowMatrix<T> storage
│   ├── distance.rs            # 8-way unrolled L2^2, Euclidean, and Inner Product kernels
│   ├── rng.rs                 # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs             # Texmex .fvecs/.ivecs I/O, SIFT downloader & seeded generator
│   ├── exact_knn.rs           # Brute-force ExactKnn (single-query, timed, & multi-threaded)
│   ├── recall.rs              # Recall@k harness, self-verification & latency percentiles
│   ├── hardware.rs            # Cross-platform CPU / RAM / OS hardware probe
│   ├── benchmark.rs           # End-to-end Phase 1 benchmark pipeline & JSON reporter
│   └── hnsw_phase1/           # Python reference implementation & FAISS comparison helpers
├── tests/
│   ├── rust_phase1_tests.rs   # Rust integration & edge-case test suite
│   ├── rust_phase2_tests.rs   # HNSW correctness, topology & recall tests
│   ├── adversarial_hnsw_tests.rs # HNSW stress / boundary / reproducibility tests
│   ├── rust_phase3_tests.rs   # Phase 3 report semantics + parallel-search input validation
│   ├── test_phase1.py         # Python reference unit tests
│   └── test_rust_parity.py    # Cross-language Rust <-> Python <-> FAISS parity tests
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
