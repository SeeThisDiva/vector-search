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

- [x] **Phase 1: Brute-Force Exact $k$-NN Ground Truth & Recall Measurement Harness** *(Current — Halted for Review)*
  - From-scratch Rust `ExactKnn` engine (`L2` squared, Euclidean, Inner Product) with canonical `(distance, id)` deterministic tie-breaking
  - Standard Texmex `.fvecs` / `.ivecs` little-endian readers & writers with full per-record header integrity checks
  - Public SIFT downloader/loader (`siftsmall` 10k $\times$ 128-d and `SIFT1M` 1M $\times$ 128-d) + deterministic seeded `Xoshiro256**` synthetic offline fallback + custom `.fvecs` support
  - Automated `Recall@k` (`Recall@1`, `Recall@5`, `Recall@10`, `Recall@100`) and `Recall@k(GT, GT) == 1.0` self-verification invariant
  - Per-query latency percentile profiler (`min`, `mean`, `p50`, `p95`, `p99`, `max`, `QPS`) & host hardware detector
  - Python + `faiss.IndexFlatL2` cross-language parity verification suite
- [ ] **Phase 2: HNSW Insert + Search, Correctness-Checked on 10k Vectors**
- [ ] **Phase 3: Scale to 1M Vectors (`SIFT1M`), Profile Memory & Build Time**
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

## Repository Structure

```text
├── Cargo.toml                 # Rust crate manifest (vector_search lib + phase1 CLI)
├── docs/
│   └── HNSW_ALGORITHM.md      # HNSW mathematical derivations, layer theory & pseudocode
├── src/
│   ├── lib.rs                 # Rust crate root (#![forbid(unsafe_code)])
│   ├── main.rs                # Rust Phase 1 CLI runner (`phase1`)
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
