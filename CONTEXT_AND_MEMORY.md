# Context & Memory: HNSW From Scratch in Rust

> **Purpose**: This file provides complete, token-efficient memory and architectural context for Claude (or any AI assistant/engineer) to resume work on this repository immediately without scanning the entire history.

---

## 1. Project Mission & Build Roadmap

Build a production-grade **Hierarchical Navigable Small World (HNSW)** Approximate Nearest Neighbor (ANN) search engine from scratch in **Rust** (`#![forbid(unsafe_code)]`), measure recall-vs-latency curves, and benchmark head-to-head against **FAISS**.

### Build Roadmap (7 Phases — Stop After Each)
- [x] **Phase 1: Brute-Force Exact $k$-NN Ground Truth & Recall Harness** *(COMPLETED & VERIFIED)*
- [x] **Phase 2: HNSW Insert + Search, Correctness-Checked on 10k Vectors** *(COMPLETED & VERIFIED)*
- [x] **Phase 3: Scale to 1M Vectors (`SIFT1M`), Profile Memory & Build Time** *(COMPLETED & VERIFIED — Gate G0)*
- [x] **Phase 3.5: Memory Optimization & Adjacency Flattening** *(COMPLETED & VERIFIED — Gate G1)*
- [x] **Phase 4: Parameter Sweep ($M$, $efConstruction$, $efSearch$), Plot Recall@10 vs. p99 Latency** *(COMPLETED & VERIFIED — formal Gate G3 tick in `AGENT_PLAN.md` optional)*
- [x] **Phase 5: Filtered Search with Metadata Predicates (Including Selective Filters)** *(Milestone M3 `T3.1`–`T3.11` COMPLETED & VERIFIED)*
- [ ] **Phase 6: Scalar (SQ8) or Product Quantization (PQ), Measure Recall Cost** *(Draft `results/phase6/DESIGN.md` ready — awaiting **Gate G5**)*
- [ ] **Phase 7: Head-to-Head Benchmark vs. FAISS on Identical Hardware** *(Draft `results/phase7/DESIGN.md` ready — awaiting **Gate G6**)*

---

## 2. Environment & Machine Profile

- **Operating System**: Windows 11 / x86_64 (PowerShell)
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores)
- **RAM**: ~16 GB (15.8 GB usable)
- **Rust Toolchain**: `rustc` / `cargo` 1.99.1 (`#![forbid(unsafe_code)]`)
- **Python Environment**: Python 3.13.5 (`numpy`, `faiss-cpu`, `pytest`, `pytest-cov`, `ruff`, `mypy`)
---

## 3. Current Repository Architecture & File Map

```text
├── Cargo.toml                  # Rust crate manifest: 'vector-search' v0.1.0, binaries phase1..5, generate_metadata, filtered_exact_knn
├── docs/
│   └── HNSW_ALGORITHM.md       # Authoritative math & pseudocode spec (Algorithms 1-5)
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)])
│   ├── main.rs                 # CLI entry point for Phase 1 runner (`phase1`)
│   ├── hnsw.rs                 # HNSW multi-layer graph index (Algorithms 1-5, filtered search, compact u32 flat adjacency)
│   ├── bin/phase2.rs           # CLI entry point for Phase 2 runner (`phase2`)
│   ├── bin/phase3.rs           # CLI entry point for Phase 3 SIFT1M runner (`phase3`)
│   ├── bin/phase4.rs           # CLI entry point for Phase 4 parameter sweep runner (`phase4`)
│   ├── bin/phase5.rs           # CLI entry point for Phase 5 filtered benchmark CLI (`phase5`)
│   ├── bin/generate_metadata.rs # Deterministic Mode A/B predicate bitset CLI
│   ├── bin/filtered_exact_knn.rs # Multi-threaded exact filtered k-NN CLI
│   ├── matrix.rs               # Contiguous row-major RowMatrix<f32>
│   ├── distance.rs             # 8-way unrolled L2^2, Euclidean, Inner Product kernels
│   ├── rng.rs                  # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs              # Texmex .fvecs/.ivecs I/O, SIFT downloader, Gaussian & Mode A/B generators
│   ├── exact_knn.rs            # ExactKnn engine (single-query, parallel, filtered, timed)
│   ├── filter.rs               # DenseBitset, metadata predicates & FilteredSearchParams
│   ├── recall.rs               # Recall@k, verify_self_recall, latency percentiles
│   ├── hardware.rs             # System CPU / RAM / OS hardware detection
│   ├── benchmark.rs            # End-to-end benchmark orchestrator & JSON reporter
│   ├── phase4.rs               # Phase 4 resumable parameter sweep runner & prefix GT caching
│   ├── phase5.rs               # Phase 5 Option C matrix, versioned GT/bitset cache & sensitivity
│   └── hnsw_phase1/            # Python reference implementation & BLAS GEMM baseline
├── scripts/
│   ├── plot_phase4.py          # Phase 4 Pareto & scaling chart generator
│   └── plot_phase5.py          # Phase 5 filtered recall, latency & crossover chart generator
├── tests/
│   ├── rust_phase1_tests.rs    # 16 Rust Phase 1 integration & edge-case tests
│   ├── rust_phase2_tests.rs    # 23 Rust Phase 2 HNSW correctness & recall tests
│   ├── adversarial_hnsw_tests.rs # 12 Rust HNSW stress & boundary tests
│   ├── adversarial_graph_fingerprint_tests.rs # 10 Rust graph fingerprint & mutation tests
│   ├── challenge_m1_p35_tests.rs # 8 Rust Phase 3.5 instrumentation, golden & layout tests
│   ├── rust_phase3_tests.rs    # 11 Rust Phase 3 report semantics, GT check & sweep tests
│   ├── rust_phase4_tests.rs    # 4 Rust Phase 4 grid parsing, GT caching & sweep tests
│   ├── rust_metadata_tests.rs  # 10 Rust Mode A/B predicate tests
│   ├── rust_filtered_knn_tests.rs # 7 Rust filtered exact k-NN tests
│   ├── rust_filtered_strategy1_tests.rs # 14 Rust S1 post-filter tests
│   ├── rust_filtered_strategy2_tests.rs # 13 Rust S2 in-traversal & eval_limit tests
│   ├── rust_filtered_strategy3_tests.rs # 14 Rust S3 pre-filter bitset tests
│   ├── rust_filtered_counters_tests.rs # 3 Rust instrumented filtered counter tests
│   ├── rust_phase5_tests.rs    # 4 Rust Phase 5 cache, crossover & matrix CLI tests
│   ├── test_phase1.py          # 23 Python reference tests
│   ├── test_rust_parity.py     # 4 Rust <-> Python <-> FAISS parity tests
│   ├── test_filtered_parity.py # 5 Filtered search & FAISS parity tests
│   └── test_phase5_pipeline.py # 1 End-to-end Phase 5 CLI + resume + plot test
├── results/
│   ├── phase3_5/               # Phase 3.5 golden fixtures, baseline/final reports & LOG.md
│   ├── phase4/                 # Phase 4 JSON reports, Pareto charts, PROPOSAL.md, G3_CHECKLIST.md
│   ├── phase5/                 # Phase 5 100k/1M JSONs, sensitivity, charts, DESIGN.md, PROPOSAL.md, FINDINGS.md
│   ├── phase6/DESIGN.md        # Draft Gate G5 design spec (SQ8 + PQ stretch goal)
│   └── phase7/DESIGN.md        # Draft Gate G6 protocol spec (Rust vs FAISS IndexHNSWFlat)
├── data/                       # SIFT corpora & gitignored Phase 4/5 GT/bitset caches
└── run_phase1.py               # Python reference runner
```

---

## 4. Phase 1 Accomplishments & Verified Invariants

1. **Algorithm Specification**: Written in `docs/HNSW_ALGORITHM.md` covering:
   - Skip-list hierarchy in metric spaces.
   - Probabilistic layer decay: $m_L = 1 / \ln M$, $l_{max} = \lfloor -\ln(U) \cdot m_L \rfloor$.
   - Degree bounds: $M$ for upper layers, $M_{max0} = 2M$ for layer 0.
   - Top-down 1-greedy zoom-in ($ef=1$) and layer 0 beam search with priority queues.
   - Small-world properties and $\mathcal{O}(\log N)$ logarithmic scaling proof.
   - Heuristic neighbor diversity selection (triangular inequality pruning).
2. **Deterministic Tie-Breaking**: Canonical `(distance, id)` ordering across all engines to prevent platform-dependent ground truth drift.
3. **Distance Precision**: Clamping to $\ge 0.0$ to eliminate floating-point non-associativity rounding negatives.
4. **Texmex I/O**: Bit-exact `.fvecs` (float32) and `.ivecs` (int32) readers/writers.
5. **SIFT Dataset Support**: Auto-downloads authentic INRIA Texmex `siftsmall` (10k vectors) / `SIFT1M` (1M vectors), with seeded Gaussian fallback (`seed=42`).
6. **Self-Verification Invariant**: `Recall@k(GT, GT) == 1.000000` (100.0%) strictly verified.
7. **FAISS Parity Alignment**: `1000/1000` (100.0%) exact neighbor ID match against `faiss.IndexFlatL2`.
8. **Test Suite Status**:
   - `cargo test`: **84/84 PASS** (16 Phase 1 + 23 Phase 2 + 12 adversarial_hnsw + 10 adversarial_graph_fingerprint + 8 challenge_m1_p35 + 11 Phase 3 + 4 Phase 4; siftsmall tests skip if `data/*.fvecs` is absent)
   - `pytest`: **27/27 PASS** (with `faiss-cpu` installed; the FAISS parity test skips otherwise)

---

## 5. Phase 2 Verified Acceptance Results

- [x] Build HNSW index on 10,000 vectors (`M=16`, `efConstruction=100`, `m_L=1/ln(16)`).
- [x] Query 100 test vectors with $k=10$.
- [x] Monotonic recall scaling: Demonstrated that as $efSearch$ increases ($10 \to 50 \to 100 \to 200$), `Recall@10` scales monotonically from $\sim 0.80$ to $\ge 0.98+$.
- [x] Latency: mean query latency $< 0.05$ ms at Recall@10 $\ge 0.98$ (met at efSearch=20, 98.6% recall).
- [x] All tests pass under `#![forbid(unsafe_code)]` with zero warnings.

---

## 6. Phase 3 & 3.5 (Milestone M1) Verified Results

- [x] **Scale to 1M Vectors (`SIFT1M`)**: Built index on 1,000,000 128-d vectors in 450.3 s (2,220.7 vec/s).
- [x] **Bit-Exact Ground Truth Alignment**: Official Texmex `sift_groundtruth.ivecs` verified with `999/1000` ID overlap (1 distance boundary tie) and 100% boundary distance compliance.
- [x] **Adjacency Flattening & Memory Breakthrough (`T1.7`)**: Replaced pointer-heavy nested structures with flat contiguous `u32` slice storage (`layer0` stride `m_max0 + 1 = 33`, `upper_arena` stride `m_max + 1 = 17`).
  - Graph-only memory dropped **56.0%** from `338.67 B/vec` to **`148.91 B/vec`** (`-180.97 MiB` at 1M).
  - Isolated 1M peak RSS dropped from `845.75 MiB` to **`646.05 MiB`** (`-199.70 MiB`).
  - 100k build time dropped **10.37%** (`30.3 s -> 27.2 s`).
  - Preserved **100% bit-exact** graph fingerprints and query results.
- [x] **Gate G0 and G1 Signed Off**: Zero-cost instrumentation (`--features instrument`), reproducible 64-bit graph fingerprinting, and strict provenance tracking merged.

---

## 7. Phase 4 (Milestone M2) Verified Results & Current Status

- [x] **100k Wide Parameter Sweep (`T2.3`)**: Swept 23 configurations ($M \in \{8, 12, 16, 24, 32, 48\}$, $\text{efC} \in \{40, 100, 200, 400\}$ with $\text{efC} \ge M$) across a 12-point `efSearch` ladder $\{10..600\}$ on cached prefix ground truth.
- [x] **Publication-Grade Pareto Visualization (`T2.4`)**: Implemented `scripts/plot_phase4.py` (`matplotlib` only, 100% typed, 0 lints) producing 4 clean charts in PNG and SVG under `results/phase4/`:
  - `chart1_recall_vs_p99`: Recall@10 vs p99 query latency (log scale) with empirical Pareto frontier.
  - `chart2_recall_vs_qps`: Recall@10 vs single-thread QPS.
  - `chart3_build_time_vs_recall`: Construction throughput vs achievable recall at fixed latency.
  - `chart4_memory_vs_m`: Graph memory bytes per vector vs $M$ demonstrating strict scale invariance.
- [x] **1M Vector Confirmation (`T2.5`)**: Benchmarked 4 empirical Pareto points on full SIFT1M with 5 timed passes pinned to P-core `0x0004`:
  - $M=8, \text{efC}=100$: 266.5 s build, 85.4 B/vec graph, 20,150.7 QPS at ef=10, 0.9947 Recall@10 at ef=600.
  - $M=16, \text{efC}=100$ (Phase 3 baseline): 450.3 s build, 148.9 B/vec graph, 0.9806 Recall@10 at ef=100 (2,383.9 QPS, 0.55 ms p99). Bit-exact graph fingerprint `0x44044886c62bca5e`.
  - $M=32, \text{efC}=200$: 1,288.9 s build, 272.3 B/vec graph, 0.9828 Recall@10 at ef=50 (2,579.7 QPS), 0.9988 at ef=200.
  - $M=48, \text{efC}=400$: 2,828.0 s build, 402.4 B/vec graph, 0.9917 Recall@10 at ef=50 (1,918.8 QPS), 0.9992 at ef=200.
- [x] **Documentation & Claims Ledger (`T2.6`)**: Authored Phase 4 section in `README.md` and registered numbers in `results/CLAIMS.md` Section 10.

---

## 7.5. Phase 5 (Milestone M3) Verified Results & Next Gates (`G5` / `G6`)

- [x] **Filtered Search & Provenance (`T3.1`–`T3.8`)**: Implemented `DenseBitset`, Mode A/B deterministic generators, `S1`/`S2`/`S3` filtered strategies, zero-cost `FilteredSearchCounters`, `DEFAULT_OPAQUE_EVAL_LIMIT = 16_384` (`FilteredSearchParams::default().eval_limit == None`), and versioned disk caches under `data/phase5/`.
- [x] **Option C Matrix & Sensitivity (`T3.9`)**: Executed all 24 `100k` cells + sensitivity (`a4621bc`, solo `0x0004`) and 24 `1M` cells + sensitivity (`c39c92b`, `0x0004/0x0010/0x0040/0x0100`). Mode A $\sigma^*_{0.95}$ shifts `0.079149` (`100k`) $\to$ `0.023799` (`1M`); Mode B shifts `0.252372` (`100k`) $\to$ `0.120868` (`1M`); `16_384` bounds `1M` Mode B `0.01` opaque `p99` from `53.729 ms` to `4.797 ms` (`11.20x`).
- [x] **Visualization & Claims Synthesis (`T3.10`–`T3.11`)**: Authored [`results/phase5/FINDINGS.md`](results/phase5/FINDINGS.md), generated PNG/SVG charts via `scripts/plot_phase5.py`, updated `README.md`, and logged Section 11 in `results/CLAIMS.md`.
- [ ] **Current Gate Status**: Drafted **`results/phase6/DESIGN.md`** (**Gate G5**) and **`results/phase7/DESIGN.md`** (**Gate G6**); paused awaiting Dev sign-off before M4/M5 implementation begins.

---

## 8. Key Commands for Quick Verification

```powershell
# Run all Rust unit & integration tests (184 tests) and instrumented checks (11 tests)
cargo test --release
cargo test --release --features instrument --test challenge_m1_p35_tests --test rust_filtered_counters_tests

# Run Rust in release mode with clippy checks
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features instrument -- -D warnings

# Check code formatting
cargo fmt --all -- --check

# Run all Python reference & parity tests (33 tests, 0 skipped)
python -m pytest tests -q -rs

# Run Python lint & type checks
ruff check . ; mypy src tests scripts

# Execute Phase 5 filtered benchmark matrix
cargo run --release --bin phase5 -- --matrix --scale 100000 --reps 5 --include-sensitivity --output-dir results/phase5
```

---

## 9. Critical Design Rules & Gotchas

1. **No Unsafe Code**: The Rust codebase enforces `#![forbid(unsafe_code)]`. Do not use `unsafe`.
2. **Canonical Ordering**: Whenever sorting top-$k$ results, use `(ordered_float(distance), node_id)` to guarantee bit-exact determinism across platforms.
3. **Cache-Friendly Visited Sets**: For high QPS, avoid allocating a `HashSet<u32>` per query. Use a reusable `VisitedTracker` with a query counter (`Vec<u32>` of generation tags) or a pre-allocated `BitVec`.
4. **Reference Specifications**: Always check `docs/HNSW_ALGORITHM.md` for pseudocode algorithms (Algorithms 1–5).
