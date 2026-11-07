# Context & Memory: HNSW From Scratch in Rust

> **Purpose**: This file provides complete, token-efficient memory and architectural context for Claude (or any AI assistant/engineer) to resume work on this repository immediately without scanning the entire history.

---

## 1. Project Mission & Build Roadmap

Build a production-grade **Hierarchical Navigable Small World (HNSW)** Approximate Nearest Neighbor (ANN) search engine from scratch in **Rust** (`#![forbid(unsafe_code)]`), measure recall-vs-latency curves, and benchmark head-to-head against **FAISS**.

### Build Roadmap (7 Phases — Stop After Each)
- [x] **Phase 1: Brute-Force Exact $k$-NN Ground Truth & Recall Harness** *(COMPLETED & VERIFIED)*
- [x] **Phase 2: HNSW Insert + Search, Correctness-Checked on 10k Vectors** *(COMPLETED & VERIFIED)*
- [x] **Phase 3: Scale to 1M Vectors (`SIFT1M`), Profile Memory & Build Time** *(IMPLEMENTED, AWAITING DEV'S REVIEW; do not start Phase 4 before that)*
- [ ] **Phase 4: Parameter Sweep ($M$, $efConstruction$, $efSearch$), Plot Recall@10 vs. p99 Latency**
- [ ] **Phase 5: Filtered Search with Metadata Predicates (Including Selective Filters)**
- [ ] **Phase 6: Scalar (SQ8) or Product Quantization (PQ), Measure Recall Cost**
- [ ] **Phase 7: Head-to-Head Benchmark vs. FAISS on Identical Hardware**

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
├── Cargo.toml                  # Rust crate manifest: 'vector-search' v0.1.0, binaries 'phase1', 'phase2', 'phase3'
├── docs/
│   └── HNSW_ALGORITHM.md       # Authoritative math & pseudocode spec (Algorithms 1-5)
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)])
│   ├── main.rs                 # CLI entry point for Phase 1 runner (`phase1`)
│   ├── hnsw.rs                 # HNSW multi-layer graph index (Algorithms 1-5, compact u32 flat adjacency)
│   ├── bin/phase2.rs           # CLI entry point for Phase 2 runner (`phase2`)
│   ├── bin/phase3.rs           # CLI entry point for Phase 3 SIFT1M runner (`phase3`)
│   ├── matrix.rs               # Contiguous row-major RowMatrix<f32>
│   ├── distance.rs             # 8-way unrolled L2^2, Euclidean, Inner Product kernels
│   ├── rng.rs                  # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs              # Texmex .fvecs/.ivecs I/O, SIFT downloader & Gaussian generator
│   ├── exact_knn.rs            # ExactKnn engine (single-query, parallel, timed)
│   ├── recall.rs               # Recall@k, verify_self_recall, latency percentiles
│   ├── hardware.rs             # System CPU / RAM / OS hardware detection
│   ├── benchmark.rs            # End-to-end benchmark orchestrator & JSON reporter
│   └── hnsw_phase1/            # Python reference implementation & BLAS GEMM baseline
├── tests/
│   ├── rust_phase1_tests.rs    # 16 Rust Phase 1 integration & edge-case tests (PASSING)
│   ├── rust_phase2_tests.rs    # 23 Rust Phase 2 HNSW correctness & recall tests (PASSING)
│   ├── adversarial_hnsw_tests.rs # 12 Rust HNSW stress & boundary tests (PASSING)
│   ├── adversarial_graph_fingerprint_tests.rs # 10 Rust graph fingerprint & mutation tests (PASSING)
│   ├── challenge_m1_p35_tests.rs # 8 Rust Phase 3.5 instrumentation, golden & layout tests (PASSING)
│   ├── rust_phase3_tests.rs    # 11 Rust Phase 3 report semantics, GT check & sweep tests (PASSING)
│   ├── test_phase1.py          # 23 Python reference tests (PASSING)
│   └── test_rust_parity.py     # 4 Rust <-> Python <-> FAISS parity tests (PASSING)
├── results/phase3_5/           # Phase 3.5 golden fixtures, baseline/final reports & LOG.md
├── data/                       # Contains SIFT corpora & JSON benchmark reports (gitignored)
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
   - `cargo test`: **80/80 PASS** (16 Phase 1 + 23 Phase 2 + 12 adversarial_hnsw + 10 adversarial_graph_fingerprint + 8 challenge_m1_p35 + 11 Phase 3; siftsmall tests skip if `data/*.fvecs` is absent)
   - `pytest`: **27/27 PASS** (with `faiss-cpu` installed; the FAISS parity test skips otherwise)

---

## 5. Phase 2 Verified Acceptance Results

- [x] Build HNSW index on 10,000 vectors (`M=16`, `efConstruction=100`, `m_L=1/ln(16)`).
- [x] Query 100 test vectors with $k=10$.
- [x] Monotonic recall scaling: Demonstrated that as $efSearch$ increases ($10 \to 50 \to 100 \to 200$), `Recall@10` scales monotonically from $\sim 0.80$ to $\ge 0.98+$.
- [x] Latency: mean query latency $< 0.05$ ms at Recall@10 $\ge 0.98$ (met at efSearch=20, 98.6% recall).
- [x] All tests pass under `#![forbid(unsafe_code)]` with zero warnings.

---

## 6. Key Commands for Quick Verification

```powershell
# Run all Rust unit & integration tests
cargo test

# Run Rust in release mode with clippy checks
cargo clippy --all-targets -- -D warnings

# Execute Phase 1 CLI benchmark (SIFT small)
cargo run --release --bin phase1 -- --dataset siftsmall --k 10

# Execute Phase 1 CLI benchmark (offline synthetic seeded)
cargo run --release --bin phase1 -- --dataset synthetic --num-base 10000 --num-query 100 --k 10 --seed 42

# Run all Python reference & parity tests
pytest
```

---

## 7. Critical Design Rules & Gotchas

1. **No Unsafe Code**: The Rust codebase enforces `#![forbid(unsafe_code)]`. Do not use `unsafe`.
2. **Canonical Ordering**: Whenever sorting top-$k$ results, use `(ordered_float(distance), node_id)` to guarantee bit-exact determinism across platforms.
3. **Cache-Friendly Visited Sets**: For high QPS, avoid allocating a `HashSet<u32>` per query. Use a reusable `VisitedTracker` with a query counter (`Vec<u32>` of generation tags) or a pre-allocated `BitVec`.
4. **Reference Specifications**: Always check `docs/HNSW_ALGORITHM.md` for pseudocode algorithms (Algorithms 1–5).
