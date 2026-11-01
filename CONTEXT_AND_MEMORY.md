# Context & Memory: HNSW From Scratch in Rust

> **Purpose**: This file provides complete, token-efficient memory and architectural context for Claude (or any AI assistant/engineer) to resume work on this repository immediately without scanning the entire history.

---

## 1. Project Mission & Build Roadmap

Build a production-grade **Hierarchical Navigable Small World (HNSW)** Approximate Nearest Neighbor (ANN) search engine from scratch in **Rust** (`#![forbid(unsafe_code)]`), measure recall-vs-latency curves, and benchmark head-to-head against **FAISS**.

### Build Roadmap (7 Phases — Stop After Each)
- [x] **Phase 1: Brute-Force Exact $k$-NN Ground Truth & Recall Harness** *(COMPLETED & VERIFIED)*
- [ ] **Phase 2: HNSW Insert + Search, Correctness-Checked on 10k Vectors** *(<-- RESUME HERE)*
- [ ] **Phase 3: Scale to 1M Vectors (`SIFT1M`), Profile Memory & Build Time**
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
- **Working Directory**: `c:\Users\Dev Gupta\OneDrive\Documents\Project\vector search`

---

## 3. Current Repository Architecture & File Map

```text
├── Cargo.toml                  # Rust crate manifest: 'vector-search' v0.1.0, binary 'phase1'
├── docs/
│   └── HNSW_ALGORITHM.md       # 513-line authoritative math & pseudocode spec (Algorithms 1-5)
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)])
│   ├── main.rs                 # CLI entry point for Phase 1 runner (`phase1`)
│   ├── matrix.rs               # Contiguous row-major RowMatrix<f32>
│   ├── distance.rs             # 8-way unrolled L2^2, Euclidean, Inner Product kernels
│   ├── rng.rs                  # Deterministic SplitMix64 + Xoshiro256** PRNG
│   ├── dataset.rs              # Texmex .fvecs/.ivecs I/O, SIFT downloader & Gaussian generator
│   ├── exact_knn.rs            # ExactKnn engine (single-query, parallel Rayon, timed)
│   ├── recall.rs               # Recall@k, verify_self_recall, latency percentiles
│   ├── hardware.rs             # System CPU / RAM / OS hardware detection
│   ├── benchmark.rs            # End-to-end benchmark orchestrator & JSON reporter
│   └── hnsw_phase1/            # Python reference implementation & BLAS GEMM baseline
├── tests/
│   ├── rust_phase1_tests.rs    # 16 Rust integration & edge-case tests (PASSING)
│   ├── test_phase1.py          # 23 Python reference tests (PASSING)
│   └── test_rust_parity.py     # 4 Rust <-> Python <-> FAISS parity tests (PASSING)
├── data/                       # Contains base.fvecs, query.fvecs, groundtruth.ivecs, groundtruth.fvecs
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
   - `cargo test`: **16/16 PASS** (0.68s)
   - `pytest`: **27/27 PASS** (2.21s)

---

## 5. Next Step: Phase 2 Detailed Implementation Guide

### Objective
Implement the core **HNSW index** (insert + approximate search) from scratch in Rust, and verify recall against the Phase 1 brute-force ground truth on 10,000 128-d vectors (`siftsmall` or synthetic).

### Required Architecture & Components for Phase 2

1. **Graph Representation (`src/hnsw/graph.rs` or `src/hnsw.rs`)**:
   - Layered adjacency list: Each vector ID ($0 \dots N-1$) has an assigned top level $l_{max}$.
   - For each level $l \in [0, l_{max}]$, store neighbor IDs: `Vec<u32>` (capped at $M_{max}$ for $l > 0$, $M_{max0} = 2M$ for $l = 0$).
   - Global state: `enter_point: Option<u32>`, `max_level: usize`.

2. **Layer Sampling (`src/rng.rs` / `src/hnsw.rs`)**:
   - Formula: $l = \lfloor -\ln(U) \cdot m_L \rfloor$ where $U \sim (0, 1)$ via `Xoshiro256**` and $m_L = 1.0 / \ln(M)$.

3. **Routing & Beam Search (`SEARCH-LAYER`)**:
   - Priority queues:
     - $C$: Min-Heap of `(distance, node_id)` (candidates to visit).
     - $W$: Max-Heap of `(distance, node_id)` (current best $ef$ elements).
     - $V$: Visited set (`BitSet` or `Vec<bool>` sized to capacity $N$ for fast $O(1)$ lookups without allocations).
   - Early exit condition: If closest candidate $c \in C$ has $d(c, q) > \text{worst}(W)$, break.

4. **Neighbor Selection (`SELECT-NEIGHBORS-HEURISTIC`)**:
   - Given candidate pool $C$ found by `SEARCH-LAYER(q, ef=efConstruction)`:
   - Sort candidates ascending by distance to $q$.
   - Iterate candidates $e$: Add $e$ to selected set $R$ iff $\forall n \in R, d(e, q) < d(e, n)$.
   - Shrink bidirectional connections of added neighbors if their degree exceeds $M_{layer}$.

5. **K-NN Search (`src/hnsw/search.rs`)**:
   - Top-down greedy walk with $ef=1$ from $l_{top}$ down to layer 1:
     - At each layer $l \ge 1$, greedily step to the closest neighbor of current entry point until no neighbor is closer to $q$.
   - Base layer search: Run `SEARCH-LAYER` at layer 0 with $ef = efSearch$.
   - Return top-$k$ nearest elements sorted ascending by distance.

### Acceptance Criteria for Phase 2
- [ ] Build HNSW index on 10,000 vectors (`M=16`, `efConstruction=100`, `m_L=1/ln(16)`).
- [ ] Query 100 test vectors with $k=10$.
- [ ] Monotonic recall scaling: Demonstrate that as $efSearch$ increases ($10 \to 50 \to 100 \to 200$), `Recall@10` scales monotonically from $\sim 0.80$ to $\ge 0.98+$.
- [ ] Latency speedup: Query latency must be $< 0.05$ ms/query (an order of magnitude faster than brute-force $\sim 0.25$ ms/query).
- [ ] All tests pass under `#![forbid(unsafe_code)]` with zero warnings.

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
