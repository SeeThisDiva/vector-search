# Milestone M4 / Phase 6: Vector Quantization (SQ8 & PQ) Design Specification

> **Document Status**: DRAFT design specification for Dev's **Gate G5** review (not signed off). Architectural estimates, byte-layout calculations, and expected trade-offs in this document are explicitly marked as **estimates or hypotheses** until traced to a clean-commit JSON report from a real run (`AGENT_PLAN.md` Ground Rule 3).  
> **Target Release**: `vector-search` v0.1.0 (Milestone M4 / Phase 6)  
> **Engineering Invariants**: 100% Safe Rust (`#![forbid(unsafe_code)]`), standard library only (`std`) for the library crate, strict preservation of the `f32` `HnswIndex` graph and search behavior, zero heap allocations inside query inner loops, deterministic `(distance, id)` tie-breaking, and clean-commit JSON provenance.

---

## 1. Executive Summary & Architectural Scope

### 1.1 Motivation: Memory Bottleneck of `f32` HNSW
In Milestones M1 and M2 (Phases 3.5 and 4), compact `u32` flat adjacency storage (`T1.7`) cut HNSW graph-only memory at `1M` (`SIFT1M`, $D=128$, $M=16$) down to **`148.91 B/vector`** (`142.0144 MiB`), leaving the raw `f32` vector matrix (`512.00 B/vector`, `488.2813 MiB`) as **77.5%** of index payload memory and driving measured process peak RSS to **`647.26 MiB`** (`results/phase4/scale_1000000_m16_efc100.json`, commit `cd48e7f`).

At $N = 10^6$ vectors, `512 B/vector` is comfortably in-RAM on a 16 GB laptop, yet it already dominates memory bandwidth during random-access graph traversal (each neighbor evaluation touches `8` cache lines of `f32` data) and determines the scaling ceiling for larger corpora. **Vector quantization** compresses each base vector into compact `u8` codes:
- **Scalar Quantization (`SQ8`)**: maps each dimension linearly into `u8` (`128 B/vector` for $D=128$, a **4x vector compression**), cutting estimated `SQ8-Only` index payload (`M=16`) from `660.91 B/vector` to **`~276.91 B/vector`** (**~58.1% payload reduction**, ~55–60% RAM reduction vs `647.26 MiB` peak RSS).
- **Product Quantization (`PQ`, Stretch Goal)**: splits $D=128$ into $M_{\text{pq}} \in \{16, 32\}$ subspaces with $K=256$ centroids per subspace (`16 B/vector` or `32 B/vector`, **32x** or **16x** vector compression), reducing vector payload below 10–18% of total memory and converting inner-loop distance evaluation into L1-resident lookup-table additions (Asymmetric Distance Computation, ADC).

### 1.2 Mission Objectives
1. **Strict Separation from `f32` `HnswIndex`**: Keep [`src/hnsw.rs`](../../src/hnsw.rs) (`HnswIndex<'a>`) and all Phase 1–5 `f32` paths untouched and verified bit-identical (`PRESERVING` non-regression guarantee). All quantized indexes live in dedicated types (`HnswSq8Index`, `HnswPqIndex`) inside a new `src/quantized/` module.
2. **SQ8 Architecture (Primary)**:
   - Per-dimension min/max calibration (`mins` + `scales` of length $D$).
   - Safe `#![forbid(unsafe_code)]`, `std`-only 8-way unrolled (`as_chunks::<8>()`) asymmetric `f32` query vs `u8` code $L_2^2$ and symmetric `u8`-vs-`u8` $L_2^2$ kernels with zero per-dimension float division in the inner loop.
   - Two explicit storage modes: **(a) Memory-First (`Sq8Only`)**, which drops raw `f32` vectors post-build (and supports two-pass streaming `.fvecs` build so even build-time Peak RSS drops by ~55–60%), and **(b) Accuracy-First (`Sq8WithRerank`)**, which retains raw `f32` vectors to rerank the top-$R$ quantized candidates.
   - Explicit empirical comparison between **graph built on `f32` then quantized** (`Float32GraphThenQuantize`) and **graph built directly on quantized distances** (`QuantizedDirect`).
3. **PQ Stretch Goal (for Gate G5 Choice)**:
   - Deterministic `Xoshiro256**` (`DeterministicRng`) seeded $k$-means++ initialization + Lloyd iterations from scratch in `std` Rust.
   - $M_{\text{pq}} \in \{16, 32\}$, $K = 256$ (`16 B/vec` and `32 B/vec` at $D=128$), precomputed query ADC lookup tables (`[f32; 256]` per subspace, fitting in L1d cache), deterministic training subsampling at `100k`/`1M`, and top-$R$ reranking option.
4. **Empirical Evaluation & Gate G5 Decision Framework**:
   - Measure recall cost, memory per vector, isolated Peak RSS, build time, p50/p99 latency, and QPS at `100k` and `1M` `SIFT1M` against the verified Phase 4 `f32` baselines, leaving Dev with clear, data-backed decisions at **Gate G5**.

### 1.3 Non-Goals & Invariants
- **No `unsafe`, No New Library Crates**: `#![forbid(unsafe_code)]` (`src/lib.rs`) and `std`-only default dependencies (`Cargo.toml`) remain invariant. No explicit CPU ISA intrinsics (AVX2/AVX-512/NEON); kernels rely strictly on safe Rust slice patterns (`as_chunks`) for compiler auto-vectorization.
- **No Unmeasured Claims**: All memory reductions (beyond exact Phase 4 JSON numbers), recall deltas, and throughput numbers in this design spec are **estimates/hypotheses** to be measured in Phase 6 runs.
- **Out of Scope for M4**:
  - Filtered quantized search (combining Phase 5 bitsets with Phase 6 quantized indexes can be evaluated after M4 if desired, though `CompactHnswGraph` preserves the `u32` neighbor arena shape needed to support it later).
  - Disk-backed / `mmap` raw vector storage for reranking (all Mode 2 rerank benchmarks keep `RowMatrix<f32>` in memory and honestly report its RAM cost).
  - `OPQ` (Optimized Product Quantization / orthogonal rotation matrix) or `4-bit` / `SQ4` SIMD-packed LUTs (PQ16/PQ32 and SQ8 establish the foundational quantization trade-off curve first).
  - `InnerProduct` recall benchmarking (consistent with `continuation.md`: HNSW diversity heuristic assumes non-negative distance-like values; primary SIFT1M evaluation uses `Metric::L2Squared`, while kernels support `L2Squared`, `Euclidean`, and `InnerProduct` mathematically for parity tests).

### 1.4 Recommended Defaults Pending Gate G5 Sign-Off

| # | Gate G5 Design Question | Recommended Default | Alternative(s) for Dev |
| :---: | :--- | :--- | :--- |
| **1** | **Scope: SQ8 only vs SQ8 + PQ** | **SQ8 + PQ** (SQ8 first in `T4.1–T4.3`, then PQ in `T4.4–T4.5` on the same benchmark charts) | SQ8-only (cuts scope/time by ~40% if Dev wants faster entry to Phase 7 FAISS comparison) |
| **2** | **SQ8 Storage Mode** | Implement & benchmark **both** `Sq8Only` (Memory-First) and `Sq8WithRerank` (Accuracy-First), with `Sq8Only` as the memory-footprint headline | `Sq8Only` only, or `Sq8WithRerank` only |
| **3** | **Graph Build Precision** | Primary: **`Float32GraphThenQuantize`** (exact Phase 4 graph quality, isolates quantization error) + **Ablation: `QuantizedDirect`** at `100k` (and `1M` `Sq8Only` streaming if 100k delta is acceptable) | `Float32GraphThenQuantize` only, or `QuantizedDirect` only |
| **4** | **Rerank Depth Sweep ($R$)** | **Fixed ladder `R in {None, 20, 30, 50, 100}`** for SQ8 (plus **`200`** for PQ), with `effective_ef = ef_search.max(R).max(k)`; plus **Full-Beam (`R = ef_search`)** to measure pure rescoring without collapsing low `efSearch` | `R = None` and `R = 100` only |
| **5** | **HNSW Graph Parameters** | Mandatory reference: **`M=16, efConstruction=100`** (`100k` and `1M`); **Optional Pareto config**: **`M=32, efConstruction=200`** at `100k` (and `1M` only after pilot budget review) | Add **`M=8, efC=100`** for minimum-RAM `Sq8Only`/`Pq16Only` envelope |
| **6** | **PQ Parameters (if PQ enabled)** | **`M_pq in {16, 32}`**, `K=256` (`16 B/vec` and `32 B/vec`), `N_train = min(N, 65_536)`, `max_iters = 25`, `seed = 42` | `M_pq = 16` only (`AGENT_PLAN.md` starting point) |

---

## 2. Strict Isolation from `f32` `HnswIndex`

### 2.1 `PRESERVING` Invariant for Phases 1–5
`AGENT_PLAN.md` §5 M4 requires that float32 paths remain completely intact and tested, because the `PRESERVING` guarantees from M1 (`graph_fingerprint()` equality and bit-identical fixed-query results) do not apply to quantized variants.

To make regressions **structurally impossible**:
1. **`src/hnsw.rs` (`HnswIndex<'a>`) is NOT modified** (zero changes to `HnswIndex` fields, `insert`, `search_layer`, `select_neighbors_heuristic`, `search_knn`, or `search_knn_filtered`).
2. **Existing `f32` golden proofs remain continuously verified by `cargo test --release`**:
   - Phase 3.5 golden `10k` and `100k` fingerprints (`0x80af03bd7aebaade` at `100k M=16, efC=100`) and fixed-query outputs.
   - Phase 4 `1M` reference (`0x44044886c62bca5e` at `1M M=16, efC=100`).
3. **New Module Separation (`src/quantized/`)**:
   ```
   src/quantized/
     ├── mod.rs        # Public traits, QuantizedBuildMode, QuantizedMemoryProfile, exports
     ├── graph.rs      # CompactHnswGraph, QuantizedDistanceSource trait, QuantizedSearchScratch
     ├── sq8.rs        # Sq8Quantizer, Sq8Vectors, HnswSq8Index, streaming .fvecs builder
     └── pq.rs         # PqTrainer (Xoshiro256** k-means++ + Lloyd), PqVectors, HnswPqIndex
   ```
   *(All exported through `pub mod quantized;` in `src/lib.rs`, with `#![forbid(unsafe_code)]` enforced crate-wide.)*

### 2.2 `CompactHnswGraph` & `QuantizedDistanceSource`
Instead of duplicating or mutating `HnswIndex<'a>`, `CompactHnswGraph` reproduces the exact compact `T1.7` adjacency layout (`HnswNodeMeta` 8-byte structs + flat `u32` `layer0` arena with stride `m_max0 + 1` + sparse `upper_arena` with stride `m_max + 1`) in a **lifetime-free owned struct**:

```rust
/// Static-dispatch distance source for quantized HNSW graph construction and search.
pub trait QuantizedDistanceSource {
    /// Reusable per-query precomputed state (e.g., shifted query or PQ ADC lookup table).
    type QueryState: Default;

    /// Number of quantized base vectors indexed.
    fn len(&self) -> usize;

    /// Vector dimensionality (`D`).
    fn dim(&self) -> usize;

    /// Configured distance metric.
    fn metric(&self) -> Metric;

    /// Prepares per-query state once before graph traversal (writes into preallocated `state`).
    fn prepare_query(&self, query: &[f32], state: &mut Self::QueryState);

    /// Computes approximate distance between prepared query `state` and base node `node_id`.
    fn query_distance(&self, state: &Self::QueryState, node_id: usize) -> f32;

    /// Computes approximate node-to-node distance (used only during `QuantizedDirect` graph build).
    fn node_distance(&self, id_a: usize, id_b: usize) -> f32;

    /// Formats internal minimization sort key to public metric output (e.g., sqrt for Euclidean).
    fn format_output_distance(&self, sort_key: f32) -> f32;

    /// Total heap payload bytes of quantized codes and codebooks/calibration tables.
    fn vector_memory_bytes(&self) -> usize;
}
```

#### Two Graph Acquisition Paths on `CompactHnswGraph`
1. **`CompactHnswGraph::from_hnsw_index(index: &HnswIndex<'_>) -> Self`** (used by `QuantizedBuildMode::Float32GraphThenQuantize`):
   - Reads topology exclusively through existing public read-only methods on `HnswIndex` (`num_vectors()`, `entry_point()`, `max_layer()`, `node_layer(id)`, `neighbors(lc, id)`, `config()`, `metric()`).
   - Populates `node_meta`, `layer0`, and `upper_arena` identically, achieving the **exact same `graph_fingerprint()`** (`0x80af03bd7aebaade` at 100k `M=16`, `0x44044886c62bca5e` at 1M `M=16`) without touching `src/hnsw.rs`!
   - Once copied, the caller can immediately drop both the temporary `HnswIndex<'_>` and `RowMatrix<f32>`.
2. **`CompactHnswGraph::build_with_source<S: QuantizedDistanceSource>(source: &S, config: HnswConfig) -> Result<Self>`** (used by `QuantizedBuildMode::QuantizedDirect`):
   - Executes the exact same Malkov & Yashunin Algorithms 1, 2, and 4 as `HnswIndex::build` (`DeterministicRng::new(config.seed)` layer sampling, `VisitedTracker`, `HeuristicScratch`, `select_neighbors_heuristic`, canonical `(distance, id)` tie-breaking), but calls `source.query_distance(...)` during `search_layer` and `source.node_distance(...)` during `select_neighbors_heuristic` / reverse-edge `shrink`.

#### Zero-Allocation `QuantizedSearchScratch<Q>`
Unlike `HnswIndex::search_layer`, which allocates two `BinaryHeap`s per layer call (`T1.6` saw no gain at 100k for `f32`, though `Phase 5` preallocated them in `FilteredSearchScratch`), quantized inner-loop evaluation is faster per node so heap-allocation overhead would constitute a larger fraction of latency. Therefore `QuantizedSearchScratch<Q>` preallocates **every** per-query buffer once:

```rust
/// Reusable per-thread scratch space for zero-allocation quantized HNSW search.
#[derive(Debug, Clone)]
pub struct QuantizedSearchScratch<Q> {
    pub tracker: VisitedTracker,
    pub frontier: BinaryHeap<Reverse<Candidate>>,
    pub results: BinaryHeap<Candidate>,
    pub raw_candidates: Vec<Candidate>,
    pub rerank_candidates: Vec<(f32, usize)>,
    pub query_state: Q,
}
```
Pre-sized at `frontier: with_capacity(max_ef + m_max0 * 4)` (grows at most once during warmup, never during timed passes—or `with_capacity(num_nodes)` when strict zero-growth is asserted), `results: with_capacity(max_ef + 1)`, and `query_state` pre-sized to `D` (SQ8) or `M_pq * 256` (PQ). Timed hot paths perform **zero heap allocations**.

---

## 3. Scalar Quantization 8-Bit (SQ8) Architecture

### 3.1 Per-Dimension Min/Max Calibration & Edge Cases
Given base vectors $V \in \mathbb{R}^{N \times D}$:
1. Validate $N > 0$, $D > 0$, $N \le \text{u32::MAX}$, and all entries finite (`v.is_finite()`).
2. For each dimension $j \in \{0, \dots, D-1\}$:
   $$\text{min}_j = \min_{0 \le i < N} v_{i,j}, \qquad \text{max}_j = \max_{0 \le i < N} v_{i,j}$$
   $$\Delta_j = \text{max}_j - \text{min}_j$$
3. Precompute scale constants once at calibration time (never inside distance loops):
   $$s_j = \begin{cases} \dfrac{\Delta_j}{255.0} & \text{if } \Delta_j > 0.0 \\ 0.0 & \text{if } \Delta_j == 0.0 \end{cases}, \qquad s_j^{-1} = \begin{cases} \dfrac{255.0}{\Delta_j} & \text{if } \Delta_j > 0.0 \\ 0.0 & \text{if } \Delta_j == 0.0 \end{cases}, \qquad s_j^2 = s_j \cdot s_j$$
   *(Computed in `f64` during calibration, then stored as `f32` arrays of length $D$: `mins`, `scales`, `inv_scales`, `scales_sq` — total `4 * 128 * 4 = 2,048 B = 2.0 KiB` for $D=128$.)*
4. **Encoding ($f32 \to u8$)**:
   For each vector $i$ and dimension $j$:
   $$c_{i,j} = \operatorname{round\_clamp}_{[0, 255]}\!\left((v_{i,j} - \text{min}_j) \cdot s_j^{-1}\right) \in \{0, \dots, 255\}$$
   In Rust: `((v - min_j) * inv_scale_j).round().clamp(0.0, 255.0) as u8`.
   - **Constant-dimension safety**: when $\text{min}_j == \text{max}_j$, `inv_scale_j == 0.0`, so `c_{i,j} == 0` and dequantization $\hat{v}_{i,j} = \text{min}_j + 0 \cdot 0.0 = \text{min}_j$ is **exact** with zero NaNs or divisions by zero.
   - **Extreme range / subnormal safety**: if $\Delta_j$ overflows `f32` (`max_j - min_j == inf` when `min_j = -3e38, max_j = +3e38`), computing `(max_j as f64 - min_j as f64) / 255.0` in `f64` keeps `scales[j]` finite (`2.35e36`). Note that `scales_sq[j]` would overflow `f32` only if $s_j > 1.84 \times 10^{19}$; symmetric kernel can fall back or compute `(s_j * diff)^2` safely (`s_j * (f32::from(ca) - f32::from(cb))`), avoiding premature squaring overflow.
   - **Reconstruction error bound**: for all base vectors, $|v_{i,j} - (\text{min}_j + s_j c_{i,j})| \le \frac{\Delta_j}{510}$ (up to 1 ULP of `f32` rounding).

### 3.2 Asymmetric & Symmetric Distance Kernels in Safe Rust (Zero Inner-Loop Division)
A naive implementation of quantized $L_2^2$ distance computes $\sum_j \big(q_j - (\text{min}_j + \frac{c_{i,j}}{255}(\text{max}_j - \text{min}_j))\big)^2$, incurring a floating-point division and three arithmetic ops per coordinate. We eliminate division completely using **per-query precomputation** and evaluate two safe-Rust 8-way unrolled $L_2^2$ kernels:

#### Kernel A: Query-Shifted Direct Asymmetric $L_2^2$ (Default, Zero Cancellation Risk, Zero Extra Per-Vector RAM)
1. **Per-query precomputation** (executed once in `prepare_query`, $O(D)$):
   $$\tilde{q}_j = q_j - \text{min}_j \qquad \forall j \in \{0, \dots, D-1\}$$
   Notice `q_shifted` is $128 \times 4 = 512\text{ B}$ — pinned inside L1d cache for the entire search.
2. **Inner loop per candidate $i$**:
   $$d_{\text{asym}}(\mathbf{q}, \mathbf{c}_i) = \sum_{j=0}^{D-1} \left(\tilde{q}_j - s_j \cdot c_{i,j}\right)^2$$
   Implemented in safe `#![forbid(unsafe_code)]` Rust using `as_chunks::<8>()` and 8 independent accumulators (matching [`src/distance.rs`](../../src/distance.rs)):
   ```rust
   pub fn l2_squared_sq8_shifted(q_shifted: &[f32], scales: &[f32], code: &[u8]) -> f32 {
       assert_eq!(q_shifted.len(), scales.len());
       assert_eq!(q_shifted.len(), code.len());

       let mut s0 = 0.0_f32; let mut s1 = 0.0_f32; let mut s2 = 0.0_f32; let mut s3 = 0.0_f32;
       let mut s4 = 0.0_f32; let mut s5 = 0.0_f32; let mut s6 = 0.0_f32; let mut s7 = 0.0_f32;

       let (q_chunks, q_rem) = q_shifted.as_chunks::<8>();
       let (sc_chunks, sc_rem) = scales.as_chunks::<8>();
       let (c_chunks, c_rem) = code.as_chunks::<8>();

       for ((cq, cs), cc) in q_chunks.iter().zip(sc_chunks.iter()).zip(c_chunks.iter()) {
           let d0 = cq[0] - cs[0] * f32::from(cc[0]);
           let d1 = cq[1] - cs[1] * f32::from(cc[1]);
           let d2 = cq[2] - cs[2] * f32::from(cc[2]);
           let d3 = cq[3] - cs[3] * f32::from(cc[3]);
           let d4 = cq[4] - cs[4] * f32::from(cc[4]);
           let d5 = cq[5] - cs[5] * f32::from(cc[5]);
           let d6 = cq[6] - cs[6] * f32::from(cc[6]);
           let d7 = cq[7] - cs[7] * f32::from(cc[7]);
           s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
           s4 += d4 * d4; s5 += d5 * d5; s6 += d6 * d6; s7 += d7 * d7;
       }
       let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
       for ((&vq, &vs), &vc) in q_rem.iter().zip(sc_rem.iter()).zip(c_rem.iter()) {
           let d = vq - vs * f32::from(vc);
           sum += d * d;
       }
       if sum <= 0.0 { 0.0 } else { sum }
   }
   ```
   - **Zero bounds checks** inside the unrolled loop (`&[f32; 8]` and `&[u8; 8]` fixed-size array indexing).
   - **Zero division** (`scales` precomputed at calibration).
   - **Zero cancellation**: every summand is a non-negative square $d_j^2 \ge 0$, so relative precision holds even when $\|\mathbf{q} - \mathbf{v}_i\| \ll \|\mathbf{q}\|$.

#### Kernel B: Analytical Norm-Expanded Asymmetric $L_2^2$ (Microbenchmarked Alternative in `T4.2`)
Expanding $(\tilde{q}_j - s_j c_{i,j})^2$ analytically yields:
$$d_{\text{asym}}(\mathbf{q}, \mathbf{c}_i) = \underbrace{\sum_{j=0}^{D-1} \tilde{q}_j^2}_{C_q\text{ (once per query)}} - 2 \sum_{j=0}^{D-1} \underbrace{(\tilde{q}_j \cdot s_j)}_{\hat{q}_j\text{ (once per query)}} c_{i,j} + \underbrace{\sum_{j=0}^{D-1} (s_j \cdot c_{i,j})^2}_{B_i\text{ (stored once per base vec, }+4\text{ B/vec)}}$$
- **Pros**: Eliminates reading `scales[j]` and eliminates the subtraction and squaring inside the inner loop (reducing inner loop to one `f32::from(cc[k])` + one FMA `s_k += q_hat[k] * f32::from(cc[k])`, enabling compiler auto-vectorization over 2 streams instead of 3).
- **Cons**: Requires `+4.00 B/vector` (`3.81 MiB` at `1M`) for `base_norms_sq: Vec<f32>`, and suffers from `f32` catastrophic cancellation if $\|\mathbf{q} - \mathbf{v}_i\|^2 \ll \|\mathbf{q} - \mathbf{min}\|^2$ (e.g., if vectors sit far from `min` with tiny pairwise separations; centering around the midpoint $\mu_j = \text{min}_j + 127.5 s_j$ with `i8`/`u8 - 128` reduces norm magnitudes somewhat, yet does not eliminate cancellation on arbitrary synthetic clusters).
- **Decision rule (`T4.2`)**: Microbenchmark Kernel A vs Kernel B (both 8-way unrolled) on `100k` `SIFT1M` and adversarial cancellation inputs. Adopt Kernel A as default unless Kernel B clears the `AGENT_PLAN.md` §4 noise-floor rule AND passes a strict relative-error bound across SIFT1M; if Kernel B is offered, keep Kernel A for arbitrary-range safety.

#### Kernel C: Symmetric Code-to-Code $L_2^2$ (for `QuantizedDirect` Graph Build)
When comparing two quantized base nodes $a$ and $b$ (in `select_neighbors_heuristic` and reverse-edge `shrink`):
$$d_{\text{sym}}(\mathbf{c}_a, \mathbf{c}_b) = \sum_{j=0}^{D-1} \left( s_j \cdot \big(\text{f32::from}(c_{a,j}) - \text{f32::from}(c_{b,j})\big) \right)^2$$
Wait: notice $\text{min}_j$ cancels identically (`(min_j + s_j * c_{a,j}) - (min_j + s_j * c_{b,j}) = s_j * (c_{a,j} - c_{b,j})`), and `f32::from(u8)` is bit-exact! We can write `scales_sq[j] * (d * d)` (precomputing `scales_sq[j] = s_j * s_j` outside the loop, falling back to `(s_j * d) * (s_j * d)` only if `scales_sq` would be non-finite) with `as_chunks::<8>()` 8-way unrolling and zero divisions.

#### Kernel D: Inner Product & Euclidean Extensions
- `Metric::Euclidean`: `l2_squared_sq8_shifted(...).sqrt()`, ordering monotonic with `L2Squared`.
- `Metric::InnerProduct`:
  $$\sum_{j=0}^{D-1} q_j (\text{min}_j + s_j c_{i,j}) = \underbrace{\sum_{j=0}^{D-1} q_j \text{min}_j}_{C_{\text{ip}}\text{ (once per query)}} + \sum_{j=0}^{D-1} (q_j s_j) c_{i,j}$$
  Again zero inner-loop division!

### 3.3 Storage Modes: Memory-First vs. Accuracy-First
To cleanly decouple deployment storage from query-time reranking without accidentally retaining raw `f32` vectors in memory-first runs, `HnswSq8Index` is **lifetime-free** and stores **only** the quantized representation (`CompactHnswGraph` + `Sq8Vectors`).

```
Mode (a): Memory-First (Sq8Only)
  [Query f32] ─► prepare_query ─► [CompactHnswGraph + Sq8Vectors (u8)] ─► Top-k (SQ8 distance)
                                   (NO RowMatrix<f32> in memory!)

Mode (b): Accuracy-First (Sq8WithRerank)
  [Query f32] ─► prepare_query ─► [CompactHnswGraph + Sq8Vectors (u8)] ─► Top-R candidates (ef = max(ef, R))
                                                                               │
                                                                               ▼
                                                     [Raw RowMatrix<f32>] ─► Exact f32 L2^2 on R IDs ─► Top-k
```

#### Memory Footprint Comparison at $N = 1{,}000{,}000$, $D = 128$, $M = 16$
The `f32` Phase 4 figures below are **verified measurements** from [`results/phase4/scale_1000000_m16_efc100.json`](../phase4/scale_1000000_m16_efc100.json) (`commit cd48e7f`); all SQ8 and PQ figures are **architectural payload estimates** (subject to verification in Phase 6 runs):

| Index Variant | Raw `f32` Vectors | Quantized Codes + Tables | Graph (`M=16`) | Estimated Payload (`B/vec` / `MiB`) | Payload Delta vs `f32` (`660.91 B/vec`) | Peak RSS Note (`PeakWorkingSet64`) |
|---|---|---|---|---|---|---|
| **Phase 4 `f32` Baseline** *(measured)* | `512.00 B` (`488.28 MiB`) | `0 B` | **`148.91 B`** (`142.01 MiB`) | **`660.91 B`** (`630.30 MiB` + `4 B` tracker) | `0.0%` (**`647.26 MiB` measured Peak RSS**) | Single-process build+query lifetime HWM |
| **SQ8-Only (`Sq8Only`, Kernel A)** *(estimate)* | **`0 B`** (dropped / not loaded) | `128.00 B` (`122.07 MiB` + `2 KiB` cal.) | `~148.91 B` (`142.01 MiB`) | **`~276.91 B`** (`~264.08 MiB`) | **`~ -58.1%`** (~`384 B/vec` saved) | See **§3.3.1** on streaming vs post-drop RSS |
| **SQ8 + Top-R Rerank (`Sq8WithRerank`)** *(estimate)* | `512.00 B` (`488.28 MiB`) | `128.00 B` (`122.07 MiB`) | `~148.91 B` (`142.01 MiB`) | **`~788.91 B`** (`~752.37 MiB`) | **`~ +19.4%`** (`+128 B/vec`) | Tests whether `128 B` cache traversal beats `f32` at matched recall |
| **PQ16-Only (`M_pq=16, K=256`)** *(estimate)* | **`0 B`** | `16.13 B` (`15.38 MiB` incl. `128 KiB` codebook) | `~148.91 B` (`142.01 MiB`) | **`~165.04 B`** (`~157.40 MiB`) | **`~ -75.0%`** (~`496 B/vec` saved) | Graph (`148.91 B`) is `90.2%` of total index |
| **PQ32-Only (`M_pq=32, K=256`)** *(estimate)* | **`0 B`** | `32.13 B` (`30.64 MiB` incl. `128 KiB` codebook) | `~148.91 B` (`142.01 MiB`) | **`~181.04 B`** (`~172.66 MiB`) | **`~ -72.6%`** | Graph (`148.91 B`) is `82.3%` of total index |
| **PQ16 + Top-R Rerank** *(estimate)* | `512.00 B` (`488.28 MiB`) | `16.13 B` (`15.38 MiB`) | `~148.91 B` (`142.01 MiB`) | **`~677.04 B`** (`~645.68 MiB`) | **`~ +2.4%`** | Near `f32` RAM; tests PQ traversal + exact rerank |

*(Note: Adding `M=8, efC=100` where Phase 4 measured `85.44 B/vec` graph-only and `583.42 MiB` Peak RSS yields estimated payloads of `~213.44 B/vec` (`~203.55 MiB`) for `SQ8-Only` and `~101.57 B/vec` (`~96.87 MiB`) for `PQ16-Only`.)*

#### 3.3.1 Critical Engineering Note on `PeakWorkingSet64` (Build-Time vs. Query-Time RSS)
In [`src/phase4.rs`](../../src/phase4.rs) and [`src/hardware.rs`](../../src/hardware.rs), `peak_rss_mb` records Windows `PeakWorkingSet64` (`peak_rss_scope: "process_lifetime_high_water_mark"`). Because a process lifetime high-water mark **never decreases** when `drop(base_f32)` runs:
- If a single process loads `RowMatrix<f32>` (`488.28 MiB`), builds an `f32` HNSW graph (`142.01 MiB`), and encodes SQ8 (`122.07 MiB`), `PeakWorkingSet64` will report **`~752+ MiB`** even after `RowMatrix<f32>` is dropped!
- To measure the **true ~60% RAM reduction** honestly via OS `PeakWorkingSet64` (without guessing or filtering numbers):
  1. **Two-Pass Streaming `.fvecs` Calibration + Quantized Build (`build_sq8_streaming_from_fvecs`)**:
     - Pass 1 streams the `.fvecs` file in a `4,096`-vector (`2.0 MiB`) chunk buffer to compute `mins`/`maxs`.
     - Pass 2 streams chunks to encode `u8` codes directly into `Vec<u8>` (`122.07 MiB`).
     - `RowMatrix<f32>` (`488.28 MiB`) is **never resident in RAM**, so even **build-time `PeakWorkingSet64`** is estimated near **`~270–290 MiB`**!
  2. **Deterministic Binary Snapshot + Isolated Query Process (`write_quantized_index` / `--internal-profile-query`)**:
     - For `Float32GraphThenQuantize` (where `f32` vectors are needed during construction), serialize the quantized index (`graph` + `codes` + `calibration`/`codebook`) to a temporary binary artifact and load it into a fresh child process (mirroring Phase 3's `--internal-profile-scale`) to measure true **query-serving `PeakWorkingSet64`**.
     - Every JSON report records **all three** memory metrics explicitly: `steady_state_payload_bytes_per_vector` (analytic sum of retained vectors + graph + calibration), `build_peak_rss_mb`, and `query_serving_peak_rss_mb`.

#### 3.3.2 Exact Top-$R$ Reranking Semantics & Mathematical Pitfall at $R = k$
When `rerank_top_r = Some(r)` is passed to `search_knn_rerank_with_scratch`:
1. Reject `r == 0` with `InvalidArgument` (use `None` for no rerank; avoids silent ambiguity).
2. Set `effective_r = r.max(k)` and **`effective_ef = ef_search.max(effective_r)`** (because Layer-0 `results` heap `W` holds at most `effective_ef` entries; if `ef_search < R`, `W` would not contain `R` candidates to rerank). Record both `requested_ef_search` and `effective_ef_search` in JSON (matching Phase 3's `ef < k` documentation in `continuation.md`).
3. Run quantized `CompactHnswGraph` traversal to collect `W` (`|W| <= effective_ef`).
4. Sort/select the `effective_r` lowest-quantized-distance candidates from `W`, compute **exact `Metric` distance** against `raw_base.row(id)` using [`src/distance.rs`](../../src/distance.rs), re-sort by canonical `(exact_distance, id)`, and return the top `k`.
5. **Crucial `R = k` invariant**: Because `Recall@k` measures **set overlap** ($|\mathcal{R}_{\text{approx}}^{(k)} \cap \mathcal{R}_{\text{exact}}^{(k)}| / k$), setting $R = k = 10$ only reorders the already-chosen 10 IDs and **cannot change `Recall@10`**. Improving `Recall@10` requires $R > k$ ($R \in \{20, 30, 50, 100, 200\}$) or `FullBeam` ($R = \max(k, \text{efSearch})$).

### 3.4 Graph Construction Precision: `Float32GraphThenQuantize` vs. `QuantizedDirect`

| Dimension | `Float32GraphThenQuantize` | `QuantizedDirect` |
|---|---|---|
| **Graph Topology & Fingerprint** | Bit-identical to Phase 4 `f32` `HnswIndex` (`0x44044886c62bca5e` at 1M `M=16, efC=100`). | New graph (`GRAPH-CHANGING`): `search_layer` uses asymmetric `f32-u8` (`q` is `f32` raw during `insert(q_id)` when available, or `u8` in streaming mode!), and `select_neighbors_heuristic` / `shrink` use symmetric `u8-u8` distances. |
| **Triangular Heuristic Fidelity** | `select_neighbors_heuristic` evaluates $d(e, r) \le d(q, e)$ in exact `f32`; zero pruning distortion. | Quantization noise $\pm \epsilon$ can occasionally flip $d(e, r) \le d(q, e)$ near boundary ties, altering Layer-0 edge diversity slightly. |
| **Build-Time Peak RAM (`1M`)** | High (`~647–770 MiB`): raw `f32` matrix must coexist with graph during build. | **Low (`~270–290 MiB` estimate for SQ8 streaming)**: raw `f32` vectors are never kept in RAM during graph construction. |
| **Build Speed Hypothesis** | Same as `f32` (`450.30 s` at 1M `M=16`) + `< 1 s` quantization pass. | Evaluates `128 B` `u8` codes instead of `512 B` `f32` vectors during the `3.74B` distance evaluations of a 1M build (from Phase 3.5 `baseline_1m_report.json`); **to be measured** whether cache locality speeds up build or graph drift alters eval count. |
| **Role in Phase 6** | **Primary ablation**: isolates query-time quantization accuracy loss on the identical Phase 4 graph. | **End-to-end quantized build ablation**: proves true sub-300 MiB build+serve feasibility and measures any recall/latency delta vs `Float32GraphThenQuantize`. |

*(Note for `QuantizedDirect` with `SQ8`: when constructing from an in-memory `RowMatrix<f32>`, `insert(q_id)` can query using the exact `f32` `q_vec` against existing `u8` neighbors in `search_layer` and use `u8-u8` only for `heuristic`/`shrink`, whereas streaming build uses `u8-u8` everywhere. Standardizing `QuantizedDirect` on symmetric `u8-u8` inside `CompactHnswGraph::build_with_source` guarantees that streaming `.fvecs` build and in-memory `QuantizedDirect` build produce the **exact same `graph_fingerprint()`**!)*

---

## 4. Product Quantization (PQ) Stretch Goal Design

If Dev approves **SQ8 + PQ** at Gate G5, Phase 6 implements **Product Quantization (PQ)** completely from scratch in `src/quantized/pq.rs` (`#![forbid(unsafe_code)]`, `std`-only).

### 4.1 Subspace Decomposition & Code Layout
- **Configuration (`PqConfig`)**:
  - `m_pq: usize` — number of subspaces, requiring `dim % m_pq == 0`. Evaluated values: **`M_pq in {16, 32}`**.
    - At $D = 128$: $M_{\text{pq}} = 16 \implies d_{\text{sub}} = 8$; $M_{\text{pq}} = 32 \implies d_{\text{sub}} = 4$.
  - `k_centroids: usize = 256` — fixed $K = 256$ centroids per subspace so each subspace code fits in a single `u8`.
  - `max_train_vectors: usize = 65_536` — training subsample cap (`256 * K`).
  - `max_lloyd_iters: usize = 25` — Lloyd refinement cap, with early stop if relative MSE reduction $< 10^{-5}$.
  - `seed: u64 = 42` — controls training subsampling, $k$-means++ initialization, and empty-cluster tie-breaking.
- **Memory Layout**:
  - `centroids: Vec<f32>` of length $M_{\text{pq}} \times 256 \times d_{\text{sub}} = 256 \times D$ (`32,768` floats = **`128.0 KiB`** for $D=128$, independent of $M_{\text{pq}}$!). Stored `[subspace m][centroid k][dim d]`.
  - `codes: Vec<u8>` of length $N \times M_{\text{pq}}$, row-major: vector `i` occupies `&codes[i * m_pq .. (i + 1) * m_pq]` (**`16 B/vec`** for $M_{\text{pq}}=16$, **`32 B/vec`** for $M_{\text{pq}}=32$).

### 4.2 Deterministic Training (`Xoshiro256**` Subsampling + $k$-means++ + Lloyd)
1. **Deterministic Subsampling**:
   - Let $N_{\text{train}} = \min(N, \text{max\_train\_vectors})$.
   - At $N = 100\text{k}$ and $N = 1\text{M}$, $N_{\text{train}} = 65{,}536$ ($65.5\%$ of `100k`, $6.55\%$ of `1M` — `256` vectors per centroid).
   - If $N_{\text{train}} == N$, use all rows `0..N`. Otherwise draw $N_{\text{train}}$ distinct indices without replacement via partial Fisher-Yates shuffle on `[0..N]` driven by `DeterministicRng::new(seed)` (`SplitMix64` + `Xoshiro256**` in [`src/rng.rs`](../../src/rng.rs)).
2. **Per-Subspace Independent $k$-means++ & Lloyd Refinement** (for each $m \in \{0, \dots, M_{\text{pq}}-1\}$):
   - Copy the $N_{\text{train}} \times d_{\text{sub}}$ subvectors into a contiguous scratch matrix (`2.0 MiB` for $M_{\text{pq}}=16$, `1.0 MiB` for $M_{\text{pq}}=32$ — L2/L3-cache resident!).
   - **Effective Centroid Count ($N_{\text{train}} < 256$ edge case)**: let $K_{\text{eff}} = \min(256, N_{\text{train}})$. Train $K_{\text{eff}}$ centroids cleanly; if $K_{\text{eff}} < 256$, populate remaining slots `K_eff..256` with copies of centroid `0` and restrict `encode` `argmin` to `0..K_eff` so degenerate small datasets never produce NaN/Inf or unused-slot matches.
   - **$k$-means++ Initialization** (mirroring `deterministic_kmeans_plus_plus` in [`src/dataset.rs`](../../src/dataset.rs) on the $d_{\text{sub}}$ slice):
     1. Pick first centroid uniformly (`rng.next_usize(n_train)`).
     2. Maintain `min_sq_dist[i]` to nearest chosen centroid so far.
     3. Sample remaining $K_{\text{eff}} - 1$ centroids with probability proportional to $D(x_i)^2$ using `f64` cumulative sum and `rng.next_f64_open01()`.
   - **Lloyd Iterations (`0..max_lloyd_iters`)**:
     1. **Assign step**: for each training subvector `i`, find $k^* = \arg\min_{k \in 0..K_{\text{eff}}} \|\mathbf{x}_{i,m} - \mathbf{C}_{m,k}\|_2^2$, breaking distance ties canonically by smallest $k$. Accumulate sums in **`f64`** (`[f64; d_sub]` per centroid) to eliminate float drift over `65,536` points, plus cluster counts and total MSE.
     2. **Update & Empty-Cluster Recovery**: for any centroid with `count == 0`, deterministically split the most populous (tie-broken by highest SSE, then smallest `k`) cluster by relocating its farthest member (or perturbing via `rng` if all members are identical, `sse == 0.0`).
     3. **Convergence check**: stop early if assignment changes `== 0` or $(\text{prev\_mse} - \text{mse}) / \text{prev\_mse} < 10^{-5}$.
3. **PQ Base Vector Encoding**:
   - Assign every base vector $i \in 0..N$ and subspace $m \in 0..M_{\text{pq}}$ to its nearest centroid `0..K_eff` (canonical tie-break on smallest centroid ID).
   - Note: training + encoding cost is $O(N_{\text{train}} \cdot K \cdot D \cdot \text{iters} + N \cdot K \cdot D)$. Pilot `T4.6` measures and budgets the exact wall-clock training + encoding time at `100k` and `1M` before the full matrix runs.

### 4.3 Query ADC Lookup Tables & Safe Zero-Bounds-Check Inner Loop
1. **Query Setup (`prepare_query`, once per query)**:
   - Allocate/reuse `adc_table: Vec<f32>` of length $M_{\text{pq}} \times 256$:
     - $M_{\text{pq}} = 16$: $16 \times 256 \times 4\text{ B} = \mathbf{16\text{ KiB}}$ (fits inside L1d cache).
     - $M_{\text{pq}} = 32$: $32 \times 256 \times 4\text{ B} = \mathbf{32\text{ KiB}}$ (fits inside L1d cache).
   - For each subspace $m \in \{0, \dots, M_{\text{pq}}-1\}$ and centroid $k \in \{0, \dots, 255\}$:
     $$\text{adc\_table}[m \cdot 256 + k] = \|\mathbf{q}_m - \mathbf{C}_{m,k}\|_2^2$$
   - Cost per query: $256 \times D = 32{,}768$ float diffs/squares (~64× smaller work than 2,000 raw distance evals!).
2. **Candidate Distance Evaluation (Inner Loop)**:
   $$d_{\text{ADC}}(\mathbf{q}, \mathbf{c}_i) = \sum_{m=0}^{M_{\text{pq}}-1} \text{adc\_table}[m \cdot 256 + c_{i,m}]$$
   **Safe Rust Bounds-Check-Free Pattern**:
   Because `c_{i,m}` has type `u8` (`0..=255`), slicing `adc_table` via `let (tables, _) = adc_table.as_chunks::<256>();` yields `&[[f32; 256]]` where every inner table has compile-time length `256`:
   ```rust
   pub fn pq_adc_distance(adc_table: &[f32], code: &[u8]) -> f32 {
       let (sub_tables, rem_t) = adc_table.as_chunks::<256>();
       debug_assert!(rem_t.is_empty());
       assert_eq!(sub_tables.len(), code.len());

       let (tbl_chunks, tbl_rem) = sub_tables.as_chunks::<8>();
       let (code_chunks, code_rem) = code.as_chunks::<8>();

       let mut s0 = 0.0_f32; let mut s1 = 0.0_f32; let mut s2 = 0.0_f32; let mut s3 = 0.0_f32;
       let mut s4 = 0.0_f32; let mut s5 = 0.0_f32; let mut s6 = 0.0_f32; let mut s7 = 0.0_f32;

       for (tc, cc) in tbl_chunks.iter().zip(code_chunks.iter()) {
           s0 += tc[0][usize::from(cc[0])];
           s1 += tc[1][usize::from(cc[1])];
           s2 += tc[2][usize::from(cc[2])];
           s3 += tc[3][usize::from(cc[3])];
           s4 += tc[4][usize::from(cc[4])];
           s5 += tc[5][usize::from(cc[5])];
           s6 += tc[6][usize::from(cc[6])];
           s7 += tc[7][usize::from(cc[7])];
       }
       let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
       for (tbl, &cb) in tbl_rem.iter().zip(code_rem.iter()) {
           sum += tbl[usize::from(cb)];
       }
       if sum <= 0.0 { 0.0 } else { sum }
   }
   ```
   Because `usize::from(u8) <= 255 < 256`, LLVM proves every `[f32; 256]` lookup is in-bounds statically — zero bounds checks in safe `#![forbid(unsafe_code)]` Rust!
3. **Symmetric Distance Computation (SDC) for Optional `QuantizedDirect` PQ Build**:
   - If building a graph directly on PQ codes (`QuantizedDirect`), precompute `sdc_table` of inter-centroid squared distances `[M_pq][256][256]` (`4.0 MiB` for $M_{\text{pq}}=16$, `8.0 MiB` for $M_{\text{pq}}=32$, temporary during build only and dropped post-build!).
   - Hypothesis (to measure in ablation): because `16 B/vec` PQ has much larger distance error than `128 B/vec` SQ8, building the HNSW graph on `f32` (`Float32GraphThenQuantize`) will show substantially better recall than `QuantizedDirect` for PQ, whereas `SQ8` may show minimal graph-build degradation.
4. **Top-$R$ Rerank for PQ (`PqWithRerank`)**:
   - Uses the exact same reranking mechanism as SQ8 (§3.3.2), extending the candidate sweep up to `R = 200` (`R in {None, 30, 50, 100, 200}`) to measure how much recall is recovered from `16 B`/`32 B` traversal when exact `f32` vectors are available for final scoring.

---

## 5. Evaluation Plan & Comparison Against Phase 4 Baselines

### 5.1 Verified Phase 4 `f32` Reference Baselines
All comparisons are anchored strictly to verified Phase 4 JSON outputs (`SIFT1M`, $D=128$, $k=10$, $Q=10{,}000$ queries, single-threaded, P-core affinity mask `0x0004`, `git_dirty: false`):

| Scale & Config | Source Report (`commit`) | Build Time (s) | Graph-Only (`B/vec`) | Peak RSS (`MiB`) | `ef=10` Recall / QPS / p99 | `ef=50` Recall / QPS / p99 | `ef=100` Recall / QPS / p99 | `ef=200` Recall / QPS / p99 |
|---|---|---:|---:|---:|---|---|---|---|
| **100k** `M=16, efC=100` *(sweep)* | `results/phase4/scale_100000_m16_efc100.json` (`b267dd8`, 3 runs) | `35.805` | `145.57` | `73.80` | `0.82103` / `16,144.6` / `0.1352 ms` | `0.98019` / `4,950.9` / `0.4981 ms` | `0.99367` / `2,299.8` / `2.8496 ms` | `0.99645` / `1,318.1` / `3.6195 ms` |
| **100k** `M=16, efC=100` *(pilot)* | `results/phase4/pilot/scale_100000_m16_efc100.json` (`26a9807`, 3 runs) | `26.066` | `145.57` | `77.61` | `0.82103` / `21,237.1` / `0.0777 ms` | `0.98019` / `6,643.8` / `0.2119 ms` | `0.99367` / `3,728.9` / `0.3768 ms` | `0.99645` / `2,136.9` / `0.6650 ms` |
| **100k** `M=32, efC=200` | `results/phase4/scale_100000_m32_efc200.json` (`b267dd8`, 3 runs) | `71.941` | `273.41` | `92.59` | `0.90974` / `13,462.0` / `0.1212 ms` | `0.99548` / `4,279.1` / `0.3412 ms` | `0.99939` / `2,541.9` / `0.5803 ms` | `0.99991` / `1,512.6` / `0.9935 ms` |
| **1M** `M=8, efC=100` | `results/phase4/scale_1000000_m8_efc100.json` (`cd48e7f`, 5 runs) | `266.541` | `85.44` | `583.42` | `0.57222` / `20,150.7` / `0.0830 ms` | `0.86072` / `6,743.3` / `0.2109 ms` | `0.93304` / `3,861.6` / `0.3390 ms` | `0.97330` / `2,071.0` / `0.6434 ms` |
| **1M** `M=16, efC=100` *(ref)* | `results/phase4/scale_1000000_m16_efc100.json` (`cd48e7f`, 5 runs) | `450.301` | `148.91` | `647.26` | `0.73315` / `12,711.7` / `0.1284 ms` | `0.94703` / `4,209.0` / `0.3230 ms` | `0.98061` / `2,383.9` / `0.5515 ms` | `0.99376` / `1,296.0` / `1.0673 ms` |
| **1M** `M=32, efC=200` *(Pareto)* | `results/phase4/scale_1000000_m32_efc200.json` (`cd48e7f`, 5 runs) | `1,288.897` | `272.33` | `767.40` | `0.84614` / `7,953.7` / `0.2041 ms` | `0.98278` / `2,579.7` / `0.5304 ms` | `0.99572` / `1,476.8` / `0.9404 ms` | `0.99881` / `842.2` / `1.6854 ms` |

*(Note: in Phase 6, we also run a fresh `f32` `M=16, efC=100` baseline on the same clean commit/environment when comparing latencies directly, so thermal/OS conditions match `AGENT_PLAN.md` §4!)*

### 5.2 Benchmark Protocol (`AGENT_PLAN.md` §4 Compliant)
- **Hardware & Environment Controls**: Plugged in, best-performance power mode, OneDrive paused / non-synced working copy, single-threaded query loop pinned to a single P-core (`0x0004`), release profile (`lto = "thin"`, `codegen-units = 1`), clean tree (`git_dirty: false`).
- **Scales & Ground Truth**:
  - `10k` (`siftsmall`): CI / parity / smoke test only (not reported as a benchmark result).
  - `100k` (`SIFT1M` prefix): full ablation scale (`3` timed repetitions after `1` untimed warmup; cached exact ground truth `data/gt_100000_k100.ivecs`).
  - `1M` (`SIFT1M` full): final evaluation scale (`5` timed repetitions after `1` untimed warmup; official `data/sift1m/sift_groundtruth.ivecs`).
- **`efSearch` Ladder**: `{10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600}`, $k = 10$, $10{,}000$ queries.
- **Pilot & Budget Gate (`T4.6`)**: Before running `1M` builds/sweeps, run a `100k` pilot of `SQ8` + `PQ16/PQ32`, compute exact training/encoding/build/sweep time estimates in `results/phase6/PROPOSAL.md`, and obtain Dev sign-off if any `1M` variant is pruned or expanded.

### 5.3 Evaluation Matrix (`100k` & `1M`)

1. **Arm Group A — SQ8 (`100k` & `1M` at `M=16, efC=100`)**:
   - `sq8_only_f32graph`: `Float32GraphThenQuantize`, `rerank = None` (also profiles isolated query RSS via snapshot reload).
   - `sq8_only_qbuild`: `QuantizedDirect`, `rerank = None` (at `100k`, plus `1M` streaming build from `.fvecs` to verify build-time Peak RSS reduction).
   - `sq8_rerank_f32graph`: `Float32GraphThenQuantize`, swept at `rerank_top_r in {20, 30, 50, 100}` plus `FullBeam` (`R = ef_search`).
   - *(Optional if Gate G5 approves `M=32, efC=200`)*: `sq8_only_m32` & `sq8_rerank_m32` at `100k` (and `1M` if budgeted in `T4.6`) to test whether denser graph edges compensate for quantization error without reranking!
2. **Arm Group B — PQ Stretch Goal (`100k` & `1M` at `M=16, efC=100`, if Gate G5 approves SQ8 + PQ)**:
   - `pq16_only_f32graph`: $M_{\text{pq}} = 16, K = 256$ (`16 B/vec`), `Float32GraphThenQuantize`, `rerank = None`.
   - `pq32_only_f32graph`: $M_{\text{pq}} = 32, K = 256$ (`32 B/vec`), `Float32GraphThenQuantize`, `rerank = None`.
   - `pq16_rerank_f32graph` & `pq32_rerank_f32graph`: `Float32GraphThenQuantize`, swept at `rerank_top_r in {30, 50, 100, 200}`.
   - `pq16_only_qbuild` (`100k` ablation only): measures how much PQ quantization error distorts HNSW construction when built directly on PQ SDC/ADC distances.

### 5.4 Metrics, Matched-Recall Comparison & Charts (`scripts/plot_phase6.py`)
Every per-config JSON records: `training_or_calibration_time_s`, `encode_time_s`, `graph_build_time_s`, `total_build_time_s`, `graph_fingerprint_hex`, `steady_state_bytes_per_vector` (broken down into `codes_bytes`, `calibration_or_codebook_bytes`, `graph_only_bytes`, `raw_vectors_retained_bytes`), `build_peak_rss_mb`, `query_serving_peak_rss_mb`, `recall_at_k`, `qps`, `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, and `spread`.

`scripts/plot_phase6.py` (matplotlib only, deterministic PNG + SVG into `results/phase6/`) generates four primary charts overlaying `f32`, `SQ8`, and `PQ` directly:
1. **`chart1_recall_vs_p99.png/svg`**: Recall@10 (x) vs p99 latency (ms, log y-axis) at `100k` and `1M` (`f32` `M=16`, `SQ8Only`, `SQ8+Rerank`, `PQ16Only`, `PQ32Only`, `PQ+Rerank`).
2. **`chart2_recall_vs_qps.png/svg`**: Recall@10 (x) vs Single-Thread QPS (y).
3. **`chart3_memory_vs_recall.png/svg`**: Total Index Payload (`B/vec`) and Query-Serving Peak RSS (`MiB`) vs Recall@10 at matched latency budget (e.g., `p99 <= 0.5 ms` and `<= 1.0 ms`) and matched `efSearch = 100`, highlighting the memory-accuracy Pareto frontier.
4. **`chart4_rerank_ablation.png/svg`**: Recall@10 recovery and latency overhead vs rerank depth $R$ across `SQ8` and `PQ16/PQ32`.

---

## 6. Unit, Adversarial & Cross-Language Parity Test Plan

All tests enforce `#![forbid(unsafe_code)]`, run under `cargo test --release` and `python -m pytest tests -q -rs` with **zero skips**, and require zero downloads (using small synthetic matrices and `siftsmall` helpers where optional):

1. **`f32` Preservation Guard (`tests/rust_phase6_tests.rs`)**:
   - Re-verify `HnswIndex` graph fingerprint and fixed-query outputs match pre-Phase-6 golden values bit-for-bit.
   - Verify `CompactHnswGraph::from_hnsw_index(&idx).graph_fingerprint() == idx.graph_fingerprint()`.
2. **SQ8 Mathematical & Kernel Tests**:
   - **Reconstruction error bound**: for synthetic vectors across varied distributions, assert $|v_{i,j} - (\text{min}_j + s_j c_{i,j})| \le \frac{\text{max}_j - \text{min}_j}{510} + 10^{-5}$.
   - **Kernel parity (`f64` reference)**: verify `l2_squared_sq8_shifted` matches `f64` dequantized $L_2^2$ across dimensions `dim in {1, 3, 7, 8, 9, 16, 128, 131}` (covering `as_chunks::<8>()` remainder tail paths!).
   - **Kernel B cancellation guard**: if `Way B` (norm-expanded) is tested, run adversarial test with large-offset vectors (`[10000.0 + eps]` where `min = 0.0`) to quantify `f32` drift vs Kernel A.
   - **Constant & extreme dimensions**: `min_j == max_j` (zero span), heterogeneous ranges (`[0, 1e-5]` vs `[-1e4, 1e4]`), zero vectors, single-vector datasets (`N=1`).
   - **Symmetric distance axioms**: `d_sym(a, a) == 0.0`, `d_sym(a, b) == d_sym(b, a) >= 0.0`.
   - **Streaming parity**: two-pass streaming `.fvecs` calibration+encoding produces **bit-identical** `mins`, `scales`, and `u8` codes as in-memory `RowMatrix<f32>` calibration+encoding.
   - **Non-finite & mismatch guards**: NaN/Inf in base vectors, NaN/Inf in query (including on empty index), wrong query dimension, `k == 0`, `rerank_top_r == Some(0)` all return `Err`.
3. **SQ8 Index & Reranking Tests**:
   - **Memory-First ownership check**: build `HnswSq8Index`, `drop(base_f32)`, run searches, serialize and reload snapshot, verify identical results.
   - **Rerank exactness**: for `R = N` (rerank all nodes), `HnswSq8Index::search_knn_rerank_with_scratch` returns the exact `ExactKnn` results; for `R = k`, Recall@k equals `rerank = None` (order-only improvement).
   - **Zero-allocation query hot path**: assert scratch buffer capacities (`frontier`, `results`, `q_shifted`, `adc_table`) remain strictly unchanged after warmup query.
   - **Deterministic `QuantizedDirect` build**: same seed $\implies$ equal `graph_fingerprint()` and bit-identical search results; different seed $\implies$ different fingerprint.
4. **PQ Trainer & Index Tests**:
   - **Determinism**: identical seed produces bit-identical centroids and codes; different seeds differ.
   - **Degenerate inputs**: $N < 256$ (e.g., $N = 10$), duplicate vectors (all identical rows), zero vectors — empty-cluster splitting never panics, returns finite distances.
   - **Division guard**: `dim % m_pq != 0` returns `Err(InvalidArgument)`.
   - **ADC parity**: `pq_adc_distance` matches naive scalar centroid sum across `m_pq in {16, 32}` and non-8-multiple `m_pq` (e.g. `m_pq = 4`).
   - **Subspace exactness**: clustered dataset with $\le 256$ tight modes recovers near-100% recall at `efSearch = 50`.
5. **Cross-Language Parity (`tests/test_phase6_parity.py`)**:
   - Calls compiled `phase6` binary (fail-if-skipped parity gate!) on synthetic `.fvecs` and checks SQ8 min/max codes, SQ8 asymmetric $L_2^2$ scores, and PQ ADC top-$k$ IDs against an independent NumPy reference.

---

## 7. Implementation Roadmap (Task Breakdown `T4.1`–`T4.9`) & Gate G5 Checklist

### 7.1 Detailed Task Breakdown (Milestone M4)

| Task ID | Deliverable | Class | Files Allowed to Change | Acceptance Criteria |
| :---: | :--- | :---: | :--- | :--- |
| **`T4.1`** | `CompactHnswGraph`, `QuantizedDistanceSource`, `QuantizedSearchScratch`, `f32` non-regression tests | `PRESERVING` | `src/quantized/mod.rs`, `src/quantized/graph.rs`, `src/lib.rs`, `tests/rust_phase6_tests.rs` | `src/hnsw.rs` untouched; `from_hnsw_index(&idx).graph_fingerprint() == idx.graph_fingerprint()`; Phase 1–5 `f32` golden fingerprints and fixed-query outputs bit-identical; full gate passes. |
| **`T4.2`** | `Sq8Quantizer`, `Sq8Vectors`, safe 8-way unrolled kernels (`Way A`, `Way B` microbench, symmetric `u8-u8`) | `PRESERVING` | `src/quantized/sq8.rs`, `tests/rust_phase6_tests.rs` | Unit tests pass for error bounds, constant/extreme dims, odd dims, non-finite guards, and `f64` parity; `Way A` vs `Way B` microbenchmark JSON determines default under noise-floor rule. |
| **`T4.3`** | `HnswSq8Index` (`Sq8Only`, `Sq8WithRerank`, `Float32GraphThenQuantize` & `QuantizedDirect`, streaming `.fvecs` build, binary snapshot) | `GRAPH-CHANGING` | `src/quantized/sq8.rs`, `src/quantized/mod.rs`, `tests/rust_phase6_tests.rs` | Memory-first `Sq8Only` drops raw `f32`; streaming `.fvecs` codes match in-memory bit-for-bit; `R = N` rerank matches `ExactKnn`; zero allocations in timed query loop; `siftsmall` smoke evaluation passes. |
| **`T4.4`** | Deterministic PQ trainer (`Xoshiro256**` $k$-means++ + Lloyd) & `PqVectors` ADC/SDC kernels *(if PQ approved)* | `PRESERVING` | `src/quantized/pq.rs`, `tests/rust_phase6_tests.rs` | Deterministic across runs; handles `N < 256` and duplicate-point empty clusters; `dim % m_pq != 0` rejected; `as_chunks::<256>()` ADC matches naive reference; zero effect on `HnswIndex`/`HnswSq8Index`. |
| **`T4.5`** | `HnswPqIndex` (`M_pq in {16, 32}`, `PqOnly` + `PqWithRerank`, `Float32GraphThenQuantize` + `QuantizedDirect`) *(if PQ approved)* | `GRAPH-CHANGING` | `src/quantized/pq.rs`, `src/quantized/mod.rs`, `tests/rust_phase6_tests.rs` | Supports `16 B/vec` and `32 B/vec`; `R = N` rerank matches `ExactKnn`; zero allocations in timed query loop; deterministic graph & query outputs for fixed seed. |
| **`T4.6`** | `100k` Pilot run & Phase 6 Time-Budget Proposal (`results/phase6/PROPOSAL.md`) | `MEASUREMENT-ONLY` | `results/phase6/PROPOSAL.md`, `results/phase6/pilot/*.json` | Pilot measures SQ8 + PQ train/encode/build/sweep times at `100k` (`M=16, efC=100`); extrapolates `1M` wall-clock and disk budget for Dev sign-off before long `1M` runs. |
| **`T4.7`** | `phase6` benchmark runner CLI, isolated query-serving RSS profiler, & NumPy parity suite | `MEASUREMENT-ONLY` | `src/phase6.rs`, `src/bin/phase6.rs`, `src/lib.rs`, `Cargo.toml`, `tests/test_phase6_parity.py` | Resumable per-config JSONs with full provenance (`git_commit`, `git_dirty`, `environment`, steady-state & peak RSS); `pytest -rs` runs `test_phase6_parity.py` with `0` skipped. |
| **`T4.8`** | Execute approved `100k` & `1M` evaluation matrix (Benchmarker, clean commit, P-core pinned) | `MEASUREMENT-ONLY` | `results/phase6/*.json` | Clean tree (`git_dirty: false`); `3` reps (`100k`) / `5` reps (`1M`) after warmup; records matched-recall latency/QPS & real Peak RSS (`Sq8Only` streaming + isolated query RSS). |
| **`T4.9`** | Plots (`scripts/plot_phase6.py`), README write-up, & `results/CLAIMS.md` audit | `MEASUREMENT-ONLY` | `scripts/plot_phase6.py`, `results/phase6/*.png`, `results/phase6/*.svg`, `README.md`, `results/CLAIMS.md` | 4 publication charts (`Recall vs p99`, `Recall vs QPS`, `RAM vs Recall`, `Rerank Ablation`); every README number mapped 1:1 to source JSON in `results/CLAIMS.md`. |

*(Note: Every task executes in separate Implementer / Reviewer / Benchmarker passes per `AGENT_PLAN.md` §2, running the full verification gate before each commit.)*

---

### 7.2 Gate G5 Decision Checklist (Dev Sign-Off Required Before Coding)

Implementers **must not** check these boxes or begin `T4.1` until Dev reviews and records Gate G5 decisions:

1. **Scope Choice (Primary Gate G5 Decision)**:
   - [ ] **Option A (`SQ8 only`)**: Implement `T4.1–T4.3` + `T4.6–T4.9` (per-dim SQ8, Memory-First + Rerank, streaming RSS proof); skip PQ (`T4.4–T4.5`) to move directly to Phase 7 (FAISS comparison).
   - [ ] **Option B (`SQ8 + PQ`)** *(Recommended)*: Implement full SQ8 (`T4.1–T4.3`) followed by PQ Stretch Goal (`T4.4–T4.5`: `M_pq in {16, 32}`, `K=256`, `N_train = min(N, 65_536)`) so both appear on the same recall-vs-latency and recall-vs-RAM Pareto charts.
2. **SQ8 Storage Mode Scope**:
   - [ ] Confirm benchmarking **both** **(a) Memory-First (`Sq8Only`)** (including two-pass streaming `.fvecs` build / isolated query RSS) and **(b) Accuracy-First (`Sq8WithRerank`)**.
3. **Top-$R$ Rerank Sweep Parameters**:
   - [ ] Confirm rerank sweep ladder: `R in {None, 20, 30, 50, 100}` for SQ8 (plus `200` for PQ if Option B chosen) + `FullBeam` (`R = ef_search`), with `effective_ef = ef_search.max(R).max(k)`.
4. **Graph Build Precision Policy**:
   - [ ] Confirm **`Float32GraphThenQuantize`** as the primary baseline (isolating quantization loss on exact Phase 4 graphs) alongside **`QuantizedDirect`** ablation at `100k` (and `1M` SQ8 streaming if approved after `T4.6` budget).
5. **HNSW Graph Parameters & Scales**:
   - [ ] Confirm `M=16, efConstruction=100` (`100k` and `1M`) as mandatory reference.
   - [ ] Decide whether **`M=32, efConstruction=200`** (Pareto high-recall config) and/or **`M=8, efConstruction=100`** (ultra-low-memory config) are included in `100k` and/or `1M` sweeps (or deferred to `T4.6` budget sign-off).
6. **Technical Sign-Off Checks**:
   - [ ] Confirm `#![forbid(unsafe_code)]`, `std`-only library constraint, and `src/hnsw.rs` untouched (`PRESERVING` non-regression on `f32` paths).
   - [ ] Confirm `T4.6` pilot budget gate before launching multi-hour `1M` builds/sweeps (same model as Phase 4 Gate G2 and Phase 5 Gate G4.5).
