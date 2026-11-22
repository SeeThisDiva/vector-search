# Milestone M3 / Phase 5: Filtered Approximate Nearest Neighbor Search Design Specification

> **Document Status**: Design specification for **Gate G4** review. Numbers in this document are hypotheses or placeholders until they are traced to a JSON report from a real run.  
> **Target Release**: `vector-search` v0.1.0 (Milestone M3)  
> **Author**: `teamwork_worker` (Specialist / Implementer Role)  
> **Engineering Invariants**: 100% Safe Rust (`#![forbid(unsafe_code)]`), standard library only (`std`), zero dynamic allocations in query hot paths, deterministic tie-breaking on `(distance, id)`, and clean-commit JSON provenance.

---

## 1. Executive Summary & Scope

### 1.1 Motivation & The Filtered Search Dilemma
Modern vector databases (e.g., Milvus, Pinecone, Qdrant, Vespa) rarely operate on pure unstructured embeddings in isolation. Production search queries almost universally specify metadata constraints alongside the target vector:
- **E-commerce**: Find products visually similar to an image *WHERE* `category == "footwear"` *AND* `price <= 120.0` *AND* `in_stock == true`.
- **Multi-Tenant SaaS**: Retrieve semantic document chunks *WHERE* `tenant_id == 0x4F2A` *AND* `acl_group IN [12, 18, 99]`.
- **Temporal Retrieval**: Search conversational context *WHERE* `timestamp >= 1704067200` *AND* `author == "user"`.

In an unfiltered Hierarchical Navigable Small World (HNSW) proximity graph, search navigates along a Delaunay-like small-world network, greedily hopping between geometric neighbors. However, when an arbitrary boolean predicate $\mathcal{P}: \mathcal{V} \to \{0, 1\}$ is applied, standard graph search faces a severe structural dilemma governed by predicate selectivity $\sigma \in (0, 1]$ (the fraction of data points satisfying $\mathcal{P}$):

1. **The Post-Filtering Collapse ($\sigma \ll 1$)**: When an index performs standard nearest neighbor search with an inflated over-fetch candidate budget $k' = \lceil k / \sigma \rceil$, low selectivities (e.g., $\sigma = 0.1\%$ or $1$ in $1,000$ points) force $k'$ into tens of thousands. The graph beam width $efSearch$ must explode to prevent candidate starvation, degrading query throughput (QPS) by orders of magnitude and frequently returning fewer than $k$ matching elements (candidate truncation).
2. **The Topological Disconnection Problem (In-Traversal Filtering)**: If graph traversal is restricted *strictly* to nodes satisfying $\mathcal{P}$, the induced subgraph $G[\mathcal{P}]$ often fragments into disjoint connected components or isolated vertices. Express entry paths in upper layers disappear, causing greedy routing to trap in catastrophic local minima and destroying recall.
3. **The Brute-Force Reversal**: As selectivity $\sigma$ drops toward zero, the count of valid candidates $N_\sigma = \sigma \cdot N$ shrinks. A scan over contiguous vectors has a sequential memory access pattern (compiler auto-vectorization only, no explicit intrinsics, because the library is safe `std` Rust), so the hypothesis is that a pre-filtered brute-force scan over surviving IDs eventually beats navigating a pointer-chasing graph. Where that happens is measured in Section 4, not assumed.

### 1.2 Mission Objectives
Phase 5 establishes a mathematically principled, empirically validated filtered vector search engine within the existing `vector-search` Rust architecture. The core objectives are:
1. **Deterministic Metadata Generation**: Formally specify and deterministically generate synthetic attribute distributions across four standardized selectivity tiers ($\sigma \in \{50\%, 10\%, 1\%, 0.1\%\}$) under both **Independent (Uniform)** and **Cluster-Correlated** spatial modes.
2. **Three Competing Search Paradigms**: Architect and implement:
   - *Strategy 1*: Post-filtering with iterative geometric over-fetch.
   - *Strategy 2*: In-traversal predicate filtering (filtered beam search on the unmodified graph: non-matching nodes are traversed for connectivity and only matching nodes enter the result set; this is not ACORN, which changes graph construction).
   - *Strategy 3*: Pre-filter brute-force fallback over compressed bitsets.
3. **Crossover Selectivity Modeling**: Formulate cost models with measured parameters and an empirical protocol to locate the crossover selectivity $\sigma^*$ where brute-force scan overtakes graph traversal at matched recall, at dataset scales $N \in \{100\text{k}, 1\text{M}\}$ (10k is a smoke test only).
4. **Authoritative Ground Truth & Evaluation Harness**: Implement exact filtered $k$-NN with deterministic `(distance, id)` canonical tie-breaking, evaluating Recall@10, latency percentiles (p50, p99), QPS, and candidate truncation rates.
5. **Rust API**: zero-allocation `FilterPredicate` trait and cache-aligned `DenseBitset`. Python bindings are deferred (Section 6.2).

### 1.3 Non-Goals & Invariants
- **Strict Invariant**: `#![forbid(unsafe_code)]` remains strictly enforced across all new library modules. No external crates are introduced into standard runtime dependencies (`std` only).
- **Zero-Allocation Hot Path**: Query routing, predicate evaluation, and scratch buffer reuse must perform zero heap allocations per query iteration.
- **Out of Scope for Phase 5**:
  - Dynamic index mutation under filtering (rebuilding graph edges per attribute value).
  - External disk-backed relational databases (attribute storage remains fully in-memory).
  - Multi-vector composite documents.
  - PyO3 / Python bindings (optional, needs Dev approval; see Section 6.2).
  - Adaptive strategy router (optional, needs Dev approval; see Section 4.4).

### 1.4 Design Decisions for Dev (recommended defaults, pending G4 sign-off)

| # | Question | Recommended decision |
|:---:|:---|:---|
| 1 | Opaque predicate hits its evaluation limit | Return partial results with flag `EvalLimitReached`. No automatic fallback: Strategy 3 needs the matching ids, and for an opaque closure the only way to get them is N predicate calls. Fallback to Strategy 3 is allowed only for bitset-backed predicates when `allow_bruteforce_fallback` is on. Benchmark arms never fall back, so failures stay visible. |
| 2 | Python bindings | Deferred to the optional section (6.2). |
| 3 | Result scales | 100k and 1M. 10k is a smoke test only (at $\sigma = 0.1\%$ it leaves 10 matches, equal to $k$). |
| 4 | Over-fetch factor $\gamma$ | Default from the confidence formula ($\delta = 0.001$), overridable per query via `overfetch_factor`, swept in at least one config. The formula holds only for independent metadata (Mode A). |
| 5 | Bridge-hop cap $H_{\max}$ | Default $3 \cdot ef$ (reviewer's value, unmeasured). Query-time field of `FilteredSearchParams`, not `HnswConfig`. Sensitivity run at $\{1, 3, 10\} \cdot ef$ on one config. |
| 6 | Over-fetch pool cap $K'_{\max}$ | Default 4,096 (hypothesis: about 7x the largest efSearch measured in Phase 4). Strategy 1 is benchmarked in two arms, capped (4,096) and uncapped ($K'_{\max} = N$). On reaching the cap with fewer than $k$ matches it returns partial results with flag `PostFilterCapReached`. |
| 7 | Crossover definition | Compared at matched recall (Section 4.3). Proposed targets 0.95 (primary) and 0.99 (secondary). Dev decides. |

---

## 2. Metadata Architecture & Synthetic Data Generation

### 2.1 Attribute Storage Model
To achieve minimal memory overhead, zero pointer chasing, and instantaneous predicate evaluation during graph search, metadata attributes are decoupled from vector row matrices and organized into a columnar layout.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ATTRIBUTE STORE                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Column 0: Boolean Bitsets (e.g. is_active, in_stock, is_verified)                     │
│ ┌──────────────────────────┬──────────────────────────┬──────────────────────────────┐ │
│ │ Word 0 (u64: bits 0..63) │ Word 1 (u64: bits 64..127│ ... Word W-1 (bits N-64..N) │ │
│ └──────────────────────────┴──────────────────────────┴──────────────────────────────┘ │
│                                                                                        │
│ Column 1: Categorical IDs (e.g. category_id in 0..K-1)                                │
│ ┌──────────────────────┬──────────────────────┬──────────────────────┬───────────────┐ │
│ │ vec_0: u16           │ vec_1: u16           │ vec_2: u16           │ ... vec_N: u16│ │
│ └──────────────────────┴──────────────────────┴──────────────────────┴───────────────┘ │
│ Inverted Index Postings: category_id -> DenseBitset (N bits)                           │
│                                                                                        │
│ Column 2: Scalar Numeric Ranges (e.g. price: f32, timestamp: i64)                      │
│ ┌──────────────────────┬──────────────────────┬──────────────────────┬───────────────┐ │
│ │ vec_0: f32           │ vec_1: f32           │ vec_2: f32           │ ... vec_N: f32│ │
│ └──────────────────────┴──────────────────────┴──────────────────────┴───────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Dense Bitset Representation (`DenseBitset`)
For boolean attributes and inverted posting lists, a dense 64-bit word array is optimal:
- **Bit-Density**: Exactly 1 bit per vector ID. At $N = 1,000,000$, a bitset consumes exactly $\lceil 10^6 / 64 \rceil \times 8 \text{ bytes} = 125,000 \text{ bytes} \approx 122.07 \text{ KiB}$.
- **Cache Alignment**: Contiguous `Vec<u64>` ensures that a single 64-byte L1 cache line holds filter membership for 512 consecutive vector IDs.
- **Constant-Time Membership**: Node $i$ membership is one shift, one mask and one load (no cycle count is claimed):
  $$\text{matches}(i) = \left( \text{words}[i \gg 6] \ \& \ (1\text{u64} \ll (i \ \& \ 63)) \right) \neq 0$$
- **Rapid Cardinality & Iteration**: `u64::count_ones` gives candidate counts and `u64::trailing_zeros` skips runs of non-matching ids. Whether these compile to hardware instructions is up to the compiler and is not assumed.

### 2.2 Target Selectivities & Statistical Framing
Phase 5 benchmarks four standardized selectivity levels:

| Selectivity Level ($\sigma$) | Target Percentage | Expected Matches ($N=10\text{k}$, smoke test only) | Expected Matches ($N=100\text{k}$) | Expected Matches ($N=1\text{M}$) |
|:---:|:---:|:---:|:---:|:---:|
| **Tier 1** | $50.0\%$ | 5,000 | 50,000 | 500,000 |
| **Tier 2** | $10.0\%$ | 1,000 | 10,000 | 100,000 |
| **Tier 3** | $1.0\%$ | 100 | 1,000 | 10,000 |
| **Tier 4** | $0.1\%$ | 10 | 100 | 1,000 |

### 2.3 Deterministic Synthetic Generation: Two Spatial Modes
Real-world metadata attributes fall into two distinct spatial regimes relative to embedding vectors:
1. **Independent Mode (Uniform Random)**: The attribute is completely orthogonal to the vector's position in metric space (e.g., whether a document was published on an odd vs. even day, or a seller's promotional tier).
2. **Cluster-Correlated Mode (Semantic Locality)**: The attribute strongly correlates with semantic embedding clusters (e.g., language tags in multilingual text embeddings, product categories in visual catalogs, or geographic regions). This represents the most challenging operational scenario for graph traversal, as matching nodes form dense, localized islands separated by vast metric expanses of non-matching nodes.

#### Mode A: Independent (Uniform) Metadata Generation
For each vector index $i \in \{0, \dots, N-1\}$, membership in the filter candidate set is drawn from an independent Bernoulli trial:
$$X_i \sim \text{Bernoulli}(\sigma)$$

To guarantee exact reproducibility across all platforms, generation is governed by `DeterministicRng` (`src/rng.rs`) (`SplitMix64` + `Xoshiro256**`). To ensure that the generated bitset contains an exact target count $K_\sigma = \lfloor \sigma \cdot N \rfloor$ without statistical variance skewing benchmark runs:
1. Initialize a bitset with all zeros.
2. Deterministically shuffle indices $[0, N-1]$ using the Fisher-Yates algorithm seeded with a dedicated PRNG seed.
3. Set the first $K_\sigma$ bits to 1.

```
Algorithm 1: Deterministic Independent Bitset Generation
Input: Dataset size N, Target selectivity σ, Seed S_attr
Output: DenseBitset of length N with exactly floor(σ * N) active bits

1: K_sigma ← floor(σ * N)
2: bitset ← DenseBitset::zeros(N)
3: indices ← [0, 1, ..., N - 1]
4: rng ← DeterministicRng::new(S_attr)
5: for i from N - 1 down to 1 do
6:     j ← rng.next_usize(i + 1)
7:     swap(indices[i], indices[j])
8: end for
9: for i from 0 to K_sigma - 1 do
10:    bitset.set_bit(indices[i], true)
11: end for
12: return bitset
```

#### Mode B: Cluster-Correlated Metadata Generation
In cluster-correlated mode, attribute values are coupled to the underlying geometric distribution of the base vectors. We formalize this using a deterministic Gaussian Mixture Model (GMM) / Voronoi Centroid partition:

1. **Centroid Selection**: Using a fixed PRNG seed $S_{\text{clust}} = 0\text{xCBF29CE484222325}$, select $C$ cluster centroids $\{\mathbf{c}_1, \dots, \mathbf{c}_C\} \subset \mathbb{R}^d$ from the base dataset using deterministic $K$-means++ initialization, where $C = \max(64, \lfloor 2 / \sigma \rfloor)$.
2. **Cluster Assignment**: Each base vector $\mathbf{v}_i \in \mathcal{V}$ is assigned to its nearest centroid:
   $$z_i = \arg\min_{c \in \{1, \dots, C\}} \|\mathbf{v}_i - \mathbf{c}_c\|_2^2$$
3. **Cluster Activation Probability**: Each cluster $c \in \{1, \dots, C\}$ is assigned an activation weight $w_c \in [0, 1]$. To model localized correlation with tunable dispersion parameter $\kappa \in [0, 1]$:
   - When $\kappa = 1.0$ (pure correlation): activate whole clusters chosen by Algorithm 2 ($w_c = 1.0$), leaving other clusters at $w_c = 0.0$. All Phase 5 result runs use $\kappa = 1.0$; $\kappa < 1.0$ is optional.
   - When $\kappa < 1.0$ (soft correlation with noise): Assign cluster activation probability:
     $$P(X_i = 1 \mid z_i = c) = \sigma \cdot \left[ 1 + \kappa \cdot \left( \frac{C \cdot \mathbb{I}(c \in \mathcal{S}_{\text{active}})}{|\mathcal{S}_{\text{active}}|} - 1 \right) \right]$$
4. **Exact Count Calibration**: Algorithm 2 adds whole clusters (largest first, ties by cluster id) while the running total stays at or below $K_\sigma$. Every cluster left out then has more members than the remaining shortfall $s = K_\sigma - \text{total}$, because it failed the fit test when the running total was no larger, so one extra cluster is enough. Pick the non-active cluster whose centroid is nearest to any active centroid (ties by cluster id; if no cluster is active yet, use the largest cluster) and activate its $s$ members with the smallest (distance to own centroid, id). The result has exactly $K_\sigma$ bits and is fully deterministic.

```
Algorithm 2: Cluster-Correlated Bitset Generation
Input: Base vector matrix V (N x d), Target selectivity σ, Centroid count C, Dispersion κ, Seeds S_clust, S_attr
Output: DenseBitset of length N with floor(σ * N) cluster-correlated active bits

1: rng_clust ← DeterministicRng::new(S_clust)
2: centroids ← DeterministicKMeansPlusPlus(V, C, rng_clust)
3: cluster_assignments ← zeros(N)
4: for i from 0 to N - 1 do
5:     cluster_assignments[i] ← argmin_{c} ||V[i] - centroids[c]||_2^2
6: end for
7: cluster_sizes ← count_by_cluster(cluster_assignments, C)
8: // Select active clusters starting from highest density centroid
9: sorted_clusters ← argsort_descending(cluster_sizes)  // ties broken by cluster id
10: active_clusters ← EmptySet()
11: accumulated_vectors ← 0
12: target_count ← floor(σ * N)
13: for c in sorted_clusters do
14:     if accumulated_vectors + cluster_sizes[c] <= target_count then
15:         active_clusters.insert(c)
16:         accumulated_vectors ← accumulated_vectors + cluster_sizes[c]
17:     end if
18: end for
19: bitset ← DenseBitset::zeros(N)
20: for i from 0 to N - 1 do
21:     if active_clusters.contains(cluster_assignments[i]) then
22:         bitset.set_bit(i, true)
23:     end if
24: end for
25: // Exact count calibration (step 4): add the shortfall from one extra cluster
26: bitset.add_shortfall_from_nearest_cluster(target_count, V, centroids, active_clusters)
27: return bitset
```

Cost and caching: assignment is $N \cdot C \cdot d$ distance work per (scale, $\sigma$), largest at $\sigma = 0.05\%$ where $C = 4000$. Bitsets are therefore generated once, written to disk with provenance (seeds, $C$, active cluster ids, popcount) and loaded by the benchmark. Generation time is reported in the T3.7 budget proposal, not assumed here.

---

## 3. The Three Filtered Search Strategies

To establish an authoritative benchmark comparison, Phase 5 implements three fundamentally distinct filtered search strategies.

```
Query + Predicate P
         │
         ├─── Strategy 1: Post-Filtering (Over-Fetch)
         │    Standard HNSW -> Get top-k' candidates -> Filter out !P(id) -> Take top-k
         │
         ├─── Strategy 2: In-Traversal Predicate Filtering (Bridge-Node Routing)
         │    Graph Traversal -> Hop through ALL nodes -> Only push P(id)==true to Result Set W
         │
         └─── Strategy 3: Pre-Filter Brute-Force Fallback (Bitset Scan)
              Evaluate P -> Bitset surviving IDs -> Flat scan over surviving vectors
```

---

### 3.1 Strategy 1: Post-Filtering with Over-Fetch

#### Algorithmic Principle
Post-filtering decouples graph search from filtering. The HNSW graph is traversed completely ignoring predicate $\mathcal{P}$, requesting an expanded top-$k'$ candidate pool. The resulting candidate stream is then evaluated against $\mathcal{P}$, discarding non-matching vectors until $k$ valid items are accumulated.

#### Mathematical Formulation of Over-Fetch Expansion
If matching items are independently distributed with selectivity $\sigma$, the number of items inspected before collecting $k$ valid matches follows a Negative Binomial distribution:
$$X \sim \text{NegBinomial}(k, \sigma)$$
The expected number of candidates required is:
$$\mathbb{E}[k'] = \frac{k}{\sigma}$$
To ensure a high probability $1 - \delta$ (e.g., $99.9\%$ confidence) of acquiring at least $k$ items, we apply an inflation factor $\gamma > 1$:
$$k' = \min\left(K'_{\max}, \ \left\lceil \frac{k}{\sigma} \cdot \gamma \right\rceil\right)$$
where $\gamma$ is dynamically scaled based on target confidence:
$$\gamma(\sigma, \delta) \approx 1 + z_{1-\delta} \sqrt{\frac{1 - \sigma}{k}}$$
Notes on this formula:
- It is the normal approximation of the negative-binomial tail. It is valid only when matches are independent of distance rank (Mode A). In Mode B the nearest neighbours are not a random sample, so the formula gives no guarantee and the widening loop below does the real work.
- $\sigma$ is exact for bitset predicates (popcount) and the caller's estimate for opaque predicates.
- Default $\delta = 0.001$ ($z_{1-\delta} \approx 3.09$). The factor is a query-time parameter (`overfetch_factor`), an explicit value overrides the formula, and at least one index config is swept over the factor.
- Pool cap: $k' \le K'_{\max}$, default $K'_{\max} = 4096$ (a hypothesis, about 7x the largest efSearch measured in Phase 4). At $k = 10$ and $\delta = 0.001$ the cap covers $\sigma \gtrsim 0.5\%$; below that Strategy 1 gives up by construction.
- Two benchmark arms: capped ($K'_{\max} = 4096$) and uncapped ($K'_{\max} = N$), so the cap cannot decide the headline result on its own.

#### Iterative Widening Protocol
If the initial over-fetch pool yields $m < k$ matching items, the query does not immediately fail. Instead, an **Iterative Widening Loop** expands the candidate beam:
1. Initialize $k'_0 = \min(K'_{\max}, \lceil k / \sigma \rceil \cdot \gamma)$.
2. Run standard `HnswIndex::search_knn(query, k'_i, efSearch = \max(efSearch, k'_i))`.
3. Filter results: let $\mathcal{M}$ be matching elements. If $|\mathcal{M}| \ge k$, return top-$k$.
4. If $|\mathcal{M}| < k$ and $k'_i < K'_{\max}$, widen: $k'_{i+1} = \min(K'_{\max}, \ \lfloor 2.5 \cdot k'_i \rfloor)$, and repeat.
5. If $k'_i = K'_{\max}$ and there are still fewer than $k$ matches, return the matches found with flag `PostFilterCapReached` (no silent fallback; with $K'_{\max} = N$ the loop ends in an exact full result).

#### Catastrophic Failure Modes of Post-Filtering
1. **Severe Memory & Heap Pressure**: For $k=10$ and $\sigma=0.001$ ($0.1\%$), the formula above gives $k' \approx 10,000 \cdot 1.98 \approx 19,800$. Maintaining a `BinaryHeap` of that size per query consumes significant memory bandwidth and invalidates L1/L2 CPU caches.
2. **Graph Beam Inefficiency**: Standard HNSW search is calibrated for $efSearch \in [10, 600]$. Forcing $efSearch \approx 20,000$ turns the search into a sprawling crawl over Layer 0, evaluating tens of thousands of distances.
3. **Cluster-Correlated Starvation**: Under Mode B (Cluster-Correlated), if the query vector is located in a non-matching semantic region, the closest $10,000$ neighbors in metric space may contain *zero* matching items! Post-filtering completely fails, returning 0 results unless $k' \to N$.

---

### 3.2 Strategy 2: In-Traversal Predicate Filtering (Bridge-Node Routing)

#### Algorithmic Principle: Filtered Beam Search with Bridge Nodes
This strategy uses the unmodified graph. (ACORN builds a denser, predicate-agnostic graph; that is out of scope.) To prevent the graph from fragmenting into disconnected components when filtering, in-traversal filtering decouples the **graph routing set** from the **result candidate set**:
- **Bridge Nodes (Routing)**: Nodes that do *not* satisfy predicate $\mathcal{P}$ are still visited, evaluated for distance, and added to the greedy traversal frontier $C$. They act as "navigational bridges" allowing search to traverse regions of non-matching space to reach distant matching clusters.
- **Result Admittance ($W$)**: Only nodes that *satisfy* $\mathcal{P}(\text{node\_id}) == \text{true}$ are permitted to enter the result tracking heap $W$.
- **Bounding & Termination**: The termination condition is strictly bounded: traversal halts when the closest candidate in frontier $C$ is further than the furthest *matching* candidate in $W$, provided $|W| \ge efSearch$.

#### Handling Isolated Components & Greedy Trapping
When $\sigma$ is very small, a standard search can wander endlessly across bridge nodes without encountering matching vertices. To guarantee high recall and bounded query latency, our in-traversal search incorporates three mechanisms:

1. **Dual Frontier Management**: Maintain two heaps:
   - $C_{\text{all}}$: Min-heap of all visited nodes (both matching and non-matching) ordered by distance from query.
   - $W_{\text{match}}$: Max-heap of matching nodes ordered by distance, bounded by size $efSearch$.
2. **Bridge-Hop Budget ($H_{\text{max}}$)**: Count consecutive node expansions since the last matching node was admitted to $W_{\text{match}}$. If the count exceeds $H_{\text{max}}$ while $|W_{\text{match}}| < k$, the layer search stops and the query returns whatever matches it has, with flag `HopBudgetExhausted`. Default $H_{\text{max}} = 3 \cdot ef$ (the reviewer's suggestion, a hypothesis, not measured). It is a query-time field of `FilteredSearchParams`, not part of `HnswConfig` (that struct holds build settings). A sensitivity run at $H_{\text{max}} \in \{1, 3, 10\} \cdot ef$ on one config is part of the benchmark matrix. Fallback to Strategy 3 is available only for bitset-backed predicates when `allow_bruteforce_fallback` is on; benchmark arms never fall back, so failures appear in the failure-rate metric.
3. **Upper-Layer Predicate Bypassing**: In upper layers ($l \ge 1$), zoom-in routing uses standard greedy search (ignoring the filter) to reach the spatial neighborhood of the query at Layer 0. Predicate gating is applied exclusively during Layer 0 beam search.

```
Algorithm 3: In-Traversal Filtered Layer Search (Algorithm 2 Filtered Extension)
Input: Query vector q, Entry point ep, Beam width ef, Target k, Predicate P, VisitedTracker tracker, Hop budget H_max
Output: Bounded Max-Heap W_match containing up to ef matching candidates

1: tracker.prepare(num_nodes)
2: tracker.mark_visited(ep)
3: d_ep ← compute_distance(q, ep)
4: C ← MinHeap()         // Holds Reverse(Candidate) for exploration
5: W_match ← MaxHeap()   // Holds Candidate matching P;  H ← 0  (expansions since last admitted match)
6: C.push(Reverse(Candidate(d_ep, ep)))
7: if P.matches(ep) then
8:     W_match.push(Candidate(d_ep, ep))
9: end if
10: while C is not empty do
11:    Reverse(curr) ← C.pop();  H ← H + 1
12:    if H > H_max and W_match.len() < k then flag(HOP_BUDGET_EXHAUSTED); break end if
12b:   // Pruning check against worst matching candidate
13:    if W_match.len() >= ef and curr.distance > W_match.peek().distance then
14:        break
15:    end if
16:    for neighbor in graph.neighbors(0, curr.id) do
17:        if not tracker.is_visited(neighbor) then
18:            tracker.mark_visited(neighbor)
19:            d_neighbor ← compute_distance(q, neighbor)
20:            cand ← Candidate(d_neighbor, neighbor)
21:            is_match ← P.matches(neighbor)
22:            if is_match then
23:                if W_match.len() < ef or cand < W_match.peek() then
24:                    W_match.push(cand);  H ← 0
25:                    if W_match.len() > ef then
26:                        W_match.pop()
27:                    end if
28:                end if
29:            end if
30:            // Always allow promising nodes into exploration frontier C
31:            if W_match.len() < ef or cand.distance < W_match.peek().distance then
32:                C.push(Reverse(cand))
33:            end if
34:        end if
35:    end for
36: end while
37: return W_match
```

---

### 3.3 Strategy 3: Pre-Filter Brute-Force Fallback (Bitset Scan)

#### Algorithmic Principle
When selectivity $\sigma$ is low ($\sigma \le 1\%$), the total count of valid candidates $N_\sigma = \lfloor \sigma \cdot N \rfloor$ is small:
- At $N = 10,000$: $N_{0.1\%} = 10$ vectors; $N_{1\%} = 100$ vectors.
- At $N = 100,000$: $N_{0.1\%} = 100$ vectors; $N_{1\%} = 1,000$ vectors.
- At $N = 1,000,000$: $N_{0.1\%} = 1,000$ vectors; $N_{1\%} = 10,000$ vectors.

Under Strategy 3, the HNSW graph is completely bypassed. The search pipeline:
1. Queries the predicate's underlying bitset or inverted index to retrieve the exact matching ID set $\mathcal{S}_{\text{match}} = \{i \mid \mathcal{P}(i) = 1\}$.
2. Performs a linear scan strictly over $\mathcal{S}_{\text{match}}$, computing distances between $\mathbf{q}$ and base vectors $\mathbf{v}_i$ ($i \in \mathcal{S}_{\text{match}}$) using the existing 8-way unrolled distance kernel (`l2_squared`).
3. Maintains a compact Max-Heap of size $k$ with deterministic `(distance, id)` canonical tie-breaking.

Strategy 3 requires a bitset-backed predicate. For an opaque closure, enumerating the matches means calling it $N$ times, and those calls count against the evaluation limit.

#### Memory Access Pattern
Hypothesis (measured by the T3.6 microbenchmark): a scan over matching ids has a more regular memory access pattern than graph traversal, so the cost per distance evaluation should be lower. No per-evaluation or per-query timing is assumed in this document. The scan has zero approximation error by construction (Recall = 100%) and cannot return fewer than $\min(k, N_\sigma)$ results.

```
Algorithm 4: Pre-Filter Brute-Force Scan
Input: Query vector q, Base matrix V (N x d), Target k, DenseBitset bitset
Output: Top-k neighbor list [(distance, id)] sorted ascending

1: heap ← MaxHeap(capacity = k)
2: for word_idx from 0 to bitset.num_words() - 1 do
3:     word ← bitset.words[word_idx]
4:     while word != 0 do
5:         bit_pos ← trailing_zeros(word)
6:         vec_id ← (word_idx << 6) + bit_pos
7:         vec_data ← V.get_row(vec_id)
8:         dist ← l2_squared_8way(q, vec_data)
9:         cand ← Candidate(dist, vec_id)
10:        if heap.len() < k then
11:            heap.push(cand)
12:        else if cand < heap.peek() then
13:            heap.pop()
14:            heap.push(cand)
15:        end if
16:        word ← word & (word - 1)  // Clear lowest set bit
17:    end while
18: end for
19: return heap.into_sorted_ascending_vec()
```

---

### 3.4 Comparative Complexity Analysis

| Dimension | Strategy 1: Post-Filtering | Strategy 2: In-Traversal (Bridge) | Strategy 3: Pre-Filter Brute Force |
|---|---|---|---|
| **Time Complexity** | $\mathcal{O}\left( \frac{ef}{\sigma} \cdot M \cdot d \right)$ | $\mathcal{O}\left( \frac{ef}{\sigma^\alpha} \cdot M \cdot d \right), \ \alpha \in (0, 1)$ | $\mathcal{O}(\sigma \cdot N \cdot d + \sigma N \log k)$ |
| **Distance Evaluations** | Grows roughly as $1/\sigma$ (measured) | Grows with $\sigma^{-\beta}$ (measured) | Exactly $\sigma \cdot N$ |
| **Memory Access Pattern** | Random pointer chasing (DRAM bound) | Random pointer chasing (DRAM bound) | Streaming / Sequential contiguous |
| **Vectorization** | Per-vector hops | Per-vector hops | Sequential scan (compiler auto-vectorization only) |
| **Recall at $\sigma = 50\%$** | Measured | Measured | Exact ($100.0\%$) |
| **Recall at $\sigma = 0.1\%$** | Measured (may truncate) | Measured (may truncate) | Exact ($100.0\%$) |
| **Cluster-Correlated Robustness** | Expected to degrade (spatial starvation); measured | Expected to degrade less; measured | Immune (exhaustive over matches) |
| **Allocation in Hot Path** | Zero if preallocated to $K'_{\max}$ and the existing search path is checked (T3.4) | Zero (preallocated scratch heaps) | Zero (fixed-size heap of size $k$) |

Entries marked "measured" are filled from Phase 5 JSON. No value in this table is assumed.

---

## 4. Theoretical and Empirical Crossover Selectivity Threshold Methodology

### 4.1 Theoretical Cost Models

The models below are hypotheses that give the crossover a predicted shape. Every constant is a parameter fitted from measurements (T3.6 microbenchmark and T3.9 runs). No value is assumed here.

#### Cost Model: Pre-Filter Brute-Force Scan ($C_{\text{brute}}$)
$$C_{\text{brute}}(\sigma, N, d) = T_{\text{scan}}(N) + \sigma N \cdot t_{\text{cand}}$$
Where:
- $T_{\text{scan}}(N) = \frac{N}{64} \cdot t_{\text{word}}$ is the bitset traversal cost.
- $t_{\text{cand}} = t_{\text{dist}}(d) + t_{\text{heap}}(k)$ is the per-matching-vector cost (distance kernel plus heap update).
- $t_{\text{word}}$, $t_{\text{dist}}$, $t_{\text{heap}}$ come from a single-threaded microbenchmark on the target machine (pinned core, Phase 4 protocol).

#### Cost Model: In-Traversal Graph Traversal ($C_{\text{graph}}$)
$$C_{\text{graph}}(\sigma, ef) = N_{\text{eval}}(\sigma, ef) \cdot t_{\text{graph\_cand}} + N_{\text{visited}} \cdot t_{\text{pred}} + C_{\text{zoom}}$$
$$N_{\text{eval}}(\sigma, ef) \approx ef \cdot M \cdot \sigma^{-\beta}$$
Where $\beta$ is a fitted dilation exponent (the power-law form is a modelling assumption, checked with a goodness-of-fit report against the measured distance-evaluation counters) and $t_{\text{graph\_cand}}$ is the measured per-evaluation cost including cache misses (measured $C_{\text{graph}}$ divided by the measured evaluation count).

---

### 4.2 Predicted Shape of the Crossover
The crossover selectivity $\sigma^*$ is where pre-filter brute force and graph traversal cost the same. Equating the two models and dropping the small $T_{\text{scan}}$, $C_{\text{zoom}}$ and predicate terms:
$$\sigma^* N \cdot t_{\text{cand}} \approx ef \cdot M \cdot (\sigma^*)^{-\beta} \cdot t_{\text{graph\_cand}} \implies (\sigma^*)^{1+\beta} \approx \frac{ef \cdot M}{N} \cdot \frac{t_{\text{graph\_cand}}}{t_{\text{cand}}}$$

What this predicts qualitatively (testable, not assumed):
1. $\sigma^*$ falls as $N$ grows, because the right-hand side scales as $1/N$.
2. $\sigma^*$ rises with $ef$ and with $M$ (a costlier graph search favours brute force at higher selectivity).
3. $\sigma^*$ rises with the cost ratio $t_{\text{graph\_cand}} / t_{\text{cand}}$.
4. If the right-hand side is $\ge 1$, brute force wins at every selectivity for that configuration. This is the expected situation at small $N$, which is one reason 10k is a smoke test only.

No numeric $\sigma^*$ is stated in this document. The prediction is compared with the measured crossover in T3.11, and a mismatch is reported, not tuned away.

---

### 4.3 Empirical Discovery Protocol

1. **Experimental Matrix**:
   - Result scales: $N \in \{100\text{k}, 1\text{M}\}$ (SIFT base prefix and full set). A 10k run (`siftsmall`) is a pipeline smoke test and is not reported as a result.
   - Queries: the 10,000 SIFT1M queries, $k = 10$.
   - Selectivity ladder (12 points):
     $$\sigma \in \{0.05\%, 0.1\%, 0.2\%, 0.5\%, 1.0\%, 2.0\%, 3.5\%, 5.0\%, 7.5\%, 10.0\%, 20.0\%, 50.0\%\}$$
   - Metadata modes: Independent (Mode A) and Cluster-Correlated (Mode B, $\kappa = 1.0$).
   - Index config: required $(M=16, efC=100)$. Optional second config $(M=32, efC=200)$ if Dev approves the extra builds.
   - Strategies: Strategy 1 capped, Strategy 1 uncapped, Strategy 2, Strategy 3.
   - efSearch ladder (Strategies 1 and 2): $\{20, 50, 100, 200, 400, 600\}$ (subset of the Phase 4 ladder).
   - Sensitivity runs (one config, one mode, one scale): $H_{\max} \in \{1, 3, 10\} \cdot ef$, and $\gamma$ at the formula default plus two fixed values.
2. **Execution Environment**: same protocol as Phase 4. Single core pinned (record the real hardware and affinity mask in the JSON), 5 timed repetitions after 1 untimed warmup, clean commit (`git_dirty` false), with full JSON provenance. Record per-query latency (p50, p99), QPS, distance counters and failure flags.
3. **Time Budget First**: before any run, T3.7 produces a budget proposal (pilot at 100k, extrapolation, worst case, disk use) for Dev, as T2.1 did for Phase 4. The grid is large (strategies x efSearch x ladder x modes x scales), so Dev may cut the matrix at that point.
4. **Crossover Extraction at Matched Recall**: brute force has Recall@10 = 1 by construction, so comparing it with graph latency at an arbitrary efSearch is not a fair test. For each (selectivity, mode, scale), take the Strategy 2 latency at the smallest efSearch that reaches the recall target (primary 0.95, secondary 0.99, both proposed, Dev decides). If no efSearch on the ladder reaches the target, record "target not reached" (brute force wins at that point). The crossover is the bracket $[\sigma_{\text{lo}}, \sigma_{\text{hi}}]$ between adjacent ladder points where the sign of (graph latency minus brute-force latency) changes, plus a linear interpolation in log-log space inside the bracket. Splines are not used: with 12 noisy points they can invent roots. Report each mode and scale separately, including a "no crossover in range" outcome where it occurs.

---

### 4.4 Optional, Deferred: Adaptive Hybrid Dispatcher (needs Dev approval)
Outside the M3 brief, and only meaningful after the crossover is measured (T3.11). If Dev approves it later, the dispatcher below is a starting point. It is not part of the G4 criteria:

```rust
pub struct AdaptiveFilterRouter {
    /// Measured crossover count threshold below which brute force was faster
    crossover_threshold_count: usize,
}

impl AdaptiveFilterRouter {
    #[inline]
    #[must_use]
    pub fn select_strategy(
        &self,
        matching_candidate_count: usize,
        k: usize,
    ) -> FilterStrategy {
        if matching_candidate_count <= k {
            // Trivial Case: The entire matching set fits within top-k
            FilterStrategy::ExhaustiveBitsetScan
        } else if matching_candidate_count < self.crossover_threshold_count {
            // Strategy 3: Pre-filter brute force is faster than graph traversal
            FilterStrategy::ExhaustiveBitsetScan
        } else {
            // Strategy 2: In-traversal graph traversal is faster
            FilterStrategy::InTraversalHnsw
        }
    }
}
```

---

## 5. Ground Truth Generation & Evaluation Metrics

### 5.1 Authoritative Filtered Ground Truth Algorithm (`FilteredExactKnn`)
To evaluate Recall@10 with absolute mathematical precision, all approximate runs are scored against an authoritative exact ground truth engine. 

#### Deterministic Canonical Tie-Breaking
Matching Phase 1 standards, all distance comparisons break ties canonically on `(distance, id)`:
$$(d_a, \text{id}_a) < (d_b, \text{id}_b) \iff d_a < d_b \lor (d_a == d_b \land \text{id}_a < \text{id}_b)$$
This guarantees bit-exact ground truth across all compilers, hardware architectures, and operating systems.

```rust
pub struct FilteredExactKnn {
    metric: Metric,
    base_vectors: RowMatrix<f32>,
}

impl FilteredExactKnn {
    /// Computes bit-exact Top-k nearest neighbors satisfying `predicate`.
    pub fn search_filtered<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
    ) -> Result<Vec<(f32, usize)>> {
        let mut heap = BinaryHeap::<HeapCandidate>::with_capacity(k);
        let num_base = self.base_vectors.rows();

        for id in 0..num_base {
            if predicate.matches(id as u32) {
                let row = self.base_vectors.row(id);
                let dist = match self.metric {
                    Metric::L2Squared => l2_squared(query, row),
                    Metric::Euclidean => l2_squared(query, row).sqrt(),
                    Metric::InnerProduct => -inner_product(query, row),
                };
                let cand = HeapCandidate { sort_key: dist, id: id as i32 };
                if heap.len() < k {
                    heap.push(cand);
                } else if cand < *heap.peek().unwrap() {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }
        // Extract and sort ascending
        let mut results = Vec::with_capacity(heap.len());
        while let Some(cand) = heap.pop() {
            results.push((cand.sort_key, cand.id as usize));
        }
        results.reverse();
        Ok(results)
    }
}
```

#### Self-Verification Invariant
The ground truth engine enforces the automated self-verification invariant:
$$\text{Recall@}k(\text{GT}_{\mathcal{P}}, \ \text{GT}_{\mathcal{P}}) \equiv 1.000000 \quad (\text{Self-Parity})$$
Self-parity is trivial, so there is also an independent check: on the 10k smoke data, `FilteredExactKnn` is compared with a NumPy brute force (as the Phase 1 parity tests do) at all four selectivity tiers.

Ground truth is computed once per (scale, mode, selectivity) over the full query set, written as `.ivecs` with a provenance JSON (git commit, seeds, bitset popcount, predicate hash) and cached like the Phase 4 prefix ground truth. Its cost is $Q \cdot \sigma N$ distance evaluations per predicate (largest at $\sigma = 50\%$, 1M), estimated in the T3.7 budget proposal, not here.

---

### 5.2 Comprehensive Evaluation Metrics Suite

Every Phase 5 benchmark run records six metrics:

1. **Recall@10 vs Selectivity**:
   $$\text{Recall@}10 = \frac{|\mathcal{R}_{\text{approx}}^{(10)} \cap \mathcal{R}_{\text{exact}}^{(10)}|}{10}$$
   Evaluated across all four selectivity tiers ($\sigma \in \{50\%, 10\%, 1\%, 0.1\%\}$).
2. **Latency Distributions & Throughput**:
   - Single-threaded query throughput: $\text{QPS} = \frac{Q}{\sum_{i=1}^Q t_i}$.
   - Latency percentiles: Mean, p50 (median), p90, p95, and p99 (tail latency in ms).
3. **Candidate Truncation Rate (Failure Rate)**:
   The fraction of queries that return fewer than $k$ results when the dataset contains at least $k$ matching vectors:
   $$\text{Truncation Rate} = \frac{1}{Q} \sum_{q=1}^Q \mathbb{I}\left( |\mathcal{R}_{\text{approx}}^{(q)}| < \min(k, \ N_\sigma) \right)$$
   *(No target. Strategy 3 is 0.0% by construction for bitset predicates; Strategies 1 and 2 report the measured rate. Each failure is counted by its flag: `PostFilterCapReached`, `HopBudgetExhausted`, `EvalLimitReached`.)*
4. **Distance Evaluations Counter**:
   Instrumented total distance kernel calls per query, categorized into:
   - Valid distance computations (matching candidates).
   - Bridge distance computations (non-matching routing vertices).
5. **Graph Traversal Efficiency Ratio**:
   $$\eta_{\text{graph}} = \frac{|\mathcal{V}_{\text{visited}} \cap \mathcal{S}_{\text{match}}|}{|\mathcal{V}_{\text{visited}}|}$$
   Measures the fraction of graph traversal effort expended on valid vs. discarded nodes.
6. **Latency at Matched Recall**:
   For each strategy, the latency at the smallest efSearch reaching Recall@10 of 0.95 (and 0.99), or "not reached". This is the quantity used for the crossover (Section 4.3).

---

## 6. Rust API / Architecture Design

### 6.1 Rust Trait & Data Structure Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE RUST TRAIT: FilterPredicate                │
├────────────────────────────────────────────────────────────────────────┤
│ + matches(&self, id: u32) -> bool                                      │
│ + bitset(&self) -> Option<&DenseBitset>                                │
│ + count_matching(&self, total: usize) -> Option<usize>                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌─────────────────────────────┐           ┌──────────────────────────────┐
│        DenseBitset          │           │       LambdaPredicate<F>     │
├─────────────────────────────┤           ├──────────────────────────────┤
│ - words: Vec<u64>           │           │ - closure: F (Fn(u32)->bool) │
│ - len: usize                │           │ - estimated_selectivity: f64 │
├─────────────────────────────┤           └──────────────────────────────┘
│ + matches(id: u32) -> bool  │
│ + count_ones() -> usize     │
│ + and(&self, other)         │
│ + or(&self, other)          │
│ + iter_ones() -> BitIter    │
└─────────────────────────────┘
```

#### The `FilterPredicate` Trait
Defined in `src/filter.rs`, `FilterPredicate` provides a zero-overhead interface for arbitrary query constraints:

```rust
//! Filter predicates and bitset acceleration structures for Milestone M3.

/// Interface for evaluating query filter constraints over database vector IDs.
pub trait FilterPredicate: Send + Sync {
    /// Evaluates whether the vector with 0-indexed database ID `id` satisfies the filter.
    fn matches(&self, id: u32) -> bool;

    /// Optional reference to an underlying contiguous bitset.
    /// When available, enables popcount-based candidate counting and
    /// the Strategy 3 brute-force path.
    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        None
    }

    /// Optional count of total matching candidates.
    #[inline]
    fn count_matching(&self, total: usize) -> Option<usize> {
        self.bitset().map(|b| b.count_ones())
    }
}
```

#### `DenseBitset` Implementation
```rust
/// Cache-line aligned dense bitset storing 1 bit per vector ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseBitset {
    words: Vec<u64>,
    len: usize,
}

impl DenseBitset {
    /// Allocates an all-zero bitset for `len` elements.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        let num_words = (len + 63) / 64;
        Self {
            words: vec![0_u64; num_words],
            len,
        }
    }

    /// Number of elements tracked by the bitset.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the bitset contains zero elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Tests whether bit `id` is set.
    #[inline]
    #[must_use]
    pub fn get_bit(&self, id: usize) -> bool {
        if id >= self.len {
            return false;
        }
        let word_idx = id >> 6;
        let bit_idx = id & 63;
        (self.words[word_idx] & (1_u64 << bit_idx)) != 0
    }

    /// Sets or clears bit `id`.
    #[inline]
    pub fn set_bit(&mut self, id: usize, value: bool) {
        assert!(id < self.len, "Bit index out of bounds");
        let word_idx = id >> 6;
        let bit_idx = id & 63;
        if value {
            self.words[word_idx] |= 1_u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1_u64 << bit_idx);
        }
    }

    /// Total count of active (1) bits (`u64::count_ones` per word).
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Performs in-place bitwise AND (`self &= other`).
    pub fn bitand_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "Bitset length mismatch in AND");
        for (w_self, w_other) in self.words.iter_mut().zip(&other.words) {
            *w_self &= *w_other;
        }
    }

    /// Performs in-place bitwise OR (`self |= other`).
    pub fn bitor_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "Bitset length mismatch in OR");
        for (w_self, w_other) in self.words.iter_mut().zip(&other.words) {
            *w_self |= *w_other;
        }
    }

    /// Returns a zero-allocation iterator over all active (1) vector IDs.
    #[inline]
    pub fn iter_ones(&self) -> BitsetOneIterator<'_> {
        BitsetOneIterator {
            bitset: self,
            word_idx: 0,
            current_word: self.words.first().copied().unwrap_or(0),
        }
    }
}

impl FilterPredicate for DenseBitset {
    #[inline]
    fn matches(&self, id: u32) -> bool {
        self.get_bit(id as usize)
    }

    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        Some(self)
    }
}
```

#### Query-Time Settings (`FilteredSearchParams`) and Result Flags
```rust
/// Query-time settings for filtered search. Not part of `HnswConfig` (build settings).
#[derive(Debug, Clone)]
pub struct FilteredSearchParams {
    pub ef_search: usize,
    /// Strategy 1 over-fetch factor. None = confidence formula with delta = 0.001.
    pub overfetch_factor: Option<f64>,
    /// Strategy 1 pool cap K'_max (default 4096; N for the uncapped arm).
    pub overfetch_cap: usize,
    /// Strategy 2 consecutive bridge-hop budget. None = 3 * ef_search.
    pub max_bridge_hops: Option<usize>,
    /// Max predicate evaluations per query, applied to predicates without a bitset.
    /// None = no limit (default value to be proposed in the T3.7 budget).
    pub eval_limit: Option<usize>,
    /// Bitset-backed predicates only: allow a fallback to Strategy 3 when a strategy
    /// gives up. Always false in benchmark arms.
    pub allow_bruteforce_fallback: bool,
}

/// Why a query returned fewer than k results (or used a fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FilteredSearchFlags {
    pub post_filter_cap_reached: bool,
    pub hop_budget_exhausted: bool,
    pub eval_limit_reached: bool,
    pub used_bruteforce_fallback: bool,
}

pub struct FilteredSearchOutput {
    pub results: Vec<(f32, usize)>,
    pub flags: FilteredSearchFlags,
}
```

#### Zero-Allocation Query Scratch Buffers (`FilteredSearchScratch`)
To guarantee zero heap allocations in inner search loops:
```rust
/// Reusable per-thread scratch buffers for filtered HNSW queries.
#[derive(Debug, Clone, Default)]
pub struct FilteredSearchScratch {
    /// Visited tracking bit/tag array
    pub tracker: VisitedTracker,
    /// Priority queue for candidate exploration (Min-Heap)
    pub frontier: BinaryHeap<Reverse<Candidate>>,
    /// Priority queue for matching results (Max-Heap)
    pub matched_results: BinaryHeap<Candidate>,
    /// Temporary candidate collection buffer
    pub raw_candidates: Vec<Candidate>,
    /// Over-fetch staging buffer for Strategy 1
    pub overfetch_buffer: Vec<(f32, usize)>,
}

impl FilteredSearchScratch {
    /// Preallocates every buffer for the largest query the caller will issue.
    /// Nothing may grow inside a query: a query that would exceed a capacity stops
    /// and sets a flag instead of reallocating.
    #[must_use]
    pub fn new(num_nodes: usize, max_ef: usize, overfetch_cap: usize) -> Self {
        Self {
            tracker: VisitedTracker::new(num_nodes),
            // Worst case: every node is pushed to the frontier once.
            frontier: BinaryHeap::with_capacity(num_nodes),
            matched_results: BinaryHeap::with_capacity(max_ef + 1),
            raw_candidates: Vec::with_capacity(overfetch_cap),
            overfetch_buffer: Vec::with_capacity(overfetch_cap),
        }
    }
}
```

#### Extended Search Signatures on `HnswIndex`
The existing unfiltered `search_knn` is untouched (class PRESERVING: graph fingerprint and unfiltered results stay bit-identical).
```rust
pub enum FilterStrategy {
    PostFilterOverfetch,
    InTraversal,
    PreFilterBruteForce,
    // Adaptive: optional, see Section 4.4
}

impl HnswIndex {
    /// Executes filtered approximate nearest neighbor search using the specified strategy.
    pub fn search_knn_filtered<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        strategy: FilterStrategy,
        params: &FilteredSearchParams,
        scratch: &mut FilteredSearchScratch,
    ) -> Result<FilteredSearchOutput> {
        // Dispatch to search_filtered_post / search_filtered_intraversal /
        // search_filtered_prefilter_brute, each filling `flags` as described in Section 3.
    }
}
```
Strategy 1 reuses the existing unfiltered search. T3.4 must state whether that path allocates per query. Zero-allocation is claimed only for code paths that were checked.

---

### 6.2 Python Access (deferred, optional)

PyO3 bindings are removed from M3. The library takes no new default dependencies, the repo has no Python extension build today, and the existing parity tests call the Rust binaries through `subprocess`. Phase 5 follows the same pattern: the `phase5` binary writes JSON and `.ivecs`, and Python scripts (plots, parity checks against a NumPy brute force) read them. If bindings are later added (for example for the Phase 7 FAISS comparison), they go in a separate optional crate or feature with their own design note.

---

## 7. Implementation Roadmap & Gate G4 Sign-Off Criteria

### 7.1 Detailed Task Card Breakdown (Milestone M3)

| Task ID | Component / Deliverable | Class | Files Allowed to Change | Acceptance Criteria |
|:---:|:---|:---:|:---|:---|
| **T3.1** | `DenseBitset` & `FilterPredicate` | `PRESERVING` | `src/filter.rs`, `src/lib.rs` | Unit tests for every public method, including edge cases (empty, N not a multiple of 64, all ones, single bit); `iter_ones` yields ascending ids. |
| **T3.2** | Deterministic metadata generator (Mode A, Mode B) | `MEASUREMENT-ONLY` | `src/dataset.rs`, `src/bin/generate_metadata.rs`, `Cargo.toml` | Popcount equals floor(σN) for every tier and mode; same seeds give byte-identical bitsets on repeat runs; Mode B active cluster ids recorded in JSON. |
| **T3.3** | `FilteredExactKnn` | `MEASUREMENT-ONLY` | `src/exact_knn.rs` | Self-parity 1.000000; matches a NumPy brute force on 10k smoke data at all four tiers; canonical `.ivecs` with provenance. |
| **T3.4** | Strategy 1: post-filter with over-fetch | `PRESERVING` | `src/hnsw.rs` | Widening loop per Section 3.1 with cap and flag; states whether the existing search path allocates; graph fingerprint and unfiltered results unchanged. |
| **T3.5** | Strategy 2: in-traversal filtering | `PRESERVING` | `src/hnsw.rs` | Matches Algorithm 3 including the hop budget; flags set correctly; fingerprint equal and unfiltered results bit-identical; truncation rate reported (not required to be zero). |
| **T3.6** | Strategy 3 + cost microbenchmark | `PRESERVING` | `src/exact_knn.rs`, `src/hnsw.rs`, `src/bin/phase5.rs` | Results equal `FilteredExactKnn`; microbenchmark JSON gives $t_{word}$, $t_{dist}$, $t_{heap}$; no throughput target. |
| **T3.7** | Matrix and time-budget proposal (no runs) | `MEASUREMENT-ONLY` | `results/phase5/PROPOSAL.md` | Pilot at 100k, extrapolated run time, worst case, disk use, metadata and ground-truth generation cost, default `eval_limit`. For Dev approval (proposed gate G4.5, Dev decides whether to fold it into G4). |
| **T3.8** | Phase 5 benchmark runner CLI | `MEASUREMENT-ONLY` | `src/bin/phase5.rs`, `src/benchmark.rs`, `Cargo.toml` | Resumable grid runner like phase4; one JSON per point with provenance, flags, counters; Section 4.3 matrix incl. sensitivity runs; 10k smoke mode. |
| **T3.9** | Run the approved matrix | `MEASUREMENT-ONLY` | `results/phase5/*.json` | Clean commit, `git_dirty` false, no concurrent load; every point has provenance. |
| **T3.10**| Plots (`plot_phase5.py`) | `MEASUREMENT-ONLY` | `scripts/plot_phase5.py` | Recall@10 vs selectivity, latency vs selectivity, failure rate vs selectivity, matched-recall latency vs selectivity with the crossover bracket; captions with hardware, scale and repetitions. |
| **T3.11**| Crossover analysis, README, CLAIMS | `MEASUREMENT-ONLY` | `results/CLAIMS.md` (Section 11), `README.md` | Every number maps to a source JSON; predicted vs measured crossover compared and any mismatch reported. |
| Optional | Adaptive router; PyO3 bindings | `PRESERVING` | (needs Dev approval) | Not part of M3 unless Dev adds them after G4. |

---

### 7.2 Gate G4 Sign-Off Criteria (Prerequisites to Coding)

Prior to committing implementation code for Milestone M3, Dev must inspect and sign off on this design specification against the following acceptance criteria:

1. **Deterministic Metadata Invariant**:
   - [ ] Independent and Cluster-Correlated synthetic metadata generation algorithms are formally specified with fixed seeds and exact count calibration.
   - [ ] Selectivity tiers $\{50\%, 10\%, 1\%, 0.1\%\}$ are explicitly addressed with analytical bounds.
2. **Strategy Architectural Soundness**:
   - [ ] Strategy 1 defines over-fetch sizing $\lceil k / \sigma \rceil \cdot \gamma$, the pool cap $K'_{\max}$, iterative widening bounds and the `PostFilterCapReached` flag.
   - [ ] Strategy 2 decouples graph routing candidates from result set admittance and defines the bridge-hop budget and the `HopBudgetExhausted` flag.
   - [ ] Strategy 3 defines a bitset-driven exact scan using safe Rust primitives.
3. **Crossover Selectivity Rigor**:
   - [ ] Cost models predict the shape of $\sigma^*$ with every constant left as a measured parameter; no numeric $\sigma^*$ is claimed before measurement.
   - [ ] Measurement protocol at 100k and 1M (10k smoke only) defines core pinning, repetitions and a matched-recall crossover.
4. **Safety & Zero-Cost Guarantees**:
   - [ ] `#![forbid(unsafe_code)]` and `std`-only invariants are preserved.
   - [ ] `FilteredSearchScratch` is preallocated for the largest query; code paths that allocate are listed explicitly.
   - [ ] Canonical `(distance, id)` tie-breaking ensures 100% deterministic, platform-independent ground truth.
5. **Open Decisions for Dev**:
   - [ ] Decisions in Section 1.4 confirmed or changed.
   - [ ] Recall targets for the matched-recall crossover (proposed 0.95 and 0.99).
   - [ ] Default `eval_limit` (to be proposed in T3.7).
   - [ ] Whether the second index config $(M=32, efC=200)$ is in scope.
   - [ ] Whether T3.7 approval is its own gate (G4.5) or part of G4.
