# Hierarchical Navigable Small World (HNSW) Algorithm Specification

**Reference**: Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 42(4), 824-836. arXiv:1603.09320.

---

## 1. Executive Overview

Approximate Nearest Neighbor (ANN) search is foundational to modern vector databases, dense semantic retrieval, multimodal embedding search, and recommendation systems. While brute-force exact K-Nearest Neighbor (KNN) search scales linearly as $\mathcal{O}(N \cdot D)$—becoming computationally prohibitive as dataset size $N$ reaches millions or billions—graph-based indices offer logarithmic query scaling $\mathcal{O}(\log N)$.

Among graph-based algorithms, the **Hierarchical Navigable Small World (HNSW)** graph stands as the gold standard for high recall and query throughput. HNSW generalizes the multi-level probabilistic skip-list to high-dimensional metric spaces, overcoming the fundamental curse of dimensionality that degrades traditional tree partition structures (e.g., KD-trees, VP-trees).

---

## 2. Multi-Layer Hierarchical Graph Structure

### 2.1 The Skip-List Analog in Metric Spaces

In 1D ordered data structures, a **Skip-List** (Pugh, 1990) builds an efficient hierarchy over a linked list:
- The base layer $L_0$ contains all $N$ elements in sorted order.
- Each successive layer $L_l$ subsamples elements with probability $p = 1/M$.
- Search starts at the topmost sparse layer with long express pointers. When a pointer overshoots the target key, the search drops down to the corresponding node in the layer below, repeating down to layer 0.
- This skip-list hierarchy yields logarithmic search time $\mathcal{O}(\log N)$ without requiring expensive tree balancing.

```
Layer 2:  [Node 1] ---------------------------------------------> [Node 9]
                \                                                       \
Layer 1:  [Node 1] -------------------> [Node 5] ---------------------> [Node 9]
                \                         \                               \
Layer 0:  [Node 1] -> [Node 2] -> [Node 3] -> [Node 5] -> [Node 7] -> [Node 8] -> [Node 9]
```

#### High-Dimensional Challenge
In metric spaces $\mathbb{R}^D$ where $D \gg 1$, there is no natural total ordering (no 1D monotonic $<, >$ relationship) that preserves spatial proximity. Furthermore, spatial partitioning trees degenerate into linear brute-force scans as $D > 20$ because hyperplanes fail to separate high-dimensional clusters efficiently.

#### HNSW's Solution
HNSW generalizes the skip-list principle to arbitrary metric spaces by replacing:
1. **1D linked lists** with **Navigable Small World (NSW) proximity graphs**.
2. **Scalar key comparisons** with **Metric distance evaluations** (e.g., Squared Euclidean distance $L_2^2$, inner product, or cosine distance).
3. **Over-the-target pointer drops** with **Local-minimum detection in greedy graph routing**.

Each layer $l \in [0, l_{max}]$ in HNSW is a proximity graph $G_l = (V_l, E_l)$ forming a nested hierarchy:
$$V_{l_{max}} \subset V_{l_{max}-1} \subset \dots \subset V_1 \subset V_0 = V$$
where $V_0 = V$ contains all $N$ indexed vectors. Higher layers contain exponentially fewer vectors, maintaining links with longer spatial span in metric space.

---

### 2.2 Probabilistic Layer Assignment & Mathematical Derivation

When a new vector $v$ is inserted into an HNSW index, its maximum layer $l_{max}(v)$ is determined probabilistically using an exponential decay distribution.

#### Continuous and Discrete Distribution
The probability that a vector is present in layer $l \ge 0$ is governed by:
$$P(\text{layer} \ge l) = \exp\left(-\frac{l}{m_L}\right)$$

where $m_L$ is the **layer scale normalization factor** (the scale parameter of the exponential distribution).

To sample the discrete maximum layer $l$ for an element, let $U \sim \text{Uniform}(0, 1)$ be a standard uniform random variable. Using the inverse transform sampling method:
$$\exp\left(-\frac{l}{m_L}\right) = U \implies -\frac{l}{m_L} = \ln(U) \implies l = -m_L \ln(U)$$

Taking the integer floor yields the exact formula specified by Malkov & Yashunin:
$$\boxed{l_{max}(v) = \left\lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \right\rfloor}$$

#### Derivation of $m_L = \frac{1}{\ln(M)}$
Why is $m_L = 1 / \ln(M)$ optimal?
In a classical skip-list with branching factor $M$, the fraction of elements surviving from layer $l-1$ to layer $l$ is:
$$\frac{|V_l|}{|V_{l-1}|} = \frac{1}{M}$$

For HNSW, the conditional probability that a node present at layer $l-1$ also appears at layer $l$ is:
$$\frac{P(\text{layer} \ge l)}{P(\text{layer} \ge l-1)} = \frac{\exp(-l / m_L)}{\exp(-(l-1) / m_L)} = \exp\left(-\frac{1}{m_L}\right)$$

Equating this ratio to the target downsampling factor $1/M$:
$$\exp\left(-\frac{1}{m_L}\right) = \frac{1}{M} \implies -\frac{1}{m_L} = -\ln(M) \implies \boxed{m_L = \frac{1}{\ln(M)}}$$

#### Consequence on Graph Layer Populations
- **Layer 0**: $|V_0| = N$ ($100\%$ of data points).
- **Layer 1**: Expected $|V_1| = N \cdot \frac{1}{M}$.
- **Layer 2**: Expected $|V_2| = N \cdot \frac{1}{M^2}$.
- **Layer $l$**: Expected $|V_l| = N \cdot M^{-l}$.
- **Expected maximum layer height**:
  $$L \approx \frac{\ln(N)}{\ln(M)} = \log_M(N)$$

For instance, with $N = 1,000,000$ (SIFT1M) and $M = 16$:
$$m_L = \frac{1}{\ln(16)} \approx 0.36067$$
$$L \approx \log_{16}(10^6) \approx 4.98 \implies \approx 5 \text{ layers above Layer 0.}$$
At layer 5, the expected node count is $10^6 / 16^5 \approx 0.95 \approx 1$ node (which acts as the top-level global entry point).

---

### 2.3 Degree Bounds and Asymmetric Layer Densities

Each node in layer $l$ maintains an adjacency list of bidirectional links. To maintain strict memory bounds and ensure predictable query latency, HNSW enforces maximum degree caps:

| Parameter | Symbol | Typical Range | Description |
|---|---|---|---|
| Max Degree for Upper Layers ($l > 0$) | $M$ (or $M_{max}$) | $12 \dots 64$ (default: 16) | Maximum bidirectional links per node in layers $l \ge 1$. |
| Max Degree for Base Layer ($l = 0$) | $M_{max0}$ | $2 \cdot M$ (default: 32) | Maximum bidirectional links per node in ground layer 0. |
| Neighbor Construction Target | $M$ | Equal to upper degree $M$ | Number of neighbors selected during node insertion. |

#### Why $M_{max0} = 2M$?
Malkov & Yashunin established empirically and theoretically that doubling the maximum degree at layer 0 ($M_{max0} = 2M$) provides significant recall benefits:
1. **Zero-Downsampling Boundary**: Ground layer 0 contains the entire dataset ($100\%$ of vectors). While upper layers serve solely for coarse navigation towards the neighborhood of the query, layer 0 is solely responsible for fine-grained discrimination between nearest neighbors.
2. **Connectivity & Clustering in High Dimensions**: In high dimensions ($D=128$), Voronoi cells share boundaries with many neighboring partitions. Doubling $M_{max0}$ boosts the local clustering coefficient without adding memory overhead to upper layers (which contain only $\approx 1/M$ of all nodes).
3. **Recall vs. Memory Optimization**: Allocating additional edge capacity strictly to layer 0 doubles graph connectivity where precision matters most, while overall index memory remains bounded by $\mathcal{O}(N \cdot M_{max0})$.

---

## 3. Greedy Walk Navigation & Search Dynamics

### 3.1 Entry Point Management

The HNSW structure maintains a single global entry point node, denoted as $ep$, located at the topmost layer $l_{top}$.
- **Initialization**: When the very first node $v_0$ is inserted, $ep \leftarrow v_0$, and $l_{top} \leftarrow l_{max}(v_0)$.
- **Dynamic Update**: During subsequent insertions, if a newly inserted node $v_{new}$ has an assigned maximum layer $l_{new} > l_{top}$, the global entry point is updated:
  $$ep \leftarrow v_{new}, \quad l_{top} \leftarrow l_{new}$$
  This ensures that navigation always initiates from the highest, sparsest available layer.

```
                    TOP-DOWN GREEDY WALK (ZOOM-IN)
                    
Layer 2 (l_top)       [ep] ----------------------------> [v_closest_2]
                                                                |
                                                           drop down
                                                                v
Layer 1               ... --------------> [v_closest_1] <--------
                                                |
                                           drop down
                                                v
Layer 0 (Base)        ... --> [Beam Search Queue ef=efSearch] --> [Top K Results]
```

---

### 3.2 Top-Down Zoom-In ($ef = 1$ Greedy Routing)

Search begins at the topmost layer $l_{top}$ and proceeds downward to layer 1. At each upper layer $l \in [l_{top}, 1]$, the algorithm executes a **1-greedy search** (beam search with capacity $ef = 1$):

1. Initialize current node $curr \leftarrow ep$.
2. Compute distance $d_{min} \leftarrow \text{dist}(curr, q)$.
3. Iterate over all neighbors $n \in \text{neighbors}_l(curr)$:
   - Compute $d \leftarrow \text{dist}(n, q)$.
   - If $d < d_{min}$, update $curr \leftarrow n$ and $d_{min} \leftarrow d$.
4. If a strictly closer neighbor was found, repeat Step 3 from the new $curr$.
5. When no neighbor of $curr$ is closer to $q$ than $curr$ itself, a **local minimum at layer $l$** has been reached.
6. Set the entry point for layer $l - 1$ to be this local minimum:
   $$ep \leftarrow curr$$
7. Decrement layer: $l \leftarrow l - 1$, and repeat until layer 0 is reached.

#### Computational Cost of Top-Down Zoom-In
Because layers $l \ge 1$ are exponentially sparse, each greedy hop traverses long spatial distances in $\mathbb{R}^D$. Furthermore, because $ef = 1$, each greedy step inspects at most $M$ neighbors. Across all $\mathcal{O}(\log_M N)$ upper layers, the total number of distance calculations in the top-down phase is bounded by:
$$T_{zoom} = \mathcal{O}(M \log_M N)$$
This zoom-in phase converges in microseconds, positioning the search cursor directly into the local metric basin containing query $q$.

---

### 3.3 Base Layer 0 Search: Beam Search with Priority Queues

Once the entry point reaches Layer 0, the search switches from a 1-greedy walk to a **multi-candidate beam search** (Algorithm 2: `SEARCH-LAYER`), parameterized by capacity $ef$ ($efConstruction$ during graph build; $efSearch$ during query).

#### Data Structures Used in Layer 0 Search
1. **Candidate Queue $C$ (Min-Heap by distance)**:
   - Stores candidate nodes to be explored, ordered ascending by distance: smallest distance at the top.
   - Allows extracting the nearest unexplored node in $\mathcal{O}(\log |C|)$ time.
2. **Dynamic Result Pool $W$ (Max-Heap by distance)**:
   - Stores the current best $ef$ nodes found so far, ordered descending by distance: largest distance at the top.
   - Peek operation $f = \text{top}(W)$ returns the current furthest element in the pool ($d(f, q) = \max_{u \in W} d(u, q)$).
   - Capacity is bounded to at most $ef$ elements.
3. **Visited Set $V$ (Hash Set or Bitset/Visited Array)**:
   - Tracks node IDs that have already been evaluated, guaranteeing no node is distance-evaluated twice.

#### The Search-Layer Invariant & Early Exit Condition
At each iteration:
1. Extract the closest candidate $c \leftarrow \text{pop\_min}(C)$.
2. Compare $c$ against the furthest element in the current result pool $f \leftarrow \text{peek\_max}(W)$:
   $$\text{If } d(c, q) > d(f, q), \quad \textbf{BREAK SEARCH.}$$
   **Termination Rationale**: Because $C$ is a min-heap, all remaining candidates in $C$ are at least as far as $c$. Since $c$ is already further than the worst element in $W$, no candidate in $C$ can possibly improve $W$.
3. For each neighbor $e \in \text{neighbors}_0(c)$:
   - If $e \notin V$:
     - Mark $e \in V$.
     - Let $f \leftarrow \text{peek\_max}(W)$.
     - If $d(e, q) < d(f, q)$ or $|W| < ef$:
       - Push $e$ into $C$ (min-heap).
       - Push $e$ into $W$ (max-heap).
       - If $|W| > ef$, evict the furthest element: $\text{pop\_max}(W)$.

Finally, the top $K$ nearest neighbors are extracted from $W$ (where $K \le efSearch$).

---

## 4. Small-World Mechanics & Logarithmic Scaling

### 4.1 Navigable Small World (NSW) Properties

The foundation of HNSW is rooted in the study of complex networks by Watts & Strogatz (1998) and Kleinberg (2000). A network exhibits **Small-World properties** if:
1. **High Clustering Coefficient $C(G)$**: The neighbors of any node are likely to be neighbors of each other (strong local triangulations).
2. **Short Average Path Length $L(G)$**: The minimum number of hops between any two arbitrary nodes scales logarithmically with graph size:
   $$L(G) \propto \log N$$

```
             NSW GRAPH TOPOLOGY (WATTS-STROGATZ MODEL)
             
            (o)----(o)                  (o)----(o)
           /   \  /   \                /   \  /   \
         (o)---(o)----(o)============(o)---(o)----(o)
               Cluster A      ^             Cluster B
                              |
                    Long-Range Shortcut
```

In a standard single-layer Navigable Small World (NSW) graph (Malkov et al., 2014):
- Nodes are connected both to their immediate spatial neighbors (providing high clustering) and to a few distant nodes (long-range shortcuts).
- Search starts at an arbitrary node and walks greedily toward the query.
- **Limitation of Single-Layer NSW**: As $N$ grows into millions, routing complexity in flat NSW degrades from $\mathcal{O}(\log N)$ to $\mathcal{O}(\log^2 N)$ or poly-logarithmic. Furthermore, if the starting point is far from the query, the greedy walk frequently gets trapped in **false local minima** (clusters whose local neighbors all point away from the target, even though a distant cluster is closer).

---

### 4.2 Why Greedy Walk Succeeds & Overcomes Local Minima

HNSW solves the local minimum trap and restores pure $\mathcal{O}(\log N)$ search complexity through **spatial scale separation**:

```
Scale Separation in HNSW:
Layer 2 (Long-range links):     Hops span 100-1000 distance units
Layer 1 (Medium-range links):   Hops span 10-100 distance units
Layer 0 (Short-range links):    Hops span 0.1-10 distance units
```

#### The Scale Separation Mechanism
1. **Upper Layer Coarse Routing**: Upper layers contain only widely dispersed nodes. The Voronoi cells of these upper nodes cover massive volumes of the metric space. A 1-greedy walk on layer $l$ cannot get trapped in small local metric fluctuations because small clusters simply do not exist at that layer.
2. **Exponential Distance Contraction**: Each hop at layer $l$ covers a distance proportional to $M^l$. Thus, the remaining metric distance to the query decays exponentially:
   $$d(curr_{t+1}, q) \le \alpha \cdot d(curr_t, q), \quad \alpha < 1$$
3. **Smooth Basin Landing**: When the top-down walk lands at layer 0, the entry point is already guaranteed to be in the immediate spatial vicinity (the same high-density basin) as the query. At this stage, beam search with $efSearch > 1$ easily traverses local triangulations without getting stuck.

---

### 4.3 Logarithmic Complexity Derivation $\mathcal{O}(\log N)$

We can formally demonstrate why HNSW search executes in $\mathcal{O}(\log N)$ time:

1. **Total Number of Layers**:
   Because nodes survive to the next layer with constant probability $p = 1/M$, the maximum layer height $L$ follows:
   $$L = \log_{1/p}(N) = \log_M(N) = \mathcal{O}(\log N)$$

2. **Hops per Layer**:
   At each upper layer $l$, the search begins at the node that was the closest in layer $l+1$. By scale invariance of the Poisson/exponential point process, the expected number of greedy steps required to find the local minimum among neighbors is a constant independent of $N$:
   $$\mathbb{E}[\text{hops}_l] \le c_1, \quad \forall l \ge 1$$

3. **Total Upper Search Complexity**:
   $$\text{Steps}_{upper} = \sum_{l=1}^{L} \mathbb{E}[\text{hops}_l] \cdot M \le L \cdot c_1 \cdot M = \mathcal{O}(M \log_M N) = \mathcal{O}(\log N)$$

4. **Base Layer Complexity**:
   At layer 0, the beam search explores an area bounded by parameter $efSearch$. Since the entry point is already within the nearest neighborhood, the number of evaluated nodes is bounded by a function of $efSearch$ and $M_{max0}$, independent of $N$:
   $$\text{Steps}_0 = \mathcal{O}(efSearch \cdot M_{max0})$$

5. **Overall Search Complexity**:
   $$\boxed{T_{search} = \mathcal{O}(\log N) + \mathcal{O}(efSearch \cdot M_{max0}) = \mathcal{O}(\log N)}$$

This provides an exponential speedup over brute-force search ($\mathcal{O}(N \cdot D)$) and improves upon flat NSW ($\mathcal{O}(\log^2 N)$).

---

### 4.4 Neighbor Selection Heuristics: Diversity vs Proximity

During graph insertion, when a node $q$ connects to candidate neighbors in layer $l$, how should the $M$ links be chosen?

```
                     NEIGHBOR SELECTION HEURISTIC
                     
              (q) -------------------> (e) Candidate
                \                      /
                 \                    /
                  \                  /  d(e, n) < d(e, q) ?
                   v                v   -> If yes: redundant!
                     (n) Selected
```

#### Simple Selection (`SELECT-NEIGHBORS-SIMPLE`)
Select the $M$ candidates that have the absolute smallest distance to $q$.
- **Problem**: In high dimensions, points often cluster along manifolds. Simple selection connects $q$ to $M$ points that are all clustered tightly together in the exact same direction. The graph loses directional diversity, isolating other angular sectors and creating "cliques" that degrade search recall.

#### Heuristic Selection with Shrinking (`SELECT-NEIGHBORS-HEURISTIC`)
Algorithm 4 of Malkov & Yashunin enforces **directional diversity** using a triangular distance test:
- Maintain selected list $R \leftarrow \emptyset$.
- For each candidate $e$ in ascending order of distance to $q$:
  - Add $e$ to $R$ **if and only if** $e$ is closer to $q$ than to any node already selected in $R$:
    $$\forall n \in R, \quad d(e, q) < d(e, n)$$
  - If $d(e, n) \le d(e, q)$ for some $n \in R$, candidate $e$ is discarded because $n$ already acts as a shortcut in that general direction.
- **Optional `keepPrunedConnections`**: If $|R| < M$ after testing all candidates, backfill $R$ with the best discarded candidates to guarantee degree connectivity.

This heuristic creates a Delaunay-like proximity graph spanning all spatial quadrants, dramatically reducing routing hops.

---

## 5. Formal Algorithmic Specifications

The following pseudocode defines the authoritative reference implementation per Malkov & Yashunin (2018).

### 5.1 Algorithm 1: INSERT

```text
Algorithm 1: INSERT(hnsw, q, M, M_max, efConstruction, m_L)
Input: 
  hnsw: HNSW index (entry point ep, top layer l_top, adjacency lists)
  q: new vector to insert
  M: target number of bidirectional connections to establish
  M_max: maximum allowed degree for layers > 0
  efConstruction: size of dynamic candidate list during construction
  m_L: layer scale normalization factor (1 / ln(M))

1:  W = empty set of nearest candidates
2:  ep = hnsw.enter_point
3:  l_top = hnsw.top_layer
4:  l = floor( -ln(uniform(0, 1)) * m_L )   // Sample layer for q

5:  // Phase A: Top-down zoom-in from l_top down to l + 1
6:  curr_ep = ep
7:  if curr_ep != null then
8:      for lc = l_top down to l + 1 do
9:          curr_ep = SEARCH-LAYER(q, curr_ep, ef=1, lc).top_closest()
10:     end for
11: end if

12: // Phase B: Connect q at layers min(l_top, l) down to 0
13: for lc = min(l_top, l) down to 0 do
14:     W = SEARCH-LAYER(q, curr_ep, ef=efConstruction, lc)
15:     M_layer = (lc == 0) ? M_max0 : M_max
16:     neighbors = SELECT-NEIGHBORS-HEURISTIC(q, W, M, lc)
17:     
18:     // Add bidirectional edges
19:     for each e in neighbors do
20:         ADD-BIDIRECTIONAL-EDGE(hnsw, lc, q, e)
21:         // Shrink connections of e if degree exceeds M_layer
22:         if degree(hnsw, lc, e) > M_layer then
23:             e_adj = get_neighbors(hnsw, lc, e)
24:             e_new_adj = SELECT-NEIGHBORS-HEURISTIC(e, e_adj, M_layer, lc)
25:             set_neighbors(hnsw, lc, e, e_new_adj)
26:         end if
27:     end for
28:     curr_ep = W.top_closest()
29: end for

30: // Phase C: Update global entry point if new node exceeded l_top
31: if l > l_top or hnsw.enter_point == null then
32:     hnsw.top_layer = l
33:     hnsw.enter_point = q
34: end if
```

---

### 5.2 Algorithm 2: SEARCH-LAYER

```text
Algorithm 2: SEARCH-LAYER(q, ep, ef, lc)
Input:
  q: query vector
  ep: enter point node
  ef: number of nearest elements to track
  lc: layer index
Output:
  W: dynamic result set containing up to ef closest elements to q found at layer lc

1:  v = { ep }                             // Set of visited elements
2:  C = Min-Heap of pairs (distance, node) // Candidates to inspect
3:  W = Max-Heap of pairs (distance, node) // Best found elements

4:  d_ep = dist(ep, q)
5:  C.push( (d_ep, ep) )
6:  W.push( (d_ep, ep) )

7:  while C is not empty do
8:      (d_c, c) = C.pop_min()              // Closest candidate
9:      (d_f, f) = W.peek_max()             // Furthest element in current pool

10:     if d_c > d_f then
11:         break                           // All remaining elements in C are worse than W
12:     end if

13:     for each neighbor e in neighbors(layer=lc, node=c) do
14:         if e not in v then
15:             v.add(e)
16:             (d_f, f) = W.peek_max()
17:             d_e = dist(e, q)
18:             if d_e < d_f or W.size() < ef then
19:                 C.push( (d_e, e) )
20:                 W.push( (d_e, e) )
21:                 if W.size() > ef then
22:                     W.pop_max()         // Evict furthest
23:                 end if
24:             end if
25:         end if
26:     end for
27: end while
28: return W
```

---

### 5.3 Algorithm 3: SELECT-NEIGHBORS-SIMPLE

```text
Algorithm 3: SELECT-NEIGHBORS-SIMPLE(q, C, M)
Input:
  q: base node
  C: candidate elements pool
  M: number of neighbors to select
Output:
  R: M closest candidates from C to q

1:  Sort candidates in C in ascending order of dist(c, q)
2:  return top M elements from sorted C
```

---

### 5.4 Algorithm 4: SELECT-NEIGHBORS-HEURISTIC

```text
Algorithm 4: SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates=false, keepPrunedConnections=false)
Input:
  q: base node
  C: candidate set
  M: maximum degree to select
  lc: layer number
  extendCandidates: boolean flag to add neighbors of candidates to C
  keepPrunedConnections: boolean flag to retain diverse pruned nodes
Output:
  R: selected diverse neighbors (size <= M)

1:  R = []
2:  W = copy of C
3:  if extendCandidates then
4:      for each e in C do
5:          for each e_adj in neighbors(layer=lc, node=e) do
6:              if e_adj not in W then W.add(e_adj)
7:          end for
8:      end for
9:  end if

10: W_d = Min-Heap of pairs (dist(e, q), e) for all e in W
11: W_pruned = empty queue

12: while W_d is not empty and |R| < M do
13:     (d_e, e) = W_d.pop_min()
14:     // Triangular diversity condition:
15:     // e is added if it is closer to q than to any existing member of R
16:     is_diverse = true
17:     for each r in R do
18:         if dist(e, r) <= d_e then
19:             is_diverse = false
20:             break
21:         end if
22:     end for
23:     if is_diverse then
24:         R.append(e)
25:     else
26:         W_pruned.append(e)
27:     end if
28: end while

29: // Keep pruned connections if R is under-filled
30: if keepPrunedConnections then
31:     while W_pruned is not empty and |R| < M do
32:         R.append(W_pruned.pop_front())
33:     end while
34: end if
35: return R
```

---

### 5.5 Algorithm 5: K-NN-SEARCH

```text
Algorithm 5: K-NN-SEARCH(hnsw, q, K, efSearch)
Input:
  hnsw: HNSW index
  q: query vector
  K: number of nearest neighbors to return (K <= efSearch)
  efSearch: beam width during query
Output:
  top_k: K nearest elements to q

1:  ep = hnsw.enter_point
2:  l_top = hnsw.top_layer
3:  if ep == null then return []

4:  curr_ep = ep
5:  // Phase 1: Top-down zoom-in (ef = 1)
6:  for lc = l_top down to 1 do
7:      curr_ep = SEARCH-LAYER(q, curr_ep, ef=1, lc).top_closest()
8:  end for

9:  // Phase 2: Base layer beam search
10: W = SEARCH-LAYER(q, curr_ep, ef=max(efSearch, K), lc=0)

11: // Extract top K elements sorted ascending by distance
12: top_k = extract_k_smallest(W, K)
13: return top_k
```

---

## 6. Summary of Phase 1 Ground Truth Role

Before constructing the HNSW proximity graphs (Phases 2–7), an unassailable baseline must be established:
1. **Brute-Force Exact Search**: Evaluates exact distances across all $N$ vectors for every query $q$.
2. **Canonical Tie-Breaking**: Guarantees bit-exact deterministic outputs via $(distance, id)$ total ordering.
3. **Automated Recall Harness**: Asserts $\text{Recall}@K(\text{Exact}, \text{Exact}) \equiv 1.0$, providing the benchmark against which HNSW graph configurations will be evaluated.

### Numerical Stability in Distance Calculations
When using the BLAS GEMM expansion $D^2 = \|q\|^2 + \|b\|^2 - 2(q \cdot b)$ for high-throughput brute-force search, IEEE 754 float32 non-associativity between C-level vector norm summation and SIMD FMA matrix multiplication can introduce rounding discrepancies of $\le 1$ to $3$ ULPs (relative error $< 10^{-6}$). Distances are clamped strictly to $D^2 \ge 0.0$. Relative positive threshold clamping is omitted to avoid collapsing distinct closely-spaced vectors and to preserve peak GEMM throughput.
