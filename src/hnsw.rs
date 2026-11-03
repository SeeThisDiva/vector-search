//! Hierarchical Navigable Small World (HNSW) Approximate Nearest Neighbor Search.
//!
//! Conforms to Malkov & Yashunin (2018) TPAMI and `docs/HNSW_ALGORITHM.md` (Algorithms 1–5).
//! Implemented in 100% safe Rust under `#![forbid(unsafe_code)]` using only `std`.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::distance::{inner_product, l2_squared, Metric};
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::{KnnSearchResult, TimedKnnSearchResult};
use crate::matrix::RowMatrix;
use crate::recall::{compute_latency_stats, LatencyStats};
use crate::rng::DeterministicRng;

/// Hyperparameters governing HNSW graph construction and query routing.
///
/// `m_max`, `m_max0` and `m_l` are optional overrides. When left as `None` they are derived
/// from `m` (see [`HnswConfig::effective_m_max`] and friends), so changing `m` alone keeps the
/// whole configuration consistent. An explicit `Some(..)` is always respected as given.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Target number of bidirectional edges to establish per node during insertion (default: 16).
    pub m: usize,
    /// Override for the maximum outgoing edges per node at upper layers l > 0 (default: `m`).
    pub m_max: Option<usize>,
    /// Override for the maximum outgoing edges per node at ground layer l = 0 (default: `2 * m`).
    pub m_max0: Option<usize>,
    /// Size of dynamic candidate beam during index construction (default: 100).
    pub ef_construction: usize,
    /// Override for the layer scale factor (default: `1.0 / ln(m)`).
    pub m_l: Option<f64>,
    /// Whether to extend candidate set with 1-hop neighbors during heuristic selection (default: false).
    pub extend_candidates: bool,
    /// Whether to retain diverse pruned candidates to guarantee degree M (default: true).
    pub keep_pruned: bool,
    /// Fixed PRNG seed for deterministic layer assignment and tie-breaking (default: 42).
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: None,
            m_max0: None,
            ef_construction: 100,
            m_l: None,
            extend_candidates: false,
            keep_pruned: true,
            seed: 42,
        }
    }
}

impl HnswConfig {
    /// Constructs a configuration with target degree `m` and beam width `ef_construction`.
    #[must_use]
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let m = m.max(2);
        Self {
            m,
            ef_construction: ef_construction.max(m),
            ..Self::default()
        }
    }

    /// Maximum outgoing edges per node at layers l > 0.
    #[must_use]
    pub fn effective_m_max(&self) -> usize {
        self.m_max.unwrap_or(self.m)
    }

    /// Maximum outgoing edges per node at layer 0.
    #[must_use]
    pub fn effective_m_max0(&self) -> usize {
        self.m_max0.unwrap_or(2 * self.m)
    }

    /// Layer scale factor used for level sampling.
    #[must_use]
    pub fn effective_m_l(&self) -> f64 {
        self.m_l.unwrap_or_else(|| 1.0 / (self.m as f64).ln())
    }

    /// Validates configuration invariants before building the index.
    pub fn validate(&self) -> Result<()> {
        if self.m < 2 {
            return Err(VectorSearchError::InvalidArgument(
                "m must be at least 2".to_string(),
            ));
        }
        if self.effective_m_max() < self.m {
            return Err(VectorSearchError::InvalidArgument(
                "m_max must be >= m".to_string(),
            ));
        }
        if self.effective_m_max0() < self.m {
            return Err(VectorSearchError::InvalidArgument(
                "m_max0 must be >= m".to_string(),
            ));
        }
        if self.ef_construction < self.m {
            return Err(VectorSearchError::InvalidArgument(
                "ef_construction must be >= m".to_string(),
            ));
        }
        let m_l = self.effective_m_l();
        if !m_l.is_finite() || m_l <= 0.0 {
            return Err(VectorSearchError::InvalidArgument(
                "m_l must be positive and finite".to_string(),
            ));
        }
        Ok(())
    }
}

/// Search candidate pairing a distance score with a 0-indexed node ID.
///
/// Implements canonical `(distance, id)` total ordering matching Phase 1 (`ExactKnn`):
/// smaller distances sort as `Less`; on identical distances, smaller IDs sort as `Less`.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub distance: f32,
    pub id: usize,
}

impl PartialEq for Candidate {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == Ordering::Equal && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Safe generation-based tag array providing O(1) visited checks and O(1) resets.
///
/// Eliminates per-query allocations and avoids O(N) memset zeroing passes.
#[derive(Debug, Clone)]
pub struct VisitedTracker {
    tags: Vec<u32>,
    curr_tag: u32,
}

impl VisitedTracker {
    /// Creates a tracker sized for up to `capacity` nodes.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            tags: vec![0; capacity],
            curr_tag: 1,
        }
    }

    /// Prepares the tracker for a new search session in O(1) time.
    #[inline]
    pub fn prepare(&mut self, node_count: usize) {
        if self.tags.len() < node_count {
            self.tags.resize(node_count, 0);
        }
        self.curr_tag = self.curr_tag.wrapping_add(1);
        if self.curr_tag == 0 {
            self.tags.fill(0);
            self.curr_tag = 1;
        }
    }

    /// Returns true if node `id` has already been visited in the current session.
    #[inline]
    #[must_use]
    pub fn is_visited(&self, id: usize) -> bool {
        id < self.tags.len() && self.tags[id] == self.curr_tag
    }

    /// Marks node `id` as visited in the current session in O(1) time.
    #[inline]
    pub fn mark_visited(&mut self, id: usize) {
        if id >= self.tags.len() {
            self.tags.resize(id + 1, 0);
        }
        self.tags[id] = self.curr_tag;
    }

    /// Heap memory footprint allocated by visited tracking tags in bytes.
    #[inline]
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.tags.capacity() * std::mem::size_of::<u32>()
    }
}

impl Default for VisitedTracker {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Reusable scratch buffers for [`HnswIndex::select_neighbors_heuristic`] and neighbor pruning.
///
/// Pre-allocated during index initialization to completely eliminate dynamic heap allocations
/// in the triangular diversity heuristic during vector insertion.
#[derive(Debug, Clone, Default)]
pub struct HeuristicScratch {
    /// Candidate buffer holding input candidates and optional 1-hop expansions.
    pub w_set: Vec<Candidate>,
    /// Min-heap sorting candidates in ascending canonical order `(distance, id)`.
    pub w_d: BinaryHeap<Reverse<Candidate>>,
    /// Selected diverse neighbor IDs for the target node.
    pub r: Vec<usize>,
    /// Rejected candidate buffer for step 5 keep-pruned backfilling.
    pub w_pruned: Vec<Candidate>,
    /// Scratch buffer for gathering candidates during degree pruning.
    pub e_cands: Vec<Candidate>,
    /// Scratch buffer storing outer neighbor IDs to prevent nested slice aliasing during degree pruning.
    pub outer_neighbors: Vec<usize>,
    /// Visited/seen set for candidate deduplication.
    pub seen: HashSet<usize>,
}

impl HeuristicScratch {
    /// Creates a new `HeuristicScratch` with preallocated capacities.
    #[must_use]
    pub fn new() -> Self {
        Self {
            w_set: Vec::with_capacity(256),
            w_d: BinaryHeap::with_capacity(256),
            r: Vec::with_capacity(64),
            w_pruned: Vec::with_capacity(256),
            e_cands: Vec::with_capacity(64),
            outer_neighbors: Vec::with_capacity(64),
            seen: HashSet::with_capacity(256),
        }
    }

    /// Estimated memory footprint allocated by scratch buffers in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.w_set.capacity() * std::mem::size_of::<Candidate>()
            + self.w_d.capacity() * std::mem::size_of::<Reverse<Candidate>>()
            + self.r.capacity() * std::mem::size_of::<usize>()
            + self.w_pruned.capacity() * std::mem::size_of::<Candidate>()
            + self.e_cands.capacity() * std::mem::size_of::<Candidate>()
            + self.outer_neighbors.capacity() * std::mem::size_of::<usize>()
            + self.seen.capacity() * (std::mem::size_of::<usize>() + 1)
    }
}

/// Detailed memory footprint breakdown for HNSW graph components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HnswGraphComponents {
    pub node_structs_bytes: usize,
    pub layer0_edges_bytes: usize,
    pub upper_layers_edges_bytes: usize,
    pub outer_neighbors_headers_bytes: usize,
    pub visited_tracker_tags_bytes: usize,
    pub heuristic_scratch_bytes: usize,
}

impl HnswGraphComponents {
    /// Total bytes allocated by all graph components.
    #[must_use]
    pub const fn total_graph_bytes(&self) -> usize {
        self.node_structs_bytes
            + self.layer0_edges_bytes
            + self.upper_layers_edges_bytes
            + self.outer_neighbors_headers_bytes
            + self.visited_tracker_tags_bytes
            + self.heuristic_scratch_bytes
    }
}

/// Internal node representing an indexed vector in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswNode {
    pub id: usize,
    pub max_layer: usize,
    pub neighbors: Vec<Vec<usize>>,
}

impl HnswNode {
    /// Creates an empty node with neighbor lists allocated up to `max_layer`.
    #[must_use]
    pub fn new(id: usize, max_layer: usize, m_max0: usize, m_max: usize) -> Self {
        let mut neighbors = Vec::with_capacity(max_layer + 1);
        for lc in 0..=max_layer {
            let cap = if lc == 0 { m_max0 + 1 } else { m_max + 1 };
            neighbors.push(Vec::with_capacity(cap));
        }
        Self {
            id,
            max_layer,
            neighbors,
        }
    }
}

/// Hierarchical Navigable Small World (HNSW) Approximate Nearest Neighbor index.
pub struct HnswIndex<'a> {
    matrix: &'a RowMatrix<f32>,
    metric: Metric,
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    enter_point: Option<usize>,
    top_layer: usize,
    rng: DeterministicRng,
    visited_tracker: VisitedTracker,
    heuristic_scratch: HeuristicScratch,
}

impl<'a> HnswIndex<'a> {
    /// Initializes an empty HNSW index over `matrix`.
    pub fn new(matrix: &'a RowMatrix<f32>, metric: Metric, config: HnswConfig) -> Self {
        let rng = DeterministicRng::new(config.seed);
        let visited_tracker = VisitedTracker::new(matrix.rows());
        let heuristic_scratch = HeuristicScratch::new();

        Self {
            matrix,
            metric,
            config,
            nodes: Vec::with_capacity(matrix.rows()),
            enter_point: None,
            top_layer: 0,
            rng,
            visited_tracker,
            heuristic_scratch,
        }
    }

    /// Builds an HNSW index from scratch by inserting all rows in `matrix`.
    pub fn build(matrix: &'a RowMatrix<f32>, metric: Metric, config: HnswConfig) -> Result<Self> {
        config.validate()?;
        let mut index = Self::new(matrix, metric, config);
        for id in 0..matrix.rows() {
            index.insert(id)?;
        }
        Ok(index)
    }

    /// Number of indexed vectors in the graph.
    #[inline]
    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.nodes.len()
    }

    /// Current maximum layer height in the graph.
    #[inline]
    #[must_use]
    pub fn max_layer(&self) -> usize {
        self.top_layer
    }

    /// Global entry point node ID, or `None` if index is empty.
    #[inline]
    #[must_use]
    pub fn entry_point(&self) -> Option<usize> {
        self.enter_point
    }

    /// Graph memory footprint allocated by index structures in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let mut total = self.nodes.capacity() * std::mem::size_of::<HnswNode>();
        for node in &self.nodes {
            total += node.neighbors.capacity() * std::mem::size_of::<Vec<usize>>();
            for layer_edges in &node.neighbors {
                total += layer_edges.capacity() * std::mem::size_of::<usize>();
            }
        }
        total += self.visited_tracker.memory_bytes();
        total
    }

    /// Graph memory footprint in mebibytes (MiB).
    #[inline]
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Average graph memory per indexed vector in bytes.
    #[inline]
    #[must_use]
    pub fn bytes_per_vector(&self) -> f64 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.memory_bytes() as f64 / self.nodes.len() as f64
        }
    }

    /// Detailed breakdown of memory allocated across individual graph components.
    #[must_use]
    pub fn graph_components(&self) -> HnswGraphComponents {
        let node_structs_bytes = self.nodes.capacity() * std::mem::size_of::<HnswNode>();
        let mut layer0_edges_bytes = 0;
        let mut upper_layers_edges_bytes = 0;
        let mut outer_neighbors_headers_bytes = 0;

        for node in &self.nodes {
            outer_neighbors_headers_bytes +=
                node.neighbors.capacity() * std::mem::size_of::<Vec<usize>>();
            for (lc, layer_edges) in node.neighbors.iter().enumerate() {
                let bytes = layer_edges.capacity() * std::mem::size_of::<usize>();
                if lc == 0 {
                    layer0_edges_bytes += bytes;
                } else {
                    upper_layers_edges_bytes += bytes;
                }
            }
        }

        let visited_tracker_tags_bytes = self.visited_tracker.memory_bytes();
        let heuristic_scratch_bytes = self.heuristic_scratch.memory_bytes();

        HnswGraphComponents {
            node_structs_bytes,
            layer0_edges_bytes,
            upper_layers_edges_bytes,
            outer_neighbors_headers_bytes,
            visited_tracker_tags_bytes,
            heuristic_scratch_bytes,
        }
    }

    /// Reference to graph configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Configured distance metric.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of nodes present at layer `layer`.
    #[must_use]
    pub fn layer_node_count(&self, layer: usize) -> usize {
        self.nodes.iter().filter(|n| n.max_layer >= layer).count()
    }

    /// Node IDs present at layer `layer`.
    #[must_use]
    pub fn nodes_at_layer(&self, layer: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.max_layer >= layer)
            .map(|n| n.id)
            .collect()
    }

    /// Returns true if node `node` exists at layer `layer`.
    #[inline]
    #[must_use]
    pub fn node_exists_at_layer(&self, layer: usize, node: usize) -> bool {
        node < self.nodes.len() && self.nodes[node].max_layer >= layer
    }

    /// Maximum layer height of node `node`.
    #[inline]
    #[must_use]
    pub fn node_layer(&self, node: usize) -> usize {
        if node < self.nodes.len() {
            self.nodes[node].max_layer
        } else {
            0
        }
    }

    /// Outgoing neighbor node IDs for `node` at layer `layer`.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, layer: usize, node: usize) -> &[usize] {
        if node < self.nodes.len() && layer <= self.nodes[node].max_layer {
            &self.nodes[node].neighbors[layer]
        } else {
            &[]
        }
    }

    /// Draws a random layer height using inverse transform sampling:
    ///
    /// $$l = \lfloor -\ln(U) \cdot m_L \rfloor$$
    #[inline]
    pub fn sample_layer(&mut self) -> usize {
        let u = self.rng.next_f64_open01();
        let l = (-u.ln() * self.config.effective_m_l()).floor();
        if l <= 0.0 || l.is_nan() {
            0
        } else {
            l as usize
        }
    }

    #[inline]
    fn compute_distance(&self, query: &[f32], base_id: usize) -> f32 {
        let base_row = self.matrix.row(base_id);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => l2_squared(query, base_row),
            Metric::InnerProduct => {
                let neg_dot = -inner_product(query, base_row);
                if neg_dot == 0.0 {
                    0.0
                } else {
                    neg_dot
                }
            }
        }
    }

    #[inline]
    fn compute_node_distance(&self, id_a: usize, id_b: usize) -> f32 {
        if id_a == id_b {
            return 0.0;
        }
        let row_a = self.matrix.row(id_a);
        let row_b = self.matrix.row(id_b);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => l2_squared(row_a, row_b),
            Metric::InnerProduct => {
                let neg_dot = -inner_product(row_a, row_b);
                if neg_dot == 0.0 {
                    0.0
                } else {
                    neg_dot
                }
            }
        }
    }

    #[inline]
    fn format_output_distance(&self, sort_key: f32) -> f32 {
        match self.metric {
            Metric::L2Squared => {
                if sort_key <= 0.0 {
                    0.0
                } else {
                    sort_key
                }
            }
            Metric::Euclidean => {
                let d = sort_key.max(0.0).sqrt();
                if d <= 0.0 {
                    0.0
                } else {
                    d
                }
            }
            Metric::InnerProduct => {
                let score = -sort_key;
                if score == 0.0 {
                    0.0
                } else {
                    score
                }
            }
        }
    }

    /// Explores the graph at layer `lc` starting from `ep` tracking up to `ef` candidates (Algorithm 2: SEARCH-LAYER).
    pub fn search_layer(
        &self,
        query: &[f32],
        ep: usize,
        ef: usize,
        lc: usize,
        tracker: &mut VisitedTracker,
    ) -> BinaryHeap<Candidate> {
        tracker.prepare(self.nodes.len());
        tracker.mark_visited(ep);

        let d_ep = self.compute_distance(query, ep);
        let cand_ep = Candidate {
            distance: d_ep,
            id: ep,
        };

        let cap = ef + self.config.effective_m_max0();
        let mut c = BinaryHeap::<Reverse<Candidate>>::with_capacity(cap);
        let mut w = BinaryHeap::<Candidate>::with_capacity(cap);

        c.push(Reverse(cand_ep));
        w.push(cand_ep);

        while let Some(Reverse(curr)) = c.pop() {
            let furthest = *w.peek().expect("W never empty during search");

            if curr.distance > furthest.distance {
                break;
            }

            if let Some(node) = self.nodes.get(curr.id) {
                if lc <= node.max_layer {
                    for &neighbor_id in &node.neighbors[lc] {
                        if !tracker.is_visited(neighbor_id) {
                            tracker.mark_visited(neighbor_id);

                            let furthest_now = *w.peek().expect("W never empty");
                            let d_neighbor = self.compute_distance(query, neighbor_id);
                            let cand_neighbor = Candidate {
                                distance: d_neighbor,
                                id: neighbor_id,
                            };

                            if cand_neighbor < furthest_now || w.len() < ef {
                                c.push(Reverse(cand_neighbor));
                                w.push(cand_neighbor);

                                if w.len() > ef {
                                    w.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        w
    }

    /// Internal implementation of Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) reusing preallocated buffers in `scratch`.
    #[allow(clippy::too_many_arguments)]
    pub fn select_neighbors_heuristic_with_scratch(
        &self,
        base_id: usize,
        candidates: &[Candidate],
        m: usize,
        lc: usize,
        extend_candidates: bool,
        keep_pruned: bool,
        scratch: &mut HeuristicScratch,
    ) {
        scratch.w_set.clear();
        scratch.seen.clear();

        // Step 1: Add input candidates (excluding base node itself)
        for &cand in candidates {
            if cand.id != base_id && scratch.seen.insert(cand.id) {
                scratch.w_set.push(cand);
            }
        }

        // Step 2: Optionally extend candidates with 1-hop neighbors
        if extend_candidates {
            for &cand in candidates {
                if let Some(node) = self.nodes.get(cand.id) {
                    if lc <= node.max_layer {
                        for &adj_id in &node.neighbors[lc] {
                            if adj_id != base_id && scratch.seen.insert(adj_id) {
                                let d_adj = self.compute_node_distance(adj_id, base_id);
                                scratch.w_set.push(Candidate {
                                    distance: d_adj,
                                    id: adj_id,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Populate Min-Heap W_d (ordered ascending by distance)
        scratch.w_d.clear();
        for &cand in &scratch.w_set {
            scratch.w_d.push(Reverse(cand));
        }

        scratch.r.clear();
        scratch.w_pruned.clear();

        // Step 4: Triangular diversity selection
        while let Some(Reverse(e)) = scratch.w_d.pop() {
            if scratch.r.len() >= m {
                break;
            }

            let mut is_diverse = true;
            for &r_id in &scratch.r {
                let d_e_r = self.compute_node_distance(e.id, r_id);
                if d_e_r <= e.distance {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                scratch.r.push(e.id);
            } else {
                scratch.w_pruned.push(e);
            }
        }

        // Step 5: Backfill from pruned candidates if keep_pruned is set and R is under-filled
        if keep_pruned && scratch.r.len() < m {
            for &pruned_cand in &scratch.w_pruned {
                if scratch.r.len() >= m {
                    break;
                }
                scratch.r.push(pruned_cand.id);
            }
        }
    }

    /// Selects up to `m` diverse neighbors from candidate set `candidates` using triangular pruning (Algorithm 4: SELECT-NEIGHBORS-HEURISTIC).
    pub fn select_neighbors_heuristic(
        &self,
        base_id: usize,
        candidates: &[Candidate],
        m: usize,
        lc: usize,
        extend_candidates: bool,
        keep_pruned: bool,
    ) -> Vec<usize> {
        let mut scratch = HeuristicScratch::new();
        self.select_neighbors_heuristic_with_scratch(
            base_id,
            candidates,
            m,
            lc,
            extend_candidates,
            keep_pruned,
            &mut scratch,
        );
        scratch.r
    }

    /// Inserts a new vector by its row `id` into the HNSW graph (Algorithm 1: INSERT).
    ///
    /// Ids must be inserted densely in order (`0, 1, 2, ...`): node `i` is stored at slot `i`,
    /// so `q_id` must equal the current [`HnswIndex::num_vectors`]. Out-of-order ids would create
    /// unreachable placeholder nodes and re-inserting an id would leave dangling edges, so both
    /// are rejected with `InvalidArgument` instead of silently corrupting the graph.
    pub fn insert(&mut self, q_id: usize) -> Result<()> {
        if q_id >= self.matrix.rows() {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Vector id {q_id} out of bounds for matrix with {} rows",
                self.matrix.rows()
            )));
        }
        if q_id != self.nodes.len() {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Vector id {q_id} must be inserted in order; next expected id is {}",
                self.nodes.len()
            )));
        }

        let q_vec = self.matrix.row(q_id);
        let l = self.sample_layer();

        let new_node = HnswNode::new(
            q_id,
            l,
            self.config.effective_m_max0(),
            self.config.effective_m_max(),
        );
        self.nodes.push(new_node);

        // Case 1: First node insertion (empty graph)
        let Some(mut curr_ep) = self.enter_point else {
            self.enter_point = Some(q_id);
            self.top_layer = l;
            return Ok(());
        };

        let l_top = self.top_layer;
        let mut tracker = std::mem::take(&mut self.visited_tracker);
        let mut scratch = std::mem::take(&mut self.heuristic_scratch);
        let mut e_cands = std::mem::take(&mut scratch.e_cands);

        // Phase A: Top-down greedy zoom-in (ef = 1) from l_top down to l + 1
        if l < l_top {
            for lc in (l + 1..=l_top).rev() {
                let w = self.search_layer(q_vec, curr_ep, 1, lc, &mut tracker);
                if let Some(best) = w.iter().min_by(|a, b| a.cmp(b)) {
                    curr_ep = best.id;
                }
            }
        }

        // Phase B: Connect q at layers min(l_top, l) down to 0
        let start_layer = l_top.min(l);
        for lc in (0..=start_layer).rev() {
            let m_layer = if lc == 0 {
                self.config.effective_m_max0()
            } else {
                self.config.effective_m_max()
            };

            let w = self.search_layer(
                q_vec,
                curr_ep,
                self.config.ef_construction,
                lc,
                &mut tracker,
            );

            let candidates: Vec<Candidate> = w.into_sorted_vec();
            self.select_neighbors_heuristic_with_scratch(
                q_id,
                &candidates,
                self.config.m,
                lc,
                self.config.extend_candidates,
                self.config.keep_pruned,
                &mut scratch,
            );

            scratch.outer_neighbors.clear();
            scratch.outer_neighbors.extend_from_slice(&scratch.r);

            // Add bidirectional edges
            for i in 0..scratch.outer_neighbors.len() {
                let e = scratch.outer_neighbors[i];
                if !self.nodes[q_id].neighbors[lc].contains(&e) {
                    self.nodes[q_id].neighbors[lc].push(e);
                }
                if !self.nodes[e].neighbors[lc].contains(&q_id) {
                    self.nodes[e].neighbors[lc].push(q_id);
                }

                // Prune neighbor e if its degree exceeds layer cap
                if self.nodes[e].neighbors[lc].len() > m_layer {
                    e_cands.clear();

                    for &adj_id in &self.nodes[e].neighbors[lc] {
                        let d = self.compute_node_distance(adj_id, e);
                        e_cands.push(Candidate {
                            distance: d,
                            id: adj_id,
                        });
                    }

                    self.select_neighbors_heuristic_with_scratch(
                        e,
                        &e_cands,
                        m_layer,
                        lc,
                        self.config.extend_candidates,
                        self.config.keep_pruned,
                        &mut scratch,
                    );
                    self.nodes[e].neighbors[lc].clear();
                    self.nodes[e].neighbors[lc].extend_from_slice(&scratch.r);
                }
            }

            // Next layer's entry point is closest node found in W
            if let Some(best) = candidates.first() {
                curr_ep = best.id;
            }
        }

        scratch.e_cands = e_cands;
        self.visited_tracker = tracker;
        self.heuristic_scratch = scratch;

        // Phase C: Update global entry point if new node exceeded l_top
        if l > l_top {
            self.top_layer = l;
            self.enter_point = Some(q_id);
        }

        Ok(())
    }

    /// Queries the index for the top-`k` nearest neighbors to `query` (Algorithm 5: K-NN-SEARCH).
    ///
    /// Returns neighbor pairs `(distance, id)` sorted ascending by canonical total order.
    pub fn search_knn(
        &mut self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(f32, usize)>> {
        let mut tracker = std::mem::take(&mut self.visited_tracker);
        let result = self.search_knn_with_tracker(query, k, ef_search, &mut tracker);
        self.visited_tracker = tracker;
        result
    }

    /// Read-only search using an externally-provided `VisitedTracker`.
    /// Enables multi-threaded concurrent searches over `&HnswIndex`.
    pub fn search_knn_with_tracker(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        tracker: &mut VisitedTracker,
    ) -> Result<Vec<(f32, usize)>> {
        if k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "k must be positive (> 0)".to_string(),
            ));
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(
                "query vector contains non-finite value (NaN/Inf)".to_string(),
            ));
        }

        let Some(mut curr_ep) = self.enter_point else {
            return Ok(Vec::new());
        };

        if self.matrix.cols() > 0 && query.len() != self.matrix.cols() {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.matrix.cols(),
                actual: query.len(),
            });
        }

        // Phase 1: Top-down greedy zoom-in (ef = 1) down to layer 1
        if self.top_layer >= 1 {
            for lc in (1..=self.top_layer).rev() {
                let w = self.search_layer(query, curr_ep, 1, lc, tracker);
                if let Some(best) = w.iter().min_by(|a, b| a.cmp(b)) {
                    curr_ep = best.id;
                }
            }
        }

        // Phase 2: Base layer beam search at layer 0
        let effective_ef = ef_search.max(k);
        let w = self.search_layer(query, curr_ep, effective_ef, 0, tracker);

        // Phase 3: Extract top-K elements ordered ascending by (distance, id)
        let sorted = w.into_sorted_vec();
        let take_k = k.min(sorted.len());
        let mut results = Vec::with_capacity(take_k);

        for cand in &sorted[..take_k] {
            results.push((self.format_output_distance(cand.distance), cand.id));
        }

        Ok(results)
    }

    /// Warms up graph caches by executing an untimed search.
    pub fn warmup(&mut self, query: &[f32], k: usize, ef_search: usize) -> Result<()> {
        let _ = self.search_knn(query, k, ef_search)?;
        Ok(())
    }

    /// Batch sequential query execution recording latency statistics.
    pub fn search_batch_timed(
        &mut self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
    ) -> Result<TimedKnnSearchResult> {
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(TimedKnnSearchResult {
                result: KnnSearchResult {
                    indices: RowMatrix::empty(),
                    distances: RowMatrix::empty(),
                },
                per_query_latency_ms: Vec::new(),
                latency: LatencyStats::default(),
            });
        }

        let effective_k = k.min(self.nodes.len());
        let mut all_indices = Vec::with_capacity(num_queries * effective_k);
        let mut all_distances = Vec::with_capacity(num_queries * effective_k);
        let mut latencies_ms = Vec::with_capacity(num_queries);

        let batch_start = std::time::Instant::now();
        for q_row in queries.iter_rows() {
            let q_start = std::time::Instant::now();
            let hits = self.search_knn(q_row, k, ef_search)?;
            let q_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            latencies_ms.push(q_ms);

            for (dist, id) in hits {
                all_indices.push(id as i32);
                all_distances.push(dist);
            }
        }
        let total_elapsed_s = batch_start.elapsed().as_secs_f64();
        let stats = compute_latency_stats(&latencies_ms, total_elapsed_s);

        Ok(TimedKnnSearchResult {
            result: KnnSearchResult {
                indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
                distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
            },
            per_query_latency_ms: latencies_ms,
            latency: stats,
        })
    }

    /// Batch sequential query execution.
    pub fn search_batch(
        &mut self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
    ) -> Result<KnnSearchResult> {
        let timed = self.search_batch_timed(queries, k, ef_search)?;
        Ok(timed.result)
    }

    /// Scoped multi-threaded batch search with per-worker visited trackers.
    pub fn search_batch_parallel(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        num_threads: usize,
    ) -> Result<KnnSearchResult> {
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        if k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "k must be positive (> 0)".to_string(),
            ));
        }

        if self.nodes.is_empty() || self.enter_point.is_none() {
            return Ok(KnnSearchResult {
                indices: RowMatrix::new(num_queries, 0, vec![])?,
                distances: RowMatrix::new(num_queries, 0, vec![])?,
            });
        }

        if self.matrix.cols() > 0 && queries.cols() != self.matrix.cols() {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.matrix.cols(),
                actual: queries.cols(),
            });
        }

        // Validate every query up front so a bad vector yields `Err` here, exactly like the
        // sequential path, instead of panicking inside a worker thread.
        if queries.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(
                "query vector contains non-finite value (NaN/Inf)".to_string(),
            ));
        }

        let effective_k = k.min(self.nodes.len());
        let workers = if num_threads == 0 {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        } else {
            num_threads
        }
        .min(num_queries)
        .max(1);

        if workers == 1 {
            let mut tracker = VisitedTracker::new(self.nodes.len());
            let mut all_indices = Vec::with_capacity(num_queries * effective_k);
            let mut all_distances = Vec::with_capacity(num_queries * effective_k);
            for q_row in queries.iter_rows() {
                let hits = self.search_knn_with_tracker(q_row, k, ef_search, &mut tracker)?;
                for (dist, id) in hits {
                    all_indices.push(id as i32);
                    all_distances.push(dist);
                }
            }
            return Ok(KnnSearchResult {
                indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
                distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
            });
        }

        let mut all_indices = vec![0_i32; num_queries * effective_k];
        let mut all_distances = vec![0.0_f32; num_queries * effective_k];
        let chunk_size = num_queries.div_ceil(workers);

        std::thread::scope(|s| {
            let mut idx_rest = all_indices.as_mut_slice();
            let mut dist_rest = all_distances.as_mut_slice();
            let mut q_offset = 0;

            while q_offset < num_queries {
                let take_q = chunk_size.min(num_queries - q_offset);
                let take_elems = take_q * effective_k;
                let (idx_slice, next_idx) = idx_rest.split_at_mut(take_elems);
                let (dist_slice, next_dist) = dist_rest.split_at_mut(take_elems);
                idx_rest = next_idx;
                dist_rest = next_dist;

                let start_q = q_offset;
                q_offset += take_q;

                s.spawn(move || {
                    let mut tracker = VisitedTracker::new(self.nodes.len());
                    for local_i in 0..take_q {
                        let q_row = queries.row(start_q + local_i);
                        let hits = self
                            .search_knn_with_tracker(q_row, k, ef_search, &mut tracker)
                            .expect("query valid");
                        let out_base = local_i * effective_k;
                        for (j, (dist, id)) in hits.into_iter().enumerate() {
                            idx_slice[out_base + j] = id as i32;
                            dist_slice[out_base + j] = dist;
                        }
                    }
                });
            }
        });

        Ok(KnnSearchResult {
            indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
            distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
        })
    }
}
