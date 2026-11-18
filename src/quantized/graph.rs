//! Lifetime-free compact HNSW graph (`T1.7` flat `u32` arenas) and static-dispatch
//! quantization traits (`T4.1`).
//!
//! Provides two graph acquisition paths:
//! 1. [`CompactHnswGraph::from_hnsw_index`] (`Float32GraphThenQuantize`) — clones topology via
//!    public accessors on [`crate::hnsw::HnswIndex`], guaranteeing bit-identical `graph_fingerprint()`.
//! 2. [`CompactHnswGraph::build_with_source`] (`QuantizedDirect`) — builds directly from any
//!    [`QuantizedDistanceSource`] using Malkov & Yashunin Algorithms 1, 2, and 4.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Read, Write};
use std::time::Instant;

use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::KnnSearchResult;
use crate::hnsw::{
    fnv1a_64_update_u64, Candidate, HeuristicScratch, HnswConfig, HnswGraphComponents, HnswIndex,
    HnswNodeMeta, VisitedTracker, FNV_OFFSET_BASIS_64,
};
use crate::matrix::RowMatrix;
use crate::recall::compute_latency_stats;
use crate::rng::DeterministicRng;

use super::{
    compute_exact_sort_key, format_metric_output_distance, metric_from_u8, metric_to_u8, read_f64,
    read_magic, read_u32, read_u32_vec, read_u64, read_u8, read_usize_u64, write_f64, write_u32,
    write_u32_slice, write_u64, write_u8,
};

const GRAPH_SNAPSHOT_MAGIC: &[u8; 8] = b"CHNSWG01";

/// Precision mode used to construct the HNSW graph topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizedBuildMode {
    /// Build HNSW on exact `f32` vectors, copy topology into [`CompactHnswGraph`], then quantize.
    #[default]
    Float32GraphThenQuantize,
    /// Quantize first and construct HNSW directly on quantized distances.
    QuantizedDirect,
}

impl QuantizedBuildMode {
    /// Canonical lowercase identifier for reports and filenames.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float32GraphThenQuantize => "f32_graph_then_quantize",
            Self::QuantizedDirect => "quantized_direct",
        }
    }
}

/// Storage policy for retaining vs. dropping raw `f32` base vectors after indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizedStorageMode {
    /// Memory-first mode: retain only quantized codes, calibration/codebooks, and compact graph.
    #[default]
    QuantizedOnly,
    /// Accuracy-first mode: retain raw `f32` base vectors alongside the quantized index for exact top-R reranking.
    WithExactRerank,
}

impl QuantizedStorageMode {
    /// Canonical lowercase identifier for reports and filenames.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QuantizedOnly => "quantized_only",
            Self::WithExactRerank => "with_exact_rerank",
        }
    }
}

/// Phase-level wall-clock timings recorded during quantized index construction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QuantizedBuildTimings {
    pub calibration_or_train_s: f64,
    pub encode_s: f64,
    pub graph_build_s: f64,
    pub total_s: f64,
}

/// Steady-state payload memory breakdown for a quantized HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QuantizedMemoryProfile {
    pub num_vectors: usize,
    pub codes_bytes: usize,
    pub calibration_or_codebook_bytes: usize,
    pub aux_vector_bytes: usize,
    pub vector_payload_bytes: usize,
    pub node_meta_bytes: usize,
    pub layer0_edges_bytes: usize,
    pub upper_layers_edges_bytes: usize,
    pub graph_only_bytes: usize,
    pub raw_vectors_retained_bytes: usize,
    pub steady_state_total_bytes: usize,
    pub vector_bytes_per_vector: f64,
    pub graph_only_bytes_per_vector: f64,
    pub steady_state_bytes_per_vector: f64,
}

impl QuantizedMemoryProfile {
    /// Constructs a memory profile from exact byte counts of index components.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn from_parts(
        num_vectors: usize,
        codes_bytes: usize,
        calibration_or_codebook_bytes: usize,
        aux_vector_bytes: usize,
        node_meta_bytes: usize,
        layer0_edges_bytes: usize,
        upper_layers_edges_bytes: usize,
        raw_vectors_retained_bytes: usize,
    ) -> Self {
        let vector_payload_bytes = codes_bytes + calibration_or_codebook_bytes + aux_vector_bytes;
        let graph_only_bytes = node_meta_bytes + layer0_edges_bytes + upper_layers_edges_bytes;
        let steady_state_total_bytes =
            vector_payload_bytes + graph_only_bytes + raw_vectors_retained_bytes;

        let denom = num_vectors as f64;
        let (vector_bytes_per_vector, graph_only_bytes_per_vector, steady_state_bytes_per_vector) =
            if num_vectors == 0 {
                (0.0, 0.0, 0.0)
            } else {
                (
                    vector_payload_bytes as f64 / denom,
                    graph_only_bytes as f64 / denom,
                    steady_state_total_bytes as f64 / denom,
                )
            };

        Self {
            num_vectors,
            codes_bytes,
            calibration_or_codebook_bytes,
            aux_vector_bytes,
            vector_payload_bytes,
            node_meta_bytes,
            layer0_edges_bytes,
            upper_layers_edges_bytes,
            graph_only_bytes,
            raw_vectors_retained_bytes,
            steady_state_total_bytes,
            vector_bytes_per_vector,
            graph_only_bytes_per_vector,
            steady_state_bytes_per_vector,
        }
    }

    /// Steady-state total payload footprint in mebibytes (MiB).
    #[inline]
    #[must_use]
    pub fn steady_state_mb(&self) -> f64 {
        self.steady_state_total_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Quantized vector payload (codes + calibration/codebook + aux) in mebibytes (MiB).
    #[inline]
    #[must_use]
    pub fn vector_payload_mb(&self) -> f64 {
        self.vector_payload_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Graph-only payload in mebibytes (MiB).
    #[inline]
    #[must_use]
    pub fn graph_only_mb(&self) -> f64 {
        self.graph_only_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Single search neighbor result pairing a 0-indexed vector ID with its output distance/score.
///
/// Implements canonical `(distance, id)` total ordering matching [`Candidate`].
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub id: usize,
    pub distance: f32,
}

impl Neighbor {
    /// Creates a [`Neighbor`] from a 0-indexed vector ID and output distance/score.
    #[inline]
    #[must_use]
    pub const fn new(id: usize, distance: f32) -> Self {
        Self { id, distance }
    }

    /// Converts to `(distance, id)` tuple matching [`crate::hnsw::HnswIndex::search_knn`].
    #[inline]
    #[must_use]
    pub const fn as_tuple(self) -> (f32, usize) {
        (self.distance, self.id)
    }
}

impl PartialEq for Neighbor {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == std::cmp::Ordering::Equal && self.id == other.id
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Batch latency, throughput, and effective-parameter metrics for quantized search.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SearchMetrics {
    pub num_queries: usize,
    pub total_s: f64,
    pub qps: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub requested_ef_search: usize,
    pub effective_ef_search: usize,
    pub rerank_top_r: Option<usize>,
}

/// Timed batch search output containing per-query neighbor lists, latencies, and aggregate metrics.
#[derive(Debug, Clone)]
pub struct TimedQuantizedSearch {
    pub results: Vec<Vec<Neighbor>>,
    pub per_query_latency_ms: Vec<f64>,
    pub metrics: SearchMetrics,
}

/// Converts per-query `Vec<Vec<Neighbor>>` outputs into a [`KnnSearchResult`] matrix bundle
/// compatible with [`crate::recall::compute_recall_at_k`].
pub fn results_to_knn_search_result(results: &[Vec<Neighbor>]) -> Result<KnnSearchResult> {
    let num_queries = results.len();
    if num_queries == 0 {
        return Ok(KnnSearchResult {
            indices: RowMatrix::empty(),
            distances: RowMatrix::empty(),
        });
    }
    let width = results.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return Ok(KnnSearchResult {
            indices: RowMatrix::empty(),
            distances: RowMatrix::empty(),
        });
    }
    let mut indices = Vec::with_capacity(num_queries * width);
    let mut distances = Vec::with_capacity(num_queries * width);
    for row in results {
        for n in row {
            indices.push(n.id as i32);
            distances.push(n.distance);
        }
        for _ in row.len()..width {
            indices.push(-1);
            distances.push(f32::INFINITY);
        }
    }
    Ok(KnnSearchResult {
        indices: RowMatrix::new(num_queries, width, indices)?,
        distances: RowMatrix::new(num_queries, width, distances)?,
    })
}

/// Static-dispatch distance source for quantized HNSW graph construction and search.
pub trait QuantizedDistanceSource {
    /// Reusable per-query precomputed state (e.g., shifted query or PQ ADC lookup table).
    type QueryState: Default;

    /// Number of quantized base vectors indexed.
    fn len(&self) -> usize;

    /// Returns `true` if the distance source contains zero vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimensionality (`D`).
    fn dim(&self) -> usize;

    /// Configured distance metric.
    fn metric(&self) -> Metric;

    /// Prepares per-query state once before graph traversal (writes into preallocated `state`).
    fn prepare_query(&self, query: &[f32], state: &mut Self::QueryState);

    /// Computes approximate distance sort key between prepared query `state` and base node `node_id`.
    fn query_distance(&self, state: &Self::QueryState, node_id: usize) -> f32;

    /// Computes approximate node-to-node distance sort key (used during `QuantizedDirect` graph build).
    fn node_distance(&self, id_a: usize, id_b: usize) -> f32;

    /// Formats internal minimization sort key to public metric output (e.g., `sqrt` for Euclidean).
    fn format_output_distance(&self, sort_key: f32) -> f32;

    /// Total heap payload bytes of quantized codes and codebooks/calibration tables.
    fn vector_memory_bytes(&self) -> usize;
}

/// Reusable per-thread scratch space for zero-allocation quantized HNSW search.
#[derive(Debug, Clone, Default)]
pub struct QuantizedSearchScratch<Q> {
    pub tracker: VisitedTracker,
    pub frontier: BinaryHeap<Reverse<Candidate>>,
    pub results: BinaryHeap<Candidate>,
    pub raw_candidates: Vec<Candidate>,
    pub rerank_candidates: Vec<(f32, usize)>,
    pub query_state: Q,
}

impl<Q: Default> QuantizedSearchScratch<Q> {
    /// Creates a pre-sized scratch buffer bundle so hot-path queries avoid heap allocations.
    ///
    /// `frontier` is sized to `num_nodes.max(max_ef + m_max0 * 4)` to guarantee zero growth even
    /// on worst-case beam traversals, while `results`, `raw_candidates`, and `rerank_candidates`
    /// are sized to `max_ef + 1`.
    #[must_use]
    pub fn new(num_nodes: usize, max_ef: usize, m_max0: usize) -> Self {
        let frontier_cap = num_nodes.max(max_ef.saturating_add(m_max0.saturating_mul(4)));
        Self::with_capacities(num_nodes, frontier_cap, max_ef)
    }

    /// Creates scratch buffers with explicit capacities for `tracker`, `frontier`, and `max_ef` pools.
    #[must_use]
    pub fn with_capacities(num_nodes: usize, frontier_cap: usize, max_ef: usize) -> Self {
        let ef_cap = max_ef.saturating_add(1);
        Self {
            tracker: VisitedTracker::new(num_nodes),
            frontier: BinaryHeap::with_capacity(frontier_cap),
            results: BinaryHeap::with_capacity(ef_cap),
            raw_candidates: Vec::with_capacity(ef_cap),
            rerank_candidates: Vec::with_capacity(ef_cap),
            query_state: Q::default(),
        }
    }
}

/// Lifetime-free compact HNSW graph storing `T1.7` flat `u32` layer arenas.
#[derive(Debug, Clone)]
pub struct CompactHnswGraph {
    metric: Metric,
    config: HnswConfig,
    node_meta: Vec<HnswNodeMeta>,
    layer0: Vec<u32>,
    upper_arena: Vec<u32>,
    l0_stride: usize,
    upper_stride: usize,
    enter_point: Option<usize>,
    top_layer: usize,
}

impl CompactHnswGraph {
    /// Copies the exact topology from an existing `f32` [`HnswIndex`] using only its public
    /// read-only accessors (`Float32GraphThenQuantize`).
    ///
    /// Guarantees bit-for-bit identical [`Self::graph_fingerprint`] with `index.graph_fingerprint()`.
    #[must_use]
    pub fn from_hnsw_index(index: &HnswIndex<'_>) -> Self {
        let config = index.config().clone();
        let metric = index.metric();
        let n = index.num_vectors();
        let m_max0 = config.effective_m_max0();
        let m_max = config.effective_m_max();
        let l0_stride = m_max0 + 1;
        let upper_stride = m_max + 1;

        let total_upper_layers: usize = (0..n).map(|id| index.node_layer(id)).sum();
        let mut node_meta = Vec::with_capacity(n);
        let mut layer0 = vec![0_u32; n * l0_stride];
        let mut upper_arena = vec![0_u32; total_upper_layers * upper_stride];

        let mut curr_upper_off = 0_usize;
        for id in 0..n {
            let max_l = index.node_layer(id);
            let nbrs0 = index.neighbors(0, id);
            assert!(
                nbrs0.len() <= m_max0,
                "HnswIndex node {id} layer 0 degree {} exceeds m_max0 {m_max0}",
                nbrs0.len()
            );
            let base0 = id * l0_stride;
            layer0[base0] = nbrs0.len() as u32;
            layer0[base0 + 1..base0 + 1 + nbrs0.len()].copy_from_slice(nbrs0);

            let upper_offset = if max_l == 0 {
                u32::MAX
            } else {
                let off = curr_upper_off as u32;
                for lc in 1..=max_l {
                    let nbrs_l = index.neighbors(lc, id);
                    assert!(
                        nbrs_l.len() <= m_max,
                        "HnswIndex node {id} layer {lc} degree {} exceeds m_max {m_max}",
                        nbrs_l.len()
                    );
                    upper_arena[curr_upper_off] = nbrs_l.len() as u32;
                    upper_arena[curr_upper_off + 1..curr_upper_off + 1 + nbrs_l.len()]
                        .copy_from_slice(nbrs_l);
                    curr_upper_off += upper_stride;
                }
                off
            };

            node_meta.push(HnswNodeMeta {
                max_layer: max_l as u8,
                upper_offset,
            });
        }

        Self {
            metric,
            config,
            node_meta,
            layer0,
            upper_arena,
            l0_stride,
            upper_stride,
            enter_point: index.entry_point(),
            top_layer: if n == 0 { 0 } else { index.max_layer() },
        }
    }

    /// Builds an HNSW graph directly from any [`QuantizedDistanceSource`] (`QuantizedDirect`),
    /// executing Malkov & Yashunin Algorithms 1, 2, and 4 with canonical `(distance, id)` tie-breaking.
    pub fn build_with_source<S: QuantizedDistanceSource>(
        source: &S,
        config: HnswConfig,
    ) -> Result<Self> {
        config.validate()?;
        let n = source.len();
        if n > u32::MAX as usize {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Source vector count ({n}) exceeds u32::MAX ({})",
                u32::MAX
            )));
        }

        let l0_stride = config.effective_m_max0() + 1;
        let upper_stride = config.effective_m_max() + 1;

        let mut graph = Self {
            metric: source.metric(),
            config: config.clone(),
            node_meta: Vec::with_capacity(n),
            layer0: Vec::with_capacity(n.saturating_mul(l0_stride)),
            upper_arena: Vec::new(),
            l0_stride,
            upper_stride,
            enter_point: None,
            top_layer: 0,
        };

        if n == 0 {
            return Ok(graph);
        }

        let mut rng = DeterministicRng::new(config.seed);
        let mut tracker = VisitedTracker::new(n);
        let mut scratch = HeuristicScratch::new();
        let mut e_cands = std::mem::take(&mut scratch.e_cands);

        let heap_cap = config.ef_construction + config.effective_m_max0();
        let mut frontier = BinaryHeap::<Reverse<Candidate>>::with_capacity(heap_cap);
        let mut results = BinaryHeap::<Candidate>::with_capacity(heap_cap);
        let mut candidates = Vec::<Candidate>::with_capacity(config.ef_construction + 1);

        let m_l = config.effective_m_l();
        let m_max = config.effective_m_max();
        let m_max0 = config.effective_m_max0();
        let keep_pruned_shrink = config.effective_keep_pruned_shrink();

        for q_id in 0..n {
            let l = sample_layer(&mut rng, m_l).min(u8::MAX as usize);
            let upper_offset = if l == 0 {
                u32::MAX
            } else {
                let off = graph.upper_arena.len();
                graph
                    .upper_arena
                    .resize(off + l * graph.upper_stride, 0_u32);
                off as u32
            };

            graph.node_meta.push(HnswNodeMeta {
                max_layer: l as u8,
                upper_offset,
            });
            graph.layer0.resize((q_id + 1) * graph.l0_stride, 0_u32);

            let Some(mut curr_ep) = graph.enter_point else {
                graph.enter_point = Some(q_id);
                graph.top_layer = l;
                continue;
            };

            let l_top = graph.top_layer;

            // Phase A: Top-down greedy zoom-in (ef = 1) from l_top down to l + 1
            if l < l_top {
                for lc in (l + 1..=l_top).rev() {
                    graph.search_layer_core(
                        curr_ep,
                        1,
                        lc,
                        &mut tracker,
                        &mut frontier,
                        &mut results,
                        |nid| source.node_distance(q_id, nid),
                    );
                    if let Some(best) = results.iter().min_by(|a, b| a.cmp(b)) {
                        curr_ep = best.id;
                    }
                }
            }

            // Phase B: Connect q_id at layers min(l_top, l) down to 0
            let start_layer = l_top.min(l);
            for lc in (0..=start_layer).rev() {
                let m_layer = if lc == 0 { m_max0 } else { m_max };

                graph.search_layer_core(
                    curr_ep,
                    config.ef_construction,
                    lc,
                    &mut tracker,
                    &mut frontier,
                    &mut results,
                    |nid| source.node_distance(q_id, nid),
                );

                drain_heap_sorted_asc(&mut results, &mut candidates);

                graph.select_neighbors_heuristic_with_scratch(
                    source,
                    q_id,
                    &candidates,
                    config.m,
                    lc,
                    config.extend_candidates,
                    config.keep_pruned,
                    &mut scratch,
                );

                scratch.outer_neighbors.clear();
                scratch.outer_neighbors.extend_from_slice(&scratch.r);

                let q_id_u32 = q_id as u32;
                for i in 0..scratch.outer_neighbors.len() {
                    let e = scratch.outer_neighbors[i];
                    graph.push_neighbor_unique(lc, q_id, e as u32);
                    let prev_deg_e = graph.degree_at(lc, e);
                    if prev_deg_e < m_layer {
                        graph.push_neighbor_unique(lc, e, q_id_u32);
                    } else if !graph.neighbors(lc, e).contains(&q_id_u32) {
                        e_cands.clear();
                        for &adj_u32 in graph.neighbors(lc, e) {
                            let adj_id = adj_u32 as usize;
                            let d = source.node_distance(adj_id, e);
                            e_cands.push(Candidate {
                                distance: d,
                                id: adj_id,
                            });
                        }
                        let d_q = source.node_distance(q_id, e);
                        e_cands.push(Candidate {
                            distance: d_q,
                            id: q_id,
                        });

                        graph.select_neighbors_heuristic_with_scratch(
                            source,
                            e,
                            &e_cands,
                            m_layer,
                            lc,
                            config.extend_candidates,
                            keep_pruned_shrink,
                            &mut scratch,
                        );
                        graph.set_neighbors_from_slice(lc, e, &scratch.r);
                    }
                }

                if let Some(best) = candidates.first() {
                    curr_ep = best.id;
                }
            }

            // Phase C: Update global entry point if new node exceeded l_top
            if l > l_top {
                graph.top_layer = l;
                graph.enter_point = Some(q_id);
            }
        }

        graph.upper_arena.shrink_to_fit();
        Ok(graph)
    }

    /// 64-bit FNV-1a fingerprint matching [`HnswIndex::graph_fingerprint`] bit-for-bit.
    #[must_use]
    pub fn graph_fingerprint(&self) -> u64 {
        let mut hash = FNV_OFFSET_BASIS_64;
        fnv1a_64_update_u64(&mut hash, self.num_vectors() as u64);
        let ep_val = self.enter_point.map(|ep| ep as u64).unwrap_or(u64::MAX);
        fnv1a_64_update_u64(&mut hash, ep_val);
        for i in 0..self.num_vectors() {
            let max_l = self.node_layer(i);
            fnv1a_64_update_u64(&mut hash, max_l as u64);
            for lc in 0..=max_l {
                for &nid in self.neighbors(lc, i) {
                    fnv1a_64_update_u64(&mut hash, nid as u64);
                }
            }
        }
        hash
    }

    /// Number of indexed vectors in the graph.
    #[inline]
    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.node_meta.len()
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
        self.node_meta
            .iter()
            .filter(|m| (m.max_layer as usize) >= layer)
            .count()
    }

    /// Node IDs present at layer `layer`.
    #[must_use]
    pub fn nodes_at_layer(&self, layer: usize) -> Vec<usize> {
        self.node_meta
            .iter()
            .enumerate()
            .filter_map(|(id, m)| ((m.max_layer as usize) >= layer).then_some(id))
            .collect()
    }

    /// Returns `true` if `node` exists at `layer`.
    #[inline]
    #[must_use]
    pub fn node_exists_at_layer(&self, layer: usize, node: usize) -> bool {
        node < self.node_meta.len() && (self.node_meta[node].max_layer as usize) >= layer
    }

    /// Maximum layer height of `node`.
    #[inline]
    #[must_use]
    pub fn node_layer(&self, node: usize) -> usize {
        self.node_meta.get(node).map_or(0, |m| m.max_layer as usize)
    }

    /// Outgoing neighbor node IDs (`u32`) for `node` at `layer`.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, layer: usize, node: usize) -> &[u32] {
        let Some(meta) = self.node_meta.get(node) else {
            return &[];
        };
        if layer > meta.max_layer as usize {
            return &[];
        }
        if layer == 0 {
            let base = node * self.l0_stride;
            let deg = self.layer0[base] as usize;
            &self.layer0[base + 1..base + 1 + deg]
        } else {
            let base = (meta.upper_offset as usize) + (layer - 1) * self.upper_stride;
            let deg = self.upper_arena[base] as usize;
            &self.upper_arena[base + 1..base + 1 + deg]
        }
    }

    /// Bytes allocated by compact node metadata (`node_meta`).
    #[inline]
    #[must_use]
    pub fn node_meta_bytes(&self) -> usize {
        self.node_meta.capacity() * std::mem::size_of::<HnswNodeMeta>()
    }

    /// Bytes allocated by Layer-0 flat adjacency arena (`layer0`).
    #[inline]
    #[must_use]
    pub fn layer0_edges_bytes(&self) -> usize {
        self.layer0.capacity() * std::mem::size_of::<u32>()
    }

    /// Bytes allocated by upper-layer flat adjacency arena (`upper_arena`).
    #[inline]
    #[must_use]
    pub fn upper_layers_edges_bytes(&self) -> usize {
        self.upper_arena.capacity() * std::mem::size_of::<u32>()
    }

    /// Total graph-only topology memory footprint in bytes.
    #[inline]
    #[must_use]
    pub fn graph_only_memory_bytes(&self) -> usize {
        self.node_meta_bytes() + self.layer0_edges_bytes() + self.upper_layers_edges_bytes()
    }

    /// Graph-only memory footprint in mebibytes (MiB).
    #[inline]
    #[must_use]
    pub fn graph_only_memory_mb(&self) -> f64 {
        self.graph_only_memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Average graph-only memory per indexed vector in bytes.
    #[inline]
    #[must_use]
    pub fn graph_only_bytes_per_vector(&self) -> f64 {
        if self.node_meta.is_empty() {
            0.0
        } else {
            self.graph_only_memory_bytes() as f64 / self.node_meta.len() as f64
        }
    }

    /// Detailed breakdown matching [`HnswGraphComponents`].
    #[must_use]
    pub fn graph_components(&self) -> HnswGraphComponents {
        HnswGraphComponents {
            node_structs_bytes: self.node_meta_bytes(),
            layer0_edges_bytes: self.layer0_edges_bytes(),
            upper_layers_edges_bytes: self.upper_layers_edges_bytes(),
            outer_neighbors_headers_bytes: 0,
            visited_tracker_tags_bytes: 0,
            heuristic_scratch_bytes: 0,
        }
    }

    /// Executes pure quantized top-`k` search reusing caller-owned `scratch`.
    pub fn search_knn_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
    ) -> Result<Vec<Neighbor>> {
        let mut out = Vec::with_capacity(k.min(self.num_vectors()));
        self.search_knn_into_with_scratch(source, query, k, ef_search, scratch, &mut out)?;
        Ok(out)
    }

    /// Zero-allocation pure quantized top-`k` search writing results into `out`.
    pub fn search_knn_into_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
        out: &mut Vec<Neighbor>,
    ) -> Result<()> {
        self.search_knn_rerank_opt_into_with_scratch(
            source, None, query, k, ef_search, None, scratch, out,
        )
    }

    /// Executes quantized search with optional exact top-`R` reranking against `raw_base`.
    ///
    /// When `rerank_top_r = Some(r)`:
    /// - Rejects `r == 0` with [`VectorSearchError::InvalidArgument`].
    /// - Sets `effective_r = r.max(k)` and `effective_ef = ef_search.max(effective_r)`.
    /// - Runs quantized traversal, computes exact `Metric` distances on the top `effective_r`
    ///   quantized candidates via `raw_base.row(id)`, sorts by canonical `(distance, id)`, and
    ///   returns the top `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn search_knn_rerank_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: &RowMatrix<f32>,
        query: &[f32],
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
    ) -> Result<Vec<Neighbor>> {
        let mut out = Vec::with_capacity(k.min(self.num_vectors()));
        self.search_knn_rerank_opt_into_with_scratch(
            source,
            Some(raw_base),
            query,
            k,
            ef_search,
            rerank_top_r,
            scratch,
            &mut out,
        )?;
        Ok(out)
    }

    /// Shared quantized + optional rerank search implementation writing into preallocated `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn search_knn_rerank_opt_into_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        query: &[f32],
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
        out: &mut Vec<Neighbor>,
    ) -> Result<()> {
        self.validate_query_params(source, raw_base, query, k, rerank_top_r)?;
        self.search_knn_validated_into_with_scratch(
            source,
            raw_base,
            query,
            k,
            ef_search,
            rerank_top_r,
            scratch,
            out,
        );
        Ok(())
    }

    /// Batch sequential search returning `Vec<Vec<Neighbor>>` (zero duplicate traversal/rerank logic).
    #[allow(clippy::too_many_arguments)]
    pub fn search_batch_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
    ) -> Result<Vec<Vec<Neighbor>>> {
        self.validate_batch_params(source, raw_base, queries, k, rerank_top_r)?;
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(Vec::new());
        }
        let effective_k = k.min(self.num_vectors());
        let mut results: Vec<Vec<Neighbor>> = (0..num_queries)
            .map(|_| Vec::with_capacity(effective_k))
            .collect();
        for (qi, q_row) in queries.iter_rows().enumerate() {
            self.search_knn_validated_into_with_scratch(
                source,
                raw_base,
                q_row,
                k,
                ef_search,
                rerank_top_r,
                scratch,
                &mut results[qi],
            );
        }
        Ok(results)
    }

    /// Timed sequential batch search recording exact per-query latencies and [`SearchMetrics`].
    ///
    /// Preallocates all output vectors before the timed loop so zero heap allocations occur
    /// during measured query execution once `scratch` is warmed up.
    #[allow(clippy::too_many_arguments)]
    pub fn search_batch_timed_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
    ) -> Result<TimedQuantizedSearch> {
        self.validate_batch_params(source, raw_base, queries, k, rerank_top_r)?;
        let effective_ef = Self::effective_ef_search(k, ef_search, rerank_top_r);
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(TimedQuantizedSearch {
                results: Vec::new(),
                per_query_latency_ms: Vec::new(),
                metrics: SearchMetrics {
                    requested_ef_search: ef_search,
                    effective_ef_search: effective_ef,
                    rerank_top_r,
                    ..SearchMetrics::default()
                },
            });
        }

        let effective_k = k.min(self.num_vectors());
        let mut results: Vec<Vec<Neighbor>> = (0..num_queries)
            .map(|_| Vec::with_capacity(effective_k))
            .collect();
        let mut per_query_latency_ms = Vec::with_capacity(num_queries);

        let batch_start = Instant::now();
        for (qi, q_row) in queries.iter_rows().enumerate() {
            let q_start = Instant::now();
            self.search_knn_validated_into_with_scratch(
                source,
                raw_base,
                q_row,
                k,
                ef_search,
                rerank_top_r,
                scratch,
                &mut results[qi],
            );
            per_query_latency_ms.push(q_start.elapsed().as_secs_f64() * 1000.0);
        }
        let total_elapsed_s = batch_start.elapsed().as_secs_f64();
        let latency = compute_latency_stats(&per_query_latency_ms, total_elapsed_s);

        Ok(TimedQuantizedSearch {
            results,
            per_query_latency_ms,
            metrics: SearchMetrics {
                num_queries: latency.num_queries,
                total_s: latency.total_s,
                qps: latency.qps,
                min_ms: latency.min_ms,
                mean_ms: latency.mean_ms,
                p50_ms: latency.p50_ms,
                p95_ms: latency.p95_ms,
                p99_ms: latency.p99_ms,
                max_ms: latency.max_ms,
                requested_ef_search: ef_search,
                effective_ef_search: effective_ef,
                rerank_top_r,
            },
        })
    }

    /// Computes effective layer-0 beam width enforcing `ef_search.max(rerank_top_r).max(k)`.
    #[inline]
    #[must_use]
    pub fn effective_ef_search(k: usize, ef_search: usize, rerank_top_r: Option<usize>) -> usize {
        let base_ef = ef_search.max(k);
        match rerank_top_r {
            Some(r) => base_ef.max(r),
            None => base_ef,
        }
    }

    /// Serializes this compact graph deterministically to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(GRAPH_SNAPSHOT_MAGIC)?;
        write_u8(writer, metric_to_u8(self.metric))?;
        write_u64(writer, self.config.m as u64)?;
        write_u64(writer, self.config.m_max.map_or(u64::MAX, |v| v as u64))?;
        write_u64(writer, self.config.m_max0.map_or(u64::MAX, |v| v as u64))?;
        write_u64(writer, self.config.ef_construction as u64)?;
        match self.config.m_l {
            Some(v) => {
                write_u8(writer, 1)?;
                write_f64(writer, v)?;
            }
            None => {
                write_u8(writer, 0)?;
                write_f64(writer, 0.0)?;
            }
        }
        write_u8(writer, u8::from(self.config.extend_candidates))?;
        write_u8(writer, u8::from(self.config.keep_pruned))?;
        write_u8(
            writer,
            match self.config.keep_pruned_shrink {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            },
        )?;
        write_u64(writer, self.config.seed)?;

        write_u64(writer, self.node_meta.len() as u64)?;
        write_u64(writer, self.enter_point.map_or(u64::MAX, |ep| ep as u64))?;
        write_u64(writer, self.top_layer as u64)?;
        write_u64(writer, self.l0_stride as u64)?;
        write_u64(writer, self.upper_stride as u64)?;

        for meta in &self.node_meta {
            write_u8(writer, meta.max_layer)?;
            write_u32(writer, meta.upper_offset)?;
        }

        write_u32_slice(writer, &self.layer0)?;
        write_u32_slice(writer, &self.upper_arena)?;
        Ok(())
    }

    /// Reads and structurally validates a [`CompactHnswGraph`] from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        read_magic(reader, GRAPH_SNAPSHOT_MAGIC, "CompactHnswGraph")?;

        let metric = metric_from_u8(read_u8(reader)?)?;
        let m = read_usize_u64(reader, "CompactHnswGraph m")?;
        let m_max = read_opt_usize_u64(reader, "CompactHnswGraph m_max")?;
        let m_max0 = read_opt_usize_u64(reader, "CompactHnswGraph m_max0")?;
        let ef_construction = read_usize_u64(reader, "CompactHnswGraph ef_construction")?;
        let has_m_l = read_u8(reader)?;
        let m_l_val = read_f64(reader)?;
        let m_l = match has_m_l {
            0 => None,
            1 => Some(m_l_val),
            other => {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid m_l tag in CompactHnswGraph snapshot: {other}"
                )));
            }
        };
        let extend_candidates = read_bool_u8(reader, "extend_candidates")?;
        let keep_pruned = read_bool_u8(reader, "keep_pruned")?;
        let keep_pruned_shrink = match read_u8(reader)? {
            0 => None,
            1 => Some(false),
            2 => Some(true),
            other => {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid keep_pruned_shrink tag: {other}"
                )));
            }
        };
        let seed = read_u64(reader)?;

        let config = HnswConfig {
            m,
            m_max,
            m_max0,
            ef_construction,
            m_l,
            extend_candidates,
            keep_pruned,
            keep_pruned_shrink,
            seed,
        };
        config.validate()?;

        let num_vectors = read_usize_u64(reader, "CompactHnswGraph num_vectors")?;
        if num_vectors > u32::MAX as usize {
            return Err(VectorSearchError::CorruptData(format!(
                "CompactHnswGraph num_vectors {num_vectors} exceeds u32::MAX"
            )));
        }

        let enter_point = match read_opt_usize_u64(reader, "CompactHnswGraph entry_point")? {
            None => None,
            Some(ep) if ep < num_vectors => Some(ep),
            Some(ep) => {
                return Err(VectorSearchError::CorruptData(format!(
                    "CompactHnswGraph entry_point {ep} out of range for {num_vectors} nodes"
                )));
            }
        };
        if num_vectors == 0 && enter_point.is_some() {
            return Err(VectorSearchError::CorruptData(
                "Empty CompactHnswGraph cannot have an entry point".to_string(),
            ));
        }

        let top_layer = read_usize_u64(reader, "CompactHnswGraph top_layer")?;
        let l0_stride = read_usize_u64(reader, "CompactHnswGraph l0_stride")?;
        let upper_stride = read_usize_u64(reader, "CompactHnswGraph upper_stride")?;

        let expected_m_max0 = config.effective_m_max0();
        let expected_m_max = config.effective_m_max();
        if l0_stride != expected_m_max0 + 1 || upper_stride != expected_m_max + 1 {
            return Err(VectorSearchError::CorruptData(
                "CompactHnswGraph strides do not match config".to_string(),
            ));
        }

        let mut node_meta = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let max_layer = read_u8(reader)?;
            let upper_offset = read_u32(reader)?;
            node_meta.push(HnswNodeMeta {
                max_layer,
                upper_offset,
            });
        }

        let expected_l0_len = num_vectors.checked_mul(l0_stride).ok_or_else(|| {
            VectorSearchError::CorruptData("CompactHnswGraph layer0 length overflows".to_string())
        })?;
        let layer0 = read_u32_vec(reader, expected_l0_len)?;
        if layer0.len() != expected_l0_len {
            return Err(VectorSearchError::CorruptData(format!(
                "CompactHnswGraph layer0 length {} != expected {expected_l0_len}",
                layer0.len()
            )));
        }

        let max_upper_len = num_vectors
            .saturating_mul(u8::MAX as usize)
            .saturating_mul(upper_stride);
        let upper_arena = read_u32_vec(reader, max_upper_len)?;

        let mut actual_top = 0_usize;
        // Structural validation of degrees and neighbor IDs so searches never panic out of bounds.
        for (id, meta) in node_meta.iter().enumerate() {
            actual_top = actual_top.max(meta.max_layer as usize);
            let base0 = id * l0_stride;
            let deg0 = layer0[base0] as usize;
            if deg0 > expected_m_max0 {
                return Err(VectorSearchError::CorruptData(format!(
                    "Node {id} layer 0 degree {deg0} exceeds m_max0 {expected_m_max0}"
                )));
            }
            for &nid in &layer0[base0 + 1..base0 + 1 + deg0] {
                if (nid as usize) >= num_vectors {
                    return Err(VectorSearchError::CorruptData(format!(
                        "Node {id} layer 0 neighbor {nid} >= num_vectors {num_vectors}"
                    )));
                }
            }

            let max_l = meta.max_layer as usize;
            if max_l > 0 {
                let off = meta.upper_offset as usize;
                let end = off.checked_add(max_l * upper_stride).ok_or_else(|| {
                    VectorSearchError::CorruptData(
                        "CompactHnswGraph upper_arena offset overflow".to_string(),
                    )
                })?;
                if end > upper_arena.len() {
                    return Err(VectorSearchError::CorruptData(format!(
                        "Node {id} upper arena slice [{off}..{end}) exceeds len {}",
                        upper_arena.len()
                    )));
                }
                for lc in 1..=max_l {
                    let base = off + (lc - 1) * upper_stride;
                    let deg = upper_arena[base] as usize;
                    if deg > expected_m_max {
                        return Err(VectorSearchError::CorruptData(format!(
                            "Node {id} layer {lc} degree {deg} exceeds m_max {expected_m_max}"
                        )));
                    }
                    for &nid in &upper_arena[base + 1..base + 1 + deg] {
                        if (nid as usize) >= num_vectors {
                            return Err(VectorSearchError::CorruptData(format!(
                                "Node {id} layer {lc} neighbor {nid} >= num_vectors {num_vectors}"
                            )));
                        }
                    }
                }
            }
        }

        if top_layer != actual_top {
            return Err(VectorSearchError::CorruptData(format!(
                "CompactHnswGraph top_layer ({top_layer}) != max node layer ({actual_top})"
            )));
        }

        Ok(Self {
            metric,
            config,
            node_meta,
            layer0,
            upper_arena,
            l0_stride,
            upper_stride,
            enter_point,
            top_layer,
        })
    }

    fn validate_common_inputs<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        k: usize,
        rerank_top_r: Option<usize>,
    ) -> Result<()> {
        if k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "k must be positive (> 0)".to_string(),
            ));
        }
        if rerank_top_r == Some(0) {
            return Err(VectorSearchError::InvalidArgument(
                "rerank_top_r must be positive (> 0) when specified; pass None to disable reranking"
                    .to_string(),
            ));
        }
        if source.len() != self.node_meta.len() {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Distance source len ({}) does not match graph node count ({})",
                source.len(),
                self.node_meta.len()
            )));
        }
        if source.metric() != self.metric {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Distance source metric ({}) does not match graph metric ({})",
                source.metric(),
                self.metric
            )));
        }
        if let Some(r) = rerank_top_r {
            let _ = r;
            let raw = raw_base.ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "Reranking requires raw f32 base vectors (WithExactRerank mode or external raw_base)"
                        .to_string(),
                )
            })?;
            if raw.rows() != self.node_meta.len() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "raw_base row count ({}) does not match graph node count ({})",
                    raw.rows(),
                    self.node_meta.len()
                )));
            }
            if source.dim() > 0 && raw.cols() != source.dim() {
                return Err(VectorSearchError::DimensionMismatch {
                    expected: source.dim(),
                    actual: raw.cols(),
                });
            }
        }
        Ok(())
    }

    fn validate_query_params<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        query: &[f32],
        k: usize,
        rerank_top_r: Option<usize>,
    ) -> Result<()> {
        self.validate_common_inputs(source, raw_base, k, rerank_top_r)?;
        if query.iter().any(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(
                "query vector contains non-finite value (NaN/Inf)".to_string(),
            ));
        }
        if source.dim() > 0 && query.len() != source.dim() {
            return Err(VectorSearchError::DimensionMismatch {
                expected: source.dim(),
                actual: query.len(),
            });
        }
        if let Some(raw) = raw_base {
            if raw.rows() > 0 && query.len() != raw.cols() {
                return Err(VectorSearchError::DimensionMismatch {
                    expected: raw.cols(),
                    actual: query.len(),
                });
            }
        }
        Ok(())
    }

    fn validate_batch_params<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        queries: &RowMatrix<f32>,
        k: usize,
        rerank_top_r: Option<usize>,
    ) -> Result<()> {
        self.validate_common_inputs(source, raw_base, k, rerank_top_r)?;
        if queries.rows() > 0 {
            if source.dim() > 0 && queries.cols() != source.dim() {
                return Err(VectorSearchError::DimensionMismatch {
                    expected: source.dim(),
                    actual: queries.cols(),
                });
            }
            if let Some(raw) = raw_base {
                if raw.rows() > 0 && queries.cols() != raw.cols() {
                    return Err(VectorSearchError::DimensionMismatch {
                        expected: raw.cols(),
                        actual: queries.cols(),
                    });
                }
            }
            if queries.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(VectorSearchError::InvalidArgument(
                    "query vector contains non-finite value (NaN/Inf)".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn search_knn_validated_into_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
        raw_base: Option<&RowMatrix<f32>>,
        query: &[f32],
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<S::QueryState>,
        out: &mut Vec<Neighbor>,
    ) {
        out.clear();
        let Some(mut curr_ep) = self.enter_point else {
            return;
        };

        let effective_ef = Self::effective_ef_search(k, ef_search, rerank_top_r);
        source.prepare_query(query, &mut scratch.query_state);

        // Phase 1: Top-down greedy zoom-in (ef = 1) down to layer 1
        if self.top_layer >= 1 {
            for lc in (1..=self.top_layer).rev() {
                self.search_layer_core(
                    curr_ep,
                    1,
                    lc,
                    &mut scratch.tracker,
                    &mut scratch.frontier,
                    &mut scratch.results,
                    |nid| source.query_distance(&scratch.query_state, nid),
                );
                if let Some(best) = scratch.results.iter().min_by(|a, b| a.cmp(b)) {
                    curr_ep = best.id;
                }
            }
        }

        // Phase 2: Base layer quantized beam search at layer 0
        self.search_layer_core(
            curr_ep,
            effective_ef,
            0,
            &mut scratch.tracker,
            &mut scratch.frontier,
            &mut scratch.results,
            |nid| source.query_distance(&scratch.query_state, nid),
        );

        drain_heap_sorted_asc(&mut scratch.results, &mut scratch.raw_candidates);

        // Phase 3: Optional exact top-R reranking or direct quantized top-k output
        match (rerank_top_r, raw_base) {
            (Some(r), Some(raw)) => {
                let effective_r = r.max(k);
                let num_rerank = effective_r.min(scratch.raw_candidates.len());
                scratch.rerank_candidates.clear();
                for cand in &scratch.raw_candidates[..num_rerank] {
                    let sort_key = compute_exact_sort_key(self.metric, query, raw.row(cand.id));
                    scratch.rerank_candidates.push((sort_key, cand.id));
                }
                scratch
                    .rerank_candidates
                    .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                let take_k = k.min(scratch.rerank_candidates.len());
                for &(sort_key, id) in &scratch.rerank_candidates[..take_k] {
                    out.push(Neighbor {
                        id,
                        distance: format_metric_output_distance(self.metric, sort_key),
                    });
                }
            }
            _ => {
                let take_k = k.min(scratch.raw_candidates.len());
                for cand in &scratch.raw_candidates[..take_k] {
                    out.push(Neighbor {
                        id: cand.id,
                        distance: source.format_output_distance(cand.distance),
                    });
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search_layer_core<F>(
        &self,
        ep: usize,
        ef: usize,
        lc: usize,
        tracker: &mut VisitedTracker,
        frontier: &mut BinaryHeap<Reverse<Candidate>>,
        results: &mut BinaryHeap<Candidate>,
        mut dist_fn: F,
    ) where
        F: FnMut(usize) -> f32,
    {
        tracker.prepare(self.node_meta.len());
        tracker.mark_visited(ep);

        let cand_ep = Candidate {
            distance: dist_fn(ep),
            id: ep,
        };

        frontier.clear();
        results.clear();

        frontier.push(Reverse(cand_ep));
        results.push(cand_ep);

        while let Some(Reverse(curr)) = frontier.pop() {
            let furthest = *results.peek().expect("W never empty during search");
            if curr.distance > furthest.distance {
                break;
            }

            for &neighbor_u32 in self.neighbors(lc, curr.id) {
                let neighbor_id = neighbor_u32 as usize;
                if !tracker.is_visited(neighbor_id) {
                    tracker.mark_visited(neighbor_id);
                    let furthest_now = *results.peek().expect("W never empty");
                    let cand_neighbor = Candidate {
                        distance: dist_fn(neighbor_id),
                        id: neighbor_id,
                    };

                    if cand_neighbor < furthest_now || results.len() < ef {
                        frontier.push(Reverse(cand_neighbor));
                        results.push(cand_neighbor);

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn select_neighbors_heuristic_with_scratch<S: QuantizedDistanceSource>(
        &self,
        source: &S,
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

        for &cand in candidates {
            if cand.id != base_id && scratch.seen.insert(cand.id) {
                scratch.w_set.push(cand);
            }
        }

        if extend_candidates {
            for &cand in candidates {
                for &adj_u32 in self.neighbors(lc, cand.id) {
                    let adj_id = adj_u32 as usize;
                    if adj_id != base_id && scratch.seen.insert(adj_id) {
                        let d_adj = source.node_distance(adj_id, base_id);
                        scratch.w_set.push(Candidate {
                            distance: d_adj,
                            id: adj_id,
                        });
                    }
                }
            }
        }

        scratch.w_d.clear();
        for &cand in &scratch.w_set {
            scratch.w_d.push(Reverse(cand));
        }

        scratch.r.clear();
        scratch.w_pruned.clear();

        while let Some(Reverse(e)) = scratch.w_d.pop() {
            if scratch.r.len() >= m {
                break;
            }

            let mut is_diverse = true;
            for &r_id in &scratch.r {
                let d_e_r = source.node_distance(e.id, r_id);
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

        if keep_pruned && scratch.r.len() < m {
            for &pruned_cand in &scratch.w_pruned {
                if scratch.r.len() >= m {
                    break;
                }
                scratch.r.push(pruned_cand.id);
            }
        }
    }

    #[inline]
    fn layer_slot_base(&self, layer: usize, node: usize) -> (bool, usize) {
        if layer == 0 {
            (true, node * self.l0_stride)
        } else {
            let base =
                (self.node_meta[node].upper_offset as usize) + (layer - 1) * self.upper_stride;
            (false, base)
        }
    }

    #[inline]
    fn degree_at(&self, layer: usize, node: usize) -> usize {
        let (is_l0, base) = self.layer_slot_base(layer, node);
        if is_l0 {
            self.layer0[base] as usize
        } else {
            self.upper_arena[base] as usize
        }
    }

    #[inline]
    fn push_neighbor_unique(&mut self, layer: usize, node: usize, target: u32) {
        let (is_l0, base) = self.layer_slot_base(layer, node);
        if is_l0 {
            let deg = self.layer0[base] as usize;
            if !self.layer0[base + 1..base + 1 + deg].contains(&target) {
                self.layer0[base + 1 + deg] = target;
                self.layer0[base] = (deg + 1) as u32;
            }
        } else {
            let deg = self.upper_arena[base] as usize;
            if !self.upper_arena[base + 1..base + 1 + deg].contains(&target) {
                self.upper_arena[base + 1 + deg] = target;
                self.upper_arena[base] = (deg + 1) as u32;
            }
        }
    }

    #[inline]
    fn set_neighbors_from_slice(&mut self, layer: usize, node: usize, selected: &[usize]) {
        let (is_l0, base) = self.layer_slot_base(layer, node);
        let deg = selected.len();
        if is_l0 {
            self.layer0[base] = deg as u32;
            for (i, &id) in selected.iter().enumerate() {
                self.layer0[base + 1 + i] = id as u32;
            }
        } else {
            self.upper_arena[base] = deg as u32;
            for (i, &id) in selected.iter().enumerate() {
                self.upper_arena[base + 1 + i] = id as u32;
            }
        }
    }
}

#[inline]
fn sample_layer(rng: &mut DeterministicRng, m_l: f64) -> usize {
    let u = rng.next_f64_open01();
    let l = (-u.ln() * m_l).floor();
    if l <= 0.0 || l.is_nan() {
        0
    } else {
        l as usize
    }
}

/// Drains a max-heap of [`Candidate`]s into `out` sorted strictly ascending without
/// consuming or shrinking `heap`'s underlying buffer capacity.
#[inline]
fn drain_heap_sorted_asc(heap: &mut BinaryHeap<Candidate>, out: &mut Vec<Candidate>) {
    out.clear();
    while let Some(cand) = heap.pop() {
        out.push(cand);
    }
    out.reverse();
}

#[inline]
fn read_opt_usize_u64<R: Read>(reader: &mut R, label: &str) -> Result<Option<usize>> {
    let raw = read_u64(reader)?;
    if raw == u64::MAX {
        Ok(None)
    } else {
        usize::try_from(raw)
            .map(Some)
            .map_err(|_| VectorSearchError::CorruptData(format!("{label} overflows usize")))
    }
}

#[inline]
fn read_bool_u8<R: Read>(reader: &mut R, label: &str) -> Result<bool> {
    match read_u8(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(VectorSearchError::CorruptData(format!(
            "Invalid {label} byte: {other}"
        ))),
    }
}
