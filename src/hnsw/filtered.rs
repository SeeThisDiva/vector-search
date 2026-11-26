//! Filtered Approximate Nearest Neighbor Search strategies over [`HnswIndex`].
//!
//! Implements Strategies 1–3 (Milestone M3):
//! - Strategy 1: Post-filter over-fetch with iterative widening ([`FilterStrategy::PostFilterOverfetch`])
//! - Strategy 2: In-traversal predicate filtering with bridge routing and hop budget ([`FilterStrategy::InTraversal`])
//! - Strategy 3: Pre-filter bitset brute-force scan ([`FilterStrategy::PreFilterBruteForce`])

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::{drain_heap_asc, Candidate, HnswIndex, VisitedTracker};
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::resolve_worker_count;
use crate::filter::{
    DenseBitset, FilterPredicate, FilteredSearchFlags, FilteredSearchOutput, FilteredSearchParams,
};
use crate::matrix::RowMatrix;

/// Execution strategy for filtered approximate nearest neighbor search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrategy {
    /// Strategy 1: Post-filtering with iterative widening over-fetch candidate pool.
    PostFilterOverfetch,
    /// Strategy 2: In-traversal predicate filtering with bridge-node routing.
    InTraversal,
    /// Strategy 3: Pre-filter brute force flat scan over bitset-backed predicate.
    PreFilterBruteForce,
}

/// Reusable per-thread scratch buffers for filtered HNSW queries.
#[derive(Debug, Clone, Default)]
pub struct FilteredSearchScratch {
    /// Visited tracking tag array.
    pub tracker: VisitedTracker,
    /// Priority queue for candidate exploration (Min-Heap).
    pub frontier: BinaryHeap<Reverse<Candidate>>,
    /// Priority queue for matching results (Max-Heap).
    pub matched_results: BinaryHeap<Candidate>,
    /// Temporary candidate collection buffer.
    pub raw_candidates: Vec<Candidate>,
    /// Over-fetch staging buffer for Strategy 1.
    pub overfetch_buffer: Vec<(f32, usize)>,
}

impl FilteredSearchScratch {
    /// Preallocates every buffer for the largest query the caller will issue.
    ///
    /// Buffers are pre-sized so inner loops do not grow scratch allocations during queries.
    #[must_use]
    pub fn new(num_nodes: usize, max_ef: usize, overfetch_cap: usize) -> Self {
        Self {
            tracker: VisitedTracker::new(num_nodes),
            frontier: BinaryHeap::with_capacity(num_nodes),
            matched_results: BinaryHeap::with_capacity(max_ef + 1),
            raw_candidates: Vec::with_capacity(overfetch_cap.max(max_ef + 1)),
            overfetch_buffer: Vec::with_capacity(overfetch_cap),
        }
    }
}

/// Per-query execution counters collected outside timed hot paths via
/// [`HnswIndex::search_knn_filtered_with_counters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FilteredSearchCounters {
    /// Total vector distance computations performed (`valid_distance_evals + bridge_distance_evals`).
    pub total_distance_evals: u64,
    /// Distance computations evaluated on base vectors matching the filter predicate.
    pub valid_distance_evals: u64,
    /// Distance computations evaluated on non-matching routing / bridge vectors.
    pub bridge_distance_evals: u64,
    /// Subset of total distance computations executed during upper-layer (`l >= 1`) zoom-in.
    pub upper_layer_distance_evals: u64,
    /// Layer 0 vertices visited (distance-evaluated) during filtered search.
    pub visited_nodes: u64,
    /// Subset of `visited_nodes` satisfying the filter predicate (`|V_visited intersect S_match|`).
    pub visited_matching_nodes: u64,
}

impl FilteredSearchCounters {
    /// Graph traversal efficiency ratio `eta_graph = |V_visited intersect S_match| / |V_visited|`.
    #[inline]
    #[must_use]
    pub fn eta_graph(&self) -> f64 {
        if self.visited_nodes == 0 {
            0.0
        } else {
            (self.visited_matching_nodes as f64) / (self.visited_nodes as f64)
        }
    }
}

#[inline]
pub(super) fn record_layer_distance_counter(
    counters: &mut Option<&mut FilteredSearchCounters>,
    nid: usize,
    layer: usize,
    predicate_bitset: Option<&DenseBitset>,
) {
    if let Some(c_out) = counters.as_deref_mut() {
        let is_match = predicate_bitset.is_some_and(|bs| bs.get_bit(nid));
        c_out.total_distance_evals += 1;
        if is_match {
            c_out.valid_distance_evals += 1;
        } else {
            c_out.bridge_distance_evals += 1;
        }
        if layer >= 1 {
            c_out.upper_layer_distance_evals += 1;
        } else {
            c_out.visited_nodes += 1;
            if is_match {
                c_out.visited_matching_nodes += 1;
            }
        }
    }
}

#[inline]
fn record_layer0_match_counter(counters: &mut Option<&mut FilteredSearchCounters>, is_match: bool) {
    if let Some(c_out) = counters.as_deref_mut() {
        c_out.total_distance_evals += 1;
        c_out.visited_nodes += 1;
        if is_match {
            c_out.valid_distance_evals += 1;
            c_out.visited_matching_nodes += 1;
        } else {
            c_out.bridge_distance_evals += 1;
        }
    }
}

impl<'a> HnswIndex<'a> {
    // =========================================================================
    // Milestone M3 Task T3.4: Filtered Search (Strategy 1 Post-Filter Over-Fetch)
    // =========================================================================

    /// Executes Strategy 1 post-filter approximate nearest neighbor search with iterative widening over-fetch.
    ///
    /// # Sizing Formula & Widening Loop
    /// - Initial candidate pool size:
    ///   $$k'_0 = \min\left(K'_{\max}, \ \left\lceil \frac{k}{\sigma} \cdot \gamma \right\rceil\right)$$
    ///   where $\gamma(\sigma, \delta) \approx 1 + z_{1-\delta} \sqrt{\frac{1 - \sigma}{k}}$ with $\delta = 0.001$ ($z_{1-\delta} \approx 3.0902323$).
    ///   If `params.overfetch_factor` is explicitly set, it overrides $\gamma$.
    /// - Widening step:
    ///   $$k'_{i+1} = \min\left(K'_{\max}, \ \max(k'_i + 1, \lfloor 2.5 \cdot k'_i \rfloor)\right)$$
    /// - Loop terminates when $m \ge k$ (with `post_filter_cap_reached = false`) or when $k'_i == K'_{\max}$ (with `post_filter_cap_reached = true`).
    pub fn search_knn_filtered_post<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
    ) -> Result<FilteredSearchOutput> {
        let mut tracker = VisitedTracker::new(self.node_meta.len());
        self.search_knn_filtered_post_with_tracker(query, k, predicate, params, &mut tracker)
    }

    /// Read-only Strategy 1 post-filtering search reusing an externally provided [`VisitedTracker`].
    pub fn search_knn_filtered_post_with_tracker<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        tracker: &mut VisitedTracker,
    ) -> Result<FilteredSearchOutput> {
        self.search_knn_filtered_post_core::<P, false>(
            query, k, predicate, params, tracker, None, None,
        )
    }

    /// Read-only Strategy 1 post-filtering search reusing preallocated [`FilteredSearchScratch`].
    pub fn search_knn_filtered_post_with_scratch<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        scratch: &mut FilteredSearchScratch,
    ) -> Result<FilteredSearchOutput> {
        let FilteredSearchScratch {
            tracker,
            frontier,
            matched_results,
            raw_candidates,
            overfetch_buffer,
        } = scratch;
        self.search_knn_filtered_post_core::<P, false>(
            query,
            k,
            predicate,
            params,
            tracker,
            Some((frontier, matched_results, raw_candidates, overfetch_buffer)),
            None,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn search_knn_filtered_post_core<P: FilterPredicate, const COLLECT_COUNTERS: bool>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        tracker: &mut VisitedTracker,
        mut scratch_bufs: Option<(
            &mut BinaryHeap<Reverse<Candidate>>,
            &mut BinaryHeap<Candidate>,
            &mut Vec<Candidate>,
            &mut Vec<(f32, usize)>,
        )>,
        mut counters: Option<&mut FilteredSearchCounters>,
    ) -> Result<FilteredSearchOutput> {
        self.validate_query_slice(query, query.len(), k)?;

        if self.node_meta.is_empty() || self.enter_point.is_none() {
            return Ok(FilteredSearchOutput {
                results: Vec::new(),
                flags: FilteredSearchFlags::default(),
            });
        }

        let is_bitset = predicate.bitset().is_some();
        if !is_bitset && matches!(params.eval_limit, Some(0)) {
            return Ok(FilteredSearchOutput {
                results: Vec::new(),
                flags: FilteredSearchFlags {
                    eval_limit_reached: true,
                    ..Default::default()
                },
            });
        }

        let total = self.node_meta.len();
        let k_max = params.overfetch_cap.min(total);
        if k_max == 0 {
            if params.allow_bruteforce_fallback && is_bitset {
                let mut fallback_out = self.search_knn_filtered_pre_inner::<P, COLLECT_COUNTERS>(
                    query,
                    k,
                    predicate,
                    tracker,
                    scratch_bufs
                        .as_mut()
                        .map(|(_, m, r, _)| (&mut **m, &mut **r)),
                    counters,
                )?;
                fallback_out.flags.post_filter_cap_reached = true;
                fallback_out.flags.used_bruteforce_fallback = true;
                return Ok(fallback_out);
            }
            return Ok(FilteredSearchOutput {
                results: Vec::new(),
                flags: FilteredSearchFlags {
                    post_filter_cap_reached: true,
                    ..Default::default()
                },
            });
        }

        // Determine selectivity sigma
        let sigma = if let Some(count) = predicate.count_matching(total) {
            if total == 0 {
                1.0
            } else {
                (count as f64) / (total as f64)
            }
        } else if let Some(est) = predicate.estimated_selectivity() {
            if est.is_finite() {
                est.clamp(0.0, 1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Determine over-fetch factor gamma
        let gamma = if let Some(user_factor) = params.overfetch_factor {
            user_factor
        } else if sigma <= 0.0 {
            1.0
        } else {
            const Z_DELTA: f64 = 3.090_232_306_143_706; // z_{1-delta} for delta = 0.001
            let variance_term = ((1.0 - sigma).max(0.0) / (k as f64)).sqrt();
            1.0 + Z_DELTA * variance_term
        };

        // Initial candidate pool size k'_0
        let mut k_prime = if sigma <= 0.0 {
            k_max
        } else {
            let raw = ((k as f64) / sigma) * gamma;
            let raw_ceil = raw.ceil();
            if raw_ceil.is_nan() || raw_ceil >= (k_max as f64) {
                k_max
            } else {
                (raw_ceil as usize).max(1).min(k_max)
            }
        };

        let mut local_best: Vec<(f32, usize)> = Vec::new();
        let mut seen_opaque: HashSet<usize> = HashSet::new();
        let mut eval_count = 0;

        let (mut heap_bufs, best_matches) = match scratch_bufs {
            Some((frontier, matched_res, raw_cands, overfetch_buf)) => {
                overfetch_buf.clear();
                (Some((frontier, matched_res, raw_cands)), overfetch_buf)
            }
            None => (None, &mut local_best),
        };

        loop {
            let current_ef = params.ef_search.max(k_prime).min(total);
            let mut eval_limit_reached = false;

            {
                let mut try_push_candidate = |d: f32, id: usize, limit_hit: &mut bool| -> bool {
                    if !is_bitset && !seen_opaque.insert(id) {
                        return true;
                    }
                    if let Some(limit) = params.eval_limit {
                        if !is_bitset && eval_count >= limit {
                            *limit_hit = true;
                            return false;
                        }
                    }
                    if !is_bitset {
                        eval_count += 1;
                    }
                    if predicate.matches(id as u32)
                        && (!is_bitset || !best_matches.iter().any(|m| m.1 == id))
                    {
                        if best_matches.capacity() > k
                            && best_matches.len() == best_matches.capacity()
                        {
                            best_matches
                                .sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                            best_matches.truncate(k);
                        }
                        best_matches.push((d, id));
                    }
                    true
                };

                let used_scratch_heaps = if let Some((frontier, matched_res, raw_cands)) =
                    heap_bufs.as_mut()
                {
                    if frontier.capacity() >= total
                        && matched_res.capacity() > current_ef
                        && raw_cands.capacity() >= current_ef
                    {
                        let ep = self.enter_point.expect("enter_point checked non-empty");
                        let pred_bitset = predicate.bitset();
                        let curr_ep = self.search_top_layers_greedy::<COLLECT_COUNTERS>(
                            query,
                            ep,
                            tracker,
                            frontier,
                            matched_res,
                            pred_bitset,
                            counters.as_deref_mut(),
                        );
                        self.search_layer_with_scratch_inner::<COLLECT_COUNTERS>(
                            query,
                            curr_ep,
                            current_ef,
                            0,
                            tracker,
                            frontier,
                            matched_res,
                            pred_bitset,
                            counters.as_deref_mut(),
                        );
                        drain_heap_asc(matched_res, raw_cands);
                        let take_k = k_prime.min(raw_cands.len());
                        for cand in &raw_cands[..take_k] {
                            if !try_push_candidate(cand.distance, cand.id, &mut eval_limit_reached)
                            {
                                break;
                            }
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !used_scratch_heaps {
                    let candidates = self.search_knn_raw_inner::<COLLECT_COUNTERS>(
                        query,
                        k_prime,
                        current_ef,
                        tracker,
                        predicate.bitset(),
                        counters.as_deref_mut(),
                    )?;
                    for cand in candidates {
                        if !try_push_candidate(cand.distance, cand.id, &mut eval_limit_reached) {
                            break;
                        }
                    }
                }
            }

            best_matches.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            if best_matches.len() >= k {
                best_matches.truncate(k);
                let results = best_matches
                    .iter()
                    .map(|&(key, id)| (self.format_output_distance(key), id))
                    .collect();
                return Ok(FilteredSearchOutput {
                    results,
                    flags: FilteredSearchFlags {
                        post_filter_cap_reached: false,
                        eval_limit_reached,
                        ..Default::default()
                    },
                });
            }

            if eval_limit_reached || k_prime == k_max {
                if params.allow_bruteforce_fallback && is_bitset {
                    let mut fallback_out = self
                        .search_knn_filtered_pre_inner::<P, COLLECT_COUNTERS>(
                            query,
                            k,
                            predicate,
                            tracker,
                            heap_bufs.as_mut().map(|(_, m, r)| (&mut **m, &mut **r)),
                            counters.as_deref_mut(),
                        )?;
                    fallback_out.flags.eval_limit_reached = eval_limit_reached;
                    fallback_out.flags.post_filter_cap_reached =
                        !eval_limit_reached && k_prime == k_max;
                    fallback_out.flags.used_bruteforce_fallback = true;
                    return Ok(fallback_out);
                }
                let results = best_matches
                    .iter()
                    .map(|&(key, id)| (self.format_output_distance(key), id))
                    .collect();
                return Ok(FilteredSearchOutput {
                    results,
                    flags: FilteredSearchFlags {
                        post_filter_cap_reached: !eval_limit_reached && k_prime == k_max,
                        eval_limit_reached,
                        ..Default::default()
                    },
                });
            }

            // Widening step: k'_{i+1} = min(K'_max, max(k'_i + 1, floor(2.5 * k'_i)))
            let next_k = (k_prime + 1).max((2.5 * k_prime as f64).floor() as usize);
            k_prime = next_k.min(k_max);
        }
    }

    /// Unified dispatch method executing filtered search under the selected strategy.
    pub fn search_knn_filtered<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        strategy: FilterStrategy,
        params: &FilteredSearchParams,
    ) -> Result<FilteredSearchOutput> {
        let mut tracker = VisitedTracker::new(self.node_meta.len());
        self.search_knn_filtered_with_tracker(query, k, predicate, strategy, params, &mut tracker)
    }

    /// Read-only unified filtered search dispatch reusing an externally provided [`VisitedTracker`].
    pub fn search_knn_filtered_with_tracker<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        strategy: FilterStrategy,
        params: &FilteredSearchParams,
        tracker: &mut VisitedTracker,
    ) -> Result<FilteredSearchOutput> {
        match strategy {
            FilterStrategy::PostFilterOverfetch => {
                self.search_knn_filtered_post_with_tracker(query, k, predicate, params, tracker)
            }
            FilterStrategy::InTraversal => self
                .search_knn_filtered_traversal_with_tracker(query, k, predicate, params, tracker),
            FilterStrategy::PreFilterBruteForce => {
                self.search_knn_filtered_pre_with_tracker(query, k, predicate, params, tracker)
            }
        }
    }

    /// Read-only unified filtered search dispatch reusing preallocated [`FilteredSearchScratch`].
    pub fn search_knn_filtered_with_scratch<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        strategy: FilterStrategy,
        params: &FilteredSearchParams,
        scratch: &mut FilteredSearchScratch,
    ) -> Result<FilteredSearchOutput> {
        match strategy {
            FilterStrategy::PostFilterOverfetch => {
                self.search_knn_filtered_post_with_scratch(query, k, predicate, params, scratch)
            }
            FilterStrategy::InTraversal => self
                .search_knn_filtered_traversal_with_scratch(query, k, predicate, params, scratch),
            FilterStrategy::PreFilterBruteForce => {
                self.search_knn_filtered_pre_with_scratch(query, k, predicate, params, scratch)
            }
        }
    }

    /// Read-only unified filtered search dispatch collecting per-query execution counters.
    ///
    /// Intended for untimed measurement passes outside the timed loop so timed calls
    /// to [`Self::search_knn_filtered_with_scratch`] pay zero counter overhead via
    /// compile-time monomorphization.
    pub fn search_knn_filtered_with_counters<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        strategy: FilterStrategy,
        params: &FilteredSearchParams,
        scratch: &mut FilteredSearchScratch,
    ) -> Result<(FilteredSearchOutput, FilteredSearchCounters)> {
        let mut counters = FilteredSearchCounters::default();
        let FilteredSearchScratch {
            tracker,
            frontier,
            matched_results,
            raw_candidates,
            overfetch_buffer,
        } = scratch;
        let output = match strategy {
            FilterStrategy::PostFilterOverfetch => self.search_knn_filtered_post_core::<P, true>(
                query,
                k,
                predicate,
                params,
                tracker,
                Some((frontier, matched_results, raw_candidates, overfetch_buffer)),
                Some(&mut counters),
            )?,
            FilterStrategy::InTraversal => self.search_knn_filtered_traversal_core::<P, true>(
                query,
                k,
                predicate,
                params,
                tracker,
                Some((frontier, matched_results, raw_candidates)),
                Some(&mut counters),
            )?,
            FilterStrategy::PreFilterBruteForce => self.search_knn_filtered_pre_inner::<P, true>(
                query,
                k,
                predicate,
                tracker,
                Some((matched_results, raw_candidates)),
                Some(&mut counters),
            )?,
        };
        Ok((output, counters))
    }

    /// Batch sequential query execution for Strategy 1 post-filtering.
    pub fn search_batch_filtered_post<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
    ) -> Result<Vec<FilteredSearchOutput>> {
        self.run_filtered_batch_seq(queries, |q_row, tracker| {
            self.search_knn_filtered_post_with_tracker(q_row, k, predicate, params, tracker)
        })
    }

    /// Multi-threaded batch search for Strategy 1 post-filtering.
    pub fn search_batch_filtered_post_parallel<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        num_threads: usize,
    ) -> Result<Vec<FilteredSearchOutput>> {
        self.run_filtered_batch_parallel(queries, k, num_threads, None, |q_row, tracker| {
            self.search_knn_filtered_post_with_tracker(q_row, k, predicate, params, tracker)
        })
    }

    // =========================================================================
    // Milestone M3 Task T3.5: Filtered Search (Strategy 2 In-Traversal Filtering)
    // =========================================================================

    /// Internal Layer 0 traversal with bridge nodes and hop budget (Algorithm 3).
    #[allow(clippy::too_many_arguments)]
    fn search_layer0_in_traversal_inner<P: FilterPredicate, const COLLECT_COUNTERS: bool>(
        &self,
        query: &[f32],
        curr_ep: usize,
        k: usize,
        effective_ef: usize,
        h_max: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        tracker: &mut VisitedTracker,
        c: &mut BinaryHeap<Reverse<Candidate>>,
        w_match: &mut BinaryHeap<Candidate>,
        mut counters: Option<&mut FilteredSearchCounters>,
    ) -> (bool, bool) {
        tracker.prepare(self.node_meta.len());
        tracker.mark_visited(curr_ep);

        let d_ep = self.compute_distance(query, curr_ep);
        #[cfg(feature = "instrument")]
        let mut dist_evals = 1_u64;

        let cand_ep = Candidate {
            distance: d_ep,
            id: curr_ep,
        };

        c.clear();
        w_match.clear();

        let mut eval_count = 0;
        let mut eval_limit_reached = false;

        if let Some(limit) = params.eval_limit {
            if predicate.bitset().is_none() && eval_count >= limit {
                eval_limit_reached = true;
            }
        }

        let ep_matches = if !eval_limit_reached {
            if predicate.bitset().is_none() {
                eval_count += 1;
            }
            predicate.matches(curr_ep as u32)
        } else {
            false
        };
        if COLLECT_COUNTERS {
            record_layer0_match_counter(&mut counters, ep_matches);
        }

        if ep_matches {
            w_match.push(cand_ep);
        }
        c.push(Reverse(cand_ep));

        let mut h = 0;
        let mut hop_budget_exhausted = false;

        'traversal: while let Some(Reverse(curr)) = c.pop() {
            if eval_limit_reached {
                break 'traversal;
            }

            h += 1;

            if h > h_max && w_match.len() < k {
                hop_budget_exhausted = true;
                break 'traversal;
            }

            // Pruning check against worst matching candidate
            if w_match.len() >= effective_ef {
                let furthest_match = *w_match.peek().expect("w_match not empty when len >= ef");
                if curr.distance > furthest_match.distance {
                    break 'traversal;
                }
            }

            for &neighbor_u32 in self.neighbors(0, curr.id) {
                let neighbor_id = neighbor_u32 as usize;
                if !tracker.is_visited(neighbor_id) {
                    if let Some(limit) = params.eval_limit {
                        if predicate.bitset().is_none() && eval_count >= limit {
                            eval_limit_reached = true;
                            break 'traversal;
                        }
                    }

                    tracker.mark_visited(neighbor_id);

                    let d_neighbor = self.compute_distance(query, neighbor_id);
                    #[cfg(feature = "instrument")]
                    {
                        dist_evals += 1;
                    }
                    let cand_neighbor = Candidate {
                        distance: d_neighbor,
                        id: neighbor_id,
                    };

                    let is_match = {
                        if predicate.bitset().is_none() {
                            eval_count += 1;
                        }
                        predicate.matches(neighbor_id as u32)
                    };
                    if COLLECT_COUNTERS {
                        record_layer0_match_counter(&mut counters, is_match);
                    }

                    let mut admitted = false;
                    if is_match
                        && (w_match.len() < effective_ef
                            || cand_neighbor < *w_match.peek().expect("w_match not empty"))
                    {
                        w_match.push(cand_neighbor);
                        h = 0;
                        if w_match.len() > effective_ef {
                            w_match.pop();
                        }
                        admitted = true;
                    }

                    // Always allow promising nodes into exploration frontier C
                    if admitted
                        || w_match.len() < effective_ef
                        || cand_neighbor < *w_match.peek().expect("w_match not empty")
                    {
                        c.push(Reverse(cand_neighbor));
                    }
                }
            }
        }

        #[cfg(feature = "instrument")]
        {
            tracker.distance_evals += dist_evals;
        }

        (hop_budget_exhausted, eval_limit_reached)
    }

    /// Executes Strategy 2 in-traversal predicate filtered approximate nearest neighbor search
    /// using bridge nodes and hop budget (Algorithm 3).
    pub fn search_knn_filtered_traversal<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
    ) -> Result<FilteredSearchOutput> {
        let mut tracker = VisitedTracker::new(self.node_meta.len());
        self.search_knn_filtered_traversal_with_tracker(query, k, predicate, params, &mut tracker)
    }

    /// Read-only Strategy 2 in-traversal search reusing an externally provided [`VisitedTracker`].
    pub fn search_knn_filtered_traversal_with_tracker<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        tracker: &mut VisitedTracker,
    ) -> Result<FilteredSearchOutput> {
        self.search_knn_filtered_traversal_core::<P, false>(
            query, k, predicate, params, tracker, None, None,
        )
    }

    /// Read-only Strategy 2 in-traversal search reusing preallocated [`FilteredSearchScratch`].
    pub fn search_knn_filtered_traversal_with_scratch<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        scratch: &mut FilteredSearchScratch,
    ) -> Result<FilteredSearchOutput> {
        let FilteredSearchScratch {
            tracker,
            frontier,
            matched_results,
            raw_candidates,
            ..
        } = scratch;
        self.search_knn_filtered_traversal_core::<P, false>(
            query,
            k,
            predicate,
            params,
            tracker,
            Some((frontier, matched_results, raw_candidates)),
            None,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn search_knn_filtered_traversal_core<P: FilterPredicate, const COLLECT_COUNTERS: bool>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        tracker: &mut VisitedTracker,
        scratch_bufs: Option<(
            &mut BinaryHeap<Reverse<Candidate>>,
            &mut BinaryHeap<Candidate>,
            &mut Vec<Candidate>,
        )>,
        mut counters: Option<&mut FilteredSearchCounters>,
    ) -> Result<FilteredSearchOutput> {
        self.validate_query_slice(query, query.len(), k)?;

        if self.node_meta.is_empty() || self.enter_point.is_none() {
            return Ok(FilteredSearchOutput {
                results: Vec::new(),
                flags: FilteredSearchFlags::default(),
            });
        }

        let effective_ef = params.ef_search.max(k).min(self.node_meta.len());
        let h_max = params
            .max_bridge_hops
            .unwrap_or_else(|| effective_ef.saturating_mul(3));

        if let Some((frontier, matched_results, raw_candidates)) = scratch_bufs.as_ref() {
            if frontier.capacity() < self.node_meta.len()
                || matched_results.capacity() <= effective_ef
                || raw_candidates.capacity() < effective_ef
            {
                return Err(VectorSearchError::InvalidArgument(
                    "FilteredSearchScratch capacity is insufficient for Strategy 2 query parameters"
                        .to_string(),
                ));
            }
        }

        let cap = effective_ef.saturating_add(self.config.effective_m_max0());
        let mut local_c = BinaryHeap::<Reverse<Candidate>>::new();
        let mut local_w = BinaryHeap::<Candidate>::new();
        let mut local_raw = Vec::<Candidate>::new();
        let using_scratch = scratch_bufs.is_some();
        let (frontier, matched_results, raw_candidates) = match scratch_bufs {
            Some((frontier, matched_results, raw_candidates)) => {
                (frontier, matched_results, raw_candidates)
            }
            None => {
                local_c.reserve(cap);
                local_w.reserve(cap);
                (&mut local_c, &mut local_w, &mut local_raw)
            }
        };

        let ep = self.enter_point.expect("enter_point checked non-empty");
        let pred_bitset = predicate.bitset();

        // Phase 1: Top-down greedy zoom-in (ef = 1) down to layer 1 (unfiltered)
        let curr_ep = self.search_top_layers_greedy::<COLLECT_COUNTERS>(
            query,
            ep,
            tracker,
            frontier,
            matched_results,
            pred_bitset,
            counters.as_deref_mut(),
        );

        // Phase 2: Base layer in-traversal search with bridge-nodes and hop budget
        let (hop_budget_exhausted, eval_limit_reached) = self
            .search_layer0_in_traversal_inner::<P, COLLECT_COUNTERS>(
                query,
                curr_ep,
                k,
                effective_ef,
                h_max,
                predicate,
                params,
                tracker,
                frontier,
                matched_results,
                counters.as_deref_mut(),
            );

        drain_heap_asc(matched_results, raw_candidates);
        let results = self.format_candidates(raw_candidates, k);

        if (hop_budget_exhausted || eval_limit_reached)
            && results.len() < k
            && params.allow_bruteforce_fallback
            && pred_bitset.is_some()
        {
            let mut fallback_out = self.search_knn_filtered_pre_inner::<P, COLLECT_COUNTERS>(
                query,
                k,
                predicate,
                tracker,
                if using_scratch {
                    Some((matched_results, raw_candidates))
                } else {
                    None
                },
                counters,
            )?;
            fallback_out.flags.hop_budget_exhausted = hop_budget_exhausted;
            fallback_out.flags.eval_limit_reached = eval_limit_reached;
            fallback_out.flags.used_bruteforce_fallback = true;
            return Ok(fallback_out);
        }

        Ok(FilteredSearchOutput {
            results,
            flags: FilteredSearchFlags {
                post_filter_cap_reached: false,
                hop_budget_exhausted,
                eval_limit_reached,
                used_bruteforce_fallback: false,
            },
        })
    }

    /// Batch sequential query execution for Strategy 2 in-traversal filtering.
    pub fn search_batch_filtered_traversal<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
    ) -> Result<Vec<FilteredSearchOutput>> {
        self.run_filtered_batch_seq(queries, |q_row, tracker| {
            self.search_knn_filtered_traversal_with_tracker(q_row, k, predicate, params, tracker)
        })
    }

    /// Multi-threaded batch search for Strategy 2 in-traversal filtering.
    pub fn search_batch_filtered_traversal_parallel<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        num_threads: usize,
    ) -> Result<Vec<FilteredSearchOutput>> {
        self.run_filtered_batch_parallel(queries, k, num_threads, None, |q_row, tracker| {
            self.search_knn_filtered_traversal_with_tracker(q_row, k, predicate, params, tracker)
        })
    }

    // =========================================================================
    // Milestone M3 Task T3.6: Filtered Search (Strategy 3 Pre-Filter Brute-Force Scan)
    // =========================================================================

    /// Single-query execution for Strategy 3: Pre-filter brute-force flat scan (Algorithm 4).
    ///
    /// Requires a bitset-backed predicate (`predicate.bitset().is_some()`). Bypasses the HNSW
    /// graph traversal entirely, iterating over all active (1) bit IDs in the bitset and
    /// computing exact vector distances using the compiler auto-vectorizable kernel. Bounded by a max-heap of
    /// size `k` with canonical `(distance, id)` tie-breaking.
    pub fn search_knn_filtered_pre<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
    ) -> Result<FilteredSearchOutput> {
        let mut tracker = VisitedTracker::new(self.node_meta.len());
        self.search_knn_filtered_pre_with_tracker(query, k, predicate, params, &mut tracker)
    }

    /// Read-only Strategy 3 pre-filter brute-force scan reusing an externally provided [`VisitedTracker`].
    pub fn search_knn_filtered_pre_with_tracker<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        _params: &FilteredSearchParams,
        #[allow(unused_variables)] tracker: &mut VisitedTracker,
    ) -> Result<FilteredSearchOutput> {
        self.search_knn_filtered_pre_inner::<P, false>(query, k, predicate, tracker, None, None)
    }

    /// Read-only Strategy 3 pre-filter brute-force scan reusing preallocated [`FilteredSearchScratch`].
    pub fn search_knn_filtered_pre_with_scratch<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        _params: &FilteredSearchParams,
        scratch: &mut FilteredSearchScratch,
    ) -> Result<FilteredSearchOutput> {
        let FilteredSearchScratch {
            tracker,
            matched_results,
            raw_candidates,
            ..
        } = scratch;
        self.search_knn_filtered_pre_inner::<P, false>(
            query,
            k,
            predicate,
            tracker,
            Some((matched_results, raw_candidates)),
            None,
        )
    }

    fn search_knn_filtered_pre_inner<P: FilterPredicate, const COLLECT_COUNTERS: bool>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
        #[allow(unused_variables)] tracker: &mut VisitedTracker,
        scratch_bufs: Option<(&mut BinaryHeap<Candidate>, &mut Vec<Candidate>)>,
        counters: Option<&mut FilteredSearchCounters>,
    ) -> Result<FilteredSearchOutput> {
        let _ = tracker;
        self.validate_query_slice(query, query.len(), k)?;

        let bitset = predicate.bitset().ok_or_else(|| {
            VectorSearchError::InvalidArgument(
                "Strategy 3 (PreFilterBruteForce) requires a bitset-backed predicate".to_string(),
            )
        })?;

        if self.node_meta.is_empty() || self.matrix.rows() == 0 {
            return Ok(FilteredSearchOutput {
                results: Vec::new(),
                flags: FilteredSearchFlags::default(),
            });
        }

        if let Some((matched_results, raw_candidates)) = scratch_bufs.as_ref() {
            if matched_results.capacity() < k || raw_candidates.capacity() < k {
                return Err(VectorSearchError::InvalidArgument(
                    "FilteredSearchScratch capacity is insufficient for Strategy 3 query k"
                        .to_string(),
                ));
            }
        }

        let num_base = self.matrix.rows().min(self.node_meta.len());
        let results = if let Some((matched_results, raw_candidates)) = scratch_bufs {
            matched_results.clear();
            self.fill_pre_filter_heap_inner::<COLLECT_COUNTERS>(
                query,
                k,
                num_base,
                bitset,
                tracker,
                matched_results,
                counters,
            );
            drain_heap_asc(matched_results, raw_candidates);
            self.format_candidates(raw_candidates, raw_candidates.len())
        } else {
            let mut heap = BinaryHeap::<Candidate>::with_capacity(k.min(num_base));
            self.fill_pre_filter_heap_inner::<COLLECT_COUNTERS>(
                query, k, num_base, bitset, tracker, &mut heap, counters,
            );
            let sorted = heap.into_sorted_vec();
            self.format_candidates(&sorted, sorted.len())
        };

        Ok(FilteredSearchOutput {
            results,
            flags: FilteredSearchFlags {
                post_filter_cap_reached: false,
                hop_budget_exhausted: false,
                eval_limit_reached: false,
                used_bruteforce_fallback: false,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_pre_filter_heap_inner<const COLLECT_COUNTERS: bool>(
        &self,
        query: &[f32],
        k: usize,
        num_base: usize,
        bitset: &DenseBitset,
        #[allow(unused_variables)] tracker: &mut VisitedTracker,
        heap: &mut BinaryHeap<Candidate>,
        mut counters: Option<&mut FilteredSearchCounters>,
    ) {
        let _ = tracker;
        for id in bitset.iter_ones() {
            if id >= num_base {
                break;
            }
            #[cfg(feature = "instrument")]
            {
                tracker.distance_evals += 1;
            }
            let dist = self.compute_distance(query, id);
            if COLLECT_COUNTERS {
                record_layer0_match_counter(&mut counters, true);
            }
            let cand = Candidate { distance: dist, id };
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(mut worst) = heap.peek_mut() {
                if cand < *worst {
                    *worst = cand;
                }
            }
        }
    }

    /// Batch sequential query execution for Strategy 3 pre-filter brute force.
    pub fn search_batch_filtered_pre<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
    ) -> Result<Vec<FilteredSearchOutput>> {
        self.run_filtered_batch_seq(queries, |q_row, tracker| {
            self.search_knn_filtered_pre_with_tracker(q_row, k, predicate, params, tracker)
        })
    }

    /// Multi-threaded batch search for Strategy 3 pre-filter brute force.
    pub fn search_batch_filtered_pre_parallel<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        params: &FilteredSearchParams,
        num_threads: usize,
    ) -> Result<Vec<FilteredSearchOutput>> {
        self.run_filtered_batch_parallel(
            queries,
            k,
            num_threads,
            Some(predicate),
            |q_row, tracker| {
                self.search_knn_filtered_pre_with_tracker(q_row, k, predicate, params, tracker)
            },
        )
    }

    fn run_filtered_batch_seq<F>(
        &self,
        queries: &RowMatrix<f32>,
        mut per_query: F,
    ) -> Result<Vec<FilteredSearchOutput>>
    where
        F: FnMut(&[f32], &mut VisitedTracker) -> Result<FilteredSearchOutput>,
    {
        let mut tracker = VisitedTracker::new(self.node_meta.len());
        let mut outputs = Vec::with_capacity(queries.rows());
        for q_row in queries.iter_rows() {
            outputs.push(per_query(q_row, &mut tracker)?);
        }
        Ok(outputs)
    }

    fn run_filtered_batch_parallel<F>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        num_threads: usize,
        require_bitset: Option<&dyn FilterPredicate>,
        per_query: F,
    ) -> Result<Vec<FilteredSearchOutput>>
    where
        F: Fn(&[f32], &mut VisitedTracker) -> Result<FilteredSearchOutput> + Sync,
    {
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(Vec::new());
        }

        self.validate_query_slice(queries.as_slice(), queries.cols(), k)?;

        if let Some(pred) = require_bitset {
            if pred.bitset().is_none() {
                return Err(VectorSearchError::InvalidArgument(
                    "Strategy 3 (PreFilterBruteForce) requires a bitset-backed predicate"
                        .to_string(),
                ));
            }
        }

        if self.node_meta.is_empty() || self.enter_point.is_none() || self.matrix.rows() == 0 {
            return Ok(vec![
                FilteredSearchOutput {
                    results: Vec::new(),
                    flags: FilteredSearchFlags::default(),
                };
                num_queries
            ]);
        }

        let workers = resolve_worker_count(num_threads, num_queries);
        if workers == 1 {
            return self.run_filtered_batch_seq(queries, per_query);
        }

        let mut outputs: Vec<Option<FilteredSearchOutput>> =
            (0..num_queries).map(|_| None).collect();
        let chunk_size = num_queries.div_ceil(workers);

        let per_query = &per_query;
        std::thread::scope(|s| {
            let mut out_rest = outputs.as_mut_slice();
            let mut q_offset = 0;

            while q_offset < num_queries {
                let take_q = chunk_size.min(num_queries - q_offset);
                let (out_slice, next_out) = out_rest.split_at_mut(take_q);
                out_rest = next_out;

                let start_q = q_offset;
                q_offset += take_q;

                s.spawn(move || {
                    let mut tracker = VisitedTracker::new(self.node_meta.len());
                    for (local_i, out_slot) in out_slice.iter_mut().enumerate() {
                        let q_row = queries.row(start_q + local_i);
                        let res = per_query(q_row, &mut tracker).expect("query valid");
                        *out_slot = Some(res);
                    }
                });
            }
        });

        Ok(outputs
            .into_iter()
            .map(|opt| opt.expect("all queries processed"))
            .collect())
    }
}
