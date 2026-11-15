//! From-scratch Brute-Force Exact K-Nearest Neighbor (KNN) search engine.
//!
//! Computes exact pairwise distances with non-negative clamping and canonical
//! `(distance, id)` deterministic tie-breaking, serving as the authoritative
//! ground truth for all HNSW recall measurements.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread;
use std::time::Instant;

use crate::distance::{inner_product, l2_squared, Metric};
use crate::error::{Result, VectorSearchError};
use crate::filter::FilterPredicate;
use crate::matrix::RowMatrix;
use crate::recall::{compute_latency_stats, LatencyStats};

/// Exact top-`k` search output across a batch of queries.
#[derive(Debug, Clone)]
pub struct KnnSearchResult {
    /// 0-indexed database neighbor IDs of shape `(num_queries, effective_k)`.
    pub indices: RowMatrix<i32>,
    /// Exact computed distances/scores of shape `(num_queries, effective_k)`.
    pub distances: RowMatrix<f32>,
}

/// Exact top-`k` search output paired with per-query latency distribution statistics.
#[derive(Debug, Clone)]
pub struct TimedKnnSearchResult {
    pub result: KnnSearchResult,
    pub per_query_latency_ms: Vec<f64>,
    pub latency: LatencyStats,
}

#[derive(Debug, Clone, Copy)]
struct HeapCandidate {
    /// Minimization key (`d^2` for L2/Euclidean, `-dot` for InnerProduct).
    sort_key: f32,
    /// 0-indexed database vector ID.
    id: i32,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key.total_cmp(&other.sort_key) == Ordering::Equal && self.id == other.id
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key
            .total_cmp(&other.sort_key)
            .then_with(|| self.id.cmp(&other.id))
    }
}

fn validate_base_matrix(base_vectors: &RowMatrix<f32>) -> Result<()> {
    if base_vectors.rows() == 0 || base_vectors.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "base_vectors cannot be empty (rows and cols must be > 0)".to_string(),
        ));
    }
    if i32::try_from(base_vectors.rows()).is_err() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors row count {} exceeds i32::MAX",
            base_vectors.rows()
        )));
    }
    if let Some(pos) = base_vectors.as_slice().iter().position(|v| !v.is_finite()) {
        let row = pos / base_vectors.cols();
        let col = pos % base_vectors.cols();
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors contains non-finite value (NaN/Inf) at ({row}, {col})"
        )));
    }
    Ok(())
}

fn validate_query_batch(base: &RowMatrix<f32>, queries: &RowMatrix<f32>, k: usize) -> Result<()> {
    if k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0)".to_string(),
        ));
    }
    if queries.rows() > 0 && queries.cols() != base.cols() {
        return Err(VectorSearchError::DimensionMismatch {
            expected: base.cols(),
            actual: queries.cols(),
        });
    }
    if let Some(pos) = queries.as_slice().iter().position(|v| !v.is_finite()) {
        let cols = queries.cols().max(1);
        let row = pos / cols;
        let col = pos % cols;
        return Err(VectorSearchError::InvalidArgument(format!(
            "queries contains non-finite value (NaN/Inf) at ({row}, {col})"
        )));
    }
    Ok(())
}

#[inline]
fn compute_metric_sort_key(metric: Metric, query: &[f32], base_row: &[f32]) -> f32 {
    match metric {
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
fn format_metric_output_distance(metric: Metric, sort_key: f32) -> f32 {
    match metric {
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

/// Brute-force exact K-Nearest Neighbor search engine.
#[derive(Debug, Clone)]
pub struct ExactKnn {
    metric: Metric,
    base_vectors: Option<RowMatrix<f32>>,
}

impl ExactKnn {
    /// Creates an unfitted `ExactKnn` engine for the specified distance `metric`.
    #[must_use]
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            base_vectors: None,
        }
    }

    /// Creates and fits an `ExactKnn` engine directly on `base_vectors`.
    pub fn from_base(metric: Metric, base_vectors: RowMatrix<f32>) -> Result<Self> {
        let mut engine = Self::new(metric);
        engine.fit(base_vectors)?;
        Ok(engine)
    }

    /// Configured distance metric.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of indexed base vectors (`0` if unfitted).
    #[inline]
    #[must_use]
    pub fn num_base(&self) -> usize {
        self.base_vectors.as_ref().map_or(0, RowMatrix::rows)
    }

    /// Vector dimensionality (`0` if unfitted).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.base_vectors.as_ref().map_or(0, RowMatrix::cols)
    }

    /// Reference to the indexed base vector matrix, if fitted.
    #[inline]
    #[must_use]
    pub fn base_vectors(&self) -> Option<&RowMatrix<f32>> {
        self.base_vectors.as_ref()
    }

    /// Indexes `base_vectors` after validating non-empty dimensions, `i32` ID bounds, and finiteness.
    pub fn fit(&mut self, base_vectors: RowMatrix<f32>) -> Result<&mut Self> {
        validate_base_matrix(&base_vectors)?;

        self.base_vectors = Some(base_vectors);
        Ok(self)
    }

    /// Searches exact top-`k` nearest neighbors for a single 1D query slice `query`.
    ///
    /// Returns `(indices, distances)` ordered ascending by `(distance, id)` (or descending
    /// by `(dot_product, -id)` for `Metric::InnerProduct`).
    pub fn search_one(&self, query: &[f32], k: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        let base = self
            .base_vectors
            .as_ref()
            .ok_or(VectorSearchError::UnfittedIndex(
                "ExactKnn::fit() must be called with base_vectors before search",
            ))?;

        if query.len() != base.cols() {
            return Err(VectorSearchError::DimensionMismatch {
                expected: base.cols(),
                actual: query.len(),
            });
        }
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

        let effective_k = k.min(base.rows());
        let mut heap = BinaryHeap::with_capacity(effective_k);
        self.scan_into_heap(base, query, effective_k, &mut heap);
        Ok(self.finalize_heap(heap))
    }

    /// Searches exact top-`k` nearest neighbors sequentially across all rows in `queries`.
    pub fn search(&self, queries: &RowMatrix<f32>, k: usize) -> Result<KnnSearchResult> {
        let base = self.validate_batch_inputs(queries, k)?;
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        let effective_k = k.min(base.rows());
        let mut all_indices = Vec::with_capacity(num_queries * effective_k);
        let mut all_distances = Vec::with_capacity(num_queries * effective_k);
        let mut heap = BinaryHeap::with_capacity(effective_k);

        for q_row in queries.iter_rows() {
            heap.clear();
            self.scan_into_heap(base, q_row, effective_k, &mut heap);
            let sorted = heap.clone().into_sorted_vec();
            for cand in sorted {
                all_indices.push(cand.id);
                all_distances.push(self.format_output_distance(cand.sort_key));
            }
        }

        Ok(KnnSearchResult {
            indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
            distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
        })
    }

    /// Searches exact top-`k` neighbors sequentially while recording individual per-query
    /// latency (`p50`, `p95`, `p99`, `mean`, `QPS`).
    ///
    /// When `warmup == true` and `queries.rows() > 0`, executes 1 untimed warm-up query
    /// first to warm instruction and vector data caches before timed measurement.
    pub fn search_timed(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        warmup: bool,
    ) -> Result<TimedKnnSearchResult> {
        let base = self.validate_batch_inputs(queries, k)?;
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

        let effective_k = k.min(base.rows());
        let mut heap = BinaryHeap::with_capacity(effective_k);

        if warmup {
            heap.clear();
            self.scan_into_heap(base, queries.row(0), effective_k, &mut heap);
        }

        let mut all_indices = Vec::with_capacity(num_queries * effective_k);
        let mut all_distances = Vec::with_capacity(num_queries * effective_k);
        let mut latencies_ms = Vec::with_capacity(num_queries);

        let batch_start = Instant::now();
        for q_row in queries.iter_rows() {
            let q_start = Instant::now();
            heap.clear();
            self.scan_into_heap(base, q_row, effective_k, &mut heap);
            let sorted = std::mem::replace(&mut heap, BinaryHeap::with_capacity(effective_k))
                .into_sorted_vec();
            let q_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            latencies_ms.push(q_ms);

            for cand in sorted {
                all_indices.push(cand.id);
                all_distances.push(self.format_output_distance(cand.sort_key));
            }
        }
        let total_elapsed_s = batch_start.elapsed().as_secs_f64();
        let latency = compute_latency_stats(&latencies_ms, total_elapsed_s);

        Ok(TimedKnnSearchResult {
            result: KnnSearchResult {
                indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
                distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
            },
            per_query_latency_ms: latencies_ms,
            latency,
        })
    }

    /// Multi-threaded batch ground-truth computation using `std::thread::scope`.
    ///
    /// Produces bit-for-bit identical `(indices, distances)` to single-threaded `search()`
    /// while scaling across CPU cores for large datasets (e.g., 1M vectors in Phase 3).
    pub fn search_parallel(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        num_threads: usize,
    ) -> Result<KnnSearchResult> {
        let base = self.validate_batch_inputs(queries, k)?;
        Self::search_parallel_validated(self.metric, base, queries, k, num_threads)
    }

    /// Multi-threaded batch ground-truth computation directly over a borrowed `base` matrix
    /// without cloning `base` into an `ExactKnn` instance.
    pub fn search_parallel_with_base(
        metric: Metric,
        base: &RowMatrix<f32>,
        queries: &RowMatrix<f32>,
        k: usize,
        num_threads: usize,
    ) -> Result<KnnSearchResult> {
        validate_base_matrix(base)?;
        validate_query_batch(base, queries, k)?;

        Self::search_parallel_validated(metric, base, queries, k, num_threads)
    }

    fn search_parallel_validated(
        metric: Metric,
        base: &RowMatrix<f32>,
        queries: &RowMatrix<f32>,
        k: usize,
        num_threads: usize,
    ) -> Result<KnnSearchResult> {
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        let engine = Self::new(metric);
        let effective_k = k.min(base.rows());
        let workers = if num_threads == 0 {
            thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        } else {
            num_threads
        }
        .min(num_queries)
        .max(1);

        if workers == 1 {
            let mut all_indices = Vec::with_capacity(num_queries * effective_k);
            let mut all_distances = Vec::with_capacity(num_queries * effective_k);
            let mut heap = BinaryHeap::with_capacity(effective_k);

            for q_row in queries.iter_rows() {
                heap.clear();
                engine.scan_into_heap(base, q_row, effective_k, &mut heap);
                let sorted = std::mem::replace(&mut heap, BinaryHeap::with_capacity(effective_k))
                    .into_sorted_vec();
                for cand in sorted {
                    all_indices.push(cand.id);
                    all_distances.push(engine.format_output_distance(cand.sort_key));
                }
            }

            return Ok(KnnSearchResult {
                indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
                distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
            });
        }

        let mut all_indices = vec![0_i32; num_queries * effective_k];
        let mut all_distances = vec![0.0_f32; num_queries * effective_k];

        let chunk_queries = num_queries.div_ceil(workers);
        let idx_chunk_len = chunk_queries * effective_k;

        thread::scope(|s| {
            let mut idx_rest = all_indices.as_mut_slice();
            let mut dist_rest = all_distances.as_mut_slice();
            let mut q_offset = 0;

            while q_offset < num_queries {
                let take_q = chunk_queries.min(num_queries - q_offset);
                let take_elems = take_q * effective_k;
                let (idx_slice, next_idx) = idx_rest.split_at_mut(take_elems.min(idx_chunk_len));
                let (dist_slice, next_dist) = dist_rest.split_at_mut(take_elems.min(idx_chunk_len));
                idx_rest = next_idx;
                dist_rest = next_dist;

                let start_q = q_offset;
                q_offset += take_q;
                let worker_engine = engine.clone();

                s.spawn(move || {
                    let mut heap = BinaryHeap::with_capacity(effective_k);
                    for local_i in 0..take_q {
                        let q_row = queries.row(start_q + local_i);
                        heap.clear();
                        worker_engine.scan_into_heap(base, q_row, effective_k, &mut heap);
                        let sorted =
                            std::mem::replace(&mut heap, BinaryHeap::with_capacity(effective_k))
                                .into_sorted_vec();

                        let out_base = local_i * effective_k;
                        for (j, cand) in sorted.into_iter().enumerate() {
                            idx_slice[out_base + j] = cand.id;
                            dist_slice[out_base + j] =
                                worker_engine.format_output_distance(cand.sort_key);
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

    fn validate_batch_inputs<'a>(
        &'a self,
        queries: &RowMatrix<f32>,
        k: usize,
    ) -> Result<&'a RowMatrix<f32>> {
        let base = self
            .base_vectors
            .as_ref()
            .ok_or(VectorSearchError::UnfittedIndex(
                "ExactKnn::fit() must be called with base_vectors before search",
            ))?;

        validate_query_batch(base, queries, k)?;

        Ok(base)
    }

    #[inline]
    fn compute_sort_key(&self, query: &[f32], base_row: &[f32]) -> f32 {
        compute_metric_sort_key(self.metric, query, base_row)
    }

    #[inline]
    fn format_output_distance(&self, sort_key: f32) -> f32 {
        format_metric_output_distance(self.metric, sort_key)
    }

    #[inline]
    fn scan_into_heap(
        &self,
        base: &RowMatrix<f32>,
        query: &[f32],
        effective_k: usize,
        heap: &mut BinaryHeap<HeapCandidate>,
    ) {
        let num_base = base.rows();
        let dim = base.cols();
        let flat = base.as_slice();

        for idx in 0..effective_k {
            let row = &flat[idx * dim..(idx + 1) * dim];
            let cand = HeapCandidate {
                sort_key: self.compute_sort_key(query, row),
                id: idx as i32,
            };
            heap.push(cand);
        }

        for idx in effective_k..num_base {
            let row = &flat[idx * dim..(idx + 1) * dim];
            let cand = HeapCandidate {
                sort_key: self.compute_sort_key(query, row),
                id: idx as i32,
            };
            if let Some(&worst) = heap.peek() {
                if cand < worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }
    }

    #[inline]
    fn finalize_heap(&self, heap: BinaryHeap<HeapCandidate>) -> (Vec<i32>, Vec<f32>) {
        let sorted = heap.into_sorted_vec();
        let mut indices = Vec::with_capacity(sorted.len());
        let mut distances = Vec::with_capacity(sorted.len());
        for cand in sorted {
            indices.push(cand.id);
            distances.push(self.format_output_distance(cand.sort_key));
        }
        (indices, distances)
    }
}

/// Brute-force exact K-Nearest Neighbor search engine with predicate filtering.
///
/// Computes bit-exact Top-`k` nearest neighbors satisfying a [`FilterPredicate`]
/// with canonical `(distance, id)` deterministic tie-breaking.
#[derive(Debug, Clone)]
pub struct FilteredExactKnn {
    metric: Metric,
    base_vectors: RowMatrix<f32>,
}

impl FilteredExactKnn {
    /// Creates a new `FilteredExactKnn` engine on `base_vectors` with `metric`.
    ///
    /// Validates non-empty matrix dimensions, `i32::MAX` row bound, and vector finiteness.
    pub fn new(metric: Metric, base_vectors: RowMatrix<f32>) -> Result<Self> {
        validate_base_matrix(&base_vectors)?;
        Ok(Self {
            metric,
            base_vectors,
        })
    }

    /// Creates a `FilteredExactKnn` engine directly from `base_vectors` (alias to [`FilteredExactKnn::new`]).
    pub fn from_base(metric: Metric, base_vectors: RowMatrix<f32>) -> Result<Self> {
        Self::new(metric, base_vectors)
    }

    /// Configured distance metric.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of indexed base vectors.
    #[inline]
    #[must_use]
    pub fn num_base(&self) -> usize {
        self.base_vectors.rows()
    }

    /// Vector dimensionality.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.base_vectors.cols()
    }

    /// Reference to the indexed base vector matrix.
    #[inline]
    #[must_use]
    pub fn base_vectors(&self) -> &RowMatrix<f32> {
        &self.base_vectors
    }

    /// Computes bit-exact Top-`k` nearest neighbors satisfying `predicate`.
    ///
    /// # Tie-Breaking
    /// Distance comparisons break ties canonically on `(distance, id)`:
    /// `(d_a, id_a) < (d_b, id_b)` iff `d_a < d_b` or `(d_a == d_b && id_a < id_b)`.
    ///
    /// # Edge Cases
    /// - If `predicate` matches fewer than `k` vectors, returns all matching vectors sorted ascending.
    /// - If 0 vectors match or `k == 0`, returns `Ok(vec![])`.
    /// - If `query` contains non-finite values (NaN / Inf), returns [`VectorSearchError::InvalidArgument`].
    /// - If `query.len() != dim`, returns [`VectorSearchError::DimensionMismatch`].
    pub fn search_filtered<P: FilterPredicate>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &P,
    ) -> Result<Vec<(f32, usize)>> {
        if query.len() != self.base_vectors.cols() {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.base_vectors.cols(),
                actual: query.len(),
            });
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(
                "query vector contains non-finite value (NaN/Inf)".to_string(),
            ));
        }
        if k == 0 || self.base_vectors.rows() == 0 {
            return Ok(Vec::new());
        }

        let num_base = self.base_vectors.rows();
        let mut heap = BinaryHeap::<HeapCandidate>::with_capacity(k);

        let push_id = |id: usize, heap: &mut BinaryHeap<HeapCandidate>| {
            let row = self.base_vectors.row(id);
            let cand = HeapCandidate {
                sort_key: self.compute_sort_key(query, row),
                id: id as i32,
            };
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(&worst) = heap.peek() {
                if cand < worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        };

        if let Some(b) = predicate.bitset() {
            for id in b.iter_ones() {
                if id >= num_base {
                    break;
                }
                push_id(id, &mut heap);
            }
        } else {
            for id in 0..num_base {
                if predicate.matches(id as u32) {
                    push_id(id, &mut heap);
                }
            }
        }

        let sorted = heap.into_sorted_vec();
        let mut results = Vec::with_capacity(sorted.len());
        for cand in sorted {
            results.push((self.format_output_distance(cand.sort_key), cand.id as usize));
        }
        Ok(results)
    }

    /// Searches exact filtered top-`k` nearest neighbors sequentially across all query rows in `queries`.
    pub fn search_batch<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
    ) -> Result<KnnSearchResult> {
        self.validate_batch_inputs(queries, k)?;
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        let num_base = self.base_vectors.rows();
        let matching_count = if let Some(count) = predicate.count_matching(num_base) {
            count
        } else {
            (0..num_base)
                .filter(|&id| predicate.matches(id as u32))
                .count()
        };

        let effective_k = k.min(matching_count);
        if effective_k == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        let mut all_indices = Vec::with_capacity(num_queries * effective_k);
        let mut all_distances = Vec::with_capacity(num_queries * effective_k);

        for q_row in queries.iter_rows() {
            let hits = self.search_filtered(q_row, effective_k, predicate)?;
            for (dist, id) in hits {
                all_indices.push(id as i32);
                all_distances.push(dist);
            }
        }

        Ok(KnnSearchResult {
            indices: RowMatrix::new(num_queries, effective_k, all_indices)?,
            distances: RowMatrix::new(num_queries, effective_k, all_distances)?,
        })
    }

    /// Multi-threaded batch filtered search using `std::thread::scope`.
    ///
    /// Produces bit-for-bit identical results to single-threaded `search_batch()`.
    pub fn search_batch_parallel<P: FilterPredicate>(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        predicate: &P,
        num_threads: usize,
    ) -> Result<KnnSearchResult> {
        self.validate_batch_inputs(queries, k)?;
        let num_queries = queries.rows();
        if num_queries == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        let num_base = self.base_vectors.rows();
        let matching_count = if let Some(count) = predicate.count_matching(num_base) {
            count
        } else {
            (0..num_base)
                .filter(|&id| predicate.matches(id as u32))
                .count()
        };

        let effective_k = k.min(matching_count);
        if effective_k == 0 {
            return Ok(KnnSearchResult {
                indices: RowMatrix::empty(),
                distances: RowMatrix::empty(),
            });
        }

        let workers = if num_threads == 0 {
            thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        } else {
            num_threads
        }
        .min(num_queries)
        .max(1);

        if workers == 1 {
            return self.search_batch(queries, k, predicate);
        }

        let mut all_indices = vec![0_i32; num_queries * effective_k];
        let mut all_distances = vec![0.0_f32; num_queries * effective_k];

        let chunk_queries = num_queries.div_ceil(workers);
        let idx_chunk_len = chunk_queries * effective_k;

        thread::scope(|s| {
            let mut idx_rest = all_indices.as_mut_slice();
            let mut dist_rest = all_distances.as_mut_slice();
            let mut q_offset = 0;

            while q_offset < num_queries {
                let take_q = chunk_queries.min(num_queries - q_offset);
                let take_elems = take_q * effective_k;
                let (idx_slice, next_idx) = idx_rest.split_at_mut(take_elems.min(idx_chunk_len));
                let (dist_slice, next_dist) = dist_rest.split_at_mut(take_elems.min(idx_chunk_len));
                idx_rest = next_idx;
                dist_rest = next_dist;

                let start_q = q_offset;
                q_offset += take_q;

                s.spawn(move || {
                    for local_i in 0..take_q {
                        let q_row = queries.row(start_q + local_i);
                        let hits = self
                            .search_filtered(q_row, effective_k, predicate)
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

    fn validate_batch_inputs(&self, queries: &RowMatrix<f32>, k: usize) -> Result<()> {
        validate_query_batch(&self.base_vectors, queries, k)
    }

    #[inline]
    fn compute_sort_key(&self, query: &[f32], base_row: &[f32]) -> f32 {
        compute_metric_sort_key(self.metric, query, base_row)
    }

    #[inline]
    fn format_output_distance(&self, sort_key: f32) -> f32 {
        format_metric_output_distance(self.metric, sort_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{DenseBitset, LambdaPredicate};
    use crate::recall::compute_recall_at_k;

    #[test]
    fn test_filtered_exact_knn_basic_and_self_parity() {
        let base_data = vec![
            1.0, 0.0, // id 0
            0.0, 2.0, // id 1
            3.0, 0.0, // id 2
            0.0, 4.0, // id 3
            5.0, 0.0, // id 4
        ];
        let base = RowMatrix::new(5, 2, base_data).unwrap();
        let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();
        assert_eq!(engine.num_base(), 5);
        assert_eq!(engine.dim(), 2);
        assert_eq!(engine.metric(), Metric::L2Squared);

        // Filter: only odd IDs (1, 3)
        let mut bitset = DenseBitset::zeros(5);
        bitset.set_bit(1, true);
        bitset.set_bit(3, true);

        let query = vec![0.0_f32, 0.0];
        let hits = engine.search_filtered(&query, 2, &bitset).unwrap();

        // id 1: d^2 = 4.0; id 3: d^2 = 16.0
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], (4.0, 1));
        assert_eq!(hits[1], (16.0, 3));

        // Self-parity check Recall@k(GT, GT) == 1.000000
        let q_mat = RowMatrix::new(1, 2, query).unwrap();
        let batch_res = engine.search_batch(&q_mat, 2, &bitset).unwrap();
        let recall = compute_recall_at_k(&batch_res.indices, &batch_res.indices, 2).unwrap();
        assert!((recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filtered_exact_knn_canonical_tie_breaking() {
        // IDs 1, 3, 4 are at identical distance d^2 = 4.0 to query [0, 0]
        let base_data = vec![
            5.0, 0.0, // id 0, d^2 = 25.0
            2.0, 0.0, // id 1, d^2 = 4.0
            5.0, 0.0, // id 2, d^2 = 25.0
            -2.0, 0.0, // id 3, d^2 = 4.0
            0.0, 2.0, // id 4, d^2 = 4.0
        ];
        let base = RowMatrix::new(5, 2, base_data).unwrap();
        let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

        // Include all in bitset
        let bitset = DenseBitset::ones(5);
        let query = vec![0.0_f32, 0.0];
        let hits = engine.search_filtered(&query, 3, &bitset).unwrap();

        assert_eq!(hits.len(), 3);
        // Ties broken by smaller ID: id 1, id 3, id 4
        assert_eq!(hits[0], (4.0, 1));
        assert_eq!(hits[1], (4.0, 3));
        assert_eq!(hits[2], (4.0, 4));
    }

    #[test]
    fn test_filtered_exact_knn_zero_matches_and_fewer_than_k() {
        let base_data = vec![
            1.0, 0.0, // id 0
            2.0, 0.0, // id 1
            3.0, 0.0, // id 2
        ];
        let base = RowMatrix::new(3, 2, base_data).unwrap();
        let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

        // 1. Zero matches
        let empty_bitset = DenseBitset::zeros(3);
        let hits_zero = engine
            .search_filtered(&[0.0, 0.0], 5, &empty_bitset)
            .unwrap();
        assert!(hits_zero.is_empty());

        // 2. Fewer than k matches (1 match, k=5)
        let mut single_bitset = DenseBitset::zeros(3);
        single_bitset.set_bit(1, true);
        let hits_few = engine
            .search_filtered(&[0.0, 0.0], 5, &single_bitset)
            .unwrap();
        assert_eq!(hits_few.len(), 1);
        assert_eq!(hits_few[0], (4.0, 1));

        // 3. k = 0
        let hits_k0 = engine
            .search_filtered(&[0.0, 0.0], 0, &single_bitset)
            .unwrap();
        assert!(hits_k0.is_empty());
    }

    #[test]
    fn test_filtered_exact_knn_validations() {
        let base = RowMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();
        let bitset = DenseBitset::ones(2);

        // Non-finite query (NaN)
        let nan_q = vec![f32::NAN, 1.0];
        assert!(matches!(
            engine.search_filtered(&nan_q, 2, &bitset),
            Err(VectorSearchError::InvalidArgument(_))
        ));

        // Non-finite query (Infinity)
        let inf_q = vec![f32::INFINITY, 1.0];
        assert!(matches!(
            engine.search_filtered(&inf_q, 2, &bitset),
            Err(VectorSearchError::InvalidArgument(_))
        ));

        // Dimension mismatch
        let mismatch_q = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            engine.search_filtered(&mismatch_q, 2, &bitset),
            Err(VectorSearchError::DimensionMismatch {
                expected: 2,
                actual: 3
            })
        ));

        // Empty base matrix
        let empty_base = RowMatrix::empty();
        assert!(matches!(
            FilteredExactKnn::new(Metric::L2Squared, empty_base),
            Err(VectorSearchError::InvalidArgument(_))
        ));

        // Non-finite base matrix
        let non_finite_base = RowMatrix::new(1, 2, vec![f32::NAN, 1.0]).unwrap();
        assert!(matches!(
            FilteredExactKnn::new(Metric::L2Squared, non_finite_base),
            Err(VectorSearchError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_filtered_exact_knn_lambda_predicate() {
        let base_data = vec![
            1.0, 1.0, // id 0
            2.0, 2.0, // id 1
            3.0, 3.0, // id 2
            4.0, 4.0, // id 3
        ];
        let base = RowMatrix::new(4, 2, base_data).unwrap();
        let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

        // LambdaPredicate without bitset: even IDs only (0, 2)
        let lambda = LambdaPredicate::new(|id| id % 2 == 0, 0.5);
        let hits = engine.search_filtered(&[0.0, 0.0], 2, &lambda).unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], (2.0, 0));
        assert_eq!(hits[1], (18.0, 2));
    }

    #[test]
    fn test_filtered_exact_knn_batch_and_parallel_parity() {
        let n = 200;
        let dim = 8;
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(((i * 17 + d * 31) % 100) as f32);
            }
        }
        let base = RowMatrix::new(n, dim, flat).unwrap();
        let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

        let num_q = 20;
        let mut q_flat = Vec::with_capacity(num_q * dim);
        for i in 0..num_q {
            for d in 0..dim {
                q_flat.push(((i * 23 + d * 7) % 50) as f32);
            }
        }
        let queries = RowMatrix::new(num_q, dim, q_flat).unwrap();

        let mut bitset = DenseBitset::zeros(n);
        for i in (0..n).step_by(3) {
            bitset.set_bit(i, true);
        }

        let k = 5;
        let seq_res = engine.search_batch(&queries, k, &bitset).unwrap();
        let par_res = engine
            .search_batch_parallel(&queries, k, &bitset, 4)
            .unwrap();

        assert_eq!(seq_res.indices.rows(), num_q);
        assert_eq!(seq_res.indices.cols(), k);
        assert_eq!(seq_res.indices.as_slice(), par_res.indices.as_slice());
        assert_eq!(seq_res.distances.as_slice(), par_res.distances.as_slice());
    }

    #[test]
    fn test_filtered_exact_knn_all_metrics() {
        let base =
            RowMatrix::from_rows(&[vec![1.0_f32, 0.0], vec![0.0, 2.0], vec![2.0, 0.0]]).unwrap();

        // Euclidean
        let engine_euc = FilteredExactKnn::new(Metric::Euclidean, base.clone()).unwrap();
        let all_bits = DenseBitset::ones(3);
        let hits_euc = engine_euc
            .search_filtered(&[0.0, 0.0], 3, &all_bits)
            .unwrap();
        assert_eq!(hits_euc[0], (1.0, 0));
        assert_eq!(hits_euc[1], (2.0, 1));
        assert_eq!(hits_euc[2], (2.0, 2));

        // Inner product (descending dot score, tie broken by ID)
        let engine_ip = FilteredExactKnn::new(Metric::InnerProduct, base).unwrap();
        let hits_ip = engine_ip
            .search_filtered(&[1.0, 1.0], 3, &all_bits)
            .unwrap();
        // Dot products: id 0 -> 1.0, id 1 -> 2.0, id 2 -> 2.0.
        // Ties between id 1 and id 2 resolved in favor of id 1.
        assert_eq!(hits_ip[0], (2.0, 1));
        assert_eq!(hits_ip[1], (2.0, 2));
        assert_eq!(hits_ip[2], (1.0, 0));
    }
}
