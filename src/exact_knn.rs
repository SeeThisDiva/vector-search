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
        if base.rows() == 0 || base.cols() == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "base_vectors cannot be empty (rows and cols must be > 0)".to_string(),
            ));
        }
        if i32::try_from(base.rows()).is_err() {
            return Err(VectorSearchError::InvalidArgument(format!(
                "base_vectors row count {} exceeds i32::MAX",
                base.rows()
            )));
        }
        if let Some(pos) = base.as_slice().iter().position(|v| !v.is_finite()) {
            let row = pos / base.cols();
            let col = pos % base.cols();
            return Err(VectorSearchError::InvalidArgument(format!(
                "base_vectors contains non-finite value (NaN/Inf) at ({row}, {col})"
            )));
        }
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

        Ok(base)
    }

    #[inline]
    fn compute_sort_key(&self, query: &[f32], base_row: &[f32]) -> f32 {
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
