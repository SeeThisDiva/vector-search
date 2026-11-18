//! Product Quantization (`PQ`) trainer, codebook, safe ADC/SDC kernels, streaming `.fvecs`
//! builder, and lifetime-free [`HnswPqIndex`] (`T4.4` & `T4.5`).
//!
//! All distance kernels and Lloyd/`k`-means++ training routines are 100% safe Rust
//! (`#![forbid(unsafe_code)]`, `std`-only) using `as_chunks::<256>()` + `as_chunks::<8>()`
//! zero-bounds-check ADC accumulation and deterministic `Xoshiro256**` PRNG state.

use std::cmp::Ordering;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::matrix::RowMatrix;
use crate::rng::DeterministicRng;

use super::graph::{
    CompactHnswGraph, Neighbor, QuantizedBuildMode, QuantizedBuildTimings, QuantizedDistanceSource,
    QuantizedMemoryProfile, QuantizedSearchScratch, QuantizedStorageMode, TimedQuantizedSearch,
};
use super::{
    format_metric_output_distance, impl_quantized_index_common, inspect_streaming_fvecs,
    metric_from_u8, metric_to_u8, read_f32_vec, read_index_snapshot, read_magic, read_u64, read_u8,
    read_usize_u64, stream_fvecs_bounded, validate_base_matrix, write_f32_slice,
    write_index_snapshot, write_u64, write_u8,
};

/// Fixed number of centroids per PQ subspace (`K = 256` for 8-bit `u8` codes).
pub const PQ_K_CENTROIDS: usize = 256;
/// Default maximum number of base vectors sampled for PQ codebook training (`256 * K`).
pub const PQ_DEFAULT_MAX_TRAIN_VECTORS: usize = 65_536;
/// Default maximum Lloyd refinement iterations per PQ subspace.
pub const PQ_DEFAULT_MAX_LLOYD_ITERS: usize = 25;
/// Default deterministic PRNG seed for PQ subsampling, `k`-means++, and empty-cluster splitting.
pub const PQ_DEFAULT_SEED: u64 = 42;

const PQ_LLOYD_REL_TOL: f64 = 1e-5;
const PQ_CODEBOOK_MAGIC: &[u8; 8] = b"PQCBK001";
const PQ_VECTORS_MAGIC: &[u8; 8] = b"PQVEC001";
const PQ_SNAPSHOT_MAGIC: &[u8; 8] = b"HPQSN001";
const ENCODE_ROW_BLOCK: usize = 1_024;
const SDC_TABLE_STRIDE: usize = PQ_K_CENTROIDS * PQ_K_CENTROIDS;

/// Configuration for Product Quantization (`PQ`) codebook training and encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PqConfig {
    /// Number of orthogonal subspaces (`M_pq`, requiring `dim % m_pq == 0`; default: `16`).
    pub m_pq: usize,
    /// Number of centroids per subspace (must equal [`PQ_K_CENTROIDS`] = `256` for `u8` codes).
    pub k_centroids: usize,
    /// Maximum base vectors sampled deterministically without replacement for training.
    pub max_train_vectors: usize,
    /// Maximum Lloyd refinement iterations per subspace.
    pub max_lloyd_iters: usize,
    /// Fixed PRNG seed controlling training subsampling, `k`-means++, and empty-cluster recovery.
    pub seed: u64,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            m_pq: 16,
            k_centroids: PQ_K_CENTROIDS,
            max_train_vectors: PQ_DEFAULT_MAX_TRAIN_VECTORS,
            max_lloyd_iters: PQ_DEFAULT_MAX_LLOYD_ITERS,
            seed: PQ_DEFAULT_SEED,
        }
    }
}

impl PqConfig {
    /// Creates a [`PqConfig`] with `m_pq` subspaces and default training hyperparameters.
    #[must_use]
    pub fn new(m_pq: usize) -> Self {
        Self {
            m_pq,
            ..Self::default()
        }
    }

    /// Returns a copy of this configuration with an explicit deterministic training `seed`.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates configuration invariants independent of vector dimension.
    pub fn validate(&self) -> Result<()> {
        if self.m_pq == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "PqConfig::m_pq must be positive (> 0)".to_string(),
            ));
        }
        if self.k_centroids != PQ_K_CENTROIDS {
            return Err(VectorSearchError::InvalidArgument(format!(
                "PqConfig::k_centroids must be {PQ_K_CENTROIDS} for 8-bit PQ codes, got {}",
                self.k_centroids
            )));
        }
        if self.max_train_vectors == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "PqConfig::max_train_vectors must be positive (> 0)".to_string(),
            ));
        }
        Ok(())
    }

    /// Validates configuration invariants for `dim` and returns subspace width `d_sub = dim / m_pq`.
    pub fn validate_for_dim(&self, dim: usize) -> Result<usize> {
        self.validate()?;
        if dim == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "Vector dimension must be positive (> 0)".to_string(),
            ));
        }
        if !dim.is_multiple_of(self.m_pq) {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Vector dimension ({dim}) must be divisible by m_pq ({})",
                self.m_pq
            )));
        }
        Ok(dim / self.m_pq)
    }
}

/// Phase-level wall-clock timings recorded during PQ index construction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PqBuildTimings {
    pub train_time_s: f64,
    pub encode_time_s: f64,
    pub graph_build_time_s: f64,
    pub total_s: f64,
}

impl PqBuildTimings {
    /// Converts PQ build timings to the shared [`QuantizedBuildTimings`] schema.
    #[inline]
    #[must_use]
    pub const fn to_quantized_timings(self) -> QuantizedBuildTimings {
        QuantizedBuildTimings {
            calibration_or_train_s: self.train_time_s,
            encode_s: self.encode_time_s,
            graph_build_s: self.graph_build_time_s,
            total_s: self.total_s,
        }
    }
}

impl From<PqBuildTimings> for QuantizedBuildTimings {
    #[inline]
    fn from(value: PqBuildTimings) -> Self {
        value.to_quantized_timings()
    }
}

/// Reusable per-query ADC lookup table state for [`PqVectors`].
#[derive(Debug, Clone, Default)]
pub struct PqQueryState {
    /// Flattened `[m_pq * 256]` lookup table (`||q_m - C_{m,k}||^2` for `L2Squared`/`Euclidean`,
    /// or `-dot(q_m, C_{m,k})` for `InnerProduct` minimization).
    pub adc_table: Vec<f32>,
}

impl PqQueryState {
    /// Creates a pre-allocated query state sized for `m_pq` subspaces (`m_pq * 256` floats).
    #[must_use]
    pub fn with_subspaces(m_pq: usize) -> Self {
        Self {
            adc_table: Vec::with_capacity(m_pq.saturating_mul(PQ_K_CENTROIDS)),
        }
    }

    /// Allocated capacity of the precomputed ADC lookup table (`adc_table`).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.adc_table.capacity()
    }
}

/// Trained Product Quantization codebook storing `[m_pq][256][d_sub]` `f32` centroids.
#[derive(Debug, Clone, PartialEq)]
pub struct PqCodebook {
    dim: usize,
    m_pq: usize,
    d_sub: usize,
    k_effective: usize,
    config: PqConfig,
    centroids: Vec<f32>,
}

impl PqCodebook {
    /// Full vector dimensionality (`D = m_pq * d_sub`).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of PQ subspaces (`M_pq`).
    #[inline]
    #[must_use]
    pub fn m_pq(&self) -> usize {
        self.m_pq
    }

    /// Subspace dimensionality (`d_sub = D / M_pq`).
    #[inline]
    #[must_use]
    pub fn d_sub(&self) -> usize {
        self.d_sub
    }

    /// Nominal centroid capacity per subspace (`256`).
    #[inline]
    #[must_use]
    pub fn k_centroids(&self) -> usize {
        self.config.k_centroids
    }

    /// Effective number of trained centroids per subspace (`min(256, N_train)`).
    #[inline]
    #[must_use]
    pub fn k_effective(&self) -> usize {
        self.k_effective
    }

    /// Training configuration used to build this codebook.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &PqConfig {
        &self.config
    }

    /// Contiguous `[m_pq * 256 * d_sub]` centroid buffer laid out `[subspace m][centroid k][dim d]`.
    #[inline]
    #[must_use]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Centroid slice `&[f32]` of length `d_sub` for subspace `m` and centroid index `k < 256`.
    ///
    /// # Panics
    /// Panics if `m >= self.m_pq()` or `k >= 256`.
    #[inline]
    #[must_use]
    pub fn centroid(&self, m: usize, k: usize) -> &[f32] {
        assert!(m < self.m_pq, "subspace index {m} >= m_pq {}", self.m_pq);
        assert!(k < PQ_K_CENTROIDS, "centroid index {k} >= {PQ_K_CENTROIDS}");
        let offset = (m * PQ_K_CENTROIDS + k) * self.d_sub;
        &self.centroids[offset..offset + self.d_sub]
    }

    /// Total heap payload bytes of the codebook (`m_pq * 256 * d_sub * sizeof(f32)`).
    #[inline]
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.centroids.capacity() * std::mem::size_of::<f32>()
    }

    /// Encodes a single finite `f32` vector of length `dim` into `m_pq` `u8` PQ codes.
    pub fn encode_row(&self, row: &[f32], out: &mut [u8]) -> Result<()> {
        if row.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: row.len(),
            });
        }
        if out.len() != self.m_pq {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.m_pq,
                actual: out.len(),
            });
        }
        if let Some(pos) = row.iter().position(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(format!(
                "row contains non-finite value (NaN/Inf) at col {pos}"
            )));
        }
        self.encode_row_unchecked(row, out);
        Ok(())
    }

    #[inline]
    fn encode_row_unchecked(&self, row: &[f32], out: &mut [u8]) {
        for (m, dst) in out.iter_mut().enumerate() {
            let sub = &row[m * self.d_sub..(m + 1) * self.d_sub];
            *dst = self.encode_subvector_unchecked(m, sub);
        }
    }

    #[inline]
    fn encode_subvector_unchecked(&self, m: usize, sub: &[f32]) -> u8 {
        let m_base = m * PQ_K_CENTROIDS * self.d_sub;
        let sub_centroids = &self.centroids[m_base..m_base + self.k_effective * self.d_sub];

        let mut best_k = 0_usize;
        let mut best_dist = f64::INFINITY;
        let mut k = 0_usize;

        while k + 4 <= self.k_effective {
            let c0 = &sub_centroids[k * self.d_sub..(k + 1) * self.d_sub];
            let c1 = &sub_centroids[(k + 1) * self.d_sub..(k + 2) * self.d_sub];
            let c2 = &sub_centroids[(k + 2) * self.d_sub..(k + 3) * self.d_sub];
            let c3 = &sub_centroids[(k + 3) * self.d_sub..(k + 4) * self.d_sub];
            let (d0, d1, d2, d3) = sub_l2_sq_4_f32_f32_in_f64(sub, c0, c1, c2, c3);
            if d0 < best_dist {
                best_dist = d0;
                best_k = k;
            }
            if d1 < best_dist {
                best_dist = d1;
                best_k = k + 1;
            }
            if d2 < best_dist {
                best_dist = d2;
                best_k = k + 2;
            }
            if d3 < best_dist {
                best_dist = d3;
                best_k = k + 3;
            }
            k += 4;
        }
        while k < self.k_effective {
            let ck = &sub_centroids[k * self.d_sub..(k + 1) * self.d_sub];
            let d = sub_l2_sq_f32_f32_in_f64(sub, ck);
            if d < best_dist {
                best_dist = d;
                best_k = k;
            }
            k += 1;
        }

        best_k as u8
    }

    /// Encodes all rows of a validated `RowMatrix<f32>` into a flat `[N * m_pq]` `u8` buffer.
    pub fn encode_matrix(&self, matrix: &RowMatrix<f32>) -> Result<Vec<u8>> {
        validate_base_matrix(matrix)?;
        if matrix.cols() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: matrix.cols(),
            });
        }
        Ok(self.encode_matrix_unchecked(matrix))
    }

    fn encode_matrix_unchecked(&self, matrix: &RowMatrix<f32>) -> Vec<u8> {
        let n = matrix.rows();
        let total_codes = n.saturating_mul(self.m_pq);
        let mut codes = vec![0_u8; total_codes];

        // Tile rows in cache-resident blocks so both the active row chunk and subspace
        // centroids stay in L1/L2/L3 while `matrix` is streamed from DRAM only once.
        let mut row_start = 0_usize;
        while row_start < n {
            let row_end = (row_start + ENCODE_ROW_BLOCK).min(n);
            for m in 0..self.m_pq {
                let col_lo = m * self.d_sub;
                let col_hi = col_lo + self.d_sub;
                for row_id in row_start..row_end {
                    let sub = &matrix.row(row_id)[col_lo..col_hi];
                    codes[row_id * self.m_pq + m] = self.encode_subvector_unchecked(m, sub);
                }
            }
            row_start = row_end;
        }

        codes
    }

    /// Reconstructs an approximate `f32` vector by concatenating the centroids indexed by `code`.
    pub fn dequantize_row(&self, code: &[u8], out: &mut [f32]) -> Result<()> {
        if code.len() != self.m_pq {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.m_pq,
                actual: code.len(),
            });
        }
        if out.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: out.len(),
            });
        }
        for (m, &cb) in code.iter().enumerate() {
            let c = self.centroid(m, usize::from(cb));
            out[m * self.d_sub..(m + 1) * self.d_sub].copy_from_slice(c);
        }
        Ok(())
    }

    /// Fills `adc_table` (resized to `m_pq * 256`) for `query` under `metric`.
    pub fn compute_adc_table(
        &self,
        metric: Metric,
        query: &[f32],
        adc_table: &mut Vec<f32>,
    ) -> Result<()> {
        if query.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if let Some(pos) = query.iter().position(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(format!(
                "query contains non-finite value (NaN/Inf) at col {pos}"
            )));
        }
        self.fill_adc_table_unchecked(metric, query, adc_table);
        Ok(())
    }

    #[inline]
    fn fill_adc_table_unchecked(&self, metric: Metric, query: &[f32], adc_table: &mut Vec<f32>) {
        let table_len = self.m_pq * PQ_K_CENTROIDS;
        if adc_table.len() != table_len {
            adc_table.resize(table_len, 0.0);
        }

        for m in 0..self.m_pq {
            let q_sub = &query[m * self.d_sub..(m + 1) * self.d_sub];
            let tbl = &mut adc_table[m * PQ_K_CENTROIDS..(m + 1) * PQ_K_CENTROIDS];
            let cent_base = m * PQ_K_CENTROIDS * self.d_sub;
            let sub_centroids = &self.centroids[cent_base..cent_base + PQ_K_CENTROIDS * self.d_sub];

            match metric {
                Metric::L2Squared | Metric::Euclidean => {
                    for k in 0..self.k_effective {
                        let ck = &sub_centroids[k * self.d_sub..(k + 1) * self.d_sub];
                        let d2 = sub_l2_sq_f32_fast(q_sub, ck);
                        tbl[k] = if d2 <= 0.0 { 0.0 } else { d2 };
                    }
                }
                Metric::InnerProduct => {
                    for k in 0..self.k_effective {
                        let ck = &sub_centroids[k * self.d_sub..(k + 1) * self.d_sub];
                        let neg_dot = -sub_dot_f32_fast(q_sub, ck);
                        tbl[k] = if neg_dot == 0.0 { 0.0 } else { neg_dot };
                    }
                }
            }

            if self.k_effective < PQ_K_CENTROIDS {
                let c0_val = tbl[0];
                tbl[self.k_effective..].fill(c0_val);
            }
        }
    }

    /// Serializes [`PqCodebook`] deterministically to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(PQ_CODEBOOK_MAGIC)?;
        write_u64(writer, self.dim as u64)?;
        write_u64(writer, self.m_pq as u64)?;
        write_u64(writer, self.d_sub as u64)?;
        write_u64(writer, self.k_effective as u64)?;
        write_u64(writer, self.config.k_centroids as u64)?;
        write_u64(writer, self.config.max_train_vectors as u64)?;
        write_u64(writer, self.config.max_lloyd_iters as u64)?;
        write_u64(writer, self.config.seed)?;
        write_f32_slice(writer, &self.centroids)?;
        Ok(())
    }

    /// Deserializes and validates [`PqCodebook`] from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        read_magic(reader, PQ_CODEBOOK_MAGIC, "PqCodebook")?;

        let dim = read_usize_u64(reader, "PqCodebook dim")?;
        let m_pq = read_usize_u64(reader, "PqCodebook m_pq")?;
        let d_sub = read_usize_u64(reader, "PqCodebook d_sub")?;
        let k_effective = read_usize_u64(reader, "PqCodebook k_effective")?;
        let k_centroids = read_usize_u64(reader, "PqCodebook k_centroids")?;
        let max_train_vectors = read_usize_u64(reader, "PqCodebook max_train_vectors")?;
        let max_lloyd_iters = read_usize_u64(reader, "PqCodebook max_lloyd_iters")?;
        let seed = read_u64(reader)?;

        let config = PqConfig {
            m_pq,
            k_centroids,
            max_train_vectors,
            max_lloyd_iters,
            seed,
        };
        let expected_d_sub = config.validate_for_dim(dim).map_err(|e| {
            VectorSearchError::CorruptData(format!("Invalid PqCodebook config: {e}"))
        })?;
        if d_sub != expected_d_sub {
            return Err(VectorSearchError::CorruptData(format!(
                "PqCodebook d_sub mismatch: expected {expected_d_sub}, got {d_sub}"
            )));
        }
        if k_effective == 0 || k_effective > PQ_K_CENTROIDS {
            return Err(VectorSearchError::CorruptData(format!(
                "Invalid PqCodebook k_effective: {k_effective} (expected 1..={PQ_K_CENTROIDS})"
            )));
        }

        let expected_centroids_len = m_pq
            .checked_mul(PQ_K_CENTROIDS)
            .and_then(|v| v.checked_mul(d_sub))
            .ok_or_else(|| {
                VectorSearchError::CorruptData("PqCodebook centroids length overflow".to_string())
            })?;
        let centroids = read_f32_vec(reader, expected_centroids_len)?;
        if let Some(pos) = centroids.iter().position(|v| !v.is_finite()) {
            return Err(VectorSearchError::CorruptData(format!(
                "Non-finite centroid float at index {pos} in PqCodebook snapshot"
            )));
        }

        Ok(Self {
            dim,
            m_pq,
            d_sub,
            k_effective,
            config,
            centroids,
        })
    }
}

/// Deterministic Product Quantization codebook trainer (`Xoshiro256**` subsampling,
/// per-subspace `k`-means++ initialization, and `f64`-accumulated Lloyd refinement).
#[derive(Debug, Clone, Copy, Default)]
pub struct PqTrainer;

impl PqTrainer {
    /// Trains a [`PqCodebook`] on `base` according to `config`.
    pub fn train(base: &RowMatrix<f32>, config: &PqConfig) -> Result<PqCodebook> {
        validate_base_matrix(base)?;
        let _d_sub = config.validate_for_dim(base.cols())?;
        let mut rng = DeterministicRng::new(config.seed);
        let train_indices = sample_train_indices(base.rows(), config.max_train_vectors, &mut rng);
        Self::train_from_indexed_rows(base, &train_indices, config, &mut rng)
    }

    fn train_from_indexed_rows(
        base: &RowMatrix<f32>,
        train_indices: &[usize],
        config: &PqConfig,
        rng: &mut DeterministicRng,
    ) -> Result<PqCodebook> {
        let dim = base.cols();
        let d_sub = config.validate_for_dim(dim)?;
        let m_pq = config.m_pq;
        let n_train = train_indices.len();
        if n_train == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "PqTrainer requires at least 1 training vector".to_string(),
            ));
        }

        let k_effective = PQ_K_CENTROIDS.min(n_train);
        let mut centroids = vec![0.0_f32; m_pq * PQ_K_CENTROIDS * d_sub];
        let mut sub_train = vec![0.0_f32; n_train * d_sub];
        let mut scratch = LloydSubspaceScratch::new(n_train, k_effective, d_sub);

        for m in 0..m_pq {
            let col_lo = m * d_sub;
            let col_hi = col_lo + d_sub;
            for (slot, &row_id) in train_indices.iter().enumerate() {
                sub_train[slot * d_sub..(slot + 1) * d_sub]
                    .copy_from_slice(&base.row(row_id)[col_lo..col_hi]);
            }

            let dst_centroids =
                &mut centroids[m * PQ_K_CENTROIDS * d_sub..(m + 1) * PQ_K_CENTROIDS * d_sub];
            train_subspace(
                &sub_train,
                n_train,
                d_sub,
                k_effective,
                config.max_lloyd_iters,
                rng,
                &mut scratch,
                dst_centroids,
            );
        }

        Ok(PqCodebook {
            dim,
            m_pq,
            d_sub,
            k_effective,
            config: *config,
            centroids,
        })
    }
}

struct LloydSubspaceScratch {
    centers: Vec<f64>,
    min_sq_dist: Vec<f64>,
    assign: Vec<usize>,
    prev_assign: Vec<usize>,
    counts: Vec<usize>,
    cluster_sums: Vec<f64>,
    cluster_sse: Vec<f64>,
}

impl LloydSubspaceScratch {
    fn new(n_train: usize, k_effective: usize, d_sub: usize) -> Self {
        Self {
            centers: vec![0.0_f64; k_effective * d_sub],
            min_sq_dist: vec![0.0_f64; n_train],
            assign: vec![0_usize; n_train],
            prev_assign: vec![0_usize; n_train],
            counts: vec![0_usize; k_effective],
            cluster_sums: vec![0.0_f64; k_effective * d_sub],
            cluster_sse: vec![0.0_f64; k_effective],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn train_subspace(
    sub_train: &[f32],
    n_train: usize,
    d_sub: usize,
    k_effective: usize,
    max_lloyd_iters: usize,
    rng: &mut DeterministicRng,
    scratch: &mut LloydSubspaceScratch,
    out_centroids: &mut [f32],
) {
    kmeans_pp_init_subspace(sub_train, n_train, d_sub, k_effective, rng, scratch);

    let mut prev_mse = f64::INFINITY;
    let mut has_prev_assign = false;

    for _iter in 0..max_lloyd_iters {
        scratch.counts.fill(0);
        scratch.cluster_sums.fill(0.0);
        scratch.cluster_sse.fill(0.0);

        let mut total_sse = 0.0_f64;
        let mut assign_changes = 0_usize;

        // 1. Assign step: nearest centroid in 0..k_effective (canonical smallest-k tie-break)
        for i in 0..n_train {
            let x = &sub_train[i * d_sub..(i + 1) * d_sub];
            let (best_k, best_dist) = nearest_centroid_f64(x, &scratch.centers, k_effective, d_sub);

            if !has_prev_assign || scratch.assign[i] != best_k {
                assign_changes += 1;
            }
            scratch.assign[i] = best_k;
            total_sse += best_dist;
            scratch.counts[best_k] += 1;
            scratch.cluster_sse[best_k] += best_dist;

            let sum_slice = &mut scratch.cluster_sums[best_k * d_sub..(best_k + 1) * d_sub];
            for (acc, &val) in sum_slice.iter_mut().zip(x.iter()) {
                *acc += f64::from(val);
            }
        }

        if has_prev_assign && assign_changes == 0 {
            break;
        }

        let mse = total_sse / (n_train as f64);

        // 2. Update non-empty centroids from f64 sums
        for k in 0..k_effective {
            let cnt = scratch.counts[k];
            if cnt > 0 {
                let inv_c = 1.0 / (cnt as f64);
                let sums = &scratch.cluster_sums[k * d_sub..(k + 1) * d_sub];
                let cent = &mut scratch.centers[k * d_sub..(k + 1) * d_sub];
                for (c_val, &s_val) in cent.iter_mut().zip(sums.iter()) {
                    *c_val = s_val * inv_c;
                }
            }
        }

        // 3. Deterministic empty-cluster recovery: split most populous cluster
        //    (tie-broken by highest SSE, then smallest k) by relocating its farthest member
        //    (or deterministic tiny jitter if all members in the donor cluster are identical).
        let mut had_empty = false;
        for empty_k in 0..k_effective {
            if scratch.counts[empty_k] != 0 {
                continue;
            }
            had_empty = true;
            let Some(donor_k) = select_donor_cluster(&scratch.counts, &scratch.cluster_sse) else {
                // Unreachable by pigeonhole principle since sum(counts) == n_train >= k_effective.
                let (head, tail) = scratch.centers.split_at_mut(empty_k * d_sub);
                if empty_k > 0 {
                    tail[..d_sub].copy_from_slice(&head[..d_sub]);
                }
                continue;
            };

            let donor_center_start = donor_k * d_sub;
            let mut far_i = 0_usize;
            let mut max_d = -1.0_f64;
            for i in 0..n_train {
                if scratch.assign[i] == donor_k {
                    let x = &sub_train[i * d_sub..(i + 1) * d_sub];
                    let d = sub_l2_sq_f32_f64(
                        x,
                        &scratch.centers[donor_center_start..donor_center_start + d_sub],
                    );
                    if d > max_d {
                        max_d = d;
                        far_i = i;
                    }
                }
            }

            let far_x = &sub_train[far_i * d_sub..(far_i + 1) * d_sub];
            scratch.assign[far_i] = empty_k;
            scratch.counts[donor_k] -= 1;
            scratch.counts[empty_k] = 1;
            scratch.cluster_sse[empty_k] = 0.0;

            let donor_sums =
                &mut scratch.cluster_sums[donor_center_start..donor_center_start + d_sub];
            for (s_val, &x_val) in donor_sums.iter_mut().zip(far_x.iter()) {
                *s_val -= f64::from(x_val);
            }

            if max_d > 0.0 {
                let inv_donor = 1.0 / (scratch.counts[donor_k] as f64);
                let donor_cent =
                    &mut scratch.centers[donor_center_start..donor_center_start + d_sub];
                for (c_val, &s_val) in donor_cent.iter_mut().zip(donor_sums.iter()) {
                    *c_val = s_val * inv_donor;
                }
                scratch.cluster_sse[donor_k] = (scratch.cluster_sse[donor_k] - max_d).max(0.0);

                let empty_cent = &mut scratch.centers[empty_k * d_sub..(empty_k + 1) * d_sub];
                for (c_val, &x_val) in empty_cent.iter_mut().zip(far_x.iter()) {
                    *c_val = f64::from(x_val);
                }
            } else {
                // Donor cluster has identical members: leave donor center exact and perturb
                // the relocated split centroid deterministically so distances stay finite.
                let empty_cent = &mut scratch.centers[empty_k * d_sub..(empty_k + 1) * d_sub];
                for (c_val, &x_val) in empty_cent.iter_mut().zip(far_x.iter()) {
                    let base_val = f64::from(x_val);
                    let step = base_val.abs().max(1.0) * 1e-6;
                    let signed_noise = (rng.next_f64_open01() - 0.5) * 2.0 * step;
                    let candidate = base_val + signed_noise;
                    *c_val = if candidate.abs() <= f64::from(f32::MAX) {
                        candidate
                    } else {
                        base_val - base_val.signum() * step
                    };
                }
            }
        }

        if mse == 0.0 {
            break;
        }
        if !had_empty
            && prev_mse > 0.0
            && prev_mse.is_finite()
            && (prev_mse - mse) / prev_mse < PQ_LLOYD_REL_TOL
        {
            break;
        }

        prev_mse = mse;
        scratch.prev_assign.copy_from_slice(&scratch.assign);
        has_prev_assign = true;
    }

    // Store trained centroids as f32 and pad k_effective..256 with copies of centroid 0.
    for k in 0..k_effective {
        let src = &scratch.centers[k * d_sub..(k + 1) * d_sub];
        let dst = &mut out_centroids[k * d_sub..(k + 1) * d_sub];
        for (d_out, &c_in) in dst.iter_mut().zip(src.iter()) {
            *d_out = c_in as f32;
        }
    }
    if k_effective < PQ_K_CENTROIDS {
        let (first, rest) = out_centroids.split_at_mut(d_sub);
        for k in k_effective..PQ_K_CENTROIDS {
            let slot_start = (k - 1) * d_sub;
            rest[slot_start..slot_start + d_sub].copy_from_slice(first);
        }
    }
}

fn kmeans_pp_init_subspace(
    sub_train: &[f32],
    n_train: usize,
    d_sub: usize,
    k_effective: usize,
    rng: &mut DeterministicRng,
    scratch: &mut LloydSubspaceScratch,
) {
    let first_idx = rng.next_usize(n_train);
    let first_src = &sub_train[first_idx * d_sub..(first_idx + 1) * d_sub];
    for (c_out, &x_in) in scratch.centers[..d_sub].iter_mut().zip(first_src.iter()) {
        *c_out = f64::from(x_in);
    }

    for i in 0..n_train {
        let x = &sub_train[i * d_sub..(i + 1) * d_sub];
        scratch.min_sq_dist[i] = sub_l2_sq_f32_f64(x, &scratch.centers[..d_sub]);
    }

    for c_idx in 1..k_effective {
        let mut total_dist = 0.0_f64;
        let mut last_positive = n_train - 1;
        for (i, &d) in scratch.min_sq_dist.iter().enumerate() {
            total_dist += d;
            if d > 0.0 {
                last_positive = i;
            }
        }

        let chosen_idx = if total_dist > 0.0 && total_dist.is_finite() {
            let threshold = rng.next_f64_open01() * total_dist;
            let mut cumsum = 0.0_f64;
            let mut chosen = last_positive;
            for (i, &d) in scratch.min_sq_dist.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            rng.next_usize(n_train)
        };

        let src = &sub_train[chosen_idx * d_sub..(chosen_idx + 1) * d_sub];
        let c_start = c_idx * d_sub;
        for (c_out, &x_in) in scratch.centers[c_start..c_start + d_sub]
            .iter_mut()
            .zip(src.iter())
        {
            *c_out = f64::from(x_in);
        }

        let new_center = &scratch.centers[c_start..c_start + d_sub];
        for (i, min_d) in scratch.min_sq_dist.iter_mut().enumerate() {
            let x = &sub_train[i * d_sub..(i + 1) * d_sub];
            let d = sub_l2_sq_f32_f64(x, new_center);
            if d < *min_d {
                *min_d = d;
            }
        }
    }
}

#[inline]
fn select_donor_cluster(counts: &[usize], cluster_sse: &[f64]) -> Option<usize> {
    let mut best_donor: Option<usize> = None;
    for k in 0..counts.len() {
        if counts[k] < 2 {
            continue;
        }
        let is_better = match best_donor {
            None => true,
            Some(best_k) => match counts[k].cmp(&counts[best_k]) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => match cluster_sse[k].total_cmp(&cluster_sse[best_k]) {
                    Ordering::Greater => true,
                    Ordering::Less => false,
                    Ordering::Equal => k < best_k,
                },
            },
        };
        if is_better {
            best_donor = Some(k);
        }
    }
    best_donor
}

#[inline]
fn nearest_centroid_f64(
    x: &[f32],
    centers: &[f64],
    k_effective: usize,
    d_sub: usize,
) -> (usize, f64) {
    let mut best_k = 0_usize;
    let mut best_dist = f64::INFINITY;
    let mut k = 0_usize;

    while k + 4 <= k_effective {
        let c0 = &centers[k * d_sub..(k + 1) * d_sub];
        let c1 = &centers[(k + 1) * d_sub..(k + 2) * d_sub];
        let c2 = &centers[(k + 2) * d_sub..(k + 3) * d_sub];
        let c3 = &centers[(k + 3) * d_sub..(k + 4) * d_sub];
        let (d0, d1, d2, d3) = sub_l2_sq_4_f32_f64(x, c0, c1, c2, c3);
        if d0 < best_dist {
            best_dist = d0;
            best_k = k;
        }
        if d1 < best_dist {
            best_dist = d1;
            best_k = k + 1;
        }
        if d2 < best_dist {
            best_dist = d2;
            best_k = k + 2;
        }
        if d3 < best_dist {
            best_dist = d3;
            best_k = k + 3;
        }
        k += 4;
    }

    while k < k_effective {
        let ck = &centers[k * d_sub..(k + 1) * d_sub];
        let d = sub_l2_sq_f32_f64(x, ck);
        if d < best_dist {
            best_dist = d;
            best_k = k;
        }
        k += 1;
    }

    (best_k, best_dist)
}

#[inline]
fn sample_train_indices(
    n_total: usize,
    max_train_vectors: usize,
    rng: &mut DeterministicRng,
) -> Vec<usize> {
    let n_train = n_total.min(max_train_vectors);
    let mut indices: Vec<usize> = (0..n_total).collect();
    if n_train < n_total {
        for i in 0..n_train {
            let j = i + rng.next_usize(n_total - i);
            indices.swap(i, j);
        }
        indices.truncate(n_train);
        indices.shrink_to_fit();
    }
    indices
}

/// Safe, bounds-check-free 8-way unrolled PQ Asymmetric Distance Computation (`ADC`) kernel.
///
/// Sums `adc_table[m * 256 + usize::from(code[m])]` across all `M_pq` subspaces and clamps
/// negative zero/rounding artifacts to `0.0`.
///
/// # Panics
/// Panics if `adc_table.len()` is not `code.len() * 256`.
#[inline]
#[must_use]
pub fn pq_adc_distance(adc_table: &[f32], code: &[u8]) -> f32 {
    let sum = pq_adc_sum_raw(adc_table, code);
    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

/// Safe, bounds-check-free 8-way unrolled PQ Symmetric Distance Computation (`SDC`) kernel
/// over a precomputed `[m_pq * 256 * 256]` inter-centroid lookup table.
///
/// # Panics
/// Panics if `sdc_table.len() != code_a.len() * 65_536` or `code_a.len() != code_b.len()`.
#[inline]
#[must_use]
pub fn pq_sdc_distance(sdc_table: &[f32], code_a: &[u8], code_b: &[u8]) -> f32 {
    let sum = pq_sdc_sum_raw(sdc_table, code_a, code_b);
    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

#[inline]
fn pq_adc_sum_raw(adc_table: &[f32], code: &[u8]) -> f32 {
    let (sub_tables, rem_t) = adc_table.as_chunks::<PQ_K_CENTROIDS>();
    assert!(
        rem_t.is_empty(),
        "adc_table length ({}) must be a multiple of {PQ_K_CENTROIDS}",
        adc_table.len()
    );
    assert_eq!(
        sub_tables.len(),
        code.len(),
        "adc_table subspace count ({}) != code length ({})",
        sub_tables.len(),
        code.len()
    );

    let (tbl_chunks, tbl_rem) = sub_tables.as_chunks::<8>();
    let (code_chunks, code_rem) = code.as_chunks::<8>();

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

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
    sum
}

#[inline]
fn pq_sdc_sum_raw(sdc_table: &[f32], code_a: &[u8], code_b: &[u8]) -> f32 {
    assert_eq!(
        code_a.len(),
        code_b.len(),
        "PQ code length mismatch: {} != {}",
        code_a.len(),
        code_b.len()
    );
    let (sub_tables, rem_t) = sdc_table.as_chunks::<SDC_TABLE_STRIDE>();
    assert!(
        rem_t.is_empty(),
        "sdc_table length ({}) must be a multiple of {SDC_TABLE_STRIDE}",
        sdc_table.len()
    );
    assert_eq!(
        sub_tables.len(),
        code_a.len(),
        "sdc_table subspace count ({}) != code length ({})",
        sub_tables.len(),
        code_a.len()
    );

    let (tbl_chunks, tbl_rem) = sub_tables.as_chunks::<8>();
    let (ca_chunks, ca_rem) = code_a.as_chunks::<8>();
    let (cb_chunks, cb_rem) = code_b.as_chunks::<8>();

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    for ((tc, ca), cb) in tbl_chunks
        .iter()
        .zip(ca_chunks.iter())
        .zip(cb_chunks.iter())
    {
        s0 += tc[0][(usize::from(ca[0]) << 8) | usize::from(cb[0])];
        s1 += tc[1][(usize::from(ca[1]) << 8) | usize::from(cb[1])];
        s2 += tc[2][(usize::from(ca[2]) << 8) | usize::from(cb[2])];
        s3 += tc[3][(usize::from(ca[3]) << 8) | usize::from(cb[3])];
        s4 += tc[4][(usize::from(ca[4]) << 8) | usize::from(cb[4])];
        s5 += tc[5][(usize::from(ca[5]) << 8) | usize::from(cb[5])];
        s6 += tc[6][(usize::from(ca[6]) << 8) | usize::from(cb[6])];
        s7 += tc[7][(usize::from(ca[7]) << 8) | usize::from(cb[7])];
    }

    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for ((tbl, &va), &vb) in tbl_rem.iter().zip(ca_rem.iter()).zip(cb_rem.iter()) {
        sum += tbl[(usize::from(va) << 8) | usize::from(vb)];
    }
    sum
}

/// Contiguous `u8` Product-Quantized base vector store implementing [`QuantizedDistanceSource`].
///
/// Steady-state memory contains only the `N * m_pq` byte codes and `128 KiB` codebook
/// (`D = 128`); no `4–8 MiB` SDC table is retained in [`PqVectors`].
#[derive(Debug, Clone, PartialEq)]
pub struct PqVectors {
    metric: Metric,
    codebook: PqCodebook,
    num_vectors: usize,
    codes: Vec<u8>,
}

impl PqVectors {
    /// Trains a [`PqCodebook`] and encodes an in-memory `RowMatrix<f32>` into [`PqVectors`].
    pub fn from_matrix(matrix: &RowMatrix<f32>, metric: Metric, config: &PqConfig) -> Result<Self> {
        let (vectors, _, _) = Self::from_matrix_timed(matrix, metric, config)?;
        Ok(vectors)
    }

    fn from_matrix_timed(
        matrix: &RowMatrix<f32>,
        metric: Metric,
        config: &PqConfig,
    ) -> Result<(Self, f64, f64)> {
        let t_train = Instant::now();
        let codebook = PqTrainer::train(matrix, config)?;
        let train_s = t_train.elapsed().as_secs_f64();

        let t_enc = Instant::now();
        let codes = codebook.encode_matrix_unchecked(matrix);
        let enc_s = t_enc.elapsed().as_secs_f64();

        Ok((
            Self {
                metric,
                codebook,
                num_vectors: matrix.rows(),
                codes,
            },
            train_s,
            enc_s,
        ))
    }

    /// Encodes `matrix` against an already-trained [`PqCodebook`].
    pub fn from_codebook_and_matrix(
        codebook: PqCodebook,
        matrix: &RowMatrix<f32>,
        metric: Metric,
    ) -> Result<Self> {
        let codes = codebook.encode_matrix(matrix)?;
        Ok(Self {
            metric,
            codebook,
            num_vectors: matrix.rows(),
            codes,
        })
    }

    /// Two-pass streaming `.fvecs` PQ training and encoding without materializing the full
    /// `RowMatrix<f32>` in memory.
    ///
    /// Pass 1 samples `<= max_train_vectors` rows into a small training buffer using the exact
    /// same `DeterministicRng::new(config.seed)` sequence as [`PqTrainer::train`], producing
    /// bit-identical codebooks and codes as [`Self::from_matrix`].
    pub fn from_fvecs_streaming<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        config: &PqConfig,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        let (vectors, _, _) = Self::from_fvecs_streaming_timed(path, metric, config, max_rows)?;
        Ok(vectors)
    }

    fn from_fvecs_streaming_timed<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        config: &PqConfig,
        max_rows: Option<usize>,
    ) -> Result<(Self, f64, f64)> {
        config.validate()?;
        let path_ref = path.as_ref();
        let (num_vecs, dim, chunk_rows) = inspect_streaming_fvecs(path_ref, max_rows)?;
        let _d_sub = config.validate_for_dim(dim)?;

        // Pass 1: Sample deterministic training subset into a compact <= 65,536-row matrix.
        let t_train = Instant::now();
        let mut rng = DeterministicRng::new(config.seed);
        let train_indices = sample_train_indices(num_vecs, config.max_train_vectors, &mut rng);
        let n_train = train_indices.len();
        let mut row_to_slot: Vec<(usize, usize)> = train_indices
            .into_iter()
            .enumerate()
            .map(|(slot, row_id)| (row_id, slot))
            .collect();
        row_to_slot.sort_unstable();

        let train_elems = n_train.checked_mul(dim).ok_or_else(|| {
            VectorSearchError::InvalidArgument(
                "PQ training matrix size overflows usize".to_string(),
            )
        })?;
        let mut train_data = vec![0.0_f32; train_elems];
        let mut pair_cursor = 0_usize;

        stream_fvecs_bounded(
            path_ref,
            max_rows,
            chunk_rows,
            |row_offset, rows_in_chunk, chunk| {
                for r in 0..rows_in_chunk {
                    let global_id = row_offset + r;
                    while pair_cursor < row_to_slot.len() && row_to_slot[pair_cursor].0 == global_id
                    {
                        let slot = row_to_slot[pair_cursor].1;
                        train_data[slot * dim..(slot + 1) * dim]
                            .copy_from_slice(&chunk[r * dim..(r + 1) * dim]);
                        pair_cursor += 1;
                    }
                }
            },
        )?;
        drop(row_to_slot);

        let train_matrix = RowMatrix::new(n_train, dim, train_data)?;
        let sequential_slots: Vec<usize> = (0..n_train).collect();
        let codebook =
            PqTrainer::train_from_indexed_rows(&train_matrix, &sequential_slots, config, &mut rng)?;
        drop(train_matrix);
        let train_s = t_train.elapsed().as_secs_f64();

        // Pass 2: Stream chunks and encode PQ u8 codes directly into [N * m_pq].
        let t_enc = Instant::now();
        let m_pq = codebook.m_pq();
        let total_codes = num_vecs.checked_mul(m_pq).ok_or_else(|| {
            VectorSearchError::InvalidArgument("PQ code buffer size overflows usize".to_string())
        })?;
        let mut codes = vec![0_u8; total_codes];

        stream_fvecs_bounded(
            path_ref,
            max_rows,
            chunk_rows,
            |row_offset, rows_in_chunk, chunk| {
                for r in 0..rows_in_chunk {
                    let global_id = row_offset + r;
                    let row = &chunk[r * dim..(r + 1) * dim];
                    let dst = &mut codes[global_id * m_pq..(global_id + 1) * m_pq];
                    codebook.encode_row_unchecked(row, dst);
                }
            },
        )?;
        let enc_s = t_enc.elapsed().as_secs_f64();

        Ok((
            Self {
                metric,
                codebook,
                num_vectors: num_vecs,
                codes,
            },
            train_s,
            enc_s,
        ))
    }

    /// Reference to the trained [`PqCodebook`].
    #[inline]
    #[must_use]
    pub fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }

    /// Number of PQ subspaces (`M_pq`).
    #[inline]
    #[must_use]
    pub fn m_pq(&self) -> usize {
        self.codebook.m_pq()
    }

    /// Returns the `m_pq`-byte PQ code slice for base vector `id`.
    ///
    /// # Panics
    /// Panics if `id >= self.len()`.
    #[inline]
    #[must_use]
    pub fn code(&self, id: usize) -> &[u8] {
        let m_pq = self.codebook.m_pq();
        let start = id * m_pq;
        &self.codes[start..start + m_pq]
    }

    /// Contiguous flat `[N * m_pq]` `u8` PQ code buffer.
    #[inline]
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Heap bytes allocated for quantized `u8` PQ codes.
    #[inline]
    #[must_use]
    pub fn codes_memory_bytes(&self) -> usize {
        self.codes.capacity()
    }

    /// Auxiliary heap bytes retained alongside codes and codebook (always `0` for steady-state [`PqVectors`]).
    #[inline]
    #[must_use]
    pub fn aux_memory_bytes(&self) -> usize {
        0
    }

    /// Wraps `self` with a temporary `[m_pq * 256 * 256]` SDC lookup table for fast
    /// `QuantizedDirect` graph construction. Call [`PqWithSdc::into_pq_vectors`] after
    /// graph build to drop the temporary table immediately.
    #[must_use]
    pub fn into_with_sdc(self) -> PqWithSdc {
        PqWithSdc::new(self)
    }

    /// Serializes [`PqVectors`] to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(PQ_VECTORS_MAGIC)?;
        write_u8(writer, metric_to_u8(self.metric))?;
        write_u64(writer, self.num_vectors as u64)?;
        self.codebook.write_to(writer)?;
        write_u64(writer, self.codes.len() as u64)?;
        writer.write_all(&self.codes)?;
        Ok(())
    }

    /// Deserializes and validates [`PqVectors`] from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        read_magic(reader, PQ_VECTORS_MAGIC, "PqVectors")?;

        let metric = metric_from_u8(read_u8(reader)?)?;
        let num_vectors = read_usize_u64(reader, "PqVectors num_vectors")?;
        if num_vectors == 0 || num_vectors > u32::MAX as usize {
            return Err(VectorSearchError::CorruptData(format!(
                "Invalid PqVectors num_vectors: {num_vectors}"
            )));
        }

        let codebook = PqCodebook::read_from(reader)?;
        let expected_codes_len = num_vectors.checked_mul(codebook.m_pq()).ok_or_else(|| {
            VectorSearchError::CorruptData("PqVectors codes length overflows usize".to_string())
        })?;
        let codes_len = read_usize_u64(reader, "PqVectors codes_len")?;
        if codes_len != expected_codes_len {
            return Err(VectorSearchError::CorruptData(format!(
                "PqVectors codes length mismatch: expected {expected_codes_len}, got {codes_len}"
            )));
        }

        let mut codes = vec![0_u8; codes_len];
        reader.read_exact(&mut codes)?;

        let k_eff = codebook.k_effective();
        if k_eff < PQ_K_CENTROIDS {
            for (pos, &cb) in codes.iter().enumerate() {
                if usize::from(cb) >= k_eff {
                    return Err(VectorSearchError::CorruptData(format!(
                        "PQ code {cb} at byte {pos} exceeds k_effective {k_eff}"
                    )));
                }
            }
        }

        Ok(Self {
            metric,
            codebook,
            num_vectors,
            codes,
        })
    }
}

impl QuantizedDistanceSource for PqVectors {
    type QueryState = PqQueryState;

    #[inline]
    fn len(&self) -> usize {
        self.num_vectors
    }

    #[inline]
    fn dim(&self) -> usize {
        self.codebook.dim()
    }

    #[inline]
    fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn prepare_query(&self, query: &[f32], state: &mut Self::QueryState) {
        assert_eq!(
            query.len(),
            self.codebook.dim(),
            "PQ query dimension mismatch"
        );
        self.codebook
            .fill_adc_table_unchecked(self.metric, query, &mut state.adc_table);
    }

    #[inline]
    fn query_distance(&self, state: &Self::QueryState, node_id: usize) -> f32 {
        let code = self.code(node_id);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => pq_adc_distance(&state.adc_table, code),
            Metric::InnerProduct => {
                let neg_dot = pq_adc_sum_raw(&state.adc_table, code);
                if neg_dot == 0.0 {
                    0.0
                } else {
                    neg_dot
                }
            }
        }
    }

    #[inline]
    fn node_distance(&self, id_a: usize, id_b: usize) -> f32 {
        let code_a = self.code(id_a);
        let code_b = self.code(id_b);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => {
                if id_a == id_b {
                    return 0.0;
                }
                let mut sum = 0.0_f32;
                for (m, (&ca, &cb)) in code_a.iter().zip(code_b.iter()).enumerate() {
                    if ca != cb {
                        let cent_a = self.codebook.centroid(m, usize::from(ca));
                        let cent_b = self.codebook.centroid(m, usize::from(cb));
                        sum += sub_l2_sq_f32_fast(cent_a, cent_b);
                    }
                }
                if sum <= 0.0 {
                    0.0
                } else {
                    sum
                }
            }
            Metric::InnerProduct => {
                let mut neg_dot = 0.0_f32;
                for (m, (&ca, &cb)) in code_a.iter().zip(code_b.iter()).enumerate() {
                    let cent_a = self.codebook.centroid(m, usize::from(ca));
                    let cent_b = self.codebook.centroid(m, usize::from(cb));
                    neg_dot += -sub_dot_f32_fast(cent_a, cent_b);
                }
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
        format_metric_output_distance(self.metric, sort_key)
    }

    #[inline]
    fn vector_memory_bytes(&self) -> usize {
        self.codes_memory_bytes() + self.codebook.memory_bytes()
    }
}

/// Temporary wrapper pairing [`PqVectors`] with an `[m_pq * 256 * 256]` inter-centroid SDC
/// lookup table for fast `QuantizedDirect` graph construction.
///
/// Convert back with [`PqWithSdc::into_pq_vectors`] immediately after graph construction so
/// steady-state index memory never retains the `4–8 MiB` SDC table.
#[derive(Debug, Clone)]
pub struct PqWithSdc {
    vectors: PqVectors,
    sdc_table: Vec<f32>,
}

impl PqWithSdc {
    /// Precomputes the symmetric inter-centroid `[m_pq * 256 * 256]` lookup table for `vectors`.
    #[must_use]
    pub fn new(vectors: PqVectors) -> Self {
        let sdc_table = build_sdc_table(&vectors.codebook, vectors.metric);
        Self { vectors, sdc_table }
    }

    /// Drops the temporary SDC table and returns the lean steady-state [`PqVectors`].
    #[inline]
    #[must_use]
    pub fn into_pq_vectors(self) -> PqVectors {
        self.vectors
    }

    /// Reference to the wrapped [`PqVectors`].
    #[inline]
    #[must_use]
    pub fn vectors(&self) -> &PqVectors {
        &self.vectors
    }

    /// Precomputed flat `[m_pq * 256 * 256]` SDC lookup table.
    #[inline]
    #[must_use]
    pub fn sdc_table(&self) -> &[f32] {
        &self.sdc_table
    }

    /// Temporary heap bytes occupied by the SDC lookup table (`m_pq * 65_536 * 4`).
    #[inline]
    #[must_use]
    pub fn sdc_memory_bytes(&self) -> usize {
        self.sdc_table.capacity() * std::mem::size_of::<f32>()
    }
}

impl QuantizedDistanceSource for PqWithSdc {
    type QueryState = PqQueryState;

    #[inline]
    fn len(&self) -> usize {
        self.vectors.len()
    }

    #[inline]
    fn dim(&self) -> usize {
        self.vectors.dim()
    }

    #[inline]
    fn metric(&self) -> Metric {
        self.vectors.metric()
    }

    #[inline]
    fn prepare_query(&self, query: &[f32], state: &mut Self::QueryState) {
        self.vectors.prepare_query(query, state);
    }

    #[inline]
    fn query_distance(&self, state: &Self::QueryState, node_id: usize) -> f32 {
        self.vectors.query_distance(state, node_id)
    }

    #[inline]
    fn node_distance(&self, id_a: usize, id_b: usize) -> f32 {
        let code_a = self.vectors.code(id_a);
        let code_b = self.vectors.code(id_b);
        match self.vectors.metric {
            Metric::L2Squared | Metric::Euclidean => {
                if id_a == id_b {
                    return 0.0;
                }
                pq_sdc_distance(&self.sdc_table, code_a, code_b)
            }
            Metric::InnerProduct => {
                let neg_dot = pq_sdc_sum_raw(&self.sdc_table, code_a, code_b);
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
        self.vectors.format_output_distance(sort_key)
    }

    #[inline]
    fn vector_memory_bytes(&self) -> usize {
        self.vectors.vector_memory_bytes() + self.sdc_memory_bytes()
    }
}

fn build_sdc_table(codebook: &PqCodebook, metric: Metric) -> Vec<f32> {
    let m_pq = codebook.m_pq();
    let k_eff = codebook.k_effective();
    let mut sdc = vec![0.0_f32; m_pq * SDC_TABLE_STRIDE];

    for m in 0..m_pq {
        let base = m * SDC_TABLE_STRIDE;
        let tbl = &mut sdc[base..base + SDC_TABLE_STRIDE];

        for ka in 0..k_eff {
            let ca = codebook.centroid(m, ka);
            let row_a = ka * PQ_K_CENTROIDS;
            for kb in ka..k_eff {
                let val = match metric {
                    Metric::L2Squared | Metric::Euclidean => {
                        if ka == kb {
                            0.0
                        } else {
                            let cb = codebook.centroid(m, kb);
                            let d2 = sub_l2_sq_f32_fast(ca, cb);
                            if d2 <= 0.0 {
                                0.0
                            } else {
                                d2
                            }
                        }
                    }
                    Metric::InnerProduct => {
                        let cb = codebook.centroid(m, kb);
                        let neg_dot = -sub_dot_f32_fast(ca, cb);
                        if neg_dot == 0.0 {
                            0.0
                        } else {
                            neg_dot
                        }
                    }
                };
                tbl[row_a + kb] = val;
                tbl[kb * PQ_K_CENTROIDS + ka] = val;
            }
        }

        if k_eff < PQ_K_CENTROIDS {
            // Replicate centroid 0 entries across padded slots k_eff..256 so all 256x256 entries
            // remain finite and symmetric.
            for ka in 0..k_eff {
                let v = tbl[ka * PQ_K_CENTROIDS];
                for kb in k_eff..PQ_K_CENTROIDS {
                    tbl[ka * PQ_K_CENTROIDS + kb] = v;
                    tbl[kb * PQ_K_CENTROIDS + ka] = v;
                }
            }
            let v00 = tbl[0];
            for ka in k_eff..PQ_K_CENTROIDS {
                for kb in k_eff..PQ_K_CENTROIDS {
                    tbl[ka * PQ_K_CENTROIDS + kb] = v00;
                }
            }
        }
    }

    sdc
}

/// Lifetime-free Product Quantization HNSW index separating memory-first (`QuantizedOnly`)
/// from optional exact top-`R` reranking (`WithExactRerank`).
#[derive(Debug, Clone)]
pub struct HnswPqIndex {
    graph: CompactHnswGraph,
    vectors: PqVectors,
    raw_base: Option<Arc<RowMatrix<f32>>>,
    build_mode: QuantizedBuildMode,
    storage_mode: QuantizedStorageMode,
    build_timings: PqBuildTimings,
}

impl HnswPqIndex {
    /// Builds a PQ HNSW index from `base`.
    ///
    /// When `storage_mode == QuantizedStorageMode::QuantizedOnly`, no raw `f32` vectors are
    /// cloned or retained on `self`.
    pub fn build(
        base: &RowMatrix<f32>,
        metric: Metric,
        config: HnswConfig,
        pq_config: PqConfig,
        build_mode: QuantizedBuildMode,
        storage_mode: QuantizedStorageMode,
    ) -> Result<Self> {
        let total_start = Instant::now();
        config.validate()?;
        validate_base_matrix(base)?;
        let _d_sub = pq_config.validate_for_dim(base.cols())?;

        let (graph, vectors, train_time_s, encode_time_s, graph_build_time_s) = match build_mode {
            QuantizedBuildMode::Float32GraphThenQuantize => {
                let t_graph = Instant::now();
                let f32_index = HnswIndex::build(base, metric, config)?;
                let graph = CompactHnswGraph::from_hnsw_index(&f32_index);
                drop(f32_index);
                let graph_s = t_graph.elapsed().as_secs_f64();

                let (vectors, train_s, enc_s) =
                    PqVectors::from_matrix_timed(base, metric, &pq_config)?;
                (graph, vectors, train_s, enc_s, graph_s)
            }
            QuantizedBuildMode::QuantizedDirect => {
                let (vectors, train_s, enc_s) =
                    PqVectors::from_matrix_timed(base, metric, &pq_config)?;
                let t_graph = Instant::now();
                let sdc_source = vectors.into_with_sdc();
                let graph = CompactHnswGraph::build_with_source(&sdc_source, config)?;
                let vectors = sdc_source.into_pq_vectors();
                let graph_s = t_graph.elapsed().as_secs_f64();
                (graph, vectors, train_s, enc_s, graph_s)
            }
        };

        let raw_base = match storage_mode {
            QuantizedStorageMode::QuantizedOnly => None,
            QuantizedStorageMode::WithExactRerank => Some(Arc::new(base.clone())),
        };

        Ok(Self {
            graph,
            vectors,
            raw_base,
            build_mode,
            storage_mode,
            build_timings: PqBuildTimings {
                train_time_s,
                encode_time_s,
                graph_build_time_s,
                total_s: total_start.elapsed().as_secs_f64(),
            },
        })
    }

    /// Builds a PQ HNSW index sharing an already-allocated `Arc<RowMatrix<f32>>` without
    /// duplicating raw `f32` storage when `storage_mode == QuantizedStorageMode::WithExactRerank`.
    pub fn build_with_shared_raw(
        base: Arc<RowMatrix<f32>>,
        metric: Metric,
        config: HnswConfig,
        pq_config: PqConfig,
        build_mode: QuantizedBuildMode,
        storage_mode: QuantizedStorageMode,
    ) -> Result<Self> {
        let mut idx = Self::build(
            &base,
            metric,
            config,
            pq_config,
            build_mode,
            QuantizedStorageMode::QuantizedOnly,
        )?;
        if storage_mode == QuantizedStorageMode::WithExactRerank {
            idx.raw_base = Some(base);
            idx.storage_mode = QuantizedStorageMode::WithExactRerank;
        }
        Ok(idx)
    }

    /// Builds a memory-first [`HnswPqIndex`] directly from an `.fvecs` file using two-pass
    /// streaming training/encoding and `QuantizedDirect` graph construction.
    ///
    /// Never materializes the full `RowMatrix<f32>` in memory, producing codebooks, codes, and
    /// `graph_fingerprint()` bit-identical to in-memory `QuantizedDirect`.
    pub fn build_streaming_from_fvecs<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        config: HnswConfig,
        pq_config: PqConfig,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        let total_start = Instant::now();
        config.validate()?;
        pq_config.validate()?;

        let (vectors, train_time_s, encode_time_s) =
            PqVectors::from_fvecs_streaming_timed(path, metric, &pq_config, max_rows)?;
        let t_graph = Instant::now();
        let sdc_source = vectors.into_with_sdc();
        let graph = CompactHnswGraph::build_with_source(&sdc_source, config)?;
        let vectors = sdc_source.into_pq_vectors();
        let graph_build_time_s = t_graph.elapsed().as_secs_f64();

        Ok(Self {
            graph,
            vectors,
            raw_base: None,
            build_mode: QuantizedBuildMode::QuantizedDirect,
            storage_mode: QuantizedStorageMode::QuantizedOnly,
            build_timings: PqBuildTimings {
                train_time_s,
                encode_time_s,
                graph_build_time_s,
                total_s: total_start.elapsed().as_secs_f64(),
            },
        })
    }

    impl_quantized_index_common!(HnswPqIndex, PqVectors, PqQueryState);

    /// Writes a deterministic quantized-only snapshot to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        write_index_snapshot(
            writer,
            PQ_SNAPSHOT_MAGIC,
            self.build_mode,
            self.build_timings.to_quantized_timings(),
            &self.graph,
            |w| self.vectors.write_to(w),
        )
    }

    /// Reads and validates a quantized-only snapshot from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let (graph, vectors, build_mode, timings) = read_index_snapshot(
            reader,
            PQ_SNAPSHOT_MAGIC,
            "HnswPqIndex",
            PqVectors::read_from,
        )?;
        Ok(Self {
            graph,
            vectors,
            raw_base: None,
            build_mode,
            storage_mode: QuantizedStorageMode::QuantizedOnly,
            build_timings: PqBuildTimings {
                train_time_s: timings.calibration_or_train_s,
                encode_time_s: timings.encode_s,
                graph_build_time_s: timings.graph_build_s,
                total_s: timings.total_s,
            },
        })
    }

    /// Reference to the trained [`PqCodebook`].
    #[inline]
    #[must_use]
    pub fn codebook(&self) -> &PqCodebook {
        self.vectors.codebook()
    }

    /// PQ training configuration used to construct this index.
    #[inline]
    #[must_use]
    pub fn pq_config(&self) -> &PqConfig {
        self.vectors.codebook().config()
    }

    /// Phase-level PQ build timings recorded during construction.
    #[inline]
    #[must_use]
    pub fn build_timings(&self) -> PqBuildTimings {
        self.build_timings
    }

    /// Phase-level build timings converted to [`QuantizedBuildTimings`].
    #[inline]
    #[must_use]
    pub fn quantized_build_timings(&self) -> QuantizedBuildTimings {
        self.build_timings.to_quantized_timings()
    }

    /// Number of PQ subspaces (`M_pq`).
    #[inline]
    #[must_use]
    pub fn m_pq(&self) -> usize {
        self.vectors.m_pq()
    }

    /// Steady-state payload memory profile across codes, codebook, graph, and optional raw vectors.
    #[must_use]
    pub fn memory_profile(&self) -> QuantizedMemoryProfile {
        QuantizedMemoryProfile::from_parts(
            self.graph.num_vectors(),
            self.vectors.codes_memory_bytes(),
            self.vectors.codebook().memory_bytes(),
            self.vectors.aux_memory_bytes(),
            self.graph.node_meta_bytes(),
            self.graph.layer0_edges_bytes(),
            self.graph.upper_layers_edges_bytes(),
            self.raw_base.as_ref().map_or(0, |m| m.memory_bytes()),
        )
    }

    /// Allocates reusable search scratch sized for `max_ef` and pre-sized for `[m_pq * 256]` ADC tables.
    #[must_use]
    pub fn new_search_scratch(&self, max_ef: usize) -> QuantizedSearchScratch<PqQueryState> {
        let mut scratch = QuantizedSearchScratch::new(
            self.graph.num_vectors(),
            max_ef,
            self.graph.config().effective_m_max0(),
        );
        scratch.query_state = PqQueryState::with_subspaces(self.m_pq());
        scratch
    }
}

#[inline]
fn sub_l2_sq_f32_f64(a: &[f32], b: &[f64]) -> f64 {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;

    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks.iter()) {
        let d0 = f64::from(ca[0]) - cb[0];
        let d1 = f64::from(ca[1]) - cb[1];
        let d2 = f64::from(ca[2]) - cb[2];
        let d3 = f64::from(ca[3]) - cb[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&va, &vb) in a_rem.iter().zip(b_rem.iter()) {
        let d = f64::from(va) - vb;
        sum += d * d;
    }
    sum
}

#[inline]
fn sub_l2_sq_4_f32_f64(
    x: &[f32],
    c0: &[f64],
    c1: &[f64],
    c2: &[f64],
    c3: &[f64],
) -> (f64, f64, f64, f64) {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;

    for (j, &xj_f32) in x.iter().enumerate() {
        let xj = f64::from(xj_f32);
        let d0 = xj - c0[j];
        let d1 = xj - c1[j];
        let d2 = xj - c2[j];
        let d3 = xj - c3[j];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    (s0, s1, s2, s3)
}

#[inline]
fn sub_l2_sq_f32_f32_in_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;

    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks.iter()) {
        let d0 = f64::from(ca[0]) - f64::from(cb[0]);
        let d1 = f64::from(ca[1]) - f64::from(cb[1]);
        let d2 = f64::from(ca[2]) - f64::from(cb[2]);
        let d3 = f64::from(ca[3]) - f64::from(cb[3]);
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&va, &vb) in a_rem.iter().zip(b_rem.iter()) {
        let d = f64::from(va) - f64::from(vb);
        sum += d * d;
    }
    sum
}

#[inline]
fn sub_l2_sq_4_f32_f32_in_f64(
    x: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> (f64, f64, f64, f64) {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;

    for (j, &xj_f32) in x.iter().enumerate() {
        let xj = f64::from(xj_f32);
        let d0 = xj - f64::from(c0[j]);
        let d1 = xj - f64::from(c1[j]);
        let d2 = xj - f64::from(c2[j]);
        let d3 = xj - f64::from(c3[j]);
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    (s0, s1, s2, s3)
}

#[inline]
fn sub_l2_sq_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;

    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks.iter()) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&va, &vb) in a_rem.iter().zip(b_rem.iter()) {
        let d = va - vb;
        sum += d * d;
    }
    if sum.is_finite() {
        sum
    } else {
        sub_l2_sq_f32_f32_in_f64(a, b) as f32
    }
}

#[inline]
fn sub_dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;

    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks.iter()) {
        s0 += ca[0] * cb[0];
        s1 += ca[1] * cb[1];
        s2 += ca[2] * cb[2];
        s3 += ca[3] * cb[3];
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&va, &vb) in a_rem.iter().zip(b_rem.iter()) {
        sum += va * vb;
    }
    if sum.is_finite() {
        sum
    } else {
        let mut f64_sum = 0.0_f64;
        for (&va, &vb) in a.iter().zip(b.iter()) {
            f64_sum += f64::from(va) * f64::from(vb);
        }
        f64_sum as f32
    }
}
