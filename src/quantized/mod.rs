//! Quantized vector search engines (`SQ8` and `PQ`) over compact lifetime-free HNSW graphs.
//!
//! Strictly isolated from [`crate::hnsw::HnswIndex`] so all Phase 1–5 `f32` graph construction
//! and search paths remain untouched (`PRESERVING` non-regression guarantee).

pub mod graph;
pub mod pq;
pub mod sq8;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::dataset::{inspect_fvecs_header, stream_fvecs_chunks, stream_fvecs_chunks_max_rows};
use crate::distance::{inner_product, l2_squared, Metric};
use crate::error::{Result, VectorSearchError};
use crate::json_util::ensure_parent_dir;
use crate::matrix::RowMatrix;

pub use graph::{
    results_to_knn_search_result, CompactHnswGraph, Neighbor, QuantizedBuildMode,
    QuantizedBuildTimings, QuantizedDistanceSource, QuantizedMemoryProfile, QuantizedSearchScratch,
    QuantizedStorageMode, SearchMetrics, TimedQuantizedSearch,
};
pub use pq::{
    pq_adc_distance, pq_sdc_distance, HnswPqIndex, PqBuildTimings, PqCodebook, PqConfig,
    PqQueryState, PqTrainer, PqVectors, PqWithSdc, PQ_DEFAULT_MAX_LLOYD_ITERS,
    PQ_DEFAULT_MAX_TRAIN_VECTORS, PQ_DEFAULT_SEED, PQ_K_CENTROIDS,
};
pub use sq8::{
    inner_product_sq8, inner_product_sq8_symmetric, l2_squared_sq8_expanded,
    l2_squared_sq8_shifted, l2_squared_sq8_symmetric, l2_squared_sq8_symmetric_from_scales,
    HnswSq8Index, Sq8KernelKind, Sq8Quantizer, Sq8QueryState, Sq8Vectors,
};

pub(crate) const IO_BUF_CAPACITY: usize = 256 * 1024;
pub(crate) const STREAMING_CHUNK_ROWS: usize = 4_096;

/// Validates that `base` is non-empty, fits within `u32` node IDs, and contains only finite floats,
/// returning `(rows, cols)`.
pub(crate) fn validate_base_matrix(base: &RowMatrix<f32>) -> Result<(usize, usize)> {
    let rows = base.rows();
    let cols = base.cols();
    if rows == 0 || cols == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "base_vectors cannot be empty (rows and cols must be > 0)".to_string(),
        ));
    }
    if rows > u32::MAX as usize {
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors row count {rows} exceeds u32::MAX"
        )));
    }
    if let Some(pos) = base.as_slice().iter().position(|v| !v.is_finite()) {
        let row = pos / cols;
        let col = pos % cols;
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors contains non-finite value (NaN/Inf) at ({row}, {col})"
        )));
    }
    Ok((rows, cols))
}

/// Validates a raw base matrix before attaching it to an already-built quantized index.
pub(crate) fn validate_attached_raw_base(
    raw_base: &RowMatrix<f32>,
    expected_rows: usize,
    expected_dim: usize,
) -> Result<()> {
    validate_base_matrix(raw_base)?;
    if raw_base.rows() != expected_rows {
        return Err(VectorSearchError::InvalidArgument(format!(
            "raw_base row count ({}) does not match index vector count ({expected_rows})",
            raw_base.rows()
        )));
    }
    if raw_base.cols() != expected_dim {
        return Err(VectorSearchError::DimensionMismatch {
            expected: expected_dim,
            actual: raw_base.cols(),
        });
    }
    Ok(())
}

/// Inspects an `.fvecs` header for streaming calibration/training and computes `(num_vecs, dim, chunk_rows)`.
pub(crate) fn inspect_streaming_fvecs(
    path: &Path,
    max_rows: Option<usize>,
) -> Result<(usize, usize, usize)> {
    if max_rows == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_rows must be positive (> 0) when specified".to_string(),
        ));
    }
    let header = inspect_fvecs_header(path)?;
    let num_vecs = max_rows.map_or(header.num_rows, |limit| header.num_rows.min(limit));
    if num_vecs > u32::MAX as usize {
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors row count {num_vecs} exceeds u32::MAX"
        )));
    }
    let chunk_rows = STREAMING_CHUNK_ROWS
        .min((16 * 1024 * 1024 / (header.dim * 4)).max(1))
        .min(num_vecs)
        .max(1);
    Ok((num_vecs, header.dim, chunk_rows))
}

/// Streams up to `max_rows` records from `path` in chunks of `chunk_rows` rows.
pub(crate) fn stream_fvecs_bounded<F>(
    path: &Path,
    max_rows: Option<usize>,
    chunk_rows: usize,
    on_chunk: F,
) -> Result<()>
where
    F: FnMut(usize, usize, &[f32]),
{
    match max_rows {
        None => {
            stream_fvecs_chunks(path, chunk_rows, on_chunk)?;
        }
        Some(_) => {
            stream_fvecs_chunks_max_rows(path, max_rows, chunk_rows, on_chunk)?;
        }
    }
    Ok(())
}

/// Computes exact canonical minimization sort key (`L2^2` or `-dot`) between two `f32` slices.
#[inline]
pub(crate) fn compute_exact_sort_key(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2Squared | Metric::Euclidean => l2_squared(a, b),
        Metric::InnerProduct => {
            let neg_dot = -inner_product(a, b);
            if neg_dot == 0.0 {
                0.0
            } else {
                neg_dot
            }
        }
    }
}

/// Converts an internal minimization sort key (`L2^2` or `-dot`) into the public `Metric` output.
#[inline]
pub(crate) fn format_metric_output_distance(metric: Metric, sort_key: f32) -> f32 {
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

#[inline]
pub(crate) fn metric_to_u8(metric: Metric) -> u8 {
    match metric {
        Metric::L2Squared => 0,
        Metric::Euclidean => 1,
        Metric::InnerProduct => 2,
    }
}

#[inline]
pub(crate) fn metric_from_u8(tag: u8) -> Result<Metric> {
    match tag {
        0 => Ok(Metric::L2Squared),
        1 => Ok(Metric::Euclidean),
        2 => Ok(Metric::InnerProduct),
        _ => Err(VectorSearchError::CorruptData(format!(
            "Invalid serialized Metric tag: {tag}"
        ))),
    }
}

#[inline]
pub(crate) fn build_mode_to_u8(mode: QuantizedBuildMode) -> u8 {
    match mode {
        QuantizedBuildMode::Float32GraphThenQuantize => 0,
        QuantizedBuildMode::QuantizedDirect => 1,
    }
}

#[inline]
pub(crate) fn build_mode_from_u8(tag: u8) -> Result<QuantizedBuildMode> {
    match tag {
        0 => Ok(QuantizedBuildMode::Float32GraphThenQuantize),
        1 => Ok(QuantizedBuildMode::QuantizedDirect),
        other => Err(VectorSearchError::CorruptData(format!(
            "Invalid QuantizedBuildMode tag in snapshot: {other}"
        ))),
    }
}

#[inline]
pub(crate) fn read_magic<R: Read>(
    reader: &mut R,
    expected_magic: &[u8; 8],
    label: &str,
) -> Result<()> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != expected_magic {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid {label} snapshot magic header"
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn read_usize_u64<R: Read>(reader: &mut R, label: &str) -> Result<usize> {
    let raw = read_u64(reader)?;
    usize::try_from(raw)
        .map_err(|_| VectorSearchError::CorruptData(format!("{label} ({raw}) overflows usize")))
}

pub(crate) fn write_snapshot_to_path<P, F>(path: P, write_fn: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut BufWriter<File>) -> Result<()>,
{
    let path_ref = path.as_ref();
    ensure_parent_dir(path_ref)?;
    let file = File::create(path_ref)?;
    let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);
    write_fn(&mut writer)?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn read_snapshot_from_path<P, T, F>(path: P, read_fn: F) -> Result<T>
where
    P: AsRef<Path>,
    F: FnOnce(&mut BufReader<File>) -> Result<T>,
{
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);
    read_fn(&mut reader)
}

pub(crate) fn write_index_snapshot<W: Write, F>(
    writer: &mut W,
    magic: &[u8; 8],
    build_mode: QuantizedBuildMode,
    timings: QuantizedBuildTimings,
    graph: &CompactHnswGraph,
    write_vectors: F,
) -> Result<()>
where
    F: FnOnce(&mut W) -> Result<()>,
{
    writer.write_all(magic)?;
    write_u8(writer, build_mode_to_u8(build_mode))?;
    write_f64(writer, timings.calibration_or_train_s)?;
    write_f64(writer, timings.encode_s)?;
    write_f64(writer, timings.graph_build_s)?;
    write_f64(writer, timings.total_s)?;
    graph.write_to(writer)?;
    write_vectors(writer)
}

pub(crate) fn read_index_snapshot<R: Read, V: QuantizedDistanceSource, F>(
    reader: &mut R,
    expected_magic: &[u8; 8],
    label: &str,
    read_vectors: F,
) -> Result<(
    CompactHnswGraph,
    V,
    QuantizedBuildMode,
    QuantizedBuildTimings,
)>
where
    F: FnOnce(&mut R) -> Result<V>,
{
    read_magic(reader, expected_magic, label)?;
    let build_mode = build_mode_from_u8(read_u8(reader)?)?;
    let calibration_or_train_s = read_f64(reader)?;
    let encode_s = read_f64(reader)?;
    let graph_build_s = read_f64(reader)?;
    let total_s = read_f64(reader)?;
    if !calibration_or_train_s.is_finite()
        || !encode_s.is_finite()
        || !graph_build_s.is_finite()
        || !total_s.is_finite()
    {
        return Err(VectorSearchError::CorruptData(format!(
            "Non-finite build timings in {label} snapshot"
        )));
    }
    let graph = CompactHnswGraph::read_from(reader)?;
    let vectors = read_vectors(reader)?;
    if graph.num_vectors() != vectors.len() {
        return Err(VectorSearchError::CorruptData(format!(
            "Snapshot node count mismatch: graph={}, vectors={}",
            graph.num_vectors(),
            vectors.len()
        )));
    }
    if graph.metric() != vectors.metric() {
        return Err(VectorSearchError::CorruptData(format!(
            "Snapshot metric mismatch: graph={}, vectors={}",
            graph.metric(),
            vectors.metric()
        )));
    }
    Ok((
        graph,
        vectors,
        build_mode,
        QuantizedBuildTimings {
            calibration_or_train_s,
            encode_s,
            graph_build_s,
            total_s,
        },
    ))
}

#[inline]
pub(crate) fn write_u8<W: Write>(w: &mut W, val: u8) -> Result<()> {
    w.write_all(&[val])?;
    Ok(())
}

#[inline]
pub(crate) fn write_u32<W: Write>(w: &mut W, val: u32) -> Result<()> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

#[inline]
pub(crate) fn write_u64<W: Write>(w: &mut W, val: u64) -> Result<()> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

#[inline]
pub(crate) fn write_f64<W: Write>(w: &mut W, val: f64) -> Result<()> {
    w.write_all(&val.to_bits().to_le_bytes())?;
    Ok(())
}

pub(crate) fn write_u32_slice<W: Write>(w: &mut W, slice: &[u32]) -> Result<()> {
    write_u64(w, slice.len() as u64)?;
    for &v in slice {
        write_u32(w, v)?;
    }
    Ok(())
}

pub(crate) fn write_f32_slice<W: Write>(w: &mut W, slice: &[f32]) -> Result<()> {
    write_u64(w, slice.len() as u64)?;
    for &v in slice {
        w.write_all(&v.to_bits().to_le_bytes())?;
    }
    Ok(())
}

#[inline]
pub(crate) fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0_u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
pub(crate) fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0_u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[inline]
pub(crate) fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0_u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[inline]
pub(crate) fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0_u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_bits(u64::from_le_bytes(buf)))
}

pub(crate) fn read_u32_vec<R: Read>(r: &mut R, max_len: usize) -> Result<Vec<u32>> {
    let len = read_usize_u64(r, "Serialized u32 vector length")?;
    if len > max_len {
        return Err(VectorSearchError::CorruptData(format!(
            "Serialized u32 vector length {len} exceeds maximum allowed {max_len}"
        )));
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_u32(r)?);
    }
    Ok(out)
}

pub(crate) fn read_f32_vec<R: Read>(r: &mut R, expected_len: usize) -> Result<Vec<f32>> {
    let len = read_usize_u64(r, "Serialized f32 vector length")?;
    if len != expected_len {
        return Err(VectorSearchError::CorruptData(format!(
            "Serialized f32 vector length mismatch: expected {expected_len}, got {len}"
        )));
    }
    let mut out = Vec::with_capacity(len);
    let mut buf = [0_u8; 4];
    for _ in 0..len {
        r.read_exact(&mut buf)?;
        out.push(f32::from_bits(u32::from_le_bytes(buf)));
    }
    Ok(out)
}

/// Implements common accessors, snapshot file wrappers, raw-base transitions, and search
/// delegation methods shared by [`HnswSq8Index`] and [`HnswPqIndex`].
macro_rules! impl_quantized_index_common {
    ($index_ty:ty, $vec_ty:ty, $query_state_ty:ty) => {
        /// Drops retained raw `f32` vectors (if any) and transitions storage mode to [`QuantizedStorageMode::QuantizedOnly`].
        #[must_use]
        pub fn into_quantized_only(mut self) -> Self {
            self.raw_base = None;
            self.storage_mode = QuantizedStorageMode::QuantizedOnly;
            self
        }

        /// Attaches shared raw `f32` vectors after validating shape and finiteness, enabling exact rerank.
        pub fn with_raw_base(mut self, raw_base: Arc<RowMatrix<f32>>) -> Result<Self> {
            super::validate_attached_raw_base(&raw_base, self.num_vectors(), self.dim())?;
            self.raw_base = Some(raw_base);
            self.storage_mode = QuantizedStorageMode::WithExactRerank;
            Ok(self)
        }

        /// Writes a deterministic quantized-only binary snapshot to `path`.
        pub fn write_snapshot<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            super::write_snapshot_to_path(path, |writer| self.write_to(writer))
        }

        /// Loads a quantized-only binary snapshot from `path` (holding zero raw `f32` vectors).
        pub fn read_snapshot<P: AsRef<Path>>(path: P) -> Result<Self> {
            super::read_snapshot_from_path(path, Self::read_from)
        }

        /// Reference to the underlying [`CompactHnswGraph`].
        #[inline]
        #[must_use]
        pub fn graph(&self) -> &CompactHnswGraph {
            &self.graph
        }

        /// Reference to the quantized vector store.
        #[inline]
        #[must_use]
        pub fn vectors(&self) -> &$vec_ty {
            &self.vectors
        }

        /// Reference to optional retained raw `f32` base vectors (`None` in `QuantizedOnly` mode).
        #[inline]
        #[must_use]
        pub fn raw_base(&self) -> Option<&RowMatrix<f32>> {
            self.raw_base.as_deref()
        }

        /// Graph build precision mode used to construct this index.
        #[inline]
        #[must_use]
        pub fn build_mode(&self) -> QuantizedBuildMode {
            self.build_mode
        }

        /// Current raw-vector storage mode (`QuantizedOnly` or `WithExactRerank`).
        #[inline]
        #[must_use]
        pub fn storage_mode(&self) -> QuantizedStorageMode {
            self.storage_mode
        }

        /// Number of indexed vectors (`N`).
        #[inline]
        #[must_use]
        pub fn num_vectors(&self) -> usize {
            self.graph.num_vectors()
        }

        /// Vector dimensionality (`D`).
        #[inline]
        #[must_use]
        pub fn dim(&self) -> usize {
            self.vectors.dim()
        }

        /// Configured distance metric.
        #[inline]
        #[must_use]
        pub fn metric(&self) -> Metric {
            self.graph.metric()
        }

        /// Maximum layer height in the graph.
        #[inline]
        #[must_use]
        pub fn max_layer(&self) -> usize {
            self.graph.max_layer()
        }

        /// Global entry point node ID, or `None` if empty.
        #[inline]
        #[must_use]
        pub fn entry_point(&self) -> Option<usize> {
            self.graph.entry_point()
        }

        /// Canonical 64-bit FNV-1a graph topology fingerprint matching `HnswIndex::graph_fingerprint()`.
        #[inline]
        #[must_use]
        pub fn graph_fingerprint(&self) -> u64 {
            self.graph.graph_fingerprint()
        }

        /// Executes an untimed warmup query to populate scratch buffer capacities and CPU caches.
        pub fn warmup(
            &self,
            query: &[f32],
            k: usize,
            ef_search: usize,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<()> {
            let _ = self.search_knn_with_scratch(query, k, ef_search, scratch)?;
            Ok(())
        }

        /// Pure quantized top-`k` search reusing `scratch`.
        pub fn search_knn_with_scratch(
            &self,
            query: &[f32],
            k: usize,
            ef_search: usize,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<Vec<Neighbor>> {
            self.graph
                .search_knn_with_scratch(&self.vectors, query, k, ef_search, scratch)
        }

        /// Quantized search with optional exact top-`R` reranking using stored raw vectors (`WithExactRerank`).
        pub fn search_knn_rerank_with_scratch(
            &self,
            query: &[f32],
            k: usize,
            ef_search: usize,
            rerank_top_r: Option<usize>,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<Vec<Neighbor>> {
            let mut out = Vec::with_capacity(k.min(self.num_vectors()));
            self.graph.search_knn_rerank_opt_into_with_scratch(
                &self.vectors,
                self.raw_base.as_deref(),
                query,
                k,
                ef_search,
                rerank_top_r,
                scratch,
                &mut out,
            )?;
            Ok(out)
        }

        /// Quantized search with optional exact top-`R` reranking against a caller-supplied `raw_base`
        /// without storing `raw_base` inside the index.
        pub fn search_knn_rerank_with_base_scratch(
            &self,
            raw_base: &RowMatrix<f32>,
            query: &[f32],
            k: usize,
            ef_search: usize,
            rerank_top_r: Option<usize>,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<Vec<Neighbor>> {
            self.graph.search_knn_rerank_with_scratch(
                &self.vectors,
                raw_base,
                query,
                k,
                ef_search,
                rerank_top_r,
                scratch,
            )
        }

        /// Batch sequential search returning `Vec<Vec<Neighbor>>`.
        pub fn search_batch_with_scratch(
            &self,
            queries: &RowMatrix<f32>,
            k: usize,
            ef_search: usize,
            rerank_top_r: Option<usize>,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<Vec<Vec<Neighbor>>> {
            self.graph.search_batch_with_scratch(
                &self.vectors,
                self.raw_base.as_deref(),
                queries,
                k,
                ef_search,
                rerank_top_r,
                scratch,
            )
        }

        /// Timed sequential batch search returning `Vec<Vec<Neighbor>>`, per-query latencies, and [`super::SearchMetrics`].
        pub fn search_batch_timed_with_scratch(
            &self,
            queries: &RowMatrix<f32>,
            k: usize,
            ef_search: usize,
            rerank_top_r: Option<usize>,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<TimedQuantizedSearch> {
            self.graph.search_batch_timed_with_scratch(
                &self.vectors,
                self.raw_base.as_deref(),
                queries,
                k,
                ef_search,
                rerank_top_r,
                scratch,
            )
        }

        /// Timed sequential batch search with caller-supplied `raw_base` (for reranking without owning raw vectors).
        pub fn search_batch_with_base_timed_with_scratch(
            &self,
            raw_base: &RowMatrix<f32>,
            queries: &RowMatrix<f32>,
            k: usize,
            ef_search: usize,
            rerank_top_r: Option<usize>,
            scratch: &mut QuantizedSearchScratch<$query_state_ty>,
        ) -> Result<TimedQuantizedSearch> {
            self.graph.search_batch_timed_with_scratch(
                &self.vectors,
                Some(raw_base),
                queries,
                k,
                ef_search,
                rerank_top_r,
                scratch,
            )
        }
    };
}

pub(crate) use impl_quantized_index_common;
