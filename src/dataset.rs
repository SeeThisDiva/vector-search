//! Texmex `.fvecs` / `.ivecs` binary serialization, deterministic synthetic SIFT generator,
//! and public SIFT (`siftsmall` & `SIFT1M`) dataset downloader with offline fallback.

use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::distance::l2_squared;
use crate::error::{Result, VectorSearchError};
use crate::filter::DenseBitset;
use crate::matrix::RowMatrix;
use crate::rng::DeterministicRng;

const IO_BUF_CAPACITY: usize = 256 * 1024;

const SIFTSMALL_URLS: &[&str] = &[
    "https://huggingface.co/datasets/vecdata/siftsmall/resolve/main/siftsmall.tar.gz",
    "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
];

const SIFT1M_BASE_URL: &str =
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs";
const SIFT1M_QUERY_URL: &str =
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs";
const SIFT1M_GT_URL: &str =
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs";
const SIFT1M_ARCHIVE_URL: &str = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz";

/// Standard public SIFT benchmark corpus presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiftPreset {
    /// Texmex `siftsmall`: 10,000 base 128-d vectors, 100 query 128-d vectors, 100x100 GT.
    SiftSmall,
    /// Texmex `SIFT1M`: 1,000,000 base 128-d vectors, 10,000 query 128-d vectors, 10,000x100 GT.
    Sift1M,
}

impl SiftPreset {
    /// Expected number of base vectors in the authentic corpus.
    #[must_use]
    pub const fn expected_num_base(self) -> usize {
        match self {
            Self::SiftSmall => 10_000,
            Self::Sift1M => 1_000_000,
        }
    }

    /// Expected number of query vectors in the authentic corpus.
    #[must_use]
    pub const fn expected_num_query(self) -> usize {
        match self {
            Self::SiftSmall => 100,
            Self::Sift1M => 10_000,
        }
    }

    /// Expected vector dimensionality (128 for SIFT descriptors).
    #[must_use]
    pub const fn expected_dim(self) -> usize {
        128
    }

    /// Expected ground-truth neighbor width per query (100 in Texmex SIFT).
    #[must_use]
    pub const fn expected_gt_k(self) -> usize {
        100
    }

    /// Canonical name string (`"siftsmall"` or `"sift1m"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SiftSmall => "siftsmall",
            Self::Sift1M => "sift1m",
        }
    }
}

impl fmt::Display for SiftPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provenance metadata recording whether vectors came from an authentic SIFT corpus,
/// user files, or deterministic seeded synthetic generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetSource {
    /// Verified authentic `siftsmall` (10k x 128-d) Texmex corpus.
    AuthenticSiftSmall { dir: PathBuf },
    /// Verified authentic `SIFT1M` (1M x 128-d) Texmex corpus.
    AuthenticSift1M { dir: PathBuf },
    /// Explicitly requested deterministic synthetic clustered Gaussian vectors.
    Synthetic { seed: u64 },
    /// Offline/unreachable fallback to deterministic synthetic vectors when SIFT fetch could not run.
    SyntheticFallback {
        preset: SiftPreset,
        reason: String,
        seed: u64,
    },
    /// User-supplied `.fvecs` (and optional `.ivecs`) files.
    CustomFiles {
        base_path: PathBuf,
        query_path: PathBuf,
    },
}

impl DatasetSource {
    /// Short identifier suitable for machine-readable reports.
    #[must_use]
    pub fn kind_tag(&self) -> &'static str {
        match self {
            Self::AuthenticSiftSmall { .. } => "authentic_siftsmall",
            Self::AuthenticSift1M { .. } => "authentic_sift1m",
            Self::Synthetic { .. } => "synthetic",
            Self::SyntheticFallback { .. } => "synthetic_fallback",
            Self::CustomFiles { .. } => "custom_files",
        }
    }

    /// Returns `true` when backed by verified authentic SIFT vectors.
    #[must_use]
    pub const fn is_authentic_sift(&self) -> bool {
        matches!(
            self,
            Self::AuthenticSiftSmall { .. } | Self::AuthenticSift1M { .. }
        )
    }
}

impl fmt::Display for DatasetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthenticSiftSmall { dir } => {
                write!(f, "Authentic Texmex siftsmall ({})", dir.display())
            }
            Self::AuthenticSift1M { dir } => {
                write!(f, "Authentic Texmex SIFT1M ({})", dir.display())
            }
            Self::Synthetic { seed } => {
                write!(f, "Deterministic Synthetic SIFT Manifold (seed={seed})")
            }
            Self::SyntheticFallback {
                preset,
                reason,
                seed,
            } => write!(
                f,
                "Deterministic Synthetic Fallback for {preset} (seed={seed}; reason: {reason})"
            ),
            Self::CustomFiles {
                base_path,
                query_path,
            } => write!(
                f,
                "Custom .fvecs (base={}, query={})",
                base_path.display(),
                query_path.display()
            ),
        }
    }
}

/// Loaded benchmark dataset bundle containing base vectors, query vectors, and optional reference GT.
#[derive(Debug, Clone)]
pub struct LoadedDataset {
    pub base: RowMatrix<f32>,
    pub query: RowMatrix<f32>,
    pub official_ground_truth: Option<RowMatrix<i32>>,
    pub source: DatasetSource,
    pub load_time_s: f64,
}

/// Reads a standard Texmex `.fvecs` little-endian binary file into a `RowMatrix<f32>`.
///
/// Record layout:
/// - 4-byte signed 32-bit integer `d` (little-endian)
/// - `d` IEEE 754 `f32` values (little-endian, `4 * d` bytes)
///
/// An empty (0-byte) file yields an empty `(0, 0)` matrix.
pub fn read_fvecs<P: AsRef<Path>>(path: P) -> Result<RowMatrix<f32>> {
    read_fvecs_max_rows(path, None)
}

/// Reads up to `max_rows` records from a standard Texmex `.fvecs` file after validating
/// full-file record alignment, avoiding allocating the entire file when only a prefix is needed.
pub fn read_fvecs_max_rows<P: AsRef<Path>>(
    path: P,
    max_rows: Option<usize>,
) -> Result<RowMatrix<f32>> {
    if max_rows == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_rows must be positive (> 0) when specified".to_string(),
        ));
    }

    let path_ref = path.as_ref();
    let meta = fs::metadata(path_ref)?;
    let file_size = meta.len();

    if file_size == 0 {
        return Ok(RowMatrix::empty());
    }
    if file_size < 4 {
        return Err(VectorSearchError::CorruptData(format!(
            "Corrupt .fvecs file (size {file_size} < 4 bytes): {}",
            path_ref.display()
        )));
    }

    let file = File::open(path_ref)?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);

    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;
    let first_dim_i32 = i32::from_le_bytes(header);
    if first_dim_i32 <= 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid dimension {first_dim_i32} in .fvecs file: {}",
            path_ref.display()
        )));
    }
    let dim = first_dim_i32 as usize;
    let record_bytes = 4_u64
        .checked_add((dim as u64).checked_mul(4).ok_or_else(|| {
            VectorSearchError::CorruptData("Dimension byte size overflow".to_string())
        })?)
        .ok_or_else(|| VectorSearchError::CorruptData("Record byte size overflow".to_string()))?;

    if file_size % record_bytes != 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "File size {file_size} is not a multiple of record size {record_bytes} (dim={dim}) in {}",
            path_ref.display()
        )));
    }

    let num_vecs_u64 = file_size / record_bytes;
    let total_vecs = usize::try_from(num_vecs_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!("Vector count {num_vecs_u64} exceeds usize"))
    })?;
    let num_vecs = match max_rows {
        Some(limit) => total_vecs.min(limit),
        None => total_vecs,
    };
    let total_floats = num_vecs.checked_mul(dim).ok_or_else(|| {
        VectorSearchError::CorruptData("Total float count overflows usize".to_string())
    })?;

    let mut data = Vec::with_capacity(total_floats);
    let mut row_bytes = vec![0_u8; dim * 4];

    // Read payload of record 0 (header already read)
    reader.read_exact(&mut row_bytes)?;
    for &chunk in row_bytes.as_chunks::<4>().0 {
        data.push(f32::from_le_bytes(chunk));
    }

    // Read remaining records 1..num_vecs, verifying every record's dimension header
    for rec_idx in 1..num_vecs {
        reader.read_exact(&mut header)?;
        let rec_dim = i32::from_le_bytes(header);
        if rec_dim != first_dim_i32 {
            return Err(VectorSearchError::CorruptData(format!(
                "Inconsistent .fvecs record dimension at row {rec_idx}: expected {dim}, got {rec_dim} in {}",
                path_ref.display()
            )));
        }
        reader.read_exact(&mut row_bytes)?;
        for &chunk in row_bytes.as_chunks::<4>().0 {
            data.push(f32::from_le_bytes(chunk));
        }
    }

    RowMatrix::new(num_vecs, dim, data)
}

/// Writes a `RowMatrix<f32>` into standard Texmex `.fvecs` little-endian binary format.
pub fn write_fvecs<P: AsRef<Path>>(path: P, matrix: &RowMatrix<f32>) -> Result<()> {
    let path_ref = path.as_ref();
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path_ref)?;
    let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

    if matrix.rows() == 0 {
        writer.flush()?;
        return Ok(());
    }

    if matrix.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Cannot write .fvecs with 0 columns and non-zero rows".to_string(),
        ));
    }
    let dim_i32 = i32::try_from(matrix.cols()).map_err(|_| {
        VectorSearchError::InvalidArgument(format!(
            "Dimension {} exceeds i32::MAX for .fvecs header",
            matrix.cols()
        ))
    })?;
    let header = dim_i32.to_le_bytes();

    for row in matrix.iter_rows() {
        writer.write_all(&header)?;
        for &val in row {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Reads a standard Texmex `.ivecs` little-endian binary file into a `RowMatrix<i32>`.
///
/// Record layout:
/// - 4-byte signed 32-bit integer `k` (little-endian)
/// - `k` signed 32-bit integer neighbor IDs (little-endian, `4 * k` bytes)
pub fn read_ivecs<P: AsRef<Path>>(path: P) -> Result<RowMatrix<i32>> {
    read_ivecs_max_rows(path, None)
}

/// Reads up to `max_rows` records from a standard Texmex `.ivecs` file after validating
/// full-file record alignment.
pub fn read_ivecs_max_rows<P: AsRef<Path>>(
    path: P,
    max_rows: Option<usize>,
) -> Result<RowMatrix<i32>> {
    if max_rows == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_rows must be positive (> 0) when specified".to_string(),
        ));
    }

    let path_ref = path.as_ref();
    let meta = fs::metadata(path_ref)?;
    let file_size = meta.len();

    if file_size == 0 {
        return Ok(RowMatrix::empty());
    }
    if file_size < 4 {
        return Err(VectorSearchError::CorruptData(format!(
            "Corrupt .ivecs file (size {file_size} < 4 bytes): {}",
            path_ref.display()
        )));
    }

    let file = File::open(path_ref)?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);

    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;
    let first_k_i32 = i32::from_le_bytes(header);
    if first_k_i32 <= 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid neighbor width {first_k_i32} in .ivecs file: {}",
            path_ref.display()
        )));
    }
    let k = first_k_i32 as usize;
    let record_bytes = 4_u64
        .checked_add((k as u64).checked_mul(4).ok_or_else(|| {
            VectorSearchError::CorruptData("Width byte size overflow".to_string())
        })?)
        .ok_or_else(|| VectorSearchError::CorruptData("Record byte size overflow".to_string()))?;

    if file_size % record_bytes != 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "File size {file_size} is not a multiple of record size {record_bytes} (k={k}) in {}",
            path_ref.display()
        )));
    }

    let num_rows_u64 = file_size / record_bytes;
    let total_rows = usize::try_from(num_rows_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!("Row count {num_rows_u64} exceeds usize"))
    })?;
    let num_rows = match max_rows {
        Some(limit) => total_rows.min(limit),
        None => total_rows,
    };
    let total_ints = num_rows.checked_mul(k).ok_or_else(|| {
        VectorSearchError::CorruptData("Total int32 count overflows usize".to_string())
    })?;

    let mut data = Vec::with_capacity(total_ints);
    let mut row_bytes = vec![0_u8; k * 4];

    reader.read_exact(&mut row_bytes)?;
    for &chunk in row_bytes.as_chunks::<4>().0 {
        data.push(i32::from_le_bytes(chunk));
    }

    for rec_idx in 1..num_rows {
        reader.read_exact(&mut header)?;
        let rec_k = i32::from_le_bytes(header);
        if rec_k != first_k_i32 {
            return Err(VectorSearchError::CorruptData(format!(
                "Inconsistent .ivecs record width at row {rec_idx}: expected {k}, got {rec_k} in {}",
                path_ref.display()
            )));
        }
        reader.read_exact(&mut row_bytes)?;
        for &chunk in row_bytes.as_chunks::<4>().0 {
            data.push(i32::from_le_bytes(chunk));
        }
    }

    RowMatrix::new(num_rows, k, data)
}

/// Writes a `RowMatrix<i32>` into standard Texmex `.ivecs` little-endian binary format.
pub fn write_ivecs<P: AsRef<Path>>(path: P, matrix: &RowMatrix<i32>) -> Result<()> {
    let path_ref = path.as_ref();
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path_ref)?;
    let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

    if matrix.rows() == 0 {
        writer.flush()?;
        return Ok(());
    }

    if matrix.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Cannot write .ivecs with 0 columns and non-zero rows".to_string(),
        ));
    }
    let k_i32 = i32::try_from(matrix.cols()).map_err(|_| {
        VectorSearchError::InvalidArgument(format!(
            "Column count {} exceeds i32::MAX for .ivecs header",
            matrix.cols()
        ))
    })?;
    let header = k_i32.to_le_bytes();

    for row in matrix.iter_rows() {
        writer.write_all(&header)?;
        for &val in row {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Generates deterministic clustered Gaussian SIFT-like `f32` base and query vectors.
///
/// Models realistic multi-cluster descriptor manifolds:
/// 1. `num_clusters` centroids normalized onto the unit hypersphere and scaled by $\sqrt{D} \cdot 10$.
/// 2. `num_base` vectors sampled around assigned cluster centroids ($\sigma = 0.15 \cdot \text{scale}$).
/// 3. `num_query` vectors: 85% sampled near cluster centroids, 15% sampled across wider global spread.
pub fn generate_synthetic_sift(
    num_base: usize,
    num_query: usize,
    dim: usize,
    num_clusters: usize,
    seed: u64,
) -> Result<(RowMatrix<f32>, RowMatrix<f32>)> {
    if num_base == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "num_base must be positive (> 0)".to_string(),
        ));
    }
    if num_query == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "num_query must be positive (> 0)".to_string(),
        ));
    }
    if dim == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "dim must be positive (> 0)".to_string(),
        ));
    }

    let clusters = num_clusters.max(1);
    let mut rng = DeterministicRng::new(seed);
    let scale_factor = (dim as f64).sqrt() * 10.0;
    let cluster_std = scale_factor * 0.15;

    // 1. Cluster centroids on scaled hypersphere
    let mut centroids = vec![0.0_f32; clusters * dim];
    for c in 0..clusters {
        let start = c * dim;
        let slice = &mut centroids[start..start + dim];
        let mut norm_sq = 0.0_f64;
        for val in slice.iter_mut() {
            let sample = rng.next_standard_normal_f64();
            *val = sample as f32;
            norm_sq += sample * sample;
        }
        let norm = if norm_sq > 0.0 { norm_sq.sqrt() } else { 1.0 };
        let factor = (scale_factor / norm) as f32;
        for val in slice.iter_mut() {
            *val *= factor;
        }
    }

    // 2. Base vectors clustered around centroids
    let base_len = num_base.checked_mul(dim).ok_or_else(|| {
        VectorSearchError::InvalidArgument("num_base * dim overflows usize".to_string())
    })?;
    let mut base_data = Vec::with_capacity(base_len);
    for _ in 0..num_base {
        let cid = rng.next_usize(clusters);
        let centroid = &centroids[cid * dim..(cid + 1) * dim];
        for &c_val in centroid {
            let noise = rng.next_normal_f32(0.0, cluster_std);
            base_data.push(c_val + noise);
        }
    }

    // 3. Query vectors (85% near-cluster, 15% wide-spread)
    let query_len = num_query.checked_mul(dim).ok_or_else(|| {
        VectorSearchError::InvalidArgument("num_query * dim overflows usize".to_string())
    })?;
    let mut query_data = Vec::with_capacity(query_len);
    let num_near = ((num_query as f64) * 0.85) as usize;
    let num_wide = num_query - num_near;

    for _ in 0..num_near {
        let cid = rng.next_usize(clusters);
        let centroid = &centroids[cid * dim..(cid + 1) * dim];
        for &c_val in centroid {
            let noise = rng.next_normal_f32(0.0, cluster_std * 0.8);
            query_data.push(c_val + noise);
        }
    }

    for _ in 0..num_wide {
        for _ in 0..dim {
            query_data.push(rng.next_normal_f32(0.0, scale_factor));
        }
    }

    let base = RowMatrix::new(num_base, dim, base_data)?;
    let query = RowMatrix::new(num_query, dim, query_data)?;
    Ok((base, query))
}

/// Loads user-provided `.fvecs` (and optional `.ivecs` ground truth) files from disk.
pub fn load_custom_dataset(
    base_path: &Path,
    query_path: &Path,
    gt_path: Option<&Path>,
) -> Result<LoadedDataset> {
    load_custom_dataset_with_options(base_path, query_path, gt_path, None, None)
}

/// Loads user-provided `.fvecs` (and optional `.ivecs` ground truth) files from disk,
/// reading at most `max_base` base rows and `max_queries` query/GT rows when specified.
pub fn load_custom_dataset_with_options(
    base_path: &Path,
    query_path: &Path,
    gt_path: Option<&Path>,
    max_base: Option<usize>,
    max_queries: Option<usize>,
) -> Result<LoadedDataset> {
    let t0 = Instant::now();
    let base = read_fvecs_max_rows(base_path, max_base)?;
    let query = read_fvecs_max_rows(query_path, max_queries)?;
    if base.rows() == 0 || query.rows() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Custom base and query .fvecs files must be non-empty".to_string(),
        ));
    }
    if base.cols() != query.cols() {
        return Err(VectorSearchError::DimensionMismatch {
            expected: base.cols(),
            actual: query.cols(),
        });
    }
    let official_ground_truth = match gt_path {
        Some(p) => Some(read_ivecs_max_rows(p, max_queries)?),
        None => None,
    };
    Ok(LoadedDataset {
        base,
        query,
        official_ground_truth,
        source: DatasetSource::CustomFiles {
            base_path: base_path.to_path_buf(),
            query_path: query_path.to_path_buf(),
        },
        load_time_s: t0.elapsed().as_secs_f64(),
    })
}

/// Loads an authentic SIFT dataset (`siftsmall` or `SIFT1M`) from `cache_dir`, downloading
/// from public HTTPS/FTP mirrors if missing and `allow_download == true`, and falling back
/// gracefully to deterministic seeded 128-d synthetic vectors if offline or unreachable.
pub fn load_or_download_sift(
    preset: SiftPreset,
    cache_dir: &Path,
    allow_download: bool,
    fallback_seed: u64,
) -> Result<LoadedDataset> {
    load_or_download_sift_with_options(
        preset,
        cache_dir,
        allow_download,
        fallback_seed,
        None,
        None,
        true,
    )
}

/// Loads an authentic SIFT dataset (`siftsmall` or `SIFT1M`) reading at most `max_base` base
/// vectors and `max_queries` query vectors directly from disk so prefix scale benchmarks do not
/// inflate process peak RSS with the full corpus.
pub fn load_or_download_sift_with_options(
    preset: SiftPreset,
    cache_dir: &Path,
    allow_download: bool,
    fallback_seed: u64,
    max_base: Option<usize>,
    max_queries: Option<usize>,
    load_ground_truth: bool,
) -> Result<LoadedDataset> {
    if max_base == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_base must be positive (> 0) when specified".to_string(),
        ));
    }
    if max_queries == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_queries must be positive (> 0) when specified".to_string(),
        ));
    }

    let t0 = Instant::now();

    // 1. Check existing local cache first
    if let Some((base, query, gt, resolved_dir)) = try_load_cached_sift_with_options(
        preset,
        cache_dir,
        max_base,
        max_queries,
        load_ground_truth,
    ) {
        let source = match preset {
            SiftPreset::SiftSmall => DatasetSource::AuthenticSiftSmall { dir: resolved_dir },
            SiftPreset::Sift1M => DatasetSource::AuthenticSift1M { dir: resolved_dir },
        };
        return Ok(LoadedDataset {
            base,
            query,
            official_ground_truth: gt,
            source,
            load_time_s: t0.elapsed().as_secs_f64(),
        });
    }

    // 2. Attempt public mirror download if permitted
    let fallback_reason = if !allow_download {
        "offline mode enabled (--offline) and local SIFT cache not present".to_string()
    } else {
        match download_sift_preset(preset, cache_dir) {
            Ok(()) => {
                if let Some((base, query, gt, resolved_dir)) = try_load_cached_sift_with_options(
                    preset,
                    cache_dir,
                    max_base,
                    max_queries,
                    load_ground_truth,
                ) {
                    let source = match preset {
                        SiftPreset::SiftSmall => {
                            DatasetSource::AuthenticSiftSmall { dir: resolved_dir }
                        }
                        SiftPreset::Sift1M => DatasetSource::AuthenticSift1M { dir: resolved_dir },
                    };
                    return Ok(LoadedDataset {
                        base,
                        query,
                        official_ground_truth: gt,
                        source,
                        load_time_s: t0.elapsed().as_secs_f64(),
                    });
                }
                format!(
                    "downloaded {preset} files failed Texmex integrity/shape validation in {}",
                    cache_dir.display()
                )
            }
            Err(err) => err.to_string(),
        }
    };

    // 3. Graceful offline fallback to deterministic seeded 128-d vectors
    let eff_base = max_base.map_or(preset.expected_num_base(), |lim| {
        preset.expected_num_base().min(lim)
    });
    let eff_query = max_queries.map_or(preset.expected_num_query(), |lim| {
        preset.expected_num_query().min(lim)
    });
    let (base, query) = generate_synthetic_sift(
        eff_base,
        eff_query,
        preset.expected_dim(),
        16,
        fallback_seed,
    )?;

    Ok(LoadedDataset {
        base,
        query,
        official_ground_truth: None,
        source: DatasetSource::SyntheticFallback {
            preset,
            reason: fallback_reason,
            seed: fallback_seed,
        },
        load_time_s: t0.elapsed().as_secs_f64(),
    })
}

fn candidate_sift_dirs(preset: SiftPreset, cache_dir: &Path) -> Vec<(PathBuf, &'static str)> {
    match preset {
        SiftPreset::SiftSmall => vec![
            (cache_dir.join("siftsmall"), "siftsmall"),
            (cache_dir.to_path_buf(), "siftsmall"),
        ],
        SiftPreset::Sift1M => vec![
            (cache_dir.join("sift1m"), "sift"),
            (cache_dir.join("sift"), "sift"),
            (cache_dir.to_path_buf(), "sift"),
        ],
    }
}

type CachedSiftBundle = (
    RowMatrix<f32>,
    RowMatrix<f32>,
    Option<RowMatrix<i32>>,
    PathBuf,
);

fn try_load_cached_sift_with_options(
    preset: SiftPreset,
    cache_dir: &Path,
    max_base: Option<usize>,
    max_queries: Option<usize>,
    load_ground_truth: bool,
) -> Option<CachedSiftBundle> {
    let expected_base_bytes =
        (preset.expected_num_base() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_query_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_gt_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_gt_k() as u64));

    for (dir, prefix) in candidate_sift_dirs(preset, cache_dir) {
        let base_path = dir.join(format!("{prefix}_base.fvecs"));
        let query_path = dir.join(format!("{prefix}_query.fvecs"));
        let gt_path = dir.join(format!("{prefix}_groundtruth.ivecs"));

        if !base_path.is_file() || !query_path.is_file() || !gt_path.is_file() {
            continue;
        }

        // Verify full-file byte sizes on disk match the authentic SIFT preset even when
        // reading only a prefix of rows into memory.
        if fs::metadata(&base_path).map(|m| m.len()).unwrap_or(0) != expected_base_bytes
            || fs::metadata(&query_path).map(|m| m.len()).unwrap_or(0) != expected_query_bytes
            || fs::metadata(&gt_path).map(|m| m.len()).unwrap_or(0) != expected_gt_bytes
        {
            continue;
        }

        let (Ok(base), Ok(query)) = (
            read_fvecs_max_rows(&base_path, max_base),
            read_fvecs_max_rows(&query_path, max_queries),
        ) else {
            continue;
        };

        let gt = if load_ground_truth {
            let Ok(g) = read_ivecs_max_rows(&gt_path, max_queries) else {
                continue;
            };
            Some(g)
        } else {
            None
        };

        if validate_authentic_sift_shapes_with_options(
            preset,
            &base,
            &query,
            gt.as_ref(),
            max_base,
            max_queries,
        ) {
            return Some((base, query, gt, dir));
        }
    }
    None
}

fn validate_authentic_sift_shapes_with_options(
    preset: SiftPreset,
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    gt: Option<&RowMatrix<i32>>,
    max_base: Option<usize>,
    max_queries: Option<usize>,
) -> bool {
    let expected_base_rows = max_base.map_or(preset.expected_num_base(), |lim| {
        preset.expected_num_base().min(lim)
    });
    let expected_query_rows = max_queries.map_or(preset.expected_num_query(), |lim| {
        preset.expected_num_query().min(lim)
    });

    if base.shape() != (expected_base_rows, preset.expected_dim()) {
        return false;
    }
    if query.shape() != (expected_query_rows, preset.expected_dim()) {
        return false;
    }
    if !base.as_slice().iter().all(|v| v.is_finite())
        || !query.as_slice().iter().all(|v| v.is_finite())
    {
        return false;
    }
    if let Some(gt_mat) = gt {
        if gt_mat.shape() != (expected_query_rows, preset.expected_gt_k()) {
            return false;
        }
        let max_id = preset.expected_num_base() as i32;
        if !gt_mat.as_slice().iter().all(|&id| id >= 0 && id < max_id) {
            return false;
        }
    }
    true
}

fn download_sift_preset(preset: SiftPreset, cache_dir: &Path) -> Result<()> {
    fs::create_dir_all(cache_dir)?;

    match preset {
        SiftPreset::SiftSmall => {
            let archive_part = cache_dir.join("siftsmall.tar.gz.part");
            let archive_path = cache_dir.join("siftsmall.tar.gz");
            let mut errors = Vec::new();

            for &url in SIFTSMALL_URLS {
                let _ = fs::remove_file(&archive_part);
                match run_curl_download(url, &archive_part, 180) {
                    Ok(()) => {
                        let _ = fs::remove_file(&archive_path);
                        fs::rename(&archive_part, &archive_path)?;
                        let extract_res = run_tar_extract(&archive_path, cache_dir);
                        let _ = fs::remove_file(&archive_path);
                        match extract_res {
                            Ok(()) => {
                                let _ = fs::remove_file(
                                    cache_dir.join("siftsmall").join("siftsmall_learn.fvecs"),
                                );
                                return Ok(());
                            }
                            Err(e) => errors.push(format!("{url} (extract): {e}")),
                        }
                    }
                    Err(e) => errors.push(format!("{url}: {e}")),
                }
            }
            let _ = fs::remove_file(&archive_part);
            Err(VectorSearchError::DownloadFailed(errors.join("; ")))
        }
        SiftPreset::Sift1M => {
            let target_dir = cache_dir.join("sift1m");
            fs::create_dir_all(&target_dir)?;

            let direct_files = [
                (SIFT1M_BASE_URL, target_dir.join("sift_base.fvecs"), 900),
                (SIFT1M_QUERY_URL, target_dir.join("sift_query.fvecs"), 180),
                (
                    SIFT1M_GT_URL,
                    target_dir.join("sift_groundtruth.ivecs"),
                    180,
                ),
            ];

            let mut direct_ok = true;
            let mut errors = Vec::new();
            for (url, dest, timeout_s) in &direct_files {
                if dest.is_file() {
                    continue;
                }
                let part = dest.with_extension("part");
                let _ = fs::remove_file(&part);
                match run_curl_download(url, &part, *timeout_s) {
                    Ok(()) => {
                        fs::rename(&part, dest)?;
                    }
                    Err(e) => {
                        let _ = fs::remove_file(&part);
                        errors.push(format!("{url}: {e}"));
                        direct_ok = false;
                        break;
                    }
                }
            }

            if direct_ok {
                return Ok(());
            }

            // Fallback to Texmex FTP archive sift.tar.gz
            let archive_part = cache_dir.join("sift.tar.gz.part");
            let archive_path = cache_dir.join("sift.tar.gz");
            let _ = fs::remove_file(&archive_part);
            match run_curl_download(SIFT1M_ARCHIVE_URL, &archive_part, 1200) {
                Ok(()) => {
                    let _ = fs::remove_file(&archive_path);
                    fs::rename(&archive_part, &archive_path)?;
                    let extract_res = run_tar_extract(&archive_path, cache_dir);
                    let _ = fs::remove_file(&archive_path);
                    if extract_res.is_ok() {
                        let _ = fs::remove_file(cache_dir.join("sift").join("sift_learn.fvecs"));
                    }
                    extract_res
                }
                Err(e) => {
                    let _ = fs::remove_file(&archive_part);
                    errors.push(format!("{SIFT1M_ARCHIVE_URL}: {e}"));
                    Err(VectorSearchError::DownloadFailed(errors.join("; ")))
                }
            }
        }
    }
}

fn run_curl_download(url: &str, dest: &Path, max_time_secs: u64) -> Result<()> {
    let max_time = max_time_secs.to_string();
    let status = Command::new("curl")
        .args([
            "-fsSL",
            "--retry",
            "2",
            "--connect-timeout",
            "15",
            "--max-time",
            &max_time,
            "-o",
        ])
        .arg(dest)
        .arg(url)
        .status()
        .map_err(|e| {
            VectorSearchError::DownloadFailed(format!("failed to invoke system curl: {e}"))
        })?;

    if !status.success() {
        return Err(VectorSearchError::DownloadFailed(format!(
            "curl exited with status {status}"
        )));
    }
    let meta = fs::metadata(dest)?;
    if meta.len() == 0 {
        return Err(VectorSearchError::DownloadFailed(
            "downloaded file is 0 bytes".to_string(),
        ));
    }
    Ok(())
}

fn run_tar_extract(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive_path)
        .arg("-C")
        .arg(dest_dir)
        .status()
        .map_err(|e| {
            VectorSearchError::DownloadFailed(format!("failed to invoke system tar: {e}"))
        })?;

    if !status.success() {
        return Err(VectorSearchError::DownloadFailed(format!(
            "tar extraction exited with status {status}"
        )));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Milestone M3 / Phase 5: Deterministic Metadata Generation (Task T3.2)
// -----------------------------------------------------------------------------

/// Canonical fixed cluster centroid seed for Mode B ($S_{\text{clust}} = 0\text{xCBF29CE484222325}$).
pub const S_CLUST_DEFAULT: u64 = 0xCBF2_9CE4_8422_2325;

/// Generates a deterministic independent (Bernoulli) bitset (Mode A).
///
/// Follows DESIGN.md Algorithm 1:
/// 1. Computes target count $K_\sigma = \lfloor \sigma \cdot N \rfloor$.
/// 2. Deterministically shuffles indices $0..N-1$ using Fisher-Yates with `DeterministicRng::new(seed)`.
/// 3. Activates the first $K_\sigma$ shuffled indices.
///
/// Guaranteed to produce exact popcount $K_\sigma$ with zero variance.
///
/// # Panics
/// Panics if `selectivity` is not in $[0.0, 1.0]$ or is NaN.
#[must_use]
pub fn generate_mode_a_bitset(num_vectors: usize, selectivity: f64, seed: u64) -> DenseBitset {
    assert!(
        (0.0..=1.0).contains(&selectivity),
        "selectivity must be in [0.0, 1.0], got {selectivity}"
    );
    let target_count = (selectivity * num_vectors as f64).floor() as usize;
    let mut bitset = DenseBitset::zeros(num_vectors);
    if target_count == 0 || num_vectors == 0 {
        return bitset;
    }
    let mut indices: Vec<usize> = (0..num_vectors).collect();
    let mut rng = DeterministicRng::new(seed);
    for i in (1..num_vectors).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
    for &id in &indices[..target_count] {
        bitset.set_bit(id, true);
    }
    bitset
}

/// Metadata diagnostics produced by Mode B generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeBMetadata {
    /// Resulting dense bitset with exact popcount $\lfloor \sigma \cdot N \rfloor$.
    pub bitset: DenseBitset,
    /// Number of centroids $C = \min(N, \max(64, \lfloor 2 / \sigma \rfloor))$.
    pub centroid_count: usize,
    /// Sorted IDs of whole clusters activated in the greedy stage.
    pub active_clusters: Vec<usize>,
    /// Cluster ID from which shortfall vectors were activated, if shortfall $s > 0$.
    pub shortfall_cluster: Option<usize>,
    /// Shortfall count $s = K_\sigma - \text{accumulated}$.
    pub shortfall_count: usize,
}

/// Selects $C$ centroids from `vectors` using deterministic $K$-means++ initialization.
///
/// # Panics
/// Panics if $C > N$.
pub fn deterministic_kmeans_plus_plus(
    vectors: &RowMatrix<f32>,
    c: usize,
    rng: &mut DeterministicRng,
) -> Vec<Vec<f32>> {
    let n = vectors.rows();
    assert!(c <= n, "centroid count c ({c}) cannot exceed n ({n})");
    if c == 0 || n == 0 {
        return Vec::new();
    }
    let mut centroids = Vec::with_capacity(c);
    // 1. Uniformly sample first centroid
    let first_idx = rng.next_usize(n);
    centroids.push(vectors.row(first_idx).to_vec());

    // 2. Track minimum squared L2 distance from each vector to nearest centroid
    let mut min_sq_dist: Vec<f32> = (0..n)
        .map(|i| l2_squared(vectors.row(i), &centroids[0]))
        .collect();

    // 3. Iteratively sample remaining centroids with probability proportional to D(x)^2
    for _ in 1..c {
        let total_dist: f64 = min_sq_dist.iter().map(|&d| d as f64).sum();
        let chosen_idx = if total_dist > 0.0 {
            let threshold = rng.next_f64_open01() * total_dist;
            let mut cumsum = 0.0;
            let mut idx = n - 1;
            for (i, &dist) in min_sq_dist.iter().enumerate() {
                cumsum += dist as f64;
                if cumsum >= threshold {
                    idx = i;
                    break;
                }
            }
            idx
        } else {
            // Degenerate case (e.g. duplicate points)
            rng.next_usize(n)
        };

        let new_centroid = vectors.row(chosen_idx).to_vec();
        for (i, dist) in min_sq_dist.iter_mut().enumerate() {
            let d = l2_squared(vectors.row(i), &new_centroid);
            if d < *dist {
                *dist = d;
            }
        }
        centroids.push(new_centroid);
    }

    centroids
}

/// Generates a deterministic cluster-correlated bitset (Mode B) with detailed metadata.
///
/// Follows DESIGN.md Algorithm 2 with exact shortfall calibration:
/// 1. $K_\sigma = \lfloor \sigma \cdot N \rfloor$.
/// 2. Centroid count $C = \max(64, \lfloor 2.0 / \sigma \rfloor)$, clamped to $C \le N$.
/// 3. Centroids chosen via deterministic $K$-means++ with `seed_clust`.
/// 4. Vector partitioning by nearest centroid (ties broken by cluster ID).
/// 5. Greedy accumulation of whole clusters sorted descending by size (ties broken by cluster ID).
/// 6. Shortfall calibration: non-active cluster nearest to any active centroid (ties by cluster ID;
///    if no active clusters, largest cluster) activates its $s$ members with smallest `(distance, id)`.
///
/// Guaranteed exact popcount $K_\sigma = \lfloor \sigma \cdot N \rfloor$ with zero variance.
///
/// # Panics
/// Panics if `selectivity` is not in $[0.0, 1.0]$ or is NaN.
pub fn generate_mode_b_bitset_detailed(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    _seed_attr: u64,
    seed_clust: u64,
) -> ModeBMetadata {
    assert!(
        (0.0..=1.0).contains(&selectivity),
        "selectivity must be in [0.0, 1.0], got {selectivity}"
    );
    let n = vectors.rows();
    if n == 0 {
        return ModeBMetadata {
            bitset: DenseBitset::zeros(0),
            centroid_count: 0,
            active_clusters: Vec::new(),
            shortfall_cluster: None,
            shortfall_count: 0,
        };
    }

    let target_count = (selectivity * n as f64).floor() as usize;
    if target_count == 0 {
        return ModeBMetadata {
            bitset: DenseBitset::zeros(n),
            centroid_count: 64.min(n),
            active_clusters: Vec::new(),
            shortfall_cluster: None,
            shortfall_count: 0,
        };
    }

    // Centroid count C = max(64, floor(2.0 / sigma)), clamped to C <= N
    let c_floor = (2.0 / selectivity).floor();
    let c_raw = if c_floor >= usize::MAX as f64 {
        n
    } else {
        c_floor as usize
    };
    let c = c_raw.max(64).min(n);

    let mut rng_clust = DeterministicRng::new(seed_clust);
    let centroids = deterministic_kmeans_plus_plus(vectors, c, &mut rng_clust);

    // Vector assignment: z_i = argmin_c ||V[i] - centroids[c]||_2^2 (ties broken by cluster ID c)
    let mut cluster_assignments = Vec::with_capacity(n);
    for i in 0..n {
        let v = vectors.row(i);
        let mut best_c = 0;
        let mut best_dist = l2_squared(v, &centroids[0]);
        for (c_idx, centroid) in centroids.iter().enumerate().skip(1) {
            let d = l2_squared(v, centroid);
            if d < best_dist {
                best_dist = d;
                best_c = c_idx;
            }
        }
        cluster_assignments.push(best_c);
    }

    // Cluster sizes
    let mut cluster_sizes = vec![0_usize; c];
    for &assign in &cluster_assignments {
        cluster_sizes[assign] += 1;
    }

    // Sort clusters descending by size, ties broken by cluster ID
    let mut sorted_clusters: Vec<usize> = (0..c).collect();
    sorted_clusters.sort_by(|&a, &b| {
        cluster_sizes[b]
            .cmp(&cluster_sizes[a])
            .then_with(|| a.cmp(&b))
    });

    // Greedily accumulate whole clusters
    let mut active_clusters = Vec::new();
    let mut is_active = vec![false; c];
    let mut accumulated = 0_usize;

    for &c_idx in &sorted_clusters {
        let sz = cluster_sizes[c_idx];
        if accumulated + sz <= target_count {
            active_clusters.push(c_idx);
            is_active[c_idx] = true;
            accumulated += sz;
        }
    }

    let mut bitset = DenseBitset::zeros(n);
    for (i, &assign) in cluster_assignments.iter().enumerate() {
        if is_active[assign] {
            bitset.set_bit(i, true);
        }
    }

    // Shortfall calibration
    let shortfall = target_count - accumulated;
    let mut shortfall_cluster = None;

    if shortfall > 0 {
        let chosen_cluster = if active_clusters.is_empty() {
            sorted_clusters[0]
        } else {
            let mut best_c = None;
            let mut best_dist = f32::INFINITY;
            for c_cand in 0..c {
                if is_active[c_cand] {
                    continue;
                }
                let mut min_to_active = f32::INFINITY;
                for &a in &active_clusters {
                    let d = l2_squared(&centroids[c_cand], &centroids[a]);
                    if d < min_to_active {
                        min_to_active = d;
                    }
                }
                if min_to_active < best_dist {
                    best_dist = min_to_active;
                    best_c = Some(c_cand);
                }
            }
            best_c.expect("Must find a non-active cluster when shortfall > 0")
        };

        shortfall_cluster = Some(chosen_cluster);

        // Member vectors sorted by (distance to own centroid, id)
        let mut members = Vec::with_capacity(cluster_sizes[chosen_cluster]);
        for (i, &assign) in cluster_assignments.iter().enumerate() {
            if assign == chosen_cluster {
                let d = l2_squared(vectors.row(i), &centroids[chosen_cluster]);
                members.push((d, i));
            }
        }

        members.sort_by(|&(d_a, id_a), &(d_b, id_b)| {
            d_a.partial_cmp(&d_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| id_a.cmp(&id_b))
        });

        assert!(
            members.len() >= shortfall,
            "Chosen shortfall cluster has {} members, but shortfall is {}",
            members.len(),
            shortfall
        );

        for &(_d, id) in members.iter().take(shortfall) {
            bitset.set_bit(id, true);
        }
    }

    assert_eq!(
        bitset.count_ones(),
        target_count,
        "Mode B generated popcount mismatch: expected {target_count}, got {}",
        bitset.count_ones()
    );

    ModeBMetadata {
        bitset,
        centroid_count: c,
        active_clusters,
        shortfall_cluster,
        shortfall_count: shortfall,
    }
}

/// Generates a deterministic cluster-correlated bitset (Mode B) with default centroid seed.
///
/// Signature matching Task T3.2 specification:
/// `generate_mode_b_bitset(vectors: &RowMatrix<f32>, selectivity: f64, seed_attr: u64) -> DenseBitset`
#[must_use]
pub fn generate_mode_b_bitset(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    seed_attr: u64,
) -> DenseBitset {
    generate_mode_b_bitset_detailed(vectors, selectivity, seed_attr, S_CLUST_DEFAULT).bitset
}

/// Generates a deterministic cluster-correlated bitset (Mode B) with explicit cluster seed.
#[must_use]
pub fn generate_mode_b_bitset_with_seed(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    seed_attr: u64,
    seed_clust: u64,
) -> DenseBitset {
    generate_mode_b_bitset_detailed(vectors, selectivity, seed_attr, seed_clust).bitset
}

// -----------------------------------------------------------------------------
// Bitset Binary Persistence (I/O)
// -----------------------------------------------------------------------------

const BITSET_MAGIC: &[u8; 8] = b"BITSET\x01\x00";

/// Writes a `DenseBitset` into a structured little-endian binary format with header validation.
///
/// Binary layout:
/// - 8 bytes: magic `b"BITSET\x01\x00"`
/// - 8 bytes: `len` as `u64` little-endian
/// - 8 bytes: `num_words` as `u64` little-endian
/// - `8 * num_words` bytes: 64-bit words as little-endian
pub fn write_bitset<P: AsRef<Path>>(path: P, bitset: &DenseBitset) -> Result<()> {
    let path_ref = path.as_ref();
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path_ref)?;
    let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

    writer.write_all(BITSET_MAGIC)?;
    writer.write_all(&(bitset.len() as u64).to_le_bytes())?;
    writer.write_all(&(bitset.words().len() as u64).to_le_bytes())?;
    for &word in bitset.words() {
        writer.write_all(&word.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

/// Reads a `DenseBitset` from a structured little-endian binary file.
pub fn read_bitset<P: AsRef<Path>>(path: P) -> Result<DenseBitset> {
    let path_ref = path.as_ref();
    let meta = fs::metadata(path_ref)?;
    let file_size = meta.len();

    if file_size < 24 {
        return Err(VectorSearchError::CorruptData(format!(
            "Bitset file too small ({file_size} < 24 bytes): {}",
            path_ref.display()
        )));
    }

    let file = File::open(path_ref)?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);

    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != BITSET_MAGIC {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid bitset file magic in {}",
            path_ref.display()
        )));
    }

    let mut buf8 = [0_u8; 8];
    reader.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let num_words = u64::from_le_bytes(buf8) as usize;

    let expected_words = len.div_ceil(64);
    if num_words != expected_words {
        return Err(VectorSearchError::CorruptData(format!(
            "Bitset word count mismatch: header claims {num_words}, expected {expected_words} for len {len}"
        )));
    }

    let expected_file_size = 24_u64 + (num_words as u64) * 8;
    if file_size != expected_file_size {
        return Err(VectorSearchError::CorruptData(format!(
            "Bitset file size mismatch: actual {file_size}, expected {expected_file_size}"
        )));
    }

    let mut words = Vec::with_capacity(num_words);
    for _ in 0..num_words {
        reader.read_exact(&mut buf8)?;
        words.push(u64::from_le_bytes(buf8));
    }

    Ok(DenseBitset::from_words(words, len))
}

/// Writes active (1) IDs of a `DenseBitset` into a Texmex-compatible `.ivecs` file.
///
/// Emits a single vector record whose dimension is the popcount of the bitset.
pub fn write_bitset_as_ivecs<P: AsRef<Path>>(path: P, bitset: &DenseBitset) -> Result<()> {
    let popcount = bitset.count_ones();
    let ids: Vec<i32> = bitset
        .iter_ones()
        .map(|id| {
            i32::try_from(id).map_err(|_| {
                VectorSearchError::InvalidArgument(format!(
                    "Vector ID {id} exceeds i32::MAX for .ivecs export"
                ))
            })
        })
        .collect::<Result<Vec<i32>>>()?;

    let matrix = if popcount == 0 {
        RowMatrix::empty()
    } else {
        RowMatrix::new(1, popcount, ids)?
    };

    write_ivecs(path, &matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_a_popcount_exactness() {
        let n = 10_000;
        let selectivities = [0.5, 0.1, 0.01, 0.001];
        for &sel in &selectivities {
            let expected_k = (sel * n as f64).floor() as usize;
            let bitset = generate_mode_a_bitset(n, sel, 42);
            assert_eq!(bitset.len(), n);
            assert_eq!(
                bitset.count_ones(),
                expected_k,
                "Mode A exact popcount failed for sel={sel}"
            );
        }

        // Non-round size and selectivity
        let non_round_n = 1234;
        let sel = 0.33;
        let expected_k = (sel * non_round_n as f64).floor() as usize;
        let bitset = generate_mode_a_bitset(non_round_n, sel, 999);
        assert_eq!(bitset.len(), non_round_n);
        assert_eq!(bitset.count_ones(), expected_k);
    }

    #[test]
    fn test_mode_a_deterministic_repeatability() {
        let n = 2000;
        let sel = 0.15;
        let seed = 12345;
        let b1 = generate_mode_a_bitset(n, sel, seed);
        let b2 = generate_mode_a_bitset(n, sel, seed);
        assert_eq!(b1, b2);
        assert_eq!(b1.words(), b2.words());

        // Different seeds should produce different bitsets
        let b3 = generate_mode_a_bitset(n, sel, seed + 1);
        assert_ne!(b1, b3);
    }

    #[test]
    fn test_mode_a_edge_cases() {
        // Selectivity 0.0
        let b0 = generate_mode_a_bitset(500, 0.0, 42);
        assert_eq!(b0.len(), 500);
        assert_eq!(b0.count_ones(), 0);

        // Selectivity 1.0
        let b1 = generate_mode_a_bitset(500, 1.0, 42);
        assert_eq!(b1.len(), 500);
        assert_eq!(b1.count_ones(), 500);

        // N = 0
        let b_empty = generate_mode_a_bitset(0, 0.5, 42);
        assert_eq!(b_empty.len(), 0);
        assert_eq!(b_empty.count_ones(), 0);

        // N = 1
        let b_one_0 = generate_mode_a_bitset(1, 0.0, 42);
        assert_eq!(b_one_0.count_ones(), 0);
        let b_one_1 = generate_mode_a_bitset(1, 1.0, 42);
        assert_eq!(b_one_1.count_ones(), 1);
        assert!(b_one_1.get_bit(0));
    }

    #[test]
    #[should_panic(expected = "selectivity must be in [0.0, 1.0]")]
    fn test_mode_a_negative_selectivity_panics() {
        let _ = generate_mode_a_bitset(100, -0.01, 42);
    }

    #[test]
    #[should_panic(expected = "selectivity must be in [0.0, 1.0]")]
    fn test_mode_a_excess_selectivity_panics() {
        let _ = generate_mode_a_bitset(100, 1.01, 42);
    }

    #[test]
    fn test_kmeans_plus_plus_determinism() {
        let (base, _) = generate_synthetic_sift(200, 5, 16, 10, 42).unwrap();
        let mut rng1 = DeterministicRng::new(0xCBF2_9CE4_8422_2325);
        let mut rng2 = DeterministicRng::new(0xCBF2_9CE4_8422_2325);
        let c1 = deterministic_kmeans_plus_plus(&base, 10, &mut rng1);
        let c2 = deterministic_kmeans_plus_plus(&base, 10, &mut rng2);
        assert_eq!(c1.len(), 10);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_mode_b_popcount_exactness() {
        let (base, _) = generate_synthetic_sift(1000, 10, 32, 20, 101).unwrap();
        let selectivities = [0.5, 0.1, 0.01, 0.001];
        for &sel in &selectivities {
            let expected_k = (sel * base.rows() as f64).floor() as usize;
            let bitset = generate_mode_b_bitset(&base, sel, 42);
            assert_eq!(bitset.len(), base.rows());
            assert_eq!(
                bitset.count_ones(),
                expected_k,
                "Mode B exact popcount failed for sel={sel}"
            );
        }
    }

    #[test]
    fn test_mode_b_deterministic_repeatability() {
        let (base, _) = generate_synthetic_sift(500, 10, 16, 10, 777).unwrap();
        let sel = 0.05;
        let meta1 = generate_mode_b_bitset_detailed(&base, sel, 42, S_CLUST_DEFAULT);
        let meta2 = generate_mode_b_bitset_detailed(&base, sel, 42, S_CLUST_DEFAULT);
        assert_eq!(meta1.bitset, meta2.bitset);
        assert_eq!(meta1.active_clusters, meta2.active_clusters);
        assert_eq!(meta1.shortfall_cluster, meta2.shortfall_cluster);
        assert_eq!(meta1.shortfall_count, meta2.shortfall_count);
    }

    #[test]
    fn test_mode_b_edge_cases() {
        let (base, _) = generate_synthetic_sift(100, 5, 16, 5, 55).unwrap();

        // Selectivity 0.0
        let b0 = generate_mode_b_bitset(&base, 0.0, 42);
        assert_eq!(b0.len(), 100);
        assert_eq!(b0.count_ones(), 0);

        // Selectivity 1.0
        let b1 = generate_mode_b_bitset(&base, 1.0, 42);
        assert_eq!(b1.len(), 100);
        assert_eq!(b1.count_ones(), 100);

        // Small dataset N < C (e.g. N = 10, C = max(64, ...) clamped to 10)
        let (small_base, _) = generate_synthetic_sift(10, 2, 8, 3, 12).unwrap();
        let b_small = generate_mode_b_bitset(&small_base, 0.3, 42);
        assert_eq!(b_small.len(), 10);
        assert_eq!(b_small.count_ones(), 3);

        // N = 0
        let empty_matrix = RowMatrix::<f32>::empty();
        let b_empty = generate_mode_b_bitset(&empty_matrix, 0.5, 42);
        assert_eq!(b_empty.len(), 0);
        assert_eq!(b_empty.count_ones(), 0);
    }

    #[test]
    fn test_mode_b_shortfall_calibration_with_odd_selectivity() {
        let (base, _) = generate_synthetic_sift(300, 5, 16, 10, 888).unwrap();
        // An odd selectivity that guarantees fractional cluster shortfall
        let sel: f64 = 0.073; // target_count = floor(0.073 * 300) = 21
        let expected_k = (sel * 300.0_f64).floor() as usize;
        let meta = generate_mode_b_bitset_detailed(&base, sel, 42, S_CLUST_DEFAULT);
        assert_eq!(meta.bitset.count_ones(), expected_k);
        // Shortfall should be calibrated cleanly
        if meta.shortfall_count > 0 {
            assert!(meta.shortfall_cluster.is_some());
        }
    }

    #[test]
    fn test_bitset_binary_io_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bitset_roundtrip.bin");

        let mut original = DenseBitset::zeros(300);
        original.set_bit(0, true);
        original.set_bit(63, true);
        original.set_bit(64, true);
        original.set_bit(127, true);
        original.set_bit(299, true);

        write_bitset(&file_path, &original).unwrap();
        let loaded = read_bitset(&file_path).unwrap();

        assert_eq!(original, loaded);
        assert_eq!(loaded.len(), 300);
        assert_eq!(loaded.count_ones(), 5);
        assert!(loaded.get_bit(0));
        assert!(loaded.get_bit(63));
        assert!(loaded.get_bit(64));
        assert!(loaded.get_bit(127));
        assert!(loaded.get_bit(299));
        assert!(!loaded.get_bit(1));

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_bitset_ivecs_io_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bitset_ivecs.ivecs");

        let mut original = DenseBitset::zeros(100);
        original.set_bit(5, true);
        original.set_bit(17, true);
        original.set_bit(88, true);

        write_bitset_as_ivecs(&file_path, &original).unwrap();
        let matrix = read_ivecs(&file_path).unwrap();
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.row(0), &[5, 17, 88]);

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_bitset_io_corruption_handling() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bitset_corrupt.bin");

        // Too small (< 24 bytes)
        std::fs::write(&file_path, b"TOO_SHORT").unwrap();
        assert!(read_bitset(&file_path).is_err());

        // Invalid magic
        let bad_magic = vec![0_u8; 32];
        std::fs::write(&file_path, bad_magic).unwrap();
        assert!(read_bitset(&file_path).is_err());

        let _ = std::fs::remove_file(file_path);
    }
}
