//! Texmex `.fvecs` / `.ivecs` binary serialization, deterministic synthetic SIFT generator,
//! and public SIFT (`siftsmall` & `SIFT1M`) dataset downloader with offline fallback.

use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::error::{Result, VectorSearchError};
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
    let num_vecs = usize::try_from(num_vecs_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!("Vector count {num_vecs_u64} exceeds usize"))
    })?;
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
    let num_rows = usize::try_from(num_rows_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!("Row count {num_rows_u64} exceeds usize"))
    })?;
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
    let t0 = Instant::now();
    let base = read_fvecs(base_path)?;
    let query = read_fvecs(query_path)?;
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
        Some(p) => Some(read_ivecs(p)?),
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
    let t0 = Instant::now();

    // 1. Check existing local cache first
    if let Some((base, query, gt, resolved_dir)) = try_load_cached_sift(preset, cache_dir) {
        let source = match preset {
            SiftPreset::SiftSmall => DatasetSource::AuthenticSiftSmall { dir: resolved_dir },
            SiftPreset::Sift1M => DatasetSource::AuthenticSift1M { dir: resolved_dir },
        };
        return Ok(LoadedDataset {
            base,
            query,
            official_ground_truth: Some(gt),
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
                if let Some((base, query, gt, resolved_dir)) =
                    try_load_cached_sift(preset, cache_dir)
                {
                    let source = match preset {
                        SiftPreset::SiftSmall => {
                            DatasetSource::AuthenticSiftSmall { dir: resolved_dir }
                        }
                        SiftPreset::Sift1M => DatasetSource::AuthenticSift1M { dir: resolved_dir },
                    };
                    return Ok(LoadedDataset {
                        base,
                        query,
                        official_ground_truth: Some(gt),
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
    let (base, query) = generate_synthetic_sift(
        preset.expected_num_base(),
        preset.expected_num_query(),
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

type CachedSiftBundle = (RowMatrix<f32>, RowMatrix<f32>, RowMatrix<i32>, PathBuf);

fn try_load_cached_sift(preset: SiftPreset, cache_dir: &Path) -> Option<CachedSiftBundle> {
    for (dir, prefix) in candidate_sift_dirs(preset, cache_dir) {
        let base_path = dir.join(format!("{prefix}_base.fvecs"));
        let query_path = dir.join(format!("{prefix}_query.fvecs"));
        let gt_path = dir.join(format!("{prefix}_groundtruth.ivecs"));

        if !base_path.is_file() || !query_path.is_file() || !gt_path.is_file() {
            continue;
        }

        let (Ok(base), Ok(query), Ok(gt)) = (
            read_fvecs(&base_path),
            read_fvecs(&query_path),
            read_ivecs(&gt_path),
        ) else {
            continue;
        };

        if validate_authentic_sift_shapes(preset, &base, &query, &gt) {
            return Some((base, query, gt, dir));
        }
    }
    None
}

fn validate_authentic_sift_shapes(
    preset: SiftPreset,
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    gt: &RowMatrix<i32>,
) -> bool {
    if base.shape() != (preset.expected_num_base(), preset.expected_dim()) {
        return false;
    }
    if query.shape() != (preset.expected_num_query(), preset.expected_dim()) {
        return false;
    }
    if gt.shape() != (preset.expected_num_query(), preset.expected_gt_k()) {
        return false;
    }
    if !base.as_slice().iter().all(|v| v.is_finite())
        || !query.as_slice().iter().all(|v| v.is_finite())
    {
        return false;
    }
    let max_id = preset.expected_num_base() as i32;
    gt.as_slice().iter().all(|&id| id >= 0 && id < max_id)
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
                            Ok(()) => return Ok(()),
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
