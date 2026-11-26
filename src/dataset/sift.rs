//! Public SIFT (`siftsmall` & `SIFT1M`) dataset loader/downloader, custom `.fvecs` loader,
//! and deterministic synthetic clustered Gaussian SIFT generator.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::texmex_io::{read_fvecs_max_rows, read_ivecs_max_rows};
use crate::error::{Result, VectorSearchError};
use crate::matrix::RowMatrix;
use crate::rng::DeterministicRng;

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
    let centroid_elems = clusters.checked_mul(dim).ok_or_else(|| {
        VectorSearchError::InvalidArgument("num_clusters * dim overflows usize".to_string())
    })?;
    let mut centroids = vec![0.0_f32; centroid_elems];
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
    if let Some(loaded) = try_load_cached_sift_bundle(
        preset,
        cache_dir,
        max_base,
        max_queries,
        load_ground_truth,
        t0,
    ) {
        return Ok(loaded);
    }

    // 2. Attempt public mirror download if permitted
    let fallback_reason = if !allow_download {
        "offline mode enabled (--offline) and local SIFT cache not present".to_string()
    } else {
        match download_sift_preset(preset, cache_dir) {
            Ok(()) => {
                if let Some(loaded) = try_load_cached_sift_bundle(
                    preset,
                    cache_dir,
                    max_base,
                    max_queries,
                    load_ground_truth,
                    t0,
                ) {
                    return Ok(loaded);
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
    let eff_base = resolve_row_limit(preset.expected_num_base(), max_base);
    let eff_query = resolve_row_limit(preset.expected_num_query(), max_queries);
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

#[inline]
fn resolve_row_limit(expected: usize, limit: Option<usize>) -> usize {
    limit.map_or(expected, |lim| expected.min(lim))
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

fn expected_preset_file_bytes(preset: SiftPreset) -> (u64, u64, u64) {
    let expected_base_bytes =
        (preset.expected_num_base() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_query_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_gt_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_gt_k() as u64));
    (expected_base_bytes, expected_query_bytes, expected_gt_bytes)
}

fn try_load_cached_sift_bundle(
    preset: SiftPreset,
    cache_dir: &Path,
    max_base: Option<usize>,
    max_queries: Option<usize>,
    load_ground_truth: bool,
    t0: Instant,
) -> Option<LoadedDataset> {
    let (expected_base_bytes, expected_query_bytes, expected_gt_bytes) =
        expected_preset_file_bytes(preset);

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
            let source = match preset {
                SiftPreset::SiftSmall => DatasetSource::AuthenticSiftSmall { dir },
                SiftPreset::Sift1M => DatasetSource::AuthenticSift1M { dir },
            };
            return Some(LoadedDataset {
                base,
                query,
                official_ground_truth: gt,
                source,
                load_time_s: t0.elapsed().as_secs_f64(),
            });
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
    let expected_base_rows = resolve_row_limit(preset.expected_num_base(), max_base);
    let expected_query_rows = resolve_row_limit(preset.expected_num_query(), max_queries);

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

            let (expected_base_bytes, expected_query_bytes, expected_gt_bytes) =
                expected_preset_file_bytes(SiftPreset::Sift1M);
            let direct_files = [
                (
                    SIFT1M_BASE_URL,
                    target_dir.join("sift_base.fvecs"),
                    900,
                    expected_base_bytes,
                ),
                (
                    SIFT1M_QUERY_URL,
                    target_dir.join("sift_query.fvecs"),
                    180,
                    expected_query_bytes,
                ),
                (
                    SIFT1M_GT_URL,
                    target_dir.join("sift_groundtruth.ivecs"),
                    180,
                    expected_gt_bytes,
                ),
            ];

            let mut direct_ok = true;
            let mut errors = Vec::new();
            for (url, dest, timeout_s, expected_bytes) in &direct_files {
                if dest.is_file() {
                    let actual_bytes = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
                    if actual_bytes == *expected_bytes {
                        continue;
                    }
                    let _ = fs::remove_file(dest);
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
