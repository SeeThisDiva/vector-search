//! Dataset & ground-truth resolution for Phase 6 (strictly disallowing silent synthetic fallback).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::dataset::{
    generate_synthetic_sift, inspect_fvecs_header, load_or_download_sift_with_options,
    read_fvecs_max_rows, read_ivecs, read_ivecs_max_rows, write_ivecs, DatasetSource, SiftPreset,
};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::matrix::RowMatrix;

use super::config::Phase6MatrixConfig;

pub(crate) struct PreparedDatasetState {
    pub source: DatasetSource,
    pub base_fvecs_path: Option<PathBuf>,
    pub query_fvecs_path: Option<PathBuf>,
    pub query: RowMatrix<f32>,
    pub base_cache: Option<Arc<RowMatrix<f32>>>,
    pub num_base: usize,
    pub dim: usize,
    pub gt_matrix: RowMatrix<i32>,
    pub gt_source: String,
    pub gt_cache_path: Option<PathBuf>,
    pub gt_cache_hit: bool,
}

impl PreparedDatasetState {
    /// Lazily loads or returns the shared `Arc<RowMatrix<f32>>` base matrix.
    pub fn get_or_load_base(&mut self) -> Result<Arc<RowMatrix<f32>>> {
        if let Some(ref arc) = self.base_cache {
            return Ok(Arc::clone(arc));
        }
        let path = self.base_fvecs_path.as_ref().ok_or_else(|| {
            VectorSearchError::InvalidArgument(
                "Internal error: missing base_fvecs_path when loading base matrix".to_string(),
            )
        })?;
        let base = read_fvecs_max_rows(path, Some(self.num_base))?;
        let arc = Arc::new(base);
        self.base_cache = Some(Arc::clone(&arc));
        Ok(arc)
    }
}

fn resolve_sift_file_paths(
    preset: SiftPreset,
    data_dir: &Path,
) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let candidates: Vec<(PathBuf, &str)> = match preset {
        SiftPreset::SiftSmall => vec![
            (data_dir.join("siftsmall"), "siftsmall"),
            (data_dir.to_path_buf(), "siftsmall"),
        ],
        SiftPreset::Sift1M => vec![
            (data_dir.join("sift1m"), "sift"),
            (data_dir.join("sift"), "sift"),
            (data_dir.to_path_buf(), "sift"),
        ],
    };
    let expected_base_bytes =
        (preset.expected_num_base() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_query_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_gt_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_gt_k() as u64));

    for (dir, prefix) in candidates {
        let base_p = dir.join(format!("{prefix}_base.fvecs"));
        let query_p = dir.join(format!("{prefix}_query.fvecs"));
        let gt_p = dir.join(format!("{prefix}_groundtruth.ivecs"));
        if !base_p.is_file() || !query_p.is_file() || !gt_p.is_file() {
            continue;
        }
        let b_len = fs::metadata(&base_p).map(|m| m.len()).unwrap_or(0);
        let q_len = fs::metadata(&query_p).map(|m| m.len()).unwrap_or(0);
        let g_len = fs::metadata(&gt_p).map(|m| m.len()).unwrap_or(0);
        if b_len == expected_base_bytes
            && q_len == expected_query_bytes
            && g_len == expected_gt_bytes
        {
            return Some((base_p, query_p, gt_p));
        }
    }
    None
}

fn slice_and_validate_gt(
    src: &RowMatrix<i32>,
    num_queries: usize,
    num_base: usize,
    k: usize,
) -> Result<RowMatrix<i32>> {
    if src.rows() < num_queries || src.cols() < k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Ground truth matrix shape ({}x{}) is smaller than required ({num_queries}x{k})",
            src.rows(),
            src.cols()
        )));
    }
    let mut data = Vec::with_capacity(num_queries * k);
    for r in 0..num_queries {
        let slice = &src.row(r)[..k];
        for &id in slice {
            if id < 0 || (id as usize) >= num_base {
                return Err(VectorSearchError::CorruptData(format!(
                    "Ground truth ID {id} at row {r} out of range [0, {num_base})"
                )));
            }
        }
        data.extend_from_slice(slice);
    }
    RowMatrix::new(num_queries, k, data)
}

pub(crate) fn prepare_dataset_and_gt(config: &Phase6MatrixConfig) -> Result<PreparedDatasetState> {
    if config.synthetic {
        let (base, query) = generate_synthetic_sift(
            config.scale,
            config.queries,
            config.dim,
            config.num_clusters,
            config.seed,
        )?;
        let num_base = base.rows();
        let num_queries = query.rows();
        let dim = base.cols();
        let source = DatasetSource::Synthetic { seed: config.seed };

        let (gt_matrix, gt_source, gt_cache_path, gt_cache_hit) = if let Some(ref explicit_gt) =
            config.gt_path
        {
            let raw_gt = read_ivecs_max_rows(explicit_gt, Some(num_queries))?;
            let sliced = slice_and_validate_gt(&raw_gt, num_queries, num_base, config.k)?;
            (
                sliced,
                format!("custom_ivecs ({})", explicit_gt.display()),
                Some(explicit_gt.clone()),
                true,
            )
        } else {
            let cache_k = 100.max(config.k).min(num_base);
            let exact =
                ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, cache_k, 0)?;
            let sliced = slice_and_validate_gt(&exact.indices, num_queries, num_base, config.k)?;
            (
                sliced,
                format!("computed_exact_knn_synthetic (metric=l2, k={cache_k})"),
                None,
                false,
            )
        };

        return Ok(PreparedDatasetState {
            source,
            base_fvecs_path: None,
            query_fvecs_path: None,
            query,
            base_cache: Some(Arc::new(base)),
            num_base,
            dim,
            gt_matrix,
            gt_source,
            gt_cache_path,
            gt_cache_hit,
        });
    }

    // Custom .fvecs path pair
    if let (Some(ref base_path), Some(ref query_path)) = (&config.base_fvecs, &config.query_fvecs) {
        let base_hdr = inspect_fvecs_header(base_path)?;
        let query_hdr = inspect_fvecs_header(query_path)?;
        if base_hdr.dim != query_hdr.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: base_hdr.dim,
                actual: query_hdr.dim,
            });
        }
        let num_base = if config.scale_explicit {
            if config.scale > base_hdr.num_rows {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --scale {}, but {} has only {} vectors",
                    config.scale,
                    base_path.display(),
                    base_hdr.num_rows
                )));
            }
            config.scale
        } else {
            base_hdr.num_rows
        };
        let num_queries = if config.queries_explicit {
            if config.queries > query_hdr.num_rows {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --queries {}, but {} has only {} queries",
                    config.queries,
                    query_path.display(),
                    query_hdr.num_rows
                )));
            }
            config.queries
        } else {
            query_hdr.num_rows
        };
        let query = read_fvecs_max_rows(query_path, Some(num_queries))?;
        let source = DatasetSource::CustomFiles {
            base_path: base_path.clone(),
            query_path: query_path.clone(),
        };

        return resolve_file_backed_gt(
            config,
            source,
            base_path.clone(),
            query_path.clone(),
            None,
            query,
            num_base,
            base_hdr.dim,
        );
    }

    // Authentic SIFT dataset under --data-dir (NEVER silently fall back to synthetic!)
    let preset = config.dataset.to_preset();
    if config.scale > preset.expected_num_base() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Requested --scale {} exceeds {} maximum base count ({})",
            config.scale,
            preset.as_str(),
            preset.expected_num_base()
        )));
    }
    if config.queries > preset.expected_num_query() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Requested --queries {} exceeds {} maximum query count ({})",
            config.queries,
            preset.as_str(),
            preset.expected_num_query()
        )));
    }

    let (base_path, query_path, official_gt_path) = if let Some(paths) =
        resolve_sift_file_paths(preset, &config.data_dir)
    {
        paths
    } else {
        // Probe / download if !offline; verify strictly authentic SIFT afterwards.
        let probe = load_or_download_sift_with_options(
            preset,
            &config.data_dir,
            !config.offline,
            config.seed,
            Some(1),
            Some(1),
            false,
        )?;
        if !probe.source.is_authentic_sift() {
            let reason = match &probe.source {
                DatasetSource::SyntheticFallback { reason, .. } => reason.as_str(),
                _ => "authentic SIFT files not found",
            };
            return Err(VectorSearchError::InvalidArgument(format!(
                    "Authentic {} dataset missing in '{}' ({reason}). Never falling back silently: pass --synthetic explicitly or place SIFT files under --data-dir.",
                    preset.as_str(),
                    config.data_dir.display()
                )));
        }
        resolve_sift_file_paths(preset, &config.data_dir).ok_or_else(|| {
            VectorSearchError::InvalidArgument(format!(
                "Unable to resolve SIFT files under '{}'",
                config.data_dir.display()
            ))
        })?
    };

    let dir = base_path
        .parent()
        .map_or_else(|| config.data_dir.clone(), Path::to_path_buf);
    let source = match preset {
        SiftPreset::SiftSmall => DatasetSource::AuthenticSiftSmall { dir },
        SiftPreset::Sift1M => DatasetSource::AuthenticSift1M { dir },
    };
    let query = read_fvecs_max_rows(&query_path, Some(config.queries))?;
    let official_opt = (config.scale == preset.expected_num_base()).then_some(official_gt_path);

    resolve_file_backed_gt(
        config,
        source,
        base_path,
        query_path,
        official_opt,
        query,
        config.scale,
        preset.expected_dim(),
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_file_backed_gt(
    config: &Phase6MatrixConfig,
    source: DatasetSource,
    base_path: PathBuf,
    query_path: PathBuf,
    official_gt_path: Option<PathBuf>,
    query: RowMatrix<f32>,
    num_base: usize,
    dim: usize,
) -> Result<PreparedDatasetState> {
    let num_queries = query.rows();

    if let Some(ref explicit_gt) = config.gt_path {
        let raw_gt = read_ivecs_max_rows(explicit_gt, Some(num_queries))?;
        let gt_matrix = slice_and_validate_gt(&raw_gt, num_queries, num_base, config.k)?;
        return Ok(PreparedDatasetState {
            source,
            base_fvecs_path: Some(base_path),
            query_fvecs_path: Some(query_path),
            query,
            base_cache: None,
            num_base,
            dim,
            gt_matrix,
            gt_source: format!("custom_ivecs ({})", explicit_gt.display()),
            gt_cache_path: Some(explicit_gt.clone()),
            gt_cache_hit: true,
        });
    }

    if let Some(ref official_path) = official_gt_path {
        let raw_gt = read_ivecs_max_rows(official_path, Some(num_queries))?;
        let gt_matrix = slice_and_validate_gt(&raw_gt, num_queries, num_base, config.k)?;
        let gt_source = match source {
            DatasetSource::AuthenticSift1M { .. } => {
                "official_texmex_sift1m_groundtruth_ivecs".to_string()
            }
            DatasetSource::AuthenticSiftSmall { .. } => {
                "official_texmex_siftsmall_groundtruth_ivecs".to_string()
            }
            _ => format!("official_ivecs ({})", official_path.display()),
        };
        return Ok(PreparedDatasetState {
            source,
            base_fvecs_path: Some(base_path),
            query_fvecs_path: Some(query_path),
            query,
            base_cache: None,
            num_base,
            dim,
            gt_matrix,
            gt_source,
            gt_cache_path: Some(official_path.clone()),
            gt_cache_hit: true,
        });
    }

    // Prefix scale (< 1M): check <data-dir>/gt_{scale}_k100.ivecs
    let cache_path = config.data_dir.join(format!("gt_{num_base}_k100.ivecs"));
    let cache_k = 100.max(config.k).min(num_base);
    if config.enable_gt_cache && cache_path.is_file() {
        if let Ok(cached) = read_ivecs(&cache_path) {
            if cached.rows() == num_queries && cached.cols() >= config.k {
                if let Ok(gt_matrix) =
                    slice_and_validate_gt(&cached, num_queries, num_base, config.k)
                {
                    return Ok(PreparedDatasetState {
                        source,
                        base_fvecs_path: Some(base_path),
                        query_fvecs_path: Some(query_path),
                        query,
                        base_cache: None,
                        num_base,
                        dim,
                        gt_matrix,
                        gt_source: format!(
                            "cached_exact_knn_prefix_ivecs ({}, metric=l2, k={})",
                            cache_path.display(),
                            cached.cols()
                        ),
                        gt_cache_path: Some(cache_path),
                        gt_cache_hit: true,
                    });
                }
            }
        }
    }

    // Cache miss: load base matrix, compute exact GT, and retain base_cache in memory.
    let base = read_fvecs_max_rows(&base_path, Some(num_base))?;
    let exact = ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, cache_k, 0)?;
    if config.enable_gt_cache {
        fs::create_dir_all(&config.data_dir)?;
        write_ivecs(&cache_path, &exact.indices)?;
    }
    let gt_matrix = slice_and_validate_gt(&exact.indices, num_queries, num_base, config.k)?;

    Ok(PreparedDatasetState {
        source,
        base_fvecs_path: Some(base_path),
        query_fvecs_path: Some(query_path),
        query,
        base_cache: Some(Arc::new(base)),
        num_base,
        dim,
        gt_matrix,
        gt_source: format!(
            "computed_exact_knn_prefix_ivecs (ExactKnn::search_parallel_with_base metric=l2 k={cache_k}{})",
            if config.enable_gt_cache {
                format!(", cached to {}", cache_path.display())
            } else {
                String::new()
            }
        ),
        gt_cache_path: config.enable_gt_cache.then_some(cache_path),
        gt_cache_hit: false,
    })
}
