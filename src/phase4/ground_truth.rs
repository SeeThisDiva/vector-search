//! Prefix Ground Truth Disk Caching and Exact Brute-Force Fallback for Phase 4.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use crate::dataset::{read_ivecs, write_ivecs, DatasetSource, LoadedDataset};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::matrix::RowMatrix;

/// Slices the first `cols` columns and first `rows` queries from a ground truth matrix.
fn slice_ground_truth_matrix(
    src: &RowMatrix<i32>,
    rows: usize,
    cols: usize,
) -> Result<RowMatrix<i32>> {
    if src.rows() == rows && src.cols() == cols {
        return Ok(src.clone());
    }
    if src.rows() < rows || src.cols() < cols {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Source ground truth matrix ({}x{}) smaller than requested ({}x{})",
            src.rows(),
            src.cols(),
            rows,
            cols
        )));
    }
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend_from_slice(&src.row(r)[..cols]);
    }
    RowMatrix::new(rows, cols, data)
}

/// Result of loading or computing Phase 4 ground truth.
#[derive(Debug, Clone)]
pub struct Phase4GroundTruth {
    pub matrix: RowMatrix<i32>,
    pub ground_truth_source: String,
    pub ground_truth_cache_path: Option<PathBuf>,
    pub ground_truth_cache_hit: bool,
}

/// Loads official ground truth (if 1M authentic SIFT) or loads/computes exact prefix ground truth.
pub fn load_or_compute_phase4_ground_truth(
    loaded: &LoadedDataset,
    k: usize,
    metric: Metric,
    data_cache_dir: &Path,
    enable_disk_cache: bool,
) -> Result<Phase4GroundTruth> {
    let num_base = loaded.base.rows();
    let num_queries = loaded.query.rows();
    if num_base == 0 || num_queries == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Base or query matrix cannot be empty".to_string(),
        ));
    }

    // Case A: Official SIFT1M or siftsmall ground truth
    if loaded.source.is_authentic_sift() {
        let is_full_sift1m = num_base == 1_000_000
            && loaded
                .official_ground_truth
                .as_ref()
                .is_some_and(|g| g.rows() == 10_000);
        let is_full_siftsmall = num_base == 10_000
            && loaded
                .official_ground_truth
                .as_ref()
                .is_some_and(|g| g.rows() == 100);
        if is_full_sift1m || is_full_siftsmall {
            if let Some(ref official) = loaded.official_ground_truth {
                if official.rows() >= num_queries && official.cols() >= k {
                    let matrix = slice_ground_truth_matrix(official, num_queries, k)?;
                    let source_name = if is_full_sift1m {
                        "official_texmex_sift1m_groundtruth_ivecs"
                    } else {
                        "official_texmex_siftsmall_groundtruth_ivecs"
                    };
                    return Ok(Phase4GroundTruth {
                        matrix,
                        ground_truth_source: source_name.to_string(),
                        ground_truth_cache_path: None,
                        ground_truth_cache_hit: true,
                    });
                }
            }
        }
    }

    // Case B: Prefix scale or synthetic/custom dataset
    let cache_filename = match &loaded.source {
        DatasetSource::AuthenticSiftSmall { .. } | DatasetSource::AuthenticSift1M { .. } => {
            format!("gt_{num_base}_k100.ivecs")
        }
        DatasetSource::Synthetic { seed } => {
            format!("gt_synthetic_seed{seed}_{num_base}_k100.ivecs")
        }
        DatasetSource::SyntheticFallback { seed, .. } => {
            format!("gt_synthetic_seed{seed}_{num_base}_k100.ivecs")
        }
        DatasetSource::CustomFiles { .. } => {
            format!("gt_custom_{num_base}_k100.ivecs")
        }
    };
    let cache_path = data_cache_dir.join(cache_filename);
    let cache_k = 100.max(k).min(num_base);

    if enable_disk_cache && cache_path.is_file() {
        if let Ok(cached) = read_ivecs(&cache_path) {
            if cached.rows() == num_queries && cached.cols() >= k {
                let matrix = slice_ground_truth_matrix(&cached, num_queries, k)?;
                let ground_truth_source = format!(
                    "cached_exact_knn_prefix_ivecs ({}, computed by ExactKnn::search_parallel_with_base metric={} k={})",
                    cache_path.display(),
                    metric,
                    cached.cols()
                );
                return Ok(Phase4GroundTruth {
                    matrix,
                    ground_truth_source,
                    ground_truth_cache_path: Some(cache_path),
                    ground_truth_cache_hit: true,
                });
            }
        }
    }

    // Compute exact brute-force ground truth on the prefix
    let exact =
        ExactKnn::search_parallel_with_base(metric, &loaded.base, &loaded.query, cache_k, 0)?;

    if enable_disk_cache {
        if !data_cache_dir.exists() {
            fs::create_dir_all(data_cache_dir)?;
        }
        write_ivecs(&cache_path, &exact.indices)?;
    }

    let matrix = slice_ground_truth_matrix(&exact.indices, num_queries, k)?;
    let ground_truth_source = format!(
        "computed_exact_knn_prefix_ivecs (ExactKnn::search_parallel_with_base metric={} k={}{})",
        metric,
        cache_k,
        if enable_disk_cache {
            format!(", cached to {}", cache_path.display())
        } else {
            String::new()
        }
    );

    Ok(Phase4GroundTruth {
        matrix,
        ground_truth_source,
        ground_truth_cache_path: if enable_disk_cache {
            Some(cache_path)
        } else {
            None
        },
        ground_truth_cache_hit: false,
    })
}
