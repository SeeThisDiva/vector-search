//! Dataset Loading and Prefix `.ivecs` Ground Truth Resolution for Phase 7.

use std::path::PathBuf;
use std::time::Instant;

use vector_search::cli_util::SiftPresetChoice;
use vector_search::dataset::{
    generate_synthetic_sift, load_custom_dataset_with_options, load_or_download_sift_with_options,
    read_ivecs_max_rows, write_ivecs, DatasetSource, LoadedDataset,
};
use vector_search::distance::Metric;
use vector_search::error::{Result, VectorSearchError};
use vector_search::exact_knn::ExactKnn;
use vector_search::matrix::RowMatrix;
use vector_search::recall::verify_self_recall;

use super::cli::Phase7CliConfig;

pub fn load_benchmark_dataset(cfg: &Phase7CliConfig) -> Result<(LoadedDataset, String)> {
    if cfg.synthetic {
        let num_base = cfg.scale.unwrap_or(100_000);
        let num_query = cfg.queries.unwrap_or(10_000);
        let t0 = Instant::now();
        let (base, query) =
            generate_synthetic_sift(num_base, num_query, cfg.dim, cfg.num_clusters, cfg.seed)?;
        let loaded = LoadedDataset {
            base,
            query,
            official_ground_truth: None,
            source: DatasetSource::Synthetic { seed: cfg.seed },
            load_time_s: t0.elapsed().as_secs_f64(),
        };
        return Ok((loaded, "synthetic".to_string()));
    }

    if let (Some(base_path), Some(query_path)) = (&cfg.base_fvecs, &cfg.query_fvecs) {
        let loaded =
            load_custom_dataset_with_options(base_path, query_path, None, cfg.scale, cfg.queries)?;
        if let Some(req_scale) = cfg.scale {
            if loaded.base.rows() < req_scale {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --scale {req_scale}, but {} contains only {} vectors",
                    base_path.display(),
                    loaded.base.rows()
                )));
            }
        }
        if let Some(req_q) = cfg.queries {
            if loaded.query.rows() < req_q {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --queries {req_q}, but {} contains only {} queries",
                    query_path.display(),
                    loaded.query.rows()
                )));
            }
        }
        return Ok((loaded, "custom".to_string()));
    }

    let preset = cfg.dataset.to_preset();
    let max_base = match cfg.scale {
        Some(s) => {
            if s > preset.expected_num_base() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --scale {s} exceeds {} base size ({})",
                    preset.as_str(),
                    preset.expected_num_base()
                )));
            }
            Some(s)
        }
        None => Some(match cfg.dataset {
            SiftPresetChoice::Sift1M => 100_000,
            SiftPresetChoice::SiftSmall => preset.expected_num_base(),
        }),
    };
    let max_queries = match cfg.queries {
        Some(q) => {
            if q > preset.expected_num_query() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --queries {q} exceeds {} query size ({})",
                    preset.as_str(),
                    preset.expected_num_query()
                )));
            }
            Some(q)
        }
        None => Some(match cfg.dataset {
            SiftPresetChoice::Sift1M => 10_000,
            SiftPresetChoice::SiftSmall => preset.expected_num_query(),
        }),
    };

    let loaded = load_or_download_sift_with_options(
        preset,
        &cfg.data_dir,
        !cfg.offline,
        cfg.seed,
        max_base,
        max_queries,
        true,
    )?;

    if !loaded.source.is_authentic_sift() {
        let reason = match &loaded.source {
            DatasetSource::SyntheticFallback { reason, .. } => reason.as_str(),
            _ => "authentic SIFT dataset unavailable",
        };
        return Err(VectorSearchError::InvalidArgument(format!(
            "Authentic {} dataset unavailable ({reason}). Pass --synthetic explicitly to run on synthetic vectors.",
            preset.as_str()
        )));
    }

    Ok((loaded, preset.as_str().to_string()))
}

pub struct GroundTruthResolved {
    pub matrix: RowMatrix<i32>,
    pub source: String,
    pub path: Option<PathBuf>,
}

fn validate_and_slice_gt(
    src: &RowMatrix<i32>,
    num_query: usize,
    num_base: usize,
    k: usize,
) -> Result<RowMatrix<i32>> {
    if src.rows() < num_query || src.cols() < k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Ground truth matrix shape ({}x{}) incompatible with Q={num_query}, k={k}",
            src.rows(),
            src.cols()
        )));
    }
    let max_id = num_base as i32;
    let mut data = Vec::with_capacity(num_query * k);
    for q in 0..num_query {
        let row = &src.row(q)[..k];
        for &id in row {
            if id < 0 || id >= max_id {
                return Err(VectorSearchError::CorruptData(format!(
                    "Ground truth ID {id} out of range [0, {max_id}) at query {q}"
                )));
            }
        }
        data.extend_from_slice(row);
    }
    let sliced = RowMatrix::new(num_query, k, data)?;
    verify_self_recall(&sliced, k)?;
    Ok(sliced)
}

pub fn prepare_exact_gt_file(loaded: &LoadedDataset, cfg: &Phase7CliConfig) -> Result<PathBuf> {
    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    if num_base < cfg.k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "num_base ({num_base}) must be >= k ({})",
            cfg.k
        )));
    }
    let cache_k = 100.max(cfg.k).min(num_base);
    let target_path = cfg
        .gt_path
        .clone()
        .unwrap_or_else(|| cfg.data_dir.join(format!("gt_{num_base}_k{cache_k}.ivecs")));

    let t0 = Instant::now();
    let exact = ExactKnn::search_parallel_with_base(
        Metric::L2Squared,
        &loaded.base,
        &loaded.query,
        cache_k,
        0,
    )?;
    verify_self_recall(&exact.indices, cfg.k)?;
    write_ivecs(&target_path, &exact.indices)?;
    let elapsed_s = t0.elapsed().as_secs_f64();

    println!(
        "[Rust Phase 7] Wrote exact ground truth ({}x{}, N={}) to {} in {:.3}s",
        num_query,
        cache_k,
        num_base,
        target_path.display(),
        elapsed_s
    );
    Ok(target_path)
}

pub fn resolve_ground_truth(
    loaded: &LoadedDataset,
    cfg: &Phase7CliConfig,
) -> Result<GroundTruthResolved> {
    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();

    if let Some(ref gt_p) = cfg.gt_path {
        let raw_gt = read_ivecs_max_rows(gt_p, Some(num_query))?;
        let matrix = validate_and_slice_gt(&raw_gt, num_query, num_base, cfg.k)?;
        return Ok(GroundTruthResolved {
            matrix,
            source: format!("custom_ivecs ({})", gt_p.display()),
            path: Some(gt_p.clone()),
        });
    }

    match &loaded.source {
        DatasetSource::AuthenticSift1M { dir } if num_base == 1_000_000 => {
            if let Some(ref official) = loaded.official_ground_truth {
                let matrix = validate_and_slice_gt(official, num_query, num_base, cfg.k)?;
                return Ok(GroundTruthResolved {
                    matrix,
                    source: "official_texmex_sift1m_groundtruth_ivecs".to_string(),
                    path: Some(dir.join("sift_groundtruth.ivecs")),
                });
            }
        }
        DatasetSource::AuthenticSiftSmall { dir } if num_base == 10_000 => {
            if let Some(ref official) = loaded.official_ground_truth {
                let matrix = validate_and_slice_gt(official, num_query, num_base, cfg.k)?;
                return Ok(GroundTruthResolved {
                    matrix,
                    source: "official_texmex_siftsmall_groundtruth_ivecs".to_string(),
                    path: Some(dir.join("siftsmall_groundtruth.ivecs")),
                });
            }
        }
        _ => {}
    }

    if loaded.source.is_authentic_sift() {
        let cache_k = 100.max(cfg.k).min(num_base);
        let cache_path = cfg.data_dir.join(format!("gt_{num_base}_k{cache_k}.ivecs"));
        if !cfg.no_gt_cache && cache_path.is_file() {
            if let Ok(cached) = read_ivecs_max_rows(&cache_path, Some(num_query)) {
                if let Ok(matrix) = validate_and_slice_gt(&cached, num_query, num_base, cfg.k) {
                    return Ok(GroundTruthResolved {
                        matrix,
                        source: format!("cached_exact_knn_prefix_ivecs ({})", cache_path.display()),
                        path: Some(cache_path),
                    });
                }
            }
        }

        let exact = ExactKnn::search_parallel_with_base(
            Metric::L2Squared,
            &loaded.base,
            &loaded.query,
            cache_k,
            0,
        )?;
        let matrix = validate_and_slice_gt(&exact.indices, num_query, num_base, cfg.k)?;
        if !cfg.no_gt_cache {
            write_ivecs(&cache_path, &exact.indices)?;
            return Ok(GroundTruthResolved {
                matrix,
                source: format!(
                    "computed_exact_knn_prefix_ivecs (ExactKnn::search_parallel_with_base metric=l2 k={cache_k}, cached to {})",
                    cache_path.display()
                ),
                path: Some(cache_path),
            });
        }
        return Ok(GroundTruthResolved {
            matrix,
            source: format!(
                "computed_exact_knn_prefix (ExactKnn::search_parallel_with_base metric=l2 k={cache_k})"
            ),
            path: None,
        });
    }

    let exact = ExactKnn::search_parallel_with_base(
        Metric::L2Squared,
        &loaded.base,
        &loaded.query,
        cfg.k,
        0,
    )?;
    let matrix = validate_and_slice_gt(&exact.indices, num_query, num_base, cfg.k)?;
    Ok(GroundTruthResolved {
        matrix,
        source: format!(
            "computed_exact_knn (ExactKnn::search_parallel_with_base metric=l2 N={num_base}, Q={num_query}, k={})",
            cfg.k
        ),
        path: None,
    })
}
