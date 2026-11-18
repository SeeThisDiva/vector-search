//! Shared dataset loading, `max_queries` slicing, and SIFT1M file path probing
//! for Phases 1–3.5 benchmarks.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataset::{
    generate_synthetic_sift, load_custom_dataset, load_custom_dataset_with_options,
    load_or_download_sift, load_or_download_sift_with_options, DatasetSource, LoadedDataset,
    SiftPreset,
};
use crate::error::{Result, VectorSearchError};

/// Dataset selection mode for Phase 1–3 execution.
#[derive(Debug, Clone)]
pub enum Phase1DatasetMode {
    /// Public Texmex SIFT preset (`siftsmall` 10k or `SIFT1M` 1M) with offline fallback.
    Sift(SiftPreset),
    /// Deterministic seeded synthetic clustered Gaussian SIFT manifold.
    Synthetic {
        num_base: usize,
        num_query: usize,
        dim: usize,
        num_clusters: usize,
    },
    /// User-supplied `.fvecs` (and optional `.ivecs` ground truth) files.
    Custom {
        base_fvecs: PathBuf,
        query_fvecs: PathBuf,
        gt_ivecs: Option<PathBuf>,
    },
}

/// Resolves a CLI `--dataset` choice and optional custom `.fvecs`/`.ivecs` paths into a [`Phase1DatasetMode`].
#[allow(clippy::too_many_arguments)]
pub fn resolve_cli_dataset_mode(
    dataset_choice: &str,
    base_fvecs: Option<PathBuf>,
    query_fvecs: Option<PathBuf>,
    gt_ivecs: Option<PathBuf>,
    num_base: usize,
    num_query: usize,
    dim: usize,
    num_clusters: usize,
) -> Result<Phase1DatasetMode> {
    if base_fvecs.is_some()
        || query_fvecs.is_some()
        || dataset_choice.eq_ignore_ascii_case("custom")
    {
        let (Some(b), Some(q)) = (base_fvecs, query_fvecs) else {
            return Err(VectorSearchError::InvalidArgument(
                "Custom dataset mode requires both --base-fvecs <path> and --query-fvecs <path>"
                    .to_string(),
            ));
        };
        Ok(Phase1DatasetMode::Custom {
            base_fvecs: b,
            query_fvecs: q,
            gt_ivecs,
        })
    } else {
        match dataset_choice.to_ascii_lowercase().as_str() {
            "siftsmall" => Ok(Phase1DatasetMode::Sift(SiftPreset::SiftSmall)),
            "sift1m" | "sift" => Ok(Phase1DatasetMode::Sift(SiftPreset::Sift1M)),
            "synthetic" => Ok(Phase1DatasetMode::Synthetic {
                num_base,
                num_query,
                dim,
                num_clusters,
            }),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Unsupported --dataset '{other}'. Expected 'siftsmall', 'sift1m', 'synthetic', or 'custom'."
            ))),
        }
    }
}

/// Loads or generates a dataset for Phase 1 or Phase 2 and applies optional `max_queries` slicing.
pub(crate) fn load_dataset_and_slice_queries(
    dataset_mode: &Phase1DatasetMode,
    data_cache_dir: &Path,
    allow_download: bool,
    seed: u64,
    max_queries: Option<usize>,
) -> Result<LoadedDataset> {
    let mut loaded: LoadedDataset = match dataset_mode {
        Phase1DatasetMode::Sift(preset) => {
            load_or_download_sift(*preset, data_cache_dir, allow_download, seed)?
        }
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let (base, query) =
                generate_synthetic_sift(*num_base, *num_query, *dim, *num_clusters, seed)?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic { seed },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => load_custom_dataset(base_fvecs, query_fvecs, gt_ivecs.as_deref())?,
    };

    if let Some(limit) = max_queries {
        if limit == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "max_queries must be positive (> 0) when specified".to_string(),
            ));
        }
        if limit < loaded.query.rows() {
            loaded.query = loaded.query.slice_first_rows(limit);
            if let Some(gt) = &loaded.official_ground_truth {
                loaded.official_ground_truth = Some(gt.slice_first_rows(limit));
            }
        }
    }

    Ok(loaded)
}

/// Loads a dataset for Phase 3, reading at most `max_base` / `max_queries` rows directly from disk
/// so per-scale Peak RSS is never polluted by unrequested rows.
///
/// Returns `(loaded_dataset, custom_full_base_rows)`.
pub(crate) fn load_phase3_dataset(
    dataset_mode: &Phase1DatasetMode,
    data_cache_dir: &Path,
    allow_download: bool,
    seed: u64,
    max_base: Option<usize>,
    max_queries: Option<usize>,
) -> Result<(LoadedDataset, Option<usize>)> {
    let mut custom_full_base_rows: Option<usize> = None;
    let loaded: LoadedDataset = match dataset_mode {
        Phase1DatasetMode::Sift(preset) => load_or_download_sift_with_options(
            *preset,
            data_cache_dir,
            allow_download,
            seed,
            max_base,
            max_queries,
            true,
        )?,
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let eff_base = max_base.map_or(*num_base, |lim| lim.min(*num_base));
            let eff_query = max_queries.map_or(*num_query, |lim| lim.min(*num_query));
            let (base, query) =
                generate_synthetic_sift(eff_base, eff_query, *dim, *num_clusters, seed)?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic { seed },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => {
            let ds = load_custom_dataset_with_options(
                base_fvecs,
                query_fvecs,
                gt_ivecs.as_deref(),
                max_base,
                max_queries,
            )?;
            let d = ds.base.cols();
            if d > 0 {
                let rec_bytes = ((1 + d) as u64) * 4;
                if let Ok(meta) = fs::metadata(base_fvecs) {
                    custom_full_base_rows = usize::try_from(meta.len() / rec_bytes).ok();
                }
            }
            ds
        }
    };

    Ok((loaded, custom_full_base_rows))
}

/// Probes `data_dir/sift1m/{sift_base.fvecs,sift_query.fvecs}` and `data_dir/{sift_base.fvecs,sift_query.fvecs}`.
pub(crate) fn resolve_sift1m_fvecs_paths(data_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let sub_base = data_dir.join("sift1m").join("sift_base.fvecs");
    let sub_query = data_dir.join("sift1m").join("sift_query.fvecs");
    if sub_base.exists() && sub_query.exists() {
        return Ok((sub_base, sub_query));
    }

    let root_base = data_dir.join("sift_base.fvecs");
    let root_query = data_dir.join("sift_query.fvecs");
    if root_base.exists() && root_query.exists() {
        return Ok((root_base, root_query));
    }

    Err(VectorSearchError::InvalidArgument(format!(
        "SIFT1M corpus files not found in {}. Base: {}, Query: {}",
        data_dir.display(),
        sub_base.display(),
        sub_query.display()
    )))
}
