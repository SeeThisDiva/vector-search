//! Dedicated CLI Parser and Runner for Phase 4 Resumable Parameter Sweeps.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use crate::benchmark::{Phase1DatasetMode, ReportProvenance};
use crate::cli_util::{next_str, parse_next, parse_positive_csv_usizes};
use crate::dataset::{DatasetSource, SiftPreset};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::hardware::HardwareInfo;

use super::grid::{Phase4GridSpec, Phase4IndexConfig};
use super::report::Phase4SweepSummary;
use super::runner::{run_phase4_sweep, Phase4SweepConfig};

/// Executes Phase 4 CLI with given arguments.
pub fn run_phase4_cli(args: &[String]) -> Result<Phase4SweepSummary> {
    let mut grid_path: Option<PathBuf> = None;
    let mut dataset_str = "sift1m".to_string();
    let mut cli_scale: Option<usize> = None;
    let mut cli_max_queries: Option<usize> = None;
    let mut cli_dim: usize = 128;
    let mut cli_k: Option<usize> = None;
    let mut cli_m_list: Option<Vec<usize>> = None;
    let mut cli_efc_list: Option<Vec<usize>> = None;
    let mut cli_ef_search_list: Option<Vec<usize>> = None;
    let mut cli_sweep_runs: Option<usize> = None;
    let mut cli_seed: Option<u64> = None;
    let mut output_dir = PathBuf::from("results/phase4");
    let mut data_dir = PathBuf::from("data");
    let mut resume = true;
    let mut allow_download = true;
    let mut enable_gt_disk_cache = true;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--help" | "-h" => {
                println!(
                    "Usage: phase4 [OPTIONS]\n\n\
                    Options:\n  \
                      --grid <path>               Path to .grid specification file\n  \
                      --dataset <name>            Dataset preset: siftsmall, sift1m, synthetic (default: sift1m)\n  \
                      --scale, --max-base <N>     Base vector scale (default: 100000)\n  \
                      --max-queries <N>           Query count limit\n  \
                      --dim <N>                   Dimensionality for synthetic dataset (default: 128)\n  \
                      --k <N>                     Number of nearest neighbors (default: 10)\n  \
                      --m-list <csv>              Comma-separated M values (default: 16)\n  \
                      --ef-construction-list <csv> Comma-separated efConstruction values (default: 100)\n  \
                      --ef-search-list <csv>      Comma-separated efSearch sweep ladder\n  \
                      --sweep-runs <N>            Repeated latency passes per efSearch (default: 3)\n  \
                      --seed <N>                  PRNG seed (default: 42)\n  \
                      --output-dir <path>         Output directory for JSON reports (default: results/phase4)\n  \
                      --data-dir <path>           Dataset and ground-truth cache directory (default: data)\n  \
                      --no-resume                 Force re-execution of previously measured configs\n  \
                      --offline                   Disallow network download of datasets\n  \
                      --no-gt-cache               Disable disk caching of prefix ground truth\n  \
                      -h, --help                  Print help information"
                );
                return Ok(Phase4SweepSummary {
                    total_configs: 0,
                    newly_measured_count: 0,
                    resumed_skipped_count: 0,
                    git_commit: "none".to_string(),
                    git_dirty: false,
                    rustc_version: "none".to_string(),
                    provenance: ReportProvenance {
                        git_commit: "none".to_string(),
                        git_dirty: false,
                        rustc_version: "none".to_string(),
                        power_mode: "none".to_string(),
                        processor_affinity_mask: "not_set".to_string(),
                    },
                    hardware: HardwareInfo {
                        cpu_model: "none".to_string(),
                        logical_cores: 0,
                        total_ram_gb: None,
                        os: "none".to_string(),
                        arch: "none".to_string(),
                    },
                    dataset_source: DatasetSource::Synthetic { seed: 42 },
                    scale_vectors: 0,
                    ground_truth_source: "none".to_string(),
                    ground_truth_cache_hit: false,
                    configs: Vec::new(),
                });
            }
            "--grid" => {
                grid_path = Some(PathBuf::from(next_str(args, &mut i, "--grid")?));
            }
            "--dataset" => {
                dataset_str = next_str(args, &mut i, "--dataset")?.to_string();
            }
            "--scale" | "--max-base" => {
                cli_scale = Some(parse_next(args, &mut i, flag)?);
            }
            "--max-queries" => {
                cli_max_queries = Some(parse_next(args, &mut i, "--max-queries")?);
            }
            "--dim" => {
                cli_dim = parse_next(args, &mut i, "--dim")?;
            }
            "--k" => {
                cli_k = Some(parse_next(args, &mut i, "--k")?);
            }
            "--m-list" => {
                let raw = next_str(args, &mut i, "--m-list")?;
                cli_m_list = Some(parse_positive_csv_usizes(raw, "--m-list")?);
            }
            "--ef-construction-list" => {
                let raw = next_str(args, &mut i, "--ef-construction-list")?;
                cli_efc_list = Some(parse_positive_csv_usizes(raw, "--ef-construction-list")?);
            }
            "--ef-search-list" => {
                let raw = next_str(args, &mut i, "--ef-search-list")?;
                cli_ef_search_list = Some(parse_positive_csv_usizes(raw, "--ef-search-list")?);
            }
            "--sweep-runs" => {
                cli_sweep_runs = Some(parse_next(args, &mut i, "--sweep-runs")?);
            }
            "--seed" => {
                cli_seed = Some(parse_next(args, &mut i, "--seed")?);
            }
            "--output-dir" => {
                output_dir = PathBuf::from(next_str(args, &mut i, "--output-dir")?);
            }
            "--data-dir" => {
                data_dir = PathBuf::from(next_str(args, &mut i, "--data-dir")?);
            }
            "--no-resume" => {
                resume = false;
                i += 1;
            }
            "--offline" => {
                allow_download = false;
                i += 1;
            }
            "--no-gt-cache" => {
                enable_gt_disk_cache = false;
                i += 1;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI argument: '{unknown}'"
                )));
            }
        }
    }

    let mut grid = if let Some(ref gp) = grid_path {
        Phase4GridSpec::from_file(gp)?
    } else {
        let m_list = cli_m_list.unwrap_or_else(|| vec![16]);
        let efc_list = cli_efc_list.unwrap_or_else(|| vec![100]);
        let seed = cli_seed.unwrap_or(42);
        let mut configs = Vec::new();
        for &m_val in &m_list {
            for &efc_val in &efc_list {
                if efc_val >= m_val {
                    configs.push(Phase4IndexConfig {
                        m: m_val,
                        m_max: None,
                        m_max0: None,
                        ef_construction: efc_val,
                        keep_pruned: true,
                        keep_pruned_shrink: None,
                        seed,
                        explicit_seed: false,
                    });
                }
            }
        }
        if configs.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "No valid configurations formed (ef_construction must be >= m)".to_string(),
            ));
        }

        Phase4GridSpec {
            scale_vectors: cli_scale.or(Some(100_000)),
            k: cli_k.unwrap_or(10),
            sweep_runs: cli_sweep_runs.unwrap_or(3),
            seed,
            ef_search_list: cli_ef_search_list
                .clone()
                .unwrap_or_else(|| vec![10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600]),
            configs,
        }
    };

    // Apply CLI overrides to grid if specified
    if let Some(s) = cli_scale {
        grid.scale_vectors = Some(s);
    } else if grid.scale_vectors.is_none() {
        grid.scale_vectors = Some(100_000);
    }
    if let Some(k_val) = cli_k {
        grid.k = k_val;
    }
    if let Some(sr) = cli_sweep_runs {
        grid.sweep_runs = sr;
    }
    if let Some(s) = cli_seed {
        grid.seed = s;
        for cfg in &mut grid.configs {
            if !cfg.explicit_seed {
                cfg.seed = s;
            }
        }
    }
    if let Some(ef_list) = cli_ef_search_list {
        grid.ef_search_list = ef_list;
    }

    grid.validate()?;

    let dataset_mode = match dataset_str.to_ascii_lowercase().as_str() {
        "siftsmall" => Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
        "sift1m" => Phase1DatasetMode::Sift(SiftPreset::Sift1M),
        "synthetic" => Phase1DatasetMode::Synthetic {
            num_base: grid.scale_vectors.unwrap_or(100_000),
            num_query: cli_max_queries.unwrap_or(10_000),
            dim: cli_dim,
            num_clusters: 100,
        },
        other => {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Unknown dataset '{other}': expected siftsmall, sift1m, or synthetic"
            )));
        }
    };

    let sweep_cfg = Phase4SweepConfig {
        dataset_mode,
        data_cache_dir: data_dir,
        output_dir,
        grid,
        metric: Metric::L2Squared,
        allow_download,
        enable_gt_disk_cache,
        resume,
        max_queries: cli_max_queries,
        expected_commit_override: None,
    };

    let summary = run_phase4_sweep(&sweep_cfg)?;
    println!(
        "[Phase 4] Parameter Sweep complete: {} total configs ({} measured, {} resumed/skipped). Output: {}",
        summary.total_configs,
        summary.newly_measured_count,
        summary.resumed_skipped_count,
        sweep_cfg.output_dir.display()
    );

    Ok(summary)
}
