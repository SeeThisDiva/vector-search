//! Resumable Phase 4 HNSW Parameter Sweep Runner.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::benchmark::{detect_provenance, Phase1DatasetMode};
use crate::dataset::{
    generate_synthetic_sift, load_custom_dataset_with_options, load_or_download_sift_with_options,
    DatasetSource, LoadedDataset, SiftPreset,
};
use crate::distance::Metric;
use crate::error::Result;
use crate::hardware::{current_utc_rfc3339, detect_hardware, get_peak_rss_mb};
use crate::hnsw::{HnswIndex, VisitedTracker};
use crate::recall::{compute_recall_at_k, summarize_repeated_latencies};

use super::grid::Phase4GridSpec;
use super::ground_truth::load_or_compute_phase4_ground_truth;
use super::report::{
    can_resume_existing_report, parse_resumed_summary_entry, Phase4EfEvaluation, Phase4IndexReport,
    Phase4IndexSummaryEntry, Phase4MemoryProfile, Phase4SweepSummary,
};

/// Complete configuration for a Phase 4 sweep execution.
#[derive(Debug, Clone)]
pub struct Phase4SweepConfig {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub grid: Phase4GridSpec,
    pub metric: Metric,
    pub allow_download: bool,
    pub enable_gt_disk_cache: bool,
    pub resume: bool,
    pub max_queries: Option<usize>,
    pub expected_commit_override: Option<String>,
}

impl Default for Phase4SweepConfig {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            data_cache_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("results/phase4"),
            grid: Phase4GridSpec::default(),
            metric: Metric::L2Squared,
            allow_download: true,
            enable_gt_disk_cache: true,
            resume: true,
            max_queries: None,
            expected_commit_override: None,
        }
    }
}

/// Executes a Phase 4 resumable parameter sweep.
pub fn run_phase4_sweep(config: &Phase4SweepConfig) -> Result<Phase4SweepSummary> {
    let mut prov = detect_provenance();
    if let Some(ref override_commit) = config.expected_commit_override {
        prov.git_commit = override_commit.clone();
    }
    let current_commit = prov.git_commit.clone();
    let hardware = detect_hardware();

    if !config.output_dir.exists() {
        fs::create_dir_all(&config.output_dir)?;
    }

    // 1. Load dataset (using grid.scale_vectors as max_base)
    let loaded: LoadedDataset = match &config.dataset_mode {
        Phase1DatasetMode::Sift(preset) => load_or_download_sift_with_options(
            *preset,
            &config.data_cache_dir,
            config.allow_download,
            config.grid.seed,
            config.grid.scale_vectors,
            config.max_queries,
            true,
        )?,
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let eff_base = config
                .grid
                .scale_vectors
                .map_or(*num_base, |lim| lim.min(*num_base));
            let eff_query = config
                .max_queries
                .map_or(*num_query, |lim| lim.min(*num_query));
            let (base, query) = generate_synthetic_sift(
                eff_base,
                eff_query,
                *dim,
                *num_clusters,
                config.grid.seed,
            )?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic {
                    seed: config.grid.seed,
                },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => load_custom_dataset_with_options(
            base_fvecs,
            query_fvecs,
            gt_ivecs.as_deref(),
            config.grid.scale_vectors,
            config.max_queries,
        )?,
    };

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();

    // 2. Load or compute prefix ground truth once for the entire grid
    let gt = load_or_compute_phase4_ground_truth(
        &loaded,
        config.grid.k,
        config.metric,
        &config.data_cache_dir,
        config.enable_gt_disk_cache,
    )?;

    let mut summary_entries = Vec::with_capacity(config.grid.configs.len());
    let mut newly_measured_count = 0;
    let mut resumed_skipped_count = 0;

    // 3. Iterate over each index configuration
    for idx_cfg in &config.grid.configs {
        let report_filename = idx_cfg.report_filename(num_base);
        let report_path = config.output_dir.join(&report_filename);

        if config.resume
            && can_resume_existing_report(
                &report_path,
                &current_commit,
                loaded.source.kind_tag(),
                num_base,
                num_query,
                config.grid.k,
                config.grid.sweep_runs,
                &config.grid.ef_search_list,
                prov.git_dirty,
                idx_cfg,
            )
        {
            resumed_skipped_count += 1;
            let entry = parse_resumed_summary_entry(&report_path, idx_cfg, &report_filename);
            summary_entries.push(entry);
            continue;
        }

        // Build the index
        let hnsw_cfg = idx_cfg.to_hnsw_config();
        let t_build_start = Instant::now();
        let index = HnswIndex::build(&loaded.base, config.metric, hnsw_cfg)?;
        let build_time_s = t_build_start.elapsed().as_secs_f64();
        let build_qps = if build_time_s > 0.0 {
            (num_base as f64) / build_time_s
        } else {
            0.0
        };

        let max_layer = index.max_layer();
        let entry_point = index.entry_point().unwrap_or(0);
        let graph_fingerprint = index.graph_fingerprint();
        let peak_rss_mb = get_peak_rss_mb().unwrap_or(0.0);
        let graph_only_bytes_per_vec = index.graph_only_bytes_per_vector();
        let bytes_per_vec = index.bytes_per_vector();
        let graph_components = index.graph_components();
        let graph_only_bytes = index.graph_only_memory_bytes();
        let graph_only_mb = index.graph_only_memory_mb();

        let memory_profile = Phase4MemoryProfile {
            peak_rss_mb,
            graph_only_bytes_per_vector: graph_only_bytes_per_vec,
            bytes_per_vector: bytes_per_vec,
            graph_only_memory_bytes: graph_only_bytes,
            graph_only_memory_mb: graph_only_mb,
            layer0_bytes: graph_components.layer0_edges_bytes,
            upper_layers_bytes: graph_components.upper_layers_edges_bytes,
        };

        // Prepare visited tracker
        let mut tracker = VisitedTracker::new(num_base);
        let mut ef_evaluations = Vec::with_capacity(config.grid.ef_search_list.len());
        let mut monotonic_recall = true;
        let mut prev_recall = -1.0;
        let mut best_recall: Option<f64> = None;
        let mut best_qps: Option<f64> = None;

        for &ef in &config.grid.ef_search_list {
            // 1 untimed warmup pass
            let _ = index.search_batch_timed_with_tracker(
                &loaded.query,
                config.grid.k,
                ef,
                &mut tracker,
            )?;

            // sweep_runs timed passes
            let mut run_latencies = Vec::with_capacity(config.grid.sweep_runs);
            let mut last_indices = None;
            for _ in 0..config.grid.sweep_runs {
                let timed_res = index.search_batch_timed_with_tracker(
                    &loaded.query,
                    config.grid.k,
                    ef,
                    &mut tracker,
                )?;
                run_latencies.push(timed_res.latency);
                last_indices = Some(timed_res.result.indices);
            }
            let summary = summarize_repeated_latencies(&run_latencies);
            let recall_at_k = if let Some(ref ind) = last_indices {
                compute_recall_at_k(ind, &gt.matrix, config.grid.k)?
            } else {
                0.0
            };

            if prev_recall >= 0.0 && recall_at_k + 1e-9 < prev_recall {
                monotonic_recall = false;
            }
            prev_recall = recall_at_k;

            best_recall = Some(best_recall.map_or(recall_at_k, |b| b.max(recall_at_k)));
            best_qps = Some(best_qps.map_or(summary.median_latency.qps, |b| {
                b.max(summary.median_latency.qps)
            }));

            ef_evaluations.push(Phase4EfEvaluation {
                ef_search: ef,
                recall_at_k,
                summary,
            });
        }

        let preset_name = match &config.dataset_mode {
            Phase1DatasetMode::Sift(SiftPreset::SiftSmall) => "siftsmall".to_string(),
            Phase1DatasetMode::Sift(SiftPreset::Sift1M) => "sift1m".to_string(),
            Phase1DatasetMode::Synthetic { .. } => "synthetic".to_string(),
            Phase1DatasetMode::Custom { .. } => "custom".to_string(),
        };

        let report = Phase4IndexReport {
            phase: 4,
            phase_name: "Phase 4 Parameter Sweep & Pareto Frontier".to_string(),
            generated_at: current_utc_rfc3339(),
            provenance: prov.clone(),
            hardware: hardware.clone(),
            dataset_source: loaded.source.clone(),
            preset_name,
            scale_vectors: num_base,
            num_base,
            num_query,
            dim,
            k: config.grid.k,
            seed: idx_cfg.seed,
            metric: config.metric,
            ground_truth_source: gt.ground_truth_source.clone(),
            ground_truth_cache_path: gt.ground_truth_cache_path.clone(),
            ground_truth_cache_hit: gt.ground_truth_cache_hit,
            hnsw_config: idx_cfg.clone(),
            build_time_s,
            build_qps,
            max_layer,
            entry_point,
            graph_fingerprint,
            memory_profile,
            warmup_performed: true,
            measured_this_run: true,
            sweep_runs: config.grid.sweep_runs,
            ef_search_sweep: ef_evaluations,
            monotonic_recall,
        };

        fs::write(&report_path, report.to_json_string())?;
        newly_measured_count += 1;

        summary_entries.push(Phase4IndexSummaryEntry {
            m: idx_cfg.m,
            ef_construction: idx_cfg.ef_construction,
            keep_pruned_shrink: idx_cfg.keep_pruned_shrink,
            report_filename,
            status: "measured".to_string(),
            graph_fingerprint: Some(graph_fingerprint),
            graph_fingerprint_hex: Some(format!("0x{graph_fingerprint:016x}")),
            build_time_s: Some(build_time_s),
            build_qps: Some(build_qps),
            peak_rss_mb: Some(peak_rss_mb),
            graph_only_bytes_per_vector: Some(graph_only_bytes_per_vec),
            best_recall,
            best_qps,
        });
    }

    let summary = Phase4SweepSummary {
        total_configs: config.grid.configs.len(),
        newly_measured_count,
        resumed_skipped_count,
        git_commit: current_commit,
        git_dirty: prov.git_dirty,
        rustc_version: prov.rustc_version.clone(),
        provenance: prov,
        hardware,
        dataset_source: loaded.source,
        scale_vectors: num_base,
        ground_truth_source: gt.ground_truth_source,
        ground_truth_cache_hit: gt.ground_truth_cache_hit,
        configs: summary_entries,
    };

    let summary_path = config.output_dir.join("summary.json");
    fs::write(&summary_path, summary.to_json_string())?;

    Ok(summary)
}
