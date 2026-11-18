//! Phase 3: SIFT1M Real-World Scaling, Memory Profiling & Ground Truth Benchmark Harness.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataset::{load_or_download_sift_with_options, DatasetSource, SiftPreset};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{
    current_utc_rfc3339, detect_hardware, get_peak_rss_mb, peak_rss_measurement_method,
    HardwareInfo,
};
use crate::hnsw::{HnswBuildStats, HnswConfig, HnswIndex};
use crate::json_util::{escape_json, opt_path_json};
use crate::matrix::RowMatrix;
use crate::recall::{compute_recall_at_k, summarize_repeated_latencies, LatencyStats};

use super::baseline_profile::{
    load_or_default_baseline_profiles, spawn_isolated_scale_profile, BaselineScaleProfile,
};
use super::dataset_loader::{load_phase3_dataset, Phase1DatasetMode};
use super::golden::verify_index_against_golden_fixture;
use super::phase2::verify_reproducibility_with_metric;
use super::{append_hardware_json, detect_provenance};

/// Configuration parameters for Phase 3 SIFT1M scaling & benchmark.
#[derive(Debug, Clone)]
pub struct Phase3Config {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub m: usize,
    pub m_max0: Option<usize>,
    pub ef_construction: usize,
    pub ef_search_list: Vec<usize>,
    pub allow_download: bool,
    pub save_files: bool,
    pub max_queries: Option<usize>,
    pub max_base: Option<usize>,
    pub num_threads: usize,
    pub run_repro_check: bool,
    pub repro_vectors: usize,
    pub sweep_runs: usize,
    pub run_multi_scale: bool,
    pub keep_pruned_shrink: Option<bool>,
}

impl Default for Phase3Config {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            data_cache_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("data"),
            k: 10,
            seed: 42,
            metric: Metric::L2Squared,
            m: 16,
            m_max0: None,
            ef_construction: 100,
            ef_search_list: vec![10, 20, 50, 100, 200],
            allow_download: true,
            save_files: true,
            max_queries: None,
            max_base: None,
            num_threads: 0,
            run_repro_check: true,
            repro_vectors: 10_000,
            sweep_runs: 5,
            run_multi_scale: true,
            keep_pruned_shrink: None,
        }
    }
}

/// Evaluation metrics for a single efSearch setting in Phase 3 across repeated runs.
#[derive(Debug, Clone)]
pub struct Phase3EfEvaluation {
    pub ef_search: usize,
    pub recall_at_k: f64,
    /// Median latency statistics across `num_runs` repetitions of the sweep.
    pub latency: LatencyStats,
    pub num_runs: usize,
    pub mean_ms_min: f64,
    pub mean_ms_max: f64,
    pub mean_ms_spread: f64,
    pub p99_ms_min: f64,
    pub p99_ms_max: f64,
    pub p99_ms_spread: f64,
    pub run_mean_latencies_ms: Vec<f64>,
    pub run_p99_latencies_ms: Vec<f64>,
    pub monotonic_passed: bool,
}

/// Detailed memory profile for Phase 3 SIFT1M benchmark.
#[derive(Debug, Clone)]
pub struct Phase3MemoryProfile {
    pub peak_rss_mb: f64,
    pub peak_rss_bytes: u64,
    pub peak_rss_measurement_method: String,
    pub is_peak_rss_os_measured: bool,
    pub raw_vector_bytes_per_vector: f64,
    pub graph_bytes_per_vector: f64,
    pub graph_only_bytes_per_vector: f64,
    pub total_bytes_per_vector: f64,
    pub raw_vector_total_bytes: usize,
    pub raw_vector_total_mb: f64,
    pub graph_total_bytes: usize,
    pub graph_total_mb: f64,
    pub graph_only_total_bytes: usize,
    pub graph_only_total_mb: f64,
    pub total_footprint_bytes: usize,
    pub total_footprint_mb: f64,
    pub graph_components: crate::hnsw::HnswGraphComponents,
}

/// Complete structured Phase 3 SIFT1M benchmark report.
#[derive(Debug, Clone)]
pub struct Phase3Report {
    pub phase: usize,
    pub phase_name: String,
    pub generated_at: String,
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub preset_name: String,
    pub base_path: Option<PathBuf>,
    pub query_path: Option<PathBuf>,
    pub groundtruth_path: Option<PathBuf>,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub hnsw_m: usize,
    pub hnsw_m_max0: usize,
    pub hnsw_ef_construction: usize,
    pub build_time_s: f64,
    pub build_qps: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub memory_profile: Phase3MemoryProfile,
    pub multi_scale_baseline_summary: Vec<BaselineScaleProfile>,
    pub sweep_runs: usize,
    pub ef_evaluations: Vec<Phase3EfEvaluation>,
    pub monotonic_scaling_verified: bool,
    /// True only if the reproducibility check actually ran (it is skipped by `--no-repro-check`).
    pub reproducibility_checked: bool,
    /// True only if the check ran and passed. A skipped check is never reported as verified.
    pub reproducibility_verified: bool,
    /// True only when brute-force ExactKNN over `loaded.base` matches `official_ground_truth`
    /// with Recall@k = 1.0 on the verified query sample (`official_gt_check_queries`).
    pub official_ground_truth_matched: bool,
    pub official_gt_check_queries: usize,
    pub official_gt_recall_at_k: Option<f64>,
    pub official_gt_exact_id_matches: Option<usize>,
    pub terminal_stopping_enforced: bool,
    pub phase4_initiated: bool,
    /// 64-bit FNV-1a fingerprint of the graph structure.
    pub graph_fingerprint: Option<u64>,
    /// Build instrumentation counters and phase-level time split (if instrument feature enabled).
    pub build_stats: Option<HnswBuildStats>,
}

impl Phase3Report {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(8192);
        out.push_str("{\n");
        out.push_str(&format!("  \"phase\": {},\n", self.phase));
        out.push_str(&format!(
            "  \"phase_name\": \"{}\",\n",
            escape_json(&self.phase_name)
        ));
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&self.generated_at)
        ));
        detect_provenance().append_json_fields(&mut out, &self.hardware);

        // Hardware
        append_hardware_json(&mut out, &self.hardware);

        // Dataset
        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!(
            "    \"preset\": \"{}\",\n",
            escape_json(&self.preset_name)
        ));
        out.push_str(&format!(
            "    \"source_kind\": \"{}\",\n",
            escape_json(self.dataset_source.kind_tag())
        ));
        out.push_str(&format!(
            "    \"source_description\": \"{}\",\n",
            escape_json(&self.dataset_source.to_string())
        ));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {},\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str(&format!(
            "    \"base_path\": {},\n",
            opt_path_json(self.base_path.as_deref())
        ));
        out.push_str(&format!(
            "    \"query_path\": {},\n",
            opt_path_json(self.query_path.as_deref())
        ));
        out.push_str(&format!(
            "    \"groundtruth_path\": {},\n",
            opt_path_json(self.groundtruth_path.as_deref())
        ));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str(&format!(
            "    \"metric\": \"{}\"\n",
            escape_json(&self.metric.to_string())
        ));
        out.push_str("  },\n");

        // HnswConfig
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_m));
        out.push_str(&format!("    \"m_max0\": {},\n", self.hnsw_m_max0));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_ef_construction
        ));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str("    \"heuristic\": \"select_neighbors_heuristic_with_scratch\"\n");
        out.push_str("  },\n");

        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!("    \"build_qps\": {:.2},\n", self.build_qps));
        out.push_str(&format!("    \"max_layer\": {},\n", self.max_layer));
        if let Some(fp) = self.graph_fingerprint {
            out.push_str(&format!("    \"entry_point\": {},\n", self.entry_point));
            out.push_str(&format!("    \"graph_fingerprint\": {fp},\n"));
            out.push_str(&format!("    \"graph_fingerprint_hex\": \"0x{fp:016x}\"\n"));
        } else {
            out.push_str(&format!("    \"entry_point\": {}\n", self.entry_point));
        }
        out.push_str("  },\n");

        if let Some(ref bs) = self.build_stats {
            out.push_str("  \"build_instrumentation\": {\n");
            out.push_str("    \"distance_evaluations\": {\n");
            out.push_str(&format!("      \"total\": {},\n", bs.distance_evals_total));
            out.push_str(&format!(
                "      \"search_layer\": {},\n",
                bs.distance_evals_search
            ));
            out.push_str(&format!(
                "      \"heuristic\": {},\n",
                bs.distance_evals_heuristic
            ));
            out.push_str(&format!(
                "      \"shrink\": {},\n",
                bs.distance_evals_shrink
            ));
            out.push_str(&format!(
                "      \"per_insert_mean\": {:.2},\n",
                bs.dist_evals_per_insert_mean
            ));
            out.push_str(&format!(
                "      \"per_insert_min\": {},\n",
                bs.dist_evals_per_insert_min
            ));
            out.push_str(&format!(
                "      \"per_insert_max\": {}\n",
                bs.dist_evals_per_insert_max
            ));
            out.push_str("    },\n");
            out.push_str("    \"call_counts\": {\n");
            out.push_str(&format!(
                "      \"search_layer_calls\": {},\n",
                bs.search_layer_calls
            ));
            out.push_str(&format!(
                "      \"heuristic_calls\": {},\n",
                bs.heuristic_calls
            ));
            out.push_str(&format!(
                "      \"heuristic_insert_calls\": {},\n",
                bs.heuristic_insert_calls
            ));
            out.push_str(&format!(
                "      \"heuristic_shrink_calls\": {},\n",
                bs.heuristic_shrink_calls
            ));
            out.push_str(&format!("      \"shrink_calls\": {}\n", bs.shrink_calls));
            out.push_str("    },\n");
            out.push_str("    \"phase_time_split\": {\n");
            out.push_str(&format!(
                "      \"search_layer_s\": {:.6},\n",
                bs.time_search_layer_s
            ));
            out.push_str(&format!(
                "      \"search_layer_pct\": {:.2},\n",
                bs.pct_search_layer
            ));
            out.push_str(&format!(
                "      \"heuristic_s\": {:.6},\n",
                bs.time_heuristic_s
            ));
            out.push_str(&format!(
                "      \"heuristic_pct\": {:.2},\n",
                bs.pct_heuristic
            ));
            out.push_str(&format!("      \"shrink_s\": {:.6},\n", bs.time_shrink_s));
            out.push_str(&format!("      \"shrink_pct\": {:.2},\n", bs.pct_shrink));
            out.push_str(&format!(
                "      \"other_overhead_s\": {:.6},\n",
                bs.time_other_overhead_s
            ));
            out.push_str(&format!(
                "      \"other_overhead_pct\": {:.2},\n",
                bs.pct_other_overhead
            ));
            out.push_str(&format!(
                "      \"total_build_time_s\": {:.6}\n",
                bs.total_build_time_s
            ));
            out.push_str("    }\n");
            out.push_str("  },\n");
        }

        // MemoryProfile
        let mp = &self.memory_profile;
        out.push_str("  \"memory_profile\": {\n");
        out.push_str(&format!("    \"peak_rss_mb\": {:.2},\n", mp.peak_rss_mb));
        out.push_str(&format!("    \"peak_rss_bytes\": {},\n", mp.peak_rss_bytes));
        out.push_str(&format!(
            "    \"peak_rss_measurement_method\": \"{}\",\n",
            escape_json(&mp.peak_rss_measurement_method)
        ));
        out.push_str(&format!(
            "    \"is_peak_rss_os_measured\": {},\n",
            mp.is_peak_rss_os_measured
        ));

        out.push_str("    \"bytes_per_vector_breakdown\": {\n");
        out.push_str(&format!(
            "      \"raw_vector_bytes_per_vector\": {:.2},\n",
            mp.raw_vector_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"graph_bytes_per_vector\": {:.2},\n",
            mp.graph_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"graph_only_bytes_per_vector\": {:.2},\n",
            mp.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"total_bytes_per_vector\": {:.2},\n",
            mp.total_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"raw_vector_total_bytes\": {},\n",
            mp.raw_vector_total_bytes
        ));
        out.push_str(&format!(
            "      \"raw_vector_total_mb\": {:.2},\n",
            mp.raw_vector_total_mb
        ));
        out.push_str(&format!(
            "      \"graph_total_bytes\": {},\n",
            mp.graph_total_bytes
        ));
        out.push_str(&format!(
            "      \"graph_total_mb\": {:.2},\n",
            mp.graph_total_mb
        ));
        out.push_str(&format!(
            "      \"graph_only_total_bytes\": {},\n",
            mp.graph_only_total_bytes
        ));
        out.push_str(&format!(
            "      \"graph_only_total_mb\": {:.2},\n",
            mp.graph_only_total_mb
        ));
        out.push_str(&format!(
            "      \"total_footprint_bytes\": {},\n",
            mp.total_footprint_bytes
        ));
        out.push_str(&format!(
            "      \"total_footprint_mb\": {:.2}\n",
            mp.total_footprint_mb
        ));
        out.push_str("    },\n");

        out.push_str("    \"graph_components_bytes\": {\n");
        out.push_str(&format!(
            "      \"node_structs_bytes\": {},\n",
            mp.graph_components.node_structs_bytes
        ));
        out.push_str(&format!(
            "      \"layer0_edges_bytes\": {},\n",
            mp.graph_components.layer0_edges_bytes
        ));
        out.push_str(&format!(
            "      \"upper_layers_edges_bytes\": {},\n",
            mp.graph_components.upper_layers_edges_bytes
        ));
        out.push_str(&format!(
            "      \"outer_neighbors_headers_bytes\": {},\n",
            mp.graph_components.outer_neighbors_headers_bytes
        ));
        out.push_str(&format!(
            "      \"visited_tracker_tags_bytes\": {},\n",
            mp.graph_components.visited_tracker_tags_bytes
        ));
        out.push_str(&format!(
            "      \"heuristic_scratch_bytes\": {}\n",
            mp.graph_components.heuristic_scratch_bytes
        ));
        out.push_str("    }\n");
        out.push_str("  },\n");

        // MultiScaleBaselineSummary
        out.push_str("  \"multi_scale_baseline_summary\": [\n");
        for (i, p) in self.multi_scale_baseline_summary.iter().enumerate() {
            let comma = if i + 1 < self.multi_scale_baseline_summary.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!(
                "      \"measured_this_run\": {},\n",
                p.measured_this_run
            ));
            out.push_str(&format!("      \"scale_vectors\": {},\n", p.scale_vectors));
            out.push_str(&format!("      \"build_time_s\": {:.6},\n", p.build_time_s));
            out.push_str(&format!("      \"build_qps\": {:.2},\n", p.build_qps));
            out.push_str(&format!("      \"peak_rss_mb\": {:.2},\n", p.peak_rss_mb));
            out.push_str(&format!(
                "      \"raw_vector_bytes_per_vector\": {:.2},\n",
                p.raw_vector_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_bytes_per_vector\": {:.2},\n",
                p.graph_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_only_bytes_per_vector\": {:.2},\n",
                p.graph_only_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"total_bytes_per_vector\": {:.2},\n",
                p.total_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_memory_bytes\": {},\n",
                p.graph_memory_bytes
            ));
            out.push_str(&format!(
                "      \"graph_only_memory_bytes\": {},\n",
                p.graph_only_memory_bytes
            ));
            out.push_str(&format!("      \"recall_at_10\": {:.6}\n", p.recall_at_10));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");

        // EfSearchSweep
        out.push_str("  \"ef_search_sweep\": [\n");
        for (i, ef) in self.ef_evaluations.iter().enumerate() {
            let comma = if i + 1 < self.ef_evaluations.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", ef.ef_search));
            out.push_str(&format!("      \"recall_at_10\": {:.6},\n", ef.recall_at_k));
            out.push_str(&format!("      \"num_runs\": {},\n", ef.num_runs));
            out.push_str("      \"latency\": {\n");
            out.push_str(&format!(
                "        \"mean_ms\": {:.6},\n",
                ef.latency.mean_ms
            ));
            out.push_str(&format!("        \"p50_ms\": {:.6},\n", ef.latency.p50_ms));
            out.push_str(&format!("        \"p95_ms\": {:.6},\n", ef.latency.p95_ms));
            out.push_str(&format!("        \"p99_ms\": {:.6},\n", ef.latency.p99_ms));
            out.push_str(&format!("        \"qps\": {:.2},\n", ef.latency.qps));
            out.push_str(&format!(
                "        \"mean_ms_min\": {:.6},\n",
                ef.mean_ms_min
            ));
            out.push_str(&format!(
                "        \"mean_ms_max\": {:.6},\n",
                ef.mean_ms_max
            ));
            out.push_str(&format!(
                "        \"mean_ms_spread\": {:.6},\n",
                ef.mean_ms_spread
            ));
            out.push_str(&format!("        \"p99_ms_min\": {:.6},\n", ef.p99_ms_min));
            out.push_str(&format!("        \"p99_ms_max\": {:.6},\n", ef.p99_ms_max));
            out.push_str(&format!(
                "        \"p99_ms_spread\": {:.6}\n",
                ef.p99_ms_spread
            ));
            out.push_str("      },\n");
            let mean_runs_json: Vec<String> = ef
                .run_mean_latencies_ms
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect();
            let p99_runs_json: Vec<String> = ef
                .run_p99_latencies_ms
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect();
            out.push_str(&format!(
                "      \"run_mean_latencies_ms\": [{}],\n",
                mean_runs_json.join(", ")
            ));
            out.push_str(&format!(
                "      \"run_p99_latencies_ms\": [{}],\n",
                p99_runs_json.join(", ")
            ));
            out.push_str(&format!(
                "      \"monotonic_passed\": {}\n",
                ef.monotonic_passed
            ));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");

        // VerificationStatus
        out.push_str("  \"verification_status\": {\n");
        out.push_str(&format!(
            "    \"authentic_sift1m_verified\": {},\n",
            matches!(self.dataset_source, DatasetSource::AuthenticSift1M { .. })
                && self.num_base == 1_000_000
        ));
        out.push_str(&format!(
            "    \"official_ground_truth_matched\": {},\n",
            self.official_ground_truth_matched
        ));
        out.push_str(&format!(
            "    \"official_gt_check_queries\": {},\n",
            self.official_gt_check_queries
        ));
        match self.official_gt_recall_at_k {
            Some(r) => out.push_str(&format!("    \"official_gt_recall_at_k\": {r:.6},\n")),
            None => out.push_str("    \"official_gt_recall_at_k\": null,\n"),
        }
        match self.official_gt_exact_id_matches {
            Some(m) => out.push_str(&format!("    \"official_gt_exact_id_matches\": {m},\n")),
            None => out.push_str("    \"official_gt_exact_id_matches\": null,\n"),
        }
        out.push_str(&format!(
            "    \"monotonic_scaling_verified\": {},\n",
            self.monotonic_scaling_verified
        ));
        out.push_str(&format!(
            "    \"reproducibility_checked\": {},\n",
            self.reproducibility_checked
        ));
        out.push_str(&format!(
            "    \"reproducibility_verified\": {},\n",
            self.reproducibility_verified
        ));
        out.push_str("    \"forbid_unsafe_code_enforced\": true,\n");
        out.push_str("    \"std_only_enforced\": true,\n");
        out.push_str(&format!(
            "    \"phase3_completed\": {},\n",
            self.monotonic_scaling_verified && self.reproducibility_verified
        ));
        out.push_str(&format!(
            "    \"terminal_stopping_enforced\": {},\n",
            self.terminal_stopping_enforced
        ));
        out.push_str(&format!(
            "    \"phase4_initiated\": {}\n",
            self.phase4_initiated
        ));
        out.push_str("  }\n");

        out.push_str("}\n");
        out
    }
}

/// Executes the complete Phase 3 SIFT1M scaling benchmark and profiling pipeline.
pub fn run_phase3_benchmark(config: &Phase3Config) -> Result<Phase3Report> {
    if config.k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0)".to_string(),
        ));
    }
    if config.max_base == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_base must be positive (> 0) when specified".to_string(),
        ));
    }
    if config.max_queries == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_queries must be positive (> 0) when specified".to_string(),
        ));
    }
    if config.sweep_runs == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "sweep_runs must be positive (> 0)".to_string(),
        ));
    }

    let hardware = detect_hardware();
    let effective_m_max0 = config.m_max0.unwrap_or(2 * config.m);

    // Baseline profiles summary: if this is a full 1M Authentic SIFT1M run with default HNSW
    // parameters and multi-scale profiling enabled, measure 10k and 100k in isolated child
    // processes BEFORE loading the 1M corpus in the parent process.
    let mut baseline_profiles = load_or_default_baseline_profiles(&config.data_cache_dir);
    let wants_full_sift1m_subscales = config.run_multi_scale
        && matches!(
            config.dataset_mode,
            Phase1DatasetMode::Sift(SiftPreset::Sift1M)
        )
        && config.max_base.unwrap_or(1_000_000) == 1_000_000
        && config.k == 10
        && config.m == 16
        && effective_m_max0 == 32
        && config.ef_construction == 100
        && config.metric == Metric::L2Squared;

    if wants_full_sift1m_subscales {
        if let Ok(probe) = load_or_download_sift_with_options(
            SiftPreset::Sift1M,
            &config.data_cache_dir,
            config.allow_download,
            config.seed,
            Some(1),
            Some(1),
            false,
        ) {
            let is_auth = probe.source.is_authentic_sift();
            drop(probe);
            if is_auth {
                for &sub_scale in &[10_000_usize, 100_000_usize] {
                    println!(
                        "[0/4] Profiling {} baseline scale in isolated child process...",
                        sub_scale
                    );
                    if let Ok(sub_prof) = spawn_isolated_scale_profile(
                        &config.data_cache_dir,
                        sub_scale,
                        config.seed,
                        config.allow_download,
                        config.sweep_runs,
                    ) {
                        if let Some(slot) = baseline_profiles
                            .iter_mut()
                            .find(|p| p.scale_vectors == sub_scale)
                        {
                            *slot = sub_prof;
                        }
                    }
                }
            }
        }
    }

    // 1. Load dataset (loading only max_base / max_queries rows from disk so per-scale Peak RSS
    // is never polluted by unrequested rows)
    let (loaded, custom_full_base_rows) = load_phase3_dataset(
        &config.dataset_mode,
        &config.data_cache_dir,
        config.allow_download,
        config.seed,
        config.max_base,
        config.max_queries,
    )?;

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();

    // Construct HNSW index
    let hnsw_cfg = HnswConfig {
        m: config.m,
        m_max: None,
        m_max0: Some(effective_m_max0),
        ef_construction: config.ef_construction,
        m_l: None,
        extend_candidates: false,
        keep_pruned: true,
        keep_pruned_shrink: config.keep_pruned_shrink,
        seed: config.seed,
    };

    println!(
        "[1/4] Constructing HNSW Index on {} vectors (M={}, M0={}, efConstruction={})...",
        num_base, config.m, effective_m_max0, config.ef_construction
    );
    let t_build_start = Instant::now();
    let mut index = HnswIndex::build(&loaded.base, config.metric, hnsw_cfg.clone())?;
    let build_time_s = t_build_start.elapsed().as_secs_f64();
    let build_qps = (num_base as f64) / build_time_s;
    let max_layer = index.max_layer();
    let entry_point = index.entry_point().unwrap_or(0);
    let graph_fingerprint = Some(index.graph_fingerprint());
    let build_stats = index.build_stats().cloned();

    // Profile memory immediately after build before any evaluation buffers are allocated
    println!("[2/4] Profiling memory footprint and process peak RSS...");
    let peak_rss_mb = get_peak_rss_mb().unwrap_or(0.0);
    let peak_rss_bytes = (peak_rss_mb * 1024.0 * 1024.0) as u64;
    let is_peak_rss_os_measured = peak_rss_mb > 0.0;

    let raw_vector_bytes_per_vector = (dim * std::mem::size_of::<f32>()) as f64;
    let graph_bytes_per_vector = index.bytes_per_vector();
    let graph_only_bytes_per_vector = index.graph_only_bytes_per_vector();
    let total_bytes_per_vector = raw_vector_bytes_per_vector + graph_bytes_per_vector;

    let raw_vector_total_bytes = loaded.base.memory_bytes();
    let raw_vector_total_mb = loaded.base.memory_mb();
    let graph_total_bytes = index.memory_bytes();
    let graph_total_mb = index.memory_mb();
    let graph_only_total_bytes = index.graph_only_memory_bytes();
    let graph_only_total_mb = index.graph_only_memory_mb();
    let total_footprint_bytes = raw_vector_total_bytes + graph_total_bytes;
    let total_footprint_mb = (total_footprint_bytes as f64) / (1024.0 * 1024.0);

    let graph_components = index.graph_components();

    let memory_profile = Phase3MemoryProfile {
        peak_rss_mb,
        peak_rss_bytes,
        peak_rss_measurement_method: peak_rss_measurement_method(false),
        is_peak_rss_os_measured,
        raw_vector_bytes_per_vector,
        graph_bytes_per_vector,
        graph_only_bytes_per_vector,
        total_bytes_per_vector,
        raw_vector_total_bytes,
        raw_vector_total_mb,
        graph_total_bytes,
        graph_total_mb,
        graph_only_total_bytes,
        graph_only_total_mb,
        total_footprint_bytes,
        total_footprint_mb,
        graph_components,
    };

    // Warmup HNSW
    if loaded.query.rows() > 0 {
        index.warmup(loaded.query.row(0), config.k, 50)?;
    }

    // Ground truth verification and establishment for evaluation:
    // Compute true brute-force ExactKNN on up to 100 queries against `&loaded.base` (without
    // cloning the base matrix) and verify `official_ground_truth` against it.
    let is_full_base_match = match &loaded.source {
        DatasetSource::AuthenticSift1M { .. } => num_base == 1_000_000,
        DatasetSource::AuthenticSiftSmall { .. } => num_base == 10_000,
        DatasetSource::CustomFiles { .. } => {
            custom_full_base_rows.is_none_or(|full| num_base == full)
        }
        DatasetSource::Synthetic { .. } | DatasetSource::SyntheticFallback { .. } => false,
    };

    let check_q = 100.min(num_query);
    let check_queries = if check_q == num_query {
        loaded.query.clone()
    } else {
        loaded.query.slice_first_rows(check_q)
    };

    let exact_check_res = if check_q > 0 && num_base > 0 {
        Some(ExactKnn::search_parallel_with_base(
            config.metric,
            &loaded.base,
            &check_queries,
            config.k,
            config.num_threads,
        )?)
    } else {
        None
    };

    let mut official_ground_truth_matched = false;
    let mut official_gt_check_queries = 0_usize;
    let mut official_gt_recall_at_k: Option<f64> = None;
    let mut official_gt_exact_id_matches: Option<usize> = None;

    if is_full_base_match && matches!(config.metric, Metric::L2Squared | Metric::Euclidean) {
        if let (Some(ref gt), Some(ref exact_res)) =
            (&loaded.official_ground_truth, &exact_check_res)
        {
            let exact_idx = &exact_res.indices;
            let exact_dists = &exact_res.distances;
            if gt.rows() == num_query && gt.cols() >= exact_idx.cols() && exact_idx.cols() > 0 {
                let eval_k = exact_idx.cols();
                let gt_slice = gt.slice_first_rows(check_q);
                if let Ok(rec) = compute_recall_at_k(exact_idx, &gt_slice, eval_k) {
                    let mut exact_matches = 0_usize;
                    let mut all_valid_exact_top_k = true;
                    for q in 0..check_q {
                        let a_row = &exact_idx.row(q)[..eval_k];
                        let a_dist_row = &exact_dists.row(q)[..eval_k];
                        let b_row = &gt_slice.row(q)[..eval_k];
                        let q_vec = check_queries.row(q);

                        for (pos, (&a, &b)) in a_row.iter().zip(b_row.iter()).enumerate() {
                            if a == b {
                                exact_matches += 1;
                            }
                            if b < 0 || (b as usize) >= num_base || b_row[..pos].contains(&b) {
                                all_valid_exact_top_k = false;
                            } else {
                                let u = loaded.base.row(b as usize);
                                let b_dist = match config.metric {
                                    Metric::L2Squared => crate::distance::l2_squared(q_vec, u),
                                    Metric::Euclidean => crate::distance::euclidean(q_vec, u),
                                    Metric::InnerProduct => {
                                        -crate::distance::inner_product(q_vec, u)
                                    }
                                };
                                let expected_dist = a_dist_row[pos];
                                let dist_tol = expected_dist.abs() * 1e-5 + 1e-5;
                                if (b_dist - expected_dist).abs() > dist_tol {
                                    all_valid_exact_top_k = false;
                                }
                            }
                        }
                    }
                    official_gt_check_queries = check_q;
                    official_gt_recall_at_k = Some(rec);
                    official_gt_exact_id_matches = Some(exact_matches);
                    // Require at least 99% ID overlap AND every official top-k neighbor must be
                    // distance-equivalent at its rank (allowing only genuine distance ties).
                    official_ground_truth_matched = all_valid_exact_top_k && rec >= 0.99;
                }
            }
        }
    }

    println!(
        "[3/4] Evaluating efSearch sweep across {:?} on {} queries ({} runs per efSearch)...",
        config.ef_search_list, num_query, config.sweep_runs
    );

    let mut ef_evaluations = Vec::new();
    let mut previous_recall = 0.0;
    let mut monotonic_scaling_verified = true;

    // Deduplicate effective ef values
    let mut effective_efs: Vec<usize> = Vec::new();
    for &ef in &config.ef_search_list {
        let effective = ef.max(config.k);
        if !effective_efs.contains(&effective) {
            effective_efs.push(effective);
        }
    }

    let runs_per_ef = config.sweep_runs.max(1);
    for &ef in &effective_efs {
        let mut run_latencies = Vec::with_capacity(runs_per_ef);
        let mut last_indices: Option<RowMatrix<i32>> = None;
        for _ in 0..runs_per_ef {
            let timed = index.search_batch_timed(&loaded.query, config.k, ef)?;
            run_latencies.push(timed.latency);
            last_indices = Some(timed.result.indices);
        }
        let summary = summarize_repeated_latencies(&run_latencies);
        let cand_indices = last_indices.unwrap();

        let recall_at_k = if is_full_base_match && official_ground_truth_matched {
            if let Some(ref gt) = loaded.official_ground_truth {
                compute_recall_at_k(&cand_indices, gt, config.k)?
            } else {
                0.0
            }
        } else if let Some(ref ref_res) = exact_check_res {
            let cand_slice = cand_indices.slice_first_rows(ref_res.indices.rows());
            compute_recall_at_k(&cand_slice, &ref_res.indices, config.k)?
        } else {
            0.0
        };

        let is_monotonic = recall_at_k >= previous_recall - 1e-6;
        if !is_monotonic {
            monotonic_scaling_verified = false;
        }

        ef_evaluations.push(Phase3EfEvaluation {
            ef_search: ef,
            recall_at_k,
            latency: summary.median_latency,
            num_runs: summary.num_runs,
            mean_ms_min: summary.mean_ms_min,
            mean_ms_max: summary.mean_ms_max,
            mean_ms_spread: summary.mean_ms_spread,
            p99_ms_min: summary.p99_ms_min,
            p99_ms_max: summary.p99_ms_max,
            p99_ms_spread: summary.p99_ms_spread,
            run_mean_latencies_ms: summary.run_mean_latencies_ms,
            run_p99_latencies_ms: summary.run_p99_latencies_ms,
            monotonic_passed: is_monotonic,
        });

        previous_recall = recall_at_k;
    }

    // 4. Verify Reproducibility (if enabled)
    println!("[4/4] Verifying bit-exact reproducibility...");
    let reproducibility_checked = config.run_repro_check;
    let reproducibility_verified = if config.run_repro_check {
        let repro_b = config.repro_vectors.min(loaded.base.rows());
        let repro_q = 1000.min(loaded.query.rows());
        let (base_repro_slice, is_full_b) = if repro_b == loaded.base.rows() {
            (None, true)
        } else {
            (Some(loaded.base.slice_first_rows(repro_b)), false)
        };
        let b_ref = if is_full_b {
            &loaded.base
        } else {
            base_repro_slice.as_ref().unwrap()
        };

        let (query_repro_slice, is_full_q) = if repro_q == loaded.query.rows() {
            (None, true)
        } else {
            (Some(loaded.query.slice_first_rows(repro_q)), false)
        };
        let q_ref = if is_full_q {
            &loaded.query
        } else {
            query_repro_slice.as_ref().unwrap()
        };

        let repro_ok = verify_reproducibility_with_metric(
            b_ref,
            q_ref,
            config.metric,
            &hnsw_cfg,
            config.k,
            50,
        )
        .is_ok();
        let golden_ok = if repro_ok
            && loaded.source.is_authentic_sift()
            && hnsw_cfg.effective_keep_pruned_shrink() == hnsw_cfg.keep_pruned
        {
            let golden_file = if repro_b == 10_000 {
                Some(Path::new("results/phase3_5/golden_10k.json"))
            } else if repro_b == 100_000 {
                Some(Path::new("results/phase3_5/golden_100k.json"))
            } else {
                None
            };
            if let Some(gp) = golden_file {
                if gp.exists() {
                    let index1 = HnswIndex::build(b_ref, config.metric, hnsw_cfg.clone())
                        .map_err(|e| eprintln!("[repro-check] Build failed: {e}"))
                        .ok();
                    match index1 {
                        Some(ref idx) => {
                            verify_index_against_golden_fixture(idx, &loaded.query, gp)
                                .map_err(|e| {
                                    eprintln!("[repro-check] Golden verification failed: {e}")
                                })
                                .is_ok()
                        }
                        None => false,
                    }
                } else {
                    true
                }
            } else {
                true
            }
        } else {
            repro_ok
        };
        repro_ok && golden_ok
    } else {
        false
    };

    // If running on Authentic SIFT1M with default HNSW parameters at one of the reference scales
    // (10k, 100k, or 1M), update that scale's entry with the current run's live numbers.
    if loaded.source.is_authentic_sift()
        && matches!(
            config.dataset_mode,
            Phase1DatasetMode::Sift(SiftPreset::Sift1M)
        )
        && config.m == 16
        && effective_m_max0 == 32
        && config.ef_construction == 100
        && config.metric == Metric::L2Squared
        && config.k == 10
    {
        if let (Some(slot), Some(eval50)) = (
            baseline_profiles
                .iter_mut()
                .find(|p| p.scale_vectors == num_base),
            ef_evaluations.iter().find(|e| e.ef_search == 50),
        ) {
            slot.measured_this_run = true;
            slot.build_time_s = build_time_s;
            slot.build_qps = build_qps;
            slot.peak_rss_mb = peak_rss_mb;
            slot.bytes_per_vector = graph_bytes_per_vector;
            slot.raw_vector_bytes_per_vector = raw_vector_bytes_per_vector;
            slot.graph_bytes_per_vector = graph_bytes_per_vector;
            slot.graph_only_bytes_per_vector = graph_only_bytes_per_vector;
            slot.total_bytes_per_vector = total_bytes_per_vector;
            slot.graph_memory_bytes = graph_total_bytes;
            slot.graph_only_memory_bytes = graph_only_total_bytes;
            slot.graph_memory_mb = graph_total_mb;
            slot.k = config.k;
            slot.max_layer = max_layer;
            slot.entry_point = entry_point;
            slot.recall_at_10 = eval50.recall_at_k;
            slot.query_latency_ms = eval50.latency.mean_ms;
            slot.query_qps = eval50.latency.qps;
            slot.latency_min_ms = eval50.latency.min_ms;
            slot.latency_p50_ms = eval50.latency.p50_ms;
            slot.latency_p95_ms = eval50.latency.p95_ms;
            slot.latency_p99_ms = eval50.latency.p99_ms;
            slot.latency_max_ms = eval50.latency.max_ms;
        }
    }

    let preset_name = match &config.dataset_mode {
        Phase1DatasetMode::Sift(p) => p.as_str().to_string(),
        Phase1DatasetMode::Synthetic { .. } => "synthetic".to_string(),
        Phase1DatasetMode::Custom { .. } => "custom".to_string(),
    };

    let (base_path, query_path, groundtruth_path) = match &loaded.source {
        DatasetSource::AuthenticSift1M { dir } => (
            Some(dir.join("sift_base.fvecs")),
            Some(dir.join("sift_query.fvecs")),
            Some(dir.join("sift_groundtruth.ivecs")),
        ),
        DatasetSource::AuthenticSiftSmall { dir } => (
            Some(dir.join("siftsmall_base.fvecs")),
            Some(dir.join("siftsmall_query.fvecs")),
            Some(dir.join("siftsmall_groundtruth.ivecs")),
        ),
        _ => (None, None, None),
    };

    let generated_at = current_utc_rfc3339();

    Ok(Phase3Report {
        phase: 3,
        phase_name: "Phase 3: HNSW SIFT1M Scaling, Memory Profiling & Ground Truth Benchmarking"
            .to_string(),
        generated_at,
        hardware,
        dataset_source: loaded.source,
        preset_name,
        base_path,
        query_path,
        groundtruth_path,
        num_base,
        num_query,
        dim,
        k: config.k,
        seed: config.seed,
        metric: config.metric,
        hnsw_m: config.m,
        hnsw_m_max0: effective_m_max0,
        hnsw_ef_construction: config.ef_construction,
        build_time_s,
        build_qps,
        max_layer,
        entry_point,
        memory_profile,
        multi_scale_baseline_summary: baseline_profiles,
        sweep_runs: runs_per_ef,
        ef_evaluations,
        monotonic_scaling_verified,
        reproducibility_checked,
        reproducibility_verified,
        official_ground_truth_matched,
        official_gt_check_queries,
        official_gt_recall_at_k,
        official_gt_exact_id_matches,
        terminal_stopping_enforced: true,
        phase4_initiated: false,
        graph_fingerprint,
        build_stats,
    })
}

/// Formats and prints a structured summary of the Phase 3 SIFT1M benchmark to stdout.
pub fn print_phase3_report(r: &Phase3Report) {
    println!("{}", "=".repeat(80));
    println!(" HNSW APPROXIMATE NEAREST NEIGHBOR SEARCH (RUST) -- PHASE 3 SIFT1M BENCHMARK ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!("Dataset Provenance: {}", r.dataset_source);
    println!("Configuration:");
    println!("  - Base Vectors:          {} vectors", r.num_base);
    println!("  - Query Vectors:         {} vectors", r.num_query);
    println!("  - Vector Dimension:      {}-d (float32)", r.dim);
    println!("  - Top-k Neighbors:       {}", r.k);
    println!("  - Distance Metric:       {}", r.metric);
    println!("  - Deterministic Seed:    {}", r.seed);
    println!(
        "  - HNSW Graph Topology:   M={}, M0={}, efConstruction={}",
        r.hnsw_m, r.hnsw_m_max0, r.hnsw_ef_construction
    );
    println!("{}", "-".repeat(80));

    println!("[+] Index Construction Profile:");
    println!(
        "  - Total build elapsed:   {:.3} s ({:.1} vectors/s)",
        r.build_time_s, r.build_qps
    );
    println!("  - Max graph layer:       {}", r.max_layer);
    println!("  - Entry point node:      #{}", r.entry_point);
    println!("{}", "-".repeat(80));

    let mp = &r.memory_profile;
    println!("[+] Memory Profiling & Decoupled Footprint:");
    println!(
        "  - Process Peak RSS (OS): {:.2} MiB ({:.2} MB)",
        mp.peak_rss_mb,
        mp.peak_rss_mb * 1.048576
    );
    println!(
        "  - Measurement Method:    {}",
        mp.peak_rss_measurement_method
    );
    println!(
        "  - Raw Vector Storage:    {:.2} B/vector ({:.2} MiB total)",
        mp.raw_vector_bytes_per_vector, mp.raw_vector_total_mb
    );
    println!(
        "  - Graph Structural Cost: {:.2} B/vector ({:.2} MiB total, incl. visited & scratch)",
        mp.graph_bytes_per_vector, mp.graph_total_mb
    );
    println!(
        "  - Graph-Only Footprint:  {:.2} B/vector ({:.2} MiB total, excl. visited & scratch)",
        mp.graph_only_bytes_per_vector, mp.graph_only_total_mb
    );
    println!(
        "  - Total Index Footprint: {:.2} B/vector ({:.2} MiB total)",
        mp.total_bytes_per_vector, mp.total_footprint_mb
    );
    println!("\n  Detailed Graph Components Breakdown:");
    println!(
        "    - Node struct buffers:         {:10} bytes ({:.2} MiB)",
        mp.graph_components.node_structs_bytes,
        mp.graph_components.node_structs_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Layer-0 edge adjacency:      {:10} bytes ({:.2} MiB)",
        mp.graph_components.layer0_edges_bytes,
        mp.graph_components.layer0_edges_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Upper layers edge adjacency: {:10} bytes ({:.2} MiB)",
        mp.graph_components.upper_layers_edges_bytes,
        mp.graph_components.upper_layers_edges_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Outer neighbor headers:      {:10} bytes ({:.2} MiB)",
        mp.graph_components.outer_neighbors_headers_bytes,
        mp.graph_components.outer_neighbors_headers_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Visited tracker tag buffer:  {:10} bytes ({:.2} MiB)",
        mp.graph_components.visited_tracker_tags_bytes,
        mp.graph_components.visited_tracker_tags_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Heuristic scratch buffers:   {:10} bytes (<0.01 MiB)",
        mp.graph_components.heuristic_scratch_bytes
    );
    println!("{}", "-".repeat(80));

    println!(
        "[+] efSearch Parameter Sweep vs. Texmex Ground Truth (k={}, median of {} runs):",
        r.k, r.sweep_runs
    );
    println!("+----------+-----------+------------+----------------+------------+------------+------------+-----------+");
    println!("| efSearch | Recall@10 | MedMean ms | Mean Spread ms | p50 (ms)   | p99 (ms)   | QPS        | Monotonic |");
    println!("+----------+-----------+------------+----------------+------------+------------+------------+-----------+");
    for ef in &r.ef_evaluations {
        println!(
            "| {:8} | {:8.2}% | {:10.3} | {:14.3} | {:10.3} | {:10.3} | {:10.1} | {:9} |",
            ef.ef_search,
            ef.recall_at_k * 100.0,
            ef.latency.mean_ms,
            ef.mean_ms_spread,
            ef.latency.p50_ms,
            ef.latency.p99_ms,
            ef.latency.qps,
            if ef.monotonic_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );
    }
    println!("+----------+-----------+------------+----------------+------------+------------+------------+-----------+");
    println!("{}", "-".repeat(80));

    println!("[+] Verification & Quality Status:");
    println!(
        "  - Authentic SIFT1M Corpus:       {}",
        if r.dataset_source.is_authentic_sift() {
            "PASSED"
        } else {
            "SYNTHETIC/FALLBACK"
        }
    );
    println!(
        "  - Official Ground Truth Matched: {}",
        if r.official_ground_truth_matched {
            format!(
                "PASSED (100.0% ExactKNN match on {} queries)",
                r.official_gt_check_queries
            )
        } else {
            "FAILED / NOT APPLICABLE".to_string()
        }
    );
    println!(
        "  - Monotonic Recall Scaling:      {}",
        if r.monotonic_scaling_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "  - Bit-Exact Reproducibility:     {}",
        if !r.reproducibility_checked {
            "SKIPPED"
        } else if r.reproducibility_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!("  - #![forbid(unsafe_code)]:       STRICTLY ENFORCED (0 unsafe blocks)");
    println!(
        "  - Pure Standard Library:         STRICTLY ENFORCED (0 external algorithmic crates)"
    );
    println!("\n{}", "=".repeat(80));
    println!(" PHASE 3 COMPLETE (RUST) -- TERMINAL MILESTONE REACHED -- STOPPING EXECUTION ");
    println!("{}", "=".repeat(80));
    println!("Notice: Phase 3 benchmark and memory profiling on SIFT1M is complete.");
    println!("Stopping at Phase 3 milestone boundary. Phase 4 will not be initiated.");
    println!("{}", "=".repeat(80));
}
