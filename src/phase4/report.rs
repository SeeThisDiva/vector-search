//! Structured JSON Reporting and Resumable Run Inspection for Phase 4.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use crate::benchmark::{append_hardware_json, ReportProvenance};
use crate::dataset::DatasetSource;
use crate::distance::Metric;
use crate::hardware::{current_utc_rfc3339, HardwareInfo};
use crate::json_util::{
    escape_json, extract_json_array_slice, extract_json_f64_field, extract_json_string_field,
    extract_json_u64_field, opt_path_json, split_top_level_json_objects,
};
use crate::recall::RepeatedSweepSummary;

use super::grid::Phase4IndexConfig;

/// Checks whether an existing per-index report file matches the current git commit, scale, dataset, and config.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn can_resume_existing_report(
    report_path: &Path,
    expected_commit: &str,
    source_kind: &str,
    scale_vectors: usize,
    num_query: usize,
    k: usize,
    sweep_runs: usize,
    ef_search_list: &[usize],
    current_git_dirty: bool,
    idx_cfg: &Phase4IndexConfig,
) -> bool {
    if expected_commit == "unknown" {
        return false;
    }
    if !report_path.is_file() {
        return false;
    }
    let content = match fs::read_to_string(report_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    if !current_git_dirty && content.contains("\"git_dirty\": true") {
        return false;
    }
    content.contains("\"measured_this_run\": true")
        && content.contains(&format!("\"git_commit\": \"{expected_commit}\""))
        && content.contains(&format!("\"source_kind\": \"{source_kind}\""))
        && content.contains(&format!("\"scale_vectors\": {scale_vectors}"))
        && content.contains(&format!("\"num_query\": {num_query}"))
        && content.contains(&format!("\"k\": {k}"))
        && content.contains(&format!("\"sweep_runs\": {sweep_runs}"))
        && content.contains(&format!("\"m\": {}", idx_cfg.m))
        && content.contains(&format!("\"ef_construction\": {}", idx_cfg.ef_construction))
        && content.contains(&format!("\"seed\": {}", idx_cfg.seed))
        && ef_search_list
            .iter()
            .all(|ef| content.contains(&format!("\"ef_search\": {ef},")))
        && content.matches("\"ef_search\":").count() == ef_search_list.len()
}

/// Single efSearch evaluation step within an index report.
#[derive(Debug, Clone)]
pub struct Phase4EfEvaluation {
    pub ef_search: usize,
    pub recall_at_k: f64,
    pub summary: RepeatedSweepSummary,
}

/// Memory profile recorded for an index.
#[derive(Debug, Clone)]
pub struct Phase4MemoryProfile {
    pub peak_rss_mb: f64,
    pub graph_only_bytes_per_vector: f64,
    pub bytes_per_vector: f64,
    pub graph_only_memory_bytes: usize,
    pub graph_only_memory_mb: f64,
    pub layer0_bytes: usize,
    pub upper_layers_bytes: usize,
}

/// Complete structured per-index report written by Phase 4.
#[derive(Debug, Clone)]
pub struct Phase4IndexReport {
    pub phase: usize,
    pub phase_name: String,
    pub generated_at: String,
    pub provenance: ReportProvenance,
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub preset_name: String,
    pub scale_vectors: usize,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub ground_truth_source: String,
    pub ground_truth_cache_path: Option<PathBuf>,
    pub ground_truth_cache_hit: bool,
    pub hnsw_config: Phase4IndexConfig,
    pub build_time_s: f64,
    pub build_qps: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub graph_fingerprint: u64,
    pub memory_profile: Phase4MemoryProfile,
    pub warmup_performed: bool,
    pub measured_this_run: bool,
    pub sweep_runs: usize,
    pub ef_search_sweep: Vec<Phase4EfEvaluation>,
    pub monotonic_recall: bool,
}

impl Phase4IndexReport {
    /// Serializes this report to a deterministic, dependency-free JSON string.
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

        // Provenance & Environment
        self.provenance.append_json_fields(&mut out, &self.hardware);

        out.push_str(&format!("  \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str(&format!(
            "  \"measured_this_run\": {},\n",
            self.measured_this_run
        ));
        out.push_str(&format!(
            "  \"warmup_performed\": {},\n",
            self.warmup_performed
        ));
        out.push_str(&format!("  \"sweep_runs\": {},\n", self.sweep_runs));

        // Hardware block
        append_hardware_json(&mut out, &self.hardware);

        // Dataset block
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
        out.push_str(&format!("    \"scale_vectors\": {},\n", self.scale_vectors));
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

        // Ground truth metadata
        out.push_str(&format!(
            "  \"ground_truth_source\": \"{}\",\n",
            escape_json(&self.ground_truth_source)
        ));
        out.push_str(&format!(
            "  \"ground_truth_cache_path\": {},\n",
            opt_path_json(self.ground_truth_cache_path.as_deref())
        ));
        out.push_str(&format!(
            "  \"ground_truth_cache_hit\": {},\n",
            self.ground_truth_cache_hit
        ));

        // HNSW Config
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_config.m));
        out.push_str(&format!(
            "    \"m_max\": {},\n",
            self.hnsw_config.m_max.unwrap_or(self.hnsw_config.m)
        ));
        out.push_str(&format!(
            "    \"m_max0\": {},\n",
            self.hnsw_config.m_max0.unwrap_or(2 * self.hnsw_config.m)
        ));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_config.ef_construction
        ));
        out.push_str(&format!(
            "    \"keep_pruned\": {},\n",
            self.hnsw_config.keep_pruned
        ));
        match self.hnsw_config.keep_pruned_shrink {
            Some(kps) => out.push_str(&format!("    \"keep_pruned_shrink\": {kps},\n")),
            None => out.push_str("    \"keep_pruned_shrink\": null,\n"),
        }
        out.push_str(&format!("    \"seed\": {}\n", self.hnsw_config.seed));
        out.push_str("  },\n");

        // Index Build
        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!("    \"build_qps\": {:.2},\n", self.build_qps));
        out.push_str(&format!("    \"max_layer\": {},\n", self.max_layer));
        out.push_str(&format!("    \"entry_point\": {},\n", self.entry_point));
        out.push_str(&format!(
            "    \"graph_fingerprint\": {},\n",
            self.graph_fingerprint
        ));
        out.push_str(&format!(
            "    \"graph_fingerprint_hex\": \"0x{:016x}\"\n",
            self.graph_fingerprint
        ));
        out.push_str("  },\n");

        // Memory Profile
        out.push_str("  \"memory_profile\": {\n");
        out.push_str(&format!(
            "    \"peak_rss_mb\": {:.2},\n",
            self.memory_profile.peak_rss_mb
        ));
        out.push_str("    \"peak_rss_scope\": \"process_lifetime_high_water_mark\",\n");
        out.push_str(&format!(
            "    \"graph_only_bytes_per_vector\": {:.2},\n",
            self.memory_profile.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"bytes_per_vector\": {:.2},\n",
            self.memory_profile.bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_bytes\": {},\n",
            self.memory_profile.graph_only_memory_bytes
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_mb\": {:.4},\n",
            self.memory_profile.graph_only_memory_mb
        ));
        out.push_str(&format!(
            "    \"layer0_bytes\": {},\n",
            self.memory_profile.layer0_bytes
        ));
        out.push_str(&format!(
            "    \"upper_layers_bytes\": {}\n",
            self.memory_profile.upper_layers_bytes
        ));
        out.push_str("  },\n");

        out.push_str(&format!(
            "  \"graph_only_bytes_per_vector\": {:.2},\n",
            self.memory_profile.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "  \"monotonic_recall\": {},\n",
            self.monotonic_recall
        ));

        // ef_search_sweep
        out.push_str("  \"ef_search_sweep\": [\n");
        for (i, eval) in self.ef_search_sweep.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", eval.ef_search));
            out.push_str(&format!(
                "      \"recall_at_k\": {:.6},\n",
                eval.recall_at_k
            ));
            out.push_str(&format!(
                "      \"qps\": {:.2},\n",
                eval.summary.median_latency.qps
            ));
            out.push_str(&format!(
                "      \"mean_ms\": {:.4},\n",
                eval.summary.median_latency.mean_ms
            ));
            out.push_str(&format!(
                "      \"p50_ms\": {:.4},\n",
                eval.summary.median_latency.p50_ms
            ));
            out.push_str(&format!(
                "      \"p95_ms\": {:.4},\n",
                eval.summary.median_latency.p95_ms
            ));
            out.push_str(&format!(
                "      \"p99_ms\": {:.4},\n",
                eval.summary.median_latency.p99_ms
            ));

            out.push_str("      \"median_latency\": {\n");
            out.push_str(&format!(
                "        \"num_queries\": {},\n",
                eval.summary.median_latency.num_queries
            ));
            out.push_str(&format!(
                "        \"total_s\": {:.6},\n",
                eval.summary.median_latency.total_s
            ));
            out.push_str(&format!(
                "        \"qps\": {:.2},\n",
                eval.summary.median_latency.qps
            ));
            out.push_str(&format!(
                "        \"min_ms\": {:.4},\n",
                eval.summary.median_latency.min_ms
            ));
            out.push_str(&format!(
                "        \"mean_ms\": {:.4},\n",
                eval.summary.median_latency.mean_ms
            ));
            out.push_str(&format!(
                "        \"p50_ms\": {:.4},\n",
                eval.summary.median_latency.p50_ms
            ));
            out.push_str(&format!(
                "        \"p95_ms\": {:.4},\n",
                eval.summary.median_latency.p95_ms
            ));
            out.push_str(&format!(
                "        \"p99_ms\": {:.4},\n",
                eval.summary.median_latency.p99_ms
            ));
            out.push_str(&format!(
                "        \"max_ms\": {:.4}\n",
                eval.summary.median_latency.max_ms
            ));
            out.push_str("      },\n");

            out.push_str(&format!(
                "      \"mean_ms_min\": {:.4},\n",
                eval.summary.mean_ms_min
            ));
            out.push_str(&format!(
                "      \"mean_ms_max\": {:.4},\n",
                eval.summary.mean_ms_max
            ));
            out.push_str(&format!(
                "      \"mean_ms_spread\": {:.4},\n",
                eval.summary.mean_ms_spread
            ));
            out.push_str(&format!(
                "      \"p99_ms_min\": {:.4},\n",
                eval.summary.p99_ms_min
            ));
            out.push_str(&format!(
                "      \"p99_ms_max\": {:.4},\n",
                eval.summary.p99_ms_max
            ));
            out.push_str(&format!(
                "      \"p99_ms_spread\": {:.4},\n",
                eval.summary.p99_ms_spread
            ));

            out.push_str("      \"run_mean_latencies_ms\": [");
            for (rj, r) in eval.summary.run_mean_latencies_ms.iter().enumerate() {
                if rj > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{r:.4}"));
            }
            out.push_str("],\n");

            out.push_str("      \"run_p99_latencies_ms\": [");
            for (rj, r) in eval.summary.run_p99_latencies_ms.iter().enumerate() {
                if rj > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{r:.4}"));
            }
            out.push_str("]\n");

            if i + 1 < self.ef_search_sweep.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ]\n");

        out.push_str("}\n");
        out
    }
}

/// Summary entry for a single configuration within `summary.json`.
#[derive(Debug, Clone)]
pub struct Phase4IndexSummaryEntry {
    pub m: usize,
    pub ef_construction: usize,
    pub keep_pruned_shrink: Option<bool>,
    pub report_filename: String,
    pub status: String,
    pub graph_fingerprint: Option<u64>,
    pub graph_fingerprint_hex: Option<String>,
    pub build_time_s: Option<f64>,
    pub build_qps: Option<f64>,
    pub peak_rss_mb: Option<f64>,
    pub graph_only_bytes_per_vector: Option<f64>,
    pub best_recall: Option<f64>,
    pub best_qps: Option<f64>,
}

/// Structured summary of an entire Phase 4 parameter sweep.
#[derive(Debug, Clone)]
pub struct Phase4SweepSummary {
    pub total_configs: usize,
    pub newly_measured_count: usize,
    pub resumed_skipped_count: usize,
    pub git_commit: String,
    pub git_dirty: bool,
    pub rustc_version: String,
    pub provenance: ReportProvenance,
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub scale_vectors: usize,
    pub ground_truth_source: String,
    pub ground_truth_cache_hit: bool,
    pub configs: Vec<Phase4IndexSummaryEntry>,
}

impl Phase4SweepSummary {
    /// Serializes the sweep summary to deterministic JSON.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str("  \"summary_version\": 1,\n");
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&current_utc_rfc3339())
        ));

        // Provenance & Environment
        self.provenance.append_json_fields(&mut out, &self.hardware);

        append_hardware_json(&mut out, &self.hardware);

        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!("    \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {}\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str("  },\n");

        out.push_str(&format!("  \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str(&format!(
            "  \"ground_truth_source\": \"{}\",\n",
            escape_json(&self.ground_truth_source)
        ));
        out.push_str(&format!(
            "  \"ground_truth_cache_hit\": {},\n",
            self.ground_truth_cache_hit
        ));
        out.push_str(&format!("  \"total_configs\": {},\n", self.total_configs));
        out.push_str(&format!(
            "  \"newly_measured_count\": {},\n",
            self.newly_measured_count
        ));
        out.push_str(&format!(
            "  \"resumed_skipped_count\": {},\n",
            self.resumed_skipped_count
        ));
        out.push_str("  \"measured_this_run\": true,\n");

        out.push_str("  \"configs\": [\n");
        for (i, entry) in self.configs.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"m\": {},\n", entry.m));
            out.push_str(&format!(
                "      \"ef_construction\": {},\n",
                entry.ef_construction
            ));
            match entry.keep_pruned_shrink {
                Some(kps) => out.push_str(&format!("      \"keep_pruned_shrink\": {kps},\n")),
                None => out.push_str("      \"keep_pruned_shrink\": null,\n"),
            }
            out.push_str(&format!(
                "      \"report_filename\": \"{}\",\n",
                escape_json(&entry.report_filename)
            ));
            out.push_str(&format!(
                "      \"status\": \"{}\",\n",
                escape_json(&entry.status)
            ));
            match &entry.graph_fingerprint_hex {
                Some(hex) => out.push_str(&format!(
                    "      \"graph_fingerprint_hex\": \"{}\",\n",
                    escape_json(hex)
                )),
                None => out.push_str("      \"graph_fingerprint_hex\": null,\n"),
            }
            match entry.graph_fingerprint {
                Some(fp) => out.push_str(&format!("      \"graph_fingerprint\": {fp},\n")),
                None => out.push_str("      \"graph_fingerprint\": null,\n"),
            }
            match entry.build_time_s {
                Some(bt) => out.push_str(&format!("      \"build_time_s\": {bt:.6},\n")),
                None => out.push_str("      \"build_time_s\": null,\n"),
            }
            match entry.build_qps {
                Some(q) => out.push_str(&format!("      \"build_qps\": {q:.2},\n")),
                None => out.push_str("      \"build_qps\": null,\n"),
            }
            match entry.peak_rss_mb {
                Some(rss) => out.push_str(&format!("      \"peak_rss_mb\": {rss:.2},\n")),
                None => out.push_str("      \"peak_rss_mb\": null,\n"),
            }
            match entry.graph_only_bytes_per_vector {
                Some(gb) => out.push_str(&format!(
                    "      \"graph_only_bytes_per_vector\": {gb:.2},\n"
                )),
                None => out.push_str("      \"graph_only_bytes_per_vector\": null,\n"),
            }
            match entry.best_recall {
                Some(r) => out.push_str(&format!("      \"best_recall\": {r:.6},\n")),
                None => out.push_str("      \"best_recall\": null,\n"),
            }
            match entry.best_qps {
                Some(q) => out.push_str(&format!("      \"best_qps\": {q:.2}\n")),
                None => out.push_str("      \"best_qps\": null\n"),
            }

            if i + 1 < self.configs.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ]\n");

        out.push_str("}\n");
        out
    }
}

pub(crate) fn parse_resumed_summary_entry(
    report_path: &Path,
    idx_cfg: &Phase4IndexConfig,
    report_filename: &str,
) -> Phase4IndexSummaryEntry {
    let content = fs::read_to_string(report_path).unwrap_or_default();
    let graph_fingerprint_hex = extract_json_string_field(&content, "graph_fingerprint_hex");
    let graph_fingerprint = extract_json_u64_field(&content, "graph_fingerprint");
    let build_time_s = extract_json_f64_field(&content, "build_time_s");
    let build_qps = extract_json_f64_field(&content, "build_qps");
    let peak_rss_mb = extract_json_f64_field(&content, "peak_rss_mb");
    let graph_only_bytes_per_vector =
        extract_json_f64_field(&content, "graph_only_bytes_per_vector");

    let mut best_recall: Option<f64> = None;
    let mut best_qps: Option<f64> = None;

    if let Some(sweep_slice) = extract_json_array_slice(&content, "ef_search_sweep") {
        for obj in split_top_level_json_objects(sweep_slice) {
            if let Some(r) = extract_json_f64_field(obj, "recall_at_k") {
                best_recall = Some(best_recall.map_or(r, |b| b.max(r)));
            }
            if let Some(q) = extract_json_f64_field(obj, "qps") {
                best_qps = Some(best_qps.map_or(q, |b| b.max(q)));
            }
        }
    }

    Phase4IndexSummaryEntry {
        m: idx_cfg.m,
        ef_construction: idx_cfg.ef_construction,
        keep_pruned_shrink: idx_cfg.keep_pruned_shrink,
        report_filename: report_filename.to_string(),
        status: "resumed".to_string(),
        graph_fingerprint,
        graph_fingerprint_hex,
        build_time_s,
        build_qps,
        peak_rss_mb,
        graph_only_bytes_per_vector,
        best_recall,
        best_qps,
    }
}
