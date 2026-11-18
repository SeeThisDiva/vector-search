//! Canonical Phase 7 JSON Report Serialization and Console Summary Formatting.

use std::path::PathBuf;

use vector_search::benchmark::ReportProvenance;
use vector_search::dataset::DatasetSource;
use vector_search::hardware::HardwareInfo;
use vector_search::hnsw::HnswConfig;
use vector_search::json_util::{
    escape_json, opt_f64_json_prec, opt_path_json, push_f64_array_prec,
};
use vector_search::recall::{median_min_max, RepeatedSweepSummary};

use super::cli::CompareMode;

#[derive(Debug, Clone)]
pub struct Mode1Summary {
    pub num_queries: usize,
    pub total_s: f64,
    pub qps: f64,
    pub mean_ms: f64,
    pub mean_ms_min: f64,
    pub mean_ms_max: f64,
    pub mean_ms_spread: f64,
    pub qps_min: f64,
    pub qps_max: f64,
    pub qps_spread: f64,
    pub run_total_s: Vec<f64>,
    pub run_mean_ms: Vec<f64>,
    pub run_qps: Vec<f64>,
}

pub fn summarize_mode1_runs(num_queries: usize, totals_s: &[f64]) -> Mode1Summary {
    let run_mean_ms: Vec<f64> = totals_s
        .iter()
        .map(|&t| (t * 1000.0) / (num_queries.max(1) as f64))
        .collect();
    let run_qps: Vec<f64> = totals_s
        .iter()
        .map(|&t| {
            if t > 0.0 {
                (num_queries as f64) / t
            } else {
                0.0
            }
        })
        .collect();

    let (med_total_s, _, _) = median_min_max(totals_s);
    let (med_mean_ms, mean_ms_min, mean_ms_max) = median_min_max(&run_mean_ms);
    let (_, qps_min, qps_max) = median_min_max(&run_qps);
    let med_qps = if med_total_s > 0.0 {
        (num_queries as f64) / med_total_s
    } else {
        0.0
    };

    Mode1Summary {
        num_queries,
        total_s: med_total_s,
        qps: med_qps,
        mean_ms: med_mean_ms,
        mean_ms_min,
        mean_ms_max,
        mean_ms_spread: (mean_ms_max - mean_ms_min).max(0.0),
        qps_min,
        qps_max,
        qps_spread: (qps_max - qps_min).max(0.0),
        run_total_s: totals_s.to_vec(),
        run_mean_ms,
        run_qps,
    }
}

#[derive(Debug, Clone)]
pub struct Mode2Summary {
    pub sweep: RepeatedSweepSummary,
    pub run_total_s: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SweepPoint {
    pub ef_search: usize,
    pub recall_at_k: f64,
    pub recall_at_k_mode2: Option<f64>,
    pub mode1: Mode1Summary,
    pub mode2: Option<Mode2Summary>,
    pub ffi_delta_us: Option<f64>,
    pub instant_overhead_us: Option<f64>,
    pub dist_evals_mean: Option<f64>,
}

pub struct Phase7Report {
    pub compare_mode: CompareMode,
    pub config_id: String,
    pub generated_at: String,
    pub provenance: ReportProvenance,
    pub affinity_mask: String,
    pub affinity_pinned: bool,
    pub isolated_process: bool,
    pub hardware: HardwareInfo,
    pub preset_name: String,
    pub dataset_source: DatasetSource,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub seed: u64,
    pub ground_truth_source: String,
    pub ground_truth_path: Option<PathBuf>,
    pub hnsw_config: HnswConfig,
    pub threads: usize,
    pub build_reps: usize,
    pub query_reps: usize,
    pub build_times_s: Vec<f64>,
    pub build_time_s: f64,
    pub build_time_min_s: f64,
    pub build_time_max_s: f64,
    pub build_time_spread_s: f64,
    pub build_qps: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub graph_fingerprint: u64,
    pub peak_rss_mb: Option<f64>,
    pub graph_only_bytes_per_vector: f64,
    pub graph_only_memory_bytes: usize,
    pub graph_only_memory_mb: f64,
    pub bytes_per_vector: f64,
    pub layer0_bytes: usize,
    pub upper_layers_bytes: usize,
    pub node_structs_bytes: usize,
    pub raw_vector_bytes_per_vector: f64,
    pub total_with_vectors_bytes_per_vector: f64,
    pub monotonic_recall: bool,
    pub sweep: Vec<SweepPoint>,
}

impl Phase7Report {
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(16_384);
        out.push_str("{\n");
        out.push_str("  \"schema_version\": 1,\n");
        out.push_str("  \"phase\": 7,\n");
        out.push_str("  \"phase_name\": \"Phase 7 Rust HnswIndex Benchmark\",\n");
        out.push_str("  \"engine\": \"rust\",\n");
        out.push_str(&format!(
            "  \"compare_mode\": \"{}\",\n",
            self.compare_mode.as_str()
        ));
        out.push_str(&format!(
            "  \"config_id\": \"{}\",\n",
            escape_json(&self.config_id)
        ));
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&self.generated_at)
        ));
        out.push_str(&format!(
            "  \"git_commit\": \"{}\",\n",
            escape_json(&self.provenance.git_commit)
        ));
        out.push_str(&format!(
            "  \"git_dirty\": {},\n",
            self.provenance.git_dirty
        ));
        out.push_str(&format!(
            "  \"rustc_version\": \"{}\",\n",
            escape_json(&self.provenance.rustc_version)
        ));
        out.push_str(&format!(
            "  \"isolated_process\": {},\n",
            self.isolated_process
        ));

        // environment
        out.push_str("  \"environment\": {\n");
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\",\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {},\n",
            opt_f64_json_prec(self.hardware.total_ram_gb, 2)
        ));
        out.push_str(&format!(
            "    \"power_mode\": \"{}\",\n",
            escape_json(&self.provenance.power_mode)
        ));
        out.push_str(&format!(
            "    \"processor_affinity_mask\": \"{}\",\n",
            escape_json(&self.affinity_mask)
        ));
        out.push_str(&format!(
            "    \"affinity_pinned\": {},\n",
            self.affinity_pinned
        ));
        out.push_str(&format!("    \"threads\": {}\n", self.threads));
        out.push_str("  },\n");

        // hardware
        out.push_str("  \"hardware\": {\n");
        out.push_str(&format!(
            "    \"summary\": \"{}\",\n",
            escape_json(&self.hardware.summary_line())
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {},\n",
            opt_f64_json_prec(self.hardware.total_ram_gb, 2)
        ));
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\"\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str("  },\n");

        // dataset
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
        out.push_str(&format!("    \"scale_vectors\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str("    \"metric\": \"l2\"\n");
        out.push_str("  },\n");

        out.push_str(&format!(
            "  \"ground_truth_source\": \"{}\",\n",
            escape_json(&self.ground_truth_source)
        ));
        out.push_str(&format!(
            "  \"ground_truth_path\": {},\n",
            opt_path_json(self.ground_truth_path.as_deref())
        ));

        // hnsw_config
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_config.m));
        out.push_str(&format!(
            "    \"m_max\": {},\n",
            self.hnsw_config.effective_m_max()
        ));
        out.push_str(&format!(
            "    \"m_max0\": {},\n",
            self.hnsw_config.effective_m_max0()
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

        out.push_str(&format!("  \"threads\": {},\n", self.threads));
        out.push_str("  \"warmup_performed\": true,\n");
        out.push_str(&format!("  \"build_reps\": {},\n", self.build_reps));
        out.push_str(&format!("  \"query_reps\": {},\n", self.query_reps));
        out.push_str(&format!("  \"scale_vectors\": {},\n", self.num_base));

        // index_build
        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!(
            "    \"build_time_min_s\": {:.6},\n",
            self.build_time_min_s
        ));
        out.push_str(&format!(
            "    \"build_time_max_s\": {:.6},\n",
            self.build_time_max_s
        ));
        out.push_str(&format!(
            "    \"build_time_spread_s\": {:.6},\n",
            self.build_time_spread_s
        ));
        out.push_str("    \"build_times_s\": ");
        push_f64_array_prec(&mut out, &self.build_times_s, 6);
        out.push_str(",\n");
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

        // memory_profile
        let peak_scope = if self.isolated_process {
            "isolated_child_process_high_water_mark"
        } else {
            "process_lifetime_high_water_mark"
        };
        out.push_str("  \"memory_profile\": {\n");
        out.push_str(&format!(
            "    \"peak_rss_mb\": {},\n",
            opt_f64_json_prec(self.peak_rss_mb, 2)
        ));
        out.push_str(&format!("    \"peak_rss_scope\": \"{peak_scope}\",\n"));
        out.push_str(&format!(
            "    \"graph_only_bytes_per_vector\": {:.2},\n",
            self.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_bytes\": {},\n",
            self.graph_only_memory_bytes
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_mb\": {:.4},\n",
            self.graph_only_memory_mb
        ));
        out.push_str(&format!(
            "    \"bytes_per_vector\": {:.2},\n",
            self.bytes_per_vector
        ));
        out.push_str(&format!("    \"layer0_bytes\": {},\n", self.layer0_bytes));
        out.push_str(&format!(
            "    \"upper_layers_bytes\": {},\n",
            self.upper_layers_bytes
        ));
        out.push_str(&format!(
            "    \"node_structs_bytes\": {},\n",
            self.node_structs_bytes
        ));
        out.push_str(&format!(
            "    \"raw_vector_bytes_per_vector\": {:.2},\n",
            self.raw_vector_bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"total_with_vectors_bytes_per_vector\": {:.2}\n",
            self.total_with_vectors_bytes_per_vector
        ));
        out.push_str("  },\n");

        out.push_str(&format!(
            "  \"graph_only_bytes_per_vector\": {:.2},\n",
            self.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "  \"monotonic_recall\": {},\n",
            self.monotonic_recall
        ));

        // ef_search_sweep
        out.push_str("  \"ef_search_sweep\": [\n");
        for (idx, pt) in self.sweep.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", pt.ef_search));
            out.push_str(&format!("      \"recall_at_k\": {:.6},\n", pt.recall_at_k));
            out.push_str(&format!(
                "      \"recall_at_k_mode2\": {},\n",
                opt_f64_json_prec(pt.recall_at_k_mode2, 6)
            ));

            // mode1_batch
            out.push_str("      \"mode1_batch\": {\n");
            out.push_str(&format!(
                "        \"num_queries\": {},\n",
                pt.mode1.num_queries
            ));
            out.push_str(&format!("        \"total_s\": {:.6},\n", pt.mode1.total_s));
            out.push_str(&format!("        \"qps\": {:.2},\n", pt.mode1.qps));
            out.push_str(&format!("        \"mean_ms\": {:.4},\n", pt.mode1.mean_ms));
            out.push_str(&format!(
                "        \"mean_ms_min\": {:.4},\n",
                pt.mode1.mean_ms_min
            ));
            out.push_str(&format!(
                "        \"mean_ms_max\": {:.4},\n",
                pt.mode1.mean_ms_max
            ));
            out.push_str(&format!(
                "        \"mean_ms_spread\": {:.4},\n",
                pt.mode1.mean_ms_spread
            ));
            out.push_str(&format!("        \"qps_min\": {:.2},\n", pt.mode1.qps_min));
            out.push_str(&format!("        \"qps_max\": {:.2},\n", pt.mode1.qps_max));
            out.push_str(&format!(
                "        \"qps_spread\": {:.2},\n",
                pt.mode1.qps_spread
            ));
            out.push_str("        \"run_total_s\": ");
            push_f64_array_prec(&mut out, &pt.mode1.run_total_s, 6);
            out.push_str(",\n");
            out.push_str("        \"run_mean_ms\": ");
            push_f64_array_prec(&mut out, &pt.mode1.run_mean_ms, 4);
            out.push_str(",\n");
            out.push_str("        \"run_qps\": ");
            push_f64_array_prec(&mut out, &pt.mode1.run_qps, 2);
            out.push_str("\n      },\n");

            // mode2_per_query
            match &pt.mode2 {
                Some(m2) => {
                    let med = &m2.sweep.median_latency;
                    out.push_str("      \"mode2_per_query\": {\n");
                    out.push_str(&format!("        \"num_queries\": {},\n", med.num_queries));
                    out.push_str(&format!("        \"total_s\": {:.6},\n", med.total_s));
                    out.push_str(&format!("        \"qps\": {:.2},\n", med.qps));
                    out.push_str(&format!("        \"min_ms\": {:.4},\n", med.min_ms));
                    out.push_str(&format!("        \"mean_ms\": {:.4},\n", med.mean_ms));
                    out.push_str(&format!("        \"p50_ms\": {:.4},\n", med.p50_ms));
                    out.push_str(&format!("        \"p95_ms\": {:.4},\n", med.p95_ms));
                    out.push_str(&format!("        \"p99_ms\": {:.4},\n", med.p99_ms));
                    out.push_str(&format!("        \"max_ms\": {:.4},\n", med.max_ms));
                    out.push_str(&format!(
                        "        \"mean_ms_min\": {:.4},\n",
                        m2.sweep.mean_ms_min
                    ));
                    out.push_str(&format!(
                        "        \"mean_ms_max\": {:.4},\n",
                        m2.sweep.mean_ms_max
                    ));
                    out.push_str(&format!(
                        "        \"mean_ms_spread\": {:.4},\n",
                        m2.sweep.mean_ms_spread
                    ));
                    out.push_str(&format!(
                        "        \"p99_ms_min\": {:.4},\n",
                        m2.sweep.p99_ms_min
                    ));
                    out.push_str(&format!(
                        "        \"p99_ms_max\": {:.4},\n",
                        m2.sweep.p99_ms_max
                    ));
                    out.push_str(&format!(
                        "        \"p99_ms_spread\": {:.4},\n",
                        m2.sweep.p99_ms_spread
                    ));
                    out.push_str("        \"run_total_s\": ");
                    push_f64_array_prec(&mut out, &m2.run_total_s, 6);
                    out.push_str(",\n");
                    out.push_str("        \"run_mean_ms\": ");
                    push_f64_array_prec(&mut out, &m2.sweep.run_mean_latencies_ms, 4);
                    out.push_str(",\n");
                    out.push_str("        \"run_p99_ms\": ");
                    push_f64_array_prec(&mut out, &m2.sweep.run_p99_latencies_ms, 4);
                    out.push_str("\n      },\n");
                }
                None => {
                    out.push_str("      \"mode2_per_query\": null,\n");
                }
            }

            out.push_str(&format!(
                "      \"ffi_delta_us\": {},\n",
                opt_f64_json_prec(pt.ffi_delta_us, 4)
            ));
            out.push_str(&format!(
                "      \"instant_overhead_us\": {},\n",
                opt_f64_json_prec(pt.instant_overhead_us, 4)
            ));
            out.push_str(&format!(
                "      \"dist_evals_mean\": {},\n",
                opt_f64_json_prec(pt.dist_evals_mean, 4)
            ));
            out.push_str(&format!("      \"mode1_qps\": {:.2},\n", pt.mode1.qps));
            out.push_str(&format!(
                "      \"mode1_mean_ms\": {:.4},\n",
                pt.mode1.mean_ms
            ));
            out.push_str(&format!("      \"qps\": {:.2},\n", pt.mode1.qps));

            let (mean_alias, p50_alias, p95_alias, p99_alias) = match &pt.mode2 {
                Some(m2) => (
                    m2.sweep.median_latency.mean_ms,
                    Some(m2.sweep.median_latency.p50_ms),
                    Some(m2.sweep.median_latency.p95_ms),
                    Some(m2.sweep.median_latency.p99_ms),
                ),
                None => (pt.mode1.mean_ms, None, None, None),
            };
            out.push_str(&format!("      \"mean_ms\": {:.4},\n", mean_alias));
            out.push_str(&format!(
                "      \"p50_ms\": {},\n",
                opt_f64_json_prec(p50_alias, 4)
            ));
            out.push_str(&format!(
                "      \"p95_ms\": {},\n",
                opt_f64_json_prec(p95_alias, 4)
            ));
            out.push_str(&format!(
                "      \"p99_ms\": {}\n",
                opt_f64_json_prec(p99_alias, 4)
            ));

            if idx + 1 < self.sweep.len() {
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

pub fn print_console_summary(report: &Phase7Report) {
    println!(
        "[Rust Phase 7] {} (M={}, efC={}, keep_pruned_shrink={:?}, mode={}, T={}, affinity={})",
        report.config_id,
        report.hnsw_config.m,
        report.hnsw_config.ef_construction,
        report.hnsw_config.keep_pruned_shrink,
        report.compare_mode.as_str(),
        report.threads,
        report.affinity_mask
    );
    println!(
        "  Hardware: {} | Power: {}",
        report.hardware.summary_line(),
        report.provenance.power_mode
    );
    println!(
        "  Dataset: {} N={} Q={} dim={} k={} | GT: {}",
        report.dataset_source.kind_tag(),
        report.num_base,
        report.num_query,
        report.dim,
        report.k,
        report.ground_truth_source
    );
    let rss_display = report
        .peak_rss_mb
        .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2} MiB"));
    println!(
        "  Build: median={:.3}s ({:.1} vec/s, reps={}) | fp=0x{:016x} | Graph={:.2} B/vec | PeakRSS={}",
        report.build_time_s,
        report.build_qps,
        report.build_reps,
        report.graph_fingerprint,
        report.graph_only_bytes_per_vector,
        rss_display
    );
    println!("  efSearch | Recall@k | Mode1 QPS | Mode1 ms | Mode2 p99 | Instant delta (us)");
    for pt in &report.sweep {
        let p99_str = pt.mode2.as_ref().map_or_else(
            || "      n/a".to_string(),
            |m2| format!("{:9.4}", m2.sweep.median_latency.p99_ms),
        );
        let delta_str = pt
            .instant_overhead_us
            .map_or_else(|| "               n/a".to_string(), |d| format!("{d:18.3}"));
        println!(
            "  {:8} | {:8.6} | {:9.2} | {:8.4} | {} | {}",
            pt.ef_search, pt.recall_at_k, pt.mode1.qps, pt.mode1.mean_ms, p99_str, delta_str
        );
    }
}
