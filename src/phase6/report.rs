//! Phase 6 per-arm and matrix summary JSON serialization and `--resume` validation.

use std::fs;
use std::path::{Path, PathBuf};

use crate::benchmark::{append_hardware_json, ReportProvenance};
use crate::hardware::{current_utc_rfc3339, HardwareInfo};
use crate::json_util::{
    escape_json, extract_json_array_slice, extract_json_bool_field, extract_json_f64_field,
    extract_json_string_field, extract_json_u64_field, opt_f64_json_prec, opt_path_json,
    push_f64_array, split_top_level_json_objects,
};
use crate::quantized::{PqConfig, QuantizedMemoryProfile};
use crate::recall::RepeatedSweepSummary;

use super::config::{Phase6ArmKind, Phase6ArmSpec, Phase6MatrixConfig, RerankDepthSpec};
use super::dataset::PreparedDatasetState;

/// Single measured point inside an arm's `sweep[]` array.
#[derive(Debug, Clone)]
pub struct Phase6SweepPointResult {
    pub ef_search: usize,
    pub effective_ef_search: usize,
    pub rerank_depth: RerankDepthSpec,
    pub recall_at_k: f64,
    pub summary: RepeatedSweepSummary,
}

/// Summary entry recorded for one arm in `summary_n{scale}.json`.
#[derive(Debug, Clone)]
pub struct Phase6ArmSummaryEntry {
    pub arm_id: String,
    pub status: String,
    pub report_filename: String,
    pub quantizer_kind: String,
    pub storage_mode: String,
    pub build_mode: String,
    pub m: usize,
    pub ef_construction: usize,
    pub graph_fingerprint_hex: Option<String>,
    pub total_build_time_s: Option<f64>,
    pub steady_state_bytes_per_vector: Option<f64>,
    pub build_peak_rss_mb: Option<f64>,
    pub query_serving_peak_rss_mb: Option<f64>,
    pub best_recall: Option<f64>,
    pub best_qps: Option<f64>,
}

/// Summary of a complete `--matrix` or `--pilot` invocation.
#[derive(Debug, Clone)]
pub struct Phase6MatrixSummary {
    pub scale: usize,
    pub total_arms: usize,
    pub newly_measured_count: usize,
    pub resumed_skipped_count: usize,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub ground_truth_source: String,
    pub ground_truth_cache_hit: bool,
    pub arms: Vec<Phase6ArmSummaryEntry>,
}

pub(crate) struct ArmReportContext<'a> {
    pub spec: &'a Phase6ArmSpec,
    pub config: &'a Phase6MatrixConfig,
    pub prov: &'a ReportProvenance,
    pub hw: &'a HardwareInfo,
    pub dataset: &'a PreparedDatasetState,
    pub training_or_calibration_time_s: f64,
    pub encode_time_s: f64,
    pub graph_build_time_s: f64,
    pub total_build_time_s: f64,
    pub graph_fingerprint: u64,
    pub streaming_build: bool,
    pub pq_config: Option<PqConfig>,
    pub mem: QuantizedMemoryProfile,
    pub build_peak_rss_mb: Option<f64>,
    pub query_serving_peak_rss_mb: Option<f64>,
    pub query_rss_profiled_isolated: bool,
    pub sweep: &'a [Phase6SweepPointResult],
}

pub(crate) fn serialize_arm_report_json(ctx: &ArmReportContext<'_>) -> String {
    let mut out = String::with_capacity(16 * 1024);
    let denom = ctx.mem.num_vectors.max(1) as f64;
    let codes_bv = ctx.mem.codes_bytes as f64 / denom;
    let cal_bv = ctx.mem.calibration_or_codebook_bytes as f64 / denom;
    let raw_bv = ctx.mem.raw_vectors_retained_bytes as f64 / denom;
    let fp_hex = format!("0x{:016x}", ctx.graph_fingerprint);
    let build_rss_json = opt_f64_json_prec(ctx.build_peak_rss_mb, 2);
    let query_rss_json = opt_f64_json_prec(ctx.query_serving_peak_rss_mb, 2);
    let rss_scope = if ctx.query_rss_profiled_isolated {
        "isolated_query_subprocess_high_water_mark"
    } else {
        "process_lifetime_high_water_mark"
    };

    out.push_str("{\n");
    out.push_str("  \"phase\": 6,\n");
    out.push_str("  \"phase_name\": \"Phase 6 Vector Quantization (SQ8 & PQ)\",\n");
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        escape_json(&current_utc_rfc3339())
    ));
    ctx.prov.append_json_fields(&mut out, ctx.hw);
    append_hardware_json(&mut out, ctx.hw);

    out.push_str(&format!(
        "  \"arm_id\": \"{}\",\n",
        escape_json(ctx.spec.arm_id)
    ));
    out.push_str(&format!(
        "  \"quantizer_kind\": \"{}\",\n",
        escape_json(ctx.spec.quantizer_kind)
    ));
    out.push_str(&format!(
        "  \"storage_mode\": \"{}\",\n",
        escape_json(ctx.spec.storage_mode)
    ));
    out.push_str(&format!(
        "  \"build_mode\": \"{}\",\n",
        escape_json(ctx.spec.build_mode_str)
    ));
    out.push_str(&format!("  \"m\": {},\n", ctx.spec.m));
    out.push_str(&format!(
        "  \"ef_construction\": {},\n",
        ctx.spec.ef_construction
    ));
    out.push_str(&format!("  \"scale\": {},\n", ctx.dataset.num_base));
    out.push_str(&format!("  \"dim\": {},\n", ctx.dataset.dim));
    out.push_str(&format!("  \"k\": {},\n", ctx.config.k));
    out.push_str(&format!(
        "  \"num_queries\": {},\n",
        ctx.dataset.query.rows()
    ));
    out.push_str(&format!("  \"reps\": {},\n", ctx.config.reps));
    out.push_str(&format!("  \"seed\": {},\n", ctx.config.seed));
    out.push_str("  \"metric\": \"l2\",\n");
    out.push_str(&format!(
        "  \"streaming_build\": {},\n",
        ctx.streaming_build
    ));
    out.push_str("  \"measured_this_run\": true,\n");
    out.push_str("  \"warmup_performed\": true,\n");
    out.push_str(&format!(
        "  \"source_kind\": \"{}\",\n",
        escape_json(ctx.dataset.source.kind_tag())
    ));

    match ctx.spec.kind {
        Phase6ArmKind::Sq8 => out.push_str("  \"sq8_kernel\": \"query_shifted\",\n"),
        _ => out.push_str("  \"sq8_kernel\": null,\n"),
    }
    match ctx.pq_config {
        Some(pq) => {
            out.push_str("  \"pq_config\": {\n");
            out.push_str(&format!("    \"m_pq\": {},\n", pq.m_pq));
            out.push_str(&format!("    \"k_centroids\": {},\n", pq.k_centroids));
            out.push_str(&format!(
                "    \"max_train_vectors\": {},\n",
                pq.max_train_vectors
            ));
            out.push_str(&format!(
                "    \"max_lloyd_iters\": {},\n",
                pq.max_lloyd_iters
            ));
            out.push_str(&format!("    \"seed\": {}\n", pq.seed));
            out.push_str("  },\n");
        }
        None => out.push_str("  \"pq_config\": null,\n"),
    }

    out.push_str("  \"dataset\": {\n");
    out.push_str(&format!(
        "    \"source_kind\": \"{}\",\n",
        escape_json(ctx.dataset.source.kind_tag())
    ));
    out.push_str(&format!(
        "    \"source_description\": \"{}\",\n",
        escape_json(&ctx.dataset.source.to_string())
    ));
    out.push_str(&format!(
        "    \"is_authentic_sift\": {},\n",
        ctx.dataset.source.is_authentic_sift()
    ));
    out.push_str(&format!(
        "    \"scale_vectors\": {},\n",
        ctx.dataset.num_base
    ));
    out.push_str(&format!("    \"num_base\": {},\n", ctx.dataset.num_base));
    out.push_str(&format!(
        "    \"num_query\": {},\n",
        ctx.dataset.query.rows()
    ));
    out.push_str(&format!("    \"dim\": {},\n", ctx.dataset.dim));
    out.push_str(&format!("    \"k\": {},\n", ctx.config.k));
    out.push_str(&format!("    \"seed\": {},\n", ctx.config.seed));
    out.push_str("    \"metric\": \"l2\"\n");
    out.push_str("  },\n");

    out.push_str(&format!(
        "  \"ground_truth_source\": \"{}\",\n",
        escape_json(&ctx.dataset.gt_source)
    ));
    out.push_str(&format!(
        "  \"ground_truth_cache_path\": {},\n",
        opt_path_json(ctx.dataset.gt_cache_path.as_deref())
    ));
    out.push_str(&format!(
        "  \"ground_truth_cache_hit\": {},\n",
        ctx.dataset.gt_cache_hit
    ));

    out.push_str(&format!(
        "  \"total_build_time_s\": {:.6},\n",
        ctx.total_build_time_s
    ));
    out.push_str(&format!(
        "  \"training_or_calibration_time_s\": {:.6},\n",
        ctx.training_or_calibration_time_s
    ));
    out.push_str(&format!("  \"encode_time_s\": {:.6},\n", ctx.encode_time_s));
    out.push_str(&format!(
        "  \"graph_build_time_s\": {:.6},\n",
        ctx.graph_build_time_s
    ));
    out.push_str(&format!(
        "  \"graph_fingerprint\": {},\n",
        ctx.graph_fingerprint
    ));
    out.push_str(&format!("  \"graph_fingerprint_hex\": \"{fp_hex}\",\n"));

    out.push_str(&format!(
        "  \"steady_state_bytes_per_vector\": {:.4},\n",
        ctx.mem.steady_state_bytes_per_vector
    ));
    out.push_str(&format!(
        "  \"graph_only_bytes_per_vector\": {:.4},\n",
        ctx.mem.graph_only_bytes_per_vector
    ));
    out.push_str(&format!("  \"codes_bytes_per_vector\": {codes_bv:.4},\n"));
    out.push_str(&format!(
        "  \"calibration_or_codebook_bytes_per_vector\": {cal_bv:.4},\n"
    ));
    out.push_str(&format!(
        "  \"raw_vectors_retained_bytes_per_vector\": {raw_bv:.4},\n"
    ));
    out.push_str(&format!("  \"codes_bytes\": {},\n", ctx.mem.codes_bytes));
    out.push_str(&format!(
        "  \"calibration_or_codebook_bytes\": {},\n",
        ctx.mem.calibration_or_codebook_bytes
    ));
    out.push_str(&format!(
        "  \"aux_vector_bytes\": {},\n",
        ctx.mem.aux_vector_bytes
    ));
    out.push_str(&format!(
        "  \"graph_only_bytes\": {},\n",
        ctx.mem.graph_only_bytes
    ));
    out.push_str(&format!(
        "  \"node_meta_bytes\": {},\n",
        ctx.mem.node_meta_bytes
    ));
    out.push_str(&format!(
        "  \"layer0_edges_bytes\": {},\n",
        ctx.mem.layer0_edges_bytes
    ));
    out.push_str(&format!(
        "  \"upper_layers_edges_bytes\": {},\n",
        ctx.mem.upper_layers_edges_bytes
    ));
    out.push_str(&format!(
        "  \"raw_vectors_retained_bytes\": {},\n",
        ctx.mem.raw_vectors_retained_bytes
    ));
    out.push_str(&format!(
        "  \"steady_state_total_bytes\": {},\n",
        ctx.mem.steady_state_total_bytes
    ));
    out.push_str(&format!("  \"build_peak_rss_mb\": {build_rss_json},\n"));
    out.push_str(&format!(
        "  \"query_serving_peak_rss_mb\": {query_rss_json},\n"
    ));
    out.push_str(&format!(
        "  \"query_rss_profiled_isolated\": {},\n",
        ctx.query_rss_profiled_isolated
    ));

    out.push_str("  \"memory_profile\": {\n");
    out.push_str(&format!(
        "    \"steady_state_bytes_per_vector\": {:.4},\n",
        ctx.mem.steady_state_bytes_per_vector
    ));
    out.push_str(&format!(
        "    \"graph_only_bytes_per_vector\": {:.4},\n",
        ctx.mem.graph_only_bytes_per_vector
    ));
    out.push_str(&format!("    \"codes_bytes_per_vector\": {codes_bv:.4},\n"));
    out.push_str(&format!(
        "    \"calibration_or_codebook_bytes_per_vector\": {cal_bv:.4},\n"
    ));
    out.push_str(&format!(
        "    \"raw_vectors_retained_bytes_per_vector\": {raw_bv:.4},\n"
    ));
    out.push_str(&format!("    \"codes_bytes\": {},\n", ctx.mem.codes_bytes));
    out.push_str(&format!(
        "    \"calibration_or_codebook_bytes\": {},\n",
        ctx.mem.calibration_or_codebook_bytes
    ));
    out.push_str(&format!(
        "    \"aux_vector_bytes\": {},\n",
        ctx.mem.aux_vector_bytes
    ));
    out.push_str(&format!(
        "    \"graph_only_bytes\": {},\n",
        ctx.mem.graph_only_bytes
    ));
    out.push_str(&format!(
        "    \"raw_vectors_retained_bytes\": {},\n",
        ctx.mem.raw_vectors_retained_bytes
    ));
    out.push_str(&format!(
        "    \"steady_state_total_bytes\": {},\n",
        ctx.mem.steady_state_total_bytes
    ));
    out.push_str(&format!("    \"build_peak_rss_mb\": {build_rss_json},\n"));
    out.push_str(&format!(
        "    \"query_serving_peak_rss_mb\": {query_rss_json},\n"
    ));
    out.push_str(&format!("    \"peak_rss_scope\": \"{rss_scope}\"\n"));
    out.push_str("  },\n");

    out.push_str("  \"sweep\": [\n");
    for (idx, pt) in ctx.sweep.iter().enumerate() {
        let comma = if idx + 1 < ctx.sweep.len() { "," } else { "" };
        out.push_str("    {\n");
        out.push_str(&format!("      \"ef_search\": {},\n", pt.ef_search));
        out.push_str(&format!(
            "      \"effective_ef_search\": {},\n",
            pt.effective_ef_search
        ));
        out.push_str(&format!(
            "      \"rerank_top_r\": {},\n",
            pt.rerank_depth.json_rerank_top_r()
        ));
        out.push_str(&format!(
            "      \"is_full_beam\": {},\n",
            pt.rerank_depth.is_full_beam()
        ));
        out.push_str(&format!("      \"recall_at_k\": {:.6},\n", pt.recall_at_k));
        out.push_str(&format!(
            "      \"qps\": {:.2},\n",
            pt.summary.median_latency.qps
        ));
        out.push_str(&format!(
            "      \"mean_ms\": {:.6},\n",
            pt.summary.median_latency.mean_ms
        ));
        out.push_str(&format!(
            "      \"p50_ms\": {:.6},\n",
            pt.summary.median_latency.p50_ms
        ));
        out.push_str(&format!(
            "      \"p95_ms\": {:.6},\n",
            pt.summary.median_latency.p95_ms
        ));
        out.push_str(&format!(
            "      \"p99_ms\": {:.6},\n",
            pt.summary.median_latency.p99_ms
        ));
        out.push_str(&format!(
            "      \"min_ms\": {:.6},\n",
            pt.summary.median_latency.min_ms
        ));
        out.push_str(&format!(
            "      \"max_ms\": {:.6},\n",
            pt.summary.median_latency.max_ms
        ));
        out.push_str("      \"spread\": {\n");
        out.push_str(&format!(
            "        \"mean_ms_min\": {:.6},\n",
            pt.summary.mean_ms_min
        ));
        out.push_str(&format!(
            "        \"mean_ms_max\": {:.6},\n",
            pt.summary.mean_ms_max
        ));
        out.push_str(&format!(
            "        \"mean_ms_spread\": {:.6},\n",
            pt.summary.mean_ms_spread
        ));
        out.push_str(&format!(
            "        \"p99_ms_min\": {:.6},\n",
            pt.summary.p99_ms_min
        ));
        out.push_str(&format!(
            "        \"p99_ms_max\": {:.6},\n",
            pt.summary.p99_ms_max
        ));
        out.push_str(&format!(
            "        \"p99_ms_spread\": {:.6},\n",
            pt.summary.p99_ms_spread
        ));
        out.push_str("        \"run_mean_latencies_ms\": ");
        push_f64_array(&mut out, &pt.summary.run_mean_latencies_ms);
        out.push_str(",\n");
        out.push_str("        \"run_p99_latencies_ms\": ");
        push_f64_array(&mut out, &pt.summary.run_p99_latencies_ms);
        out.push_str("\n      }\n");
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

pub(crate) fn serialize_matrix_summary_json(
    summary: &Phase6MatrixSummary,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    config: &Phase6MatrixConfig,
) -> String {
    let mut out = String::with_capacity(8192);
    out.push_str("{\n");
    out.push_str("  \"phase\": 6,\n");
    out.push_str(&format!(
        "  \"mode\": \"{}\",\n",
        if config.is_pilot { "pilot" } else { "matrix" }
    ));
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        escape_json(&current_utc_rfc3339())
    ));
    prov.append_json_fields(&mut out, hw);
    append_hardware_json(&mut out, hw);
    out.push_str(&format!("  \"scale\": {},\n", summary.scale));
    out.push_str(&format!("  \"k\": {},\n", config.k));
    out.push_str(&format!("  \"reps\": {},\n", config.reps));
    out.push_str(&format!("  \"seed\": {},\n", config.seed));
    out.push_str(&format!(
        "  \"ground_truth_source\": \"{}\",\n",
        escape_json(&summary.ground_truth_source)
    ));
    out.push_str(&format!(
        "  \"ground_truth_cache_hit\": {},\n",
        summary.ground_truth_cache_hit
    ));
    out.push_str(&format!("  \"total_arms\": {},\n", summary.total_arms));
    out.push_str(&format!(
        "  \"newly_measured_count\": {},\n",
        summary.newly_measured_count
    ));
    out.push_str(&format!(
        "  \"resumed_skipped_count\": {},\n",
        summary.resumed_skipped_count
    ));
    out.push_str("  \"measured_this_run\": true,\n");
    out.push_str("  \"arms\": [\n");
    for (idx, arm) in summary.arms.iter().enumerate() {
        let comma = if idx + 1 < summary.arms.len() {
            ","
        } else {
            ""
        };
        let fp_json = arm
            .graph_fingerprint_hex
            .as_deref()
            .map_or_else(|| "null".to_string(), |s| format!("\"{}\"", escape_json(s)));

        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"arm_id\": \"{}\",\n",
            escape_json(&arm.arm_id)
        ));
        out.push_str(&format!(
            "      \"status\": \"{}\",\n",
            escape_json(&arm.status)
        ));
        out.push_str(&format!(
            "      \"report_filename\": \"{}\",\n",
            escape_json(&arm.report_filename)
        ));
        out.push_str(&format!(
            "      \"quantizer_kind\": \"{}\",\n",
            escape_json(&arm.quantizer_kind)
        ));
        out.push_str(&format!(
            "      \"storage_mode\": \"{}\",\n",
            escape_json(&arm.storage_mode)
        ));
        out.push_str(&format!(
            "      \"build_mode\": \"{}\",\n",
            escape_json(&arm.build_mode)
        ));
        out.push_str(&format!("      \"m\": {},\n", arm.m));
        out.push_str(&format!(
            "      \"ef_construction\": {},\n",
            arm.ef_construction
        ));
        out.push_str(&format!("      \"graph_fingerprint_hex\": {fp_json},\n"));
        out.push_str(&format!(
            "      \"total_build_time_s\": {},\n",
            opt_f64_json_prec(arm.total_build_time_s, 6)
        ));
        out.push_str(&format!(
            "      \"steady_state_bytes_per_vector\": {},\n",
            opt_f64_json_prec(arm.steady_state_bytes_per_vector, 4)
        ));
        out.push_str(&format!(
            "      \"build_peak_rss_mb\": {},\n",
            opt_f64_json_prec(arm.build_peak_rss_mb, 2)
        ));
        out.push_str(&format!(
            "      \"query_serving_peak_rss_mb\": {},\n",
            opt_f64_json_prec(arm.query_serving_peak_rss_mb, 2)
        ));
        out.push_str(&format!(
            "      \"best_recall\": {},\n",
            opt_f64_json_prec(arm.best_recall, 6)
        ));
        out.push_str(&format!(
            "      \"best_qps\": {}\n",
            opt_f64_json_prec(arm.best_qps, 2)
        ));
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Checks whether an existing per-arm JSON file matches the current clean commit and config.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn can_resume_existing_arm_report(
    report_path: &Path,
    expected_commit: &str,
    current_git_dirty: bool,
    source_kind: &str,
    scale: usize,
    dim: usize,
    k: usize,
    num_queries: usize,
    reps: usize,
    seed: u64,
    ef_search_list: &[usize],
    rerank_ladder: &[RerankDepthSpec],
    profile_query_rss: bool,
    arm: &Phase6ArmSpec,
) -> bool {
    if expected_commit == "unknown" || !report_path.is_file() {
        return false;
    }
    let Ok(content) = fs::read_to_string(report_path) else {
        return false;
    };
    if !current_git_dirty && extract_json_bool_field(&content, "git_dirty") == Some(true) {
        return false;
    }
    if extract_json_u64_field(&content, "phase") != Some(6)
        || extract_json_bool_field(&content, "measured_this_run") != Some(true)
        || extract_json_string_field(&content, "git_commit").as_deref() != Some(expected_commit)
        || extract_json_string_field(&content, "source_kind").as_deref() != Some(source_kind)
        || extract_json_string_field(&content, "arm_id").as_deref() != Some(arm.arm_id)
        || extract_json_string_field(&content, "quantizer_kind").as_deref()
            != Some(arm.quantizer_kind)
        || extract_json_string_field(&content, "storage_mode").as_deref() != Some(arm.storage_mode)
        || extract_json_string_field(&content, "build_mode").as_deref() != Some(arm.build_mode_str)
        || extract_json_u64_field(&content, "m") != Some(arm.m as u64)
        || extract_json_u64_field(&content, "ef_construction") != Some(arm.ef_construction as u64)
        || extract_json_u64_field(&content, "scale") != Some(scale as u64)
        || extract_json_u64_field(&content, "dim") != Some(dim as u64)
        || extract_json_u64_field(&content, "k") != Some(k as u64)
        || extract_json_u64_field(&content, "num_queries") != Some(num_queries as u64)
        || extract_json_u64_field(&content, "reps") != Some(reps as u64)
        || extract_json_u64_field(&content, "seed") != Some(seed)
    {
        return false;
    }

    if arm.isolated_snapshot_kind().is_some()
        && profile_query_rss
        && extract_json_f64_field(&content, "query_serving_peak_rss_mb").is_none()
    {
        return false;
    }

    let Some(sweep_slice) = extract_json_array_slice(&content, "sweep") else {
        return false;
    };
    let objects = split_top_level_json_objects(sweep_slice);
    let expected_len = ef_search_list.len() * rerank_ladder.len();
    if objects.len() != expected_len {
        return false;
    }

    let mut idx = 0_usize;
    for &r_spec in rerank_ladder {
        for &ef in ef_search_list {
            let obj = objects[idx];
            idx += 1;
            let (_, expected_eff_ef) = r_spec.resolve_params(k, ef);
            if extract_json_u64_field(obj, "ef_search") != Some(ef as u64)
                || extract_json_u64_field(obj, "effective_ef_search")
                    != Some(expected_eff_ef as u64)
                || extract_json_bool_field(obj, "is_full_beam") != Some(r_spec.is_full_beam())
            {
                return false;
            }
            match r_spec {
                RerankDepthSpec::None => {
                    if !obj.contains("\"rerank_top_r\": null") {
                        return false;
                    }
                }
                RerankDepthSpec::Fixed(r) => {
                    if extract_json_u64_field(obj, "rerank_top_r") != Some(r as u64) {
                        return false;
                    }
                }
                RerankDepthSpec::FullBeam => {
                    if extract_json_string_field(obj, "rerank_top_r").as_deref()
                        != Some("full_beam")
                    {
                        return false;
                    }
                }
            }
        }
    }
    true
}

pub(crate) fn parse_resumed_arm_summary(
    report_path: &Path,
    arm: &Phase6ArmSpec,
    report_filename: &str,
) -> Phase6ArmSummaryEntry {
    let content = fs::read_to_string(report_path).unwrap_or_default();
    let mut best_recall: Option<f64> = None;
    let mut best_qps: Option<f64> = None;
    if let Some(sweep_slice) = extract_json_array_slice(&content, "sweep") {
        for obj in split_top_level_json_objects(sweep_slice) {
            if let Some(r) = extract_json_f64_field(obj, "recall_at_k") {
                best_recall = Some(best_recall.map_or(r, |b| b.max(r)));
            }
            if let Some(q) = extract_json_f64_field(obj, "qps") {
                best_qps = Some(best_qps.map_or(q, |b| b.max(q)));
            }
        }
    }

    Phase6ArmSummaryEntry {
        arm_id: arm.arm_id.to_string(),
        status: "resumed".to_string(),
        report_filename: report_filename.to_string(),
        quantizer_kind: arm.quantizer_kind.to_string(),
        storage_mode: arm.storage_mode.to_string(),
        build_mode: arm.build_mode_str.to_string(),
        m: arm.m,
        ef_construction: arm.ef_construction,
        graph_fingerprint_hex: extract_json_string_field(&content, "graph_fingerprint_hex"),
        total_build_time_s: extract_json_f64_field(&content, "total_build_time_s"),
        steady_state_bytes_per_vector: extract_json_f64_field(
            &content,
            "steady_state_bytes_per_vector",
        ),
        build_peak_rss_mb: extract_json_f64_field(&content, "build_peak_rss_mb"),
        query_serving_peak_rss_mb: extract_json_f64_field(&content, "query_serving_peak_rss_mb"),
        best_recall,
        best_qps,
    }
}
