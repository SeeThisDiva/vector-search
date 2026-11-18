//! Per-Cell and Summary JSON Serialization, Parsing, and Resume Verification for Phase 5.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use crate::benchmark::{append_hardware_json, ReportProvenance};
use crate::dataset::LoadedDataset;
use crate::error::{Result, VectorSearchError};
use crate::filter::DEFAULT_OPAQUE_EVAL_LIMIT;
use crate::hardware::{current_utc_rfc3339, HardwareInfo};
use crate::hnsw::HnswConfig;
use crate::json_util::{
    escape_json, extract_json_array_slice, extract_json_bool_field, extract_json_f64_field,
    extract_json_string_field, extract_json_u64_field, opt_path_json, split_top_level_json_objects,
};
use crate::recall::{LatencyStats, RepeatedSweepSummary};

use super::cache::{FilteredGtCacheArtifact, PredicateCacheArtifact};
use super::crossover::{CrossoverModeResult, Phase5CellRecord};
use super::metrics::Phase5ArmRecord;
use super::{
    cell_report_filename, should_evaluate_s1_uncapped, sigma_slug, MetadataMode, GT_GEN_VERSION,
    PRED_GEN_VERSION, S1_UNCAPPED_SIGMA_MIN,
};

/// Top-level summary returned by `run_phase5_matrix`.
#[derive(Debug, Clone)]
pub struct Phase5Summary {
    pub scale: usize,
    pub total_cells: usize,
    pub newly_measured_count: usize,
    pub resumed_skipped_count: usize,
    pub sensitivity_executed: bool,
    pub opaque_1m_spotcheck_executed: bool,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub cells: Vec<Phase5CellRecord>,
    pub crossovers_primary: Vec<CrossoverModeResult>,
    pub crossovers_secondary: Vec<CrossoverModeResult>,
}

/// Checks whether a previously generated per-cell JSON can be cleanly reused under `--resume`.
#[allow(clippy::too_many_arguments)]
pub fn can_resume_phase5_cell(
    cell_path: &Path,
    expected_commit: &str,
    current_git_dirty: bool,
    source_kind: &str,
    scale: usize,
    mode: MetadataMode,
    sigma: f64,
    k: usize,
    reps: usize,
) -> bool {
    if expected_commit == "unknown" || !cell_path.is_file() {
        return false;
    }
    let Ok(content) = fs::read_to_string(cell_path) else {
        return false;
    };
    if !current_git_dirty && content.contains("\"git_dirty\": true") {
        return false;
    }
    let expected_s1_uncapped = should_evaluate_s1_uncapped(sigma);
    content.contains("\"phase\": 5")
        && content.contains("\"measured_this_run\": true")
        && content.contains(&format!("\"git_commit\": \"{expected_commit}\""))
        && content.contains(&format!("\"dataset_source_kind\": \"{source_kind}\""))
        && content.contains(&format!("\"scale\": {scale}"))
        && content.contains(&format!("\"mode\": \"{}\"", mode.as_str()))
        && content.contains(&format!("\"sigma_slug\": \"{}\"", sigma_slug(sigma)))
        && content.contains(&format!("\"k\": {k}"))
        && content.contains(&format!("\"reps\": {reps}"))
        && content.contains(&format!("\"pred_gen_version\": {PRED_GEN_VERSION}"))
        && content.contains(&format!("\"gt_gen_version\": {GT_GEN_VERSION}"))
        && content.contains(&format!(
            "\"s1_uncapped_evaluated\": {expected_s1_uncapped}"
        ))
        && parse_phase5_cell_json(&content, &cell_report_filename(scale, mode, sigma)).is_ok()
}

/// Parses a `Phase5CellRecord` from a written per-cell JSON file (zero external crates).
pub fn parse_phase5_cell_json(content: &str, filename: &str) -> Result<Phase5CellRecord> {
    let parse_err = |field: &str| {
        VectorSearchError::InvalidArgument(format!(
            "Failed parsing '{field}' from Phase 5 cell JSON '{filename}'"
        ))
    };
    let scale =
        extract_json_u64_field(content, "scale").ok_or_else(|| parse_err("scale"))? as usize;
    let mode_raw = extract_json_string_field(content, "mode").ok_or_else(|| parse_err("mode"))?;
    let mode = MetadataMode::parse(&mode_raw)?;
    let sigma = extract_json_f64_field(content, "sigma").ok_or_else(|| parse_err("sigma"))?;
    let k = extract_json_u64_field(content, "k").ok_or_else(|| parse_err("k"))? as usize;
    let reps = extract_json_u64_field(content, "reps").ok_or_else(|| parse_err("reps"))? as usize;
    let expected_popcount = extract_json_u64_field(content, "expected_popcount")
        .ok_or_else(|| parse_err("expected_popcount"))? as usize;
    let actual_popcount = extract_json_u64_field(content, "actual_popcount")
        .ok_or_else(|| parse_err("actual_popcount"))? as usize;
    let clusters_c = extract_json_u64_field(content, "clusters_c")
        .ok_or_else(|| parse_err("clusters_c"))? as usize;
    let s1_uncapped_evaluated = extract_json_bool_field(content, "s1_uncapped_evaluated")
        .ok_or_else(|| parse_err("s1_uncapped_evaluated"))?;
    let predicate_cache_hit = extract_json_bool_field(content, "predicate_cache_hit")
        .ok_or_else(|| parse_err("predicate_cache_hit"))?;
    let predicate_gen_time_s = extract_json_f64_field(content, "predicate_gen_time_s")
        .ok_or_else(|| parse_err("predicate_gen_time_s"))?;
    let gt_cache_hit = extract_json_bool_field(content, "gt_cache_hit")
        .ok_or_else(|| parse_err("gt_cache_hit"))?;
    let gt_gen_time_s = extract_json_f64_field(content, "gt_gen_time_s")
        .ok_or_else(|| parse_err("gt_gen_time_s"))?;

    let arms_slice = extract_json_array_slice(content, "arms").ok_or_else(|| parse_err("arms"))?;
    let mut arms = Vec::new();
    for obj in split_top_level_json_objects(arms_slice) {
        arms.push(parse_arm_object(obj, filename)?);
    }

    Ok(Phase5CellRecord {
        report_filename: filename.to_string(),
        status: "resumed".to_string(),
        scale,
        mode,
        sigma,
        k,
        reps,
        expected_popcount,
        actual_popcount,
        clusters_c,
        s1_uncapped_evaluated,
        predicate_cache_hit,
        predicate_gen_time_s,
        gt_cache_hit,
        gt_gen_time_s,
        arms,
    })
}

fn parse_arm_object(obj: &str, filename: &str) -> Result<Phase5ArmRecord> {
    let parse_err = |field: &str| {
        VectorSearchError::InvalidArgument(format!(
            "Failed parsing arm field '{field}' in '{filename}'"
        ))
    };
    let num_queries = extract_json_u64_field(obj, "num_queries")
        .ok_or_else(|| parse_err("num_queries"))? as usize;
    let mean_ms = extract_json_f64_field(obj, "mean_ms").ok_or_else(|| parse_err("mean_ms"))?;
    let p50_ms = extract_json_f64_field(obj, "p50_ms").ok_or_else(|| parse_err("p50_ms"))?;
    let p95_ms = extract_json_f64_field(obj, "p95_ms").ok_or_else(|| parse_err("p95_ms"))?;
    let p99_ms = extract_json_f64_field(obj, "p99_ms").ok_or_else(|| parse_err("p99_ms"))?;
    let qps = extract_json_f64_field(obj, "qps").ok_or_else(|| parse_err("qps"))?;
    let summary = RepeatedSweepSummary {
        num_runs: extract_json_u64_field(obj, "reps").unwrap_or(1) as usize,
        median_latency: LatencyStats {
            num_queries,
            total_s: if qps > 0.0 {
                (num_queries as f64) / qps
            } else {
                0.0
            },
            qps,
            min_ms: 0.0,
            mean_ms,
            p50_ms,
            p95_ms,
            p99_ms,
            max_ms: 0.0,
        },
        mean_ms_min: extract_json_f64_field(obj, "mean_ms_min").unwrap_or(mean_ms),
        mean_ms_max: extract_json_f64_field(obj, "mean_ms_max").unwrap_or(mean_ms),
        mean_ms_spread: extract_json_f64_field(obj, "mean_ms_spread").unwrap_or(0.0),
        p99_ms_min: extract_json_f64_field(obj, "p99_ms_min").unwrap_or(p99_ms),
        p99_ms_max: extract_json_f64_field(obj, "p99_ms_max").unwrap_or(p99_ms),
        p99_ms_spread: extract_json_f64_field(obj, "p99_ms_spread").unwrap_or(0.0),
        run_mean_latencies_ms: Vec::new(),
        run_p99_latencies_ms: Vec::new(),
    };

    Ok(Phase5ArmRecord {
        strategy: extract_json_string_field(obj, "strategy")
            .ok_or_else(|| parse_err("strategy"))?,
        strategy_label: extract_json_string_field(obj, "strategy_label")
            .ok_or_else(|| parse_err("strategy_label"))?,
        ef_search: extract_json_u64_field(obj, "ef_search").ok_or_else(|| parse_err("ef_search"))?
            as usize,
        overfetch_cap: extract_json_u64_field(obj, "overfetch_cap")
            .ok_or_else(|| parse_err("overfetch_cap"))? as usize,
        overfetch_factor: extract_json_f64_field(obj, "overfetch_factor"),
        max_bridge_hops: extract_json_u64_field(obj, "max_bridge_hops").map(|v| v as usize),
        eval_limit: extract_json_u64_field(obj, "eval_limit").map(|v| v as usize),
        num_queries,
        is_subquery_probe: extract_json_bool_field(obj, "is_subquery_probe")
            .ok_or_else(|| parse_err("is_subquery_probe"))?,
        reps: extract_json_u64_field(obj, "reps").unwrap_or(1) as usize,
        recall_at_k: extract_json_f64_field(obj, "recall_at_k")
            .ok_or_else(|| parse_err("recall_at_k"))?,
        fewer_than_k_count: extract_json_u64_field(obj, "fewer_than_k_count").unwrap_or(0) as usize,
        fewer_than_k_rate: extract_json_f64_field(obj, "fewer_than_k_rate").unwrap_or(0.0),
        post_filter_cap_reached_rate: extract_json_f64_field(obj, "post_filter_cap_reached_rate")
            .unwrap_or(0.0),
        hop_budget_exhausted_rate: extract_json_f64_field(obj, "hop_budget_exhausted_rate")
            .unwrap_or(0.0),
        eval_limit_reached_rate: extract_json_f64_field(obj, "eval_limit_reached_rate")
            .unwrap_or(0.0),
        used_bruteforce_fallback_rate: extract_json_f64_field(obj, "used_bruteforce_fallback_rate")
            .unwrap_or(0.0),
        summary,
        dist_evals_mean: extract_json_f64_field(obj, "dist_evals_mean").unwrap_or(0.0),
        dist_evals_p50: extract_json_f64_field(obj, "dist_evals_p50").unwrap_or(0.0),
        dist_evals_p99: extract_json_f64_field(obj, "dist_evals_p99").unwrap_or(0.0),
        valid_dist_evals_mean: extract_json_f64_field(obj, "valid_dist_evals_mean").unwrap_or(0.0),
        valid_dist_evals_p50: extract_json_f64_field(obj, "valid_dist_evals_p50").unwrap_or(0.0),
        valid_dist_evals_p99: extract_json_f64_field(obj, "valid_dist_evals_p99").unwrap_or(0.0),
        bridge_dist_evals_mean: extract_json_f64_field(obj, "bridge_dist_evals_mean")
            .unwrap_or(0.0),
        bridge_dist_evals_p50: extract_json_f64_field(obj, "bridge_dist_evals_p50").unwrap_or(0.0),
        bridge_dist_evals_p99: extract_json_f64_field(obj, "bridge_dist_evals_p99").unwrap_or(0.0),
        upper_layer_dist_evals_mean: extract_json_f64_field(obj, "upper_layer_dist_evals_mean")
            .unwrap_or(0.0),
        visited_nodes_mean: extract_json_f64_field(obj, "visited_nodes_mean").unwrap_or(0.0),
        visited_matching_nodes_mean: extract_json_f64_field(obj, "visited_matching_nodes_mean")
            .unwrap_or(0.0),
        eta_graph_mean: extract_json_f64_field(obj, "eta_graph_mean").unwrap_or(0.0),
        eta_graph_pooled: extract_json_f64_field(obj, "eta_graph_pooled").unwrap_or(0.0),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn format_cell_json(
    cell: &Phase5CellRecord,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    loaded: &LoadedDataset,
    pred: &PredicateCacheArtifact,
    gt: &FilteredGtCacheArtifact,
    build_time_s: f64,
    build_qps: f64,
    graph_fingerprint: u64,
    graph_only_bpv: f64,
    peak_rss_mb: Option<f64>,
    hnsw_cfg: &HnswConfig,
    attr_seed: u64,
    clust_seed: u64,
) -> String {
    let mut hw_block = String::new();
    append_hardware_json(&mut hw_block, hw);

    let effective_cs = match cell.mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    let peak_rss_json = peak_rss_mb.map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));
    let keep_pruned_json = hnsw_cfg
        .keep_pruned_shrink
        .map_or_else(|| "null".to_string(), |v| v.to_string());
    let total_ram_json = hw
        .total_ram_gb
        .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));

    let arms_json = cell
        .arms
        .iter()
        .map(|a| a.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        r#"{{
  "phase": 5,
  "phase_name": "Phase 5 Filtered Search Benchmark",
  "benchmark": "phase5_cell",
  "task": "Milestone M3 Task T3.8 / T3.9",
  "generated_at": "{}",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "measured_this_run": true,
  "provenance": {{
    "os": "{}",
    "arch": "{}",
    "cpu_model": "{}",
    "logical_cores": {},
    "total_ram_gb": {total_ram_json},
    "power_mode": "{}",
    "processor_affinity_mask": "{}"
  }},
{hw_block}
  "dataset_source": "{}",
  "dataset_source_kind": "{}",
  "is_authentic_sift": {},
  "scale": {},
  "num_base": {},
  "num_query_total": {},
  "dim": {},
  "k": {},
  "reps": {},
  "metric": "l2_squared",
  "index_build": {{
    "m": {},
    "ef_construction": {},
    "seed": {},
    "keep_pruned_shrink": {keep_pruned_json},
    "build_time_s": {build_time_s:.6},
    "build_qps": {build_qps:.2},
    "graph_fingerprint": {graph_fingerprint},
    "graph_fingerprint_hex": "0x{graph_fingerprint:016x}",
    "graph_only_bytes_per_vector": {graph_only_bpv:.2},
    "peak_rss_mb": {peak_rss_json}
  }},
  "mode": "{}",
  "sigma": {:.6},
  "sigma_slug": "{}",
  "expected_popcount": {},
  "actual_popcount": {},
  "clusters_c": {},
  "attr_seed": {attr_seed},
  "clust_seed": {effective_cs},
  "pred_gen_version": {PRED_GEN_VERSION},
  "gt_gen_version": {GT_GEN_VERSION},
  "predicate_cache_path": {},
  "predicate_cache_hit": {},
  "predicate_gen_time_s": {:.6},
  "gt_ivecs_path": {},
  "gt_fvecs_path": {},
  "gt_cache_hit": {},
  "gt_gen_time_s": {:.6},
  "option_c_schedule": {{
    "tier": "Option C",
    "s1_uncapped_evaluated_sigma_min": {:.6},
    "s1_uncapped_evaluated": {}
  }},
  "arms": [
{arms_json}
  ]
}}
"#,
        escape_json(&current_utc_rfc3339()),
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        escape_json(&hw.os),
        escape_json(&hw.arch),
        escape_json(&hw.cpu_model),
        hw.logical_cores,
        escape_json(&prov.power_mode),
        escape_json(&prov.processor_affinity_mask),
        escape_json(&loaded.source.to_string()),
        escape_json(loaded.source.kind_tag()),
        loaded.source.is_authentic_sift(),
        cell.scale,
        loaded.base.rows(),
        loaded.query.rows(),
        loaded.base.cols(),
        cell.k,
        cell.reps,
        hnsw_cfg.m,
        hnsw_cfg.ef_construction,
        hnsw_cfg.seed,
        cell.mode.as_str(),
        cell.sigma,
        sigma_slug(cell.sigma),
        cell.expected_popcount,
        cell.actual_popcount,
        cell.clusters_c,
        opt_path_json(Some(&pred.cache_path)),
        cell.predicate_cache_hit,
        cell.predicate_gen_time_s,
        opt_path_json(Some(&gt.ivecs_path)),
        opt_path_json(Some(&gt.fvecs_path)),
        cell.gt_cache_hit,
        cell.gt_gen_time_s,
        S1_UNCAPPED_SIGMA_MIN,
        cell.s1_uncapped_evaluated,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn format_summary_json(
    scale: usize,
    total_cells: usize,
    newly_measured_count: usize,
    resumed_skipped_count: usize,
    sensitivity_executed: bool,
    opaque_1m_spotcheck_executed: bool,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    dataset_source: &str,
    is_authentic_sift: bool,
    cells: &[Phase5CellRecord],
    crossovers_primary: &[CrossoverModeResult],
    crossovers_secondary: &[CrossoverModeResult],
) -> String {
    let mut hw_block = String::new();
    append_hardware_json(&mut hw_block, hw);

    let cells_json = cells
        .iter()
        .map(|c| {
            format!(
                r#"    {{
      "report_filename": "{}",
      "status": "{}",
      "mode": "{}",
      "sigma": {:.6},
      "s1_uncapped_evaluated": {},
      "predicate_cache_hit": {},
      "gt_cache_hit": {},
      "num_arms": {}
    }}"#,
                escape_json(&c.report_filename),
                escape_json(&c.status),
                c.mode.as_str(),
                c.sigma,
                c.s1_uncapped_evaluated,
                c.predicate_cache_hit,
                c.gt_cache_hit,
                c.arms.len(),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let primary_json = crossovers_primary
        .iter()
        .map(|c| c.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");
    let secondary_json = crossovers_secondary
        .iter()
        .map(|c| c.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        r#"{{
  "phase": 5,
  "phase_name": "Phase 5 Filtered Search Summary",
  "task": "Milestone M3 Task T3.8 / T3.9",
  "generated_at": "{}",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "dataset_source": "{}",
  "is_authentic_sift": {},
  "scale": {scale},
  "total_cells": {total_cells},
  "newly_measured_count": {newly_measured_count},
  "resumed_skipped_count": {resumed_skipped_count},
  "sensitivity_executed": {sensitivity_executed},
  "opaque_1m_spotcheck_executed": {opaque_1m_spotcheck_executed},
  "option_c_schedule": {{
    "tier": "Option C",
    "default_opaque_eval_limit": {DEFAULT_OPAQUE_EVAL_LIMIT},
    "s1_uncapped_evaluated_sigma_min": {S1_UNCAPPED_SIGMA_MIN:.6}
  }},
{hw_block}
  "cells": [
{cells_json}
  ],
  "crossovers_primary_0_95": [
{primary_json}
  ],
  "crossovers_secondary_0_99": [
{secondary_json}
  ]
}}
"#,
        escape_json(&current_utc_rfc3339()),
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        escape_json(dataset_source),
        is_authentic_sift,
    )
}

pub(crate) fn collect_all_scale_cells_in_dir(
    output_dir: &Path,
    scale: usize,
) -> Vec<Phase5CellRecord> {
    let prefix = format!("cell_n{scale}_");
    let mut records = Vec::new();
    let Ok(entries) = fs::read_dir(output_dir) else {
        return records;
    };
    let mut files: Vec<(String, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_string_lossy().to_string();
            if p.is_file() && name.starts_with(&prefix) && name.ends_with(".json") {
                Some((name, p))
            } else {
                None
            }
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, path) in files {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(parsed) = parse_phase5_cell_json(&content, &name) {
                records.push(parsed);
            }
        }
    }
    records
}
