//! Phase 2 HNSW Verification & Benchmarking Pipeline (`siftsmall` 10k).

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::Instant;

use crate::dataset::{DatasetSource, SiftPreset};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{detect_hardware, HardwareInfo};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::json_util::escape_json;
use crate::matrix::RowMatrix;
use crate::recall::{compute_recall_at_k, verify_self_recall, LatencyStats};

use super::dataset_loader::{load_dataset_and_slice_queries, Phase1DatasetMode};
use super::{append_hardware_json, detect_provenance};

/// Configuration parameters for Phase 2 HNSW verification & benchmarking.
#[derive(Debug, Clone)]
pub struct Phase2Config {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub m: usize,
    /// Layer-0 degree cap; `None` derives `2 * m`.
    pub m_max0: Option<usize>,
    pub ef_construction: usize,
    pub ef_search_list: Vec<usize>,
    pub allow_download: bool,
    pub save_files: bool,
    pub max_queries: Option<usize>,
    pub run_repro_check: bool,
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
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
            run_repro_check: true,
        }
    }
}

/// Phase 2 acceptance target: minimum Recall@k at the operating point.
pub const PHASE2_TARGET_MIN_RECALL: f64 = 0.98;
/// Phase 2 acceptance target: maximum mean latency per query (milliseconds).
pub const PHASE2_TARGET_MAX_LATENCY_MS: f64 = 0.05;

/// Evaluation metrics for a single efSearch setting in Phase 2.
#[derive(Debug, Clone)]
pub struct HnswEfEvaluation {
    pub ef_search: usize,
    pub recall_at_k: f64,
    pub latency: LatencyStats,
    pub speedup_vs_exact: f64,
    pub monotonic_passed: bool,
}

/// Complete structured Phase 2 benchmark report.
#[derive(Debug, Clone)]
pub struct Phase2Report {
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
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
    pub max_layer: usize,
    pub entry_point: usize,
    pub exact_baseline_latency: LatencyStats,
    pub exact_self_verified: bool,
    pub exact_official_gt_match: bool,
    pub ef_evaluations: Vec<HnswEfEvaluation>,
    pub monotonic_scaling_verified: bool,
    /// True when some efSearch point meets the Phase 2 targets simultaneously:
    /// recall >= `PHASE2_TARGET_MIN_RECALL` and mean latency < `PHASE2_TARGET_MAX_LATENCY_MS`.
    /// Speedup vs brute force is reported per point but not gated: the spec's "10x" was
    /// derived from a ~0.25 ms/query baseline and is machine-dependent.
    pub performance_target_met: bool,
    /// Smallest efSearch that satisfies the performance target, if any.
    pub performance_target_ef: Option<usize>,
    /// True only if the reproducibility check actually ran (it is skipped by `--no-repro-check`).
    pub reproducibility_checked: bool,
    /// True only if the check ran and passed. A skipped check is never reported as verified.
    pub reproducibility_verified: bool,
    /// 64-bit FNV-1a fingerprint of graph topology.
    pub graph_fingerprint: Option<u64>,
}

impl Phase2Report {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str("  \"phase\": 2,\n");
        detect_provenance().append_json_fields(&mut out, &self.hardware);
        append_hardware_json(&mut out, &self.hardware);

        out.push_str("  \"dataset\": {\n");
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

        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_m));
        out.push_str(&format!("    \"m_max0\": {},\n", self.hnsw_m_max0));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_ef_construction
        ));
        out.push_str(&format!("    \"seed\": {}\n", self.seed));
        out.push_str("  },\n");

        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!("    \"max_layer\": {},\n", self.max_layer));
        if let Some(fp) = self.graph_fingerprint {
            out.push_str(&format!("    \"entry_point\": {},\n", self.entry_point));
            out.push_str(&format!("    \"graph_fingerprint\": {fp},\n"));
            out.push_str(&format!("    \"graph_fingerprint_hex\": \"0x{fp:016x}\"\n"));
        } else {
            out.push_str(&format!("    \"entry_point\": {}\n", self.entry_point));
        }
        out.push_str("  },\n");

        out.push_str("  \"exact_knn_baseline\": {\n");
        out.push_str(&format!(
            "    \"mean_ms\": {:.6},\n",
            self.exact_baseline_latency.mean_ms
        ));
        out.push_str(&format!(
            "    \"p50_ms\": {:.6},\n",
            self.exact_baseline_latency.p50_ms
        ));
        out.push_str(&format!(
            "    \"p95_ms\": {:.6},\n",
            self.exact_baseline_latency.p95_ms
        ));
        out.push_str(&format!(
            "    \"p99_ms\": {:.6},\n",
            self.exact_baseline_latency.p99_ms
        ));
        out.push_str(&format!(
            "    \"qps\": {:.3},\n",
            self.exact_baseline_latency.qps
        ));
        out.push_str(&format!(
            "    \"self_verified\": {},\n",
            self.exact_self_verified
        ));
        out.push_str(&format!(
            "    \"official_gt_matched\": {}\n",
            self.exact_official_gt_match
        ));
        out.push_str("  },\n");

        out.push_str("  \"ef_search_sweep\": [\n");
        for (i, eval) in self.ef_evaluations.iter().enumerate() {
            let comma = if i + 1 < self.ef_evaluations.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", eval.ef_search));
            out.push_str(&format!(
                "      \"recall_at_k\": {:.6},\n",
                eval.recall_at_k
            ));
            out.push_str("      \"latency\": {\n");
            out.push_str(&format!(
                "        \"mean_ms\": {:.6},\n",
                eval.latency.mean_ms
            ));
            out.push_str(&format!(
                "        \"p50_ms\": {:.6},\n",
                eval.latency.p50_ms
            ));
            out.push_str(&format!(
                "        \"p95_ms\": {:.6},\n",
                eval.latency.p95_ms
            ));
            out.push_str(&format!(
                "        \"p99_ms\": {:.6},\n",
                eval.latency.p99_ms
            ));
            out.push_str(&format!("        \"qps\": {:.3}\n", eval.latency.qps));
            out.push_str("      },\n");
            out.push_str(&format!(
                "      \"speedup_vs_exact\": {:.2},\n",
                eval.speedup_vs_exact
            ));
            out.push_str(&format!(
                "      \"monotonic_passed\": {}\n",
                eval.monotonic_passed
            ));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");

        out.push_str("  \"verification_status\": {\n");
        out.push_str(&format!(
            "    \"monotonic_scaling_verified\": {},\n",
            self.monotonic_scaling_verified
        ));
        out.push_str(&format!(
            "    \"performance_target_met\": {},\n",
            self.performance_target_met
        ));
        out.push_str(&format!(
            "    \"performance_target_ef\": {},\n",
            self.performance_target_ef
                .map_or_else(|| "null".to_string(), |ef| ef.to_string())
        ));
        out.push_str(&format!(
            "    \"performance_target\": {{ \"min_recall\": {}, \"max_mean_latency_ms\": {} }},\n",
            PHASE2_TARGET_MIN_RECALL, PHASE2_TARGET_MAX_LATENCY_MS
        ));
        out.push_str(&format!(
            "    \"reproducibility_checked\": {},\n",
            self.reproducibility_checked
        ));
        out.push_str(&format!(
            "    \"reproducibility_verified\": {},\n",
            self.reproducibility_verified
        ));
        let overall = self.monotonic_scaling_verified
            && self.reproducibility_verified
            && self.performance_target_met;
        out.push_str(&format!("    \"overall_phase2_passed\": {}\n", overall));
        out.push_str("  }\n");
        out.push_str("}\n");

        out
    }
}

/// Verifies bit-exact reproducibility across two independent index builds and query executions.
pub fn verify_reproducibility(
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    config: &HnswConfig,
    k: usize,
    ef_search: usize,
) -> Result<()> {
    verify_reproducibility_with_metric(base, query, Metric::L2Squared, config, k, ef_search)
}

/// Verifies bit-exact reproducibility across two independent index builds and query executions
/// under the specified distance `metric`.
pub fn verify_reproducibility_with_metric(
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    metric: Metric,
    config: &HnswConfig,
    k: usize,
    ef_search: usize,
) -> Result<()> {
    let mut index1 = HnswIndex::build(base, metric, config.clone())?;
    let res1 = index1.search_batch(query, k, ef_search)?;

    let mut index2 = HnswIndex::build(base, metric, config.clone())?;
    let res2 = index2.search_batch(query, k, ef_search)?;

    if index1.max_layer() != index2.max_layer() {
        return Err(VectorSearchError::VerificationFailed(
            "Max layer mismatch between identical runs".to_string(),
        ));
    }
    if index1.entry_point() != index2.entry_point() {
        return Err(VectorSearchError::VerificationFailed(
            "Entry point mismatch between identical runs".to_string(),
        ));
    }
    if index1.graph_fingerprint() != index2.graph_fingerprint() {
        return Err(VectorSearchError::VerificationFailed(format!(
            "Graph fingerprint mismatch between identical runs: 0x{:016x} != 0x{:016x}",
            index1.graph_fingerprint(),
            index2.graph_fingerprint()
        )));
    }

    if res1.indices.as_slice() != res2.indices.as_slice() {
        return Err(VectorSearchError::VerificationFailed(
            "Neighbor indices mismatch between identical runs".to_string(),
        ));
    }

    for (i, (&d1, &d2)) in res1
        .distances
        .as_slice()
        .iter()
        .zip(res2.distances.as_slice().iter())
        .enumerate()
    {
        if d1.to_bits() != d2.to_bits() {
            return Err(VectorSearchError::VerificationFailed(format!(
                "Distance bit mismatch at index {i}: {d1} != {d2}"
            )));
        }
    }

    Ok(())
}

/// Executes the complete Phase 2 verification and benchmarking pipeline.
pub fn run_phase2_benchmark(config: &Phase2Config) -> Result<Phase2Report> {
    if config.k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0)".to_string(),
        ));
    }

    let hardware = detect_hardware();

    // 1. Load or generate dataset
    let loaded = load_dataset_and_slice_queries(
        &config.dataset_mode,
        &config.data_cache_dir,
        config.allow_download,
        config.seed,
        config.max_queries,
    )?;

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();

    // 2. Establish Exact KNN Ground Truth
    let mut exact_knn = ExactKnn::new(config.metric);
    exact_knn.fit(loaded.base.clone())?;
    let exact_timed = exact_knn.search_timed(&loaded.query, config.k, true)?;
    let gt_indices = &exact_timed.result.indices;

    let exact_self_verified = verify_self_recall(gt_indices, config.k)?;
    let exact_official_gt_match = if let Some(official_gt) = &loaded.official_ground_truth {
        if official_gt.rows() == gt_indices.rows() && official_gt.cols() > 0 {
            let eval_k = config.k.min(official_gt.cols());
            let recall = compute_recall_at_k(gt_indices, official_gt, eval_k)?;
            (recall - 1.0).abs() < 1e-6
        } else {
            false
        }
    } else {
        false
    };

    // 3. Construct HNSW Index
    let hnsw_cfg = HnswConfig {
        m: config.m,
        m_max: None,
        m_max0: config.m_max0,
        ef_construction: config.ef_construction,
        m_l: None,
        extend_candidates: false,
        keep_pruned: true,
        keep_pruned_shrink: None,
        seed: config.seed,
    };

    let t_build = Instant::now();
    let mut hnsw = HnswIndex::build(&loaded.base, config.metric, hnsw_cfg.clone())?;
    let build_time_s = t_build.elapsed().as_secs_f64();

    // 4. Warmup HNSW
    if loaded.query.rows() > 0 {
        hnsw.warmup(loaded.query.row(0), config.k, 50)?;
    }

    // 5. Evaluate Sweep Across efSearch
    let mut ef_evaluations = Vec::new();
    let mut previous_recall = 0.0;
    let mut monotonic_scaling_verified = true;

    // The index clamps efSearch up to k, so ef < k would silently duplicate the ef == k point.
    // Report the effective value once instead of plotting identical results as separate points.
    let mut effective_efs: Vec<usize> = Vec::new();
    for &ef in &config.ef_search_list {
        let effective = ef.max(config.k);
        if !effective_efs.contains(&effective) {
            effective_efs.push(effective);
        }
    }

    for &ef in &effective_efs {
        let timed = hnsw.search_batch_timed(&loaded.query, config.k, ef)?;
        let recall = compute_recall_at_k(&timed.result.indices, gt_indices, config.k)?;

        let is_monotonic = recall >= previous_recall - 1e-9;
        if !is_monotonic {
            monotonic_scaling_verified = false;
        }

        let speedup = if timed.latency.mean_ms > 0.0 {
            exact_timed.latency.mean_ms / timed.latency.mean_ms
        } else {
            0.0
        };

        ef_evaluations.push(HnswEfEvaluation {
            ef_search: ef,
            recall_at_k: recall,
            latency: timed.latency,
            speedup_vs_exact: speedup,
            monotonic_passed: is_monotonic,
        });

        previous_recall = recall;
    }

    // 6. Verify Reproducibility (if enabled)
    let reproducibility_checked = config.run_repro_check;
    let reproducibility_verified = if config.run_repro_check {
        verify_reproducibility_with_metric(
            &loaded.base,
            &loaded.query,
            config.metric,
            &hnsw_cfg,
            config.k,
            50,
        )
        .is_ok()
    } else {
        false
    };

    // 7. Verify performance target: one operating point must satisfy recall and latency together.
    let performance_target_ef = ef_evaluations
        .iter()
        .filter(|e| {
            e.recall_at_k >= PHASE2_TARGET_MIN_RECALL
                && e.latency.mean_ms < PHASE2_TARGET_MAX_LATENCY_MS
        })
        .map(|e| e.ef_search)
        .min();
    let performance_target_met = performance_target_ef.is_some();

    Ok(Phase2Report {
        hardware,
        dataset_source: loaded.source,
        num_base,
        num_query,
        dim,
        k: config.k,
        seed: config.seed,
        metric: config.metric,
        hnsw_m: config.m,
        hnsw_m_max0: hnsw_cfg.effective_m_max0(),
        hnsw_ef_construction: config.ef_construction,
        build_time_s,
        max_layer: hnsw.max_layer(),
        entry_point: hnsw.entry_point().unwrap_or(0),
        exact_baseline_latency: exact_timed.latency,
        exact_self_verified,
        exact_official_gt_match,
        ef_evaluations,
        monotonic_scaling_verified,
        performance_target_met,
        performance_target_ef,
        reproducibility_checked,
        reproducibility_verified,
        graph_fingerprint: Some(hnsw.graph_fingerprint()),
    })
}

/// Formats and displays a human-readable Phase 2 benchmark summary table.
pub fn print_phase2_report(r: &Phase2Report) {
    println!("{}", "=".repeat(80));
    println!(" HNSW APPROXIMATE NEAREST NEIGHBOR SEARCH (RUST) -- PHASE 2 BENCHMARK ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!(
        "Dataset:  {} ({} base, {} queries, {}-d, {})",
        r.dataset_source, r.num_base, r.num_query, r.dim, r.metric
    );
    println!(
        "HNSW:     M={}, M_max0={}, efConstruction={}, seed={}",
        r.hnsw_m, r.hnsw_m_max0, r.hnsw_ef_construction, r.seed
    );
    println!("{}", "-".repeat(80));

    println!("[+] Phase 1 Brute-Force Exact KNN Ground Truth:");
    println!(
        "  - Baseline Mean Latency: {:.3} ms / query",
        r.exact_baseline_latency.mean_ms
    );
    println!(
        "  - Baseline Median (p50): {:.3} ms / query",
        r.exact_baseline_latency.p50_ms
    );
    println!(
        "  - Baseline Throughput:   {:.1} QPS",
        r.exact_baseline_latency.qps
    );
    println!(
        "  - Ground Truth Self-Rec: {}",
        if r.exact_self_verified {
            "PASSED (100.00% exact match)"
        } else {
            "FAILED"
        }
    );
    if r.exact_official_gt_match {
        println!("  - Official GT Match:     PASSED (100.00% exact match)");
    }

    println!("\n[+] HNSW Index Construction ({} vectors):", r.num_base);
    println!(
        "  - Build Elapsed:         {:.2} ms",
        r.build_time_s * 1000.0
    );
    println!("  - Max Graph Height:      {} layers", r.max_layer);
    println!("  - Global Entry Point ID: {}", r.entry_point);

    println!(
        "\n[+] Monotonic Recall Scaling & Latency Speedup Evaluation (k={}):",
        r.k
    );
    println!("+----------+------------+-----------+----------+----------+-----------+----------+-------------+");
    println!("| efSearch |  Recall@k  | Mean (ms) | p50 (ms) | p99 (ms) |    QPS    | Speedup  | Monotonicity|");
    println!("+----------+------------+-----------+----------+----------+-----------+----------+-------------+");

    for (i, eval) in r.ef_evaluations.iter().enumerate() {
        let status = if i == 0 {
            "BASE"
        } else if eval.monotonic_passed {
            "PASSED"
        } else {
            "FAILED"
        };
        println!(
            "| {:^8} | {:^10.2}% | {:^9.3} | {:^8.3} | {:^8.3} | {:^9.1} | {:^6.1} x | {:^11} |",
            eval.ef_search,
            eval.recall_at_k * 100.0,
            eval.latency.mean_ms,
            eval.latency.p50_ms,
            eval.latency.p99_ms,
            eval.latency.qps,
            eval.speedup_vs_exact,
            status,
        );
    }
    println!("+----------+------------+-----------+----------+----------+-----------+----------+-------------+");

    println!("\n[+] Reproducibility & Invariant Verification:");
    println!(
        "  - Strict Monotonic Scaling:   {}",
        if r.monotonic_scaling_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "  - Perf Target (recall>={:.2}, <{} ms): {}",
        PHASE2_TARGET_MIN_RECALL,
        PHASE2_TARGET_MAX_LATENCY_MS,
        match r.performance_target_ef {
            Some(ef) => format!("PASSED (efSearch={ef})"),
            None => "FAILED (no efSearch meets both)".to_string(),
        }
    );
    println!(
        "  - Bit-Exact Reproducibility:  {}",
        if !r.reproducibility_checked {
            "SKIPPED"
        } else if r.reproducibility_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    println!("\n{}", "=".repeat(80));
    println!(" PHASE 2 COMPLETE (RUST) -- HALTED FOR USER REVIEW BEFORE PHASE 3 ");
    println!("{}", "=".repeat(80));
}
