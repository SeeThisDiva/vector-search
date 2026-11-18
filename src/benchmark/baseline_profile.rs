//! Multi-Scale Baseline HNSW Profiling (10k, 100k, 1M) in Per-Scale Isolated OS Child Processes.

#![forbid(unsafe_code)]

use std::path::Path;
use std::time::Instant;

use crate::dataset::{load_or_download_sift_with_options, DatasetSource, SiftPreset};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{
    detect_hardware, get_peak_rss_mb, peak_rss_measurement_method, HardwareInfo,
};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::json_util::escape_json;
use crate::recall::{compute_recall_at_k, summarize_repeated_latencies};

use super::{append_hardware_json, detect_provenance};

/// Profile measurement for an individual scale in the baseline multi-scale profile.
#[derive(Debug, Clone)]
pub struct BaselineScaleProfile {
    /// False for rows copied from an earlier recorded run (see `load_or_default_baseline_profiles`);
    /// true only for rows measured by the run that produced the report.
    pub measured_this_run: bool,
    pub scale_vectors: usize,
    pub build_time_s: f64,
    pub build_qps: f64,
    pub peak_rss_mb: f64,
    pub bytes_per_vector: f64,
    pub raw_vector_bytes_per_vector: f64,
    pub graph_bytes_per_vector: f64,
    pub graph_only_bytes_per_vector: f64,
    pub total_bytes_per_vector: f64,
    pub graph_memory_bytes: usize,
    pub graph_only_memory_bytes: usize,
    pub graph_memory_mb: f64,
    pub query_latency_ms: f64,
    pub query_qps: f64,
    pub k: usize,
    pub ef_search: usize,
    pub recall_at_10: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub latency_min_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_max_ms: f64,
    pub graph_fingerprint: Option<u64>,
}

/// Structured multi-scale baseline profiling report.
#[derive(Debug, Clone)]
pub struct BaselineMultiScaleReport {
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub hnsw_m: usize,
    pub hnsw_m_max0: usize,
    pub hnsw_ef_construction: usize,
    pub seed: u64,
    pub metric: Metric,
    pub peak_rss_measurement_method: String,
    pub scale_profiles: Vec<BaselineScaleProfile>,
}

impl BaselineMultiScaleReport {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str("  \"phase\": \"2_baseline_profile\",\n");
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
        out.push_str("    \"num_base_total\": 1000000,\n");
        out.push_str("    \"dim\": 128,\n");
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

        out.push_str(&format!(
            "  \"peak_rss_measurement_method\": \"{}\",\n",
            escape_json(&self.peak_rss_measurement_method)
        ));

        out.push_str("  \"scale_profiles\": [\n");
        for (i, p) in self.scale_profiles.iter().enumerate() {
            let comma = if i + 1 < self.scale_profiles.len() {
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
                "      \"bytes_per_vector\": {:.2},\n",
                p.bytes_per_vector
            ));
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
            out.push_str(&format!(
                "      \"graph_memory_mb\": {:.2},\n",
                p.graph_memory_mb
            ));
            out.push_str(&format!(
                "      \"query_latency_ms\": {:.6},\n",
                p.query_latency_ms
            ));
            out.push_str(&format!("      \"query_qps\": {:.2},\n", p.query_qps));
            out.push_str(&format!("      \"k\": {},\n", p.k));
            out.push_str(&format!("      \"ef_search\": {},\n", p.ef_search));
            out.push_str(&format!("      \"recall_at_10\": {:.6},\n", p.recall_at_10));
            out.push_str(&format!("      \"max_layer\": {},\n", p.max_layer));
            out.push_str(&format!("      \"entry_point\": {},\n", p.entry_point));
            out.push_str(&format!(
                "      \"latency_min_ms\": {:.6},\n",
                p.latency_min_ms
            ));
            out.push_str(&format!(
                "      \"latency_p50_ms\": {:.6},\n",
                p.latency_p50_ms
            ));
            out.push_str(&format!(
                "      \"latency_p95_ms\": {:.6},\n",
                p.latency_p95_ms
            ));
            out.push_str(&format!(
                "      \"latency_p99_ms\": {:.6},\n",
                p.latency_p99_ms
            ));
            out.push_str(&format!(
                "      \"latency_max_ms\": {:.6}\n",
                p.latency_max_ms
            ));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }
}

/// Profiles a single SIFT1M prefix scale (`scale` base vectors) by loading ONLY `scale` rows
/// from disk so the process high-water mark RSS reflects only that scale's memory footprint.
pub fn profile_single_sift1m_scale(
    data_cache_dir: &Path,
    scale: usize,
    seed: u64,
    allow_download: bool,
    sweep_runs: usize,
) -> Result<BaselineScaleProfile> {
    if scale == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "scale must be positive (> 0)".to_string(),
        ));
    }
    let num_eval_queries = if scale >= 1_000_000 { 10_000 } else { 1_000 };
    let load_gt = scale == 1_000_000;
    let loaded = load_or_download_sift_with_options(
        SiftPreset::Sift1M,
        data_cache_dir,
        allow_download,
        seed,
        Some(scale),
        Some(num_eval_queries),
        load_gt,
    )?;

    let actual_scale = loaded.base.rows();
    let hnsw_cfg = HnswConfig {
        m: 16,
        m_max: None,
        m_max0: Some(32),
        ef_construction: 100,
        m_l: None,
        extend_candidates: false,
        keep_pruned: true,
        keep_pruned_shrink: None,
        seed,
    };

    let t_build_start = Instant::now();
    let mut index = HnswIndex::build(&loaded.base, Metric::L2Squared, hnsw_cfg)?;
    let build_time_s = t_build_start.elapsed().as_secs_f64();
    let build_qps = (actual_scale as f64) / build_time_s;

    let graph_memory_bytes = index.memory_bytes();
    let graph_only_memory_bytes = index.graph_only_memory_bytes();
    let graph_memory_mb = index.memory_mb();
    let bytes_per_vector = index.bytes_per_vector();
    let graph_only_bytes_per_vector = index.graph_only_bytes_per_vector();
    let peak_rss_mb = get_peak_rss_mb().unwrap_or(0.0);
    let max_layer = index.max_layer();
    let entry_point = index.entry_point().unwrap_or(0);
    let graph_fingerprint = Some(index.graph_fingerprint());

    if loaded.query.rows() > 0 {
        index.warmup(loaded.query.row(0), 10, 50)?;
    }

    let runs = sweep_runs.max(1);
    let mut run_stats = Vec::with_capacity(runs);
    let mut last_indices = None;
    for _ in 0..runs {
        let timed = index.search_batch_timed(&loaded.query, 10, 50)?;
        run_stats.push(timed.latency);
        last_indices = Some(timed.result.indices);
    }
    let summary = summarize_repeated_latencies(&run_stats);
    let cand_indices = last_indices.unwrap();

    let recall_at_10 = if actual_scale == 1_000_000 {
        if let Some(ref gt) = loaded.official_ground_truth {
            compute_recall_at_k(&cand_indices, gt, 10)?
        } else {
            let exact_queries = loaded.query.slice_first_rows(100.min(loaded.query.rows()));
            let exact_res = ExactKnn::search_parallel_with_base(
                Metric::L2Squared,
                &loaded.base,
                &exact_queries,
                10,
                0,
            )?;
            let cand_slice = cand_indices.slice_first_rows(exact_queries.rows());
            compute_recall_at_k(&cand_slice, &exact_res.indices, 10)?
        }
    } else {
        let exact_res = ExactKnn::search_parallel_with_base(
            Metric::L2Squared,
            &loaded.base,
            &loaded.query,
            10,
            0,
        )?;
        compute_recall_at_k(&cand_indices, &exact_res.indices, 10)?
    };

    let raw_vector_bytes_per_vector = (loaded.base.cols() * std::mem::size_of::<f32>()) as f64;
    let graph_bytes_per_vector = bytes_per_vector;
    let total_bytes_per_vector = raw_vector_bytes_per_vector + graph_bytes_per_vector;

    Ok(BaselineScaleProfile {
        measured_this_run: true,
        scale_vectors: actual_scale,
        build_time_s,
        build_qps,
        peak_rss_mb,
        bytes_per_vector,
        raw_vector_bytes_per_vector,
        graph_bytes_per_vector,
        graph_only_bytes_per_vector,
        total_bytes_per_vector,
        graph_memory_bytes,
        graph_only_memory_bytes,
        graph_memory_mb,
        query_latency_ms: summary.median_latency.mean_ms,
        query_qps: summary.median_latency.qps,
        k: 10,
        ef_search: 50,
        recall_at_10,
        max_layer,
        entry_point,
        latency_min_ms: summary.median_latency.min_ms,
        latency_p50_ms: summary.median_latency.p50_ms,
        latency_p95_ms: summary.median_latency.p95_ms,
        latency_p99_ms: summary.median_latency.p99_ms,
        latency_max_ms: summary.median_latency.max_ms,
        graph_fingerprint,
    })
}

pub(crate) fn format_scale_profile_kv(p: &BaselineScaleProfile) -> String {
    let fp_str = p
        .graph_fingerprint
        .map(|v| v.to_string())
        .unwrap_or_default();
    format!(
        "SCALE_PROFILE_BEGIN\n\
         measured_this_run={}\n\
         scale_vectors={}\n\
         build_time_s={:.9}\n\
         build_qps={:.6}\n\
         peak_rss_mb={:.6}\n\
         bytes_per_vector={:.6}\n\
         raw_vector_bytes_per_vector={:.6}\n\
         graph_bytes_per_vector={:.6}\n\
         graph_only_bytes_per_vector={:.6}\n\
         total_bytes_per_vector={:.6}\n\
         graph_memory_bytes={}\n\
         graph_only_memory_bytes={}\n\
         graph_memory_mb={:.6}\n\
         query_latency_ms={:.9}\n\
         query_qps={:.6}\n\
         k={}\n\
         ef_search={}\n\
         recall_at_10={:.9}\n\
         max_layer={}\n\
         entry_point={}\n\
         latency_min_ms={:.9}\n\
         latency_p50_ms={:.9}\n\
         latency_p95_ms={:.9}\n\
         latency_p99_ms={:.9}\n\
         latency_max_ms={:.9}\n\
         graph_fingerprint={}\n\
         SCALE_PROFILE_END\n",
        p.measured_this_run,
        p.scale_vectors,
        p.build_time_s,
        p.build_qps,
        p.peak_rss_mb,
        p.bytes_per_vector,
        p.raw_vector_bytes_per_vector,
        p.graph_bytes_per_vector,
        p.graph_only_bytes_per_vector,
        p.total_bytes_per_vector,
        p.graph_memory_bytes,
        p.graph_only_memory_bytes,
        p.graph_memory_mb,
        p.query_latency_ms,
        p.query_qps,
        p.k,
        p.ef_search,
        p.recall_at_10,
        p.max_layer,
        p.entry_point,
        p.latency_min_ms,
        p.latency_p50_ms,
        p.latency_p95_ms,
        p.latency_p99_ms,
        p.latency_max_ms,
        fp_str,
    )
}

pub(crate) fn parse_scale_profile_kv(output: &str) -> Result<BaselineScaleProfile> {
    let mut in_block = false;
    let mut p = BaselineScaleProfile {
        measured_this_run: true,
        scale_vectors: 0,
        build_time_s: 0.0,
        build_qps: 0.0,
        peak_rss_mb: 0.0,
        bytes_per_vector: 0.0,
        raw_vector_bytes_per_vector: 0.0,
        graph_bytes_per_vector: 0.0,
        graph_only_bytes_per_vector: 0.0,
        total_bytes_per_vector: 0.0,
        graph_memory_bytes: 0,
        graph_only_memory_bytes: 0,
        graph_memory_mb: 0.0,
        query_latency_ms: 0.0,
        query_qps: 0.0,
        k: 10,
        ef_search: 50,
        recall_at_10: 0.0,
        max_layer: 0,
        entry_point: 0,
        latency_min_ms: 0.0,
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
        latency_p99_ms: 0.0,
        latency_max_ms: 0.0,
        graph_fingerprint: None,
    };
    let mut found_end = false;

    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line == "SCALE_PROFILE_BEGIN" {
            in_block = true;
            continue;
        }
        if line == "SCALE_PROFILE_END" {
            if in_block {
                found_end = true;
                break;
            }
            continue;
        }
        if !in_block {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            match k {
                "measured_this_run" => p.measured_this_run = v.parse().unwrap_or(true),
                "scale_vectors" => p.scale_vectors = v.parse().unwrap_or(0),
                "build_time_s" => p.build_time_s = v.parse().unwrap_or(0.0),
                "build_qps" => p.build_qps = v.parse().unwrap_or(0.0),
                "graph_fingerprint" => p.graph_fingerprint = v.parse().ok(),
                "peak_rss_mb" => p.peak_rss_mb = v.parse().unwrap_or(0.0),
                "bytes_per_vector" => p.bytes_per_vector = v.parse().unwrap_or(0.0),
                "raw_vector_bytes_per_vector" => {
                    p.raw_vector_bytes_per_vector = v.parse().unwrap_or(0.0)
                }
                "graph_bytes_per_vector" => p.graph_bytes_per_vector = v.parse().unwrap_or(0.0),
                "graph_only_bytes_per_vector" => {
                    p.graph_only_bytes_per_vector = v.parse().unwrap_or(0.0)
                }
                "total_bytes_per_vector" => p.total_bytes_per_vector = v.parse().unwrap_or(0.0),
                "graph_memory_bytes" => p.graph_memory_bytes = v.parse().unwrap_or(0),
                "graph_only_memory_bytes" => p.graph_only_memory_bytes = v.parse().unwrap_or(0),
                "graph_memory_mb" => p.graph_memory_mb = v.parse().unwrap_or(0.0),
                "query_latency_ms" => p.query_latency_ms = v.parse().unwrap_or(0.0),
                "query_qps" => p.query_qps = v.parse().unwrap_or(0.0),
                "k" => p.k = v.parse().unwrap_or(10),
                "ef_search" => p.ef_search = v.parse().unwrap_or(50),
                "recall_at_10" => p.recall_at_10 = v.parse().unwrap_or(0.0),
                "max_layer" => p.max_layer = v.parse().unwrap_or(0),
                "entry_point" => p.entry_point = v.parse().unwrap_or(0),
                "latency_min_ms" => p.latency_min_ms = v.parse().unwrap_or(0.0),
                "latency_p50_ms" => p.latency_p50_ms = v.parse().unwrap_or(0.0),
                "latency_p95_ms" => p.latency_p95_ms = v.parse().unwrap_or(0.0),
                "latency_p99_ms" => p.latency_p99_ms = v.parse().unwrap_or(0.0),
                "latency_max_ms" => p.latency_max_ms = v.parse().unwrap_or(0.0),
                _ => {}
            }
        }
    }

    if !found_end || p.scale_vectors == 0 {
        return Err(VectorSearchError::VerificationFailed(
            "Failed to parse isolated child process scale profile output".to_string(),
        ));
    }
    Ok(p)
}

/// Spawns a fresh OS child process to measure a single SIFT1M prefix scale with an unpolluted
/// Peak RSS high-water mark, falling back to in-process execution if invoked from a test harness.
pub fn spawn_isolated_scale_profile(
    data_cache_dir: &Path,
    scale: usize,
    seed: u64,
    allow_download: bool,
    sweep_runs: usize,
) -> Result<BaselineScaleProfile> {
    if let Ok(exe) = std::env::current_exe() {
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("--phase")
            .arg("3")
            .arg("--internal-profile-scale")
            .arg(scale.to_string())
            .arg("--data-dir")
            .arg(data_cache_dir.as_os_str())
            .arg("--seed")
            .arg(seed.to_string())
            .arg("--sweep-runs")
            .arg(sweep_runs.max(1).to_string());
        if !allow_download {
            cmd.arg("--offline");
        }
        if let Ok(output) = cmd.output() {
            if output.status.success() {
                if let Ok(stdout) = String::from_utf8(output.stdout) {
                    if let Ok(profile) = parse_scale_profile_kv(&stdout) {
                        return Ok(profile);
                    }
                }
            }
        }
    }
    profile_single_sift1m_scale(data_cache_dir, scale, seed, allow_download, sweep_runs)
}

/// Executes multi-scale baseline HNSW profiling across the provided vector scales.
/// Each scale runs in its own isolated OS process and loads only `scale` vectors from disk
/// so that per-scale Peak RSS is never polluted by earlier or larger allocations.
pub fn run_baseline_multi_scale_profile(
    data_cache_dir: &Path,
    scales: &[usize],
    seed: u64,
    allow_download: bool,
) -> Result<BaselineMultiScaleReport> {
    println!("{}", "=".repeat(80));
    println!(" HNSW MULTI-SCALE BASELINE PROFILING (10k, 100k, 1M) ");
    println!("{}", "=".repeat(80));

    let hw = detect_hardware();
    println!("Hardware: {hw}");

    let dataset_source = load_or_download_sift_with_options(
        SiftPreset::Sift1M,
        data_cache_dir,
        allow_download,
        seed,
        Some(1),
        Some(1),
        false,
    )?
    .source;
    println!("  - Dataset Provenance: {dataset_source}");

    let eval_scales = if scales.is_empty() {
        &[10_000, 100_000, 1_000_000][..]
    } else {
        scales
    };

    let mut profiles = Vec::with_capacity(eval_scales.len());

    for (idx, &scale) in eval_scales.iter().enumerate() {
        println!(
            "\n[{}/{}] Profiling Scale in Isolated Process: {} vectors...",
            idx + 1,
            eval_scales.len(),
            scale
        );
        let p = spawn_isolated_scale_profile(data_cache_dir, scale, seed, allow_download, 5)?;
        println!(
            "  - Build finished in {:.3} s ({:.1} vectors/s)",
            p.build_time_s, p.build_qps
        );
        println!(
            "  - Graph memory: {:.2} MiB ({:.2} B/vec total, {:.2} B/vec graph-only)",
            p.graph_memory_mb, p.graph_bytes_per_vector, p.graph_only_bytes_per_vector
        );
        println!("  - Isolated Process Peak RSS: {:.2} MiB", p.peak_rss_mb);
        println!("  - Recall@10: {:.2}%", p.recall_at_10 * 100.0);
        println!(
            "  - Query latency (median of 5 runs): {:.3} ms ({:.1} QPS)",
            p.query_latency_ms, p.query_qps
        );
        profiles.push(p);
    }

    Ok(BaselineMultiScaleReport {
        hardware: hw,
        dataset_source,
        hnsw_m: 16,
        hnsw_m_max0: 32,
        hnsw_ef_construction: 100,
        seed,
        metric: Metric::L2Squared,
        peak_rss_measurement_method: peak_rss_measurement_method(true),
        scale_profiles: profiles,
    })
}

/// Formats and prints a multi-scale baseline profiling summary table to stdout.
pub fn print_baseline_profile_report(r: &BaselineMultiScaleReport) {
    println!("\n{}", "=".repeat(80));
    println!(" MULTI-SCALE BASELINE HNSW PROFILING SUMMARY ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!("Dataset:  Authentic SIFT1M (1,000,000 base vectors, 128-d, L2 metric)");
    println!("Config:   M=16, M0=32, efConstruction=100, seed={}", r.seed);
    println!("Peak RSS: {}", r.peak_rss_measurement_method);
    println!(
        "\n+---------+------------+-----------+-------------+------------+-------------+--------------+-------------+--------------------+-----------+------------+"
    );
    println!(
        "|  Scale  | Build Time | Build QPS | Peak RSS MB | Raw B/Vec  | Graph B/Vec | Graph-Only B | Total B/Vec | Query Latency (ms) | Query QPS | Recall@10  |"
    );
    println!(
        "+---------+------------+-----------+-------------+------------+-------------+--------------+-------------+--------------------+-----------+------------+"
    );

    for p in &r.scale_profiles {
        let scale_label = if p.scale_vectors == 1_000_000 {
            "1M".to_string()
        } else {
            format!("{}k", p.scale_vectors / 1000)
        };
        println!(
            "| {:^7} | {:^8.2} s | {:^9.1} | {:^11.2} | {:^10.1} | {:^11.2} | {:^12.2} | {:^11.2} | {:^18.3} | {:^9.1} | {:^9.2}% |",
            scale_label,
            p.build_time_s,
            p.build_qps,
            p.peak_rss_mb,
            p.raw_vector_bytes_per_vector,
            p.graph_bytes_per_vector,
            p.graph_only_bytes_per_vector,
            p.total_bytes_per_vector,
            p.query_latency_ms,
            p.query_qps,
            p.recall_at_10 * 100.0,
        );
    }
    println!(
        "+---------+------------+-----------+-------------+------------+-------------+--------------+-------------+--------------------+-----------+------------+"
    );
}

pub(crate) fn load_or_default_baseline_profiles(_data_dir: &Path) -> Vec<BaselineScaleProfile> {
    vec![
        BaselineScaleProfile {
            measured_this_run: false,
            scale_vectors: 10_000,
            build_time_s: 1.482878,
            build_qps: 6743.64,
            peak_rss_mb: 13.64,
            bytes_per_vector: 342.98,
            raw_vector_bytes_per_vector: 512.00,
            graph_bytes_per_vector: 342.98,
            graph_only_bytes_per_vector: 338.98,
            total_bytes_per_vector: 854.98,
            graph_memory_bytes: 3_429_760,
            graph_only_memory_bytes: 3_389_760,
            graph_memory_mb: 3.27,
            query_latency_ms: 0.062140,
            query_qps: 16062.58,
            k: 10,
            ef_search: 50,
            recall_at_10: 0.997500,
            max_layer: 3,
            entry_point: 3015,
            latency_min_ms: 0.027000,
            latency_p50_ms: 0.063200,
            latency_p95_ms: 0.078000,
            latency_p99_ms: 0.090408,
            latency_max_ms: 0.182800,
            graph_fingerprint: None,
        },
        BaselineScaleProfile {
            measured_this_run: false,
            scale_vectors: 100_000,
            build_time_s: 28.623980,
            build_qps: 3493.57,
            peak_rss_mb: 88.60,
            bytes_per_vector: 342.66,
            raw_vector_bytes_per_vector: 512.00,
            graph_bytes_per_vector: 342.66,
            graph_only_bytes_per_vector: 338.66,
            total_bytes_per_vector: 854.66,
            graph_memory_bytes: 34_266_240,
            graph_only_memory_bytes: 33_866_240,
            graph_memory_mb: 32.68,
            query_latency_ms: 0.155355,
            query_qps: 6426.16,
            k: 10,
            ef_search: 50,
            recall_at_10: 0.981400,
            max_layer: 4,
            entry_point: 56204,
            latency_min_ms: 0.041500,
            latency_p50_ms: 0.158800,
            latency_p95_ms: 0.196000,
            latency_p99_ms: 0.224216,
            latency_max_ms: 0.367500,
            graph_fingerprint: None,
        },
        BaselineScaleProfile {
            measured_this_run: false,
            scale_vectors: 1_000_000,
            build_time_s: 472.189209,
            build_qps: 2117.80,
            peak_rss_mb: 845.77,
            bytes_per_vector: 342.67,
            raw_vector_bytes_per_vector: 512.00,
            graph_bytes_per_vector: 342.67,
            graph_only_bytes_per_vector: 338.67,
            total_bytes_per_vector: 854.67,
            graph_memory_bytes: 342_668_320,
            graph_only_memory_bytes: 338_668_320,
            graph_memory_mb: 326.79,
            query_latency_ms: 0.249254,
            query_qps: 4009.49,
            k: 10,
            ef_search: 50,
            recall_at_10: 0.947030,
            max_layer: 5,
            entry_point: 829262,
            latency_min_ms: 0.084700,
            latency_p50_ms: 0.254800,
            latency_p95_ms: 0.307405,
            latency_p99_ms: 0.342101,
            latency_max_ms: 2.842400,
            graph_fingerprint: None,
        },
    ]
}
