//! Phase 5 Filtered Search CLI, Strategy 3 Cost Model Microbenchmark (`T3.6`),
//! 100k Pilot Runner (`T3.7`), and Matrix CLI Dispatcher (`T3.8 / T3.9`).

#![forbid(unsafe_code)]

use std::collections::{BinaryHeap, HashSet};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::benchmark::{detect_provenance, ReportProvenance};
use crate::cli_util::{next_str, parse_csv_list, parse_next};
use crate::dataset::{
    generate_mode_a_bitset, generate_mode_b_bitset_detailed, load_or_download_sift_with_options,
    read_ivecs, SiftPreset, S_CLUST_DEFAULT,
};
use crate::distance::{l2_squared, Metric};
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::FilteredExactKnn;
use crate::filter::{DenseBitset, FilteredSearchOutput, FilteredSearchParams, LambdaPredicate};
use crate::hardware::{detect_hardware, get_peak_rss_mb, HardwareInfo};
use crate::hnsw::{
    Candidate, FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex, VisitedTracker,
};
use crate::json_util::{escape_json, write_file_with_parents};
use crate::matrix::RowMatrix;
use crate::recall::{
    compute_latency_stats, compute_recall_at_k, median_min_max, percentile_sorted,
    summarize_repeated_latencies, LatencyStats,
};
use crate::rng::DeterministicRng;

use super::runner::run_phase5_matrix;
use super::{MetadataMode, Phase5MatrixConfig, SELECTIVITY_LADDER_12, SMOKE_SELECTIVITY_LADDER};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase5RunMode {
    Microbench,
    Pilot,
    Matrix,
    Help,
}

#[derive(Debug, Clone)]
struct CliConfig {
    mode: Phase5RunMode,
    output_path: Option<PathBuf>,
    output_dir: PathBuf,
    output_dir_explicit: bool,
    data_dir: PathBuf,
    scale: usize,
    scale_explicit: bool,
    queries: usize,
    queries_explicit: bool,
    slow_probe_queries: usize,
    uncapped_probe_queries: usize,
    dim: usize,
    k: usize,
    reps: usize,
    num_words: usize,
    num_dists: usize,
    num_heap_ops: usize,
    modes: Vec<MetadataMode>,
    sigmas: Vec<f64>,
    sigmas_explicit: bool,
    smoke: bool,
    synthetic: bool,
    offline: bool,
    resume: bool,
    include_sensitivity: bool,
    sensitivity_only: bool,
    include_1m_opaque_spotcheck: bool,
    opaque_1m_spotcheck_only: bool,
    gt_threads: usize,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            mode: Phase5RunMode::Microbench,
            output_path: None,
            output_dir: PathBuf::from("results/phase5"),
            output_dir_explicit: false,
            data_dir: PathBuf::from("data"),
            scale: 100_000,
            scale_explicit: false,
            queries: 10_000,
            queries_explicit: false,
            slow_probe_queries: 500,
            uncapped_probe_queries: 200,
            dim: 128,
            k: 10,
            reps: 5,
            num_words: 500_000,
            num_dists: 50_000,
            num_heap_ops: 50_000,
            modes: vec![MetadataMode::Independent, MetadataMode::Correlated],
            sigmas: SELECTIVITY_LADDER_12.to_vec(),
            sigmas_explicit: false,
            smoke: false,
            synthetic: false,
            offline: false,
            resume: true,
            include_sensitivity: false,
            sensitivity_only: false,
            include_1m_opaque_spotcheck: false,
            opaque_1m_spotcheck_only: false,
            gt_threads: 0,
        }
    }
}

fn print_usage() {
    println!(
        r#"Phase 5 Filtered Vector Search CLI (Milestone M3 Tasks T3.6 / T3.7 / T3.8 / T3.9)

USAGE:
    phase5 [OPTIONS]

OPTIONS:
    --microbench, -m    Run the Strategy 3 cost model microbenchmark (default).
    --pilot             Run the T3.7 100k pilot benchmark suite across Strategies 1-3.
    --matrix            Run the Option C resumable Phase 5 matrix runner (T3.8 / T3.9).
    --smoke             Run fast 10k smoke matrix (defaults: scale=10000, 4 tiers,
                        output-dir=results/phase5/smoke).
    --output, -o <PATH> Output path for structured JSON report (microbench / pilot).
    --output-dir <PATH> Output directory for per-cell & summary JSONs (default: results/phase5).
    --data-dir <PATH>   Dataset & Phase 5 versioned cache directory (default: data).
    --scale <USIZE>     Base vector scale N (default: 100000, or 10000 for --smoke).
    --modes <CSV>       Comma-separated metadata modes: independent,correlated.
    --sigmas <CSV>      Comma-separated selectivity values (e.g. 0.5,0.1,0.01).
    --queries <USIZE>   Query count override Q (default: full dataset query count / Option C).
    --synthetic         Use offline deterministic synthetic vectors instead of SIFT files.
    --offline           Disallow automatic SIFT dataset download if missing locally.
    --resume            Skip cells whose clean-commit JSON matches current config (default).
    --no-resume         Force re-execution of all requested cells.
    --include-sensitivity
                        Also run H_max, overfetch_factor, and opaque eval_limit sensitivity.
    --sensitivity-only  Run only the sensitivity suite and write sensitivity_n<N>.json.
    --include-1m-opaque-spotcheck
                        Include the 1M opaque LambdaPredicate eval_limit spot check in sensitivity.
    --1m-opaque-spotcheck-only
                        Execute only the 1M opaque spot check (writes opaque_spotcheck_1m.json).
    --gt-threads <USIZE> Threads for exact filtered GT cache misses (0 = all cores; default: 0).
    --slow-probe-queries <USIZE>
                        Pilot query count for slow HNSW arms at sigma <= 0.01 (default: 500).
    --uncapped-probe-queries <USIZE>
                        Pilot query count for Strategy 1 uncapped probes at sigma <= 0.01 (default: 200).
    --dim <USIZE>       Vector dimensionality for microbenchmark or --synthetic (default: 128).
    --k <USIZE>         Top-k neighbor count (default: 10).
    --reps <USIZE>      Timed repetitions after 1 warmup (default: 5).
    --words <USIZE>     Word count for bitset scan microbenchmark (default: 500000).
    --dists <USIZE>     Evaluations count for distance microbenchmark (default: 50000).
    --heap-ops <USIZE>  Evaluations count for heap microbenchmark (default: 50000).
    --help, -h          Print this help message."#
    );
}

fn parse_cli_args(args: &[String]) -> Result<CliConfig> {
    let mut config = CliConfig::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--microbench" | "-m" => {
                config.mode = Phase5RunMode::Microbench;
                i += 1;
            }
            "--pilot" => {
                config.mode = Phase5RunMode::Pilot;
                i += 1;
            }
            "--matrix" => {
                config.mode = Phase5RunMode::Matrix;
                i += 1;
            }
            "--smoke" => {
                config.mode = Phase5RunMode::Matrix;
                config.smoke = true;
                i += 1;
            }
            "--output" | "-o" => {
                config.output_path = Some(PathBuf::from(next_str(args, &mut i, "--output")?));
            }
            "--output-dir" => {
                config.output_dir = PathBuf::from(next_str(args, &mut i, "--output-dir")?);
                config.output_dir_explicit = true;
            }
            "--data-dir" => {
                config.data_dir = PathBuf::from(next_str(args, &mut i, "--data-dir")?);
            }
            "--scale" => {
                config.scale = parse_next(args, &mut i, "--scale")?;
                config.scale_explicit = true;
            }
            "--modes" => {
                let raw = next_str(args, &mut i, "--modes")?;
                let mut parsed = Vec::new();
                for part in raw.split(',').filter(|p| !p.trim().is_empty()) {
                    let m = MetadataMode::parse(part)?;
                    if !parsed.contains(&m) {
                        parsed.push(m);
                    }
                }
                if parsed.is_empty() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--modes list cannot be empty".to_string(),
                    ));
                }
                config.modes = parsed;
            }
            "--sigmas" => {
                let raw = next_str(args, &mut i, "--sigmas")?;
                let parsed: Vec<f64> = parse_csv_list(raw, "--sigmas")?;
                for &s in &parsed {
                    if !s.is_finite() || s <= 0.0 || s > 1.0 {
                        return Err(VectorSearchError::InvalidArgument(format!(
                            "Sigma must lie in (0.0, 1.0], got {s}"
                        )));
                    }
                }
                config.sigmas = parsed;
                config.sigmas_explicit = true;
            }
            "--queries" => {
                config.queries = parse_next(args, &mut i, "--queries")?;
                config.queries_explicit = true;
            }
            "--synthetic" => {
                config.synthetic = true;
                i += 1;
            }
            "--offline" => {
                config.offline = true;
                i += 1;
            }
            "--resume" => {
                config.resume = true;
                i += 1;
            }
            "--no-resume" => {
                config.resume = false;
                i += 1;
            }
            "--include-sensitivity" => {
                config.include_sensitivity = true;
                i += 1;
            }
            "--sensitivity-only" => {
                config.mode = Phase5RunMode::Matrix;
                config.sensitivity_only = true;
                config.include_sensitivity = true;
                i += 1;
            }
            "--include-1m-opaque-spotcheck" => {
                config.include_1m_opaque_spotcheck = true;
                i += 1;
            }
            "--1m-opaque-spotcheck-only" => {
                config.mode = Phase5RunMode::Matrix;
                config.opaque_1m_spotcheck_only = true;
                config.include_1m_opaque_spotcheck = true;
                i += 1;
            }
            "--gt-threads" => {
                config.gt_threads = parse_next(args, &mut i, "--gt-threads")?;
            }
            "--slow-probe-queries" => {
                config.slow_probe_queries = parse_next(args, &mut i, "--slow-probe-queries")?;
            }
            "--uncapped-probe-queries" => {
                config.uncapped_probe_queries =
                    parse_next(args, &mut i, "--uncapped-probe-queries")?;
            }
            "--dim" => config.dim = parse_next(args, &mut i, "--dim")?,
            "--k" => config.k = parse_next(args, &mut i, "--k")?,
            "--reps" => config.reps = parse_next(args, &mut i, "--reps")?,
            "--words" => config.num_words = parse_next(args, &mut i, "--words")?,
            "--dists" => config.num_dists = parse_next(args, &mut i, "--dists")?,
            "--heap-ops" => {
                config.num_heap_ops = parse_next(args, &mut i, "--heap-ops")?;
            }
            "--help" | "-h" => {
                config.mode = Phase5RunMode::Help;
                return Ok(config);
            }
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized option: {other}"
                )));
            }
        }
    }

    if config.smoke {
        if !config.scale_explicit {
            config.scale = 10_000;
        }
        if !config.sigmas_explicit {
            config.sigmas = SMOKE_SELECTIVITY_LADDER.to_vec();
        }
        if !config.output_dir_explicit {
            config.output_dir = PathBuf::from("results/phase5/smoke");
        }
    }

    Ok(config)
}

/// Measures the average nanoseconds required to scan one 64-bit bitset word ($t_{\text{word}}$).
#[must_use]
pub fn measure_t_word(num_words: usize, reps: usize) -> f64 {
    let mut words = vec![0_u64; num_words];
    let mut rng = DeterministicRng::new(0x1234_5678);
    for w in words.iter_mut() {
        if rng.next_u64().is_multiple_of(100) {
            *w = rng.next_u64();
        }
    }

    // 1 untimed warmup run
    let mut sum: u64 = 0;
    for &w in std::hint::black_box(words.as_slice()) {
        sum ^= w;
    }
    std::hint::black_box(sum);

    // Timed repetitions
    let mut latencies_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let mut s: u64 = 0;
        for &w in std::hint::black_box(words.as_slice()) {
            s ^= w;
        }
        std::hint::black_box(s);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        latencies_ns.push(elapsed_ns / num_words as f64);
    }

    median_min_max(&latencies_ns).0
}

/// Measures the average nanoseconds per vector distance computation ($t_{\text{dist}}(d)$).
#[must_use]
pub fn measure_t_dist(dim: usize, num_evals: usize, reps: usize) -> f64 {
    let mut rng = DeterministicRng::new(0x8765_4321);
    let mut q = vec![0.0_f32; dim];
    for v in q.iter_mut() {
        *v = (rng.next_u64() % 1000) as f32 / 100.0;
    }
    let num_cached_vectors = 64;
    let mut db_vecs = vec![0.0_f32; dim * num_cached_vectors];
    for v in db_vecs.iter_mut() {
        *v = (rng.next_u64() % 1000) as f32 / 100.0;
    }

    // 1 untimed warmup run
    let q_slice = std::hint::black_box(q.as_slice());
    let db_slice = std::hint::black_box(db_vecs.as_slice());
    let mut dist_sum = 0.0_f32;
    for i in 0..num_evals {
        let offset = (i % num_cached_vectors) * dim;
        let v = &db_slice[offset..offset + dim];
        dist_sum += l2_squared(q_slice, v);
    }
    std::hint::black_box(dist_sum);

    // Timed repetitions
    let mut latencies_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let q_ref = std::hint::black_box(q.as_slice());
        let db_ref = std::hint::black_box(db_vecs.as_slice());
        let start = Instant::now();
        let mut acc = 0.0_f32;
        for i in 0..num_evals {
            let offset = (i % num_cached_vectors) * dim;
            let v = &db_ref[offset..offset + dim];
            acc += l2_squared(q_ref, v);
        }
        std::hint::black_box(acc);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        latencies_ns.push(elapsed_ns / num_evals as f64);
    }

    median_min_max(&latencies_ns).0
}

/// Measures the average nanoseconds per bounded heap operation ($t_{\text{heap}}(k)$).
#[must_use]
pub fn measure_t_heap(k: usize, num_ops: usize, reps: usize) -> f64 {
    let mut rng = DeterministicRng::new(0xABCD_EF01);
    let candidates: Vec<Candidate> = (0..num_ops)
        .map(|i| Candidate {
            distance: (rng.next_u64() % 100_000) as f32 / 10.0,
            id: i,
        })
        .collect();

    // 1 untimed warmup run
    let mut heap = BinaryHeap::<Candidate>::with_capacity(k);
    for &cand in std::hint::black_box(candidates.as_slice()) {
        if heap.len() < k {
            heap.push(cand);
        } else if let Some(mut worst) = heap.peek_mut() {
            if cand < *worst {
                *worst = cand;
            }
        }
    }
    std::hint::black_box(&heap);

    // Timed repetitions
    let mut latencies_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut h = BinaryHeap::<Candidate>::with_capacity(k);
        let cand_slice = std::hint::black_box(candidates.as_slice());
        let start = Instant::now();
        for &cand in cand_slice {
            if h.len() < k {
                h.push(cand);
            } else if let Some(mut worst) = h.peek_mut() {
                if cand < *worst {
                    *worst = cand;
                }
            }
        }
        std::hint::black_box(&h);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        latencies_ns.push(elapsed_ns / num_ops as f64);
    }

    median_min_max(&latencies_ns).0
}

/// Evaluates the theoretical brute-force cost model:
/// $$C_{\text{brute}}(\sigma, N, d) = \frac{N}{64} t_{\text{word}} + \sigma N (t_{\text{dist}}(d) + t_{\text{heap}}(k))$$
#[must_use]
pub fn predict_c_brute_ns(
    sigma: f64,
    n: usize,
    t_word_ns: f64,
    t_dist_ns: f64,
    t_heap_ns: f64,
) -> f64 {
    let t_scan = (n as f64 / 64.0) * t_word_ns;
    let t_cand = t_dist_ns + t_heap_ns;
    t_scan + sigma * (n as f64) * t_cand
}

fn format_environment_json(hw: &HardwareInfo, prov: &ReportProvenance) -> String {
    format!(
        r#"  "environment": {{
    "os": "{}",
    "arch": "{}",
    "cpu_model": "{}",
    "logical_cores": {},
    "total_ram_gb": {},
    "power_mode": "{}",
    "processor_affinity_mask": "{}"
  }}"#,
        escape_json(&hw.os),
        escape_json(&hw.arch),
        escape_json(&hw.cpu_model),
        hw.logical_cores,
        hw.total_ram_gb
            .map_or("null".to_string(), |r| format!("{r:.2}")),
        escape_json(&prov.power_mode),
        escape_json(&prov.processor_affinity_mask)
    )
}

fn build_json_report(config: &CliConfig, t_word_ns: f64, t_dist_ns: f64, t_heap_ns: f64) -> String {
    let hw = detect_hardware();
    let prov = detect_provenance();
    let t_cand_ns = t_dist_ns + t_heap_ns;

    let test_scales = [10_000_usize, 100_000, 1_000_000];
    let test_sigmas = [0.0005_f64, 0.001, 0.01, 0.1, 0.5];

    let mut predictions_json = String::new();
    for (si, &n) in test_scales.iter().enumerate() {
        for (sj, &sigma) in test_sigmas.iter().enumerate() {
            let cost = predict_c_brute_ns(sigma, n, t_word_ns, t_dist_ns, t_heap_ns);
            let is_last = si == test_scales.len() - 1 && sj == test_sigmas.len() - 1;
            let key = format!("n{}_sigma{:.4}", n, sigma);
            predictions_json.push_str(&format!(
                "      \"{}\": {:.4}{}\n",
                key,
                cost,
                if is_last { "" } else { "," }
            ));
        }
    }

    let env_block = format_environment_json(&hw, &prov);

    format!(
        r#"{{
  "benchmark": "cost_microbenchmark",
  "task": "Milestone M3 Task T3.6",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "measured_this_run": true,
{},
  "hardware": {{
    "cpu_model": "{}",
    "logical_cores": {},
    "total_ram_gb": {},
    "os": "{}",
    "arch": "{}",
    "processor_affinity_mask": "{}"
  }},
  "parameters": {{
    "dim": {},
    "k": {},
    "reps": {},
    "num_words": {},
    "num_dists": {},
    "num_heap_ops": {}
  }},
  "measurements": {{
    "t_word_ns": {:.6},
    "t_dist_ns": {:.6},
    "t_heap_ns": {:.6},
    "t_cand_ns": {:.6}
  }},
  "theoretical_model": {{
    "formula": "C_brute(sigma, N, d) = (N / 64) * t_word + sigma * N * (t_dist(d) + t_heap(k))",
    "t_scan_formula": "(N / 64) * t_word",
    "t_cand_formula": "t_dist(d) + t_heap(k)",
    "predicted_c_brute_ns": {{
{}    }}
  }}
}}
"#,
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        env_block,
        escape_json(&hw.cpu_model),
        hw.logical_cores,
        hw.total_ram_gb
            .map_or("null".to_string(), |r| format!("{r:.1}")),
        escape_json(&hw.os),
        escape_json(&hw.arch),
        escape_json(&prov.processor_affinity_mask),
        config.dim,
        config.k,
        config.reps,
        config.num_words,
        config.num_dists,
        config.num_heap_ops,
        t_word_ns,
        t_dist_ns,
        t_heap_ns,
        t_cand_ns,
        predictions_json
    )
}

/// Computes per-query average filtered `Recall@k` for the T3.7 pilot suite, skipping queries
/// with zero valid ground-truth neighbors (`denom = k.min(gt_set.len())`).
fn compute_per_query_filtered_recall(
    outputs: &[FilteredSearchOutput],
    gt_indices: &RowMatrix<i32>,
    k: usize,
) -> f64 {
    let q_count = outputs.len().min(gt_indices.rows());
    if q_count == 0 {
        return 1.0;
    }
    let mut recall_sum = 0.0_f64;
    let mut valid_q = 0_usize;
    let mut gt_set = HashSet::with_capacity(k * 2);
    let mut cand_seen = HashSet::with_capacity(k * 2);

    for (qi, out) in outputs.iter().enumerate().take(q_count) {
        gt_set.clear();
        cand_seen.clear();
        for &id in &gt_indices.row(qi)[..k.min(gt_indices.cols())] {
            if id >= 0 {
                gt_set.insert(id as usize);
            }
        }
        let denom = k.min(gt_set.len());
        if denom == 0 {
            continue;
        }
        valid_q += 1;
        let mut hits = 0_usize;
        for &(_, id) in out.results.iter().take(k) {
            if cand_seen.insert(id) && gt_set.contains(&id) {
                hits += 1;
            }
        }
        recall_sum += (hits as f64) / (denom as f64);
    }

    if valid_q == 0 {
        1.0
    } else {
        recall_sum / (valid_q as f64)
    }
}

struct PilotArmRecord {
    mode: String,
    sigma: f64,
    strategy_label: String,
    ef_search: usize,
    overfetch_cap: usize,
    num_queries: usize,
    is_subquery_probe: bool,
    recall_at_k: f64,
    fewer_than_k_rate: f64,
    cap_reached_rate: f64,
    hop_exhausted_rate: f64,
    eval_limit_rate: f64,
    median_total_s: f64,
    median_qps: f64,
    median_mean_ms: f64,
    median_p50_ms: f64,
    median_p99_ms: f64,
    mean_ms_spread: f64,
    p99_ms_spread: f64,
}

#[allow(clippy::too_many_arguments)]
fn measure_filtered_arm(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    num_queries: usize,
    k: usize,
    bitset: &DenseBitset,
    gt_indices: &RowMatrix<i32>,
    strategy: FilterStrategy,
    strategy_label: &str,
    mode_label: &str,
    sigma: f64,
    params: &FilteredSearchParams,
    reps: usize,
    scratch: &mut FilteredSearchScratch,
    is_subquery_probe: bool,
) -> Result<PilotArmRecord> {
    let take_q = num_queries.min(queries.rows());
    let mut last_outputs = Vec::with_capacity(take_q);

    // 1 untimed warmup pass
    for qi in 0..take_q {
        let out = index.search_knn_filtered_with_scratch(
            queries.row(qi),
            k,
            bitset,
            strategy,
            params,
            scratch,
        )?;
        std::hint::black_box(&out.results);
    }

    let mut run_stats: Vec<LatencyStats> = Vec::with_capacity(reps);
    let mut per_q_ms: Vec<f64> = Vec::with_capacity(take_q);

    for r_idx in 0..reps {
        per_q_ms.clear();
        if r_idx + 1 == reps {
            last_outputs.clear();
        }
        let batch_start = Instant::now();
        for qi in 0..take_q {
            let q_start = Instant::now();
            let out = index.search_knn_filtered_with_scratch(
                queries.row(qi),
                k,
                bitset,
                strategy,
                params,
                scratch,
            )?;
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&out.results);
            per_q_ms.push(elapsed_ms);
            if r_idx + 1 == reps {
                last_outputs.push(out);
            }
        }
        let total_s = batch_start.elapsed().as_secs_f64();
        run_stats.push(compute_latency_stats(&per_q_ms, total_s));
    }

    let summary = summarize_repeated_latencies(&run_stats);
    let recall_at_k = compute_per_query_filtered_recall(&last_outputs, gt_indices, k);

    let mut fewer = 0_usize;
    let mut cap_hit = 0_usize;
    let mut hop_hit = 0_usize;
    let mut eval_hit = 0_usize;
    for out in &last_outputs {
        if out.results.len() < k {
            fewer += 1;
        }
        if out.flags.post_filter_cap_reached {
            cap_hit += 1;
        }
        if out.flags.hop_budget_exhausted {
            hop_hit += 1;
        }
        if out.flags.eval_limit_reached {
            eval_hit += 1;
        }
    }
    let denom = take_q.max(1) as f64;

    Ok(PilotArmRecord {
        mode: mode_label.to_string(),
        sigma,
        strategy_label: strategy_label.to_string(),
        ef_search: params.ef_search,
        overfetch_cap: params.overfetch_cap,
        num_queries: take_q,
        is_subquery_probe,
        recall_at_k,
        fewer_than_k_rate: (fewer as f64) / denom,
        cap_reached_rate: (cap_hit as f64) / denom,
        hop_exhausted_rate: (hop_hit as f64) / denom,
        eval_limit_rate: (eval_hit as f64) / denom,
        median_total_s: summary.median_latency.total_s,
        median_qps: summary.median_latency.qps,
        median_mean_ms: summary.median_latency.mean_ms,
        median_p50_ms: summary.median_latency.p50_ms,
        median_p99_ms: summary.median_latency.p99_ms,
        mean_ms_spread: summary.mean_ms_spread,
        p99_ms_spread: summary.p99_ms_spread,
    })
}

struct MetadataCostRecord {
    mode: String,
    sigma: f64,
    gen_time_s: f64,
    gt_gen_time_s: f64,
    bitset_bytes: usize,
    gt_bytes: usize,
    actual_popcount: usize,
    centroid_count: usize,
    shortfall_count: usize,
}

struct EvalProbeRecord {
    mode: String,
    sigma: f64,
    strategy_label: String,
    ef_search: usize,
    num_queries: usize,
    mean_evals: f64,
    p50_evals: f64,
    p95_evals: f64,
    p99_evals: f64,
    max_evals: usize,
    max_to_ef_ratio: f64,
}

struct EvalProbeSpec<'a> {
    probe_q: usize,
    k: usize,
    mode_label: &'a str,
    sigma: f64,
    strategy: FilterStrategy,
    strategy_label: &'a str,
    params: &'a FilteredSearchParams,
}

fn measure_eval_probe(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    bitset: &DenseBitset,
    spec: &EvalProbeSpec<'_>,
) -> Result<EvalProbeRecord> {
    let take_q = spec.probe_q.min(queries.rows());
    let mut tracker = VisitedTracker::new(index.num_vectors());
    let mut counts = Vec::with_capacity(take_q);
    for qi in 0..take_q {
        let counter = AtomicUsize::new(0);
        let pred = LambdaPredicate::new(
            |id| {
                counter.fetch_add(1, Ordering::Relaxed);
                bitset.get_bit(id as usize)
            },
            spec.sigma,
        );
        let _ = index.search_knn_filtered_with_tracker(
            queries.row(qi),
            spec.k,
            &pred,
            spec.strategy,
            spec.params,
            &mut tracker,
        )?;
        counts.push(counter.load(Ordering::Relaxed) as f64);
    }
    counts.sort_by(f64::total_cmp);
    let sum: f64 = counts.iter().sum();
    let max_evals = counts.last().copied().unwrap_or(0.0) as usize;
    Ok(EvalProbeRecord {
        mode: spec.mode_label.to_string(),
        sigma: spec.sigma,
        strategy_label: spec.strategy_label.to_string(),
        ef_search: spec.params.ef_search,
        num_queries: take_q,
        mean_evals: sum / (take_q.max(1) as f64),
        p50_evals: percentile_sorted(&counts, 50.0),
        p95_evals: percentile_sorted(&counts, 95.0),
        p99_evals: percentile_sorted(&counts, 99.0),
        max_evals,
        max_to_ef_ratio: (max_evals as f64) / (spec.params.ef_search.max(1) as f64),
    })
}

fn run_pilot(config: &CliConfig) -> Result<()> {
    let total_t0 = Instant::now();
    let hw = detect_hardware();
    let prov = detect_provenance();
    let out_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("results/phase5/pilot/pilot_100k.json"));

    eprintln!(
        "[INFO] T3.7 Phase 5 100k Pilot starting (scale={}, queries={}, k={}, reps={}, affinity={})",
        config.scale, config.queries, config.k, config.reps, prov.processor_affinity_mask
    );

    let loaded = load_or_download_sift_with_options(
        SiftPreset::Sift1M,
        &config.data_dir,
        false,
        42,
        Some(config.scale),
        Some(config.queries),
        false,
    )?;
    if !loaded.source.is_authentic_sift() {
        return Err(VectorSearchError::InvalidArgument(
            "Authentic SIFT1M dataset is required under --data-dir for the 100k pilot".to_string(),
        ));
    }

    let num_base = loaded.base.rows();
    let num_queries = loaded.query.rows();
    eprintln!(
        "[INFO] Loaded authentic SIFT1M prefix: N={}, Q={}, D={} in {:.3} s",
        num_base,
        num_queries,
        loaded.base.cols(),
        loaded.load_time_s
    );

    let hnsw_cfg = HnswConfig::new(16, 100);
    let build_start = Instant::now();
    let index = HnswIndex::build(&loaded.base, Metric::L2Squared, hnsw_cfg.clone())?;
    let build_time_s = build_start.elapsed().as_secs_f64();
    let fp = index.graph_fingerprint();
    let graph_only_bpv = index.graph_only_bytes_per_vector();
    eprintln!(
        "[INFO] Built HNSW (M=16, efC=100) in {:.3} s | fingerprint=0x{fp:016x} | graph={graph_only_bpv:.2} B/vec",
        build_time_s
    );

    // Unfiltered sanity check against cached 100k exact GT if present
    let unfiltered_gt_path = config.data_dir.join(format!("gt_{num_base}_k100.ivecs"));
    let mut unfiltered_recall_ef100: Option<f64> = None;
    let mut unfiltered_qps_ef100: Option<f64> = None;
    if unfiltered_gt_path.is_file() {
        let gt_mat = read_ivecs(&unfiltered_gt_path)?;
        if gt_mat.rows() == num_queries {
            let mut t = VisitedTracker::new(num_base);
            let _ = index.search_knn_with_tracker(loaded.query.row(0), config.k, 100, &mut t)?;
            let timed =
                index.search_batch_timed_with_tracker(&loaded.query, config.k, 100, &mut t)?;
            let rec = compute_recall_at_k(&timed.result.indices, &gt_mat, config.k)?;
            unfiltered_recall_ef100 = Some(rec);
            unfiltered_qps_ef100 = Some(timed.latency.qps);
            eprintln!(
                "[INFO] Unfiltered ef=100 check: Recall@{}={:.6}, QPS={:.2}",
                config.k, rec, timed.latency.qps
            );
        } else {
            eprintln!(
                "[WARN] Unfiltered GT rows ({}) != queries ({}); skipping unfiltered recall check",
                gt_mat.rows(),
                num_queries
            );
        }
    }

    // Prepare reusable scratches and exact filtered GT engine
    let exact_engine = FilteredExactKnn::from_base(Metric::L2Squared, loaded.base.clone())?;
    let mut scratch_s1_cap = FilteredSearchScratch::new(num_base, 4096, 4096);
    let mut scratch_s1_uncap = FilteredSearchScratch::new(num_base, num_base, num_base);
    let mut scratch_s2_s3 = FilteredSearchScratch::new(num_base, 600, 0);

    let pilot_sigmas = [0.0005_f64, 0.001, 0.01, 0.10, 0.50];
    let ef_s2_ladder = [20_usize, 50, 100, 200, 400, 600];
    let ef_s1_high_sigma = [20_usize, 100, 600];
    let ef_s1_low_sigma = [100_usize];

    let mut metadata_records: Vec<MetadataCostRecord> = Vec::new();
    let mut arm_records: Vec<PilotArmRecord> = Vec::new();
    let mut eval_probes: Vec<EvalProbeRecord> = Vec::new();

    for &mode_label in &["independent", "correlated"] {
        for &sigma in &pilot_sigmas {
            eprintln!("[INFO] --- Mode={mode_label}, sigma={sigma} ---");

            // 1. Metadata generation
            let meta_start = Instant::now();
            let (bitset, centroid_count, shortfall_count) = if mode_label == "independent" {
                (
                    generate_mode_a_bitset(num_base, sigma, 42),
                    0_usize,
                    0_usize,
                )
            } else {
                let meta =
                    generate_mode_b_bitset_detailed(&loaded.base, sigma, 42, S_CLUST_DEFAULT);
                let c_cnt = meta.centroid_count;
                let sf_cnt = meta.shortfall_count;
                (meta.bitset, c_cnt, sf_cnt)
            };
            let gen_time_s = meta_start.elapsed().as_secs_f64();
            let bitset_bytes = bitset.words().len() * 8 + 16;

            // 2. Filtered GT generation (all logical cores)
            let gt_start = Instant::now();
            let gt_batch =
                exact_engine.search_batch_parallel(&loaded.query, config.k, &bitset, 0)?;
            let gt_gen_time_s = gt_start.elapsed().as_secs_f64();
            let gt_bytes = num_queries * (4 + 4 * config.k);

            eprintln!(
                "[INFO] Metadata gen: {gen_time_s:.3} s ({bitset_bytes} B, pop={}) | Filtered GT gen (multi-core): {gt_gen_time_s:.3} s ({gt_bytes} B)",
                bitset.count_ones()
            );

            metadata_records.push(MetadataCostRecord {
                mode: mode_label.to_string(),
                sigma,
                gen_time_s,
                gt_gen_time_s,
                bitset_bytes,
                gt_bytes,
                actual_popcount: bitset.count_ones(),
                centroid_count,
                shortfall_count,
            });

            // 3. Strategy 3 (PreFilterBruteForce)
            let s3_params = FilteredSearchParams::default();
            let s3_rec = measure_filtered_arm(
                &index,
                &loaded.query,
                num_queries,
                config.k,
                &bitset,
                &gt_batch.indices,
                FilterStrategy::PreFilterBruteForce,
                "S3_pre_filter",
                mode_label,
                sigma,
                &s3_params,
                config.reps,
                &mut scratch_s2_s3,
                false,
            )?;
            eprintln!(
                "[INFO] S3_pre_filter: mean={:.4} ms, QPS={:.1}, Recall@10={:.6}, total={:.2} s",
                s3_rec.median_mean_ms, s3_rec.median_qps, s3_rec.recall_at_k, s3_rec.median_total_s
            );
            arm_records.push(s3_rec);

            let hnsw_q = if sigma <= 0.01 && config.slow_probe_queries > 0 {
                config.slow_probe_queries.min(num_queries)
            } else if sigma < 0.5 && config.slow_probe_queries > 0 {
                (config.slow_probe_queries * 2).min(num_queries)
            } else {
                num_queries
            };
            let hnsw_is_probe = hnsw_q < num_queries;

            // 4. Strategy 2 (InTraversal, all 6 efSearch points)
            for &ef in &ef_s2_ladder {
                let s2_params = FilteredSearchParams {
                    ef_search: ef,
                    max_bridge_hops: Some(3 * ef),
                    ..Default::default()
                };
                let s2_rec = measure_filtered_arm(
                    &index,
                    &loaded.query,
                    hnsw_q,
                    config.k,
                    &bitset,
                    &gt_batch.indices,
                    FilterStrategy::InTraversal,
                    "S2_in_traversal",
                    mode_label,
                    sigma,
                    &s2_params,
                    config.reps,
                    &mut scratch_s2_s3,
                    hnsw_is_probe,
                )?;
                eprintln!(
                    "[INFO] S2 ef={ef:3} (q={hnsw_q}, probe={hnsw_is_probe}): mean={:.4} ms, QPS={:.1}, Recall@10={:.4}, fewer={:.3}%, hop_exh={:.3}%",
                    s2_rec.median_mean_ms,
                    s2_rec.median_qps,
                    s2_rec.recall_at_k,
                    s2_rec.fewer_than_k_rate * 100.0,
                    s2_rec.hop_exhausted_rate * 100.0
                );
                arm_records.push(s2_rec);
            }

            // 5. Strategy 1 Capped (overfetch_cap = 4096)
            let s1_efs = if sigma >= 0.05 {
                &ef_s1_high_sigma[..]
            } else {
                &ef_s1_low_sigma[..]
            };
            for &ef in s1_efs {
                let s1_cap_params = FilteredSearchParams {
                    ef_search: ef,
                    overfetch_cap: 4096,
                    ..Default::default()
                };
                let rec = measure_filtered_arm(
                    &index,
                    &loaded.query,
                    hnsw_q,
                    config.k,
                    &bitset,
                    &gt_batch.indices,
                    FilterStrategy::PostFilterOverfetch,
                    "S1_post_capped",
                    mode_label,
                    sigma,
                    &s1_cap_params,
                    config.reps,
                    &mut scratch_s1_cap,
                    hnsw_is_probe,
                )?;
                eprintln!(
                    "[INFO] S1_cap ef={ef:3} (q={hnsw_q}, probe={hnsw_is_probe}): mean={:.4} ms, QPS={:.1}, Recall@10={:.4}, fewer={:.3}%, cap={:.3}%",
                    rec.median_mean_ms,
                    rec.median_qps,
                    rec.recall_at_k,
                    rec.fewer_than_k_rate * 100.0,
                    rec.cap_reached_rate * 100.0
                );
                arm_records.push(rec);
            }

            // 6. Strategy 1 Uncapped (overfetch_cap = N)
            let uncap_q = if sigma <= 0.01 && config.uncapped_probe_queries > 0 {
                config.uncapped_probe_queries.min(num_queries)
            } else {
                hnsw_q
            };
            let is_probe = uncap_q < num_queries;
            let s1_uncap_params = FilteredSearchParams {
                ef_search: 100,
                overfetch_cap: num_base,
                ..Default::default()
            };
            let uncap_rec = measure_filtered_arm(
                &index,
                &loaded.query,
                uncap_q,
                config.k,
                &bitset,
                &gt_batch.indices,
                FilterStrategy::PostFilterOverfetch,
                "S1_post_uncapped",
                mode_label,
                sigma,
                &s1_uncap_params,
                config.reps,
                &mut scratch_s1_uncap,
                is_probe,
            )?;
            eprintln!(
                "[INFO] S1_uncap (q={uncap_q}, probe={is_probe}): mean={:.4} ms, QPS={:.1}, Recall@10={:.4}",
                uncap_rec.median_mean_ms, uncap_rec.median_qps, uncap_rec.recall_at_k
            );
            arm_records.push(uncap_rec);

            // 7. Opaque LambdaPredicate eval count profiling (first 1000 queries)
            for &ef in &[20_usize, 100, 600] {
                let ep_params = FilteredSearchParams {
                    ef_search: ef,
                    max_bridge_hops: Some(3 * ef),
                    ..Default::default()
                };
                let ep = measure_eval_probe(
                    &index,
                    &loaded.query,
                    &bitset,
                    &EvalProbeSpec {
                        probe_q: 1000,
                        k: config.k,
                        mode_label,
                        sigma,
                        strategy: FilterStrategy::InTraversal,
                        strategy_label: "S2_in_traversal",
                        params: &ep_params,
                    },
                )?;
                eval_probes.push(ep);
            }
            let s1_ep_params = FilteredSearchParams {
                ef_search: 100,
                overfetch_cap: 4096,
                ..Default::default()
            };
            let ep_s1 = measure_eval_probe(
                &index,
                &loaded.query,
                &bitset,
                &EvalProbeSpec {
                    probe_q: 1000,
                    k: config.k,
                    mode_label,
                    sigma,
                    strategy: FilterStrategy::PostFilterOverfetch,
                    strategy_label: "S1_post_capped",
                    params: &s1_ep_params,
                },
            )?;
            eval_probes.push(ep_s1);
        }
    }

    let peak_rss_mb = get_peak_rss_mb();
    let total_wall_s = total_t0.elapsed().as_secs_f64();

    // Format JSON report
    let mut meta_json = String::new();
    for (idx, m) in metadata_records.iter().enumerate() {
        let comma = if idx + 1 == metadata_records.len() {
            ""
        } else {
            ","
        };
        meta_json.push_str(&format!(
            "    {{\"mode\": \"{}\", \"sigma\": {:.4}, \"gen_time_s\": {:.6}, \"gt_gen_time_s\": {:.6}, \"bitset_bytes\": {}, \"gt_bytes\": {}, \"actual_popcount\": {}, \"centroid_count\": {}, \"shortfall_count\": {}}}{comma}\n",
            m.mode, m.sigma, m.gen_time_s, m.gt_gen_time_s, m.bitset_bytes, m.gt_bytes, m.actual_popcount, m.centroid_count, m.shortfall_count
        ));
    }

    let mut arms_json = String::new();
    for (idx, a) in arm_records.iter().enumerate() {
        let comma = if idx + 1 == arm_records.len() {
            ""
        } else {
            ","
        };
        arms_json.push_str(&format!(
            "    {{\"mode\": \"{}\", \"sigma\": {:.4}, \"strategy\": \"{}\", \"ef_search\": {}, \"overfetch_cap\": {}, \"num_queries\": {}, \"is_subquery_probe\": {}, \"recall_at_k\": {:.6}, \"fewer_than_k_rate\": {:.6}, \"post_filter_cap_reached_rate\": {:.6}, \"hop_budget_exhausted_rate\": {:.6}, \"eval_limit_reached_rate\": {:.6}, \"median_total_s\": {:.6}, \"median_qps\": {:.2}, \"median_mean_ms\": {:.6}, \"median_p50_ms\": {:.6}, \"median_p99_ms\": {:.6}, \"mean_ms_spread\": {:.6}, \"p99_ms_spread\": {:.6}}}{comma}\n",
            a.mode, a.sigma, a.strategy_label, a.ef_search, a.overfetch_cap, a.num_queries, a.is_subquery_probe, a.recall_at_k, a.fewer_than_k_rate, a.cap_reached_rate, a.hop_exhausted_rate, a.eval_limit_rate, a.median_total_s, a.median_qps, a.median_mean_ms, a.median_p50_ms, a.median_p99_ms, a.mean_ms_spread, a.p99_ms_spread
        ));
    }

    let mut probes_json = String::new();
    for (idx, p) in eval_probes.iter().enumerate() {
        let comma = if idx + 1 == eval_probes.len() {
            ""
        } else {
            ","
        };
        probes_json.push_str(&format!(
            "    {{\"mode\": \"{}\", \"sigma\": {:.4}, \"strategy\": \"{}\", \"ef_search\": {}, \"num_queries\": {}, \"mean_evals\": {:.2}, \"p50_evals\": {:.2}, \"p95_evals\": {:.2}, \"p99_evals\": {:.2}, \"max_evals\": {}, \"max_to_ef_ratio\": {:.3}}}{comma}\n",
            p.mode, p.sigma, p.strategy_label, p.ef_search, p.num_queries, p.mean_evals, p.p50_evals, p.p95_evals, p.p99_evals, p.max_evals, p.max_to_ef_ratio
        ));
    }

    let env_block = format_environment_json(&hw, &prov);
    let report = format!(
        r#"{{
  "benchmark": "phase5_t37_100k_pilot",
  "task": "Milestone M3 Task T3.7",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "measured_this_run": true,
  "warmup_performed": true,
  "reps": {},
{},
  "dataset": {{
    "preset": "sift1m_prefix",
    "is_authentic_sift": true,
    "num_base": {},
    "num_query": {},
    "dim": {},
    "k": {},
    "load_time_s": {:.6}
  }},
  "index_build": {{
    "m": {},
    "ef_construction": {},
    "seed": {},
    "build_time_s": {:.6},
    "build_qps": {:.2},
    "graph_fingerprint": {},
    "graph_fingerprint_hex": "0x{:016x}",
    "graph_only_bytes": {},
    "graph_only_bytes_per_vector": {:.2},
    "unfiltered_recall_ef100": {},
    "unfiltered_qps_ef100": {}
  }},
  "total_pilot_wall_s": {:.6},
  "peak_rss_mb": {},
  "metadata_and_gt_costs": [
{}  ],
  "pilot_results": [
{}  ],
  "opaque_eval_probes": [
{}  ]
}}
"#,
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        config.reps,
        env_block,
        num_base,
        num_queries,
        loaded.base.cols(),
        config.k,
        loaded.load_time_s,
        hnsw_cfg.m,
        hnsw_cfg.ef_construction,
        hnsw_cfg.seed,
        build_time_s,
        (num_base as f64) / build_time_s,
        fp,
        fp,
        index.graph_only_memory_bytes(),
        graph_only_bpv,
        unfiltered_recall_ef100.map_or("null".to_string(), |v| format!("{v:.6}")),
        unfiltered_qps_ef100.map_or("null".to_string(), |v| format!("{v:.2}")),
        total_wall_s,
        peak_rss_mb.map_or("null".to_string(), |v| format!("{v:.2}")),
        meta_json,
        arms_json,
        probes_json
    );

    write_file_with_parents(&out_path, &report)?;
    eprintln!(
        "[SUCCESS] Wrote T3.7 100k pilot JSON to {} (total wall {:.2} s)",
        out_path.display(),
        total_wall_s
    );
    Ok(())
}

fn run_microbenchmark(config: &CliConfig) -> Result<()> {
    eprintln!("[INFO] Running Cost Model Microbenchmarks (DESIGN.md §3.6, §4.1)...");
    eprintln!(
        "[INFO] Parameters: dim={}, k={}, reps={} (after 1 warmup)",
        config.dim, config.k, config.reps
    );

    let t_word_ns = measure_t_word(config.num_words, config.reps);
    eprintln!("[INFO] Measured t_word: {:.4} ns/word", t_word_ns);

    let t_dist_ns = measure_t_dist(config.dim, config.num_dists, config.reps);
    eprintln!(
        "[INFO] Measured t_dist(d={}): {:.4} ns/dist",
        config.dim, t_dist_ns
    );

    let t_heap_ns = measure_t_heap(config.k, config.num_heap_ops, config.reps);
    eprintln!(
        "[INFO] Measured t_heap(k={}): {:.4} ns/op",
        config.k, t_heap_ns
    );

    let t_cand_ns = t_dist_ns + t_heap_ns;
    eprintln!(
        "[INFO] Combined t_cand: {:.4} ns/matching_vector",
        t_cand_ns
    );

    let json_report = build_json_report(config, t_word_ns, t_dist_ns, t_heap_ns);

    if let Some(ref path) = config.output_path {
        if path.as_os_str() != "-" {
            write_file_with_parents(path, &json_report)?;
            eprintln!(
                "[INFO] Wrote cost microbenchmark report to {}",
                path.display()
            );
            return Ok(());
        }
    }

    println!("{json_report}");
    Ok(())
}

fn run_matrix_cli(config: &CliConfig) -> Result<()> {
    let matrix_cfg = Phase5MatrixConfig {
        data_dir: config.data_dir.clone(),
        output_dir: config.output_dir.clone(),
        scale: config.scale,
        modes: config.modes.clone(),
        sigmas: config.sigmas.clone(),
        k: config.k,
        reps: config.reps,
        synthetic: config.synthetic,
        synthetic_dim: config.dim,
        synthetic_queries: if config.queries_explicit {
            Some(config.queries)
        } else {
            None
        },
        allow_download: !config.offline,
        smoke: config.smoke,
        resume: config.resume,
        include_sensitivity: config.include_sensitivity,
        sensitivity_only: config.sensitivity_only,
        include_1m_opaque_spotcheck: config.include_1m_opaque_spotcheck,
        opaque_1m_spotcheck_only: config.opaque_1m_spotcheck_only,
        gt_threads: config.gt_threads,
        ..Default::default()
    };

    let summary = run_phase5_matrix(&matrix_cfg)?;
    eprintln!(
        "[DONE] Phase 5 matrix: scale={}, total_cells={}, newly_measured={}, resumed={}, summary={}",
        summary.scale,
        summary.total_cells,
        summary.newly_measured_count,
        summary.resumed_skipped_count,
        summary.summary_path.display()
    );
    Ok(())
}

/// Executes the Phase 5 CLI from `std::env::args()` and returns a process [`ExitCode`].
#[must_use]
pub fn run_phase5_cli() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let config = match parse_cli_args(&args) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let res = match config.mode {
        Phase5RunMode::Help => {
            print_usage();
            Ok(())
        }
        Phase5RunMode::Microbench => run_microbenchmark(&config),
        Phase5RunMode::Pilot => run_pilot(&config),
        Phase5RunMode::Matrix => run_matrix_cli(&config),
    };

    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            ExitCode::FAILURE
        }
    }
}
