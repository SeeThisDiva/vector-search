//! Phase 5 Filtered Search CLI & Cost Microbenchmark Runner (Milestone M3 Task T3.6).
//!
//! Emits structured JSON reporting measured cost model parameters:
//! - `t_word`: time in nanoseconds per 64-bit word scan
//! - `t_dist(d)`: time in nanoseconds per distance computation (d = 128)
//! - `t_heap(k)`: time in nanoseconds per heap update (k = 10)
//!
//! Theoretical cost model:
//! $$C_{\text{brute}}(\sigma, N, d) = \frac{N}{64} t_{\text{word}} + \sigma N (t_{\text{dist}}(d) + t_{\text{heap}}(k))$$

#![forbid(unsafe_code)]

use std::collections::BinaryHeap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::Instant;

use vector_search::distance::l2_squared;
use vector_search::error::{Result, VectorSearchError};
use vector_search::hardware::detect_hardware;
use vector_search::hnsw::Candidate;
use vector_search::rng::DeterministicRng;

#[derive(Debug, Clone)]
struct CliConfig {
    run_microbench: bool,
    output_path: Option<PathBuf>,
    dim: usize,
    k: usize,
    reps: usize,
    num_words: usize,
    num_dists: usize,
    num_heap_ops: usize,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            run_microbench: true,
            output_path: None,
            dim: 128,
            k: 10,
            reps: 5,
            num_words: 500_000,
            num_dists: 50_000,
            num_heap_ops: 50_000,
        }
    }
}

fn print_usage() {
    eprintln!(
        r#"Phase 5 Filtered Vector Search CLI (Milestone M3 Task T3.6)

USAGE:
    phase5 [OPTIONS]

OPTIONS:
    --microbench, -m
                        Run the Strategy 3 cost model microbenchmark (default).
    --output, -o <PATH>
                        Output path for structured JSON benchmark report.
                        If omitted or '-', JSON is written to stdout.
    --dim <USIZE>
                        Vector dimensionality for distance microbenchmark (default: 128).
    --k <USIZE>
                        Top-k neighbor count for heap microbenchmark (default: 10).
    --reps <USIZE>
                        Number of timed repetitions after 1 warmup (default: 5).
    --words <USIZE>
                        Word count for bitset scan microbenchmark (default: 500000).
    --dists <USIZE>
                        Evaluations count for distance microbenchmark (default: 50000).
    --heap-ops <USIZE>
                        Evaluations count for heap microbenchmark (default: 50000).
    --help, -h
                        Print this help message.
"#
    );
}

fn parse_cli_args(args: &[String]) -> Result<CliConfig> {
    let mut config = CliConfig::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--microbench" | "-m" => {
                config.run_microbench = true;
                i += 1;
            }
            "--output" | "-o" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --output".to_string(),
                    ));
                }
                config.output_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--dim" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --dim".to_string(),
                    ));
                }
                config.dim = args[i + 1].parse::<usize>().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --dim: {e}"))
                })?;
                i += 2;
            }
            "--k" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --k".to_string(),
                    ));
                }
                config.k = args[i + 1]
                    .parse::<usize>()
                    .map_err(|e| VectorSearchError::InvalidArgument(format!("Invalid --k: {e}")))?;
                i += 2;
            }
            "--reps" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --reps".to_string(),
                    ));
                }
                config.reps = args[i + 1].parse::<usize>().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --reps: {e}"))
                })?;
                i += 2;
            }
            "--words" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --words".to_string(),
                    ));
                }
                config.num_words = args[i + 1].parse::<usize>().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --words: {e}"))
                })?;
                i += 2;
            }
            "--dists" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --dists".to_string(),
                    ));
                }
                config.num_dists = args[i + 1].parse::<usize>().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --dists: {e}"))
                })?;
                i += 2;
            }
            "--heap-ops" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing argument for --heap-ops".to_string(),
                    ));
                }
                config.num_heap_ops = args[i + 1].parse::<usize>().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --heap-ops: {e}"))
                })?;
                i += 2;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized option: {other}"
                )));
            }
        }
    }

    Ok(config)
}

/// Measures the average nanoseconds required to scan one 64-bit bitset word ($t_{\text{word}}$).
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
    for &w in &words {
        sum ^= std::hint::black_box(w);
    }
    std::hint::black_box(sum);

    // Timed repetitions
    let mut latencies_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let mut s: u64 = 0;
        for &w in &words {
            s ^= std::hint::black_box(w);
        }
        std::hint::black_box(s);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        latencies_ns.push(elapsed_ns / num_words as f64);
    }

    latencies_ns.sort_by(|a, b| a.total_cmp(b));
    latencies_ns[latencies_ns.len() / 2]
}

/// Measures the average nanoseconds per vector distance computation ($t_{\text{dist}}(d)$).
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
    let mut dist_sum = 0.0_f32;
    for i in 0..num_evals {
        let offset = (i % num_cached_vectors) * dim;
        let v = &db_vecs[offset..offset + dim];
        dist_sum += l2_squared(std::hint::black_box(&q), std::hint::black_box(v));
    }
    std::hint::black_box(dist_sum);

    // Timed repetitions
    let mut latencies_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let mut acc = 0.0_f32;
        for i in 0..num_evals {
            let offset = (i % num_cached_vectors) * dim;
            let v = &db_vecs[offset..offset + dim];
            acc += l2_squared(std::hint::black_box(&q), std::hint::black_box(v));
        }
        std::hint::black_box(acc);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        latencies_ns.push(elapsed_ns / num_evals as f64);
    }

    latencies_ns.sort_by(|a, b| a.total_cmp(b));
    latencies_ns[latencies_ns.len() / 2]
}

/// Measures the average nanoseconds per bounded heap operation ($t_{\text{heap}}(k)$).
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
    for &cand in &candidates {
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
        let start = Instant::now();
        for &cand in &candidates {
            if h.len() < k {
                h.push(std::hint::black_box(cand));
            } else if let Some(mut worst) = h.peek_mut() {
                if cand < *worst {
                    *worst = std::hint::black_box(cand);
                }
            }
        }
        std::hint::black_box(&h);
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        latencies_ns.push(elapsed_ns / num_ops as f64);
    }

    latencies_ns.sort_by(|a, b| a.total_cmp(b));
    latencies_ns[latencies_ns.len() / 2]
}

fn get_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|out| !out.stdout.is_empty())
        .unwrap_or(false)
}

/// Evaluates the theoretical brute-force cost model:
/// $$C_{\text{brute}}(\sigma, N, d) = \frac{N}{64} t_{\text{word}} + \sigma N (t_{\text{dist}}(d) + t_{\text{heap}}(k))$$
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

fn build_json_report(config: &CliConfig, t_word_ns: f64, t_dist_ns: f64, t_heap_ns: f64) -> String {
    let hw = detect_hardware();
    let git_commit = get_git_commit();
    let git_dirty = get_git_dirty();
    let affinity_mask =
        env::var("VECTOR_SEARCH_AFFINITY_MASK").unwrap_or_else(|_| "0x0001".to_string());
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

    format!(
        r#"{{
  "benchmark": "cost_microbenchmark",
  "task": "Milestone M3 Task T3.6",
  "git_commit": "{}",
  "git_dirty": {},
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
        git_commit,
        git_dirty,
        hw.cpu_model,
        hw.logical_cores,
        hw.total_ram_gb
            .map_or("null".to_string(), |r| format!("{r:.1}")),
        hw.os,
        hw.arch,
        affinity_mask,
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
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(VectorSearchError::Io)?;
                }
            }
            fs::write(path, &json_report).map_err(VectorSearchError::Io)?;
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

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let config = match parse_cli_args(&args) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    match run_microbenchmark(&config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            ExitCode::FAILURE
        }
    }
}
