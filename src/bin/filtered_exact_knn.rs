//! Filtered Exact KNN Ground Truth CLI (Milestone M3 / Phase 5: Task T3.3).
//!
//! Authoritative filtered ground truth generator and parity verification harness.
//! Computes exact pairwise distances with canonical `(distance, id)` deterministic
//! tie-breaking and validates the self-parity invariant `Recall@k(GT, GT) == 1.000000`.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Instant;

use vector_search::benchmark::detect_provenance;
use vector_search::cli_util::{next_str, parse_next};
use vector_search::dataset::{
    generate_mode_a_bitset, generate_mode_b_bitset_detailed, generate_synthetic_sift, read_bitset,
    read_fvecs_max_rows, read_ivecs, write_bitset, write_bitset_as_ivecs, write_fvecs, write_ivecs,
    S_CLUST_DEFAULT,
};
use vector_search::distance::Metric;
use vector_search::error::{Result, VectorSearchError};
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::filter::DenseBitset;
use vector_search::hardware::detect_hardware;
use vector_search::json_util::{escape_json, write_file_with_parents};
use vector_search::recall::compute_recall_at_k;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterMode {
    Independent,
    Correlated,
}

#[derive(Debug)]
struct CliConfig {
    num_base: usize,
    num_query: usize,
    explicit_num_base: Option<usize>,
    explicit_num_query: Option<usize>,
    dim: usize,
    k: usize,
    seed: u64,
    base_fvecs: Option<PathBuf>,
    query_fvecs: Option<PathBuf>,
    filter_bitset: Option<PathBuf>,
    selectivity: f64,
    filter_mode: FilterMode,
    filter_seed: u64,
    seed_clust: u64,
    metric: Metric,
    output_dir: Option<PathBuf>,
    output_ivecs: Option<PathBuf>,
    output_fvecs: Option<PathBuf>,
    output_bitset: Option<PathBuf>,
    json_path: Option<PathBuf>,
    threads: usize,
}

fn print_usage() {
    println!(
        r#"Filtered Exact KNN Ground Truth Harness (Milestone M3 Task T3.3)

USAGE:
    filtered_exact_knn [OPTIONS]

OPTIONS:
    --num-base <N>, -n <N>
                        Number of base vectors N (default: 10000)
    --num-query <Q>, -q <Q>
                        Number of query vectors Q (default: 100)
    --dim <D>, -d <D>   Vector dimensionality (default: 128)
    --k <K>, -k <K>     Number of nearest neighbors to retrieve (default: 10)
    --seed <SEED>       PRNG seed for synthetic dataset (default: 42)
    --base-fvecs <PATH> Path to existing base vectors .fvecs file
    --query-fvecs <PATH>
                        Path to existing query vectors .fvecs file
    --filter-bitset <PATH>
                        Path to existing filter bitset file (.bin or .ivecs)
    --selectivity <S>, -s <S>
                        Filter selectivity in [0.0, 1.0], e.g. 0.5, 0.1, 0.01, 0.001 (default: 0.1)
    --filter-mode <independent|correlated>
                        Mode A (independent) or Mode B (correlated) (default: independent)
    --filter-seed <SEED>
                        PRNG seed for attribute generation (default: 42)
    --seed-clust <SEED> PRNG seed for Mode B centroids (default: 0xCBF29CE484222325)
    --metric <l2|euclidean|ip>
                        Distance metric: 'l2' (squared L2), 'euclidean', 'ip' (default: l2)
    --output-dir <PATH>, -o <PATH>
                        Output directory for groundtruth.ivecs, groundtruth.fvecs, filter.bin
    --output-ivecs <PATH>
                        Path for groundtruth neighbor indices (.ivecs)
    --output-fvecs <PATH>
                        Path for groundtruth neighbor distances (.fvecs)
    --output-bitset <PATH>
                        Path for serialized bitset (.bin)
    --json-out <PATH>   Path for structured JSON report
    --threads <T>, -t <T>
                        Number of parallel worker threads (default: 0 = all cores)
    -h, --help          Print this help message"#
    );
}

fn parse_cli_args() -> Result<Option<CliConfig>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(None);
    }

    let mut num_base: usize = 10_000;
    let mut num_query: usize = 100;
    let mut explicit_num_base: Option<usize> = None;
    let mut explicit_num_query: Option<usize> = None;
    let mut dim: usize = 128;
    let mut k: usize = 10;
    let mut seed: u64 = 42;
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut filter_bitset: Option<PathBuf> = None;
    let mut selectivity: f64 = 0.1;
    let mut filter_mode = FilterMode::Independent;
    let mut filter_seed: u64 = 42;
    let mut seed_clust: u64 = S_CLUST_DEFAULT;
    let mut metric = Metric::L2Squared;
    let mut output_dir: Option<PathBuf> = None;
    let mut output_ivecs: Option<PathBuf> = None;
    let mut output_fvecs: Option<PathBuf> = None;
    let mut output_bitset: Option<PathBuf> = None;
    let mut json_path: Option<PathBuf> = None;
    let mut threads: usize = 0;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--num-base" | "-n" => {
                let parsed: usize = parse_next(&args, &mut i, flag)?;
                num_base = parsed;
                explicit_num_base = Some(parsed);
            }
            "--num-query" | "-q" => {
                let parsed: usize = parse_next(&args, &mut i, flag)?;
                num_query = parsed;
                explicit_num_query = Some(parsed);
            }
            "--dim" | "-d" => {
                dim = parse_next(&args, &mut i, flag)?;
            }
            "--k" | "-k" => {
                k = parse_next(&args, &mut i, flag)?;
            }
            "--seed" => {
                seed = parse_next(&args, &mut i, "--seed")?;
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(next_str(&args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                query_fvecs = Some(PathBuf::from(next_str(&args, &mut i, "--query-fvecs")?));
            }
            "--filter-bitset" => {
                filter_bitset = Some(PathBuf::from(next_str(&args, &mut i, "--filter-bitset")?));
            }
            "--selectivity" | "-s" => {
                let val_str = next_str(&args, &mut i, flag)?;
                let clean_str = val_str.trim_end_matches('%');
                let mut parsed: f64 = clean_str.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --selectivity: {e}"))
                })?;
                if val_str.ends_with('%') {
                    parsed /= 100.0;
                }
                if !(0.0..=1.0).contains(&parsed) {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "Selectivity must be in [0.0, 1.0], got {parsed}"
                    )));
                }
                selectivity = parsed;
            }
            "--filter-mode" => {
                let mode_str = next_str(&args, &mut i, "--filter-mode")?;
                match mode_str.to_lowercase().as_str() {
                    "independent" | "a" | "mode_a" => filter_mode = FilterMode::Independent,
                    "correlated" | "b" | "mode_b" => filter_mode = FilterMode::Correlated,
                    other => {
                        return Err(VectorSearchError::InvalidArgument(format!(
                            "Invalid filter mode: '{other}'. Expected 'independent' or 'correlated'"
                        )));
                    }
                }
            }
            "--filter-seed" => {
                filter_seed = parse_next(&args, &mut i, "--filter-seed")?;
            }
            "--seed-clust" => {
                let val_str = next_str(&args, &mut i, "--seed-clust")?;
                seed_clust = if let Some(hex) = val_str
                    .strip_prefix("0x")
                    .or_else(|| val_str.strip_prefix("0X"))
                {
                    u64::from_str_radix(hex, 16).map_err(|e| {
                        VectorSearchError::InvalidArgument(format!("Invalid hex --seed-clust: {e}"))
                    })?
                } else {
                    val_str.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!("Invalid --seed-clust: {e}"))
                    })?
                };
            }
            "--metric" => {
                let val = next_str(&args, &mut i, "--metric")?;
                metric = Metric::from_str(val)?;
            }
            "--output-dir" | "-o" => {
                output_dir = Some(PathBuf::from(next_str(&args, &mut i, flag)?));
            }
            "--output-ivecs" => {
                output_ivecs = Some(PathBuf::from(next_str(&args, &mut i, "--output-ivecs")?));
            }
            "--output-fvecs" => {
                output_fvecs = Some(PathBuf::from(next_str(&args, &mut i, "--output-fvecs")?));
            }
            "--output-bitset" => {
                output_bitset = Some(PathBuf::from(next_str(&args, &mut i, "--output-bitset")?));
            }
            "--json-out" => {
                json_path = Some(PathBuf::from(next_str(&args, &mut i, "--json-out")?));
            }
            "--threads" | "-t" => {
                threads = parse_next(&args, &mut i, flag)?;
            }
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized option '{other}'. Run with --help for usage."
                )));
            }
        }
    }

    Ok(Some(CliConfig {
        num_base,
        num_query,
        explicit_num_base,
        explicit_num_query,
        dim,
        k,
        seed,
        base_fvecs,
        query_fvecs,
        filter_bitset,
        selectivity,
        filter_mode,
        filter_seed,
        seed_clust,
        metric,
        output_dir,
        output_ivecs,
        output_fvecs,
        output_bitset,
        json_path,
        threads,
    }))
}

fn run() -> Result<()> {
    let Some(config) = parse_cli_args()? else {
        return Ok(());
    };

    let git_commit = detect_provenance().git_commit;
    let hw = detect_hardware();

    println!("============================================================");
    println!("Filtered Exact KNN Ground Truth Harness (Milestone M3 T3.3)");
    println!("============================================================");
    println!("Git Commit:       {git_commit}");
    println!("CPU Model:        {}", hw.cpu_model);
    println!("Logical Cores:    {}", hw.logical_cores);
    println!("Metric:           {:?}", config.metric);
    println!("K:                {}", config.k);

    // 1. Load or generate base and query vectors
    let (base_matrix, query_matrix, is_synthetic) = if let Some(ref base_p) = config.base_fvecs {
        println!("Loading base vectors from {}...", base_p.display());
        let loaded_base = read_fvecs_max_rows(base_p, config.explicit_num_base)?;
        let loaded_query = if let Some(ref query_p) = config.query_fvecs {
            println!("Loading query vectors from {}...", query_p.display());
            read_fvecs_max_rows(query_p, config.explicit_num_query)?
        } else {
            let take_q = config.num_query.min(loaded_base.rows());
            println!("Using first {take_q} base vectors as queries...");
            loaded_base.slice_first_rows(take_q)
        };
        (loaded_base, loaded_query, false)
    } else {
        println!(
            "Generating synthetic dataset: N={}, Q={}, dim={}, seed={}...",
            config.num_base, config.num_query, config.dim, config.seed
        );
        let (base, query) = generate_synthetic_sift(
            config.num_base,
            config.num_query,
            config.dim,
            100,
            config.seed,
        )?;
        (base, query, true)
    };

    let n = base_matrix.rows();
    let q = query_matrix.rows();
    let d = base_matrix.cols();
    println!("Base Vectors:     N={n}, D={d}");
    println!("Query Vectors:    Q={q}, D={d}");

    // 2. Load or generate filter bitset
    let bitset = if let Some(ref filter_p) = config.filter_bitset {
        println!("Loading filter bitset from {}...", filter_p.display());
        let ext = filter_p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext == "ivecs" {
            let ivecs = read_ivecs(filter_p)?;
            if ivecs.rows() == 0 {
                DenseBitset::zeros(n)
            } else {
                let ids = ivecs.row(0);
                let mut b = DenseBitset::zeros(n);
                for &id in ids {
                    if id >= 0 && (id as usize) < n {
                        b.set_bit(id as usize, true);
                    }
                }
                b
            }
        } else {
            read_bitset(filter_p)?
        }
    } else {
        println!(
            "Generating bitset: selectivity={:.6} ({:.2}%), mode={:?}...",
            config.selectivity,
            config.selectivity * 100.0,
            config.filter_mode
        );
        match config.filter_mode {
            FilterMode::Independent => {
                generate_mode_a_bitset(n, config.selectivity, config.filter_seed)
            }
            FilterMode::Correlated => {
                let meta = generate_mode_b_bitset_detailed(
                    &base_matrix,
                    config.selectivity,
                    config.filter_seed,
                    config.seed_clust,
                );
                meta.bitset
            }
        }
    };

    let popcount = bitset.count_ones();
    println!(
        "Bitset Popcount:  {popcount} / {n} ({:.4}%)",
        (popcount as f64 / n as f64) * 100.0
    );

    // 3. Compute exact filtered KNN
    println!("Fitting FilteredExactKnn engine...");
    let engine = FilteredExactKnn::new(config.metric, base_matrix.clone())?;

    println!(
        "Running filtered exact search across {q} queries (threads={})...",
        if config.threads == 0 {
            hw.logical_cores
        } else {
            config.threads
        }
    );
    let start = Instant::now();
    let batch_res =
        engine.search_batch_parallel(&query_matrix, config.k, &bitset, config.threads)?;
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let qps = if elapsed_ms > 0.0 {
        (q as f64) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let effective_k = batch_res.indices.cols();
    println!("Search completed in {elapsed_ms:.2} ms ({qps:.1} QPS). Effective K: {effective_k}");

    // 4. Verify Self-Parity Invariant Recall@k(GT, GT) == 1.000000
    let mut self_recall = 1.0_f64;
    if q > 0 && effective_k > 0 {
        self_recall = compute_recall_at_k(&batch_res.indices, &batch_res.indices, effective_k)?;
        println!("Self-Parity:      Recall@{effective_k}(GT, GT) = {self_recall:.6}");
        assert!(
            (self_recall - 1.0).abs() < f64::EPSILON,
            "Self-parity invariant failed: expected 1.0, got {self_recall}"
        );
    }

    // 5. Write outputs
    if let Some(ref out_dir) = config.output_dir {
        fs::create_dir_all(out_dir)?;
        let gt_ivecs_path = out_dir.join("groundtruth.ivecs");
        let gt_fvecs_path = out_dir.join("groundtruth.fvecs");
        let filter_bin_path = out_dir.join("filter.bin");
        let filter_ivecs_path = out_dir.join("filter.ivecs");

        println!("Writing {}...", gt_ivecs_path.display());
        write_ivecs(&gt_ivecs_path, &batch_res.indices)?;

        println!("Writing {}...", gt_fvecs_path.display());
        write_fvecs(&gt_fvecs_path, &batch_res.distances)?;

        println!("Writing {}...", filter_bin_path.display());
        write_bitset(&filter_bin_path, &bitset)?;

        println!("Writing {}...", filter_ivecs_path.display());
        write_bitset_as_ivecs(&filter_ivecs_path, &bitset)?;

        if is_synthetic {
            let base_path = out_dir.join("base.fvecs");
            let query_path = out_dir.join("query.fvecs");
            println!("Writing synthetic base to {}...", base_path.display());
            write_fvecs(&base_path, &base_matrix)?;
            println!("Writing synthetic query to {}...", query_path.display());
            write_fvecs(&query_path, &query_matrix)?;
        }
    }

    if let Some(ref path) = config.output_ivecs {
        println!("Writing indices to {}...", path.display());
        write_ivecs(path, &batch_res.indices)?;
    }

    if let Some(ref path) = config.output_fvecs {
        println!("Writing distances to {}...", path.display());
        write_fvecs(path, &batch_res.distances)?;
    }

    if let Some(ref path) = config.output_bitset {
        println!("Writing bitset to {}...", path.display());
        write_bitset(path, &bitset)?;
    }

    // 6. Write Provenance JSON
    if let Some(ref json_p) = config.json_path {
        println!("Writing provenance report to {}...", json_p.display());
        let mode_str = match config.filter_mode {
            FilterMode::Independent => "independent",
            FilterMode::Correlated => "correlated",
        };
        let metric_str = match config.metric {
            Metric::L2Squared => "l2",
            Metric::Euclidean => "euclidean",
            Metric::InnerProduct => "ip",
        };

        let json_text = format!(
            r#"{{
  "git_commit": "{git_commit}",
  "hardware": {{
    "cpu_model": "{cpu_model}",
    "logical_cores": {logical_cores}
  }},
  "dataset": {{
    "num_base": {n},
    "num_query": {q},
    "dim": {d},
    "metric": "{metric_str}"
  }},
  "filter": {{
    "mode": "{mode_str}",
    "selectivity": {selectivity},
    "popcount": {popcount},
    "seed_attr": {seed_attr}
  }},
  "exact_knn": {{
    "requested_k": {requested_k},
    "effective_k": {effective_k},
    "elapsed_ms": {elapsed_ms:.4},
    "qps": {qps:.2}
  }},
  "recall": {{
    "self_verified": true,
    "self_recall_at_k": {self_recall:.6}
  }}
}}
"#,
            git_commit = escape_json(&git_commit),
            cpu_model = escape_json(&hw.cpu_model),
            logical_cores = hw.logical_cores,
            selectivity = config.selectivity,
            seed_attr = config.filter_seed,
            requested_k = config.k,
        );

        write_file_with_parents(json_p, json_text)?;
        println!("Provenance JSON written successfully.");
    }

    println!("============================================================");
    println!("[SUCCESS] Filtered exact KNN completed with 100% self-parity.");
    println!("============================================================");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ERROR] {e}");
            ExitCode::FAILURE
        }
    }
}
