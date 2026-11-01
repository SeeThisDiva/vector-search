//! Phase 1 CLI Runner: Brute-Force Exact KNN Ground Truth & Recall Measurement Harness.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

use vector_search::benchmark::{
    run_phase1_benchmark, Phase1Config, Phase1DatasetMode, Phase1Report,
};
use vector_search::dataset::SiftPreset;
use vector_search::distance::Metric;
use vector_search::error::{Result, VectorSearchError};

struct CliOptions {
    config: Phase1Config,
    json_out: Option<PathBuf>,
    show_help: bool,
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            ExitCode::FAILURE
        }
    }
}

fn try_main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let opts = parse_cli_args(&args)?;

    if opts.show_help {
        print_usage();
        return Ok(());
    }

    let report = run_phase1_benchmark(&opts.config)?;
    print_report(&report);

    if let Some(json_path) = &opts.json_out {
        if let Some(parent) = json_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(json_path, report.to_json_string())?;
        println!("  - JSON Summary Report:   {}", json_path.display());
    }

    Ok(())
}

fn parse_cli_args(args: &[String]) -> Result<CliOptions> {
    let mut dataset_choice = String::from("siftsmall");
    let mut num_base: usize = 10_000;
    let mut num_query: usize = 100;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 16;
    let mut k: usize = 10;
    let mut seed: u64 = 42;
    let mut metric = Metric::L2Squared;
    let mut data_cache_dir = PathBuf::from("data");
    let mut output_dir = PathBuf::from("data");
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut gt_ivecs: Option<PathBuf> = None;
    let mut max_queries: Option<usize> = None;
    let mut allow_download = true;
    let mut save_files = true;
    let mut json_out: Option<PathBuf> = None;
    let mut show_help = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                show_help = true;
                i += 1;
            }
            "--dataset" => {
                dataset_choice = next_arg(args, &mut i, "--dataset")?.to_ascii_lowercase();
            }
            "--num-base" => {
                num_base = parse_num(next_arg(args, &mut i, "--num-base")?, "--num-base")?;
            }
            "--num-query" => {
                num_query = parse_num(next_arg(args, &mut i, "--num-query")?, "--num-query")?;
            }
            "--max-queries" => {
                max_queries = Some(parse_num(
                    next_arg(args, &mut i, "--max-queries")?,
                    "--max-queries",
                )?);
            }
            "--dim" => {
                dim = parse_num(next_arg(args, &mut i, "--dim")?, "--dim")?;
            }
            "--num-clusters" => {
                num_clusters =
                    parse_num(next_arg(args, &mut i, "--num-clusters")?, "--num-clusters")?;
            }
            "--k" => {
                k = parse_num(next_arg(args, &mut i, "--k")?, "--k")?;
            }
            "--seed" => {
                seed = parse_num(next_arg(args, &mut i, "--seed")?, "--seed")?;
            }
            "--metric" => {
                metric = Metric::from_str(next_arg(args, &mut i, "--metric")?)?;
            }
            "--data-dir" => {
                data_cache_dir = PathBuf::from(next_arg(args, &mut i, "--data-dir")?);
            }
            "--output-dir" => {
                output_dir = PathBuf::from(next_arg(args, &mut i, "--output-dir")?);
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(next_arg(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                query_fvecs = Some(PathBuf::from(next_arg(args, &mut i, "--query-fvecs")?));
            }
            "--gt-ivecs" => {
                gt_ivecs = Some(PathBuf::from(next_arg(args, &mut i, "--gt-ivecs")?));
            }
            "--json-out" => {
                json_out = Some(PathBuf::from(next_arg(args, &mut i, "--json-out")?));
            }
            "--offline" => {
                allow_download = false;
                i += 1;
            }
            "--no-save" => {
                save_files = false;
                i += 1;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI flag '{unknown}'. Run with --help for usage."
                )));
            }
        }
    }

    let dataset_mode = if base_fvecs.is_some()
        || query_fvecs.is_some()
        || dataset_choice == "custom"
    {
        let (Some(b), Some(q)) = (base_fvecs, query_fvecs) else {
            return Err(VectorSearchError::InvalidArgument(
                "Custom dataset mode requires both --base-fvecs <path> and --query-fvecs <path>"
                    .to_string(),
            ));
        };
        Phase1DatasetMode::Custom {
            base_fvecs: b,
            query_fvecs: q,
            gt_ivecs,
        }
    } else {
        match dataset_choice.as_str() {
            "siftsmall" => Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
            "sift1m" | "sift" => Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            "synthetic" => Phase1DatasetMode::Synthetic {
                num_base,
                num_query,
                dim,
                num_clusters,
            },
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unsupported --dataset '{other}'. Expected 'siftsmall', 'sift1m', 'synthetic', or 'custom'."
                )));
            }
        }
    };

    Ok(CliOptions {
        config: Phase1Config {
            dataset_mode,
            data_cache_dir,
            output_dir,
            k,
            seed,
            metric,
            allow_download,
            save_files,
            max_queries,
        },
        json_out,
        show_help,
    })
}

fn next_arg<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str> {
    if *idx + 1 >= args.len() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Missing value after '{flag}'"
        )));
    }
    let val = &args[*idx + 1];
    *idx += 2;
    Ok(val.as_str())
}

fn parse_num<T: FromStr>(raw: &str, flag: &str) -> Result<T> {
    raw.parse::<T>().map_err(|_| {
        VectorSearchError::InvalidArgument(format!("Invalid numeric value '{raw}' for {flag}"))
    })
}

fn print_usage() {
    println!(
        "\
HNSW Vector Search (Rust) -- Phase 1: Brute-Force Ground Truth & Recall Harness

USAGE:
    cargo run --release --bin phase1 -- [OPTIONS]

OPTIONS:
    --dataset <siftsmall|sift1m|synthetic|custom>
                                Dataset preset to load or download (default: siftsmall)
    --data-dir <DIR>            Cache directory for public SIFT corpora (default: data)
    --output-dir <DIR>          Output directory for .fvecs/.ivecs artifacts (default: data)
    --base-fvecs <PATH>         Custom base vectors .fvecs path (implies --dataset custom)
    --query-fvecs <PATH>        Custom query vectors .fvecs path (implies --dataset custom)
    --gt-ivecs <PATH>           Optional reference ground truth .ivecs for custom mode
    --num-base <N>              Base vector count for --dataset synthetic (default: 10000)
    --num-query <Q>             Query vector count for --dataset synthetic (default: 100)
    --max-queries <Q>           Limit benchmark evaluation to first Q queries
    --dim <D>                   Vector dimension for --dataset synthetic (default: 128)
    --k <K>                     Top-k nearest neighbors to retrieve (default: 10)
    --seed <SEED>               Deterministic PRNG seed (default: 42)
    --metric <l2|euclidean|ip>  Distance metric (default: l2)
    --offline                   Disable network download; use local cache or seeded fallback
    --no-save                   Skip writing output .fvecs / .ivecs artifacts to disk
    --json-out <PATH>           Write structured JSON benchmark report to PATH
    -h, --help                  Print this help message"
    );
}

fn print_report(r: &Phase1Report) {
    println!("{}", "=".repeat(80));
    println!(" HNSW APPROXIMATE NEAREST NEIGHBOR SEARCH (RUST) -- PHASE 1 EXECUTION ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!("Dataset Provenance: {}", r.dataset_source);
    println!("Configuration:");
    println!("  - Base Vectors:          {} vectors", r.num_base);
    println!("  - Query Vectors:         {} vectors", r.num_query);
    println!("  - Vector Dimension:      {}-d (float32)", r.dim);
    println!(
        "  - Top-k Neighbors:       {} (effective k={})",
        r.k, r.effective_k
    );
    println!("  - Distance Metric:       {}", r.metric);
    println!("  - Deterministic Seed:    {}", r.seed);
    println!("{}", "-".repeat(80));

    println!("[+] Dataset Memory & Load Profile:");
    println!("  - Base vectors memory:   {:.2} MiB", r.base_memory_mb);
    println!("  - Query vectors memory:  {:.3} MiB", r.query_memory_mb);
    println!(
        "  - Load / gen elapsed:    {:.2} ms",
        r.dataset_load_time_s * 1000.0
    );

    println!("\n[+] Brute-Force Exact KNN Latency & Throughput:");
    println!("  - Index fit elapsed:     {:.3} ms", r.fit_time_s * 1000.0);
    println!(
        "  - Total search elapsed:  {:.2} ms",
        r.latency.total_s * 1000.0
    );
    println!("  - Query throughput:      {:.1} QPS", r.latency.qps);
    println!(
        "  - Latency min / mean:    {:.3} ms / {:.3} ms",
        r.latency.min_ms, r.latency.mean_ms
    );
    println!("  - Latency p50 (median):  {:.3} ms", r.latency.p50_ms);
    println!("  - Latency p95:           {:.3} ms", r.latency.p95_ms);
    println!("  - Latency p99:           {:.3} ms", r.latency.p99_ms);
    println!("  - Latency max:           {:.3} ms", r.latency.max_ms);

    println!("\n[+] Recall Harness & Self-Verification:");
    for (metric_name, val) in &r.recall_summary {
        println!(
            "  - {:14}:        {:.2}% ({:.6})",
            metric_name,
            val * 100.0,
            val
        );
    }
    println!(
        "  - Self-Verification:     {}",
        if r.self_verified {
            "PASSED (100.0% Exact Match)"
        } else {
            "FAILED"
        }
    );

    if let Some(cmp) = &r.official_gt_comparison {
        println!("\n[+] Authentic Corpus Ground Truth Cross-Check:");
        println!(
            "  - Recall@{} vs Official: {:.2}% ({:.6})",
            cmp.eval_k,
            cmp.recall_at_k * 100.0,
            cmp.recall_at_k
        );
        println!(
            "  - Ordered ID Alignment:  {}/{} ({:.2}%)",
            cmp.exact_id_matches,
            cmp.total_compared,
            (cmp.exact_id_matches as f64 / cmp.total_compared.max(1) as f64) * 100.0
        );
    }

    println!("\n[+] Binary Dataset Artifacts (Texmex Specification):");
    if let Some(base_p) = &r.artifacts.base_fvecs {
        println!("  - Base Vectors:          {}", base_p.display());
        if let Some(p) = &r.artifacts.query_fvecs {
            println!("  - Query Vectors:         {}", p.display());
        }
        if let Some(p) = &r.artifacts.groundtruth_ivecs {
            println!("  - Ground Truth Indices:  {}", p.display());
        }
        if let Some(p) = &r.artifacts.groundtruth_fvecs {
            println!("  - Ground Truth Distances:{}", p.display());
        }
    } else {
        println!("  - Binary serialization skipped (--no-save).");
    }

    println!("\n{}", "=".repeat(80));
    println!(" PHASE 1 COMPLETE (RUST) -- HALTED FOR USER REVIEW BEFORE PHASE 2 ");
    println!("{}", "=".repeat(80));
}
