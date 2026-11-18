//! CLI Runners for Phase 2 and Phase 3 Benchmarks.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::str::FromStr;

use crate::cli_util::{next_str, parse_csv_list, parse_next};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::json_util::write_file_with_parents;

use super::baseline_profile::{
    format_scale_profile_kv, print_baseline_profile_report, profile_single_sift1m_scale,
    run_baseline_multi_scale_profile, BaselineMultiScaleReport,
};
use super::dataset_loader::resolve_cli_dataset_mode;
use super::golden::{generate_golden_fixtures, run_phase3_5_baseline_measurement};
use super::phase2::{print_phase2_report, run_phase2_benchmark, Phase2Config};
use super::phase3::{print_phase3_report, run_phase3_benchmark, Phase3Config};

/// Entry point for CLI execution of Phase 2 benchmark.
pub fn run_phase2_cli(args: &[String]) -> Result<()> {
    let mut dataset_choice = String::from("siftsmall");
    let mut num_base: usize = 10_000;
    let mut num_query: usize = 100;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 16;
    let mut k: usize = 10;
    let mut seed: u64 = 42;
    let mut metric = Metric::L2Squared;
    let mut m: usize = 16;
    let mut m_max0: Option<usize> = None;
    let mut ef_construction: usize = 100;
    let mut ef_search_list = vec![10_usize, 20, 50, 100, 200];
    let mut data_cache_dir = PathBuf::from("data");
    let mut output_dir = PathBuf::from("data");
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut gt_ivecs: Option<PathBuf> = None;
    let mut max_queries: Option<usize> = None;
    let mut allow_download = true;
    let mut save_files = true;
    let mut run_repro_check = true;
    let mut json_out: Option<PathBuf> = None;
    let mut profile_baseline = false;
    let mut profile_scales: Vec<usize> = Vec::new();
    let mut internal_profile_scale: Option<usize> = None;
    let mut sweep_runs: usize = 5;
    let mut show_help = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                show_help = true;
                i += 1;
            }
            "--dataset" => {
                dataset_choice = next_str(args, &mut i, "--dataset")?.to_ascii_lowercase();
            }
            "--num-base" => {
                num_base = parse_next(args, &mut i, "--num-base")?;
            }
            "--num-query" => {
                num_query = parse_next(args, &mut i, "--num-query")?;
            }
            "--max-queries" => {
                max_queries = Some(parse_next(args, &mut i, "--max-queries")?);
            }
            "--dim" => {
                dim = parse_next(args, &mut i, "--dim")?;
            }
            "--num-clusters" => {
                num_clusters = parse_next(args, &mut i, "--num-clusters")?;
            }
            "--k" => {
                k = parse_next(args, &mut i, "--k")?;
            }
            "--seed" => {
                seed = parse_next(args, &mut i, "--seed")?;
            }
            "--metric" => {
                metric = Metric::from_str(next_str(args, &mut i, "--metric")?)?;
            }
            "--m" => {
                m = parse_next(args, &mut i, "--m")?;
            }
            "--m-max0" => {
                m_max0 = Some(parse_next(args, &mut i, "--m-max0")?);
            }
            "--ef-construction" => {
                ef_construction = parse_next(args, &mut i, "--ef-construction")?;
            }
            "--ef-search" => {
                let raw = next_str(args, &mut i, "--ef-search")?;
                ef_search_list = parse_csv_list(raw, "--ef-search")?;
            }
            "--data-dir" => {
                data_cache_dir = PathBuf::from(next_str(args, &mut i, "--data-dir")?);
            }
            "--output-dir" => {
                output_dir = PathBuf::from(next_str(args, &mut i, "--output-dir")?);
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                query_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--query-fvecs")?));
            }
            "--gt-ivecs" => {
                gt_ivecs = Some(PathBuf::from(next_str(args, &mut i, "--gt-ivecs")?));
            }
            "--json-out" => {
                json_out = Some(PathBuf::from(next_str(args, &mut i, "--json-out")?));
            }
            "--offline" => {
                allow_download = false;
                i += 1;
            }
            "--no-save" => {
                save_files = false;
                i += 1;
            }
            "--repro-check" => {
                run_repro_check = true;
                i += 1;
            }
            "--no-repro-check" => {
                run_repro_check = false;
                i += 1;
            }
            "--phase" => {
                let _ = next_str(args, &mut i, "--phase")?;
            }
            "--profile-baseline" | "--baseline-profile" => {
                profile_baseline = true;
                i += 1;
            }
            "--profile-scales" => {
                let raw = next_str(args, &mut i, "--profile-scales")?;
                profile_scales = parse_csv_list(raw, "--profile-scales")?;
                profile_baseline = true;
            }
            "--internal-profile-scale" => {
                internal_profile_scale =
                    Some(parse_next(args, &mut i, "--internal-profile-scale")?);
            }
            "--sweep-runs" => {
                sweep_runs = parse_next(args, &mut i, "--sweep-runs")?;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI flag '{unknown}'"
                )));
            }
        }
    }

    if let Some(scale) = internal_profile_scale {
        let profile = profile_single_sift1m_scale(
            &data_cache_dir,
            scale,
            seed,
            allow_download,
            sweep_runs.max(1),
        )?;
        print!("{}", format_scale_profile_kv(&profile));
        return Ok(());
    }

    if show_help {
        println!(
            "\
HNSW Vector Search (Rust) -- Phase 2: 10,000 Vectors Verification & Benchmarking

USAGE:
    cargo run --release --bin phase2 -- [OPTIONS]
    cargo run --release --bin phase1 -- --phase 2 [OPTIONS]

OPTIONS:
    --dataset <siftsmall|sift1m|synthetic|custom>
                                Dataset preset to load or download (default: siftsmall)
    --data-dir <DIR>            Cache directory for public SIFT corpora (default: data)
    --output-dir <DIR>          Output directory for artifacts (default: data)
    --num-base <N>              Base vector count for synthetic dataset (default: 10000)
    --num-query <Q>             Query vector count for synthetic dataset (default: 100)
    --dim <D>                   Vector dimension for synthetic dataset (default: 128)
    --k <K>                     Top-k nearest neighbors to retrieve (default: 10)
    --seed <SEED>               Deterministic PRNG seed (default: 42)
    --metric <l2|euclidean|ip>  Distance metric (default: l2)
    --m <M>                     Max outgoing edges per node at layers > 0 (default: 16)
    --m-max0 <M0>               Max outgoing edges per node at layer 0 (default: 2*M)
    --ef-construction <EFC>     Candidate beam width during index build (default: 100)
    --ef-search <LIST>          Comma-separated efSearch values to sweep (default: 10,20,50,100,200)
    --profile-baseline          Profile HNSW across multiple scales (10k, 100k, 1M)
    --profile-scales <SCALES>   Comma-separated vector scales to profile (default: 10000,100000,1000000)
    --json-out <PATH>           Write structured JSON benchmark report to PATH
    --offline                   Disable internet downloads; use local cache or synthetic
    --no-save                   Skip writing intermediate artifacts to disk
    --repro-check               Verify bit-exact reproducibility across independent runs
    --no-repro-check            Skip reproducibility check
    -h, --help                  Print this help message"
        );
        return Ok(());
    }

    if profile_baseline {
        let report = run_baseline_multi_scale_profile(
            &data_cache_dir,
            &profile_scales,
            seed,
            allow_download,
        )?;
        print_baseline_profile_report(&report);

        let target_json =
            json_out.unwrap_or_else(|| PathBuf::from("data/baseline_profile_report.json"));
        write_file_with_parents(&target_json, report.to_json_string())?;
        println!("  - JSON Baseline Report:  {}", target_json.display());
        return Ok(());
    }

    let dataset_mode = resolve_cli_dataset_mode(
        &dataset_choice,
        base_fvecs,
        query_fvecs,
        gt_ivecs,
        num_base,
        num_query,
        dim,
        num_clusters,
    )?;

    let config = Phase2Config {
        dataset_mode,
        data_cache_dir,
        output_dir,
        k,
        seed,
        metric,
        m,
        m_max0,
        ef_construction,
        ef_search_list,
        allow_download,
        save_files,
        max_queries,
        run_repro_check,
    };

    let report = run_phase2_benchmark(&config)?;
    print_phase2_report(&report);

    if let Some(json_path) = &json_out {
        write_file_with_parents(json_path, report.to_json_string())?;
        println!("  - JSON Summary Report:   {}", json_path.display());
    }

    Ok(())
}

/// Entry point for CLI execution of Phase 3 benchmark.
pub fn run_phase3_cli(args: &[String]) -> Result<()> {
    let mut dataset_choice = String::from("sift1m");
    let mut num_base: usize = 1_000_000;
    let mut num_query: usize = 10_000;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 16;
    let mut k: usize = 10;
    let mut seed: u64 = 42;
    let mut metric = Metric::L2Squared;
    let mut m: usize = 16;
    let mut m_max0: Option<usize> = None;
    let mut ef_construction: usize = 100;
    let mut ef_search_list = vec![10_usize, 20, 50, 100, 200];
    let mut data_cache_dir = PathBuf::from("data");
    let mut output_dir = PathBuf::from("data");
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut gt_ivecs: Option<PathBuf> = None;
    let mut max_queries: Option<usize> = None;
    let mut max_base: Option<usize> = None;
    let mut num_threads: usize = 0;
    let mut allow_download = true;
    let mut save_files = true;
    let mut run_repro_check = true;
    let mut repro_vectors = 10_000;
    let mut sweep_runs: usize = 5;
    let mut run_multi_scale = true;
    let mut keep_pruned_shrink: Option<bool> = None;
    let mut internal_profile_scale: Option<usize> = None;
    let mut generate_golden = false;
    let mut measure_baseline = false;
    let mut json_out: Option<PathBuf> = None;
    let mut show_help = false;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-h" | "--help" => {
                show_help = true;
                i += 1;
            }
            "--dataset" => {
                dataset_choice = next_str(args, &mut i, "--dataset")?.to_ascii_lowercase();
            }
            "--num-base" | "--max-base" | "--scale" => {
                let val: usize = parse_next(args, &mut i, flag)?;
                num_base = val;
                max_base = Some(val);
            }
            "--num-query" => {
                num_query = parse_next(args, &mut i, "--num-query")?;
            }
            "--max-queries" => {
                max_queries = Some(parse_next(args, &mut i, "--max-queries")?);
            }
            "--dim" => {
                dim = parse_next(args, &mut i, "--dim")?;
            }
            "--num-clusters" => {
                num_clusters = parse_next(args, &mut i, "--num-clusters")?;
            }
            "--k" => {
                k = parse_next(args, &mut i, "--k")?;
            }
            "--seed" => {
                seed = parse_next(args, &mut i, "--seed")?;
            }
            "--metric" => {
                metric = Metric::from_str(next_str(args, &mut i, "--metric")?)?;
            }
            "--m" => {
                m = parse_next(args, &mut i, "--m")?;
            }
            "--m-max0" => {
                m_max0 = Some(parse_next(args, &mut i, "--m-max0")?);
            }
            "--ef-construction" => {
                ef_construction = parse_next(args, &mut i, "--ef-construction")?;
            }
            "--ef-search" => {
                let raw = next_str(args, &mut i, "--ef-search")?;
                ef_search_list = parse_csv_list(raw, "--ef-search")?;
            }
            "--threads" | "--num-threads" => {
                num_threads = parse_next(args, &mut i, flag)?;
            }
            "--data-dir" => {
                data_cache_dir = PathBuf::from(next_str(args, &mut i, "--data-dir")?);
            }
            "--output-dir" => {
                output_dir = PathBuf::from(next_str(args, &mut i, "--output-dir")?);
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                query_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--query-fvecs")?));
            }
            "--gt-ivecs" => {
                gt_ivecs = Some(PathBuf::from(next_str(args, &mut i, "--gt-ivecs")?));
            }
            "--json-out" => {
                json_out = Some(PathBuf::from(next_str(args, &mut i, "--json-out")?));
            }
            "--offline" => {
                allow_download = false;
                i += 1;
            }
            "--no-save" => {
                save_files = false;
                i += 1;
            }
            "--repro-check" => {
                run_repro_check = true;
                i += 1;
            }
            "--no-repro-check" => {
                run_repro_check = false;
                i += 1;
            }
            "--repro-vectors" => {
                repro_vectors = parse_next(args, &mut i, "--repro-vectors")?;
            }
            "--sweep-runs" => {
                sweep_runs = parse_next(args, &mut i, "--sweep-runs")?;
            }
            "--multi-scale" => {
                run_multi_scale = true;
                i += 1;
            }
            "--no-multi-scale" => {
                run_multi_scale = false;
                i += 1;
            }
            "--no-keep-pruned-shrink" => {
                keep_pruned_shrink = Some(false);
                i += 1;
            }
            "--keep-pruned-shrink" => {
                keep_pruned_shrink = Some(true);
                i += 1;
            }
            "--internal-profile-scale" => {
                internal_profile_scale =
                    Some(parse_next(args, &mut i, "--internal-profile-scale")?);
            }
            "--phase" => {
                let _ = next_str(args, &mut i, "--phase")?;
            }
            "--generate-golden" => {
                generate_golden = true;
                i += 1;
            }
            "--measure-baseline" => {
                measure_baseline = true;
                i += 1;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI flag '{unknown}'"
                )));
            }
        }
    }

    if let Some(scale) = internal_profile_scale {
        let profile = profile_single_sift1m_scale(
            &data_cache_dir,
            scale,
            seed,
            allow_download,
            sweep_runs.max(1),
        )?;
        print!("{}", format_scale_profile_kv(&profile));
        return Ok(());
    }

    if show_help {
        println!(
            "\
HNSW Vector Search (Rust) -- Phase 3: SIFT1M Real-World Scaling & Benchmark Harness

USAGE:
    cargo run --release --bin phase3 -- [OPTIONS]
    cargo run --release --bin phase1 -- --phase 3 [OPTIONS]

OPTIONS:
    --dataset <sift1m|siftsmall|synthetic|custom>
                                Dataset preset to load or download (default: sift1m)
    --data-dir <DIR>           Cache directory for public SIFT corpora (default: data)
    --output-dir <DIR>         Output directory for artifacts (default: data)
    --k <K>                    Top-k nearest neighbors to retrieve (default: 10)
    --seed <SEED>              Deterministic PRNG seed (default: 42)
    --metric <l2|euclidean|ip> Distance metric (default: l2)
    --m <M>                    Max outgoing edges per node at layers > 0 (default: 16)
    --m-max0 <M0>              Max outgoing edges per node at layer 0 (default: 32)
    --ef-construction <EFC>    Candidate beam width during index build (default: 100)
    --ef-search <LIST>         Comma-separated efSearch values to sweep (default: 10,20,50,100,200)
    --sweep-runs <N>           Repetitions per efSearch setting for median & spread (default: 5)
    --no-multi-scale           Skip isolated 10k and 100k child-process baseline profiling
    --max-queries <Q>          Limit search evaluation to first Q queries
    --max-base <N>             Limit index construction to first N base vectors (smoke test)
    --threads <T>              Number of worker threads (default: 0 for all logical cores)
    --json-out <PATH>          Write structured JSON benchmark report to PATH (default: data/phase3_report.json)
    --offline                  Disable internet downloads; use local cache or synthetic
    --no-save                  Skip writing intermediate artifacts to disk
    --repro-check              Verify bit-exact reproducibility across independent builds
    --no-repro-check           Skip reproducibility check
    --repro-vectors <N>        Vector slice size for reproducibility check (default: 10000)
    -h, --help                 Print this help message"
        );
        return Ok(());
    }

    if generate_golden {
        let golden_dir = PathBuf::from("results/phase3_5");
        return generate_golden_fixtures(&data_cache_dir, &golden_dir);
    }

    if measure_baseline {
        let baseline_dir = PathBuf::from("results/phase3_5");
        return run_phase3_5_baseline_measurement(&data_cache_dir, &baseline_dir);
    }

    let dataset_mode = resolve_cli_dataset_mode(
        &dataset_choice,
        base_fvecs,
        query_fvecs,
        gt_ivecs,
        num_base,
        num_query,
        dim,
        num_clusters,
    )?;

    let config = Phase3Config {
        dataset_mode,
        data_cache_dir,
        output_dir,
        k,
        seed,
        metric,
        m,
        m_max0,
        ef_construction,
        ef_search_list,
        allow_download,
        save_files,
        max_queries,
        max_base,
        num_threads,
        run_repro_check,
        repro_vectors,
        sweep_runs,
        run_multi_scale,
        keep_pruned_shrink,
    };

    let report = run_phase3_benchmark(&config)?;
    print_phase3_report(&report);

    let target_json = json_out.unwrap_or_else(|| PathBuf::from("data/phase3_report.json"));
    write_file_with_parents(&target_json, report.to_json_string())?;
    println!("  - JSON Summary Report:   {}", target_json.display());

    if config.save_files
        && !report.multi_scale_baseline_summary.is_empty()
        && report
            .multi_scale_baseline_summary
            .iter()
            .all(|p| p.measured_this_run)
    {
        let baseline_report = BaselineMultiScaleReport {
            hardware: report.hardware.clone(),
            dataset_source: report.dataset_source.clone(),
            hnsw_m: report.hnsw_m,
            hnsw_m_max0: report.hnsw_m_max0,
            hnsw_ef_construction: report.hnsw_ef_construction,
            seed: report.seed,
            metric: report.metric,
            peak_rss_measurement_method: format!(
                "isolated_per_scale_process ({})",
                report.memory_profile.peak_rss_measurement_method
            ),
            scale_profiles: report.multi_scale_baseline_summary.clone(),
        };
        let baseline_json = config.output_dir.join("baseline_profile_report.json");
        write_file_with_parents(&baseline_json, baseline_report.to_json_string())?;
        println!("  - Multi-Scale Report:    {}", baseline_json.display());
    }

    Ok(())
}
