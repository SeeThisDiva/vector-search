//! Phase 7 Head-to-Head Rust HNSW Benchmark Runner (Milestone M5 / Task T5.1).
//!
//! Benchmarks `HnswIndex` (`Metric::L2Squared`, `#![forbid(unsafe_code)]`) head-to-head
//! against FAISS `IndexHNSWFlat` using the canonical Phase 7 JSON schema consumed by
//! `scripts/plot_phase7.py`:
//! - Mode 1 pure-batch throughput via `HnswIndex::search_batch_parallel` (`1` thread in
//!   `single` mode, or `--threads` in `multithread` mode), eliminating per-query `Instant::now()`.
//! - Mode 2 per-query latency distribution (`p50`, `p95`, `p99`, `min`, `max`) via
//!   `HnswIndex::search_batch_timed_with_tracker` (in `single` mode), verifying bit-exact
//!   recall parity with Mode 1 and reporting `ffi_delta_us = 0.0` plus `instant_overhead_us`.
//! - Optional diagnostic `dist_evals_mean` pass outside timed loops when compiled with
//!   `--features instrument` (`null` otherwise).
//! - Prefix `.ivecs` ground-truth caching and `--prepare-100k-gt` generation via
//!   `ExactKnn::search_parallel_with_base`.

#![forbid(unsafe_code)]

#[path = "phase7_parts/cli.rs"]
mod cli;
#[path = "phase7_parts/dataset_gt.rs"]
mod dataset_gt;
#[path = "phase7_parts/report.rs"]
mod report;
#[path = "phase7_parts/runner.rs"]
mod runner;

use std::env;
use std::process::ExitCode;

use vector_search::json_util::write_file_with_parents;

use cli::{parse_cli_args, print_usage, spawn_isolated_child, ParseOutcome};
use dataset_gt::{load_benchmark_dataset, prepare_exact_gt_file};
use report::print_console_summary;
use runner::run_benchmark;

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let cfg = match parse_cli_args(&raw_args) {
        Ok(ParseOutcome::Help) => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Ok(ParseOutcome::Run(cfg)) => cfg,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            return ExitCode::FAILURE;
        }
    };

    if cfg.profile_rss_isolated && !cfg.internal_isolated_child {
        return match spawn_isolated_child(&raw_args) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(code) => {
                eprintln!("[ERROR] Isolated child process exited with code {code}");
                ExitCode::FAILURE
            }
            Err(err) => {
                eprintln!("[ERROR] Failed to spawn isolated child process: {err}");
                ExitCode::FAILURE
            }
        };
    }

    if cfg.prepare_100k_gt {
        return match load_benchmark_dataset(&cfg).and_then(|(loaded, _)| {
            prepare_exact_gt_file(&loaded, &cfg)?;
            Ok(())
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("[ERROR] {err}");
                ExitCode::FAILURE
            }
        };
    }

    match run_benchmark(&cfg) {
        Ok(report) => {
            if let Some(ref json_path) = cfg.json_out {
                let json_str = report.to_json_string();
                if let Err(err) = write_file_with_parents(json_path, &json_str) {
                    eprintln!("[ERROR] Failed to write JSON report: {err}");
                    return ExitCode::FAILURE;
                }
                println!("[Rust Phase 7] Wrote JSON report: {}", json_path.display());
            }
            print_console_summary(&report);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ERROR] {err}");
            ExitCode::FAILURE
        }
    }
}
