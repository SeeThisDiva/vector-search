//! Phase 2 Dedicated CLI Runner: HNSW Approximate Nearest Neighbor Search Benchmark.

#![forbid(unsafe_code)]

use std::env;
use std::process::ExitCode;

use vector_search::benchmark::run_phase2_cli;
use vector_search::error::Result;

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
    run_phase2_cli(&args)
}
