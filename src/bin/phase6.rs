//! Phase 6 Dedicated CLI Runner (`SQ8` & `PQ` Benchmarks, Kernel Microbenchmark,
//! Isolated Query RSS Profiler, and Cross-Language Parity Hook).

#![forbid(unsafe_code)]

use std::env;
use std::process::ExitCode;

use vector_search::error::Result;
use vector_search::phase6::run_phase6_cli;

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
    run_phase6_cli(&args)
}
