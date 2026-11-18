//! Phase 5 Dedicated CLI Runner: Filtered Search Benchmarks, Cost Model Microbenchmark,
//! 100k Pilot Suite, and Resumable Option C Matrix Runner.

#![forbid(unsafe_code)]

use std::process::ExitCode;

use vector_search::phase5::run_phase5_cli;

fn main() -> ExitCode {
    run_phase5_cli()
}
