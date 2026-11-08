//! Phase 4 Dedicated CLI Runner: Resumable HNSW Parameter Sweep.

#![forbid(unsafe_code)]

use std::env;
use std::process::ExitCode;

use vector_search::error::Result;
use vector_search::phase4::run_phase4_cli;

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
    run_phase4_cli(&args)?;
    Ok(())
}
