//! Phase 6 Vector Quantization (`SQ8` & `PQ`) Benchmark Runner, Isolated Query-Serving
//! RSS Profiler, Kernel Microbenchmark, and Cross-Language Parity CLI (`T4.7`).
//!
//! Strictly follows `results/phase6/DESIGN.md`, `results/phase6/G5_DECISIONS.md`, and the
//! JSON contract consumed by `scripts/plot_phase6.py`:
//! - Per-arm reports written as `arm_n{scale}_{arm_id}.json` plus `summary_n{scale}.json`
//! - Resumable runs skipping clean-commit (`git_dirty: false`) matching reports
//! - Streaming `.fvecs` build for `sq8_only_qbuild` / `pq16_only_qbuild`
//! - Isolated child-process query-serving `PeakWorkingSet64` profiling (`--internal-profile-query`)
//! - Kernel microbenchmark (`Way A` vs `Way B` vs `PQ16` vs `PQ32` vs raw `f32`)
//! - `--parity-check` exporting deterministic SQ8/PQ outputs for NumPy parity tests

#![forbid(unsafe_code)]

pub mod config;
pub mod dataset;
pub mod microbench;
pub mod parity;
pub mod report;
pub mod rss_profile;
pub mod runner;

use crate::error::Result;

pub use config::{
    arm_report_filename, parse_phase6_cli_args, phase6_usage_text, resolve_arm_spec, Phase6ArmKind,
    Phase6ArmSpec, Phase6CliCommand, Phase6DatasetChoice, Phase6InternalQueryProfileConfig,
    Phase6MatrixConfig, Phase6MicrobenchConfig, Phase6ParityConfig, Phase6RunMode, RerankDepthSpec,
    SnapshotKind, CANONICAL_MATRIX_ARM_IDS, DEFAULT_EF_SEARCH_LIST, DEFAULT_PQ_RERANK_LADDER,
    DEFAULT_SQ8_RERANK_LADDER, PILOT_ARM_IDS,
};
pub use microbench::{run_phase6_microbench, KernelMicrobenchResult, Phase6MicrobenchReport};
pub use parity::run_phase6_parity_check;
pub use report::{
    can_resume_existing_arm_report, Phase6ArmSummaryEntry, Phase6MatrixSummary,
    Phase6SweepPointResult,
};
pub use rss_profile::run_phase6_internal_profile_query;
pub use runner::run_phase6_matrix;

/// Parses CLI args and executes the requested Phase 6 workflow.
pub fn run_phase6_cli(args: &[String]) -> Result<()> {
    match parse_phase6_cli_args(args)? {
        Phase6CliCommand::Help => {
            println!("{}", phase6_usage_text());
            Ok(())
        }
        Phase6CliCommand::Microbench(cfg) => {
            let _ = run_phase6_microbench(&cfg)?;
            Ok(())
        }
        Phase6CliCommand::Matrix(cfg) => {
            let summary = run_phase6_matrix(&cfg)?;
            eprintln!(
                "[DONE] Phase 6 {}: scale={}, total_arms={}, newly_measured={}, resumed={}, summary={}",
                if cfg.is_pilot { "pilot" } else { "matrix" },
                summary.scale,
                summary.total_arms,
                summary.newly_measured_count,
                summary.resumed_skipped_count,
                summary.summary_path.display()
            );
            Ok(())
        }
        Phase6CliCommand::ParityCheck(cfg) => run_phase6_parity_check(&cfg),
        Phase6CliCommand::InternalProfileQuery(cfg) => run_phase6_internal_profile_query(&cfg),
    }
}
