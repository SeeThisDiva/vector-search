//! Phase 4 Parameter Sweep Runner: Resumable HNSW Grid Exploration & Pareto Frontier.
//!
//! Provides:
//! - Declarative grid specification parsing ([`Phase4GridSpec`], [`Phase4IndexConfig`])
//! - Prefix ground-truth disk caching ([`load_or_compute_phase4_ground_truth`])
//! - Resumable run semantics skipping previously measured configs ([`can_resume_existing_report`])
//! - End-to-end multi-configuration sweep execution ([`run_phase4_sweep`])
//! - Dedicated CLI argument parsing and execution ([`run_phase4_cli`])

#![forbid(unsafe_code)]

pub mod cli;
pub mod grid;
pub mod ground_truth;
pub mod report;
pub mod runner;

pub use crate::benchmark::current_utc_rfc3339;
pub use cli::run_phase4_cli;
pub use grid::{Phase4GridSpec, Phase4IndexConfig};
pub use ground_truth::{load_or_compute_phase4_ground_truth, Phase4GroundTruth};
pub use report::{
    can_resume_existing_report, Phase4EfEvaluation, Phase4IndexReport, Phase4IndexSummaryEntry,
    Phase4MemoryProfile, Phase4SweepSummary,
};
pub use runner::{run_phase4_sweep, Phase4SweepConfig};
