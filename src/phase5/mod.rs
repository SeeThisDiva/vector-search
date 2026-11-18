//! Phase 5 Filtered Search Benchmark Runner, Versioned Cache & Sensitivity Suite (`T3.8`).
//!
//! Implements Gate G4.5 decisions:
//! - Option C query schedules (`reps = 5` default after 1 warmup), recording `num_queries`
//!   and `is_subquery_probe` on every arm and scoping `S1_post_uncapped` strictly to `sigma >= 0.05`.
//! - Versioned & verified disk cache under `data/phase5/` storing `.bin` predicate bitsets,
//!   `.ivecs` ground-truth indices, `.fvecs` ground-truth distances, and `.json` provenance.
//! - Resumable per-cell JSON outputs with zero-overhead latency timing (`COLLECT_COUNTERS = false`)
//!   and separate untimed counter/flag/recall pass (`COLLECT_COUNTERS = true`).
//! - Sensitivity suite covering `H_max in {1, 3, 10}*ef`, `overfetch_factor`, and opaque
//!   `LambdaPredicate` `eval_limit` probes including a 1M spot-check mode.
//! - Matched-recall (`0.95`, `0.99`) crossover bracket and log-log interpolation extraction.

#![forbid(unsafe_code)]

pub mod cache;
pub mod cli;
pub mod crossover;
pub mod metrics;
pub mod report;
pub mod runner;
pub mod sensitivity;

use std::path::PathBuf;

use crate::dataset::S_CLUST_DEFAULT;
use crate::error::{Result, VectorSearchError};
use crate::filter::DEFAULT_OPAQUE_EVAL_LIMIT;

pub use cache::{
    compute_bitset_hash, compute_f32_matrix_hash, dataset_cache_tag,
    load_or_compute_filtered_gt_cache, load_or_generate_predicate_cache, FilteredGtCacheArtifact,
    PredicateCacheArtifact,
};
pub use cli::{measure_t_dist, measure_t_heap, measure_t_word, predict_c_brute_ns, run_phase5_cli};
pub use crossover::{compute_crossover_analysis, CrossoverModeResult, Phase5CellRecord};
pub use metrics::{
    compute_filtered_recall_at_k, measure_arm, measure_opaque_eval_probe, OpaqueEvalProbeRecord,
    Phase5ArmRecord,
};
pub use report::{can_resume_phase5_cell, parse_phase5_cell_json, Phase5Summary};
pub use runner::run_phase5_matrix;
pub use sensitivity::run_1m_opaque_spotcheck;

/// Predicate bitset cache generator schema version.
pub const PRED_GEN_VERSION: u32 = 1;
/// Filtered exact ground truth cache schema version.
pub const GT_GEN_VERSION: u32 = 1;
/// Default top-k count evaluated across Phase 5.
pub const DEFAULT_K: usize = 10;
/// Default deterministic seed for attribute / shuffle generation.
pub const DEFAULT_ATTR_SEED: u64 = 42;
/// Default timed repetition count after 1 untimed warmup pass (Gate G4.5 Decision 1).
pub const DEFAULT_REPS: usize = 5;

/// Canonical reference HNSW build parameter `M` for Phase 5 (`PROPOSAL.md`).
pub const HNSW_M: usize = 16;
/// Canonical reference HNSW build parameter `efConstruction` for Phase 5 (`PROPOSAL.md`).
pub const HNSW_EF_CONSTRUCTION: usize = 100;
/// Canonical reference HNSW build RNG seed.
pub const HNSW_BUILD_SEED: u64 = 42;

/// Default capped over-fetch candidate pool size $K'_{\max}$ for Strategy 1.
pub const S1_OVERFETCH_CAP: usize = 4_096;
/// Minimum selectivity where Strategy 1 uncapped ($K'_{\max} = N$) is evaluated under Option C.
pub const S1_UNCAPPED_SIGMA_MIN: f64 = 0.05;

/// Full 12-point logarithmic selectivity ladder (`DESIGN.md` §4.3).
pub const SELECTIVITY_LADDER_12: [f64; 12] = [
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.20, 0.50,
];
/// Standard 4-tier smoke selectivity ladder (`50%, 10%, 1%, 0.1%`).
pub const SMOKE_SELECTIVITY_LADDER: [f64; 4] = [0.50, 0.10, 0.01, 0.001];

/// Strategy 2 `efSearch` ladder (`DESIGN.md` §4.3).
pub const S2_EF_LADDER: [usize; 6] = [20, 50, 100, 200, 400, 600];
/// Strategy 1 capped `efSearch` ladder at high selectivity ($\sigma \ge 5\%$).
pub const S1_CAP_EF_HIGH_SIGMA: [usize; 3] = [20, 100, 600];
/// Strategy 1 capped `efSearch` ladder below $5\%$.
pub const S1_CAP_EF_LOW_SIGMA: [usize; 1] = [100];

/// Primary matched-recall crossover target (`DESIGN.md` §4.3).
pub const RECALL_TARGET_PRIMARY: f64 = 0.95;
/// Secondary matched-recall crossover target (`DESIGN.md` §4.3).
pub const RECALL_TARGET_SECONDARY: f64 = 0.99;

/// Bridge-hop multiplier sweep values (`H_max = mult * ef`) in the sensitivity suite.
pub const SENSITIVITY_HMAX_MULTS: [usize; 3] = [1, 3, 10];
/// `efSearch` values swept in `H_max` sensitivity.
pub const SENSITIVITY_HMAX_EFS: [usize; 3] = [20, 100, 400];
/// Selectivity tiers swept in `H_max` sensitivity.
pub const SENSITIVITY_HMAX_SIGMAS: [f64; 3] = [0.001, 0.01, 0.10];

/// Over-fetch factor sweep values (`None` uses confidence formula $\delta = 0.001$).
pub const SENSITIVITY_GAMMA_VALUES: [Option<f64>; 4] = [None, Some(1.0), Some(2.0), Some(5.0)];
/// Selectivity tiers swept in `overfetch_factor` sensitivity.
pub const SENSITIVITY_GAMMA_SIGMAS: [f64; 2] = [0.01, 0.10];

/// Opaque predicate `eval_limit` policies compared in sensitivity probes.
pub const SENSITIVITY_EVAL_LIMITS: [Option<usize>; 3] =
    [None, Some(DEFAULT_OPAQUE_EVAL_LIMIT), Some(10_000)];

/// Spatial distribution mode for synthetic metadata attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetadataMode {
    /// Mode A: Uniform independent random sampling (`generate_mode_a_bitset`).
    Independent,
    /// Mode B: $K$-means++ Voronoi cluster-correlated sampling (`generate_mode_b_bitset_detailed`).
    Correlated,
}

impl MetadataMode {
    /// Canonical lowercase string representation used in file names and JSON reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Correlated => "correlated",
        }
    }

    /// Parses mode name (`independent`/`mode_a`/`a` or `correlated`/`mode_b`/`b`).
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "independent" | "mode_a" | "a" => Ok(Self::Independent),
            "correlated" | "mode_b" | "b" => Ok(Self::Correlated),
            _ => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid metadata mode '{raw}' (expected 'independent' or 'correlated')"
            ))),
        }
    }
}

/// Canonical filename slug for selectivity `sigma` (e.g. `s0.100000`).
#[must_use]
pub fn sigma_slug(sigma: f64) -> String {
    format!("s{sigma:.6}")
}

/// Canonical cell JSON filename under `output_dir`.
#[must_use]
pub fn cell_report_filename(scale: usize, mode: MetadataMode, sigma: f64) -> String {
    format!("cell_n{scale}_{}_{}.json", mode.as_str(), sigma_slug(sigma))
}

/// Configuration for the Phase 5 filtered search matrix and sensitivity runner.
#[derive(Debug, Clone)]
pub struct Phase5MatrixConfig {
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub scale: usize,
    pub modes: Vec<MetadataMode>,
    pub sigmas: Vec<f64>,
    pub k: usize,
    pub reps: usize,
    pub attr_seed: u64,
    pub clust_seed: u64,
    pub synthetic: bool,
    pub synthetic_dim: usize,
    pub synthetic_queries: Option<usize>,
    pub allow_download: bool,
    pub smoke: bool,
    pub resume: bool,
    pub include_sensitivity: bool,
    pub sensitivity_only: bool,
    pub include_1m_opaque_spotcheck: bool,
    pub opaque_1m_spotcheck_only: bool,
    pub gt_threads: usize,
}

impl Default for Phase5MatrixConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("results/phase5"),
            scale: 100_000,
            modes: vec![MetadataMode::Independent, MetadataMode::Correlated],
            sigmas: SELECTIVITY_LADDER_12.to_vec(),
            k: DEFAULT_K,
            reps: DEFAULT_REPS,
            attr_seed: DEFAULT_ATTR_SEED,
            clust_seed: S_CLUST_DEFAULT,
            synthetic: false,
            synthetic_dim: 128,
            synthetic_queries: None,
            allow_download: true,
            smoke: false,
            resume: true,
            include_sensitivity: false,
            sensitivity_only: false,
            include_1m_opaque_spotcheck: false,
            opaque_1m_spotcheck_only: false,
            gt_threads: 0,
        }
    }
}

/// Returns `(num_queries, is_subquery_probe)` for Strategy 3 (`S3_pre_filter`) under Option C.
///
/// - At `scale <= 10_000` (10k smoke or tiny synthetic fixtures), evaluates all `available_queries`.
/// - At `100k`, evaluates full `10,000` queries across the entire ladder.
/// - At `1M`, evaluates `10,000` for `sigma <= 0.05` and `2,000` sub-query probe for `sigma >= 0.075`.
#[must_use]
pub fn option_c_s3_query_count(
    scale: usize,
    sigma: f64,
    available_queries: usize,
) -> (usize, bool) {
    if scale <= 10_000 {
        return (available_queries, false);
    }
    let desired = if scale >= 1_000_000 && sigma >= 0.074_999 {
        2_000
    } else {
        10_000
    };
    let take = desired.min(available_queries);
    (take, take < available_queries)
}

/// Returns `(num_queries, is_subquery_probe)` for HNSW arms (`S2`, `S1_cap`, `S1_uncap`) under Option C.
///
/// - `scale <= 10_000`: all `available_queries` (`is_subquery_probe = false`).
/// - `sigma <= 1%`: `300` queries.
/// - `1% < sigma <= 10%`: `1,000` queries.
/// - `sigma >= 20%`: `2,000` queries.
#[must_use]
pub fn option_c_hnsw_query_count(
    scale: usize,
    sigma: f64,
    available_queries: usize,
) -> (usize, bool) {
    if scale <= 10_000 {
        return (available_queries, false);
    }
    let desired = if sigma <= 0.010_001 {
        300
    } else if sigma <= 0.100_001 {
        1_000
    } else {
        2_000
    };
    let take = desired.min(available_queries);
    (take, take < available_queries)
}

/// Returns `true` when Strategy 1 uncapped (`overfetch_cap = N`) is evaluated (`sigma >= 0.05`).
#[must_use]
pub fn should_evaluate_s1_uncapped(sigma: f64) -> bool {
    sigma + 1e-12 >= S1_UNCAPPED_SIGMA_MIN
}
