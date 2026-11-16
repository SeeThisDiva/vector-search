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

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::benchmark::{
    append_hardware_json, current_utc_rfc3339, detect_provenance, escape_json, opt_path_json,
    summarize_repeated_latencies, RepeatedSweepSummary, ReportProvenance,
};
use crate::dataset::{
    generate_synthetic_sift, load_or_download_sift_with_options, read_fvecs, read_fvecs_max_rows,
    read_ivecs, read_ivecs_max_rows, write_fvecs, write_ivecs, DatasetSource, SiftPreset,
};
use crate::distance::{l2_squared, Metric};
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{detect_hardware, get_peak_rss_mb, HardwareInfo};
use crate::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use crate::matrix::RowMatrix;
use crate::quantized::{
    l2_squared_sq8_expanded, l2_squared_sq8_shifted, pq_adc_distance, results_to_knn_search_result,
    HnswPqIndex, HnswSq8Index, PqConfig, PqQueryState, PqVectors, QuantizedBuildMode,
    QuantizedDistanceSource, QuantizedMemoryProfile, QuantizedSearchScratch, QuantizedStorageMode,
    Sq8KernelKind, Sq8QueryState, Sq8Vectors, TimedQuantizedSearch, PQ_DEFAULT_MAX_LLOYD_ITERS,
    PQ_DEFAULT_MAX_TRAIN_VECTORS, PQ_K_CENTROIDS,
};
use crate::recall::{compute_recall_at_k, percentile_sorted, LatencyStats};
use crate::rng::DeterministicRng;

/// Default `efSearch` ladder evaluated across Phase 6 sweeps.
pub const DEFAULT_EF_SEARCH_LIST: [usize; 12] =
    [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600];

/// Default SQ8 top-`R` rerank ladder (`R in {20, 30, 50, 100}` plus `FullBeam`).
pub const DEFAULT_SQ8_RERANK_LADDER: [RerankDepthSpec; 5] = [
    RerankDepthSpec::Fixed(20),
    RerankDepthSpec::Fixed(30),
    RerankDepthSpec::Fixed(50),
    RerankDepthSpec::Fixed(100),
    RerankDepthSpec::FullBeam,
];

/// Default PQ top-`R` rerank ladder (`R in {30, 50, 100, 200}` plus `FullBeam`).
pub const DEFAULT_PQ_RERANK_LADDER: [RerankDepthSpec; 5] = [
    RerankDepthSpec::Fixed(30),
    RerankDepthSpec::Fixed(50),
    RerankDepthSpec::Fixed(100),
    RerankDepthSpec::Fixed(200),
    RerankDepthSpec::FullBeam,
];

/// Canonical full evaluation matrix arm IDs (`G5_DECISIONS.md` + `DESIGN.md` §5.3).
pub const CANONICAL_MATRIX_ARM_IDS: [&str; 11] = [
    "f32_m16",
    "sq8_only_f32graph",
    "sq8_only_qbuild",
    "sq8_rerank_f32graph",
    "sq8_only_m32",
    "sq8_rerank_m32",
    "pq16_only_f32graph",
    "pq32_only_f32graph",
    "pq16_only_qbuild",
    "pq16_rerank_f32graph",
    "pq32_rerank_f32graph",
];

/// Representative `M=16, efC=100` pilot arm IDs (`T4.6`).
pub const PILOT_ARM_IDS: [&str; 8] = [
    "f32_m16",
    "sq8_only_f32graph",
    "sq8_only_qbuild",
    "sq8_rerank_f32graph",
    "pq16_only_f32graph",
    "pq32_only_f32graph",
    "pq16_rerank_f32graph",
    "pq32_rerank_f32graph",
];

/// Top-level Phase 6 CLI subcommand mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase6RunMode {
    Microbench,
    Pilot,
    Matrix,
    ParityCheck,
    InternalProfileQuery,
}

/// Public SIFT preset selector for Phase 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase6DatasetChoice {
    Sift1M,
    SiftSmall,
}

impl Phase6DatasetChoice {
    /// Parses `sift1m` or `siftsmall`.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sift1m" => Ok(Self::Sift1M),
            "siftsmall" => Ok(Self::SiftSmall),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid --dataset '{other}': expected 'sift1m' or 'siftsmall'"
            ))),
        }
    }

    #[must_use]
    pub const fn to_preset(self) -> SiftPreset {
        match self {
            Self::Sift1M => SiftPreset::Sift1M,
            Self::SiftSmall => SiftPreset::SiftSmall,
        }
    }
}

/// Snapshot quantizer kind used by `--internal-profile-query`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotKind {
    Sq8,
    Pq,
}

impl SnapshotKind {
    /// Parses `sq8` or `pq`.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sq8" => Ok(Self::Sq8),
            "pq" | "pq16" | "pq32" => Ok(Self::Pq),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid --kind '{other}': expected 'sq8' or 'pq'"
            ))),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sq8 => "sq8",
            Self::Pq => "pq",
        }
    }
}

/// Rerank depth specification within a Phase 6 sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankDepthSpec {
    None,
    Fixed(usize),
    FullBeam,
}

impl RerankDepthSpec {
    /// Parses a `--rerank-list` element (`none`, `full_beam`/`full`, or positive integer `R`).
    pub fn parse(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        let lower = trimmed.to_ascii_lowercase();
        match lower.as_str() {
            "none" | "null" => Ok(Self::None),
            "full_beam" | "fullbeam" | "full" => Ok(Self::FullBeam),
            _ => {
                let r = trimmed.parse::<usize>().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Invalid rerank depth '{trimmed}': {e}"
                    ))
                })?;
                if r == 0 {
                    return Err(VectorSearchError::InvalidArgument(
                        "Rerank depth R must be >= 1 (use 'none' to disable reranking)".to_string(),
                    ));
                }
                Ok(Self::Fixed(r))
            }
        }
    }

    /// Resolves `(rerank_top_r_arg, effective_ef_search)` for query execution.
    #[inline]
    #[must_use]
    pub fn resolve_params(self, k: usize, ef_search: usize) -> (Option<usize>, usize) {
        match self {
            Self::None => (None, ef_search.max(k)),
            Self::Fixed(r) => (Some(r), ef_search.max(r).max(k)),
            Self::FullBeam => {
                let beam = ef_search.max(k);
                (Some(beam), beam)
            }
        }
    }

    /// JSON literal representation for `"rerank_top_r"`.
    #[must_use]
    pub fn json_rerank_top_r(self) -> String {
        match self {
            Self::None => "null".to_string(),
            Self::Fixed(r) => r.to_string(),
            Self::FullBeam => "\"full_beam\"".to_string(),
        }
    }

    /// Whether this operating point represents `FullBeam` (`R = ef_search`).
    #[inline]
    #[must_use]
    pub const fn is_full_beam(self) -> bool {
        matches!(self, Self::FullBeam)
    }
}

/// Quantizer implementation family for an evaluation arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase6ArmKind {
    F32,
    Sq8,
    Pq { m_pq: usize },
}

/// Full specification of a single Phase 6 benchmark arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase6ArmSpec {
    pub arm_id: &'static str,
    pub kind: Phase6ArmKind,
    pub quantizer_kind: &'static str,
    pub storage_mode: &'static str,
    pub build_mode_str: &'static str,
    pub build_mode: QuantizedBuildMode,
    pub storage_enum: QuantizedStorageMode,
    pub m: usize,
    pub ef_construction: usize,
    pub default_reranks: &'static [RerankDepthSpec],
}

impl Phase6ArmSpec {
    /// Returns `true` when this arm supports streaming `.fvecs` build without materializing `RowMatrix<f32>`.
    #[inline]
    #[must_use]
    pub const fn supports_streaming_build(&self) -> bool {
        matches!(
            (self.kind, self.build_mode, self.storage_enum),
            (
                Phase6ArmKind::Sq8 | Phase6ArmKind::Pq { .. },
                QuantizedBuildMode::QuantizedDirect,
                QuantizedStorageMode::QuantizedOnly
            )
        )
    }

    /// Returns the snapshot kind if this is a `QuantizedOnly` arm eligible for isolated query RSS profiling.
    #[inline]
    #[must_use]
    pub const fn isolated_snapshot_kind(&self) -> Option<SnapshotKind> {
        if !matches!(self.storage_enum, QuantizedStorageMode::QuantizedOnly) {
            return None;
        }
        match self.kind {
            Phase6ArmKind::F32 => None,
            Phase6ArmKind::Sq8 => Some(SnapshotKind::Sq8),
            Phase6ArmKind::Pq { .. } => Some(SnapshotKind::Pq),
        }
    }

    /// Resolves the active rerank ladder for this arm given an optional `--rerank-list` override.
    #[must_use]
    pub fn active_rerank_ladder(
        &self,
        override_list: Option<&[RerankDepthSpec]>,
    ) -> Vec<RerankDepthSpec> {
        if !matches!(self.kind, Phase6ArmKind::F32)
            && matches!(self.storage_enum, QuantizedStorageMode::WithExactRerank)
        {
            if let Some(custom) = override_list {
                return custom.to_vec();
            }
        }
        self.default_reranks.to_vec()
    }
}

const SINGLE_NONE_RERANK: [RerankDepthSpec; 1] = [RerankDepthSpec::None];

/// Resolves a canonical Phase 6 arm ID into its [`Phase6ArmSpec`].
pub fn resolve_arm_spec(raw_id: &str) -> Result<Phase6ArmSpec> {
    let norm = raw_id.trim();
    match norm {
        "f32_m16" => Ok(Phase6ArmSpec {
            arm_id: "f32_m16",
            kind: Phase6ArmKind::F32,
            quantizer_kind: "f32",
            storage_mode: "f32",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::WithExactRerank,
            m: 16,
            ef_construction: 100,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "sq8_only_f32graph" => Ok(Phase6ArmSpec {
            arm_id: "sq8_only_f32graph",
            kind: Phase6ArmKind::Sq8,
            quantizer_kind: "sq8",
            storage_mode: "only",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::QuantizedOnly,
            m: 16,
            ef_construction: 100,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "sq8_only_qbuild" => Ok(Phase6ArmSpec {
            arm_id: "sq8_only_qbuild",
            kind: Phase6ArmKind::Sq8,
            quantizer_kind: "sq8",
            storage_mode: "only",
            build_mode_str: "quantized_direct",
            build_mode: QuantizedBuildMode::QuantizedDirect,
            storage_enum: QuantizedStorageMode::QuantizedOnly,
            m: 16,
            ef_construction: 100,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "sq8_rerank_f32graph" => Ok(Phase6ArmSpec {
            arm_id: "sq8_rerank_f32graph",
            kind: Phase6ArmKind::Sq8,
            quantizer_kind: "sq8",
            storage_mode: "rerank",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::WithExactRerank,
            m: 16,
            ef_construction: 100,
            default_reranks: &DEFAULT_SQ8_RERANK_LADDER,
        }),
        "sq8_only_m32" => Ok(Phase6ArmSpec {
            arm_id: "sq8_only_m32",
            kind: Phase6ArmKind::Sq8,
            quantizer_kind: "sq8",
            storage_mode: "only",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::QuantizedOnly,
            m: 32,
            ef_construction: 200,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "sq8_rerank_m32" => Ok(Phase6ArmSpec {
            arm_id: "sq8_rerank_m32",
            kind: Phase6ArmKind::Sq8,
            quantizer_kind: "sq8",
            storage_mode: "rerank",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::WithExactRerank,
            m: 32,
            ef_construction: 200,
            default_reranks: &DEFAULT_SQ8_RERANK_LADDER,
        }),
        "pq16_only_f32graph" => Ok(Phase6ArmSpec {
            arm_id: "pq16_only_f32graph",
            kind: Phase6ArmKind::Pq { m_pq: 16 },
            quantizer_kind: "pq16",
            storage_mode: "only",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::QuantizedOnly,
            m: 16,
            ef_construction: 100,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "pq32_only_f32graph" => Ok(Phase6ArmSpec {
            arm_id: "pq32_only_f32graph",
            kind: Phase6ArmKind::Pq { m_pq: 32 },
            quantizer_kind: "pq32",
            storage_mode: "only",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::QuantizedOnly,
            m: 16,
            ef_construction: 100,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "pq16_only_qbuild" => Ok(Phase6ArmSpec {
            arm_id: "pq16_only_qbuild",
            kind: Phase6ArmKind::Pq { m_pq: 16 },
            quantizer_kind: "pq16",
            storage_mode: "only",
            build_mode_str: "quantized_direct",
            build_mode: QuantizedBuildMode::QuantizedDirect,
            storage_enum: QuantizedStorageMode::QuantizedOnly,
            m: 16,
            ef_construction: 100,
            default_reranks: &SINGLE_NONE_RERANK,
        }),
        "pq16_rerank_f32graph" => Ok(Phase6ArmSpec {
            arm_id: "pq16_rerank_f32graph",
            kind: Phase6ArmKind::Pq { m_pq: 16 },
            quantizer_kind: "pq16",
            storage_mode: "rerank",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::WithExactRerank,
            m: 16,
            ef_construction: 100,
            default_reranks: &DEFAULT_PQ_RERANK_LADDER,
        }),
        "pq32_rerank_f32graph" => Ok(Phase6ArmSpec {
            arm_id: "pq32_rerank_f32graph",
            kind: Phase6ArmKind::Pq { m_pq: 32 },
            quantizer_kind: "pq32",
            storage_mode: "rerank",
            build_mode_str: "f32_graph",
            build_mode: QuantizedBuildMode::Float32GraphThenQuantize,
            storage_enum: QuantizedStorageMode::WithExactRerank,
            m: 16,
            ef_construction: 100,
            default_reranks: &DEFAULT_PQ_RERANK_LADDER,
        }),
        other => Err(VectorSearchError::InvalidArgument(format!(
            "Unknown Phase 6 arm_id '{other}'. Valid arms: {}",
            CANONICAL_MATRIX_ARM_IDS.join(", ")
        ))),
    }
}

/// Canonical per-arm JSON filename (`arm_n{scale}_{arm_id}.json`).
#[must_use]
pub fn arm_report_filename(scale: usize, arm_id: &str) -> String {
    format!("arm_n{scale}_{arm_id}.json")
}

/// Configuration for `--matrix` and `--pilot` runs.
#[derive(Debug, Clone)]
pub struct Phase6MatrixConfig {
    pub is_pilot: bool,
    pub dataset: Phase6DatasetChoice,
    pub base_fvecs: Option<PathBuf>,
    pub query_fvecs: Option<PathBuf>,
    pub gt_path: Option<PathBuf>,
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub scale: usize,
    pub scale_explicit: bool,
    pub queries: usize,
    pub queries_explicit: bool,
    pub k: usize,
    pub reps: usize,
    pub reps_explicit: bool,
    pub seed: u64,
    pub dim: usize,
    pub num_clusters: usize,
    pub arm_ids: Vec<String>,
    pub ef_search_list: Vec<usize>,
    pub rerank_list_override: Option<Vec<RerankDepthSpec>>,
    pub resume: bool,
    pub synthetic: bool,
    pub offline: bool,
    pub enable_gt_cache: bool,
    pub profile_query_rss: bool,
    pub affinity: Option<String>,
    pub expected_commit_override: Option<String>,
}

impl Default for Phase6MatrixConfig {
    fn default() -> Self {
        Self {
            is_pilot: false,
            dataset: Phase6DatasetChoice::Sift1M,
            base_fvecs: None,
            query_fvecs: None,
            gt_path: None,
            data_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("results/phase6"),
            scale: 100_000,
            scale_explicit: false,
            queries: 10_000,
            queries_explicit: false,
            k: 10,
            reps: 3,
            reps_explicit: false,
            seed: 42,
            dim: 128,
            num_clusters: 16,
            arm_ids: CANONICAL_MATRIX_ARM_IDS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            ef_search_list: DEFAULT_EF_SEARCH_LIST.to_vec(),
            rerank_list_override: None,
            resume: true,
            synthetic: false,
            offline: false,
            enable_gt_cache: true,
            profile_query_rss: true,
            affinity: None,
            expected_commit_override: None,
        }
    }
}

impl Phase6MatrixConfig {
    /// Validates matrix/pilot configuration invariants.
    pub fn validate(&self) -> Result<()> {
        if self.scale == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--scale must be positive (> 0)".to_string(),
            ));
        }
        if self.queries == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--queries must be positive (> 0)".to_string(),
            ));
        }
        if self.k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--k must be positive (> 0)".to_string(),
            ));
        }
        if self.reps == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--reps must be positive (> 0)".to_string(),
            ));
        }
        if self.dim == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--dim must be positive (> 0)".to_string(),
            ));
        }
        if self.num_clusters == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--num-clusters must be positive (> 0)".to_string(),
            ));
        }
        if self.ef_search_list.is_empty() || self.ef_search_list.contains(&0) {
            return Err(VectorSearchError::InvalidArgument(
                "--ef-search-list must contain positive integers (>= 1)".to_string(),
            ));
        }
        if self.arm_ids.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "--arms cannot be empty".to_string(),
            ));
        }
        for id in &self.arm_ids {
            let _ = resolve_arm_spec(id)?;
        }
        if self.base_fvecs.is_some() != self.query_fvecs.is_some() {
            return Err(VectorSearchError::InvalidArgument(
                "--base-fvecs and --query-fvecs must be provided together".to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration for `--microbench`.
#[derive(Debug, Clone)]
pub struct Phase6MicrobenchConfig {
    pub output_dir: PathBuf,
    pub output_path: Option<PathBuf>,
    pub num_vectors: usize,
    pub num_evals: usize,
    pub dim: usize,
    pub reps: usize,
    pub seed: u64,
    pub affinity: Option<String>,
}

impl Default for Phase6MicrobenchConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("results/phase6"),
            output_path: None,
            num_vectors: 4_096,
            num_evals: 200_000,
            dim: 128,
            reps: 5,
            seed: 42,
            affinity: None,
        }
    }
}

impl Phase6MicrobenchConfig {
    /// Resolves output JSON file path (default `<output-dir>/microbench_kernels.json`).
    #[must_use]
    pub fn resolved_output_path(&self) -> PathBuf {
        self.output_path
            .clone()
            .unwrap_or_else(|| self.output_dir.join("microbench_kernels.json"))
    }
}

/// Configuration for `--parity-check`.
#[derive(Debug, Clone)]
pub struct Phase6ParityConfig {
    pub base_fvecs: PathBuf,
    pub query_fvecs: PathBuf,
    pub m_pq: usize,
    pub k: usize,
    pub ef: usize,
    pub seed: u64,
    pub max_train_vectors: usize,
    pub max_lloyd_iters: usize,
    pub json_out: Option<PathBuf>,
}

/// Configuration for `--internal-profile-query`.
#[derive(Debug, Clone)]
pub struct Phase6InternalQueryProfileConfig {
    pub snapshot_path: PathBuf,
    pub kind: SnapshotKind,
    pub query_fvecs: Option<PathBuf>,
    pub synthetic: bool,
    pub num_queries: usize,
    pub dim: usize,
    pub k: usize,
    pub ef: usize,
    pub seed: u64,
    pub affinity: Option<String>,
}

/// Parsed Phase 6 CLI command ready for dispatch.
#[derive(Debug, Clone)]
pub enum Phase6CliCommand {
    Help,
    Microbench(Phase6MicrobenchConfig),
    Matrix(Box<Phase6MatrixConfig>),
    ParityCheck(Phase6ParityConfig),
    InternalProfileQuery(Phase6InternalQueryProfileConfig),
}

/// Single measured point inside an arm's `sweep[]` array.
#[derive(Debug, Clone)]
pub struct Phase6SweepPointResult {
    pub ef_search: usize,
    pub effective_ef_search: usize,
    pub rerank_depth: RerankDepthSpec,
    pub recall_at_k: f64,
    pub summary: RepeatedSweepSummary,
}

/// Summary entry recorded for one arm in `summary_n{scale}.json`.
#[derive(Debug, Clone)]
pub struct Phase6ArmSummaryEntry {
    pub arm_id: String,
    pub status: String,
    pub report_filename: String,
    pub quantizer_kind: String,
    pub storage_mode: String,
    pub build_mode: String,
    pub m: usize,
    pub ef_construction: usize,
    pub graph_fingerprint_hex: Option<String>,
    pub total_build_time_s: Option<f64>,
    pub steady_state_bytes_per_vector: Option<f64>,
    pub build_peak_rss_mb: Option<f64>,
    pub query_serving_peak_rss_mb: Option<f64>,
    pub best_recall: Option<f64>,
    pub best_qps: Option<f64>,
}

/// Summary of a complete `--matrix` or `--pilot` invocation.
#[derive(Debug, Clone)]
pub struct Phase6MatrixSummary {
    pub scale: usize,
    pub total_arms: usize,
    pub newly_measured_count: usize,
    pub resumed_skipped_count: usize,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub ground_truth_source: String,
    pub ground_truth_cache_hit: bool,
    pub arms: Vec<Phase6ArmSummaryEntry>,
}

/// Single kernel measurement inside `microbench_kernels.json`.
#[derive(Debug, Clone)]
pub struct KernelMicrobenchResult {
    pub kernel_id: String,
    pub label: String,
    pub bytes_per_vector: usize,
    pub median_ns_per_eval: f64,
    pub min_ns_per_eval: f64,
    pub max_ns_per_eval: f64,
    pub spread_ns_per_eval: f64,
    pub evals_per_sec: f64,
    pub run_ns_per_eval: Vec<f64>,
}

/// Complete kernel microbenchmark report (`Way A` vs `Way B` vs `PQ16` vs `PQ32` vs `f32`).
#[derive(Debug, Clone)]
pub struct Phase6MicrobenchReport {
    pub output_path: PathBuf,
    pub dim: usize,
    pub num_vectors: usize,
    pub num_evals: usize,
    pub reps: usize,
    pub seed: u64,
    pub kernels: Vec<KernelMicrobenchResult>,
    pub standard_max_rel_err_way_a: f64,
    pub standard_max_rel_err_way_b: f64,
    pub adversarial_offset_max_abs_err_way_a: f64,
    pub adversarial_offset_max_abs_err_way_b: f64,
    pub speedup_way_b_over_way_a: f64,
    pub recommended_default_kernel: String,
}

/// Prints Phase 6 CLI usage information.
#[must_use]
pub fn phase6_usage_text() -> &'static str {
    r#"Phase 6 Vector Quantization (SQ8 & PQ) Benchmark & Parity CLI (Milestone M4)

USAGE:
    phase6 [MODE] [OPTIONS]

MODES:
    --microbench                   Benchmark SQ8 Way A vs Way B, PQ16, PQ32, and f32 kernels (default)
    --pilot                        Run fast representative M=16 pilot suite (default output: results/phase6/pilot)
    --matrix                       Run full resumable Phase 6 evaluation matrix (default output: results/phase6)
    --parity-check                 Export SQ8 + PQ encodings, distances, and top-k IDs for NumPy parity
    --internal-profile-query <P>   Internal isolated query-serving RSS profiler child mode
    --help, -h                     Print help information

MATRIX / PILOT OPTIONS:
    --scale <USIZE>                Base vector count N (default: 100000; 10000 for --dataset siftsmall)
    --queries <USIZE>              Query vector count Q (default: 10000; 100 for --dataset siftsmall)
    --k <USIZE>                    Top-k neighbors (default: 10)
    --reps <USIZE>                 Timed repetitions after 1 warmup (default: 3 at <1M, 5 at >=1M)
    --seed <U64>                   Deterministic PRNG seed (default: 42)
    --arms <CSV>                   Subset of canonical arm IDs:
                                     f32_m16, sq8_only_f32graph, sq8_only_qbuild, sq8_rerank_f32graph,
                                     sq8_only_m32, sq8_rerank_m32, pq16_only_f32graph, pq32_only_f32graph,
                                     pq16_only_qbuild, pq16_rerank_f32graph, pq32_rerank_f32graph
    --ef-search-list <CSV>         Comma-separated efSearch values (default: 10,20,30,40,50,70,100,150,200,300,400,600)
    --rerank-list <CSV>            Override rerank ladder for rerank arms (e.g. 20,30,50,100,full_beam)
    --data-dir <PATH>              Dataset and GT cache directory (default: data)
    --output-dir <PATH>            Output directory for arm JSONs (default: results/phase6 or results/phase6/pilot)
    --dataset <sift1m|siftsmall>   Authentic SIFT corpus preset (default: sift1m)
    --base-fvecs <PATH>            Custom base .fvecs input
    --query-fvecs <PATH>           Custom query .fvecs input
    --gt-path <PATH>               Explicit .ivecs ground truth path
    --synthetic                    Use deterministic synthetic clustered data (never silent fallback)
    --dim <USIZE>                  Dimensionality for --synthetic or --microbench (default: 128)
    --num-clusters <USIZE>         Cluster count for --synthetic (default: 16)
    --offline                      Disallow network download of SIFT files
    --no-gt-cache                  Disable disk caching of gt_<scale>_k100.ivecs
    --resume                       Skip matching clean-commit reports (default)
    --no-resume                    Re-measure all selected arms
    --profile-query-rss            Profile QuantizedOnly query RSS in isolated subprocess (default: on)
    --no-isolated-query-rss        Disable isolated child-process query RSS profiling
    --affinity <MASK>              Optional Windows CPU affinity mask (e.g. 0x0004)

MICROBENCH OPTIONS:
    --output, --json-out, -o <P>   Output JSON path (default: <output-dir>/microbench_kernels.json)
    --evals, --dists <USIZE>       Distance evaluations per repetition (default: 200000)
    --num-vectors, --scale <N>     Base pool vector count (default: 4096)

PARITY-CHECK OPTIONS:
    --base-fvecs <PATH>            Required input base .fvecs path
    --query-fvecs <PATH>           Required input query .fvecs path
    --m-pq <USIZE>                 PQ subspace count (default: 16)
    --k <USIZE>                    Top-k neighbors (default: 10)
    --ef <USIZE>                   HNSW efSearch for index parity output (default: 50)
    --json-out, --output <PATH>    Output JSON path (prints to stdout if omitted)
"#
}

/// Parses raw CLI argument tokens into a [`Phase6CliCommand`].
pub fn parse_phase6_cli_args(args: &[String]) -> Result<Phase6CliCommand> {
    let mut mode = Phase6RunMode::Microbench;
    let mut is_pilot = false;

    let mut dataset = Phase6DatasetChoice::Sift1M;
    let mut dataset_explicit = false;
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut gt_path: Option<PathBuf> = None;
    let mut data_dir = PathBuf::from("data");
    let mut output_dir = PathBuf::from("results/phase6");
    let mut output_dir_explicit = false;
    let mut output_file: Option<PathBuf> = None;

    let mut scale: Option<usize> = None;
    let mut queries: Option<usize> = None;
    let mut k: usize = 10;
    let mut ef: usize = 50;
    let mut ef_explicit = false;
    let mut reps: Option<usize> = None;
    let mut seed: u64 = 42;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 16;
    let mut num_evals: usize = 200_000;
    let mut m_pq: usize = 16;
    let mut max_train_vectors: usize = PQ_DEFAULT_MAX_TRAIN_VECTORS;
    let mut max_lloyd_iters: usize = PQ_DEFAULT_MAX_LLOYD_ITERS;

    let mut arms: Option<Vec<String>> = None;
    let mut ef_search_list: Vec<usize> = DEFAULT_EF_SEARCH_LIST.to_vec();
    let mut rerank_list_override: Option<Vec<RerankDepthSpec>> = None;

    let mut resume = true;
    let mut synthetic = false;
    let mut offline = false;
    let mut enable_gt_cache = true;
    let mut profile_query_rss = true;
    let mut affinity: Option<String> = None;

    let mut internal_snapshot: Option<PathBuf> = None;
    let mut snapshot_kind: Option<SnapshotKind> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(Phase6CliCommand::Help),
            "--microbench" => {
                mode = Phase6RunMode::Microbench;
                i += 1;
            }
            "--pilot" => {
                mode = Phase6RunMode::Pilot;
                is_pilot = true;
                i += 1;
            }
            "--matrix" => {
                mode = Phase6RunMode::Matrix;
                is_pilot = false;
                i += 1;
            }
            "--parity-check" => {
                mode = Phase6RunMode::ParityCheck;
                i += 1;
            }
            "--internal-profile-query" => {
                mode = Phase6RunMode::InternalProfileQuery;
                internal_snapshot = Some(PathBuf::from(parse_next_str(
                    args,
                    &mut i,
                    "--internal-profile-query",
                )?));
            }
            "--kind" => {
                let raw = parse_next_str(args, &mut i, "--kind")?;
                snapshot_kind = Some(SnapshotKind::parse(raw)?);
            }
            "--dataset" => {
                let raw = parse_next_str(args, &mut i, "--dataset")?;
                dataset = Phase6DatasetChoice::parse(raw)?;
                dataset_explicit = true;
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(parse_next_str(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                query_fvecs = Some(PathBuf::from(parse_next_str(
                    args,
                    &mut i,
                    "--query-fvecs",
                )?));
            }
            "--gt-path" => {
                gt_path = Some(PathBuf::from(parse_next_str(args, &mut i, "--gt-path")?));
            }
            "--data-dir" => {
                data_dir = PathBuf::from(parse_next_str(args, &mut i, "--data-dir")?);
            }
            "--output-dir" => {
                output_dir = PathBuf::from(parse_next_str(args, &mut i, "--output-dir")?);
                output_dir_explicit = true;
            }
            "--output" | "-o" | "--json-out" => {
                let flag = args[i].as_str();
                output_file = Some(PathBuf::from(parse_next_str(args, &mut i, flag)?));
            }
            "--scale" | "--num-vectors" => {
                let flag = args[i].as_str();
                scale = Some(parse_next_usize(args, &mut i, flag)?);
            }
            "--queries" => {
                queries = Some(parse_next_usize(args, &mut i, "--queries")?);
            }
            "--k" => {
                k = parse_next_usize(args, &mut i, "--k")?;
            }
            "--ef" => {
                ef = parse_next_usize(args, &mut i, "--ef")?;
                ef_explicit = true;
            }
            "--reps" => {
                reps = Some(parse_next_usize(args, &mut i, "--reps")?);
            }
            "--seed" => {
                seed = parse_next_u64(args, &mut i, "--seed")?;
            }
            "--dim" => {
                dim = parse_next_usize(args, &mut i, "--dim")?;
            }
            "--num-clusters" => {
                num_clusters = parse_next_usize(args, &mut i, "--num-clusters")?;
            }
            "--evals" | "--dists" => {
                let flag = args[i].as_str();
                num_evals = parse_next_usize(args, &mut i, flag)?;
            }
            "--m-pq" => {
                m_pq = parse_next_usize(args, &mut i, "--m-pq")?;
            }
            "--max-train-vectors" => {
                max_train_vectors = parse_next_usize(args, &mut i, "--max-train-vectors")?;
            }
            "--max-lloyd-iters" => {
                max_lloyd_iters = parse_next_usize(args, &mut i, "--max-lloyd-iters")?;
            }
            "--arms" => {
                let raw = parse_next_str(args, &mut i, "--arms")?;
                let mut parsed = Vec::new();
                let mut seen = BTreeSet::new();
                for part in raw.split(',').filter(|p| !p.trim().is_empty()) {
                    let spec = resolve_arm_spec(part)?;
                    if seen.insert(spec.arm_id) {
                        parsed.push(spec.arm_id.to_string());
                    }
                }
                if parsed.is_empty() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--arms list cannot be empty".to_string(),
                    ));
                }
                arms = Some(parsed);
            }
            "--ef-search-list" => {
                let raw = parse_next_str(args, &mut i, "--ef-search-list")?;
                ef_search_list = parse_csv_usizes(raw, "--ef-search-list")?;
            }
            "--rerank-list" => {
                let raw = parse_next_str(args, &mut i, "--rerank-list")?;
                let mut parsed = Vec::new();
                for part in raw.split(',').filter(|p| !p.trim().is_empty()) {
                    let r = RerankDepthSpec::parse(part)?;
                    if !parsed.contains(&r) {
                        parsed.push(r);
                    }
                }
                if parsed.is_empty() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--rerank-list cannot be empty".to_string(),
                    ));
                }
                rerank_list_override = Some(parsed);
            }
            "--resume" => {
                resume = true;
                i += 1;
            }
            "--no-resume" => {
                resume = false;
                i += 1;
            }
            "--synthetic" => {
                synthetic = true;
                i += 1;
            }
            "--offline" => {
                offline = true;
                i += 1;
            }
            "--no-gt-cache" => {
                enable_gt_cache = false;
                i += 1;
            }
            "--profile-query-rss" => {
                profile_query_rss = true;
                i += 1;
            }
            "--no-isolated-query-rss" => {
                profile_query_rss = false;
                i += 1;
            }
            "--affinity" => {
                affinity = Some(parse_next_str(args, &mut i, "--affinity")?.to_string());
            }
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI argument: '{other}'"
                )));
            }
        }
    }

    match mode {
        Phase6RunMode::Microbench => Ok(Phase6CliCommand::Microbench(Phase6MicrobenchConfig {
            output_dir,
            output_path: output_file,
            num_vectors: scale.unwrap_or(4_096),
            num_evals,
            dim,
            reps: reps.unwrap_or(5),
            seed,
            affinity,
        })),
        Phase6RunMode::Pilot | Phase6RunMode::Matrix => {
            let eff_output_dir = if is_pilot && !output_dir_explicit {
                PathBuf::from("results/phase6/pilot")
            } else {
                output_dir
            };
            let default_scale = if dataset_explicit && dataset == Phase6DatasetChoice::SiftSmall {
                10_000
            } else {
                100_000
            };
            let resolved_scale = scale.unwrap_or(default_scale);
            let default_queries = if dataset_explicit && dataset == Phase6DatasetChoice::SiftSmall {
                100
            } else {
                10_000
            };
            let resolved_queries = queries.unwrap_or(default_queries);
            let default_reps = if resolved_scale >= 1_000_000 { 5 } else { 3 };
            let resolved_reps = reps.unwrap_or(default_reps);
            let resolved_arms = arms.unwrap_or_else(|| {
                if is_pilot {
                    PILOT_ARM_IDS.iter().map(|s| (*s).to_string()).collect()
                } else {
                    CANONICAL_MATRIX_ARM_IDS
                        .iter()
                        .map(|s| (*s).to_string())
                        .collect()
                }
            });

            let cfg = Phase6MatrixConfig {
                is_pilot,
                dataset,
                base_fvecs,
                query_fvecs,
                gt_path,
                data_dir,
                output_dir: eff_output_dir,
                scale: resolved_scale,
                scale_explicit: scale.is_some(),
                queries: resolved_queries,
                queries_explicit: queries.is_some(),
                k,
                reps: resolved_reps,
                reps_explicit: reps.is_some(),
                seed,
                dim,
                num_clusters,
                arm_ids: resolved_arms,
                ef_search_list,
                rerank_list_override,
                resume,
                synthetic,
                offline,
                enable_gt_cache,
                profile_query_rss,
                affinity,
                expected_commit_override: None,
            };
            cfg.validate()?;
            Ok(Phase6CliCommand::Matrix(Box::new(cfg)))
        }
        Phase6RunMode::ParityCheck => {
            let base = base_fvecs.ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "--parity-check requires --base-fvecs <PATH>".to_string(),
                )
            })?;
            let query = query_fvecs.ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "--parity-check requires --query-fvecs <PATH>".to_string(),
                )
            })?;
            if k == 0 || ef == 0 || m_pq == 0 {
                return Err(VectorSearchError::InvalidArgument(
                    "--k, --ef, and --m-pq must all be positive (> 0)".to_string(),
                ));
            }
            Ok(Phase6CliCommand::ParityCheck(Phase6ParityConfig {
                base_fvecs: base,
                query_fvecs: query,
                m_pq,
                k,
                ef,
                seed,
                max_train_vectors,
                max_lloyd_iters,
                json_out: output_file,
            }))
        }
        Phase6RunMode::InternalProfileQuery => {
            let snapshot_path = internal_snapshot.ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "--internal-profile-query requires a snapshot path".to_string(),
                )
            })?;
            let kind = snapshot_kind.ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "--internal-profile-query requires --kind <sq8|pq>".to_string(),
                )
            })?;
            if query_fvecs.is_none() && !synthetic {
                return Err(VectorSearchError::InvalidArgument(
                    "--internal-profile-query requires --query-fvecs <PATH> or --synthetic"
                        .to_string(),
                ));
            }
            Ok(Phase6CliCommand::InternalProfileQuery(
                Phase6InternalQueryProfileConfig {
                    snapshot_path,
                    kind,
                    query_fvecs,
                    synthetic,
                    num_queries: queries.unwrap_or(10_000),
                    dim,
                    k,
                    ef: if ef_explicit { ef } else { 100 },
                    seed,
                    affinity,
                },
            ))
        }
    }
}

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

fn parse_next_str<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str> {
    if *idx + 1 >= args.len() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Missing argument after {flag}"
        )));
    }
    let val = args[*idx + 1].as_str();
    *idx += 2;
    Ok(val)
}

fn parse_next_usize(args: &[String], idx: &mut usize, flag: &str) -> Result<usize> {
    let raw = parse_next_str(args, idx, flag)?;
    raw.parse::<usize>().map_err(|e| {
        VectorSearchError::InvalidArgument(format!("Invalid integer for {flag} ('{raw}'): {e}"))
    })
}

fn parse_next_u64(args: &[String], idx: &mut usize, flag: &str) -> Result<u64> {
    let raw = parse_next_str(args, idx, flag)?;
    raw.parse::<u64>().map_err(|e| {
        VectorSearchError::InvalidArgument(format!("Invalid integer for {flag} ('{raw}'): {e}"))
    })
}

fn parse_csv_usizes(raw: &str, flag: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in raw.split(',').filter(|p| !p.trim().is_empty()) {
        let v = part.trim().parse::<usize>().map_err(|e| {
            VectorSearchError::InvalidArgument(format!("Invalid integer in {flag} ('{part}'): {e}"))
        })?;
        if v == 0 {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Values in {flag} must be >= 1, got 0"
            )));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "{flag} list cannot be empty"
        )));
    }
    Ok(out)
}

// ============================================================================
// Affinity Pinning Helper (mirrors src/bin/phase7.rs)
// ============================================================================

fn parse_affinity_mask(raw: &str) -> Result<u64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(VectorSearchError::InvalidArgument(
            "Affinity mask cannot be empty".to_string(),
        ));
    }
    let mask = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16)
    } else {
        trimmed.parse::<u64>()
    }
    .map_err(|e| {
        VectorSearchError::InvalidArgument(format!("Invalid affinity mask '{trimmed}': {e}"))
    })?;
    if mask == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Affinity mask must be positive (> 0)".to_string(),
        ));
    }
    Ok(mask)
}

fn apply_affinity_pinning(affinity_arg: Option<&str>, prov: &mut ReportProvenance) -> Result<()> {
    let Some(raw) = affinity_arg else {
        return Ok(());
    };
    let requested_mask = parse_affinity_mask(raw)?;
    let verified_mask = pin_process_affinity_os(requested_mask)?;
    prov.processor_affinity_mask = format!("0x{verified_mask:04x}");
    Ok(())
}

#[cfg(target_os = "windows")]
fn pin_process_affinity_os(requested_mask: u64) -> Result<u64> {
    let pid = std::process::id();
    let script = format!(
        "$p = Get-Process -Id {pid}; $p.ProcessorAffinity = [IntPtr]{requested_mask}; [int64](Get-Process -Id {pid}).ProcessorAffinity"
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .map_err(VectorSearchError::Io)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(VectorSearchError::InvalidArgument(format!(
            "Failed to set process affinity mask 0x{requested_mask:04x}: {}",
            stderr.trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let observed = stdout.trim().parse::<u64>().map_err(|e| {
        VectorSearchError::VerificationFailed(format!(
            "Failed to parse verified process affinity output '{}': {e}",
            stdout.trim()
        ))
    })?;
    if observed != requested_mask {
        return Err(VectorSearchError::VerificationFailed(format!(
            "Process affinity verification failed: expected 0x{requested_mask:04x}, got 0x{observed:04x}"
        )));
    }
    Ok(observed)
}

#[cfg(not(target_os = "windows"))]
fn pin_process_affinity_os(_requested_mask: u64) -> Result<u64> {
    Err(VectorSearchError::InvalidArgument(
        "--affinity pinning inside phase6 is only supported on Windows; pin externally (e.g. taskset) and set VECTOR_SEARCH_AFFINITY_MASK".to_string(),
    ))
}

// ============================================================================
// 1. Microbenchmark (`--microbench`)
// ============================================================================

fn time_kernel_repetitions<F>(num_evals: usize, reps: usize, mut run_pass: F) -> Vec<f64>
where
    F: FnMut() -> f32,
{
    // 1 untimed warmup pass
    std::hint::black_box(run_pass());

    let mut run_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let acc = run_pass();
        let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
        std::hint::black_box(acc);
        run_ns.push(elapsed_ns / (num_evals.max(1) as f64));
    }
    run_ns
}

fn summarize_kernel_bench(
    kernel_id: &str,
    label: &str,
    bytes_per_vector: usize,
    run_ns_per_eval: Vec<f64>,
) -> KernelMicrobenchResult {
    let mut sorted = run_ns_per_eval.clone();
    sorted.sort_by(f64::total_cmp);
    let median_ns = percentile_sorted(&sorted, 50.0);
    let min_ns = sorted.first().copied().unwrap_or(0.0);
    let max_ns = sorted.last().copied().unwrap_or(0.0);
    let evals_per_sec = if median_ns > 0.0 {
        1e9 / median_ns
    } else {
        0.0
    };
    KernelMicrobenchResult {
        kernel_id: kernel_id.to_string(),
        label: label.to_string(),
        bytes_per_vector,
        median_ns_per_eval: median_ns,
        min_ns_per_eval: min_ns,
        max_ns_per_eval: max_ns,
        spread_ns_per_eval: (max_ns - min_ns).max(0.0),
        evals_per_sec,
        run_ns_per_eval,
    }
}

fn compute_sq8_f64_reference_errors(
    base: &RowMatrix<f32>,
    queries: &RowMatrix<f32>,
    sq8_shifted: &Sq8Vectors,
    sq8_expanded: &Sq8Vectors,
) -> (f64, f64) {
    let quantizer = sq8_shifted.quantizer();
    let dim = base.cols();
    let mut recon = vec![0.0_f32; dim];
    let mut state_a = Sq8QueryState::default();
    let mut state_b = Sq8QueryState::default();
    let mut max_rel_a = 0.0_f64;
    let mut max_rel_b = 0.0_f64;

    let sample_rows = base.rows().min(256);
    for qi in 0..queries.rows() {
        let q = queries.row(qi);
        sq8_shifted.prepare_query(q, &mut state_a);
        sq8_expanded.prepare_query(q, &mut state_b);
        for row_id in 0..sample_rows {
            let _ = quantizer.dequantize_row(sq8_shifted.code(row_id), &mut recon);
            let mut ref_f64 = 0.0_f64;
            for (j, &qj) in q.iter().enumerate().take(dim) {
                let d = f64::from(qj)
                    - (f64::from(quantizer.mins()[j])
                        + f64::from(quantizer.scales()[j])
                            * f64::from(sq8_shifted.code(row_id)[j]));
                ref_f64 += d * d;
            }
            let da = f64::from(sq8_shifted.query_distance(&state_a, row_id));
            let db = f64::from(sq8_expanded.query_distance(&state_b, row_id));
            let denom = ref_f64.max(1e-12);
            max_rel_a = max_rel_a.max((da - ref_f64).abs() / denom);
            max_rel_b = max_rel_b.max((db - ref_f64).abs() / denom);
        }
    }
    (max_rel_a, max_rel_b)
}

fn measure_adversarial_cancellation_errors() -> Result<(f64, f64)> {
    // Anchor at 0.0 sets min_j = 0.0 while clustered rows live near 10_000.0 + tiny offsets.
    let base = RowMatrix::new(
        4,
        4,
        vec![
            0.0, 0.0, 0.0, 0.0, //
            10_000.0, 10_000.0, 10_000.0, 10_000.0, //
            10_000.05, 10_000.02, 9_999.98, 10_000.01, //
            10_000.1, 9_999.95, 10_000.04, 10_000.0, //
        ],
    )?;
    let query = [10_000.01_f32, 10_000.01, 10_000.01, 10_000.01];
    let sq8_a =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::QueryShifted)?;
    let sq8_b =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::NormExpanded)?;
    let quantizer = sq8_a.quantizer();

    let mut state_a = Sq8QueryState::default();
    let mut state_b = Sq8QueryState::default();
    sq8_a.prepare_query(&query, &mut state_a);
    sq8_b.prepare_query(&query, &mut state_b);

    let mut max_abs_a = 0.0_f64;
    let mut max_abs_b = 0.0_f64;
    for row_id in 1..4 {
        let code = sq8_a.code(row_id);
        let mut ref_f64 = 0.0_f64;
        for j in 0..4 {
            let recon = f64::from(quantizer.mins()[j])
                + f64::from(quantizer.scales()[j]) * f64::from(code[j]);
            let d = f64::from(query[j]) - recon;
            ref_f64 += d * d;
        }
        let da = f64::from(sq8_a.query_distance(&state_a, row_id));
        let db = f64::from(sq8_b.query_distance(&state_b, row_id));
        max_abs_a = max_abs_a.max((da - ref_f64).abs());
        max_abs_b = max_abs_b.max((db - ref_f64).abs());
    }
    Ok((max_abs_a, max_abs_b))
}

/// Runs the Phase 6 distance kernel microbenchmark (`Way A`, `Way B`, `PQ16`, `PQ32`, `f32`).
pub fn run_phase6_microbench(config: &Phase6MicrobenchConfig) -> Result<Phase6MicrobenchReport> {
    if config.num_vectors == 0 || config.num_evals == 0 || config.dim == 0 || config.reps == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Microbench parameters (num_vectors, num_evals, dim, reps) must all be > 0".to_string(),
        ));
    }

    let mut prov = detect_provenance();
    apply_affinity_pinning(config.affinity.as_deref(), &mut prov)?;
    let hw = detect_hardware();

    let num_queries = 8_usize;
    let (base, queries) =
        generate_synthetic_sift(config.num_vectors, num_queries, config.dim, 16, config.seed)?;

    let mut rng = DeterministicRng::new(config.seed ^ 0x9E37_79B9_7F4A_7C15);
    let access_ids: Vec<usize> = (0..config.num_evals)
        .map(|_| rng.next_usize(config.num_vectors))
        .collect();

    let sq8_a =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::QueryShifted)?;
    let sq8_b =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::NormExpanded)?;
    let (std_rel_a, std_rel_b) = compute_sq8_f64_reference_errors(&base, &queries, &sq8_a, &sq8_b);
    let (adv_abs_a, adv_abs_b) = measure_adversarial_cancellation_errors()?;

    let mut kernels = Vec::new();

    // 1. Raw f32 l2_squared
    let q_f32 = queries.row(0);
    let f32_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
        let q = std::hint::black_box(q_f32);
        let mut acc = 0.0_f32;
        for &id in std::hint::black_box(access_ids.as_slice()) {
            acc += l2_squared(q, base.row(id));
        }
        acc
    });
    kernels.push(summarize_kernel_bench(
        "f32_l2_squared",
        "Raw f32 L2Squared (8-way unrolled)",
        config.dim * std::mem::size_of::<f32>(),
        f32_runs,
    ));

    // 2. SQ8 Way A (QueryShifted)
    let mut state_a = Sq8QueryState::default();
    sq8_a.prepare_query(q_f32, &mut state_a);
    let scales_a = sq8_a.quantizer().scales();
    let way_a_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
        let qs = std::hint::black_box(state_a.q_shifted.as_slice());
        let sc = std::hint::black_box(scales_a);
        let mut acc = 0.0_f32;
        for &id in std::hint::black_box(access_ids.as_slice()) {
            acc += l2_squared_sq8_shifted(qs, sc, sq8_a.code(id));
        }
        acc
    });
    let way_a_res = summarize_kernel_bench(
        "sq8_query_shifted",
        "SQ8 Way A: Query-Shifted Asymmetric L2^2",
        config.dim,
        way_a_runs,
    );
    let way_a_median_ns = way_a_res.median_ns_per_eval;
    kernels.push(way_a_res);

    // 3. SQ8 Way B (NormExpanded)
    let mut state_b = Sq8QueryState::default();
    sq8_b.prepare_query(q_f32, &mut state_b);
    let norms_b = sq8_b.base_norms_sq().unwrap_or(&[]);
    let way_b_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
        let q_norm = std::hint::black_box(state_b.scalar);
        let q_scaled = std::hint::black_box(state_b.q_shifted.as_slice());
        let mut acc = 0.0_f32;
        for &id in std::hint::black_box(access_ids.as_slice()) {
            acc += l2_squared_sq8_expanded(q_norm, q_scaled, sq8_b.code(id), norms_b[id]);
        }
        acc
    });
    let way_b_res = summarize_kernel_bench(
        "sq8_norm_expanded",
        "SQ8 Way B: Norm-Expanded Asymmetric L2^2 (+4 B/vec)",
        config.dim + std::mem::size_of::<f32>(),
        way_b_runs,
    );
    let way_b_median_ns = way_b_res.median_ns_per_eval;
    kernels.push(way_b_res);

    // 4 & 5. PQ16 and PQ32 ADC kernels when dim is divisible
    for m_pq in [16_usize, 32_usize] {
        if !config.dim.is_multiple_of(m_pq) {
            continue;
        }
        let pq_cfg = PqConfig {
            m_pq,
            seed: config.seed,
            max_train_vectors: config.num_vectors.min(PQ_DEFAULT_MAX_TRAIN_VECTORS),
            max_lloyd_iters: 15,
            ..PqConfig::default()
        };
        let pq_vecs = PqVectors::from_matrix(&base, Metric::L2Squared, &pq_cfg)?;
        let mut pq_state = PqQueryState::with_subspaces(m_pq);
        pq_vecs.prepare_query(q_f32, &mut pq_state);
        let pq_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
            let tbl = std::hint::black_box(pq_state.adc_table.as_slice());
            let mut acc = 0.0_f32;
            for &id in std::hint::black_box(access_ids.as_slice()) {
                acc += pq_adc_distance(tbl, pq_vecs.code(id));
            }
            acc
        });
        kernels.push(summarize_kernel_bench(
            &format!("pq{m_pq}_adc"),
            &format!("PQ{m_pq} ADC Lookup Table Distance ({m_pq} B/vec)"),
            m_pq,
            pq_runs,
        ));
    }

    let speedup_b_over_a = if way_b_median_ns > 0.0 {
        way_a_median_ns / way_b_median_ns
    } else {
        1.0
    };
    let recommended_default_kernel = "query_shifted".to_string();

    let output_path = config.resolved_output_path();
    let report = Phase6MicrobenchReport {
        output_path: output_path.clone(),
        dim: config.dim,
        num_vectors: config.num_vectors,
        num_evals: config.num_evals,
        reps: config.reps,
        seed: config.seed,
        kernels,
        standard_max_rel_err_way_a: std_rel_a,
        standard_max_rel_err_way_b: std_rel_b,
        adversarial_offset_max_abs_err_way_a: adv_abs_a,
        adversarial_offset_max_abs_err_way_b: adv_abs_b,
        speedup_way_b_over_way_a: speedup_b_over_a,
        recommended_default_kernel,
    };

    let json = serialize_microbench_json(&report, &prov, &hw);
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output_path, json)?;
    eprintln!(
        "[INFO] Wrote Phase 6 kernel microbenchmark JSON to {}",
        output_path.display()
    );
    Ok(report)
}

fn serialize_microbench_json(
    r: &Phase6MicrobenchReport,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("{\n");
    out.push_str("  \"phase\": 6,\n");
    out.push_str("  \"benchmark\": \"kernel_microbenchmark\",\n");
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        escape_json(&current_utc_rfc3339())
    ));
    prov.append_json_fields(&mut out, hw);
    append_hardware_json(&mut out, hw);
    out.push_str("  \"measured_this_run\": true,\n");
    out.push_str("  \"warmup_performed\": true,\n");
    out.push_str("  \"parameters\": {\n");
    out.push_str(&format!("    \"dim\": {},\n", r.dim));
    out.push_str(&format!("    \"num_vectors\": {},\n", r.num_vectors));
    out.push_str(&format!("    \"num_evals\": {},\n", r.num_evals));
    out.push_str(&format!("    \"reps\": {},\n", r.reps));
    out.push_str(&format!("    \"seed\": {}\n", r.seed));
    out.push_str("  },\n");

    out.push_str("  \"kernels\": [\n");
    for (idx, k) in r.kernels.iter().enumerate() {
        let comma = if idx + 1 < r.kernels.len() { "," } else { "" };
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"kernel_id\": \"{}\",\n",
            escape_json(&k.kernel_id)
        ));
        out.push_str(&format!(
            "      \"label\": \"{}\",\n",
            escape_json(&k.label)
        ));
        out.push_str(&format!(
            "      \"bytes_per_vector\": {},\n",
            k.bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"median_ns_per_eval\": {:.4},\n",
            k.median_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"min_ns_per_eval\": {:.4},\n",
            k.min_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"max_ns_per_eval\": {:.4},\n",
            k.max_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"spread_ns_per_eval\": {:.4},\n",
            k.spread_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"evals_per_sec\": {:.2},\n",
            k.evals_per_sec
        ));
        out.push_str("      \"run_ns_per_eval\": [");
        for (ri, &val) in k.run_ns_per_eval.iter().enumerate() {
            if ri > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{val:.4}"));
        }
        out.push_str("]\n");
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ],\n");

    out.push_str("  \"way_a_vs_way_b\": {\n");
    out.push_str(&format!(
        "    \"speedup_way_b_over_way_a\": {:.4},\n",
        r.speedup_way_b_over_way_a
    ));
    out.push_str(&format!(
        "    \"standard_max_rel_err_way_a\": {:.12e},\n",
        r.standard_max_rel_err_way_a
    ));
    out.push_str(&format!(
        "    \"standard_max_rel_err_way_b\": {:.12e},\n",
        r.standard_max_rel_err_way_b
    ));
    out.push_str(&format!(
        "    \"adversarial_offset_max_abs_err_way_a\": {:.12e},\n",
        r.adversarial_offset_max_abs_err_way_a
    ));
    out.push_str(&format!(
        "    \"adversarial_offset_max_abs_err_way_b\": {:.12e},\n",
        r.adversarial_offset_max_abs_err_way_b
    ));
    out.push_str(&format!(
        "    \"recommended_default_kernel\": \"{}\"\n",
        escape_json(&r.recommended_default_kernel)
    ));
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

// ============================================================================
// 2. Parity Check Subcommand (`--parity-check`)
// ============================================================================

fn canonical_topk_from_distances(dists: &[f32], k: usize) -> Vec<usize> {
    let mut pairs: Vec<(f32, usize)> = dists
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();
    pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    pairs.into_iter().take(k).map(|(_, id)| id).collect()
}

fn format_f32_json(val: f32) -> Result<String> {
    if !val.is_finite() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Cannot serialize non-finite f32 value ({val}) into parity JSON"
        )));
    }
    Ok(format!("{val:?}"))
}

fn push_f32_array(out: &mut String, vals: &[f32]) -> Result<()> {
    out.push('[');
    for (idx, &v) in vals.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&format_f32_json(v)?);
    }
    out.push(']');
    Ok(())
}

fn push_usize_2d_array(out: &mut String, rows: &[Vec<usize>], indent: &str) {
    out.push_str("[\n");
    for (ri, row) in rows.iter().enumerate() {
        out.push_str(indent);
        out.push_str("  [");
        for (ci, &val) in row.iter().enumerate() {
            if ci > 0 {
                out.push_str(", ");
            }
            out.push_str(&val.to_string());
        }
        out.push(']');
        if ri + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

fn push_u8_2d_array(out: &mut String, data: &[u8], rows: usize, cols: usize, indent: &str) {
    out.push_str("[\n");
    for ri in 0..rows {
        out.push_str(indent);
        out.push_str("  [");
        let row = &data[ri * cols..(ri + 1) * cols];
        for (ci, &val) in row.iter().enumerate() {
            if ci > 0 {
                out.push_str(", ");
            }
            out.push_str(&val.to_string());
        }
        out.push(']');
        if ri + 1 < rows {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

fn push_f32_2d_array(out: &mut String, rows: &[Vec<f32>], indent: &str) -> Result<()> {
    out.push_str("[\n");
    for (ri, row) in rows.iter().enumerate() {
        out.push_str(indent);
        out.push_str("  ");
        push_f32_array(out, row)?;
        if ri + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
    Ok(())
}

/// Runs `--parity-check` exporting deterministic SQ8 + PQ encodings, distances, and top-`k` IDs.
pub fn run_phase6_parity_check(config: &Phase6ParityConfig) -> Result<()> {
    let base = read_fvecs(&config.base_fvecs)?;
    let queries = read_fvecs(&config.query_fvecs)?;
    if base.rows() == 0 || queries.rows() == 0 || base.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Base and query matrices for --parity-check must be non-empty".to_string(),
        ));
    }
    if base.cols() != queries.cols() {
        return Err(VectorSearchError::DimensionMismatch {
            expected: base.cols(),
            actual: queries.cols(),
        });
    }
    let num_base = base.rows();
    let num_queries = queries.rows();
    let dim = base.cols();
    let eff_k = config.k.min(num_base);

    let exact_gt =
        ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &queries, eff_k, 0)?;
    let exact_gt_rows: Vec<Vec<usize>> = (0..num_queries)
        .map(|qi| {
            exact_gt
                .indices
                .row(qi)
                .iter()
                .map(|&v| v as usize)
                .collect()
        })
        .collect();

    let hnsw_cfg = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: config.seed,
        ..HnswConfig::default()
    };

    // 1. SQ8 Calibration, Encode, Exhaustive Distances, and HNSW Index Search
    let sq8_vecs = Sq8Vectors::from_matrix(&base, Metric::L2Squared)?;
    let mut sq8_state = Sq8QueryState::default();
    let mut sq8_dists = Vec::with_capacity(num_queries);
    let mut sq8_exhaustive_topk = Vec::with_capacity(num_queries);
    for q_row in queries.iter_rows() {
        sq8_vecs.prepare_query(q_row, &mut sq8_state);
        let row_dists: Vec<f32> = (0..num_base)
            .map(|id| sq8_vecs.query_distance(&sq8_state, id))
            .collect();
        sq8_exhaustive_topk.push(canonical_topk_from_distances(&row_dists, eff_k));
        sq8_dists.push(row_dists);
    }

    let sq8_idx_only = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        hnsw_cfg.clone(),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )?;
    let mut sq8_scratch = sq8_idx_only.new_search_scratch(config.ef.max(eff_k));
    let sq8_only_res = sq8_idx_only.search_batch_with_scratch(
        &queries,
        eff_k,
        config.ef,
        None,
        &mut sq8_scratch,
    )?;
    let sq8_rerank_res = sq8_idx_only.search_batch_with_base_timed_with_scratch(
        &base,
        &queries,
        eff_k,
        config.ef,
        Some(config.ef.max(eff_k)),
        &mut sq8_scratch,
    )?;
    let sq8_only_ids: Vec<Vec<usize>> = sq8_only_res
        .iter()
        .map(|row| row.iter().map(|n| n.id).collect())
        .collect();
    let sq8_rerank_ids: Vec<Vec<usize>> = sq8_rerank_res
        .results
        .iter()
        .map(|row| row.iter().map(|n| n.id).collect())
        .collect();

    // 2. PQ Training, Encode, Exhaustive ADC Distances, and HNSW Index Search
    let pq_cfg = PqConfig {
        m_pq: config.m_pq,
        k_centroids: PQ_K_CENTROIDS,
        max_train_vectors: config.max_train_vectors,
        max_lloyd_iters: config.max_lloyd_iters,
        seed: config.seed,
    };
    let _d_sub = pq_cfg.validate_for_dim(dim)?;
    let pq_vecs = PqVectors::from_matrix(&base, Metric::L2Squared, &pq_cfg)?;
    let mut pq_state = PqQueryState::with_subspaces(config.m_pq);
    let mut pq_dists = Vec::with_capacity(num_queries);
    let mut pq_exhaustive_topk = Vec::with_capacity(num_queries);
    for q_row in queries.iter_rows() {
        pq_vecs.prepare_query(q_row, &mut pq_state);
        let row_dists: Vec<f32> = (0..num_base)
            .map(|id| pq_vecs.query_distance(&pq_state, id))
            .collect();
        pq_exhaustive_topk.push(canonical_topk_from_distances(&row_dists, eff_k));
        pq_dists.push(row_dists);
    }

    let pq_idx_only = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        hnsw_cfg,
        pq_cfg,
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )?;
    let mut pq_scratch = pq_idx_only.new_search_scratch(config.ef.max(eff_k));
    let pq_only_res =
        pq_idx_only.search_batch_with_scratch(&queries, eff_k, config.ef, None, &mut pq_scratch)?;
    let pq_rerank_res = pq_idx_only.search_batch_with_base_timed_with_scratch(
        &base,
        &queries,
        eff_k,
        config.ef,
        Some(config.ef.max(eff_k)),
        &mut pq_scratch,
    )?;
    let pq_only_ids: Vec<Vec<usize>> = pq_only_res
        .iter()
        .map(|row| row.iter().map(|n| n.id).collect())
        .collect();
    let pq_rerank_ids: Vec<Vec<usize>> = pq_rerank_res
        .results
        .iter()
        .map(|row| row.iter().map(|n| n.id).collect())
        .collect();

    // Serialize JSON
    let mut out = String::with_capacity(64 * 1024);
    out.push_str("{\n");
    out.push_str("  \"command\": \"parity_check\",\n");
    out.push_str(&format!("  \"num_base\": {num_base},\n"));
    out.push_str(&format!("  \"num_queries\": {num_queries},\n"));
    out.push_str(&format!("  \"dim\": {dim},\n"));
    out.push_str(&format!("  \"k\": {eff_k},\n"));
    out.push_str(&format!("  \"ef\": {},\n", config.ef));
    out.push_str(&format!("  \"m_pq\": {},\n", config.m_pq));
    out.push_str(&format!("  \"seed\": {},\n", config.seed));
    out.push_str("  \"exact_gt_topk_ids\": ");
    push_usize_2d_array(&mut out, &exact_gt_rows, "  ");
    out.push_str(",\n");

    // SQ8 section
    let quant = sq8_vecs.quantizer();
    out.push_str("  \"sq8\": {\n");
    out.push_str("    \"mins\": ");
    push_f32_array(&mut out, quant.mins())?;
    out.push_str(",\n");
    out.push_str("    \"scales\": ");
    push_f32_array(&mut out, quant.scales())?;
    out.push_str(",\n");
    out.push_str("    \"inv_scales\": ");
    push_f32_array(&mut out, quant.inv_scales())?;
    out.push_str(",\n");
    out.push_str("    \"codes\": ");
    push_u8_2d_array(&mut out, sq8_vecs.codes(), num_base, dim, "    ");
    out.push_str(",\n");
    out.push_str("    \"asymmetric_l2_sq\": ");
    push_f32_2d_array(&mut out, &sq8_dists, "    ")?;
    out.push_str(",\n");
    out.push_str("    \"exhaustive_topk_ids\": ");
    push_usize_2d_array(&mut out, &sq8_exhaustive_topk, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_only_topk_ids\": ");
    push_usize_2d_array(&mut out, &sq8_only_ids, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_rerank_topk_ids\": ");
    push_usize_2d_array(&mut out, &sq8_rerank_ids, "    ");
    out.push('\n');
    out.push_str("  },\n");

    // PQ section
    let cb = pq_vecs.codebook();
    out.push_str("  \"pq\": {\n");
    out.push_str(&format!("    \"m_pq\": {},\n", cb.m_pq()));
    out.push_str(&format!("    \"k_centroids\": {},\n", cb.k_centroids()));
    out.push_str(&format!("    \"k_effective\": {},\n", cb.k_effective()));
    out.push_str(&format!("    \"d_sub\": {},\n", cb.d_sub()));
    out.push_str("    \"centroids\": [\n");
    for m in 0..cb.m_pq() {
        out.push_str("      [\n");
        for c in 0..cb.k_centroids() {
            out.push_str("        ");
            push_f32_array(&mut out, cb.centroid(m, c))?;
            if c + 1 < cb.k_centroids() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("      ]");
        if m + 1 < cb.m_pq() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("    ],\n");
    out.push_str("    \"codes\": ");
    push_u8_2d_array(&mut out, pq_vecs.codes(), num_base, cb.m_pq(), "    ");
    out.push_str(",\n");
    out.push_str("    \"adc_distances\": ");
    push_f32_2d_array(&mut out, &pq_dists, "    ")?;
    out.push_str(",\n");
    out.push_str("    \"exhaustive_topk_ids\": ");
    push_usize_2d_array(&mut out, &pq_exhaustive_topk, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_only_topk_ids\": ");
    push_usize_2d_array(&mut out, &pq_only_ids, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_rerank_topk_ids\": ");
    push_usize_2d_array(&mut out, &pq_rerank_ids, "    ");
    out.push('\n');
    out.push_str("  }\n");
    out.push_str("}\n");

    if let Some(ref json_out) = config.json_out {
        if let Some(parent) = json_out.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(json_out, &out)?;
    } else {
        print!("{out}");
    }
    Ok(())
}

// ============================================================================
// 3. Isolated Query-Serving RSS Profiler (`--internal-profile-query`)
// ============================================================================

/// Executes inside a fresh child OS process: loads ONLY the serialized quantized snapshot and
/// query vectors (never `RowMatrix<f32>`), runs 1 warmup + 1 query batch at `ef`, and prints
/// `query_serving_peak_rss_mb`.
pub fn run_phase6_internal_profile_query(config: &Phase6InternalQueryProfileConfig) -> Result<()> {
    if config.k == 0 || config.ef == 0 || config.num_queries == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "--k, --ef, and --queries must be positive (> 0)".to_string(),
        ));
    }
    let mut prov = detect_provenance();
    apply_affinity_pinning(config.affinity.as_deref(), &mut prov)?;

    match config.kind {
        SnapshotKind::Sq8 => {
            let index = HnswSq8Index::read_snapshot(&config.snapshot_path)?;
            run_isolated_query_profile_for(&index, config)?;
        }
        SnapshotKind::Pq => {
            let index = HnswPqIndex::read_snapshot(&config.snapshot_path)?;
            run_isolated_query_profile_for(&index, config)?;
        }
    }
    Ok(())
}

fn run_isolated_query_profile_for<T: QuantizedSweepTarget>(
    index: &T,
    config: &Phase6InternalQueryProfileConfig,
) -> Result<()> {
    let queries = load_isolated_queries(config, index.q_dim())?;
    let mut scratch = index.q_new_scratch(config.ef.max(config.k));
    index.q_warmup(queries.row(0), config.k, config.ef, &mut scratch)?;
    let res = index.q_search_batch_timed(&queries, config.k, config.ef, None, &mut scratch)?;
    std::hint::black_box(&res.results);
    let peak_rss = get_peak_rss_mb();
    print_query_profile_kv(
        index.q_num_vectors(),
        index.q_dim(),
        queries.rows(),
        config.ef,
        index.q_graph_fingerprint(),
        peak_rss,
    );
    Ok(())
}

fn load_isolated_queries(
    config: &Phase6InternalQueryProfileConfig,
    expected_dim: usize,
) -> Result<RowMatrix<f32>> {
    let queries = if let Some(ref q_path) = config.query_fvecs {
        read_fvecs_max_rows(q_path, Some(config.num_queries))?
    } else {
        // Synthetic query-only generation (never allocates base RowMatrix<f32>)
        let mut rng = DeterministicRng::new(config.seed ^ 0x517C_C1B7_2722_0A95);
        let total = config
            .num_queries
            .checked_mul(expected_dim)
            .ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "Synthetic query buffer overflows usize".to_string(),
                )
            })?;
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            let u = rng.next_f64_open01() * 200.0 - 100.0;
            data.push(u as f32);
        }
        RowMatrix::new(config.num_queries, expected_dim, data)?
    };
    if queries.rows() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Query matrix for --internal-profile-query cannot be empty".to_string(),
        ));
    }
    if queries.cols() != expected_dim {
        return Err(VectorSearchError::DimensionMismatch {
            expected: expected_dim,
            actual: queries.cols(),
        });
    }
    Ok(queries)
}

fn print_query_profile_kv(
    num_vectors: usize,
    dim: usize,
    num_queries: usize,
    ef: usize,
    graph_fingerprint: u64,
    peak_rss_mb: Option<f64>,
) {
    println!("PHASE6_QUERY_PROFILE_BEGIN");
    println!("num_vectors={num_vectors}");
    println!("dim={dim}");
    println!("num_queries={num_queries}");
    println!("ef={ef}");
    println!("graph_fingerprint_hex=0x{graph_fingerprint:016x}");
    match peak_rss_mb {
        Some(mb) => println!("query_serving_peak_rss_mb={mb:.4}"),
        None => println!("query_serving_peak_rss_mb=null"),
    }
    println!("PHASE6_QUERY_PROFILE_END");
}

fn parse_query_profile_kv(stdout: &str) -> Option<f64> {
    let mut in_block = false;
    let mut found_end = false;
    let mut rss: Option<f64> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed == "PHASE6_QUERY_PROFILE_BEGIN" {
            in_block = true;
            continue;
        }
        if trimmed == "PHASE6_QUERY_PROFILE_END" {
            found_end = true;
            break;
        }
        if !in_block {
            continue;
        }
        if let Some((k, v)) = trimmed.split_once('=') {
            if k == "query_serving_peak_rss_mb" && v != "null" {
                rss = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0);
            }
        }
    }
    if found_end {
        rss
    } else {
        None
    }
}

struct TempFileCleanup {
    paths: Vec<PathBuf>,
    tmp_dir: PathBuf,
}

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = fs::remove_file(p);
        }
        let _ = fs::remove_dir(&self.tmp_dir);
    }
}

fn resolve_phase6_executable() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let stem = exe.file_stem()?.to_string_lossy().to_ascii_lowercase();
    if stem == "phase6" {
        return Some(exe);
    }
    let bin_name = if cfg!(windows) {
        "phase6.exe"
    } else {
        "phase6"
    };
    if let Some(parent) = exe.parent() {
        let direct = parent.join(bin_name);
        if direct.is_file() {
            return Some(direct);
        }
        if let Some(grandparent) = parent.parent() {
            let sibling = grandparent.join(bin_name);
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn measure_isolated_query_rss<F>(
    kind: SnapshotKind,
    arm_id: &str,
    scale: usize,
    k: usize,
    data_dir: &Path,
    query_fvecs_path: Option<&Path>,
    queries: &RowMatrix<f32>,
    affinity: Option<&str>,
    write_snapshot_fn: F,
) -> Result<(Option<f64>, bool)>
where
    F: FnOnce(&Path) -> Result<()>,
{
    let Some(phase6_exe) = resolve_phase6_executable() else {
        return Ok((None, false));
    };

    let tmp_dir = data_dir.join("phase6_tmp");
    fs::create_dir_all(&tmp_dir)?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let snapshot_path = tmp_dir.join(format!("snapshot_n{scale}_{arm_id}_{pid}_{nanos}.bin"));
    let mut cleanup = TempFileCleanup {
        paths: vec![snapshot_path.clone()],
        tmp_dir,
    };

    write_snapshot_fn(&snapshot_path)?;

    let effective_query_fvecs = if let Some(q_path) = query_fvecs_path {
        q_path.to_path_buf()
    } else {
        let tmp_q = cleanup
            .tmp_dir
            .join(format!("queries_n{scale}_{arm_id}_{pid}_{nanos}.fvecs"));
        cleanup.paths.push(tmp_q.clone());
        write_fvecs(&tmp_q, queries)?;
        tmp_q
    };

    let mut cmd = Command::new(&phase6_exe);
    cmd.arg("--internal-profile-query")
        .arg(&snapshot_path)
        .arg("--kind")
        .arg(kind.as_str())
        .arg("--query-fvecs")
        .arg(&effective_query_fvecs)
        .arg("--queries")
        .arg(queries.rows().to_string())
        .arg("--k")
        .arg(k.to_string())
        .arg("--ef")
        .arg("100");
    if let Some(mask) = affinity {
        cmd.arg("--affinity").arg(mask);
    }

    let output = cmd.output().map_err(VectorSearchError::Io)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(VectorSearchError::VerificationFailed(format!(
            "Isolated query RSS child process failed for arm '{arm_id}': {}",
            stderr.trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed_rss = parse_query_profile_kv(&stdout);
    Ok((parsed_rss, true))
}

// ============================================================================
// 4. Dataset & Ground-Truth Resolution (No Silent Synthetic Fallback!)
// ============================================================================

struct PreparedDatasetState {
    source: DatasetSource,
    base_fvecs_path: Option<PathBuf>,
    query_fvecs_path: Option<PathBuf>,
    query: RowMatrix<f32>,
    base_cache: Option<Arc<RowMatrix<f32>>>,
    num_base: usize,
    dim: usize,
    gt_matrix: RowMatrix<i32>,
    gt_source: String,
    gt_cache_path: Option<PathBuf>,
    gt_cache_hit: bool,
}

impl PreparedDatasetState {
    /// Lazily loads or returns the shared `Arc<RowMatrix<f32>>` base matrix.
    fn get_or_load_base(&mut self) -> Result<Arc<RowMatrix<f32>>> {
        if let Some(ref arc) = self.base_cache {
            return Ok(Arc::clone(arc));
        }
        let path = self.base_fvecs_path.as_ref().ok_or_else(|| {
            VectorSearchError::InvalidArgument(
                "Internal error: missing base_fvecs_path when loading base matrix".to_string(),
            )
        })?;
        let base = read_fvecs_max_rows(path, Some(self.num_base))?;
        let arc = Arc::new(base);
        self.base_cache = Some(Arc::clone(&arc));
        Ok(arc)
    }
}

fn inspect_fvecs_shape(path: &Path) -> Result<(usize, usize)> {
    let meta = fs::metadata(path)?;
    let file_size = meta.len();
    if file_size == 0 {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Empty .fvecs file: {}",
            path.display()
        )));
    }
    if file_size < 4 {
        return Err(VectorSearchError::CorruptData(format!(
            "Corrupt .fvecs file (size {file_size} < 4): {}",
            path.display()
        )));
    }
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hdr = [0_u8; 4];
    reader.read_exact(&mut hdr)?;
    let dim_i32 = i32::from_le_bytes(hdr);
    if dim_i32 <= 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid .fvecs dim {dim_i32} in {}",
            path.display()
        )));
    }
    let dim = dim_i32 as usize;
    let rec_bytes = 4_u64 + (dim as u64) * 4;
    if file_size % rec_bytes != 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "File size {file_size} is not a multiple of record size {rec_bytes} in {}",
            path.display()
        )));
    }
    let num_vecs = usize::try_from(file_size / rec_bytes).map_err(|_| {
        VectorSearchError::CorruptData(format!(
            "Vector count in {} overflows usize",
            path.display()
        ))
    })?;
    Ok((num_vecs, dim))
}

fn resolve_sift_file_paths(
    preset: SiftPreset,
    data_dir: &Path,
) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let candidates: Vec<(PathBuf, &str)> = match preset {
        SiftPreset::SiftSmall => vec![
            (data_dir.join("siftsmall"), "siftsmall"),
            (data_dir.to_path_buf(), "siftsmall"),
        ],
        SiftPreset::Sift1M => vec![
            (data_dir.join("sift1m"), "sift"),
            (data_dir.join("sift"), "sift"),
            (data_dir.to_path_buf(), "sift"),
        ],
    };
    let expected_base_bytes =
        (preset.expected_num_base() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_query_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_dim() as u64));
    let expected_gt_bytes =
        (preset.expected_num_query() as u64) * (4 + 4 * (preset.expected_gt_k() as u64));

    for (dir, prefix) in candidates {
        let base_p = dir.join(format!("{prefix}_base.fvecs"));
        let query_p = dir.join(format!("{prefix}_query.fvecs"));
        let gt_p = dir.join(format!("{prefix}_groundtruth.ivecs"));
        if !base_p.is_file() || !query_p.is_file() || !gt_p.is_file() {
            continue;
        }
        let b_len = fs::metadata(&base_p).map(|m| m.len()).unwrap_or(0);
        let q_len = fs::metadata(&query_p).map(|m| m.len()).unwrap_or(0);
        let g_len = fs::metadata(&gt_p).map(|m| m.len()).unwrap_or(0);
        if b_len == expected_base_bytes
            && q_len == expected_query_bytes
            && g_len == expected_gt_bytes
        {
            return Some((base_p, query_p, gt_p));
        }
    }
    None
}

fn slice_and_validate_gt(
    src: &RowMatrix<i32>,
    num_queries: usize,
    num_base: usize,
    k: usize,
) -> Result<RowMatrix<i32>> {
    if src.rows() < num_queries || src.cols() < k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Ground truth matrix shape ({}x{}) is smaller than required ({num_queries}x{k})",
            src.rows(),
            src.cols()
        )));
    }
    let mut data = Vec::with_capacity(num_queries * k);
    for r in 0..num_queries {
        let slice = &src.row(r)[..k];
        for &id in slice {
            if id < 0 || (id as usize) >= num_base {
                return Err(VectorSearchError::CorruptData(format!(
                    "Ground truth ID {id} at row {r} out of range [0, {num_base})"
                )));
            }
        }
        data.extend_from_slice(slice);
    }
    RowMatrix::new(num_queries, k, data)
}

fn prepare_dataset_and_gt(config: &Phase6MatrixConfig) -> Result<PreparedDatasetState> {
    if config.synthetic {
        let (base, query) = generate_synthetic_sift(
            config.scale,
            config.queries,
            config.dim,
            config.num_clusters,
            config.seed,
        )?;
        let num_base = base.rows();
        let num_queries = query.rows();
        let dim = base.cols();
        let source = DatasetSource::Synthetic { seed: config.seed };

        let (gt_matrix, gt_source, gt_cache_path, gt_cache_hit) = if let Some(ref explicit_gt) =
            config.gt_path
        {
            let raw_gt = read_ivecs_max_rows(explicit_gt, Some(num_queries))?;
            let sliced = slice_and_validate_gt(&raw_gt, num_queries, num_base, config.k)?;
            (
                sliced,
                format!("custom_ivecs ({})", explicit_gt.display()),
                Some(explicit_gt.clone()),
                true,
            )
        } else {
            let cache_k = 100.max(config.k).min(num_base);
            let exact =
                ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, cache_k, 0)?;
            let sliced = slice_and_validate_gt(&exact.indices, num_queries, num_base, config.k)?;
            (
                sliced,
                format!("computed_exact_knn_synthetic (metric=l2, k={cache_k})"),
                None,
                false,
            )
        };

        return Ok(PreparedDatasetState {
            source,
            base_fvecs_path: None,
            query_fvecs_path: None,
            query,
            base_cache: Some(Arc::new(base)),
            num_base,
            dim,
            gt_matrix,
            gt_source,
            gt_cache_path,
            gt_cache_hit,
        });
    }

    // Custom .fvecs path pair
    if let (Some(ref base_path), Some(ref query_path)) = (&config.base_fvecs, &config.query_fvecs) {
        let (total_base, base_dim) = inspect_fvecs_shape(base_path)?;
        let (total_queries, query_dim) = inspect_fvecs_shape(query_path)?;
        if base_dim != query_dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: base_dim,
                actual: query_dim,
            });
        }
        let num_base = if config.scale_explicit {
            if config.scale > total_base {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --scale {}, but {} has only {total_base} vectors",
                    config.scale,
                    base_path.display()
                )));
            }
            config.scale
        } else {
            total_base
        };
        let num_queries = if config.queries_explicit {
            if config.queries > total_queries {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --queries {}, but {} has only {total_queries} queries",
                    config.queries,
                    query_path.display()
                )));
            }
            config.queries
        } else {
            total_queries
        };
        let query = read_fvecs_max_rows(query_path, Some(num_queries))?;
        let source = DatasetSource::CustomFiles {
            base_path: base_path.clone(),
            query_path: query_path.clone(),
        };

        return resolve_file_backed_gt(
            config,
            source,
            base_path.clone(),
            query_path.clone(),
            None,
            query,
            num_base,
            base_dim,
        );
    }

    // Authentic SIFT dataset under --data-dir (NEVER silently fall back to synthetic!)
    let preset = config.dataset.to_preset();
    if config.scale > preset.expected_num_base() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Requested --scale {} exceeds {} maximum base count ({})",
            config.scale,
            preset.as_str(),
            preset.expected_num_base()
        )));
    }
    if config.queries > preset.expected_num_query() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Requested --queries {} exceeds {} maximum query count ({})",
            config.queries,
            preset.as_str(),
            preset.expected_num_query()
        )));
    }

    let (base_path, query_path, official_gt_path) = if let Some(paths) =
        resolve_sift_file_paths(preset, &config.data_dir)
    {
        paths
    } else {
        // Probe / download if !offline; verify strictly authentic SIFT afterwards.
        let probe = load_or_download_sift_with_options(
            preset,
            &config.data_dir,
            !config.offline,
            config.seed,
            Some(1),
            Some(1),
            false,
        )?;
        if !probe.source.is_authentic_sift() {
            let reason = match &probe.source {
                DatasetSource::SyntheticFallback { reason, .. } => reason.as_str(),
                _ => "authentic SIFT files not found",
            };
            return Err(VectorSearchError::InvalidArgument(format!(
                    "Authentic {} dataset missing in '{}' ({reason}). Never falling back silently: pass --synthetic explicitly or place SIFT files under --data-dir.",
                    preset.as_str(),
                    config.data_dir.display()
                )));
        }
        resolve_sift_file_paths(preset, &config.data_dir).ok_or_else(|| {
            VectorSearchError::InvalidArgument(format!(
                "Unable to resolve SIFT files under '{}'",
                config.data_dir.display()
            ))
        })?
    };

    let source = match preset {
        SiftPreset::SiftSmall => DatasetSource::AuthenticSiftSmall {
            dir: base_path
                .parent()
                .map_or_else(|| config.data_dir.clone(), Path::to_path_buf),
        },
        SiftPreset::Sift1M => DatasetSource::AuthenticSift1M {
            dir: base_path
                .parent()
                .map_or_else(|| config.data_dir.clone(), Path::to_path_buf),
        },
    };
    let query = read_fvecs_max_rows(&query_path, Some(config.queries))?;
    let official_opt = if config.scale == preset.expected_num_base() {
        Some(official_gt_path)
    } else {
        None
    };

    resolve_file_backed_gt(
        config,
        source,
        base_path,
        query_path,
        official_opt,
        query,
        config.scale,
        preset.expected_dim(),
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_file_backed_gt(
    config: &Phase6MatrixConfig,
    source: DatasetSource,
    base_path: PathBuf,
    query_path: PathBuf,
    official_gt_path: Option<PathBuf>,
    query: RowMatrix<f32>,
    num_base: usize,
    dim: usize,
) -> Result<PreparedDatasetState> {
    let num_queries = query.rows();

    if let Some(ref explicit_gt) = config.gt_path {
        let raw_gt = read_ivecs_max_rows(explicit_gt, Some(num_queries))?;
        let gt_matrix = slice_and_validate_gt(&raw_gt, num_queries, num_base, config.k)?;
        return Ok(PreparedDatasetState {
            source,
            base_fvecs_path: Some(base_path),
            query_fvecs_path: Some(query_path),
            query,
            base_cache: None,
            num_base,
            dim,
            gt_matrix,
            gt_source: format!("custom_ivecs ({})", explicit_gt.display()),
            gt_cache_path: Some(explicit_gt.clone()),
            gt_cache_hit: true,
        });
    }

    if let Some(ref official_path) = official_gt_path {
        let raw_gt = read_ivecs_max_rows(official_path, Some(num_queries))?;
        let gt_matrix = slice_and_validate_gt(&raw_gt, num_queries, num_base, config.k)?;
        let gt_source = match source {
            DatasetSource::AuthenticSift1M { .. } => {
                "official_texmex_sift1m_groundtruth_ivecs".to_string()
            }
            DatasetSource::AuthenticSiftSmall { .. } => {
                "official_texmex_siftsmall_groundtruth_ivecs".to_string()
            }
            _ => format!("official_ivecs ({})", official_path.display()),
        };
        return Ok(PreparedDatasetState {
            source,
            base_fvecs_path: Some(base_path),
            query_fvecs_path: Some(query_path),
            query,
            base_cache: None,
            num_base,
            dim,
            gt_matrix,
            gt_source,
            gt_cache_path: Some(official_path.clone()),
            gt_cache_hit: true,
        });
    }

    // Prefix scale (< 1M): check <data-dir>/gt_{scale}_k100.ivecs
    let cache_path = config.data_dir.join(format!("gt_{num_base}_k100.ivecs"));
    let cache_k = 100.max(config.k).min(num_base);
    if config.enable_gt_cache && cache_path.is_file() {
        if let Ok(cached) = read_ivecs(&cache_path) {
            if cached.rows() == num_queries && cached.cols() >= config.k {
                if let Ok(gt_matrix) =
                    slice_and_validate_gt(&cached, num_queries, num_base, config.k)
                {
                    return Ok(PreparedDatasetState {
                        source,
                        base_fvecs_path: Some(base_path),
                        query_fvecs_path: Some(query_path),
                        query,
                        base_cache: None,
                        num_base,
                        dim,
                        gt_matrix,
                        gt_source: format!(
                            "cached_exact_knn_prefix_ivecs ({}, metric=l2, k={})",
                            cache_path.display(),
                            cached.cols()
                        ),
                        gt_cache_path: Some(cache_path),
                        gt_cache_hit: true,
                    });
                }
            }
        }
    }

    // Cache miss: load base matrix, compute exact GT, and retain base_cache in memory.
    let base = read_fvecs_max_rows(&base_path, Some(num_base))?;
    let exact = ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, cache_k, 0)?;
    if config.enable_gt_cache {
        fs::create_dir_all(&config.data_dir)?;
        write_ivecs(&cache_path, &exact.indices)?;
    }
    let gt_matrix = slice_and_validate_gt(&exact.indices, num_queries, num_base, config.k)?;

    Ok(PreparedDatasetState {
        source,
        base_fvecs_path: Some(base_path),
        query_fvecs_path: Some(query_path),
        query,
        base_cache: Some(Arc::new(base)),
        num_base,
        dim,
        gt_matrix,
        gt_source: format!(
            "computed_exact_knn_prefix_ivecs (ExactKnn::search_parallel_with_base metric=l2 k={cache_k}{})",
            if config.enable_gt_cache {
                format!(", cached to {}", cache_path.display())
            } else {
                String::new()
            }
        ),
        gt_cache_path: config.enable_gt_cache.then_some(cache_path),
        gt_cache_hit: false,
    })
}

// ============================================================================
// 5. Generic Quantized Arm Evaluation + f32 Baseline Evaluation
// ============================================================================

trait QuantizedSweepTarget {
    type QueryState: Default;

    fn q_num_vectors(&self) -> usize;
    fn q_dim(&self) -> usize;
    fn q_graph_fingerprint(&self) -> u64;
    fn q_memory_profile(&self) -> QuantizedMemoryProfile;
    fn q_new_scratch(&self, max_ef: usize) -> QuantizedSearchScratch<Self::QueryState>;
    fn q_warmup(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<Self::QueryState>,
    ) -> Result<()>;
    fn q_search_batch_timed(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Self::QueryState>,
    ) -> Result<TimedQuantizedSearch>;
    fn q_write_snapshot(&self, path: &Path) -> Result<()>;
}

impl QuantizedSweepTarget for HnswSq8Index {
    type QueryState = Sq8QueryState;

    fn q_num_vectors(&self) -> usize {
        self.num_vectors()
    }

    fn q_dim(&self) -> usize {
        self.dim()
    }

    fn q_graph_fingerprint(&self) -> u64 {
        self.graph_fingerprint()
    }

    fn q_memory_profile(&self) -> QuantizedMemoryProfile {
        self.memory_profile()
    }

    fn q_new_scratch(&self, max_ef: usize) -> QuantizedSearchScratch<Self::QueryState> {
        self.new_search_scratch(max_ef)
    }

    fn q_warmup(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<Self::QueryState>,
    ) -> Result<()> {
        self.warmup(query, k, ef_search, scratch)
    }

    fn q_search_batch_timed(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Self::QueryState>,
    ) -> Result<TimedQuantizedSearch> {
        self.search_batch_timed_with_scratch(queries, k, ef_search, rerank_top_r, scratch)
    }

    fn q_write_snapshot(&self, path: &Path) -> Result<()> {
        self.write_snapshot(path)
    }
}

impl QuantizedSweepTarget for HnswPqIndex {
    type QueryState = PqQueryState;

    fn q_num_vectors(&self) -> usize {
        self.num_vectors()
    }

    fn q_dim(&self) -> usize {
        self.dim()
    }

    fn q_graph_fingerprint(&self) -> u64 {
        self.graph_fingerprint()
    }

    fn q_memory_profile(&self) -> QuantizedMemoryProfile {
        self.memory_profile()
    }

    fn q_new_scratch(&self, max_ef: usize) -> QuantizedSearchScratch<Self::QueryState> {
        self.new_search_scratch(max_ef)
    }

    fn q_warmup(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<Self::QueryState>,
    ) -> Result<()> {
        self.warmup(query, k, ef_search, scratch)
    }

    fn q_search_batch_timed(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Self::QueryState>,
    ) -> Result<TimedQuantizedSearch> {
        self.search_batch_timed_with_scratch(queries, k, ef_search, rerank_top_r, scratch)
    }

    fn q_write_snapshot(&self, path: &Path) -> Result<()> {
        self.write_snapshot(path)
    }
}

fn to_latency_stats(timed: &TimedQuantizedSearch) -> LatencyStats {
    LatencyStats {
        num_queries: timed.metrics.num_queries,
        total_s: timed.metrics.total_s,
        qps: timed.metrics.qps,
        min_ms: timed.metrics.min_ms,
        mean_ms: timed.metrics.mean_ms,
        p50_ms: timed.metrics.p50_ms,
        p95_ms: timed.metrics.p95_ms,
        p99_ms: timed.metrics.p99_ms,
        max_ms: timed.metrics.max_ms,
    }
}

fn max_ef_for_sweep(k: usize, ef_list: &[usize], rerank_ladder: &[RerankDepthSpec]) -> usize {
    let mut max_ef = k;
    for &ef in ef_list {
        for &r in rerank_ladder {
            let (_, eff) = r.resolve_params(k, ef);
            max_ef = max_ef.max(eff);
        }
    }
    max_ef
}

fn evaluate_quantized_sweep<T: QuantizedSweepTarget>(
    index: &T,
    queries: &RowMatrix<f32>,
    gt_matrix: &RowMatrix<i32>,
    k: usize,
    reps: usize,
    ef_search_list: &[usize],
    rerank_ladder: &[RerankDepthSpec],
) -> Result<Vec<Phase6SweepPointResult>> {
    let max_ef = max_ef_for_sweep(k, ef_search_list, rerank_ladder);
    let mut scratch = index.q_new_scratch(max_ef);
    let mut points = Vec::with_capacity(ef_search_list.len() * rerank_ladder.len());

    for &rerank_depth in rerank_ladder {
        for &ef_search in ef_search_list {
            let (rerank_arg, effective_ef) = rerank_depth.resolve_params(k, ef_search);

            // 1 untimed warmup pass
            let _ = index.q_search_batch_timed(queries, k, ef_search, rerank_arg, &mut scratch)?;

            let mut run_latencies = Vec::with_capacity(reps);
            let mut last_results = None;
            for _ in 0..reps {
                let timed =
                    index.q_search_batch_timed(queries, k, ef_search, rerank_arg, &mut scratch)?;
                run_latencies.push(to_latency_stats(&timed));
                last_results = Some(timed.results);
            }

            let summary = summarize_repeated_latencies(&run_latencies);
            let recall_at_k = if let Some(ref res) = last_results {
                let knn_res = results_to_knn_search_result(res)?;
                compute_recall_at_k(&knn_res.indices, gt_matrix, k)?
            } else {
                0.0
            };

            points.push(Phase6SweepPointResult {
                ef_search,
                effective_ef_search: effective_ef,
                rerank_depth,
                recall_at_k,
                summary,
            });
        }
    }

    Ok(points)
}

fn evaluate_f32_sweep(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    gt_matrix: &RowMatrix<i32>,
    k: usize,
    reps: usize,
    ef_search_list: &[usize],
) -> Result<Vec<Phase6SweepPointResult>> {
    let mut tracker = VisitedTracker::new(index.num_vectors());
    let mut points = Vec::with_capacity(ef_search_list.len());

    for &ef_search in ef_search_list {
        let effective_ef = ef_search.max(k);
        // 1 untimed warmup pass
        let _ = index.search_batch_timed_with_tracker(queries, k, ef_search, &mut tracker)?;

        let mut run_latencies = Vec::with_capacity(reps);
        let mut last_indices = None;
        for _ in 0..reps {
            let timed =
                index.search_batch_timed_with_tracker(queries, k, ef_search, &mut tracker)?;
            run_latencies.push(timed.latency);
            last_indices = Some(timed.result.indices);
        }
        let summary = summarize_repeated_latencies(&run_latencies);
        let recall_at_k = if let Some(ref ind) = last_indices {
            compute_recall_at_k(ind, gt_matrix, k)?
        } else {
            0.0
        };

        points.push(Phase6SweepPointResult {
            ef_search,
            effective_ef_search: effective_ef,
            rerank_depth: RerankDepthSpec::None,
            recall_at_k,
            summary,
        });
    }

    Ok(points)
}

// ============================================================================
// 6. Report Serialization, Resume Logic, & Summary
// ============================================================================

struct ArmReportContext<'a> {
    spec: &'a Phase6ArmSpec,
    config: &'a Phase6MatrixConfig,
    prov: &'a ReportProvenance,
    hw: &'a HardwareInfo,
    dataset: &'a PreparedDatasetState,
    training_or_calibration_time_s: f64,
    encode_time_s: f64,
    graph_build_time_s: f64,
    total_build_time_s: f64,
    graph_fingerprint: u64,
    streaming_build: bool,
    pq_config: Option<PqConfig>,
    mem: QuantizedMemoryProfile,
    build_peak_rss_mb: Option<f64>,
    query_serving_peak_rss_mb: Option<f64>,
    query_rss_profiled_isolated: bool,
    sweep: &'a [Phase6SweepPointResult],
}

fn serialize_arm_report_json(ctx: &ArmReportContext<'_>) -> String {
    let mut out = String::with_capacity(16 * 1024);
    let denom = ctx.mem.num_vectors.max(1) as f64;
    let codes_bv = ctx.mem.codes_bytes as f64 / denom;
    let cal_bv = ctx.mem.calibration_or_codebook_bytes as f64 / denom;
    let raw_bv = ctx.mem.raw_vectors_retained_bytes as f64 / denom;
    let fp_hex = format!("0x{:016x}", ctx.graph_fingerprint);
    let build_rss_json = ctx
        .build_peak_rss_mb
        .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));
    let query_rss_json = ctx
        .query_serving_peak_rss_mb
        .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));
    let rss_scope = if ctx.query_rss_profiled_isolated {
        "isolated_query_subprocess_high_water_mark"
    } else {
        "process_lifetime_high_water_mark"
    };

    out.push_str("{\n");
    out.push_str("  \"phase\": 6,\n");
    out.push_str("  \"phase_name\": \"Phase 6 Vector Quantization (SQ8 & PQ)\",\n");
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        escape_json(&current_utc_rfc3339())
    ));
    ctx.prov.append_json_fields(&mut out, ctx.hw);
    append_hardware_json(&mut out, ctx.hw);

    out.push_str(&format!(
        "  \"arm_id\": \"{}\",\n",
        escape_json(ctx.spec.arm_id)
    ));
    out.push_str(&format!(
        "  \"quantizer_kind\": \"{}\",\n",
        escape_json(ctx.spec.quantizer_kind)
    ));
    out.push_str(&format!(
        "  \"storage_mode\": \"{}\",\n",
        escape_json(ctx.spec.storage_mode)
    ));
    out.push_str(&format!(
        "  \"build_mode\": \"{}\",\n",
        escape_json(ctx.spec.build_mode_str)
    ));
    out.push_str(&format!("  \"m\": {},\n", ctx.spec.m));
    out.push_str(&format!(
        "  \"ef_construction\": {},\n",
        ctx.spec.ef_construction
    ));
    out.push_str(&format!("  \"scale\": {},\n", ctx.dataset.num_base));
    out.push_str(&format!("  \"dim\": {},\n", ctx.dataset.dim));
    out.push_str(&format!("  \"k\": {},\n", ctx.config.k));
    out.push_str(&format!(
        "  \"num_queries\": {},\n",
        ctx.dataset.query.rows()
    ));
    out.push_str(&format!("  \"reps\": {},\n", ctx.config.reps));
    out.push_str(&format!("  \"seed\": {},\n", ctx.config.seed));
    out.push_str("  \"metric\": \"l2\",\n");
    out.push_str(&format!(
        "  \"streaming_build\": {},\n",
        ctx.streaming_build
    ));
    out.push_str("  \"measured_this_run\": true,\n");
    out.push_str("  \"warmup_performed\": true,\n");
    out.push_str(&format!(
        "  \"source_kind\": \"{}\",\n",
        escape_json(ctx.dataset.source.kind_tag())
    ));

    match ctx.spec.kind {
        Phase6ArmKind::Sq8 => out.push_str("  \"sq8_kernel\": \"query_shifted\",\n"),
        _ => out.push_str("  \"sq8_kernel\": null,\n"),
    }
    match ctx.pq_config {
        Some(pq) => {
            out.push_str("  \"pq_config\": {\n");
            out.push_str(&format!("    \"m_pq\": {},\n", pq.m_pq));
            out.push_str(&format!("    \"k_centroids\": {},\n", pq.k_centroids));
            out.push_str(&format!(
                "    \"max_train_vectors\": {},\n",
                pq.max_train_vectors
            ));
            out.push_str(&format!(
                "    \"max_lloyd_iters\": {},\n",
                pq.max_lloyd_iters
            ));
            out.push_str(&format!("    \"seed\": {}\n", pq.seed));
            out.push_str("  },\n");
        }
        None => out.push_str("  \"pq_config\": null,\n"),
    }

    out.push_str("  \"dataset\": {\n");
    out.push_str(&format!(
        "    \"source_kind\": \"{}\",\n",
        escape_json(ctx.dataset.source.kind_tag())
    ));
    out.push_str(&format!(
        "    \"source_description\": \"{}\",\n",
        escape_json(&ctx.dataset.source.to_string())
    ));
    out.push_str(&format!(
        "    \"is_authentic_sift\": {},\n",
        ctx.dataset.source.is_authentic_sift()
    ));
    out.push_str(&format!(
        "    \"scale_vectors\": {},\n",
        ctx.dataset.num_base
    ));
    out.push_str(&format!("    \"num_base\": {},\n", ctx.dataset.num_base));
    out.push_str(&format!(
        "    \"num_query\": {},\n",
        ctx.dataset.query.rows()
    ));
    out.push_str(&format!("    \"dim\": {},\n", ctx.dataset.dim));
    out.push_str(&format!("    \"k\": {},\n", ctx.config.k));
    out.push_str(&format!("    \"seed\": {},\n", ctx.config.seed));
    out.push_str("    \"metric\": \"l2\"\n");
    out.push_str("  },\n");

    out.push_str(&format!(
        "  \"ground_truth_source\": \"{}\",\n",
        escape_json(&ctx.dataset.gt_source)
    ));
    out.push_str(&format!(
        "  \"ground_truth_cache_path\": {},\n",
        opt_path_json(ctx.dataset.gt_cache_path.as_deref())
    ));
    out.push_str(&format!(
        "  \"ground_truth_cache_hit\": {},\n",
        ctx.dataset.gt_cache_hit
    ));

    out.push_str(&format!(
        "  \"total_build_time_s\": {:.6},\n",
        ctx.total_build_time_s
    ));
    out.push_str(&format!(
        "  \"training_or_calibration_time_s\": {:.6},\n",
        ctx.training_or_calibration_time_s
    ));
    out.push_str(&format!("  \"encode_time_s\": {:.6},\n", ctx.encode_time_s));
    out.push_str(&format!(
        "  \"graph_build_time_s\": {:.6},\n",
        ctx.graph_build_time_s
    ));
    out.push_str(&format!(
        "  \"graph_fingerprint\": {},\n",
        ctx.graph_fingerprint
    ));
    out.push_str(&format!("  \"graph_fingerprint_hex\": \"{fp_hex}\",\n"));

    out.push_str(&format!(
        "  \"steady_state_bytes_per_vector\": {:.4},\n",
        ctx.mem.steady_state_bytes_per_vector
    ));
    out.push_str(&format!(
        "  \"graph_only_bytes_per_vector\": {:.4},\n",
        ctx.mem.graph_only_bytes_per_vector
    ));
    out.push_str(&format!("  \"codes_bytes_per_vector\": {codes_bv:.4},\n"));
    out.push_str(&format!(
        "  \"calibration_or_codebook_bytes_per_vector\": {cal_bv:.4},\n"
    ));
    out.push_str(&format!(
        "  \"raw_vectors_retained_bytes_per_vector\": {raw_bv:.4},\n"
    ));
    out.push_str(&format!("  \"codes_bytes\": {},\n", ctx.mem.codes_bytes));
    out.push_str(&format!(
        "  \"calibration_or_codebook_bytes\": {},\n",
        ctx.mem.calibration_or_codebook_bytes
    ));
    out.push_str(&format!(
        "  \"aux_vector_bytes\": {},\n",
        ctx.mem.aux_vector_bytes
    ));
    out.push_str(&format!(
        "  \"graph_only_bytes\": {},\n",
        ctx.mem.graph_only_bytes
    ));
    out.push_str(&format!(
        "  \"node_meta_bytes\": {},\n",
        ctx.mem.node_meta_bytes
    ));
    out.push_str(&format!(
        "  \"layer0_edges_bytes\": {},\n",
        ctx.mem.layer0_edges_bytes
    ));
    out.push_str(&format!(
        "  \"upper_layers_edges_bytes\": {},\n",
        ctx.mem.upper_layers_edges_bytes
    ));
    out.push_str(&format!(
        "  \"raw_vectors_retained_bytes\": {},\n",
        ctx.mem.raw_vectors_retained_bytes
    ));
    out.push_str(&format!(
        "  \"steady_state_total_bytes\": {},\n",
        ctx.mem.steady_state_total_bytes
    ));
    out.push_str(&format!("  \"build_peak_rss_mb\": {build_rss_json},\n"));
    out.push_str(&format!(
        "  \"query_serving_peak_rss_mb\": {query_rss_json},\n"
    ));
    out.push_str(&format!(
        "  \"query_rss_profiled_isolated\": {},\n",
        ctx.query_rss_profiled_isolated
    ));

    out.push_str("  \"memory_profile\": {\n");
    out.push_str(&format!(
        "    \"steady_state_bytes_per_vector\": {:.4},\n",
        ctx.mem.steady_state_bytes_per_vector
    ));
    out.push_str(&format!(
        "    \"graph_only_bytes_per_vector\": {:.4},\n",
        ctx.mem.graph_only_bytes_per_vector
    ));
    out.push_str(&format!("    \"codes_bytes_per_vector\": {codes_bv:.4},\n"));
    out.push_str(&format!(
        "    \"calibration_or_codebook_bytes_per_vector\": {cal_bv:.4},\n"
    ));
    out.push_str(&format!(
        "    \"raw_vectors_retained_bytes_per_vector\": {raw_bv:.4},\n"
    ));
    out.push_str(&format!("    \"codes_bytes\": {},\n", ctx.mem.codes_bytes));
    out.push_str(&format!(
        "    \"calibration_or_codebook_bytes\": {},\n",
        ctx.mem.calibration_or_codebook_bytes
    ));
    out.push_str(&format!(
        "    \"aux_vector_bytes\": {},\n",
        ctx.mem.aux_vector_bytes
    ));
    out.push_str(&format!(
        "    \"graph_only_bytes\": {},\n",
        ctx.mem.graph_only_bytes
    ));
    out.push_str(&format!(
        "    \"raw_vectors_retained_bytes\": {},\n",
        ctx.mem.raw_vectors_retained_bytes
    ));
    out.push_str(&format!(
        "    \"steady_state_total_bytes\": {},\n",
        ctx.mem.steady_state_total_bytes
    ));
    out.push_str(&format!("    \"build_peak_rss_mb\": {build_rss_json},\n"));
    out.push_str(&format!(
        "    \"query_serving_peak_rss_mb\": {query_rss_json},\n"
    ));
    out.push_str(&format!("    \"peak_rss_scope\": \"{rss_scope}\"\n"));
    out.push_str("  },\n");

    out.push_str("  \"sweep\": [\n");
    for (idx, pt) in ctx.sweep.iter().enumerate() {
        let comma = if idx + 1 < ctx.sweep.len() { "," } else { "" };
        out.push_str("    {\n");
        out.push_str(&format!("      \"ef_search\": {},\n", pt.ef_search));
        out.push_str(&format!(
            "      \"effective_ef_search\": {},\n",
            pt.effective_ef_search
        ));
        out.push_str(&format!(
            "      \"rerank_top_r\": {},\n",
            pt.rerank_depth.json_rerank_top_r()
        ));
        out.push_str(&format!(
            "      \"is_full_beam\": {},\n",
            pt.rerank_depth.is_full_beam()
        ));
        out.push_str(&format!("      \"recall_at_k\": {:.6},\n", pt.recall_at_k));
        out.push_str(&format!(
            "      \"qps\": {:.2},\n",
            pt.summary.median_latency.qps
        ));
        out.push_str(&format!(
            "      \"mean_ms\": {:.6},\n",
            pt.summary.median_latency.mean_ms
        ));
        out.push_str(&format!(
            "      \"p50_ms\": {:.6},\n",
            pt.summary.median_latency.p50_ms
        ));
        out.push_str(&format!(
            "      \"p95_ms\": {:.6},\n",
            pt.summary.median_latency.p95_ms
        ));
        out.push_str(&format!(
            "      \"p99_ms\": {:.6},\n",
            pt.summary.median_latency.p99_ms
        ));
        out.push_str(&format!(
            "      \"min_ms\": {:.6},\n",
            pt.summary.median_latency.min_ms
        ));
        out.push_str(&format!(
            "      \"max_ms\": {:.6},\n",
            pt.summary.median_latency.max_ms
        ));
        out.push_str("      \"spread\": {\n");
        out.push_str(&format!(
            "        \"mean_ms_min\": {:.6},\n",
            pt.summary.mean_ms_min
        ));
        out.push_str(&format!(
            "        \"mean_ms_max\": {:.6},\n",
            pt.summary.mean_ms_max
        ));
        out.push_str(&format!(
            "        \"mean_ms_spread\": {:.6},\n",
            pt.summary.mean_ms_spread
        ));
        out.push_str(&format!(
            "        \"p99_ms_min\": {:.6},\n",
            pt.summary.p99_ms_min
        ));
        out.push_str(&format!(
            "        \"p99_ms_max\": {:.6},\n",
            pt.summary.p99_ms_max
        ));
        out.push_str(&format!(
            "        \"p99_ms_spread\": {:.6},\n",
            pt.summary.p99_ms_spread
        ));
        out.push_str("        \"run_mean_latencies_ms\": [");
        for (ri, &v) in pt.summary.run_mean_latencies_ms.iter().enumerate() {
            if ri > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{v:.6}"));
        }
        out.push_str("],\n");
        out.push_str("        \"run_p99_latencies_ms\": [");
        for (ri, &v) in pt.summary.run_p99_latencies_ms.iter().enumerate() {
            if ri > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{v:.6}"));
        }
        out.push_str("]\n");
        out.push_str("      }\n");
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn serialize_matrix_summary_json(
    summary: &Phase6MatrixSummary,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    config: &Phase6MatrixConfig,
) -> String {
    let mut out = String::with_capacity(8192);
    out.push_str("{\n");
    out.push_str("  \"phase\": 6,\n");
    out.push_str(&format!(
        "  \"mode\": \"{}\",\n",
        if config.is_pilot { "pilot" } else { "matrix" }
    ));
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        escape_json(&current_utc_rfc3339())
    ));
    prov.append_json_fields(&mut out, hw);
    append_hardware_json(&mut out, hw);
    out.push_str(&format!("  \"scale\": {},\n", summary.scale));
    out.push_str(&format!("  \"k\": {},\n", config.k));
    out.push_str(&format!("  \"reps\": {},\n", config.reps));
    out.push_str(&format!("  \"seed\": {},\n", config.seed));
    out.push_str(&format!(
        "  \"ground_truth_source\": \"{}\",\n",
        escape_json(&summary.ground_truth_source)
    ));
    out.push_str(&format!(
        "  \"ground_truth_cache_hit\": {},\n",
        summary.ground_truth_cache_hit
    ));
    out.push_str(&format!("  \"total_arms\": {},\n", summary.total_arms));
    out.push_str(&format!(
        "  \"newly_measured_count\": {},\n",
        summary.newly_measured_count
    ));
    out.push_str(&format!(
        "  \"resumed_skipped_count\": {},\n",
        summary.resumed_skipped_count
    ));
    out.push_str("  \"measured_this_run\": true,\n");
    out.push_str("  \"arms\": [\n");
    for (idx, arm) in summary.arms.iter().enumerate() {
        let comma = if idx + 1 < summary.arms.len() {
            ","
        } else {
            ""
        };
        let fp_json = arm
            .graph_fingerprint_hex
            .as_deref()
            .map_or_else(|| "null".to_string(), |s| format!("\"{}\"", escape_json(s)));
        let build_s_json = arm
            .total_build_time_s
            .map_or_else(|| "null".to_string(), |v| format!("{v:.6}"));
        let steady_json = arm
            .steady_state_bytes_per_vector
            .map_or_else(|| "null".to_string(), |v| format!("{v:.4}"));
        let build_rss_json = arm
            .build_peak_rss_mb
            .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));
        let query_rss_json = arm
            .query_serving_peak_rss_mb
            .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));
        let recall_json = arm
            .best_recall
            .map_or_else(|| "null".to_string(), |v| format!("{v:.6}"));
        let qps_json = arm
            .best_qps
            .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));

        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"arm_id\": \"{}\",\n",
            escape_json(&arm.arm_id)
        ));
        out.push_str(&format!(
            "      \"status\": \"{}\",\n",
            escape_json(&arm.status)
        ));
        out.push_str(&format!(
            "      \"report_filename\": \"{}\",\n",
            escape_json(&arm.report_filename)
        ));
        out.push_str(&format!(
            "      \"quantizer_kind\": \"{}\",\n",
            escape_json(&arm.quantizer_kind)
        ));
        out.push_str(&format!(
            "      \"storage_mode\": \"{}\",\n",
            escape_json(&arm.storage_mode)
        ));
        out.push_str(&format!(
            "      \"build_mode\": \"{}\",\n",
            escape_json(&arm.build_mode)
        ));
        out.push_str(&format!("      \"m\": {},\n", arm.m));
        out.push_str(&format!(
            "      \"ef_construction\": {},\n",
            arm.ef_construction
        ));
        out.push_str(&format!("      \"graph_fingerprint_hex\": {fp_json},\n"));
        out.push_str(&format!("      \"total_build_time_s\": {build_s_json},\n"));
        out.push_str(&format!(
            "      \"steady_state_bytes_per_vector\": {steady_json},\n"
        ));
        out.push_str(&format!("      \"build_peak_rss_mb\": {build_rss_json},\n"));
        out.push_str(&format!(
            "      \"query_serving_peak_rss_mb\": {query_rss_json},\n"
        ));
        out.push_str(&format!("      \"best_recall\": {recall_json},\n"));
        out.push_str(&format!("      \"best_qps\": {qps_json}\n"));
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Checks whether an existing per-arm JSON file matches the current clean commit and config.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn can_resume_existing_arm_report(
    report_path: &Path,
    expected_commit: &str,
    current_git_dirty: bool,
    source_kind: &str,
    scale: usize,
    dim: usize,
    k: usize,
    num_queries: usize,
    reps: usize,
    seed: u64,
    ef_search_list: &[usize],
    rerank_ladder: &[RerankDepthSpec],
    profile_query_rss: bool,
    arm: &Phase6ArmSpec,
) -> bool {
    if expected_commit == "unknown" || !report_path.is_file() {
        return false;
    }
    let Ok(content) = fs::read_to_string(report_path) else {
        return false;
    };
    if !current_git_dirty && content.contains("\"git_dirty\": true") {
        return false;
    }
    if extract_json_u64_field(&content, "phase") != Some(6)
        || extract_json_bool_field(&content, "measured_this_run") != Some(true)
        || extract_json_string_field(&content, "git_commit").as_deref() != Some(expected_commit)
        || extract_json_string_field(&content, "source_kind").as_deref() != Some(source_kind)
        || extract_json_string_field(&content, "arm_id").as_deref() != Some(arm.arm_id)
        || extract_json_string_field(&content, "quantizer_kind").as_deref()
            != Some(arm.quantizer_kind)
        || extract_json_string_field(&content, "storage_mode").as_deref() != Some(arm.storage_mode)
        || extract_json_string_field(&content, "build_mode").as_deref() != Some(arm.build_mode_str)
        || extract_json_u64_field(&content, "m") != Some(arm.m as u64)
        || extract_json_u64_field(&content, "ef_construction") != Some(arm.ef_construction as u64)
        || extract_json_u64_field(&content, "scale") != Some(scale as u64)
        || extract_json_u64_field(&content, "dim") != Some(dim as u64)
        || extract_json_u64_field(&content, "k") != Some(k as u64)
        || extract_json_u64_field(&content, "num_queries") != Some(num_queries as u64)
        || extract_json_u64_field(&content, "reps") != Some(reps as u64)
        || extract_json_u64_field(&content, "seed") != Some(seed)
    {
        return false;
    }

    if arm.isolated_snapshot_kind().is_some()
        && profile_query_rss
        && content.contains("\"query_serving_peak_rss_mb\": null")
    {
        return false;
    }

    let Some(sweep_slice) = extract_json_array_slice(&content, "sweep") else {
        return false;
    };
    let objects = split_top_level_json_objects(sweep_slice);
    let expected_len = ef_search_list.len() * rerank_ladder.len();
    if objects.len() != expected_len {
        return false;
    }

    let mut idx = 0_usize;
    for &r_spec in rerank_ladder {
        for &ef in ef_search_list {
            let obj = objects[idx];
            idx += 1;
            let (rerank_arg, expected_eff_ef) = r_spec.resolve_params(k, ef);
            if extract_json_u64_field(obj, "ef_search") != Some(ef as u64)
                || extract_json_u64_field(obj, "effective_ef_search")
                    != Some(expected_eff_ef as u64)
                || extract_json_bool_field(obj, "is_full_beam") != Some(r_spec.is_full_beam())
            {
                return false;
            }
            match r_spec {
                RerankDepthSpec::None => {
                    if !obj.contains("\"rerank_top_r\": null") {
                        return false;
                    }
                }
                RerankDepthSpec::Fixed(r) => {
                    let _ = rerank_arg;
                    if extract_json_u64_field(obj, "rerank_top_r") != Some(r as u64) {
                        return false;
                    }
                }
                RerankDepthSpec::FullBeam => {
                    if extract_json_string_field(obj, "rerank_top_r").as_deref()
                        != Some("full_beam")
                    {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn parse_resumed_arm_summary(
    report_path: &Path,
    arm: &Phase6ArmSpec,
    report_filename: &str,
) -> Phase6ArmSummaryEntry {
    let content = fs::read_to_string(report_path).unwrap_or_default();
    let mut best_recall: Option<f64> = None;
    let mut best_qps: Option<f64> = None;
    if let Some(sweep_slice) = extract_json_array_slice(&content, "sweep") {
        for obj in split_top_level_json_objects(sweep_slice) {
            if let Some(r) = extract_json_f64_field(obj, "recall_at_k") {
                best_recall = Some(best_recall.map_or(r, |b| b.max(r)));
            }
            if let Some(q) = extract_json_f64_field(obj, "qps") {
                best_qps = Some(best_qps.map_or(q, |b| b.max(q)));
            }
        }
    }

    Phase6ArmSummaryEntry {
        arm_id: arm.arm_id.to_string(),
        status: "resumed".to_string(),
        report_filename: report_filename.to_string(),
        quantizer_kind: arm.quantizer_kind.to_string(),
        storage_mode: arm.storage_mode.to_string(),
        build_mode: arm.build_mode_str.to_string(),
        m: arm.m,
        ef_construction: arm.ef_construction,
        graph_fingerprint_hex: extract_json_string_field(&content, "graph_fingerprint_hex"),
        total_build_time_s: extract_json_f64_field(&content, "total_build_time_s"),
        steady_state_bytes_per_vector: extract_json_f64_field(
            &content,
            "steady_state_bytes_per_vector",
        ),
        build_peak_rss_mb: extract_json_f64_field(&content, "build_peak_rss_mb"),
        query_serving_peak_rss_mb: extract_json_f64_field(&content, "query_serving_peak_rss_mb"),
        best_recall,
        best_qps,
    }
}

// ============================================================================
// 7. Main Matrix / Pilot Orchestrator
// ============================================================================

/// Executes the resumable Phase 6 evaluation matrix or pilot suite.
pub fn run_phase6_matrix(config: &Phase6MatrixConfig) -> Result<Phase6MatrixSummary> {
    config.validate()?;
    let mut prov = detect_provenance();
    if let Some(ref commit_override) = config.expected_commit_override {
        prov.git_commit = commit_override.clone();
    }
    apply_affinity_pinning(config.affinity.as_deref(), &mut prov)?;
    let hw = detect_hardware();

    fs::create_dir_all(&config.output_dir)?;

    let mut dataset = prepare_dataset_and_gt(config)?;
    let num_base = dataset.num_base;
    let dim = dataset.dim;
    let num_queries = dataset.query.rows();
    let source_kind = dataset.source.kind_tag().to_string();

    // Validate PQ arm subspace divisibility up front before spending any build time.
    let mut specs = Vec::with_capacity(config.arm_ids.len());
    for id in &config.arm_ids {
        let spec = resolve_arm_spec(id)?;
        if let Phase6ArmKind::Pq { m_pq } = spec.kind {
            if !dim.is_multiple_of(m_pq) {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Arm '{}' requires vector dim ({dim}) to be divisible by m_pq ({m_pq})",
                    spec.arm_id
                )));
            }
        }
        specs.push(spec);
    }

    // Partition arms into resumed vs pending, ordering streaming arms first when `base_cache`
    // is not yet loaded so streaming `build_peak_rss_mb` stays unpolluted by `RowMatrix<f32>`.
    let mut arm_entries: Vec<Option<Phase6ArmSummaryEntry>> = vec![None; specs.len()];
    let mut pending_indices: Vec<usize> = Vec::new();
    let mut resumed_skipped_count = 0_usize;

    for (idx, spec) in specs.iter().enumerate() {
        let filename = arm_report_filename(num_base, spec.arm_id);
        let path = config.output_dir.join(&filename);
        let rerank_ladder = spec.active_rerank_ladder(config.rerank_list_override.as_deref());
        if config.resume
            && can_resume_existing_arm_report(
                &path,
                &prov.git_commit,
                prov.git_dirty,
                &source_kind,
                num_base,
                dim,
                config.k,
                num_queries,
                config.reps,
                config.seed,
                &config.ef_search_list,
                &rerank_ladder,
                config.profile_query_rss,
                spec,
            )
        {
            resumed_skipped_count += 1;
            arm_entries[idx] = Some(parse_resumed_arm_summary(&path, spec, &filename));
        } else {
            pending_indices.push(idx);
        }
    }

    // Order pending arms: streaming PQ -> streaming SQ8 -> non-streaming arms (preserving order).
    pending_indices.sort_by_key(|&idx| {
        let s = &specs[idx];
        if dataset.base_fvecs_path.is_some() && s.supports_streaming_build() {
            match s.kind {
                Phase6ArmKind::Pq { .. } => (0_usize, idx),
                Phase6ArmKind::Sq8 => (1_usize, idx),
                Phase6ArmKind::F32 => (2_usize, idx),
            }
        } else {
            (2_usize, idx)
        }
    });

    let mut newly_measured_count = 0_usize;
    for idx in pending_indices {
        let spec = &specs[idx];
        let entry = execute_single_arm(spec, config, &prov, &hw, &mut dataset)?;
        arm_entries[idx] = Some(entry);
        newly_measured_count += 1;
    }

    let final_entries: Vec<Phase6ArmSummaryEntry> = arm_entries
        .into_iter()
        .map(|opt| opt.expect("every arm must be either resumed or measured"))
        .collect();

    let summary = Phase6MatrixSummary {
        scale: num_base,
        total_arms: specs.len(),
        newly_measured_count,
        resumed_skipped_count,
        output_dir: config.output_dir.clone(),
        summary_path: config.output_dir.join(format!("summary_n{num_base}.json")),
        ground_truth_source: dataset.gt_source.clone(),
        ground_truth_cache_hit: dataset.gt_cache_hit,
        arms: final_entries,
    };

    let summary_json = serialize_matrix_summary_json(&summary, &prov, &hw, config);
    fs::write(&summary.summary_path, summary_json)?;
    Ok(summary)
}

struct ArmBuildTimings {
    cal_or_train_s: f64,
    encode_s: f64,
    graph_s: f64,
    total_s: f64,
}

struct ArmExecutionOutcome {
    sweep: Vec<Phase6SweepPointResult>,
    timings: ArmBuildTimings,
    fp: u64,
    streaming_build: bool,
    pq_config_used: Option<PqConfig>,
    mem: QuantizedMemoryProfile,
    build_peak_rss_mb: Option<f64>,
    query_serving_peak_rss_mb: Option<f64>,
    query_rss_profiled_isolated: bool,
}

#[allow(clippy::too_many_arguments)]
fn finalize_quantized_arm<T: QuantizedSweepTarget>(
    index: T,
    snapshot_kind: SnapshotKind,
    spec: &Phase6ArmSpec,
    config: &Phase6MatrixConfig,
    dataset: &PreparedDatasetState,
    rerank_ladder: &[RerankDepthSpec],
    timings: ArmBuildTimings,
    streaming_build: bool,
    pq_config_used: Option<PqConfig>,
) -> Result<ArmExecutionOutcome> {
    let fp = index.q_graph_fingerprint();
    let mem = index.q_memory_profile();
    let build_peak_rss_mb = get_peak_rss_mb();
    let (query_serving_peak_rss_mb, query_rss_profiled_isolated) =
        if config.profile_query_rss && spec.isolated_snapshot_kind().is_some() {
            measure_isolated_query_rss(
                snapshot_kind,
                spec.arm_id,
                dataset.num_base,
                config.k,
                &config.data_dir,
                dataset.query_fvecs_path.as_deref(),
                &dataset.query,
                config.affinity.as_deref(),
                |p| index.q_write_snapshot(p),
            )?
        } else if matches!(spec.storage_enum, QuantizedStorageMode::WithExactRerank) {
            (build_peak_rss_mb, false)
        } else {
            (None, false)
        };

    let sweep = evaluate_quantized_sweep(
        &index,
        &dataset.query,
        &dataset.gt_matrix,
        config.k,
        config.reps,
        &config.ef_search_list,
        rerank_ladder,
    )?;
    drop(index);
    Ok(ArmExecutionOutcome {
        sweep,
        timings,
        fp,
        streaming_build,
        pq_config_used,
        mem,
        build_peak_rss_mb,
        query_serving_peak_rss_mb,
        query_rss_profiled_isolated,
    })
}

fn execute_single_arm(
    spec: &Phase6ArmSpec,
    config: &Phase6MatrixConfig,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    dataset: &mut PreparedDatasetState,
) -> Result<Phase6ArmSummaryEntry> {
    let hnsw_cfg = HnswConfig {
        m: spec.m,
        ef_construction: spec.ef_construction,
        seed: config.seed,
        ..HnswConfig::default()
    };
    let rerank_ladder = spec.active_rerank_ladder(config.rerank_list_override.as_deref());
    let filename = arm_report_filename(dataset.num_base, spec.arm_id);
    let report_path = config.output_dir.join(&filename);

    eprintln!(
        "[INFO] Running Phase 6 arm '{}' (M={}, efC={}, scale={}, Q={})...",
        spec.arm_id,
        spec.m,
        spec.ef_construction,
        dataset.num_base,
        dataset.query.rows()
    );

    let outcome = match spec.kind {
        Phase6ArmKind::F32 => {
            let base = dataset.get_or_load_base()?;
            let t0 = Instant::now();
            let index = HnswIndex::build(&base, Metric::L2Squared, hnsw_cfg)?;
            let build_s = t0.elapsed().as_secs_f64();
            let fp = index.graph_fingerprint();
            let comps = index.graph_components();
            let mem = QuantizedMemoryProfile::from_parts(
                index.num_vectors(),
                0,
                0,
                0,
                comps.node_structs_bytes,
                comps.layer0_edges_bytes,
                comps.upper_layers_edges_bytes,
                base.memory_bytes(),
            );
            let build_rss = get_peak_rss_mb();
            let sweep = evaluate_f32_sweep(
                &index,
                &dataset.query,
                &dataset.gt_matrix,
                config.k,
                config.reps,
                &config.ef_search_list,
            )?;
            drop(index);
            ArmExecutionOutcome {
                sweep,
                timings: ArmBuildTimings {
                    cal_or_train_s: 0.0,
                    encode_s: 0.0,
                    graph_s: build_s,
                    total_s: build_s,
                },
                fp,
                streaming_build: false,
                pq_config_used: None,
                mem,
                build_peak_rss_mb: build_rss,
                query_serving_peak_rss_mb: build_rss,
                query_rss_profiled_isolated: false,
            }
        }
        Phase6ArmKind::Sq8 => {
            let (index, streaming) = if spec.supports_streaming_build() {
                if let Some(ref base_path) = dataset.base_fvecs_path {
                    (
                        HnswSq8Index::build_streaming_from_fvecs(
                            base_path,
                            Metric::L2Squared,
                            hnsw_cfg,
                            Some(dataset.num_base),
                        )?,
                        true,
                    )
                } else {
                    let base = dataset.get_or_load_base()?;
                    (
                        HnswSq8Index::build(
                            &base,
                            Metric::L2Squared,
                            hnsw_cfg,
                            spec.build_mode,
                            spec.storage_enum,
                        )?,
                        false,
                    )
                }
            } else {
                let base = dataset.get_or_load_base()?;
                (
                    HnswSq8Index::build_with_shared_raw(
                        base,
                        Metric::L2Squared,
                        hnsw_cfg,
                        spec.build_mode,
                        spec.storage_enum,
                        Sq8KernelKind::QueryShifted,
                    )?,
                    false,
                )
            };

            let timings = index.build_timings();
            finalize_quantized_arm(
                index,
                SnapshotKind::Sq8,
                spec,
                config,
                dataset,
                &rerank_ladder,
                ArmBuildTimings {
                    cal_or_train_s: timings.calibration_or_train_s,
                    encode_s: timings.encode_s,
                    graph_s: timings.graph_build_s,
                    total_s: timings.total_s,
                },
                streaming,
                None,
            )?
        }
        Phase6ArmKind::Pq { m_pq } => {
            let pq_cfg = PqConfig {
                m_pq,
                k_centroids: PQ_K_CENTROIDS,
                max_train_vectors: PQ_DEFAULT_MAX_TRAIN_VECTORS,
                max_lloyd_iters: PQ_DEFAULT_MAX_LLOYD_ITERS,
                seed: config.seed,
            };
            let (index, streaming) = if spec.supports_streaming_build() {
                if let Some(ref base_path) = dataset.base_fvecs_path {
                    (
                        HnswPqIndex::build_streaming_from_fvecs(
                            base_path,
                            Metric::L2Squared,
                            hnsw_cfg,
                            pq_cfg,
                            Some(dataset.num_base),
                        )?,
                        true,
                    )
                } else {
                    let base = dataset.get_or_load_base()?;
                    (
                        HnswPqIndex::build(
                            &base,
                            Metric::L2Squared,
                            hnsw_cfg,
                            pq_cfg,
                            spec.build_mode,
                            spec.storage_enum,
                        )?,
                        false,
                    )
                }
            } else {
                let base = dataset.get_or_load_base()?;
                (
                    HnswPqIndex::build_with_shared_raw(
                        base,
                        Metric::L2Squared,
                        hnsw_cfg,
                        pq_cfg,
                        spec.build_mode,
                        spec.storage_enum,
                    )?,
                    false,
                )
            };

            let timings = index.build_timings();
            finalize_quantized_arm(
                index,
                SnapshotKind::Pq,
                spec,
                config,
                dataset,
                &rerank_ladder,
                ArmBuildTimings {
                    cal_or_train_s: timings.train_time_s,
                    encode_s: timings.encode_time_s,
                    graph_s: timings.graph_build_time_s,
                    total_s: timings.total_s,
                },
                streaming,
                Some(pq_cfg),
            )?
        }
    };

    let ctx = ArmReportContext {
        spec,
        config,
        prov,
        hw,
        dataset,
        training_or_calibration_time_s: outcome.timings.cal_or_train_s,
        encode_time_s: outcome.timings.encode_s,
        graph_build_time_s: outcome.timings.graph_s,
        total_build_time_s: outcome.timings.total_s,
        graph_fingerprint: outcome.fp,
        streaming_build: outcome.streaming_build,
        pq_config: outcome.pq_config_used,
        mem: outcome.mem,
        build_peak_rss_mb: outcome.build_peak_rss_mb,
        query_serving_peak_rss_mb: outcome.query_serving_peak_rss_mb,
        query_rss_profiled_isolated: outcome.query_rss_profiled_isolated,
        sweep: &outcome.sweep,
    };
    let json = serialize_arm_report_json(&ctx);
    fs::write(&report_path, json)?;

    let best_recall = outcome
        .sweep
        .iter()
        .map(|p| p.recall_at_k)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let best_qps = outcome
        .sweep
        .iter()
        .map(|p| p.summary.median_latency.qps)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    Ok(Phase6ArmSummaryEntry {
        arm_id: spec.arm_id.to_string(),
        status: "measured".to_string(),
        report_filename: filename,
        quantizer_kind: spec.quantizer_kind.to_string(),
        storage_mode: spec.storage_mode.to_string(),
        build_mode: spec.build_mode_str.to_string(),
        m: spec.m,
        ef_construction: spec.ef_construction,
        graph_fingerprint_hex: Some(format!("0x{:016x}", outcome.fp)),
        total_build_time_s: Some(outcome.timings.total_s),
        steady_state_bytes_per_vector: Some(outcome.mem.steady_state_bytes_per_vector),
        build_peak_rss_mb: outcome.build_peak_rss_mb,
        query_serving_peak_rss_mb: outcome.query_serving_peak_rss_mb,
        best_recall,
        best_qps,
    })
}

// ============================================================================
// 8. Minimal String-Escape-Aware JSON Helpers (for --resume)
// ============================================================================

fn extract_json_string_field(content: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\": \"");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let bytes = content.as_bytes();
    let mut i = start;
    let mut out = String::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            return Some(out);
        }
        if b == b'\\' && i + 1 < bytes.len() {
            let esc = bytes[i + 1];
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                other => out.push(other as char),
            }
            i += 2;
        } else {
            out.push(b as char);
            i += 1;
        }
    }
    None
}

fn extract_json_f64_field(content: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\": ");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let rem = content[start..].trim_start();
    let val_str: String = rem
        .chars()
        .take_while(|c| {
            c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || *c == 'e' || *c == 'E'
        })
        .collect();
    val_str.parse().ok()
}

fn extract_json_u64_field(content: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\": ");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let rem = content[start..].trim_start();
    let val_str: String = rem.chars().take_while(|c| c.is_ascii_digit()).collect();
    val_str.parse().ok()
}

fn extract_json_bool_field(content: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{key}\": ");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let rem = content[start..].trim_start();
    if rem.starts_with("true") {
        Some(true)
    } else if rem.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn extract_json_array_slice<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{key}\":");
    let pos = content.find(&pattern)?;
    let after = &content[pos + pattern.len()..];
    let open_rel = after.find('[')?;
    let bytes = after.as_bytes();
    let mut depth = 0_usize;
    let mut in_str = false;
    let mut escape = false;

    for (idx, &b) in bytes.iter().enumerate().skip(open_rel) {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' => depth += 1,
            b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&after[open_rel + 1..idx]);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_json_objects(slice: &str) -> Vec<&str> {
    let bytes = slice.as_bytes();
    let mut objs = Vec::new();
    let mut depth = 0_usize;
    let mut obj_start: Option<usize> = None;
    let mut in_str = false;
    let mut escape = false;

    for (idx, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    obj_start = Some(idx);
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(start) = obj_start.take() {
                        objs.push(&slice[start..=idx]);
                    }
                }
            }
            _ => {}
        }
    }
    objs
}
