//! Phase 6 CLI configuration, arm specifications, rerank depth ladders, and argument parsing.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::benchmark::ReportProvenance;
use crate::cli_util::{next_str, parse_next, parse_positive_csv_usizes, SiftPresetChoice};
use crate::error::{Result, VectorSearchError};
use crate::hardware::{parse_affinity_mask, pin_process_affinity_os};
use crate::quantized::{
    QuantizedBuildMode, QuantizedStorageMode, PQ_DEFAULT_MAX_LLOYD_ITERS,
    PQ_DEFAULT_MAX_TRAIN_VECTORS,
};

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

const SINGLE_NONE_RERANK: [RerankDepthSpec; 1] = [RerankDepthSpec::None];

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
pub type Phase6DatasetChoice = SiftPresetChoice;

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
        match trimmed.to_ascii_lowercase().as_str() {
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

const fn make_arm_spec(
    arm_id: &'static str,
    kind: Phase6ArmKind,
    build_mode: QuantizedBuildMode,
    storage_enum: QuantizedStorageMode,
    m: usize,
    ef_construction: usize,
) -> Phase6ArmSpec {
    let quantizer_kind = match kind {
        Phase6ArmKind::F32 => "f32",
        Phase6ArmKind::Sq8 => "sq8",
        Phase6ArmKind::Pq { m_pq: 16 } => "pq16",
        Phase6ArmKind::Pq { m_pq: 32 } => "pq32",
        Phase6ArmKind::Pq { .. } => "pq",
    };
    let storage_mode = match (kind, storage_enum) {
        (Phase6ArmKind::F32, _) => "f32",
        (_, QuantizedStorageMode::QuantizedOnly) => "only",
        (_, QuantizedStorageMode::WithExactRerank) => "rerank",
    };
    let build_mode_str = match build_mode {
        QuantizedBuildMode::Float32GraphThenQuantize => "f32_graph",
        QuantizedBuildMode::QuantizedDirect => "quantized_direct",
    };
    let default_reranks: &'static [RerankDepthSpec] = match (kind, storage_enum) {
        (Phase6ArmKind::F32, _) | (_, QuantizedStorageMode::QuantizedOnly) => &SINGLE_NONE_RERANK,
        (Phase6ArmKind::Sq8, QuantizedStorageMode::WithExactRerank) => &DEFAULT_SQ8_RERANK_LADDER,
        (Phase6ArmKind::Pq { .. }, QuantizedStorageMode::WithExactRerank) => {
            &DEFAULT_PQ_RERANK_LADDER
        }
    };
    Phase6ArmSpec {
        arm_id,
        kind,
        quantizer_kind,
        storage_mode,
        build_mode_str,
        build_mode,
        storage_enum,
        m,
        ef_construction,
        default_reranks,
    }
}

/// Resolves a canonical Phase 6 arm ID into its [`Phase6ArmSpec`].
pub fn resolve_arm_spec(raw_id: &str) -> Result<Phase6ArmSpec> {
    use Phase6ArmKind::{Pq, Sq8, F32};
    use QuantizedBuildMode::{Float32GraphThenQuantize as F32Graph, QuantizedDirect as QDirect};
    use QuantizedStorageMode::{QuantizedOnly as Only, WithExactRerank as Rerank};

    let spec = match raw_id.trim() {
        "f32_m16" => make_arm_spec("f32_m16", F32, F32Graph, Rerank, 16, 100),
        "sq8_only_f32graph" => make_arm_spec("sq8_only_f32graph", Sq8, F32Graph, Only, 16, 100),
        "sq8_only_qbuild" => make_arm_spec("sq8_only_qbuild", Sq8, QDirect, Only, 16, 100),
        "sq8_rerank_f32graph" => {
            make_arm_spec("sq8_rerank_f32graph", Sq8, F32Graph, Rerank, 16, 100)
        }
        "sq8_only_m32" => make_arm_spec("sq8_only_m32", Sq8, F32Graph, Only, 32, 200),
        "sq8_rerank_m32" => make_arm_spec("sq8_rerank_m32", Sq8, F32Graph, Rerank, 32, 200),
        "pq16_only_f32graph" => make_arm_spec(
            "pq16_only_f32graph",
            Pq { m_pq: 16 },
            F32Graph,
            Only,
            16,
            100,
        ),
        "pq32_only_f32graph" => make_arm_spec(
            "pq32_only_f32graph",
            Pq { m_pq: 32 },
            F32Graph,
            Only,
            16,
            100,
        ),
        "pq16_only_qbuild" => {
            make_arm_spec("pq16_only_qbuild", Pq { m_pq: 16 }, QDirect, Only, 16, 100)
        }
        "pq16_rerank_f32graph" => make_arm_spec(
            "pq16_rerank_f32graph",
            Pq { m_pq: 16 },
            F32Graph,
            Rerank,
            16,
            100,
        ),
        "pq32_rerank_f32graph" => make_arm_spec(
            "pq32_rerank_f32graph",
            Pq { m_pq: 32 },
            F32Graph,
            Rerank,
            16,
            100,
        ),
        other => {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Unknown Phase 6 arm_id '{other}'. Valid arms: {}",
                CANONICAL_MATRIX_ARM_IDS.join(", ")
            )));
        }
    };
    Ok(spec)
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
        for (val, flag) in [
            (self.scale, "--scale"),
            (self.queries, "--queries"),
            (self.k, "--k"),
            (self.reps, "--reps"),
            (self.dim, "--dim"),
            (self.num_clusters, "--num-clusters"),
        ] {
            if val == 0 {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "{flag} must be positive (> 0)"
                )));
            }
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
                internal_snapshot = Some(PathBuf::from(next_str(
                    args,
                    &mut i,
                    "--internal-profile-query",
                )?));
            }
            "--kind" => {
                let raw = next_str(args, &mut i, "--kind")?;
                snapshot_kind = Some(SnapshotKind::parse(raw)?);
            }
            "--dataset" => {
                let raw = next_str(args, &mut i, "--dataset")?;
                dataset = Phase6DatasetChoice::parse(raw)?;
                dataset_explicit = true;
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                query_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--query-fvecs")?));
            }
            "--gt-path" => {
                gt_path = Some(PathBuf::from(next_str(args, &mut i, "--gt-path")?));
            }
            "--data-dir" => {
                data_dir = PathBuf::from(next_str(args, &mut i, "--data-dir")?);
            }
            "--output-dir" => {
                output_dir = PathBuf::from(next_str(args, &mut i, "--output-dir")?);
                output_dir_explicit = true;
            }
            "--output" | "-o" | "--json-out" => {
                let flag = args[i].as_str();
                output_file = Some(PathBuf::from(next_str(args, &mut i, flag)?));
            }
            "--scale" | "--num-vectors" => {
                let flag = args[i].as_str();
                scale = Some(parse_next::<usize>(args, &mut i, flag)?);
            }
            "--queries" => {
                queries = Some(parse_next::<usize>(args, &mut i, "--queries")?);
            }
            "--k" => {
                k = parse_next::<usize>(args, &mut i, "--k")?;
            }
            "--ef" => {
                ef = parse_next::<usize>(args, &mut i, "--ef")?;
                ef_explicit = true;
            }
            "--reps" => {
                reps = Some(parse_next::<usize>(args, &mut i, "--reps")?);
            }
            "--seed" => {
                seed = parse_next::<u64>(args, &mut i, "--seed")?;
            }
            "--dim" => {
                dim = parse_next::<usize>(args, &mut i, "--dim")?;
            }
            "--num-clusters" => {
                num_clusters = parse_next::<usize>(args, &mut i, "--num-clusters")?;
            }
            "--evals" | "--dists" => {
                let flag = args[i].as_str();
                num_evals = parse_next::<usize>(args, &mut i, flag)?;
            }
            "--m-pq" => {
                m_pq = parse_next::<usize>(args, &mut i, "--m-pq")?;
            }
            "--max-train-vectors" => {
                max_train_vectors = parse_next::<usize>(args, &mut i, "--max-train-vectors")?;
            }
            "--max-lloyd-iters" => {
                max_lloyd_iters = parse_next::<usize>(args, &mut i, "--max-lloyd-iters")?;
            }
            "--arms" => {
                let raw = next_str(args, &mut i, "--arms")?;
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
                let raw = next_str(args, &mut i, "--ef-search-list")?;
                ef_search_list = parse_positive_csv_usizes(raw, "--ef-search-list")?;
            }
            "--rerank-list" => {
                let raw = next_str(args, &mut i, "--rerank-list")?;
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
                affinity = Some(next_str(args, &mut i, "--affinity")?.to_string());
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
            let is_siftsmall = dataset_explicit && dataset == Phase6DatasetChoice::SiftSmall;
            let resolved_scale = scale.unwrap_or(if is_siftsmall { 10_000 } else { 100_000 });
            let resolved_queries = queries.unwrap_or(if is_siftsmall { 100 } else { 10_000 });
            let resolved_reps = reps.unwrap_or(if resolved_scale >= 1_000_000 { 5 } else { 3 });
            let resolved_arms = arms.unwrap_or_else(|| {
                let slice: &[&str] = if is_pilot {
                    &PILOT_ARM_IDS
                } else {
                    &CANONICAL_MATRIX_ARM_IDS
                };
                slice.iter().map(|s| (*s).to_string()).collect()
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

pub(crate) fn apply_affinity_pinning(
    affinity_arg: Option<&str>,
    prov: &mut ReportProvenance,
) -> Result<()> {
    let Some(raw) = affinity_arg else {
        return Ok(());
    };
    let requested_mask = parse_affinity_mask(raw)?;
    let verified_mask = pin_process_affinity_os(requested_mask)?;
    prov.processor_affinity_mask = format!("0x{verified_mask:04x}");
    Ok(())
}
