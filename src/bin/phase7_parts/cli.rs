//! CLI Argument Parsing, Validation, and Affinity / Process Isolation Helpers for Phase 7.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use vector_search::benchmark::ReportProvenance;
use vector_search::cli_util::{next_str, parse_next, parse_positive_csv_usizes, SiftPresetChoice};
use vector_search::error::{Result, VectorSearchError};
use vector_search::hardware::{parse_affinity_mask, pin_process_affinity_os};
use vector_search::hnsw::HnswConfig;

pub const DEFAULT_EF_SEARCH_LIST: [usize; 12] =
    [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareMode {
    Single,
    Multithread,
}

impl CompareMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Multithread => "multithread",
        }
    }

    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "single" => Ok(Self::Single),
            "multithread" => Ok(Self::Multithread),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid --compare-mode '{other}': expected 'single' or 'multithread'"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Phase7CliConfig {
    pub dataset: SiftPresetChoice,
    pub base_fvecs: Option<PathBuf>,
    pub query_fvecs: Option<PathBuf>,
    pub gt_path: Option<PathBuf>,
    pub data_dir: PathBuf,
    pub scale: Option<usize>,
    pub queries: Option<usize>,
    pub dim: usize,
    pub num_clusters: usize,
    pub k: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search_list: Vec<usize>,
    pub seed: u64,
    pub keep_pruned_shrink: Option<bool>,
    pub config_id: Option<String>,
    pub build_reps: Option<usize>,
    pub query_reps: usize,
    pub compare_mode: CompareMode,
    pub threads: usize,
    pub affinity: Option<String>,
    pub synthetic: bool,
    pub offline: bool,
    pub no_gt_cache: bool,
    pub prepare_100k_gt: bool,
    pub json_out: Option<PathBuf>,
    pub profile_rss_isolated: bool,
    pub internal_isolated_child: bool,
}

impl Default for Phase7CliConfig {
    fn default() -> Self {
        Self {
            dataset: SiftPresetChoice::Sift1M,
            base_fvecs: None,
            query_fvecs: None,
            gt_path: None,
            data_dir: PathBuf::from("data"),
            scale: None,
            queries: None,
            dim: 128,
            num_clusters: 10,
            k: 10,
            m: 16,
            ef_construction: 100,
            ef_search_list: DEFAULT_EF_SEARCH_LIST.to_vec(),
            seed: 42,
            keep_pruned_shrink: None,
            config_id: None,
            build_reps: None,
            query_reps: 5,
            compare_mode: CompareMode::Single,
            threads: 1,
            affinity: None,
            synthetic: false,
            offline: false,
            no_gt_cache: false,
            prepare_100k_gt: false,
            json_out: None,
            profile_rss_isolated: false,
            internal_isolated_child: false,
        }
    }
}

impl Phase7CliConfig {
    pub fn validate(&self) -> Result<()> {
        if self.k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--k must be positive (> 0)".to_string(),
            ));
        }
        if self.scale == Some(0) {
            return Err(VectorSearchError::InvalidArgument(
                "--scale must be positive (> 0)".to_string(),
            ));
        }
        if self.queries == Some(0) {
            return Err(VectorSearchError::InvalidArgument(
                "--queries must be positive (> 0)".to_string(),
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
        if self.query_reps == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--query-reps must be positive (> 0)".to_string(),
            ));
        }
        if self.build_reps == Some(0) {
            return Err(VectorSearchError::InvalidArgument(
                "--build-reps must be positive (> 0)".to_string(),
            ));
        }
        if self.threads == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "--threads must be positive (> 0)".to_string(),
            ));
        }
        if self.compare_mode == CompareMode::Single && self.threads != 1 {
            return Err(VectorSearchError::InvalidArgument(
                "--compare-mode single requires --threads 1 (use --compare-mode multithread for multi-thread Mode 1)".to_string(),
            ));
        }
        if self.ef_search_list.is_empty() || self.ef_search_list.contains(&0) {
            return Err(VectorSearchError::InvalidArgument(
                "--ef-search-list must contain only positive integers".to_string(),
            ));
        }
        if self.base_fvecs.is_some() != self.query_fvecs.is_some() {
            return Err(VectorSearchError::InvalidArgument(
                "Both --base-fvecs and --query-fvecs must be provided together".to_string(),
            ));
        }
        if self.synthetic && self.base_fvecs.is_some() {
            return Err(VectorSearchError::InvalidArgument(
                "Cannot combine --synthetic with --base-fvecs/--query-fvecs".to_string(),
            ));
        }
        if let Some(ref cid) = self.config_id {
            if cid.trim().is_empty() {
                return Err(VectorSearchError::InvalidArgument(
                    "--config-id cannot be empty".to_string(),
                ));
            }
        }
        self.to_hnsw_config().validate()?;
        Ok(())
    }

    pub fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            ef_construction: self.ef_construction,
            keep_pruned: true,
            keep_pruned_shrink: self.keep_pruned_shrink,
            seed: self.seed,
            ..HnswConfig::default()
        }
    }

    pub fn resolve_config_id(&self) -> String {
        if let Some(ref explicit) = self.config_id {
            return explicit.trim().to_string();
        }
        if self.keep_pruned_shrink == Some(false) {
            if (self.m, self.ef_construction) == (16, 100) {
                return "C2-nkp".to_string();
            }
            return format!("M{}_efC{}-nkp", self.m, self.ef_construction);
        }
        match (self.m, self.ef_construction) {
            (8, 100) => "C1".to_string(),
            (16, 100) => "C2".to_string(),
            (32, 200) => "C3".to_string(),
            (48, 400) => "C4".to_string(),
            _ => format!("M{}_efC{}", self.m, self.ef_construction),
        }
    }
}

pub fn print_usage() {
    println!(
        r#"Phase 7 Head-to-Head Rust HNSW Benchmark Runner (Milestone M5 / Task T5.1)

USAGE:
    phase7 [OPTIONS]

OPTIONS:
    --dataset <sift1m|siftsmall>
                            Named Texmex dataset preset (default: sift1m)
    --base-fvecs <PATH>     Optional override path to base .fvecs file
    --query-fvecs <PATH>    Optional override path to query .fvecs file
    --gt-path <PATH>        Optional path to .ivecs ground truth (or output path with --prepare-100k-gt)
    --data-dir <PATH>       Dataset and GT cache directory (default: data)
    --scale <USIZE>         Base vector count (default: 100000 for sift1m/synthetic, 10000 for siftsmall)
    --queries <USIZE>       Query vector count (default: 10000 for sift1m/synthetic, 100 for siftsmall)
    --dim <USIZE>           Vector dimensionality for --synthetic (default: 128)
    --num-clusters <USIZE>  Cluster count for --synthetic (default: 10)
    --k <USIZE>             Top-k nearest neighbors (default: 10)
    --m <USIZE>             HNSW degree parameter M (default: 16)
    --ef-construction <USIZE>
                            HNSW efConstruction parameter (default: 100)
    --ef-search-list <CSV>  Comma-separated efSearch sweep ladder
                            (default: 10,20,30,40,50,70,100,150,200,300,400,600)
    --seed <U64>            Deterministic PRNG seed (default: 42)
    --no-keep-pruned-shrink Disable keep_pruned during reverse-edge shrink (config_id e.g. C2-nkp)
    --config-id <STR>       Optional explicit config label (e.g. C1, C2, C3, C2-nkp)
    --build-reps <USIZE>    Index build repetitions (default: 3 for N < 500k, 1 for N >= 500k)
    --query-reps <USIZE>    Timed query sweep passes after 1 untimed warmup (default: 5)
    --compare-mode <single|multithread>
                            single (Mode 1 batch + Mode 2 per-query) or multithread (Mode 1 batch)
    --threads <USIZE>       Thread count for Mode 1 search_batch_parallel (default: 1)
    --affinity <MASK>       Optional hex/decimal CPU affinity mask (e.g. 0x0004, 0x0555, 0xFFFF)
    --synthetic             Use deterministic offline synthetic vectors (never silent fallback if absent)
    --offline               Disallow network download of SIFT datasets
    --no-gt-cache           Disable reading/writing prefix GT cache files
    --prepare-100k-gt       Compute exact brute-force GT via ExactKnn::search_parallel_with_base,
                            verify self-recall, write .ivecs (default data/gt_<scale>_k100.ivecs), and exit
    --json-out <PATH>       Output path to write canonical Phase 7 JSON report
    --profile-rss-isolated  Spawn a fresh child process for uncontaminated PeakWorkingSet64
    -h, --help              Print help information"#
    );
}

pub enum ParseOutcome {
    Help,
    Run(Box<Phase7CliConfig>),
}

pub fn parse_cli_args(args: &[String]) -> Result<ParseOutcome> {
    let mut cfg = Phase7CliConfig::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(ParseOutcome::Help),
            "--dataset" => {
                let val = next_str(args, &mut i, "--dataset")?;
                cfg.dataset = SiftPresetChoice::parse(val)?;
            }
            "--base-fvecs" => {
                cfg.base_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                cfg.query_fvecs = Some(PathBuf::from(next_str(args, &mut i, "--query-fvecs")?));
            }
            "--gt-path" => {
                cfg.gt_path = Some(PathBuf::from(next_str(args, &mut i, "--gt-path")?));
            }
            "--data-dir" => {
                cfg.data_dir = PathBuf::from(next_str(args, &mut i, "--data-dir")?);
            }
            "--scale" => {
                cfg.scale = Some(parse_next(args, &mut i, "--scale")?);
            }
            "--queries" => {
                cfg.queries = Some(parse_next(args, &mut i, "--queries")?);
            }
            "--dim" => {
                cfg.dim = parse_next(args, &mut i, "--dim")?;
            }
            "--num-clusters" => {
                cfg.num_clusters = parse_next(args, &mut i, "--num-clusters")?;
            }
            "--k" => {
                cfg.k = parse_next(args, &mut i, "--k")?;
            }
            "--m" => {
                cfg.m = parse_next(args, &mut i, "--m")?;
            }
            "--ef-construction" => {
                cfg.ef_construction = parse_next(args, &mut i, "--ef-construction")?;
            }
            "--ef-search-list" => {
                let raw = next_str(args, &mut i, "--ef-search-list")?;
                cfg.ef_search_list = parse_positive_csv_usizes(raw, "--ef-search-list")?;
            }
            "--seed" => {
                cfg.seed = parse_next(args, &mut i, "--seed")?;
            }
            "--no-keep-pruned-shrink" => {
                cfg.keep_pruned_shrink = Some(false);
                i += 1;
            }
            "--config-id" => {
                cfg.config_id = Some(next_str(args, &mut i, "--config-id")?.to_string());
            }
            "--build-reps" => {
                cfg.build_reps = Some(parse_next(args, &mut i, "--build-reps")?);
            }
            "--query-reps" => {
                cfg.query_reps = parse_next(args, &mut i, "--query-reps")?;
            }
            "--compare-mode" => {
                let raw = next_str(args, &mut i, "--compare-mode")?;
                cfg.compare_mode = CompareMode::parse(raw)?;
            }
            "--threads" => {
                cfg.threads = parse_next(args, &mut i, "--threads")?;
            }
            "--affinity" => {
                cfg.affinity = Some(next_str(args, &mut i, "--affinity")?.to_string());
            }
            "--synthetic" => {
                cfg.synthetic = true;
                i += 1;
            }
            "--offline" => {
                cfg.offline = true;
                i += 1;
            }
            "--no-gt-cache" => {
                cfg.no_gt_cache = true;
                i += 1;
            }
            "--prepare-100k-gt" => {
                cfg.prepare_100k_gt = true;
                i += 1;
            }
            "--json-out" => {
                cfg.json_out = Some(PathBuf::from(next_str(args, &mut i, "--json-out")?));
            }
            "--profile-rss-isolated" => {
                cfg.profile_rss_isolated = true;
                i += 1;
            }
            "--internal-isolated-child" => {
                cfg.internal_isolated_child = true;
                i += 1;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI argument: '{unknown}'"
                )));
            }
        }
    }

    cfg.validate()?;
    Ok(ParseOutcome::Run(Box::new(cfg)))
}

pub fn spawn_isolated_child(raw_args: &[String]) -> Result<i32> {
    let exe = env::current_exe().map_err(VectorSearchError::Io)?;
    let mut cmd = Command::new(exe);
    for arg in raw_args {
        if arg != "--profile-rss-isolated" {
            cmd.arg(arg);
        }
    }
    if !raw_args.iter().any(|a| a == "--internal-isolated-child") {
        cmd.arg("--internal-isolated-child");
    }
    let status = cmd.status().map_err(VectorSearchError::Io)?;
    Ok(status.code().unwrap_or(1))
}

pub fn apply_affinity_pinning(
    affinity_arg: Option<&str>,
    prov: &ReportProvenance,
) -> Result<(String, bool)> {
    let Some(raw) = affinity_arg else {
        let pinned = prov.processor_affinity_mask != "not_set";
        return Ok((prov.processor_affinity_mask.clone(), pinned));
    };

    let requested_mask = parse_affinity_mask(raw)?;
    let verified_mask = pin_process_affinity_os(requested_mask)?;
    Ok((format!("0x{verified_mask:04x}"), true))
}
