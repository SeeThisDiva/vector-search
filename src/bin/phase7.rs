//! Phase 7 Head-to-Head Rust HNSW Benchmark Runner (Milestone M5 / Task T5.1).
//!
//! Benchmarks `HnswIndex` (`Metric::L2Squared`, `#![forbid(unsafe_code)]`) head-to-head
//! against FAISS `IndexHNSWFlat` using the canonical Phase 7 JSON schema consumed by
//! `scripts/plot_phase7.py`:
//! - Mode 1 pure-batch throughput via `HnswIndex::search_batch_parallel` (`1` thread in
//!   `single` mode, or `--threads` in `multithread` mode), eliminating per-query `Instant::now()`.
//! - Mode 2 per-query latency distribution (`p50`, `p95`, `p99`, `min`, `max`) via
//!   `HnswIndex::search_batch_timed_with_tracker` (in `single` mode), verifying bit-exact
//!   recall parity with Mode 1 and reporting `ffi_delta_us = 0.0` plus `instant_overhead_us`.
//! - Optional diagnostic `dist_evals_mean` pass outside timed loops when compiled with
//!   `--features instrument` (`null` otherwise).
//! - Prefix `.ivecs` ground-truth caching and `--prepare-100k-gt` generation via
//!   `ExactKnn::search_parallel_with_base`.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use vector_search::benchmark::{
    current_utc_rfc3339, detect_provenance, escape_json, opt_path_json,
    summarize_repeated_latencies, RepeatedSweepSummary, ReportProvenance,
};
use vector_search::dataset::{
    generate_synthetic_sift, load_custom_dataset_with_options, load_or_download_sift_with_options,
    read_ivecs_max_rows, write_ivecs, DatasetSource, LoadedDataset, SiftPreset,
};
use vector_search::distance::Metric;
use vector_search::error::{Result, VectorSearchError};
use vector_search::exact_knn::ExactKnn;
use vector_search::hardware::{detect_hardware, get_peak_rss_mb, HardwareInfo};
use vector_search::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use vector_search::matrix::RowMatrix;
use vector_search::recall::{
    compute_recall_at_k, percentile_sorted, verify_self_recall, LatencyStats,
};

const DEFAULT_EF_SEARCH_LIST: [usize; 12] = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareMode {
    Single,
    Multithread,
}

impl CompareMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Multithread => "multithread",
        }
    }

    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "single" => Ok(Self::Single),
            "multithread" => Ok(Self::Multithread),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid --compare-mode '{other}': expected 'single' or 'multithread'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetChoice {
    Sift1M,
    SiftSmall,
}

impl DatasetChoice {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sift1m" => Ok(Self::Sift1M),
            "siftsmall" => Ok(Self::SiftSmall),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid --dataset '{other}': expected 'sift1m' or 'siftsmall'"
            ))),
        }
    }

    fn to_preset(self) -> SiftPreset {
        match self {
            Self::Sift1M => SiftPreset::Sift1M,
            Self::SiftSmall => SiftPreset::SiftSmall,
        }
    }
}

#[derive(Debug, Clone)]
struct Phase7CliConfig {
    dataset: DatasetChoice,
    base_fvecs: Option<PathBuf>,
    query_fvecs: Option<PathBuf>,
    gt_path: Option<PathBuf>,
    data_dir: PathBuf,
    scale: Option<usize>,
    queries: Option<usize>,
    dim: usize,
    num_clusters: usize,
    k: usize,
    m: usize,
    ef_construction: usize,
    ef_search_list: Vec<usize>,
    seed: u64,
    keep_pruned_shrink: Option<bool>,
    config_id: Option<String>,
    build_reps: Option<usize>,
    query_reps: usize,
    compare_mode: CompareMode,
    threads: usize,
    affinity: Option<String>,
    synthetic: bool,
    offline: bool,
    no_gt_cache: bool,
    prepare_100k_gt: bool,
    json_out: Option<PathBuf>,
    profile_rss_isolated: bool,
    internal_isolated_child: bool,
}

impl Default for Phase7CliConfig {
    fn default() -> Self {
        Self {
            dataset: DatasetChoice::Sift1M,
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
    fn validate(&self) -> Result<()> {
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

    fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            ef_construction: self.ef_construction,
            keep_pruned: true,
            keep_pruned_shrink: self.keep_pruned_shrink,
            seed: self.seed,
            ..HnswConfig::default()
        }
    }

    fn resolve_config_id(&self) -> String {
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

fn print_usage() {
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

fn parse_next_str<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str> {
    if *idx + 1 >= args.len() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Missing argument for {flag}"
        )));
    }
    let val = args[*idx + 1].as_str();
    *idx += 2;
    Ok(val)
}

fn parse_next_usize(args: &[String], idx: &mut usize, flag: &str) -> Result<usize> {
    let raw = parse_next_str(args, idx, flag)?;
    raw.parse::<usize>()
        .map_err(|e| VectorSearchError::InvalidArgument(format!("Invalid {flag} '{raw}': {e}")))
}

fn parse_next_u64(args: &[String], idx: &mut usize, flag: &str) -> Result<u64> {
    let raw = parse_next_str(args, idx, flag)?;
    raw.parse::<u64>()
        .map_err(|e| VectorSearchError::InvalidArgument(format!("Invalid {flag} '{raw}': {e}")))
}

fn parse_ef_search_list(raw: &str) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v = trimmed.parse::<usize>().map_err(|e| {
            VectorSearchError::InvalidArgument(format!(
                "Invalid value '{trimmed}' in --ef-search-list: {e}"
            ))
        })?;
        values.push(v);
    }
    if values.is_empty() || values.contains(&0) {
        return Err(VectorSearchError::InvalidArgument(
            "--ef-search-list must contain at least one positive integer".to_string(),
        ));
    }
    Ok(values)
}

enum ParseOutcome {
    Help,
    Run(Box<Phase7CliConfig>),
}

fn parse_cli_args(args: &[String]) -> Result<ParseOutcome> {
    let mut cfg = Phase7CliConfig::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(ParseOutcome::Help),
            "--dataset" => {
                let val = parse_next_str(args, &mut i, "--dataset")?;
                cfg.dataset = DatasetChoice::parse(val)?;
            }
            "--base-fvecs" => {
                cfg.base_fvecs = Some(PathBuf::from(parse_next_str(args, &mut i, "--base-fvecs")?));
            }
            "--query-fvecs" => {
                cfg.query_fvecs = Some(PathBuf::from(parse_next_str(
                    args,
                    &mut i,
                    "--query-fvecs",
                )?));
            }
            "--gt-path" => {
                cfg.gt_path = Some(PathBuf::from(parse_next_str(args, &mut i, "--gt-path")?));
            }
            "--data-dir" => {
                cfg.data_dir = PathBuf::from(parse_next_str(args, &mut i, "--data-dir")?);
            }
            "--scale" => {
                cfg.scale = Some(parse_next_usize(args, &mut i, "--scale")?);
            }
            "--queries" => {
                cfg.queries = Some(parse_next_usize(args, &mut i, "--queries")?);
            }
            "--dim" => {
                cfg.dim = parse_next_usize(args, &mut i, "--dim")?;
            }
            "--num-clusters" => {
                cfg.num_clusters = parse_next_usize(args, &mut i, "--num-clusters")?;
            }
            "--k" => {
                cfg.k = parse_next_usize(args, &mut i, "--k")?;
            }
            "--m" => {
                cfg.m = parse_next_usize(args, &mut i, "--m")?;
            }
            "--ef-construction" => {
                cfg.ef_construction = parse_next_usize(args, &mut i, "--ef-construction")?;
            }
            "--ef-search-list" => {
                let raw = parse_next_str(args, &mut i, "--ef-search-list")?;
                cfg.ef_search_list = parse_ef_search_list(raw)?;
            }
            "--seed" => {
                cfg.seed = parse_next_u64(args, &mut i, "--seed")?;
            }
            "--no-keep-pruned-shrink" => {
                cfg.keep_pruned_shrink = Some(false);
                i += 1;
            }
            "--config-id" => {
                cfg.config_id = Some(parse_next_str(args, &mut i, "--config-id")?.to_string());
            }
            "--build-reps" => {
                cfg.build_reps = Some(parse_next_usize(args, &mut i, "--build-reps")?);
            }
            "--query-reps" => {
                cfg.query_reps = parse_next_usize(args, &mut i, "--query-reps")?;
            }
            "--compare-mode" => {
                let raw = parse_next_str(args, &mut i, "--compare-mode")?;
                cfg.compare_mode = CompareMode::parse(raw)?;
            }
            "--threads" => {
                cfg.threads = parse_next_usize(args, &mut i, "--threads")?;
            }
            "--affinity" => {
                cfg.affinity = Some(parse_next_str(args, &mut i, "--affinity")?.to_string());
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
                cfg.json_out = Some(PathBuf::from(parse_next_str(args, &mut i, "--json-out")?));
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

fn spawn_isolated_child(raw_args: &[String]) -> Result<i32> {
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

fn apply_affinity_pinning(
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
        "--affinity pinning inside phase7 is only supported on Windows; pin externally (e.g. taskset) and set VECTOR_SEARCH_AFFINITY_MASK".to_string(),
    ))
}

fn load_benchmark_dataset(cfg: &Phase7CliConfig) -> Result<(LoadedDataset, String)> {
    if cfg.synthetic {
        let num_base = cfg.scale.unwrap_or(100_000);
        let num_query = cfg.queries.unwrap_or(10_000);
        let t0 = Instant::now();
        let (base, query) =
            generate_synthetic_sift(num_base, num_query, cfg.dim, cfg.num_clusters, cfg.seed)?;
        let loaded = LoadedDataset {
            base,
            query,
            official_ground_truth: None,
            source: DatasetSource::Synthetic { seed: cfg.seed },
            load_time_s: t0.elapsed().as_secs_f64(),
        };
        return Ok((loaded, "synthetic".to_string()));
    }

    if let (Some(base_path), Some(query_path)) = (&cfg.base_fvecs, &cfg.query_fvecs) {
        let loaded =
            load_custom_dataset_with_options(base_path, query_path, None, cfg.scale, cfg.queries)?;
        if let Some(req_scale) = cfg.scale {
            if loaded.base.rows() < req_scale {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --scale {req_scale}, but {} contains only {} vectors",
                    base_path.display(),
                    loaded.base.rows()
                )));
            }
        }
        if let Some(req_q) = cfg.queries {
            if loaded.query.rows() < req_q {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --queries {req_q}, but {} contains only {} queries",
                    query_path.display(),
                    loaded.query.rows()
                )));
            }
        }
        return Ok((loaded, "custom".to_string()));
    }

    let preset = cfg.dataset.to_preset();
    let max_base = match cfg.scale {
        Some(s) => {
            if s > preset.expected_num_base() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --scale {s} exceeds {} base size ({})",
                    preset.as_str(),
                    preset.expected_num_base()
                )));
            }
            Some(s)
        }
        None => Some(match cfg.dataset {
            DatasetChoice::Sift1M => 100_000,
            DatasetChoice::SiftSmall => preset.expected_num_base(),
        }),
    };
    let max_queries = match cfg.queries {
        Some(q) => {
            if q > preset.expected_num_query() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Requested --queries {q} exceeds {} query size ({})",
                    preset.as_str(),
                    preset.expected_num_query()
                )));
            }
            Some(q)
        }
        None => Some(match cfg.dataset {
            DatasetChoice::Sift1M => 10_000,
            DatasetChoice::SiftSmall => preset.expected_num_query(),
        }),
    };

    let loaded = load_or_download_sift_with_options(
        preset,
        &cfg.data_dir,
        !cfg.offline,
        cfg.seed,
        max_base,
        max_queries,
        true,
    )?;

    if !loaded.source.is_authentic_sift() {
        let reason = match &loaded.source {
            DatasetSource::SyntheticFallback { reason, .. } => reason.as_str(),
            _ => "authentic SIFT dataset unavailable",
        };
        return Err(VectorSearchError::InvalidArgument(format!(
            "Authentic {} dataset unavailable ({reason}). Pass --synthetic explicitly to run on synthetic vectors.",
            preset.as_str()
        )));
    }

    Ok((loaded, preset.as_str().to_string()))
}

struct GroundTruthResolved {
    matrix: RowMatrix<i32>,
    source: String,
    path: Option<PathBuf>,
}

fn validate_and_slice_gt(
    src: &RowMatrix<i32>,
    num_query: usize,
    num_base: usize,
    k: usize,
) -> Result<RowMatrix<i32>> {
    if src.rows() < num_query || src.cols() < k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Ground truth matrix shape ({}x{}) incompatible with Q={num_query}, k={k}",
            src.rows(),
            src.cols()
        )));
    }
    let max_id = num_base as i32;
    let mut data = Vec::with_capacity(num_query * k);
    for q in 0..num_query {
        let row = &src.row(q)[..k];
        for &id in row {
            if id < 0 || id >= max_id {
                return Err(VectorSearchError::CorruptData(format!(
                    "Ground truth ID {id} out of range [0, {max_id}) at query {q}"
                )));
            }
        }
        data.extend_from_slice(row);
    }
    let sliced = RowMatrix::new(num_query, k, data)?;
    verify_self_recall(&sliced, k)?;
    Ok(sliced)
}

fn prepare_exact_gt_file(loaded: &LoadedDataset, cfg: &Phase7CliConfig) -> Result<PathBuf> {
    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    if num_base < cfg.k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "num_base ({num_base}) must be >= k ({})",
            cfg.k
        )));
    }
    let cache_k = 100.max(cfg.k).min(num_base);
    let target_path = cfg
        .gt_path
        .clone()
        .unwrap_or_else(|| cfg.data_dir.join(format!("gt_{num_base}_k{cache_k}.ivecs")));

    let t0 = Instant::now();
    let exact = ExactKnn::search_parallel_with_base(
        Metric::L2Squared,
        &loaded.base,
        &loaded.query,
        cache_k,
        0,
    )?;
    verify_self_recall(&exact.indices, cfg.k)?;
    write_ivecs(&target_path, &exact.indices)?;
    let elapsed_s = t0.elapsed().as_secs_f64();

    println!(
        "[Rust Phase 7] Wrote exact ground truth ({}x{}, N={}) to {} in {:.3}s",
        num_query,
        cache_k,
        num_base,
        target_path.display(),
        elapsed_s
    );
    Ok(target_path)
}

fn resolve_ground_truth(
    loaded: &LoadedDataset,
    cfg: &Phase7CliConfig,
) -> Result<GroundTruthResolved> {
    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();

    if let Some(ref gt_p) = cfg.gt_path {
        let raw_gt = read_ivecs_max_rows(gt_p, Some(num_query))?;
        let matrix = validate_and_slice_gt(&raw_gt, num_query, num_base, cfg.k)?;
        return Ok(GroundTruthResolved {
            matrix,
            source: format!("custom_ivecs ({})", gt_p.display()),
            path: Some(gt_p.clone()),
        });
    }

    match &loaded.source {
        DatasetSource::AuthenticSift1M { dir } if num_base == 1_000_000 => {
            if let Some(ref official) = loaded.official_ground_truth {
                let matrix = validate_and_slice_gt(official, num_query, num_base, cfg.k)?;
                return Ok(GroundTruthResolved {
                    matrix,
                    source: "official_texmex_sift1m_groundtruth_ivecs".to_string(),
                    path: Some(dir.join("sift_groundtruth.ivecs")),
                });
            }
        }
        DatasetSource::AuthenticSiftSmall { dir } if num_base == 10_000 => {
            if let Some(ref official) = loaded.official_ground_truth {
                let matrix = validate_and_slice_gt(official, num_query, num_base, cfg.k)?;
                return Ok(GroundTruthResolved {
                    matrix,
                    source: "official_texmex_siftsmall_groundtruth_ivecs".to_string(),
                    path: Some(dir.join("siftsmall_groundtruth.ivecs")),
                });
            }
        }
        _ => {}
    }

    if loaded.source.is_authentic_sift() {
        let cache_k = 100.max(cfg.k).min(num_base);
        let cache_path = cfg.data_dir.join(format!("gt_{num_base}_k{cache_k}.ivecs"));
        if !cfg.no_gt_cache && cache_path.is_file() {
            if let Ok(cached) = read_ivecs_max_rows(&cache_path, Some(num_query)) {
                if let Ok(matrix) = validate_and_slice_gt(&cached, num_query, num_base, cfg.k) {
                    return Ok(GroundTruthResolved {
                        matrix,
                        source: format!("cached_exact_knn_prefix_ivecs ({})", cache_path.display()),
                        path: Some(cache_path),
                    });
                }
            }
        }

        let exact = ExactKnn::search_parallel_with_base(
            Metric::L2Squared,
            &loaded.base,
            &loaded.query,
            cache_k,
            0,
        )?;
        let matrix = validate_and_slice_gt(&exact.indices, num_query, num_base, cfg.k)?;
        if !cfg.no_gt_cache {
            write_ivecs(&cache_path, &exact.indices)?;
            return Ok(GroundTruthResolved {
                matrix,
                source: format!(
                    "computed_exact_knn_prefix_ivecs (ExactKnn::search_parallel_with_base metric=l2 k={cache_k}, cached to {})",
                    cache_path.display()
                ),
                path: Some(cache_path),
            });
        }
        return Ok(GroundTruthResolved {
            matrix,
            source: format!(
                "computed_exact_knn_prefix (ExactKnn::search_parallel_with_base metric=l2 k={cache_k})"
            ),
            path: None,
        });
    }

    let exact = ExactKnn::search_parallel_with_base(
        Metric::L2Squared,
        &loaded.base,
        &loaded.query,
        cfg.k,
        0,
    )?;
    let matrix = validate_and_slice_gt(&exact.indices, num_query, num_base, cfg.k)?;
    Ok(GroundTruthResolved {
        matrix,
        source: format!(
            "computed_exact_knn (ExactKnn::search_parallel_with_base metric=l2 N={num_base}, Q={num_query}, k={})",
            cfg.k
        ),
        path: None,
    })
}

fn median_min_max(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        percentile_sorted(&sorted, 50.0),
        sorted[0],
        sorted[sorted.len() - 1],
    )
}

#[derive(Debug, Clone)]
struct Mode1Summary {
    num_queries: usize,
    total_s: f64,
    qps: f64,
    mean_ms: f64,
    mean_ms_min: f64,
    mean_ms_max: f64,
    mean_ms_spread: f64,
    qps_min: f64,
    qps_max: f64,
    qps_spread: f64,
    run_total_s: Vec<f64>,
    run_mean_ms: Vec<f64>,
    run_qps: Vec<f64>,
}

fn summarize_mode1_runs(num_queries: usize, totals_s: &[f64]) -> Mode1Summary {
    let run_mean_ms: Vec<f64> = totals_s
        .iter()
        .map(|&t| (t * 1000.0) / (num_queries.max(1) as f64))
        .collect();
    let run_qps: Vec<f64> = totals_s
        .iter()
        .map(|&t| {
            if t > 0.0 {
                (num_queries as f64) / t
            } else {
                0.0
            }
        })
        .collect();

    let (med_total_s, _, _) = median_min_max(totals_s);
    let (med_mean_ms, mean_ms_min, mean_ms_max) = median_min_max(&run_mean_ms);
    let (_, qps_min, qps_max) = median_min_max(&run_qps);
    let med_qps = if med_total_s > 0.0 {
        (num_queries as f64) / med_total_s
    } else {
        0.0
    };

    Mode1Summary {
        num_queries,
        total_s: med_total_s,
        qps: med_qps,
        mean_ms: med_mean_ms,
        mean_ms_min,
        mean_ms_max,
        mean_ms_spread: (mean_ms_max - mean_ms_min).max(0.0),
        qps_min,
        qps_max,
        qps_spread: (qps_max - qps_min).max(0.0),
        run_total_s: totals_s.to_vec(),
        run_mean_ms,
        run_qps,
    }
}

#[derive(Debug, Clone)]
struct Mode2Summary {
    sweep: RepeatedSweepSummary,
    run_total_s: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SweepPoint {
    ef_search: usize,
    recall_at_k: f64,
    recall_at_k_mode2: Option<f64>,
    mode1: Mode1Summary,
    mode2: Option<Mode2Summary>,
    ffi_delta_us: Option<f64>,
    instant_overhead_us: Option<f64>,
    dist_evals_mean: Option<f64>,
}

#[cfg(feature = "instrument")]
fn measure_diag_dist_evals(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    k: usize,
    ef_search: usize,
) -> Result<Option<f64>> {
    let mut diag_tracker = VisitedTracker::new(index.num_vectors());
    let _ = index.search_batch_timed_with_tracker(queries, k, ef_search, &mut diag_tracker)?;
    Ok(Some(
        (diag_tracker.distance_evals as f64) / (queries.rows().max(1) as f64),
    ))
}

#[cfg(not(feature = "instrument"))]
fn measure_diag_dist_evals(
    _index: &HnswIndex<'_>,
    _queries: &RowMatrix<f32>,
    _k: usize,
    _ef_search: usize,
) -> Result<Option<f64>> {
    Ok(None)
}

struct Phase7Report {
    compare_mode: CompareMode,
    config_id: String,
    generated_at: String,
    provenance: ReportProvenance,
    affinity_mask: String,
    affinity_pinned: bool,
    isolated_process: bool,
    hardware: HardwareInfo,
    preset_name: String,
    dataset_source: DatasetSource,
    num_base: usize,
    num_query: usize,
    dim: usize,
    k: usize,
    seed: u64,
    ground_truth_source: String,
    ground_truth_path: Option<PathBuf>,
    hnsw_config: HnswConfig,
    threads: usize,
    build_reps: usize,
    query_reps: usize,
    build_times_s: Vec<f64>,
    build_time_s: f64,
    build_time_min_s: f64,
    build_time_max_s: f64,
    build_time_spread_s: f64,
    build_qps: f64,
    max_layer: usize,
    entry_point: usize,
    graph_fingerprint: u64,
    peak_rss_mb: Option<f64>,
    graph_only_bytes_per_vector: f64,
    graph_only_memory_bytes: usize,
    graph_only_memory_mb: f64,
    bytes_per_vector: f64,
    layer0_bytes: usize,
    upper_layers_bytes: usize,
    node_structs_bytes: usize,
    raw_vector_bytes_per_vector: f64,
    total_with_vectors_bytes_per_vector: f64,
    monotonic_recall: bool,
    sweep: Vec<SweepPoint>,
}

fn push_f64_array(out: &mut String, vals: &[f64], decimals: usize) {
    out.push('[');
    for (idx, v) in vals.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{v:.decimals$}"));
    }
    out.push(']');
}

fn opt_f64_json(val: Option<f64>, decimals: usize) -> String {
    match val {
        Some(v) if v.is_finite() => format!("{v:.decimals$}"),
        _ => "null".to_string(),
    }
}

impl Phase7Report {
    fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(16_384);
        out.push_str("{\n");
        out.push_str("  \"schema_version\": 1,\n");
        out.push_str("  \"phase\": 7,\n");
        out.push_str("  \"phase_name\": \"Phase 7 Rust HnswIndex Benchmark\",\n");
        out.push_str("  \"engine\": \"rust\",\n");
        out.push_str(&format!(
            "  \"compare_mode\": \"{}\",\n",
            self.compare_mode.as_str()
        ));
        out.push_str(&format!(
            "  \"config_id\": \"{}\",\n",
            escape_json(&self.config_id)
        ));
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&self.generated_at)
        ));
        out.push_str(&format!(
            "  \"git_commit\": \"{}\",\n",
            escape_json(&self.provenance.git_commit)
        ));
        out.push_str(&format!(
            "  \"git_dirty\": {},\n",
            self.provenance.git_dirty
        ));
        out.push_str(&format!(
            "  \"rustc_version\": \"{}\",\n",
            escape_json(&self.provenance.rustc_version)
        ));
        out.push_str(&format!(
            "  \"isolated_process\": {},\n",
            self.isolated_process
        ));

        // environment
        out.push_str("  \"environment\": {\n");
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\",\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {},\n",
            opt_f64_json(self.hardware.total_ram_gb, 2)
        ));
        out.push_str(&format!(
            "    \"power_mode\": \"{}\",\n",
            escape_json(&self.provenance.power_mode)
        ));
        out.push_str(&format!(
            "    \"processor_affinity_mask\": \"{}\",\n",
            escape_json(&self.affinity_mask)
        ));
        out.push_str(&format!(
            "    \"affinity_pinned\": {},\n",
            self.affinity_pinned
        ));
        out.push_str(&format!("    \"threads\": {}\n", self.threads));
        out.push_str("  },\n");

        // hardware
        out.push_str("  \"hardware\": {\n");
        out.push_str(&format!(
            "    \"summary\": \"{}\",\n",
            escape_json(&self.hardware.summary_line())
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {},\n",
            opt_f64_json(self.hardware.total_ram_gb, 2)
        ));
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\"\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str("  },\n");

        // dataset
        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!(
            "    \"preset\": \"{}\",\n",
            escape_json(&self.preset_name)
        ));
        out.push_str(&format!(
            "    \"source_kind\": \"{}\",\n",
            self.dataset_source.kind_tag()
        ));
        out.push_str(&format!(
            "    \"source_description\": \"{}\",\n",
            escape_json(&self.dataset_source.to_string())
        ));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {},\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str(&format!("    \"scale_vectors\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str("    \"metric\": \"l2\"\n");
        out.push_str("  },\n");

        out.push_str(&format!(
            "  \"ground_truth_source\": \"{}\",\n",
            escape_json(&self.ground_truth_source)
        ));
        out.push_str(&format!(
            "  \"ground_truth_path\": {},\n",
            opt_path_json(self.ground_truth_path.as_deref())
        ));

        // hnsw_config
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_config.m));
        out.push_str(&format!(
            "    \"m_max\": {},\n",
            self.hnsw_config.effective_m_max()
        ));
        out.push_str(&format!(
            "    \"m_max0\": {},\n",
            self.hnsw_config.effective_m_max0()
        ));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_config.ef_construction
        ));
        out.push_str(&format!(
            "    \"keep_pruned\": {},\n",
            self.hnsw_config.keep_pruned
        ));
        match self.hnsw_config.keep_pruned_shrink {
            Some(kps) => out.push_str(&format!("    \"keep_pruned_shrink\": {kps},\n")),
            None => out.push_str("    \"keep_pruned_shrink\": null,\n"),
        }
        out.push_str(&format!("    \"seed\": {}\n", self.hnsw_config.seed));
        out.push_str("  },\n");

        out.push_str(&format!("  \"threads\": {},\n", self.threads));
        out.push_str("  \"warmup_performed\": true,\n");
        out.push_str(&format!("  \"build_reps\": {},\n", self.build_reps));
        out.push_str(&format!("  \"query_reps\": {},\n", self.query_reps));
        out.push_str(&format!("  \"scale_vectors\": {},\n", self.num_base));

        // index_build
        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!(
            "    \"build_time_min_s\": {:.6},\n",
            self.build_time_min_s
        ));
        out.push_str(&format!(
            "    \"build_time_max_s\": {:.6},\n",
            self.build_time_max_s
        ));
        out.push_str(&format!(
            "    \"build_time_spread_s\": {:.6},\n",
            self.build_time_spread_s
        ));
        out.push_str("    \"build_times_s\": ");
        push_f64_array(&mut out, &self.build_times_s, 6);
        out.push_str(",\n");
        out.push_str(&format!("    \"build_qps\": {:.2},\n", self.build_qps));
        out.push_str(&format!("    \"max_layer\": {},\n", self.max_layer));
        out.push_str(&format!("    \"entry_point\": {},\n", self.entry_point));
        out.push_str(&format!(
            "    \"graph_fingerprint\": {},\n",
            self.graph_fingerprint
        ));
        out.push_str(&format!(
            "    \"graph_fingerprint_hex\": \"0x{:016x}\"\n",
            self.graph_fingerprint
        ));
        out.push_str("  },\n");

        // memory_profile
        let peak_scope = if self.isolated_process {
            "isolated_child_process_high_water_mark"
        } else {
            "process_lifetime_high_water_mark"
        };
        out.push_str("  \"memory_profile\": {\n");
        out.push_str(&format!(
            "    \"peak_rss_mb\": {},\n",
            opt_f64_json(self.peak_rss_mb, 2)
        ));
        out.push_str(&format!("    \"peak_rss_scope\": \"{peak_scope}\",\n"));
        out.push_str(&format!(
            "    \"graph_only_bytes_per_vector\": {:.2},\n",
            self.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_bytes\": {},\n",
            self.graph_only_memory_bytes
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_mb\": {:.4},\n",
            self.graph_only_memory_mb
        ));
        out.push_str(&format!(
            "    \"bytes_per_vector\": {:.2},\n",
            self.bytes_per_vector
        ));
        out.push_str(&format!("    \"layer0_bytes\": {},\n", self.layer0_bytes));
        out.push_str(&format!(
            "    \"upper_layers_bytes\": {},\n",
            self.upper_layers_bytes
        ));
        out.push_str(&format!(
            "    \"node_structs_bytes\": {},\n",
            self.node_structs_bytes
        ));
        out.push_str(&format!(
            "    \"raw_vector_bytes_per_vector\": {:.2},\n",
            self.raw_vector_bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"total_with_vectors_bytes_per_vector\": {:.2}\n",
            self.total_with_vectors_bytes_per_vector
        ));
        out.push_str("  },\n");

        out.push_str(&format!(
            "  \"graph_only_bytes_per_vector\": {:.2},\n",
            self.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "  \"monotonic_recall\": {},\n",
            self.monotonic_recall
        ));

        // ef_search_sweep
        out.push_str("  \"ef_search_sweep\": [\n");
        for (idx, pt) in self.sweep.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", pt.ef_search));
            out.push_str(&format!("      \"recall_at_k\": {:.6},\n", pt.recall_at_k));
            out.push_str(&format!(
                "      \"recall_at_k_mode2\": {},\n",
                opt_f64_json(pt.recall_at_k_mode2, 6)
            ));

            // mode1_batch
            out.push_str("      \"mode1_batch\": {\n");
            out.push_str(&format!(
                "        \"num_queries\": {},\n",
                pt.mode1.num_queries
            ));
            out.push_str(&format!("        \"total_s\": {:.6},\n", pt.mode1.total_s));
            out.push_str(&format!("        \"qps\": {:.2},\n", pt.mode1.qps));
            out.push_str(&format!("        \"mean_ms\": {:.4},\n", pt.mode1.mean_ms));
            out.push_str(&format!(
                "        \"mean_ms_min\": {:.4},\n",
                pt.mode1.mean_ms_min
            ));
            out.push_str(&format!(
                "        \"mean_ms_max\": {:.4},\n",
                pt.mode1.mean_ms_max
            ));
            out.push_str(&format!(
                "        \"mean_ms_spread\": {:.4},\n",
                pt.mode1.mean_ms_spread
            ));
            out.push_str(&format!("        \"qps_min\": {:.2},\n", pt.mode1.qps_min));
            out.push_str(&format!("        \"qps_max\": {:.2},\n", pt.mode1.qps_max));
            out.push_str(&format!(
                "        \"qps_spread\": {:.2},\n",
                pt.mode1.qps_spread
            ));
            out.push_str("        \"run_total_s\": ");
            push_f64_array(&mut out, &pt.mode1.run_total_s, 6);
            out.push_str(",\n");
            out.push_str("        \"run_mean_ms\": ");
            push_f64_array(&mut out, &pt.mode1.run_mean_ms, 4);
            out.push_str(",\n");
            out.push_str("        \"run_qps\": ");
            push_f64_array(&mut out, &pt.mode1.run_qps, 2);
            out.push_str("\n      },\n");

            // mode2_per_query
            match &pt.mode2 {
                Some(m2) => {
                    let med = &m2.sweep.median_latency;
                    out.push_str("      \"mode2_per_query\": {\n");
                    out.push_str(&format!("        \"num_queries\": {},\n", med.num_queries));
                    out.push_str(&format!("        \"total_s\": {:.6},\n", med.total_s));
                    out.push_str(&format!("        \"qps\": {:.2},\n", med.qps));
                    out.push_str(&format!("        \"min_ms\": {:.4},\n", med.min_ms));
                    out.push_str(&format!("        \"mean_ms\": {:.4},\n", med.mean_ms));
                    out.push_str(&format!("        \"p50_ms\": {:.4},\n", med.p50_ms));
                    out.push_str(&format!("        \"p95_ms\": {:.4},\n", med.p95_ms));
                    out.push_str(&format!("        \"p99_ms\": {:.4},\n", med.p99_ms));
                    out.push_str(&format!("        \"max_ms\": {:.4},\n", med.max_ms));
                    out.push_str(&format!(
                        "        \"mean_ms_min\": {:.4},\n",
                        m2.sweep.mean_ms_min
                    ));
                    out.push_str(&format!(
                        "        \"mean_ms_max\": {:.4},\n",
                        m2.sweep.mean_ms_max
                    ));
                    out.push_str(&format!(
                        "        \"mean_ms_spread\": {:.4},\n",
                        m2.sweep.mean_ms_spread
                    ));
                    out.push_str(&format!(
                        "        \"p99_ms_min\": {:.4},\n",
                        m2.sweep.p99_ms_min
                    ));
                    out.push_str(&format!(
                        "        \"p99_ms_max\": {:.4},\n",
                        m2.sweep.p99_ms_max
                    ));
                    out.push_str(&format!(
                        "        \"p99_ms_spread\": {:.4},\n",
                        m2.sweep.p99_ms_spread
                    ));
                    out.push_str("        \"run_total_s\": ");
                    push_f64_array(&mut out, &m2.run_total_s, 6);
                    out.push_str(",\n");
                    out.push_str("        \"run_mean_ms\": ");
                    push_f64_array(&mut out, &m2.sweep.run_mean_latencies_ms, 4);
                    out.push_str(",\n");
                    out.push_str("        \"run_p99_ms\": ");
                    push_f64_array(&mut out, &m2.sweep.run_p99_latencies_ms, 4);
                    out.push_str("\n      },\n");
                }
                None => {
                    out.push_str("      \"mode2_per_query\": null,\n");
                }
            }

            out.push_str(&format!(
                "      \"ffi_delta_us\": {},\n",
                opt_f64_json(pt.ffi_delta_us, 4)
            ));
            out.push_str(&format!(
                "      \"instant_overhead_us\": {},\n",
                opt_f64_json(pt.instant_overhead_us, 4)
            ));
            out.push_str(&format!(
                "      \"dist_evals_mean\": {},\n",
                opt_f64_json(pt.dist_evals_mean, 4)
            ));
            out.push_str(&format!("      \"mode1_qps\": {:.2},\n", pt.mode1.qps));
            out.push_str(&format!(
                "      \"mode1_mean_ms\": {:.4},\n",
                pt.mode1.mean_ms
            ));
            out.push_str(&format!("      \"qps\": {:.2},\n", pt.mode1.qps));

            let (mean_alias, p50_alias, p95_alias, p99_alias) = match &pt.mode2 {
                Some(m2) => (
                    m2.sweep.median_latency.mean_ms,
                    Some(m2.sweep.median_latency.p50_ms),
                    Some(m2.sweep.median_latency.p95_ms),
                    Some(m2.sweep.median_latency.p99_ms),
                ),
                None => (pt.mode1.mean_ms, None, None, None),
            };
            out.push_str(&format!("      \"mean_ms\": {:.4},\n", mean_alias));
            out.push_str(&format!(
                "      \"p50_ms\": {},\n",
                opt_f64_json(p50_alias, 4)
            ));
            out.push_str(&format!(
                "      \"p95_ms\": {},\n",
                opt_f64_json(p95_alias, 4)
            ));
            out.push_str(&format!(
                "      \"p99_ms\": {}\n",
                opt_f64_json(p99_alias, 4)
            ));

            if idx + 1 < self.sweep.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }
}

fn print_console_summary(report: &Phase7Report) {
    println!(
        "[Rust Phase 7] {} (M={}, efC={}, keep_pruned_shrink={:?}, mode={}, T={}, affinity={})",
        report.config_id,
        report.hnsw_config.m,
        report.hnsw_config.ef_construction,
        report.hnsw_config.keep_pruned_shrink,
        report.compare_mode.as_str(),
        report.threads,
        report.affinity_mask
    );
    println!(
        "  Hardware: {} | Power: {}",
        report.hardware.summary_line(),
        report.provenance.power_mode
    );
    println!(
        "  Dataset: {} N={} Q={} dim={} k={} | GT: {}",
        report.dataset_source.kind_tag(),
        report.num_base,
        report.num_query,
        report.dim,
        report.k,
        report.ground_truth_source
    );
    let rss_display = report
        .peak_rss_mb
        .map_or_else(|| "n/a".to_string(), |v| format!("{v:.2} MiB"));
    println!(
        "  Build: median={:.3}s ({:.1} vec/s, reps={}) | fp=0x{:016x} | Graph={:.2} B/vec | PeakRSS={}",
        report.build_time_s,
        report.build_qps,
        report.build_reps,
        report.graph_fingerprint,
        report.graph_only_bytes_per_vector,
        rss_display
    );
    println!("  efSearch | Recall@k | Mode1 QPS | Mode1 ms | Mode2 p99 | Instant delta (us)");
    for pt in &report.sweep {
        let p99_str = pt.mode2.as_ref().map_or_else(
            || "      n/a".to_string(),
            |m2| format!("{:9.4}", m2.sweep.median_latency.p99_ms),
        );
        let delta_str = pt
            .instant_overhead_us
            .map_or_else(|| "               n/a".to_string(), |d| format!("{d:18.3}"));
        println!(
            "  {:8} | {:8.6} | {:9.2} | {:8.4} | {} | {}",
            pt.ef_search, pt.recall_at_k, pt.mode1.qps, pt.mode1.mean_ms, p99_str, delta_str
        );
    }
}

fn run_benchmark(cfg: &Phase7CliConfig) -> Result<Phase7Report> {
    let provenance = detect_provenance();
    let (affinity_mask, affinity_pinned) =
        apply_affinity_pinning(cfg.affinity.as_deref(), &provenance)?;
    let hardware = detect_hardware();

    let (loaded, preset_name) = load_benchmark_dataset(cfg)?;
    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();
    if num_base < cfg.k {
        return Err(VectorSearchError::InvalidArgument(format!(
            "num_base ({num_base}) must be >= k ({})",
            cfg.k
        )));
    }

    let gt = resolve_ground_truth(&loaded, cfg)?;
    let hnsw_cfg = cfg.to_hnsw_config();
    let build_reps = cfg
        .build_reps
        .unwrap_or(if num_base >= 500_000 { 1 } else { 3 });

    let mut build_times_s = Vec::with_capacity(build_reps);
    let mut built_index: Option<HnswIndex<'_>> = None;
    for _ in 0..build_reps {
        drop(built_index.take());
        let t0 = Instant::now();
        let candidate = HnswIndex::build(&loaded.base, Metric::L2Squared, hnsw_cfg.clone())?;
        build_times_s.push(t0.elapsed().as_secs_f64());
        built_index = Some(candidate);
    }
    let index = built_index.expect("build_reps >= 1 guarantees built index");
    let (build_time_s, build_time_min_s, build_time_max_s) = median_min_max(&build_times_s);
    let build_time_spread_s = (build_time_max_s - build_time_min_s).max(0.0);
    let build_qps = if build_time_s > 0.0 {
        (num_base as f64) / build_time_s
    } else {
        0.0
    };

    let mode1_threads = match cfg.compare_mode {
        CompareMode::Single => 1,
        CompareMode::Multithread => cfg.threads,
    };

    let mut tracker = VisitedTracker::new(num_base);
    let mut prev_recall = -1.0_f64;
    let mut monotonic_recall = true;
    let mut sweep = Vec::with_capacity(cfg.ef_search_list.len());

    for &ef_search in &cfg.ef_search_list {
        // Mode 1: Pure Batch Throughput (1 untimed warmup + query_reps timed passes)
        let warmup_m1 =
            index.search_batch_parallel(&loaded.query, cfg.k, ef_search, mode1_threads)?;
        std::hint::black_box(&warmup_m1.indices);

        let mut m1_totals_s = Vec::with_capacity(cfg.query_reps);
        let mut last_m1_indices: Option<RowMatrix<i32>> = None;
        for rep in 0..cfg.query_reps {
            let t0 = Instant::now();
            let res =
                index.search_batch_parallel(&loaded.query, cfg.k, ef_search, mode1_threads)?;
            let elapsed_s = t0.elapsed().as_secs_f64();
            std::hint::black_box(&res.indices);
            m1_totals_s.push(elapsed_s);
            if rep + 1 == cfg.query_reps {
                last_m1_indices = Some(res.indices);
            }
        }
        let m1_summary = summarize_mode1_runs(num_query, &m1_totals_s);
        let m1_indices = last_m1_indices.expect("query_reps >= 1");
        let recall_at_k = compute_recall_at_k(&m1_indices, &gt.matrix, cfg.k)?;

        if prev_recall >= 0.0 && recall_at_k + 1e-12 < prev_recall {
            monotonic_recall = false;
        }
        prev_recall = recall_at_k;

        // Mode 2: Per-Query Distribution (only in single mode)
        let (mode2, recall_at_k_mode2, ffi_delta_us, instant_overhead_us) = if cfg.compare_mode
            == CompareMode::Single
        {
            let warmup_m2 = index.search_batch_timed_with_tracker(
                &loaded.query,
                cfg.k,
                ef_search,
                &mut tracker,
            )?;
            std::hint::black_box(&warmup_m2.result.indices);

            let mut m2_runs: Vec<LatencyStats> = Vec::with_capacity(cfg.query_reps);
            let mut m2_totals: Vec<f64> = Vec::with_capacity(cfg.query_reps);
            let mut last_m2_indices: Option<RowMatrix<i32>> = None;
            for rep in 0..cfg.query_reps {
                let timed = index.search_batch_timed_with_tracker(
                    &loaded.query,
                    cfg.k,
                    ef_search,
                    &mut tracker,
                )?;
                std::hint::black_box(&timed.result.indices);
                m2_totals.push(timed.latency.total_s);
                m2_runs.push(timed.latency);
                if rep + 1 == cfg.query_reps {
                    last_m2_indices = Some(timed.result.indices);
                }
            }
            let m2_sweep = summarize_repeated_latencies(&m2_runs);
            let m2_indices = last_m2_indices.expect("query_reps >= 1");
            let r2 = compute_recall_at_k(&m2_indices, &gt.matrix, cfg.k)?;
            if m1_indices.as_slice() != m2_indices.as_slice() || (recall_at_k - r2).abs() > 1e-12 {
                return Err(VectorSearchError::VerificationFailed(format!(
                        "Mode 1 vs Mode 2 recall parity mismatch at ef_search={ef_search}: {recall_at_k} != {r2}"
                    )));
            }
            let overhead_us = (m2_sweep.median_latency.mean_ms - m1_summary.mean_ms) * 1000.0;
            (
                Some(Mode2Summary {
                    sweep: m2_sweep,
                    run_total_s: m2_totals,
                }),
                Some(r2),
                Some(0.0_f64),
                Some(overhead_us),
            )
        } else {
            (None, None, None, None)
        };

        let dist_evals_mean = measure_diag_dist_evals(&index, &loaded.query, cfg.k, ef_search)?;

        sweep.push(SweepPoint {
            ef_search,
            recall_at_k,
            recall_at_k_mode2,
            mode1: m1_summary,
            mode2,
            ffi_delta_us,
            instant_overhead_us,
            dist_evals_mean,
        });
    }

    let peak_rss_mb = get_peak_rss_mb();
    let components = index.graph_components();
    let graph_only_bytes_per_vector = index.graph_only_bytes_per_vector();
    let graph_only_memory_bytes = index.graph_only_memory_bytes();
    let graph_only_memory_mb = index.graph_only_memory_mb();
    let bytes_per_vector = index.bytes_per_vector();
    let raw_vector_bytes_per_vector = (dim * std::mem::size_of::<f32>()) as f64;
    let total_with_vectors_bytes_per_vector =
        graph_only_bytes_per_vector + raw_vector_bytes_per_vector;

    Ok(Phase7Report {
        compare_mode: cfg.compare_mode,
        config_id: cfg.resolve_config_id(),
        generated_at: current_utc_rfc3339(),
        provenance,
        affinity_mask,
        affinity_pinned,
        isolated_process: cfg.internal_isolated_child,
        hardware,
        preset_name,
        dataset_source: loaded.source,
        num_base,
        num_query,
        dim,
        k: cfg.k,
        seed: cfg.seed,
        ground_truth_source: gt.source,
        ground_truth_path: gt.path,
        hnsw_config: hnsw_cfg,
        threads: cfg.threads,
        build_reps,
        query_reps: cfg.query_reps,
        build_times_s,
        build_time_s,
        build_time_min_s,
        build_time_max_s,
        build_time_spread_s,
        build_qps,
        max_layer: index.max_layer(),
        entry_point: index.entry_point().unwrap_or(0),
        graph_fingerprint: index.graph_fingerprint(),
        peak_rss_mb,
        graph_only_bytes_per_vector,
        graph_only_memory_bytes,
        graph_only_memory_mb,
        bytes_per_vector,
        layer0_bytes: components.layer0_edges_bytes,
        upper_layers_bytes: components.upper_layers_edges_bytes,
        node_structs_bytes: components.node_structs_bytes,
        raw_vector_bytes_per_vector,
        total_with_vectors_bytes_per_vector,
        monotonic_recall,
        sweep,
    })
}

fn write_json_report(path: &Path, json: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, json)?;
    Ok(())
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let cfg = match parse_cli_args(&raw_args) {
        Ok(ParseOutcome::Help) => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Ok(ParseOutcome::Run(cfg)) => cfg,
        Err(err) => {
            eprintln!("[ERROR] {err}");
            return ExitCode::FAILURE;
        }
    };

    if cfg.profile_rss_isolated && !cfg.internal_isolated_child {
        return match spawn_isolated_child(&raw_args) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(code) => {
                eprintln!("[ERROR] Isolated child process exited with code {code}");
                ExitCode::FAILURE
            }
            Err(err) => {
                eprintln!("[ERROR] Failed to spawn isolated child process: {err}");
                ExitCode::FAILURE
            }
        };
    }

    if cfg.prepare_100k_gt {
        return match load_benchmark_dataset(&cfg).and_then(|(loaded, _)| {
            prepare_exact_gt_file(&loaded, &cfg)?;
            Ok(())
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("[ERROR] {err}");
                ExitCode::FAILURE
            }
        };
    }

    match run_benchmark(&cfg) {
        Ok(report) => {
            if let Some(ref json_path) = cfg.json_out {
                let json_str = report.to_json_string();
                if let Err(err) = write_json_report(json_path, &json_str) {
                    eprintln!("[ERROR] Failed to write JSON report: {err}");
                    return ExitCode::FAILURE;
                }
                println!("[Rust Phase 7] Wrote JSON report: {}", json_path.display());
            }
            print_console_summary(&report);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ERROR] {err}");
            ExitCode::FAILURE
        }
    }
}
