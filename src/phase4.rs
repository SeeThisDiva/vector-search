//! Phase 4 Parameter Sweep Runner: Resumable HNSW Grid Exploration & Pareto Frontier.
//!
//! Provides:
//! - Declarative grid specification parsing ([`Phase4GridSpec`], [`Phase4IndexConfig`])
//! - Prefix ground-truth disk caching ([`load_or_compute_phase4_ground_truth`])
//! - Resumable run semantics skipping previously measured configs ([`can_resume_existing_report`])
//! - End-to-end multi-configuration sweep execution ([`run_phase4_sweep`])
//! - Dedicated CLI argument parsing and execution ([`run_phase4_cli`])

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::benchmark::{
    detect_provenance, summarize_repeated_latencies, Phase1DatasetMode, RepeatedSweepSummary,
    ReportProvenance,
};
use crate::dataset::{
    generate_synthetic_sift, load_custom_dataset_with_options, load_or_download_sift_with_options,
    read_ivecs, write_ivecs, DatasetSource, LoadedDataset, SiftPreset,
};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{detect_hardware, HardwareInfo};
use crate::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use crate::matrix::RowMatrix;
use crate::recall::compute_recall_at_k;

/// Individual HNSW index construction parameters in a Phase 4 sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase4IndexConfig {
    pub m: usize,
    pub m_max: Option<usize>,
    pub m_max0: Option<usize>,
    pub ef_construction: usize,
    pub keep_pruned: bool,
    pub keep_pruned_shrink: Option<bool>,
    pub seed: u64,
}

impl Default for Phase4IndexConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: None,
            m_max0: None,
            ef_construction: 100,
            keep_pruned: true,
            keep_pruned_shrink: None,
            seed: 42,
        }
    }
}

impl Phase4IndexConfig {
    /// Constructs a configuration with target degree `m` and beam `ef_construction`.
    #[must_use]
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            ef_construction,
            ..Self::default()
        }
    }

    /// Converts this grid configuration into an internal [`HnswConfig`].
    #[must_use]
    pub fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            m_max: self.m_max,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            keep_pruned: self.keep_pruned,
            keep_pruned_shrink: self.keep_pruned_shrink,
            seed: self.seed,
            ..HnswConfig::default()
        }
    }

    /// Generates the deterministic per-index report filename for a given dataset scale.
    #[must_use]
    pub fn report_filename(&self, scale_vectors: usize) -> String {
        format!(
            "scale_{}_m{}_efc{}{}.json",
            scale_vectors,
            self.m,
            self.ef_construction,
            if self.keep_pruned_shrink == Some(false) {
                "_no_kp_shrink"
            } else {
                ""
            }
        )
    }
}

/// Declarative grid specification for Phase 4 parameter sweeps.
#[derive(Debug, Clone, PartialEq)]
pub struct Phase4GridSpec {
    pub scale_vectors: Option<usize>,
    pub k: usize,
    pub sweep_runs: usize,
    pub seed: u64,
    pub ef_search_list: Vec<usize>,
    pub configs: Vec<Phase4IndexConfig>,
}

impl Default for Phase4GridSpec {
    fn default() -> Self {
        Self {
            scale_vectors: None,
            k: 10,
            sweep_runs: 3,
            seed: 42,
            ef_search_list: vec![10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600],
            configs: Vec::new(),
        }
    }
}

impl Phase4GridSpec {
    /// Parses a grid specification from a file on disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::parse_str(&content)
    }

    /// Parses a grid specification from a directive string.
    pub fn parse_str(input: &str) -> Result<Self> {
        let mut scale_vectors: Option<usize> = None;
        let mut k: usize = 10;
        let mut sweep_runs: usize = 3;
        let mut seed: u64 = 42;
        let mut ef_search_list: Option<Vec<usize>> = None;
        let mut m_list: Vec<usize> = Vec::new();
        let mut ef_construction_list: Vec<usize> = Vec::new();
        let mut explicit_configs: Vec<Phase4IndexConfig> = Vec::new();

        for (line_idx, raw_line) in input.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            // Strip any trailing comments (e.g., # or //)
            let line_clean = if let Some(idx) = trimmed.find('#') {
                trimmed[..idx].trim()
            } else if let Some(idx) = trimmed.find("//") {
                trimmed[..idx].trim()
            } else {
                trimmed
            };
            if line_clean.is_empty() {
                continue;
            }

            // Split into key and value by '=' or ':'
            let (key, val) = if let Some(pos) = line_clean.find('=') {
                (line_clean[..pos].trim(), line_clean[pos + 1..].trim())
            } else if let Some(pos) = line_clean.find(':') {
                (line_clean[..pos].trim(), line_clean[pos + 1..].trim())
            } else {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Line {line_num}: invalid syntax (expected key = value): '{line_clean}'"
                )));
            };

            let key_norm = key.to_ascii_lowercase();
            match key_norm.as_str() {
                "scale_vectors" | "scale" | "max_base" => {
                    let v: usize = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid scale_vectors '{val}': {e}"
                        ))
                    })?;
                    scale_vectors = Some(v);
                }
                "k" => {
                    k = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid k '{val}': {e}"
                        ))
                    })?;
                }
                "sweep_runs" | "repetitions" | "runs" => {
                    sweep_runs = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid sweep_runs '{val}': {e}"
                        ))
                    })?;
                }
                "seed" => {
                    seed = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid seed '{val}': {e}"
                        ))
                    })?;
                }
                "ef_search" | "ef_search_list" => {
                    let list = parse_comma_separated_usizes(val, line_num, "ef_search")?;
                    ef_search_list = Some(list);
                }
                "m" | "m_list" => {
                    let list = parse_comma_separated_usizes(val, line_num, "m")?;
                    m_list = list;
                }
                "ef_construction" | "ef_construction_list" => {
                    let list = parse_comma_separated_usizes(val, line_num, "ef_construction")?;
                    ef_construction_list = list;
                }
                "config" => {
                    let cfg = parse_explicit_config_line(val, line_num, seed)?;
                    explicit_configs.push(cfg);
                }
                unknown => {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: unrecognized grid directive '{unknown}'"
                    )));
                }
            }
        }

        let mut configs = Vec::new();
        // Cartesian expansion of (m, ef_construction) subject to ef_construction >= m
        if !m_list.is_empty() && !ef_construction_list.is_empty() {
            for &m_val in &m_list {
                for &efc_val in &ef_construction_list {
                    if efc_val >= m_val {
                        configs.push(Phase4IndexConfig {
                            m: m_val,
                            m_max: None,
                            m_max0: None,
                            ef_construction: efc_val,
                            keep_pruned: true,
                            keep_pruned_shrink: None,
                            seed,
                        });
                    }
                }
            }
        }

        // Add explicit configs
        configs.extend(explicit_configs);

        if configs.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "Grid specification resulted in 0 configurations".to_string(),
            ));
        }

        if k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "k must be >= 1".to_string(),
            ));
        }
        if sweep_runs == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "sweep_runs must be >= 1".to_string(),
            ));
        }

        let final_ef_search_list = ef_search_list
            .unwrap_or_else(|| vec![10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600]);
        if final_ef_search_list.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "ef_search_list cannot be empty".to_string(),
            ));
        }
        for &ef in &final_ef_search_list {
            if ef == 0 {
                return Err(VectorSearchError::InvalidArgument(
                    "ef_search values must be >= 1".to_string(),
                ));
            }
        }

        // Validate each configuration
        for cfg in &configs {
            cfg.to_hnsw_config().validate()?;
        }

        Ok(Self {
            scale_vectors,
            k,
            sweep_runs,
            seed,
            ef_search_list: final_ef_search_list,
            configs,
        })
    }
}

fn parse_comma_separated_usizes(
    val: &str,
    line_num: usize,
    field_name: &str,
) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in val.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let v: usize = p.parse().map_err(|e| {
            VectorSearchError::InvalidArgument(format!(
                "Line {line_num}: invalid {field_name} value '{p}': {e}"
            ))
        })?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: {field_name} list cannot be empty"
        )));
    }
    Ok(out)
}

fn parse_explicit_config_line(
    val: &str,
    line_num: usize,
    default_seed: u64,
) -> Result<Phase4IndexConfig> {
    let mut m: Option<usize> = None;
    let mut m_max: Option<usize> = None;
    let mut m_max0: Option<usize> = None;
    let mut ef_construction: Option<usize> = None;
    let mut keep_pruned = true;
    let mut keep_pruned_shrink: Option<bool> = None;
    let mut seed = default_seed;

    for part in val.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let (k, v) = if let Some(pos) = p.find(':') {
            (p[..pos].trim(), p[pos + 1..].trim())
        } else if let Some(pos) = p.find('=') {
            (p[..pos].trim(), p[pos + 1..].trim())
        } else {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Line {line_num}: invalid config field '{p}' (expected key:value)"
            )));
        };

        match k.to_ascii_lowercase().as_str() {
            "m" => {
                m = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid m '{v}': {e}"
                    ))
                })?);
            }
            "m_max" => {
                m_max = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid m_max '{v}': {e}"
                    ))
                })?);
            }
            "m_max0" => {
                m_max0 = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid m_max0 '{v}': {e}"
                    ))
                })?);
            }
            "ef_construction" | "efc" => {
                ef_construction = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid ef_construction '{v}': {e}"
                    ))
                })?);
            }
            "keep_pruned" => {
                keep_pruned = parse_bool(v, line_num, "keep_pruned")?;
            }
            "keep_pruned_shrink" => {
                keep_pruned_shrink = Some(parse_bool(v, line_num, "keep_pruned_shrink")?);
            }
            "seed" => {
                seed = v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid seed '{v}': {e}"
                    ))
                })?;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Line {line_num}: unrecognized config field '{unknown}'"
                )));
            }
        }
    }

    let m = m.ok_or_else(|| {
        VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: explicit config missing required field 'm'"
        ))
    })?;
    let ef_construction = ef_construction.ok_or_else(|| {
        VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: explicit config missing required field 'ef_construction'"
        ))
    })?;

    Ok(Phase4IndexConfig {
        m,
        m_max,
        m_max0,
        ef_construction,
        keep_pruned,
        keep_pruned_shrink,
        seed,
    })
}

fn parse_bool(v: &str, line_num: usize, name: &str) -> Result<bool> {
    match v.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: invalid boolean for {name}: '{v}'"
        ))),
    }
}

/// Slices the first `k` columns and first `rows` queries from a ground truth matrix.
fn slice_ground_truth_matrix(
    src: &RowMatrix<i32>,
    rows: usize,
    cols: usize,
) -> Result<RowMatrix<i32>> {
    if src.rows() == rows && src.cols() == cols {
        return Ok(src.clone());
    }
    if src.rows() < rows || src.cols() < cols {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Source ground truth matrix ({}x{}) smaller than requested ({}x{})",
            src.rows(),
            src.cols(),
            rows,
            cols
        )));
    }
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        data.extend_from_slice(&src.row(r)[..cols]);
    }
    RowMatrix::new(rows, cols, data)
}

/// Result of loading or computing Phase 4 ground truth.
#[derive(Debug, Clone)]
pub struct Phase4GroundTruth {
    pub matrix: RowMatrix<i32>,
    pub ground_truth_source: String,
    pub ground_truth_cache_path: Option<PathBuf>,
    pub ground_truth_cache_hit: bool,
}

/// Loads official ground truth (if 1M authentic SIFT) or loads/computes exact prefix ground truth.
pub fn load_or_compute_phase4_ground_truth(
    loaded: &LoadedDataset,
    k: usize,
    metric: Metric,
    data_cache_dir: &Path,
    enable_disk_cache: bool,
) -> Result<Phase4GroundTruth> {
    let num_base = loaded.base.rows();
    let num_queries = loaded.query.rows();
    if num_base == 0 || num_queries == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Base or query matrix cannot be empty".to_string(),
        ));
    }

    // Case A: Official SIFT1M ground truth (authentic SIFT corpus, exactly 1M base vectors)
    if loaded.source.is_authentic_sift() && num_base == 1_000_000 {
        if let Some(ref official) = loaded.official_ground_truth {
            if official.rows() >= num_queries && official.cols() >= k {
                let matrix = slice_ground_truth_matrix(official, num_queries, k)?;
                return Ok(Phase4GroundTruth {
                    matrix,
                    ground_truth_source: "official_texmex_sift1m_groundtruth_ivecs".to_string(),
                    ground_truth_cache_path: None,
                    ground_truth_cache_hit: true,
                });
            }
        }
    }

    // Case B: Prefix scale or synthetic/custom dataset
    let cache_filename = format!("gt_{num_base}_k100.ivecs");
    let cache_path = data_cache_dir.join(cache_filename);
    let cache_k = 100.max(k).min(num_base);

    if enable_disk_cache && cache_path.is_file() {
        if let Ok(cached) = read_ivecs(&cache_path) {
            if cached.rows() == num_queries && cached.cols() >= k {
                let matrix = slice_ground_truth_matrix(&cached, num_queries, k)?;
                let ground_truth_source = format!(
                    "cached_exact_knn_prefix_ivecs ({}, computed by ExactKnn::search_parallel_with_base metric={} k={})",
                    cache_path.display(),
                    metric,
                    cached.cols()
                );
                return Ok(Phase4GroundTruth {
                    matrix,
                    ground_truth_source,
                    ground_truth_cache_path: Some(cache_path),
                    ground_truth_cache_hit: true,
                });
            }
        }
    }

    // Compute exact brute-force ground truth on the prefix
    let exact =
        ExactKnn::search_parallel_with_base(metric, &loaded.base, &loaded.query, cache_k, 0)?;

    if enable_disk_cache {
        if !data_cache_dir.exists() {
            fs::create_dir_all(data_cache_dir)?;
        }
        write_ivecs(&cache_path, &exact.indices)?;
    }

    let matrix = slice_ground_truth_matrix(&exact.indices, num_queries, k)?;
    let ground_truth_source = format!(
        "computed_exact_knn_prefix_ivecs (ExactKnn::search_parallel_with_base metric={} k={}{})",
        metric,
        cache_k,
        if enable_disk_cache {
            format!(", cached to {}", cache_path.display())
        } else {
            String::new()
        }
    );

    Ok(Phase4GroundTruth {
        matrix,
        ground_truth_source,
        ground_truth_cache_path: if enable_disk_cache {
            Some(cache_path)
        } else {
            None
        },
        ground_truth_cache_hit: false,
    })
}

/// Checks whether an existing per-index report file matches the current git commit, scale, and config.
#[must_use]
pub fn can_resume_existing_report(
    report_path: &Path,
    expected_commit: &str,
    scale_vectors: usize,
    idx_cfg: &Phase4IndexConfig,
) -> bool {
    if !report_path.is_file() {
        return false;
    }
    let content = match fs::read_to_string(report_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    content.contains("\"measured_this_run\": true")
        && content.contains(&format!("\"git_commit\": \"{expected_commit}\""))
        && content.contains(&format!("\"scale_vectors\": {scale_vectors}"))
        && content.contains(&format!("\"m\": {}", idx_cfg.m))
        && content.contains(&format!("\"ef_construction\": {}", idx_cfg.ef_construction))
}

/// Complete configuration for a Phase 4 sweep execution.
#[derive(Debug, Clone)]
pub struct Phase4SweepConfig {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub grid: Phase4GridSpec,
    pub metric: Metric,
    pub allow_download: bool,
    pub enable_gt_disk_cache: bool,
    pub resume: bool,
    pub max_queries: Option<usize>,
    pub expected_commit_override: Option<String>,
}

impl Default for Phase4SweepConfig {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            data_cache_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("results/phase4"),
            grid: Phase4GridSpec::default(),
            metric: Metric::L2Squared,
            allow_download: true,
            enable_gt_disk_cache: true,
            resume: true,
            max_queries: None,
            expected_commit_override: None,
        }
    }
}

/// Single efSearch evaluation step within an index report.
#[derive(Debug, Clone)]
pub struct Phase4EfEvaluation {
    pub ef_search: usize,
    pub recall_at_k: f64,
    pub summary: RepeatedSweepSummary,
}

/// Memory profile recorded for an index.
#[derive(Debug, Clone)]
pub struct Phase4MemoryProfile {
    pub peak_rss_mb: f64,
    pub graph_only_bytes_per_vector: f64,
    pub bytes_per_vector: f64,
    pub graph_only_memory_bytes: usize,
    pub graph_only_memory_mb: f64,
    pub layer0_bytes: usize,
    pub upper_layers_bytes: usize,
}

/// Complete structured per-index report written by Phase 4.
#[derive(Debug, Clone)]
pub struct Phase4IndexReport {
    pub phase: usize,
    pub phase_name: String,
    pub generated_at: String,
    pub provenance: ReportProvenance,
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub preset_name: String,
    pub scale_vectors: usize,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub ground_truth_source: String,
    pub ground_truth_cache_path: Option<PathBuf>,
    pub ground_truth_cache_hit: bool,
    pub hnsw_config: Phase4IndexConfig,
    pub build_time_s: f64,
    pub build_qps: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub graph_fingerprint: u64,
    pub memory_profile: Phase4MemoryProfile,
    pub warmup_performed: bool,
    pub measured_this_run: bool,
    pub sweep_runs: usize,
    pub ef_search_sweep: Vec<Phase4EfEvaluation>,
    pub monotonic_recall: bool,
}

impl Phase4IndexReport {
    /// Serializes this report to a deterministic, dependency-free JSON string.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(8192);
        out.push_str("{\n");
        out.push_str(&format!("  \"phase\": {},\n", self.phase));
        out.push_str(&format!(
            "  \"phase_name\": \"{}\",\n",
            escape_json(&self.phase_name)
        ));
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&self.generated_at)
        ));

        // Provenance & Environment
        self.provenance.append_json_fields(&mut out, &self.hardware);

        out.push_str(&format!("  \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str(&format!(
            "  \"measured_this_run\": {},\n",
            self.measured_this_run
        ));
        out.push_str(&format!(
            "  \"warmup_performed\": {},\n",
            self.warmup_performed
        ));
        out.push_str(&format!("  \"sweep_runs\": {},\n", self.sweep_runs));

        // Hardware block
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
            "    \"total_ram_gb\": {:.2},\n",
            self.hardware.total_ram_gb.unwrap_or(0.0)
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

        // Dataset block
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
        out.push_str(&format!("    \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str(&format!("    \"metric\": \"{}\"\n", self.metric));
        out.push_str("  },\n");

        // Ground truth metadata
        out.push_str(&format!(
            "  \"ground_truth_source\": \"{}\",\n",
            escape_json(&self.ground_truth_source)
        ));
        out.push_str(&format!(
            "  \"ground_truth_cache_path\": {},\n",
            opt_path_json(self.ground_truth_cache_path.as_deref())
        ));
        out.push_str(&format!(
            "  \"ground_truth_cache_hit\": {},\n",
            self.ground_truth_cache_hit
        ));

        // HNSW Config
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_config.m));
        out.push_str(&format!(
            "    \"m_max\": {},\n",
            self.hnsw_config.m_max.unwrap_or(self.hnsw_config.m)
        ));
        out.push_str(&format!(
            "    \"m_max0\": {},\n",
            self.hnsw_config.m_max0.unwrap_or(2 * self.hnsw_config.m)
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

        // Index Build
        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
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

        // Memory Profile
        out.push_str("  \"memory_profile\": {\n");
        out.push_str(&format!(
            "    \"peak_rss_mb\": {:.2},\n",
            self.memory_profile.peak_rss_mb
        ));
        out.push_str(&format!(
            "    \"graph_only_bytes_per_vector\": {:.2},\n",
            self.memory_profile.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"bytes_per_vector\": {:.2},\n",
            self.memory_profile.bytes_per_vector
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_bytes\": {},\n",
            self.memory_profile.graph_only_memory_bytes
        ));
        out.push_str(&format!(
            "    \"graph_only_memory_mb\": {:.4},\n",
            self.memory_profile.graph_only_memory_mb
        ));
        out.push_str(&format!(
            "    \"layer0_bytes\": {},\n",
            self.memory_profile.layer0_bytes
        ));
        out.push_str(&format!(
            "    \"upper_layers_bytes\": {}\n",
            self.memory_profile.upper_layers_bytes
        ));
        out.push_str("  },\n");

        out.push_str(&format!(
            "  \"graph_only_bytes_per_vector\": {:.2},\n",
            self.memory_profile.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "  \"monotonic_recall\": {},\n",
            self.monotonic_recall
        ));

        // ef_search_sweep
        out.push_str("  \"ef_search_sweep\": [\n");
        for (i, eval) in self.ef_search_sweep.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", eval.ef_search));
            out.push_str(&format!(
                "      \"recall_at_k\": {:.6},\n",
                eval.recall_at_k
            ));
            out.push_str(&format!(
                "      \"qps\": {:.2},\n",
                eval.summary.median_latency.qps
            ));
            out.push_str(&format!(
                "      \"mean_ms\": {:.4},\n",
                eval.summary.median_latency.mean_ms
            ));
            out.push_str(&format!(
                "      \"p50_ms\": {:.4},\n",
                eval.summary.median_latency.p50_ms
            ));
            out.push_str(&format!(
                "      \"p95_ms\": {:.4},\n",
                eval.summary.median_latency.p95_ms
            ));
            out.push_str(&format!(
                "      \"p99_ms\": {:.4},\n",
                eval.summary.median_latency.p99_ms
            ));

            out.push_str("      \"median_latency\": {\n");
            out.push_str(&format!(
                "        \"num_queries\": {},\n",
                eval.summary.median_latency.num_queries
            ));
            out.push_str(&format!(
                "        \"total_s\": {:.6},\n",
                eval.summary.median_latency.total_s
            ));
            out.push_str(&format!(
                "        \"qps\": {:.2},\n",
                eval.summary.median_latency.qps
            ));
            out.push_str(&format!(
                "        \"min_ms\": {:.4},\n",
                eval.summary.median_latency.min_ms
            ));
            out.push_str(&format!(
                "        \"mean_ms\": {:.4},\n",
                eval.summary.median_latency.mean_ms
            ));
            out.push_str(&format!(
                "        \"p50_ms\": {:.4},\n",
                eval.summary.median_latency.p50_ms
            ));
            out.push_str(&format!(
                "        \"p95_ms\": {:.4},\n",
                eval.summary.median_latency.p95_ms
            ));
            out.push_str(&format!(
                "        \"p99_ms\": {:.4},\n",
                eval.summary.median_latency.p99_ms
            ));
            out.push_str(&format!(
                "        \"max_ms\": {:.4}\n",
                eval.summary.median_latency.max_ms
            ));
            out.push_str("      },\n");

            out.push_str(&format!(
                "      \"mean_ms_min\": {:.4},\n",
                eval.summary.mean_ms_min
            ));
            out.push_str(&format!(
                "      \"mean_ms_max\": {:.4},\n",
                eval.summary.mean_ms_max
            ));
            out.push_str(&format!(
                "      \"mean_ms_spread\": {:.4},\n",
                eval.summary.mean_ms_spread
            ));
            out.push_str(&format!(
                "      \"p99_ms_min\": {:.4},\n",
                eval.summary.p99_ms_min
            ));
            out.push_str(&format!(
                "      \"p99_ms_max\": {:.4},\n",
                eval.summary.p99_ms_max
            ));
            out.push_str(&format!(
                "      \"p99_ms_spread\": {:.4},\n",
                eval.summary.p99_ms_spread
            ));

            out.push_str("      \"run_mean_latencies_ms\": [");
            for (rj, r) in eval.summary.run_mean_latencies_ms.iter().enumerate() {
                if rj > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{r:.4}"));
            }
            out.push_str("],\n");

            out.push_str("      \"run_p99_latencies_ms\": [");
            for (rj, r) in eval.summary.run_p99_latencies_ms.iter().enumerate() {
                if rj > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{r:.4}"));
            }
            out.push_str("]\n");

            if i + 1 < self.ef_search_sweep.len() {
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

/// Summary entry for a single configuration within `summary.json`.
#[derive(Debug, Clone)]
pub struct Phase4IndexSummaryEntry {
    pub m: usize,
    pub ef_construction: usize,
    pub keep_pruned_shrink: Option<bool>,
    pub report_filename: String,
    pub status: String,
    pub graph_fingerprint: Option<u64>,
    pub graph_fingerprint_hex: Option<String>,
    pub build_time_s: Option<f64>,
    pub build_qps: Option<f64>,
    pub peak_rss_mb: Option<f64>,
    pub graph_only_bytes_per_vector: Option<f64>,
    pub best_recall: Option<f64>,
    pub best_qps: Option<f64>,
}

/// Structured summary of an entire Phase 4 parameter sweep.
#[derive(Debug, Clone)]
pub struct Phase4SweepSummary {
    pub total_configs: usize,
    pub newly_measured_count: usize,
    pub resumed_skipped_count: usize,
    pub git_commit: String,
    pub git_dirty: bool,
    pub rustc_version: String,
    pub provenance: ReportProvenance,
    pub hardware: HardwareInfo,
    pub scale_vectors: usize,
    pub ground_truth_source: String,
    pub ground_truth_cache_hit: bool,
    pub configs: Vec<Phase4IndexSummaryEntry>,
}

impl Phase4SweepSummary {
    /// Serializes the sweep summary to deterministic JSON.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str("  \"summary_version\": 1,\n");
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&current_utc_rfc3339())
        ));

        // Provenance & Environment
        self.provenance.append_json_fields(&mut out, &self.hardware);

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
            "    \"total_ram_gb\": {:.2},\n",
            self.hardware.total_ram_gb.unwrap_or(0.0)
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

        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!("    \"scale_vectors\": {}\n", self.scale_vectors));
        out.push_str("  },\n");

        out.push_str(&format!("  \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str(&format!(
            "  \"ground_truth_source\": \"{}\",\n",
            escape_json(&self.ground_truth_source)
        ));
        out.push_str(&format!(
            "  \"ground_truth_cache_hit\": {},\n",
            self.ground_truth_cache_hit
        ));
        out.push_str(&format!("  \"total_configs\": {},\n", self.total_configs));
        out.push_str(&format!(
            "  \"newly_measured_count\": {},\n",
            self.newly_measured_count
        ));
        out.push_str(&format!(
            "  \"resumed_skipped_count\": {},\n",
            self.resumed_skipped_count
        ));
        out.push_str("  \"measured_this_run\": true,\n");

        out.push_str("  \"configs\": [\n");
        for (i, entry) in self.configs.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"m\": {},\n", entry.m));
            out.push_str(&format!(
                "      \"ef_construction\": {},\n",
                entry.ef_construction
            ));
            match entry.keep_pruned_shrink {
                Some(kps) => out.push_str(&format!("      \"keep_pruned_shrink\": {kps},\n")),
                None => out.push_str("      \"keep_pruned_shrink\": null,\n"),
            }
            out.push_str(&format!(
                "      \"report_filename\": \"{}\",\n",
                escape_json(&entry.report_filename)
            ));
            out.push_str(&format!(
                "      \"status\": \"{}\",\n",
                escape_json(&entry.status)
            ));
            match &entry.graph_fingerprint_hex {
                Some(hex) => {
                    out.push_str(&format!("      \"graph_fingerprint_hex\": \"{hex}\",\n"))
                }
                None => out.push_str("      \"graph_fingerprint_hex\": null,\n"),
            }
            match entry.graph_fingerprint {
                Some(fp) => out.push_str(&format!("      \"graph_fingerprint\": {fp},\n")),
                None => out.push_str("      \"graph_fingerprint\": null,\n"),
            }
            match entry.build_time_s {
                Some(bt) => out.push_str(&format!("      \"build_time_s\": {bt:.6},\n")),
                None => out.push_str("      \"build_time_s\": null,\n"),
            }
            match entry.build_qps {
                Some(q) => out.push_str(&format!("      \"build_qps\": {q:.2},\n")),
                None => out.push_str("      \"build_qps\": null,\n"),
            }
            match entry.peak_rss_mb {
                Some(rss) => out.push_str(&format!("      \"peak_rss_mb\": {rss:.2},\n")),
                None => out.push_str("      \"peak_rss_mb\": null,\n"),
            }
            match entry.graph_only_bytes_per_vector {
                Some(gb) => out.push_str(&format!(
                    "      \"graph_only_bytes_per_vector\": {gb:.2},\n"
                )),
                None => out.push_str("      \"graph_only_bytes_per_vector\": null,\n"),
            }
            match entry.best_recall {
                Some(r) => out.push_str(&format!("      \"best_recall\": {r:.6},\n")),
                None => out.push_str("      \"best_recall\": null,\n"),
            }
            match entry.best_qps {
                Some(q) => out.push_str(&format!("      \"best_qps\": {q:.2}\n")),
                None => out.push_str("      \"best_qps\": null\n"),
            }

            if i + 1 < self.configs.len() {
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

/// Executes a Phase 4 resumable parameter sweep.
pub fn run_phase4_sweep(config: &Phase4SweepConfig) -> Result<Phase4SweepSummary> {
    let mut prov = detect_provenance();
    if let Some(ref override_commit) = config.expected_commit_override {
        prov.git_commit = override_commit.clone();
    }
    let current_commit = prov.git_commit.clone();
    let hardware = detect_hardware();

    if !config.output_dir.exists() {
        fs::create_dir_all(&config.output_dir)?;
    }

    // 1. Load dataset (using grid.scale_vectors as max_base)
    let loaded: LoadedDataset = match &config.dataset_mode {
        Phase1DatasetMode::Sift(preset) => load_or_download_sift_with_options(
            *preset,
            &config.data_cache_dir,
            config.allow_download,
            config.grid.seed,
            config.grid.scale_vectors,
            config.max_queries,
            true,
        )?,
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let eff_base = config
                .grid
                .scale_vectors
                .map_or(*num_base, |lim| lim.min(*num_base));
            let eff_query = config
                .max_queries
                .map_or(*num_query, |lim| lim.min(*num_query));
            let (base, query) = generate_synthetic_sift(
                eff_base,
                eff_query,
                *dim,
                *num_clusters,
                config.grid.seed,
            )?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic {
                    seed: config.grid.seed,
                },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => load_custom_dataset_with_options(
            base_fvecs,
            query_fvecs,
            gt_ivecs.as_deref(),
            config.grid.scale_vectors,
            config.max_queries,
        )?,
    };

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();

    // 2. Load or compute prefix ground truth once for the entire grid
    let gt = load_or_compute_phase4_ground_truth(
        &loaded,
        config.grid.k,
        config.metric,
        &config.data_cache_dir,
        config.enable_gt_disk_cache,
    )?;

    let mut summary_entries = Vec::with_capacity(config.grid.configs.len());
    let mut newly_measured_count = 0;
    let mut resumed_skipped_count = 0;

    // 3. Iterate over each index configuration
    for idx_cfg in &config.grid.configs {
        let report_filename = idx_cfg.report_filename(num_base);
        let report_path = config.output_dir.join(&report_filename);

        if config.resume
            && can_resume_existing_report(&report_path, &current_commit, num_base, idx_cfg)
        {
            resumed_skipped_count += 1;
            let entry = parse_resumed_summary_entry(&report_path, idx_cfg, &report_filename);
            summary_entries.push(entry);
            continue;
        }

        // Build the index
        let hnsw_cfg = idx_cfg.to_hnsw_config();
        let t_build_start = Instant::now();
        let index = HnswIndex::build(&loaded.base, config.metric, hnsw_cfg)?;
        let build_time_s = t_build_start.elapsed().as_secs_f64();
        let build_qps = if build_time_s > 0.0 {
            (num_base as f64) / build_time_s
        } else {
            0.0
        };

        let max_layer = index.max_layer();
        let entry_point = index.entry_point().unwrap_or(0);
        let graph_fingerprint = index.graph_fingerprint();
        let peak_rss_mb = crate::hardware::get_peak_rss_mb().unwrap_or(0.0);
        let graph_only_bytes_per_vec = index.graph_only_bytes_per_vector();
        let bytes_per_vec = index.bytes_per_vector();
        let graph_components = index.graph_components();
        let graph_only_bytes = index.graph_only_memory_bytes();
        let graph_only_mb = index.graph_only_memory_mb();

        let memory_profile = Phase4MemoryProfile {
            peak_rss_mb,
            graph_only_bytes_per_vector: graph_only_bytes_per_vec,
            bytes_per_vector: bytes_per_vec,
            graph_only_memory_bytes: graph_only_bytes,
            graph_only_memory_mb: graph_only_mb,
            layer0_bytes: graph_components.layer0_edges_bytes,
            upper_layers_bytes: graph_components.upper_layers_edges_bytes,
        };

        // Prepare visited tracker
        let mut tracker = VisitedTracker::new(num_base);
        let mut ef_evaluations = Vec::with_capacity(config.grid.ef_search_list.len());
        let mut monotonic_recall = true;
        let mut prev_recall = -1.0;
        let mut best_recall: Option<f64> = None;
        let mut best_qps: Option<f64> = None;

        for &ef in &config.grid.ef_search_list {
            // 1 untimed warmup pass
            let _ = index.search_batch_timed_with_tracker(
                &loaded.query,
                config.grid.k,
                ef,
                &mut tracker,
            )?;

            // sweep_runs timed passes
            let mut run_latencies = Vec::with_capacity(config.grid.sweep_runs);
            let mut last_indices = None;
            for _ in 0..config.grid.sweep_runs {
                let timed_res = index.search_batch_timed_with_tracker(
                    &loaded.query,
                    config.grid.k,
                    ef,
                    &mut tracker,
                )?;
                run_latencies.push(timed_res.latency);
                last_indices = Some(timed_res.result.indices);
            }
            let summary = summarize_repeated_latencies(&run_latencies);
            let recall_at_k = if let Some(ref ind) = last_indices {
                compute_recall_at_k(ind, &gt.matrix, config.grid.k)?
            } else {
                0.0
            };

            if prev_recall >= 0.0 && recall_at_k + 1e-9 < prev_recall {
                monotonic_recall = false;
            }
            prev_recall = recall_at_k;

            best_recall = Some(best_recall.map_or(recall_at_k, |b| b.max(recall_at_k)));
            best_qps = Some(best_qps.map_or(summary.median_latency.qps, |b| {
                b.max(summary.median_latency.qps)
            }));

            ef_evaluations.push(Phase4EfEvaluation {
                ef_search: ef,
                recall_at_k,
                summary,
            });
        }

        let preset_name = match &config.dataset_mode {
            Phase1DatasetMode::Sift(SiftPreset::SiftSmall) => "siftsmall".to_string(),
            Phase1DatasetMode::Sift(SiftPreset::Sift1M) => "sift1m".to_string(),
            Phase1DatasetMode::Synthetic { .. } => "synthetic".to_string(),
            Phase1DatasetMode::Custom { .. } => "custom".to_string(),
        };

        let report = Phase4IndexReport {
            phase: 4,
            phase_name: "Phase 4 Parameter Sweep & Pareto Frontier".to_string(),
            generated_at: current_utc_rfc3339(),
            provenance: prov.clone(),
            hardware: hardware.clone(),
            dataset_source: loaded.source.clone(),
            preset_name,
            scale_vectors: num_base,
            num_base,
            num_query,
            dim,
            k: config.grid.k,
            seed: idx_cfg.seed,
            metric: config.metric,
            ground_truth_source: gt.ground_truth_source.clone(),
            ground_truth_cache_path: gt.ground_truth_cache_path.clone(),
            ground_truth_cache_hit: gt.ground_truth_cache_hit,
            hnsw_config: idx_cfg.clone(),
            build_time_s,
            build_qps,
            max_layer,
            entry_point,
            graph_fingerprint,
            memory_profile,
            warmup_performed: true,
            measured_this_run: true,
            sweep_runs: config.grid.sweep_runs,
            ef_search_sweep: ef_evaluations,
            monotonic_recall,
        };

        fs::write(&report_path, report.to_json_string())?;
        newly_measured_count += 1;

        summary_entries.push(Phase4IndexSummaryEntry {
            m: idx_cfg.m,
            ef_construction: idx_cfg.ef_construction,
            keep_pruned_shrink: idx_cfg.keep_pruned_shrink,
            report_filename,
            status: "measured".to_string(),
            graph_fingerprint: Some(graph_fingerprint),
            graph_fingerprint_hex: Some(format!("0x{graph_fingerprint:016x}")),
            build_time_s: Some(build_time_s),
            build_qps: Some(build_qps),
            peak_rss_mb: Some(peak_rss_mb),
            graph_only_bytes_per_vector: Some(graph_only_bytes_per_vec),
            best_recall,
            best_qps,
        });
    }

    let summary = Phase4SweepSummary {
        total_configs: config.grid.configs.len(),
        newly_measured_count,
        resumed_skipped_count,
        git_commit: current_commit,
        git_dirty: prov.git_dirty,
        rustc_version: prov.rustc_version.clone(),
        provenance: prov,
        hardware,
        scale_vectors: num_base,
        ground_truth_source: gt.ground_truth_source,
        ground_truth_cache_hit: gt.ground_truth_cache_hit,
        configs: summary_entries,
    };

    let summary_path = config.output_dir.join("summary.json");
    fs::write(&summary_path, summary.to_json_string())?;

    Ok(summary)
}

fn parse_resumed_summary_entry(
    report_path: &Path,
    idx_cfg: &Phase4IndexConfig,
    report_filename: &str,
) -> Phase4IndexSummaryEntry {
    let content = fs::read_to_string(report_path).unwrap_or_default();
    let graph_fingerprint_hex = extract_json_string_field(&content, "graph_fingerprint_hex");
    let graph_fingerprint = extract_json_u64_field(&content, "graph_fingerprint");
    let build_time_s = extract_json_f64_field(&content, "build_time_s");
    let build_qps = extract_json_f64_field(&content, "build_qps");
    let peak_rss_mb = extract_json_f64_field(&content, "peak_rss_mb");
    let graph_only_bytes_per_vector =
        extract_json_f64_field(&content, "graph_only_bytes_per_vector");

    Phase4IndexSummaryEntry {
        m: idx_cfg.m,
        ef_construction: idx_cfg.ef_construction,
        keep_pruned_shrink: idx_cfg.keep_pruned_shrink,
        report_filename: report_filename.to_string(),
        status: "resumed".to_string(),
        graph_fingerprint,
        graph_fingerprint_hex,
        build_time_s,
        build_qps,
        peak_rss_mb,
        graph_only_bytes_per_vector,
        best_recall: None,
        best_qps: None,
    }
}

fn extract_json_string_field(content: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\": \"");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let rem = &content[start..];
    let end = rem.find('"')?;
    Some(rem[..end].to_string())
}

fn extract_json_f64_field(content: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\": ");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let rem = &content[start..];
    let val_str: String = rem
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
        .collect();
    val_str.parse().ok()
}

fn extract_json_u64_field(content: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\": ");
    let pos = content.find(&pattern)?;
    let start = pos + pattern.len();
    let rem = &content[start..];
    let val_str: String = rem.chars().take_while(|c| c.is_ascii_digit()).collect();
    val_str.parse().ok()
}

/// Generates a current UTC RFC3339 timestamp with std only.
#[must_use]
pub fn current_utc_rfc3339() -> String {
    let now = std::time::SystemTime::now();
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let rem_secs = secs % 86400;
    let hours = rem_secs / 3600;
    let mins = (rem_secs % 3600) / 60;
    let s = rem_secs % 60;

    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn opt_path_json(p: Option<&Path>) -> String {
    match p {
        Some(path) => format!("\"{}\"", escape_json(&path.display().to_string())),
        None => "null".to_string(),
    }
}

/// Executes Phase 4 CLI with given arguments.
pub fn run_phase4_cli(args: &[String]) -> Result<Phase4SweepSummary> {
    let mut grid_path: Option<PathBuf> = None;
    let mut dataset_str = "sift1m".to_string();
    let mut cli_scale: Option<usize> = None;
    let mut cli_max_queries: Option<usize> = None;
    let mut cli_dim: usize = 128;
    let mut cli_k: Option<usize> = None;
    let mut cli_m_list: Option<Vec<usize>> = None;
    let mut cli_efc_list: Option<Vec<usize>> = None;
    let mut cli_ef_search_list: Option<Vec<usize>> = None;
    let mut cli_sweep_runs: Option<usize> = None;
    let mut cli_seed: Option<u64> = None;
    let mut output_dir = PathBuf::from("results/phase4");
    let mut data_dir = PathBuf::from("data");
    let mut resume = true;
    let mut allow_download = true;
    let mut enable_gt_disk_cache = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    "Usage: phase4 [OPTIONS]\n\n\
                    Options:\n  \
                      --grid <path>               Path to .grid specification file\n  \
                      --dataset <name>            Dataset preset: siftsmall, sift1m, synthetic (default: sift1m)\n  \
                      --scale, --max-base <N>     Base vector scale (default: 100000)\n  \
                      --max-queries <N>           Query count limit\n  \
                      --dim <N>                   Dimensionality for synthetic dataset (default: 128)\n  \
                      --k <N>                     Number of nearest neighbors (default: 10)\n  \
                      --m-list <csv>              Comma-separated M values (default: 16)\n  \
                      --ef-construction-list <csv> Comma-separated efConstruction values (default: 100)\n  \
                      --ef-search-list <csv>      Comma-separated efSearch sweep ladder\n  \
                      --sweep-runs <N>            Repeated latency passes per efSearch (default: 3)\n  \
                      --seed <N>                  PRNG seed (default: 42)\n  \
                      --output-dir <path>         Output directory for JSON reports (default: results/phase4)\n  \
                      --data-dir <path>           Dataset and ground-truth cache directory (default: data)\n  \
                      --no-resume                 Force re-execution of previously measured configs\n  \
                      --offline                   Disallow network download of datasets\n  \
                      --no-gt-cache               Disable disk caching of prefix ground truth\n  \
                      -h, --help                  Print help information"
                );
                return Ok(Phase4SweepSummary {
                    total_configs: 0,
                    newly_measured_count: 0,
                    resumed_skipped_count: 0,
                    git_commit: "none".to_string(),
                    git_dirty: false,
                    rustc_version: "none".to_string(),
                    provenance: detect_provenance(),
                    hardware: detect_hardware(),
                    scale_vectors: 0,
                    ground_truth_source: "none".to_string(),
                    ground_truth_cache_hit: false,
                    configs: Vec::new(),
                });
            }
            "--grid" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--grid requires a path argument".to_string(),
                    ));
                }
                grid_path = Some(PathBuf::from(&args[i]));
            }
            "--dataset" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--dataset requires a value".to_string(),
                    ));
                }
                dataset_str = args[i].clone();
            }
            "--scale" | "--max-base" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--scale requires an integer value".to_string(),
                    ));
                }
                cli_scale = Some(args[i].parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --scale: {e}"))
                })?);
            }
            "--max-queries" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--max-queries requires an integer value".to_string(),
                    ));
                }
                cli_max_queries = Some(args[i].parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --max-queries: {e}"))
                })?);
            }
            "--dim" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--dim requires an integer value".to_string(),
                    ));
                }
                cli_dim = args[i].parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --dim: {e}"))
                })?;
            }
            "--k" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--k requires an integer value".to_string(),
                    ));
                }
                cli_k = Some(args[i].parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --k: {e}"))
                })?);
            }
            "--m-list" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--m-list requires a csv list".to_string(),
                    ));
                }
                cli_m_list = Some(parse_comma_separated_usizes(&args[i], 0, "--m-list")?);
            }
            "--ef-construction-list" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--ef-construction-list requires a csv list".to_string(),
                    ));
                }
                cli_efc_list = Some(parse_comma_separated_usizes(
                    &args[i],
                    0,
                    "--ef-construction-list",
                )?);
            }
            "--ef-search-list" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--ef-search-list requires a csv list".to_string(),
                    ));
                }
                cli_ef_search_list = Some(parse_comma_separated_usizes(
                    &args[i],
                    0,
                    "--ef-search-list",
                )?);
            }
            "--sweep-runs" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--sweep-runs requires an integer value".to_string(),
                    ));
                }
                cli_sweep_runs = Some(args[i].parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --sweep-runs: {e}"))
                })?);
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--seed requires an integer value".to_string(),
                    ));
                }
                cli_seed = Some(args[i].parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --seed: {e}"))
                })?);
            }
            "--output-dir" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--output-dir requires a path argument".to_string(),
                    ));
                }
                output_dir = PathBuf::from(&args[i]);
            }
            "--data-dir" => {
                i += 1;
                if i >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "--data-dir requires a path argument".to_string(),
                    ));
                }
                data_dir = PathBuf::from(&args[i]);
            }
            "--no-resume" => {
                resume = false;
            }
            "--offline" => {
                allow_download = false;
            }
            "--no-gt-cache" => {
                enable_gt_disk_cache = false;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI argument: '{unknown}'"
                )));
            }
        }
        i += 1;
    }

    let mut grid = if let Some(ref gp) = grid_path {
        Phase4GridSpec::from_file(gp)?
    } else {
        let m_list = cli_m_list.unwrap_or_else(|| vec![16]);
        let efc_list = cli_efc_list.unwrap_or_else(|| vec![100]);
        let seed = cli_seed.unwrap_or(42);
        let mut configs = Vec::new();
        for &m_val in &m_list {
            for &efc_val in &efc_list {
                if efc_val >= m_val {
                    configs.push(Phase4IndexConfig {
                        m: m_val,
                        m_max: None,
                        m_max0: None,
                        ef_construction: efc_val,
                        keep_pruned: true,
                        keep_pruned_shrink: None,
                        seed,
                    });
                }
            }
        }
        if configs.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "No valid configurations formed (ef_construction must be >= m)".to_string(),
            ));
        }

        Phase4GridSpec {
            scale_vectors: cli_scale.or(Some(100_000)),
            k: cli_k.unwrap_or(10),
            sweep_runs: cli_sweep_runs.unwrap_or(3),
            seed,
            ef_search_list: cli_ef_search_list
                .clone()
                .unwrap_or_else(|| vec![10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600]),
            configs,
        }
    };

    // Apply CLI overrides to grid if specified
    if let Some(s) = cli_scale {
        grid.scale_vectors = Some(s);
    } else if grid.scale_vectors.is_none() {
        grid.scale_vectors = Some(100_000);
    }
    if let Some(k_val) = cli_k {
        grid.k = k_val;
    }
    if let Some(sr) = cli_sweep_runs {
        grid.sweep_runs = sr;
    }
    if let Some(s) = cli_seed {
        grid.seed = s;
    }
    if let Some(ef_list) = cli_ef_search_list {
        grid.ef_search_list = ef_list;
    }

    let dataset_mode = match dataset_str.to_ascii_lowercase().as_str() {
        "siftsmall" => Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
        "sift1m" => Phase1DatasetMode::Sift(SiftPreset::Sift1M),
        "synthetic" => Phase1DatasetMode::Synthetic {
            num_base: grid.scale_vectors.unwrap_or(100_000),
            num_query: cli_max_queries.unwrap_or(10_000),
            dim: cli_dim,
            num_clusters: 100,
        },
        other => {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Unknown dataset '{other}': expected siftsmall, sift1m, or synthetic"
            )));
        }
    };

    let sweep_cfg = Phase4SweepConfig {
        dataset_mode,
        data_cache_dir: data_dir,
        output_dir,
        grid,
        metric: Metric::L2Squared,
        allow_download,
        enable_gt_disk_cache,
        resume,
        max_queries: cli_max_queries,
        expected_commit_override: None,
    };

    let summary = run_phase4_sweep(&sweep_cfg)?;
    println!(
        "[Phase 4] Parameter Sweep complete: {} total configs ({} measured, {} resumed/skipped). Output: {}",
        summary.total_configs,
        summary.newly_measured_count,
        summary.resumed_skipped_count,
        sweep_cfg.output_dir.display()
    );

    Ok(summary)
}
