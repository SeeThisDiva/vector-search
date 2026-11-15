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

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::benchmark::{
    append_hardware_json, compute_query_results_hash, current_utc_rfc3339, detect_provenance,
    escape_json, opt_path_json, summarize_repeated_latencies, RepeatedSweepSummary,
    ReportProvenance,
};
use crate::dataset::{
    generate_mode_a_bitset, generate_mode_b_bitset_detailed, generate_synthetic_sift,
    load_or_download_sift_with_options, read_bitset, read_fvecs, read_ivecs, write_bitset,
    write_fvecs, write_ivecs, DatasetSource, LoadedDataset, SiftPreset, S_CLUST_DEFAULT,
};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::FilteredExactKnn;
use crate::filter::{
    DenseBitset, FilteredSearchOutput, FilteredSearchParams, LambdaPredicate,
    DEFAULT_OPAQUE_EVAL_LIMIT,
};
use crate::hardware::{detect_hardware, get_peak_rss_mb, HardwareInfo};
use crate::hnsw::{
    fnv1a_64_update_u64, FilterStrategy, FilteredSearchCounters, FilteredSearchScratch, HnswConfig,
    HnswIndex, FNV_OFFSET_BASIS_64,
};
use crate::matrix::RowMatrix;
use crate::recall::{compute_latency_stats, percentile_sorted, LatencyStats};

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

/// Computes canonical 64-bit FNV-1a hash over a `RowMatrix<f32>` (dimensions and exact IEEE bits).
#[must_use]
pub fn compute_f32_matrix_hash(matrix: &RowMatrix<f32>) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    fnv1a_64_update_u64(&mut hash, matrix.rows() as u64);
    fnv1a_64_update_u64(&mut hash, matrix.cols() as u64);
    for &val in matrix.as_slice() {
        fnv1a_64_update_u64(&mut hash, val.to_bits() as u64);
    }
    hash
}

/// Computes canonical 64-bit FNV-1a hash over a `DenseBitset` (`len` and packed 64-bit words).
#[must_use]
pub fn compute_bitset_hash(bitset: &DenseBitset) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    fnv1a_64_update_u64(&mut hash, bitset.len() as u64);
    for &w in bitset.words() {
        fnv1a_64_update_u64(&mut hash, w);
    }
    hash
}

/// Returns canonical dataset subdirectory tag under `data/phase5/`.
#[must_use]
pub fn dataset_cache_tag(
    source: &DatasetSource,
    scale: usize,
    dim: usize,
    attr_seed: u64,
) -> String {
    match source {
        DatasetSource::AuthenticSiftSmall { .. } => "siftsmall".to_string(),
        DatasetSource::AuthenticSift1M { .. } => "sift1m".to_string(),
        _ => format!("{}_n{scale}_d{dim}_s{attr_seed}", source.kind_tag()),
    }
}

/// Cached predicate bitset and its verified provenance.
#[derive(Debug, Clone)]
pub struct PredicateCacheArtifact {
    pub bitset: DenseBitset,
    pub cache_path: PathBuf,
    pub provenance_path: PathBuf,
    pub cache_hit: bool,
    pub gen_time_s: f64,
    pub expected_popcount: usize,
    pub actual_popcount: usize,
    pub clusters_c: usize,
    pub bitset_hash_hex: String,
}

/// Cached filtered exact ground truth (`.ivecs` indices + `.fvecs` distances) and provenance.
#[derive(Debug, Clone)]
pub struct FilteredGtCacheArtifact {
    pub indices: RowMatrix<i32>,
    pub distances: RowMatrix<f32>,
    pub ivecs_path: PathBuf,
    pub fvecs_path: PathBuf,
    pub provenance_path: PathBuf,
    pub cache_hit: bool,
    pub gen_time_s: f64,
    pub gt_hash_hex: String,
}

#[allow(clippy::too_many_arguments)]
fn validate_predicate_provenance(
    prov_path: &Path,
    expected_scale: usize,
    mode: MetadataMode,
    sigma: f64,
    attr_seed: u64,
    clust_seed: u64,
    base_hash_hex: &str,
    expected_popcount: usize,
    bitset_hash_hex: &str,
) -> Option<usize> {
    let content = fs::read_to_string(prov_path).ok()?;
    let ver = extract_json_u64_field(&content, "pred_gen_version")? as u32;
    if ver != PRED_GEN_VERSION {
        return None;
    }
    if extract_json_u64_field(&content, "scale")? as usize != expected_scale {
        return None;
    }
    if extract_json_string_field(&content, "mode")? != mode.as_str() {
        return None;
    }
    if extract_json_string_field(&content, "sigma_slug")? != sigma_slug(sigma) {
        return None;
    }
    if extract_json_u64_field(&content, "attr_seed")? != attr_seed {
        return None;
    }
    let expected_cs = match mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    if extract_json_u64_field(&content, "clust_seed")? != expected_cs {
        return None;
    }
    if extract_json_string_field(&content, "base_hash_hex")? != base_hash_hex {
        return None;
    }
    if extract_json_u64_field(&content, "actual_popcount")? as usize != expected_popcount {
        return None;
    }
    if extract_json_string_field(&content, "bitset_hash_hex")? != bitset_hash_hex {
        return None;
    }
    let clusters_c = extract_json_u64_field(&content, "clusters_c")? as usize;
    Some(clusters_c)
}

/// Loads a predicate bitset from `data/phase5/<tag>/` if versioned provenance and popcount/hash verify,
/// otherwise generates it deterministically and caches `.bin` + `.json`.
#[allow(clippy::too_many_arguments)]
pub fn load_or_generate_predicate_cache(
    cache_dir: &Path,
    base: &RowMatrix<f32>,
    base_hash_hex: &str,
    mode: MetadataMode,
    sigma: f64,
    attr_seed: u64,
    clust_seed: u64,
) -> Result<PredicateCacheArtifact> {
    if !sigma.is_finite() || sigma < 0.0 || sigma > 1.0 {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Selectivity sigma must lie in [0.0, 1.0], got {sigma}"
        )));
    }
    fs::create_dir_all(cache_dir)?;

    let scale = base.rows();
    let expected_popcount = (sigma * (scale as f64)).floor() as usize;
    let effective_cs = match mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    let stem = format!(
        "pred_n{scale}_{}_{}_as{attr_seed}_cs{effective_cs}_pv{PRED_GEN_VERSION}",
        mode.as_str(),
        sigma_slug(sigma),
    );
    let bin_path = cache_dir.join(format!("{stem}.bin"));
    let json_path = cache_dir.join(format!("{stem}.json"));

    if bin_path.is_file() && json_path.is_file() {
        if let Ok(bitset) = read_bitset(&bin_path) {
            let actual_popcount = bitset.count_ones();
            if bitset.len() == scale && actual_popcount == expected_popcount {
                let hash_hex = format!("0x{:016x}", compute_bitset_hash(&bitset));
                if let Some(clusters_c) = validate_predicate_provenance(
                    &json_path,
                    scale,
                    mode,
                    sigma,
                    attr_seed,
                    clust_seed,
                    base_hash_hex,
                    expected_popcount,
                    &hash_hex,
                ) {
                    return Ok(PredicateCacheArtifact {
                        bitset,
                        cache_path: bin_path,
                        provenance_path: json_path,
                        cache_hit: true,
                        gen_time_s: 0.0,
                        expected_popcount,
                        actual_popcount,
                        clusters_c,
                        bitset_hash_hex: hash_hex,
                    });
                }
            }
        }
        eprintln!(
            "[WARN] Invalid or stale predicate cache at {}; regenerating...",
            bin_path.display()
        );
    }

    let t0 = Instant::now();
    let (bitset, clusters_c, active_clusters, shortfall_cluster, shortfall_count) = match mode {
        MetadataMode::Independent => (
            generate_mode_a_bitset(scale, sigma, attr_seed),
            0_usize,
            Vec::new(),
            None,
            0_usize,
        ),
        MetadataMode::Correlated => {
            let meta = generate_mode_b_bitset_detailed(base, sigma, attr_seed, effective_cs);
            (
                meta.bitset,
                meta.centroid_count,
                meta.active_clusters,
                meta.shortfall_cluster,
                meta.shortfall_count,
            )
        }
    };
    let gen_time_s = t0.elapsed().as_secs_f64();
    let actual_popcount = bitset.count_ones();
    if actual_popcount != expected_popcount {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Generated bitset popcount {actual_popcount} != expected {expected_popcount}"
        )));
    }

    let bitset_hash_hex = format!("0x{:016x}", compute_bitset_hash(&bitset));
    write_bitset(&bin_path, &bitset)?;

    let active_json = active_clusters
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let shortfall_json = shortfall_cluster.map_or_else(|| "null".to_string(), |c| c.to_string());
    let meta_json = format!(
        r#"{{
  "pred_gen_version": {PRED_GEN_VERSION},
  "scale": {scale},
  "mode": "{}",
  "sigma": {sigma:.6},
  "sigma_slug": "{}",
  "attr_seed": {attr_seed},
  "clust_seed": {effective_cs},
  "base_hash_hex": "{}",
  "expected_popcount": {expected_popcount},
  "actual_popcount": {actual_popcount},
  "bitset_hash_hex": "{}",
  "clusters_c": {clusters_c},
  "active_clusters": [{active_json}],
  "shortfall_cluster": {shortfall_json},
  "shortfall_count": {shortfall_count},
  "gen_time_s": {gen_time_s:.6},
  "generated_at": "{}"
}}
"#,
        mode.as_str(),
        sigma_slug(sigma),
        escape_json(base_hash_hex),
        escape_json(&bitset_hash_hex),
        escape_json(&current_utc_rfc3339()),
    );
    fs::write(&json_path, meta_json)?;

    Ok(PredicateCacheArtifact {
        bitset,
        cache_path: bin_path,
        provenance_path: json_path,
        cache_hit: false,
        gen_time_s,
        expected_popcount,
        actual_popcount,
        clusters_c,
        bitset_hash_hex,
    })
}

fn verify_loaded_gt_semantics(
    indices: &RowMatrix<i32>,
    distances: &RowMatrix<f32>,
    num_queries: usize,
    effective_k: usize,
    scale: usize,
    bitset: &DenseBitset,
) -> bool {
    if indices.rows() != num_queries || distances.rows() != num_queries {
        return false;
    }
    if indices.cols() != effective_k || distances.cols() != effective_k {
        return false;
    }
    if effective_k == 0 {
        return true;
    }
    let mut seen = HashSet::with_capacity(effective_k * 2);
    for q in 0..num_queries {
        seen.clear();
        let idx_row = indices.row(q);
        let dist_row = distances.row(q);
        let mut prev_key: Option<(f32, i32)> = None;
        for col in 0..effective_k {
            let id = idx_row[col];
            let dist = dist_row[col];
            if id < 0 || (id as usize) >= scale {
                return false;
            }
            if !bitset.get_bit(id as usize) {
                return false;
            }
            if !dist.is_finite() || dist < 0.0 || !seen.insert(id) {
                return false;
            }
            if let Some((prev_d, prev_id)) = prev_key {
                if dist < prev_d || (dist == prev_d && id <= prev_id) {
                    return false;
                }
            }
            prev_key = Some((dist, id));
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn validate_gt_provenance(
    prov_path: &Path,
    scale: usize,
    mode: MetadataMode,
    sigma: f64,
    num_queries: usize,
    k: usize,
    effective_k: usize,
    attr_seed: u64,
    clust_seed: u64,
    base_hash_hex: &str,
    query_hash_hex: &str,
    bitset_hash_hex: &str,
    gt_hash_hex: &str,
) -> bool {
    let Ok(content) = fs::read_to_string(prov_path) else {
        return false;
    };
    let effective_cs = match mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    extract_json_u64_field(&content, "pred_gen_version") == Some(u64::from(PRED_GEN_VERSION))
        && extract_json_u64_field(&content, "gt_gen_version") == Some(u64::from(GT_GEN_VERSION))
        && extract_json_u64_field(&content, "scale") == Some(scale as u64)
        && extract_json_string_field(&content, "mode").as_deref() == Some(mode.as_str())
        && extract_json_string_field(&content, "sigma_slug").as_deref() == Some(&sigma_slug(sigma))
        && extract_json_u64_field(&content, "num_queries") == Some(num_queries as u64)
        && extract_json_u64_field(&content, "k") == Some(k as u64)
        && extract_json_u64_field(&content, "effective_k") == Some(effective_k as u64)
        && extract_json_u64_field(&content, "attr_seed") == Some(attr_seed)
        && extract_json_u64_field(&content, "clust_seed") == Some(effective_cs)
        && extract_json_string_field(&content, "base_hash_hex").as_deref() == Some(base_hash_hex)
        && extract_json_string_field(&content, "query_hash_hex").as_deref() == Some(query_hash_hex)
        && extract_json_string_field(&content, "bitset_hash_hex").as_deref()
            == Some(bitset_hash_hex)
        && extract_json_string_field(&content, "gt_hash_hex").as_deref() == Some(gt_hash_hex)
}

/// Loads cached `.ivecs` + `.fvecs` filtered ground truth if valid, or computes it via `FilteredExactKnn`.
#[allow(clippy::too_many_arguments)]
pub fn load_or_compute_filtered_gt_cache(
    cache_dir: &Path,
    exact_engine: Option<&FilteredExactKnn>,
    base: &RowMatrix<f32>,
    queries: &RowMatrix<f32>,
    base_hash_hex: &str,
    query_hash_hex: &str,
    pred: &PredicateCacheArtifact,
    mode: MetadataMode,
    sigma: f64,
    k: usize,
    attr_seed: u64,
    clust_seed: u64,
    gt_threads: usize,
) -> Result<FilteredGtCacheArtifact> {
    fs::create_dir_all(cache_dir)?;

    let scale = base.rows();
    let num_queries = queries.rows();
    let effective_k = k.min(pred.actual_popcount);
    let effective_cs = match mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    let stem = format!(
        "gt_n{scale}_{}_{}_q{num_queries}_k{k}_as{attr_seed}_cs{effective_cs}_pv{PRED_GEN_VERSION}_gv{GT_GEN_VERSION}",
        mode.as_str(),
        sigma_slug(sigma),
    );
    let ivecs_path = cache_dir.join(format!("{stem}.ivecs"));
    let fvecs_path = cache_dir.join(format!("{stem}.fvecs"));
    let json_path = cache_dir.join(format!("{stem}.json"));

    if ivecs_path.is_file() && fvecs_path.is_file() && json_path.is_file() {
        if let (Ok(indices), Ok(distances)) = (read_ivecs(&ivecs_path), read_fvecs(&fvecs_path)) {
            if verify_loaded_gt_semantics(
                &indices,
                &distances,
                num_queries,
                effective_k,
                scale,
                &pred.bitset,
            ) {
                let gt_hash_hex = format!(
                    "0x{:016x}",
                    compute_query_results_hash(&indices, &distances)
                );
                if validate_gt_provenance(
                    &json_path,
                    scale,
                    mode,
                    sigma,
                    num_queries,
                    k,
                    effective_k,
                    attr_seed,
                    clust_seed,
                    base_hash_hex,
                    query_hash_hex,
                    &pred.bitset_hash_hex,
                    &gt_hash_hex,
                ) {
                    return Ok(FilteredGtCacheArtifact {
                        indices,
                        distances,
                        ivecs_path,
                        fvecs_path,
                        provenance_path: json_path,
                        cache_hit: true,
                        gen_time_s: 0.0,
                        gt_hash_hex,
                    });
                }
            }
        }
        eprintln!(
            "[WARN] Invalid or stale GT cache at {}; regenerating...",
            ivecs_path.display()
        );
    }

    let t0 = Instant::now();
    let owned_engine;
    let engine = match exact_engine {
        Some(e) => e,
        None => {
            owned_engine = FilteredExactKnn::new(Metric::L2Squared, base.clone())?;
            &owned_engine
        }
    };
    let gt = engine.search_batch_parallel(queries, k, &pred.bitset, gt_threads)?;
    let gen_time_s = t0.elapsed().as_secs_f64();

    let gt_hash_hex = format!(
        "0x{:016x}",
        compute_query_results_hash(&gt.indices, &gt.distances)
    );
    write_ivecs(&ivecs_path, &gt.indices)?;
    write_fvecs(&fvecs_path, &gt.distances)?;

    let meta_json = format!(
        r#"{{
  "pred_gen_version": {PRED_GEN_VERSION},
  "gt_gen_version": {GT_GEN_VERSION},
  "scale": {scale},
  "mode": "{}",
  "sigma": {sigma:.6},
  "sigma_slug": "{}",
  "num_queries": {num_queries},
  "k": {k},
  "effective_k": {effective_k},
  "attr_seed": {attr_seed},
  "clust_seed": {effective_cs},
  "base_hash_hex": "{}",
  "query_hash_hex": "{}",
  "bitset_hash_hex": "{}",
  "gt_hash_hex": "{}",
  "gen_time_s": {gen_time_s:.6},
  "generated_at": "{}"
}}
"#,
        mode.as_str(),
        sigma_slug(sigma),
        escape_json(base_hash_hex),
        escape_json(query_hash_hex),
        escape_json(&pred.bitset_hash_hex),
        escape_json(&gt_hash_hex),
        escape_json(&current_utc_rfc3339()),
    );
    fs::write(&json_path, meta_json)?;

    Ok(FilteredGtCacheArtifact {
        indices: gt.indices,
        distances: gt.distances,
        ivecs_path,
        fvecs_path,
        provenance_path: json_path,
        cache_hit: false,
        gen_time_s,
        gt_hash_hex,
    })
}

/// Complete measured statistics for one benchmark arm within a `(scale, mode, sigma)` cell.
#[derive(Debug, Clone)]
pub struct Phase5ArmRecord {
    pub strategy: String,
    pub strategy_label: String,
    pub ef_search: usize,
    pub overfetch_cap: usize,
    pub overfetch_factor: Option<f64>,
    pub max_bridge_hops: Option<usize>,
    pub eval_limit: Option<usize>,
    pub num_queries: usize,
    pub is_subquery_probe: bool,
    pub reps: usize,
    pub recall_at_k: f64,
    pub fewer_than_k_count: usize,
    pub fewer_than_k_rate: f64,
    pub post_filter_cap_reached_rate: f64,
    pub hop_budget_exhausted_rate: f64,
    pub eval_limit_reached_rate: f64,
    pub used_bruteforce_fallback_rate: f64,
    pub summary: RepeatedSweepSummary,
    pub dist_evals_mean: f64,
    pub dist_evals_p50: f64,
    pub dist_evals_p99: f64,
    pub valid_dist_evals_mean: f64,
    pub valid_dist_evals_p50: f64,
    pub valid_dist_evals_p99: f64,
    pub bridge_dist_evals_mean: f64,
    pub bridge_dist_evals_p50: f64,
    pub bridge_dist_evals_p99: f64,
    pub upper_layer_dist_evals_mean: f64,
    pub visited_nodes_mean: f64,
    pub visited_matching_nodes_mean: f64,
    pub eta_graph_mean: f64,
    pub eta_graph_pooled: f64,
}

impl Phase5ArmRecord {
    fn to_json_string(&self, indent: &str) -> String {
        let overfetch_factor_json = self
            .overfetch_factor
            .map_or_else(|| "null".to_string(), |v| format!("{v:.6}"));
        let max_hops_json = self
            .max_bridge_hops
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let eval_limit_json = self
            .eval_limit
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let run_means = self
            .summary
            .run_mean_latencies_ms
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"{indent}{{
{indent}  "strategy": "{}",
{indent}  "strategy_label": "{}",
{indent}  "ef_search": {},
{indent}  "overfetch_cap": {},
{indent}  "overfetch_factor": {overfetch_factor_json},
{indent}  "max_bridge_hops": {max_hops_json},
{indent}  "eval_limit": {eval_limit_json},
{indent}  "num_queries": {},
{indent}  "is_subquery_probe": {},
{indent}  "reps": {},
{indent}  "recall_at_k": {:.6},
{indent}  "fewer_than_k_count": {},
{indent}  "fewer_than_k_rate": {:.6},
{indent}  "post_filter_cap_reached_rate": {:.6},
{indent}  "hop_budget_exhausted_rate": {:.6},
{indent}  "eval_limit_reached_rate": {:.6},
{indent}  "used_bruteforce_fallback_rate": {:.6},
{indent}  "mean_ms": {:.6},
{indent}  "p50_ms": {:.6},
{indent}  "p95_ms": {:.6},
{indent}  "p99_ms": {:.6},
{indent}  "qps": {:.2},
{indent}  "mean_ms_min": {:.6},
{indent}  "mean_ms_max": {:.6},
{indent}  "mean_ms_spread": {:.6},
{indent}  "p99_ms_min": {:.6},
{indent}  "p99_ms_max": {:.6},
{indent}  "p99_ms_spread": {:.6},
{indent}  "run_mean_latencies_ms": [{run_means}],
{indent}  "dist_evals_mean": {:.4},
{indent}  "dist_evals_p50": {:.4},
{indent}  "dist_evals_p99": {:.4},
{indent}  "valid_dist_evals_mean": {:.4},
{indent}  "valid_dist_evals_p50": {:.4},
{indent}  "valid_dist_evals_p99": {:.4},
{indent}  "bridge_dist_evals_mean": {:.4},
{indent}  "bridge_dist_evals_p50": {:.4},
{indent}  "bridge_dist_evals_p99": {:.4},
{indent}  "upper_layer_dist_evals_mean": {:.4},
{indent}  "visited_nodes_mean": {:.4},
{indent}  "visited_matching_nodes_mean": {:.4},
{indent}  "eta_graph_mean": {:.6},
{indent}  "eta_graph_pooled": {:.6}
{indent}}}"#,
            escape_json(&self.strategy),
            escape_json(&self.strategy_label),
            self.ef_search,
            self.overfetch_cap,
            self.num_queries,
            self.is_subquery_probe,
            self.reps,
            self.recall_at_k,
            self.fewer_than_k_count,
            self.fewer_than_k_rate,
            self.post_filter_cap_reached_rate,
            self.hop_budget_exhausted_rate,
            self.eval_limit_reached_rate,
            self.used_bruteforce_fallback_rate,
            self.summary.median_latency.mean_ms,
            self.summary.median_latency.p50_ms,
            self.summary.median_latency.p95_ms,
            self.summary.median_latency.p99_ms,
            self.summary.median_latency.qps,
            self.summary.mean_ms_min,
            self.summary.mean_ms_max,
            self.summary.mean_ms_spread,
            self.summary.p99_ms_min,
            self.summary.p99_ms_max,
            self.summary.p99_ms_spread,
            self.dist_evals_mean,
            self.dist_evals_p50,
            self.dist_evals_p99,
            self.valid_dist_evals_mean,
            self.valid_dist_evals_p50,
            self.valid_dist_evals_p99,
            self.bridge_dist_evals_mean,
            self.bridge_dist_evals_p50,
            self.bridge_dist_evals_p99,
            self.upper_layer_dist_evals_mean,
            self.visited_nodes_mean,
            self.visited_matching_nodes_mean,
            self.eta_graph_mean,
            self.eta_graph_pooled,
        )
    }
}

/// Computes `Recall@k` over filtered search outputs (`fewer_than_k` padded implicitly with misses).
#[must_use]
pub fn compute_filtered_recall_at_k(
    outputs: &[FilteredSearchOutput],
    gt_indices: &RowMatrix<i32>,
    k: usize,
) -> f64 {
    let take_q = outputs.len().min(gt_indices.rows());
    let effective_k = k.min(gt_indices.cols());
    if take_q == 0 || effective_k == 0 {
        return 1.0;
    }
    let mut total_hits = 0_usize;
    let mut gt_set = HashSet::with_capacity(effective_k * 2);
    let mut cand_seen = HashSet::with_capacity(effective_k * 2);
    for (qi, out) in outputs[..take_q].iter().enumerate() {
        gt_set.clear();
        cand_seen.clear();
        for &gt_id in &gt_indices.row(qi)[..effective_k] {
            gt_set.insert(gt_id as usize);
        }
        for &(_, cid) in out.results.iter().take(effective_k) {
            if cand_seen.insert(cid) && gt_set.contains(&cid) {
                total_hits += 1;
            }
        }
    }
    (total_hits as f64) / ((take_q * effective_k) as f64)
}

fn summarize_counter_distribution(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        mean,
        percentile_sorted(&sorted, 50.0),
        percentile_sorted(&sorted, 99.0),
    )
}

/// Executes 1 warmup + `reps` timed passes using zero-counter [`HnswIndex::search_knn_filtered_with_scratch`],
/// followed by 1 untimed pass using [`HnswIndex::search_knn_filtered_with_counters`].
#[allow(clippy::too_many_arguments)]
pub fn measure_arm(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    take_q: usize,
    is_subquery_probe: bool,
    k: usize,
    bitset: &DenseBitset,
    gt_indices: &RowMatrix<i32>,
    strategy: FilterStrategy,
    strategy_name: &str,
    strategy_label: &str,
    params: &FilteredSearchParams,
    reps: usize,
    scratch: &mut FilteredSearchScratch,
) -> Result<Phase5ArmRecord> {
    let take_q = take_q.min(queries.rows());
    let effective_reps = reps.max(1);
    let expected_match_k = k.min(bitset.count_ones());

    // 1 untimed warmup pass
    for qi in 0..take_q {
        let _ = index.search_knn_filtered_with_scratch(
            queries.row(qi),
            k,
            bitset,
            strategy,
            params,
            scratch,
        )?;
    }

    // Timed passes (`COLLECT_COUNTERS = false`)
    let mut rep_stats: Vec<LatencyStats> = Vec::with_capacity(effective_reps);
    let mut per_query_ms = Vec::with_capacity(take_q);
    for _ in 0..effective_reps {
        per_query_ms.clear();
        let batch_t0 = Instant::now();
        for qi in 0..take_q {
            let q_t0 = Instant::now();
            let out = index.search_knn_filtered_with_scratch(
                queries.row(qi),
                k,
                bitset,
                strategy,
                params,
                scratch,
            )?;
            std::hint::black_box(out.results.len());
            per_query_ms.push(q_t0.elapsed().as_secs_f64() * 1000.0);
        }
        let batch_s = batch_t0.elapsed().as_secs_f64();
        rep_stats.push(compute_latency_stats(&per_query_ms, batch_s));
    }
    let summary = summarize_repeated_latencies(&rep_stats);

    // 1 untimed accuracy + flags + counters pass (`COLLECT_COUNTERS = true`)
    let mut outputs = Vec::with_capacity(take_q);
    let mut counters_list: Vec<FilteredSearchCounters> = Vec::with_capacity(take_q);
    for qi in 0..take_q {
        let (out, cnt) = index.search_knn_filtered_with_counters(
            queries.row(qi),
            k,
            bitset,
            strategy,
            params,
            scratch,
        )?;
        outputs.push(out);
        counters_list.push(cnt);
    }

    let recall_at_k = compute_filtered_recall_at_k(&outputs, gt_indices, k);
    let mut fewer_than_k_count = 0_usize;
    let mut post_cap_count = 0_usize;
    let mut hop_ex_count = 0_usize;
    let mut eval_limit_count = 0_usize;
    let mut fallback_count = 0_usize;
    for out in &outputs {
        if out.results.len() < expected_match_k {
            fewer_than_k_count += 1;
        }
        if out.flags.post_filter_cap_reached {
            post_cap_count += 1;
        }
        if out.flags.hop_budget_exhausted {
            hop_ex_count += 1;
        }
        if out.flags.eval_limit_reached {
            eval_limit_count += 1;
        }
        if out.flags.used_bruteforce_fallback {
            fallback_count += 1;
        }
    }

    let denom_q = take_q.max(1) as f64;
    let totals: Vec<f64> = counters_list
        .iter()
        .map(|c| c.total_distance_evals as f64)
        .collect();
    let valids: Vec<f64> = counters_list
        .iter()
        .map(|c| c.valid_distance_evals as f64)
        .collect();
    let bridges: Vec<f64> = counters_list
        .iter()
        .map(|c| c.bridge_distance_evals as f64)
        .collect();
    let (dist_evals_mean, dist_evals_p50, dist_evals_p99) = summarize_counter_distribution(&totals);
    let (valid_dist_evals_mean, valid_dist_evals_p50, valid_dist_evals_p99) =
        summarize_counter_distribution(&valids);
    let (bridge_dist_evals_mean, bridge_dist_evals_p50, bridge_dist_evals_p99) =
        summarize_counter_distribution(&bridges);

    let upper_layer_dist_evals_mean = counters_list
        .iter()
        .map(|c| c.upper_layer_distance_evals as f64)
        .sum::<f64>()
        / denom_q;
    let sum_visited: u64 = counters_list.iter().map(|c| c.visited_nodes).sum();
    let sum_matching: u64 = counters_list.iter().map(|c| c.visited_matching_nodes).sum();
    let visited_nodes_mean = (sum_visited as f64) / denom_q;
    let visited_matching_nodes_mean = (sum_matching as f64) / denom_q;
    let eta_graph_mean = counters_list
        .iter()
        .map(FilteredSearchCounters::eta_graph)
        .sum::<f64>()
        / denom_q;
    let eta_graph_pooled = if sum_visited == 0 {
        0.0
    } else {
        (sum_matching as f64) / (sum_visited as f64)
    };

    Ok(Phase5ArmRecord {
        strategy: strategy_name.to_string(),
        strategy_label: strategy_label.to_string(),
        ef_search: params.ef_search,
        overfetch_cap: params.overfetch_cap,
        overfetch_factor: params.overfetch_factor,
        max_bridge_hops: params.max_bridge_hops,
        eval_limit: params.eval_limit,
        num_queries: take_q,
        is_subquery_probe,
        reps: effective_reps,
        recall_at_k,
        fewer_than_k_count,
        fewer_than_k_rate: (fewer_than_k_count as f64) / denom_q,
        post_filter_cap_reached_rate: (post_cap_count as f64) / denom_q,
        hop_budget_exhausted_rate: (hop_ex_count as f64) / denom_q,
        eval_limit_reached_rate: (eval_limit_count as f64) / denom_q,
        used_bruteforce_fallback_rate: (fallback_count as f64) / denom_q,
        summary,
        dist_evals_mean,
        dist_evals_p50,
        dist_evals_p99,
        valid_dist_evals_mean,
        valid_dist_evals_p50,
        valid_dist_evals_p99,
        bridge_dist_evals_mean,
        bridge_dist_evals_p50,
        bridge_dist_evals_p99,
        upper_layer_dist_evals_mean,
        visited_nodes_mean,
        visited_matching_nodes_mean,
        eta_graph_mean,
        eta_graph_pooled,
    })
}

/// Result of an opaque `LambdaPredicate` `eval_limit` sensitivity probe.
#[derive(Debug, Clone)]
pub struct OpaqueEvalProbeRecord {
    pub scale: usize,
    pub mode: MetadataMode,
    pub sigma: f64,
    pub strategy_label: String,
    pub ef_search: usize,
    pub eval_limit: Option<usize>,
    pub num_queries: usize,
    pub is_subquery_probe: bool,
    pub recall_at_k: f64,
    pub mean_ms: f64,
    pub p99_ms: f64,
    pub eval_count_mean: f64,
    pub eval_count_p95: f64,
    pub eval_count_p99: f64,
    pub eval_count_max: usize,
    pub eval_limit_reached_rate: f64,
    pub fewer_than_k_rate: f64,
}

impl OpaqueEvalProbeRecord {
    fn to_json_string(&self, indent: &str) -> String {
        let eval_limit_json = self
            .eval_limit
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        format!(
            r#"{indent}{{
{indent}  "scale": {},
{indent}  "mode": "{}",
{indent}  "sigma": {:.6},
{indent}  "strategy_label": "{}",
{indent}  "ef_search": {},
{indent}  "eval_limit": {eval_limit_json},
{indent}  "num_queries": {},
{indent}  "is_subquery_probe": {},
{indent}  "recall_at_k": {:.6},
{indent}  "mean_ms": {:.6},
{indent}  "p99_ms": {:.6},
{indent}  "eval_count_mean": {:.2},
{indent}  "eval_count_p95": {:.2},
{indent}  "eval_count_p99": {:.2},
{indent}  "eval_count_max": {},
{indent}  "eval_limit_reached_rate": {:.6},
{indent}  "fewer_than_k_rate": {:.6}
{indent}}}"#,
            self.scale,
            self.mode.as_str(),
            self.sigma,
            escape_json(&self.strategy_label),
            self.ef_search,
            self.num_queries,
            self.is_subquery_probe,
            self.recall_at_k,
            self.mean_ms,
            self.p99_ms,
            self.eval_count_mean,
            self.eval_count_p95,
            self.eval_count_p99,
            self.eval_count_max,
            self.eval_limit_reached_rate,
            self.fewer_than_k_rate,
        )
    }
}

/// Measures an opaque `LambdaPredicate` probe (no bitset fast-path) recording closure invocation stats.
#[allow(clippy::too_many_arguments)]
pub fn measure_opaque_eval_probe(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    take_q: usize,
    is_subquery_probe: bool,
    k: usize,
    scale: usize,
    mode: MetadataMode,
    sigma: f64,
    bitset: &DenseBitset,
    gt_indices: &RowMatrix<i32>,
    strategy: FilterStrategy,
    strategy_label: &str,
    params: &FilteredSearchParams,
    scratch: &mut FilteredSearchScratch,
) -> Result<OpaqueEvalProbeRecord> {
    let take_q = take_q.min(queries.rows());
    let expected_match_k = k.min(bitset.count_ones());
    let eval_calls = AtomicUsize::new(0);
    let lambda = LambdaPredicate::new(
        |id| {
            eval_calls.fetch_add(1, Ordering::Relaxed);
            bitset.get_bit(id as usize)
        },
        sigma,
    );

    if take_q > 0 {
        let _ = index.search_knn_filtered_with_scratch(
            queries.row(0),
            k,
            &lambda,
            strategy,
            params,
            scratch,
        )?;
    }

    let mut outputs = Vec::with_capacity(take_q);
    let mut latencies_ms = Vec::with_capacity(take_q);
    let mut per_query_evals = Vec::with_capacity(take_q);
    let mut limit_hits = 0_usize;
    let mut fewer_hits = 0_usize;

    let t_total = Instant::now();
    for qi in 0..take_q {
        eval_calls.store(0, Ordering::Relaxed);
        let q_t0 = Instant::now();
        let out = index.search_knn_filtered_with_scratch(
            queries.row(qi),
            k,
            &lambda,
            strategy,
            params,
            scratch,
        )?;
        let elapsed_ms = q_t0.elapsed().as_secs_f64() * 1000.0;
        let used = eval_calls.load(Ordering::Relaxed);
        if out.flags.eval_limit_reached {
            limit_hits += 1;
        }
        if out.results.len() < expected_match_k {
            fewer_hits += 1;
        }
        latencies_ms.push(elapsed_ms);
        per_query_evals.push(used as f64);
        outputs.push(out);
    }
    let stats = compute_latency_stats(&latencies_ms, t_total.elapsed().as_secs_f64());
    let recall_at_k = compute_filtered_recall_at_k(&outputs, gt_indices, k);

    let eval_count_mean = if per_query_evals.is_empty() {
        0.0
    } else {
        per_query_evals.iter().sum::<f64>() / (per_query_evals.len() as f64)
    };
    per_query_evals.sort_by(f64::total_cmp);
    let eval_count_p95 = percentile_sorted(&per_query_evals, 95.0);
    let eval_count_p99 = percentile_sorted(&per_query_evals, 99.0);
    let eval_count_max = per_query_evals.last().map_or(0_usize, |v| *v as usize);
    let denom = take_q.max(1) as f64;

    Ok(OpaqueEvalProbeRecord {
        scale,
        mode,
        sigma,
        strategy_label: strategy_label.to_string(),
        ef_search: params.ef_search,
        eval_limit: params.eval_limit,
        num_queries: take_q,
        is_subquery_probe,
        recall_at_k,
        mean_ms: stats.mean_ms,
        p99_ms: stats.p99_ms,
        eval_count_mean,
        eval_count_p95,
        eval_count_p99,
        eval_count_max,
        eval_limit_reached_rate: (limit_hits as f64) / denom,
        fewer_than_k_rate: (fewer_hits as f64) / denom,
    })
}

/// One cell report covering all benchmark arms at a `(scale, mode, sigma)` grid coordinate.
#[derive(Debug, Clone)]
pub struct Phase5CellRecord {
    pub report_filename: String,
    pub status: String,
    pub scale: usize,
    pub mode: MetadataMode,
    pub sigma: f64,
    pub k: usize,
    pub reps: usize,
    pub expected_popcount: usize,
    pub actual_popcount: usize,
    pub clusters_c: usize,
    pub s1_uncapped_evaluated: bool,
    pub predicate_cache_hit: bool,
    pub predicate_gen_time_s: f64,
    pub gt_cache_hit: bool,
    pub gt_gen_time_s: f64,
    pub arms: Vec<Phase5ArmRecord>,
}

/// Matched-recall crossover result (`DESIGN.md` §4.3) for one `(mode, target_recall)` series.
#[derive(Debug, Clone)]
pub struct CrossoverModeResult {
    pub mode: MetadataMode,
    pub target_recall: f64,
    pub outcome: String,
    pub bracket_lo_sigma: Option<f64>,
    pub bracket_hi_sigma: Option<f64>,
    pub crossover_sigma_loglog: Option<f64>,
    pub bracket_lo_s2_ef: Option<usize>,
    pub bracket_hi_s2_ef: Option<usize>,
    pub note: String,
}

impl CrossoverModeResult {
    fn to_json_string(&self, indent: &str) -> String {
        let opt_f64 = |v: Option<f64>| v.map_or_else(|| "null".to_string(), |x| format!("{x:.6}"));
        let opt_usize = |v: Option<usize>| v.map_or_else(|| "null".to_string(), |x| x.to_string());
        format!(
            r#"{indent}{{
{indent}  "mode": "{}",
{indent}  "target_recall": {:.2},
{indent}  "outcome": "{}",
{indent}  "bracket_lo_sigma": {},
{indent}  "bracket_hi_sigma": {},
{indent}  "crossover_sigma_loglog": {},
{indent}  "bracket_lo_s2_ef": {},
{indent}  "bracket_hi_s2_ef": {},
{indent}  "note": "{}"
{indent}}}"#,
            self.mode.as_str(),
            self.target_recall,
            escape_json(&self.outcome),
            opt_f64(self.bracket_lo_sigma),
            opt_f64(self.bracket_hi_sigma),
            opt_f64(self.crossover_sigma_loglog),
            opt_usize(self.bracket_lo_s2_ef),
            opt_usize(self.bracket_hi_s2_ef),
            escape_json(&self.note),
        )
    }
}

/// Extracts matched-recall crossover brackets and log-log interpolated $\sigma^*$ from measured cells.
#[must_use]
pub fn compute_crossover_analysis(
    cells: &[Phase5CellRecord],
    target_recall: f64,
) -> Vec<CrossoverModeResult> {
    let mut modes_present: Vec<MetadataMode> = Vec::new();
    for cell in cells {
        if !modes_present.contains(&cell.mode) {
            modes_present.push(cell.mode);
        }
    }
    modes_present.sort();

    let mut results = Vec::new();
    for mode in modes_present {
        let mut mode_cells: Vec<&Phase5CellRecord> =
            cells.iter().filter(|c| c.mode == mode).collect();
        mode_cells.sort_by(|a, b| a.sigma.total_cmp(&b.sigma));

        #[derive(Clone, Copy)]
        struct PointEval {
            sigma: f64,
            s3_ms: f64,
            s2_ef: Option<usize>,
            s2_ms: Option<f64>,
            s2_wins: bool,
        }

        let mut points = Vec::new();
        for c in mode_cells {
            let Some(s3) = c.arms.iter().find(|a| a.strategy == "S3_pre_filter") else {
                continue;
            };
            let s3_ms = s3.summary.median_latency.mean_ms;
            let mut s2_candidates: Vec<&Phase5ArmRecord> = c
                .arms
                .iter()
                .filter(|a| a.strategy == "S2_in_traversal" && a.recall_at_k >= target_recall)
                .collect();
            s2_candidates.sort_by_key(|a| a.ef_search);
            let matched = s2_candidates.first().copied();
            let s2_ef = matched.map(|a| a.ef_search);
            let s2_ms = matched.map(|a| a.summary.median_latency.mean_ms);
            let s2_wins = s2_ms.is_some_and(|ms| ms <= s3_ms);
            points.push(PointEval {
                sigma: c.sigma,
                s3_ms,
                s2_ef,
                s2_ms,
                s2_wins,
            });
        }

        let mut found: Option<CrossoverModeResult> = None;
        for w in points.windows(2) {
            let lo = w[0];
            let hi = w[1];
            if !lo.s2_wins && hi.s2_wins {
                let crossover_sigma_loglog = match (lo.s2_ms, hi.s2_ms) {
                    (Some(s2_lo), Some(s2_hi))
                        if s2_lo > 0.0 && s2_hi > 0.0 && lo.s3_ms > 0.0 && hi.s3_ms > 0.0 =>
                    {
                        let x0 = lo.sigma.ln();
                        let x1 = hi.sigma.ln();
                        let d0 = s2_lo.ln() - lo.s3_ms.ln();
                        let d1 = s2_hi.ln() - hi.s3_ms.ln();
                        let denom = d1 - d0;
                        if d0 >= 0.0 && d1 <= 0.0 && denom.abs() > 1e-12 {
                            let t = (-d0 / denom).clamp(0.0, 1.0);
                            Some((x0 + t * (x1 - x0)).exp())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                let note = if lo.s2_ef.is_none() {
                    "target_not_reached_at_lower_bracket".to_string()
                } else {
                    "log_log_bracket_interpolated".to_string()
                };
                found = Some(CrossoverModeResult {
                    mode,
                    target_recall,
                    outcome: "crossover_found".to_string(),
                    bracket_lo_sigma: Some(lo.sigma),
                    bracket_hi_sigma: Some(hi.sigma),
                    crossover_sigma_loglog,
                    bracket_lo_s2_ef: lo.s2_ef,
                    bracket_hi_s2_ef: hi.s2_ef,
                    note,
                });
                break;
            }
        }

        if let Some(res) = found {
            results.push(res);
        } else {
            let note = if points.is_empty() {
                "no_cells_with_s3_available".to_string()
            } else if points.iter().all(|p| !p.s2_wins) {
                "s3_wins_across_all_measured_selectivities".to_string()
            } else if points.iter().all(|p| p.s2_wins) {
                "s2_wins_across_all_measured_selectivities".to_string()
            } else {
                "no_monotone_s3_to_s2_crossover_in_range".to_string()
            };
            results.push(CrossoverModeResult {
                mode,
                target_recall,
                outcome: "no_crossover_in_range".to_string(),
                bracket_lo_sigma: None,
                bracket_hi_sigma: None,
                crossover_sigma_loglog: None,
                bracket_lo_s2_ef: None,
                bracket_hi_s2_ef: None,
                note,
            });
        }
    }
    results
}

/// Top-level summary returned by `run_phase5_matrix`.
#[derive(Debug, Clone)]
pub struct Phase5Summary {
    pub scale: usize,
    pub total_cells: usize,
    pub newly_measured_count: usize,
    pub resumed_skipped_count: usize,
    pub sensitivity_executed: bool,
    pub opaque_1m_spotcheck_executed: bool,
    pub output_dir: PathBuf,
    pub summary_path: PathBuf,
    pub cells: Vec<Phase5CellRecord>,
    pub crossovers_primary: Vec<CrossoverModeResult>,
    pub crossovers_secondary: Vec<CrossoverModeResult>,
}

/// Checks whether a previously generated per-cell JSON can be cleanly reused under `--resume`.
#[allow(clippy::too_many_arguments)]
pub fn can_resume_phase5_cell(
    cell_path: &Path,
    expected_commit: &str,
    current_git_dirty: bool,
    source_kind: &str,
    scale: usize,
    mode: MetadataMode,
    sigma: f64,
    k: usize,
    reps: usize,
) -> bool {
    if expected_commit == "unknown" || !cell_path.is_file() {
        return false;
    }
    let Ok(content) = fs::read_to_string(cell_path) else {
        return false;
    };
    if !current_git_dirty && content.contains("\"git_dirty\": true") {
        return false;
    }
    let expected_s1_uncapped = should_evaluate_s1_uncapped(sigma);
    content.contains("\"phase\": 5")
        && content.contains("\"measured_this_run\": true")
        && content.contains(&format!("\"git_commit\": \"{expected_commit}\""))
        && content.contains(&format!("\"dataset_source_kind\": \"{source_kind}\""))
        && content.contains(&format!("\"scale\": {scale}"))
        && content.contains(&format!("\"mode\": \"{}\"", mode.as_str()))
        && content.contains(&format!("\"sigma_slug\": \"{}\"", sigma_slug(sigma)))
        && content.contains(&format!("\"k\": {k}"))
        && content.contains(&format!("\"reps\": {reps}"))
        && content.contains(&format!("\"pred_gen_version\": {PRED_GEN_VERSION}"))
        && content.contains(&format!("\"gt_gen_version\": {GT_GEN_VERSION}"))
        && content.contains(&format!(
            "\"s1_uncapped_evaluated\": {expected_s1_uncapped}"
        ))
        && parse_phase5_cell_json(&content, &cell_report_filename(scale, mode, sigma)).is_ok()
}

/// Parses a `Phase5CellRecord` from a written per-cell JSON file (zero external crates).
pub fn parse_phase5_cell_json(content: &str, filename: &str) -> Result<Phase5CellRecord> {
    let parse_err = |field: &str| {
        VectorSearchError::InvalidArgument(format!(
            "Failed parsing '{field}' from Phase 5 cell JSON '{filename}'"
        ))
    };
    let scale =
        extract_json_u64_field(content, "scale").ok_or_else(|| parse_err("scale"))? as usize;
    let mode_raw = extract_json_string_field(content, "mode").ok_or_else(|| parse_err("mode"))?;
    let mode = MetadataMode::parse(&mode_raw)?;
    let sigma = extract_json_f64_field(content, "sigma").ok_or_else(|| parse_err("sigma"))?;
    let k = extract_json_u64_field(content, "k").ok_or_else(|| parse_err("k"))? as usize;
    let reps = extract_json_u64_field(content, "reps").ok_or_else(|| parse_err("reps"))? as usize;
    let expected_popcount = extract_json_u64_field(content, "expected_popcount")
        .ok_or_else(|| parse_err("expected_popcount"))? as usize;
    let actual_popcount = extract_json_u64_field(content, "actual_popcount")
        .ok_or_else(|| parse_err("actual_popcount"))? as usize;
    let clusters_c = extract_json_u64_field(content, "clusters_c")
        .ok_or_else(|| parse_err("clusters_c"))? as usize;
    let s1_uncapped_evaluated = extract_json_bool_field(content, "s1_uncapped_evaluated")
        .ok_or_else(|| parse_err("s1_uncapped_evaluated"))?;
    let predicate_cache_hit = extract_json_bool_field(content, "predicate_cache_hit")
        .ok_or_else(|| parse_err("predicate_cache_hit"))?;
    let predicate_gen_time_s = extract_json_f64_field(content, "predicate_gen_time_s")
        .ok_or_else(|| parse_err("predicate_gen_time_s"))?;
    let gt_cache_hit = extract_json_bool_field(content, "gt_cache_hit")
        .ok_or_else(|| parse_err("gt_cache_hit"))?;
    let gt_gen_time_s = extract_json_f64_field(content, "gt_gen_time_s")
        .ok_or_else(|| parse_err("gt_gen_time_s"))?;

    let arms_slice = extract_json_array_slice(content, "arms").ok_or_else(|| parse_err("arms"))?;
    let mut arms = Vec::new();
    for obj in split_top_level_json_objects(arms_slice) {
        arms.push(parse_arm_object(obj, filename)?);
    }

    Ok(Phase5CellRecord {
        report_filename: filename.to_string(),
        status: "resumed".to_string(),
        scale,
        mode,
        sigma,
        k,
        reps,
        expected_popcount,
        actual_popcount,
        clusters_c,
        s1_uncapped_evaluated,
        predicate_cache_hit,
        predicate_gen_time_s,
        gt_cache_hit,
        gt_gen_time_s,
        arms,
    })
}

fn parse_arm_object(obj: &str, filename: &str) -> Result<Phase5ArmRecord> {
    let parse_err = |field: &str| {
        VectorSearchError::InvalidArgument(format!(
            "Failed parsing arm field '{field}' in '{filename}'"
        ))
    };
    let num_queries = extract_json_u64_field(obj, "num_queries")
        .ok_or_else(|| parse_err("num_queries"))? as usize;
    let mean_ms = extract_json_f64_field(obj, "mean_ms").ok_or_else(|| parse_err("mean_ms"))?;
    let p50_ms = extract_json_f64_field(obj, "p50_ms").ok_or_else(|| parse_err("p50_ms"))?;
    let p95_ms = extract_json_f64_field(obj, "p95_ms").ok_or_else(|| parse_err("p95_ms"))?;
    let p99_ms = extract_json_f64_field(obj, "p99_ms").ok_or_else(|| parse_err("p99_ms"))?;
    let qps = extract_json_f64_field(obj, "qps").ok_or_else(|| parse_err("qps"))?;
    let summary = RepeatedSweepSummary {
        num_runs: extract_json_u64_field(obj, "reps").unwrap_or(1) as usize,
        median_latency: LatencyStats {
            num_queries,
            total_s: if qps > 0.0 {
                (num_queries as f64) / qps
            } else {
                0.0
            },
            qps,
            min_ms: 0.0,
            mean_ms,
            p50_ms,
            p95_ms,
            p99_ms,
            max_ms: 0.0,
        },
        mean_ms_min: extract_json_f64_field(obj, "mean_ms_min").unwrap_or(mean_ms),
        mean_ms_max: extract_json_f64_field(obj, "mean_ms_max").unwrap_or(mean_ms),
        mean_ms_spread: extract_json_f64_field(obj, "mean_ms_spread").unwrap_or(0.0),
        p99_ms_min: extract_json_f64_field(obj, "p99_ms_min").unwrap_or(p99_ms),
        p99_ms_max: extract_json_f64_field(obj, "p99_ms_max").unwrap_or(p99_ms),
        p99_ms_spread: extract_json_f64_field(obj, "p99_ms_spread").unwrap_or(0.0),
        run_mean_latencies_ms: Vec::new(),
        run_p99_latencies_ms: Vec::new(),
    };

    Ok(Phase5ArmRecord {
        strategy: extract_json_string_field(obj, "strategy")
            .ok_or_else(|| parse_err("strategy"))?,
        strategy_label: extract_json_string_field(obj, "strategy_label")
            .ok_or_else(|| parse_err("strategy_label"))?,
        ef_search: extract_json_u64_field(obj, "ef_search").ok_or_else(|| parse_err("ef_search"))?
            as usize,
        overfetch_cap: extract_json_u64_field(obj, "overfetch_cap")
            .ok_or_else(|| parse_err("overfetch_cap"))? as usize,
        overfetch_factor: extract_json_f64_field(obj, "overfetch_factor"),
        max_bridge_hops: extract_json_u64_field(obj, "max_bridge_hops").map(|v| v as usize),
        eval_limit: extract_json_u64_field(obj, "eval_limit").map(|v| v as usize),
        num_queries,
        is_subquery_probe: extract_json_bool_field(obj, "is_subquery_probe")
            .ok_or_else(|| parse_err("is_subquery_probe"))?,
        reps: extract_json_u64_field(obj, "reps").unwrap_or(1) as usize,
        recall_at_k: extract_json_f64_field(obj, "recall_at_k")
            .ok_or_else(|| parse_err("recall_at_k"))?,
        fewer_than_k_count: extract_json_u64_field(obj, "fewer_than_k_count").unwrap_or(0) as usize,
        fewer_than_k_rate: extract_json_f64_field(obj, "fewer_than_k_rate").unwrap_or(0.0),
        post_filter_cap_reached_rate: extract_json_f64_field(obj, "post_filter_cap_reached_rate")
            .unwrap_or(0.0),
        hop_budget_exhausted_rate: extract_json_f64_field(obj, "hop_budget_exhausted_rate")
            .unwrap_or(0.0),
        eval_limit_reached_rate: extract_json_f64_field(obj, "eval_limit_reached_rate")
            .unwrap_or(0.0),
        used_bruteforce_fallback_rate: extract_json_f64_field(obj, "used_bruteforce_fallback_rate")
            .unwrap_or(0.0),
        summary,
        dist_evals_mean: extract_json_f64_field(obj, "dist_evals_mean").unwrap_or(0.0),
        dist_evals_p50: extract_json_f64_field(obj, "dist_evals_p50").unwrap_or(0.0),
        dist_evals_p99: extract_json_f64_field(obj, "dist_evals_p99").unwrap_or(0.0),
        valid_dist_evals_mean: extract_json_f64_field(obj, "valid_dist_evals_mean").unwrap_or(0.0),
        valid_dist_evals_p50: extract_json_f64_field(obj, "valid_dist_evals_p50").unwrap_or(0.0),
        valid_dist_evals_p99: extract_json_f64_field(obj, "valid_dist_evals_p99").unwrap_or(0.0),
        bridge_dist_evals_mean: extract_json_f64_field(obj, "bridge_dist_evals_mean")
            .unwrap_or(0.0),
        bridge_dist_evals_p50: extract_json_f64_field(obj, "bridge_dist_evals_p50").unwrap_or(0.0),
        bridge_dist_evals_p99: extract_json_f64_field(obj, "bridge_dist_evals_p99").unwrap_or(0.0),
        upper_layer_dist_evals_mean: extract_json_f64_field(obj, "upper_layer_dist_evals_mean")
            .unwrap_or(0.0),
        visited_nodes_mean: extract_json_f64_field(obj, "visited_nodes_mean").unwrap_or(0.0),
        visited_matching_nodes_mean: extract_json_f64_field(obj, "visited_matching_nodes_mean")
            .unwrap_or(0.0),
        eta_graph_mean: extract_json_f64_field(obj, "eta_graph_mean").unwrap_or(0.0),
        eta_graph_pooled: extract_json_f64_field(obj, "eta_graph_pooled").unwrap_or(0.0),
    })
}

fn load_dataset_for_config(config: &Phase5MatrixConfig) -> Result<LoadedDataset> {
    if config.synthetic {
        let num_q = config.synthetic_queries.unwrap_or(100);
        let t0 = Instant::now();
        let (base, query) = generate_synthetic_sift(
            config.scale,
            num_q,
            config.synthetic_dim,
            64,
            config.attr_seed,
        )?;
        let load_time_s = t0.elapsed().as_secs_f64();
        return Ok(LoadedDataset {
            base,
            query,
            official_ground_truth: None,
            source: DatasetSource::Synthetic {
                seed: config.attr_seed,
            },
            load_time_s,
        });
    }

    let loaded = if config.scale <= 10_000 {
        load_or_download_sift_with_options(
            SiftPreset::SiftSmall,
            &config.data_dir,
            config.allow_download,
            config.attr_seed,
            Some(config.scale),
            config.synthetic_queries,
            false,
        )?
    } else {
        load_or_download_sift_with_options(
            SiftPreset::Sift1M,
            &config.data_dir,
            config.allow_download,
            config.attr_seed,
            Some(config.scale),
            config.synthetic_queries,
            false,
        )?
    };
    if !loaded.source.is_authentic_sift() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Authentic Texmex SIFT corpus is required for scale {} unless --synthetic is passed (loaded '{}'). Ensure SIFT files exist under '{}' or omit --offline.",
            config.scale,
            loaded.source,
            config.data_dir.display()
        )));
    }
    Ok(loaded)
}

struct PreparedCellData {
    mode: MetadataMode,
    sigma: f64,
    pred: PredicateCacheArtifact,
    gt: FilteredGtCacheArtifact,
}

#[allow(clippy::too_many_arguments)]
fn format_cell_json(
    cell: &Phase5CellRecord,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    loaded: &LoadedDataset,
    pred: &PredicateCacheArtifact,
    gt: &FilteredGtCacheArtifact,
    build_time_s: f64,
    build_qps: f64,
    graph_fingerprint: u64,
    graph_only_bpv: f64,
    peak_rss_mb: Option<f64>,
    hnsw_cfg: &HnswConfig,
    attr_seed: u64,
    clust_seed: u64,
) -> String {
    let mut hw_block = String::new();
    append_hardware_json(&mut hw_block, hw);

    let effective_cs = match cell.mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    let peak_rss_json = peak_rss_mb.map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));
    let keep_pruned_json = hnsw_cfg
        .keep_pruned_shrink
        .map_or_else(|| "null".to_string(), |v| v.to_string());
    let total_ram_json = hw
        .total_ram_gb
        .map_or_else(|| "null".to_string(), |v| format!("{v:.2}"));

    let arms_json = cell
        .arms
        .iter()
        .map(|a| a.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        r#"{{
  "phase": 5,
  "phase_name": "Phase 5 Filtered Search Benchmark",
  "benchmark": "phase5_cell",
  "task": "Milestone M3 Task T3.8 / T3.9",
  "generated_at": "{}",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "measured_this_run": true,
  "provenance": {{
    "os": "{}",
    "arch": "{}",
    "cpu_model": "{}",
    "logical_cores": {},
    "total_ram_gb": {total_ram_json},
    "power_mode": "{}",
    "processor_affinity_mask": "{}"
  }},
{hw_block}
  "dataset_source": "{}",
  "dataset_source_kind": "{}",
  "is_authentic_sift": {},
  "scale": {},
  "num_base": {},
  "num_query_total": {},
  "dim": {},
  "k": {},
  "reps": {},
  "metric": "l2_squared",
  "index_build": {{
    "m": {},
    "ef_construction": {},
    "seed": {},
    "keep_pruned_shrink": {keep_pruned_json},
    "build_time_s": {build_time_s:.6},
    "build_qps": {build_qps:.2},
    "graph_fingerprint": {graph_fingerprint},
    "graph_fingerprint_hex": "0x{graph_fingerprint:016x}",
    "graph_only_bytes_per_vector": {graph_only_bpv:.2},
    "peak_rss_mb": {peak_rss_json}
  }},
  "mode": "{}",
  "sigma": {:.6},
  "sigma_slug": "{}",
  "expected_popcount": {},
  "actual_popcount": {},
  "clusters_c": {},
  "attr_seed": {attr_seed},
  "clust_seed": {effective_cs},
  "pred_gen_version": {PRED_GEN_VERSION},
  "gt_gen_version": {GT_GEN_VERSION},
  "predicate_cache_path": {},
  "predicate_cache_hit": {},
  "predicate_gen_time_s": {:.6},
  "gt_ivecs_path": {},
  "gt_fvecs_path": {},
  "gt_cache_hit": {},
  "gt_gen_time_s": {:.6},
  "option_c_schedule": {{
    "tier": "Option C",
    "s1_uncapped_evaluated_sigma_min": {:.6},
    "s1_uncapped_evaluated": {}
  }},
  "arms": [
{arms_json}
  ]
}}
"#,
        escape_json(&current_utc_rfc3339()),
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        escape_json(&hw.os),
        escape_json(&hw.arch),
        escape_json(&hw.cpu_model),
        hw.logical_cores,
        escape_json(&prov.power_mode),
        escape_json(&prov.processor_affinity_mask),
        escape_json(&loaded.source.to_string()),
        escape_json(loaded.source.kind_tag()),
        loaded.source.is_authentic_sift(),
        cell.scale,
        loaded.base.rows(),
        loaded.query.rows(),
        loaded.base.cols(),
        cell.k,
        cell.reps,
        hnsw_cfg.m,
        hnsw_cfg.ef_construction,
        hnsw_cfg.seed,
        cell.mode.as_str(),
        cell.sigma,
        sigma_slug(cell.sigma),
        cell.expected_popcount,
        cell.actual_popcount,
        cell.clusters_c,
        opt_path_json(Some(&pred.cache_path)),
        cell.predicate_cache_hit,
        cell.predicate_gen_time_s,
        opt_path_json(Some(&gt.ivecs_path)),
        opt_path_json(Some(&gt.fvecs_path)),
        cell.gt_cache_hit,
        cell.gt_gen_time_s,
        S1_UNCAPPED_SIGMA_MIN,
        cell.s1_uncapped_evaluated,
    )
}

#[derive(Debug, Clone)]
struct SensitivitySweepItem {
    sweep_group: String,
    mode: MetadataMode,
    sigma: f64,
    arm: Phase5ArmRecord,
}

impl SensitivitySweepItem {
    fn to_json_string(&self, indent: &str) -> String {
        let arm_json = self.arm.to_json_string(&format!("{indent}    "));
        format!(
            r#"{indent}{{
{indent}  "sweep_group": "{}",
{indent}  "mode": "{}",
{indent}  "sigma": {:.6},
{indent}  "arm":
{arm_json}
{indent}}}"#,
            escape_json(&self.sweep_group),
            self.mode.as_str(),
            self.sigma,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sensitivity_suite(
    config: &Phase5MatrixConfig,
    loaded: &LoadedDataset,
    cache_dir: &Path,
    base_hash_hex: &str,
    query_hash_hex: &str,
    index: &HnswIndex<'_>,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    hnsw_cfg: &HnswConfig,
    graph_fingerprint: u64,
) -> Result<(Vec<SensitivitySweepItem>, Vec<OpaqueEvalProbeRecord>)> {
    let mut items = Vec::new();
    let mut opaque_probes = Vec::new();
    let num_base = loaded.base.rows();
    let sens_q = if config.scale <= 10_000 {
        loaded.query.rows()
    } else {
        500_usize.min(loaded.query.rows())
    };
    let is_probe = sens_q < loaded.query.rows();

    let max_ef = *SENSITIVITY_HMAX_EFS.iter().max().unwrap_or(&400);
    let mut scratch = FilteredSearchScratch::new(
        num_base,
        max_ef.max(S1_OVERFETCH_CAP.min(num_base)).max(config.k),
        S1_OVERFETCH_CAP.min(num_base).max(config.k),
    );
    let mut lazy_exact: Option<FilteredExactKnn> = None;

    // 1. Bridge-hop budget sensitivity: H_max in {1, 3, 10} * ef across sigma in {0.001, 0.01, 0.10}
    let active_sigmas_hmax: Vec<f64> = SENSITIVITY_HMAX_SIGMAS
        .iter()
        .copied()
        .filter(|&s| (s * num_base as f64).floor() as usize >= 1)
        .collect();
    for &mode in &config.modes {
        for &sigma in &active_sigmas_hmax {
            let cell_cache = prepare_single_cell_cache(
                cache_dir,
                &mut lazy_exact,
                loaded,
                base_hash_hex,
                query_hash_hex,
                mode,
                sigma,
                config.k,
                config.attr_seed,
                config.clust_seed,
                config.gt_threads,
            )?;
            for &ef in &SENSITIVITY_HMAX_EFS {
                for &mult in &SENSITIVITY_HMAX_MULTS {
                    let h_max = mult * ef;
                    let params = FilteredSearchParams {
                        ef_search: ef,
                        max_bridge_hops: Some(h_max),
                        overfetch_cap: S1_OVERFETCH_CAP.min(num_base),
                        allow_bruteforce_fallback: false,
                        ..Default::default()
                    };
                    let label = format!("S2_ef{ef}_hmax{mult}x");
                    let arm = measure_arm(
                        index,
                        &loaded.query,
                        sens_q,
                        is_probe,
                        config.k,
                        &cell_cache.pred.bitset,
                        &cell_cache.gt.indices,
                        FilterStrategy::InTraversal,
                        "S2_in_traversal",
                        &label,
                        &params,
                        config.reps,
                        &mut scratch,
                    )?;
                    items.push(SensitivitySweepItem {
                        sweep_group: "h_max_sweep".to_string(),
                        mode,
                        sigma,
                        arm,
                    });
                }
            }
        }
    }

    // 2. Over-fetch factor gamma sensitivity on Strategy 1 (`ef=100`, `sigma in {0.01, 0.10}`)
    let active_sigmas_gamma: Vec<f64> = SENSITIVITY_GAMMA_SIGMAS
        .iter()
        .copied()
        .filter(|&s| (s * num_base as f64).floor() as usize >= 1)
        .collect();
    for &mode in &config.modes {
        for &sigma in &active_sigmas_gamma {
            let cell_cache = prepare_single_cell_cache(
                cache_dir,
                &mut lazy_exact,
                loaded,
                base_hash_hex,
                query_hash_hex,
                mode,
                sigma,
                config.k,
                config.attr_seed,
                config.clust_seed,
                config.gt_threads,
            )?;
            for &gamma in &SENSITIVITY_GAMMA_VALUES {
                let params = FilteredSearchParams {
                    ef_search: 100,
                    overfetch_factor: gamma,
                    overfetch_cap: S1_OVERFETCH_CAP.min(num_base),
                    allow_bruteforce_fallback: false,
                    ..Default::default()
                };
                let label = match gamma {
                    None => "S1_cap4096_ef100_gamma_formula".to_string(),
                    Some(g) => format!("S1_cap4096_ef100_gamma{g:.1}"),
                };
                let arm = measure_arm(
                    index,
                    &loaded.query,
                    sens_q,
                    is_probe,
                    config.k,
                    &cell_cache.pred.bitset,
                    &cell_cache.gt.indices,
                    FilterStrategy::PostFilterOverfetch,
                    "S1_post_capped",
                    &label,
                    &params,
                    config.reps,
                    &mut scratch,
                )?;
                items.push(SensitivitySweepItem {
                    sweep_group: "overfetch_factor_sweep".to_string(),
                    mode,
                    sigma,
                    arm,
                });
            }
        }
    }
    drop(lazy_exact);

    // 3. Opaque closure `eval_limit` sensitivity (`None`, `16_384`, `10_000`)
    let eval_q = if config.scale <= 10_000 {
        loaded.query.rows()
    } else if config.scale >= 1_000_000 {
        200_usize.min(loaded.query.rows())
    } else {
        300_usize.min(loaded.query.rows())
    };
    let opaque_specs: Vec<(MetadataMode, f64)> = [
        (MetadataMode::Independent, 0.10_f64),
        (MetadataMode::Correlated, 0.01_f64),
        (MetadataMode::Correlated, 0.10_f64),
    ]
    .into_iter()
    .filter(|(m, _)| config.modes.contains(m))
    .collect();
    let sens_probes = run_opaque_eval_probe_grid(
        config,
        loaded,
        cache_dir,
        base_hash_hex,
        query_hash_hex,
        index,
        &opaque_specs,
        eval_q,
        &mut scratch,
    )?;
    opaque_probes.extend(sens_probes);

    let spotcheck_note = if config.include_1m_opaque_spotcheck && config.scale < 1_000_000 {
        // Explicit 1M spot check requested alongside smaller-scale sensitivity run
        run_1m_opaque_spotcheck(config)?
    } else {
        Vec::new()
    };
    opaque_probes.extend(spotcheck_note);

    let sens_path = config
        .output_dir
        .join(format!("sensitivity_n{}.json", config.scale));
    write_sensitivity_json(
        &sens_path,
        config.scale,
        prov,
        hw,
        loaded,
        hnsw_cfg,
        graph_fingerprint,
        &items,
        &opaque_probes,
    )?;

    Ok((items, opaque_probes))
}

#[allow(clippy::too_many_arguments)]
fn run_opaque_eval_probe_grid(
    config: &Phase5MatrixConfig,
    loaded: &LoadedDataset,
    cache_dir: &Path,
    base_hash_hex: &str,
    query_hash_hex: &str,
    index: &HnswIndex<'_>,
    specs: &[(MetadataMode, f64)],
    eval_q: usize,
    scratch: &mut FilteredSearchScratch,
) -> Result<Vec<OpaqueEvalProbeRecord>> {
    let num_base = loaded.base.rows();
    let eval_is_probe = eval_q < loaded.query.rows();
    let mut prepared = Vec::with_capacity(specs.len());
    {
        let mut lazy_exact: Option<FilteredExactKnn> = None;
        for &(mode, sigma) in specs {
            prepared.push(prepare_single_cell_cache(
                cache_dir,
                &mut lazy_exact,
                loaded,
                base_hash_hex,
                query_hash_hex,
                mode,
                sigma,
                config.k,
                config.attr_seed,
                config.clust_seed,
                config.gt_threads,
            )?);
        }
    }

    let mut probes = Vec::new();
    for cell_data in &prepared {
        for &eval_limit in &SENSITIVITY_EVAL_LIMITS {
            for (strategy, label) in [
                (FilterStrategy::InTraversal, "S2_ef100"),
                (FilterStrategy::PostFilterOverfetch, "S1_cap4096_ef100"),
            ] {
                let params = FilteredSearchParams {
                    ef_search: 100,
                    overfetch_cap: S1_OVERFETCH_CAP.min(num_base),
                    eval_limit,
                    allow_bruteforce_fallback: false,
                    ..Default::default()
                };
                probes.push(measure_opaque_eval_probe(
                    index,
                    &loaded.query,
                    eval_q,
                    eval_is_probe,
                    config.k,
                    num_base,
                    cell_data.mode,
                    cell_data.sigma,
                    &cell_data.pred.bitset,
                    &cell_data.gt.indices,
                    strategy,
                    label,
                    &params,
                    scratch,
                )?);
            }
        }
    }
    Ok(probes)
}

#[allow(clippy::too_many_arguments)]
fn prepare_single_cell_cache(
    cache_dir: &Path,
    lazy_exact: &mut Option<FilteredExactKnn>,
    loaded: &LoadedDataset,
    base_hash_hex: &str,
    query_hash_hex: &str,
    mode: MetadataMode,
    sigma: f64,
    k: usize,
    attr_seed: u64,
    clust_seed: u64,
    gt_threads: usize,
) -> Result<PreparedCellData> {
    let pred = load_or_generate_predicate_cache(
        cache_dir,
        &loaded.base,
        base_hash_hex,
        mode,
        sigma,
        attr_seed,
        clust_seed,
    )?;
    let num_base = loaded.base.rows();
    let num_query_total = loaded.query.rows();
    let effective_cs = match mode {
        MetadataMode::Independent => 0_u64,
        MetadataMode::Correlated => clust_seed,
    };
    let gt_stem = format!(
        "gt_n{num_base}_{}_{}_q{num_query_total}_k{k}_as{attr_seed}_cs{effective_cs}_pv{PRED_GEN_VERSION}_gv{GT_GEN_VERSION}",
        mode.as_str(),
        sigma_slug(sigma),
    );
    let maybe_cache_miss = !cache_dir.join(format!("{gt_stem}.ivecs")).is_file()
        || !cache_dir.join(format!("{gt_stem}.fvecs")).is_file();
    if maybe_cache_miss && lazy_exact.is_none() {
        *lazy_exact = Some(FilteredExactKnn::new(
            Metric::L2Squared,
            loaded.base.clone(),
        )?);
    }
    let gt = load_or_compute_filtered_gt_cache(
        cache_dir,
        lazy_exact.as_ref(),
        &loaded.base,
        &loaded.query,
        base_hash_hex,
        query_hash_hex,
        &pred,
        mode,
        sigma,
        k,
        attr_seed,
        clust_seed,
        gt_threads,
    )?;
    Ok(PreparedCellData {
        mode,
        sigma,
        pred,
        gt,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_sensitivity_json(
    path: &Path,
    scale: usize,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    loaded: &LoadedDataset,
    hnsw_cfg: &HnswConfig,
    graph_fingerprint: u64,
    items: &[SensitivitySweepItem],
    opaque_probes: &[OpaqueEvalProbeRecord],
) -> Result<()> {
    let mut hw_block = String::new();
    append_hardware_json(&mut hw_block, hw);

    let items_json = items
        .iter()
        .map(|it| it.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");
    let probes_json = opaque_probes
        .iter()
        .map(|p| p.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");

    let json = format!(
        r#"{{
  "phase": 5,
  "benchmark": "phase5_sensitivity",
  "task": "Milestone M3 Task T3.8",
  "generated_at": "{}",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "measured_this_run": true,
  "default_opaque_eval_limit": {DEFAULT_OPAQUE_EVAL_LIMIT},
  "default_opaque_eval_limit_origin": "100k pilot evaluation-count distribution (4x K_max=4096)",
  "dataset_source": "{}",
  "is_authentic_sift": {},
  "scale": {scale},
  "index_build": {{
    "m": {},
    "ef_construction": {},
    "seed": {},
    "graph_fingerprint": {graph_fingerprint},
    "graph_fingerprint_hex": "0x{graph_fingerprint:016x}"
  }},
{hw_block}
  "sensitivity_arms": [
{items_json}
  ],
  "opaque_eval_probes": [
{probes_json}
  ]
}}
"#,
        escape_json(&current_utc_rfc3339()),
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        escape_json(&loaded.source.to_string()),
        loaded.source.is_authentic_sift(),
        hnsw_cfg.m,
        hnsw_cfg.ef_construction,
        hnsw_cfg.seed,
    );
    fs::write(path, json)?;
    Ok(())
}

/// Runs the Gate G4.5 Decision 2 **1M opaque-closure spot check** (`S2 ef=100` & `S1_cap ef=100`
/// at `eval_limit in {None, 16_384, 10_000}`), and writes `opaque_spotcheck_1m.json` if `opaque_1m_spotcheck_only`.
pub fn run_1m_opaque_spotcheck(config: &Phase5MatrixConfig) -> Result<Vec<OpaqueEvalProbeRecord>> {
    let mut cfg_1m = config.clone();
    if cfg_1m.scale < 1_000_000 && !cfg_1m.synthetic {
        cfg_1m.scale = 1_000_000;
    }
    let loaded = load_dataset_for_config(&cfg_1m)?;
    let num_base = loaded.base.rows();
    let tag = dataset_cache_tag(
        &loaded.source,
        num_base,
        loaded.base.cols(),
        cfg_1m.attr_seed,
    );
    let cache_dir = cfg_1m.data_dir.join("phase5").join(tag);
    let base_hash_hex = format!("0x{:016x}", compute_f32_matrix_hash(&loaded.base));
    let query_hash_hex = format!("0x{:016x}", compute_f32_matrix_hash(&loaded.query));

    let spot_specs: Vec<(MetadataMode, f64)> = [
        (MetadataMode::Independent, 0.10_f64),
        (MetadataMode::Correlated, 0.01_f64),
    ]
    .into_iter()
    .filter(|&(_, s)| (s * num_base as f64).floor() as usize >= 1)
    .collect();

    let hnsw_cfg = HnswConfig::new(HNSW_M, HNSW_EF_CONSTRUCTION);
    let index = HnswIndex::build(&loaded.base, Metric::L2Squared, hnsw_cfg)?;
    let spot_q = if num_base <= 10_000 {
        loaded.query.rows()
    } else {
        200_usize.min(loaded.query.rows())
    };
    let mut scratch = FilteredSearchScratch::new(
        num_base,
        S1_OVERFETCH_CAP.min(num_base).max(100).max(cfg_1m.k),
        S1_OVERFETCH_CAP.min(num_base).max(cfg_1m.k),
    );
    let probes = run_opaque_eval_probe_grid(
        &cfg_1m,
        &loaded,
        &cache_dir,
        &base_hash_hex,
        &query_hash_hex,
        &index,
        &spot_specs,
        spot_q,
        &mut scratch,
    )?;

    if cfg_1m.opaque_1m_spotcheck_only {
        fs::create_dir_all(&cfg_1m.output_dir)?;
        let hw = detect_hardware();
        let prov = detect_provenance();
        let mut hw_block = String::new();
        append_hardware_json(&mut hw_block, &hw);
        let probes_json = probes
            .iter()
            .map(|p| p.to_json_string("    "))
            .collect::<Vec<_>>()
            .join(",\n");
        let fp = index.graph_fingerprint();
        let json = format!(
            r#"{{
  "phase": 5,
  "benchmark": "phase5_1m_opaque_spotcheck",
  "task": "Milestone M3 Task T3.8 (Gate G4.5 Decision 2)",
  "generated_at": "{}",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "measured_this_run": true,
  "default_opaque_eval_limit": {DEFAULT_OPAQUE_EVAL_LIMIT},
  "dataset_source": "{}",
  "is_authentic_sift": {},
  "scale": {num_base},
  "graph_fingerprint": {fp},
  "graph_fingerprint_hex": "0x{fp:016x}",
{hw_block}
  "opaque_eval_probes": [
{probes_json}
  ]
}}
"#,
            escape_json(&current_utc_rfc3339()),
            escape_json(&prov.git_commit),
            prov.git_dirty,
            escape_json(&prov.rustc_version),
            escape_json(&loaded.source.to_string()),
            loaded.source.is_authentic_sift(),
        );
        fs::write(cfg_1m.output_dir.join("opaque_spotcheck_1m.json"), json)?;
    }

    Ok(probes)
}

#[allow(clippy::too_many_arguments)]
fn format_summary_json(
    scale: usize,
    total_cells: usize,
    newly_measured_count: usize,
    resumed_skipped_count: usize,
    sensitivity_executed: bool,
    opaque_1m_spotcheck_executed: bool,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
    dataset_source: &str,
    is_authentic_sift: bool,
    cells: &[Phase5CellRecord],
    crossovers_primary: &[CrossoverModeResult],
    crossovers_secondary: &[CrossoverModeResult],
) -> String {
    let mut hw_block = String::new();
    append_hardware_json(&mut hw_block, hw);

    let cells_json = cells
        .iter()
        .map(|c| {
            format!(
                r#"    {{
      "report_filename": "{}",
      "status": "{}",
      "mode": "{}",
      "sigma": {:.6},
      "s1_uncapped_evaluated": {},
      "predicate_cache_hit": {},
      "gt_cache_hit": {},
      "num_arms": {}
    }}"#,
                escape_json(&c.report_filename),
                escape_json(&c.status),
                c.mode.as_str(),
                c.sigma,
                c.s1_uncapped_evaluated,
                c.predicate_cache_hit,
                c.gt_cache_hit,
                c.arms.len(),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let primary_json = crossovers_primary
        .iter()
        .map(|c| c.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");
    let secondary_json = crossovers_secondary
        .iter()
        .map(|c| c.to_json_string("    "))
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        r#"{{
  "phase": 5,
  "phase_name": "Phase 5 Filtered Search Summary",
  "task": "Milestone M3 Task T3.8 / T3.9",
  "generated_at": "{}",
  "git_commit": "{}",
  "git_dirty": {},
  "rustc_version": "{}",
  "dataset_source": "{}",
  "is_authentic_sift": {},
  "scale": {scale},
  "total_cells": {total_cells},
  "newly_measured_count": {newly_measured_count},
  "resumed_skipped_count": {resumed_skipped_count},
  "sensitivity_executed": {sensitivity_executed},
  "opaque_1m_spotcheck_executed": {opaque_1m_spotcheck_executed},
  "option_c_schedule": {{
    "tier": "Option C",
    "default_opaque_eval_limit": {DEFAULT_OPAQUE_EVAL_LIMIT},
    "s1_uncapped_evaluated_sigma_min": {S1_UNCAPPED_SIGMA_MIN:.6}
  }},
{hw_block}
  "cells": [
{cells_json}
  ],
  "crossovers_primary_0_95": [
{primary_json}
  ],
  "crossovers_secondary_0_99": [
{secondary_json}
  ]
}}
"#,
        escape_json(&current_utc_rfc3339()),
        escape_json(&prov.git_commit),
        prov.git_dirty,
        escape_json(&prov.rustc_version),
        escape_json(dataset_source),
        is_authentic_sift,
    )
}

fn collect_all_scale_cells_in_dir(output_dir: &Path, scale: usize) -> Vec<Phase5CellRecord> {
    let prefix = format!("cell_n{scale}_");
    let mut records = Vec::new();
    let Ok(entries) = fs::read_dir(output_dir) else {
        return records;
    };
    let mut files: Vec<(String, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_string_lossy().to_string();
            if p.is_file() && name.starts_with(&prefix) && name.ends_with(".json") {
                Some((name, p))
            } else {
                None
            }
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, path) in files {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(parsed) = parse_phase5_cell_json(&content, &name) {
                records.push(parsed);
            }
        }
    }
    records
}

/// Executes the resumable Phase 5 filtered benchmark matrix, sensitivity suite, and summary synthesis.
pub fn run_phase5_matrix(config: &Phase5MatrixConfig) -> Result<Phase5Summary> {
    if config.modes.is_empty() {
        return Err(VectorSearchError::InvalidArgument(
            "At least one MetadataMode must be configured".to_string(),
        ));
    }
    if config.sigmas.is_empty() && !config.sensitivity_only && !config.opaque_1m_spotcheck_only {
        return Err(VectorSearchError::InvalidArgument(
            "At least one selectivity sigma must be configured".to_string(),
        ));
    }
    if config.k == 0 || config.reps == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k and reps must both be >= 1".to_string(),
        ));
    }

    if config.opaque_1m_spotcheck_only {
        let _ = run_1m_opaque_spotcheck(config)?;
        return Ok(Phase5Summary {
            scale: if config.scale < 1_000_000 && !config.synthetic {
                1_000_000
            } else {
                config.scale
            },
            total_cells: 0,
            newly_measured_count: 0,
            resumed_skipped_count: 0,
            sensitivity_executed: false,
            opaque_1m_spotcheck_executed: true,
            output_dir: config.output_dir.clone(),
            summary_path: config.output_dir.join("opaque_spotcheck_1m.json"),
            cells: Vec::new(),
            crossovers_primary: Vec::new(),
            crossovers_secondary: Vec::new(),
        });
    }

    fs::create_dir_all(&config.output_dir)?;
    let prov = detect_provenance();
    let hw = detect_hardware();

    let expected_source_kind = if config.synthetic {
        "synthetic"
    } else if config.scale <= 10_000 {
        "authentic_siftsmall"
    } else {
        "authentic_sift1m"
    };

    // Check which cells can already be cleanly resumed before loading matrices/building HNSW.
    let mut cell_records: Vec<Phase5CellRecord> = Vec::new();
    let mut pending_coords: Vec<(MetadataMode, f64)> = Vec::new();
    let mut resumed_skipped_count = 0_usize;

    if !config.sensitivity_only {
        for &mode in &config.modes {
            for &sigma in &config.sigmas {
                let filename = cell_report_filename(config.scale, mode, sigma);
                let cell_path = config.output_dir.join(&filename);
                if config.resume
                    && can_resume_phase5_cell(
                        &cell_path,
                        &prov.git_commit,
                        prov.git_dirty,
                        expected_source_kind,
                        config.scale,
                        mode,
                        sigma,
                        config.k,
                        config.reps,
                    )
                {
                    let content = fs::read_to_string(&cell_path)?;
                    let parsed = parse_phase5_cell_json(&content, &filename)?;
                    cell_records.push(parsed);
                    resumed_skipped_count += 1;
                } else {
                    pending_coords.push((mode, sigma));
                }
            }
        }
    }

    let need_sensitivity = config.include_sensitivity || config.sensitivity_only;
    if pending_coords.is_empty() && !need_sensitivity {
        let all_cells = collect_all_scale_cells_in_dir(&config.output_dir, config.scale);
        let crossovers_primary = compute_crossover_analysis(&all_cells, RECALL_TARGET_PRIMARY);
        let crossovers_secondary = compute_crossover_analysis(&all_cells, RECALL_TARGET_SECONDARY);
        let summary_path = config
            .output_dir
            .join(format!("summary_n{}.json", config.scale));
        let ds_label = if config.synthetic {
            format!("Synthetic SIFT ({})", config.scale)
        } else if config.scale <= 10_000 {
            format!("siftsmall ({})", config.scale)
        } else {
            format!("SIFT1M prefix ({})", config.scale)
        };
        let summary_json = format_summary_json(
            config.scale,
            cell_records.len(),
            0,
            resumed_skipped_count,
            false,
            false,
            &prov,
            &hw,
            &ds_label,
            !config.synthetic,
            &cell_records,
            &crossovers_primary,
            &crossovers_secondary,
        );
        fs::write(&summary_path, summary_json)?;
        return Ok(Phase5Summary {
            scale: config.scale,
            total_cells: cell_records.len(),
            newly_measured_count: 0,
            resumed_skipped_count,
            sensitivity_executed: false,
            opaque_1m_spotcheck_executed: false,
            output_dir: config.output_dir.clone(),
            summary_path,
            cells: cell_records,
            crossovers_primary,
            crossovers_secondary,
        });
    }

    let loaded = load_dataset_for_config(config)?;
    let num_base = loaded.base.rows();
    let num_query_total = loaded.query.rows();
    let tag = dataset_cache_tag(
        &loaded.source,
        num_base,
        loaded.base.cols(),
        config.attr_seed,
    );
    let cache_dir = config.data_dir.join("phase5").join(tag);
    let base_hash_hex = format!("0x{:016x}", compute_f32_matrix_hash(&loaded.base));
    let query_hash_hex = format!("0x{:016x}", compute_f32_matrix_hash(&loaded.query));

    // Stage 1: Prepare & verify predicate + GT disk caches BEFORE building HnswIndex so any
    // temporary FilteredExactKnn allocation is freed prior to HNSW graph residency.
    let mut prepared: Vec<PreparedCellData> = Vec::with_capacity(pending_coords.len());
    {
        let mut lazy_exact: Option<FilteredExactKnn> = None;
        for &(mode, sigma) in &pending_coords {
            prepared.push(prepare_single_cell_cache(
                &cache_dir,
                &mut lazy_exact,
                &loaded,
                &base_hash_hex,
                &query_hash_hex,
                mode,
                sigma,
                config.k,
                config.attr_seed,
                config.clust_seed,
                config.gt_threads,
            )?);
        }
    }

    // Stage 2: Build canonical reference HNSW index once (`M=16, efConstruction=100, seed=42`).
    let hnsw_cfg = HnswConfig::new(HNSW_M, HNSW_EF_CONSTRUCTION);
    let build_t0 = Instant::now();
    let index = HnswIndex::build(&loaded.base, Metric::L2Squared, hnsw_cfg.clone())?;
    let build_time_s = build_t0.elapsed().as_secs_f64();
    let build_qps = if build_time_s > 0.0 {
        (num_base as f64) / build_time_s
    } else {
        0.0
    };
    let graph_fingerprint = index.graph_fingerprint();
    let graph_only_bpv = index.graph_only_bytes_per_vector();

    let max_ef_needed = S2_EF_LADDER
        .iter()
        .copied()
        .chain(S1_CAP_EF_HIGH_SIGMA.iter().copied())
        .max()
        .unwrap_or(600)
        .max(S1_OVERFETCH_CAP.min(num_base))
        .max(config.k);
    let mut main_scratch = FilteredSearchScratch::new(
        num_base,
        max_ef_needed,
        S1_OVERFETCH_CAP.min(num_base).max(config.k),
    );
    let mut uncap_scratch: Option<FilteredSearchScratch> = None;

    let mut newly_measured_count = 0_usize;
    for cell_data in &prepared {
        let mode = cell_data.mode;
        let sigma = cell_data.sigma;
        let mut arms = Vec::new();

        // Arm A: Strategy 3 pre-filter brute force
        let (s3_q, s3_probe) = option_c_s3_query_count(num_base, sigma, num_query_total);
        let s3_params = FilteredSearchParams {
            ef_search: 100,
            overfetch_cap: S1_OVERFETCH_CAP.min(num_base),
            allow_bruteforce_fallback: false,
            ..Default::default()
        };
        arms.push(measure_arm(
            &index,
            &loaded.query,
            s3_q,
            s3_probe,
            config.k,
            &cell_data.pred.bitset,
            &cell_data.gt.indices,
            FilterStrategy::PreFilterBruteForce,
            "S3_pre_filter",
            "S3_pre_filter",
            &s3_params,
            config.reps,
            &mut main_scratch,
        )?);

        // Arm B: Strategy 2 in-traversal sweep across efSearch in {20, 50, 100, 200, 400, 600}
        let (hnsw_q, hnsw_probe) = option_c_hnsw_query_count(num_base, sigma, num_query_total);
        for &ef in &S2_EF_LADDER {
            let s2_params = FilteredSearchParams {
                ef_search: ef,
                overfetch_cap: S1_OVERFETCH_CAP.min(num_base),
                allow_bruteforce_fallback: false,
                ..Default::default()
            };
            arms.push(measure_arm(
                &index,
                &loaded.query,
                hnsw_q,
                hnsw_probe,
                config.k,
                &cell_data.pred.bitset,
                &cell_data.gt.indices,
                FilterStrategy::InTraversal,
                "S2_in_traversal",
                &format!("S2_ef{ef}"),
                &s2_params,
                config.reps,
                &mut main_scratch,
            )?);
        }

        // Arm C: Strategy 1 post-filter capped (`K'_max = 4096`)
        let s1_efs: &[usize] = if sigma + 1e-12 >= S1_UNCAPPED_SIGMA_MIN {
            &S1_CAP_EF_HIGH_SIGMA
        } else {
            &S1_CAP_EF_LOW_SIGMA
        };
        for &ef in s1_efs {
            let s1_params = FilteredSearchParams {
                ef_search: ef,
                overfetch_cap: S1_OVERFETCH_CAP.min(num_base),
                allow_bruteforce_fallback: false,
                ..Default::default()
            };
            arms.push(measure_arm(
                &index,
                &loaded.query,
                hnsw_q,
                hnsw_probe,
                config.k,
                &cell_data.pred.bitset,
                &cell_data.gt.indices,
                FilterStrategy::PostFilterOverfetch,
                "S1_post_capped",
                &format!("S1_cap{}_ef{ef}", S1_OVERFETCH_CAP.min(num_base)),
                &s1_params,
                config.reps,
                &mut main_scratch,
            )?);
        }

        // Arm D: Strategy 1 post-filter uncapped (`K'_max = N`) — Option C scopes strictly to `sigma >= 0.05`
        let eval_s1_uncapped = should_evaluate_s1_uncapped(sigma);
        if eval_s1_uncapped {
            let scratch_uncap = uncap_scratch
                .get_or_insert_with(|| FilteredSearchScratch::new(num_base, num_base, num_base));
            let uncap_params = FilteredSearchParams {
                ef_search: 100,
                overfetch_cap: num_base,
                allow_bruteforce_fallback: false,
                ..Default::default()
            };
            arms.push(measure_arm(
                &index,
                &loaded.query,
                hnsw_q,
                hnsw_probe,
                config.k,
                &cell_data.pred.bitset,
                &cell_data.gt.indices,
                FilterStrategy::PostFilterOverfetch,
                "S1_post_uncapped",
                "S1_uncap_ef100",
                &uncap_params,
                config.reps,
                scratch_uncap,
            )?);
        }

        let filename = cell_report_filename(config.scale, mode, sigma);
        let cell_rec = Phase5CellRecord {
            report_filename: filename.clone(),
            status: "measured".to_string(),
            scale: config.scale,
            mode,
            sigma,
            k: config.k,
            reps: config.reps,
            expected_popcount: cell_data.pred.expected_popcount,
            actual_popcount: cell_data.pred.actual_popcount,
            clusters_c: cell_data.pred.clusters_c,
            s1_uncapped_evaluated: eval_s1_uncapped,
            predicate_cache_hit: cell_data.pred.cache_hit,
            predicate_gen_time_s: cell_data.pred.gen_time_s,
            gt_cache_hit: cell_data.gt.cache_hit,
            gt_gen_time_s: cell_data.gt.gen_time_s,
            arms,
        };

        let peak_rss_mb = get_peak_rss_mb();
        let cell_json = format_cell_json(
            &cell_rec,
            &prov,
            &hw,
            &loaded,
            &cell_data.pred,
            &cell_data.gt,
            build_time_s,
            build_qps,
            graph_fingerprint,
            graph_only_bpv,
            peak_rss_mb,
            &hnsw_cfg,
            config.attr_seed,
            config.clust_seed,
        );
        fs::write(config.output_dir.join(&filename), cell_json)?;
        newly_measured_count += 1;
        cell_records.push(cell_rec);
    }

    let mut opaque_1m_spotcheck_executed = false;
    if need_sensitivity {
        let (_, opaque_probes) = run_sensitivity_suite(
            config,
            &loaded,
            &cache_dir,
            &base_hash_hex,
            &query_hash_hex,
            &index,
            &prov,
            &hw,
            &hnsw_cfg,
            graph_fingerprint,
        )?;
        opaque_1m_spotcheck_executed = opaque_probes.iter().any(|p| p.scale >= 1_000_000);
    }

    let all_cells = collect_all_scale_cells_in_dir(&config.output_dir, config.scale);
    let crossovers_primary = compute_crossover_analysis(&all_cells, RECALL_TARGET_PRIMARY);
    let crossovers_secondary = compute_crossover_analysis(&all_cells, RECALL_TARGET_SECONDARY);
    let summary_path = config
        .output_dir
        .join(format!("summary_n{}.json", config.scale));
    let summary_json = format_summary_json(
        config.scale,
        cell_records.len(),
        newly_measured_count,
        resumed_skipped_count,
        need_sensitivity,
        opaque_1m_spotcheck_executed,
        &prov,
        &hw,
        &loaded.source.to_string(),
        loaded.source.is_authentic_sift(),
        &cell_records,
        &crossovers_primary,
        &crossovers_secondary,
    );
    fs::write(&summary_path, summary_json)?;

    Ok(Phase5Summary {
        scale: config.scale,
        total_cells: cell_records.len(),
        newly_measured_count,
        resumed_skipped_count,
        sensitivity_executed: need_sensitivity,
        opaque_1m_spotcheck_executed,
        output_dir: config.output_dir.clone(),
        summary_path,
        cells: cell_records,
        crossovers_primary,
        crossovers_secondary,
    })
}

// =============================================================================
// Lightweight JSON extraction helpers (std-only, matching phase4.rs convention)
// =============================================================================

fn find_key_value_start<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let pos = content.find(&needle)?;
    let rest = content[pos + needle.len()..].trim_start();
    Some(rest)
}

fn extract_json_string_field(content: &str, key: &str) -> Option<String> {
    let rest = find_key_value_start(content, key)?;
    let inner = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(esc) = chars.next() {
                match esc {
                    '"' | '\\' | '/' => out.push(esc),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    other => out.push(other),
                }
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_json_token<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let rest = find_key_value_start(content, key)?;
    let end = rest.find([',', '\n', '\r', '}', ']']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn extract_json_f64_field(content: &str, key: &str) -> Option<f64> {
    let tok = extract_json_token(content, key)?;
    if tok == "null" {
        return None;
    }
    tok.parse::<f64>().ok()
}

fn extract_json_u64_field(content: &str, key: &str) -> Option<u64> {
    let tok = extract_json_token(content, key)?;
    if tok == "null" {
        return None;
    }
    tok.parse::<u64>().ok()
}

fn extract_json_bool_field(content: &str, key: &str) -> Option<bool> {
    let tok = extract_json_token(content, key)?;
    match tok {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn extract_json_array_slice<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let rest = find_key_value_start(content, key)?;
    let start_idx = rest.find('[')?;
    let bytes = rest.as_bytes();
    let mut depth = 0_usize;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes[start_idx..].iter().enumerate() {
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
        if b == b'"' {
            in_str = true;
        } else if b == b'[' {
            depth += 1;
        } else if b == b']' {
            depth -= 1;
            if depth == 0 {
                return Some(&rest[start_idx + 1..start_idx + i]);
            }
        }
    }
    None
}

fn split_top_level_json_objects(array_inner: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let bytes = array_inner.as_bytes();
    let mut depth = 0_usize;
    let mut obj_start = None;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
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
        if b == b'"' {
            in_str = true;
        } else if b == b'{' {
            if depth == 0 {
                obj_start = Some(i);
            }
            depth += 1;
        } else if b == b'}' && depth > 0 {
            depth -= 1;
            if depth == 0 {
                if let Some(start) = obj_start.take() {
                    objects.push(&array_inner[start..=i]);
                }
            }
        }
    }
    objects
}
