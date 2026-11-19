//! Versioned Predicate Bitset and Filtered Exact Ground Truth Disk Cache (`data/phase5/`).

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::benchmark::compute_query_results_hash;
use crate::dataset::{
    generate_mode_a_bitset, generate_mode_b_bitset_detailed, generate_synthetic_sift,
    load_or_download_sift_with_options, read_bitset, read_fvecs, read_ivecs, write_bitset,
    write_fvecs, write_ivecs, DatasetSource, LoadedDataset, SiftPreset,
};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::FilteredExactKnn;
use crate::filter::DenseBitset;
use crate::hardware::current_utc_rfc3339;
use crate::hnsw::{fnv1a_64_update_u64, FNV_OFFSET_BASIS_64};
use crate::json_util::{escape_json, extract_json_string_field, extract_json_u64_field};
use crate::matrix::RowMatrix;

use super::{sigma_slug, MetadataMode, Phase5MatrixConfig, GT_GEN_VERSION, PRED_GEN_VERSION};

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
    if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
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

pub(crate) struct PreparedCellData {
    pub(crate) mode: MetadataMode,
    pub(crate) sigma: f64,
    pub(crate) pred: PredicateCacheArtifact,
    pub(crate) gt: FilteredGtCacheArtifact,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_single_cell_cache(
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

pub(crate) fn load_dataset_for_config(config: &Phase5MatrixConfig) -> Result<LoadedDataset> {
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
