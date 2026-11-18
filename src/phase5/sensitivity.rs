//! Bridge-Hop (`H_max`), Over-Fetch (`overfetch_factor`), and Opaque Closure (`eval_limit`)
//! Sensitivity Suite (`T3.8`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

use crate::benchmark::{append_hardware_json, detect_provenance, ReportProvenance};
use crate::dataset::LoadedDataset;
use crate::distance::Metric;
use crate::error::Result;
use crate::exact_knn::FilteredExactKnn;
use crate::filter::{FilteredSearchParams, DEFAULT_OPAQUE_EVAL_LIMIT};
use crate::hardware::{current_utc_rfc3339, detect_hardware, HardwareInfo};
use crate::hnsw::{FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex};
use crate::json_util::escape_json;

use super::cache::{
    compute_f32_matrix_hash, dataset_cache_tag, load_dataset_for_config, prepare_single_cell_cache,
};
use super::metrics::{
    measure_arm, measure_opaque_eval_probe, OpaqueEvalProbeRecord, Phase5ArmRecord,
};
use super::{
    MetadataMode, Phase5MatrixConfig, HNSW_EF_CONSTRUCTION, HNSW_M, S1_OVERFETCH_CAP,
    SENSITIVITY_EVAL_LIMITS, SENSITIVITY_GAMMA_SIGMAS, SENSITIVITY_GAMMA_VALUES,
    SENSITIVITY_HMAX_EFS, SENSITIVITY_HMAX_MULTS, SENSITIVITY_HMAX_SIGMAS,
};

#[derive(Debug, Clone)]
pub(crate) struct SensitivitySweepItem {
    pub(crate) sweep_group: String,
    pub(crate) mode: MetadataMode,
    pub(crate) sigma: f64,
    pub(crate) arm: Phase5ArmRecord,
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
pub(crate) fn run_sensitivity_suite(
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
