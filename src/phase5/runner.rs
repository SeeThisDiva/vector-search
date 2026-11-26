//! Resumable Phase 5 Filtered Search Benchmark Matrix Runner (`T3.8 / T3.9`).

#![forbid(unsafe_code)]

use std::fs;
use std::time::Instant;

use crate::benchmark::detect_provenance;
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::FilteredExactKnn;
use crate::filter::FilteredSearchParams;
use crate::hardware::{detect_hardware, get_peak_rss_mb};
use crate::hnsw::{FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex};

use super::cache::{
    compute_f32_matrix_hash, dataset_cache_tag, load_dataset_for_config, prepare_single_cell_cache,
    PreparedCellData,
};
use super::crossover::{compute_crossover_analysis, Phase5CellRecord};
use super::metrics::measure_arm;
use super::report::{
    can_resume_phase5_cell, collect_all_scale_cells_in_dir, format_cell_json, format_summary_json,
    parse_phase5_cell_json, Phase5Summary,
};
use super::sensitivity::{run_1m_opaque_spotcheck, run_sensitivity_suite};
use super::{
    cell_report_filename, option_c_hnsw_query_count, option_c_s3_query_count,
    should_evaluate_s1_uncapped, MetadataMode, Phase5MatrixConfig, HNSW_EF_CONSTRUCTION, HNSW_M,
    RECALL_TARGET_PRIMARY, RECALL_TARGET_SECONDARY, S1_CAP_EF_HIGH_SIGMA, S1_CAP_EF_LOW_SIGMA,
    S1_OVERFETCH_CAP, S1_UNCAPPED_SIGMA_MIN, S2_EF_LADDER,
};

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

    let expected_dim = if config.synthetic {
        config.synthetic_dim
    } else {
        crate::dataset::SiftPreset::Sift1M.expected_dim()
    };
    let expected_num_query_total = if config.synthetic {
        config.synthetic_queries.unwrap_or(100)
    } else {
        let preset = if config.scale <= 10_000 {
            crate::dataset::SiftPreset::SiftSmall
        } else {
            crate::dataset::SiftPreset::Sift1M
        };
        config
            .synthetic_queries
            .map_or(preset.expected_num_query(), |q| {
                q.min(preset.expected_num_query())
            })
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
                        config.attr_seed,
                        config.clust_seed,
                        expected_dim,
                        expected_num_query_total,
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
