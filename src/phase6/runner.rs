//! Phase 6 quantized and `f32` sweep runner and resumable matrix orchestrator.

use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::benchmark::{detect_provenance, ReportProvenance};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::hardware::{detect_hardware, get_peak_rss_mb, HardwareInfo};
use crate::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use crate::json_util::write_file_with_parents;
use crate::matrix::RowMatrix;
use crate::quantized::{
    results_to_knn_search_result, HnswPqIndex, HnswSq8Index, PqConfig, PqQueryState,
    QuantizedBuildTimings, QuantizedMemoryProfile, QuantizedSearchScratch, QuantizedStorageMode,
    Sq8KernelKind, Sq8QueryState, TimedQuantizedSearch, PQ_DEFAULT_MAX_LLOYD_ITERS,
    PQ_DEFAULT_MAX_TRAIN_VECTORS, PQ_K_CENTROIDS,
};
use crate::recall::{compute_recall_at_k, summarize_repeated_latencies, LatencyStats};

use super::config::{
    apply_affinity_pinning, arm_report_filename, resolve_arm_spec, Phase6ArmKind, Phase6ArmSpec,
    Phase6MatrixConfig, RerankDepthSpec, SnapshotKind,
};
use super::dataset::{prepare_dataset_and_gt, PreparedDatasetState};
use super::report::{
    can_resume_existing_arm_report, parse_resumed_arm_summary, serialize_arm_report_json,
    serialize_matrix_summary_json, ArmReportContext, Phase6ArmSummaryEntry, Phase6MatrixSummary,
    Phase6SweepPointResult,
};
use super::rss_profile::measure_isolated_query_rss;

pub(crate) trait QuantizedSweepTarget {
    type QueryState: Default;

    fn q_num_vectors(&self) -> usize;
    fn q_dim(&self) -> usize;
    fn q_graph_fingerprint(&self) -> u64;
    fn q_memory_profile(&self) -> QuantizedMemoryProfile;
    fn q_build_timings(&self) -> QuantizedBuildTimings;
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

macro_rules! impl_quantized_sweep_target {
    ($index_ty:ty, $state_ty:ty, $timings_expr:ident => $timings_conv:expr) => {
        impl QuantizedSweepTarget for $index_ty {
            type QueryState = $state_ty;

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

            fn q_build_timings(&self) -> QuantizedBuildTimings {
                let $timings_expr = self;
                $timings_conv
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
    };
}

impl_quantized_sweep_target!(HnswSq8Index, Sq8QueryState, s => s.build_timings());
impl_quantized_sweep_target!(HnswPqIndex, PqQueryState, s => s.quantized_build_timings());

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

/// Shared repeated-run sweep point measurement (1 untimed warmup + `reps` timed passes).
fn run_repeated_sweep_point<F>(
    ef_search: usize,
    effective_ef_search: usize,
    rerank_depth: RerankDepthSpec,
    reps: usize,
    gt_matrix: &RowMatrix<i32>,
    k: usize,
    mut run_once: F,
) -> Result<Phase6SweepPointResult>
where
    F: FnMut() -> Result<(LatencyStats, RowMatrix<i32>)>,
{
    // 1 untimed warmup pass
    let _ = run_once()?;

    let mut run_latencies = Vec::with_capacity(reps);
    let mut last_indices = None;
    for _ in 0..reps {
        let (lat, indices) = run_once()?;
        run_latencies.push(lat);
        last_indices = Some(indices);
    }

    let summary = summarize_repeated_latencies(&run_latencies);
    let recall_at_k = match last_indices {
        Some(ref ind) => compute_recall_at_k(ind, gt_matrix, k)?,
        None => 0.0,
    };

    Ok(Phase6SweepPointResult {
        ef_search,
        effective_ef_search,
        rerank_depth,
        recall_at_k,
        summary,
    })
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
            let pt = run_repeated_sweep_point(
                ef_search,
                effective_ef,
                rerank_depth,
                reps,
                gt_matrix,
                k,
                || {
                    let timed = index.q_search_batch_timed(
                        queries,
                        k,
                        ef_search,
                        rerank_arg,
                        &mut scratch,
                    )?;
                    let knn = results_to_knn_search_result(&timed.results)?;
                    Ok((to_latency_stats(&timed), knn.indices))
                },
            )?;
            points.push(pt);
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
        let pt = run_repeated_sweep_point(
            ef_search,
            effective_ef,
            RerankDepthSpec::None,
            reps,
            gt_matrix,
            k,
            || {
                let timed =
                    index.search_batch_timed_with_tracker(queries, k, ef_search, &mut tracker)?;
                Ok((timed.latency, timed.result.indices))
            },
        )?;
        points.push(pt);
    }

    Ok(points)
}

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
    write_file_with_parents(&summary.summary_path, summary_json)?;
    Ok(summary)
}

struct ArmExecutionOutcome {
    sweep: Vec<Phase6SweepPointResult>,
    timings: QuantizedBuildTimings,
    fp: u64,
    streaming_build: bool,
    pq_config_used: Option<PqConfig>,
    mem: QuantizedMemoryProfile,
    build_peak_rss_mb: Option<f64>,
    query_serving_peak_rss_mb: Option<f64>,
    query_rss_profiled_isolated: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_and_finalize_quantized_arm<T, FStream, FMem>(
    snapshot_kind: SnapshotKind,
    spec: &Phase6ArmSpec,
    config: &Phase6MatrixConfig,
    dataset: &mut PreparedDatasetState,
    rerank_ladder: &[RerankDepthSpec],
    pq_config_used: Option<PqConfig>,
    build_streaming: FStream,
    build_in_memory: FMem,
) -> Result<ArmExecutionOutcome>
where
    T: QuantizedSweepTarget,
    FStream: FnOnce(&Path, usize) -> Result<T>,
    FMem: FnOnce(Arc<RowMatrix<f32>>) -> Result<T>,
{
    let (index, streaming_build) = if spec.supports_streaming_build() {
        if let Some(ref base_path) = dataset.base_fvecs_path {
            (build_streaming(base_path, dataset.num_base)?, true)
        } else {
            let base = dataset.get_or_load_base()?;
            (build_in_memory(base)?, false)
        }
    } else {
        let base = dataset.get_or_load_base()?;
        (build_in_memory(base)?, false)
    };

    let timings = index.q_build_timings();
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
                timings: QuantizedBuildTimings {
                    calibration_or_train_s: 0.0,
                    encode_s: 0.0,
                    graph_build_s: build_s,
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
        Phase6ArmKind::Sq8 => build_and_finalize_quantized_arm(
            SnapshotKind::Sq8,
            spec,
            config,
            dataset,
            &rerank_ladder,
            None,
            |base_path, num_base| {
                HnswSq8Index::build_streaming_from_fvecs(
                    base_path,
                    Metric::L2Squared,
                    hnsw_cfg.clone(),
                    Some(num_base),
                )
            },
            |base| {
                HnswSq8Index::build_with_shared_raw(
                    base,
                    Metric::L2Squared,
                    hnsw_cfg.clone(),
                    spec.build_mode,
                    spec.storage_enum,
                    Sq8KernelKind::QueryShifted,
                )
            },
        )?,
        Phase6ArmKind::Pq { m_pq } => {
            let pq_cfg = PqConfig {
                m_pq,
                k_centroids: PQ_K_CENTROIDS,
                max_train_vectors: PQ_DEFAULT_MAX_TRAIN_VECTORS,
                max_lloyd_iters: PQ_DEFAULT_MAX_LLOYD_ITERS,
                seed: config.seed,
            };
            build_and_finalize_quantized_arm(
                SnapshotKind::Pq,
                spec,
                config,
                dataset,
                &rerank_ladder,
                Some(pq_cfg),
                |base_path, num_base| {
                    HnswPqIndex::build_streaming_from_fvecs(
                        base_path,
                        Metric::L2Squared,
                        hnsw_cfg.clone(),
                        pq_cfg,
                        Some(num_base),
                    )
                },
                |base| {
                    HnswPqIndex::build_with_shared_raw(
                        base,
                        Metric::L2Squared,
                        hnsw_cfg.clone(),
                        pq_cfg,
                        spec.build_mode,
                        spec.storage_enum,
                    )
                },
            )?
        }
    };

    let ctx = ArmReportContext {
        spec,
        config,
        prov,
        hw,
        dataset,
        training_or_calibration_time_s: outcome.timings.calibration_or_train_s,
        encode_time_s: outcome.timings.encode_s,
        graph_build_time_s: outcome.timings.graph_build_s,
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
    write_file_with_parents(&report_path, json)?;

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
