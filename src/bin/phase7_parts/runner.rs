//! Phase 7 Mode 1 (Pure Batch) + Mode 2 (Per-Query Distribution) Execution Loop.

use std::time::Instant;

use vector_search::benchmark::detect_provenance;
use vector_search::distance::Metric;
use vector_search::error::{Result, VectorSearchError};
use vector_search::hardware::{current_utc_rfc3339, detect_hardware, get_peak_rss_mb};
use vector_search::hnsw::{HnswIndex, VisitedTracker};
use vector_search::matrix::RowMatrix;
use vector_search::recall::{
    compute_recall_at_k, median_min_max, summarize_repeated_latencies, LatencyStats,
};

use super::cli::{apply_affinity_pinning, CompareMode, Phase7CliConfig};
use super::dataset_gt::{load_benchmark_dataset, resolve_ground_truth};
use super::report::{summarize_mode1_runs, Mode2Summary, Phase7Report, SweepPoint};

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

pub fn run_benchmark(cfg: &Phase7CliConfig) -> Result<Phase7Report> {
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
