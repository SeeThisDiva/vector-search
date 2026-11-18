//! Filtered Search Arm Measurement, Recall Evaluation, and Opaque Predicate Probes.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::error::Result;
use crate::filter::{DenseBitset, FilteredSearchOutput, FilteredSearchParams, LambdaPredicate};
use crate::hnsw::{FilterStrategy, FilteredSearchCounters, FilteredSearchScratch, HnswIndex};
use crate::json_util::escape_json;
use crate::matrix::RowMatrix;
use crate::recall::{
    compute_latency_stats, percentile_sorted, summarize_repeated_latencies, LatencyStats,
    RepeatedSweepSummary,
};

use super::MetadataMode;

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
    pub(crate) fn to_json_string(&self, indent: &str) -> String {
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
    pub(crate) fn to_json_string(&self, indent: &str) -> String {
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
