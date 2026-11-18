//! Recall@k evaluation harness, self-verification invariant checks, and latency percentile metrics.

use std::collections::HashSet;

use crate::error::{Result, VectorSearchError};
use crate::matrix::RowMatrix;

/// Query latency summary statistics across a benchmark batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    pub num_queries: usize,
    pub total_s: f64,
    pub qps: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            num_queries: 0,
            total_s: 0.0,
            qps: 0.0,
            min_ms: 0.0,
            mean_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            max_ms: 0.0,
        }
    }
}

/// Computes linear-interpolated `p`-th percentile (`p` in `[0.0, 100.0]`) on a sorted slice.
///
/// Matches standard `numpy.percentile(..., method="linear")` behavior.
#[must_use]
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped_p = p.clamp(0.0, 100.0);
    let rank = (clamped_p / 100.0) * ((sorted.len() - 1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = rank - (lo as f64);
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

/// Computes `min`, `mean`, `p50`, `p95`, `p99`, `max`, and `QPS` from per-query latencies (in ms).
#[must_use]
pub fn compute_latency_stats(latencies_ms: &[f64], total_elapsed_s: f64) -> LatencyStats {
    if latencies_ms.is_empty() {
        return LatencyStats::default();
    }

    let mut sorted: Vec<f64> = latencies_ms
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    if sorted.is_empty() {
        return LatencyStats::default();
    }
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len();
    let sum_ms: f64 = sorted.iter().sum();
    let mean_ms = sum_ms / (n as f64);
    let effective_total_s = if total_elapsed_s > 0.0 {
        total_elapsed_s
    } else {
        sum_ms / 1000.0
    };
    let qps = if effective_total_s > 0.0 {
        (n as f64) / effective_total_s
    } else {
        0.0
    };

    LatencyStats {
        num_queries: n,
        total_s: effective_total_s,
        qps,
        min_ms: sorted[0],
        mean_ms,
        p50_ms: percentile_sorted(&sorted, 50.0),
        p95_ms: percentile_sorted(&sorted, 95.0),
        p99_ms: percentile_sorted(&sorted, 99.0),
        max_ms: sorted[n - 1],
    }
}

/// Computes standard mean `Recall@k` across all queries:
///
/// $$\text{Recall}@k = \frac{1}{Q} \sum_{q=1}^{Q} \frac{|\text{set}(C_{q, :k'}) \cap \text{set}(G_{q, :k'})|}{k'}$$
///
/// where $k' = \min(k, \text{ground\_truth.cols()})$.
///
/// Defensive properties:
/// - Deduplicates candidate IDs per query so duplicate predictions never inflate recall.
/// - Preserves denominator $k'$ even when candidate width is smaller than $k'$.
/// - Validates query row alignment and positive `k`.
pub fn compute_recall_at_k(
    candidate_indices: &RowMatrix<i32>,
    ground_truth_indices: &RowMatrix<i32>,
    k: usize,
) -> Result<f64> {
    if k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0) for Recall@k".to_string(),
        ));
    }

    let num_queries = ground_truth_indices.rows();
    if candidate_indices.rows() != num_queries {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Query count mismatch: candidates have {} rows, ground truth has {num_queries} rows",
            candidate_indices.rows()
        )));
    }

    if num_queries == 0 {
        return Ok(1.0);
    }

    if ground_truth_indices.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Ground truth matrix has 0 columns; cannot compute recall".to_string(),
        ));
    }

    let effective_k = k.min(ground_truth_indices.cols());
    let cand_cols = candidate_indices.cols();
    let cand_take = effective_k.min(cand_cols);

    let mut total_matches: usize = 0;
    let mut gt_set = HashSet::with_capacity(effective_k * 2);
    let mut cand_seen = HashSet::with_capacity(cand_take * 2);

    for q in 0..num_queries {
        gt_set.clear();
        cand_seen.clear();

        for &gt_id in &ground_truth_indices.row(q)[..effective_k] {
            gt_set.insert(gt_id);
        }

        if cand_take > 0 {
            for &cand_id in &candidate_indices.row(q)[..cand_take] {
                if cand_seen.insert(cand_id) && gt_set.contains(&cand_id) {
                    total_matches += 1;
                }
            }
        }
    }

    let denom = (num_queries * effective_k) as f64;
    Ok((total_matches as f64) / denom)
}

/// Verifies the fundamental ground-truth self-consistency invariant:
///
/// $$\text{Recall}@k(\text{GroundTruth}, \text{GroundTruth}) \equiv 1.0$$
///
/// Fails if ground truth contains duplicate neighbor IDs within top-`k` or fails exact 1.0 recall.
pub fn verify_self_recall(ground_truth_indices: &RowMatrix<i32>, k: usize) -> Result<bool> {
    let recall = compute_recall_at_k(ground_truth_indices, ground_truth_indices, k)?;
    if (recall - 1.0).abs() > 1e-9 {
        return Err(VectorSearchError::SelfVerificationFailed {
            k,
            actual_recall: recall,
        });
    }
    Ok(true)
}

/// Evaluates `Recall@k` across multiple rank thresholds `k_values` (filtering to `1 <= k <= gt.cols()`).
///
/// Deduplicates `k_values` while preserving ascending/specified order.
pub fn evaluate_recall_summary(
    candidate_indices: &RowMatrix<i32>,
    ground_truth_indices: &RowMatrix<i32>,
    k_values: &[usize],
) -> Result<Vec<(String, f64)>> {
    let gt_max_k = ground_truth_indices.cols();
    let mut seen = HashSet::new();
    let mut results = Vec::new();

    for &k in k_values {
        if k >= 1 && k <= gt_max_k && seen.insert(k) {
            let val = compute_recall_at_k(candidate_indices, ground_truth_indices, k)?;
            results.push((format!("Recall@{k}"), val));
        }
    }
    Ok(results)
}

/// Computes `(median, min, max)` over `values`, returning `(0.0, 0.0, 0.0)` when empty.
#[must_use]
pub fn median_min_max(values: &[f64]) -> (f64, f64, f64) {
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

/// Summary statistics across repeated `efSearch` sweep runs.
#[derive(Debug, Clone, PartialEq)]
pub struct RepeatedSweepSummary {
    pub num_runs: usize,
    pub median_latency: LatencyStats,
    pub mean_ms_min: f64,
    pub mean_ms_max: f64,
    pub mean_ms_spread: f64,
    pub p99_ms_min: f64,
    pub p99_ms_max: f64,
    pub p99_ms_spread: f64,
    pub run_mean_latencies_ms: Vec<f64>,
    pub run_p99_latencies_ms: Vec<f64>,
}

/// Computes median `LatencyStats` and min/max/spread across `N` repeated batch search runs.
#[must_use]
pub fn summarize_repeated_latencies(runs: &[LatencyStats]) -> RepeatedSweepSummary {
    if runs.is_empty() {
        return RepeatedSweepSummary {
            num_runs: 0,
            median_latency: LatencyStats::default(),
            mean_ms_min: 0.0,
            mean_ms_max: 0.0,
            mean_ms_spread: 0.0,
            p99_ms_min: 0.0,
            p99_ms_max: 0.0,
            p99_ms_spread: 0.0,
            run_mean_latencies_ms: Vec::new(),
            run_p99_latencies_ms: Vec::new(),
        };
    }

    let num_runs = runs.len();
    let num_queries = runs[0].num_queries;
    let run_mean_latencies_ms: Vec<f64> = runs.iter().map(|r| r.mean_ms).collect();
    let run_p99_latencies_ms: Vec<f64> = runs.iter().map(|r| r.p99_ms).collect();

    let extract_stat = |extract: fn(&LatencyStats) -> f64| -> (f64, f64, f64) {
        let vals: Vec<f64> = runs.iter().map(extract).collect();
        median_min_max(&vals)
    };

    let (total_s, _, _) = extract_stat(|r| r.total_s);
    let (min_ms, _, _) = extract_stat(|r| r.min_ms);
    let (mean_ms, mean_ms_min, mean_ms_max) = extract_stat(|r| r.mean_ms);
    let (p50_ms, _, _) = extract_stat(|r| r.p50_ms);
    let (p95_ms, _, _) = extract_stat(|r| r.p95_ms);
    let (p99_ms, p99_ms_min, p99_ms_max) = extract_stat(|r| r.p99_ms);
    let (max_ms, _, _) = extract_stat(|r| r.max_ms);
    let qps = if total_s > 0.0 {
        (num_queries as f64) / total_s
    } else {
        0.0
    };

    RepeatedSweepSummary {
        num_runs,
        median_latency: LatencyStats {
            num_queries,
            total_s,
            qps,
            min_ms,
            mean_ms,
            p50_ms,
            p95_ms,
            p99_ms,
            max_ms,
        },
        mean_ms_min,
        mean_ms_max,
        mean_ms_spread: (mean_ms_max - mean_ms_min).max(0.0),
        p99_ms_min,
        p99_ms_max,
        p99_ms_spread: (p99_ms_max - p99_ms_min).max(0.0),
        run_mean_latencies_ms,
        run_p99_latencies_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_min_max_and_repeated_summary() {
        assert_eq!(median_min_max(&[]), (0.0, 0.0, 0.0));
        assert_eq!(
            median_min_max(&[3.0, 1.0, 2.0, 10.0, 4.0]),
            (3.0, 1.0, 10.0)
        );

        let empty_summary = summarize_repeated_latencies(&[]);
        assert_eq!(empty_summary.num_runs, 0);
        assert_eq!(empty_summary.median_latency, LatencyStats::default());

        let runs = vec![
            LatencyStats {
                num_queries: 100,
                total_s: 0.2,
                qps: 500.0,
                min_ms: 0.5,
                mean_ms: 2.0,
                p50_ms: 1.9,
                p95_ms: 2.5,
                p99_ms: 3.0,
                max_ms: 4.0,
            },
            LatencyStats {
                num_queries: 100,
                total_s: 0.1,
                qps: 1000.0,
                min_ms: 0.3,
                mean_ms: 1.0,
                p50_ms: 0.9,
                p95_ms: 1.4,
                p99_ms: 2.0,
                max_ms: 2.5,
            },
            LatencyStats {
                num_queries: 100,
                total_s: 0.4,
                qps: 250.0,
                min_ms: 0.8,
                mean_ms: 4.0,
                p50_ms: 3.9,
                p95_ms: 5.0,
                p99_ms: 6.0,
                max_ms: 8.0,
            },
        ];
        let summary = summarize_repeated_latencies(&runs);
        assert_eq!(summary.num_runs, 3);
        assert!((summary.median_latency.mean_ms - 2.0).abs() < 1e-12);
        assert!((summary.median_latency.qps - 500.0).abs() < 1e-12);
        assert!((summary.mean_ms_min - 1.0).abs() < 1e-12);
        assert!((summary.mean_ms_max - 4.0).abs() < 1e-12);
        assert!((summary.mean_ms_spread - 3.0).abs() < 1e-12);
        assert!((summary.p99_ms_min - 2.0).abs() < 1e-12);
        assert!((summary.p99_ms_max - 6.0).abs() < 1e-12);
        assert!((summary.p99_ms_spread - 4.0).abs() < 1e-12);
    }
}
