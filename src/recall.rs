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
