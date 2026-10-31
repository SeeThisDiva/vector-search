"""Automated recall evaluation harness and self-verification for vector search."""

import numpy as np


def compute_recall_at_k(
    candidate_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k: int = 10,
) -> float:
    """
    Computes standard mean Recall@k across queries.

    For each query q:
        Recall@k(q) = |set(candidate_indices[q, :k]) ∩ set(ground_truth_indices[q, :k])| / k

    Mean Recall@k is the average over all queries:
        Recall@k = (1 / num_queries) * sum_q Recall@k(q)

    Defensive features:
    - Deduplicates candidate IDs so duplicate predictions are properly penalized.
    - Preserves denominator k if candidate list is incomplete (size < k).
    - Validates query count alignment.

    Parameters
    ----------
    candidate_indices : np.ndarray
        2D array of shape (num_queries, num_candidates) with candidate neighbor IDs.
    ground_truth_indices : np.ndarray
        2D array of shape (num_queries, num_gt) with exact ground truth neighbor IDs.
    k : int
        Recall rank threshold. Default 10.

    Returns
    -------
    float
        Mean Recall@k in range [0.0, 1.0].
    """
    cand = np.asarray(candidate_indices, dtype=np.int32)
    gt = np.asarray(ground_truth_indices, dtype=np.int32)

    if cand.ndim == 1:
        cand = cand.reshape(1, -1)
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)

    if cand.ndim != 2 or gt.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got candidate shape {cand.shape} and ground truth shape {gt.shape}"
        )

    num_queries = gt.shape[0]
    if cand.shape[0] != num_queries:
        raise ValueError(
            f"Query count mismatch: candidates have {cand.shape[0]} queries, "
            f"ground truth has {num_queries} queries"
        )

    if num_queries == 0:
        return 1.0

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Effective k cannot exceed ground truth width
    effective_k = min(k, gt.shape[1])
    if effective_k <= 0:
        raise ValueError("Ground truth has 0 columns; cannot compute recall.")

    total_matches = 0
    cand_width = cand.shape[1]

    for q in range(num_queries):
        gt_set = set(gt[q, :effective_k].tolist())
        cand_slice = cand[q, :min(effective_k, cand_width)].tolist()
        cand_set = set(cand_slice)
        matches = len(gt_set.intersection(cand_set))
        total_matches += matches

    mean_recall = float(total_matches) / float(num_queries * effective_k)
    return mean_recall


def verify_self_recall(
    ground_truth_indices: np.ndarray,
    k: int = 10,
) -> bool:
    """
    Verifies the fundamental self-consistency invariant:
        Recall@k(GroundTruth, GroundTruth) == 1.0 (100.0%)

    Parameters
    ----------
    ground_truth_indices : np.ndarray
        2D array of ground truth neighbor IDs.
    k : int
        Rank threshold.

    Returns
    -------
    bool
        True if recall is exactly 1.0.

    Raises
    ------
    AssertionError
        If self-recall is not identically 1.0.
    """
    recall = compute_recall_at_k(ground_truth_indices, ground_truth_indices, k=k)
    if not np.isclose(recall, 1.0, atol=1e-7):
        raise AssertionError(
            f"Self-verification failed: Recall@{k}(GroundTruth, GroundTruth) = {recall} (expected 1.0)"
        )
    return True


def evaluate_recall_summary(
    candidate_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10, 50, 100),
) -> dict[str, float]:
    """
    Evaluates Recall@k across multiple rank thresholds.

    Parameters
    ----------
    candidate_indices : np.ndarray
        Candidate neighbor indices (Q, K_cand).
    ground_truth_indices : np.ndarray
        Ground truth neighbor indices (Q, K_gt).
    k_values : list[int]
        Rank thresholds to evaluate.

    Returns
    -------
    dict[str, float]
        Mapping from "Recall@k" to computed recall value.
    """
    gt_max_k = ground_truth_indices.shape[1] if ground_truth_indices.ndim == 2 else 0
    results: dict[str, float] = {}
    for k in k_values:
        if 1 <= k <= gt_max_k:
            results[f"Recall@{k}"] = compute_recall_at_k(candidate_indices, ground_truth_indices, k=k)
    return results
