"""Phase 1 Benchmark Pipeline and Ground Truth Generator."""

import os
import time
from typing import Any

import numpy as np

from src.hnsw_phase1.dataset import (
    generate_synthetic_sift,
    write_fvecs,
    write_ivecs,
)
from src.hnsw_phase1.exact_knn import ExactKNN
from src.hnsw_phase1.recall import (
    compute_recall_at_k,
    evaluate_recall_summary,
    verify_self_recall,
)


def run_phase1_benchmark(
    num_base: int = 10_000,
    num_query: int = 100,
    dim: int = 128,
    k: int = 10,
    seed: int = 42,
    output_dir: str = "data",
    save_files: bool = True,
) -> dict[str, Any]:
    """
    Executes the full Phase 1 benchmark pipeline.

    1. Generates reproducible synthetic SIFT-like vectors (seed=42).
    2. Writes base.fvecs and query.fvecs to disk.
    3. Fits ExactKNN and computes exact ground truth top-k neighbors and distances.
    4. Writes groundtruth.ivecs and groundtruth.fvecs to disk.
    5. Evaluates recall harness and verifies 100% self-recall invariant.
    6. Cross-verifies against FAISS IndexFlatL2 if available.

    Parameters
    ----------
    num_base : int
        Number of base vectors. Default 10,000.
    num_query : int
        Number of query vectors. Default 100.
    dim : int
        Vector dimension. Default 128.
    k : int
        Top-k nearest neighbors to compute. Default 10.
    seed : int
        Random seed for reproducibility. Default 42.
    output_dir : str
        Directory to store .fvecs and .ivecs binary files. Default "data".
    save_files : bool
        Whether to serialize vectors and ground truth to disk. Default True.

    Returns
    -------
    dict[str, Any]
        Structured benchmark results, timings, recall metrics, and parity status.
    """
    if save_files:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset Generation
    t_gen_start = time.perf_counter()
    base_vectors, query_vectors = generate_synthetic_sift(
        num_base=num_base,
        num_query=num_query,
        dim=dim,
        seed=seed,
    )
    t_gen_elapsed = time.perf_counter() - t_gen_start

    base_fvecs_path = os.path.join(output_dir, "base.fvecs")
    query_fvecs_path = os.path.join(output_dir, "query.fvecs")
    gt_ivecs_path = os.path.join(output_dir, "groundtruth.ivecs")
    gt_fvecs_path = os.path.join(output_dir, "groundtruth.fvecs")

    if save_files:
        write_fvecs(base_fvecs_path, base_vectors)
        write_fvecs(query_fvecs_path, query_vectors)

    # 2. Exact Brute-Force KNN Search
    knn = ExactKNN(metric="l2")

    t_fit_start = time.perf_counter()
    knn.fit(base_vectors)
    t_fit_elapsed = time.perf_counter() - t_fit_start

    t_search_start = time.perf_counter()
    gt_indices, gt_distances = knn.search(query_vectors, k=k)
    t_search_elapsed = time.perf_counter() - t_search_start

    qps = num_query / t_search_elapsed if t_search_elapsed > 0 else 0.0
    latency_ms = (t_search_elapsed / num_query) * 1000.0 if num_query > 0 else 0.0

    if save_files:
        write_ivecs(gt_ivecs_path, gt_indices)
        write_fvecs(gt_fvecs_path, gt_distances)

    # 3. Recall Verification
    self_recall = compute_recall_at_k(gt_indices, gt_indices, k=k)
    self_verified = verify_self_recall(gt_indices, k=k)
    recall_summary = evaluate_recall_summary(
        gt_indices, gt_indices, k_values=[1, min(5, k), k]
    )

    # 4. FAISS Parity Verification (if FAISS is installed)
    faiss_available = False
    faiss_parity = False
    faiss_max_dist_diff = 0.0
    faiss_index_matches = 0
    effective_k = int(gt_indices.shape[1])
    total_elements = num_query * effective_k

    try:
        import faiss

        faiss_available = True
        index = faiss.IndexFlatL2(dim)
        index.add(base_vectors)
        faiss_dists, faiss_indices = index.search(query_vectors, k)
        faiss_dists = faiss_dists[:, :effective_k]
        faiss_indices = faiss_indices[:, :effective_k]

        # Check index ID alignment
        faiss_index_matches = int(np.sum(faiss_indices == gt_indices))

        # Check distance numerical difference (squared L2)
        dist_diff = np.abs(faiss_dists - gt_distances)
        faiss_max_dist_diff = float(np.max(dist_diff))
        faiss_max_rel_diff = float(np.max(dist_diff / np.maximum(1.0, gt_distances)))

        # FAISS matches if recall@k is 1.0 and relative distance difference is within float32 tolerance
        faiss_recall = compute_recall_at_k(faiss_indices, gt_indices, k=k)
        if faiss_recall == 1.0 and faiss_max_rel_diff < 1e-4:
            faiss_parity = True

    except ImportError:
        faiss_available = False

    results: dict[str, Any] = {
        "dataset": {
            "num_base": num_base,
            "num_query": num_query,
            "dim": dim,
            "k": k,
            "seed": seed,
            "base_memory_mb": float(base_vectors.nbytes) / (1024 * 1024),
            "query_memory_mb": float(query_vectors.nbytes) / (1024 * 1024),
            "generation_time_s": t_gen_elapsed,
        },
        "exact_knn": {
            "fit_time_s": t_fit_elapsed,
            "search_time_s": t_search_elapsed,
            "qps": qps,
            "latency_ms_per_query": latency_ms,
        },
        "recall": {
            "self_recall": self_recall,
            "self_verified": self_verified,
            "metrics": recall_summary,
        },
        "files": {
            "base_fvecs": base_fvecs_path if save_files else None,
            "query_fvecs": query_fvecs_path if save_files else None,
            "groundtruth_ivecs": gt_ivecs_path if save_files else None,
            "groundtruth_fvecs": gt_fvecs_path if save_files else None,
        },
        "faiss_verification": {
            "faiss_available": faiss_available,
            "faiss_parity": faiss_parity,
            "faiss_max_dist_diff": faiss_max_dist_diff,
            "faiss_index_matches": faiss_index_matches,
            "faiss_total_elements": total_elements,
        },
    }

    return results
