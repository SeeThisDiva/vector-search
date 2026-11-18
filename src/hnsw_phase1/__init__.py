"""HNSW Phase 1: Foundations, Exact Brute-Force Ground Truth & Recall Harness."""

from src.hnsw_phase1.benchmark import run_phase1_benchmark
from src.hnsw_phase1.dataset import (
    generate_synthetic_sift,
    read_fvecs,
    read_fvecs_prefix,
    read_ivecs,
    read_ivecs_prefix,
    write_fvecs,
    write_ivecs,
)
from src.hnsw_phase1.exact_knn import ExactKNN
from src.hnsw_phase1.recall import (
    compute_recall_at_k,
    evaluate_recall_summary,
    verify_self_recall,
)

__all__ = [
    "ExactKNN",
    "compute_recall_at_k",
    "evaluate_recall_summary",
    "generate_synthetic_sift",
    "read_fvecs",
    "read_fvecs_prefix",
    "read_ivecs",
    "read_ivecs_prefix",
    "run_phase1_benchmark",
    "verify_self_recall",
    "write_fvecs",
    "write_ivecs",
]
