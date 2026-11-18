"""Comprehensive automated test suite for Phase 1.

Verifies:
1. Dataset generation & deterministic seeding
2. Texmex binary I/O (.fvecs, .ivecs) roundtrip and error handling
3. ExactKNN brute-force accuracy vs manual calculation and FAISS IndexFlatL2
4. Non-negative distance clamping & deterministic tie-breaking
5. Recall harness metrics, edge cases (duplicates, incomplete lists), and 100% self-recall invariant
6. Benchmark runner integration
"""

import os
import tempfile

import faiss  # type: ignore[import-untyped]
import numpy as np
import pytest

from src.hnsw_phase1.benchmark import run_phase1_benchmark
from src.hnsw_phase1.dataset import (
    generate_synthetic_sift,
    read_fvecs,
    read_ivecs,
    write_fvecs,
    write_ivecs,
)
from src.hnsw_phase1.exact_knn import ExactKNN
from src.hnsw_phase1.recall import (
    compute_recall_at_k,
    evaluate_recall_summary,
    verify_self_recall,
)

# ============================================================================
# 1. Dataset Generation & Seeding Tests
# ============================================================================

def test_dataset_generation_shapes_and_types():
    """Verify shapes and dtype of generated synthetic dataset."""
    num_base = 500
    num_query = 50
    dim = 64
    base, query = generate_synthetic_sift(
        num_base=num_base,
        num_query=num_query,
        dim=dim,
        num_clusters=5,
        seed=42,
    )

    assert base.shape == (num_base, dim)
    assert query.shape == (num_query, dim)
    assert base.dtype == np.float32
    assert query.dtype == np.float32
    assert np.all(np.isfinite(base))
    assert np.all(np.isfinite(query))


def test_dataset_generation_seed_reproducibility():
    """Verify identical seeds produce bit-exact identical vectors."""
    base1, query1 = generate_synthetic_sift(num_base=200, num_query=20, dim=32, seed=123)
    base2, query2 = generate_synthetic_sift(num_base=200, num_query=20, dim=32, seed=123)

    assert np.array_equal(base1, base2)
    assert np.array_equal(query1, query2)


def test_dataset_generation_different_seeds():
    """Verify different seeds produce distinct vector sets."""
    base1, query1 = generate_synthetic_sift(num_base=100, num_query=10, dim=16, seed=42)
    base2, query2 = generate_synthetic_sift(num_base=100, num_query=10, dim=16, seed=999)

    assert not np.array_equal(base1, base2)
    assert not np.array_equal(query1, query2)


def test_dataset_generation_invalid_args():
    """Verify ValueError is raised on invalid dimensions or sizes."""
    with pytest.raises(ValueError):
        generate_synthetic_sift(num_base=0)
    with pytest.raises(ValueError):
        generate_synthetic_sift(num_query=-5)
    with pytest.raises(ValueError):
        generate_synthetic_sift(dim=0)


# ============================================================================
# 2. Texmex Binary I/O (.fvecs, .ivecs) Tests
# ============================================================================

def test_fvecs_io_roundtrip():
    """Verify .fvecs write and read preserves float32 values bit-for-bit."""
    rng = np.random.default_rng(42)
    orig_data = rng.standard_normal((100, 32), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.fvecs")
        write_fvecs(fpath, orig_data)
        loaded_data = read_fvecs(fpath)

        assert loaded_data.shape == orig_data.shape
        assert loaded_data.dtype == np.float32
        assert np.array_equal(orig_data, loaded_data)


def test_ivecs_io_roundtrip():
    """Verify .ivecs write and read preserves int32 values bit-for-bit."""
    rng = np.random.default_rng(42)
    orig_data = rng.integers(0, 10000, size=(100, 10), dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.ivecs")
        write_ivecs(fpath, orig_data)
        loaded_data = read_ivecs(fpath)

        assert loaded_data.shape == orig_data.shape
        assert loaded_data.dtype == np.int32
        assert np.array_equal(orig_data, loaded_data)


def test_fvecs_empty_file():
    """Verify empty file handling for .fvecs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "empty.fvecs")
        write_fvecs(fpath, np.empty((0, 32), dtype=np.float32))
        loaded = read_fvecs(fpath)
        assert loaded.shape == (0, 0)
        assert loaded.dtype == np.float32


def test_ivecs_empty_file():
    """Verify empty file handling for .ivecs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "empty.ivecs")
        write_ivecs(fpath, np.empty((0, 10), dtype=np.int32))
        loaded = read_ivecs(fpath)
        assert loaded.shape == (0, 0)
        assert loaded.dtype == np.int32


def test_io_file_not_found():
    """Verify FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        read_fvecs("non_existent_file.fvecs")
    with pytest.raises(FileNotFoundError):
        read_ivecs("non_existent_file.ivecs")


def test_io_corrupt_files():
    """Verify ValueError for corrupt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "corrupt.fvecs")
        with open(fpath, "wb") as f:
            f.write(b"\x02\x00\x00\x00\x01\x02")  # partial record
        with pytest.raises(ValueError):
            read_fvecs(fpath)


# ============================================================================
# 3. Exact KNN Brute-Force Engine Tests
# ============================================================================

def test_exact_knn_manual_calculation():
    """Verify ExactKNN on known vectors matches analytical squared L2 distances."""
    # Base: 4 points in 2D
    # Point 0: (0, 0)
    # Point 1: (3, 0)
    # Point 2: (0, 4)
    # Point 3: (3, 4)
    base = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [0.0, 4.0],
        [3.0, 4.0],
    ], dtype=np.float32)

    # Query: (1, 1)
    # d^2 to 0: 1^2 + 1^2 = 2
    # d^2 to 1: (1-3)^2 + 1^2 = 4 + 1 = 5
    # d^2 to 2: 1^2 + (1-4)^2 = 1 + 9 = 10
    # d^2 to 3: (1-3)^2 + (1-4)^2 = 4 + 9 = 13
    query = np.array([[1.0, 1.0]], dtype=np.float32)

    knn = ExactKNN(metric="l2")
    knn.fit(base)
    indices, distances = knn.search(query, k=4)

    assert np.array_equal(indices[0], [0, 1, 2, 3])
    assert np.allclose(distances[0], [2.0, 5.0, 10.0, 13.0], atol=1e-5)


def test_exact_knn_tie_breaking():
    """Verify canonical (distance, id) tie-breaking selects lower IDs first."""
    # Base contains duplicate equidistant points at indices 0, 1, 3
    base = np.array([
        [2.0, 0.0],  # id 0, d^2=4
        [2.0, 0.0],  # id 1, d^2=4
        [5.0, 0.0],  # id 2, d^2=25
        [2.0, 0.0],  # id 3, d^2=4
    ], dtype=np.float32)

    query = np.array([[0.0, 0.0]], dtype=np.float32)

    knn = ExactKNN(metric="l2")
    knn.fit(base)

    # Top-2 must strictly select IDs 0 and 1
    indices, distances = knn.search(query, k=2)
    assert np.array_equal(indices[0], [0, 1])
    assert np.allclose(distances[0], [4.0, 4.0])

    # Top-3 must strictly select IDs 0, 1, 3
    indices, distances = knn.search(query, k=3)
    assert np.array_equal(indices[0], [0, 1, 3])
    assert np.allclose(distances[0], [4.0, 4.0, 4.0])


def test_exact_knn_distance_clamping():
    """Verify distance is clamped to >= 0.0 when query is identical to base vector."""
    base = np.array([[1.23456, 7.89101, -3.14159]], dtype=np.float32)
    query = np.array([[1.23456, 7.89101, -3.14159]], dtype=np.float32)

    knn = ExactKNN(metric="l2")
    knn.fit(base)
    indices, distances = knn.search(query, k=1)

    assert indices[0, 0] == 0
    assert distances[0, 0] == 0.0
    assert distances[0, 0] >= 0.0


def test_exact_knn_faiss_parity():
    """Verify ExactKNN matches FAISS IndexFlatL2 results 100%."""
    base, query = generate_synthetic_sift(num_base=1_000, num_query=50, dim=64, seed=101)
    k = 10

    # 1. ExactKNN
    knn = ExactKNN(metric="l2")
    knn.fit(base)
    our_indices, our_distances = knn.search(query, k=k)

    # 2. FAISS IndexFlatL2
    index = faiss.IndexFlatL2(64)
    index.add(base)
    faiss_distances, faiss_indices = index.search(query, k)

    # Verify ID alignment
    assert np.array_equal(our_indices, faiss_indices)
    # Verify squared L2 distance alignment
    assert np.allclose(our_distances, faiss_distances, atol=1e-3, rtol=1e-4)


def test_exact_knn_metric_modes():
    """Verify euclidean and inner product metrics."""
    base = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    query = np.array([[0.0, 0.0]], dtype=np.float32)

    # Euclidean (sqrt of squared L2)
    knn_euc = ExactKNN(metric="euclidean")
    knn_euc.fit(base)
    _, euc_dists = knn_euc.search(query, k=2)
    assert np.allclose(euc_dists[0], [1.0, 2.0])

    # Inner product (closest is highest dot product)
    knn_ip = ExactKNN(metric="ip")
    knn_ip.fit(base)
    # Query: (1, 1) -> dot(base[0]) = 1.0, dot(base[1]) = 2.0
    query_ip = np.array([[1.0, 1.0]], dtype=np.float32)
    ip_indices, ip_dists = knn_ip.search(query_ip, k=2)
    assert np.array_equal(ip_indices[0], [1, 0])  # base[1] has dot=2, base[0] has dot=1
    assert np.allclose(ip_dists[0], [2.0, 1.0])


def test_exact_knn_unfitted_or_dim_mismatch():
    """Verify runtime errors on unfitted calls or dimension mismatches."""
    knn = ExactKNN()
    with pytest.raises(RuntimeError):
        knn.search(np.zeros((1, 10), dtype=np.float32))

    base = np.zeros((10, 8), dtype=np.float32)
    knn.fit(base)
    with pytest.raises(ValueError):
        knn.search(np.zeros((1, 4), dtype=np.float32))


# ============================================================================
# 4. Recall Harness Tests
# ============================================================================

def test_recall_perfect_match():
    """Verify self-recall yields exactly 1.0 (100%)."""
    gt = np.array([
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
    ], dtype=np.int32)

    recall = compute_recall_at_k(gt, gt, k=5)
    assert recall == 1.0
    assert verify_self_recall(gt, k=5) is True


def test_recall_partial_and_disjoint():
    """Verify recall calculation for partial matches and completely disjoint sets."""
    gt = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)

    # Disjoint -> 0.0
    disjoint = np.array([[6, 7, 8, 9, 10]], dtype=np.int32)
    assert compute_recall_at_k(disjoint, gt, k=5) == 0.0

    # 3 matches out of 5 -> 0.6
    partial = np.array([[1, 2, 3, 99, 100]], dtype=np.int32)
    assert pytest.approx(compute_recall_at_k(partial, gt, k=5), 1e-6) == 0.6


def test_recall_duplicate_candidates_penalized():
    """Verify duplicate candidate IDs do not falsely inflate recall."""
    gt = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    # Repeated candidate ID 1 five times: should only count as 1 match out of 5
    dup = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
    assert pytest.approx(compute_recall_at_k(dup, gt, k=5), 1e-6) == 0.2


def test_recall_incomplete_candidates():
    """Verify incomplete candidate list preserves denominator k."""
    gt = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    # Candidate list has only 2 items: [1, 2]
    cand = np.array([[1, 2]], dtype=np.int32)
    assert pytest.approx(compute_recall_at_k(cand, gt, k=5), 1e-6) == 0.4


def test_verify_self_recall_failure():
    """Verify AssertionError is raised when ground truth has duplicates breaking self-consistency."""
    invalid_gt = np.array([[1, 1, 2]], dtype=np.int32)
    with pytest.raises(AssertionError, match="Self-verification failed"):
        verify_self_recall(invalid_gt, k=3)


def test_evaluate_recall_summary():
    """Verify recall summary across multiple k ranks."""
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)
    cand = np.array([[0, 1, 2, 3, 4, 90, 91, 92, 93, 94]], dtype=np.int32)

    summary = evaluate_recall_summary(cand, gt, k_values=[1, 5, 10])
    assert summary["Recall@1"] == 1.0
    assert summary["Recall@5"] == 1.0
    assert summary["Recall@10"] == 0.5


# ============================================================================
# 5. Benchmark Pipeline Integration Test
# ============================================================================

def test_run_phase1_benchmark_pipeline():
    """Verify the full Phase 1 benchmark runner executes end-to-end cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_phase1_benchmark(
            num_base=500,
            num_query=20,
            dim=32,
            k=5,
            seed=42,
            output_dir=tmpdir,
            save_files=True,
        )

        assert results["dataset"]["num_base"] == 500
        assert results["dataset"]["num_query"] == 20
        assert results["recall"]["self_recall"] == 1.0
        assert results["recall"]["self_verified"] is True
        assert results["exact_knn"]["qps"] > 0

        # Check binary files exist and reload properly
        base_fvecs = results["files"]["base_fvecs"]
        gt_ivecs = results["files"]["groundtruth_ivecs"]

        assert os.path.exists(base_fvecs)
        assert os.path.exists(gt_ivecs)

        reloaded_base = read_fvecs(base_fvecs)
        reloaded_gt = read_ivecs(gt_ivecs)

        assert reloaded_base.shape == (500, 32)
        assert reloaded_gt.shape == (20, 5)
