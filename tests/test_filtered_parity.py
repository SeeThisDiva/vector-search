"""Parity and ground-truth verification tests for FilteredExactKnn (Task T3.3).

Verifies bit-exact mathematical parity between the compiled Rust `filtered_exact_knn`
binary (invoked strictly via `subprocess.run`) and pure Python NumPy across all four
selectivity tiers {50%, 10%, 1%, 0.1%} on 10k smoke datasets.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.hnsw_phase1.dataset import read_fvecs, read_ivecs, write_fvecs, write_ivecs

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_rust_filtered_exact_knn_binary() -> Path:
    """Locates or builds the compiled Rust `filtered_exact_knn` release binary."""
    exe_name = "filtered_exact_knn.exe" if os.name == "nt" else "filtered_exact_knn"
    release_bin = REPO_ROOT / "target" / "release" / exe_name
    debug_bin = REPO_ROOT / "target" / "debug" / exe_name

    if release_bin.is_file():
        return release_bin

    cargo = shutil.which("cargo")
    if cargo is not None:
        proc = subprocess.run(
            [cargo, "build", "--release", "--bin", "filtered_exact_knn"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and release_bin.is_file():
            return release_bin

    if debug_bin.is_file():
        return debug_bin

    raise RuntimeError(
        f"Rust `filtered_exact_knn` binary not found at {release_bin} or {debug_bin}, "
        "and failed to build via cargo."
    )


def numpy_filtered_exact_knn(
    base: np.ndarray,
    queries: np.ndarray,
    matching_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes exact Top-k filtered nearest neighbors using pure NumPy.

    Enforces canonical (distance, id) tie-breaking:
    (d_a, id_a) < (d_b, id_b) <=> d_a < d_b or (d_a == d_b and id_a < id_b).
    """
    valid_ids = np.asarray(matching_ids, dtype=np.int32)
    num_queries = len(queries)
    effective_k = min(k, len(valid_ids))

    if effective_k == 0 or num_queries == 0:
        return np.empty((num_queries, 0), dtype=np.int32), np.empty((num_queries, 0), dtype=np.float32)

    matching_base = base[valid_ids]
    out_indices = np.empty((num_queries, effective_k), dtype=np.int32)
    out_distances = np.empty((num_queries, effective_k), dtype=np.float32)

    for q_idx, q in enumerate(queries):
        diff = matching_base - q
        # Pairwise squared Euclidean distance
        dists = np.sum(diff * diff, axis=1, dtype=np.float32)
        np.maximum(0.0, dists, out=dists)

        # Canonical tie-breaking: primary = dists, secondary = valid_ids
        sort_order = np.lexsort((valid_ids, dists))
        top_k_order = sort_order[:effective_k]

        out_indices[q_idx] = valid_ids[top_k_order]
        out_distances[q_idx] = dists[top_k_order]

    return out_indices, out_distances


def test_filtered_parity_sift_10k_all_four_selectivity_tiers() -> None:
    """Verifies 100% ID and numerical parity on 10k SIFT smoke data across 4 tiers."""
    rust_bin = _resolve_rust_filtered_exact_knn_binary()
    base_path = REPO_ROOT / "data" / "siftsmall" / "siftsmall_base.fvecs"
    query_path = REPO_ROOT / "data" / "siftsmall" / "siftsmall_query.fvecs"

    assert base_path.is_file(), f"SIFT smoke base file not found at {base_path}"
    assert query_path.is_file(), f"SIFT smoke query file not found at {query_path}"

    base = read_fvecs(str(base_path))
    query = read_fvecs(str(query_path))
    num_base = len(base)
    assert num_base == 10_000, f"Expected 10k base vectors, got {num_base}"

    tiers = [0.5, 0.1, 0.01, 0.001]
    k = 10

    for sigma in tiers:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_report_path = os.path.join(tmpdir, "report.json")
            proc = subprocess.run(
                [
                    str(rust_bin),
                    "--base-fvecs",
                    str(base_path),
                    "--query-fvecs",
                    str(query_path),
                    "--selectivity",
                    str(sigma),
                    "--k",
                    str(k),
                    "--output-dir",
                    tmpdir,
                    "--json-out",
                    json_report_path,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, (
                f"Rust binary failed for sigma={sigma}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

            # 1. Inspect provenance report
            with open(json_report_path, encoding="utf-8") as f:
                report = json.load(f)

            expected_popcount = int(np.floor(sigma * num_base))
            assert report["filter"]["selectivity"] == sigma
            assert report["filter"]["popcount"] == expected_popcount
            assert report["recall"]["self_verified"] is True
            assert report["recall"]["self_recall_at_k"] == 1.0
            assert report["exact_knn"]["effective_k"] == min(k, expected_popcount)

            # 2. Load outputs
            rust_indices = read_ivecs(os.path.join(tmpdir, "groundtruth.ivecs"))
            rust_distances = read_fvecs(os.path.join(tmpdir, "groundtruth.fvecs"))
            filter_ids = read_ivecs(os.path.join(tmpdir, "filter.ivecs"))[0]

            assert len(filter_ids) == expected_popcount

            # 3. Compute pure NumPy reference
            py_indices, py_distances = numpy_filtered_exact_knn(base, query, filter_ids, k)

            # 4. Strict parity assertion (100% ID and numerical equality)
            assert np.array_equal(rust_indices, py_indices), (
                f"Index mismatch for tier sigma={sigma}!\nRust:\n{rust_indices[:3]}\nNumPy:\n{py_indices[:3]}"
            )
            assert np.allclose(rust_distances, py_distances, atol=1e-5, rtol=1e-5), (
                f"Distance mismatch for tier sigma={sigma}!\nRust:\n{rust_distances[:3]}\nNumPy:\n{py_distances[:3]}"
            )
            # SIFT coordinates are integers, so squared distances are bit-exact integers
            assert np.array_equal(rust_distances, py_distances)


def test_filtered_parity_synthetic_10k_all_four_selectivity_tiers() -> None:
    """Verifies parity on 10k synthetic dataset across all 4 selectivity tiers."""
    rust_bin = _resolve_rust_filtered_exact_knn_binary()
    tiers = [0.5, 0.1, 0.01, 0.001]
    k = 10

    for sigma in tiers:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_report_path = os.path.join(tmpdir, "report.json")
            proc = subprocess.run(
                [
                    str(rust_bin),
                    "--num-base",
                    "10000",
                    "--num-query",
                    "20",
                    "--dim",
                    "64",
                    "--seed",
                    "77",
                    "--selectivity",
                    str(sigma),
                    "--k",
                    str(k),
                    "--output-dir",
                    tmpdir,
                    "--json-out",
                    json_report_path,
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, (
                f"Synthetic run failed for sigma={sigma}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

            # Read back generated base, query, bitset, and groundtruth
            base = read_fvecs(os.path.join(tmpdir, "base.fvecs"))
            query = read_fvecs(os.path.join(tmpdir, "query.fvecs"))
            rust_indices = read_ivecs(os.path.join(tmpdir, "groundtruth.ivecs"))
            rust_distances = read_fvecs(os.path.join(tmpdir, "groundtruth.fvecs"))
            filter_ids = read_ivecs(os.path.join(tmpdir, "filter.ivecs"))[0]

            assert base.shape == (10_000, 64)
            assert query.shape == (20, 64)

            # Pure NumPy reference
            py_indices, py_distances = numpy_filtered_exact_knn(base, query, filter_ids, k)

            assert np.array_equal(rust_indices, py_indices)
            assert np.allclose(rust_distances, py_distances, atol=1e-5, rtol=1e-5)


def test_filtered_parity_canonical_tie_breaking() -> None:
    """Verifies canonical (distance, id) tie-breaking between Rust and NumPy."""
    rust_bin = _resolve_rust_filtered_exact_knn_binary()

    # Query at [0, 0, 0, 0]
    # IDs 1, 3, 5 all have d^2 = 4.0; IDs 0, 4 have d^2 = 25.0
    base = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],   # id 0, d^2 = 25.0
            [2.0, 0.0, 0.0, 0.0],   # id 1, d^2 = 4.0
            [10.0, 0.0, 0.0, 0.0],  # id 2, d^2 = 100.0
            [-2.0, 0.0, 0.0, 0.0],  # id 3, d^2 = 4.0
            [0.0, 5.0, 0.0, 0.0],   # id 4, d^2 = 25.0
            [0.0, -2.0, 0.0, 0.0],  # id 5, d^2 = 4.0
        ],
        dtype=np.float32,
    )
    query = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, "tied_base.fvecs")
        query_file = os.path.join(tmpdir, "tied_query.fvecs")
        filter_file = os.path.join(tmpdir, "tied_filter.ivecs")
        out_ivecs = os.path.join(tmpdir, "gt.ivecs")
        out_fvecs = os.path.join(tmpdir, "gt.fvecs")

        write_fvecs(base_file, base)
        write_fvecs(query_file, query)
        # Include all 6 IDs in filter
        write_ivecs(filter_file, np.arange(6, dtype=np.int32).reshape(1, -1))

        proc = subprocess.run(
            [
                str(rust_bin),
                "--base-fvecs",
                base_file,
                "--query-fvecs",
                query_file,
                "--filter-bitset",
                filter_file,
                "--k",
                "4",
                "--output-ivecs",
                out_ivecs,
                "--output-fvecs",
                out_fvecs,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"

        rust_idx = read_ivecs(out_ivecs)
        rust_dist = read_fvecs(out_fvecs)

        py_idx, py_dist = numpy_filtered_exact_knn(base, query, np.arange(6, dtype=np.int32), k=4)

        # Expected top-4:
        # Distance 4.0 tied among 1, 3, 5 -> ordered 1, 3, 5
        # 4th element: distance 25.0 tied among 0, 4 -> ordered 0
        expected_ids = np.array([[1, 3, 5, 0]], dtype=np.int32)
        expected_dists = np.array([[4.0, 4.0, 4.0, 25.0]], dtype=np.float32)

        assert np.array_equal(rust_idx, expected_ids)
        assert np.array_equal(rust_idx, py_idx)
        assert np.allclose(rust_dist, expected_dists, atol=1e-6)
        assert np.allclose(rust_dist, py_dist, atol=1e-6)


def test_filtered_parity_edge_cases_zero_matches_and_fewer_than_k() -> None:
    """Verifies edge cases: 0 matches (empty result) and fewer than k matches."""
    rust_bin = _resolve_rust_filtered_exact_knn_binary()

    base = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([[0.0, 0.0]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, "edge_base.fvecs")
        query_file = os.path.join(tmpdir, "edge_query.fvecs")
        write_fvecs(base_file, base)
        write_fvecs(query_file, query)

        # 1. Zero matches (filter has 0 items)
        filter_empty = os.path.join(tmpdir, "filter_empty.ivecs")
        write_ivecs(filter_empty, np.empty((0, 0), dtype=np.int32))
        out_zero_ivecs = os.path.join(tmpdir, "zero.ivecs")

        proc_zero = subprocess.run(
            [
                str(rust_bin),
                "--base-fvecs",
                base_file,
                "--query-fvecs",
                query_file,
                "--filter-bitset",
                filter_empty,
                "--k",
                "5",
                "--output-ivecs",
                out_zero_ivecs,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_zero.returncode == 0, proc_zero.stderr
        zero_res = read_ivecs(out_zero_ivecs)
        assert len(zero_res) == 0 or zero_res.shape[1] == 0

        # 2. Fewer than k matches (2 items match, requested k=10)
        filter_two = os.path.join(tmpdir, "filter_two.ivecs")
        write_ivecs(filter_two, np.array([[1, 3]], dtype=np.int32))
        out_two_ivecs = os.path.join(tmpdir, "two.ivecs")
        out_two_fvecs = os.path.join(tmpdir, "two.fvecs")

        proc_two = subprocess.run(
            [
                str(rust_bin),
                "--base-fvecs",
                base_file,
                "--query-fvecs",
                query_file,
                "--filter-bitset",
                filter_two,
                "--k",
                "10",
                "--output-ivecs",
                out_two_ivecs,
                "--output-fvecs",
                out_two_fvecs,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_two.returncode == 0, proc_two.stderr
        two_idx = read_ivecs(out_two_ivecs)
        two_dist = read_fvecs(out_two_fvecs)

        # Exactly 2 items returned, ordered ascending
        assert two_idx.shape == (1, 2)
        assert two_dist.shape == (1, 2)
        assert np.array_equal(two_idx, np.array([[1, 3]], dtype=np.int32))
        assert np.allclose(two_dist, np.array([[4.0, 16.0]], dtype=np.float32))


def test_filtered_parity_strict_validations() -> None:
    """Verifies that non-finite query components and dimension mismatches trigger non-zero exit."""
    rust_bin = _resolve_rust_filtered_exact_knn_binary()

    base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, "val_base.fvecs")
        nan_query_file = os.path.join(tmpdir, "nan_query.fvecs")
        dim_query_file = os.path.join(tmpdir, "dim_query.fvecs")

        write_fvecs(base_file, base)
        # Query with NaN
        write_fvecs(nan_query_file, np.array([[np.nan, 1.0]], dtype=np.float32))
        # Query with wrong dimension (3 instead of 2)
        write_fvecs(dim_query_file, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

        # 1. NaN query rejection
        proc_nan = subprocess.run(
            [
                str(rust_bin),
                "--base-fvecs",
                base_file,
                "--query-fvecs",
                nan_query_file,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_nan.returncode != 0
        assert "non-finite value" in proc_nan.stderr or "non-finite value" in proc_nan.stdout

        # 2. Dimension mismatch rejection
        proc_dim = subprocess.run(
            [
                str(rust_bin),
                "--base-fvecs",
                base_file,
                "--query-fvecs",
                dim_query_file,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_dim.returncode != 0
        assert "dimension mismatch" in proc_dim.stderr.lower() or "dimension mismatch" in proc_dim.stdout.lower()
