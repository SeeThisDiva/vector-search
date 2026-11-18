"""Cross-language parity tests verifying the Rust Phase 1 binary against Python & FAISS IndexFlatL2."""

import json
import os
import subprocess
import tempfile

import faiss  # type: ignore[import-untyped]
import numpy as np

from src.hnsw_phase1.dataset import read_fvecs, read_ivecs, write_fvecs
from src.hnsw_phase1.exact_knn import ExactKNN
from src.hnsw_phase1.recall import compute_recall_at_k
from tests.helpers import REPO_ROOT, resolve_rust_binary


def test_rust_synthetic_fvecs_ivecs_and_faiss_parity() -> None:
    """Verify Rust synthetic vector generation, Texmex binary I/O, JSON report, and FAISS parity."""
    rust_bin = resolve_rust_binary("phase1")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_json_path = os.path.join(tmpdir, "report.json")
        proc = subprocess.run(
            [
                str(rust_bin),
                "--dataset",
                "synthetic",
                "--num-base",
                "1000",
                "--num-query",
                "40",
                "--dim",
                "64",
                "--k",
                "10",
                "--seed",
                "101",
                "--output-dir",
                tmpdir,
                "--json-out",
                report_json_path,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Rust phase1 failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        # 1. Verify structured JSON report & hardware/latency metadata
        with open(report_json_path, encoding="utf-8") as f:
            report = json.load(f)

        assert report["hardware"]["logical_cores"] >= 1
        assert len(report["hardware"]["cpu_model"]) > 0
        assert report["dataset"]["num_base"] == 1000
        assert report["dataset"]["num_query"] == 40
        assert report["dataset"]["dim"] == 64
        assert report["recall"]["self_verified"] is True
        assert report["recall"]["self_recall_at_k"] == 1.0
        assert report["recall"]["metrics"]["Recall@10"] == 1.0
        assert report["exact_knn"]["qps"] > 0.0
        assert 0.0 <= report["exact_knn"]["p50_ms"] <= report["exact_knn"]["p95_ms"]
        assert report["exact_knn"]["p95_ms"] <= report["exact_knn"]["p99_ms"]

        # 2. Load Rust's Texmex binary outputs in Python
        base = read_fvecs(os.path.join(tmpdir, "base.fvecs"))
        query = read_fvecs(os.path.join(tmpdir, "query.fvecs"))
        rust_indices = read_ivecs(os.path.join(tmpdir, "groundtruth.ivecs"))
        rust_distances = read_fvecs(os.path.join(tmpdir, "groundtruth.fvecs"))

        assert base.shape == (1000, 64)
        assert query.shape == (40, 64)
        assert rust_indices.shape == (40, 10)
        assert rust_distances.shape == (40, 10)

        # 3. Verify 100% parity against Python ExactKNN
        py_knn = ExactKNN(metric="l2").fit(base)
        py_indices, py_distances = py_knn.search(query, k=10)
        assert np.array_equal(rust_indices, py_indices)
        assert np.allclose(rust_distances, py_distances, atol=1e-2, rtol=1e-4)

        # 4. Verify 100% parity against FAISS IndexFlatL2
        index = faiss.IndexFlatL2(64)
        index.add(base)
        faiss_distances, faiss_indices = index.search(query, 10)

        assert compute_recall_at_k(rust_indices, faiss_indices, k=10) == 1.0
        assert np.array_equal(rust_indices, faiss_indices)
        assert np.allclose(rust_distances, faiss_distances, atol=1e-2, rtol=1e-4)


def test_rust_custom_tie_breaking_and_zero_clamping() -> None:
    """Verify Rust canonical (distance, id) tie-breaking and zero clamping on Python-written .fvecs."""
    rust_bin = resolve_rust_binary("phase1")

    base = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],  # id 0, d^2 = 25 to query 0
            [2.0, 0.0, 0.0, 0.0],  # id 1, d^2 = 4 to query 0
            [5.0, 0.0, 0.0, 0.0],  # id 2, d^2 = 25 to query 0
            [-2.0, 0.0, 0.0, 0.0],  # id 3, d^2 = 4 to query 0
            [0.0, 2.0, 0.0, 0.0],  # id 4, d^2 = 4 to query 0
        ],
        dtype=np.float32,
    )
    query = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # tied with IDs 1, 3, 4 at d^2 = 4.0
            [5.0, 0.0, 0.0, 0.0],  # identical to IDs 0 and 2 at d^2 = 0.0
        ],
        dtype=np.float32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_base = os.path.join(tmpdir, "in_base.fvecs")
        in_query = os.path.join(tmpdir, "in_query.fvecs")
        out_dir = os.path.join(tmpdir, "out")
        write_fvecs(in_base, base)
        write_fvecs(in_query, query)

        proc = subprocess.run(
            [
                str(rust_bin),
                "--base-fvecs",
                in_base,
                "--query-fvecs",
                in_query,
                "--k",
                "3",
                "--output-dir",
                out_dir,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr

        rust_idx = read_ivecs(os.path.join(out_dir, "groundtruth.ivecs"))
        rust_dist = read_fvecs(os.path.join(out_dir, "groundtruth.fvecs"))

        assert np.array_equal(rust_idx[0], [1, 3, 4])
        assert np.allclose(rust_dist[0], [4.0, 4.0, 4.0])

        assert np.array_equal(rust_idx[1, :2], [0, 2])
        assert rust_dist[1, 0] == 0.0
        assert rust_dist[1, 1] == 0.0


def test_rust_siftsmall_vs_faiss_parity() -> None:
    """Verify Rust ExactKNN on authentic cached siftsmall (or offline fallback) matches FAISS 100%."""
    rust_bin = resolve_rust_binary("phase1")
    siftsmall_cached = (REPO_ROOT / "data" / "siftsmall" / "siftsmall_base.fvecs").is_file()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "siftsmall_report.json")
        cmd = [
            str(rust_bin),
            "--dataset",
            "siftsmall",
            "--data-dir",
            str(REPO_ROOT / "data"),
            "--output-dir",
            tmpdir,
            "--k",
            "10",
            "--json-out",
            report_path,
        ]
        if not siftsmall_cached:
            cmd.append("--offline")

        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        assert report["dataset"]["num_base"] == 10_000
        assert report["dataset"]["num_query"] == 100
        assert report["dataset"]["dim"] == 128
        if siftsmall_cached:
            assert report["dataset"]["is_authentic_sift"] is True
            assert report["official_ground_truth"]["recall_at_k"] == 1.0

        base = read_fvecs(os.path.join(tmpdir, "base.fvecs"))
        query = read_fvecs(os.path.join(tmpdir, "query.fvecs"))
        rust_idx = read_ivecs(os.path.join(tmpdir, "groundtruth.ivecs"))
        rust_dist = read_fvecs(os.path.join(tmpdir, "groundtruth.fvecs"))

        index = faiss.IndexFlatL2(128)
        index.add(base)
        faiss_dist, faiss_idx = index.search(query, 10)

        assert compute_recall_at_k(rust_idx, faiss_idx, k=10) == 1.0
        assert np.allclose(rust_dist, faiss_dist, atol=1e-1, rtol=1e-4)


def test_rust_cli_invalid_arguments_rejected() -> None:
    """Verify Rust CLI rejects invalid flags and zero k cleanly without panicking."""
    rust_bin = resolve_rust_binary("phase1")

    for bad_args in (
        ["--k", "0", "--offline", "--no-save"],
        ["--dataset", "nonexistent_corpus", "--no-save"],
        ["--bogus-flag"],
    ):
        proc = subprocess.run(
            [str(rust_bin), *bad_args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        assert "[ERROR]" in proc.stderr
        assert "panicked" not in proc.stderr.lower()
