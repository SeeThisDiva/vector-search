"""Unit and end-to-end smoke tests for Phase 7 FAISS harness and plotter (T5.2 & T5.6)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
import pytest

from scripts.benchmark_faiss import (
    compute_matched_recall_comparison,
    enforce_omp_threads,
    format_affinity_mask,
    get_process_affinity_mask,
    parse_affinity_mask,
    read_fvecs_prefix,
    read_ivecs_prefix,
    run_faiss_benchmark,
    set_process_affinity_mask,
)
from scripts.plot_phase7 import generate_all_phase7_charts
from src.hnsw_phase1.dataset import generate_synthetic_sift, write_fvecs, write_ivecs
from src.hnsw_phase1.exact_knn import ExactKNN
from src.hnsw_phase1.recall import compute_recall_at_k

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CHART_STEMS: tuple[str, ...] = (
    "chart1_recall_vs_p99",
    "chart2_recall_vs_qps",
    "chart3_matched_recall_ratios",
    "chart4_multithread_and_build",
)


def _pick_single_cpu_mask(system_mask: int) -> int:
    if system_mask & 0x0004:
        return 0x0004
    return system_mask & -system_mask


def test_windows_affinity_helper_and_restore() -> None:
    """Verify affinity parsing, 64-bit Set/GetProcessAffinityMask, and original mask restore."""
    assert parse_affinity_mask("0x0004") == 4
    assert parse_affinity_mask("0x0555") == 0x0555
    assert parse_affinity_mask("0xFFFF") == 0xFFFF
    assert format_affinity_mask(4) == "0x0004"

    with pytest.raises(ValueError):
        parse_affinity_mask("")
    with pytest.raises(ValueError):
        parse_affinity_mask("0")
    with pytest.raises(ValueError):
        set_process_affinity_mask(0)

    info = get_process_affinity_mask()
    if info is not None:
        orig_mask, sys_mask = info
        target_mask = _pick_single_cpu_mask(sys_mask)
        try:
            prev_mask, verified_mask = set_process_affinity_mask(target_mask)
            assert prev_mask == orig_mask
            assert verified_mask == target_mask
            post_info = get_process_affinity_mask()
            assert post_info is not None
            assert post_info[0] == target_mask
        finally:
            set_process_affinity_mask(orig_mask)
            restored = get_process_affinity_mask()
            assert restored is not None
            assert restored[0] == orig_mask

        with pytest.raises(ValueError):
            set_process_affinity_mask(sys_mask + 1)
    else:
        with pytest.raises(RuntimeError):
            set_process_affinity_mask(1)


def test_omp_set_num_threads_enforced() -> None:
    """Verify FAISS OpenMP thread enforcement and validation."""
    with pytest.raises(ValueError):
        enforce_omp_threads(0)

    orig = int(faiss.omp_get_max_threads())
    try:
        assert enforce_omp_threads(1) == 1
        assert int(faiss.omp_get_max_threads()) == 1
        assert enforce_omp_threads(2) == 2
        assert int(faiss.omp_get_max_threads()) == 2
    finally:
        enforce_omp_threads(max(1, orig))


def test_recall_parity_vs_faiss_index_flat_l2() -> None:
    """Verify compute_recall_at_k achieves exact parity against FAISS IndexFlatL2."""
    base, query = generate_synthetic_sift(
        num_base=300, num_query=30, dim=32, num_clusters=5, seed=42
    )
    k = 10

    exact = ExactKNN(metric="l2").fit(base)
    exact_ids, exact_dists = exact.search(query, k=k)

    flat = faiss.IndexFlatL2(32)
    flat.add(base)
    flat_dists, flat_ids = flat.search(query, k)

    assert np.array_equal(exact_ids, flat_ids)
    assert np.allclose(exact_dists, flat_dists, atol=1e-3, rtol=1e-4)
    assert compute_recall_at_k(flat_ids, exact_ids, k=k) == 1.0

    enforce_omp_threads(1)
    faiss_dyn: Any = faiss
    hnsw: Any = faiss.IndexHNSWFlat(32, 16, faiss.METRIC_L2)
    hnsw.hnsw.efConstruction = 80
    hnsw.hnsw.rng = faiss_dyn.RandomGenerator(42)
    hnsw.add(base)
    hnsw.hnsw.efSearch = 100
    _, hnsw_ids = hnsw.search(query, k)

    recall_harness = compute_recall_at_k(hnsw_ids, exact_ids, k=k)
    manual_overlap = float(
        sum(len(set(hnsw_ids[q, :k].tolist()) & set(exact_ids[q, :k].tolist())) for q in range(30))
    ) / float(30 * k)
    assert math.isclose(recall_harness, manual_overlap, abs_tol=1e-12)
    assert recall_harness >= 0.90


def test_prefix_fvecs_ivecs_readers(tmp_path: Path) -> None:
    """Verify prefix .fvecs and .ivecs reading avoids loading trailing records."""
    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((20, 16), dtype=np.float32)
    ids = rng.integers(0, 100, size=(20, 10), dtype=np.int32)

    f_path = tmp_path / "vecs.fvecs"
    i_path = tmp_path / "ids.ivecs"
    write_fvecs(str(f_path), vecs)
    write_ivecs(str(i_path), ids)

    sub_vecs = read_fvecs_prefix(f_path, max_rows=7)
    sub_ids = read_ivecs_prefix(i_path, max_rows=7)
    assert sub_vecs.shape == (7, 16)
    assert sub_ids.shape == (7, 10)
    assert np.array_equal(sub_vecs, vecs[:7])
    assert np.array_equal(sub_ids, ids[:7])

    with pytest.raises(ValueError):
        read_fvecs_prefix(f_path, max_rows=25)
    with pytest.raises(ValueError):
        read_ivecs_prefix(i_path, max_rows=25)


def test_mode1_mode2_ffi_delta_and_ndis_stats() -> None:
    """Verify Mode 1 batch, Mode 2 search_c, ffi_delta_us, memory accounting, and ndis stats."""
    report = run_faiss_benchmark(
        synthetic=True,
        scale=400,
        queries=40,
        dim=32,
        num_clusters=6,
        k=10,
        m=16,
        ef_construction=40,
        ef_search_list=(10, 40),
        threads=1,
        build_reps=1,
        query_reps=2,
        seed=42,
        compare_mode="single",
    )

    assert report["phase"] == 7
    assert report["engine"] == "faiss"
    assert report["faiss"]["version"] == faiss.__version__
    assert report["faiss"]["omp_max_threads"] == 1
    assert isinstance(report["faiss"]["simd_level_name"], str)
    assert isinstance(report["faiss"]["has_dynamic_dispatch"], bool)
    assert isinstance(report["faiss"]["compiled_simd_levels"], int)
    assert report["environment"]["power_mode"]

    mem = report["memory_profile"]
    expected_graph_bytes = mem["levels_bytes"] + mem["offsets_bytes"] + mem["neighbors_bytes"]
    assert mem["levels_bytes"] == 400 * 4
    assert mem["offsets_bytes"] == 401 * 8
    assert mem["graph_only_memory_bytes"] == expected_graph_bytes
    assert math.isclose(
        mem["graph_only_bytes_per_vector"],
        round(expected_graph_bytes / 400.0, 2),
        abs_tol=1e-6,
    )
    assert mem["peak_rss_mb"] is not None and mem["peak_rss_mb"] > 0.0
    assert report["index_build"]["build_time_s"] > 0.0
    assert report["index_build"]["build_qps"] > 0.0

    sweep = report["ef_search_sweep"]
    assert len(sweep) == 2
    for pt in sweep:
        m1 = pt["mode1_batch"]
        m2 = pt["mode2_per_query"]
        assert m1["qps"] > 0.0
        assert m1["mean_ms"] > 0.0
        assert len(m1["run_qps"]) == 2
        assert m2 is not None
        assert 0.0 <= m2["min_ms"] <= m2["mean_ms"] <= m2["max_ms"]
        assert m2["p50_ms"] <= m2["p95_ms"] <= m2["p99_ms"]
        assert len(m2["run_p99_ms"]) == 2
        assert math.isclose(pt["recall_at_k"], pt["recall_at_k_mode2"], abs_tol=1e-9)
        assert pt["dist_evals_mean"] > 0.0
        expected_ffi_delta = (m2["total_s"] - m1["total_s"]) * 1_000_000.0 / 40.0
        assert math.isclose(pt["ffi_delta_us"], expected_ffi_delta, abs_tol=0.05)

    assert sweep[1]["dist_evals_mean"] >= sweep[0]["dist_evals_mean"]


def test_matched_recall_interpolation_and_no_extrapolation() -> None:
    """Verify log-linear interpolation inside [min, max] recall and strict no-extrapolation."""
    rust_curve = [
        {"ef_search": 10, "recall_at_k": 0.90, "mode1_qps": 1000.0, "mode2_p99_ms": 1.0},
        {"ef_search": 40, "recall_at_k": 0.98, "mode1_qps": 250.0, "mode2_p99_ms": 4.0},
        {"ef_search": 100, "recall_at_k": 0.995, "mode1_qps": 100.0, "mode2_p99_ms": 10.0},
    ]
    faiss_curve = [
        {"ef_search": 10, "recall_at_k": 0.90, "mode1_qps": 500.0, "mode2_p99_ms": 2.0},
        {"ef_search": 40, "recall_at_k": 0.98, "mode1_qps": 125.0, "mode2_p99_ms": 8.0},
    ]

    results = compute_matched_recall_comparison(
        rust_curve,
        faiss_curve,
        targets=(0.85, 0.90, 0.94, 0.98, 0.995, 0.999),
    )
    by_target = {r["target_recall"]: r for r in results}

    # Below min_recall (0.85) -> strictly out_of_range, no extrapolation
    assert by_target[0.85]["status"] == "out_of_range"
    assert by_target[0.85]["rust"] is None
    assert by_target[0.85]["faiss"] is None
    assert by_target[0.85]["qps_ratio_rust_over_faiss"] is None

    # Exact endpoint 0.90 -> matched
    assert by_target[0.90]["status"] == "matched"
    assert math.isclose(by_target[0.90]["rust"]["interpolated_qps"], 1000.0, rel_tol=1e-9)
    assert math.isclose(by_target[0.90]["qps_ratio_rust_over_faiss"], 2.0, rel_tol=1e-9)

    # Midpoint 0.94 (alpha=0.5) -> geometric mean sqrt(1000*250)=500 QPS, sqrt(1*4)=2 ms p99
    mid = by_target[0.94]
    assert mid["status"] == "matched"
    assert math.isclose(mid["rust"]["interpolated_qps"], 500.0, rel_tol=1e-9)
    assert math.isclose(mid["rust"]["interpolated_p99_ms"], 2.0, rel_tol=1e-9)
    assert mid["rust"]["upper_bracket"]["ef_search"] == 40
    assert mid["rust"]["upper_bracket"]["recall_at_k"] == 0.98
    assert math.isclose(mid["faiss"]["interpolated_qps"], 250.0, rel_tol=1e-9)
    assert math.isclose(mid["faiss"]["interpolated_p99_ms"], 4.0, rel_tol=1e-9)
    assert math.isclose(mid["qps_ratio_rust_over_faiss"], 2.0, rel_tol=1e-9)
    assert math.isclose(mid["p99_ratio_faiss_over_rust"], 2.0, rel_tol=1e-9)

    # Exact endpoint 0.98 -> matched
    assert by_target[0.98]["status"] == "matched"
    assert math.isclose(by_target[0.98]["rust"]["interpolated_qps"], 250.0, rel_tol=1e-9)

    # 0.995 attained by Rust only, not FAISS -> no extrapolation for FAISS, ratio is None
    asym = by_target[0.995]
    assert asym["rust_status"] == "in_range"
    assert asym["faiss_status"] == "out_of_range"
    assert asym["status"] == "out_of_range"
    assert asym["faiss"] is None
    assert asym["qps_ratio_rust_over_faiss"] is None

    # 0.999 unattained by both -> strictly out_of_range
    assert by_target[0.999]["status"] == "out_of_range"
    assert by_target[0.999]["rust"] is None
    assert by_target[0.999]["faiss"] is None


def test_cli_synthetic_end_to_end_and_plot_phase7(tmp_path: Path) -> None:
    """Run CLI benchmark (--synthetic, --profile-rss-isolated, multithread) and plot_phase7.py."""
    results_dir = tmp_path / "results"
    charts_dir = tmp_path / "charts"
    faiss_only_dir = tmp_path / "faiss_only"
    faiss_only_charts = tmp_path / "faiss_only_charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    faiss_only_dir.mkdir(parents=True, exist_ok=True)

    faiss_single_json = results_dir / "faiss_c2_n400.json"
    faiss_mt_json = results_dir / "faiss_c2_n400_t2.json"

    single_cmd = [
        sys.executable,
        "scripts/benchmark_faiss.py",
        "--synthetic",
        "--scale",
        "400",
        "--queries",
        "40",
        "--dim",
        "32",
        "--k",
        "10",
        "--m",
        "16",
        "--ef-construction",
        "40",
        "--ef-search-list",
        "10,30,60",
        "--threads",
        "1",
        "--build-reps",
        "1",
        "--query-reps",
        "2",
        "--profile-rss-isolated",
        "--json-out",
        str(faiss_single_json),
    ]
    affinity_info = get_process_affinity_mask()
    expected_affinity = "not_set"
    if affinity_info is not None:
        _, sys_mask = affinity_info
        target_mask = _pick_single_cpu_mask(sys_mask)
        expected_affinity = format_affinity_mask(target_mask)
        single_cmd.extend(["--affinity", expected_affinity])

    subprocess.run(single_cmd, cwd=str(REPO_ROOT), check=True)
    assert faiss_single_json.is_file()

    faiss_single = json.loads(faiss_single_json.read_text(encoding="utf-8"))
    assert faiss_single["isolated_process"] is True
    assert faiss_single["memory_profile"]["peak_rss_scope"] == "isolated_child_process_high_water_mark"
    assert faiss_single["dataset"]["num_base"] == 400
    assert faiss_single["dataset"]["num_query"] == 40
    assert len(faiss_single["ef_search_sweep"]) == 3
    if affinity_info is not None:
        assert faiss_single["environment"]["affinity_pinned"] is True
        assert faiss_single["environment"]["processor_affinity_mask"] == expected_affinity

    # Also test partial / FAISS-only chart generation into tmp_path
    faiss_only_copy = faiss_only_dir / "faiss_c2_n400.json"
    faiss_only_copy.write_text(faiss_single_json.read_text(encoding="utf-8"), encoding="utf-8")
    partial_created = generate_all_phase7_charts(
        results_dir=faiss_only_dir,
        output_dir=faiss_only_charts,
    )
    assert len(partial_created) == 8
    for p in partial_created:
        assert p.is_file() and p.stat().st_size > 1000

    # Run multithread FAISS CLI arm
    mt_cmd = [
        sys.executable,
        "scripts/benchmark_faiss.py",
        "--synthetic",
        "--scale",
        "400",
        "--queries",
        "40",
        "--dim",
        "32",
        "--k",
        "10",
        "--m",
        "16",
        "--ef-construction",
        "40",
        "--ef-search-list",
        "10,30,60",
        "--threads",
        "2",
        "--build-reps",
        "1",
        "--query-reps",
        "2",
        "--compare-mode",
        "multithread",
        "--json-out",
        str(faiss_mt_json),
    ]
    subprocess.run(mt_cmd, cwd=str(REPO_ROOT), check=True)
    faiss_mt = json.loads(faiss_mt_json.read_text(encoding="utf-8"))
    assert faiss_mt["compare_mode"] == "multithread"
    assert faiss_mt["threads"] == 2
    assert all(pt["mode2_per_query"] is None for pt in faiss_mt["ef_search_sweep"])

    # Create synthetic paired Rust single-thread + multithread reports so all 4 charts plot full comparisons
    rust_single = {
        "phase": 7,
        "engine": "rust",
        "compare_mode": "single",
        "config_id": "M16_efC40",
        "git_commit": faiss_single["git_commit"],
        "git_dirty": False,
        "environment": {
            "processor_affinity_mask": expected_affinity,
            "power_mode": faiss_single["environment"]["power_mode"],
        },
        "hardware": faiss_single["hardware"],
        "scale_vectors": 400,
        "threads": 1,
        "build_reps": 1,
        "query_reps": 2,
        "hnsw_config": {"m": 16, "ef_construction": 40, "keep_pruned_shrink": None},
        "index_build": {"build_time_s": 0.015, "build_qps": 26666.7},
        "memory_profile": {"graph_only_bytes_per_vector": 146.0, "peak_rss_mb": 12.5},
        "ef_search_sweep": [
            {
                "ef_search": 10,
                "recall_at_k": 0.88,
                "mode1_batch": {"qps": 18000.0, "mean_ms": 0.055},
                "mode2_per_query": {"p99_ms": 0.14, "mean_ms": 0.06},
            },
            {
                "ef_search": 30,
                "recall_at_k": 0.96,
                "mode1_batch": {"qps": 9000.0, "mean_ms": 0.111},
                "mode2_per_query": {"p99_ms": 0.28, "mean_ms": 0.12},
            },
            {
                "ef_search": 60,
                "recall_at_k": 0.995,
                "mode1_batch": {"qps": 5000.0, "mean_ms": 0.200},
                "mode2_per_query": {"p99_ms": 0.45, "mean_ms": 0.21},
            },
        ],
    }
    rust_mt = {
        **rust_single,
        "compare_mode": "multithread",
        "threads": 2,
        "ef_search_sweep": [
            {"ef_search": 10, "recall_at_k": 0.88, "mode1_batch": {"qps": 31000.0, "mean_ms": 0.032}},
            {"ef_search": 30, "recall_at_k": 0.96, "mode1_batch": {"qps": 15500.0, "mean_ms": 0.064}},
            {"ef_search": 60, "recall_at_k": 0.995, "mode1_batch": {"qps": 8800.0, "mean_ms": 0.113}},
        ],
    }
    (results_dir / "rust_c2_n400.json").write_text(json.dumps(rust_single, indent=2), encoding="utf-8")
    (results_dir / "rust_c2_n400_t2.json").write_text(json.dumps(rust_mt, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_phase7.py",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(charts_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    for stem in EXPECTED_CHART_STEMS:
        png = charts_dir / f"{stem}.png"
        svg = charts_dir / f"{stem}.svg"
        assert png.is_file() and png.stat().st_size > 1000
        assert svg.is_file() and svg.stat().st_size > 1000
