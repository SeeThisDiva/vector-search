"""End-to-end integration tests for Milestone M3 Tasks T3.8 and T3.10.

Exercises the compiled Rust `phase5` binary (`--smoke --synthetic`), verifies:
- Versioned cache files (`.bin`, `.ivecs`, `.fvecs`, `.json`) under `--data-dir`
- Cell JSON provenance, Option C `num_queries`/`is_subquery_probe`, zero-overhead counters
- Strict `S1_post_uncapped` gating (`sigma >= 0.05` only)
- Sensitivity JSON (`h_max_sweep`, `overfetch_factor_sweep`, `opaque_eval_probes`)
- Resumable run skipping cleanly (`newly_measured_count == 0`, `resumed_skipped_count == 6`)
- `scripts/plot_phase5.py` chart generation (both multi-sigma + single-cell edge case, PNG + SVG)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.plot_phase5 import generate_all_phase5_charts

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CHART_STEMS: tuple[str, ...] = (
    "chart1_recall_vs_selectivity",
    "chart2_latency_vs_selectivity",
    "chart3_failure_rate_vs_selectivity",
    "chart4_matched_recall_crossover",
    "chart5_distance_evals_and_sensitivity",
)


def _ensure_phase5_release_binary() -> Path:
    exe_name = "phase5.exe" if os.name == "nt" else "phase5"
    bin_path = REPO_ROOT / "target" / "release" / exe_name
    if not bin_path.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "phase5"],
            cwd=REPO_ROOT,
            check=True,
        )
    assert bin_path.is_file(), f"Expected release binary at {bin_path}"
    return bin_path


def test_phase5_synthetic_pipeline_resume_and_charts(tmp_path: Path) -> None:
    """Run synthetic micro-matrix, resume skip, and plot_phase5.py multi-cell + single-cell."""
    phase5_bin = _ensure_phase5_release_binary()
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    charts_dir = tmp_path / "charts"

    base_cmd = [
        str(phase5_bin),
        "--smoke",
        "--synthetic",
        "--scale",
        "250",
        "--queries",
        "12",
        "--dim",
        "8",
        "--reps",
        "2",
        "--sigmas",
        "0.5,0.1,0.01",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(out_dir),
    ]

    subprocess.run([*base_cmd, "--include-sensitivity"], cwd=REPO_ROOT, check=True)

    # Check cache artifacts generated under synthetic cache namespace
    cache_files = list((data_dir / "phase5").rglob("*"))
    assert any(p.suffix == ".bin" for p in cache_files)
    assert any(p.suffix == ".ivecs" for p in cache_files)
    assert any(p.suffix == ".fvecs" for p in cache_files)
    assert any(p.suffix == ".json" for p in cache_files)

    summary_path = out_dir / "summary_n250.json"
    sens_path = out_dir / "sensitivity_n250.json"
    assert summary_path.is_file()
    assert sens_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["scale"] == 250
    assert summary["total_cells"] == 6
    assert summary["newly_measured_count"] == 6
    assert summary["resumed_skipped_count"] == 0
    assert summary["sensitivity_executed"] is True
    assert abs(float(summary["option_c_schedule"]["s1_uncapped_evaluated_sigma_min"]) - 0.05) < 1e-9

    sens = json.loads(sens_path.read_text(encoding="utf-8"))
    assert sens["default_opaque_eval_limit"] == 16384
    assert len(sens["sensitivity_arms"]) > 0
    assert len(sens["opaque_eval_probes"]) > 0

    # Verify S1_uncap scoping and per-arm invariants
    low_cell = json.loads((out_dir / "cell_n250_independent_s0.010000.json").read_text(encoding="utf-8"))
    high_cell = json.loads((out_dir / "cell_n250_correlated_s0.100000.json").read_text(encoding="utf-8"))

    assert low_cell["option_c_schedule"]["s1_uncapped_evaluated"] is False
    assert all(arm["strategy"] != "S1_post_uncapped" for arm in low_cell["arms"])

    assert high_cell["option_c_schedule"]["s1_uncapped_evaluated"] is True
    assert any(arm["strategy"] == "S1_post_uncapped" for arm in high_cell["arms"])
    assert high_cell["provenance"]["processor_affinity_mask"]

    for arm in high_cell["arms"]:
        assert arm["num_queries"] == 12
        assert arm["is_subquery_probe"] is False
        assert arm["reps"] == 2
        total_evals = float(arm["dist_evals_mean"])
        split_sum = float(arm["valid_dist_evals_mean"]) + float(arm["bridge_dist_evals_mean"])
        assert abs(total_evals - split_sum) < 1e-2

    # Re-run with --resume (no sensitivity flag) -> all 6 cells skipped cleanly
    subprocess.run([*base_cmd, "--resume"], cwd=REPO_ROOT, check=True)
    summary_resumed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_resumed["newly_measured_count"] == 0
    assert summary_resumed["resumed_skipped_count"] == 6

    # Chart generation across multi-cell + sensitivity payload
    created = generate_all_phase5_charts(
        results_dir=out_dir,
        output_dir=charts_dir,
        scale=250,
    )
    assert len(created) == 10
    for stem in EXPECTED_CHART_STEMS:
        png = charts_dir / f"{stem}.png"
        svg = charts_dir / f"{stem}.svg"
        assert png.is_file() and png.stat().st_size > 1000
        assert svg.is_file() and svg.stat().st_size > 1000

    # Single-cell CLI smoke test via scripts/plot_phase5.py
    single_dir = tmp_path / "single_cell"
    single_charts = tmp_path / "single_charts"
    single_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "cell_n250_correlated_s0.100000.json", single_dir / "cell_n250_correlated_s0.100000.json")

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_phase5.py",
            "--results-dir",
            str(single_dir),
            "--output-dir",
            str(single_charts),
            "--scale",
            "250",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    for stem in EXPECTED_CHART_STEMS:
        assert (single_charts / f"{stem}.png").is_file()
        assert (single_charts / f"{stem}.svg").is_file()
