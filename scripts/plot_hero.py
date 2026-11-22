#!/usr/bin/env python3
"""Hero Chart Generator: Safe Rust HNSW vs FAISS on SIFT1M (1M vectors).

Reads ONLY the seven Phase 7 single-thread 1M benchmark JSON reports:
  - results/phase7/rust_n1000000_{C1,C2,C2-nkp,C3}.json
  - results/phase7/faiss_n1000000_{C1,C2,C3}.json

Produces ``results/hero_recall_vs_latency.png`` (200 DPI) and
``results/hero_recall_vs_latency.svg`` showing:
  1. Left panel: Recall@10 vs single-thread median (p50) per-query latency [ms]
     (log scale) as two strictly monotonic lower-envelope Pareto frontiers
     (Rust vs FAISS).
  2. Right side panel: Matched M=16, efC=100 (C2 / C2-nkp) 1-thread build time,
     isolated peak RSS, and graph-only bytes/vector.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.plot_common import (
    OKABE_ITO_BLUE,
    OKABE_ITO_ORANGE,
    OKABE_ITO_SKY,
    compute_pareto_frontier,
    configure_ax,
    configure_matplotlib,
    safe_float,
    safe_int,
    safe_opt_float,
    save_figure,
)

configure_matplotlib(svg_hashsalt="hero-sift1m-pareto", font_family="DejaVu Sans")

RUST_1M_FILES: tuple[tuple[str, str], ...] = (
    ("C1", "rust_n1000000_C1.json"),
    ("C2", "rust_n1000000_C2.json"),
    ("C2-nkp", "rust_n1000000_C2-nkp.json"),
    ("C3", "rust_n1000000_C3.json"),
)
FAISS_1M_FILES: tuple[tuple[str, str], ...] = (
    ("C1", "faiss_n1000000_C1.json"),
    ("C2", "faiss_n1000000_C2.json"),
    ("C3", "faiss_n1000000_C3.json"),
)

# Carefully chosen anchor (config_id, ef_search) markers to label lightly without overlaps.
# Rust labels sit up-and-left (perpendicular away from the rising curve);
# FAISS labels sit down-and-right (or left near x=1.0).
RUST_LABEL_ANCHORS: dict[tuple[str, int], tuple[tuple[int, int], str]] = {
    ("C2-nkp", 20): ((-5, 6), "right"),
    ("C2-nkp", 30): ((-5, 6), "right"),
    ("C2-nkp", 40): ((-5, 6), "right"),
    ("C3", 30): ((-6, 6), "right"),
    ("C3", 50): ((-6, 6), "right"),
    ("C3", 100): ((-8, 4), "right"),
    ("C3", 200): ((-9, 2), "right"),
    ("C3", 600): ((-9, -2), "right"),
}
FAISS_LABEL_ANCHORS: dict[tuple[str, int], tuple[tuple[int, int], str]] = {
    ("C2", 20): ((5, -6), "left"),
    ("C3", 20): ((5, -7), "left"),
    ("C3", 30): ((5, -7), "left"),
    ("C3", 40): ((5, -7), "left"),
    ("C3", 70): ((6, -12), "center"),
    ("C3", 600): ((-9, -6), "right"),
}


@dataclass(frozen=True)
class SweepPoint:
    """Single efSearch operating point parsed from a Phase 7 1M JSON."""

    engine: str
    config_id: str
    ef_search: int
    recall_at_k: float
    p50_ms: float
    mode2_mean_ms: float
    mode2_p99_ms: float
    mode1_mean_ms: float
    mode1_qps: float
    ffi_delta_us: float | None


@dataclass(frozen=True)
class ReportSummary:
    """Build, memory, and execution metadata parsed from a Phase 7 1M JSON."""

    engine: str
    config_id: str
    m: int
    ef_construction: int
    threads: int
    env_threads: int
    affinity_mask: str
    cpu_model: str
    num_base: int
    num_query: int
    dim: int
    k: int
    ground_truth_path: str
    build_time_s: float
    peak_rss_mb: float
    graph_only_bytes_per_vector: float
    faiss_version: str | None
    faiss_index_class: str | None
    faiss_simd_level_name: str | None
    faiss_avx2_available: bool | None
    faiss_omp_max_threads: int | None
    points: tuple[SweepPoint, ...]


def _as_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    val = payload.get(key)
    return val if isinstance(val, dict) else {}


def load_report(path: Path, expected_engine: str, expected_config: str) -> ReportSummary:
    """Load and validate one Phase 7 1M single-thread JSON report."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected JSON object in {path}")

    engine = str(raw.get("engine", expected_engine)).strip().lower()
    config_id = str(raw.get("config_id", expected_config)).strip()
    env = _as_dict(raw, "environment")
    hw = _as_dict(raw, "hardware")
    ds = _as_dict(raw, "dataset")
    hnsw = _as_dict(raw, "hnsw_config")
    ib = _as_dict(raw, "index_build")
    mem = _as_dict(raw, "memory_profile")
    fx = _as_dict(raw, "faiss")

    threads = safe_int(raw.get("threads"), default=0)
    env_threads = safe_int(env.get("threads"), default=0)
    if threads != 1 or env_threads != 1:
        raise ValueError(
            f"Expected 1-thread report in {path}, got threads={threads}, env.threads={env_threads}"
        )

    omp_max = safe_int(fx.get("omp_max_threads"), default=1) if fx else None
    if omp_max is not None and omp_max != 1:
        raise ValueError(f"Expected omp_max_threads=1 in {path}, got {omp_max}")

    raw_sweep = raw.get("ef_search_sweep")
    if not isinstance(raw_sweep, list) or not raw_sweep:
        raise ValueError(f"Missing ef_search_sweep in {path}")

    points: list[SweepPoint] = []
    for entry in raw_sweep:
        if not isinstance(entry, dict):
            continue
        m1 = _as_dict(entry, "mode1_batch")
        m2 = _as_dict(entry, "mode2_per_query")
        points.append(
            SweepPoint(
                engine=engine,
                config_id=config_id,
                ef_search=safe_int(entry.get("ef_search")),
                recall_at_k=safe_float(entry.get("recall_at_k")),
                p50_ms=safe_float(m2.get("p50_ms")),
                mode2_mean_ms=safe_float(m2.get("mean_ms")),
                mode2_p99_ms=safe_float(m2.get("p99_ms")),
                mode1_mean_ms=safe_float(m1.get("mean_ms")),
                mode1_qps=safe_float(m1.get("qps")),
                ffi_delta_us=safe_opt_float(entry.get("ffi_delta_us")),
            )
        )

    return ReportSummary(
        engine=engine,
        config_id=config_id,
        m=safe_int(hnsw.get("m")),
        ef_construction=safe_int(hnsw.get("ef_construction")),
        threads=threads,
        env_threads=env_threads,
        affinity_mask=str(env.get("processor_affinity_mask", "n/a")),
        cpu_model=str(hw.get("cpu_model", env.get("cpu_model", "Unknown CPU"))),
        num_base=safe_int(ds.get("num_base")),
        num_query=safe_int(ds.get("num_query")),
        dim=safe_int(ds.get("dim")),
        k=safe_int(ds.get("k")),
        ground_truth_path=str(raw.get("ground_truth_path", "data/sift1m/sift_groundtruth.ivecs")),
        build_time_s=safe_float(ib.get("build_time_s")),
        peak_rss_mb=safe_float(mem.get("peak_rss_mb")),
        graph_only_bytes_per_vector=safe_float(
            mem.get("graph_only_bytes_per_vector", raw.get("graph_only_bytes_per_vector"))
        ),
        faiss_version=str(fx["version"]) if "version" in fx else None,
        faiss_index_class=str(fx["index_class"]) if "index_class" in fx else None,
        faiss_simd_level_name=str(fx["simd_level_name"]) if "simd_level_name" in fx else None,
        faiss_avx2_available=(
            bool(fx["avx2_available"]) if "avx2_available" in fx else None
        ),
        faiss_omp_max_threads=omp_max,
        points=tuple(points),
    )


def load_all_1m_reports(phase7_dir: Path) -> tuple[list[ReportSummary], list[ReportSummary]]:
    """Load the 4 Rust and 3 FAISS 1M JSON reports from ``phase7_dir``."""
    rust_reports = [
        load_report(phase7_dir / filename, "rust", cfg_id)
        for cfg_id, filename in RUST_1M_FILES
    ]
    faiss_reports = [
        load_report(phase7_dir / filename, "faiss", cfg_id)
        for cfg_id, filename in FAISS_1M_FILES
    ]
    return rust_reports, faiss_reports


def _plot_frontier_panel(
    ax: Axes,
    rust_frontier: list[SweepPoint],
    faiss_frontier: list[SweepPoint],
) -> None:
    """Render the main Recall@10 vs Median (p50) Per-Query Latency Pareto curves."""
    configure_ax(ax, log_x=False, log_y=True, minor_alpha=0.16)

    r_xs = [pt.recall_at_k for pt in rust_frontier]
    r_ys = [pt.p50_ms for pt in rust_frontier]
    f_xs = [pt.recall_at_k for pt in faiss_frontier]
    f_ys = [pt.p50_ms for pt in faiss_frontier]

    ax.plot(
        r_xs,
        r_ys,
        color=OKABE_ITO_BLUE,
        linestyle="-",
        linewidth=2.4,
        marker="o",
        markersize=5.6,
        markerfacecolor=OKABE_ITO_BLUE,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
        label="Rust HnswIndex Pareto frontier (C1, C2, C2-nkp, C3)",
    )
    ax.plot(
        f_xs,
        f_ys,
        color=OKABE_ITO_ORANGE,
        linestyle="--",
        linewidth=2.2,
        marker="^",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=OKABE_ITO_ORANGE,
        markeredgewidth=1.6,
        zorder=4,
        label="FAISS IndexHNSWFlat Pareto frontier (C1, C2, C3)",
    )

    x_min = 0.80
    x_max = 1.004
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.066, 4.8)

    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))

    y_ticks = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.5", "1.0", "2.0", "4.0"]))

    for pt in rust_frontier:
        spec = RUST_LABEL_ANCHORS.get((pt.config_id, pt.ef_search))
        if spec is None or pt.recall_at_k < x_min:
            continue
        offset, ha = spec
        ax.annotate(
            f"{pt.config_id} ef={pt.ef_search}",
            xy=(pt.recall_at_k, pt.p50_ms),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=7.3,
            color="#0B4F7A",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            zorder=6,
        )

    for pt in faiss_frontier:
        spec = FAISS_LABEL_ANCHORS.get((pt.config_id, pt.ef_search))
        if spec is None or pt.recall_at_k < x_min:
            continue
        offset, ha = spec
        ax.annotate(
            f"{pt.config_id} ef={pt.ef_search}",
            xy=(pt.recall_at_k, pt.p50_ms),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=7.3,
            color="#9E4200",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            zorder=6,
        )

    ax.set_xlabel("Recall@10", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(
        "Single-Thread Median (p50) Per-Query Latency [ms] (log scale)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.set_title(
        "Recall@10 vs Per-Query Latency Pareto Frontier (Lower is Better)",
        fontsize=11.0,
        fontweight="bold",
        pad=8,
    )
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.94,
        fontsize=8.6,
    )


def _plot_side_bars(
    ax: Axes,
    labels: list[str],
    values: list[float],
    colors: list[str],
    hatches: list[str],
    value_fmt: str,
    title: str,
    xlabel: str,
    xlim_scale: float,
    annotation: str | None = None,
) -> None:
    """Render one compact horizontal bar comparison panel in the right column."""
    configure_ax(ax, log_x=False, log_y=False, minor_alpha=0.0)
    ax.grid(axis="y", visible=False)
    ys = list(range(len(values)))
    bars = ax.barh(
        ys,
        values,
        height=0.58,
        color=colors,
        edgecolor="#222222",
        linewidth=0.75,
        alpha=0.90,
        zorder=4,
    )
    for bar, hatch, val in zip(bars, hatches, values, strict=True):
        if hatch:
            bar.set_hatch(hatch)
        ax.text(
            val + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2.0,
            value_fmt.format(val),
            va="center",
            ha="left",
            fontsize=8.1,
            fontweight="bold",
            color="#222222",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(values) * xlim_scale)
    ax.set_xlabel(xlabel, fontsize=8.6, fontweight="bold")
    ax.set_title(title, fontsize=9.6, fontweight="bold", pad=5)
    if annotation:
        ax.text(
            0.98,
            0.67,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.6,
            color="#0B4F7A",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#F0F8FF",
                "edgecolor": OKABE_ITO_BLUE,
                "alpha": 0.94,
            },
            zorder=6,
        )


def _plot_matched_c2_side_panels(
    axes_side: tuple[Axes, Axes, Axes],
    rust_c2: ReportSummary,
    rust_c2_nkp: ReportSummary,
    faiss_c2: ReportSummary,
) -> None:
    """Render build time, peak RSS, and graph B/vec panels for M=16, efC=100."""
    labels = [
        f"Rust C2 (default, {rust_c2.threads}T)",
        f"Rust C2-nkp ({rust_c2_nkp.threads}T)",
        f"FAISS C2 ({faiss_c2.threads}T)",
    ]
    colors = [OKABE_ITO_BLUE, OKABE_ITO_SKY, OKABE_ITO_ORANGE]
    hatches = ["", "", "//"]

    _plot_side_bars(
        axes_side[0],
        labels=labels,
        values=[rust_c2.build_time_s, rust_c2_nkp.build_time_s, faiss_c2.build_time_s],
        colors=colors,
        hatches=hatches,
        value_fmt="{:.1f} s",
        title="Matched M=16, efC=100: Build Time (1-Thread)",
        xlabel="Wall-clock seconds [s]",
        xlim_scale=1.24,
    )

    rss_gap = faiss_c2.peak_rss_mb - rust_c2.peak_rss_mb
    _plot_side_bars(
        axes_side[1],
        labels=labels,
        values=[rust_c2.peak_rss_mb, rust_c2_nkp.peak_rss_mb, faiss_c2.peak_rss_mb],
        colors=colors,
        hatches=hatches,
        value_fmt="{:,.1f} MiB",
        title="Isolated Peak RSS (PeakWorkingSet64)",
        xlabel="Peak working set [MiB]",
        xlim_scale=1.62,
        annotation=f"Gap: {rss_gap:.2f} MiB\n(~1 base-matrix copy)",
    )

    _plot_side_bars(
        axes_side[2],
        labels=labels,
        values=[
            rust_c2.graph_only_bytes_per_vector,
            rust_c2_nkp.graph_only_bytes_per_vector,
            faiss_c2.graph_only_bytes_per_vector,
        ],
        colors=colors,
        hatches=hatches,
        value_fmt="{:.2f} B/v",
        title="Graph-Only Memory Footprint",
        xlabel="Bytes per vector [B/vec]",
        xlim_scale=1.28,
    )


def _build_footer_text(
    rust_reports: list[ReportSummary],
    faiss_reports: list[ReportSummary],
) -> str:
    rep = rust_reports[0]
    fx = faiss_reports[0]
    ffi_deltas = [
        pt.ffi_delta_us
        for rpt in faiss_reports
        for pt in rpt.points
        if pt.ffi_delta_us is not None
    ]
    ffi_med = statistics.median(ffi_deltas) if ffi_deltas else 0.0
    avx2_str = "false" if fx.faiss_avx2_available is False else str(fx.faiss_avx2_available)
    gt_name = Path(rep.ground_truth_path).name

    line1 = (
        f"CPU: {rep.cpu_model} | Dataset: SIFT1M ({rep.num_base:,} x {rep.dim}-d float32, "
        f"{rep.num_query:,} queries, k={rep.k}, L2) | "
        f"Ground truth: exact brute force / Texmex ({gt_name})"
    )
    line2 = (
        f"FAISS {fx.faiss_version} {fx.faiss_index_class} "
        f"(SIMD={fx.faiss_simd_level_name} / SSE2 baseline, avx2_available={avx2_str}, "
        f"Mode 2 median FFI delta {ffi_med:.2f} us/q) vs "
        f"Rust #![forbid(unsafe_code)] std-only | All runs 1-thread pinned"
    )
    return f"{line1}\n{line2}"


def generate_hero_chart(
    phase7_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 200,
) -> list[Path]:
    """Build the SIFT1M hero Pareto frontier + matched build/memory figure."""
    rust_reports, faiss_reports = load_all_1m_reports(phase7_dir)

    rust_points = [pt for rpt in rust_reports for pt in rpt.points]
    faiss_points = [pt for rpt in faiss_reports for pt in rpt.points]

    rust_frontier = compute_pareto_frontier(
        rust_points,
        x_fn=lambda p: p.recall_at_k,
        y_fn=lambda p: p.p50_ms,
        maximize_x=True,
        maximize_y=False,
    )
    faiss_frontier = compute_pareto_frontier(
        faiss_points,
        x_fn=lambda p: p.recall_at_k,
        y_fn=lambda p: p.p50_ms,
        maximize_x=True,
        maximize_y=False,
    )

    rust_by_cfg = {rpt.config_id: rpt for rpt in rust_reports}
    faiss_by_cfg = {rpt.config_id: rpt for rpt in faiss_reports}

    fig = plt.figure(figsize=(13.6, 6.5))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[2.15, 1.05],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.34,
        hspace=0.58,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    axes_side = (
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 1]),
    )

    _plot_frontier_panel(ax_main, rust_frontier, faiss_frontier)
    _plot_matched_c2_side_panels(
        axes_side,
        rust_by_cfg["C2"],
        rust_by_cfg["C2-nkp"],
        faiss_by_cfg["C2"],
    )

    fig.suptitle(
        "Safe Rust HNSW vs FAISS on SIFT1M (1M Vectors, Single-Thread)",
        fontsize=13.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.012,
        _build_footer_text(rust_reports, faiss_reports),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.068, right=0.965, top=0.885, bottom=0.145)

    return save_figure(
        fig,
        output_dir,
        "hero_recall_vs_latency",
        dpi=dpi,
        bbox_inches="tight",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the hero chart generator."""
    parser = argparse.ArgumentParser(
        description="Generate the SIFT1M hero Pareto frontier chart (PNG + SVG)."
    )
    parser.add_argument(
        "--phase7-dir",
        type=Path,
        default=Path("results/phase7"),
        help="Directory containing Phase 7 1M JSON reports (default: results/phase7).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write hero_recall_vs_latency.{png,svg} (default: results).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output PNG DPI (default: 200).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating the SIFT1M hero chart."""
    args = parse_args(argv)
    created = generate_hero_chart(args.phase7_dir, args.output_dir, dpi=args.dpi)
    print(f"[Hero Plotter] Generated {len(created)} files:")
    for path in created:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
