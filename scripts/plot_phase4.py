#!/usr/bin/env python3
"""Phase 4 Parameter Sweep Plotter (Milestone T2.4).

Generates deterministic, publication-grade benchmark charts from Phase 4 JSON reports:
1. Chart 1: Recall@10 vs p99 Latency (ms, log scale) with explicit Pareto frontier.
2. Chart 2: Recall@10 vs QPS (single thread) with explicit Pareto frontier.
3. Chart 3: Build Time (s) vs Best Achievable Recall (at fixed latency budget).
4. Chart 4: Graph Memory per Vector (B/vec) vs M.

All charts are saved in both PNG (300 DPI) and SVG formats.
Follows the AGENT_PLAN.md Milestone T2.4 specification.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import matplotlib  # type: ignore

# Use Agg backend for headless, deterministic rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


@dataclass
class SweepPoint:
    ef_search: int
    recall: float
    qps: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class IndexReport:
    filename: str
    scale_vectors: int
    m: int
    ef_construction: int
    build_time_s: float
    build_qps: float
    graph_only_bytes_per_vec: float
    total_bytes_per_vec: float
    peak_rss_mb: float
    graph_fingerprint_hex: str
    hardware_summary: str
    cpu_model: str
    affinity_mask: str
    dataset_description: str
    sweep_runs: int
    points: list[SweepPoint]

    @property
    def label(self) -> str:
        return f"M={self.m}, efC={self.ef_construction}"


def parse_report_file(path: str) -> IndexReport | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
        return None

    if data.get("phase") != 4:
        return None

    scale_vectors = data.get("scale_vectors", 0)
    hnsw_cfg = data.get("hnsw_config", {})
    m = hnsw_cfg.get("m", 0)
    efc = hnsw_cfg.get("ef_construction", 0)

    build_info = data.get("index_build", {})
    build_time_s = build_info.get("build_time_s", 0.0)
    build_qps = build_info.get("build_qps", 0.0)
    fingerprint_hex = build_info.get("graph_fingerprint_hex", "")

    mem_profile = data.get("memory_profile", {})
    graph_only_bytes = mem_profile.get("graph_only_bytes_per_vector", 0.0)
    total_bytes = mem_profile.get("bytes_per_vector", 0.0)
    peak_rss = mem_profile.get("peak_rss_mb", 0.0)

    hw = data.get("hardware", {})
    hw_summary = hw.get("summary", "")
    cpu_model = hw.get("cpu_model", "")

    env = data.get("environment", {})
    affinity_mask = env.get("processor_affinity_mask", "not_set")

    dataset = data.get("dataset", {})
    dataset_desc = dataset.get("source_description", "SIFT")
    sweep_runs = data.get("sweep_runs", 3)

    raw_points = data.get("ef_search_sweep", [])
    points: list[SweepPoint] = []
    for pt in raw_points:
        points.append(
            SweepPoint(
                ef_search=pt.get("ef_search", 0),
                recall=pt.get("recall_at_k", 0.0),
                qps=pt.get("qps", 0.0),
                mean_ms=pt.get("mean_ms", 0.0),
                p50_ms=pt.get("p50_ms", 0.0),
                p95_ms=pt.get("p95_ms", 0.0),
                p99_ms=pt.get("p99_ms", 0.0),
            )
        )

    return IndexReport(
        filename=os.path.basename(path),
        scale_vectors=scale_vectors,
        m=m,
        ef_construction=efc,
        build_time_s=build_time_s,
        build_qps=build_qps,
        graph_only_bytes_per_vec=graph_only_bytes,
        total_bytes_per_vec=total_bytes,
        peak_rss_mb=peak_rss,
        graph_fingerprint_hex=fingerprint_hex,
        hardware_summary=hw_summary,
        cpu_model=cpu_model,
        affinity_mask=affinity_mask,
        dataset_description=dataset_desc,
        sweep_runs=sweep_runs,
        points=points,
    )


def compute_pareto_frontier_latency(
    all_points: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Compute Pareto frontier for Recall vs Latency (higher recall, lower latency).

    all_points: list of (recall, latency_ms, label)
    Returns sorted Pareto optimal points.
    """
    # Sort primarily by recall ascending, then latency ascending
    sorted_pts = sorted(all_points, key=lambda x: (x[0], x[1]))
    frontier: list[tuple[float, float, str]] = []

    # Traverse from right (highest recall) to left:
    # A point is Pareto optimal if its latency is strictly lower than all points with higher recall.
    min_latency = float("inf")
    for pt in reversed(sorted_pts):
        _r, lat, _lbl = pt
        if lat < min_latency:
            frontier.append(pt)
            min_latency = lat

    frontier.reverse()
    return frontier


def compute_pareto_frontier_qps(
    all_points: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Compute Pareto frontier for Recall vs QPS (higher recall, higher QPS).

    all_points: list of (recall, qps, label)
    Returns sorted Pareto optimal points.
    """
    sorted_pts = sorted(all_points, key=lambda x: (x[0], -x[1]))
    frontier: list[tuple[float, float, str]] = []

    # Traverse from right (highest recall) to left:
    # A point is Pareto optimal if its QPS is strictly higher than all points with higher recall.
    max_qps = -1.0
    for pt in reversed(sorted_pts):
        _r, qps, _lbl = pt
        if qps > max_qps:
            frontier.append(pt)
            max_qps = qps

    frontier.reverse()
    return frontier


def distinct_color_for_m(m: int) -> str:
    palette = {
        8: "#1f77b4",   # Blue
        12: "#2ca02c",  # Green
        16: "#ff7f0e",  # Orange
        24: "#d62728",  # Red
        32: "#9467bd",  # Purple
        48: "#8c564b",  # Brown
    }
    return palette.get(m, "#333333")


def linestyle_for_efc(efc: int) -> str:
    styles = {
        40: ":",       # dotted
        100: "-",      # solid
        200: "--",     # dashed
        400: "-.",     # dash-dot
    }
    return styles.get(efc, "-")


def marker_for_efc(efc: int) -> str:
    markers = {
        40: "o",
        100: "s",
        200: "^",
        400: "D",
    }
    return markers.get(efc, "o")


def plot_chart1_recall_vs_latency(
    reports: list[IndexReport],
    reports_1m: list[IndexReport],
    out_dir: str,
) -> None:
    """Chart 1: Recall@10 (x) vs p99 Latency (ms, log scale) with explicit Pareto frontier."""
    _fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    all_eval_pts: list[tuple[float, float, str]] = []

    # Plot 100k series
    for rpt in sorted(reports, key=lambda r: (r.m, r.ef_construction)):
        xs = [pt.recall for pt in rpt.points]
        ys = [pt.p99_ms for pt in rpt.points]
        for pt in rpt.points:
            all_eval_pts.append((pt.recall, pt.p99_ms, f"{rpt.label} (ef={pt.ef_search})"))

        color = distinct_color_for_m(rpt.m)
        ls = linestyle_for_efc(rpt.ef_construction)
        mk = marker_for_efc(rpt.ef_construction)

        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=3.5,
            alpha=0.65,
            linewidth=1.2,
            label=rpt.label,
        )

    # Compute and plot Pareto frontier for 100k
    pareto_pts = compute_pareto_frontier_latency(all_eval_pts)
    if pareto_pts:
        pxs = [pt[0] for pt in pareto_pts]
        pys = [pt[1] for pt in pareto_pts]
        ax.plot(
            pxs,
            pys,
            color="#000000",
            linestyle="-",
            linewidth=2.5,
            marker="*",
            markersize=8,
            label="Pareto Frontier (100k)",
            zorder=10,
        )

    # Plot 1M series if present
    for rpt in reports_1m:
        xs = [pt.recall for pt in rpt.points]
        ys = [pt.p99_ms for pt in rpt.points]
        ax.plot(
            xs,
            ys,
            linestyle="-",
            marker="X",
            markersize=6,
            linewidth=2.0,
            label=f"1M Final: {rpt.label}",
            zorder=9,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=11, fontweight="bold")
    ax.set_ylabel("p99 Latency [ms] (log scale)", fontsize=11, fontweight="bold")
    ax.set_title(
        "HNSW Parameter Sweep: Recall@10 vs p99 Latency (100k SIFT1M Prefix)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Focus on the high-recall region if appropriate
    min_recall = min((pt[0] for pt in all_eval_pts), default=0.7)
    x_lower = max(0.60, min(0.85, (min_recall // 0.05) * 0.05))
    ax.set_xlim(x_lower, 1.002)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Metadata caption
    rep_sample = reports[0] if reports else None
    hw_str = rep_sample.cpu_model if rep_sample else "Unknown CPU"
    caption = (
        f"Single thread (Affinity: P-core 0x0004) | CPU: {hw_str}\n"
        f"SIFT1M 100k prefix (10k queries, k=10) | 1 warmup + 3 timed reps per efSearch"
    )
    plt.figtext(
        0.5,
        -0.02,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7.5,
        frameon=True,
        title="Index Configs",
        title_fontsize=8.5,
    )
    plt.tight_layout()

    png_path = os.path.join(out_dir, "chart1_recall_vs_p99.png")
    svg_path = os.path.join(out_dir, "chart1_recall_vs_p99.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[Plotter] Saved: {png_path} & {svg_path}")


def plot_chart2_recall_vs_qps(reports: list[IndexReport], out_dir: str) -> None:
    """Chart 2: Recall@10 (x) vs Single-Thread QPS (y) with explicit Pareto frontier."""
    _fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    all_eval_pts: list[tuple[float, float, str]] = []

    for rpt in sorted(reports, key=lambda r: (r.m, r.ef_construction)):
        xs = [pt.recall for pt in rpt.points]
        ys = [pt.qps for pt in rpt.points]
        for pt in rpt.points:
            all_eval_pts.append((pt.recall, pt.qps, f"{rpt.label} (ef={pt.ef_search})"))

        color = distinct_color_for_m(rpt.m)
        ls = linestyle_for_efc(rpt.ef_construction)
        mk = marker_for_efc(rpt.ef_construction)

        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=3.5,
            alpha=0.65,
            linewidth=1.2,
            label=rpt.label,
        )

    pareto_pts = compute_pareto_frontier_qps(all_eval_pts)
    if pareto_pts:
        pxs = [pt[0] for pt in pareto_pts]
        pys = [pt[1] for pt in pareto_pts]
        ax.plot(
            pxs,
            pys,
            color="#000000",
            linestyle="-",
            linewidth=2.5,
            marker="*",
            markersize=8,
            label="Pareto Frontier",
            zorder=10,
        )

    ax.set_xlabel("Recall@10", fontsize=11, fontweight="bold")
    ax.set_ylabel("Throughput [Queries / Second]", fontsize=11, fontweight="bold")
    ax.set_title(
        "HNSW Parameter Sweep: Recall@10 vs QPS (100k SIFT1M Prefix)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    min_recall = min((pt[0] for pt in all_eval_pts), default=0.7)
    x_lower = max(0.60, min(0.85, (min_recall // 0.05) * 0.05))
    ax.set_xlim(x_lower, 1.002)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    rep_sample = reports[0] if reports else None
    hw_str = rep_sample.cpu_model if rep_sample else "Unknown CPU"
    caption = (
        f"Single thread (Affinity: P-core 0x0004) | CPU: {hw_str}\n"
        f"SIFT1M 100k prefix (10k queries, k=10) | 1 warmup + 3 timed reps per efSearch"
    )
    plt.figtext(
        0.5,
        -0.02,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.0),
        fontsize=7.5,
        frameon=True,
        title="Index Configs",
        title_fontsize=8.5,
    )
    plt.tight_layout()

    png_path = os.path.join(out_dir, "chart2_recall_vs_qps.png")
    svg_path = os.path.join(out_dir, "chart2_recall_vs_qps.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[Plotter] Saved: {png_path} & {svg_path}")


def plot_chart3_build_time_vs_recall(
    reports: list[IndexReport],
    out_dir: str,
    latency_budget_ms: float = 1.0,
) -> None:
    """Chart 3: Build time vs best achievable recall at a fixed latency budget, per config."""
    _fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    for rpt in sorted(reports, key=lambda r: (r.m, r.ef_construction)):
        # Best recall where p99 latency <= latency_budget_ms
        eligible = [pt for pt in rpt.points if pt.p99_ms <= latency_budget_ms]
        if eligible:
            best_r = max(pt.recall for pt in eligible)
        else:
            best_r = min(pt.recall for pt in rpt.points)  # lowest recall if none meet budget

        color = distinct_color_for_m(rpt.m)
        mk = marker_for_efc(rpt.ef_construction)

        ax.scatter(
            rpt.build_time_s,
            best_r,
            color=color,
            marker=mk,
            s=80,
            edgecolors="#222222",
            linewidth=0.8,
            alpha=0.85,
            label=rpt.label,
        )
        # Small text label for clarity
        ax.annotate(
            f"({rpt.m},{rpt.ef_construction})",
            (rpt.build_time_s, best_r),
            textcoords="offset points",
            xytext=(4, -2),
            fontsize=6.5,
            alpha=0.8,
        )

    ax.set_xlabel("Index Build Time [seconds]", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        f"Achievable Recall@10 (at p99 ≤ {latency_budget_ms:.1f} ms)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"Construction Efficiency: Build Time vs Best Achievable Recall (p99 ≤ {latency_budget_ms:.1f} ms)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    rep_sample = reports[0] if reports else None
    hw_str = rep_sample.cpu_model if rep_sample else "Unknown CPU"
    caption = (
        f"Single thread build & search (P-core 0x0004) | CPU: {hw_str}\n"
        f"100k SIFT1M prefix | Fixed latency constraint: p99 ≤ {latency_budget_ms:.1f} ms"
    )
    plt.figtext(
        0.5,
        -0.02,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )

    plt.tight_layout()
    png_path = os.path.join(out_dir, "chart3_build_time_vs_recall.png")
    svg_path = os.path.join(out_dir, "chart3_build_time_vs_recall.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[Plotter] Saved: {png_path} & {svg_path}")


def plot_chart4_memory_vs_m(reports: list[IndexReport], out_dir: str) -> None:
    """Chart 4: Graph memory per vector vs M."""
    _fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Group by M and average/report graph_only_bytes_per_vec
    m_values: list[int] = []
    mem_values: list[float] = []

    # Map each report
    for rpt in sorted(reports, key=lambda r: (r.m, r.ef_construction)):
        m_values.append(rpt.m)
        mem_values.append(rpt.graph_only_bytes_per_vec)

    # Unique M sorted
    unique_m = sorted(set(m_values))
    mean_mem_by_m = [
        sum(mem for m, mem in zip(m_values, mem_values) if m == um)
        / m_values.count(um)
        for um in unique_m
    ]

    ax.scatter(
        m_values,
        mem_values,
        color="#1f77b4",
        alpha=0.6,
        s=60,
        edgecolors="black",
        label="Measured Configurations",
        zorder=4,
    )
    ax.plot(
        unique_m,
        mean_mem_by_m,
        color="#d62728",
        linestyle="-",
        linewidth=2.0,
        marker="o",
        label="Empirical Mean (u32 Flat Adjacency)",
        zorder=5,
    )

    # Theoretical line: (2*M + 1) * 4 bytes for layer0 + upper arena overhead (~M * 1.5 * 4 bytes) + 8B metadata
    # Layer0 = (2*M + 1) * 4 bytes.
    # Total graph ~ 8*M + 4 + upper_layers + 8 B
    for um, mem in zip(unique_m, mean_mem_by_m):
        ax.annotate(
            f"{mem:.1f} B/vec",
            (um, mem),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlabel("Degree Bound M", fontsize=11, fontweight="bold")
    ax.set_ylabel("Graph Memory per Vector [Bytes]", fontsize=11, fontweight="bold")
    ax.set_title(
        "HNSW Graph Memory Scaling: Bytes per Vector vs M (Safe Rust u32 Flat Layout)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(unique_m)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    rep_sample = reports[0] if reports else None
    hw_str = rep_sample.cpu_model if rep_sample else "Unknown CPU"
    caption = (
        f"Contiguous u32 flat adjacency layout (Phase 3.5 T1.7)\n"
        f"Zero external pointers | 100k SIFT1M Prefix | Architecture: {hw_str}"
    )
    plt.figtext(
        0.5,
        -0.02,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )

    ax.legend(loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "chart4_memory_vs_m.png")
    svg_path = os.path.join(out_dir, "chart4_memory_vs_m.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[Plotter] Saved: {png_path} & {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Parameter Sweep Plotter (Milestone T2.4)"
    )
    parser.add_argument(
        "--results-dir",
        default="results/phase4",
        help="Directory containing scale_100000_*.json and summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase4",
        help="Directory to output generated charts",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100_000,
        help="Scale to filter reports (default: 100000)",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=1.0,
        help="Fixed latency budget for Chart 3 (default: 1.0 ms)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    pattern_100k = os.path.join(results_dir, f"scale_{args.scale}_m*_efc*.json")
    files_100k = glob.glob(pattern_100k)

    if not files_100k:
        print(f"[Plotter] No JSON reports found matching pattern: {pattern_100k}")
        sys.exit(1)

    reports_100k: list[IndexReport] = []
    for fp in files_100k:
        rpt = parse_report_file(fp)
        if rpt and rpt.scale_vectors == args.scale:
            reports_100k.append(rpt)

    print(f"[Plotter] Loaded {len(reports_100k)} reports for scale={args.scale} from {results_dir}")

    # Check for 1M final reports
    pattern_1m = os.path.join(results_dir, "scale_1000000_m*_efc*.json")
    files_1m = glob.glob(pattern_1m)
    reports_1m: list[IndexReport] = []
    for fp in files_1m:
        rpt = parse_report_file(fp)
        if rpt and rpt.scale_vectors == 1_000_000:
            reports_1m.append(rpt)

    if reports_1m:
        print(f"[Plotter] Loaded {len(reports_1m)} reports for 1M finals.")

    # Generate all 4 charts
    plot_chart1_recall_vs_latency(reports_100k, reports_1m, out_dir)
    plot_chart2_recall_vs_qps(reports_100k, out_dir)
    plot_chart3_build_time_vs_recall(reports_100k, out_dir, args.latency_budget_ms)
    plot_chart4_memory_vs_m(reports_100k, out_dir)
    print(f"[Plotter] All 4 charts successfully generated in {out_dir}")


if __name__ == "__main__":
    main()
