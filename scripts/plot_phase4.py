#!/usr/bin/env python3
"""Phase 4 Parameter Sweep Plotter (Milestone T2.4).

Generates deterministic, publication-grade benchmark charts from Phase 4 JSON reports:
1. Chart 1: Recall@10 vs p99 Latency (ms, log scale) with explicit Pareto frontier.
2. Chart 2: Recall@10 vs QPS (single thread) with explicit Pareto frontier.
3. Chart 3: Build Time (s) vs Best Achievable Recall (at fixed latency budget).
4. Chart 4: Graph Memory per Vector (B/vec) vs M.

All charts are saved in both PNG (300 DPI) and deterministic SVG formats.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.plot_common import (
    OKABE_ITO_AMBER,
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_GREEN,
    OKABE_ITO_ORANGE,
    OKABE_ITO_PINK,
    OKABE_ITO_SKY,
    compute_pareto_frontier,
    compute_recall_xlim,
    configure_ax,
    configure_matplotlib,
    safe_float,
    safe_int,
    save_figure,
)

configure_matplotlib(svg_hashsalt="vector-search-phase4")


@dataclass(frozen=True)
class SweepPoint:
    """Single efSearch operating point from a Phase 4 report."""

    ef_search: int
    recall: float
    qps: float
    p99_ms: float


@dataclass(frozen=True)
class IndexReport:
    """Normalized Phase 4 index configuration report."""

    scale_vectors: int
    m: int
    ef_construction: int
    build_time_s: float
    graph_only_bytes_per_vec: float
    cpu_model: str
    affinity_mask: str
    dataset_description: str
    num_queries: int
    k: int
    sweep_runs: int
    points: list[SweepPoint]

    @property
    def label(self) -> str:
        return f"M={self.m}, efC={self.ef_construction}"


def _dict_field(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key)
    return val if isinstance(val, dict) else {}


def parse_report_file(path: str | Path) -> IndexReport | None:
    """Parse one Phase 4 JSON report into an ``IndexReport``, returning ``None`` on mismatch."""
    import json

    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to parse {file_path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, dict) or data.get("phase") != 4:
        return None

    hnsw_cfg = _dict_field(data, "hnsw_config")
    build_info = _dict_field(data, "index_build")
    mem_profile = _dict_field(data, "memory_profile")
    hw = _dict_field(data, "hardware")
    env = _dict_field(data, "environment")
    dataset = _dict_field(data, "dataset")

    raw_points = data.get("ef_search_sweep", [])
    if not isinstance(raw_points, list):
        return None

    points = [
        SweepPoint(
            ef_search=safe_int(pt.get("ef_search")),
            recall=safe_float(pt.get("recall_at_k")),
            qps=safe_float(pt.get("qps")),
            p99_ms=safe_float(pt.get("p99_ms")),
        )
        for pt in raw_points
        if isinstance(pt, dict)
    ]

    cpu_model = str(hw.get("cpu_model") or env.get("cpu_model") or "Unknown CPU")
    affinity_mask = str(env.get("processor_affinity_mask") or "not_set")
    dataset_desc = str(dataset.get("source_description") or "SIFT")

    return IndexReport(
        scale_vectors=safe_int(data.get("scale_vectors")),
        m=safe_int(hnsw_cfg.get("m")),
        ef_construction=safe_int(hnsw_cfg.get("ef_construction")),
        build_time_s=safe_float(build_info.get("build_time_s")),
        graph_only_bytes_per_vec=safe_float(
            mem_profile.get("graph_only_bytes_per_vector", data.get("graph_only_bytes_per_vector"))
        ),
        cpu_model=cpu_model,
        affinity_mask=affinity_mask,
        dataset_description=dataset_desc,
        num_queries=safe_int(dataset.get("num_query"), default=10_000),
        k=safe_int(dataset.get("k"), default=10),
        sweep_runs=safe_int(data.get("sweep_runs"), default=3),
        points=points,
    )


def distinct_color_for_m(m: int) -> str:
    palette = {
        8: OKABE_ITO_BLUE,
        12: OKABE_ITO_GREEN,
        16: OKABE_ITO_ORANGE,
        24: OKABE_ITO_PINK,
        32: OKABE_ITO_AMBER,
        48: OKABE_ITO_SKY,
    }
    return palette.get(m, "#333333")


def linestyle_for_efc(efc: int) -> str:
    return {40: ":", 100: "-", 200: "--", 400: "-."}.get(efc, "-")


def marker_for_efc(efc: int) -> str:
    return {40: "o", 100: "s", 200: "^", 400: "D"}.get(efc, "o")


def _format_scale_tag(scale_vectors: int) -> str:
    if scale_vectors >= 1_000_000 and scale_vectors % 1_000_000 == 0:
        return f"{scale_vectors // 1_000_000}M"
    if scale_vectors >= 1_000 and scale_vectors % 1_000 == 0:
        return f"{scale_vectors // 1_000}k"
    return f"{scale_vectors:,}" if scale_vectors > 0 else "n/a"


def _build_provenance_caption(reports: list[IndexReport], detail_line: str | None = None) -> str:
    rep = reports[0] if reports else None
    hw_str = rep.cpu_model if rep else "Unknown CPU"
    affinity = rep.affinity_mask if rep else "not_set"
    scale_tag = _format_scale_tag(rep.scale_vectors) if rep else "n/a"
    ds_desc = rep.dataset_description if rep else "SIFT"
    runs = rep.sweep_runs if rep else 3
    num_q = rep.num_queries if rep else 10_000
    k_val = rep.k if rep else 10

    line1 = f"Single thread (Affinity: {affinity}) | CPU: {hw_str}"
    line2 = detail_line or (
        f"{ds_desc} ({scale_tag}, {num_q:,} queries, k={k_val}) | 1 warmup + {runs} timed reps per efSearch"
    )
    return f"{line1}\n{line2}"


def _plot_recall_pareto_chart(
    reports: list[IndexReport],
    out_dir: str | Path,
    *,
    y_fn: Callable[[SweepPoint], float],
    maximize_y: bool,
    log_y: bool,
    ylabel: str,
    title: str,
    stem: str,
    pareto_label: str,
    legend_loc: Literal["upper left", "upper right"],
    legend_bbox: tuple[float, float],
    reports_1m: list[IndexReport] | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    configure_ax(ax, log_x=False, log_y=log_y)

    all_eval_pts: list[tuple[float, float]] = []
    for rpt in sorted(reports, key=lambda r: (r.m, r.ef_construction)):
        xs = [pt.recall for pt in rpt.points]
        ys = [y_fn(pt) for pt in rpt.points]
        all_eval_pts.extend(zip(xs, ys, strict=True))
        ax.plot(
            xs,
            ys,
            color=distinct_color_for_m(rpt.m),
            linestyle=linestyle_for_efc(rpt.ef_construction),
            marker=marker_for_efc(rpt.ef_construction),
            markersize=3.5,
            alpha=0.65,
            linewidth=1.2,
            label=rpt.label,
        )

    pareto_pts = compute_pareto_frontier(
        all_eval_pts,
        x_fn=lambda p: p[0],
        y_fn=lambda p: p[1],
        maximize_x=True,
        maximize_y=maximize_y,
    )
    if pareto_pts:
        ax.plot(
            [pt[0] for pt in pareto_pts],
            [pt[1] for pt in pareto_pts],
            color=OKABE_ITO_BLACK,
            linestyle="-",
            linewidth=2.5,
            marker="*",
            markersize=8,
            label=pareto_label,
            zorder=10,
        )

    for rpt in reports_1m or ():
        ax.plot(
            [pt.recall for pt in rpt.points],
            [y_fn(pt) for pt in rpt.points],
            linestyle="-",
            marker="X",
            markersize=6,
            linewidth=2.0,
            label=f"1M Final: {rpt.label}",
            zorder=9,
        )

    recalls = [pt[0] for pt in all_eval_pts]
    ax.set_xlim(*compute_recall_xlim(recalls, default=(0.70, 1.002), hi_cap=1.002))
    k_val = reports[0].k if reports else 10
    ax.set_xlabel(f"Recall@{k_val}", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.text(
        0.5,
        -0.02,
        _build_provenance_caption(reports),
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=7.5,
        frameon=True,
        title="Index Configs",
        title_fontsize=8.5,
    )
    fig.tight_layout()
    saved = save_figure(fig, out_dir, stem, bbox_inches="tight")
    print(f"[Plotter] Saved: {saved[0]} & {saved[1]}")
    return saved


def plot_chart1_recall_vs_latency(
    reports: list[IndexReport],
    reports_1m: list[IndexReport],
    out_dir: str | Path,
) -> list[Path]:
    """Chart 1: Recall@10 (x) vs p99 Latency (ms, log scale) with explicit Pareto frontier."""
    scale_tag = _format_scale_tag(reports[0].scale_vectors) if reports else "100k"
    return _plot_recall_pareto_chart(
        reports,
        out_dir,
        y_fn=lambda pt: pt.p99_ms,
        maximize_y=False,
        log_y=True,
        ylabel="p99 Latency [ms] (log scale)",
        title=f"HNSW Parameter Sweep: Recall@10 vs p99 Latency ({scale_tag} SIFT1M Prefix)",
        stem="chart1_recall_vs_p99",
        pareto_label=f"Pareto Frontier ({scale_tag})",
        legend_loc="upper left",
        legend_bbox=(1.02, 1.0),
        reports_1m=reports_1m,
    )


def plot_chart2_recall_vs_qps(
    reports: list[IndexReport],
    out_dir: str | Path,
) -> list[Path]:
    """Chart 2: Recall@10 (x) vs Single-Thread QPS (y) with explicit Pareto frontier."""
    scale_tag = _format_scale_tag(reports[0].scale_vectors) if reports else "100k"
    return _plot_recall_pareto_chart(
        reports,
        out_dir,
        y_fn=lambda pt: pt.qps,
        maximize_y=True,
        log_y=False,
        ylabel="Throughput [Queries / Second]",
        title=f"HNSW Parameter Sweep: Recall@10 vs QPS ({scale_tag} SIFT1M Prefix)",
        stem="chart2_recall_vs_qps",
        pareto_label="Pareto Frontier",
        legend_loc="upper right",
        legend_bbox=(1.25, 1.0),
    )


def plot_chart3_build_time_vs_recall(
    reports: list[IndexReport],
    out_dir: str | Path,
    latency_budget_ms: float = 1.0,
) -> list[Path]:
    """Chart 3: Build time vs best achievable recall at a fixed latency budget, per config."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    configure_ax(ax, log_x=False, log_y=False)

    for rpt in sorted(reports, key=lambda r: (r.m, r.ef_construction)):
        eligible = [pt for pt in rpt.points if pt.p99_ms <= latency_budget_ms]
        best_r = max(pt.recall for pt in eligible) if eligible else min(pt.recall for pt in rpt.points)
        ax.scatter(
            rpt.build_time_s,
            best_r,
            color=distinct_color_for_m(rpt.m),
            marker=marker_for_efc(rpt.ef_construction),
            s=80,
            edgecolors="#222222",
            linewidth=0.8,
            alpha=0.85,
            label=rpt.label,
        )
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
        f"Achievable Recall@10 (at p99 <= {latency_budget_ms:.1f} ms)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"Construction Efficiency: Build Time vs Best Achievable Recall (p99 <= {latency_budget_ms:.1f} ms)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    scale_tag = _format_scale_tag(reports[0].scale_vectors) if reports else "100k"
    fig.text(
        0.5,
        -0.02,
        _build_provenance_caption(
            reports,
            f"{scale_tag} SIFT1M prefix | Fixed latency constraint: p99 <= {latency_budget_ms:.1f} ms",
        ),
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    saved = save_figure(fig, out_dir, "chart3_build_time_vs_recall", bbox_inches="tight")
    print(f"[Plotter] Saved: {saved[0]} & {saved[1]}")
    return saved


def plot_chart4_memory_vs_m(
    reports: list[IndexReport],
    out_dir: str | Path,
) -> list[Path]:
    """Chart 4: Graph memory per vector vs M."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    configure_ax(ax, log_x=False, log_y=False)

    ordered = sorted(reports, key=lambda r: (r.m, r.ef_construction))
    m_values = [rpt.m for rpt in ordered]
    mem_values = [rpt.graph_only_bytes_per_vec for rpt in ordered]
    unique_m = sorted(set(m_values))
    mean_mem_by_m = [
        sum(mem for m, mem in zip(m_values, mem_values, strict=True) if m == um)
        / m_values.count(um)
        for um in unique_m
    ]

    ax.scatter(
        m_values,
        mem_values,
        color=OKABE_ITO_BLUE,
        alpha=0.6,
        s=60,
        edgecolors="black",
        label="Measured Configurations",
        zorder=4,
    )
    ax.plot(
        unique_m,
        mean_mem_by_m,
        color=OKABE_ITO_ORANGE,
        linestyle="-",
        linewidth=2.0,
        marker="o",
        label="Empirical Mean (u32 Flat Adjacency)",
        zorder=5,
    )
    for um, mem in zip(unique_m, mean_mem_by_m, strict=True):
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
    scale_tag = _format_scale_tag(reports[0].scale_vectors) if reports else "100k"
    rep_cpu = reports[0].cpu_model if reports else "Unknown CPU"
    fig.text(
        0.5,
        -0.02,
        f"Contiguous u32 flat adjacency layout (Phase 3.5 T1.7)\n"
        f"Zero external pointers | {scale_tag} SIFT1M Prefix | Architecture: {rep_cpu}",
        wrap=True,
        horizontalalignment="center",
        fontsize=8.5,
        color="#444444",
    )
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    saved = save_figure(fig, out_dir, "chart4_memory_vs_m", bbox_inches="tight")
    print(f"[Plotter] Saved: {saved[0]} & {saved[1]}")
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 Parameter Sweep Plotter (Milestone T2.4)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase4"),
        help="Directory containing scale_100000_*.json and summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4"),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_primary = sorted(results_dir.glob(f"scale_{args.scale}_m*_efc*.json"))
    if not files_primary:
        print(
            f"[Plotter] No JSON reports found matching pattern: "
            f"{results_dir / f'scale_{args.scale}_m*_efc*.json'}"
        )
        return 1

    reports_primary = [
        rpt
        for fp in files_primary
        if (rpt := parse_report_file(fp)) is not None and rpt.scale_vectors == args.scale
    ]
    print(f"[Plotter] Loaded {len(reports_primary)} reports for scale={args.scale} from {results_dir}")

    reports_1m = [
        rpt
        for fp in sorted(results_dir.glob("scale_1000000_m*_efc*.json"))
        if (rpt := parse_report_file(fp)) is not None and rpt.scale_vectors == 1_000_000
    ]
    if reports_1m:
        print(f"[Plotter] Loaded {len(reports_1m)} reports for 1M finals.")

    plot_chart1_recall_vs_latency(reports_primary, reports_1m, out_dir)
    plot_chart2_recall_vs_qps(reports_primary, out_dir)
    plot_chart3_build_time_vs_recall(reports_primary, out_dir, args.latency_budget_ms)
    plot_chart4_memory_vs_m(reports_primary, out_dir)
    print(f"[Plotter] All 4 charts successfully generated in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
