#!/usr/bin/env python3
"""Generate the 1M SIFT1M filtered-search post-filter collapse figure (PNG + SVG).

Reads existing Phase 5 1M cell JSONs (`cell_n1000000_*.json`) and summary JSON
(`summary_n1000000.json`) under `--results-dir` (no benchmark re-execution) and
renders a two-panel figure illustrating why "search then filter" (`S1`) collapses
at low selectivity—especially under spatially correlated predicates (Mode B)—and
how `S2` (in-traversal) and `S3` (exact `DenseBitset` pre-filter) resolve it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.plot_common import (
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_ORANGE,
    OKABE_ITO_PINK,
    configure_ax,
    configure_matplotlib,
    safe_float,
    safe_opt_float,
    save_figure,
)

configure_matplotlib(svg_hashsalt="vector-search-filter-collapse")

TARGET_SCALE = 1_000_000
KEY_ARMS: tuple[str, ...] = (
    "S3_pre_filter",
    "S2_ef100",
    "S1_cap4096_ef100",
    "S1_uncap_ef100",
)


@dataclass(frozen=True)
class ArmPoint:
    """Single `(sigma, metrics)` observation extracted from a Phase 5 cell JSON."""

    sigma: float
    recall_at_k: float
    p99_ms: float
    mean_ms: float
    qps: float
    fewer_than_k_rate: float
    cap_reached_rate: float


@dataclass(frozen=True)
class CollapseDataset:
    """Normalized 1M Phase 5 curves and crossovers for plotting."""

    scale: int
    series: dict[tuple[str, str], list[ArmPoint]]
    crossover_mode_a: float
    crossover_mode_b: float
    cpu_model: str
    git_commit: str
    reps: int


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected JSON object in '{path}'.")
    return loaded


def load_collapse_dataset(results_dir: Path, scale: int = TARGET_SCALE) -> CollapseDataset:
    """Load 1M cell JSONs and crossover summary from ``results_dir``."""
    cell_paths = sorted(results_dir.glob(f"cell_n{scale}_*.json"))
    if not cell_paths:
        raise FileNotFoundError(
            f"No Phase 5 cell JSON files matching 'cell_n{scale}_*.json' in '{results_dir}'."
        )

    series: dict[tuple[str, str], list[ArmPoint]] = {}
    first_cell: dict[str, Any] | None = None
    for cell_path in cell_paths:
        cell = _load_json(cell_path)
        if first_cell is None:
            first_cell = cell
        mode = str(cell.get("mode", ""))
        sigma = safe_float(cell.get("sigma"), 0.0)
        if mode not in {"independent", "correlated"} or sigma <= 0.0:
            continue

        for arm in cell.get("arms", []):
            if not isinstance(arm, dict):
                continue
            label = str(arm.get("strategy_label", ""))
            if label not in KEY_ARMS:
                continue
            if label == "S1_uncap_ef100" and sigma < 0.05 - 1e-12:
                continue
            point = ArmPoint(
                sigma=sigma,
                recall_at_k=safe_float(arm.get("recall_at_k")),
                p99_ms=max(safe_float(arm.get("p99_ms"), 1e-4), 1e-4),
                mean_ms=max(safe_float(arm.get("mean_ms"), 1e-4), 1e-4),
                qps=safe_float(arm.get("qps")),
                fewer_than_k_rate=safe_float(arm.get("fewer_than_k_rate")),
                cap_reached_rate=safe_float(arm.get("post_filter_cap_reached_rate")),
            )
            series.setdefault((mode, label), []).append(point)

    for pts in series.values():
        pts.sort(key=lambda p: p.sigma)

    summary_path = results_dir / f"summary_n{scale}.json"
    summary = _load_json(summary_path)
    crossovers_95 = summary.get("crossovers_primary_0_95", [])
    cross_a: float | None = None
    cross_b: float | None = None
    for entry in crossovers_95:
        if not isinstance(entry, dict):
            continue
        m_name = str(entry.get("mode", ""))
        val = safe_opt_float(entry.get("crossover_sigma_loglog"))
        if m_name == "independent":
            cross_a = val
        elif m_name == "correlated":
            cross_b = val
    if cross_a is None or cross_b is None:
        raise ValueError(f"Missing R@10>=0.95 crossovers in '{summary_path}'.")

    assert first_cell is not None
    prov = first_cell.get("provenance", {})
    cpu_model = str(prov.get("cpu_model", "unknown CPU"))
    git_commit = str(first_cell.get("git_commit", "unknown"))[:10]
    reps = int(safe_float(first_cell.get("reps"), 0.0))

    return CollapseDataset(
        scale=scale,
        series=series,
        crossover_mode_a=cross_a,
        crossover_mode_b=cross_b,
        cpu_model=cpu_model,
        git_commit=git_commit,
        reps=reps,
    )


def _point_at(points: list[ArmPoint], target_sigma: float) -> ArmPoint | None:
    for pt in points:
        if abs(pt.sigma - target_sigma) <= 1e-9:
            return pt
    return None


def _configure_sigma_axis(ax: Axes, sigmas: list[float]) -> None:
    lo, hi = min(sigmas), max(sigmas)
    ax.set_xlim(lo * 0.72, hi * 1.38)
    ticks = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.0005", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"], fontsize=8.2)
    ax.xaxis.set_minor_formatter(FormatStrFormatter(""))
    ax.tick_params(axis="x", which="minor", bottom=False)


def _add_regime_guides(
    ax: Axes,
    *,
    lo_sigma: float,
    cross_a: float | None,
    cross_b: float,
) -> None:
    ax.axvspan(lo_sigma * 0.72, cross_b, color=OKABE_ITO_ORANGE, alpha=0.07, zorder=0)
    if cross_a is not None:
        ax.axvspan(lo_sigma * 0.72, 0.0025, color=OKABE_ITO_ORANGE, alpha=0.07, zorder=0)
        ax.axvline(cross_a, color="#4D6A88", linestyle="--", linewidth=1.25, alpha=0.85, zorder=2)
        ax.text(
            cross_a,
            0.03,
            rf"$\sigma^*_A={cross_a:.4f}$",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.8,
            color="#35516E",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "#4D6A88", "alpha": 0.92},
        )
    ax.axvline(cross_b, color="#A63A00", linestyle="--", linewidth=1.4, alpha=0.9, zorder=2)
    ax.text(
        cross_b,
        0.03,
        rf"$\sigma^*_B={cross_b:.4f}$",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#8C2D00",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "#A63A00", "alpha": 0.92},
    )


def plot_filter_collapse(data: CollapseDataset, output_dir: Path) -> list[Path]:
    """Build the 2-panel 1M filter-collapse figure and save PNG + SVG."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.6, 5.5))
    ax_recall: Axes = axes_arr[0]
    ax_p99: Axes = axes_arr[1]

    s3_b = data.series[("correlated", "S3_pre_filter")]
    s2_b = data.series[("correlated", "S2_ef100")]
    s2_a = data.series[("independent", "S2_ef100")]
    s1_cap_b = data.series[("correlated", "S1_cap4096_ef100")]
    s1_cap_a = data.series[("independent", "S1_cap4096_ef100")]
    s1_uncap_b = data.series[("correlated", "S1_uncap_ef100")]

    all_sigmas = [pt.sigma for pt in s3_b]
    lo_sigma = min(all_sigmas)

    # Panel 1: Recall@10 vs selectivity (Mode B primary + Mode A dashed comparison)
    configure_ax(ax_recall, log_x=True, log_y=False)
    _configure_sigma_axis(ax_recall, all_sigmas)
    _add_regime_guides(
        ax_recall,
        lo_sigma=lo_sigma,
        cross_a=data.crossover_mode_a,
        cross_b=data.crossover_mode_b,
    )
    ax_recall.axhline(
        0.95,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        alpha=0.85,
        zorder=1,
        label="Target Recall@10 = 0.95",
    )

    ax_recall.plot(
        [p.sigma for p in s3_b],
        [p.recall_at_k for p in s3_b],
        color=OKABE_ITO_BLACK,
        linestyle="-",
        marker="s",
        linewidth=2.1,
        markersize=5.0,
        zorder=6,
        label="S3 Pre-filter (exact DenseBitset)",
    )
    ax_recall.plot(
        [p.sigma for p in s2_b],
        [p.recall_at_k for p in s2_b],
        color=OKABE_ITO_BLUE,
        linestyle="-",
        marker="^",
        linewidth=2.2,
        markersize=5.4,
        zorder=5,
        label="S2 In-traversal (ef=100, Mode B corr.)",
    )
    ax_recall.plot(
        [p.sigma for p in s2_a],
        [p.recall_at_k for p in s2_a],
        color=OKABE_ITO_BLUE,
        linestyle="--",
        marker="^",
        linewidth=1.5,
        markersize=4.2,
        alpha=0.55,
        zorder=3,
        label="S2 In-traversal (ef=100, Mode A ind.)",
    )
    ax_recall.plot(
        [p.sigma for p in s1_cap_b],
        [p.recall_at_k for p in s1_cap_b],
        color=OKABE_ITO_ORANGE,
        linestyle="-",
        marker="v",
        linewidth=2.2,
        markersize=5.4,
        zorder=5,
        label="S1 Post-filter cap=4096 (ef=100, Mode B corr.)",
    )
    ax_recall.plot(
        [p.sigma for p in s1_cap_a],
        [p.recall_at_k for p in s1_cap_a],
        color=OKABE_ITO_ORANGE,
        linestyle="--",
        marker="v",
        linewidth=1.5,
        markersize=4.2,
        alpha=0.58,
        zorder=3,
        label="S1 Post-filter cap=4096 (ef=100, Mode A ind.)",
    )
    ax_recall.plot(
        [p.sigma for p in s1_uncap_b],
        [p.recall_at_k for p in s1_uncap_b],
        color=OKABE_ITO_PINK,
        linestyle="-.",
        marker="P",
        linewidth=2.0,
        markersize=5.8,
        zorder=5,
        label=r"S1 Post-filter uncapped (ef=100, Mode B, $\sigma \geq 0.05$)",
    )

    ax_recall.set_ylim(-0.04, 1.06)
    ax_recall.set_title("Recall@10 Collapse (Mode B Solid, Mode A Dashed)", fontsize=10.8, fontweight="bold")
    ax_recall.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=10)
    ax_recall.set_ylabel("Filtered Recall@10", fontsize=10)

    # Panel 1 annotations anchored strictly to observed JSON points
    ax_recall.annotate(
        "S3 pre-filter\n(R@10 = 1.000, 0% <k)",
        xy=(0.002, 1.0),
        xytext=(0.0045, 0.81),
        fontsize=7.5,
        color=OKABE_ITO_BLACK,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#F7F7F7", "edgecolor": "#888888", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0, "connectionstyle": "arc3,rad=-0.12"},
        zorder=7,
    )

    pt_a_001 = _point_at(s1_cap_a, 0.001)
    if pt_a_001 is not None:
        ax_recall.annotate(
            f"Mode A hits K'=4096 wall\n({pt_a_001.cap_reached_rate * 100:.1f}% capped, R={pt_a_001.recall_at_k:.3f})",
            xy=(pt_a_001.sigma, pt_a_001.recall_at_k),
            xytext=(0.0029, 0.56),
            fontsize=7.5,
            color="#8C2D00",
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFDF8", "edgecolor": OKABE_ITO_ORANGE, "alpha": 0.94},
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO_ORANGE, "lw": 1.1},
            zorder=7,
        )

    pt_a_01 = _point_at(s1_cap_a, 0.01)
    pt_b_01 = _point_at(s1_cap_b, 0.01)
    if pt_a_01 is not None and pt_b_01 is not None:
        ax_recall.annotate(
            f"Correlated Mode B collapse\n(σ=0.01: R={pt_b_01.recall_at_k:.3f} vs {pt_a_01.recall_at_k:.3f} Mode A)",
            xy=(pt_b_01.sigma, pt_b_01.recall_at_k),
            xytext=(0.027, 0.16),
            fontsize=7.4,
            color="#8C2D00",
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFDF8", "edgecolor": OKABE_ITO_ORANGE, "alpha": 0.94},
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO_ORANGE, "lw": 1.1, "connectionstyle": "arc3,rad=0.14"},
            zorder=7,
        )

    # Panel 2: p99 Latency (ms, log scale) vs selectivity (Mode B correlated)
    configure_ax(ax_p99, log_x=True, log_y=True)
    _configure_sigma_axis(ax_p99, all_sigmas)
    _add_regime_guides(
        ax_p99,
        lo_sigma=lo_sigma,
        cross_a=None,
        cross_b=data.crossover_mode_b,
    )

    ax_p99.plot(
        [p.sigma for p in s3_b],
        [p.p99_ms for p in s3_b],
        color=OKABE_ITO_BLACK,
        linestyle="-",
        marker="s",
        linewidth=2.1,
        markersize=5.0,
        zorder=6,
    )
    ax_p99.plot(
        [p.sigma for p in s2_b],
        [p.p99_ms for p in s2_b],
        color=OKABE_ITO_BLUE,
        linestyle="-",
        marker="^",
        linewidth=2.2,
        markersize=5.4,
        zorder=5,
    )
    ax_p99.plot(
        [p.sigma for p in s1_cap_b],
        [p.p99_ms for p in s1_cap_b],
        color=OKABE_ITO_ORANGE,
        linestyle="-",
        marker="v",
        linewidth=2.2,
        markersize=5.4,
        zorder=5,
    )
    ax_p99.plot(
        [p.sigma for p in s1_uncap_b],
        [p.p99_ms for p in s1_uncap_b],
        color=OKABE_ITO_PINK,
        linestyle="-.",
        marker="P",
        linewidth=2.0,
        markersize=5.8,
        zorder=6,
    )

    ax_p99.set_ylim(0.03, 5500.0)
    ax_p99.set_title("p99 Tail Latency vs Selectivity (Mode B Correlated)", fontsize=10.8, fontweight="bold")
    ax_p99.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=10)
    ax_p99.set_ylabel("p99 Query Latency (ms, log scale)", fontsize=10)

    max_p99_le_01 = max(p.p99_ms for p in s3_b if p.sigma <= 0.01 + 1e-12)
    ax_p99.text(
        0.00058,
        3000.0,
        f"S3 exact (R=1.0) wins σ < σ*_B\n(p99 ≤ {max_p99_le_01:.2f} ms for σ ≤ 0.01)",
        fontsize=7.4,
        color="#8C2D00",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFDF8", "edgecolor": "#D9A88F", "alpha": 0.92},
        zorder=7,
    )

    pt_uncap_075 = _point_at(s1_uncap_b, 0.075)
    if pt_uncap_075 is not None:
        ax_p99.annotate(
            f"Uncapped S1 explodes\n(σ=0.075: {pt_uncap_075.p99_ms:,.0f} ms p99)",
            xy=(pt_uncap_075.sigma, pt_uncap_075.p99_ms),
            xytext=(0.022, 1300.0),
            fontsize=7.5,
            color="#7A2E5D",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFF8FC", "edgecolor": OKABE_ITO_PINK, "alpha": 0.94},
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO_PINK, "lw": 1.15},
            zorder=7,
        )

    pt_s3_001_b = _point_at(s3_b, 0.001)
    if pt_s3_001_b is not None:
        ax_p99.annotate(
            f"S3 exact scan wins low σ\n({pt_s3_001_b.p99_ms:.3f} ms p99 at σ=0.001)",
            xy=(pt_s3_001_b.sigma, pt_s3_001_b.p99_ms),
            xytext=(0.0042, 0.14),
            fontsize=7.5,
            color=OKABE_ITO_BLACK,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#F7F7F7", "edgecolor": "#666666", "alpha": 0.94},
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO_BLACK, "lw": 1.1},
            zorder=7,
        )

    pt_s2_50_b = _point_at(s2_b, 0.5)
    pt_s3_50_b = _point_at(s3_b, 0.5)
    if pt_s2_50_b is not None and pt_s3_50_b is not None:
        ax_p99.annotate(
            f"S2 wins high σ\n({pt_s2_50_b.p99_ms:.1f} vs S3 {pt_s3_50_b.p99_ms:.1f} ms)",
            xy=(pt_s2_50_b.sigma, pt_s2_50_b.p99_ms),
            xytext=(0.24, 1.35),
            fontsize=7.5,
            color="#004D7A",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#F4FAFF", "edgecolor": OKABE_ITO_BLUE, "alpha": 0.94},
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO_BLUE, "lw": 1.1},
            zorder=7,
        )

    handles, labels = ax_recall.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        fontsize=7.9,
        columnspacing=1.2,
    )

    fig.suptitle(
        '1M SIFT1M Filtered Search: Why "Search-Then-Filter" (S1) Collapses vs S2 & S3',
        fontsize=12.8,
        fontweight="bold",
        y=0.985,
    )
    caption = (
        f"SIFT1M N={data.scale:,}, M=16, efC=100, k=10, reps={data.reps} | {data.cpu_model} | git={data.git_commit}\n"
        f"Crossovers σ* (matched R@10≥0.95 mean latency): Mode A (independent)={data.crossover_mode_a:.6f}, "
        f"Mode B (correlated)={data.crossover_mode_b:.6f}"
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=7.6, color="#444444")

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.84))
    return save_figure(fig, output_dir, "filter_collapse", dpi=240)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1M filtered-search post-filter collapse chart (PNG + SVG)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase5"),
        help="Directory containing Phase 5 cell_n1000000_*.json and summary_n1000000.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for filter_collapse.png and filter_collapse.svg.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=TARGET_SCALE,
        help="Dataset scale N (default: 1000000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_collapse_dataset(results_dir=args.results_dir, scale=args.scale)
    created = plot_filter_collapse(data=data, output_dir=args.output_dir)
    print(f"Generated {len(created)} files:")
    for path in created:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
