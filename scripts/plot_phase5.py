#!/usr/bin/env python3
"""Generate publication-grade SVG and PNG charts for Phase 5 filtered HNSW benchmarks.

Reads per-cell JSON files (`cell_n{scale}_{mode}_s{sigma}.json`) and optional
`summary_n{scale}.json`, `sensitivity_n{scale}.json`, and `opaque_spotcheck_1m.json`
outputs produced by `cargo run --release --bin phase5`.

All outputs obey Gate G4.5 reporting constraints:
- Option C `num_queries` and `is_subquery_probe` scope are surfaced in chart captions.
- `S1_post_uncapped` (`S1_uncap`) is plotted and labeled strictly for `sigma >= 0.05`.
- Works cleanly across 1 cell (spot verification), partial sweeps, or the full 12-point ladder.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

MODE_ORDER: tuple[str, str] = ("independent", "correlated")
MODE_TITLES: dict[str, str] = {
    "independent": "Mode A: Independent Bernoulli",
    "correlated": "Mode B: Spatially Correlated Clusters",
}

STRATEGY_STYLES: dict[str, dict[str, Any]] = {
    "S3_pre_filter": {
        "label": "S3 Pre-filter (exact BF)",
        "color": "#111111",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 5.5,
        "zorder": 6,
    },
    "S2_in_traversal_ef20": {
        "label": "S2 In-traversal (ef=20)",
        "color": "#56B4E9",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.6,
        "markersize": 4.8,
        "zorder": 4,
    },
    "S2_in_traversal_ef100": {
        "label": "S2 In-traversal (ef=100)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "^",
        "linewidth": 2.1,
        "markersize": 5.6,
        "zorder": 5,
    },
    "S2_in_traversal_ef600": {
        "label": "S2 In-traversal (ef=600)",
        "color": "#009E73",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.0,
        "markersize": 5.0,
        "zorder": 5,
    },
    "S1_post_capped_ef100": {
        "label": "S1 Post-filter capped (cap=4096, ef=100)",
        "color": "#D55E00",
        "linestyle": ":",
        "marker": "v",
        "linewidth": 2.0,
        "markersize": 5.6,
        "zorder": 4,
    },
    "S1_post_uncapped_ef100": {
        "label": "S1 Post-filter uncapped (ef=100, sigma >= 0.05)",
        "color": "#CC79A7",
        "linestyle": "-.",
        "marker": "P",
        "linewidth": 2.0,
        "markersize": 6.0,
        "zorder": 5,
    },
}

DEFAULT_STRATEGY_STYLE: dict[str, Any] = {
    "color": "#666666",
    "linestyle": "-",
    "marker": "x",
    "linewidth": 1.4,
    "markersize": 4.5,
    "zorder": 3,
}

PREFERRED_ARM_KEYS: tuple[str, ...] = (
    "S3_pre_filter",
    "S2_in_traversal_ef20",
    "S2_in_traversal_ef100",
    "S2_in_traversal_ef600",
    "S1_post_capped_ef100",
    "S1_post_uncapped_ef100",
)


@dataclass(frozen=True)
class LoadedPhase5Data:
    """Normalized Phase 5 results bundle for plotting."""

    scale: int
    cells: list[dict[str, Any]]
    summary: dict[str, Any] | None
    sensitivity: dict[str, Any] | None
    spotcheck_1m: dict[str, Any] | None


def _style_for_arm_key(arm_key: str) -> dict[str, Any]:
    style = dict(DEFAULT_STRATEGY_STYLE)
    style["label"] = arm_key
    if arm_key in STRATEGY_STYLES:
        style.update(STRATEGY_STYLES[arm_key])
    return style


def _canonical_arm_key(arm: dict[str, Any]) -> str:
    strategy = str(arm.get("strategy", "unknown"))
    ef = int(arm.get("ef_search", 0))
    if strategy == "S3_pre_filter":
        return "S3_pre_filter"
    if strategy == "S2_in_traversal":
        return f"S2_in_traversal_ef{ef}"
    if strategy == "S1_post_capped":
        return f"S1_post_capped_ef{ef}"
    if strategy == "S1_post_uncapped":
        return f"S1_post_uncapped_ef{ef}"
    return str(arm.get("strategy_label", strategy))


def _load_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if isinstance(loaded, dict):
        return loaded
    return None


def _discover_scale(results_dir: Path) -> int:
    scales: set[int] = set()
    for p in results_dir.glob("cell_n*_*.json"):
        name = p.name
        if not name.startswith("cell_n"):
            continue
        rest = name[len("cell_n") :]
        head = rest.split("_", 1)[0]
        if head.isdigit():
            scales.add(int(head))
    if not scales:
        raise FileNotFoundError(f"No Phase 5 cell JSON files found in '{results_dir}'.")
    if len(scales) > 1:
        sorted_scales = sorted(scales)
        raise ValueError(
            f"Multiple scales found in '{results_dir}' ({sorted_scales}); pass --scale explicitly."
        )
    return next(iter(scales))


def load_phase5_results(
    results_dir: Path,
    scale: int | None,
    sensitivity_json: Path | None = None,
    summary_json: Path | None = None,
) -> LoadedPhase5Data:
    """Load cell JSON reports and optional summary/sensitivity artifacts for `scale`."""
    resolved_scale = _discover_scale(results_dir) if scale is None else scale
    cell_paths = sorted(results_dir.glob(f"cell_n{resolved_scale}_*.json"))
    if not cell_paths:
        raise FileNotFoundError(
            f"No Phase 5 cell JSON files matching 'cell_n{resolved_scale}_*.json' in '{results_dir}'."
        )

    cells: list[dict[str, Any]] = []
    for path in cell_paths:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            cells.append(payload)

    cells.sort(key=lambda c: (str(c.get("mode", "")), float(c.get("sigma", 0.0))))

    summary_path = summary_json or (results_dir / f"summary_n{resolved_scale}.json")
    sens_path = sensitivity_json or (results_dir / f"sensitivity_n{resolved_scale}.json")
    spot_path = results_dir / "opaque_spotcheck_1m.json"

    return LoadedPhase5Data(
        scale=resolved_scale,
        cells=cells,
        summary=_load_json_optional(summary_path),
        sensitivity=_load_json_optional(sens_path),
        spotcheck_1m=_load_json_optional(spot_path),
    )


def _format_query_scope(cells: list[dict[str, Any]], strategy_prefix: str) -> str:
    counts: set[int] = set()
    probes: set[bool] = set()
    for cell in cells:
        for arm in cell.get("arms", []):
            strat = str(arm.get("strategy", ""))
            if strat.startswith(strategy_prefix):
                counts.add(int(arm.get("num_queries", 0)))
                probes.add(bool(arm.get("is_subquery_probe", False)))
    if not counts:
        return "n/a"
    c_str = str(next(iter(counts))) if len(counts) == 1 else f"{min(counts)}..{max(counts)}"
    if probes == {True}:
        return f"{c_str} (subquery probe)"
    if probes == {False}:
        return f"{c_str} (full query set)"
    return f"{c_str} (mixed full/probe)"


def _build_provenance_caption(data: LoadedPhase5Data) -> str:
    first = data.cells[0]
    prov = first.get("provenance", {})
    cpu = str(prov.get("cpu_model", "unknown CPU"))
    os_name = str(prov.get("os", "unknown OS"))
    affinity = str(prov.get("processor_affinity_mask", "unknown"))
    commit = str(first.get("git_commit", "unknown"))[:10]
    dirty_suffix = "-dirty" if bool(first.get("git_dirty", False)) else ""
    reps = int(first.get("reps", 0))
    source = str(first.get("dataset_source_kind", first.get("dataset_source", "unknown")))
    s2_q = _format_query_scope(data.cells, "S2_")
    s3_q = _format_query_scope(data.cells, "S3_")
    return (
        f"N={data.scale:,} ({source}) | {cpu} ({os_name}) | affinity={affinity} | reps={reps} | "
        f"git={commit}{dirty_suffix}\n"
        f"Queries: S2/S1={s2_q}, S3={s3_q} | S1_uncap evaluated only for sigma >= 0.05"
    )


def _configure_ax_style(ax: Axes, *, log_x: bool = True, log_y: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.28, color="#888888")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.16, color="#AAAAAA")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def _set_sigma_xlim(ax: Axes, sigmas: list[float]) -> None:
    pos = [s for s in sigmas if s > 0.0]
    if not pos:
        return
    lo, hi = min(pos), max(pos)
    if math.isclose(lo, hi, rel_tol=1e-9):
        ax.set_xlim(lo * 0.45, hi * 2.2)
    else:
        ax.set_xlim(lo * 0.72, hi * 1.38)


def _save_figure(fig: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)


def _collect_mode_series(
    cells: list[dict[str, Any]],
    mode: str,
) -> tuple[list[float], dict[str, list[tuple[float, dict[str, Any]]]]]:
    mode_cells = [c for c in cells if str(c.get("mode", "")) == mode]
    mode_cells.sort(key=lambda c: float(c.get("sigma", 0.0)))
    sigmas: list[float] = [float(c.get("sigma", 0.0)) for c in mode_cells]
    by_key: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for cell in mode_cells:
        sigma = float(cell.get("sigma", 0.0))
        for arm in cell.get("arms", []):
            key = _canonical_arm_key(arm)
            if key.startswith("S1_post_uncapped") and sigma < 0.05 - 1e-12:
                continue
            by_key.setdefault(key, []).append((sigma, arm))
    return sigmas, by_key


def _ordered_arm_keys(keys: set[str]) -> list[str]:
    ordered = [k for k in PREFERRED_ARM_KEYS if k in keys]
    extras = sorted(k for k in keys if k not in PREFERRED_ARM_KEYS)
    return ordered + extras


def _render_no_data_panel(ax: Axes, mode: str) -> None:
    ax.set_title(MODE_TITLES.get(mode, mode), fontsize=11, fontweight="bold")
    ax.text(
        0.5,
        0.5,
        "No cells measured for this mode",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="#666666",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _collect_shared_legend(axes: list[Axes]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        h_list, l_list = ax.get_legend_handles_labels()
        for h, lbl in zip(h_list, l_list, strict=True):
            if lbl not in seen:
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)
    return handles, labels


def plot_chart1_recall_vs_selectivity(data: LoadedPhase5Data, output_dir: Path) -> None:
    """Chart 1: Recall@10 vs log selectivity across Mode A and Mode B."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]

    all_sigmas = [float(c.get("sigma", 0.0)) for c in data.cells]
    for ax, mode in zip(axes, MODE_ORDER, strict=True):
        sigmas, by_key = _collect_mode_series(data.cells, mode)
        if not sigmas:
            _render_no_data_panel(ax, mode)
            continue
        _configure_ax_style(ax, log_x=True, log_y=False)
        _set_sigma_xlim(ax, all_sigmas if len(all_sigmas) > 1 else sigmas)
        ax.axhline(0.95, color="#555555", linestyle="--", linewidth=1.0, alpha=0.65, label="Target 0.95")
        ax.axhline(0.99, color="#888888", linestyle=":", linewidth=1.0, alpha=0.65, label="Target 0.99")

        for key in _ordered_arm_keys(set(by_key.keys())):
            pts = sorted(by_key[key], key=lambda item: item[0])
            xs = [p[0] for p in pts]
            ys = [float(p[1].get("recall_at_k", 0.0)) for p in pts]
            st = _style_for_arm_key(key)
            ax.plot(xs, ys, **st)

        ax.set_title(MODE_TITLES.get(mode, mode), fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=10)
        ax.set_ylim(-0.02, 1.05)

    axes[0].set_ylabel("Filtered Recall@10", fontsize=10)
    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=False,
            fontsize=8.3,
        )
    fig.suptitle(f"Phase 5: Filtered Recall@10 vs Selectivity (N={data.scale:,})", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, _build_provenance_caption(data), ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.84))
    _save_figure(fig, output_dir, "chart1_recall_vs_selectivity")


def plot_chart2_latency_vs_selectivity(data: LoadedPhase5Data, output_dir: Path) -> None:
    """Chart 2: Mean (solid/styled) and p99 (upper whiskers/markers) latency on log y-axis."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.2, 5.5), sharey=True)
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]

    all_sigmas = [float(c.get("sigma", 0.0)) for c in data.cells]
    for ax, mode in zip(axes, MODE_ORDER, strict=True):
        sigmas, by_key = _collect_mode_series(data.cells, mode)
        if not sigmas:
            _render_no_data_panel(ax, mode)
            continue
        _configure_ax_style(ax, log_x=True, log_y=True)
        _set_sigma_xlim(ax, all_sigmas if len(all_sigmas) > 1 else sigmas)

        for key in _ordered_arm_keys(set(by_key.keys())):
            pts = sorted(by_key[key], key=lambda item: item[0])
            xs = [p[0] for p in pts]
            y_mean = [max(float(p[1].get("mean_ms", 1e-4)), 1e-4) for p in pts]
            y_p99 = [max(float(p[1].get("p99_ms", 1e-4)), 1e-4) for p in pts]
            st = _style_for_arm_key(key)
            ax.plot(xs, y_mean, **st)
            # Draw vertical mean->p99 stem + small hollow p99 cap so tail spread stays legible
            for x_val, m_val, p99_val in zip(xs, y_mean, y_p99, strict=True):
                if p99_val > m_val:
                    ax.plot(
                        [x_val, x_val],
                        [m_val, p99_val],
                        color=st["color"],
                        alpha=0.35,
                        linewidth=1.1,
                        zorder=2,
                    )
            ax.scatter(
                xs,
                y_p99,
                facecolors="none",
                edgecolors=st["color"],
                marker="o",
                s=22,
                alpha=0.7,
                linewidths=1.0,
                zorder=3,
            )

        ax.set_title(MODE_TITLES.get(mode, mode), fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=10)

    axes[0].set_ylabel("Query Latency (ms, log scale; stems extend to p99)", fontsize=10)
    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=False,
            fontsize=8.3,
        )
    fig.suptitle(
        f"Phase 5: Query Latency (Mean & p99 Tails) vs Selectivity (N={data.scale:,})",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(0.5, 0.01, _build_provenance_caption(data), ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.84))
    _save_figure(fig, output_dir, "chart2_latency_vs_selectivity")


def plot_chart3_failure_rate_vs_selectivity(data: LoadedPhase5Data, output_dir: Path) -> None:
    """Chart 3: Under-filled (< k results) failure rate vs selectivity across strategies."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]

    all_sigmas = [float(c.get("sigma", 0.0)) for c in data.cells]
    for ax, mode in zip(axes, MODE_ORDER, strict=True):
        sigmas, by_key = _collect_mode_series(data.cells, mode)
        if not sigmas:
            _render_no_data_panel(ax, mode)
            continue
        _configure_ax_style(ax, log_x=True, log_y=False)
        _set_sigma_xlim(ax, all_sigmas if len(all_sigmas) > 1 else sigmas)

        for key in _ordered_arm_keys(set(by_key.keys())):
            pts = sorted(by_key[key], key=lambda item: item[0])
            xs = [p[0] for p in pts]
            ys = [float(p[1].get("fewer_than_k_rate", 0.0)) for p in pts]
            st = _style_for_arm_key(key)
            ax.plot(xs, ys, **st)

        ax.set_title(MODE_TITLES.get(mode, mode), fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=10)
        ax.set_ylim(-0.02, 1.05)

    axes[0].set_ylabel("Fewer-than-k Failure Rate", fontsize=10)
    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=False,
            fontsize=8.3,
        )
    fig.suptitle(
        f"Phase 5: Under-Filled Result Rate (< k) vs Selectivity (N={data.scale:,})",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(0.5, 0.01, _build_provenance_caption(data), ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.84))
    _save_figure(fig, output_dir, "chart3_failure_rate_vs_selectivity")


def _extract_matched_s2_curve(
    mode_cells: list[dict[str, Any]],
    target_recall: float,
) -> tuple[list[float], list[float], list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    efs: list[int] = []
    for cell in sorted(mode_cells, key=lambda c: float(c.get("sigma", 0.0))):
        sigma = float(cell.get("sigma", 0.0))
        candidates = [
            arm
            for arm in cell.get("arms", [])
            if str(arm.get("strategy", "")) == "S2_in_traversal"
            and float(arm.get("recall_at_k", 0.0)) >= target_recall
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda a: int(a.get("ef_search", 10**9)))
        xs.append(sigma)
        ys.append(max(float(best.get("mean_ms", 1e-4)), 1e-4))
        efs.append(int(best.get("ef_search", 0)))
    return xs, ys, efs


def _compute_crossover_fallback(mode_cells: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    ordered = sorted(mode_cells, key=lambda c: float(c.get("sigma", 0.0)))
    points: list[tuple[float, float, float | None, bool]] = []
    for cell in ordered:
        sigma = float(cell.get("sigma", 0.0))
        arms = cell.get("arms", [])
        s3 = next((a for a in arms if str(a.get("strategy", "")) == "S3_pre_filter"), None)
        if s3 is None:
            continue
        s3_ms = float(s3.get("mean_ms", 0.0))
        s2_cands = [
            a
            for a in arms
            if str(a.get("strategy", "")) == "S2_in_traversal"
            and float(a.get("recall_at_k", 0.0)) >= target_recall
        ]
        if not s2_cands:
            points.append((sigma, s3_ms, None, False))
            continue
        best = min(s2_cands, key=lambda a: int(a.get("ef_search", 10**9)))
        s2_ms = float(best.get("mean_ms", 0.0))
        points.append((sigma, s3_ms, s2_ms, s2_ms <= s3_ms))

    if len(points) < 2:
        return {"outcome": "insufficient_points"}

    for idx in range(len(points) - 1):
        s_lo, s3_lo, s2_lo, win_lo = points[idx]
        s_hi, s3_hi, s2_hi, win_hi = points[idx + 1]
        if not win_lo and win_hi and s2_lo is not None and s2_hi is not None:
            d_lo = s2_lo - s3_lo
            d_hi = s2_hi - s3_hi
            denom = d_lo - d_hi
            t = 0.5 if abs(denom) < 1e-12 else max(0.0, min(1.0, d_lo / denom))
            cross = math.exp(math.log(s_lo) + t * (math.log(s_hi) - math.log(s_lo)))
            return {
                "outcome": "crossover_found",
                "bracket_lo_sigma": s_lo,
                "bracket_hi_sigma": s_hi,
                "crossover_sigma_loglog": cross,
            }
    return {"outcome": "no_crossover_in_range"}


def _lookup_crossover(
    data: LoadedPhase5Data,
    mode: str,
    target_recall: float,
    mode_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    key = "crossovers_primary_0_95" if target_recall <= 0.96 else "crossovers_secondary_0_99"
    if data.summary is not None:
        for entry in data.summary.get(key, []):
            if str(entry.get("mode", "")) == mode:
                return dict(entry)
    return _compute_crossover_fallback(mode_cells, target_recall)


def plot_chart4_matched_recall_crossover(data: LoadedPhase5Data, output_dir: Path) -> None:
    """Chart 4: Matched-recall (0.95 & 0.99) S2 latency vs S3 pre-filter with shaded crossover bracket."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]
    all_sigmas = [float(c.get("sigma", 0.0)) for c in data.cells]

    for ax, mode in zip(axes, MODE_ORDER, strict=True):
        mode_cells = [c for c in data.cells if str(c.get("mode", "")) == mode]
        mode_cells.sort(key=lambda c: float(c.get("sigma", 0.0)))
        if not mode_cells:
            _render_no_data_panel(ax, mode)
            continue

        _configure_ax_style(ax, log_x=True, log_y=True)
        _set_sigma_xlim(ax, all_sigmas if len(all_sigmas) > 1 else [float(c.get("sigma", 0.0)) for c in mode_cells])

        # Plot S3 brute-force baseline
        s3_pts = [
            (
                float(c.get("sigma", 0.0)),
                max(float(a.get("mean_ms", 1e-4)), 1e-4),
            )
            for c in mode_cells
            for a in c.get("arms", [])
            if str(a.get("strategy", "")) == "S3_pre_filter"
        ]
        if s3_pts:
            ax.plot(
                [p[0] for p in s3_pts],
                [p[1] for p in s3_pts],
                label="S3 Pre-filter (exact BF)",
                color="#111111",
                linestyle="--",
                marker="s",
                linewidth=2.0,
                markersize=5.5,
                zorder=5,
            )

        # Matched S2 curves at 0.95 (primary) and 0.99 (secondary)
        x95, y95, ef95 = _extract_matched_s2_curve(mode_cells, 0.95)
        if x95:
            ax.plot(
                x95,
                y95,
                label="S2 Recall-matched >= 0.95",
                color="#0072B2",
                linestyle="-",
                marker="o",
                linewidth=2.2,
                markersize=5.8,
                zorder=6,
            )
            for x_v, y_v, ef_v in zip(x95, y95, ef95, strict=True):
                ax.annotate(
                    f"ef{ef_v}",
                    (x_v, y_v),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7.2,
                    color="#004D7A",
                )

        x99, y99, _ = _extract_matched_s2_curve(mode_cells, 0.99)
        if x99:
            ax.plot(
                x99,
                y99,
                label="S2 Recall-matched >= 0.99",
                color="#009E73",
                linestyle="-.",
                marker="^",
                linewidth=1.9,
                markersize=5.4,
                zorder=6,
            )

        cross_95 = _lookup_crossover(data, mode, 0.95, mode_cells)
        outcome_95 = str(cross_95.get("outcome", ""))
        if outcome_95 == "crossover_found":
            lo = float(cross_95["bracket_lo_sigma"])
            hi = float(cross_95["bracket_hi_sigma"])
            star = float(cross_95["crossover_sigma_loglog"])
            ax.axvspan(lo, hi, color="#F0E442", alpha=0.28, label="Crossover bracket (R@10 >= 0.95)")
            ax.axvline(star, color="#D55E00", linestyle="--", linewidth=1.6, zorder=4)
            ax.text(
                star,
                0.96,
                f"sigma*={star:.4f}\n[{lo:.4g}, {hi:.4g}]",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFDF0", "edgecolor": "#D55E00", "alpha": 0.9},
            )
        else:
            cross_99 = _lookup_crossover(data, mode, 0.99, mode_cells)
            note_lines = [f"0.95: {outcome_95 or 'n/a'}", f"0.99: {cross_99.get('outcome', 'n/a')}"]
            ax.text(
                0.03,
                0.96,
                "\n".join(note_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8F8F8", "edgecolor": "#BBBBBB"},
            )

        ax.set_title(MODE_TITLES.get(mode, mode), fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=10)

    axes[0].set_ylabel("Matched-Recall Mean Latency (ms, log scale)", fontsize=10)
    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=False,
            fontsize=8.3,
        )
    fig.suptitle(
        f"Phase 5: Matched-Recall Crossover Latency (S2 vs S3, N={data.scale:,})",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(0.5, 0.01, _build_provenance_caption(data), ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.84))
    _save_figure(fig, output_dir, "chart4_matched_recall_crossover")


def _plot_evals_and_eta_row(data: LoadedPhase5Data, ax_left: Axes, ax_right: Axes) -> None:
    all_sigmas = sorted({float(c.get("sigma", 0.0)) for c in data.cells})
    single_sigma = len(all_sigmas) == 1

    if single_sigma:
        # Grouped valid vs bridge stacked horizontal/vertical bars for each arm when only 1 cell sigma exists
        _configure_ax_style(ax_left, log_x=False, log_y=True)
        labels: list[str] = []
        valid_vals: list[float] = []
        bridge_vals: list[float] = []
        for cell in data.cells:
            mode_tag = "A" if str(cell.get("mode", "")) == "independent" else "B"
            for arm in cell.get("arms", []):
                key = _canonical_arm_key(arm)
                if key == "S2_in_traversal_ef20":
                    continue
                labels.append(f"{mode_tag}:{arm.get('strategy_label', key)}")
                valid_vals.append(max(float(arm.get("valid_dist_evals_mean", 0.0)), 0.0))
                bridge_vals.append(max(float(arm.get("bridge_dist_evals_mean", 0.0)), 0.0))
        x_idx = list(range(len(labels)))
        ax_left.bar(x_idx, valid_vals, color="#0072B2", label="Valid dist evals")
        ax_left.bar(x_idx, bridge_vals, bottom=valid_vals, color="#D55E00", alpha=0.85, label="Bridge dist evals")
        ax_left.set_xticks(x_idx)
        ax_left.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.8)
        ax_left.set_ylabel("Mean Distance Evals / Query (log)", fontsize=9.5)
        ax_left.set_title(f"Distance Evaluations Breakdown (sigma={all_sigmas[0]:.3g})", fontsize=10.5, fontweight="bold")
        ax_left.legend(frameon=False, fontsize=8.0)

        _configure_ax_style(ax_right, log_x=False, log_y=False)
        eta_vals = []
        bar_colors = []
        for cell in data.cells:
            for arm in cell.get("arms", []):
                key = _canonical_arm_key(arm)
                if key == "S2_in_traversal_ef20":
                    continue
                eta_vals.append(float(arm.get("eta_graph_mean", 0.0)))
                bar_colors.append(_style_for_arm_key(key)["color"])
        ax_right.bar(x_idx, eta_vals, color=bar_colors, alpha=0.88)
        ax_right.set_xticks(x_idx)
        ax_right.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.8)
        ax_right.set_ylim(0.0, 1.05)
        ax_right.set_ylabel(r"Graph Efficiency $\eta_{\mathrm{graph}}$", fontsize=9.5)
        ax_right.set_title(r"Graph Navigation Purity $\eta_{\mathrm{graph}}$", fontsize=10.5, fontweight="bold")
        return

    _configure_ax_style(ax_left, log_x=True, log_y=True)
    _configure_ax_style(ax_right, log_x=True, log_y=False)
    _set_sigma_xlim(ax_left, all_sigmas)
    _set_sigma_xlim(ax_right, all_sigmas)

    for mode in MODE_ORDER:
        mode_marker_suffix = " (A)" if mode == "independent" else " (B)"
        ls_override = "-" if mode == "independent" else "--"
        _, by_key = _collect_mode_series(data.cells, mode)
        for key in ("S3_pre_filter", "S2_in_traversal_ef100", "S1_post_capped_ef100"):
            if key not in by_key:
                continue
            pts = sorted(by_key[key], key=lambda item: item[0])
            xs = [p[0] for p in pts]
            tot = [max(float(p[1].get("dist_evals_mean", 1e-2)), 1e-2) for p in pts]
            etas = [float(p[1].get("eta_graph_mean", 0.0)) for p in pts]
            st = _style_for_arm_key(key)
            ax_left.plot(
                xs,
                tot,
                color=st["color"],
                linestyle=ls_override,
                marker=st["marker"],
                linewidth=1.8,
                markersize=4.8,
                label=f"{st['label']}{mode_marker_suffix}",
            )
            ax_right.plot(
                xs,
                etas,
                color=st["color"],
                linestyle=ls_override,
                marker=st["marker"],
                linewidth=1.8,
                markersize=4.8,
                label=f"{st['label']}{mode_marker_suffix}",
            )

    ax_left.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=9.5)
    ax_left.set_ylabel("Mean Distance Evals / Query (log)", fontsize=9.5)
    ax_left.set_title("Total Distance Evaluations vs Selectivity", fontsize=10.5, fontweight="bold")
    ax_left.legend(frameon=False, fontsize=7.5, loc="best")

    ax_right.set_ylim(-0.02, 1.05)
    ax_right.set_xlabel(r"Filter Selectivity $\sigma$ (log scale)", fontsize=9.5)
    ax_right.set_ylabel(r"$\eta_{\mathrm{graph}}$ (visited matching / visited total)", fontsize=9.5)
    ax_right.set_title(r"Graph Navigation Purity $\eta_{\mathrm{graph}}$ (A=solid, B=dashed)", fontsize=10.5, fontweight="bold")


def _plot_sensitivity_row(data: LoadedPhase5Data, ax_left: Axes, ax_right: Axes) -> None:
    sens = data.sensitivity or {}
    items: list[dict[str, Any]] = list(sens.get("sensitivity_arms", []))
    probes: list[dict[str, Any]] = list(sens.get("opaque_eval_probes", []))
    if data.spotcheck_1m is not None:
        probes.extend(list(data.spotcheck_1m.get("opaque_eval_probes", [])))

    # Left: H_max / gamma recall sensitivity
    _configure_ax_style(ax_left, log_x=False, log_y=False)
    hmax_items = [it for it in items if str(it.get("sweep_group", "")) == "h_max_sweep"]
    gamma_items = [it for it in items if str(it.get("sweep_group", "")) == "overfetch_factor_sweep"]

    if hmax_items:
        # Plot Recall@10 grouped by (mode, sigma) over multiplier = H_max / ef_search
        groups: dict[str, list[tuple[float, float]]] = {}
        for it in hmax_items:
            arm = it.get("arm", {})
            ef = max(int(arm.get("ef_search", 100)), 1)
            hmax = arm.get("max_bridge_hops")
            if hmax is None:
                continue
            mult = float(hmax) / float(ef)
            g_label = f"{str(it.get('mode', ''))[:4]} s={float(it.get('sigma', 0.0)):.3g} ef={ef}"
            groups.setdefault(g_label, []).append((mult, float(arm.get("recall_at_k", 0.0))))
        for g_label, pts in sorted(groups.items()):
            pts_sorted = sorted(pts, key=lambda x: x[0])
            ax_left.plot(
                [p[0] for p in pts_sorted],
                [p[1] for p in pts_sorted],
                marker="o",
                linewidth=1.5,
                markersize=4.2,
                label=g_label,
            )
        ax_left.set_xlabel(r"Bridge-Hop Budget Multiplier ($H_{\max} / ef$)", fontsize=9.5)
        ax_left.set_ylabel("Filtered Recall@10", fontsize=9.5)
        ax_left.set_ylim(-0.02, 1.05)
        ax_left.set_title(r"$H_{\max}$ Bridge-Hop Sensitivity", fontsize=10.5, fontweight="bold")
        if len(groups) <= 8:
            ax_left.legend(frameon=False, fontsize=7.0, ncol=2)
    elif gamma_items:
        xs = [str(it.get("arm", {}).get("strategy_label", "")) for it in gamma_items]
        ys = [float(it.get("arm", {}).get("recall_at_k", 0.0)) for it in gamma_items]
        ax_left.bar(range(len(xs)), ys, color="#009E73")
        ax_left.set_xticks(range(len(xs)))
        ax_left.set_xticklabels(xs, rotation=25, ha="right", fontsize=7.5)
        ax_left.set_ylim(-0.02, 1.05)
        ax_left.set_ylabel("Filtered Recall@10", fontsize=9.5)
        ax_left.set_title(r"Overfetch Factor $\gamma$ Sensitivity", fontsize=10.5, fontweight="bold")
    else:
        ax_left.text(0.5, 0.5, "No sensitivity_arms present", ha="center", va="center", transform=ax_left.transAxes)

    # Right: Opaque closure eval_limit sensitivity (max / mean evals vs cap)
    _configure_ax_style(ax_right, log_x=False, log_y=False)
    if probes:
        probe_labels = []
        mean_evals = []
        max_evals = []
        limit_rates = []
        for p in probes:
            lim = p.get("eval_limit")
            lim_str = "None" if lim is None else str(int(lim))
            sc = int(p.get("scale", data.scale))
            sc_tag = f"{sc // 1000}k" if sc < 1_000_000 else "1M"
            strat = "S2" if "S2" in str(p.get("strategy_label", "")) else "S1"
            probe_labels.append(f"{sc_tag} {strat}\nlim={lim_str}\ns={float(p.get('sigma', 0.0)):.2g}")
            mean_evals.append(float(p.get("eval_count_mean", 0.0)))
            max_evals.append(float(p.get("eval_count_max", 0.0)))
            limit_rates.append(float(p.get("eval_limit_reached_rate", 0.0)))
        x_p = list(range(len(probe_labels)))
        width = 0.38
        ax_right.bar([x - width / 2 for x in x_p], mean_evals, width=width, color="#0072B2", label="Mean predicate calls")
        ax_right.bar([x + width / 2 for x in x_p], max_evals, width=width, color="#E69F00", label="Max predicate calls")
        ax_right.axhline(16_384, color="#CC79A7", linestyle="--", linewidth=1.2, label="DEFAULT_OPAQUE_EVAL_LIMIT=16,384")
        ax_right.set_xticks(x_p)
        ax_right.set_xticklabels(probe_labels, fontsize=6.8, rotation=22, ha="right")
        ax_right.set_ylabel("Predicate Closure Invocations", fontsize=9.5)
        ax_right.set_title(
            f"Opaque Closure eval_limit Probe (max hit rate={max(limit_rates) * 100:.1f}%)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_right.legend(frameon=False, fontsize=7.5)
    else:
        ax_right.text(0.5, 0.5, "No opaque_eval_probes present", ha="center", va="center", transform=ax_right.transAxes)


def plot_chart5_distance_evals_and_sensitivity(data: LoadedPhase5Data, output_dir: Path) -> None:
    """Chart 5: Distance evaluations + graph efficiency (and sensitivity sweeps if present)."""
    has_sens = data.sensitivity is not None or data.spotcheck_1m is not None
    if has_sens:
        fig, axes_grid = plt.subplots(2, 2, figsize=(13.5, 9.5))
        _plot_evals_and_eta_row(data, axes_grid[0, 0], axes_grid[0, 1])
        _plot_sensitivity_row(data, axes_grid[1, 0], axes_grid[1, 1])
        top_rect = 0.93
    else:
        fig, axes_arr = plt.subplots(1, 2, figsize=(13.2, 5.4))
        _plot_evals_and_eta_row(data, axes_arr[0], axes_arr[1])
        top_rect = 0.89

    fig.suptitle(
        f"Phase 5: Distance Evaluations, Graph Efficiency & Sensitivity (N={data.scale:,})",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.5, 0.01, _build_provenance_caption(data), ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, top_rect))
    _save_figure(fig, output_dir, "chart5_distance_evals_and_sensitivity")


def generate_all_phase5_charts(
    results_dir: Path,
    output_dir: Path,
    scale: int | None = None,
    sensitivity_json: Path | None = None,
    summary_json: Path | None = None,
) -> list[Path]:
    """Generate all 5 Phase 5 charts (PNG + SVG) and return created file paths."""
    data = load_phase5_results(
        results_dir=results_dir,
        scale=scale,
        sensitivity_json=sensitivity_json,
        summary_json=summary_json,
    )
    plot_chart1_recall_vs_selectivity(data, output_dir)
    plot_chart2_latency_vs_selectivity(data, output_dir)
    plot_chart3_failure_rate_vs_selectivity(data, output_dir)
    plot_chart4_matched_recall_crossover(data, output_dir)
    plot_chart5_distance_evals_and_sensitivity(data, output_dir)

    stems = [
        "chart1_recall_vs_selectivity",
        "chart2_latency_vs_selectivity",
        "chart3_failure_rate_vs_selectivity",
        "chart4_matched_recall_crossover",
        "chart5_distance_evals_and_sensitivity",
    ]
    created: list[Path] = []
    for stem in stems:
        created.append(output_dir / f"{stem}.png")
        created.append(output_dir / f"{stem}.svg")
    return created


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Phase 5 filtered HNSW vs pre/post-filter benchmark charts (PNG + SVG)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase5"),
        help="Directory containing Phase 5 cell JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated PNG/SVG charts (defaults to --results-dir).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Dataset scale N (e.g., 10000, 100000, 1000000). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--sensitivity-json",
        type=Path,
        default=None,
        help="Optional override path to sensitivity_n{scale}.json.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional override path to summary_n{scale}.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.output_dir if args.output_dir is not None else args.results_dir
    created = generate_all_phase5_charts(
        results_dir=args.results_dir,
        output_dir=out_dir,
        scale=args.scale,
        sensitivity_json=args.sensitivity_json,
        summary_json=args.summary_json,
    )
    print(f"Generated {len(created)} chart files in {out_dir}")
    for path in created:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
