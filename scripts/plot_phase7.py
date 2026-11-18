#!/usr/bin/env python3
"""Phase 7 Head-to-Head Benchmark Plotter (Milestone T5.6).

Generates deterministic, publication-grade PNG (300 DPI) and SVG charts comparing
Rust ``HnswIndex`` against FAISS ``IndexHNSWFlat``:
1. ``chart1_recall_vs_p99.{png,svg}``: Mode 2 Recall@10 vs p99 latency (log scale),
   across configs and scales, disclosing FAISS per-query SWIG ``ffi_delta_us``.
2. ``chart2_recall_vs_qps.{png,svg}``: Mode 1 pure-batch Recall@10 vs single-thread
   QPS (zero per-query FFI / timer tax, log scale).
3. ``chart3_matched_recall_ratios.{png,svg}``: Interpolated Mode 1 QPS and Mode 2
   p99 ratios at standardized recall targets R* in {0.90, 0.95, 0.98, 0.99, 0.995}
   without extrapolation beyond observed [min_recall, max_recall].
4. ``chart4_multithread_and_build.{png,svg}``: Multi-thread Mode 1 batch QPS scaling
   (T in {1, 6, 16}), build wall-clock time (explicitly annotating Rust single-thread
   build asymmetry), graph-only bytes/vector, and isolated process peak RSS.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._metrics import (
    CONFIG_ID_MAP,
    MATCHED_RECALL_TARGETS,
    compute_matched_recall_comparison,
)
from scripts.plot_common import (
    OKABE_ITO_AMBER,
    OKABE_ITO_BLACK,
    OKABE_ITO_BLUE,
    OKABE_ITO_GREEN,
    OKABE_ITO_ORANGE,
    OKABE_ITO_PINK,
    OKABE_ITO_SKY,
    compute_recall_xlim,
    configure_ax,
    configure_matplotlib,
    finalize_figure_with_top_legend,
    render_empty_panel,
    safe_float,
    safe_int,
    safe_opt_float,
    save_figure,
)

configure_matplotlib(svg_hashsalt="phase7", font_family="DejaVu Sans")

CONFIG_ORDER: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C2-nkp")
CONFIG_COLORS: dict[str, str] = {
    "C1": OKABE_ITO_BLUE,
    "C2": OKABE_ITO_ORANGE,
    "C3": OKABE_ITO_GREEN,
    "C4": OKABE_ITO_PINK,
    "C2-nkp": OKABE_ITO_AMBER,
}
FALLBACK_COLORS: tuple[str, ...] = (
    OKABE_ITO_SKY,
    "#7F7F7F",
    "#8C564B",
    "#17BECF",
)

ENGINE_STYLE: dict[str, dict[str, Any]] = {
    "rust": {
        "label_prefix": "Rust",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.1,
        "markersize": 5.0,
        "alpha": 0.92,
        "zorder": 5,
    },
    "faiss": {
        "label_prefix": "FAISS",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.8,
        "markersize": 5.2,
        "alpha": 0.88,
        "zorder": 4,
    },
}


@dataclass(frozen=True)
class SweepPoint:
    """Normalized operating point for a single ``efSearch`` value."""

    ef_search: int
    recall_at_k: float
    mode1_qps: float | None
    mode1_mean_ms: float | None
    mode2_p99_ms: float | None
    ffi_delta_us: float | None


@dataclass(frozen=True)
class Phase7Report:
    """Normalized benchmark report from Rust or FAISS."""

    phase: int
    engine: str
    compare_mode: str
    config_id: str
    scale_vectors: int
    m: int
    ef_construction: int
    threads: int
    affinity_mask: str
    power_mode: str
    cpu_model: str
    git_commit: str
    git_dirty: bool
    build_reps: int
    query_reps: int
    build_time_s: float
    graph_only_bytes_per_vector: float
    peak_rss_mb: float | None
    faiss_version: str | None
    faiss_simd_level: str | None
    points: list[SweepPoint]

    def to_curve_dicts(self) -> list[dict[str, Any]]:
        """Convert normalized sweep points to ``compute_matched_recall_comparison`` format."""
        return [
            {
                "ef_search": pt.ef_search,
                "recall_at_k": pt.recall_at_k,
                "mode1_qps": pt.mode1_qps,
                "mode1_mean_ms": pt.mode1_mean_ms,
                "mode2_p99_ms": pt.mode2_p99_ms,
            }
            for pt in self.points
        ]


def _dict_field(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key)
    return val if isinstance(val, dict) else {}


def _format_scale(scale: int) -> str:
    if scale >= 1_000_000 and scale % 1_000_000 == 0:
        return f"{scale // 1_000_000}M"
    if scale >= 1_000 and scale % 1_000 == 0:
        return f"{scale // 1_000}k"
    return f"{scale:,}"


def _config_sort_key(config_id: str) -> tuple[int, str]:
    if config_id in CONFIG_ORDER:
        return (CONFIG_ORDER.index(config_id), config_id)
    return (len(CONFIG_ORDER), config_id)


def _color_for_config(config_id: str) -> str:
    if config_id in CONFIG_COLORS:
        return CONFIG_COLORS[config_id]
    idx = sum(ord(ch) for ch in config_id) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def _infer_engine(data: dict[str, Any]) -> str:
    raw = str(data.get("engine", "")).strip().lower()
    if raw in ("rust", "faiss"):
        return raw
    if isinstance(data.get("faiss"), dict):
        return "faiss"
    if "rustc_version" in data or data.get("phase") == 4:
        return "rust"
    return "unknown"


def _infer_config_id(
    data: dict[str, Any],
    engine: str,
    m: int,
    efc: int,
    keep_pruned_shrink: bool | None,
) -> str:
    explicit = str(data.get("config_id", "")).strip()
    if explicit:
        return explicit
    if engine == "rust" and keep_pruned_shrink is False:
        return "C2-nkp" if (m, efc) == (16, 100) else f"M{m}_efC{efc}-nkp"
    return CONFIG_ID_MAP.get((m, efc), f"M{m}_efC{efc}")


def _first_opt_float(*vals: Any) -> float | None:
    for v in vals:
        parsed = safe_opt_float(v)
        if parsed is not None:
            return parsed
    return None


def _parse_sweep_points(raw_points: list[Any]) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for pt in raw_points:
        if not isinstance(pt, dict):
            continue
        ef = safe_int(pt.get("ef_search"), default=0)
        recall_opt = _first_opt_float(pt.get("recall_at_k"), pt.get("recall"))
        recall = recall_opt if recall_opt is not None else 0.0

        mode1 = _dict_field(pt, "mode1_batch")
        mode2 = _dict_field(pt, "mode2_per_query")
        median_lat = _dict_field(pt, "median_latency")

        mode1_qps = _first_opt_float(
            mode1.get("qps"), pt.get("mode1_qps"), pt.get("qps"), median_lat.get("qps")
        )
        mode1_mean = _first_opt_float(
            mode1.get("mean_ms"), pt.get("mode1_mean_ms"), pt.get("mean_ms"), median_lat.get("mean_ms")
        )
        mode2_p99 = _first_opt_float(
            mode2.get("p99_ms"), pt.get("mode2_p99_ms"), pt.get("p99_ms"), median_lat.get("p99_ms")
        )
        points.append(
            SweepPoint(
                ef_search=ef,
                recall_at_k=recall,
                mode1_qps=mode1_qps,
                mode1_mean_ms=mode1_mean,
                mode2_p99_ms=mode2_p99,
                ffi_delta_us=safe_opt_float(pt.get("ffi_delta_us")),
            )
        )
    points.sort(key=lambda p: p.ef_search)
    return points


def _parse_single_report(data: dict[str, Any]) -> Phase7Report | None:
    raw_sweep = data.get("ef_search_sweep")
    if not isinstance(raw_sweep, list) or not raw_sweep:
        return None

    engine = _infer_engine(data)
    if engine == "unknown":
        return None

    hnsw_cfg = _dict_field(data, "hnsw_config")
    m = safe_int(hnsw_cfg.get("m"), default=0)
    efc = safe_int(hnsw_cfg.get("ef_construction"), default=0)
    kps_raw = hnsw_cfg.get("keep_pruned_shrink")
    keep_pruned_shrink = bool(kps_raw) if kps_raw is not None else None
    config_id = _infer_config_id(data, engine, m, efc, keep_pruned_shrink)

    ds = _dict_field(data, "dataset")
    scale = safe_int(data.get("scale_vectors", ds.get("scale_vectors", ds.get("num_base"))), default=0)
    if scale <= 0:
        return None

    env = _dict_field(data, "environment")
    hw = _dict_field(data, "hardware")
    ib = _dict_field(data, "index_build")
    mem = _dict_field(data, "memory_profile")
    fx = _dict_field(data, "faiss")

    threads = safe_int(data.get("threads", env.get("threads")), default=1)
    compare_mode = str(
        data.get("compare_mode", "multithread" if threads > 1 else "single")
    ).lower()

    points = _parse_sweep_points(raw_sweep)
    if not points:
        return None

    return Phase7Report(
        phase=safe_int(data.get("phase"), default=7),
        engine=engine,
        compare_mode=compare_mode,
        config_id=config_id,
        scale_vectors=scale,
        m=m,
        ef_construction=efc,
        threads=threads,
        affinity_mask=str(env.get("processor_affinity_mask", "not_set")),
        power_mode=str(env.get("power_mode", "unknown")),
        cpu_model=str(hw.get("cpu_model", env.get("cpu_model", "Unknown CPU"))),
        git_commit=str(data.get("git_commit", "unknown")),
        git_dirty=bool(data.get("git_dirty", False)),
        build_reps=safe_int(data.get("build_reps"), default=1),
        query_reps=safe_int(data.get("query_reps", data.get("sweep_runs")), default=5),
        build_time_s=safe_float(ib.get("build_time_s"), default=0.0),
        graph_only_bytes_per_vector=safe_float(
            mem.get("graph_only_bytes_per_vector", data.get("graph_only_bytes_per_vector")),
            default=0.0,
        ),
        peak_rss_mb=safe_opt_float(mem.get("peak_rss_mb")),
        faiss_version=str(fx["version"]) if "version" in fx else None,
        faiss_simd_level=str(fx["simd_level_name"]) if "simd_level_name" in fx else None,
        points=points,
    )


def _expand_report_json(data: dict[str, Any]) -> list[Phase7Report]:
    """Parse either a standalone report or a multi-thread bundle report."""
    reports: list[Phase7Report] = []
    single = _parse_single_report(data)
    if single is not None:
        reports.append(single)

    for bundle_key in ("thread_runs", "multithread_runs", "runs"):
        entries = data.get(bundle_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            merged = dict(data)
            for k, v in entry.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            merged.setdefault("compare_mode", "multithread")
            parsed = _parse_single_report(merged)
            if parsed is not None:
                reports.append(parsed)
    return reports


def load_phase7_reports(
    results_dir: str | Path,
    *,
    rust_dir: str | Path | None = None,
    faiss_dir: str | Path | None = None,
    scales: tuple[int, ...] | None = None,
) -> list[Phase7Report]:
    """Discover, normalize, and deduplicate Rust and FAISS Phase 7 JSON reports."""
    search_dirs: list[Path] = [Path(results_dir)]
    if rust_dir is not None:
        search_dirs.append(Path(rust_dir))
    if faiss_dir is not None:
        search_dirs.append(Path(faiss_dir))

    seen_files: set[Path] = set()
    raw_reports: list[Phase7Report] = []

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for json_path in sorted(directory.glob("*.json")):
            resolved = json_path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARN] Skipping unreadable JSON {json_path}: {exc}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                continue
            for rpt in _expand_report_json(payload):
                if scales is None or rpt.scale_vectors in scales:
                    raw_reports.append(rpt)

    # Deduplicate by (engine, scale, config_id, threads, compare_mode), preferring phase 7 over phase 4.
    dedup: dict[tuple[str, int, str, int, str], Phase7Report] = {}
    for rpt in raw_reports:
        key = (rpt.engine, rpt.scale_vectors, rpt.config_id, rpt.threads, rpt.compare_mode)
        existing = dedup.get(key)
        if existing is None or (rpt.phase == 7 and existing.phase != 7):
            dedup[key] = rpt

    reports = list(dedup.values())
    reports.sort(
        key=lambda r: (
            r.scale_vectors,
            _config_sort_key(r.config_id),
            0 if r.engine == "rust" else 1,
            r.threads,
        )
    )
    return reports


def _select_single_thread_reports(reports: list[Phase7Report]) -> list[Phase7Report]:
    by_key: dict[tuple[str, int, str], Phase7Report] = {}
    for rpt in reports:
        if rpt.threads != 1:
            continue
        key = (rpt.engine, rpt.scale_vectors, rpt.config_id)
        existing = by_key.get(key)
        if existing is None or (rpt.compare_mode == "single" and existing.compare_mode != "single"):
            by_key[key] = rpt
    return sorted(
        by_key.values(),
        key=lambda r: (r.scale_vectors, _config_sort_key(r.config_id), 0 if r.engine == "rust" else 1),
    )


def _build_provenance_caption(reports: list[Phase7Report]) -> str:
    if not reports:
        return "No Phase 7 reports loaded."
    cpu_models = sorted({r.cpu_model for r in reports if r.cpu_model})
    cpu = cpu_models[0] if cpu_models else "Unknown CPU"

    faiss_rpts = [r for r in reports if r.engine == "faiss"]
    if faiss_rpts:
        fx_ver = sorted({r.faiss_version or "?" for r in faiss_rpts})
        fx_simd = sorted({r.faiss_simd_level or "?" for r in faiss_rpts})
        faiss_tag = f"FAISS {'/'.join(fx_ver)} (SIMD={'+'.join(fx_simd)})"
    else:
        faiss_tag = "FAISS n/a"

    commits = sorted({f"{r.git_commit}{'-dirty' if r.git_dirty else ''}" for r in reports})
    commit_tag = ",".join(commits[:3]) + ("..." if len(commits) > 3 else "")
    affinities = sorted({r.affinity_mask for r in reports if r.affinity_mask})
    power_modes = sorted({r.power_mode for r in reports if r.power_mode})
    q_reps = sorted({r.query_reps for r in reports})
    b_reps = sorted({r.build_reps for r in reports})

    return (
        f"CPU: {cpu} | {faiss_tag} | Affinity: {','.join(affinities)} | "
        f"Power: {','.join(power_modes)}\n"
        f"Build reps: {','.join(str(x) for x in b_reps)} | Query reps: 1 warmup + "
        f"{','.join(str(x) for x in q_reps)} timed | git: {commit_tag}"
    )


def _plot_single_thread_recall_chart(
    reports: list[Phase7Report],
    output_dir: str | Path,
    *,
    y_fn: Callable[[SweepPoint], float | None],
    ylabel: str,
    suptitle: str,
    empty_title: str,
    empty_panel_msg: str,
    stem: str,
    caption_extra_fn: Callable[[list[Phase7Report]], str] | None = None,
) -> list[Path]:
    single_rpts = _select_single_thread_reports(reports)
    scales = sorted({r.scale_vectors for r in single_rpts})
    if not scales:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        render_empty_panel(ax, empty_title, "No single-thread Phase 7 reports found.")
        return save_figure(fig, output_dir, stem, bbox_inches="tight")

    ncols = len(scales)
    fig, axes_raw = plt.subplots(1, ncols, figsize=(6.8 * ncols + 0.6, 5.7), squeeze=False)
    axes: list[Axes] = [axes_raw[0, idx] for idx in range(ncols)]

    for ax, scale in zip(axes, scales, strict=True):
        configure_ax(ax, log_x=False, log_y=True, minor_alpha=0.15)
        scale_rpts = [r for r in single_rpts if r.scale_vectors == scale]
        recalls: list[float] = []

        for rpt in scale_rpts:
            pts_xy = [
                (pt.recall_at_k, yv)
                for pt in rpt.points
                if (yv := y_fn(pt)) is not None and yv > 0.0
            ]
            if not pts_xy:
                continue
            xs = [xy[0] for xy in pts_xy]
            ys = [xy[1] for xy in pts_xy]
            recalls.extend(xs)

            color = _color_for_config(rpt.config_id)
            est = ENGINE_STYLE.get(rpt.engine, ENGINE_STYLE["faiss"])
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=est["linestyle"],
                marker=est["marker"],
                linewidth=est["linewidth"],
                markersize=est["markersize"],
                markerfacecolor=color if rpt.engine == "rust" else "white",
                markeredgewidth=1.4,
                alpha=est["alpha"],
                zorder=est["zorder"],
                label=f"{est['label_prefix']} {rpt.config_id} (M={rpt.m}, efC={rpt.ef_construction})",
            )

        if not recalls:
            render_empty_panel(ax, f"N = {_format_scale(scale)} vectors", empty_panel_msg)
        else:
            ax.set_xlim(*compute_recall_xlim(recalls, default=(0.65, 1.005), hi_cap=1.004))
            ax.set_xlabel("Recall@10", fontsize=10, fontweight="bold")
            ax.set_title(f"N = {_format_scale(scale)} vectors", fontsize=11, fontweight="bold")

    axes[0].set_ylabel(ylabel, fontsize=10, fontweight="bold")
    extra_note = caption_extra_fn(single_rpts) if caption_extra_fn is not None else ""
    finalize_figure_with_top_legend(
        fig,
        axes,
        title=suptitle,
        caption=_build_provenance_caption(single_rpts) + extra_note,
        max_ncol=4,
        legend_y=0.915,
        suptitle_y=0.98,
        legend_fontsize=8.0,
        bottom_margin=0.06,
        top_margin=0.82,
    )
    return save_figure(fig, output_dir, stem, bbox_inches="tight")


def _median_faiss_ffi_note(single_rpts: list[Phase7Report]) -> str:
    deltas = [
        pt.ffi_delta_us
        for rpt in single_rpts
        if rpt.engine == "faiss"
        for pt in rpt.points
        if pt.ffi_delta_us is not None and pt.mode2_p99_ms is not None and pt.mode2_p99_ms > 0.0
    ]
    if not deltas:
        return ""
    med_ffi = sorted(deltas)[len(deltas) // 2]
    return f" | FAISS Mode 2 search_c includes median FFI delta {med_ffi:.2f} us/q"


def plot_chart1_recall_vs_p99(
    reports: list[Phase7Report],
    output_dir: str | Path,
) -> list[Path]:
    """Chart 1: Mode 2 Recall@10 vs p99 latency (log scale), Rust vs FAISS."""
    return _plot_single_thread_recall_chart(
        reports,
        output_dir,
        y_fn=lambda pt: pt.mode2_p99_ms,
        ylabel="Mode 2 p99 Latency [ms] (log scale)",
        suptitle="Phase 7: Mode 2 Recall@10 vs p99 Tail Latency (Single-Thread Pinned)",
        empty_title="Phase 7: Mode 2 Recall@10 vs p99 Latency",
        empty_panel_msg="No Mode 2 p99 latency points present",
        stem="chart1_recall_vs_p99",
        caption_extra_fn=_median_faiss_ffi_note,
    )


def plot_chart2_recall_vs_qps(
    reports: list[Phase7Report],
    output_dir: str | Path,
) -> list[Path]:
    """Chart 2: Mode 1 pure-batch Recall@10 vs single-thread QPS (zero FFI tax)."""
    return _plot_single_thread_recall_chart(
        reports,
        output_dir,
        y_fn=lambda pt: pt.mode1_qps,
        ylabel="Mode 1 Pure-Batch QPS [queries/s] (log scale)",
        suptitle="Phase 7: Mode 1 Pure-Batch Recall@10 vs Throughput (Zero Per-Query FFI Tax)",
        empty_title="Phase 7: Mode 1 Pure-Batch Recall@10 vs QPS",
        empty_panel_msg="No Mode 1 QPS points present",
        stem="chart2_recall_vs_qps",
    )


def _render_empty_matched_recall_chart(single_rpts: list[Phase7Report], output_dir: str | Path) -> list[Path]:
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.6, 5.7))
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]
    for ax, title in zip(
        axes,
        ("Mode 1 Batch QPS Ratio (Rust / FAISS)", "Mode 2 p99 Latency Ratio (FAISS / Rust)"),
        strict=True,
    ):
        render_empty_panel(
            ax,
            title,
            "Paired Rust & FAISS single-thread reports required\nfor matched-recall interpolation.",
        )
    fig.suptitle(
        "Phase 7: Matched-Recall Speed & Tail-Latency Ratios (No Extrapolation)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        _build_provenance_caption(single_rpts),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.91))
    return save_figure(fig, output_dir, "chart3_matched_recall_ratios", bbox_inches="tight")


def _plot_matched_ratio_panel(
    ax: Axes,
    targets: tuple[float, ...],
    paired_keys: list[tuple[int, str]],
    pair_comparisons: dict[tuple[int, str], list[dict[str, Any]]],
    *,
    ratio_key: str,
    title: str,
    ylabel: str,
    add_legend_labels: bool,
) -> None:
    configure_ax(ax, log_x=False, log_y=False, minor_alpha=0.15)
    ax.axhline(1.0, color=OKABE_ITO_BLACK, linestyle="--", linewidth=1.2, alpha=0.75, zorder=2)

    n_pairs = len(paired_keys)
    bar_width = 0.78 / max(1, n_pairs)
    for p_idx, sc_cfg in enumerate(paired_keys):
        scale, cfg = sc_cfg
        comps = pair_comparisons[sc_cfg]
        offset = (p_idx - (n_pairs - 1) / 2.0) * bar_width
        xs_bars: list[float] = []
        ys_bars: list[float] = []
        for t_idx, comp in enumerate(comps):
            val = comp.get(ratio_key)
            if val is not None and val > 0.0:
                xs_bars.append(float(t_idx) + offset)
                ys_bars.append(float(val))

        if not xs_bars:
            continue
        bars = ax.bar(
            xs_bars,
            ys_bars,
            width=bar_width * 0.92,
            color=_color_for_config(cfg),
            hatch="//" if scale >= 1_000_000 else "",
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.88,
            label=f"{_format_scale(scale)} {cfg}" if add_legend_labels else None,
            zorder=4,
        )
        for bar, val in zip(bars, ys_bars, strict=True):
            ax.annotate(
                f"{val:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2.0, val),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90 if n_pairs >= 4 else 0,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                zorder=6,
            )

    ax.set_xticks(list(range(len(targets))))
    ax.set_xticklabels([f"R*={t:g}" for t in targets], fontsize=9.0)
    ax.set_xlabel("Matched Recall@10 Target (No Extrapolation)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontsize=10.8, fontweight="bold")
    _, ymax = ax.get_ylim()
    ax.set_ylim(0.0, max(1.3, ymax * 1.18))


def plot_chart3_matched_recall_ratios(
    reports: list[Phase7Report],
    output_dir: str | Path,
    targets: tuple[float, ...] = MATCHED_RECALL_TARGETS,
) -> list[Path]:
    """Chart 3: Matched-recall Mode 1 QPS and Mode 2 p99 ratios without extrapolation."""
    single_rpts = _select_single_thread_reports(reports)
    rust_by_key = {(r.scale_vectors, r.config_id): r for r in single_rpts if r.engine == "rust"}
    faiss_by_key = {(r.scale_vectors, r.config_id): r for r in single_rpts if r.engine == "faiss"}
    paired_keys = sorted(
        set(rust_by_key.keys()) & set(faiss_by_key.keys()),
        key=lambda sc_cfg: (sc_cfg[0], _config_sort_key(sc_cfg[1])),
    )
    if not paired_keys:
        return _render_empty_matched_recall_chart(single_rpts, output_dir)

    pair_comparisons = {
        sc_cfg: compute_matched_recall_comparison(
            rust_by_key[sc_cfg].to_curve_dicts(),
            faiss_by_key[sc_cfg].to_curve_dicts(),
            targets=targets,
        )
        for sc_cfg in paired_keys
    }

    fig, axes_arr = plt.subplots(1, 2, figsize=(13.6, 5.7))
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]
    panel_specs = (
        (
            axes[0],
            "qps_ratio_rust_over_faiss",
            "Mode 1 Pure-Batch QPS Ratio (Rust / FAISS)",
            "Speedup Ratio (>1.0x = Rust faster)",
            True,
        ),
        (
            axes[1],
            "p99_ratio_faiss_over_rust",
            "Mode 2 p99 Latency Speedup (FAISS p99 / Rust p99)",
            "Tail-Latency Ratio (>1.0x = Rust lower p99)",
            False,
        ),
    )
    for ax, ratio_key, title, ylabel, add_labels in panel_specs:
        _plot_matched_ratio_panel(
            ax,
            targets,
            paired_keys,
            pair_comparisons,
            ratio_key=ratio_key,
            title=title,
            ylabel=ylabel,
            add_legend_labels=add_labels,
        )

    finalize_figure_with_top_legend(
        fig,
        axes,
        title="Phase 7: Matched-Recall Performance Ratios (Log-Linear Interpolation Within Observed Range)",
        caption=_build_provenance_caption(single_rpts),
        max_ncol=5,
        legend_y=0.915,
        suptitle_y=0.98,
        legend_fontsize=8.2,
        bottom_margin=0.06,
        top_margin=0.83,
    )
    return save_figure(fig, output_dir, "chart3_matched_recall_ratios", bbox_inches="tight")


def _plot_mt_qps_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Top-left of Chart 4: Multi-thread Mode 1 batch QPS scaling across T."""
    mt_reports = [r for r in reports if r.threads > 1]
    target_configs = sorted(
        {(r.scale_vectors, r.config_id) for r in mt_reports},
        key=lambda sc_cfg: (sc_cfg[0], _config_sort_key(sc_cfg[1])),
    )
    if not target_configs:
        target_configs = sorted(
            {(r.scale_vectors, r.config_id) for r in reports if r.config_id == "C2"}
            or {(r.scale_vectors, r.config_id) for r in reports},
            key=lambda sc_cfg: (sc_cfg[0], _config_sort_key(sc_cfg[1])),
        )[:1]

    if not target_configs:
        render_empty_panel(ax, "Multi-Thread Mode 1 Batch QPS Scaling", "No throughput data loaded.")
        return

    configure_ax(ax, log_x=False, log_y=True, minor_alpha=0.15)
    chosen_sc_cfg = target_configs[-1]
    for sc_cfg in reversed(target_configs):
        if sc_cfg[1] == "C2":
            chosen_sc_cfg = sc_cfg
            break

    scale_sel, cfg_sel = chosen_sc_cfg
    by_et: dict[tuple[str, int], Phase7Report] = {}
    for r in reports:
        if r.scale_vectors != scale_sel or r.config_id != cfg_sel:
            continue
        key = (r.engine, r.threads)
        existing = by_et.get(key)
        if existing is None or (r.compare_mode == "multithread" and existing.compare_mode != "multithread"):
            by_et[key] = r

    common_efs_sets = [
        {p.ef_search for p in r.points if p.mode1_qps is not None and p.mode1_qps > 0.0}
        for r in by_et.values()
    ]
    common_efs = sorted(set.intersection(*common_efs_sets)) if common_efs_sets else []
    if not common_efs:
        render_empty_panel(
            ax,
            f"Multi-Thread Mode 1 Scaling ({_format_scale(scale_sel)} {cfg_sel})",
            "No common efSearch points across thread counts.",
        )
        return

    preferred_efs: list[int] = [ef for ef in (50, 100) if ef in common_efs]
    if not preferred_efs:
        preferred_efs = [common_efs[len(common_efs) // 2]]
    thread_vals = sorted({t for _, t in by_et})
    x_map = {t: idx for idx, t in enumerate(thread_vals)}
    ef_colors = {0: OKABE_ITO_BLUE, 1: OKABE_ITO_ORANGE}

    for ef_idx, ef_target in enumerate(preferred_efs):
        base_color = ef_colors.get(ef_idx, OKABE_ITO_GREEN)
        for engine in ("rust", "faiss"):
            xs: list[int] = []
            ys: list[float] = []
            for t in thread_vals:
                rpt = by_et.get((engine, t))
                if rpt is None:
                    continue
                pt = next((p for p in rpt.points if p.ef_search == ef_target), None)
                if pt is not None and pt.mode1_qps is not None and pt.mode1_qps > 0.0:
                    xs.append(x_map[t])
                    ys.append(pt.mode1_qps)
            if xs:
                est = ENGINE_STYLE[engine]
                line_color = base_color if (engine == "rust" or len(preferred_efs) > 1) else "#333333"
                ax.plot(
                    xs,
                    ys,
                    color=line_color,
                    linestyle="-" if engine == "rust" else "--",
                    marker=est["marker"],
                    linewidth=2.0,
                    markersize=5.6,
                    markerfacecolor=base_color if engine == "rust" else "white",
                    label=f"{est['label_prefix']} ef={ef_target}",
                )

    ax.set_xticks(list(range(len(thread_vals))))
    ax.set_xticklabels([f"T={t}" for t in thread_vals], fontsize=9.0)
    ax.set_xlabel("Thread Count T", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Mode 1 Batch QPS (log scale)", fontsize=9.5, fontweight="bold")
    ax.set_title(
        f"Multi-Thread Mode 1 Batch QPS ({_format_scale(scale_sel)} {cfg_sel})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=7.8, loc="best")
    if len(thread_vals) == 1:
        ax.text(
            0.5,
            0.12,
            "Multi-thread (T=6, 16) runs not yet loaded",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.0,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8F8F8", "edgecolor": "#CCCCCC"},
        )


def _render_hatched_bar_panel(
    ax: Axes,
    entries: list[tuple[str, float, str, str]],
    *,
    value_fmt: Callable[[float], str],
    ylabel: str,
    title: str,
    empty_title: str,
    empty_msg: str,
    ylim_scale: float = 1.18,
    label_fontsize: float = 7.0,
) -> bool:
    """Shared hatched-bar renderer for Rust (solid) vs FAISS (hatched) panels."""
    if not entries:
        render_empty_panel(ax, empty_title, empty_msg)
        return False

    configure_ax(ax, log_x=False, log_y=False, minor_alpha=0.15)
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    hatches = [e[3] for e in entries]

    xs = list(range(len(vals)))
    bars = ax.bar(xs, vals, color=colors, edgecolor="#222222", linewidth=0.7, alpha=0.88)
    for bar, hatch, val in zip(bars, hatches, vals, strict=True):
        if hatch:
            bar.set_hatch(hatch)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            value_fmt(val),
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.0, rotation=28, ha="right")
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontsize=10.2, fontweight="bold")
    _, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax * ylim_scale)
    return True


def _bar_entry_label(rpt: Phase7Report, *, include_threads: bool = False) -> str:
    eng_lbl = "Rust" if rpt.engine == "rust" else "FAISS"
    suffix = f" ({rpt.threads}T)" if include_threads else ""
    return f"{_format_scale(rpt.scale_vectors)} {rpt.config_id}\n{eng_lbl}{suffix}"


def _plot_build_time_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Top-right of Chart 4: Build time (s) with Rust single-thread build asymmetry note."""
    ordered = sorted(
        reports,
        key=lambda r: (r.scale_vectors, _config_sort_key(r.config_id), 0 if r.engine == "rust" else 1, r.threads),
    )
    entries = [
        (
            _bar_entry_label(rpt, include_threads=True),
            rpt.build_time_s,
            _color_for_config(rpt.config_id),
            "" if rpt.engine == "rust" else "//",
        )
        for rpt in ordered
        if rpt.build_time_s > 0.0
    ]
    plotted = _render_hatched_bar_panel(
        ax,
        entries,
        value_fmt=lambda v: f"{v:.1f}s" if v >= 1.0 else f"{v:.3f}s",
        ylabel="Build Time [seconds]",
        title="Index Build Time (Solid=Rust 1T only, Hatched=FAISS OpenMP)",
        empty_title="Index Build Wall-Clock Time",
        empty_msg="No positive build timings present." if reports else "No build data loaded.",
        ylim_scale=1.24,
        label_fontsize=6.8,
    )
    if plotted:
        ax.text(
            0.02,
            0.96,
            "Asymmetry note: Rust HnswIndex::build is strictly 1-thread\n"
            "(deterministic); FAISS IndexHNSWFlat::add supports OpenMP T>1.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.3,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFDF2", "edgecolor": OKABE_ITO_ORANGE, "alpha": 0.9},
        )


def _plot_graph_memory_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Bottom-left of Chart 4: Graph-only memory bytes per vector."""
    single_rpts = _select_single_thread_reports(reports) or reports
    ordered = sorted(
        single_rpts,
        key=lambda r: (r.scale_vectors, _config_sort_key(r.config_id), 0 if r.engine == "rust" else 1),
    )
    entries = [
        (
            _bar_entry_label(rpt),
            rpt.graph_only_bytes_per_vector,
            _color_for_config(rpt.config_id),
            "" if rpt.engine == "rust" else "//",
        )
        for rpt in ordered
        if rpt.graph_only_bytes_per_vector > 0.0
    ]
    _render_hatched_bar_panel(
        ax,
        entries,
        value_fmt=lambda v: f"{v:.1f}",
        ylabel="Graph-Only Memory [B/vector]",
        title="Graph-Only Memory / Vector (Solid=Rust u32 flat, Hatched=FAISS)",
        empty_title="Graph-Only Memory Footprint",
        empty_msg="No graph memory measurements present." if single_rpts else "No memory profile data loaded.",
    )


def _plot_peak_rss_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Bottom-right of Chart 4: Process Peak RSS (PeakWorkingSet64, MiB)."""
    single_rpts = _select_single_thread_reports(reports) or reports
    ordered = sorted(
        (r for r in single_rpts if r.peak_rss_mb is not None and r.peak_rss_mb > 0.0),
        key=lambda r: (r.scale_vectors, _config_sort_key(r.config_id), 0 if r.engine == "rust" else 1),
    )
    entries = [
        (
            _bar_entry_label(rpt),
            float(rpt.peak_rss_mb),
            _color_for_config(rpt.config_id),
            "" if rpt.engine == "rust" else "//",
        )
        for rpt in ordered
        if rpt.peak_rss_mb is not None
    ]
    _render_hatched_bar_panel(
        ax,
        entries,
        value_fmt=lambda v: f"{v:.1f}",
        ylabel="Peak Working Set [MiB]",
        title="Process Peak RSS (Solid=Rust, Hatched=FAISS)",
        empty_title="Process Peak RSS (PeakWorkingSet64)",
        empty_msg="No peak RSS measurements present.",
    )


def plot_chart4_multithread_and_build(
    reports: list[Phase7Report],
    output_dir: str | Path,
) -> list[Path]:
    """Chart 4: Multi-thread Mode 1 QPS scaling + build time & graph-only memory comparison."""
    fig, axes_grid = plt.subplots(2, 2, figsize=(13.8, 9.6))
    _plot_mt_qps_panel(axes_grid[0, 0], reports)
    _plot_build_time_panel(axes_grid[0, 1], reports)
    _plot_graph_memory_panel(axes_grid[1, 0], reports)
    _plot_peak_rss_panel(axes_grid[1, 1], reports)

    fig.suptitle(
        "Phase 7: Multi-Thread Search Scaling, Build Asymmetry & Memory Footprint",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        _build_provenance_caption(reports),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.955))
    return save_figure(fig, output_dir, "chart4_multithread_and_build", bbox_inches="tight")


def generate_all_phase7_charts(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    rust_dir: str | Path | None = None,
    faiss_dir: str | Path | None = None,
    scales: tuple[int, ...] | None = None,
) -> list[Path]:
    """Load Phase 7 results and generate all 4 PNG+SVG chart pairs."""
    reports = load_phase7_reports(
        results_dir=results_dir,
        rust_dir=rust_dir,
        faiss_dir=faiss_dir,
        scales=scales,
    )
    created: list[Path] = []
    created.extend(plot_chart1_recall_vs_p99(reports, output_dir))
    created.extend(plot_chart2_recall_vs_qps(reports, output_dir))
    created.extend(plot_chart3_matched_recall_ratios(reports, output_dir))
    created.extend(plot_chart4_multithread_and_build(reports, output_dir))
    return created


def _parse_scales(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Phase 7 plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot Phase 7 Rust vs FAISS head-to-head benchmark charts (PNG + SVG)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase7"),
        help="Primary directory containing Phase 7 JSON reports (default: results/phase7).",
    )
    parser.add_argument(
        "--rust-dir",
        type=Path,
        default=None,
        help="Optional extra directory to load Rust JSON reports from.",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=None,
        help="Optional extra directory to load FAISS JSON reports from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated PNG/SVG charts (defaults to --results-dir).",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=None,
        help="Optional comma-separated dataset scales to include (e.g. 100000,1000000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase 7 deterministic chart generation."""
    args = parse_args(argv)
    out_dir: Path = args.output_dir if args.output_dir is not None else args.results_dir
    created = generate_all_phase7_charts(
        results_dir=args.results_dir,
        output_dir=out_dir,
        rust_dir=args.rust_dir,
        faiss_dir=args.faiss_dir,
        scales=_parse_scales(args.scales),
    )
    print(f"[Phase 7 Plotter] Generated {len(created)} chart files in {out_dir}")
    for path in created:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
