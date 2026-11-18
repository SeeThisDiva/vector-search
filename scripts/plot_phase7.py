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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark_faiss import (
    MATCHED_RECALL_TARGETS,
    compute_matched_recall_comparison,
)

plt.rcParams["svg.hashsalt"] = "phase7"
plt.rcParams["font.family"] = "DejaVu Sans"

CONFIG_ORDER: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C2-nkp")
CONFIG_COLORS: dict[str, str] = {
    "C1": "#0072B2",
    "C2": "#D55E00",
    "C3": "#009E73",
    "C4": "#CC79A7",
    "C2-nkp": "#E69F00",
}
FALLBACK_COLORS: tuple[str, ...] = (
    "#56B4E9",
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
    mode2_mean_ms: float | None
    ffi_delta_us: float | None
    dist_evals_mean: float | None


@dataclass(frozen=True)
class Phase7Report:
    """Normalized benchmark report from Rust or FAISS."""

    source_path: str
    phase: int
    engine: str
    compare_mode: str
    config_id: str
    scale_vectors: int
    m: int
    ef_construction: int
    keep_pruned_shrink: bool | None
    threads: int
    affinity_mask: str
    power_mode: str
    cpu_model: str
    git_commit: str
    git_dirty: bool
    build_reps: int
    query_reps: int
    build_time_s: float
    build_qps: float
    graph_only_bytes_per_vector: float
    peak_rss_mb: float | None
    faiss_version: str | None
    faiss_simd_level: str | None
    points: list[SweepPoint]

    @property
    def config_desc(self) -> str:
        suffix = ", no-kp-shrink" if (self.engine == "rust" and self.keep_pruned_shrink is False) else ""
        return f"{self.config_id} (M={self.m}, efC={self.ef_construction}{suffix})"

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
    if "faiss" in data and isinstance(data["faiss"], dict):
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
    mapping = {(8, 100): "C1", (16, 100): "C2", (32, 200): "C3", (48, 400): "C4"}
    return mapping.get((m, efc), f"M{m}_efC{efc}")


def _opt_float(val: Any) -> float | None:
    if val is None:
        return None
    return float(val)


def _first_float(*vals: Any) -> float | None:
    for v in vals:
        if v is not None:
            return float(v)
    return None


def _parse_sweep_points(raw_points: list[dict[str, Any]]) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for pt in raw_points:
        if not isinstance(pt, dict):
            continue
        ef = int(pt.get("ef_search") or 0)
        recall_opt = _first_float(pt.get("recall_at_k"), pt.get("recall"))
        recall = recall_opt if recall_opt is not None else 0.0

        raw_m1 = pt.get("mode1_batch")
        raw_m2 = pt.get("mode2_per_query")
        raw_med = pt.get("median_latency")
        mode1: dict[str, Any] = raw_m1 if isinstance(raw_m1, dict) else {}
        mode2: dict[str, Any] = raw_m2 if isinstance(raw_m2, dict) else {}
        median_lat: dict[str, Any] = raw_med if isinstance(raw_med, dict) else {}

        mode1_qps = _first_float(
            mode1.get("qps"),
            pt.get("mode1_qps"),
            pt.get("qps"),
            median_lat.get("qps"),
        )
        mode1_mean = _first_float(
            mode1.get("mean_ms"),
            pt.get("mode1_mean_ms"),
            pt.get("mean_ms"),
            median_lat.get("mean_ms"),
        )
        mode2_p99 = _first_float(
            mode2.get("p99_ms"),
            pt.get("mode2_p99_ms"),
            pt.get("p99_ms"),
            median_lat.get("p99_ms"),
        )
        mode2_mean = _first_float(
            mode2.get("mean_ms"),
            pt.get("mode2_mean_ms"),
            pt.get("mean_ms"),
            median_lat.get("mean_ms"),
        )
        ffi_delta = _opt_float(pt.get("ffi_delta_us"))
        ndis = _opt_float(pt.get("dist_evals_mean"))

        points.append(
            SweepPoint(
                ef_search=ef,
                recall_at_k=recall,
                mode1_qps=mode1_qps,
                mode1_mean_ms=mode1_mean,
                mode2_p99_ms=mode2_p99,
                mode2_mean_ms=mode2_mean,
                ffi_delta_us=ffi_delta,
                dist_evals_mean=ndis,
            )
        )
    points.sort(key=lambda p: p.ef_search)
    return points


def _parse_single_report(data: dict[str, Any], source_path: Path) -> Phase7Report | None:
    raw_sweep = data.get("ef_search_sweep")
    if not isinstance(raw_sweep, list) or not raw_sweep:
        return None

    engine = _infer_engine(data)
    if engine == "unknown":
        return None

    hnsw_cfg = data.get("hnsw_config", {}) if isinstance(data.get("hnsw_config"), dict) else {}
    m = int(hnsw_cfg.get("m", 0))
    efc = int(hnsw_cfg.get("ef_construction", 0))
    kps_raw = hnsw_cfg.get("keep_pruned_shrink")
    keep_pruned_shrink = bool(kps_raw) if kps_raw is not None else None
    config_id = _infer_config_id(data, engine, m, efc, keep_pruned_shrink)

    ds = data.get("dataset", {}) if isinstance(data.get("dataset"), dict) else {}
    scale = int(data.get("scale_vectors", ds.get("scale_vectors", ds.get("num_base", 0))))
    if scale <= 0:
        return None

    env = data.get("environment", {}) if isinstance(data.get("environment"), dict) else {}
    hw = data.get("hardware", {}) if isinstance(data.get("hardware"), dict) else {}
    ib = data.get("index_build", {}) if isinstance(data.get("index_build"), dict) else {}
    mem = data.get("memory_profile", {}) if isinstance(data.get("memory_profile"), dict) else {}
    fx = data.get("faiss", {}) if isinstance(data.get("faiss"), dict) else {}

    threads = int(data.get("threads", env.get("threads", 1)))
    compare_mode = str(
        data.get("compare_mode", "multithread" if threads > 1 else "single")
    ).lower()

    points = _parse_sweep_points(raw_sweep)
    if not points:
        return None

    return Phase7Report(
        source_path=str(source_path),
        phase=int(data.get("phase", 7)),
        engine=engine,
        compare_mode=compare_mode,
        config_id=config_id,
        scale_vectors=scale,
        m=m,
        ef_construction=efc,
        keep_pruned_shrink=keep_pruned_shrink,
        threads=threads,
        affinity_mask=str(env.get("processor_affinity_mask", "not_set")),
        power_mode=str(env.get("power_mode", "unknown")),
        cpu_model=str(hw.get("cpu_model", env.get("cpu_model", "Unknown CPU"))),
        git_commit=str(data.get("git_commit", "unknown")),
        git_dirty=bool(data.get("git_dirty", False)),
        build_reps=int(data.get("build_reps", 1)),
        query_reps=int(data.get("query_reps", data.get("sweep_runs", 5))),
        build_time_s=float(ib.get("build_time_s", 0.0)),
        build_qps=float(ib.get("build_qps", 0.0)),
        graph_only_bytes_per_vector=float(
            mem.get("graph_only_bytes_per_vector", data.get("graph_only_bytes_per_vector", 0.0))
        ),
        peak_rss_mb=_opt_float(mem.get("peak_rss_mb")),
        faiss_version=str(fx["version"]) if "version" in fx else None,
        faiss_simd_level=str(fx["simd_level_name"]) if "simd_level_name" in fx else None,
        points=points,
    )


def _expand_report_json(data: dict[str, Any], path: Path) -> list[Phase7Report]:
    """Parse either a standalone report or a multi-thread bundle report."""
    reports: list[Phase7Report] = []
    single = _parse_single_report(data, path)
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
            parsed = _parse_single_report(merged, path)
            if parsed is not None:
                reports.append(parsed)
    return reports


def load_phase7_reports(
    results_dir: Path,
    *,
    rust_dir: Path | None = None,
    faiss_dir: Path | None = None,
    scales: tuple[int, ...] | None = None,
) -> list[Phase7Report]:
    """Discover, normalize, and deduplicate Rust and FAISS Phase 7 JSON reports."""
    search_dirs: list[Path] = [results_dir]
    if rust_dir is not None:
        search_dirs.append(rust_dir)
    if faiss_dir is not None:
        search_dirs.append(faiss_dir)

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
            for rpt in _expand_report_json(payload, json_path):
                if scales is not None and rpt.scale_vectors not in scales:
                    continue
                raw_reports.append(rpt)

    # Deduplicate by (engine, scale, config_id, threads, compare_mode), preferring phase 7 over phase 4
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


def _configure_ax(ax: Axes, *, log_x: bool = False, log_y: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.28, color="#888888")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.15, color="#AAAAAA")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def _collect_shared_legend(axes: list[Axes]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        h_list, l_list = ax.get_legend_handles_labels()
        for handle, label in zip(h_list, l_list, strict=True):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    return handles, labels


def _render_empty_panel(ax: Axes, title: str, message: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#555555",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def _save_figure(fig: Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(svg_path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return [png_path, svg_path]


def _compute_recall_xlim(recalls: list[float]) -> tuple[float, float]:
    if not recalls:
        return (0.65, 1.005)
    min_r = min(recalls)
    lo = max(0.0, math.floor((min_r - 0.03) * 20.0) / 20.0)
    if lo > 0.88:
        lo = 0.85
    return (lo, 1.004)


def plot_chart1_recall_vs_p99(
    reports: list[Phase7Report],
    output_dir: Path,
) -> list[Path]:
    """Chart 1: Mode 2 Recall@10 vs p99 latency (log scale), Rust vs FAISS."""
    single_rpts = _select_single_thread_reports(reports)
    scales = sorted({r.scale_vectors for r in single_rpts})
    if not scales:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        _render_empty_panel(
            ax,
            "Phase 7: Mode 2 Recall@10 vs p99 Latency",
            "No single-thread Phase 7 reports found.",
        )
        return _save_figure(fig, output_dir, "chart1_recall_vs_p99")

    ncols = len(scales)
    fig, axes_raw = plt.subplots(1, ncols, figsize=(6.8 * ncols + 0.6, 5.7), squeeze=False)
    axes: list[Axes] = [axes_raw[0, idx] for idx in range(ncols)]

    all_ffi_deltas: list[float] = []
    for ax, scale in zip(axes, scales, strict=True):
        _configure_ax(ax, log_x=False, log_y=True)
        scale_rpts = [r for r in single_rpts if r.scale_vectors == scale]
        recalls: list[float] = []

        for rpt in scale_rpts:
            pts = [p for p in rpt.points if p.mode2_p99_ms is not None and p.mode2_p99_ms > 0.0]
            if not pts:
                continue
            xs = [p.recall_at_k for p in pts]
            ys = [float(p.mode2_p99_ms) for p in pts if p.mode2_p99_ms is not None]
            recalls.extend(xs)
            for p in pts:
                if rpt.engine == "faiss" and p.ffi_delta_us is not None:
                    all_ffi_deltas.append(p.ffi_delta_us)

            color = _color_for_config(rpt.config_id)
            est = ENGINE_STYLE.get(rpt.engine, ENGINE_STYLE["faiss"])
            mfc = color if rpt.engine == "rust" else "white"
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=est["linestyle"],
                marker=est["marker"],
                linewidth=est["linewidth"],
                markersize=est["markersize"],
                markerfacecolor=mfc,
                markeredgewidth=1.4,
                alpha=est["alpha"],
                zorder=est["zorder"],
                label=f"{est['label_prefix']} {rpt.config_id} (M={rpt.m}, efC={rpt.ef_construction})",
            )

        if not recalls:
            _render_empty_panel(
                ax,
                f"N = {_format_scale(scale)} vectors",
                "No Mode 2 p99 latency points present",
            )
        else:
            ax.set_xlim(*_compute_recall_xlim(recalls))
            ax.set_xlabel("Recall@10", fontsize=10, fontweight="bold")
            ax.set_title(f"N = {_format_scale(scale)} vectors", fontsize=11, fontweight="bold")

    axes[0].set_ylabel("Mode 2 p99 Latency [ms] (log scale)", fontsize=10, fontweight="bold")
    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=min(4, max(1, len(handles))),
            frameon=False,
            fontsize=8.0,
        )

    ffi_note = ""
    if all_ffi_deltas:
        med_ffi = sorted(all_ffi_deltas)[len(all_ffi_deltas) // 2]
        ffi_note = f" | FAISS Mode 2 search_c includes median FFI delta {med_ffi:.2f} us/q"
    fig.suptitle(
        "Phase 7: Mode 2 Recall@10 vs p99 Tail Latency (Single-Thread Pinned)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    caption = _build_provenance_caption(single_rpts) + ffi_note
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    return _save_figure(fig, output_dir, "chart1_recall_vs_p99")


def plot_chart2_recall_vs_qps(
    reports: list[Phase7Report],
    output_dir: Path,
) -> list[Path]:
    """Chart 2: Mode 1 pure-batch Recall@10 vs single-thread QPS (zero FFI tax)."""
    single_rpts = _select_single_thread_reports(reports)
    scales = sorted({r.scale_vectors for r in single_rpts})
    if not scales:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        _render_empty_panel(
            ax,
            "Phase 7: Mode 1 Pure-Batch Recall@10 vs QPS",
            "No single-thread Phase 7 reports found.",
        )
        return _save_figure(fig, output_dir, "chart2_recall_vs_qps")

    ncols = len(scales)
    fig, axes_raw = plt.subplots(1, ncols, figsize=(6.8 * ncols + 0.6, 5.7), squeeze=False)
    axes: list[Axes] = [axes_raw[0, idx] for idx in range(ncols)]

    for ax, scale in zip(axes, scales, strict=True):
        _configure_ax(ax, log_x=False, log_y=True)
        scale_rpts = [r for r in single_rpts if r.scale_vectors == scale]
        recalls: list[float] = []

        for rpt in scale_rpts:
            pts = [p for p in rpt.points if p.mode1_qps is not None and p.mode1_qps > 0.0]
            if not pts:
                continue
            xs = [p.recall_at_k for p in pts]
            ys = [float(p.mode1_qps) for p in pts if p.mode1_qps is not None]
            recalls.extend(xs)

            color = _color_for_config(rpt.config_id)
            est = ENGINE_STYLE.get(rpt.engine, ENGINE_STYLE["faiss"])
            mfc = color if rpt.engine == "rust" else "white"
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=est["linestyle"],
                marker=est["marker"],
                linewidth=est["linewidth"],
                markersize=est["markersize"],
                markerfacecolor=mfc,
                markeredgewidth=1.4,
                alpha=est["alpha"],
                zorder=est["zorder"],
                label=f"{est['label_prefix']} {rpt.config_id} (M={rpt.m}, efC={rpt.ef_construction})",
            )

        ax.set_xlim(*_compute_recall_xlim(recalls))
        ax.set_xlabel("Recall@10", fontsize=10, fontweight="bold")
        ax.set_title(f"N = {_format_scale(scale)} vectors", fontsize=11, fontweight="bold")

    axes[0].set_ylabel("Mode 1 Pure-Batch QPS [queries/s] (log scale)", fontsize=10, fontweight="bold")
    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=min(4, max(1, len(handles))),
            frameon=False,
            fontsize=8.0,
        )

    fig.suptitle(
        "Phase 7: Mode 1 Pure-Batch Recall@10 vs Throughput (Zero Per-Query FFI Tax)",
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
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    return _save_figure(fig, output_dir, "chart2_recall_vs_qps")


def plot_chart3_matched_recall_ratios(
    reports: list[Phase7Report],
    output_dir: Path,
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

    fig, axes_arr = plt.subplots(1, 2, figsize=(13.6, 5.7))
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]

    if not paired_keys:
        for ax, title in zip(
            axes,
            (
                "Mode 1 Batch QPS Ratio (Rust / FAISS)",
                "Mode 2 p99 Latency Ratio (FAISS / Rust)",
            ),
            strict=True,
        ):
            _render_empty_panel(
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
        return _save_figure(fig, output_dir, "chart3_matched_recall_ratios")

    pair_comparisons: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for sc_cfg in paired_keys:
        r_rpt = rust_by_key[sc_cfg]
        f_rpt = faiss_by_key[sc_cfg]
        pair_comparisons[sc_cfg] = compute_matched_recall_comparison(
            r_rpt.to_curve_dicts(),
            f_rpt.to_curve_dicts(),
            targets=targets,
        )

    x_positions = list(range(len(targets)))
    n_pairs = len(paired_keys)
    total_group_width = 0.78
    bar_width = total_group_width / max(1, n_pairs)

    for ax_idx, (ax, ratio_key, title, ylabel) in enumerate(
        (
            (
                axes[0],
                "qps_ratio_rust_over_faiss",
                "Mode 1 Pure-Batch QPS Ratio (Rust / FAISS)",
                "Speedup Ratio (>1.0x = Rust faster)",
            ),
            (
                axes[1],
                "p99_ratio_faiss_over_rust",
                "Mode 2 p99 Latency Speedup (FAISS p99 / Rust p99)",
                "Tail-Latency Ratio (>1.0x = Rust lower p99)",
            ),
        )
    ):
        _configure_ax(ax, log_x=False, log_y=False)
        ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.2, alpha=0.75, zorder=2)

        for p_idx, sc_cfg in enumerate(paired_keys):
            scale, cfg = sc_cfg
            comps = pair_comparisons[sc_cfg]
            color = _color_for_config(cfg)
            hatch = "//" if scale >= 1_000_000 else ""
            offset = (p_idx - (n_pairs - 1) / 2.0) * bar_width
            xs_bars: list[float] = []
            ys_bars: list[float] = []
            for t_idx, comp in enumerate(comps):
                val = comp.get(ratio_key)
                if val is not None and val > 0.0:
                    xs_bars.append(float(t_idx) + offset)
                    ys_bars.append(float(val))

            label = f"{_format_scale(scale)} {cfg}" if ax_idx == 0 else None
            if xs_bars:
                bars = ax.bar(
                    xs_bars,
                    ys_bars,
                    width=bar_width * 0.92,
                    color=color,
                    hatch=hatch,
                    edgecolor="#222222",
                    linewidth=0.7,
                    alpha=0.88,
                    label=label,
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
                        bbox={
                            "boxstyle": "round,pad=0.10",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.88,
                        },
                        zorder=6,
                    )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"R*={t:g}" for t in targets], fontsize=9.0)
        ax.set_xlabel("Matched Recall@10 Target (No Extrapolation)", fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        _, ymax = ax.get_ylim()
        ax.set_ylim(0.0, max(1.3, ymax * 1.18))

    handles, labels = _collect_shared_legend(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=min(5, max(1, len(handles))),
            frameon=False,
            fontsize=8.2,
        )
    fig.suptitle(
        "Phase 7: Matched-Recall Performance Ratios (Log-Linear Interpolation Within Observed Range)",
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
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.83))
    return _save_figure(fig, output_dir, "chart3_matched_recall_ratios")


def _plot_mt_qps_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Top-left of Chart 4: Multi-thread Mode 1 batch QPS scaling across T."""
    mt_reports = [r for r in reports if r.threads > 1]
    target_configs = sorted(
        {(r.scale_vectors, r.config_id) for r in mt_reports},
        key=lambda sc_cfg: (sc_cfg[0], _config_sort_key(sc_cfg[1])),
    )
    if not target_configs:
        # Fall back to C2 or whatever single-thread configs exist so panel stays informative
        target_configs = sorted(
            {(r.scale_vectors, r.config_id) for r in reports if r.config_id == "C2"}
            or {(r.scale_vectors, r.config_id) for r in reports},
            key=lambda sc_cfg: (sc_cfg[0], _config_sort_key(sc_cfg[1])),
        )[:1]

    if not target_configs:
        _render_empty_panel(ax, "Multi-Thread Mode 1 Batch QPS Scaling", "No throughput data loaded.")
        return

    _configure_ax(ax, log_x=False, log_y=True)

    # Prefer C2 if present among MT configs; show largest scale if multiple
    chosen_sc_cfg = target_configs[-1]
    for sc_cfg in reversed(target_configs):
        if sc_cfg[1] == "C2":
            chosen_sc_cfg = sc_cfg
            break

    scale_sel, cfg_sel = chosen_sc_cfg
    relevant = [
        r
        for r in reports
        if r.scale_vectors == scale_sel and r.config_id == cfg_sel
    ]
    # Keep one report per (engine, threads), preferring compare_mode == "multithread"
    by_et: dict[tuple[str, int], Phase7Report] = {}
    for r in relevant:
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
        _render_empty_panel(
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
    ef_styles = {
        0: ("#0072B2", "-"),
        1: ("#D55E00", "--"),
    }

    for ef_idx, ef_target in enumerate(preferred_efs):
        base_color, _ = ef_styles.get(ef_idx, ("#009E73", ":"))
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
                ax.plot(
                    xs,
                    ys,
                    color=base_color if engine == "rust" else "#333333" if len(preferred_efs) == 1 else base_color,
                    linestyle="-" if engine == "rust" else "--",
                    marker=est["marker"],
                    linewidth=2.0,
                    markersize=5.6,
                    markerfacecolor=(base_color if engine == "rust" else "white"),
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


def _plot_build_time_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Top-right of Chart 4: Build time (s) with Rust single-thread build asymmetry note."""
    if not reports:
        _render_empty_panel(ax, "Index Build Wall-Clock Time", "No build data loaded.")
        return

    _configure_ax(ax, log_x=False, log_y=False)
    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []

    ordered = sorted(
        reports,
        key=lambda r: (
            r.scale_vectors,
            _config_sort_key(r.config_id),
            0 if r.engine == "rust" else 1,
            r.threads,
        ),
    )
    for rpt in ordered:
        if rpt.build_time_s <= 0.0:
            continue
        eng_lbl = "Rust" if rpt.engine == "rust" else "FAISS"
        t_lbl = f"{rpt.threads}T"
        labels.append(f"{_format_scale(rpt.scale_vectors)} {rpt.config_id}\n{eng_lbl} ({t_lbl})")
        vals.append(rpt.build_time_s)
        colors.append(_color_for_config(rpt.config_id))
        hatches.append("" if rpt.engine == "rust" else "//")

    if not vals:
        _render_empty_panel(ax, "Index Build Wall-Clock Time", "No positive build timings present.")
        return

    xs = list(range(len(vals)))
    bars = ax.bar(xs, vals, color=colors, edgecolor="#222222", linewidth=0.7, alpha=0.88)
    for bar, hatch in zip(bars, hatches, strict=True):
        if hatch:
            bar.set_hatch(hatch)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.1f}s" if val >= 1.0 else f"{val:.3f}s",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.0, rotation=28, ha="right")
    ax.set_ylabel("Build Time [seconds]", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "Index Build Time (Solid=Rust 1T only, Hatched=FAISS OpenMP)",
        fontsize=10.3,
        fontweight="bold",
    )
    _, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax * 1.24)
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
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFDF2", "edgecolor": "#D55E00", "alpha": 0.9},
    )


def _plot_graph_memory_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Bottom-left of Chart 4: Graph-only memory bytes per vector."""
    single_rpts = _select_single_thread_reports(reports) or reports
    if not single_rpts:
        _render_empty_panel(ax, "Graph-Only Memory Footprint", "No memory profile data loaded.")
        return

    _configure_ax(ax, log_x=False, log_y=False)
    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []

    for rpt in sorted(
        single_rpts,
        key=lambda r: (r.scale_vectors, _config_sort_key(r.config_id), 0 if r.engine == "rust" else 1),
    ):
        if rpt.graph_only_bytes_per_vector <= 0.0:
            continue
        eng_lbl = "Rust" if rpt.engine == "rust" else "FAISS"
        labels.append(f"{_format_scale(rpt.scale_vectors)} {rpt.config_id}\n{eng_lbl}")
        vals.append(rpt.graph_only_bytes_per_vector)
        colors.append(_color_for_config(rpt.config_id))
        hatches.append("" if rpt.engine == "rust" else "//")

    if not vals:
        _render_empty_panel(ax, "Graph-Only Memory Footprint", "No graph memory measurements present.")
        return

    xs = list(range(len(vals)))
    bars = ax.bar(xs, vals, color=colors, edgecolor="#222222", linewidth=0.7, alpha=0.88)
    for bar, hatch in zip(bars, hatches, strict=True):
        if hatch:
            bar.set_hatch(hatch)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.0, rotation=28, ha="right")
    ax.set_ylabel("Graph-Only Memory [B/vector]", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "Graph-Only Memory / Vector (Solid=Rust u32 flat, Hatched=FAISS)",
        fontsize=10.2,
        fontweight="bold",
    )
    _, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax * 1.18)


def _plot_peak_rss_panel(ax: Axes, reports: list[Phase7Report]) -> None:
    """Bottom-right of Chart 4: Process Peak RSS (PeakWorkingSet64, MiB)."""
    single_rpts = _select_single_thread_reports(reports) or reports
    rss_rpts = [r for r in single_rpts if r.peak_rss_mb is not None and r.peak_rss_mb > 0.0]
    if not rss_rpts:
        _render_empty_panel(ax, "Process Peak RSS (PeakWorkingSet64)", "No peak RSS measurements present.")
        return

    _configure_ax(ax, log_x=False, log_y=False)
    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []

    for rpt in sorted(
        rss_rpts,
        key=lambda r: (r.scale_vectors, _config_sort_key(r.config_id), 0 if r.engine == "rust" else 1),
    ):
        assert rpt.peak_rss_mb is not None
        eng_lbl = "Rust" if rpt.engine == "rust" else "FAISS"
        labels.append(f"{_format_scale(rpt.scale_vectors)} {rpt.config_id}\n{eng_lbl}")
        vals.append(rpt.peak_rss_mb)
        colors.append(_color_for_config(rpt.config_id))
        hatches.append("" if rpt.engine == "rust" else "//")

    xs = list(range(len(vals)))
    bars = ax.bar(xs, vals, color=colors, edgecolor="#222222", linewidth=0.7, alpha=0.88)
    for bar, hatch in zip(bars, hatches, strict=True):
        if hatch:
            bar.set_hatch(hatch)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.0, rotation=28, ha="right")
    ax.set_ylabel("Peak Working Set [MiB]", fontsize=9.5, fontweight="bold")
    ax.set_title("Process Peak RSS (Solid=Rust, Hatched=FAISS)", fontsize=10.2, fontweight="bold")
    _, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax * 1.18)


def plot_chart4_multithread_and_build(
    reports: list[Phase7Report],
    output_dir: Path,
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
    return _save_figure(fig, output_dir, "chart4_multithread_and_build")


def generate_all_phase7_charts(
    results_dir: Path,
    output_dir: Path,
    *,
    rust_dir: Path | None = None,
    faiss_dir: Path | None = None,
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
