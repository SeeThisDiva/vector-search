#!/usr/bin/env python3
"""Generate publication-grade PNG and SVG charts for Phase 6 quantized HNSW benchmarks.

Reads Phase 6 per-arm JSON reports (`phase6_*.json` or `arm_*.json`) from
`--results-dir`, plus optional Phase 4 `f32` baseline reports (`scale_*.json`)
from `--phase4-dir` or `--results-dir`.

Generates 4 deterministic, publication-grade charts (both 300 DPI PNG and SVG):
1. `chart1_recall_vs_p99.{png,svg}`: Recall@10 vs p99 latency (ms, log y-axis).
2. `chart2_recall_vs_qps.{png,svg}`: Recall@10 vs single-thread QPS (log y-axis).
3. `chart3_memory_vs_recall.{png,svg}`: Total payload (B/vec) & query-serving Peak RSS (MiB)
   vs Recall@10 at matched `p99 <= 1.0 ms` (or matched `efSearch = 100`), with Pareto frontier.
4. `chart4_rerank_ablation.{png,svg}`: Recall@10 recovery and mean latency vs rerank
   depth `R` across SQ8, PQ16, and PQ32.

Handles partial/subset runs (e.g., SQ8-only without PQ or without rerank) cleanly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

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
    OKABE_ITO_YELLOW,
    compute_pareto_frontier,
    compute_recall_xlim,
    configure_ax,
    configure_matplotlib,
    discover_scale,
    finalize_figure_with_top_legend,
    safe_float,
    safe_int,
    safe_opt_float,
    save_figure,
)

configure_matplotlib(svg_hashsalt="vector-search-phase6")

_PNG_METADATA: dict[str, Any] = {"Software": "vector-search/plot_phase6"}

QUANTIZER_ORDER: dict[str, int] = {"f32": 0, "sq8": 1, "pq16": 2, "pq32": 3}
STORAGE_ORDER: dict[str, int] = {"f32": 0, "only": 1, "rerank": 2}
BUILD_ORDER: dict[str, int] = {"f32_graph": 0, "quantized_direct": 1}

# Okabe-Ito colorblind-safe palette keyed by (quantizer_kind, storage_mode, build_mode).
ARM_STYLE_MAP: dict[tuple[str, str, str], dict[str, Any]] = {
    ("f32", "f32", "f32_graph"): {
        "color": OKABE_ITO_BLACK,
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 5.2,
        "zorder": 8,
    },
    ("sq8", "only", "f32_graph"): {
        "color": OKABE_ITO_BLUE,
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 5.0,
        "zorder": 6,
    },
    ("sq8", "only", "quantized_direct"): {
        "color": OKABE_ITO_SKY,
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.9,
        "markersize": 5.2,
        "zorder": 5,
    },
    ("sq8", "rerank", "f32_graph"): {
        "color": OKABE_ITO_GREEN,
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.0,
        "markersize": 4.8,
        "zorder": 7,
    },
    ("sq8", "rerank", "quantized_direct"): {
        "color": "#66C2A5",
        "linestyle": "--",
        "marker": "v",
        "linewidth": 1.8,
        "markersize": 5.0,
        "zorder": 5,
    },
    ("pq16", "only", "f32_graph"): {
        "color": OKABE_ITO_ORANGE,
        "linestyle": "-",
        "marker": "v",
        "linewidth": 2.0,
        "markersize": 5.2,
        "zorder": 6,
    },
    ("pq16", "only", "quantized_direct"): {
        "color": "#FC8D62",
        "linestyle": "--",
        "marker": "<",
        "linewidth": 1.8,
        "markersize": 5.0,
        "zorder": 5,
    },
    ("pq16", "rerank", "f32_graph"): {
        "color": OKABE_ITO_AMBER,
        "linestyle": "-",
        "marker": "P",
        "linewidth": 2.0,
        "markersize": 5.4,
        "zorder": 6,
    },
    ("pq16", "rerank", "quantized_direct"): {
        "color": OKABE_ITO_YELLOW,
        "linestyle": "--",
        "marker": "X",
        "linewidth": 1.8,
        "markersize": 5.2,
        "zorder": 5,
    },
    ("pq32", "only", "f32_graph"): {
        "color": OKABE_ITO_PINK,
        "linestyle": "-",
        "marker": ">",
        "linewidth": 2.0,
        "markersize": 5.2,
        "zorder": 6,
    },
    ("pq32", "only", "quantized_direct"): {
        "color": "#E78AC3",
        "linestyle": "--",
        "marker": "h",
        "linewidth": 1.8,
        "markersize": 5.2,
        "zorder": 5,
    },
    ("pq32", "rerank", "f32_graph"): {
        "color": "#7570B3",
        "linestyle": "-",
        "marker": "*",
        "linewidth": 2.1,
        "markersize": 6.4,
        "zorder": 7,
    },
    ("pq32", "rerank", "quantized_direct"): {
        "color": "#9E9AC8",
        "linestyle": "--",
        "marker": "p",
        "linewidth": 1.8,
        "markersize": 5.2,
        "zorder": 5,
    },
}

DEFAULT_ARM_STYLE: dict[str, Any] = {
    "color": "#666666",
    "linestyle": "-",
    "marker": "o",
    "linewidth": 1.6,
    "markersize": 4.5,
    "zorder": 4,
}

_SCATTER_CANDIDATES: tuple[tuple[int, int, str], ...] = (
    (8, 8, "left"),
    (8, -10, "left"),
    (-8, 8, "right"),
    (-8, -10, "right"),
    (0, 12, "center"),
    (0, -14, "center"),
    (10, 18, "left"),
    (-10, 18, "right"),
    (10, -20, "left"),
    (-10, -20, "right"),
    (0, 26, "center"),
    (0, -28, "center"),
    (16, 28, "left"),
    (-16, 28, "right"),
    (16, -32, "left"),
    (-16, -32, "right"),
    (0, 38, "center"),
    (0, -40, "center"),
    (24, 34, "left"),
    (-24, 34, "right"),
    (24, -44, "left"),
    (-24, -44, "right"),
    (0, -54, "center"),
    (32, -56, "left"),
    (-32, -56, "right"),
    (0, -68, "center"),
    (-55, -24, "right"),
    (55, -24, "left"),
    (-65, -45, "right"),
    (65, -45, "left"),
)

CHART_STEMS: tuple[str, ...] = (
    "chart1_recall_vs_p99",
    "chart2_recall_vs_qps",
    "chart3_memory_vs_recall",
    "chart4_rerank_ablation",
)


@dataclass(frozen=True)
class Phase6SweepPoint:
    """Single ``(ef_search, rerank_top_r)`` benchmark operating point."""

    ef_search: int
    effective_ef_search: int
    rerank_top_r: int | None
    is_full_beam: bool
    recall_at_k: float
    qps: float
    mean_ms: float
    p99_ms: float


@dataclass(frozen=True)
class Phase6ArmReport:
    """Normalized representation of a Phase 6 arm JSON or Phase 4 ``f32`` baseline JSON."""

    arm_id: str
    filename: str
    quantizer_kind: str
    storage_mode: str
    build_mode: str
    m: int
    ef_construction: int
    scale: int
    k: int
    num_queries: int
    steady_state_bytes_per_vector: float
    build_peak_rss_mb: float | None
    query_serving_peak_rss_mb: float | None
    reps: int
    cpu_model: str
    os_name: str
    affinity_mask: str
    git_commit: str
    git_dirty: bool
    dataset_description: str
    source_phase: int
    sweep: tuple[Phase6SweepPoint, ...]


@dataclass(frozen=True)
class LoadedPhase6Data:
    """Loaded and normalized Phase 6 benchmark collection for a single scale."""

    scale: int
    arms: tuple[Phase6ArmReport, ...]


def _dict_field(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key)
    return val if isinstance(val, dict) else {}


def _normalize_quantizer_kind(raw: Any, arm_id: str, data: dict[str, Any]) -> str:
    text = str(raw).strip().lower() if raw is not None else ""
    aliases = {
        "f32": "f32",
        "float32": "f32",
        "raw": "f32",
        "sq8": "sq8",
        "sq": "sq8",
        "scalar8": "sq8",
        "pq16": "pq16",
        "pq_16": "pq16",
        "pq32": "pq32",
        "pq_32": "pq32",
    }
    if text in aliases:
        return aliases[text]
    if text == "pq":
        pq_cfg = _dict_field(data, "pq_config")
        m_pq = safe_int(data.get("m_pq", pq_cfg.get("m_pq")), default=0)
        if m_pq > 0:
            return f"pq{m_pq}"

    lowered = arm_id.lower()
    for candidate in ("pq32", "pq16", "sq8", "f32"):
        if candidate in lowered:
            return candidate
    return text if text else "sq8"


def _normalize_storage_mode(raw: Any, arm_id: str, quantizer_kind: str) -> str:
    if quantizer_kind == "f32" and (raw is None or str(raw).strip() == ""):
        return "f32"
    text = str(raw).strip().lower() if raw is not None else ""
    if text in {"f32", "raw", "baseline"}:
        return "f32"
    if text in {
        "only",
        "sq8only",
        "sq8_only",
        "pqonly",
        "pq_only",
        "pq16only",
        "pq32only",
        "memory_first",
        "quantized_only",
    }:
        return "only"
    if text in {
        "rerank",
        "with_rerank",
        "withrerank",
        "sq8withrerank",
        "sq8_with_rerank",
        "pqwithrerank",
        "pq_with_rerank",
        "pq16withrerank",
        "pq32withrerank",
        "accuracy_first",
    }:
        return "rerank"

    lowered = arm_id.lower()
    if "rerank" in lowered:
        return "rerank"
    if "only" in lowered:
        return "only"
    return "f32" if quantizer_kind == "f32" else "only"


def _normalize_build_mode(raw: Any, arm_id: str) -> str:
    text = str(raw).strip().lower() if raw is not None else ""
    if text in {
        "f32_graph",
        "f32graph",
        "float32_graph_then_quantize",
        "float32graphthenquantize",
        "f32",
    }:
        return "f32_graph"
    if text in {
        "quantized_direct",
        "quantizeddirect",
        "qbuild",
        "q_direct",
        "direct",
        "streaming",
        "sq8_streaming",
    }:
        return "quantized_direct"

    lowered = arm_id.lower()
    if any(tok in lowered for tok in ("qbuild", "q_direct", "quantized_direct", "streaming")):
        return "quantized_direct"
    return "f32_graph"


def _parse_rerank_field(
    pt: dict[str, Any],
    top_rerank: Any,
    top_full_beam: bool,
) -> tuple[int | None, bool]:
    raw_mode = str(pt.get("rerank_mode", "")).strip().lower()
    is_full = bool(pt.get("is_full_beam", False)) or raw_mode in {"full_beam", "fullbeam", "full"}
    raw_r = pt.get("rerank_top_r", top_rerank)
    if isinstance(raw_r, str):
        norm = raw_r.strip().lower()
        if norm in {"full_beam", "fullbeam", "full", "ef", "ef_search"}:
            return None, True
        if norm in {"none", "null", ""}:
            return None, is_full
        try:
            return int(norm), is_full
        except ValueError:
            return None, is_full

    if raw_r is not None and not isinstance(raw_r, bool):
        r_int = safe_int(raw_r, default=0)
        if r_int > 0:
            return r_int, (is_full or top_full_beam)

    return None, (is_full or top_full_beam)


def _extract_scale(data: dict[str, Any]) -> int:
    dataset = _dict_field(data, "dataset")
    for candidate in (
        data.get("scale"),
        data.get("scale_vectors"),
        data.get("num_base"),
        dataset.get("scale_vectors"),
        dataset.get("num_base"),
    ):
        val = safe_int(candidate, default=0)
        if val > 0:
            return val
    return 0


def _extract_sweep_points(
    raw_sweep: list[Any],
    default_k: int,
    top_rerank: Any,
    top_full_beam: bool,
) -> list[Phase6SweepPoint]:
    points: list[Phase6SweepPoint] = []
    for item in raw_sweep:
        if not isinstance(item, dict):
            continue
        ef_search = safe_int(item.get("ef_search", item.get("requested_ef_search")), default=0)
        if ef_search <= 0:
            continue
        rerank_r, is_full_beam = _parse_rerank_field(
            item, top_rerank=top_rerank, top_full_beam=top_full_beam
        )
        eff_default = max(ef_search, rerank_r or 0, default_k)
        effective_ef = safe_int(item.get("effective_ef_search"), default=eff_default)
        med = _dict_field(item, "median_latency")

        recall = safe_float(
            item.get("recall_at_k", item.get("recall", item.get("recall_at_10"))),
            default=0.0,
        )
        qps = safe_float(item.get("qps", med.get("qps")), default=0.0)
        mean_ms = safe_float(item.get("mean_ms", med.get("mean_ms")), default=0.0)
        p50_ms = safe_float(item.get("p50_ms", med.get("p50_ms")), default=mean_ms)
        p95_ms = safe_float(item.get("p95_ms", med.get("p95_ms")), default=p50_ms)
        p99_ms = safe_float(item.get("p99_ms", med.get("p99_ms")), default=p95_ms)

        points.append(
            Phase6SweepPoint(
                ef_search=ef_search,
                effective_ef_search=effective_ef,
                rerank_top_r=rerank_r,
                is_full_beam=is_full_beam,
                recall_at_k=recall,
                qps=qps,
                mean_ms=mean_ms,
                p99_ms=p99_ms,
            )
        )
    return points


def _extract_steady_state_bytes_per_vector(
    data: dict[str, Any],
    mem: dict[str, Any],
    scale: int,
    dim: int,
    quantizer_kind: str,
    is_phase4: bool,
) -> float:
    graph_only = safe_float(
        data.get("graph_only_bytes_per_vector", mem.get("graph_only_bytes_per_vector")),
        default=0.0,
    )
    if graph_only <= 0.0 and scale > 0:
        graph_total = safe_float(
            data.get("graph_only_bytes", mem.get("graph_only_bytes", mem.get("graph_only_memory_bytes"))),
            default=0.0,
        )
        if graph_total > 0.0:
            graph_only = graph_total / float(scale)

    for key in (
        "steady_state_bytes_per_vector",
        "steady_state_payload_bytes_per_vector",
        "total_payload_bytes_per_vector",
    ):
        val = safe_float(data.get(key, mem.get(key)), default=0.0)
        if val > 0.0:
            return val

    codes_bv = safe_float(data.get("codes_bytes_per_vector", mem.get("codes_bytes_per_vector")), 0.0)
    cal_bv = safe_float(
        data.get("calibration_or_codebook_bytes_per_vector", mem.get("calibration_or_codebook_bytes_per_vector")),
        0.0,
    )
    raw_bv = safe_float(
        data.get("raw_vectors_retained_bytes_per_vector", mem.get("raw_vectors_retained_bytes_per_vector")),
        0.0,
    )
    if codes_bv + cal_bv + raw_bv > 0.0 and graph_only > 0.0:
        return codes_bv + cal_bv + graph_only + raw_bv

    if scale > 0:
        codes_b = safe_float(data.get("codes_bytes", mem.get("codes_bytes")), 0.0)
        cal_b = safe_float(
            data.get("calibration_or_codebook_bytes", mem.get("calibration_or_codebook_bytes")),
            0.0,
        )
        raw_b = safe_float(
            data.get("raw_vectors_retained_bytes", mem.get("raw_vectors_retained_bytes")),
            0.0,
        )
        if codes_b + cal_b + raw_b > 0.0:
            return graph_only + (codes_b + cal_b + raw_b) / float(scale)

    if (is_phase4 or quantizer_kind == "f32") and graph_only > 0.0:
        # Phase 4 JSON records graph_only_bytes_per_vector without the raw D * 4 B f32 matrix.
        return graph_only + float(dim) * 4.0

    return safe_float(mem.get("bytes_per_vector"), default=0.0)


def _resolve_arm_classification(
    data: dict[str, Any],
    path: Path,
    *,
    is_phase4: bool,
    m_val: int,
    efc_val: int,
) -> tuple[str, str, str, str]:
    raw_arm_id = str(data.get("arm_id", data.get("id", path.stem)))
    if is_phase4:
        return f"f32_m{m_val}_efc{efc_val}", "f32", "f32", "f32_graph"
    quantizer_kind = _normalize_quantizer_kind(
        data.get("quantizer_kind", data.get("quantizer")), raw_arm_id, data
    )
    storage_mode = _normalize_storage_mode(
        data.get("storage_mode", data.get("mode")), raw_arm_id, quantizer_kind
    )
    build_mode = _normalize_build_mode(
        data.get("build_mode", data.get("graph_build_mode")), raw_arm_id
    )
    return raw_arm_id, quantizer_kind, storage_mode, build_mode


def _extract_provenance_meta(
    data: dict[str, Any],
    env_dict: dict[str, Any],
    prov_dict: dict[str, Any],
    hw_dict: dict[str, Any],
    dataset_dict: dict[str, Any],
) -> tuple[str, str, str, str, bool, str]:
    cpu_model = str(
        env_dict.get("cpu_model")
        or prov_dict.get("cpu_model")
        or hw_dict.get("cpu_model")
        or "unknown CPU"
    )
    os_name = str(
        env_dict.get("os")
        or prov_dict.get("os")
        or hw_dict.get("os")
        or "unknown OS"
    )
    affinity = str(
        env_dict.get("processor_affinity_mask")
        or prov_dict.get("processor_affinity_mask")
        or data.get("processor_affinity_mask")
        or "not_set"
    )
    git_commit = str(data.get("git_commit", "unknown"))[:10]
    git_dirty = bool(data.get("git_dirty", False))
    dataset_desc = str(
        dataset_dict.get("source_description")
        or data.get("dataset_source")
        or data.get("dataset_source_kind")
        or dataset_dict.get("source_kind")
        or "SIFT"
    )
    return cpu_model, os_name, affinity, git_commit, git_dirty, dataset_desc


def _extract_rss_metrics(
    data: dict[str, Any],
    mem_dict: dict[str, Any],
    *,
    is_phase4: bool,
) -> tuple[float | None, float | None]:
    build_peak_rss = safe_opt_float(
        data.get("build_peak_rss_mb", mem_dict.get("build_peak_rss_mb", mem_dict.get("peak_rss_mb")))
    )
    query_peak_rss = safe_opt_float(
        data.get(
            "query_serving_peak_rss_mb",
            mem_dict.get(
                "query_serving_peak_rss_mb",
                mem_dict.get("query_peak_rss_mb", mem_dict.get("peak_rss_mb") if is_phase4 else None),
            ),
        )
    )
    return build_peak_rss, query_peak_rss


def parse_arm_report(path: str | Path) -> Phase6ArmReport | None:
    """Parse one Phase 6 arm JSON or Phase 4 baseline JSON report cleanly."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    raw_sweep = data.get("sweep", data.get("ef_search_sweep"))
    if not isinstance(raw_sweep, list) or not raw_sweep:
        return None

    scale = _extract_scale(data)
    if scale <= 0:
        return None

    is_phase4 = data.get("phase") == 4
    hnsw_dict = _dict_field(data, "hnsw_config")
    dataset_dict = _dict_field(data, "dataset")
    mem_dict = _dict_field(data, "memory_profile")
    env_dict = _dict_field(data, "environment")
    prov_dict = _dict_field(data, "provenance")
    hw_dict = _dict_field(data, "hardware")

    m_val = safe_int(data.get("m", hnsw_dict.get("m")), default=16)
    efc_val = safe_int(data.get("ef_construction", hnsw_dict.get("ef_construction")), default=100)
    dim = safe_int(data.get("dim", dataset_dict.get("dim")), default=128)
    k_val = safe_int(data.get("k", dataset_dict.get("k")), default=10)
    num_queries = safe_int(
        data.get("num_queries", data.get("num_query", dataset_dict.get("num_query"))),
        default=10_000,
    )

    arm_id, quantizer_kind, storage_mode, build_mode = _resolve_arm_classification(
        data, file_path, is_phase4=is_phase4, m_val=m_val, efc_val=efc_val
    )
    top_full_beam = bool(data.get("is_full_beam", False)) or (
        str(data.get("rerank_mode", "")).strip().lower() in {"full_beam", "fullbeam"}
    )
    sweep_points = _extract_sweep_points(
        raw_sweep, default_k=k_val, top_rerank=data.get("rerank_top_r"), top_full_beam=top_full_beam
    )
    if not sweep_points:
        return None

    steady_bv = _extract_steady_state_bytes_per_vector(
        data=data, mem=mem_dict, scale=scale, dim=dim, quantizer_kind=quantizer_kind, is_phase4=is_phase4
    )
    build_peak_rss, query_peak_rss = _extract_rss_metrics(data, mem_dict, is_phase4=is_phase4)
    reps = safe_int(data.get("reps", data.get("sweep_runs", data.get("num_reps"))), default=0)
    cpu_model, os_name, affinity, git_commit, git_dirty, dataset_desc = _extract_provenance_meta(
        data, env_dict, prov_dict, hw_dict, dataset_dict
    )

    return Phase6ArmReport(
        arm_id=arm_id,
        filename=file_path.name,
        quantizer_kind=quantizer_kind,
        storage_mode=storage_mode,
        build_mode=build_mode,
        m=m_val,
        ef_construction=efc_val,
        scale=scale,
        k=k_val,
        num_queries=num_queries,
        steady_state_bytes_per_vector=steady_bv,
        build_peak_rss_mb=build_peak_rss,
        query_serving_peak_rss_mb=query_peak_rss,
        reps=reps,
        cpu_model=cpu_model,
        os_name=os_name,
        affinity_mask=affinity,
        git_commit=git_commit,
        git_dirty=git_dirty,
        dataset_description=dataset_desc,
        source_phase=4 if is_phase4 else 6,
        sweep=tuple(sweep_points),
    )


def _arm_sort_key(rpt: Phase6ArmReport) -> tuple[int, int, int, int, int, str]:
    return (
        QUANTIZER_ORDER.get(rpt.quantizer_kind, 99),
        STORAGE_ORDER.get(rpt.storage_mode, 99),
        BUILD_ORDER.get(rpt.build_mode, 99),
        rpt.m,
        rpt.ef_construction,
        rpt.arm_id,
    )


def _discover_scale(reports: list[Phase6ArmReport]) -> int:
    phase6_scales = [r.scale for r in reports if r.source_phase != 4]
    scales = phase6_scales if phase6_scales else [r.scale for r in reports]
    return discover_scale(
        scales,
        empty_message="No valid Phase 6 or Phase 4 arm JSON reports found.",
        multi_message="Multiple scales found in results directory ({scales}); pass --scale explicitly.",
    )


def _filter_phase4_baselines(
    phase4_reports: list[Phase6ArmReport],
    phase6_reports: list[Phase6ArmReport],
) -> list[Phase6ArmReport]:
    if not phase4_reports:
        return []
    standard = [r for r in phase4_reports if "no_kp_shrink" not in r.filename.lower()]
    candidates = standard if standard else phase4_reports

    if not phase6_reports:
        preferred = [r for r in candidates if (r.m, r.ef_construction) in {(16, 100), (32, 200)}]
        return preferred if preferred else candidates

    phase6_cfgs = {(r.m, r.ef_construction) for r in phase6_reports}
    phase6_cfgs.add((16, 100))
    phase6_f32_cfgs = {(r.m, r.ef_construction) for r in phase6_reports if r.quantizer_kind == "f32"}

    return [
        r
        for r in candidates
        if (r.m, r.ef_construction) in phase6_cfgs and (r.m, r.ef_construction) not in phase6_f32_cfgs
    ]


def load_phase6_results(
    results_dir: str | Path,
    scale: int | None = None,
    phase4_dir: str | Path | None = None,
) -> LoadedPhase6Data:
    """Discover and normalize Phase 6 arm JSONs and optional Phase 4 ``f32`` baselines."""
    res_dir = Path(results_dir)
    p4_dir = Path(phase4_dir) if phase4_dir is not None else None
    if not res_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: '{res_dir}'")

    candidate_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for pattern in ("phase6_*.json", "arm_*.json", "scale_*.json"):
        for p in sorted(res_dir.glob(pattern)):
            resolved = p.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                candidate_paths.append(p)

    if p4_dir is not None and p4_dir.is_dir():
        for p in sorted(p4_dir.glob("scale_*.json")):
            resolved = p.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                candidate_paths.append(p)

    parsed = [rpt for path in candidate_paths if (rpt := parse_arm_report(path)) is not None]
    if not parsed:
        raise FileNotFoundError(
            f"No valid Phase 6 (`phase6_*.json` / `arm_*.json`) or baseline JSON reports in '{res_dir}'."
        )

    resolved_scale = _discover_scale(parsed) if scale is None else scale
    at_scale = [r for r in parsed if r.scale == resolved_scale]
    if not at_scale:
        available = sorted({r.scale for r in parsed})
        raise FileNotFoundError(
            f"No reports found for scale={resolved_scale} (available scales: {available})."
        )

    phase6_arms = [r for r in at_scale if r.source_phase != 4]
    phase4_arms = _filter_phase4_baselines(
        [r for r in at_scale if r.source_phase == 4],
        phase6_arms,
    )

    dedup: dict[tuple[str, str, str, str, int, int], Phase6ArmReport] = {}
    for arm in sorted(phase6_arms + phase4_arms, key=_arm_sort_key):
        dedup_key = (
            arm.arm_id,
            arm.quantizer_kind,
            arm.storage_mode,
            arm.build_mode,
            arm.m,
            arm.ef_construction,
        )
        dedup.setdefault(dedup_key, arm)

    ordered_arms = tuple(sorted(dedup.values(), key=_arm_sort_key))
    return LoadedPhase6Data(scale=resolved_scale, arms=ordered_arms)


def _format_quantizer_prefix(quantizer_kind: str) -> str:
    return {"f32": "f32", "sq8": "SQ8", "pq16": "PQ16", "pq32": "PQ32"}.get(
        quantizer_kind, quantizer_kind.upper()
    )


def _format_arm_label(arm: Phase6ArmReport, multi_m: bool, *, short: bool = False) -> str:
    q_name = _format_quantizer_prefix(arm.quantizer_kind)
    if arm.quantizer_kind == "f32":
        if short:
            return f"f32 M={arm.m}" if multi_m else "f32"
        src_note = " Phase 4" if arm.source_phase == 4 else ""
        return f"f32{src_note} (M={arm.m}, efC={arm.ef_construction})"

    base = f"{q_name}-Only" if arm.storage_mode == "only" else f"{q_name}+Rerank"
    if arm.build_mode == "quantized_direct":
        base += " [Q-Direct]" if not short else " [QD]"
    if multi_m:
        base += f" (M={arm.m})" if not short else f" M{arm.m}"
    return base


def _style_for_arm(arm: Phase6ArmReport, multi_m: bool) -> dict[str, Any]:
    key = (arm.quantizer_kind, arm.storage_mode, arm.build_mode)
    style = dict(ARM_STYLE_MAP.get(key, DEFAULT_ARM_STYLE))
    if multi_m and arm.m != 16:
        # Differentiate M=32 / M=8 variants visually when overlaid alongside M=16.
        style["linestyle"] = "-." if arm.build_mode == "f32_graph" else ":"
        style["marker"] = "D" if arm.m > 16 else "o"
        style["alpha"] = 0.88
    return style


def _build_provenance_caption(data: LoadedPhase6Data) -> str:
    rep = next((a for a in data.arms if a.source_phase != 4), data.arms[0])
    affinities = sorted({a.affinity_mask for a in data.arms if a.affinity_mask and a.affinity_mask != "not_set"})
    affinity_str = ",".join(affinities) if affinities else rep.affinity_mask
    reps_vals = sorted({a.reps for a in data.arms if a.reps > 0})
    if not reps_vals:
        reps_str = "n/a"
    elif len(reps_vals) == 1:
        reps_str = str(reps_vals[0])
    else:
        reps_str = f"{reps_vals[0]}..{reps_vals[-1]}"
    dirty_suffix = "-dirty" if rep.git_dirty else ""
    phase4_count = sum(1 for a in data.arms if a.source_phase == 4)
    p4_note = f" | includes {phase4_count} Phase 4 f32 baseline(s)" if phase4_count > 0 else ""
    return (
        f"N={data.scale:,} ({rep.dataset_description}, Q={rep.num_queries:,}, k={rep.k}) | "
        f"{rep.cpu_model} ({rep.os_name})\n"
        f"Single-thread affinity={affinity_str} | reps={reps_str} | "
        f"git={rep.git_commit}{dirty_suffix}{p4_note}"
    )


def _bucket_sweep_by_rerank(
    arm: Phase6ArmReport,
) -> list[tuple[tuple[int, int], str, list[Phase6SweepPoint]]]:
    buckets: dict[tuple[bool, int | None], list[Phase6SweepPoint]] = {}
    for pt in arm.sweep:
        key = (pt.is_full_beam, None if pt.is_full_beam else pt.rerank_top_r)
        buckets.setdefault(key, []).append(pt)

    ordered: list[tuple[tuple[int, int], str, list[Phase6SweepPoint]]] = []
    for (is_full, r_val), pts in buckets.items():
        pts_sorted = sorted(pts, key=lambda p: (p.ef_search, p.effective_ef_search))
        if is_full:
            ordered.append(((2, 0), "FullBeam", pts_sorted))
        elif r_val is None:
            ordered.append(((0, 0), "none", pts_sorted))
        else:
            ordered.append(((1, r_val), f"R={r_val}", pts_sorted))
    ordered.sort(key=lambda item: item[0])
    return ordered


def _select_series_for_sweep_charts(
    arm: Phase6ArmReport,
    multi_m: bool,
    all_rerank_curves: bool,
) -> list[tuple[str, dict[str, Any], list[Phase6SweepPoint]]]:
    base_label = _format_arm_label(arm, multi_m=multi_m, short=False)
    base_style = _style_for_arm(arm, multi_m=multi_m)
    buckets = _bucket_sweep_by_rerank(arm)

    if len(buckets) <= 1:
        if not buckets:
            return []
        _sort_k, tag, pts = buckets[0]
        lbl = f"{base_label} ({tag})" if (arm.storage_mode == "rerank" and tag != "none") else base_label
        return [(lbl, base_style, pts)]

    if all_rerank_curves or len(buckets) <= 2:
        selected = buckets
    else:
        # Pick the primary high-depth fixed-R curve plus FullBeam so Charts 1 & 2 stay legible.
        numeric = [b for b in buckets if b[0][0] == 1]
        full_b = [b for b in buckets if b[0][0] == 2]
        chosen: list[tuple[tuple[int, int], str, list[Phase6SweepPoint]]] = []
        if numeric:
            pref_100 = [b for b in numeric if b[0][1] == 100]
            chosen.append(pref_100[0] if pref_100 else numeric[-1])
        if full_b:
            chosen.append(full_b[0])
        selected = chosen if chosen else [buckets[-1]]

    out: list[tuple[str, dict[str, Any], list[Phase6SweepPoint]]] = []
    alt_linestyles = ["-", "-.", ":", "--"]
    alt_markers = [None, "o", "d", "s"]
    for idx, (_sort_k, tag, pts) in enumerate(selected):
        st = dict(base_style)
        if idx > 0:
            st["linestyle"] = alt_linestyles[idx % len(alt_linestyles)]
            alt_m = alt_markers[idx % len(alt_markers)]
            if alt_m is not None and alt_m != st.get("marker"):
                st["marker"] = alt_m
        lbl = base_label if tag == "none" else f"{base_label} ({tag})"
        out.append((lbl, st, pts))
    return out


def _plot_recall_sweep_chart(
    data: LoadedPhase6Data,
    output_dir: Path,
    *,
    y_fn: Callable[[Phase6SweepPoint], float],
    ylabel: str,
    title: str,
    stem: str,
    all_rerank_curves: bool,
    latency_budget_ms: float | None = None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 6.3), dpi=300)
    configure_ax(ax, log_x=False, log_y=True)

    multi_m = len({a.m for a in data.arms}) > 1
    all_recalls: list[float] = []
    for arm in data.arms:
        for lbl, st, pts in _select_series_for_sweep_charts(arm, multi_m, all_rerank_curves):
            xs = [pt.recall_at_k for pt in pts]
            ys = [y_fn(pt) for pt in pts]
            all_recalls.extend(xs)
            ax.plot(xs, ys, label=lbl, **st)

    ax.set_xlim(*compute_recall_xlim(all_recalls, default=(0.0, 1.01), hi_cap=1.01))
    if latency_budget_ms is not None:
        ax.axhline(latency_budget_ms, color="#666666", linestyle=":", linewidth=1.1, alpha=0.8)
        x_lo, _ = ax.get_xlim()
        ax.text(
            x_lo + 0.005,
            latency_budget_ms * 1.06,
            f"p99 budget = {latency_budget_ms:g} ms",
            fontsize=7.8,
            color="#555555",
            va="bottom",
        )

    ax.set_xlabel("Recall@10", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    finalize_figure_with_top_legend(
        fig,
        [ax],
        title=title,
        caption=_build_provenance_caption(data),
        max_ncol=3,
        legend_y=0.945,
        suptitle_y=0.985,
        legend_fontsize=8.2,
        caption_y=0.012,
        caption_fontsize=7.6,
        bottom_margin=0.085,
    )
    return save_figure(fig, output_dir, stem, metadata=_PNG_METADATA)


def plot_chart1_recall_vs_p99(
    data: LoadedPhase6Data,
    output_dir: Path,
    *,
    latency_budget_ms: float = 1.0,
    all_rerank_curves: bool = False,
) -> None:
    """Chart 1: Recall@10 vs p99 latency (ms, log y-axis) across f32, SQ8, and PQ."""
    _plot_recall_sweep_chart(
        data,
        output_dir,
        y_fn=lambda pt: max(pt.p99_ms, 1e-4),
        ylabel="p99 Query Latency [ms] (log scale)",
        title=f"Phase 6: Recall@10 vs p99 Latency (N={data.scale:,})",
        stem="chart1_recall_vs_p99",
        all_rerank_curves=all_rerank_curves,
        latency_budget_ms=latency_budget_ms,
    )


def plot_chart2_recall_vs_qps(
    data: LoadedPhase6Data,
    output_dir: Path,
    *,
    all_rerank_curves: bool = False,
) -> None:
    """Chart 2: Recall@10 vs single-thread throughput QPS (log y-axis)."""
    _plot_recall_sweep_chart(
        data,
        output_dir,
        y_fn=lambda pt: max(pt.qps, 1.0),
        ylabel="Single-Thread Throughput [QPS] (log scale)",
        title=f"Phase 6: Recall@10 vs Single-Thread QPS (N={data.scale:,})",
        stem="chart2_recall_vs_qps",
        all_rerank_curves=all_rerank_curves,
    )


def _select_memory_operating_point(
    arm: Phase6ArmReport,
    latency_budget_ms: float,
    match_mode: str,
) -> tuple[Phase6SweepPoint, str]:
    pts = list(arm.sweep)
    if match_mode in {"auto", "budget"}:
        eligible = [p for p in pts if p.p99_ms <= latency_budget_ms]
        if eligible:
            best = max(eligible, key=lambda p: (p.recall_at_k, -p.p99_ms, -p.ef_search))
            return best, f"p99<={latency_budget_ms:g}ms"
        if match_mode == "budget":
            fastest = min(pts, key=lambda p: (p.p99_ms, -p.recall_at_k))
            return fastest, f"min_p99={fastest.p99_ms:.2g}ms"

    at_100 = [p for p in pts if p.ef_search == 100]
    if at_100:
        best_100 = max(at_100, key=lambda p: (p.recall_at_k, -p.p99_ms))
        return best_100, "ef=100"

    nearest = min(pts, key=lambda p: (abs(p.ef_search - 100), -p.recall_at_k))
    return nearest, f"ef={nearest.ef_search}"


def _score_scatter_candidate(
    ax: Axes,
    renderer: Any,
    ax_box: Bbox,
    pad_px: float,
    marker_boxes: list[Bbox],
    placed_boxes: list[Bbox],
    idx: int,
    xv: float,
    yv: float,
    text: str,
    rank: int,
    dx: int,
    dy: int,
    ha: str,
) -> tuple[float, Bbox, bool]:
    ann = ax.annotate(
        text,
        (xv, yv),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va="center",
        fontsize=7.3,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        zorder=9,
    )
    bb = ann.get_window_extent(renderer=renderer)
    ann.remove()

    out_penalty = (
        max(0.0, ax_box.x0 + 2.0 - bb.x0)
        + max(0.0, bb.x1 - (ax_box.x1 - 2.0))
        + max(0.0, ax_box.y0 + 2.0 - bb.y0)
        + max(0.0, bb.y1 - (ax_box.y1 - 2.0))
    ) * 100.0
    pbb = Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)

    overlap_area = 0.0
    for ob in placed_boxes:
        if pbb.overlaps(ob):
            ix = max(0.0, min(pbb.x1, ob.x1) - max(pbb.x0, ob.x0))
            iy = max(0.0, min(pbb.y1, ob.y1) - max(pbb.y0, ob.y0))
            overlap_area += ix * iy + 500.0
    for j, mb in enumerate(marker_boxes):
        chk = bb if j == idx else pbb
        if chk.overlaps(mb):
            ix = max(0.0, min(chk.x1, mb.x1) - max(chk.x0, mb.x0))
            iy = max(0.0, min(chk.y1, mb.y1) - max(chk.y0, mb.y0))
            overlap_area += ix * iy + (200.0 if j != idx else 80.0)

    penalty = overlap_area * 10.0 + out_penalty + rank * 0.5 + math.hypot(dx, dy) * 0.2
    return penalty, bb, (overlap_area == 0.0 and out_penalty == 0.0)


def _place_scatter_annotation(
    ax: Axes,
    xv: float,
    yv: float,
    text: str,
    dx: int,
    dy: int,
    ha: str,
) -> None:
    use_leader = math.hypot(dx, dy) > 15.0
    ax.annotate(
        text,
        (xv, yv),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va="center",
        fontsize=7.3,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "#CCCCCC" if use_leader else "none",
            "linewidth": 0.4,
            "alpha": 0.88,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "#666666",
            "linewidth": 0.75,
            "shrinkA": 0,
            "shrinkB": 3.5,
        }
        if use_leader
        else None,
        zorder=9,
    )


def _annotate_scatter_points(ax: Axes, entries: list[tuple[float, float, str]]) -> None:
    if not entries:
        return
    fig = ax.get_figure()
    if not isinstance(fig, Figure) or not isinstance(fig.canvas, FigureCanvasAgg):
        return
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=renderer)
    dpi_scale = fig.dpi / 72.0
    pad_px = 2.5 * dpi_scale
    marker_r_px = 4.2 * dpi_scale
    marker_boxes = [
        Bbox.from_bounds(px - marker_r_px, py - marker_r_px, 2.0 * marker_r_px, 2.0 * marker_r_px)
        for px, py in (ax.transData.transform((xv, yv)) for xv, yv, _ in entries)
    ]

    order = sorted(
        range(len(entries)),
        key=lambda i: (round(entries[i][0], 2), -entries[i][1], entries[i][2]),
    )
    placed_boxes: list[Bbox] = []
    for idx in order:
        xv, yv, text = entries[idx]
        best_choice: tuple[int, int, str, Bbox] | None = None
        best_penalty = float("inf")
        for rank, (dx, dy, ha) in enumerate(_SCATTER_CANDIDATES):
            penalty, bb, is_clean = _score_scatter_candidate(
                ax, renderer, ax_box, pad_px, marker_boxes, placed_boxes, idx, xv, yv, text, rank, dx, dy, ha
            )
            if penalty < best_penalty:
                best_penalty = penalty
                best_choice = (dx, dy, ha, bb)
            if is_clean:
                break
        assert best_choice is not None
        dx, dy, ha, bb = best_choice
        placed_boxes.append(bb)
        _place_scatter_annotation(ax, xv, yv, text, dx, dy, ha)


def _plot_memory_scatter_panel(
    ax: Axes,
    entries: list[tuple[float, float, str, str, dict[str, Any]]],
    *,
    empty_msg: str,
    xlabel: str,
    title: str,
) -> list[tuple[float, float, str]]:
    if not entries:
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes, color="#666666")
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        return []

    pts: list[tuple[float, float]] = []
    annots: list[tuple[float, float, str]] = []
    for x_val, y_val, lbl, short_lbl, st in entries:
        ax.scatter(
            x_val,
            y_val,
            color=st["color"],
            marker=st["marker"],
            s=76,
            edgecolors="#222222",
            linewidths=0.7,
            label=lbl,
            zorder=int(st.get("zorder", 5)),
        )
        pts.append((x_val, y_val))
        annots.append((x_val, y_val, short_lbl))

    frontier = compute_pareto_frontier(
        pts,
        x_fn=lambda p: p[0],
        y_fn=lambda p: p[1],
        maximize_x=False,
        maximize_y=True,
    )
    if len(frontier) >= 2:
        ax.plot(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            color=OKABE_ITO_BLACK,
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            label="Memory-Recall Pareto",
            zorder=3,
        )

    ax.set_xlim(left=0.0, right=max(p[0] for p in pts) * 1.18)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10.8, fontweight="bold")
    return annots


def plot_chart3_memory_vs_recall(
    data: LoadedPhase6Data,
    output_dir: Path,
    *,
    latency_budget_ms: float = 1.0,
    memory_match_mode: str = "auto",
) -> None:
    """Chart 3: Total payload (B/vec) and query-serving Peak RSS (MiB) vs Recall@10."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.4, 5.8), sharey=True)
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]
    configure_ax(axes[0], log_x=False, log_y=False)
    configure_ax(axes[1], log_x=False, log_y=False)

    multi_m = len({a.m for a in data.arms}) > 1
    payload_entries: list[tuple[float, float, str, str, dict[str, Any]]] = []
    rss_entries: list[tuple[float, float, str, str, dict[str, Any]]] = []
    all_recalls: list[float] = []
    match_tags: set[str] = set()

    for arm in data.arms:
        pt, tag = _select_memory_operating_point(arm, latency_budget_ms, memory_match_mode)
        match_tags.add(tag)
        all_recalls.append(pt.recall_at_k)
        st = _style_for_arm(arm, multi_m=multi_m)
        lbl = _format_arm_label(arm, multi_m=multi_m, short=False)
        short_lbl = _format_arm_label(arm, multi_m=multi_m, short=True)

        if arm.steady_state_bytes_per_vector > 0.0:
            payload_entries.append((arm.steady_state_bytes_per_vector, pt.recall_at_k, lbl, short_lbl, st))

        if arm.query_serving_peak_rss_mb is not None and arm.query_serving_peak_rss_mb > 0.0:
            rss_entries.append((arm.query_serving_peak_rss_mb, pt.recall_at_k, lbl, short_lbl, st))
        elif arm.build_peak_rss_mb is not None and arm.build_peak_rss_mb > 0.0:
            rss_entries.append((arm.build_peak_rss_mb, pt.recall_at_k, lbl, f"{short_lbl} (build)", st))

    match_summary = ", ".join(sorted(match_tags)) if match_tags else f"p99<={latency_budget_ms:g}ms"
    payload_annots = _plot_memory_scatter_panel(
        axes[0],
        payload_entries,
        empty_msg="No steady-state payload B/vec recorded",
        xlabel="Total Steady-State Index Payload [B/vector]",
        title=f"Analytic Payload Footprint ({match_summary})",
    )
    rss_annots = _plot_memory_scatter_panel(
        axes[1],
        rss_entries,
        empty_msg="No Peak RSS telemetry recorded",
        xlabel="Query-Serving Peak RSS [MiB]",
        title=f"OS Peak Working Set ({match_summary})",
    )

    if all_recalls:
        y_lo = max(0.0, math.floor(min(all_recalls) * 20.0) / 20.0 - 0.04)
        axes[0].set_ylim(y_lo, 1.07)
    axes[0].set_ylabel("Achievable Recall@10", fontsize=10.5, fontweight="bold")

    finalize_figure_with_top_legend(
        fig,
        axes,
        title=f"Phase 6: Memory Footprint vs Achievable Recall@10 (N={data.scale:,})",
        caption=_build_provenance_caption(data),
        max_ncol=3,
        legend_y=0.945,
        suptitle_y=0.985,
        legend_fontsize=8.2,
        caption_y=0.012,
        caption_fontsize=7.6,
        bottom_margin=0.085,
    )
    fig.canvas.draw()
    if payload_annots:
        _annotate_scatter_points(axes[0], payload_annots)
    if rss_annots:
        _annotate_scatter_points(axes[1], rss_annots)
    save_figure(fig, output_dir, "chart3_memory_vs_recall", metadata=_PNG_METADATA)


def _select_ablation_ef(arms: tuple[Phase6ArmReport, ...], explicit_ef: int | None) -> int:
    if explicit_ef is not None and explicit_ef > 0:
        return explicit_ef

    rerank_ef_counts: dict[int, int] = {}
    all_ef_counts: dict[int, int] = {}
    for arm in arms:
        if arm.quantizer_kind == "f32":
            continue
        seen_all: set[int] = set()
        seen_rerank: set[int] = set()
        for pt in arm.sweep:
            seen_all.add(pt.ef_search)
            if pt.rerank_top_r is not None or pt.is_full_beam or arm.storage_mode == "rerank":
                seen_rerank.add(pt.ef_search)
        for ef in seen_all:
            all_ef_counts[ef] = all_ef_counts.get(ef, 0) + 1
        for ef in seen_rerank:
            rerank_ef_counts[ef] = rerank_ef_counts.get(ef, 0) + 1

    counts = rerank_ef_counts if rerank_ef_counts else all_ef_counts
    if 100 in counts or not counts:
        return 100
    return max(counts.keys(), key=lambda ef: (counts[ef], -abs(ef - 100), ef))


def _r_bucket_sort_key(bucket: tuple[str, int | None]) -> tuple[int, int]:
    kind, val = bucket
    if kind == "none":
        return (0, 0)
    if kind == "fixed":
        return (1, val or 0)
    return (2, 0)


def _format_r_bucket_label(bucket: tuple[str, int | None], target_ef: int) -> str:
    kind, val = bucket
    if kind == "none":
        return "None\n(Only)"
    if kind == "fixed":
        return f"R={val}"
    return f"FullBeam\n(R={target_ef})"


def _collect_ablation_families(
    arms: tuple[Phase6ArmReport, ...],
    target_ef: int,
) -> tuple[
    dict[tuple[str, str, int, int], dict[tuple[str, int | None], Phase6SweepPoint]],
    dict[tuple[str, str, int, int], Phase6ArmReport],
    bool,
]:
    families: dict[tuple[str, str, int, int], dict[tuple[str, int | None], Phase6SweepPoint]] = {}
    family_reps: dict[tuple[str, str, int, int], Phase6ArmReport] = {}

    for arm in arms:
        if arm.quantizer_kind == "f32":
            continue
        fam_key = (arm.quantizer_kind, arm.build_mode, arm.m, arm.ef_construction)
        family_reps.setdefault(fam_key, arm)
        fam_pts = families.setdefault(fam_key, {})

        ef_points = [p for p in arm.sweep if p.ef_search == target_ef]
        if not ef_points and arm.sweep:
            nearest_ef = min(arm.sweep, key=lambda p: abs(p.ef_search - target_ef)).ef_search
            ef_points = [p for p in arm.sweep if p.ef_search == nearest_ef]

        for pt in ef_points:
            if pt.is_full_beam:
                b_key: tuple[str, int | None] = ("fixed", pt.rerank_top_r or target_ef)
            elif pt.rerank_top_r is None:
                b_key = ("none", None)
            else:
                b_key = ("fixed", pt.rerank_top_r)
            fam_pts.setdefault(b_key, pt)

    rerank_families = {fk: pts for fk, pts in families.items() if any(bk[0] != "none" for bk in pts)}
    has_rerank = bool(rerank_families)
    return (rerank_families if has_rerank else families), family_reps, has_rerank


def _plot_f32_ablation_baselines(axes: list[Axes], arms: tuple[Phase6ArmReport, ...], target_ef: int) -> None:
    f32_arm = next((a for a in arms if a.quantizer_kind == "f32" and a.m == 16), None)
    if f32_arm is None:
        f32_arm = next((a for a in arms if a.quantizer_kind == "f32"), None)
    if f32_arm is None or not f32_arm.sweep:
        return
    f32_pt = min(f32_arm.sweep, key=lambda p: abs(p.ef_search - target_ef))
    lbl = f"f32 M={f32_arm.m} (ef={f32_pt.ef_search})"
    for ax, y_val in ((axes[0], f32_pt.recall_at_k), (axes[1], f32_pt.mean_ms)):
        ax.axhline(y_val, color=OKABE_ITO_BLACK, linestyle="--", linewidth=1.4, alpha=0.75, label=lbl, zorder=3)


def _plot_ablation_family_curves(
    axes: list[Axes],
    active_families: dict[tuple[str, str, int, int], dict[tuple[str, int | None], Phase6SweepPoint]],
    family_reps: dict[tuple[str, str, int, int], Phase6ArmReport],
    target_ef: int,
    multi_m: bool,
) -> tuple[list[float], list[float]]:
    all_buckets = sorted({bk for pts in active_families.values() for bk in pts}, key=_r_bucket_sort_key)
    if not active_families or not all_buckets:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No quantized rerank ablation points in loaded reports",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666666",
            )
        return [], []

    bucket_to_x = {bk: idx for idx, bk in enumerate(all_buckets)}
    sorted_fams = sorted(
        active_families.keys(),
        key=lambda fk: (QUANTIZER_ORDER.get(fk[0], 99), BUILD_ORDER.get(fk[1], 99), fk[2], fk[3]),
    )
    recalls_plotted: list[float] = []
    latencies_plotted: list[float] = []

    for fk in sorted_fams:
        rep = family_reps[fk]
        fam_pts = active_families[fk]
        ordered_bks = sorted(fam_pts.keys(), key=_r_bucket_sort_key)
        xs = [bucket_to_x[bk] for bk in ordered_bks]
        ys_recall = [fam_pts[bk].recall_at_k for bk in ordered_bks]
        ys_ms = [fam_pts[bk].mean_ms for bk in ordered_bks]
        recalls_plotted.extend(ys_recall)
        latencies_plotted.extend(ys_ms)

        style_key = (fk[0], "rerank" if any(bk[0] != "none" for bk in ordered_bks) else rep.storage_mode, fk[1])
        st = dict(ARM_STYLE_MAP.get(style_key, DEFAULT_ARM_STYLE))
        if multi_m and fk[2] != 16:
            st["linestyle"] = "-."
        lbl = _format_quantizer_prefix(fk[0])
        if fk[1] == "quantized_direct":
            lbl += " [Q-Direct]"
        if multi_m:
            lbl += f" (M={fk[2]})"

        axes[0].plot(xs, ys_recall, label=lbl, **st)
        axes[1].plot(xs, ys_ms, label=lbl, **st)

    x_ticks = list(range(len(all_buckets)))
    x_labels = [_format_r_bucket_label(bk, target_ef) for bk in all_buckets]
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_xlim(-0.35, max(len(all_buckets) - 0.65, 0.35))
    return recalls_plotted, latencies_plotted


def plot_chart4_rerank_ablation(
    data: LoadedPhase6Data,
    output_dir: Path,
    *,
    ablation_ef: int | None = None,
) -> None:
    """Chart 4: Recall@10 recovery and mean latency vs rerank depth R across SQ8/PQ16/PQ32."""
    fig, axes_arr = plt.subplots(1, 2, figsize=(13.4, 5.8), sharex=True)
    axes: list[Axes] = [axes_arr[0], axes_arr[1]]
    configure_ax(axes[0], log_x=False, log_y=False)
    configure_ax(axes[1], log_x=False, log_y=False)

    target_ef = _select_ablation_ef(data.arms, ablation_ef)
    multi_m = len({a.m for a in data.arms if a.quantizer_kind != "f32"}) > 1
    active_families, family_reps, has_rerank = _collect_ablation_families(data.arms, target_ef)
    _plot_f32_ablation_baselines(axes, data.arms, target_ef)
    recalls_plotted, latencies_plotted = _plot_ablation_family_curves(
        axes, active_families, family_reps, target_ef, multi_m
    )

    if recalls_plotted:
        y_lo = max(0.0, math.floor(min(recalls_plotted) * 20.0) / 20.0 - 0.04)
        axes[0].set_ylim(y_lo, 1.02)
    if latencies_plotted:
        pos_lat = [v for v in latencies_plotted if v > 0.0]
        if pos_lat and max(pos_lat) / min(pos_lat) > 15.0:
            axes[1].set_yscale("log")
        else:
            axes[1].set_ylim(bottom=0.0, top=max(latencies_plotted) * 1.15)

    if not has_rerank and active_families:
        axes[0].text(
            0.03,
            0.05,
            "Partial run: quantized-only baselines (no R>0 rerank arms)",
            transform=axes[0].transAxes,
            fontsize=8.0,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8F8F8", "edgecolor": "#CCCCCC"},
        )

    axes[0].set_xlabel("Rerank Depth R", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Rerank Depth R", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Recall@10", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Mean Query Latency [ms]", fontsize=10.5, fontweight="bold")
    axes[0].set_title(f"Recall@10 Recovery vs R (efSearch={target_ef})", fontsize=10.8, fontweight="bold")
    axes[1].set_title(f"Latency Overhead vs R (efSearch={target_ef})", fontsize=10.8, fontweight="bold")

    finalize_figure_with_top_legend(
        fig,
        axes,
        title=f"Phase 6: Top-R Rerank Depth Ablation (N={data.scale:,}, efSearch={target_ef})",
        caption=_build_provenance_caption(data),
        max_ncol=3,
        legend_y=0.945,
        suptitle_y=0.985,
        legend_fontsize=8.2,
        caption_y=0.012,
        caption_fontsize=7.6,
        bottom_margin=0.085,
    )
    save_figure(fig, output_dir, "chart4_rerank_ablation", metadata=_PNG_METADATA)


def generate_all_phase6_charts(
    results_dir: str | Path,
    output_dir: str | Path,
    scale: int | None = None,
    phase4_dir: str | Path | None = None,
    latency_budget_ms: float = 1.0,
    ablation_ef: int | None = None,
    memory_match_mode: str = "auto",
    all_rerank_curves: bool = False,
) -> list[Path]:
    """Generate all 4 Phase 6 charts (PNG + SVG) and return the 8 created paths."""
    out_dir = Path(output_dir)
    data = load_phase6_results(
        results_dir=results_dir,
        scale=scale,
        phase4_dir=phase4_dir,
    )
    plot_chart1_recall_vs_p99(
        data,
        out_dir,
        latency_budget_ms=latency_budget_ms,
        all_rerank_curves=all_rerank_curves,
    )
    plot_chart2_recall_vs_qps(
        data,
        out_dir,
        all_rerank_curves=all_rerank_curves,
    )
    plot_chart3_memory_vs_recall(
        data,
        out_dir,
        latency_budget_ms=latency_budget_ms,
        memory_match_mode=memory_match_mode,
    )
    plot_chart4_rerank_ablation(
        data,
        out_dir,
        ablation_ef=ablation_ef,
    )

    created: list[Path] = []
    for stem in CHART_STEMS:
        created.append(out_dir / f"{stem}.png")
        created.append(out_dir / f"{stem}.svg")
    return created


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Phase 6 quantized HNSW (SQ8 / PQ vs f32) benchmark charts (PNG + SVG)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase6"),
        help="Directory containing Phase 6 arm JSON files (`phase6_*.json` / `arm_*.json`).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write PNG/SVG charts (defaults to --results-dir).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Dataset scale N (e.g. 100000 or 1000000). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--phase4-dir",
        "--baseline-dir",
        dest="phase4_dir",
        type=Path,
        default=None,
        help="Optional directory containing Phase 4 `f32` baseline JSONs (`scale_*.json`).",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=1.0,
        help="p99 latency budget (ms) used for Chart 1 reference and Chart 3 operating points.",
    )
    parser.add_argument(
        "--ablation-ef",
        type=int,
        default=None,
        help="efSearch slice for Chart 4 rerank ablation (defaults to 100 if available).",
    )
    parser.add_argument(
        "--memory-match-mode",
        choices=["auto", "budget", "ef100"],
        default="auto",
        help="Chart 3 operating point match criterion (default: auto = p99<=budget, ef=100 fallback).",
    )
    parser.add_argument(
        "--all-rerank-curves",
        action="store_true",
        help="Overlay every rerank depth R curve on Charts 1 and 2 instead of representative depths.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.output_dir if args.output_dir is not None else args.results_dir
    created = generate_all_phase6_charts(
        results_dir=args.results_dir,
        output_dir=out_dir,
        scale=args.scale,
        phase4_dir=args.phase4_dir,
        latency_budget_ms=args.latency_budget_ms,
        ablation_ef=args.ablation_ef,
        memory_match_mode=args.memory_match_mode,
        all_rerank_curves=args.all_rerank_curves,
    )
    print(f"Generated {len(created)} chart files in {out_dir}")
    for path in created:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
