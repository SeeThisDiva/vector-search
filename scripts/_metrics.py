"""FAISS-free matched-recall interpolation metrics and config identifiers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

MATCHED_RECALL_TARGETS: tuple[float, ...] = (0.90, 0.95, 0.98, 0.99, 0.995)

CONFIG_ID_MAP: dict[tuple[int, int], str] = {
    (8, 100): "C1",
    (16, 100): "C2",
    (32, 200): "C3",
    (48, 400): "C4",
}


def _extract_curve_point(pt: Mapping[str, Any]) -> dict[str, Any] | None:
    recall_raw = pt.get("recall_at_k", pt.get("recall"))
    if recall_raw is None:
        return None
    recall = float(recall_raw)
    ef_search = int(pt.get("ef_search", 0))

    mode1 = pt.get("mode1_batch")
    if isinstance(mode1, Mapping):
        qps_raw = mode1.get("qps", pt.get("mode1_qps", pt.get("qps")))
        mean_raw = mode1.get("mean_ms", pt.get("mode1_mean_ms", pt.get("mean_ms")))
    else:
        qps_raw = pt.get("mode1_qps", pt.get("qps"))
        mean_raw = pt.get("mode1_mean_ms", pt.get("mean_ms"))

    mode2 = pt.get("mode2_per_query")
    if isinstance(mode2, Mapping):
        p99_raw = mode2.get("p99_ms", pt.get("mode2_p99_ms", pt.get("p99_ms")))
    else:
        p99_raw = pt.get("mode2_p99_ms", pt.get("p99_ms"))

    return {
        "ef_search": ef_search,
        "recall_at_k": recall,
        "qps": float(qps_raw) if qps_raw is not None else None,
        "p99_ms": float(p99_raw) if p99_raw is not None else None,
        "mean_ms": float(mean_raw) if mean_raw is not None else None,
    }


def _log_linear_interp(y_lo: float | None, y_hi: float | None, alpha: float) -> float | None:
    if y_lo is None or y_hi is None or y_lo <= 0.0 or y_hi <= 0.0:
        return None
    return math.exp((1.0 - alpha) * math.log(y_lo) + alpha * math.log(y_hi))


def _interpolate_curve_at_target(
    curve: Sequence[Mapping[str, Any]],
    target_recall: float,
) -> dict[str, Any] | None:
    """Interpolate log-QPS and log-latency linearly in recall without extrapolation."""
    pts = [p for raw in curve if (p := _extract_curve_point(raw)) is not None]
    if not pts:
        return None

    pts_sorted = sorted(pts, key=lambda p: (float(p["recall_at_k"]), int(p["ef_search"])))
    min_r = float(pts_sorted[0]["recall_at_k"])
    max_r = float(pts_sorted[-1]["recall_at_k"])
    tol = 1e-12

    # Never extrapolate beyond observed [min_recall, max_recall].
    if target_recall < min_r - tol or target_recall > max_r + tol:
        return None

    hi_idx = next(
        (idx for idx, pt in enumerate(pts_sorted) if float(pt["recall_at_k"]) >= target_recall - tol),
        len(pts_sorted) - 1,
    )
    hi_pt = pts_sorted[hi_idx]
    hi_r = float(hi_pt["recall_at_k"])

    if abs(hi_r - target_recall) <= tol or hi_idx == 0:
        lo_pt = hi_pt
        alpha = 0.0
    else:
        lo_pt = pts_sorted[hi_idx - 1]
        lo_r = float(lo_pt["recall_at_k"])
        span = hi_r - lo_r
        alpha = 0.0 if span <= 1e-15 else max(0.0, min(1.0, (target_recall - lo_r) / span))

    return {
        "status": "in_range",
        "interpolated_qps": _log_linear_interp(lo_pt["qps"], hi_pt["qps"], alpha),
        "interpolated_p99_ms": _log_linear_interp(lo_pt["p99_ms"], hi_pt["p99_ms"], alpha),
        "interpolated_mean_ms": _log_linear_interp(lo_pt["mean_ms"], hi_pt["mean_ms"], alpha),
        "alpha": round(alpha, 6),
        "lower_bracket": {
            "ef_search": lo_pt["ef_search"],
            "recall_at_k": lo_pt["recall_at_k"],
            "qps": lo_pt["qps"],
            "p99_ms": lo_pt["p99_ms"],
        },
        "upper_bracket": {
            "ef_search": hi_pt["ef_search"],
            "recall_at_k": hi_pt["recall_at_k"],
            "qps": hi_pt["qps"],
            "p99_ms": hi_pt["p99_ms"],
        },
    }


def compute_matched_recall_comparison(
    rust_curve: Sequence[Mapping[str, Any]],
    faiss_curve: Sequence[Mapping[str, Any]],
    targets: Sequence[float] = MATCHED_RECALL_TARGETS,
) -> list[dict[str, Any]]:
    """Compare Rust vs FAISS at standardized recall targets without extrapolation.

    For each target ``R*`` inside both curves' observed ``[min_recall, max_recall]``,
    performs linear interpolation of ``log(QPS)`` and ``log(p99_ms)`` against recall
    and reports speed ratios alongside discrete upper-bracket operating points.
    Targets outside either curve's range return ``status = "out_of_range"`` and ``None``.
    """
    comparisons: list[dict[str, Any]] = []
    for target in targets:
        target_f = float(target)
        rust_interp = _interpolate_curve_at_target(rust_curve, target_f)
        faiss_interp = _interpolate_curve_at_target(faiss_curve, target_f)

        rust_status = "in_range" if rust_interp is not None else "out_of_range"
        faiss_status = "in_range" if faiss_interp is not None else "out_of_range"
        matched = rust_interp is not None and faiss_interp is not None

        qps_ratio: float | None = None
        p99_ratio_faiss_over_rust: float | None = None
        p99_ratio_rust_over_faiss: float | None = None
        mean_ratio_faiss_over_rust: float | None = None

        if rust_interp is not None and faiss_interp is not None:
            r_qps = rust_interp["interpolated_qps"]
            f_qps = faiss_interp["interpolated_qps"]
            if r_qps is not None and f_qps is not None and f_qps > 0.0:
                qps_ratio = r_qps / f_qps

            r_p99 = rust_interp["interpolated_p99_ms"]
            f_p99 = faiss_interp["interpolated_p99_ms"]
            if r_p99 is not None and f_p99 is not None and r_p99 > 0.0 and f_p99 > 0.0:
                p99_ratio_faiss_over_rust = f_p99 / r_p99
                p99_ratio_rust_over_faiss = r_p99 / f_p99

            r_mean = rust_interp["interpolated_mean_ms"]
            f_mean = faiss_interp["interpolated_mean_ms"]
            if r_mean is not None and f_mean is not None and r_mean > 0.0:
                mean_ratio_faiss_over_rust = f_mean / r_mean

        comparisons.append(
            {
                "target_recall": target_f,
                "status": "matched" if matched else "out_of_range",
                "rust_status": rust_status,
                "faiss_status": faiss_status,
                "rust": rust_interp,
                "faiss": faiss_interp,
                "qps_ratio_rust_over_faiss": qps_ratio,
                "p99_ratio_faiss_over_rust": p99_ratio_faiss_over_rust,
                "p99_ratio_rust_over_faiss": p99_ratio_rust_over_faiss,
                "mean_ms_ratio_faiss_over_rust": mean_ratio_faiss_over_rust,
            }
        )
    return comparisons
