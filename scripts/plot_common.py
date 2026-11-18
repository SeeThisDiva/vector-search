"""Shared Matplotlib styling, layout, Pareto, and parsing utilities for Phase 4-7 plotters."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_T = TypeVar("_T")

# Canonical Okabe-Ito colorblind-safe palette constants
OKABE_ITO_BLACK = "#111111"
OKABE_ITO_BLUE = "#0072B2"
OKABE_ITO_ORANGE = "#D55E00"
OKABE_ITO_GREEN = "#009E73"
OKABE_ITO_PINK = "#CC79A7"
OKABE_ITO_AMBER = "#E69F00"
OKABE_ITO_SKY = "#56B4E9"
OKABE_ITO_YELLOW = "#F0E442"

OKABE_ITO_CYCLE: tuple[str, ...] = (
    OKABE_ITO_BLUE,
    OKABE_ITO_ORANGE,
    OKABE_ITO_GREEN,
    OKABE_ITO_PINK,
    OKABE_ITO_AMBER,
    OKABE_ITO_SKY,
    OKABE_ITO_YELLOW,
    OKABE_ITO_BLACK,
)


def configure_matplotlib(
    *,
    svg_hashsalt: str | None = None,
    font_family: str | None = None,
) -> None:
    """Configure headless Agg backend and deterministic Matplotlib RC parameters."""
    matplotlib.use("Agg")
    if svg_hashsalt is not None:
        plt.rcParams["svg.hashsalt"] = svg_hashsalt
    if font_family is not None:
        plt.rcParams["font.family"] = font_family


def configure_ax(
    ax: Axes,
    *,
    log_x: bool = False,
    log_y: bool = False,
    minor_alpha: float = 0.16,
) -> None:
    """Apply consistent spine hiding, grid lines, and optional log scaling to an Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.28, color="#888888")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=minor_alpha, color="#AAAAAA")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def save_figure(
    fig: Figure,
    output_dir: str | Path,
    stem: str,
    *,
    dpi: int = 300,
    bbox_inches: str | None = None,
    metadata: dict[str, Any] | None = None,
    svg_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """Save figure as 300-DPI PNG and deterministic SVG, close it, and return both paths."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    png_path = out_path / f"{stem}.png"
    svg_path = out_path / f"{stem}.svg"

    png_meta = {"Software": None} if metadata is None else dict(metadata)
    svg_meta = {"Date": None} if svg_metadata is None else dict(svg_metadata)

    if bbox_inches is not None:
        fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, metadata=png_meta)
        fig.savefig(svg_path, format="svg", bbox_inches=bbox_inches, metadata=svg_meta)
    else:
        fig.savefig(png_path, dpi=dpi, metadata=png_meta)
        fig.savefig(svg_path, format="svg", metadata=svg_meta)

    plt.close(fig)
    return [png_path, svg_path]


def collect_shared_legend(axes: Sequence[Axes]) -> tuple[list[Any], list[str]]:
    """Collect deduplicated ``(handles, labels)`` in first-seen order across ``axes``."""
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


def render_empty_panel(ax: Axes, title: str, message: str) -> None:
    """Render an informative placeholder panel when no plot data is available."""
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


def finalize_figure_with_top_legend(
    fig: Figure,
    axes: Sequence[Axes],
    *,
    title: str,
    caption: str | None = None,
    max_ncol: int = 3,
    legend_y: float = 0.94,
    suptitle_y: float = 0.985,
    legend_fontsize: float = 8.2,
    caption_y: float = 0.01,
    caption_fontsize: float = 7.5,
    bottom_margin: float | None = None,
    top_margin: float | None = None,
) -> None:
    """Attach shared top legend, suptitle, bottom provenance caption, and tight layout."""
    handles, labels = collect_shared_legend(axes)
    if handles:
        ncol = min(max_ncol, max(1, len(handles)))
        n_rows = math.ceil(len(handles) / ncol)
        computed_top = max(0.66, (legend_y - 0.02) - 0.036 * n_rows)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
        )
    else:
        computed_top = 0.92

    fig.suptitle(title, fontsize=13, fontweight="bold", y=suptitle_y)
    if caption:
        fig.text(
            0.5,
            caption_y,
            caption,
            ha="center",
            va="bottom",
            fontsize=caption_fontsize,
            color="#444444",
        )

    resolved_bottom = (0.065 if caption else 0.025) if bottom_margin is None else bottom_margin
    resolved_top = computed_top if top_margin is None else top_margin
    fig.tight_layout(rect=(0.0, resolved_bottom, 1.0, resolved_top))


def compute_recall_xlim(
    recalls: Sequence[float],
    *,
    default: tuple[float, float] = (0.65, 1.005),
    min_span: float = 0.08,
    max_lo: float = 0.85,
    hi_cap: float = 1.005,
) -> tuple[float, float]:
    """Compute a clean Recall@k x-axis range covering all observed points without clipping."""
    if not recalls:
        return default
    min_r = min(recalls)
    max_r = max(recalls)
    lo = max(0.0, math.floor((min_r - 0.02) * 20.0) / 20.0)
    lo = min(lo, max_lo)
    hi = min(hi_cap, max(max_r + 0.015, lo + min_span))
    if hi - lo < min_span:
        lo = max(0.0, hi - min_span)
    return (lo, hi)


def compute_pareto_frontier(
    items: Sequence[_T],
    *,
    x_fn: Callable[[_T], float],
    y_fn: Callable[[_T], float],
    maximize_x: bool = True,
    maximize_y: bool = False,
) -> list[_T]:
    """Extract non-dominated 2D Pareto frontier points sorted by ``x`` ascending."""
    if not items:
        return []

    def _sort_key(item: _T) -> tuple[float, float]:
        xv = x_fn(item)
        yv = y_fn(item)
        return (-xv if maximize_x else xv, -yv if maximize_y else yv)

    ordered = sorted(items, key=_sort_key)
    frontier: list[_T] = []
    best_y = float("-inf") if maximize_y else float("inf")
    eps = 1e-12
    for item in ordered:
        yv = y_fn(item)
        is_better = (yv > best_y + eps) if maximize_y else (yv < best_y - eps)
        if is_better:
            frontier.append(item)
            best_y = yv

    if maximize_x:
        frontier.reverse()
    return frontier


def discover_scale(
    scales: Iterable[int],
    *,
    empty_message: str = "No valid benchmark reports found.",
    multi_message: str = "Multiple scales found ({scales}); pass --scale explicitly.",
) -> int:
    """Validate that discovered dataset scales resolve unambiguously to a single scale."""
    unique = sorted(set(scales))
    if not unique:
        raise FileNotFoundError(empty_message)
    if len(unique) > 1:
        raise ValueError(multi_message.format(scales=unique))
    return unique[0]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Parse finite float from JSON scalar, falling back to ``default``."""
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def safe_opt_float(value: Any) -> float | None:
    """Parse finite optional float from JSON scalar, returning ``None`` when absent/invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_int(value: Any, default: int = 0) -> int:
    """Parse int from JSON scalar, falling back to ``default``."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
