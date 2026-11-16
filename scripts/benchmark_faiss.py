#!/usr/bin/env python3
"""Phase 7 FAISS IndexHNSWFlat Benchmark Harness (Milestone T5.2).

Measures FAISS ``IndexHNSWFlat(dim, M, faiss.METRIC_L2)`` head-to-head against
the Rust HNSW engine under strict fairness controls:
- 64-bit Windows ``ctypes.wintypes`` ``SetProcessAffinityMask`` +
  ``GetProcessAffinityMask`` verification (asserting ``masked == requested``).
- Explicit ``faiss.omp_set_num_threads(threads)`` enforcement.
- Deterministic ``faiss.RandomGenerator(seed)`` initialization.
- Mode 1 Pure Batch (``idx.search(Q, k, D=..., I=...)``) with 1 warmup + N reps.
- Mode 2 Per-Query (``idx.search_c(1, q_ptr, k, d_ptr, i_ptr)``) with
  preallocated contiguous buffers, 1 warmup + N reps, percentile stats, and
  explicit per-query SWIG/FFI overhead ``ffi_delta_us``.
- Outside-timer diagnostic pass collecting ``faiss.cvar.hnsw_stats.ndis / Q``.
- Exact FAISS graph-only memory accounting (``levels`` + ``offsets`` +
  ``neighbors``) plus process ``PeakWorkingSet64`` via Windows
  ``GetProcessMemoryInfo`` (or POSIX ``getrusage``), with optional isolated
  child-process execution (``--profile-rss-isolated``).
- Reusable no-extrapolation matched-recall log-linear interpolation helper
  ``compute_matched_recall_comparison``.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from ctypes import wintypes
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.hnsw_phase1.dataset import generate_synthetic_sift
from src.hnsw_phase1.recall import compute_recall_at_k, verify_self_recall

MATCHED_RECALL_TARGETS: tuple[float, ...] = (0.90, 0.95, 0.98, 0.99, 0.995)
DEFAULT_EF_SEARCH_LIST: tuple[int, ...] = (
    10,
    20,
    30,
    40,
    50,
    70,
    100,
    150,
    200,
    300,
    400,
    600,
)
CONFIG_ID_MAP: dict[tuple[int, int], str] = {
    (8, 100): "C1",
    (16, 100): "C2",
    (32, 200): "C3",
    (48, 400): "C4",
}

if sys.platform == "win32":

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]


def _win_kernel32() -> Any:
    """Return a configured 64-bit-safe kernel32 WinDLL handle on Windows."""
    if sys.platform != "win32":
        raise RuntimeError("_win_kernel32 is only available on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    return kernel32


def parse_affinity_mask(mask_str: str) -> int:
    """Parse a hex (e.g. ``0x0004``) or decimal CPU affinity bitmask."""
    raw = mask_str.strip()
    if not raw:
        raise ValueError("Affinity mask string cannot be empty")
    mask = int(raw, 0)
    if mask <= 0:
        raise ValueError(f"Affinity mask must be positive, got {mask_str!r}")
    return mask


def format_affinity_mask(mask: int) -> str:
    """Format an integer CPU affinity mask as lowercase 4+-digit hex."""
    return f"0x{mask:04x}"


def get_process_affinity_mask() -> tuple[int, int] | None:
    """Return ``(process_mask, system_mask)`` where supported, else ``None``."""
    if sys.platform == "win32":
        kernel32 = _win_kernel32()
        handle = kernel32.GetCurrentProcess()
        proc_mask = ctypes.c_size_t(0)
        sys_mask = ctypes.c_size_t(0)
        ok = kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(proc_mask), ctypes.byref(sys_mask)
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        return int(proc_mask.value), int(sys_mask.value)

    if hasattr(os, "sched_getaffinity"):
        cpus = os.sched_getaffinity(0)
        proc_mask_val = 0
        for cpu in cpus:
            proc_mask_val |= 1 << cpu
        n_cpus = os.cpu_count() or 1
        sys_mask_val = (1 << n_cpus) - 1
        return proc_mask_val, sys_mask_val

    return None


def set_process_affinity_mask(requested_mask: int) -> tuple[int, int]:
    """Pin process affinity to ``requested_mask`` and verify ``masked == requested``.

    Returns ``(previous_mask, verified_mask)``.
    """
    if requested_mask <= 0:
        raise ValueError(f"Requested affinity mask must be positive, got {requested_mask}")

    if sys.platform == "win32":
        kernel32 = _win_kernel32()
        handle = kernel32.GetCurrentProcess()
        prev_c = ctypes.c_size_t(0)
        sys_c = ctypes.c_size_t(0)
        if not kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(prev_c), ctypes.byref(sys_c)
        ):
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        previous_mask = int(prev_c.value)
        system_mask = int(sys_c.value)
        if (requested_mask & system_mask) != requested_mask:
            raise ValueError(
                f"Requested affinity {format_affinity_mask(requested_mask)} exceeds "
                f"system affinity mask {format_affinity_mask(system_mask)}"
            )
        if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(requested_mask)):
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

        post_c = ctypes.c_size_t(0)
        post_sys_c = ctypes.c_size_t(0)
        if not kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(post_c), ctypes.byref(post_sys_c)
        ):
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        masked = int(post_c.value)
        assert masked == requested_mask, (
            f"Process affinity verification failed: expected {format_affinity_mask(requested_mask)}, "
            f"got {format_affinity_mask(masked)}"
        )
        return previous_mask, masked

    if hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
        prev_cpus = os.sched_getaffinity(0)
        previous_mask = 0
        for cpu in prev_cpus:
            previous_mask |= 1 << cpu
        target_cpus = {
            bit for bit in range(requested_mask.bit_length()) if (requested_mask >> bit) & 1
        }
        os.sched_setaffinity(0, target_cpus)
        actual_cpus = os.sched_getaffinity(0)
        masked = 0
        for cpu in actual_cpus:
            masked |= 1 << cpu
        assert masked == requested_mask, (
            f"Process affinity verification failed: expected {format_affinity_mask(requested_mask)}, "
            f"got {format_affinity_mask(masked)}"
        )
        return previous_mask, masked

    raise RuntimeError(f"CPU affinity pinning is not supported on platform {sys.platform!r}")


def enforce_omp_threads(threads: int) -> int:
    """Set FAISS OpenMP thread count and verify ``omp_get_max_threads() == threads``."""
    if threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}")
    faiss.omp_set_num_threads(threads)
    actual = int(faiss.omp_get_max_threads())
    assert actual == threads, (
        f"faiss.omp_set_num_threads({threads}) failed: omp_get_max_threads() == {actual}"
    )
    return actual


def get_peak_rss_mb() -> float | None:
    """Return process peak resident working set in MiB via OS APIs."""
    if sys.platform == "win32":
        kernel32 = _win_kernel32()
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)
        return float(usage.ru_maxrss) / 1024.0
    except (ImportError, OSError, AttributeError):
        return None


def detect_windows_power_scheme() -> tuple[str, str]:
    """Return ``(normalized_power_mode, raw_powercfg_output)``."""
    env_override = os.environ.get("VECTOR_SEARCH_POWER_MODE", "").strip()
    if sys.platform != "win32":
        return (env_override if env_override else "unknown"), ""

    raw_output = ""
    normalized = "unknown"
    try:
        proc = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            raw_output = proc.stdout.strip()
            lowered = raw_output.lower()
            if (
                "high performance" in lowered
                or "ultimate performance" in lowered
                or "best performance" in lowered
                or "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" in lowered
                or "e9a42b02-d5df-448d-aa00-03f14749eb61" in lowered
            ):
                normalized = "high_performance"
            elif (
                "balanced" in lowered
                or "381b4222-f694-41f0-9685-ff5bb260df2e" in lowered
            ):
                normalized = "balanced"
            elif (
                "power saver" in lowered
                or "a1841308-3541-4fab-bc81-f71556f20b4a" in lowered
            ):
                normalized = "power_saver"
    except (OSError, subprocess.SubprocessError):
        raw_output = ""

    if env_override:
        return env_override, raw_output
    return normalized, raw_output


def _detect_cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                text = str(val).strip()
                if text:
                    return text
        except OSError:
            pass
    elif Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    _, _, rhs = line.partition(":")
                    name = rhs.strip()
                    if name:
                        return name
        except OSError:
            pass

    fallback = platform.processor().strip()
    return fallback if fallback else f"Unknown {platform.machine()} CPU"


def _detect_total_ram_gb() -> float | None:
    if sys.platform == "win32":
        try:
            kernel32 = _win_kernel32()
            kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(float(status.ullTotalPhys) / (1024.0**3), 2)
        except (OSError, AttributeError):
            return None
    elif Path("/proc/meminfo").is_file():
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb_str = line.split(":", 1)[1].strip().rstrip("kB").strip()
                    return round(float(int(kb_str)) / (1024.0 * 1024.0), 2)
        except (OSError, ValueError):
            return None
    return None


def _detect_git_provenance() -> tuple[str, bool]:
    commit = "unknown"
    dirty = False
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if rev.returncode == 0 and rev.stdout.strip():
            commit = rev.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            dirty = status.returncode == 0 and bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    return commit, dirty


def collect_hardware_and_environment(
    affinity_mask: str,
    affinity_pinned: bool,
    threads: int,
) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    """Collect hardware, OS, power scheme, affinity, and git provenance metadata."""
    os_name = "windows" if sys.platform == "win32" else platform.system().lower()
    machine = platform.machine().lower()
    arch = "x86_64" if machine in ("amd64", "x86_64") else machine
    cpu_model = _detect_cpu_model()
    logical_cores = os.cpu_count() or 1
    total_ram_gb = _detect_total_ram_gb()
    power_mode, power_scheme_raw = detect_windows_power_scheme()
    git_commit, git_dirty = _detect_git_provenance()

    ram_summary = f", {total_ram_gb:.1f} GB RAM" if total_ram_gb is not None else ""
    summary = f"{cpu_model} ({logical_cores} logical cores{ram_summary}, {os_name}/{arch})"

    hardware = {
        "summary": summary,
        "cpu_model": cpu_model,
        "logical_cores": logical_cores,
        "total_ram_gb": total_ram_gb,
        "os": os_name,
        "arch": arch,
    }
    environment = {
        "os": os_name,
        "arch": arch,
        "cpu_model": cpu_model,
        "logical_cores": logical_cores,
        "total_ram_gb": total_ram_gb,
        "power_mode": power_mode,
        "power_scheme_raw": power_scheme_raw,
        "processor_affinity_mask": affinity_mask,
        "affinity_pinned": affinity_pinned,
        "threads": threads,
        "python_version": platform.python_version(),
    }
    return hardware, environment, git_commit, git_dirty


def collect_faiss_metadata(threads: int) -> dict[str, Any]:
    """Collect installed FAISS version, SIMDConfig flags, compile options, and OpenMP threads."""
    simd_cfg = faiss.SIMDConfig
    get_level_name = str(simd_cfg.get_level_name())
    simd_level = int(simd_cfg.get_level())
    has_dynamic_dispatch = bool(simd_cfg.has_dynamic_dispatch())

    if hasattr(faiss, "compiled_simd_levels") and callable(faiss.compiled_simd_levels):
        compiled_levels = int(faiss.compiled_simd_levels())
    elif hasattr(simd_cfg, "compiled_simd_levels") and callable(simd_cfg.compiled_simd_levels):
        compiled_levels = int(simd_cfg.compiled_simd_levels())
    else:
        compiled_levels = int(simd_cfg().supported_simd_levels)

    avx2_available = bool(
        hasattr(faiss, "SIMDLevel_AVX2")
        and simd_cfg.is_simd_level_available(faiss.SIMDLevel_AVX2)
    )
    compile_options = str(faiss.get_compile_options()) if hasattr(faiss, "get_compile_options") else ""
    supported_isa = (
        sorted(str(x) for x in faiss.supported_instruction_sets())
        if hasattr(faiss, "supported_instruction_sets")
        else []
    )
    return {
        "version": str(faiss.__version__),
        "index_class": "IndexHNSWFlat",
        "metric_type": "METRIC_L2",
        "simd_level_name": get_level_name,
        "simd_level": simd_level,
        "has_dynamic_dispatch": has_dynamic_dispatch,
        "compiled_simd_levels": compiled_levels,
        "avx2_available": avx2_available,
        "compile_options": compile_options,
        "supported_instruction_sets": supported_isa,
        "omp_threads_requested": threads,
        "omp_max_threads": int(faiss.omp_get_max_threads()),
    }


def read_fvecs_prefix(file_path: str | Path, max_rows: int | None = None) -> np.ndarray:
    """Read up to ``max_rows`` vectors from a Texmex ``.fvecs`` file without loading the full file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError(f"max_rows must be positive when set, got {max_rows}")

    file_size = path.stat().st_size
    if file_size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if file_size < 4:
        raise ValueError(f"Corrupt .fvecs file (size {file_size} < 4): {path}")

    with path.open("rb") as fh:
        dim = int(np.frombuffer(fh.read(4), dtype=np.int32)[0])
        if dim <= 0:
            raise ValueError(f"Invalid vector dimension {dim} in {path}")
        record_bytes = 4 + 4 * dim
        if file_size % record_bytes != 0:
            raise ValueError(
                f"File size {file_size} is not a multiple of record size {record_bytes} (dim={dim}) in {path}"
            )
        total_rows = file_size // record_bytes
        if max_rows is not None and max_rows > total_rows:
            raise ValueError(
                f"Requested {max_rows} rows, but {path} contains only {total_rows} vectors"
            )
        rows_to_read = total_rows if max_rows is None else min(total_rows, max_rows)
        fh.seek(0)
        raw = fh.read(rows_to_read * record_bytes)

    view = np.frombuffer(raw, dtype=np.float32).reshape(rows_to_read, dim + 1)
    return np.ascontiguousarray(view[:, 1:], dtype=np.float32)


def read_ivecs_prefix(file_path: str | Path, max_rows: int | None = None) -> np.ndarray:
    """Read up to ``max_rows`` rows from a Texmex ``.ivecs`` file without loading the full file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError(f"max_rows must be positive when set, got {max_rows}")

    file_size = path.stat().st_size
    if file_size == 0:
        return np.empty((0, 0), dtype=np.int32)
    if file_size < 4:
        raise ValueError(f"Corrupt .ivecs file (size {file_size} < 4): {path}")

    with path.open("rb") as fh:
        k = int(np.frombuffer(fh.read(4), dtype=np.int32)[0])
        if k <= 0:
            raise ValueError(f"Invalid column count {k} in {path}")
        record_bytes = 4 + 4 * k
        if file_size % record_bytes != 0:
            raise ValueError(
                f"File size {file_size} is not a multiple of record size {record_bytes} (k={k}) in {path}"
            )
        total_rows = file_size // record_bytes
        if max_rows is not None and max_rows > total_rows:
            raise ValueError(
                f"Requested {max_rows} rows, but {path} contains only {total_rows} rows"
            )
        rows_to_read = total_rows if max_rows is None else min(total_rows, max_rows)
        fh.seek(0)
        raw = fh.read(rows_to_read * record_bytes)

    view = np.frombuffer(raw, dtype=np.int32).reshape(rows_to_read, k + 1)
    return np.ascontiguousarray(view[:, 1:], dtype=np.int32)


def _compute_exact_l2_gt_faiss(
    base: np.ndarray,
    query: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute exact squared-L2 ground truth neighbor IDs via ``faiss.IndexFlatL2``."""
    flat = faiss.IndexFlatL2(int(base.shape[1]))
    flat.add(base)
    _, gt_ids = flat.search(query, k)
    return np.ascontiguousarray(gt_ids, dtype=np.int32)


def compute_faiss_memory_profile(
    index: Any,
    num_vectors: int,
    dim: int,
    peak_rss_mb: float | None,
    isolated_process: bool,
) -> dict[str, Any]:
    """Compute exact FAISS graph-only byte footprint (``levels`` + ``offsets`` + ``neighbors``)."""
    hnsw = index.hnsw
    levels_bytes = int(hnsw.levels.size()) * 4
    offsets_bytes = int(hnsw.offsets.size()) * 8
    neighbors_bytes = int(hnsw.neighbors.size()) * 4
    graph_only_bytes = levels_bytes + offsets_bytes + neighbors_bytes
    graph_only_b_per_vec = float(graph_only_bytes) / float(num_vectors) if num_vectors > 0 else 0.0
    raw_vec_b_per_vec = float(dim * 4)

    return {
        "peak_rss_mb": round(peak_rss_mb, 2) if peak_rss_mb is not None else None,
        "peak_rss_scope": (
            "isolated_child_process_high_water_mark"
            if isolated_process
            else "process_lifetime_high_water_mark"
        ),
        "graph_only_bytes_per_vector": round(graph_only_b_per_vec, 2),
        "graph_only_memory_bytes": graph_only_bytes,
        "graph_only_memory_mb": round(float(graph_only_bytes) / (1024.0 * 1024.0), 4),
        "levels_bytes": levels_bytes,
        "offsets_bytes": offsets_bytes,
        "neighbors_bytes": neighbors_bytes,
        "raw_vector_bytes_per_vector": round(raw_vec_b_per_vec, 2),
        "total_with_vectors_bytes_per_vector": round(graph_only_b_per_vec + raw_vec_b_per_vec, 2),
    }


def _summarize_mode1_runs(num_queries: int, run_total_s: list[float]) -> dict[str, Any]:
    run_mean_ms = [(t * 1000.0) / float(num_queries) for t in run_total_s]
    run_qps = [float(num_queries) / t if t > 0.0 else 0.0 for t in run_total_s]
    med_total_s = float(np.median(run_total_s))
    med_mean_ms = float(np.median(run_mean_ms))
    med_qps = float(num_queries) / med_total_s if med_total_s > 0.0 else 0.0
    return {
        "num_queries": num_queries,
        "total_s": round(med_total_s, 6),
        "qps": round(med_qps, 2),
        "mean_ms": round(med_mean_ms, 4),
        "mean_ms_min": round(min(run_mean_ms), 4),
        "mean_ms_max": round(max(run_mean_ms), 4),
        "mean_ms_spread": round(max(0.0, max(run_mean_ms) - min(run_mean_ms)), 4),
        "qps_min": round(min(run_qps), 2),
        "qps_max": round(max(run_qps), 2),
        "qps_spread": round(max(0.0, max(run_qps) - min(run_qps)), 2),
        "run_total_s": [round(x, 6) for x in run_total_s],
        "run_mean_ms": [round(x, 4) for x in run_mean_ms],
        "run_qps": [round(x, 2) for x in run_qps],
    }


def _summarize_mode2_runs(
    num_queries: int,
    run_stats: list[dict[str, float]],
) -> dict[str, Any]:
    totals = [r["total_s"] for r in run_stats]
    mins = [r["min_ms"] for r in run_stats]
    means = [r["mean_ms"] for r in run_stats]
    p50s = [r["p50_ms"] for r in run_stats]
    p95s = [r["p95_ms"] for r in run_stats]
    p99s = [r["p99_ms"] for r in run_stats]
    maxs = [r["max_ms"] for r in run_stats]

    med_total_s = float(np.median(totals))
    med_qps = float(num_queries) / med_total_s if med_total_s > 0.0 else 0.0
    return {
        "num_queries": num_queries,
        "total_s": round(med_total_s, 6),
        "qps": round(med_qps, 2),
        "min_ms": round(float(np.median(mins)), 4),
        "mean_ms": round(float(np.median(means)), 4),
        "p50_ms": round(float(np.median(p50s)), 4),
        "p95_ms": round(float(np.median(p95s)), 4),
        "p99_ms": round(float(np.median(p99s)), 4),
        "max_ms": round(float(np.median(maxs)), 4),
        "mean_ms_min": round(min(means), 4),
        "mean_ms_max": round(max(means), 4),
        "mean_ms_spread": round(max(0.0, max(means) - min(means)), 4),
        "p99_ms_min": round(min(p99s), 4),
        "p99_ms_max": round(max(p99s), 4),
        "p99_ms_spread": round(max(0.0, max(p99s) - min(p99s)), 4),
        "run_total_s": [round(x, 6) for x in totals],
        "run_mean_ms": [round(x, 4) for x in means],
        "run_p99_ms": [round(x, 4) for x in p99s],
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


def _resolve_config_id(m: int, ef_construction: int) -> str:
    return CONFIG_ID_MAP.get((m, ef_construction), f"M{m}_efC{ef_construction}")


def _load_dataset_arrays(
    *,
    dataset: str,
    synthetic: bool,
    scale: int | None,
    queries: int | None,
    dim: int,
    num_clusters: int,
    seed: int,
    k: int,
    base_fvecs: str | None,
    query_fvecs: str | None,
    gt_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str, str | None]:
    if synthetic:
        num_base = 10_000 if scale is None else scale
        num_query = 100 if queries is None else queries
        base, query = generate_synthetic_sift(
            num_base=num_base,
            num_query=num_query,
            dim=dim,
            num_clusters=num_clusters,
            seed=seed,
        )
        if gt_path is not None:
            gt = read_ivecs_prefix(gt_path, max_rows=num_query)
            gt_source = f"custom_ivecs ({gt_path})"
            resolved_gt_path: str | None = str(gt_path)
        else:
            gt = _compute_exact_l2_gt_faiss(base, query, k=k)
            gt_source = f"computed_faiss_index_flat_l2 (synthetic N={num_base}, Q={num_query}, k={k})"
            resolved_gt_path = None
        dataset_meta = {
            "preset": "synthetic",
            "source_kind": "synthetic",
            "source_description": f"Synthetic Gaussian SIFT (N={num_base}, Q={num_query}, dim={dim}, seed={seed})",
            "is_authentic_sift": False,
            "scale_vectors": num_base,
            "num_base": num_base,
            "num_query": num_query,
            "dim": dim,
            "k": k,
            "seed": seed,
            "metric": "l2",
        }
        return base, query, gt, dataset_meta, gt_source, resolved_gt_path

    if dataset == "siftsmall":
        default_base = _REPO_ROOT / "data" / "siftsmall" / "siftsmall_base.fvecs"
        default_query = _REPO_ROOT / "data" / "siftsmall" / "siftsmall_query.fvecs"
        default_gt = _REPO_ROOT / "data" / "siftsmall" / "siftsmall_groundtruth.ivecs"
    else:
        default_base = _REPO_ROOT / "data" / "sift1m" / "sift_base.fvecs"
        default_query = _REPO_ROOT / "data" / "sift1m" / "sift_query.fvecs"
        default_gt = _REPO_ROOT / "data" / "sift1m" / "sift_groundtruth.ivecs"

    base_path = Path(base_fvecs) if base_fvecs is not None else default_base
    query_path = Path(query_fvecs) if query_fvecs is not None else default_query

    base = read_fvecs_prefix(base_path, max_rows=scale)
    query = read_fvecs_prefix(query_path, max_rows=queries)
    num_base, actual_dim = int(base.shape[0]), int(base.shape[1])
    num_query = int(query.shape[0])
    if int(query.shape[1]) != actual_dim:
        raise ValueError(
            f"Dimension mismatch: base has dim={actual_dim}, query has dim={query.shape[1]}"
        )

    if gt_path is not None:
        gt = read_ivecs_prefix(gt_path, max_rows=num_query)
        gt_source = f"custom_ivecs ({gt_path})"
        resolved_gt_path = str(gt_path)
    elif dataset == "siftsmall" and num_base == 10_000 and default_gt.is_file():
        gt = read_ivecs_prefix(default_gt, max_rows=num_query)
        gt_source = "official_texmex_siftsmall_groundtruth_ivecs"
        resolved_gt_path = str(default_gt)
    elif dataset == "sift1m" and num_base == 1_000_000 and default_gt.is_file():
        gt = read_ivecs_prefix(default_gt, max_rows=num_query)
        gt_source = "official_texmex_sift1m_groundtruth_ivecs"
        resolved_gt_path = str(default_gt)
    elif (
        dataset == "sift1m"
        and num_base == 100_000
        and (_REPO_ROOT / "data" / "gt_100000_k100.ivecs").is_file()
    ):
        cached_100k_gt = _REPO_ROOT / "data" / "gt_100000_k100.ivecs"
        gt = read_ivecs_prefix(cached_100k_gt, max_rows=num_query)
        gt_source = f"cached_exact_knn_prefix_ivecs ({cached_100k_gt})"
        resolved_gt_path = str(cached_100k_gt)
    else:
        gt = _compute_exact_l2_gt_faiss(base, query, k=k)
        gt_source = f"computed_faiss_index_flat_l2 (N={num_base}, Q={num_query}, k={k})"
        resolved_gt_path = None

    is_authentic = dataset in ("sift1m", "siftsmall") and base_fvecs is None
    dataset_meta = {
        "preset": dataset,
        "source_kind": f"authentic_{dataset}" if is_authentic else "custom_fvecs",
        "source_description": f"Texmex {dataset} ({base_path})",
        "is_authentic_sift": is_authentic,
        "scale_vectors": num_base,
        "num_base": num_base,
        "num_query": num_query,
        "dim": actual_dim,
        "k": k,
        "seed": seed,
        "metric": "l2",
    }
    return base, query, gt, dataset_meta, gt_source, resolved_gt_path


def run_faiss_benchmark(
    *,
    base: np.ndarray | None = None,
    query: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    dataset: str = "sift1m",
    synthetic: bool = False,
    scale: int | None = None,
    queries: int | None = None,
    dim: int = 128,
    num_clusters: int = 10,
    k: int = 10,
    m: int = 16,
    ef_construction: int = 100,
    ef_search_list: Sequence[int] = DEFAULT_EF_SEARCH_LIST,
    threads: int = 1,
    affinity: str | None = None,
    restore_affinity: bool = False,
    build_reps: int | None = None,
    query_reps: int = 5,
    seed: int = 42,
    base_fvecs: str | None = None,
    query_fvecs: str | None = None,
    gt_path: str | None = None,
    compare_mode: str = "single",
    isolated_process: bool = False,
) -> dict[str, Any]:
    """Run FAISS ``IndexHNSWFlat`` build and Mode 1 / Mode 2 search benchmark."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if ef_construction <= 0:
        raise ValueError(f"ef_construction must be positive, got {ef_construction}")
    if query_reps <= 0:
        raise ValueError(f"query_reps must be positive, got {query_reps}")
    if not ef_search_list or any(ef <= 0 for ef in ef_search_list):
        raise ValueError(f"ef_search_list must contain positive integers, got {ef_search_list}")
    if compare_mode not in ("single", "multithread"):
        raise ValueError(f"compare_mode must be 'single' or 'multithread', got {compare_mode!r}")

    previous_affinity: int | None = None
    affinity_pinned = False
    verified_affinity_str = "not_set"
    if affinity is not None:
        requested_mask = parse_affinity_mask(affinity)
        previous_affinity, verified_mask = set_process_affinity_mask(requested_mask)
        affinity_pinned = True
        verified_affinity_str = format_affinity_mask(verified_mask)

    try:
        enforce_omp_threads(threads)

        if base is not None and query is not None:
            base_arr = np.ascontiguousarray(base, dtype=np.float32)
            query_arr = np.ascontiguousarray(query, dtype=np.float32)
            num_base, actual_dim = int(base_arr.shape[0]), int(base_arr.shape[1])
            num_query = int(query_arr.shape[0])
            if gt is not None:
                gt_arr = np.ascontiguousarray(gt, dtype=np.int32)
                gt_source = "provided_in_memory_gt"
            else:
                gt_arr = _compute_exact_l2_gt_faiss(base_arr, query_arr, k=k)
                gt_source = f"computed_faiss_index_flat_l2 (N={num_base}, Q={num_query}, k={k})"
            resolved_gt_path: str | None = None
            dataset_meta: dict[str, Any] = {
                "preset": "in_memory",
                "source_kind": "in_memory",
                "source_description": f"In-memory float32 arrays (N={num_base}, Q={num_query}, dim={actual_dim})",
                "is_authentic_sift": False,
                "scale_vectors": num_base,
                "num_base": num_base,
                "num_query": num_query,
                "dim": actual_dim,
                "k": k,
                "seed": seed,
                "metric": "l2",
            }
        else:
            base_arr, query_arr, gt_arr, dataset_meta, gt_source, resolved_gt_path = (
                _load_dataset_arrays(
                    dataset=dataset,
                    synthetic=synthetic,
                    scale=scale,
                    queries=queries,
                    dim=dim,
                    num_clusters=num_clusters,
                    seed=seed,
                    k=k,
                    base_fvecs=base_fvecs,
                    query_fvecs=query_fvecs,
                    gt_path=gt_path,
                )
            )
            num_base, actual_dim = int(base_arr.shape[0]), int(base_arr.shape[1])
            num_query = int(query_arr.shape[0])

        if num_base < k:
            raise ValueError(f"num_base ({num_base}) must be >= k ({k})")
        if gt_arr.shape[0] != num_query or gt_arr.shape[1] < k:
            raise ValueError(
                f"Ground truth shape {gt_arr.shape} incompatible with Q={num_query}, k={k}"
            )
        verify_self_recall(gt_arr, k=k)

        effective_build_reps = (1 if num_base >= 500_000 else 3) if build_reps is None else build_reps
        if effective_build_reps <= 0:
            raise ValueError(f"build_reps must be positive, got {effective_build_reps}")

        hardware, environment, git_commit, git_dirty = collect_hardware_and_environment(
            affinity_mask=verified_affinity_str,
            affinity_pinned=affinity_pinned,
            threads=threads,
        )
        faiss_meta = collect_faiss_metadata(threads=threads)

        faiss_dyn: Any = faiss
        # Build timing passes (releasing prior index between reps to preserve clean peak RSS)
        build_times_s: list[float] = []
        index: Any = None
        for _rep in range(effective_build_reps):
            if index is not None:
                del index
                gc.collect()
            enforce_omp_threads(threads)
            candidate_idx: Any = faiss.IndexHNSWFlat(actual_dim, m, faiss.METRIC_L2)
            assert candidate_idx.metric_type == faiss.METRIC_L2
            candidate_idx.hnsw.efConstruction = ef_construction
            rng = faiss_dyn.RandomGenerator(seed)
            candidate_idx.hnsw.rng = rng
            t0 = time.perf_counter_ns()
            candidate_idx.add(base_arr)
            t1 = time.perf_counter_ns()
            build_times_s.append(float(t1 - t0) / 1e9)
            index = candidate_idx

        assert index is not None
        med_build_s = float(np.median(build_times_s))
        build_qps = float(num_base) / med_build_s if med_build_s > 0.0 else 0.0

        # Preallocate output arrays and SWIG row pointers once outside timed loops
        D_batch = np.empty((num_query, k), dtype=np.float32)
        I_batch = np.empty((num_query, k), dtype=np.int64)

        run_mode2 = compare_mode == "single"
        if run_mode2:
            D_loop = np.empty((num_query, k), dtype=np.float32)
            I_loop = np.empty((num_query, k), dtype=np.int64)
            q_ptrs = [faiss.swig_ptr(query_arr[i]) for i in range(num_query)]
            d_ptrs = [faiss.swig_ptr(D_loop[i]) for i in range(num_query)]
            i_ptrs = [faiss.swig_ptr(I_loop[i]) for i in range(num_query)]
            latencies_ms = np.empty(num_query, dtype=np.float64)

        sweep_results: list[dict[str, Any]] = []
        prev_recall = -1.0
        monotonic_recall = True

        for ef_search in ef_search_list:
            index.hnsw.efSearch = int(ef_search)
            enforce_omp_threads(threads)

            # Mode 1 Pure Batch: 1 untimed warmup + query_reps timed passes
            index.search(query_arr, k, D=D_batch, I=I_batch)
            mode1_run_total_s: list[float] = []
            for _ in range(query_reps):
                t0 = time.perf_counter_ns()
                index.search(query_arr, k, D=D_batch, I=I_batch)
                t1 = time.perf_counter_ns()
                mode1_run_total_s.append(float(t1 - t0) / 1e9)

            mode1_summary = _summarize_mode1_runs(num_query, mode1_run_total_s)
            recall_val = compute_recall_at_k(I_batch, gt_arr, k=k)
            if recall_val + 1e-12 < prev_recall:
                monotonic_recall = False
            prev_recall = recall_val

            mode2_summary: dict[str, Any] | None = None
            recall_mode2: float | None = None
            ffi_delta_us: float | None = None

            if run_mode2:
                # Mode 2 Per-Query: 1 untimed warmup + query_reps timed passes
                for i in range(num_query):
                    index.search_c(1, q_ptrs[i], k, d_ptrs[i], i_ptrs[i])

                mode2_run_stats: list[dict[str, float]] = []
                for _ in range(query_reps):
                    t_pass_start = time.perf_counter_ns()
                    for i in range(num_query):
                        tq0 = time.perf_counter_ns()
                        index.search_c(1, q_ptrs[i], k, d_ptrs[i], i_ptrs[i])
                        tq1 = time.perf_counter_ns()
                        latencies_ms[i] = float(tq1 - tq0) / 1_000_000.0
                    t_pass_end = time.perf_counter_ns()
                    p50, p95, p99 = np.percentile(latencies_ms, [50.0, 95.0, 99.0], method="linear")
                    mode2_run_stats.append(
                        {
                            "total_s": float(t_pass_end - t_pass_start) / 1e9,
                            "min_ms": float(np.min(latencies_ms)),
                            "mean_ms": float(np.mean(latencies_ms)),
                            "p50_ms": float(p50),
                            "p95_ms": float(p95),
                            "p99_ms": float(p99),
                            "max_ms": float(np.max(latencies_ms)),
                        }
                    )

                mode2_summary = _summarize_mode2_runs(num_query, mode2_run_stats)
                recall_mode2 = compute_recall_at_k(I_loop, gt_arr, k=k)
                assert abs(recall_mode2 - recall_val) <= 1e-9, (
                    f"Mode 1 vs Mode 2 recall mismatch at efSearch={ef_search}: "
                    f"{recall_val} != {recall_mode2}"
                )
                ffi_delta_us = round(
                    (float(mode2_summary["total_s"]) - float(mode1_summary["total_s"]))
                    * 1_000_000.0
                    / float(num_query),
                    4,
                )

            # Diagnostic Pass (strictly outside timed loops)
            cvar_dyn: Any = faiss.cvar
            stats: Any = cvar_dyn.hnsw_stats
            stats.reset()
            index.search(query_arr, k, D=D_batch, I=I_batch)
            dist_evals_mean = float(stats.ndis) / float(num_query)
            nhops_mean = float(stats.nhops) / float(num_query)
            n1_mean = float(stats.n1) / float(num_query)

            sweep_results.append(
                {
                    "ef_search": int(ef_search),
                    "recall_at_k": round(recall_val, 6),
                    "recall_at_k_mode2": round(recall_mode2, 6) if recall_mode2 is not None else None,
                    "mode1_batch": mode1_summary,
                    "mode2_per_query": mode2_summary,
                    "ffi_delta_us": ffi_delta_us,
                    "dist_evals_mean": round(dist_evals_mean, 4),
                    "nhops_mean": round(nhops_mean, 4),
                    "n1_mean": round(n1_mean, 4),
                    "mode1_qps": mode1_summary["qps"],
                    "mode1_mean_ms": mode1_summary["mean_ms"],
                    "qps": mode1_summary["qps"],
                    "mean_ms": (
                        mode2_summary["mean_ms"]
                        if mode2_summary is not None
                        else mode1_summary["mean_ms"]
                    ),
                    "p50_ms": mode2_summary["p50_ms"] if mode2_summary is not None else None,
                    "p95_ms": mode2_summary["p95_ms"] if mode2_summary is not None else None,
                    "p99_ms": mode2_summary["p99_ms"] if mode2_summary is not None else None,
                }
            )

        peak_rss_mb = get_peak_rss_mb()
        memory_profile = compute_faiss_memory_profile(
            index=index,
            num_vectors=num_base,
            dim=actual_dim,
            peak_rss_mb=peak_rss_mb,
            isolated_process=isolated_process,
        )

        config_id = _resolve_config_id(m, ef_construction)
        generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "schema_version": 1,
            "phase": 7,
            "phase_name": "Phase 7 FAISS IndexHNSWFlat Benchmark",
            "engine": "faiss",
            "compare_mode": compare_mode,
            "config_id": config_id,
            "generated_at": generated_at,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "isolated_process": isolated_process,
            "environment": environment,
            "hardware": hardware,
            "faiss": faiss_meta,
            "dataset": dataset_meta,
            "ground_truth_source": gt_source,
            "ground_truth_path": resolved_gt_path,
            "hnsw_config": {
                "m": m,
                "m_max": m,
                "m_max0": 2 * m,
                "ef_construction": ef_construction,
                "keep_pruned": False,
                "keep_pruned_shrink": False,
                "prune_headroom": float(index.hnsw.prune_headroom),
                "search_bounded_queue": bool(index.hnsw.search_bounded_queue),
                "check_relative_distance": bool(index.hnsw.check_relative_distance),
                "seed": seed,
            },
            "threads": threads,
            "warmup_performed": True,
            "build_reps": effective_build_reps,
            "query_reps": query_reps,
            "scale_vectors": num_base,
            "index_build": {
                "num_vectors": num_base,
                "build_time_s": round(med_build_s, 6),
                "build_time_min_s": round(min(build_times_s), 6),
                "build_time_max_s": round(max(build_times_s), 6),
                "build_time_spread_s": round(max(0.0, max(build_times_s) - min(build_times_s)), 6),
                "build_times_s": [round(x, 6) for x in build_times_s],
                "build_qps": round(build_qps, 2),
                "max_layer": int(index.hnsw.max_level),
                "entry_point": int(index.hnsw.entry_point),
            },
            "memory_profile": memory_profile,
            "graph_only_bytes_per_vector": memory_profile["graph_only_bytes_per_vector"],
            "monotonic_recall": monotonic_recall,
            "ef_search_sweep": sweep_results,
        }
    finally:
        if restore_affinity and previous_affinity is not None:
            set_process_affinity_mask(previous_affinity)


def _parse_ef_search_list(raw: str) -> list[int]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("ef-search-list cannot be empty")
    values = [int(x) for x in items]
    if any(v <= 0 for v in values):
        raise ValueError(f"All efSearch values must be positive, got {values}")
    return values


def _print_console_summary(report: Mapping[str, Any]) -> None:
    ds = report["dataset"]
    hw = report["hardware"]
    env = report["environment"]
    fx = report["faiss"]
    ib = report["index_build"]
    mem = report["memory_profile"]

    print(
        f"[FAISS Phase 7] {report['config_id']} (M={report['hnsw_config']['m']}, "
        f"efC={report['hnsw_config']['ef_construction']}, mode={report['compare_mode']}, "
        f"T={report['threads']}, affinity={env['processor_affinity_mask']})"
    )
    print(f"  Hardware: {hw['summary']} | Power: {env['power_mode']}")
    print(
        f"  FAISS {fx['version']} | SIMD={fx['simd_level_name']} "
        f"(dyn={fx['has_dynamic_dispatch']}, levels={fx['compiled_simd_levels']}) | "
        f"OMP={fx['omp_max_threads']}"
    )
    print(
        f"  Dataset: {ds['source_kind']} N={ds['num_base']:,} Q={ds['num_query']:,} "
        f"dim={ds['dim']} k={ds['k']}"
    )
    print(
        f"  Build: median={ib['build_time_s']:.3f}s ({ib['build_qps']:.1f} vec/s, "
        f"reps={report['build_reps']}) | Graph={mem['graph_only_bytes_per_vector']:.2f} B/vec | "
        f"PeakRSS={mem['peak_rss_mb']} MiB ({mem['peak_rss_scope']})"
    )
    print(
        "  efSearch | Recall@k | Mode1 QPS | Mode1 ms | Mode2 p99 | FFI delta (us) | ndis/q"
    )
    for pt in report["ef_search_sweep"]:
        p99_str = f"{pt['p99_ms']:9.4f}" if pt["p99_ms"] is not None else "      n/a"
        ffi_str = f"{pt['ffi_delta_us']:14.3f}" if pt["ffi_delta_us"] is not None else "           n/a"
        print(
            f"  {pt['ef_search']:8d} | {pt['recall_at_k']:8.6f} | {pt['mode1_qps']:9.2f} | "
            f"{pt['mode1_mean_ms']:8.4f} | {p99_str} | {ffi_str} | {pt['dist_evals_mean']:8.1f}"
        )


def _spawn_isolated_rss_child(raw_argv: list[str]) -> int:
    child_args = [arg for arg in raw_argv if arg != "--profile-rss-isolated"]
    if "--internal-isolated-child" not in child_args:
        child_args.append("--internal-isolated-child")
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Phase 7 FAISS benchmarks."""
    parser = argparse.ArgumentParser(
        description="Phase 7 FAISS IndexHNSWFlat Benchmark Runner (Milestone T5.2)"
    )
    parser.add_argument(
        "--dataset",
        choices=("sift1m", "siftsmall"),
        default="sift1m",
        help="Named Texmex dataset preset when not using --synthetic (default: sift1m).",
    )
    parser.add_argument(
        "--base-fvecs",
        default=None,
        help="Optional override path to base .fvecs file.",
    )
    parser.add_argument(
        "--query-fvecs",
        default=None,
        help="Optional override path to query .fvecs file.",
    )
    parser.add_argument(
        "--gt-path",
        default=None,
        help="Optional path to .ivecs ground truth file.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Number of base vectors to load (prefix read for .fvecs; default 10000 under --synthetic).",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Optional maximum number of query vectors to load (default 100 under --synthetic).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Vector dimensionality for --synthetic mode (default: 128).",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=10,
        help="Cluster count for --synthetic mode (default: 10).",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k neighbors (default: 10).")
    parser.add_argument("--m", type=int, default=16, help="HNSW degree parameter M (default: 16).")
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="HNSW efConstruction parameter (default: 100).",
    )
    parser.add_argument(
        "--ef-search-list",
        default=",".join(str(x) for x in DEFAULT_EF_SEARCH_LIST),
        help="Comma-separated efSearch values (default: 10,20,30,40,50,70,100,150,200,300,400,600).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="OpenMP thread count enforced via faiss.omp_set_num_threads (default: 1).",
    )
    parser.add_argument(
        "--affinity",
        default=None,
        help="Optional CPU affinity mask in hex or decimal (e.g. 0x0004, 0x0555, 0xFFFF).",
    )
    parser.add_argument(
        "--build-reps",
        type=int,
        default=None,
        help="Number of index builds (default: 3 for N < 500k, 1 for N >= 500k).",
    )
    parser.add_argument(
        "--query-reps",
        type=int,
        default=5,
        help="Number of timed query sweep passes after 1 untimed warmup (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic offline synthetic Gaussian SIFT vectors.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path to write structured JSON report.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("single", "multithread"),
        default="single",
        help="Benchmark mode: single (Mode 1 + Mode 2 + FFI delta) or multithread (Mode 1 batch).",
    )
    parser.add_argument(
        "--profile-rss-isolated",
        action="store_true",
        help="Spawn a fresh isolated child process so PeakWorkingSet64 is uncontaminated.",
    )
    parser.add_argument(
        "--internal-isolated-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Phase 7 FAISS benchmark harness."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    if args.profile_rss_isolated and not args.internal_isolated_child:
        return _spawn_isolated_rss_child(raw_argv)

    ef_search_list = _parse_ef_search_list(args.ef_search_list)
    report = run_faiss_benchmark(
        dataset=args.dataset,
        synthetic=bool(args.synthetic),
        scale=args.scale,
        queries=args.queries,
        dim=args.dim,
        num_clusters=args.num_clusters,
        k=args.k,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search_list=ef_search_list,
        threads=args.threads,
        affinity=args.affinity,
        restore_affinity=False,
        build_reps=args.build_reps,
        query_reps=args.query_reps,
        seed=args.seed,
        base_fvecs=args.base_fvecs,
        query_fvecs=args.query_fvecs,
        gt_path=args.gt_path,
        compare_mode=args.compare_mode,
        isolated_process=bool(args.internal_isolated_child),
    )

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[FAISS Phase 7] Wrote JSON report: {out_path}")

    _print_console_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
