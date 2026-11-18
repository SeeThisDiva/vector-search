"""OS/hardware provenance, process CPU affinity pinning, and peak RSS helpers."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from ctypes import wintypes
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

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


def _win_get_affinity(kernel32: Any, handle: Any) -> tuple[int, int]:
    """Query ``(process_mask, system_mask)`` via Win32 ``GetProcessAffinityMask``."""
    if sys.platform != "win32":
        raise RuntimeError("_win_get_affinity is only available on Windows")
    proc_c = ctypes.c_size_t(0)
    sys_c = ctypes.c_size_t(0)
    if not kernel32.GetProcessAffinityMask(handle, ctypes.byref(proc_c), ctypes.byref(sys_c)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(proc_c.value), int(sys_c.value)


def _cpus_to_mask(cpus: Iterable[int]) -> int:
    mask = 0
    for cpu in cpus:
        mask |= 1 << cpu
    return mask


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
        return _win_get_affinity(kernel32, kernel32.GetCurrentProcess())

    if hasattr(os, "sched_getaffinity"):
        n_cpus = os.cpu_count() or 1
        return _cpus_to_mask(os.sched_getaffinity(0)), (1 << n_cpus) - 1

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
        previous_mask, system_mask = _win_get_affinity(kernel32, handle)
        if (requested_mask & system_mask) != requested_mask:
            raise ValueError(
                f"Requested affinity {format_affinity_mask(requested_mask)} exceeds "
                f"system affinity mask {format_affinity_mask(system_mask)}"
            )
        if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(requested_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        masked, _ = _win_get_affinity(kernel32, handle)
    elif hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
        n_cpus = os.cpu_count() or 1
        system_mask = (1 << n_cpus) - 1
        if (requested_mask & system_mask) != requested_mask:
            raise ValueError(
                f"Requested affinity {format_affinity_mask(requested_mask)} exceeds "
                f"system affinity mask {format_affinity_mask(system_mask)}"
            )
        previous_mask = _cpus_to_mask(os.sched_getaffinity(0))
        target_cpus = {
            bit for bit in range(requested_mask.bit_length()) if (requested_mask >> bit) & 1
        }
        try:
            os.sched_setaffinity(0, target_cpus)
        except OSError as exc:
            raise ValueError(
                f"Failed to set CPU affinity {format_affinity_mask(requested_mask)}: {exc}"
            ) from exc
        masked = _cpus_to_mask(os.sched_getaffinity(0))
    else:
        raise RuntimeError(f"CPU affinity pinning is not supported on platform {sys.platform!r}")

    assert masked == requested_mask, (
        f"Process affinity verification failed: expected {format_affinity_mask(requested_mask)}, "
        f"got {format_affinity_mask(masked)}"
    )
    return previous_mask, masked


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


def _detect_git_provenance(repo_root: Path | None = None) -> tuple[str, bool]:
    cwd = str(_REPO_ROOT if repo_root is None else repo_root)
    commit = "unknown"
    dirty = False
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if rev.returncode == 0 and rev.stdout.strip():
            commit = rev.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
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
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    """Collect hardware, OS, power scheme, affinity, and git provenance metadata."""
    os_name = "windows" if sys.platform == "win32" else platform.system().lower()
    machine = platform.machine().lower()
    arch = "x86_64" if machine in ("amd64", "x86_64") else machine
    cpu_model = _detect_cpu_model()
    logical_cores = os.cpu_count() or 1
    total_ram_gb = _detect_total_ram_gb()
    power_mode, power_scheme_raw = detect_windows_power_scheme()
    git_commit, git_dirty = _detect_git_provenance(repo_root)

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
