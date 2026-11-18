"""Shared test helpers for Python unit and cross-language parity suites."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_rust_binary(bin_name: str) -> Path:
    """Locate or build the compiled Rust release binary `<bin_name>`, raising RuntimeError on failure."""
    exe_name = f"{bin_name}.exe" if os.name == "nt" else bin_name
    release_bin = REPO_ROOT / "target" / "release" / exe_name
    if release_bin.is_file():
        return release_bin

    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError(
            f"Compiled Rust binary `{bin_name}` not found at {release_bin} and `cargo` is not on PATH."
        )

    proc = subprocess.run(
        [cargo, "build", "--release", "--bin", bin_name],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not release_bin.is_file():
        raise RuntimeError(
            f"Failed to build release `{bin_name}` binary (rc={proc.returncode}):\n{proc.stderr}"
        )
    return release_bin


def assert_charts_created(
    output_dir: Path,
    stems: Sequence[str],
    min_bytes: int = 1000,
) -> None:
    """Verify that PNG and SVG chart artifacts exist, exceed `min_bytes`, and have valid headers."""
    for stem in stems:
        png = output_dir / f"{stem}.png"
        svg = output_dir / f"{stem}.svg"
        assert png.is_file(), f"Missing PNG chart: {png}"
        assert svg.is_file(), f"Missing SVG chart: {svg}"
        assert png.stat().st_size > min_bytes, f"Empty/tiny PNG chart: {png}"
        assert svg.stat().st_size > min_bytes, f"Empty/tiny SVG chart: {svg}"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG header in {png}"
        svg_text = svg.read_text(encoding="utf-8")
        assert "<svg" in svg_text and "</svg>" in svg_text, f"Invalid SVG content in {svg}"
