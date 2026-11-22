#!/usr/bin/env python3
"""Verify numeric values in results/CLAIMS.md against their source JSON reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JSON_REF_RE = re.compile(r"results/[A-Za-z0-9_./-]+\.json")
_KEY_PATH_RE = re.compile(r"[A-Za-z0-9_]+(\[\d+\])?(\.[A-Za-z0-9_]+(\[\d+\])?)*")
_KEY_TOKEN_RE = re.compile(r"[^.\[\]]+|\[\d+\]")
_SEP_CELL_RE = re.compile(r":?-+:?")
_NUM_LITERAL_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
_SOURCE_COL_NAMES = ("Source Report JSON", "Report JSON Path", "Source JSON Path")


@dataclass(frozen=True)
class ClaimMismatch:
    """Single mismatch or resolution error discovered in results/CLAIMS.md."""

    line_no: int
    source_file: str
    key_path: str
    reason: str
    documented_actual: str
    resolved_value: str | None = None


@dataclass(frozen=True)
class ClaimsCheckResult:
    """Aggregate summary returned by `check_claims`."""

    total_rows: int
    checked: int
    skipped: int
    failures: tuple[ClaimMismatch, ...]


def resolve_json_path(obj: Any, path_str: str) -> tuple[bool, Any]:
    """Resolve dotted field names and `[n]` list indices on parsed JSON data."""
    if not _KEY_PATH_RE.fullmatch(path_str):
        return False, None
    tokens = _KEY_TOKEN_RE.findall(path_str)
    cur: Any = obj
    for tok in tokens:
        if tok.startswith("["):
            idx = int(tok[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return False, None
            cur = cur[tok]
    return True, cur


def check_claims(
    claims_path: Path,
    repo_root: Path,
    rel_tol: float = 0.005,
) -> ClaimsCheckResult:
    """Validate single-JSON numeric rows in `claims_path` within `rel_tol`."""
    lines = claims_path.read_text(encoding="utf-8").splitlines()
    json_cache: dict[str, Any] = {}
    headers: list[str] = []
    section_source: str | None = None
    total_rows = 0
    checked = 0
    skipped = 0
    failures: list[ClaimMismatch] = []

    for lineno, line in enumerate(lines, start=1):
        if line.startswith("## "):
            section_source = None
            headers = []
            continue

        if not line.startswith("|"):
            if "Source report:" in line:
                found = _JSON_REF_RE.findall(line)
                section_source = found[0] if len(found) == 1 else None
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and all(_SEP_CELL_RE.fullmatch(c) for c in cells):
            continue
        if "JSON Key Path" in cells or "Actual JSON Value" in cells:
            headers = cells
            continue
        if not headers or len(cells) != len(headers):
            continue

        total_rows += 1
        row = dict(zip(headers, cells, strict=True))

        src_cell = ""
        has_source_col = any(col in row for col in _SOURCE_COL_NAMES)
        if has_source_col:
            for col in _SOURCE_COL_NAMES:
                if col in row:
                    src_cell = row[col]
                    break
        elif section_source is not None:
            src_cell = f"`{section_source}`"

        json_refs = _JSON_REF_RE.findall(src_cell)
        if len(json_refs) != 1 or "*" in src_cell or "{" in src_cell or "}" in src_cell:
            skipped += 1
            continue

        key_cell = row.get("JSON Key Path", "").strip("`").strip()
        if not _KEY_PATH_RE.fullmatch(key_cell):
            skipped += 1
            continue

        rel_json = json_refs[0]
        val_cell = row.get("Actual JSON Value", "").strip()

        if rel_json not in json_cache:
            target = repo_root / rel_json
            if not target.is_file():
                failures.append(
                    ClaimMismatch(
                        line_no=lineno,
                        source_file=rel_json,
                        key_path=key_cell,
                        reason="source JSON file not found",
                        documented_actual=val_cell,
                    )
                )
                continue
            json_cache[rel_json] = json.loads(target.read_text(encoding="utf-8"))

        ok, resolved = resolve_json_path(json_cache[rel_json], key_cell)
        if not ok:
            failures.append(
                ClaimMismatch(
                    line_no=lineno,
                    source_file=rel_json,
                    key_path=key_cell,
                    reason="JSON key path did not resolve",
                    documented_actual=val_cell,
                )
            )
            continue

        if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
            skipped += 1
            continue

        vclean = val_cell.strip("`").strip().replace(",", "")
        if not _NUM_LITERAL_RE.fullmatch(vclean):
            skipped += 1
            continue

        expected = float(vclean)
        actual = float(resolved)
        denom = max(abs(actual), 1e-12)
        rel_err = abs(expected - actual) / denom
        checked += 1
        if rel_err > rel_tol:
            failures.append(
                ClaimMismatch(
                    line_no=lineno,
                    source_file=rel_json,
                    key_path=key_cell,
                    reason=f"relative mismatch {rel_err * 100.0:.4f}% > {rel_tol * 100.0:.2f}%",
                    documented_actual=vclean,
                    resolved_value=repr(resolved),
                )
            )

    return ClaimsCheckResult(
        total_rows=total_rows,
        checked=checked,
        skipped=skipped,
        failures=tuple(failures),
    )


def main() -> int:
    """CLI entrypoint for validating results/CLAIMS.md."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Verify single-JSON numeric rows in results/CLAIMS.md."
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=repo_root / "results" / "CLAIMS.md",
        help="Path to CLAIMS.md ledger.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.005,
        help="Relative tolerance fraction (default: 0.005 = 0.5%%).",
    )
    args = parser.parse_args()

    result = check_claims(args.claims, repo_root, rel_tol=args.rel_tol)
    print(
        f"CLAIMS ledger check: {result.checked} checked, {result.skipped} skipped, "
        f"{len(result.failures)} failed (total {result.total_rows} rows)"
    )
    for item in result.failures:
        print(
            f"  FAIL L{item.line_no} {item.source_file}::{item.key_path}: "
            f"{item.reason} (doc={item.documented_actual!r}, json={item.resolved_value!r})",
            file=sys.stderr,
        )
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
