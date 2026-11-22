"""Ledger guard tests verifying results/CLAIMS.md against tracked JSON artifacts."""

from __future__ import annotations

from pathlib import Path

from scripts.check_claims import check_claims
from tests.helpers import REPO_ROOT


def test_claims_ledger_matches_source_jsons() -> None:
    """Every single-JSON numeric row in results/CLAIMS.md matches within 0.5%."""
    claims_path: Path = REPO_ROOT / "results" / "CLAIMS.md"
    result = check_claims(claims_path=claims_path, repo_root=REPO_ROOT, rel_tol=0.005)
    print(
        f"claims_ledger: checked={result.checked} skipped={result.skipped} "
        f"total={result.total_rows}"
    )
    assert result.total_rows >= 280, f"Unexpectedly few CLAIMS.md rows: {result.total_rows}"
    assert result.checked >= 140, f"Unexpectedly few resolvable numeric rows: {result.checked}"
    assert not result.failures, (
        f"CLAIMS.md ledger mismatches ({len(result.failures)}):\n"
        + "\n".join(
            f"  L{m.line_no} {m.source_file}::{m.key_path}: {m.reason} "
            f"(doc={m.documented_actual!r}, json={m.resolved_value!r})"
            for m in result.failures
        )
    )
