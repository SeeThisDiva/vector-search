"""Batch 1 Python regression tests for PY-01, PY-02, PY-03, and PY-06."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_faiss
from scripts.check_claims import check_claims
from src.hnsw_phase1.benchmark import run_phase1_benchmark
from src.hnsw_phase1.dataset import write_fvecs, write_ivecs
from src.hnsw_phase1.exact_knn import ExactKNN


def test_regress_py01_custom_fvecs_ignores_cached_sift_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom --base-fvecs / --query-fvecs must not load cached SIFT ground truth."""
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    # Create a fake cached 100k GT file in monkeypatched _REPO_ROOT/data
    write_ivecs(fake_data / "gt_100000_k100.ivecs", np.zeros((5, 10), dtype=np.int32))
    monkeypatch.setattr(benchmark_faiss, "_REPO_ROOT", tmp_path)

    rng = np.random.default_rng(1)
    base = rng.standard_normal((100_000, 2), dtype=np.float32)
    query = rng.standard_normal((5, 2), dtype=np.float32)
    b_path = tmp_path / "custom_base.fvecs"
    q_path = tmp_path / "custom_query.fvecs"
    write_fvecs(b_path, base)
    write_fvecs(q_path, query)

    _, _, gt, _, gt_source, resolved_gt = benchmark_faiss._load_dataset_arrays(
        dataset="sift1m",
        synthetic=False,
        scale=None,
        queries=None,
        dim=2,
        num_clusters=1,
        seed=42,
        k=5,
        base_fvecs=str(b_path),
        query_fvecs=str(q_path),
        gt_path=None,
    )
    assert gt_source.startswith("computed_faiss_index_flat_l2")
    assert resolved_gt is None
    assert gt.shape == (5, 5)


def test_regress_py02_phase1_benchmark_k_gt_num_base() -> None:
    """run_phase1_benchmark handles k > num_base without FAISS broadcast ValueError."""
    res = run_phase1_benchmark(
        num_base=5,
        num_query=2,
        dim=8,
        k=10,
        seed=42,
        save_files=False,
    )
    assert res["recall"]["self_recall"] == pytest.approx(1.0)
    assert res["faiss_verification"]["faiss_available"] is True
    assert res["faiss_verification"]["faiss_parity"] is True
    assert res["faiss_verification"]["faiss_total_elements"] == 2 * 5


def test_regress_py03_exact_knn_rejects_nan_inf_zero_dim_and_bad_batch_size() -> None:
    """ExactKNN validates finite values, dim > 0, and batch_size > 0."""
    with pytest.raises(ValueError, match="at least 1 column"):
        ExactKNN(metric="l2").fit(np.empty((3, 0), dtype=np.float32))

    with pytest.raises(ValueError, match="non-finite"):
        ExactKNN(metric="l2").fit(np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="non-finite"):
        ExactKNN(metric="l2").fit(np.array([[np.inf, 0.0], [1.0, 0.0]], dtype=np.float32))

    knn = ExactKNN(metric="l2").fit(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        knn.search(np.array([[np.nan, 0.0]], dtype=np.float32), k=1)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        knn.search(np.array([[0.0, 0.0]], dtype=np.float32), k=1, batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        knn.search(np.array([[0.0, 0.0]], dtype=np.float32), k=1, batch_size=-1)


def test_regress_py06_check_claims_nan_malformed_json_and_hyphen_keys(tmp_path: Path) -> None:
    """check_claims reports NaN and malformed JSON as failures, and resolves hyphenated keys."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "nan.json").write_text('{"my-metric": NaN}', encoding="utf-8")
    (tmp_path / "results" / "broken.json").write_text("{broken json", encoding="utf-8")
    (tmp_path / "results" / "ok.json").write_text(
        '{"sub-key": 12.5, "flag": true}', encoding="utf-8"
    )

    claims_md = tmp_path / "CLAIMS.md"
    claims_md.write_text(
        "| Metric | Source Report JSON | JSON Key Path | Actual JSON Value |\n"
        "|---|---|---|---|\n"
        "| NaN | `results/nan.json` | `my-metric` | `123.45` |\n"
        "| Broken | `results/broken.json` | `x` | `1` |\n"
        "| Hyphen | `results/ok.json` | `sub-key` | `12.5` |\n"
        "| Flag | `results/ok.json` | `flag` | `true` |\n",
        encoding="utf-8",
    )

    res = check_claims(claims_md, tmp_path)
    assert res.total_rows == 4
    assert res.checked == 2  # NaN row + Hyphen row
    assert res.skipped == 1  # Flag row
    assert len(res.failures) == 2
    reasons = {f.reason for f in res.failures}
    assert "non-finite JSON value" in reasons
    assert any(r.startswith("malformed JSON") for r in reasons)
    assert dict(res.skip_reasons) == {"boolean JSON value": 1}
