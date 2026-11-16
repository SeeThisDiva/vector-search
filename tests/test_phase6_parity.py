"""Independent NumPy SQ8/PQ reference implementation, parity tests, and Phase 6 plotter tests.

Covers Milestone M4 Tasks T4.7b and T4.9:
- Pure NumPy reference implementations for SQ8 per-dimension min/max calibration,
  encoding, decoding, Kernel A asymmetric L2^2, symmetric u8-u8 L2^2, and inner product.
- Pure NumPy reference implementations for PQ training (deterministic subsampling,
  k-means++ initialization, f64 Lloyd refinement with empty-cluster recovery),
  encoding, ADC lookup tables, and direct centroid reconstruction scoring.
- Reranking pipeline invariance (`R = k` equals `R = None`) vs recall lift (`R > k`,
  and `R = N` exact match) on deterministic synthetic clustered data.
- End-to-end tests of `scripts/plot_phase6.py` (full matrix, partial subset runs,
  Phase 4 baseline merge, scale auto-discovery, and deterministic SVG output) into `tmp_path`.

Note: Cross-binary Rust `phase6.exe` CLI parity checks will be added in Stage 2 once
`src/bin/phase6.rs` lands. Every test in this module runs and passes standalone with 0 skips.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.plot_phase6 import generate_all_phase6_charts

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PHASE6_CHART_STEMS: tuple[str, ...] = (
    "chart1_recall_vs_p99",
    "chart2_recall_vs_qps",
    "chart3_memory_vs_recall",
    "chart4_rerank_ablation",
)


# ============================================================================
# Reference SQ8 Implementation (Independent NumPy Reference per DESIGN.md §3)
# ============================================================================


@dataclass(frozen=True)
class Sq8Calibration:
    """Per-dimension min/max SQ8 calibration parameters."""

    mins: NDArray[np.float32]
    maxs: NDArray[np.float32]
    scales: NDArray[np.float32]
    inv_scales: NDArray[np.float32]
    scales_sq: NDArray[np.float32]


def _validate_2d_finite_f32(arr: NDArray[np.floating[Any]], name: str) -> NDArray[np.float32]:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite (NaN/Inf) entries")
    return out


def _validate_1d_finite_f32(arr: NDArray[np.floating[Any]], dim: int, name: str) -> NDArray[np.float32]:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 1 or out.shape[0] != dim:
        raise ValueError(f"{name} must have shape ({dim},), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite (NaN/Inf) entries")
    return out


def _round_half_away_from_zero(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Match Rust `f32::round` / `f64::round` half-away-from-zero semantics."""
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5))


def sq8_calibrate(base: NDArray[np.float32]) -> Sq8Calibration:
    """Calibrate per-dimension min/max SQ8 quantizer parameters using f64 spans."""
    mat = _validate_2d_finite_f32(base, "base")
    mins = np.min(mat, axis=0).astype(np.float32)
    maxs = np.max(mat, axis=0).astype(np.float32)

    delta_f64 = maxs.astype(np.float64) - mins.astype(np.float64)
    scales_f64 = np.divide(delta_f64, 255.0, out=np.zeros_like(delta_f64), where=delta_f64 > 0.0)
    inv_scales_f64 = np.divide(255.0, delta_f64, out=np.zeros_like(delta_f64), where=delta_f64 > 0.0)

    scales = scales_f64.astype(np.float32)
    inv_scales = inv_scales_f64.astype(np.float32)
    with np.errstate(over="ignore"):
        scales_sq = (scales.astype(np.float64) ** 2).astype(np.float32)

    return Sq8Calibration(
        mins=mins,
        maxs=maxs,
        scales=scales,
        inv_scales=inv_scales,
        scales_sq=scales_sq,
    )


def sq8_encode(vectors: NDArray[np.float32], cal: Sq8Calibration) -> NDArray[np.uint8]:
    """Quantize `(N, D)` float32 vectors to `(N, D)` uint8 SQ8 codes."""
    mat = _validate_2d_finite_f32(vectors, "vectors")
    dim = cal.mins.shape[0]
    if mat.shape[1] != dim:
        raise ValueError(f"Dimension mismatch: expected {dim}, got {mat.shape[1]}")

    # Compute shifted & scaled values in f64 to avoid f32 span overflow on extreme ranges.
    shifted = mat.astype(np.float64) - cal.mins.astype(np.float64)
    scaled = shifted * cal.inv_scales.astype(np.float64)
    rounded = _round_half_away_from_zero(scaled)
    clamped = np.clip(rounded, 0.0, 255.0)
    return clamped.astype(np.uint8)


def sq8_decode(codes: NDArray[np.uint8], cal: Sq8Calibration) -> NDArray[np.float32]:
    """Dequantize `(N, D)` uint8 SQ8 codes back to float32 approximations."""
    u8 = np.asarray(codes, dtype=np.uint8)
    if u8.ndim != 2 or u8.shape[1] != cal.mins.shape[0]:
        raise ValueError(f"Expected codes shape (N, {cal.mins.shape[0]}), got {u8.shape}")
    recon = cal.mins.astype(np.float64) + u8.astype(np.float64) * cal.scales.astype(np.float64)
    return recon.astype(np.float32)


def sq8_decode_f64(codes: NDArray[np.uint8], cal: Sq8Calibration) -> NDArray[np.float64]:
    """Dequantize `(N, D)` uint8 SQ8 codes to high-precision float64 reference."""
    u8 = np.asarray(codes, dtype=np.uint8)
    if u8.ndim != 2 or u8.shape[1] != cal.mins.shape[0]:
        raise ValueError(f"Expected codes shape (N, {cal.mins.shape[0]}), got {u8.shape}")
    return cal.mins.astype(np.float64) + u8.astype(np.float64) * cal.scales.astype(np.float64)


def sq8_asymmetric_l2_squared(
    query: NDArray[np.float32],
    codes: NDArray[np.uint8],
    cal: Sq8Calibration,
) -> NDArray[np.float32]:
    """Compute Kernel A query-shifted asymmetric L2^2 distances in float32."""
    dim = cal.mins.shape[0]
    q = _validate_1d_finite_f32(query, dim, "query")
    u8 = np.asarray(codes, dtype=np.uint8)
    if u8.ndim != 2 or u8.shape[1] != dim:
        raise ValueError(f"Expected codes shape (N, {dim}), got {u8.shape}")

    q_shifted = (q - cal.mins).astype(np.float32)
    diff = q_shifted - (u8.astype(np.float32) * cal.scales)
    d2 = np.sum(diff * diff, axis=1, dtype=np.float32)
    return np.maximum(np.float32(0.0), d2).astype(np.float32)


def sq8_asymmetric_l2_squared_f64(
    query: NDArray[np.float32],
    codes: NDArray[np.uint8],
    cal: Sq8Calibration,
) -> NDArray[np.float64]:
    """Compute high-precision float64 dequantized asymmetric L2^2 reference."""
    dim = cal.mins.shape[0]
    q = _validate_1d_finite_f32(query, dim, "query").astype(np.float64)
    recon = sq8_decode_f64(codes, cal)
    diff = q - recon
    d2 = np.sum(diff * diff, axis=1, dtype=np.float64)
    return np.maximum(0.0, d2)


def sq8_symmetric_l2_squared(
    code_a: NDArray[np.uint8],
    code_b: NDArray[np.uint8],
    cal: Sq8Calibration,
) -> float:
    """Compute symmetric SQ8 code-to-code L2^2 distance (`float`)."""
    dim = cal.mins.shape[0]
    a = np.asarray(code_a, dtype=np.uint8)
    b = np.asarray(code_b, dtype=np.uint8)
    if a.shape != (dim,) or b.shape != (dim,):
        raise ValueError(f"Expected 1D codes of length {dim}, got {a.shape} and {b.shape}")
    diff = (a.astype(np.float32) - b.astype(np.float32)) * cal.scales
    d2 = float(np.sum(diff * diff, dtype=np.float32))
    return max(0.0, d2)


def sq8_asymmetric_inner_product(
    query: NDArray[np.float32],
    codes: NDArray[np.uint8],
    cal: Sq8Calibration,
) -> NDArray[np.float32]:
    """Compute Kernel D asymmetric SQ8 inner product `<q, v_hat>` without inner-loop division."""
    dim = cal.mins.shape[0]
    q = _validate_1d_finite_f32(query, dim, "query")
    u8 = np.asarray(codes, dtype=np.uint8)
    if u8.ndim != 2 or u8.shape[1] != dim:
        raise ValueError(f"Expected codes shape (N, {dim}), got {u8.shape}")
    c_ip = np.sum(q * cal.mins, dtype=np.float32)
    q_scaled = (q * cal.scales).astype(np.float32)
    dots = np.sum(u8.astype(np.float32) * q_scaled, axis=1, dtype=np.float32)
    return (c_ip + dots).astype(np.float32)


# ============================================================================
# Reference PQ Implementation (Independent NumPy Reference per DESIGN.md §4)
# ============================================================================


@dataclass(frozen=True)
class PqCodebook:
    """Product Quantization codebook trained across `m_pq` orthogonal subspaces."""

    m_pq: int
    k_centroids: int
    k_eff: int
    d_sub: int
    centroids: NDArray[np.float32]  # shape (m_pq, k_centroids, d_sub)


def _pairwise_sq_dists_f64(
    points: NDArray[np.float64],
    centers: NDArray[np.float64],
    batch_size: int = 2048,
) -> NDArray[np.float64]:
    """Compute exact chunked `(N, K)` squared Euclidean distances without norm cancellation."""
    n_pts = points.shape[0]
    k_cnt = centers.shape[0]
    out = np.empty((n_pts, k_cnt), dtype=np.float64)
    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        diff = points[start:end, None, :] - centers[None, :, :]
        out[start:end] = np.sum(diff * diff, axis=2)
    return out


def _deterministic_kmeans_pp_init(
    sub_train: NDArray[np.float64],
    k_eff: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Initialize `k_eff` centroids on `sub_train` via deterministic k-means++."""
    n_train, d_sub = sub_train.shape
    centers = np.empty((k_eff, d_sub), dtype=np.float64)

    first_idx = int(rng.integers(0, n_train))
    centers[0] = sub_train[first_idx]
    diff0 = sub_train - centers[0]
    min_sq = np.sum(diff0 * diff0, axis=1)

    for c_idx in range(1, k_eff):
        total = float(np.sum(min_sq))
        if total > 0.0 and np.isfinite(total):
            probs = min_sq / total
            chosen = int(rng.choice(n_train, p=probs))
        else:
            chosen = int(rng.integers(0, n_train))
        centers[c_idx] = sub_train[chosen]
        diff_c = sub_train - centers[c_idx]
        d_sq = np.sum(diff_c * diff_c, axis=1)
        np.minimum(min_sq, d_sq, out=min_sq)

    return centers


def pq_train(
    base: NDArray[np.float32],
    m_pq: int,
    k_centroids: int = 256,
    max_train_vectors: int = 65_536,
    max_lloyd_iters: int = 25,
    seed: int = 42,
    tol: float = 1e-5,
) -> PqCodebook:
    """Train PQ centroids independently per subspace with deterministic k-means++ + Lloyd."""
    mat = _validate_2d_finite_f32(base, "base")
    num_base, dim = mat.shape
    if m_pq <= 0 or dim % m_pq != 0:
        raise ValueError(f"dim ({dim}) must be divisible by positive m_pq ({m_pq})")
    if not (1 <= k_centroids <= 256):
        raise ValueError(f"k_centroids must be in [1, 256], got {k_centroids}")
    if max_train_vectors <= 0 or max_lloyd_iters <= 0:
        raise ValueError("max_train_vectors and max_lloyd_iters must be positive")

    d_sub = dim // m_pq
    rng = np.random.default_rng(seed)

    n_train = min(num_base, max_train_vectors)
    if n_train == num_base:
        train_mat = mat
    else:
        indices = rng.choice(num_base, size=n_train, replace=False)
        train_mat = mat[indices]

    k_eff = min(k_centroids, n_train)
    centroids_out = np.zeros((m_pq, k_centroids, d_sub), dtype=np.float32)

    for m_idx in range(m_pq):
        col_lo = m_idx * d_sub
        col_hi = col_lo + d_sub
        sub_train = np.ascontiguousarray(train_mat[:, col_lo:col_hi], dtype=np.float64)

        centers = _deterministic_kmeans_pp_init(sub_train, k_eff, rng)
        prev_assign: NDArray[np.intp] | None = None
        prev_mse = float("inf")

        for _it in range(max_lloyd_iters):
            dists = _pairwise_sq_dists_f64(sub_train, centers)
            # np.argmin breaks ties canonically by smallest centroid index.
            assign = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(n_train), assign]
            mse = float(np.mean(min_dists))

            if prev_assign is not None and np.array_equal(assign, prev_assign):
                break

            counts = np.bincount(assign, minlength=k_eff).astype(np.int64)
            new_centers = np.zeros_like(centers)
            for c_idx in range(k_eff):
                if counts[c_idx] > 0:
                    new_centers[c_idx] = np.mean(sub_train[assign == c_idx], axis=0)

            # Empty-cluster recovery: split the highest-SSE donor cluster with >= 2 points.
            for empty_k in np.flatnonzero(counts == 0):
                cluster_sse = np.bincount(assign, weights=min_dists, minlength=k_eff)
                eligible = np.flatnonzero(counts >= 2)
                if eligible.size == 0:
                    new_centers[empty_k] = new_centers[0]
                    continue
                donor_k = int(
                    max(
                        eligible,
                        key=lambda ck: (float(cluster_sse[ck]), int(counts[ck]), -int(ck)),
                    )
                )
                donor_members = np.flatnonzero(assign == donor_k)
                donor_diff = sub_train[donor_members] - new_centers[donor_k]
                donor_sq = np.sum(donor_diff * donor_diff, axis=1)
                far_local = int(np.argmax(donor_sq))
                far_global = int(donor_members[far_local])

                new_centers[empty_k] = sub_train[far_global]
                assign[far_global] = int(empty_k)
                counts[donor_k] -= 1
                counts[empty_k] = 1
                if counts[donor_k] > 0:
                    new_centers[donor_k] = np.mean(sub_train[assign == donor_k], axis=0)
                min_dists[far_global] = 0.0

            centers = new_centers
            if prev_mse > 0.0 and np.isfinite(prev_mse) and (prev_mse - mse) / prev_mse < tol:
                break
            prev_mse = mse
            prev_assign = assign

        centroids_f32 = centers.astype(np.float32)
        centroids_out[m_idx, :k_eff, :] = centroids_f32
        if k_eff < k_centroids:
            centroids_out[m_idx, k_eff:, :] = centroids_f32[0:1, :]

    return PqCodebook(
        m_pq=m_pq,
        k_centroids=k_centroids,
        k_eff=k_eff,
        d_sub=d_sub,
        centroids=centroids_out,
    )


def pq_encode(vectors: NDArray[np.float32], codebook: PqCodebook) -> NDArray[np.uint8]:
    """Encode `(N, D)` float32 vectors into `(N, m_pq)` uint8 PQ codes."""
    mat = _validate_2d_finite_f32(vectors, "vectors")
    expected_dim = codebook.m_pq * codebook.d_sub
    if mat.shape[1] != expected_dim:
        raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {mat.shape[1]}")

    codes = np.empty((mat.shape[0], codebook.m_pq), dtype=np.uint8)
    for m_idx in range(codebook.m_pq):
        col_lo = m_idx * codebook.d_sub
        col_hi = col_lo + codebook.d_sub
        sub = mat[:, col_lo:col_hi].astype(np.float64)
        cents = codebook.centroids[m_idx, : codebook.k_eff, :].astype(np.float64)
        dists = _pairwise_sq_dists_f64(sub, cents)
        codes[:, m_idx] = np.argmin(dists, axis=1).astype(np.uint8)
    return codes


def pq_compute_adc_table(query: NDArray[np.float32], codebook: PqCodebook) -> NDArray[np.float32]:
    """Precompute `(m_pq, k_centroids)` query-to-centroid squared distance table."""
    dim = codebook.m_pq * codebook.d_sub
    q = _validate_1d_finite_f32(query, dim, "query")
    table = np.empty((codebook.m_pq, codebook.k_centroids), dtype=np.float32)
    for m_idx in range(codebook.m_pq):
        col_lo = m_idx * codebook.d_sub
        col_hi = col_lo + codebook.d_sub
        q_sub = q[col_lo:col_hi].astype(np.float64)
        cents = codebook.centroids[m_idx].astype(np.float64)
        diff = cents - q_sub[None, :]
        table[m_idx] = np.sum(diff * diff, axis=1).astype(np.float32)
    return table


def pq_adc_distances(adc_table: NDArray[np.float32], codes: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Score `(N, m_pq)` PQ codes against precomputed `(m_pq, k_centroids)` ADC lookup table."""
    tbl = np.asarray(adc_table, dtype=np.float32)
    u8 = np.asarray(codes, dtype=np.uint8)
    if tbl.ndim != 2 or u8.ndim != 2 or u8.shape[1] != tbl.shape[0]:
        raise ValueError(f"Incompatible ADC table {tbl.shape} and codes {u8.shape}")
    subspace_indices = np.arange(tbl.shape[0])[None, :]
    gathered = tbl[subspace_indices, u8]
    d2 = np.sum(gathered, axis=1, dtype=np.float32)
    return np.maximum(np.float32(0.0), d2).astype(np.float32)


def pq_direct_centroid_distances(
    query: NDArray[np.float32],
    codes: NDArray[np.uint8],
    codebook: PqCodebook,
) -> NDArray[np.float32]:
    """Score PQ codes by explicitly reconstructing vectors from centroids and summing L2^2."""
    dim = codebook.m_pq * codebook.d_sub
    q = _validate_1d_finite_f32(query, dim, "query")
    u8 = np.asarray(codes, dtype=np.uint8)
    if u8.ndim != 2 or u8.shape[1] != codebook.m_pq:
        raise ValueError(f"Expected codes shape (N, {codebook.m_pq}), got {u8.shape}")

    recon = np.empty((u8.shape[0], dim), dtype=np.float32)
    for m_idx in range(codebook.m_pq):
        col_lo = m_idx * codebook.d_sub
        col_hi = col_lo + codebook.d_sub
        recon[:, col_lo:col_hi] = codebook.centroids[m_idx][u8[:, m_idx]]
    diff = recon - q[None, :]
    d2 = np.sum(diff * diff, axis=1, dtype=np.float32)
    return np.maximum(np.float32(0.0), d2).astype(np.float32)


# ============================================================================
# Reference Rerank & Exact Top-k Helpers
# ============================================================================


def exact_l2_squared_topk(
    queries: NDArray[np.float32],
    base: NDArray[np.float32],
    k: int,
) -> NDArray[np.int32]:
    """Compute canonical `(distance, id)` exact top-k L2^2 neighbor IDs."""
    base_mat = _validate_2d_finite_f32(base, "base")
    query_mat = _validate_2d_finite_f32(queries, "queries")
    if base_mat.shape[1] != query_mat.shape[1]:
        raise ValueError("Base and query dimensions must match")
    if k <= 0:
        raise ValueError("k must be >= 1")

    eff_k = min(k, base_mat.shape[0])
    ids_all = np.arange(base_mat.shape[0], dtype=np.int32)
    out = np.empty((query_mat.shape[0], eff_k), dtype=np.int32)
    for q_idx, q in enumerate(query_mat):
        diff = base_mat - q[None, :]
        dists = np.sum(diff * diff, axis=1, dtype=np.float32)
        np.maximum(np.float32(0.0), dists, out=dists)
        order = np.lexsort((ids_all, dists))
        out[q_idx] = ids_all[order[:eff_k]]
    return out


def rerank_top_r_search(
    query: NDArray[np.float32],
    base: NDArray[np.float32],
    approx_dists: NDArray[np.float32],
    k: int,
    ef_search: int,
    rerank_top_r: int | None,
) -> NDArray[np.int32]:
    """Simulate quantized candidate selection (`ef_eff = max(ef, R, k)`) + exact top-R rerank."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if rerank_top_r == 0:
        raise ValueError("rerank_top_r == 0 is invalid; pass None to disable reranking")

    base_mat = _validate_2d_finite_f32(base, "base")
    q = _validate_1d_finite_f32(query, base_mat.shape[1], "query")
    approx = np.asarray(approx_dists, dtype=np.float32)
    if approx.shape != (base_mat.shape[0],):
        raise ValueError(f"approx_dists shape {approx.shape} != ({base_mat.shape[0]},)")

    n_base = base_mat.shape[0]
    eff_k = min(k, n_base)
    ids_all = np.arange(n_base, dtype=np.int32)
    approx_order = np.lexsort((ids_all, approx))

    if rerank_top_r is None:
        return approx_order[:eff_k].astype(np.int32)

    eff_r = min(max(rerank_top_r, eff_k), n_base)
    eff_ef = min(max(ef_search, eff_r), n_base)
    beam_ids = approx_order[:eff_ef]
    cand_ids = beam_ids[:eff_r]

    diff = base_mat[cand_ids] - q[None, :]
    exact_dists = np.sum(diff * diff, axis=1, dtype=np.float32)
    np.maximum(np.float32(0.0), exact_dists, out=exact_dists)
    rerank_order = np.lexsort((cand_ids, exact_dists))
    return cand_ids[rerank_order[:eff_k]].astype(np.int32)


def compute_recall_at_k(
    approx_ids: NDArray[np.int32],
    gt_ids: NDArray[np.int32],
    k: int,
) -> float:
    """Compute average set-overlap Recall@k across all queries."""
    if approx_ids.shape[0] != gt_ids.shape[0] or approx_ids.shape[0] == 0:
        raise ValueError("Mismatched or empty query counts in compute_recall_at_k")
    total = 0.0
    for q_idx in range(approx_ids.shape[0]):
        gt_set = set(gt_ids[q_idx, :k].tolist())
        ap_set = set(approx_ids[q_idx, :k].tolist())
        total += float(len(gt_set & ap_set)) / float(len(gt_set))
    return total / float(approx_ids.shape[0])


# ============================================================================
# SQ8 Parity & Mathematical Tests
# ============================================================================


def test_sq8_calibration_and_reconstruction_error_bound() -> None:
    """Verify SQ8 reconstruction error satisfies |v - recon| <= delta/510 + 1e-5."""
    rng = np.random.default_rng(2026)
    base_uniform = rng.uniform(-50.0, 175.0, size=(300, 64)).astype(np.float32)
    base_gaussian = (rng.standard_normal((300, 64)) * 25.0 - 12.5).astype(np.float32)
    base_skewed = rng.exponential(scale=12.0, size=(300, 64)).astype(np.float32)
    # Mix vastly different per-dimension scales ([0, 1e-5] vs [-1e4, 1e4]).
    scale_vec = np.concatenate([np.full(32, 1e-5), np.full(32, 1e4)]).astype(np.float32)
    base_mixed = (rng.uniform(-1.0, 1.0, size=(300, 64)) * scale_vec[None, :]).astype(np.float32)

    for mat in (base_uniform, base_gaussian, base_skewed, base_mixed):
        cal = sq8_calibrate(mat)
        codes = sq8_encode(mat, cal)
        recon = sq8_decode(codes, cal)
        delta = cal.maxs.astype(np.float64) - cal.mins.astype(np.float64)
        max_allowed = (delta / 510.0) + 1e-5
        abs_err = np.abs(mat.astype(np.float64) - recon.astype(np.float64))
        assert np.all(abs_err <= max_allowed[None, :]), (
            f"Max error {np.max(abs_err - max_allowed[None, :])} exceeded delta/510 + 1e-5"
        )


def test_sq8_f64_calibration_precision_and_extreme_range() -> None:
    """Verify f64 span calculation prevents f32 overflow on [-3e38, 3e38] inputs."""
    base = np.array(
        [
            [-3.0e38, -1.0e-5, 0.0],
            [3.0e38, 1.0e-5, 255.0],
            [0.0, 0.0, 127.5],
        ],
        dtype=np.float32,
    )
    cal = sq8_calibrate(base)
    assert np.all(np.isfinite(cal.scales))
    assert np.all(np.isfinite(cal.inv_scales))
    assert cal.scales[0] > 2.0e36

    codes = sq8_encode(base, cal)
    assert codes.dtype == np.uint8
    assert codes[0, 0] == 0
    assert codes[1, 0] == 255
    assert codes[2, 0] in (127, 128)
    assert codes[0, 2] == 0
    assert codes[1, 2] == 255


def test_sq8_constant_dimension_and_single_vector_handling() -> None:
    """Verify constant dimensions (min == max), all-zeros, and N=1 produce exact codes."""
    base = np.array(
        [
            [42.5, 0.0, -7.25, 1.0],
            [42.5, 0.0, -7.25, 10.0],
            [42.5, 0.0, -7.25, 4.0],
        ],
        dtype=np.float32,
    )
    cal = sq8_calibrate(base)
    assert cal.scales[0] == 0.0 and cal.inv_scales[0] == 0.0
    assert cal.scales[1] == 0.0 and cal.inv_scales[1] == 0.0
    assert cal.scales[2] == 0.0 and cal.inv_scales[2] == 0.0
    assert cal.scales[3] > 0.0

    codes = sq8_encode(base, cal)
    assert np.all(codes[:, :3] == 0)
    recon = sq8_decode(codes, cal)
    assert np.array_equal(recon[:, :3], base[:, :3])

    single = np.array([[3.25, -9.5, 0.0]], dtype=np.float32)
    cal_single = sq8_calibrate(single)
    codes_single = sq8_encode(single, cal_single)
    assert np.all(codes_single == 0)
    assert np.array_equal(sq8_decode(codes_single, cal_single), single)


def test_sq8_asymmetric_l2_squared_f64_parity_across_dims() -> None:
    """Verify Kernel A f32 asymmetric L2^2 matches f64 dequantized reference across odd/even dims."""
    rng = np.random.default_rng(77)
    for dim in (1, 3, 7, 8, 9, 16, 128, 131):
        base = rng.normal(loc=10.0, scale=15.0, size=(64, dim)).astype(np.float32)
        query = rng.normal(loc=10.0, scale=15.0, size=(dim,)).astype(np.float32)
        cal = sq8_calibrate(base)
        codes = sq8_encode(base, cal)

        d_f32 = sq8_asymmetric_l2_squared(query, codes, cal)
        d_f64 = sq8_asymmetric_l2_squared_f64(query, codes, cal)
        assert d_f32.dtype == np.float32
        assert np.all(d_f32 >= 0.0)
        assert np.allclose(d_f32.astype(np.float64), d_f64, rtol=1e-5, atol=1e-4), (
            f"Mismatch at dim={dim}"
        )

        # Also verify Inner Product kernel parity vs f64 dot(q, v_hat)
        ip_f32 = sq8_asymmetric_inner_product(query, codes, cal)
        ip_f64 = sq8_decode_f64(codes, cal) @ query.astype(np.float64)
        assert np.allclose(ip_f32.astype(np.float64), ip_f64, rtol=1e-4, atol=1e-3)


def test_sq8_shifted_kernel_resists_cancellation() -> None:
    """Verify query-shifted Kernel A avoids catastrophic cancellation at large offsets."""
    # Anchor row sets min = 0.0 while clustered rows live at 10000.0 + tiny eps.
    base = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            [10_000.05, 10_000.02, 9_999.98, 10_000.01],
            [10_000.10, 9_999.95, 10_000.04, 10_000.00],
        ],
        dtype=np.float32,
    )
    query = np.array([10_000.01, 10_000.01, 10_000.01, 10_000.01], dtype=np.float32)
    cal = sq8_calibrate(base)
    codes = sq8_encode(base, cal)

    d_shifted = sq8_asymmetric_l2_squared(query, codes, cal)
    d_f64 = sq8_asymmetric_l2_squared_f64(query, codes, cal)
    # Query-shifted Kernel A avoids squaring 10,000^2 and stays accurate within 1e-4,
    # whereas norm-expanded f32 (`||q'||^2 - 2 q_hat*c + ||c'||^2`) loses > 1.0 to cancellation.
    assert np.allclose(d_shifted[1:].astype(np.float64), d_f64[1:], rtol=1e-2, atol=1e-4)

    q_shifted = (query - cal.mins).astype(np.float32)
    c_q = np.sum(q_shifted * q_shifted, dtype=np.float32)
    q_hat = (q_shifted * cal.scales).astype(np.float32)
    scaled_c = codes.astype(np.float32) * cal.scales
    b_norms = np.sum(scaled_c * scaled_c, axis=1, dtype=np.float32)
    d_expanded = c_q - np.float32(2.0) * np.sum(codes.astype(np.float32) * q_hat, axis=1, dtype=np.float32) + b_norms
    err_shifted = np.max(np.abs(d_shifted[1:].astype(np.float64) - d_f64[1:]))
    err_expanded = np.max(np.abs(d_expanded[1:].astype(np.float64) - d_f64[1:]))
    assert err_shifted < 1e-4
    assert err_expanded > 1.0


def test_sq8_symmetric_distance_axioms_and_validation() -> None:
    """Verify symmetric u8-u8 axioms and non-finite / dimension-mismatch rejection."""
    rng = np.random.default_rng(11)
    base = rng.uniform(-10.0, 10.0, size=(20, 16)).astype(np.float32)
    cal = sq8_calibrate(base)
    codes = sq8_encode(base, cal)
    recon_f64 = sq8_decode_f64(codes, cal)

    for i in range(5):
        assert sq8_symmetric_l2_squared(codes[i], codes[i], cal) == 0.0
        for j in range(i + 1, 6):
            dab = sq8_symmetric_l2_squared(codes[i], codes[j], cal)
            dba = sq8_symmetric_l2_squared(codes[j], codes[i], cal)
            ref = float(np.sum((recon_f64[i] - recon_f64[j]) ** 2))
            assert dab >= 0.0
            assert dab == dba
            assert np.isclose(dab, ref, rtol=1e-5, atol=1e-5)

    try:
        bad_base = base.copy()
        bad_base[0, 0] = np.nan
        sq8_calibrate(bad_base)
        raise AssertionError("Expected ValueError for NaN base vector")
    except ValueError:
        pass

    try:
        sq8_asymmetric_l2_squared(np.zeros(15, dtype=np.float32), codes, cal)
        raise AssertionError("Expected ValueError for query dimension mismatch")
    except ValueError:
        pass


# ============================================================================
# PQ Reference & ADC Parity Tests
# ============================================================================


def test_pq_training_determinism_and_validation() -> None:
    """Verify PQ training is deterministic per seed and rejects invalid m_pq or NaNs."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal((128, 32)).astype(np.float32)

    cb1 = pq_train(base, m_pq=8, k_centroids=16, max_lloyd_iters=10, seed=42)
    cb2 = pq_train(base, m_pq=8, k_centroids=16, max_lloyd_iters=10, seed=42)
    cb3 = pq_train(base, m_pq=8, k_centroids=16, max_lloyd_iters=10, seed=99)

    assert np.array_equal(cb1.centroids, cb2.centroids)
    assert np.array_equal(pq_encode(base, cb1), pq_encode(base, cb2))
    assert not np.array_equal(cb1.centroids, cb3.centroids)

    try:
        pq_train(base, m_pq=6)
        raise AssertionError("Expected ValueError when dim % m_pq != 0")
    except ValueError:
        pass

    try:
        bad = base.copy()
        bad[3, 1] = np.inf
        pq_train(bad, m_pq=8)
        raise AssertionError("Expected ValueError on non-finite input")
    except ValueError:
        pass


def test_pq_adc_table_matches_direct_centroid_sum() -> None:
    """Verify PQ ADC lookup scoring matches explicit centroid reconstruction across m_pq."""
    rng = np.random.default_rng(123)
    dim = 128
    base = rng.normal(loc=5.0, scale=2.5, size=(150, dim)).astype(np.float32)
    queries = rng.normal(loc=5.0, scale=2.5, size=(5, dim)).astype(np.float32)

    for m_pq in (4, 8, 16, 32):
        cb = pq_train(base, m_pq=m_pq, k_centroids=32, max_lloyd_iters=12, seed=42)
        codes = pq_encode(base, cb)
        assert codes.shape == (150, m_pq)

        for q in queries:
            adc_table = pq_compute_adc_table(q, cb)
            adc_dists = pq_adc_distances(adc_table, codes)
            direct_dists = pq_direct_centroid_distances(q, codes, cb)
            assert adc_dists.dtype == np.float32
            assert np.all(adc_dists >= 0.0)
            assert np.allclose(adc_dists, direct_dists, rtol=1e-5, atol=1e-4), (
                f"PQ ADC mismatch at m_pq={m_pq}"
            )


def test_pq_degenerate_small_n_duplicates_and_zeros() -> None:
    """Verify PQ handles N < 256, duplicate vectors, and all-zero vectors cleanly."""
    zeros = np.zeros((12, 32), dtype=np.float32)
    cb_zero = pq_train(zeros, m_pq=8, k_centroids=256, max_lloyd_iters=10, seed=42)
    codes_zero = pq_encode(zeros, cb_zero)
    assert cb_zero.k_eff == 12
    assert np.all(np.isfinite(cb_zero.centroids))
    assert np.all(codes_zero < cb_zero.k_eff)

    dups = np.tile(np.arange(32, dtype=np.float32), (20, 1))
    dups[:4] += 1.5
    cb_dup = pq_train(dups, m_pq=16, k_centroids=256, max_lloyd_iters=15, seed=7)
    codes_dup = pq_encode(dups, cb_dup)
    q = np.ones(32, dtype=np.float32)
    dists = pq_adc_distances(pq_compute_adc_table(q, cb_dup), codes_dup)
    assert np.all(np.isfinite(dists))
    assert np.all(dists >= 0.0)


def test_pq_subspace_exactness_tight_clusters() -> None:
    """Verify PQ with K >= num_modes recovers 1.0 Recall@10 on well-separated clusters."""
    rng = np.random.default_rng(55)
    num_modes = 32
    dim = 64
    k = 10
    modes = rng.uniform(-50.0, 50.0, size=(num_modes, dim)).astype(np.float32)
    # Place exactly k=10 members per mode so the exact top-10 set is unambiguously the mode itself.
    assignments = np.repeat(np.arange(num_modes), k)
    base = (modes[assignments] + rng.normal(0.0, 0.01, size=(len(assignments), dim))).astype(
        np.float32
    )
    queries = (modes[:10] + rng.normal(0.0, 0.01, size=(10, dim))).astype(np.float32)

    cb = pq_train(base, m_pq=16, k_centroids=64, max_lloyd_iters=20, seed=42)
    codes = pq_encode(base, cb)
    gt_ids = exact_l2_squared_topk(queries, base, k=k)

    approx_ids = np.empty((10, k), dtype=np.int32)
    for q_idx, q in enumerate(queries):
        dists = pq_adc_distances(pq_compute_adc_table(q, cb), codes)
        approx_ids[q_idx] = rerank_top_r_search(
            q, base, dists, k=k, ef_search=50, rerank_top_r=None
        )

    recall = compute_recall_at_k(approx_ids, gt_ids, k=k)
    assert recall == 1.0, f"Expected exact recall=1.0 on separated modes, got {recall}"


# ============================================================================
# Reranking Invariance (R=k) vs Recall Lift (R>k) Test
# ============================================================================


def test_rerank_r_equals_k_invariance_and_r_gt_k_lift() -> None:
    """Verify R=k yields identical Recall@k to R=None, while R>k strictly lifts Recall@k."""
    rng = np.random.default_rng(42)
    n_base = 600
    n_query = 30
    dim = 64
    k = 10
    ef_search = 100

    centers = rng.normal(0.0, 15.0, size=(16, dim)).astype(np.float32)
    c_assign = rng.integers(0, 16, size=n_base)
    base = (centers[c_assign] + rng.normal(0.0, 4.5, size=(n_base, dim))).astype(np.float32)
    q_assign = rng.integers(0, 16, size=n_query)
    queries = (centers[q_assign] + rng.normal(0.0, 4.5, size=(n_query, dim))).astype(np.float32)

    # Train coarse PQ (m_pq=8, k=16) so quantized top-10 omits several exact neighbors
    # that still reside inside the top-50 / top-150 quantized candidates.
    cb = pq_train(base, m_pq=8, k_centroids=16, max_lloyd_iters=15, seed=42)
    codes = pq_encode(base, cb)
    gt_ids = exact_l2_squared_topk(queries, base, k=k)

    # Confirm invalid rerank_top_r == 0 is rejected
    q0_dists = pq_adc_distances(pq_compute_adc_table(queries[0], cb), codes)
    try:
        rerank_top_r_search(queries[0], base, q0_dists, k=k, ef_search=ef_search, rerank_top_r=0)
        raise AssertionError("Expected ValueError for rerank_top_r=0")
    except ValueError:
        pass

    results_by_r: dict[int | None, NDArray[np.int32]] = {}
    for r_val in (None, k, 50, 150, n_base):
        out = np.empty((n_query, k), dtype=np.int32)
        for q_idx, q in enumerate(queries):
            approx_dists = pq_adc_distances(pq_compute_adc_table(q, cb), codes)
            out[q_idx] = rerank_top_r_search(
                q,
                base,
                approx_dists,
                k=k,
                ef_search=ef_search,
                rerank_top_r=r_val,
            )
        results_by_r[r_val] = out

    # 1. R=k must produce the exact same set of IDs per query as R=None -> identical Recall@k
    for q_idx in range(n_query):
        assert set(results_by_r[None][q_idx].tolist()) == set(results_by_r[k][q_idx].tolist())
    recall_none = compute_recall_at_k(results_by_r[None], gt_ids, k=k)
    recall_k = compute_recall_at_k(results_by_r[k], gt_ids, k=k)
    assert recall_none == recall_k

    # 2. R > k must recover substantial recall over R = k
    recall_r50 = compute_recall_at_k(results_by_r[50], gt_ids, k=k)
    recall_r150 = compute_recall_at_k(results_by_r[150], gt_ids, k=k)
    assert recall_r50 > recall_k, f"Expected R=50 ({recall_r50}) > R=k ({recall_k})"
    assert recall_r150 >= recall_r50

    # 3. R = N must match ExactKnn bit-for-bit
    assert np.array_equal(results_by_r[n_base], gt_ids)
    assert compute_recall_at_k(results_by_r[n_base], gt_ids, k=k) == 1.0


# ============================================================================
# End-to-End Plotter Tests (`scripts/plot_phase6.py`)
# ============================================================================


def _make_synthetic_arm_json(
    *,
    arm_id: str,
    quantizer_kind: str,
    storage_mode: str,
    build_mode: str,
    m: int,
    ef_construction: int,
    scale: int,
    steady_bv: float,
    build_rss_mb: float | None,
    query_rss_mb: float | None,
    recall_offset: float,
    latency_scale: float,
    rerank_depths: list[int | str] | None = None,
) -> dict[str, Any]:
    efs = [10, 20, 50, 100, 200]
    sweep: list[dict[str, Any]] = []
    depths: list[int | str | None] = list(rerank_depths) if rerank_depths else [None]

    for r_depth in depths:
        for ef in efs:
            is_full = r_depth == "full_beam"
            num_r: int | None = None if (r_depth is None or is_full) else int(r_depth)
            eff_ef = max(ef, num_r or 0, 10)
            r_bonus = 0.0
            lat_penalty = 0.0
            if storage_mode == "rerank":
                effective_r = ef if is_full else (num_r or 10)
                r_bonus = min(0.12, 0.03 * np.log2(max(effective_r / 10.0, 1.0)))
                lat_penalty = 0.0012 * effective_r
            recall = float(
                np.clip(0.76 + recall_offset + 0.055 * np.log10(ef / 10.0) + r_bonus, 0.1, 0.999)
            )
            mean_ms = float(latency_scale * (0.04 + 0.0022 * ef) + lat_penalty)
            p50_ms = mean_ms * 0.92
            p95_ms = mean_ms * 1.45
            p99_ms = mean_ms * 1.95
            qps = float(1000.0 / max(mean_ms, 1e-4))
            pt: dict[str, Any] = {
                "ef_search": ef,
                "effective_ef_search": eff_ef,
                "rerank_top_r": r_depth,
                "is_full_beam": is_full,
                "recall_at_k": round(recall, 6),
                "qps": round(qps, 2),
                "mean_ms": round(mean_ms, 4),
                "p50_ms": round(p50_ms, 4),
                "p95_ms": round(p95_ms, 4),
                "p99_ms": round(p99_ms, 4),
            }
            sweep.append(pt)

    payload: dict[str, Any] = {
        "phase": 6,
        "arm_id": arm_id,
        "quantizer_kind": quantizer_kind,
        "storage_mode": storage_mode,
        "build_mode": build_mode,
        "m": m,
        "ef_construction": ef_construction,
        "scale": scale,
        "dim": 128,
        "k": 10,
        "num_queries": 1000,
        "reps": 3,
        "git_commit": "abc1234",
        "git_dirty": False,
        "environment": {
            "os": "windows",
            "arch": "x86_64",
            "cpu_model": "Synthetic Test CPU",
            "processor_affinity_mask": "0x0004",
        },
        "dataset": {
            "source_description": "Synthetic SIFT",
            "scale_vectors": scale,
            "num_query": 1000,
            "dim": 128,
            "k": 10,
        },
        "steady_state_bytes_per_vector": steady_bv,
        "build_peak_rss_mb": build_rss_mb,
        "query_serving_peak_rss_mb": query_rss_mb,
        "sweep": sweep,
    }
    return payload


def _assert_valid_charts_created(output_dir: Path) -> None:
    for stem in EXPECTED_PHASE6_CHART_STEMS:
        png = output_dir / f"{stem}.png"
        svg = output_dir / f"{stem}.svg"
        assert png.is_file(), f"Missing PNG chart: {png}"
        assert svg.is_file(), f"Missing SVG chart: {svg}"
        assert png.stat().st_size > 1000, f"Empty/tiny PNG chart: {png}"
        assert svg.stat().st_size > 1000, f"Empty/tiny SVG chart: {svg}"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG header in {png}"
        svg_text = svg.read_text(encoding="utf-8")
        assert "<svg" in svg_text and "</svg>" in svg_text, f"Invalid SVG content in {svg}"


def test_plot_phase6_end_to_end_synthetic_full_matrix(tmp_path: Path) -> None:
    """Run scripts/plot_phase6.py end-to-end on synthetic SQ8 + PQ + f32 JSON reports."""
    results_dir = tmp_path / "results"
    charts_dir = tmp_path / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    scale = 100_000

    arms = [
        _make_synthetic_arm_json(
            arm_id="f32_m16",
            quantizer_kind="f32",
            storage_mode="f32",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=660.91,
            build_rss_mb=647.3,
            query_rss_mb=647.3,
            recall_offset=0.14,
            latency_scale=1.05,
        ),
        _make_synthetic_arm_json(
            arm_id="sq8_only_f32graph",
            quantizer_kind="sq8",
            storage_mode="only",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=276.91,
            build_rss_mb=720.0,
            query_rss_mb=275.4,
            recall_offset=0.12,
            latency_scale=0.82,
        ),
        _make_synthetic_arm_json(
            arm_id="sq8_only_qbuild",
            quantizer_kind="sq8",
            storage_mode="only",
            build_mode="quantized_direct",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=276.91,
            build_rss_mb=280.1,
            query_rss_mb=275.4,
            recall_offset=0.11,
            latency_scale=0.84,
        ),
        _make_synthetic_arm_json(
            arm_id="sq8_rerank_f32graph",
            quantizer_kind="sq8",
            storage_mode="rerank",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=788.91,
            build_rss_mb=760.0,
            query_rss_mb=752.0,
            recall_offset=0.10,
            latency_scale=0.86,
            rerank_depths=[20, 30, 50, 100, "full_beam"],
        ),
        _make_synthetic_arm_json(
            arm_id="pq16_only_f32graph",
            quantizer_kind="pq16",
            storage_mode="only",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=165.04,
            build_rss_mb=680.0,
            query_rss_mb=168.5,
            recall_offset=-0.06,
            latency_scale=0.65,
        ),
        _make_synthetic_arm_json(
            arm_id="pq32_only_f32graph",
            quantizer_kind="pq32",
            storage_mode="only",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=181.04,
            build_rss_mb=690.0,
            query_rss_mb=184.2,
            recall_offset=0.02,
            latency_scale=0.72,
        ),
        _make_synthetic_arm_json(
            arm_id="pq16_rerank_f32graph",
            quantizer_kind="pq16",
            storage_mode="rerank",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=677.04,
            build_rss_mb=710.0,
            query_rss_mb=650.0,
            recall_offset=-0.06,
            latency_scale=0.68,
            rerank_depths=[30, 50, 100, 200, "full_beam"],
        ),
        _make_synthetic_arm_json(
            arm_id="pq32_rerank_f32graph",
            quantizer_kind="pq32",
            storage_mode="rerank",
            build_mode="f32_graph",
            m=16,
            ef_construction=100,
            scale=scale,
            steady_bv=693.04,
            build_rss_mb=720.0,
            query_rss_mb=666.0,
            recall_offset=0.02,
            latency_scale=0.74,
            rerank_depths=[30, 50, 100, 200, "full_beam"],
        ),
    ]

    for arm in arms:
        (results_dir / f"phase6_{arm['arm_id']}.json").write_text(
            json.dumps(arm, indent=2),
            encoding="utf-8",
        )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/plot_phase6.py",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(charts_dir),
            "--scale",
            str(scale),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    _assert_valid_charts_created(charts_dir)


def test_plot_phase6_partial_subset_and_phase4_baseline_merge(tmp_path: Path) -> None:
    """Verify partial subset run (SQ8-only without PQ/rerank/query RSS) + Phase 4 merge."""
    results_dir = tmp_path / "partial"
    phase4_dir = tmp_path / "phase4"
    charts_dir = tmp_path / "charts_partial"
    results_dir.mkdir(parents=True, exist_ok=True)
    phase4_dir.mkdir(parents=True, exist_ok=True)

    partial_arm = _make_synthetic_arm_json(
        arm_id="sq8_only_partial",
        quantizer_kind="sq8",
        storage_mode="only",
        build_mode="quantized_direct",
        m=16,
        ef_construction=100,
        scale=10_000,
        steady_bv=274.5,
        build_rss_mb=32.0,
        query_rss_mb=None,
        recall_offset=0.10,
        latency_scale=0.80,
    )
    (results_dir / "arm_sq8_only.json").write_text(json.dumps(partial_arm), encoding="utf-8")

    # Phase 4-style baseline report using `scale_vectors`, `hnsw_config`, `ef_search_sweep`
    phase4_report = {
        "phase": 4,
        "git_commit": "cd48e7f",
        "git_dirty": False,
        "scale_vectors": 10_000,
        "sweep_runs": 3,
        "environment": {
            "os": "windows",
            "cpu_model": "Synthetic Test CPU",
            "processor_affinity_mask": "0x0004",
        },
        "dataset": {
            "source_description": "Synthetic SIFT",
            "scale_vectors": 10_000,
            "num_query": 100,
            "dim": 128,
            "k": 10,
        },
        "hnsw_config": {"m": 16, "ef_construction": 100},
        "index_build": {"build_time_s": 1.5},
        "memory_profile": {
            "peak_rss_mb": 48.0,
            "graph_only_bytes_per_vector": 144.0,
            "bytes_per_vector": 148.0,
        },
        "ef_search_sweep": [
            {
                "ef_search": 20,
                "recall_at_k": 0.90,
                "qps": 12000.0,
                "mean_ms": 0.08,
                "p50_ms": 0.07,
                "p95_ms": 0.12,
                "p99_ms": 0.16,
            },
            {
                "ef_search": 100,
                "recall_at_k": 0.985,
                "qps": 4500.0,
                "mean_ms": 0.22,
                "p50_ms": 0.20,
                "p95_ms": 0.33,
                "p99_ms": 0.48,
            },
        ],
    }
    (phase4_dir / "scale_10000_m16_efc100.json").write_text(
        json.dumps(phase4_report), encoding="utf-8"
    )

    # Omit `scale` to verify auto-discovery works cleanly
    created = generate_all_phase6_charts(
        results_dir=results_dir,
        output_dir=charts_dir,
        scale=None,
        phase4_dir=phase4_dir,
    )
    assert len(created) == 8
    _assert_valid_charts_created(charts_dir)


def test_plot_phase6_deterministic_svg_and_multiscale_error(tmp_path: Path) -> None:
    """Verify deterministic SVG output across repeated runs and ValueError on ambiguous scales."""
    results_dir = tmp_path / "multi"
    out_a = tmp_path / "charts_a"
    out_b = tmp_path / "charts_b"
    results_dir.mkdir(parents=True, exist_ok=True)

    arm_10k = _make_synthetic_arm_json(
        arm_id="sq8_only_10k",
        quantizer_kind="sq8",
        storage_mode="only",
        build_mode="f32_graph",
        m=16,
        ef_construction=100,
        scale=10_000,
        steady_bv=274.5,
        build_rss_mb=32.0,
        query_rss_mb=24.0,
        recall_offset=0.10,
        latency_scale=0.8,
    )
    (results_dir / "phase6_n10000_sq8_only.json").write_text(
        json.dumps(arm_10k), encoding="utf-8"
    )

    generate_all_phase6_charts(results_dir=results_dir, output_dir=out_a, scale=10_000)
    generate_all_phase6_charts(results_dir=results_dir, output_dir=out_b, scale=10_000)
    for stem in EXPECTED_PHASE6_CHART_STEMS:
        assert (out_a / f"{stem}.svg").read_bytes() == (out_b / f"{stem}.svg").read_bytes()

    # Add a second scale -> scale=None must raise ValueError cleanly
    arm_100k = _make_synthetic_arm_json(
        arm_id="sq8_only_100k",
        quantizer_kind="sq8",
        storage_mode="only",
        build_mode="f32_graph",
        m=16,
        ef_construction=100,
        scale=100_000,
        steady_bv=276.9,
        build_rss_mb=280.0,
        query_rss_mb=275.0,
        recall_offset=0.10,
        latency_scale=0.9,
    )
    (results_dir / "phase6_n100000_sq8_only.json").write_text(
        json.dumps(arm_100k), encoding="utf-8"
    )
    try:
        generate_all_phase6_charts(results_dir=results_dir, output_dir=out_a, scale=None)
        raise AssertionError("Expected ValueError when multiple scales exist and scale=None")
    except ValueError:
        pass
