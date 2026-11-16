//! 8-bit Scalar Quantization (`SQ8`) calibrator, safe SIMD-friendly kernels, two-pass
//! streaming `.fvecs` encoder, and lifetime-free [`HnswSq8Index`] (`T4.2` & `T4.3`).
//!
//! All distance kernels are 100% safe Rust (`#![forbid(unsafe_code)]`, `std`-only) using
//! `as_chunks::<8>()` 8-way unrolling and zero per-dimension float divisions in hot loops.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::matrix::RowMatrix;

use super::graph::{
    CompactHnswGraph, Neighbor, QuantizedBuildMode, QuantizedBuildTimings, QuantizedDistanceSource,
    QuantizedMemoryProfile, QuantizedSearchScratch, QuantizedStorageMode, TimedQuantizedSearch,
};
use super::{
    format_metric_output_distance, metric_from_u8, metric_to_u8, read_f32_vec, read_f64, read_u64,
    read_u8, write_f32_slice, write_f64, write_u64, write_u8,
};

const SQ8_VECTORS_MAGIC: &[u8; 8] = b"SQ8VEC01";
const SQ8_SNAPSHOT_MAGIC: &[u8; 8] = b"HSQ8SN01";
const IO_BUF_CAPACITY: usize = 256 * 1024;
const STREAMING_CHUNK_ROWS: usize = 4_096;

/// Asymmetric `L2^2` SQ8 kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sq8KernelKind {
    /// Kernel A (Default): Query-shifted direct asymmetric `sum_j ((q_j - min_j) - s_j * c_ij)^2`.
    /// Zero extra per-vector memory and zero catastrophic cancellation.
    #[default]
    QueryShifted,
    /// Kernel B (Ablation): Analytical norm-expanded `||q'||^2 - 2 * <q' * s, c_i> + ||s * c_i||^2`.
    /// Requires `+4 B/vector` for `base_norms_sq` and is susceptible to `f32` cancellation on
    /// large-offset clustered vectors.
    NormExpanded,
}

impl Sq8KernelKind {
    /// Canonical lowercase identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QueryShifted => "query_shifted",
            Self::NormExpanded => "norm_expanded",
        }
    }

    #[inline]
    fn to_u8(self) -> u8 {
        match self {
            Self::QueryShifted => 0,
            Self::NormExpanded => 1,
        }
    }

    #[inline]
    fn from_u8(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::QueryShifted),
            1 => Ok(Self::NormExpanded),
            _ => Err(VectorSearchError::CorruptData(format!(
                "Invalid Sq8KernelKind tag: {tag}"
            ))),
        }
    }
}

/// Reusable per-query precomputed state for [`Sq8Vectors`].
#[derive(Debug, Clone, Default)]
pub struct Sq8QueryState {
    /// Precomputed per-dimension query buffer (`q_j - min_j` for `QueryShifted`,
    /// `(q_j - min_j) * s_j` for `NormExpanded`, or `q_j * s_j` for `InnerProduct`).
    pub q_shifted: Vec<f32>,
    /// Precomputed scalar term (`||q - min||^2` for `NormExpanded`, `<q, min>` for `InnerProduct`, `0.0` otherwise).
    pub scalar: f32,
}

impl Sq8QueryState {
    /// Allocated capacity of the precomputed query vector buffer (`q_shifted`).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.q_shifted.capacity()
    }
}

/// Per-dimension min/max SQ8 quantizer storing precomputed `f32` scale constants.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8Quantizer {
    dim: usize,
    mins: Vec<f32>,
    scales: Vec<f32>,
    inv_scales: Vec<f32>,
    scales_sq: Vec<f32>,
    has_non_finite_scales_sq: bool,
}

impl Sq8Quantizer {
    /// Calibrates per-dimension `[min_j, max_j]` ranges from an in-memory `RowMatrix<f32>`.
    pub fn calibrate(matrix: &RowMatrix<f32>) -> Result<Self> {
        validate_base_matrix(matrix)?;
        let dim = matrix.cols();
        let mut mins = matrix.row(0).to_vec();
        let mut maxs = mins.clone();

        for row in matrix.iter_rows().skip(1) {
            update_min_max_row(&mut mins, &mut maxs, row);
        }

        Self::from_min_max(mins, maxs).inspect(|q| {
            debug_assert_eq!(q.dim(), dim);
        })
    }

    /// Builds calibration scale tables in `f64` from precomputed per-dimension `mins` and `maxs`.
    pub fn from_min_max(mins: Vec<f32>, maxs: Vec<f32>) -> Result<Self> {
        let dim = mins.len();
        if dim == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "Sq8Quantizer dimension must be > 0".to_string(),
            ));
        }
        if maxs.len() != dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: dim,
                actual: maxs.len(),
            });
        }

        let mut scales = Vec::with_capacity(dim);
        let mut inv_scales = Vec::with_capacity(dim);
        let mut scales_sq = Vec::with_capacity(dim);
        let mut has_non_finite_scales_sq = false;

        for j in 0..dim {
            let min_j = mins[j];
            let max_j = maxs[j];
            if !min_j.is_finite() || !max_j.is_finite() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Non-finite min/max bound at dim {j}: min={min_j}, max={max_j}"
                )));
            }
            if max_j < min_j {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Invalid calibration range at dim {j}: max ({max_j}) < min ({min_j})"
                )));
            }

            let span = f64::from(max_j) - f64::from(min_j);
            let (scale, inv_scale, scale_sq) = if span > 0.0 {
                let s = (span / 255.0) as f32;
                if s == 0.0 {
                    // Subnormal span < ~1.8e-43 underflows f32 scale; treat safely as constant dim.
                    (0.0_f32, 0.0_f32, 0.0_f32)
                } else {
                    let inv = (255.0 / span) as f32;
                    let s2 = s * s;
                    if !s2.is_finite() {
                        has_non_finite_scales_sq = true;
                    }
                    (s, inv, s2)
                }
            } else {
                (0.0_f32, 0.0_f32, 0.0_f32)
            };

            scales.push(scale);
            inv_scales.push(inv_scale);
            scales_sq.push(scale_sq);
        }

        Ok(Self {
            dim,
            mins,
            scales,
            inv_scales,
            scales_sq,
            has_non_finite_scales_sq,
        })
    }

    /// Vector dimensionality (`D`).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Per-dimension minimum values (`min_j`).
    #[inline]
    #[must_use]
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Per-dimension quantization step sizes (`(max_j - min_j) / 255.0`).
    #[inline]
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Per-dimension inverse scales (`255.0 / (max_j - min_j)`).
    #[inline]
    #[must_use]
    pub fn inv_scales(&self) -> &[f32] {
        &self.inv_scales
    }

    /// Precomputed squared scales (`scale_j^2`) for fast symmetric `u8`-vs-`u8` kernels.
    #[inline]
    #[must_use]
    pub fn scales_sq(&self) -> &[f32] {
        &self.scales_sq
    }

    /// Returns `true` if any `scales_sq[j]` overflowed `f32` (extreme dynamic range).
    #[inline]
    #[must_use]
    pub fn has_non_finite_scales_sq(&self) -> bool {
        self.has_non_finite_scales_sq
    }

    /// Total heap payload bytes of the calibration tables (`4 * dim * sizeof(f32)`).
    #[inline]
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.mins.capacity()
            + self.scales.capacity()
            + self.inv_scales.capacity()
            + self.scales_sq.capacity())
            * std::mem::size_of::<f32>()
    }

    /// Encodes a single `f32` vector into `u8` SQ8 codes after checking dimension and finiteness.
    pub fn encode_row(&self, row: &[f32], out: &mut [u8]) -> Result<()> {
        if row.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: row.len(),
            });
        }
        if out.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: out.len(),
            });
        }
        if let Some(pos) = row.iter().position(|v| !v.is_finite()) {
            return Err(VectorSearchError::InvalidArgument(format!(
                "row contains non-finite value (NaN/Inf) at col {pos}"
            )));
        }
        self.encode_row_unchecked(row, out);
        Ok(())
    }

    #[inline]
    fn encode_row_unchecked(&self, row: &[f32], out: &mut [u8]) {
        for j in 0..self.dim {
            let scale = self.scales[j];
            if scale == 0.0 {
                out[j] = 0;
            } else {
                let min_j = self.mins[j];
                let inv_scale = self.inv_scales[j];
                // Compute (v - min) in f64 so extreme ranges like [-3e38, +3e38] never
                // overflow to +inf prior to scaling.
                let diff = f64::from(row[j]) - f64::from(min_j);
                let scaled = if inv_scale.is_finite() {
                    diff * f64::from(inv_scale)
                } else {
                    diff / f64::from(scale)
                };
                out[j] = scaled.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    /// Reconstructs approximate `f32` coordinates `min_j + scale_j * code_j` into `out`.
    pub fn dequantize_row(&self, code: &[u8], out: &mut [f32]) -> Result<()> {
        if code.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: code.len(),
            });
        }
        if out.len() != self.dim {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim,
                actual: out.len(),
            });
        }
        for j in 0..self.dim {
            out[j] = self.mins[j] + self.scales[j] * f32::from(code[j]);
        }
        Ok(())
    }

    /// Serializes quantizer tables to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        write_u64(writer, self.dim as u64)?;
        write_f32_slice(writer, &self.mins)?;
        write_f32_slice(writer, &self.scales)?;
        write_f32_slice(writer, &self.inv_scales)?;
        write_f32_slice(writer, &self.scales_sq)?;
        Ok(())
    }

    /// Deserializes and validates quantizer tables from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let dim = usize::try_from(read_u64(reader)?).map_err(|_| {
            VectorSearchError::CorruptData("Sq8Quantizer dim overflows usize".to_string())
        })?;
        if dim == 0 {
            return Err(VectorSearchError::CorruptData(
                "Sq8Quantizer dim cannot be 0".to_string(),
            ));
        }

        let mins = read_f32_vec(reader, dim)?;
        let scales = read_f32_vec(reader, dim)?;
        let inv_scales = read_f32_vec(reader, dim)?;
        let scales_sq = read_f32_vec(reader, dim)?;

        let mut has_non_finite_scales_sq = false;
        for j in 0..dim {
            if !mins[j].is_finite() {
                return Err(VectorSearchError::CorruptData(format!(
                    "Non-finite mins[{j}] in Sq8Quantizer snapshot"
                )));
            }
            if !scales[j].is_finite() || scales[j] < 0.0 {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid scales[{j}]={} in Sq8Quantizer snapshot",
                    scales[j]
                )));
            }
            if inv_scales[j].is_nan() || inv_scales[j] < 0.0 {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid inv_scales[{j}]={} in Sq8Quantizer snapshot",
                    inv_scales[j]
                )));
            }
            if scales_sq[j].is_nan() || scales_sq[j] < 0.0 {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid scales_sq[{j}]={} in Sq8Quantizer snapshot",
                    scales_sq[j]
                )));
            }
            if !scales_sq[j].is_finite() {
                has_non_finite_scales_sq = true;
            }
        }

        Ok(Self {
            dim,
            mins,
            scales,
            inv_scales,
            scales_sq,
            has_non_finite_scales_sq,
        })
    }
}

/// Safe 8-way unrolled Kernel A (`QueryShifted`):
/// `sum_j (q_shifted[j] - scales[j] * f32::from(code[j]))^2`.
///
/// Performs zero per-dimension float divisions and zero catastrophic cancellation.
///
/// # Panics
/// Panics if `q_shifted.len() != scales.len()` or `q_shifted.len() != code.len()`.
#[inline]
#[must_use]
pub fn l2_squared_sq8_shifted(q_shifted: &[f32], scales: &[f32], code: &[u8]) -> f32 {
    assert_eq!(q_shifted.len(), scales.len());
    assert_eq!(q_shifted.len(), code.len());

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (q_chunks, q_rem) = q_shifted.as_chunks::<8>();
    let (sc_chunks, sc_rem) = scales.as_chunks::<8>();
    let (c_chunks, c_rem) = code.as_chunks::<8>();

    for ((cq, cs), cc) in q_chunks.iter().zip(sc_chunks.iter()).zip(c_chunks.iter()) {
        let d0 = cq[0] - cs[0] * f32::from(cc[0]);
        let d1 = cq[1] - cs[1] * f32::from(cc[1]);
        let d2 = cq[2] - cs[2] * f32::from(cc[2]);
        let d3 = cq[3] - cs[3] * f32::from(cc[3]);
        let d4 = cq[4] - cs[4] * f32::from(cc[4]);
        let d5 = cq[5] - cs[5] * f32::from(cc[5]);
        let d6 = cq[6] - cs[6] * f32::from(cc[6]);
        let d7 = cq[7] - cs[7] * f32::from(cc[7]);

        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        s4 += d4 * d4;
        s5 += d5 * d5;
        s6 += d6 * d6;
        s7 += d7 * d7;
    }

    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for ((&vq, &vs), &vc) in q_rem.iter().zip(sc_rem.iter()).zip(c_rem.iter()) {
        let d = vq - vs * f32::from(vc);
        sum += d * d;
    }

    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

/// Safe 8-way unrolled Kernel B (`NormExpanded`, `T4.2` ablation):
/// `q_norm_sq + base_norm_sq - 2.0 * sum_j q_scaled[j] * f32::from(code[j])`
/// where `q_scaled[j] = (q[j] - min[j]) * scales[j]`, `q_norm_sq = ||q - min||^2`,
/// and `base_norm_sq = ||scales * code||^2`.
///
/// # Panics
/// Panics if `q_scaled.len() != code.len()`.
#[inline]
#[must_use]
pub fn l2_squared_sq8_expanded(
    q_norm_sq: f32,
    q_scaled: &[f32],
    code: &[u8],
    base_norm_sq: f32,
) -> f32 {
    assert_eq!(q_scaled.len(), code.len());

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (q_chunks, q_rem) = q_scaled.as_chunks::<8>();
    let (c_chunks, c_rem) = code.as_chunks::<8>();

    for (cq, cc) in q_chunks.iter().zip(c_chunks.iter()) {
        s0 += cq[0] * f32::from(cc[0]);
        s1 += cq[1] * f32::from(cc[1]);
        s2 += cq[2] * f32::from(cc[2]);
        s3 += cq[3] * f32::from(cc[3]);
        s4 += cq[4] * f32::from(cc[4]);
        s5 += cq[5] * f32::from(cc[5]);
        s6 += cq[6] * f32::from(cc[6]);
        s7 += cq[7] * f32::from(cc[7]);
    }

    let mut dot = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (&vq, &vc) in q_rem.iter().zip(c_rem.iter()) {
        dot += vq * f32::from(vc);
    }

    let dist = (q_norm_sq + base_norm_sq) - 2.0 * dot;
    if dist <= 0.0 {
        0.0
    } else {
        dist
    }
}

/// Safe 8-way unrolled symmetric `u8`-vs-`u8` `L2^2` kernel using precomputed `scales_sq`:
/// `sum_j scales_sq[j] * (f32::from(code_a[j]) - f32::from(code_b[j]))^2`.
///
/// Callers with extreme float ranges where `scales_sq` contains non-finite values should use
/// [`l2_squared_sq8_symmetric_from_scales`] to avoid `inf * 0.0 = NaN`.
///
/// # Panics
/// Panics if slice lengths differ.
#[inline]
#[must_use]
pub fn l2_squared_sq8_symmetric(scales_sq: &[f32], code_a: &[u8], code_b: &[u8]) -> f32 {
    assert_eq!(scales_sq.len(), code_a.len());
    assert_eq!(scales_sq.len(), code_b.len());

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (ss_chunks, ss_rem) = scales_sq.as_chunks::<8>();
    let (ca_chunks, ca_rem) = code_a.as_chunks::<8>();
    let (cb_chunks, cb_rem) = code_b.as_chunks::<8>();

    for ((css, ca), cb) in ss_chunks.iter().zip(ca_chunks.iter()).zip(cb_chunks.iter()) {
        let d0 = f32::from(ca[0]) - f32::from(cb[0]);
        let d1 = f32::from(ca[1]) - f32::from(cb[1]);
        let d2 = f32::from(ca[2]) - f32::from(cb[2]);
        let d3 = f32::from(ca[3]) - f32::from(cb[3]);
        let d4 = f32::from(ca[4]) - f32::from(cb[4]);
        let d5 = f32::from(ca[5]) - f32::from(cb[5]);
        let d6 = f32::from(ca[6]) - f32::from(cb[6]);
        let d7 = f32::from(ca[7]) - f32::from(cb[7]);

        s0 += css[0] * (d0 * d0);
        s1 += css[1] * (d1 * d1);
        s2 += css[2] * (d2 * d2);
        s3 += css[3] * (d3 * d3);
        s4 += css[4] * (d4 * d4);
        s5 += css[5] * (d5 * d5);
        s6 += css[6] * (d6 * d6);
        s7 += css[7] * (d7 * d7);
    }

    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for ((&vss, &va), &vb) in ss_rem.iter().zip(ca_rem.iter()).zip(cb_rem.iter()) {
        let d = f32::from(va) - f32::from(vb);
        sum += vss * (d * d);
    }

    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

/// Overflow-safe 8-way unrolled symmetric `u8`-vs-`u8` `L2^2` kernel computing
/// `(scales[j] * (f32::from(code_a[j]) - f32::from(code_b[j])))^2`.
///
/// Prevents `inf * 0.0 = NaN` on equal coordinates when `scales[j]^2 > f32::MAX`.
///
/// # Panics
/// Panics if slice lengths differ.
#[inline]
#[must_use]
pub fn l2_squared_sq8_symmetric_from_scales(scales: &[f32], code_a: &[u8], code_b: &[u8]) -> f32 {
    assert_eq!(scales.len(), code_a.len());
    assert_eq!(scales.len(), code_b.len());

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (sc_chunks, sc_rem) = scales.as_chunks::<8>();
    let (ca_chunks, ca_rem) = code_a.as_chunks::<8>();
    let (cb_chunks, cb_rem) = code_b.as_chunks::<8>();

    for ((csc, ca), cb) in sc_chunks.iter().zip(ca_chunks.iter()).zip(cb_chunks.iter()) {
        let d0 = csc[0] * (f32::from(ca[0]) - f32::from(cb[0]));
        let d1 = csc[1] * (f32::from(ca[1]) - f32::from(cb[1]));
        let d2 = csc[2] * (f32::from(ca[2]) - f32::from(cb[2]));
        let d3 = csc[3] * (f32::from(ca[3]) - f32::from(cb[3]));
        let d4 = csc[4] * (f32::from(ca[4]) - f32::from(cb[4]));
        let d5 = csc[5] * (f32::from(ca[5]) - f32::from(cb[5]));
        let d6 = csc[6] * (f32::from(ca[6]) - f32::from(cb[6]));
        let d7 = csc[7] * (f32::from(ca[7]) - f32::from(cb[7]));

        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        s4 += d4 * d4;
        s5 += d5 * d5;
        s6 += d6 * d6;
        s7 += d7 * d7;
    }

    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for ((&vs, &va), &vb) in sc_rem.iter().zip(ca_rem.iter()).zip(cb_rem.iter()) {
        let d = vs * (f32::from(va) - f32::from(vb));
        sum += d * d;
    }

    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

/// Safe 8-way unrolled asymmetric inner product `q_min_dot + sum_j q_scaled[j] * f32::from(code[j])`
/// where `q_min_dot = sum_j q[j] * min[j]` and `q_scaled[j] = q[j] * scales[j]`.
///
/// # Panics
/// Panics if `q_scaled.len() != code.len()`.
#[inline]
#[must_use]
pub fn inner_product_sq8(q_min_dot: f32, q_scaled: &[f32], code: &[u8]) -> f32 {
    assert_eq!(q_scaled.len(), code.len());

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (q_chunks, q_rem) = q_scaled.as_chunks::<8>();
    let (c_chunks, c_rem) = code.as_chunks::<8>();

    for (cq, cc) in q_chunks.iter().zip(c_chunks.iter()) {
        s0 += cq[0] * f32::from(cc[0]);
        s1 += cq[1] * f32::from(cc[1]);
        s2 += cq[2] * f32::from(cc[2]);
        s3 += cq[3] * f32::from(cc[3]);
        s4 += cq[4] * f32::from(cc[4]);
        s5 += cq[5] * f32::from(cc[5]);
        s6 += cq[6] * f32::from(cc[6]);
        s7 += cq[7] * f32::from(cc[7]);
    }

    let mut dot = q_min_dot + (((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)));
    for (&vq, &vc) in q_rem.iter().zip(c_rem.iter()) {
        dot += vq * f32::from(vc);
    }

    if dot == 0.0 {
        0.0
    } else {
        dot
    }
}

/// Safe 8-way unrolled symmetric inner product over dequantized SQ8 codes:
/// `sum_j (mins[j] + scales[j] * code_a[j]) * (mins[j] + scales[j] * code_b[j])`.
///
/// # Panics
/// Panics if slice lengths differ.
#[inline]
#[must_use]
pub fn inner_product_sq8_symmetric(
    mins: &[f32],
    scales: &[f32],
    code_a: &[u8],
    code_b: &[u8],
) -> f32 {
    assert_eq!(mins.len(), scales.len());
    assert_eq!(mins.len(), code_a.len());
    assert_eq!(mins.len(), code_b.len());

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (m_chunks, m_rem) = mins.as_chunks::<8>();
    let (sc_chunks, sc_rem) = scales.as_chunks::<8>();
    let (ca_chunks, ca_rem) = code_a.as_chunks::<8>();
    let (cb_chunks, cb_rem) = code_b.as_chunks::<8>();

    for (((cm, cs), ca), cb) in m_chunks
        .iter()
        .zip(sc_chunks.iter())
        .zip(ca_chunks.iter())
        .zip(cb_chunks.iter())
    {
        s0 += (cm[0] + cs[0] * f32::from(ca[0])) * (cm[0] + cs[0] * f32::from(cb[0]));
        s1 += (cm[1] + cs[1] * f32::from(ca[1])) * (cm[1] + cs[1] * f32::from(cb[1]));
        s2 += (cm[2] + cs[2] * f32::from(ca[2])) * (cm[2] + cs[2] * f32::from(cb[2]));
        s3 += (cm[3] + cs[3] * f32::from(ca[3])) * (cm[3] + cs[3] * f32::from(cb[3]));
        s4 += (cm[4] + cs[4] * f32::from(ca[4])) * (cm[4] + cs[4] * f32::from(cb[4]));
        s5 += (cm[5] + cs[5] * f32::from(ca[5])) * (cm[5] + cs[5] * f32::from(cb[5]));
        s6 += (cm[6] + cs[6] * f32::from(ca[6])) * (cm[6] + cs[6] * f32::from(cb[6]));
        s7 += (cm[7] + cs[7] * f32::from(ca[7])) * (cm[7] + cs[7] * f32::from(cb[7]));
    }

    let mut dot = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (((&vm, &vs), &va), &vb) in m_rem
        .iter()
        .zip(sc_rem.iter())
        .zip(ca_rem.iter())
        .zip(cb_rem.iter())
    {
        dot += (vm + vs * f32::from(va)) * (vm + vs * f32::from(vb));
    }

    if dot == 0.0 {
        0.0
    } else {
        dot
    }
}

/// Contiguous `u8` SQ8 base vector store implementing [`QuantizedDistanceSource`].
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8Vectors {
    metric: Metric,
    kernel_kind: Sq8KernelKind,
    quantizer: Sq8Quantizer,
    num_vectors: usize,
    codes: Vec<u8>,
    base_norms_sq: Option<Vec<f32>>,
}

impl Sq8Vectors {
    /// Calibrates and encodes an in-memory `RowMatrix<f32>` using the default [`Sq8KernelKind::QueryShifted`].
    pub fn from_matrix(matrix: &RowMatrix<f32>, metric: Metric) -> Result<Self> {
        Self::from_matrix_with_kernel(matrix, metric, Sq8KernelKind::QueryShifted)
    }

    /// Calibrates and encodes an in-memory `RowMatrix<f32>` with an explicit [`Sq8KernelKind`].
    pub fn from_matrix_with_kernel(
        matrix: &RowMatrix<f32>,
        metric: Metric,
        kernel_kind: Sq8KernelKind,
    ) -> Result<Self> {
        let (vectors, _, _) = Self::from_matrix_timed(matrix, metric, kernel_kind)?;
        Ok(vectors)
    }

    fn from_matrix_timed(
        matrix: &RowMatrix<f32>,
        metric: Metric,
        kernel_kind: Sq8KernelKind,
    ) -> Result<(Self, f64, f64)> {
        let t_cal = Instant::now();
        let quantizer = Sq8Quantizer::calibrate(matrix)?;
        let cal_s = t_cal.elapsed().as_secs_f64();

        let t_enc = Instant::now();
        let num_vectors = matrix.rows();
        let dim = matrix.cols();
        let total_codes = num_vectors.checked_mul(dim).ok_or_else(|| {
            VectorSearchError::InvalidArgument("SQ8 code buffer size overflows usize".to_string())
        })?;
        let mut codes = vec![0_u8; total_codes];
        let mut base_norms_sq =
            (kernel_kind == Sq8KernelKind::NormExpanded).then(|| Vec::with_capacity(num_vectors));

        for (i, row) in matrix.iter_rows().enumerate() {
            let code_slice = &mut codes[i * dim..(i + 1) * dim];
            quantizer.encode_row_unchecked(row, code_slice);
            if let Some(norms) = base_norms_sq.as_mut() {
                norms.push(compute_code_norm_sq(quantizer.scales(), code_slice));
            }
        }
        let enc_s = t_enc.elapsed().as_secs_f64();

        Ok((
            Self {
                metric,
                kernel_kind,
                quantizer,
                num_vectors,
                codes,
                base_norms_sq,
            },
            cal_s,
            enc_s,
        ))
    }

    /// Two-pass streaming `.fvecs` calibration and quantization using a `<= 4,096`-row chunk buffer.
    ///
    /// Never materializes `RowMatrix<f32>` in memory, producing codes and calibration tables
    /// bit-identical to [`Self::from_matrix`].
    pub fn from_fvecs_streaming<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        Self::from_fvecs_streaming_with_kernel(path, metric, Sq8KernelKind::QueryShifted, max_rows)
    }

    /// Two-pass streaming `.fvecs` calibration and quantization with an explicit [`Sq8KernelKind`].
    pub fn from_fvecs_streaming_with_kernel<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        kernel_kind: Sq8KernelKind,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        let (vectors, _, _) =
            Self::from_fvecs_streaming_timed(path, metric, kernel_kind, max_rows)?;
        Ok(vectors)
    }

    fn from_fvecs_streaming_timed<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        kernel_kind: Sq8KernelKind,
        max_rows: Option<usize>,
    ) -> Result<(Self, f64, f64)> {
        if max_rows == Some(0) {
            return Err(VectorSearchError::InvalidArgument(
                "max_rows must be positive (> 0) when specified".to_string(),
            ));
        }

        let path_ref = path.as_ref();
        let meta = fs::metadata(path_ref)?;
        let file_size = meta.len();
        if file_size == 0 {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Cannot calibrate SQ8 on empty .fvecs file: {}",
                path_ref.display()
            )));
        }
        if file_size < 4 {
            return Err(VectorSearchError::CorruptData(format!(
                "Corrupt .fvecs file (size {file_size} < 4 bytes): {}",
                path_ref.display()
            )));
        }

        let file = File::open(path_ref)?;
        let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);

        let mut header = [0_u8; 4];
        reader.read_exact(&mut header)?;
        let first_dim_i32 = i32::from_le_bytes(header);
        if first_dim_i32 <= 0 {
            return Err(VectorSearchError::CorruptData(format!(
                "Invalid dimension {first_dim_i32} in .fvecs file: {}",
                path_ref.display()
            )));
        }
        let dim = first_dim_i32 as usize;
        let record_bytes = 4_u64
            .checked_add((dim as u64).checked_mul(4).ok_or_else(|| {
                VectorSearchError::CorruptData("Dimension byte size overflow".to_string())
            })?)
            .ok_or_else(|| {
                VectorSearchError::CorruptData("Record byte size overflow".to_string())
            })?;

        if file_size % record_bytes != 0 {
            return Err(VectorSearchError::CorruptData(format!(
                "File size {file_size} is not a multiple of record size {record_bytes} (dim={dim}) in {}",
                path_ref.display()
            )));
        }

        let total_vecs_u64 = file_size / record_bytes;
        let total_vecs = usize::try_from(total_vecs_u64).map_err(|_| {
            VectorSearchError::CorruptData(format!("Vector count {total_vecs_u64} exceeds usize"))
        })?;
        let num_vecs = match max_rows {
            Some(limit) => total_vecs.min(limit),
            None => total_vecs,
        };
        if num_vecs > u32::MAX as usize {
            return Err(VectorSearchError::InvalidArgument(format!(
                "base_vectors row count {num_vecs} exceeds u32::MAX"
            )));
        }

        // Small reusable chunk buffer (4,096 rows = 2 MiB at D=128) so RowMatrix<f32> is never materialized.
        let chunk_rows = STREAMING_CHUNK_ROWS
            .min((16 * 1024 * 1024 / (dim * 4)).max(1))
            .min(num_vecs);
        let mut chunk_f32 = vec![0.0_f32; chunk_rows * dim];
        let mut row_bytes = vec![0_u8; dim * 4];

        // Pass 1: Calibrate mins and maxs while checking finiteness and per-record headers.
        let t_cal = Instant::now();
        reader.seek(SeekFrom::Start(0))?;
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        stream_fvecs_chunks(
            &mut reader,
            path_ref,
            first_dim_i32,
            dim,
            num_vecs,
            chunk_rows,
            &mut row_bytes,
            &mut chunk_f32,
            |row_offset, rows_in_chunk, chunk| {
                for r in 0..rows_in_chunk {
                    let row = &chunk[r * dim..(r + 1) * dim];
                    if row_offset + r == 0 {
                        mins.extend_from_slice(row);
                        maxs.extend_from_slice(row);
                    } else {
                        update_min_max_row(&mut mins, &mut maxs, row);
                    }
                }
            },
        )?;
        let quantizer = Sq8Quantizer::from_min_max(mins, maxs)?;
        let cal_s = t_cal.elapsed().as_secs_f64();

        // Pass 2: Encode quantized u8 codes directly into compact output Vec<u8>.
        let t_enc = Instant::now();
        let total_codes = num_vecs.checked_mul(dim).ok_or_else(|| {
            VectorSearchError::InvalidArgument("SQ8 code buffer size overflows usize".to_string())
        })?;
        let mut codes = vec![0_u8; total_codes];
        let mut base_norms_sq =
            (kernel_kind == Sq8KernelKind::NormExpanded).then(|| Vec::with_capacity(num_vecs));

        reader.seek(SeekFrom::Start(0))?;
        stream_fvecs_chunks(
            &mut reader,
            path_ref,
            first_dim_i32,
            dim,
            num_vecs,
            chunk_rows,
            &mut row_bytes,
            &mut chunk_f32,
            |row_offset, rows_in_chunk, chunk| {
                for r in 0..rows_in_chunk {
                    let global_id = row_offset + r;
                    let row = &chunk[r * dim..(r + 1) * dim];
                    let code_slice = &mut codes[global_id * dim..(global_id + 1) * dim];
                    quantizer.encode_row_unchecked(row, code_slice);
                    if let Some(norms) = base_norms_sq.as_mut() {
                        norms.push(compute_code_norm_sq(quantizer.scales(), code_slice));
                    }
                }
            },
        )?;
        let enc_s = t_enc.elapsed().as_secs_f64();

        Ok((
            Self {
                metric,
                kernel_kind,
                quantizer,
                num_vectors: num_vecs,
                codes,
                base_norms_sq,
            },
            cal_s,
            enc_s,
        ))
    }

    /// Reference to the per-dimension [`Sq8Quantizer`].
    #[inline]
    #[must_use]
    pub fn quantizer(&self) -> &Sq8Quantizer {
        &self.quantizer
    }

    /// Configured asymmetric `L2^2` kernel variant.
    #[inline]
    #[must_use]
    pub fn kernel_kind(&self) -> Sq8KernelKind {
        self.kernel_kind
    }

    /// Returns the `u8` quantized code slice for vector `id`.
    ///
    /// # Panics
    /// Panics if `id >= self.len()`.
    #[inline]
    #[must_use]
    pub fn code(&self, id: usize) -> &[u8] {
        let dim = self.quantizer.dim();
        let start = id * dim;
        &self.codes[start..start + dim]
    }

    /// Contiguous flat `[N * dim]` `u8` code buffer.
    #[inline]
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Optional precomputed base code norms (`||s * c_i||^2`), populated only for [`Sq8KernelKind::NormExpanded`].
    #[inline]
    #[must_use]
    pub fn base_norms_sq(&self) -> Option<&[f32]> {
        self.base_norms_sq.as_deref()
    }

    /// Heap bytes allocated for quantized `u8` codes.
    #[inline]
    #[must_use]
    pub fn codes_memory_bytes(&self) -> usize {
        self.codes.capacity()
    }

    /// Auxiliary heap bytes (`base_norms_sq` when `NormExpanded`, else `0`).
    #[inline]
    #[must_use]
    pub fn aux_memory_bytes(&self) -> usize {
        self.base_norms_sq
            .as_ref()
            .map_or(0, |v| v.capacity() * std::mem::size_of::<f32>())
    }

    /// Serializes [`Sq8Vectors`] to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(SQ8_VECTORS_MAGIC)?;
        write_u8(writer, metric_to_u8(self.metric))?;
        write_u8(writer, self.kernel_kind.to_u8())?;
        write_u64(writer, self.num_vectors as u64)?;
        self.quantizer.write_to(writer)?;
        write_u64(writer, self.codes.len() as u64)?;
        writer.write_all(&self.codes)?;
        match &self.base_norms_sq {
            Some(norms) => {
                write_u8(writer, 1)?;
                write_f32_slice(writer, norms)?;
            }
            None => {
                write_u8(writer, 0)?;
            }
        }
        Ok(())
    }

    /// Deserializes and validates [`Sq8Vectors`] from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != SQ8_VECTORS_MAGIC {
            return Err(VectorSearchError::CorruptData(
                "Invalid Sq8Vectors snapshot magic header".to_string(),
            ));
        }
        let metric = metric_from_u8(read_u8(reader)?)?;
        let kernel_kind = Sq8KernelKind::from_u8(read_u8(reader)?)?;
        let num_vectors = usize::try_from(read_u64(reader)?).map_err(|_| {
            VectorSearchError::CorruptData("Sq8Vectors num_vectors overflows usize".to_string())
        })?;
        if num_vectors == 0 || num_vectors > u32::MAX as usize {
            return Err(VectorSearchError::CorruptData(format!(
                "Invalid Sq8Vectors num_vectors: {num_vectors}"
            )));
        }

        let quantizer = Sq8Quantizer::read_from(reader)?;
        let expected_codes_len = num_vectors.checked_mul(quantizer.dim()).ok_or_else(|| {
            VectorSearchError::CorruptData("Sq8Vectors codes length overflows usize".to_string())
        })?;
        let codes_len = usize::try_from(read_u64(reader)?).map_err(|_| {
            VectorSearchError::CorruptData("Sq8Vectors codes_len overflows usize".to_string())
        })?;
        if codes_len != expected_codes_len {
            return Err(VectorSearchError::CorruptData(format!(
                "Sq8Vectors codes length mismatch: expected {expected_codes_len}, got {codes_len}"
            )));
        }
        let mut codes = vec![0_u8; codes_len];
        reader.read_exact(&mut codes)?;

        let has_norms = read_u8(reader)?;
        let base_norms_sq = match has_norms {
            0 => {
                if kernel_kind == Sq8KernelKind::NormExpanded {
                    return Err(VectorSearchError::CorruptData(
                        "NormExpanded Sq8Vectors snapshot missing base_norms_sq".to_string(),
                    ));
                }
                None
            }
            1 => Some(read_f32_vec(reader, num_vectors)?),
            other => {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid has_norms tag in Sq8Vectors snapshot: {other}"
                )));
            }
        };

        Ok(Self {
            metric,
            kernel_kind,
            quantizer,
            num_vectors,
            codes,
            base_norms_sq,
        })
    }
}

impl QuantizedDistanceSource for Sq8Vectors {
    type QueryState = Sq8QueryState;

    #[inline]
    fn len(&self) -> usize {
        self.num_vectors
    }

    #[inline]
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    #[inline]
    fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn prepare_query(&self, query: &[f32], state: &mut Self::QueryState) {
        let dim = self.quantizer.dim();
        assert_eq!(query.len(), dim, "SQ8 query dimension mismatch");
        if state.q_shifted.len() != dim {
            state.q_shifted.resize(dim, 0.0);
        }

        let mins = self.quantizer.mins();
        let scales = self.quantizer.scales();

        match self.metric {
            Metric::L2Squared | Metric::Euclidean => match self.kernel_kind {
                Sq8KernelKind::QueryShifted => {
                    state.scalar = 0.0;
                    for j in 0..dim {
                        state.q_shifted[j] = query[j] - mins[j];
                    }
                }
                Sq8KernelKind::NormExpanded => {
                    let mut norm_sq = 0.0_f64;
                    for j in 0..dim {
                        let q_diff = query[j] - mins[j];
                        norm_sq += f64::from(q_diff) * f64::from(q_diff);
                        state.q_shifted[j] = q_diff * scales[j];
                    }
                    state.scalar = norm_sq as f32;
                }
            },
            Metric::InnerProduct => {
                let mut min_dot = 0.0_f64;
                for j in 0..dim {
                    min_dot += f64::from(query[j]) * f64::from(mins[j]);
                    state.q_shifted[j] = query[j] * scales[j];
                }
                state.scalar = min_dot as f32;
            }
        }
    }

    #[inline]
    fn query_distance(&self, state: &Self::QueryState, node_id: usize) -> f32 {
        let code = self.code(node_id);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => match self.kernel_kind {
                Sq8KernelKind::QueryShifted => {
                    l2_squared_sq8_shifted(&state.q_shifted, self.quantizer.scales(), code)
                }
                Sq8KernelKind::NormExpanded => {
                    let base_norm_sq = self
                        .base_norms_sq
                        .as_ref()
                        .expect("base_norms_sq present for NormExpanded")[node_id];
                    l2_squared_sq8_expanded(state.scalar, &state.q_shifted, code, base_norm_sq)
                }
            },
            Metric::InnerProduct => {
                let neg_dot = -inner_product_sq8(state.scalar, &state.q_shifted, code);
                if neg_dot == 0.0 {
                    0.0
                } else {
                    neg_dot
                }
            }
        }
    }

    #[inline]
    fn node_distance(&self, id_a: usize, id_b: usize) -> f32 {
        let code_a = self.code(id_a);
        let code_b = self.code(id_b);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => {
                if id_a == id_b {
                    return 0.0;
                }
                if self.quantizer.has_non_finite_scales_sq() {
                    l2_squared_sq8_symmetric_from_scales(self.quantizer.scales(), code_a, code_b)
                } else {
                    l2_squared_sq8_symmetric(self.quantizer.scales_sq(), code_a, code_b)
                }
            }
            Metric::InnerProduct => {
                let neg_dot = -inner_product_sq8_symmetric(
                    self.quantizer.mins(),
                    self.quantizer.scales(),
                    code_a,
                    code_b,
                );
                if neg_dot == 0.0 {
                    0.0
                } else {
                    neg_dot
                }
            }
        }
    }

    #[inline]
    fn format_output_distance(&self, sort_key: f32) -> f32 {
        format_metric_output_distance(self.metric, sort_key)
    }

    #[inline]
    fn vector_memory_bytes(&self) -> usize {
        self.codes_memory_bytes() + self.quantizer.memory_bytes() + self.aux_memory_bytes()
    }
}

/// Lifetime-free SQ8 HNSW index separating memory-first (`QuantizedOnly`) from optional
/// exact top-`R` reranking (`WithExactRerank`).
#[derive(Debug, Clone)]
pub struct HnswSq8Index {
    graph: CompactHnswGraph,
    vectors: Sq8Vectors,
    raw_base: Option<Arc<RowMatrix<f32>>>,
    build_mode: QuantizedBuildMode,
    storage_mode: QuantizedStorageMode,
    build_timings: QuantizedBuildTimings,
}

impl HnswSq8Index {
    /// Builds an SQ8 HNSW index using the default [`Sq8KernelKind::QueryShifted`] kernel.
    ///
    /// When `storage_mode == QuantizedStorageMode::QuantizedOnly`, no raw `f32` vectors are
    /// cloned or retained on `self`.
    pub fn build(
        base: &RowMatrix<f32>,
        metric: Metric,
        config: HnswConfig,
        build_mode: QuantizedBuildMode,
        storage_mode: QuantizedStorageMode,
    ) -> Result<Self> {
        Self::build_with_kernel(
            base,
            metric,
            config,
            build_mode,
            storage_mode,
            Sq8KernelKind::QueryShifted,
        )
    }

    /// Builds an SQ8 HNSW index with an explicit [`Sq8KernelKind`].
    pub fn build_with_kernel(
        base: &RowMatrix<f32>,
        metric: Metric,
        config: HnswConfig,
        build_mode: QuantizedBuildMode,
        storage_mode: QuantizedStorageMode,
        kernel_kind: Sq8KernelKind,
    ) -> Result<Self> {
        let total_start = Instant::now();
        config.validate()?;
        validate_base_matrix(base)?;

        let (graph, vectors, cal_s, enc_s, graph_s) = match build_mode {
            QuantizedBuildMode::Float32GraphThenQuantize => {
                let t_graph = Instant::now();
                let f32_index = HnswIndex::build(base, metric, config)?;
                let graph = CompactHnswGraph::from_hnsw_index(&f32_index);
                drop(f32_index);
                let graph_s = t_graph.elapsed().as_secs_f64();

                let (vectors, cal_s, enc_s) =
                    Sq8Vectors::from_matrix_timed(base, metric, kernel_kind)?;
                (graph, vectors, cal_s, enc_s, graph_s)
            }
            QuantizedBuildMode::QuantizedDirect => {
                let (vectors, cal_s, enc_s) =
                    Sq8Vectors::from_matrix_timed(base, metric, kernel_kind)?;
                let t_graph = Instant::now();
                let graph = CompactHnswGraph::build_with_source(&vectors, config)?;
                let graph_s = t_graph.elapsed().as_secs_f64();
                (graph, vectors, cal_s, enc_s, graph_s)
            }
        };

        let raw_base = match storage_mode {
            QuantizedStorageMode::QuantizedOnly => None,
            QuantizedStorageMode::WithExactRerank => Some(Arc::new(base.clone())),
        };

        Ok(Self {
            graph,
            vectors,
            raw_base,
            build_mode,
            storage_mode,
            build_timings: QuantizedBuildTimings {
                calibration_or_train_s: cal_s,
                encode_s: enc_s,
                graph_build_s: graph_s,
                total_s: total_start.elapsed().as_secs_f64(),
            },
        })
    }

    /// Builds an SQ8 HNSW index sharing an already-allocated `Arc<RowMatrix<f32>>` without
    /// duplicating raw `f32` storage when `storage_mode == QuantizedStorageMode::WithExactRerank`.
    pub fn build_with_shared_raw(
        base: Arc<RowMatrix<f32>>,
        metric: Metric,
        config: HnswConfig,
        build_mode: QuantizedBuildMode,
        storage_mode: QuantizedStorageMode,
        kernel_kind: Sq8KernelKind,
    ) -> Result<Self> {
        let mut idx = Self::build_with_kernel(
            &base,
            metric,
            config,
            build_mode,
            QuantizedStorageMode::QuantizedOnly,
            kernel_kind,
        )?;
        if storage_mode == QuantizedStorageMode::WithExactRerank {
            idx.raw_base = Some(base);
            idx.storage_mode = QuantizedStorageMode::WithExactRerank;
        }
        Ok(idx)
    }

    /// Builds a memory-first [`HnswSq8Index`] directly from an `.fvecs` file using two-pass
    /// streaming calibration and `QuantizedDirect` graph construction.
    ///
    /// Never materializes `RowMatrix<f32>` in memory, enabling low build-time and query-time RSS.
    pub fn build_streaming_from_fvecs<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        config: HnswConfig,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        Self::build_streaming_from_fvecs_with_kernel(
            path,
            metric,
            config,
            Sq8KernelKind::QueryShifted,
            max_rows,
        )
    }

    /// Two-pass streaming `.fvecs` index builder with explicit [`Sq8KernelKind`].
    pub fn build_streaming_from_fvecs_with_kernel<P: AsRef<Path>>(
        path: P,
        metric: Metric,
        config: HnswConfig,
        kernel_kind: Sq8KernelKind,
        max_rows: Option<usize>,
    ) -> Result<Self> {
        let total_start = Instant::now();
        config.validate()?;

        let (vectors, cal_s, enc_s) =
            Sq8Vectors::from_fvecs_streaming_timed(path, metric, kernel_kind, max_rows)?;
        let t_graph = Instant::now();
        let graph = CompactHnswGraph::build_with_source(&vectors, config)?;
        let graph_s = t_graph.elapsed().as_secs_f64();

        Ok(Self {
            graph,
            vectors,
            raw_base: None,
            build_mode: QuantizedBuildMode::QuantizedDirect,
            storage_mode: QuantizedStorageMode::QuantizedOnly,
            build_timings: QuantizedBuildTimings {
                calibration_or_train_s: cal_s,
                encode_s: enc_s,
                graph_build_s: graph_s,
                total_s: total_start.elapsed().as_secs_f64(),
            },
        })
    }

    /// Drops retained raw `f32` vectors (if any) and transitions storage mode to [`QuantizedStorageMode::QuantizedOnly`].
    #[must_use]
    pub fn into_quantized_only(mut self) -> Self {
        self.raw_base = None;
        self.storage_mode = QuantizedStorageMode::QuantizedOnly;
        self
    }

    /// Attaches shared raw `f32` vectors after validating shape and finiteness, enabling exact rerank.
    pub fn with_raw_base(mut self, raw_base: Arc<RowMatrix<f32>>) -> Result<Self> {
        validate_base_matrix(&raw_base)?;
        if raw_base.rows() != self.num_vectors() {
            return Err(VectorSearchError::InvalidArgument(format!(
                "raw_base row count ({}) does not match index vector count ({})",
                raw_base.rows(),
                self.num_vectors()
            )));
        }
        if raw_base.cols() != self.dim() {
            return Err(VectorSearchError::DimensionMismatch {
                expected: self.dim(),
                actual: raw_base.cols(),
            });
        }
        self.raw_base = Some(raw_base);
        self.storage_mode = QuantizedStorageMode::WithExactRerank;
        Ok(self)
    }

    /// Writes a deterministic quantized-only binary snapshot (`graph + Sq8Vectors`) to `path`.
    pub fn write_snapshot<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path_ref)?;
        let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);
        self.write_to(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Loads a quantized-only [`HnswSq8Index`] snapshot from `path` (holding zero raw `f32` vectors).
    pub fn read_snapshot<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);
        Self::read_from(&mut reader)
    }

    /// Writes a deterministic quantized-only snapshot to a binary stream.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(SQ8_SNAPSHOT_MAGIC)?;
        write_u8(
            writer,
            match self.build_mode {
                QuantizedBuildMode::Float32GraphThenQuantize => 0,
                QuantizedBuildMode::QuantizedDirect => 1,
            },
        )?;
        write_f64(writer, self.build_timings.calibration_or_train_s)?;
        write_f64(writer, self.build_timings.encode_s)?;
        write_f64(writer, self.build_timings.graph_build_s)?;
        write_f64(writer, self.build_timings.total_s)?;
        self.graph.write_to(writer)?;
        self.vectors.write_to(writer)?;
        Ok(())
    }

    /// Reads and validates a quantized-only snapshot from a binary stream.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != SQ8_SNAPSHOT_MAGIC {
            return Err(VectorSearchError::CorruptData(
                "Invalid HnswSq8Index snapshot magic header".to_string(),
            ));
        }
        let build_mode = match read_u8(reader)? {
            0 => QuantizedBuildMode::Float32GraphThenQuantize,
            1 => QuantizedBuildMode::QuantizedDirect,
            other => {
                return Err(VectorSearchError::CorruptData(format!(
                    "Invalid QuantizedBuildMode tag in snapshot: {other}"
                )));
            }
        };
        let calibration_or_train_s = read_f64(reader)?;
        let encode_s = read_f64(reader)?;
        let graph_build_s = read_f64(reader)?;
        let total_s = read_f64(reader)?;
        if !calibration_or_train_s.is_finite()
            || !encode_s.is_finite()
            || !graph_build_s.is_finite()
            || !total_s.is_finite()
        {
            return Err(VectorSearchError::CorruptData(
                "Non-finite build timings in HnswSq8Index snapshot".to_string(),
            ));
        }
        let graph = CompactHnswGraph::read_from(reader)?;
        let vectors = Sq8Vectors::read_from(reader)?;
        if graph.num_vectors() != vectors.len() {
            return Err(VectorSearchError::CorruptData(format!(
                "Snapshot node count mismatch: graph={}, vectors={}",
                graph.num_vectors(),
                vectors.len()
            )));
        }
        if graph.metric() != vectors.metric() {
            return Err(VectorSearchError::CorruptData(format!(
                "Snapshot metric mismatch: graph={}, vectors={}",
                graph.metric(),
                vectors.metric()
            )));
        }

        Ok(Self {
            graph,
            vectors,
            raw_base: None,
            build_mode,
            storage_mode: QuantizedStorageMode::QuantizedOnly,
            build_timings: QuantizedBuildTimings {
                calibration_or_train_s,
                encode_s,
                graph_build_s,
                total_s,
            },
        })
    }

    /// Reference to the underlying [`CompactHnswGraph`].
    #[inline]
    #[must_use]
    pub fn graph(&self) -> &CompactHnswGraph {
        &self.graph
    }

    /// Reference to the quantized [`Sq8Vectors`] store.
    #[inline]
    #[must_use]
    pub fn vectors(&self) -> &Sq8Vectors {
        &self.vectors
    }

    /// Reference to optional retained raw `f32` base vectors (`None` in `QuantizedOnly` mode).
    #[inline]
    #[must_use]
    pub fn raw_base(&self) -> Option<&RowMatrix<f32>> {
        self.raw_base.as_deref()
    }

    /// Graph build precision mode used to construct this index.
    #[inline]
    #[must_use]
    pub fn build_mode(&self) -> QuantizedBuildMode {
        self.build_mode
    }

    /// Current raw-vector storage mode (`QuantizedOnly` or `WithExactRerank`).
    #[inline]
    #[must_use]
    pub fn storage_mode(&self) -> QuantizedStorageMode {
        self.storage_mode
    }

    /// Phase-level build timings recorded during construction.
    #[inline]
    #[must_use]
    pub fn build_timings(&self) -> QuantizedBuildTimings {
        self.build_timings
    }

    /// Number of indexed vectors (`N`).
    #[inline]
    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.graph.num_vectors()
    }

    /// Vector dimensionality (`D`).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vectors.dim()
    }

    /// Configured distance metric.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.graph.metric()
    }

    /// Maximum layer height in the graph.
    #[inline]
    #[must_use]
    pub fn max_layer(&self) -> usize {
        self.graph.max_layer()
    }

    /// Global entry point node ID, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn entry_point(&self) -> Option<usize> {
        self.graph.entry_point()
    }

    /// Canonical 64-bit FNV-1a graph topology fingerprint matching `HnswIndex::graph_fingerprint()`.
    #[inline]
    #[must_use]
    pub fn graph_fingerprint(&self) -> u64 {
        self.graph.graph_fingerprint()
    }

    /// Steady-state payload memory profile across codes, calibration, graph, and optional raw vectors.
    #[must_use]
    pub fn memory_profile(&self) -> QuantizedMemoryProfile {
        QuantizedMemoryProfile::from_parts(
            self.graph.num_vectors(),
            self.vectors.codes_memory_bytes(),
            self.vectors.quantizer().memory_bytes(),
            self.vectors.aux_memory_bytes(),
            self.graph.node_meta_bytes(),
            self.graph.layer0_edges_bytes(),
            self.graph.upper_layers_edges_bytes(),
            self.raw_base.as_ref().map_or(0, |m| m.memory_bytes()),
        )
    }

    /// Allocates reusable search scratch sized for `max_ef`.
    #[must_use]
    pub fn new_search_scratch(&self, max_ef: usize) -> QuantizedSearchScratch<Sq8QueryState> {
        QuantizedSearchScratch::new(
            self.graph.num_vectors(),
            max_ef,
            self.graph.config().effective_m_max0(),
        )
    }

    /// Executes an untimed warmup query to populate scratch buffer capacities and CPU caches.
    pub fn warmup(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<()> {
        let _ = self.search_knn_with_scratch(query, k, ef_search, scratch)?;
        Ok(())
    }

    /// Pure quantized top-`k` search reusing `scratch`.
    pub fn search_knn_with_scratch(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<Vec<Neighbor>> {
        self.graph
            .search_knn_with_scratch(&self.vectors, query, k, ef_search, scratch)
    }

    /// Quantized search with optional exact top-`R` reranking using stored raw vectors (`WithExactRerank`).
    pub fn search_knn_rerank_with_scratch(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<Vec<Neighbor>> {
        let mut out = Vec::with_capacity(k.min(self.num_vectors()));
        self.graph.search_knn_rerank_opt_into_with_scratch(
            &self.vectors,
            self.raw_base.as_deref(),
            query,
            k,
            ef_search,
            rerank_top_r,
            scratch,
            &mut out,
        )?;
        Ok(out)
    }

    /// Quantized search with optional exact top-`R` reranking against a caller-supplied `raw_base`
    /// without storing `raw_base` inside [`HnswSq8Index`].
    pub fn search_knn_rerank_with_base_scratch(
        &self,
        raw_base: &RowMatrix<f32>,
        query: &[f32],
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<Vec<Neighbor>> {
        self.graph.search_knn_rerank_with_scratch(
            &self.vectors,
            raw_base,
            query,
            k,
            ef_search,
            rerank_top_r,
            scratch,
        )
    }

    /// Batch sequential search returning `Vec<Vec<Neighbor>>`.
    pub fn search_batch_with_scratch(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<Vec<Vec<Neighbor>>> {
        self.graph.search_batch_with_scratch(
            &self.vectors,
            self.raw_base.as_deref(),
            queries,
            k,
            ef_search,
            rerank_top_r,
            scratch,
        )
    }

    /// Timed sequential batch search returning `Vec<Vec<Neighbor>>`, per-query latencies, and [`super::SearchMetrics`].
    pub fn search_batch_timed_with_scratch(
        &self,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<TimedQuantizedSearch> {
        self.graph.search_batch_timed_with_scratch(
            &self.vectors,
            self.raw_base.as_deref(),
            queries,
            k,
            ef_search,
            rerank_top_r,
            scratch,
        )
    }

    /// Timed sequential batch search with caller-supplied `raw_base` (for reranking without owning raw vectors).
    pub fn search_batch_with_base_timed_with_scratch(
        &self,
        raw_base: &RowMatrix<f32>,
        queries: &RowMatrix<f32>,
        k: usize,
        ef_search: usize,
        rerank_top_r: Option<usize>,
        scratch: &mut QuantizedSearchScratch<Sq8QueryState>,
    ) -> Result<TimedQuantizedSearch> {
        self.graph.search_batch_timed_with_scratch(
            &self.vectors,
            Some(raw_base),
            queries,
            k,
            ef_search,
            rerank_top_r,
            scratch,
        )
    }
}

fn validate_base_matrix(base_vectors: &RowMatrix<f32>) -> Result<()> {
    if base_vectors.rows() == 0 || base_vectors.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "base_vectors cannot be empty (rows and cols must be > 0)".to_string(),
        ));
    }
    if base_vectors.rows() > u32::MAX as usize {
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors row count {} exceeds u32::MAX",
            base_vectors.rows()
        )));
    }
    if let Some(pos) = base_vectors.as_slice().iter().position(|v| !v.is_finite()) {
        let row = pos / base_vectors.cols();
        let col = pos % base_vectors.cols();
        return Err(VectorSearchError::InvalidArgument(format!(
            "base_vectors contains non-finite value (NaN/Inf) at ({row}, {col})"
        )));
    }
    Ok(())
}

#[inline]
fn update_min_max_row(mins: &mut [f32], maxs: &mut [f32], row: &[f32]) {
    for j in 0..row.len() {
        let v = row[j];
        if v < mins[j] {
            mins[j] = v;
        }
        if v > maxs[j] {
            maxs[j] = v;
        }
    }
}

#[inline]
fn compute_code_norm_sq(scales: &[f32], code: &[u8]) -> f32 {
    let mut sum = 0.0_f64;
    for (&s, &c) in scales.iter().zip(code.iter()) {
        let d = f64::from(s) * f64::from(c);
        sum += d * d;
    }
    sum as f32
}

#[allow(clippy::too_many_arguments)]
fn stream_fvecs_chunks<R: Read, F>(
    reader: &mut R,
    path_ref: &Path,
    expected_dim_i32: i32,
    dim: usize,
    num_vecs: usize,
    chunk_rows: usize,
    row_bytes: &mut [u8],
    chunk_f32: &mut [f32],
    mut on_chunk: F,
) -> Result<()>
where
    F: FnMut(usize, usize, &[f32]),
{
    let mut header = [0_u8; 4];
    let mut rows_read = 0_usize;

    while rows_read < num_vecs {
        let take = chunk_rows.min(num_vecs - rows_read);
        for r in 0..take {
            let global_row = rows_read + r;
            reader.read_exact(&mut header)?;
            let rec_dim = i32::from_le_bytes(header);
            if rec_dim != expected_dim_i32 {
                return Err(VectorSearchError::CorruptData(format!(
                    "Inconsistent .fvecs record dimension at row {global_row}: expected {dim}, got {rec_dim} in {}",
                    path_ref.display()
                )));
            }
            reader.read_exact(row_bytes)?;
            let dst = &mut chunk_f32[r * dim..(r + 1) * dim];
            for (col, (&bytes, out_val)) in row_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.iter_mut())
                .enumerate()
            {
                let v = f32::from_le_bytes(bytes);
                if !v.is_finite() {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "base_vectors contains non-finite value (NaN/Inf) at ({global_row}, {col})"
                    )));
                }
                *out_val = v;
            }
        }
        on_chunk(rows_read, take, &chunk_f32[..take * dim]);
        rows_read += take;
    }

    Ok(())
}
