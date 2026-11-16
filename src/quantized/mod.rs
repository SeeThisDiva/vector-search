//! Quantized vector search engines (`SQ8`, and future `PQ`) over compact lifetime-free HNSW graphs.
//!
//! Strictly isolated from [`crate::hnsw::HnswIndex`] so all Phase 1–5 `f32` graph construction
//! and search paths remain untouched (`PRESERVING` non-regression guarantee).

pub mod graph;
pub mod sq8;

use std::io::{Read, Write};

use crate::distance::{inner_product, l2_squared, Metric};
use crate::error::{Result, VectorSearchError};

pub use graph::{
    results_to_knn_search_result, CompactHnswGraph, Neighbor, QuantizedBuildMode,
    QuantizedBuildTimings, QuantizedDistanceSource, QuantizedMemoryProfile, QuantizedSearchScratch,
    QuantizedStorageMode, SearchMetrics, TimedQuantizedSearch,
};
pub use sq8::{
    inner_product_sq8, inner_product_sq8_symmetric, l2_squared_sq8_expanded,
    l2_squared_sq8_shifted, l2_squared_sq8_symmetric, l2_squared_sq8_symmetric_from_scales,
    HnswSq8Index, Sq8KernelKind, Sq8Quantizer, Sq8QueryState, Sq8Vectors,
};

/// Computes exact canonical minimization sort key (`L2^2` or `-dot`) between two `f32` slices.
#[inline]
pub(crate) fn compute_exact_sort_key(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2Squared | Metric::Euclidean => l2_squared(a, b),
        Metric::InnerProduct => {
            let neg_dot = -inner_product(a, b);
            if neg_dot == 0.0 {
                0.0
            } else {
                neg_dot
            }
        }
    }
}

/// Converts an internal minimization sort key (`L2^2` or `-dot`) into the public `Metric` output.
#[inline]
pub(crate) fn format_metric_output_distance(metric: Metric, sort_key: f32) -> f32 {
    match metric {
        Metric::L2Squared => {
            if sort_key <= 0.0 {
                0.0
            } else {
                sort_key
            }
        }
        Metric::Euclidean => {
            let d = sort_key.max(0.0).sqrt();
            if d <= 0.0 {
                0.0
            } else {
                d
            }
        }
        Metric::InnerProduct => {
            let score = -sort_key;
            if score == 0.0 {
                0.0
            } else {
                score
            }
        }
    }
}

#[inline]
pub(crate) fn metric_to_u8(metric: Metric) -> u8 {
    match metric {
        Metric::L2Squared => 0,
        Metric::Euclidean => 1,
        Metric::InnerProduct => 2,
    }
}

#[inline]
pub(crate) fn metric_from_u8(tag: u8) -> Result<Metric> {
    match tag {
        0 => Ok(Metric::L2Squared),
        1 => Ok(Metric::Euclidean),
        2 => Ok(Metric::InnerProduct),
        _ => Err(VectorSearchError::CorruptData(format!(
            "Invalid serialized Metric tag: {tag}"
        ))),
    }
}

#[inline]
pub(crate) fn write_u8<W: Write>(w: &mut W, val: u8) -> Result<()> {
    w.write_all(&[val])?;
    Ok(())
}

#[inline]
pub(crate) fn write_u32<W: Write>(w: &mut W, val: u32) -> Result<()> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

#[inline]
pub(crate) fn write_u64<W: Write>(w: &mut W, val: u64) -> Result<()> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

#[inline]
pub(crate) fn write_f64<W: Write>(w: &mut W, val: f64) -> Result<()> {
    w.write_all(&val.to_bits().to_le_bytes())?;
    Ok(())
}

pub(crate) fn write_u32_slice<W: Write>(w: &mut W, slice: &[u32]) -> Result<()> {
    write_u64(w, slice.len() as u64)?;
    for &v in slice {
        write_u32(w, v)?;
    }
    Ok(())
}

pub(crate) fn write_f32_slice<W: Write>(w: &mut W, slice: &[f32]) -> Result<()> {
    write_u64(w, slice.len() as u64)?;
    for &v in slice {
        w.write_all(&v.to_bits().to_le_bytes())?;
    }
    Ok(())
}

#[inline]
pub(crate) fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0_u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
pub(crate) fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0_u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[inline]
pub(crate) fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0_u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[inline]
pub(crate) fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0_u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_bits(u64::from_le_bytes(buf)))
}

pub(crate) fn read_u32_vec<R: Read>(r: &mut R, max_len: usize) -> Result<Vec<u32>> {
    let len_u64 = read_u64(r)?;
    let len = usize::try_from(len_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!(
            "Serialized u32 vector length {len_u64} overflows usize"
        ))
    })?;
    if len > max_len {
        return Err(VectorSearchError::CorruptData(format!(
            "Serialized u32 vector length {len} exceeds maximum allowed {max_len}"
        )));
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_u32(r)?);
    }
    Ok(out)
}

pub(crate) fn read_f32_vec<R: Read>(r: &mut R, expected_len: usize) -> Result<Vec<f32>> {
    let len_u64 = read_u64(r)?;
    let len = usize::try_from(len_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!(
            "Serialized f32 vector length {len_u64} overflows usize"
        ))
    })?;
    if len != expected_len {
        return Err(VectorSearchError::CorruptData(format!(
            "Serialized f32 vector length mismatch: expected {expected_len}, got {len}"
        )));
    }
    let mut out = Vec::with_capacity(len);
    let mut buf = [0_u8; 4];
    for _ in 0..len {
        r.read_exact(&mut buf)?;
        out.push(f32::from_bits(u32::from_le_bytes(buf)));
    }
    Ok(out)
}
