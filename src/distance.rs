//! Vector distance kernels with SIMD-friendly 8-way unrolled pairwise accumulation.

use std::fmt;
use std::str::FromStr;

use crate::error::{Result, VectorSearchError};

/// Supported vector similarity/distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// Squared Euclidean distance `sum_i (a_i - b_i)^2` (default for SIFT1M & `faiss.IndexFlatL2`).
    #[default]
    L2Squared,
    /// True Euclidean distance `sqrt(sum_i (a_i - b_i)^2)`.
    Euclidean,
    /// Inner product `sum_i a_i * b_i` (higher dot product = closer neighbor).
    InnerProduct,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::L2Squared => write!(f, "l2"),
            Self::Euclidean => write!(f, "euclidean"),
            Self::InnerProduct => write!(f, "ip"),
        }
    }
}

impl FromStr for Metric {
    type Err = VectorSearchError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "l2" | "l2_squared" | "sqeuclidean" => Ok(Self::L2Squared),
            "euclidean" | "l2_sqrt" => Ok(Self::Euclidean),
            "ip" | "inner_product" | "dot" => Ok(Self::InnerProduct),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Unsupported metric '{other}'. Expected 'l2', 'euclidean', or 'ip'."
            ))),
        }
    }
}

/// Computes exact squared Euclidean distance $\sum_{j=0}^{d-1} (a_j - b_j)^2$.
///
/// Uses 8 independent accumulators with pairwise tree reduction so LLVM
/// auto-vectorizes across SIMD lanes while minimizing floating-point rounding drift.
/// Clamped to `>= 0.0` with canonical `+0.0`.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "l2_squared dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        let d4 = ca[4] - cb[4];
        let d5 = ca[5] - cb[5];
        let d6 = ca[6] - cb[6];
        let d7 = ca[7] - cb[7];

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
    for (&va, &vb) in rem_a.iter().zip(rem_b.iter()) {
        let d = va - vb;
        sum += d * d;
    }

    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

/// Computes exact squared Euclidean distance $\sum_{j=0}^{d-1} (a_j - b_j)^2$ in `f64` precision.
///
/// Accumulating in `f64` guarantees zero intermediate overflow and zero underflow across the
/// entire finite IEEE-754 `f32` domain (`[-3.40e38, +3.40e38]`, including subnormals down to
/// `1.40e-45`), making it suitable for authoritative ground-truth ranking.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn l2_squared_f64(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "l2_squared_f64 dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );

    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut s4 = 0.0_f64;
    let mut s5 = 0.0_f64;
    let mut s6 = 0.0_f64;
    let mut s7 = 0.0_f64;

    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        let d0 = f64::from(ca[0]) - f64::from(cb[0]);
        let d1 = f64::from(ca[1]) - f64::from(cb[1]);
        let d2 = f64::from(ca[2]) - f64::from(cb[2]);
        let d3 = f64::from(ca[3]) - f64::from(cb[3]);
        let d4 = f64::from(ca[4]) - f64::from(cb[4]);
        let d5 = f64::from(ca[5]) - f64::from(cb[5]);
        let d6 = f64::from(ca[6]) - f64::from(cb[6]);
        let d7 = f64::from(ca[7]) - f64::from(cb[7]);

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
    for (&va, &vb) in rem_a.iter().zip(rem_b.iter()) {
        let d = f64::from(va) - f64::from(vb);
        sum += d * d;
    }

    if sum <= 0.0 {
        0.0
    } else {
        sum
    }
}

/// Computes true Euclidean distance $\sqrt{\sum_{j=0}^{d-1} (a_j - b_j)^2}$.
///
/// Evaluates the sum of squares in `f64` prior to taking the square root so finite `f32`
/// inputs never overflow (`|a_j - b_j| > 1.84e19`) or underflow (`|a_j - b_j| < 1.18e-22`)
/// the intermediate squared sum.
#[inline]
#[must_use]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let sq = l2_squared_f64(a, b).sqrt() as f32;
    if sq <= 0.0 {
        0.0
    } else {
        sq
    }
}

/// Computes inner product $\sum_{j=0}^{d-1} a_j \cdot b_j$.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "inner_product dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );

    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;

    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        s0 += ca[0] * cb[0];
        s1 += ca[1] * cb[1];
        s2 += ca[2] * cb[2];
        s3 += ca[3] * cb[3];
        s4 += ca[4] * cb[4];
        s5 += ca[5] * cb[5];
        s6 += ca[6] * cb[6];
        s7 += ca[7] * cb[7];
    }

    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (&va, &vb) in rem_a.iter().zip(rem_b.iter()) {
        sum += va * vb;
    }

    if sum == 0.0 {
        0.0
    } else {
        sum
    }
}

/// Computes exact inner product $\sum_{j=0}^{d-1} a_j \cdot b_j$ in `f64` precision.
///
/// Guarantees zero intermediate overflow for any finite `f32` vectors.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn inner_product_f64(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "inner_product_f64 dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );

    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let mut s4 = 0.0_f64;
    let mut s5 = 0.0_f64;
    let mut s6 = 0.0_f64;
    let mut s7 = 0.0_f64;

    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        s0 += f64::from(ca[0]) * f64::from(cb[0]);
        s1 += f64::from(ca[1]) * f64::from(cb[1]);
        s2 += f64::from(ca[2]) * f64::from(cb[2]);
        s3 += f64::from(ca[3]) * f64::from(cb[3]);
        s4 += f64::from(ca[4]) * f64::from(cb[4]);
        s5 += f64::from(ca[5]) * f64::from(cb[5]);
        s6 += f64::from(ca[6]) * f64::from(cb[6]);
        s7 += f64::from(ca[7]) * f64::from(cb[7]);
    }

    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (&va, &vb) in rem_a.iter().zip(rem_b.iter()) {
        sum += f64::from(va) * f64::from(vb);
    }

    if sum == 0.0 {
        0.0
    } else {
        sum
    }
}
