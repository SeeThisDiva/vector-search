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

/// Computes internal 8-way unrolled sum with pairwise tree reduction, preserving exact
/// lane accumulator order `s0..s7` across SIMD lanes and remainder elements.
macro_rules! unrolled_8way_sum {
    ($a:expr, $b:expr, $acc_ty:ty, $zero:expr, $term:expr) => {{
        let mut s0: $acc_ty = $zero;
        let mut s1: $acc_ty = $zero;
        let mut s2: $acc_ty = $zero;
        let mut s3: $acc_ty = $zero;
        let mut s4: $acc_ty = $zero;
        let mut s5: $acc_ty = $zero;
        let mut s6: $acc_ty = $zero;
        let mut s7: $acc_ty = $zero;

        let (chunks_a, rem_a) = $a.as_chunks::<8>();
        let (chunks_b, rem_b) = $b.as_chunks::<8>();

        for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
            s0 += $term(ca[0], cb[0]);
            s1 += $term(ca[1], cb[1]);
            s2 += $term(ca[2], cb[2]);
            s3 += $term(ca[3], cb[3]);
            s4 += $term(ca[4], cb[4]);
            s5 += $term(ca[5], cb[5]);
            s6 += $term(ca[6], cb[6]);
            s7 += $term(ca[7], cb[7]);
        }

        let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        for (&va, &vb) in rem_a.iter().zip(rem_b.iter()) {
            sum += $term(va, vb);
        }
        sum
    }};
}

#[inline(always)]
fn sq_diff_f32(x: f32, y: f32) -> f32 {
    let d = x - y;
    d * d
}

#[inline(always)]
fn sq_diff_f64(x: f32, y: f32) -> f64 {
    let d = f64::from(x) - f64::from(y);
    d * d
}

#[inline(always)]
fn mul_f32(x: f32, y: f32) -> f32 {
    x * y
}

#[inline(always)]
fn mul_f64(x: f32, y: f32) -> f64 {
    f64::from(x) * f64::from(y)
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

    let sum = unrolled_8way_sum!(a, b, f32, 0.0_f32, sq_diff_f32);
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

    let sum = unrolled_8way_sum!(a, b, f64, 0.0_f64, sq_diff_f64);
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

    let sum = unrolled_8way_sum!(a, b, f32, 0.0_f32, mul_f32);
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

    let sum = unrolled_8way_sum!(a, b, f64, 0.0_f64, mul_f64);
    if sum == 0.0 {
        0.0
    } else {
        sum
    }
}

/// Computes the canonical minimization sort key (`L2^2` or `-dot`) for `metric`.
#[inline]
pub(crate) fn compute_metric_sort_key(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
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

/// Converts an internal minimization sort key (`L2^2` or `-dot`) into the public `Metric` distance/score.
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
