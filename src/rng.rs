//! Deterministic, platform-independent seeded PRNG (`SplitMix64` + `Xoshiro256**`).
//!
//! Ensures bit-exact reproducible vector generation and HNSW layer sampling across
//! operating systems, architectures, and compiler versions without external crate drift.

use std::f64::consts::TAU;

/// Deterministic 256-bit `Xoshiro256**` pseudo-random number generator initialized via `SplitMix64`.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl DeterministicRng {
    /// Initializes a deterministic generator from any 64-bit `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_splitmix = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        Self {
            state: [
                next_splitmix(),
                next_splitmix(),
                next_splitmix(),
                next_splitmix(),
            ],
            spare_normal: None,
        }
    }

    /// Generates the next uniformly distributed 64-bit unsigned integer (`Xoshiro256**`).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Generates a uniform `f64` strictly inside the open interval `(0.0, 1.0)`.
    ///
    /// Never returns `0.0` or `1.0`, making `-ln(u)` safe across all $2^{64}$ states
    /// for both Box-Muller Gaussians and HNSW exponential layer sampling.
    #[inline]
    pub fn next_f64_open01(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        let bits53 = self.next_u64() >> 11;
        (bits53 as f64 + 0.5) * SCALE
    }

    /// Generates an unbiased uniform integer in `0..upper_exclusive` via rejection sampling.
    ///
    /// # Panics
    /// Panics if `upper_exclusive == 0`.
    #[inline]
    pub fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        assert!(upper_exclusive > 0, "upper_exclusive must be > 0");
        let bound = upper_exclusive as u64;
        if bound.is_power_of_two() {
            return (self.next_u64() & (bound - 1)) as usize;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return (r % bound) as usize;
            }
        }
    }

    /// Samples a standard normal $\mathcal{N}(0, 1)$ variate via Box-Muller transform.
    #[inline]
    pub fn next_standard_normal_f64(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        self.spare_normal = Some(radius * theta.sin());
        radius * theta.cos()
    }

    /// Samples a normal $\mathcal{N}(\mu, \sigma^2)$ variate as `f32`.
    #[inline]
    pub fn next_normal_f32(&mut self, mean: f64, std_dev: f64) -> f32 {
        (mean + std_dev * self.next_standard_normal_f64()) as f32
    }
}
