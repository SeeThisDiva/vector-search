//! Filter predicates and bitset acceleration structures for Milestone M3.
//!
//! Provides:
//! - [`FilterPredicate`]: Abstraction for evaluating boolean filter constraints over vector IDs.
//! - [`DenseBitset`]: Cache-friendly, 1-bit-per-vector bitset with zero-allocation iteration.
//! - [`BitsetOneIterator`]: Zero-allocation ascending iterator over active (1) bit IDs.
//! - [`LambdaPredicate`]: Opaque closure-based predicate with estimated selectivity.
//! - Query interface structs: [`FilteredSearchParams`], [`FilteredSearchFlags`], [`FilteredSearchOutput`].

/// Interface for evaluating query filter constraints over database vector IDs.
pub trait FilterPredicate: Send + Sync {
    /// Evaluates whether the vector with 0-indexed database ID `id` satisfies the filter.
    fn matches(&self, id: u32) -> bool;

    /// Optional reference to an underlying contiguous bitset.
    /// When available, enables popcount-based candidate counting and
    /// the Strategy 3 brute-force path.
    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        None
    }

    /// Optional count of total matching candidates.
    /// Defaults to the popcount of the underlying bitset if available.
    #[inline]
    fn count_matching(&self, _total: usize) -> Option<usize> {
        self.bitset().map(|b| b.count_ones())
    }

    /// Optional caller-supplied selectivity estimate in `(0.0, 1.0]` when an exact
    /// match count is unavailable (for example, [`LambdaPredicate`]).
    #[inline]
    fn estimated_selectivity(&self) -> Option<f64> {
        None
    }
}

impl<P: FilterPredicate + ?Sized> FilterPredicate for &P {
    #[inline]
    fn matches(&self, id: u32) -> bool {
        (**self).matches(id)
    }

    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        (**self).bitset()
    }

    #[inline]
    fn count_matching(&self, total: usize) -> Option<usize> {
        (**self).count_matching(total)
    }

    #[inline]
    fn estimated_selectivity(&self) -> Option<f64> {
        (**self).estimated_selectivity()
    }
}

impl<P: FilterPredicate + ?Sized> FilterPredicate for Box<P> {
    #[inline]
    fn matches(&self, id: u32) -> bool {
        (**self).matches(id)
    }

    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        (**self).bitset()
    }

    #[inline]
    fn count_matching(&self, total: usize) -> Option<usize> {
        (**self).count_matching(total)
    }

    #[inline]
    fn estimated_selectivity(&self) -> Option<f64> {
        (**self).estimated_selectivity()
    }
}

impl<P: FilterPredicate + ?Sized> FilterPredicate for std::sync::Arc<P> {
    #[inline]
    fn matches(&self, id: u32) -> bool {
        (**self).matches(id)
    }

    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        (**self).bitset()
    }

    #[inline]
    fn count_matching(&self, total: usize) -> Option<usize> {
        (**self).count_matching(total)
    }

    #[inline]
    fn estimated_selectivity(&self) -> Option<f64> {
        (**self).estimated_selectivity()
    }
}

/// Cache-line aligned dense bitset storing 1 bit per vector ID in contiguous 64-bit words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseBitset {
    words: Vec<u64>,
    len: usize,
}

impl DenseBitset {
    /// Allocates an all-zero bitset for `len` elements.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        let num_words = len.div_ceil(64);
        Self {
            words: vec![0_u64; num_words],
            len,
        }
    }

    /// Allocates an all-one bitset for `len` elements.
    ///
    /// Any unused padding bits in the final 64-bit word are guaranteed to be zeroed.
    #[must_use]
    pub fn ones(len: usize) -> Self {
        let num_words = len.div_ceil(64);
        if num_words == 0 {
            return Self {
                words: Vec::new(),
                len: 0,
            };
        }
        let mut words = vec![!0_u64; num_words];
        let rem = len % 64;
        if rem != 0 {
            words[num_words - 1] = (1_u64 << rem) - 1;
        }
        Self { words, len }
    }

    /// Constructs a bitset from existing 64-bit words and exact length.
    ///
    /// Any unused padding bits beyond `len` in the final word are automatically zeroed.
    ///
    /// # Panics
    /// Panics if `words.len() != (len + 63) / 64`.
    #[must_use]
    pub fn from_words(mut words: Vec<u64>, len: usize) -> Self {
        let expected_words = len.div_ceil(64);
        assert_eq!(
            words.len(),
            expected_words,
            "DenseBitset::from_words: expected {} words for len {}, got {}",
            expected_words,
            len,
            words.len()
        );
        let rem = len % 64;
        if rem != 0 && !words.is_empty() {
            let mask = (1_u64 << rem) - 1;
            words[expected_words - 1] &= mask;
        }
        Self { words, len }
    }

    /// Constructs a bitset with active bits at the specified indices.
    ///
    /// # Panics
    /// Panics if any index is `>= len`.
    #[must_use]
    pub fn from_indices(len: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let mut bitset = Self::zeros(len);
        for id in indices {
            bitset.set_bit(id, true);
        }
        bitset
    }

    /// Number of elements tracked by the bitset.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitset contains zero elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slice view of underlying 64-bit words.
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Tests whether bit `id` is set.
    ///
    /// Returns `false` safely without panicking if `id >= self.len`.
    #[inline]
    #[must_use]
    pub fn get_bit(&self, id: usize) -> bool {
        if id >= self.len {
            return false;
        }
        let word_idx = id >> 6;
        let bit_idx = id & 63;
        (self.words[word_idx] & (1_u64 << bit_idx)) != 0
    }

    /// Sets or clears bit `id`.
    ///
    /// # Panics
    /// Panics with `"Bit index out of bounds"` if `id >= self.len`.
    #[inline]
    pub fn set_bit(&mut self, id: usize, value: bool) {
        assert!(
            id < self.len,
            "Bit index out of bounds: id={}, len={}",
            id,
            self.len
        );
        let word_idx = id >> 6;
        let bit_idx = id & 63;
        if value {
            self.words[word_idx] |= 1_u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1_u64 << bit_idx);
        }
    }

    /// Total count of active (1) bits (`u64::count_ones` per word).
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Total count of inactive (0) bits within valid bounds.
    #[inline]
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Performs in-place bitwise AND (`self &= other`).
    ///
    /// # Panics
    /// Panics if `self.len != other.len`.
    pub fn bitand_assign(&mut self, other: &Self) {
        assert_eq!(
            self.len, other.len,
            "Bitset length mismatch in AND: self.len={}, other.len={}",
            self.len, other.len
        );
        for (w_self, w_other) in self.words.iter_mut().zip(&other.words) {
            *w_self &= *w_other;
        }
    }

    /// Performs in-place bitwise OR (`self |= other`).
    ///
    /// # Panics
    /// Panics if `self.len != other.len`.
    pub fn bitor_assign(&mut self, other: &Self) {
        assert_eq!(
            self.len, other.len,
            "Bitset length mismatch in OR: self.len={}, other.len={}",
            self.len, other.len
        );
        for (w_self, w_other) in self.words.iter_mut().zip(&other.words) {
            *w_self |= *w_other;
        }
    }

    /// Returns a zero-allocation iterator over all active (1) vector IDs strictly `< self.len`.
    #[inline]
    #[must_use]
    pub fn iter_ones(&self) -> BitsetOneIterator<'_> {
        BitsetOneIterator {
            bitset: self,
            word_idx: 0,
            current_word: self.words.first().copied().unwrap_or(0),
        }
    }
}

impl std::ops::BitAndAssign<&DenseBitset> for DenseBitset {
    #[inline]
    fn bitand_assign(&mut self, rhs: &DenseBitset) {
        self.bitand_assign(rhs);
    }
}

impl std::ops::BitOrAssign<&DenseBitset> for DenseBitset {
    #[inline]
    fn bitor_assign(&mut self, rhs: &DenseBitset) {
        self.bitor_assign(rhs);
    }
}

impl std::ops::BitAnd<&DenseBitset> for &DenseBitset {
    type Output = DenseBitset;

    #[inline]
    fn bitand(self, rhs: &DenseBitset) -> Self::Output {
        let mut out = self.clone();
        out.bitand_assign(rhs);
        out
    }
}

impl std::ops::BitOr<&DenseBitset> for &DenseBitset {
    type Output = DenseBitset;

    #[inline]
    fn bitor(self, rhs: &DenseBitset) -> Self::Output {
        let mut out = self.clone();
        out.bitor_assign(rhs);
        out
    }
}

impl FilterPredicate for DenseBitset {
    #[inline]
    fn matches(&self, id: u32) -> bool {
        self.get_bit(id as usize)
    }

    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        Some(self)
    }

    #[inline]
    fn count_matching(&self, _total: usize) -> Option<usize> {
        Some(self.count_ones())
    }
}

/// Zero-allocation ascending iterator over active (1) bit IDs of a [`DenseBitset`].
///
/// Uses hardware/compiler trailing zero count (`trailing_zeros`) and clears the lowest
/// set bit with `w & (w - 1)` for fast skipping of sparse zeroes.
#[derive(Debug, Clone)]
pub struct BitsetOneIterator<'a> {
    bitset: &'a DenseBitset,
    word_idx: usize,
    current_word: u64,
}

impl Iterator for BitsetOneIterator<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_word != 0 {
                let tz = self.current_word.trailing_zeros() as usize;
                let id = (self.word_idx << 6) | tz;
                self.current_word &= self.current_word - 1;
                if id < self.bitset.len {
                    return Some(id);
                }
                // Padding bits in the final word beyond len: terminate immediately.
                self.current_word = 0;
                self.word_idx = self.bitset.words.len();
                return None;
            }

            if self.word_idx >= self.bitset.words.len() {
                return None;
            }

            self.word_idx += 1;
            if self.word_idx < self.bitset.words.len() {
                self.current_word = self.bitset.words[self.word_idx];
            } else {
                return None;
            }
        }
    }
}

impl std::iter::FusedIterator for BitsetOneIterator<'_> {}

/// Opaque closure-based predicate wrapper with an estimated selectivity $\sigma \in [0.0, 1.0]$.
pub struct LambdaPredicate<F>
where
    F: Fn(u32) -> bool + Send + Sync,
{
    closure: F,
    /// Estimated selectivity fraction in $[0.0, 1.0]$.
    pub est_sel: f64,
}

impl<F> LambdaPredicate<F>
where
    F: Fn(u32) -> bool + Send + Sync,
{
    /// Creates a new `LambdaPredicate` with given closure and estimated selectivity.
    pub fn new(closure: F, est_sel: f64) -> Self {
        Self { closure, est_sel }
    }

    /// Estimated selectivity fraction.
    #[inline]
    #[must_use]
    pub fn est_sel(&self) -> f64 {
        self.est_sel
    }
}

impl<F> std::fmt::Debug for LambdaPredicate<F>
where
    F: Fn(u32) -> bool + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LambdaPredicate")
            .field("est_sel", &self.est_sel)
            .finish_non_exhaustive()
    }
}

impl<F> FilterPredicate for LambdaPredicate<F>
where
    F: Fn(u32) -> bool + Send + Sync,
{
    #[inline]
    fn matches(&self, id: u32) -> bool {
        (self.closure)(id)
    }

    #[inline]
    fn bitset(&self) -> Option<&DenseBitset> {
        None
    }

    #[inline]
    fn estimated_selectivity(&self) -> Option<f64> {
        Some(self.est_sel)
    }
}

/// Query-time parameter settings for filtered approximate nearest neighbor search.
#[derive(Debug, Clone)]
pub struct FilteredSearchParams {
    /// Beam width during the ground layer search traversal.
    pub ef_search: usize,
    /// Strategy 1 over-fetch factor multiplier.
    /// `None` uses analytical confidence formula $\gamma(\sigma, \delta = 0.001)$.
    pub overfetch_factor: Option<f64>,
    /// Strategy 1 candidate pool ceiling $K'_{\max}$ (default 4096; $N$ for uncapped arm).
    pub overfetch_cap: usize,
    /// Strategy 2 consecutive bridge-hop budget $H_{\max}$.
    /// `None` defaults to $3 \cdot efSearch$.
    pub max_bridge_hops: Option<usize>,
    /// Maximum allowed predicate evaluations per query for opaque predicates without bitset.
    /// `None` means unconstrained evaluation.
    pub eval_limit: Option<usize>,
    /// Whether bitset-backed predicates may fallback to Strategy 3 upon truncation/budget exhaustion.
    pub allow_bruteforce_fallback: bool,
}

impl Default for FilteredSearchParams {
    fn default() -> Self {
        Self {
            ef_search: 50,
            overfetch_factor: None,
            overfetch_cap: 4096,
            max_bridge_hops: None,
            eval_limit: None,
            allow_bruteforce_fallback: false,
        }
    }
}

/// Diagnostic outcome flags explaining truncation or execution path transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FilteredSearchFlags {
    /// Strategy 1 candidate pool reached $K'_{\max}$ with fewer than $k$ matching items.
    pub post_filter_cap_reached: bool,
    /// Strategy 2 bridge hop counter exceeded $H_{\max}$ with fewer than $k$ matching items.
    pub hop_budget_exhausted: bool,
    /// Predicate evaluations reached `eval_limit`.
    pub eval_limit_reached: bool,
    /// Query redirected to Strategy 3 pre-filtered brute-force scan.
    pub used_bruteforce_fallback: bool,
}

/// Filtered approximate nearest neighbor search result package.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredSearchOutput {
    /// Ascending sorted (distance, vector_id) pairs satisfying the filter predicate.
    pub results: Vec<(f32, usize)>,
    /// Diagnostic flags tracking truncation or fallback events.
    pub flags: FilteredSearchFlags,
}

impl FilteredSearchOutput {
    /// Creates a new `FilteredSearchOutput`.
    #[must_use]
    pub fn new(results: Vec<(f32, usize)>, flags: FilteredSearchFlags) -> Self {
        Self { results, flags }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_bitset() {
        let b = DenseBitset::zeros(0);
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
        assert_eq!(b.count_ones(), 0);
        assert_eq!(b.count_zeros(), 0);
        assert_eq!(b.words().len(), 0);
        assert!(!b.get_bit(0));
        assert!(!b.get_bit(63));
        assert!(!b.get_bit(100));

        let mut it = b.iter_ones();
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);

        let ones = DenseBitset::ones(0);
        assert_eq!(ones.len(), 0);
        assert!(ones.is_empty());
        assert_eq!(ones.count_ones(), 0);
        assert_eq!(ones.words().len(), 0);

        // FilterPredicate trait evaluation on empty bitset
        assert!(!b.matches(0));
        assert!(!b.matches(42));
        assert!(b.bitset().is_some());
        assert_eq!(b.count_matching(0), Some(0));
        assert_eq!(b.count_matching(100), Some(0));
    }

    #[test]
    fn test_non_multiple_of_64_lengths() {
        let test_lengths = [1, 2, 7, 63, 65, 100, 127, 129, 200];
        for &len in &test_lengths {
            let mut b = DenseBitset::zeros(len);
            assert_eq!(b.len(), len);
            assert!(!b.is_empty());
            assert_eq!(b.count_ones(), 0);
            assert_eq!(b.count_zeros(), len);
            assert_eq!(b.words().len(), len.div_ceil(64));

            // Out-of-bounds safe reads
            assert!(!b.get_bit(len));
            assert!(!b.get_bit(len + 1));
            assert!(!b.get_bit(len + 64));

            // Set all bits
            for i in 0..len {
                b.set_bit(i, true);
                assert!(b.get_bit(i));
            }
            assert_eq!(b.count_ones(), len);
            assert_eq!(b.count_zeros(), 0);

            // Ascending iterator completeness
            let active_ids: Vec<usize> = b.iter_ones().collect();
            let expected: Vec<usize> = (0..len).collect();
            assert_eq!(active_ids, expected);

            // Clear alternating bits
            for i in (0..len).step_by(2) {
                b.set_bit(i, false);
                assert!(!b.get_bit(i));
            }
            let odd_ids: Vec<usize> = b.iter_ones().collect();
            let expected_odds: Vec<usize> = (1..len).step_by(2).collect();
            assert_eq!(odd_ids, expected_odds);
            assert_eq!(b.count_ones(), expected_odds.len());
        }
    }

    #[test]
    fn test_all_ones_bitset() {
        let lengths = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 1000];
        for &len in &lengths {
            let b = DenseBitset::ones(len);
            assert_eq!(b.len(), len);
            assert_eq!(b.count_ones(), len);
            assert_eq!(b.count_zeros(), 0);

            for i in 0..len {
                assert!(b.get_bit(i), "Bit {} should be set for len {}", i, len);
            }
            assert!(!b.get_bit(len));
            assert!(!b.get_bit(len + 10));

            let collected: Vec<usize> = b.iter_ones().collect();
            let expected: Vec<usize> = (0..len).collect();
            assert_eq!(collected, expected);
        }
    }

    #[test]
    fn test_word_boundaries() {
        let len = 256;
        let boundary_indices = [0, 63, 64, 127, 128, 191, 192, 255];

        for &target_id in &boundary_indices {
            let mut b = DenseBitset::zeros(len);
            b.set_bit(target_id, true);

            assert!(b.get_bit(target_id));
            assert_eq!(b.count_ones(), 1);

            for other_id in 0..len {
                if other_id != target_id {
                    assert!(
                        !b.get_bit(other_id),
                        "Bit {} should be false when only {} is set",
                        other_id,
                        target_id
                    );
                }
            }

            let collected: Vec<usize> = b.iter_ones().collect();
            assert_eq!(collected, vec![target_id]);
        }

        // Multiple boundary bits active simultaneously
        let mut b = DenseBitset::zeros(len);
        for &id in &boundary_indices {
            b.set_bit(id, true);
        }
        assert_eq!(b.count_ones(), boundary_indices.len());
        let collected: Vec<usize> = b.iter_ones().collect();
        assert_eq!(collected, boundary_indices.to_vec());
    }

    #[test]
    fn test_bounds_safety_get_bit() {
        let b = DenseBitset::ones(100);
        assert!(b.get_bit(0));
        assert!(b.get_bit(99));
        assert!(!b.get_bit(100));
        assert!(!b.get_bit(101));
        assert!(!b.get_bit(128));
        assert!(!b.get_bit(usize::MAX));
        assert!(!b.get_bit(1_000_000));
    }

    #[test]
    #[should_panic(expected = "Bit index out of bounds")]
    fn test_set_bit_panics_at_exact_len() {
        let mut b = DenseBitset::zeros(100);
        b.set_bit(100, true);
    }

    #[test]
    #[should_panic(expected = "Bit index out of bounds")]
    fn test_set_bit_panics_on_empty() {
        let mut b = DenseBitset::zeros(0);
        b.set_bit(0, true);
    }

    #[test]
    #[should_panic(expected = "Bit index out of bounds")]
    fn test_set_bit_panics_at_word_boundary() {
        let mut b = DenseBitset::zeros(64);
        b.set_bit(64, true);
    }

    #[test]
    fn test_bitwise_and_and_or_operations() {
        let len = 200;
        let mut a = DenseBitset::zeros(len);
        let mut b = DenseBitset::zeros(len);

        // a: even indices (0, 2, 4, ...)
        for i in (0..len).step_by(2) {
            a.set_bit(i, true);
        }
        // b: multiples of 3 (0, 3, 6, ...)
        for i in (0..len).step_by(3) {
            b.set_bit(i, true);
        }

        // Bitwise AND
        let mut a_and = a.clone();
        a_and.bitand_assign(&b);
        for i in 0..len {
            let expected = (i % 2 == 0) && (i % 3 == 0);
            assert_eq!(a_and.get_bit(i), expected, "AND mismatch at index {}", i);
        }
        let and_collected: Vec<usize> = a_and.iter_ones().collect();
        let expected_and: Vec<usize> = (0..len).filter(|&i| i % 6 == 0).collect();
        assert_eq!(and_collected, expected_and);

        // Operator &
        let and_op = &a & &b;
        assert_eq!(and_op, a_and);

        // Bitwise OR
        let mut a_or = a.clone();
        a_or.bitor_assign(&b);
        for i in 0..len {
            let expected = (i % 2 == 0) || (i % 3 == 0);
            assert_eq!(a_or.get_bit(i), expected, "OR mismatch at index {}", i);
        }
        let or_collected: Vec<usize> = a_or.iter_ones().collect();
        let expected_or: Vec<usize> = (0..len).filter(|&i| i % 2 == 0 || i % 3 == 0).collect();
        assert_eq!(or_collected, expected_or);

        // Operator |
        let or_op = &a | &b;
        assert_eq!(or_op, a_or);
    }

    #[test]
    #[should_panic(expected = "Bitset length mismatch in AND")]
    fn test_bitand_mismatched_lengths_panics() {
        let mut a = DenseBitset::zeros(64);
        let b = DenseBitset::zeros(65);
        a.bitand_assign(&b);
    }

    #[test]
    #[should_panic(expected = "Bitset length mismatch in OR")]
    fn test_bitor_mismatched_lengths_panics() {
        let mut a = DenseBitset::zeros(100);
        let b = DenseBitset::zeros(99);
        a.bitor_assign(&b);
    }

    #[test]
    fn test_iter_ones_ascending_order_and_exhaustion() {
        let len = 500;
        let mut b = DenseBitset::zeros(len);
        let target_ids = [5, 12, 63, 64, 99, 127, 128, 255, 300, 499];
        for &id in &target_ids {
            b.set_bit(id, true);
        }

        let mut it = b.iter_ones();
        for &expected_id in &target_ids {
            assert_eq!(it.next(), Some(expected_id));
        }
        // Fused exhaustion
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_from_words_and_from_indices() {
        let words = vec![0x1, 0x3];
        let bitset = DenseBitset::from_words(words, 66);
        assert_eq!(bitset.len(), 66);
        assert!(bitset.get_bit(0));
        assert!(bitset.get_bit(64));
        assert!(bitset.get_bit(65));
        assert!(!bitset.get_bit(66)); // out of bounds

        let indices = vec![1, 4, 10, 63, 70];
        let from_idx = DenseBitset::from_indices(100, indices.clone());
        assert_eq!(from_idx.len(), 100);
        assert_eq!(from_idx.count_ones(), indices.len());
        let collected: Vec<usize> = from_idx.iter_ones().collect();
        assert_eq!(collected, indices);
    }

    #[test]
    fn test_filter_predicate_evaluation() {
        let len = 150;
        let mut bitset = DenseBitset::zeros(len);
        bitset.set_bit(10, true);
        bitset.set_bit(50, true);
        bitset.set_bit(149, true);

        // DenseBitset as FilterPredicate
        let pred: &dyn FilterPredicate = &bitset;
        assert!(pred.matches(10));
        assert!(pred.matches(50));
        assert!(pred.matches(149));
        assert!(!pred.matches(0));
        assert!(!pred.matches(11));
        assert!(!pred.matches(150));
        assert!(!pred.matches(1000));
        assert_eq!(pred.count_matching(len), Some(3));
        assert!(pred.bitset().is_some());
        assert_eq!(pred.bitset().unwrap().len(), len);
        assert_eq!(pred.estimated_selectivity(), None);

        // LambdaPredicate evaluation
        let lambda = LambdaPredicate::new(|id| id % 10 == 0, 0.10);
        assert_eq!(lambda.est_sel(), 0.10);
        assert!(lambda.matches(0));
        assert!(lambda.matches(10));
        assert!(lambda.matches(100));
        assert!(!lambda.matches(1));
        assert!(!lambda.matches(9));
        assert!(lambda.bitset().is_none());
        assert_eq!(lambda.count_matching(1000), None);
        assert_eq!(lambda.estimated_selectivity(), Some(0.10));

        // Debug formatting does not panic
        let debug_str = format!("{:?}", lambda);
        assert!(debug_str.contains("LambdaPredicate"));
        assert!(debug_str.contains("0.1"));

        // Boxed and borrowed predicate dispatch
        let borrowed = &lambda;
        assert_eq!(borrowed.estimated_selectivity(), Some(0.10));
        let arc_pred: std::sync::Arc<dyn FilterPredicate> =
            std::sync::Arc::new(LambdaPredicate::new(|id| id % 5 == 0, 0.20));
        assert_eq!(arc_pred.estimated_selectivity(), Some(0.20));
        let boxed: Box<dyn FilterPredicate> = Box::new(lambda);
        assert!(boxed.matches(20));
        assert!(!boxed.matches(21));
        assert!(boxed.bitset().is_none());
        assert_eq!(boxed.estimated_selectivity(), Some(0.10));
    }

    #[test]
    fn test_filtered_search_params_and_flags_defaults() {
        let params = FilteredSearchParams::default();
        assert_eq!(params.ef_search, 50);
        assert_eq!(params.overfetch_cap, 4096);
        assert_eq!(params.overfetch_factor, None);
        assert_eq!(params.max_bridge_hops, None);
        assert_eq!(params.eval_limit, None);
        assert!(!params.allow_bruteforce_fallback);

        let flags = FilteredSearchFlags::default();
        assert!(!flags.post_filter_cap_reached);
        assert!(!flags.hop_budget_exhausted);
        assert!(!flags.eval_limit_reached);
        assert!(!flags.used_bruteforce_fallback);

        let output = FilteredSearchOutput::new(vec![(0.5, 3)], flags);
        assert_eq!(output.results.len(), 1);
        assert_eq!(output.flags, flags);
    }
}
