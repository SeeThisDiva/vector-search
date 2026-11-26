//! Deterministic filtered search metadata generation (Mode A & Mode B) and bitset binary I/O.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::texmex_io::{write_ivecs, IO_BUF_CAPACITY};
use crate::distance::l2_squared;
use crate::error::{Result, VectorSearchError};
use crate::filter::DenseBitset;
use crate::json_util::ensure_parent_dir;
use crate::matrix::RowMatrix;
use crate::rng::DeterministicRng;

/// Canonical fixed cluster centroid seed for Mode B ($S_{\text{clust}} = 0\text{xCBF29CE484222325}$).
pub const S_CLUST_DEFAULT: u64 = 0xCBF2_9CE4_8422_2325;

/// Generates a deterministic independent (Bernoulli) bitset (Mode A).
///
/// Follows DESIGN.md Algorithm 1:
/// 1. Computes target count $K_\sigma = \lfloor \sigma \cdot N \rfloor$.
/// 2. Deterministically shuffles indices $0..N-1$ using Fisher-Yates with `DeterministicRng::new(seed)`.
/// 3. Activates the first $K_\sigma$ shuffled indices.
///
/// Guaranteed to produce exact popcount $K_\sigma$ with zero variance.
///
/// # Panics
/// Panics if `selectivity` is not in $[0.0, 1.0]$ or is NaN.
#[must_use]
pub fn generate_mode_a_bitset(num_vectors: usize, selectivity: f64, seed: u64) -> DenseBitset {
    assert!(
        (0.0..=1.0).contains(&selectivity),
        "selectivity must be in [0.0, 1.0], got {selectivity}"
    );
    let target_count = (selectivity * num_vectors as f64).floor() as usize;
    let mut bitset = DenseBitset::zeros(num_vectors);
    if target_count == 0 || num_vectors == 0 {
        return bitset;
    }
    let mut indices: Vec<usize> = (0..num_vectors).collect();
    let mut rng = DeterministicRng::new(seed);
    for i in (1..num_vectors).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
    for &id in &indices[..target_count] {
        bitset.set_bit(id, true);
    }
    bitset
}

/// Metadata diagnostics produced by Mode B generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeBMetadata {
    /// Resulting dense bitset with exact popcount $\lfloor \sigma \cdot N \rfloor$.
    pub bitset: DenseBitset,
    /// Number of centroids $C = \min(N, \max(64, \lfloor 2 / \sigma \rfloor))$.
    pub centroid_count: usize,
    /// Sorted IDs of whole clusters activated in the greedy stage.
    pub active_clusters: Vec<usize>,
    /// Cluster ID from which shortfall vectors were activated, if shortfall $s > 0$.
    pub shortfall_cluster: Option<usize>,
    /// Shortfall count $s = K_\sigma - \text{accumulated}$.
    pub shortfall_count: usize,
}

/// Selects $C$ centroids from `vectors` using deterministic $K$-means++ initialization.
///
/// # Panics
/// Panics if $C > N$.
pub fn deterministic_kmeans_plus_plus(
    vectors: &RowMatrix<f32>,
    c: usize,
    rng: &mut DeterministicRng,
) -> Vec<Vec<f32>> {
    let n = vectors.rows();
    assert!(c <= n, "centroid count c ({c}) cannot exceed n ({n})");
    if c == 0 || n == 0 {
        return Vec::new();
    }
    let mut centroids = Vec::with_capacity(c);
    // 1. Uniformly sample first centroid
    let first_idx = rng.next_usize(n);
    centroids.push(vectors.row(first_idx).to_vec());

    // 2. Track minimum squared L2 distance from each vector to nearest centroid
    let mut min_sq_dist: Vec<f32> = (0..n)
        .map(|i| l2_squared(vectors.row(i), &centroids[0]))
        .collect();

    // 3. Iteratively sample remaining centroids with probability proportional to D(x)^2
    for _ in 1..c {
        let total_dist: f64 = min_sq_dist.iter().map(|&d| d as f64).sum();
        let chosen_idx = if total_dist > 0.0 {
            let threshold = rng.next_f64_open01() * total_dist;
            let mut cumsum = 0.0;
            let mut idx = n - 1;
            for (i, &dist) in min_sq_dist.iter().enumerate() {
                cumsum += dist as f64;
                if cumsum >= threshold {
                    idx = i;
                    break;
                }
            }
            idx
        } else {
            // Degenerate case (e.g. duplicate points)
            rng.next_usize(n)
        };

        let new_centroid = vectors.row(chosen_idx).to_vec();
        for (i, dist) in min_sq_dist.iter_mut().enumerate() {
            let d = l2_squared(vectors.row(i), &new_centroid);
            if d < *dist {
                *dist = d;
            }
        }
        centroids.push(new_centroid);
    }

    centroids
}

fn generate_mode_b_bitset_core(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    seed_clust: u64,
) -> ModeBMetadata {
    assert!(
        (0.0..=1.0).contains(&selectivity),
        "selectivity must be in [0.0, 1.0], got {selectivity}"
    );
    let n = vectors.rows();
    if n == 0 {
        return ModeBMetadata {
            bitset: DenseBitset::zeros(0),
            centroid_count: 0,
            active_clusters: Vec::new(),
            shortfall_cluster: None,
            shortfall_count: 0,
        };
    }

    let target_count = (selectivity * n as f64).floor() as usize;
    if target_count == 0 {
        return ModeBMetadata {
            bitset: DenseBitset::zeros(n),
            centroid_count: 64.min(n),
            active_clusters: Vec::new(),
            shortfall_cluster: None,
            shortfall_count: 0,
        };
    }

    // Centroid count C = max(64, floor(2.0 / sigma)), clamped to C <= N
    let c_floor = (2.0 / selectivity).floor();
    let c_raw = if c_floor >= usize::MAX as f64 {
        n
    } else {
        c_floor as usize
    };
    let c = c_raw.max(64).min(n);

    let mut rng_clust = DeterministicRng::new(seed_clust);
    let centroids = deterministic_kmeans_plus_plus(vectors, c, &mut rng_clust);

    // Vector assignment: z_i = argmin_c ||V[i] - centroids[c]||_2^2 (ties broken by cluster ID c)
    let mut cluster_assignments = Vec::with_capacity(n);
    for i in 0..n {
        let v = vectors.row(i);
        let mut best_c = 0;
        let mut best_dist = l2_squared(v, &centroids[0]);
        for (c_idx, centroid) in centroids.iter().enumerate().skip(1) {
            let d = l2_squared(v, centroid);
            if d < best_dist {
                best_dist = d;
                best_c = c_idx;
            }
        }
        cluster_assignments.push(best_c);
    }

    // Cluster sizes
    let mut cluster_sizes = vec![0_usize; c];
    for &assign in &cluster_assignments {
        cluster_sizes[assign] += 1;
    }

    // Sort clusters descending by size, ties broken by cluster ID
    let mut sorted_clusters: Vec<usize> = (0..c).collect();
    sorted_clusters.sort_by(|&a, &b| {
        cluster_sizes[b]
            .cmp(&cluster_sizes[a])
            .then_with(|| a.cmp(&b))
    });

    // Greedily accumulate whole clusters
    let mut active_clusters = Vec::new();
    let mut is_active = vec![false; c];
    let mut accumulated = 0_usize;

    for &c_idx in &sorted_clusters {
        let sz = cluster_sizes[c_idx];
        if accumulated + sz <= target_count {
            active_clusters.push(c_idx);
            is_active[c_idx] = true;
            accumulated += sz;
        }
    }

    let mut bitset = DenseBitset::zeros(n);
    for (i, &assign) in cluster_assignments.iter().enumerate() {
        if is_active[assign] {
            bitset.set_bit(i, true);
        }
    }

    // Shortfall calibration
    let shortfall = target_count - accumulated;
    let mut shortfall_cluster = None;

    if shortfall > 0 {
        let chosen_cluster = if active_clusters.is_empty() {
            sorted_clusters[0]
        } else {
            let mut best_c = None;
            let mut best_dist = f32::INFINITY;
            for c_cand in 0..c {
                if is_active[c_cand] {
                    continue;
                }
                let mut min_to_active = f32::INFINITY;
                for &a in &active_clusters {
                    let d = l2_squared(&centroids[c_cand], &centroids[a]);
                    if d < min_to_active {
                        min_to_active = d;
                    }
                }
                let key = if min_to_active.is_nan() {
                    f32::INFINITY
                } else {
                    min_to_active
                };
                if best_c.is_none() || key < best_dist {
                    best_dist = key;
                    best_c = Some(c_cand);
                }
            }
            best_c.expect("Must find a non-active cluster when shortfall > 0")
        };

        shortfall_cluster = Some(chosen_cluster);

        // Member vectors sorted by (distance to own centroid, id)
        let mut members = Vec::with_capacity(cluster_sizes[chosen_cluster]);
        for (i, &assign) in cluster_assignments.iter().enumerate() {
            if assign == chosen_cluster {
                let d = l2_squared(vectors.row(i), &centroids[chosen_cluster]);
                members.push((d, i));
            }
        }

        members.sort_by(|&(d_a, id_a), &(d_b, id_b)| {
            d_a.partial_cmp(&d_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| id_a.cmp(&id_b))
        });

        assert!(
            members.len() >= shortfall,
            "Chosen shortfall cluster has {} members, but shortfall is {}",
            members.len(),
            shortfall
        );

        for &(_d, id) in members.iter().take(shortfall) {
            bitset.set_bit(id, true);
        }
    }

    assert_eq!(
        bitset.count_ones(),
        target_count,
        "Mode B generated popcount mismatch: expected {target_count}, got {}",
        bitset.count_ones()
    );

    ModeBMetadata {
        bitset,
        centroid_count: c,
        active_clusters,
        shortfall_cluster,
        shortfall_count: shortfall,
    }
}

/// Generates a deterministic cluster-correlated bitset (Mode B) with detailed metadata.
///
/// Follows DESIGN.md Algorithm 2 with exact shortfall calibration:
/// 1. $K_\sigma = \lfloor \sigma \cdot N \rfloor$.
/// 2. Centroid count $C = \max(64, \lfloor 2.0 / \sigma \rfloor)$, clamped to $C \le N$.
/// 3. Centroids chosen via deterministic $K$-means++ with `seed_clust`.
/// 4. Vector partitioning by nearest centroid (ties broken by cluster ID).
/// 5. Greedy accumulation of whole clusters sorted descending by size (ties broken by cluster ID).
/// 6. Shortfall calibration: non-active cluster nearest to any active centroid (ties by cluster ID;
///    if no active clusters, largest cluster) activates its $s$ members with smallest `(distance, id)`.
///
/// Guaranteed exact popcount $K_\sigma = \lfloor \sigma \cdot N \rfloor$ with zero variance.
///
/// Note: `_seed_attr` is retained for signature compatibility with Task T3.2 callers.
///
/// # Panics
/// Panics if `selectivity` is not in $[0.0, 1.0]$ or is NaN.
pub fn generate_mode_b_bitset_detailed(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    _seed_attr: u64,
    seed_clust: u64,
) -> ModeBMetadata {
    generate_mode_b_bitset_core(vectors, selectivity, seed_clust)
}

/// Generates a deterministic cluster-correlated bitset (Mode B) with default centroid seed.
///
/// Signature matching Task T3.2 specification:
/// `generate_mode_b_bitset(vectors: &RowMatrix<f32>, selectivity: f64, seed_attr: u64) -> DenseBitset`
#[must_use]
pub fn generate_mode_b_bitset(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    _seed_attr: u64,
) -> DenseBitset {
    generate_mode_b_bitset_core(vectors, selectivity, S_CLUST_DEFAULT).bitset
}

/// Generates a deterministic cluster-correlated bitset (Mode B) with explicit cluster seed.
#[must_use]
pub fn generate_mode_b_bitset_with_seed(
    vectors: &RowMatrix<f32>,
    selectivity: f64,
    _seed_attr: u64,
    seed_clust: u64,
) -> DenseBitset {
    generate_mode_b_bitset_core(vectors, selectivity, seed_clust).bitset
}

// -----------------------------------------------------------------------------
// Bitset Binary Persistence (I/O)
// -----------------------------------------------------------------------------

const BITSET_MAGIC: &[u8; 8] = b"BITSET\x01\x00";

/// Writes a `DenseBitset` into a structured little-endian binary format with header validation.
///
/// Binary layout:
/// - 8 bytes: magic `b"BITSET\x01\x00"`
/// - 8 bytes: `len` as `u64` little-endian
/// - 8 bytes: `num_words` as `u64` little-endian
/// - `8 * num_words` bytes: 64-bit words as little-endian
pub fn write_bitset<P: AsRef<Path>>(path: P, bitset: &DenseBitset) -> Result<()> {
    let path_ref = path.as_ref();
    ensure_parent_dir(path_ref)?;
    let file = File::create(path_ref)?;
    let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

    writer.write_all(BITSET_MAGIC)?;
    writer.write_all(&(bitset.len() as u64).to_le_bytes())?;
    writer.write_all(&(bitset.words().len() as u64).to_le_bytes())?;
    for &word in bitset.words() {
        writer.write_all(&word.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

/// Reads a `DenseBitset` from a structured little-endian binary file.
pub fn read_bitset<P: AsRef<Path>>(path: P) -> Result<DenseBitset> {
    let path_ref = path.as_ref();
    let meta = fs::metadata(path_ref)?;
    let file_size = meta.len();

    if file_size < 24 {
        return Err(VectorSearchError::CorruptData(format!(
            "Bitset file too small ({file_size} < 24 bytes): {}",
            path_ref.display()
        )));
    }

    let file = File::open(path_ref)?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);

    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != BITSET_MAGIC {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid bitset file magic in {}",
            path_ref.display()
        )));
    }

    let mut buf8 = [0_u8; 8];
    reader.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let num_words = u64::from_le_bytes(buf8) as usize;

    let expected_words = len.div_ceil(64);
    if num_words != expected_words {
        return Err(VectorSearchError::CorruptData(format!(
            "Bitset word count mismatch: header claims {num_words}, expected {expected_words} for len {len}"
        )));
    }

    let expected_file_size = 24_u64 + (num_words as u64) * 8;
    if file_size != expected_file_size {
        return Err(VectorSearchError::CorruptData(format!(
            "Bitset file size mismatch: actual {file_size}, expected {expected_file_size}"
        )));
    }

    let mut words = Vec::with_capacity(num_words);
    for _ in 0..num_words {
        reader.read_exact(&mut buf8)?;
        words.push(u64::from_le_bytes(buf8));
    }

    Ok(DenseBitset::from_words(words, len))
}

/// Writes active (1) IDs of a `DenseBitset` into a Texmex-compatible `.ivecs` file.
///
/// Emits a single vector record whose dimension is the popcount of the bitset.
pub fn write_bitset_as_ivecs<P: AsRef<Path>>(path: P, bitset: &DenseBitset) -> Result<()> {
    let popcount = bitset.count_ones();
    let ids: Vec<i32> = bitset
        .iter_ones()
        .map(|id| {
            i32::try_from(id).map_err(|_| {
                VectorSearchError::InvalidArgument(format!(
                    "Vector ID {id} exceeds i32::MAX for .ivecs export"
                ))
            })
        })
        .collect::<Result<Vec<i32>>>()?;

    let matrix = if popcount == 0 {
        RowMatrix::empty()
    } else {
        RowMatrix::new(1, popcount, ids)?
    };

    write_ivecs(path, &matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::sift::generate_synthetic_sift;
    use crate::dataset::texmex_io::read_ivecs;

    #[test]
    fn test_mode_a_popcount_exactness() {
        let n = 10_000;
        let selectivities = [0.5, 0.1, 0.01, 0.001];
        for &sel in &selectivities {
            let expected_k = (sel * n as f64).floor() as usize;
            let bitset = generate_mode_a_bitset(n, sel, 42);
            assert_eq!(bitset.len(), n);
            assert_eq!(
                bitset.count_ones(),
                expected_k,
                "Mode A exact popcount failed for sel={sel}"
            );
        }

        // Non-round size and selectivity
        let non_round_n = 1234;
        let sel = 0.33;
        let expected_k = (sel * non_round_n as f64).floor() as usize;
        let bitset = generate_mode_a_bitset(non_round_n, sel, 999);
        assert_eq!(bitset.len(), non_round_n);
        assert_eq!(bitset.count_ones(), expected_k);
    }

    #[test]
    fn test_mode_a_deterministic_repeatability() {
        let n = 2000;
        let sel = 0.15;
        let seed = 12345;
        let b1 = generate_mode_a_bitset(n, sel, seed);
        let b2 = generate_mode_a_bitset(n, sel, seed);
        assert_eq!(b1, b2);
        assert_eq!(b1.words(), b2.words());

        // Different seeds should produce different bitsets
        let b3 = generate_mode_a_bitset(n, sel, seed + 1);
        assert_ne!(b1, b3);
    }

    #[test]
    fn test_mode_a_edge_cases() {
        // Selectivity 0.0
        let b0 = generate_mode_a_bitset(500, 0.0, 42);
        assert_eq!(b0.len(), 500);
        assert_eq!(b0.count_ones(), 0);

        // Selectivity 1.0
        let b1 = generate_mode_a_bitset(500, 1.0, 42);
        assert_eq!(b1.len(), 500);
        assert_eq!(b1.count_ones(), 500);

        // N = 0
        let b_empty = generate_mode_a_bitset(0, 0.5, 42);
        assert_eq!(b_empty.len(), 0);
        assert_eq!(b_empty.count_ones(), 0);

        // N = 1
        let b_one_0 = generate_mode_a_bitset(1, 0.0, 42);
        assert_eq!(b_one_0.count_ones(), 0);
        let b_one_1 = generate_mode_a_bitset(1, 1.0, 42);
        assert_eq!(b_one_1.count_ones(), 1);
        assert!(b_one_1.get_bit(0));
    }

    #[test]
    #[should_panic(expected = "selectivity must be in [0.0, 1.0]")]
    fn test_mode_a_negative_selectivity_panics() {
        let _ = generate_mode_a_bitset(100, -0.01, 42);
    }

    #[test]
    #[should_panic(expected = "selectivity must be in [0.0, 1.0]")]
    fn test_mode_a_excess_selectivity_panics() {
        let _ = generate_mode_a_bitset(100, 1.01, 42);
    }

    #[test]
    fn test_kmeans_plus_plus_determinism() {
        let (base, _) = generate_synthetic_sift(200, 5, 16, 10, 42).unwrap();
        let mut rng1 = DeterministicRng::new(0xCBF2_9CE4_8422_2325);
        let mut rng2 = DeterministicRng::new(0xCBF2_9CE4_8422_2325);
        let c1 = deterministic_kmeans_plus_plus(&base, 10, &mut rng1);
        let c2 = deterministic_kmeans_plus_plus(&base, 10, &mut rng2);
        assert_eq!(c1.len(), 10);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_mode_b_popcount_exactness() {
        let (base, _) = generate_synthetic_sift(1000, 10, 32, 20, 101).unwrap();
        let selectivities = [0.5, 0.1, 0.01, 0.001];
        for &sel in &selectivities {
            let expected_k = (sel * base.rows() as f64).floor() as usize;
            let bitset = generate_mode_b_bitset(&base, sel, 42);
            assert_eq!(bitset.len(), base.rows());
            assert_eq!(
                bitset.count_ones(),
                expected_k,
                "Mode B exact popcount failed for sel={sel}"
            );
        }
    }

    #[test]
    fn test_mode_b_deterministic_repeatability() {
        let (base, _) = generate_synthetic_sift(500, 10, 16, 10, 777).unwrap();
        let sel = 0.05;
        let meta1 = generate_mode_b_bitset_detailed(&base, sel, 42, S_CLUST_DEFAULT);
        let meta2 = generate_mode_b_bitset_detailed(&base, sel, 42, S_CLUST_DEFAULT);
        assert_eq!(meta1.bitset, meta2.bitset);
        assert_eq!(meta1.active_clusters, meta2.active_clusters);
        assert_eq!(meta1.shortfall_cluster, meta2.shortfall_cluster);
        assert_eq!(meta1.shortfall_count, meta2.shortfall_count);
    }

    #[test]
    fn test_mode_b_edge_cases() {
        let (base, _) = generate_synthetic_sift(100, 5, 16, 5, 55).unwrap();

        // Selectivity 0.0
        let b0 = generate_mode_b_bitset(&base, 0.0, 42);
        assert_eq!(b0.len(), 100);
        assert_eq!(b0.count_ones(), 0);

        // Selectivity 1.0
        let b1 = generate_mode_b_bitset(&base, 1.0, 42);
        assert_eq!(b1.len(), 100);
        assert_eq!(b1.count_ones(), 100);

        // Small dataset N < C (e.g. N = 10, C = max(64, ...) clamped to 10)
        let (small_base, _) = generate_synthetic_sift(10, 2, 8, 3, 12).unwrap();
        let b_small = generate_mode_b_bitset(&small_base, 0.3, 42);
        assert_eq!(b_small.len(), 10);
        assert_eq!(b_small.count_ones(), 3);

        // N = 0
        let empty_matrix = RowMatrix::<f32>::empty();
        let b_empty = generate_mode_b_bitset(&empty_matrix, 0.5, 42);
        assert_eq!(b_empty.len(), 0);
        assert_eq!(b_empty.count_ones(), 0);
    }

    #[test]
    fn test_mode_b_shortfall_calibration_with_odd_selectivity() {
        let (base, _) = generate_synthetic_sift(300, 5, 16, 10, 888).unwrap();
        // An odd selectivity that guarantees fractional cluster shortfall
        let sel: f64 = 0.073; // target_count = floor(0.073 * 300) = 21
        let expected_k = (sel * 300.0_f64).floor() as usize;
        let meta = generate_mode_b_bitset_detailed(&base, sel, 42, S_CLUST_DEFAULT);
        assert_eq!(meta.bitset.count_ones(), expected_k);
        // Shortfall should be calibrated cleanly
        if meta.shortfall_count > 0 {
            assert!(meta.shortfall_cluster.is_some());
        }
    }

    #[test]
    fn test_bitset_binary_io_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bitset_roundtrip.bin");

        let mut original = DenseBitset::zeros(300);
        original.set_bit(0, true);
        original.set_bit(63, true);
        original.set_bit(64, true);
        original.set_bit(127, true);
        original.set_bit(299, true);

        write_bitset(&file_path, &original).unwrap();
        let loaded = read_bitset(&file_path).unwrap();

        assert_eq!(original, loaded);
        assert_eq!(loaded.len(), 300);
        assert_eq!(loaded.count_ones(), 5);
        assert!(loaded.get_bit(0));
        assert!(loaded.get_bit(63));
        assert!(loaded.get_bit(64));
        assert!(loaded.get_bit(127));
        assert!(loaded.get_bit(299));
        assert!(!loaded.get_bit(1));

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_bitset_ivecs_io_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bitset_ivecs.ivecs");

        let mut original = DenseBitset::zeros(100);
        original.set_bit(5, true);
        original.set_bit(17, true);
        original.set_bit(88, true);

        write_bitset_as_ivecs(&file_path, &original).unwrap();
        let matrix = read_ivecs(&file_path).unwrap();
        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.row(0), &[5, 17, 88]);

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_bitset_io_corruption_handling() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bitset_corrupt.bin");

        // Too small (< 24 bytes)
        std::fs::write(&file_path, b"TOO_SHORT").unwrap();
        assert!(read_bitset(&file_path).is_err());

        // Invalid magic
        let bad_magic = vec![0_u8; 32];
        std::fs::write(&file_path, bad_magic).unwrap();
        assert!(read_bitset(&file_path).is_err());

        let _ = std::fs::remove_file(file_path);
    }
}
