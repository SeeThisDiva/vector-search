//! Milestone M3 Task T3.2: Deterministic Metadata Generator Integration Tests.
//!
//! Verifies:
//! 1. Mode A popcount exactness for selectivities {0.5, 0.1, 0.01, 0.001}.
//! 2. Mode A deterministic repeatability across calls with identical seeds.
//! 3. Mode B popcount exactness for selectivities {0.5, 0.1, 0.01, 0.001}.
//! 4. Mode B deterministic repeatability across calls.
//! 5. Edge cases: selectivity 0.0, selectivity 1.0, small datasets N < C, empty datasets N = 0.
//! 6. Mode B spatial locality (semantic island clustering).
//! 7. Binary persistence (.bin & .ivecs) round-trips.
//! 8. CLI subprocess execution with provenance JSON validation.

mod common;

use std::fs;
use std::process::Command;

use common::TempTestDir;
use vector_search::dataset::{
    generate_mode_a_bitset, generate_mode_b_bitset, generate_mode_b_bitset_detailed,
    generate_synthetic_sift, read_bitset, read_ivecs, write_bitset, write_bitset_as_ivecs,
    S_CLUST_DEFAULT,
};
use vector_search::distance::l2_squared;
use vector_search::matrix::RowMatrix;

#[test]
fn test_integration_mode_a_four_tiers_popcount_exactness() {
    let n = 10_000;
    let tiers = [0.5, 0.1, 0.01, 0.001];

    for &sigma in &tiers {
        let expected_k = (sigma * n as f64).floor() as usize;
        let bitset = generate_mode_a_bitset(n, sigma, 42);

        assert_eq!(bitset.len(), n);
        assert_eq!(
            bitset.count_ones(),
            expected_k,
            "Mode A exact popcount failed for sigma={sigma}"
        );
    }
}

#[test]
fn test_integration_mode_a_deterministic_repeatability() {
    let n = 5_000;
    let sigma = 0.1;
    let seed = 0xDEAD_BEEF_CAFE_BABE;

    let b1 = generate_mode_a_bitset(n, sigma, seed);
    let b2 = generate_mode_a_bitset(n, sigma, seed);

    assert_eq!(b1.len(), b2.len());
    assert_eq!(b1.count_ones(), b2.count_ones());
    assert_eq!(b1.words(), b2.words());
    assert_eq!(b1, b2);

    // Independent seeds produce distinct bitsets
    let b3 = generate_mode_a_bitset(n, sigma, seed.wrapping_add(1));
    assert_ne!(b1, b3);
}

#[test]
fn test_integration_mode_b_four_tiers_popcount_exactness() {
    // Generate 10k synthetic 128-d SIFT vectors
    let (base_vectors, _) = generate_synthetic_sift(10_000, 10, 128, 100, 2026).unwrap();
    let n = base_vectors.rows();
    let tiers = [0.5, 0.1, 0.01, 0.001];

    for &sigma in &tiers {
        let expected_k = (sigma * n as f64).floor() as usize;
        let meta = generate_mode_b_bitset_detailed(&base_vectors, sigma, 42, S_CLUST_DEFAULT);

        assert_eq!(meta.bitset.len(), n);
        assert_eq!(
            meta.bitset.count_ones(),
            expected_k,
            "Mode B exact popcount failed for sigma={sigma}"
        );

        // Verification of C formula: C = max(64, floor(2.0 / sigma)), clamped to C <= N
        let expected_c = ((2.0 / sigma).floor() as usize).max(64).min(n);
        assert_eq!(meta.centroid_count, expected_c);
    }
}

#[test]
fn test_integration_mode_b_deterministic_repeatability() {
    let (base_vectors, _) = generate_synthetic_sift(2_000, 5, 64, 50, 42).unwrap();
    let sigma = 0.05;

    let meta1 = generate_mode_b_bitset_detailed(&base_vectors, sigma, 100, S_CLUST_DEFAULT);
    let meta2 = generate_mode_b_bitset_detailed(&base_vectors, sigma, 100, S_CLUST_DEFAULT);

    assert_eq!(meta1.bitset, meta2.bitset);
    assert_eq!(meta1.active_clusters, meta2.active_clusters);
    assert_eq!(meta1.shortfall_cluster, meta2.shortfall_cluster);
    assert_eq!(meta1.shortfall_count, meta2.shortfall_count);
}

#[test]
fn test_integration_edge_cases_selectivity_0_and_1() {
    let (base, _) = generate_synthetic_sift(200, 5, 32, 10, 7).unwrap();
    let n = base.rows();

    // Mode A: selectivity 0.0
    let a0 = generate_mode_a_bitset(n, 0.0, 42);
    assert_eq!(a0.count_ones(), 0);
    assert_eq!(a0.count_zeros(), n);

    // Mode A: selectivity 1.0
    let a1 = generate_mode_a_bitset(n, 1.0, 42);
    assert_eq!(a1.count_ones(), n);
    assert_eq!(a1.count_zeros(), 0);

    // Mode B: selectivity 0.0
    let b0 = generate_mode_b_bitset(&base, 0.0, 42);
    assert_eq!(b0.count_ones(), 0);
    assert_eq!(b0.count_zeros(), n);

    // Mode B: selectivity 1.0
    let b1 = generate_mode_b_bitset(&base, 1.0, 42);
    assert_eq!(b1.count_ones(), n);
    assert_eq!(b1.count_zeros(), 0);
}

#[test]
fn test_integration_edge_cases_small_dataset_clamped_c() {
    // When N = 8, C = max(64, floor(2 / 0.1) = 20) is clamped to N = 8
    let (base, _) = generate_synthetic_sift(8, 2, 16, 4, 99).unwrap();
    let sigma = 0.25; // target_count = floor(0.25 * 8) = 2
    let meta = generate_mode_b_bitset_detailed(&base, sigma, 42, S_CLUST_DEFAULT);

    assert_eq!(meta.bitset.len(), 8);
    assert_eq!(meta.bitset.count_ones(), 2);
    assert_eq!(meta.centroid_count, 8);
}

#[test]
fn test_integration_edge_cases_empty_and_single_element() {
    let empty = RowMatrix::<f32>::empty();

    let a_empty = generate_mode_a_bitset(0, 0.5, 42);
    assert_eq!(a_empty.len(), 0);
    assert!(a_empty.is_empty());

    let b_empty = generate_mode_b_bitset(&empty, 0.5, 42);
    assert_eq!(b_empty.len(), 0);
    assert!(b_empty.is_empty());

    let single = RowMatrix::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b_single_0 = generate_mode_b_bitset(&single, 0.0, 42);
    assert_eq!(b_single_0.count_ones(), 0);
    let b_single_1 = generate_mode_b_bitset(&single, 1.0, 42);
    assert_eq!(b_single_1.count_ones(), 1);
}

#[test]
fn test_integration_mode_b_spatial_locality() {
    // Spatial locality invariant:
    // In Mode B, active vectors belong to localized clusters in metric space.
    // Therefore, the average pairwise distance between active vectors in Mode B
    // should be significantly lower than between uniformly distributed active vectors in Mode A.
    let (base, _) = generate_synthetic_sift(1_000, 5, 32, 20, 1234).unwrap();
    let sigma = 0.1; // 100 vectors

    let mode_a = generate_mode_a_bitset(1_000, sigma, 42);
    let mode_b = generate_mode_b_bitset(&base, sigma, 42);

    let active_a: Vec<usize> = mode_a.iter_ones().collect();
    let active_b: Vec<usize> = mode_b.iter_ones().collect();
    assert_eq!(active_a.len(), 100);
    assert_eq!(active_b.len(), 100);

    // Compute sample mean pairwise distance for first 50 pairs
    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut pairs = 0;
    for i in 0..50 {
        for j in (i + 1)..50 {
            let dist_a = l2_squared(base.row(active_a[i]), base.row(active_a[j]));
            let dist_b = l2_squared(base.row(active_b[i]), base.row(active_b[j]));
            sum_a += dist_a as f64;
            sum_b += dist_b as f64;
            pairs += 1;
        }
    }
    let mean_a = sum_a / pairs as f64;
    let mean_b = sum_b / pairs as f64;

    assert!(
        mean_b < mean_a,
        "Mode B mean pairwise distance ({mean_b}) must be smaller than Mode A ({mean_a}) due to cluster correlation"
    );
}

#[test]
fn test_integration_bitset_persistence_roundtrip() {
    let tmp = TempTestDir::new("metadata_bitset");
    let bin_path = tmp.path().join("integration_bitset.bin");
    let ivecs_path = tmp.path().join("integration_bitset.ivecs");

    let bitset = generate_mode_a_bitset(1_500, 0.05, 777);
    assert_eq!(bitset.count_ones(), 75);

    // Binary format round-trip
    write_bitset(&bin_path, &bitset).unwrap();
    let loaded_bin = read_bitset(&bin_path).unwrap();
    assert_eq!(bitset, loaded_bin);

    // Texmex .ivecs format round-trip
    write_bitset_as_ivecs(&ivecs_path, &bitset).unwrap();
    let loaded_ivecs = read_ivecs(&ivecs_path).unwrap();
    assert_eq!(loaded_ivecs.rows(), 1);
    assert_eq!(loaded_ivecs.cols(), 75);

    let bitset_ids: Vec<i32> = bitset.iter_ones().map(|id| id as i32).collect();
    assert_eq!(loaded_ivecs.row(0), bitset_ids.as_slice());
}

#[test]
fn test_integration_cli_subprocess_execution() {
    let tmp = TempTestDir::new("metadata_cli");
    let bin_out = tmp.path().join("cli_test_out.bin");
    let json_out = tmp.path().join("cli_test_out.json");

    // Run CLI executable
    let status = Command::new(env!("CARGO_BIN_EXE_generate_metadata"))
        .args([
            "--mode",
            "correlated",
            "--num-base",
            "500",
            "--selectivity",
            "0.1",
            "--seed",
            "123",
            "--output",
            bin_out.to_str().unwrap(),
            "--json-out",
            json_out.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute generate_metadata CLI binary");

    assert!(status.success(), "CLI binary failed with status: {status}");
    assert!(bin_out.exists(), "CLI did not create output bitset binary");
    assert!(json_out.exists(), "CLI did not create output JSON");

    // Verify loaded bitset from CLI
    let loaded_bitset = read_bitset(&bin_out).unwrap();
    assert_eq!(loaded_bitset.len(), 500);
    assert_eq!(loaded_bitset.count_ones(), 50);

    // Verify loaded JSON from CLI
    let json_text = fs::read_to_string(&json_out).unwrap();
    assert!(json_text.contains("\"mode\": \"correlated\""));
    assert!(json_text.contains("\"num_vectors\": 500"));
    assert!(json_text.contains("\"target_popcount\": 50"));
    assert!(json_text.contains("\"actual_popcount\": 50"));
    assert!(json_text.contains("\"git_commit\":"));
    assert!(json_text.contains("\"active_cluster_ids\":"));
}
