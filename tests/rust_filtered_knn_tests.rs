//! Milestone M3 Task T3.3: FilteredExactKnn Integration & Self-Parity Tests.
//!
//! Verifies:
//! 1. 10k SIFT smoke dataset ground truth computation across 4 tiers {50%, 10%, 1%, 0.1%}.
//! 2. Self-parity invariant `Recall@k(GT, GT) == 1.000000` at every tier.
//! 3. Canonical `(distance, id)` deterministic tie-breaking on tied distances.
//! 4. 0 matches returns `Ok(vec![])`.
//! 5. Fewer than `k` matches returns all matching vectors sorted ascending.
//! 6. Input validation: non-finite query (NaN/Inf) returns `InvalidArgument`.
//! 7. Input validation: query dimension mismatch returns `DimensionMismatch`.
//! 8. Sequential vs multi-threaded parallel batch execution produces bit-identical output.
//! 9. Opaque closure `LambdaPredicate` produces identical results to `DenseBitset`.
//! 10. CLI subprocess execution with provenance JSON validation.

#![forbid(unsafe_code)]

mod common;

use std::fs;
use std::process::Command;

use common::TempTestDir;
use vector_search::dataset::{generate_mode_a_bitset, read_fvecs, read_ivecs};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::filter::{DenseBitset, LambdaPredicate};
use vector_search::matrix::RowMatrix;
use vector_search::recall::compute_recall_at_k;

#[test]
fn test_integration_sift_10k_four_tiers_self_parity() {
    let Some(loaded) =
        common::load_siftsmall_or_skip("test_integration_sift_10k_four_tiers_self_parity")
    else {
        return;
    };
    let base = loaded.base;
    let query = loaded.query;
    let n = base.rows();
    assert_eq!(n, 10_000);

    let engine = FilteredExactKnn::new(Metric::L2Squared, base).expect("Engine fit failed");
    let tiers = [0.5, 0.1, 0.01, 0.001];
    let k = 10;

    for &sigma in &tiers {
        let expected_popcount = (sigma * n as f64).floor() as usize;
        let bitset = generate_mode_a_bitset(n, sigma, 42);
        assert_eq!(bitset.count_ones(), expected_popcount);

        let batch_res = engine
            .search_batch_parallel(&query, k, &bitset, 4)
            .expect("Search failed");

        let effective_k = batch_res.indices.cols();
        assert_eq!(effective_k, k.min(expected_popcount));

        // Self-parity invariant
        let self_recall = compute_recall_at_k(&batch_res.indices, &batch_res.indices, effective_k)
            .expect("Recall calculation failed");
        assert!(
            (self_recall - 1.0).abs() < f64::EPSILON,
            "Self-parity failed for tier sigma={sigma}: recall={self_recall}"
        );
    }
}

#[test]
fn test_integration_canonical_tie_breaking_order() {
    // Vector 0: distance 25.0
    // Vector 1: distance 4.0
    // Vector 2: distance 100.0
    // Vector 3: distance 4.0 (tied with 1)
    // Vector 4: distance 25.0 (tied with 0)
    // Vector 5: distance 4.0 (tied with 1, 3)
    let base = RowMatrix::from_rows(&[
        vec![5.0_f32, 0.0],
        vec![2.0, 0.0],
        vec![10.0, 0.0],
        vec![-2.0, 0.0],
        vec![0.0, 5.0],
        vec![0.0, -2.0],
    ])
    .unwrap();

    let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();
    let bitset = DenseBitset::ones(6);

    let hits = engine.search_filtered(&[0.0, 0.0], 4, &bitset).unwrap();
    assert_eq!(hits.len(), 4);

    // Distance 4.0 ties among 1, 3, 5 -> must resolve to [1, 3, 5]
    assert_eq!(hits[0], (4.0, 1));
    assert_eq!(hits[1], (4.0, 3));
    assert_eq!(hits[2], (4.0, 5));
    // Distance 25.0 ties among 0, 4 -> must resolve to 0
    assert_eq!(hits[3], (25.0, 0));
}

#[test]
fn test_integration_zero_and_fewer_than_k_matches() {
    let base = RowMatrix::from_rows(&[
        vec![1.0_f32, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![4.0, 0.0],
    ])
    .unwrap();
    let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

    // 0 matches
    let empty_bitset = DenseBitset::zeros(4);
    let zero_hits = engine
        .search_filtered(&[0.0, 0.0], 10, &empty_bitset)
        .unwrap();
    assert!(zero_hits.is_empty());

    let q_mat = RowMatrix::from_rows(&[vec![0.0_f32, 0.0]]).unwrap();
    let zero_batch = engine.search_batch(&q_mat, 10, &empty_bitset).unwrap();
    assert_eq!(zero_batch.indices.rows(), 0);
    assert_eq!(zero_batch.indices.cols(), 0);

    // Fewer than k matches (2 matches with k=10)
    let mut bitset_two = DenseBitset::zeros(4);
    bitset_two.set_bit(1, true);
    bitset_two.set_bit(3, true);

    let few_hits = engine
        .search_filtered(&[0.0, 0.0], 10, &bitset_two)
        .unwrap();
    assert_eq!(few_hits.len(), 2);
    assert_eq!(few_hits[0], (4.0, 1));
    assert_eq!(few_hits[1], (16.0, 3));

    let few_batch = engine.search_batch(&q_mat, 10, &bitset_two).unwrap();
    assert_eq!(few_batch.indices.rows(), 1);
    assert_eq!(few_batch.indices.cols(), 2);
    assert_eq!(few_batch.indices.row(0), &[1, 3]);
}

#[test]
fn test_integration_validations() {
    let base = RowMatrix::from_rows(&[vec![1.0_f32, 2.0], vec![3.0, 4.0]]).unwrap();
    let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();
    let bitset = DenseBitset::ones(2);

    // NaN query
    let nan_res = engine.search_filtered(&[f32::NAN, 1.0], 2, &bitset);
    assert!(matches!(
        nan_res,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Inf query
    let inf_res = engine.search_filtered(&[f32::INFINITY, 1.0], 2, &bitset);
    assert!(matches!(
        inf_res,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Dimension mismatch query
    let dim_res = engine.search_filtered(&[1.0, 2.0, 3.0], 2, &bitset);
    assert!(matches!(
        dim_res,
        Err(VectorSearchError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));
}

#[test]
fn test_integration_parallel_and_sequential_batch_parity() {
    let n = 500;
    let dim = 16;
    let mut flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        for d in 0..dim {
            flat.push(((i * 13 + d * 19) % 100) as f32);
        }
    }
    let base = RowMatrix::new(n, dim, flat).unwrap();
    let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

    let num_q = 30;
    let mut q_flat = Vec::with_capacity(num_q * dim);
    for i in 0..num_q {
        for d in 0..dim {
            q_flat.push(((i * 7 + d * 23) % 80) as f32);
        }
    }
    let queries = RowMatrix::new(num_q, dim, q_flat).unwrap();

    let bitset = generate_mode_a_bitset(n, 0.1, 999);
    let k = 10;

    let seq = engine.search_batch(&queries, k, &bitset).unwrap();
    let par = engine
        .search_batch_parallel(&queries, k, &bitset, 4)
        .unwrap();

    assert_eq!(seq.indices.rows(), num_q);
    assert_eq!(seq.indices.cols(), k);
    assert_eq!(seq.indices.as_slice(), par.indices.as_slice());
    assert_eq!(seq.distances.as_slice(), par.distances.as_slice());
}

#[test]
fn test_integration_lambda_predicate_bitset_parity() {
    let n = 200;
    let dim = 8;
    let mut flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        for d in 0..dim {
            flat.push((i + d) as f32);
        }
    }
    let base = RowMatrix::new(n, dim, flat).unwrap();
    let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();

    // DenseBitset predicate: multiples of 4
    let mut bitset = DenseBitset::zeros(n);
    for i in (0..n).step_by(4) {
        bitset.set_bit(i, true);
    }

    // Lambda predicate: same constraint
    let lambda = LambdaPredicate::new(|id| id % 4 == 0, 0.25);

    let query = vec![10.0_f32; dim];
    let hits_bitset = engine.search_filtered(&query, 5, &bitset).unwrap();
    let hits_lambda = engine.search_filtered(&query, 5, &lambda).unwrap();

    assert_eq!(hits_bitset, hits_lambda);
}

#[test]
fn test_integration_cli_subprocess_execution() {
    let tmp = TempTestDir::new("fek_cli_test_out");
    let run_dir = tmp.path();
    let json_file = run_dir.join("report.json");

    let status = Command::new(env!("CARGO_BIN_EXE_filtered_exact_knn"))
        .args([
            "--num-base",
            "500",
            "--num-query",
            "10",
            "--dim",
            "32",
            "--k",
            "5",
            "--selectivity",
            "0.1",
            "--output-dir",
            run_dir.to_str().unwrap(),
            "--json-out",
            json_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute filtered_exact_knn subprocess");

    assert!(status.success());
    assert!(run_dir.join("groundtruth.ivecs").exists());
    assert!(run_dir.join("groundtruth.fvecs").exists());
    assert!(run_dir.join("filter.bin").exists());
    assert!(run_dir.join("filter.ivecs").exists());
    assert!(json_file.exists());

    let json_text = fs::read_to_string(&json_file).expect("Failed to read report.json");
    assert!(json_text.contains("\"self_verified\": true"));
    assert!(json_text.contains("\"self_recall_at_k\": 1.000000"));
    assert!(json_text.contains("\"effective_k\": 5"));

    // Verify .ivecs and .fvecs can be read back cleanly
    let gt_idx = read_ivecs(run_dir.join("groundtruth.ivecs")).unwrap();
    let gt_dist = read_fvecs(run_dir.join("groundtruth.fvecs")).unwrap();
    assert_eq!(gt_idx.rows(), 10);
    assert_eq!(gt_idx.cols(), 5);
    assert_eq!(gt_dist.rows(), 10);
    assert_eq!(gt_dist.cols(), 5);
}
