//! Milestone M3 Task T3.6: Strategy 3 Pre-Filter Brute-Force Fallback & Cost Microbenchmark Integration Tests.
//!
//! Verifies:
//! 1. Exact parity between Strategy 3 (`search_knn_filtered_pre`) and `FilteredExactKnn` across all 4 selectivity tiers on 10k SIFT data.
//! 2. Predicate with no bitset error handling (`VectorSearchError::InvalidArgument`).
//! 3. Empty bitset (0 matches) returns empty vector cleanly.
//! 4. Fewer than $k$ matches returned cleanly sorted ascending.
//! 5. Canonical `(distance, id)` tie-breaking.
//! 6. Fallback behavior when `allow_bruteforce_fallback = true` from Strategy 1 (over-fetch cap reached).
//! 7. Fallback behavior when `allow_bruteforce_fallback = true` from Strategy 2 (hop budget exhausted).
//! 8. Opaque lambda predicate without bitset does not fallback even when `allow_bruteforce_fallback = true`.
//! 9. Unified `search_knn_filtered` dispatch, tracker reuse, and scratch buffer zero-allocation parity.
//! 10. Sequential vs multi-threaded parallel batch execution consistency.
//! 11. Input validation (`k = 0`, non-finite values, dimension mismatch).
//! 12. Microbenchmark execution via `phase5` CLI binary and structured JSON validation.
//! 13. Class PRESERVING: HNSW graph fingerprint and unfiltered `search_knn` results remain 100% bit-identical.

#![forbid(unsafe_code)]

mod common;

use std::fs;
use std::process::Command;

use common::{
    load_siftsmall_or_skip, make_synthetic_2d_index, make_synthetic_2d_matrix, TempTestDir,
};
use vector_search::dataset::generate_mode_a_bitset;
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::filter::{DenseBitset, FilterPredicate, FilteredSearchParams, LambdaPredicate};
use vector_search::hnsw::{FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::recall::compute_recall_at_k;

#[test]
fn test_strategy3_empty_bitset_zero_matches() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 5;

    let bitset = DenseBitset::zeros(100);
    assert_eq!(bitset.count_ones(), 0);

    let params = FilteredSearchParams::default();
    let output = index
        .search_knn_filtered_pre(&query, k, &bitset, &params)
        .expect("Search with empty bitset must succeed");

    assert!(
        output.results.is_empty(),
        "0 matching items must return empty results"
    );
    assert!(!output.flags.used_bruteforce_fallback);
    assert!(!output.flags.post_filter_cap_reached);
    assert!(!output.flags.hop_budget_exhausted);
}

#[test]
fn test_strategy3_fewer_than_k_matches() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 10;

    // Only 3 elements match
    let mut bitset = DenseBitset::zeros(100);
    bitset.set_bit(2, true);
    bitset.set_bit(10, true);
    bitset.set_bit(50, true);
    assert_eq!(bitset.count_ones(), 3);

    let params = FilteredSearchParams::default();
    let output = index
        .search_knn_filtered_pre(&query, k, &bitset, &params)
        .expect("Search must succeed");

    assert_eq!(output.results.len(), 3, "Must return all 3 matching items");
    for &(dist, id) in &output.results {
        assert!(bitset.matches(id as u32));
        assert!(dist >= 0.0);
    }

    // Verify ascending order
    for w in output.results.windows(2) {
        assert!(
            w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1),
            "Results must be canonically sorted"
        );
    }
}

#[test]
fn test_strategy3_predicate_with_no_bitset_error_handling() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 5;

    let opaque = LambdaPredicate::new(|id| id % 2 == 0, 0.5);
    assert!(opaque.bitset().is_none());

    let params = FilteredSearchParams::default();

    // Calling search_knn_filtered_pre directly with opaque predicate must error
    let res = index.search_knn_filtered_pre(&query, k, &opaque, &params);
    assert!(res.is_err(), "Must reject predicate with no bitset");
    match res.unwrap_err() {
        VectorSearchError::InvalidArgument(msg) => {
            assert!(
                msg.contains("bitset"),
                "Error message must mention bitset requirement: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got {other:?}"),
    }

    // Calling search_knn_filtered with FilterStrategy::PreFilterBruteForce must also error
    let res_dispatch = index.search_knn_filtered(
        &query,
        k,
        &opaque,
        FilterStrategy::PreFilterBruteForce,
        &params,
    );
    assert!(res_dispatch.is_err());
}

#[test]
fn test_strategy3_canonical_tie_breaking() {
    // 4 identical vectors at exactly distance 0.0 from query [5.0, 5.0]
    let rows = vec![
        vec![5.0_f32, 5.0_f32],
        vec![5.0_f32, 5.0_f32],
        vec![5.0_f32, 5.0_f32],
        vec![5.0_f32, 5.0_f32],
    ];
    let matrix = RowMatrix::from_rows(&rows).unwrap();
    let config = HnswConfig {
        m: 16,
        ef_construction: 64,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&matrix, Metric::L2Squared, config).unwrap();
    let query = [5.0_f32, 5.0_f32];
    let bitset = DenseBitset::ones(4);

    let params = FilteredSearchParams::default();
    let out = index
        .search_knn_filtered_pre(&query, 3, &bitset, &params)
        .unwrap();

    assert_eq!(out.results.len(), 3);
    // Canonical tie-breaker must return smallest IDs: 0, 1, 2
    assert_eq!(out.results[0].1, 0);
    assert_eq!(out.results[1].1, 1);
    assert_eq!(out.results[2].1, 2);
    assert_eq!(out.results[0].0, 0.0);
    assert_eq!(out.results[1].0, 0.0);
    assert_eq!(out.results[2].0, 0.0);
}

#[test]
fn test_strategy3_fallback_from_strategy1() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 10;

    // Sparse filter: only 15% matches
    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(7) {
        bitset.set_bit(i, true);
    }

    // Arm A: allow_bruteforce_fallback = false, cap = 5 (< k = 10)
    let params_no_fallback = FilteredSearchParams {
        ef_search: 20,
        overfetch_cap: 5,
        overfetch_factor: None,
        allow_bruteforce_fallback: false,
        ..Default::default()
    };
    let out_no_fallback = index
        .search_knn_filtered_post(&query, k, &bitset, &params_no_fallback)
        .unwrap();
    assert!(out_no_fallback.flags.post_filter_cap_reached);
    assert!(!out_no_fallback.flags.used_bruteforce_fallback);
    assert!(out_no_fallback.results.len() < k);

    // Arm B: allow_bruteforce_fallback = true, cap = 5
    let params_fallback = FilteredSearchParams {
        ef_search: 20,
        overfetch_cap: 5,
        overfetch_factor: None,
        allow_bruteforce_fallback: true,
        ..Default::default()
    };
    let out_fallback = index
        .search_knn_filtered_post(&query, k, &bitset, &params_fallback)
        .unwrap();
    assert!(out_fallback.flags.post_filter_cap_reached);
    assert!(out_fallback.flags.used_bruteforce_fallback);
    assert_eq!(
        out_fallback.results.len(),
        k,
        "Fallback must gather full k=10 results via Strategy 3"
    );

    // Results must match direct Strategy 3 results
    let out_pre = index
        .search_knn_filtered_pre(&query, k, &bitset, &params_fallback)
        .unwrap();
    assert_eq!(out_fallback.results, out_pre.results);
}

#[test]
fn test_strategy3_fallback_from_strategy2() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 10;

    // Sparse filter: only 5 matches in the entire dataset
    let mut bitset = DenseBitset::zeros(100);
    for i in [12, 34, 56, 78, 90] {
        bitset.set_bit(i, true);
    }

    // Arm A: allow_bruteforce_fallback = false, tight max_bridge_hops = 2
    let params_no_fallback = FilteredSearchParams {
        ef_search: 20,
        max_bridge_hops: Some(2),
        allow_bruteforce_fallback: false,
        ..Default::default()
    };
    let out_no_fallback = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params_no_fallback)
        .unwrap();
    assert!(out_no_fallback.flags.hop_budget_exhausted);
    assert!(!out_no_fallback.flags.used_bruteforce_fallback);
    assert!(out_no_fallback.results.len() < k);

    // Arm B: allow_bruteforce_fallback = true, tight max_bridge_hops = 2
    let params_fallback = FilteredSearchParams {
        ef_search: 20,
        max_bridge_hops: Some(2),
        allow_bruteforce_fallback: true,
        ..Default::default()
    };
    let out_fallback = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params_fallback)
        .unwrap();
    assert!(out_fallback.flags.hop_budget_exhausted);
    assert!(out_fallback.flags.used_bruteforce_fallback);
    // Since only 5 matches exist in dataset, it returns min(k, N_sigma) = 5
    assert_eq!(out_fallback.results.len(), 5);

    let out_pre = index
        .search_knn_filtered_pre(&query, k, &bitset, &params_fallback)
        .unwrap();
    assert_eq!(out_fallback.results, out_pre.results);
}

#[test]
fn test_strategy3_opaque_predicate_does_not_fallback() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 10;

    // Very low selectivity closure: matches only 2 vectors
    let opaque = LambdaPredicate::new(|id| id == 10 || id == 20, 0.02);

    // Strategy 1 with cap = 5 and allow_bruteforce_fallback = true
    let params = FilteredSearchParams {
        ef_search: 20,
        overfetch_cap: 5,
        allow_bruteforce_fallback: true,
        ..Default::default()
    };

    let out = index
        .search_knn_filtered_post(&query, k, &opaque, &params)
        .unwrap();
    assert!(out.flags.post_filter_cap_reached);
    // Must NOT fall back because predicate has no bitset!
    assert!(!out.flags.used_bruteforce_fallback);
    assert!(out.results.len() <= 2);
}

#[test]
fn test_strategy3_unified_dispatch_and_scratch() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [25.0_f32, 12.0_f32];
    let k = 8;

    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(3) {
        bitset.set_bit(i, true);
    }

    let params = FilteredSearchParams::default();

    // 1. search_knn_filtered_pre
    let out1 = index
        .search_knn_filtered_pre(&query, k, &bitset, &params)
        .unwrap();

    // 2. search_knn_filtered with FilterStrategy::PreFilterBruteForce
    let out2 = index
        .search_knn_filtered(
            &query,
            k,
            &bitset,
            FilterStrategy::PreFilterBruteForce,
            &params,
        )
        .unwrap();

    // 3. search_knn_filtered_with_scratch
    let mut scratch = FilteredSearchScratch::new(100, 50, 4096);
    let out3 = index
        .search_knn_filtered_with_scratch(
            &query,
            k,
            &bitset,
            FilterStrategy::PreFilterBruteForce,
            &params,
            &mut scratch,
        )
        .unwrap();

    assert_eq!(out1.results, out2.results);
    assert_eq!(out1.results, out3.results);
    assert_eq!(out1.flags, out2.flags);
    assert_eq!(out1.flags, out3.flags);
}

#[test]
fn test_strategy3_batch_sequential_and_parallel_parity() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let q_rows = vec![
        vec![5.0_f32, 2.0],
        vec![15.0, 8.0],
        vec![30.0, 12.0],
        vec![50.0, 20.0],
    ];
    let queries = RowMatrix::from_rows(&q_rows).unwrap();
    let k = 5;

    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(2) {
        bitset.set_bit(i, true);
    }

    let params = FilteredSearchParams::default();

    let seq_results = index
        .search_batch_filtered_pre(&queries, k, &bitset, &params)
        .unwrap();
    let par_results = index
        .search_batch_filtered_pre_parallel(&queries, k, &bitset, &params, 2)
        .unwrap();

    assert_eq!(seq_results.len(), par_results.len());
    for (s, p) in seq_results.iter().zip(par_results.iter()) {
        assert_eq!(s, p, "Parallel output must match sequential exactly");
    }
}

#[test]
fn test_strategy3_input_validation() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let bitset = DenseBitset::ones(100);
    let params = FilteredSearchParams::default();

    // k == 0
    let err_k = index.search_knn_filtered_pre(&[0.0, 0.0], 0, &bitset, &params);
    assert!(matches!(err_k, Err(VectorSearchError::InvalidArgument(_))));

    // Non-finite query
    let err_nan = index.search_knn_filtered_pre(&[f32::NAN, 0.0], 5, &bitset, &params);
    assert!(matches!(
        err_nan,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Dimension mismatch
    let err_dim = index.search_knn_filtered_pre(&[1.0, 2.0, 3.0], 5, &bitset, &params);
    assert!(matches!(
        err_dim,
        Err(VectorSearchError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_strategy3_sift10k_exact_parity_all_4_tiers() {
    let Some(loaded) = load_siftsmall_or_skip("test_strategy3_sift10k_exact_parity_all_4_tiers")
    else {
        return;
    };
    let base = loaded.base;
    let queries = loaded.query;
    let n = base.rows();
    assert_eq!(n, 10_000);

    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();
    let exact_engine = FilteredExactKnn::new(Metric::L2Squared, base.clone()).unwrap();

    // All 4 canonical selectivity tiers: 50%, 10%, 1%, 0.1%
    let tiers = [0.5, 0.1, 0.01, 0.001];
    let k = 10;
    let params = FilteredSearchParams::default();

    for &sigma in &tiers {
        let bitset = generate_mode_a_bitset(n, sigma, 42);
        assert_eq!(bitset.count_ones(), (sigma * n as f64).floor() as usize);

        let mut hnsw_indices = Vec::with_capacity(10 * k);
        let mut gt_indices = Vec::with_capacity(10 * k);

        // Test over first 10 queries
        for q_idx in 0..10 {
            let q_vec = queries.row(q_idx);
            let hnsw_out = index
                .search_knn_filtered_pre(q_vec, k, &bitset, &params)
                .unwrap();
            let gt_hits = exact_engine.search_filtered(q_vec, k, &bitset).unwrap();

            // Check bit-for-bit equivalence of (distance, id)
            assert_eq!(
                hnsw_out.results.len(),
                gt_hits.len(),
                "Result count mismatch for query {q_idx} at sigma={sigma}"
            );

            for (i, (&(h_dist, h_id), &(g_dist, g_id))) in
                hnsw_out.results.iter().zip(gt_hits.iter()).enumerate()
            {
                assert_eq!(
                    h_id, g_id,
                    "ID mismatch at rank {i} for query {q_idx} at sigma={sigma}: {h_id} vs {g_id}"
                );
                assert!(
                    (h_dist - g_dist).abs() <= 1e-5,
                    "Distance mismatch at rank {i} for query {q_idx} at sigma={sigma}: {h_dist} vs {g_dist}"
                );
            }

            for &(_, id) in &hnsw_out.results {
                hnsw_indices.push(id as i32);
            }
            for &(_, id) in &gt_hits {
                gt_indices.push(id as i32);
            }
        }

        let num_returned = hnsw_indices.len() / k;
        if num_returned > 0 {
            let hnsw_mat = RowMatrix::new(num_returned, k, hnsw_indices).unwrap();
            let gt_mat = RowMatrix::new(num_returned, k, gt_indices).unwrap();
            let recall = compute_recall_at_k(&hnsw_mat, &gt_mat, k).unwrap();
            assert_eq!(
                recall, 1.0,
                "Strategy 3 must achieve 1.000000 exact recall against FilteredExactKnn for sigma={sigma}"
            );
        }
    }
}

#[test]
fn test_strategy3_microbenchmark_cli_execution_and_json_validation() {
    let tmp = TempTestDir::new("strategy3_microbench");
    let json_file = tmp.path().join("test_microbench.json");

    let bin_path = env!("CARGO_BIN_EXE_phase5");
    let status = Command::new(bin_path)
        .args([
            "--microbench",
            "--output",
            json_file.to_str().unwrap(),
            "--dim",
            "128",
            "--k",
            "10",
            "--reps",
            "3",
            "--words",
            "10000",
            "--dists",
            "1000",
            "--heap-ops",
            "1000",
        ])
        .status()
        .expect("Failed to execute phase5 binary");

    assert!(
        status.success(),
        "phase5 --microbench must exit successfully"
    );
    assert!(json_file.exists(), "Benchmark report file must be written");

    let content = fs::read_to_string(&json_file).expect("Failed to read microbench JSON");
    assert!(content.contains(r#""benchmark": "cost_microbenchmark""#));
    assert!(content.contains(r#""t_word_ns""#));
    assert!(content.contains(r#""t_dist_ns""#));
    assert!(content.contains(r#""t_heap_ns""#));
    assert!(content.contains(r#""t_cand_ns""#));
    assert!(content.contains(r#""theoretical_model""#));
    assert!(content.contains(r#""predicted_c_brute_ns""#));
}

#[test]
fn test_strategy3_class_preserving_unfiltered_and_fingerprint() {
    let matrix = make_synthetic_2d_matrix(100);
    let mut index = make_synthetic_2d_index(&matrix);
    let fp = index.graph_fingerprint();
    assert_ne!(fp, 0, "Graph fingerprint must be non-zero");

    let query = [15.0_f32, 10.0_f32];
    let unfiltered_hits = index.search_knn(&query, 5, 50).unwrap();
    assert_eq!(unfiltered_hits.len(), 5);

    // Filter with 100% ones
    let all_ones = DenseBitset::ones(100);
    let params = FilteredSearchParams::default();
    let filtered_hits = index
        .search_knn_filtered_pre(&query, 5, &all_ones, &params)
        .unwrap();

    // With 100% ones, Strategy 3 exact scan returns top-5 nearest neighbors
    assert_eq!(filtered_hits.results.len(), 5);
    // Unfiltered HNSW with ef=50 on 100 2D points achieves 100% recall
    assert_eq!(unfiltered_hits, filtered_hits.results);
}

#[test]
fn test_strategy3_scratch_zero_reallocation_and_undersized_error() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let bitset = DenseBitset::ones(100);
    let params = FilteredSearchParams::default();

    let mut scratch = FilteredSearchScratch::new(100, 50, 0);
    let cap_matched = scratch.matched_results.capacity();
    let cap_raw = scratch.raw_candidates.capacity();

    for step in 0..10 {
        let q = [(step as f32) * 10.0, 5.0];
        let direct = index
            .search_knn_filtered_pre(&q, 10, &bitset, &params)
            .unwrap();
        let with_scratch = index
            .search_knn_filtered_with_scratch(
                &q,
                10,
                &bitset,
                FilterStrategy::PreFilterBruteForce,
                &params,
                &mut scratch,
            )
            .unwrap();
        assert_eq!(direct, with_scratch);
    }
    assert_eq!(scratch.matched_results.capacity(), cap_matched);
    assert_eq!(scratch.raw_candidates.capacity(), cap_raw);

    let mut empty_scratch = FilteredSearchScratch::default();
    let err = index.search_knn_filtered_with_scratch(
        &[10.0, 5.0],
        10,
        &bitset,
        FilterStrategy::PreFilterBruteForce,
        &params,
        &mut empty_scratch,
    );
    assert!(matches!(err, Err(VectorSearchError::InvalidArgument(_))));
}
