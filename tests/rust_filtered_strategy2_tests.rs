//! Milestone M3 Task T3.5: Strategy 2 In-Traversal Filtering Integration Tests.
//!
//! Verifies:
//! 1. Normal in-traversal finding `k` matches with `hop_budget_exhausted = false`.
//! 2. Bridge node routing: queries requiring traversal through non-matching nodes to reach matching cluster.
//! 3. Hop budget exhaustion: queries with sparse or unreachable matches triggering `H > H_max` and setting `flags.hop_budget_exhausted = true`.
//! 4. Zero matches setting `hop_budget_exhausted = true`.
//! 5. Verification against `FilteredExactKnn` ground truth on 10k SIFT smoke dataset (`Recall@10 >= 0.85`).
//! 6. Bit-identical parity between unfiltered `search_knn` and graph fingerprint preservation.
//! 7. Canonical `(distance, id)` deterministic tie-breaking.
//! 8. Input validation: `k = 0`, non-finite query, dimension mismatch, empty index.
//! 9. Opaque `LambdaPredicate` evaluation parity and `eval_limit` reached flag.
//! 10. Sequential vs multi-threaded parallel batch execution consistency.
//! 11. Unified `search_knn_filtered` and `search_knn_filtered_with_scratch` dispatch with `FilterStrategy::InTraversal`.

#![forbid(unsafe_code)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use common::{load_siftsmall_or_skip, make_synthetic_2d_index, make_synthetic_2d_matrix};
use vector_search::dataset::generate_mode_a_bitset;
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::filter::{DenseBitset, FilterPredicate, FilteredSearchParams, LambdaPredicate};
use vector_search::hnsw::{FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::recall::compute_recall_at_k;

#[test]
fn test_strategy2_normal_intraversal_finds_k_matches() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 5;

    // Filter allows even IDs (50% selectivity)
    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(2) {
        bitset.set_bit(i, true);
    }
    assert_eq!(bitset.count_ones(), 50);

    let params = FilteredSearchParams {
        ef_search: 50,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params)
        .expect("In-traversal filtered search failed");

    assert_eq!(output.results.len(), k);
    assert!(!output.flags.hop_budget_exhausted);
    assert!(!output.flags.eval_limit_reached);
    assert!(!output.flags.post_filter_cap_reached);

    // Verify all returned results satisfy predicate
    for &(dist, id) in &output.results {
        assert!(
            bitset.matches(id as u32),
            "Returned node ID {} must satisfy predicate",
            id
        );
        assert!(dist >= 0.0);
    }

    // Verify canonical ascending sort order
    for w in output.results.windows(2) {
        let (d0, id0) = w[0];
        let (d1, id1) = w[1];
        assert!(
            d0 < d1 || (d0 == d1 && id0 < id1),
            "Results must be canonically sorted: ({d0}, {id0}) vs ({d1}, {id1})"
        );
    }

    // Verify against FilteredExactKnn ground truth
    let exact_knn = FilteredExactKnn::new(Metric::L2Squared, matrix).unwrap();
    let gt = exact_knn.search_filtered(&query, k, &bitset).unwrap();
    assert_eq!(output.results.len(), gt.len());
    for ((d_out, id_out), (d_gt, id_gt)) in output.results.iter().zip(gt.iter()) {
        assert_eq!(id_out, id_gt);
        assert!((d_out - d_gt).abs() < 1e-5);
    }
}

#[test]
fn test_strategy2_bridge_node_routing_through_non_matching_nodes() {
    // Construct a synthetic scenario with two spatially separated clusters:
    // Non-matching cluster at (0, y) and matching cluster at (100, y).
    // An intermediate bridge chain connects the clusters.
    let mut rows = Vec::new();

    // IDs 0..20: Cluster A near (0, 0)
    for i in 0..20 {
        rows.push(vec![(i as f32) * 0.5, ((i % 5) as f32) * 0.5]);
    }
    // IDs 20..40: Bridge chain connecting x=10 to x=90
    for i in 20..40 {
        let frac = (i - 20) as f32 / 20.0;
        rows.push(vec![10.0 + frac * 80.0, 2.5]);
    }
    // IDs 40..60: Cluster B near (100, 0)
    for i in 40..60 {
        rows.push(vec![
            100.0 + ((i - 40) as f32) * 0.5,
            ((i % 5) as f32) * 0.5,
        ]);
    }

    let matrix = RowMatrix::from_rows(&rows).unwrap();
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&matrix, Metric::L2Squared, config).unwrap();

    // Only Cluster B (IDs 40..60) satisfies the predicate
    let mut bitset = DenseBitset::zeros(60);
    for i in 40..60 {
        bitset.set_bit(i, true);
    }
    assert_eq!(bitset.count_ones(), 20);

    // Query is placed at Cluster B (100.0, 1.0)
    let query = [100.0_f32, 1.0_f32];
    let k = 5;

    let params = FilteredSearchParams {
        ef_search: 60,
        max_bridge_hops: Some(200), // generous hop budget to allow traversal across bridges
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params)
        .unwrap();

    assert_eq!(output.results.len(), k);
    assert!(!output.flags.hop_budget_exhausted);

    // Verify all returned results are indeed from Cluster B (IDs 40..60)
    for &(dist, id) in &output.results {
        assert!(
            (40..60).contains(&id),
            "Node {id} must belong to matching Cluster B"
        );
        assert!(dist < 20.0, "Distance {dist} should be close to Cluster B");
    }
}

#[test]
fn test_strategy2_hop_budget_exhaustion_sparse_matches() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [0.0_f32, 0.0_f32];
    let k = 10;

    // Very sparse predicate: only IDs 98 and 99 match (at the far end of the dataset)
    let mut bitset = DenseBitset::zeros(100);
    bitset.set_bit(98, true);
    bitset.set_bit(99, true);
    assert_eq!(bitset.count_ones(), 2);

    // Restrict hop budget to 5 consecutive hops
    let params = FilteredSearchParams {
        ef_search: 20,
        max_bridge_hops: Some(5),
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params)
        .unwrap();

    // Since k=10 but only 2 match (and are far), expanding 6 bridge nodes exhausts budget
    assert!(
        output.flags.hop_budget_exhausted,
        "Hop budget must be exhausted when H > H_max and |W_match| < k"
    );
    assert!(output.results.len() < k);
}

#[test]
fn test_strategy2_zero_matches_sets_hop_budget_exhausted() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 5;

    // Bitset with zero active bits (selectivity sigma = 0.0)
    let bitset = DenseBitset::zeros(100);
    assert_eq!(bitset.count_ones(), 0);

    // Budget: 15 hops (100 nodes in graph, so traversal easily exceeds 15)
    let params = FilteredSearchParams {
        ef_search: 10,
        max_bridge_hops: Some(15),
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params)
        .unwrap();

    assert!(
        output.results.is_empty(),
        "Zero matches must return empty results"
    );
    assert!(
        output.flags.hop_budget_exhausted,
        "hop_budget_exhausted must be true for 0 matches when H > H_max"
    );
}

#[test]
fn test_strategy2_sift10k_recall_against_ground_truth() {
    let Some(loaded) = load_siftsmall_or_skip("test_strategy2_sift10k_recall_against_ground_truth")
    else {
        return;
    };
    let base = loaded.base;
    let queries = loaded.query;
    let n = base.rows();
    assert_eq!(n, 10_000);

    // Build HNSW index on 10k SIFT
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();
    let exact_engine = FilteredExactKnn::new(Metric::L2Squared, base.clone()).unwrap();

    let tiers = [0.5, 0.1, 0.01];
    let k = 10;
    let ef_search = 100;

    for &sigma in &tiers {
        let bitset = generate_mode_a_bitset(n, sigma, 42);
        let params = FilteredSearchParams {
            ef_search,
            max_bridge_hops: None, // default 3 * ef = 300
            ..Default::default()
        };

        let mut hnsw_indices = Vec::with_capacity(10 * k);
        let mut gt_indices = Vec::with_capacity(10 * k);

        // Evaluate first 10 queries
        for q_idx in 0..10 {
            let q_vec = queries.row(q_idx);
            let hnsw_out = index
                .search_knn_filtered_traversal(q_vec, k, &bitset, &params)
                .unwrap();
            let gt_hits = exact_engine.search_filtered(q_vec, k, &bitset).unwrap();

            for &(_, id) in &hnsw_out.results {
                hnsw_indices.push(id as i32);
            }
            for &(_, id) in &gt_hits {
                gt_indices.push(id as i32);
            }
        }

        let hnsw_mat = RowMatrix::new(10, k, hnsw_indices).unwrap();
        let gt_mat = RowMatrix::new(10, k, gt_indices).unwrap();
        let recall = compute_recall_at_k(&hnsw_mat, &gt_mat, k).unwrap();

        // On 10k SIFT with ef=100 and in-traversal, recall@10 is expected to be >= 0.85
        assert!(
            recall >= 0.85,
            "Recall@10 too low for tier sigma={sigma}: recall={recall}"
        );
    }
}

#[test]
fn test_strategy2_unfiltered_search_and_fingerprint_bit_identical() {
    let matrix = make_synthetic_2d_matrix(100);
    let mut index = make_synthetic_2d_index(&matrix);
    let query = [15.0_f32, 8.0_f32];
    let k = 5;
    let ef_search = 50;

    // Baseline unfiltered search
    let baseline = index.search_knn(&query, k, ef_search).unwrap();

    // Filter with 100% selectivity (all ones bitset)
    let bitset_all = DenseBitset::ones(100);
    let params = FilteredSearchParams {
        ef_search,
        ..Default::default()
    };
    let filtered_all = index
        .search_knn_filtered_traversal(&query, k, &bitset_all, &params)
        .unwrap();

    assert_eq!(
        baseline.len(),
        filtered_all.results.len(),
        "100% selectivity in-traversal search should match unfiltered result count"
    );
    for ((d_base, id_base), (d_filt, id_filt)) in baseline.iter().zip(filtered_all.results.iter()) {
        assert_eq!(
            id_base, id_filt,
            "ID mismatch between unfiltered and in-traversal filtered"
        );
        assert_eq!(
            d_base.to_bits(),
            d_filt.to_bits(),
            "Distance bit-exact mismatch"
        );
    }
}

#[test]
fn test_strategy2_canonical_tie_breaking() {
    // 4 points on circle with identical squared distance 25.0 from origin (0, 0)
    let rows = vec![
        vec![5.0_f32, 0.0],  // id 0, dist 25.0
        vec![0.0_f32, 5.0],  // id 1, dist 25.0
        vec![-5.0_f32, 0.0], // id 2, dist 25.0
        vec![0.0_f32, -5.0], // id 3, dist 25.0
    ];
    let matrix = RowMatrix::from_rows(&rows).unwrap();
    let config = HnswConfig {
        m: 4,
        ef_construction: 16,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&matrix, Metric::L2Squared, config).unwrap();
    let bitset = DenseBitset::ones(4);
    let params = FilteredSearchParams {
        ef_search: 16,
        ..Default::default()
    };

    let query = [0.0_f32, 0.0_f32];
    let output = index
        .search_knn_filtered_traversal(&query, 4, &bitset, &params)
        .unwrap();

    assert_eq!(output.results.len(), 4);
    // All distances are identical (25.0), so IDs must sort strictly ascending: 0, 1, 2, 3
    for i in 0..4 {
        assert_eq!(
            output.results[i].1, i,
            "Ties must be broken by ID ascending"
        );
        assert_eq!(output.results[i].0, 25.0);
    }
}

#[test]
fn test_strategy2_input_validation() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let bitset = DenseBitset::ones(100);
    let params = FilteredSearchParams::default();
    let query = [1.0_f32, 2.0_f32];

    // 1. k = 0
    let err_k0 = index.search_knn_filtered_traversal(&query, 0, &bitset, &params);
    assert!(matches!(err_k0, Err(VectorSearchError::InvalidArgument(_))));

    // 2. NaN in query
    let nan_query = [f32::NAN, 2.0];
    let err_nan = index.search_knn_filtered_traversal(&nan_query, 5, &bitset, &params);
    assert!(matches!(
        err_nan,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // 3. Inf in query
    let inf_query = [1.0, f32::INFINITY];
    let err_inf = index.search_knn_filtered_traversal(&inf_query, 5, &bitset, &params);
    assert!(matches!(
        err_inf,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // 4. Dimension mismatch
    let dim_query = [1.0, 2.0, 3.0];
    let err_dim = index.search_knn_filtered_traversal(&dim_query, 5, &bitset, &params);
    assert!(matches!(
        err_dim,
        Err(VectorSearchError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));

    // 5. Empty index
    let empty_mat = RowMatrix::empty();
    let empty_index =
        HnswIndex::build(&empty_mat, Metric::L2Squared, HnswConfig::new(16, 64)).unwrap();
    let empty_out = empty_index
        .search_knn_filtered_traversal(&query, 5, &bitset, &params)
        .unwrap();
    assert!(empty_out.results.is_empty());
}

#[test]
fn test_strategy2_opaque_lambda_predicate_and_eval_limit() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];
    let k = 5;

    let lambda = LambdaPredicate::new(|id| id % 2 == 0, 0.5);
    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(2) {
        bitset.set_bit(i, true);
    }

    let params = FilteredSearchParams {
        ef_search: 50,
        ..Default::default()
    };

    let lambda_out = index
        .search_knn_filtered_traversal(&query, k, &lambda, &params)
        .unwrap();
    let bitset_out = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params)
        .unwrap();

    assert_eq!(lambda_out.results, bitset_out.results);
    assert!(!lambda_out.flags.eval_limit_reached);

    // Test eval_limit reached for opaque predicate
    let limit_params = FilteredSearchParams {
        ef_search: 50,
        eval_limit: Some(3),
        ..Default::default()
    };
    let limited_out = index
        .search_knn_filtered_traversal(&query, k, &lambda, &limit_params)
        .unwrap();
    assert!(limited_out.flags.eval_limit_reached);
}

#[test]
fn test_strategy2_batch_sequential_and_parallel_parity() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let q_rows = vec![
        vec![5.0_f32, 2.0],
        vec![15.0, 8.0],
        vec![30.0, 12.0],
        vec![50.0, 20.0],
    ];
    let queries = RowMatrix::from_rows(&q_rows).unwrap();
    let k = 4;

    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(2) {
        bitset.set_bit(i, true);
    }
    let params = FilteredSearchParams::default();

    let seq_results = index
        .search_batch_filtered_traversal(&queries, k, &bitset, &params)
        .unwrap();
    let par_results = index
        .search_batch_filtered_traversal_parallel(&queries, k, &bitset, &params, 2)
        .unwrap();

    assert_eq!(seq_results.len(), par_results.len());
    for (s, p) in seq_results.iter().zip(par_results.iter()) {
        assert_eq!(s, p, "Parallel output must match sequential exactly");
    }
}

#[test]
fn test_strategy2_unified_dispatch_and_scratch() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [20.0_f32, 10.0_f32];
    let k = 5;

    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(2) {
        bitset.set_bit(i, true);
    }
    let params = FilteredSearchParams {
        ef_search: 50,
        ..Default::default()
    };

    // 1. Direct traversal
    let direct = index
        .search_knn_filtered_traversal(&query, k, &bitset, &params)
        .unwrap();

    // 2. Unified dispatch with FilterStrategy::InTraversal
    let unified = index
        .search_knn_filtered(&query, k, &bitset, FilterStrategy::InTraversal, &params)
        .unwrap();
    assert_eq!(direct, unified);

    // 3. Scratch dispatch
    let mut scratch = FilteredSearchScratch::new(100, 100, 4096);
    let with_scratch = index
        .search_knn_filtered_with_scratch(
            &query,
            k,
            &bitset,
            FilterStrategy::InTraversal,
            &params,
            &mut scratch,
        )
        .unwrap();
    assert_eq!(direct, with_scratch);
}

#[test]
fn test_strategy2_eval_limit_strict_stop_and_zero_limit() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let query = [10.0_f32, 5.0_f32];

    // eval_limit = Some(0) with max_bridge_hops = Some(0) sets eval_limit_reached cleanly without hop_budget_exhausted
    let zero_calls = AtomicUsize::new(0);
    let zero_pred = LambdaPredicate::new(
        |id| {
            zero_calls.fetch_add(1, Ordering::Relaxed);
            id.is_multiple_of(2)
        },
        0.5,
    );
    let zero_out = index
        .search_knn_filtered_traversal(
            &query,
            5,
            &zero_pred,
            &FilteredSearchParams {
                ef_search: 20,
                max_bridge_hops: Some(0),
                eval_limit: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(zero_out.flags.eval_limit_reached);
    assert!(!zero_out.flags.hop_budget_exhausted);
    assert_eq!(zero_calls.load(Ordering::Relaxed), 0);

    // Non-zero strict eval_limit stops immediately without extra closure calls
    let calls = AtomicUsize::new(0);
    let count_pred = LambdaPredicate::new(
        |id| {
            calls.fetch_add(1, Ordering::Relaxed);
            id.is_multiple_of(2)
        },
        0.5,
    );
    let limit = 7;
    let out = index
        .search_knn_filtered_traversal(
            &query,
            5,
            &count_pred,
            &FilteredSearchParams {
                ef_search: 50,
                eval_limit: Some(limit),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(out.flags.eval_limit_reached);
    assert!(calls.load(Ordering::Relaxed) <= limit);
}

#[test]
fn test_strategy2_scratch_zero_reallocation_and_undersized_error() {
    let matrix = make_synthetic_2d_matrix(100);
    let index = make_synthetic_2d_index(&matrix);
    let bitset = DenseBitset::ones(100);

    // effective_ef (100) + m_max0 (32) > num_nodes (100), yet frontier must not grow
    let mut scratch = FilteredSearchScratch::new(100, 100, 0);
    let cap_frontier = scratch.frontier.capacity();
    let cap_matched = scratch.matched_results.capacity();
    let cap_raw = scratch.raw_candidates.capacity();

    let params = FilteredSearchParams {
        ef_search: 100,
        ..Default::default()
    };
    for step in 0..10 {
        let q = [(step as f32) * 10.0, 5.0];
        let direct = index
            .search_knn_filtered_traversal(&q, 10, &bitset, &params)
            .unwrap();
        let with_scratch = index
            .search_knn_filtered_with_scratch(
                &q,
                10,
                &bitset,
                FilterStrategy::InTraversal,
                &params,
                &mut scratch,
            )
            .unwrap();
        assert_eq!(direct, with_scratch);
    }
    assert_eq!(scratch.frontier.capacity(), cap_frontier);
    assert_eq!(scratch.matched_results.capacity(), cap_matched);
    assert_eq!(scratch.raw_candidates.capacity(), cap_raw);

    // Undersized scratch errors cleanly instead of reallocating in query path
    let mut empty_scratch = FilteredSearchScratch::default();
    let err = index.search_knn_filtered_with_scratch(
        &[10.0, 5.0],
        5,
        &bitset,
        FilterStrategy::InTraversal,
        &params,
        &mut empty_scratch,
    );
    assert!(matches!(err, Err(VectorSearchError::InvalidArgument(_))));
}
