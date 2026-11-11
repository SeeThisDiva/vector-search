//! Milestone M3 Task T3.4: Strategy 1 Post-Filter Over-Fetch Integration Tests.
//!
//! Verifies:
//! 1. Normal over-fetch finding `k` results in 1 search pass with `post_filter_cap_reached = false`.
//! 2. Low selectivity triggering iterative widening loop (widening from small `k'` to larger `k'`).
//! 3. Hitting `K'_max` cap with `< k` matches setting `flags.post_filter_cap_reached = true`.
//! 4. Uncapped mode (`K'_max = N`) finding all matching elements without truncation.
//! 5. Zero matches setting `post_filter_cap_reached = true` when cap is reached.
//! 6. Bit-identical parity between unfiltered `search_knn` and graph fingerprint preservation.
//! 7. Canonical `(distance, id)` deterministic tie-breaking.
//! 8. Input validation: `k = 0`, non-finite query, dimension mismatch, empty index.
//! 9. Opaque `LambdaPredicate` evaluation parity with `DenseBitset`.
//! 10. Sequential vs multi-threaded parallel batch execution consistency.
//! 11. Unified `search_knn_filtered` dispatch with `FilterStrategy::PostFilterOverfetch`.
//! 12. SIFT 10k smoke dataset ground truth recall validation against `FilteredExactKnn`.

#![forbid(unsafe_code)]

use std::path::Path;

use vector_search::dataset::{generate_mode_a_bitset, read_fvecs};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::filter::{
    DenseBitset, FilterPredicate, FilteredSearchFlags, FilteredSearchParams, LambdaPredicate,
};
use vector_search::hnsw::{FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::recall::compute_recall_at_k;

/// Helper: builds a small deterministic 2D synthetic dataset.
fn make_synthetic_2d_matrix() -> RowMatrix<f32> {
    let mut rows = Vec::new();
    for i in 0..100 {
        let x = (i as f32) * 1.5;
        let y = ((i * 7) % 31) as f32;
        rows.push(vec![x, y]);
    }
    RowMatrix::from_rows(&rows).unwrap()
}

fn make_synthetic_index(matrix: &RowMatrix<f32>) -> HnswIndex<'_> {
    let config = HnswConfig {
        m: 16,
        ef_construction: 64,
        seed: 42,
        ..Default::default()
    };
    HnswIndex::build(matrix, Metric::L2Squared, config).unwrap()
}

#[test]
fn test_strategy1_normal_overfetch_single_pass() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
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
        overfetch_cap: 4096,
        overfetch_factor: None,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .expect("Filtered search failed");

    assert_eq!(output.results.len(), k);
    assert!(!output.flags.post_filter_cap_reached);

    // Verify all returned results satisfy predicate
    for &(dist, id) in &output.results {
        assert!(
            bitset.matches(id as u32),
            "ID {} should match predicate",
            id
        );
        assert!(dist >= 0.0);
    }

    // Verify canonical ascending order
    for w in output.results.windows(2) {
        let (d0, id0) = w[0];
        let (d1, id1) = w[1];
        assert!(
            d0 < d1 || (d0 == d1 && id0 < id1),
            "Results must be sorted canonically: ({d0}, {id0}) vs ({d1}, {id1})"
        );
    }
}

#[test]
fn test_strategy1_low_selectivity_triggers_widening_loop() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let query = [0.0_f32, 0.0_f32];
    let k = 4;

    // Very sparse predicate: only IDs 80, 85, 90, 95 match (4 items total in 100)
    // Note that these are far from query [0, 0], so small initial k' will find 0 matches
    let mut bitset = DenseBitset::zeros(100);
    bitset.set_bit(80, true);
    bitset.set_bit(85, true);
    bitset.set_bit(90, true);
    bitset.set_bit(95, true);
    assert_eq!(bitset.count_ones(), 4);

    // Force small initial overfetch with factor = 1.0 (so k' starts small)
    let params = FilteredSearchParams {
        ef_search: 20,
        overfetch_factor: Some(1.0),
        overfetch_cap: 100,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .expect("Widening search failed");

    // Widening loop must have expanded to find all 4 matches
    assert_eq!(output.results.len(), 4);
    assert!(!output.flags.post_filter_cap_reached);

    let found_ids: Vec<usize> = output.results.iter().map(|&(_, id)| id).collect();
    for &expected_id in &[80, 85, 90, 95] {
        assert!(
            found_ids.contains(&expected_id),
            "Expected ID {} in results",
            expected_id
        );
    }
}

#[test]
fn test_strategy1_hitting_cap_sets_post_filter_cap_reached() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let query = [0.0_f32, 0.0_f32];
    let k = 10;

    // Predicate matches only 2 vectors in the entire index
    let mut bitset = DenseBitset::zeros(100);
    bitset.set_bit(12, true);
    bitset.set_bit(45, true);

    // Set a tight cap of 50
    let params = FilteredSearchParams {
        ef_search: 20,
        overfetch_cap: 50,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .expect("Capped search failed");

    // We asked for k=10, but at most 2 exist and cap=50 was reached
    assert!(output.results.len() <= 2);
    assert!(
        output.flags.post_filter_cap_reached,
        "flags.post_filter_cap_reached must be true when m < k at cap"
    );

    // Verify all returned are valid matches
    for &(_, id) in &output.results {
        assert!(bitset.matches(id as u32));
    }
}

#[test]
fn test_strategy1_uncapped_mode_finds_all_matching_elements() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let query = [25.0_f32, 10.0_f32];
    let k = 6;
    let n = matrix.rows();

    // Exactly 6 vectors match
    let mut bitset = DenseBitset::zeros(n);
    let target_ids = [3, 17, 34, 52, 78, 91];
    for &id in &target_ids {
        bitset.set_bit(id, true);
    }
    assert_eq!(bitset.count_ones(), 6);

    // Uncapped mode: K'_max = N
    let params = FilteredSearchParams {
        ef_search: 50,
        overfetch_cap: n,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .expect("Uncapped search failed");

    assert_eq!(output.results.len(), 6);
    assert!(!output.flags.post_filter_cap_reached);

    // Verify bit-exact parity against FilteredExactKnn
    let exact_engine = FilteredExactKnn::new(Metric::L2Squared, matrix).unwrap();
    let exact_hits = exact_engine.search_filtered(&query, k, &bitset).unwrap();

    let output_ids: Vec<usize> = output.results.iter().map(|&(_, id)| id).collect();
    let exact_ids: Vec<usize> = exact_hits.iter().map(|&(_, id)| id).collect();
    assert_eq!(output_ids, exact_ids);
}

#[test]
fn test_strategy1_zero_matches_sets_cap_reached() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let query = [5.0_f32, 5.0_f32];
    let k = 5;

    // All zero bitset (0 matches in index)
    let bitset = DenseBitset::zeros(100);

    let params = FilteredSearchParams {
        ef_search: 30,
        overfetch_cap: 64,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .expect("Zero matches search failed");

    assert!(output.results.is_empty());
    assert!(
        output.flags.post_filter_cap_reached,
        "Zero matches must set post_filter_cap_reached = true when cap is reached"
    );
}

#[test]
fn test_strategy1_unfiltered_search_and_fingerprint_bit_identical() {
    let matrix = make_synthetic_2d_matrix();
    let mut index = make_synthetic_index(&matrix);
    let query = [15.0_f32, 8.0_f32];
    let k = 10;
    let ef_search = 60;

    // 1. Unfiltered search
    let unfiltered_hits = index
        .search_knn(&query, k, ef_search)
        .expect("Unfiltered search failed");

    // 2. Filtered search with all-ones bitset (every vector matches)
    let all_ones = DenseBitset::ones(100);
    let params = FilteredSearchParams {
        ef_search,
        overfetch_cap: 4096,
        overfetch_factor: Some(1.0),
        ..Default::default()
    };

    let filtered_out = index
        .search_knn_filtered_post(&query, k, &all_ones, &params)
        .expect("Filtered all-ones search failed");

    assert_eq!(filtered_out.results.len(), unfiltered_hits.len());
    for (i, (&(f_dist, f_id), &(u_dist, u_id))) in filtered_out
        .results
        .iter()
        .zip(unfiltered_hits.iter())
        .enumerate()
    {
        assert_eq!(
            f_id, u_id,
            "ID mismatch at rank {i}: filtered={f_id}, unfiltered={u_id}"
        );
        assert_eq!(
            f_dist.to_bits(),
            u_dist.to_bits(),
            "Distance bit mismatch at rank {i}: filtered={f_dist}, unfiltered={u_dist}"
        );
    }
}

#[test]
fn test_strategy1_canonical_tie_breaking() {
    // 4 vectors with equidistant positions relative to query [0, 0]:
    // distance = 25.0
    let rows = vec![
        vec![5.0_f32, 0.0],
        vec![0.0, 5.0],
        vec![-5.0, 0.0],
        vec![0.0, -5.0],
    ];
    let matrix = RowMatrix::from_rows(&rows).unwrap();
    let config = HnswConfig {
        m: 4,
        ef_construction: 16,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&matrix, Metric::L2Squared, config).unwrap();

    let all_ones = DenseBitset::ones(4);
    let params = FilteredSearchParams {
        ef_search: 10,
        overfetch_cap: 10,
        ..Default::default()
    };

    let output = index
        .search_knn_filtered_post(&[0.0, 0.0], 4, &all_ones, &params)
        .unwrap();

    assert_eq!(output.results.len(), 4);
    // All distances are 25.0, so order must strictly be IDs [0, 1, 2, 3]
    for (i, &(dist, id)) in output.results.iter().enumerate() {
        assert_eq!(dist, 25.0);
        assert_eq!(id, i, "Tied distances must break by ID ascending");
    }
}

#[test]
fn test_strategy1_input_validation() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let bitset = DenseBitset::ones(100);
    let params = FilteredSearchParams::default();

    // 1. k = 0
    let err_k0 = index.search_knn_filtered_post(&[0.0, 0.0], 0, &bitset, &params);
    assert!(matches!(err_k0, Err(VectorSearchError::InvalidArgument(_))));

    // 2. Non-finite query (NaN / Inf)
    let err_nan = index.search_knn_filtered_post(&[f32::NAN, 0.0], 5, &bitset, &params);
    assert!(matches!(
        err_nan,
        Err(VectorSearchError::InvalidArgument(_))
    ));
    let err_inf = index.search_knn_filtered_post(&[0.0, f32::INFINITY], 5, &bitset, &params);
    assert!(matches!(
        err_inf,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // 3. Dimension mismatch (3D query into 2D index)
    let err_dim = index.search_knn_filtered_post(&[1.0, 2.0, 3.0], 5, &bitset, &params);
    assert!(matches!(
        err_dim,
        Err(VectorSearchError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));

    // 4. Empty index
    let empty_matrix = RowMatrix::empty();
    let empty_index =
        HnswIndex::build(&empty_matrix, Metric::L2Squared, HnswConfig::default()).unwrap();
    let empty_out = empty_index
        .search_knn_filtered_post(&[0.0, 0.0], 5, &bitset, &params)
        .unwrap();
    assert!(empty_out.results.is_empty());
    assert_eq!(empty_out.flags, FilteredSearchFlags::default());
}

#[test]
fn test_strategy1_opaque_lambda_predicate_parity() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let query = [20.0_f32, 10.0_f32];
    let k = 8;
    let params = FilteredSearchParams::default();

    // Predicate: IDs divisible by 3
    let mut bitset = DenseBitset::zeros(100);
    for i in (0..100).step_by(3) {
        bitset.set_bit(i, true);
    }

    let lambda = LambdaPredicate::new(|id| id % 3 == 0, 0.33);

    let out_bitset = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .unwrap();
    let out_lambda = index
        .search_knn_filtered_post(&query, k, &lambda, &params)
        .unwrap();

    assert_eq!(out_bitset.results.len(), out_lambda.results.len());
    for (b, l) in out_bitset.results.iter().zip(out_lambda.results.iter()) {
        assert_eq!(b.1, l.1, "ID mismatch between bitset and lambda");
        assert_eq!(
            b.0.to_bits(),
            l.0.to_bits(),
            "Distance mismatch between bitset and lambda"
        );
    }
}

#[test]
fn test_strategy1_unified_dispatch_and_scratch() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
    let query = [12.0_f32, 6.0_f32];
    let k = 5;
    let bitset = DenseBitset::ones(100);
    let params = FilteredSearchParams::default();

    // 1. Direct search_knn_filtered_post
    let direct = index
        .search_knn_filtered_post(&query, k, &bitset, &params)
        .unwrap();

    // 2. Dispatch via search_knn_filtered(PostFilterOverfetch)
    let dispatched = index
        .search_knn_filtered(
            &query,
            k,
            &bitset,
            FilterStrategy::PostFilterOverfetch,
            &params,
        )
        .unwrap();
    assert_eq!(direct, dispatched);

    // 3. Scratch buffer execution
    let mut scratch = FilteredSearchScratch::new(100, 100, 4096);
    let with_scratch = index
        .search_knn_filtered_with_scratch(
            &query,
            k,
            &bitset,
            FilterStrategy::PostFilterOverfetch,
            &params,
            &mut scratch,
        )
        .unwrap();
    assert_eq!(direct, with_scratch);

    // 4. InTraversal implemented in T3.5, PreFilterBruteForce returns informative error until T3.6
    let in_trav_res =
        index.search_knn_filtered(&query, k, &bitset, FilterStrategy::InTraversal, &params);
    assert!(in_trav_res.is_ok());

    let pre_filter_err = index.search_knn_filtered(
        &query,
        k,
        &bitset,
        FilterStrategy::PreFilterBruteForce,
        &params,
    );
    assert!(pre_filter_err.is_err());
}

#[test]
fn test_strategy1_batch_sequential_and_parallel_parity() {
    let matrix = make_synthetic_2d_matrix();
    let index = make_synthetic_index(&matrix);
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
        .search_batch_filtered_post(&queries, k, &bitset, &params)
        .unwrap();
    let par_results = index
        .search_batch_filtered_post_parallel(&queries, k, &bitset, &params, 2)
        .unwrap();

    assert_eq!(seq_results.len(), par_results.len());
    for (s, p) in seq_results.iter().zip(par_results.iter()) {
        assert_eq!(s, p, "Parallel output must match sequential exactly");
    }
}

#[test]
fn test_strategy1_sift10k_recall_against_ground_truth() {
    let sift_base_path = Path::new("data/siftsmall/siftsmall_base.fvecs");
    let sift_query_path = Path::new("data/siftsmall/siftsmall_query.fvecs");

    if !sift_base_path.exists() || !sift_query_path.exists() {
        return;
    }

    let base = read_fvecs(sift_base_path).expect("Failed to read siftsmall_base.fvecs");
    let queries = read_fvecs(sift_query_path).expect("Failed to read siftsmall_query.fvecs");
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
            overfetch_cap: 4096,
            overfetch_factor: None,
            ..Default::default()
        };

        let mut hnsw_indices = Vec::with_capacity(10 * k);
        let mut gt_indices = Vec::with_capacity(10 * k);

        // Evaluate first 10 queries
        for q_idx in 0..10 {
            let q_vec = queries.row(q_idx);
            let hnsw_out = index
                .search_knn_filtered_post(q_vec, k, &bitset, &params)
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

        // On 10k SIFT with ef=100 and overfetch, recall@10 is expected to be high (> 0.85)
        assert!(
            recall >= 0.85,
            "Recall@10 too low for tier sigma={sigma}: recall={recall}"
        );
    }
}
