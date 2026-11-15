//! Tests for zero-overhead monomorphized filtered search counters (`T3.8` Commit 2).

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use vector_search::dataset::generate_mode_a_bitset;
use vector_search::distance::Metric;
use vector_search::filter::{DenseBitset, FilteredSearchParams, LambdaPredicate};
use vector_search::hnsw::{FilterStrategy, FilteredSearchScratch, HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;

fn make_deterministic_matrix(n: usize, dims: usize) -> RowMatrix<f32> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(dims);
        for d in 0..dims {
            let v = (((i * 31 + d * 17) % 1000) as f32) / 100.0;
            row.push(v);
        }
        rows.push(row);
    }
    RowMatrix::from_rows(&rows).unwrap()
}

fn make_index(matrix: &RowMatrix<f32>) -> HnswIndex<'_> {
    let config = HnswConfig {
        m: 12,
        ef_construction: 48,
        seed: 7,
        ..Default::default()
    };
    HnswIndex::build(matrix, Metric::L2Squared, config).unwrap()
}

#[test]
fn test_counters_bit_exact_parity_and_invariants_across_strategies() {
    let n = 400;
    let dims = 8;
    let matrix = make_deterministic_matrix(n, dims);
    let index = make_index(&matrix);
    let bitset = generate_mode_a_bitset(n, 0.15, 42);
    let popcount = bitset.count_ones() as u64;
    let k = 10;

    let mut scratch_timed = FilteredSearchScratch::new(n, 128, n);
    let mut scratch_cnt = FilteredSearchScratch::new(n, 128, n);

    let params = FilteredSearchParams {
        ef_search: 32,
        overfetch_cap: n,
        allow_bruteforce_fallback: false,
        ..Default::default()
    };

    for q_idx in [0_usize, 7, 42] {
        let query = matrix.row(q_idx);
        for strategy in [
            FilterStrategy::PostFilterOverfetch,
            FilterStrategy::InTraversal,
            FilterStrategy::PreFilterBruteForce,
        ] {
            let out_timed = index
                .search_knn_filtered_with_scratch(
                    query,
                    k,
                    &bitset,
                    strategy,
                    &params,
                    &mut scratch_timed,
                )
                .unwrap();

            let (out_cnt, counters) = index
                .search_knn_filtered_with_counters(
                    query,
                    k,
                    &bitset,
                    strategy,
                    &params,
                    &mut scratch_cnt,
                )
                .unwrap();

            assert_eq!(out_timed.results, out_cnt.results);
            assert_eq!(out_timed.flags, out_cnt.flags);

            assert_eq!(
                counters.total_distance_evals,
                counters.valid_distance_evals + counters.bridge_distance_evals
            );
            assert_eq!(
                counters.total_distance_evals,
                counters.upper_layer_distance_evals + counters.visited_nodes
            );
            assert!(counters.total_distance_evals > 0);
            assert!(counters.eta_graph() >= 0.0 && counters.eta_graph() <= 1.0);

            if matches!(strategy, FilterStrategy::PreFilterBruteForce) {
                assert_eq!(counters.bridge_distance_evals, 0);
                assert_eq!(counters.upper_layer_distance_evals, 0);
                assert_eq!(counters.valid_distance_evals, popcount);
                assert_eq!(counters.total_distance_evals, popcount);
                assert_eq!(counters.visited_matching_nodes, popcount);
                assert!((counters.eta_graph() - 1.0).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn test_counters_do_not_inflate_opaque_closure_evaluations() {
    let n = 300;
    let dims = 6;
    let matrix = make_deterministic_matrix(n, dims);
    let index = make_index(&matrix);
    let query = matrix.row(5);
    let k = 10;

    let params = FilteredSearchParams {
        ef_search: 24,
        overfetch_cap: 150,
        ..FilteredSearchParams::with_opaque_eval_limit()
    };

    for strategy in [
        FilterStrategy::PostFilterOverfetch,
        FilterStrategy::InTraversal,
    ] {
        let mut scratch_timed = FilteredSearchScratch::new(n, 200, n);
        let mut scratch_cnt = FilteredSearchScratch::new(n, 200, n);

        let calls_timed = AtomicUsize::new(0);
        let pred_timed = LambdaPredicate::new(
            |id| {
                calls_timed.fetch_add(1, Ordering::Relaxed);
                id.is_multiple_of(5)
            },
            0.20,
        );

        let calls_cnt = AtomicUsize::new(0);
        let pred_cnt = LambdaPredicate::new(
            |id| {
                calls_cnt.fetch_add(1, Ordering::Relaxed);
                id.is_multiple_of(5)
            },
            0.20,
        );

        let out_timed = index
            .search_knn_filtered_with_scratch(
                query,
                k,
                &pred_timed,
                strategy,
                &params,
                &mut scratch_timed,
            )
            .unwrap();
        let (out_cnt, counters) = index
            .search_knn_filtered_with_counters(
                query,
                k,
                &pred_cnt,
                strategy,
                &params,
                &mut scratch_cnt,
            )
            .unwrap();

        assert_eq!(out_timed.results, out_cnt.results);
        assert_eq!(out_timed.flags, out_cnt.flags);
        assert_eq!(
            calls_timed.load(Ordering::Relaxed),
            calls_cnt.load(Ordering::Relaxed),
            "Counter mode must not perform extra predicate.matches calls for {strategy:?}"
        );
        assert_eq!(
            counters.total_distance_evals,
            counters.valid_distance_evals + counters.bridge_distance_evals
        );
    }
}

#[test]
fn test_counters_accumulate_when_fallback_triggers() {
    let n = 200;
    let dims = 4;
    let matrix = make_deterministic_matrix(n, dims);
    let index = make_index(&matrix);
    let query = matrix.row(0);

    let mut bitset = DenseBitset::zeros(n);
    for i in (0..10).map(|j| j * 19) {
        bitset.set_bit(i, true);
    }
    let popcount = bitset.count_ones() as u64;

    let params = FilteredSearchParams {
        ef_search: 12,
        overfetch_cap: 10,
        max_bridge_hops: Some(0),
        allow_bruteforce_fallback: true,
        ..Default::default()
    };

    let mut scratch = FilteredSearchScratch::new(n, 64, n);
    let (out, counters) = index
        .search_knn_filtered_with_counters(
            query,
            10,
            &bitset,
            FilterStrategy::InTraversal,
            &params,
            &mut scratch,
        )
        .unwrap();

    assert!(out.flags.used_bruteforce_fallback);
    assert!(counters.total_distance_evals > popcount);
    assert_eq!(
        counters.total_distance_evals,
        counters.valid_distance_evals + counters.bridge_distance_evals
    );
    assert_eq!(
        counters.total_distance_evals,
        counters.upper_layer_distance_evals + counters.visited_nodes
    );
}
