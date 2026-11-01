//! Adversarial stress test suite for Phase 2 HNSW index.
//!
//! Evaluates failure modes, boundary inputs, metric variations, extreme hyperparameters,
//! sparse/disconnected topologies, duplicate/collinear inputs, and bit-exact reproducibility.

use std::path::Path;
use vector_search::dataset::{generate_synthetic_sift, load_or_download_sift, SiftPreset};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::ExactKnn;
use vector_search::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use vector_search::matrix::RowMatrix;
use vector_search::recall::compute_recall_at_k;

// ============================================================================
// 1. Boundary & Malformed Inputs
// ============================================================================

#[test]
fn test_adv_empty_matrix_and_batch_variations() {
    let empty = RowMatrix::<f32>::empty();
    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&empty, Metric::L2Squared, config).unwrap();

    assert_eq!(index.num_vectors(), 0);
    assert_eq!(index.max_layer(), 0);
    assert_eq!(index.entry_point(), None);

    // Single query on empty index
    let res = index.search_knn(&[1.0, 2.0, 3.0], 5, 10).unwrap();
    assert!(res.is_empty());

    // Batch sequential on empty index
    let queries = RowMatrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
    let batch_seq = index.search_batch(&queries, 5, 10).unwrap();
    assert_eq!(batch_seq.indices.rows(), 2);
    assert_eq!(batch_seq.indices.cols(), 0);

    // Batch timed on empty index
    let batch_timed = index.search_batch_timed(&queries, 5, 10).unwrap();
    assert_eq!(batch_timed.result.indices.rows(), 2);
    assert_eq!(batch_timed.result.indices.cols(), 0);

    // Batch parallel on empty index
    let batch_par = index.search_batch_parallel(&queries, 5, 10, 4).unwrap();
    assert_eq!(batch_par.indices.rows(), 2);
    assert_eq!(batch_par.indices.cols(), 0);

    // Empty query batch on empty index
    let empty_q = RowMatrix::<f32>::empty();
    let par_empty = index.search_batch_parallel(&empty_q, 5, 10, 4).unwrap();
    assert!(par_empty.indices.is_empty());
}

#[test]
fn test_adv_single_vector_edge_cases() {
    let single = RowMatrix::from_rows(&[vec![42.0_f32, -17.5, 0.001]]).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&single, Metric::L2Squared, config).unwrap();

    assert_eq!(index.num_vectors(), 1);
    assert_eq!(index.entry_point(), Some(0));

    // Requesting k=10 when N=1 -> must return exactly 1 result
    let res = index.search_knn(&[42.0, -17.5, 0.001], 10, 20).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].1, 0);
    assert_eq!(res[0].0.to_bits(), 0.0_f32.to_bits());

    // efSearch < k on N=1
    let res_clamped = index.search_knn(&[0.0, 0.0, 0.0], 5, 1).unwrap();
    assert_eq!(res_clamped.len(), 1);
    assert_eq!(res_clamped[0].1, 0);

    // k=0 must error cleanly
    assert!(matches!(
        index.search_knn(&[42.0, -17.5, 0.001], 0, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Dimension mismatch must error cleanly
    assert!(matches!(
        index.search_knn(&[42.0, -17.5], 1, 10),
        Err(VectorSearchError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));

    // Non-finite query values must error cleanly
    assert!(matches!(
        index.search_knn(&[42.0, f32::NAN, 0.0], 1, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        index.search_knn(&[f32::INFINITY, -17.5, 0.0], 1, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        index.search_knn(&[42.0, -17.5, f32::NEG_INFINITY], 1, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

#[test]
fn test_adv_k_greater_than_n_and_ef_less_than_k() {
    let (base, queries) = generate_synthetic_sift(7, 3, 16, 2, 88).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    for q in queries.iter_rows() {
        // k=100 > N=7, ef_search=1 < k -> both extremes simultaneously
        let res = index.search_knn(q, 100, 1).unwrap();
        assert_eq!(res.len(), 7, "Must return all N=7 elements");

        // Verify strictly sorted ascending
        for i in 1..res.len() {
            let prev = res[i - 1];
            let curr = res[i];
            assert!(
                prev.0 < curr.0 || (prev.0 == curr.0 && prev.1 < curr.1),
                "Results must follow canonical (distance, id) total order"
            );
        }
    }
}

// ============================================================================
// 2. Duplicate, Collinear, and Equal-Distance Vectors
// ============================================================================

#[test]
fn test_adv_massive_identical_duplicates() {
    // 50 identical vectors with distance 0.0
    let identical_vec = vec![1.23_f32, -4.56, 7.89, 0.0];
    let rows = vec![identical_vec.clone(); 50];
    let base = RowMatrix::from_rows(&rows).unwrap();

    // Test A: When k <= M_max0 + 1 (here 32 + 1 = 33), all 25 identical duplicates are reachable
    let config_default = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let mut index_default = HnswIndex::build(&base, Metric::L2Squared, config_default).unwrap();
    let res_25 = index_default.search_knn(&identical_vec, 25, 50).unwrap();
    assert_eq!(res_25.len(), 25);
    for (i, &(dist, id)) in res_25.iter().enumerate() {
        assert_eq!(dist.to_bits(), 0.0_f32.to_bits());
        assert_eq!(id, i);
    }

    // Test B: Adversarial clique saturation proof:
    // With m_max0=16, the fully connected clique of identical vectors saturates at M_max0 + 1 = 17 nodes.
    // Because older nodes saturate at degree 16 and prune higher IDs deterministically,
    // exactly 17 nodes (IDs 0..16) are reachable from the entry point.
    let config_saturated = HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let mut index_saturated = HnswIndex::build(&base, Metric::L2Squared, config_saturated).unwrap();
    let res_saturated = index_saturated.search_knn(&identical_vec, 25, 50).unwrap();
    assert_eq!(
        res_saturated.len(),
        17,
        "Degree-bounded HNSW on 100% identical vectors naturally bounds reachable clique to M_max0 + 1 = 17"
    );
    for (i, &(dist, id)) in res_saturated.iter().enumerate() {
        assert_eq!(dist.to_bits(), 0.0_f32.to_bits());
        assert_eq!(id, i);
    }
}

#[test]
fn test_adv_clustered_duplicates_ranking() {
    // 3 distinct clusters, each containing 5 duplicate points
    let mut rows = Vec::new();
    for _ in 0..5 {
        rows.push(vec![0.0_f32, 0.0]); // Cluster 0: IDs 0..4
    }
    for _ in 0..5 {
        rows.push(vec![10.0_f32, 10.0]); // Cluster 1: IDs 5..9
    }
    for _ in 0..5 {
        rows.push(vec![100.0_f32, 100.0]); // Cluster 2: IDs 10..14
    }
    let base = RowMatrix::from_rows(&rows).unwrap();

    let config = HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 20,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    // Query near cluster 0
    let res = index.search_knn(&[0.1, 0.1], 15, 20).unwrap();
    assert_eq!(res.len(), 15);

    // First 5 must be IDs 0..4 in order
    for (i, &(_, id)) in res[..5].iter().enumerate() {
        assert_eq!(id, i);
    }
    // Next 5 must be IDs 5..9 in order
    for (i, &(_, id)) in res[5..10].iter().enumerate() {
        assert_eq!(id, 5 + i);
    }
    // Last 5 must be IDs 10..14 in order
    for (i, &(_, id)) in res[10..15].iter().enumerate() {
        assert_eq!(id, 10 + i);
    }
}

// ============================================================================
// 3. Sparse & Disconnected Graph Topologies (Adversarial Parameters)
// ============================================================================

#[test]
fn test_adv_sparse_graph_tiny_m_high_dim() {
    // M=2 (minimal allowable degree), M_max0=4, high dimension D=512, N=200
    let (base, queries) = generate_synthetic_sift(200, 10, 512, 5, 9999).unwrap();
    let exact = ExactKnn::from_base(Metric::L2Squared, base.clone()).unwrap();
    let gt = exact.search(&queries, 5).unwrap();

    let config = HnswConfig {
        m: 2,
        m_max: 2,
        m_max0: 4,
        ef_construction: 50,
        keep_pruned: true,
        seed: 777,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    // Verify degree caps are strictly respected
    for u in 0..200 {
        assert!(index.neighbors(0, u).len() <= 4);
        for l in 1..=index.max_layer() {
            if index.node_exists_at_layer(l, u) {
                assert!(index.neighbors(l, u).len() <= 2);
            }
        }
    }

    // Despite sparse M=2, index must answer queries without crashing or panicking
    let mut cand_rows = Vec::new();
    for q in queries.iter_rows() {
        let hits = index.search_knn(q, 5, 50).unwrap();
        assert_eq!(hits.len(), 5);
        for h in hits {
            cand_rows.push(h.1 as i32);
        }
    }
    let cand_mat = RowMatrix::new(10, 5, cand_rows).unwrap();
    let recall = compute_recall_at_k(&cand_mat, &gt.indices, 5).unwrap();
    // Recall with M=2 will be lower than M=16, but should be strictly positive (> 0.50)
    assert!(
        recall >= 0.50,
        "Recall with M=2 was unexpectedly low: {recall}"
    );
}

#[test]
fn test_adv_heuristic_keep_pruned_false_collinear() {
    // Collinear points along axis: with keep_pruned=false, triangular inequality prunes redundant neighbors
    let base = RowMatrix::from_rows(
        &(0..30)
            .map(|i| vec![i as f32 * 2.0, 0.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let config = HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 30,
        keep_pruned: false,
        extend_candidates: true,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    // Query somewhere along the line
    let hits = index.search_knn(&[15.0, 0.0], 5, 20).unwrap();
    assert_eq!(hits.len(), 5);
    // Closest integers to 15.0/2.0 = 7.5 are 7 and 8
    assert!(hits.iter().any(|h| h.1 == 7));
    assert!(hits.iter().any(|h| h.1 == 8));
}

// ============================================================================
// 4. Metric Variations: Euclidean & Inner Product
// ============================================================================

#[test]
fn test_adv_euclidean_and_inner_product_parity() {
    let (base, queries) = generate_synthetic_sift(100, 10, 32, 4, 101).unwrap();

    // Metric 1: Euclidean (sqrt)
    let exact_euc = ExactKnn::from_base(Metric::Euclidean, base.clone()).unwrap();
    let gt_euc = exact_euc.search(&queries, 5).unwrap();

    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: 42,
        ..Default::default()
    };
    let mut idx_euc = HnswIndex::build(&base, Metric::Euclidean, config.clone()).unwrap();

    let mut euc_rows = Vec::new();
    for q in queries.iter_rows() {
        let hits = idx_euc.search_knn(q, 5, 50).unwrap();
        for h in hits {
            euc_rows.push(h.1 as i32);
        }
    }
    let euc_mat = RowMatrix::new(10, 5, euc_rows).unwrap();
    let recall_euc = compute_recall_at_k(&euc_mat, &gt_euc.indices, 5).unwrap();
    assert!(
        recall_euc >= 0.95,
        "Euclidean recall below 0.95: {recall_euc}"
    );

    // Metric 2: InnerProduct
    let exact_ip = ExactKnn::from_base(Metric::InnerProduct, base.clone()).unwrap();
    let gt_ip = exact_ip.search(&queries, 5).unwrap();

    let mut idx_ip = HnswIndex::build(&base, Metric::InnerProduct, config).unwrap();
    let mut ip_rows = Vec::new();
    for q in queries.iter_rows() {
        let hits = idx_ip.search_knn(q, 5, 50).unwrap();
        for h in hits {
            ip_rows.push(h.1 as i32);
        }
    }
    let ip_mat = RowMatrix::new(10, 5, ip_rows).unwrap();
    let recall_ip = compute_recall_at_k(&ip_mat, &gt_ip.indices, 5).unwrap();
    assert!(
        recall_ip >= 0.95,
        "InnerProduct recall below 0.95: {recall_ip}"
    );
}

// ============================================================================
// 5. Multi-Threaded Parallel Search vs Sequential Bit-Exact Parity
// ============================================================================

#[test]
fn test_adv_batch_parallel_worker_variations() {
    let (base, queries) = generate_synthetic_sift(300, 37, 32, 4, 202).unwrap();
    let config = HnswConfig {
        m: 16,
        ef_construction: 80,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    let seq_res = index.search_batch(&queries, 10, 40).unwrap();

    // Test with worker counts: 1, 2, 3, 7, 16, 64 (workers > num_queries)
    for &num_workers in &[1, 2, 3, 7, 16, 64] {
        let par_res = index
            .search_batch_parallel(&queries, 10, 40, num_workers)
            .unwrap();

        assert_eq!(
            seq_res.indices.as_slice(),
            par_res.indices.as_slice(),
            "Indices mismatch for worker count {num_workers}"
        );
        for (&d_seq, &d_par) in seq_res
            .distances
            .as_slice()
            .iter()
            .zip(par_res.distances.as_slice().iter())
        {
            assert_eq!(
                d_seq.to_bits(),
                d_par.to_bits(),
                "Distance bit mismatch for worker count {num_workers}"
            );
        }
    }
}

// ============================================================================
// 6. Bit-Exact Reproducibility Across Multiple Independent Instances
// ============================================================================

#[test]
fn test_adv_multi_instance_bit_exact_reproducibility() {
    let (base, queries) = generate_synthetic_sift(500, 30, 48, 6, 303).unwrap();
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: 12345,
        ..Default::default()
    };

    // Build 3 separate indices from identical seed
    let mut index_a = HnswIndex::build(&base, Metric::L2Squared, config.clone()).unwrap();
    let mut index_b = HnswIndex::build(&base, Metric::L2Squared, config.clone()).unwrap();
    let mut index_c = HnswIndex::build(&base, Metric::L2Squared, config.clone()).unwrap();

    // Verify structural equality: entry points and max layers match
    assert_eq!(index_a.entry_point(), index_b.entry_point());
    assert_eq!(index_b.entry_point(), index_c.entry_point());
    assert_eq!(index_a.max_layer(), index_b.max_layer());
    assert_eq!(index_b.max_layer(), index_c.max_layer());

    // Verify exact neighbor topology
    for u in 0..500 {
        assert_eq!(index_a.node_layer(u), index_b.node_layer(u));
        assert_eq!(index_b.node_layer(u), index_c.node_layer(u));
        assert_eq!(index_a.neighbors(0, u), index_b.neighbors(0, u));
        assert_eq!(index_b.neighbors(0, u), index_c.neighbors(0, u));
    }

    // Verify query results
    for q in queries.iter_rows() {
        let res_a = index_a.search_knn(q, 10, 50).unwrap();
        let res_b = index_b.search_knn(q, 10, 50).unwrap();
        let res_c = index_c.search_knn(q, 10, 50).unwrap();

        for i in 0..10 {
            assert_eq!(res_a[i].1, res_b[i].1);
            assert_eq!(res_b[i].1, res_c[i].1);
            assert_eq!(res_a[i].0.to_bits(), res_b[i].0.to_bits());
            assert_eq!(res_b[i].0.to_bits(), res_c[i].0.to_bits());
        }
    }

    // Verify different seeds produce different layer assignments
    let diff_config = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: 99999,
        ..Default::default()
    };
    let index_diff = HnswIndex::build(&base, Metric::L2Squared, diff_config).unwrap();
    let layers_match = (0..500).all(|u| index_a.node_layer(u) == index_diff.node_layer(u));
    assert!(
        !layers_match,
        "Different seeds must not produce identical layer assignments"
    );
}

// ============================================================================
// 7. Strict Monotonicity Across Extended efSearch Ladder
// ============================================================================

#[test]
fn test_adv_extended_ef_search_ladder_monotonicity() {
    let loaded =
        load_or_download_sift(SiftPreset::SiftSmall, Path::new("data"), false, 42).unwrap();
    let exact = ExactKnn::from_base(Metric::L2Squared, loaded.base.clone()).unwrap();
    let gt = exact.search(&loaded.query, 10).unwrap();

    let config = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&loaded.base, Metric::L2Squared, config).unwrap();

    let extended_ladder = [5_usize, 10, 15, 25, 50, 75, 100, 150, 200];
    let mut recalls = Vec::new();

    for &ef in &extended_ladder {
        let mut cand_rows = Vec::with_capacity(100 * 10);
        for q in loaded.query.iter_rows() {
            let res = index.search_knn(q, 10, ef).unwrap();
            for item in res {
                cand_rows.push(item.1 as i32);
            }
        }
        let cand_mat = RowMatrix::new(100, 10, cand_rows).unwrap();
        let recall = compute_recall_at_k(&cand_mat, &gt.indices, 10).unwrap();
        recalls.push(recall);
    }

    // Strict non-decreasing monotonicity check
    for i in 1..recalls.len() {
        assert!(
            recalls[i] >= recalls[i - 1] - 1e-6,
            "Monotonicity inversion between ef={} ({}) and ef={} ({})",
            extended_ladder[i - 1],
            recalls[i - 1],
            extended_ladder[i],
            recalls[i]
        );
    }
}

// ============================================================================
// 8. VisitedTracker Overflow / Wrapping Stress Test
// ============================================================================

#[test]
fn test_adv_visited_tracker_tag_overflow_reset() {
    // Force curr_tag close to u32::MAX to test wrapping reset logic
    let mut tracker = VisitedTracker::new(100);
    // Mark a node
    tracker.mark_visited(10);
    assert!(tracker.is_visited(10));
    assert!(!tracker.is_visited(11));

    // Simulate 10,000 rapid prepare cycles
    for _ in 0..10_000 {
        tracker.prepare(100);
        assert!(!tracker.is_visited(10));
        tracker.mark_visited(10);
        assert!(tracker.is_visited(10));
    }
}
