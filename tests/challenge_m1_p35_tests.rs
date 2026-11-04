//! Empirical Challenger Stress Tests for Milestone 1 (M1_P35)
//!
//! Stress-tests:
//! 1. Distance evaluation count identities across scales, metrics, and hyperparameter topologies
//! 2. Zero-cost verification when `--features instrument` is disabled vs enabled
//! 3. Canonical (distance, id) tie-breaking under duplicate, equidistant, and collinear inputs

use std::cmp::Ordering;
use vector_search::dataset::generate_synthetic_sift;
use vector_search::distance::Metric;
use vector_search::exact_knn::ExactKnn;
use vector_search::hnsw::{Candidate, HeuristicScratch, HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;

// ============================================================================
// 1. Distance Evaluation Count Identity Invariants Across Scales & Topologies
// ============================================================================

#[test]
fn test_challenge_distance_evals_identity_empty_and_single_node() {
    let empty_base = RowMatrix::<f32>::new(0, 16, vec![]).unwrap();
    let cfg = HnswConfig {
        seed: 42,
        ..Default::default()
    };
    let idx_empty = HnswIndex::build(&empty_base, Metric::L2Squared, cfg.clone()).unwrap();

    #[cfg(feature = "instrument")]
    {
        let stats = idx_empty.build_stats().expect("build_stats present");
        assert_eq!(stats.inserts_count, 0);
        assert_eq!(stats.distance_evals_total, 0);
        assert_eq!(stats.distance_evals_search, 0);
        assert_eq!(stats.distance_evals_heuristic, 0);
        assert_eq!(stats.distance_evals_shrink, 0);
        assert_eq!(stats.search_layer_calls, 0);
        assert_eq!(stats.heuristic_calls, 0);
        assert_eq!(stats.shrink_calls, 0);
        assert_eq!(
            stats.distance_evals_total,
            stats.distance_evals_search
                + stats.distance_evals_heuristic
                + stats.distance_evals_shrink
        );
    }
    #[cfg(not(feature = "instrument"))]
    {
        assert!(idx_empty.build_stats().is_none());
    }

    // Single vector (N = 1)
    let single_base = RowMatrix::from_rows(&[vec![1.0; 16]]).unwrap();
    let _idx_single = HnswIndex::build(&single_base, Metric::L2Squared, cfg).unwrap();

    #[cfg(feature = "instrument")]
    {
        let stats = _idx_single.build_stats().expect("build_stats present");
        assert_eq!(stats.inserts_count, 1);
        assert_eq!(stats.distance_evals_total, 0);
        assert_eq!(stats.distance_evals_search, 0);
        assert_eq!(stats.distance_evals_heuristic, 0);
        assert_eq!(stats.distance_evals_shrink, 0);
        assert_eq!(stats.search_layer_calls, 0);
        assert_eq!(stats.heuristic_calls, 0);
        assert_eq!(stats.shrink_calls, 0);
        assert_eq!(
            stats.distance_evals_total,
            stats.distance_evals_search
                + stats.distance_evals_heuristic
                + stats.distance_evals_shrink
        );
    }
}

#[test]
fn test_challenge_distance_evals_identity_analytical_n2() {
    let (base, _) = generate_synthetic_sift(2, 1, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        m: 16,
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let _index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();

    #[cfg(feature = "instrument")]
    {
        let stats = _index.build_stats().expect("build_stats present");
        let l0 = _index.node_layer(0);

        assert_eq!(stats.inserts_count, 2);
        assert_eq!(stats.search_layer_calls as usize, l0 + 1);
        assert_eq!(stats.distance_evals_search as usize, l0 + 1);
        assert_eq!(stats.distance_evals_heuristic, 0);
        assert_eq!(stats.distance_evals_shrink, 0);
        assert_eq!(stats.shrink_calls, 0);
        assert_eq!(
            stats.heuristic_calls,
            (l0.min(_index.node_layer(1)) + 1) as u64
        );
        assert_eq!(
            stats.distance_evals_total,
            stats.distance_evals_search
                + stats.distance_evals_heuristic
                + stats.distance_evals_shrink
        );
    }
}

#[test]
fn test_challenge_distance_evals_identity_multi_scale_sweep() {
    let scales = [10_usize, 25, 50, 100, 250, 500, 1000];
    let metrics = [Metric::L2Squared, Metric::Euclidean, Metric::InnerProduct];

    for &scale in &scales {
        for &metric in &metrics {
            let (base, _) = generate_synthetic_sift(scale, 1, 16, 4, 1000 + scale as u64).unwrap();

            // Test with both keep_pruned=true and keep_pruned=false
            for &keep_pruned in &[true, false] {
                // Test with both extend_candidates=false and extend_candidates=true
                for &extend_candidates in &[false, true] {
                    let cfg = HnswConfig {
                        m: 16,
                        m_max0: Some(32),
                        ef_construction: 40,
                        seed: 42,
                        extend_candidates,
                        keep_pruned,
                        ..Default::default()
                    };

                    let _index = HnswIndex::build(&base, metric, cfg).unwrap();

                    #[cfg(feature = "instrument")]
                    {
                        let stats = _index.build_stats().expect("build_stats present");

                        // 1. Core count identity invariant
                        assert_eq!(
                            stats.distance_evals_total,
                            stats.distance_evals_search
                                + stats.distance_evals_heuristic
                                + stats.distance_evals_shrink,
                            "Distance evals identity failed for scale={scale}, metric={metric:?}, kp={keep_pruned}, ec={extend_candidates}"
                        );

                        // 2. Call counts invariants
                        assert_eq!(
                            stats.heuristic_calls,
                            stats.heuristic_insert_calls + stats.heuristic_shrink_calls,
                            "Heuristic call count decomposition failed for scale={scale}"
                        );
                        assert_eq!(
                            stats.shrink_calls, stats.heuristic_shrink_calls,
                            "Shrink calls != heuristic shrink calls for scale={scale}"
                        );

                        // 3. Distribution sanity checks
                        assert_eq!(stats.inserts_count, scale);
                        assert!(
                            stats.dist_evals_per_insert_min <= stats.dist_evals_per_insert_max,
                            "min > max: min={}, max={}",
                            stats.dist_evals_per_insert_min,
                            stats.dist_evals_per_insert_max
                        );

                        let expected_mean = stats.distance_evals_total as f64 / scale as f64;
                        assert!(
                            (stats.dist_evals_per_insert_mean - expected_mean).abs() < 1e-6,
                            "Mean mismatch: reported {}, expected {}",
                            stats.dist_evals_per_insert_mean,
                            expected_mean
                        );

                        // 4. Shrink degree threshold invariant:
                        // If scale <= M (16) and M0 (32), degree shrink CANNOT trigger!
                        if scale <= 16 {
                            assert_eq!(
                                stats.shrink_calls, 0,
                                "Shrink calls should be 0 when scale ({scale}) <= M (16)"
                            );
                            assert_eq!(
                                stats.distance_evals_shrink, 0,
                                "Shrink evals should be 0 when scale ({scale}) <= M (16)"
                            );
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2. Zero-Cost Verification When Feature `instrument` is Disabled
// ============================================================================

#[test]
fn test_challenge_zero_cost_memory_and_stats() {
    let (base, _) = generate_synthetic_sift(50, 1, 16, 2, 99).unwrap();
    let cfg = HnswConfig {
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();

    #[cfg(not(feature = "instrument"))]
    {
        // When instrument feature is OFF:
        // 1. build_stats() MUST return None
        assert!(
            index.build_stats().is_none(),
            "build_stats() must return None when feature 'instrument' is disabled"
        );

        // 2. Struct size verification:
        // VisitedTracker has tags: Vec<u32> (24 bytes) + curr_tag: u32 (4 bytes) + padding (4 bytes) = 32 bytes
        // No distance_evals: u64 field!
        assert_eq!(
            std::mem::size_of::<vector_search::hnsw::VisitedTracker>(),
            32
        );

        // HeuristicScratch: w_set (24) + w_d (24) + r (24) + w_pruned (24) + e_cands (24) + outer_neighbors (24) + seen (48) = 192 bytes
        // No distance_evals: u64 field!
        assert_eq!(std::mem::size_of::<HeuristicScratch>(), 192);

        // HnswIndex is 416 bytes without build_stats
        assert_eq!(std::mem::size_of::<HnswIndex>(), 416);
    }

    #[cfg(feature = "instrument")]
    {
        // When instrument feature is ON:
        // 1. build_stats() MUST return Some
        assert!(
            index.build_stats().is_some(),
            "build_stats() must return Some when feature 'instrument' is enabled"
        );
        let stats = index.build_stats().unwrap();
        assert!(stats.distance_evals_total > 0);
        assert!(stats.total_build_time_s >= 0.0);

        // 2. Struct size verification with instrument feature:
        // VisitedTracker has additional distance_evals: u64 (+8 bytes) = 40 bytes
        assert_eq!(
            std::mem::size_of::<vector_search::hnsw::VisitedTracker>(),
            40
        );

        // HeuristicScratch has additional distance_evals: u64 (+8 bytes) = 200 bytes
        assert_eq!(std::mem::size_of::<HeuristicScratch>(), 200);

        // HnswBuildStats is exactly 176 bytes (22 8-byte fields)
        assert_eq!(
            std::mem::size_of::<vector_search::hnsw::HnswBuildStats>(),
            176
        );

        // HnswIndex with build_stats: 416 + 8 (tracker) + 8 (scratch) + 176 (build_stats) = 608 bytes
        assert_eq!(std::mem::size_of::<HnswIndex>(), 608);
    }
}

// ============================================================================
// 3. Canonical (Distance, ID) Tie-Breaking Stress Tests
// ============================================================================

#[test]
fn test_challenge_candidate_ordering_canonical_contract() {
    // Direct unit test of Candidate comparison contract
    let c1 = Candidate {
        distance: 1.0,
        id: 10,
    };
    let c2 = Candidate {
        distance: 1.0,
        id: 20,
    };
    let c3 = Candidate {
        distance: 2.0,
        id: 5,
    };

    // Smaller ID must be Less when distances are equal
    assert_eq!(c1.cmp(&c2), Ordering::Less);
    assert_eq!(c2.cmp(&c1), Ordering::Greater);
    assert_eq!(c1.cmp(&c1), Ordering::Equal);

    // Distance dominates ID
    assert_eq!(c2.cmp(&c3), Ordering::Less);
    assert_eq!(c3.cmp(&c2), Ordering::Greater);
}

#[test]
fn test_challenge_canonical_tie_breaking_identical_duplicate_vectors() {
    // 50 completely identical vectors
    let num_vecs = 50;
    let dim = 16;
    let base_data = vec![std::f32::consts::PI; num_vecs * dim];
    let base = RowMatrix::new(num_vecs, dim, base_data).unwrap();

    let query_data = vec![std::f32::consts::PI; dim];
    let query_mat = RowMatrix::new(1, dim, query_data.clone()).unwrap();

    // Authoritative ExactKnn ground truth
    let exact = ExactKnn::from_base(Metric::L2Squared, base.clone()).unwrap();
    let gt = exact.search(&query_mat, 10).unwrap();

    // ExactKnn must break ties by strictly ascending ID: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let gt_ids = gt.indices.row(0);
    for (i, &gt_id) in gt_ids.iter().enumerate().take(10) {
        assert_eq!(gt_id, i as i32, "ExactKnn failed tie-breaking at rank {i}");
    }

    // Build HNSW index
    let cfg = HnswConfig {
        m: 16,
        m_max0: Some(32),
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();

    // Query HNSW with high efSearch (>= num_vecs so entire graph is accessible)
    let hits = index.search_knn(&query_data, 10, 50).unwrap();
    assert_eq!(hits.len(), 10);

    for (rank, &(dist, id)) in hits.iter().enumerate() {
        assert_eq!(dist, 0.0, "Distance to identical vector must be 0.0");
        assert_eq!(
            id, rank,
            "HNSW canonical tie-breaking violated at rank {rank}: expected ID {rank}, got {id}"
        );
    }
}

#[test]
fn test_challenge_canonical_tie_breaking_orthogonal_hypercube_basis() {
    // 32 standard basis vectors e_0..e_31 in R^32
    let dim = 32;
    let num_vecs = 32;
    let mut data = vec![0.0_f32; num_vecs * dim];
    for i in 0..num_vecs {
        data[i * dim + i] = 1.0;
    }
    let base = RowMatrix::new(num_vecs, dim, data).unwrap();

    // Query from origin: distance to every e_i is exactly 1.0 (L2Squared)
    let query = vec![0.0_f32; dim];
    let query_mat = RowMatrix::new(1, dim, query.clone()).unwrap();

    let exact = ExactKnn::from_base(Metric::L2Squared, base.clone()).unwrap();
    let gt = exact.search(&query_mat, 16).unwrap();

    // Ground truth MUST be 0..15 in strictly ascending order
    let gt_ids = gt.indices.row(0);
    for (i, &gt_id) in gt_ids.iter().enumerate().take(16) {
        assert_eq!(gt_id, i as i32);
        assert_eq!(gt.distances.row(0)[i], 1.0);
    }

    let cfg = HnswConfig {
        m: 16,
        m_max0: Some(32),
        ef_construction: 64,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();

    // Query HNSW with ef=32 so all equidistant nodes are explored
    let hits = index.search_knn(&query, 16, 32).unwrap();
    assert_eq!(hits.len(), 16);

    for (rank, &(dist, id)) in hits.iter().enumerate() {
        assert_eq!(
            dist, 1.0,
            "Distance from origin to basis vector must be 1.0"
        );
        assert_eq!(
            id, rank,
            "HNSW canonical tie-breaking mismatch on basis vectors at rank {rank}: expected {rank}, got {id}"
        );
    }
}

#[test]
fn test_challenge_canonical_tie_breaking_heuristic_scratch_direct() {
    // Direct stress-test of select_neighbors_heuristic_with_scratch on equidistant candidates
    let (base, _) = generate_synthetic_sift(20, 1, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::new(&base, Metric::L2Squared, cfg);

    let mut scratch = HeuristicScratch::new();

    // Provide candidates with identical distance 5.0, but scrambled IDs
    let cands = vec![
        Candidate {
            distance: 5.0,
            id: 15,
        },
        Candidate {
            distance: 5.0,
            id: 3,
        },
        Candidate {
            distance: 5.0,
            id: 8,
        },
        Candidate {
            distance: 5.0,
            id: 1,
        },
        Candidate {
            distance: 5.0,
            id: 12,
        },
    ];

    // Select m=3 neighbors from base_id=0
    index.select_neighbors_heuristic_with_scratch(
        0,
        &cands,
        3,
        0,
        false, // extend_candidates
        true,  // keep_pruned
        &mut scratch,
    );

    // The heuristic Min-Heap pop must process candidates in strictly ascending order: ID 1, 3, 8...
    // If triangular diversity does not prune, the first selected neighbor must be ID 1.
    assert!(!scratch.r.is_empty());
    assert_eq!(
        scratch.r[0], 1,
        "First selected candidate among equidistant must have minimum ID (1), got {}",
        scratch.r[0]
    );
}
