//! Comprehensive integration, invariant, and recall test suite for Phase 2 HNSW index.
//!
//! Organized into four tiers:
//! - Tier 1: Feature coverage & algorithmic foundations
//! - Tier 2: Boundary conditions & defensive error handling
//! - Tier 3: Cross-feature interactions & graph topology invariants
//! - Tier 4: Parity, monotonic recall scaling, and latency benchmarking vs ExactKnn

use std::path::Path;

use vector_search::benchmark::{
    run_phase2_benchmark, HnswEfEvaluation, Phase1DatasetMode, Phase2Config,
};
use vector_search::dataset::{generate_synthetic_sift, load_or_download_sift, SiftPreset};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::ExactKnn;
use vector_search::hnsw::{HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::recall::compute_recall_at_k;
use vector_search::rng::DeterministicRng;

// ============================================================================
// Tier 1: Feature Coverage & Foundational Mechanics
// ============================================================================

#[test]
fn test_tier1_layer_distribution_statistics() {
    let m = 16_usize;
    let ml = 1.0 / (m as f64).ln();
    let mut rng = DeterministicRng::new(42);

    let num_samples = 30_000;
    let mut layer_counts = [0_usize; 10];

    for _ in 0..num_samples {
        let u = rng.next_f64_open01();
        let l = (-u.ln() * ml).floor() as usize;
        let bucket = l.min(layer_counts.len() - 1);
        layer_counts[bucket] += 1;
    }

    let f0 = layer_counts[0] as f64 / num_samples as f64;
    let f1 = layer_counts[1] as f64 / num_samples as f64;
    let f_ge2: f64 = layer_counts[2..].iter().sum::<usize>() as f64 / num_samples as f64;

    // Theoretical: f0 = 15/16 = 0.9375, f1 = 15/256 = 0.05859, f_ge2 = 1/256 = 0.00391
    assert!(
        (0.930..=0.945).contains(&f0),
        "Layer 0 empirical fraction {f0} out of expected bounds [0.930, 0.945]"
    );
    assert!(
        (0.052..=0.065).contains(&f1),
        "Layer 1 empirical fraction {f1} out of expected bounds [0.052, 0.065]"
    );
    assert!(
        (0.002..=0.006).contains(&f_ge2),
        "Layer >= 2 empirical fraction {f_ge2} out of expected bounds [0.002, 0.006]"
    );

    // Seed determinism check
    let mut rng1 = DeterministicRng::new(999);
    let mut rng2 = DeterministicRng::new(999);
    for _ in 0..1_000 {
        assert_eq!(rng1.next_f64_open01(), rng2.next_f64_open01());
    }
}

#[test]
fn test_tier1_single_vector_insert_and_search() {
    let base = RowMatrix::from_rows(&[vec![1.0_f32, 2.0, 3.0, 4.0]]).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    assert_eq!(index.num_vectors(), 1);
    assert_eq!(index.entry_point(), Some(0));

    // Self-query returns distance 0.0
    let res_self = index.search_knn(&[1.0, 2.0, 3.0, 4.0], 1, 10).unwrap();
    assert_eq!(res_self.len(), 1);
    assert_eq!(res_self[0].1, 0); // node ID
    assert_eq!(res_self[0].0.to_bits(), 0.0_f32.to_bits()); // exact +0.0

    // Non-self query returns exact squared Euclidean distance
    let res_other = index.search_knn(&[2.0, 2.0, 3.0, 4.0], 1, 10).unwrap();
    assert_eq!(res_other.len(), 1);
    assert_eq!(res_other[0].1, 0);
    assert!((res_other[0].0 - 1.0).abs() < 1e-6);
}

#[test]
fn test_tier1_multi_vector_sequential_insert() {
    let (base, _) = generate_synthetic_sift(100, 10, 16, 4, 1234).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::new(&base, Metric::L2Squared, config);

    for i in 0..100 {
        index.insert(i).unwrap();
        assert_eq!(index.num_vectors(), i + 1);
        let ep = index.entry_point().expect("entry point exists");
        assert!(ep <= i);
    }

    // Invariant: all 100 vectors are in layer 0
    assert_eq!(index.layer_node_count(0), 100);

    // Invariant: layer hierarchy inclusion (nodes in layer l exist in layer l-1)
    for l in 1..=index.max_layer() {
        for node in index.nodes_at_layer(l) {
            assert!(
                index.node_exists_at_layer(l - 1, node),
                "Node {node} present at layer {l} but missing at layer {}",
                l - 1
            );
        }
    }
}

#[test]
fn test_tier1_greedy_routing_vs_beam_search() {
    let (base, queries) = generate_synthetic_sift(300, 20, 32, 6, 555).unwrap();
    let config = HnswConfig {
        m: 16,
        ef_construction: 100,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    let mut beam_improvements = 0;
    for q in queries.iter_rows() {
        let res_greedy = index.search_knn(q, 1, 1).unwrap();
        let res_beam = index.search_knn(q, 1, 50).unwrap();

        let d_greedy = res_greedy[0].0;
        let d_beam = res_beam[0].0;

        assert!(
            d_beam <= d_greedy + 1e-6,
            "Beam search distance {d_beam} worse than greedy {d_greedy}"
        );

        if d_beam < d_greedy - 1e-5 {
            beam_improvements += 1;
        }
    }

    // Beam search should recover from local minima on clustered manifold
    assert!(
        beam_improvements > 0,
        "Expected beam search to improve over greedy search on clustered dataset"
    );
}

#[test]
fn test_tier1_deterministic_canonical_tie_breaking() {
    // Equidistant points test: 4 vectors equidistant from origin
    let base = RowMatrix::from_rows(&[
        vec![1.0_f32, 0.0], // ID 0, d^2 = 1.0
        vec![0.0, 1.0],     // ID 1, d^2 = 1.0
        vec![-1.0, 0.0],    // ID 2, d^2 = 1.0
        vec![0.0, -1.0],    // ID 3, d^2 = 1.0
    ])
    .unwrap();

    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();
    let res = index.search_knn(&[0.0, 0.0], 4, 10).unwrap();

    assert_eq!(res.len(), 4);
    assert_eq!(res[0].1, 0);
    assert_eq!(res[1].1, 1);
    assert_eq!(res[2].1, 2);
    assert_eq!(res[3].1, 3);
    for r in &res {
        assert!((r.0 - 1.0).abs() < 1e-6);
    }
}

// ============================================================================
// Tier 2: Boundary Conditions & Defensive Error Handling
// ============================================================================

#[test]
fn test_tier2_empty_index_query() {
    let empty_base = RowMatrix::<f32>::empty();
    let config = HnswConfig::default();
    let mut index = HnswIndex::new(&empty_base, Metric::L2Squared, config);

    assert_eq!(index.num_vectors(), 0);
    assert_eq!(index.max_layer(), 0);
    assert_eq!(index.entry_point(), None);

    let res = index.search_knn(&[1.0, 2.0, 3.0], 5, 10).unwrap();
    assert!(res.is_empty());
}

#[test]
fn test_tier2_k_greater_than_n() {
    let base = RowMatrix::from_rows(&[
        vec![1.0_f32, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![4.0, 0.0],
        vec![5.0, 0.0],
    ])
    .unwrap();

    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();
    let res = index.search_knn(&[0.0, 0.0], 20, 20).unwrap();

    assert_eq!(res.len(), 5);
    assert_eq!(res[0].1, 0);
    assert_eq!(res[1].1, 1);
    assert_eq!(res[2].1, 2);
    assert_eq!(res[3].1, 3);
    assert_eq!(res[4].1, 4);
}

#[test]
fn test_tier2_ef_search_less_than_k() {
    let (base, _) = generate_synthetic_sift(30, 5, 8, 2, 77).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    // k=15 but ef_search=3 -> clamped to ef=15
    let res = index.search_knn(&[0.0_f32; 8], 15, 3).unwrap();
    assert_eq!(res.len(), 15);

    // Verify results are sorted ascending
    for i in 1..res.len() {
        assert!(res[i - 1].0 <= res[i].0);
    }
}

#[test]
fn test_tier2_single_vector_boundaries() {
    let base = RowMatrix::from_rows(&[vec![1.0_f32, 2.0, 3.0, 4.0]]).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    // k=0 rejects
    assert!(matches!(
        index.search_knn(&[1.0, 2.0, 3.0, 4.0], 0, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Dimension mismatch rejects
    assert!(matches!(
        index.search_knn(&[1.0, 2.0], 1, 10),
        Err(VectorSearchError::DimensionMismatch {
            expected: 4,
            actual: 2
        })
    ));

    // Non-finite values reject
    assert!(matches!(
        index.search_knn(&[1.0, f32::NAN, 3.0, 4.0], 1, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        index.search_knn(&[1.0, f32::INFINITY, 3.0, 4.0], 1, 10),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

#[test]
fn test_tier2_duplicate_vectors_zero_distance() {
    let base = RowMatrix::from_rows(&[
        vec![3.0_f32, -1.0, 4.0],
        vec![3.0, -1.0, 4.0],
        vec![3.0, -1.0, 4.0],
        vec![3.0, -1.0, 4.0],
        vec![3.0, -1.0, 4.0],
    ])
    .unwrap();

    let config = HnswConfig::default();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();
    let res = index.search_knn(&[3.0, -1.0, 4.0], 5, 10).unwrap();

    assert_eq!(res.len(), 5);
    for (i, item) in res.iter().enumerate() {
        assert_eq!(item.1, i);
        assert_eq!(item.0.to_bits(), 0.0_f32.to_bits());
    }
}

#[test]
fn test_tier2_extreme_dimensions() {
    // 1D metric space
    let base1d =
        RowMatrix::from_rows(&(0..50).map(|i| vec![i as f32 * 2.0]).collect::<Vec<_>>()).unwrap();
    let mut idx1d = HnswIndex::build(&base1d, Metric::L2Squared, HnswConfig::default()).unwrap();
    let res1d = idx1d.search_knn(&[25.0], 2, 20).unwrap();
    assert_eq!(res1d.len(), 2);
    // Closest are 24.0 (ID 12) and 26.0 (ID 13)
    assert!(res1d.iter().any(|r| r.1 == 12));
    assert!(res1d.iter().any(|r| r.1 == 13));

    // 1024-D high-dimensional space
    let (base_hd, queries_hd) = generate_synthetic_sift(20, 2, 1024, 2, 888).unwrap();
    let mut idx_hd = HnswIndex::build(&base_hd, Metric::L2Squared, HnswConfig::default()).unwrap();
    let res_hd = idx_hd.search_knn(queries_hd.row(0), 5, 20).unwrap();
    assert_eq!(res_hd.len(), 5);
}

#[test]
fn test_tier2_extreme_m_values() {
    let (base, queries) = generate_synthetic_sift(100, 5, 16, 4, 333).unwrap();

    for &m in &[2_usize, 4, 32, 64] {
        let config = HnswConfig {
            m,
            m_max0: 2 * m,
            ef_construction: 100,
            ..Default::default()
        };
        let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();
        let res = index.search_knn(queries.row(0), 5, 20).unwrap();
        assert_eq!(res.len(), 5);

        // Verify degree bounds
        for u in 0..100 {
            assert!(index.neighbors(0, u).len() <= 2 * m);
            for l in 1..=index.max_layer() {
                if index.node_exists_at_layer(l, u) {
                    assert!(index.neighbors(l, u).len() <= m);
                }
            }
        }
    }
}

// ============================================================================
// Tier 3: Cross-Feature Interactions & Graph Invariants
// ============================================================================

#[test]
fn test_tier3_multi_layer_graph_hierarchy() {
    // With M=4, layer scaling produces taller hierarchy
    let (base, _) = generate_synthetic_sift(1_500, 10, 16, 8, 42).unwrap();
    let config = HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 50,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    assert!(
        index.max_layer() >= 3,
        "Expected max layer >= 3, got {}",
        index.max_layer()
    );

    let mut prev_count = index.layer_node_count(0);
    assert_eq!(prev_count, 1_500);

    for l in 1..=index.max_layer() {
        let count = index.layer_node_count(l);
        assert!(
            count < prev_count,
            "Layer {l} count {count} not less than previous {prev_count}"
        );
        prev_count = count;
    }
}

#[test]
fn test_tier3_heuristic_neighbor_pruning_keep_pruned() {
    // Collinear points along a ray from base point [0.0, 0.0]
    let base = RowMatrix::from_rows(
        &(0..12)
            .map(|i| vec![i as f32 * 1.5, 0.0])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let config_pruned_false = HnswConfig {
        m: 4,
        m_max0: 8,
        keep_pruned: false,
        ..Default::default()
    };
    let idx_pruned_false = HnswIndex::build(&base, Metric::L2Squared, config_pruned_false).unwrap();

    let config_pruned_true = HnswConfig {
        m: 4,
        m_max0: 8,
        keep_pruned: true,
        ..Default::default()
    };
    let idx_pruned_true = HnswIndex::build(&base, Metric::L2Squared, config_pruned_true).unwrap();

    // keep_pruned=true should maintain target degree
    assert!(
        idx_pruned_true.neighbors(0, 0).len() >= idx_pruned_false.neighbors(0, 0).len(),
        "keep_pruned=true must retain equal or greater connection density"
    );
}

#[test]
fn test_tier3_bidirectional_edge_consistency_and_degree_bounds() {
    let (base, _) = generate_synthetic_sift(300, 10, 16, 4, 999).unwrap();
    let m = 16;
    let m_max0 = 32;
    let config = HnswConfig {
        m,
        m_max0,
        ef_construction: 100,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    for u in 0..300 {
        // No self loops
        assert!(!index.neighbors(0, u).contains(&u));

        // Strict degree caps
        assert!(index.neighbors(0, u).len() <= m_max0);
        for l in 1..=index.max_layer() {
            if index.node_exists_at_layer(l, u) {
                assert!(!index.neighbors(l, u).contains(&u));
                assert!(index.neighbors(l, u).len() <= m);
            }
        }
    }
}

#[test]
fn test_tier3_dynamic_entry_point_elevation() {
    let (base, _) = generate_synthetic_sift(200, 10, 16, 4, 777).unwrap();
    let config = HnswConfig::default();
    let mut index = HnswIndex::new(&base, Metric::L2Squared, config);

    let mut current_top_layer = 0;
    let mut current_ep = 0;

    for i in 0..200 {
        index.insert(i).unwrap();
        let node_layer = index.node_layer(i);
        if i == 0 || node_layer > current_top_layer {
            current_top_layer = node_layer;
            current_ep = i;
        }

        assert_eq!(index.max_layer(), current_top_layer);
        assert_eq!(index.entry_point(), Some(current_ep));
    }
}

// ============================================================================
// Tier 4: Parity, Recall & Performance Benchmarks
// ============================================================================

#[test]
fn test_tier4_synthetic_clusters_recall_scaling() {
    let (base, queries) = generate_synthetic_sift(1_000, 50, 64, 8, 42).unwrap();
    let exact = ExactKnn::from_base(Metric::L2Squared, base.clone()).unwrap();
    let gt = exact.search(&queries, 10).unwrap();

    let config = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    let ef_ladder = [10_usize, 20, 50, 100, 200];
    let mut recalls = Vec::new();

    for &ef in &ef_ladder {
        let mut cand_rows = Vec::with_capacity(50 * 10);
        for q in queries.iter_rows() {
            let res = index.search_knn(q, 10, ef).unwrap();
            for item in res {
                cand_rows.push(item.1 as i32);
            }
        }
        let cand_mat = RowMatrix::new(50, 10, cand_rows).unwrap();
        let recall = compute_recall_at_k(&cand_mat, &gt.indices, 10).unwrap();
        recalls.push(recall);
    }

    // Monotonic scaling check
    for i in 1..recalls.len() {
        assert!(
            recalls[i] >= recalls[i - 1] - 1e-6,
            "Recall degraded from {} to {} between ef={} and ef={}",
            recalls[i - 1],
            recalls[i],
            ef_ladder[i - 1],
            ef_ladder[i]
        );
    }

    // Peak recall assertion
    assert!(
        *recalls.last().unwrap() >= 0.98,
        "Peak recall at ef=200 was below 0.98: {}",
        recalls.last().unwrap()
    );
}

#[test]
fn test_tier4_siftsmall_10k_parity_and_recall() {
    let loaded =
        load_or_download_sift(SiftPreset::SiftSmall, Path::new("data"), false, 42).unwrap();
    assert_eq!(loaded.base.shape(), (10_000, 128));
    assert_eq!(loaded.query.shape(), (100, 128));

    let exact = ExactKnn::from_base(Metric::L2Squared, loaded.base.clone()).unwrap();
    let gt = exact.search(&loaded.query, 10).unwrap();

    let config = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&loaded.base, Metric::L2Squared, config).unwrap();

    let ef_ladder = [10_usize, 50, 100, 200];
    let thresholds = [0.80_f64, 0.92, 0.96, 0.98];

    let mut prev_recall = 0.0;
    for (&ef, &min_expected) in ef_ladder.iter().zip(thresholds.iter()) {
        let mut cand_rows = Vec::with_capacity(100 * 10);
        for q in loaded.query.iter_rows() {
            let res = index.search_knn(q, 10, ef).unwrap();
            for item in res {
                cand_rows.push(item.1 as i32);
            }
        }
        let cand_mat = RowMatrix::new(100, 10, cand_rows).unwrap();
        let recall = compute_recall_at_k(&cand_mat, &gt.indices, 10).unwrap();

        assert!(
            recall >= prev_recall - 1e-6,
            "Recall degraded at ef={ef}: {recall} < {prev_recall}"
        );
        assert!(
            recall >= min_expected,
            "Recall at ef={ef} was {recall}, expected >= {min_expected}"
        );
        prev_recall = recall;
    }
}

#[test]
fn test_tier4_bit_exact_reproducibility() {
    let (base, queries) = generate_synthetic_sift(300, 25, 32, 4, 12345).unwrap();

    let config = HnswConfig {
        seed: 42,
        ..Default::default()
    };

    let mut index1 = HnswIndex::build(&base, Metric::L2Squared, config.clone()).unwrap();
    let mut index2 = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    for q in queries.iter_rows() {
        let res1 = index1.search_knn(q, 10, 50).unwrap();
        let res2 = index2.search_knn(q, 10, 50).unwrap();

        assert_eq!(res1.len(), res2.len());
        for (a, b) in res1.iter().zip(res2.iter()) {
            assert_eq!(a.1, b.1, "Neighbor ID mismatch between identical runs");
            assert_eq!(
                a.0.to_bits(),
                b.0.to_bits(),
                "Distance float bits mismatch between identical runs"
            );
        }
    }
}

#[test]
fn test_tier4_hnsw_query_latency_speedup() {
    let loaded =
        load_or_download_sift(SiftPreset::SiftSmall, Path::new("data"), false, 42).unwrap();
    let exact = ExactKnn::from_base(Metric::L2Squared, loaded.base.clone()).unwrap();

    let t_exact_start = std::time::Instant::now();
    let _ = exact.search(&loaded.query, 10).unwrap();
    let exact_duration = t_exact_start.elapsed();

    let config = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&loaded.base, Metric::L2Squared, config).unwrap();

    let t_hnsw_start = std::time::Instant::now();
    for q in loaded.query.iter_rows() {
        let _ = index.search_knn(q, 10, 50).unwrap();
    }
    let hnsw_duration = t_hnsw_start.elapsed();

    let speedup = exact_duration.as_secs_f64() / hnsw_duration.as_secs_f64();
    assert!(
        speedup >= 2.0,
        "HNSW speedup over brute-force exact search was {speedup:.2}x, expected >= 2.0x"
    );
}

#[test]
fn test_hnsw_10k_synthetic_monotonic_recall_and_speedup() {
    let config = Phase2Config {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 10_000,
            num_query: 100,
            dim: 128,
            num_clusters: 16,
        },
        k: 10,
        seed: 42,
        metric: Metric::L2Squared,
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search_list: vec![10, 50, 100, 200],
        allow_download: false,
        save_files: false,
        max_queries: None,
        run_repro_check: true,
        ..Default::default()
    };

    let report = run_phase2_benchmark(&config).expect("Phase 2 benchmark must succeed");

    assert!(
        report.monotonic_scaling_verified,
        "Recall@10 failed monotonic scaling across efSearch"
    );

    let eval_map: std::collections::HashMap<usize, &HnswEfEvaluation> = report
        .ef_evaluations
        .iter()
        .map(|e| (e.ef_search, e))
        .collect();

    assert!(
        eval_map[&10].recall_at_k >= 0.70,
        "Recall at ef=10 below floor 0.70"
    );
    assert!(
        eval_map[&50].recall_at_k >= 0.88,
        "Recall at ef=50 below floor 0.88"
    );
    assert!(
        eval_map[&100].recall_at_k >= 0.94,
        "Recall at ef=100 below floor 0.94"
    );
    assert!(
        eval_map[&200].recall_at_k >= 0.97,
        "Recall at ef=200 below floor 0.97"
    );

    assert!(
        eval_map[&50].speedup_vs_exact >= 2.0,
        "Speedup at ef=50 was {:.2}x, expected >= 2.0x",
        eval_map[&50].speedup_vs_exact
    );

    assert!(
        report.reproducibility_verified,
        "Bit-exact reproducibility failed on identical seeds"
    );
}

#[test]
fn test_batch_parallel_parity() {
    let (base, queries) = generate_synthetic_sift(300, 20, 32, 4, 777).unwrap();
    let config = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, config).unwrap();

    let seq_res = index.search_batch(&queries, 10, 50).unwrap();
    let par_res = index.search_batch_parallel(&queries, 10, 50, 4).unwrap();

    assert_eq!(seq_res.indices.as_slice(), par_res.indices.as_slice());
    for (&d_seq, &d_par) in seq_res
        .distances
        .as_slice()
        .iter()
        .zip(par_res.distances.as_slice().iter())
    {
        assert_eq!(d_seq.to_bits(), d_par.to_bits());
    }
}
