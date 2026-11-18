//! Phase 6 Stage 2 tests (`T4.4` & `T4.5`):
//! - Deterministic `PqTrainer` (`Xoshiro256**` subsampling + `k`-means++ + `f64` Lloyd) & seed variation
//! - Degenerate inputs (`N < 256`, `N = 1`, all-identical, duplicate modes, all-zero, extreme ranges)
//! - Configuration & non-finite / `dim % m_pq != 0` / query error guards
//! - `pq_adc_distance` & `PqVectors` / `PqWithSdc` `f64` parity across `m_pq in {4, 7, 16, 32}`
//!   and `Metric::{L2Squared, Euclidean, InnerProduct}`
//! - Clustered `<= 256`-mode near-100% recall check (`efSearch = 50`)
//! - `HnswPqIndex` (`M_pq in {16, 32}`, `16 B/vec` & `32 B/vec`, `QuantizedOnly` vs `WithExactRerank`,
//!   `Float32GraphThenQuantize` vs `QuantizedDirect`, no retained SDC table, `R = N` exact parity,
//!   `R = k` recall invariance, zero scratch growth after warmup, snapshot roundtrip, streaming `.fvecs`)

mod common;

use std::io::Cursor;
use std::sync::Arc;

use common::{make_random_matrix, TempTestDir};
use vector_search::dataset::{generate_synthetic_sift, read_fvecs_max_rows, write_fvecs};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::ExactKnn;
use vector_search::hnsw::{HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::quantized::{
    pq_adc_distance, pq_sdc_distance, results_to_knn_search_result, HnswPqIndex, Neighbor,
    PqCodebook, PqConfig, PqQueryState, PqTrainer, PqVectors, PqWithSdc, QuantizedBuildMode,
    QuantizedDistanceSource, QuantizedStorageMode, PQ_K_CENTROIDS,
};
use vector_search::recall::compute_recall_at_k;
use vector_search::rng::DeterministicRng;

// ============================================================================
// 1. T4.4: PqTrainer Determinism, Subsampling, Degenerate & Validation Guards
// ============================================================================

#[test]
fn phase6_pq_training_determinism_and_seed_variation() {
    // 1. Without subsampling (N <= max_train_vectors)
    let (base_small, _) = generate_synthetic_sift(300, 5, 64, 16, 101).unwrap();
    let cfg_a1 = PqConfig::new(16).with_seed(42);
    let cfg_a2 = PqConfig::new(16).with_seed(42);
    let cfg_b = PqConfig::new(16).with_seed(777);

    let vecs_a1 = PqVectors::from_matrix(&base_small, Metric::L2Squared, &cfg_a1).unwrap();
    let vecs_a2 = PqVectors::from_matrix(&base_small, Metric::L2Squared, &cfg_a2).unwrap();
    let vecs_b = PqVectors::from_matrix(&base_small, Metric::L2Squared, &cfg_b).unwrap();

    assert_eq!(vecs_a1.codebook(), vecs_a2.codebook());
    assert_eq!(vecs_a1.codes(), vecs_a2.codes());
    assert_ne!(
        vecs_a1.codebook().centroids(),
        vecs_b.codebook().centroids()
    );

    // 2. With subsampling (N > max_train_vectors)
    let (base_sub, _) = generate_synthetic_sift(450, 5, 64, 12, 202).unwrap();
    let mut cfg_sub1 = PqConfig::new(16).with_seed(42);
    cfg_sub1.max_train_vectors = 128;
    let mut cfg_sub2 = cfg_sub1;
    let cfg_sub3 = cfg_sub1.with_seed(999);

    let sub1 = PqVectors::from_matrix(&base_sub, Metric::L2Squared, &cfg_sub1).unwrap();
    let sub2 = PqVectors::from_matrix(&base_sub, Metric::L2Squared, &cfg_sub2).unwrap();
    let sub3 = PqVectors::from_matrix(&base_sub, Metric::L2Squared, &cfg_sub3).unwrap();

    assert_eq!(sub1.codebook().k_effective(), 128);
    assert_eq!(sub1.codebook(), sub2.codebook());
    assert_eq!(sub1.codes(), sub2.codes());
    assert_ne!(sub1.codebook().centroids(), sub3.codebook().centroids());
    cfg_sub2.max_lloyd_iters = 0; // smoke k-means++ only path (0 Lloyd iters)
    let cb_init_only = PqTrainer::train(&base_sub, &cfg_sub2).unwrap();
    assert_eq!(cb_init_only.k_effective(), 128);
    assert!(cb_init_only.centroids().iter().all(|v| v.is_finite()));
}

#[test]
fn phase6_pq_small_n_lt_256_all_identical_zeros_and_extreme_ranges() {
    // 1. N = 10 (< 256): verify k_effective = 10, unused slots padded with centroid 0, codes < 10
    let small = make_random_matrix(10, 64, -20.0, 20.0, 303);
    let cfg = PqConfig::new(16).with_seed(42);
    let vecs_small = PqVectors::from_matrix(&small, Metric::L2Squared, &cfg).unwrap();
    let cb = vecs_small.codebook();
    assert_eq!(cb.k_effective(), 10);
    assert_eq!(cb.centroids().len(), 16 * PQ_K_CENTROIDS * 4);
    assert!(cb.centroids().iter().all(|v| v.is_finite()));

    for m in 0..16 {
        let c0 = cb.centroid(m, 0);
        for k in 10..PQ_K_CENTROIDS {
            assert_eq!(cb.centroid(m, k), c0, "Unused slot {k} not padded with c0");
        }
    }
    for &code_byte in vecs_small.codes() {
        assert!(
            usize::from(code_byte) < 10,
            "Encoded code {code_byte} >= k_effective 10"
        );
    }

    // 2. All-zero vectors (both N < 256 and N > 256)
    for n_rows in [10_usize, 300] {
        let zeros = RowMatrix::new(n_rows, 32, vec![0.0_f32; n_rows * 32]).unwrap();
        let vecs_z = PqVectors::from_matrix(&zeros, Metric::L2Squared, &PqConfig::new(16)).unwrap();
        assert!(vecs_z.codebook().centroids().iter().all(|v| v.is_finite()));
        let mut st = PqQueryState::default();
        let q = vec![1.0_f32; 32];
        vecs_z.prepare_query(&q, &mut st);
        for id in 0..n_rows {
            let d = vecs_z.query_distance(&st, id);
            assert!(d.is_finite() && (d - 32.0).abs() < 1e-4, "got d={d}");
        }
    }

    // 3. All-identical non-zero vectors
    let mut identical_data = Vec::with_capacity(40 * 32);
    for _ in 0..40 {
        for j in 0..32 {
            identical_data.push((j as f32) * 1.5 - 10.0);
        }
    }
    let identical = RowMatrix::new(40, 32, identical_data).unwrap();
    let vecs_id = PqVectors::from_matrix(&identical, Metric::L2Squared, &PqConfig::new(8)).unwrap();
    let mut recon = vec![0.0_f32; 32];
    vecs_id
        .codebook()
        .dequantize_row(vecs_id.code(0), &mut recon)
        .unwrap();
    assert_eq!(recon, identical.row(0));

    // 4. Two duplicate clusters repeated across 300 rows (> 256 -> empty clusters every sub)
    let mut dup_rows = Vec::with_capacity(300);
    for i in 0..300 {
        if i % 3 == 0 {
            dup_rows.push(vec![10.0_f32; 32]);
        } else {
            dup_rows.push(vec![-5.0_f32; 32]);
        }
    }
    let dups = RowMatrix::from_rows(&dup_rows).unwrap();
    let vecs_dups = PqVectors::from_matrix(&dups, Metric::L2Squared, &PqConfig::new(8)).unwrap();
    assert!(vecs_dups
        .codebook()
        .centroids()
        .iter()
        .all(|v| v.is_finite()));
    let mut recon_a = vec![0.0_f32; 32];
    let mut recon_b = vec![0.0_f32; 32];
    vecs_dups
        .codebook()
        .dequantize_row(vecs_dups.code(0), &mut recon_a)
        .unwrap();
    vecs_dups
        .codebook()
        .dequantize_row(vecs_dups.code(1), &mut recon_b)
        .unwrap();
    assert_eq!(recon_a, vec![10.0_f32; 32]);
    assert_eq!(recon_b, vec![-5.0_f32; 32]);

    // 5. Extreme dynamic range [-3e38, 3e38] & N = 1
    let extreme = RowMatrix::from_rows(&[
        vec![-3.0e38_f32; 16],
        vec![3.0e38_f32; 16],
        vec![0.0_f32; 16],
    ])
    .unwrap();
    let cb_ext = PqTrainer::train(&extreme, &PqConfig::new(4)).unwrap();
    assert!(cb_ext.centroids().iter().all(|v| v.is_finite()));

    let single =
        RowMatrix::from_rows(&[vec![1.25_f32, -3.5, 0.0, 9.0, 2.0, -1.0, 4.5, 0.25]]).unwrap();
    let idx_single = HnswPqIndex::build(
        &single,
        Metric::L2Squared,
        HnswConfig::default(),
        PqConfig::new(4),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let mut scratch = idx_single.new_search_scratch(10);
    let hits = idx_single
        .search_knn_with_scratch(single.row(0), 5, 10, &mut scratch)
        .unwrap();
    assert_eq!(hits, vec![Neighbor::new(0, 0.0)]);
}

#[test]
fn phase6_pq_error_guards_divisibility_config_and_non_finite() {
    let base = make_random_matrix(32, 128, -5.0, 5.0, 404);

    // dim % m_pq != 0
    for bad_m in [3_usize, 6, 15, 24, 33, 256] {
        assert!(
            matches!(
                PqTrainer::train(&base, &PqConfig::new(bad_m)),
                Err(VectorSearchError::InvalidArgument(_))
            ),
            "Expected InvalidArgument for dim=128, m_pq={bad_m}"
        );
        assert!(matches!(
            HnswPqIndex::build(
                &base,
                Metric::L2Squared,
                HnswConfig::default(),
                PqConfig::new(bad_m),
                QuantizedBuildMode::Float32GraphThenQuantize,
                QuantizedStorageMode::QuantizedOnly,
            ),
            Err(VectorSearchError::InvalidArgument(_))
        ));
    }

    // Invalid PqConfig fields
    assert!(matches!(
        PqTrainer::train(&base, &PqConfig::new(0)),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    let mut bad_k = PqConfig::new(16);
    bad_k.k_centroids = 128;
    assert!(matches!(
        PqTrainer::train(&base, &bad_k),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    let mut bad_train = PqConfig::new(16);
    bad_train.max_train_vectors = 0;
    assert!(matches!(
        PqTrainer::train(&base, &bad_train),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Empty matrix
    assert!(matches!(
        PqTrainer::train(&RowMatrix::empty(), &PqConfig::new(16)),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Non-finite base entries
    for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad_mat = base.clone();
        bad_mat.as_mut_slice()[130] = bad_val;
        assert!(matches!(
            PqTrainer::train(&bad_mat, &PqConfig::new(16)),
            Err(VectorSearchError::InvalidArgument(_))
        ));
        assert!(matches!(
            PqVectors::from_matrix(&bad_mat, Metric::L2Squared, &PqConfig::new(16)),
            Err(VectorSearchError::InvalidArgument(_))
        ));
        assert!(matches!(
            HnswPqIndex::build(
                &bad_mat,
                Metric::L2Squared,
                HnswConfig::default(),
                PqConfig::new(16),
                QuantizedBuildMode::QuantizedDirect,
                QuantizedStorageMode::QuantizedOnly,
            ),
            Err(VectorSearchError::InvalidArgument(_))
        ));
    }

    // Query guards on populated HnswPqIndex
    let idx = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(8, 20),
        PqConfig::new(16),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::WithExactRerank,
    )
    .unwrap();
    let mut scratch = idx.new_search_scratch(20);

    for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut q = vec![0.25_f32; 128];
        q[17] = bad_val;
        assert!(matches!(
            idx.search_knn_with_scratch(&q, 5, 20, &mut scratch),
            Err(VectorSearchError::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        idx.search_knn_with_scratch(&[1.0_f32; 64], 5, 20, &mut scratch),
        Err(VectorSearchError::DimensionMismatch {
            expected: 128,
            actual: 64
        })
    ));
    assert!(matches!(
        idx.search_knn_with_scratch(&[1.0_f32; 128], 0, 20, &mut scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        idx.search_knn_rerank_with_scratch(&[1.0_f32; 128], 5, 20, Some(0), &mut scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Rerank on QuantizedOnly returns Err
    let idx_only = idx.into_quantized_only();
    assert!(matches!(
        idx_only.search_knn_rerank_with_scratch(&[1.0_f32; 128], 5, 20, Some(10), &mut scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

// ============================================================================
// 2. T4.4: ADC & SDC Kernel Parity vs f64 Reference Across m_pq and Metrics
// ============================================================================

#[test]
fn phase6_pq_adc_and_sdc_f64_parity_across_m_pq_and_metrics() {
    // Test m_pq in {4, 16, 32} (at D=128) plus remainder m_pq=7 (at D=28)
    let cases = [(128_usize, 4_usize), (128, 16), (128, 32), (28, 7)];

    for &(dim, m_pq) in &cases {
        let base = make_random_matrix(80, dim, -25.0, 25.0, 500 + m_pq as u64);
        let queries = make_random_matrix(6, dim, -25.0, 25.0, 600 + m_pq as u64);
        let pq_cfg = PqConfig::new(m_pq).with_seed(42);

        for metric in [Metric::L2Squared, Metric::Euclidean, Metric::InnerProduct] {
            let vecs = PqVectors::from_matrix(&base, metric, &pq_cfg).unwrap();
            let cb = vecs.codebook();
            let mut state = PqQueryState::with_subspaces(m_pq);
            let mut recon = vec![0.0_f32; dim];

            // ADC parity vs f64 centroid reconstruction
            for q_row in queries.iter_rows() {
                vecs.prepare_query(q_row, &mut state);
                assert_eq!(state.adc_table.len(), m_pq * PQ_K_CENTROIDS);

                for id in 0..base.rows() {
                    let code = vecs.code(id);
                    cb.dequantize_row(code, &mut recon).unwrap();
                    let actual = vecs.format_output_distance(vecs.query_distance(&state, id));

                    let (expected_f64, scale) = match metric {
                        Metric::L2Squared => {
                            let direct_adc = pq_adc_distance(&state.adc_table, code);
                            assert_eq!(direct_adc.to_bits(), actual.to_bits());
                            let sum = q_row
                                .iter()
                                .zip(recon.iter())
                                .map(|(&q, &r)| {
                                    let d = f64::from(q) - f64::from(r);
                                    d * d
                                })
                                .sum::<f64>();
                            (sum, sum.abs().max(1.0))
                        }
                        Metric::Euclidean => {
                            let dist = q_row
                                .iter()
                                .zip(recon.iter())
                                .map(|(&q, &r)| {
                                    let d = f64::from(q) - f64::from(r);
                                    d * d
                                })
                                .sum::<f64>()
                                .sqrt();
                            (dist, dist.abs().max(1.0))
                        }
                        Metric::InnerProduct => {
                            let dot = q_row
                                .iter()
                                .zip(recon.iter())
                                .map(|(&q, &r)| f64::from(q) * f64::from(r))
                                .sum::<f64>();
                            let abs_energy = q_row
                                .iter()
                                .zip(recon.iter())
                                .map(|(&q, &r)| (f64::from(q) * f64::from(r)).abs())
                                .sum::<f64>();
                            (dot, abs_energy.max(1.0))
                        }
                    };

                    let tol = 1e-5 * scale;
                    assert!(
                        (f64::from(actual) - expected_f64).abs() <= tol,
                        "ADC parity failure dim={dim}, m_pq={m_pq}, metric={metric}: actual={actual}, expected_f64={expected_f64}, tol={tol}"
                    );
                }
            }

            // SDC wrapper parity vs PqVectors::node_distance & symmetry
            let sdc_wrap = PqWithSdc::new(vecs.clone());
            assert_eq!(sdc_wrap.vectors(), &vecs);
            assert_eq!(
                sdc_wrap.sdc_memory_bytes(),
                m_pq * PQ_K_CENTROIDS * PQ_K_CENTROIDS * std::mem::size_of::<f32>()
            );
            let mut recon_a = vec![0.0_f32; dim];
            let mut recon_b = vec![0.0_f32; dim];
            for a in 0..20 {
                cb.dequantize_row(vecs.code(a), &mut recon_a).unwrap();
                for b in 0..20 {
                    cb.dequantize_row(vecs.code(b), &mut recon_b).unwrap();
                    let d_vec = vecs.node_distance(a, b);
                    let d_sdc = sdc_wrap.node_distance(a, b);
                    let d_sdc_rev = sdc_wrap.node_distance(b, a);
                    assert_eq!(d_sdc.to_bits(), d_sdc_rev.to_bits());
                    let sdc_scale = if metric == Metric::InnerProduct {
                        recon_a
                            .iter()
                            .zip(recon_b.iter())
                            .map(|(&xa, &xb)| (xa * xb).abs())
                            .sum::<f32>()
                            .max(1.0)
                    } else {
                        d_vec.abs().max(1.0)
                    };
                    assert!(
                        (d_vec - d_sdc).abs() <= 1e-5 * sdc_scale,
                        "SDC mismatch at ({a}, {b}), m_pq={m_pq}, metric={metric}: {d_vec} vs {d_sdc}"
                    );
                    if matches!(metric, Metric::L2Squared | Metric::Euclidean) {
                        let direct_sdc =
                            pq_sdc_distance(sdc_wrap.sdc_table(), vecs.code(a), vecs.code(b));
                        if a == b {
                            assert_eq!(d_sdc, 0.0);
                        } else {
                            assert_eq!(d_sdc.to_bits(), direct_sdc.to_bits());
                        }
                    }
                }
            }
            let recovered = sdc_wrap.into_pq_vectors();
            assert_eq!(recovered, vecs);
        }
    }
}

#[test]
fn phase6_pq_clustered_256_modes_near_perfect_recall() {
    // 128 distinct well-separated modes (<= 256), 10 members per mode (1,280 vectors, D = 64).
    // With 128 modes <= K = 256 and tight sigma = 0.02, PQ16 at efSearch = 50 should achieve >= 0.99 Recall@10.
    let num_modes = 128_usize;
    let per_mode = 10_usize;
    let dim = 64_usize;
    let k = 10_usize;

    let mut rng = DeterministicRng::new(777);
    let mut modes = vec![0.0_f32; num_modes * dim];
    for val in &mut modes {
        *val = (rng.next_f64_open01() as f32) * 100.0 - 50.0;
    }

    let mut base_data = Vec::with_capacity(num_modes * per_mode * dim);
    for m in 0..num_modes {
        let mode = &modes[m * dim..(m + 1) * dim];
        for _ in 0..per_mode {
            for &c in mode {
                base_data.push(c + rng.next_normal_f32(0.0, 0.02));
            }
        }
    }
    let mut query_data = Vec::with_capacity(40 * dim);
    for m in 0..40 {
        let mode = &modes[m * dim..(m + 1) * dim];
        for &c in mode {
            query_data.push(c + rng.next_normal_f32(0.0, 0.02));
        }
    }

    let base = RowMatrix::new(num_modes * per_mode, dim, base_data).unwrap();
    let queries = RowMatrix::new(40, dim, query_data).unwrap();

    let exact = ExactKnn::from_base(Metric::L2Squared, base.clone()).unwrap();
    let gt = exact.search(&queries, k).unwrap();

    let idx = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        PqConfig::new(16),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();

    let mut scratch = idx.new_search_scratch(50);
    let res = idx
        .search_batch_with_scratch(&queries, k, 50, None, &mut scratch)
        .unwrap();
    let knn = results_to_knn_search_result(&res).unwrap();
    let recall = compute_recall_at_k(&knn.indices, &gt.indices, k).unwrap();
    assert!(
        recall >= 0.99,
        "Expected near-100% Recall@10 on 128 tight modes, got {recall:.4}"
    );
}

// ============================================================================
// 3. T4.5: HnswPqIndex Ownership, Memory, Rerank, Zero-Alloc, Snapshot, Streaming
// ============================================================================

#[test]
fn phase6_pq16_and_pq32_ownership_memory_profile_and_no_retained_sdc() {
    let (base, queries) = generate_synthetic_sift(400, 12, 128, 16, 42).unwrap();
    let cfg = HnswConfig::new(16, 100);
    let f32_idx = HnswIndex::build(&base, Metric::L2Squared, cfg.clone()).unwrap();
    let f32_fp = f32_idx.graph_fingerprint();
    drop(f32_idx);

    for m_pq in [16_usize, 32] {
        // Float32GraphThenQuantize preserves exact f32 graph_fingerprint
        let idx_f32g = HnswPqIndex::build(
            &base,
            Metric::L2Squared,
            cfg.clone(),
            PqConfig::new(m_pq),
            QuantizedBuildMode::Float32GraphThenQuantize,
            QuantizedStorageMode::QuantizedOnly,
        )
        .unwrap();
        assert_eq!(idx_f32g.graph_fingerprint(), f32_fp);
        assert!(idx_f32g.raw_base().is_none());

        let prof = idx_f32g.memory_profile();
        assert_eq!(prof.codes_bytes, 400 * m_pq);
        assert_eq!(
            prof.calibration_or_codebook_bytes,
            256 * 128 * std::mem::size_of::<f32>()
        );
        assert_eq!(prof.aux_vector_bytes, 0);
        assert_eq!(prof.raw_vectors_retained_bytes, 0);
        assert_eq!(idx_f32g.vectors().aux_memory_bytes(), 0);
        assert_eq!(
            idx_f32g.vectors().vector_memory_bytes(),
            400 * m_pq + 256 * 128 * 4
        );

        // QuantizedDirect also drops temporary SDC table immediately after build
        let idx_direct = HnswPqIndex::build(
            &base,
            Metric::L2Squared,
            cfg.clone(),
            PqConfig::new(m_pq),
            QuantizedBuildMode::QuantizedDirect,
            QuantizedStorageMode::QuantizedOnly,
        )
        .unwrap();
        let prof_direct = idx_direct.memory_profile();
        assert_eq!(prof_direct.aux_vector_bytes, 0);
        assert_eq!(idx_direct.vectors().aux_memory_bytes(), 0);
        assert_eq!(prof_direct.codes_bytes, 400 * m_pq);
        assert!(idx_direct.build_timings().total_s >= 0.0);
        let q_timings = idx_direct.quantized_build_timings();
        assert_eq!(
            q_timings.calibration_or_train_s,
            idx_direct.build_timings().train_time_s
        );
        assert_eq!(q_timings.encode_s, idx_direct.build_timings().encode_time_s);
        assert_eq!(
            q_timings.graph_build_s,
            idx_direct.build_timings().graph_build_time_s
        );

        // Verify search after dropping base clone
        let mut s = idx_f32g.new_search_scratch(40);
        let hits = idx_f32g
            .search_batch_with_scratch(&queries, 10, 40, None, &mut s)
            .unwrap();
        assert_eq!(hits.len(), queries.rows());
        assert_eq!(hits[0].len(), 10);
    }
}

#[test]
fn phase6_pq_rerank_exactness_r_eq_n_and_r_eq_k_invariance() {
    let (base, queries) = generate_synthetic_sift(200, 20, 64, 8, 314).unwrap();
    let base_arc = Arc::new(base.clone());
    let cfg = HnswConfig::new(16, 100);
    let pq_cfg = PqConfig::new(16);
    let k = 10_usize;

    for metric in [Metric::L2Squared, Metric::Euclidean, Metric::InnerProduct] {
        let exact = ExactKnn::from_base(metric, base.clone()).unwrap();
        let gt = exact.search(&queries, k).unwrap();

        let idx = HnswPqIndex::build_with_shared_raw(
            Arc::clone(&base_arc),
            metric,
            cfg.clone(),
            pq_cfg,
            QuantizedBuildMode::Float32GraphThenQuantize,
            QuantizedStorageMode::WithExactRerank,
        )
        .unwrap();
        let mut scratch = idx.new_search_scratch(base.rows());

        // 1. R = N matches ExactKnn bit-for-bit
        let timed_all = idx
            .search_batch_timed_with_scratch(&queries, k, 10, Some(base.rows()), &mut scratch)
            .unwrap();
        assert_eq!(timed_all.metrics.requested_ef_search, 10);
        assert_eq!(timed_all.metrics.effective_ef_search, base.rows());
        assert_eq!(timed_all.metrics.rerank_top_r, Some(base.rows()));

        for (qi, hits) in timed_all.results.iter().enumerate() {
            let gt_ids = gt.indices.row(qi);
            let gt_dists = gt.distances.row(qi);
            assert_eq!(hits.len(), k);
            for rank in 0..k {
                assert_eq!(
                    hits[rank].id as i32, gt_ids[rank],
                    "PQ R=N id mismatch at metric={metric}, q={qi}, rank={rank}"
                );
                assert_eq!(
                    hits[rank].distance.to_bits(),
                    gt_dists[rank].to_bits(),
                    "PQ R=N distance bits mismatch at metric={metric}, q={qi}, rank={rank}"
                );
            }
        }

        // 2. R = k leaves Recall@k identical to rerank = None
        let res_none = idx
            .search_batch_with_scratch(&queries, k, 30, None, &mut scratch)
            .unwrap();
        let res_k = idx
            .search_batch_with_scratch(&queries, k, 30, Some(k), &mut scratch)
            .unwrap();

        let knn_none = results_to_knn_search_result(&res_none).unwrap();
        let knn_k = results_to_knn_search_result(&res_k).unwrap();
        let recall_none = compute_recall_at_k(&knn_none.indices, &gt.indices, k).unwrap();
        let recall_k = compute_recall_at_k(&knn_k.indices, &gt.indices, k).unwrap();
        assert!(
            (recall_none - recall_k).abs() < 1e-12,
            "PQ R=k must leave Recall@k invariant ({recall_none} vs {recall_k})"
        );

        // 3. External raw_base helpers match stored raw_base bit-for-bit
        let idx_only = idx.clone().into_quantized_only();
        let res_ext = idx_only
            .search_batch_with_base_timed_with_scratch(
                &base,
                &queries,
                k,
                10,
                Some(base.rows()),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(res_ext.results, timed_all.results);
        let single_ext = idx_only
            .search_knn_rerank_with_base_scratch(
                &base,
                queries.row(0),
                k,
                10,
                Some(base.rows()),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(single_ext, timed_all.results[0]);

        // Re-attaching raw_base via with_raw_base restores WithExactRerank
        let reattached = idx_only.with_raw_base(Arc::clone(&base_arc)).unwrap();
        assert_eq!(
            reattached.storage_mode(),
            QuantizedStorageMode::WithExactRerank
        );
    }
}

#[test]
fn phase6_pq_zero_allocation_query_hot_path() {
    let (base, queries) = generate_synthetic_sift(450, 25, 128, 12, 999).unwrap();
    let idx = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        PqConfig::new(32),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::WithExactRerank,
    )
    .unwrap();

    let max_ef = 100_usize;
    let mut scratch = idx.new_search_scratch(max_ef);
    assert!(scratch.query_state.capacity() >= 32 * PQ_K_CENTROIDS);

    idx.warmup(queries.row(0), 10, max_ef, &mut scratch)
        .unwrap();
    let _ = idx
        .search_knn_rerank_with_scratch(queries.row(0), 10, 50, Some(50), &mut scratch)
        .unwrap();

    let cap_frontier = scratch.frontier.capacity();
    let cap_results = scratch.results.capacity();
    let cap_raw = scratch.raw_candidates.capacity();
    let cap_rerank = scratch.rerank_candidates.capacity();
    let cap_qstate = scratch.query_state.capacity();

    assert!(cap_frontier >= base.rows());
    assert!(cap_results > max_ef);
    assert!(cap_raw > max_ef);
    assert!(cap_rerank > max_ef);
    assert!(cap_qstate >= 32 * PQ_K_CENTROIDS);

    let _ = idx
        .search_batch_timed_with_scratch(&queries, 10, max_ef, None, &mut scratch)
        .unwrap();
    let _ = idx
        .search_batch_timed_with_scratch(&queries, 10, 50, Some(50), &mut scratch)
        .unwrap();

    assert_eq!(scratch.frontier.capacity(), cap_frontier);
    assert_eq!(scratch.results.capacity(), cap_results);
    assert_eq!(scratch.raw_candidates.capacity(), cap_raw);
    assert_eq!(scratch.rerank_candidates.capacity(), cap_rerank);
    assert_eq!(scratch.query_state.capacity(), cap_qstate);
}

#[test]
fn phase6_pq_quantized_direct_determinism_and_snapshot_roundtrip() {
    let tmp = TempTestDir::new("phase6_pq_snapshot");
    let snap_path = tmp.path().join("pq16_index.bin");

    let (base, queries) = generate_synthetic_sift(350, 15, 128, 12, 404).unwrap();
    let idx_a = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        PqConfig::new(16),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let idx_b = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        PqConfig::new(16),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let idx_c = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        PqConfig::new(16).with_seed(99),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let mut hnsw_other = HnswConfig::new(16, 100);
    hnsw_other.seed = 99;
    let idx_d = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        hnsw_other,
        PqConfig::new(16),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();

    assert_eq!(idx_a.graph_fingerprint(), idx_b.graph_fingerprint());
    assert_eq!(idx_a.vectors(), idx_b.vectors());
    assert_ne!(idx_a.graph_fingerprint(), idx_c.graph_fingerprint());
    assert_ne!(idx_a.graph_fingerprint(), idx_d.graph_fingerprint());

    let mut sa = idx_a.new_search_scratch(50);
    let before = idx_a
        .search_batch_with_scratch(&queries, 10, 50, None, &mut sa)
        .unwrap();

    // Write & reload snapshot
    idx_a.write_snapshot(&snap_path).unwrap();
    let reloaded = HnswPqIndex::read_snapshot(&snap_path).unwrap();
    assert!(reloaded.raw_base().is_none());
    assert_eq!(reloaded.storage_mode(), QuantizedStorageMode::QuantizedOnly);
    assert_eq!(reloaded.build_mode(), QuantizedBuildMode::QuantizedDirect);
    assert_eq!(reloaded.graph_fingerprint(), idx_a.graph_fingerprint());
    assert_eq!(reloaded.vectors(), idx_a.vectors());
    assert_eq!(reloaded.memory_profile(), idx_a.memory_profile());

    let mut sr = reloaded.new_search_scratch(50);
    let after = reloaded
        .search_batch_with_scratch(&queries, 10, 50, None, &mut sr)
        .unwrap();
    assert_eq!(before.len(), after.len());
    for (row_a, row_b) in before.iter().zip(after.iter()) {
        assert_eq!(row_a.len(), row_b.len());
        for (a, b) in row_a.iter().zip(row_b.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        }
    }

    // Corrupt snapshot guards
    let mut raw_bytes = Vec::new();
    idx_a.write_to(&mut raw_bytes).unwrap();

    let mut bad_magic = raw_bytes.clone();
    bad_magic[0] ^= 0xFF;
    assert!(matches!(
        HnswPqIndex::read_from(&mut Cursor::new(&bad_magic)),
        Err(VectorSearchError::CorruptData(_))
    ));

    let truncated = &raw_bytes[..raw_bytes.len() / 2];
    assert!(HnswPqIndex::read_from(&mut Cursor::new(truncated)).is_err());

    // Out-of-range code byte when k_effective < 256
    let small_mat = make_random_matrix(12, 32, 0.0, 1.0, 12);
    let small_vecs =
        PqVectors::from_matrix(&small_mat, Metric::L2Squared, &PqConfig::new(8)).unwrap();
    let mut vec_bytes = Vec::new();
    small_vecs.write_to(&mut vec_bytes).unwrap();
    let last = vec_bytes.len() - 1;
    vec_bytes[last] = 200; // >= k_effective (12)
    assert!(matches!(
        PqVectors::read_from(&mut Cursor::new(&vec_bytes)),
        Err(VectorSearchError::CorruptData(_))
    ));

    // PqCodebook standalone snapshot roundtrip
    let mut cb_bytes = Vec::new();
    idx_a.codebook().write_to(&mut cb_bytes).unwrap();
    let cb_reloaded = PqCodebook::read_from(&mut Cursor::new(&cb_bytes)).unwrap();
    assert_eq!(&cb_reloaded, idx_a.codebook());
}

#[test]
fn phase6_pq_streaming_fvecs_bit_identical_parity() {
    let tmp = TempTestDir::new("phase6_pq_streaming");
    // 500 rows with max_train_vectors=128 exercises deterministic subsampling parity too
    let mat = make_random_matrix(500, 64, -15.0, 45.0, 808);
    let fvecs_path = tmp.path().join("pq_stream.fvecs");
    write_fvecs(&fvecs_path, &mat).unwrap();
    let queries = make_random_matrix(10, 64, -15.0, 45.0, 809);

    let hnsw_cfg = HnswConfig::new(8, 40);
    let mut pq_cfg = PqConfig::new(16).with_seed(42);
    pq_cfg.max_train_vectors = 128;
    pq_cfg.max_lloyd_iters = 15;

    for max_rows in [None, Some(300)] {
        let expected_mat = read_fvecs_max_rows(&fvecs_path, max_rows).unwrap();
        let mem_idx = HnswPqIndex::build(
            &expected_mat,
            Metric::L2Squared,
            hnsw_cfg.clone(),
            pq_cfg,
            QuantizedBuildMode::QuantizedDirect,
            QuantizedStorageMode::QuantizedOnly,
        )
        .unwrap();
        let stream_idx = HnswPqIndex::build_streaming_from_fvecs(
            &fvecs_path,
            Metric::L2Squared,
            hnsw_cfg.clone(),
            pq_cfg,
            max_rows,
        )
        .unwrap();

        assert_eq!(mem_idx.codebook(), stream_idx.codebook());
        assert_eq!(mem_idx.vectors().codes(), stream_idx.vectors().codes());
        assert_eq!(mem_idx.graph_fingerprint(), stream_idx.graph_fingerprint());

        let mut sm = mem_idx.new_search_scratch(40);
        let mut ss = stream_idx.new_search_scratch(40);
        let rm = mem_idx
            .search_batch_with_scratch(&queries, 10, 40, None, &mut sm)
            .unwrap();
        let rs = stream_idx
            .search_batch_with_scratch(&queries, 10, 40, None, &mut ss)
            .unwrap();
        assert_eq!(rm, rs);
    }
}

#[test]
fn phase6_pq_siftsmall_smoke_if_present() {
    let Some(loaded) = common::load_siftsmall_or_skip("phase6_pq_siftsmall_smoke_if_present")
    else {
        return;
    };

    let k = 10_usize;
    let ef = 100_usize;
    let cfg = HnswConfig::new(16, 100);
    let exact = ExactKnn::from_base(Metric::L2Squared, loaded.base.clone()).unwrap();
    let gt = exact.search(&loaded.query, k).unwrap();

    for m_pq in [16_usize, 32] {
        let pq_only = HnswPqIndex::build(
            &loaded.base,
            Metric::L2Squared,
            cfg.clone(),
            PqConfig::new(m_pq),
            QuantizedBuildMode::Float32GraphThenQuantize,
            QuantizedStorageMode::QuantizedOnly,
        )
        .unwrap();
        let pq_rerank = pq_only
            .clone()
            .with_raw_base(Arc::new(loaded.base.clone()))
            .unwrap();

        let mut scratch = pq_only.new_search_scratch(200);
        let r_only = results_to_knn_search_result(
            &pq_only
                .search_batch_with_scratch(&loaded.query, k, ef, None, &mut scratch)
                .unwrap(),
        )
        .unwrap();
        let r_rerank_100 = results_to_knn_search_result(
            &pq_rerank
                .search_batch_with_scratch(&loaded.query, k, ef, Some(100), &mut scratch)
                .unwrap(),
        )
        .unwrap();

        let rec_only = compute_recall_at_k(&r_only.indices, &gt.indices, k).unwrap();
        let rec_r100 = compute_recall_at_k(&r_rerank_100.indices, &gt.indices, k).unwrap();
        assert!(rec_r100 >= rec_only);
        assert!(
            rec_r100 >= 0.90,
            "PQ{m_pq} + R=100 recall too low: {rec_r100}"
        );
        assert!(pq_only.memory_profile().steady_state_bytes_per_vector < 220.0);
    }
}

#[test]
fn phase6_pq_codebook_helpers_and_from_codebook_and_matrix() {
    let (base, queries) = generate_synthetic_sift(64, 4, 32, 4, 515).unwrap();
    let cfg = PqConfig::new(8).with_seed(42);
    let cb = PqTrainer::train(&base, &cfg).unwrap();
    let vecs = PqVectors::from_codebook_and_matrix(cb.clone(), &base, Metric::L2Squared).unwrap();
    assert_eq!(vecs.codes(), cb.encode_matrix(&base).unwrap().as_slice());

    let mut code = vec![0_u8; 8];
    cb.encode_row(base.row(0), &mut code).unwrap();
    assert_eq!(code.as_slice(), vecs.code(0));

    let mut table = Vec::new();
    cb.compute_adc_table(Metric::L2Squared, queries.row(0), &mut table)
        .unwrap();
    let d_direct = pq_adc_distance(&table, &code);
    let mut state = PqQueryState::default();
    vecs.prepare_query(queries.row(0), &mut state);
    assert_eq!(d_direct.to_bits(), vecs.query_distance(&state, 0).to_bits());

    // Error paths on codebook helpers
    assert!(matches!(
        cb.encode_row(&[0.0_f32; 16], &mut code),
        Err(VectorSearchError::DimensionMismatch {
            expected: 32,
            actual: 16
        })
    ));
    let mut bad_code = vec![0_u8; 4];
    assert!(matches!(
        cb.encode_row(base.row(0), &mut bad_code),
        Err(VectorSearchError::DimensionMismatch {
            expected: 8,
            actual: 4
        })
    ));
    let mut nan_row = base.row(0).to_vec();
    nan_row[3] = f32::NAN;
    assert!(matches!(
        cb.encode_row(&nan_row, &mut code),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        cb.compute_adc_table(Metric::L2Squared, &nan_row, &mut table),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        cb.compute_adc_table(Metric::L2Squared, &[0.0_f32; 16], &mut table),
        Err(VectorSearchError::DimensionMismatch {
            expected: 32,
            actual: 16
        })
    ));
}
