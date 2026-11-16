//! Phase 6 Stage 1 tests (`T4.1`, `T4.2`, `T4.3`):
//! - `CompactHnswGraph` parity with `f32` `HnswIndex` (`graph_fingerprint()`, topology, golden fixture)
//! - `CompactHnswGraph::build_with_source` Malkov & Yashunin Algorithms 1, 2, 4 parity
//! - `Sq8Quantizer` calibration, reconstruction error bounds, constant & extreme float ranges
//! - Safe 8-way unrolled SQ8 kernels (`Way A` `QueryShifted`, `Way B` `NormExpanded` + cancellation diagnostic,
//!   symmetric `u8-u8`, `Euclidean`, `InnerProduct`) across remainder dimensions
//! - Two-pass streaming `.fvecs` calibration & encoding bit-exact parity with in-memory `RowMatrix<f32>`
//! - `HnswSq8Index` `Sq8Only` vs `WithExactRerank`, snapshot roundtrip, `R=N` exactness, `R=k` recall
//!   invariance, zero-allocation hot-path scratch stability, `QuantizedDirect` determinism, and `siftsmall` smoke.

mod common;

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use vector_search::benchmark::verify_index_against_golden_fixture;
use vector_search::dataset::{generate_synthetic_sift, read_fvecs_max_rows, write_fvecs};
use vector_search::distance::{inner_product, l2_squared, Metric};
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::ExactKnn;
use vector_search::hnsw::{HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::quantized::{
    inner_product_sq8, l2_squared_sq8_expanded, l2_squared_sq8_shifted, l2_squared_sq8_symmetric,
    l2_squared_sq8_symmetric_from_scales, results_to_knn_search_result, CompactHnswGraph,
    HnswSq8Index, Neighbor, QuantizedBuildMode, QuantizedDistanceSource, QuantizedSearchScratch,
    QuantizedStorageMode, Sq8KernelKind, Sq8Quantizer, Sq8Vectors,
};
use vector_search::recall::compute_recall_at_k;
use vector_search::rng::DeterministicRng;

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), nanos));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp test dir");
        Self { path }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Test-only exact `f32` adapter over `QuantizedDistanceSource` used to prove
/// `CompactHnswGraph::build_with_source` produces bit-identical graphs and search results
/// to `HnswIndex::build`.
struct F32SourceAdapter<'a> {
    matrix: &'a RowMatrix<f32>,
    metric: Metric,
}

#[derive(Default)]
struct F32QueryState {
    vec: Vec<f32>,
}

impl QuantizedDistanceSource for F32SourceAdapter<'_> {
    type QueryState = F32QueryState;

    fn len(&self) -> usize {
        self.matrix.rows()
    }

    fn dim(&self) -> usize {
        self.matrix.cols()
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn prepare_query(&self, query: &[f32], state: &mut Self::QueryState) {
        state.vec.clear();
        state.vec.extend_from_slice(query);
    }

    fn query_distance(&self, state: &Self::QueryState, node_id: usize) -> f32 {
        let row = self.matrix.row(node_id);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => l2_squared(&state.vec, row),
            Metric::InnerProduct => {
                let nd = -inner_product(&state.vec, row);
                if nd == 0.0 {
                    0.0
                } else {
                    nd
                }
            }
        }
    }

    fn node_distance(&self, id_a: usize, id_b: usize) -> f32 {
        if id_a == id_b {
            return 0.0;
        }
        let a = self.matrix.row(id_a);
        let b = self.matrix.row(id_b);
        match self.metric {
            Metric::L2Squared | Metric::Euclidean => l2_squared(a, b),
            Metric::InnerProduct => {
                let nd = -inner_product(a, b);
                if nd == 0.0 {
                    0.0
                } else {
                    nd
                }
            }
        }
    }

    fn format_output_distance(&self, sort_key: f32) -> f32 {
        match self.metric {
            Metric::L2Squared => {
                if sort_key <= 0.0 {
                    0.0
                } else {
                    sort_key
                }
            }
            Metric::Euclidean => {
                let d = sort_key.max(0.0).sqrt();
                if d <= 0.0 {
                    0.0
                } else {
                    d
                }
            }
            Metric::InnerProduct => {
                let score = -sort_key;
                if score == 0.0 {
                    0.0
                } else {
                    score
                }
            }
        }
    }

    fn vector_memory_bytes(&self) -> usize {
        self.matrix.memory_bytes()
    }
}

fn make_random_matrix(rows: usize, cols: usize, lo: f64, hi: f64, seed: u64) -> RowMatrix<f32> {
    let mut rng = DeterministicRng::new(seed);
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        let u = rng.next_f64_open01();
        data.push((lo + u * (hi - lo)) as f32);
    }
    RowMatrix::new(rows, cols, data).unwrap()
}

fn assert_compact_graph_matches_hnsw(compact: &CompactHnswGraph, hnsw: &HnswIndex<'_>) {
    assert_eq!(compact.num_vectors(), hnsw.num_vectors());
    assert_eq!(compact.entry_point(), hnsw.entry_point());
    assert_eq!(compact.max_layer(), hnsw.max_layer());
    assert_eq!(compact.metric(), hnsw.metric());
    assert_eq!(compact.graph_fingerprint(), hnsw.graph_fingerprint());

    for id in 0..hnsw.num_vectors() {
        let max_l = hnsw.node_layer(id);
        assert_eq!(compact.node_layer(id), max_l);
        for lc in 0..=max_l {
            assert_eq!(
                compact.neighbors(lc, id),
                hnsw.neighbors(lc, id),
                "Neighbor mismatch at node {id}, layer {lc}"
            );
        }
    }
}

// ============================================================================
// 1. T4.1: f32 Preservation & CompactHnswGraph Parity
// ============================================================================

#[test]
fn phase6_compact_graph_from_hnsw_matches_fingerprint_and_topology() {
    // Empty graph
    let empty = RowMatrix::<f32>::empty();
    let empty_idx = HnswIndex::build(&empty, Metric::L2Squared, HnswConfig::default()).unwrap();
    let empty_compact = CompactHnswGraph::from_hnsw_index(&empty_idx);
    assert_compact_graph_matches_hnsw(&empty_compact, &empty_idx);

    // Single vector
    let single = RowMatrix::from_rows(&[vec![1.0_f32, -2.0, 3.5, 0.0]]).unwrap();
    let single_idx = HnswIndex::build(&single, Metric::L2Squared, HnswConfig::default()).unwrap();
    let single_compact = CompactHnswGraph::from_hnsw_index(&single_idx);
    assert_compact_graph_matches_hnsw(&single_compact, &single_idx);

    // Multi-layer graphs across config variants
    let (base, _) = generate_synthetic_sift(400, 10, 32, 8, 2026).unwrap();
    let configs = [
        HnswConfig {
            m: 8,
            ef_construction: 40,
            seed: 42,
            ..Default::default()
        },
        HnswConfig {
            m: 16,
            ef_construction: 100,
            extend_candidates: true,
            keep_pruned: true,
            keep_pruned_shrink: Some(false),
            seed: 77,
            ..Default::default()
        },
    ];
    for cfg in configs {
        let idx = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
        let compact = CompactHnswGraph::from_hnsw_index(&idx);
        assert_compact_graph_matches_hnsw(&compact, &idx);
        assert_eq!(
            compact.graph_only_memory_bytes(),
            compact.node_meta_bytes()
                + compact.layer0_edges_bytes()
                + compact.upper_layers_edges_bytes()
        );
    }
}

#[test]
fn phase6_f32_golden_10k_preservation_and_compact_graph_parity() {
    let base_path = Path::new("data/sift1m/sift_base.fvecs");
    let query_path = Path::new("data/sift1m/sift_query.fvecs");
    let golden_path = Path::new("results/phase3_5/golden_10k.json");
    if !base_path.exists() || !query_path.exists() || !golden_path.exists() {
        eprintln!(
            "SKIP phase6_f32_golden_10k_preservation_and_compact_graph_parity: sift1m/golden_10k not present"
        );
        return;
    }

    let base = read_fvecs_max_rows(base_path, Some(10_000)).unwrap();
    let queries = read_fvecs_max_rows(query_path, Some(1_000)).unwrap();
    let cfg = HnswConfig {
        m: 16,
        m_max0: Some(32),
        ef_construction: 100,
        seed: 42,
        ..Default::default()
    };
    let idx = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    verify_index_against_golden_fixture(&idx, &queries, golden_path).unwrap();
    assert_eq!(idx.graph_fingerprint(), 0x7eb9_ba68_df12_b0a0);

    let compact = CompactHnswGraph::from_hnsw_index(&idx);
    assert_eq!(compact.graph_fingerprint(), idx.graph_fingerprint());
}

#[test]
fn phase6_build_with_source_f32_adapter_matches_hnsw_index_bit_for_bit() {
    let (base, queries) = generate_synthetic_sift(350, 20, 24, 6, 99).unwrap();

    for metric in [Metric::L2Squared, Metric::Euclidean, Metric::InnerProduct] {
        for kp_shrink in [None, Some(false)] {
            let cfg = HnswConfig {
                m: 8,
                ef_construction: 50,
                keep_pruned: true,
                keep_pruned_shrink: kp_shrink,
                seed: 42,
                ..Default::default()
            };
            let mut hnsw = HnswIndex::build(&base, metric, cfg.clone()).unwrap();
            let adapter = F32SourceAdapter {
                matrix: &base,
                metric,
            };
            let compact = CompactHnswGraph::build_with_source(&adapter, cfg).unwrap();
            assert_compact_graph_matches_hnsw(&compact, &hnsw);

            let mut scratch = QuantizedSearchScratch::new(
                compact.num_vectors(),
                50,
                compact.config().effective_m_max0(),
            );
            for q in queries.iter_rows() {
                let ref_hits = hnsw.search_knn(q, 10, 50).unwrap();
                let new_hits = compact
                    .search_knn_with_scratch(&adapter, q, 10, 50, &mut scratch)
                    .unwrap();
                assert_eq!(ref_hits.len(), new_hits.len());
                for (r, n) in ref_hits.iter().zip(new_hits.iter()) {
                    assert_eq!(r.1, n.id);
                    assert_eq!(r.0.to_bits(), n.distance.to_bits());
                }
            }
        }
    }
}

#[test]
fn phase6_compact_graph_snapshot_roundtrip_and_corruption_guards() {
    let (base, _) = generate_synthetic_sift(150, 5, 16, 4, 17).unwrap();
    let cfg = HnswConfig::new(8, 40);
    let idx = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let compact = CompactHnswGraph::from_hnsw_index(&idx);

    let mut buf = Vec::new();
    compact.write_to(&mut buf).unwrap();
    let reloaded = CompactHnswGraph::read_from(&mut Cursor::new(&buf)).unwrap();
    assert_compact_graph_matches_hnsw(&reloaded, &idx);

    // Corrupt magic
    let mut bad_magic = buf.clone();
    bad_magic[0] ^= 0xFF;
    assert!(matches!(
        CompactHnswGraph::read_from(&mut Cursor::new(&bad_magic)),
        Err(VectorSearchError::CorruptData(_))
    ));

    // Truncated stream
    let truncated = &buf[..buf.len() / 2];
    assert!(CompactHnswGraph::read_from(&mut Cursor::new(truncated)).is_err());
}

// ============================================================================
// 2. T4.2: SQ8 Mathematical Bounds, Kernels & Edge Cases
// ============================================================================

#[test]
fn phase6_sq8_reconstruction_error_bound() {
    let datasets = [
        make_random_matrix(120, 128, 0.0, 255.0, 11),
        make_random_matrix(120, 64, -500.0, 1200.0, 22),
        generate_synthetic_sift(150, 5, 37, 5, 33).unwrap().0,
    ];

    for mat in &datasets {
        let q = Sq8Quantizer::calibrate(mat).unwrap();
        let mut mins = vec![f32::INFINITY; mat.cols()];
        let mut maxs = vec![f32::NEG_INFINITY; mat.cols()];
        for row in mat.iter_rows() {
            for (j, &v) in row.iter().enumerate() {
                mins[j] = mins[j].min(v);
                maxs[j] = maxs[j].max(v);
            }
        }

        let mut code = vec![0_u8; mat.cols()];
        let mut recon = vec![0.0_f32; mat.cols()];
        for row in mat.iter_rows() {
            q.encode_row(row, &mut code).unwrap();
            q.dequantize_row(&code, &mut recon).unwrap();
            for j in 0..mat.cols() {
                let span = (maxs[j] as f64) - (mins[j] as f64);
                let bound = span / 510.0 + 1e-4 * span.max(1.0);
                let err = ((row[j] as f64) - (recon[j] as f64)).abs();
                assert!(
                    err <= bound,
                    "Reconstruction error {err} exceeded bound {bound} at dim {j} (span={span})"
                );
            }
        }
    }
}

#[test]
fn phase6_sq8_way_a_f64_parity_across_dimensions() {
    let dims = [1_usize, 3, 7, 8, 9, 16, 128, 131];

    for &dim in &dims {
        let base = make_random_matrix(40, dim, 0.0, 255.0, 100 + dim as u64);
        let queries = make_random_matrix(10, dim, 0.0, 255.0, 200 + dim as u64);

        for metric in [Metric::L2Squared, Metric::Euclidean, Metric::InnerProduct] {
            let vecs = Sq8Vectors::from_matrix(&base, metric).unwrap();
            let quant = vecs.quantizer();
            let mut state = Default::default();
            let mut recon = vec![0.0_f32; dim];

            for q_row in queries.iter_rows() {
                vecs.prepare_query(q_row, &mut state);

                // Also test direct l2_squared_sq8_shifted & inner_product_sq8 helper parity
                for id in 0..base.rows() {
                    let code = vecs.code(id);
                    quant.dequantize_row(code, &mut recon).unwrap();
                    let actual = vecs.format_output_distance(vecs.query_distance(&state, id));

                    let expected_f64 = match metric {
                        Metric::L2Squared => {
                            let mut sum = 0.0_f64;
                            for j in 0..dim {
                                let d = f64::from(q_row[j]) - f64::from(recon[j]);
                                sum += d * d;
                            }
                            let shifted =
                                l2_squared_sq8_shifted(&state.q_shifted, quant.scales(), code);
                            assert!((shifted - actual).abs() <= 1e-6);
                            sum
                        }
                        Metric::Euclidean => {
                            let mut sum = 0.0_f64;
                            for j in 0..dim {
                                let d = f64::from(q_row[j]) - f64::from(recon[j]);
                                sum += d * d;
                            }
                            sum.sqrt()
                        }
                        Metric::InnerProduct => {
                            let mut dot = 0.0_f64;
                            for j in 0..dim {
                                dot += f64::from(q_row[j]) * f64::from(recon[j]);
                            }
                            let direct_ip = inner_product_sq8(state.scalar, &state.q_shifted, code);
                            assert!((direct_ip - actual).abs() <= 1e-6);
                            dot
                        }
                    };

                    let tol = 1e-4 * expected_f64.abs().max(1.0);
                    assert!(
                        (f64::from(actual) - expected_f64).abs() <= tol,
                        "Way A parity failure dim={dim}, metric={metric}: actual={actual}, f64={expected_f64}"
                    );
                }
            }
        }
    }
}

#[test]
fn phase6_sq8_way_b_parity_and_cancellation_diagnostic() {
    // 1. On normal SIFT-like [0, 255] distributions, Way B matches Way A and f64 reference
    for &dim in &[1_usize, 7, 8, 13, 128, 131] {
        let base = make_random_matrix(30, dim, 0.0, 255.0, 300 + dim as u64);
        let queries = make_random_matrix(5, dim, 0.0, 255.0, 400 + dim as u64);

        let vecs_a = Sq8Vectors::from_matrix_with_kernel(
            &base,
            Metric::L2Squared,
            Sq8KernelKind::QueryShifted,
        )
        .unwrap();
        let vecs_b = Sq8Vectors::from_matrix_with_kernel(
            &base,
            Metric::L2Squared,
            Sq8KernelKind::NormExpanded,
        )
        .unwrap();
        assert!(vecs_a.base_norms_sq().is_none());
        assert_eq!(vecs_b.base_norms_sq().unwrap().len(), base.rows());

        let mut st_a = Default::default();
        let mut st_b = Default::default();
        for q_row in queries.iter_rows() {
            vecs_a.prepare_query(q_row, &mut st_a);
            vecs_b.prepare_query(q_row, &mut st_b);
            for id in 0..base.rows() {
                let da = vecs_a.query_distance(&st_a, id);
                let db = vecs_b.query_distance(&st_b, id);
                let base_norm_sq = vecs_b.base_norms_sq().unwrap()[id];
                let direct_b = l2_squared_sq8_expanded(
                    st_b.scalar,
                    &st_b.q_shifted,
                    vecs_b.code(id),
                    base_norm_sq,
                );
                assert_eq!(db.to_bits(), direct_b.to_bits());
                // Way B computes (||q'||^2 + ||s*c||^2 - 2*dot), so rounding scales with
                // (||q'||^2 + ||s*c||^2) rather than with the residual distance da.
                let energy_scale =
                    (f64::from(st_b.scalar) + f64::from(base_norm_sq) + f64::from(da)).max(1.0);
                let abs_diff = (f64::from(da) - f64::from(db)).abs();
                assert!(
                    abs_diff <= 1e-5 * energy_scale,
                    "Way A vs Way B exceeded f32 rounding bound on [0,255] dim={dim}: da={da}, db={db}, energy={energy_scale}"
                );
            }
        }
    }

    // 2. Microcheck on dim=128 [0, 255]: compare accuracy + kernel throughput of Way A vs Way B
    {
        let base_128 = make_random_matrix(2_000, 128, 0.0, 255.0, 450);
        let queries_128 = make_random_matrix(50, 128, 0.0, 255.0, 451);
        let vecs_a = Sq8Vectors::from_matrix_with_kernel(
            &base_128,
            Metric::L2Squared,
            Sq8KernelKind::QueryShifted,
        )
        .unwrap();
        let vecs_b = Sq8Vectors::from_matrix_with_kernel(
            &base_128,
            Metric::L2Squared,
            Sq8KernelKind::NormExpanded,
        )
        .unwrap();
        let mut st_a = Default::default();
        let mut st_b = Default::default();

        let t_a = std::time::Instant::now();
        let mut acc_a = 0.0_f64;
        for q_row in queries_128.iter_rows() {
            vecs_a.prepare_query(q_row, &mut st_a);
            for id in 0..base_128.rows() {
                acc_a += f64::from(std::hint::black_box(vecs_a.query_distance(&st_a, id)));
            }
        }
        let elapsed_a_ms = t_a.elapsed().as_secs_f64() * 1000.0;

        let t_b = std::time::Instant::now();
        let mut acc_b = 0.0_f64;
        for q_row in queries_128.iter_rows() {
            vecs_b.prepare_query(q_row, &mut st_b);
            for id in 0..base_128.rows() {
                acc_b += f64::from(std::hint::black_box(vecs_b.query_distance(&st_b, id)));
            }
        }
        let elapsed_b_ms = t_b.elapsed().as_secs_f64() * 1000.0;

        let mut max_rel_diff = 0.0_f64;
        for q_row in queries_128.iter_rows() {
            vecs_a.prepare_query(q_row, &mut st_a);
            vecs_b.prepare_query(q_row, &mut st_b);
            for id in 0..base_128.rows() {
                let da = f64::from(vecs_a.query_distance(&st_a, id));
                let db = f64::from(vecs_b.query_distance(&st_b, id));
                max_rel_diff = max_rel_diff.max((da - db).abs() / da.max(1.0));
            }
        }
        assert!(
            max_rel_diff < 1e-4,
            "dim=128 Way A vs Way B rel diff: {max_rel_diff}"
        );
        eprintln!(
            "[T4.2 Way A vs Way B Microcheck 2k x 50q d=128] way_a={elapsed_a_ms:.3} ms (sum={acc_a:.1}), way_b={elapsed_b_ms:.3} ms (sum={acc_b:.1}), max_rel_diff={max_rel_diff:.3e}"
        );
    }

    // 3. Adversarial cancellation diagnostic (DESIGN.md §6.2):
    // Row 0 sits at 0.0 (anchoring min = 0.0), while cluster vectors & query sit at 10_000.0 + eps.
    let dim = 128_usize;
    let mut rows = Vec::new();
    rows.push(vec![0.0_f32; dim]);
    let mut rng = DeterministicRng::new(9001);
    for _ in 1..40 {
        let r: Vec<f32> = (0..dim)
            .map(|_| 10_000.0 + (rng.next_f64_open01() as f32) * 0.5)
            .collect();
        rows.push(r);
    }
    let adv_base = RowMatrix::from_rows(&rows).unwrap();
    let adv_query: Vec<f32> = (0..dim)
        .map(|_| 10_000.0 + (rng.next_f64_open01() as f32) * 0.5)
        .collect();

    let vecs_a = Sq8Vectors::from_matrix_with_kernel(
        &adv_base,
        Metric::L2Squared,
        Sq8KernelKind::QueryShifted,
    )
    .unwrap();
    let vecs_b = Sq8Vectors::from_matrix_with_kernel(
        &adv_base,
        Metric::L2Squared,
        Sq8KernelKind::NormExpanded,
    )
    .unwrap();

    let mut st_a = Default::default();
    let mut st_b = Default::default();
    vecs_a.prepare_query(&adv_query, &mut st_a);
    vecs_b.prepare_query(&adv_query, &mut st_b);

    let mut recon = vec![0.0_f32; dim];
    let mut max_rel_err_a = 0.0_f64;
    let mut max_rel_err_b = 0.0_f64;
    for id in 1..adv_base.rows() {
        vecs_a
            .quantizer()
            .dequantize_row(vecs_a.code(id), &mut recon)
            .unwrap();
        let ref_f64: f64 = adv_query
            .iter()
            .zip(recon.iter())
            .map(|(&q, &r)| {
                let d = f64::from(q) - f64::from(r);
                d * d
            })
            .sum();
        let da = f64::from(vecs_a.query_distance(&st_a, id));
        let db = f64::from(vecs_b.query_distance(&st_b, id));
        max_rel_err_a = max_rel_err_a.max((da - ref_f64).abs() / ref_f64.max(1e-6));
        max_rel_err_b = max_rel_err_b.max((db - ref_f64).abs() / ref_f64.max(1e-6));
    }

    eprintln!(
        "[T4.2 Way A vs Way B Cancellation Diagnostic] max_rel_err_way_a={max_rel_err_a:.3e}, max_rel_err_way_b={max_rel_err_b:.3e}"
    );
    assert!(
        max_rel_err_a < 1e-4,
        "Way A (QueryShifted) lost precision under large offset: {max_rel_err_a}"
    );
    assert!(
        max_rel_err_b > 1e-2 && max_rel_err_b > 100.0 * max_rel_err_a,
        "Expected Way B cancellation drift >> Way A (got Way B={max_rel_err_b}, Way A={max_rel_err_a})"
    );
}

#[test]
fn phase6_sq8_constant_and_extreme_dimensions() {
    // Heterogeneous + constant + subnormal + extreme dynamic ranges
    let rows = vec![
        vec![0.0_f32, -42.5, 0.0, -3.0e38, 1.0e-40],
        vec![0.0_f32, -42.5, 1.0e-5, 3.0e38, 2.0e-40],
        vec![0.0_f32, -42.5, 5.0e-6, 1.0e38, 1.5e-40],
    ];
    let mat = RowMatrix::from_rows(&rows).unwrap();
    let vecs = Sq8Vectors::from_matrix(&mat, Metric::L2Squared).unwrap();
    let q = vecs.quantizer();

    // Dim 0 and Dim 1 are constant -> scale == 0, inv_scale == 0, code == 0
    assert_eq!(q.scales()[0], 0.0);
    assert_eq!(q.inv_scales()[0], 0.0);
    assert_eq!(q.scales()[1], 0.0);
    assert_eq!(q.inv_scales()[1], 0.0);
    for id in 0..3 {
        assert_eq!(vecs.code(id)[0], 0);
        assert_eq!(vecs.code(id)[1], 0);
    }

    // Dim 3 has span 6e38 -> finite scale & inv_scale, non-finite scales_sq handled safely
    assert!(q.scales()[3].is_finite() && q.scales()[3] > 0.0);
    assert!(q.inv_scales()[3].is_finite() && q.inv_scales()[3] > 0.0);
    assert!(q.has_non_finite_scales_sq());
    // Row 2 has 1e38 -> fraction 4/6 * 255 = 170 (not clamped to 255 by f32 subtraction overflow!)
    assert!((i32::from(vecs.code(2)[3]) - 170).abs() <= 1);

    // Dim 4 subnormal [1e-40, 2e-40] -> midpoint encodes near 128
    assert!((i32::from(vecs.code(2)[4]) - 128).abs() <= 1);

    // Symmetric distances must never produce NaN on extreme ranges (inf * 0.0 guard)
    for a in 0..3 {
        let d_self = vecs.node_distance(a, a);
        assert_eq!(d_self, 0.0);
        assert!(!d_self.is_nan());
    }
    let d_01 = vecs.node_distance(0, 1);
    let d_10 = vecs.node_distance(1, 0);
    assert_eq!(d_01.to_bits(), d_10.to_bits());
    assert!(!d_01.is_nan());

    // Single-vector dataset (N = 1)
    let single = RowMatrix::from_rows(&[vec![1.5_f32, -2.5, 0.0, 9.0]]).unwrap();
    let idx = HnswSq8Index::build(
        &single,
        Metric::L2Squared,
        HnswConfig::default(),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let mut scratch = idx.new_search_scratch(10);
    let hits = idx
        .search_knn_with_scratch(&[1.5_f32, -2.5, 0.0, 9.0], 10, 10, &mut scratch)
        .unwrap();
    assert_eq!(hits, vec![Neighbor::new(0, 0.0)]);
}

#[test]
fn phase6_sq8_symmetric_distance_axioms() {
    for &dim in &[1_usize, 7, 8, 9, 128, 131] {
        let base = make_random_matrix(25, dim, -50.0, 200.0, 500 + dim as u64);
        let vecs = Sq8Vectors::from_matrix(&base, Metric::L2Squared).unwrap();
        let mut recon_a = vec![0.0_f32; dim];
        let mut recon_b = vec![0.0_f32; dim];

        for a in 0..base.rows() {
            assert_eq!(vecs.node_distance(a, a), 0.0);
            vecs.quantizer()
                .dequantize_row(vecs.code(a), &mut recon_a)
                .unwrap();

            for b in 0..base.rows() {
                let d_ab = vecs.node_distance(a, b);
                let d_ba = vecs.node_distance(b, a);
                assert!(d_ab >= 0.0 && d_ab.is_finite());
                assert_eq!(d_ab.to_bits(), d_ba.to_bits());

                let direct = l2_squared_sq8_symmetric(
                    vecs.quantizer().scales_sq(),
                    vecs.code(a),
                    vecs.code(b),
                );
                let from_scales = l2_squared_sq8_symmetric_from_scales(
                    vecs.quantizer().scales(),
                    vecs.code(a),
                    vecs.code(b),
                );
                assert!(
                    (d_ab - from_scales).abs() <= 1e-4 * d_ab.max(1.0),
                    "Symmetric scales fallback diverged: {d_ab} vs {from_scales}"
                );
                if a != b {
                    assert_eq!(d_ab.to_bits(), direct.to_bits());
                }

                vecs.quantizer()
                    .dequantize_row(vecs.code(b), &mut recon_b)
                    .unwrap();
                let ref_f64: f64 = recon_a
                    .iter()
                    .zip(recon_b.iter())
                    .map(|(&xa, &xb)| {
                        let d = f64::from(xa) - f64::from(xb);
                        d * d
                    })
                    .sum();
                assert!((f64::from(d_ab) - ref_f64).abs() <= 1e-4 * ref_f64.max(1.0));
            }
        }
    }
}

#[test]
fn phase6_sq8_streaming_fvecs_bit_identical_parity() {
    let tmp = TempDirGuard::new("phase6_sq8_streaming");

    // 9,005 rows exercises > 2 full 4,096-row streaming chunks + remainder chunk.
    for &dim in &[13_usize, 128] {
        let mat = make_random_matrix(9_005, dim, -10.0, 255.0, 700 + dim as u64);
        let fvecs_path = tmp.path.join(format!("vecs_d{dim}.fvecs"));
        write_fvecs(&fvecs_path, &mat).unwrap();

        for max_rows in [None, Some(5_000)] {
            let expected_mat = read_fvecs_max_rows(&fvecs_path, max_rows).unwrap();
            for kernel in [Sq8KernelKind::QueryShifted, Sq8KernelKind::NormExpanded] {
                let in_mem =
                    Sq8Vectors::from_matrix_with_kernel(&expected_mat, Metric::L2Squared, kernel)
                        .unwrap();
                let streamed = Sq8Vectors::from_fvecs_streaming_with_kernel(
                    &fvecs_path,
                    Metric::L2Squared,
                    kernel,
                    max_rows,
                )
                .unwrap();

                assert_eq!(streamed.len(), in_mem.len());
                assert_eq!(streamed.dim(), in_mem.dim());
                assert_eq!(streamed.quantizer(), in_mem.quantizer());
                assert_eq!(streamed.codes(), in_mem.codes());
                assert_eq!(streamed.base_norms_sq(), in_mem.base_norms_sq());
            }
        }
    }

    // Verify full streaming HnswSq8Index matches in-memory QuantizedDirect bit-for-bit
    let small_mat = make_random_matrix(300, 16, 0.0, 100.0, 808);
    let small_path = tmp.path.join("small_build.fvecs");
    write_fvecs(&small_path, &small_mat).unwrap();
    let queries = make_random_matrix(10, 16, 0.0, 100.0, 809);

    let cfg = HnswConfig::new(8, 40);
    let idx_mem = HnswSq8Index::build(
        &small_mat,
        Metric::L2Squared,
        cfg.clone(),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let idx_stream =
        HnswSq8Index::build_streaming_from_fvecs(&small_path, Metric::L2Squared, cfg, None)
            .unwrap();
    assert_eq!(idx_mem.graph_fingerprint(), idx_stream.graph_fingerprint());

    let mut s1 = idx_mem.new_search_scratch(40);
    let mut s2 = idx_stream.new_search_scratch(40);
    let res1 = idx_mem
        .search_batch_with_scratch(&queries, 10, 40, None, &mut s1)
        .unwrap();
    let res2 = idx_stream
        .search_batch_with_scratch(&queries, 10, 40, None, &mut s2)
        .unwrap();
    assert_eq!(res1, res2);

    // Streaming error guards
    assert!(matches!(
        Sq8Vectors::from_fvecs_streaming(&small_path, Metric::L2Squared, Some(0)),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    let empty_fvecs = tmp.path.join("empty.fvecs");
    fs::write(&empty_fvecs, []).unwrap();
    assert!(matches!(
        Sq8Vectors::from_fvecs_streaming(&empty_fvecs, Metric::L2Squared, None),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    let corrupt_fvecs = tmp.path.join("corrupt.fvecs");
    fs::write(&corrupt_fvecs, [1_u8, 2, 3]).unwrap();
    assert!(matches!(
        Sq8Vectors::from_fvecs_streaming(&corrupt_fvecs, Metric::L2Squared, None),
        Err(VectorSearchError::CorruptData(_))
    ));

    let mut nan_mat = make_random_matrix(10, 8, 0.0, 1.0, 12);
    nan_mat.as_mut_slice()[3] = f32::NAN;
    let nan_path = tmp.path.join("nan.fvecs");
    write_fvecs(&nan_path, &nan_mat).unwrap();
    assert!(matches!(
        Sq8Vectors::from_fvecs_streaming(&nan_path, Metric::L2Squared, None),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

#[test]
fn phase6_sq8_non_finite_and_invalid_argument_guards() {
    // 1. Base non-finite rejection
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut mat = make_random_matrix(16, 8, 0.0, 10.0, 55);
        mat.as_mut_slice()[11] = bad;
        assert!(matches!(
            Sq8Quantizer::calibrate(&mat),
            Err(VectorSearchError::InvalidArgument(_))
        ));
        assert!(matches!(
            Sq8Vectors::from_matrix(&mat, Metric::L2Squared),
            Err(VectorSearchError::InvalidArgument(_))
        ));
        assert!(matches!(
            HnswSq8Index::build(
                &mat,
                Metric::L2Squared,
                HnswConfig::default(),
                QuantizedBuildMode::Float32GraphThenQuantize,
                QuantizedStorageMode::QuantizedOnly
            ),
            Err(VectorSearchError::InvalidArgument(_))
        ));
    }

    // Empty matrix rejection
    assert!(matches!(
        Sq8Quantizer::calibrate(&RowMatrix::empty()),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // 2. Query guards on populated index
    let (base, _) = generate_synthetic_sift(64, 4, 8, 4, 71).unwrap();
    let idx = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(8, 20),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::WithExactRerank,
    )
    .unwrap();
    let mut scratch = idx.new_search_scratch(20);

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut q = vec![0.5_f32; 8];
        q[4] = bad;
        assert!(matches!(
            idx.search_knn_with_scratch(&q, 5, 20, &mut scratch),
            Err(VectorSearchError::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        idx.search_knn_with_scratch(&[1.0_f32; 4], 5, 20, &mut scratch),
        Err(VectorSearchError::DimensionMismatch {
            expected: 8,
            actual: 4
        })
    ));
    assert!(matches!(
        idx.search_knn_with_scratch(&[1.0_f32; 8], 0, 20, &mut scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        idx.search_knn_rerank_with_scratch(&[1.0_f32; 8], 5, 20, Some(0), &mut scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // 3. Rerank requested on QuantizedOnly index must return Err
    let idx_only = idx.clone().into_quantized_only();
    assert!(matches!(
        idx_only.search_knn_rerank_with_scratch(&[1.0_f32; 8], 5, 20, Some(10), &mut scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // 4. Empty CompactHnswGraph must STILL reject NaN/Inf queries, k=0, rerank=0, and dim mismatch
    let empty_mat = RowMatrix::<f32>::empty();
    let empty_hnsw =
        HnswIndex::build(&empty_mat, Metric::L2Squared, HnswConfig::default()).unwrap();
    let empty_compact = CompactHnswGraph::from_hnsw_index(&empty_hnsw);
    let empty_src_mat = RowMatrix::new(0, 0, vec![]).unwrap();
    let adapter = F32SourceAdapter {
        matrix: &empty_src_mat,
        metric: Metric::L2Squared,
    };
    let mut f32_scratch = QuantizedSearchScratch::default();
    assert!(matches!(
        empty_compact.search_knn_with_scratch(&adapter, &[f32::NAN; 8], 5, 10, &mut f32_scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        empty_compact.search_knn_with_scratch(&adapter, &[1.0_f32; 8], 0, 10, &mut f32_scratch),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        empty_compact.search_knn_rerank_with_scratch(
            &adapter,
            &empty_src_mat,
            &[1.0_f32; 8],
            5,
            10,
            Some(0),
            &mut f32_scratch,
        ),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

// ============================================================================
// 3. T4.3: SQ8 Index Ownership, Snapshot, Reranking, & Zero-Alloc Tests
// ============================================================================

#[test]
fn phase6_sq8_only_drop_raw_and_snapshot_roundtrip() {
    let tmp = TempDirGuard::new("phase6_sq8_snapshot");
    let snap_path = tmp.path.join("sq8_index.bin");

    let (base, queries) = generate_synthetic_sift(400, 15, 128, 8, 42).unwrap();
    let cfg = HnswConfig::new(16, 100);
    let f32_idx = HnswIndex::build(&base, Metric::L2Squared, cfg.clone()).unwrap();
    let f32_fp = f32_idx.graph_fingerprint();
    let f32_bytes_per_vec = f32_idx.graph_only_bytes_per_vector() + (128.0 * 4.0);
    drop(f32_idx);

    let idx = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        cfg,
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    // Explicitly drop raw base matrix to verify lifetime-free ownership
    drop(base);

    assert!(idx.raw_base().is_none());
    assert_eq!(idx.graph_fingerprint(), f32_fp);
    let profile = idx.memory_profile();
    assert_eq!(profile.raw_vectors_retained_bytes, 0);
    assert_eq!(profile.codes_bytes, 400 * 128);
    assert!(
        profile.steady_state_bytes_per_vector < f32_bytes_per_vec * 0.50,
        "Expected >50% payload reduction: sq8={} vs f32={}",
        profile.steady_state_bytes_per_vector,
        f32_bytes_per_vec
    );

    let mut scratch = idx.new_search_scratch(50);
    let before = idx
        .search_batch_with_scratch(&queries, 10, 50, None, &mut scratch)
        .unwrap();

    idx.write_snapshot(&snap_path).unwrap();
    let reloaded = HnswSq8Index::read_snapshot(&snap_path).unwrap();
    assert!(reloaded.raw_base().is_none());
    assert_eq!(reloaded.storage_mode(), QuantizedStorageMode::QuantizedOnly);
    assert_eq!(reloaded.graph_fingerprint(), idx.graph_fingerprint());
    assert_eq!(reloaded.memory_profile(), profile);

    let mut scratch2 = reloaded.new_search_scratch(50);
    let after = reloaded
        .search_batch_with_scratch(&queries, 10, 50, None, &mut scratch2)
        .unwrap();
    assert_eq!(before.len(), after.len());
    for (row_a, row_b) in before.iter().zip(after.iter()) {
        assert_eq!(row_a.len(), row_b.len());
        for (a, b) in row_a.iter().zip(row_b.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        }
    }
}

#[test]
fn phase6_sq8_rerank_exactness_r_eq_n_and_r_eq_k_invariance() {
    let (base, queries) = generate_synthetic_sift(200, 20, 32, 6, 314).unwrap();
    let base_arc = Arc::new(base.clone());
    let cfg = HnswConfig::new(16, 100);
    let k = 10_usize;

    for metric in [Metric::L2Squared, Metric::Euclidean, Metric::InnerProduct] {
        let exact = ExactKnn::from_base(metric, base.clone()).unwrap();
        let gt = exact.search(&queries, k).unwrap();

        let idx = HnswSq8Index::build_with_shared_raw(
            Arc::clone(&base_arc),
            metric,
            cfg.clone(),
            QuantizedBuildMode::Float32GraphThenQuantize,
            QuantizedStorageMode::WithExactRerank,
            Sq8KernelKind::QueryShifted,
        )
        .unwrap();
        let mut scratch = idx.new_search_scratch(base.rows());

        // 1. R = N returns exact ExactKnn results bit-for-bit
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
                    "R=N id mismatch at metric={metric}, q={qi}, rank={rank}"
                );
                assert_eq!(
                    hits[rank].distance.to_bits(),
                    gt_dists[rank].to_bits(),
                    "R=N distance bits mismatch at metric={metric}, q={qi}, rank={rank}"
                );
            }
        }

        // 2. R = k gives identical Recall@k to rerank = None (reorders only the same k IDs)
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
            "R=k must leave Recall@k invariant ({recall_none} vs {recall_k})"
        );
    }
}

#[test]
fn phase6_sq8_zero_allocation_query_hot_path() {
    let (base, queries) = generate_synthetic_sift(500, 30, 128, 8, 999).unwrap();
    let idx = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::WithExactRerank,
    )
    .unwrap();

    let max_ef = 100_usize;
    let mut scratch = idx.new_search_scratch(max_ef);
    // Run 1 warmup query (populates query_state capacity = dim)
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
    assert!(cap_qstate >= 128);

    // Run batch searches across both pure quantized and rerank modes
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
fn phase6_sq8_quantized_direct_determinism() {
    let (base, queries) = generate_synthetic_sift(300, 15, 64, 6, 404).unwrap();

    let idx_a = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let idx_b = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        HnswConfig::new(16, 100),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    let mut cfg_other = HnswConfig::new(16, 100);
    cfg_other.seed = 99;
    let idx_c = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        cfg_other,
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();

    assert_eq!(idx_a.graph_fingerprint(), idx_b.graph_fingerprint());
    assert_ne!(idx_a.graph_fingerprint(), idx_c.graph_fingerprint());

    let mut sa = idx_a.new_search_scratch(50);
    let mut sb = idx_b.new_search_scratch(50);
    let ra = idx_a
        .search_batch_with_scratch(&queries, 10, 50, None, &mut sa)
        .unwrap();
    let rb = idx_b
        .search_batch_with_scratch(&queries, 10, 50, None, &mut sb)
        .unwrap();
    assert_eq!(ra, rb);
}

#[test]
fn phase6_sq8_siftsmall_smoke_if_present() {
    let Some(loaded) = common::load_siftsmall_or_skip("phase6_sq8_siftsmall_smoke_if_present")
    else {
        return;
    };

    let k = 10_usize;
    let ef = 100_usize;
    let cfg = HnswConfig::new(16, 100);

    let exact = ExactKnn::from_base(Metric::L2Squared, loaded.base.clone()).unwrap();
    let gt = exact.search(&loaded.query, k).unwrap();

    let mut f32_idx = HnswIndex::build(&loaded.base, Metric::L2Squared, cfg.clone()).unwrap();
    let f32_res = f32_idx.search_batch(&loaded.query, k, ef).unwrap();
    let f32_recall = compute_recall_at_k(&f32_res.indices, &gt.indices, k).unwrap();

    let sq8_only = HnswSq8Index::build(
        &loaded.base,
        Metric::L2Squared,
        cfg.clone(),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();
    assert_eq!(sq8_only.graph_fingerprint(), f32_idx.graph_fingerprint());

    let sq8_direct = HnswSq8Index::build(
        &loaded.base,
        Metric::L2Squared,
        cfg,
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .unwrap();

    let sq8_rerank = sq8_only
        .clone()
        .with_raw_base(Arc::new(loaded.base.clone()))
        .unwrap();

    let mut scratch = sq8_only.new_search_scratch(ef);
    let r_only = results_to_knn_search_result(
        &sq8_only
            .search_batch_with_scratch(&loaded.query, k, ef, None, &mut scratch)
            .unwrap(),
    )
    .unwrap();
    let r_direct = results_to_knn_search_result(
        &sq8_direct
            .search_batch_with_scratch(&loaded.query, k, ef, None, &mut scratch)
            .unwrap(),
    )
    .unwrap();
    let r_rerank_k = results_to_knn_search_result(
        &sq8_rerank
            .search_batch_with_scratch(&loaded.query, k, ef, Some(k), &mut scratch)
            .unwrap(),
    )
    .unwrap();
    let r_rerank_50 = results_to_knn_search_result(
        &sq8_rerank
            .search_batch_with_scratch(&loaded.query, k, ef, Some(50), &mut scratch)
            .unwrap(),
    )
    .unwrap();

    let rec_only = compute_recall_at_k(&r_only.indices, &gt.indices, k).unwrap();
    let rec_direct = compute_recall_at_k(&r_direct.indices, &gt.indices, k).unwrap();
    let rec_rerank_k = compute_recall_at_k(&r_rerank_k.indices, &gt.indices, k).unwrap();
    let rec_rerank_50 = compute_recall_at_k(&r_rerank_50.indices, &gt.indices, k).unwrap();

    eprintln!(
        "[siftsmall ef={ef}] f32={f32_recall:.4}, sq8_only={rec_only:.4}, sq8_direct={rec_direct:.4}, sq8_r10={rec_rerank_k:.4}, sq8_r50={rec_rerank_50:.4}, sq8_only_B/vec={:.2}",
        sq8_only.memory_profile().steady_state_bytes_per_vector
    );

    assert!(
        rec_only >= 0.90,
        "siftsmall sq8_only recall too low: {rec_only}"
    );
    assert!(
        rec_direct >= 0.90,
        "siftsmall sq8_direct recall too low: {rec_direct}"
    );
    assert!((rec_rerank_k - rec_only).abs() < 1e-12);
    assert!(
        rec_rerank_50 >= rec_only,
        "R=50 rerank ({rec_rerank_50}) should not degrade below Sq8Only ({rec_only})"
    );
    assert!(
        rec_rerank_50 >= 0.96,
        "siftsmall sq8_rerank_50 recall too low: {rec_rerank_50}"
    );
    assert!(sq8_only.memory_profile().steady_state_bytes_per_vector < 300.0);
}
