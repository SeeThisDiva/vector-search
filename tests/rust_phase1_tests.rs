//! Comprehensive integration & edge-case test suite for the Rust Phase 1 implementation.

use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use vector_search::benchmark::{run_phase1_benchmark, Phase1Config, Phase1DatasetMode};
use vector_search::dataset::{
    generate_synthetic_sift, load_custom_dataset, load_or_download_sift, read_fvecs, read_ivecs,
    write_fvecs, write_ivecs, DatasetSource, SiftPreset,
};
use vector_search::distance::{euclidean, inner_product, l2_squared, Metric};
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::ExactKnn;
use vector_search::hardware::detect_hardware;
use vector_search::matrix::RowMatrix;
use vector_search::recall::{
    compute_latency_stats, compute_recall_at_k, evaluate_recall_summary, percentile_sorted,
    verify_self_recall,
};
use vector_search::rng::DeterministicRng;

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("vs_rust_test_{prefix}_{pid}_{nanos}"));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self { path: dir }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

// ============================================================================
// 1. Deterministic PRNG & Synthetic SIFT Dataset Tests
// ============================================================================

#[test]
fn test_rng_open_unit_interval_and_bounds() {
    let mut rng = DeterministicRng::new(42);
    for _ in 0..10_000 {
        let u = rng.next_f64_open01();
        assert!(u > 0.0 && u < 1.0, "u out of open (0, 1): {u}");
        let idx7 = rng.next_usize(7);
        assert!(idx7 < 7);
        let idx8 = rng.next_usize(8);
        assert!(idx8 < 8);
        let idx1 = rng.next_usize(1);
        assert_eq!(idx1, 0);
        let g = rng.next_normal_f32(0.0, 1.0);
        assert!(g.is_finite());
    }
}

#[test]
fn test_dataset_generation_shapes_and_seed_determinism() {
    let (base1, query1) = generate_synthetic_sift(300, 25, 64, 5, 123).unwrap();
    let (base2, query2) = generate_synthetic_sift(300, 25, 64, 5, 123).unwrap();
    let (base_diff, query_diff) = generate_synthetic_sift(300, 25, 64, 5, 999).unwrap();

    assert_eq!(base1.shape(), (300, 64));
    assert_eq!(query1.shape(), (25, 64));
    assert!(base1.as_slice().iter().all(|v| v.is_finite()));
    assert!(query1.as_slice().iter().all(|v| v.is_finite()));

    // Bit-exact reproducibility for identical seeds
    assert_eq!(base1, base2);
    assert_eq!(query1, query2);

    // Distinct outputs for distinct seeds
    assert_ne!(base1, base_diff);
    assert_ne!(query1, query_diff);
}

#[test]
fn test_dataset_generation_invalid_args() {
    assert!(matches!(
        generate_synthetic_sift(0, 10, 32, 4, 42),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        generate_synthetic_sift(10, 0, 32, 4, 42),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        generate_synthetic_sift(10, 5, 0, 4, 42),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

// ============================================================================
// 2. Texmex Binary I/O (.fvecs, .ivecs) & Corruption Tests
// ============================================================================

#[test]
fn test_fvecs_and_ivecs_bit_exact_roundtrip() {
    let tmp = TempDirGuard::new("roundtrip");
    let (base, _) = generate_synthetic_sift(77, 5, 37, 4, 2026).unwrap();

    let fvecs_path = tmp.path.join("nested").join("vectors.fvecs");
    write_fvecs(&fvecs_path, &base).unwrap();
    let loaded_base = read_fvecs(&fvecs_path).unwrap();
    assert_eq!(loaded_base.shape(), (77, 37));
    // Bit-for-bit equality (including sign bits)
    for (&a, &b) in base.as_slice().iter().zip(loaded_base.as_slice().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let mut ids = Vec::with_capacity(50 * 10);
    for r in 0..50_i32 {
        for c in 0..10_i32 {
            ids.push(r * 100 + c);
        }
    }
    let ivecs_mat = RowMatrix::new(50, 10, ids).unwrap();
    let ivecs_path = tmp.path.join("neighbors.ivecs");
    write_ivecs(&ivecs_path, &ivecs_mat).unwrap();
    let loaded_ids = read_ivecs(&ivecs_path).unwrap();
    assert_eq!(loaded_ids, ivecs_mat);
}

#[test]
fn test_fvecs_and_ivecs_empty_files() {
    let tmp = TempDirGuard::new("empty_io");
    let empty_f = tmp.path.join("empty.fvecs");
    let empty_i = tmp.path.join("empty.ivecs");

    write_fvecs(&empty_f, &RowMatrix::<f32>::empty()).unwrap();
    write_ivecs(&empty_i, &RowMatrix::<i32>::empty()).unwrap();

    assert_eq!(fs::metadata(&empty_f).unwrap().len(), 0);
    assert_eq!(fs::metadata(&empty_i).unwrap().len(), 0);

    let rf = read_fvecs(&empty_f).unwrap();
    let ri = read_ivecs(&empty_i).unwrap();
    assert_eq!(rf.shape(), (0, 0));
    assert_eq!(ri.shape(), (0, 0));
}

#[test]
fn test_io_missing_and_corrupt_files() {
    let tmp = TempDirGuard::new("corrupt_io");

    // Missing file
    assert!(matches!(
        read_fvecs(tmp.path.join("missing.fvecs")),
        Err(VectorSearchError::Io(_))
    ));
    assert!(matches!(
        read_ivecs(tmp.path.join("missing.ivecs")),
        Err(VectorSearchError::Io(_))
    ));

    // < 4 bytes
    let tiny = tmp.path.join("tiny.fvecs");
    fs::write(&tiny, [1_u8, 2, 3]).unwrap();
    assert!(matches!(
        read_fvecs(&tiny),
        Err(VectorSearchError::CorruptData(_))
    ));

    // Zero or negative dimension header
    let neg_dim = tmp.path.join("neg.fvecs");
    fs::write(&neg_dim, (-4_i32).to_le_bytes()).unwrap();
    assert!(matches!(
        read_fvecs(&neg_dim),
        Err(VectorSearchError::CorruptData(_))
    ));

    // Truncated record (dim=2 needs 12 bytes, write 8)
    let trunc = tmp.path.join("trunc.fvecs");
    let mut trunc_bytes = Vec::new();
    trunc_bytes.extend_from_slice(&2_i32.to_le_bytes());
    trunc_bytes.extend_from_slice(&1.0_f32.to_le_bytes());
    fs::write(&trunc, trunc_bytes).unwrap();
    assert!(matches!(
        read_fvecs(&trunc),
        Err(VectorSearchError::CorruptData(_))
    ));

    // Inconsistent dimension header on record 1 (file size IS a multiple of record 0's 12 bytes!)
    let mismatch_hdr = tmp.path.join("mismatch_hdr.ivecs");
    let mut bad_records = Vec::new();
    // Record 0: k=2 -> 12 bytes
    bad_records.extend_from_slice(&2_i32.to_le_bytes());
    bad_records.extend_from_slice(&10_i32.to_le_bytes());
    bad_records.extend_from_slice(&20_i32.to_le_bytes());
    // Record 1: header claims k=99, padded to 12 bytes total so file_size == 24 (multiple of 12)
    bad_records.extend_from_slice(&99_i32.to_le_bytes());
    bad_records.extend_from_slice(&30_i32.to_le_bytes());
    bad_records.extend_from_slice(&40_i32.to_le_bytes());
    fs::write(&mismatch_hdr, bad_records).unwrap();
    assert!(matches!(
        read_ivecs(&mismatch_hdr),
        Err(VectorSearchError::CorruptData(_))
    ));
}

// ============================================================================
// 3. Distance Kernels & ExactKNN Brute-Force Engine Tests
// ============================================================================

#[test]
fn test_distance_kernels_across_chunk_and_remainder_dims() {
    for dim in [1_usize, 3, 7, 8, 9, 15, 16, 63, 128, 131] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * -0.25 + 1.5).collect();

        let expected_l2: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum();
        let expected_ip: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let got_l2 = l2_squared(&a, &b);
        let got_euc = euclidean(&a, &b);
        let got_ip = inner_product(&a, &b);

        assert!(
            (got_l2 - expected_l2).abs() <= 1e-3 * expected_l2.max(1.0),
            "dim={dim}: got_l2={got_l2}, expected={expected_l2}"
        );
        assert!((got_euc - expected_l2.sqrt()).abs() <= 1e-4 * expected_l2.sqrt().max(1.0));
        assert!((got_ip - expected_ip).abs() <= 1e-3 * expected_ip.abs().max(1.0));

        // Exact zero self-distance with positive +0.0 sign bit
        let self_d = l2_squared(&a, &a);
        assert_eq!(self_d.to_bits(), 0.0_f32.to_bits());
    }
}

#[test]
fn test_exact_knn_analytical_distances() {
    let base = RowMatrix::from_rows(&[
        vec![0.0_f32, 0.0],
        vec![3.0, 0.0],
        vec![0.0, 4.0],
        vec![3.0, 4.0],
    ])
    .unwrap();
    let query = RowMatrix::from_rows(&[vec![1.0_f32, 1.0]]).unwrap();

    let knn = ExactKnn::from_base(Metric::L2Squared, base).unwrap();
    let res = knn.search(&query, 4).unwrap();

    assert_eq!(res.indices.row(0), &[0, 1, 2, 3]);
    let dists = res.distances.row(0);
    for (&got, &exp) in dists.iter().zip([2.0_f32, 5.0, 10.0, 13.0].iter()) {
        assert!((got - exp).abs() < 1e-6);
    }
}

#[test]
fn test_exact_knn_deterministic_tie_breaking() {
    // Equidistant points interspersed with farther points
    let base = RowMatrix::from_rows(&[
        vec![5.0_f32, 0.0], // id 0, d^2 = 25
        vec![2.0, 0.0],     // id 1, d^2 = 4
        vec![5.0, 0.0],     // id 2, d^2 = 25
        vec![2.0, 0.0],     // id 3, d^2 = 4
        vec![-2.0, 0.0],    // id 4, d^2 = 4
    ])
    .unwrap();
    let query = RowMatrix::from_rows(&[vec![0.0_f32, 0.0]]).unwrap();

    let knn = ExactKnn::from_base(Metric::L2Squared, base).unwrap();

    let r1 = knn.search(&query, 1).unwrap();
    assert_eq!(r1.indices.row(0), &[1]);
    assert_eq!(r1.distances.row(0), &[4.0]);

    let r2 = knn.search(&query, 2).unwrap();
    assert_eq!(r2.indices.row(0), &[1, 3]);
    assert_eq!(r2.distances.row(0), &[4.0, 4.0]);

    let r3 = knn.search(&query, 3).unwrap();
    assert_eq!(r3.indices.row(0), &[1, 3, 4]);
    assert_eq!(r3.distances.row(0), &[4.0, 4.0, 4.0]);

    // k > num_base clamps to num_base = 5 with full deterministic tie order
    let r_all = knn.search(&query, 20).unwrap();
    assert_eq!(r_all.indices.row(0), &[1, 3, 4, 0, 2]);
    assert_eq!(r_all.distances.row(0), &[4.0, 4.0, 4.0, 25.0, 25.0]);
}

#[test]
fn test_exact_knn_euclidean_and_inner_product_metrics() {
    assert_eq!(Metric::from_str("l2").unwrap(), Metric::L2Squared);
    assert_eq!(Metric::from_str("euclidean").unwrap(), Metric::Euclidean);
    assert_eq!(Metric::from_str("ip").unwrap(), Metric::InnerProduct);
    assert!(Metric::from_str("cosine_invalid").is_err());

    let base = RowMatrix::from_rows(&[
        vec![1.0_f32, 0.0], // dot with (1,1) = 1.0
        vec![0.0, 2.0],     // dot with (1,1) = 2.0
        vec![2.0, 0.0],     // dot with (1,1) = 2.0 (tie with id 1 -> id 1 wins!)
    ])
    .unwrap();

    // Euclidean
    let knn_euc = ExactKnn::from_base(Metric::Euclidean, base.clone()).unwrap();
    let q_zero = RowMatrix::from_rows(&[vec![0.0_f32, 0.0]]).unwrap();
    let res_euc = knn_euc.search(&q_zero, 3).unwrap();
    assert_eq!(res_euc.indices.row(0), &[0, 1, 2]);
    assert_eq!(res_euc.distances.row(0), &[1.0, 2.0, 2.0]);

    // Inner Product (descending score, ascending ID on ties)
    let knn_ip = ExactKnn::from_base(Metric::InnerProduct, base).unwrap();
    let q_ones = RowMatrix::from_rows(&[vec![1.0_f32, 1.0]]).unwrap();
    let res_ip = knn_ip.search(&q_ones, 3).unwrap();
    assert_eq!(res_ip.indices.row(0), &[1, 2, 0]);
    assert_eq!(res_ip.distances.row(0), &[2.0, 2.0, 1.0]);
}

#[test]
fn test_exact_knn_parallel_and_timed_parity() {
    let (base, query) = generate_synthetic_sift(1_200, 35, 64, 8, 777).unwrap();
    let knn = ExactKnn::from_base(Metric::L2Squared, base).unwrap();

    let seq = knn.search(&query, 10).unwrap();
    let par = knn.search_parallel(&query, 10, 4).unwrap();
    let timed = knn.search_timed(&query, 10, true).unwrap();

    assert_eq!(seq.indices, par.indices);
    assert_eq!(seq.distances, par.distances);
    assert_eq!(seq.indices, timed.result.indices);
    assert_eq!(seq.distances, timed.result.distances);

    assert_eq!(timed.per_query_latency_ms.len(), 35);
    assert!(timed.latency.qps > 0.0);
    assert!(timed.latency.min_ms <= timed.latency.p50_ms);
    assert!(timed.latency.p50_ms <= timed.latency.p95_ms);
    assert!(timed.latency.p95_ms <= timed.latency.p99_ms);
    assert!(timed.latency.p99_ms <= timed.latency.max_ms);
}

#[test]
fn test_exact_knn_error_paths_and_empty_queries() {
    // Unfitted
    let unfitted = ExactKnn::new(Metric::L2Squared);
    let q = RowMatrix::from_rows(&[vec![1.0_f32, 2.0]]).unwrap();
    assert!(matches!(
        unfitted.search(&q, 1),
        Err(VectorSearchError::UnfittedIndex(_))
    ));
    assert!(matches!(
        unfitted.search_one(&[1.0, 2.0], 1),
        Err(VectorSearchError::UnfittedIndex(_))
    ));

    // Empty base
    assert!(matches!(
        ExactKnn::from_base(Metric::L2Squared, RowMatrix::empty()),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // NaN in base
    let nan_base = RowMatrix::from_rows(&[vec![1.0_f32, f32::NAN]]).unwrap();
    assert!(matches!(
        ExactKnn::from_base(Metric::L2Squared, nan_base),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    let valid_base = RowMatrix::from_rows(&[vec![1.0_f32, 2.0, 3.0]]).unwrap();
    let knn = ExactKnn::from_base(Metric::L2Squared, valid_base).unwrap();

    // Dimension mismatch
    assert!(matches!(
        knn.search(&q, 1),
        Err(VectorSearchError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));

    // k == 0
    let valid_q = RowMatrix::from_rows(&[vec![1.0_f32, 2.0, 3.0]]).unwrap();
    assert!(matches!(
        knn.search(&valid_q, 0),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Inf in query
    let inf_q = RowMatrix::from_rows(&[vec![1.0_f32, f32::INFINITY, 3.0]]).unwrap();
    assert!(matches!(
        knn.search(&inf_q, 1),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // Empty query batch returns empty (0, 0) matrices cleanly
    let empty_res = knn.search(&RowMatrix::empty(), 5).unwrap();
    assert!(empty_res.indices.is_empty());
    assert!(empty_res.distances.is_empty());
}

// ============================================================================
// 4. Recall@k Harness & Percentile Edge Cases
// ============================================================================

#[test]
fn test_recall_harness_all_cases() {
    let gt = RowMatrix::from_rows(&[vec![1, 2, 3, 4, 5], vec![10, 20, 30, 40, 50]]).unwrap();
    assert_eq!(compute_recall_at_k(&gt, &gt, 5).unwrap(), 1.0);
    assert!(verify_self_recall(&gt, 5).unwrap());

    let gt_1 = RowMatrix::from_rows(&[vec![1, 2, 3, 4, 5]]).unwrap();

    // Disjoint -> 0.0
    let disjoint = RowMatrix::from_rows(&[vec![6, 7, 8, 9, 10]]).unwrap();
    assert_eq!(compute_recall_at_k(&disjoint, &gt_1, 5).unwrap(), 0.0);

    // Partial 3/5 -> 0.6
    let partial = RowMatrix::from_rows(&[vec![1, 2, 3, 99, 100]]).unwrap();
    assert!((compute_recall_at_k(&partial, &gt_1, 5).unwrap() - 0.6).abs() < 1e-12);

    // Duplicate candidates penalized -> 1/5 = 0.2
    let dup_cand = RowMatrix::from_rows(&[vec![1, 1, 1, 1, 1]]).unwrap();
    assert!((compute_recall_at_k(&dup_cand, &gt_1, 5).unwrap() - 0.2).abs() < 1e-12);

    // Incomplete candidate list preserves denominator k=5 -> 2/5 = 0.4
    let short_cand = RowMatrix::from_rows(&[vec![1, 2]]).unwrap();
    assert!((compute_recall_at_k(&short_cand, &gt_1, 5).unwrap() - 0.4).abs() < 1e-12);

    // Invalid ground truth with duplicates fails self-verification
    let invalid_gt = RowMatrix::from_rows(&[vec![1, 1, 2]]).unwrap();
    assert!(matches!(
        verify_self_recall(&invalid_gt, 3),
        Err(VectorSearchError::SelfVerificationFailed { k: 3, .. })
    ));

    // Multi-rank summary
    let gt10 = RowMatrix::from_rows(&[vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).unwrap();
    let cand10 = RowMatrix::from_rows(&[vec![0, 1, 2, 3, 4, 90, 91, 92, 93, 94]]).unwrap();
    let summary = evaluate_recall_summary(&cand10, &gt10, &[1, 5, 10, 50]).unwrap();
    assert_eq!(
        summary,
        vec![
            ("Recall@1".to_string(), 1.0),
            ("Recall@5".to_string(), 1.0),
            ("Recall@10".to_string(), 0.5),
        ]
    );

    // Error checks
    assert!(compute_recall_at_k(&gt, &gt, 0).is_err());
    assert!(compute_recall_at_k(&gt_1, &gt, 5).is_err());
}

#[test]
fn test_latency_percentiles() {
    assert_eq!(percentile_sorted(&[], 50.0), 0.0);
    assert_eq!(percentile_sorted(&[3.5], 99.0), 3.5);

    let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let stats = compute_latency_stats(&vals, 1.0);
    assert_eq!(stats.num_queries, 100);
    assert!((stats.qps - 100.0).abs() < 1e-9);
    assert!((stats.min_ms - 0.0).abs() < 1e-9);
    assert!((stats.max_ms - 99.0).abs() < 1e-9);
    assert!((stats.p50_ms - 49.5).abs() < 1e-9);
    assert!((stats.p95_ms - 94.05).abs() < 1e-9);
    assert!((stats.p99_ms - 98.01).abs() < 1e-9);
}

// ============================================================================
// 5. Offline SIFT Fallback, Custom Files & End-to-End Benchmark Pipeline
// ============================================================================

#[test]
fn test_sift_offline_fallback_and_cached_loading() {
    let tmp = TempDirGuard::new("sift_fallback");

    // 1. Offline mode on empty cache -> deterministic SyntheticFallback (10,000 x 128, 100 x 128)
    let fallback = load_or_download_sift(SiftPreset::SiftSmall, &tmp.path, false, 42).unwrap();
    assert_eq!(fallback.base.shape(), (10_000, 128));
    assert_eq!(fallback.query.shape(), (100, 128));
    assert!(fallback.official_ground_truth.is_none());
    assert!(matches!(
        fallback.source,
        DatasetSource::SyntheticFallback {
            preset: SiftPreset::SiftSmall,
            seed: 42,
            ..
        }
    ));

    // 2. Populate valid siftsmall cache layout and verify offline cache hit
    let siftsmall_dir = tmp.path.join("siftsmall");
    write_fvecs(siftsmall_dir.join("siftsmall_base.fvecs"), &fallback.base).unwrap();
    write_fvecs(siftsmall_dir.join("siftsmall_query.fvecs"), &fallback.query).unwrap();
    let dummy_gt = RowMatrix::new(100, 100, vec![0_i32; 10_000]).unwrap();
    write_ivecs(siftsmall_dir.join("siftsmall_groundtruth.ivecs"), &dummy_gt).unwrap();

    let cached = load_or_download_sift(SiftPreset::SiftSmall, &tmp.path, false, 42).unwrap();
    assert!(cached.source.is_authentic_sift());
    assert!(cached.official_ground_truth.is_some());
}

#[test]
fn test_benchmark_pipeline_and_hardware_detection() {
    let hw = detect_hardware();
    assert!(!hw.cpu_model.is_empty());
    assert!(hw.logical_cores >= 1);
    assert!(!hw.summary_line().is_empty());

    let tmp = TempDirGuard::new("bench_e2e");
    let config = Phase1Config {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 400,
            num_query: 20,
            dim: 32,
            num_clusters: 4,
        },
        data_cache_dir: tmp.path.clone(),
        output_dir: tmp.path.clone(),
        k: 10,
        seed: 42,
        metric: Metric::L2Squared,
        allow_download: false,
        save_files: true,
        max_queries: None,
    };

    let report = run_phase1_benchmark(&config).unwrap();
    assert_eq!(report.num_base, 400);
    assert_eq!(report.num_query, 20);
    assert_eq!(report.dim, 32);
    assert_eq!(report.self_recall_at_k, 1.0);
    assert!(report.self_verified);
    assert!(report.latency.qps > 0.0);

    let json = report.to_json_string();
    assert!(json.contains("\"self_verified\": true"));
    assert!(json.contains("\"Recall@10\": 1.000000"));

    // Verify saved artifacts reload cleanly via load_custom_dataset
    let custom = load_custom_dataset(
        report.artifacts.base_fvecs.as_deref().unwrap(),
        report.artifacts.query_fvecs.as_deref().unwrap(),
        report.artifacts.groundtruth_ivecs.as_deref(),
    )
    .unwrap();
    assert_eq!(custom.base.shape(), (400, 32));
    assert_eq!(custom.query.shape(), (20, 32));
    assert_eq!(custom.official_ground_truth.unwrap().shape(), (20, 10));
}
