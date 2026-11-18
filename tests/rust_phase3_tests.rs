//! Phase 3 regression tests. They run on small synthetic data so they need no SIFT download.

mod common;

use common::TempTestDir;
use vector_search::benchmark::{
    detect_provenance, detect_provenance_with_commands, run_phase1_benchmark, run_phase2_benchmark,
    run_phase3_benchmark, summarize_repeated_latencies, BaselineMultiScaleReport, Phase1Config,
    Phase1DatasetMode, Phase2Config, Phase3Config,
};
use vector_search::dataset::{
    generate_synthetic_sift, read_fvecs_max_rows, read_ivecs_max_rows, write_fvecs, write_ivecs,
};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::ExactKnn;
use vector_search::hnsw::{HnswConfig, HnswIndex, VisitedTracker};
use vector_search::matrix::RowMatrix;
use vector_search::recall::LatencyStats;

fn small_phase2_config(run_repro_check: bool) -> Phase2Config {
    Phase2Config {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 400,
            num_query: 20,
            dim: 16,
            num_clusters: 8,
        },
        k: 10,
        ef_search_list: vec![10, 50],
        allow_download: false,
        save_files: false,
        run_repro_check,
        ..Default::default()
    }
}

fn small_config(run_repro_check: bool) -> Phase3Config {
    Phase3Config {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 600,
            num_query: 20,
            dim: 16,
            num_clusters: 8,
        },
        k: 10,
        ef_search_list: vec![10, 50],
        allow_download: false,
        save_files: false,
        run_repro_check,
        repro_vectors: 300,
        sweep_runs: 2,
        run_multi_scale: false,
        ..Default::default()
    }
}

#[test]
fn phase3_skipped_repro_check_is_not_reported_as_verified() {
    let report2 = run_phase2_benchmark(&small_phase2_config(false)).expect("phase 2 must run");
    assert!(!report2.reproducibility_checked);
    assert!(!report2.reproducibility_verified);
    let json2 = report2.to_json_string();
    assert!(json2.contains("\"reproducibility_checked\": false"));
    assert!(json2.contains("\"reproducibility_verified\": false"));
    assert!(json2.contains("\"overall_phase2_passed\": false"));

    let report = run_phase3_benchmark(&small_config(false)).expect("phase 3 must run");
    assert!(!report.reproducibility_checked);
    assert!(!report.reproducibility_verified);
    let json = report.to_json_string();
    assert!(json.contains("\"reproducibility_checked\": false"));
    assert!(json.contains("\"reproducibility_verified\": false"));
    assert!(json.contains("\"phase3_completed\": false"));
}

#[test]
fn phase3_repro_check_passes_and_completes() {
    let report2 = run_phase2_benchmark(&small_phase2_config(true)).expect("phase 2 must run");
    assert!(report2.reproducibility_checked);
    assert!(report2.reproducibility_verified);
    assert!(report2.monotonic_scaling_verified);
    let json2 = report2.to_json_string();
    assert!(json2.contains("\"reproducibility_checked\": true"));
    assert!(json2.contains("\"reproducibility_verified\": true"));

    let report = run_phase3_benchmark(&small_config(true)).expect("phase 3 must run");
    assert!(report.reproducibility_checked);
    assert!(report.reproducibility_verified);
    assert!(report.monotonic_scaling_verified);
    assert!(report
        .to_json_string()
        .contains("\"phase3_completed\": true"));
}

#[test]
fn phase3_baseline_rows_not_from_this_run_are_flagged() {
    // A 600-vector run cannot have measured the 10k / 100k / 1M baseline rows, so none of
    // them may claim to have been measured in this run.
    let report = run_phase3_benchmark(&small_config(false)).unwrap();
    assert!(!report.multi_scale_baseline_summary.is_empty());
    assert!(report
        .multi_scale_baseline_summary
        .iter()
        .all(|p| !p.measured_this_run));
    assert!(!report
        .to_json_string()
        .contains("\"measured_this_run\": true"));
}

#[test]
fn parallel_search_rejects_non_finite_query_instead_of_panicking() {
    let (base, _) = generate_synthetic_sift(300, 4, 8, 4, 7).unwrap();
    let index = HnswIndex::build(&base, Metric::L2Squared, HnswConfig::default()).unwrap();

    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = vec![vec![0.5_f32; 8]; 8];
        bad[5][3] = non_finite;
        let queries = RowMatrix::from_rows(&bad).unwrap();

        for threads in [1, 4] {
            let err = index
                .search_batch_parallel(&queries, 5, 20, threads)
                .expect_err("non-finite query must be rejected");
            assert!(matches!(err, VectorSearchError::InvalidArgument(_)));
        }
    }

    // Even on an empty index (0 inserted vectors), non-finite queries and dimension
    // mismatches must be rejected consistently with the sequential path.
    let mut empty_index = HnswIndex::new(&base, Metric::L2Squared, HnswConfig::default());
    let nan_queries = RowMatrix::from_rows(&[vec![f32::NAN; 8]]).unwrap();
    assert!(matches!(
        empty_index.search_batch(&nan_queries, 5, 20),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        empty_index.search_batch_parallel(&nan_queries, 5, 20, 4),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    let wrong_dim_queries = RowMatrix::from_rows(&[vec![1.0_f32; 4]]).unwrap();
    assert!(matches!(
        empty_index.search_batch(&wrong_dim_queries, 5, 20),
        Err(VectorSearchError::DimensionMismatch {
            expected: 8,
            actual: 4
        })
    ));
    assert!(matches!(
        empty_index.search_batch_parallel(&wrong_dim_queries, 5, 20, 4),
        Err(VectorSearchError::DimensionMismatch {
            expected: 8,
            actual: 4
        })
    ));
}

#[test]
fn phase3_official_ground_truth_check_detects_corrupted_gt_and_truncated_base() {
    let tmp = TempTestDir::new("phase3_gt");
    let tmp_dir = tmp.path();

    let (base, query) = generate_synthetic_sift(200, 20, 16, 4, 123).unwrap();
    let true_gt = ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, 10, 2)
        .unwrap()
        .indices;

    let base_path = tmp_dir.join("base.fvecs");
    let query_path = tmp_dir.join("query.fvecs");
    let true_gt_path = tmp_dir.join("true_gt.ivecs");
    let bad_gt_path = tmp_dir.join("bad_gt.ivecs");

    write_fvecs(&base_path, &base).unwrap();
    write_fvecs(&query_path, &query).unwrap();
    write_ivecs(&true_gt_path, &true_gt).unwrap();

    // Create a corrupted ground-truth file where each row still has 10 distinct IDs (which
    // would fool a self-recall check `verify_self_recall(gt, 10)`), but shifted by +50 mod 200
    // so they are not the true nearest neighbors.
    let mut bad_rows = Vec::with_capacity(true_gt.rows());
    for q in 0..true_gt.rows() {
        let shifted: Vec<i32> = true_gt.row(q).iter().map(|&id| (id + 50) % 200).collect();
        bad_rows.push(shifted);
    }
    let bad_gt = RowMatrix::from_rows(&bad_rows).unwrap();
    write_ivecs(&bad_gt_path, &bad_gt).unwrap();

    // 1. Authentic ground truth on full base -> official_ground_truth_matched must be true
    let cfg_good = Phase3Config {
        dataset_mode: Phase1DatasetMode::Custom {
            base_fvecs: base_path.clone(),
            query_fvecs: query_path.clone(),
            gt_ivecs: Some(true_gt_path.clone()),
        },
        k: 10,
        ef_search_list: vec![10, 50],
        allow_download: false,
        save_files: false,
        run_repro_check: false,
        sweep_runs: 1,
        run_multi_scale: false,
        ..Default::default()
    };
    let rep_good = run_phase3_benchmark(&cfg_good).unwrap();
    assert!(rep_good.official_ground_truth_matched);
    assert_eq!(rep_good.official_gt_check_queries, 20);
    assert!((rep_good.official_gt_recall_at_k.unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(rep_good.official_gt_exact_id_matches, Some(200));
    assert!(rep_good
        .to_json_string()
        .contains("\"official_ground_truth_matched\": true"));

    // 2. Corrupted ground truth (still distinct per row) -> must be rejected as false
    let cfg_bad = Phase3Config {
        dataset_mode: Phase1DatasetMode::Custom {
            base_fvecs: base_path.clone(),
            query_fvecs: query_path.clone(),
            gt_ivecs: Some(bad_gt_path),
        },
        ..cfg_good.clone()
    };
    let rep_bad = run_phase3_benchmark(&cfg_bad).unwrap();
    assert!(!rep_bad.official_ground_truth_matched);
    assert!(rep_bad.official_gt_recall_at_k.unwrap() < 0.5);
    assert!(rep_bad
        .to_json_string()
        .contains("\"official_ground_truth_matched\": false"));

    // 3. Single non-optimal neighbor (199/200 match => 0.995 recall, but 1 neighbor has distance > kth exact)
    // -> must be rejected as false
    let mut one_bad_rows: Vec<Vec<i32>> = (0..true_gt.rows())
        .map(|q| true_gt.row(q).to_vec())
        .collect();
    let full_nn = ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, 200, 2)
        .unwrap()
        .indices;
    // Replace the 10th neighbor of query 0 with the farthest vector (rank 199)
    one_bad_rows[0][9] = full_nn.row(0)[199];
    let one_bad_gt = RowMatrix::from_rows(&one_bad_rows).unwrap();
    let one_bad_gt_path = tmp_dir.join("one_bad_gt.ivecs");
    write_ivecs(&one_bad_gt_path, &one_bad_gt).unwrap();
    let cfg_one_bad = Phase3Config {
        dataset_mode: Phase1DatasetMode::Custom {
            base_fvecs: base_path.clone(),
            query_fvecs: query_path.clone(),
            gt_ivecs: Some(one_bad_gt_path),
        },
        ..cfg_good.clone()
    };
    let rep_one_bad = run_phase3_benchmark(&cfg_one_bad).unwrap();
    assert!((rep_one_bad.official_gt_recall_at_k.unwrap() - 0.995).abs() < 1e-6);
    assert!(!rep_one_bad.official_ground_truth_matched);

    // 4. Genuine k-th boundary distance tie (199/200 ID match => 0.995 recall, and the 1 differing ID
    // has the exact same distance to query 0 as the 10th nearest neighbor) -> must be accepted as true
    let mut tied_base_rows: Vec<Vec<f32>> =
        (0..base.rows()).map(|r| base.row(r).to_vec()).collect();
    let tenth_id = true_gt.row(0)[9] as usize;
    let farthest_id = full_nn.row(0)[199] as usize;
    tied_base_rows[farthest_id] = tied_base_rows[tenth_id].clone();
    let tied_base = RowMatrix::from_rows(&tied_base_rows).unwrap();
    let tied_base_path = tmp_dir.join("tied_base.fvecs");
    write_fvecs(&tied_base_path, &tied_base).unwrap();

    let tied_exact =
        ExactKnn::search_parallel_with_base(Metric::L2Squared, &tied_base, &query, 10, 2)
            .unwrap()
            .indices;
    let mut tied_gt_rows: Vec<Vec<i32>> = (0..tied_exact.rows())
        .map(|q| tied_exact.row(q).to_vec())
        .collect();
    let other_tied_id = if tied_gt_rows[0][9] == tenth_id as i32 {
        farthest_id as i32
    } else {
        tenth_id as i32
    };
    tied_gt_rows[0][9] = other_tied_id;
    let tied_gt = RowMatrix::from_rows(&tied_gt_rows).unwrap();
    let tied_gt_path = tmp_dir.join("tied_gt.ivecs");
    write_ivecs(&tied_gt_path, &tied_gt).unwrap();

    let cfg_tied = Phase3Config {
        dataset_mode: Phase1DatasetMode::Custom {
            base_fvecs: tied_base_path,
            query_fvecs: query_path.clone(),
            gt_ivecs: Some(tied_gt_path),
        },
        ..cfg_good.clone()
    };
    let rep_tied = run_phase3_benchmark(&cfg_tied).unwrap();
    assert!((rep_tied.official_gt_recall_at_k.unwrap() - 0.995).abs() < 1e-6);
    assert!(rep_tied.official_ground_truth_matched);

    // 5. Truncated base (--max-base 100 on 200-vector file) -> full-corpus GT does not apply
    let cfg_trunc = Phase3Config {
        max_base: Some(100),
        ..cfg_good
    };
    let rep_trunc = run_phase3_benchmark(&cfg_trunc).unwrap();
    assert_eq!(rep_trunc.num_base, 100);
    assert!(!rep_trunc.official_ground_truth_matched);
    assert_eq!(rep_trunc.official_gt_check_queries, 0);
}

#[test]
fn phase3_repeated_ef_search_sweep_reports_median_and_spread() {
    // 1. Direct unit test on summarize_repeated_latencies with 5 synthetic runs
    let synthetic_runs: Vec<LatencyStats> = [0.10, 0.50, 0.20, 0.30, 0.40]
        .into_iter()
        .map(|m| LatencyStats {
            num_queries: 100,
            total_s: m * 0.1,
            qps: 1000.0 / m,
            min_ms: m * 0.5,
            mean_ms: m,
            p50_ms: m * 0.9,
            p95_ms: m * 1.5,
            p99_ms: m * 2.0,
            max_ms: m * 3.0,
        })
        .collect();

    let summary = summarize_repeated_latencies(&synthetic_runs);
    assert_eq!(summary.num_runs, 5);
    assert!((summary.median_latency.mean_ms - 0.30).abs() < 1e-9);
    assert!((summary.mean_ms_min - 0.10).abs() < 1e-9);
    assert!((summary.mean_ms_max - 0.50).abs() < 1e-9);
    assert!((summary.mean_ms_spread - 0.40).abs() < 1e-9);
    assert!((summary.median_latency.p99_ms - 0.60).abs() < 1e-9);
    assert!((summary.p99_ms_min - 0.20).abs() < 1e-9);
    assert!((summary.p99_ms_max - 1.00).abs() < 1e-9);
    assert!((summary.p99_ms_spread - 0.80).abs() < 1e-9);

    // 2. End-to-end Phase 3 benchmark with sweep_runs = 3
    let mut cfg = small_config(false);
    cfg.sweep_runs = 3;
    let report = run_phase3_benchmark(&cfg).unwrap();
    assert_eq!(report.sweep_runs, 3);
    for eval in &report.ef_evaluations {
        assert_eq!(eval.num_runs, 3);
        assert_eq!(eval.run_mean_latencies_ms.len(), 3);
        assert_eq!(eval.run_p99_latencies_ms.len(), 3);
        assert!(eval.mean_ms_min <= eval.latency.mean_ms + 1e-9);
        assert!(eval.latency.mean_ms <= eval.mean_ms_max + 1e-9);
        assert!((eval.mean_ms_max - eval.mean_ms_min - eval.mean_ms_spread).abs() < 1e-9);
        assert!(eval.p99_ms_min <= eval.latency.p99_ms + 1e-9);
        assert!(eval.latency.p99_ms <= eval.p99_ms_max + 1e-9);
        assert!((eval.p99_ms_max - eval.p99_ms_min - eval.p99_ms_spread).abs() < 1e-9);
    }

    let json = report.to_json_string();
    assert!(json.contains("\"num_runs\": 3"));
    assert!(json.contains("\"mean_ms_spread\":"));
    assert!(json.contains("\"p99_ms_spread\":"));
    assert!(json.contains("\"run_mean_latencies_ms\":"));
    assert!(json.contains("\"run_p99_latencies_ms\":"));
}

#[test]
fn phase3_prefix_fvecs_loading_graph_only_bytes_and_config_validation() {
    // 1. HnswConfig::new must not silently clamp invalid parameters; validate() must reject them
    let bad_m = HnswConfig::new(0, 100);
    assert_eq!(bad_m.m, 0);
    assert!(bad_m.validate().is_err());

    let bad_ef = HnswConfig::new(16, 8);
    assert_eq!(bad_ef.ef_construction, 8);
    assert!(bad_ef.validate().is_err());

    let good_cfg = HnswConfig::new(16, 100);
    assert!(good_cfg.validate().is_ok());

    // 2. Prefix .fvecs / .ivecs loading reads only max_rows while validating full file alignment
    let tmp = TempTestDir::new("phase3_prefix");
    let tmp_dir = tmp.path();

    let (base, _) = generate_synthetic_sift(80, 5, 12, 4, 77).unwrap();
    let fvecs_path = tmp_dir.join("test.fvecs");
    write_fvecs(&fvecs_path, &base).unwrap();

    let prefix_25 = read_fvecs_max_rows(&fvecs_path, Some(25)).unwrap();
    assert_eq!(prefix_25.rows(), 25);
    assert_eq!(prefix_25.cols(), 12);
    assert_eq!(prefix_25.as_slice(), base.slice_first_rows(25).as_slice());

    let gt = ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &prefix_25, 5, 2)
        .unwrap()
        .indices;
    let ivecs_path = tmp_dir.join("test.ivecs");
    write_ivecs(&ivecs_path, &gt).unwrap();
    let gt_prefix_10 = read_ivecs_max_rows(&ivecs_path, Some(10)).unwrap();
    assert_eq!(gt_prefix_10.rows(), 10);
    assert_eq!(gt_prefix_10.cols(), 5);
    assert_eq!(gt_prefix_10.as_slice(), gt.slice_first_rows(10).as_slice());

    // 3. Graph-only memory accounting and search_batch_timed_with_tracker(&self, ...)
    let index = HnswIndex::build(&base, Metric::L2Squared, good_cfg).unwrap();
    let gc = index.graph_components();
    assert_eq!(index.graph_only_memory_bytes(), gc.graph_only_bytes());
    assert_eq!(
        index.memory_bytes(),
        index.graph_only_memory_bytes() + gc.visited_tracker_tags_bytes
    );
    assert_eq!(
        gc.total_graph_bytes(),
        index.memory_bytes() + gc.heuristic_scratch_bytes
    );
    assert!((index.bytes_per_vector() - index.graph_only_bytes_per_vector() - 4.0).abs() < 1e-9);

    let mut tracker = VisitedTracker::new(index.num_vectors());
    let timed = index
        .search_batch_timed_with_tracker(&prefix_25, 5, 20, &mut tracker)
        .unwrap();
    assert_eq!(timed.result.indices.rows(), 25);
    assert_eq!(timed.result.indices.cols(), 5);
}

#[test]
fn test_graph_fingerprint_properties() {
    let (base, _) = generate_synthetic_sift(100, 5, 16, 4, 42).unwrap();
    let cfg1 = HnswConfig {
        seed: 42,
        ..Default::default()
    };
    let index1 = HnswIndex::build(&base, Metric::L2Squared, cfg1.clone()).unwrap();
    let fp1 = index1.graph_fingerprint();

    // 1. Deterministic: same seed produces identical fingerprint
    let cfg2 = HnswConfig {
        seed: 42,
        ..Default::default()
    };
    let index2 = HnswIndex::build(&base, Metric::L2Squared, cfg2).unwrap();
    let fp2 = index2.graph_fingerprint();
    assert_eq!(fp1, fp2);
    assert_ne!(fp1, 0);

    // 2. Seed-sensitive: different seed produces different fingerprint
    let cfg3 = HnswConfig {
        seed: 43,
        ..Default::default()
    };
    let index3 = HnswIndex::build(&base, Metric::L2Squared, cfg3).unwrap();
    let fp3 = index3.graph_fingerprint();
    assert_ne!(fp1, fp3);

    // 3. Non-empty vs empty index behavior
    let empty_base = RowMatrix::<f32>::new(0, 16, vec![]).unwrap();
    let empty_index = HnswIndex::new(&empty_base, Metric::L2Squared, cfg1);
    let fp_empty = empty_index.graph_fingerprint();
    assert_ne!(fp_empty, fp1);
    assert_ne!(fp_empty, 0);
}

#[test]
fn test_hnsw_build_instrumentation_zero_cost_and_metrics() {
    let (base, _) = generate_synthetic_sift(100, 5, 16, 4, 42).unwrap();
    let cfg = HnswConfig {
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();

    #[cfg(feature = "instrument")]
    {
        let stats = index
            .build_stats()
            .expect("build_stats must be Some when instrument enabled");
        assert_eq!(stats.inserts_count, 100);
        assert!(stats.distance_evals_total > 0);
        assert_eq!(
            stats.distance_evals_total,
            stats.distance_evals_search
                + stats.distance_evals_heuristic
                + stats.distance_evals_shrink
        );
        assert!(stats.search_layer_calls > 0);
        assert!(stats.heuristic_calls > 0);
        assert!(stats.dist_evals_per_insert_mean > 0.0);
        assert!(stats.dist_evals_per_insert_max >= stats.dist_evals_per_insert_min);
        assert!(stats.total_build_time_s >= 0.0);
    }

    #[cfg(not(feature = "instrument"))]
    {
        assert!(index.build_stats().is_none());
    }
}

#[test]
fn test_keep_pruned_shrink_config_override() {
    let (base, queries) = generate_synthetic_sift(200, 10, 16, 4, 42).unwrap();
    let cfg_default = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 40,
        seed: 42,
        ..Default::default()
    };
    assert!(cfg_default.effective_keep_pruned_shrink());

    let cfg_explicit_true = HnswConfig {
        keep_pruned_shrink: Some(true),
        ..cfg_default.clone()
    };
    assert!(cfg_explicit_true.effective_keep_pruned_shrink());

    let cfg_shrink_false = HnswConfig {
        keep_pruned_shrink: Some(false),
        ..cfg_default.clone()
    };
    assert!(!cfg_shrink_false.effective_keep_pruned_shrink());

    let idx_default = HnswIndex::build(&base, Metric::L2Squared, cfg_default).unwrap();
    let idx_explicit_true = HnswIndex::build(&base, Metric::L2Squared, cfg_explicit_true).unwrap();
    let mut idx_shrink_false =
        HnswIndex::build(&base, Metric::L2Squared, cfg_shrink_false).unwrap();

    // Explicit Some(true) must match default (None) bit-for-bit
    assert_eq!(
        idx_default.graph_fingerprint(),
        idx_explicit_true.graph_fingerprint()
    );
    // keep_pruned_shrink = Some(false) alters shrink pruning when degree overflows m_layer
    assert_ne!(
        idx_default.graph_fingerprint(),
        idx_shrink_false.graph_fingerprint()
    );

    let res = idx_shrink_false.search_batch(&queries, 5, 40).unwrap();
    assert_eq!(res.indices.rows(), 10);
    assert_eq!(res.indices.cols(), 5);
}

#[test]
fn test_report_provenance_fields_and_unknown_fallback() {
    // 1. Live provenance detection succeeds in this repository
    let live_prov = detect_provenance();
    assert!(!live_prov.git_commit.is_empty());
    assert!(!live_prov.rustc_version.is_empty());
    assert!(!live_prov.power_mode.is_empty());
    assert!(!live_prov.processor_affinity_mask.is_empty());

    // 2. Fallback path when git / rustc binaries are unavailable returns "unknown" and dirty=false
    let fallback_prov = detect_provenance_with_commands(
        "__nonexistent_git_binary_for_t04_test__",
        "__nonexistent_rustc_binary_for_t04_test__",
    );
    assert_eq!(fallback_prov.git_commit, "unknown");
    assert!(!fallback_prov.git_dirty);
    assert_eq!(fallback_prov.rustc_version, "unknown");

    // 3. All report writers (Phase 1, Phase 2, BaselineMultiScaleReport, Phase 3) emit
    //    git_commit, git_dirty, rustc_version, and environment in JSON
    let cfg1 = Phase1Config {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 100,
            num_query: 10,
            dim: 8,
            num_clusters: 4,
        },
        k: 5,
        allow_download: false,
        save_files: false,
        ..Default::default()
    };
    let rep1 = run_phase1_benchmark(&cfg1).expect("phase 1 must run");
    let json1 = rep1.to_json_string();
    let rep2 = run_phase2_benchmark(&small_phase2_config(false)).expect("phase 2 must run");
    let json2 = rep2.to_json_string();
    let rep3 = run_phase3_benchmark(&small_config(false)).expect("phase 3 must run");
    let json3 = rep3.to_json_string();

    let multi_rep = BaselineMultiScaleReport {
        hardware: rep3.hardware.clone(),
        dataset_source: rep3.dataset_source.clone(),
        hnsw_m: rep3.hnsw_m,
        hnsw_m_max0: rep3.hnsw_m_max0,
        hnsw_ef_construction: rep3.hnsw_ef_construction,
        seed: rep3.seed,
        metric: rep3.metric,
        peak_rss_measurement_method: "test".to_string(),
        scale_profiles: rep3.multi_scale_baseline_summary.clone(),
    };
    let json_multi = multi_rep.to_json_string();

    for (name, json) in [
        ("Phase1Report", &json1),
        ("Phase2Report", &json2),
        ("BaselineMultiScaleReport", &json_multi),
        ("Phase3Report", &json3),
    ] {
        assert!(
            json.contains("\"git_commit\":"),
            "{name} missing git_commit"
        );
        assert!(json.contains("\"git_dirty\":"), "{name} missing git_dirty");
        assert!(
            json.contains("\"rustc_version\":"),
            "{name} missing rustc_version"
        );
        assert!(
            json.contains("\"environment\": {"),
            "{name} missing environment block"
        );
        assert!(
            json.contains("\"power_mode\":"),
            "{name} missing power_mode"
        );
        assert!(
            json.contains("\"processor_affinity_mask\":"),
            "{name} missing processor_affinity_mask"
        );
    }
}
