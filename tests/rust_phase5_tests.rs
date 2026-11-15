//! Integration & unit tests for Milestone M3 Task T3.8 (`phase5` runner & versioned cache).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use vector_search::benchmark::RepeatedSweepSummary;
use vector_search::dataset::S_CLUST_DEFAULT;
use vector_search::dataset::{generate_synthetic_sift, read_ivecs, write_fvecs};
use vector_search::distance::Metric;
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::matrix::RowMatrix;
use vector_search::phase5::{
    cell_report_filename, compute_crossover_analysis, compute_f32_matrix_hash,
    load_or_compute_filtered_gt_cache, load_or_generate_predicate_cache, option_c_hnsw_query_count,
    option_c_s3_query_count, parse_phase5_cell_json, run_1m_opaque_spotcheck, run_phase5_matrix,
    should_evaluate_s1_uncapped, MetadataMode, Phase5ArmRecord, Phase5CellRecord,
    Phase5MatrixConfig, DEFAULT_ATTR_SEED, DEFAULT_K,
};
use vector_search::recall::LatencyStats;

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vs_phase5_{prefix}_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_option_c_schedules_and_s1_uncapped_scope() {
    assert_eq!(option_c_s3_query_count(10_000, 0.50, 100), (100, false));
    assert_eq!(
        option_c_s3_query_count(100_000, 0.50, 10_000),
        (10_000, false)
    );
    assert_eq!(
        option_c_s3_query_count(100_000, 0.0005, 10_000),
        (10_000, false)
    );
    assert_eq!(
        option_c_s3_query_count(1_000_000, 0.05, 10_000),
        (10_000, false)
    );
    assert_eq!(
        option_c_s3_query_count(1_000_000, 0.075, 10_000),
        (2_000, true)
    );
    assert_eq!(
        option_c_s3_query_count(1_000_000, 0.50, 10_000),
        (2_000, true)
    );

    assert_eq!(option_c_hnsw_query_count(10_000, 0.01, 100), (100, false));
    assert_eq!(
        option_c_hnsw_query_count(100_000, 0.01, 10_000),
        (300, true)
    );
    assert_eq!(
        option_c_hnsw_query_count(100_000, 0.05, 10_000),
        (1_000, true)
    );
    assert_eq!(
        option_c_hnsw_query_count(100_000, 0.10, 10_000),
        (1_000, true)
    );
    assert_eq!(
        option_c_hnsw_query_count(1_000_000, 0.20, 10_000),
        (2_000, true)
    );

    assert!(!should_evaluate_s1_uncapped(0.035));
    assert!(!should_evaluate_s1_uncapped(0.01));
    assert!(should_evaluate_s1_uncapped(0.05));
    assert!(should_evaluate_s1_uncapped(0.50));
}

#[test]
fn test_versioned_cache_hit_and_corruption_invalidation() {
    let tmp = TempTestDir::new("cache");
    let cache_dir = tmp.0.join("phase5_cache");
    let (base, queries) = generate_synthetic_sift(300, 8, 16, 16, 42).unwrap();
    let base_hash_hex = format!("0x{:016x}", compute_f32_matrix_hash(&base));
    let query_hash_hex = format!("0x{:016x}", compute_f32_matrix_hash(&queries));

    for mode in [MetadataMode::Independent, MetadataMode::Correlated] {
        let pred1 = load_or_generate_predicate_cache(
            &cache_dir,
            &base,
            &base_hash_hex,
            mode,
            0.10,
            DEFAULT_ATTR_SEED,
            S_CLUST_DEFAULT,
        )
        .unwrap();
        assert!(!pred1.cache_hit);
        assert!(pred1.cache_path.is_file());
        assert!(pred1.provenance_path.is_file());
        assert_eq!(pred1.actual_popcount, 30);

        let pred2 = load_or_generate_predicate_cache(
            &cache_dir,
            &base,
            &base_hash_hex,
            mode,
            0.10,
            DEFAULT_ATTR_SEED,
            S_CLUST_DEFAULT,
        )
        .unwrap();
        assert!(pred2.cache_hit);
        assert_eq!(pred1.bitset_hash_hex, pred2.bitset_hash_hex);

        let engine = FilteredExactKnn::new(Metric::L2Squared, base.clone()).unwrap();
        let gt1 = load_or_compute_filtered_gt_cache(
            &cache_dir,
            Some(&engine),
            &base,
            &queries,
            &base_hash_hex,
            &query_hash_hex,
            &pred2,
            mode,
            0.10,
            DEFAULT_K,
            DEFAULT_ATTR_SEED,
            S_CLUST_DEFAULT,
            2,
        )
        .unwrap();
        assert!(!gt1.cache_hit);
        assert!(gt1.ivecs_path.is_file());
        assert!(gt1.fvecs_path.is_file());
        assert!(gt1.provenance_path.is_file());

        let gt2 = load_or_compute_filtered_gt_cache(
            &cache_dir,
            None,
            &base,
            &queries,
            &base_hash_hex,
            &query_hash_hex,
            &pred2,
            mode,
            0.10,
            DEFAULT_K,
            DEFAULT_ATTR_SEED,
            S_CLUST_DEFAULT,
            2,
        )
        .unwrap();
        assert!(gt2.cache_hit);
        assert_eq!(gt1.gt_hash_hex, gt2.gt_hash_hex);
    }

    // Mutate GT provenance version -> forces clean GT recomputation
    let pred = load_or_generate_predicate_cache(
        &cache_dir,
        &base,
        &base_hash_hex,
        MetadataMode::Independent,
        0.10,
        DEFAULT_ATTR_SEED,
        S_CLUST_DEFAULT,
    )
    .unwrap();
    let gt = load_or_compute_filtered_gt_cache(
        &cache_dir,
        None,
        &base,
        &queries,
        &base_hash_hex,
        &query_hash_hex,
        &pred,
        MetadataMode::Independent,
        0.10,
        DEFAULT_K,
        DEFAULT_ATTR_SEED,
        S_CLUST_DEFAULT,
        1,
    )
    .unwrap();
    assert!(gt.cache_hit);

    let orig_meta = fs::read_to_string(&gt.provenance_path).unwrap();
    fs::write(
        &gt.provenance_path,
        orig_meta.replace("\"gt_gen_version\": 1", "\"gt_gen_version\": 999"),
    )
    .unwrap();
    let gt_rebuilt = load_or_compute_filtered_gt_cache(
        &cache_dir,
        None,
        &base,
        &queries,
        &base_hash_hex,
        &query_hash_hex,
        &pred,
        MetadataMode::Independent,
        0.10,
        DEFAULT_K,
        DEFAULT_ATTR_SEED,
        S_CLUST_DEFAULT,
        1,
    )
    .unwrap();
    assert!(!gt_rebuilt.cache_hit);
    assert_eq!(gt_rebuilt.gt_hash_hex, gt.gt_hash_hex);

    // Corrupt a cached .fvecs file -> forces clean GT recomputation
    let bad_fvecs = RowMatrix::new(
        queries.rows(),
        DEFAULT_K,
        vec![-1.0_f32; queries.rows() * DEFAULT_K],
    )
    .unwrap();
    write_fvecs(&gt_rebuilt.fvecs_path, &bad_fvecs).unwrap();
    let gt_fixed = load_or_compute_filtered_gt_cache(
        &cache_dir,
        None,
        &base,
        &queries,
        &base_hash_hex,
        &query_hash_hex,
        &pred,
        MetadataMode::Independent,
        0.10,
        DEFAULT_K,
        DEFAULT_ATTR_SEED,
        S_CLUST_DEFAULT,
        1,
    )
    .unwrap();
    assert!(!gt_fixed.cache_hit);
    let _ = read_ivecs(&gt_fixed.ivecs_path).unwrap();
}

fn make_arm(strategy: &str, ef: usize, recall: f64, mean_ms: f64) -> Phase5ArmRecord {
    Phase5ArmRecord {
        strategy: strategy.to_string(),
        strategy_label: format!("{strategy}_ef{ef}"),
        ef_search: ef,
        overfetch_cap: 4096,
        overfetch_factor: None,
        max_bridge_hops: None,
        eval_limit: None,
        num_queries: 1000,
        is_subquery_probe: true,
        reps: 5,
        recall_at_k: recall,
        fewer_than_k_count: 0,
        fewer_than_k_rate: 0.0,
        post_filter_cap_reached_rate: 0.0,
        hop_budget_exhausted_rate: 0.0,
        eval_limit_reached_rate: 0.0,
        used_bruteforce_fallback_rate: 0.0,
        summary: RepeatedSweepSummary {
            num_runs: 5,
            median_latency: LatencyStats {
                num_queries: 1000,
                total_s: mean_ms,
                qps: 1000.0 / mean_ms.max(1e-6),
                min_ms: mean_ms,
                mean_ms,
                p50_ms: mean_ms,
                p95_ms: mean_ms,
                p99_ms: mean_ms * 1.2,
                max_ms: mean_ms * 1.5,
            },
            mean_ms_min: mean_ms,
            mean_ms_max: mean_ms,
            mean_ms_spread: 0.0,
            p99_ms_min: mean_ms * 1.2,
            p99_ms_max: mean_ms * 1.2,
            p99_ms_spread: 0.0,
            run_mean_latencies_ms: vec![mean_ms; 5],
            run_p99_latencies_ms: vec![mean_ms * 1.2; 5],
        },
        dist_evals_mean: 100.0,
        dist_evals_p50: 100.0,
        dist_evals_p99: 150.0,
        valid_dist_evals_mean: 60.0,
        valid_dist_evals_p50: 60.0,
        valid_dist_evals_p99: 90.0,
        bridge_dist_evals_mean: 40.0,
        bridge_dist_evals_p50: 40.0,
        bridge_dist_evals_p99: 60.0,
        upper_layer_dist_evals_mean: 5.0,
        visited_nodes_mean: 95.0,
        visited_matching_nodes_mean: 60.0,
        eta_graph_mean: 60.0 / 95.0,
        eta_graph_pooled: 60.0 / 95.0,
    }
}

#[test]
fn test_crossover_loglog_interpolation_and_edge_outcomes() {
    let make_cell = |mode: MetadataMode, sigma: f64, s3_ms: f64, s2_arms: Vec<Phase5ArmRecord>| {
        let mut arms = vec![make_arm("S3_pre_filter", 100, 1.0, s3_ms)];
        arms.extend(s2_arms);
        Phase5CellRecord {
            report_filename: cell_report_filename(100_000, mode, sigma),
            status: "measured".to_string(),
            scale: 100_000,
            mode,
            sigma,
            k: 10,
            reps: 5,
            expected_popcount: 1000,
            actual_popcount: 1000,
            clusters_c: 64,
            s1_uncapped_evaluated: sigma >= 0.05,
            predicate_cache_hit: false,
            predicate_gen_time_s: 0.0,
            gt_cache_hit: false,
            gt_gen_time_s: 0.0,
            arms,
        }
    };

    // Symmetry check: at sigma=0.01 (s3=0.2, s2=2.0 -> 10x slower) and sigma=0.10 (s3=2.0, s2=0.2 -> 10x faster),
    // exact log-log midpoint is sqrt(0.01 * 0.10) = 0.0316227766...
    let cells = vec![
        make_cell(
            MetadataMode::Independent,
            0.01,
            0.2,
            vec![make_arm("S2_in_traversal", 20, 0.96, 2.0)],
        ),
        make_cell(
            MetadataMode::Independent,
            0.10,
            2.0,
            vec![make_arm("S2_in_traversal", 20, 0.97, 0.2)],
        ),
        make_cell(
            MetadataMode::Correlated,
            0.01,
            0.1,
            vec![make_arm("S2_in_traversal", 600, 0.50, 5.0)],
        ),
        make_cell(
            MetadataMode::Correlated,
            0.10,
            0.9,
            vec![make_arm("S2_in_traversal", 400, 0.92, 4.0)],
        ),
    ];

    let res = compute_crossover_analysis(&cells, 0.95);
    let ind = res
        .iter()
        .find(|r| r.mode == MetadataMode::Independent)
        .unwrap();
    assert_eq!(ind.outcome, "crossover_found");
    assert!((ind.bracket_lo_sigma.unwrap() - 0.01).abs() < 1e-12);
    assert!((ind.bracket_hi_sigma.unwrap() - 0.10).abs() < 1e-12);
    let expected_mid = (0.01_f64 * 0.10_f64).sqrt();
    assert!((ind.crossover_sigma_loglog.unwrap() - expected_mid).abs() < 1e-9);

    let corr = res
        .iter()
        .find(|r| r.mode == MetadataMode::Correlated)
        .unwrap();
    assert_eq!(corr.outcome, "no_crossover_in_range");
    assert_eq!(corr.note, "s3_wins_across_all_measured_selectivities");
}

#[test]
fn test_resumable_synthetic_matrix_and_spotcheck_end_to_end() {
    let tmp = TempTestDir::new("matrix");
    let cfg = Phase5MatrixConfig {
        data_dir: tmp.0.join("data"),
        output_dir: tmp.0.join("out"),
        scale: 300,
        modes: vec![MetadataMode::Independent, MetadataMode::Correlated],
        sigmas: vec![0.01, 0.10],
        k: 10,
        reps: 2,
        synthetic: true,
        synthetic_dim: 8,
        synthetic_queries: Some(15),
        smoke: true,
        resume: true,
        include_sensitivity: true,
        ..Default::default()
    };

    let summary1 = run_phase5_matrix(&cfg).unwrap();
    assert_eq!(summary1.total_cells, 4);
    assert_eq!(summary1.newly_measured_count, 4);
    assert_eq!(summary1.resumed_skipped_count, 0);
    assert!(summary1.sensitivity_executed);
    assert!(summary1.summary_path.is_file());
    assert!(cfg.output_dir.join("sensitivity_n300.json").is_file());

    // Inspect cell schemas & S1_uncap sigma >= 0.05 scoping
    let cell_low_path =
        cfg.output_dir
            .join(cell_report_filename(300, MetadataMode::Independent, 0.01));
    let cell_high_path =
        cfg.output_dir
            .join(cell_report_filename(300, MetadataMode::Independent, 0.10));
    let cell_low =
        parse_phase5_cell_json(&fs::read_to_string(&cell_low_path).unwrap(), "low.json").unwrap();
    let cell_high =
        parse_phase5_cell_json(&fs::read_to_string(&cell_high_path).unwrap(), "high.json").unwrap();
    assert!(!cell_low.s1_uncapped_evaluated);
    assert!(!cell_low
        .arms
        .iter()
        .any(|a| a.strategy == "S1_post_uncapped"));
    assert!(cell_high.s1_uncapped_evaluated);
    assert!(cell_high
        .arms
        .iter()
        .any(|a| a.strategy == "S1_post_uncapped"));

    for arm in &cell_high.arms {
        assert_eq!(arm.num_queries, 15);
        assert!(!arm.is_subquery_probe);
        assert!(
            (arm.dist_evals_mean - (arm.valid_dist_evals_mean + arm.bridge_dist_evals_mean)).abs()
                < 1e-2
        );
        if arm.strategy == "S3_pre_filter" {
            assert!((arm.eta_graph_mean - 1.0).abs() < 1e-6);
            assert!(arm.bridge_dist_evals_mean.abs() < 1e-6);
        }
    }

    // Resume pass with no changes -> skips all 4 cells cleanly
    let mut cfg_resume = cfg.clone();
    cfg_resume.include_sensitivity = false;
    let summary2 = run_phase5_matrix(&cfg_resume).unwrap();
    assert_eq!(summary2.newly_measured_count, 0);
    assert_eq!(summary2.resumed_skipped_count, 4);

    // Mutate git_commit in 1 cell -> rebuilds only that 1 cell using warm disk caches
    let orig = fs::read_to_string(&cell_low_path).unwrap();
    let mutated = orig.replacen("\"git_commit\": \"", "\"git_commit\": \"stale_", 1);
    fs::write(&cell_low_path, mutated).unwrap();

    let summary3 = run_phase5_matrix(&cfg_resume).unwrap();
    assert_eq!(summary3.newly_measured_count, 1);
    assert_eq!(summary3.resumed_skipped_count, 3);

    // Spot check mode on small synthetic fixture
    let spot_cfg = Phase5MatrixConfig {
        data_dir: tmp.0.join("data_spot"),
        output_dir: tmp.0.join("out_spot"),
        scale: 250,
        synthetic: true,
        synthetic_dim: 8,
        synthetic_queries: Some(12),
        opaque_1m_spotcheck_only: true,
        ..Default::default()
    };
    let probes = run_1m_opaque_spotcheck(&spot_cfg).unwrap();
    assert!(!probes.is_empty());
    assert!(spot_cfg
        .output_dir
        .join("opaque_spotcheck_1m.json")
        .is_file());
}
