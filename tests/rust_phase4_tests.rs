//! Phase 4 Integration & Regression Tests.
//!
//! Verifies:
//! 1. Grid specification parsing, Cartesian (M, efConstruction) expansion with efConstruction >= M filtering,
//!    explicit config directives, and invalid syntax/parameter rejection.
//! 2. Prefix ground-truth disk caching (`gt_<scale>_k100.ivecs`), hit/miss detection, and recall reproducibility.
//! 3. Resumable sweep runner behavior: skips matching git_commit, rebuilds mismatched commits.
//! 4. Per-index and summary JSON schema conformance, provenance recording, and hardware metadata.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use vector_search::benchmark::Phase1DatasetMode;
use vector_search::dataset::{DatasetSource, LoadedDataset};
use vector_search::distance::Metric;
use vector_search::matrix::RowMatrix;
use vector_search::phase4::{
    can_resume_existing_report, load_or_compute_phase4_ground_truth, run_phase4_sweep,
    Phase4GridSpec, Phase4IndexConfig, Phase4SweepConfig,
};

#[test]
fn test_phase4_grid_parsing_cartesian_and_validation() {
    let grid_text = r#"
# Phase 4 Grid specification test
scale = 50000
k = 10
sweep_runs = 2
seed = 42
ef_search = 10, 20, 50
m = 8, 16, 48
ef_construction = 40, 100
config = m:32, ef_construction:150, keep_pruned_shrink:false
"#;
    let spec = Phase4GridSpec::parse_str(grid_text).expect("Grid parsing must succeed");
    assert_eq!(spec.scale_vectors, Some(50_000));
    assert_eq!(spec.k, 10);
    assert_eq!(spec.sweep_runs, 2);
    assert_eq!(spec.seed, 42);
    assert_eq!(spec.ef_search_list, vec![10, 20, 50]);

    // Cartesian product of [8, 16, 48] x [40, 100] filtering to ef_construction >= m:
    // (8, 40)   -> 40 >= 8   [keep]
    // (8, 100)  -> 100 >= 8  [keep]
    // (16, 40)  -> 40 >= 16  [keep]
    // (16, 100) -> 100 >= 16 [keep]
    // (48, 40)  -> 40 < 48   [skipped!]
    // (48, 100) -> 100 >= 48 [keep]
    // Total Cartesian: 5 configs.
    // Plus 1 explicit config (32, 150) -> 6 configs total.
    assert_eq!(spec.configs.len(), 6);
    assert_eq!(spec.configs[0].m, 8);
    assert_eq!(spec.configs[0].ef_construction, 40);
    assert_eq!(spec.configs[1].m, 8);
    assert_eq!(spec.configs[1].ef_construction, 100);
    assert_eq!(spec.configs[2].m, 16);
    assert_eq!(spec.configs[2].ef_construction, 40);
    assert_eq!(spec.configs[3].m, 16);
    assert_eq!(spec.configs[3].ef_construction, 100);
    assert_eq!(spec.configs[4].m, 48);
    assert_eq!(spec.configs[4].ef_construction, 100);
    assert_eq!(spec.configs[5].m, 32);
    assert_eq!(spec.configs[5].ef_construction, 150);
    assert_eq!(spec.configs[5].keep_pruned_shrink, Some(false));

    // Validation & Error path tests
    // 1. Invalid syntax
    assert!(Phase4GridSpec::parse_str("not_a_valid_directive").is_err());
    // 2. Empty configs
    assert!(Phase4GridSpec::parse_str("k = 10\nseed = 42").is_err());
    // 3. ef_construction < m in explicit config
    assert!(Phase4GridSpec::parse_str("config = m:48, ef_construction:20").is_err());
    // 4. m < 2
    assert!(Phase4GridSpec::parse_str("config = m:1, ef_construction:20").is_err());
    // 5. k == 0
    assert!(Phase4GridSpec::parse_str("k = 0\nconfig = m:8, ef_construction:20").is_err());
    // 6. sweep_runs == 0
    assert!(Phase4GridSpec::parse_str("sweep_runs = 0\nconfig = m:8, ef_construction:20").is_err());
    // 7. ef_search containing 0
    assert!(
        Phase4GridSpec::parse_str("ef_search = 10, 0, 50\nconfig = m:8, ef_construction:20")
            .is_err()
    );

    // 8. Seed ordering inheritance: config preceding seed inherits grid seed unless explicitly specified
    let seed_order_text = r#"
config = m:16, ef_construction:100
config = m:8, ef_construction:40, seed:77
seed = 99
"#;
    let spec_seed =
        Phase4GridSpec::parse_str(seed_order_text).expect("Seed ordering grid must parse");
    assert_eq!(spec_seed.seed, 99);
    assert_eq!(spec_seed.configs[0].seed, 99);
    assert_eq!(spec_seed.configs[1].seed, 77);

    // 9. Explicit validate() method
    let mut bad_grid = spec_seed.clone();
    bad_grid.k = 0;
    assert!(bad_grid.validate().is_err());
    let mut bad_grid2 = spec_seed.clone();
    bad_grid2.sweep_runs = 0;
    assert!(bad_grid2.validate().is_err());

    // 10. Canonical 100k grid file from repository
    let grid_100k_path = std::path::Path::new("results/phase4/grid_100k.grid");
    let grid_100k =
        Phase4GridSpec::from_file(grid_100k_path).expect("grid_100k.grid must parse cleanly");
    assert_eq!(
        grid_100k.configs.len(),
        23,
        "100k grid must contain exactly 23 valid configs (efConstruction >= M)"
    );
    assert_eq!(grid_100k.scale_vectors, Some(100_000));
    assert_eq!(grid_100k.ef_search_list.len(), 12);
}

#[test]
fn test_phase4_prefix_ground_truth_disk_caching() {
    let mut tmp_dir: PathBuf = std::env::temp_dir();
    tmp_dir.push(format!(
        "vecsearch_p4_gt_cache_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let data_dir = tmp_dir.join("data");
    let out_dir1 = tmp_dir.join("out1");
    let out_dir2 = tmp_dir.join("out2");
    fs::create_dir_all(&data_dir).unwrap();

    let grid = Phase4GridSpec {
        scale_vectors: Some(250),
        k: 10,
        sweep_runs: 1,
        seed: 42,
        ef_search_list: vec![10, 20],
        configs: vec![Phase4IndexConfig::new(8, 20)],
    };

    let cfg1 = Phase4SweepConfig {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 250,
            num_query: 20,
            dim: 16,
            num_clusters: 4,
        },
        data_cache_dir: data_dir.clone(),
        output_dir: out_dir1,
        grid: grid.clone(),
        metric: Metric::L2Squared,
        allow_download: false,
        enable_gt_disk_cache: true,
        resume: false,
        max_queries: Some(20),
        expected_commit_override: Some("testcommit".to_string()),
    };

    let summary1 = run_phase4_sweep(&cfg1).expect("Run 1 must succeed");
    assert!(!summary1.ground_truth_cache_hit);
    let gt_cache_file = data_dir.join("gt_synthetic_seed42_250_k100.ivecs");
    assert!(
        gt_cache_file.is_file(),
        "gt_synthetic_seed42_250_k100.ivecs must exist after run 1"
    );

    let cfg2 = Phase4SweepConfig {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 250,
            num_query: 20,
            dim: 16,
            num_clusters: 4,
        },
        data_cache_dir: data_dir.clone(),
        output_dir: out_dir2,
        grid,
        metric: Metric::L2Squared,
        allow_download: false,
        enable_gt_disk_cache: true,
        resume: false,
        max_queries: Some(20),
        expected_commit_override: Some("testcommit".to_string()),
    };

    let summary2 = run_phase4_sweep(&cfg2).expect("Run 2 must succeed");
    assert!(summary2.ground_truth_cache_hit);
    assert_eq!(
        summary1.configs[0].best_recall,
        summary2.configs[0].best_recall
    );

    // Verify Authentic SIFT prefix derivation produces exact "gt_<scale>_k100.ivecs"
    let fake_authentic = LoadedDataset {
        base: RowMatrix::new(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
        query: RowMatrix::new(1, 4, vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        official_ground_truth: None,
        source: DatasetSource::AuthenticSift1M {
            dir: PathBuf::from("data/sift"),
        },
        load_time_s: 0.01,
    };
    let authentic_gt =
        load_or_compute_phase4_ground_truth(&fake_authentic, 1, Metric::L2Squared, &data_dir, true)
            .expect("Authentic SIFT prefix GT computation must succeed");
    assert!(
        data_dir.join("gt_2_k100.ivecs").is_file(),
        "Authentic SIFT prefix GT must produce gt_<scale>_k100.ivecs"
    );
    assert_eq!(
        authentic_gt.ground_truth_cache_path,
        Some(data_dir.join("gt_2_k100.ivecs"))
    );

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_phase4_resumable_sweep_skips_matching_commit_and_rebuilds_mismatched() {
    let mut tmp_dir: PathBuf = std::env::temp_dir();
    tmp_dir.push(format!(
        "vecsearch_p4_resume_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let data_dir = tmp_dir.join("data");
    let out_dir = tmp_dir.join("out");
    fs::create_dir_all(&data_dir).unwrap();

    let grid = Phase4GridSpec {
        scale_vectors: Some(200),
        k: 10,
        sweep_runs: 1,
        seed: 42,
        ef_search_list: vec![10, 20],
        configs: vec![
            Phase4IndexConfig::new(8, 20),
            Phase4IndexConfig::new(12, 24),
        ],
    };

    let cfg = Phase4SweepConfig {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 200,
            num_query: 20,
            dim: 16,
            num_clusters: 4,
        },
        data_cache_dir: data_dir,
        output_dir: out_dir.clone(),
        grid,
        metric: Metric::L2Squared,
        allow_download: false,
        enable_gt_disk_cache: false,
        resume: true,
        max_queries: Some(20),
        expected_commit_override: Some("abc1234".to_string()),
    };

    // First run: measures both configs
    let s1 = run_phase4_sweep(&cfg).expect("Sweep 1 must succeed");
    assert_eq!(s1.newly_measured_count, 2);
    assert_eq!(s1.resumed_skipped_count, 0);

    // Second run with same commit: skips both configs
    let s2 = run_phase4_sweep(&cfg).expect("Sweep 2 must succeed");
    assert_eq!(s2.newly_measured_count, 0);
    assert_eq!(s2.resumed_skipped_count, 2);
    assert!(
        s2.configs[0].best_recall.is_some(),
        "Resumed entry must retain best_recall"
    );
    assert!(
        s2.configs[0].best_qps.is_some(),
        "Resumed entry must retain best_qps"
    );
    let recall_diff =
        (s2.configs[0].best_recall.unwrap() - s1.configs[0].best_recall.unwrap()).abs();
    assert!(recall_diff < 1e-5, "Resumed recall must match original");
    let qps_diff = (s2.configs[0].best_qps.unwrap() - s1.configs[0].best_qps.unwrap()).abs();
    assert!(qps_diff < 1e-1, "Resumed QPS must match original");

    // Mismatched commit: "unknown" commit override must never resume
    let mut cfg_unknown = cfg.clone();
    cfg_unknown.expected_commit_override = Some("unknown".to_string());
    let s_unknown = run_phase4_sweep(&cfg_unknown).expect("Sweep must succeed");
    assert_eq!(s_unknown.newly_measured_count, 2);
    assert_eq!(s_unknown.resumed_skipped_count, 0);

    // Changing sweep_runs must trigger rebuild (smoke run does not block full run)
    let mut cfg_runs = cfg.clone();
    cfg_runs.grid.sweep_runs = 2;
    let s_runs = run_phase4_sweep(&cfg_runs).expect("Sweep must succeed");
    assert_eq!(s_runs.newly_measured_count, 2);
    assert_eq!(s_runs.resumed_skipped_count, 0);

    // Changing max_queries must trigger rebuild
    let mut cfg_queries = cfg.clone();
    cfg_queries.max_queries = Some(10);
    let s_queries = run_phase4_sweep(&cfg_queries).expect("Sweep must succeed");
    assert_eq!(s_queries.newly_measured_count, 2);
    assert_eq!(s_queries.resumed_skipped_count, 0);

    // Dirty-tree report must not be resumed by a clean-tree runner
    let cfg1_file = out_dir.join("scale_200_m8_efc20.json");
    let c1_clean = fs::read_to_string(&cfg1_file).unwrap();
    let c1_dirty = c1_clean.replace("\"git_dirty\": false", "\"git_dirty\": true");
    fs::write(&cfg1_file, c1_dirty).unwrap();
    let can_resume_dirty = can_resume_existing_report(
        &cfg1_file,
        "abc1234",
        "synthetic",
        200,
        10,
        10,
        1,
        &cfg.grid.ef_search_list,
        false, // current runner is clean
        &cfg.grid.configs[0],
    );
    assert!(
        !can_resume_dirty,
        "Clean runner must not resume dirty-tree report"
    );

    // Re-run original clean cfg to re-establish matching report
    let _ = run_phase4_sweep(&cfg).expect("Sweep must succeed");

    // Changing ef_search_list must trigger rebuild
    let mut cfg_ef = cfg.clone();
    cfg_ef.grid.ef_search_list = vec![10, 20, 30];
    let s_ef = run_phase4_sweep(&cfg_ef).expect("Sweep must succeed");
    assert_eq!(s_ef.newly_measured_count, 2);
    assert_eq!(s_ef.resumed_skipped_count, 0);

    // Re-run original clean cfg to re-establish matching report for commit test
    let _ = run_phase4_sweep(&cfg).expect("Sweep must succeed");

    // Modifying config 1's report to have mismatched git_commit: "old0000"
    let c1_content = fs::read_to_string(&cfg1_file).unwrap();
    let corrupted =
        c1_content.replace("\"git_commit\": \"abc1234\"", "\"git_commit\": \"old0000\"");
    fs::write(&cfg1_file, corrupted).unwrap();

    // Third run: rebuilds config 1 and skips config 2
    let s3 = run_phase4_sweep(&cfg).expect("Sweep 3 must succeed");
    assert_eq!(s3.newly_measured_count, 1);
    assert_eq!(s3.resumed_skipped_count, 1);
    assert_eq!(s3.configs[0].status, "measured");
    assert_eq!(s3.configs[1].status, "resumed");

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_phase4_report_json_schema_and_provenance() {
    let mut tmp_dir: PathBuf = std::env::temp_dir();
    tmp_dir.push(format!(
        "vecsearch_p4_schema_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let data_dir = tmp_dir.join("data");
    let out_dir = tmp_dir.join("out");
    fs::create_dir_all(&data_dir).unwrap();

    let grid = Phase4GridSpec {
        scale_vectors: Some(150),
        k: 10,
        sweep_runs: 1,
        seed: 42,
        ef_search_list: vec![10, 25],
        configs: vec![Phase4IndexConfig::new(8, 20)],
    };

    let cfg = Phase4SweepConfig {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 150,
            num_query: 20,
            dim: 16,
            num_clusters: 4,
        },
        data_cache_dir: data_dir,
        output_dir: out_dir.clone(),
        grid,
        metric: Metric::L2Squared,
        allow_download: false,
        enable_gt_disk_cache: false,
        resume: true,
        max_queries: Some(20),
        expected_commit_override: Some("schema_commit_hash".to_string()),
    };

    let _summary = run_phase4_sweep(&cfg).expect("Sweep must succeed");

    let report_file = out_dir.join("scale_150_m8_efc20.json");
    let report_json = fs::read_to_string(&report_file).expect("Report file must exist");

    let required_fields = [
        "\"git_commit\"",
        "\"git_dirty\"",
        "\"rustc_version\"",
        "\"environment\"",
        "\"hardware\"",
        "\"dataset\"",
        "\"ground_truth_source\"",
        "\"hnsw_config\"",
        "\"index_build\"",
        "\"graph_fingerprint_hex\"",
        "\"memory_profile\"",
        "\"graph_only_bytes_per_vector\"",
        "\"ef_search_sweep\"",
        "\"warmup_performed\"",
        "\"measured_this_run\"",
        "\"is_authentic_sift\"",
        "\"peak_rss_scope\"",
    ];

    for field in &required_fields {
        assert!(
            report_json.contains(field),
            "Per-index report JSON must contain field {field}"
        );
    }

    let summary_file = out_dir.join("summary.json");
    let summary_json = fs::read_to_string(&summary_file).expect("Summary file must exist");

    let required_summary_fields = [
        "\"git_commit\"",
        "\"git_dirty\"",
        "\"rustc_version\"",
        "\"environment\"",
        "\"hardware\"",
        "\"dataset\"",
        "\"ground_truth_source\"",
        "\"scale_vectors\"",
        "\"total_configs\"",
        "\"newly_measured_count\"",
        "\"resumed_skipped_count\"",
        "\"is_authentic_sift\"",
        "\"configs\"",
    ];

    for field in &required_summary_fields {
        assert!(
            summary_json.contains(field),
            "Summary JSON must contain field {field}"
        );
    }

    let _ = fs::remove_dir_all(&tmp_dir);
}
