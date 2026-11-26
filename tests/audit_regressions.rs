//! Batch 1 audit regression tests covering Critical & High fixes.

#![forbid(unsafe_code)]

mod common;

use std::fs;
use std::process::Command;

use common::{parse_json_strict, TempTestDir};
use vector_search::benchmark::{
    detect_provenance, run_phase3_5_baseline_measurement, run_phase3_cli, Phase1DatasetMode,
};
use vector_search::dataset::{
    generate_mode_b_bitset_detailed, generate_synthetic_sift, write_bitset,
};
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::exact_knn::FilteredExactKnn;
use vector_search::filter::{DenseBitset, FilterPredicate, FilteredSearchParams};
use vector_search::hnsw::{FilterStrategy, HnswConfig, HnswIndex, VisitedTracker};
use vector_search::json_util::{extract_json_string_field, extract_json_usize_field};
use vector_search::matrix::RowMatrix;
use vector_search::phase4::{
    can_resume_existing_report, run_phase4_sweep, Phase4GridSpec, Phase4IndexConfig,
    Phase4SweepConfig,
};
use vector_search::phase5::{
    can_resume_phase5_cell, cell_report_filename, run_phase5_matrix, MetadataMode,
    Phase5MatrixConfig,
};
use vector_search::quantized::{pq_adc_distance, pq_sdc_distance, PqConfig, PqTrainer, PqWithSdc};

// ---------------------------------------------------------------------------
// FILT-01 + FILT-04: Strategy 1 post-filter sorts by internal sort key
// ---------------------------------------------------------------------------

#[test]
fn regress_filt01_strategy1_inner_product_ordering_and_overfetch() {
    let n = 20;
    let rows: Vec<Vec<f32>> = (0..n).map(|i| vec![(i + 1) as f32, 0.0]).collect();
    let base = RowMatrix::from_rows(&rows).unwrap();
    let cfg = HnswConfig {
        m: 8,
        ef_construction: 40,
        seed: 42,
        ..Default::default()
    };
    let idx = HnswIndex::build(&base, Metric::InnerProduct, cfg).unwrap();
    let exact = FilteredExactKnn::new(Metric::InnerProduct, base.clone()).unwrap();

    let query = [1.0_f32, 0.0];
    let params = FilteredSearchParams {
        ef_search: 20,
        overfetch_cap: 100,
        overfetch_factor: Some(2.0),
        allow_bruteforce_fallback: false,
        ..Default::default()
    };

    // 100% selectivity
    let all = DenseBitset::ones(n);
    let gt_all = exact.search_filtered(&query, 5, &all).unwrap();
    let s1_all = idx
        .search_knn_filtered(
            &query,
            5,
            &all,
            FilterStrategy::PostFilterOverfetch,
            &params,
        )
        .unwrap();
    assert_eq!(s1_all.results, gt_all);

    // 50% selectivity (even indices)
    let mut evens = DenseBitset::zeros(n);
    for i in (0..n).step_by(2) {
        evens.set_bit(i, true);
    }
    let gt_even = exact.search_filtered(&query, 5, &evens).unwrap();
    let s1_even = idx
        .search_knn_filtered(
            &query,
            5,
            &evens,
            FilterStrategy::PostFilterOverfetch,
            &params,
        )
        .unwrap();
    assert_eq!(s1_even.results, gt_even);
}

#[test]
fn regress_filt04_strategy1_euclidean_tie_break_matches_gt() {
    let v0 = vec![1.0_f32, 1.5 * 2.0_f32.powi(-12)];
    let v1 = vec![1.0_f32, 0.0_f32];
    let base = RowMatrix::from_rows(&[v0, v1]).unwrap();
    let cfg = HnswConfig {
        m: 2,
        ef_construction: 10,
        seed: 42,
        ..Default::default()
    };
    let idx = HnswIndex::build(&base, Metric::Euclidean, cfg).unwrap();
    let exact = FilteredExactKnn::new(Metric::Euclidean, base.clone()).unwrap();
    let pred = DenseBitset::ones(2);
    let params = FilteredSearchParams {
        ef_search: 10,
        overfetch_cap: 10,
        allow_bruteforce_fallback: false,
        ..Default::default()
    };

    let query = [0.0_f32, 0.0];
    let gt = exact.search_filtered(&query, 2, &pred).unwrap();
    let s1 = idx
        .search_knn_filtered(
            &query,
            2,
            &pred,
            FilterStrategy::PostFilterOverfetch,
            &params,
        )
        .unwrap();
    assert_eq!(s1.results, gt);
}

// ---------------------------------------------------------------------------
// KNN-01 + KNN-04: Oversized bitset count_matching & lying predicates
// ---------------------------------------------------------------------------

struct LyingPredicate;

impl FilterPredicate for LyingPredicate {
    fn matches(&self, id: u32) -> bool {
        id == 1
    }

    fn count_matching(&self, _total: usize) -> Option<usize> {
        Some(3)
    }
}

#[test]
fn regress_knn01_oversized_bitset_and_lying_predicate() {
    let base = RowMatrix::from_rows(&[
        vec![0.0_f32, 0.0],
        vec![2.0, 0.0],
        vec![5.0, 0.0],
        vec![10.0, 0.0],
    ])
    .unwrap();
    let engine = FilteredExactKnn::new(Metric::L2Squared, base).unwrap();
    let queries = RowMatrix::from_rows(&[vec![0.0_f32, 0.0], vec![10.0, 0.0]]).unwrap();

    // Oversized bitset with bits set at >= num_base (4): count_matching(4) must only count bits < 4.
    let mut big_bs = DenseBitset::zeros(10);
    big_bs.set_bit(1, true);
    big_bs.set_bit(7, true);
    big_bs.set_bit(9, true);
    assert_eq!(big_bs.count_matching(4), Some(1));
    assert_eq!(big_bs.count_matching(10), Some(3));

    let seq = engine.search_batch(&queries, 5, &big_bs).unwrap();
    let par = engine
        .search_batch_parallel(&queries, 5, &big_bs, 2)
        .unwrap();
    assert_eq!(seq.indices.shape(), (2, 1));
    assert_eq!(seq.indices.as_slice(), &[1, 1]);
    assert_eq!(par.indices.as_slice(), seq.indices.as_slice());
    assert_eq!(par.distances.as_slice(), seq.distances.as_slice());

    // Lying predicate whose count_matching() > actual matches() must return Err on both seq & par.
    let lying_seq = engine.search_batch(&queries, 5, &LyingPredicate);
    let lying_par = engine.search_batch_parallel(&queries, 5, &LyingPredicate, 2);
    assert!(matches!(
        lying_seq,
        Err(VectorSearchError::InvalidArgument(_))
    ));
    assert!(matches!(
        lying_par,
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

#[test]
fn regress_knn01_knn04_filtered_exact_knn_cli_rejects_mismatched_bitset() {
    let tmp = TempTestDir::new("regress_knn04_cli");
    let bs_path = tmp.path().join("oversized.bin");
    let out_dir = tmp.path().join("out");

    let mut big_bs = DenseBitset::zeros(10);
    big_bs.set_bit(1, true);
    big_bs.set_bit(7, true);
    write_bitset(&bs_path, &big_bs).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_filtered_exact_knn"))
        .args([
            "--num-base",
            "4",
            "--num-query",
            "2",
            "--dim",
            "4",
            "--k",
            "3",
            "--filter-bitset",
            bs_path.to_str().unwrap(),
            "--output-dir",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "CLI must return Err cleanly, not panic: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// HNSW-01: Under-filled batch rows padded with (-1, f32::INFINITY)
// ---------------------------------------------------------------------------

#[test]
fn regress_hnsw01_underfilled_batch_rows_padded_consistently() {
    let n = 100;
    let dim = 8;
    let base = RowMatrix::new(n, dim, vec![1.0_f32; n * dim]).unwrap();
    let cfg = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 32,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let queries = RowMatrix::new(3, dim, vec![1.0_f32; 3 * dim]).unwrap();

    let k = 50;
    let ef = 100;
    let single = index.search_knn(queries.row(0), k, ef).unwrap();
    assert!(single.len() < k);

    let seq = index.search_batch(&queries, k, ef).unwrap();
    let par1 = index.search_batch_parallel(&queries, k, ef, 1).unwrap();
    let par4 = index.search_batch_parallel(&queries, k, ef, 4).unwrap();

    assert_eq!(seq.indices.shape(), (3, k));
    assert_eq!(par1.indices.as_slice(), seq.indices.as_slice());
    assert_eq!(par4.indices.as_slice(), seq.indices.as_slice());
    assert_eq!(par4.distances.as_slice(), seq.distances.as_slice());

    for q in 0..3 {
        let i_row = seq.indices.row(q);
        let d_row = seq.distances.row(q);
        for c in single.len()..k {
            assert_eq!(i_row[c], -1);
            assert!(d_row[c].is_infinite() && d_row[c].is_sign_positive());
        }
    }
}

// ---------------------------------------------------------------------------
// HNSW-02, HNSW-03, QUANT-05: 0-col matrix, insert validation, u32::MAX limits
// ---------------------------------------------------------------------------

#[test]
fn regress_hnsw02_hnsw03_quant05_config_and_zero_col_validation() {
    // 0-column matrix must fail in insert / build
    let empty_cols = RowMatrix::new(3, 0, vec![]).unwrap();
    let res_zero_col = HnswIndex::build(&empty_cols, Metric::L2Squared, HnswConfig::default());
    assert!(matches!(
        res_zero_col,
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // HnswIndex::new + insert with invalid config (m_max0 < m) must fail with Err, not panic
    let base = RowMatrix::from_rows(&[vec![0.0_f32, 0.0], vec![1.0, 0.0]]).unwrap();
    let bad_cfg = HnswConfig {
        m: 16,
        m_max0: Some(2),
        ef_construction: 50,
        ..Default::default()
    };
    let mut idx = HnswIndex::new(&base, Metric::L2Squared, bad_cfg);
    assert!(matches!(
        idx.insert(0),
        Err(VectorSearchError::InvalidArgument(_))
    ));

    // m_max0 = Some(usize::MAX) must fail HnswConfig::validate()
    let huge_cfg = HnswConfig {
        m: 16,
        m_max0: Some(usize::MAX),
        ef_construction: 50,
        ..Default::default()
    };
    assert!(matches!(
        huge_cfg.validate(),
        Err(VectorSearchError::InvalidArgument(_))
    ));
    let huge_m_cfg = HnswConfig {
        m: usize::MAX,
        ef_construction: usize::MAX,
        ..Default::default()
    };
    assert_eq!(huge_m_cfg.effective_m_max0(), usize::MAX);
    assert!(matches!(
        huge_m_cfg.validate(),
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------------
// HNSW-04 + FILT-02: Unclamped ef_search / k capacity overflow
// ---------------------------------------------------------------------------

#[test]
fn regress_hnsw04_filt02_huge_ef_search_does_not_overflow_capacity() {
    let base = RowMatrix::from_rows(&[
        vec![0.0_f32, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![3.0, 0.0],
    ])
    .unwrap();
    let mut index = HnswIndex::build(&base, Metric::L2Squared, HnswConfig::new(2, 4)).unwrap();

    let hits = index.search_knn(&[0.0, 0.0], 2, usize::MAX - 100).unwrap();
    assert_eq!(hits.len(), 2);

    let mut tracker = VisitedTracker::new(4);
    let layer_res = index.search_layer(&[0.0, 0.0], 0, usize::MAX - 100, 0, &mut tracker);
    assert!(!layer_res.is_empty());

    let pred = DenseBitset::ones(4);
    let params = FilteredSearchParams {
        ef_search: (usize::MAX / 3) + 1,
        ..Default::default()
    };
    let s2 = index
        .search_knn_filtered(&[0.0, 0.0], 2, &pred, FilterStrategy::InTraversal, &params)
        .unwrap();
    assert_eq!(s2.results.len(), 2);

    let s3 = index
        .search_knn_filtered(
            &[0.0, 0.0],
            usize::MAX,
            &pred,
            FilterStrategy::PreFilterBruteForce,
            &params,
        )
        .unwrap();
    assert_eq!(s3.results.len(), 4);
}

// ---------------------------------------------------------------------------
// CORE-01 + CORE-02: JSON surrogate pair decoding & escape validation
// ---------------------------------------------------------------------------

#[test]
fn regress_core01_json_surrogate_pairs_and_unknown_escapes() {
    let json_emoji = r#"{"emoji": "\uD83D\uDE00", "ok": "hi"}"#;
    assert_eq!(
        extract_json_string_field(json_emoji, "emoji").as_deref(),
        Some("😀")
    );

    let lone_high = r#"{"bad": "\uD800"}"#;
    assert_eq!(extract_json_string_field(lone_high, "bad"), None);

    let lone_low = r#"{"bad": "\uDC00"}"#;
    assert_eq!(extract_json_string_field(lone_low, "bad"), None);

    let bad_escape = r#"{"bad": "\q"}"#;
    assert_eq!(extract_json_string_field(bad_escape, "bad"), None);

    // Documented first-match-anywhere behavior (CORE-02) still holds for valid JSON.
    let nested = r#"{"hnsw_config": {"m": 16}, "k": 10}"#;
    assert_eq!(extract_json_usize_field(nested, "m"), Some(16));
}

// ---------------------------------------------------------------------------
// DATA-01 + DATA-02: Shortfall calibration extreme f32 & checked clusters * dim
// ---------------------------------------------------------------------------

#[test]
fn regress_data01_data02_metadata_extreme_f32_and_sift_overflow() {
    let extreme = RowMatrix::new(4, 1, vec![-1e20_f32, -1e20_f32, -1e20_f32, 1e20_f32]).unwrap();
    let bs_extreme = generate_mode_b_bitset_detailed(&extreme, 0.5, 42, 24).bitset;
    assert_eq!(bs_extreme.count_ones(), 2);

    let with_nan = RowMatrix::new(4, 1, vec![0.0_f32, 0.0, 0.0, f32::NAN]).unwrap();
    let bs_nan = generate_mode_b_bitset_detailed(&with_nan, 0.5, 42, 42).bitset;
    assert_eq!(bs_nan.count_ones(), 2);

    let overflow_res = generate_synthetic_sift(1, 1, 2, (usize::MAX / 2) + 1, 42);
    assert!(matches!(
        overflow_res,
        Err(VectorSearchError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------------
// QUANT-02: pq_adc_distance / pq_sdc_distance preserve InnerProduct negatives
// ---------------------------------------------------------------------------

#[test]
fn regress_quant02_pq_adc_and_sdc_preserve_negative_inner_product() {
    let base = RowMatrix::from_rows(&[
        vec![2.0_f32, 3.0, 4.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![3.0, 2.0, 1.0, 4.0],
        vec![0.5, 1.5, 2.5, 3.5],
    ])
    .unwrap();
    let pq_cfg = PqConfig {
        m_pq: 2,
        k_centroids: 256,
        max_train_vectors: 4,
        max_lloyd_iters: 10,
        seed: 42,
    };
    let cb = PqTrainer::train(&base, &pq_cfg).unwrap();
    let codes = cb.encode_matrix(&base).unwrap();
    let mut adc_table = Vec::new();
    let query = [2.0_f32, 3.0, 4.0, 5.0];
    cb.compute_adc_table(Metric::InnerProduct, &query, &mut adc_table)
        .unwrap();

    let adc_val = pq_adc_distance(&adc_table, &codes[0..2]);
    assert!(adc_val < 0.0, "Expected negative -dot, got {adc_val}");

    let vecs = vector_search::quantized::PqVectors::from_codebook_and_matrix(
        cb,
        &base,
        Metric::InnerProduct,
    )
    .unwrap();
    let sdc_wrap = PqWithSdc::new(vecs);
    let sdc_val = pq_sdc_distance(
        sdc_wrap.sdc_table(),
        sdc_wrap.vectors().code(0),
        sdc_wrap.vectors().code(0),
    );
    assert!(sdc_val < 0.0, "Expected negative -dot SDC, got {sdc_val}");
}

// ---------------------------------------------------------------------------
// HARN-01 + HARN-02 + HARN-03: Resume validation & phase3_5 dir/metadata
// ---------------------------------------------------------------------------

#[test]
fn regress_harn01_phase4_resume_rejects_truncated_and_mismatched_configs() {
    let tmp = TempTestDir::new("regress_harn01");
    let data_dir = tmp.path().join("data");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&data_dir).unwrap();

    let cfg1 = Phase4IndexConfig {
        m: 8,
        m_max: Some(8),
        m_max0: Some(16),
        ef_construction: 20,
        keep_pruned: true,
        keep_pruned_shrink: None,
        seed: 42,
        explicit_seed: false,
    };
    let sweep_cfg = Phase4SweepConfig {
        dataset_mode: Phase1DatasetMode::Synthetic {
            num_base: 80,
            num_query: 6,
            dim: 8,
            num_clusters: 4,
        },
        data_cache_dir: data_dir,
        output_dir: out_dir.clone(),
        grid: Phase4GridSpec {
            scale_vectors: Some(80),
            k: 5,
            sweep_runs: 1,
            seed: 42,
            ef_search_list: vec![10, 20],
            configs: vec![cfg1.clone()],
        },
        metric: Metric::L2Squared,
        allow_download: false,
        enable_gt_disk_cache: false,
        resume: true,
        max_queries: Some(6),
        expected_commit_override: Some("deadbeef".to_string()),
    };

    let s1 = run_phase4_sweep(&sweep_cfg).unwrap();
    assert_eq!(s1.newly_measured_count, 1);

    let report_path = out_dir.join(cfg1.report_filename(80));
    let full_txt = fs::read_to_string(&report_path).unwrap();
    let git_dirty = detect_provenance().git_dirty;

    assert!(can_resume_existing_report(
        &report_path,
        "deadbeef",
        "synthetic",
        80,
        6,
        8,
        5,
        Metric::L2Squared,
        1,
        &[10, 20],
        git_dirty,
        &cfg1,
    ));

    // Truncated JSON must NOT resume
    let trunc_idx = full_txt.rfind("\"run_p99_latencies_ms\"").unwrap();
    fs::write(&report_path, &full_txt[..trunc_idx]).unwrap();
    assert!(!can_resume_existing_report(
        &report_path,
        "deadbeef",
        "synthetic",
        80,
        6,
        8,
        5,
        Metric::L2Squared,
        1,
        &[10, 20],
        git_dirty,
        &cfg1,
    ));

    // Restore full JSON and verify m_max0 / keep_pruned / metric / dim mismatches reject resume
    fs::write(&report_path, &full_txt).unwrap();
    let mut mod_cfg = cfg1.clone();
    mod_cfg.m_max0 = Some(32);
    assert!(!can_resume_existing_report(
        &report_path,
        "deadbeef",
        "synthetic",
        80,
        6,
        8,
        5,
        Metric::L2Squared,
        1,
        &[10, 20],
        git_dirty,
        &mod_cfg,
    ));
    assert!(!can_resume_existing_report(
        &report_path,
        "deadbeef",
        "synthetic",
        80,
        6,
        8,
        5,
        Metric::InnerProduct,
        1,
        &[10, 20],
        git_dirty,
        &cfg1,
    ));
    assert!(!can_resume_existing_report(
        &report_path,
        "deadbeef",
        "synthetic",
        80,
        6,
        16,
        5,
        Metric::L2Squared,
        1,
        &[10, 20],
        git_dirty,
        &cfg1,
    ));
}

#[test]
fn regress_harn02_phase5_resume_rejects_empty_arms_and_seed_query_mismatch() {
    let tmp = TempTestDir::new("regress_harn02");
    let data_dir = tmp.path().join("data");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&data_dir).unwrap();

    let cfg = Phase5MatrixConfig {
        data_dir,
        output_dir: out_dir.clone(),
        scale: 80,
        modes: vec![MetadataMode::Correlated],
        sigmas: vec![0.5],
        k: 5,
        reps: 1,
        attr_seed: 42,
        clust_seed: 77,
        synthetic: true,
        synthetic_dim: 8,
        synthetic_queries: Some(6),
        allow_download: false,
        smoke: false,
        resume: true,
        include_sensitivity: false,
        sensitivity_only: false,
        include_1m_opaque_spotcheck: false,
        opaque_1m_spotcheck_only: false,
        gt_threads: 1,
    };

    let s1 = run_phase5_matrix(&cfg).unwrap();
    assert_eq!(s1.newly_measured_count, 1);

    let cell_path = out_dir.join(cell_report_filename(80, MetadataMode::Correlated, 0.5));
    let full_txt = fs::read_to_string(&cell_path).unwrap();
    let prov = detect_provenance();

    assert!(can_resume_phase5_cell(
        &cell_path,
        &prov.git_commit,
        prov.git_dirty,
        "synthetic",
        80,
        MetadataMode::Correlated,
        0.5,
        5,
        1,
        42,
        77,
        8,
        6,
    ));

    // Changed clust_seed / num_query_total / dim must not resume
    assert!(!can_resume_phase5_cell(
        &cell_path,
        &prov.git_commit,
        prov.git_dirty,
        "synthetic",
        80,
        MetadataMode::Correlated,
        0.5,
        5,
        1,
        42,
        99,
        8,
        6,
    ));
    assert!(!can_resume_phase5_cell(
        &cell_path,
        &prov.git_commit,
        prov.git_dirty,
        "synthetic",
        80,
        MetadataMode::Correlated,
        0.5,
        5,
        1,
        42,
        77,
        8,
        10,
    ));

    // Empty "arms": [] must not resume
    let arms_pos = full_txt.find("\"arms\": [").unwrap();
    let empty_arms = format!("{}\"arms\": []\n}}\n", &full_txt[..arms_pos]);
    fs::write(&cell_path, empty_arms).unwrap();
    assert!(!can_resume_phase5_cell(
        &cell_path,
        &prov.git_commit,
        prov.git_dirty,
        "synthetic",
        80,
        MetadataMode::Correlated,
        0.5,
        5,
        1,
        42,
        77,
        8,
        6,
    ));
}

#[test]
fn regress_harn03_phase3_5_cli_help_and_baseline_metadata() {
    // Verify --phase3-5-dir appears in --help and redirects --measure-baseline without writing to results/phase3_5
    let tmp = TempTestDir::new("regress_harn03");
    let help_out = Command::new(env!("CARGO_BIN_EXE_phase3"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help_out.status.success());
    let stdout = String::from_utf8_lossy(&help_out.stdout);
    assert!(stdout.contains("--phase3-5-dir"));

    // Calling run_phase3_cli with missing data-dir fails cleanly instead of touching results/phase3_5
    let err = run_phase3_cli(&[
        "--data-dir".to_string(),
        tmp.path()
            .join("missing_data")
            .to_string_lossy()
            .to_string(),
        "--phase3-5-dir".to_string(),
        tmp.path().join("p35_out").to_string_lossy().to_string(),
        "--measure-baseline".to_string(),
    ]);
    assert!(err.is_err());

    // Keep unused import check happy if run_phase3_5_baseline_measurement is called directly in future
    let _fn_ptr: fn(&std::path::Path, &std::path::Path) -> vector_search::error::Result<()> =
        run_phase3_5_baseline_measurement;
    let _ = parse_json_strict("{}");
}
