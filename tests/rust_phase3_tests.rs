//! Phase 3 regression tests. They run on small synthetic data so they need no SIFT download.

use vector_search::benchmark::{
    run_phase2_benchmark, run_phase3_benchmark, Phase1DatasetMode, Phase2Config, Phase3Config,
};
use vector_search::dataset::generate_synthetic_sift;
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::hnsw::{HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;

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
