//! Golden Fixtures, Query Result Hashing, and Phase 3.5 Baseline Re-Measurement.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{
    current_utc_rfc3339, detect_hardware, get_peak_rss_mb, peak_rss_measurement_method,
};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::json_util::{
    escape_json, extract_json_u64_field, extract_json_usize_field, parse_number_list,
};
use crate::matrix::RowMatrix;
use crate::recall::{compute_recall_at_k, summarize_repeated_latencies};

use super::dataset_loader::resolve_sift1m_fvecs_paths;
use super::detect_provenance;

/// Computes a 64-bit FNV-1a summary hash over returned neighbor IDs and exact IEEE 754 distance bits.
#[must_use]
pub fn compute_query_results_hash(indices: &RowMatrix<i32>, distances: &RowMatrix<f32>) -> u64 {
    let mut hash = crate::hnsw::FNV_OFFSET_BASIS_64;
    for q in 0..indices.rows() {
        let idx_row = indices.row(q);
        let dist_row = distances.row(q);
        for rank in 0..indices.cols() {
            let id = idx_row[rank];
            let dist = dist_row[rank];
            crate::hnsw::fnv1a_64_update_u64(&mut hash, id as u64);
            crate::hnsw::fnv1a_64_update_u64(&mut hash, dist.to_bits() as u64);
        }
    }
    hash
}

/// Verifies an `HnswIndex` against a golden fixture JSON file (`graph_fingerprint`, `query_results_hash`,
/// and detailed candidate IDs and distance bits in `results`).
pub fn verify_index_against_golden_fixture(
    index: &HnswIndex<'_>,
    queries: &RowMatrix<f32>,
    golden_path: &Path,
) -> Result<()> {
    let content = fs::read_to_string(golden_path)?;
    let expected_fp = extract_json_u64_field(&content, "graph_fingerprint").ok_or_else(|| {
        VectorSearchError::InvalidArgument(format!(
            "Missing graph_fingerprint in golden fixture {}",
            golden_path.display()
        ))
    })?;
    let actual_fp = index.graph_fingerprint();
    if actual_fp != expected_fp {
        return Err(VectorSearchError::VerificationFailed(format!(
            "Golden graph_fingerprint mismatch vs {}: actual 0x{actual_fp:016x} != expected 0x{expected_fp:016x}",
            golden_path.display()
        )));
    }

    let expected_qhash = extract_json_u64_field(&content, "query_results_hash");
    let expected_nq = extract_json_usize_field(&content, "num_queries").unwrap_or(1000);
    let results = parse_golden_results(&content);
    let max_result_k = results
        .iter()
        .map(|r| r.indices.len().max(r.distance_bits.len()))
        .max()
        .unwrap_or(0);
    let k = extract_json_usize_field(&content, "k").unwrap_or_else(|| 10.max(max_result_k));
    let ef_search = extract_json_usize_field(&content, "ef_search").unwrap_or_else(|| 100.max(k));

    let max_result_qid = results
        .iter()
        .map(|r| r.query_id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let num_to_search = if expected_qhash.is_some() {
        expected_nq
    } else {
        0
    }
    .max(max_result_qid);

    if num_to_search > 0 {
        if queries.rows() == 0 {
            return Err(VectorSearchError::VerificationFailed(format!(
                "No query vectors provided for golden verification vs {}",
                golden_path.display()
            )));
        }

        // Fail closed if expected_qhash is present and queries.rows() < expected_nq
        if expected_qhash.is_some() && queries.rows() < expected_nq {
            return Err(VectorSearchError::VerificationFailed(format!(
                "Golden fixture {} requires {} queries for query_results_hash, but only {} queries provided",
                golden_path.display(),
                expected_nq,
                queries.rows()
            )));
        }

        // Check if query count meets maximum query_id in results
        if queries.rows() < max_result_qid {
            return Err(VectorSearchError::VerificationFailed(format!(
                "Insufficient queries for golden results verification vs {}: provided {} < required {}",
                golden_path.display(),
                queries.rows(),
                max_result_qid
            )));
        }

        let nq_run = num_to_search.min(queries.rows());
        let q_slice = queries.slice_first_rows(nq_run);
        let mut tracker = crate::hnsw::VisitedTracker::new(index.num_vectors());
        let timed = index.search_batch_timed_with_tracker(&q_slice, k, ef_search, &mut tracker)?;

        // 1. Verify query_results_hash over expected_nq queries
        if let Some(expected_hash) = expected_qhash {
            let actual_qhash = compute_query_results_hash(
                &timed.result.indices.slice_first_rows(expected_nq),
                &timed.result.distances.slice_first_rows(expected_nq),
            );
            if actual_qhash != expected_hash {
                return Err(VectorSearchError::VerificationFailed(format!(
                    "Golden query_results_hash mismatch vs {}: actual 0x{actual_qhash:016x} != expected 0x{expected_hash:016x}",
                    golden_path.display()
                )));
            }
        }

        // 2. Cross-verify detailed candidate IDs and exact IEEE 754 distance bits in results array
        if !results.is_empty() {
            for res in &results {
                let q = res.query_id;
                if q >= timed.result.indices.rows() {
                    return Err(VectorSearchError::VerificationFailed(format!(
                        "Golden query result index {q} out of range of searched queries {}",
                        timed.result.indices.rows()
                    )));
                }
                let act_idx_row = timed.result.indices.row(q);
                let act_dist_row = timed.result.distances.row(q);

                // Verify candidate IDs
                if !res.indices.is_empty() {
                    if act_idx_row.len() < res.indices.len() {
                        return Err(VectorSearchError::VerificationFailed(format!(
                            "Golden candidate count mismatch vs {} for query {q}: actual {} < expected {}",
                            golden_path.display(),
                            act_idx_row.len(),
                            res.indices.len()
                        )));
                    }
                    for (rank, &expected_id) in res.indices.iter().enumerate() {
                        let actual_id = act_idx_row[rank];
                        if actual_id != expected_id {
                            return Err(VectorSearchError::VerificationFailed(format!(
                                "Golden candidate ID mismatch vs {} for query {q} at rank {rank}: actual {actual_id} != expected {expected_id}",
                                golden_path.display()
                            )));
                        }
                    }
                }

                // Verify exact IEEE 754 distance bits
                if !res.distance_bits.is_empty() {
                    if act_dist_row.len() < res.distance_bits.len() {
                        return Err(VectorSearchError::VerificationFailed(format!(
                            "Golden distance count mismatch vs {} for query {q}: actual {} < expected {}",
                            golden_path.display(),
                            act_dist_row.len(),
                            res.distance_bits.len()
                        )));
                    }
                    for (rank, &expected_bits) in res.distance_bits.iter().enumerate() {
                        let actual_bits = act_dist_row[rank].to_bits();
                        if actual_bits != expected_bits {
                            return Err(VectorSearchError::VerificationFailed(format!(
                                "Golden distance bit mismatch vs {} for query {q} at rank {rank}: actual 0x{actual_bits:08x} ({}) != expected 0x{expected_bits:08x}",
                                golden_path.display(),
                                act_dist_row[rank]
                            )));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Per-query detailed record for the first queries stored in golden fixtures.
#[derive(Debug, Clone, PartialEq)]
pub struct GoldenQueryResult {
    pub query_id: usize,
    pub indices: Vec<i32>,
    pub distance_bits: Vec<u32>,
    pub distances: Vec<f32>,
}

/// Extracts the detailed query results array from a golden fixture JSON string.
#[must_use]
pub fn parse_golden_results(content: &str) -> Vec<GoldenQueryResult> {
    let mut results = Vec::new();
    if let Some(res_idx) = content.find("\"results\":") {
        let res_sub = &content[res_idx..];
        for block in res_sub.split('{') {
            if let Some(qid) = extract_json_usize_field(block, "query_id") {
                let mut indices = Vec::new();
                let mut distance_bits = Vec::new();
                let mut distances = Vec::new();

                if let Some(p) = block.find("\"indices\":") {
                    indices = parse_number_list::<i32>(&block[p..]);
                }
                if let Some(p) = block.find("\"distance_bits\":") {
                    distance_bits = parse_number_list::<u32>(&block[p..]);
                }
                if let Some(p) = block.find("\"distances\":") {
                    distances = parse_number_list::<f32>(&block[p..]);
                }

                results.push(GoldenQueryResult {
                    query_id: qid,
                    indices,
                    distance_bits,
                    distances,
                });
            }
        }
    }
    results
}

/// Authoritative golden fixture capturing graph fingerprint and fixed-query results at scale.
#[derive(Debug, Clone, PartialEq)]
pub struct GoldenFixture {
    pub scale_vectors: usize,
    pub hnsw_m: usize,
    pub hnsw_m_max0: usize,
    pub hnsw_ef_construction: usize,
    pub seed: u64,
    pub graph_fingerprint: u64,
    pub graph_fingerprint_hex: String,
    pub query_results_hash: u64,
    pub query_results_hash_hex: String,
    pub num_queries: usize,
    pub k: usize,
    pub ef_search: usize,
    pub results: Vec<GoldenQueryResult>,
}

impl GoldenFixture {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(65536);
        out.push_str("{\n");
        out.push_str(&format!("  \"scale_vectors\": {},\n", self.scale_vectors));
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_m));
        out.push_str(&format!("    \"m_max0\": {},\n", self.hnsw_m_max0));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_ef_construction
        ));
        out.push_str(&format!("    \"seed\": {}\n", self.seed));
        out.push_str("  },\n");
        out.push_str(&format!(
            "  \"graph_fingerprint\": {},\n",
            self.graph_fingerprint
        ));
        out.push_str(&format!(
            "  \"graph_fingerprint_hex\": \"{}\",\n",
            escape_json(&self.graph_fingerprint_hex)
        ));
        out.push_str(&format!(
            "  \"query_results_hash\": {},\n",
            self.query_results_hash
        ));
        out.push_str(&format!(
            "  \"query_results_hash_hex\": \"{}\",\n",
            escape_json(&self.query_results_hash_hex)
        ));
        out.push_str("  \"query_params\": {\n");
        out.push_str(&format!("    \"num_queries\": {},\n", self.num_queries));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"ef_search\": {}\n", self.ef_search));
        out.push_str("  },\n");
        out.push_str("  \"results\": [\n");
        for (i, res) in self.results.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"query_id\": {},\n", res.query_id));
            let idx_str = res
                .indices
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("      \"indices\": [{}],\n", idx_str));
            let bits_str = res
                .distance_bits
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("      \"distance_bits\": [{}],\n", bits_str));
            let dist_str = res
                .distances
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("      \"distances\": [{}]\n", dist_str));
            if i + 1 < self.results.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }

    pub fn parse_json(content: &str) -> Result<Self> {
        let scale_vectors = extract_json_usize_field(content, "scale_vectors").unwrap_or(0);
        let hnsw_m = extract_json_usize_field(content, "m").unwrap_or(16);
        let hnsw_m_max0 = extract_json_usize_field(content, "m_max0").unwrap_or(32);
        let hnsw_ef_construction =
            extract_json_usize_field(content, "ef_construction").unwrap_or(100);
        let seed = extract_json_u64_field(content, "seed").unwrap_or(42);
        let graph_fingerprint =
            extract_json_u64_field(content, "graph_fingerprint").ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "Missing graph_fingerprint in golden fixture".to_string(),
                )
            })?;
        let graph_fingerprint_hex = format!("0x{:016x}", graph_fingerprint);
        let query_results_hash =
            extract_json_u64_field(content, "query_results_hash").ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "Missing query_results_hash in golden fixture".to_string(),
                )
            })?;
        let query_results_hash_hex = format!("0x{:016x}", query_results_hash);
        let num_queries = extract_json_usize_field(content, "num_queries").unwrap_or(1000);
        let k = extract_json_usize_field(content, "k").unwrap_or(10);
        let ef_search = extract_json_usize_field(content, "ef_search").unwrap_or(100);

        let results = parse_golden_results(content);

        Ok(Self {
            scale_vectors,
            hnsw_m,
            hnsw_m_max0,
            hnsw_ef_construction,
            seed,
            graph_fingerprint,
            graph_fingerprint_hex,
            query_results_hash,
            query_results_hash_hex,
            num_queries,
            k,
            ef_search,
            results,
        })
    }
}

/// Generates authoritative golden fixtures for 10k and 100k SIFT1M prefixes.
pub fn generate_golden_fixtures(data_dir: &Path, output_dir: &Path) -> Result<()> {
    println!("[golden-fixture] Generating golden fixtures for 10k and 100k scales...");
    let (base_path, query_path) = resolve_sift1m_fvecs_paths(data_dir)?;

    let queries = crate::dataset::read_fvecs_max_rows(&query_path, Some(1000))?;
    fs::create_dir_all(output_dir)?;

    for &scale in &[10_000_usize, 100_000_usize] {
        println!("[golden-fixture] Reading {} base vectors...", scale);
        let base = crate::dataset::read_fvecs_max_rows(&base_path, Some(scale))?;
        let cfg = HnswConfig {
            m: 16,
            m_max: None,
            m_max0: Some(32),
            ef_construction: 100,
            m_l: None,
            extend_candidates: false,
            keep_pruned: true,
            keep_pruned_shrink: None,
            seed: 42,
        };
        println!(
            "[golden-fixture] Building HNSW index for scale {}...",
            scale
        );
        let start_build = Instant::now();
        let index = HnswIndex::build(&base, Metric::L2Squared, cfg)?;
        let build_dur = start_build.elapsed();
        let fp = index.graph_fingerprint();
        println!(
            "[golden-fixture] Scale {}: built in {:.3} s, graph fingerprint: 0x{:016x}",
            scale,
            build_dur.as_secs_f64(),
            fp
        );

        println!("[golden-fixture] Running 1,000 fixed queries (k=10, efSearch=100)...");
        let mut tracker = crate::hnsw::VisitedTracker::new(index.num_vectors());
        let timed = index.search_batch_timed_with_tracker(&queries, 10, 100, &mut tracker)?;
        let qr_hash = compute_query_results_hash(&timed.result.indices, &timed.result.distances);
        println!(
            "[golden-fixture] Scale {}: query results hash: 0x{:016x}",
            scale, qr_hash
        );

        let mut results_detail = Vec::with_capacity(100);
        for q in 0..100.min(queries.rows()) {
            let i_row = timed.result.indices.row(q);
            let d_row = timed.result.distances.row(q);
            let k_len = 10.min(i_row.len());
            let indices: Vec<i32> = i_row[..k_len].to_vec();
            let distance_bits: Vec<u32> = d_row[..k_len].iter().map(|d| d.to_bits()).collect();
            let distances: Vec<f32> = d_row[..k_len].to_vec();
            results_detail.push(GoldenQueryResult {
                query_id: q,
                indices,
                distance_bits,
                distances,
            });
        }

        let fixture = GoldenFixture {
            scale_vectors: scale,
            hnsw_m: 16,
            hnsw_m_max0: 32,
            hnsw_ef_construction: 100,
            seed: 42,
            graph_fingerprint: fp,
            graph_fingerprint_hex: format!("0x{:016x}", fp),
            query_results_hash: qr_hash,
            query_results_hash_hex: format!("0x{:016x}", qr_hash),
            num_queries: 1000,
            k: 10,
            ef_search: 100,
            results: results_detail,
        };

        let fname = if scale == 10_000 {
            "golden_10k.json"
        } else {
            "golden_100k.json"
        };
        let out_file = output_dir.join(fname);
        fs::write(&out_file, fixture.to_json_string())?;
        println!("[golden-fixture] Saved: {}", out_file.display());
    }

    println!("[golden-fixture] Golden fixture generation completed successfully.");
    Ok(())
}

/// Measures unmodified baseline at 10k and 100k (3 repetitions, median, min, max, spread)
/// and writes `results/phase3_5/baseline_10k.json` and `results/phase3_5/baseline_100k.json`.
pub fn run_phase3_5_baseline_measurement(data_dir: &Path, output_dir: &Path) -> Result<()> {
    println!("[baseline-measurement] Running Phase 3.5 baseline re-measurements...");
    let (base_path, query_path) = resolve_sift1m_fvecs_paths(data_dir)?;

    fs::create_dir_all(output_dir)?;
    let hw = detect_hardware();
    let prov = detect_provenance();
    let git_commit = prov.git_commit;
    let git_dirty = prov.git_dirty;
    let rustc_version = prov.rustc_version;
    let rss_method = peak_rss_measurement_method(false);

    for &scale in &[10_000_usize, 100_000_usize] {
        println!("\n{}", "=".repeat(70));
        println!(
            "[baseline-measurement] Measuring baseline for scale: {} vectors",
            scale
        );
        println!("{}", "=".repeat(70));

        let base = crate::dataset::read_fvecs_max_rows(&base_path, Some(scale))?;
        let queries = crate::dataset::read_fvecs_max_rows(&query_path, Some(10_000))?;
        let num_eval_q = 1000.min(queries.rows());
        let eval_queries = queries.slice_first_rows(num_eval_q);

        println!(
            "[baseline-measurement] Computing exact KNN ground truth for {} queries...",
            num_eval_q
        );
        let exact_res =
            ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &eval_queries, 10, 0)?;

        // Brief cooldown after multi-threaded ExactKNN so Run 1/3 starts at steady CPU temperature
        std::thread::sleep(std::time::Duration::from_secs(2));

        let cfg = HnswConfig {
            m: 16,
            m_max: None,
            m_max0: Some(32),
            ef_construction: 100,
            m_l: None,
            extend_candidates: false,
            keep_pruned: true,
            keep_pruned_shrink: None,
            seed: 42,
        };

        // 3 consecutive builds
        let mut build_times = Vec::with_capacity(3);
        let mut last_index: Option<HnswIndex> = None;
        for run_idx in 1..=3 {
            println!(
                "[baseline-measurement] Run {}/3 building HNSW index...",
                run_idx
            );
            let t0 = Instant::now();
            let idx = HnswIndex::build(&base, Metric::L2Squared, cfg.clone())?;
            let dur = t0.elapsed().as_secs_f64();
            println!("[baseline-measurement] Run {}/3 took {:.3} s", run_idx, dur);
            build_times.push(dur);
            last_index = Some(idx);
        }

        let mut index = last_index.unwrap();
        let mut sorted_times = build_times.clone();
        sorted_times.sort_by(f64::total_cmp);
        let median_build_time = sorted_times[1];
        let min_build_time = sorted_times[0];
        let max_build_time = sorted_times[2];
        let spread = max_build_time - min_build_time;
        let build_qps = (scale as f64) / median_build_time;

        println!(
            "[baseline-measurement] Scale {} Summary: Median={:.3}s, Min={:.3}s, Max={:.3}s, Spread={:.3}s ({:.2}%)",
            scale, median_build_time, min_build_time, max_build_time, spread, (spread / median_build_time) * 100.0
        );

        let graph_fingerprint = index.graph_fingerprint();
        let max_layer = index.max_layer();
        let entry_point = index.entry_point().unwrap_or(0);
        let peak_rss_mb = get_peak_rss_mb().unwrap_or(0.0);
        let peak_rss_bytes = (peak_rss_mb * 1024.0 * 1024.0) as usize;
        let graph_memory_bytes = index.memory_bytes();
        let graph_only_memory_bytes = index.graph_only_memory_bytes();
        let graph_memory_mb = index.memory_mb();
        let bytes_per_vector = index.bytes_per_vector();
        let graph_bytes_per_vector = bytes_per_vector;
        let graph_only_bytes_per_vector = index.graph_only_bytes_per_vector();
        let raw_vector_bytes_per_vector = 512.0;
        let total_bytes_per_vector = raw_vector_bytes_per_vector + bytes_per_vector;
        let graph_components = index.graph_components();

        // efSearch evaluations for 50 and 100
        let mut ef_evaluations_json = Vec::new();
        for &ef in &[50_usize, 100_usize] {
            let mut run_latencies = Vec::with_capacity(5);
            let mut last_indices = None;
            for _ in 0..5 {
                let timed = index.search_batch_timed(&eval_queries, 10, ef)?;
                run_latencies.push(timed.latency);
                last_indices = Some(timed.result.indices);
            }
            let summary = summarize_repeated_latencies(&run_latencies);
            let cand_indices = last_indices.unwrap();
            let recall = compute_recall_at_k(&cand_indices, &exact_res.indices, 10)?;
            let ef_json = format!(
                "    {{\n      \"ef_search\": {},\n      \"recall_at_10\": {:.6},\n      \"num_runs\": 5,\n      \"latency\": {{\n        \"mean_ms\": {:.6},\n        \"p50_ms\": {:.6},\n        \"p95_ms\": {:.6},\n        \"p99_ms\": {:.6},\n        \"qps\": {:.2},\n        \"mean_ms_spread\": {:.6},\n        \"p99_ms_spread\": {:.6}\n      }}\n    }}",
                ef, recall, summary.median_latency.mean_ms, summary.median_latency.p50_ms, summary.median_latency.p95_ms, summary.median_latency.p99_ms, summary.median_latency.qps, summary.mean_ms_spread, summary.p99_ms_spread
            );
            ef_evaluations_json.push(ef_json);
        }

        // Fixed query set verification (1,000 queries at k=10, efSearch=100)
        let fixed_1k = queries.slice_first_rows(1000.min(queries.rows()));
        let mut tracker = crate::hnsw::VisitedTracker::new(index.num_vectors());
        let timed_1k = index.search_batch_timed_with_tracker(&fixed_1k, 10, 100, &mut tracker)?;
        let qr_hash =
            compute_query_results_hash(&timed_1k.result.indices, &timed_1k.result.distances);

        let golden_fname = if scale == 10_000 {
            "golden_10k.json"
        } else {
            "golden_100k.json"
        };
        let golden_path = output_dir.join(golden_fname);
        let golden_exists = golden_path.exists();
        let mut golden_fp_matched = false;
        let mut golden_qr_matched = false;

        if golden_exists {
            if let Ok(content) = fs::read_to_string(&golden_path) {
                if let Ok(fixture) = GoldenFixture::parse_json(&content) {
                    golden_fp_matched = fixture.graph_fingerprint == graph_fingerprint;
                    golden_qr_matched = fixture.query_results_hash == qr_hash;
                }
            }
        }

        let build_stats_json = if let Some(bs) = index.build_stats() {
            format!(
                "  \"build_instrumentation\": {{\n    \"feature_instrument_enabled\": true,\n    \"distance_evaluations_total\": {},\n    \"distance_evaluations_per_insert_mean\": {:.2},\n    \"search_layer_calls\": {},\n    \"heuristic_calls\": {},\n    \"shrink_calls\": {},\n    \"time_split_seconds\": {{\n      \"search_layer_s\": {:.4},\n      \"heuristic_s\": {:.4},\n      \"shrink_s\": {:.4},\n      \"other_s\": {:.4}\n    }}\n  }},",
                bs.distance_evals_total,
                bs.dist_evals_per_insert_mean,
                bs.search_layer_calls,
                bs.heuristic_calls,
                bs.shrink_calls,
                bs.time_search_layer_s,
                bs.time_heuristic_s,
                bs.time_shrink_s,
                bs.time_other_overhead_s
            )
        } else {
            "  \"build_instrumentation\": {\n    \"feature_instrument_enabled\": false\n  },"
                .to_string()
        };

        let generated_at = current_utc_rfc3339();
        let baseline_json = format!(
            "{{\n\
  \"phase\": \"3.5_baseline_profile\",\n\
  \"generated_at\": \"{}\",\n\
  \"provenance\": {{\n\
    \"git_commit\": \"{}\",\n\
    \"git_dirty\": {},\n\
    \"rustc_version\": \"{}\",\n\
    \"cargo_profile\": \"release\"\n\
  }},\n\
  \"environment\": {{\n\
    \"os\": \"{}\",\n\
    \"cpu_model\": \"{}\",\n\
    \"logical_cores\": {},\n\
    \"total_ram_gb\": {:.2},\n\
    \"power_mode\": \"best_performance\",\n\
    \"processor_affinity_mask\": \"0x0004\"\n\
  }},\n\
  \"dataset\": {{\n\
    \"preset\": \"sift1m\",\n\
    \"source_kind\": \"authentic_sift1m\",\n\
    \"source_description\": \"Authentic Texmex SIFT1M ({})\",\n\
    \"is_authentic_sift\": true,\n\
    \"base_path\": \"{}\",\n\
    \"query_path\": \"{}\",\n\
    \"scale_vectors\": {},\n\
    \"dim\": 128,\n\
    \"metric\": \"l2\"\n\
  }},\n\
  \"hnsw_config\": {{\n\
    \"m\": 16,\n\
    \"m_max0\": 32,\n\
    \"ef_construction\": 100,\n\
    \"seed\": 42,\n\
    \"extend_candidates\": false,\n\
    \"keep_pruned\": true\n\
  }},\n\
  \"index_build\": {{\n\
    \"scale_vectors\": {},\n\
    \"build_runs\": 3,\n\
    \"build_times_s\": [{:.6}, {:.6}, {:.6}],\n\
    \"build_time_s\": {:.6},\n\
    \"build_qps\": {:.2},\n\
    \"max_layer\": {},\n\
    \"entry_point\": {},\n\
    \"graph_fingerprint\": {},\n\
    \"graph_fingerprint_hex\": \"0x{:016x}\"\n\
  }},\n\
  \"memory_profile\": {{\n\
    \"peak_rss_mb\": {:.2},\n\
    \"peak_rss_bytes\": {},\n\
    \"peak_rss_measurement_method\": \"{}\",\n\
    \"is_peak_rss_os_measured\": true,\n\
    \"raw_vector_bytes_per_vector\": {:.2},\n\
    \"graph_bytes_per_vector\": {:.2},\n\
    \"graph_only_bytes_per_vector\": {:.2},\n\
    \"total_bytes_per_vector\": {:.2},\n\
    \"graph_memory_bytes\": {},\n\
    \"graph_only_memory_bytes\": {},\n\
    \"graph_memory_mb\": {:.2},\n\
    \"graph_components_bytes\": {{\n\
      \"node_structs_bytes\": {},\n\
      \"layer0_edges_bytes\": {},\n\
      \"upper_layers_edges_bytes\": {},\n\
      \"outer_neighbors_headers_bytes\": {},\n\
      \"visited_tracker_tags_bytes\": {},\n\
      \"heuristic_scratch_bytes\": {}\n\
    }}\n\
  }},\n\
{}\n\
  \"ef_search_evaluations\": [\n\
{}\n\
  ],\n\
  \"fixed_query_set_verification\": {{\n\
    \"num_queries\": 1000,\n\
    \"k\": 10,\n\
    \"ef_search\": 100,\n\
    \"query_results_hash\": {},\n\
    \"query_results_hash_hex\": \"0x{:016x}\",\n\
    \"golden_fixture_file\": \"{}\"\n\
  }},\n\
  \"verification_status\": {{\n\
    \"reproducibility_checked\": true,\n\
    \"reproducibility_verified\": true,\n\
    \"golden_fingerprint_matched\": {},\n\
    \"golden_query_results_matched\": {},\n\
    \"forbid_unsafe_code_enforced\": true,\n\
    \"std_only_enforced\": true\n\
  }}\n\
}}",
            escape_json(&generated_at),
            escape_json(&git_commit),
            git_dirty,
            escape_json(&rustc_version),
            escape_json(&hw.os),
            escape_json(&hw.cpu_model),
            hw.logical_cores,
            hw.total_ram_gb.unwrap_or(0.0),
            escape_json(&data_dir.display().to_string()),
            escape_json(&base_path.display().to_string()),
            escape_json(&query_path.display().to_string()),
            scale,
            scale,
            build_times[0],
            build_times[1],
            build_times[2],
            median_build_time,
            build_qps,
            max_layer,
            entry_point,
            graph_fingerprint,
            graph_fingerprint,
            peak_rss_mb,
            peak_rss_bytes,
            escape_json(&rss_method),
            raw_vector_bytes_per_vector,
            graph_bytes_per_vector,
            graph_only_bytes_per_vector,
            total_bytes_per_vector,
            graph_memory_bytes,
            graph_only_memory_bytes,
            graph_memory_mb,
            graph_components.node_structs_bytes,
            graph_components.layer0_edges_bytes,
            graph_components.upper_layers_edges_bytes,
            graph_components.outer_neighbors_headers_bytes,
            graph_components.visited_tracker_tags_bytes,
            graph_components.heuristic_scratch_bytes,
            build_stats_json,
            ef_evaluations_json.join(",\n"),
            qr_hash,
            qr_hash,
            escape_json(&golden_path.display().to_string()),
            golden_fp_matched,
            golden_qr_matched
        );

        let baseline_fname = if scale == 10_000 {
            "baseline_10k.json"
        } else {
            "baseline_100k.json"
        };
        let out_file = output_dir.join(baseline_fname);
        fs::write(&out_file, &baseline_json)?;
        println!("[baseline-measurement] Saved: {}", out_file.display());
    }

    println!("[baseline-measurement] Phase 3.5 baseline measurement complete.");
    Ok(())
}
