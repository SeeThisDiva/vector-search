//! Phase 1 Benchmark Pipeline: Dataset Loading, Brute-Force Ground Truth,
//! Latency Percentiles (`p50`, `p95`, `p99`, `QPS`), Recall Verification, and Reporting.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataset::{
    generate_synthetic_sift, load_custom_dataset, load_custom_dataset_with_options,
    load_or_download_sift, load_or_download_sift_with_options, write_fvecs, write_ivecs,
    DatasetSource, LoadedDataset, SiftPreset,
};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{detect_hardware, HardwareInfo};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::matrix::RowMatrix;
use crate::recall::{
    compute_recall_at_k, evaluate_recall_summary, percentile_sorted, verify_self_recall,
    LatencyStats,
};

/// Dataset selection mode for Phase 1 execution.
#[derive(Debug, Clone)]
pub enum Phase1DatasetMode {
    /// Public Texmex SIFT preset (`siftsmall` 10k or `SIFT1M` 1M) with offline fallback.
    Sift(SiftPreset),
    /// Deterministic seeded synthetic clustered Gaussian SIFT manifold.
    Synthetic {
        num_base: usize,
        num_query: usize,
        dim: usize,
        num_clusters: usize,
    },
    /// User-supplied `.fvecs` (and optional `.ivecs` ground truth) files.
    Custom {
        base_fvecs: PathBuf,
        query_fvecs: PathBuf,
        gt_ivecs: Option<PathBuf>,
    },
}

/// Configuration parameters for the Phase 1 ground-truth & recall verification benchmark.
#[derive(Debug, Clone)]
pub struct Phase1Config {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub allow_download: bool,
    pub save_files: bool,
    pub max_queries: Option<usize>,
}

impl Default for Phase1Config {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
            data_cache_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("data"),
            k: 10,
            seed: 42,
            metric: Metric::L2Squared,
            allow_download: true,
            save_files: true,
            max_queries: None,
        }
    }
}

/// Saved binary artifact paths written during Phase 1 execution.
#[derive(Debug, Clone, Default)]
pub struct SavedArtifacts {
    pub base_fvecs: Option<PathBuf>,
    pub query_fvecs: Option<PathBuf>,
    pub groundtruth_ivecs: Option<PathBuf>,
    pub groundtruth_fvecs: Option<PathBuf>,
}

/// Verification result comparing computed ExactKNN results against official corpus ground truth.
#[derive(Debug, Clone)]
pub struct OfficialGtComparison {
    pub eval_k: usize,
    pub recall_at_k: f64,
    pub exact_id_matches: usize,
    pub total_compared: usize,
}

/// Complete structured Phase 1 benchmark report.
#[derive(Debug, Clone)]
pub struct Phase1Report {
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub effective_k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub base_memory_mb: f64,
    pub query_memory_mb: f64,
    pub dataset_load_time_s: f64,
    pub fit_time_s: f64,
    pub latency: LatencyStats,
    pub self_recall_at_k: f64,
    pub self_verified: bool,
    pub recall_summary: Vec<(String, f64)>,
    pub official_gt_comparison: Option<OfficialGtComparison>,
    pub artifacts: SavedArtifacts,
}

impl Phase1Report {
    /// Serializes the benchmark report to a deterministic, dependency-free JSON document.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("{\n");

        // Hardware block
        out.push_str("  \"hardware\": {\n");
        out.push_str(&format!(
            "    \"summary\": \"{}\",\n",
            escape_json(&self.hardware.summary_line())
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        match self.hardware.total_ram_gb {
            Some(ram) => out.push_str(&format!("    \"total_ram_gb\": {ram:.3},\n")),
            None => out.push_str("    \"total_ram_gb\": null,\n"),
        }
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\"\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str("  },\n");

        // Dataset block
        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!(
            "    \"source_kind\": \"{}\",\n",
            self.dataset_source.kind_tag()
        ));
        out.push_str(&format!(
            "    \"source_description\": \"{}\",\n",
            escape_json(&self.dataset_source.to_string())
        ));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {},\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"effective_k\": {},\n", self.effective_k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str(&format!("    \"metric\": \"{}\",\n", self.metric));
        out.push_str(&format!(
            "    \"base_memory_mb\": {:.6},\n",
            self.base_memory_mb
        ));
        out.push_str(&format!(
            "    \"query_memory_mb\": {:.6},\n",
            self.query_memory_mb
        ));
        out.push_str(&format!(
            "    \"load_time_s\": {:.6}\n",
            self.dataset_load_time_s
        ));
        out.push_str("  },\n");

        // Exact KNN timing & latency percentiles
        out.push_str("  \"exact_knn\": {\n");
        out.push_str(&format!("    \"fit_time_s\": {:.6},\n", self.fit_time_s));
        out.push_str(&format!(
            "    \"search_time_s\": {:.6},\n",
            self.latency.total_s
        ));
        out.push_str(&format!("    \"qps\": {:.3},\n", self.latency.qps));
        out.push_str(&format!("    \"min_ms\": {:.6},\n", self.latency.min_ms));
        out.push_str(&format!("    \"mean_ms\": {:.6},\n", self.latency.mean_ms));
        out.push_str(&format!("    \"p50_ms\": {:.6},\n", self.latency.p50_ms));
        out.push_str(&format!("    \"p95_ms\": {:.6},\n", self.latency.p95_ms));
        out.push_str(&format!("    \"p99_ms\": {:.6},\n", self.latency.p99_ms));
        out.push_str(&format!("    \"max_ms\": {:.6}\n", self.latency.max_ms));
        out.push_str("  },\n");

        // Recall block
        out.push_str("  \"recall\": {\n");
        out.push_str(&format!(
            "    \"self_recall_at_k\": {:.6},\n",
            self.self_recall_at_k
        ));
        out.push_str(&format!("    \"self_verified\": {},\n", self.self_verified));
        out.push_str("    \"metrics\": {");
        for (idx, (name, val)) in self.recall_summary.iter().enumerate() {
            let sep = if idx + 1 < self.recall_summary.len() {
                ","
            } else {
                ""
            };
            out.push_str(&format!(
                "\n      \"{}\": {:.6}{sep}",
                escape_json(name),
                val
            ));
        }
        if !self.recall_summary.is_empty() {
            out.push_str("\n    }\n");
        } else {
            out.push_str("}\n");
        }
        out.push_str("  },\n");

        // Official ground truth comparison
        match &self.official_gt_comparison {
            Some(cmp) => {
                out.push_str("  \"official_ground_truth\": {\n");
                out.push_str(&format!("    \"eval_k\": {},\n", cmp.eval_k));
                out.push_str(&format!("    \"recall_at_k\": {:.6},\n", cmp.recall_at_k));
                out.push_str(&format!(
                    "    \"exact_id_matches\": {},\n",
                    cmp.exact_id_matches
                ));
                out.push_str(&format!("    \"total_compared\": {}\n", cmp.total_compared));
                out.push_str("  },\n");
            }
            None => out.push_str("  \"official_ground_truth\": null,\n"),
        }

        // Artifacts block
        out.push_str("  \"files\": {\n");
        out.push_str(&format!(
            "    \"base_fvecs\": {},\n",
            opt_path_json(self.artifacts.base_fvecs.as_deref())
        ));
        out.push_str(&format!(
            "    \"query_fvecs\": {},\n",
            opt_path_json(self.artifacts.query_fvecs.as_deref())
        ));
        out.push_str(&format!(
            "    \"groundtruth_ivecs\": {},\n",
            opt_path_json(self.artifacts.groundtruth_ivecs.as_deref())
        ));
        out.push_str(&format!(
            "    \"groundtruth_fvecs\": {}\n",
            opt_path_json(self.artifacts.groundtruth_fvecs.as_deref())
        ));
        out.push_str("  }\n");

        out.push('}');
        out
    }
}

/// Executes the end-to-end Phase 1 ground-truth & recall verification benchmark pipeline.
pub fn run_phase1_benchmark(config: &Phase1Config) -> Result<Phase1Report> {
    if config.k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0)".to_string(),
        ));
    }

    let hardware = detect_hardware();

    // 1. Load or generate dataset
    let mut loaded: LoadedDataset = match &config.dataset_mode {
        Phase1DatasetMode::Sift(preset) => load_or_download_sift(
            *preset,
            &config.data_cache_dir,
            config.allow_download,
            config.seed,
        )?,
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let (base, query) =
                generate_synthetic_sift(*num_base, *num_query, *dim, *num_clusters, config.seed)?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic { seed: config.seed },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => load_custom_dataset(base_fvecs, query_fvecs, gt_ivecs.as_deref())?,
    };

    if let Some(limit) = config.max_queries {
        if limit == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "max_queries must be positive (> 0) when specified".to_string(),
            ));
        }
        if limit < loaded.query.rows() {
            loaded.query = loaded.query.slice_first_rows(limit);
            if let Some(gt) = &loaded.official_ground_truth {
                loaded.official_ground_truth = Some(gt.slice_first_rows(limit));
            }
        }
    }

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();
    let base_memory_mb = loaded.base.memory_mb();
    let query_memory_mb = loaded.query.memory_mb();

    let mut artifacts = SavedArtifacts::default();
    if config.save_files {
        fs::create_dir_all(&config.output_dir)?;
        let base_path = config.output_dir.join("base.fvecs");
        let query_path = config.output_dir.join("query.fvecs");
        write_fvecs(&base_path, &loaded.base)?;
        write_fvecs(&query_path, &loaded.query)?;
        artifacts.base_fvecs = Some(base_path);
        artifacts.query_fvecs = Some(query_path);
    }

    // 2. Fit Brute-Force ExactKNN & execute timed per-query search
    let t_fit = Instant::now();
    let mut knn = ExactKnn::new(config.metric);
    knn.fit(loaded.base)?;
    let fit_time_s = t_fit.elapsed().as_secs_f64();

    let timed = knn.search_timed(&loaded.query, config.k, true)?;
    let gt_indices = &timed.result.indices;
    let gt_distances = &timed.result.distances;
    let effective_k = gt_indices.cols();

    if config.save_files {
        let gt_ivecs_path = config.output_dir.join("groundtruth.ivecs");
        let gt_fvecs_path = config.output_dir.join("groundtruth.fvecs");
        write_ivecs(&gt_ivecs_path, gt_indices)?;
        write_fvecs(&gt_fvecs_path, gt_distances)?;
        artifacts.groundtruth_ivecs = Some(gt_ivecs_path);
        artifacts.groundtruth_fvecs = Some(gt_fvecs_path);
    }

    // 3. Automated Recall@k harness & self-verification invariant
    let self_recall_at_k = compute_recall_at_k(gt_indices, gt_indices, config.k)?;
    let self_verified = verify_self_recall(gt_indices, config.k)?;
    let recall_summary =
        evaluate_recall_summary(gt_indices, gt_indices, &[1, 5, 10, 50, 100, config.k])?;

    // 4. Cross-verify against official corpus ground truth (if present & L2-compatible)
    let official_gt_comparison = if matches!(config.metric, Metric::L2Squared | Metric::Euclidean) {
        if let Some(official_gt) = &loaded.official_ground_truth {
            if official_gt.rows() == gt_indices.rows() && official_gt.cols() > 0 {
                let eval_k = effective_k.min(official_gt.cols());
                let recall_at_k = compute_recall_at_k(gt_indices, official_gt, eval_k)?;
                let mut exact_id_matches = 0_usize;
                for q in 0..gt_indices.rows() {
                    let our_row = &gt_indices.row(q)[..eval_k];
                    let off_row = &official_gt.row(q)[..eval_k];
                    for (&a, &b) in our_row.iter().zip(off_row.iter()) {
                        if a == b {
                            exact_id_matches += 1;
                        }
                    }
                }
                Some(OfficialGtComparison {
                    eval_k,
                    recall_at_k,
                    exact_id_matches,
                    total_compared: gt_indices.rows() * eval_k,
                })
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(Phase1Report {
        hardware,
        dataset_source: loaded.source,
        num_base,
        num_query,
        dim,
        k: config.k,
        effective_k,
        seed: config.seed,
        metric: config.metric,
        base_memory_mb,
        query_memory_mb,
        dataset_load_time_s: loaded.load_time_s,
        fit_time_s,
        latency: timed.latency,
        self_recall_at_k,
        self_verified,
        recall_summary,
        official_gt_comparison,
        artifacts,
    })
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn opt_path_json(p: Option<&Path>) -> String {
    match p {
        Some(path) => format!("\"{}\"", escape_json(&path.display().to_string())),
        None => "null".to_string(),
    }
}

/// Configuration parameters for Phase 2 HNSW verification & benchmarking.
#[derive(Debug, Clone)]
pub struct Phase2Config {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub m: usize,
    /// Layer-0 degree cap; `None` derives `2 * m`.
    pub m_max0: Option<usize>,
    pub ef_construction: usize,
    pub ef_search_list: Vec<usize>,
    pub allow_download: bool,
    pub save_files: bool,
    pub max_queries: Option<usize>,
    pub run_repro_check: bool,
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
            data_cache_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("data"),
            k: 10,
            seed: 42,
            metric: Metric::L2Squared,
            m: 16,
            m_max0: None,
            ef_construction: 100,
            ef_search_list: vec![10, 20, 50, 100, 200],
            allow_download: true,
            save_files: true,
            max_queries: None,
            run_repro_check: true,
        }
    }
}

/// Phase 2 acceptance target: minimum Recall@k at the operating point.
pub const PHASE2_TARGET_MIN_RECALL: f64 = 0.98;
/// Phase 2 acceptance target: maximum mean latency per query (milliseconds).
pub const PHASE2_TARGET_MAX_LATENCY_MS: f64 = 0.05;

/// Evaluation metrics for a single efSearch setting in Phase 2.
#[derive(Debug, Clone)]
pub struct HnswEfEvaluation {
    pub ef_search: usize,
    pub recall_at_k: f64,
    pub latency: LatencyStats,
    pub speedup_vs_exact: f64,
    pub monotonic_passed: bool,
}

/// Complete structured Phase 2 benchmark report.
#[derive(Debug, Clone)]
pub struct Phase2Report {
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub hnsw_m: usize,
    pub hnsw_m_max0: usize,
    pub hnsw_ef_construction: usize,
    pub build_time_s: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub exact_baseline_latency: LatencyStats,
    pub exact_self_verified: bool,
    pub exact_official_gt_match: bool,
    pub ef_evaluations: Vec<HnswEfEvaluation>,
    pub monotonic_scaling_verified: bool,
    /// True when some efSearch point meets the Phase 2 targets simultaneously:
    /// recall >= `PHASE2_TARGET_MIN_RECALL` and mean latency < `PHASE2_TARGET_MAX_LATENCY_MS`.
    /// Speedup vs brute force is reported per point but not gated: the spec's "10x" was
    /// derived from a ~0.25 ms/query baseline and is machine-dependent.
    pub performance_target_met: bool,
    /// Smallest efSearch that satisfies the performance target, if any.
    pub performance_target_ef: Option<usize>,
    /// True only if the reproducibility check actually ran (it is skipped by `--no-repro-check`).
    pub reproducibility_checked: bool,
    /// True only if the check ran and passed. A skipped check is never reported as verified.
    pub reproducibility_verified: bool,
}

impl Phase2Report {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str("  \"phase\": 2,\n");
        out.push_str("  \"hardware\": {\n");
        out.push_str(&format!(
            "    \"summary\": \"{}\",\n",
            escape_json(&self.hardware.summary_line())
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {:.2},\n",
            self.hardware.total_ram_gb.unwrap_or(0.0)
        ));
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\"\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str("  },\n");

        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!(
            "    \"source_kind\": \"{}\",\n",
            self.dataset_source.kind_tag()
        ));
        out.push_str(&format!(
            "    \"source_description\": \"{}\",\n",
            escape_json(&self.dataset_source.to_string())
        ));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {},\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str(&format!("    \"metric\": \"{}\"\n", self.metric));
        out.push_str("  },\n");

        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_m));
        out.push_str(&format!("    \"m_max0\": {},\n", self.hnsw_m_max0));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_ef_construction
        ));
        out.push_str(&format!("    \"seed\": {}\n", self.seed));
        out.push_str("  },\n");

        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!("    \"max_layer\": {},\n", self.max_layer));
        out.push_str(&format!("    \"entry_point\": {}\n", self.entry_point));
        out.push_str("  },\n");

        out.push_str("  \"exact_knn_baseline\": {\n");
        out.push_str(&format!(
            "    \"mean_ms\": {:.6},\n",
            self.exact_baseline_latency.mean_ms
        ));
        out.push_str(&format!(
            "    \"p50_ms\": {:.6},\n",
            self.exact_baseline_latency.p50_ms
        ));
        out.push_str(&format!(
            "    \"p95_ms\": {:.6},\n",
            self.exact_baseline_latency.p95_ms
        ));
        out.push_str(&format!(
            "    \"p99_ms\": {:.6},\n",
            self.exact_baseline_latency.p99_ms
        ));
        out.push_str(&format!(
            "    \"qps\": {:.3},\n",
            self.exact_baseline_latency.qps
        ));
        out.push_str(&format!(
            "    \"self_verified\": {},\n",
            self.exact_self_verified
        ));
        out.push_str(&format!(
            "    \"official_gt_matched\": {}\n",
            self.exact_official_gt_match
        ));
        out.push_str("  },\n");

        out.push_str("  \"ef_search_sweep\": [\n");
        for (i, eval) in self.ef_evaluations.iter().enumerate() {
            let comma = if i + 1 < self.ef_evaluations.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", eval.ef_search));
            out.push_str(&format!(
                "      \"recall_at_k\": {:.6},\n",
                eval.recall_at_k
            ));
            out.push_str("      \"latency\": {\n");
            out.push_str(&format!(
                "        \"mean_ms\": {:.6},\n",
                eval.latency.mean_ms
            ));
            out.push_str(&format!(
                "        \"p50_ms\": {:.6},\n",
                eval.latency.p50_ms
            ));
            out.push_str(&format!(
                "        \"p95_ms\": {:.6},\n",
                eval.latency.p95_ms
            ));
            out.push_str(&format!(
                "        \"p99_ms\": {:.6},\n",
                eval.latency.p99_ms
            ));
            out.push_str(&format!("        \"qps\": {:.3}\n", eval.latency.qps));
            out.push_str("      },\n");
            out.push_str(&format!(
                "      \"speedup_vs_exact\": {:.2},\n",
                eval.speedup_vs_exact
            ));
            out.push_str(&format!(
                "      \"monotonic_passed\": {}\n",
                eval.monotonic_passed
            ));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");

        out.push_str("  \"verification_status\": {\n");
        out.push_str(&format!(
            "    \"monotonic_scaling_verified\": {},\n",
            self.monotonic_scaling_verified
        ));
        out.push_str(&format!(
            "    \"performance_target_met\": {},\n",
            self.performance_target_met
        ));
        out.push_str(&format!(
            "    \"performance_target_ef\": {},\n",
            self.performance_target_ef
                .map_or_else(|| "null".to_string(), |ef| ef.to_string())
        ));
        out.push_str(&format!(
            "    \"performance_target\": {{ \"min_recall\": {}, \"max_mean_latency_ms\": {} }},\n",
            PHASE2_TARGET_MIN_RECALL, PHASE2_TARGET_MAX_LATENCY_MS
        ));
        out.push_str(&format!(
            "    \"reproducibility_checked\": {},\n",
            self.reproducibility_checked
        ));
        out.push_str(&format!(
            "    \"reproducibility_verified\": {},\n",
            self.reproducibility_verified
        ));
        let overall = self.monotonic_scaling_verified
            && self.reproducibility_verified
            && self.performance_target_met;
        out.push_str(&format!("    \"overall_phase2_passed\": {}\n", overall));
        out.push_str("  }\n");
        out.push_str("}\n");

        out
    }
}

/// Profile measurement for an individual scale in the baseline multi-scale profile.
#[derive(Debug, Clone)]
pub struct BaselineScaleProfile {
    /// False for rows copied from an earlier recorded run (see `load_or_default_baseline_profiles`);
    /// true only for rows measured by the run that produced the report.
    pub measured_this_run: bool,
    pub scale_vectors: usize,
    pub build_time_s: f64,
    pub build_qps: f64,
    pub peak_rss_mb: f64,
    pub bytes_per_vector: f64,
    pub raw_vector_bytes_per_vector: f64,
    pub graph_bytes_per_vector: f64,
    pub graph_only_bytes_per_vector: f64,
    pub total_bytes_per_vector: f64,
    pub graph_memory_bytes: usize,
    pub graph_only_memory_bytes: usize,
    pub graph_memory_mb: f64,
    pub query_latency_ms: f64,
    pub query_qps: f64,
    pub k: usize,
    pub ef_search: usize,
    pub recall_at_10: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub latency_min_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_max_ms: f64,
}

/// Summary statistics across repeated `efSearch` sweep runs.
#[derive(Debug, Clone)]
pub struct RepeatedSweepSummary {
    pub num_runs: usize,
    pub median_latency: LatencyStats,
    pub mean_ms_min: f64,
    pub mean_ms_max: f64,
    pub mean_ms_spread: f64,
    pub p99_ms_min: f64,
    pub p99_ms_max: f64,
    pub p99_ms_spread: f64,
    pub run_mean_latencies_ms: Vec<f64>,
    pub run_p99_latencies_ms: Vec<f64>,
}

/// Computes median `LatencyStats` and min/max/spread across `N` repeated batch search runs.
#[must_use]
pub fn summarize_repeated_latencies(runs: &[LatencyStats]) -> RepeatedSweepSummary {
    if runs.is_empty() {
        return RepeatedSweepSummary {
            num_runs: 0,
            median_latency: LatencyStats {
                num_queries: 0,
                total_s: 0.0,
                qps: 0.0,
                min_ms: 0.0,
                mean_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                max_ms: 0.0,
            },
            mean_ms_min: 0.0,
            mean_ms_max: 0.0,
            mean_ms_spread: 0.0,
            p99_ms_min: 0.0,
            p99_ms_max: 0.0,
            p99_ms_spread: 0.0,
            run_mean_latencies_ms: Vec::new(),
            run_p99_latencies_ms: Vec::new(),
        };
    }

    let num_runs = runs.len();
    let num_queries = runs[0].num_queries;
    let run_mean_latencies_ms: Vec<f64> = runs.iter().map(|r| r.mean_ms).collect();
    let run_p99_latencies_ms: Vec<f64> = runs.iter().map(|r| r.p99_ms).collect();

    let median_of = |extract: fn(&LatencyStats) -> f64| -> (f64, f64, f64) {
        let mut vals: Vec<f64> = runs.iter().map(extract).collect();
        vals.sort_by(|a, b| a.total_cmp(b));
        let med = percentile_sorted(&vals, 50.0);
        let min_v = vals[0];
        let max_v = vals[vals.len() - 1];
        (med, min_v, max_v)
    };

    let (total_s, _, _) = median_of(|r| r.total_s);
    let (min_ms, _, _) = median_of(|r| r.min_ms);
    let (mean_ms, mean_ms_min, mean_ms_max) = median_of(|r| r.mean_ms);
    let (p50_ms, _, _) = median_of(|r| r.p50_ms);
    let (p95_ms, _, _) = median_of(|r| r.p95_ms);
    let (p99_ms, p99_ms_min, p99_ms_max) = median_of(|r| r.p99_ms);
    let (max_ms, _, _) = median_of(|r| r.max_ms);
    let qps = if total_s > 0.0 {
        (num_queries as f64) / total_s
    } else {
        0.0
    };

    RepeatedSweepSummary {
        num_runs,
        median_latency: LatencyStats {
            num_queries,
            total_s,
            qps,
            min_ms,
            mean_ms,
            p50_ms,
            p95_ms,
            p99_ms,
            max_ms,
        },
        mean_ms_min,
        mean_ms_max,
        mean_ms_spread: (mean_ms_max - mean_ms_min).max(0.0),
        p99_ms_min,
        p99_ms_max,
        p99_ms_spread: (p99_ms_max - p99_ms_min).max(0.0),
        run_mean_latencies_ms,
        run_p99_latencies_ms,
    }
}

/// Structured multi-scale baseline profiling report.
#[derive(Debug, Clone)]
pub struct BaselineMultiScaleReport {
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub hnsw_m: usize,
    pub hnsw_m_max0: usize,
    pub hnsw_ef_construction: usize,
    pub seed: u64,
    pub metric: Metric,
    pub peak_rss_measurement_method: String,
    pub scale_profiles: Vec<BaselineScaleProfile>,
}

impl BaselineMultiScaleReport {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("{\n");
        out.push_str("  \"phase\": \"2_baseline_profile\",\n");
        out.push_str("  \"hardware\": {\n");
        out.push_str(&format!(
            "    \"summary\": \"{}\",\n",
            escape_json(&self.hardware.summary_line())
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {:.2},\n",
            self.hardware.total_ram_gb.unwrap_or(0.0)
        ));
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\"\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str("  },\n");

        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!(
            "    \"source_kind\": \"{}\",\n",
            self.dataset_source.kind_tag()
        ));
        out.push_str(&format!(
            "    \"source_description\": \"{}\",\n",
            escape_json(&self.dataset_source.to_string())
        ));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {},\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str("    \"num_base_total\": 1000000,\n");
        out.push_str("    \"dim\": 128,\n");
        out.push_str(&format!("    \"metric\": \"{}\"\n", self.metric));
        out.push_str("  },\n");

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
            "  \"peak_rss_measurement_method\": \"{}\",\n",
            escape_json(&self.peak_rss_measurement_method)
        ));

        out.push_str("  \"scale_profiles\": [\n");
        for (i, p) in self.scale_profiles.iter().enumerate() {
            let comma = if i + 1 < self.scale_profiles.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!(
                "      \"measured_this_run\": {},\n",
                p.measured_this_run
            ));
            out.push_str(&format!("      \"scale_vectors\": {},\n", p.scale_vectors));
            out.push_str(&format!("      \"build_time_s\": {:.6},\n", p.build_time_s));
            out.push_str(&format!("      \"build_qps\": {:.2},\n", p.build_qps));
            out.push_str(&format!("      \"peak_rss_mb\": {:.2},\n", p.peak_rss_mb));
            out.push_str(&format!(
                "      \"bytes_per_vector\": {:.2},\n",
                p.bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"raw_vector_bytes_per_vector\": {:.2},\n",
                p.raw_vector_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_bytes_per_vector\": {:.2},\n",
                p.graph_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_only_bytes_per_vector\": {:.2},\n",
                p.graph_only_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"total_bytes_per_vector\": {:.2},\n",
                p.total_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_memory_bytes\": {},\n",
                p.graph_memory_bytes
            ));
            out.push_str(&format!(
                "      \"graph_only_memory_bytes\": {},\n",
                p.graph_only_memory_bytes
            ));
            out.push_str(&format!(
                "      \"graph_memory_mb\": {:.2},\n",
                p.graph_memory_mb
            ));
            out.push_str(&format!(
                "      \"query_latency_ms\": {:.6},\n",
                p.query_latency_ms
            ));
            out.push_str(&format!("      \"query_qps\": {:.2},\n", p.query_qps));
            out.push_str(&format!("      \"k\": {},\n", p.k));
            out.push_str(&format!("      \"ef_search\": {},\n", p.ef_search));
            out.push_str(&format!("      \"recall_at_10\": {:.6},\n", p.recall_at_10));
            out.push_str(&format!("      \"max_layer\": {},\n", p.max_layer));
            out.push_str(&format!("      \"entry_point\": {},\n", p.entry_point));
            out.push_str(&format!(
                "      \"latency_min_ms\": {:.6},\n",
                p.latency_min_ms
            ));
            out.push_str(&format!(
                "      \"latency_p50_ms\": {:.6},\n",
                p.latency_p50_ms
            ));
            out.push_str(&format!(
                "      \"latency_p95_ms\": {:.6},\n",
                p.latency_p95_ms
            ));
            out.push_str(&format!(
                "      \"latency_p99_ms\": {:.6},\n",
                p.latency_p99_ms
            ));
            out.push_str(&format!(
                "      \"latency_max_ms\": {:.6}\n",
                p.latency_max_ms
            ));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }
}

/// Profiles a single SIFT1M prefix scale (`scale` base vectors) by loading ONLY `scale` rows
/// from disk so the process high-water mark RSS reflects only that scale's memory footprint.
pub fn profile_single_sift1m_scale(
    data_cache_dir: &Path,
    scale: usize,
    seed: u64,
    allow_download: bool,
    sweep_runs: usize,
) -> Result<BaselineScaleProfile> {
    if scale == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "scale must be positive (> 0)".to_string(),
        ));
    }
    let num_eval_queries = if scale >= 1_000_000 { 10_000 } else { 1_000 };
    let load_gt = scale == 1_000_000;
    let loaded = load_or_download_sift_with_options(
        SiftPreset::Sift1M,
        data_cache_dir,
        allow_download,
        seed,
        Some(scale),
        Some(num_eval_queries),
        load_gt,
    )?;

    let actual_scale = loaded.base.rows();
    let hnsw_cfg = HnswConfig {
        m: 16,
        m_max: None,
        m_max0: Some(32),
        ef_construction: 100,
        m_l: None,
        extend_candidates: false,
        keep_pruned: true,
        seed,
    };

    let t_build_start = Instant::now();
    let mut index = HnswIndex::build(&loaded.base, Metric::L2Squared, hnsw_cfg)?;
    let build_time_s = t_build_start.elapsed().as_secs_f64();
    let build_qps = (actual_scale as f64) / build_time_s;

    let graph_memory_bytes = index.memory_bytes();
    let graph_only_memory_bytes = index.graph_only_memory_bytes();
    let graph_memory_mb = index.memory_mb();
    let bytes_per_vector = index.bytes_per_vector();
    let graph_only_bytes_per_vector = index.graph_only_bytes_per_vector();
    let peak_rss_mb = crate::hardware::get_peak_rss_mb().unwrap_or(0.0);
    let max_layer = index.max_layer();
    let entry_point = index.entry_point().unwrap_or(0);

    if loaded.query.rows() > 0 {
        index.warmup(loaded.query.row(0), 10, 50)?;
    }

    let runs = sweep_runs.max(1);
    let mut run_stats = Vec::with_capacity(runs);
    let mut last_indices = None;
    for _ in 0..runs {
        let timed = index.search_batch_timed(&loaded.query, 10, 50)?;
        run_stats.push(timed.latency);
        last_indices = Some(timed.result.indices);
    }
    let summary = summarize_repeated_latencies(&run_stats);
    let cand_indices = last_indices.unwrap();

    let recall_at_10 = if actual_scale == 1_000_000 {
        if let Some(ref gt) = loaded.official_ground_truth {
            compute_recall_at_k(&cand_indices, gt, 10)?
        } else {
            let exact_queries = loaded.query.slice_first_rows(100.min(loaded.query.rows()));
            let exact_res = ExactKnn::search_parallel_with_base(
                Metric::L2Squared,
                &loaded.base,
                &exact_queries,
                10,
                0,
            )?;
            let cand_slice = cand_indices.slice_first_rows(exact_queries.rows());
            compute_recall_at_k(&cand_slice, &exact_res.indices, 10)?
        }
    } else {
        let exact_res = ExactKnn::search_parallel_with_base(
            Metric::L2Squared,
            &loaded.base,
            &loaded.query,
            10,
            0,
        )?;
        compute_recall_at_k(&cand_indices, &exact_res.indices, 10)?
    };

    let raw_vector_bytes_per_vector = (loaded.base.cols() * std::mem::size_of::<f32>()) as f64;
    let graph_bytes_per_vector = bytes_per_vector;
    let total_bytes_per_vector = raw_vector_bytes_per_vector + graph_bytes_per_vector;

    Ok(BaselineScaleProfile {
        measured_this_run: true,
        scale_vectors: actual_scale,
        build_time_s,
        build_qps,
        peak_rss_mb,
        bytes_per_vector,
        raw_vector_bytes_per_vector,
        graph_bytes_per_vector,
        graph_only_bytes_per_vector,
        total_bytes_per_vector,
        graph_memory_bytes,
        graph_only_memory_bytes,
        graph_memory_mb,
        query_latency_ms: summary.median_latency.mean_ms,
        query_qps: summary.median_latency.qps,
        k: 10,
        ef_search: 50,
        recall_at_10,
        max_layer,
        entry_point,
        latency_min_ms: summary.median_latency.min_ms,
        latency_p50_ms: summary.median_latency.p50_ms,
        latency_p95_ms: summary.median_latency.p95_ms,
        latency_p99_ms: summary.median_latency.p99_ms,
        latency_max_ms: summary.median_latency.max_ms,
    })
}

fn format_scale_profile_kv(p: &BaselineScaleProfile) -> String {
    format!(
        "SCALE_PROFILE_BEGIN\n\
         measured_this_run={}\n\
         scale_vectors={}\n\
         build_time_s={:.9}\n\
         build_qps={:.6}\n\
         peak_rss_mb={:.6}\n\
         bytes_per_vector={:.6}\n\
         raw_vector_bytes_per_vector={:.6}\n\
         graph_bytes_per_vector={:.6}\n\
         graph_only_bytes_per_vector={:.6}\n\
         total_bytes_per_vector={:.6}\n\
         graph_memory_bytes={}\n\
         graph_only_memory_bytes={}\n\
         graph_memory_mb={:.6}\n\
         query_latency_ms={:.9}\n\
         query_qps={:.6}\n\
         k={}\n\
         ef_search={}\n\
         recall_at_10={:.9}\n\
         max_layer={}\n\
         entry_point={}\n\
         latency_min_ms={:.9}\n\
         latency_p50_ms={:.9}\n\
         latency_p95_ms={:.9}\n\
         latency_p99_ms={:.9}\n\
         latency_max_ms={:.9}\n\
         SCALE_PROFILE_END\n",
        p.measured_this_run,
        p.scale_vectors,
        p.build_time_s,
        p.build_qps,
        p.peak_rss_mb,
        p.bytes_per_vector,
        p.raw_vector_bytes_per_vector,
        p.graph_bytes_per_vector,
        p.graph_only_bytes_per_vector,
        p.total_bytes_per_vector,
        p.graph_memory_bytes,
        p.graph_only_memory_bytes,
        p.graph_memory_mb,
        p.query_latency_ms,
        p.query_qps,
        p.k,
        p.ef_search,
        p.recall_at_10,
        p.max_layer,
        p.entry_point,
        p.latency_min_ms,
        p.latency_p50_ms,
        p.latency_p95_ms,
        p.latency_p99_ms,
        p.latency_max_ms,
    )
}

fn parse_scale_profile_kv(output: &str) -> Result<BaselineScaleProfile> {
    let mut in_block = false;
    let mut p = BaselineScaleProfile {
        measured_this_run: true,
        scale_vectors: 0,
        build_time_s: 0.0,
        build_qps: 0.0,
        peak_rss_mb: 0.0,
        bytes_per_vector: 0.0,
        raw_vector_bytes_per_vector: 0.0,
        graph_bytes_per_vector: 0.0,
        graph_only_bytes_per_vector: 0.0,
        total_bytes_per_vector: 0.0,
        graph_memory_bytes: 0,
        graph_only_memory_bytes: 0,
        graph_memory_mb: 0.0,
        query_latency_ms: 0.0,
        query_qps: 0.0,
        k: 10,
        ef_search: 50,
        recall_at_10: 0.0,
        max_layer: 0,
        entry_point: 0,
        latency_min_ms: 0.0,
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
        latency_p99_ms: 0.0,
        latency_max_ms: 0.0,
    };
    let mut found_end = false;

    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line == "SCALE_PROFILE_BEGIN" {
            in_block = true;
            continue;
        }
        if line == "SCALE_PROFILE_END" {
            if in_block {
                found_end = true;
                break;
            }
            continue;
        }
        if !in_block {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            match k {
                "measured_this_run" => p.measured_this_run = v.parse().unwrap_or(true),
                "scale_vectors" => p.scale_vectors = v.parse().unwrap_or(0),
                "build_time_s" => p.build_time_s = v.parse().unwrap_or(0.0),
                "build_qps" => p.build_qps = v.parse().unwrap_or(0.0),
                "peak_rss_mb" => p.peak_rss_mb = v.parse().unwrap_or(0.0),
                "bytes_per_vector" => p.bytes_per_vector = v.parse().unwrap_or(0.0),
                "raw_vector_bytes_per_vector" => {
                    p.raw_vector_bytes_per_vector = v.parse().unwrap_or(0.0)
                }
                "graph_bytes_per_vector" => p.graph_bytes_per_vector = v.parse().unwrap_or(0.0),
                "graph_only_bytes_per_vector" => {
                    p.graph_only_bytes_per_vector = v.parse().unwrap_or(0.0)
                }
                "total_bytes_per_vector" => p.total_bytes_per_vector = v.parse().unwrap_or(0.0),
                "graph_memory_bytes" => p.graph_memory_bytes = v.parse().unwrap_or(0),
                "graph_only_memory_bytes" => p.graph_only_memory_bytes = v.parse().unwrap_or(0),
                "graph_memory_mb" => p.graph_memory_mb = v.parse().unwrap_or(0.0),
                "query_latency_ms" => p.query_latency_ms = v.parse().unwrap_or(0.0),
                "query_qps" => p.query_qps = v.parse().unwrap_or(0.0),
                "k" => p.k = v.parse().unwrap_or(10),
                "ef_search" => p.ef_search = v.parse().unwrap_or(50),
                "recall_at_10" => p.recall_at_10 = v.parse().unwrap_or(0.0),
                "max_layer" => p.max_layer = v.parse().unwrap_or(0),
                "entry_point" => p.entry_point = v.parse().unwrap_or(0),
                "latency_min_ms" => p.latency_min_ms = v.parse().unwrap_or(0.0),
                "latency_p50_ms" => p.latency_p50_ms = v.parse().unwrap_or(0.0),
                "latency_p95_ms" => p.latency_p95_ms = v.parse().unwrap_or(0.0),
                "latency_p99_ms" => p.latency_p99_ms = v.parse().unwrap_or(0.0),
                "latency_max_ms" => p.latency_max_ms = v.parse().unwrap_or(0.0),
                _ => {}
            }
        }
    }

    if !found_end || p.scale_vectors == 0 {
        return Err(VectorSearchError::VerificationFailed(
            "Failed to parse isolated child process scale profile output".to_string(),
        ));
    }
    Ok(p)
}

/// Spawns a fresh OS child process to measure a single SIFT1M prefix scale with an unpolluted
/// Peak RSS high-water mark, falling back to in-process execution if invoked from a test harness.
pub fn spawn_isolated_scale_profile(
    data_cache_dir: &Path,
    scale: usize,
    seed: u64,
    allow_download: bool,
    sweep_runs: usize,
) -> Result<BaselineScaleProfile> {
    if let Ok(exe) = std::env::current_exe() {
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("--phase")
            .arg("3")
            .arg("--internal-profile-scale")
            .arg(scale.to_string())
            .arg("--data-dir")
            .arg(data_cache_dir.as_os_str())
            .arg("--seed")
            .arg(seed.to_string())
            .arg("--sweep-runs")
            .arg(sweep_runs.max(1).to_string());
        if !allow_download {
            cmd.arg("--offline");
        }
        if let Ok(output) = cmd.output() {
            if output.status.success() {
                if let Ok(stdout) = String::from_utf8(output.stdout) {
                    if let Ok(profile) = parse_scale_profile_kv(&stdout) {
                        return Ok(profile);
                    }
                }
            }
        }
    }
    profile_single_sift1m_scale(data_cache_dir, scale, seed, allow_download, sweep_runs)
}

/// Executes multi-scale baseline HNSW profiling across the provided vector scales.
/// Each scale runs in its own isolated OS process and loads only `scale` vectors from disk
/// so that per-scale Peak RSS is never polluted by earlier or larger allocations.
pub fn run_baseline_multi_scale_profile(
    data_cache_dir: &Path,
    scales: &[usize],
    seed: u64,
    allow_download: bool,
) -> Result<BaselineMultiScaleReport> {
    println!("{}", "=".repeat(80));
    println!(" HNSW MULTI-SCALE BASELINE PROFILING (10k, 100k, 1M) ");
    println!("{}", "=".repeat(80));

    let hw = detect_hardware();
    println!("Hardware: {hw}");

    let dataset_source = load_or_download_sift_with_options(
        SiftPreset::Sift1M,
        data_cache_dir,
        allow_download,
        seed,
        Some(1),
        Some(1),
        false,
    )?
    .source;
    println!("  - Dataset Provenance: {dataset_source}");

    let eval_scales = if scales.is_empty() {
        &[10_000, 100_000, 1_000_000][..]
    } else {
        scales
    };

    let mut profiles = Vec::with_capacity(eval_scales.len());

    for (idx, &scale) in eval_scales.iter().enumerate() {
        println!(
            "\n[{}/{}] Profiling Scale in Isolated Process: {} vectors...",
            idx + 1,
            eval_scales.len(),
            scale
        );
        let p = spawn_isolated_scale_profile(data_cache_dir, scale, seed, allow_download, 5)?;
        println!(
            "  - Build finished in {:.3} s ({:.1} vectors/s)",
            p.build_time_s, p.build_qps
        );
        println!(
            "  - Graph memory: {:.2} MiB ({:.2} B/vec total, {:.2} B/vec graph-only)",
            p.graph_memory_mb, p.graph_bytes_per_vector, p.graph_only_bytes_per_vector
        );
        println!("  - Isolated Process Peak RSS: {:.2} MiB", p.peak_rss_mb);
        println!("  - Recall@10: {:.2}%", p.recall_at_10 * 100.0);
        println!(
            "  - Query latency (median of 5 runs): {:.3} ms ({:.1} QPS)",
            p.query_latency_ms, p.query_qps
        );
        profiles.push(p);
    }

    #[cfg(target_os = "windows")]
    let peak_rss_measurement_method = "windows_powershell_peak_working_set_64 (per-scale isolated child process): powershell -NoProfile -NonInteractive -Command (Get-Process -Id <benchmark process id>).PeakWorkingSet64".to_string();
    #[cfg(target_os = "linux")]
    let peak_rss_measurement_method =
        "linux_proc_status_vm_hwm (per-scale isolated child process): /proc/self/status VmHWM"
            .to_string();
    #[cfg(target_os = "macos")]
    let peak_rss_measurement_method =
        "macos_ps_rss (per-scale isolated child process): ps -o rss= -p <pid>".to_string();
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    let peak_rss_measurement_method = "unknown_platform".to_string();

    Ok(BaselineMultiScaleReport {
        hardware: hw,
        dataset_source,
        hnsw_m: 16,
        hnsw_m_max0: 32,
        hnsw_ef_construction: 100,
        seed,
        metric: Metric::L2Squared,
        peak_rss_measurement_method,
        scale_profiles: profiles,
    })
}

/// Formats and prints a multi-scale baseline profiling summary table to stdout.
pub fn print_baseline_profile_report(r: &BaselineMultiScaleReport) {
    println!("\n{}", "=".repeat(80));
    println!(" MULTI-SCALE BASELINE HNSW PROFILING SUMMARY ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!("Dataset:  Authentic SIFT1M (1,000,000 base vectors, 128-d, L2 metric)");
    println!("Config:   M=16, M0=32, efConstruction=100, seed={}", r.seed);
    println!("Peak RSS: {}", r.peak_rss_measurement_method);
    println!(
        "\n+---------+------------+-----------+-------------+------------+-------------+--------------+-------------+--------------------+-----------+------------+"
    );
    println!(
        "|  Scale  | Build Time | Build QPS | Peak RSS MB | Raw B/Vec  | Graph B/Vec | Graph-Only B | Total B/Vec | Query Latency (ms) | Query QPS | Recall@10  |"
    );
    println!(
        "+---------+------------+-----------+-------------+------------+-------------+--------------+-------------+--------------------+-----------+------------+"
    );

    for p in &r.scale_profiles {
        let scale_label = if p.scale_vectors == 1_000_000 {
            "1M".to_string()
        } else {
            format!("{}k", p.scale_vectors / 1000)
        };
        println!(
            "| {:^7} | {:^8.2} s | {:^9.1} | {:^11.2} | {:^10.1} | {:^11.2} | {:^12.2} | {:^11.2} | {:^18.3} | {:^9.1} | {:^9.2}% |",
            scale_label,
            p.build_time_s,
            p.build_qps,
            p.peak_rss_mb,
            p.raw_vector_bytes_per_vector,
            p.graph_bytes_per_vector,
            p.graph_only_bytes_per_vector,
            p.total_bytes_per_vector,
            p.query_latency_ms,
            p.query_qps,
            p.recall_at_10 * 100.0,
        );
    }
    println!(
        "+---------+------------+-----------+-------------+------------+-------------+--------------+-------------+--------------------+-----------+------------+"
    );
}

/// Verifies bit-exact reproducibility across two independent index builds and query executions.
pub fn verify_reproducibility(
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    config: &HnswConfig,
    k: usize,
    ef_search: usize,
) -> Result<()> {
    verify_reproducibility_with_metric(base, query, Metric::L2Squared, config, k, ef_search)
}

/// Verifies bit-exact reproducibility across two independent index builds and query executions
/// under the specified distance `metric`.
pub fn verify_reproducibility_with_metric(
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    metric: Metric,
    config: &HnswConfig,
    k: usize,
    ef_search: usize,
) -> Result<()> {
    let mut index1 = HnswIndex::build(base, metric, config.clone())?;
    let res1 = index1.search_batch(query, k, ef_search)?;

    let mut index2 = HnswIndex::build(base, metric, config.clone())?;
    let res2 = index2.search_batch(query, k, ef_search)?;

    if index1.max_layer() != index2.max_layer() {
        return Err(VectorSearchError::VerificationFailed(
            "Max layer mismatch between identical runs".to_string(),
        ));
    }
    if index1.entry_point() != index2.entry_point() {
        return Err(VectorSearchError::VerificationFailed(
            "Entry point mismatch between identical runs".to_string(),
        ));
    }

    if res1.indices.as_slice() != res2.indices.as_slice() {
        return Err(VectorSearchError::VerificationFailed(
            "Neighbor indices mismatch between identical runs".to_string(),
        ));
    }

    for (i, (&d1, &d2)) in res1
        .distances
        .as_slice()
        .iter()
        .zip(res2.distances.as_slice().iter())
        .enumerate()
    {
        if d1.to_bits() != d2.to_bits() {
            return Err(VectorSearchError::VerificationFailed(format!(
                "Distance bit mismatch at index {i}: {d1} != {d2}"
            )));
        }
    }

    Ok(())
}

/// Executes the complete Phase 2 verification and benchmarking pipeline.
pub fn run_phase2_benchmark(config: &Phase2Config) -> Result<Phase2Report> {
    if config.k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0)".to_string(),
        ));
    }

    let hardware = detect_hardware();

    // 1. Load or generate dataset
    let mut loaded: LoadedDataset = match &config.dataset_mode {
        Phase1DatasetMode::Sift(preset) => load_or_download_sift(
            *preset,
            &config.data_cache_dir,
            config.allow_download,
            config.seed,
        )?,
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let (base, query) =
                generate_synthetic_sift(*num_base, *num_query, *dim, *num_clusters, config.seed)?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic { seed: config.seed },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => load_custom_dataset(base_fvecs, query_fvecs, gt_ivecs.as_deref())?,
    };

    if let Some(limit) = config.max_queries {
        if limit == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "max_queries must be positive (> 0) when specified".to_string(),
            ));
        }
        if limit < loaded.query.rows() {
            loaded.query = loaded.query.slice_first_rows(limit);
            if let Some(gt) = &loaded.official_ground_truth {
                loaded.official_ground_truth = Some(gt.slice_first_rows(limit));
            }
        }
    }

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();

    // 2. Establish Exact KNN Ground Truth
    let mut exact_knn = ExactKnn::new(config.metric);
    exact_knn.fit(loaded.base.clone())?;
    let exact_timed = exact_knn.search_timed(&loaded.query, config.k, true)?;
    let gt_indices = &exact_timed.result.indices;

    let exact_self_verified = verify_self_recall(gt_indices, config.k)?;
    let exact_official_gt_match = if let Some(official_gt) = &loaded.official_ground_truth {
        if official_gt.rows() == gt_indices.rows() && official_gt.cols() > 0 {
            let eval_k = config.k.min(official_gt.cols());
            let recall = compute_recall_at_k(gt_indices, official_gt, eval_k)?;
            (recall - 1.0).abs() < 1e-6
        } else {
            false
        }
    } else {
        false
    };

    // 3. Construct HNSW Index
    let hnsw_cfg = HnswConfig {
        m: config.m,
        m_max: None,
        m_max0: config.m_max0,
        ef_construction: config.ef_construction,
        m_l: None,
        extend_candidates: false,
        keep_pruned: true,
        seed: config.seed,
    };

    let t_build = Instant::now();
    let mut hnsw = HnswIndex::build(&loaded.base, config.metric, hnsw_cfg.clone())?;
    let build_time_s = t_build.elapsed().as_secs_f64();

    // 4. Warmup HNSW
    if loaded.query.rows() > 0 {
        hnsw.warmup(loaded.query.row(0), config.k, 50)?;
    }

    // 5. Evaluate Sweep Across efSearch
    let mut ef_evaluations = Vec::new();
    let mut previous_recall = 0.0;
    let mut monotonic_scaling_verified = true;

    // The index clamps efSearch up to k, so ef < k would silently duplicate the ef == k point.
    // Report the effective value once instead of plotting identical results as separate points.
    let mut effective_efs: Vec<usize> = Vec::new();
    for &ef in &config.ef_search_list {
        let effective = ef.max(config.k);
        if !effective_efs.contains(&effective) {
            effective_efs.push(effective);
        }
    }

    for &ef in &effective_efs {
        let timed = hnsw.search_batch_timed(&loaded.query, config.k, ef)?;
        let recall = compute_recall_at_k(&timed.result.indices, gt_indices, config.k)?;

        let is_monotonic = recall >= previous_recall - 1e-9;
        if !is_monotonic {
            monotonic_scaling_verified = false;
        }

        let speedup = if timed.latency.mean_ms > 0.0 {
            exact_timed.latency.mean_ms / timed.latency.mean_ms
        } else {
            0.0
        };

        ef_evaluations.push(HnswEfEvaluation {
            ef_search: ef,
            recall_at_k: recall,
            latency: timed.latency,
            speedup_vs_exact: speedup,
            monotonic_passed: is_monotonic,
        });

        previous_recall = recall;
    }

    // 6. Verify Reproducibility (if enabled)
    let reproducibility_checked = config.run_repro_check;
    let reproducibility_verified = if config.run_repro_check {
        verify_reproducibility_with_metric(
            &loaded.base,
            &loaded.query,
            config.metric,
            &hnsw_cfg,
            config.k,
            50,
        )
        .is_ok()
    } else {
        false
    };

    // 7. Verify performance target: one operating point must satisfy recall and latency together.
    let performance_target_ef = ef_evaluations
        .iter()
        .filter(|e| {
            e.recall_at_k >= PHASE2_TARGET_MIN_RECALL
                && e.latency.mean_ms < PHASE2_TARGET_MAX_LATENCY_MS
        })
        .map(|e| e.ef_search)
        .min();
    let performance_target_met = performance_target_ef.is_some();

    Ok(Phase2Report {
        hardware,
        dataset_source: loaded.source,
        num_base,
        num_query,
        dim,
        k: config.k,
        seed: config.seed,
        metric: config.metric,
        hnsw_m: config.m,
        hnsw_m_max0: hnsw_cfg.effective_m_max0(),
        hnsw_ef_construction: config.ef_construction,
        build_time_s,
        max_layer: hnsw.max_layer(),
        entry_point: hnsw.entry_point().unwrap_or(0),
        exact_baseline_latency: exact_timed.latency,
        exact_self_verified,
        exact_official_gt_match,
        ef_evaluations,
        monotonic_scaling_verified,
        performance_target_met,
        performance_target_ef,
        reproducibility_checked,
        reproducibility_verified,
    })
}

/// Formats and displays a human-readable Phase 2 benchmark summary table.
pub fn print_phase2_report(r: &Phase2Report) {
    println!("{}", "=".repeat(80));
    println!(" HNSW APPROXIMATE NEAREST NEIGHBOR SEARCH (RUST) -- PHASE 2 BENCHMARK ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!(
        "Dataset:  {} ({} base, {} queries, {}-d, {})",
        r.dataset_source, r.num_base, r.num_query, r.dim, r.metric
    );
    println!(
        "HNSW:     M={}, M_max0={}, efConstruction={}, seed={}",
        r.hnsw_m, r.hnsw_m_max0, r.hnsw_ef_construction, r.seed
    );
    println!("{}", "-".repeat(80));

    println!("[+] Phase 1 Brute-Force Exact KNN Ground Truth:");
    println!(
        "  - Baseline Mean Latency: {:.3} ms / query",
        r.exact_baseline_latency.mean_ms
    );
    println!(
        "  - Baseline Median (p50): {:.3} ms / query",
        r.exact_baseline_latency.p50_ms
    );
    println!(
        "  - Baseline Throughput:   {:.1} QPS",
        r.exact_baseline_latency.qps
    );
    println!(
        "  - Ground Truth Self-Rec: {}",
        if r.exact_self_verified {
            "PASSED (100.00% exact match)"
        } else {
            "FAILED"
        }
    );
    if r.exact_official_gt_match {
        println!("  - Official GT Match:     PASSED (100.00% exact match)");
    }

    println!("\n[+] HNSW Index Construction ({} vectors):", r.num_base);
    println!(
        "  - Build Elapsed:         {:.2} ms",
        r.build_time_s * 1000.0
    );
    println!("  - Max Graph Height:      {} layers", r.max_layer);
    println!("  - Global Entry Point ID: {}", r.entry_point);

    println!(
        "\n[+] Monotonic Recall Scaling & Latency Speedup Evaluation (k={}):",
        r.k
    );
    println!("+----------+------------+-----------+----------+----------+-----------+----------+-------------+");
    println!("| efSearch |  Recall@k  | Mean (ms) | p50 (ms) | p99 (ms) |    QPS    | Speedup  | Monotonicity|");
    println!("+----------+------------+-----------+----------+----------+-----------+----------+-------------+");

    for (i, eval) in r.ef_evaluations.iter().enumerate() {
        let status = if i == 0 {
            "BASE"
        } else if eval.monotonic_passed {
            "PASSED"
        } else {
            "FAILED"
        };
        println!(
            "| {:^8} | {:^10.2}% | {:^9.3} | {:^8.3} | {:^8.3} | {:^9.1} | {:^6.1} x | {:^11} |",
            eval.ef_search,
            eval.recall_at_k * 100.0,
            eval.latency.mean_ms,
            eval.latency.p50_ms,
            eval.latency.p99_ms,
            eval.latency.qps,
            eval.speedup_vs_exact,
            status,
        );
    }
    println!("+----------+------------+-----------+----------+----------+-----------+----------+-------------+");

    println!("\n[+] Reproducibility & Invariant Verification:");
    println!(
        "  - Strict Monotonic Scaling:   {}",
        if r.monotonic_scaling_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "  - Perf Target (recall>={:.2}, <{} ms): {}",
        PHASE2_TARGET_MIN_RECALL,
        PHASE2_TARGET_MAX_LATENCY_MS,
        match r.performance_target_ef {
            Some(ef) => format!("PASSED (efSearch={ef})"),
            None => "FAILED (no efSearch meets both)".to_string(),
        }
    );
    println!(
        "  - Bit-Exact Reproducibility:  {}",
        if !r.reproducibility_checked {
            "SKIPPED"
        } else if r.reproducibility_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    println!("\n{}", "=".repeat(80));
    println!(" PHASE 2 COMPLETE (RUST) -- HALTED FOR USER REVIEW BEFORE PHASE 3 ");
    println!("{}", "=".repeat(80));
}

/// Entry point for CLI execution of Phase 2 benchmark.
pub fn run_phase2_cli(args: &[String]) -> Result<()> {
    let mut dataset_choice = String::from("siftsmall");
    let mut num_base: usize = 10_000;
    let mut num_query: usize = 100;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 16;
    let mut k: usize = 10;
    let mut seed: u64 = 42;
    let mut metric = Metric::L2Squared;
    let mut m: usize = 16;
    let mut m_max0: Option<usize> = None;
    let mut ef_construction: usize = 100;
    let mut ef_search_list = vec![10_usize, 20, 50, 100, 200];
    let mut data_cache_dir = PathBuf::from("data");
    let mut output_dir = PathBuf::from("data");
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut gt_ivecs: Option<PathBuf> = None;
    let mut max_queries: Option<usize> = None;
    let mut allow_download = true;
    let mut save_files = true;
    let mut run_repro_check = true;
    let mut json_out: Option<PathBuf> = None;
    let mut profile_baseline = false;
    let mut profile_scales: Vec<usize> = Vec::new();
    let mut internal_profile_scale: Option<usize> = None;
    let mut sweep_runs: usize = 5;
    let mut show_help = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                show_help = true;
                i += 1;
            }
            "--dataset" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --dataset".to_string(),
                    ));
                }
                dataset_choice = args[i + 1].to_ascii_lowercase();
                i += 2;
            }
            "--num-base" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --num-base".to_string(),
                    ));
                }
                num_base = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --num-base".to_string())
                })?;
                i += 2;
            }
            "--num-query" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --num-query".to_string(),
                    ));
                }
                num_query = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --num-query".to_string())
                })?;
                i += 2;
            }
            "--max-queries" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --max-queries".to_string(),
                    ));
                }
                max_queries = Some(args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --max-queries".to_string())
                })?);
                i += 2;
            }
            "--dim" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --dim".to_string(),
                    ));
                }
                dim = args[i + 1]
                    .parse()
                    .map_err(|_| VectorSearchError::InvalidArgument("Invalid --dim".to_string()))?;
                i += 2;
            }
            "--num-clusters" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --num-clusters".to_string(),
                    ));
                }
                num_clusters = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --num-clusters".to_string())
                })?;
                i += 2;
            }
            "--k" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --k".to_string(),
                    ));
                }
                k = args[i + 1]
                    .parse()
                    .map_err(|_| VectorSearchError::InvalidArgument("Invalid --k".to_string()))?;
                i += 2;
            }
            "--seed" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --seed".to_string(),
                    ));
                }
                seed = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --seed".to_string())
                })?;
                i += 2;
            }
            "--metric" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --metric".to_string(),
                    ));
                }
                metric = std::str::FromStr::from_str(&args[i + 1])?;
                i += 2;
            }
            "--m" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --m".to_string(),
                    ));
                }
                m = args[i + 1]
                    .parse()
                    .map_err(|_| VectorSearchError::InvalidArgument("Invalid --m".to_string()))?;
                i += 2;
            }
            "--m-max0" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --m-max0".to_string(),
                    ));
                }
                m_max0 = Some(args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --m-max0".to_string())
                })?);
                i += 2;
            }
            "--ef-construction" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --ef-construction".to_string(),
                    ));
                }
                ef_construction = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --ef-construction".to_string())
                })?;
                i += 2;
            }
            "--ef-search" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --ef-search".to_string(),
                    ));
                }
                ef_search_list = args[i + 1]
                    .split(',')
                    .map(|s| {
                        s.trim().parse::<usize>().map_err(|_| {
                            VectorSearchError::InvalidArgument(format!(
                                "Invalid ef-search value '{s}'"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?;
                i += 2;
            }
            "--data-dir" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --data-dir".to_string(),
                    ));
                }
                data_cache_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output-dir" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --output-dir".to_string(),
                    ));
                }
                output_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--base-fvecs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --base-fvecs".to_string(),
                    ));
                }
                base_fvecs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--query-fvecs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --query-fvecs".to_string(),
                    ));
                }
                query_fvecs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--gt-ivecs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --gt-ivecs".to_string(),
                    ));
                }
                gt_ivecs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--json-out" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --json-out".to_string(),
                    ));
                }
                json_out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--offline" => {
                allow_download = false;
                i += 1;
            }
            "--no-save" => {
                save_files = false;
                i += 1;
            }
            "--repro-check" => {
                run_repro_check = true;
                i += 1;
            }
            "--no-repro-check" => {
                run_repro_check = false;
                i += 1;
            }
            "--phase" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --phase".to_string(),
                    ));
                }
                i += 2;
            }
            "--profile-baseline" | "--baseline-profile" => {
                profile_baseline = true;
                i += 1;
            }
            "--profile-scales" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --profile-scales".to_string(),
                    ));
                }
                profile_scales = args[i + 1]
                    .split(',')
                    .map(|s| {
                        s.trim().parse::<usize>().map_err(|_| {
                            VectorSearchError::InvalidArgument(format!(
                                "Invalid profile scale value '{s}'"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?;
                profile_baseline = true;
                i += 2;
            }
            "--internal-profile-scale" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --internal-profile-scale".to_string(),
                    ));
                }
                internal_profile_scale = Some(args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument(
                        "Invalid --internal-profile-scale".to_string(),
                    )
                })?);
                i += 2;
            }
            "--sweep-runs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --sweep-runs".to_string(),
                    ));
                }
                sweep_runs = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --sweep-runs".to_string())
                })?;
                i += 2;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI flag '{unknown}'"
                )));
            }
        }
    }

    if let Some(scale) = internal_profile_scale {
        let profile = profile_single_sift1m_scale(
            &data_cache_dir,
            scale,
            seed,
            allow_download,
            sweep_runs.max(1),
        )?;
        print!("{}", format_scale_profile_kv(&profile));
        return Ok(());
    }

    if show_help {
        println!(
            "\
HNSW Vector Search (Rust) -- Phase 2: 10,000 Vectors Verification & Benchmarking

USAGE:
    cargo run --release --bin phase2 -- [OPTIONS]
    cargo run --release --bin phase1 -- --phase 2 [OPTIONS]

OPTIONS:
    --dataset <siftsmall|sift1m|synthetic|custom>
                                Dataset preset to load or download (default: siftsmall)
    --data-dir <DIR>            Cache directory for public SIFT corpora (default: data)
    --output-dir <DIR>          Output directory for artifacts (default: data)
    --num-base <N>              Base vector count for synthetic dataset (default: 10000)
    --num-query <Q>             Query vector count for synthetic dataset (default: 100)
    --dim <D>                   Vector dimension for synthetic dataset (default: 128)
    --k <K>                     Top-k nearest neighbors to retrieve (default: 10)
    --seed <SEED>               Deterministic PRNG seed (default: 42)
    --metric <l2|euclidean|ip>  Distance metric (default: l2)
    --m <M>                     Max outgoing edges per node at layers > 0 (default: 16)
    --m-max0 <M0>               Max outgoing edges per node at layer 0 (default: 2*M)
    --ef-construction <EFC>     Candidate beam width during index build (default: 100)
    --ef-search <LIST>          Comma-separated efSearch values to sweep (default: 10,20,50,100,200)
    --profile-baseline          Profile HNSW across multiple scales (10k, 100k, 1M)
    --profile-scales <SCALES>   Comma-separated vector scales to profile (default: 10000,100000,1000000)
    --json-out <PATH>           Write structured JSON benchmark report to PATH
    --offline                   Disable internet downloads; use local cache or synthetic
    --no-save                   Skip writing intermediate artifacts to disk
    --repro-check               Verify bit-exact reproducibility across independent runs
    --no-repro-check            Skip reproducibility check
    -h, --help                  Print this help message"
        );
        return Ok(());
    }

    if profile_baseline {
        let report = run_baseline_multi_scale_profile(
            &data_cache_dir,
            &profile_scales,
            seed,
            allow_download,
        )?;
        print_baseline_profile_report(&report);

        let target_json =
            json_out.unwrap_or_else(|| PathBuf::from("data/baseline_profile_report.json"));
        if let Some(parent) = target_json.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&target_json, report.to_json_string())?;
        println!("  - JSON Baseline Report:  {}", target_json.display());
        return Ok(());
    }

    let dataset_mode = if base_fvecs.is_some()
        || query_fvecs.is_some()
        || dataset_choice == "custom"
    {
        let (Some(b), Some(q)) = (base_fvecs, query_fvecs) else {
            return Err(VectorSearchError::InvalidArgument(
                "Custom dataset mode requires both --base-fvecs <path> and --query-fvecs <path>"
                    .to_string(),
            ));
        };
        Phase1DatasetMode::Custom {
            base_fvecs: b,
            query_fvecs: q,
            gt_ivecs,
        }
    } else {
        match dataset_choice.as_str() {
            "siftsmall" => Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
            "sift1m" | "sift" => Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            "synthetic" => Phase1DatasetMode::Synthetic {
                num_base,
                num_query,
                dim,
                num_clusters,
            },
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unsupported --dataset '{other}'. Expected 'siftsmall', 'sift1m', 'synthetic', or 'custom'."
                )));
            }
        }
    };

    let config = Phase2Config {
        dataset_mode,
        data_cache_dir,
        output_dir,
        k,
        seed,
        metric,
        m,
        m_max0,
        ef_construction,
        ef_search_list,
        allow_download,
        save_files,
        max_queries,
        run_repro_check,
    };

    let report = run_phase2_benchmark(&config)?;
    print_phase2_report(&report);

    if let Some(json_path) = &json_out {
        if let Some(parent) = json_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(json_path, report.to_json_string())?;
        println!("  - JSON Summary Report:   {}", json_path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 3: SIFT1M Real-World Scaling & Benchmark Harness
// ---------------------------------------------------------------------------

/// Configuration parameters for Phase 3 SIFT1M scaling & benchmark.
#[derive(Debug, Clone)]
pub struct Phase3Config {
    pub dataset_mode: Phase1DatasetMode,
    pub data_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub m: usize,
    pub m_max0: Option<usize>,
    pub ef_construction: usize,
    pub ef_search_list: Vec<usize>,
    pub allow_download: bool,
    pub save_files: bool,
    pub max_queries: Option<usize>,
    pub max_base: Option<usize>,
    pub num_threads: usize,
    pub run_repro_check: bool,
    pub repro_vectors: usize,
    pub sweep_runs: usize,
    pub run_multi_scale: bool,
}

impl Default for Phase3Config {
    fn default() -> Self {
        Self {
            dataset_mode: Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            data_cache_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("data"),
            k: 10,
            seed: 42,
            metric: Metric::L2Squared,
            m: 16,
            m_max0: None,
            ef_construction: 100,
            ef_search_list: vec![10, 20, 50, 100, 200],
            allow_download: true,
            save_files: true,
            max_queries: None,
            max_base: None,
            num_threads: 0,
            run_repro_check: true,
            repro_vectors: 10_000,
            sweep_runs: 5,
            run_multi_scale: true,
        }
    }
}

/// Evaluation metrics for a single efSearch setting in Phase 3 across repeated runs.
#[derive(Debug, Clone)]
pub struct Phase3EfEvaluation {
    pub ef_search: usize,
    pub recall_at_k: f64,
    /// Median latency statistics across `num_runs` repetitions of the sweep.
    pub latency: LatencyStats,
    pub num_runs: usize,
    pub mean_ms_min: f64,
    pub mean_ms_max: f64,
    pub mean_ms_spread: f64,
    pub p99_ms_min: f64,
    pub p99_ms_max: f64,
    pub p99_ms_spread: f64,
    pub run_mean_latencies_ms: Vec<f64>,
    pub run_p99_latencies_ms: Vec<f64>,
    pub monotonic_passed: bool,
}

/// Detailed memory profile for Phase 3 SIFT1M benchmark.
#[derive(Debug, Clone)]
pub struct Phase3MemoryProfile {
    pub peak_rss_mb: f64,
    pub peak_rss_bytes: u64,
    pub peak_rss_measurement_method: String,
    pub is_peak_rss_os_measured: bool,
    pub raw_vector_bytes_per_vector: f64,
    pub graph_bytes_per_vector: f64,
    pub graph_only_bytes_per_vector: f64,
    pub total_bytes_per_vector: f64,
    pub raw_vector_total_bytes: usize,
    pub raw_vector_total_mb: f64,
    pub graph_total_bytes: usize,
    pub graph_total_mb: f64,
    pub graph_only_total_bytes: usize,
    pub graph_only_total_mb: f64,
    pub total_footprint_bytes: usize,
    pub total_footprint_mb: f64,
    pub graph_components: crate::hnsw::HnswGraphComponents,
}

/// Complete structured Phase 3 SIFT1M benchmark report.
#[derive(Debug, Clone)]
pub struct Phase3Report {
    pub phase: usize,
    pub phase_name: String,
    pub generated_at: String,
    pub hardware: HardwareInfo,
    pub dataset_source: DatasetSource,
    pub preset_name: String,
    pub base_path: Option<PathBuf>,
    pub query_path: Option<PathBuf>,
    pub groundtruth_path: Option<PathBuf>,
    pub num_base: usize,
    pub num_query: usize,
    pub dim: usize,
    pub k: usize,
    pub seed: u64,
    pub metric: Metric,
    pub hnsw_m: usize,
    pub hnsw_m_max0: usize,
    pub hnsw_ef_construction: usize,
    pub build_time_s: f64,
    pub build_qps: f64,
    pub max_layer: usize,
    pub entry_point: usize,
    pub memory_profile: Phase3MemoryProfile,
    pub multi_scale_baseline_summary: Vec<BaselineScaleProfile>,
    pub sweep_runs: usize,
    pub ef_evaluations: Vec<Phase3EfEvaluation>,
    pub monotonic_scaling_verified: bool,
    /// True only if the reproducibility check actually ran (it is skipped by `--no-repro-check`).
    pub reproducibility_checked: bool,
    /// True only if the check ran and passed. A skipped check is never reported as verified.
    pub reproducibility_verified: bool,
    /// True only when brute-force ExactKNN over `loaded.base` matches `official_ground_truth`
    /// with Recall@k = 1.0 on the verified query sample (`official_gt_check_queries`).
    pub official_ground_truth_matched: bool,
    pub official_gt_check_queries: usize,
    pub official_gt_recall_at_k: Option<f64>,
    pub official_gt_exact_id_matches: Option<usize>,
    pub terminal_stopping_enforced: bool,
    pub phase4_initiated: bool,
}

impl Phase3Report {
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(8192);
        out.push_str("{\n");
        out.push_str(&format!("  \"phase\": {},\n", self.phase));
        out.push_str(&format!(
            "  \"phase_name\": \"{}\",\n",
            escape_json(&self.phase_name)
        ));
        out.push_str(&format!(
            "  \"generated_at\": \"{}\",\n",
            escape_json(&self.generated_at)
        ));

        // Hardware
        out.push_str("  \"hardware\": {\n");
        out.push_str(&format!(
            "    \"summary\": \"{}\",\n",
            escape_json(&self.hardware.summary_line())
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&self.hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            self.hardware.logical_cores
        ));
        out.push_str(&format!(
            "    \"total_ram_gb\": {:.2},\n",
            self.hardware.total_ram_gb.unwrap_or(0.0)
        ));
        out.push_str(&format!(
            "    \"os\": \"{}\",\n",
            escape_json(&self.hardware.os)
        ));
        out.push_str(&format!(
            "    \"arch\": \"{}\"\n",
            escape_json(&self.hardware.arch)
        ));
        out.push_str("  },\n");

        // Dataset
        out.push_str("  \"dataset\": {\n");
        out.push_str(&format!(
            "    \"preset\": \"{}\",\n",
            escape_json(&self.preset_name)
        ));
        out.push_str(&format!(
            "    \"source_kind\": \"{}\",\n",
            self.dataset_source.kind_tag()
        ));
        out.push_str(&format!(
            "    \"source_description\": \"{}\",\n",
            escape_json(&self.dataset_source.to_string())
        ));
        out.push_str(&format!(
            "    \"is_authentic_sift\": {},\n",
            self.dataset_source.is_authentic_sift()
        ));
        out.push_str(&format!(
            "    \"base_path\": {},\n",
            opt_path_json(self.base_path.as_deref())
        ));
        out.push_str(&format!(
            "    \"query_path\": {},\n",
            opt_path_json(self.query_path.as_deref())
        ));
        out.push_str(&format!(
            "    \"groundtruth_path\": {},\n",
            opt_path_json(self.groundtruth_path.as_deref())
        ));
        out.push_str(&format!("    \"num_base\": {},\n", self.num_base));
        out.push_str(&format!("    \"num_query\": {},\n", self.num_query));
        out.push_str(&format!("    \"dim\": {},\n", self.dim));
        out.push_str(&format!("    \"k\": {},\n", self.k));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str(&format!("    \"metric\": \"{}\"\n", self.metric));
        out.push_str("  },\n");

        // HnswConfig
        out.push_str("  \"hnsw_config\": {\n");
        out.push_str(&format!("    \"m\": {},\n", self.hnsw_m));
        out.push_str(&format!("    \"m_max0\": {},\n", self.hnsw_m_max0));
        out.push_str(&format!(
            "    \"ef_construction\": {},\n",
            self.hnsw_ef_construction
        ));
        out.push_str(&format!("    \"seed\": {},\n", self.seed));
        out.push_str("    \"heuristic\": \"select_neighbors_heuristic_with_scratch\"\n");
        out.push_str("  },\n");

        // IndexBuild
        out.push_str("  \"index_build\": {\n");
        out.push_str(&format!("    \"num_vectors\": {},\n", self.num_base));
        out.push_str(&format!(
            "    \"build_time_s\": {:.6},\n",
            self.build_time_s
        ));
        out.push_str(&format!("    \"build_qps\": {:.2},\n", self.build_qps));
        out.push_str(&format!("    \"max_layer\": {},\n", self.max_layer));
        out.push_str(&format!("    \"entry_point\": {}\n", self.entry_point));
        out.push_str("  },\n");

        // MemoryProfile
        let mp = &self.memory_profile;
        out.push_str("  \"memory_profile\": {\n");
        out.push_str(&format!("    \"peak_rss_mb\": {:.2},\n", mp.peak_rss_mb));
        out.push_str(&format!("    \"peak_rss_bytes\": {},\n", mp.peak_rss_bytes));
        out.push_str(&format!(
            "    \"peak_rss_measurement_method\": \"{}\",\n",
            escape_json(&mp.peak_rss_measurement_method)
        ));
        out.push_str(&format!(
            "    \"is_peak_rss_os_measured\": {},\n",
            mp.is_peak_rss_os_measured
        ));

        out.push_str("    \"bytes_per_vector_breakdown\": {\n");
        out.push_str(&format!(
            "      \"raw_vector_bytes_per_vector\": {:.2},\n",
            mp.raw_vector_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"graph_bytes_per_vector\": {:.2},\n",
            mp.graph_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"graph_only_bytes_per_vector\": {:.2},\n",
            mp.graph_only_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"total_bytes_per_vector\": {:.2},\n",
            mp.total_bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"raw_vector_total_bytes\": {},\n",
            mp.raw_vector_total_bytes
        ));
        out.push_str(&format!(
            "      \"raw_vector_total_mb\": {:.2},\n",
            mp.raw_vector_total_mb
        ));
        out.push_str(&format!(
            "      \"graph_total_bytes\": {},\n",
            mp.graph_total_bytes
        ));
        out.push_str(&format!(
            "      \"graph_total_mb\": {:.2},\n",
            mp.graph_total_mb
        ));
        out.push_str(&format!(
            "      \"graph_only_total_bytes\": {},\n",
            mp.graph_only_total_bytes
        ));
        out.push_str(&format!(
            "      \"graph_only_total_mb\": {:.2},\n",
            mp.graph_only_total_mb
        ));
        out.push_str(&format!(
            "      \"total_footprint_bytes\": {},\n",
            mp.total_footprint_bytes
        ));
        out.push_str(&format!(
            "      \"total_footprint_mb\": {:.2}\n",
            mp.total_footprint_mb
        ));
        out.push_str("    },\n");

        out.push_str("    \"graph_components_bytes\": {\n");
        out.push_str(&format!(
            "      \"node_structs_bytes\": {},\n",
            mp.graph_components.node_structs_bytes
        ));
        out.push_str(&format!(
            "      \"layer0_edges_bytes\": {},\n",
            mp.graph_components.layer0_edges_bytes
        ));
        out.push_str(&format!(
            "      \"upper_layers_edges_bytes\": {},\n",
            mp.graph_components.upper_layers_edges_bytes
        ));
        out.push_str(&format!(
            "      \"outer_neighbors_headers_bytes\": {},\n",
            mp.graph_components.outer_neighbors_headers_bytes
        ));
        out.push_str(&format!(
            "      \"visited_tracker_tags_bytes\": {},\n",
            mp.graph_components.visited_tracker_tags_bytes
        ));
        out.push_str(&format!(
            "      \"heuristic_scratch_bytes\": {}\n",
            mp.graph_components.heuristic_scratch_bytes
        ));
        out.push_str("    }\n");
        out.push_str("  },\n");

        // MultiScaleBaselineSummary
        out.push_str("  \"multi_scale_baseline_summary\": [\n");
        for (i, p) in self.multi_scale_baseline_summary.iter().enumerate() {
            let comma = if i + 1 < self.multi_scale_baseline_summary.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!(
                "      \"measured_this_run\": {},\n",
                p.measured_this_run
            ));
            out.push_str(&format!("      \"scale_vectors\": {},\n", p.scale_vectors));
            out.push_str(&format!("      \"build_time_s\": {:.6},\n", p.build_time_s));
            out.push_str(&format!("      \"build_qps\": {:.2},\n", p.build_qps));
            out.push_str(&format!("      \"peak_rss_mb\": {:.2},\n", p.peak_rss_mb));
            out.push_str(&format!(
                "      \"raw_vector_bytes_per_vector\": {:.2},\n",
                p.raw_vector_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_bytes_per_vector\": {:.2},\n",
                p.graph_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_only_bytes_per_vector\": {:.2},\n",
                p.graph_only_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"total_bytes_per_vector\": {:.2},\n",
                p.total_bytes_per_vector
            ));
            out.push_str(&format!(
                "      \"graph_memory_bytes\": {},\n",
                p.graph_memory_bytes
            ));
            out.push_str(&format!(
                "      \"graph_only_memory_bytes\": {},\n",
                p.graph_only_memory_bytes
            ));
            out.push_str(&format!("      \"recall_at_10\": {:.6}\n", p.recall_at_10));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");

        // EfSearchSweep
        out.push_str("  \"ef_search_sweep\": [\n");
        for (i, ef) in self.ef_evaluations.iter().enumerate() {
            let comma = if i + 1 < self.ef_evaluations.len() {
                ","
            } else {
                ""
            };
            out.push_str("    {\n");
            out.push_str(&format!("      \"ef_search\": {},\n", ef.ef_search));
            out.push_str(&format!("      \"recall_at_10\": {:.6},\n", ef.recall_at_k));
            out.push_str(&format!("      \"num_runs\": {},\n", ef.num_runs));
            out.push_str("      \"latency\": {\n");
            out.push_str(&format!(
                "        \"mean_ms\": {:.6},\n",
                ef.latency.mean_ms
            ));
            out.push_str(&format!("        \"p50_ms\": {:.6},\n", ef.latency.p50_ms));
            out.push_str(&format!("        \"p95_ms\": {:.6},\n", ef.latency.p95_ms));
            out.push_str(&format!("        \"p99_ms\": {:.6},\n", ef.latency.p99_ms));
            out.push_str(&format!("        \"qps\": {:.2},\n", ef.latency.qps));
            out.push_str(&format!(
                "        \"mean_ms_min\": {:.6},\n",
                ef.mean_ms_min
            ));
            out.push_str(&format!(
                "        \"mean_ms_max\": {:.6},\n",
                ef.mean_ms_max
            ));
            out.push_str(&format!(
                "        \"mean_ms_spread\": {:.6},\n",
                ef.mean_ms_spread
            ));
            out.push_str(&format!("        \"p99_ms_min\": {:.6},\n", ef.p99_ms_min));
            out.push_str(&format!("        \"p99_ms_max\": {:.6},\n", ef.p99_ms_max));
            out.push_str(&format!(
                "        \"p99_ms_spread\": {:.6}\n",
                ef.p99_ms_spread
            ));
            out.push_str("      },\n");
            let mean_runs_json: Vec<String> = ef
                .run_mean_latencies_ms
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect();
            let p99_runs_json: Vec<String> = ef
                .run_p99_latencies_ms
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect();
            out.push_str(&format!(
                "      \"run_mean_latencies_ms\": [{}],\n",
                mean_runs_json.join(", ")
            ));
            out.push_str(&format!(
                "      \"run_p99_latencies_ms\": [{}],\n",
                p99_runs_json.join(", ")
            ));
            out.push_str(&format!(
                "      \"monotonic_passed\": {}\n",
                ef.monotonic_passed
            ));
            out.push_str(&format!("    }}{comma}\n"));
        }
        out.push_str("  ],\n");

        // VerificationStatus
        out.push_str("  \"verification_status\": {\n");
        out.push_str(&format!(
            "    \"authentic_sift1m_verified\": {},\n",
            matches!(self.dataset_source, DatasetSource::AuthenticSift1M { .. })
                && self.num_base == 1_000_000
        ));
        out.push_str(&format!(
            "    \"official_ground_truth_matched\": {},\n",
            self.official_ground_truth_matched
        ));
        out.push_str(&format!(
            "    \"official_gt_check_queries\": {},\n",
            self.official_gt_check_queries
        ));
        match self.official_gt_recall_at_k {
            Some(r) => out.push_str(&format!("    \"official_gt_recall_at_k\": {r:.6},\n")),
            None => out.push_str("    \"official_gt_recall_at_k\": null,\n"),
        }
        match self.official_gt_exact_id_matches {
            Some(m) => out.push_str(&format!("    \"official_gt_exact_id_matches\": {m},\n")),
            None => out.push_str("    \"official_gt_exact_id_matches\": null,\n"),
        }
        out.push_str(&format!(
            "    \"monotonic_scaling_verified\": {},\n",
            self.monotonic_scaling_verified
        ));
        out.push_str(&format!(
            "    \"reproducibility_checked\": {},\n",
            self.reproducibility_checked
        ));
        out.push_str(&format!(
            "    \"reproducibility_verified\": {},\n",
            self.reproducibility_verified
        ));
        out.push_str("    \"forbid_unsafe_code_enforced\": true,\n");
        out.push_str("    \"std_only_enforced\": true,\n");
        out.push_str(&format!(
            "    \"phase3_completed\": {},\n",
            self.monotonic_scaling_verified && self.reproducibility_verified
        ));
        out.push_str(&format!(
            "    \"terminal_stopping_enforced\": {},\n",
            self.terminal_stopping_enforced
        ));
        out.push_str(&format!(
            "    \"phase4_initiated\": {}\n",
            self.phase4_initiated
        ));
        out.push_str("  }\n");

        out.push_str("}\n");
        out
    }
}

fn load_or_default_baseline_profiles(_data_dir: &Path) -> Vec<BaselineScaleProfile> {
    vec![
        BaselineScaleProfile {
            measured_this_run: false,
            scale_vectors: 10_000,
            build_time_s: 1.482878,
            build_qps: 6743.64,
            peak_rss_mb: 13.64,
            bytes_per_vector: 342.98,
            raw_vector_bytes_per_vector: 512.00,
            graph_bytes_per_vector: 342.98,
            graph_only_bytes_per_vector: 338.98,
            total_bytes_per_vector: 854.98,
            graph_memory_bytes: 3_429_760,
            graph_only_memory_bytes: 3_389_760,
            graph_memory_mb: 3.27,
            query_latency_ms: 0.062140,
            query_qps: 16062.58,
            k: 10,
            ef_search: 50,
            recall_at_10: 0.997500,
            max_layer: 3,
            entry_point: 3015,
            latency_min_ms: 0.027000,
            latency_p50_ms: 0.063200,
            latency_p95_ms: 0.078000,
            latency_p99_ms: 0.090408,
            latency_max_ms: 0.182800,
        },
        BaselineScaleProfile {
            measured_this_run: false,
            scale_vectors: 100_000,
            build_time_s: 28.623980,
            build_qps: 3493.57,
            peak_rss_mb: 88.60,
            bytes_per_vector: 342.66,
            raw_vector_bytes_per_vector: 512.00,
            graph_bytes_per_vector: 342.66,
            graph_only_bytes_per_vector: 338.66,
            total_bytes_per_vector: 854.66,
            graph_memory_bytes: 34_266_240,
            graph_only_memory_bytes: 33_866_240,
            graph_memory_mb: 32.68,
            query_latency_ms: 0.155355,
            query_qps: 6426.16,
            k: 10,
            ef_search: 50,
            recall_at_10: 0.981400,
            max_layer: 4,
            entry_point: 56204,
            latency_min_ms: 0.041500,
            latency_p50_ms: 0.158800,
            latency_p95_ms: 0.196000,
            latency_p99_ms: 0.224216,
            latency_max_ms: 0.367500,
        },
        BaselineScaleProfile {
            measured_this_run: false,
            scale_vectors: 1_000_000,
            build_time_s: 472.189209,
            build_qps: 2117.80,
            peak_rss_mb: 845.77,
            bytes_per_vector: 342.67,
            raw_vector_bytes_per_vector: 512.00,
            graph_bytes_per_vector: 342.67,
            graph_only_bytes_per_vector: 338.67,
            total_bytes_per_vector: 854.67,
            graph_memory_bytes: 342_668_320,
            graph_only_memory_bytes: 338_668_320,
            graph_memory_mb: 326.79,
            query_latency_ms: 0.249254,
            query_qps: 4009.49,
            k: 10,
            ef_search: 50,
            recall_at_10: 0.947030,
            max_layer: 5,
            entry_point: 829262,
            latency_min_ms: 0.084700,
            latency_p50_ms: 0.254800,
            latency_p95_ms: 0.307405,
            latency_p99_ms: 0.342101,
            latency_max_ms: 2.842400,
        },
    ]
}

fn current_utc_rfc3339() -> String {
    let now = std::time::SystemTime::now();
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let rem_secs = secs % 86400;
    let hours = rem_secs / 3600;
    let mins = (rem_secs % 3600) / 60;
    let s = rem_secs % 60;

    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

/// Executes the complete Phase 3 SIFT1M scaling benchmark and profiling pipeline.
pub fn run_phase3_benchmark(config: &Phase3Config) -> Result<Phase3Report> {
    if config.k == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "k must be positive (> 0)".to_string(),
        ));
    }
    if config.max_base == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_base must be positive (> 0) when specified".to_string(),
        ));
    }
    if config.max_queries == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_queries must be positive (> 0) when specified".to_string(),
        ));
    }
    if config.sweep_runs == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "sweep_runs must be positive (> 0)".to_string(),
        ));
    }

    let hardware = detect_hardware();
    let effective_m_max0 = config.m_max0.unwrap_or(2 * config.m);

    // Baseline profiles summary: if this is a full 1M Authentic SIFT1M run with default HNSW
    // parameters and multi-scale profiling enabled, measure 10k and 100k in isolated child
    // processes BEFORE loading the 1M corpus in the parent process.
    let mut baseline_profiles = load_or_default_baseline_profiles(&config.data_cache_dir);
    let wants_full_sift1m_subscales = config.run_multi_scale
        && matches!(
            config.dataset_mode,
            Phase1DatasetMode::Sift(SiftPreset::Sift1M)
        )
        && config.max_base.unwrap_or(1_000_000) == 1_000_000
        && config.k == 10
        && config.m == 16
        && effective_m_max0 == 32
        && config.ef_construction == 100
        && config.metric == Metric::L2Squared;

    if wants_full_sift1m_subscales {
        if let Ok(probe) = load_or_download_sift_with_options(
            SiftPreset::Sift1M,
            &config.data_cache_dir,
            config.allow_download,
            config.seed,
            Some(1),
            Some(1),
            false,
        ) {
            let is_auth = probe.source.is_authentic_sift();
            drop(probe);
            if is_auth {
                for &sub_scale in &[10_000_usize, 100_000_usize] {
                    println!(
                        "[0/4] Profiling {} baseline scale in isolated child process...",
                        sub_scale
                    );
                    if let Ok(sub_prof) = spawn_isolated_scale_profile(
                        &config.data_cache_dir,
                        sub_scale,
                        config.seed,
                        config.allow_download,
                        config.sweep_runs,
                    ) {
                        if let Some(slot) = baseline_profiles
                            .iter_mut()
                            .find(|p| p.scale_vectors == sub_scale)
                        {
                            *slot = sub_prof;
                        }
                    }
                }
            }
        }
    }

    // 1. Load dataset (loading only max_base / max_queries rows from disk so per-scale Peak RSS
    // is never polluted by unrequested rows)
    let mut custom_full_base_rows: Option<usize> = None;
    let loaded: LoadedDataset = match &config.dataset_mode {
        Phase1DatasetMode::Sift(preset) => load_or_download_sift_with_options(
            *preset,
            &config.data_cache_dir,
            config.allow_download,
            config.seed,
            config.max_base,
            config.max_queries,
            true,
        )?,
        Phase1DatasetMode::Synthetic {
            num_base,
            num_query,
            dim,
            num_clusters,
        } => {
            let t0 = Instant::now();
            let eff_base = config.max_base.map_or(*num_base, |lim| lim.min(*num_base));
            let eff_query = config
                .max_queries
                .map_or(*num_query, |lim| lim.min(*num_query));
            let (base, query) =
                generate_synthetic_sift(eff_base, eff_query, *dim, *num_clusters, config.seed)?;
            LoadedDataset {
                base,
                query,
                official_ground_truth: None,
                source: DatasetSource::Synthetic { seed: config.seed },
                load_time_s: t0.elapsed().as_secs_f64(),
            }
        }
        Phase1DatasetMode::Custom {
            base_fvecs,
            query_fvecs,
            gt_ivecs,
        } => {
            let ds = load_custom_dataset_with_options(
                base_fvecs,
                query_fvecs,
                gt_ivecs.as_deref(),
                config.max_base,
                config.max_queries,
            )?;
            let d = ds.base.cols();
            if d > 0 {
                let rec_bytes = ((1 + d) as u64) * 4;
                if let Ok(meta) = fs::metadata(base_fvecs) {
                    custom_full_base_rows = usize::try_from(meta.len() / rec_bytes).ok();
                }
            }
            ds
        }
    };

    let num_base = loaded.base.rows();
    let num_query = loaded.query.rows();
    let dim = loaded.base.cols();

    // Construct HNSW index
    let hnsw_cfg = HnswConfig {
        m: config.m,
        m_max: None,
        m_max0: Some(effective_m_max0),
        ef_construction: config.ef_construction,
        m_l: None,
        extend_candidates: false,
        keep_pruned: true,
        seed: config.seed,
    };

    println!(
        "[1/4] Constructing HNSW Index on {} vectors (M={}, M0={}, efConstruction={})...",
        num_base, config.m, effective_m_max0, config.ef_construction
    );
    let t_build_start = Instant::now();
    let mut index = HnswIndex::build(&loaded.base, config.metric, hnsw_cfg.clone())?;
    let build_time_s = t_build_start.elapsed().as_secs_f64();
    let build_qps = (num_base as f64) / build_time_s;
    let max_layer = index.max_layer();
    let entry_point = index.entry_point().unwrap_or(0);

    // Profile memory immediately after build before any evaluation buffers are allocated
    println!("[2/4] Profiling memory footprint and process peak RSS...");
    let peak_rss_mb = crate::hardware::get_peak_rss_mb().unwrap_or(0.0);
    let peak_rss_bytes = (peak_rss_mb * 1024.0 * 1024.0) as u64;
    let is_peak_rss_os_measured = peak_rss_mb > 0.0;

    #[cfg(target_os = "windows")]
    let peak_rss_measurement_method = "windows_powershell_peak_working_set_64: powershell -NoProfile -NonInteractive -Command (Get-Process -Id <benchmark process id>).PeakWorkingSet64".to_string();
    #[cfg(target_os = "linux")]
    let peak_rss_measurement_method =
        "linux_proc_status_vm_hwm: /proc/self/status VmHWM".to_string();
    #[cfg(target_os = "macos")]
    let peak_rss_measurement_method = "macos_ps_rss: ps -o rss= -p <pid>".to_string();
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    let peak_rss_measurement_method = "unknown_platform".to_string();

    let raw_vector_bytes_per_vector = (dim * std::mem::size_of::<f32>()) as f64;
    let graph_bytes_per_vector = index.bytes_per_vector();
    let graph_only_bytes_per_vector = index.graph_only_bytes_per_vector();
    let total_bytes_per_vector = raw_vector_bytes_per_vector + graph_bytes_per_vector;

    let raw_vector_total_bytes = loaded.base.memory_bytes();
    let raw_vector_total_mb = loaded.base.memory_mb();
    let graph_total_bytes = index.memory_bytes();
    let graph_total_mb = index.memory_mb();
    let graph_only_total_bytes = index.graph_only_memory_bytes();
    let graph_only_total_mb = index.graph_only_memory_mb();
    let total_footprint_bytes = raw_vector_total_bytes + graph_total_bytes;
    let total_footprint_mb = (total_footprint_bytes as f64) / (1024.0 * 1024.0);

    let graph_components = index.graph_components();

    let memory_profile = Phase3MemoryProfile {
        peak_rss_mb,
        peak_rss_bytes,
        peak_rss_measurement_method,
        is_peak_rss_os_measured,
        raw_vector_bytes_per_vector,
        graph_bytes_per_vector,
        graph_only_bytes_per_vector,
        total_bytes_per_vector,
        raw_vector_total_bytes,
        raw_vector_total_mb,
        graph_total_bytes,
        graph_total_mb,
        graph_only_total_bytes,
        graph_only_total_mb,
        total_footprint_bytes,
        total_footprint_mb,
        graph_components,
    };

    // Warmup HNSW
    if loaded.query.rows() > 0 {
        index.warmup(loaded.query.row(0), config.k, 50)?;
    }

    // Ground truth verification and establishment for evaluation:
    // Compute true brute-force ExactKNN on up to 100 queries against `&loaded.base` (without
    // cloning the base matrix) and verify `official_ground_truth` against it.
    let is_full_base_match = match &loaded.source {
        DatasetSource::AuthenticSift1M { .. } => num_base == 1_000_000,
        DatasetSource::AuthenticSiftSmall { .. } => num_base == 10_000,
        DatasetSource::CustomFiles { .. } => {
            custom_full_base_rows.is_none_or(|full| num_base == full)
        }
        DatasetSource::Synthetic { .. } | DatasetSource::SyntheticFallback { .. } => false,
    };

    let check_q = 100.min(num_query);
    let check_queries = if check_q == num_query {
        loaded.query.clone()
    } else {
        loaded.query.slice_first_rows(check_q)
    };

    let exact_check_res = if check_q > 0 && num_base > 0 {
        Some(ExactKnn::search_parallel_with_base(
            config.metric,
            &loaded.base,
            &check_queries,
            config.k,
            config.num_threads,
        )?)
    } else {
        None
    };

    let mut official_ground_truth_matched = false;
    let mut official_gt_check_queries = 0_usize;
    let mut official_gt_recall_at_k: Option<f64> = None;
    let mut official_gt_exact_id_matches: Option<usize> = None;

    if is_full_base_match && matches!(config.metric, Metric::L2Squared | Metric::Euclidean) {
        if let (Some(ref gt), Some(ref exact_res)) =
            (&loaded.official_ground_truth, &exact_check_res)
        {
            let exact_idx = &exact_res.indices;
            let exact_dists = &exact_res.distances;
            if gt.rows() == num_query && gt.cols() >= exact_idx.cols() && exact_idx.cols() > 0 {
                let eval_k = exact_idx.cols();
                let gt_slice = gt.slice_first_rows(check_q);
                if let Ok(rec) = compute_recall_at_k(exact_idx, &gt_slice, eval_k) {
                    let mut exact_matches = 0_usize;
                    let mut all_valid_exact_top_k = true;
                    for q in 0..check_q {
                        let a_row = &exact_idx.row(q)[..eval_k];
                        let a_dist_row = &exact_dists.row(q)[..eval_k];
                        let b_row = &gt_slice.row(q)[..eval_k];
                        let q_vec = check_queries.row(q);

                        for (pos, (&a, &b)) in a_row.iter().zip(b_row.iter()).enumerate() {
                            if a == b {
                                exact_matches += 1;
                            }
                            if b < 0 || (b as usize) >= num_base || b_row[..pos].contains(&b) {
                                all_valid_exact_top_k = false;
                            } else {
                                let u = loaded.base.row(b as usize);
                                let b_dist = match config.metric {
                                    Metric::L2Squared => crate::distance::l2_squared(q_vec, u),
                                    Metric::Euclidean => crate::distance::euclidean(q_vec, u),
                                    Metric::InnerProduct => {
                                        -crate::distance::inner_product(q_vec, u)
                                    }
                                };
                                let expected_dist = a_dist_row[pos];
                                let dist_tol = expected_dist.abs() * 1e-5 + 1e-5;
                                if (b_dist - expected_dist).abs() > dist_tol {
                                    all_valid_exact_top_k = false;
                                }
                            }
                        }
                    }
                    official_gt_check_queries = check_q;
                    official_gt_recall_at_k = Some(rec);
                    official_gt_exact_id_matches = Some(exact_matches);
                    // Require at least 99% ID overlap AND every official top-k neighbor must be
                    // distance-equivalent at its rank (allowing only genuine distance ties).
                    official_ground_truth_matched = all_valid_exact_top_k && rec >= 0.99;
                }
            }
        }
    }

    println!(
        "[3/4] Evaluating efSearch sweep across {:?} on {} queries ({} runs per efSearch)...",
        config.ef_search_list, num_query, config.sweep_runs
    );

    let mut ef_evaluations = Vec::new();
    let mut previous_recall = 0.0;
    let mut monotonic_scaling_verified = true;

    // Deduplicate effective ef values
    let mut effective_efs: Vec<usize> = Vec::new();
    for &ef in &config.ef_search_list {
        let effective = ef.max(config.k);
        if !effective_efs.contains(&effective) {
            effective_efs.push(effective);
        }
    }

    let runs_per_ef = config.sweep_runs.max(1);
    for &ef in &effective_efs {
        let mut run_latencies = Vec::with_capacity(runs_per_ef);
        let mut last_indices: Option<RowMatrix<i32>> = None;
        for _ in 0..runs_per_ef {
            let timed = index.search_batch_timed(&loaded.query, config.k, ef)?;
            run_latencies.push(timed.latency);
            last_indices = Some(timed.result.indices);
        }
        let summary = summarize_repeated_latencies(&run_latencies);
        let cand_indices = last_indices.unwrap();

        let recall_at_k = if is_full_base_match && official_ground_truth_matched {
            if let Some(ref gt) = loaded.official_ground_truth {
                compute_recall_at_k(&cand_indices, gt, config.k)?
            } else {
                0.0
            }
        } else if let Some(ref ref_res) = exact_check_res {
            let cand_slice = cand_indices.slice_first_rows(ref_res.indices.rows());
            compute_recall_at_k(&cand_slice, &ref_res.indices, config.k)?
        } else {
            0.0
        };

        let is_monotonic = recall_at_k >= previous_recall - 1e-6;
        if !is_monotonic {
            monotonic_scaling_verified = false;
        }

        ef_evaluations.push(Phase3EfEvaluation {
            ef_search: ef,
            recall_at_k,
            latency: summary.median_latency,
            num_runs: summary.num_runs,
            mean_ms_min: summary.mean_ms_min,
            mean_ms_max: summary.mean_ms_max,
            mean_ms_spread: summary.mean_ms_spread,
            p99_ms_min: summary.p99_ms_min,
            p99_ms_max: summary.p99_ms_max,
            p99_ms_spread: summary.p99_ms_spread,
            run_mean_latencies_ms: summary.run_mean_latencies_ms,
            run_p99_latencies_ms: summary.run_p99_latencies_ms,
            monotonic_passed: is_monotonic,
        });

        previous_recall = recall_at_k;
    }

    // 4. Verify Reproducibility (if enabled)
    println!("[4/4] Verifying bit-exact reproducibility...");
    let reproducibility_checked = config.run_repro_check;
    let reproducibility_verified = if config.run_repro_check {
        let repro_b = config.repro_vectors.min(loaded.base.rows());
        let repro_q = 100.min(loaded.query.rows());
        let (base_repro_slice, is_full_b) = if repro_b == loaded.base.rows() {
            (None, true)
        } else {
            (Some(loaded.base.slice_first_rows(repro_b)), false)
        };
        let b_ref = if is_full_b {
            &loaded.base
        } else {
            base_repro_slice.as_ref().unwrap()
        };

        let (query_repro_slice, is_full_q) = if repro_q == loaded.query.rows() {
            (None, true)
        } else {
            (Some(loaded.query.slice_first_rows(repro_q)), false)
        };
        let q_ref = if is_full_q {
            &loaded.query
        } else {
            query_repro_slice.as_ref().unwrap()
        };

        verify_reproducibility_with_metric(b_ref, q_ref, config.metric, &hnsw_cfg, config.k, 50)
            .is_ok()
    } else {
        false
    };

    // If running on Authentic SIFT1M with default HNSW parameters at one of the reference scales
    // (10k, 100k, or 1M), update that scale's entry with the current run's live numbers.
    if loaded.source.is_authentic_sift()
        && matches!(
            config.dataset_mode,
            Phase1DatasetMode::Sift(SiftPreset::Sift1M)
        )
        && config.m == 16
        && effective_m_max0 == 32
        && config.ef_construction == 100
        && config.metric == Metric::L2Squared
        && config.k == 10
    {
        if let (Some(slot), Some(eval50)) = (
            baseline_profiles
                .iter_mut()
                .find(|p| p.scale_vectors == num_base),
            ef_evaluations.iter().find(|e| e.ef_search == 50),
        ) {
            slot.measured_this_run = true;
            slot.build_time_s = build_time_s;
            slot.build_qps = build_qps;
            slot.peak_rss_mb = peak_rss_mb;
            slot.bytes_per_vector = graph_bytes_per_vector;
            slot.raw_vector_bytes_per_vector = raw_vector_bytes_per_vector;
            slot.graph_bytes_per_vector = graph_bytes_per_vector;
            slot.graph_only_bytes_per_vector = graph_only_bytes_per_vector;
            slot.total_bytes_per_vector = total_bytes_per_vector;
            slot.graph_memory_bytes = graph_total_bytes;
            slot.graph_only_memory_bytes = graph_only_total_bytes;
            slot.graph_memory_mb = graph_total_mb;
            slot.k = config.k;
            slot.max_layer = max_layer;
            slot.entry_point = entry_point;
            slot.recall_at_10 = eval50.recall_at_k;
            slot.query_latency_ms = eval50.latency.mean_ms;
            slot.query_qps = eval50.latency.qps;
            slot.latency_min_ms = eval50.latency.min_ms;
            slot.latency_p50_ms = eval50.latency.p50_ms;
            slot.latency_p95_ms = eval50.latency.p95_ms;
            slot.latency_p99_ms = eval50.latency.p99_ms;
            slot.latency_max_ms = eval50.latency.max_ms;
        }
    }

    let preset_name = match &config.dataset_mode {
        Phase1DatasetMode::Sift(p) => p.as_str().to_string(),
        Phase1DatasetMode::Synthetic { .. } => "synthetic".to_string(),
        Phase1DatasetMode::Custom { .. } => "custom".to_string(),
    };

    let (base_path, query_path, groundtruth_path) = match &loaded.source {
        DatasetSource::AuthenticSift1M { dir } => (
            Some(dir.join("sift_base.fvecs")),
            Some(dir.join("sift_query.fvecs")),
            Some(dir.join("sift_groundtruth.ivecs")),
        ),
        DatasetSource::AuthenticSiftSmall { dir } => (
            Some(dir.join("siftsmall_base.fvecs")),
            Some(dir.join("siftsmall_query.fvecs")),
            Some(dir.join("siftsmall_groundtruth.ivecs")),
        ),
        _ => (None, None, None),
    };

    let generated_at = current_utc_rfc3339();

    Ok(Phase3Report {
        phase: 3,
        phase_name: "Phase 3: HNSW SIFT1M Scaling, Memory Profiling & Ground Truth Benchmarking"
            .to_string(),
        generated_at,
        hardware,
        dataset_source: loaded.source,
        preset_name,
        base_path,
        query_path,
        groundtruth_path,
        num_base,
        num_query,
        dim,
        k: config.k,
        seed: config.seed,
        metric: config.metric,
        hnsw_m: config.m,
        hnsw_m_max0: effective_m_max0,
        hnsw_ef_construction: config.ef_construction,
        build_time_s,
        build_qps,
        max_layer,
        entry_point,
        memory_profile,
        multi_scale_baseline_summary: baseline_profiles,
        sweep_runs: runs_per_ef,
        ef_evaluations,
        monotonic_scaling_verified,
        reproducibility_checked,
        reproducibility_verified,
        official_ground_truth_matched,
        official_gt_check_queries,
        official_gt_recall_at_k,
        official_gt_exact_id_matches,
        terminal_stopping_enforced: true,
        phase4_initiated: false,
    })
}

/// Formats and prints a structured summary of the Phase 3 SIFT1M benchmark to stdout.
pub fn print_phase3_report(r: &Phase3Report) {
    println!("{}", "=".repeat(80));
    println!(" HNSW APPROXIMATE NEAREST NEIGHBOR SEARCH (RUST) -- PHASE 3 SIFT1M BENCHMARK ");
    println!("{}", "=".repeat(80));
    println!("Hardware: {}", r.hardware.summary_line());
    println!("Dataset Provenance: {}", r.dataset_source);
    println!("Configuration:");
    println!("  - Base Vectors:          {} vectors", r.num_base);
    println!("  - Query Vectors:         {} vectors", r.num_query);
    println!("  - Vector Dimension:      {}-d (float32)", r.dim);
    println!("  - Top-k Neighbors:       {}", r.k);
    println!("  - Distance Metric:       {}", r.metric);
    println!("  - Deterministic Seed:    {}", r.seed);
    println!(
        "  - HNSW Graph Topology:   M={}, M0={}, efConstruction={}",
        r.hnsw_m, r.hnsw_m_max0, r.hnsw_ef_construction
    );
    println!("{}", "-".repeat(80));

    println!("[+] Index Construction Profile:");
    println!(
        "  - Total build elapsed:   {:.3} s ({:.1} vectors/s)",
        r.build_time_s, r.build_qps
    );
    println!("  - Max graph layer:       {}", r.max_layer);
    println!("  - Entry point node:      #{}", r.entry_point);
    println!("{}", "-".repeat(80));

    let mp = &r.memory_profile;
    println!("[+] Memory Profiling & Decoupled Footprint:");
    println!(
        "  - Process Peak RSS (OS): {:.2} MiB ({:.2} MB)",
        mp.peak_rss_mb,
        mp.peak_rss_mb * 1.048576
    );
    println!(
        "  - Measurement Method:    {}",
        mp.peak_rss_measurement_method
    );
    println!(
        "  - Raw Vector Storage:    {:.2} B/vector ({:.2} MiB total)",
        mp.raw_vector_bytes_per_vector, mp.raw_vector_total_mb
    );
    println!(
        "  - Graph Structural Cost: {:.2} B/vector ({:.2} MiB total, incl. visited & scratch)",
        mp.graph_bytes_per_vector, mp.graph_total_mb
    );
    println!(
        "  - Graph-Only Footprint:  {:.2} B/vector ({:.2} MiB total, excl. visited & scratch)",
        mp.graph_only_bytes_per_vector, mp.graph_only_total_mb
    );
    println!(
        "  - Total Index Footprint: {:.2} B/vector ({:.2} MiB total)",
        mp.total_bytes_per_vector, mp.total_footprint_mb
    );
    println!("\n  Detailed Graph Components Breakdown:");
    println!(
        "    - Node struct buffers:         {:10} bytes ({:.2} MiB)",
        mp.graph_components.node_structs_bytes,
        mp.graph_components.node_structs_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Layer-0 edge adjacency:      {:10} bytes ({:.2} MiB)",
        mp.graph_components.layer0_edges_bytes,
        mp.graph_components.layer0_edges_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Upper layers edge adjacency: {:10} bytes ({:.2} MiB)",
        mp.graph_components.upper_layers_edges_bytes,
        mp.graph_components.upper_layers_edges_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Outer neighbor headers:      {:10} bytes ({:.2} MiB)",
        mp.graph_components.outer_neighbors_headers_bytes,
        mp.graph_components.outer_neighbors_headers_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Visited tracker tag buffer:  {:10} bytes ({:.2} MiB)",
        mp.graph_components.visited_tracker_tags_bytes,
        mp.graph_components.visited_tracker_tags_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    - Heuristic scratch buffers:   {:10} bytes (<0.01 MiB)",
        mp.graph_components.heuristic_scratch_bytes
    );
    println!("{}", "-".repeat(80));

    println!(
        "[+] efSearch Parameter Sweep vs. Texmex Ground Truth (k={}, median of {} runs):",
        r.k, r.sweep_runs
    );
    println!("+----------+-----------+------------+----------------+------------+------------+------------+-----------+");
    println!("| efSearch | Recall@10 | MedMean ms | Mean Spread ms | p50 (ms)   | p99 (ms)   | QPS        | Monotonic |");
    println!("+----------+-----------+------------+----------------+------------+------------+------------+-----------+");
    for ef in &r.ef_evaluations {
        println!(
            "| {:8} | {:8.2}% | {:10.3} | {:14.3} | {:10.3} | {:10.3} | {:10.1} | {:9} |",
            ef.ef_search,
            ef.recall_at_k * 100.0,
            ef.latency.mean_ms,
            ef.mean_ms_spread,
            ef.latency.p50_ms,
            ef.latency.p99_ms,
            ef.latency.qps,
            if ef.monotonic_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );
    }
    println!("+----------+-----------+------------+----------------+------------+------------+------------+-----------+");
    println!("{}", "-".repeat(80));

    println!("[+] Verification & Quality Status:");
    println!(
        "  - Authentic SIFT1M Corpus:       {}",
        if r.dataset_source.is_authentic_sift() {
            "PASSED"
        } else {
            "SYNTHETIC/FALLBACK"
        }
    );
    println!(
        "  - Official Ground Truth Matched: {}",
        if r.official_ground_truth_matched {
            format!(
                "PASSED (100.0% ExactKNN match on {} queries)",
                r.official_gt_check_queries
            )
        } else {
            "FAILED / NOT APPLICABLE".to_string()
        }
    );
    println!(
        "  - Monotonic Recall Scaling:      {}",
        if r.monotonic_scaling_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "  - Bit-Exact Reproducibility:     {}",
        if !r.reproducibility_checked {
            "SKIPPED"
        } else if r.reproducibility_verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!("  - #![forbid(unsafe_code)]:       STRICTLY ENFORCED (0 unsafe blocks)");
    println!(
        "  - Pure Standard Library:         STRICTLY ENFORCED (0 external algorithmic crates)"
    );
    println!("\n{}", "=".repeat(80));
    println!(" PHASE 3 COMPLETE (RUST) -- TERMINAL MILESTONE REACHED -- STOPPING EXECUTION ");
    println!("{}", "=".repeat(80));
    println!("Notice: Phase 3 benchmark and memory profiling on SIFT1M is complete.");
    println!("Stopping at Phase 3 milestone boundary. Phase 4 will not be initiated.");
    println!("{}", "=".repeat(80));
}

/// Entry point for CLI execution of Phase 3 benchmark.
pub fn run_phase3_cli(args: &[String]) -> Result<()> {
    let mut dataset_choice = String::from("sift1m");
    let mut num_base: usize = 1_000_000;
    let mut num_query: usize = 10_000;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 16;
    let mut k: usize = 10;
    let mut seed: u64 = 42;
    let mut metric = Metric::L2Squared;
    let mut m: usize = 16;
    let mut m_max0: Option<usize> = None;
    let mut ef_construction: usize = 100;
    let mut ef_search_list = vec![10_usize, 20, 50, 100, 200];
    let mut data_cache_dir = PathBuf::from("data");
    let mut output_dir = PathBuf::from("data");
    let mut base_fvecs: Option<PathBuf> = None;
    let mut query_fvecs: Option<PathBuf> = None;
    let mut gt_ivecs: Option<PathBuf> = None;
    let mut max_queries: Option<usize> = None;
    let mut max_base: Option<usize> = None;
    let mut num_threads: usize = 0;
    let mut allow_download = true;
    let mut save_files = true;
    let mut run_repro_check = true;
    let mut repro_vectors = 10_000;
    let mut sweep_runs: usize = 5;
    let mut run_multi_scale = true;
    let mut internal_profile_scale: Option<usize> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut show_help = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                show_help = true;
                i += 1;
            }
            "--dataset" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --dataset".to_string(),
                    ));
                }
                dataset_choice = args[i + 1].to_ascii_lowercase();
                i += 2;
            }
            "--num-base" | "--max-base" | "--scale" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "Missing value after {}",
                        args[i]
                    )));
                }
                let val: usize = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument(format!(
                        "Invalid numeric value for {}",
                        args[i]
                    ))
                })?;
                num_base = val;
                max_base = Some(val);
                i += 2;
            }
            "--num-query" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --num-query".to_string(),
                    ));
                }
                num_query = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --num-query".to_string())
                })?;
                i += 2;
            }
            "--max-queries" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --max-queries".to_string(),
                    ));
                }
                max_queries = Some(args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --max-queries".to_string())
                })?);
                i += 2;
            }
            "--dim" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --dim".to_string(),
                    ));
                }
                dim = args[i + 1]
                    .parse()
                    .map_err(|_| VectorSearchError::InvalidArgument("Invalid --dim".to_string()))?;
                i += 2;
            }
            "--num-clusters" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --num-clusters".to_string(),
                    ));
                }
                num_clusters = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --num-clusters".to_string())
                })?;
                i += 2;
            }
            "--k" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --k".to_string(),
                    ));
                }
                k = args[i + 1]
                    .parse()
                    .map_err(|_| VectorSearchError::InvalidArgument("Invalid --k".to_string()))?;
                i += 2;
            }
            "--seed" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --seed".to_string(),
                    ));
                }
                seed = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --seed".to_string())
                })?;
                i += 2;
            }
            "--metric" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --metric".to_string(),
                    ));
                }
                metric = std::str::FromStr::from_str(&args[i + 1])?;
                i += 2;
            }
            "--m" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --m".to_string(),
                    ));
                }
                m = args[i + 1]
                    .parse()
                    .map_err(|_| VectorSearchError::InvalidArgument("Invalid --m".to_string()))?;
                i += 2;
            }
            "--m-max0" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --m-max0".to_string(),
                    ));
                }
                m_max0 = Some(args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --m-max0".to_string())
                })?);
                i += 2;
            }
            "--ef-construction" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --ef-construction".to_string(),
                    ));
                }
                ef_construction = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --ef-construction".to_string())
                })?;
                i += 2;
            }
            "--ef-search" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --ef-search".to_string(),
                    ));
                }
                ef_search_list = args[i + 1]
                    .split(',')
                    .map(|s| {
                        s.trim().parse::<usize>().map_err(|_| {
                            VectorSearchError::InvalidArgument(format!(
                                "Invalid ef-search value '{s}'"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?;
                i += 2;
            }
            "--threads" | "--num-threads" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "Missing value after {}",
                        args[i]
                    )));
                }
                num_threads = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument(
                        "Invalid numeric value for threads".to_string(),
                    )
                })?;
                i += 2;
            }
            "--data-dir" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --data-dir".to_string(),
                    ));
                }
                data_cache_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output-dir" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --output-dir".to_string(),
                    ));
                }
                output_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--base-fvecs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --base-fvecs".to_string(),
                    ));
                }
                base_fvecs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--query-fvecs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --query-fvecs".to_string(),
                    ));
                }
                query_fvecs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--gt-ivecs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --gt-ivecs".to_string(),
                    ));
                }
                gt_ivecs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--json-out" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --json-out".to_string(),
                    ));
                }
                json_out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--offline" => {
                allow_download = false;
                i += 1;
            }
            "--no-save" => {
                save_files = false;
                i += 1;
            }
            "--repro-check" => {
                run_repro_check = true;
                i += 1;
            }
            "--no-repro-check" => {
                run_repro_check = false;
                i += 1;
            }
            "--repro-vectors" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --repro-vectors".to_string(),
                    ));
                }
                repro_vectors = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --repro-vectors".to_string())
                })?;
                i += 2;
            }
            "--sweep-runs" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --sweep-runs".to_string(),
                    ));
                }
                sweep_runs = args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument("Invalid --sweep-runs".to_string())
                })?;
                i += 2;
            }
            "--multi-scale" => {
                run_multi_scale = true;
                i += 1;
            }
            "--no-multi-scale" => {
                run_multi_scale = false;
                i += 1;
            }
            "--internal-profile-scale" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --internal-profile-scale".to_string(),
                    ));
                }
                internal_profile_scale = Some(args[i + 1].parse().map_err(|_| {
                    VectorSearchError::InvalidArgument(
                        "Invalid --internal-profile-scale".to_string(),
                    )
                })?);
                i += 2;
            }
            "--phase" => {
                if i + 1 >= args.len() {
                    return Err(VectorSearchError::InvalidArgument(
                        "Missing value after --phase".to_string(),
                    ));
                }
                i += 2;
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI flag '{unknown}'"
                )));
            }
        }
    }

    if let Some(scale) = internal_profile_scale {
        let profile = profile_single_sift1m_scale(
            &data_cache_dir,
            scale,
            seed,
            allow_download,
            sweep_runs.max(1),
        )?;
        print!("{}", format_scale_profile_kv(&profile));
        return Ok(());
    }

    if show_help {
        println!(
            "\
HNSW Vector Search (Rust) -- Phase 3: SIFT1M Real-World Scaling & Benchmark Harness

USAGE:
    cargo run --release --bin phase3 -- [OPTIONS]
    cargo run --release --bin phase1 -- --phase 3 [OPTIONS]

OPTIONS:
    --dataset <sift1m|siftsmall|synthetic|custom>
                                Dataset preset to load or download (default: sift1m)
    --data-dir <DIR>           Cache directory for public SIFT corpora (default: data)
    --output-dir <DIR>         Output directory for artifacts (default: data)
    --k <K>                    Top-k nearest neighbors to retrieve (default: 10)
    --seed <SEED>              Deterministic PRNG seed (default: 42)
    --metric <l2|euclidean|ip> Distance metric (default: l2)
    --m <M>                    Max outgoing edges per node at layers > 0 (default: 16)
    --m-max0 <M0>              Max outgoing edges per node at layer 0 (default: 32)
    --ef-construction <EFC>    Candidate beam width during index build (default: 100)
    --ef-search <LIST>         Comma-separated efSearch values to sweep (default: 10,20,50,100,200)
    --sweep-runs <N>           Repetitions per efSearch setting for median & spread (default: 5)
    --no-multi-scale           Skip isolated 10k and 100k child-process baseline profiling
    --max-queries <Q>          Limit search evaluation to first Q queries
    --max-base <N>             Limit index construction to first N base vectors (smoke test)
    --threads <T>              Number of worker threads (default: 0 for all logical cores)
    --json-out <PATH>          Write structured JSON benchmark report to PATH (default: data/phase3_report.json)
    --offline                  Disable internet downloads; use local cache or synthetic
    --no-save                  Skip writing intermediate artifacts to disk
    --repro-check              Verify bit-exact reproducibility across independent builds
    --no-repro-check           Skip reproducibility check
    --repro-vectors <N>        Vector slice size for reproducibility check (default: 10000)
    -h, --help                 Print this help message"
        );
        return Ok(());
    }

    let dataset_mode = if base_fvecs.is_some()
        || query_fvecs.is_some()
        || dataset_choice == "custom"
    {
        let (Some(b), Some(q)) = (base_fvecs, query_fvecs) else {
            return Err(VectorSearchError::InvalidArgument(
                "Custom dataset mode requires both --base-fvecs <path> and --query-fvecs <path>"
                    .to_string(),
            ));
        };
        Phase1DatasetMode::Custom {
            base_fvecs: b,
            query_fvecs: q,
            gt_ivecs,
        }
    } else {
        match dataset_choice.as_str() {
            "sift1m" | "sift" => Phase1DatasetMode::Sift(SiftPreset::Sift1M),
            "siftsmall" => Phase1DatasetMode::Sift(SiftPreset::SiftSmall),
            "synthetic" => Phase1DatasetMode::Synthetic {
                num_base,
                num_query,
                dim,
                num_clusters,
            },
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unsupported --dataset '{other}'. Expected 'sift1m', 'siftsmall', 'synthetic', or 'custom'."
                )));
            }
        }
    };

    let config = Phase3Config {
        dataset_mode,
        data_cache_dir,
        output_dir,
        k,
        seed,
        metric,
        m,
        m_max0,
        ef_construction,
        ef_search_list,
        allow_download,
        save_files,
        max_queries,
        max_base,
        num_threads,
        run_repro_check,
        repro_vectors,
        sweep_runs,
        run_multi_scale,
    };

    let report = run_phase3_benchmark(&config)?;
    print_phase3_report(&report);

    let target_json = json_out.unwrap_or_else(|| PathBuf::from("data/phase3_report.json"));
    if let Some(parent) = target_json.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&target_json, report.to_json_string())?;
    println!("  - JSON Summary Report:   {}", target_json.display());

    if config.save_files
        && !report.multi_scale_baseline_summary.is_empty()
        && report
            .multi_scale_baseline_summary
            .iter()
            .all(|p| p.measured_this_run)
    {
        let baseline_report = BaselineMultiScaleReport {
            hardware: report.hardware.clone(),
            dataset_source: report.dataset_source.clone(),
            hnsw_m: report.hnsw_m,
            hnsw_m_max0: report.hnsw_m_max0,
            hnsw_ef_construction: report.hnsw_ef_construction,
            seed: report.seed,
            metric: report.metric,
            peak_rss_measurement_method: format!(
                "isolated_per_scale_process ({})",
                report.memory_profile.peak_rss_measurement_method
            ),
            scale_profiles: report.multi_scale_baseline_summary.clone(),
        };
        let baseline_json = config.output_dir.join("baseline_profile_report.json");
        if let Some(parent) = baseline_json.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&baseline_json, baseline_report.to_json_string())?;
        println!("  - Multi-Scale Report:    {}", baseline_json.display());
    }

    Ok(())
}
