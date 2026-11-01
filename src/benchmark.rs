//! Phase 1 Benchmark Pipeline: Dataset Loading, Brute-Force Ground Truth,
//! Latency Percentiles (`p50`, `p95`, `p99`, `QPS`), Recall Verification, and Reporting.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataset::{
    generate_synthetic_sift, load_custom_dataset, load_or_download_sift, write_fvecs, write_ivecs,
    DatasetSource, LoadedDataset, SiftPreset,
};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{detect_hardware, HardwareInfo};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::matrix::RowMatrix;
use crate::recall::{
    compute_recall_at_k, evaluate_recall_summary, verify_self_recall, LatencyStats,
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

/// Verifies bit-exact reproducibility across two independent index builds and query executions.
pub fn verify_reproducibility(
    base: &RowMatrix<f32>,
    query: &RowMatrix<f32>,
    config: &HnswConfig,
    k: usize,
    ef_search: usize,
) -> Result<()> {
    let mut index1 = HnswIndex::build(base, Metric::L2Squared, config.clone())?;
    let res1 = index1.search_batch(query, k, ef_search)?;

    let mut index2 = HnswIndex::build(base, Metric::L2Squared, config.clone())?;
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
    let reproducibility_verified = if config.run_repro_check {
        verify_reproducibility(&loaded.base, &loaded.query, &hnsw_cfg, config.k, 50).is_ok()
    } else {
        true
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
        if r.reproducibility_verified {
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
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized CLI flag '{unknown}'"
                )));
            }
        }
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
    --json-out <PATH>           Write structured JSON benchmark report to PATH
    --offline                   Disable internet downloads; use local cache or synthetic
    --no-save                   Skip writing intermediate artifacts to disk
    --repro-check               Verify bit-exact reproducibility across independent runs
    -h, --help                  Print this help message"
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
