//! Phase 1 Benchmark Pipeline: Dataset Loading, Brute-Force Ground Truth,
//! Latency Percentiles (`p50`, `p95`, `p99`, `QPS`), Recall Verification, and Reporting.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use crate::dataset::{write_fvecs, write_ivecs, DatasetSource, SiftPreset};
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hardware::{detect_hardware, HardwareInfo};
use crate::json_util::{escape_json, opt_path_json};
use crate::recall::{
    compute_recall_at_k, evaluate_recall_summary, verify_self_recall, LatencyStats,
};

use super::dataset_loader::{load_dataset_and_slice_queries, Phase1DatasetMode};
use super::detect_provenance;

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
        detect_provenance().append_json_fields(&mut out, &self.hardware);

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
            escape_json(self.dataset_source.kind_tag())
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
        out.push_str(&format!(
            "    \"metric\": \"{}\",\n",
            escape_json(&self.metric.to_string())
        ));
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
    let loaded = load_dataset_and_slice_queries(
        &config.dataset_mode,
        &config.data_cache_dir,
        config.allow_download,
        config.seed,
        config.max_queries,
    )?;

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
