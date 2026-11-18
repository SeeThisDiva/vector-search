//! Isolated child-process query-serving `PeakWorkingSet64` / RSS profiler (`--internal-profile-query`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::benchmark::detect_provenance;
use crate::dataset::{read_fvecs_max_rows, write_fvecs};
use crate::error::{Result, VectorSearchError};
use crate::hardware::get_peak_rss_mb;
use crate::matrix::RowMatrix;
use crate::quantized::{HnswPqIndex, HnswSq8Index};
use crate::rng::DeterministicRng;

use super::config::{apply_affinity_pinning, Phase6InternalQueryProfileConfig, SnapshotKind};
use super::runner::QuantizedSweepTarget;

/// Executes inside a fresh child OS process: loads ONLY the serialized quantized snapshot and
/// query vectors (never `RowMatrix<f32>`), runs 1 warmup + 1 query batch at `ef`, and prints
/// `query_serving_peak_rss_mb`.
pub fn run_phase6_internal_profile_query(config: &Phase6InternalQueryProfileConfig) -> Result<()> {
    if config.k == 0 || config.ef == 0 || config.num_queries == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "--k, --ef, and --queries must be positive (> 0)".to_string(),
        ));
    }
    let mut prov = detect_provenance();
    apply_affinity_pinning(config.affinity.as_deref(), &mut prov)?;

    match config.kind {
        SnapshotKind::Sq8 => {
            let index = HnswSq8Index::read_snapshot(&config.snapshot_path)?;
            run_isolated_query_profile_for(&index, config)?;
        }
        SnapshotKind::Pq => {
            let index = HnswPqIndex::read_snapshot(&config.snapshot_path)?;
            run_isolated_query_profile_for(&index, config)?;
        }
    }
    Ok(())
}

fn run_isolated_query_profile_for<T: QuantizedSweepTarget>(
    index: &T,
    config: &Phase6InternalQueryProfileConfig,
) -> Result<()> {
    let queries = load_isolated_queries(config, index.q_dim())?;
    let mut scratch = index.q_new_scratch(config.ef.max(config.k));
    index.q_warmup(queries.row(0), config.k, config.ef, &mut scratch)?;
    let res = index.q_search_batch_timed(&queries, config.k, config.ef, None, &mut scratch)?;
    std::hint::black_box(&res.results);
    let peak_rss = get_peak_rss_mb();
    print_query_profile_kv(
        index.q_num_vectors(),
        index.q_dim(),
        queries.rows(),
        config.ef,
        index.q_graph_fingerprint(),
        peak_rss,
    );
    Ok(())
}

fn load_isolated_queries(
    config: &Phase6InternalQueryProfileConfig,
    expected_dim: usize,
) -> Result<RowMatrix<f32>> {
    let queries = if let Some(ref q_path) = config.query_fvecs {
        read_fvecs_max_rows(q_path, Some(config.num_queries))?
    } else {
        // Synthetic query-only generation (never allocates base RowMatrix<f32>)
        let mut rng = DeterministicRng::new(config.seed ^ 0x517C_C1B7_2722_0A95);
        let total = config
            .num_queries
            .checked_mul(expected_dim)
            .ok_or_else(|| {
                VectorSearchError::InvalidArgument(
                    "Synthetic query buffer overflows usize".to_string(),
                )
            })?;
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            let u = rng.next_f64_open01() * 200.0 - 100.0;
            data.push(u as f32);
        }
        RowMatrix::new(config.num_queries, expected_dim, data)?
    };
    if queries.rows() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Query matrix for --internal-profile-query cannot be empty".to_string(),
        ));
    }
    if queries.cols() != expected_dim {
        return Err(VectorSearchError::DimensionMismatch {
            expected: expected_dim,
            actual: queries.cols(),
        });
    }
    Ok(queries)
}

fn print_query_profile_kv(
    num_vectors: usize,
    dim: usize,
    num_queries: usize,
    ef: usize,
    graph_fingerprint: u64,
    peak_rss_mb: Option<f64>,
) {
    println!("PHASE6_QUERY_PROFILE_BEGIN");
    println!("num_vectors={num_vectors}");
    println!("dim={dim}");
    println!("num_queries={num_queries}");
    println!("ef={ef}");
    println!("graph_fingerprint_hex=0x{graph_fingerprint:016x}");
    match peak_rss_mb {
        Some(mb) => println!("query_serving_peak_rss_mb={mb:.4}"),
        None => println!("query_serving_peak_rss_mb=null"),
    }
    println!("PHASE6_QUERY_PROFILE_END");
}

fn parse_query_profile_kv(stdout: &str) -> Option<f64> {
    let mut in_block = false;
    let mut found_end = false;
    let mut rss: Option<f64> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed == "PHASE6_QUERY_PROFILE_BEGIN" {
            in_block = true;
            continue;
        }
        if trimmed == "PHASE6_QUERY_PROFILE_END" {
            found_end = true;
            break;
        }
        if !in_block {
            continue;
        }
        if let Some((k, v)) = trimmed.split_once('=') {
            if k == "query_serving_peak_rss_mb" && v != "null" {
                rss = v.parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0);
            }
        }
    }
    if found_end {
        rss
    } else {
        None
    }
}

struct TempFileCleanup {
    paths: Vec<PathBuf>,
    tmp_dir: PathBuf,
}

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = fs::remove_file(p);
        }
        let _ = fs::remove_dir(&self.tmp_dir);
    }
}

fn resolve_phase6_executable() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let stem = exe.file_stem()?.to_string_lossy().to_ascii_lowercase();
    if stem == "phase6" {
        return Some(exe);
    }
    let bin_name = if cfg!(windows) {
        "phase6.exe"
    } else {
        "phase6"
    };
    if let Some(parent) = exe.parent() {
        let direct = parent.join(bin_name);
        if direct.is_file() {
            return Some(direct);
        }
        if let Some(grandparent) = parent.parent() {
            let sibling = grandparent.join(bin_name);
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_isolated_query_rss<F>(
    kind: SnapshotKind,
    arm_id: &str,
    scale: usize,
    k: usize,
    data_dir: &Path,
    query_fvecs_path: Option<&Path>,
    queries: &RowMatrix<f32>,
    affinity: Option<&str>,
    write_snapshot_fn: F,
) -> Result<(Option<f64>, bool)>
where
    F: FnOnce(&Path) -> Result<()>,
{
    let Some(phase6_exe) = resolve_phase6_executable() else {
        return Ok((None, false));
    };

    let tmp_dir = data_dir.join("phase6_tmp");
    fs::create_dir_all(&tmp_dir)?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let snapshot_path = tmp_dir.join(format!("snapshot_n{scale}_{arm_id}_{pid}_{nanos}.bin"));
    let mut cleanup = TempFileCleanup {
        paths: vec![snapshot_path.clone()],
        tmp_dir,
    };

    write_snapshot_fn(&snapshot_path)?;

    let effective_query_fvecs = if let Some(q_path) = query_fvecs_path {
        q_path.to_path_buf()
    } else {
        let tmp_q = cleanup
            .tmp_dir
            .join(format!("queries_n{scale}_{arm_id}_{pid}_{nanos}.fvecs"));
        cleanup.paths.push(tmp_q.clone());
        write_fvecs(&tmp_q, queries)?;
        tmp_q
    };

    let mut cmd = Command::new(&phase6_exe);
    cmd.arg("--internal-profile-query")
        .arg(&snapshot_path)
        .arg("--kind")
        .arg(kind.as_str())
        .arg("--query-fvecs")
        .arg(&effective_query_fvecs)
        .arg("--queries")
        .arg(queries.rows().to_string())
        .arg("--k")
        .arg(k.to_string())
        .arg("--ef")
        .arg("100");
    if let Some(mask) = affinity {
        cmd.arg("--affinity").arg(mask);
    }

    let output = cmd.output().map_err(VectorSearchError::Io)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(VectorSearchError::VerificationFailed(format!(
            "Isolated query RSS child process failed for arm '{arm_id}': {}",
            stderr.trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed_rss = parse_query_profile_kv(&stdout);
    Ok((parsed_rss, true))
}
