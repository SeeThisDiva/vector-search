//! Deterministic Metadata Generator CLI (Milestone M3 / Phase 5: Task T3.2).
//!
//! Generates synthetic metadata bitsets for filtered vector search benchmarks:
//! - Mode A: Independent (Bernoulli) with exact popcount $\lfloor \sigma \cdot N \rfloor$.
//! - Mode B: Cluster-Correlated via deterministic $K$-means++ with exact shortfall calibration.
//!
//! Supports structured binary bitset output (`.bin`), Texmex-compatible `.ivecs`,
//! and clean provenance JSON reporting git commit, parameters, cluster IDs, and popcount.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use vector_search::dataset::{
    generate_mode_a_bitset, generate_mode_b_bitset_detailed, generate_synthetic_sift, read_fvecs,
    write_bitset, write_bitset_as_ivecs, ModeBMetadata, S_CLUST_DEFAULT,
};
use vector_search::error::{Result, VectorSearchError};
use vector_search::filter::DenseBitset;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationMode {
    Independent,
    Correlated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    RawBinary,
    Ivecs,
}

struct CliConfig {
    mode: GenerationMode,
    num_base: usize,
    selectivity: f64,
    seed_attr: u64,
    seed_clust: u64,
    base_fvecs: Option<PathBuf>,
    dim: usize,
    num_clusters: usize,
    output_path: Option<PathBuf>,
    format: OutputFormat,
    json_path: Option<PathBuf>,
}

fn print_usage() {
    eprintln!(
        r#"Deterministic Metadata Generator (Milestone M3 Task T3.2)

USAGE:
    generate_metadata [OPTIONS]

OPTIONS:
    --mode <independent|correlated>
                        Metadata distribution mode:
                          'independent' (Mode A): uniform Bernoulli
                          'correlated'  (Mode B): cluster-correlated (default: independent)
    --num-base <N>, -n <N>
                        Number of base vectors N (default: 10000)
    --selectivity <S>, -s <S>
                        Filter selectivity in [0.0, 1.0], e.g. 0.1, 0.01, 10% (default: 0.1)
    --seed <SEED>       PRNG seed for attribute generation (default: 42)
    --seed-clust <SEED> PRNG seed for Mode B centroid selection (default: 0xCBF29CE484222325)
    --base-fvecs <PATH> Path to existing base vectors .fvecs file (Mode B)
    --dim <D>           Vector dimension if generating synthetic base vectors (default: 128)
    --num-clusters <C>  Clusters for synthetic base vectors if needed (default: 100)
    --output <PATH>, -o <PATH>
                        Destination path for output bitset (.bin or .ivecs)
    --format <raw|ivecs>
                        Bitset serialization format: 'raw' (.bin) or 'ivecs' (default: raw)
    --json-out <PATH>   Destination path for provenance JSON report
    -h, --help          Print this help message
"#
    );
}

fn parse_cli_args() -> Result<Option<CliConfig>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(None);
    }

    let mut mode = GenerationMode::Independent;
    let mut num_base: usize = 10_000;
    let mut selectivity: f64 = 0.1;
    let mut seed_attr: u64 = 42;
    let mut seed_clust: u64 = S_CLUST_DEFAULT;
    let mut base_fvecs: Option<PathBuf> = None;
    let mut dim: usize = 128;
    let mut num_clusters: usize = 100;
    let mut output_path: Option<PathBuf> = None;
    let mut format = OutputFormat::RawBinary;
    let mut json_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        let mut next_val = || -> Result<&str> {
            i += 1;
            if i >= args.len() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Missing argument value for flag '{flag}'"
                )));
            }
            Ok(&args[i])
        };

        match flag.as_str() {
            "--mode" => {
                let m = next_val()?.trim().to_ascii_lowercase();
                mode = match m.as_str() {
                    "independent" | "mode_a" | "a" | "uniform" => GenerationMode::Independent,
                    "correlated" | "mode_b" | "b" | "cluster" => GenerationMode::Correlated,
                    other => {
                        return Err(VectorSearchError::InvalidArgument(format!(
                            "Invalid mode '{other}'. Expected 'independent' or 'correlated'."
                        )))
                    }
                };
            }
            "--num-base" | "-n" | "--num-vectors" => {
                let val_str = next_val()?;
                num_base = val_str.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Invalid --num-base '{val_str}': {e}"
                    ))
                })?;
            }
            "--selectivity" | "-s" | "--sel" => {
                let mut val_str = next_val()?.trim().to_string();
                let is_pct = val_str.ends_with('%');
                if is_pct {
                    val_str.pop();
                }
                let mut s: f64 = val_str.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Invalid --selectivity '{val_str}': {e}"
                    ))
                })?;
                if is_pct {
                    s /= 100.0;
                }
                if !(0.0..=1.0).contains(&s) || s.is_nan() {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "Selectivity must be in [0.0, 1.0], got {s}"
                    )));
                }
                selectivity = s;
            }
            "--seed" => {
                let val_str = next_val()?;
                seed_attr = val_str.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --seed '{val_str}': {e}"))
                })?;
            }
            "--seed-clust" => {
                let val_str = next_val()?;
                seed_clust = if val_str.starts_with("0x") || val_str.starts_with("0X") {
                    u64::from_str_radix(&val_str[2..], 16)
                } else {
                    val_str.parse()
                }
                .map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Invalid --seed-clust '{val_str}': {e}"
                    ))
                })?;
            }
            "--base-fvecs" => {
                base_fvecs = Some(PathBuf::from(next_val()?));
            }
            "--dim" => {
                let val_str = next_val()?;
                dim = val_str.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!("Invalid --dim '{val_str}': {e}"))
                })?;
            }
            "--num-clusters" => {
                let val_str = next_val()?;
                num_clusters = val_str.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Invalid --num-clusters '{val_str}': {e}"
                    ))
                })?;
            }
            "--output" | "-o" | "--out" => {
                let p = PathBuf::from(next_val()?);
                if p.extension().is_some_and(|ext| ext == "ivecs") {
                    format = OutputFormat::Ivecs;
                }
                output_path = Some(p);
            }
            "--format" => {
                let fmt_str = next_val()?.trim().to_ascii_lowercase();
                format = match fmt_str.as_str() {
                    "raw" | "bin" | "binary" => OutputFormat::RawBinary,
                    "ivecs" | "texmex" => OutputFormat::Ivecs,
                    other => {
                        return Err(VectorSearchError::InvalidArgument(format!(
                            "Invalid format '{other}'. Expected 'raw' or 'ivecs'."
                        )))
                    }
                };
            }
            "--json-out" => {
                json_path = Some(PathBuf::from(next_val()?));
            }
            other => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Unrecognized argument '{other}'. Run with --help for usage."
                )))
            }
        }
        i += 1;
    }

    Ok(Some(CliConfig {
        mode,
        num_base,
        selectivity,
        seed_attr,
        seed_clust,
        base_fvecs,
        dim,
        num_clusters,
        output_path,
        format,
        json_path,
    }))
}

fn get_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn run() -> Result<()> {
    let Some(config) = parse_cli_args()? else {
        return Ok(());
    };

    let git_commit = get_git_commit();
    let num_vectors = config.num_base;
    let target_popcount = (config.selectivity * num_vectors as f64).floor() as usize;

    println!("============================================================");
    println!("Deterministic Metadata Generator (Milestone M3 Task T3.2)");
    println!("============================================================");
    println!("Git Commit:       {git_commit}");
    println!("Mode:             {:?}", config.mode);
    println!("Num Vectors N:    {}", config.num_base);
    println!(
        "Selectivity:      {:.6} ({:.2}%)",
        config.selectivity,
        config.selectivity * 100.0
    );
    println!("Target Popcount:  {target_popcount}");
    println!("Seed (Attr):      {}", config.seed_attr);

    let (bitset, mode_b_meta): (DenseBitset, Option<ModeBMetadata>) = match config.mode {
        GenerationMode::Independent => {
            let bs = generate_mode_a_bitset(config.num_base, config.selectivity, config.seed_attr);
            (bs, None)
        }
        GenerationMode::Correlated => {
            println!("Seed (Centroids): 0x{:016X}", config.seed_clust);
            let base_matrix = if let Some(ref path) = config.base_fvecs {
                println!("Loading base vectors from {}...", path.display());
                let loaded = read_fvecs(path)?;
                if loaded.rows() < config.num_base {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "File {} contains {} vectors, but --num-base is {}",
                        path.display(),
                        loaded.rows(),
                        config.num_base
                    )));
                }
                if loaded.rows() > config.num_base {
                    loaded.slice_first_rows(config.num_base)
                } else {
                    loaded
                }
            } else {
                println!(
                    "Generating synthetic base vectors (N={}, dim={}, clusters={})...",
                    config.num_base, config.dim, config.num_clusters
                );
                let (base, _) = generate_synthetic_sift(
                    config.num_base,
                    10,
                    config.dim,
                    config.num_clusters,
                    config.seed_clust,
                )?;
                base
            };

            let meta = generate_mode_b_bitset_detailed(
                &base_matrix,
                config.selectivity,
                config.seed_attr,
                config.seed_clust,
            );
            (meta.bitset.clone(), Some(meta))
        }
    };

    let actual_popcount = bitset.count_ones();
    println!("Actual Popcount:  {actual_popcount}");
    assert_eq!(
        actual_popcount, target_popcount,
        "Generated bitset popcount {actual_popcount} does not match target {target_popcount}"
    );

    if let Some(ref meta) = mode_b_meta {
        println!("Centroid Count C: {}", meta.centroid_count);
        println!("Active Clusters:  {}", meta.active_clusters.len());
        if let Some(sc) = meta.shortfall_cluster {
            println!(
                "Shortfall Calib:  cluster={sc}, activated_members={}",
                meta.shortfall_count
            );
        } else {
            println!("Shortfall Calib:  exact cluster fit (shortfall=0)");
        }
    }

    if let Some(ref out_path) = config.output_path {
        println!(
            "Writing bitset to {} ({:?})...",
            out_path.display(),
            config.format
        );
        match config.format {
            OutputFormat::RawBinary => write_bitset(out_path, &bitset)?,
            OutputFormat::Ivecs => write_bitset_as_ivecs(out_path, &bitset)?,
        }
        let bytes = fs::metadata(out_path)?.len();
        println!("Bitset written successfully ({bytes} bytes).");
    }

    if let Some(ref json_p) = config.json_path {
        println!("Writing provenance JSON to {}...", json_p.display());
        let mode_str = match config.mode {
            GenerationMode::Independent => "independent",
            GenerationMode::Correlated => "correlated",
        };
        let seed_clust_json = if config.mode == GenerationMode::Correlated {
            format!("{}", config.seed_clust)
        } else {
            "null".to_string()
        };
        let centroid_count_json = mode_b_meta
            .as_ref()
            .map_or("null".to_string(), |m| m.centroid_count.to_string());
        let shortfall_cluster_json = mode_b_meta
            .as_ref()
            .and_then(|m| m.shortfall_cluster)
            .map_or("null".to_string(), |c| c.to_string());
        let shortfall_count_json = mode_b_meta
            .as_ref()
            .map_or("null".to_string(), |m| m.shortfall_count.to_string());
        let active_clusters_str = mode_b_meta.as_ref().map_or(String::new(), |m| {
            m.active_clusters
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });

        let json_text = format!(
            r#"{{
  "git_commit": "{git_commit}",
  "mode": "{mode_str}",
  "num_vectors": {num_vectors},
  "selectivity": {selectivity},
  "target_popcount": {target_popcount},
  "actual_popcount": {actual_popcount},
  "seed_attr": {seed_attr},
  "seed_clust": {seed_clust_json},
  "centroid_count": {centroid_count_json},
  "active_cluster_ids": [{active_clusters_str}],
  "shortfall_cluster_id": {shortfall_cluster_json},
  "shortfall_count": {shortfall_count_json}
}}
"#,
            selectivity = config.selectivity,
            seed_attr = config.seed_attr,
        );

        if let Some(parent) = Path::new(json_p).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(json_p, json_text)?;
        println!("Provenance JSON written successfully.");
    }

    println!("============================================================");
    println!("[SUCCESS] Deterministic metadata generated cleanly.");
    println!("============================================================");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ERROR] {e}");
            ExitCode::FAILURE
        }
    }
}
