//! Declarative grid specification and per-index configuration for Phase 4 parameter sweeps.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

use crate::cli_util::parse_positive_csv_usizes;
use crate::error::{Result, VectorSearchError};
use crate::hnsw::HnswConfig;

/// Individual HNSW index construction parameters in a Phase 4 sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase4IndexConfig {
    pub m: usize,
    pub m_max: Option<usize>,
    pub m_max0: Option<usize>,
    pub ef_construction: usize,
    pub keep_pruned: bool,
    pub keep_pruned_shrink: Option<bool>,
    pub seed: u64,
    pub explicit_seed: bool,
}

impl Default for Phase4IndexConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: None,
            m_max0: None,
            ef_construction: 100,
            keep_pruned: true,
            keep_pruned_shrink: None,
            seed: 42,
            explicit_seed: false,
        }
    }
}

impl Phase4IndexConfig {
    /// Constructs a configuration with target degree `m` and beam `ef_construction`.
    #[must_use]
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            ef_construction,
            ..Self::default()
        }
    }

    /// Converts this grid configuration into an internal [`HnswConfig`].
    #[must_use]
    pub fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            m_max: self.m_max,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            keep_pruned: self.keep_pruned,
            keep_pruned_shrink: self.keep_pruned_shrink,
            seed: self.seed,
            ..HnswConfig::default()
        }
    }

    /// Generates the deterministic per-index report filename for a given dataset scale.
    #[must_use]
    pub fn report_filename(&self, scale_vectors: usize) -> String {
        format!(
            "scale_{}_m{}_efc{}{}.json",
            scale_vectors,
            self.m,
            self.ef_construction,
            if self.keep_pruned_shrink == Some(false) {
                "_no_kp_shrink"
            } else {
                ""
            }
        )
    }
}

/// Declarative grid specification for Phase 4 parameter sweeps.
#[derive(Debug, Clone, PartialEq)]
pub struct Phase4GridSpec {
    pub scale_vectors: Option<usize>,
    pub k: usize,
    pub sweep_runs: usize,
    pub seed: u64,
    pub ef_search_list: Vec<usize>,
    pub configs: Vec<Phase4IndexConfig>,
}

impl Default for Phase4GridSpec {
    fn default() -> Self {
        Self {
            scale_vectors: None,
            k: 10,
            sweep_runs: 3,
            seed: 42,
            ef_search_list: vec![10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600],
            configs: Vec::new(),
        }
    }
}

impl Phase4GridSpec {
    /// Validates all parameters and configurations in the grid specification.
    pub fn validate(&self) -> Result<()> {
        if self.configs.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "Grid specification resulted in 0 configurations".to_string(),
            ));
        }

        if self.k == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "k must be >= 1".to_string(),
            ));
        }
        if self.sweep_runs == 0 {
            return Err(VectorSearchError::InvalidArgument(
                "sweep_runs must be >= 1".to_string(),
            ));
        }

        if self.ef_search_list.is_empty() {
            return Err(VectorSearchError::InvalidArgument(
                "ef_search_list cannot be empty".to_string(),
            ));
        }
        for &ef in &self.ef_search_list {
            if ef == 0 {
                return Err(VectorSearchError::InvalidArgument(
                    "ef_search values must be >= 1".to_string(),
                ));
            }
        }

        for cfg in &self.configs {
            cfg.to_hnsw_config().validate()?;
        }

        Ok(())
    }

    /// Parses a grid specification from a file on disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::parse_str(&content)
    }

    /// Parses a grid specification from a directive string.
    pub fn parse_str(input: &str) -> Result<Self> {
        let mut scale_vectors: Option<usize> = None;
        let mut k: usize = 10;
        let mut sweep_runs: usize = 3;
        let mut seed: u64 = 42;
        let mut ef_search_list: Option<Vec<usize>> = None;
        let mut m_list: Vec<usize> = Vec::new();
        let mut ef_construction_list: Vec<usize> = Vec::new();
        let mut explicit_configs: Vec<Phase4IndexConfig> = Vec::new();

        for (line_idx, raw_line) in input.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            // Strip any trailing comments (e.g., # or //)
            let line_clean = if let Some(idx) = trimmed.find('#') {
                trimmed[..idx].trim()
            } else if let Some(idx) = trimmed.find("//") {
                trimmed[..idx].trim()
            } else {
                trimmed
            };
            if line_clean.is_empty() {
                continue;
            }

            // Split into key and value by '=' or ':'
            let (key, val) = if let Some(pos) = line_clean.find('=') {
                (line_clean[..pos].trim(), line_clean[pos + 1..].trim())
            } else if let Some(pos) = line_clean.find(':') {
                (line_clean[..pos].trim(), line_clean[pos + 1..].trim())
            } else {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Line {line_num}: invalid syntax (expected key = value): '{line_clean}'"
                )));
            };

            let key_norm = key.to_ascii_lowercase();
            match key_norm.as_str() {
                "scale_vectors" | "scale" | "max_base" => {
                    let v: usize = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid scale_vectors '{val}': {e}"
                        ))
                    })?;
                    scale_vectors = Some(v);
                }
                "k" => {
                    k = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid k '{val}': {e}"
                        ))
                    })?;
                }
                "sweep_runs" | "repetitions" | "runs" => {
                    sweep_runs = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid sweep_runs '{val}': {e}"
                        ))
                    })?;
                }
                "seed" => {
                    seed = val.parse().map_err(|e| {
                        VectorSearchError::InvalidArgument(format!(
                            "Line {line_num}: invalid seed '{val}': {e}"
                        ))
                    })?;
                }
                "ef_search" | "ef_search_list" => {
                    let flag_ctx = format!("Line {line_num} ef_search");
                    let list = parse_positive_csv_usizes(val, &flag_ctx)?;
                    ef_search_list = Some(list);
                }
                "m" | "m_list" => {
                    let flag_ctx = format!("Line {line_num} m");
                    m_list = parse_positive_csv_usizes(val, &flag_ctx)?;
                }
                "ef_construction" | "ef_construction_list" => {
                    let flag_ctx = format!("Line {line_num} ef_construction");
                    ef_construction_list = parse_positive_csv_usizes(val, &flag_ctx)?;
                }
                "config" => {
                    let cfg = parse_explicit_config_line(val, line_num, seed)?;
                    explicit_configs.push(cfg);
                }
                unknown => {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: unrecognized grid directive '{unknown}'"
                    )));
                }
            }
        }

        let mut configs = Vec::new();
        // Cartesian expansion of (m, ef_construction) subject to ef_construction >= m
        if !m_list.is_empty() && !ef_construction_list.is_empty() {
            for &m_val in &m_list {
                for &efc_val in &ef_construction_list {
                    if efc_val >= m_val {
                        configs.push(Phase4IndexConfig {
                            m: m_val,
                            m_max: None,
                            m_max0: None,
                            ef_construction: efc_val,
                            keep_pruned: true,
                            keep_pruned_shrink: None,
                            seed,
                            explicit_seed: false,
                        });
                    }
                }
            }
        }

        configs.extend(explicit_configs);

        // Inherit final grid seed for any config without explicit per-config seed
        for cfg in &mut configs {
            if !cfg.explicit_seed {
                cfg.seed = seed;
            }
        }

        let final_ef_search_list = ef_search_list
            .unwrap_or_else(|| vec![10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600]);

        let spec = Self {
            scale_vectors,
            k,
            sweep_runs,
            seed,
            ef_search_list: final_ef_search_list,
            configs,
        };
        spec.validate()?;
        Ok(spec)
    }
}

fn parse_explicit_config_line(
    val: &str,
    line_num: usize,
    default_seed: u64,
) -> Result<Phase4IndexConfig> {
    let mut m: Option<usize> = None;
    let mut m_max: Option<usize> = None;
    let mut m_max0: Option<usize> = None;
    let mut ef_construction: Option<usize> = None;
    let mut keep_pruned = true;
    let mut keep_pruned_shrink: Option<bool> = None;
    let mut seed: Option<u64> = None;

    for part in val.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let (k, v) = if let Some(pos) = p.find(':') {
            (p[..pos].trim(), p[pos + 1..].trim())
        } else if let Some(pos) = p.find('=') {
            (p[..pos].trim(), p[pos + 1..].trim())
        } else {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Line {line_num}: invalid config field '{p}' (expected key:value)"
            )));
        };

        match k.to_ascii_lowercase().as_str() {
            "m" => {
                m = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid m '{v}': {e}"
                    ))
                })?);
            }
            "m_max" => {
                m_max = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid m_max '{v}': {e}"
                    ))
                })?);
            }
            "m_max0" => {
                m_max0 = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid m_max0 '{v}': {e}"
                    ))
                })?);
            }
            "ef_construction" | "efc" => {
                ef_construction = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid ef_construction '{v}': {e}"
                    ))
                })?);
            }
            "keep_pruned" => {
                keep_pruned = parse_bool(v, line_num, "keep_pruned")?;
            }
            "keep_pruned_shrink" => {
                keep_pruned_shrink = Some(parse_bool(v, line_num, "keep_pruned_shrink")?);
            }
            "seed" => {
                seed = Some(v.parse().map_err(|e| {
                    VectorSearchError::InvalidArgument(format!(
                        "Line {line_num}: invalid seed '{v}': {e}"
                    ))
                })?);
            }
            unknown => {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Line {line_num}: unrecognized config field '{unknown}'"
                )));
            }
        }
    }

    let m = m.ok_or_else(|| {
        VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: explicit config missing required field 'm'"
        ))
    })?;
    let ef_construction = ef_construction.ok_or_else(|| {
        VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: explicit config missing required field 'ef_construction'"
        ))
    })?;

    let explicit_seed = seed.is_some();
    let seed = seed.unwrap_or(default_seed);

    Ok(Phase4IndexConfig {
        m,
        m_max,
        m_max0,
        ef_construction,
        keep_pruned,
        keep_pruned_shrink,
        seed,
        explicit_seed,
    })
}

fn parse_bool(v: &str, line_num: usize, name: &str) -> Result<bool> {
    match v.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(VectorSearchError::InvalidArgument(format!(
            "Line {line_num}: invalid boolean for {name}: '{v}'"
        ))),
    }
}
