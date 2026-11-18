//! Phases 1–3.5 Benchmark Pipelines, Provenance Metadata, Multi-Scale Profiling,
//! Golden Fixture Verification, and CLI Runners.

#![forbid(unsafe_code)]

pub mod baseline_profile;
pub mod cli;
pub mod dataset_loader;
pub mod golden;
pub mod phase1;
pub mod phase2;
pub mod phase3;

use crate::hardware::HardwareInfo;

pub use crate::hardware::current_utc_rfc3339;
pub use crate::json_util::{escape_json, opt_path_json};
pub use crate::recall::{summarize_repeated_latencies, RepeatedSweepSummary};

pub use baseline_profile::{
    print_baseline_profile_report, profile_single_sift1m_scale, run_baseline_multi_scale_profile,
    spawn_isolated_scale_profile, BaselineMultiScaleReport, BaselineScaleProfile,
};
pub use cli::{run_phase2_cli, run_phase3_cli};
pub use dataset_loader::{resolve_cli_dataset_mode, Phase1DatasetMode};
pub use golden::{
    compute_query_results_hash, generate_golden_fixtures, parse_golden_results,
    run_phase3_5_baseline_measurement, verify_index_against_golden_fixture, GoldenFixture,
    GoldenQueryResult,
};
pub use phase1::{
    run_phase1_benchmark, OfficialGtComparison, Phase1Config, Phase1Report, SavedArtifacts,
};
pub use phase2::{
    print_phase2_report, run_phase2_benchmark, verify_reproducibility,
    verify_reproducibility_with_metric, HnswEfEvaluation, Phase2Config, Phase2Report,
    PHASE2_TARGET_MAX_LATENCY_MS, PHASE2_TARGET_MIN_RECALL,
};
pub use phase3::{
    print_phase3_report, run_phase3_benchmark, Phase3Config, Phase3EfEvaluation,
    Phase3MemoryProfile, Phase3Report,
};

/// Provenance and runtime environment metadata recorded in every benchmark JSON report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportProvenance {
    pub git_commit: String,
    pub git_dirty: bool,
    pub rustc_version: String,
    pub power_mode: String,
    pub processor_affinity_mask: String,
}

impl ReportProvenance {
    /// Appends `git_commit`, `git_dirty`, `rustc_version`, and the `environment` block
    /// to an in-progress top-level JSON object (with 2-space indentation).
    pub fn append_json_fields(&self, out: &mut String, hardware: &HardwareInfo) {
        out.push_str(&format!(
            "  \"git_commit\": \"{}\",\n",
            escape_json(&self.git_commit)
        ));
        out.push_str(&format!("  \"git_dirty\": {},\n", self.git_dirty));
        out.push_str(&format!(
            "  \"rustc_version\": \"{}\",\n",
            escape_json(&self.rustc_version)
        ));
        out.push_str("  \"environment\": {\n");
        out.push_str(&format!("    \"os\": \"{}\",\n", escape_json(&hardware.os)));
        out.push_str(&format!(
            "    \"arch\": \"{}\",\n",
            escape_json(&hardware.arch)
        ));
        out.push_str(&format!(
            "    \"cpu_model\": \"{}\",\n",
            escape_json(&hardware.cpu_model)
        ));
        out.push_str(&format!(
            "    \"logical_cores\": {},\n",
            hardware.logical_cores
        ));
        match hardware.total_ram_gb {
            Some(ram) => out.push_str(&format!("    \"total_ram_gb\": {ram:.2},\n")),
            None => out.push_str("    \"total_ram_gb\": null,\n"),
        }
        out.push_str(&format!(
            "    \"power_mode\": \"{}\",\n",
            escape_json(&self.power_mode)
        ));
        out.push_str(&format!(
            "    \"processor_affinity_mask\": \"{}\"\n",
            escape_json(&self.processor_affinity_mask)
        ));
        out.push_str("  },\n");
    }
}

/// Appends the standard `"hardware": { ... },` JSON object (2-space indented, 2-decimal RAM)
/// to an in-progress top-level JSON report.
pub fn append_hardware_json(out: &mut String, hardware: &HardwareInfo) {
    out.push_str("  \"hardware\": {\n");
    out.push_str(&format!(
        "    \"summary\": \"{}\",\n",
        escape_json(&hardware.summary_line())
    ));
    out.push_str(&format!(
        "    \"cpu_model\": \"{}\",\n",
        escape_json(&hardware.cpu_model)
    ));
    out.push_str(&format!(
        "    \"logical_cores\": {},\n",
        hardware.logical_cores
    ));
    out.push_str(&format!(
        "    \"total_ram_gb\": {:.2},\n",
        hardware.total_ram_gb.unwrap_or(0.0)
    ));
    out.push_str(&format!("    \"os\": \"{}\",\n", escape_json(&hardware.os)));
    out.push_str(&format!(
        "    \"arch\": \"{}\"\n",
        escape_json(&hardware.arch)
    ));
    out.push_str("  },\n");
}

/// Detects provenance (`git_commit`, `git_dirty`, `rustc_version`) and environment settings
/// using the specified command binaries, falling back to `"unknown"` when a binary is unavailable
/// or exits non-zero.
#[must_use]
pub fn detect_provenance_with_commands(git_bin: &str, rustc_bin: &str) -> ReportProvenance {
    let git_commit = std::process::Command::new(git_bin)
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    let git_dirty = if git_commit == "unknown" {
        false
    } else {
        std::process::Command::new(git_bin)
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .map(|o| o.status.success() && !o.stdout.is_empty())
            .unwrap_or(false)
    };

    let rustc_version = std::process::Command::new(rustc_bin)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    let power_mode = std::env::var("VECTOR_SEARCH_POWER_MODE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(detect_os_power_mode);

    let processor_affinity_mask = std::env::var("VECTOR_SEARCH_AFFINITY_MASK")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "not_set".to_string());

    ReportProvenance {
        git_commit,
        git_dirty,
        rustc_version,
        power_mode,
        processor_affinity_mask,
    }
}

fn detect_os_power_mode() -> String {
    if cfg!(target_os = "windows") {
        if let Ok(out) = std::process::Command::new("powercfg")
            .arg("/getactivescheme")
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).to_ascii_lowercase();
                if s.contains("high performance")
                    || s.contains("ultimate performance")
                    || s.contains("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")
                    || s.contains("e9a42b02-d5df-448d-aa00-03f14749eb61")
                {
                    return "high_performance".to_string();
                } else if s.contains("balanced")
                    || s.contains("381b4222-f694-41f0-9685-ff5bb260df2e")
                {
                    return "balanced".to_string();
                } else if s.contains("power saver")
                    || s.contains("a1841308-3541-4fab-bc81-f71556f20b4a")
                {
                    return "power_saver".to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

/// Returns the cached provenance and environment snapshot for the current process.
#[must_use]
pub fn detect_provenance() -> ReportProvenance {
    static PROVENANCE: std::sync::OnceLock<ReportProvenance> = std::sync::OnceLock::new();
    PROVENANCE
        .get_or_init(|| detect_provenance_with_commands("git", "rustc"))
        .clone()
}
