//! Cross-platform host machine hardware detection, peak RSS metadata, CPU affinity pinning,
//! and UTC timestamp formatting.

use std::env;
use std::fmt;
#[cfg(any(target_os = "windows", target_os = "macos"))]
use std::process::Command;
use std::sync::OnceLock;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, VectorSearchError};

/// Detected host CPU, memory, and OS specifications for reproducible benchmark reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub logical_cores: usize,
    pub total_ram_gb: Option<f64>,
    pub os: String,
    pub arch: String,
}

impl HardwareInfo {
    /// Single-line summary formatted for benchmark headers and reports.
    #[must_use]
    pub fn summary_line(&self) -> String {
        match self.total_ram_gb {
            Some(ram) => format!(
                "{} ({} logical cores, {:.1} GB RAM, {}/{})",
                self.cpu_model, self.logical_cores, ram, self.os, self.arch
            ),
            None => format!(
                "{} ({} logical cores, {}/{})",
                self.cpu_model, self.logical_cores, self.os, self.arch
            ),
        }
    }
}

impl fmt::Display for HardwareInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary_line())
    }
}

/// Detects host hardware specifications (cached after first probe).
#[must_use]
pub fn detect_hardware() -> HardwareInfo {
    static CACHED: OnceLock<HardwareInfo> = OnceLock::new();
    CACHED.get_or_init(probe_hardware).clone()
}

fn probe_hardware() -> HardwareInfo {
    let logical_cores = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let os = env::consts::OS.to_string();
    let arch = env::consts::ARCH.to_string();

    let (cpu_model, total_ram_gb) = probe_platform_cpu_and_ram();

    HardwareInfo {
        cpu_model: if cpu_model.trim().is_empty() {
            format!("Unknown {arch} CPU")
        } else {
            cpu_model.trim().to_string()
        },
        logical_cores,
        total_ram_gb,
        os,
        arch,
    }
}

#[cfg(target_os = "windows")]
fn probe_platform_cpu_and_ram() -> (String, Option<f64>) {
    let cpu = probe_windows_cpu_registry().unwrap_or_default();
    let ram = probe_windows_ram_gb();
    (cpu, ram)
}

#[cfg(target_os = "windows")]
fn probe_windows_cpu_registry() -> Option<String> {
    let output = Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        if let Some(idx) = line.find("REG_SZ") {
            let name = line[idx + "REG_SZ".len()..].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn probe_windows_ram_gb() -> Option<f64> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let bytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some((bytes as f64) / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(target_os = "linux")]
fn probe_platform_cpu_and_ram() -> (String, Option<f64>) {
    let mut cpu = String::new();
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some(colon) = rest.find(':') {
                    cpu = rest[colon + 1..].trim().to_string();
                    if !cpu.is_empty() {
                        break;
                    }
                }
            }
        }
    }

    let mut ram_gb = None;
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb_str = rest.trim().trim_end_matches("kB").trim();
                if let Ok(kb) = kb_str.parse::<u64>() {
                    ram_gb = Some((kb as f64) / (1024.0 * 1024.0));
                    break;
                }
            }
        }
    }

    (cpu, ram_gb)
}

#[cfg(target_os = "macos")]
fn probe_platform_cpu_and_ram() -> (String, Option<f64>) {
    let cpu = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let ram_gb = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|bytes| (bytes as f64) / (1024.0 * 1024.0 * 1024.0));

    (cpu, ram_gb)
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn probe_platform_cpu_and_ram() -> (String, Option<f64>) {
    (String::new(), None)
}

/// Returns the process peak resident set size (or peak working set on Windows) in mebibytes (MiB).
///
/// Completely safe: enforces `#![forbid(unsafe_code)]` with zero unsafe blocks.
#[must_use]
pub fn get_peak_rss_mb() -> Option<f64> {
    #[cfg(target_os = "windows")]
    {
        get_windows_peak_working_set_mb()
    }

    #[cfg(target_os = "linux")]
    {
        get_linux_peak_rss_mb()
    }

    #[cfg(target_os = "macos")]
    {
        get_macos_peak_rss_mb()
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "windows")]
fn get_windows_peak_working_set_mb() -> Option<f64> {
    let pid = std::process::id();
    let cmd = format!("(Get-Process -Id {pid}).PeakWorkingSet64");
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &cmd])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let bytes: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;

    Some((bytes as f64) / (1024.0 * 1024.0))
}

#[cfg(target_os = "linux")]
fn get_linux_peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb_str = rest.trim().trim_end_matches("kB").trim();
            if let Ok(kb) = kb_str.parse::<u64>() {
                return Some((kb as f64) / 1024.0);
            }
        }
    }
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.trim().trim_end_matches("kB").trim();
            if let Ok(kb) = kb_str.parse::<u64>() {
                return Some((kb as f64) / 1024.0);
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn get_macos_peak_rss_mb() -> Option<f64> {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let kb: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;

    Some((kb as f64) / 1024.0)
}

/// Returns a descriptive platform identifier for how peak process RSS / working set is measured.
#[must_use]
pub fn peak_rss_measurement_method(isolated_child: bool) -> String {
    let scope = if isolated_child {
        " (per-scale isolated child process)"
    } else {
        ""
    };

    #[cfg(target_os = "windows")]
    {
        format!(
            "windows_powershell_peak_working_set_64{scope}: powershell -NoProfile -NonInteractive -Command (Get-Process -Id <benchmark process id>).PeakWorkingSet64"
        )
    }

    #[cfg(target_os = "linux")]
    {
        format!("linux_proc_status_vm_hwm{scope}: /proc/self/status VmHWM")
    }

    #[cfg(target_os = "macos")]
    {
        format!("macos_ps_rss{scope}: ps -o rss= -p <pid>")
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        let _ = scope;
        "unknown_platform".to_string()
    }
}

/// Parses a positive hexadecimal (`0x...`) or decimal CPU affinity bitmask.
pub fn parse_affinity_mask(raw: &str) -> Result<u64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(VectorSearchError::InvalidArgument(
            "Affinity mask cannot be empty".to_string(),
        ));
    }
    let mask = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16)
    } else {
        trimmed.parse::<u64>()
    }
    .map_err(|e| {
        VectorSearchError::InvalidArgument(format!("Invalid affinity mask '{trimmed}': {e}"))
    })?;
    if mask == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Affinity mask must be positive (> 0)".to_string(),
        ));
    }
    Ok(mask)
}

/// Pins the current process CPU affinity mask and verifies the applied OS mask.
#[cfg(target_os = "windows")]
pub fn pin_process_affinity_os(requested_mask: u64) -> Result<u64> {
    let pid = std::process::id();
    let script = format!(
        "$p = Get-Process -Id {pid}; $p.ProcessorAffinity = [IntPtr]{requested_mask}; [int64](Get-Process -Id {pid}).ProcessorAffinity"
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .map_err(VectorSearchError::Io)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(VectorSearchError::InvalidArgument(format!(
            "Failed to set process affinity mask 0x{requested_mask:04x}: {}",
            stderr.trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let observed = stdout.trim().parse::<u64>().map_err(|e| {
        VectorSearchError::VerificationFailed(format!(
            "Failed to parse verified process affinity output '{}': {e}",
            stdout.trim()
        ))
    })?;
    if observed != requested_mask {
        return Err(VectorSearchError::VerificationFailed(format!(
            "Process affinity verification failed: expected 0x{requested_mask:04x}, got 0x{observed:04x}"
        )));
    }
    Ok(observed)
}

/// Pins the current process CPU affinity mask (unsupported natively outside Windows).
#[cfg(not(target_os = "windows"))]
pub fn pin_process_affinity_os(_requested_mask: u64) -> Result<u64> {
    Err(VectorSearchError::InvalidArgument(
        "--affinity pinning is only supported on Windows; pin externally (e.g. taskset) and set VECTOR_SEARCH_AFFINITY_MASK".to_string(),
    ))
}

/// Generates a current UTC RFC3339 timestamp (`YYYY-MM-DDTHH:MM:SSZ`) using `std` only.
#[must_use]
pub fn current_utc_rfc3339() -> String {
    let now = SystemTime::now();
    let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let days = secs / 86_400;
    let rem_secs = secs % 86_400;
    let hours = rem_secs / 3_600;
    let mins = (rem_secs % 3_600) / 60;
    let s = rem_secs % 60;

    // Howard Hinnant's civil-from-days algorithm
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_affinity_mask() {
        assert_eq!(parse_affinity_mask("0x0004").unwrap(), 4);
        assert_eq!(parse_affinity_mask(" 0X0555 ").unwrap(), 0x0555);
        assert_eq!(parse_affinity_mask("16").unwrap(), 16);
        assert!(parse_affinity_mask("").is_err());
        assert!(parse_affinity_mask("0").is_err());
        assert!(parse_affinity_mask("0x0").is_err());
        assert!(parse_affinity_mask("xyz").is_err());
    }

    #[test]
    fn test_peak_rss_measurement_method_and_timestamp() {
        let single = peak_rss_measurement_method(false);
        let child = peak_rss_measurement_method(true);
        assert!(!single.is_empty());
        assert!(!child.is_empty());
        if single != "unknown_platform" {
            assert!(child.contains("per-scale isolated child process"));
        }

        let ts = current_utc_rfc3339();
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
    }
}
