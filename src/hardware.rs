//! Cross-platform host machine hardware detection (`Hardware: [YOUR MACHINE]`).

use std::env;
use std::fmt;
use std::process::Command;
use std::sync::OnceLock;
use std::thread;

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
