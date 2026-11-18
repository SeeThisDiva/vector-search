//! Matched-Recall Crossover Bracket & Log-Log Interpolation Analysis (`DESIGN.md` §4.3).

#![forbid(unsafe_code)]

use crate::json_util::escape_json;

use super::metrics::Phase5ArmRecord;
use super::MetadataMode;

/// One cell report covering all benchmark arms at a `(scale, mode, sigma)` grid coordinate.
#[derive(Debug, Clone)]
pub struct Phase5CellRecord {
    pub report_filename: String,
    pub status: String,
    pub scale: usize,
    pub mode: MetadataMode,
    pub sigma: f64,
    pub k: usize,
    pub reps: usize,
    pub expected_popcount: usize,
    pub actual_popcount: usize,
    pub clusters_c: usize,
    pub s1_uncapped_evaluated: bool,
    pub predicate_cache_hit: bool,
    pub predicate_gen_time_s: f64,
    pub gt_cache_hit: bool,
    pub gt_gen_time_s: f64,
    pub arms: Vec<Phase5ArmRecord>,
}

/// Matched-recall crossover result (`DESIGN.md` §4.3) for one `(mode, target_recall)` series.
#[derive(Debug, Clone)]
pub struct CrossoverModeResult {
    pub mode: MetadataMode,
    pub target_recall: f64,
    pub outcome: String,
    pub bracket_lo_sigma: Option<f64>,
    pub bracket_hi_sigma: Option<f64>,
    pub crossover_sigma_loglog: Option<f64>,
    pub bracket_lo_s2_ef: Option<usize>,
    pub bracket_hi_s2_ef: Option<usize>,
    pub note: String,
}

impl CrossoverModeResult {
    pub(crate) fn to_json_string(&self, indent: &str) -> String {
        let opt_f64 = |v: Option<f64>| v.map_or_else(|| "null".to_string(), |x| format!("{x:.6}"));
        let opt_usize = |v: Option<usize>| v.map_or_else(|| "null".to_string(), |x| x.to_string());
        format!(
            r#"{indent}{{
{indent}  "mode": "{}",
{indent}  "target_recall": {:.2},
{indent}  "outcome": "{}",
{indent}  "bracket_lo_sigma": {},
{indent}  "bracket_hi_sigma": {},
{indent}  "crossover_sigma_loglog": {},
{indent}  "bracket_lo_s2_ef": {},
{indent}  "bracket_hi_s2_ef": {},
{indent}  "note": "{}"
{indent}}}"#,
            self.mode.as_str(),
            self.target_recall,
            escape_json(&self.outcome),
            opt_f64(self.bracket_lo_sigma),
            opt_f64(self.bracket_hi_sigma),
            opt_f64(self.crossover_sigma_loglog),
            opt_usize(self.bracket_lo_s2_ef),
            opt_usize(self.bracket_hi_s2_ef),
            escape_json(&self.note),
        )
    }
}

/// Extracts matched-recall crossover brackets and log-log interpolated $\sigma^*$ from measured cells.
#[must_use]
pub fn compute_crossover_analysis(
    cells: &[Phase5CellRecord],
    target_recall: f64,
) -> Vec<CrossoverModeResult> {
    let mut modes_present: Vec<MetadataMode> = Vec::new();
    for cell in cells {
        if !modes_present.contains(&cell.mode) {
            modes_present.push(cell.mode);
        }
    }
    modes_present.sort();

    let mut results = Vec::new();
    for mode in modes_present {
        let mut mode_cells: Vec<&Phase5CellRecord> =
            cells.iter().filter(|c| c.mode == mode).collect();
        mode_cells.sort_by(|a, b| a.sigma.total_cmp(&b.sigma));

        #[derive(Clone, Copy)]
        struct PointEval {
            sigma: f64,
            s3_ms: f64,
            s2_ef: Option<usize>,
            s2_ms: Option<f64>,
            s2_wins: bool,
        }

        let mut points = Vec::new();
        for c in mode_cells {
            let Some(s3) = c.arms.iter().find(|a| a.strategy == "S3_pre_filter") else {
                continue;
            };
            let s3_ms = s3.summary.median_latency.mean_ms;
            let mut s2_candidates: Vec<&Phase5ArmRecord> = c
                .arms
                .iter()
                .filter(|a| a.strategy == "S2_in_traversal" && a.recall_at_k >= target_recall)
                .collect();
            s2_candidates.sort_by_key(|a| a.ef_search);
            let matched = s2_candidates.first().copied();
            let s2_ef = matched.map(|a| a.ef_search);
            let s2_ms = matched.map(|a| a.summary.median_latency.mean_ms);
            let s2_wins = s2_ms.is_some_and(|ms| ms <= s3_ms);
            points.push(PointEval {
                sigma: c.sigma,
                s3_ms,
                s2_ef,
                s2_ms,
                s2_wins,
            });
        }

        let mut found: Option<CrossoverModeResult> = None;
        for w in points.windows(2) {
            let lo = w[0];
            let hi = w[1];
            if !lo.s2_wins && hi.s2_wins {
                let crossover_sigma_loglog = match (lo.s2_ms, hi.s2_ms) {
                    (Some(s2_lo), Some(s2_hi))
                        if s2_lo > 0.0 && s2_hi > 0.0 && lo.s3_ms > 0.0 && hi.s3_ms > 0.0 =>
                    {
                        let x0 = lo.sigma.ln();
                        let x1 = hi.sigma.ln();
                        let d0 = s2_lo.ln() - lo.s3_ms.ln();
                        let d1 = s2_hi.ln() - hi.s3_ms.ln();
                        let denom = d1 - d0;
                        if d0 >= 0.0 && d1 <= 0.0 && denom.abs() > 1e-12 {
                            let t = (-d0 / denom).clamp(0.0, 1.0);
                            Some((x0 + t * (x1 - x0)).exp())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                let note = if lo.s2_ef.is_none() {
                    "target_not_reached_at_lower_bracket".to_string()
                } else {
                    "log_log_bracket_interpolated".to_string()
                };
                found = Some(CrossoverModeResult {
                    mode,
                    target_recall,
                    outcome: "crossover_found".to_string(),
                    bracket_lo_sigma: Some(lo.sigma),
                    bracket_hi_sigma: Some(hi.sigma),
                    crossover_sigma_loglog,
                    bracket_lo_s2_ef: lo.s2_ef,
                    bracket_hi_s2_ef: hi.s2_ef,
                    note,
                });
                break;
            }
        }

        if let Some(res) = found {
            results.push(res);
        } else {
            let note = if points.is_empty() {
                "no_cells_with_s3_available".to_string()
            } else if points.iter().all(|p| !p.s2_wins) {
                "s3_wins_across_all_measured_selectivities".to_string()
            } else if points.iter().all(|p| p.s2_wins) {
                "s2_wins_across_all_measured_selectivities".to_string()
            } else {
                "no_monotone_s3_to_s2_crossover_in_range".to_string()
            };
            results.push(CrossoverModeResult {
                mode,
                target_recall,
                outcome: "no_crossover_in_range".to_string(),
                bracket_lo_sigma: None,
                bracket_hi_sigma: None,
                crossover_sigma_loglog: None,
                bracket_lo_s2_ef: None,
                bracket_hi_s2_ef: None,
                note,
            });
        }
    }
    results
}
