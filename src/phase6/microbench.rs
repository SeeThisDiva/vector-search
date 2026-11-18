//! Phase 6 distance kernel microbenchmark (`Way A` vs `Way B` vs `PQ16` vs `PQ32` vs `f32`).

use std::path::PathBuf;
use std::time::Instant;

use crate::benchmark::{append_hardware_json, detect_provenance, ReportProvenance};
use crate::dataset::generate_synthetic_sift;
use crate::distance::{l2_squared, Metric};
use crate::error::{Result, VectorSearchError};
use crate::hardware::{current_utc_rfc3339, detect_hardware, HardwareInfo};
use crate::json_util::{escape_json, push_f64_array_prec, write_file_with_parents};
use crate::matrix::RowMatrix;
use crate::quantized::{
    l2_squared_sq8_expanded, l2_squared_sq8_shifted, pq_adc_distance, PqConfig, PqQueryState,
    PqVectors, QuantizedDistanceSource, Sq8KernelKind, Sq8QueryState, Sq8Vectors,
    PQ_DEFAULT_MAX_TRAIN_VECTORS,
};
use crate::recall::median_min_max;
use crate::rng::DeterministicRng;

use super::config::{apply_affinity_pinning, Phase6MicrobenchConfig};

/// Single kernel measurement inside `microbench_kernels.json`.
#[derive(Debug, Clone)]
pub struct KernelMicrobenchResult {
    pub kernel_id: String,
    pub label: String,
    pub bytes_per_vector: usize,
    pub median_ns_per_eval: f64,
    pub min_ns_per_eval: f64,
    pub max_ns_per_eval: f64,
    pub spread_ns_per_eval: f64,
    pub evals_per_sec: f64,
    pub run_ns_per_eval: Vec<f64>,
}

/// Complete kernel microbenchmark report (`Way A` vs `Way B` vs `PQ16` vs `PQ32` vs `f32`).
#[derive(Debug, Clone)]
pub struct Phase6MicrobenchReport {
    pub output_path: PathBuf,
    pub dim: usize,
    pub num_vectors: usize,
    pub num_evals: usize,
    pub reps: usize,
    pub seed: u64,
    pub kernels: Vec<KernelMicrobenchResult>,
    pub standard_max_rel_err_way_a: f64,
    pub standard_max_rel_err_way_b: f64,
    pub adversarial_offset_max_abs_err_way_a: f64,
    pub adversarial_offset_max_abs_err_way_b: f64,
    pub speedup_way_b_over_way_a: f64,
    pub recommended_default_kernel: String,
}

fn time_kernel_repetitions<F>(num_evals: usize, reps: usize, mut run_pass: F) -> Vec<f64>
where
    F: FnMut() -> f32,
{
    // 1 untimed warmup pass
    std::hint::black_box(run_pass());

    let mut run_ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let acc = run_pass();
        let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
        std::hint::black_box(acc);
        run_ns.push(elapsed_ns / (num_evals.max(1) as f64));
    }
    run_ns
}

fn summarize_kernel_bench(
    kernel_id: &str,
    label: &str,
    bytes_per_vector: usize,
    run_ns_per_eval: Vec<f64>,
) -> KernelMicrobenchResult {
    let (median_ns, min_ns, max_ns) = median_min_max(&run_ns_per_eval);
    let evals_per_sec = if median_ns > 0.0 {
        1e9 / median_ns
    } else {
        0.0
    };
    KernelMicrobenchResult {
        kernel_id: kernel_id.to_string(),
        label: label.to_string(),
        bytes_per_vector,
        median_ns_per_eval: median_ns,
        min_ns_per_eval: min_ns,
        max_ns_per_eval: max_ns,
        spread_ns_per_eval: (max_ns - min_ns).max(0.0),
        evals_per_sec,
        run_ns_per_eval,
    }
}

fn ref_dequantized_l2_sq_f64(query: &[f32], mins: &[f32], scales: &[f32], code: &[u8]) -> f64 {
    query
        .iter()
        .zip(mins.iter())
        .zip(scales.iter())
        .zip(code.iter())
        .map(|(((&qj, &min_j), &scale_j), &cj)| {
            let recon = f64::from(min_j) + f64::from(scale_j) * f64::from(cj);
            let d = f64::from(qj) - recon;
            d * d
        })
        .sum()
}

fn compute_sq8_f64_reference_errors(
    base: &RowMatrix<f32>,
    queries: &RowMatrix<f32>,
    sq8_shifted: &Sq8Vectors,
    sq8_expanded: &Sq8Vectors,
) -> (f64, f64) {
    let quantizer = sq8_shifted.quantizer();
    let mut state_a = Sq8QueryState::default();
    let mut state_b = Sq8QueryState::default();
    let mut max_rel_a = 0.0_f64;
    let mut max_rel_b = 0.0_f64;

    let sample_rows = base.rows().min(256);
    for qi in 0..queries.rows() {
        let q = queries.row(qi);
        sq8_shifted.prepare_query(q, &mut state_a);
        sq8_expanded.prepare_query(q, &mut state_b);
        for row_id in 0..sample_rows {
            let ref_f64 = ref_dequantized_l2_sq_f64(
                q,
                quantizer.mins(),
                quantizer.scales(),
                sq8_shifted.code(row_id),
            );
            let da = f64::from(sq8_shifted.query_distance(&state_a, row_id));
            let db = f64::from(sq8_expanded.query_distance(&state_b, row_id));
            let denom = ref_f64.max(1e-12);
            max_rel_a = max_rel_a.max((da - ref_f64).abs() / denom);
            max_rel_b = max_rel_b.max((db - ref_f64).abs() / denom);
        }
    }
    (max_rel_a, max_rel_b)
}

fn measure_adversarial_cancellation_errors() -> Result<(f64, f64)> {
    // Anchor at 0.0 sets min_j = 0.0 while clustered rows live near 10_000.0 + tiny offsets.
    let base = RowMatrix::new(
        4,
        4,
        vec![
            0.0, 0.0, 0.0, 0.0, //
            10_000.0, 10_000.0, 10_000.0, 10_000.0, //
            10_000.05, 10_000.02, 9_999.98, 10_000.01, //
            10_000.1, 9_999.95, 10_000.04, 10_000.0, //
        ],
    )?;
    let query = [10_000.01_f32, 10_000.01, 10_000.01, 10_000.01];
    let sq8_a =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::QueryShifted)?;
    let sq8_b =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::NormExpanded)?;
    let quantizer = sq8_a.quantizer();

    let mut state_a = Sq8QueryState::default();
    let mut state_b = Sq8QueryState::default();
    sq8_a.prepare_query(&query, &mut state_a);
    sq8_b.prepare_query(&query, &mut state_b);

    let mut max_abs_a = 0.0_f64;
    let mut max_abs_b = 0.0_f64;
    for row_id in 1..4 {
        let ref_f64 = ref_dequantized_l2_sq_f64(
            &query,
            quantizer.mins(),
            quantizer.scales(),
            sq8_a.code(row_id),
        );
        let da = f64::from(sq8_a.query_distance(&state_a, row_id));
        let db = f64::from(sq8_b.query_distance(&state_b, row_id));
        max_abs_a = max_abs_a.max((da - ref_f64).abs());
        max_abs_b = max_abs_b.max((db - ref_f64).abs());
    }
    Ok((max_abs_a, max_abs_b))
}

/// Runs the Phase 6 distance kernel microbenchmark (`Way A`, `Way B`, `PQ16`, `PQ32`, `f32`).
pub fn run_phase6_microbench(config: &Phase6MicrobenchConfig) -> Result<Phase6MicrobenchReport> {
    if config.num_vectors == 0 || config.num_evals == 0 || config.dim == 0 || config.reps == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Microbench parameters (num_vectors, num_evals, dim, reps) must all be > 0".to_string(),
        ));
    }

    let mut prov = detect_provenance();
    apply_affinity_pinning(config.affinity.as_deref(), &mut prov)?;
    let hw = detect_hardware();

    let (base, queries) =
        generate_synthetic_sift(config.num_vectors, 8, config.dim, 16, config.seed)?;

    let mut rng = DeterministicRng::new(config.seed ^ 0x9E37_79B9_7F4A_7C15);
    let access_ids: Vec<usize> = (0..config.num_evals)
        .map(|_| rng.next_usize(config.num_vectors))
        .collect();

    let sq8_a =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::QueryShifted)?;
    let sq8_b =
        Sq8Vectors::from_matrix_with_kernel(&base, Metric::L2Squared, Sq8KernelKind::NormExpanded)?;
    let (std_rel_a, std_rel_b) = compute_sq8_f64_reference_errors(&base, &queries, &sq8_a, &sq8_b);
    let (adv_abs_a, adv_abs_b) = measure_adversarial_cancellation_errors()?;

    let mut kernels = Vec::new();
    let q_f32 = queries.row(0);

    // 1. Raw f32 l2_squared
    let f32_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
        let q = std::hint::black_box(q_f32);
        let mut acc = 0.0_f32;
        for &id in std::hint::black_box(access_ids.as_slice()) {
            acc += l2_squared(q, base.row(id));
        }
        acc
    });
    kernels.push(summarize_kernel_bench(
        "f32_l2_squared",
        "Raw f32 L2Squared (8-way unrolled)",
        config.dim * std::mem::size_of::<f32>(),
        f32_runs,
    ));

    // 2. SQ8 Way A (QueryShifted)
    let mut state_a = Sq8QueryState::default();
    sq8_a.prepare_query(q_f32, &mut state_a);
    let scales_a = sq8_a.quantizer().scales();
    let way_a_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
        let qs = std::hint::black_box(state_a.q_shifted.as_slice());
        let sc = std::hint::black_box(scales_a);
        let mut acc = 0.0_f32;
        for &id in std::hint::black_box(access_ids.as_slice()) {
            acc += l2_squared_sq8_shifted(qs, sc, sq8_a.code(id));
        }
        acc
    });
    let way_a_res = summarize_kernel_bench(
        "sq8_query_shifted",
        "SQ8 Way A: Query-Shifted Asymmetric L2^2",
        config.dim,
        way_a_runs,
    );
    let way_a_median_ns = way_a_res.median_ns_per_eval;
    kernels.push(way_a_res);

    // 3. SQ8 Way B (NormExpanded)
    let mut state_b = Sq8QueryState::default();
    sq8_b.prepare_query(q_f32, &mut state_b);
    let norms_b = sq8_b.base_norms_sq().unwrap_or(&[]);
    let way_b_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
        let q_norm = std::hint::black_box(state_b.scalar);
        let q_scaled = std::hint::black_box(state_b.q_shifted.as_slice());
        let mut acc = 0.0_f32;
        for &id in std::hint::black_box(access_ids.as_slice()) {
            acc += l2_squared_sq8_expanded(q_norm, q_scaled, sq8_b.code(id), norms_b[id]);
        }
        acc
    });
    let way_b_res = summarize_kernel_bench(
        "sq8_norm_expanded",
        "SQ8 Way B: Norm-Expanded Asymmetric L2^2 (+4 B/vec)",
        config.dim + std::mem::size_of::<f32>(),
        way_b_runs,
    );
    let way_b_median_ns = way_b_res.median_ns_per_eval;
    kernels.push(way_b_res);

    // 4 & 5. PQ16 and PQ32 ADC kernels when dim is divisible
    for m_pq in [16_usize, 32_usize] {
        if !config.dim.is_multiple_of(m_pq) {
            continue;
        }
        let pq_cfg = PqConfig {
            m_pq,
            seed: config.seed,
            max_train_vectors: config.num_vectors.min(PQ_DEFAULT_MAX_TRAIN_VECTORS),
            max_lloyd_iters: 15,
            ..PqConfig::default()
        };
        let pq_vecs = PqVectors::from_matrix(&base, Metric::L2Squared, &pq_cfg)?;
        let mut pq_state = PqQueryState::with_subspaces(m_pq);
        pq_vecs.prepare_query(q_f32, &mut pq_state);
        let pq_runs = time_kernel_repetitions(config.num_evals, config.reps, || {
            let tbl = std::hint::black_box(pq_state.adc_table.as_slice());
            let mut acc = 0.0_f32;
            for &id in std::hint::black_box(access_ids.as_slice()) {
                acc += pq_adc_distance(tbl, pq_vecs.code(id));
            }
            acc
        });
        kernels.push(summarize_kernel_bench(
            &format!("pq{m_pq}_adc"),
            &format!("PQ{m_pq} ADC Lookup Table Distance ({m_pq} B/vec)"),
            m_pq,
            pq_runs,
        ));
    }

    let speedup_b_over_a = if way_b_median_ns > 0.0 {
        way_a_median_ns / way_b_median_ns
    } else {
        1.0
    };
    let output_path = config.resolved_output_path();
    let report = Phase6MicrobenchReport {
        output_path: output_path.clone(),
        dim: config.dim,
        num_vectors: config.num_vectors,
        num_evals: config.num_evals,
        reps: config.reps,
        seed: config.seed,
        kernels,
        standard_max_rel_err_way_a: std_rel_a,
        standard_max_rel_err_way_b: std_rel_b,
        adversarial_offset_max_abs_err_way_a: adv_abs_a,
        adversarial_offset_max_abs_err_way_b: adv_abs_b,
        speedup_way_b_over_way_a: speedup_b_over_a,
        recommended_default_kernel: "query_shifted".to_string(),
    };

    let json = serialize_microbench_json(&report, &prov, &hw);
    write_file_with_parents(&output_path, json)?;
    eprintln!(
        "[INFO] Wrote Phase 6 kernel microbenchmark JSON to {}",
        output_path.display()
    );
    Ok(report)
}

fn serialize_microbench_json(
    r: &Phase6MicrobenchReport,
    prov: &ReportProvenance,
    hw: &HardwareInfo,
) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("{\n");
    out.push_str("  \"phase\": 6,\n");
    out.push_str("  \"benchmark\": \"kernel_microbenchmark\",\n");
    out.push_str(&format!(
        "  \"generated_at\": \"{}\",\n",
        escape_json(&current_utc_rfc3339())
    ));
    prov.append_json_fields(&mut out, hw);
    append_hardware_json(&mut out, hw);
    out.push_str("  \"measured_this_run\": true,\n");
    out.push_str("  \"warmup_performed\": true,\n");
    out.push_str("  \"parameters\": {\n");
    out.push_str(&format!("    \"dim\": {},\n", r.dim));
    out.push_str(&format!("    \"num_vectors\": {},\n", r.num_vectors));
    out.push_str(&format!("    \"num_evals\": {},\n", r.num_evals));
    out.push_str(&format!("    \"reps\": {},\n", r.reps));
    out.push_str(&format!("    \"seed\": {}\n", r.seed));
    out.push_str("  },\n");

    out.push_str("  \"kernels\": [\n");
    for (idx, k) in r.kernels.iter().enumerate() {
        let comma = if idx + 1 < r.kernels.len() { "," } else { "" };
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"kernel_id\": \"{}\",\n",
            escape_json(&k.kernel_id)
        ));
        out.push_str(&format!(
            "      \"label\": \"{}\",\n",
            escape_json(&k.label)
        ));
        out.push_str(&format!(
            "      \"bytes_per_vector\": {},\n",
            k.bytes_per_vector
        ));
        out.push_str(&format!(
            "      \"median_ns_per_eval\": {:.4},\n",
            k.median_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"min_ns_per_eval\": {:.4},\n",
            k.min_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"max_ns_per_eval\": {:.4},\n",
            k.max_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"spread_ns_per_eval\": {:.4},\n",
            k.spread_ns_per_eval
        ));
        out.push_str(&format!(
            "      \"evals_per_sec\": {:.2},\n",
            k.evals_per_sec
        ));
        out.push_str("      \"run_ns_per_eval\": ");
        push_f64_array_prec(&mut out, &k.run_ns_per_eval, 4);
        out.push('\n');
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ],\n");

    out.push_str("  \"way_a_vs_way_b\": {\n");
    out.push_str(&format!(
        "    \"speedup_way_b_over_way_a\": {:.4},\n",
        r.speedup_way_b_over_way_a
    ));
    out.push_str(&format!(
        "    \"standard_max_rel_err_way_a\": {:.12e},\n",
        r.standard_max_rel_err_way_a
    ));
    out.push_str(&format!(
        "    \"standard_max_rel_err_way_b\": {:.12e},\n",
        r.standard_max_rel_err_way_b
    ));
    out.push_str(&format!(
        "    \"adversarial_offset_max_abs_err_way_a\": {:.12e},\n",
        r.adversarial_offset_max_abs_err_way_a
    ));
    out.push_str(&format!(
        "    \"adversarial_offset_max_abs_err_way_b\": {:.12e},\n",
        r.adversarial_offset_max_abs_err_way_b
    ));
    out.push_str(&format!(
        "    \"recommended_default_kernel\": \"{}\"\n",
        escape_json(&r.recommended_default_kernel)
    ));
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}
