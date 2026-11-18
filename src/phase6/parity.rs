//! Phase 6 cross-language NumPy parity check (`--parity-check`).

use crate::dataset::read_fvecs;
use crate::distance::Metric;
use crate::error::{Result, VectorSearchError};
use crate::exact_knn::ExactKnn;
use crate::hnsw::HnswConfig;
use crate::json_util::write_file_with_parents;
use crate::matrix::RowMatrix;
use crate::quantized::{
    HnswPqIndex, HnswSq8Index, Neighbor, PqConfig, PqVectors, QuantizedBuildMode,
    QuantizedDistanceSource, QuantizedStorageMode, Sq8Vectors, PQ_K_CENTROIDS,
};

use super::config::Phase6ParityConfig;

fn canonical_topk_from_distances(dists: &[f32], k: usize) -> Vec<usize> {
    let mut pairs: Vec<(f32, usize)> = dists
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();
    pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    pairs.into_iter().take(k).map(|(_, id)| id).collect()
}

fn eval_exhaustive_distances<S: QuantizedDistanceSource>(
    source: &S,
    queries: &RowMatrix<f32>,
    k: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let num_base = source.len();
    let num_queries = queries.rows();
    let mut state = S::QueryState::default();
    let mut dists = Vec::with_capacity(num_queries);
    let mut exhaustive_topk = Vec::with_capacity(num_queries);
    for q_row in queries.iter_rows() {
        source.prepare_query(q_row, &mut state);
        let row_dists: Vec<f32> = (0..num_base)
            .map(|id| source.query_distance(&state, id))
            .collect();
        exhaustive_topk.push(canonical_topk_from_distances(&row_dists, k));
        dists.push(row_dists);
    }
    (dists, exhaustive_topk)
}

fn extract_neighbor_ids(rows: &[Vec<Neighbor>]) -> Vec<Vec<usize>> {
    rows.iter()
        .map(|row| row.iter().map(|n| n.id).collect())
        .collect()
}

fn format_f32_json(val: f32) -> Result<String> {
    if !val.is_finite() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Cannot serialize non-finite f32 value ({val}) into parity JSON"
        )));
    }
    Ok(format!("{val:?}"))
}

fn push_f32_array(out: &mut String, vals: &[f32]) -> Result<()> {
    out.push('[');
    for (idx, &v) in vals.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&format_f32_json(v)?);
    }
    out.push(']');
    Ok(())
}

fn push_usize_2d_array(out: &mut String, rows: &[Vec<usize>], indent: &str) {
    out.push_str("[\n");
    for (ri, row) in rows.iter().enumerate() {
        out.push_str(indent);
        out.push_str("  [");
        for (ci, &val) in row.iter().enumerate() {
            if ci > 0 {
                out.push_str(", ");
            }
            out.push_str(&val.to_string());
        }
        out.push(']');
        if ri + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

fn push_u8_2d_array(out: &mut String, data: &[u8], rows: usize, cols: usize, indent: &str) {
    out.push_str("[\n");
    for ri in 0..rows {
        out.push_str(indent);
        out.push_str("  [");
        let row = &data[ri * cols..(ri + 1) * cols];
        for (ci, &val) in row.iter().enumerate() {
            if ci > 0 {
                out.push_str(", ");
            }
            out.push_str(&val.to_string());
        }
        out.push(']');
        if ri + 1 < rows {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

fn push_f32_2d_array(out: &mut String, rows: &[Vec<f32>], indent: &str) -> Result<()> {
    out.push_str("[\n");
    for (ri, row) in rows.iter().enumerate() {
        out.push_str(indent);
        out.push_str("  ");
        push_f32_array(out, row)?;
        if ri + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
    Ok(())
}

/// Runs `--parity-check` exporting deterministic SQ8 + PQ encodings, distances, and top-`k` IDs.
pub fn run_phase6_parity_check(config: &Phase6ParityConfig) -> Result<()> {
    let base = read_fvecs(&config.base_fvecs)?;
    let queries = read_fvecs(&config.query_fvecs)?;
    if base.rows() == 0 || queries.rows() == 0 || base.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "Base and query matrices for --parity-check must be non-empty".to_string(),
        ));
    }
    if base.cols() != queries.cols() {
        return Err(VectorSearchError::DimensionMismatch {
            expected: base.cols(),
            actual: queries.cols(),
        });
    }
    let num_base = base.rows();
    let num_queries = queries.rows();
    let dim = base.cols();
    let eff_k = config.k.min(num_base);

    let exact_gt =
        ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &queries, eff_k, 0)?;
    let exact_gt_rows: Vec<Vec<usize>> = (0..num_queries)
        .map(|qi| {
            exact_gt
                .indices
                .row(qi)
                .iter()
                .map(|&v| v as usize)
                .collect()
        })
        .collect();

    let hnsw_cfg = HnswConfig {
        m: 16,
        ef_construction: 100,
        seed: config.seed,
        ..HnswConfig::default()
    };

    // 1. SQ8 Calibration, Encode, Exhaustive Distances, and HNSW Index Search
    let sq8_vecs = Sq8Vectors::from_matrix(&base, Metric::L2Squared)?;
    let (sq8_dists, sq8_exhaustive_topk) = eval_exhaustive_distances(&sq8_vecs, &queries, eff_k);

    let sq8_idx_only = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        hnsw_cfg.clone(),
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )?;
    let mut sq8_scratch = sq8_idx_only.new_search_scratch(config.ef.max(eff_k));
    let sq8_only_res = sq8_idx_only.search_batch_with_scratch(
        &queries,
        eff_k,
        config.ef,
        None,
        &mut sq8_scratch,
    )?;
    let sq8_rerank_res = sq8_idx_only.search_batch_with_base_timed_with_scratch(
        &base,
        &queries,
        eff_k,
        config.ef,
        Some(config.ef.max(eff_k)),
        &mut sq8_scratch,
    )?;
    let sq8_only_ids = extract_neighbor_ids(&sq8_only_res);
    let sq8_rerank_ids = extract_neighbor_ids(&sq8_rerank_res.results);

    // 2. PQ Training, Encode, Exhaustive ADC Distances, and HNSW Index Search
    let pq_cfg = PqConfig {
        m_pq: config.m_pq,
        k_centroids: PQ_K_CENTROIDS,
        max_train_vectors: config.max_train_vectors,
        max_lloyd_iters: config.max_lloyd_iters,
        seed: config.seed,
    };
    let _d_sub = pq_cfg.validate_for_dim(dim)?;
    let pq_vecs = PqVectors::from_matrix(&base, Metric::L2Squared, &pq_cfg)?;
    let (pq_dists, pq_exhaustive_topk) = eval_exhaustive_distances(&pq_vecs, &queries, eff_k);

    let pq_idx_only = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        hnsw_cfg,
        pq_cfg,
        QuantizedBuildMode::Float32GraphThenQuantize,
        QuantizedStorageMode::QuantizedOnly,
    )?;
    let mut pq_scratch = pq_idx_only.new_search_scratch(config.ef.max(eff_k));
    let pq_only_res =
        pq_idx_only.search_batch_with_scratch(&queries, eff_k, config.ef, None, &mut pq_scratch)?;
    let pq_rerank_res = pq_idx_only.search_batch_with_base_timed_with_scratch(
        &base,
        &queries,
        eff_k,
        config.ef,
        Some(config.ef.max(eff_k)),
        &mut pq_scratch,
    )?;
    let pq_only_ids = extract_neighbor_ids(&pq_only_res);
    let pq_rerank_ids = extract_neighbor_ids(&pq_rerank_res.results);

    // Serialize JSON
    let mut out = String::with_capacity(64 * 1024);
    out.push_str("{\n");
    out.push_str("  \"command\": \"parity_check\",\n");
    out.push_str(&format!("  \"num_base\": {num_base},\n"));
    out.push_str(&format!("  \"num_queries\": {num_queries},\n"));
    out.push_str(&format!("  \"dim\": {dim},\n"));
    out.push_str(&format!("  \"k\": {eff_k},\n"));
    out.push_str(&format!("  \"ef\": {},\n", config.ef));
    out.push_str(&format!("  \"m_pq\": {},\n", config.m_pq));
    out.push_str(&format!("  \"seed\": {},\n", config.seed));
    out.push_str("  \"exact_gt_topk_ids\": ");
    push_usize_2d_array(&mut out, &exact_gt_rows, "  ");
    out.push_str(",\n");

    let quant = sq8_vecs.quantizer();
    out.push_str("  \"sq8\": {\n");
    out.push_str("    \"mins\": ");
    push_f32_array(&mut out, quant.mins())?;
    out.push_str(",\n");
    out.push_str("    \"scales\": ");
    push_f32_array(&mut out, quant.scales())?;
    out.push_str(",\n");
    out.push_str("    \"inv_scales\": ");
    push_f32_array(&mut out, quant.inv_scales())?;
    out.push_str(",\n");
    out.push_str("    \"codes\": ");
    push_u8_2d_array(&mut out, sq8_vecs.codes(), num_base, dim, "    ");
    out.push_str(",\n");
    out.push_str("    \"asymmetric_l2_sq\": ");
    push_f32_2d_array(&mut out, &sq8_dists, "    ")?;
    out.push_str(",\n");
    out.push_str("    \"exhaustive_topk_ids\": ");
    push_usize_2d_array(&mut out, &sq8_exhaustive_topk, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_only_topk_ids\": ");
    push_usize_2d_array(&mut out, &sq8_only_ids, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_rerank_topk_ids\": ");
    push_usize_2d_array(&mut out, &sq8_rerank_ids, "    ");
    out.push('\n');
    out.push_str("  },\n");

    let cb = pq_vecs.codebook();
    out.push_str("  \"pq\": {\n");
    out.push_str(&format!("    \"m_pq\": {},\n", cb.m_pq()));
    out.push_str(&format!("    \"k_centroids\": {},\n", cb.k_centroids()));
    out.push_str(&format!("    \"k_effective\": {},\n", cb.k_effective()));
    out.push_str(&format!("    \"d_sub\": {},\n", cb.d_sub()));
    out.push_str("    \"centroids\": [\n");
    for m in 0..cb.m_pq() {
        out.push_str("      [\n");
        for c in 0..cb.k_centroids() {
            out.push_str("        ");
            push_f32_array(&mut out, cb.centroid(m, c))?;
            if c + 1 < cb.k_centroids() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("      ]");
        if m + 1 < cb.m_pq() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("    ],\n");
    out.push_str("    \"codes\": ");
    push_u8_2d_array(&mut out, pq_vecs.codes(), num_base, cb.m_pq(), "    ");
    out.push_str(",\n");
    out.push_str("    \"adc_distances\": ");
    push_f32_2d_array(&mut out, &pq_dists, "    ")?;
    out.push_str(",\n");
    out.push_str("    \"exhaustive_topk_ids\": ");
    push_usize_2d_array(&mut out, &pq_exhaustive_topk, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_only_topk_ids\": ");
    push_usize_2d_array(&mut out, &pq_only_ids, "    ");
    out.push_str(",\n");
    out.push_str("    \"hnsw_rerank_topk_ids\": ");
    push_usize_2d_array(&mut out, &pq_rerank_ids, "    ");
    out.push('\n');
    out.push_str("  }\n");
    out.push_str("}\n");

    if let Some(ref json_out) = config.json_out {
        write_file_with_parents(json_out, &out)?;
    } else {
        print!("{out}");
    }
    Ok(())
}
