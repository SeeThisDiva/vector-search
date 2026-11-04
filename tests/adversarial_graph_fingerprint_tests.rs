//! Adversarial challenge test suite for HNSW Graph Fingerprinting and Golden Fixtures.
//!
//! Evaluates:
//! 1. `HnswIndex::graph_fingerprint()` sensitivity to:
//!    - Single-edge swaps, additions, deletions, and reversals
//!    - Node layer promotion/demotion
//!    - Entry point variations (and architectural blindness to entry point)
//!    - Empty indices and single-element graphs (panic safety)
//!    - Deterministic seed variations and collision resistance
//! 2. Golden Fixtures & `--repro-check` verification:
//!    - Sensitivity to graph fingerprint corruption
//!    - Sensitivity (and loophole bypass) to query results hash corruption
//!    - Failure to validate detailed `results` array (distance bits / IDs)

use std::fs;
use vector_search::benchmark::{
    compute_query_results_hash, verify_index_against_golden_fixture, GoldenFixture,
};
use vector_search::dataset::generate_synthetic_sift;
use vector_search::distance::Metric;
use vector_search::error::VectorSearchError;
use vector_search::hnsw::{fnv1a_64_update_u64, HnswConfig, HnswIndex, FNV_OFFSET_BASIS_64};
use vector_search::matrix::RowMatrix;

/// Helper function to compute fingerprint matching `HnswIndex::graph_fingerprint` logic
/// directly on synthetic topology: (num_vectors, enter_point, Vec<(max_layer, Vec<Vec<usize>>)>).
fn compute_fingerprint_from_topology(
    num_vectors: usize,
    enter_point: Option<usize>,
    nodes: &[(usize, Vec<Vec<usize>>)],
) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    fnv1a_64_update_u64(&mut hash, num_vectors as u64);
    let ep_val = enter_point.map(|ep| ep as u64).unwrap_or(u64::MAX);
    fnv1a_64_update_u64(&mut hash, ep_val);
    for (max_l, layers) in nodes {
        fnv1a_64_update_u64(&mut hash, *max_l as u64);
        for lc in 0..=*max_l {
            if let Some(nbrs) = layers.get(lc) {
                for &nid in nbrs {
                    fnv1a_64_update_u64(&mut hash, nid as u64);
                }
            }
        }
    }
    hash
}

// ============================================================================
// 1. Empty & Single-Element Panic Safety
// ============================================================================

#[test]
fn test_adv_fingerprint_empty_index_panic_safety() {
    let empty_matrix = RowMatrix::<f32>::empty();
    let cfg = HnswConfig::default();
    let index = HnswIndex::build(&empty_matrix, Metric::L2Squared, cfg).unwrap();

    assert_eq!(index.num_vectors(), 0);
    assert_eq!(index.entry_point(), None);

    // Fingerprint must not panic and must equal FNV-1a basis updated with 0_u64 and u64::MAX
    let mut expected = FNV_OFFSET_BASIS_64;
    fnv1a_64_update_u64(&mut expected, 0);
    fnv1a_64_update_u64(&mut expected, u64::MAX);
    assert_eq!(index.graph_fingerprint(), expected);
}

#[test]
fn test_adv_fingerprint_single_vector_panic_safety() {
    let single = RowMatrix::from_rows(&[vec![1.0_f32, 2.0, 3.0, 4.0]]).unwrap();
    let cfg = HnswConfig::default();
    let index = HnswIndex::build(&single, Metric::L2Squared, cfg.clone()).unwrap();

    assert_eq!(index.num_vectors(), 1);
    assert_eq!(index.entry_point(), Some(0));

    let fp = index.graph_fingerprint();
    assert_ne!(fp, 0);
    assert_ne!(fp, FNV_OFFSET_BASIS_64);

    // Verify consistency: second build produces exact same fingerprint
    let index2 = HnswIndex::build(&single, Metric::L2Squared, cfg).unwrap();
    assert_eq!(index.graph_fingerprint(), index2.graph_fingerprint());
}

// ============================================================================
// 2. Single-Edge and Topology Perturbation Sensitivity (1,000 Adversarial Trials)
// ============================================================================

#[test]
fn test_adv_fingerprint_single_edge_perturbations_zero_collisions() {
    // Construct a base graph topology of 20 nodes with multiple layers
    let n = 20;
    let mut base_nodes: Vec<(usize, Vec<Vec<usize>>)> = Vec::with_capacity(n);
    for i in 0..n {
        let max_l = if i % 5 == 0 {
            2
        } else if i % 2 == 0 {
            1
        } else {
            0
        };
        let mut layers = Vec::new();
        for lc in 0..=max_l {
            let mut nbrs = Vec::new();
            for step in 1..=4 {
                let neighbor = (i + step * (lc + 1)) % n;
                if neighbor != i && !nbrs.contains(&neighbor) {
                    nbrs.push(neighbor);
                }
            }
            layers.push(nbrs);
        }
        base_nodes.push((max_l, layers));
    }

    let ep = Some(0);
    let base_fp = compute_fingerprint_from_topology(n, ep, &base_nodes);

    let mut seen_hashes = std::collections::HashSet::new();
    seen_hashes.insert(base_fp);

    let mut trials = 0;

    // Perturbation 1: Single edge swap (replace neighbor with another)
    for node_id in 0..n {
        let max_l = base_nodes[node_id].0;
        for lc in 0..=max_l {
            let nbrs_len = base_nodes[node_id].1[lc].len();
            for idx in 0..nbrs_len {
                let orig_nbr = base_nodes[node_id].1[lc][idx];
                for cand in 0..n {
                    if cand != node_id && cand != orig_nbr {
                        let mut perturbed = base_nodes.clone();
                        perturbed[node_id].1[lc][idx] = cand;
                        let p_fp = compute_fingerprint_from_topology(n, ep, &perturbed);
                        assert_ne!(
                            p_fp, base_fp,
                            "Single edge swap at node {node_id} lc {lc} idx {idx} to {cand} collided with base!"
                        );
                        seen_hashes.insert(p_fp);
                        trials += 1;
                    }
                }
            }
        }
    }

    // Perturbation 2: Single edge deletion
    for node_id in 0..n {
        let max_l = base_nodes[node_id].0;
        for lc in 0..=max_l {
            if base_nodes[node_id].1[lc].len() > 1 {
                let mut perturbed = base_nodes.clone();
                perturbed[node_id].1[lc].pop();
                let p_fp = compute_fingerprint_from_topology(n, ep, &perturbed);
                assert_ne!(p_fp, base_fp, "Edge deletion at node {node_id} collided!");
                seen_hashes.insert(p_fp);
                trials += 1;
            }
        }
    }

    // Perturbation 3: Single edge addition
    for node_id in 0..n {
        let max_l = base_nodes[node_id].0;
        for lc in 0..=max_l {
            let mut perturbed = base_nodes.clone();
            perturbed[node_id].1[lc].push(999);
            let p_fp = compute_fingerprint_from_topology(n, ep, &perturbed);
            assert_ne!(p_fp, base_fp, "Edge addition at node {node_id} collided!");
            seen_hashes.insert(p_fp);
            trials += 1;
        }
    }

    // Perturbation 4: Edge reordering (non-commutative sequence test)
    for node_id in 0..n {
        let max_l = base_nodes[node_id].0;
        for lc in 0..=max_l {
            if base_nodes[node_id].1[lc].len() >= 2 {
                let mut perturbed = base_nodes.clone();
                perturbed[node_id].1[lc].swap(0, 1);
                let p_fp = compute_fingerprint_from_topology(n, ep, &perturbed);
                assert_ne!(
                    p_fp, base_fp,
                    "Edge order reversal at node {node_id} collided! Sequence order must matter."
                );
                seen_hashes.insert(p_fp);
                trials += 1;
            }
        }
    }

    // Perturbation 5: Node layer promotion/demotion
    for node_id in 0..n {
        let (max_l, _layers) = &base_nodes[node_id];
        let mut perturbed = base_nodes.clone();
        let new_l = if *max_l > 0 { *max_l - 1 } else { *max_l + 1 };
        perturbed[node_id].0 = new_l;
        if new_l > *max_l {
            perturbed[node_id].1.push(vec![1, 2]);
        }
        let p_fp = compute_fingerprint_from_topology(n, ep, &perturbed);
        assert_ne!(
            p_fp, base_fp,
            "Layer change at node {node_id} ({max_l} -> {new_l}) collided!"
        );
        seen_hashes.insert(p_fp);
        trials += 1;
    }

    // Perturbation 6: Entry point variation
    for ep_cand in 1..n {
        let p_fp = compute_fingerprint_from_topology(n, Some(ep_cand), &base_nodes);
        assert_ne!(p_fp, base_fp, "Entry point change to {ep_cand} collided!");
        seen_hashes.insert(p_fp);
        trials += 1;
    }

    assert!(trials > 500, "Must have executed > 500 adversarial trials");
    // Verify zero collisions among all generated perturbed hashes
    assert_eq!(
        seen_hashes.len(),
        trials + 1,
        "Collision detected among perturbed fingerprints!"
    );
}

// ============================================================================
// 3. Entry Point Sensitivity Challenge (Finding the Architectural Blindness)
// ============================================================================

#[test]
fn test_adv_fingerprint_blindness_to_entry_point() {
    // In HnswIndex::graph_fingerprint(), `enter_point` is now hashed into the stream
    // immediately after `num_vectors`. Shifting entry point MUST produce distinct fingerprints.
    let (base, _) = generate_synthetic_sift(50, 10, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let ep = index.entry_point();
    assert!(ep.is_some());

    // Compute expected fingerprint by formula
    let fp = index.graph_fingerprint();

    let nodes_topology: Vec<(usize, Vec<Vec<usize>>)> = (0..index.num_vectors())
        .map(|i| {
            let max_l = index.node_layer(i);
            let layers = (0..=max_l)
                .map(|lc| index.neighbors(lc, i).to_vec())
                .collect();
            (max_l, layers)
        })
        .collect();

    let computed_fp = compute_fingerprint_from_topology(index.num_vectors(), ep, &nodes_topology);
    assert_eq!(fp, computed_fp);

    // Verify sensitivity: Shifting entry_point from ep to Some(999) produces a DISTINCT fingerprint!
    let alt_ep = if ep == Some(999) { Some(0) } else { Some(999) };
    let alt_fp = compute_fingerprint_from_topology(index.num_vectors(), alt_ep, &nodes_topology);
    assert_ne!(
        fp, alt_fp,
        "Shifting entry point must produce distinct fingerprints!"
    );

    // Verify sensitivity: None entry point produces a DISTINCT fingerprint
    let none_fp = compute_fingerprint_from_topology(index.num_vectors(), None, &nodes_topology);
    assert_ne!(
        fp, none_fp,
        "None entry point must produce distinct fingerprint!"
    );
}

// ============================================================================
// 4. Seed Variation Sensitivity on Real HNSW Indices
// ============================================================================

#[test]
fn test_adv_seed_variation_produces_distinct_fingerprints() {
    let (base, _) = generate_synthetic_sift(100, 10, 16, 3, 42).unwrap();
    let seeds = [1_u64, 42, 100, 999, 1234567];
    let mut fps = std::collections::HashSet::new();

    for &seed in &seeds {
        let cfg = HnswConfig {
            m: 8,
            m_max0: Some(16),
            ef_construction: 50,
            seed,
            ..Default::default()
        };
        let index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
        let fp = index.graph_fingerprint();
        assert!(
            fps.insert(fp),
            "Seed {seed} produced duplicate fingerprint 0x{fp:016x}!"
        );
    }
}

// ============================================================================
// 5. Golden Fixtures Verification & Loophole Exposure
// ============================================================================

#[test]
fn test_adv_golden_fixture_corruption_fingerprint_mismatch() {
    let (base, queries) = generate_synthetic_sift(100, 20, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let actual_fp = index.graph_fingerprint();

    // Create a golden fixture JSON with corrupted graph_fingerprint
    let corrupt_fp = actual_fp ^ 0xdeadbeef;
    let golden_json = format!(
        r#"{{
  "scale_vectors": 100,
  "graph_fingerprint": {},
  "query_results_hash": 12345,
  "num_queries": 20
}}"#,
        corrupt_fp
    );

    let temp_dir = std::env::temp_dir();
    let temp_golden = temp_dir.join("test_adv_corrupt_fp.json");
    fs::write(&temp_golden, golden_json).unwrap();

    let res = verify_index_against_golden_fixture(&index, &queries, &temp_golden);
    let _ = fs::remove_file(&temp_golden);

    assert!(
        res.is_err(),
        "Must fail when graph_fingerprint is corrupted"
    );
    match res.unwrap_err() {
        VectorSearchError::VerificationFailed(msg) => {
            assert!(msg.contains("Golden graph_fingerprint mismatch"));
        }
        other => panic!("Unexpected error type: {other:?}"),
    }
}

#[test]
fn test_adv_golden_fixture_corruption_query_hash_when_queries_match() {
    let (base, queries) = generate_synthetic_sift(100, 50, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let actual_fp = index.graph_fingerprint();

    // Compute real query hash for 50 queries
    let timed = index.search_batch_timed(&queries, 10, 100).unwrap();
    let actual_qhash = compute_query_results_hash(&timed.result.indices, &timed.result.distances);

    // Corrupt query_results_hash
    let corrupt_qhash = actual_qhash ^ 0x1337;
    let golden_json = format!(
        r#"{{
  "scale_vectors": 100,
  "graph_fingerprint": {},
  "query_results_hash": {},
  "num_queries": 50
}}"#,
        actual_fp, corrupt_qhash
    );

    let temp_dir = std::env::temp_dir();
    let temp_golden = temp_dir.join("test_adv_corrupt_qhash.json");
    fs::write(&temp_golden, golden_json).unwrap();

    let res = verify_index_against_golden_fixture(&index, &queries, &temp_golden);
    let _ = fs::remove_file(&temp_golden);

    assert!(
        res.is_err(),
        "Must fail when query_results_hash is corrupted and queries.rows() >= num_queries"
    );
    match res.unwrap_err() {
        VectorSearchError::VerificationFailed(msg) => {
            assert!(msg.contains("Golden query_results_hash mismatch"));
        }
        other => panic!("Unexpected error type: {other:?}"),
    }
}

#[test]
fn test_adv_golden_fixture_loophole_query_hash_bypassed_when_queries_under_expected() {
    // EMPIRICAL DEMONSTRATION OF THE BUG / HOLE:
    // When `--repro-check` runs in phase3, `repro_q` is capped at 100:
    //   `let repro_q = 100.min(loaded.query.rows());`
    // But `golden_10k.json` was generated with:
    //   `"num_queries": 1000`
    // In `verify_index_against_golden_fixture`:
    //   `if queries.rows() >= expected_nq && expected_nq > 0 { ... }`
    // Because 100 < 1000, the condition is FALSE.
    // The query hash check is COMPLETELY BYPASSED!
    let (base, queries_100) = generate_synthetic_sift(100, 100, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let actual_fp = index.graph_fingerprint();

    // Fabricate a golden fixture with totally bogus query_results_hash and num_queries=1000
    let bogus_qhash = 0xdeadbeefcafebabe_u64;
    let golden_json = format!(
        r#"{{
  "scale_vectors": 100,
  "graph_fingerprint": {},
  "query_results_hash": {},
  "num_queries": 1000
}}"#,
        actual_fp, bogus_qhash
    );

    let temp_dir = std::env::temp_dir();
    let temp_golden = temp_dir.join("test_adv_bypass_qhash.json");
    fs::write(&temp_golden, golden_json).unwrap();

    // Passes queries_100 (which only has 100 rows, < 1000)
    let res = verify_index_against_golden_fixture(&index, &queries_100, &temp_golden);
    let _ = fs::remove_file(&temp_golden);

    // Fail closed: must return Err when queries.rows() < expected_nq
    assert!(
        res.is_err(),
        "verify_index_against_golden_fixture must fail closed when queries.rows() < expected_nq"
    );
    match res.unwrap_err() {
        VectorSearchError::VerificationFailed(msg) => {
            assert!(
                msg.contains("requires 1000 queries for query_results_hash"),
                "Expected query count error message, got: {msg}"
            );
        }
        other => panic!("Unexpected error type: {other:?}"),
    }
}

#[test]
fn test_adv_golden_fixture_results_array_corruption_detected() {
    let (base, queries) = generate_synthetic_sift(100, 10, 16, 2, 42).unwrap();
    let cfg = HnswConfig {
        m: 8,
        m_max0: Some(16),
        ef_construction: 50,
        seed: 42,
        ..Default::default()
    };
    let mut index = HnswIndex::build(&base, Metric::L2Squared, cfg).unwrap();
    let actual_fp = index.graph_fingerprint();

    // 1. Corrupted candidate IDs must fail verification
    let golden_json = format!(
        r#"{{
  "scale_vectors": 100,
  "graph_fingerprint": {},
  "results": [
    {{
      "query_id": 0,
      "indices": [999999, 888888],
      "distance_bits": [0, 0],
      "distances": [999999.0, 999999.0]
    }}
  ]
}}"#,
        actual_fp
    );

    let temp_dir = std::env::temp_dir();
    let temp_golden = temp_dir.join("test_adv_corrupt_results_array.json");
    fs::write(&temp_golden, golden_json).unwrap();

    let res = verify_index_against_golden_fixture(&index, &queries, &temp_golden);
    let _ = fs::remove_file(&temp_golden);

    assert!(
        res.is_err(),
        "verify_index_against_golden_fixture must detect corrupted candidate IDs in results array"
    );
    match res.unwrap_err() {
        VectorSearchError::VerificationFailed(msg) => {
            assert!(
                msg.contains("Golden candidate ID mismatch"),
                "Expected candidate ID mismatch error, got: {msg}"
            );
        }
        other => panic!("Unexpected error type: {other:?}"),
    }

    // 2. Corrupted distance bits (with correct IDs) must also fail verification
    let timed = index
        .search_batch(&queries.slice_first_rows(1), 2, 100)
        .unwrap();
    let corrupt_bits_json = format!(
        r#"{{
  "scale_vectors": 100,
  "graph_fingerprint": {},
  "results": [
    {{
      "query_id": 0,
      "indices": [{}, {}],
      "distance_bits": [{}, {}]
    }}
  ]
}}"#,
        actual_fp,
        timed.indices.row(0)[0],
        timed.indices.row(0)[1],
        timed.distances.row(0)[0].to_bits() ^ 0x01,
        timed.distances.row(0)[1].to_bits(),
    );
    let temp_bits = temp_dir.join("test_adv_corrupt_bits.json");
    fs::write(&temp_bits, corrupt_bits_json).unwrap();
    let res_bits = verify_index_against_golden_fixture(&index, &queries, &temp_bits);
    let _ = fs::remove_file(&temp_bits);
    assert!(
        res_bits.is_err(),
        "verify_index_against_golden_fixture must detect corrupted distance bits"
    );

    // 3. Valid results array must succeed
    let valid_json = format!(
        r#"{{
  "scale_vectors": 100,
  "graph_fingerprint": {},
  "results": [
    {{
      "query_id": 0,
      "indices": [{}, {}],
      "distance_bits": [{}, {}]
    }}
  ]
}}"#,
        actual_fp,
        timed.indices.row(0)[0],
        timed.indices.row(0)[1],
        timed.distances.row(0)[0].to_bits(),
        timed.distances.row(0)[1].to_bits(),
    );
    let temp_valid = temp_dir.join("test_adv_valid_results_array.json");
    fs::write(&temp_valid, valid_json).unwrap();
    let res_valid = verify_index_against_golden_fixture(&index, &queries, &temp_valid);
    let _ = fs::remove_file(&temp_valid);
    assert!(
        res_valid.is_ok(),
        "Valid results array must pass verification"
    );
}

#[test]
fn test_adv_golden_fixture_parse_json_does_not_verify_hash_consistency() {
    // When GoldenFixture::parse_json parses a JSON string, it does NOT verify
    // that query_results_hash matches the distance_bits or indices in `results`.
    // It is purely a deserializer without internal cross-validation.
    let bogus_json = r#"{
  "scale_vectors": 100,
  "graph_fingerprint": 12345,
  "query_results_hash": 67890,
  "num_queries": 1,
  "results": [
    {
      "query_id": 0,
      "indices": [1, 2, 3],
      "distance_bits": [100, 200, 300],
      "distances": [1.0, 2.0, 3.0]
    }
  ]
}"#;

    let parsed = GoldenFixture::parse_json(bogus_json);
    assert!(parsed.is_ok());
    let fixture = parsed.unwrap();
    assert_eq!(fixture.query_results_hash, 67890);
    assert_eq!(fixture.results.len(), 1);
    // Notice: 67890 is completely arbitrary and does not match the results, but parse_json succeeds.
}
