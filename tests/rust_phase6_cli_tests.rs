//! Integration and CLI tests for the Phase 6 benchmark runner (`phase6`).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use vector_search::benchmark::detect_provenance;
use vector_search::dataset::{generate_synthetic_sift, write_fvecs};
use vector_search::distance::Metric;
use vector_search::hnsw::HnswConfig;
use vector_search::matrix::RowMatrix;
use vector_search::phase6::{can_resume_existing_arm_report, resolve_arm_spec, RerankDepthSpec};
use vector_search::quantized::{
    HnswPqIndex, HnswSq8Index, PqConfig, QuantizedBuildMode, QuantizedStorageMode,
};

// ============================================================================
// Self-contained RAII TempDir & Binary Runner
// ============================================================================

static TEMP_SEQ: AtomicUsize = AtomicUsize::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "vector_search_{prefix}_{}_{nanos}_{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self { path: dir }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn phase6_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_phase6"))
}

fn run_phase6(args: &[&str]) -> Output {
    Command::new(phase6_bin_path())
        .args(args)
        .output()
        .expect("spawn phase6 binary")
}

// ============================================================================
// Strict JSON Parser (rejects duplicate object keys)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn as_object(&self) -> &BTreeMap<String, JsonValue> {
        match self {
            Self::Object(map) => map,
            other => panic!("Expected JSON object, got {other:?}"),
        }
    }

    fn as_array(&self) -> &[JsonValue] {
        match self {
            Self::Array(arr) => arr,
            other => panic!("Expected JSON array, got {other:?}"),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Str(s) => s,
            other => panic!("Expected JSON string, got {other:?}"),
        }
    }

    fn as_f64(&self) -> f64 {
        match self {
            Self::Number(v) => *v,
            other => panic!("Expected JSON number, got {other:?}"),
        }
    }

    fn as_usize(&self) -> usize {
        let v = self.as_f64();
        assert!(v >= 0.0 && (v - v.round()).abs() < 1e-9, "Not usize: {v}");
        v.round() as usize
    }

    fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            other => panic!("Expected JSON bool, got {other:?}"),
        }
    }

    fn get(&self, key: &str) -> &JsonValue {
        self.as_object()
            .get(key)
            .unwrap_or_else(|| panic!("Missing JSON key '{key}'"))
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn parse(text: &'a str) -> JsonValue {
        let mut p = Self {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let val = p.parse_value();
        p.skip_ws();
        assert_eq!(p.pos, p.bytes.len(), "Trailing characters in JSON");
        val
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, expected: u8) {
        self.skip_ws();
        assert!(self.pos < self.bytes.len(), "Unexpected EOF");
        assert_eq!(
            self.bytes[self.pos], expected,
            "Expected '{}' at byte {}",
            expected as char, self.pos
        );
        self.pos += 1;
    }

    fn parse_value(&mut self) -> JsonValue {
        match self.peek().expect("Unexpected EOF in JSON value") {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => JsonValue::Str(self.parse_string()),
            b't' => {
                self.consume_literal(b"true");
                JsonValue::Bool(true)
            }
            b'f' => {
                self.consume_literal(b"false");
                JsonValue::Bool(false)
            }
            b'n' => {
                self.consume_literal(b"null");
                JsonValue::Null
            }
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => panic!("Unexpected token '{}' at byte {}", other as char, self.pos),
        }
    }

    fn consume_literal(&mut self, lit: &[u8]) {
        self.skip_ws();
        assert!(self.bytes[self.pos..].starts_with(lit), "Bad JSON literal");
        self.pos += lit.len();
    }

    fn parse_string(&mut self) -> String {
        self.expect(b'"');
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return out;
            }
            if b == b'\\' {
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => panic!("Unsupported escape \\{}", other as char),
                }
            } else {
                out.push(b as char);
            }
        }
        panic!("Unterminated JSON string");
    }

    fn parse_number(&mut self) -> JsonValue {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).expect("utf8 number");
        JsonValue::Number(s.parse::<f64>().expect("parse f64"))
    }

    fn parse_array(&mut self) -> JsonValue {
        self.expect(b'[');
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return JsonValue::Array(items);
        }
        loop {
            items.push(self.parse_value());
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                other => panic!("Expected ',' or ']' in array, got {other:?}"),
            }
        }
        JsonValue::Array(items)
    }

    fn parse_object(&mut self) -> JsonValue {
        self.expect(b'{');
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return JsonValue::Object(map);
        }
        loop {
            let key = self.parse_string();
            assert!(
                !map.contains_key(&key),
                "Duplicate JSON key '{key}' emitted"
            );
            self.expect(b':');
            let val = self.parse_value();
            map.insert(key, val);
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                other => panic!("Expected ',' or '}}' in object, got {other:?}"),
            }
        }
        JsonValue::Object(map)
    }
}

fn read_json_file(path: &Path) -> JsonValue {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed reading JSON '{}': {e}", path.display()));
    JsonParser::parse(&text)
}

// ============================================================================
// CLI Validation & Dataset Strictness Tests
// ============================================================================

#[test]
fn test_phase6_help_and_invalid_cli_errors() {
    let help = run_phase6(&["--help"]);
    assert!(
        help.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&help.stderr)
    );
    let stdout = String::from_utf8_lossy(&help.stdout);
    for expected in [
        "--microbench",
        "--pilot",
        "--matrix",
        "--parity-check",
        "--internal-profile-query",
        "sq8_only_qbuild",
        "pq16_rerank_f32graph",
    ] {
        assert!(stdout.contains(expected), "Help missing '{expected}'");
    }

    let bogus = run_phase6(&["--definitely-unknown-flag"]);
    assert!(!bogus.status.success(), "Unknown flag must fail");

    let tmp = TempDir::new("phase6_strict");
    let data_dir = tmp.path().join("empty_data");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&data_dir).expect("create empty data dir");

    // Unknown arm must fail cleanly.
    let bad_arm = run_phase6(&[
        "--matrix",
        "--synthetic",
        "--scale",
        "40",
        "--queries",
        "4",
        "--arms",
        "nonexistent_arm",
        "--output-dir",
        out_dir.to_str().unwrap(),
    ]);
    assert!(!bad_arm.status.success(), "Unknown arm must fail");

    // Empty data-dir + --offline without --synthetic MUST fail loudly (never silent fallback).
    let missing_sift = run_phase6(&[
        "--matrix",
        "--dataset",
        "sift1m",
        "--scale",
        "100",
        "--queries",
        "10",
        "--arms",
        "sq8_only_f32graph",
        "--ef-search-list",
        "10",
        "--offline",
        "--data-dir",
        data_dir.to_str().unwrap(),
        "--output-dir",
        out_dir.to_str().unwrap(),
    ]);
    assert!(
        !missing_sift.status.success(),
        "Missing SIFT1M without --synthetic must error instead of falling back silently"
    );
    let stderr = String::from_utf8_lossy(&missing_sift.stderr);
    assert!(
        stderr.contains("Authentic") || stderr.contains("missing") || stderr.contains("not found"),
        "Expected clear missing-dataset error message, got: {stderr}"
    );
}

// ============================================================================
// Microbenchmark CLI Test (Way A vs Way B + PQ16 + PQ32 + f32)
// ============================================================================

#[test]
fn test_phase6_microbench_cli_writes_way_a_vs_way_b_json() {
    let tmp = TempDir::new("phase6_microbench");
    let json_path = tmp.path().join("microbench_kernels.json");

    let out = run_phase6(&[
        "--microbench",
        "--dim",
        "128",
        "--num-vectors",
        "256",
        "--evals",
        "2000",
        "--reps",
        "2",
        "--output",
        json_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "--microbench failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(json_path.is_file(), "Microbench JSON was not created");

    let doc = read_json_file(&json_path);
    assert_eq!(doc.get("phase").as_usize(), 6);
    assert_eq!(doc.get("benchmark").as_str(), "kernel_microbenchmark");
    assert!(doc.get("measured_this_run").as_bool());
    assert!(doc.get("warmup_performed").as_bool());

    let kernels = doc.get("kernels").as_array();
    let ids: HashSet<&str> = kernels
        .iter()
        .map(|k| k.get("kernel_id").as_str())
        .collect();
    for required in [
        "f32_l2_squared",
        "sq8_query_shifted",
        "sq8_norm_expanded",
        "pq16_adc",
        "pq32_adc",
    ] {
        assert!(ids.contains(required), "Missing kernel '{required}'");
    }
    for k in kernels {
        assert!(k.get("median_ns_per_eval").as_f64() > 0.0);
        assert!(k.get("evals_per_sec").as_f64() > 0.0);
        assert_eq!(k.get("run_ns_per_eval").as_array().len(), 2);
    }

    let ab = doc.get("way_a_vs_way_b");
    assert!(ab.get("speedup_way_b_over_way_a").as_f64() > 0.0);
    assert!(ab.get("standard_max_rel_err_way_a").as_f64() < 0.05);
    assert!(ab.get("standard_max_rel_err_way_b").as_f64() < 0.05);
    assert_eq!(
        ab.get("recommended_default_kernel").as_str(),
        "query_shifted"
    );
    assert!(
        ab.get("adversarial_offset_max_abs_err_way_a").as_f64()
            < ab.get("adversarial_offset_max_abs_err_way_b").as_f64(),
        "Way A should be much tighter than Way B on large-offset adversarial inputs"
    );
}

// ============================================================================
// Matrix + Streaming + Isolated Query RSS + Resume Tests
// ============================================================================

#[test]
fn test_phase6_matrix_synthetic_subset_and_resume() {
    let tmp = TempDir::new("phase6_matrix");
    let data_dir = tmp.path().join("data");
    let out_dir = tmp.path().join("out");

    let arms_arg = "f32_m16,sq8_only_f32graph,sq8_only_qbuild,sq8_rerank_f32graph,pq16_only_f32graph,pq16_rerank_f32graph";
    let base_args = [
        "--matrix",
        "--synthetic",
        "--scale",
        "120",
        "--queries",
        "12",
        "--dim",
        "32",
        "--num-clusters",
        "4",
        "--k",
        "5",
        "--reps",
        "2",
        "--ef-search-list",
        "10,20",
        "--rerank-list",
        "15,full_beam",
        "--arms",
        arms_arg,
        "--data-dir",
        data_dir.to_str().unwrap(),
        "--output-dir",
        out_dir.to_str().unwrap(),
        "--seed",
        "42",
    ];

    let first = run_phase6(&base_args);
    assert!(
        first.status.success(),
        "--matrix synthetic failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );

    let summary_path = out_dir.join("summary_n120.json");
    assert!(summary_path.is_file(), "Missing summary_n120.json");
    let summary = read_json_file(&summary_path);
    assert_eq!(summary.get("phase").as_usize(), 6);
    assert_eq!(summary.get("mode").as_str(), "matrix");
    assert_eq!(summary.get("scale").as_usize(), 120);
    assert_eq!(summary.get("total_arms").as_usize(), 6);
    assert_eq!(summary.get("newly_measured_count").as_usize(), 6);
    assert_eq!(summary.get("resumed_skipped_count").as_usize(), 0);

    for arm in [
        "f32_m16",
        "sq8_only_f32graph",
        "sq8_only_qbuild",
        "sq8_rerank_f32graph",
        "pq16_only_f32graph",
        "pq16_rerank_f32graph",
    ] {
        let p = out_dir.join(format!("arm_n120_{arm}.json"));
        assert!(p.is_file(), "Missing arm report {}", p.display());
        let doc = read_json_file(&p);
        assert_eq!(doc.get("phase").as_usize(), 6);
        assert_eq!(doc.get("arm_id").as_str(), arm);
        assert_eq!(doc.get("scale").as_usize(), 120);
        assert_eq!(doc.get("dim").as_usize(), 32);
        assert_eq!(doc.get("k").as_usize(), 5);
        assert_eq!(doc.get("num_queries").as_usize(), 12);
        assert_eq!(doc.get("reps").as_usize(), 2);
        assert!(doc.get("measured_this_run").as_bool());
        assert!(doc.get("steady_state_bytes_per_vector").as_f64() > 0.0);

        let sweep = doc.get("sweep").as_array();
        if arm.contains("_only_") {
            assert_eq!(sweep.len(), 2);
            assert!(
                doc.get("query_rss_profiled_isolated").as_bool(),
                "{arm} must profile query RSS in isolated child process"
            );
            assert!(doc.get("query_serving_peak_rss_mb").as_f64() > 0.0);
        } else if arm.contains("_rerank_") {
            assert_eq!(sweep.len(), 4);
            let saw_fixed_15 = sweep.iter().any(|pt| {
                matches!(pt.get("rerank_top_r"), JsonValue::Number(v) if (*v - 15.0).abs() < 1e-9)
                    && !pt.get("is_full_beam").as_bool()
            });
            let saw_full_beam = sweep.iter().any(|pt| {
                matches!(pt.get("rerank_top_r"), JsonValue::Str(s) if s == "full_beam")
                    && pt.get("is_full_beam").as_bool()
                    && pt.get("effective_ef_search").as_usize()
                        == pt.get("ef_search").as_usize().max(5)
            });
            assert!(saw_fixed_15, "{arm} missing numeric rerank_top_r=15 point");
            assert!(saw_full_beam, "{arm} missing FullBeam point");
        } else {
            assert_eq!(sweep.len(), 2);
        }

        for pt in sweep {
            let recall = pt.get("recall_at_k").as_f64();
            assert!((0.0..=1.0).contains(&recall), "Bad recall {recall}");
            assert!(pt.get("qps").as_f64() > 0.0);
            let spread = pt.get("spread");
            assert!(spread.get("mean_ms_spread").as_f64() >= 0.0);
            assert!(spread.get("p99_ms_spread").as_f64() >= 0.0);
            assert_eq!(spread.get("run_mean_latencies_ms").as_array().len(), 2);
            assert_eq!(spread.get("run_p99_latencies_ms").as_array().len(), 2);
        }
    }

    // Second run with --resume skips all 6 matching reports if git commit is known.
    let prov = detect_provenance();
    if prov.git_commit != "unknown" {
        let second = run_phase6(&base_args);
        assert!(
            second.status.success(),
            "--matrix resume failed: {}",
            String::from_utf8_lossy(&second.stderr)
        );
        let summary2 = read_json_file(&summary_path);
        assert_eq!(summary2.get("resumed_skipped_count").as_usize(), 6);
        assert_eq!(summary2.get("newly_measured_count").as_usize(), 0);

        // --no-resume forces re-measurement.
        let mut no_resume_args = base_args.to_vec();
        no_resume_args.push("--no-resume");
        let third = run_phase6(&no_resume_args);
        assert!(third.status.success());
        let summary3 = read_json_file(&summary_path);
        assert_eq!(summary3.get("newly_measured_count").as_usize(), 6);
        assert_eq!(summary3.get("resumed_skipped_count").as_usize(), 0);
    }

    // Unit check: dirty reports must be rejected when checking for a clean commit.
    let sq8_report = out_dir.join("arm_n120_sq8_only_f32graph.json");
    let sq8_spec = resolve_arm_spec("sq8_only_f32graph").expect("resolve sq8_only_f32graph");
    assert!(
        !can_resume_existing_arm_report(
            &sq8_report,
            "unknown",
            false,
            "synthetic_clustered",
            120,
            32,
            5,
            12,
            2,
            42,
            &[10, 20],
            &[RerankDepthSpec::None],
            true,
            &sq8_spec,
        ),
        "unknown commit must never resume"
    );

    let injected_dirty = fs::read_to_string(&sq8_report)
        .expect("read sq8 report")
        .replace("\"git_dirty\": false", "\"git_dirty\": true");
    let dirty_report_path = tmp.path().join("dirty_arm.json");
    fs::write(&dirty_report_path, injected_dirty).expect("write dirty report copy");
    assert!(
        !can_resume_existing_arm_report(
            &dirty_report_path,
            &prov.git_commit,
            false,
            "synthetic_clustered",
            120,
            32,
            5,
            12,
            2,
            42,
            &[10, 20],
            &[RerankDepthSpec::None],
            true,
            &sq8_spec,
        ),
        "Dirty report must be rejected when current working tree is clean"
    );
}

#[test]
fn test_phase6_streaming_fvecs_qbuild_and_gt_cache() {
    let tmp = TempDir::new("phase6_stream");
    let data_dir = tmp.path().join("data");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&data_dir).expect("mkdir data");

    let (base, query) = generate_synthetic_sift(80, 10, 32, 4, 42).expect("generate synthetic");
    let base_path = data_dir.join("base.fvecs");
    let query_path = data_dir.join("query.fvecs");
    write_fvecs(&base_path, &base).expect("write base.fvecs");
    write_fvecs(&query_path, &query).expect("write query.fvecs");

    let res = run_phase6(&[
        "--matrix",
        "--base-fvecs",
        base_path.to_str().unwrap(),
        "--query-fvecs",
        query_path.to_str().unwrap(),
        "--k",
        "5",
        "--reps",
        "1",
        "--ef-search-list",
        "12",
        "--arms",
        "sq8_only_qbuild,pq16_only_qbuild",
        "--data-dir",
        data_dir.to_str().unwrap(),
        "--output-dir",
        out_dir.to_str().unwrap(),
        "--no-resume",
    ]);
    assert!(
        res.status.success(),
        "Streaming file-backed run failed: {}",
        String::from_utf8_lossy(&res.stderr)
    );

    for arm in ["sq8_only_qbuild", "pq16_only_qbuild"] {
        let doc = read_json_file(&out_dir.join(format!("arm_n80_{arm}.json")));
        assert!(
            doc.get("streaming_build").as_bool(),
            "{arm} should use streaming .fvecs build when file-backed"
        );
        assert!(doc.get("query_rss_profiled_isolated").as_bool());
        assert!(doc.get("query_serving_peak_rss_mb").as_f64() > 0.0);
    }
}

// ============================================================================
// Direct Isolated Query Profiler & Parity Check CLI Tests
// ============================================================================

#[test]
fn test_phase6_internal_profile_query_subprocess() {
    let tmp = TempDir::new("phase6_iso_query");
    let (base, _query) = generate_synthetic_sift(64, 8, 32, 4, 7).expect("generate synthetic");
    let hnsw_cfg = HnswConfig {
        m: 8,
        ef_construction: 32,
        seed: 7,
        ..HnswConfig::default()
    };

    let sq8 = HnswSq8Index::build(
        &base,
        Metric::L2Squared,
        hnsw_cfg.clone(),
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .expect("build sq8 snapshot index");
    let sq8_snap = tmp.path().join("sq8.bin");
    sq8.write_snapshot(&sq8_snap).expect("write sq8 snapshot");

    let sq8_out = run_phase6(&[
        "--internal-profile-query",
        sq8_snap.to_str().unwrap(),
        "--kind",
        "sq8",
        "--synthetic",
        "--queries",
        "8",
        "--k",
        "4",
        "--ef",
        "16",
    ]);
    assert!(
        sq8_out.status.success(),
        "sq8 internal profile failed: {}",
        String::from_utf8_lossy(&sq8_out.stderr)
    );
    let sq8_stdout = String::from_utf8_lossy(&sq8_out.stdout);
    assert!(sq8_stdout.contains("PHASE6_QUERY_PROFILE_BEGIN"));
    assert!(sq8_stdout.contains("query_serving_peak_rss_mb="));
    assert!(sq8_stdout.contains("PHASE6_QUERY_PROFILE_END"));

    let pq_cfg = PqConfig {
        m_pq: 8,
        k_centroids: 256,
        max_train_vectors: 64,
        max_lloyd_iters: 10,
        seed: 7,
    };
    let pq = HnswPqIndex::build(
        &base,
        Metric::L2Squared,
        hnsw_cfg,
        pq_cfg,
        QuantizedBuildMode::QuantizedDirect,
        QuantizedStorageMode::QuantizedOnly,
    )
    .expect("build pq snapshot index");
    let pq_snap = tmp.path().join("pq.bin");
    pq.write_snapshot(&pq_snap).expect("write pq snapshot");

    let pq_out = run_phase6(&[
        "--internal-profile-query",
        pq_snap.to_str().unwrap(),
        "--kind",
        "pq",
        "--synthetic",
        "--queries",
        "8",
        "--k",
        "4",
        "--ef",
        "16",
    ]);
    assert!(
        pq_out.status.success(),
        "pq internal profile failed: {}",
        String::from_utf8_lossy(&pq_out.stderr)
    );
    let pq_stdout = String::from_utf8_lossy(&pq_out.stdout);
    assert!(pq_stdout.contains("PHASE6_QUERY_PROFILE_BEGIN"));
    assert!(pq_stdout.contains("query_serving_peak_rss_mb="));
    assert!(pq_stdout.contains("PHASE6_QUERY_PROFILE_END"));
}

#[test]
fn test_phase6_parity_check_cli_json() {
    let tmp = TempDir::new("phase6_parity_cli");
    let (base, query) = generate_synthetic_sift(64, 5, 32, 4, 19).expect("generate synthetic");
    let base_path = tmp.path().join("base.fvecs");
    let query_path = tmp.path().join("query.fvecs");
    let json_path = tmp.path().join("parity.json");
    write_fvecs(&base_path, &base).expect("write base");
    write_fvecs(&query_path, &query).expect("write query");

    let out = run_phase6(&[
        "--parity-check",
        "--base-fvecs",
        base_path.to_str().unwrap(),
        "--query-fvecs",
        query_path.to_str().unwrap(),
        "--m-pq",
        "8",
        "--k",
        "5",
        "--ef",
        "24",
        "--seed",
        "19",
        "--json-out",
        json_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "--parity-check failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let doc = read_json_file(&json_path);
    assert_eq!(doc.get("command").as_str(), "parity_check");
    assert_eq!(doc.get("num_base").as_usize(), 64);
    assert_eq!(doc.get("num_queries").as_usize(), 5);
    assert_eq!(doc.get("dim").as_usize(), 32);
    assert_eq!(doc.get("k").as_usize(), 5);
    assert_eq!(doc.get("m_pq").as_usize(), 8);
    assert_eq!(doc.get("exact_gt_topk_ids").as_array().len(), 5);

    let sq8 = doc.get("sq8");
    assert_eq!(sq8.get("mins").as_array().len(), 32);
    assert_eq!(sq8.get("scales").as_array().len(), 32);
    assert_eq!(sq8.get("inv_scales").as_array().len(), 32);
    assert_eq!(sq8.get("codes").as_array().len(), 64);
    assert_eq!(sq8.get("asymmetric_l2_sq").as_array().len(), 5);
    assert_eq!(sq8.get("exhaustive_topk_ids").as_array().len(), 5);
    assert_eq!(sq8.get("hnsw_only_topk_ids").as_array().len(), 5);
    assert_eq!(sq8.get("hnsw_rerank_topk_ids").as_array().len(), 5);

    let pq = doc.get("pq");
    assert_eq!(pq.get("m_pq").as_usize(), 8);
    assert_eq!(pq.get("k_centroids").as_usize(), 256);
    assert_eq!(pq.get("d_sub").as_usize(), 4);
    assert_eq!(pq.get("centroids").as_array().len(), 8);
    assert_eq!(pq.get("codes").as_array().len(), 64);
    assert_eq!(pq.get("adc_distances").as_array().len(), 5);
    assert_eq!(pq.get("exhaustive_topk_ids").as_array().len(), 5);
    assert_eq!(pq.get("hnsw_only_topk_ids").as_array().len(), 5);
    assert_eq!(pq.get("hnsw_rerank_topk_ids").as_array().len(), 5);
}

#[test]
fn test_phase6_row_matrix_import_sanity() {
    // Ensures RowMatrix remains directly usable for callers constructing synthetic parity inputs.
    let m = RowMatrix::new(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("RowMatrix");
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 4);
}
