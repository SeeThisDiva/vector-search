//! Integration & CLI Tests for Phase 7 Rust Benchmark Binary (`src/bin/phase7.rs`).
//!
//! Verifies:
//! 1. Single-thread Mode 1 (`search_batch_parallel`) + Mode 2 (`search_batch_timed_with_tracker`)
//!    execution on `--synthetic`, exact recall parity, `ffi_delta_us == 0.0`,
//!    `instant_overhead_us`, and canonical JSON schema matching `scripts/plot_phase7.py`.
//! 2. `--no-keep-pruned-shrink` behavior (`keep_pruned_shrink = false`, distinct
//!    `graph_fingerprint`, and `-nkp` / `C2-nkp` `config_id` naming) plus `--config-id` override.
//! 3. Multi-thread Mode 1 (`--compare-mode multithread --threads 4`) emitting `mode2_per_query: null`
//!    and matching single-thread recall.
//! 4. `--prepare-100k-gt` and `--gt-path` `.ivecs` caching roundtrip on small synthetic/custom files.
//! 5. Strict failure (no silent fallback to synthetic) when SIFT files are missing and `--synthetic`
//!    is absent, plus invalid CLI flag rejection.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use vector_search::dataset::{generate_synthetic_sift, read_ivecs, write_fvecs};
use vector_search::distance::Metric;
use vector_search::exact_knn::ExactKnn;
use vector_search::recall::verify_self_recall;

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "vecsearch_p7_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
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
            Self::String(s) => s,
            other => panic!("Expected JSON string, got {other:?}"),
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            other => panic!("Expected JSON bool, got {other:?}"),
        }
    }

    fn as_f64(&self) -> f64 {
        match self {
            Self::Number(n) => *n,
            other => panic!("Expected JSON number, got {other:?}"),
        }
    }

    fn as_usize(&self) -> usize {
        let n = self.as_f64();
        assert!(
            n >= 0.0 && (n - n.round()).abs() < 1e-9,
            "Expected non-negative integer JSON number, got {n}"
        );
        n.round() as usize
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
    fn parse(input: &'a str) -> JsonValue {
        let mut p = Self {
            bytes: input.as_bytes(),
            pos: 0,
        };
        let val = p.parse_value();
        p.skip_ws();
        assert_eq!(
            p.pos,
            p.bytes.len(),
            "Trailing characters in JSON at byte {}",
            p.pos
        );
        val
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> u8 {
        assert!(self.pos < self.bytes.len(), "Unexpected EOF in JSON");
        self.bytes[self.pos]
    }

    fn expect_bytes(&mut self, expected: &[u8]) {
        assert!(
            self.pos + expected.len() <= self.bytes.len()
                && &self.bytes[self.pos..self.pos + expected.len()] == expected,
            "Expected {:?} at pos {}",
            String::from_utf8_lossy(expected),
            self.pos
        );
        self.pos += expected.len();
    }

    fn parse_value(&mut self) -> JsonValue {
        self.skip_ws();
        match self.peek() {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => JsonValue::String(self.parse_string()),
            b't' => {
                self.expect_bytes(b"true");
                JsonValue::Bool(true)
            }
            b'f' => {
                self.expect_bytes(b"false");
                JsonValue::Bool(false)
            }
            b'n' => {
                self.expect_bytes(b"null");
                JsonValue::Null
            }
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => panic!("Unexpected byte {} at pos {}", other as char, self.pos),
        }
    }

    fn parse_object(&mut self) -> JsonValue {
        self.expect_bytes(b"{");
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == b'}' {
            self.pos += 1;
            return JsonValue::Object(map);
        }
        loop {
            self.skip_ws();
            let key = self.parse_string();
            self.skip_ws();
            self.expect_bytes(b":");
            let val = self.parse_value();
            assert!(
                map.insert(key.clone(), val).is_none(),
                "Duplicate JSON key '{key}'"
            );
            self.skip_ws();
            match self.peek() {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    break;
                }
                other => panic!("Expected ',' or '}}' in object, got {}", other as char),
            }
        }
        JsonValue::Object(map)
    }

    fn parse_array(&mut self) -> JsonValue {
        self.expect_bytes(b"[");
        self.skip_ws();
        let mut arr = Vec::new();
        if self.peek() == b']' {
            self.pos += 1;
            return JsonValue::Array(arr);
        }
        loop {
            arr.push(self.parse_value());
            self.skip_ws();
            match self.peek() {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    break;
                }
                other => panic!("Expected ',' or ']' in array, got {}", other as char),
            }
        }
        JsonValue::Array(arr)
    }

    fn parse_string(&mut self) -> String {
        self.expect_bytes(b"\"");
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return out,
                b'\\' => {
                    assert!(
                        self.pos < self.bytes.len(),
                        "Dangling escape in JSON string"
                    );
                    let esc = self.bytes[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'u' => {
                            assert!(self.pos + 4 <= self.bytes.len(), "Truncated \\uXXXX escape");
                            let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                .expect("valid utf8 hex");
                            let code = u32::from_str_radix(hex, 16).expect("valid hex escape");
                            self.pos += 4;
                            out.push(char::from_u32(code).expect("valid unicode scalar"));
                        }
                        other => panic!("Unknown JSON escape \\{}", other as char),
                    }
                }
                _ => {
                    self.pos -= 1;
                    let rem = std::str::from_utf8(&self.bytes[self.pos..]).expect("valid utf8");
                    let ch = rem.chars().next().expect("non-empty char");
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        panic!("Unterminated JSON string");
    }

    fn parse_number(&mut self) -> JsonValue {
        let start = self.pos;
        if self.bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_digit()
                || matches!(self.bytes[self.pos], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).expect("valid utf8 number");
        let n = s
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("Invalid JSON number '{s}': {e}"));
        assert!(n.is_finite(), "Non-finite JSON number '{s}'");
        JsonValue::Number(n)
    }
}

fn phase7_bin() -> &'static str {
    env!("CARGO_BIN_EXE_phase7")
}

#[test]
fn test_phase7_cli_single_mode_and_no_keep_pruned_shrink_synthetic() {
    let tmp = TempTestDir::new("single_nkp");
    let single_json_path = tmp.0.join("rust_single.json");
    let nkp_json_path = tmp.0.join("rust_nkp.json");

    let out_single = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "350",
            "--queries",
            "35",
            "--dim",
            "32",
            "--m",
            "16",
            "--ef-construction",
            "40",
            "--ef-search-list",
            "10,25,60",
            "--k",
            "10",
            "--build-reps",
            "2",
            "--query-reps",
            "3",
            "--compare-mode",
            "single",
            "--json-out",
        ])
        .arg(&single_json_path)
        .output()
        .expect("execute phase7 single");
    assert!(
        out_single.status.success(),
        "phase7 single failed: stderr={}",
        String::from_utf8_lossy(&out_single.stderr)
    );

    let out_nkp = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "350",
            "--queries",
            "35",
            "--dim",
            "32",
            "--m",
            "16",
            "--ef-construction",
            "40",
            "--ef-search-list",
            "10,25,60",
            "--k",
            "10",
            "--build-reps",
            "1",
            "--query-reps",
            "2",
            "--compare-mode",
            "single",
            "--no-keep-pruned-shrink",
            "--json-out",
        ])
        .arg(&nkp_json_path)
        .output()
        .expect("execute phase7 single nkp");
    assert!(
        out_nkp.status.success(),
        "phase7 nkp failed: stderr={}",
        String::from_utf8_lossy(&out_nkp.stderr)
    );

    let single = JsonParser::parse(&fs::read_to_string(&single_json_path).unwrap());
    let nkp = JsonParser::parse(&fs::read_to_string(&nkp_json_path).unwrap());

    assert_eq!(single.get("schema_version").as_usize(), 1);
    assert_eq!(single.get("phase").as_usize(), 7);
    assert_eq!(single.get("engine").as_str(), "rust");
    assert_eq!(single.get("compare_mode").as_str(), "single");
    assert_eq!(single.get("config_id").as_str(), "M16_efC40");
    assert_eq!(single.get("threads").as_usize(), 1);
    assert!(single.get("warmup_performed").as_bool());
    assert_eq!(single.get("build_reps").as_usize(), 2);
    assert_eq!(single.get("query_reps").as_usize(), 3);
    assert_eq!(single.get("scale_vectors").as_usize(), 350);
    assert!(!single.get("git_commit").as_str().is_empty());
    assert!(!single.get("rustc_version").as_str().is_empty());
    assert!(!single.get("generated_at").as_str().is_empty());
    assert_eq!(*single.get("ground_truth_path"), JsonValue::Null);

    let env_obj = single.get("environment");
    assert!(!env_obj.get("os").as_str().is_empty());
    assert!(!env_obj.get("arch").as_str().is_empty());
    assert!(!env_obj.get("cpu_model").as_str().is_empty());
    assert!(env_obj.get("logical_cores").as_usize() >= 1);
    assert!(!env_obj.get("power_mode").as_str().is_empty());
    assert!(!env_obj.get("processor_affinity_mask").as_str().is_empty());
    assert_eq!(env_obj.get("threads").as_usize(), 1);

    let hw_obj = single.get("hardware");
    assert!(!hw_obj.get("summary").as_str().is_empty());
    assert_eq!(
        hw_obj.get("cpu_model").as_str(),
        env_obj.get("cpu_model").as_str()
    );

    let ds_obj = single.get("dataset");
    assert_eq!(ds_obj.get("preset").as_str(), "synthetic");
    assert_eq!(ds_obj.get("source_kind").as_str(), "synthetic");
    assert!(!ds_obj.get("is_authentic_sift").as_bool());
    assert_eq!(ds_obj.get("num_base").as_usize(), 350);
    assert_eq!(ds_obj.get("num_query").as_usize(), 35);
    assert_eq!(ds_obj.get("dim").as_usize(), 32);
    assert_eq!(ds_obj.get("k").as_usize(), 10);
    assert_eq!(ds_obj.get("metric").as_str(), "l2");

    let cfg_single = single.get("hnsw_config");
    assert_eq!(cfg_single.get("m").as_usize(), 16);
    assert_eq!(cfg_single.get("m_max").as_usize(), 16);
    assert_eq!(cfg_single.get("m_max0").as_usize(), 32);
    assert_eq!(cfg_single.get("ef_construction").as_usize(), 40);
    assert!(cfg_single.get("keep_pruned").as_bool());
    assert_eq!(*cfg_single.get("keep_pruned_shrink"), JsonValue::Null);

    let cfg_nkp = nkp.get("hnsw_config");
    assert!(!cfg_nkp.get("keep_pruned_shrink").as_bool());
    assert_eq!(nkp.get("config_id").as_str(), "M16_efC40-nkp");

    let ib_single = single.get("index_build");
    let ib_nkp = nkp.get("index_build");
    assert_eq!(ib_single.get("num_vectors").as_usize(), 350);
    assert_eq!(ib_single.get("build_times_s").as_array().len(), 2);
    assert!(ib_single.get("build_time_s").as_f64() > 0.0);
    assert!(ib_single.get("build_qps").as_f64() > 0.0);
    assert!(
        ib_single.get("build_time_min_s").as_f64() <= ib_single.get("build_time_s").as_f64()
            && ib_single.get("build_time_s").as_f64() <= ib_single.get("build_time_max_s").as_f64()
    );
    assert_ne!(
        ib_single.get("graph_fingerprint_hex").as_str(),
        ib_nkp.get("graph_fingerprint_hex").as_str(),
        "--no-keep-pruned-shrink must alter graph topology fingerprint"
    );

    let mem = single.get("memory_profile");
    assert_eq!(
        mem.get("peak_rss_scope").as_str(),
        "process_lifetime_high_water_mark"
    );
    assert!(mem.get("graph_only_bytes_per_vector").as_f64() > 0.0);
    assert!(mem.get("graph_only_memory_bytes").as_usize() > 0);
    assert!((mem.get("raw_vector_bytes_per_vector").as_f64() - 128.0).abs() < 1e-6);
    assert!(
        (mem.get("total_with_vectors_bytes_per_vector").as_f64()
            - (mem.get("graph_only_bytes_per_vector").as_f64() + 128.0))
            .abs()
            < 0.05
    );

    let sweep = single.get("ef_search_sweep").as_array();
    assert_eq!(sweep.len(), 3);
    for (idx, expected_ef) in [10_usize, 25, 60].iter().enumerate() {
        let pt = &sweep[idx];
        assert_eq!(pt.get("ef_search").as_usize(), *expected_ef);
        let r1 = pt.get("recall_at_k").as_f64();
        let r2 = pt.get("recall_at_k_mode2").as_f64();
        assert!(
            (r1 - r2).abs() < 1e-9,
            "Mode 1 and Mode 2 recall must match"
        );
        assert!((0.0..=1.0).contains(&r1));

        let m1 = pt.get("mode1_batch");
        assert_eq!(m1.get("num_queries").as_usize(), 35);
        assert!(m1.get("total_s").as_f64() > 0.0);
        assert!(m1.get("qps").as_f64() > 0.0);
        assert!(m1.get("mean_ms").as_f64() > 0.0);
        assert_eq!(m1.get("run_total_s").as_array().len(), 3);
        assert_eq!(m1.get("run_mean_ms").as_array().len(), 3);
        assert_eq!(m1.get("run_qps").as_array().len(), 3);

        let m2 = pt.get("mode2_per_query");
        assert_eq!(m2.get("num_queries").as_usize(), 35);
        assert!(m2.get("min_ms").as_f64() <= m2.get("mean_ms").as_f64());
        assert!(m2.get("mean_ms").as_f64() <= m2.get("max_ms").as_f64());
        assert!(m2.get("p50_ms").as_f64() <= m2.get("p95_ms").as_f64());
        assert!(m2.get("p95_ms").as_f64() <= m2.get("p99_ms").as_f64());
        assert_eq!(m2.get("run_total_s").as_array().len(), 3);
        assert_eq!(m2.get("run_mean_ms").as_array().len(), 3);
        assert_eq!(m2.get("run_p99_ms").as_array().len(), 3);

        assert!((pt.get("ffi_delta_us").as_f64()).abs() < 1e-9);
        let expected_overhead = (m2.get("mean_ms").as_f64() - m1.get("mean_ms").as_f64()) * 1000.0;
        assert!(
            (pt.get("instant_overhead_us").as_f64() - expected_overhead).abs() < 0.25,
            "instant_overhead_us mismatch: got {}, expected {}",
            pt.get("instant_overhead_us").as_f64(),
            expected_overhead
        );
        assert!((pt.get("mode1_qps").as_f64() - m1.get("qps").as_f64()).abs() < 1e-6);
        assert!((pt.get("p99_ms").as_f64() - m2.get("p99_ms").as_f64()).abs() < 1e-6);
    }
}

#[test]
fn test_phase7_cli_multithread_mode_and_config_id_override() {
    let tmp = TempTestDir::new("multithread");
    let single_c2_path = tmp.0.join("c2_single.json");
    let mt_c2_path = tmp.0.join("c2_mt4.json");
    let custom_cid_path = tmp.0.join("custom_cid.json");

    let out_single = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "350",
            "--queries",
            "35",
            "--dim",
            "32",
            "--m",
            "16",
            "--ef-construction",
            "100",
            "--ef-search-list",
            "10,30",
            "--k",
            "10",
            "--build-reps",
            "1",
            "--query-reps",
            "2",
            "--compare-mode",
            "single",
            "--json-out",
        ])
        .arg(&single_c2_path)
        .output()
        .expect("execute single C2");
    assert!(out_single.status.success());

    let out_mt = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "350",
            "--queries",
            "35",
            "--dim",
            "32",
            "--m",
            "16",
            "--ef-construction",
            "100",
            "--ef-search-list",
            "10,30",
            "--k",
            "10",
            "--build-reps",
            "1",
            "--query-reps",
            "2",
            "--compare-mode",
            "multithread",
            "--threads",
            "4",
            "--json-out",
        ])
        .arg(&mt_c2_path)
        .output()
        .expect("execute multithread C2");
    assert!(
        out_mt.status.success(),
        "phase7 multithread failed: stderr={}",
        String::from_utf8_lossy(&out_mt.stderr)
    );

    let single_c2 = JsonParser::parse(&fs::read_to_string(&single_c2_path).unwrap());
    let mt_c2 = JsonParser::parse(&fs::read_to_string(&mt_c2_path).unwrap());

    assert_eq!(single_c2.get("config_id").as_str(), "C2");
    assert_eq!(mt_c2.get("config_id").as_str(), "C2");
    assert_eq!(mt_c2.get("compare_mode").as_str(), "multithread");
    assert_eq!(mt_c2.get("threads").as_usize(), 4);
    assert_eq!(mt_c2.get("environment").get("threads").as_usize(), 4);
    assert_eq!(
        single_c2.get("index_build").get("graph_fingerprint_hex"),
        mt_c2.get("index_build").get("graph_fingerprint_hex")
    );

    let s_sweep = single_c2.get("ef_search_sweep").as_array();
    let m_sweep = mt_c2.get("ef_search_sweep").as_array();
    assert_eq!(m_sweep.len(), 2);
    for idx in 0..2 {
        let pt = &m_sweep[idx];
        assert_eq!(*pt.get("mode2_per_query"), JsonValue::Null);
        assert_eq!(*pt.get("recall_at_k_mode2"), JsonValue::Null);
        assert_eq!(*pt.get("ffi_delta_us"), JsonValue::Null);
        assert_eq!(*pt.get("instant_overhead_us"), JsonValue::Null);
        assert_eq!(*pt.get("p50_ms"), JsonValue::Null);
        assert_eq!(*pt.get("p95_ms"), JsonValue::Null);
        assert_eq!(*pt.get("p99_ms"), JsonValue::Null);
        let m1 = pt.get("mode1_batch");
        assert!(m1.get("qps").as_f64() > 0.0);
        assert_eq!(m1.get("run_qps").as_array().len(), 2);
        assert!((pt.get("mean_ms").as_f64() - m1.get("mean_ms").as_f64()).abs() < 1e-6);
        assert!(
            (pt.get("recall_at_k").as_f64() - s_sweep[idx].get("recall_at_k").as_f64()).abs()
                < 1e-9,
            "Multithread search_batch_parallel recall must match single-thread"
        );
    }

    // Verify explicit --config-id override with --no-keep-pruned-shrink
    let out_cid = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "200",
            "--queries",
            "20",
            "--dim",
            "16",
            "--m",
            "16",
            "--ef-construction",
            "100",
            "--ef-search-list",
            "10",
            "--build-reps",
            "1",
            "--query-reps",
            "1",
            "--no-keep-pruned-shrink",
            "--config-id",
            "C2-nkp",
            "--json-out",
        ])
        .arg(&custom_cid_path)
        .output()
        .expect("execute custom config-id");
    assert!(out_cid.status.success());
    let custom_cid = JsonParser::parse(&fs::read_to_string(&custom_cid_path).unwrap());
    assert_eq!(custom_cid.get("config_id").as_str(), "C2-nkp");
    assert!(!custom_cid
        .get("hnsw_config")
        .get("keep_pruned_shrink")
        .as_bool());
}

#[test]
fn test_phase7_prepare_gt_and_ivecs_roundtrip() {
    let tmp = TempTestDir::new("gt_roundtrip");
    let data_dir = tmp.0.join("data");
    let base_path = tmp.0.join("base.fvecs");
    let query_path = tmp.0.join("query.fvecs");

    let (base, query) = generate_synthetic_sift(300, 30, 32, 6, 42).unwrap();
    write_fvecs(&base_path, &base).unwrap();
    write_fvecs(&query_path, &query).unwrap();

    // 1. Prepare cached exact GT file via --prepare-100k-gt on custom .fvecs
    let out_prep = Command::new(phase7_bin())
        .arg("--base-fvecs")
        .arg(&base_path)
        .arg("--query-fvecs")
        .arg(&query_path)
        .arg("--data-dir")
        .arg(&data_dir)
        .arg("--prepare-100k-gt")
        .output()
        .expect("execute prepare-100k-gt");
    assert!(
        out_prep.status.success(),
        "prepare-100k-gt failed: stderr={}",
        String::from_utf8_lossy(&out_prep.stderr)
    );

    let gt_file = data_dir.join("gt_300_k100.ivecs");
    assert!(
        gt_file.is_file(),
        "Expected prepared GT file at {}",
        gt_file.display()
    );
    let gt_loaded = read_ivecs(&gt_file).unwrap();
    assert_eq!(gt_loaded.shape(), (30, 100));
    verify_self_recall(&gt_loaded, 10).unwrap();

    let expected_exact =
        ExactKnn::search_parallel_with_base(Metric::L2Squared, &base, &query, 100, 0).unwrap();
    assert_eq!(gt_loaded.as_slice(), expected_exact.indices.as_slice());

    // 2. Roundtrip using --gt-path and compare recall against in-memory exact GT
    let json_with_gt = tmp.0.join("with_gt.json");
    let json_mem_gt = tmp.0.join("mem_gt.json");

    let out_with_gt = Command::new(phase7_bin())
        .arg("--base-fvecs")
        .arg(&base_path)
        .arg("--query-fvecs")
        .arg(&query_path)
        .arg("--gt-path")
        .arg(&gt_file)
        .args([
            "--m",
            "8",
            "--ef-construction",
            "40",
            "--ef-search-list",
            "10,40",
            "--build-reps",
            "1",
            "--query-reps",
            "2",
            "--json-out",
        ])
        .arg(&json_with_gt)
        .output()
        .expect("execute phase7 with gt-path");
    assert!(
        out_with_gt.status.success(),
        "phase7 with --gt-path failed: stderr={}",
        String::from_utf8_lossy(&out_with_gt.stderr)
    );

    let out_mem_gt = Command::new(phase7_bin())
        .arg("--base-fvecs")
        .arg(&base_path)
        .arg("--query-fvecs")
        .arg(&query_path)
        .args([
            "--m",
            "8",
            "--ef-construction",
            "40",
            "--ef-search-list",
            "10,40",
            "--build-reps",
            "1",
            "--query-reps",
            "2",
            "--json-out",
        ])
        .arg(&json_mem_gt)
        .output()
        .expect("execute phase7 without gt-path");
    assert!(out_mem_gt.status.success());

    let parsed_gt = JsonParser::parse(&fs::read_to_string(&json_with_gt).unwrap());
    let parsed_mem = JsonParser::parse(&fs::read_to_string(&json_mem_gt).unwrap());
    assert!(parsed_gt
        .get("ground_truth_source")
        .as_str()
        .starts_with("custom_ivecs"));
    assert_ne!(*parsed_gt.get("ground_truth_path"), JsonValue::Null);
    assert_eq!(*parsed_mem.get("ground_truth_path"), JsonValue::Null);
    for idx in 0..2 {
        assert!(
            (parsed_gt.get("ef_search_sweep").as_array()[idx]
                .get("recall_at_k")
                .as_f64()
                - parsed_mem.get("ef_search_sweep").as_array()[idx]
                    .get("recall_at_k")
                    .as_f64())
            .abs()
                < 1e-12
        );
    }

    // 3. --prepare-100k-gt with explicit --gt-path on --synthetic
    let explicit_gt = tmp.0.join("explicit_gt.ivecs");
    let out_syn_prep = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "180",
            "--queries",
            "15",
            "--dim",
            "16",
            "--prepare-100k-gt",
            "--gt-path",
        ])
        .arg(&explicit_gt)
        .output()
        .expect("execute synthetic prepare-100k-gt");
    assert!(out_syn_prep.status.success());
    let syn_gt = read_ivecs(&explicit_gt).unwrap();
    assert_eq!(syn_gt.shape(), (15, 100));
}

#[test]
fn test_phase7_no_silent_synthetic_fallback_and_cli_validation() {
    let tmp = TempTestDir::new("validation");
    let empty_data_dir = tmp.0.join("empty_data");
    fs::create_dir_all(&empty_data_dir).unwrap();

    // Missing SIFT1M + --offline without --synthetic MUST fail loudly (never silent fallback)
    let out_missing = Command::new(phase7_bin())
        .arg("--dataset")
        .arg("sift1m")
        .arg("--data-dir")
        .arg(&empty_data_dir)
        .args(["--offline", "--scale", "100", "--queries", "10"])
        .output()
        .expect("execute offline missing sift");
    assert!(
        !out_missing.status.success(),
        "phase7 must NOT silently fall back to synthetic when --synthetic is absent"
    );

    // --compare-mode single with --threads 4 must fail
    let out_bad_threads = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "100",
            "--queries",
            "10",
            "--compare-mode",
            "single",
            "--threads",
            "4",
        ])
        .output()
        .expect("execute invalid single threads");
    assert!(!out_bad_threads.status.success());

    // Invalid HNSW config (ef_construction < m) must fail
    let out_bad_efc = Command::new(phase7_bin())
        .args([
            "--synthetic",
            "--scale",
            "100",
            "--queries",
            "10",
            "--m",
            "16",
            "--ef-construction",
            "8",
        ])
        .output()
        .expect("execute invalid efC");
    assert!(!out_bad_efc.status.success());
}
