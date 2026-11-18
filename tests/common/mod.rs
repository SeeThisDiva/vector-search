//! Shared helpers for integration tests.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use vector_search::dataset::{load_or_download_sift, DatasetSource, LoadedDataset, SiftPreset};
use vector_search::distance::Metric;
use vector_search::hnsw::{HnswConfig, HnswIndex};
use vector_search::matrix::RowMatrix;
use vector_search::rng::DeterministicRng;

static TEMP_SEQ: AtomicUsize = AtomicUsize::new(0);

/// RAII temporary directory that is automatically deleted on drop.
pub struct TempTestDir {
    path: PathBuf,
}

impl TempTestDir {
    pub fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "vs_test_{prefix}_{}_{nanos}_{seq}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp test dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Loads the authentic `siftsmall` corpus from `data/`, or returns `None` (so the caller can
/// skip) when the gitignored `.fvecs` files are not present, e.g. on a fresh clone or in CI.
///
/// The loader silently falls back to synthetic vectors when the cache is missing; tests that
/// assert SIFT-specific shapes or recall must not run against that fallback.
pub fn load_siftsmall_or_skip(test_name: &str) -> Option<LoadedDataset> {
    let loaded = load_or_download_sift(SiftPreset::SiftSmall, Path::new("data"), false, 42).ok()?;
    if matches!(loaded.source, DatasetSource::AuthenticSiftSmall { .. }) {
        Some(loaded)
    } else {
        eprintln!("SKIP {test_name}: data/siftsmall/*.fvecs not found (see README to download)");
        None
    }
}

/// Builds a deterministic 2D synthetic matrix of `n` vectors.
pub fn make_synthetic_2d_matrix(n: usize) -> RowMatrix<f32> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f32) * 1.5;
        let y = ((i * 7) % 31) as f32;
        rows.push(vec![x, y]);
    }
    RowMatrix::from_rows(&rows).unwrap()
}

/// Builds a deterministic HNSW index over a 2D synthetic matrix.
pub fn make_synthetic_2d_index(matrix: &RowMatrix<f32>) -> HnswIndex<'_> {
    let config = HnswConfig {
        m: 16,
        ef_construction: 64,
        seed: 42,
        ..Default::default()
    };
    HnswIndex::build(matrix, Metric::L2Squared, config).unwrap()
}

/// Generates a deterministic random `f32` matrix in `[lo, hi)`.
pub fn make_random_matrix(rows: usize, cols: usize, lo: f64, hi: f64, seed: u64) -> RowMatrix<f32> {
    let mut rng = DeterministicRng::new(seed);
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        let u = rng.next_f64_open01();
        data.push((lo + u * (hi - lo)) as f32);
    }
    RowMatrix::new(rows, cols, data).unwrap()
}

/// Strict JSON AST node for CLI and report verification in integration tests.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_object(&self) -> &BTreeMap<String, JsonValue> {
        match self {
            Self::Object(map) => map,
            other => panic!("Expected JSON object, got {other:?}"),
        }
    }

    pub fn as_array(&self) -> &[JsonValue] {
        match self {
            Self::Array(arr) => arr,
            other => panic!("Expected JSON array, got {other:?}"),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Str(s) => s,
            other => panic!("Expected JSON string, got {other:?}"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            other => panic!("Expected JSON bool, got {other:?}"),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Number(n) => *n,
            other => panic!("Expected JSON number, got {other:?}"),
        }
    }

    pub fn as_u64(&self) -> u64 {
        let n = self.as_f64();
        assert!(
            n >= 0.0 && (n - n.round()).abs() < 1e-9,
            "Expected non-negative integer JSON number, got {n}"
        );
        n.round() as u64
    }

    pub fn as_usize(&self) -> usize {
        self.as_u64() as usize
    }

    pub fn get(&self, key: &str) -> &JsonValue {
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
            b'"' => JsonValue::Str(self.parse_string()),
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

/// Parses a JSON string strictly (rejecting duplicate object keys and trailing bytes).
pub fn parse_json_strict(raw: &str) -> JsonValue {
    JsonParser::parse(raw)
}

/// Reads and strictly parses a JSON file from disk.
pub fn read_json_file(path: &Path) -> JsonValue {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed reading JSON '{}': {e}", path.display()));
    parse_json_strict(&text)
}
