//! Dependency-free JSON serialization, file writing, and escape-aware extraction utilities.
//!
//! Internal support module shared across benchmark runners and CLI binaries.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use crate::error::Result;

/// Escapes a string for JSON string literal emission.
#[must_use]
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// Formats an optional file path as a JSON string literal or `null`.
#[must_use]
pub fn opt_path_json(path: Option<&Path>) -> String {
    match path {
        Some(p) => format!("\"{}\"", escape_json(&p.display().to_string())),
        None => "null".to_string(),
    }
}

/// Appends a JSON array of `f64` values formatted with `decimals` fractional digits.
pub fn push_f64_array_prec(out: &mut String, values: &[f64], decimals: usize) {
    out.push('[');
    for (idx, &v) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        if v.is_finite() {
            let _ = write!(out, "{v:.decimals$}");
        } else {
            out.push_str("null");
        }
    }
    out.push(']');
}

/// Appends a JSON array of `f64` values formatted with 6 fractional digits.
pub fn push_f64_array(out: &mut String, values: &[f64]) {
    push_f64_array_prec(out, values, 6);
}

/// Formats an optional `f64` as a JSON number with `decimals` fractional digits or `null`.
#[must_use]
pub fn opt_f64_json_prec(v: Option<f64>, decimals: usize) -> String {
    match v {
        Some(val) if val.is_finite() => format!("{val:.decimals$}"),
        _ => "null".to_string(),
    }
}

/// Formats an optional `f64` as a JSON number with 6 fractional digits or `null`.
#[must_use]
pub fn opt_f64_json(v: Option<f64>) -> String {
    opt_f64_json_prec(v, 6)
}

/// Ensures the parent directory of `path` exists when non-empty.
pub fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Creates parent directories if needed and writes `contents` to `path`.
pub fn write_file_with_parents(path: &Path, contents: impl AsRef<[u8]>) -> Result<()> {
    ensure_parent_dir(path)?;
    fs::write(path, contents)?;
    Ok(())
}

/// Parses 4 hex digits from `chars` into a `u16`.
fn parse_hex4(chars: &mut std::str::CharIndices<'_>) -> Option<u16> {
    let mut hex = [0_u8; 4];
    for slot in &mut hex {
        let (_, h) = chars.next()?;
        if !h.is_ascii_hexdigit() {
            return None;
        }
        *slot = h as u8;
    }
    let hex_str = std::str::from_utf8(&hex).ok()?;
    u16::from_str_radix(hex_str, 16).ok()
}

/// Parses a JSON string literal starting immediately after the opening `"` at byte index 0,
/// returning `(decoded_string, byte_index_after_closing_quote)`.
fn parse_json_string_at_quote(after_open_quote: &str) -> Option<(String, usize)> {
    let mut out = String::new();
    let mut chars = after_open_quote.char_indices();
    while let Some((idx, c)) = chars.next() {
        match c {
            '"' => return Some((out, idx + 1)),
            '\\' => {
                let (_, esc) = chars.next()?;
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let code = parse_hex4(&mut chars)?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            let (_, backslash) = chars.next()?;
                            let (_, u_ch) = chars.next()?;
                            if backslash != '\\' || u_ch != 'u' {
                                return None;
                            }
                            let low = parse_hex4(&mut chars)?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return None;
                            }
                            let cp = 0x10000
                                + (((u32::from(code) - 0xD800) << 10) | (u32::from(low) - 0xDC00));
                            out.push(char::from_u32(cp)?);
                        } else if (0xDC00..=0xDFFF).contains(&code) {
                            return None;
                        } else {
                            out.push(char::from_u32(u32::from(code))?);
                        }
                    }
                    _ => return None,
                }
            }
            _ => out.push(c),
        }
    }
    None
}

/// Finds the start of the JSON value for `key` using first-match-anywhere semantics while
/// skipping string literal contents so `"key":` inside escaped strings never triggers false matches.
///
/// Note: Intentionally does not track `{}`/`[]` nesting depth because benchmark callers (e.g.,
/// `golden.rs`) query nested fields such as `"m"` and `"graph_fingerprint"` directly by key name.
fn find_key_value_start<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let bytes = content.as_bytes();
    let mut i = 0_usize;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let inner_start = i + 1;
        let (parsed_key, consumed) = parse_json_string_at_quote(&content[inner_start..])?;
        let mut cursor = inner_start + consumed;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b':' && parsed_key == key {
            return Some(content[cursor + 1..].trim_start());
        }
        i = inner_start + consumed;
    }
    None
}

fn extract_json_scalar_token<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let rest = find_key_value_start(content, key)?;
    let end = rest.find([',', '\n', '\r', '}', ']']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

/// Extracts an escape-aware string field `"key": "..."` from `content`, returning `None`
/// if missing or `null`.
#[must_use]
pub fn extract_json_string_field(content: &str, key: &str) -> Option<String> {
    let rest = find_key_value_start(content, key)?;
    let inner = rest.strip_prefix('"')?;
    parse_json_string_at_quote(inner).map(|(s, _)| s)
}

/// Extracts a finite `f64` field (including negative and scientific notation) from `content`.
#[must_use]
pub fn extract_json_f64_field(content: &str, key: &str) -> Option<f64> {
    let tok = extract_json_scalar_token(content, key)?;
    if tok == "null" {
        return None;
    }
    let first = *tok.as_bytes().first()?;
    if !(first.is_ascii_digit() || first == b'-' || first == b'+') {
        return None;
    }
    tok.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Extracts a `u64` field from `content`, returning `None` if missing, `null`, or non-integer.
#[must_use]
pub fn extract_json_u64_field(content: &str, key: &str) -> Option<u64> {
    let tok = extract_json_scalar_token(content, key)?;
    if tok == "null" {
        return None;
    }
    tok.parse::<u64>().ok()
}

/// Extracts a `usize` field from `content`, returning `None` if missing, `null`, or non-integer.
#[must_use]
pub fn extract_json_usize_field(content: &str, key: &str) -> Option<usize> {
    let tok = extract_json_scalar_token(content, key)?;
    if tok == "null" {
        return None;
    }
    tok.parse::<usize>().ok()
}

/// Extracts a boolean field (`true` / `false`) from `content`.
#[must_use]
pub fn extract_json_bool_field(content: &str, key: &str) -> Option<bool> {
    match extract_json_scalar_token(content, key)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Extracts the inner slice `[...]` for an array field `"key": [...]` using escape-aware
/// bracket depth tracking.
#[must_use]
pub fn extract_json_array_slice<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    let rest = find_key_value_start(content, key)?;
    if !rest.starts_with('[') {
        return None;
    }
    let bytes = rest.as_bytes();
    let mut depth = 0_usize;
    let mut in_str = false;
    let mut escape = false;

    for (idx, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' => depth += 1,
            b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&rest[1..idx]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Splits a JSON array body slice into its top-level `{ ... }` object slices, respecting
/// nested objects and escaped string literals.
#[must_use]
pub fn split_top_level_json_objects(slice: &str) -> Vec<&str> {
    let bytes = slice.as_bytes();
    let mut objs = Vec::new();
    let mut depth = 0_usize;
    let mut obj_start: Option<usize> = None;
    let mut in_str = false;
    let mut escape = false;

    for (idx, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    obj_start = Some(idx);
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start.take() {
                        objs.push(&slice[start..=idx]);
                    }
                }
            }
            _ => {}
        }
    }
    objs
}

/// Parses a comma-separated list of numbers from either a bracketed slice (`"[1, 2, 3]"` or
/// `"\"key\": [1, 2, 3]"`) or an unwrapped comma-separated array body (`"1, 2, 3"`).
#[must_use]
pub fn parse_number_list<T: FromStr>(slice: &str) -> Vec<T> {
    let body = if let Some(start) = slice.find('[') {
        let after = &slice[start + 1..];
        let Some(end) = after.find(']') else {
            return Vec::new();
        };
        &after[..end]
    } else {
        slice
    };
    body.split(',')
        .filter_map(|s| s.trim().parse::<T>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_and_formatting_helpers() {
        assert_eq!(escape_json("a\"b\\c\n\t\x01"), "a\\\"b\\\\c\\n\\t\\u0001");
        assert_eq!(opt_path_json(None), "null");
        assert_eq!(
            opt_path_json(Some(Path::new(r"C:\tmp\a.bin"))),
            "\"C:\\\\tmp\\\\a.bin\""
        );

        let mut arr = String::new();
        push_f64_array(&mut arr, &[1.5, -2.25]);
        assert_eq!(arr, "[1.500000, -2.250000]");

        let mut arr2 = String::new();
        push_f64_array_prec(&mut arr2, &[1.5, f64::NAN], 2);
        assert_eq!(arr2, "[1.50, null]");

        assert_eq!(opt_f64_json(Some(0.25)), "0.250000");
        assert_eq!(opt_f64_json(None), "null");
        assert_eq!(opt_f64_json(Some(f64::INFINITY)), "null");
        assert_eq!(opt_f64_json_prec(Some(12.345), 2), "12.35");
    }

    #[test]
    fn test_escape_aware_extractors() {
        let json = r#"{
  "desc": "escaped \"k\": 999 and path C:\\dir\\\"next\"",
  "utf8": "café µs",
  "k": 10,
  "neg_float": -1.25e-4,
  "pos_exp": 3.5E+2,
  "null_num": null,
  "flag": true,
  "empty_flag": false,
  "arr": [
    {"id": 1, "label": "a[0] }"},
    {"id": 2, "label": "b\\"}
  ]
}"#;
        assert_eq!(
            extract_json_string_field(json, "desc").as_deref(),
            Some("escaped \"k\": 999 and path C:\\dir\\\"next\"")
        );
        assert_eq!(
            extract_json_string_field(json, "utf8").as_deref(),
            Some("café µs")
        );
        assert_eq!(extract_json_usize_field(json, "k"), Some(10));
        assert_eq!(extract_json_u64_field(json, "k"), Some(10));
        assert_eq!(extract_json_f64_field(json, "neg_float"), Some(-1.25e-4));
        assert_eq!(extract_json_f64_field(json, "pos_exp"), Some(350.0));
        assert_eq!(extract_json_f64_field(json, "null_num"), None);
        assert_eq!(extract_json_u64_field(json, "null_num"), None);
        assert_eq!(extract_json_bool_field(json, "flag"), Some(true));
        assert_eq!(extract_json_bool_field(json, "empty_flag"), Some(false));

        let arr = extract_json_array_slice(json, "arr").unwrap();
        let objs = split_top_level_json_objects(arr);
        assert_eq!(objs.len(), 2);
        assert_eq!(extract_json_u64_field(objs[0], "id"), Some(1));
        assert_eq!(
            extract_json_string_field(objs[0], "label").as_deref(),
            Some("a[0] }")
        );
        assert_eq!(extract_json_u64_field(objs[1], "id"), Some(2));
        assert_eq!(
            extract_json_string_field(objs[1], "label").as_deref(),
            Some("b\\")
        );

        assert_eq!(
            parse_number_list::<i32>("\"indices\": [10, -2, 30]"),
            vec![10, -2, 30]
        );
        assert_eq!(parse_number_list::<u32>("10, 20, 30"), vec![10, 20, 30]);
        assert!(parse_number_list::<i32>("[1, 2").is_empty());
    }
}
