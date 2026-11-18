//! Shared CLI argument cursor, parsing helpers, and common SIFT dataset preset choices.
//!
//! Internal support module shared across benchmark binaries and runners.

#![forbid(unsafe_code)]

use std::fmt;
use std::str::FromStr;

use crate::dataset::SiftPreset;
use crate::error::{Result, VectorSearchError};

/// Stateful cursor over CLI argument tokens (`args: &[String]`).
#[derive(Debug, Clone)]
pub struct ArgCursor<'a> {
    args: &'a [String],
    idx: usize,
}

impl<'a> ArgCursor<'a> {
    /// Creates a new cursor starting at index `0`.
    #[must_use]
    pub const fn new(args: &'a [String]) -> Self {
        Self { args, idx: 0 }
    }

    /// Current zero-based token index.
    #[inline]
    #[must_use]
    pub const fn position(&self) -> usize {
        self.idx
    }

    /// Returns `true` when no unread arguments remain.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.idx >= self.args.len()
    }

    /// Peeks at the next unread argument without advancing the cursor.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<&'a str> {
        self.args.get(self.idx).map(String::as_str)
    }

    /// Advances past and returns the next argument token.
    #[inline]
    pub fn next_flag(&mut self) -> Option<&'a str> {
        let tok = self.args.get(self.idx)?.as_str();
        self.idx += 1;
        Some(tok)
    }

    /// Consumes the value token immediately following `flag`.
    pub fn next_value(&mut self, flag: &str) -> Result<&'a str> {
        let val = self.args.get(self.idx).ok_or_else(|| {
            VectorSearchError::InvalidArgument(format!("Missing argument for {flag}"))
        })?;
        self.idx += 1;
        Ok(val.as_str())
    }

    /// Consumes and parses the value token immediately following `flag`.
    pub fn parse_value<T>(&mut self, flag: &str) -> Result<T>
    where
        T: FromStr,
        T::Err: fmt::Display,
    {
        let raw = self.next_value(flag)?;
        raw.parse::<T>().map_err(|e| {
            VectorSearchError::InvalidArgument(format!("Invalid value for {flag} ('{raw}'): {e}"))
        })
    }
}

/// Reads the argument value immediately following `flag` at `args[*idx]` and advances `*idx` by 2.
pub fn next_str<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str> {
    if *idx + 1 >= args.len() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Missing argument for {flag}"
        )));
    }
    let val = args[*idx + 1].as_str();
    *idx += 2;
    Ok(val)
}

/// Reads and parses the argument value immediately following `flag` at `args[*idx]`,
/// advancing `*idx` by 2.
pub fn parse_next<T>(args: &[String], idx: &mut usize, flag: &str) -> Result<T>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    let raw = next_str(args, idx, flag)?;
    raw.parse::<T>().map_err(|e| {
        VectorSearchError::InvalidArgument(format!("Invalid value for {flag} ('{raw}'): {e}"))
    })
}

/// Parses a non-empty comma-separated list of `T` values for `flag`.
pub fn parse_csv_list<T>(raw: &str, flag: &str) -> Result<Vec<T>>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    let mut values = Vec::new();
    for part in raw.split(',').filter(|p| !p.trim().is_empty()) {
        let trimmed = part.trim();
        let val = trimmed.parse::<T>().map_err(|e| {
            VectorSearchError::InvalidArgument(format!("Invalid value '{trimmed}' in {flag}: {e}"))
        })?;
        values.push(val);
    }
    if values.is_empty() {
        return Err(VectorSearchError::InvalidArgument(format!(
            "{flag} list cannot be empty"
        )));
    }
    Ok(values)
}

/// Parses a non-empty comma-separated list of strictly positive (`>= 1`) `usize` values.
pub fn parse_positive_csv_usizes(raw: &str, flag: &str) -> Result<Vec<usize>> {
    let values = parse_csv_list::<usize>(raw, flag)?;
    if values.contains(&0) {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Values in {flag} must be >= 1, got 0"
        )));
    }
    Ok(values)
}

/// Standard CLI `--dataset` choice selecting between authentic SIFT presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiftPresetChoice {
    Sift1M,
    SiftSmall,
}

impl SiftPresetChoice {
    /// Parses `"sift1m"` or `"siftsmall"` (case-insensitive, trimmed).
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sift1m" => Ok(Self::Sift1M),
            "siftsmall" => Ok(Self::SiftSmall),
            other => Err(VectorSearchError::InvalidArgument(format!(
                "Invalid --dataset '{other}': expected 'sift1m' or 'siftsmall'"
            ))),
        }
    }

    /// Canonical lowercase CLI string (`"sift1m"` or `"siftsmall"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sift1M => "sift1m",
            Self::SiftSmall => "siftsmall",
        }
    }

    /// Converts to [`SiftPreset`].
    #[must_use]
    pub const fn to_preset(self) -> SiftPreset {
        match self {
            Self::Sift1M => SiftPreset::Sift1M,
            Self::SiftSmall => SiftPreset::SiftSmall,
        }
    }
}

impl From<SiftPresetChoice> for SiftPreset {
    fn from(choice: SiftPresetChoice) -> Self {
        choice.to_preset()
    }
}

impl fmt::Display for SiftPresetChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SiftPresetChoice {
    type Err = VectorSearchError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arg_cursor_and_index_helpers() {
        let args = vec![
            "--k".to_string(),
            "10".to_string(),
            "--flag".to_string(),
            "--name".to_string(),
            "abc".to_string(),
        ];

        let mut i = 0;
        assert_eq!(parse_next::<usize>(&args, &mut i, "--k").unwrap(), 10);
        assert_eq!(i, 2);
        assert_eq!(args[i], "--flag");
        i += 1;
        assert_eq!(next_str(&args, &mut i, "--name").unwrap(), "abc");
        assert_eq!(i, 5);
        assert!(next_str(&args, &mut i, "--missing").is_err());

        let mut cur = ArgCursor::new(&args);
        assert_eq!(cur.next_flag(), Some("--k"));
        assert_eq!(cur.parse_value::<usize>("--k").unwrap(), 10);
        assert_eq!(cur.next_flag(), Some("--flag"));
        assert_eq!(cur.next_flag(), Some("--name"));
        assert_eq!(cur.next_value("--name").unwrap(), "abc");
        assert!(cur.is_empty());
        assert!(cur.next_value("--eof").is_err());
    }

    #[test]
    fn test_csv_and_preset_helpers() {
        assert_eq!(
            parse_csv_list::<f64>("0.5, 0.1, , 0.01", "--sigmas").unwrap(),
            vec![0.5, 0.1, 0.01]
        );
        assert_eq!(
            parse_positive_csv_usizes("10, 20, 50", "--ef-search-list").unwrap(),
            vec![10, 20, 50]
        );
        assert!(parse_positive_csv_usizes("10, 0, 50", "--ef-search-list").is_err());
        assert!(parse_positive_csv_usizes(" , ", "--ef-search-list").is_err());

        assert_eq!(
            SiftPresetChoice::parse(" SIFT1M ").unwrap().to_preset(),
            SiftPreset::Sift1M
        );
        assert_eq!(
            SiftPresetChoice::parse("siftsmall").unwrap().as_str(),
            "siftsmall"
        );
        assert!(SiftPresetChoice::parse("other").is_err());
    }
}
