//! Shared helpers for integration tests.

use std::path::Path;

use vector_search::dataset::{load_or_download_sift, DatasetSource, LoadedDataset, SiftPreset};

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
