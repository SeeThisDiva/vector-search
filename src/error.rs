//! Error types for the vector search engine and benchmark harness.

use std::fmt;
use std::io;

/// Unified error type across dataset I/O, exact KNN search, and evaluation harnesses.
#[derive(Debug)]
pub enum VectorSearchError {
    /// Underlying filesystem or process I/O error.
    Io(io::Error),
    /// Invalid user/caller parameter (e.g., `k == 0`, `dim == 0`, non-finite float).
    InvalidArgument(String),
    /// Malformed or truncated `.fvecs` / `.ivecs` binary payload.
    CorruptData(String),
    /// Dimensionality mismatch between base index and query vectors.
    DimensionMismatch { expected: usize, actual: usize },
    /// Search invoked before base vectors were indexed.
    UnfittedIndex(&'static str),
    /// Self-recall verification failed (`Recall@k(GT, GT) != 1.0`).
    SelfVerificationFailed { k: usize, actual_recall: f64 },
    /// Remote dataset download or archive extraction failed.
    DownloadFailed(String),
    /// Algorithmic verification or invariant check failed.
    VerificationFailed(String),
}

impl fmt::Display for VectorSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::CorruptData(msg) => write!(f, "Corrupt binary data: {msg}"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "Dimension mismatch: expected dim={expected}, got dim={actual}"
            ),
            Self::UnfittedIndex(msg) => write!(f, "Unfitted index: {msg}"),
            Self::SelfVerificationFailed { k, actual_recall } => write!(
                f,
                "Self-verification failed: Recall@{k}(GroundTruth, GroundTruth) = {actual_recall:.6} (expected 1.0)"
            ),
            Self::DownloadFailed(msg) => write!(f, "Dataset download failed: {msg}"),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {msg}"),
        }
    }
}

impl std::error::Error for VectorSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for VectorSearchError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Convenience result alias for vector search operations.
pub type Result<T> = std::result::Result<T, VectorSearchError>;
