//! From-scratch Vector Search & HNSW Approximate Nearest Neighbor Benchmark Suite (Phase 1).
//!
//! Provides:
//! - Contiguous row-major vector storage ([`matrix::RowMatrix`])
//! - Exact vector distance kernels ([`distance::Metric`], [`distance::l2_squared`])
//! - Platform-independent deterministic PRNG ([`rng::DeterministicRng`])
//! - Standard Texmex `.fvecs` / `.ivecs` binary readers/writers and SIFT downloader ([`dataset`])
//! - Brute-force exact K-Nearest Neighbor engine with canonical `(distance, id)` tie-breaking ([`exact_knn::ExactKnn`])
//! - Automated `Recall@k` harness, self-verification invariant, and latency percentiles ([`recall`])
//! - Cross-platform host hardware detection ([`hardware`])
//! - End-to-end Phase 1 benchmark runner ([`benchmark`])

#![forbid(unsafe_code)]

pub mod benchmark;
pub mod dataset;
pub mod distance;
pub mod error;
pub mod exact_knn;
pub mod filter;
pub mod hardware;
pub mod hnsw;
pub mod matrix;
pub mod phase4;
pub mod recall;
pub mod rng;

pub use filter::{
    BitsetOneIterator, DenseBitset, FilterPredicate, FilteredSearchFlags, FilteredSearchOutput,
    FilteredSearchParams, LambdaPredicate,
};
