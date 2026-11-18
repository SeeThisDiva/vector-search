//! From-scratch Vector Search & HNSW Approximate Nearest Neighbor Benchmark Suite.
//!
//! Implemented in 100% safe Rust (`#![forbid(unsafe_code)]`) with zero external crate
//! dependencies (`std`-only).
//!
//! ## Core Library Architecture
//! - Contiguous row-major vector and ID storage ([`matrix::RowMatrix`])
//! - SIMD-friendly 8-way unrolled vector distance kernels ([`distance::Metric`],
//!   [`distance::l2_squared`], [`distance::l2_squared_f64`], [`distance::euclidean`],
//!   [`distance::inner_product`])
//! - Platform-independent deterministic `Xoshiro256**` PRNG ([`rng::DeterministicRng`])
//! - Texmex `.fvecs` / `.ivecs` streaming I/O, SIFT loader/downloader, and filtered metadata
//!   generation ([`dataset`])
//! - Exact brute-force KNN and filtered exact KNN engines with canonical `(distance, id)`
//!   tie-breaking ([`exact_knn::ExactKnn`], [`exact_knn::FilteredExactKnn`])
//! - Hierarchical Navigable Small World graph construction and search ([`hnsw::HnswIndex`],
//!   [`hnsw::HnswConfig`])
//! - Filtered HNSW search strategies (post-filter over-fetch, in-traversal bridge routing, and
//!   bitset pre-filter brute force) with dense bitsets and opaque predicates ([`filter`],
//!   [`hnsw::FilterStrategy`])
//! - Scalar Quantization (`SQ8`) and Product Quantization (`PQ`) over compact HNSW graphs
//!   ([`quantized::HnswSq8Index`], [`quantized::HnswPqIndex`])
//! - Automated `Recall@k`, self-verification invariants, and latency summary statistics ([`recall`])
//! - Cross-platform host hardware detection, peak RSS metadata, and affinity helpers ([`hardware`])
//!
//! ## Benchmark & Evaluation Harnesses
//! - Phases 1–3 core ground-truth, HNSW verification, and SIFT1M scaling ([`benchmark`])
//! - Phase 4 resumable hyperparameter sweep and Pareto frontier ([`phase4`])
//! - Phase 5 filtered search benchmark matrix and sensitivity suite ([`phase5`])
//! - Phase 6 SQ8/PQ quantization matrix and isolated query RSS profiler ([`phase6`])

#![forbid(unsafe_code)]

pub mod benchmark;
#[doc(hidden)]
pub mod cli_util;
pub mod dataset;
pub mod distance;
pub mod error;
pub mod exact_knn;
pub mod filter;
pub mod hardware;
pub mod hnsw;
#[doc(hidden)]
pub mod json_util;
pub mod matrix;
pub mod phase4;
pub mod phase5;
pub mod phase6;
pub mod quantized;
pub mod recall;
pub mod rng;

pub use distance::Metric;
pub use exact_knn::{ExactKnn, FilteredExactKnn, KnnSearchResult, TimedKnnSearchResult};
pub use filter::{
    BitsetOneIterator, DenseBitset, FilterPredicate, FilteredSearchFlags, FilteredSearchOutput,
    FilteredSearchParams, LambdaPredicate, DEFAULT_OPAQUE_EVAL_LIMIT,
};
pub use hnsw::{FilteredSearchCounters, HnswConfig, HnswIndex};
pub use matrix::RowMatrix;
pub use quantized::{HnswPqIndex, HnswSq8Index};
