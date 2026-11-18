//! Texmex `.fvecs` / `.ivecs` binary serialization, deterministic synthetic SIFT generator,
//! public SIFT (`siftsmall` & `SIFT1M`) dataset downloader, and filtered metadata generation.

pub mod metadata;
pub mod sift;
pub mod texmex_io;

pub use metadata::{
    deterministic_kmeans_plus_plus, generate_mode_a_bitset, generate_mode_b_bitset,
    generate_mode_b_bitset_detailed, generate_mode_b_bitset_with_seed, read_bitset, write_bitset,
    write_bitset_as_ivecs, ModeBMetadata, S_CLUST_DEFAULT,
};
pub use sift::{
    generate_synthetic_sift, load_custom_dataset, load_custom_dataset_with_options,
    load_or_download_sift, load_or_download_sift_with_options, DatasetSource, LoadedDataset,
    SiftPreset,
};
pub use texmex_io::{
    inspect_fvecs_header, read_fvecs, read_fvecs_max_rows, read_ivecs, read_ivecs_max_rows,
    stream_fvecs_chunks, stream_fvecs_chunks_max_rows, write_fvecs, write_ivecs, FvecsHeader,
};
