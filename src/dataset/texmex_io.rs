//! Texmex `.fvecs` / `.ivecs` little-endian binary readers, writers, and streaming chunk helpers.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::json_util::ensure_parent_dir;
use crate::matrix::RowMatrix;

pub(crate) const IO_BUF_CAPACITY: usize = 256 * 1024;

/// Validated metadata header for a uniform-dimension `.fvecs` file on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FvecsHeader {
    /// Total number of vector records in the file.
    pub num_rows: usize,
    /// Vector dimensionality (`d` per record).
    pub dim: usize,
}

fn validate_vecs_header(
    path: &Path,
    file_size: u64,
    first_dim_i32: i32,
    ext_label: &str,
    width_label: &str,
) -> Result<usize> {
    if first_dim_i32 <= 0 {
        return Err(VectorSearchError::CorruptData(format!(
            "Invalid {width_label} {first_dim_i32} in {ext_label} file: {}",
            path.display()
        )));
    }
    let dim = first_dim_i32 as usize;
    let record_bytes = 4_u64
        .checked_add((dim as u64).checked_mul(4).ok_or_else(|| {
            VectorSearchError::CorruptData(format!("{width_label} byte size overflow"))
        })?)
        .ok_or_else(|| VectorSearchError::CorruptData("Record byte size overflow".to_string()))?;

    if !file_size.is_multiple_of(record_bytes) {
        return Err(VectorSearchError::CorruptData(format!(
            "File size {file_size} is not a multiple of record size {record_bytes} (dim={dim}) in {}",
            path.display()
        )));
    }

    let total_rows_u64 = file_size / record_bytes;
    usize::try_from(total_rows_u64).map_err(|_| {
        VectorSearchError::CorruptData(format!("Row count {total_rows_u64} exceeds usize"))
    })
}

/// Validates and inspects the header and record alignment of a non-empty `.fvecs` file.
pub fn inspect_fvecs_header(path: &Path) -> Result<FvecsHeader> {
    let meta = fs::metadata(path)?;
    let file_size = meta.len();
    if file_size == 0 {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Empty .fvecs file: {}",
            path.display()
        )));
    }
    if file_size < 4 {
        return Err(VectorSearchError::CorruptData(format!(
            "Corrupt .fvecs file (size {file_size} < 4 bytes): {}",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;
    let first_dim_i32 = i32::from_le_bytes(header);
    let num_rows = validate_vecs_header(path, file_size, first_dim_i32, ".fvecs", "dimension")?;
    Ok(FvecsHeader {
        num_rows,
        dim: first_dim_i32 as usize,
    })
}

/// Streams all rows from an `.fvecs` file in chunks of at most `chunk_rows`, validating
/// per-record dimension headers and float finiteness without materializing `RowMatrix<f32>`.
///
/// The callback receives `(row_offset, rows_in_chunk, chunk_f32_slice)`.
pub fn stream_fvecs_chunks<F>(path: &Path, chunk_rows: usize, f: F) -> Result<FvecsHeader>
where
    F: FnMut(usize, usize, &[f32]),
{
    stream_fvecs_chunks_max_rows(path, None, chunk_rows, f)
}

/// Streams up to `max_rows` rows from an `.fvecs` file in chunks of at most `chunk_rows`,
/// validating per-record dimension headers and float finiteness.
///
/// The callback receives `(row_offset, rows_in_chunk, chunk_f32_slice)`.
pub fn stream_fvecs_chunks_max_rows<F>(
    path: &Path,
    max_rows: Option<usize>,
    chunk_rows: usize,
    mut on_chunk: F,
) -> Result<FvecsHeader>
where
    F: FnMut(usize, usize, &[f32]),
{
    if chunk_rows == 0 {
        return Err(VectorSearchError::InvalidArgument(
            "chunk_rows must be positive (> 0)".to_string(),
        ));
    }
    if max_rows == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_rows must be positive (> 0) when specified".to_string(),
        ));
    }

    let header = inspect_fvecs_header(path)?;
    let dim = header.dim;
    let expected_dim_i32 = dim as i32;
    let num_vecs = match max_rows {
        Some(limit) => header.num_rows.min(limit),
        None => header.num_rows,
    };

    let eff_chunk_rows = chunk_rows.min(num_vecs).max(1);
    let mut chunk_f32 = vec![0.0_f32; eff_chunk_rows * dim];
    let mut row_bytes = vec![0_u8; dim * 4];
    let mut rec_hdr = [0_u8; 4];

    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);
    let mut rows_read = 0_usize;

    while rows_read < num_vecs {
        let take = eff_chunk_rows.min(num_vecs - rows_read);
        for r in 0..take {
            let global_row = rows_read + r;
            reader.read_exact(&mut rec_hdr)?;
            let rec_dim = i32::from_le_bytes(rec_hdr);
            if rec_dim != expected_dim_i32 {
                return Err(VectorSearchError::CorruptData(format!(
                    "Inconsistent .fvecs record dimension at row {global_row}: expected {dim}, got {rec_dim} in {}",
                    path.display()
                )));
            }
            reader.read_exact(&mut row_bytes)?;
            let dst = &mut chunk_f32[r * dim..(r + 1) * dim];
            for (col, (&bytes, out_val)) in row_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.iter_mut())
                .enumerate()
            {
                let v = f32::from_le_bytes(bytes);
                if !v.is_finite() {
                    return Err(VectorSearchError::InvalidArgument(format!(
                        "base_vectors contains non-finite value (NaN/Inf) at ({global_row}, {col})"
                    )));
                }
                *out_val = v;
            }
        }
        on_chunk(rows_read, take, &chunk_f32[..take * dim]);
        rows_read += take;
    }

    Ok(FvecsHeader {
        num_rows: num_vecs,
        dim,
    })
}

fn read_vecs_max_rows<T, F>(
    path: &Path,
    max_rows: Option<usize>,
    ext_label: &str,
    width_label: &str,
    decode: F,
) -> Result<RowMatrix<T>>
where
    F: Fn([u8; 4]) -> T,
{
    if max_rows == Some(0) {
        return Err(VectorSearchError::InvalidArgument(
            "max_rows must be positive (> 0) when specified".to_string(),
        ));
    }

    let meta = fs::metadata(path)?;
    let file_size = meta.len();

    if file_size == 0 {
        return Ok(RowMatrix::empty());
    }
    if file_size < 4 {
        return Err(VectorSearchError::CorruptData(format!(
            "Corrupt {ext_label} file (size {file_size} < 4 bytes): {}",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(IO_BUF_CAPACITY, file);

    let mut header = [0_u8; 4];
    reader.read_exact(&mut header)?;
    let first_width_i32 = i32::from_le_bytes(header);
    let total_rows =
        validate_vecs_header(path, file_size, first_width_i32, ext_label, width_label)?;
    let width = first_width_i32 as usize;

    let num_rows = match max_rows {
        Some(limit) => total_rows.min(limit),
        None => total_rows,
    };
    let total_elems = num_rows.checked_mul(width).ok_or_else(|| {
        VectorSearchError::CorruptData("Total element count overflows usize".to_string())
    })?;

    let mut data = Vec::with_capacity(total_elems);
    let mut row_bytes = vec![0_u8; width * 4];

    // Record 0 payload (header already read)
    reader.read_exact(&mut row_bytes)?;
    for &chunk in row_bytes.as_chunks::<4>().0 {
        data.push(decode(chunk));
    }

    // Remaining records 1..num_rows, verifying every record's width header
    for rec_idx in 1..num_rows {
        reader.read_exact(&mut header)?;
        let rec_width = i32::from_le_bytes(header);
        if rec_width != first_width_i32 {
            return Err(VectorSearchError::CorruptData(format!(
                "Inconsistent {ext_label} record {width_label} at row {rec_idx}: expected {width}, got {rec_width} in {}",
                path.display()
            )));
        }
        reader.read_exact(&mut row_bytes)?;
        for &chunk in row_bytes.as_chunks::<4>().0 {
            data.push(decode(chunk));
        }
    }

    RowMatrix::new(num_rows, width, data)
}

fn write_vecs<T: Copy, F>(
    path: &Path,
    matrix: &RowMatrix<T>,
    ext_label: &str,
    width_label: &str,
    encode: F,
) -> Result<()>
where
    F: Fn(T) -> [u8; 4],
{
    ensure_parent_dir(path)?;

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

    if matrix.rows() == 0 {
        writer.flush()?;
        return Ok(());
    }

    if matrix.cols() == 0 {
        return Err(VectorSearchError::InvalidArgument(format!(
            "Cannot write {ext_label} with 0 columns and non-zero rows"
        )));
    }
    let width_i32 = i32::try_from(matrix.cols()).map_err(|_| {
        VectorSearchError::InvalidArgument(format!(
            "{width_label} {} exceeds i32::MAX for {ext_label} header",
            matrix.cols()
        ))
    })?;
    let header = width_i32.to_le_bytes();

    for row in matrix.iter_rows() {
        writer.write_all(&header)?;
        for &val in row {
            writer.write_all(&encode(val))?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Reads a standard Texmex `.fvecs` little-endian binary file into a `RowMatrix<f32>`.
///
/// Record layout:
/// - 4-byte signed 32-bit integer `d` (little-endian)
/// - `d` IEEE 754 `f32` values (little-endian, `4 * d` bytes)
///
/// An empty (0-byte) file yields an empty `(0, 0)` matrix.
pub fn read_fvecs<P: AsRef<Path>>(path: P) -> Result<RowMatrix<f32>> {
    read_fvecs_max_rows(path, None)
}

/// Reads up to `max_rows` records from a standard Texmex `.fvecs` file after validating
/// full-file record alignment, avoiding allocating the entire file when only a prefix is needed.
pub fn read_fvecs_max_rows<P: AsRef<Path>>(
    path: P,
    max_rows: Option<usize>,
) -> Result<RowMatrix<f32>> {
    read_vecs_max_rows(
        path.as_ref(),
        max_rows,
        ".fvecs",
        "dimension",
        f32::from_le_bytes,
    )
}

/// Writes a `RowMatrix<f32>` into standard Texmex `.fvecs` little-endian binary format.
pub fn write_fvecs<P: AsRef<Path>>(path: P, matrix: &RowMatrix<f32>) -> Result<()> {
    write_vecs(
        path.as_ref(),
        matrix,
        ".fvecs",
        "Dimension",
        f32::to_le_bytes,
    )
}

/// Reads a standard Texmex `.ivecs` little-endian binary file into a `RowMatrix<i32>`.
///
/// Record layout:
/// - 4-byte signed 32-bit integer `k` (little-endian)
/// - `k` signed 32-bit integer neighbor IDs (little-endian, `4 * k` bytes)
pub fn read_ivecs<P: AsRef<Path>>(path: P) -> Result<RowMatrix<i32>> {
    read_ivecs_max_rows(path, None)
}

/// Reads up to `max_rows` records from a standard Texmex `.ivecs` file after validating
/// full-file record alignment.
pub fn read_ivecs_max_rows<P: AsRef<Path>>(
    path: P,
    max_rows: Option<usize>,
) -> Result<RowMatrix<i32>> {
    read_vecs_max_rows(
        path.as_ref(),
        max_rows,
        ".ivecs",
        "neighbor width",
        i32::from_le_bytes,
    )
}

/// Writes a `RowMatrix<i32>` into standard Texmex `.ivecs` little-endian binary format.
pub fn write_ivecs<P: AsRef<Path>>(path: P, matrix: &RowMatrix<i32>) -> Result<()> {
    write_vecs(
        path.as_ref(),
        matrix,
        ".ivecs",
        "Column count",
        i32::to_le_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_inspect_and_stream_fvecs_chunks() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "vs_stream_fvecs_test_{}_{}.fvecs",
            std::process::id(),
            nanos
        ));

        let rows = 11_usize;
        let dim = 4_usize;
        let data: Vec<f32> = (0..(rows * dim)).map(|v| v as f32 * 0.5).collect();
        let mat = RowMatrix::new(rows, dim, data.clone()).unwrap();
        write_fvecs(&path, &mat).unwrap();

        let hdr = inspect_fvecs_header(&path).unwrap();
        assert_eq!(
            hdr,
            FvecsHeader {
                num_rows: 11,
                dim: 4
            }
        );

        let mut collected = Vec::new();
        let mut offsets = Vec::new();
        let streamed_hdr = stream_fvecs_chunks(&path, 4, |offset, count, slice| {
            offsets.push((offset, count));
            collected.extend_from_slice(slice);
        })
        .unwrap();
        assert_eq!(streamed_hdr, hdr);
        assert_eq!(offsets, vec![(0, 4), (4, 4), (8, 3)]);
        assert_eq!(collected, data);

        let mut partial = Vec::new();
        let partial_hdr = stream_fvecs_chunks_max_rows(&path, Some(5), 3, |_, _, slice| {
            partial.extend_from_slice(slice);
        })
        .unwrap();
        assert_eq!(
            partial_hdr,
            FvecsHeader {
                num_rows: 5,
                dim: 4
            }
        );
        assert_eq!(partial, data[..5 * dim]);

        let _ = fs::remove_file(path);
    }
}
