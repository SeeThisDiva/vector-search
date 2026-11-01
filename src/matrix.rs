//! Contiguous row-major 2D matrix container for dense vectors and neighbor ID tables.

use std::mem;

use crate::error::{Result, VectorSearchError};

/// Flat, cache-friendly row-major 2D matrix of `rows x cols` elements.
#[derive(Debug, Clone, PartialEq)]
pub struct RowMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> RowMatrix<T> {
    /// Creates an empty `(0, 0)` matrix.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }

    /// Constructs a `RowMatrix` from dimensions and flat row-major backing storage.
    ///
    /// If `rows == 0` and `data` is empty, dimensions are normalized to `(0, 0)`.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self> {
        if rows == 0 {
            if !data.is_empty() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "RowMatrix with 0 rows must have empty data, got {} elements",
                    data.len()
                )));
            }
            return Ok(Self::empty());
        }
        if cols == 0 {
            if !data.is_empty() {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "RowMatrix with 0 cols must have empty data, got {} elements",
                    data.len()
                )));
            }
            return Ok(Self {
                rows,
                cols: 0,
                data,
            });
        }
        let expected_len = rows.checked_mul(cols).ok_or_else(|| {
            VectorSearchError::InvalidArgument(format!(
                "Matrix dimensions overflow: rows={rows}, cols={cols}"
            ))
        })?;
        if data.len() != expected_len {
            return Err(VectorSearchError::InvalidArgument(format!(
                "Matrix data length {} does not match rows ({rows}) * cols ({cols}) = {expected_len}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Number of rows (e.g., vector count `N` or query count `Q`).
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (e.g., dimensionality `D` or neighbor count `K`).
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Shape tuple `(rows, cols)`.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns `true` if there are zero elements (`rows == 0 || cols == 0`).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat contiguous slice over all `rows * cols` elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable flat slice over all `rows * cols` elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the matrix and returns the underlying flat `Vec<T>`.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Returns a slice for row `r`.
    ///
    /// # Panics
    /// Panics if `r >= self.rows`.
    #[inline]
    #[must_use]
    pub fn row(&self, r: usize) -> &[T] {
        assert!(
            r < self.rows,
            "row index {r} out of bounds for {} rows",
            self.rows
        );
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Checked row lookup returning `None` if `r >= self.rows`.
    #[inline]
    #[must_use]
    pub fn try_row(&self, r: usize) -> Option<&[T]> {
        if r >= self.rows {
            return None;
        }
        let start = r * self.cols;
        Some(&self.data[start..start + self.cols])
    }

    /// Iterator over contiguous row slices.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |r| self.row(r))
    }

    /// Heap payload memory footprint in bytes (`data.len() * size_of::<T>()`).
    #[inline]
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * mem::size_of::<T>()
    }

    /// Heap payload memory footprint in mebibytes (MiB).
    #[inline]
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

impl<T: Clone> RowMatrix<T> {
    /// Constructs a `RowMatrix` from nested row vectors, verifying uniform width.
    pub fn from_rows(rows: &[Vec<T>]) -> Result<Self> {
        if rows.is_empty() {
            return Ok(Self::empty());
        }
        let cols = rows[0].len();
        let total = rows.len().checked_mul(cols).ok_or_else(|| {
            VectorSearchError::InvalidArgument("RowMatrix::from_rows overflow".to_string())
        })?;
        let mut data = Vec::with_capacity(total);
        for (idx, row) in rows.iter().enumerate() {
            if row.len() != cols {
                return Err(VectorSearchError::InvalidArgument(format!(
                    "Ragged row {idx}: expected length {cols}, got {}",
                    row.len()
                )));
            }
            data.extend_from_slice(row);
        }
        Self::new(rows.len(), cols, data)
    }

    /// Returns a new matrix containing the first `max_rows.min(self.rows)` rows.
    #[must_use]
    pub fn slice_first_rows(&self, max_rows: usize) -> Self {
        if max_rows == 0 || self.rows == 0 {
            return Self::empty();
        }
        let take = max_rows.min(self.rows);
        let end = take * self.cols;
        Self {
            rows: take,
            cols: self.cols,
            data: self.data[..end].to_vec(),
        }
    }
}
