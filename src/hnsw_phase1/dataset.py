"""Dataset generator and standard Texmex binary I/O for vector search benchmarks."""

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "generate_synthetic_sift",
    "read_fvecs",
    "read_fvecs_prefix",
    "read_ivecs",
    "read_ivecs_prefix",
    "write_fvecs",
    "write_ivecs",
]


def generate_synthetic_sift(
    num_base: int = 10_000,
    num_query: int = 100,
    dim: int = 128,
    num_clusters: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic 128-d SIFT-like vectors using clustered Gaussians.

    Clustered vectors faithfully reproduce realistic manifold topologies
    of high-dimensional visual descriptors, avoiding the concentration-of-measure
    degeneracy inherent in uniform hypercube distributions.

    Parameters
    ----------
    num_base : int
        Number of database vectors to generate. Default 10,000.
    num_query : int
        Number of query vectors to generate. Default 100.
    dim : int
        Dimensionality of each vector. Default 128.
    num_clusters : int
        Number of Gaussian cluster centers. Default 10.
    seed : int
        Fixed random seed for deterministic reproducibility. Default 42.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (base_vectors, query_vectors), both float32 ndarrays of shapes
        (num_base, dim) and (num_query, dim).
    """
    if num_base <= 0:
        raise ValueError(f"num_base must be positive, got {num_base}")
    if num_query <= 0:
        raise ValueError(f"num_query must be positive, got {num_query}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if num_clusters <= 0:
        num_clusters = 1

    rng = np.random.default_rng(seed)

    # 1. Generate cluster centroids on unit hypersphere
    raw_centroids = rng.standard_normal((num_clusters, dim), dtype=np.float32)
    norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    centroids = raw_centroids / norms

    # Scale centroids to characteristic SIFT magnitude (~100-200)
    scale_factor = np.sqrt(float(dim)) * 10.0
    centroids *= scale_factor

    # 2. Generate base vectors clustered around centroids
    cluster_assignments = rng.integers(0, num_clusters, size=num_base)
    base_centroids = centroids[cluster_assignments]
    cluster_std = scale_factor * 0.15
    base_noise = rng.normal(0.0, cluster_std, size=(num_base, dim)).astype(np.float32)
    base_vectors = np.ascontiguousarray(base_centroids + base_noise, dtype=np.float32)

    # 3. Generate query vectors
    # 85% sampled near existing clusters (true positive nearest neighbors)
    # 15% sampled with wider spread to simulate challenging queries
    num_near_cluster = int(num_query * 0.85)
    num_wide_spread = num_query - num_near_cluster

    query_parts = []
    if num_near_cluster > 0:
        q_assignments = rng.integers(0, num_clusters, size=num_near_cluster)
        q_centroids = centroids[q_assignments]
        q_noise = rng.normal(0.0, cluster_std * 0.8, size=(num_near_cluster, dim)).astype(np.float32)
        query_parts.append(q_centroids + q_noise)

    if num_wide_spread > 0:
        wide_noise = rng.normal(0.0, scale_factor, size=(num_wide_spread, dim)).astype(np.float32)
        query_parts.append(wide_noise)

    query_vectors = np.ascontiguousarray(np.vstack(query_parts), dtype=np.float32)

    return base_vectors, query_vectors


def _read_vecs(
    file_path: str | Path,
    *,
    dtype: type[np.float32] | type[np.int32],
    label: str,
    max_rows: int | None = None,
) -> np.ndarray:
    """Read up to ``max_rows`` vectors (or all rows when None) from a Texmex ``.fvecs``/``.ivecs`` file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError(f"max_rows must be positive when set, got {max_rows}")

    file_size = path.stat().st_size
    if file_size == 0:
        return np.empty((0, 0), dtype=dtype)
    if file_size < 4:
        raise ValueError(f"Corrupt {label} file (size {file_size} < 4 bytes): {path}")

    with path.open("rb") as f:
        width = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        if width <= 0:
            raise ValueError(f"Invalid record width {width} in {label} file: {path}")
        record_bytes = 4 + 4 * width
        if file_size % record_bytes != 0:
            raise ValueError(
                f"File size {file_size} is not a multiple of record size {record_bytes} (width={width}) in {path}"
            )
        total_rows = file_size // record_bytes
        if max_rows is not None and max_rows > total_rows:
            raise ValueError(
                f"Requested {max_rows} rows, but {path} contains only {total_rows} rows"
            )
        rows_to_read = total_rows if max_rows is None else max_rows
        f.seek(0)
        data = f.read(rows_to_read * record_bytes)

    # View 4-byte words as target dtype; column 0 is width header, columns 1..width are payload.
    array_view: np.ndarray = np.frombuffer(data, dtype=dtype).reshape(rows_to_read, width + 1)
    return np.ascontiguousarray(array_view[:, 1:], dtype=dtype)


def _write_vecs(
    file_path: str | Path,
    data: np.ndarray,
    *,
    dtype: type[np.float32] | type[np.int32],
) -> None:
    """Write a 1D or 2D ndarray to a standard Texmex ``.fvecs`` or ``.ivecs`` binary file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr: np.ndarray[Any, Any] = np.ascontiguousarray(data, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

    num_vecs, width = arr.shape
    if num_vecs == 0:
        path.write_bytes(b"")
        return

    # Prefix each row with its 4-byte int32 width reinterpreted as `dtype`.
    width_col = np.full((num_vecs, 1), width, dtype=np.int32).view(dtype)
    record = np.hstack((width_col, arr))
    path.write_bytes(record.tobytes())


def read_fvecs(file_path: str | Path) -> np.ndarray:
    """
    Reads a standard Texmex .fvecs binary file into a 2D float32 ndarray.

    In the .fvecs format, each record consists of:
    - 4-byte int32 specifying vector dimension d
    - d * 4 bytes of IEEE 754 float32 components

    Parameters
    ----------
    file_path : str | Path
        Path to the .fvecs file.

    Returns
    -------
    np.ndarray
        Array of shape (N, d), dtype float32. Empty array (0, 0) if file is empty.
    """
    return _read_vecs(file_path, dtype=np.float32, label=".fvecs")


def read_fvecs_prefix(file_path: str | Path, max_rows: int | None = None) -> np.ndarray:
    """Read up to ``max_rows`` vectors from a Texmex ``.fvecs`` file without loading the full file."""
    return _read_vecs(file_path, dtype=np.float32, label=".fvecs", max_rows=max_rows)


def write_fvecs(file_path: str | Path, data: np.ndarray) -> None:
    """
    Writes a 2D float32 ndarray into standard Texmex .fvecs binary format.

    Parameters
    ----------
    file_path : str | Path
        Path to output .fvecs file. Parent directories are created if missing.
    data : np.ndarray
        2D array of shape (N, d), or 1D array of shape (d,).
    """
    _write_vecs(file_path, data, dtype=np.float32)


def read_ivecs(file_path: str | Path) -> np.ndarray:
    """
    Reads a standard Texmex .ivecs binary file into a 2D int32 ndarray.

    In the .ivecs format, each record consists of:
    - 4-byte int32 specifying number of neighbor indices k
    - k * 4 bytes of 32-bit signed integer neighbor IDs

    Parameters
    ----------
    file_path : str | Path
        Path to the .ivecs file.

    Returns
    -------
    np.ndarray
        Array of shape (N, k), dtype int32. Empty array (0, 0) if file is empty.
    """
    return _read_vecs(file_path, dtype=np.int32, label=".ivecs")


def read_ivecs_prefix(file_path: str | Path, max_rows: int | None = None) -> np.ndarray:
    """Read up to ``max_rows`` rows from a Texmex ``.ivecs`` file without loading the full file."""
    return _read_vecs(file_path, dtype=np.int32, label=".ivecs", max_rows=max_rows)


def write_ivecs(file_path: str | Path, data: np.ndarray) -> None:
    """
    Writes a 2D int32 ndarray into standard Texmex .ivecs binary format.

    Parameters
    ----------
    file_path : str | Path
        Path to output .ivecs file. Parent directories are created if missing.
    data : np.ndarray
        2D array of shape (N, k), or 1D array of shape (k,).
    """
    _write_vecs(file_path, data, dtype=np.int32)
