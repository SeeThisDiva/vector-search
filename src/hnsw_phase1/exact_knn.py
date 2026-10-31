"""Brute-force exact K-Nearest Neighbor (KNN) engine with deterministic tie-breaking."""

import numpy as np


class ExactKNN:
    """
    Exact brute-force K-Nearest Neighbors search engine.

    Computes exact distances using batched matrix multiplication (BLAS GEMM)
    with non-negative clamping and canonical (distance, id) deterministic tie-breaking.
    Serves as the benchmark ground truth reference for all ANN evaluation.
    """

    def __init__(self, metric: str = "l2"):
        """
        Parameters
        ----------
        metric : str
            Distance metric to optimize.
            - "l2" or "l2_squared" (default): Squared Euclidean distance (standard for SIFT1M and FAISS IndexFlatL2)
            - "euclidean": True Euclidean distance sqrt(L2^2)
            - "ip" or "inner_product": Negative dot product (closest = highest dot product)
        """
        metric_normalized = metric.lower()
        if metric_normalized in ("l2", "l2_squared", "sqeuclidean"):
            self.metric = "l2"
        elif metric_normalized == "euclidean":
            self.metric = "euclidean"
        elif metric_normalized in ("ip", "inner_product"):
            self.metric = "ip"
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Expected 'l2', 'euclidean', or 'ip'.")

        self.base_vectors: np.ndarray | None = None
        self.base_norms_sq: np.ndarray | None = None  # (1, N)
        self.num_base: int = 0
        self.dim: int = 0

    def fit(self, base_vectors: np.ndarray) -> "ExactKNN":
        """
        Stores reference database vectors and precomputes squared norms for GEMM acceleration.

        Parameters
        ----------
        base_vectors : np.ndarray
            2D float32 array of shape (N, d).

        Returns
        -------
        self : ExactKNN
        """
        arr = np.ascontiguousarray(base_vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"base_vectors must be a 2D array, got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("base_vectors cannot be empty")

        self.base_vectors = arr
        self.num_base, self.dim = arr.shape

        if self.metric in ("l2", "euclidean"):
            # Precompute row squared norms for base vectors: shape (1, N)
            self.base_norms_sq = np.sum(arr ** 2, axis=1, keepdims=True).T
        else:
            self.base_norms_sq = None

        return self

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        batch_size: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Searches for top-k exact nearest neighbors for each query vector.

        Parameters
        ----------
        query_vectors : np.ndarray
            2D float32 array of shape (Q, d), or 1D array of shape (d,).
        k : int
            Number of nearest neighbors to retrieve per query (default 10).
        batch_size : int
            Number of queries to process concurrently via BLAS GEMM (default 500).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (indices, distances) where:
            - indices is int32 ndarray of shape (Q, k) containing 0-indexed database IDs
            - distances is float32 ndarray of shape (Q, k) containing exact computed distances
            Both ordered ascending by (distance, id) for deterministic reproducibility.
        """
        if self.base_vectors is None or self.num_base == 0:
            raise RuntimeError("ExactKNN must be fitted with base_vectors before search()")

        q_arr = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        elif q_arr.ndim != 2:
            raise ValueError(f"query_vectors must be 1D or 2D array, got shape {q_arr.shape}")

        num_queries, q_dim = q_arr.shape
        if q_dim != self.dim:
            raise ValueError(
                f"Dimension mismatch: queries have dim={q_dim}, base vectors have dim={self.dim}"
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        effective_k = min(k, self.num_base)

        all_indices = np.empty((num_queries, effective_k), dtype=np.int32)
        all_distances = np.empty((num_queries, effective_k), dtype=np.float32)

        # Process queries in batches to bound working memory
        for b_start in range(0, num_queries, batch_size):
            b_end = min(b_start + batch_size, num_queries)
            q_batch = q_arr[b_start:b_end]
            curr_batch_size = b_end - b_start

            # 1. Compute pairwise distances via BLAS GEMM
            if self.metric in ("l2", "euclidean"):
                # D^2 = ||Q||^2 + ||B||^2 - 2 * Q @ B^T
                # Note: GEMM expansion uses OpenBLAS SGEMM with AVX2 FMA kernels.
                # Due to floating-point non-associativity between pairwise summation
                # and SIMD FMA block accumulation, identical vectors (q = b) may yield
                # tiny positive residuals on the order of 1 ULP (~1e-7 relative error).
                # Distances are clamped to >= 0.0 to prevent negative rounding errors;
                # positive threshold clamping is intentionally avoided to preserve metric
                # separation for closely-spaced distinct points and avoid array allocation overhead.
                assert self.base_norms_sq is not None
                q_norms_sq = np.sum(q_batch ** 2, axis=1, keepdims=True)  # (batch, 1)
                dists = q_norms_sq + self.base_norms_sq - 2.0 * np.dot(q_batch, self.base_vectors.T)
                # Non-negative clamping to eliminate negative floating-point rounding errors
                np.maximum(0.0, dists, out=dists)
            else:  # "ip"
                # Minimize negative dot product -> highest dot product is closest
                dists = -np.dot(q_batch, self.base_vectors.T)

            # 2. Extract top-k with deterministic (distance, id) tie-breaking
            for i in range(curr_batch_size):
                global_q_idx = b_start + i
                d_row = dists[i]

                if effective_k >= self.num_base:
                    # Sort entire database
                    order = np.lexsort((np.arange(self.num_base, dtype=np.int32), d_row))
                    sel_indices = order[:effective_k]
                else:
                    # Fast partition of top-k
                    part = np.argpartition(d_row, effective_k - 1)[:effective_k]
                    max_part_d = np.max(d_row[part])

                    # Check for ties spanning the partition boundary
                    boundary_ties = np.flatnonzero(d_row <= max_part_d)
                    if len(boundary_ties) > effective_k:
                        # Tie detected at boundary: sort all candidate tied elements deterministically
                        sorted_ties = boundary_ties[np.lexsort((boundary_ties, d_row[boundary_ties]))]
                        sel_indices = sorted_ties[:effective_k]
                    else:
                        # No ties spanning boundary: sort top-k partition deterministically
                        sel_indices = part[np.lexsort((part, d_row[part]))]

                all_indices[global_q_idx] = sel_indices
                sel_dists = d_row[sel_indices]

                if self.metric == "euclidean":
                    sel_dists = np.sqrt(sel_dists)
                elif self.metric == "ip":
                    sel_dists = -sel_dists  # Restore original dot product

                all_distances[global_q_idx] = sel_dists

        return all_indices, all_distances
