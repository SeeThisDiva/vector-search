# Vector Search: Hierarchical Navigable Small World (HNSW)

A high-performance Approximate Nearest Neighbor (ANN) vector search engine implemented from first principles in Python, grounded in the seminal paper:
> *Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE TPAMI.*

---

## Overview

High-dimensional vector search powers modern dense retrieval, recommendation engines, and retrieval-augmented generation (RAG). While brute-force exact $k$-NN scales linearly as $\mathcal{O}(N \cdot D)$, HNSW organizes vectors into a multi-layer probabilistic graph hierarchy that delivers $\mathcal{O}(\log N)$ logarithmic search complexity with high recall.

This repository tracks the step-by-step implementation, optimization, and benchmarking of HNSW from mathematical foundations to production-grade performance.

---

## Architecture & Roadmap

```
Layer L_max :   [Node A] --------------------> [Node B]            (Long-range routing)
                     \                              \
Layer L_1   :   [Node A] -----> [Node C] -----> [Node B]           (Intermediate density)
                     \             \                \
Layer L_0   :   [Node A] -> [D] -> [C] -> [E] -> [Node B] -> ...   (All N elements)
```

- [x] **Phase 1: Foundations & Exact $k$-NN Ground Truth**
  - Synthetic 128-d SIFT-like Gaussian cluster generator
  - Texmex binary I/O formats (`.fvecs`, `.ivecs`)
  - Brute-force exact $k$-NN baseline (L2 Euclidean distance)
  - Recall@k metrics and verification harness
- [ ] **Phase 2: NSW & Entry Point Greedy Routing**
- [ ] **Phase 3: Multi-Layer Hierarchical Graph Construction**
- [ ] **Phase 4: Heuristic Neighbor Selection (Shrinking & Diversity)**
- [ ] **Phase 5: Search-Layer Beam Search & Priority Queues (`efSearch`)**
- [ ] **Phase 6: Dynamic Deletions & Soft-Tombstones**
- [ ] **Phase 7: Quantization & SIMD Vectorization**
- [ ] **Phase 8: FAISS & ScaNN Comparative Benchmarking**

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/SeeThisDiva/vector-search.git
cd vector-search
pip install -r requirements.txt
```

### 2. Run Phase 1 Ground Truth & Verification

Generate synthetic SIFT vectors and verify exact $k$-NN:

```bash
python run_phase1.py --num-base 10000 --num-query 100 --dim 128 --k 10
```

### 3. Run Test Suite

```bash
python -m pytest tests/ -v --cov=src
```

### 4. Code Quality & Linting

```bash
ruff check .
mypy src tests
```

---

## Detailed Specifications

For in-depth mathematical formulations, pseudocode, and complexity proofs, see [docs/HNSW_ALGORITHM.md](docs/HNSW_ALGORITHM.md).
