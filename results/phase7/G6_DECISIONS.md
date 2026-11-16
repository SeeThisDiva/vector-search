# Gate G6 Decisions — Approved Protocol (`T5.1`–`T5.7`)

**Date Recorded:** 2026-11-16 (Dev delegated optimal speed + full quality choices across [`results/phase7/DESIGN.md`](./DESIGN.md) §8)

## Approved Protocol

1. **Configuration Scope:** **Option A (`Core Frontier + keep_pruned Control`)** across `100k` (`3` builds, `5` query reps) and `1M` (`1` build, `5` query reps):
   - `C1`: `M=8, efC=100`
   - `C2`: `M=16, efC=100`
   - `C3`: `M=32, efC=200`
   - `C2-nkp`: `M=16, efC=100, keep_pruned_shrink=false` (isolating `keep_pruned` shrink backfill vs FAISS; `C4 (48, 400)` omitted to avoid ~47m extra 1M rebuild while preserving full low-RAM, baseline, and high-recall Pareto coverage).
2. **SIMD Parity Scope:** **Option A (`Stock Baseline SSE2 Parity`)** — compare PyPI `faiss-cpu 1.15.1` (`SIMDLevel_NONE`) directly against default-target Rust (`opt-level = 3, lto = "thin"`), with explicit disclosure that neither uses AVX2.
3. **Multi-Thread Evaluation:** Benchmark `C2 (M=16, efC=100)` Mode 1 batch query throughput across `T in {1 (0x0004), 6 P-cores (0x0555), 16 all logical (0xFFFF)}` for Rust `search_batch_parallel` vs FAISS OpenMP, plus record FAISS `T in {6, 16}` OpenMP build times while explicitly disclosing Rust's builder is single-threaded.
4. **Matched-Recall Targets:** $R^* \in \{0.90, 0.95, 0.98, 0.99, 0.995\}$ via log-latency/log-QPS interpolation without extrapolation.
5. **Fresh Clean-Commit Rust + FAISS Runs:** Execute Rust Mode 1 (`search_batch_parallel(..., 1)`) + Mode 2 (`search_batch_timed_with_tracker`) and FAISS Mode 1 (`idx.search`) + Mode 2 (`idx.search_c`) on the same clean Phase 7 commit (`0x0004` isolated processes), verifying identical Phase 4 graph fingerprints and recording host Windows power scheme in every JSON.
