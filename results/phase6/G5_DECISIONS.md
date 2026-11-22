# Gate G5 Decisions — Approved Scope (`T4.1`–`T4.9`)

**Date Recorded:** 2026-11-16 (full speed and quality choices approved across [`results/phase6/DESIGN.md`](./DESIGN.md) §7.2)

## Approved Parameters

1. **Scope:** **Option B (`SQ8 + PQ`, `T4.1–T4.9`)** — Implement `SQ8` (`T4.1–T4.3`) followed by `PQ` (`T4.4–T4.5`: $M_{\text{pq}} \in \{16, 32\}$, $K = 256$, $N_{\text{train}} = \min(N, 65{,}536)$, `max_lloyd_iters = 25`, `seed = 42`).
2. **Storage Modes:** Benchmark **both** Memory-First (`Sq8Only`, `Pq16Only`, `Pq32Only` — including streaming two-pass `.fvecs` build & isolated query-serving `PeakWorkingSet64`) and Accuracy-First (`Sq8WithRerank`, `Pq16WithRerank`, `Pq32WithRerank`).
3. **Top-$R$ Rerank Ladder:** `R in {None, 20, 30, 50, 100}` for SQ8 (extending to `R = 200` for PQ) plus `FullBeam` (`R = ef_search`), enforcing `effective_ef = ef_search.max(R).max(k)`.
4. **Build Precision:** Primary **`Float32GraphThenQuantize`** across all configs + **`QuantizedDirect`** ablation at `100k` (plus `1M` `Sq8Only` streaming build if `T4.6` budget permits).
5. **Graph Configs & Scales:** Mandatory **`M=16, efConstruction=100`** (`100k` and `1M`), plus **`M=32, efConstruction=200`** at `100k` (and `1M` `Sq8Only` pending `T4.6` timing audit).
6. **Hard Invariants & Pilot Gate:** `#![forbid(unsafe_code)]`, `std`-only crate, [`src/hnsw/mod.rs`](../../src/hnsw/mod.rs) untouched (`PRESERVING` on `f32` paths), and `T4.6` `100k` pilot check before multi-hour `1M` execution.
