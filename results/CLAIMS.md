# Benchmark Claims Ledger & Verification Mapping (`results/CLAIMS.md`)

This document establishes the authoritative provenance and claim-verification ledger for all quantitative metrics, benchmark tables, and memory/performance claims published in `README.md` and project reports.

Every documented number is mapped to its exact document location, source JSON report path, JSON key path, actual serialized value, generating Git commit, and test environment.

---

## 1. Provenance & Execution Environment

| Metric / Parameter | Documented Value | Doc + Line | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit | Environment |
|---|---|---|---|---|---|---|---|
| CPU Model | `12th Gen Intel(R) Core(TM) i7-12650H` | `README.md:49,62,82` | `results/phase3/phase3_report.json` | `hardware.cpu_model` | `"12th Gen Intel(R) Core(TM) i7-12650H"` | `78e8fcb` | Windows 11 / x86_64 |
| Logical Cores | `16 logical cores` | `README.md:49,62,82` | `results/phase3/phase3_report.json` | `hardware.logical_cores` | `16` | `78e8fcb` | 6 P-cores + 4 E-cores (16 threads) |
| Usable System RAM | `15.8 GB RAM` | `README.md:49,62,82` | `results/phase3/phase3_report.json` | `hardware.total_ram_gb` | `15.78` | `78e8fcb` | Host RAM (16 GB physical) |
| Host Operating System | `windows` | `README.md:49,62,82` | `results/phase3/phase3_report.json` | `hardware.os` | `"windows"` | `78e8fcb` | Windows 11 Home / PowerShell |
| Host Architecture | `x86_64` | `README.md:49,62,82` | `results/phase3/phase3_report.json` | `hardware.arch` | `"x86_64"` | `78e8fcb` | AMD64 / AVX2 supported |
| Hardware Summary String | `12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)` | `README.md:49,62,82` | `results/phase3/phase3_report.json` | `hardware.summary` | `"12th Gen Intel(R) Core(TM) i7-12650H (16 logical cores, 15.8 GB RAM, windows/x86_64)"` | `78e8fcb` | Standardized hardware probe |
| Compiler Profile | `--release` | `README.md:75,115` | `results/phase3_5/LOG.md` | `Section 0` | `opt-level = 3, lto = "thin", codegen-units = 1` | `78e8fcb` | rustc >= 1.88 (`1.99.1`) |

---

## 2. Phase 1 Exact Brute-Force Baseline Claims (`siftsmall` 10k x 128-d)

Documented in `README.md` lines 47–57 from official Phase 1 validation against INRIA Texmex ground truth and `faiss.IndexFlatL2`:

| Metric Name | Documented Value (`README.md`) | Doc + Line | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit | Notes / Status |
|---|---|---|---|---|---|---|---|
| Base Vectors ($N$) | `10,000` | `README.md:47,50` | `results/phase3/phase2_report.json` | `dataset.num_base` | `10000` | `027b2d5` | Authentic `siftsmall` base vectors |
| Query Vectors ($Q$) | `100` | `README.md:50` | `results/phase3/phase2_report.json` | `dataset.num_query` | `100` | `027b2d5` | Authentic `siftsmall` query vectors |
| Vector Dimension ($D$) | `128` | `README.md:47,50` | `results/phase3/phase2_report.json` | `dataset.dim` | `128` | `027b2d5` | 128-d single-precision float32 |
| Query $k$ | `10` | `README.md:54,55` | `results/phase3/phase2_report.json` | `dataset.k` | `10` | `027b2d5` | Top-$k$ nearest neighbors |
| Exact Brute-Force Throughput | `~3,640+ QPS` | `README.md:52` | `results/phase3/phase2_report.json` | `exact_knn_baseline.qps` | `4033.208` | `96ff2a3` (initial) / `027b2d5` (regenerated) | Single-threaded sequential exact search (>3,640 QPS) |
| Exact Latency p50 | `0.250 ms` | `README.md:53` | `results/phase3/phase2_report.json` | `exact_knn_baseline.p50_ms` | `0.243550` | `96ff2a3` (initial) / `027b2d5` (regenerated) | Baseline query latency median |
| Exact Latency p95 | `0.379 ms` | `README.md:53` | `results/phase3/phase2_report.json` | `exact_knn_baseline.p95_ms` | `0.258285` | `96ff2a3` (initial) / `027b2d5` (regenerated) | Baseline 95th percentile latency |
| Exact Latency p99 | `0.520 ms` | `README.md:53` | `results/phase3/phase2_report.json` | `exact_knn_baseline.p99_ms` | `0.287739` | `96ff2a3` (initial) / `027b2d5` (regenerated) | Baseline 99th percentile latency |
| Self-Recall Invariant | `100.00% (1.000000)` | `README.md:54` | `results/phase3/phase2_report.json` | `exact_knn_baseline.self_verified` | `true` | `96ff2a3` / `027b2d5` | `Recall@10(GT, GT) == 1.0` |
| Official Texmex Ground Truth Parity | `100.00% (1000/1000)` | `README.md:55` | `results/phase3/phase2_report.json` | `exact_knn_baseline.official_gt_matched` | `true` | `96ff2a3` / `027b2d5` | Exact match on 100 queries x 10 neighbors |
| FAISS `IndexFlatL2` Parity | `100.00% (100% ID alignment)` | `README.md:56` | Python test harness | `tests/test_rust_parity.py` | `test_rust_siftsmall_vs_faiss_parity: PASSED` | `96ff2a3` | Identical IDs, distances within float32 tol |

---

## 3. Phase 2 HNSW Correctness & Performance Claims (`siftsmall` 10k x 128-d, $M=16$, $efC=100$, $k=10$)

Documented in `README.md` lines 60–76. HNSW construction seed is fixed to `42`.
Source report: `results/phase3/phase2_report.json` regenerated on clean commit `027b2d5`.

| Metric / Parameter | Documented Value (`README.md`) | Doc + Line | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit | Verification / Notes |
|---|---|---|---|---|---|---|---|
| Index Build Time | `1413.55 ms (~1.41 s)` | Phase 2 stdout | `results/phase3/phase2_report.json` | `index_build.build_time_s` | `1.413554` | `027b2d5` | Build throughput: ~7,074 vec/s |
| Max Graph Layer | `3` | Phase 2 stdout | `results/phase3/phase2_report.json` | `index_build.max_layer` | `3` | `027b2d5` | $\lfloor -\ln(U) / \ln(16) \rfloor$ |
| Graph Entry Point | `#3015` | Phase 2 stdout | `results/phase3/phase2_report.json` | `index_build.entry_point` | `3015` | `027b2d5` | Highest layer entry point vector ID |
| `efSearch = 10` Recall@10 | `95.1%` | `README.md:66` | `results/phase3/phase2_report.json` | `ef_search_sweep[0].recall_at_k` | `0.951000` | `027b2d5` | **Exact match** (deterministic) |
| `efSearch = 10` Mean Latency | `0.018 ms` | `README.md:66` | `results/phase3/phase2_report.json` | `ef_search_sweep[0].latency.mean_ms` | `0.018357` | `027b2d5` | **Exact match** (rounds to 0.018 ms) |
| `efSearch = 10` p99 Latency | `0.027 ms` | `README.md:66` | `results/phase3/phase2_report.json` | `ef_search_sweep[0].latency.p99_ms` | `0.027019` | `027b2d5` | **Exact match** (rounds to 0.027 ms) |
| `efSearch = 10` Speedup vs Brute Force | `13.5x` | `README.md:66` | `results/phase3/phase2_report.json` | `ef_search_sweep[0].speedup_vs_exact` | `13.49` | `027b2d5` | **Exact match** (rounds to 13.5x) |
| `efSearch = 20` Recall@10 | `98.6%` | `README.md:67` | `results/phase3/phase2_report.json` | `ef_search_sweep[1].recall_at_k` | `0.986000` | `027b2d5` | **Exact match** (deterministic) |
| `efSearch = 20` Mean Latency | `0.027 ms` | `README.md:67` | `results/phase3/phase2_report.json` | `ef_search_sweep[1].latency.mean_ms` | `0.027308` | `027b2d5` | **Exact match** (rounds to 0.027 ms) |
| `efSearch = 20` p99 Latency | `0.037 ms` | `README.md:67` | `results/phase3/phase2_report.json` | `ef_search_sweep[1].latency.p99_ms` | `0.036759` | `027b2d5` | **Exact match** (rounds to 0.037 ms) |
| `efSearch = 20` Speedup vs Brute Force | `9.1x` | `README.md:67` | `results/phase3/phase2_report.json` | `ef_search_sweep[1].speedup_vs_exact` | `9.07` | `027b2d5` | **Exact match** (rounds to 9.1x) |
| `efSearch = 50` Recall@10 | `100.0%` | `README.md:68` | `results/phase3/phase2_report.json` | `ef_search_sweep[2].recall_at_k` | `1.000000` | `027b2d5` | **Exact match** (deterministic saturation) |
| `efSearch = 50` Mean Latency | `0.057 ms` | `README.md:68` | `results/phase3/phase2_report.json` | `ef_search_sweep[2].latency.mean_ms` | `0.056846` | `027b2d5` | **Exact match** (rounds to 0.057 ms) |
| `efSearch = 50` p99 Latency | `0.082 ms` | `README.md:68` | `results/phase3/phase2_report.json` | `ef_search_sweep[2].latency.p99_ms` | `0.082104` | `027b2d5` | **Exact match** (rounds to 0.082 ms) |
| `efSearch = 50` Speedup vs Brute Force | `4.4x` | `README.md:68` | `results/phase3/phase2_report.json` | `ef_search_sweep[2].speedup_vs_exact` | `4.36` | `027b2d5` | **Exact match** (rounds to 4.4x) |
| `efSearch = 100` Recall@10 | `100.0%` | `README.md:69` | `results/phase3/phase2_report.json` | `ef_search_sweep[3].recall_at_k` | `1.000000` | `027b2d5` | **Exact match** (deterministic saturation) |
| `efSearch = 100` Mean Latency | `0.096 ms` | `README.md:69` | `results/phase3/phase2_report.json` | `ef_search_sweep[3].latency.mean_ms` | `0.096457` | `027b2d5` | **Exact match** (rounds to 0.096 ms) |
| `efSearch = 100` p99 Latency | `0.135 ms` | `README.md:69` | `results/phase3/phase2_report.json` | `ef_search_sweep[3].latency.p99_ms` | `0.134609` | `027b2d5` | **Exact match** (rounds to 0.135 ms) |
| `efSearch = 100` Speedup vs Brute Force | `2.6x` | `README.md:69` | `results/phase3/phase2_report.json` | `ef_search_sweep[3].speedup_vs_exact` | `2.57` | `027b2d5` | **Exact match** (rounds to 2.6x) |
| `efSearch = 200` Recall@10 | `100.0%` | `README.md:70` | `results/phase3/phase2_report.json` | `ef_search_sweep[4].recall_at_k` | `1.000000` | `027b2d5` | **Exact match** (deterministic saturation) |
| `efSearch = 200` Mean Latency | `0.165 ms` | `README.md:70` | `results/phase3/phase2_report.json` | `ef_search_sweep[4].latency.mean_ms` | `0.164986` | `027b2d5` | **Exact match** (rounds to 0.165 ms) |
| `efSearch = 200` p99 Latency | `0.242 ms` | `README.md:70` | `results/phase3/phase2_report.json` | `ef_search_sweep[4].latency.p99_ms` | `0.241539` | `027b2d5` | **Exact match** (rounds to 0.242 ms) |
| `efSearch = 200` Speedup vs Brute Force | `1.5x` | `README.md:70` | `results/phase3/phase2_report.json` | `ef_search_sweep[4].speedup_vs_exact` | `1.50` | `027b2d5` | **Exact match** (rounds to 1.5x) |
| Performance Target Check | Met at `efSearch=20` (Recall $\ge 0.98$ at $< 0.05$ ms) | `README.md:72` | `results/phase3/phase2_report.json` | `verification_status.performance_target_met` | `true` | `027b2d5` | Target met: 0.027 ms < 0.05 ms, Recall 98.6% >= 98% |
| Monotonic Scaling Verified | `true` | Phase 2 stdout | `results/phase3/phase2_report.json` | `verification_status.monotonic_scaling_verified` | `true` | `027b2d5` | Recall non-decreasing with `efSearch` |
| Reproducibility Verified | `true` | Phase 2 stdout | `results/phase3/phase2_report.json` | `verification_status.reproducibility_verified` | `true` | `027b2d5` | Multi-instance bit-exact reproducibility |

---

## 4. Phase 3 SIFT1M Search Sweep Claims (`SIFT1M` 1,000,000 x 128-d, $M=16$, $efC=100$, $k=10$, 5-Run Sweep)

Documented in `README.md` lines 84–90. Source report: `results/phase3/phase3_report.json` (`78e8fcb`).

| `efSearch` | Metric | Documented Value (`README.md`) | Doc + Line | JSON Key Path | Actual JSON Value | Generating Commit | Verification Status |
|:---:|---|---|---|---|---|---|---|
| **10** | Recall@10 | `73.315%` | `README.md:86` | `ef_search_sweep[0].recall_at_10` | `0.733150` | `78e8fcb` | **EXACT MATCH** |
| 10 | Median Mean (ms) | `0.100` | `README.md:86` | `ef_search_sweep[0].latency.mean_ms` | `0.100350` | `78e8fcb` | **EXACT MATCH** (rounds to 0.100) |
| 10 | Mean Spread (ms) | `0.007` | `README.md:86` | `ef_search_sweep[0].latency.mean_ms_spread` | `0.007304` | `78e8fcb` | **EXACT MATCH** (rounds to 0.007) |
| 10 | Median p50 (ms) | `0.097` | `README.md:86` | `ef_search_sweep[0].latency.p50_ms` | `0.097200` | `78e8fcb` | **EXACT MATCH** (rounds to 0.097) |
| 10 | Median p99 (ms) | `0.175` | `README.md:86` | `ef_search_sweep[0].latency.p99_ms` | `0.175203` | `78e8fcb` | **EXACT MATCH** (rounds to 0.175) |
| 10 | p99 Spread (ms) | `0.020` | `README.md:86` | `ef_search_sweep[0].latency.p99_ms_spread` | `0.019804` | `78e8fcb` | **EXACT MATCH** (rounds to 0.020) |
| 10 | Median QPS | `9,942.1` | `README.md:86` | `ef_search_sweep[0].latency.qps` | `9942.10` | `78e8fcb` | **EXACT MATCH** (9942.10) |
| **20** | Recall@10 | `84.949%` | `README.md:87` | `ef_search_sweep[1].recall_at_10` | `0.849490` | `78e8fcb` | **EXACT MATCH** |
| 20 | Median Mean (ms) | `0.153` | `README.md:87` | `ef_search_sweep[1].latency.mean_ms` | `0.153094` | `78e8fcb` | **EXACT MATCH** (rounds to 0.153) |
| 20 | Mean Spread (ms) | `0.003` | `README.md:87` | `ef_search_sweep[1].latency.mean_ms_spread` | `0.003287` | `78e8fcb` | **EXACT MATCH** (rounds to 0.003) |
| 20 | Median p50 (ms) | `0.150` | `README.md:87` | `ef_search_sweep[1].latency.p50_ms` | `0.150300` | `78e8fcb` | **EXACT MATCH** (rounds to 0.150) |
| 20 | Median p99 (ms) | `0.255` | `README.md:87` | `ef_search_sweep[1].latency.p99_ms` | `0.254701` | `78e8fcb` | **EXACT MATCH** (rounds to 0.255) |
| 20 | p99 Spread (ms) | `0.013` | `README.md:87` | `ef_search_sweep[1].latency.p99_ms_spread` | `0.013498` | `78e8fcb` | **EXACT MATCH** (rounds to 0.013) |
| 20 | Median QPS | `6,523.0` | `README.md:87` | `ef_search_sweep[1].latency.qps` | `6523.01` | `78e8fcb` | **EXACT MATCH** (6523.01) |
| **50** | Recall@10 | `94.703%` | `README.md:88` | `ef_search_sweep[2].recall_at_10` | `0.947030` | `78e8fcb` | **EXACT MATCH** |
| 50 | Median Mean (ms) | `0.299` | `README.md:88` | `ef_search_sweep[2].latency.mean_ms` | `0.298723` | `78e8fcb` | **EXACT MATCH** (rounds to 0.299) |
| 50 | Mean Spread (ms) | `0.030` | `README.md:88` | `ef_search_sweep[2].latency.mean_ms_spread` | `0.030087` | `78e8fcb` | **EXACT MATCH** (rounds to 0.030) |
| 50 | Median p50 (ms) | `0.299` | `README.md:88` | `ef_search_sweep[2].latency.p50_ms` | `0.299400` | `78e8fcb` | **EXACT MATCH** (rounds to 0.299) |
| 50 | Median p99 (ms) | `0.477` | `README.md:88` | `ef_search_sweep[2].latency.p99_ms` | `0.476917` | `78e8fcb` | **EXACT MATCH** (rounds to 0.477) |
| 50 | p99 Spread (ms) | `0.270` | `README.md:88` | `ef_search_sweep[2].latency.p99_ms_spread` | `0.270232` | `78e8fcb` | **EXACT MATCH** (rounds to 0.270) |
| 50 | Median QPS | `3,344.9` | `README.md:88` | `ef_search_sweep[2].latency.qps` | `3344.90` | `78e8fcb` | **EXACT MATCH** (3344.90) |
| **100** | Recall@10 | `98.061%` | `README.md:89` | `ef_search_sweep[3].recall_at_10` | `0.980610` | `78e8fcb` | **EXACT MATCH** |
| 100 | Median Mean (ms) | `0.490` | `README.md:89` | `ef_search_sweep[3].latency.mean_ms` | `0.489827` | `78e8fcb` | **EXACT MATCH** (rounds to 0.490) |
| 100 | Mean Spread (ms) | `0.033` | `README.md:89` | `ef_search_sweep[3].latency.mean_ms_spread` | `0.032888` | `78e8fcb` | **EXACT MATCH** (rounds to 0.033) |
| 100 | Median p50 (ms) | `0.501` | `README.md:89` | `ef_search_sweep[3].latency.p50_ms` | `0.501200` | `78e8fcb` | **EXACT MATCH** (rounds to 0.501) |
| 100 | Median p99 (ms) | `0.723` | `README.md:89` | `ef_search_sweep[3].latency.p99_ms` | `0.723405` | `78e8fcb` | **EXACT MATCH** (rounds to 0.723) |
| 100 | p99 Spread (ms) | `0.096` | `README.md:89` | `ef_search_sweep[3].latency.p99_ms_spread` | `0.096101` | `78e8fcb` | **EXACT MATCH** (rounds to 0.096) |
| 100 | Median QPS | `2,040.4` | `README.md:89` | `ef_search_sweep[3].latency.qps` | `2040.44` | `78e8fcb` | **EXACT MATCH** (2040.44) |
| **200** | Recall@10 | `99.376%` | `README.md:90` | `ef_search_sweep[4].recall_at_10` | `0.993760` | `78e8fcb` | **EXACT MATCH** |
| 200 | Median Mean (ms) | `0.828` | `README.md:90` | `ef_search_sweep[4].latency.mean_ms` | `0.827589` | `78e8fcb` | **EXACT MATCH** (rounds to 0.828) |
| 200 | Mean Spread (ms) | `0.035` | `README.md:90` | `ef_search_sweep[4].latency.mean_ms_spread` | `0.034641` | `78e8fcb` | **EXACT MATCH** (rounds to 0.035) |
| 200 | Median p50 (ms) | `0.861` | `README.md:90` | `ef_search_sweep[4].latency.p50_ms` | `0.860700` | `78e8fcb` | **EXACT MATCH** (rounds to 0.861) |
| 200 | Median p99 (ms) | `1.177` | `README.md:90` | `ef_search_sweep[4].latency.p99_ms` | `1.177406` | `78e8fcb` | **EXACT MATCH** (rounds to 1.177) |
| 200 | p99 Spread (ms) | `0.098` | `README.md:90` | `ef_search_sweep[4].latency.p99_ms_spread` | `0.098015` | `78e8fcb` | **EXACT MATCH** (rounds to 0.098) |
| 200 | Median QPS | `1,208.0` | `README.md:90` | `ef_search_sweep[4].latency.qps` | `1207.98` | `78e8fcb` | **EXACT MATCH** (1207.98) |

---

## 5. Phase 3 SIFT1M Build Time & Memory Footprint Claims

Documented in `README.md` lines 92–99. Source reports: `results/phase3/phase3_report.json`, `results/phase3_5/baseline_1m_report.json`, and `results/phase3_5/exp_no_keep_pruned_shrink_1m.json`.

| Metric Name | Documented Value (`README.md`) | Doc + Line | Report JSON Path | JSON Key Path | Actual JSON Value | Generating Commit | Verification Status |
|---|---|---|---|---|---|---|---|
| 1M Build Time | `493.8 s` | `README.md:94` | `results/phase3/phase3_report.json` | `index_build.build_time_s` | `493.782889` | `78e8fcb` | **EXACT MATCH** (rounds to 493.8 s) |
| 1M Build Throughput | `2,025.2 vectors/s` | `README.md:94` | `results/phase3/phase3_report.json` | `index_build.build_qps` | `2025.18` | `78e8fcb` | **EXACT MATCH** (rounds to 2,025.2 vec/s) |
| Top Graph Layer | `top layer 5` | `README.md:94` | `results/phase3/phase3_report.json` | `index_build.max_layer` | `5` | `78e8fcb` | **EXACT MATCH** ($\approx \ln(10^6)/\ln(16) = 4.98$) |
| Graph Entry Point Node | `#829262` | `README.md:94` | `results/phase3/phase3_report.json` | `index_build.entry_point` | `829262` | `78e8fcb` | **EXACT MATCH** |
| Graph Fingerprint | `0x44044886c62bca5e` | `README.md:94` | `results/phase3/phase3_report.json` | `index_build.graph_fingerprint_hex` | `"0x44044886c62bca5e"` | `78e8fcb` | **EXACT MATCH** (`4901122038173518430`) |
| Build Time (`--no-keep-pruned-shrink`) | `415.1 s` | `README.md:94` | `results/phase3_5/exp_no_keep_pruned_shrink_1m.json` | `index_build.build_time_s` | `415.075613` | `366f185` | **EXACT MATCH** (rounds to 415.1 s) |
| Build Throughput (`--no-keep-pruned-shrink`) | `2,409.2 vectors/s` | `README.md:94` | `results/phase3_5/exp_no_keep_pruned_shrink_1m.json` | `index_build.build_qps` | `2409.20` | `366f185` | **EXACT MATCH** (rounds to 2,409.2 vec/s) |
| Raw Vector Bytes per Vector | `512.00 B/vector` | `README.md:95` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.raw_vector_bytes_per_vector` | `512.00` | `78e8fcb` | **EXACT MATCH** ($128 \times 4$ B) |
| Raw Vectors Total Memory | `488.28 MiB` | `README.md:95` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.raw_vector_total_mb` | `488.28` | `78e8fcb` | **EXACT MATCH** ($512\text{ MB} / 1.048576$) |
| Graph-Only Memory per Vector | `148.91 B/vector` | `README.md:96` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.graph_only_bytes_per_vector` | `148.91` | `78e8fcb` | **EXACT MATCH** |
| Graph-Only Total Memory | `142.01 MiB` | `README.md:96` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.graph_only_total_mb` | `142.01` | `78e8fcb` | **EXACT MATCH** |
| Baseline Graph-Only Memory | `338.67 B/vector` | `README.md:96` | `results/phase3_5/baseline_1m_report.json` | `memory_profile.bytes_per_vector_breakdown.graph_only_bytes_per_vector` | `338.67` | `170b9d7` | **EXACT MATCH** |
| Graph Memory Reduction Percentage | `down 56.0%` | `README.md:96` | Computed from reports | `(338.67 - 148.91) / 338.67` | `56.03%` | `d6d36ff` | **EXACT MATCH** (via compact `u32` adjacency) |
| Graph Total Bytes per Vector | `152.91 B/vector` | `README.md:97` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.graph_bytes_per_vector` | `152.91` | `78e8fcb` | **EXACT MATCH** (incl. 4 B/vec visited tag) |
| Graph Total Memory | `145.83 MiB` | `README.md:97` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.graph_total_mb` | `145.83` | `78e8fcb` | **EXACT MATCH** |
| Total Index Footprint per Vector | `664.91 B/vector` | `README.md:98` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.total_bytes_per_vector` | `664.91` | `78e8fcb` | **EXACT MATCH** ($512.00 + 152.91$) |
| Total Index Footprint Memory | `634.11 MiB` | `README.md:98` | `results/phase3/phase3_report.json` | `memory_profile.bytes_per_vector_breakdown.total_footprint_mb` | `634.11` | `78e8fcb` | **EXACT MATCH** |
| Peak Working Set (1M) | `646.05 MiB` | `README.md:99` | `results/phase3/phase3_report.json` | `memory_profile.peak_rss_mb` | `646.05` | `78e8fcb` | **EXACT MATCH** (real OS `PeakWorkingSet64`) |
| Baseline Peak Working Set (1M) | `845.75 MiB` | `README.md:99` | `results/phase3_5/baseline_1m_report.json` | `memory_profile.peak_rss_mb` | `845.75` | `170b9d7` | **EXACT MATCH** |
| Peak Working Set Absolute Drop | `down 199.7 MiB` | `README.md:99` | Computed from reports | `845.75 - 646.05` | `199.70 MiB` | `d6d36ff` | **EXACT MATCH** |

---

## 6. Multi-Scale Baseline Profile Claims (Isolated Process per Scale)

Documented in `README.md` lines 101–108. Source reports: `results/phase3/baseline_profile_report.json` and `results/phase3/phase3_report.json` (`multi_scale_baseline_summary`).

| Scale ($N$) | Metric | Documented Value (`README.md`) | Doc + Line | JSON Key Path | Actual JSON Value | Generating Commit | Verification Status |
|:---:|---|---|---|---|---|---|---|
| **10,000** | Build Time (s) | `1.494` | `README.md:105` | `scale_profiles[0].build_time_s` | `1.493908` | `78e8fcb` | **EXACT MATCH** (rounds to 1.494 s) |
| 10,000 | Build QPS | `6,693.9` | `README.md:105` | `scale_profiles[0].build_qps` | `6693.85` | `78e8fcb` | **EXACT MATCH** (rounds to 6,693.9) |
| 10,000 | Isolated Peak RSS (MiB) | `11.94` | `README.md:105` | `scale_profiles[0].peak_rss_mb` | `11.94` | `78e8fcb` | **EXACT MATCH** |
| 10,000 | Graph-Only (B/vec) | `146.96` | `README.md:105` | `scale_profiles[0].graph_only_bytes_per_vector` | `146.96` | `78e8fcb` | **EXACT MATCH** |
| 10,000 | Graph Total (B/vec) | `150.96` | `README.md:105` | `scale_profiles[0].graph_bytes_per_vector` | `150.96` | `78e8fcb` | **EXACT MATCH** |
| 10,000 | Total (B/vec) | `662.96` | `README.md:105` | `scale_profiles[0].total_bytes_per_vector` | `662.96` | `78e8fcb` | **EXACT MATCH** |
| 10,000 | Recall@10 (`ef=50`) | `99.750%` | `README.md:105` | `scale_profiles[0].recall_at_10` | `0.997500` | `78e8fcb` | **EXACT MATCH** |
| **100,000** | Build Time (s) | `28.564` | `README.md:106` | `scale_profiles[1].build_time_s` | `28.563818` | `78e8fcb` | **EXACT MATCH** (rounds to 28.564 s) |
| 100,000 | Build QPS | `3,500.9` | `README.md:106` | `scale_profiles[1].build_qps` | `3500.93` | `78e8fcb` | **EXACT MATCH** (rounds to 3,500.9) |
| 100,000 | Isolated Peak RSS (MiB) | `69.21` | `README.md:106` | `scale_profiles[1].peak_rss_mb` | `69.21` | `78e8fcb` | **EXACT MATCH** |
| 100,000 | Graph-Only (B/vec) | `145.57` | `README.md:106` | `scale_profiles[1].graph_only_bytes_per_vector` | `145.57` | `78e8fcb` | **EXACT MATCH** |
| 100,000 | Graph Total (B/vec) | `149.57` | `README.md:106` | `scale_profiles[1].graph_bytes_per_vector` | `149.57` | `78e8fcb` | **EXACT MATCH** |
| 100,000 | Total (B/vec) | `661.57` | `README.md:106` | `scale_profiles[1].total_bytes_per_vector` | `661.57` | `78e8fcb` | **EXACT MATCH** |
| 100,000 | Recall@10 (`ef=50`) | `98.140%` | `README.md:106` | `scale_profiles[1].recall_at_10` | `0.981400` | `78e8fcb` | **EXACT MATCH** |
| **1,000,000** | Build Time (s) | `493.783` | `README.md:107` | `scale_profiles[2].build_time_s` | `493.782889` | `78e8fcb` | **EXACT MATCH** (rounds to 493.783 s) |
| 1,000,000 | Build QPS | `2,025.2` | `README.md:107` | `scale_profiles[2].build_qps` | `2025.18` | `78e8fcb` | **EXACT MATCH** (rounds to 2,025.2) |
| 1,000,000 | Isolated Peak RSS (MiB) | `646.05` | `README.md:107` | `scale_profiles[2].peak_rss_mb` | `646.05` | `78e8fcb` | **EXACT MATCH** |
| 1,000,000 | Graph-Only (B/vec) | `148.91` | `README.md:107` | `scale_profiles[2].graph_only_bytes_per_vector` | `148.91` | `78e8fcb` | **EXACT MATCH** |
| 1,000,000 | Graph Total (B/vec) | `152.91` | `README.md:107` | `scale_profiles[2].graph_bytes_per_vector` | `152.91` | `78e8fcb` | **EXACT MATCH** |
| 1,000,000 | Total (B/vec) | `664.91` | `README.md:107` | `scale_profiles[2].total_bytes_per_vector` | `664.91` | `78e8fcb` | **EXACT MATCH** |
| 1,000,000 | Recall@10 (`ef=50`) | `94.703%` | `README.md:107` | `scale_profiles[2].recall_at_10` | `0.947030` | `78e8fcb` | **EXACT MATCH** |

---

## 7. Phase 3 Ground-Truth & Invariant Verification Claims

Documented in `README.md` lines 109–113. Source report: `results/phase3/phase3_report.json`.

| Claim / Note Statement | Doc + Line | Source JSON Path | JSON Key Path | Actual JSON Value | Generating Commit | Verification Method / Rationale |
|---|---|---|---|---|---|---|
| Isolated per-scale process profiling | `README.md:110` | `results/phase3/baseline_profile_report.json` | `scale_profiles[*].measured_this_run` | `[true, true, true]` | `78e8fcb` | Child process loads prefix rows only to avoid memory pollution |
| PeakWorkingSet64 OS RSS measurement | `README.md:110` | `results/phase3/phase3_report.json` | `memory_profile.peak_rss_measurement_method` | `"windows_powershell_peak_working_set_64: powershell -NoProfile -NonInteractive -Command (Get-Process -Id <benchmark process id>).PeakWorkingSet64"` | `78e8fcb` | Real OS working set queried via PowerShell |
| Official Ground Truth Match | `README.md:111` | `results/phase3/phase3_report.json` | `verification_status.official_ground_truth_matched` | `true` | `78e8fcb` | Brute-force `ExactKnn` against full 1M base for first 100 queries |
| Queries Checked Against Official GT | `README.md:111` | `results/phase3/phase3_report.json` | `verification_status.official_gt_check_queries` | `100` | `78e8fcb` | 100 queries $\times k=10 = 1,000$ neighbors |
| ID Alignment vs Official GT | `README.md:111` | `results/phase3/phase3_report.json` | `verification_status.official_gt_recall_at_k` | `0.999000` | `78e8fcb` | 999/1000 ID alignment (1 boundary distance tie) |
| Exact Neighbor Rank Alignment | `README.md:111` | `results/phase3/phase3_report.json` | `verification_status.official_gt_exact_id_matches` | `991` | `78e8fcb` | 991/1000 exact rank matches |
| 100% Top-10 Bounded Distance | `README.md:111` | `results/phase3/phase3_report.json` | `verification_status.official_ground_truth_matched` | `true` | `78e8fcb` | 100% of official top-10 neighbors have distance $\le d_{10}$ |
| Monotonic Recall Scaling | `README.md:112` | `results/phase3/phase3_report.json` | `verification_status.monotonic_scaling_verified` | `true` | `78e8fcb` | Recall non-decreasing across `efSearch` sweep |
| Reproducibility Verified | README / Test Suite | `results/phase3/phase3_report.json` | `verification_status.reproducibility_verified` | `true` | `78e8fcb` | Bit-exact graph construction verified |
| Forbid Unsafe Code Invariant | Codebase Invariant | `results/phase3/phase3_report.json` | `verification_status.forbid_unsafe_code_enforced` | `true` | `78e8fcb` | `#![forbid(unsafe_code)]` compiler enforcement |
| Standard Library Only Invariant | Cargo Invariant | `results/phase3/phase3_report.json` | `verification_status.std_only_enforced` | `true` | `78e8fcb` | Zero external algorithm crates |

---

## 8. Phase 3.5 Profiling Counters Reference (SIFT1M 1M Build)

Documented in `results/phase3_5/LOG.md` and `results/phase3_5/baseline_1m_report.json` from safe `--features instrument` profiling:

| Metric Name | Documented Value | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit |
|---|---|---|---|---|---|
| Total Distance Evaluations | `3,744,913,479` | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.distance_evaluations.total` | `3744913479` | `170b9d7` |
| Distance Evals per Insert (Mean) | `3,744.91` | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.distance_evaluations.per_insert_mean` | `3744.91` | `170b9d7` |
| `search_layer` Distance Evals | `1,715,511,861` (45.8%) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.distance_evaluations.search_layer` | `1715511861` | `170b9d7` |
| `heuristic` Distance Evals | `282,127,479` (7.5%) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.distance_evaluations.heuristic` | `282127479` | `170b9d7` |
| `shrink` Distance Evals | `1,747,274,139` (46.7%) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.distance_evaluations.shrink` | `1747274139` | `170b9d7` |
| `search_layer` Calls | `5,111,179` (5.11/vec) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.call_counts.search_layer_calls` | `5111179` | `170b9d7` |
| `heuristic` Calls | `8,631,013` (8.63/vec) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.call_counts.heuristic_calls` | `8631013` | `170b9d7` |
| `shrink` Calls | `7,564,342` (7.56/vec) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.call_counts.shrink_calls` | `7564342` | `170b9d7` |
| `search_layer` Time Share | `403.01 s` (`78.83%`) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.phase_time_split.search_layer_s` | `403.009306` | `170b9d7` |
| `heuristic` Time Share | `24.56 s` (`4.80%`) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.phase_time_split.heuristic_s` | `24.555598` | `170b9d7` |
| `shrink` Time Share | `72.57 s` (`14.19%`) | `results/phase3_5/baseline_1m_report.json` | `build_instrumentation.phase_time_split.shrink_s` | `72.567086` | `170b9d7` |

---

## 9. Phase 4 Pilot Measurement Claims (100k SIFT1M Prefix, $M=16, \text{efConstruction}=100$, $k=10$)

Documented in `CLAUDE.md`, `continuation.md`, and `results/phase4/PROPOSAL.md`. Source report: `results/phase4/pilot/scale_100000_m16_efc100.json` on clean commit `26a9807`.

| Metric / Parameter | Documented Value | Doc + Line | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit | Verification / Notes |
|---|---|---|---|---|---|---|---|
| Pilot Base Scale ($N$) | `100,000` | `CLAUDE.md:7`, `continuation.md:27`, `PROPOSAL.md:20` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `scale_vectors` | `100000` | `26a9807` | Authentic SIFT1M prefix |
| Pilot Query Count ($Q$) | `10,000` | `PROPOSAL.md:20,32` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `dataset.num_query` | `10000` | `26a9807` | 10k query vectors |
| Pilot Construction Time | `26.066 s` | `CLAUDE.md:7`, `continuation.md:27`, `PROPOSAL.md:22` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `index_build.build_time_s` | `26.066005` | `26a9807` | Single thread, P-core `0x0004` |
| Pilot Build Throughput | `3,836.41 vec/s` | `CLAUDE.md:7`, `PROPOSAL.md:22` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `index_build.build_qps` | `3836.41` | `26a9807` | 100k vectors / 26.066 s |
| Pilot Graph Fingerprint | `0x80af03bd7aebaade` | `PROPOSAL.md:24` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `index_build.graph_fingerprint_hex` | `"0x80af03bd7aebaade"` | `26a9807` | Bit-exact golden match |
| Pilot Graph-Only Memory | `145.57 B/vector` (`13.88 MiB`) | `CLAUDE.md:7`, `PROPOSAL.md:25,197` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `memory_profile.graph_only_bytes_per_vector` | `145.57` | `26a9807` | `13.8827` MiB graph-only total |
| Pilot Peak RSS | `77.61 MiB` | `CLAUDE.md:7`, `PROPOSAL.md:26,197` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `memory_profile.peak_rss_mb` | `77.61` | `26a9807` | Lifetime high-water mark |
| Pilot Peak RSS Scope | `process_lifetime_high_water_mark` | `PROPOSAL.md:26` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `memory_profile.peak_rss_scope` | `"process_lifetime_high_water_mark"` | `26a9807` | Distinguishes process RSS |
| Pilot 12-Point Sweep Time | `173.49 s (2.89 min)` | `CLAUDE.md:7`, `continuation.md:27`, `PROPOSAL.md:27,48` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `ef_search_sweep` | `sum(total_s) * 4 = 173.493 s` | `26a9807` | 1 warmup + 3 timed repetitions |
| Pilot Recall@10 (ef=10) | `0.821030` | `CLAUDE.md:7`, `continuation.md:27`, `PROPOSAL.md:36` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `ef_search_sweep[0].recall_at_k` | `0.821030` | `26a9807` | Throughput: 21,237.13 QPS |
| Pilot Recall@10 (ef=50) | `0.980190` | `PROPOSAL.md:40` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `ef_search_sweep[4].recall_at_k` | `0.980190` | `26a9807` | Throughput: 6,643.81 QPS |
| Pilot Recall@10 (ef=100) | `0.993670` | `PROPOSAL.md:42` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `ef_search_sweep[6].recall_at_k` | `0.993670` | `26a9807` | Throughput: 3,728.92 QPS |
| Pilot Recall@10 (ef=600) | `0.997020` | `CLAUDE.md:7`, `continuation.md:27`, `PROPOSAL.md:47` | `results/phase4/pilot/scale_100000_m16_efc100.json` | `ef_search_sweep[11].recall_at_k` | `0.997020` | `26a9807` | Monotonic peak recall |

---

## 10. Phase 4 Parameter Sweep & 1M Finals Claims (`README.md:197-248`)

Documented in `README.md` lines 197–248. All benchmarks executed single-threaded, pinned to Performance core `0x0004` on a 12th Gen Intel(R) Core(TM) i7-12650H under clean Git commits (`cd48e7f` / `e897092`).

| Metric / Parameter | Documented Value (`README.md`) | Doc + Line | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit | Verification / Notes |
|---|---|---|---|---|---|---|---|
| **100k Sweep Grid Size** | `23 configurations` | `README.md:199` | `results/phase4/summary.json` | `total_configs` | `23` | `e897092` | Cartesian product of $M \times \text{efC}$ with $\text{efC} \ge M$ |
| **100k Sweep Base Scale** | `100,000` | `README.md:199` | `results/phase4/summary.json` | `scale_vectors` | `100000` | `e897092` | Texmex SIFT1M prefix |
| **1M Finals Scale** | `1,000,000` | `README.md:199,213` | `results/phase4/scale_1000000_*.json` | `scale_vectors` | `1000000` | `cd48e7f` | Full authentic Texmex SIFT1M base |
| **1M Finals Query Count** | `10,000` | `README.md:215` | `results/phase4/scale_1000000_*.json` | `dataset.num_query` | `10000` | `cd48e7f` | Standard SIFT query set |
| **1M Finals Repetitions** | `5 timed passes` | `README.md:215` | `results/phase4/scale_1000000_*.json` | `sweep_runs` | `5` | `cd48e7f` | 1 untimed warmup + 5 timed runs |
| **1M Finals Ground Truth** | `official Texmex 100-NN` | `README.md:215` | `results/phase4/scale_1000000_*.json` | `ground_truth_source` | `"official_texmex_sift1m_groundtruth_ivecs"` | `cd48e7f` | `data/sift1m/sift_groundtruth.ivecs` |
| **$M=8, \text{efC}=100$ (1M) Build Time** | `266.541 s` | `README.md:219` | `results/phase4/scale_1000000_m8_efc100.json` | `index_build.build_time_s` | `266.540707` | `cd48e7f` | Build throughput: 3,751.8 vec/s |
| **$M=8, \text{efC}=100$ (1M) Graph Memory** | `85.44 B/vec` (`81.5 MiB`) | `README.md:219,227` | `results/phase4/scale_1000000_m8_efc100.json` | `memory_profile.graph_only_bytes_per_vector` | `85.44` | `cd48e7f` | Graph total: 81.48 MiB |
| **$M=8, \text{efC}=100$ (1M) Peak RSS** | `583.42 MiB` | `README.md:219,227` | `results/phase4/scale_1000000_m8_efc100.json` | `memory_profile.peak_rss_mb` | `583.42` | `cd48e7f` | Lowest RSS among 1M configurations |
| **$M=8, \text{efC}=100$ (1M) Recall (ef=10)** | `0.57222` | `README.md:219` | `results/phase4/scale_1000000_m8_efc100.json` | `ef_search_sweep[0].recall_at_k` | `0.572220` | `cd48e7f` | QPS: 20,150.7, p99: 0.0830 ms |
| **$M=8, \text{efC}=100$ (1M) Recall (ef=50)** | `0.86072` | `README.md:219` | `results/phase4/scale_1000000_m8_efc100.json` | `ef_search_sweep[4].recall_at_k` | `0.860720` | `cd48e7f` | QPS: 6,743.3, p99: 0.2109 ms |
| **$M=8, \text{efC}=100$ (1M) Recall (ef=100)** | `0.93304` | `README.md:219` | `results/phase4/scale_1000000_m8_efc100.json` | `ef_search_sweep[6].recall_at_k` | `0.933040` | `cd48e7f` | QPS: 3,861.6, p99: 0.3390 ms |
| **$M=8, \text{efC}=100$ (1M) Recall (ef=200)** | `0.97330` | `README.md:219` | `results/phase4/scale_1000000_m8_efc100.json` | `ef_search_sweep[8].recall_at_k` | `0.973300` | `cd48e7f` | QPS: 2,071.0, p99: 0.6434 ms |
| **$M=8, \text{efC}=100$ (1M) Recall (ef=600)** | `0.99470` | `README.md:219` | `results/phase4/scale_1000000_m8_efc100.json` | `ef_search_sweep[11].recall_at_k` | `0.994700` | `cd48e7f` | QPS: 781.7, p99: 1.6119 ms |
| **$M=16, \text{efC}=100$ (1M) Build Time** | `450.301 s` | `README.md:220` | `results/phase4/scale_1000000_m16_efc100.json` | `index_build.build_time_s` | `450.301138` | `cd48e7f` | Build throughput: 2,220.7 vec/s |
| **$M=16, \text{efC}=100$ (1M) Graph Memory** | `148.91 B/vec` (`142.0 MiB`) | `README.md:220,231` | `results/phase4/scale_1000000_m16_efc100.json` | `memory_profile.graph_only_bytes_per_vector` | `148.91` | `cd48e7f` | Matches Phase 3 baseline memory exactly |
| **$M=16, \text{efC}=100$ (1M) Peak RSS** | `647.26 MiB` | `README.md:220,231` | `results/phase4/scale_1000000_m16_efc100.json` | `memory_profile.peak_rss_mb` | `647.26` | `cd48e7f` | Process lifetime high-water mark |
| **$M=16, \text{efC}=100$ (1M) Graph Fingerprint** | `0x44044886c62bca5e` | Log / Verification | `results/phase4/scale_1000000_m16_efc100.json` | `index_build.graph_fingerprint_hex` | `"0x44044886c62bca5e"` | `cd48e7f` | **Bit-for-bit identical** to Phase 3.5 baseline |
| **$M=16, \text{efC}=100$ (1M) Recall (ef=10)** | `0.73315` | `README.md:220` | `results/phase4/scale_1000000_m16_efc100.json` | `ef_search_sweep[0].recall_at_k` | `0.733150` | `cd48e7f` | QPS: 12,711.7, p99: 0.1284 ms |
| **$M=16, \text{efC}=100$ (1M) Recall (ef=50)** | `0.94703` | `README.md:220,245` | `results/phase4/scale_1000000_m16_efc100.json` | `ef_search_sweep[4].recall_at_k` | `0.947030` | `cd48e7f` | Exact match with Phase 3 report |
| **$M=16, \text{efC}=100$ (1M) Recall (ef=100)** | `0.98061` | `README.md:220,232` | `results/phase4/scale_1000000_m16_efc100.json` | `ef_search_sweep[6].recall_at_k` | `0.980610` | `cd48e7f` | QPS: 2,383.9, p99: 0.5515 ms |
| **$M=16, \text{efC}=100$ (1M) Recall (ef=200)** | `0.99376` | `README.md:220,232` | `results/phase4/scale_1000000_m16_efc100.json` | `ef_search_sweep[8].recall_at_k` | `0.993760` | `cd48e7f` | QPS: 1,296.0, p99: 1.0673 ms |
| **$M=16, \text{efC}=100$ (1M) Recall (ef=600)** | `0.99804` | `README.md:220` | `results/phase4/scale_1000000_m16_efc100.json` | `ef_search_sweep[11].recall_at_k` | `0.998040` | `cd48e7f` | QPS: 515.5, p99: 2.6370 ms |
| **$M=32, \text{efC}=200$ (1M) Build Time** | `1,288.897 s` | `README.md:221` | `results/phase4/scale_1000000_m32_efc200.json` | `index_build.build_time_s` | `1288.896758` | `cd48e7f` | Build throughput: 775.9 vec/s |
| **$M=32, \text{efC}=200$ (1M) Graph Memory** | `272.33 B/vec` (`259.7 MiB`) | `README.md:221` | `results/phase4/scale_1000000_m32_efc200.json` | `memory_profile.graph_only_bytes_per_vector` | `272.33` | `cd48e7f` | Graph total: 259.71 MiB |
| **$M=32, \text{efC}=200$ (1M) Peak RSS** | `767.40 MiB` | `README.md:221` | `results/phase4/scale_1000000_m32_efc200.json` | `memory_profile.peak_rss_mb` | `767.40` | `cd48e7f` | Lifetime high-water mark |
| **$M=32, \text{efC}=200$ (1M) Recall (ef=10)** | `0.84614` | `README.md:221` | `results/phase4/scale_1000000_m32_efc200.json` | `ef_search_sweep[0].recall_at_k` | `0.846140` | `cd48e7f` | QPS: 7,953.7, p99: 0.2041 ms |
| **$M=32, \text{efC}=200$ (1M) Recall (ef=50)** | `0.98278` | `README.md:221,235` | `results/phase4/scale_1000000_m32_efc200.json` | `ef_search_sweep[4].recall_at_k` | `0.982780` | `cd48e7f` | QPS: 2,579.7, p99: 0.5304 ms |
| **$M=32, \text{efC}=200$ (1M) Recall (ef=100)** | `0.99572` | `README.md:221,235` | `results/phase4/scale_1000000_m32_efc200.json` | `ef_search_sweep[6].recall_at_k` | `0.995720` | `cd48e7f` | QPS: 1,476.8, p99: 0.9404 ms |
| **$M=32, \text{efC}=200$ (1M) Recall (ef=200)** | `0.99881` | `README.md:221,235` | `results/phase4/scale_1000000_m32_efc200.json` | `ef_search_sweep[8].recall_at_k` | `0.998810` | `cd48e7f` | QPS: 842.2, p99: 1.6854 ms |
| **$M=32, \text{efC}=200$ (1M) Recall (ef=600)** | `0.99944` | `README.md:221` | `results/phase4/scale_1000000_m32_efc200.json` | `ef_search_sweep[11].recall_at_k` | `0.999440` | `cd48e7f` | QPS: 348.6, p99: 4.2463 ms |
| **$M=48, \text{efC}=400$ (1M) Build Time** | `2,827.980 s` | `README.md:222` | `results/phase4/scale_1000000_m48_efc400.json` | `index_build.build_time_s` | `2827.980183` | `cd48e7f` | Build throughput: 353.6 vec/s |
| **$M=48, \text{efC}=400$ (1M) Graph Memory** | `402.42 B/vec` (`383.8 MiB`) | `README.md:222` | `results/phase4/scale_1000000_m48_efc400.json` | `memory_profile.graph_only_bytes_per_vector` | `402.42` | `cd48e7f` | Graph total: 383.78 MiB |
| **$M=48, \text{efC}=400$ (1M) Peak RSS** | `889.69 MiB` | `README.md:222` | `results/phase4/scale_1000000_m48_efc400.json` | `memory_profile.peak_rss_mb` | `889.69` | `cd48e7f` | Safe: 5.6% of 15.78 GB host RAM |
| **$M=48, \text{efC}=400$ (1M) Recall (ef=10)** | `0.89372` | `README.md:222` | `results/phase4/scale_1000000_m48_efc400.json` | `ef_search_sweep[0].recall_at_k` | `0.893720` | `cd48e7f` | QPS: 5,905.2, p99: 0.2685 ms |
| **$M=48, \text{efC}=400$ (1M) Recall (ef=50)** | `0.99165` | `README.md:222,239` | `results/phase4/scale_1000000_m48_efc400.json` | `ef_search_sweep[4].recall_at_k` | `0.991650` | `cd48e7f` | QPS: 1,918.8, p99: 0.7254 ms |
| **$M=48, \text{efC}=400$ (1M) Recall (ef=100)** | `0.99803` | `README.md:222` | `results/phase4/scale_1000000_m48_efc400.json` | `ef_search_sweep[6].recall_at_k` | `0.998030` | `cd48e7f` | QPS: 1,110.1, p99: 1.2947 ms |
| **$M=48, \text{efC}=400$ (1M) Recall (ef=200)** | `0.99922` | `README.md:222` | `results/phase4/scale_1000000_m48_efc400.json` | `ef_search_sweep[8].recall_at_k` | `0.999220` | `cd48e7f` | QPS: 643.9, p99: 2.3104 ms |
| **$M=48, \text{efC}=400$ (1M) Recall (ef=600)** | `0.99944` | `README.md:222,239` | `results/phase4/scale_1000000_m48_efc400.json` | `ef_search_sweep[11].recall_at_k` | `0.999440` | `cd48e7f` | Accuracy saturation point |

---

## 11. Phase 5 Filtered Search & Crossover Claims (`T3.6`–`T3.11`, Option C Matrix + Sensitivity)

Documented in [`results/phase5/FINDINGS.md`](phase5/FINDINGS.md) and [`README.md`](../README.md). Executed on authentic INRIA Texmex `SIFT1M` ($M=16, \text{efConstruction}=100, \text{seed}=42, k=10, \text{reps}=5$) under clean commits (`84e3c81` for `T3.6`/`T3.7`, `a4621bc` for `100k` matrix + sensitivity on solo P-core `0x0004`, `c39c92b` for `1M` matrix + sensitivity across P-cores `0x0004,0x0010,0x0040,0x0100`). Every value traces directly to a repository JSON artifact with `"git_dirty": false`.

| Metric / Claim | Documented Value | Doc Reference | Source Report JSON | JSON Key Path | Actual JSON Value | Generating Commit | Verification / Notes |
|---|---|---|---|---|---|---|---|
| **T3.6 Cost Model Constants** | `t_word=0.3042ns`, `t_cand=23.878ns` | `FINDINGS.md` §3, `README.md` | `results/phase5/pilot/microbench_cost.json` | `measurements.{t_word_ns,t_cand_ns}` | `0.3042`, `23.878` | `84e3c81` | L1-hot primitive timings (`dim=128, k=10`) |
| **100k Matrix Graph Fingerprint** | `0x80af03bd7aebaade` | `FINDINGS.md` §1 | `results/phase5/cell_n100000_independent_s0.000500.json` | `index_build.graph_fingerprint_hex` | `"0x80af03bd7aebaade"` | `a4621bc` | Bit-exact Phase 4 100k M=16 parity (`27.365 s` build) |
| **100k Mode A Crossover (`R>=0.95` & `0.99`)** | $\sigma^* = 0.079149$ (`[0.075, 0.10]`) | `FINDINGS.md` §2, `README.md` | `results/phase5/summary_n100000.json` | `crossovers_primary_0_95[0].crossover_sigma_loglog` | `0.079149` | `a4621bc` | `S2_ef20` (`[lo,hi]=[0.075,0.1]`); identical for `0.99` |
| **100k Mode B Crossover (`R>=0.95`)** | $\sigma^* = 0.252372$ (`[0.20, 0.50]`) | `FINDINGS.md` §2, `README.md` | `results/phase5/summary_n100000.json` | `crossovers_primary_0_95[1].crossover_sigma_loglog` | `0.252372` | `a4621bc` | `S2 ef=400 -> ef=50` (`0.99`: bracket `[0.20,0.50]`, `ef=100` at `hi`) |
| **1M Matrix Graph Fingerprint & RSS** | `0x44044886c62bca5e`, `1,015.69 MiB` | `FINDINGS.md` §1 | `results/phase5/cell_n1000000_independent_s0.001000.json` | `index_build.{graph_fingerprint_hex,peak_rss_mb}` | `"0x44044886c62bca5e"`, `1015.69` | `c39c92b` | Bit-exact Phase 3/4 1M M=16 parity (`606.488 s` build) |
| **1M Mode A Crossover (`R>=0.95` & `0.99`)** | $\sigma^* = 0.023799$ (`[0.020, 0.035]`) | `FINDINGS.md` §2, `README.md` | `results/phase5/summary_n1000000.json` | `crossovers_primary_0_95[0].crossover_sigma_loglog` | `0.023799` | `c39c92b` | `3.33x` leftward shift vs 100k; `S2_ef20` at both brackets |
| **1M Mode B Crossover (`R>=0.95`)** | $\sigma^* = 0.120868$ (`[0.10, 0.20]`) | `FINDINGS.md` §2, `README.md` | `results/phase5/summary_n1000000.json` | `crossovers_primary_0_95[1].crossover_sigma_loglog` | `0.120868` | `c39c92b` | `2.09x` leftward shift vs 100k (`S2 ef=400 -> 100`) |
| **1M Mode B Crossover (`R>=0.99`)** | Bracket `[0.10, 0.20]` (`ef=400` at `0.20`) | `FINDINGS.md` §2, `README.md` | `results/phase5/summary_n1000000.json` | `crossovers_secondary_0_99[1]` | `bracket=[0.10,0.20], hi_ef=400, null` | `c39c92b` | `target_not_reached_at_lower_bracket` (`0.10`) |
| **1M Mode A `S3` vs `S2_ef20` (`sigma=0.035`)** | `S3` `55.38 QPS` vs `S2_ef20` `92.37 QPS` | `FINDINGS.md` §2, §4.3 | `results/phase5/cell_n1000000_independent_s0.035000.json` | `arms[0].qps`, `arms[1].{qps,recall_at_k}` | `55.38`, `92.37` (`0.990700`) | `c39c92b` | Post-crossover `S2` win at `1M` `0.035` (`0x0004`) |
| **1M Mode A `S3` (`sigma=0.50`) Speedup** | `S2_ef50` `1,328.34 QPS` vs `S3` `15.84 QPS` | `FINDINGS.md` §2, §4.3 | `results/phase5/cell_n1000000_independent_s0.500000.json` | `arms[2].qps`, `arms[0].qps` | `1328.34`, `15.84` | `c39c92b` | `83.9x` speedup at `R@10=0.969050` (`0x0010`) |
| **1M Mode B `S1_uncap` vs `S3` (`sigma=0.05`)** | `S1u` `5.45 QPS` (`183.4ms`) vs `S3` `131.50 QPS` | `FINDINGS.md` §4.4, §5 | `results/phase5/cell_n1000000_correlated_s0.050000.json` | `arms[10].{qps,mean_ms,recall_at_k}`, `arms[0].qps` | `5.45`, `183.413132`, `0.993800`, `131.50` | `c39c92b` | `S2_ef600` max `0.7860`, `S1_cap4096` `0.4482` |
| **1M Opaque Guardrail (Ind `0.10`, `limit=16384`)** | `R@10=0.998500`, `1.0%` hit (vs `69.0%` at `10k`) | `FINDINGS.md` §6, `README.md` | `results/phase5/sensitivity_n1000000.json` | `opaque_eval_probes[2,4].{recall_at_k,eval_limit_reached_rate}` | `0.998500`, `0.010000` (`0.690000` at `10k`) | `c39c92b` | Preserves recall of `None` (`0.998500`, max `16,503`) |
| **1M Opaque Tail Bound (Corr `0.01`, `limit=16384`)** | `p99` `53.729ms -> 4.797ms` (`11.20x` cap) | `FINDINGS.md` §6, `README.md` | `results/phase5/sensitivity_n1000000.json` | `opaque_eval_probes[6,8].{p99_ms,eval_count_max}` | `53.728891` (`max=196017`) -> `4.797383` (`16384`) | `c39c92b` | Bounds runaway opaque closure evaluations |
| **1M Bridge-Hop Sensitivity (`H_max`, Ind `0.001`)** | `ef=100`: `0.3000 -> 0.8046 -> 0.9920` | `FINDINGS.md` §5 | `results/phase5/sensitivity_n1000000.json` | `sensitivity_arms[6..8].arm.{recall_at_k,mean_ms}` | `0.3000 -> 0.8046 -> 0.9920` (`11.8 -> 121.8ms`) | `c39c92b` | Mode B `0.001` capped at `0.2548` even at `10*ef` |



