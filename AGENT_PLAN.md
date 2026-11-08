# AGENT_PLAN.md: framework and next steps for coding agents

Read order for any agent: `continuation.md` (current state), then this file, then `CLAUDE.md`, `CONTEXT_AND_MEMORY.md`, `docs/HNSW_ALGORITHM.md`.

Scope of this file: it replaces only the "How to move forward" section of `continuation.md`. Everything else in `continuation.md` (working agreement, review log, design decisions, known limitations) stays in force. When you finish a task, update `continuation.md` current-state sections; do not copy status into this file except to tick boxes.

Project: from-scratch HNSW in Rust (`#![forbid(unsafe_code)]`, std only), measured on SIFT1M, compared against FAISS in the end. Dev designs and decides. Agents implement, measure and report.

---

## 1. Ground rules

Inherited, unchanged:
1. Dev designs, agents code. Do not start a milestone before the previous gate is signed by Dev.
2. Free tools and datasets only. If a task seems to need a paid tool, stop and ask.
3. Never claim a number you did not measure. Every number in a doc must trace to a JSON file produced by a real run.
4. Never weaken a pass condition to get green. A missed target is reported as missed.
5. A check that did not run is never reported as passed.
6. `#![forbid(unsafe_code)]` and std only for the library. Dev tooling (Python scripts, optional Cargo features) may use more, but nothing new in the library's default dependencies.

New:
7. **Every change is classified before it is written** as one of:
   - `PRESERVING`: must not change any query result or the graph. Proven by an equal graph fingerprint (see T1.2) plus bit-identical search output on a fixed query set.
   - `GRAPH-CHANGING`: changes the graph or results. Needs a recall-vs-latency comparison at matched recall and a Dev gate before it becomes a default.
   - `MEASUREMENT-ONLY`: adds instrumentation or reporting, no effect on results when the feature is off.
8. **One change per commit.** Message format: `<class>: <what> (T<id>)`, for example `preserving: replace HashSet dedup with generation tags (T1.4)`.
9. **Provenance in every report.** Each JSON report records `git_commit`, `git_dirty`, hardware, dataset source, config, and `measured_this_run` per row. A report with `git_dirty: true` is never cited in the README.
10. **No hardcoded results.** No literal measured value in source except in tests that assert a documented invariant.
11. **Stop and ask** (write the question into `continuation.md` under "Open questions for Dev" and halt) when any of these happens:
    - a test fails and the cause is not obvious in 15 minutes of reading;
    - a `PRESERVING` change shows a different fingerprint;
    - a pass target is missed;
    - the spec in `docs/HNSW_ALGORITHM.md` is ambiguous or seems wrong;
    - a run exceeds its time budget by more than 25 percent;
    - a needed tool is paid or unavailable.

---

## 2. Roles and the work loop

Use separate agent sessions per role. A reviewer must start with fresh context and must not be the agent that wrote the change.

| Role | Does | Must not |
|---|---|---|
| Implementer | Writes code and tests for one task card | Run the final benchmark numbers that go into docs; edit pass thresholds |
| Reviewer | Reads the diff against the task card, re-runs the full gate, checks every claim in the PR description against a file | Fix the code itself (reports findings only) |
| Benchmarker | Runs the agreed protocol (section 4) on a clean commit and writes the JSON | Change source; run on a dirty tree |

Loop per task:
1. Implementer reads the task card, restates the acceptance criteria in its own words in the PR description.
2. Implementer writes the failing test first when the task is `PRESERVING` or fixes a bug.
3. Implementer runs the **gate** (below) and attaches the output tail.
4. Reviewer runs the gate independently and audits against the checklist.
5. Benchmarker runs only if the task needs numbers.
6. Implementer updates `continuation.md` (state, results, open questions). Dev reviews at the gates in section 6.

**Gate (run before every commit, in this order):**
```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo build --release --bin phase1
python -m pytest tests -q -rs && ruff check . && mypy src tests
```
Parity tests must report 0 skipped. If `-rs` shows a skip of `test_rust_parity.py`, the gate failed.

**Reviewer checklist:**
- Is the change class stated, and does the evidence match the class?
- Does the diff touch anything outside the task card's file list? If yes, why?
- Any threshold, tolerance, `#[ignore]`, or `skip` added or loosened?
- Any new constant that looks like a measured result?
- Does every sentence with a number in README, CLAUDE.md, `continuation.md` match a JSON field? List them in the claim ledger (section 3).
- Test count in docs equals `cargo test` and `pytest` output.

---

## 3. Evidence standard

**Task card template** (copy per task into the PR description):
```
Task: T<id> <title>
Class: PRESERVING | GRAPH-CHANGING | MEASUREMENT-ONLY
Files allowed to change: ...
Goal (one sentence):
Acceptance criteria (checkable):
Evidence to attach: ...
Stop conditions specific to this task: ...
```

**Claim ledger**: `results/CLAIMS.md`, one row per number that appears in any doc.
```
| Claim (as written) | Doc + line | Source file | JSON path | Commit |
```
The reviewer rejects any doc number missing from the ledger.

**Fixed query set for equality checks**: the first 1,000 queries of `sift_query.fvecs` at `k=10`, `efSearch=100`. For `PRESERVING` changes, results (ids and distances, bit for bit) on this set at 10k and 100k must equal the pre-change results.

**Results directory**: commit reports under `results/` (not `data/`, which stays gitignored for corpora). Layout:
```
results/
  CLAIMS.md
  phase3/   phase3_report.json  baseline_profile_report.json  phase2_report.json
  phase3_5/ LOG.md  baseline_*.json  exp_*.json
  phase4/   <one JSON per index config>  summary.json  *.png  *.svg
```

---

## 4. Benchmark protocol (applies to every number that goes in a doc)

Dev's machine is a hybrid-core laptop CPU (i7-12650H, 6 P-cores + 4 E-cores, 16 logical threads), which makes timing noisy unless controlled.

Before a run, the benchmarker checks and records in the JSON `environment` block:
- plugged in, Windows power mode set to best performance;
- no heavy background apps (browser, IDE indexers, OneDrive sync paused; the repo currently lives in OneDrive, so run from a non-synced copy or pause sync);
- process pinned to one P-core for single-threaded latency runs (for example `start /affinity` with a P-core mask, or `Start-Process` plus `ProcessorAffinity` in PowerShell; record the mask). If pinning is impossible, record that and treat results as lower confidence;
- release build with the profile in `Cargo.toml` (thin LTO, `codegen-units = 1`); never the fast test-only overrides;
- clean tree (`git_dirty: false`).

Measurement rules:
- One untimed warmup pass over all queries, then timed passes.
- Latency is per query, single-threaded, `k=10`.
- Repetitions: 5 for final published numbers, 3 for wide sweeps. Report median plus min and max.
- "Noise floor" for a given benchmark = observed spread (max minus min) of the baseline runs. A change counts as an improvement only if the median moves by more than 2x that spread and at least 5 percent.
- Build time: median of 3 builds at 100k; 1 build at 1M (record the one run, do not average).
- Ground truth: official `sift_groundtruth.ivecs` is valid only for the full 1M base. For any prefix scale (10k, 100k) compute brute-force ground truth with `ExactKnn` on that prefix and cache it as `data/gt_<scale>_k100.ivecs` (gitignored); record how it was produced.

---

## 5. Milestones

Status boxes: tick when merged and gate passed.

### M0. Close Phase 3 (small, do first)

- [x] **T0.1 Line endings.** Class: MEASUREMENT-ONLY. Add `.gitattributes` (provided in `refine_small.patch`), run `git add --renormalize .`, confirm `git status` is clean and `git diff --ignore-cr-at-eol` is empty. Do not commit a whole-file rewrite of `hnsw.rs`, `README.md`, `CLAUDE.md`, `CONTEXT_AND_MEMORY.md`.
- [x] **T0.2 RSS label.** Apply the two string fixes in `benchmark.rs` (`$PID` label). Add one README line explaining why the process shells out to PowerShell (`forbid(unsafe_code)` rules out `GetProcessMemoryInfo`).
- [x] **T0.3 Commit reports.** Move `phase2_report.json`, `phase3_report.json`, `baseline_profile_report.json` into `results/phase3/`, un-ignore them (`!results/**/*.json`), and update README source lines to the new paths. Verify each README Phase 2 and Phase 3 number against the JSON and start `results/CLAIMS.md`.
- [x] **T0.4 Provenance fields.** Class: MEASUREMENT-ONLY. Add `git_commit`, `git_dirty`, `rustc_version` and an `environment` block (power mode if detectable, affinity if set) to all report writers. Fall back to `"unknown"` when git is not available. Tests: report JSON contains the fields; fallback path covered.
- [x] **T0.5 Docs hygiene.** Remove the personal Windows path from `CONTEXT_AND_MEMORY.md`; delete its stale "Phase 2 Implementation Guide" section (Phase 2 is done; keep the acceptance results as one line); fix "513-line" to the real line count or remove the number; fix the "Phase 2" wording in `CLAUDE.md`. Dev decides whether agent handoff files stay in the public repo (open question, section 7).
- [x] **T0.6 CI strictness.** In `.github/workflows/ci.yml` run pytest with `-rs` and fail if any parity test is skipped. Confirm CI is green on a fresh clone after pushing.
- [x] **Gate G0: Dev signs off Phase 3.**

### M1. Profile first, then speed up the 1M build (call it Phase 3.5)

Goal: cut build time and graph memory without changing recall, then decide on any graph-changing experiment with data. Baseline: 471.0 s for 1M, 338.67 B/vector graph-only, recall@10 94.703% at efSearch 50 and 98.061% at 100.

Fast loop: do all iteration at 100k (about 29 s per build on the current code). Only confirm at 1M at the end of the milestone.

- [x] **T1.1 Instrumentation (feature `instrument`).** Class: MEASUREMENT-ONLY. Behind a Cargo feature that is off by default and costs nothing when off, count and report per build: distance evaluations total and per insert, `search_layer` calls, heuristic calls, shrink calls (reverse-edge prunes), and time split between `search_layer`, heuristic, and shrink using `Instant` at phase level (not per distance call). Report in the JSON. No `unsafe`.
- [x] **T1.2 Graph fingerprint.** Class: MEASUREMENT-ONLY. Add `HnswIndex::graph_fingerprint() -> u64`: FNV-1a 64 over node count, then per node `max_layer` and each layer's neighbor ids in stored order. Define it on logical content so it stays comparable after the storage layout changes in T1.7. Add the fingerprint to Phase 2/3 reports and baseline profile rows. Tests: same seed gives same fingerprint; changing the seed or one edge changes it.
- [x] **T1.3 Baseline freeze.** Benchmarker records, on the current code, at 10k and 100k: fingerprint, build time (median of 3), instrument counters, time split, recall@10 at efSearch 50 and 100, query results hash on the fixed query set, peak RSS in an isolated process. Save as `results/phase3_5/baseline_{10k,100k}.json`. All later `PRESERVING` changes are compared to these.
- [x] **T1.4 Profile.** Using the T1.1 counters and, if it installs and works, a free sampling profiler (candidates: `samply`, Intel VTune). Verify the tool works on this machine before relying on it. Deliverable: `results/phase3_5/LOG.md` section "Profile" listing the top costs by share of build time, with the evidence for each. The profile decides the order of T1.5 to T1.8. Do not optimize something the profile does not show.
- [x] **T1.5 Replace the `HashSet<usize>` dedup in the neighbor heuristic.** Class: PRESERVING. Use a generation-tagged `Vec<u32>` (same idea as `VisitedTracker`). Only matters when `extend_candidates` is true, so also check whether the default path pays for `seen.insert` at all; if yes, that is where the win is. Acceptance: fingerprint equal at 10k and 100k, fixed-query results bit-identical, build time improvement per the noise rule, or revert and log "no gain".
- [x] **T1.6 Reuse search scratch.** Class: PRESERVING. `search_layer` allocates two `BinaryHeap`s per call. Move them into a reusable `SearchScratch` passed by `&mut`. Keep `search_knn_with_tracker(&self, ...)` working with a per-thread scratch. Same acceptance as T1.5.
- [x] **T1.7 Compact adjacency: `u32` ids, flat storage.** Class: PRESERVING.
  - Layer 0: one `Vec<u32>` with fixed stride `m_max0 + 1` per node (first slot is the degree).
  - Upper layers: sparse (about 1 in 16 nodes), so use an arena `Vec<u32>` with a per-node offset, not a stride for every node.
  - Row ids as `u32` (error if `rows > u32::MAX`).
  - Public accessors such as `neighbors(layer, node)` change to `&[u32]`; update tests mechanically, never loosen assertions.
  - Remove the redundant `HnswNode.id` field.
  - Hypothesis, not a claim: layer-0 edges drop from 264 B to about 132 B per vector and total graph to roughly 140 B per vector. Measure it and report the real number.
  - Acceptance: fingerprint equal (it is defined on logical content), fixed-query results bit-identical, `graph_only_bytes_per_vector` reported before and after, build and query time not worse beyond noise.
- [ ] **T1.8 Optional, only if the profile shows distance computation dominates.** Class: PRESERVING. Make the metric a compile-time generic (or function pointer chosen once) so `compute_distance` does not branch on `Metric` per call. Check the 8-way unrolled L2 kernel for bounds-check overhead using `as_chunks` and fixed-size array patterns (safe Rust only). Same acceptance as T1.5. Skip if the profile does not support it.
- [x] **T1.9 Graph-changing experiment: no backfill when shrinking (`keep_pruned = false` in the shrink step only).** Class: GRAPH-CHANGING. Add a config field (for example `keep_pruned_shrink`) defaulting to the current behavior, so Phase 3 results stay reproducible. Measure at 100k, then 1M if promising: build time, recall-vs-latency over the efSearch sweep (compare at matched recall, not at equal efSearch), average degree, memory. Report honestly if recall at equal latency gets worse. Do not change the default. Hand the data to Dev.
- [x] **Gate G1 (Dev):** review `LOG.md`. Decide whether T1.9 becomes the default. If yes, regenerate the Phase 2/3 reports and README tables from a fresh run and keep the old ones under `results/phase3/legacy_keep_pruned/`.
- [x] **T1.10 Final 1M confirmation.** Benchmarker runs the Phase 3 command on the final commit (protocol in section 4) and regenerates the Phase 3 JSON and README table. Compare with the 471.0 s baseline and report the real ratio. If any `PRESERVING` change altered recall at all, that is a bug; find it.

Memory side effect to keep in mind: Phase 6 (quantization) and Phase 7 (FAISS comparison) both benefit from the compact layout, so T1.7 should land before them.

### M2. Phase 4: parameter sweep and Recall@10 vs p99 plot

Do not start until G1 is signed.

- [x] **T2.1 Propose the grid and a time budget (no runs).** Agent writes `results/phase4/PROPOSAL.md` with the grid, a pilot-based time estimate, and disk use. Starting proposal for Dev to edit:
  - 100k wide sweep: M in {8, 12, 16, 24, 32, 48} and efConstruction in {40, 100, 200, 400}, 24 indexes, subject to `efConstruction >= M`.
  - efSearch for every index: {10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600}, k=10, 10,000 queries, 3 repetitions.
  - 1M final: 3 to 4 configs picked from the 100k Pareto frontier (include M=16, efC=100 as the Phase 3 reference), 5 repetitions, one build each.
  - The agent must run one pilot build (M=16, efC=100 at 100k) and one pilot sweep, then extrapolate time. Build time grows with both M and efC, so the biggest configs dominate; give a worst-case figure. My rough guess before measuring is a few hours for the 100k grid and under an hour per 1M build at the larger settings, which is exactly why this needs a budget before anything runs.
- [x] **Gate G2 (Dev):** approves grid and budget (23 configurations, full 12-point ladder, ~95 min budget approved).
- [x] **T2.2 Sweep runner.** Class: MEASUREMENT-ONLY. A `phase4` binary: reads a grid file, builds each index, runs the efSearch sweep, writes one JSON per index (same conventions as Phase 3: hardware, dataset source, config, fingerprint, provenance, `measured_this_run`). Must be resumable (skip configs whose JSON exists and matches the current commit) so a crash or a laptop sleep does not lose hours. Prefix ground truth cached per section 4. Tests on small synthetic data for grid parsing, resume logic, and JSON schema.
- [ ] **T2.3 Run the 100k sweep** (Benchmarker, protocol in section 4, clean commit).
- [ ] **T2.4 Plot script.** `scripts/plot_phase4.py`, matplotlib only, deterministic, writes PNG and SVG into `results/phase4/`. Charts:
  1. Recall@10 (x) vs p99 latency in ms (y, log scale), one line per (M, efC), Pareto frontier drawn explicitly. Zoom x to the interesting recall range (for example 0.90 to 1.00).
  2. Same with QPS (single thread) instead of p99.
  3. Build time vs best achievable recall at a fixed latency budget, per config.
  4. Graph memory per vector vs M.
  Plain, readable styling: labeled axes with units, legend, no decoration. Each chart's caption states hardware, dataset, scale, repetitions.
- [ ] **T2.5 Run the 1M finals** (Benchmarker) and add them to chart 1 as a separate marker style.
- [ ] **T2.6 Write-up.** README section with the frontier, the recommended operating points, and what changed with scale. Every number goes in `CLAIMS.md`.
- [ ] **Gate G3 (Dev):** signs off Phase 4.

### M3. Phase 5: filtered search (design brief; expand after G3)

Agent produces `results/phase5/DESIGN.md` first; no code before Dev approves it. The design must cover:
- Metadata: synthetic attributes, generated deterministically from the seed. Two modes: independent of vector position, and correlated with vector clusters (the hard case, because matching nodes are far apart in the graph). Selectivities to test: 50%, 10%, 1%, 0.1%.
- Strategies to compare: (a) post-filter with over-fetch, (b) in-traversal predicate (traverse non-matching nodes for connectivity, only admit matching ones to the result set), (c) pre-filter with brute force over the matching ids when selectivity is low. Find the selectivity where the fallback beats traversal, and report it.
- Ground truth: filtered brute force per predicate.
- Metrics: Recall@10 vs selectivity, latency vs selectivity, and the failure mode where the traversal returns fewer than k results.
- Gate G4 (Dev): approves the design.

### M4. Phase 6: quantization (design brief; expand after G4)

- SQ8 first: per-dimension min and max, `u8` codes, asymmetric distance (float query vs `u8` code) in safe Rust, optional rerank of top-R with full-precision vectors. Report recall cost, memory per vector, latency.
- PQ as a stretch goal: deterministic k-means (own implementation, seeded), 8-bit codes with 16 subquantizers as the starting point (16 B per vector), ADC lookup tables. Compare against SQ8 on the same recall-vs-latency chart.
- Constraint: the `PRESERVING` guarantees from M1 do not apply here, so each variant is `GRAPH-CHANGING` or a new index type. Keep float32 paths intact and tested.
- Gate G5 (Dev): approves the design.

### M5. Phase 7: head-to-head with FAISS (design brief; expand after G5)

Protocol decisions Dev must confirm before any run:
- Same machine, same dataset, same k, same query set, same recall definition.
- FAISS `HNSWFlat`, same M and efConstruction. FAISS builds with OpenMP by default, so run it with 1 thread for the single-thread comparison and, separately, report multi-thread numbers for both sides if the Rust side has a parallel build by then.
- Match efSearch semantics and compare at matched recall (curves), not at equal efSearch.
- Record FAISS version, how it was installed (Windows wheel vs built), CPU instruction set in use, and the `keep_pruned` setting on the Rust side, since the graph structures will differ.
- Report where Rust loses as plainly as where it wins.
- Gate G6 (Dev): approves the protocol.

---

## 6. Gates summary (what Dev decides)

| Gate | When | Dev decides |
|---|---|---|
| G0 | After M0 | Phase 3 signed off; whether handoff files stay in the public repo |
| G1 | After T1.9 data | Whether no-backfill-on-shrink becomes the default; whether to regenerate Phase 3 numbers |
| G2 | After T2.1 | Phase 4 grid and time budget |
| G3 | After T2.6 | Phase 4 signed off |
| G4 | After Phase 5 design | Filter strategies, selectivities, correlated-metadata mode |
| G5 | After Phase 6 design | SQ8 only, or SQ8 plus PQ |
| G6 | After Phase 7 protocol | FAISS comparison protocol |

---

## 7. Open questions for Dev

1. Do `CLAUDE.md`, `CONTEXT_AND_MEMORY.md` and `continuation.md` stay in the public repo, or move to a private branch or `docs/dev/`? They currently state the agents-code, Dev-designs workflow openly.
2. Is a parallel (multi-threaded) build in scope? It would help Phase 4 and 7 wall-clock but breaks bit-exact reproducibility unless designed carefully. Default in this plan: out of scope.
3. 1M builds for Phase 4: how many hours are you willing to leave the laptop running?
4. Should the repo move out of OneDrive before benchmarking? Sync activity adds noise.
