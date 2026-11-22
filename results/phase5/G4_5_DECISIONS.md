# Gate G4.5 Decisions — Sign-Off (`T3.8`–`T3.11`)

**Date Recorded:** 2026-11-15 (authorized prior to `T3.8`–`T3.11` execution)  
**Context:** Response to [`results/phase5/PROPOSAL.md`](./PROPOSAL.md) (`T3.6` microbenchmark & `T3.7` 100k pilot).

## Approved Decisions

1. **Execution Tier (`T3.9`):** **Option C** (`reps = 5` timed repetitions after 1 untimed warmup).
   - Record `num_queries` and `is_subquery_probe` on every arm.
   - Scope `S1_post_uncapped` (`S1_uncap`) strictly to $\sigma \ge 0.05$ (and document claims strictly where measured).
2. **Opaque Closure Guardrail (`eval_limit`):**
   - Keep `FilteredSearchParams::default().eval_limit` as `None`.
   - Expose `pub const DEFAULT_OPAQUE_EVAL_LIMIT: usize = 16_384` and `FilteredSearchParams::with_opaque_eval_limit()`, documenting its `100k` origin ($4 \times K'_{\max} = 4 \times 4,096$) and adding a `1M` opaque-closure spot check (`sensitivity_n1000000.json`).
3. **Versioned Disk Cache:**
   - Cache predicate `.bin` bitsets, `.ivecs` ground truth, and `.fvecs` distances (plus `.json` provenance) under gitignored `data/phase5/`, keyed by `scale / mode / sigma / seed / generator_version` and verified on load.
4. **Execution Progression:**
   - Authorized `T3.8` & `T3.10` immediately, followed by `10k` smoke + single `100k` cell checkpoint review, and then `proceed` approval for the full `100k` and `1M` Option C matrix + sensitivity runs (`T3.9`, `T3.11`).
