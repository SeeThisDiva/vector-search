# Milestone M3 / Phase 5 Design Changelog

This document records the exact changes made to `results/phase5/DESIGN.md` in preparation for Dev's Gate G4 review, in accordance with `AGENT_PLAN.md` Ground Rules 1–11 and orchestrator instructions.

---

## Summary of Revisions

| # | Patch Item | Section / Lines in Prior | Revision Made | Rationale / Rule |
|---|---|---|---|---|
| 1 | Instruction-Set Terms | L5-6, L22, L78, L185, L293-297, L334, L349, L598, L671, L835, L840, L859 | Removed all occurrences of "SIMD", "AVX2", "AVX-512", "hardware POPCNT", "SSE", "NEON". Replaced with "compiler auto-vectorization" and standard library methods (`u64::count_ones`, `u64::trailing_zeros`). | Ground Rule 6; strict compiler auto-vectorization framing only; no hardware instruction set claims without specific feature flags. |
| 2 | Unmeasured Latencies & Cache Penalties | L297 | Deleted "< 0.05 ms for 1,000 distances" and "60-80 ns per cache miss"; reframed as hypotheses to be measured in T3.8. | Ground Rule 3; no unmeasured numbers in docs. |
| 3 | Membership Cycle Count | L76 | Dropped "membership check in 2 CPU cycles". Described as word lookup and bitwise shift. | Ground Rule 3 & 10; cycle counts vary across microarchitectures and must not be asserted as constants. |
| 4 | Unmeasured Throughput Claim | L825 | Deleted "bitset creation > 10 GB/s". | Ground Rule 3; no unmeasured throughput targets. |
| 5 | Broken Crossover Model Example | L388-398 | Removed the inconsistent worked example ($[1.254]^{0.714} \approx 1.18 \implies \sigma^* \approx 10\%\text{--}20\%$). Stated explicitly that closed-form models are qualitative hypotheses only, and the true crossover must be determined via empirical measurement (§4.3). | Ground Rule 3 & 11; uncalibrated closed-form formulas break down at small $N$ ($1.18 > 100\%$). |
| 6 | PyO3 Bindings & Workspace | L33, L784-824, T3.10 | Replaced in-tree PyO3 / `crates/vector_search_py/` with cross-language subprocess parity tests in `tests/test_phase5_parity.py` calling the compiled Rust `phase5` binary via `subprocess.run` (matching Phase 1). Moved PyO3 to Optional section (§8.2). | Ground Rule 6; library remains `#![forbid(unsafe_code)]` and `std` only with zero external dependencies. |
| 7 | QPS Target Assertion | T3.6 (L840) | Changed "> 2,000 QPS at 0.1% on 1M" to "measure and report QPS". | Ground Rule 3 & 4; targets are measured and reported, never asserted without measurement. |
| 8 | Adaptive Filter Router | §4.4, T3.7 | Moved Adaptive Hybrid Router out of core M3 into "Optional Features (Needs Dev Approval)" (§8.1), excluded from Gate G4 criteria. Replaced T3.7 with Empirical Crossover Measurement & Analysis. | Scope discipline; core M3 focuses on evaluating all three strategies across selectivities. |
| 9 | Result Scale Demarcation | L31, L422, T3.8 | Standardized 100k and 1M as the primary result scales; designated 10k strictly as a fast smoke test / CI check. | Consistent with Milestone M2/Phase 4 evaluation scale decisions. |
| 10 | Bridge-Hop Budget Cap | §3.2, L234, Algorithm 3 | Set $H_{\text{max}} = 3 \times ef$. Specified behavior upon cap exhaustion: immediately abandon graph traversal and fall back to Strategy 3 (pre-filter brute force). Instrumented fallback frequency as a reported metric. | Bounds worst-case traversal latency in sparse/disconnected graphs while guaranteeing zero candidate truncation. |
| 11 | Zero-Allocation Scratch Buffers | §6.1, L718-750 | Specified explicit preallocation sizing based on index capacity $N$, maximum degree $M_0$, and maximum query beam $efSearch$. Described verification via capacity preservation assertions (`debug_assert_eq!`) and automated unit tests. | Strictly preserves zero-allocation hot path invariant. |
| 12 | Opaque Closure Evaluation Budget | §6.1, L574 | Added configurable per-query predicate evaluation budget $E_{\text{max}} = C_{\text{pred}} \times efSearch$ for `LambdaPredicate<F>`. Defined fallback to Strategy 3 upon exhaustion to guarantee zero truncation and bounded runtime. Added metric tracking. | Protects against unbounded CPU expenditure in adversarial or zero-match closure queries. |
| 13 | Task T3.1 Acceptance Criteria | T3.1 (L835) | Replaced "100% unit test coverage" with "unit tests for every public method, including edge cases". | Realistic, verifiable engineering contract. |
| 14 | Gate G4 Checklist Hygiene | §7.2 | Ensured all checklist checkboxes are unticked (`- [ ]`) for Dev's independent evaluation. | Dev designs and signs gates; implementers do not self-sign. |
| 15 | Task Card Classification | §7.1 | Assigned strict AGENT_PLAN classes (`PRESERVING`, `GRAPH-CHANGING`, `MEASUREMENT-ONLY`) to all 11 task cards. | Ground Rule 7 compliance. |
| 16 | Analytical Cost Model Framing | §4.1 | Labeled all parameters ($t_{\text{dist}} \approx 4.2 \text{ ns}$, etc.) as `Hypothesis (to be measured in T3.8)`. | Ground Rule 3 compliance. |
| 17 | Open Questions for Dev | §9 (New) | Added 6 specific architectural open questions for Dev's review. | Ground Rule 1 & 11 compliance. |
| 18 | Degradation at Low Selectivity | §1.1 (L37) | Replaced "degrading query throughput by orders of magnitude" with statement that throughput degradation at low selectivity is expected and will be measured in T3.8. | Ground Rule 3 compliance. |
| 19 | Brute-Force Reversal Wording | §1.1 (L39) | Replaced assertion that scan "becomes faster ... !" with an explicit hypothesis statement to be measured empirically in T3.7/T3.8. | Ground Rule 3 compliance. |
| 20 | Table 3.4 Distance Evaluations | §3.4 (L367) | Replaced "$\ge 10^4$" and "$10^2 \dots 5 \times 10^3$" with "Measure and report (hypothesis: high / moderate to high)". | Ground Rule 3; no unmeasured numbers in docs. |
| 21 | Table 3.4 Recall Framing | §3.4 (L370-371) | Replaced "High (>98%)" and "100.0%" with "Measure and report" for Strategies 1 & 2, and "Exact by construction when $\ge k$ matches exist" for Strategy 3. | Ground Rule 3 & 10 compliance. |
| 22 | Truncation Target Framing | §3.4 (L373), §5.2 (L520) | Removed "(0.0% targeted via fallback)" and "0.0% guaranteed"; framed as "Measure and report; fallback to Strategy 3 at $H_{\text{max}}$ is designed to avoid fewer-than-k results" and "Exact by construction when $\ge k$ matches exist". | Ground Rule 4 compliance. |
| 23 | Parity Reference Clarification | §6.2 (L801), T3.10 | Defined NumPy filtered brute force as the authoritative exact ground truth reference. Clarified FAISS is strictly an optional cross-check using existing dependencies (`faiss-cpu`), with no FAISS `IDSelector` requirement in M3. | Scope discipline and zero new dependencies. |
| 24 | Scratch Buffer Consistency & Cap | §3.1, §3.4, §6.1 | Replaced "dynamic vector resizing" with preallocated over-fetch bound $K'_{\text{max}}$. If $k' > K'_{\text{max}}$, Strategy 1 falls back to Strategy 3 and records `overfetch_cap_exhausted_fallbacks`. Nothing grows on the query hot path. | Zero-allocation hot path invariant. |
| 25 | Sweep for Unmeasured Numbers | §1.1, §2.1, §3.1, §3.3 | Replaced $10^6$ in bitset formula with $1{,}000{,}000$, reframed over-fetch heap pressure qualitatively, and added Question 6 on $K'_{\text{max}}$ sizing to §9. | Ground Rule 3 compliance. |

---

## 2026-11-10: Dev Revision Superseded Agent Revision

On 2026-11-10, Dev (project owner) provided a revised `DESIGN.md` and accompanying `DESIGN.patch` in `design patch.zip`. Dev's version superseded the prior agent revision of `results/phase5/DESIGN.md` byte-for-byte. Dev signed off Gate G4 with decisions recorded in `results/phase5/G4_DECISIONS.md`.

