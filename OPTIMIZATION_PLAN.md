# GSPIKE GPU solver — optimization plan

**Update (feature branch `perf/blocked-spikes`):** Phase 0 is fully integrated
and Phase 1 (blocked TRSM for spike generation) is implemented and measured.
At `n = 50000, klu = 500, nrhs = 2` mixed factorization went from ~314 ms
(initial Phase-0 state) to ~170 ms; the spike phase alone went from 187 ms to
43 ms.  `opts::auto_full_spikes` now enables the Phase-0 full-spike correction
when memory allows, and `--profile` reports phase timings.  Items below remain
as the record of the investigation; Phase 1 should be considered DONE.

Status of the earlier experiments: main tree was at `3b2ea98`, experiments in
worktree `/tmp/gspike-opt`, branch `opt-recip`:

- `5a74a2a` — reciprocal diagonal (validated)
- `604c11e` — NB=4 panels + full-spike correction (validated prototype)

Hardware used for all measurements: NVIDIA L40, CUDA 12.6, driver 560.35.03.
Benchmark protocol: `./build/spike_bench --type {float,double,mixed}` (default
sweep) plus forced `--type <t> 50000 500 2 0` and `80000 800 1 0`; correctness via
`ctest` (dense reference, pivoting, fuzz, reuse).

Constraints for every item below:

1. `relres` must stay at current levels (fp32 ~1e-6, fp64 ~1e-15, mixed ~1e-12).
2. Default `opts` must remain memory-safe (currently peaks ~2.5 GB at n=80k/klu=800).
3. No new library dependencies; no loss of pivoting/non-dominant correctness.
4. Any new tunable must be validated on at least one other architecture (T4/A100)
   before becoming a default.

---

## Phase 0 — validated wins, ready to integrate (do first)

| # | Change | Files | Measured effect | Risk |
|---|--------|-------|-----------------|------|
| 0.1 | Precompute `1/U(ii)` once (`invert_diag` kernel, `d_diag` / `d_diagR`), replace `/` with `*` in `band_solve`, `band_solve_warp`, `band_solve_block` | `src/kernels.cuh`, `src/spike.cu` | fp64 solve **2.7–3.4×**, fp64 factor 1.2–1.45×, fp32/mixed solve 1.2–2× | low (≤1 ulp change); tests pass |
| 0.2 | Band-LU panel width NB 16 → 4 in `launch_band_lu` | `src/spike.cu` | factor **1.10–1.26×**; NB=2/8/32/64 all measured worse | low; re-check on T4 |
| 0.3 | Save full spike solutions and replace 2nd band solve of `solve_inplace()` with `x = x1 − Vp·w − Vm·z` (`full_correct`) | `include/gspike/spike.h`, `src/kernels.cuh`, `src/spike.cu` | solve **1.4–3×** on top of 0.1/0.2 | medium — memory cost |

Combined (base → optimized): mixed 80k/800 **1368/158 ms → 1078/91 ms**;
double 80k/800 **2558/204 ms → 2127/66 ms**; float 50k/500 **386/37 ms → 316/20 ms**.

### Integration steps for 0.3 (memory safety)

1. Add `bool full_spikes = false;` to `gspike::opts` with a doc comment describing
   the `2·klu·n` cost (~977 MiB fp64 / 512 MiB fp32 at n=80k, klu=800).
2. Allocate `d_fs` and copy the solved spike columns only when the flag is set;
   `solve_inplace()` already branches on `d_fs`.
3. Wire `solver`/`mixed_solver` to pass the flag through; enable it in
   `spike_bench` via a CLI option (e.g. `--full-spikes`) so both paths stay covered.
4. Add a test that runs the same problem with the flag on and off and compares
   solutions bit-for-bit-ish (residual + elementwise tolerance).
5. Update README benchmark table with both configurations.

---

## Phase 1 — factorization: blocked TRSM for spike generation (largest remaining prize)

**Why:** after Phase 0, spike solves are still ~36 % of factorization time. ncu shows
the kernel is now **L2-bandwidth bound (89 % of peak, 99.7 % L2 hit rate)**: the same
band factors are streamed from L2 once per each of the `2·klu` right-hand sides with
no reuse. A classic blocked triangular solve can reuse a tile of L/U across many RHS.

**Target:** 20–40 % off factorization; overall mixed solve at 50k/500 from ~360 ms to
~280–300 ms.

**Design sketch (mirror `band_lu_update`):**

1. Forward `L Y = P C`: process `NB_R` RHS columns at a time.
   - Panel pass: solve the first `B` rows of Y sequentially (one warp per RHS),
     load L panel into shared memory.
   - Trailing pass: one block per (row tile × RHS tile) applies
     `Y_trail -= L21 · Y_panel`, accumulating in a 4×4 register micro-tile.
2. Backward `U X = Y`: symmetric, from the bottom; use the Phase 0 reciprocal array
   for the diagonal scale (no divisions).
3. Keep the scalar warp-per-column kernel as fallback for the pivoted path, exactly
   as `band_lu_panel` does today (`d_pivoted` flag already exists).
4. Start with `NB_R = 16`, `B = 64`; tune `NB_R ∈ {8,16,32}` and micro-tile shape on L40.

**Validation:** full `ctest`, then compare `factor_ms` for 25k/250, 50k/500, 80k/800
in fp32/fp64. Acceptance: ≥15 % factor reduction with unchanged residuals.

**Risk:** high implementation complexity (band-storage indexing, P swaps, edge rows
of the last partition). Mitigation: keep old path behind a compile-time switch and
bisect with the dense-reference test.

---

## Phase 2 — reduced system: break the single-block bottleneck

**Why:** the `m × m` interface system (m = 2·klu·(p−1)) is factored and solved by a
single CUDA block on one SM. It is ~25–30 % of factorization and ~1/3 of each solve.

**Work items:**

1. Add SPIKE level 2: split the reduced system into 2 (later 4) diagonal blocks,
   factor them in parallel, compute their spikes and a tiny third-level interface,
   solve it with the existing single-block path. Reuse all current kernels — this is
   mostly host orchestration (`factor()` and `solve_inplace()`) plus a small
   re-partitioning of `d_R`.
2. Alternatively/additionally, use `band_solve_warp` (W=4/8) instead of
   `band_solve_block` for the reduced solve when nrhs is large enough.
3. Acceptance: ≥20 % off reduced-system time; net ≥10 % off end-to-end solve for
   p ≥ 3 cases; no change for p=1.

**Risk:** medium; recursion depth must be bounded (m gets small quickly).

---

## Phase 3 — lower-risk micro-optimizations (measure each, keep only clear wins)

| # | Idea | Where | Expected | Notes |
|---|------|-------|----------|-------|
| 3.1 | Replace the global reload of `r[i]` in back substitution with a warp shuffle of lane 0's updated value; keep the current row in registers | `band_solve_warp<1>`, `band_solve_block` | 5–10 % of solve | ncu: 38 % of stall cycles are L1TEX data waits |
| 3.2 | CUDA-graph capture of the deterministic, no-pivot factor/solve sequence | `spike.cu` (host side) | noticeable only for small n | NB=4 means ~8k launches per factorization; graph must be invalidated when `nrhs` changes; keep scalar-fallback path outside graph |
| 3.3 | Re-tune auto partition heuristic constants (`bal_rule` 1.3 and 2000) after Phase 0/1 changed the cost balance; consider making `p` depend on `nrhs` and on `full_spikes` | `spike.cu` constructor | a few % | current sweep shows p=4 best at 50k/500, p=3 at 80k/800; do not hard-code |
| 3.4 | Fuse the conversion/copy kernels in `mixed.cu` (B→float, float→B, residual, correction) into wider kernels | `src/mixed.cu` | <5 % | low priority |

**Explicitly rejected experiments** (do not redo without new evidence):

- Vectorized `double4` stores in `store_band4`: ≤0.5 % change (measured 542.1 vs 543.9 ms at 50k/500).
- Smaller `band_lu_panel` block (64/128/256/512 vs 1024): <1 % change.
- `TILE` values other than 64 in `band_lu_update`: **incorrect results** (relres 6e-4 instead of 1e-15); see Phase 5.

---

## Phase 4 — alternative fast path (product decision, optional)

`bench/duel.cu` already contains a band-aware Jacobi-preconditioned BiCGStab that is
10–50× faster than any direct solver on the benchmark's well-conditioned bands, but
stagnates on non-dominant bands. Proposal:

1. Extract the band SpMV + BiCGStab into an opt-in `iterative_gbsv()` API with a
   residual-based fallback to `spike_gbsv()` if it fails to converge.
2. Document the regimes: few RHS + diagonally dominant → iterative; many RHS,
   pivoting needed, or tight residual → SPIKE (possibly `full_spikes` enabled).

---

## Phase 5 — hardening and documentation

1. **Correctness fix:** `band_lu_update` is templated on `TILE = 64` but the 256-thread
   4×4 micro-tile only covers 64×64. Replace the default with a fixed constant and
   add `static_assert(TILE == 64)` (or make it a plain constant) so TILE≠64 cannot
   silently produce wrong answers.
2. Update all stale comments that still say "16-column panel" and "reused sixteen
   times" after NB=4.
3. Regenerate README benchmark tables after each phase; keep the CPU/MKL comparison
   rows in sync.
4. Add a CI job (or at least a `ctest` case) that validates the `full_spikes` opt-in
   path when it lands.
5. Re-run `build-asan` tests after each phase; device errors still need
   `compute-sanitizer`.

---

## Suggested execution order

1. Phase 0.1 → 0.2 → 0.3 (integrate, benchmark, document) — ~1 day.
2. Phase 1 (blocked TRSM) — largest payoff, ~3–5 days.
3. Phase 2 (reduced system) — ~2–3 days.
4. Phase 3.1 → 3.2 → 3.3, with measurements after each — ~2 days.
5. Phase 5 continuously; Phase 4 only if the product wants an iterative mode.

## Definition of done for each item

- `ctest --test-dir build --output-on-failure` passes (Release **and** Debug/ASan).
- `spike_bench` default sweep shows no regression >2 % at any size in the targeted
  metric, with `relres` unchanged.
- Changes are measured on L40 and at least one other GPU before becoming defaults.
- README and comments are updated in the same commit as the code.
