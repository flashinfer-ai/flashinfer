# TODO: knob selection flexible enough for vLLM — heuristic, not hot-path tuning

Status: analysis only, nothing implemented yet (2026-07-16). Motivated by the
2026-07-15 vLLM e2e runs (`$LUSTRE_ROOT/moe_ep_benchmark/vllm_e2e/RUNS.md`,
run 13 + the "knobs=auto paradox" section).

## Why the current mechanism doesn't work in an engine

Two independent failures, both observed e2e:

1. **`knobs="auto"` is unusable in the hot path.** The autotune fires at the
   first `compute()` per encountered shape
   (`backends/mega/kernel/{nvfp4,mxfp8}_cutedsl/backend.py` →
   `kernel_src/cutedsl_megamoe/shim/autotune.py`): `dist.barrier()`, ~24
   `cute.compile`s and 576+ candidate timings per shape in a decode engine
   that sees many token counts. Minutes of stall, collective host-side work
   mid-serving, and a CUDA-graph blocker ([[todo_cuda_graph]] blocker 3).
2. **Even offline, the tuner's winner doesn't transfer.** The auto winner
   (ikr + mma 256x256 + flag_batch 8 + standalone_warps) measured 710 µs in
   the tuner harness vs 1464 µs default / 1176 µs deep_gemm — a 2x
   kernel-level win — yet lost **~13% e2e on both prefill and decode**
   (prefill 22348 vs 25843 tok/s default). The tuner metric (synchronized
   launches, median of max-across-ranks) does not model the pipelined
   engine; ikr's cross-rank atomics under real skew are the prime suspect.
   Correctness was fine (smoke run 15) — this is purely a perf-transfer gap.

3. Background: the shipped `default_knobs` profiles were derived at
   hidden-7168/top-8; at DSV4-Flash geometry (4096/2048/256/top6) the
   cutedsl kernel runs ~25% SLOWER than deep_gemm (1464 vs 1176 µs) — the
   default table has geometry holes.

## Goal

A pure-lookup hot path: knob resolution = a cheap deterministic function of
(geometry, token bucket) evaluated at layer init / first shape encounter,
no compiles, no collectives, no timing in `forward()`. Tuning becomes an
**offline** tool that emits validated dicts feeding that lookup.

## Plan

1. **Extend the kernel-repo sweep grid** with the DSV4-Flash point
   (hidden 4096, inter 2048, 256 experts, top-6) and any other production
   geometries; record per-(geometry, num_tokens-bucket) winners in
   `kernel_src/cutedsl_megamoe/TUNING.md` as usual.
2. **Derive the heuristic** from the sweep data: a
   `default_knobs(geometry, tokens)` table keyed on coarse buckets
   (e.g. tokens/rank in {<=16, <=128, <=512, <=2048, >2048}) with
   interpolation-by-nearest for unswept geometries. Keep it a plain dict +
   selection function in the shim — inspectable, override-able.
3. **e2e-validate before pinning**: a knob profile only enters the table if
   it wins (or ties) in the *pipelined* engine, not just the tuner harness —
   the 07-15 paradox makes tuner-only validation insufficient. Cheapest
   proxy to investigate: an unsynchronized/skewed multi-rank timing mode in
   `shim/tuner.py` so the offline tuner ranks candidates under realistic
   skew (launch back-to-back without barriers, measure steady-state).
4. **Pinning interface**: `FI_MOE_EP_KNOBS='{...}'` (dict) already works —
   document it as THE production mechanism; make `knobs="auto"` warn loudly
   (or refuse) when world_size > 1 inside a live engine, pointing at the
   offline flow.
5. **Offline tuner CLI**: wrap the existing autotune sweep as a standalone
   `python -m ...` entry that runs the sweep at given geometry/token grid
   and prints the pinned-dict JSON, so integrators never trigger it
   implicitly.
6. Re-run the vLLM offline matrix with the new geometry-correct defaults —
   expectation from the microbench data is fi_nvfp4 ≥ fi_dg once the kernel
   isn't running 7168-derived knobs.

Related: [[todo_cuda_graph]] (autotune is capture-fatal; a pure-lookup hot
path removes that blocker), [[todo_fi_dg_nondeterminism]] (skew-sensitive
behavior shows up in both).
