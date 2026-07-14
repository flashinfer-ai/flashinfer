# CuTeDSL MegaMoE tuning + performance notes

What was changed, measured, and learned while tuning the cutedsl mega
backends (2026-07-14, 4x GB200, DeepSeek-V3-like geometry: 256 experts,
top-8, hidden 7168, inter 2048, EP=4, unless noted).  Companion to
`SKILL.md` (which covers the drop-update workflow); this file covers the
tuning surface and the measurement methodology.

## Headline result

`nvfp4_cutedsl` is **1.3-1.7x FASTER than `deep_gemm_mega` at every token
count 1..8192** through the full FI forward path (steady-state timing).
An earlier "cutedsl is 2x slower than dg" reading was a measurement
artifact — see "Benchmarking lessons" below.  `mxfp8_cutedsl` trails dg
0.5-0.7x by construction (fp8 weights move 2x the bytes of dg's fp4;
different accuracy point — dg is not its fair baseline).

Full-sweep p50 (e2e_pipelined, µs): see
`moe_ep_benchmark/results/sweep_20260714_141327_fi_mega.csv`.  Key points:

| tok/rank | dg     | nvfp4 (vs dg) | mxfp8  |
|---------:|-------:|--------------:|-------:|
| 8        | 213.0  | 133.4 (1.60x) | 380.5  |
| 64       | 286.7  | 165.9 (1.73x) | 538.7  |
| 512      | 346.1  | 234.5 (1.48x) | 626.7  |
| 1024     | 475.1  | 332.3 (1.43x) | 769.0  |
| 2048     | 823.3  | 537.6 (1.53x) | 1198.5 |
| 8192     | 3055.8 | 1841.1 (1.66x)| 4716.3 |

## The knob system (`shim/tuner.py`, `shim/autotune.py`)

- `tuner.py` mirrors the kernel team's `tester/solvers/inference_solver.py`
  knob taxonomy (re-audit on every drop).  Correctness knobs change a code
  path/output (`token_back_mode`, `mma_tiler_mnk`, `in_kernel_fc2_reduce`,
  ...); perf knobs are output-invariant (`group_hint`, `flag_batch`,
  `epi_flag_batch`).
- `default_knobs(num_tokens, dtype=...)` — the measured per-size profiles
  (provenance in the profile dict comments).  NVFP4 has FOUR profiles; the
  dominant axis is `token_back_mode`:
  - `epi_warps` wins at small batch but falls off a cliff mid-range
    (+18% at 512 tokens, +35% at 1024 — every dispatch-warp candidate beat
    every epi_warps candidate there).
  - tile (256x128 vs 256x256) and `flag_batch` are second-order (~1-5%).
  MXFP8 has ONE profile (fb4 + epi_warps at all sizes); the NVFP4-large
  fb8+dispatch-warp schedule measured ~5% slower for MXFP8 at 2048.
- Backend configs (`Nvfp4/Mxfp8CutedslMegaMoeConfig.knobs`): explicit dict
  overrides the heuristic ENTIRELY (pin every knob you care about);
  `"auto"` runs the online autotuner at the first forward.
- `autotune.py` — collective online tuner: every EP rank compiles+times the
  same candidate list in lockstep, per-candidate medians are all-reduced
  MAX (slowest rank = collective latency), argmin winner applied
  identically everywhere.  Cost: one `cute.compile` per candidate
  (~1-2 min), once per session.  Candidates mirror the tester sweep
  restriction minus `in_kernel_fc2_reduce` (changes output placement /
  determinism; config-owned until the sym-output plumbing lands — see
  `../todo_trtllm_import.md`).
- The kernel-repo tester remains the wide-sweep tool
  (`torchrun -m tester.tester --mode Perf --sweep --use_knob ...`); winners
  transfer via the `knobs=` dict.  Its problems (`nvfp4_perf.jsonl`) are
  top-6 with different inter/experts — never compare its numbers to
  FI-default-geometry runs directly.

## Launch-path work (why FI now matches the bare kernel)

Steady-state cost of the full FI forward over the bare kernel launch is now
**2-13 µs** (mostly the output copy at large tokens).  What it took:

1. **Launch-kwargs cache** (`_CompiledMega.launch_*`): `frontend.run()`
   used to rebuild all 12 cute tensor views (`from_dlpack`) + a
   `SymBufferHost` on EVERY launch.  Now keyed on input data_ptrs + token
   count + stream; a hit also skips input validation (validated when the
   entry was built).  Any config change nulls the cache.
2. **No per-launch workspace reset**: workspaces are allocated zeroed and
   the kernel TAIL-CLEANS its own counters/flags (the kernel-team drivers
   and tester never host-reset; the NVLink phase slots must NOT be reset).
   `run(reset_counters=False)` is the default; `True` only recovers an
   aborted launch.  Guarded by repeated-forward bit-exact tests.
3. **Async wrappers**: `nvfp4/mxfp8_mega_moe(sync=False)` default — kernel
   + output copy are stream-enqueued, no host sync (matches dg).  Anything
   timing with `perf_counter` must pass `sync=True` (the autotuner does).
4. **Launch thunks** (`{nvfp4,mxfp8}_mega_launch_thunk`): prebuilt bare
   launch closures matching the tester's `perf_run` timed region — for
   benchmark loops, tuners, and (future) CUDA-graph capture.

## Benchmarking lessons (how the 2x-slower misread happened)

Use `moe_ep_benchmark` (`RUNBOOK.md` there) with `MEGA_TIMING`:

- `kernel` — tester-parity bare launch (back-to-back, per-iter events,
  L2 flush outside the window).  Compare THIS against the kernel team's
  tester numbers, at MATCHED geometry.
- `e2e_pipelined` — full FI forward, back-to-back.  The serving-relevant
  number.
- `e2e` — full FI forward, each iter from a global barrier + idle GPU.
  A cold-start stress case: it adds ~85-90 µs of **from-idle collective
  start skew** (the collective dispatch stalls on the slowest rank's
  from-idle launch; dg shows the same effect at ~34 µs).  A pipelined
  workload never pays this; if a workload truly launches from global idle,
  CUDA-graph capture of the thunk is the fix.

The original misread combined (a) barrier-cold e2e timing of a then-heavy
launch path against dg's thin wrapper, and (b) a geometry mismatch vs the
tester problems.  Rule: match BOTH the problem shape and the timed region
before comparing MoE backends.

## Next levers

See `../todo_trtllm_import.md` (ideas from TRT-LLM PR #16190): combine wire
formats (`16e2m1xbf16` / `32e4m3xe8m0`, 2-4x less NVLink combine traffic),
`in_kernel_fc2_reduce` in-flight reduction (tester's overall winners are
ikr candidates, +1-2%, plus a multi-GB shared-workspace saving), CUDA-graph
capture (`../todo_cuda_graph.md`).
