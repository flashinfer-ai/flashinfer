# TODO: CUDA graph compatibility for the mega_moe path (cutedsl backends)

Status: IMPLEMENTED for single-rank (2026-07-16). Blocker 1 was already
resolved by the earlier `sync=False` default on the public entry points
(commit 68a29e6a); this change added the rest of the restoration plan:
capture guards (`ensure_not_capturing` in `shim/comm.py`, wired into
`_ensure_mega_compiled` / `set_gate_up_clamp` / `apply_knobs` /
`_release_workspace` / `autotune_knobs` / the layer's `_ensure_workspace`),
the `MoEEpMegaLayer.warmup()` contract, and capture-safe `sync` gating.
Verified on GB200 by `tests/moe_ep/test_mega_cuda_graph.py` (nvfp4 + mxfp8:
capture post-warmup, 3x replay bit-exact vs eager, replay over mutated input
buffers, and loud failure when capturing without warmup). Remaining
follow-ups: the 2-rank lockstep-replay test (item 5 below), the 2-rank
skewed-rank staging stress test (break-risk item 2), and vLLM-side adoption
(shared-workspace warmup + dropping VLLM_ENFORCE_EAGER).

Original analysis (2026-07-14) below.

Update 2026-07-16: priority raised by the vLLM 0.25.1 e2e integration
(`$LUSTRE_ROOT/moe_ep_benchmark/vllm_e2e/RUNS.md`) — every run there needed
`VLLM_ENFORCE_EAGER=1`, and the nsys attribution showed the fi paths are
host-gap-bound (fi_dg has LESS total GPU work than native yet lower
throughput; cudaLaunchKernel 330k fi_dg / 946k fi_nvfp4 vs 100k native), so
graph capture is a first-order perf item, not just a compatibility checkbox.
Two additions to the analysis below from that work: (a) the vLLM wrapper now
shares one geometry-keyed workspace across the 43 layers — the warmup
contract and capture guards must handle externally shared workspaces, not
just the per-layer lazy `_ensure_workspace`; (b) `knobs="auto"` must never
run in-engine regardless of graphs (see `todo_vllm_knob_heuristic.md`), which
turns blocker 3 into "assert not capturing + point at the offline flow".
See also `todo_fi_dg_nondeterminism.md` before baking replays into tests —
eager fi_dg is currently not run-to-run deterministic, so a graph-replay
test that compares against eager needs statistical rather than bit-exact
tolerance until that's resolved.

Nothing in `moe_ep` is graph-aware today (no `is_current_stream_capturing` /
`CUDAGraph` handling anywhere in the tree). The compiled CuTeDSL kernels
themselves are graph-capturable in principle — all cross-rank comm is
device-side through pre-baked NVSHMEM peer pointers, and the steady-state
launch is a plain arg-packed `cuLaunch`. What breaks capture is entirely in
the host shim: an unconditional `torch.cuda.synchronize()` on every forward,
plus lazy first-call work (compile, symmetric-heap alloc, autotune) with no
warmup contract to force it before capture.

## Hard blockers in the captured region

1. **`torch.cuda.synchronize()` inside `Frontend.run()`** —
   `kernel_src/cutedsl_megamoe/shim/mxfp8.py` (`run()`, `sync=True` default)
   and the same pattern in `shim/nvfp4.py`. The public entry points
   `mxfp8_mega_moe` / `nvfp4_mega_moe` call `run()` without `sync=False`.
   A device synchronize during stream capture aborts the capture immediately.
   Single biggest and easiest fix (see "Restoration plan" and "Break-risk"
   below for the two variants).

2. **Lazy compile + allocation on first launch** — `_ensure_mega_compiled`
   (`shim/mxfp8.py` / `shim/nvfp4.py`) does `cute.compile`, `torch.zeros` for
   the local workspace, and `sym_zeros` → `nvshmem.core.tensor` — a
   **collective host-side symmetric-heap allocation**. Fatal if it fires
   during capture. Same for the layer's lazy `_ensure_workspace`
   (`modes/mega_layer.py`).

3. **`knobs="auto"` autotune at first `compute()`** —
   `backends/mega/kernel/{mxfp8,nvfp4}_cutedsl/backend.py` →
   `shim/autotune.py`. The sweep does `dist.barrier()`, one `cute.compile`
   per candidate, wall-clock timing with internal syncs, an `all_reduce`,
   and `.item()`. Must complete on all ranks, in lockstep, before any rank
   starts capturing.

4. **Per-forward recompile triggers** — `set_gate_up_clamp()` is invoked on
   every `mxfp8_mega_moe` call when a clamp is passed (no-op only if the
   value is unchanged). If it (or `apply_knobs`) changes mid-capture, it
   frees the NVSHMEM workspace (host collective free) and recompiles on the
   next run — silently corrupting the capture. These paths should raise a
   clear error when capturing.

## Already graph-safe (verified by reading, not yet by test)

- Steady-state launch: `launch_kwargs` are cached keyed on data pointers
  **and the current stream** (`_launch_cache_key`), so capture on a side
  stream correctly rebuilds kwargs with the capture stream. The
  `mega.compiled(**kwargs)` call is a plain kernel launch.
- `reset_compiled_mega_workspaces` (`shim/comm.py`) is device-side
  `zero_()`s — it gets baked into the graph and replays, matching eager. The
  NVLink barrier phase counter is deliberately excluded from the zeroing,
  and the kernel tail-cleans its own flags (per the `make_launch_thunk`
  docstring), so repeated replay should be sound — deserves an explicit
  multi-replay, multi-rank test.
- Staging + quantize helpers (`stage_mega_moe_inputs`,
  `mxfp8_quantize_per_block_32`, `nvfp4_quantize_per_block_16`) are pure
  device-side torch ops; allocations during capture land in the graph's
  private pool.
- Per-forward validation is shape/dtype only — no hidden device→host syncs
  (`.any()` / `.item()`).
- The kernel launches the full padded buffer with pad rows marked
  `topk_idx == -1`, so a graph captured at `max_tokens` naturally supports
  variable live-token counts at replay via staged data.

## Restoration plan

1. **Kill/gate the per-forward sync** (two variants, see break-risk below).
2. **Explicit warmup contract**: add `warmup()` on
   `MoEEpMegaLayer` / `MegaKernelBackend` that runs one full eager forward —
   workspace alloc, autotune sweep (if `knobs="auto"`), `cute.compile`, and
   one real launch (flushes lazy `cuModuleLoad`) — then synchronizes.
   Document "call on all ranks before capture". The frontend's existing
   `warmup()` only compiles without launching and isn't exposed at the
   layer level.
3. **Capture guards**: in `_ensure_mega_compiled`, `set_gate_up_clamp`,
   `apply_knobs`, `_release_workspace`, raise if
   `torch.cuda.is_current_stream_capturing()` and a compile/alloc/free would
   be needed — fail loudly instead of corrupting the capture.
4. **Verify the CuTeDSL launch under capture** — `JitExecutor.__call__`
   should be a pure launch (cute-dsl FMHA is captured elsewhere in
   flashinfer), but confirm empirically for this kernel, including NVSHMEM
   device-side barrier behavior across replays.
5. **Tests**: capture `layer.forward` in a `torch.cuda.CUDAGraph`
   post-warmup, replay several times against the eager reference —
   single-rank (`MEGA_NO_DIST=1`) first, then 2-rank with lockstep replays
   (the kernel has global device-side barriers, so ranks must replay
   together, same as eager).

Caller-facing semantic note: under graphs the output tensor `y` allocated in
`mega_layer.forward` is fixed at capture — users consume the captured tensor
across replays (standard graph practice; document on the layer).

## Break-risk analysis (will the fix break existing functionality?)

Items 2–5 above are purely additive. Item 1 (sync removal) has two real
dependents today:

1. **Autotune timing loop** (`shim/autotune.py`, `autotune_knobs`): times
   candidates with `time.perf_counter()` around `launch()` and documents
   "launch() syncs internally". If the mega entry points stop syncing,
   autotune silently measures launch overhead and picks a garbage winner.
   Any async change must add an explicit `torch.cuda.synchronize()` inside
   the autotune timing loop.
2. **Multi-rank pacing in eager mode**: correctness should hold without the
   sync (per-rank stream ordering + back-to-back no-sync launches are the
   proven `make_launch_thunk` perf-loop pattern), but one race needs a test
   rather than an assumption: `stage_inputs` writes iteration N+1's
   activations into the *symmetric* buffers as soon as the local kernel N
   finishes — if a slower peer's kernel N is still reading this rank's
   buffers over NVLink and the kernel does **not** end with a cross-rank
   exit barrier, staging can clobber data mid-read. The tail-clean /
   phase-preserving barrier machinery strongly suggests an exit barrier
   exists (and the local `synchronize()` never protected against this
   anyway — it's local-only; it just throttled rank skew as a side effect).
   Settle with a 2-rank stress test with deliberately skewed ranks.

Tests and smoke scripts are fine either way (implicit syncs via `.cpu()` /
comparisons; `_main()` drivers sync explicitly).

### Two variants for the sync fix

- **Zero-break (recommended first step)**: gate instead of remove —
  `if sync and not torch.cuda.is_current_stream_capturing():
  torch.cuda.synchronize()`. Eager behavior bit-identical for every existing
  caller (autotune included); capture just works. Cost: keeps the
  per-forward sync overhead in the eager serving path (arguably a
  pre-existing perf bug, separate discussion).
- **Async eager (follow-up)**: keep `run(sync=True)` and public shim
  defaults unchanged; add `sync=` to `mxfp8_mega_moe` / `nvfp4_mega_moe`
  defaulting to `True`; have only the backend `compute()` pass
  `sync=False`. Requires the autotune sync fix and the 2-rank skew stress
  test as its gate.
