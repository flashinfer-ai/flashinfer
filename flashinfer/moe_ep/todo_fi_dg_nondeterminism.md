# TODO: fi_dg (deep_gemm_mega) run-to-run nondeterminism

Status: analysis only, nothing implemented yet (2026-07-16). Found during the
2026-07-15 vLLM 0.25.1 e2e integration (DeepSeek-V4-Flash, 4x GB200, TP4+EP4,
eager). Full run log: `$LUSTRE_ROOT/moe_ep_benchmark/vllm_e2e/RUNS.md`
(runs 3/4/11) and `FINDINGS.md` there.

## Symptom

Greedy decoding, 64 tokens x 8 prompts, per-token logprobs:

- native vLLM `deep_gemm_mega_moe` vs its own rerun: **8/8 bit-exact**.
- fi moe_ep `deep_gemm_mega` vs its own rerun: **2/8 exact**, mean
  |dlogprob| 0.01-0.08/tok on the divergent prompts.
- fi_dg vs native: 3/8 exact, |dlp| 0.01-0.06 — statistically at parity, so
  this is a determinism bug, not a correctness bug. Unchanged by the
  fast-path + shared-workspace wrapper (run 11).

## What has been ruled out (2026-07-15 evidence)

- **Staging**: proven bit-identical — vLLM's fused
  `_prepare_megamoe_inputs_kernel` output vs fi's `stage_inputs`
  (`backends/mega/kernel/deep_gemm_mega/staging.py`, `per_token_cast_to_fp8`
  + copies) compared tensor-by-tensor, exact match.
- **Kernel arguments**: the deep_gemm mega launch was argument-identical
  between native and fi paths.

Same kernel, same args, same staged bytes → the divergence must enter through
state the kernel reads that is NOT an argument, or through cross-rank timing:

1. **Symmetric/workspace buffer residual state** — native vLLM manages symm
   buffers via a class-level `_symm_buffer_cache`; fi allocates its own
   workspace (`modes/mega_layer.py::_ensure_workspace` →
   `prepare_workspace`). Different zeroing/reset discipline or residual
   garbage in pad regions could feed the kernel differing bits.
2. **Cross-rank arrival order** — if the mega kernel's combine/reduction
   accumulates in arrival order (atomics), rank skew changes summation
   order. But native uses the same kernel and is deterministic, so this only
   works as an explanation if fi's launch pattern (extra staging launches,
   per-forward sync) produces materially different skew.
3. **Allocator address sensitivity** — fresh per-step tensor addresses on
   the fi path vs native's reused buffers, if deep_gemm dispatches on
   alignment/address.

## Debug plan

1. Single-rank repro first (`MEGA_NO_DIST=1`, fixed seed, same input twice
   through `MoEEpMegaLayer.forward`): if nondeterminism persists at world
   size 1, cross-rank ordering (hypothesis 2) is out and it's workspace or
   allocator state.
2. Checksum the full workspace (symm buffers included) immediately before
   each launch across reruns — any delta pinpoints the residual-state path.
3. If single-rank is clean: 2-rank rerun pairs with forced skew (one rank
   sleeps) to see if divergence rate tracks skew.
4. Bisect the wrapper differences vs native (workspace provenance, staging
   kernel count, sync placement) one at a time on the vLLM side.

The e2e comparison harness already exists:
`moe_ep_benchmark/vllm_e2e/compare_outputs.py` + `smoke_infer.py`.

Related: [[todo_cuda_graph]] — graph capture would freeze launch timing and
addresses, which may mask (not fix) this; determinism should be understood
first so graph replay doesn't bake in a wrong conclusion.
