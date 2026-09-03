# Engine integration sketches for the unified paged-prefill API

Two reviewable diffs showing what vLLM and sglang delete and gain by moving
their paged-prefill path to `flashinfer.attention.unified` (draft PR #4015).

**Verification status — read this first.**  These diffs have NOT been run
against the live engines.  What IS machine-checked, on real GPUs, is the
data flow they rely on: `tests/attention/test_unified_prefill_engine_shapes.py`
replicates each engine's exact metadata pipeline and drives the unified API
under a zero-sync guard against an independent fp32 oracle.  The diffs were
then adversarially audited against the pinned engine clones; every claim
below survived that audit (several earlier claims did not and were removed).

## Friction found (the honest list — this is the point of the exercise)

Library-side, fixed during the exercise:
- **CSR→dense derivation tail reads (was silent-NaN class)**: cuDNN gathers
  K/V pages by dense-table *width* before masking, so derived rows' tail
  columns are dereferenced.  The derivation now clamps each row's tail to
  the request's own last page (fuzzer regression:
  `csr_overallocated_nan_tail`, found by a NaN-page probe).
- **`window_left < -1` was backend-divergent** (trtllm's launcher
  special-cases exactly -1; -2 became a force-enabled negative window =
  garbage on SM100 only).  Now rejected at plan/resolve.
- fa2/fa3 paged `(192,128)` requires `k_page_stride == v_page_stride`;
  separately-allocated K/V pools violate it → not declared for the fa
  family (cudnn covers it).

Engine-side, to carry in the real PRs:
- **vLLM**: unified's unconditional value-validation needs host mirrors, so
  the all-trtllm async-mode path that today *skips* `seq_lens_cpu`
  retrieval would pay it again (or production relaxes validation to
  maxes-only for mirror-free plans, proposal §3.1).  DCP rewrites
  `seq_lens_cpu` to DCP-local lengths while the GPU `seq_lens` stays
  global — the mirror-consistency contract needs a device-side local-lens
  step, so DCP stays out of the v1 diff.  The `q_data_type` un-quantize
  mutation is shared with decode and must stay until the decode follow-up.
  `logits_soft_cap` / sinks models keep the current path (absent capability
  axes — production should make them rejectable, not just absent).
- **sglang**: no host copy of `paged_kernel_lens` exists at the prefill
  plan site today — the diff adds that plumbing (host is the origin, so it
  is a copy-forward, not a sync).  The radix-extend cascade's paged half is
  OUT of the v1 envelope (it runs `causal=False` with `kv = prefix_lens`,
  which contains ZERO rows for no-prefix requests — the envelope requires
  kv_len >= 1).  `fast_prefill_plan` is CUDA-graph replay machinery and
  stays until unified's capture mode lands.
