# Plan: import speedup ideas from the TRT-LLM MegaMoE integration

> STATUS 2026-07-14: families 1 and 2a LANDED (see TUNING.md "TRT-LLM-import
> knobs"). `in_kernel_fc2_reduce` + `combine_dtype` are plumbed through
> `get_symm_buffer_for_mega_moe` and `Nvfp4CutedslMegaMoeConfig`; the MXFP8
> twin's rank-local-output latent bug is fixed (output is now always
> sym-heap); `frontend.run()` / launch thunks enqueue the ikr
> `output.zero_()`; NVFP4 autotune sweeps ikr (24 candidates); multirank
> tests cover ikr (tolerance + repeat-launch zero guard) and both quantized
> wires (bit-exact vs same-wire reference + rel-L2 band vs bf16).
> Remaining: 2b streaming weight reload (on demand), 2d CUDA-graph capture
> (`todo_cuda_graph.md`).

Source: TRT-LLM PR #16190 (`MEGAMOE_CUTEDSL` backend for DeepSeek-V4 DEP
serving, Blackwell) — same kernel drop we vendor.  The TRT-LLM team reports
all three combine wire formats beating deep_gemm.  Two idea families to
import: **in-flight reduction** and **unification** (combine wire formats +
lifecycle).

## 1. In-flight combine reduction (`in_kernel_fc2_reduce=True`)

What: REDG atomic-add collapses the top-k combine **in flight** as peer
combine data arrives, instead of staging the per-topk `(T, K, H)` tensor in
`shared_workspace` and running the in-`__call__` K-ordered fp32 tail reduce.

Measured expectation (our tester reference sweep, 7168/3072/384/top-6, bf16
combine, max-across-ranks median): the OVERALL best candidate at both 8 and
2048 tokens was an in-flight-reduce candidate — 285.7 vs 289.1 us @8,
676.7 vs 692.2 us @2048 (~1-2%).  Likely somewhat larger at our top-8
geometry (tail reduce is K-proportional).  The second benefit may matter
more: the internal combine staging is the multi-GB part of
`shared_workspace`, and ikr removes it -> big symmetric-heap footprint
reduction.

Constraints (kernel/config validation, tester `filter_invalid`):
- requires `apply_topk_in_fc1=True` (our default) — topk weights folded pre-fc2
- bf16 combine only (quantized wire formats use the explicit reduce path)
- `output_activation` becomes the cross-rank REDG atomic target ->
  MUST live on the symmetric heap and be ZEROED before every launch
  (accumulate-from-zero contract)
- nondeterministic accumulation order -> tests need a tolerance verdict
  (the tester bounds it against the pre-reduce terms; exact-compare is out)

Work items:
1. `get_symm_buffer_for_mega_moe`: expose `in_kernel_fc2_reduce`, allocate
   `output_activation` via `sym_zeros` when set (track in `_sym_roots`).
   NOTE: the MXFP8 twin already takes the param but still allocates the
   output rank-locally — latent bug if anyone enables it; fix both.
2. `frontend.run()`: enqueue `output_activation.zero_()` before the launch
   when `config.fc2_reduces_topk` (stream-ordered, ~10 us at 2048 tokens).
3. `Nvfp4CutedslMegaMoeConfig` gains `in_kernel_fc2_reduce` (mxfp8's has it).
4. Multirank test variant with tolerance-based compare vs the plain-sum
   reference (mirror the tester's nondet verdict).
5. Autotune: add `in_kernel_fc2_reduce` to the candidate space once the
   buffer is allocated ikr-capable (sym output serves both modes; the knob
   then flips per-compile).

## 2. Unification

Candidate readings from the PR (confirm which the TRT-LLM folks mean):

a. **Combine wire formats as a unified config knob** (bf16 default,
   `32e4m3xe8m0`, `16e2m1xbf16`): the cross-rank combine traffic shrinks
   2x/4x.  All three reportedly beat dg.  FI status: the shim's NVFP4
   config already has `combine_dtype` + validation, and `tuner.is_valid`
   knows the quantized-combine constraints, but neither
   `get_symm_buffer_for_mega_moe` nor the FI backend config exposes it.
   Plumbing + an accuracy check (wire quantization is a numerics tradeoff);
   our tester reference problems (nvfp4_perf.jsonl) include both quantized
   variants for validation.  Mutually exclusive with ikr (see constraints).
b. **Streaming weight load/reload lifecycle** with partial-group coverage
   tracking — serving-oriented live weight updates; import only if the FI
   serving integration needs reload.
c. **Workspace/fence lifecycle keyed on symmetric-buffer identity** — we
   already have the equivalent (launch-kwargs cache keyed on buffer
   pointers; no-reset default relying on kernel tail cleanup).
d. **CUDA-graph capture-safe integration** — pairs with our
   `todo_cuda_graph.md`; the launch thunk (single prebuilt kernel launch,
   stable pointers) is the graphable unit; ikr adds a graphable
   `output.zero_()` node before it.

## Proposed order

1. Combine wire formats (`16e2m1xbf16` first): biggest expected win at
   large tokens / multi-node (4x less NVLink combine traffic), pure
   plumbing on our side, reference data already exists.  Needs an
   accuracy-budget decision.
2. `in_kernel_fc2_reduce`: 1-2% latency + large shared-workspace memory
   saving; also completes the autotune space (the tester's overall winners
   are ikr candidates).
3. CUDA-graph capture of the launch thunk: removes the from-idle launch
   penalty (~85 us measured barrier-cold); TRT-LLM's capture-safe design is
   evidence of feasibility.
4. Streaming weight reload: on demand.
