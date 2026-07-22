# MoE Routing Test Decomposition — Working Doc

> **Status: work in progress.** Temporary tracking doc for decomposing the
> `tests/moe` combinatorial test matrix. Delete (or move to
> `docs/design_docs/`) before this branch is merged.
>
> Branch: `moe-routing-test-decomposition` (fork `aleozlx/flashinfer`, based on
> upstream main @ `ce29c1a5`). One commit per completed step, so each step is a
> revertible checkpoint.

## Problem

The `tests/moe` suite multiplies routing axes against GEMM/quant axes:
`(routing methods) × (quant impls × shapes × weight layouts × activations × ...)`.

The blow-up is concentrated in the trtllm-gen tests:

| File | Raw combos (pre-skip) | Routing multiplier |
|---|---|---|
| `test_trtllm_gen_fused_moe.py::test_deepseekv3_routing` | ~7,938 | 7 routing configs |
| `test_trtllm_gen_routed_fused_moe.py::test_trtllm_gen_routed_fused_moe` | 3,456 | routing_method[3] × packed/unpacked[2] = 6× on a 576-cell GEMM grid |
| `test_trtllm_gen_fused_moe_routing_renormalize_{bf16,fp4,fp8}.py` | full 9-axis cross-product each | sharded by quant mode only |

cutlass / cute_dsl / b12x tests have **no** routing-method axis (those backends
consume precomputed `topk_ids`/`topk_weights`), so they need no changes.

## Approach

Decompose `(routing methods) × (other params)` into
`(routing methods, dense but cheap) + (few routing methods) × (other params)`:

1. Routing already runs as a distinct kernel inside the fused launcher
   (`Routing::Runner::run` in `csrc/trtllm_fused_moe_kernel_launcher.cu`).
   Expose it standalone through TVM-FFI (precedent: `NoAuxTc`, `hash_topk`,
   `flashinfer_moe_sort`).
2. Test routing math densely against the existing host oracles
   (`routing_reference_*` in `tests/moe/trtllm_gen_fused_moe_utils.py`) —
   routing kernels are tiny, so a dense matrix is cheap.
3. Collapse the routing-method axis in the fused tests to 1–2 representatives,
   plus a small per-(method × launcher) from-logits smoke grid to guard the
   routing→GEMM interface.
4. Keep `test_unified_moe_fuzz.py` as the Monte-Carlo cross-term safety net.

## Steps / checkpoints

Each step lands as its own commit. `[ ]` → `[x]` as they complete.

- [x] **Step 0** — this tracking doc.
- [x] **Step 1 (Phase 1)** — standalone routing FFI op (`trtllm_gen_routing`).
  Code complete; GPU validation happens in Step 3.
  - `Routing::Runner` (the routing dispatcher) moved from
    `csrc/trtllm_fused_moe_runner.cu` into a new TU
    `csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_runner.cu` so it
    can be linked without the batched-GEMM stack.
  - New binding `csrc/trtllm_fused_moe_routing_binding.cu` exporting
    `trtllm_gen_routing` (caller-allocated outputs: packed top-k ids+weights,
    bf16 weights, permutation and CTA bookkeeping tensors).
  - New lightweight JIT module `gen_trtllm_gen_routing_module()` in
    `flashinfer/jit/fused_moe.py` (routing kernels only — no GEMM cubins, so
    the routing test compiles in a fraction of the fused-module time);
    registered in `flashinfer/aot.py`.
  - Python API `flashinfer.fused_moe.trtllm_gen_routing(...)` in
    `flashinfer/fused_moe/trtllm_gen_routing.py`
    (returns `TrtllmGenRoutingResult`), following the `hash_topk` custom-op
    pattern.
  - Notes: `tile_tokens_dim` is an explicit arg (permutation/padding depends
    on it). Routing weights are always bf16 (dispatcher hard-codes output
    dtype). `numTokensPerExpert`/`dtypeElt`/`useRoutingScalesOnInput`/
    `useDeepSeekFp8` are unused by the dispatcher — not exposed.
  - TODO (follow-up): TraceTemplate + `tests/trace/example.py` entry per the
    CLAUDE.md checklist, once the API shape has settled after GPU validation.
- [x] **Step 2 (Phase 2)** — dense routing-only test. Code complete; GPU
  validation happens in Step 3.
  - `tests/moe/test_trtllm_gen_routing.py`: 9 routing methods ×
    {top_k, num_experts, n_group/topk_group, bias dtype, logits dtype,
    token counts incl. edges, tile_tokens_dim, load skew} vs the
    `routing_reference_*` oracles (~2.2k cases, all routing-kernel-only).
  - Selection/weights checked strictly (tie-free positive logits by
    construction — the shared `routing_reference` oracle ranks the masked
    dense weight matrix, which breaks for negative routed weights);
    permutation checked via invariants (round-trip, per-expert padded
    segments, uniqueness) since intra-expert order is not part of the
    contract.
  - Deferred: EP shards (`local_expert_offset > 0`),
    `num_fused_shared_experts > 0`, routing-replay output — follow-ups after
    the base op is validated.
- [x] **Step 3** — validated on B200 (umb-b200-239, sm100a):
  **555/555 passed in 15.9 s** (post-compile; cold JIT of the routing-only
  module is ~10 min vs ~25 min for the full fused stack).
  Two bugs found and fixed on the way:
  - `RoutingMethodType` lives in the nested `Routing` namespace (f0d6c1e4).
  - The kernels emit no expert ids in from-logits mode — `mPtrTopKIds` is
    input-only and `mPtrTopKPacked` is pipeline scratch (confirmed by GPU
    probe). `topk_ids` is now reconstructed from the permutation:
    `cta_idx_xy_to_batch_idx[slot // tile_tokens_dim]` (f3a280a1).
- [~] **Step 4 (Phase 3)** — collapse routing axis in fused tests
  (code done, fused-suite GPU validation pending; see Step 5).
  - [x] `test_trtllm_gen_routed_fused_moe.py`: dense grid pinned to
    Renormalize/packed (3,456 → 576 raw) + new 18-case
    `..._format_parity` smoke over method[3] × format[2] × quant[3].
  - [x] `test_trtllm_gen_fused_moe.py::test_deepseekv3_routing`: dropped
    kimi_k2 / DSLite / GLM4_MoE configs (routing-shape variety moved to the
    standalone routing test, incl. nemotron 512/22) and pruned orphaned
    intermediate sizes (~7.9k → ~3.5k raw; roughly halves the passing
    volume). Kept: DSv3, nemotron_3_super, both fused_shared variants
    (shared-expert fusion isn't covered standalone yet).
  - [x] Renormalize shard files (decision: shrink in place, no file reorg):
    `RENORMALIZE_ROUTING_CONFIGS` now keeps only the Renormalize configs
    (Qwen3_MOE, Qwen3_next, 2048-expert stress) — the method GPT-OSS / Qwen
    route with; DeepSeek's method has its own dense test. Dropped
    Default / SigmoidRenorm / MiniMax2 configs (~64% of the shard files'
    1,398 passing cases per the prior audit): routing math moves to the
    standalone routing test, from-logits launcher plumbing keeps smoke
    coverage via `test_routing_dtype_flexibility` (4 methods × dtypes).
    No new smoke grid needed.
- [ ] **Step 5** — validate the reduced fused matrix on B200; record
  before/after collected-case counts and wall-clock here.
- [ ] **Step 6 (Phase 4)** — follow-ups: fuzz suite visibility in CI, doc
  cleanup, upstream PR.

## Results log

(append measurements / decisions as steps complete)

- 2026-07-21: audit + plan (see PR description / this doc). Branch created.
- 2026-07-21 evening: standalone routing suite green on B200
  (555/555, 15.9 s post-compile).
- 2026-07-21 night — fused validation on B200 (umb-b200-240) + collected-count
  accounting (baseline `ce29c1a5` vs branch, `pytest --collect-only`):

  | file | collected before | collected after | validated on B200 |
  |---|---|---|---|
  | `test_trtllm_gen_routed_fused_moe.py` | 3,568 | **706** (−80%) | 706 passed, 13m56s incl. cold JIT |
  | `test_trtllm_gen_fused_moe.py` | 10,040 | **4,424** (−56%) | `-k deepseekv3_routing` slice passed (exit 0) |
  | renormalize shards (bf16/fp8/fp4) | 1,008 / 1,512 / 1,512 | shrink pending re-measure (configs 7→3) | rerun pending (first attempt raced a stale worktree pull) |
  | `test_trtllm_gen_routing.py` (new) | — | 651 | 555/555 at af314b4e; +96 model-shape dsv3 cases added later, rerun pending |

  Note: collected ≠ executed (runtime `skip_checks` trims further); wall-clock
  is the honest currency — routed file went from "SOLO-class, hours" to ~14 min
  cold.
