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
- [ ] **Step 1 (Phase 1)** — standalone routing FFI op.
  - New binding `trtllm_moe_routing` wrapping `Routing::Runner::run`,
    returning `topk_ids`, `topk_weights` (expert weights),
    `num_tokens_per_expert`, `expanded_idx_to_permuted_idx`,
    `permuted_idx_to_token_idx` (and packed format compatible with the
    `*_routed_moe` ops).
  - Files: `csrc/trtllm_fused_moe_routing_binding.cu` (new),
    `flashinfer/jit/fused_moe.py` (add source),
    `flashinfer/fused_moe/core.py` + `flashinfer/fused_moe/__init__.py`
    (Python wrapper `trtllm_moe_routing`).
  - Note: `Routing::Runner` is constructed with `tile_tokens_dim`; the
    permutation/padding outputs depend on it, so the API takes it as an
    explicit arg.
- [ ] **Step 2 (Phase 2)** — dense routing-only test.
  - `tests/moe/test_trtllm_gen_routing.py`: all routing methods ×
    {top_k, num_experts, n_group/topk_group, routed_scaling, bias,
    logits dtype, token counts incl. edges, tile_tokens_dim} vs
    `routing_reference_*` oracles.
- [ ] **Step 3** — validate Steps 1–2 on a B200 (remote ws1 workflow).
- [ ] **Step 4 (Phase 3)** — collapse routing axis in fused tests.
  - `test_trtllm_gen_routed_fused_moe.py`: routing_method 3→1 on the flagship
    test (3,456 → ~600).
  - `test_trtllm_gen_fused_moe.py::test_deepseekv3_routing`: 7 → 2 routing
    configs (variety moves to the routing-only test).
  - New small per-(method × launcher) from-logits smoke grid (~50–100 cases).
  - Decide fate of the three `routing_renormalize_*` shard files.
- [ ] **Step 5** — validate the reduced fused matrix on B200; record
  before/after collected-case counts and wall-clock here.
- [ ] **Step 6 (Phase 4)** — follow-ups: fuzz suite visibility in CI, doc
  cleanup, upstream PR.

## Results log

(append measurements / decisions as steps complete)

- 2026-07-21: audit + plan (see PR description / this doc). Branch created.
