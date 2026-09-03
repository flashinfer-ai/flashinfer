# MoE Routing Test Decomposition

Why the trtllm-gen MoE test matrices pin their routing-method axis, and where
routing coverage lives instead.

## Problem

`tests/moe` used to multiply routing axes against GEMM/quant axes:

```
(routing methods) × (quant impls × shapes × weight layouts × activations × ...)
```

Routing correctness is orthogonal to quantization — they share no kernel code
path — so that product bought almost nothing per case while dominating the
suite's wall-clock. The blow-up was concentrated in the trtllm-gen files;
`cutlass`/`cute_dsl`/`b12x` tests have **no** routing-method axis (those
backends consume precomputed `topk_ids`/`topk_weights`), so they were already
fine.

## Decomposition

Split `(routing methods) × (other params)` into
`(routing methods, dense but cheap) + (few routing methods) × (other params)`:

1. Routing runs as a distinct kernel inside the fused launcher
   (`Routing::Runner::run`). It is exposed standalone through TVM-FFI as
   `flashinfer.fused_moe.trtllm_gen_routing` (precedent: `NoAuxTc`,
   `hash_topk`, `flashinfer_moe_sort`), backed by a lightweight JIT module
   with the routing kernels only — no batched-GEMM stack, so it compiles in a
   fraction of the fused module's time.
2. Routing math is tested densely against the existing host oracles
   (`routing_reference_*` in `tests/moe/trtllm_gen_fused_moe_utils.py`) in
   `tests/moe/test_trtllm_gen_routing.py`. Routing kernels are tiny, so a
   dense matrix here is cheap.
3. The fused tests pin the routing-method axis to one or two representatives,
   plus a small per-(method × launcher) from-logits smoke grid that guards the
   routing→GEMM interface.
4. `tests/moe/test_unified_moe_fuzz.py` remains the Monte-Carlo cross-term
   safety net.

## Where to add coverage

| New coverage for… | Goes in |
|---|---|
| a routing method, `top_k`/`num_experts`/group shape, logits or bias dtype, `tile_tokens_dim`, load skew, expert-parallel shard | `tests/moe/test_trtllm_gen_routing.py` |
| a quant mode, weight layout, activation, GEMM shape | the fused matrices, at the **pinned** routing method |
| the routing→GEMM interface for a method that has no standalone launcher coverage | the from-logits smoke grids (`..._format_parity`, `test_routing_dtype_flexibility`) |

Adding a routing method back into a fused matrix re-introduces the
cross-product this decomposition removed; prefer the standalone file.

## Standalone routing op — design notes

- `tile_tokens_dim` is an explicit argument: the permutation and padding
  outputs depend on the token-tile size the downstream grouped GEMM would use.
- Routing weights are always `bfloat16` — `Routing::Runner` hard-codes
  `mDtypeOutput = Bfloat16` for every method, regardless of the logits dtype.
- The kernels emit **no** expert ids in from-logits mode: `mPtrTopKIds` is
  input-only and `mPtrTopKPacked` is pipeline scratch (confirmed by GPU probe).
  `topk_ids` is reconstructed from the permutation —
  `cta_idx_xy_to_batch_idx[slot // tile_tokens_dim] + local_expert_offset` —
  with `-1` for slots whose expert falls outside the local expert-parallel
  shard.
- `numTokensPerExpert` / `dtypeElt` / `useRoutingScalesOnInput` /
  `useDeepSeekFp8` are unused by the routing dispatcher and are not exposed.
- Group parameters (`n_group`/`topk_group`) are validated in the binding,
  mirroring `FusedMoeLauncher::check_routing()`: the dispatcher itself only
  guards `top_k <= 22` and `topk_group <= 4`, so an unchecked combination
  would reach the kernel as a device-side fault rather than a clean error.
- Test construction: logits are positive and tie-free by construction, because
  the shared `routing_reference` oracle ranks the masked dense weight matrix
  (zero entries would outrank negative routed weights for TopK/Sigmoid
  methods). The permutation is checked by invariant — round-trip through
  `permuted_idx_to_token_idx`, per-expert padded segments, uniqueness — since
  the kernel's ordering *within* an expert's padded segment is not part of the
  contract.

## Measured impact

Collected counts, `pytest --collect-only`, baseline = merge base `46fc99b9`
(upstream main as of the 2026-08-21 merge) vs this branch. Note that main
independently trimmed shape axes in the routed file, so the two reductions
compound — these numbers are smaller than the ones measured against the
original `ce29c1a5` baseline.

| file | before | after | delta |
|---|---|---|---|
| `test_trtllm_gen_routed_fused_moe.py` | 466 | **364** | −22% |
| … of which the dense grid `test_trtllm_gen_routed_fused_moe` | 144 | **24** | −83% |
| `test_trtllm_gen_fused_moe.py` | 8,173 | **4,141** | −49% |
| renormalize shards (bf16 + fp8 + fp4) | 2,688 | **1,152** | −57% |
| **subtotal** | **11,327** | **5,657** | **−50%** |
| `test_trtllm_gen_routing.py` (new, routing-kernel only) | — | 688 | — |

Collected ≠ executed (runtime `skip_checks` trims further), and wall-clock is
the honest currency. On B200 (sm100a):

| file | B200 result |
|---|---|
| `test_trtllm_gen_routed_fused_moe.py` | 0 failures; hours-class → ~14 min including cold JIT |
| `test_trtllm_gen_fused_moe.py` (deepseekv3 slice) | 69 executed / 0 failed, ~22 min |
| renormalize shards (3 files) | 201 executed / 0 failed, ~93 min total incl. cold JIT + autotune (pre-shrink: 1,398 executed, ~123 min) |
| `test_trtllm_gen_routing.py` | ~16 s warm; ~11 min including cold routing-module JIT |

Executed-volume cut on the shards: 1,398 → 201 (−86%).

## What the fused matrices kept

- `test_trtllm_gen_routed_fused_moe.py`: dense grid pinned to
  Renormalize/packed, plus an 18-case `..._format_parity` smoke over
  method[3] × format[2] × quant[3]. Routing here is host-precomputed, so the
  method axis only varied reference math — which is now tested directly.
- `test_trtllm_gen_fused_moe.py::test_deepseekv3_routing`: kept `DSv3`,
  `nemotron_3_super`, and both fused-shared-expert variants (that fusion has no
  standalone coverage yet). Dropped `kimi_k2` / `DSLite` / `GLM4_MoE` and
  orphaned intermediate sizes — those routing shapes moved to the standalone
  test, including nemotron 512 experts / top_k 22.
- Renormalize shard trio: `RENORMALIZE_ROUTING_CONFIGS` keeps only the
  Renormalize configs (the method GPT-OSS / Qwen3 / Qwen3-Next / Mixtral use).
  Default / SigmoidRenorm / MiniMax2 from-logits plumbing keeps smoke coverage
  via `test_routing_dtype_flexibility`.

## Follow-ups

- `num_fused_shared_experts > 0` and routing-replay output are not yet covered
  in the standalone routing test.
- The from-logits grids could thin further once an fp8-per-tensor routed entry
  point exists; it currently has no pre-routed counterpart.
