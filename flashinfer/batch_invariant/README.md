# Batch-invariant kernels (`flashinfer.batch_invariant`)

`flashinfer.batch_invariant` provides deterministic, batch-invariant implementations of FlashInfer operators, selected
with the keyword `batch_invariant=True` on the operator's public API. Generated implementations use an
engine that fixes the order of every floating-point reduction and the exact instructions the kernels issue, tensor-core
instructions included; the reduction orders are proved in Lean and the generated kernels are checked bit for bit on the
device.

## The guarantee

With `batch_invariant=True` an output's bits depend on the input values and the fixed dimensions alone:

| | Property | Meaning |
|---|---|---|
| R1 | Run-to-run determinism | same inputs, same path: same bits, across processes, streams and CUDA-graph replay |
| R2 | Batch invariance | a row's bits do not depend on batch size, padding, position, neighbours or SM count |
| R3 | Configuration agreement | every configuration the dispatcher or the autotuner may pick gives the same bits |
| R4 | Cross-path agreement | the members of one class (for example the operand layouts an operator accepts, or two implementations of one reduction) give the same bits |

Each class has one fixed dataflow, named and hashed into a **fingerprint**; `flashinfer.batch_invariant.fingerprint(op,
**dims)` returns it, scoped to (fingerprint, FlashInfer version, architecture). Bitwise equality is the acceptance
criterion of every test of the mode (`tests/gemm/test_batch_invariant_mm_bf16.py`, `tests/gemm/test_batch_invariant_bmm_fp8.py`, `tests/gemm/test_batch_invariant_gemm_fp8_blockscale.py`, `tests/gemm/test_batch_invariant_mm_mxfp8.py`, `tests/gemm/test_batch_invariant_mm_fp4.py`,
`tests/gemm/test_batch_invariant_mm_mxfp8_mxfp4.py`, `tests/moe/test_unified_moe_batch_invariant.py`, `tests/batch_invariant/test_lean.py`).

The SM100/SM103/SM107 BF16 MegaMoE class additionally has an ordered arithmetic spec at
`moe_ep/batch_invariant/spec.py`, a shared pinned activation chain, the Lean
model `lean/BIKernels/MegaMoE.lean` and independent device conformance. It
reuses the existing MegaMoE kernel rather than the GEMM generation engine.
Its receipt also fixes the compiler stack, EP topology and expert placement;
its model does not prove CUDA synchronization or remote communication.

BF16 MegaMoE admits `M=128/256`, `N=128/256` and `K=64/128/256` tiles.
M128 uses a single CTA and cluster `(1,1,1)`; M256 uses paired CTAs and
cluster `(2,1,1)`. A larger K tile is admitted only when both
`round_up(hidden,64)` and `round_up(intermediate,64)` are divisible by K.
This preserves every K16 MMA step, including padding and the initial
non-accumulating instruction. Plans must also fit shared memory with at least
one A/B pipeline stage. Online autotuning filters these constraints before
profiling the geometry, scheduler and token-back combinations. The arithmetic
specification and existing fingerprints stay unchanged across admitted tiles.
SM107 uses the same BF16 backend with native `sm_107a` compilation and its
334848-byte oversized shared-memory budget; SM100/SM103 use 232448 bytes. The
SM107 CUDA/CuTe stack must support oversized shared-memory launches, distinct
from the driver's standard opt-in limit. Architecture
is part of the receipt, so these classes do not promise matching bits across
architectures. A previously captured compatibility target is rejected on SM107
before compiled-kernel workspace allocation.

Multi-GPU SM107 NVLink execution also requires a Rubin-aware NCCL;
[NCCL 2.32.3 introduced this support](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-32-3.html).
Earlier NCCL versions can fail during communicator initialization on Rubin
systems with 36 NVLink ports per GPU, before launching MegaMoE.

## Supported operators

| API | Selection | Classes and fingerprint | Supported inputs |
|---|---|---|---|
| `MoEEpMegaLayer` with `Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig` | `batch_invariant=True` on the kernel config | `fingerprint("megamoe_bf16", hidden=, intermediate=, top_k=, num_experts=, apply_routing_weights_before_fc2=True, gate_up_clamp=None, arch="sm_100")`; ordered padded K16 MMA folds, pinned approximate SwiGLU, BF16 handoff/wire and slot-order sum or FMA. Early/late weight placement are different classes; all admitted schedules of one class agree. `layer.batch_invariant_fingerprint` returns the receipt. | BF16, precomputed routing; SM100 / SM103 / SM107 (native `sm_107a`); fixed EP topology and expert placement. `hidden % 32 == 0`, `intermediate % 64 == 0`, `1 <= top_k <= 32`. Finalized output only. Router computation is outside this implementation. Outer `AutoConfig` complete-EP selection admits finalized Mega candidates with the same fingerprint. |
| `MoEEpMegaLayer` with the SM100 or SM107 NVFP4 config | `batch_invariant=True` on the kernel config | `fingerprint("megamoe_nvfp4", hidden=, intermediate=, top_k=, num_experts=, arch=, implementation=, input_norm_const=, apply_routing_weights_before_fc2=, gate_up_clamp=)`; ordered native K64 (SM100/SM103) or K128 (SM107) folds, specified activation/block-16 quantization and ordered combine. Architecture, implementation (`legacy`, `standard`, `genphase`), routing-weight placement, activation/output recipe and clamp define separate classes. `layer.batch_invariant_fingerprint` returns the actual receipt. | NVFP4 packed inputs and weights, or aligned contiguous BF16 inputs through the pinned fused stager; SM100 / SM103 / SM107. Fixed geometry allowlist; custom SwiGLU/SiTU, quantized combine and deferred output have provider-specific limits. Fixed EP topology, weights/scales and expert placement. Public omitted per-expert scales resolve to ones. See the [NVFP4 mode contract](../../docs/design_docs/moe_ep_architecture.md#batch-invariant-nvfp4-megamoe). |
| `MoEEpMegaLayer` with an MXFP4-weight config | `batch_invariant=True` on the kernel config | `fingerprint("megamoe_mxfp4", hidden=, intermediate=, top_k=, num_experts=, arch=, quant_kind=, implementation=, apply_routing_weights_before_fc2=, gate_up_clamp=)`; UE8M0/block32 quantization, specified activation, ordered native MMA and combine. Architecture, pure/mixed format, standard/GenPhase and early/late placement have separate receipts. | SM100/SM103: MXFP8 E4M3 activations with MXFP4 weights; SM107: also pure MXFP4. H/I multiples of 128, fixed geometry/flag allowlist. Custom SwiGLU/SiTU, quantized combine and deferred output have provider-specific limits. No global alpha/norm tensors. See the [MXFP4 mode contract](../../docs/design_docs/moe_ep_architecture.md#batch-invariant-mxfp4-megamoe). |
| `mm_bf16` | `backend="auto", batch_invariant=True` | `gemm.bf16.kfold` (one fp32 fold over the K steps) or `gemm.bf16.kchunk[C]` (K cut into `C` chunks, a function of the weight shape alone, so a narrow weight runs it at every M); `fingerprint("mm_bf16", N=, K=)`, and with a bias `fingerprint("mm_bf16", N=, K=, bias=True)`: the biased chain `cvt_bf16(add_f32(fold, bias[n]))` (the bf16 bias widened exactly and added once to the fully reduced accumulator before the single rounding) is a different specification | bf16 `a (M, K)` row-major and bf16 `b (K, N)` row-major or the transpose of a row-major `(N, K)` weight, `K` and `N` multiples of 8, any `M`; optional bf16 contiguous `bias (N,)`; bf16 output; `pdl` accepted (plain stream order). SM100 / SM103 / SM107 |
| `bmm_fp8` | `backend="auto", batch_invariant=True` | `gemm.e4m3.kfold.scaled` / `gemm.e5m2.kfold.scaled` (one fp32 fold over the 32-wide fp8 K steps, then `mul.rn` by `A_scale * B_scale` formed once on the device, then one `cvt.rn`) or `gemm.<fmt>.kchunk[C].scaled` (K cut into `C` chunks of whole 32-wide steps, a function of the weight shape and the format alone); every batch element is one launch of the class, so the batch is not a dimension; `fingerprint("bmm_fp8", N=, K=, dtype=, out_dtype=)` | `A (b, M, K)` and `B (b, K, N)` both e4m3 or both e5m2, `B` column-major (the transpose of a row-major `(b, N, K)` weight, the documented layout) or row-major, `K` a multiple of 16 and `N` of 8, any `b` and `M`; the two per-tensor scales as 1-element fp32 tensors; bf16 or fp16 output. SM100 / SM103 / SM107 |
| `gemm_fp8_nt_groupwise` | `backend="auto", batch_invariant=True` | `gemm.e4m3.kblock128`: the DeepSeek-style block-scale class -- each 128-element K block is one fp32 fold over its four 32-wide fp8 steps from +0 (the block partial), and the block partials are folded in block order from +0 with `fma.rn(mul.rn(a_scale[m, kb], b_scale[n // 128, kb]), P_kb, acc)` (the two fp32 block scales multiplied once on the device), then one `cvt.rn`; no split variant (the block fold already partitions K), so `N` is not a dimension; `fingerprint("gemm_fp8_nt_groupwise", K=, out_dtype=)` | e4m3 `a (M, K)` and e4m3 `b (N, K)` row-major (the operator's NT weight, K contiguous), `K` a multiple of 128 (whole scale blocks) and `N` of 8, any `M`; `scale_granularity_mnk = (1, 128, 128)`; the fp32 scales `a_scale (M, K / 128)` and `b_scale (ceil(N / 128), K / 128)` in the `K` layout or their transposes in the `MN` layout (`scale_major_mode`; two stagings of the `K` layout's scales, the same fold: up to 64 rows with 16-byte-aligned scale bases the kernel reads the tensors as they are, four blocks of a row's scales per copy; otherwise `a_scale` is copied once to the MN layout inside the API, a layout copy, no arithmetic); `mma_sm` accepted and ignored (a placement of the backends); bf16 or fp16 output. SM100 / SM103 / SM107 |
| `mm_mxfp8` | `backend="auto", batch_invariant=True` | `gemm.mxfp8.kfold` (one fp32 fold over the 32-wide block-scaled steps of the SM100 / SM103 instruction, the UE8M0 block scales applied inside each step, then one `cvt.rn`) or `gemm.mxfp8.kchunk[C]`, and on SM107 `gemm.mxfp8.kfold64` / `gemm.mxfp8.kchunk64[C]` (the SM107 instruction folds 64 elements per step: a different class, so the architecture is a dimension of the specification); `C` is a function of the weight shape alone (chunks of whole 128-element granules); `fingerprint("mm_mxfp8", N=, K=, out_dtype=, arch=)` (`arch` defaults to the current device) | e4m3 `a (M, K)` row-major and e4m3 `b (K, N)` the transpose of a row-major `(N, K)` weight (the block-scaled instruction takes K-major operands only), `K` a multiple of 32 and `N` of 8 within the operator's own limits (`N, K >= 128`), any `M`; the scale factors as the 1-D uint8 buffers of `mxfp8_quantize(..., is_sf_swizzled_layout=True)` (the 128x4 swizzled layout; where K is not a multiple of the class's step the last step's scales are read from the buffers' padding, which must be finite, as the quantizers write it); bf16 or fp16 output. SM100 / SM103 / SM107 |
| `mm_fp4` | `backend="auto", batch_invariant=True` | NVFP4 (`use_nvfp4=True`): `gemm.nvfp4.kfold` (one fp32 fold over the 64-wide block-scaled steps of the SM100 / SM103 instruction, the e4m3 block scales of 16-element blocks applied inside each step, then one `mul.rn` by the caller's `alpha` and one `cvt.rn`) or `gemm.nvfp4.kchunk[C]`, and on SM107 `gemm.nvfp4.kfold128` for the weight shapes the chunk rule leaves unsplit (the SM107 instruction folds 128 elements per step; a split weight shape runs `gemm.nvfp4.kchunk[C]` with the 64-wide step on SM107 too); MXFP4 (`use_nvfp4=False`): `gemm.mxfp4.kfold` / `gemm.mxfp4.kchunk[C]` (the UE8M0 block scales of 32-element blocks), and on SM107 `gemm.mxfp4.kfold128` / `gemm.mxfp4.kchunk128[C]`; `C` is a function of the weight shape and the format alone (chunks of whole granules: 128 elements for the 64-wide step, 256 for the 128-wide); `fingerprint("mm_fp4", N=, K=, out_dtype=, use_nvfp4=, arch=)` (`arch` defaults to the current device) | packed-fp4 `a (M, K / 2)` row-major and `b (K / 2, N)` the transpose of a row-major `(N, K / 2)` weight (uint8 or `float4_e2m1fn_x2`; the block-scaled instruction takes K-major operands only), `K` a multiple of 32 and `N` of 8, any `M`; the scale factors as the 1-D swizzled buffers of `nvfp4_quantize(..., sfLayout=SfLayout.layout_128x4, do_shuffle=False)` / `mxfp4_quantize(...)` (the 128x4 layout; e4m3 or uint8 bytes); `alpha` a 1-element fp32 tensor or None (1.0; a per-token alpha is not a class of the mode); where K is not a multiple of the class's step (64; 128 on SM107) the last step's scales are read from the buffers' padding, which must be finite, as the quantizers write it; bf16 or fp16 output. SM100 / SM103 / SM107 |
| `mm_mxfp8_mxfp4` | `batch_invariant=True`; `batch_invariant=False` (the default) runs the same class today, since FlashInfer has no other dense kernel for e2m1 weights against e4m3 activations (its `tgv_gemm_sm100` low-latency path takes the pair for a few rows only) -- a future non-invariant kernel may take the `False` path, `True` keeps the guarantee | the mixed MX classes: `gemm.mxfp4w8a.kfold` (one fp32 fold over the 32-wide steps of the SM100 / SM103 `kind::mxf8f6f4` instruction with mixed operand formats -- e4m3 A against e2m1 B, one UE8M0 scale per 32 elements on each, applied inside the step -- then one `mul.rn` by the caller's `alpha` and one `cvt.rn`) or `gemm.mxfp4w8a.kchunk[C]`, and on SM107 `gemm.mxfp4w8a.kfold64` / `gemm.mxfp4w8a.kchunk64[C]` (the SM107 instruction folds 64 elements per step: a different class, so the architecture is a dimension); the mixed step is its own fold -- the MXFP8 class on the e2m1 values re-coded to e4m3 gives different bits (pinned as an observation in the suite), so neither `mm_mxfp8` nor any FlashInfer backend is an oracle of it; `C` is a function of the weight shape and the format alone: the MX rule (chunks of whole 128-element granules) with the mixed classes' lower bound of 8 chunks for `N <= 512`, `K >= 16384` (K a multiple of 1024; constants declared empirical, the same on every part), so weights in that region carry `kchunk[8]` fingerprints; `fingerprint("mm_mxfp8_mxfp4", N=, K=, out_dtype=, arch=)` (`arch` defaults to the current device) | e4m3 `a (M, K)` row-major (`mxfp8_quantize`'s output) and the packed-e2m1 weight `b (K / 2, N)` as the transpose of a row-major `(N, K / 2)` tensor (`mxfp4_quantize`'s output; uint8 or `float4_e2m1fn_x2`; the block-scaled instruction takes K-major operands only), `K` a multiple of 32 and `N` of 8, any `M`; **on SM100 / SM103 `K` a multiple of 128** (the instruction reads the e2m1 operand through the TMA's unpacking format over whole 64-byte runs of the row; smaller K is rejected, not padded); the scale factors as the quantizers' 1-D swizzled uint8 buffers (the 128x4 layout; on SM107, where K is not a multiple of 64, the last step's scales are read from the buffers' padding, which must be finite); `alpha` a 1-element fp32 tensor or None (1.0); bf16 or fp16 output; `enable_pdl=None` follows the device (programmatic dependent launch, a placement: the same bits). SM100 / SM103 / SM107 |
| `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper` | `backend="auto", batch_invariant=True` (constructor) | `attention.bf16` per head dim and mask mode: 128-key tiles and 512-key chunks on the absolute KV axis, the online softmax step and the chunk merge of `attention/batch_invariant/spec.py`; `fingerprint(<wrapper name>, head_dim=..., causal=True)` for the causal class (a token at KV position `p` sees the keys `[0, p]`, or `[max(0, p - window_left), p]` with a window, `window_left >= 0`) and `fingerprint(<wrapper name>, head_dim=..., causal=False)` for the non-causal class (a token sees every key of its request, `[0, kv_len)`; the same dataflow over a different visible set, so a different term and fingerprint). `causal` is a required dimension of the two prefill wrappers' receipt, to be given as the value passed to `plan()`: their `plan()` defaults to `causal=False`, so a plan that omits `causal` runs the non-causal class and `causal=True` selects the causal one; for the decode wrapper it is optional and an omitted or `None` `causal` is the causal class, as `is_causal=None` is in its `plan()`. The three wrappers (prefill and decode, paged and ragged KV) compute one class per mode: a token's bits depend on its KV position (causal) or its request's key count (non-causal), its head and the window alone. With `return_lse=True` they also return the class's LSE member, a second output of the row's final maximum `m` (raw score units) and sum `l`: `lse2 = add.rn(mul.rn(m, c), lg2.approx.ftz(l))` with `c = fp32(sm_scale * log2 e)` (base 2, `lse_base="log2"`) and `mul.rn(lse2, fp32(ln 2))` (`lse_base="ln"`, computed in the kernel), fp32; its receipt is its own, `fingerprint(<wrapper name>, head_dim=..., causal=..., lse_base=...)`, and the output's receipt and bits are the same with and without it. With attention sinks (`sinks`, per-head logits) they run the class's sink member: one more merge per row, after its last chunk merge and before the epilogue, of the final `(m, l)` with a virtual key of logit `sinks[h]` (natural-log units of the scaled score: `sm_scale` is not applied to it) and zero value, in base-2 units with `LOG2E = fp32(log2 e)`: `ms = mul.rn(m, c)`, `s2 = mul.rn(sinks[h], LOG2E)`, `m' = max(ms, s2)`, `a = ex2(add.rn(ms, -m'))`, `b = ex2(add.rn(s2, -m'))`, `l' = fma.rn(l, a, b)`; the output is `cvt_bf16(add.rn(mul.rn(O, mul.rn(a, div.rn(1, l'))), +0))` and the LSE `add.rn(m', lg2.approx.ftz(l'))` (times `fp32(ln 2)` in base e), so the LSE includes the sink. The step is taken once per row whatever the tiling, the unit kind, the kernel body and prefill or decode; a call without sinks does not take it and keeps its bits and receipts, and the sink member has its own, `fingerprint(<wrapper name>, head_dim=..., causal=..., sinks=True)` and `fingerprint(<wrapper name>, head_dim=..., causal=..., sinks=True, lse_base=...)`. Both members exist in both mask modes: they read a row's final `(m, l)`, which the non-causal class forms by the same fold over `[0, kv_len)`, so under `causal=False` a row's LSE depends on its request's key count, its head and its own query alone and the sink joins that key set; the receipts carry the mode before the member (`attn-bf16-d<D>-nc-lse-<base>-...`, `attn-bf16-d<D>-nc-sink-...`, `attn-bf16-d<D>-nc-sink-lse-<base>-...`) | bf16 q / K / V / out; head dim 64 / 128 / 256; GQA (`num_qo_heads` a multiple of `num_kv_heads`, up to 128 per group); causal (bottom-right aligned) with an optional `window_left`, or non-causal (prefill `causal=False`; decode `q_len_per_req > 1` with `is_causal=False`; no window, any `q_len` against at least one key; a single-token decode step is the causal class either way; chunked prefill reproduces the one-shot bits only when each query chunk is planned over the request's full key set, since a token's term is a function of `kv_len`); page sizes 1..128 (powers of two); NHD / HND. CUDA graphs: capture `run()` after `plan()`; a replay reproduces the planned batch and a new `plan()` needs a new capture (`use_cuda_graph=True` raises). `plan()` runs a host-side planner (of the order of 10 us per request). LSE output: `run(..., return_lse=True)` / `run_return_lse` give `(out, lse)` with `lse` fp32 `(total_q, num_qo_heads)` (an `lse=` buffer is written in place); the prefill wrappers take `lse_base` (`"log2"` default, `"ln"`) and `lse_layout` (`"NH"`, or `"HN"`: the same bits transposed), the decode wrapper returns base 2. Every row has at least one visible key (`kv_len >= q_len` is required under the causal class, at least one key per request with query tokens under the non-causal class), so no LSE is that of an empty row (the term's value there would be `-inf`). Attention sinks: `run(..., sinks=)` on the paged wrappers and the `_sinks` attribute on the ragged one, a contiguous fp32 `[num_qo_heads]` tensor on the device (other dtypes, shapes and devices are refused), read in place at every launch (a captured graph replays with the tensor's current values); with a sliding window the sink joins the window's keys, under `causal=False` the request's keys. `plan(..., has_sinks=True)` or `plan(..., has_sinks=False)` (the same in both mask modes) declares that every `run()` of the plan is, or is not, given sinks, and a run that departs from the declaration raises; under the default `None` each run decides, so a caller that passes sinks to one wrapper and not to another runs two different members without an error. The check is made in Python at `run()`: a captured graph is unaffected. A sink of `-inf` gives the bits of the call without sinks; a finite sink far above the row's scores gives `+0` outputs as long as its base-2 value `mul.rn(sinks[h], LOG2E)` is finite; a `+inf` or NaN sink gives NaN on its head's rows (not guarded, like non-finite operands). A finite sink whose base-2 value overflows binary32 acts as the sink of that infinity: above about `2.36e38` the step sees `+inf` and the head's rows are NaN; below about `-2.36e38` it sees `-inf` and the rows keep the bits of the call without sinks. Not: soft cap, custom masks, fused RoPE, a window with `causal=False`, fp8 / fp4 KV, q / k / v scales |
| `MoELayer` with `BatchInvariantBf16Config` (`flashinfer.fused_moe`) | `MoEConfig(..., backend=BackendOptions((BatchInvariantBf16Config(),)))`: named explicitly, never a default candidate (alone in the candidates the layer skips its shootout) | one class per typed activation of the layer (gated, on the (gate, up) pair: `SwiGLU` with any `alpha` / `beta` / `limit`, `SwiGLUStep`, `GeGLU`, `GeGLUTanh`, `SiTU`, `PowLU`; ungated, on one accumulator: `ReLU2`, `Identity`, `GELU`, `ReLU`, `SiLU`; the activation's scalars are binary32 constants of the chain, so a parameter set is a class): the class's own routing (the declared `exp`, the strict selection order `(score desc, index asc)` on exact binary32 scores, the method's weight formula; a NaN logit is skipped by the row maximum, contributes `+0` to the softmax sum and is never selected), the deterministic permutation of the expanded rows (expert order, then token-slot order), the two grouped GEMMs as placements of the unsplit member of the dense class, `gemm.bf16.kfold` at `K=H` / `K=I` (per expert one fp32 fold over all of K whatever the weight shape: the dense operator's chunk rule, which runs `gemm.bf16.kchunk[C]` for a narrow weight, does not apply, so `mm_bf16(batch_invariant=True)` on one expert's weight reproduces these bits only where that rule leaves K unsplit; the gate / up pair of a SwiGLU term in one thread's fragment; the chain's declared arithmetic on the fp32 accumulators: `cvt_bf16(mul(silu(g), u))` for the default `SwiGLU`, every sigmoid / tanh through the declared `exp`, GELU / GeGLU through a declared `erf`, PowLU through a declared `log` and the correctly rounded `sqrt`, the clamps as `fmin` / `fmax`), and the slot-order fp32 combine `cvt_bf16(fold_r add(acc, mul(y_r, w_r)))`; `fingerprint("fused_moe", H=, I=, top_k=, num_experts=, method=, act=, n_group=, topk_group=)` (`method` a `RoutingMethodType` the class implements, the groups for `DeepSeekV3`'s group-limited selection, `act` an `ActivationConfig` of the layer or `"swiglu"` / `"relu2"`) | bf16 `hidden_states (T, H)`, `H` and `I` multiples of 8, `1 <= top_k <= min(E, 16)`, `E <= 1024`; routing from logits (`RoutingInputMode.FromLogits`: fp32 or bf16 `(T, E)` logits, methods `Default`, `Renormalize`, `RenormalizeNaive`, `Sigmoid`, `SigmoidRenorm`, `TopK` (the logits as weights), `Llama4` (`top_k = 1`) and `TopKSigmoid` (the sigmoid of the chosen logits as weights), `DeepSeekV3` with or without expert groups (`n_group` dividing `E` into groups of at least two experts, `1 <= topk_group <= n_group <= 16`, `top_k <= topk_group * E / n_group`: the group score is the sum of a group's two largest biased scores), a bias exactly for the methods that select on `sigmoid(l) + bias`) or pre-routed ids and weights (`PackedPrecomputed` / `UnpackedPrecomputed`, bf16 or fp32 weights; any `RoutingConfig`, since the selection is the caller's: a routing term the from-logits mode has no class for is refused at a from-logits call, not at construction); weights from `BatchInvariantBf16Config.prepare_weights(w1 [E, 2I, H] (rows up, gate) or [E, I, H] for the ungated activations, w2 [E, H, I], activation=...)` with `E` the local experts of a shard; bf16 output, or with `do_finalize=False` the layer's triple `[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]` (the rows in the class's permutation, the weights as the combine folds them -- fp32 from logits, where the other backends return bf16: the fold widens a bf16 weight exactly, so a receiver that needs bf16 rounds them itself -- and the flat map with `-1` at a dropped slot: the class's slot-order fold over it is the finalized output bit for bit, as the public op `fused_moe.batch_invariant.api.combine_unfinalized(*triple)` computes it; `use_fused_finalize` is ignored); expert-parallel shards (`ExpertConfig.local_expert_offset` / `local_num_experts`: the rank's output is the dense fold restricted to its experts, the other slots dropped; the receiver's slot-order fold over the ranks' unfinalized rows is the dense output, the fixed-order sum of finalized partials the split path's within-placement form); no fused shared experts. SM100 / SM103 / SM107 |
| `MoELayer` with `BatchInvariantFp8PerTensorConfig` under `QuantConfig(FP8PerTensor, FP8PerTensor)` (`flashinfer.fused_moe`) | `MoEConfig(..., quant=QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor), backend=BackendOptions((BatchInvariantFp8PerTensorConfig(),)))`: named explicitly, never a default candidate | the fp8 per-tensor (W8A8) classes `moe.e4m3.<act>`, one per typed activation and parameter set as the bf16 backend: the bf16 classes' routing (every method and group form of the bf16 row), permutation and slot-order combine; the two grouped GEMMs as placements of the dense fp8 class `gemm.e4m3.kfold[K=H]` / `[K=I]` (the 32-wide fp8 step); the static per-tensor scales in FlashInfer's multiplier convention (`q = x * s`), the class's dequantization scales their correctly rounded reciprocals formed once at preparation, the per-tile alpha `mul(s_a, s_w[e])` formed on the device; GEMM1's fused quantization `cvt.rn.satfinite.e4m3(mul(act(alpha-scaled accumulators), intermediate_scale_global))` (the activation chain of the bf16 class on the dequantized fp32 accumulators, no bf16 rounding, the activation's scalars in dequantized units) and GEMM2's `cvt_bf16(mul(acc, mul(s_inter, s_w2[e])))`; `fingerprint("fused_moe_fp8", H=, I=, top_k=, num_experts=, method=, act=, n_group=, topk_group=)` (the dimensions of `fused_moe`; the scales' values are inputs, not dimensions) | e4m3 `hidden_states (T, H)` from `BatchInvariantFp8PerTensorConfig.prepare_activations(x_bf16, hidden_states_scale_global=)` (no activation scale tensor), `H` and `I` multiples of 16, `1 <= top_k <= min(E, 16)`, `E <= 1024`; the routing modes, methods and groups of the bf16 backend through the same route kernels (its NaN-logit rule included; a term outside the from-logits classes is refused at the from-logits call, so a pre-routed layer builds under any `RoutingConfig`); weights from `BatchInvariantFp8PerTensorConfig.prepare_weights(w1 [E, 2I, H] (rows up, gate) or [E, I, H], w2 [E, H, I], hidden_states_scale_global=, intermediate_scale_global=, activation=...)` (the experts quantized here with one `448 / amax` multiplier per expert and GEMM); bf16 output; `do_finalize=True` only (the unfinalized triple is refused) and dense expert placement only (`local_expert_offset` / `local_num_experts` shards are refused: the class's EP form is not part of this backend); no fused shared experts. SM100 / SM103 / SM107 |
| `MoELayer` with `BatchInvariantFp8BlockConfig` under `QuantConfig(MXFP8, MXFP8)` (`flashinfer.fused_moe`) | `MoEConfig(..., quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8), backend=BackendOptions((BatchInvariantFp8BlockConfig(),)))`: named explicitly, never a default candidate; the fp8-block backend (`backend_key = "batch_invariant_fp8_block"`) groups the fp8 formats with block scales under one config and key as TRTLLM-Gen's `TrtllmFp8BlockConfig` does, the quantization pair of `QuantConfig` selecting the classes (MXFP8 x MXFP8 is the pair served; another pair is refused by the layer and by `prepare_weights` / `prepare_activations`, naming it); the weight view names its pair (`quant`) and a view without it is declined | the classes `moe.mxfp8.<act>`, one per typed activation of the layer as for the bf16 backend: the expert weights and the activations MXFP8 (e4m3 values, one UE8M0 scale per 32 elements), both by the quantizer class `mxfp8.quant` of the mode (the experts per expert at `prepare_weights`, the activations per token by `prepare_activations` or any producer of the canonical MXFP8 pack); the class's own routing, the deterministic permutation of the expanded rows into expert groups padded to whole 128-row scale-factor atoms (one scatter re-tiles the token-major scales into the swizzled layout the instruction reads; the padding rows are never stored), the two grouped GEMMs as placements of the dense MXFP8 class of the part (`gemm.mxfp8.kfold[K=H]` / `[K=I]` on SM100 / SM103, `gemm.mxfp8.kfold64[...]` on SM107: the SM107 instruction folds 64 elements per step, so the architecture is a dimension of the class, as for `mm_mxfp8`; the UE8M0 block scales are operands of the block-scaled step, so the chain sees dequantised fp32 accumulators and has no scale term), the activation chain of the bf16 classes on those accumulators (one bf16 rounding), the intermediate quantized by the same quantizer class inside GEMM1's epilogue (each 32-element block of the bf16-rounded chain value: the block amax, the UE8M0 scale and the e4m3 codes as the standalone quantizer computes them, so a placement of its declared second pass; the standalone pass over the materialised bf16 rows stays selectable and gives the same bytes), GEMM2's plain `cvt_bf16`, and the slot-order fp32 combine; `fingerprint("fused_moe_mxfp8", H=, I=, top_k=, num_experts=, method=, act=, n_group=, topk_group=, arch=)` (`arch` defaults to the current device; the receipt applies the class's shape rule and the runners' routing-term rule). The arithmetic is exact: the chain is the bf16 classes' with every step correctly rounded, so no SFU table enters the bits (the fp4 classes' chains take the part's SFU), and the instruction class of the part is the one architecture dimension | the canonical MXFP8 activation pack: e4m3 `hidden_states_q (T, H)` with the token-major UE8M0 `hidden_states_scale (T, H / 32)` (uint8 or `float8_e8m0fnu`; the linear layout alone, as `TrtllmFp8BlockConfig.prepare_activations` / `CutlassMxfp8Config.prepare_activations` / this config's `prepare_activations` produce it, taken as given: a swizzled or flattened buffer is refused, never reshaped into rows, as are `QuantConfig(swizzled_scale_factors=True)` and a `per_token_scale`; rows and scale bytes 16-byte aligned; `T = 0` gives the empty output), `H` and `I` multiples of 32, `1 <= top_k <= min(E, 16)`, `E <= 1024`, expert weight shapes the dense MXFP8 class folds in one chunk; routing from logits (the bf16 backend's methods; up to 16 tokens the routing and the padded permutation plan run as one launch, the bf16 backend's `route_plan` with the 128-row alignment) or pre-routed ids and weights; weights from `BatchInvariantFp8BlockConfig.prepare_weights(w1 [E, 2I, H] (rows up, gate) or [E, I, H] for the ungated activations, w2 [E, H, I], quant=..., activation=..., device=...)` (bf16 or a wider floating dtype, CPU-resident or on the device: quantized one expert at a time into the pack on a CUDA device, the geometry arguments checked against the weights; the experts' 128x4 scale buffers and the fused intermediate's carry zero pad columns); bf16 output; the tactics are the dispatcher's ranked record pairs (`schedule_mxfp8.default_configs_mx`: the dense MXFP8 class's roofline model over the grouped and the swapped record spaces, declared empirical constants and no measured table), any tactic the same bits; `do_finalize=True` only (the slot-order combine is part of the class), `use_fused_finalize` ignored; fused shared experts and expert-parallel shards refused by name. The layer's forward captures in a CUDA graph (the constants host-built outside the capture, a capture's own constants never cached) and replays the eager bits. SM100 / SM103 / SM107 |
| `MoELayer` with `BatchInvariantFp4Config` (`flashinfer.fused_moe`), the NVFP4 x NVFP4 pair | `MoEConfig(..., quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4), backend=BackendOptions((BatchInvariantFp4Config(),)))`: named explicitly, never a default candidate; the fp4 backend serves one quantization pair per `QuantConfig` as `TrtllmFp4Config` does (its weight view and records under the key `batch_invariant_fp4`), a pair it does not serve is refused and named | one class per typed activation of the layer, `moe.nvfp4.<act>` (W4A4): NVFP4 experts with one per-tensor global scale per expert and GEMM (`448 * 6 / amax`, the mode's per-tensor quantizer, 128x4 swizzled scales) and per-token NVFP4 activations (the mode's per-token quantizer, TransformerEngine's recipe) -- the bf16 classes' routing, permutation and combine around two members of the dense NVFP4 GEMM classes per expert (`gemm.nvfp4.kfold` on SM100 / SM103, `gemm.nvfp4.kfold128` on SM107 for the unsplit expert weight shapes: the instruction class per weight shape by the dense rule, so the architecture family is a dimension -- SM100 and SM103 are one family and share the fingerprint), with the W4A4 chains on the raw accumulators: GEMM1 `cvt_bf16(glu_pre(mul(g, s), mul(u, s)))` (gated) / `cvt_bf16(act(mul(acc, s)))` (ungated) with `s = mul(alpha1[e], pts[token])` (the expert's alpha as given times the token's per-token scale), the materialised bf16 intermediate quantized per token by the same class (the class contains the per-token quantization class at row lengths `H` and `I`), GEMM2 `cvt_bf16(mul(acc, mul(alpha2[e], pts2)))`, the slot-order combine; `BatchInvariantFp4Config(intermediate="static")` selects the static-intermediate class `moe.nvfp4.<act>.static`: GEMM1's epilogue quantizes the fp32 gated value per 16-column block with the expert's static scale (`prepare_weights(..., intermediate="static", intermediate_scales=[E])`, default ones) and GEMM2 carries the per-expert `1 / scale`, no bf16 intermediate; `fingerprint("fused_moe_nvfp4", H=, I=, top_k=, num_experts=, method=, act=, arch=, intermediate=)` | bf16 `hidden_states (T, H)` (quantized per token by the class) or packed fp4 `uint8 (T, H / 2)` with `hidden_states_scale` (the 128x4 swizzled scales of `nvfp4_quantize(..., per_token_activation=True, batch_invariant=True)` or the linear `(T, H / 16)` layout; at `T` a multiple of 128 and `H` of 64 the two have one shape, so `QuantConfig(swizzled_scale_factors=True)` names the swizzled buffer and `False` the linear rows, or the swizzled buffer is passed flattened; the 128x4 buffer's pad columns, present where `H` is not a multiple of 64, are zero, as every FlashInfer quantizer writes them -- the scatter and the gather copy a row's last 4-byte group as given, so a non-zero pad byte is read as the scale of a zero-filled block) and `per_token_scale (T,)` under `QuantConfig(..., per_token_scale=True)` (the layer's pack contract: a pack carrying one without the flag is refused by `pack_inputs`; unit scales when absent: the per-tensor pack of `TrtllmFp4Config.prepare_activations`); `H` and `I` multiples of 32, the expert weight shapes unsplit by the dense class's chunk rule, `1 <= top_k <= min(E, 16)`, `E <= 1024`; the routing modes and methods of the bf16 backend; the finalized output alone (`do_finalize=True`: the slot-order combine is part of the class, `do_finalize=False` is refused) and the full expert set (`local_expert_offset` / `local_num_experts` naming a shard are refused); weights from `BatchInvariantFp4Config.prepare_weights(w1, w2, activation=...)` (`quant=None` selects the pair) (bf16 canonical layouts, quantized per expert); bf16 output; no fused shared experts, no expert-parallel shards. SM100 / SM103 / SM107. The activation chains with an exponential or a division (SwiGLU and its variants, GeGLU, GeGLUTanh, SiTU, SiLU, GELU) compute those steps with the part's SFU (`ex2.approx.ftz` / `rcp.approx.ftz`): deterministic and batch-invariant per part, the bits a property of the part (the fingerprint carries the architecture family's instruction class and the tests pin the SFU tables' digest per part; the fourth hardware hypothesis below); the ReLU family and PowLU are the declared sequences. |
| `MoELayer` with `BatchInvariantFp4Config` (`flashinfer.fused_moe`), the MXFP4 x MXFP4 pair | `MoEConfig(..., quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP4), backend=BackendOptions((BatchInvariantFp4Config(),)))`: the same backend, selected by the pair; never a default candidate | one class per typed activation of the layer, `moe.mxfp4.<act>`: MXFP4 experts and MXFP4 activations (e2m1 elements with one UE8M0 scale per 32 elements on both operands, the mode's `mxfp4.quant` class -- TransformerEngine's recipe -- for the activations and the weights; no global and no per-token scale) -- the bf16 classes' routing, permutation and combine around two members of the dense MXFP4 GEMM classes per expert (`gemm.mxfp4.kfold` on SM100 / SM103, `gemm.mxfp4.kfold128` on SM107: the instruction class is the part's, so the architecture family is a dimension -- SM100 and SM103 are one family), with the bf16 class's chains on the raw accumulators (the block scales are applied inside the instruction): GEMM1 `cvt_bf16(glu_pre(g, u))` (gated) / `cvt_bf16(act(acc))` (ungated), the materialised bf16 intermediate quantized per 32-element block by `mxfp4.quant` (the class contains the quantization class), GEMM2 `cvt_bf16(acc)`, the slot-order combine; `fingerprint("fused_moe_mxfp4", H=, I=, top_k=, num_experts=, method=, act=, arch=)` | bf16 `hidden_states (T, H)` (quantized by the class) or packed fp4 `uint8 (T, H / 2)` with `hidden_states_scale` (the 128x4 swizzled scales of `mxfp4_quantize(..., batch_invariant=True)` or the linear `(T, H / 32)` layout; at `T` a multiple of 128 and `H` of 128 the two have one shape, so `QuantConfig(swizzled_scale_factors=True)` names the swizzled buffer and `False` the linear rows, or the swizzled buffer is passed flattened; the 128x4 buffer's pad columns, present where `H` is not a multiple of 128, are zero, as every FlashInfer quantizer writes them -- the scatter and the gather copy a row's last 4-byte group as given, so a non-zero pad byte is read as the scale of a zero-filled block), no `per_token_scale` (`QuantConfig(per_token_scale=True)` is refused); `H` and `I` multiples of 32, the expert weight shapes unsplit by the dense class's chunk rule, `1 <= top_k <= min(E, 16)`, `E <= 1024`; the routing modes and methods of the bf16 backend; the finalized output alone (`do_finalize=True`: the slot-order combine is part of the class, `do_finalize=False` is refused) and the full expert set (`local_expert_offset` / `local_num_experts` naming a shard are refused); weights from `BatchInvariantFp4Config.prepare_weights(w1, w2, quant=QuantConfig(weight=MXFP4, activation=MXFP4), activation=...)` (bf16 canonical layouts, quantized per expert; no static-intermediate class); bf16 output; no fused shared experts, no expert-parallel shards. SM100 / SM103 / SM107. The activation chains' exponential and divisions on the part's SFU as for the NVFP4 pair. |
| `MoELayer` with `BatchInvariantFp4Config` (`flashinfer.fused_moe`), the MXFP4 x MXFP8 pair | `MoEConfig(..., quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8), backend=BackendOptions((BatchInvariantFp4Config(),)))`: the same backend, selected by the pair (its third pair, the one `TrtllmFp4Config` serves under the same `QuantConfig`); never a default candidate; the weight view names its pair and its intermediate class (`quant`, `intermediate`) as every view of the backend, and a view without the keys or of another pair is declined | the classes `moe.mxfp4w8a.<act>`, one per typed activation of the layer as for the bf16 backend: the expert weights MXFP4 (e2m1 values, one UE8M0 scale per 32 elements) by the quantizer class `mxfp4.quant` of the mode (per expert at `prepare_weights`, as for the MXFP4 x MXFP4 pair), the activations MXFP8 (e4m3 values, one UE8M0 scale per 32) by the quantizer class `mxfp8.quant` (per token by `prepare_activations` or any producer of the canonical MXFP8 pack); the class's own routing, the deterministic permutation of the expanded rows into expert groups on one of two packings (a placement: whole 128-row scale-factor atoms per group, or the rows back to back with the scale factors at the padded positions -- the swapped decode records' packing, which the grouped body reads too; one scatter re-tiles the token-major scales into the swizzled layout the instruction reads and writes nothing for a dropped slot), the two grouped GEMMs as placements of the dense MIXED class of the part (`gemm.mxfp4w8a.kfold[K=H]` / `[K=I]` on SM100 / SM103, `gemm.mxfp4w8a.kfold64[...]` on SM107: the SM107 instruction folds 64 elements per step, so the architecture is a dimension of the class, as for `mm_mxfp8_mxfp4`; both operands' UE8M0 block scales are operands of the mixed-format block-scaled step, so the chain sees dequantised fp32 accumulators and has no scale term; the mixed step is its own fold, so the MXFP8 classes on the e2m1 values re-coded to e4m3 are a different specification), on the backend's unified grouped and swapped bodies (the swapped decode records run the e2m1 weight as the instruction's A operand on the twin instruction variant; the MX op of the MXFP8 backend at the mixed class, the box form alone), the activation chain of the bf16 classes on those accumulators (one bf16 rounding), the intermediate quantized by the `mxfp8.quant` class inside GEMM1's epilogue (each 32-element block of the bf16-rounded chain value: the block amax, the UE8M0 scale and the e4m3 codes as the standalone quantizer computes them, so a placement of its declared second pass; the standalone pass over the materialised bf16 rows stays selectable and gives the same bytes), GEMM2's plain `cvt_bf16`, and the slot-order fp32 combine; `fingerprint("fused_moe_mxfp4w8a", H=, I=, top_k=, num_experts=, method=, act=, arch=, n_group=, topk_group=)` (`arch` defaults to the current device; the receipt applies the pair's complete shape rule at the part's class and the runners' routing-term rule). The arithmetic is exact: the chain is the bf16 classes' with every step correctly rounded, so no SFU table enters the bits (the NVFP4 and MXFP4 pairs' chains take the part's SFU), and the instruction class of the part is the one architecture dimension | the canonical MXFP8 activation pack: e4m3 `hidden_states_q (T, H)` with the token-major UE8M0 `hidden_states_scale (T, H / 32)` (uint8, `float8_e8m0fnu`, or the e4m3-typed carrier `TrtllmFp4Config.prepare_activations` gives this pair; the linear layout alone, as `TrtllmFp8BlockConfig.prepare_activations` / `CutlassMxfp8Config.prepare_activations` / this config's `prepare_activations` produce it, taken as given: a swizzled or flattened buffer is refused, never reshaped into rows, as are `QuantConfig(swizzled_scale_factors=True)` and a `per_token_scale`; rows and scale bytes 16-byte aligned; `T = 0` gives the empty output), `H` and `I` multiples of 32 on SM107 and **of 128 on SM100 / SM103** (the SM100 / SM103 instruction reads the e2m1 weight through the TMA's unpacking format over whole 64-byte runs of the row), `1 <= top_k <= min(E, 16)`, `E <= 1024`, expert weight shapes the dense mixed class folds in one chunk; routing from logits (the bf16 backend's methods; up to 16 tokens the routing and the permutation plan run as one launch, in the form of the packing the ranked records run on) or pre-routed ids and weights; weights from `BatchInvariantFp4Config.prepare_weights(w1 [E, 2I, H] (rows up, gate) or [E, I, H] for the ungated activations, w2 [E, H, I], quant=QuantConfig(weight=MXFP4, activation=MXFP8), activation=..., device=...)` (bf16 or a wider floating dtype, CPU-resident or on the device: quantized one expert at a time into the pack on a CUDA device, the geometry arguments checked against the weights; the experts' 128x4 scale buffers and the fused intermediate's carry zero pad columns; no static-intermediate class); bf16 output; the tactics are the dispatcher's ranked record pairs (`schedule_mxfp4w8a.default_configs_mx`: the dense mixed class's roofline model over the grouped and the swapped record spaces, declared empirical constants and no measured table), any tactic the same bits; `do_finalize=True` only (the slot-order combine is part of the class), `use_fused_finalize` ignored; fused shared experts and expert-parallel shards refused by name. The layer's forward captures in a CUDA graph (the constants host-built outside the capture, a capture's own constants never cached) and replays the eager bits. SM100 / SM103 / SM107 |
| `chunk_gated_delta_rule` | `backend="auto", batch_invariant=True` | `gdn.bf16.c64`: the native SM100 / SM103 / SM107 chunked GDN prefill kernel without context parallelism (64-token chunks, fp32 state, log-domain gates, the Q / K l2-norm); decode computes row `p` of that kernel from rows cached since the last chunk boundary (a step kernel that issues the same tensor-core folds as `mma.sync`), so decode row `p` is prefill row `p` and prefill, chunked prefill, decode and multi-token steps are one class; `fingerprint("chunk_gated_delta_rule")` (no dimension) | bf16 packed q / k `[T, HQ, 128]` and v `[T, HV, 128]` with `HV` a multiple of `HQ`, fp32 log gates `g` and fp32 `beta` `[T, HV]` (`gate_domain="log"`), `use_qk_l2norm_in_kernel=True`; the state an fp32 pool `[slots, HV, 128, 128]` addressed by int32 `state_indices` and updated in place (a slot holds the state at the last 64-token boundary; the rows since it live next to the pool); a call with one token per sequence is a decode step, a call with more tokens is a prefill from the slot's state as a chunk boundary, and `prefix_lens` continues an open chunk (multi-token steps, pieces cut anywhere); `cu_seqlens` / `cu_seqlens_cpu`, `output`, `output_final_state` (the pool). CUDA graphs: a decode step over fixed buffers is capturable once no slot of the pool is waiting for its rebuild, i.e. after one eager decode step (or call of fewer than 64 tokens per sequence) on every slot prefilled since (while a slot the pool addresses waits, capturing a step raises). Not: `use_cp`, checkpoints, `output_state_indices`, the fused gate (`A_log` / `dt_bias` / `beta_is_logit`), linear-domain gates, a bf16 or fp16 state, fp16 activations, other head sizes. SM100 / SM103 / SM107 |
| `recurrent_kda` | `backend="auto", batch_invariant=True` | `kda.bf16.c16`: the native SM100 / SM103 / SM107 chunked KDA prefill kernel on its pinned persistent route (16-token chunks, fp32 state, in-kernel Q / K l2-norm, gate `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` and beta sigmoid); decode computes row `p` of that prefill in one launch of a step kernel (`include/flashinfer/kda/batch_invariant_decode.cuh`: the boundary state plus cached rows of the open chunk, every tensor-core product an `mma.sync` from registers), so decode row `p` is prefill row `p` and prefill, chunked prefill and decode are one class; `fingerprint("recurrent_kda")` (no dimension; it digests both kernels, the step's launcher and build flags and the installed `nvidia-cutlass-dsl` version, so it differs between environments with different DSL builds) | bf16 q / k / v / g / beta of head dim 128, `HV == H` a multiple of 8, or 1 to 10 or 12 heads (what the pinned prefill route carries; `flashinfer.batch_invariant.kda.spec.heads_supported`); the state an fp32 pool `[slots, HV, 128, 128]` addressed by `ssm_state_indices` and updated in place (a slot holds the state at the last 16-token boundary; what the open chunk needs lives next to the pool, so a chunked prefill cuts at multiples of 16 tokens except for its last piece, and a call with one token per sequence is a decode step); `A_log` / `dt_bias` with `use_gate_in_kernel`, `use_qk_l2norm_in_kernel`, `beta_is_logit`, a finite negative `lower_bound`; `cu_seqlens` packed or `[B, T]` fixed layouts, `output`, `output_final_state` (the pool); without `initial_state` the stateless call (whole sequences from the zero state, the same rows, nothing kept, no `grad_fn`: a trainer's forward pass). CUDA graphs: a decode step over fixed buffers is capturable after one eager step of the same batch size. Finite inputs: with a non-finite value inside a 16-token chunk the prefill shows it from the chunk's first row and a decode step from its own row, so the two differ inside that chunk (the sequence's neighbours stay exact). Not: speculative decode (`num_spec_tokens`), checkpoints with a state pool or at another interval than 16 (the stateless call takes `checkpoint_every_n_tokens=16`), `checkpoint_state_indices`, `initial_state_source`, `seq_order`, `prefill_workspace`, the frozen-verify arguments, a bf16 state. SM100 / SM103 / SM107 |

## The dispatcher and the launch grid

Each class has an enumerated configuration space (tile, cluster shape, 2-CTA pairing, raster order, swizzle band, chunk
fold); every record computes the class, so the choice among them moves time only (R3). `gemm/batch_invariant/heuristic.py`
orders the records for a shape, layout, format and part by a per-tile roofline estimate: waves of tiles over the part's
co-resident cluster capacity, each the slower of its MMA and its L2-to-SMEM traffic per k-step, the aggregate L2 bound,
the operands' HBM bytes plus a share of the bytes the scheduler's tile order fetches again (which operand panels stay
resident in the L2 across a swizzle band), the epilogue and the split's fixup, with a few empirical constants shared by
every part and format (none fitted per part). The first record is the default; `autotune(True)` profiles the first ten
(the best five, each with its raster twin) and keeps the bits by construction. The persistent grid is the device's
co-resident cluster count for the record's cluster shape (the CuTe DSL's GPC-aware occupancy query, asked once per
device): a launch parameter that decides which CTA computes which tile and never a bit. The heuristic takes the device's
queried capacity and L2 size once the API has recorded them, else the family's nominal table; the generated Lean dispatch
cases are the nominal parts' (`nominal=True`).

The MoE classes' grouped GEMMs are the dense class under a different scheduler (one segment of rows per expert), so
their dispatcher is the dense ranking at `M =` the average rows per active expert, mapped onto the grouped space (the
records of swizzle band 1 whose N tile the staged epilogue lays out) and filtered by the grouped body's acceptance
(`fused_moe/batch_invariant/schedule.py`). The space also holds the swapped records (`swap_ab`: the weights as the MMA's A
operand, the tokens as its B, the accumulator transposed; 8 / 16 / 32-token tiles against 64- and 128-row weight tiles), which
the dispatcher ranks with the grouped ones for the decode regime, and GEMM1 on a swapped record with a narrow token tile takes
the gather form: it gathers the token rows through the plan's slots (the TMA's gather4 mode) instead of a separate gather
launch over a permuted copy. Each is a placement of the class with its discharging test in
`tests/moe/test_unified_moe_batch_invariant.py`: the grouped GEMMs against the dense class per expert and the CPU chain
(`test_layer_matches_the_class_composition`), every swapped record against a grouped one (`test_swapped_records_equal_the_grouped_records`),
the gather form against the materialised path (`test_gather_form_of_the_op_equals_the_materialised_path`). The MXFP8 classes rank the grouped MX space
and their swapped records through the same contract (`schedule_mxfp8.py`). The autotuner's
tactics are the pairs of the top records of the two GEMMs; every pair computes the class. The persistent grid of a grouped GEMM
is the same launch parameter as the dense one's. The fp4 MoE classes' dispatcher (`fused_moe/batch_invariant/schedule_fp4.py`) ranks the same way at the
dense 4-bit class's instruction class: the ranking's input is the unpadded average rows per active expert (`record_cost` takes
it with the active expert count), while the eligibility bound pads that row count to the 128-row scale-factor atom (the grouped
kernel computes whole atoms per group) and keeps only the records whose M tile fits the padded group: a taller tile computes
padding the dense estimate does not price. At few rows per active expert the ranking is led by the swapped
decode body (`fused_moe/batch_invariant/blocks/swap.py`: the weights on the MMA's M side, the group's tokens a narrow
N tile of 8 / 16 / 32 rows read from the dense packing, their scale factors gathered from the padded positions). The dispatch
is `default_configs_fp4()`: the first record of each GEMM's ranked list, the swapped records ranked with the grouped ones;
`token_tile_default()` is a reference heuristic beside it, not the dispatch -- the narrowest swapped tile when the rows per
active expert fill it to at most three quarters, the padded body otherwise -- stating the regime the ranking leads with the
swapped body in. At decode the GEMM traffic is then the weights read once instead of a 128-row token atom per weight tile. Both bodies compute the same
bits (the block-scaled step is symmetric in its operands); the dense record and the swapped record are two placements the
autotuner may profile.

## How to check the proofs

`lean/` holds one Lean 4 project (no Mathlib). Per class it models the dataflow as terms over opaque, correctly rounded
primitives (one opaque tensor-core step per operand format: `mma16` for bf16, `mma32` for the fp8 formats, so the fp8
classes are different terms from the bf16 class even at equal K; the block-scale class folds `mma32` blocks under `fma` with its
scale leaves, `kblock128`, a different term from the per-tensor fp8 fold at every K) and proves that every implementation of the class the
kernels may run (for example a split reduction with its index-addressed workspace, or a pipelined loop with any stage
count) is the class's term, including its epilogue chain, and the launch-model statements of R1 and R2; an elementwise
postfix of the chain such as the bias lifts every identity by congruence, and the biased and the unbiased chain are proved
to be different terms. The model gives a configuration no way to reach an output element (R3 is immediate in it, and
is instantiated on the whole enumerated configuration space); that the kernel realises the model is what the engine's
lowering trace and the GPU configuration-agreement tests establish. What Lean cannot state is the hardware hypothesis that the instruction variants a
configuration space selects compute the same per-element function; the GPU configuration-agreement tests check it bit
for bit. A second hardware hypothesis of the block-scaled classes: the block-scaled step gives the same bits with its two operands exchanged, which the swapped decode records (`swap_ab`: the weights as the kernel's A, the tokens as its B) rely on; the dispatcher takes those records up to 64 rows and the dense records above, so the rows of a 64-row and a 65-row call are compared bit for bit in `test_batch_invariance_along_rows` of the mxfp8, fp4 and mxfp8_mxfp4 suites (M = 63, 64, 65 among the row counts), and every swapped record against the dense one in `test_swapped_decode_records_compute_the_class`, on each part. The bf16 MoE classes' swapped records rest on the same hypothesis for the bf16 step (`mma16` with its two operands exchanged: the weights as the kernel's A, the tokens as its B, the accumulator transposed), which `test_swapped_records_equal_the_grouped_records` of `tests/moe/test_unified_moe_batch_invariant.py` discharges bit for bit on each part: every swapped record, in its box and its gather form, against a grouped record over groups of one to several token tiles. The fp8 per-tensor MoE classes' swapped records rest on it for the 32-wide fp8 step, discharged the same way by `test_swapped_records_equal_the_grouped_records` of `tests/moe/test_unified_moe_batch_invariant_fp8.py`. The MX MoE classes' swapped records (the MXFP8 pair and the mixed MXFP4 x MXFP8 pair, whose swapped records run the e2m1 weight as the instruction's A operand on the twin variant of the `kind::mxf8f6f4` instruction) rest on it for their block-scaled steps, discharged the same way by the swapped-body tests of `tests/moe/test_unified_moe_batch_invariant_mxfp8.py` and `tests/moe/test_unified_moe_batch_invariant_mxfp4w8a.py`. A third, of the block-scale fp8 class's promotion: the packed `fma.rn.f32x2` folds two elements per instruction and each of its lanes is the same correctly rounded fma as the scalar `fma.rn.f32`; the exact-model tests of `tests/gemm/test_batch_invariant_gemm_fp8_blockscale.py` discharge it bit for bit, the kernel's output against the exact CPU promotion (correctly rounded scalar fmas over the device block partials), on each part.

The fp4 MoE classes (`moe.nvfp4.<act>`, `moe.nvfp4.<act>.static`, `moe.mxfp4.<act>`) stand on a fourth hardware hypothesis
for the activations whose chain contains an exponential or a division (SwiGLU and its variants, GeGLU, GeGLUTanh, SiTU, SiLU,
GELU), stated here as their verification standard: the chain computes the exponential as the part's `ex2.approx.ftz.f32` and
each division as `rcp.approx.ftz.f32` times the numerator (SFU instructions whose results PTX leaves implementation-specific),
so the class is deterministic and batch-invariant per part and its bits are a property of the part's SFU tables: the fingerprint
carries the architecture family (its instruction class: SM100 and SM103 are one family), the tests pin the SFU tables' digest
per part, and the same class name on two parts promises nothing across them. The CPU model
(`fp_interface/sfu.py`) takes the two primitives as data: the device evaluates `ex2` over every fraction of `[0, 1)` and `rcp`
over every significand of `[1, 2)` once per process, and the model composes the result from the table entry and an exact
exponent shift with the instructions' fixed-point, flush-to-zero and special-value rules; `sfu.check_model` evaluates the device
against the model over random inputs of every exponent and the special values, and the tests pin the tables' digest per
part (SM100 and SM107 pinned, with one digest: their tables are identical; an unpinned part such as SM103 reports its digest
for pinning), so the exact-equality tests of the chains run on the same part. Lean takes the two primitives as opaque per-part
functions (`ex2Approx`, `rcpApprox`: opaque constants, not axioms) and states the chains with the same fold and invariance
theorems as the bf16 class's. The bf16, fp8 and mixed MXFP4 x MXFP8 MoE classes stand on the specification alone (every step a correctly rounded
binary32 operation, reproduced bit for bit on the CPU); so do the fp4 classes' ReLU family and PowLU, whose chains have no
SFU step.

`Cases.lean` and the family modules it imports (`lean/BIKernels/Cases/GEMM.lean`, `Attention1.lean`, ...: one module per
kernel family, a family larger than the generator's per-file budget as numbered parts chained by import) are generated
from the Python specification and engine (`python -m flashinfer.batch_invariant.lean_cases`):
the Lean model reproduces the Python S-expressions node for node, re-derives the dispatchers' shape rules and checks
every record of a configuration space and every dispatcher pick against the class's constraints. The device-free test
`tests/batch_invariant/test_lean.py` byte-compares the committed modules and fails on a stale or missing generated one; its `lake`
steps run when the toolchain is installed.

The MoE classes have their own tier. `lean/BIKernels/MoE.lean` (and `MoEFp8.lean` for the fp8 per-tensor classes: the per-tile alpha from the multipliers' reciprocals, the fused e4m3 quantization of every chain as the standalone quantization term of the activation value, GEMM2's scaled chain) states the declared `exp`, sigmoid, tanh, erf and log
chains, the GEMM1 chains of every activation the classes take as a parameter (the gated `GAct`: SwiGLU, GeGLU-tanh,
GeGLU, SwiGLU with FlashInfer's `alpha` / `beta` / `limit`, SwiGLUStep, SiTU and PowLU with their binary32 scalars as
constants of the term; the ungated `UAct`: ReLU^2, identity, ReLU, SiLU, GELU), the routing weights of every method, the
slot-order combine, the launch model (determinism, batch / permutation / sub-batch invariance) and the grouped GEMM as a
placement of the dense class (`grouped_eq_dense`: every row of the grouped kernel is the dense `kfold`, whatever the
grouping, the expert load and the row's position). `lake exe moeconformance` prints its terms for the generated cases
(`lean/BIKernels/MoECases.lean`, rendered by `python -m flashinfer.fused_moe.batch_invariant.lean_cases`: one case per
chain and parameter set, the chains up to `THEOREM_MAX_BYTES` as kernel `decide` theorems, the larger SiTU / PowLU chains
and the routing weights as conformance lines), which the device-free test `tests/batch_invariant/test_moe_lean.py`
compares with the Python specification (`fused_moe/batch_invariant/spec.py`).

`lean/BIKernels/MoENvfp4.lean` states the NVFP4 MoE classes on the same model: the W4A4 row scale, the GEMM1 chains of every gated and ungated activation on the scaled accumulators (`gluScaledT` / `actScaledT` on `MoE.gluHwT` / `MoE.actElemHwT`: the bf16 chains with the scale entering once per accumulator and the exponential and divisions as the opaque per-part `ex2Approx` / `rcpApprox`), GEMM2's scaled conversion, the instruction class per expert weight shape and the grouped block-scaled space as a subset of the dense NVFP4 space. `lake exe moenvfp4conformance` prints its terms for the generated cases (`lean/BIKernels/MoENvfp4Cases.lean`, rendered by `python -m flashinfer.fused_moe.batch_invariant.lean_cases_nvfp4`), which `tests/batch_invariant/test_moe_nvfp4_lean.py` compares with the Python specification (`fused_moe/batch_invariant/spec_nvfp4.py`).
`lean/BIKernels/MoEMxfp4.lean` states the MXFP4 MoE classes on the same model: the GEMM1 chains of every gated and ungated activation on the raw accumulators (`mxGluT` / `mxActT` on `MoE.gluHwT` / `MoE.actElemHwT`: the bf16 chains as they are, no scale entering the epilogue), GEMM2's conversion, the instruction class as a function of the architecture family alone and the grouped record spaces as subsets of the dense MXFP4 space. `lake exe moemxfp4conformance` prints its terms for the generated cases (`lean/BIKernels/MoEMxfp4Cases.lean`, rendered by `python -m flashinfer.fused_moe.batch_invariant.lean_cases_mxfp4`), which `tests/batch_invariant/test_moe_mxfp4_lean.py` compares with the Python specification (`fused_moe/batch_invariant/spec_mxfp4.py`).

Trust statement: **Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`), plus `Lean.ofReduceBool` for the
named exhaustive checks listed in `lean/Axioms.lean`** (the configuration-space and dispatcher-membership checks over
hundreds of records per format, decided by `native_decide`; every other generated case is kernel `decide`). `Lean.ofReduceBool` trusts the compiled evaluator for
those closed Boolean terms; `lake exe axioms` enforces exactly this list and fails on any `sorry`.

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"            # the pinned toolchain is lean/lean-toolchain (leanprover/lean4:v4.15.0)
cd flashinfer/batch_invariant/lean
lake build && lake build conformance axioms    # ~1 min cold
lake exe conformance                           # the Lean model's terms for the generated cases
lake exe axioms                                # the axiom policy: 0 with sorry, 0 violations
```

The Lean steps are not run in CI. The `lean/` directory is part of the repository only: it has no `__init__.py` and
no package data entry, so no `.lean` file is in the wheel.

## Quantization

The quantizers have no cross-row reduction: a block's scale and codes depend on that block alone (per-token NVFP4: on
its row, whose amax sets the row's global scale), so FlashInfer's own kernels are already batch invariant in the R1 / R2
sense. What `batch_invariant=True` adds is a specified, fingerprinted
dataflow (R3 across the enumerated configurations, R4 across the scale-factor layouts and the input dtypes that widen
exactly), a closed-form dispatcher whose every pick is a member of one enumerated space, and the exact CPU model and Lean
statements of the chains. The flag requires `backend="auto"` (the default, which runs the CUDA kernel without the flag) on the
three operators and on `fp4_quantize`, their shared entry point.

| API | Classes and fingerprint | Numerics | Supported inputs |
|---|---|---|---|
| `mxfp8_quantize` | `mxfp8.quant`; `fingerprint("mxfp8_quantize")` (no dimension: one specification per format) | TransformerEngine's MXFP8 chain: block amax as a pairwise `max` tree over `abs`, the UE8M0 scale `cvt.rp.satfinite.ue8m0(amax * fp32(1/448))`, the exact inverse `2^(127 - e8)` at every byte, the elements `cvt.rn.satfinite.e4m3(mul.rn(x, inv))`. The corners are part of the chain: a block whose amax is below `448 * 2^-127` (scale byte 0) keeps its values under the exact inverse, a NaN element gives `0x7F`, `-inf` gives `-448` | bf16 / f16 / fp32 inputs of shape `(..., K)`, `K` a multiple of 32 and rows of at most 128 KiB, `alignment` padding; the 128x4, 8x4 and linear scale layouts (placements of the store: the same bytes); SM100 / SM103 / SM107 |
| `mxfp4_quantize` | `mxfp4.quant`; `fingerprint("mxfp4_quantize")` | TransformerEngine's MXFP4 recipe: the same chain carried to e2m1 (`fp32(1/6)`, `cvt.rn.satfinite.e2m1`; the tiny-block threshold `6 * 2^-127`) | as above (`K` a multiple of 32); the three layouts |
| `nvfp4_quantize` | per-tensor `nvfp4.quant` (`fingerprint("nvfp4_quantize")`), per-token `nvfp4.quant.pertoken` (`fingerprint("nvfp4_quantize_per_token", K=)`: the row amax tree spans the row, so the row length is a dimension) | TransformerEngine's NVFP4 recipe with every operation correctly rounded or exact: the block amax as a pairwise `max` tree over `abs`, `sf = cvt.rn.satfinite.e4m3(amax * (gs * fp32(1/6)))`, `inv = fmin(1 / (sf * (1 / gs)), FLT_MAX)` with correctly rounded divisions, the elements `cvt.rn.satfinite.e2m1(mul.rn(x, inv))`; per token the row's global scale `gs = nzor(fmin((1 / gs_inv) / ramax, FLT_MAX), 1)` (`ramax` the row amax tree; `FLT_MAX` for an all-zero row) and the returned per-token scale `1 / gs` | bf16 / f16 / fp32 inputs, `K` a multiple of 16; the per-tensor global scale an input (a float or a 1-element fp32 tensor), the per-token inverse base scale; the three layouts and `do_shuffle` (FlashInfer's own host shuffles applied to the class's linear-layout bytes; a multiple of 32 rows, as FlashInfer's shuffle requires); `sf_vec_size` 16; no 4over6 recipe, no fp8 input, no `expanded_idx_to_permuted_idx`, no `out_scale`; SM100 / SM103 / SM107 |
| `fp4_quantize` | the entry point behind `nvfp4_quantize` (per tensor) and `mxfp4_quantize`: `sf_vec_size=16, sf_use_ue8m0=False` runs `nvfp4.quant`, `sf_vec_size=32, sf_use_ue8m0=True` runs `mxfp4.quant` | those chains | its own argument forms: `is_sf_swizzled_layout` / `is_sf_8x4_layout`, `global_scale` None as the scale 1.0, `is_global_scale_inversed` as one correctly rounded fp32 division, a column-major input quantized as the transpose of its row-major view (the entry point's placement) and the `(rows, columns)` scale view; other recipes, fp8 inputs and a 4over6 recipe raise |

The chains are the contract: named by the fingerprints, stated in Lean and checked against their exact CPU model, they are
what the classes promise with respect to TransformerEngine's recipes. FlashInfer's other quantizer backends are not a
reference for the classes and nothing is promised relative to them: their numerics may change within their own tolerances.

The quantizers' dispatcher is a closed form of `(M, K, element bytes, SM count, layout)`: one configuration per call, no
autotuning (`quantization/mx_schedule.py`, `quantization/nvfp4_schedule.py`), fitted on CUPTI sweeps of both parts.
The Lean model (`lean/BIKernels/Canon.lean`, `Semantics.lean`, `MXQuant.lean`, `Quant.lean`) states the engine's reduction
orders and the chains; `lake exe quantconformance` prints its terms for the generated cases
(`lean/BIKernels/QuantCases.lean`, rendered by `quantization/lean_cases.py`), which the device-free test
`tests/batch_invariant/test_quant_lean.py` compares with the Python specification. Tests:
`tests/quantization/test_batch_invariant_mx_quantize.py` (both MX formats) and `test_batch_invariant_nvfp4_quantize.py`.

## Sampling

`flashinfer.sampling.softmax(logits, temperature, batch_invariant=True[, out_dtype=])` and the new entries
`flashinfer.sampling.log_softmax(logits, temperature)` and `flashinfer.sampling.log_probs(logits, token_ids,
temperature, return_lse=)` run one class, `lse.row`: the row maximum `m` and the sum `s` of the exponentials as the
engine's canonical trees over the vocabulary (an 8-element fold per slot, pairwise over the blocks of 256 elements,
pairwise over the 32 lanes: a function of the vocabulary alone, never of the batch, the configuration or the device),
and fixed chains over them. The operator's own softmax selects its reduction path by batch size, so a row's bits there
depend on the batch; the class is what the mode promises instead.

| API | Class and fingerprint | Numerics | Supported inputs |
|---|---|---|---|
| `softmax(..., batch_invariant=True, out_dtype=)` | `lse.row`; `fingerprint("softmax", N=, dtype=, temperature=[, out_dtype=])` (the vocabulary, the input dtype and whether a temperature enters the scale are the specification's dimensions; `out_dtype=torch.float32` names the fp32 input's specification: the leaves of a 16-bit input are its exact widenings, so only the output conversion differs) | `c = log2(e)` or `div.rn(log2(e), T)`; `z_j = mul.rn(add.rn(x_j, neg(m)), c)`; `e_j = ex2.approx.ftz(z_j)`; `s` the `add.rn` tree over the `e_j`; `p_j = cvt.rn(mul.rn(e_j, div.rn(1, s)))` in the input dtype, or `mul.rn(e_j, div.rn(1, s))` itself with `out_dtype=torch.float32`. `m` is the `max.f32` tree (a NaN operand yields the other; a NaN logit then propagates through the sum), a `-inf` logit weighs `+0` | bf16 / f16 / fp32 logits `(..., N)`, any `N` up to 2^21 (beyond it where the row's alignment admits a configuration; an odd fp32 row of 2^21 + 1 has none); temperature `None`, a positive scalar, or one per row (f16 / bf16 / f32 / f64, the last rounded to f32); same-dtype or fp32 output; `enable_pdl`; SM100 / SM103 / SM107 |
| `log_softmax(logits, temperature[, out_dtype=])` | the same class; `fingerprint("log_softmax", N=, dtype=, temperature=[, out_dtype=])` -- the class fingerprint, or with `out_dtype` the input's 16-bit dtype the 16-bit output member's (the converted element is a different output, so a different receipt) | the row `logp_j = mul.rn(add.rn(z_j, neg(lg2.approx.ftz(s))), ln 2)` in fp32, shape `logits.shape`; with `out_dtype` the input's 16-bit dtype the same row converted once, `cvt.rn(logp_j)` (bit for bit torch's `.to(dtype)` of the fp32 row: the `aten::_log_softmax(bf16) -> bf16` shape the trainers override) | as above; `out_dtype` `None` / `torch.float32` (the fp32 row) or the input dtype (16-bit inputs); `batch_invariant=True` is the pinned contract, `batch_invariant=False` (the default) is the operator's regular kernel for this entry, today the same class (a future release may route it elsewhere) |
| `log_probs(logits, token_ids, temperature, return_lse=[, top_k=, top_p=, min_p=, filter_apply_order=])` | the same class and fingerprint (`fingerprint("log_probs", ...)`); with a filter the kept-set log-probability's receipt, the row class's joined with `support.row.v1`'s (`fingerprint("log_probs", N=, dtype=, temperature=, top_k=, top_p=, min_p=, filter_apply_order=)`: `lse-row-<h>+support-row-<h>`) | the log-softmax row above gathered at the row's token ids (NaN for an id outside `[0, N)`), and with `return_lse` the row's `lse = mul.rn(add.rn(mul.rn(m, c), lg2.approx.ftz(s)), ln 2)`; fp32 outputs of the ids' shape / `logits.shape[:-1]`. With `top_k` / `top_p` / `min_p` the log-probability under the sampler's kept-set distribution instead: `support.row.v1`'s kept set and mass at the row (the truncated samplers' selection, verbatim: ties at the cut kept), then `mul.rn(add.rn(z_t, neg(lg2.approx.ftz(mass))), ln 2)` -- bit for bit the `kept_logp` the samplers return for the token they drew, the engines' processed log-probability; `-inf` for a token outside the kept set (its probability there is zero), NaN for an id outside `[0, N)` or a row whose kept mass is not a positive finite number | as above; `token_ids` int32 / int64 of shape `logits.shape[:-1]` (one id per row) or `logits.shape[:-1] + (K,)` (`K` ids per row in one launch; one per row with a filter); the filters as the truncated samplers take them (scalars or one per row, `filter_apply_order` `"top_k_first"` / `"joint"`), `return_lse` refused with a filter, `batch_invariant=True` required for them; the `batch_invariant` flag otherwise as for `log_softmax` |
| `sampling_from_probs(probs, indices, generator, seed, offset, return_valid, batch_invariant=True)` | `sample.icdf.v1`; `fingerprint("sampling_from_probs", N=, dtype=)` (the vocabulary is the dimension; the dtype is validated and enters nothing: the leaves of a 16-bit input are its exact widenings and the draw converts no output) | the inverse CDF read off the row class's canonical sum tree `S` of the given weights (the same `add.rn` tree as `lse.row`'s `s`): the threshold `t = mul.rn(u, S)` with `u` the fp32 uniform of the row's stream; the vocabulary walked in the tree's order (lane-major, then block, then element), the inclusive prefix of an element `add.rn(outer, fold_e)` where `fold_e` is the slot fold's running value and `outer` the `add.rn` chain, in ascending bit order from `+0`, of the left-sibling nodes of the tree the element's address names (every one a value the kernel's placement already holds: no new reduction, no division); the token is the first present element with a positive weight whose prefix is `>= t` (the positivity is part of the rule: the last prefix of a slot and the `outer` of the next are different `add.rn` chains over the same nodes, so a zero-weight slot head could otherwise be the first crossing), else the last with a positive weight; a row whose `S` is not a positive finite number is invalid (token `-1`, `valid` False) | bf16 / f16 / fp32 weights `>= 0` of shape `(..., N)`, any `N` as for `softmax`; `indices` int32 / int64 (the gather; the output takes its dtype; an index outside the source rows makes its output row invalid: token `-1`); `seed` / `offset` as the operator takes them, one key per output row (int64 / uint64 tensors of length M; a scalar, a length-one tensor or the pair reserved from `generator` exactly as the operator's path reserves it serves a single row -- shared by several rows it is refused, since every row would draw at the same uniform); `deterministic` ignored (deterministic by construction), `check_nan` the same host check; int32 tokens; SM100 / SM103 / SM107 |
| `sampling_from_logits(logits, indices, generator, seed, offset, batch_invariant=True[, temperature=, return_logprob=])` | `sample.icdf.v1`; `fingerprint("sampling_from_logits", N=, dtype=[, temperature=])` (the logits form embeds `lse.row`'s trees: a different fingerprint per vocabulary from the probabilities form, and another with a temperature in the scale); the returned log-probability carries `lse.row`'s receipt (`fingerprint("log_probs", N=, dtype=, temperature=)`) | the same draw over the `lse.row` weights `e_j = ex2.approx.ftz(mul.rn(add.rn(x_j, neg(m)), c))` of the row, `c = log2(e)` or `div.rn(log2(e), T)`, without materialising the probabilities; `S` is `lse.row`'s `s`, so the drawn token's log-probability under `log_probs` is the probability it was drawn with (R4, stated in Lean: `Draw.total_logits_eq_specSum`); with `return_logprob` that log-probability is returned from the same launch, `mul.rn(add.rn(z_t, neg(lg2.approx.ftz(S))), ln 2)` = `log_probs(logits, samples, temperature)` bit for bit | as above for logits (a row of `-inf` logits, or with a NaN, is invalid, its log-probability NaN); `temperature` `None`, a positive scalar or one per row as for `softmax`; the outputs `samples` or `(samples, logp)` |
| `top_k_sampling_from_probs(probs, top_k, indices, generator, seed, offset, return_valid, batch_invariant=True)` | `sample.icdf.v1` over `support.row.v1`; `fingerprint("top_k_sampling_from_probs", N=, dtype=)` (the two classes' fingerprints joined: the draw's tree and the kept set's rules at the vocabulary; the dtype validated, it enters nothing) | the kept set `{j : w_j >= t}` with `t` the k-th largest weight, found by an exact radix selection over the weights' 31-bit patterns (two to four passes of integer histograms by input kind: order-free; ties at the cut kept, `k <= 0` keeps nothing, `k >= N` everything), then `sampling_from_probs`'s draw over the masked weights (the dropped ones `+0`): the threshold `mul.rn(u, mass)` with `mass` the canonical `add.rn` tree over the kept weights, the token the first kept element whose prefix reaches it (the prefix nodes those of the masked tree); a row with nothing kept, no positive kept weight or a non-finite sum is invalid (token `-1`) | bf16 / f16 / fp32 weights `>= 0` of shape `(..., N)`, `N` whose padded row (blocks of 256 rounded up to a power of two) fits the staging budget of a 16-CTA cluster (`MAX_CLUSTER * SMEM_BUDGET`: `N <= 2^20` for bf16 / f16, `N <= 2^19` for fp32; beyond it `BatchInvariantUnsupportedError`); `top_k` an int or an int32 / int64 tensor of one value or one per row (an int64 value is clamped to `[0, N]` before the int32 cast; `top_k <= 0` keeps nothing, where the operator's kernel treats it as no filter) (one per *source* row when `indices` is given, read at `indices[i]` for output `i`: the operator's rule; `temperature` is one per output row); `indices`, `seed` / `offset` / `generator` as for `sampling_from_probs` (one key per output row); `deterministic` ignored (deterministic by construction), `check_nan` the same host check; int32 tokens (the indices' dtype with `indices`), `return_valid`; SM100 / SM103 / SM107 |
| `top_p_sampling_from_probs(probs, top_p, ..., batch_invariant=True)` | the same classes; `fingerprint("top_p_sampling_from_probs", N=, dtype=)` | the kept set the smallest upward-closed set whose declared fixed-point mass -- `q_j = trunc(mul.rn(p_j, div.rn(2^44, S)))` (`cvt.rzi`), an exact integer sum -- reaches `trunc(mul.rn(p, 2^44))`, `S` the row's canonical tree (ties kept; every positive weight when `p >= 1` or the positive mass does not reach the target, zero weights never; a row whose sum is below 2^-84, where the fixed-point scale `div(2^44, S)` overflows, keeps nothing and is invalid), then the draw over it | as above; `top_p` a float or an fp32 tensor of one value or one per row |
| `min_p_sampling_from_probs(probs, min_p, ..., batch_invariant=True)` | the same classes; `fingerprint("min_p_sampling_from_probs", N=, dtype=)` | the kept set `{j : p_j >= mul.rn(p_max, min_p)}` with `p_max` the row's `max.f32` tree, then the draw over it | as above; `min_p` a float or an fp32 tensor of one value or one per row |
| `top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices, filter_apply_order, ..., batch_invariant=True)` | the same classes; `fingerprint("top_k_top_p_sampling_from_probs", N=, dtype=[, filter_apply_order=])` (the order is a dimension) | `top_k_first`: the top-k cut, then the top-p cut with the kept mass as `S` (the masked tree); `joint`: both cuts on the whole row, intersected; then the draw over the kept set -- one kernel either way | as above; `top_k_max` ignored (no fast path) |
| `top_k_top_p_sampling_from_logits(logits, top_k, top_p, indices, filter_apply_order, ..., batch_invariant=True[, temperature=, return_logprob=, return_kept_logprob=])` | the same classes; `fingerprint("top_k_top_p_sampling_from_logits", N=, dtype=[, filter_apply_order=, temperature=])` (the logits form embeds `lse.row`'s trees: another fingerprint per vocabulary, and another with a temperature in the scale); the returned log-probabilities carry `lse.row`'s receipt (`fingerprint("log_probs", N=, dtype=, temperature=)`) | the same over the `lse.row` weights `e_j = ex2.approx.ftz(mul.rn(add.rn(x_j, neg(m)), c))` of the row, `c = log2(e)` or `div.rn(log2(e), T)`, without materialising the probabilities: the row's `S` is `lse.row`'s `s`, the kept set and the draw are those of the row's softmax at the temperature; with `return_logprob` the drawn token's log-probability under the **full** distribution, `mul.rn(add.rn(z_t, neg(lg2.approx.ftz(S))), ln 2)` = `log_probs(logits, samples, temperature)` bit for bit (never renormalised over the kept set); with `return_kept_logprob` the one renormalised over the kept set, `mul.rn(add.rn(z_t, neg(lg2.approx.ftz(mass))), ln 2)` (the probability it was drawn with) | bf16 / f16 / fp32 logits of shape `(..., N)`; `temperature` `None`, a positive scalar or one per row as for `softmax`; a row of `-inf` logits, or with a NaN, is invalid (its log-probabilities NaN); the outputs `samples[, logp][, kept_logp]`, or into the caller-owned buffers `out=` / `logprob_out=` / `kept_logprob_out=` (written in place, returned as given; every truncated sampler takes `out=`: int32, or int64 with int64 `indices`; the buffers are refused, never copied, when they do not fit, and may not overlap an input or each other; the default path refuses them; the kernel's validity word and, unless the caller asks for it, the kept mass go to a per-device scratch, so a step allocates nothing it does not return) |

**The stream.** Under the flag the uniform of a row comes from `stream.philox4x32-10.splitmix-key.v1` (`sampling/stream.py`,
`lean/BIKernels/Stream.lean`): the key `K = mix64(seed ^ mix64(offset))` (SplitMix64's finalizer), Philox4x32-10 on the
counter `(0, 0, 0, 0)` under `(K mod 2^32, K >> 32)`, word 0 converted as `curand_uniform` converts it,
`add.rn(mul.rn(cvt.rn.f32.u32(w), 2^-32), 2^-33)` in `(0, 1]`. The row's slot in the batch enters nothing: the same row
with the same `(seed, offset)` draws the same token alone, at any position of any batch, under any configuration. The
operator's own samplers seed `curand_init(seed, slot, offset)` (the slot is the Philox subsequence), so their uniforms
differ from the class's for every slot and their tokens move with the batch position; nothing is promised relative to
them. A `generator` without `seed` / `offset` reserves the pair as the operator's path does (`get_seed_and_offset`, the
same increment), so a call with only a generator is reproducible from the generator's state.

A row view into a wider tensor (the `[..., :vocab]` slice of a padded lm_head) whose row stride is a multiple of 16 bytes
is read in place; other views (a strided column view, a non-contiguous leading layout) are copied first (a placement). A
per-row temperature tensor has shape `logits.shape[:-1]` or `(rows,)`; a scalar temperature is validated on the host, a
tensor's values are not validated on the device (a check would need a synchronisation and would break CUDA-graph capture):
every value must be positive and finite, a `0` yields a NaN row, a negative or non-finite value an undefined row. For a
sampler and a trainer to agree bit for bit they must call with the same arguments: raw logits and the temperature (the
class scales inside its chain), or identically pre-scaled logits and `temperature=None`; a greedy row's temperature of 0
(vLLM, SGLang carry it in the tensor) must be replaced by `1.0` or `None` before the call: the greedy log-probabilities are
read from the unscaled row.

**The training forward.** The RL trainers (verl, Miles, NeMo RL, Megatron-Core's batch-invariant mode) re-score the
rollout's tokens with a forward pass whose log-softmax must give the engine's bits; none of them needs a backward kernel
from this mode (verl and Megatron-Core override `aten::_log_softmax` forward only, Miles takes `torch.log_softmax` after
its all-gather, NeMo RL re-runs its own autograd function on the saved logits for the gradient), so the mode provides the
forward alone and the frameworks own the adapter -- the one they already have, pointed at the class:

```python
def _log_softmax_bi(input, dim, half_to_float):            # the aten::_log_softmax override of the trainers' modes
    assert dim in (-1, input.dim() - 1)
    out_dtype = torch.float32 if half_to_float else input.dtype
    return flashinfer.sampling.log_softmax(input, batch_invariant=True, out_dtype=out_dtype)
```

The contract both sides must call with, written down because the frameworks disagree today (vLLM's `raw_logprobs` are
before the temperature and every processor, its `processed_logprobs` after temperature and filters over the kept
support, SGLang's default after the temperature over the full vocabulary, Miles divides by the rollout temperature and
casts to bf16, NeMo RL's zero-mismatch mode takes raw logits at temperature 1): the temperature is applied inside the
chain (`c = log2(e) / T`, a scalar or one per row), the normaliser is over the full vocabulary (the row as given: a
padded lm_head's rows must be sliced to the vocabulary, or padded with `-inf` on both sides, since `N` is in the receipt),
the value is fp32 (or its one conversion to the input's 16-bit dtype), and two callers agree bit for bit exactly when both
pass the same logits bits with the same temperature (or identically pre-scaled logits and `temperature=None`) on the same
architecture (the two MUFU tables are identical on SM100 and SM107, measured over every input); a greedy row's temperature
of 0 is replaced by `1.0` or `None` before the call. The draw's `return_logprob` is this value -- the full
distribution's, never renormalised over a filtered support (`top_k_top_p_sampling_from_logits`
gives the same `return_logprob`, and the renormalised value the engines' `processed_logprobs`
of a filtered mode report as a separate named output, `return_kept_logprob`) -- so
the engine's returned log-probability and the trainer's `log_probs` / `log_softmax` of the same row are one definition.
A trainer that re-scores under the sampler's filters instead (NeMo RL's sampler replay; a comparison against the engine's
processed log-probabilities) calls `log_probs(logits, token_ids, temperature, batch_invariant=True, top_k=, top_p=, min_p=,
filter_apply_order=)`: the same kept set as the sampler's (ties at the cut kept, as the class defines it -- a torch
re-implementation breaks ties by position, and bf16 logits widened to fp32 tie constantly across a 150k vocabulary) and
the same chain, so the value is the engine's `return_kept_logprob` bit for bit, `-inf` for a token outside the kept set.
Nothing is stored and no RNG state crosses to the trainer: the draw is a function of the row, its `(seed, offset)` key
and the temperature, and the trainer never re-draws.

A vocabulary-parallel trainer (Megatron's `--tensor-model-parallel-size` with the lm_head split over the ranks) uses
the class after an all-gather of the shards into full rows (the `_use_full_vocab_log_softmax` path of NeMo RL, Miles's
`_calculate_log_probs_and_entropy_true_on_policy`): the per-rank max / sum-exp exchange of the distributed log-softmax
is a different rounding from the full row's and is not this class. Where it plugs in: vLLM's sampler log-probabilities
(`vllm/v1/worker/gpu/sample/logprob.py`) and the FlashInfer samplers it already calls; SGLang's deterministic sampler
(`python/sglang/srt/layers/sampler.py`) and `layers/logprob_processor.py`; verl's `logprobs_from_logits` /
`_sampler_chunk_logprobs` (`verl/utils/torch_functional.py`) and its aten override installer
(`verl/workers/engine/utils.py`); Miles's `math_utils` log-probability path; NeMo RL's `_GenAlignedLogprob` and
Megatron-Core's `batch_invariant_kernels.py` registration of `aten::_log_softmax`.

**Caller-owned outputs.** Every sampling entry under the flag takes its outputs as optional caller-owned buffers --
`softmax(..., out=)`, `log_softmax(..., out=)`, `log_probs(..., out=, lse_out=)`, `sampling_from_probs(..., out=)`,
`sampling_from_logits(..., out=, logprob_out=)` -- written in place and returned as given, so a step allocates no output (a Python-scalar filter of the truncated samplers
is broadcast into a per-row tensor per call; per-row tensors avoid it)
(the kernels' validity word, which the draws do not return unless asked, goes to a per-device scratch grown to the largest
batch seen and never freed once handed out: a graph that captured it, or a kernel still pending on another stream, keeps
writing to its address after the cache has grown): CUDA graphs captured on the caller's buffers replay into them, and a framework's persistent structures (a decode step's
token and log-probability buffers, a trainer's chunk buffer for the batch x vocabulary log-softmax row) take the kernel's
stores directly. A buffer must have the output dtype and shape, be contiguous and (the row outputs) 16-byte aligned; one
that does not fit is refused with `ValueError`, never copied. The row output may be the logits tensor itself
(`softmax(x, out=x)`, `log_softmax(x, out_dtype=x.dtype, out=x)`: every thread writes the elements it read, after reading
them, so the in-place call gives the out-of-place bits -- tested over the register, re-read and staged members); any other
overlap between an output and an input, or between two outputs (a dtype view included), is refused. The samples buffer is int32 (the kernel's output), or int64
with int64 `indices` (filled by a cast from the kernel's int32 tokens, the operator's dtype contract). Without the buffers
the entries allocate their outputs as the operator's own kernels do (graph-safe through torch's private pools). The
inputs are read in place when they are in the kernels' form -- int64 or uint64 keys, an fp32 per-row temperature, int64
indices, contiguous rows -- and conditioned (a cast or a copy) otherwise; the validity mask is formed only when asked
for (`return_valid`). A probability row with a negative weight is outside the contract (the samplers' weights are non-negative) and is invalid on every path and placement: token `-1`, `valid` false, NaN mass and log-probabilities; `-0.0` is a zero, a NaN weight of either sign invalidates the row as before, and a logits input cannot produce a negative weight (`ex2` is never negative). The test costs nothing: every candidate filter compares the value (`w >= threshold`, the pattern test's set for non-negative weights), the list paths read the flag off the selection's own candidate count, and the dense tail's masked tree flags its leaves and poisons one partial (NaN mass) -- no extra pass, no extra exchange. `softmax`, `sampling_from_probs` and `sampling_from_logits` refuse the buffers on their default path
(`batch_invariant=False`: the operator's kernels allocate their own outputs) rather than copying into them; `log_softmax`
and `log_probs` run the class under either flag value and take them.

`ex2.approx.ftz.f32` and `lg2.approx.ftz.f32` are the class's two opaque primitives: deterministic per architecture,
pinned as inline PTX in the kernel, and tabulated on the device for the exact CPU model (`sampling/mufu_device.py`).
Everything else is a correctly rounded `.rn` operation or exact. The chains are the contract, named by the
fingerprints, stated in Lean and checked against the exact CPU model (`sampling/reference.py`); nothing is promised
relative to the operator's other softmax kernels or to torch.

The kernel (`sampling/kernel.py`) runs the two trees as the engine's plan over a placement of the configuration space
(`sampling/schedule.py`: 32 .. 1024 threads per row, rows per CTA, the vector width, fragments kept in registers /
re-read from L2 / staged in shared memory -- every member issuing at most 1024 loads per thread per pass, since the re-read
and staged forms unroll them -- and clusters of 2 .. 16 CTAs per row whose per-lane partials cross the
cluster through distributed shared memory), then writes the requested outputs in a third pass; the vocabulary is a
runtime argument (one kernel per power-of-two block count; elements beyond the row read `-inf`, the identity of both
trees). The dispatcher is a closed form of `(N, M, element bytes, SM count, which outputs)` with empirical constants
(`default_config`): clusters when there are fewer rows than SMs, one CTA per 64 / 128 KB of the row otherwise; every
pick is a member of the enumerated space and every member computes the class (the configuration-agreement tests).

The Lean model (`lean/BIKernels/Sampling.lean`) instantiates the engine's invariance theorems on the two trees
(`max_invariance`, `sum_invariance`), carries them through the chains (`lse_end_to_end`: for every valid placement and
unit the kernel's probabilities, log-softmax elements and `lse` are the spec's up to the commutativity the engine
normalises, hence `lse_bits_equal` under any commutative interpretation), and states the `-inf`-padded bucket kernel
bit for bit under the restricted identity hypotheses the spec's digits claim (`max_padded_bits`, `sum_padded_bits`).
`lake exe samplingconformance` prints its terms for the generated cases (`lean/BIKernels/SamplingCases.lean`,
rendered by `sampling/lean_cases.py`; kernel `decide` only), which the device-free test
`tests/batch_invariant/test_sampling_lean.py` compares with the Python specification. Tests:
`tests/utils/test_batch_invariant_sampling.py`.

The draw (`sampling/draw_kernel.py`) runs the row class's passes with a recording peers object (the butterfly partners
and the exchanged warp / CTA siblings of the sum pass are the tree's nodes), then one crossing pass per thread over its
own blocks and an exact integer minimum over the lanes, warps and CTAs of the first positive-weight crossing address; the uniform is
formed in the kernel from the row's key (`sampling/stream_kernel.py`: the Philox helpers as inline PTX integer ops, so
the compiler cannot reassociate them). Its configuration space is the row class's, at one row per CTA; its dispatcher
(`sampling/draw_schedule.py`) is a closed form with empirical constants: the two-pass pick when there are fewer rows than
SMs, a 256..512-thread re-read of the row otherwise, never a share above 512 elements per thread (the crossing pass
unrolls per element). The Lean model (`Draw.lean`) states the prefix dataflow as nodes of the canonical tree (so it is
as placement-invariant as the tree), the selection rule (`select`; `select_pos`: a drawn element has a positive weight, so a masked token is never drawn) and the coupling of the logits form's `S` with `lse.row`'s `s`; `Stream.lean`
decides the published Philox4x32-10 and SplitMix64 vectors; the generated `draw*` / `stream*` cases compare the terms
and the blocks with the Python specification. Exact CPU model: `sampling/draw_reference.py`. Tests:
`tests/utils/test_batch_invariant_sampling_draw.py`.

The support class `support.row.v1` (`sampling/support_spec.py`, `support_reference.py`, `support_schedule.py`; the device
code of its selection passes in `sampling/support_kernel.py`) is the kept set of a row under top-k / top-p / min-p filters
-- a function of the row's own values and the parameters alone, ties at the cut kept -- with its effective threshold, kept
count and kept mass: the selection stage of the five truncated samplers above (`sampling/draw_support_kernel.py`: the
support's passes, then the draw kernel's recording sum pass over the masked weights and its crossing pass, in one launch;
the generic entry `draw_support_api.draw_support`; exact model `draw_reference.draw_support_reference`) and of
`log_probs(..., top_k / top_p / min_p)` (the score form of the same kernel, `draw_support_api.kept_log_probs`). It has no
entry of its own: `top_k_renorm_probs` / `top_p_renorm_probs` / `top_k_mask_logits` keep their `is_deterministic=True`
kernels in this release (a flag there is deferred). The kernel
stages the CTA's share of the row in shared memory once (the draw kernel's loader) and runs every pass from it: the row
class's trees, then per filter a radix selection over the weights' bit patterns -- passes of 256-bin integer histograms
(`red.shared.add.u32`, hardware-aggregated; the top-p masses in 32-bit chunks of lane-spread replicas) each scanned from
the top by every warp with an exact integer prefix, a cluster's CTAs reducing their bins through `mapa`; a bf16
probability's fp32 pattern has 16 zero low bits and an f16 one 13, so two / three passes determine its threshold (fp32
inputs and the `ex2` weights of logits take four), and for the mass selection an early finish replaces the last passes by
one compaction sweep and an exact rank selection when the selected bin holds at most 64 candidates (a placement pinned
against the plain passes); with both filters in the `top_k_first` order the top-p candidates are the top-k kept set, so
they are compacted once and selected by rank whenever they fit the list (about `k` of them), the mass passes running
only when they do not; and for the four-pass kernels at the 128 KB shares the count selection's own passes after the
first run on a gathered list of the selected bin's candidates (CTA-local, no cluster reduction) whenever those fit it
(`count_list`, a placement pinned both ways) -- then the kept mass as the canonical tree over the kept weights. Every
number the kernel compares or sums is an integer or a declared `.rn` / `cvt.rzi` result, so the kept set and the threshold
are bit-identical in every member of the row class's space at one row per CTA; its dispatcher
(`sampling/support_schedule.py`) is a closed form of `(N, M, elem_bytes, sm_count)` fitted on regret sweeps of that space on
Rubin NVL8 and B200 (three regimes by how many rows share the device: the staged share per CTA, the cluster, the threads). The Lean model
(`Support.lean`) states the threshold procedures over pattern lists and proves what the class declares of them (top-k
keeps at least `k` and the cut one step up fewer; top-p reaches the target, no cut at a larger positive weight does, keeps
only positive weights and in the fallback every positive weight; the intersection of cuts is the cut at the larger
threshold) and the kept mass tree's placement invariance (`mass_invariance`); the generated `sup*` cases evaluate the
procedures on concrete pattern lists against the Python reference and the conformance executable prints the mass and
renormalisation terms. Exact CPU model: `sampling/support_reference.py`. Tests: the model's properties, the dispatcher's
membership and the placement pins in `tests/utils/test_batch_invariant_sampling_draw_support.py`.

## Linear attention (GDN)

`flashinfer.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=<fp32 pool>, state_indices=, cu_seqlens=,
use_qk_l2norm_in_kernel=True, gate_domain="log", batch_invariant=True)` runs one class, `gdn.bf16.c64`, for every
phase. The term is a kernel FlashInfer already ships: the chunked GDN prefill of
`gdn_kernels/blackwell/gated_delta_net_chunked.py` without context parallelism, which is deterministic, batch-invariant
and prefix-stable, and whose fp32 state at a multiple of 64 tokens is an exact resume point. The operator's other
routes (the context-parallel programs `auto` picks at low parallelism, cuDNN, Cake) compute different bits; the class
never takes them. Today's GDN decode kernels run a per-token recurrence whose bits differ from the prefill's, so a
token's decode output and its prefill output disagree; the class is what the mode promises instead.

A decode step does not update the state token by token. The pool slot holds the state at the sequence's last chunk
boundary; the class keeps, next to the pool and keyed by the pool's storage, the rows of the open chunk's solve that
row `p` depends on (the normalized keys, the log2 gates and betas, the M and X rows of the blocked inverse, the VKS
and NV rows), a bf16 copy of the boundary state and the open-row count. The step kernel
(`include/flashinfer/gdn/batch_invariant_decode.cuh`, one CTA per (request, v-head), one launch per step) computes row
`p` of the prefill from them: every tensor-core product of the prefill is a tcgen05 `kind::f16` fold, and the bf16
`mma.sync.aligned.m16n8k16` fold with the same operands, k-order and starting accumulator gives the same fp32 bits at
every stage shape, so the step issues them as `mma.sync` from registers and repeats the prefill's scalar epilogues
operation for operation (the file is built without fast math and with `-fmad=false`; the one fused multiply-add is the
prefill's own, in the 8 x 8 Gauss-Jordan). The 64th row of a chunk folds the next boundary state and writes it to the
pool. The step reads its count on the device and nothing on the host.

Calls with several tokens per sequence take placements of the same term, chosen by the call's shape alone
(`spec.STEP_MAX_TOKENS`). A call whose sequences bring fewer than 64 tokens is one launch of the step kernel's span
form: it appends each sequence's run of rows to its slot in a loop, with the open chunk's tiles resident in shared
memory, commits a chunk inside the loop when a run crosses a boundary, and checks `prefix_lens` against the slot's
count on the device, so it reads nothing on the host and can be captured in a CUDA graph (with 64 or more tokens in
the call, pass `cu_seqlens_cpu`: without it the offsets are read from the device). A sequence whose
`prefix_lens` is not where its slot's open chunk ends gets NaN rows and its slot is left as it was, as for a slot index
outside the pool: never another sequence's bits. A call that brings whole chunks runs them through the prefill kernel:
from chunk boundaries it is one prefill call in place over the caller's tensors, which also returns the rows of the
open tails and stores each sequence's last boundary state as a checkpoint in its pool slot. Such a call reads its slot
indices on the host and refuses a repeated or out-of-range slot before anything is written (inside a CUDA graph
capture a whole-chunk prefill does not read them and trusts the caller). One more launch keeps the
tails' raw rows next to the pool; their cached rows are built by one span launch before the slot's next step. Until
then the slot's count cell is negative and a step launched without that rebuild (a captured graph) returns NaN rows
for the slot. While any slot of the pool waits for its rebuild, the pool's next step or span call starts with that
rebuild; a slot restarted since is skipped on the device, and nothing is read on the host. Such a call cannot be captured
in a CUDA graph and raises, whichever slots it touches: a tail waiting in one slot blocks the capture of a step on
another, so run one eager call on the pool between a ragged prefill and a capture. The first call on a pool allocates
the state next to it and cannot be captured either (it raises). A sequence that continues an open chunk and brings whole
chunks first completes its chunk in a span launch. Per slot the class holds
`HV * (32 KB + 48.5 KB + 16 KB)` bytes next to the pool (4.6 MB at 16 / 48 heads), and
`64 * (HQ * 256 + HV * 264)` more (1.1 MB) once a prefill has left an open tail in the pool. It is allocated at the
pool's first call for every slot the pool's storage holds from the pool's offset, not for the slots of the view that
was passed (which is what lets views of one buffer share it): a pool carved from the start of a larger buffer pays for
the whole remainder, so give the pool a storage of its own.

What the caller owes the class:

* A call that carries a sequence's state passes the pool. Without `initial_state` (and without the other state
  arguments) the call does not raise: it is the stateless call below and returns rows from the zero state, so a decode
  step that drops its pool returns every token as a first token.
* A new sequence loaded into a slot starts with a prefill call (more tokens than sequences) or carries `prefix_lens`.
  A one-token call without `prefix_lens` continues what the slot holds: on a reused slot it continues the previous
  sequence, with finite rows that are not the new sequence's (nothing detects a slot whose state the caller replaced).
* A sequence moves to another slot, or is saved for reuse (a prefix cache), only at a multiple of 64 tokens, where the
  slot's fp32 state is the sequence's state. Inside a chunk the open rows live next to the pool and are not exported,
  so a copy of the fp32 state does not continue the sequence.
* Every call on a pool passes a view that starts at the pool's first slot (a view that starts at another slot is
  another pool, with its own open rows).
* Slot indices are unique and inside the pool. The calls that bring whole chunks check them on the host (except a
  whole-chunk prefill inside a CUDA graph capture); a call of fewer than 64 tokens per sequence checks the range on
  the device (NaN rows) and does not detect a repeated slot.
* `cu_seqlens_cpu` agrees with `cu_seqlens` and spares a call of 64 tokens or more the read of `cu_seqlens`; a decode
  call's `cu_seqlens` are `arange(N + 1)`. A call of fewer than 64 tokens without `cu_seqlens_cpu` does not check a
  device `cu_seqlens`: it stays inside its buffers, and rows that no sequence covers are left unwritten.
* The inputs, `output` and the pool are 16-byte aligned (a typed error otherwise), and `output` does not partly
  overlap an input (not checked, as for the operator without the flag).
* The inputs are finite. A NaN or inf in `k`, `v`, `g` or `beta` reaches every row of its 64-token chunk in a
  prefill and only the rows from its own on in decode, so the two differ on the rows before it; another sequence's
  rows are never affected.

`output_final_state` returns the pool. The decode entries (`gated_delta_rule_decode_pretranspose`,
`gated_delta_rule_mtp`) do not take the flag: they compute the gates from the raw parameters and keep another state
layout; a decode step of the class is a one-token call of `chunk_gated_delta_rule`.

### Stateless call: a trainer's forward pass

Without `initial_state` the same entry is a stateless call, the form a training forward takes: packed whole sequences,
every one from the zero state, rows only.

```python
out = torch.empty_like(v)          # once, reused every step
flashinfer.chunk_gated_delta_rule(
    q, k, v, g, beta, cu_seqlens=cu, cu_seqlens_cpu=offsets, output=out,
    use_qk_l2norm_in_kernel=True, gate_domain="log", batch_invariant=True,
)
```

* The rows of a sequence are bit for bit the rows the pool calls return for it (prefill, chunked prefill and decode),
  whatever the other sequences of the call, their order and their number. It is one launch of the class's prefill
  kernel with every sequence reading the zero state and storing none.
* No pool, no state next to a pool, no state returned: `state_indices`, `prefix_lens`, `output_state` and
  `output_final_state=True` raise the typed error without `initial_state`. A sequence that has to continue later is a
  pool call.
* With `output=` and a contiguous int32 `cu_seqlens` on the inputs' device the call keeps nothing after it returns and
  adds no allocation of its own, and reads the offsets back to check them. With `cu_seqlens_cpu` it runs from those
  host offsets instead (no read of a device tensor), so device and host offsets cannot disagree in it; only their lengths are compared.
  The host offsets are copied to the device once and the last 16 copies are kept (across streams, devices and the
  two forms of the call; a few int32 per sequence, and per stored state with `train_states`), so a call that repeats
  the offsets of an earlier one on the same stream copies nothing. Offsets that are not 1-D integers raise the typed error. The
  prefill underneath allocates, as in every prefill of the class, the normalized `q` and `k` (`2 * T * HQ * 256`
  bytes), the linear gates (`4 * T * HV` bytes) and a small kernel workspace, all freed at return. Per device and
  v-head count the class keeps one state tensor the kernel is pointed at and never reads or writes (3.1 MB at 48 v
  heads) and, for the call without `train_states`, an int32 index tensor, allocated at the first stateless call and
  regrown to twice the sequence count when a call brings more sequences than it holds (the replaced ones are kept,
  a few bytes per sequence).
* `output` may share no memory with an input, the offsets, `train_states` or one of the class's state pools (a pool
  owns its storage from its first slot to the storage's end), and an `output` allocated under `inference_mode` is
  refused in a call outside it (typed errors, before anything is written). The call bumps its autograd version (an
  inference tensor has none).
* It cannot be captured in a CUDA graph (it raises).
* **The rows carry no `grad_fn`.** The class has no backward: inputs that require grad are read as plain values and
  `out.requires_grad` is `False`. A trainer that needs gradients runs its own backward, for example its existing
  differentiable op on the same inputs wrapped in an `autograd.Function` whose forward returns these rows. The
  shortcut `ref + (out - ref).detach()` rounds in bf16 and does not return these bits. What that gradient is, and
  whether it is batch-invariant, is the trainer's.

**Intermediates for a backward pass: `train_states`.** The same call stores the state every 64-token chunk of every
sequence starts from in a caller-owned buffer:

```python
from flashinfer.batch_invariant.gdn import training_state_rows, training_state_slots

states = torch.empty(training_state_slots(seq_lens), HV, 128, 128, device=v.device)   # fp32, once per layer
flashinfer.chunk_gated_delta_rule(
    q, k, v, g, beta, cu_seqlens=cu, cu_seqlens_cpu=offsets, output=out, train_states=states,
    use_qk_l2norm_in_kernel=True, gate_domain="log", batch_invariant=True,
)
rows = training_state_rows(seq_lens)   # rows[i][j]: the state chunk j of sequence i starts from (zero for j = 0)
```

* A backward over the class's recurrence works from the inputs, the output gradient and, per 64-token chunk, the
  state the chunk starts from. That state is the one term it cannot recompute from the chunk's own inputs; without
  it a backward first re-runs the state scan over the sequence (FLA's backward and cuDNN's default path do). Every
  other term of a chunk (the normalized `q` / `k`, the cumulative gates, the triangular solve, the corrected values)
  is a function of the chunk's own 64 input rows and that state, so a backward evaluates the chunks independently of
  each other and only the state adjoint runs from chunk to chunk. Those per-chunk terms are internal to the kernel
  and are not exported, and no backward surveyed takes them from a forward.
* **The layout is the one cuDNN's GDN backward takes** (`cudnn::gated_delta_net_bwd`, argument `state_checkpoints`
  with `checkpoint_every_n_tokens=64`): `(n - 1) // 64 + 1` rows per sequence of `n` tokens, sequences in call order,
  a sequence's first row the zero state its first chunk starts from (the call writes the zeros) and row `j` the state
  after its first `64 * j` tokens; a state is `[HV, 128 (value), 128 (key)]`, one per v head (`state @ k` is the
  predicted value). The state after a sequence's last token is not stored. With the same `q`, `k`, `v`, `g`, `beta`,
  `cu_seqlens` and `scale`, `use_qk_l2norm_in_kernel=True` and `gate_domain="log"`, cuDNN frontend 1.32.3 takes the
  buffer as it is; a frontend that requires the series in the input dtype takes `states.to(torch.bfloat16)`. The
  gradients it returns from this series agree with the ones from its own recomputed series, and with an fp64 autograd
  of the token recurrence, to bf16 rounding (`tests/gdn/test_batch_invariant_gdn_stateless.py`, skipped without the
  frontend). cuDNN's own series is a bf16 rounding of its forward's states and is not these bits; the gradient is
  the backward's, and this class makes no bitwise statement about it. FLA's `h` is the transpose,
  `[..., key, value]`.
* Each stored state is bit for bit the state an inference prefill of the same prefix leaves in its pool, whatever the
  other sequences of the call. The output rows are the rows of the call without `train_states`.
* 64 KiB per v head and chunk, which approaches 1 KiB per token and v head for long sequences (48 KiB per token at
  48 v heads, four times the output) and is one state per sequence for sequences of at most 64 tokens. Rows past
  `training_state_slots` are left as they are. The layout's index tensors are kept with the offsets' copies.
* The buffer's autograd version is bumped by the call, so a trainer's `autograd.Function` that saved it can tell
  that a later call has overwritten it. `train_states` with `initial_state`, or a buffer that is not 16-byte aligned
  or shares memory with one of the class's state pools, the output, an input or the offsets, raises the typed error.
  A pool owns its storage from its first slot to the storage's end, so a buffer carved from the same allocation
  after a pool is refused too: give `train_states` an allocation of its own.

Tests: `tests/gdn/test_batch_invariant_gdn_stateless.py`.

`lean/BIKernels/Gdn.lean` models the two kernels as an opaque chunk step and an opaque row step and defines the
prefill as the fold of the chunk step over 64-row chunks. Four device facts carry the model, all checked bit for bit
on the device, for finite inputs, together with determinism and independence of the batch. Two are hypotheses of the
theorems: the chunk step is causal (the outputs of a chunk prefix are the prefix of the chunk's outputs) and the row
step returns the chunk step's last output on the same rows. Two are built into the definitions: a long prefill is the
fold of the chunk step (`stateAfter`, `prefillRow`), and the commit on a chunk's 64th row stores the chunk step's
state (`poolState`).
Proved from them: the decode row equals the prefill row at every position (`decode_eq_prefill`), the prefill kernel
run over whole chunks and an open tail returns the prefill's rows (`replay_eq_prefill`), the commit rule (the boundary state advances
on a chunk's 64th row) carries the prefill's boundary states (`pool_eq_stateAfter`), and a call's whole chunks and
tail are the position arithmetic (`extend_boundary`). The packed call is modelled as the per-sequence terms by
definition (`batch_term` only unfolds it); the pool's bookkeeping in `pool.py` (pending rebuilds, count cells) is
not modelled and is covered by the device tests. R3 is not claimed as a configuration-space statement: the kernels are hand-written and expose no configuration space; the fingerprint digests their sources.
`lake exe gdnconformance` prints the arithmetic for the generated cases (`lean/BIKernels/GdnCases.lean`, rendered by
`gdn/lean_cases.py`; kernel `decide` only), which the device-free test `tests/batch_invariant/test_gdn_lean.py`
compares with the Python specification. Tests: `tests/gdn/test_batch_invariant_gdn.py` (every comparison
`torch.equal`; one test runs a set of cases in two fresh processes and asserts that the second returns the hashes and
the case set of the first, and `FLASHINFER_GDN_BI_HASH_DIR` does the same for the whole file across two runs).
## Linear attention (KDA)

`flashinfer.recurrent_kda(q, k, v, g, beta, A_log=, dt_bias=, lower_bound=, initial_state=<fp32 pool>, ssm_state_indices=, cu_seqlens=,
use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, beta_is_logit=True, batch_invariant=True)` runs one class, `kda.bf16.c16`,
for every phase. The term is a kernel FlashInfer already ships: the chunked KDA prefill of `kda_kernels/kda_chunked_bt16.py` on
its persistent `engine` route, which is deterministic, batch-invariant and prefix-stable, and whose fp32 state at a multiple
of 16 tokens is an exact resume point. The kernel's other route (the two-kernel decomposition the operator picks when
`n_seq * heads * 2 <= sm_count`) computes different bits; the class never takes it. Today's `recurrent_kda` decode runs a
different kernel (a rank-1 recurrence on a bf16 or fp32 state) whose bits differ from the prefill's, so a token's decode
output and its prefill output disagree; the class is what the mode promises instead.

A decode step therefore does not update the state token by token. The pool slot holds the state at the sequence's last
chunk boundary; the class keeps, next to the pool and keyed by the pool tensor, what the open chunk needs per (slot, head):
a bf16 copy of the boundary state, the rows of the chunk's solve that later rows read (the decayed-inverse keys in bf16 and
fp32, the `rhs` rows, the bf16 `update` rows, the rows of `L`, the running log2 decay prefix; 21 KB), the open-row
count and a stamp of the state's first values. The term of decode row `p` is the prefill of the open chunk's rows from the
boundary state; the step kernel is a placement of it that computes only what row `p` adds: the two state projections against
the new key and query, row `p` of `K K^T` and `Q K^T`, the 16 x 16 inverse from the cached `L` rows, one `update` column and
the output row, each tensor-core product issued as `mma.sync` from registers with the prefill's operands, order and starting
accumulator, each scalar step with the prefill's operations (its fused gate increment included). The 16th row of a chunk
completes it in the same launch: the fp32 state is decayed, receives the chunk's update and is written back with its bf16
copy. A step is one launch that reads its count on the device, so it needs no host metadata and is one node of a captured
CUDA graph. A prefill takes the slot's state as a chunk boundary (any open rows are discarded), runs the sequence's whole
chunks through the chunked prefill kernel, which writes the state at the last boundary to the slot, and its tail through the
step kernel, which leaves the tail's rows cached. A slot whose state the caller replaced since the class last wrote it (a
new request) is seen by the stamp and starts a new chunk; a caller that loads a state with the same first eight values per head into a slot with open rows (a reset after a request shorter than 16 tokens, a saved boundary state loaded back)
must prefill the new request with more than one token, since a one-token call is a decode step and that slot's stamp has not
moved, or call `flashinfer.batch_invariant.kda.api.recurrent_kda_batch_invariant_reset(pool, slots)`, the explicit reset; without either the new request's decode rows are wrong and nothing raises (zeroing a slot that already holds zeros after five steps is the usual case). A decode step does not read its slots back: a sequence whose slot is outside the pool gets NaN rows and nothing else is written for it, and two sequences that name one slot in a step are the caller's error. The
call shape decides for every sequence of a call: in a prefill call a sequence with a single token also starts from its slot's
state as a boundary. The side state is keyed by the pool's storage, offset and shape, so any alias of the same extent (a full view, a `detach()`) is the
same pool, and a view of another extent that overlaps a pool with open chunks (`pool[:n]`) is refused; with none open the new extent takes over, the earlier extent's side state is kept until the buffer is freed (a graph captured on the earlier extent is stale from then on), and every side state of a buffer is released when its storage is; an `output` that shares memory with the pool is refused; the first call on a pool must be eager (a pool first used inside a graph capture is refused). A multi-token decode step (several tokens appended to an open chunk) has no spelling in `recurrent_kda`'s arguments
yet: `flashinfer.batch_invariant.kda.api.recurrent_kda_batch_invariant_extend` appends `[N, T]` tokens: fewer than 32 per sequence in one launch of the step kernel, which loops over the rows and commits the chunks it completes; longer calls with the rows that form whole chunks from a boundary through the prefill kernel and every other row through the step kernel, a split that is a function of the slot's row count and the call's shape alone. Besides the pool the class holds per slot `HV * (32 KB + 21 KB)`
(the bf16 state copy and the cached rows).

Performance is outside the guarantee. The prefill is the pinned route of the kernel `auto` already picks at most shapes;
the decode step moves the bf16 state (32 KB per (slot, head)) and the cached rows (up to 12 KB) per token where today's decode
reads and writes a bf16 state (64 KB), and the completing step of each chunk reads and rewrites the fp32 state on top
(about 212 KB), so it runs near a third of today's decode while the other fifteen run near or above it at large batches;
at small batches the step is bound by its dependent stages, not by bytes. Measured ratios are in the pull request that
added the kernel.

`lean/BIKernels/Kda.lean` models the kernel as an opaque chunk step and the prefill as its fold over the 16-row chunks: under the hypothesis that
the step is causal (the outputs of a chunk prefix are the prefix of the chunk's outputs, checked bit for bit on the device,
as is the step's determinism and its independence of the batch), the replayed decode row equals the prefill row at every
position (`decode_eq_prefill`), the pool's commit rule carries the prefill's boundary states (`pool_eq_stateAfter`), and
a packed call is the per-sequence terms (`batch_term`). R3 is not claimed: the kernel is hand-written and exposes no
configuration space; the fingerprint digests its source. `lake exe kdaconformance` prints the replay arithmetic for the
generated cases (`lean/BIKernels/KdaCases.lean`, rendered by `kda/lean_cases.py`; kernel `decide` only), which the
device-free test `tests/batch_invariant/test_kda_lean.py` compares with the Python specification. Tests:
`tests/kda/test_batch_invariant_recurrent_kda.py` (every comparison bitwise; `FLASHINFER_KDA_BI_HASH_DIR` records the
output hashes in one process and asserts them in another).

### Stateless call: a trainer's forward pass

Without `initial_state` the same entry is a stateless call, the form a training forward takes: whole sequences, every
one from the zero state, rows only.

```python
out = torch.empty_like(v, memory_format=torch.contiguous_format)  # once, reused every step
flashinfer.recurrent_kda(
    q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, cu_seqlens=cu, output=out,
    use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, beta_is_logit=True, batch_invariant=True,
)
```

* The rows of a sequence are bit for bit the rows the pool calls return for it (prefill, chunked prefill and decode),
  whatever the other sequences of the call, their order and their number. Every row comes from one launch of the
  class's prefill kernel over the whole sequences, built without a state input and without a state output; the step
  kernel is not involved, the sequence's last rows off a multiple of 16 included. A packed call is preceded, as a pool
  prefill is, by the kernel's small launch that orders the sequences on the device; it is sized for serving batches
  and its time grows with the square of the sequence count.
* Layouts and inputs are the pool calls': packed `[1, T, HV, 128]` with `cu_seqlens`, or `[B, T, HV, 128]`; bf16 `q` /
  `k` / `v` / `g`, bf16 `beta` logits, fp32 `A_log` and `dt_bias`, the same head counts. No call shape is a decode step:
  a sequence of one token is a first token.
* No pool, no state next to a pool, no state returned: `ssm_state_indices` and `output_final_state=True` raise the typed
  error without `initial_state`. A sequence that has to continue later is a pool call. A pool call that drops its pool
  no longer raises: it returns rows from the zero state, so a decode step without its pool returns every token as a
  first token.
* With `output=` (16-byte aligned, as allocator tensors are), int64 `cu_seqlens` and contiguous inputs a call
  allocates nothing on the device once its sequence count (packed) or its `[B, T]` (fixed layout) has been seen on the
  stream: the kernel on this route has no workspace and writes the caller's buffer. What stays behind is the prefill
  kernel's own metadata, one int32 per sequence for each sequence count seen (fixed layout: its offsets and order per
  `[B, T]`), never a state. The call reads `cu_seqlens` back once to check it. Inputs or gate parameters that are not
  contiguous, inputs that are not 16-byte aligned and an int32 `cu_seqlens` each cost one copy, a misaligned `output`
  a buffer and a copy.
* One `output` buffer per layer: the next call overwrites it, so a graph that saved its contents must not outlive
  that call (the call bumps the buffer's autograd version, so such a backward raises instead of reading the new rows).
* It cannot be captured in a CUDA graph (it raises).
* One call takes at most 65535 sequences (more raises).
* **What a backward needs from the forward.** `checkpoint_every_n_tokens=16` also returns `state_checkpoints`, the
  state before every 16-token chunk of every sequence, and the call then returns `(rows, None, state_checkpoints)`. Row
  zero of a sequence is its (zero) initial state, the next rows the states before tokens 16, 32, ...; a sequence of `n`
  tokens has `ceil(n / 16)` rows and the sequences follow each other in call order. In fp32 a row is bit for bit the
  pool's slot after a prefill of that many tokens; in bf16 (the default, and what the cuDNN frontend's KDA backward
  reads on its `state_checkpoints` port) it is that state rounded. The rows of the call do not change. Pass a
  `state_checkpoints` buffer (bf16 or fp32 `[rows, HV, 128, 128]`, at least that many rows; further rows are left
  alone) and `checkpoint_cu_starts` (int64 `[N + 1]`, the running count of rows) to keep the call allocation-free
  (the call reads `checkpoint_cu_starts` back on the host to check it);
  without them the call allocates both. One row is 32 KB per head in bf16, so 2 KB per head and token. Only 16, and only
  without a pool.
* **The rows carry no `grad_fn`.** The class has no backward: inputs that require grad are read as plain values and
  `out.requires_grad` is `False`. A trainer that needs gradients runs its own backward, for example its existing
  differentiable op on the same inputs wrapped in an `autograd.Function` whose forward returns these rows. The
  shortcut `ref + (out - ref).detach()` rounds in bf16 and does not return these bits. What that gradient is, and
  whether it is batch-invariant, is the trainer's.

Tests: `tests/kda/test_batch_invariant_kda_stateless.py`.
