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
`tests/moe/test_unified_moe_batch_invariant.py`, `tests/batch_invariant/test_lean.py`).

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
| `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper` | `backend="auto", batch_invariant=True` (constructor) | `attention.bf16` per head dim: 128-key tiles and 512-key chunks on the absolute KV axis, the online softmax step and the chunk merge of `attention/batch_invariant/spec.py`; `fingerprint(<wrapper name>, head_dim=...)`. The three wrappers (prefill and decode, paged and ragged KV) compute one class: a token's bits depend on its KV position, head and window alone | bf16 q / K / V / out; head dim 64 / 128 / 256; GQA (`num_qo_heads` a multiple of `num_kv_heads`, up to 128 per group); causal (bottom-right aligned) with an optional `window_left`; page sizes 1..128 (powers of two); NHD / HND. CUDA graphs: capture `run()` after `plan()`; a replay reproduces the planned batch and a new `plan()` needs a new capture (`use_cuda_graph=True` raises). `plan()` runs a host-side planner (of the order of 10 us per request). Not: soft cap, custom masks, fused RoPE, non-causal prefill, fp8 / fp4 KV, q / k / v scales, LSE output, sinks |
| `MoELayer` with `BatchInvariantBf16Config` (`flashinfer.fused_moe`) | `MoEConfig(..., backend=BackendOptions((BatchInvariantBf16Config(),)))`: named explicitly, never a default candidate (alone in the candidates the layer skips its shootout) | one class per typed activation of the layer (gated, on the (gate, up) pair: `SwiGLU` with any `alpha` / `beta` / `limit`, `SwiGLUStep`, `GeGLU`, `GeGLUTanh`, `SiTU`, `PowLU`; ungated, on one accumulator: `ReLU2`, `Identity`, `GELU`, `ReLU`, `SiLU`; the activation's scalars are binary32 constants of the chain, so a parameter set is a class): the class's own routing (the declared `exp`, the strict selection order `(score desc, index asc)` on exact binary32 scores, the method's weight formula; a NaN logit is skipped by the row maximum, contributes `+0` to the softmax sum and is never selected), the deterministic permutation of the expanded rows (expert order, then token-slot order), the two grouped GEMMs as placements of the unsplit member of the dense class, `gemm.bf16.kfold` at `K=H` / `K=I` (per expert one fp32 fold over all of K whatever the weight shape: the dense operator's chunk rule, which runs `gemm.bf16.kchunk[C]` for a narrow weight, does not apply, so `mm_bf16(batch_invariant=True)` on one expert's weight reproduces these bits only where that rule leaves K unsplit; the gate / up pair of a SwiGLU term in one thread's fragment; the chain's declared arithmetic on the fp32 accumulators: `cvt_bf16(mul(silu(g), u))` for the default `SwiGLU`, every sigmoid / tanh through the declared `exp`, GELU / GeGLU through a declared `erf`, PowLU through a declared `log` and the correctly rounded `sqrt`, the clamps as `fmin` / `fmax`), and the slot-order fp32 combine `cvt_bf16(fold_r add(acc, mul(y_r, w_r)))`; `fingerprint("fused_moe", H=, I=, top_k=, num_experts=, method=, act=, n_group=, topk_group=)` (`method` a `RoutingMethodType` the class implements, the groups for `DeepSeekV3`'s group-limited selection, `act` an `ActivationConfig` of the layer or `"swiglu"` / `"relu2"`) | bf16 `hidden_states (T, H)`, `H` and `I` multiples of 8, `1 <= top_k <= min(E, 16)`, `E <= 1024`; routing from logits (`RoutingInputMode.FromLogits`: fp32 or bf16 `(T, E)` logits, methods `Default`, `Renormalize`, `RenormalizeNaive`, `Sigmoid`, `SigmoidRenorm`, `TopK` (the logits as weights), `Llama4` (`top_k = 1`) and `TopKSigmoid` (the sigmoid of the chosen logits as weights), `DeepSeekV3` with or without expert groups (`n_group` dividing `E` into groups of at least two experts, `1 <= topk_group <= n_group <= 16`, `top_k <= topk_group * E / n_group`: the group score is the sum of a group's two largest biased scores), a bias exactly for the methods that select on `sigmoid(l) + bias`) or pre-routed ids and weights (`PackedPrecomputed` / `UnpackedPrecomputed`, bf16 or fp32 weights; any `RoutingConfig`, since the selection is the caller's: a routing term the from-logits mode has no class for is refused at a from-logits call, not at construction); weights from `BatchInvariantBf16Config.prepare_weights(w1 [E, 2I, H] (rows up, gate) or [E, I, H] for the ungated activations, w2 [E, H, I], activation=...)` with `E` the local experts of a shard; bf16 output, or with `do_finalize=False` the layer's triple `[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]` (the rows in the class's permutation, the weights as the combine folds them -- fp32 from logits, where the other backends return bf16: the fold widens a bf16 weight exactly, so a receiver that needs bf16 rounds them itself -- and the flat map with `-1` at a dropped slot: the class's slot-order fold over it is the finalized output bit for bit, as the public op `fused_moe.batch_invariant.api.combine_unfinalized(*triple)` computes it; `use_fused_finalize` is ignored); expert-parallel shards (`ExpertConfig.local_expert_offset` / `local_num_experts`: the rank's output is the dense fold restricted to its experts, the other slots dropped; the receiver's slot-order fold over the ranks' unfinalized rows is the dense output, the fixed-order sum of finalized partials the split path's within-placement form); no fused shared experts. SM100 / SM103 / SM107 |

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
the gather form against the materialised path (`test_gather_form_of_the_op_equals_the_materialised_path`). The autotuner's
tactics are the pairs of the top records of the two GEMMs; every pair computes the class. The persistent grid of a grouped GEMM
is the same launch parameter as the dense one's.

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
for bit. A second hardware hypothesis of the block-scaled classes: the block-scaled step gives the same bits with its two operands exchanged, which the swapped decode records (`swap_ab`: the weights as the kernel's A, the tokens as its B) rely on; the dispatcher takes those records up to 64 rows and the dense records above, so the rows of a 64-row and a 65-row call are compared bit for bit in `test_batch_invariance_along_rows` of the mxfp8 and fp4 suites (M = 63, 64, 65 among the row counts), and every swapped record against the dense one in `test_swapped_decode_records_compute_the_class`, on each part. The bf16 MoE classes' swapped records rest on the same hypothesis for the bf16 step (`mma16` with its two operands exchanged: the weights as the kernel's A, the tokens as its B, the accumulator transposed), which `test_swapped_records_equal_the_grouped_records` of `tests/moe/test_unified_moe_batch_invariant.py` discharges bit for bit on each part: every swapped record, in its box and its gather form, against a grouped record over groups of one to several token tiles. A third, of the block-scale fp8 class's promotion: the packed `fma.rn.f32x2` folds two elements per instruction and each of its lanes is the same correctly rounded fma as the scalar `fma.rn.f32`; the exact-model tests of `tests/gemm/test_batch_invariant_gemm_fp8_blockscale.py` discharge it bit for bit, the kernel's output against the exact CPU promotion (correctly rounded scalar fmas over the device block partials), on each part.

`Cases.lean` is generated from the Python specification and engine (`python -m flashinfer.batch_invariant.lean_cases`):
the Lean model reproduces the Python S-expressions node for node, re-derives the dispatchers' shape rules and checks
every record of a configuration space and every dispatcher pick against the class's constraints. The device-free test
`tests/batch_invariant/test_lean.py` byte-compares the committed file; its `lake` steps run when the toolchain is
installed.

The MoE classes have their own tier. `lean/BIKernels/MoE.lean` states the declared `exp`, sigmoid, tanh, erf and log
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
| `log_softmax(logits, temperature)` | the same class and fingerprint (`fingerprint("log_softmax", ...)`) | the row `logp_j = mul.rn(add.rn(z_j, neg(lg2.approx.ftz(s))), ln 2)` in fp32, shape `logits.shape` | as above; `batch_invariant=True` is the pinned contract, `batch_invariant=False` (the default) is the operator's regular kernel for this entry, today the same class (a future release may route it elsewhere) |
| `log_probs(logits, token_ids, temperature, return_lse=)` | the same class and fingerprint (`fingerprint("log_probs", ...)`) | the log-softmax row above gathered at the row's token ids (NaN for an id outside `[0, N)`), and with `return_lse` the row's `lse = mul.rn(add.rn(mul.rn(m, c), lg2.approx.ftz(s)), ln 2)`; fp32 outputs of the ids' shape / `logits.shape[:-1]` | as above; `token_ids` int32 / int64 of shape `logits.shape[:-1]` (one id per row) or `logits.shape[:-1] + (K,)` (`K` ids per row in one launch); the `batch_invariant` flag as for `log_softmax` |
| `sampling_from_probs(probs, indices, generator, seed, offset, return_valid, batch_invariant=True)` | `sample.icdf.v1`; `fingerprint("sampling_from_probs", N=, dtype=)` (the vocabulary is the dimension; the dtype is validated and enters nothing: the leaves of a 16-bit input are its exact widenings and the draw converts no output) | the inverse CDF read off the row class's canonical sum tree `S` of the given weights (the same `add.rn` tree as `lse.row`'s `s`): the threshold `t = mul.rn(u, S)` with `u` the fp32 uniform of the row's stream; the vocabulary walked in the tree's order (lane-major, then block, then element), the inclusive prefix of an element `add.rn(outer, fold_e)` where `fold_e` is the slot fold's running value and `outer` the `add.rn` chain, in ascending bit order from `+0`, of the left-sibling nodes of the tree the element's address names (every one a value the kernel's placement already holds: no new reduction, no division); the token is the first present element with a positive weight whose prefix is `>= t` (the positivity is part of the rule: the last prefix of a slot and the `outer` of the next are different `add.rn` chains over the same nodes, so a zero-weight slot head could otherwise be the first crossing), else the last with a positive weight; a row whose `S` is not a positive finite number is invalid (token `-1`, `valid` False) | bf16 / f16 / fp32 weights `>= 0` of shape `(..., N)`, any `N` as for `softmax`; `indices` int32 / int64 (the gather; the output takes its dtype; an index outside the source rows makes its output row invalid: token `-1`); `seed` / `offset` as the operator takes them, one key per output row (int64 / uint64 tensors of length M; a scalar, a length-one tensor or the pair reserved from `generator` exactly as the operator's path reserves it serves a single row -- shared by several rows it is refused, since every row would draw at the same uniform); `deterministic` ignored (deterministic by construction), `check_nan` the same host check; int32 tokens; SM100 / SM103 / SM107 |
| `sampling_from_logits(logits, indices, generator, seed, offset, batch_invariant=True)` | `sample.icdf.v1`; `fingerprint("sampling_from_logits", N=, dtype=)` (the logits form embeds `lse.row`'s trees: a different fingerprint per vocabulary from the probabilities form) | the same draw over the `lse.row` weights `e_j = ex2.approx.ftz(mul.rn(add.rn(x_j, neg(m)), log2(e)))` of the row (no temperature: the entry has none), without materialising the probabilities; `S` is `lse.row`'s `s`, so the drawn token's log-probability under `log_probs` is the probability it was drawn with (R4, stated in Lean: `Draw.total_logits_eq_specSum`) | as above for logits (a row of `-inf` logits, or with a NaN, is invalid) |

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
