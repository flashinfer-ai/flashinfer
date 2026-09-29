# Cake DSA sparse-attention training backend (SM100 / SM103)

Experimental backend for the native 64-query-head DeepSeek Sparse Attention
training kernels (top-k sparse MLA with absorbed queries; GLM-5.2 geometry:
64 heads, 512 latent + 64 rope query/key dimensions, 512 value dimensions,
top-k 2048).  Tracking issue: flashinfer-ai/flashinfer#5657.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture and SM103 (B300 / GB300)
is compiled from the same sources.  The feature is JIT-only and does not
participate in automatic backend selection, autotuning or trace-apply.

## Public entry points (`flashinfer/dsa_sparse_attention.py`)

```python
from flashinfer.dsa_sparse_attention import dsa_sparse_attention, dsa_sparse_attention_varlen

out = dsa_sparse_attention(q_latent, q_rope, kv_latent, k_rope, indices,
                           topk_length=None, softmax_scale=None, return_lse=False)
out = dsa_sparse_attention_varlen(q_latent, q_rope, kv_latent, k_rope, gather_kv_indices,
                                  cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                  topk_length=None, softmax_scale=None, return_lse=False)
```

* `q_latent [T, 64, 512]`, `q_rope [T, 64, 64]` BF16 (views of a packed
  `q [T, 64, 576]` are accepted); `kv_latent [S, 512]` (K = V) and
  `k_rope [S, 64]` BF16 (views of a packed `kv [S, 576]` are accepted).
* `indices [T, topk]` int32 hold **global** key rows; `-1` or `>= S` marks an
  invalid slot anywhere in the row; `topk_length [T]` int32 optionally
  invalidates slots `>= topk_length[t]`; any positive `topk`.
* The varlen form takes per-document indices (`gather_kv_indices`, relative
  to `cu_seqlens_k[d]`) and offsets them on device before the flat kernels
  run; a query's key set is fully described by its index row (the kernels
  apply no positional mask -- causality is the top-k selector's job).
* Forward: `out [T, 64, 512]` BF16 and (with `return_lse=True`) the natural-log
  `lse [T, 64]` FP32 over the valid keys; fully masked rows give `out = 0` and
  `lse = -inf`.  The autograd wrapper also keeps the BF16 output residual
  `o_lo = bf16(fp32(O) - bf16(O))` for the backward's exact `delta`.
* Backward: `dq_latent`, `dq_rope` (computed once per row, bitwise
  deterministic), `dkv_latent [S, 512]`, `dk_rope [S, 64]` (FP32 `red.global`
  accumulation, then a cast to natural-layout BF16).  BF16 MMA operands, FP32
  accumulation throughout.  `cake_backend.backward(..., dkv_fp32=True)` (and a
  runner prepared with `dkv_fp32=True`) returns the dK/dV gradients as
  natural-layout FP32 tensors instead; the kernels' internal accumulator layout
  (registry field `dkv_acc_layout`, `natural` or `permuted`) never crosses the
  API boundary -- a permuted program serves the FP32 mode through its cast
  stage.

Explicit forward / backward entry points without autograd, a prepared
allocation-free runner (`prepare_dsa_train`, CUDA-graph capturable) and the
workspace sizing helper live in `cake_backend.py`.  The eager entry points
(and the autograd wrapper behind the public API) validate and bind once per
input binding -- `(data_ptr, shape, stride, dtype)` of every input plus the
scale -- and launch later calls from the remembered argument plans with
freshly allocated outputs (`cake_backend.BINDING_CACHE`: no caller tensor
pinned, workspace scratch owned per binding under a FIFO capacity and a byte
budget; `FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE=0` disables it).

Host cost through the autograd wrapper: the `Function.backward` runs on
PyTorch's autograd device thread, where the two thread handoffs (about 30 us
each with an idle GPU, about 170 us each while kernels are queued on the
stream) and the Python body (3-4x slower there than on the main thread) add
roughly 400 us per backward at 4k tokens on B200 that are not in this package
-- a trivial `autograd.Function` with the same saved tensors and gradient
shapes shows the same cost, and no synchronization is involved.  In a
GPU-bound training step this is hidden behind the backward kernels (6-8 ms at
4k tokens).  Host-bound loops should call `cake_backend.forward` /
`cake_backend.backward` directly (about 25 / 40 us per call with a remembered
binding) or capture the prepared runner into a CUDA graph.

## Kernel structure of one training step

* `fwd`: one CTA per query token gathers its top-k keys once (TMA gather) and
  produces `out`, the natural-log `lse` and the BF16 output residual `o_lo`.
* `bwd_delta`: `delta = rowsum(dO * (O + O_lo))`, one (token, head) row per warp.
* `bwd_main`: one CTA per query token (20 warps: gather, compute, reduce, MMA,
  load and metadata roles) recomputes S and P from the BF16 Q and the gathered
  K, forms dP and dS, accumulates dQ / dQ_rope in tensor memory (written once
  per row: bitwise deterministic) and scatters the per-token dK/dV and dK_rope
  contributions with vectorized FP32 `red.global.add` into the accumulators.
* `bwd_cast`: converts the FP32 accumulators to the natural `[S, 512]` /
  `[S, 64]` BF16 outputs (or FP32 in the `dkv_fp32` mode).

Grid rules live in the registry record (`num_queries` CTAs for `fwd` and
`bwd_main`, `num_queries*8` for `bwd_delta`, `num_kv*18/256` for `bwd_cast`)
and are evaluated by the host from the problem scalars.

## Layout of this package

* `cake_jit.py` -- `MODULES` registry (one record per architecture, filled by
  the generated-program export), stage names and the JIT specs.
* `cake_backend.py` -- validation, workspace layout, varlen index offsetting,
  argument-plan binding, the prepared runner, the autograd `Function` and the
  eager entry points.
* `csrc/cake_dsa_h64_train/<arch>/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

The registry holds one record per architecture (`sm_100a`, `sm_103a`) with the
forward, backward preprocess, backward main and cast stages, exported from the
kernel snapshot named in the pull request.  The host
binding supports two argument profiles, selected by the record's `abi` field:
`dsa_h64_v1` (the native kernels) and `flashmla_v41_prefill_seed` (a
forward-only FlashMLA-derived prefill program used to exercise the export
pipeline; it produces no output residual, so backward is unavailable with it).

Tests: `tests/experimental/test_cake_dsa_train.py` (skips without a registered
program or a compute capability 10.0 / 10.3 device).  Benchmark:
`benchmarks/bench_cake_dsa_train.py`.
