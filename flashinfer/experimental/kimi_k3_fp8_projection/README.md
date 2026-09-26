# Experimental Kimi-K3 FP8_PB_WO projection GEMMs (Cake backend)

Both the API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4568 (Kimi-K3 kernels),
tracker #4254.

`flashinfer.gemm.kimi_k3_fp8_projection(x, prepared, out=None)` runs one
TP-local KDA / MLA projection of `nvidia/Kimi-K3-NVFP4` whose weight is
serialized as `FP8_PB_WO` (E4M3 `[N_pad128, K]` + ModelOpt 128x128 FP32
`weight_scale`) on SM100 (B200) and SM103 (B300/GB300):

```
a_q[m, k] = E4M3_rn(x[m, k] / s_a[m, k // 128]),   s_a = 2^ceil(log2(max(amax_128(x[m]), 1e-4) / 448))
w2, s2    = requant_ue8m0(weight, weight_scale)     (once: s2 = 2^ceil(log2 scale), w2 = E4M3_rn(weight * scale / s2))
out       = bf16(sum_k a_q s_a * w2 s2)              [M, n_valid], FP32 accumulation, one BF16 rounding
```

The activation recipe is DeepGEMM's `per_token_cast_to_fp8(use_ue8m0=True)`
(bit-exact), the weight recipe vLLM's `requant_weight_ue8m0`; the GEMM is
tcgen05 `kind::mxf8f6f4` block-scaled MMA with hardware scale application.
Every launch of the call is a generated Cake program:

| Route (host dispatch) | Programs | When |
| --- | --- | --- |
| quantization launch + persistent 2-CTA GEMM | `quant:u<units>`, `gemm` | `M > 256`, and the tabulated `(N, K)` families whose measured best route is the GEMM |
| quantization launch + decode | `quant:u1`, `decode:t<tok>_p<stages>` | `M <= 256`, measured table entry with `fused = false` |
| fused decode | `decode:t<tok>_p<stages>_fused[_res]` | `M <= 256`, measured table entry with `fused = true` (the token tile is quantized in-CTA; `_res` keeps the quantized token tiles resident for `K <= 256`) |

`decode_table.py` is the measured per-architecture dispatch table
(`"<n_tiles128>,<num_k_iters>,<m_bucket>"`, buckets `M <= 1 / 8 / 64 / 256`)
over the 22 representative families (TP8 and TP1 `q_proj`, `fused_qkvg`,
`in_proj_qkvgfab`, `f_a`, `f_b`, `b_proj`, `kv_a`, `fused_qkv_a`, `q_b`,
`kv_b`, `o_proj`); it is generated from the Cake sweeps and mirrors the
production dispatcher row for row. Families outside the table take the
quantization + GEMM route (the Cake dispatcher falls back to a calibrated cost
model there; every representative Kimi-K3 family is covered on both
architectures).

| Tensor | Shape | dtype |
| --- | --- | --- |
| `weight` | `[N_pad128, K]` serialized checkpoint tensor (128-row block padding), contiguous; `K % 128 == 0` | float8_e4m3fn |
| `weight_scale` | `[N_pad128 / 128, 1, K / 128, 1]` ModelOpt block scale (or its 2-D view) | float32 |
| `x` | `[M, K]` activations, contiguous | bfloat16 |
| `out` | `[M, n_valid]` view with unit column stride and an even row stride (any view into a wider buffer; only the `n_valid` columns are written) | bfloat16 |
| `workspace.q` | `[M, K]` quantized activation (caller-owned) | float8_e4m3fn |
| `workspace.sf` | `[bytes]` activation scale tiles + split-K partials + per-tile counters (caller-owned, zero initialised once) | uint8 |

```python
import torch
from flashinfer.gemm import (
    allocate_kimi_k3_fp8_projection_workspace,
    prepare_kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection_weights,
)

prepared = prepare_kimi_k3_fp8_projection_weights(weight, weight_scale, n_valid=6144, splits=(1536, 1536, 1536, 1536))
workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, M)      # once per M
runner = prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)     # binds the launch sequence
runner()                                                                 # no allocation; CUDA-graph capturable
q, k, v, g = prepared.output_views(out)                                  # fused projections: split views
```

`prepare_kimi_k3_fp8_projection_weights` requantizes the weight to UE8M0
scales, pads it to the 256-row CTA-pair tile and stores it as 128x128 E4M3
tiles in the order the TMA streams (one contiguous 32 KB box per pipeline
stage) together with the swizzled weight scale tiles; do it once per weight.
`allocate_kimi_k3_fp8_projection_workspace` sizes the byte workspace for the
route the device's table selects for `M`. `prepare_kimi_k3_fp8_projection`
validates the binding, resolves the route and binds the generated argument
plans; the runner reads `x` on device at every launch, so a CUDA graph
capturing it stays valid when new activations are written into `x`.

Correctness: `|out - ref| <= 1e-2 + 1e-2 |ref| + 2 bf16 ulp` element-wise
against the exact quantized-operand emulation (zero budget), quantized
activation bit-exact vs the DeepGEMM recipe; see
`tests/experimental/test_cake_kimi_k3_fp8_projection.py`.

Benchmark: `python benchmarks/bench_cake_kimi_k3_fp8_projection.py [--cupti]`
times the 132 representative rows (22 families x `M in {1, 8, 64, 256, 4096,
16384}`) as CUDA-graph replays with a cold L2 against FlashInfer's existing
`per_token_group_quant_8bit` + `gemm_fp8_nt_groupwise` chains (`cutlass` sm1 /
sm2, `trtllm`, `cutile`) on the same weight.

Generated sources live under `csrc/cake_kimi_k3_fp8_projection/<arch>/` and are
registered in `cake_jit.py` (`MODULES`, `KERNELS`) by the Cake
generated-program export; nothing there is hand-written.
