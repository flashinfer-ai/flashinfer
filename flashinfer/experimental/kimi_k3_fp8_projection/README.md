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
| quantization launch + persistent 2-CTA GEMM | `quant:u<units>`, `gemm_tstore` (16-byte aligned output base and a row stride that is a multiple of 8 elements: TMA-store epilogue) or `gemm` (other strides: register epilogue); the `_n192` instances of the same epilogues on the tabulated `gemm_bn` rows (the N = 576 `kv_a` family at M > 256: three 192-column tiles instead of 768 padded columns) ; decode programs carry `_px<D>` when the row's table key `pfx` prefetches the BF16 token tile into L2 D stages ahead of its TMA load (round-6 next loop, lever PX on the small fused rows: the tile's DRAM access precedes the weight burst instead of queueing behind it) | `M > 256` unless the family is tabulated for the decode kernel at that row count, and the tabulated `(N, K)` families whose measured best route is the GEMM |
| quantization launch + decode | `quant:u1`, `decode:t<tok>_p<stages>` | measured table entry with `fused = false` |
| fused decode | `decode:t<tok>_p<stages>_fused[_res]` | measured table entry with `fused = true` (the token tile is quantized in-CTA; `_res` keeps the quantized token tiles resident for `K <= 256`); above 256 rows only the single-N-tile families (`f_a`, `b_proj`) are tabulated |

`decode_table.py` is the measured dispatch table
(`"<n_tiles128>,<num_k_iters>,<m_bucket>"`, buckets `M <= 1 / 8 / 64 / 256` for
every family and `M <= 4096 / 16384` for the families measured faster on the
decode kernel than on the GEMM)
over the 22 representative families (TP8 and TP1 `q_proj`, `fused_qkvg`,
`in_proj_qkvgfab`, `f_a`, `f_b`, `b_proj`, `kv_a`, `fused_qkv_a`, `q_b`,
`kv_b`, `o_proj`): one table shared by SM100 and SM103 (`DECODE_TABLE`) plus
the few cells whose measured best route differs per architecture
(`DECODE_TABLE_OVERRIDES[arch]`; `decode_table(arch)` merges them). It is
generated from the Cake sweeps and mirrors the production dispatcher row for
row. Families outside the table take the quantization + GEMM route (the Cake
dispatcher falls back to a calibrated cost model there; every representative
Kimi-K3 family is covered on both architectures).

| Tensor | Shape | dtype |
| --- | --- | --- |
| `weight` | `[N_pad128, K]` serialized checkpoint tensor (128-row block padding), contiguous; `K % 128 == 0` | float8_e4m3fn |
| `weight_scale` | `[N_pad128 / 128, 1, K / 128, 1]` ModelOpt block scale (or its 2-D view) | float32 |
| `x` | `[M, K]` activations, contiguous (the quantization program reads rows at stride `K`) | bfloat16 |
| `out` | `[M, n_valid]` view with unit column stride and an even row stride (any view into a wider buffer; only the `n_valid` columns are written) | bfloat16 |
| `workspace.q` | `[M, K]` quantized activation (caller-owned) | float8_e4m3fn |
| `workspace.sf` | `[bytes]` activation scale tiles + per-tile counters (zero initialised once by the allocation) + split-K partials (written before read; caller-owned) | uint8 |

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
route the device's table selects for `M` and zero-fills only its scale-tile and
counter bytes. `prepare_kimi_k3_fp8_projection` validates the binding, resolves
the route once (architecture and SM count are read once per device and cached)
and binds the generated argument plans; the runner reads `x` on device at
every launch, so a CUDA graph capturing it stays valid when new activations
are written into `x`.

Correctness: `|out - ref| <= 1e-2 + 1e-2 |ref| + 2 bf16 ulp` element-wise
against the exact quantized-operand emulation (zero budget), quantized
activation bit-exact vs the DeepGEMM recipe; see
`tests/experimental/test_cake_kimi_k3_fp8_projection.py`.

Benchmark: `python benchmarks/bench_cake_kimi_k3_fp8_projection.py [--cupti]`
times the 198 representative rows (22 families x `M in {1, 8, 64, 256, 512,
1024, 2048, 4096, 16384}`) as CUDA-graph replays with a cold L2 against FlashInfer's existing
`per_token_group_quant_8bit` + `gemm_fp8_nt_groupwise` chains (`cutlass` sm1 /
sm2, `trtllm`, `cutile`) on the same weight.

Generated sources live under `csrc/cake_kimi_k3_fp8_projection/`: one kernel
and one launcher source per program for both architectures (the
architecture-specific lowering sits behind `__CUDA_ARCH__` guards; the JIT
compiles each source with the flags of the device it runs on), the shared
device-helper and host-helper headers, and nothing else. `cake_jit.py`
registers them (`MODULES`: one record per program; `KERNELS`: logical kernel
key -> program and compile-line defines, e.g. the three quantization widths
are one program with `-DQUANT_UNITS=1|2|4`). Both are written by the Cake
generated-program export; nothing there is hand-written.
