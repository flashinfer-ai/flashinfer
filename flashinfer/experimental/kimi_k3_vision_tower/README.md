# Experimental Kimi-K3 vision tower (Cake backend)

Both the API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4568 (Kimi-K3 vision
tower + PatchMergerV2 kernels), tracker #4254.

`flashinfer.kimi_k3_vision.kimi_k3_vision_tower(pixel_values, grid_thws,
weights, out=None)` runs the complete vision path of `nvidia/Kimi-K3-NVFP4`
(`modeling_kimi_k3.py`: `MoonViT3dPretrainedModel` with `merge_type =
sd2_tpool` + `PatchMergerMLPV2`) on SM100 (B200) and SM103 (B300/GB300) as a
sequence of generated tcgen05 programs:

| Stage | Program | Math (BF16 tensors, FP32 accumulation) |
| --- | --- | --- |
| `patch_embed` | `gemm:pos:*` | `x = bf16(pixels[T, 588] @ Wpe^T) + pos_rows` (Conv2d 14x14/14 as a GEMM, bilinear-resized 64x64 positional table + sincos time rows) |
| `layer_norm_qkv_rope` (x27) | `gemm:norm_qkv_rope:*` | `q, k, v = RoPE2D(bf16(RMSNorm(x, norm0) @ Wqkv^T))`, RMSNorm folded into the weight, row `rstd` in FP32 in the kernel, interleaved-pair 2-D RoPE in the epilogue, head-split outputs |
| `layer_attention` (x27) | `attention:tiles2` / `attention:tiles1` | packed-varlen noncausal attention per `grid_thw` segment, 12 heads x 128, FP32 softmax, one BF16 rounding |
| `layer_out_proj` (x27) | `gemm:residual_wo:*` | `x += bf16(a @ Wo^T)` |
| `layer_norm_fc0_gelu` (x27) | `gemm:norm_gelu:*` | `f = bf16(gelu_tanh(bf16(RMSNorm(x, norm1) @ Wfc0^T)))` |
| `layer_fc1` (x27) | `gemm:residual_fc1:*` | `x += bf16(f @ Wfc1^T)` |
| `final_norm_merge` | `merge` | `m = mean_t(bf16(RMSNorm(x, final_norm)))` over 2x2 spatial windows, `[N, 4096]` |
| `merger_gemm0` | `gemm:gelu_erf:*` | `h = bf16(gelu_erf(bf16(m @ Wp0^T)))` |
| `merger_gemm1` + `merger_rmsnorm_apply` | `gemm:rmsnorm:*`, `rmsnorm_apply` | `out = bf16(RMSNorm(bf16(h @ Wp1^T), post_norm, eps = 1e-5))` |

`RMSNorm` uses `eps = 2^-7` (`nn.RMSNorm(eps=None)` on BF16) except the
projector's post norm (`1e-5`). `*` is the production tile configuration the
host selects per token count (`cake_backend.select_tile_config`: 128x64 /
128x128 single-CTA tiles and a split-K cluster for small `T`, 256x256 /
256x128 2-CTA pair tiles above 1024 rows); the attention unit layout is chosen
per `grid_thws` batch from the LPT makespans of both layouts
(`cake_backend.select_tiles_per_cta`).

| Tensor | Shape | dtype |
| --- | --- | --- |
| `pixel_values` | `[T, 3, 14, 14]` normalized patches in packed `grid_thws` order (`t` slow, then `y`, then `x`), contiguous, 16-byte aligned | bfloat16 |
| `grid_thws` | host list of `(t, h, w)`; `1 <= t <= 4`, `h`, `w` even and `<= 512`; `T = sum t*h*w` | int |
| `weights` | `nn.Linear` `[out, in]` BF16 parameters without biases (below) | bfloat16 |
| `out` | `[N, 7168]`, `N = sum (h/2)*(w/2)` (optional, caller-owned) | bfloat16 |

`weights` is a dict: `patch_proj [1024, 588]` (the Conv2d weight flattened),
`pos_emb [64, 64, 1024]`, `time_weight [4, 1024]`
(`cake_backend.sincos_time_table()`), `final_norm [1024]`, `merger_proj0
[4096, 4096]`, `merger_proj1 [7168, 4096]`, `post_norm [7168]` and `layers`
= 27 x `{"norm0": [1024], "wqkv": [4608, 1024], "wo": [1024, 1536], "norm1":
[1024], "fc0": [4096, 1024], "fc1": [1024, 4096]}`. Prepare it once per model
with `prepare_kimi_k3_vision_weights` (norm folding, patch-projection
padding); the dict form is folded on every call.

```python
import torch
from flashinfer.kimi_k3_vision import kimi_k3_vision_tower, prepare_kimi_k3_vision_tower
from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import prepare_kimi_k3_vision_weights

prepared = prepare_kimi_k3_vision_weights(weights)          # once per model
grid_thws = [(1, 56, 74), (4, 52, 92)]                       # one 1024x768 image + one 4-frame 720p group
pixels = ...                                                 # bf16 [T, 3, 14, 14], T = 56*74 + 4*52*92
out = kimi_k3_vision_tower(pixels, grid_thws, prepared)      # bf16 [N, 7168], N = 28*37 + 26*46

# Serving form: plan once per grid_thws batch, launch without allocation, capture into a CUDA graph.
runner = prepare_kimi_k3_vision_tower(pixels, grid_thws, prepared, out)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    runner.launch()
graph.replay()                                               # new pixel values in the same buffer
```

`prepare_kimi_k3_vision_tower` derives everything a serving runtime keeps per
`grid_thws` batch (`cu_seqlens`, RoPE `cos`/`sin`, positional rows, the merge
table, the attention segment plan and unit table, the tile selection and all
workspaces) and binds the 27 x 5 + 5 launches; `launch()` performs no
allocation and no host synchronization. Pass a cached
`build_kimi_k3_vision_plan(...)` / `pos_emb_rows(...)` to skip the derivation.
Per-stage launches for tests and profiling: `runner.stages[name]()`.

Correctness (tests in `tests/experimental/test_cake_kimi_k3_vision_tower.py`):
every stage against the FP32 oracle of its operator on BF16 inputs at
`atol = rtol = 1e-2`; the complete call against the FP32 tower oracle with the
oracle-fairness gate of the Cake evaluation contract (error no worse than the
HF BF16 chain's); CUDA-graph replay bit-identical to the eager launch; no
allocation in `launch()`. Benchmark against the HF torch chain with the
FlashInfer ragged BF16 attention route:
`benchmarks/bench_cake_kimi_k3_vision_tower.py`.

The generated sources under `csrc/cake_kimi_k3_vision_tower/<arch>/` and the
`MODULES` / `KERNELS` registries in `cake_jit.py` are written by the Cake
generated-program export (`exports/kimi_k3_vision_tower/export.py` in the Cake
repository); do not edit them by hand. Every module is an exact-architecture
program compiled with FlashInfer's `sm100a` / `sm103a` flag sets; on a device
without a registered program the entry points raise `NotImplementedError`.
