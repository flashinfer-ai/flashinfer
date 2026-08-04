# WAN FlashInfer Backend Benchmark

End-to-end forward-pass latency of the FlashInfer WAN transformer with every
supported GEMM backend, on Hopper (H100 PCIe) and Blackwell (B200).

## Methodology

- **Script**: `examples/pytorch/wan/transformer_wan_flashinfer.py`
- **Attention backend**: `single` (fixed across runs to isolate GEMM cost)
- **Batch size**: 1
- **Input**: `num_frames=12, height=64, width=64` → visual seq length 12288
  - WAN patch size is `(1, 2, 2)`, so visual tokens = `frames × (H/2) × (W/2)`
- **Iterations**: 2 warmup + 5 measured, `torch.cuda.synchronize` between.
  CUDA-graph runs use 3+ stream-warmup forwards before capture, then 10
  replays.
- **Dtype**: bfloat16 weights + activations (cast inside quantized backends)
- **Stack**: PyTorch 2.11 (`nvcr.io/nvidia/pytorch:26.03-py3`), CUDA 13,
  flashinfer 0.6.7 (editable), diffusers 0.38.0, `nvidia-cutlass-dsl` 4.5.1
- **GPUs**:
  - H100 PCIe 80 GB (compute capability 9.0) — `ipp1-3210` *(early online-quant sweep)*
  - H100 80 GB HBM3 (SXM5, compute capability 9.0) — `ipp2-0061` *(torch.compile + offline-quant sweep)*
  - B200 192 GB (compute capability 10.0) — `umbriel-b200-047` *(early sweep)*
  - B300 SXM6 AC 192 GB (compute capability 10.3) — `umb-b300-020`, `umb-b300-dp-141` *(torch.compile sweeps)*
- **Activation quantization**: the early per-GPU tables below use the
  `online_act_quant` default (compute scale from the activation tensor).
  The `torch.compile + CUDA graph` section near the bottom switches to
  `--offline-act-quant` (fixed placeholder scale) for clean cuda-graph
  capture — see that section and the "Online vs. offline" caveat near
  the end of this doc.
- **Models**:
  - **FP4 = NVFP4.** The `fp4` backend in this example uses
  `flashinfer.nvfp4_quantize` + `flashinfer.mm_fp4(..., use_nvfp4=True)`
  (the default). MXFP4 (`mxfp4_quantize`) is a separate quantization scheme
  not wired into this example.

Models:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (30 blocks, hidden=1536, ffn=8960, 306
    Linear layers)
  - `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (40 blocks, hidden=5120, ffn=13824, 406
    Linear layers) — only the `transformer` subfolder, treated as a single
    14B transformer (the full A14B pipeline has a second high-noise twin
    transformer; we don't load that here)

`Speedup = torch_baseline_time / backend_time` (>1 means faster than torch).

## H100 PCIe (sm90)

### WAN-1.3B (baseline: torch = 254.19 ms)

| Backend       | online (ms) | offline (ms) | best speedup vs torch |
|---------------|------------:|-------------:|----------------------:|
| **torch**     |    **254.19** |          —  | 1.00× |
| fp8_sm90      |      277.70 |       278.28 | 0.92× |
| bmm_fp8       |      329.18 |       339.60 | 0.77× |

Backends gated by `_check_gemm_backend_support` to SM100+ (`bf16`,
`fp8_groupwise`, `fp8_blockscaled`, `fp4`, `mxfp8`, `bmm_bf16`,
`bmm_mxfp8`) silently fall back to torch on sm90 with a warning listing
the required SM range, so we don't list them again.

### WAN-14B (baseline: torch = 1432.52 ms)

| Backend       | online (ms) | offline (ms) | best speedup vs torch |
|---------------|------------:|-------------:|----------------------:|
| torch         |    1432.52  |          —  | 1.00× |
| **fp8_sm90**  |  **1326.51**|     1333.50 | **1.08×** |
| bmm_fp8       |    1456.80  |     1438.10 | 1.00× |

**Finding on H100 PCIe:** `fp8_sm90` clearly beats the cuBLAS bf16 baseline on
the 14B model (~8% faster). The same backend was 9% slower on 1.3B — the
crossover point sits between these two sizes, exactly where the FP8 memory-
bandwidth savings start to exceed the per-call quantization overhead. PCIe
H100s, which are bandwidth-limited relative to SXM5, are precisely the regime
where this backend pays off.

## B200 (sm100)

### WAN-1.3B (baseline: torch = 152.74 ms)

| Backend            | online (ms) | offline (ms) | best speedup vs torch |
|--------------------|------------:|-------------:|----------------------:|
| **torch**          | **152.74**  |          —  | 1.00× |
| mxfp8 ¹            |   164.02    |          —  | 0.93× |
| bmm_bf16           |   166.26    |          —  | 0.92× |
| bf16               |   167.35    |          —  | 0.91× |
| fp8_groupwise      |   245.76    |    **181.71** | 0.84× |
| fp8_blockscaled    |   249.14    |    **184.40** | 0.83× |
| bmm_fp8            |   190.93    |     205.40  | 0.80× |
| fp4                |   201.15    |          —  | 0.76× |
| fp8                |   285.50    |     293.02  | 0.52× |
| batch_deepgemm_fp8 |   ❌ cubin not registered | ❌ | — |

### WAN-14B (baseline: torch = 692.47 ms)

| Backend            | online (ms) | offline (ms) | best speedup vs torch |
|--------------------|------------:|-------------:|----------------------:|
| **torch**          | **692.47**  |          —  | 1.00× |
| mxfp8 ¹            |   711.27    |          —  | 0.97× |
| bmm_bf16           |   743.44    |          —  | 0.93× |
| bf16               |   743.48    |          —  | 0.93× |
| fp4                |   792.23    |          —  | 0.87× |
| fp8_blockscaled    |  1022.61    |    **797.12** | 0.87× |
| fp8_groupwise      |  1023.68    |    **798.05** | 0.87× |
| bmm_fp8            |   791.49    |     797.96  | 0.87× |
| fp8                |  1454.61    |    1460.73  | 0.48× |

¹ `mxfp8` falls back to torch for the single `Linear(*, N=64)` projection in
the modulation head (the kernel needs `N ≥ 128` and `K ≥ 128`); the other 305
(1.3B) / 405 (14B) Linear layers go through `mm_mxfp8`.

**Finding on B200:** no FlashInfer GEMM backend currently beats the cuBLAS
bf16 baseline on either WAN size **at the end-to-end model level** — the
closest is `mxfp8` at 3% slower on 14B. But this is **NOT** because the
underlying kernels are slow. An isolated kernel benchmark
([`bench_kernels_isolated.py`](bench_kernels_isolated.py),
WAN-14B Linear shapes, M=12288) shows the opposite story:

| Kernel                   | hidden→hidden (N=5120) | hidden→ffn_up (N=13824) | ffn_down (K=13824, N=5120) |
|--------------------------|-----------------------:|------------------------:|---------------------------:|
| torch.matmul bf16         |                0.396 ms |                  1.071 ms |                    1.066 ms |
| mm_bf16                   |     0.403 ms (0.98×)   |       1.078 ms (0.99×)  |         1.092 ms (0.98×)   |
| **mm_fp4 / NVFP4**        |  **0.137 ms (2.89×)**  |   **0.306 ms (3.51×)**  |     **0.287 ms (3.72×)**   |
| **mm_mxfp8**              |  **0.233 ms (1.70×)**  |   **0.653 ms (1.64×)**  |     **0.668 ms (1.60×)**   |
| gemm_fp8_nt_groupwise     |  0.326 ms (1.22×)      |       0.902 ms (1.19×)  |         0.931 ms (1.14×)   |
| **mm_fp8 (TRT-LLM)**      |  **1.677 ms (0.24×)**  |   **4.512 ms (0.24×)**  |     **4.017 ms (0.27×)**   |

NVFP4 and MXFP8 kernels are **1.6×–3.7× faster** than cuBLAS bf16 on B200, and
the groupwise FP8 path is 1.1×–1.2× faster. So the kernels themselves do
deliver the expected Blackwell tensor-core speedup. The end-to-end gap comes
from the `FlashInferLinear` Python wrapper, which spends per-layer time on:

- Activation cast / quantize on every forward (even `--offline-act-quant`
  still runs the `(x * scale).clamp().to(fp8)` elementwise pass).
- Separate bias add (`out + bias.to(out.dtype)`).
- Output dtype cast back to bf16.
- One additional Python frame and a handful of attribute lookups per layer.

WAN-14B has **406 Linear layers**; even ~0.1 ms of pre/post per layer adds
~40 ms to the forward pass, which is enough to wipe out the kernel-level
savings on this workload. The torch baseline runs each Linear as a single
fused cuBLAS call from a C++ stub, so it has near-zero per-layer overhead.

**`mm_fp8` is genuinely slow** (4× slower than bf16): it's the TRT-LLM
"low-latency" per-tensor FP8 path which is tuned for *batch=1 decode* shapes,
not the 12K-token prefill GEMMs WAN runs.

### WAN 720p × 5s (realistic inference shape)

The 12×64×64 shape used above keeps GEMM cost a small fraction of the
forward; at the realistic WAN-2.2 720p × 5s shape
(`num_frames=21, height=90, width=160`, visual seq ≈ 75K tokens), the
attention compute is much larger and GEMM choice matters less in absolute
terms. We still ran the full matrix here because it's the shape users will
actually deploy.

WAN-14B, batch=1, 720p × 5s latent, B200, after wrapper optimization
(see "Wrapper hot-paths" below):

| Backend            | online (ms) | offline (ms) | best vs torch |
|--------------------|------------:|-------------:|--------------:|
| torch (baseline)   |       14185 |        14197 | 1.00× |
| **fp4 (NVFP4)**    |   **13680** |    **13694** | **1.037×** ✓ |
| mxfp8 ¹            |       14216 |        14210 | 1.00× (noise) |
| bf16               |       14440 |        14445 | 0.98× |
| bmm_bf16           |       14448 |        14441 | 0.98× |
| fp8_groupwise      |       15492 |    **14813** | 0.96× |
| fp8_blockscaled    |       15461 |    **14747** | 0.96× |

Same shape on H100 PCIe, after wrapper optimization:

| Backend            | online (ms) | offline (ms) | best vs torch |
|--------------------|------------:|-------------:|--------------:|
| torch (baseline)   |       17503 |        17686 | 1.00× |
| fp8_sm90           |       18276 |        18734 | 0.96× |
| bmm_fp8            |       18944 |        18760 | 0.92× |

Note that H100's available FlashInfer backends (`fp8_sm90`, `bmm_fp8`) don't
go through the same code path the wrapper opt fixed — they were already
using the simpler `_quantize_activation_fp8_per_tensor` (one amax pass).
The opt is essentially a no-op on H100. Despite this, an **isolated kernel
benchmark on H100 at exactly the 720p Linear shapes** shows the FP8 kernels
are 1.3×–1.9× faster than cuBLAS bf16:

| Kernel @ M=75600                | qkv/out (N=5120) | ffn_up (N=13824) | ffn_down (N=5120) |
|---------------------------------|-----------------:|-----------------:|------------------:|
| torch.matmul bf16               |          7.83 ms |         24.65 ms |          24.96 ms |
| **fp8_blockscale_gemm_sm90**    | **6.11 (1.28×)** | **16.93 (1.46×)**| **19.83 (1.26×)** |
| **bmm_fp8 (cublas FP8)**        | **4.17 (1.88×)** | **12.87 (1.92×)**| **14.99 (1.67×)** |

So the kernels themselves should save ~1.4 s of GEMM time on H100. nsys
shows the FP8 GEMMs do come in faster (combined ~2.4 s vs torch's ~3.9 s),
but ~2.2 s of unaccounted time on the FP8 path eats the win. The
candidates we identified from the profile (and which can't be addressed
inside the example):

- `tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel` — the
  internal activation quantizer that `fp8_blockscale_gemm_sm90` runs as a
  prologue, 406 calls × ~1 ms = ~400 ms.
- `direct_copy_kernel` — 362 calls × ~1.3 ms = ~480 ms of dtype-cast
  and `.contiguous()` traffic around the FP8 GEMM.
- Output-side `out + bias.to(bf16)` and any internal bf16↔fp8 casts.

The example can't fold those into the GEMM — that would require the
kernel itself to expose a fused-bias / fused-cast epilogue.

At 720p the workload is dominated by self-attention
(~79% of GPU time per nsys — 80 calls × ~142 ms each for the 75K-token
`SinglePrefillWithKVCacheKernel`), so even a 3× FP4 GEMM kernel only moves
the total by a few percent. To get bigger end-to-end wins at this shape
you'd need a faster attention path (sparse / FP8 KV cache / TGV).

### Wrapper hot-paths (why quantized backends started out slower)

An nsys profile of `fp4` at 720p showed the dominant non-attention,
non-GEMM kernels per forward:

| Kernel                        | calls | total | mean |
|-------------------------------|------:|------:|-----:|
| `SinglePrefillWithKVCacheKernel` (attention) | 80 | 11,407 ms | 142.6 ms |
| `direct_copy_kernel` (contiguous / dtype cast) | 808 | 650 ms | 0.80 ms |
| FP4 GEMM (`nvjet_*_Avec16UE4M3_*`) | 320 | 297 ms | 0.93 ms |
| `AbsFunctor<float>` (from `.float().abs()`) | 446 | 174 ms | 0.39 ms |
| `nan_to_num_kernel` (from `.nan_to_num()`) | 446 | 172 ms | 0.39 ms |
| `reduce_kernel<float>` (from `.max()`) | 446 | 99 ms | 0.22 ms |

The `x_global_sf = (448*6) / x.float().abs().nan_to_num().max()` line was
spending more time **promoting bf16 to fp32 and running paranoia kernels**
than the FP4 GEMM was saving. After replacing it with
`x.abs().amax().to(torch.float32).clamp(min=1e-12)` and adding a fast path
to skip the padding+copy in `_quantize_activation_fp8_blockwise` when `K`
is already a multiple of `block_size`:

| Backend (B200 14B 720p)  | before opt  | after opt  | delta |
|--------------------------|------------:|-----------:|------:|
| torch                    | 14202 ms    | 14185 ms   | ≈0 |
| **fp4 (NVFP4)**          | 14340 ms    | **13680 ms** | **−660 ms** |
| fp8_groupwise (offline)  | 14833 ms    | 14813 ms   | small |
| fp8_groupwise (online)   |             | 15492 ms   | (per-block amax dominates; needs a fused activation-quant kernel) |

This is the simple lesson hiding behind the original "FlashInfer is slower
than torch" measurement: the GEMM was already faster, but the
`FlashInferLinear` activation-quant prologue was running 4–5 unnecessary
elementwise passes per layer × 406 layers. Stripping `.float()` and
`.nan_to_num()` recovers a 3.7 % end-to-end win for fp4.

### CUDA-graph attempt (does **not** help on WAN-sized workloads)

The example also supports `--cuda-graph`, which captures the forward pass
once and replays it (stream-warmup ≥3 iters, then capture, then time
replays). Tested at three shapes on B200, and at the realistic
**720p × 5s** latent shape (`num_frames=21, height=90, width=160`,
visual seq ≈ 75K tokens) on both H100 PCIe and B200:

WAN-14B, 12×64×64 input, B200:

| Backend                 | no graph | `--cuda-graph` | delta |
|-------------------------|---------:|---------------:|------:|
| torch                   | 783.40 ms | 782.32 ms     |  -0.1% |
| fp4 (NVFP4)             | 818.53 ms | 815.28 ms     |  -0.4% |
| mxfp8                   | 787.65 ms | 785.88 ms     |  -0.2% |
| fp8_groupwise (offline) | 880.37 ms | 879.83 ms     |  -0.1% |

WAN-14B, **720p × 5s** latent (21,90,160), B200:

| Backend                 | no graph    | `--cuda-graph` | delta |
|-------------------------|------------:|---------------:|------:|
| torch                   | 14202.23 ms | 14220.65 ms    | +0.1% |
| fp4 (NVFP4)             | 14340.88 ms | 14337.92 ms    |  0.0% |
| mxfp8                   | 14217.36 ms | 14208.39 ms    | -0.1% |
| fp8_groupwise (offline) | 14833.50 ms | 14819.67 ms    | -0.1% |

WAN-14B, **720p × 5s** latent (21,90,160), H100 PCIe:

| Backend     | no graph    | `--cuda-graph` | delta |
|-------------|------------:|---------------:|------:|
| torch       | 17546.00 ms | 17827.20 ms    | +1.6% |
| fp8_sm90    | 17854.92 ms | 18676.41 ms    | +4.6% |
| bmm_fp8     | 18472.24 ms | 18983.98 ms    | +2.8% |

(On H100 PCIe the captured graph is consistently a bit *slower*; we suspect
the cached allocator under the cuda-graph mempool gets less efficient when
the 28 GB 14B weights leave little headroom on the 80 GB card. Worth
revisiting on H100 SXM.)

CUDA graph captures only the CPU-side per-launch overhead, so it can only
help when the GPU has idle time between kernels. nsys on the 720p forward
shows ~2,638 kernel launches taking ~10 ms of cumulative CPU time, while
the GPU spends ~14,200 ms running the captured kernels — i.e. CPU launch is
already <0.1% of wallclock. There's nothing for the graph to save on this
shape.

**The implementation is verified.** A synthetic sanity check
([`cuda_graph_sanity.py`](cuda_graph_sanity.py) — 200 stacked tiny
`Linear(256, 256)` layers, batch=1) shows the same `--cuda-graph` capture
pattern delivers a **6.80× speedup** (2.27 → 0.33 ms/iter) when the GPU is
genuinely idle between launches. So the negligible WAN result is workload
behavior, not a broken capture.

WAN inference would benefit from CUDA graph in two regimes the
single-forward benchmark doesn't exercise:

- **Multi-step diffusion sampling** (25–50 steps): each diffusion step is a
  forward pass, and capturing the step lets the host skip 2.6K launches
  per step. Still a small absolute saving but adds up over a 50-step run.
- **Smaller resolutions / shorter clips** where each kernel takes
  microseconds instead of milliseconds. There the launch overhead is a
  measurable fraction of wallclock.

→ **Takeaway:** the path to real end-to-end speedup on WAN 720p is
**epilogue / prologue fusion**, not graph capture. The activation
quantization and bias add need to live inside the GEMM kernel call (as
cuBLAS/CUTLASS epilogues do), or the activations need to stay in FP8/FP4
across layers so the cast isn't repeated. The example as it stands is a
correctness reference, not a performance reference.

### torch.compile + CUDA graph (the combined knob)

The example supports both `--torch-compile` (Inductor fusion across each
`WanTransformerBlock`) and `--cuda-graph` (manual outer capture). Used
together — `--torch-compile --torch-compile-mode default --cuda-graph` —
Inductor folds the per-layer activation-quant / bias-add / dtype-cast
prologue (the "Wrapper hot-paths" bottleneck above) into the surrounding
ops, then the captured graph replays once per forward with no Python
overhead. The `reduce-overhead` / `max-autotune` modes do their *own*
internal CUDA-graph capture and conflict with the outer `--cuda-graph`
(stale-pointer errors when FlashInfer kernels break the graph); the
example refuses that combination and recommends `mode=default` +
`--cuda-graph` instead.

WAN-14B, **720p × 5s** latent (21, 90, 160), PyTorch 2.11
(`nvcr.io/nvidia/pytorch:26.03-py3`), flashinfer 0.6.7,
`--attention-backend single`, **`--offline-act-quant`** (fixed
placeholder activation scale — speed-only proxy, not a drop-in
production-accuracy setting), 2 warmup + 5 timed iters. Every config
re-loads the same checkpoint, then runs in two modes: **eager** (no
flags) and **compile + cg** (`--torch-compile --torch-compile-mode
default --cuda-graph`). Each `gemm_backend` is reported with its
default kernel; the `<base>-<kernel>` suffix syntax is exercised
side-by-side where the kernel matters.

#### H100 80 GB HBM3 (sm90)

| `--gemm-backend`         | eager (ms) | compile + cg (ms) | vs torch/eager |
|--------------------------|-----------:|------------------:|---------------:|
| torch (baseline)         |   11377.85 |          9768.78  | 1.165× |
| fp8_sm90                 | ❌ kernel asserts on offline scale | 9392.15 | 1.211× |
| bmm_fp8 (`cublas`)       |   11474.63 |     **8995.77**   | **1.265× ✓ best** |
| bmm_fp8-`cudnn`          |   11656.78 | ❌ cudnn/dynamo stream bug | — |
| bmm_fp8-`cutlass`        | ❌ "does not support backend 'cutlass' with capability 90" | — | — |

`bmm_fp8` (default cuBLAS backend) under `compile + cg` is the best
config on H100 at this shape — **1.265× over the cuBLAS-bf16 baseline**.
`fp8_sm90` is close behind at 1.211×; it can only be measured under
compile here because the offline-quant placeholder scale
(`1.0 / fp8_max`) is small enough to trigger a `result == CUDA_SUCCESS`
assertion in `fp8_blockscale_gemm_sm90`'s W8A8 path in eager.

#### B300 SXM6 AC (sm103)

| `--gemm-backend`              | eager (ms) | compile + cg (ms) | vs torch/eager |
|-------------------------------|-----------:|------------------:|---------------:|
| torch (baseline)              |   12552.12 |         11563.60  | 1.085× |
| bf16 (default `cudnn`)        |   12819.93 | ❌ cudnn/dynamo  | — |
| bf16-`cutlass`                |   12820.00 | ❌ still routes through cudnn | — |
| fp4 (default `auto`)          |   12376.32 |         11003.52  | 1.141× |
| **fp4-`cutlass`**             |   12364.93 |     **11001.18**  | **1.141× ✓ best** |
| fp4-`trtllm`                  |   12510.98 |         11125.28  | 1.128× |
| mxfp8 (default `auto`)        |   12676.61 |         11477.84  | 1.094× |
| mxfp8-`cute-dsl`              |   13294.11 |         12309.48  | 1.020× |
| fp8_groupwise (default `cutlass`) | 13129.49 |     11721.11  | 1.071× |
| fp8_groupwise-`trtllm`        |   15199.82 |         13620.51  | 0.922× |
| bmm_bf16 (default `cudnn`)    |   12820.01 | ❌ cudnn/dynamo  | — |
| bmm_bf16-`cutlass`            |   14934.78 |         13721.43  | 0.915× |
| bmm_fp8 (default `cublas`)    |   13007.39 |         11134.11  | 1.127× |
| bmm_fp8-`cudnn`               |   13008.49 | ❌ cudnn/dynamo  | — |

The `fp4-cutlass + compile + cg` and `fp4 (auto)` numbers are within
2 ms of each other — the autotuner picks the cutlass kernel on this
shape anyway. The earlier (online-quant, run-2) **regression** to
22 s on `fp4 + compile + cg` is gone: it was caused by the offline-scale
path allocating fresh `torch.tensor(...)` / `torch.full(...)` buffers
inside the captured graph, which made cuda-graph replay land on stale
pointers. The wrapper now caches those constant scale tensors
(`_offline_*_scale` lazily-allocated buffers) so the captured graph
sees a stable address.

**Observations across both GPUs.**

1. **`<base>-<kernel>` suffix matters most for the bf16 family.**
   `bf16` and `bmm_bf16` default to `cudnn`, which still hits
   `'torch.Stream' object has no attribute 'cuda_stream'` under dynamo
   (a flashinfer-main bug, not this example's). On B300 even
   `bf16-cutlass` is currently routed through the cudnn graph builder
   internally — so picking the suffix doesn't yet rescue this family
   from the dynamo crash. `bmm_bf16-cutlass` does run under compile + cg
   (no crash), but is slower than the cuBLAS bf16 baseline.
2. **FP8 paths win on both GPUs.** `bmm_fp8 + cublas + compile + cg` is
   the universal winner — 1.265× on H100, 1.127× on B300 — because
   per-tensor amax + scale + cast around a cuBLAS FP8 GEMM is exactly
   the shape Inductor can fuse cleanly.
3. **`fp8_groupwise` benefits enormously from compile** (the prior
   online-quant version was the most dramatic case — 30 s eager → 11.7 s
   compile+cg). In offline mode the eager number is already reasonable
   (13.1 s on B300) so the compile win shrinks to 1.12×, but the
   ordering against torch flips from "slower" to "faster".
4. **NVFP4 / MXFP8 autotuner suffix variants compare cleanly.**
   For `fp4`: `cutlass` ≈ `auto` (≈ 11.00 s under compile+cg),
   `trtllm` is ~1% slower. For `mxfp8`: explicit `cute-dsl` is ~7%
   slower than `auto` (which picks the SM-tuned CUTLASS path).
   The earlier run-2 regression on these backends was the offline
   scale-tensor bug above, not an autotuner instability.
5. **`fp8_groupwise-trtllm` is consistently slower than the cutlass
   default**, both eager and compile + cg — not worth picking on
   either GPU at this shape.

**Verified backend × GPU × suffix matrix under `compile + cg` (works ✓ /
crashes ❌ / regressed earlier ⚠️):**

| | H100 (sm90) | B300 (sm103) |
|---|---|---|
| torch                       | ✓ 9768.78 | ✓ 11563.60 |
| bf16 (cudnn / cutlass / tgv)| n/a       | ❌ ❌ — |
| bmm_bf16 (cudnn / cutlass)  | n/a       | ❌ ✓ 13721.43 |
| fp8_sm90                    | ✓ 9392.15 | n/a |
| bmm_fp8 (cublas / cudnn / cutlass) | ✓ 8995.77 / ❌ / n/a | ✓ 11134.11 / ❌ / — |
| fp4 (auto / cutlass / trtllm) | n/a    | ✓ ✓ ✓ |
| mxfp8 (auto / cute-dsl)     | n/a       | ✓ ✓ |
| fp8_groupwise (cutlass / trtllm) | n/a  | ✓ ✓ |

**Usage.**
```bash
# Best on H100 80 GB HBM3 (sm90) at 720p × 5s
python examples/pytorch/wan/transformer_wan_flashinfer.py \
  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --subfolder transformer \
  --batch-size 1 --num-frames 21 --height 90 --width 160 \
  --warmup-iters 2 --benchmark-iters 5 \
  --attention-backend single \
  --gemm-backend bmm_fp8 --offline-act-quant \
  --torch-compile --torch-compile-mode default --cuda-graph

# Best on B300 SXM6 AC (sm103) at 720p × 5s
python examples/pytorch/wan/transformer_wan_flashinfer.py \
  ... --gemm-backend fp4-cutlass --offline-act-quant \
  --torch-compile --torch-compile-mode default --cuda-graph
```

## VSA + NVFP4 + Ulysses on 8× B200 (720p × 5s, end-to-end generation)

Everything above measures a single transformer forward. This section measures
**full video generation** — 50 denoising steps, VAE decode, mp4 export — on
8× B200 with Ulysses context parallelism, and adds the two knobs the earlier
sections identified as missing: a **faster attention path** (VSA) and a
**faster all-to-all** (FlashInfer's NVLink-P2P Ulysses kernel).

### Setup

- **Model**: `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` — Wan2.1-14B
  finetuned *with* VSA, so the checkpoint carries the extra per-block
  `to_gate_compress` projection VSA's gated combine needs. Every row uses this
  same checkpoint; only the attention and GEMM paths change.
- **Config** (FastVideo's published inference settings for this checkpoint):
  720×1280, 81 frames, 50 steps, guidance 5.0, flow-shift 5.0, seed 1024,
  VSA sparsity 0.9, fps 16.
- **Shapes**: latent 21×90×160 → post-patch grid **21×45×80 = 75,600 tokens**,
  sharded 9,450 tokens/rank across 8 ranks; 40 heads → 5 heads/rank.
- **Hardware**: `umbriel-b200-094`, 8× B200 180 GB on NVSwitch, driver-level
  all-pairs NVLink.
- **Stack**: `nvcr.io/nvidia/pytorch:26.03-py3` (PyTorch 2.11, CUDA 13),
  flashinfer 0.6.17 (editable), diffusers 0.38, `quack` from source.
- **Timing**: 2 warmup denoising steps (absorbs JIT + autotune), then 50 timed
  steps via `callback_on_step_end`; the first timed interval is dropped from
  the per-step mean. Model load is excluded.
- **Attention baseline**: `--attention-backend torch` (SDPA) everywhere, so
  the VSA rows change *only* self-attention. Cross-attention stays SDPA in all
  rows.

### Results

| # | Config | GEMM | Self-attention | Ulysses | denoise (s) | ms/step | speedup |
|---|--------|------|----------------|---------|------------:|--------:|--------:|
| 1 | baseline | torch bf16 (cuBLAS) | torch SDPA | NCCL | 83.26 | 1664.6 | 1.00× |
| 2 | nvfp4 dynamic | `mm_fp4`, online act-quant | torch SDPA | NCCL | 80.21 | 1603.5 | 1.04× |
| 3 | nvfp4 static ¹ | `mm_fp4`, offline act-quant | torch SDPA | NCCL | 76.86 | 1536.8 | 1.08× |
| 4 | **nvfp4 static ¹ + VSA** | `mm_fp4`, offline | **VSA (sparsity 0.9)** | NCCL | **64.11** | **1281.4** | **1.30×** |
| 5 | nvfp4 static ¹ + VSA + NVLink A2A | `mm_fp4`, offline | VSA (sparsity 0.9) | **NVLink-P2P** | 64.28 | 1284.5 | 1.30× |
| 6 | nvfp4 dynamic + VSA | `mm_fp4`, online | VSA (sparsity 0.9) | NCCL | 67.61 | 1351.4 | 1.23× |
| 7 | nvfp4 dynamic + VSA + NVLink A2A | `mm_fp4`, online | VSA (sparsity 0.9) | NVLink-P2P | 67.78 | 1354.7 | 1.23× |

¹ Speed upper bound only — see the caveat below. Rows 3–5 are not
quality-preserving; **row 6 is the fastest numerically valid configuration**
at 1.23×.

The two quantization modes and the two Ulysses backends compose cleanly:
static buys the same ~4% over dynamic whether or not VSA is on (1536.8/1603.5
= 0.96, 1281.4/1351.4 = 0.95), and NVLink-vs-NCCL is a wash in both pairs.

Each row was captured with a generated video (`--output`), a timing JSON
(`--timing-json`), and an Nsight Systems trace of the timed denoising steps;
see "Reproducing" below for the exact invocations. On the run machine these
live under `/home/scratch.forrestl_wwfo/wan_vsa_videos/`.

### VSA is where the win is (+17–18% over nvfp4 alone)

VSA cuts 252 ms/step off row 2 (dynamic) and 255 ms/step off row 3 (static) —
the same absolute saving, as expected for a change that only touches attention.
An isolated measurement at exactly the per-rank shape (75,600 tokens, 5 heads,
head_dim 128, bf16) accounts for almost all of it:

| Self-attention, per rank per layer | time | ×2 CFG × 40 layers | vs dense |
|------------------------------------|-----:|-------------------:|---------:|
| torch SDPA (dense)                  | 10.26 ms | 0.82 s/step | 1.00× |
| **VSA sparsity 0.9** (topk 144/1440)| **7.22 ms** | **0.58 s/step** | **1.42×** |
| VSA sparsity 0.75 (topk 360)        | 10.03 ms | 0.80 s/step | 1.02× |
| VSA sparsity 0.50 (topk 720)        | 15.08 ms | 1.21 s/step | 0.68× |

Predicted saving 0.24 s/step, measured 0.25 s/step — the end-to-end delta is
fully explained by the attention path.

Two things are worth noting. First, **0.9 sparsity buys only 1.42×, not 10×**,
and VSA is a *loss* below ~0.8 sparsity. Second, the sparse kernel is only half
the cost. Stage breakdown at sparsity 0.9 (sums to the 7.22 ms above):

| Stage | time | share |
|-------|-----:|------:|
| `bsa_attn_blk64_fwd` (fine stage) | 3.45 ms | 48% |
| **tile: cube gather + padded scatter** | **2.02 ms** | **28%** |
| top-k(144) + sort | 0.66 ms | 9% |
| block masked mean ×3 | 0.64 ms | 9% |
| combine + untile | 0.27 ms | 4% |
| stack q/k/v/gate | 0.12 ms | 2% |
| coarse attention (1440 pooled tokens) | 0.07 ms | 1% |

The **tile step is the obvious target**: it moves ~860 MB per layer, which at
B200 HBM bandwidth should take ~0.1 ms, but costs 2.02 ms because it is an
`int64` fancy-index gather feeding an uncoalesced scatter. A fused
tile-into-padded-buffer kernel would recover ~1.9 ms/layer ≈ 0.15 s/step,
roughly another 11% end-to-end. The `(4,4,4)` cube layout also forces 21.9%
padding at this grid (21×45×80 → 24×48×80), which inflates the fine stage by
~1.5× on its own.

### Where the GPU time goes (nsys)

Nsight Systems traces of the timed denoising steps, all 8 ranks, bucketed by
kernel. Summed kernel time ÷ (8 ranks × 3 steps) reproduces the measured
per-step latency to within 1%, so the GPU is essentially never idle and these
shares are directly meaningful.

| Config | GPU s (8 ranks × 3 steps) | attention | VSA tiling | GEMM | all-to-all | other |
|--------|--------------------------:|----------:|-----------:|-----:|-----------:|------:|
| 1 baseline | 39.8 | **47%** (cuDNN SDPA) | — | 18% | 8% | 27% |
| 2 nvfp4 dynamic | 38.3 | 48% (SDPA) | — | 6% | 9% | 38% |
| 3 nvfp4 static | 36.8 | 50% (SDPA) | — | 6% | 8% | 36% |
| 4 nvfp4 static + VSA | 30.4 | **14%** (VSA fine) | **12%** | 7% | 6% | 60% |
| 5 … + NVLink A2A | 30.2 | 14% | 12% | 7% | 9% | 57% |
| 6 nvfp4 dynamic + VSA | 31.9 | 13% (VSA fine) | 11% | 7% | 6% | 62% |
| 7 … + NVLink A2A | 31.6 | 13% | 12% | 7% | 9% | 59% |

Three things the profile confirms independently of the microbenchmarks:

1. **Dense attention really is ~half the workload** (47%, 9.66 ms per layer per
   rank for `cudnn_generated_..._sdpa_sm100_flash_fprop_...` — the isolated
   measurement said 10.26 ms). That is why VSA moves the needle and NVFP4 does
   not: dropping GEMM from 18% to 6% only buys 4%.
2. **The tile permute costs nearly as much as the sparse kernel it feeds.** The
   `index_elementwise_kernel` gather (948 µs) and `index_put_kernel` scatter
   (946 µs) per layer per rank sum to 1.89 ms against the VSA fine stage's
   2.20 ms — 12% vs 14% of total GPU time. This matches the 2.02 ms measured in
   the stage breakdown above, from a completely different measurement path, and
   makes the fused-tile-kernel optimization the clearest remaining win.
3. **The NVLink kernel shows a *larger* all-to-all share (9% vs 6%) at equal
   total time**, which is exactly what the decomposition below predicts: it
   absorbs the permute that NCCL performs separately (and which lands in
   "other" for the NCCL rows), so it does more work inside the collective.

The "other" bucket grows in the VSA rows (60% vs 27%) partly because the total
shrinks and partly because VSA adds real elementwise work — masked block means,
top-k, the gated combine, and the extra `to_gate_compress` projection.

Regenerate any of these with the `nsys profile` invocation in "Reproducing"
below.

### The NVLink Ulysses kernel ties here — WAN 720p sits exactly at its crossover

Rows 4 and 5 are identical within noise (64.11 vs 64.28 s), as are rows 6 and 7
(67.61 vs 67.78 s). That is *not*
because there is nothing to win: the NCCL path (`_nccl_scatter_heads`) really
does run a full-tensor permute+`.contiguous()` before the collective, which the
fused NVLink-P2P kernel folds into its cross-GPU writes. At the production
shape that permute costs **0.115 ms, 39% of NCCL's 0.300 ms scatter**.

The fused kernel gives that back on raw transfer. Decomposing the collective
across payload sizes (8 ranks, 40 heads, head_dim 128, bf16; each number is a
burst of 20 collectives, barriered, median of 10 bursts):

| S_local | payload | permute alone | raw `all_to_all_single` | NCCL total | NVLink fused | NVLink win |
|--------:|--------:|--------------:|------------------------:|-----------:|-------------:|-----------:|
| 128 | 1.3 MB | 0.008 ms | 0.026 ms | 0.040 ms | **0.020 ms** | **2.07×** |
| 512 | 5.2 MB | 0.008 | 0.029 | 0.041 | **0.027** | **1.54×** |
| 2048 | 21.0 MB | 0.021 | 0.063 | 0.087 | **0.056** | **1.57×** |
| 3072 | 31.5 MB | 0.029 | 0.083 | 0.124 | **0.099** | **1.25×** |
| 4096 | 41.9 MB | 0.041 | 0.101 | 0.155 | **0.141** | 1.10× |
| 6144 | 62.9 MB | 0.076 | 0.132 | 0.210 | 0.202 | 1.04× |
| **9450 (WAN 720p)** | **96.8 MB** | 0.115 | 0.183 | **0.300** | **0.302** | **0.99×** |
| 18900 | 193.5 MB | 0.226 | 0.328 | 0.556 | 0.586 | 0.95× |

Two regimes, and the reason is bandwidth, not topology:

- **Small payloads (≤ ~30 MB): the fused kernel wins 1.25–2.07×.** NCCL pays a
  separate permute launch plus the collective; the fused kernel does both in one
  kernel. At 1.3 MB the permute and the transfer are both launch-bound, so
  collapsing two launches into one is worth 2×.
- **Large payloads (≥ ~97 MB): it ties, then loses.** NCCL's raw all-to-all
  reaches 529 GB/s at 96.8 MB and 590 GB/s at 193.5 MB (payload ÷ time), while
  the fused kernel plateaus at ~320–330 GB/s. So the fused path saves the
  0.115 ms permute but spends ~0.12 ms more moving the same bytes. Net zero at
  WAN's 96.8 MB, and slightly negative beyond.

So the optimization is real and the intuition behind it is right — the WAN 720p
shape just lands on the crossover. It should pay off for LLM-scale Ulysses
(4K–32K tokens per rank is a few MB per collective, squarely in the 1.5–2×
region), and it would pay off for video diffusion too if the fused kernel closed
the ~1.7× raw-bandwidth gap against NCCL at ~100 MB payloads. That gap, not
NVSwitch topology, is what to chase.

For context on scale: all-to-all is ~120 ms of a 1281 ms VSA step (9.4%), so
even a 2× faster collective would only buy ~5% end-to-end at this shape.

The two backends are also **numerically identical**: the same seed produces
latents that match to the last bit.

### Correctness

`vsa_sanity.py` covers the VSA path itself: tile/untile is an exact
round-trip, the cube permutation really groups each `(4,4,4)` neighbourhood,
the kernel matches an independent eager reference to ≤6e-3 relative error at
sparsity 0/0.5/0.9, and splitting heads across ranks does not change the result.

For the sequence-parallel path, comparing 1-GPU against 8-GPU latents after one
denoising step:

| Path | cosine similarity, 1 GPU vs 8 GPU |
|------|----------------------------------:|
| dense SDPA | 0.99983 |
| VSA, sparsity 0.0 (all blocks kept) | 0.99979 |
| VSA, sparsity 0.9 | 0.917 |

The first two show Ulysses is correct. The third is **expected and not a bug**:
top-k block selection is discrete, so bf16-level differences in the pooled
scores flip which blocks a query block attends to, and 40 layers × 2 CFG passes
amplify that. Dense attention cannot exhibit this because it is
permutation-invariant once RoPE is baked in. The practical consequence is that
**VSA output is not reproducible across different world sizes** — the samples
are equally valid, not equal. Setting sparsity to 0 restores reproducibility.

### Caveats

- **The static rows (3, 4, 5) are speed numbers only.** `--offline-act-quant`
  uses a fixed placeholder scale rather than calibrated per-layer scales (see
  the "Online vs. offline" section below). Their ~4% over the matching dynamic
  row is a real upper bound on what removing the online amax pass can buy, but
  the videos are visibly washed out — frame mean 124/255 and std 21 for row 3
  against 55/73 for the baseline. **Row 6 (dynamic + VSA, 1.23×) is the
  fastest configuration that is also numerically valid.**
- **VSA needs a VSA-finetuned checkpoint.** The gated combine reads
  `to_gate_compress`, which stock Wan checkpoints do not have; running
  `--use-vsa` on one silently uses a randomly initialized gate.
- Rows 1–3 run *dense* attention on a VSA-finetuned checkpoint. That is the
  controlled comparison (same weights, only attention changes), but it is not
  the same as the original Wan2.1-14B release.
- The `vsa_blackwell_blk64` kernel is SM100-only, bf16-only, and requires
  `head_dim == 128`.

### Reproducing

```bash
torchrun --nproc_per_node=8 examples/pytorch/wan/pipeline_wan_flashinfer.py \
  --model-id FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 --flow-shift 5.0 --seed 1024 \
  --gemm-backend fp4 --attention-backend torch \
  --use-vsa --vsa-sparsity 0.9 --ulysses-backend nccl \
  --warmup-steps 2 --timing-json timing.json --output wan_vsa.mp4
```

That is row 6. Add `--offline-act-quant` for row 4, swap
`--ulysses-backend nvlink` for rows 5/7, drop `--use-vsa` for rows 2/3, or use
`--gemm-backend torch` for row 1. `--check-rank-sync` asserts every rank holds
identical latents at every step.

To capture an Nsight Systems trace of just the denoising steps (the model load
would otherwise dominate the report):

```bash
nsys profile -o row6 --trace=cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  torchrun --nproc_per_node=8 examples/pytorch/wan/pipeline_wan_flashinfer.py \
    ... --num-inference-steps 3 --warmup-steps 2 --output-type latent \
    --cuda-profiler-range
```

## Online vs. offline activation quantization

The `--offline-act-quant` flag only affects backends that go through
`_quantize_activation_fp8_per_tensor` or `_quantize_activation_fp8_blockwise`
(per-tensor: `fp8`, `bmm_fp8`; blockwise: `fp8_groupwise`, `fp8_blockscaled`,
`batch_deepgemm_fp8`). Backends with their own quantizer (`fp4`, `mxfp8`,
`bmm_mxfp8`) ignore the flag.

The split is bimodal:

| Backend          | online → offline speedup (1.3B) | online → offline speedup (14B) |
|------------------|--------------------------------:|-------------------------------:|
| fp8              | 1.00× (essentially noise)       | 1.00× |
| bmm_fp8          | 0.93×                           | 0.99× |
| **fp8_groupwise**| **1.35×**                       | **1.28×** |
| **fp8_blockscaled** | **1.35×**                    | **1.28×** |

For per-tensor FP8 quantization, the activation amax+scale is one CUDA call
and cheap; online vs offline is in the noise. For block/groupwise FP8, the
amax must be computed over each `(M, K_block)` tile (~3× more reductions per
activation), and that step dominates — going offline saves 25–35%.

**⚠️ Caveat:** the current `--offline-act-quant` implementation uses a
**fixed** scale (`1.0` for per-tensor, `1.0 / fp8_max` for blockwise). This
is a placeholder. Real offline quantization requires per-layer scales
collected from a calibration pass; using `1.0` everywhere will produce
incorrect activations whenever the actual amax differs significantly from
`fp8_max`. Treat the offline numbers in this report as **upper-bound speed
estimates**, not as drop-in production settings.

## Known issues discovered during benchmarking

- **`batch_deepgemm_fp8` cubin missing** on B200 (and presumably elsewhere):
  ```
  ValueError: cubin not registered: kernel.fp8_m_grouped_gemm.fe49fe4304f7
  ```
  Both online and offline trigger the same lookup miss. The grouped-FP8
  cubin appears to not be registered in flashinfer's JIT/cubin loader.
  Not related to the WAN example — it's a flashinfer-main packaging issue.
- **`nvidia-cutlass-dsl` upgrade quirk**: a fresh container with `4.3.5`
  pre-installed silently leaves stale state when `pip install flashinfer`
  upgrades to `>= 4.5.0`, breaking `import cutlass`. Force-reinstall fixes
  it. `requirements.txt` excludes `4.5.2` specifically (the only release
  that consistently shipped without `cutlass.__init__.py`); the
  upgrade-from-4.3.5 cleanup issue is upstream.

## Reproducing

```bash
# Container
docker run -d --net=host --gpus all --runtime=nvidia --ipc=host \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --shm-size 20g --security-opt seccomp=unconfined \
  --mount type=bind,source=/path/to/flashinfer/parent/,target=/path/to/flashinfer/parent/ \
  --name fi-bench nvcr.io/nvidia/pytorch:26.03-py3 sleep infinity

docker exec -w /path/to/flashinfer fi-bench bash -c "
  pip install --no-build-isolation -e .
  pip install diffusers transformers accelerate ftfy einops sentencepiece
  pip install --force-reinstall 'nvidia-cutlass-dsl>=4.4.2,!=4.5.2'
"

# Single backend
docker exec -e HF_HOME=/path/to/hf_cache -w /path/to/flashinfer fi-bench \
  python examples/pytorch/wan/transformer_wan_flashinfer.py \
    --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --subfolder transformer \
    --batch-size 1 --num-frames 12 --height 64 --width 64 \
    --warmup-iters 2 --benchmark-iters 5 \
    --attention-backend single \
    --gemm-backend fp8_sm90
```

`--offline-act-quant` to switch off online activation scaling; see the
README in `examples/pytorch/` for the full flag list.

## Practical recommendations

| You're running on… | Best practical backend for WAN |
|--------------------|-------------------------------|
| H100 PCIe, 14B model | `fp8_sm90` (8% faster than torch baseline) |
| H100 PCIe, smaller model | `torch` (FP8 overhead not amortized) |
| **H100 80 GB HBM3, WAN-14B 720p × 5s** | **`bmm_fp8` (cuBLAS) + `--torch-compile --cuda-graph --offline-act-quant`** (1.265× vs cuBLAS bf16; `fp8_sm90` is a close second at 1.211× and only measurable under compile because the offline placeholder scale trips a kernel assert in eager) |
| B200, any WAN size tested | `torch` (cuBLAS bf16 baseline is hard to beat at these shapes) |
| **8× B200, VSA-finetuned WAN-14B, 720p × 5s** | **`fp4` + `--use-vsa --vsa-sparsity 0.9` + Ulysses** (1.23× vs bf16/dense, or 1.30× with the not-quality-preserving `--offline-act-quant`; VSA supplies ~18 of those points, nvfp4 the rest. Either Ulysses backend — the NVLink-P2P kernel is 1.5–2× faster below ~30 MB per collective but ties WAN's 97 MB payload) |
| **B300 SXM6, WAN-14B 720p × 5s** | **`fp4-cutlass` + `--torch-compile --cuda-graph --offline-act-quant`** (1.141× vs cuBLAS bf16; `bmm_fp8` is right behind at 1.127×, `mxfp8` at 1.094×) |

To make FlashInfer's quantized paths actually win on B200, the workload
needs to move further into the bandwidth-bound regime: more attention
context (longer videos / higher resolution), larger batch size, or multi-step
diffusion where the per-step quantization cost is amortized across many
forwards. For the configurations in this report the per-call kernel-launch
overhead plus online activation quantization eats more than the FP8 GEMM
saves.
