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
8× B200 with Ulysses context parallelism, adding a sparse attention path (VSA)
on top of the quantized GEMM backends.

### Setup

- **Model**: `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` — Wan2.1-14B
  finetuned *with* VSA, so the checkpoint carries the extra per-block
  `to_gate_compress` projection VSA's gated combine needs. Every row uses this
  same checkpoint; only the attention and GEMM paths change.
- **Config** (FastVideo's published inference settings for this checkpoint):
  720×1280, 81 frames, 50 steps, guidance 5.0, flow-shift 5.0, seed 1024,
  fps 16. **VSA sparsity 0.75** — see "Choosing the sparsity" below for why
  not the 0.9 the model card suggests.
- **Shapes**: latent 21×90×160 → post-patch grid **21×45×80 = 75,600 tokens**,
  sharded 9,450 tokens/rank across 8 ranks; 40 heads → 5 heads/rank.
- **Hardware**: 8× B200 180 GB on NVSwitch (`umbriel-b200-094` for the
  Ulysses-backend study, `umb-b200-237` for the final table).
- **Stack**: `nvcr.io/nvidia/pytorch:26.03-py3` (PyTorch 2.11, CUDA 13),
  flashinfer 0.6.17 (editable), diffusers 0.39, `quack` from source.
- **Timing**: 3 warmup denoising steps (absorbs JIT, autotune, and the offline
  activation-scale calibration), then 50 timed steps via
  `callback_on_step_end`; the first timed interval is dropped from the mean.
  Model load is excluded.
- **Attention baseline**: `--attention-backend torch` (SDPA) everywhere, so the
  VSA rows change *only* self-attention. Cross-attention stays SDPA in all rows.

### Results

Cumulative, at FastVideo's recommended **sparsity 0.9**. Read these as
performance numbers: 0.9 does not preserve video quality on this checkpoint —
see "Choosing the sparsity" below, and the 0.75 table after it.

**8× B200 (Ulysses, NCCL):**

| # | Config | GEMM | Self-attention | denoise (s) | ms/step | speedup | incremental |
|---|--------|------|----------------|------------:|--------:|--------:|------------:|
| 1 | baseline | torch bf16 (cuBLAS) | torch SDPA | 83.13 | 1662.1 | 1.00× | — |
| 2 | + VSA | torch bf16 | VSA (sparsity 0.9) | 64.72 | 1293.7 | 1.28× | +28.4% |
| 3 | + nvfp4 dynamic | `mm_fp4`, online act-quant | VSA (0.9) | 62.60 | 1251.1 | 1.33× | +3.4% |
| 4 | **+ nvfp4 static** | `mm_fp4`, calibrated offline | VSA (0.9) | **59.16** | **1182.4** | **1.41×** | +5.8% |

**Single B200:**

| # | Config | denoise (s) | ms/step | speedup | incremental |
|---|--------|------------:|--------:|--------:|------------:|
| 1 | baseline | 611.0 | 12220.4 | 1.00× | — |
| 2 | + VSA (sparsity 0.9) | 467.4 | 9346.9 | 1.31× | +30.7% |
| 3 | + nvfp4 dynamic | 422.7 | 8453.4 | 1.45× | +10.6% |
| 4 | **+ nvfp4 static** | **403.3** | **8065.0** | **1.52×** | +4.8% |

Ulysses scaling between the two tables: 12220 → 1662 ms/step for the baseline
(**7.35×** on 8 GPUs, 92% efficiency) and 8065 → 1182 for the fully-featured row
(**6.82×**, 85%). Sparse attention shards slightly less well because the
all-to-all payload is unchanged while the compute it overlaps with has shrunk.

**At the quality-preserving sparsity 0.75** (8× B200, same otherwise), VSA is
roughly break-even and the GEMM backends carry the win: 84.13 / 83.16 / 78.81 /
75.47 s → 1.00× / 1.01× / 1.07× / **1.11×**. Those four videos all render
cleanly; frame statistics (mean/std/inter-frame delta) are 55.3/72.9/6.25 for
the baseline then 46.2/74.1/15.28, 46.6/74.0/15.07, 46.5/72.6/15.50 — a
different but equally valid sample, not a degraded one.

Every row was captured with a generated video (`--output`), a timing JSON
(`--timing-json`) and an Nsight Systems trace of the timed denoising steps; see
"Reproducing" below for the exact invocations. On the run machine the
sparsity-0.9 artifacts live under
`/home/scratch.forrestl_wwfo/wan_final/{gpu1,gpu8}/`.

For a narrative version of this section, see [`BLOG.md`](BLOG.md).

### Choosing the sparsity: 0.9 does not work here

The model card asks for sparsity 0.9. At 0.9 the output is badly degraded —
blown-out colour, lost detail, global composition roughly intact. Quality
against sparsity, all else fixed:

| sparsity | topk / 1440 blocks | video | frame mean/std/Δ |
|---------:|-------------------:|-------|------------------|
| 0.0 (all blocks) | 1440 | clean | 66.9 / 72.5 / 9.29 |
| 0.5 | 720 | clean | 55.4 / 78.0 / 7.63 |
| **0.75** | **360** | **clean** | **46.2 / 74.1 / 15.28** |
| 0.9 | 144 | **degraded** | 38.6 / 77.4 / 16.07 |
| *(dense reference)* | — | clean | 55.3 / 72.9 / 6.25 |

**This is not an implementation bug — FastVideo's own stack does the same
thing.** Running their pipeline, their DiT and their kernel end to end on this
checkpoint at sparsity 0.9 for 50 steps produces the same failure mode
(oversaturated colour, smeared detail, composition intact):

| output | frame mean | std | inter-frame Δ |
|--------|-----------:|----:|--------------:|
| FastVideo, own pipeline, sparsity 0.9 | 39.1 | 77.5 | 17.72 |
| this example, sparsity 0.9 | 38.6 | 77.4 | 16.07 |
| dense baseline | 55.3 | 72.9 | 6.25 |

It is worth being explicit about what FastVideo does *not* do, since a natural
guess is that they keep some layers or some denoising steps dense:

- **No per-layer gating.** `wanvideo.py` picks the block class once for the
  whole stack (`WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN"
  else WanTransformerBlock`); all 40 blocks are sparse.
- **No per-step schedule.** `VideoSparseAttentionMetadata` carries
  `current_timestep`, but `_compute_cur_topk()` only reads the constant
  `VSA_sparsity`. The `VSA_decay_rate` / `VSA_decay_interval_steps` knobs that
  do ramp sparsity live in `TrainingArgs`, i.e. they are for finetuning.
- Their own inference default is `VSA_sparsity = 0.0`; 0.9 only appears when
  passed explicitly, as the model card's example does.

The port was additionally verified against FastVideo's `fastvideo_kernel` at
every level:

| Checked | Result |
|---|---|
| tile tables (`tile_partition_indices`, `variable_block_sizes`, `non_pad_index`) | bit-identical to FastVideo `vsa_utils` |
| pooled block scores, **real activations** | 0.28% relative |
| top-k selection, real activations | 143/144 set overlap; diagonal-block rate 61.5% vs 61.5% |
| fine-stage kernels under a **forced identical selection** | FlashInfer blk64 vs FastVideo `block_sparse_attn` within 0.03–0.2%; both mask padding |
| head splitting (the invariant Ulysses relies on), production grid | difference exactly 0 |
| RoPE tables and application | bit-identical to diffusers (`apply_rotary_emb` character-for-character) |
| `to_gate_compress` weights | raw checkpoint std 2.696e-4 == loaded |

The decisive check: **swapping FastVideo's own `video_sparse_attn` into this
pipeline, changing nothing else, produces an identically degraded video**
(frame mean 38.6, std 78.0, against 38.6/77.4 for ours). Two independent
kernels fed the same inputs fail the same way.

Also ruled out, each with a dedicated run: NVFP4 (row 2 of the older matrix —
fp4 with dense attention — renders perfectly); Ulysses (reproduces on a single
GPU, and 1-GPU vs 8-GPU VSA agree); partial edge blocks (the model's training
resolution 77×768×1280 gives 1200 blocks that are all exactly 64 tokens — still
degraded); CFG amplification (guidance 1.0 still degraded); and the multistep
solver (`FlowMatchEulerDiscrete` instead of `UniPCMultistep` still degraded).
`block_sizes` indexing — global block id vs position in the selected list, a
difference that would be invisible at sparsity 0.0 — was checked against an
eager reference and matches to 0.2%.

So the sparse *algorithm* at 0.9 is what this checkpoint cannot absorb, at
least through diffusers' `WanPipeline`. 0.75 is the highest sparsity that
still renders cleanly.

### Against FastVideo's own implementation

VSA's end-to-end win came in under expectation, so the obvious question is
whether this port is simply slower than the reference. It is not — it is faster
at both the kernel and the whole-model level.

**Attention op, per rank per layer** (75,600 tokens, 5 heads, head_dim 128, bf16):

| sparsity | ours | FastVideo | ours vs dense | FastVideo vs dense | **ours vs FastVideo** |
|---------:|-----:|----------:|--------------:|-------------------:|----------------------:|
| 0.90 | **5.90 ms** | 7.50 ms | 1.77× | 1.39× | **1.27×** |
| 0.75 | **8.93 ms** | 14.17 ms | 1.17× | 0.74× | **1.59×** |
| 0.50 | 14.88 ms | 25.24 ms | 0.70× | 0.41× | **1.70×** |

(dense torch SDPA = 10.46 ms.) Restricted to the fine stage under an
*identical* block selection, FlashInfer's `bsa_attn_blk64_fwd` beats FastVideo's
ThunderKittens/Triton `block_sparse_attn` by **1.75×–1.89×** across the same
sparsities.

**Whole DiT, end to end** (single B200, 720p × 5s, 20 steps, bf16 GEMM), which
also separates the attention op from the rest of the model:

| sparsity | ours | our DiT + FastVideo's op | FastVideo end to end | ours vs FastVideo |
|---------:|-----:|-------------------------:|---------------------:|------------------:|
| dense | 12.31 s/step | — | *cannot load* | — |
| 0.90 | **9.48 s/step** | 10.32 s/step | 11.97 s/step | **1.26×** |
| 0.75 | **11.67 s/step** | — | 15.64 s/step | **1.34×** |

Swapping only the op inside an identical DiT (9.48 → 10.32) attributes 1.09× to
the attention kernel; the remaining 1.16× (10.32 → 11.97) is the rest of the
model. So roughly a third of the end-to-end lead is VSA itself and two thirds is
the surrounding transformer.

Two things worth recording:

- **FastVideo cannot run this checkpoint densely.** Selecting a non-VSA
  attention backend builds a block without `to_gate_compress`, and loading dies
  with `ValueError: Parameter blocks.27.to_gate_compress.bias not found in
  custom model state dict`. Their own dense baseline is therefore unobtainable
  on this model, which is why the dense row above only has our number — and why
  only this example can quote "VSA versus dense" for this checkpoint at all.
- **At sparsity 0.75, FastVideo's VSA (15.64 s/step) is slower than our dense
  attention (12.31 s/step).** The sparse path only pays off there against a
  slow enough dense baseline.

### Why VSA only buys 1% at a usable sparsity

Isolated self-attention at exactly the per-rank shape (75,600 tokens, 5 heads,
head_dim 128, bf16), after the tiling optimization below:

| Self-attention, per rank per layer | time | vs dense |
|------------------------------------|-----:|---------:|
| torch SDPA (dense) | 10.42 ms | 1.00× |
| VSA sparsity 0.9 (topk 144) | 6.77 ms | **1.54×** |
| **VSA sparsity 0.75 (topk 360)** | **9.77 ms** | **1.07×** |
| VSA sparsity 0.5 (topk 720) | 15.27 ms | 0.68× |

**The quality-preserving range (≤0.75) and the speed-positive range (≥0.9)
barely overlap on this checkpoint at this shape.** VSA is a real 1.54× at 0.9
and a wash at 0.75. Two structural reasons, both visible in the numbers:

1. The `(4,4,4)` cube tiling pads 75,600 tokens to 92,160 (**+21.9%**) at this
   grid, so the sparse kernel starts 1.22× behind on token count alone.
2. Block-sparse attention does not convert density into time linearly: at
   sparsity 0.75 the kernel touches 25% of the blocks but runs only ~1.5×
   faster than dense.

**Tiling optimization (kept).** The cube permutation was originally
zero-allocate + gather + scatter, costing 2.02 ms per layer — 28% of VSA's
total, and ~7× off what the byte count justifies. Rewriting it as a single
gather (every padded slot reads some arbitrary token; padding is masked
downstream by `block_valid_mask` for the pooled mean and by
`variable_block_sizes` for the sparse kernel) brought it to **1.09 ms**, which
is what moved sparsity 0.9 from 1.42× to 1.54× and 0.75 from 1.02× to 1.07×.
`vsa_sanity.py` still passes with identical numbers, since its eager reference
does its own zero-padded tiling independently.

### NVFP4 static: calibrated, not a placeholder

Rows 3→4 show static beating dynamic by 4%, and unlike the earlier revision of
this document **the static row is now numerically usable**. The offline scale
used to be a fixed `448 * 6`, which assumes the activation amax is exactly 1.0;
measured WAN activations run 4–28, so it over-scaled by 4–28× and clipped —
producing visibly washed-out video (frame mean 124/255, std 21). It now
calibrates over the first `FLASHINFER_OFFLINE_CALIB_STEPS` (default 3) forwards
and freezes:

```
step 0: x_amax= 4.250  sf=632.5   step 2: x_amax=16.750  sf=160.5  <- frozen
step 1: x_amax= 8.750  sf=307.2   step 3+: reuse, no amax reduction
```

Calibration lands in warmup, so timed steps still skip the amax pass, and the
scale tensor keeps a stable address so CUDA-graph capture is unaffected. The
FP8 offline paths are still fixed placeholders — see the "Online vs. offline"
section.

### Ulysses backend: NCCL vs the NVLink-P2P kernel

Measured separately (on the 8× B200 NVSwitch node), with VSA at 0.9. The two
backends tie end-to-end (67.61 s vs 67.78 s) and are **numerically identical** —
same seed, latents matching to the last bit. That is *not* because there is
nothing to win: the NCCL path really does run a full-tensor permute before the
collective, worth **0.115 ms, 39% of NCCL's 0.300 ms scatter**, which the fused
kernel folds into its cross-GPU writes. It gives that back on raw transfer:

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

Small payloads (≤ ~30 MB) win 1.25–2.07× because collapsing two kernel launches
into one dominates. Large payloads tie then lose: NCCL's raw all-to-all reaches
529 GB/s at 96.8 MB and 590 GB/s at 193.5 MB (payload ÷ time), while the fused
kernel plateaus at ~320–330 GB/s. WAN 720p lands exactly on the crossover. The
optimization is real and the intuition behind it is right — it should pay off
for LLM-scale Ulysses (a few MB per collective, squarely in the 1.5–2× region),
and it would pay off here too if the fused kernel closed that ~1.7×
raw-bandwidth gap. That gap, not NVSwitch topology, is what to chase.

For scale: all-to-all is ~120 ms of a ~1500 ms step (8%), so even a 2× faster
collective buys ~4% end-to-end at this shape.

### Correctness

`vsa_sanity.py` covers the VSA path itself: tile/untile is an exact round-trip,
the cube permutation really groups each `(4,4,4)` neighbourhood, the kernel
matches an independent eager reference to ≤6e-3 relative error at sparsity
0/0.5/0.9, and splitting heads across ranks does not change the result.

For the sequence-parallel path, comparing 1-GPU against 8-GPU latents after one
denoising step:

| Path | cosine similarity, 1 GPU vs 8 GPU |
|------|----------------------------------:|
| dense SDPA | 0.99983 |
| VSA, sparsity 0.0 (all blocks kept) | 0.99979 |
| VSA, sparsity 0.9 | 0.917 |

The first two show Ulysses is correct. The third is **expected**: top-k block
selection is discrete, so bf16-level differences in the pooled scores flip which
blocks a query block attends to, and 40 layers × 2 CFG passes amplify that.
Dense attention cannot exhibit this because it is permutation-invariant once
RoPE is baked in. So **VSA output is not reproducible across world sizes** — the
samples are equally valid, not equal. Sparsity 0 restores reproducibility.

### Caveats

- **VSA needs a VSA-finetuned checkpoint.** The gated combine reads
  `to_gate_compress`, which stock Wan checkpoints do not have; running
  `--use-vsa` on one silently uses a randomly initialized gate.
- Row 1 runs *dense* attention on a VSA-finetuned checkpoint. That is the
  controlled comparison (same weights, only attention changes), not the same as
  the original Wan2.1-14B release.
- The `vsa_blackwell_blk64` kernel is SM100-only, bf16-only, and requires
  `head_dim == 128`.
- `flashinfer.cute_dsl.sparse`'s package `__init__` imports the blk128 backend
  unconditionally, so `quack` must be installed even to reach the pure-CUDA
  blk64 path used here.

### Reproducing

```bash
torchrun --nproc_per_node=8 examples/pytorch/wan/pipeline_wan_flashinfer.py \
  --model-id FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 --flow-shift 5.0 --seed 1024 \
  --gemm-backend fp4 --offline-act-quant --attention-backend torch \
  --use-vsa --vsa-sparsity 0.75 --ulysses-backend nccl \
  --warmup-steps 3 --timing-json timing.json --output wan_vsa.mp4
```

That is row 4. Drop `--offline-act-quant` for row 3, use `--gemm-backend torch`
for row 2, and drop `--use-vsa` as well for row 1. `--check-rank-sync` asserts
every rank holds identical latents at every step. `--scheduler flow-euler`
swaps the multistep solver for a first-order one.

To capture an Nsight Systems trace of just the denoising steps (the model load
would otherwise dominate the report):

```bash
nsys profile -o row4 --trace=cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  torchrun --nproc_per_node=8 examples/pytorch/wan/pipeline_wan_flashinfer.py \
    ... --num-inference-steps 3 --warmup-steps 3 --output-type latent \
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

**NVFP4 offline is now calibrated.** `--offline-act-quant` on the `fp4`
backend observes the real activation amax over the first
`FLASHINFER_OFFLINE_CALIB_STEPS` (default 3) forwards and then freezes the
scale, so the timed steps still skip the amax reduction but the scale is
representative. The previous fixed `448 * 6` is the scale you would get if the
activation amax were exactly 1.0; measured WAN activations run 4–28, so the
old placeholder over-scaled by 4–28× and clipped hard — that is why the
earlier `nvfp4 static` videos came out washed out (frame mean 124/255, std 21,
against 55/73 for the baseline). With calibration the static output matches
the dynamic one (46.5/72.6 vs 46.6/74.1). Put the calibration in warmup
(`--warmup-steps ≥ 2`) so it finishes before timing and before any CUDA-graph
capture; the scale tensor keeps a stable address so capture is unaffected.

**⚠️ The FP8 offline paths are still placeholders:** `--offline-act-quant`
uses a **fixed** scale (`1.0` for per-tensor, `1.0 / fp8_max` for blockwise)
for `fp8`, `bmm_fp8`, `fp8_groupwise`, `fp8_blockscaled` and
`batch_deepgemm_fp8`. Treat *those* offline numbers as **upper-bound speed
estimates**, not drop-in production settings; they would need the same
calibration treatment as `fp4`.

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
| **8× B200, VSA-finetuned WAN-14B, 720p × 5s** | **`fp4 --offline-act-quant` + `--use-vsa --vsa-sparsity 0.75` + Ulysses** (1.11× vs bf16/dense; nvfp4 supplies ~10 of those points and VSA only ~1, because the sparsity that preserves quality on this checkpoint is not the sparsity where VSA is fast. Do not use sparsity 0.9 — it is 1.54× on attention but visibly degrades the video. Either Ulysses backend — the NVLink-P2P kernel is 1.5–2× faster below ~30 MB per collective but ties WAN's 97 MB payload) |
| **B300 SXM6, WAN-14B 720p × 5s** | **`fp4-cutlass` + `--torch-compile --cuda-graph --offline-act-quant`** (1.141× vs cuBLAS bf16; `bmm_fp8` is right behind at 1.127×, `mxfp8` at 1.094×) |

To make FlashInfer's quantized paths actually win on B200, the workload
needs to move further into the bandwidth-bound regime: more attention
context (longer videos / higher resolution), larger batch size, or multi-step
diffusion where the per-step quantization cost is amortized across many
forwards. For the configurations in this report the per-call kernel-launch
overhead plus online activation quantization eats more than the FP8 GEMM
saves.
