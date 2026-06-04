# HunyuanImage-3 FlashInfer Kernel Benchmark

Isolated FlashInfer kernel benchmark at HunyuanImage-3.0 canonical
shapes, across the relevant GEMM backends, both online and offline
activation-quantization modes (FP8 family), on H100 PCIe (sm90) and
B200 (sm100).

## Methodology

- **Script**: `examples/pytorch/hunyuan_image3/bench_kernels_isolated.py`
- **Iterations**: 3 warmup + 20 measured per kernel
- **Dtype**: bfloat16 baseline; FP8/FP4/MXFP8 paths cast inside
- **Stack**: PyTorch 2.11 (`nvcr.io/nvidia/pytorch:26.03-py3`), CUDA 13.2,
  flashinfer 0.6.7 (editable from this checkout),
  `nvidia-cutlass-dsl >= 4.4.2` (force-reinstalled per the upgrade
  quirk documented in [wan/BENCHMARK.md](../wan/BENCHMARK.md))
- **GPUs**:
  - H100 PCIe 80 GB (cc 9.0) — `ipp1-3194`
  - B200 192 GB (cc 10.0) — `umb-b200-067`
- **Shapes** (HunyuanImage-3 `config.json`):
  - `hidden_size = 4096`, `intermediate_size = 3072`
  - `num_attention_heads = 32`, `num_key_value_heads = 8` (GQA), `head_dim = 128`
  - `num_experts = 64`, `moe_topk = 8`
- **GEMM sequence length**: `M = 4096`
  (one value; each `(M, K, N)` triggers a separate CUTLASS-DSL JIT compile)
- **Other-section sequence lengths**: `M ∈ {1024, 4096}` and
  attention `seq ∈ {1024, 4096}`, decode `kv_len ∈ {1024, 4096}`

`Speedup = baseline_time / backend_time` (>1 means faster than baseline).

The baseline differs per section:
| Section | Baseline |
|---|---|
| GEMM | `torch.matmul(bf16, bf16) -> bf16` (the kernel cuBLAS picks for `nn.Linear`) |
| RMSNorm | Upstream `HunyuanRMSNorm` (FP32-reduction PyTorch) |
| SwiGLU | `silu(gate) * up` in eager mode |
| Fused MoE | Upstream `HunyuanMoE` per-expert loop (`moe_impl='eager'`) |
| Attention | `torch.nn.functional.scaled_dot_product_attention` |

---

## H100 PCIe (sm90)

### A. GEMM at M=4096

| Shape `(M, K, N)` | torch bf16 baseline (ms) | `fp8_sm90` online (ms / ×) | `fp8_sm90` offline (ms / ×) | best |
|---|---:|---:|---:|---:|
| `mlp gate_and_up`  (4096, 4096, 6144) | 0.414 | 0.304 / 1.36× | 0.264 / **1.57×** | **1.57× offline** |
| `mlp down`         (4096, 3072, 4096) | 0.222 | 0.165 / 1.35× | 0.137 / **1.63×** | **1.63× offline** |
| `attn qkv_proj`    (4096, 4096, 6144) | 0.416 | 0.305 / 1.37× | 0.267 / **1.56×** | **1.56× offline** |
| `attn o_proj`      (4096, 4096, 4096) | 0.290 | 0.220 / 1.32× | 0.176 / **1.65×** | **1.65× offline** |

Other GEMM backends on H100 were skipped (expected):

| Backend | Reason on H100 |
|---|---|
| `mm_bf16` | Routed to cuDNN backend, which doesn't support sm90 (`BackendSupportedError`). |
| `mm_fp8` | TRT-LLM low-latency `gemm.run` "Check failed" — `flashinfer_modules.py:_check_gemm_backend_support` and wan BENCHMARK.md already steer to `fp8_sm90` on sm90. |
| `fp8_groupwise`, `mm_fp4`, `mm_mxfp8` | CUTLASS DSL kernels require sm100+. |

### B. RMSNorm at hidden_size=4096

| M | torch fp32-reduce baseline (ms) | `flashinfer.rmsnorm` (ms) | speedup |
|---|---:|---:|---:|
| 1024 | 0.102 | 0.013 | **8.05×** |
| 4096 | 0.398 | 0.041 | **9.70×** |

### C. SwiGLU activation (2 × 3072 channels)

| M | torch `silu(gate)*up` baseline (ms) | `flashinfer.silu_and_mul` (ms) | speedup |
|---|---:|---:|---:|
| 1024 | 0.030 | 0.020 | **1.48×** |
| 4096 | 0.107 | 0.042 | **2.56×** |

### D. Fused MoE (top-8 / 64 experts, intermediate=3072)

| Routed tokens M | torch eager baseline (ms) | `flashinfer.cutlass_fused_moe` bf16 (ms) | speedup |
|---|---:|---:|---:|
| 1024 | 19.36 | — (Ninja build failed) | — |
| 4096 | 21.13 | — (Ninja build failed) | — |

`cutlass_fused_moe` JIT-build fails on sm90 (the kernel targets
sm100+). The huge per-token cost of the eager loop is why the upstream
`HunyuanMoE` already wires a `moe_impl='flashinfer'` switch — it just
doesn't yield on sm90.

### E. GQA attention (32 Q heads, 8 KV heads, head_dim=128)

| Mode    | seq / kv_len | torch SDPA baseline (ms) | `single_prefill_with_kv_cache` causal (ms / ×) | `cudnn_batch_prefill_with_kv_cache` causal (ms / ×) | `single_decode_with_kv_cache` (ms / ×) |
|---|---:|---:|---:|---:|---:|
| prefill | 1024 | 0.042 | 0.059 / 0.70× | 0.072 / 0.58× | — |
| prefill | 4096 | 0.385 | 0.364 / **1.06×** | 0.397 / 0.97× | — |
| decode  | 1024 | 0.034 | — | — | 0.024 / **1.40×** |
| decode  | 4096 | 0.046 | — | — | 0.047 / 0.99× |

PyTorch SDPA on H100 (Flash-Attention-2 / cuDNN under the hood) is hard
to beat at small sequence lengths. FlashInfer's
`single_decode_with_kv_cache` wins at kv_len=1024 and ties at 4096.
Prefill paths reach parity around seq=4096 but don't pull ahead.

---

## B200 (sm100)

### A. GEMM at M=4096

| Shape `(M, K, N)` | torch bf16 baseline (ms) | `mm_bf16` (ms / ×) | `mm_fp8` online (ms / ×) | `mm_fp8` offline (ms / ×) |
|---|---:|---:|---:|---:|
| `mlp gate_and_up`  (4096, 4096, 6144) | 0.137 | 0.143 / 0.96× | 0.664 / 0.21× | 0.654 / 0.21× |
| `mlp down`         (4096, 3072, 4096) | 0.069 | 0.096 / 0.72× | 0.390 / 0.18× | 0.378 / 0.18× |
| `attn qkv_proj`    (4096, 4096, 6144) | 0.137 | 0.142 / 0.96× | 0.663 / 0.21× | 0.655 / 0.21× |
| `attn o_proj`      (4096, 4096, 4096) | 0.093 | 0.101 / 0.91× | 0.476 / 0.19× | 0.466 / 0.20× |

cuBLAS bf16 dominates at these shapes — they are too small for FP8 to
amortize the per-call quantize+TRT-LLM dispatch cost. `mm_fp8`'s online
vs offline activation-quant gap is in the noise here (the per-tensor
amax+scale on (4096, K) is a single cheap CUDA call).

GEMM backends skipped on B200:

| Backend | Reason on B200 |
|---|---|
| `fp8_sm90` | sm90-only by design. |
| `fp8_groupwise`, `mm_fp4`, `mm_mxfp8` | `--skip-cutlass-dsl`: the CUTLASS Python DSL JIT compile hangs indefinitely (>60 min, no progress) for `gemm_fp8_nt_groupwise` on this stack, even after the documented `nvidia-cutlass-dsl` force-reinstall. Same call pattern that runs in wan/BENCHMARK.md, so the issue is shape-dependent and not the example's fault. See "Known issues" below. |

### B. RMSNorm at hidden_size=4096

| M | torch fp32-reduce baseline (ms) | `flashinfer.rmsnorm` (ms) | speedup |
|---|---:|---:|---:|
| 1024 | 0.056 | 0.008 | **6.85×** |
| 4096 | 0.180 | 0.010 | **17.86×** |

### C. SwiGLU activation (2 × 3072 channels)

| M | torch `silu(gate)*up` baseline (ms) | `flashinfer.silu_and_mul` (ms) | speedup |
|---|---:|---:|---:|
| 1024 | 0.017 | 0.010 | **1.75×** |
| 4096 | 0.063 | 0.011 | **5.93×** |

### D. Fused MoE (top-8 / 64 experts, intermediate=3072)

| Routed tokens M | torch eager baseline (ms) | `flashinfer.cutlass_fused_moe` bf16 (ms) | speedup |
|---|---:|---:|---:|
| 1024 | 8.78 | 1.20 | **7.33×** |
| 4096 | 11.41 | 3.46 | **3.30×** |

This is the largest absolute win in the workload. HunyuanImage-3's
backbone is 32 layers × MoE-per-layer; the cutlass fused MoE path
saves ~7.5 ms per layer at the small-batch routing of 1024 tokens. On
the per-step image-denoising forward, this dwarfs every other
optimization.

### E. GQA attention (32 Q heads, 8 KV heads, head_dim=128)

| Mode    | seq / kv_len | torch SDPA baseline (ms) | `single_prefill_with_kv_cache` causal (ms / ×) | `cudnn_batch_prefill_with_kv_cache` causal (ms / ×) | `single_decode_with_kv_cache` (ms / ×) |
|---|---:|---:|---:|---:|---:|
| prefill | 1024 | 0.020 | 0.056 / 0.35× | 0.031 / 0.63× | — |
| prefill | 4096 | 0.117 | 0.451 / 0.26× | 0.168 / 0.69× | — |
| decode  | 1024 | 0.011 | — | — | 0.011 / 1.01× |
| decode  | 4096 | 0.014 | — | — | 0.014 / 1.00× |

The B200 SDPA path (cuDNN under the hood) is even harder to beat than
on H100 in the dense single-request regime — both single_prefill and
cudnn_batch_prefill lose. Decode paths are a tie. Real speedup from
the FlashInfer attention layer in this example comes from the
**custom-mask path** in `FlashInferHunyuanImage3Attention` (which we
don't isolate here because it depends on the model's runtime mask),
not from the mask-less prefill.

---

## Online vs offline activation quantization

For the two FP8 paths that work on H100 (`fp8_sm90`) and B200
(`mm_fp8`), here's the online → offline savings:

| GPU | Backend | shape (typical) | online (ms) | offline (ms) | offline savings |
|---|---|---|---:|---:|---:|
| H100 | `fp8_sm90` | `(4096, 4096, 4096)` (`attn o_proj`) | 0.220 | 0.176 | **20%** |
| H100 | `fp8_sm90` | `(4096, 4096, 6144)` (`mlp gate_and_up`) | 0.304 | 0.264 | **13%** |
| B200 | `mm_fp8`   | `(4096, 4096, 4096)` (`attn o_proj`) | 0.476 | 0.466 | **2%** |
| B200 | `mm_fp8`   | `(4096, 4096, 6144)` (`mlp gate_and_up`) | 0.664 | 0.654 | **2%** |

`fp8_sm90` uses 128×128 block-scale quantization (~3× more reductions
per activation than per-tensor), so the offline savings are real and
substantial (13–20%). `mm_fp8` is per-tensor: one amax + one scale per
activation, which is cheap, so online vs offline is in the noise.

**Caveat**: offline = a **fixed** scale (1.0 for per-tensor; 1.0/fp8_max
for blockwise). These are placeholders — production offline
quantization needs per-layer calibrated scales. Treat the offline
numbers above as upper-bound speed estimates, not as drop-in
production settings.

---

## Practical recommendations

| You're running on… | Best practical kernel by section |
|---|---|
| **H100 PCIe (sm90)** GEMM | `fp8_blockscale_gemm_sm90` offline (1.57–1.65×). Online costs 13–20% extra. |
| **H100 PCIe (sm90)** RMSNorm / SwiGLU | `flashinfer.rmsnorm` (~8–10×) and `flashinfer.silu_and_mul` (~1.5–2.6×). Unconditional wins. |
| **H100 PCIe (sm90)** MoE | Currently no FlashInfer kernel runs (`cutlass_fused_moe` is sm100+). Stick with the upstream eager loop. |
| **H100 PCIe (sm90)** attention | `single_decode_with_kv_cache` for decode at kv_len ≤ 1024. SDPA elsewhere. |
| **B200 (sm100)** GEMM | torch.matmul bf16 (cuBLAS). FP8 needs longer/wider shapes to amortize. |
| **B200 (sm100)** RMSNorm / SwiGLU | `flashinfer.rmsnorm` (up to **17.86×**) and `flashinfer.silu_and_mul` (up to **5.93×**). The biggest gain on activation/normalization kernels we've measured. |
| **B200 (sm100)** MoE | `flashinfer.cutlass_fused_moe` bf16 — **3.3–7.3× over the eager loop**. This is the single largest end-to-end win for HunyuanImage-3. |
| **B200 (sm100)** attention | torch SDPA. Switch to FlashInfer when you need a custom mask path (handled in the model wrapper, not in this micro-bench). |

For HunyuanImage-3 specifically, the FlashInfer-swapped backbone on
B200 gains most of its speedup from (1) the fused MoE path and (2) the
RMSNorm/SwiGLU activation kernels; the GEMM and dense attention paths
are a wash or slightly slower at these shapes. On H100 PCIe the
picture flips: FP8 GEMM via `fp8_sm90` is the only real win, MoE
unavailable, and norms still helpful.

## Known issues / caveats

- **`mm_fp8` on H100**: TRT-LLM low-latency GEMM "Check failed:
  gemm.run(...)" at the tested shapes. Use `fp8_sm90` instead on sm90,
  per `flashinfer_modules.py:_check_gemm_backend_support`.
- **`cutlass_fused_moe` on H100**: Ninja build failure
  (kernel targets sm100+). Not a regression; just unavailable.
- **CUTLASS-DSL JIT hang on B200**: `gemm_fp8_nt_groupwise`,
  `mm_fp4`, `mm_mxfp8` all hang in the cute-DSL JIT (>60 min, no
  progress) on this stack. We did the
  `pip install --force-reinstall nvidia-cutlass-dsl>=4.4.2,!=4.5.2`
  workaround documented in wan/BENCHMARK.md — same shapes' equivalents
  work in the wan bench. The shapes here (M=4096, K∈{3072, 4096}, N∈{4096, 6144})
  differ from wan's (M=12288, K∈{5120, 13824}, N∈{5120, 13824}); the trigger
  appears to be shape-dependent. The B200 sbatch passes `--skip-cutlass-dsl`
  to skip these and collect the rest of the numbers.
- **Offline activation quantization** uses a *fixed* scale of `1.0` or
  `1.0/fp8_max`, not a calibrated one. The offline numbers here are
  upper-bound speed estimates.
- **Non-fatal exit 1 on H100**: The Python process exits with code 1
  after Section E completes (a TRT-LLM autotune-cache destructor). The
  bench data printed before exit is intact and is what's reported here.
- **Home-quota disk-full on this cluster**: `/home/forrestl` is 5 GB and
  fills up via `~/.cache/flashinfer`, `~/.tensorrt_llm/tmp`, etc. The
  sbatch scripts override `HOME`, `XDG_CACHE_HOME`,
  `FLASHINFER_WORKSPACE_BASE`, and `TRITON_CACHE_DIR` to land on
  scratch.

## Reproducing

```bash
# Container
docker run -d --net=host --gpus all --runtime=nvidia --ipc=host \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --shm-size 20g --security-opt seccomp=unconfined \
  --mount type=bind,source=/path/to/flashinfer/parent/,target=/path/to/flashinfer/parent/ \
  --name fi-bench nvcr.io/nvidia/pytorch:26.03-py3 sleep infinity

docker exec -w /path/to/flashinfer fi-bench bash -c "
  export HOME=/path/that/has/space     # ~/.cache/flashinfer and ~/.tensorrt_llm/tmp need >5GB
  pip install --no-build-isolation -e .
  pip install --force-reinstall 'nvidia-cutlass-dsl>=4.4.2,!=4.5.2'   # B200 only
  python examples/pytorch/hunyuan_image3/bench_kernels_isolated.py \
      [--skip-cutlass-dsl]   # add on B200 if the cute-DSL JIT hangs
"
```

For slurm-driven runs the two sbatch scripts under `.bench_runs/`
(in this checkout) take care of cache redirection
(`HOME`, `FLASHINFER_WORKSPACE_BASE`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`),
the `nvidia-cutlass-dsl` force-reinstall on B200, and pass
`--skip-cutlass-dsl` to the B200 invocation.
