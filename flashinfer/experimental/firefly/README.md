# Firefly: NVFP4 routed MoE for SM100 / SM103 (experimental)

This backend is experimental and may change or be removed without
compatibility guarantees. Listing `FireflyNvfp4Config()` among the
`MoEConfig` backend candidates (`flashinfer.fused_moe`) is the explicit
opt-in and emits FlashInfer's experimental backend warning once. Firefly is
never part of the default candidate list: a `MoELayer` that does not list it
never selects it. Once listed, it takes part in `MoELayer`'s selection like
any listed backend: with other eligible candidates whose weight views are
prepared, the layer times each one (CUDA graph replays) and keeps the fastest
per input shape, which can be Firefly; the autotuner has a single tactic to
choose for it. Tracking issue: flashinfer-ai/flashinfer#9994.

Firefly runs the routed experts of a MoE layer on datacenter Blackwell
(tested on B200/SM100 and GB300/SM103) from per-token or per-tensor NVFP4
activations and NVFP4 weights, with CuTe DSL kernels that choose their schedule from the
batch size (from 1 token to `tune_max_num_tokens`).

## Contract

| | |
| --- | --- |
| Weights x activations | NVFP4 x NVFP4, per-token (`QuantConfig(per_token_scale=True)`) or per-tensor (`per_token_scale=False` or `None`) activation scales |
| Activation pack | `uint8 [M, H/2]` E2M1 values (even element in the low nibble), linear `float8_e4m3fn [M, H/16]` block scales; per-token: `float32 [M]` per-token scales, x ~ e2m1 * block scale * per-token scale; per-tensor: a unit global scale, x ~ e2m1 * block scale (a calibrated activation scale is folded into the view's per-expert `w1_gs`, as other backends fold it into `w1_alpha`) |
| Routing | pre-routed (`PackedPrecomputed` or `UnpackedPrecomputed`): `int32 [M, top_k]` ids, `float32` / `bfloat16` / `float16 [M, top_k]` weights used as given; ids outside this rank's experts (negative ones included) are skipped; each token's ids must name distinct experts, as a top-k selection does (not checked, which would take a device-to-host sync per call: repeated ids cause no out-of-bounds access, but their result is unspecified) |
| Activations | `SwiGLU(alpha, beta, limit)`, `SwiGLUStep`, `GeGLU`, `GeGLUTanh`, `SiTU` (no `clamp_limit`), `ReLU2` |
| Finalize | `do_finalize=True`; `use_fused_finalize=False` sums the top_k rows in a fixed FP32 order (deterministic), `True` adds them into the output from the down GEMM (BF16 atomics / bulk reductions, not deterministic; one BF16 rounding per add, so the error grows with top_k: relative RMSE about 0.004 at top_k 8 and 0.007 at 32, as FlashInfer's `cute_dsl` fused finalize, against about 0.0025 unfused) |
| Expert parallelism | `local_expert_offset` / `local_num_experts` (the output is this rank's partial sum) |
| PDL | `ExecutionConfig(enable_pdl=)` (default on) |
| Shapes | H % 256 == 0, I % 128 == 0, 1 <= top_k <= min(E, 32), E <= 1024 |
| Output | BF16 `[M, H]` |
| Not supported | NVFP4 4over6, fused shared experts, `do_finalize=False`, in-kernel routing (`FromLogits`) |

The intermediate (the activation's output, rounded to BF16) is quantized to
NVFP4 with E4M3 block scales only (the second-level scale is 1), except on the
per-token direct path: M <= 4 with a gated activation, I = 512 or 768 and
H >= 512 (512 <= I <= 768 and I % 256 == 0), which uses a per-token scale as
well. Every other shape and batch size (e.g. M = 2 at H = 1024, I = 640) uses
block scales only.

The weight view is Firefly's own (gate/up rows interleaved per 8, block
scales swizzled 128x4, one FP32 global scale per expert from its amax): build
it with `FireflyNvfp4Config.prepare_weights` from the canonical BF16 experts
and register it under `"firefly_nvfp4"`.

## Usage

```python
from flashinfer.fused_moe import (
    BackendOptions, ExecutionConfig, ExpertConfig, FireflyNvfp4Config,
    MoEActivationPack, MoEConfig, MoELayer, MoEWeightPack, QuantConfig,
    QuantFormat, RoutingConfig, SwiGLU,
)

config = MoEConfig(
    routing=RoutingConfig(num_experts=E, top_k=K),
    quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4,
                      per_token_scale=True),
    experts=ExpertConfig(intermediate_size=I),
    activation=SwiGLU(),
    execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
)
weights = MoEWeightPack()
weights.prepare_for("firefly_nvfp4", FireflyNvfp4Config.prepare_weights(
    w1_bf16, w2_bf16, num_local_experts=E, hidden_size=H,
    intermediate_size=I, activation=SwiGLU()))
layer = MoELayer(config)
out = layer(MoEActivationPack(hidden_states_q=q, hidden_states_scale=sf,
                              topk_ids=ids, topk_weights=w,
                              per_token_scale=gs), weights)
```

Per-tensor activations: `per_token_scale=False` in the `QuantConfig` and no
`per_token_scale` in the pack; the kernels then load no row scales (except the
gate/up of small batches without a scatter, which reads a constant vector of
ones: it is faster that way). The output equals a per-token call with every
per-token scale 1.

One runner serves every batch size up to `tune_max_num_tokens` with one
workspace and every layer of the same shape (the weights come with each
call). The first call of a batch size compiles its kernels. Before capturing
a CUDA graph, call (or let `MoELayer` prepare) each batch size with each
routing-weight dtype the capture uses (FP32, BF16 or FP16 weights take
separate kernels); for M <= 4 (the direct GEMMs) use the same weight tensors as
the capture, since their TMA descriptors are built per weight tensor on first
use, which a capture cannot do (a missing one raises `RuntimeError`). The
layer's first call (which allocates and zeroes its workspace) must also come
before any capture, or it raises `RuntimeError`.

## Package layout

| Module | Content |
| --- | --- |
| `moe.py` | `CuteMoE`: the per-batch-size schedule (`select_schedule`), plans, workspace and launch sequence |
| `router.py` | `SelectExperts`: reads the routing, counts and ranks the routes per expert |
| `placement.py` | `ScanCounts` (long batches), L2 prefetch |
| `scatter.py` | `Scatter`, `ScatterChunks`: row placement and copies |
| `gemm.py` | `GroupedGemm`: the persistent, warp-specialized grouped GEMM (gate/up with the activation and NVFP4 quantization in the epilogue; down, optionally with the fused finalize) |
| `direct.py` | `DirectGemm`: the M <= 4 path |
| `quantize.py` | `RouteInput` (M <= 4) |
| `reduce.py` | `ReduceRoutes`: the unfused finalize |
| `activation.py` | the activations, accepting FlashInfer's `ActivationConfig` objects |
| `weights.py` | weight quantization and the weight view |
| `HANDWRITTEN_PTX.md` | where the kernels use inline PTX and why |

`gemm.py` sets `.maxnreg` on the grouped GEMM through CuTe DSL's private
`CuTeDSL._generate_kernel_attrs` (checked on CuTe DSL 4.8.0 and 4.8.1);
without that method, or if CuTe DSL stops calling it, the grouped GEMM is
launched with the public launch bounds instead (`gemm.launch_regs`).
