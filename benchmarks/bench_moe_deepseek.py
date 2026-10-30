#!/usr/bin/env python3
"""MoE Performance Benchmark - CuteDSL vs TRTLLM.

Compares NVFP4, MXFP8, MXFP4x8, and BF16 MoE backends:
- CuteDSL W4A4: NVFP4 activations and weights
- CuteDSL PER-TOKEN: per-token NVFP4 activations with the explicit 4over6 recipe
- CuteDSL W4A16: BF16 activations with NVFP4 weights decoded online
- CuteDSL MXFP8 / MXFP4x8 (opt-in): MXFP8 activations with MXFP8 / MXFP4 weights,
  using the generic split kernels and deterministic unfused combine
- CUTLASS (explicit --backends cutlass only): NVIDIA CUTLASS implementation
- TRTLLM NVFP4: TensorRT-LLM's NVFP4 implementation
- TRTLLM PER-TOKEN: the same activation recipe with native TRTLLM MoE
- TRTLLM MXFP8 / MXFP4x8 (opt-in): native block-scaled MoE baselines
- TRTLLM BF16: unquantized BF16 activations and weights

Usage:
    # Kimi K3 routed latent experts, EP8, SiTU(4, 25); unsupported arms print N/A
    python bench_moe_deepseek.py --model kimi-k3 --no-fused-finalize

    # Nemotron-3 Ultra routed latent experts, EP8, ReLU2
    python bench_moe_deepseek.py --model nemo3-ultra --no-fused-finalize

    # V4.1-Flash EP4: shared sqrtsoftplus routes are prepared outside timing
    python bench_moe_deepseek.py --model v41-flash --ep 4 --no-fused-finalize

    # Separate MXFP8 / MXFP4x8 table with matched TRTLLM baselines
    python bench_moe_deepseek.py --quant-family mxfp --ep 8 --swap-ab --no-fused-finalize

    # Throughput benchmark (large batches: 128-4096 tokens)
    python bench_moe_deepseek.py

    # Generation phase benchmark (small batches: 1-128 tokens)
    python bench_moe_deepseek.py --gen-phase

    # With Expert Parallelism simulation
    python bench_moe_deepseek.py --ep 1    # 256 local experts (no parallelism)
    python bench_moe_deepseek.py --ep 8    # 32 local experts (8-way EP)
    python bench_moe_deepseek.py --ep 16   # 16 local experts (16-way EP)

    # With Tensor Parallelism simulation
    python bench_moe_deepseek.py --tp 8    # 256-wide local expert intermediate

    # Custom token counts
    python bench_moe_deepseek.py --num-tokens 64,128,256

    # Include activation quantization for FP4-activation backends
    python bench_moe_deepseek.py --include-activation-quant

    # Compare CuTe DSL W4A4/W4A16 against pure BF16 TRTLLM MoE
    python bench_moe_deepseek.py --backends cutedsl,trtllm-bf16

    # Compare the two standalone per-token activation backends
    python bench_moe_deepseek.py --backends cute-dsl-per-token,trtllm-per-token

    # Compare TRTLLM NVFP4 and BF16
    python bench_moe_deepseek.py --backends trtllm-nvfp4,trtllm-bf16

    # Disable CUDA graph (useful for debugging or profiling)
    python bench_moe_deepseek.py --no-cuda-graph

    # Disable CUPTI (use CUDA events for timing instead)
    python bench_moe_deepseek.py --no-cupti

    # Capture one backend for an external CUDA profiler
    python bench_moe_deepseek.py --num-tokens 128 \
        --profile-cuda --profile-backend cute-dsl

    # Nsight Systems: trace individual graph kernels after tuning/warmup.
    # The moe NVTX range excludes the separately labeled cold-L2 flush.
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
        --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o moe_bf16 python bench_moe_deepseek.py --num-tokens 32 --ep 8 \
        --profile-cuda --profile-backend trtllm-bf16 --profile-iters 20

    # CuTe DSL / CUTLASS finalize modes (TRTLLM keeps its native finalize)
    python bench_moe_deepseek.py --functional-api  # fused allowed (default)
    python bench_moe_deepseek.py --functional-api --no-fused-finalize  # deterministic

PER-TOKEN arms use MSE 4over6 activation quantization with E4M3 max=256 and
4over6 error fast math enabled, while base FP4 quantization fast math is disabled.
The same explicit recipe applies to initial activations and FC2 inputs; weights
retain standard NVFP4 quantization. Other arms are unaffected.

Metrics:
    - ms: Latency in milliseconds
    - TFLOPS: Computational throughput
    - Speedup: named TRTLLM baseline latency / comparison-arm latency
"""

import argparse
import contextlib
import gc
import os
from dataclasses import dataclass, replace
import numpy as np
import torch

from flashinfer.tllm_enums import DEFAULT_SWIGLU_LIMIT


@dataclass
class DeepSeekConfig:
    hidden_size: int = 7168
    intermediate_size: int = 2048
    num_experts: int = 256
    n_group: int = 8
    topk_group: int = 4
    top_k: int = 8
    routed_scaling_factor: float = 2.5
    name: str = "DeepSeek-V3"
    routing: str = "deepseek_v3"
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT
    default_ep: int = 8
    activation: str = "swiglu"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None

    @property
    def gemm1_rows(self):
        if self.activation in ("swiglu", "situ"):
            return 2 * self.intermediate_size
        elif self.activation == "relu2":
            return self.intermediate_size
        else:
            raise ValueError(f"Unsupported model activation: {self.activation}")


MODEL_CONFIGS = {
    "v3": DeepSeekConfig(),
    # https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json
    # The official inference/model.py Gate and Expert define routing/clamping.
    "v41-flash": DeepSeekConfig(
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=384,
        n_group=1,
        topk_group=1,
        top_k=6,
        routed_scaling_factor=1.5,
        name="DeepSeek-V4.1-Flash",
        routing="sqrtsoftplus",
        swiglu_limit=10.0,
        default_ep=4,
    ),
    # Same routed-latent-expert contract as bench_cute_dsl_moe_distributed.py.
    "kimi-k3": DeepSeekConfig(
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=896,
        n_group=1,
        topk_group=1,
        top_k=16,
        routed_scaling_factor=1.0,
        name="Kimi-K3",
        routing="sigmoid",
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        default_ep=8,
    ),
    # https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/77df655d5e9f8362164ed14dd8b48f8bce657498/config.json
    # Routed experts operate in the 2048-wide latent space, not model width 8192.
    "nemo3-ultra": DeepSeekConfig(
        hidden_size=2048,
        intermediate_size=5120,
        num_experts=512,
        n_group=1,
        topk_group=1,
        top_k=22,
        routed_scaling_factor=5.0,
        name="Nemotron-3-Ultra",
        routing="sigmoid",
        activation="relu2",
        default_ep=8,
    ),
}
CFG = replace(MODEL_CONFIGS["v3"])
BASE_INTERMEDIATE_SIZE = CFG.intermediate_size
TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096]

# Generation phase token counts (small batches typical in decode)
GEN_PHASE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

# Expert Parallelism configurations
# EP=1: all 256 experts on single GPU
# EP=8: 32 experts per GPU (256/8)
# EP=16: 16 experts per GPU (256/16)
EP_CONFIGS = {
    1: {"num_local_experts": 256, "local_expert_offset": 0},
    2: {"num_local_experts": 128, "local_expert_offset": 0},
    4: {"num_local_experts": 64, "local_expert_offset": 0},
    8: {"num_local_experts": 32, "local_expert_offset": 0},
    16: {"num_local_experts": 16, "local_expert_offset": 0},
}


def _autotune_context(do_autotune, cache, measurement_policy=None):
    """Use the same managed cache and measurement policy for tuning and replay."""
    from flashinfer.autotune_cache import autotune_v2

    return autotune_v2(
        mode="tune" if do_autotune else "replay",
        persistent_cache=cache is not None,
        cache_root=cache,
        cache_scope="context",
        measurement_policy=measurement_policy,
    )


def is_sm100_family():
    """Check for SM100 family (Blackwell: SM100, SM103).

    CuteDSL MoE NVFP4 kernels are optimized for SM10x architecture.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) in ((10, 0), (10, 3))


def calc_tflops(n, ms, num_local_experts=None):
    """Calculate TFLOPS for MoE computation.

    With EP, only tokens routed to local experts are computed.
    Assumes uniform routing distribution across experts.
    """
    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    # Fraction of work done locally (assuming uniform distribution)
    local_fraction = num_local_experts / CFG.num_experts

    flops = (
        n
        * CFG.top_k
        * local_fraction  # Only local expert pairs are computed
        * (
            2 * CFG.hidden_size * CFG.gemm1_rows
            + 2 * CFG.intermediate_size * CFG.hidden_size
        )
    )
    return flops / (ms * 1e-3) / 1e12


def create_inputs(n, dev="cuda", routing_bias_scale=0.01):
    """Create inputs for all backends (CuteDSL, CUTLASS, TRTLLM)."""
    from flashinfer.fp4_quantization import fp4_quantize

    torch.manual_seed(42)
    sv = 16
    FP8M = torch.finfo(torch.float8_e4m3fn).max
    FP4M = 6.0

    # Router logits and bias
    rl = torch.randn(n, CFG.num_experts, device=dev, dtype=torch.float32)
    rb = (
        torch.randn(
            CFG.num_experts,
            device=dev,
            dtype=torch.float32 if CFG.routing == "sqrtsoftplus" else torch.bfloat16,
        )
        * routing_bias_scale
    )

    # Hidden states
    hb = torch.randn(n, CFG.hidden_size, device=dev, dtype=torch.bfloat16) / 10
    hg = FP8M * FP4M / hb.abs().max().float()

    # Weights (BF16)
    w1b = (
        torch.randn(
            CFG.num_experts,
            CFG.gemm1_rows,
            CFG.hidden_size,
            device=dev,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2b = (
        torch.randn(
            CFG.num_experts,
            CFG.hidden_size,
            CFG.intermediate_size,
            device=dev,
            dtype=torch.bfloat16,
        )
        / 10
    )

    # Compute per-expert global scales
    w1_gs_list, w2_gs_list = [], []
    for e in range(CFG.num_experts):
        w1_gs_list.append(FP8M * FP4M / w1b[e].abs().max().float())
        w2_gs_list.append(FP8M * FP4M / w2b[e].abs().max().float())
    w1_gs = torch.tensor(w1_gs_list, device=dev)
    w2_gs = torch.tensor(w2_gs_list, device=dev)

    # CUTLASS format: quantize with swizzled scale factors
    w1_fp4_list, w1_sf_list = [], []
    w2_fp4_list, w2_sf_list = [], []
    for e in range(CFG.num_experts):
        q1, s1 = fp4_quantize(w1b[e], w1_gs[e], sv, False, True)  # swizzled
        w1_fp4_list.append(q1)
        w1_sf_list.append(s1)
        q2, s2 = fp4_quantize(w2b[e], w2_gs[e], sv, False, True)  # swizzled
        w2_fp4_list.append(q2)
        w2_sf_list.append(s2)

    return {
        "router_logits": rl,
        "routing_bias": rb,
        "hidden_bf16": hb,
        "hidden_gs": hg,
        "w1_bf16": w1b,
        "w1_gs": w1_gs,
        "w2_bf16": w2b,
        "w2_gs": w2_gs,
        # CUTLASS specific
        "w1_fp4": torch.stack(w1_fp4_list),
        "w1_sf": torch.stack(w1_sf_list),
        "w2_fp4": torch.stack(w2_fp4_list),
        "w2_sf": torch.stack(w2_sf_list),
    }


# =============================================================================
# Benchmark Functions
# =============================================================================


def _measure_or_profile(
    run,
    input_kwargs,
    warmup,
    iters,
    use_cuda_graph,
    use_cupti,
    profile_cuda,
    profile_iters,
):
    from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size

    da_capture = use_cuda_graph and os.getenv(
        "FLASHINFER_DIST_AWARE_AUTOTUNE", "0"
    ).lower() not in ("", "0", "false", "no", "off", "none")

    if not profile_cuda and da_capture:
        from flashinfer.fused_moe import trtllm_moe_acquire_da_graph_leases

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(**input_kwargs)
        leases = trtllm_moe_acquire_da_graph_leases(graph)
        l2_flush = torch.empty(2 * get_l2_cache_size(), device="cuda", dtype=torch.int8)
        times = []
        try:
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                l2_flush.zero_()
                torch.cuda.synchronize()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
        finally:
            torch.cuda.synchronize()
            graph.reset()
            for lease in leases:
                lease.release()
        return np.median(times)

    if not profile_cuda:
        times = bench_gpu_time(
            run,
            dry_run_iters=warmup,
            repeat_iters=iters,
            cold_l2_cache=True,
            enable_cupti=use_cupti,
            use_cuda_graph=use_cuda_graph,
            input_kwargs=input_kwargs,
        )
        return np.median(times)

    def runner():
        return run(**input_kwargs)

    leases = ()
    if use_cuda_graph:
        from flashinfer.fused_moe import trtllm_moe_acquire_da_graph_leases

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(**input_kwargs)
        leases = trtllm_moe_acquire_da_graph_leases(graph)
        runner = graph.replay
    for _ in range(3):
        runner()
    torch.cuda.synchronize()

    l2_flush = torch.empty(2 * get_l2_cache_size(), device="cuda", dtype=torch.int8)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(profile_iters):
        with torch.cuda.nvtx.range("l2_flush"):
            l2_flush.zero_()
            torch.cuda.synchronize()
        with torch.cuda.nvtx.range("moe"):
            runner()
            torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    if use_cuda_graph:
        torch.cuda.synchronize()
        graph.reset()
        for lease in leases:
            lease.release()
    return float("nan")


def _routing_tensors(inputs):
    """Allocate V3 outputs or share the selected model's precomputed routes."""
    logits = inputs["router_logits"]
    if CFG.routing == "deepseek_v3":
        shape = (logits.shape[0], CFG.top_k)
        return (
            torch.empty(shape, dtype=torch.float32, device=logits.device),
            torch.empty(shape, dtype=torch.int32, device=logits.device),
        )
    elif CFG.routing in ("sqrtsoftplus", "sigmoid"):
        if "_model_routes" not in inputs:
            # Match the distributed benchmark's model-specific routing on CPU,
            # outside tuning/timing/graphs, and share it across all backend arms.
            if CFG.routing == "sqrtsoftplus":
                scores = torch.nn.functional.softplus(logits.float().cpu()).sqrt()
            elif CFG.routing == "sigmoid":
                scores = logits.float().cpu().sigmoid()
            else:
                raise ValueError(f"Unsupported precomputed routing: {CFG.routing}")
            bias = inputs["routing_bias"].float().cpu()
            ids = torch.argsort(scores + bias, dim=-1, descending=True, stable=True)[
                :, : CFG.top_k
            ]
            # Match the distributed benchmark's lower-ID tie policy.
            weights = scores.gather(1, ids)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            weights = weights * CFG.routed_scaling_factor
            inputs["_model_routes"] = (
                weights.to(device=logits.device, dtype=torch.float32),
                ids.to(device=logits.device, dtype=torch.int32),
            )
        return inputs["_model_routes"]
    else:
        raise ValueError(f"Unsupported model routing: {CFG.routing}")


def _route_tokens(router_logits, routing_bias, topk_values, topk_indices):
    if CFG.routing == "deepseek_v3":
        from flashinfer.fused_moe import fused_topk_deepseek

        fused_topk_deepseek(
            scores=router_logits,
            bias=routing_bias,
            n_group=CFG.n_group,
            topk_group=CFG.topk_group,
            topk=CFG.top_k,
            routed_scaling_factor=CFG.routed_scaling_factor,
            topk_values=topk_values,
            topk_indices=topk_indices,
        )
    elif CFG.routing in ("sqrtsoftplus", "sigmoid"):
        # _routing_tensors supplied shared immutable routes before timing.
        return
    else:
        raise ValueError(f"Unsupported model routing: {CFG.routing}")


def _model_activation():
    from flashinfer.fused_moe import ReLU2, SiTU, SwiGLU

    if CFG.activation == "swiglu":
        return SwiGLU(limit=CFG.swiglu_limit)
    elif CFG.activation == "situ":
        if CFG.situ_beta is None or CFG.situ_linear_beta is None:
            raise ValueError("The SiTU preset requires gate and linear scales")
        return SiTU(gate_scale=CFG.situ_beta, linear_scale=CFG.situ_linear_beta)
    elif CFG.activation == "relu2":
        return ReLU2()
    else:
        raise ValueError(f"Unsupported model activation: {CFG.activation}")


def _backend_skip_reason(backend, swap_ab=False, use_fused_finalize=True):
    """Return an unsupported contract reason, never hide execution failures."""
    if backend not in (
        "CuteDSL W4A4",
        "CuteDSL PER-TOKEN",
        "CuteDSL W4A16",
        "CuteDSL MXFP8",
        "CuteDSL MXFP4x8",
        "CUTLASS",
        "TRTLLM NVFP4",
        "TRTLLM PER-TOKEN",
        "TRTLLM BF16",
        "TRTLLM MXFP8",
        "TRTLLM MXFP4x8",
    ):
        raise ValueError(f"Unknown benchmark backend: {backend}")
    if backend in ("CuteDSL MXFP8", "CuteDSL MXFP4x8") and not swap_ab:
        if CFG.activation == "situ":
            return f"{backend} requires --swap-ab for SiTU"
        if not use_fused_finalize:
            return f"{backend} requires --swap-ab for unfused finalize"
    if CFG.activation == "swiglu":
        return None
    elif CFG.activation == "situ":
        if backend in ("TRTLLM BF16", "TRTLLM MXFP8"):
            return f"{backend} has no SiTU kernel"
        elif backend == "TRTLLM PER-TOKEN":
            return "TRTLLM PER-TOKEN has no per-token SiTU kernel"
        elif backend == "CUTLASS":
            return "The CUTLASS benchmark arm does not implement SiTU"
        return None
    elif CFG.activation == "relu2":
        if backend == "CUTLASS":
            return "The CUTLASS benchmark arm does not implement ReLU2"
        return None
    else:
        raise ValueError(f"Unsupported model activation: {CFG.activation}")


@contextlib.contextmanager
def _per_token_quantization(enabled):
    """Disable base FP4 fast math only for per-token activation quantization.

    Weight preparation stays outside this scope. The separate 4over6 error
    calculation uses its explicitly configured fast-math setting.
    """
    if not enabled:
        yield
        return
    key = "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def bench_cute_dsl(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    use_wrapper=False,
    do_autotune=True,
    quant_mode="w4a4",
    use_per_token_activation=False,
    include_activation_quant=False,
    use_fused_finalize=True,
    swap_ab=False,
    profile_cuda=False,
    profile_iters=10,
    autotune_cache=None,
):
    """Benchmark CuteDSL MoE.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API (recommended for CUDA graph).
                    If False, use cute_dsl_fused_moe functional API.
        do_autotune: If True, run the pre-warm pass with tuning enabled so the
                    autotuner profiles all buckets and populates its cache. The
                    measurement loop runs with tuning disabled so that
                    choose_one cache lookups don't appear inside the CUDA-event
                    interval when bench_gpu_time falls back to events (i.e. when
                    both CUDA graphs and CUPTI are disabled).
        quant_mode: ``"w4a4"``, ``"w4a16"``, ``"mxfp8"``, or ``"w4a8"``.
            MX modes support deterministic combine with ``swap_ab=True``.
        swap_ab: Opt in to Blackwell activations-on-B kernels and weight layout.
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.
        include_activation_quant: Include the initial activation FP4
            quantization in the measured CuTe DSL W4A4 path. W4A16 consumes
            BF16 activations directly, so this option does not affect it.
        profile_cuda: Capture steady-state CUDA graph replays between
            cudaProfilerStart/Stop instead of benchmarking.
        profile_iters: Number of cold-L2 graph replays to capture.
    """
    from flashinfer.fused_moe import CuteDslConfig, QuantConfig, QuantFormat
    from flashinfer.fp4_quantization import fp4_quantize

    if use_per_token_activation and quant_mode != "w4a4":
        raise ValueError("Per-token NVFP4 requires quant_mode='w4a4'")
    activation_recipe = None
    if num_local_experts is None:
        num_local_experts = CFG.num_experts
    if quant_mode not in ("w4a4", "w4a16", "mxfp8", "w4a8"):
        raise ValueError(f"Unsupported CuTe DSL quant mode: {quant_mode}")
    if quant_mode in ("mxfp8", "w4a8") and (
        use_per_token_activation or include_activation_quant
    ):
        raise ValueError("CuTe DSL MX modes require prequantized activations")

    n, dev = inputs["router_logits"].shape[0], inputs["hidden_bf16"].device
    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts
    # Slice weights to LOCAL experts only
    w1_local = inputs["w1_bf16"][expert_start:expert_end]
    w2_local = inputs["w2_bf16"][expert_start:expert_end]
    tv, ti = _routing_tensors(inputs)
    # Alpha sized for LOCAL experts only
    alpha = torch.ones(num_local_experts, dtype=torch.float32, device=dev)
    # Pre-convert routing bias to float32
    routing_bias_f32 = inputs["routing_bias"].float()

    if quant_mode in ("w4a4", "w4a16"):
        from flashinfer import SfLayout, nvfp4_quantize
        from flashinfer.quantization.nvfp4_quantization_utils import (
            NVFP44Over6Config,
            make_nvfp4_global_scale,
        )

        activation_recipe = (
            NVFP44Over6Config(e4m3_max=256, err_mode="MSE", err_use_fast_math=True)
            if use_per_token_activation
            else None
        )
        sv = 16
        gs1 = torch.tensor([1.0], device=dev)

        activation_global_scale = None
        if quant_mode == "w4a4":
            activation_global_scale = (
                make_nvfp4_global_scale(
                    inputs["hidden_bf16"],
                    per_token_activation=True,
                    nvfp4_4over6_config=activation_recipe,
                )
                if use_per_token_activation
                else gs1
            )

        if quant_mode == "w4a16" or include_activation_quant:
            xf = inputs["hidden_bf16"]
            xs = None
            hidden_per_token_scale = None
        elif use_per_token_activation:
            with _per_token_quantization(True):
                xf, xs, hidden_per_token_scale = nvfp4_quantize(
                    inputs["hidden_bf16"],
                    activation_global_scale,
                    sfLayout=SfLayout.layout_linear,
                    per_token_activation=True,
                    backend="cute-dsl",
                    nvfp4_4over6=activation_recipe,
                )
        else:
            xf, xs = fp4_quantize(
                inputs["hidden_bf16"], gs1, sv, False, False, nvfp4_4over6=None
            )
            hidden_per_token_scale = None
        if xs is not None:
            xs = xs.unsqueeze(-1)

        def prepare_activation(x, x_sf):
            per_token_scale = hidden_per_token_scale
            if quant_mode == "w4a16":
                return x, None, None
            if include_activation_quant:
                if use_per_token_activation:
                    x, x_sf, per_token_scale = nvfp4_quantize(
                        x,
                        activation_global_scale,
                        sfLayout=SfLayout.layout_linear,
                        per_token_activation=True,
                        backend="cute-dsl",
                        nvfp4_4over6=activation_recipe,
                    )
                else:
                    x, x_sf = fp4_quantize(
                        x, gs1, sv, False, False, backend="cute-dsl", nvfp4_4over6=None
                    )
                x_sf = x_sf.unsqueeze(-1)
            return x, x_sf, per_token_scale

        weights = CuteDslConfig.prepare_weights(
            w1_local,
            w2_local,
            quant=QuantConfig(
                weight=QuantFormat.NVFP4,
                activation=(
                    QuantFormat.BF16 if quant_mode == "w4a16" else QuantFormat.NVFP4
                ),
            ),
            num_local_experts=num_local_experts,
            hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size,
            activation=_model_activation(),
            device=dev,
            nvfp4_4over6=None,
            swap_ab=swap_ab,
        )
        w1q, w1s = weights["w1_weight"], weights["w1_weight_sf"]
        w2q, w2s = weights["w2_weight"], weights["w2_weight_sf"]

        if quant_mode == "w4a16":
            fc2sc = None
        elif use_per_token_activation:
            # Both activation quantizers use the same inverse 256 * 6 scale.
            fc2sc = activation_global_scale
        else:
            fc2sc = torch.tensor([1.0], device=dev)

    elif quant_mode in ("mxfp8", "w4a8"):
        from flashinfer import mxfp8_quantize

        xf, xs = mxfp8_quantize(
            inputs["hidden_bf16"],
            is_sf_swizzled_layout=False,
            alignment=128,
            backend="cuda",
        )
        xs = xs.view(torch.uint8).reshape(n, CFG.hidden_size // 32)
        if quant_mode == "mxfp8":
            weight_format = QuantFormat.MXFP8
        elif quant_mode == "w4a8":
            weight_format = QuantFormat.MXFP4
        else:
            raise ValueError(quant_mode)
        weights = CuteDslConfig.prepare_weights(
            w1_local,
            w2_local,
            quant=QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
            num_local_experts=num_local_experts,
            hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size,
            activation=_model_activation(),
            device=dev,
            swap_ab=swap_ab,
        )
        w1q, w1s = weights["w1_weight"], weights["w1_weight_sf"]
        w2q, w2s = weights["w2_weight"], weights["w2_weight_sf"]

        fc2sc = None

        def prepare_activation(x, x_sf):
            return x, x_sf, None

    else:
        raise ValueError(f"Unsupported CuTe DSL quant mode: {quant_mode}")

    if CFG.activation == "swiglu":
        activation_kwargs = {"swiglu_limit": CFG.swiglu_limit}
    elif CFG.activation == "situ":
        activation = _model_activation()
        # CuTe DSL encodes SiTU as SwiGLU with explicit SiTU scales.
        activation_kwargs = {
            "situ_beta": activation.gate_scale,
            "situ_linear_beta": activation.linear_scale,
        }
    elif CFG.activation == "relu2":
        activation_kwargs = {"activation_type": _model_activation().type.value}
    else:
        raise ValueError(f"Unsupported model activation: {CFG.activation}")

    if use_wrapper:
        # Use CuteDslMoEWrapper (recommended for CUDA graph)
        from flashinfer import CuteDslMoEWrapper

        moe = CuteDslMoEWrapper(
            num_experts=CFG.num_experts,
            top_k=CFG.top_k,
            hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=n,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=use_fused_finalize,
            quant_mode=quant_mode,
            swap_ab=swap_ab,
            **activation_kwargs,
        )

        def run_moe(x, x_sf, topk_values, topk_indices, per_token_scale):
            return moe.run(
                x=x,
                x_sf=x_sf,
                token_selected_experts=topk_indices,
                token_final_scales=topk_values,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                per_token_scale=per_token_scale,
                nvfp4_4over6=activation_recipe,
            )
    else:
        # Use functional API
        from flashinfer import cute_dsl_fused_moe

        def run_moe(x, x_sf, topk_values, topk_indices, per_token_scale):
            return cute_dsl_fused_moe(
                x=x,
                x_sf=x_sf,
                token_selected_experts=topk_indices,
                token_final_scales=topk_values,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                num_experts=CFG.num_experts,
                top_k=CFG.top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                quant_mode=quant_mode,
                swap_ab=swap_ab,
                per_token_scale=per_token_scale,
                nvfp4_4over6=activation_recipe,
                use_fused_finalize=use_fused_finalize,
                **activation_kwargs,
            )

    def run(x, x_sf, router_logits, routing_bias, topk_values, topk_indices):
        x, x_sf, per_token_scale = prepare_activation(x, x_sf)
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
        return run_moe(x, x_sf, topk_values, topk_indices, per_token_scale)

    # Pass input tensors via input_kwargs for cold L2 cache rotation
    input_kwargs = {
        "x": xf,
        "x_sf": xs,
        "router_logits": inputs["router_logits"],
        "routing_bias": routing_bias_f32,
        "topk_values": tv,
        "topk_indices": ti,
    }

    # Pre-warm: run once on the default stream so that autotuning (which
    # allocates tensors and profiles multiple tactics) finishes before
    # bench_gpu_time moves execution to a side stream for CUDA-graph
    # capture.  Autotuning on a non-default stream triggers illegal-
    # memory-access errors in the CuteDSL persistent-tile-scheduler
    # kernels.  Autotune is scoped to the pre-warm only — bench_gpu_time
    # runs outside the autotune context so that choose_one in the
    # measurement loop returns the cached tactic via the non-tuning fast
    # path (no host-side tactic-walking inside the CUDA-event interval).
    measurement_policy = None
    if use_cuda_graph and quant_mode in ("w4a8", "mxfp8"):
        from flashinfer.autotune_cache import MeasurementPolicy

        measurement_policy = MeasurementPolicy(execution_mode="cuda_graph")
    with _per_token_quantization(use_per_token_activation):
        with _autotune_context(do_autotune, autotune_cache, measurement_policy):
            run(**input_kwargs)
            torch.cuda.synchronize()

        # Capture-time lookup must use the same policy as tuning.
        with _autotune_context(False, autotune_cache, measurement_policy):
            return _measure_or_profile(
                run,
                input_kwargs,
                warmup,
                iters,
                use_cuda_graph,
                use_cupti,
                profile_cuda,
                profile_iters,
            )


def bench_cutlass(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    do_autotune=True,
    include_activation_quant=False,
    use_fused_finalize=True,
    profile_cuda=False,
    profile_iters=10,
    autotune_cache=None,
):
    """Benchmark CUTLASS MoE.

    Args:
        do_autotune: See ``bench_cute_dsl`` for the autotune-scope rationale.
        use_fused_finalize: Allow fused-finalize tactics when True; exclude
            them when False. Autotuning may still choose an unfused tactic.
    """
    from flashinfer.fused_moe import SwiGLU, cutlass_fused_moe
    from flashinfer.fused_moe.runners import _cutlass_activation_params
    from flashinfer.fp4_quantization import fp4_quantize

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"

    tv, ti = _routing_tensors(inputs)
    activation_params = _cutlass_activation_params(
        SwiGLU(limit=CFG.swiglu_limit), num_local_experts, torch.device(dev)
    )

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    # Slice weights to LOCAL experts only (for fair EP comparison)
    w1_fp4_local = inputs["w1_fp4"][expert_start:expert_end]
    w1_sf_local = inputs["w1_sf"][expert_start:expert_end]
    w1_gs_local = inputs["w1_gs"][expert_start:expert_end]
    w2_fp4_local = inputs["w2_fp4"][expert_start:expert_end]
    w2_sf_local = inputs["w2_sf"][expert_start:expert_end]
    w2_gs_local = inputs["w2_gs"][expert_start:expert_end]

    # Prepare CUTLASS inputs
    a1_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)

    quant_scales = [
        a1_gs,
        w1_sf_local.view(torch.int32),
        1.0 / (a1_gs * w1_gs_local),
        a2_gs,
        w2_sf_local.view(torch.int32),
        1.0 / (a2_gs * w2_gs_local),
    ]

    hidden_fp4, input_sf = fp4_quantize(inputs["hidden_bf16"], a1_gs, sv, False, True)
    output = torch.empty(n, CFG.hidden_size, dtype=torch.bfloat16, device=dev)

    # Pre-convert routing bias to float32
    routing_bias_f32 = inputs["routing_bias"].float()

    # Pre-compute values that need conversion
    w1_fp4_view = w1_fp4_local.contiguous().view(torch.long)
    w2_fp4_view = w2_fp4_local.contiguous().view(torch.long)

    # Compute EP size from config
    ep_size = CFG.num_experts // num_local_experts

    def run(hidden, sf, router_logits, routing_bias, topk_values, topk_indices):
        if include_activation_quant:
            hidden, sf = fp4_quantize(hidden, a1_gs, sv, False, True)
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
        cutlass_fused_moe(
            hidden,
            topk_indices.to(torch.int),
            topk_values,
            w1_fp4_view,
            w2_fp4_view,
            torch.bfloat16,
            quant_scales=quant_scales,
            input_sf=sf,
            output=output,
            ep_size=ep_size,
            ep_rank=0,  # Simulating rank 0 of EP
            use_fused_finalize=use_fused_finalize,
            swiglu_alpha=activation_params["swiglu_alpha"],
            swiglu_beta=activation_params["swiglu_beta"],
            swiglu_limit=activation_params["swiglu_limit"],
        )
        return output

    input_kwargs = {
        "hidden": inputs["hidden_bf16"] if include_activation_quant else hidden_fp4,
        "sf": None if include_activation_quant else input_sf,
        "router_logits": inputs["router_logits"],
        "routing_bias": routing_bias_f32,
        "topk_values": tv,
        "topk_indices": ti,
    }

    # Pre-warm under autotune; measurement runs outside (see bench_cute_dsl).
    with _autotune_context(do_autotune, autotune_cache):
        run(**input_kwargs)
        torch.cuda.synchronize()

    with _autotune_context(False, autotune_cache):
        return _measure_or_profile(
            run,
            input_kwargs,
            warmup,
            iters,
            use_cuda_graph,
            use_cupti,
            profile_cuda,
            profile_iters,
        )


def bench_trtllm(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    do_autotune=True,
    use_per_token_activation=False,
    include_activation_quant=False,
    profile_cuda=False,
    profile_iters=10,
    autotune_cache=None,
    routing_input_mode="logits",
    distributions=("uniform",),
    precision="nvfp4",
):
    """Benchmark TRTLLM quantized or pure BF16 MoE with native finalize.

    Args:
        do_autotune: See ``bench_cute_dsl`` for the autotune-scope rationale.
        precision: ``"nvfp4"``, ``"mxfp8"``, ``"mxfp4"`` (MXFP4 weights
            with MXFP8 activations), or ``"bf16"``. Weight preparation is
            outside timing. NVFP4 per-token/4over6 flags do not affect the
            MX modes or BF16.
    """
    from flashinfer.fused_moe import RoutingMethodType
    from flashinfer.fused_moe.prepare import _activation_param_view
    from flashinfer.fused_moe.da_tuner import (
        RoutingRealizationFactory,
        RoutingRealizationKey,
    )

    if precision not in ("nvfp4", "mxfp8", "mxfp4", "bf16"):
        raise ValueError(f"Unsupported TRTLLM precision: {precision}")
    backend = {
        "nvfp4": "TRTLLM PER-TOKEN" if use_per_token_activation else "TRTLLM NVFP4",
        "mxfp8": "TRTLLM MXFP8",
        "mxfp4": "TRTLLM MXFP4x8",
        "bf16": "TRTLLM BF16",
    }[precision]
    skip_reason = _backend_skip_reason(backend)
    if skip_reason is not None:
        raise ValueError(skip_reason)
    if use_per_token_activation and precision != "nvfp4":
        raise ValueError("Per-token NVFP4 requires precision='nvfp4'")
    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, dev = inputs["router_logits"].shape[0], inputs["router_logits"].device
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts
    input_kwargs = {
        "routing_logits": inputs["router_logits"],
        "routing_bias": inputs["routing_bias"],
        "hidden_states": inputs["hidden_bf16"],
    }
    moe_kwargs = dict(
        num_experts=CFG.num_experts,
        top_k=CFG.top_k,
        n_group=CFG.n_group,
        topk_group=CFG.topk_group,
        intermediate_size=CFG.intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=num_local_experts,
        do_finalize=True,
    )

    if precision == "bf16":
        from flashinfer.fused_moe import (
            TrtllmBf16Config,
            trtllm_bf16_moe,
            trtllm_bf16_routed_moe,
        )

        # Canonical [up, gate] weights need both the gated row reorder and
        # BlockMajorK shuffle. The public helper preserves their BF16 values.
        weights = TrtllmBf16Config.prepare_weights(
            inputs["w1_bf16"][expert_start:expert_end],
            inputs["w2_bf16"][expert_start:expert_end],
            num_local_experts=num_local_experts,
            hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size,
            activation=_model_activation(),
            device=dev,
        )
        moe_kwargs.update(
            gemm1_weights=weights["gemm1_weights"],
            gemm2_weights=weights["gemm2_weights"],
        )
        logits_moe, routed_moe = trtllm_bf16_moe, trtllm_bf16_routed_moe
    elif precision in ("mxfp8", "mxfp4"):
        from flashinfer import SfLayout
        from flashinfer.fused_moe import (
            QuantConfig,
            QuantFormat,
            TrtllmFp4Config,
            TrtllmFp8BlockConfig,
            trtllm_fp4_block_scale_moe,
            trtllm_fp4_block_scale_routed_moe,
            trtllm_fp8_block_scale_moe,
            trtllm_fp8_block_scale_routed_moe,
        )
        from flashinfer.tllm_enums import Fp8QuantizationType, WeightLayout

        if precision == "mxfp8":
            config = TrtllmFp8BlockConfig
            weight_format = QuantFormat.MXFP8
            logits_moe = trtllm_fp8_block_scale_moe
            routed_moe = trtllm_fp8_block_scale_routed_moe
        elif precision == "mxfp4":
            config = TrtllmFp4Config
            weight_format = QuantFormat.MXFP4
            logits_moe = trtllm_fp4_block_scale_moe
            routed_moe = trtllm_fp4_block_scale_routed_moe
        else:
            raise ValueError(f"Unsupported MX precision: {precision}")
        quant = QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8)
        # Public helpers handle gated row order and each native scale layout.
        weights = config.prepare_weights(
            inputs["w1_bf16"][expert_start:expert_end],
            inputs["w2_bf16"][expert_start:expert_end],
            quant=quant,
            num_local_experts=num_local_experts,
            hidden_size=CFG.hidden_size,
            intermediate_size=CFG.intermediate_size,
            activation=_model_activation(),
            device=dev,
        )
        moe_kwargs.update(
            gemm1_weights=weights["gemm1_weights"],
            gemm1_weights_scale=weights["gemm1_weights_scale"],
            gemm2_weights=weights["gemm2_weights"],
            gemm2_weights_scale=weights["gemm2_weights_scale"],
            tune_max_num_tokens=max(n, 8192),
        )
        if precision == "mxfp8":
            moe_kwargs.update(
                fp8_quantization_type=Fp8QuantizationType.MxFp8,
                use_shuffled_weight=True,
                weight_layout=WeightLayout.MajorK,
            )
        elif precision == "mxfp4":
            moe_kwargs.update(
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_bias=None,
                output1_scale_scalar=weights["output1_scale_scalar"],
                output1_scale_gate_scalar=weights["output1_scale_gate_scalar"],
                output2_scale_scalar=weights["output2_scale_scalar"],
                hidden_states_scale_layout=SfLayout.layout_linear,
            )
        else:
            raise ValueError(f"Unsupported MX precision: {precision}")

        def quantize_hidden(hidden_states):
            return config.prepare_activations(hidden_states, quant=quant)

        if not include_activation_quant:
            hidden_states, hidden_states_scale = quantize_hidden(inputs["hidden_bf16"])
            input_kwargs.update(
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
            )
    elif precision == "nvfp4":
        from flashinfer import SfLayout, nvfp4_quantize
        from flashinfer.fused_moe import (
            trtllm_fp4_block_scale_moe,
            trtllm_fp4_block_scale_routed_moe,
        )
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )
        from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
        from flashinfer.quantization.nvfp4_quantization_utils import (
            NVFP44Over6Config,
            make_nvfp4_global_scale,
        )

        activation_recipe = (
            NVFP44Over6Config(e4m3_max=256, err_mode="MSE", err_use_fast_math=True)
            if use_per_token_activation
            else None
        )
        sv, etm, cache = 16, 128, {}
        hidden_global_scale = inputs["hidden_gs"]
        if use_per_token_activation:
            hidden_global_scale = make_nvfp4_global_scale(
                inputs["hidden_bf16"],
                per_token_activation=True,
                nvfp4_4over6_config=activation_recipe,
            )

        def quantize_hidden(hidden_states):
            if use_per_token_activation:
                hidden_states, hidden_states_scale, per_token_scale = nvfp4_quantize(
                    hidden_states,
                    hidden_global_scale,
                    sfLayout=SfLayout.layout_linear,
                    per_token_activation=True,
                    backend="cute-dsl",
                    nvfp4_4over6=activation_recipe,
                )
            else:
                hidden_states, hidden_states_scale = fp4_quantize(
                    hidden_states,
                    hidden_global_scale,
                    sv,
                    False,
                    True,
                    nvfp4_4over6=None,
                )
                per_token_scale = None
            hidden_states = hidden_states.view(torch.uint8).reshape(
                n, CFG.hidden_size // 2
            )
            hidden_states_scale = (
                hidden_states_scale.view(torch.float8_e4m3fn)
                .flatten()[: n * CFG.hidden_size // sv]
                .reshape(n, CFG.hidden_size // sv)
            )
            return hidden_states, hidden_states_scale, per_token_scale

        with _per_token_quantization(use_per_token_activation):
            hfp, hsc, hidden_per_token_scale = quantize_hidden(inputs["hidden_bf16"])

        def prep(bf16, gs, M, K):
            """Prepare weights for LOCAL experts only."""
            fl, sl = [], []
            for e in range(expert_start, expert_end):
                q, s = fp4_quantize(bf16[e], gs[e], sv, False, False, nvfp4_4over6=None)
                fl.append(q.view(torch.uint8).reshape(M, K // 2))
                sl.append(s.view(torch.float8_e4m3fn).reshape(M, K // sv))
            return torch.stack(fl), torch.stack(sl)

        w1f, w1s = prep(
            inputs["w1_bf16"],
            inputs["w1_gs"],
            CFG.gemm1_rows,
            CFG.hidden_size,
        )
        w2f, w2s = prep(
            inputs["w2_bf16"], inputs["w2_gs"], CFG.hidden_size, CFG.intermediate_size
        )

        def shuf(fp4, sf, perm_fn):
            """Shuffle weights for LOCAL experts only."""
            fsh, ssh = [], []
            for i in range(num_local_experts):
                p = perm_fn(cache, fp4[i], etm)
                fsh.append(fp4[i][p.to(dev)].contiguous())
                ps = perm_fn(cache, sf[i].view(torch.uint8), etm, sv)
                ssh.append(
                    block_scale_interleave(
                        sf[i].view(torch.uint8)[ps.to(dev)].contiguous()
                    )
                )
            return torch.stack(fsh), torch.stack(ssh)

        w1_permute = (
            _maybe_get_cached_w3_w1_permute_indices
            if _model_activation().is_gated
            else get_w2_permute_indices_with_cache
        )
        w1f, w1s = shuf(w1f, w1s, w1_permute)
        w2f, w2s = shuf(w2f, w2s, get_w2_permute_indices_with_cache)
        w1s = w1s.view(torch.float8_e4m3fn).reshape(
            num_local_experts, CFG.gemm1_rows, CFG.hidden_size // sv
        )
        w2s = w2s.view(torch.float8_e4m3fn).reshape(
            num_local_experts, CFG.hidden_size, CFG.intermediate_size // sv
        )

        # Scale tensors sized for LOCAL experts only
        sc = torch.ones(num_local_experts, device=dev, dtype=torch.float32)
        moe_kwargs.update(
            gemm1_weights=w1f,
            gemm1_weights_scale=w1s,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2f,
            gemm2_weights_scale=w2s,
            gemm2_bias=None,
            output1_scale_scalar=sc,
            output1_scale_gate_scalar=sc,
            output2_scale_scalar=sc,
            nvfp4_4over6=activation_recipe,
        )
        input_kwargs.update(
            hidden_states=inputs["hidden_bf16"] if include_activation_quant else hfp,
            hidden_states_scale=None if include_activation_quant else hsc,
        )
        logits_moe = trtllm_fp4_block_scale_moe
        routed_moe = trtllm_fp4_block_scale_routed_moe
    else:
        raise ValueError(f"Unsupported TRTLLM precision: {precision}")

    moe_kwargs.update(
        _activation_param_view(
            _model_activation(), num_local_experts, torch.device(dev)
        )
    )
    if CFG.activation in ("situ", "relu2"):
        moe_kwargs["activation_type"] = _model_activation().type.value
    routing_ids = None
    routing_weights = None
    is_routed = routing_input_mode == "routed" or CFG.routing in (
        "sqrtsoftplus",
        "sigmoid",
    )
    if CFG.routing in ("sqrtsoftplus", "sigmoid"):
        routing_weights, routing_ids = _routing_tensors(inputs)
        moe_kwargs.update(n_group=None, topk_group=None)
    elif is_routed:
        if precision == "mxfp8":
            # The FP8 Renormalize path consumes precomputed, ungrouped routes.
            moe_kwargs.update(n_group=None, topk_group=None)
        # Share this row's cached routes across precisions, including after
        # autotuning advances the RNG. Release the cache with the input row.
        factory = inputs.get("_trtllm_routing_factory")
        if factory is None:
            factory = RoutingRealizationFactory()
            inputs["_trtllm_routing_factory"] = factory
        realized = factory.get_or_create(
            RoutingRealizationKey(
                device=torch.device(dev),
                num_tokens=n,
                distribution=distributions[0],
                sample_index=0,
                local_expert_offset=0,
                num_local_experts=CFG.num_experts,
                top_k=CFG.top_k,
                routing_rule_fingerprint="bench_moe_deepseek:renormalize",
                routed_scaling_factor=1.0,
            )
        )
        routing_ids = realized.expert_ids
        routing_weights = realized.routing_weights

    moe = routed_moe if is_routed else logits_moe
    moe_kwargs.update(
        routed_scaling_factor=1.0 if is_routed else CFG.routed_scaling_factor,
        routing_method_type=(
            RoutingMethodType.Renormalize.value
            if is_routed
            else RoutingMethodType.DeepSeekV3
        ),
    )

    def run(routing_logits, routing_bias, hidden_states, hidden_states_scale=None):
        activation_kwargs = {}
        if precision == "nvfp4":
            per_token_scale = hidden_per_token_scale
            if include_activation_quant:
                hidden_states, hidden_states_scale, per_token_scale = quantize_hidden(
                    hidden_states
                )
            activation_kwargs.update(
                hidden_states_scale=hidden_states_scale,
                per_token_scale=per_token_scale,
            )
        elif precision in ("mxfp8", "mxfp4"):
            if include_activation_quant:
                hidden_states, hidden_states_scale = quantize_hidden(hidden_states)
            activation_kwargs["hidden_states_scale"] = hidden_states_scale
        if is_routed:
            routing_kwargs = {"topk_ids": (routing_ids, routing_weights)}
            if precision in ("nvfp4", "mxfp8", "mxfp4"):
                routing_kwargs["routing_bias"] = None
        else:
            routing_kwargs = dict(
                routing_logits=routing_logits, routing_bias=routing_bias
            )
        return moe(
            hidden_states=hidden_states,
            **routing_kwargs,
            **activation_kwargs,
            **moe_kwargs,
        )

    # Pre-warm under autotune; measurement runs outside (see bench_cute_dsl).
    with _per_token_quantization(use_per_token_activation):
        with _autotune_context(do_autotune, autotune_cache):
            run(**input_kwargs)
            torch.cuda.synchronize()

        with _autotune_context(False, autotune_cache):
            return _measure_or_profile(
                run,
                input_kwargs,
                warmup,
                iters,
                use_cuda_graph,
                use_cupti,
                profile_cuda,
                profile_iters,
            )


# =============================================================================
# Main Benchmark
# =============================================================================


@dataclass
class BenchResult:
    """Single benchmark result for one backend at one token count."""

    backend: str
    tokens: int
    latency_ms: float
    tflops: float


def _family_backends(quant_family):
    if quant_family == "nvfp4":
        return (
            "cutedsl",
            "cute-dsl-per-token",
            "trtllm-nvfp4",
            "trtllm-per-token",
            "trtllm-bf16",
        )
    elif quant_family == "mxfp":
        return (
            "cute-dsl-w4a8",
            "cute-dsl-mxfp8",
            "trtllm-mxfp4",
            "trtllm-mxfp8",
            "trtllm-bf16",
        )
    else:
        raise ValueError(f"Unsupported quantization family: {quant_family}")


def _resolve_output_format(output_format, backends, quant_family):
    if output_format is None:
        if backends is None:
            output_format = "table"
        else:
            output_format = "csv"
    elif output_format not in ("table", "csv", "both"):
        raise ValueError("output_format must be 'table', 'csv', or 'both'")
    if output_format in ("table", "both"):
        outside_family = set(backends or ()) - set(_family_backends(quant_family))
        if outside_family:
            raise ValueError(
                f"The {quant_family} table does not include: {', '.join(sorted(outside_family))}; "
                "select the matching --quant-family or use --output-format csv"
            )
    return output_format


def run_benchmark(
    token_counts,
    warmup=10,
    iters=100,
    ep_config=None,
    tp_config=1,
    do_autotune=True,
    verbose=True,
    use_cuda_graph=True,
    use_cupti=True,
    use_wrapper=True,
    routing_bias_scale=0.01,
    include_activation_quant=False,
    use_fused_finalize=True,
    swap_ab=False,
    profile_cuda=False,
    profile_iters=10,
    profile_backend=None,
    backends=None,
    routing_input_mode="logits",
    distributions=("uniform",),
    autotune_cache=None,
    quant_family="nvfp4",
    output_format=None,
):
    """
    Unified benchmark for the selected MoE model and precision family.

    Autotuning runs in each backend's pre-warm step
    so the autotuner profiles all tactics on the default stream before
    ``bench_gpu_time`` switches to a side stream / captures into a CUDA graph.
    This guarantees the autotuner sees the same API (wrapper vs functional),
    EP config, and weight shapes as the timed runs, avoiding cache-key
    mismatches; the measurement loop itself runs with tuning disabled
    so ``choose_one`` takes the non-tuning fast path and host-side cache
    walks don't appear inside the CUDA-event interval.

    All output is buffered and printed after the benchmark (and autotuning)
    completes, so autotuner log messages do not interleave with the results
    table.

    Args:
        token_counts: List of token counts to benchmark
        warmup: Warmup iterations
        iters: Benchmark iterations
        ep_config: Expert parallelism degree; defaults to the selected model's EP
        tp_config: Tensor Parallelism degree used to reduce expert intermediate size
        do_autotune: Whether to autotune during benchmarking
        verbose: Print results to stdout
        use_cuda_graph: Whether to use CUDA graph for benchmarking
        use_cupti: Whether to use CUPTI for accurate GPU timing
        use_wrapper: Whether to use CuteDslMoEWrapper API (recommended)
        swap_ab: Opt in to Blackwell activations-on-B kernels and weight layout.
        routing_bias_scale: Scale for random routing bias generation
        include_activation_quant: Include initial activation quantization where
            supported; W4A16 and TRTLLM BF16 consume BF16 directly.
        use_fused_finalize: Use atomic fused CuTe DSL finalize and allow
            CUTLASS fused-finalize tactics; otherwise use non-fused finalize.
        profile_cuda: Capture one backend for an external CUDA profiler.
        profile_iters: Number of cold-L2 graph replays to capture.
        profile_backend: Backend to run when profile_cuda is enabled.
        quant_family: Default backend set and table, ``nvfp4`` or ``mxfp``.
            Explicit ``backends`` overrides the set.
        output_format: ``table``, ``csv``, or ``both``. Defaults to CSV with
            explicit ``backends``, otherwise table. Table output requires
            backends from the selected quantization family.

    Returns:
        List of BenchResult objects
    """
    output_format = _resolve_output_format(output_format, backends, quant_family)
    if ep_config is None:
        ep_config = CFG.default_ep
    if tp_config < 1 or BASE_INTERMEDIATE_SIZE % tp_config != 0:
        raise ValueError(
            f"tp_config must be a positive divisor of {BASE_INTERMEDIATE_SIZE}"
        )

    if CFG.num_experts % ep_config != 0:
        raise ValueError("ep_config must divide num_experts")
    num_local = CFG.num_experts // ep_config
    local_offset = 0
    CFG.intermediate_size = BASE_INTERMEDIATE_SIZE // tp_config

    results = []
    rows_and_histograms = []

    # Note: tuning is scoped to each bench_*'s pre-warm rather than
    # wrapping the entire measurement loop.  See bench_cute_dsl for rationale.
    for n in token_counts:
        row, histogram_record = _benchmark_single(
            n,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            use_wrapper=use_wrapper,
            swap_ab=swap_ab,
            routing_bias_scale=routing_bias_scale,
            do_autotune=do_autotune,
            include_activation_quant=include_activation_quant,
            use_fused_finalize=use_fused_finalize,
            profile_cuda=profile_cuda,
            profile_iters=profile_iters,
            profile_backend=profile_backend,
            backends=backends,
            routing_input_mode=routing_input_mode,
            distributions=distributions,
            autotune_cache=autotune_cache,
            quant_family=quant_family,
        )
        results.extend(row)
        rows_and_histograms.append((n, row, histogram_record))
        # Each row rebuilds full-model weights; release cached allocations so
        # measurements do not depend on the token-count scan order.
        gc.collect()
        torch.cuda.empty_cache()

    if verbose and output_format in ("csv", "both"):
        print("backend,tokens,latency_ms,tflops")
        for _, row, _ in rows_and_histograms:
            for result in row:
                print(
                    f"{result.backend},{result.tokens},{result.latency_ms:.6f},"
                    f"{result.tflops:.6f}"
                )
    if verbose and output_format in ("table", "both"):
        _print_header(
            ep_config,
            tp_config,
            num_local,
            use_cuda_graph,
            use_cupti,
            routing_bias_scale,
            include_activation_quant=include_activation_quant,
            use_fused_finalize=use_fused_finalize,
            quant_family=quant_family,
            swap_ab=swap_ab,
        )
        for n, row, histogram_record in rows_and_histograms:
            _print_row(
                row,
                histogram_record,
                n,
                quant_family=quant_family,
            )
        _print_footer(quant_family=quant_family)

    return results


def _benchmark_single(
    n,
    warmup,
    iters,
    num_local,
    local_offset,
    use_cuda_graph,
    use_cupti,
    use_wrapper=True,
    routing_bias_scale=0.01,
    do_autotune=True,
    include_activation_quant=False,
    use_fused_finalize=True,
    swap_ab=False,
    profile_cuda=False,
    profile_iters=10,
    profile_backend=None,
    backends=None,
    routing_input_mode="logits",
    distributions=("uniform",),
    autotune_cache=None,
    quant_family="nvfp4",
):
    """Benchmark all backends for a single token count.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API for CuteDSL.
        do_autotune: Forwarded to each bench_* function — wraps pre-warm only.
    """
    selected = set(backends or _family_backends(quant_family))
    if backends is None and profile_backend in (
        "cute-dsl-mxfp8",
        "cute-dsl-w4a8",
        "trtllm-mxfp8",
        "trtllm-mxfp4",
    ):
        selected = {profile_backend}
    trtllm_backends = (
        ("trtllm-nvfp4", "nvfp4", False, "TRTLLM NVFP4"),
        ("trtllm-per-token", "nvfp4", True, "TRTLLM PER-TOKEN"),
        ("trtllm-bf16", "bf16", False, "TRTLLM BF16"),
        ("trtllm-mxfp4", "mxfp4", False, "TRTLLM MXFP4x8"),
        ("trtllm-mxfp8", "mxfp8", False, "TRTLLM MXFP8"),
    )
    for selector, _, _, label in trtllm_backends:
        if selector in selected and profile_backend in (None, selector):
            skip_reason = _backend_skip_reason(label)
            if skip_reason is not None:
                if profile_backend is not None:
                    raise ValueError(skip_reason)
                if backends is not None:
                    print(f"Skipping {label}: {skip_reason}.")
                selected.remove(selector)
    if "cutlass" in selected and profile_backend in (None, "cutlass"):
        skip_reason = _backend_skip_reason("CUTLASS")
        if skip_reason is not None:
            if profile_backend is not None:
                raise ValueError(skip_reason)
            print(f"Skipping CUTLASS: {skip_reason}.")
            selected.remove("cutlass")
    if not selected:
        return [], {}
    inputs = create_inputs(n, routing_bias_scale=routing_bias_scale)
    histogram_record = _collect_expert_histogram(inputs, num_local, local_offset)
    run_cutlass = "cutlass" in selected and profile_backend in (None, "cutlass")
    lat = {}
    for selector, profile_selector, quant_mode, per_token, label in (
        ("cutedsl", "cute-dsl", "w4a4", False, "CuteDSL W4A4"),
        (
            "cute-dsl-per-token",
            "cute-dsl-per-token",
            "w4a4",
            True,
            "CuteDSL PER-TOKEN",
        ),
        ("cutedsl", "cute-dsl-w4a16", "w4a16", False, "CuteDSL W4A16"),
        ("cute-dsl-mxfp8", "cute-dsl-mxfp8", "mxfp8", False, "CuteDSL MXFP8"),
        ("cute-dsl-w4a8", "cute-dsl-w4a8", "w4a8", False, "CuteDSL MXFP4x8"),
    ):
        if selector in selected and profile_backend in (None, profile_selector):
            skip_reason = _backend_skip_reason(label, swap_ab, use_fused_finalize)
            if skip_reason is not None:
                if profile_backend is not None:
                    raise ValueError(skip_reason)
                if backends is not None:
                    print(f"Skipping {label}: {skip_reason}.")
                continue
            if quant_mode in ("mxfp8", "w4a8") and routing_input_mode != "logits":
                raise ValueError(f"{selector} requires logits routing")
            lat[label] = bench_cute_dsl(
                inputs,
                warmup,
                iters,
                num_local,
                local_offset,
                use_cuda_graph,
                use_cupti,
                use_wrapper=use_wrapper,
                swap_ab=swap_ab,
                do_autotune=do_autotune,
                quant_mode=quant_mode,
                use_per_token_activation=per_token,
                include_activation_quant=include_activation_quant,
                use_fused_finalize=use_fused_finalize,
                profile_cuda=profile_cuda,
                profile_iters=profile_iters,
                autotune_cache=autotune_cache,
            )
    if run_cutlass:
        lat["CUTLASS"] = bench_cutlass(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            do_autotune=do_autotune,
            include_activation_quant=include_activation_quant,
            use_fused_finalize=use_fused_finalize,
            profile_cuda=profile_cuda,
            profile_iters=profile_iters,
            autotune_cache=autotune_cache,
        )
    for selector, precision, per_token, label in trtllm_backends:
        if selector in selected and profile_backend in (None, selector):
            lat[label] = bench_trtllm(
                inputs,
                warmup,
                iters,
                num_local,
                local_offset,
                use_cuda_graph,
                use_cupti,
                do_autotune=do_autotune,
                use_per_token_activation=per_token,
                include_activation_quant=include_activation_quant,
                profile_cuda=profile_cuda,
                profile_iters=profile_iters,
                autotune_cache=autotune_cache,
                routing_input_mode=routing_input_mode,
                distributions=distributions,
                precision=precision,
            )

    # Build results
    results = []
    for backend, latency in lat.items():
        results.append(
            BenchResult(
                backend=backend,
                tokens=n,
                latency_ms=latency,
                tflops=calc_tflops(n, latency, num_local),
            )
        )
    return results, histogram_record


def _table_layout(quant_family):
    if quant_family == "nvfp4":
        labels = (
            "CuteDSL W4A4",
            "CuteDSL PER-TOKEN",
            "CuteDSL W4A16",
            "TRTLLM NVFP4",
            "TRTLLM PER-TOKEN",
            "TRTLLM BF16",
        )
        comparisons = (
            (
                "TRTLLM NVFP4",
                tuple((index, "TRTLLM NVFP4", labels[index]) for index in (0, 2, 1, 4)),
            ),
        )
        winner_width = max(map(len, labels))
    elif quant_family == "mxfp":
        labels = (
            "CuteDSL MXFP4x8",
            "CuteDSL MXFP8",
            "TRTLLM MXFP4x8",
            "TRTLLM MXFP8",
            "TRTLLM BF16",
        )
        comparisons = (
            (
                "TRTLLM",
                ((0, "TRTLLM MXFP4x8", "MXFP4x8"), (1, "TRTLLM MXFP8", "MXFP8")),
            ),
        )
        winner_width = 0
    else:
        raise ValueError(f"Unsupported quantization family: {quant_family}")
    backends = [(label, label, max(15, len(label))) for label in labels]
    baselines = [
        (
            baseline,
            max(
                len("Speedup vs " + baseline),
                sum(max(8, len(label)) for _, _, label in pairs) + len(pairs) - 1,
            ),
            pairs,
        )
        for baseline, pairs in comparisons
    ]
    widths = [
        6,
        *[width for _, _, width in backends],
        *[width for _, width, _ in baselines],
        *([winner_width] if winner_width else []),
        7,
        15,
    ]
    table_width = sum(widths) + 3 * (len(widths) - 1)
    return labels, backends, baselines, winner_width, table_width


def _print_header(
    ep_config,
    tp_config,
    num_local,
    use_cuda_graph,
    use_cupti,
    routing_bias_scale,
    include_activation_quant=False,
    use_fused_finalize=True,
    quant_family="nvfp4",
    swap_ab=False,
):
    """Print benchmark header."""
    _, backends, baselines, winner_width, table_width = _table_layout(quant_family)
    print("\n" + "=" * table_width)
    if quant_family == "mxfp":
        print(
            f"{CFG.name} MoE Benchmark: CuteDSL MXFP4x8 / MXFP8 vs TRTLLM MXFP4x8 / MXFP8 / BF16 "
            f"(EP={ep_config}, TP={tp_config})"
        )
    elif quant_family == "nvfp4":
        print(
            f"{CFG.name} MoE Benchmark: CuteDSL W4A4 / PER-TOKEN / W4A16 vs TRTLLM NVFP4 / PER-TOKEN / BF16 "
            f"(EP={ep_config}, TP={tp_config})"
        )
    print("=" * table_width)
    print(
        f"Model: hidden={CFG.hidden_size}, intermediate={CFG.intermediate_size}, "
        f"experts={CFG.num_experts}, top_k={CFG.top_k}"
    )
    print(
        f"EP Config: {num_local} local experts (simulating {CFG.num_experts // num_local}-way parallelism)"
    )
    print(
        f"TP Config: intermediate size {CFG.intermediate_size} "
        f"(simulating {tp_config}-way parallelism)"
    )
    print(
        f"CUDA Graph: {'enabled' if use_cuda_graph else 'disabled'}, CUPTI: {'enabled' if use_cupti else 'disabled'}"
    )
    print(
        f"Routing bias scale: {routing_bias_scale} "
        f"(larger values tend to create expert imbalance)"
    )
    if CFG.routing == "sqrtsoftplus":
        print(
            "Routing: shared precomputed sqrt(softplus(logits)), bias selects only, "
            f"top-{CFG.top_k} normalized then scaled by {CFG.routed_scaling_factor}; "
            "routing excluded from every backend's timing."
        )
        print(
            f"Activation: SwiGLU, gate clamped above {CFG.swiglu_limit:g}, "
            f"up clamped to [-{CFG.swiglu_limit:g}, {CFG.swiglu_limit:g}]."
        )
    elif CFG.routing == "sigmoid":
        print(
            "Routing: shared precomputed sigmoid(logits), bias selects only, "
            f"top-{CFG.top_k} normalized then scaled by {CFG.routed_scaling_factor}; "
            "routing excluded from every backend's timing."
        )
        if CFG.activation == "situ":
            print(f"Activation: SiTU({CFG.situ_beta:g}, {CFG.situ_linear_beta:g}).")
        elif CFG.activation == "relu2":
            print("Activation: ReLU2 (non-gated squared ReLU).")
        else:
            raise ValueError(f"Unsupported sigmoid-preset activation: {CFG.activation}")
        print(
            "Scope: routed latent experts only; latent projections, RMSNorm, "
            "shared experts, router GEMM and communication excluded."
        )
    for backend, _, _ in backends:
        skip_reason = _backend_skip_reason(backend, swap_ab, use_fused_finalize)
        if skip_reason is not None:
            print(f"N/A: {skip_reason}; skipped before preparation and timing.")
    if quant_family == "nvfp4":
        print(
            "Timed initial activation quantization for FP4-activation backends: "
            f"{'included' if include_activation_quant else 'excluded'}; "
            "W4A16 and TRTLLM BF16 consume BF16 directly"
        )
    elif quant_family == "mxfp":
        print("MXFP8 activations: native UE8M0 scale per 32 elements along K.")
        print(
            "Timed initial activation quantization: "
            f"{'included' if include_activation_quant else 'excluded'}; "
            "weight preparation and tuning excluded. TRTLLM BF16 consumes BF16 directly."
        )
    print(
        "CuteDSL finalize: "
        f"{'atomic fused' if use_fused_finalize else 'deterministic two-stage'}"
    )
    if quant_family == "nvfp4":
        print(
            "TRTLLM NVFP4 / PER-TOKEN / BF16 finalize: native (unaffected by --no-fused-finalize)."
        )
    elif quant_family == "mxfp":
        print(
            "TRTLLM MXFP4x8 / MXFP8 / BF16 finalize: native (unaffected by --no-fused-finalize)."
        )
    print("-" * table_width)
    headers = [f"{'Tokens':>6}"]
    headers.extend(f"{label:^{width}}" for _, label, width in backends)
    headers.extend(f"{'Speedup vs ' + label:^{width}}" for label, width, _ in baselines)
    if winner_width:
        headers.append(f"{'Winner':^{winner_width}}")
    headers.extend((f"{'Active':^7}", f"{'Stats':^15}"))
    print(" | ".join(headers))
    units = [f"{'':>6}"]
    units.extend(
        f"{('ms'.rjust(7) + ' ' + 'TFLOPS'.rjust(7)):>{width}}"
        for _, _, width in backends
    )
    for _, width, pairs in baselines:
        labels = " ".join(label.rjust(max(8, len(label))) for _, _, label in pairs)
        units.append(f"{labels:>{width}}")
    if winner_width:
        units.append(f"{'':^{winner_width}}")
    units.extend((f"{'experts':^7}", f"{'min/max/median':^15}"))
    print(" | ".join(units))
    print("-" * table_width)


def _print_row(
    results,
    histogram_record,
    tokens,
    quant_family="nvfp4",
):
    """Print a single row of benchmark results."""
    # Extract values by backend
    rows = {result.backend: result for result in results}
    _, backends, baselines, winner_width, _ = _table_layout(quant_family)
    fields = [f"{tokens:>6}"]
    for key, _, width in backends:
        if key in rows:
            latency = f"{rows[key].latency_ms:>7.3f} {rows[key].tflops:>7.1f}"
        else:
            latency = "N/A"
        fields.append(f"{latency:>{width}}")
    for _, width, pairs in baselines:
        speedups = []
        for index, key, label in pairs:
            comparison = backends[index][0]
            ratio_width = max(8, len(label))
            if key in rows and comparison in rows:
                speedups.append(
                    f"{rows[key].latency_ms / rows[comparison].latency_ms:>{ratio_width - 1}.2f}x"
                )
            else:
                speedups.append(f"{'N/A':>{ratio_width}}")
        fields.append(f"{' '.join(speedups):>{width}}")
    # Find winner
    if winner_width:
        available = [entry for entry in backends if entry[0] in rows]
        if available:
            _, winner, _ = min(available, key=lambda entry: rows[entry[0]].latency_ms)
        else:
            winner = "N/A"
        fields.append(f"{winner:^{winner_width}}")
    if histogram_record:
        stats = (
            f"{histogram_record['min_count']:>3}/"
            f"{histogram_record['max_count']:>3}/"
            f"{histogram_record['median_count']:>7.2f}"
        )
        fields.extend(
            (
                f"{histogram_record['active_local_experts']:>7}",
                f"{stats:>15}",
            )
        )
    else:
        fields.extend((f"{'N/A':>7}", f"{'N/A':>15}"))
    print(" | ".join(fields))


def _print_footer(quant_family="nvfp4"):
    """Print benchmark footer."""
    table_width = _table_layout(quant_family)[-1]
    print("-" * table_width)
    print(
        "Speedup > 1.0 means the named comparison arm is faster than its TRTLLM baseline."
    )


def _collect_expert_histogram(inputs, num_local, local_offset):
    topk_values, topk_indices = _routing_tensors(inputs)
    _route_tokens(
        inputs["router_logits"],
        inputs["routing_bias"].float(),
        topk_values,
        topk_indices,
    )

    expert_hist = torch.bincount(
        topk_indices.reshape(-1).to(torch.int64), minlength=CFG.num_experts
    )
    local_hist = expert_hist[local_offset : local_offset + num_local]
    local_hist_f32 = local_hist.to(torch.float32)
    active_local_experts = int((local_hist > 0).sum().item())
    if local_hist.numel() > 0:
        min_count = int(local_hist.min().item())
        max_count = int(local_hist.max().item())
        median_count = float(torch.quantile(local_hist_f32, 0.5).item())
    else:
        min_count = 0
        max_count = 0
        median_count = 0.0

    return {
        "active_local_experts": active_local_experts,
        "min_count": min_count,
        "max_count": max_count,
        "median_count": median_count,
    }


def main():
    global CFG, BASE_INTERMEDIATE_SIZE

    parser = argparse.ArgumentParser(description="MoE Performance Benchmark")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="v3",
        help="Model shape, routing and activation preset. V4.1-Flash, Kimi K3 and Nemotron-3 Ultra share precomputed routes across all backends, excluded from timing.",
    )
    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="Comma-separated token counts (default: 128-4096 for throughput, 1-128 for gen-phase)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--no-autotune",
        "--skip-autotune",
        dest="no_autotune",
        action="store_true",
        help="Disable profiling; a supplied cache is still loaded",
    )
    parser.add_argument(
        "--cache",
        "--tuning-cache",
        "--bundle-output",
        dest="cache",
        help="Managed autotuning cache directory shared across backends",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Override the model's number of global routing experts",
    )
    parser.add_argument(
        "--quant-family",
        choices=("nvfp4", "mxfp"),
        default="nvfp4",
        help="Default comparison table: NVFP4 W4A4/PER-TOKEN/W4A16 or MXFP4x8/MXFP8; both include TRTLLM BF16.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        help="Override the family with a comma-separated subset of cutedsl,cute-dsl-per-token,cutlass,trtllm-nvfp4,trtllm-per-token,trtllm-bf16,cute-dsl-mxfp8,cute-dsl-w4a8,trtllm-mxfp8,trtllm-mxfp4; defaults to CSV output.",
    )
    parser.add_argument(
        "--output-format",
        choices=("table", "csv", "both"),
        help="Result format (default: CSV with --backends, otherwise table). Tables use --quant-family columns; unselected or unsupported backends show N/A.",
    )
    parser.add_argument(
        "--distributions",
        type=str,
        help="Comma-separated DA tuner distributions",
    )
    parser.add_argument(
        "--routing-input-mode",
        choices=("logits", "routed"),
        default="logits",
        help="TRTLLM routing input representation",
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--gen-phase",
        action="store_true",
        help="Use generation phase token counts (1-128 instead of 128-4096)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=None,
        choices=[1, 2, 4, 8, 16],
        help="Expert parallelism simulation (default: V3 EP8, V4.1-Flash EP4, Kimi K3 EP8, Nemotron-3 Ultra EP8).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor Parallelism simulation: divide the expert intermediate size by TP.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph for benchmarking (enabled by default)",
    )
    parser.add_argument(
        "--no-cupti",
        action="store_true",
        help="Disable CUPTI for GPU timing (enabled by default)",
    )
    parser.add_argument(
        "--functional-api",
        action="store_true",
        help="Use functional API instead of CuteDslMoEWrapper for CuteDSL benchmark",
    )
    parser.add_argument(
        "--include-activation-quant",
        action="store_true",
        help="Include initial activation quantization for NVFP4 and TRTLLM MX modes; CuTe DSL MX modes require it excluded. BF16 is unaffected.",
    )
    parser.add_argument(
        "--swap-ab",
        action="store_true",
        help="Use the Blackwell activations-on-B CuTe DSL kernels and weight layout.",
    )
    parser.add_argument(
        "--no-fused-finalize",
        action="store_false",
        dest="use_fused_finalize",
        help="Use deterministic two-stage CuTe DSL finalize and disable CUTLASS fused-finalize tactics.",
    )
    parser.add_argument(
        "--profile-cuda",
        action="store_true",
        help="Capture steady-state iterations between cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
        help="Number of CUDA graph replays captured by --profile-cuda.",
    )
    parser.add_argument(
        "--profile-backend",
        choices=[
            "cute-dsl",
            "cute-dsl-w4a16",
            "cute-dsl-per-token",
            "cute-dsl-mxfp8",
            "cute-dsl-w4a8",
            "cutlass",
            "trtllm-nvfp4",
            "trtllm-per-token",
            "trtllm-bf16",
            "trtllm-mxfp8",
            "trtllm-mxfp4",
        ],
        help="Backend captured by --profile-cuda.",
    )
    parser.add_argument(
        "--routing-bias-scale",
        type=float,
        default=0.01,
        help="Scale for random routing bias. Larger values tend to create expert imbalance.",
    )
    args = parser.parse_args()

    CFG = replace(MODEL_CONFIGS[args.model])
    BASE_INTERMEDIATE_SIZE = CFG.intermediate_size
    if args.ep is None:
        args.ep = CFG.default_ep
    if args.num_experts is None:
        args.num_experts = CFG.num_experts
    if args.tp < 1:
        parser.error("--tp must be positive")
    if args.num_experts < 1:
        parser.error("--num-experts must be positive")
    if args.num_experts % args.ep != 0:
        parser.error("--ep must divide --num-experts")
    backends = None
    if args.backends:
        backends = tuple(
            item.strip() for item in args.backends.split(",") if item.strip()
        )
        unknown = sorted(
            set(backends)
            - {
                "cutedsl",
                "cute-dsl-per-token",
                "cutlass",
                "trtllm-nvfp4",
                "trtllm-per-token",
                "trtllm-bf16",
                "cute-dsl-mxfp8",
                "cute-dsl-w4a8",
                "trtllm-mxfp8",
                "trtllm-mxfp4",
            }
        )
        if unknown:
            parser.error(f"unknown --backends value(s): {', '.join(unknown)}")
    try:
        output_format = _resolve_output_format(
            args.output_format, backends, args.quant_family
        )
    except ValueError as error:
        parser.error(str(error))
    distributions = tuple(
        item.strip()
        for item in (args.distributions or "uniform").split(",")
        if item.strip()
    )
    if not distributions:
        parser.error("--distributions must not be empty")
    if BASE_INTERMEDIATE_SIZE % args.tp != 0:
        parser.error(
            f"--tp must divide the expert intermediate size ({BASE_INTERMEDIATE_SIZE})"
        )
    if args.profile_iters < 1:
        parser.error("--profile-iters must be positive")
    if args.profile_cuda != (args.profile_backend is not None):
        parser.error("--profile-cuda and --profile-backend must be specified together")
    if backends and args.profile_backend is not None:
        profile_backend = (
            "cutedsl"
            if args.profile_backend in ("cute-dsl", "cute-dsl-w4a16")
            else args.profile_backend
        )
        if profile_backend not in backends:
            parser.error(
                f"--profile-backend {args.profile_backend} is excluded by --backends"
            )
    selected = set(backends or _family_backends(args.quant_family))
    if args.profile_backend is not None:
        selected = {args.profile_backend}
    if selected & {"cute-dsl-mxfp8", "cute-dsl-w4a8"}:
        if args.include_activation_quant:
            parser.error("CuTe DSL MX modes do not support --include-activation-quant")
        if args.routing_input_mode != "logits":
            parser.error("CuTe DSL MX modes require --routing-input-mode logits")
    if not is_sm100_family():
        print("ERROR: Requires SM100 family GPU (Blackwell: SM100, SM103)")
        return 1

    CFG.num_experts = args.num_experts
    if args.routing_input_mode == "routed" and not args.no_cuda_graph:
        os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "1"
        os.environ["FLASHINFER_DA_DISTRIBUTIONS"] = ",".join(distributions)
    else:
        os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "0"

    # Determine token counts
    if args.num_tokens:
        tokens = [int(x) for x in args.num_tokens.split(",")]
    elif args.gen_phase:
        tokens = GEN_PHASE_TOKENS  # [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        tokens = TOKEN_COUNTS  # [128, 256, 512, 1024, 2048, 4096]
    if args.profile_cuda and len(tokens) != 1:
        parser.error("--profile-cuda requires exactly one token count")
    print(f"\n{CFG.name} MoE Performance Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CuteDSL API: {'Functional' if args.functional_api else 'Wrapper'}")
    if selected & {"cute-dsl-per-token", "trtllm-per-token"}:
        print(
            "PER-TOKEN activation recipe (initial and FC2): 4over6, MSE, E4M3 max=256, "
            "4over6 error fast math enabled; base FP4 quantization fast math disabled. "
            "Weights retain standard NVFP4 quantization."
        )
    print(f"Initial activation quantization: {args.include_activation_quant}")
    print(f"Expert parallelism simulation: EP={args.ep}")
    if CFG.routing == "sqrtsoftplus":
        print(
            "Routing: model sqrtsoftplus routes precomputed and shared by all backends; "
            "routing excluded from timing, TRTLLM uses routed APIs."
        )
    elif CFG.routing == "sigmoid":
        print(
            "Routing: model sigmoid routes precomputed and shared by all backends; "
            "routing excluded from timing, TRTLLM uses routed APIs."
        )
    elif CFG.routing == "deepseek_v3":
        print(f"Routing input mode: {args.routing_input_mode}")
    else:
        raise ValueError(f"Unsupported model routing: {CFG.routing}")
    if args.quant_family == "nvfp4":
        print(
            "CuteDSL modes: W4A4 / PER-TOKEN / W4A16; TRTLLM: NVFP4 / PER-TOKEN / BF16"
        )
    elif args.quant_family == "mxfp":
        print(
            "CuteDSL modes: MXFP4x8 and MXFP8; baselines: matching TRTLLM modes and BF16"
        )
    print(f"Tensor parallelism simulation: TP={args.tp}")
    print(f"CUDA profiler capture: {args.profile_cuda}")
    print(
        "CuteDSL finalize: "
        f"{'atomic fused' if args.use_fused_finalize else 'deterministic two-stage'}"
    )
    if args.quant_family == "nvfp4":
        print(
            "TRTLLM NVFP4 / PER-TOKEN / BF16 finalize: native (unaffected by --no-fused-finalize)."
        )
    elif args.quant_family == "mxfp":
        print(
            "TRTLLM MXFP4x8 / MXFP8 / BF16 finalize: native (unaffected by --no-fused-finalize)."
        )

    print(f"CuTe DSL swap_ab: {args.swap_ab}")
    run_benchmark(
        token_counts=tokens,
        warmup=args.warmup,
        iters=args.iters,
        ep_config=args.ep,
        tp_config=args.tp,
        do_autotune=not args.no_autotune,
        verbose=not args.quiet and not args.profile_cuda,
        use_cuda_graph=not args.no_cuda_graph,
        use_cupti=not args.no_cupti,
        use_wrapper=not args.functional_api,
        routing_bias_scale=args.routing_bias_scale,
        include_activation_quant=args.include_activation_quant,
        use_fused_finalize=args.use_fused_finalize,
        swap_ab=args.swap_ab,
        profile_cuda=args.profile_cuda,
        profile_iters=args.profile_iters,
        profile_backend=args.profile_backend,
        backends=backends,
        routing_input_mode=args.routing_input_mode,
        distributions=distributions,
        autotune_cache=args.cache,
        quant_family=args.quant_family,
        output_format=output_format,
    )

    return 0


if __name__ == "__main__":
    exit(main())
