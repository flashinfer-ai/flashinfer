#!/usr/bin/env python3
"""DeepSeek-V3 MoE Performance Benchmark - CuteDSL vs CUTLASS vs TRTLLM.

Compares three NVFP4 MoE backends on DeepSeek-V3 configuration:
- CuteDSL: FlashInfer's CuteDSL-based implementation
- CUTLASS: NVIDIA CUTLASS-based implementation
- TRTLLM: TensorRT-LLM's implementation

Usage:
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

    # Real 8-GPU EP or TP, including communication
    torchrun --standalone --nproc-per-node=8 bench_moe_deepseek.py \
        --num-gpus 8 --ep 8 --cute-dsl-only --use-bf16-activation
    torchrun --standalone --nproc-per-node=8 bench_moe_deepseek.py \
        --num-gpus 8 --tp 8 --cute-dsl-only --use-bf16-activation

    # Custom token counts
    python bench_moe_deepseek.py --num-tokens 64,128,256

    # Disable CUDA graph (useful for debugging or profiling)
    python bench_moe_deepseek.py --no-cuda-graph

    # Disable CUPTI (use CUDA events for timing instead)
    python bench_moe_deepseek.py --no-cupti

    # CuTe DSL finalize modes
    python bench_moe_deepseek.py --functional-api  # atomic fused (default)
    python bench_moe_deepseek.py --functional-api --no-fused-finalize  # deterministic

Metrics:
    - ms: Latency in milliseconds
    - TFLOPS: Computational throughput
    - Speedup: CuteDSL latency / other backend latency (>1 = CuteDSL faster)
"""

import argparse
import gc
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class DeepSeekConfig:
    hidden_size: int = 7168
    intermediate_size: int = 2048
    num_experts: int = 256
    n_group: int = 8
    topk_group: int = 4
    top_k: int = 8
    routed_scaling_factor: float = 2.5


CFG = DeepSeekConfig()
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


@dataclass
class DistributedBenchResult:
    num_tokens: int
    e2e_ms: float


def is_sm100_family():
    """Check for SM100 family (Blackwell: SM100, SM103).

    CuteDSL MoE NVFP4 kernels are optimized for SM10x architecture.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10


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
            2 * CFG.hidden_size * 2 * CFG.intermediate_size
            + 2 * CFG.intermediate_size * CFG.hidden_size
        )
    )
    return flops / (ms * 1e-3) / 1e12


def interleave(x, gs=64):
    M, K = x.shape[-2], x.shape[-1]
    return (
        x.view(*x.shape[:-2], 2, M // (gs * 2), gs, K)
        .transpose(-4, -3)
        .contiguous()
        .view(*x.shape)
    )


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
        torch.randn(CFG.num_experts, device=dev, dtype=torch.bfloat16)
        * routing_bias_scale
    )

    # Hidden states
    hb = torch.randn(n, CFG.hidden_size, device=dev, dtype=torch.bfloat16) / 10
    hg = FP8M * FP4M / hb.abs().max().float()

    # Weights (BF16)
    w1b = (
        torch.randn(
            CFG.num_experts,
            2 * CFG.intermediate_size,
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
    use_per_token_activation=False,
    use_bf16_activation=False,
    include_activation_quant=False,
    use_fused_finalize=True,
    enable_pdl=True,
    profile_cuda=False,
    profile_iters=10,
):
    """Benchmark CuteDSL MoE.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API (recommended for CUDA graph).
                    If False, use cute_dsl_fused_moe_nvfp4 functional API.
        do_autotune: If True, run the pre-warm pass under autotune(True) so the
                    autotuner profiles all buckets and populates its cache. The
                    measurement loop runs OUTSIDE the autotune context so that
                    choose_one cache lookups don't appear inside the CUDA-event
                    interval when bench_gpu_time falls back to events (i.e. when
                    both CUDA graphs and CUPTI are disabled).
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.
        use_bf16_activation: Keep activations in BF16 and dequantize NVFP4
            weights online in both GEMMs.
        include_activation_quant: Include the initial activation FP4
            quantization in the measured CuTe DSL W4A4 path.
        enable_pdl: Enable Programmatic Dependent Launch for CuTe DSL kernels.
        profile_cuda: Capture steady-state CUDA graph replays between
            cudaProfilerStart/Stop instead of benchmarking.
        profile_iters: Number of cold-L2 graph replays to capture.
    """
    import contextlib

    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import fused_topk_deepseek
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"
    gs1 = torch.tensor([1.0], device=dev)

    if include_activation_quant and use_bf16_activation:
        raise ValueError(
            "include_activation_quant is incompatible with BF16 activation"
        )

    tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

    activation_global_scale = (
        make_nvfp4_global_scale(
            inputs["hidden_bf16"],
            per_token_activation=True,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
        if use_per_token_activation
        else gs1
    )

    if use_bf16_activation or include_activation_quant:
        xf = inputs["hidden_bf16"]
        xs = None
        hidden_per_token_scale = None
    elif use_per_token_activation:
        xf, xs, hidden_per_token_scale = nvfp4_quantize(
            inputs["hidden_bf16"],
            activation_global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        xf, xs = fp4_quantize(inputs["hidden_bf16"], gs1, sv, False, False)
        hidden_per_token_scale = None
    if xs is not None:
        xs = xs.unsqueeze(-1)

    def prepare_activation(x, x_sf):
        per_token_scale = hidden_per_token_scale
        if include_activation_quant:
            if use_per_token_activation:
                x, x_sf, per_token_scale = nvfp4_quantize(
                    x,
                    activation_global_scale,
                    sfLayout=SfLayout.layout_linear,
                    per_token_activation=True,
                    backend="cute-dsl",
                )
            else:
                x, x_sf = fp4_quantize(x, gs1, sv, False, False, backend="cute-dsl")
            x_sf = x_sf.unsqueeze(-1)
        return x, x_sf, per_token_scale

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    # Slice weights to LOCAL experts only
    w1_local = inputs["w1_bf16"][expert_start:expert_end]
    w2_local = inputs["w2_bf16"][expert_start:expert_end]

    w1i = interleave(w1_local, 64)
    w1f = w1i.view(num_local_experts * 2 * CFG.intermediate_size, CFG.hidden_size)
    w1q, w1s = fp4_quantize(w1f, gs1, sv, False, True)
    w1q = w1q.view(num_local_experts, 2 * CFG.intermediate_size, CFG.hidden_size // 2)
    w1s = convert_sf_to_mma_layout(
        w1s, 2 * CFG.intermediate_size, CFG.hidden_size, num_local_experts, sv
    )

    w2f = w2_local.view(num_local_experts * CFG.hidden_size, CFG.intermediate_size)
    w2q, w2s = fp4_quantize(w2f, gs1, sv, False, True)
    w2q = w2q.view(num_local_experts, CFG.hidden_size, CFG.intermediate_size // 2)
    w2s = convert_sf_to_mma_layout(
        w2s, CFG.hidden_size, CFG.intermediate_size, num_local_experts, sv
    )

    # Alpha sized for LOCAL experts only
    alpha, fc2sc = (
        torch.ones(num_local_experts, device=dev),
        torch.tensor([1.0], device=dev),
    )

    # Pre-convert routing bias to float32
    routing_bias_f32 = inputs["routing_bias"].float()

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
            enable_pdl=enable_pdl,
        )

        def run(x, x_sf, router_logits, routing_bias, topk_values, topk_indices):
            x, x_sf, per_token_scale = prepare_activation(x, x_sf)
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
            return moe.run(
                x=x,
                x_sf=x_sf,
                token_selected_experts=topk_indices,
                token_final_scales=topk_values,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=None if use_bf16_activation else fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                per_token_scale=per_token_scale,
            )
    else:
        # Use functional API
        from flashinfer import cute_dsl_fused_moe_nvfp4

        def run(x, x_sf, router_logits, routing_bias, topk_values, topk_indices):
            x, x_sf, per_token_scale = prepare_activation(x, x_sf)
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
            return cute_dsl_fused_moe_nvfp4(
                x=x,
                x_sf=x_sf,
                token_selected_experts=topk_indices,
                token_final_scales=topk_values,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=None if use_bf16_activation else fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                num_experts=CFG.num_experts,
                top_k=CFG.top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                per_token_scale=per_token_scale,
                use_fused_finalize=use_fused_finalize,
                enable_pdl=enable_pdl,
            )

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
    with autotune(True) if do_autotune else contextlib.nullcontext():
        run(**input_kwargs)
        torch.cuda.synchronize()

    if profile_cuda:
        from flashinfer.testing.utils import get_l2_cache_size

        runner = lambda: run(**input_kwargs)
        if use_cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(**input_kwargs)
            runner = graph.replay
        for _ in range(3):
            runner()
        torch.cuda.synchronize()

        l2_flush = torch.empty(2 * get_l2_cache_size(), device=dev, dtype=torch.int8)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(profile_iters):
            l2_flush.zero_()
            torch.cuda.synchronize()
            runner()
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return float("nan")

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


def bench_cutlass(
    inputs,
    warmup=10,
    iters=100,
    num_local_experts=None,
    local_expert_offset=0,
    use_cuda_graph=True,
    use_cupti=True,
    do_autotune=True,
):
    """Benchmark CUTLASS MoE.

    Args:
        do_autotune: See ``bench_cute_dsl`` for the autotune-scope rationale.
    """
    import contextlib

    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import fused_topk_deepseek, cutlass_fused_moe
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"

    tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

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
        # Routing (included in timing for fair comparison with TRTLLM)
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
        )
        return output

    input_kwargs = {
        "hidden": hidden_fp4,
        "sf": input_sf,
        "router_logits": inputs["router_logits"],
        "routing_bias": routing_bias_f32,
        "topk_values": tv,
        "topk_indices": ti,
    }

    # Pre-warm under autotune; measurement runs outside (see bench_cute_dsl).
    with autotune(True) if do_autotune else contextlib.nullcontext():
        run(**input_kwargs)
        torch.cuda.synchronize()

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
):
    """Benchmark TRT-LLM-Gen MoE.

    Args:
        do_autotune: See ``bench_cute_dsl`` for the autotune-scope rationale.
    """
    import contextlib

    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe, RoutingMethodType
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, dev = inputs["router_logits"].shape[0], inputs["router_logits"].device
    sv, etm, cache = 16, 128, {}

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    hg = inputs["hidden_gs"]
    if use_per_token_activation:
        hidden_global_scale = make_nvfp4_global_scale(
            inputs["hidden_bf16"],
            per_token_activation=True,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
        hfp, hsf, hidden_per_token_scale = nvfp4_quantize(
            inputs["hidden_bf16"],
            hidden_global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        hfp, hsf = fp4_quantize(inputs["hidden_bf16"], hg, sv, False, True)
        hidden_per_token_scale = None
    hfp = hfp.view(torch.uint8).reshape(n, CFG.hidden_size // 2)
    hsc = (
        hsf.view(torch.float8_e4m3fn)
        .flatten()[: n * CFG.hidden_size // sv]
        .reshape(n, CFG.hidden_size // sv)
    )

    def prep(bf16, gs, M, K):
        """Prepare weights for LOCAL experts only."""
        fl, sl = [], []
        for e in range(expert_start, expert_end):
            q, s = fp4_quantize(bf16[e], gs[e], sv, False, False)
            fl.append(q.view(torch.uint8).reshape(M, K // 2))
            sl.append(s.view(torch.float8_e4m3fn).reshape(M, K // sv))
        return torch.stack(fl), torch.stack(sl)

    w1f, w1s = prep(
        inputs["w1_bf16"], inputs["w1_gs"], 2 * CFG.intermediate_size, CFG.hidden_size
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
                block_scale_interleave(sf[i].view(torch.uint8)[ps.to(dev)].contiguous())
            )
        return torch.stack(fsh), torch.stack(ssh)

    w1f, w1s = shuf(w1f, w1s, _maybe_get_cached_w3_w1_permute_indices)
    w2f, w2s = shuf(w2f, w2s, get_w2_permute_indices_with_cache)
    w1s = w1s.view(torch.float8_e4m3fn).reshape(
        num_local_experts, 2 * CFG.intermediate_size, CFG.hidden_size // sv
    )
    w2s = w2s.view(torch.float8_e4m3fn).reshape(
        num_local_experts, CFG.hidden_size, CFG.intermediate_size // sv
    )

    # Scale tensors sized for LOCAL experts only
    sc = torch.ones(num_local_experts, device=dev, dtype=torch.float32)

    def run(routing_logits, routing_bias, hidden_states, hidden_states_scale):
        return trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
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
            num_experts=CFG.num_experts,
            top_k=CFG.top_k,
            n_group=CFG.n_group,
            topk_group=CFG.topk_group,
            intermediate_size=CFG.intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=num_local_experts,
            routed_scaling_factor=CFG.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            per_token_scale=hidden_per_token_scale,
            do_finalize=True,
        )

    input_kwargs = {
        "routing_logits": inputs["router_logits"],
        "routing_bias": inputs["routing_bias"],
        "hidden_states": hfp,
        "hidden_states_scale": hsc,
    }

    # Pre-warm under autotune; measurement runs outside (see bench_cute_dsl).
    with autotune(True) if do_autotune else contextlib.nullcontext():
        run(**input_kwargs)
        torch.cuda.synchronize()

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


# =============================================================================
# Autotune
# =============================================================================


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


def run_benchmark(
    token_counts,
    warmup=10,
    iters=100,
    ep_config=1,
    tp_config=1,
    do_autotune=True,
    verbose=True,
    use_cuda_graph=True,
    use_cupti=True,
    use_wrapper=True,
    routing_bias_scale=0.01,
    use_per_token_activation=False,
    use_bf16_activation=False,
    include_activation_quant=False,
    use_fused_finalize=True,
    cute_dsl_only=False,
    enable_pdl=True,
    profile_cuda=False,
    profile_iters=10,
):
    """
    Unified benchmark for DeepSeek-V3 MoE backends.

    Autotuning runs in each backend's pre-warm step (under ``autotune(True)``)
    so the autotuner profiles all tactics on the default stream before
    ``bench_gpu_time`` switches to a side stream / captures into a CUDA graph.
    This guarantees the autotuner sees the same API (wrapper vs functional),
    EP config, and weight shapes as the timed runs, avoiding cache-key
    mismatches; the measurement loop itself runs outside ``autotune(True)``
    so ``choose_one`` takes the non-tuning fast path and host-side cache
    walks don't appear inside the CUDA-event interval.

    All output is buffered and printed after the benchmark (and autotuning)
    completes, so autotuner log messages do not interleave with the results
    table.

    Args:
        token_counts: List of token counts to benchmark
        warmup: Warmup iterations
        iters: Benchmark iterations
        ep_config: Expert Parallelism config (1, 8, or 16)
        tp_config: Tensor Parallelism degree used to reduce expert intermediate size
        do_autotune: Whether to autotune during benchmarking
        verbose: Print results to stdout
        use_cuda_graph: Whether to use CUDA graph for benchmarking
        use_cupti: Whether to use CUPTI for accurate GPU timing
        use_wrapper: Whether to use CuteDslMoEWrapper API (recommended)
        routing_bias_scale: Scale for random routing bias generation
        use_per_token_activation: Whether supported FP4 MoE backends should use
            per-token NVFP4 activation scaling.
        use_bf16_activation: Whether CuTe DSL should keep activations in BF16
            and dequantize NVFP4 weights online.
        include_activation_quant: Include the initial activation FP4
            quantization in the CuTe DSL timing.
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.
        cute_dsl_only: Skip CUTLASS and TRTLLM backends.
        enable_pdl: Enable Programmatic Dependent Launch for CuTe DSL kernels.
        profile_cuda: Capture the CuTe DSL pipeline for an external CUDA profiler.
        profile_iters: Number of cold-L2 graph replays to capture.

    Returns:
        List of BenchResult objects
    """
    if tp_config < 1 or BASE_INTERMEDIATE_SIZE % tp_config != 0:
        raise ValueError(
            f"tp_config must be a positive divisor of {BASE_INTERMEDIATE_SIZE}"
        )

    # Get EP configuration
    ep_cfg = EP_CONFIGS.get(ep_config, EP_CONFIGS[1])
    num_local = ep_cfg["num_local_experts"]
    local_offset = ep_cfg["local_expert_offset"]
    CFG.intermediate_size = BASE_INTERMEDIATE_SIZE // tp_config

    results = []
    rows_and_histograms = []

    # Note: autotune(True) is now scoped to each bench_*'s pre-warm rather than
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
            routing_bias_scale=routing_bias_scale,
            do_autotune=do_autotune,
            use_per_token_activation=use_per_token_activation,
            use_bf16_activation=use_bf16_activation,
            include_activation_quant=include_activation_quant,
            use_fused_finalize=use_fused_finalize,
            cute_dsl_only=cute_dsl_only,
            enable_pdl=enable_pdl,
            profile_cuda=profile_cuda,
            profile_iters=profile_iters,
        )
        results.extend(row)
        rows_and_histograms.append((row, histogram_record))
        # Each row rebuilds full-model weights; release cached allocations so
        # measurements do not depend on the token-count scan order.
        gc.collect()
        torch.cuda.empty_cache()

    if verbose:
        _print_header(
            ep_config,
            tp_config,
            num_local,
            use_cuda_graph,
            use_cupti,
            routing_bias_scale,
            use_per_token_activation=use_per_token_activation,
            use_bf16_activation=use_bf16_activation,
            include_activation_quant=include_activation_quant,
            use_fused_finalize=use_fused_finalize,
            cute_dsl_only=cute_dsl_only,
            enable_pdl=enable_pdl,
        )
        for row, histogram_record in rows_and_histograms:
            _print_row(row, histogram_record)
        _print_footer(ep_config, num_local, cute_dsl_only=cute_dsl_only)

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
    use_per_token_activation=False,
    use_bf16_activation=False,
    include_activation_quant=False,
    use_fused_finalize=True,
    cute_dsl_only=False,
    enable_pdl=True,
    profile_cuda=False,
    profile_iters=10,
):
    """Benchmark all backends for a single token count.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API for CuteDSL.
        do_autotune: Forwarded to each bench_* function — wraps pre-warm only.
    """
    inputs = create_inputs(n, routing_bias_scale=routing_bias_scale)
    histogram_record = _collect_expert_histogram(inputs, num_local, local_offset)

    lat = {}
    lat["CuteDSL"] = bench_cute_dsl(
        inputs,
        warmup,
        iters,
        num_local,
        local_offset,
        use_cuda_graph,
        use_cupti,
        use_wrapper=use_wrapper,
        do_autotune=do_autotune,
        use_per_token_activation=use_per_token_activation,
        use_bf16_activation=use_bf16_activation,
        include_activation_quant=include_activation_quant,
        use_fused_finalize=use_fused_finalize,
        enable_pdl=enable_pdl,
        profile_cuda=profile_cuda,
        profile_iters=profile_iters,
    )
    if not cute_dsl_only and not use_per_token_activation:
        lat["CUTLASS"] = bench_cutlass(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            do_autotune=do_autotune,
        )
    if not cute_dsl_only:
        lat["TRTLLM"] = bench_trtllm(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            do_autotune=do_autotune,
            use_per_token_activation=use_per_token_activation,
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


def _print_header(
    ep_config,
    tp_config,
    num_local,
    use_cuda_graph,
    use_cupti,
    routing_bias_scale,
    use_per_token_activation=False,
    use_bf16_activation=False,
    include_activation_quant=False,
    use_fused_finalize=True,
    cute_dsl_only=False,
    enable_pdl=True,
):
    """Print benchmark header."""
    print("\n" + "=" * 120)
    if cute_dsl_only:
        print(f"DeepSeek-V3 MoE Benchmark: CuteDSL (EP={ep_config}, TP={tp_config})")
    elif use_per_token_activation:
        print(
            "DeepSeek-V3 MoE Benchmark: CuteDSL vs TRTLLM "
            f"(EP={ep_config}, TP={tp_config})"
        )
    else:
        print(
            "DeepSeek-V3 MoE Benchmark: CuteDSL vs CUTLASS vs TRTLLM "
            f"(EP={ep_config}, TP={tp_config})"
        )
    print("=" * 120)
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
    print(f"CuTe DSL activation: {'BF16' if use_bf16_activation else 'NVFP4'}")
    print(
        "Initial CuTe DSL activation quantization: "
        f"{'included' if include_activation_quant else 'excluded'}"
    )
    print(f"CuTe DSL PDL: {'enabled' if enable_pdl else 'disabled'}")
    print(
        "CuteDSL finalize: "
        f"{'atomic fused' if use_fused_finalize else 'deterministic two-stage'}"
    )
    if cute_dsl_only:
        print("CUTLASS and TRTLLM omitted by --cute-dsl-only.")
    if cute_dsl_only:
        print(f"{'Tokens':>6} | {'CuteDSL':^15} | {'Active':^7} | {'Stats':^14}")
        print(
            f"{'':>6} | {'ms':>7} {'TFLOPS':>7} | "
            f"{'experts':^7} | {'min/max/median':^14}"
        )
    elif use_per_token_activation:
        print("CUTLASS omitted: it does not consume the per-token activation scale.")
    print("-" * 120)
    if cute_dsl_only:
        return
    if use_per_token_activation:
        print(
            f"{'Tokens':>6} | "
            f"{'CuteDSL':^15} | "
            f"{'TRTLLM':^15} | "
            f"{'Speedup':^9} | "
            f"{'Winner':^8} | "
            f"{'Active':^7} | "
            f"{'Stats':^14}"
        )
        print(
            f"{'':>6} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'TRTLLM':>9} | "
            f"{'':^8} | "
            f"{'experts':^7} | "
            f"{'min/max/median':^14}"
        )
    else:
        print(
            f"{'Tokens':>6} | "
            f"{'CuteDSL':^15} | "
            f"{'CUTLASS':^15} | "
            f"{'TRTLLM':^15} | "
            f"{'Speedup (CuteDSL/X)':^18} | "
            f"{'Winner':^8} | "
            f"{'Active':^7} | "
            f"{'Stats':^14}"
        )
        print(
            f"{'':>6} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'ms':>7} {'TFLOPS':>7} | "
            f"{'CUTLASS':>9} {'TRTLLM':>9} | "
            f"{'':^8} | "
            f"{'experts':^7} | "
            f"{'min/max/median':^14}"
        )
    print("-" * 120)


def _print_row(results, histogram_record):
    """Print a single row of benchmark results."""
    # Extract values by backend
    r = {r.backend: r for r in results}
    cute = r["CuteDSL"]
    cutlass = r.get("CUTLASS")

    active_experts = f"{histogram_record['active_local_experts']:>3}"
    stats = (
        f"{histogram_record['min_count']:>3}/"
        f"{histogram_record['max_count']:>3}/"
        f"{histogram_record['median_count']:>7.2f}"
    )
    if len(r) == 1:
        print(
            f"{cute.tokens:>6} | "
            f"{cute.latency_ms:>7.3f} {cute.tflops:>7.1f} | "
            f"{active_experts:>7} | "
            f"{stats:>14}"
        )
        return

    trtllm = r["TRTLLM"]
    speedup_trtllm = trtllm.latency_ms / cute.latency_ms
    winner = min(r.values(), key=lambda x: x.latency_ms).backend
    if cutlass is None:
        print(
            f"{cute.tokens:>6} | "
            f"{cute.latency_ms:>7.3f} {cute.tflops:>7.1f} | "
            f"{trtllm.latency_ms:>7.3f} {trtllm.tflops:>7.1f} | "
            f"{speedup_trtllm:>8.2f}x | "
            f"{winner:^8} | "
            f"{active_experts:>7} | "
            f"{stats:>14}"
        )
    else:
        speedup_cutlass = cutlass.latency_ms / cute.latency_ms
        print(
            f"{cute.tokens:>6} | "
            f"{cute.latency_ms:>7.3f} {cute.tflops:>7.1f} | "
            f"{cutlass.latency_ms:>7.3f} {cutlass.tflops:>7.1f} | "
            f"{trtllm.latency_ms:>7.3f} {trtllm.tflops:>7.1f} | "
            f"{speedup_cutlass:>8.2f}x {speedup_trtllm:>8.2f}x | "
            f"{winner:^8} | "
            f"{active_experts:>7} | "
            f"{stats:>14}"
        )


def _print_footer(ep_config, num_local, cute_dsl_only=False):
    """Print benchmark footer."""
    print("-" * 120)
    if not cute_dsl_only:
        print("Speedup > 1.0 means CuteDSL is faster than that backend")


def _collect_expert_histogram(inputs, num_local, local_offset):
    from flashinfer.fused_moe import fused_topk_deepseek

    num_tokens = inputs["router_logits"].shape[0]
    dev = inputs["router_logits"].device
    topk_values = torch.empty(num_tokens, CFG.top_k, dtype=torch.float32, device=dev)
    topk_indices = torch.empty(num_tokens, CFG.top_k, dtype=torch.int32, device=dev)

    fused_topk_deepseek(
        scores=inputs["router_logits"],
        bias=inputs["routing_bias"].float(),
        n_group=CFG.n_group,
        topk_group=CFG.topk_group,
        topk=CFG.top_k,
        routed_scaling_factor=CFG.routed_scaling_factor,
        topk_values=topk_values,
        topk_indices=topk_indices,
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


def _distributed_moe_config(
    args,
    intermediate_size,
    num_local_experts,
    local_expert_offset,
    tune_max_num_tokens,
):
    from flashinfer.fused_moe import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=CFG.num_experts, top_k=CFG.top_k),
        quant=QuantConfig(
            variant=QuantVariant.NVFP4,
            per_token_scale=args.use_per_token_activation,
            quantize_input=not args.use_bf16_activation,
        ),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=num_local_experts,
        ),
        backend=BackendOptions(candidates=(CuteDslConfig(),)),
        execution=ExecutionConfig(
            enable_pdl=args.enable_pdl,
            tune_max_num_tokens=tune_max_num_tokens,
            use_fused_finalize=args.use_fused_finalize,
        ),
    )


def _create_distributed_weights(num_local_experts, intermediate_size, rank, device):
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    w13 = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            CFG.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_local_experts,
            CFG.hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    return w13, w2


def _token_partition(num_tokens, rank, world_size):
    tokens_per_rank, remainder = divmod(num_tokens, world_size)
    local_num_tokens = tokens_per_rank + int(rank < remainder)
    token_offset = rank * tokens_per_rank + min(rank, remainder)
    return local_num_tokens, token_offset


def _create_distributed_inputs(num_tokens, rank, world_size, device):
    local_num_tokens, token_offset = _token_partition(num_tokens, rank, world_size)
    input_generator = torch.Generator(device=device).manual_seed(42 + rank)
    hidden_states = (
        torch.randn(
            local_num_tokens,
            CFG.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=input_generator,
        )
        / 10
    )
    routing_generator = torch.Generator(device=device).manual_seed(137)
    global_router_logits = torch.randn(
        num_tokens,
        CFG.num_experts,
        dtype=torch.float32,
        device=device,
        generator=routing_generator,
    )
    router_logits = global_router_logits.narrow(0, token_offset, local_num_tokens)
    bias_generator = torch.Generator(device=device).manual_seed(911)
    routing_bias = (
        torch.randn(
            CFG.num_experts,
            dtype=torch.bfloat16,
            device=device,
            generator=bias_generator,
        )
        * 0.01
    ).float()
    return hidden_states, router_logits, global_router_logits, routing_bias


def _route_tokens(router_logits, routing_bias, topk_values, topk_indices):
    from flashinfer.fused_moe import fused_topk_deepseek

    if router_logits.shape[0] == 0:
        return
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


def _max_rank_sample(values, dist, device):
    sample = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(sample, op=dist.ReduceOp.MAX)
    return sample.cpu().tolist()


def _median_distributed_samples(num_tokens, samples):
    medians = np.median(np.asarray(samples), axis=0)
    return DistributedBenchResult(num_tokens, *medians.tolist())


def _run_distributed_iterations(args, run_once, l2_flush, dist, device):
    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    dist.barrier()

    if args.profile_cuda:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_iters):
            l2_flush.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            run_once()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.cudart().cudaProfilerStop()
        return None

    samples = []
    for _ in range(args.iters):
        l2_flush.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        e2e_start = torch.cuda.Event(enable_timing=True)
        e2e_end = torch.cuda.Event(enable_timing=True)
        e2e_start.record()
        run_once()
        e2e_end.record()
        e2e_end.synchronize()
        samples.append(
            _max_rank_sample([e2e_start.elapsed_time(e2e_end)], dist, device)
        )
    return samples


def _benchmark_distributed_ep(args, num_tokens, rank, world_size, device):
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        SplitConfig,
    )
    from flashinfer.testing.utils import get_l2_cache_size

    local_num_tokens, _ = _token_partition(num_tokens, rank, world_size)
    max_tokens_per_rank = (num_tokens + world_size - 1) // world_size
    num_local_experts = CFG.num_experts // world_size
    local_expert_offset = rank * num_local_experts
    moe_config = _distributed_moe_config(
        args,
        CFG.intermediate_size,
        num_local_experts,
        local_expert_offset,
        max_tokens_per_rank * world_size,
    )
    w13, w2 = _create_distributed_weights(
        num_local_experts, CFG.intermediate_size, rank, device
    )
    weights = MoEWeightPack(w13=w13, w2=w2)
    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    fleet_params = FleetParams(
        num_experts=CFG.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        token_hidden_size=CFG.hidden_size,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=EpLayout.RANK_MAJOR,
    )
    layer = MoEEpLayer(
        bootstrap,
        fleet_params,
        weights=weights,
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )
    hidden_states, router_logits, _, routing_bias = _create_distributed_inputs(
        num_tokens, rank, world_size, device
    )
    topk_values = torch.empty(
        local_num_tokens, CFG.top_k, dtype=torch.float32, device=device
    )
    topk_indices = torch.empty(
        local_num_tokens, CFG.top_k, dtype=torch.int32, device=device
    )
    tensors = MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_indices,
        topk_weights=topk_values,
    )
    l2_flush = torch.empty(2 * get_l2_cache_size(), dtype=torch.int8, device=device)

    def run_once():
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
        layer.forward(tensors)

    try:
        samples = _run_distributed_iterations(args, run_once, l2_flush, dist, device)
        if samples is None:
            return None
        return _median_distributed_samples(num_tokens, samples)
    finally:
        layer.destroy()


def _make_tp_activation_pack(
    args,
    hidden_states,
    topk_indices,
    topk_values,
    global_scale,
):
    from flashinfer.fused_moe import MoEActivationPack

    if args.use_bf16_activation:
        return MoEActivationPack(
            hidden_states_q=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_indices,
            topk_weights=topk_values,
        )

    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.fp4_quantization import fp4_quantize

    if args.use_per_token_activation:
        hidden_states_q, hidden_states_scale, per_token_scale = nvfp4_quantize(
            hidden_states,
            global_scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
    else:
        hidden_states_q, hidden_states_scale = fp4_quantize(
            hidden_states,
            global_scale,
            16,
            False,
            False,
            backend="cute-dsl",
        )
        per_token_scale = None
    if hidden_states_scale.dim() > 2:
        hidden_states_scale = hidden_states_scale.squeeze(-1)
    return MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        topk_ids=topk_indices,
        topk_weights=topk_values,
        per_token_scale=per_token_scale,
    )


def _benchmark_distributed_tp(
    args, num_tokens, rank, world_size, device, report_backend=False
):
    import torch.distributed as dist

    from flashinfer.comm import (
        AllReduceFusionPattern,
        allreduce_fusion,
        create_allreduce_fusion_workspace,
    )
    from flashinfer.comm.mnnvl import TorchDistBackend
    from flashinfer.fused_moe import CuteDslConfig, MoELayer, MoEWeightPack
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.testing.utils import get_l2_cache_size

    intermediate_size = BASE_INTERMEDIATE_SIZE // world_size
    moe_config = _distributed_moe_config(
        args,
        intermediate_size,
        CFG.num_experts,
        0,
        num_tokens,
    )
    w13, w2 = _create_distributed_weights(
        CFG.num_experts, intermediate_size, rank, device
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "cute_dsl_nvfp4",
        CuteDslConfig.prepare_weights(
            w13,
            w2,
            num_local_experts=CFG.num_experts,
            hidden_size=CFG.hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        ),
    )
    layer = MoELayer(moe_config, device=device)
    local_hidden_states, _, router_logits, routing_bias = _create_distributed_inputs(
        num_tokens, rank, world_size, device
    )
    topk_values = torch.empty(num_tokens, CFG.top_k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(num_tokens, CFG.top_k, dtype=torch.int32, device=device)
    hidden_states = torch.empty(
        num_tokens,
        CFG.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    gathered_hidden_states = list(
        hidden_states.split(
            [_token_partition(num_tokens, r, world_size)[0] for r in range(world_size)]
        )
    )
    dist.all_gather(gathered_hidden_states, local_hidden_states)
    global_scale = make_nvfp4_global_scale(
        hidden_states,
        per_token_activation=args.use_per_token_activation,
        nvfp4_4over6_config=current_nvfp4_4over6_config(),
    )
    allreduce_output = torch.empty_like(hidden_states)
    workspace = create_allreduce_fusion_workspace(
        backend=args.allreduce_backend,
        world_size=world_size,
        rank=rank,
        max_token_num=num_tokens,
        hidden_dim=CFG.hidden_size,
        dtype=torch.bfloat16,
        gpus_per_node=world_size,
        comm_backend=TorchDistBackend(),
    )
    if rank == 0 and report_backend:
        print(f"FlashInfer all-reduce backend: {workspace.backend}")
    l2_flush = torch.empty(2 * get_l2_cache_size(), dtype=torch.int8, device=device)

    def run_once():
        dist.all_gather(gathered_hidden_states, local_hidden_states)
        _route_tokens(router_logits, routing_bias, topk_values, topk_indices)
        activation_pack = _make_tp_activation_pack(
            args,
            hidden_states,
            topk_indices,
            topk_values,
            global_scale,
        )
        local_output = layer(activation_pack, weight_pack)
        allreduce_fusion(
            input=local_output,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=args.enable_pdl,
            output=allreduce_output,
        )

    try:
        samples = _run_distributed_iterations(args, run_once, l2_flush, dist, device)
        if samples is None:
            return None
        return _median_distributed_samples(num_tokens, samples)
    finally:
        workspace.destroy()


def _run_distributed_benchmark(args, token_counts):
    import gc
    import os
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    try:
        if world_size != args.num_gpus:
            raise ValueError(
                f"torchrun world size {world_size} does not match --num-gpus "
                f"{args.num_gpus}"
            )
        mode = "ep" if args.ep == world_size else "tp"
        variant = "w4a16" if args.use_bf16_activation else "w4a4"
        if rank == 0:
            print("\nDeepSeek-V3 distributed MoE benchmark")
            if args.profile_cuda:
                print(
                    f"Mode: real {mode.upper()}{world_size}, variant={variant}, "
                    "capture=cudaProfilerStart/Stop, cache=cold L2"
                )
            else:
                print(
                    f"Mode: real {mode.upper()}{world_size}, variant={variant}, "
                    "timing=max rank CUDA events, cache=cold L2"
                )
                print("global tokens | end-to-end (ms)")

        results = []
        for num_tokens in token_counts:
            if mode == "ep":
                result = _benchmark_distributed_ep(
                    args, num_tokens, rank, world_size, device
                )
            else:
                result = _benchmark_distributed_tp(
                    args,
                    num_tokens,
                    rank,
                    world_size,
                    device,
                    report_backend=not results,
                )
            if rank == 0:
                if result is None:
                    print(
                        f"Captured {args.profile_iters} iteration(s) at "
                        f"{num_tokens} global tokens"
                    )
                else:
                    print(f"{result.num_tokens:>13} | {result.e2e_ms:>10.3f}")
                    print(
                        "DISTRIBUTED_CSV,"
                        f"{mode},{variant},{num_tokens},{world_size},"
                        f"{result.e2e_ms:.6f}"
                    )
            if result is not None:
                results.append(result)
            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()
        return 0
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 MoE Performance Benchmark"
    )
    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="Comma-separated global token counts (default: 128-4096 for throughput, 1-128 for gen-phase)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--no-autotune", action="store_true", help="Disable autotune")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--gen-phase",
        action="store_true",
        help="Use generation phase token counts (1-128 instead of 128-4096)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Expert parallelism simulation, or real EP size with --num-gpus.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor Parallelism simulation: divide the expert intermediate size by TP.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPU count. One preserves simulated EP/TP; values above one require torchrun and enable real communication.",
    )
    parser.add_argument(
        "--allreduce-backend",
        choices=["auto", "trtllm", "mnnvl"],
        default="auto",
        help="FlashInfer all-reduce backend for real TP.",
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
        "--cute-dsl-only",
        action="store_true",
        help="Benchmark only the CuTe DSL backend.",
    )
    parser.add_argument(
        "--use-per-token-activation",
        action="store_true",
        help="Use per-token NVFP4 activation scaling for supported FP4 MoE backends.",
    )
    parser.add_argument(
        "--use-bf16-activation",
        action="store_true",
        help="Use BF16 activations with online NVFP4 weight dequantization in CuTe DSL.",
    )
    parser.add_argument(
        "--include-activation-quant",
        action="store_true",
        help="Include initial activation FP4 quantization in CuTe DSL timing.",
    )
    parser.add_argument(
        "--no-fused-finalize",
        action="store_false",
        dest="use_fused_finalize",
        help="Use deterministic two-stage CuTe DSL finalize instead of atomic fused finalize.",
    )
    parser.add_argument(
        "--no-pdl",
        action="store_false",
        dest="enable_pdl",
        help="Disable Programmatic Dependent Launch for CuTe DSL kernels.",
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
        "--routing-bias-scale",
        type=float,
        default=0.01,
        help="Scale for random routing bias. Larger values tend to create expert imbalance.",
    )
    args = parser.parse_args()

    if args.use_bf16_activation and args.use_per_token_activation:
        parser.error(
            "--use-bf16-activation and --use-per-token-activation are mutually exclusive"
        )
    if args.include_activation_quant and args.use_bf16_activation:
        parser.error("--include-activation-quant is incompatible with BF16 activation")
    if args.include_activation_quant and not args.cute_dsl_only:
        parser.error("--include-activation-quant requires --cute-dsl-only")
    if args.tp < 1:
        parser.error("--tp must be positive")
    if BASE_INTERMEDIATE_SIZE % args.tp != 0:
        parser.error(
            f"--tp must divide the expert intermediate size ({BASE_INTERMEDIATE_SIZE})"
        )
    if args.profile_cuda and not args.cute_dsl_only:
        parser.error("--profile-cuda requires --cute-dsl-only")
    if args.profile_iters < 1:
        parser.error("--profile-iters must be positive")
    if not 1 <= args.num_gpus <= 8:
        parser.error("--num-gpus must be between 1 and 8")
    if args.num_gpus > 1:
        real_ep = args.ep == args.num_gpus and args.tp == 1
        real_tp = args.tp == args.num_gpus and args.ep == 1
        if real_ep == real_tp:
            parser.error(
                "real multi-GPU mode requires exactly one of --ep or --tp to "
                "equal --num-gpus; combined EP x TP is not supported"
            )
        if not args.cute_dsl_only:
            parser.error("real multi-GPU mode requires --cute-dsl-only")
        if not args.use_bf16_activation and not args.include_activation_quant:
            parser.error(
                "real W4A4 mode starts from BF16 and requires "
                "--include-activation-quant"
            )

    if not is_sm100_family():
        print("ERROR: Requires SM100 family GPU (Blackwell: SM100, SM103)")
        return 1

    # Determine token counts
    if args.num_tokens:
        tokens = [int(x) for x in args.num_tokens.split(",")]
    elif args.gen_phase:
        tokens = GEN_PHASE_TOKENS  # [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        tokens = TOKEN_COUNTS  # [128, 256, 512, 1024, 2048, 4096]
    if args.profile_cuda and len(tokens) != 1:
        parser.error("--profile-cuda requires exactly one token count")
    if args.num_gpus > 1:
        return _run_distributed_benchmark(args, tokens)

    print("\nDeepSeek-V3 MoE Performance Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CuteDSL API: {'Functional' if args.functional_api else 'Wrapper'}")
    print(f"Per-token activation: {args.use_per_token_activation}")
    print(f"BF16 activation: {args.use_bf16_activation}")
    print(f"Initial activation quantization: {args.include_activation_quant}")
    print(f"Tensor parallelism simulation: TP={args.tp}")
    print(f"CUDA profiler capture: {args.profile_cuda}")
    print(f"CuTe DSL PDL: {'enabled' if args.enable_pdl else 'disabled'}")
    print(
        "CuteDSL finalize: "
        f"{'atomic fused' if args.use_fused_finalize else 'deterministic two-stage'}"
    )

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
        use_per_token_activation=args.use_per_token_activation,
        use_bf16_activation=args.use_bf16_activation,
        include_activation_quant=args.include_activation_quant,
        use_fused_finalize=args.use_fused_finalize,
        cute_dsl_only=args.cute_dsl_only,
        enable_pdl=args.enable_pdl,
        profile_cuda=args.profile_cuda,
        profile_iters=args.profile_iters,
    )

    return 0


if __name__ == "__main__":
    exit(main())
