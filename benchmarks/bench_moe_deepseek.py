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

    # Custom token counts
    python bench_moe_deepseek.py --num-tokens 64,128,256

    # Disable CUDA graph (useful for debugging or profiling)
    python bench_moe_deepseek.py --no-cuda-graph

    # Disable CUPTI (use CUDA events for timing instead)
    python bench_moe_deepseek.py --no-cupti

Metrics:
    - ms: Latency in milliseconds
    - TFLOPS: Computational throughput
    - Speedup: CuteDSL latency / other backend latency (>1 = CuteDSL faster)
"""

import argparse
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
TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096]

# Generation phase token counts (small batches typical in decode)
GEN_PHASE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

# Expert Parallelism configurations
# EP=1: all 256 experts on single GPU
# EP=8: 32 experts per GPU (256/8)
# EP=16: 16 experts per GPU (256/16)
EP_CONFIGS = {
    1: {"num_local_experts": 256, "local_expert_offset": 0},
    8: {"num_local_experts": 32, "local_expert_offset": 0},
    16: {"num_local_experts": 16, "local_expert_offset": 0},
}


def is_sm100_family():
    """Check for SM100 family (Blackwell: SM100, SM103, SM110).

    CuteDSL MoE NVFP4 kernels are optimized for SM100 architecture.
    SM120+ (Rubin) may have different shared memory/TMEM configurations.
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


def create_inputs(n, dev="cuda"):
    """Create inputs for all backends (CuteDSL, CUTLASS, TRTLLM)."""
    from flashinfer.fp4_quantization import fp4_quantize

    torch.manual_seed(42)
    sv = 16
    FP8M = torch.finfo(torch.float8_e4m3fn).max
    FP4M = 6.0

    # Router logits and bias
    rl = torch.randn(n, CFG.num_experts, device=dev, dtype=torch.float32)
    rb = torch.randn(CFG.num_experts, device=dev, dtype=torch.bfloat16)

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
):
    """Benchmark CuteDSL MoE.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API (recommended for CUDA graph).
                    If False, use cute_dsl_fused_moe_nvfp4 functional API.
    """
    from flashinfer.fused_moe import fused_topk_deepseek
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"
    gs1 = torch.tensor([1.0], device=dev)

    tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

    xf, xs = fp4_quantize(inputs["hidden_bf16"], gs1, sv, False, False)
    xs = xs.unsqueeze(-1)

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
        )

        def run(x, x_sf, router_logits, routing_bias, topk_values, topk_indices):
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
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
            )
    else:
        # Use functional API
        from flashinfer import cute_dsl_fused_moe_nvfp4

        def run(x, x_sf, router_logits, routing_bias, topk_values, topk_indices):
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
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                num_experts=CFG.num_experts,
                top_k=CFG.top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
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
):
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
):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import (
        RoutingMethodType,
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
    from flashinfer.testing.utils import bench_gpu_time

    if num_local_experts is None:
        num_local_experts = CFG.num_experts

    n, dev = inputs["router_logits"].shape[0], inputs["router_logits"].device
    sv, etm, cache = 16, 128, {}

    # Expert range for this EP partition
    expert_start = local_expert_offset
    expert_end = local_expert_offset + num_local_experts

    hg = inputs["hidden_gs"]
    hfp, hsf = fp4_quantize(inputs["hidden_bf16"], hg, sv, False, True)
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
            do_finalize=True,
        )

    input_kwargs = {
        "routing_logits": inputs["router_logits"],
        "routing_bias": inputs["routing_bias"],
        "hidden_states": hfp,
        "hidden_states_scale": hsc,
    }

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


def run_autotune(inputs, verbose=True):
    from flashinfer.fused_moe import (
        fused_topk_deepseek,
        cutlass_fused_moe,
        trtllm_fp4_block_scale_moe,
    )
    from flashinfer.fused_moe.core import (
        RoutingMethodType,
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer import cute_dsl_fused_moe_nvfp4
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
    from flashinfer.autotuner import autotune

    if verbose:
        print("\nRunning autotune warmup for all backends...")
        print("-" * 80)

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"
    gs1 = torch.tensor([1.0], device=dev)

    tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)
    fused_topk_deepseek(
        scores=inputs["router_logits"],
        bias=inputs["routing_bias"].float(),
        n_group=CFG.n_group,
        topk_group=CFG.topk_group,
        topk=CFG.top_k,
        routed_scaling_factor=CFG.routed_scaling_factor,
        topk_values=tv,
        topk_indices=ti,
    )

    # -------------------------------------------------------------------------
    # CuteDSL autotune
    # -------------------------------------------------------------------------
    if verbose:
        print("Autotuning CuteDSL...")

    xf, xs = fp4_quantize(inputs["hidden_bf16"], gs1, sv, False, False)
    xs = xs.unsqueeze(-1)

    w1i = interleave(inputs["w1_bf16"], 64)
    w1f = w1i.view(CFG.num_experts * 2 * CFG.intermediate_size, CFG.hidden_size)
    w1q, w1s = fp4_quantize(w1f, gs1, sv, False, True)
    w1q = w1q.view(CFG.num_experts, 2 * CFG.intermediate_size, CFG.hidden_size // 2)
    w1s = convert_sf_to_mma_layout(
        w1s, 2 * CFG.intermediate_size, CFG.hidden_size, CFG.num_experts, sv
    )

    w2f = inputs["w2_bf16"].view(
        CFG.num_experts * CFG.hidden_size, CFG.intermediate_size
    )
    w2q, w2s = fp4_quantize(w2f, gs1, sv, False, True)
    w2q = w2q.view(CFG.num_experts, CFG.hidden_size, CFG.intermediate_size // 2)
    w2s = convert_sf_to_mma_layout(
        w2s, CFG.hidden_size, CFG.intermediate_size, CFG.num_experts, sv
    )

    alpha, fc2sc = (
        torch.ones(CFG.num_experts, device=dev),
        torch.tensor([1.0], device=dev),
    )

    with autotune(True):
        for _ in range(10):
            cute_dsl_fused_moe_nvfp4(
                x=xf,
                x_sf=xs,
                token_selected_experts=ti,
                token_final_scales=tv,
                w1_weight=w1q,
                w1_weight_sf=w1s,
                w1_alpha=alpha,
                fc2_input_scale=fc2sc,
                w2_weight=w2q,
                w2_weight_sf=w2s,
                w2_alpha=alpha,
                num_experts=CFG.num_experts,
                top_k=CFG.top_k,
                num_local_experts=CFG.num_experts,
                local_expert_offset=0,
            )
    torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # CUTLASS autotune
    # -------------------------------------------------------------------------
    if verbose:
        print("Autotuning CUTLASS...")

    a1_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)
    quant_scales = [
        a1_gs,
        inputs["w1_sf"].view(torch.int32),
        1.0 / (a1_gs * inputs["w1_gs"]),
        a2_gs,
        inputs["w2_sf"].view(torch.int32),
        1.0 / (a2_gs * inputs["w2_gs"]),
    ]
    hidden_fp4, input_sf = fp4_quantize(inputs["hidden_bf16"], a1_gs, sv, False, True)
    output_cutlass = torch.empty(n, CFG.hidden_size, dtype=torch.bfloat16, device=dev)

    with autotune(True):
        for _ in range(10):
            cutlass_fused_moe(
                hidden_fp4,
                ti.to(torch.int),
                tv,
                inputs["w1_fp4"].contiguous().view(torch.long),
                inputs["w2_fp4"].contiguous().view(torch.long),
                torch.bfloat16,
                quant_scales=quant_scales,
                input_sf=input_sf,
                output=output_cutlass,
            )
    torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # TRTLLM Gen autotune
    # -------------------------------------------------------------------------
    if verbose:
        print("Autotuning TRTLLM Gen...")

    etm, cache = 128, {}
    hg = inputs["hidden_gs"]
    hfp, hsf = fp4_quantize(inputs["hidden_bf16"], hg, sv, False, True)
    hfp = hfp.view(torch.uint8).reshape(n, CFG.hidden_size // 2)
    hsc = (
        hsf.view(torch.float8_e4m3fn)
        .flatten()[: n * CFG.hidden_size // sv]
        .reshape(n, CFG.hidden_size // sv)
    )

    def prep(bf16, gs, M, K):
        fl, sl = [], []
        for e in range(CFG.num_experts):
            q, s = fp4_quantize(bf16[e], gs[e], sv, False, False)
            fl.append(q.view(torch.uint8).reshape(M, K // 2))
            sl.append(s.view(torch.float8_e4m3fn).reshape(M, K // sv))
        return torch.stack(fl), torch.stack(sl)

    w1f_trt, w1s_trt = prep(
        inputs["w1_bf16"], inputs["w1_gs"], 2 * CFG.intermediate_size, CFG.hidden_size
    )
    w2f_trt, w2s_trt = prep(
        inputs["w2_bf16"], inputs["w2_gs"], CFG.hidden_size, CFG.intermediate_size
    )

    def shuf(fp4, sf, perm_fn):
        fsh, ssh = [], []
        for i in range(CFG.num_experts):
            p = perm_fn(cache, fp4[i], etm)
            fsh.append(fp4[i][p.to(dev)].contiguous())
            ps = perm_fn(cache, sf[i].view(torch.uint8), etm, sv)
            ssh.append(
                block_scale_interleave(sf[i].view(torch.uint8)[ps.to(dev)].contiguous())
            )
        return torch.stack(fsh), torch.stack(ssh)

    w1f_trt, w1s_trt = shuf(w1f_trt, w1s_trt, _maybe_get_cached_w3_w1_permute_indices)
    w2f_trt, w2s_trt = shuf(w2f_trt, w2s_trt, get_w2_permute_indices_with_cache)
    w1s_trt = w1s_trt.view(torch.float8_e4m3fn).reshape(
        CFG.num_experts, 2 * CFG.intermediate_size, CFG.hidden_size // sv
    )
    w2s_trt = w2s_trt.view(torch.float8_e4m3fn).reshape(
        CFG.num_experts, CFG.hidden_size, CFG.intermediate_size // sv
    )

    sc = torch.ones(CFG.num_experts, device=dev, dtype=torch.float32)

    with autotune(True):
        for _ in range(10):
            trtllm_fp4_block_scale_moe(
                routing_logits=inputs["router_logits"],
                routing_bias=inputs["routing_bias"],
                hidden_states=hfp,
                hidden_states_scale=hsc,
                gemm1_weights=w1f_trt,
                gemm1_weights_scale=w1s_trt,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=w2f_trt,
                gemm2_weights_scale=w2s_trt,
                gemm2_bias=None,
                output1_scale_scalar=sc,
                output1_scale_gate_scalar=sc,
                output2_scale_scalar=sc,
                num_experts=CFG.num_experts,
                top_k=CFG.top_k,
                n_group=CFG.n_group,
                topk_group=CFG.topk_group,
                intermediate_size=CFG.intermediate_size,
                local_expert_offset=0,
                local_num_experts=CFG.num_experts,
                routed_scaling_factor=CFG.routed_scaling_factor,
                routing_method_type=RoutingMethodType.DeepSeekV3,
                do_finalize=True,
            )
    torch.cuda.synchronize()

    if verbose:
        print("-" * 80)
        print("Autotune complete for all backends.\n")


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
    do_autotune=True,
    verbose=True,
    use_cuda_graph=True,
    use_cupti=True,
    use_wrapper=True,
):
    """
    Unified benchmark for DeepSeek-V3 MoE backends.

    Args:
        token_counts: List of token counts to benchmark
        warmup: Warmup iterations
        iters: Benchmark iterations
        ep_config: Expert Parallelism config (1, 8, or 16)
        do_autotune: Whether to run autotune before benchmarking
        verbose: Print results to stdout
        use_cuda_graph: Whether to use CUDA graph for benchmarking
        use_cupti: Whether to use CUPTI for accurate GPU timing
        use_wrapper: Whether to use CuteDslMoEWrapper API (recommended)

    Returns:
        List of BenchResult objects
    """
    # Get EP configuration
    ep_cfg = EP_CONFIGS.get(ep_config, EP_CONFIGS[1])
    num_local = ep_cfg["num_local_experts"]
    local_offset = ep_cfg["local_expert_offset"]

    # Run autotune if requested (BEFORE printing header to avoid interleaved output)
    if do_autotune:
        run_autotune(create_inputs(max(token_counts)), verbose=verbose)

    # Print header AFTER autotune completes
    if verbose:
        _print_header(ep_config, num_local, use_cuda_graph, use_cupti)

    # Run benchmarks
    results = []
    for n in token_counts:
        row = _benchmark_single(
            n,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            use_wrapper=use_wrapper,
        )
        results.extend(row)
        if verbose:
            _print_row(row)

    # Print footer
    if verbose:
        _print_footer(ep_config, num_local)

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
):
    """Benchmark all backends for a single token count.

    Args:
        use_wrapper: If True, use CuteDslMoEWrapper API for CuteDSL.
    """
    inputs = create_inputs(n)

    # Run all three backends
    lat = {
        "CuteDSL": bench_cute_dsl(
            inputs,
            warmup,
            iters,
            num_local,
            local_offset,
            use_cuda_graph,
            use_cupti,
            use_wrapper=use_wrapper,
        ),
        "CUTLASS": bench_cutlass(
            inputs, warmup, iters, num_local, local_offset, use_cuda_graph, use_cupti
        ),
        "TRTLLM": bench_trtllm(
            inputs, warmup, iters, num_local, local_offset, use_cuda_graph, use_cupti
        ),
    }

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
    return results


def _print_header(ep_config, num_local, use_cuda_graph, use_cupti):
    """Print benchmark header."""
    print("\n" + "=" * 100)
    print(f"DeepSeek-V3 MoE Benchmark: CuteDSL vs CUTLASS vs TRTLLM (EP={ep_config})")
    print("=" * 100)
    print(
        f"Model: hidden={CFG.hidden_size}, intermediate={CFG.intermediate_size}, "
        f"experts={CFG.num_experts}, top_k={CFG.top_k}"
    )
    print(
        f"EP Config: {num_local} local experts (simulating {CFG.num_experts // num_local}-way parallelism)"
    )
    print(
        f"CUDA Graph: {'enabled' if use_cuda_graph else 'disabled'}, CUPTI: {'enabled' if use_cupti else 'disabled'}"
    )
    print("-" * 100)
    print(
        f"{'Tokens':>6} | "
        f"{'CuteDSL':^15} | "
        f"{'CUTLASS':^15} | "
        f"{'TRTLLM':^15} | "
        f"{'Speedup (CuteDSL/X)':^18} | "
        f"{'Winner':^8}"
    )
    print(
        f"{'':>6} | "
        f"{'ms':>7} {'TFLOPS':>7} | "
        f"{'ms':>7} {'TFLOPS':>7} | "
        f"{'ms':>7} {'TFLOPS':>7} | "
        f"{'CUTLASS':>8} {'TRTLLM':>8} |"
    )
    print("-" * 100)


def _print_row(results):
    """Print a single row of benchmark results."""
    # Extract values by backend
    r = {r.backend: r for r in results}
    cute, cutlass, trtllm = r["CuteDSL"], r["CUTLASS"], r["TRTLLM"]

    # Calculate speedups (> 1.0 means CuteDSL is faster)
    speedup_cutlass = cutlass.latency_ms / cute.latency_ms
    speedup_trtllm = trtllm.latency_ms / cute.latency_ms

    # Find winner
    winner = min(r.values(), key=lambda x: x.latency_ms).backend

    print(
        f"{cute.tokens:>6} | "
        f"{cute.latency_ms:>7.3f} {cute.tflops:>7.1f} | "
        f"{cutlass.latency_ms:>7.3f} {cutlass.tflops:>7.1f} | "
        f"{trtllm.latency_ms:>7.3f} {trtllm.tflops:>7.1f} | "
        f"{speedup_cutlass:>7.2f}x {speedup_trtllm:>7.2f}x | "
        f"{winner:^8}"
    )


def _print_footer(ep_config, num_local):
    """Print benchmark footer."""
    print("-" * 100)
    print("Speedup > 1.0 means CuteDSL is faster than that backend")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 MoE Performance Benchmark"
    )
    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="Comma-separated token counts (default: 128-4096 for throughput, 1-128 for gen-phase)",
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
        choices=[1, 8, 16],
        help="Expert Parallelism: 1 (256 local), 8 (32 local), 16 (16 local)",
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
    args = parser.parse_args()

    if not is_sm100_family():
        print("ERROR: Requires SM100 family GPU (Blackwell: SM100, SM103, SM110)")
        return 1

    # Determine token counts
    if args.num_tokens:
        tokens = [int(x) for x in args.num_tokens.split(",")]
    elif args.gen_phase:
        tokens = GEN_PHASE_TOKENS  # [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        tokens = TOKEN_COUNTS  # [128, 256, 512, 1024, 2048, 4096]

    print("\nDeepSeek-V3 MoE Performance Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CuteDSL API: {'Functional' if args.functional_api else 'Wrapper'}")

    run_benchmark(
        token_counts=tokens,
        warmup=args.warmup,
        iters=args.iters,
        ep_config=args.ep,
        do_autotune=not args.no_autotune,
        verbose=not args.quiet,
        use_cuda_graph=not args.no_cuda_graph,
        use_cupti=not args.no_cupti,
        use_wrapper=not args.functional_api,
    )

    return 0


if __name__ == "__main__":
    exit(main())
