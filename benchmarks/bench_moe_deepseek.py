#!/usr/bin/env python3
"""DeepSeek-V3 MoE Performance Benchmark - CuteDSL vs CUTLASS vs TRTLLM Gen.

Usage:
    python bench_moe_deepseek.py                       # Run perf test with default token counts
    python bench_moe_deepseek.py --num-tokens 128,256  # Custom token counts
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


def is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 10


def calc_tflops(n, ms):
    flops = (
        n
        * CFG.top_k
        * (
            2 * CFG.hidden_size * 2 * CFG.intermediate_size
            + 2 * CFG.intermediate_size * CFG.hidden_size
        )
    )
    return flops / (ms * 1e-3) / 1e12


def calc_bw(n, ms):
    bpe = 0.5 + 1 / 16
    w = (
        2 * CFG.intermediate_size * CFG.hidden_size
        + CFG.hidden_size * CFG.intermediate_size
    ) * bpe
    act = min(CFG.num_experts, CFG.top_k * n)
    return (act * w + n * CFG.hidden_size * (bpe + 2)) / (ms * 1e-3) / 1e12


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


def bench_cute_dsl(inputs, warmup=10, iters=100):
    from flashinfer.fused_moe import fused_topk_deepseek
    from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.testing.utils import bench_gpu_time

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"
    gs1 = torch.tensor([1.0], device=dev)

    tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

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

    def run():
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
        return cute_dsl_fused_moe_nvfp4(
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

    times = bench_gpu_time(
        run, dry_run_iters=warmup, repeat_iters=iters, cold_l2_cache=True
    )
    return np.median(times)


def bench_cutlass(inputs, warmup=10, iters=100):
    from flashinfer.fused_moe import fused_topk_deepseek, cutlass_fused_moe
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.testing.utils import bench_gpu_time

    n, sv, dev = inputs["router_logits"].shape[0], 16, "cuda"

    tv = torch.empty(n, CFG.top_k, dtype=torch.float32, device=dev)
    ti = torch.empty(n, CFG.top_k, dtype=torch.int32, device=dev)

    # Prepare CUTLASS inputs
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
    output = torch.empty(n, CFG.hidden_size, dtype=torch.bfloat16, device=dev)

    def run():
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
        cutlass_fused_moe(
            hidden_fp4,
            ti.to(torch.int),
            tv,
            inputs["w1_fp4"].contiguous().view(torch.long),
            inputs["w2_fp4"].contiguous().view(torch.long),
            torch.bfloat16,
            quant_scales=quant_scales,
            input_sf=input_sf,
            output=output,
        )
        return output

    times = bench_gpu_time(
        run, dry_run_iters=warmup, repeat_iters=iters, cold_l2_cache=True
    )
    return np.median(times)


def bench_trtllm(inputs, warmup=10, iters=100):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import (
        RoutingMethodType,
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.fp4_quantization import fp4_quantize, block_scale_interleave
    from flashinfer.testing.utils import bench_gpu_time

    n, dev = inputs["router_logits"].shape[0], inputs["router_logits"].device
    sv, etm, cache = 16, 128, {}

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

    w1f, w1s = prep(
        inputs["w1_bf16"], inputs["w1_gs"], 2 * CFG.intermediate_size, CFG.hidden_size
    )
    w2f, w2s = prep(
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

    w1f, w1s = shuf(w1f, w1s, _maybe_get_cached_w3_w1_permute_indices)
    w2f, w2s = shuf(w2f, w2s, get_w2_permute_indices_with_cache)
    w1s = w1s.view(torch.float8_e4m3fn).reshape(
        CFG.num_experts, 2 * CFG.intermediate_size, CFG.hidden_size // sv
    )
    w2s = w2s.view(torch.float8_e4m3fn).reshape(
        CFG.num_experts, CFG.hidden_size, CFG.intermediate_size // sv
    )

    sc = torch.ones(CFG.num_experts, device=dev, dtype=torch.float32)

    def run():
        return trtllm_fp4_block_scale_moe(
            routing_logits=inputs["router_logits"],
            routing_bias=inputs["routing_bias"],
            hidden_states=hfp,
            hidden_states_scale=hsc,
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
            local_expert_offset=0,
            local_num_experts=CFG.num_experts,
            routed_scaling_factor=CFG.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
        )

    times = bench_gpu_time(
        run, dry_run_iters=warmup, repeat_iters=iters, cold_l2_cache=True
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
    from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4
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
class PerfResult:
    backend: str
    num_tokens: int
    latency_ms: float
    tflops: float
    tb_s: float


def run_perf_test(token_counts, warmup=10, iters=100, do_autotune=True, verbose=True):
    if verbose:
        print("\n" + "=" * 120)
        print("Performance Benchmark: CuteDSL vs CUTLASS vs TRTLLM Gen (DeepSeek-V3)")
        print("=" * 120)
        print(
            f"Config: hidden={CFG.hidden_size}, inter={CFG.intermediate_size}, "
            f"experts={CFG.num_experts}, top_k={CFG.top_k}"
        )

    if do_autotune:
        run_autotune(create_inputs(max(token_counts)), verbose=verbose)

    results = []
    if verbose:
        print("-" * 120)
        print(
            f"{'Tokens':<8} | {'CuteDSL':<20} | {'CUTLASS':<20} | {'TRTLLM':<20} | "
            f"{'CuteDSL/CUTLASS':>15} {'CuteDSL/TRTLLM':>15} | {'Fastest':<10}"
        )
        print(
            f"{'':8} | {'ms':>8} {'TFLOPS':>10} | {'ms':>8} {'TFLOPS':>10} | {'ms':>8} {'TFLOPS':>10} |"
        )
        print("-" * 120)

    for n in token_counts:
        inputs = create_inputs(n)

        # Benchmark all three backends
        lat_cute = bench_cute_dsl(inputs, warmup, iters)
        lat_cutlass = bench_cutlass(inputs, warmup, iters)
        lat_trtllm = bench_trtllm(inputs, warmup, iters)

        tflops_cute = calc_tflops(n, lat_cute)
        tflops_cutlass = calc_tflops(n, lat_cutlass)
        tflops_trtllm = calc_tflops(n, lat_trtllm)

        results.append(
            PerfResult("CuteDSL", n, lat_cute, tflops_cute, calc_bw(n, lat_cute))
        )
        results.append(
            PerfResult(
                "CUTLASS", n, lat_cutlass, tflops_cutlass, calc_bw(n, lat_cutlass)
            )
        )
        results.append(
            PerfResult("TRTLLM", n, lat_trtllm, tflops_trtllm, calc_bw(n, lat_trtllm))
        )

        # Calculate speedups (CuteDSL / others, > 1 means CuteDSL is faster)
        speedup_vs_cutlass = lat_cutlass / lat_cute
        speedup_vs_trtllm = lat_trtllm / lat_cute

        # Determine fastest backend
        latencies = {"CuteDSL": lat_cute, "CUTLASS": lat_cutlass, "TRTLLM": lat_trtllm}
        fastest = min(latencies, key=latencies.get)

        if verbose:
            print(
                f"{n:<8} | {lat_cute:>8.3f} {tflops_cute:>10.1f} | "
                f"{lat_cutlass:>8.3f} {tflops_cutlass:>10.1f} | "
                f"{lat_trtllm:>8.3f} {tflops_trtllm:>10.1f} | "
                f"{speedup_vs_cutlass:>14.2f}x {speedup_vs_trtllm:>14.2f}x | {fastest:<10}"
            )

    if verbose:
        print("-" * 120)
        print("Note: Speedup > 1.0 means CuteDSL is faster")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 MoE Performance Benchmark"
    )
    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="Comma-separated token counts (e.g., 128,256,512)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--no-autotune", action="store_true", help="Disable autotune")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not is_blackwell():
        print("ERROR: Requires Blackwell GPU (SM100+)")
        return 1

    tokens = (
        [int(x) for x in args.num_tokens.split(",")]
        if args.num_tokens
        else TOKEN_COUNTS
    )
    verbose = not args.quiet

    print("\nDeepSeek-V3 MoE Performance Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_perf_test(tokens, args.warmup, args.iters, not args.no_autotune, verbose)

    return 0


if __name__ == "__main__":
    exit(main())
