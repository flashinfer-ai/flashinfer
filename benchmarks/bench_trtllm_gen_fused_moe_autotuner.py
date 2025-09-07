import argparse
from typing import Optional, Literal
import torch
import numpy as np
from flashinfer import (
    RoutingMethodType,
    GatedActType,
    fp4_quantize,
    mxfp8_quantize,
    next_positive_power_of_2,
)
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.autotuner import autotune
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import device_support_pdl


def get_tile_tokens_dim(num_tokens, num_experts, top_k):
    # Factor to account for the imbalance of the experts.
    # factor equals to the
    # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
    # - 1.0 means perfect expert distribution.
    # - > 1.0 means some experts have more
    #     tokens than the perfect distribution.
    # - < 1.0 does not make sense.
    imbalance_factor = 1.3
    # Calculate the number of tokens per expert
    # assuming perfect distribution.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # Apply the imbalance factor.
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile
    # as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


def bench_trtllm_gen_fused_moe_autotuner(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = torch.randn(num_tokens, hidden_size, device=device).to(
        torch.bfloat16
    )
    if quant_mode == "NvFP4xNvFP4":
        hidden_states, hidden_states_scale = fp4_quantize(
            hidden_states,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0 / 448.0 / 6.0
    elif quant_mode == "MxFP4xMxFP8":
        hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0
    else:  # MxFP4xBf16
        hidden_states_scale = None
        hidden_states_global_scale = 1.0

    w13 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.bfloat16
    )
    if quant_mode == "NvFP4xNvFP4":
        w13, w13_scale = fp4_quantize(
            w13,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0 / 448.0 / 6.0
        w2_global_scale = 1.0 / 448.0 / 6.0
    else:
        w13, w13_scale = fp4_quantize(
            w13, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0
        w2_global_scale = 1.0
    bias13 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10
    bias2 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10

    tile_tokens_dim = get_tile_tokens_dim(num_tokens, num_experts, top_k)
    output1_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output1_scale_gate_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output2_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w2_global_scale] * num_experts, device=device
    )
    fn = lambda: trtllm_fp4_block_scale_moe(
        routing_logits,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        bias13,
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2,
        w2_scale,
        bias2,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        tile_tokens_dim,
        RoutingMethodType.Renormalize.value[0],
        True,
        enable_pdl,
        GatedActType.SwiGlu.value,  # gated_act_type
        None,
        num_tokens if tune_max_num_tokens is None else tune_max_num_tokens,
    )

    def bench(do_autotune):
        # warmup
        with autotune(do_autotune):
            for _ in range(warmups):
                fn()
        ms_list = bench_gpu_time(
            fn,
            repeat_iters=iterations,
        )
        median_ms = np.median(ms_list)
        return median_ms

    ms = bench(do_autotune=False)
    ms_tuned = bench(do_autotune=True)
    print(
        f"num tokens: {num_tokens}, num experts: {num_experts}, hidden size: {hidden_size}, intermediate size: {intermediate_size}, top k: {top_k}"
    )
    print(f"No autotune: {ms:.3f} ms; with autotune: {ms_tuned:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="MxFP4xMxFP8",
        choices=["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
        help="Quantization mode",
    )
    parser.add_argument("--num-tokens", type=int, default=512, help="Number of tokens")
    parser.add_argument(
        "--tune-max-num-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens for tunning",
    )
    parser.add_argument(
        "--num-experts", type=int, default=128, help="Number of experts"
    )
    parser.add_argument("--hidden-size", type=int, default=3072, help="Hidden size")
    parser.add_argument(
        "--intermediate-size", type=int, default=3072, help="Intermediate size"
    )
    parser.add_argument("--top-k", type=int, default=4, help="Top-k experts per token")
    parser.add_argument(
        "--warmups", type=int, default=100, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    args = parser.parse_args()
    bench_trtllm_gen_fused_moe_autotuner(
        args.tune_max_num_tokens,
        args.quant_mode,
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.warmups,
        args.iterations,
    )
