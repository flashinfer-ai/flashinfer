import argparse
import torch
import numpy as np
from flashinfer import (
    fp4_quantize,
    mxfp8_quantize,
    next_positive_power_of_2,
    RoutingMethodType,
    shuffle_matrix_a,
    reorder_rows_for_gated_act_gemm,
)
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    WeightLayout,
)
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
    quant_mode,
    num_tokens,
    num_experts,
    hidden_size,
    intermediate_size,
    top_k,
    warmups,
    iterations,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    # Adjust parameters for specific routing methods
    if quant_mode == "FP8_per_tensor_scale":
        # Llama4 routing requires top_k=1
        top_k = 1
    # FP8_block_scale (DeepSeekV3) requires float, FP8_per_tensor_scale (Llama4) uses bfloat16
    if quant_mode == "FP8_block_scale":
        routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
            torch.float
        )
    else:
        routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
            torch.bfloat16
        )
    hidden_states = torch.randn(num_tokens, hidden_size, device=device).to(
        torch.bfloat16
    )

    # Create routing bias for FP8 modes that use it
    if quant_mode in ["FP8_block_scale", "FP8_per_tensor_scale"]:
        routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)
    else:
        routing_bias = None
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
    elif quant_mode == "MxFP4xBf16":
        hidden_states_scale = None
        hidden_states_global_scale = 1.0
    elif quant_mode == "FP8_block_scale":
        # FP8 block scale: no pre-quantization of hidden states needed
        hidden_states_scale = None
        hidden_states_global_scale = None
    elif quant_mode == "FP8_per_tensor_scale":
        # FP8 per-tensor: quantize hidden states with global scale
        hidden_states_scale_factor = 448.0 / hidden_states.float().abs().max()
        hidden_states = (hidden_states * hidden_states_scale_factor).to(
            torch.float8_e4m3fn
        )
        hidden_states_scale = None
        hidden_states_global_scale = hidden_states_scale_factor
    else:
        raise ValueError(f"Invalid quantization mode: {quant_mode}")

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
    elif quant_mode == "MxFP4xBf16" or quant_mode == "MxFP4xMxFP8":
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
    elif quant_mode == "FP8_block_scale":
        # FP8 block scale: quantize weights to FP8 with block scales
        w13 = w13.to(torch.float8_e4m3fn)
        w13_scale = 2 * torch.rand(
            (num_experts, intermediate_size * 2 // 128, hidden_size // 128),
            device=device,
        ).to(torch.float)
        w2 = w2.to(torch.float8_e4m3fn)
        w2_scale = 2 * torch.rand(
            (num_experts, hidden_size // 128, intermediate_size // 128), device=device
        ).to(torch.float)
        w13_global_scale = None
        w2_global_scale = None
    elif quant_mode == "FP8_per_tensor_scale":
        # FP8 per-tensor: quantize weights to FP8 with global scales
        w13_scale_factor = 448.0 / w13.float().abs().max()
        w13 = (w13 * w13_scale_factor).to(torch.float8_e4m3fn)
        w13_scale = None
        w13_global_scale = w13_scale_factor

        w2_scale_factor = 448.0 / w2.float().abs().max()
        w2 = (w2 * w2_scale_factor).to(torch.float8_e4m3fn)
        w2_scale = None
        w2_global_scale = w2_scale_factor
    else:
        raise ValueError(f"Invalid quantization mode: {quant_mode}")

    bias13 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10
    bias2 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10

    tile_tokens_dim = get_tile_tokens_dim(num_tokens, num_experts, top_k)

    # Handle scaling factors for different quantization modes
    if quant_mode in ["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"]:
        output1_scale_scalar = torch.tensor(
            [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
        )
        output1_scale_gate_scalar = torch.tensor(
            [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
        )
        output2_scale_scalar = torch.tensor(
            [hidden_states_global_scale * w2_global_scale] * num_experts, device=device
        )
    elif quant_mode == "FP8_per_tensor_scale":
        # FP8 per-tensor uses global scale factors
        output1_scale_scalar = torch.tensor(
            [1.0 / w13_global_scale / hidden_states_global_scale] * num_experts,
            device=device,
        )
        output1_scale_gate_scalar = torch.tensor(
            [1.0 / w13_global_scale / hidden_states_global_scale] * num_experts,
            device=device,
        )
        output2_scale_scalar = torch.tensor(
            [1.0 / w2_global_scale] * num_experts, device=device
        )
    else:
        # FP8 block scale doesn't use these scaling factors
        output1_scale_scalar = None
        output1_scale_gate_scalar = None
        output2_scale_scalar = None
    if quant_mode in ["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"]:
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
            1,
            True,
            enable_pdl,
        )
    elif quant_mode == "FP8_block_scale":
        # Prepare weights for FP8 block scale kernel
        epilogue_tile_m = 64
        w13_shuffled = []
        w2_shuffled = []
        for i in range(num_experts):
            w13_shuffled.append(
                shuffle_matrix_a(w13[i].view(torch.uint8), epilogue_tile_m)
            )
            w2_shuffled.append(
                shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m)
            )
        w13_kernel = torch.stack(w13_shuffled).view(torch.float8_e4m3fn)
        w2_kernel = torch.stack(w2_shuffled).view(torch.float8_e4m3fn)

        fn = lambda: trtllm_fp8_block_scale_moe(
            routing_logits,
            routing_bias,  # routing_bias for DeepSeekV3
            hidden_states.to(torch.float8_e4m3fn),
            2.0
            * torch.ones(
                (hidden_size // 128, num_tokens), device=device, dtype=torch.float
            ),
            w13_kernel,
            w13_scale,
            w2_kernel,
            w2_scale,
            num_experts,
            top_k,
            8,  # n_group for DeepSeekV3
            4,  # topk_group for DeepSeekV3
            intermediate_size,
            0,  # local_expert_offset
            num_experts,
            2.5,  # routed_scaling_factor for DeepSeekV3
            tile_tokens_dim,
            RoutingMethodType.DeepSeekV3,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=1024,
        )
    elif quant_mode == "FP8_per_tensor_scale":
        # Prepare weights for FP8 per-tensor kernel
        epilogue_tile_m = 128

        # Reorder rows of W1 for fused gated activation
        w13_interleaved = []
        for i in range(num_experts):
            w13_interleaved.append(reorder_rows_for_gated_act_gemm(w13[i].clone()))
        w13_interleaved = torch.stack(w13_interleaved)

        # Shuffle weights for transposed mma output
        w13_shuffled = []
        w2_shuffled = []
        for i in range(num_experts):
            w13_shuffled.append(
                shuffle_matrix_a(w13_interleaved[i].view(torch.uint8), epilogue_tile_m)
            )
            w2_shuffled.append(
                shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m)
            )
        w13_kernel = torch.stack(w13_shuffled).view(torch.float8_e4m3fn)
        w2_kernel = torch.stack(w2_shuffled).view(torch.float8_e4m3fn)

        fn = lambda: trtllm_fp8_per_tensor_scale_moe(
            routing_logits,
            routing_bias,  # routing_bias for Llama4
            hidden_states,
            w13_kernel,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            w2_kernel,
            output2_scale_scalar,
            num_experts,
            top_k,  # top_k (adjusted to 1 for Llama4)
            0,  # n_group for Llama4
            0,  # topk_group for Llama4
            intermediate_size,
            0,  # local_expert_offset
            num_experts,
            2.5,  # routed_scaling_factor for Llama4
            True,  # use_routing_scales_on_input for Llama4
            tile_tokens_dim,
            RoutingMethodType.Llama4,
            tune_max_num_tokens=1024,
        )
    else:
        raise ValueError(f"Invalid quantization mode: {quant_mode}")

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
        choices=[
            "NvFP4xNvFP4",
            "MxFP4xMxFP8",
            "MxFP4xBf16",
            "FP8_block_scale",
            "FP8_per_tensor_scale",
        ],
        help="Quantization mode",
    )
    parser.add_argument("--num-tokens", type=int, default=512, help="Number of tokens")
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
        args.quant_mode,
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.warmups,
        args.iterations,
    )
