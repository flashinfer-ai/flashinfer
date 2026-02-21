import argparse
from typing import Optional, Literal
import torch
import numpy as np
from functools import partial
from flashinfer import (
    RoutingMethodType,
    ActivationType,
    fp4_quantize,
    mxfp8_quantize,
)
from flashinfer.fused_moe import (
    Fp8QuantizationType,
    trtllm_fp4_block_scale_moe,
    trtllm_mxint4_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_fp8_block_scale_moe,
    WeightLayout,
)
from flashinfer.autotuner import autotune
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import device_support_pdl
from routines.flashinfer_benchmark_utils import enum_type

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0


def fp8_quantize(x) -> tuple[torch.Tensor, torch.Tensor]:
    max = x.abs().max().float()
    scale = FLOAT8_E4M3_MAX / max
    x = (x * scale).to(torch.float8_e4m3fn)
    return x, 1.0 / scale


def mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].to(torch.float32)
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].to(torch.float32)
    x_max = x_max * 8.0 / 7.0
    amax = torch.where(x_max > -x_min, x_max, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    )
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return x_int4.reshape(*x.shape[:-1], x.shape[-1] // 2).view(
        torch.uint8
    ), scales.reshape(-1, sf_vec_size)


def bench_trtllm_gen_fused_moe_autotuner_fp8(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["Fp8-Per-Tensor", "Fp8-Block", "MxFP8xMxFP8"],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
    activation_type: int,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.float32
    )
    hidden_states = torch.randn(num_tokens, hidden_size, device=device).to(
        torch.bfloat16
    )
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)
    w13 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.bfloat16
    )

    is_block_scale = quant_mode != "Fp8-Per-Tensor"
    if quant_mode == "Fp8-Per-Tensor":
        hidden_states, hidden_states_scale = fp8_quantize(hidden_states)
        w13, w13_scale = fp8_quantize(w13)
        w2, w2_scale = fp8_quantize(w2)
    else:
        scale_vec_size = 128 if quant_mode == "Fp8-Block" else 32
        if quant_mode == "Fp8-Block":
            # block scale quantization is too slow, so we use per-tensor quantization for now
            hidden_states, hidden_states_scale = fp8_quantize(
                hidden_states
            )  # scalar quantization
            w13, w13_scale = fp8_quantize(w13)  # scalar quantization
            w2, w2_scale = fp8_quantize(w2)  # scalar quantization
            hidden_states_scale = torch.full(
                (hidden_size // scale_vec_size, num_tokens),
                hidden_states_scale.item(),
                device=device,
            )
            w13_scale = torch.full(
                (
                    num_experts,
                    intermediate_size * 2 // scale_vec_size,
                    hidden_size // scale_vec_size,
                ),
                w13_scale.item(),
                device=device,
            )
            w2_scale = torch.full(
                (
                    num_experts,
                    hidden_size // scale_vec_size,
                    intermediate_size // scale_vec_size,
                ),
                w2_scale.item(),
                device=device,
            )
        else:  # MxFP8xMxFP8
            hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
            w13, w13_scale = mxfp8_quantize(w13, True)
            w2, w2_scale = mxfp8_quantize(w2, True)
            hidden_states_scale = hidden_states_scale.view(torch.uint8).reshape(
                num_tokens, -1
            )
            w13_scale = w13_scale.view(torch.uint8).reshape(
                num_experts, intermediate_size * 2, -1
            )
            w2_scale = w2_scale.view(torch.uint8).reshape(num_experts, hidden_size, -1)

    output1_scale_scalar = (
        torch.tensor([hidden_states_scale * w13_scale] * num_experts, device=device)
        if not is_block_scale
        else None
    )
    output1_scales_gate_scalar = (
        torch.ones(num_experts, device=device, dtype=torch.float32)
        if not is_block_scale
        else None
    )
    output2_scale_scalar = (
        torch.tensor([hidden_states_scale * w2_scale] * num_experts, device=device)
        if not is_block_scale
        else None
    )

    if is_block_scale:
        assert activation_type == ActivationType.Swiglu.value, (
            "Only Swiglu activation is supported for FP8 block scale MoE."
        )
        fn = partial(
            trtllm_fp8_block_scale_moe,
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            num_experts=num_experts,
            top_k=top_k,
            n_group=8,
            topk_group=4,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=2.5,
            routing_method_type=RoutingMethodType.DeepSeekV3.value,
            use_shuffled_weight=quant_mode == "MxFP8xMxFP8",
            weight_layout=WeightLayout.MajorK.value,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=num_tokens
            if tune_max_num_tokens is None
            else tune_max_num_tokens,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8
            if quant_mode == "Fp8-Block"
            else Fp8QuantizationType.MxFp8,
        )
    else:
        fn = partial(
            trtllm_fp8_per_tensor_scale_moe,
            routing_logits=routing_logits.to(torch.bfloat16),
            routing_bias=None,
            output1_scales_scalar=output1_scale_scalar,
            output1_scales_gate_scalar=output1_scales_gate_scalar,
            output2_scales_scalar=output2_scale_scalar,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
            routing_method_type=RoutingMethodType.TopK.value,
            enable_pdl=enable_pdl,
            tune_max_num_tokens=num_tokens
            if tune_max_num_tokens is None
            else tune_max_num_tokens,
            activation_type=activation_type,
        )
    input_kwargs = {
        "hidden_states": hidden_states,
        "gemm1_weights": w13,
        "gemm2_weights": w2,
    }
    if is_block_scale:
        input_kwargs["hidden_states_scale"] = hidden_states_scale
        input_kwargs["gemm1_weights_scale"] = w13_scale
        input_kwargs["gemm2_weights_scale"] = w2_scale

    def bench(do_autotune):
        with autotune(do_autotune):
            fn(**input_kwargs)
        ms_list = bench_gpu_time(
            fn,
            dry_run_iters=warmups,
            repeat_iters=iterations,
            enable_cupti=True,
            use_cuda_graph=True,
            input_kwargs=input_kwargs,
            cold_l2_cache=True,
        )
        median_ms = np.median(ms_list)
        return median_ms

    ms = bench(do_autotune=False)
    ms_tuned = bench(do_autotune=True)
    print(
        f"num tokens: {num_tokens}, num experts: {num_experts}, hidden size: {hidden_size}, intermediate size: {intermediate_size}, top k: {top_k}"
    )
    print(f"No autotune: {ms:.3f} ms; with autotune: {ms_tuned:.3f} ms")


def bench_trtllm_gen_fused_moe_autotuner_fp4(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
    activation_type: int,
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
            is_sf_swizzled_layout=False,
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
        assert activation_type != ActivationType.Relu2.value, (
            "Relu2 activation is supported for FP4 only with 'NvFP4xNvFP4' quant mode"
        )
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

    output1_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output1_scale_gate_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output2_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w2_global_scale] * num_experts, device=device
    )
    fn = partial(
        trtllm_fp4_block_scale_moe,
        routing_logits=routing_logits,
        routing_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=True,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        output=None,
        tune_max_num_tokens=num_tokens
        if tune_max_num_tokens is None
        else tune_max_num_tokens,
    )

    input_kwargs = {
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13,
        "gemm1_weights_scale": w13_scale,
        "gemm2_weights": w2,
        "gemm2_weights_scale": w2_scale,
        "gemm1_bias": bias13,
        "gemm2_bias": bias2,
    }

    def bench(do_autotune):
        with autotune(do_autotune):
            fn(**input_kwargs)
        ms_list = bench_gpu_time(
            fn,
            dry_run_iters=warmups,
            repeat_iters=iterations,
            enable_cupti=True,
            use_cuda_graph=True,
            input_kwargs=input_kwargs,
            cold_l2_cache=True,
        )
        median_ms = np.median(ms_list)
        return median_ms

    ms = bench(do_autotune=False)
    ms_tuned = bench(do_autotune=True)
    print(
        f"num tokens: {num_tokens}, num experts: {num_experts}, hidden size: {hidden_size}, intermediate size: {intermediate_size}, top k: {top_k}"
    )
    print(f"No autotune: {ms:.3f} ms; with autotune: {ms_tuned:.3f} ms")


def bench_trtllm_gen_fused_moe_autotuner_mxint4(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["MxInt4xBf16"],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
    activation_type: int,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(num_tokens, num_experts, device=device).float()
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(num_tokens, hidden_size, device=device).to(
        torch.bfloat16
    )

    w13 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.bfloat16
    )
    w13, w13_scale = mxint4_quantize(w13, 32)
    w13_scale = w13_scale.to(torch.bfloat16).reshape(
        num_experts,
        2 * intermediate_size,
        hidden_size // 32,
    )
    w2, w2_scale = mxint4_quantize(w2, 32)
    w2_scale = w2_scale.to(torch.bfloat16).reshape(
        num_experts,
        hidden_size,
        intermediate_size // 32,
    )

    assert activation_type == ActivationType.Swiglu, (
        "only SwiGlu activation is supported for MxInt4 MoE currently"
    )
    fn = partial(
        trtllm_mxint4_block_scale_moe,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        num_experts=num_experts,
        top_k=top_k,
        n_group=1,
        topk_group=1,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.DeepSeekV3.value,
        enable_pdl=enable_pdl,
        output=None,
        tune_max_num_tokens=num_tokens
        if tune_max_num_tokens is None
        else tune_max_num_tokens,
    )

    input_kwargs = {
        "hidden_states": hidden_states,
        "gemm1_weights": w13,
        "gemm1_weights_scale": w13_scale,
        "gemm2_weights": w2,
        "gemm2_weights_scale": w2_scale,
    }

    def bench(do_autotune):
        with autotune(do_autotune):
            fn(**input_kwargs)
        ms_list = bench_gpu_time(
            fn,
            dry_run_iters=warmups,
            repeat_iters=iterations,
            enable_cupti=True,
            use_cuda_graph=True,
            input_kwargs=input_kwargs,
            cold_l2_cache=True,
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
            "MxInt4xBf16",
            "MxFP8xMxFP8",
            "Fp8-Per-Tensor",
            "Fp8-Block",
        ],
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
    parser.add_argument(
        "--activation-type",
        type=enum_type(ActivationType),
        metavar=str([e.name for e in ActivationType]),
        required=False,
        default=ActivationType.Swiglu,
        help=f"Type of activation function: {[e.name for e in ActivationType]}",
    )
    args = parser.parse_args()
    fn = (
        bench_trtllm_gen_fused_moe_autotuner_fp8
        if args.quant_mode in ["Fp8-Per-Tensor", "Fp8-Block", "MxFP8xMxFP8"]
        else bench_trtllm_gen_fused_moe_autotuner_mxint4
        if args.quant_mode == "MxInt4xBf16"
        else bench_trtllm_gen_fused_moe_autotuner_fp4
    )
    fn(
        args.tune_max_num_tokens,
        args.quant_mode,
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.warmups,
        args.iterations,
        args.activation_type,
    )
