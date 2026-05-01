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
    trtllm_fp4_block_scale_routed_moe,
    trtllm_mxint4_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    WeightLayout,
)
from flashinfer.autotuner import autotune, AutoTuner
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import device_support_pdl
from routines.flashinfer_benchmark_utils import enum_type

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0


def _pack_topk(
    num_tokens: int, top_k: int, num_experts: int, device: torch.device
) -> torch.Tensor:
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    raw_w = torch.rand(num_tokens, top_k, device=device)
    weights = (raw_w / raw_w.sum(-1, keepdim=True)).to(torch.bfloat16)
    return (topk_ids << 16) | weights.view(torch.int16).to(torch.int32)


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


def _print_table(results: list[tuple[int, float, float]], config_str: str):
    print(f"\n{config_str}")
    col0, col1, col2, col3 = 12, 18, 16, 9
    header = f"  {'num_tokens':>{col0}}  {'no_autotune (ms)':>{col1}}  {'autotuned (ms)':>{col2}}  {'speedup':>{col3}}"
    sep = f"  {'-' * col0}  {'-' * col1}  {'-' * col2}  {'-' * col3}"
    print(header)
    print(sep)
    for num_tokens, ms, ms_tuned in results:
        speedup = ms / ms_tuned
        print(
            f"  {num_tokens:>{col0}}  {ms:>{col1}.3f}  {ms_tuned:>{col2}.3f}  {speedup:>{col3}.2f}x"
        )


def _measure(fn, input_kwargs, warmups, iterations):
    ms_list = bench_gpu_time(
        fn,
        dry_run_iters=warmups,
        repeat_iters=iterations,
        enable_cupti=True,
        use_cuda_graph=True,
        input_kwargs=input_kwargs,
        cold_l2_cache=True,
    )
    return np.median(ms_list)


def _run_benchmark(
    setups: list[tuple[int, callable, dict]],
    warmups: int,
    iterations: int,
    config_str: str,
):
    AutoTuner.get().clear_cache()

    measure = partial(_measure, warmups=warmups, iterations=iterations)

    # measure untuned
    ms_no_autotune = [measure(fn, kw) for _, fn, kw in setups]

    # tune once — covers all buckets up to tune_max
    _, first_fn, first_kw = setups[0]
    with autotune(True):
        first_fn(**first_kw)

    # measure tuned
    results = [
        (batch_size, ms, measure(fn, kw))
        for (batch_size, fn, kw), ms in zip(setups, ms_no_autotune, strict=True)
    ]

    _print_table(results, config_str)


def bench_trtllm_gen_fused_moe_autotuner_fp8(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["Fp8-Per-Tensor", "Fp8-Block", "MxFP8xMxFP8"],
    num_tokens_list: list[int],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
    activation_type: int,
    routed: bool = False,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    is_block_scale = quant_mode != "Fp8-Per-Tensor"
    tune_max = (
        max(num_tokens_list) if tune_max_num_tokens is None else tune_max_num_tokens
    )

    # --- num_tokens-independent setup ---
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)
    w13 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.bfloat16
    )

    scale_vec_size = 128 if quant_mode == "Fp8-Block" else 32
    if quant_mode == "Fp8-Per-Tensor":
        _, hidden_states_scale_scalar = fp8_quantize(w13[:1, :1, :1])  # dummy for scale
        w13, w13_scale = fp8_quantize(w13)
        w2, w2_scale = fp8_quantize(w2)
        _, hidden_states_scale_scalar = fp8_quantize(
            torch.randn(1, hidden_size, device=device).to(torch.bfloat16)
        )
        output1_scale_scalar = torch.tensor(
            [hidden_states_scale_scalar * w13_scale] * num_experts, device=device
        )
        output1_scales_gate_scalar = torch.ones(
            num_experts, device=device, dtype=torch.float32
        )
        output2_scale_scalar = torch.tensor(
            [hidden_states_scale_scalar * w2_scale] * num_experts, device=device
        )
    elif quant_mode == "Fp8-Block":
        w13, w13_scalar = fp8_quantize(w13)
        w2, w2_scalar = fp8_quantize(w2)
        w13_scale = torch.full(
            (
                num_experts,
                intermediate_size * 2 // scale_vec_size,
                hidden_size // scale_vec_size,
            ),
            w13_scalar.item(),
            device=device,
        )
        w2_scale = torch.full(
            (
                num_experts,
                hidden_size // scale_vec_size,
                intermediate_size // scale_vec_size,
            ),
            w2_scalar.item(),
            device=device,
        )
    else:  # MxFP8xMxFP8
        w13, w13_scale = mxfp8_quantize(w13, True)
        w2, w2_scale = mxfp8_quantize(w2, True)
        w13_scale = w13_scale.view(torch.uint8).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2_scale = w2_scale.view(torch.uint8).reshape(num_experts, hidden_size, -1)

    if is_block_scale:
        assert activation_type == ActivationType.Swiglu.value, (
            "Only Swiglu activation is supported for FP8 block scale MoE."
        )

    setups = []
    for batch_size in num_tokens_list:
        hidden_states_bf16 = torch.randn(batch_size, hidden_size, device=device).to(
            torch.bfloat16
        )

        if quant_mode == "Fp8-Per-Tensor":
            hidden_states, hs_scale = fp8_quantize(hidden_states_bf16)
            fn = partial(
                trtllm_fp8_per_tensor_scale_moe,
                routing_logits=torch.rand(batch_size, num_experts, device=device).to(
                    torch.bfloat16
                ),
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
                tune_max_num_tokens=tune_max,
                activation_type=activation_type,
            )
            input_kwargs = {
                "hidden_states": hidden_states,
                "gemm1_weights": w13,
                "gemm2_weights": w2,
            }
        else:
            if quant_mode == "Fp8-Block":
                hidden_states, hs_scalar = fp8_quantize(hidden_states_bf16)
                hidden_states_scale = torch.full(
                    (hidden_size // scale_vec_size, batch_size),
                    hs_scalar.item(),
                    device=device,
                )
            else:  # MxFP8xMxFP8
                hidden_states, hs_scale = mxfp8_quantize(hidden_states_bf16, False)
                hidden_states_scale = hs_scale.view(torch.uint8).reshape(batch_size, -1)

            block_scale_kwargs = dict(
                routing_bias=routing_bias,
                num_experts=num_experts,
                top_k=top_k,
                n_group=None if routed else 8,
                topk_group=None if routed else 4,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=2.5,
                use_shuffled_weight=quant_mode == "MxFP8xMxFP8",
                weight_layout=WeightLayout.MajorK.value,
                enable_pdl=enable_pdl,
                tune_max_num_tokens=tune_max,
                fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8
                if quant_mode == "Fp8-Block"
                else Fp8QuantizationType.MxFp8,
            )
            if routed:
                fn = partial(
                    trtllm_fp8_block_scale_routed_moe,
                    topk_ids=_pack_topk(batch_size, top_k, num_experts, device),
                    routing_method_type=RoutingMethodType.Renormalize.value,
                    **block_scale_kwargs,
                )
            else:
                fn = partial(
                    trtllm_fp8_block_scale_moe,
                    routing_logits=torch.rand(
                        batch_size, num_experts, device=device
                    ).to(torch.float32),
                    routing_method_type=RoutingMethodType.DeepSeekV3.value,
                    **block_scale_kwargs,
                )
            input_kwargs = {
                "hidden_states": hidden_states,
                "hidden_states_scale": hidden_states_scale,
                "gemm1_weights": w13,
                "gemm1_weights_scale": w13_scale,
                "gemm2_weights": w2,
                "gemm2_weights_scale": w2_scale,
            }
        setups.append((batch_size, fn, input_kwargs))

    mode_str = "routed" if routed else "non_routed"
    _run_benchmark(
        setups,
        warmups,
        iterations,
        f"quant_mode={quant_mode}  routing={mode_str}  experts={num_experts}"
        f"  hidden={hidden_size}  intermediate={intermediate_size}  top_k={top_k}",
    )


def bench_trtllm_gen_fused_moe_autotuner_fp4(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
    num_tokens_list: list[int],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
    activation_type: int,
    routed: bool = False,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    tune_max = (
        max(num_tokens_list) if tune_max_num_tokens is None else tune_max_num_tokens
    )

    # --- num_tokens-independent setup ---
    w13 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.bfloat16
    )
    bias13 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10
    bias2 = torch.randn(num_experts, hidden_size, device=device) * 10

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
        w13_global_scale = w2_global_scale = 1.0 / 448.0 / 6.0
        hidden_states_global_scale = 1.0 / 448.0 / 6.0
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
        w13_global_scale = w2_global_scale = 1.0
        hidden_states_global_scale = 1.0

    output1_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output1_scale_gate_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output2_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w2_global_scale] * num_experts, device=device
    )

    fp4_kwargs = dict(
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
        tune_max_num_tokens=tune_max,
    )

    setups = []
    for batch_size in num_tokens_list:
        hidden_states = torch.randn(batch_size, hidden_size, device=device).to(
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
                batch_size, -1
            )
        elif quant_mode == "MxFP4xMxFP8":
            hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
                batch_size, -1
            )
        else:  # MxFP4xBf16
            hidden_states_scale = None

        if routed:
            fn = partial(
                trtllm_fp4_block_scale_routed_moe,
                topk_ids=_pack_topk(batch_size, top_k, num_experts, device),
                **fp4_kwargs,
            )
        else:
            fn = partial(
                trtllm_fp4_block_scale_moe,
                routing_logits=torch.rand(batch_size, num_experts, device=device).to(
                    torch.bfloat16
                ),
                **fp4_kwargs,
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
        setups.append((batch_size, fn, input_kwargs))

    mode_str = "routed" if routed else "non_routed"
    _run_benchmark(
        setups,
        warmups,
        iterations,
        f"quant_mode={quant_mode}  routing={mode_str}  experts={num_experts}"
        f"  hidden={hidden_size}  intermediate={intermediate_size}  top_k={top_k}",
    )


def bench_trtllm_gen_fused_moe_autotuner_mxint4(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["MxInt4xBf16"],
    num_tokens_list: list[int],
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
    tune_max = (
        max(num_tokens_list) if tune_max_num_tokens is None else tune_max_num_tokens
    )

    # --- num_tokens-independent setup ---
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)
    w13 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.bfloat16
    )
    w13, w13_scale = mxint4_quantize(w13, 32)
    w13_scale = w13_scale.to(torch.bfloat16).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 32
    )
    w2, w2_scale = mxint4_quantize(w2, 32)
    w2_scale = w2_scale.to(torch.bfloat16).reshape(
        num_experts, hidden_size, intermediate_size // 32
    )

    assert activation_type == ActivationType.Swiglu, (
        "only SwiGlu activation is supported for MxInt4 MoE currently"
    )

    setups = []
    for batch_size in num_tokens_list:
        hidden_states = torch.randn(batch_size, hidden_size, device=device).to(
            torch.bfloat16
        )
        fn = partial(
            trtllm_mxint4_block_scale_moe,
            routing_logits=torch.rand(batch_size, num_experts, device=device).float(),
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
            tune_max_num_tokens=tune_max,
        )
        input_kwargs = {
            "hidden_states": hidden_states,
            "gemm1_weights": w13,
            "gemm1_weights_scale": w13_scale,
            "gemm2_weights": w2,
            "gemm2_weights_scale": w2_scale,
        }
        setups.append((batch_size, fn, input_kwargs))

    _run_benchmark(
        setups,
        warmups,
        iterations,
        f"quant_mode={quant_mode}  experts={num_experts}"
        f"  hidden={hidden_size}  intermediate={intermediate_size}  top_k={top_k}",
    )


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
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[512],
        help="Number of tokens (one or more)",
    )
    parser.add_argument(
        "--tune-max-num-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens for tuning (defaults to max of --num-tokens)",
    )
    parser.add_argument(
        "--num-experts", type=int, default=128, help="Number of experts"
    )
    parser.add_argument("--hidden-size", type=int, default=3072, help="Hidden size")
    parser.add_argument(
        "--intermediate-size", type=int, default=3072, help="Intermediate size"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism degree; divides intermediate-size",
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
    parser.add_argument(
        "--routed",
        action="store_true",
        default=False,
        help="Use pre-computed topk_ids (routed) path instead of routing_logits. "
        "Not supported for Fp8-Per-Tensor or MxInt4xBf16.",
    )
    args = parser.parse_args()
    args.intermediate_size //= args.tp

    is_fp8 = args.quant_mode in ["Fp8-Per-Tensor", "Fp8-Block", "MxFP8xMxFP8"]
    is_mxint4 = args.quant_mode == "MxInt4xBf16"

    if args.routed and args.quant_mode == "Fp8-Per-Tensor":
        raise ValueError("--routed is not supported for Fp8-Per-Tensor.")
    if args.routed and is_mxint4:
        raise ValueError("--routed is not supported for MxInt4xBf16.")

    if is_fp8:
        bench_trtllm_gen_fused_moe_autotuner_fp8(
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
            routed=args.routed,
        )
    elif is_mxint4:
        bench_trtllm_gen_fused_moe_autotuner_mxint4(
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
    else:
        bench_trtllm_gen_fused_moe_autotuner_fp4(
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
            routed=args.routed,
        )
