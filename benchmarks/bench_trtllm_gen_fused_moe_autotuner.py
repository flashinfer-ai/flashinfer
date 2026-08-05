import argparse
from dataclasses import dataclass
from typing import Callable, Literal, Optional
import torch
import numpy as np
from functools import partial
from flashinfer import (
    RoutingMethodType,
    ActivationType,
    fp4_quantize,
    mxfp8_quantize,
)
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    Fp8QuantizationType,
    prims_ts_fp4_block_scale_moe,
    prims_ts_fp4_block_scale_routed_moe,
    prims_ts_fp8_block_scale_moe,
    prims_ts_fp8_block_scale_routed_moe,
    prims_ts_fp8_per_tensor_scale_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_mxint4_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    WeightLayout,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.autotuner import autotune, AutoTuner
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import device_support_pdl
from routines.flashinfer_benchmark_utils import enum_type
from flashinfer.fused_moe.utils import make_random_topk_ids

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0
BACKENDS = ("trtllm", "prims_ts")


@dataclass(frozen=True)
class BenchmarkSetup:
    batch_size: int
    backend: str
    fn: Callable
    input_kwargs: dict


@dataclass(frozen=True)
class BenchmarkResult:
    batch_size: int
    backend: str
    no_autotune_ms: float
    tuned_ms: float


def _pack_topk(
    num_tokens: int, top_k: int, num_experts: int, device: torch.device
) -> torch.Tensor:
    topk_ids = make_random_topk_ids(num_experts, num_tokens, top_k, device)
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


def _print_table(results: list[BenchmarkResult], config_str: str):
    print(f"\n{config_str}")
    col0, col1, col2, col3, col4, col5 = 12, 10, 18, 16, 9, 15
    header = (
        f"  {'num_tokens':>{col0}}  {'backend':>{col1}}"
        f"  {'no_autotune (ms)':>{col2}}  {'autotuned (ms)':>{col3}}"
        f"  {'speedup':>{col4}}  {'vs trtllm':>{col5}}"
    )
    sep = (
        f"  {'-' * col0}  {'-' * col1}  {'-' * col2}  {'-' * col3}"
        f"  {'-' * col4}  {'-' * col5}"
    )
    print(header)
    print(sep)
    trtllm_tuned = {
        result.batch_size: result.tuned_ms
        for result in results
        if result.backend == "trtllm"
    }
    for result in sorted(results, key=lambda item: (item.batch_size, item.backend)):
        speedup = result.no_autotune_ms / result.tuned_ms
        baseline = trtllm_tuned.get(result.batch_size)
        backend_ratio = baseline / result.tuned_ms if baseline is not None else None
        backend_ratio_str = (
            f"{backend_ratio:.2f}x" if backend_ratio is not None else "n/a"
        )
        print(
            f"  {result.batch_size:>{col0}}  {result.backend:>{col1}}"
            f"  {result.no_autotune_ms:>{col2}.3f}"
            f"  {result.tuned_ms:>{col3}.3f}  {speedup:>{col4}.2f}x"
            f"  {backend_ratio_str:>{col5}}"
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
    setups: list[BenchmarkSetup],
    warmups: int,
    iterations: int,
    config_str: str,
    tuning_buckets: Optional[list[int]] = None,
):
    AutoTuner.get().clear_cache()

    measure = partial(_measure, warmups=warmups, iterations=iterations)

    # measure untuned
    ms_no_autotune = [measure(setup.fn, setup.input_kwargs) for setup in setups]

    # Tune each backend once.  The tuning config controls whether this covers
    # all buckets up to tune_max or only the explicit user-requested buckets.
    tuned_backends = set()
    tuning_buckets_tuple = None if tuning_buckets is None else tuple(tuning_buckets)
    for setup in setups:
        if setup.backend in tuned_backends:
            continue
        with autotune(True, tuning_buckets=tuning_buckets_tuple):
            setup.fn(**setup.input_kwargs)
        tuned_backends.add(setup.backend)

    # measure tuned
    results = [
        BenchmarkResult(
            setup.batch_size,
            setup.backend,
            ms,
            measure(setup.fn, setup.input_kwargs),
        )
        for setup, ms in zip(setups, ms_no_autotune, strict=True)
    ]

    _print_table(results, config_str)


def _normalize_backends(backends: list[str], quant_mode: str) -> list[str]:
    if "both" in backends:
        resolved = list(BACKENDS)
    else:
        resolved = []
        for backend in backends:
            if backend not in resolved:
                resolved.append(backend)

    if "prims_ts" in resolved:
        if quant_mode == "MxInt4xBf16":
            raise ValueError(
                "Prims-TS is not wired for MxInt4xBf16 in this benchmark"
            )
        if not is_prims_ts_available():
            raise RuntimeError(
                "Prims-TS backend requested but dependencies are unavailable"
            )
    return resolved


def _fp4_ops(backend: str, routed: bool) -> Callable:
    if backend == "trtllm":
        return (
            trtllm_fp4_block_scale_routed_moe
            if routed
            else trtllm_fp4_block_scale_moe
        )
    if backend == "prims_ts":
        return (
            prims_ts_fp4_block_scale_routed_moe
            if routed
            else prims_ts_fp4_block_scale_moe
        )
    raise ValueError(f"Unknown backend: {backend}")


def _fp8_per_tensor_op(backend: str) -> Callable:
    if backend == "trtllm":
        return trtllm_fp8_per_tensor_scale_moe
    if backend == "prims_ts":
        return prims_ts_fp8_per_tensor_scale_moe
    raise ValueError(f"Unknown backend: {backend}")


def _fp8_block_ops(backend: str, routed: bool) -> Callable:
    if backend == "trtllm":
        return (
            trtllm_fp8_block_scale_routed_moe
            if routed
            else trtllm_fp8_block_scale_moe
        )
    if backend == "prims_ts":
        return (
            prims_ts_fp8_block_scale_routed_moe
            if routed
            else prims_ts_fp8_block_scale_moe
        )
    raise ValueError(f"Unknown backend: {backend}")


def _shuffle_fp4_major_k(
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    sf_vec_size: int,
    gemm1_bias: Optional[torch.Tensor] = None,
    gemm2_bias: Optional[torch.Tensor] = None,
) -> dict[str, Optional[torch.Tensor]]:
    """Convert canonical FP4 expert tensors to the shuffled MajorK layout.

    The FP4 MoE kernels benchmarked here consume the TRT-LLM/Prims-TS native
    layout: gated FC1 rows and FC2 rows are epilogue-tile permuted, and their
    block-scale tensors are interleaved to match the packed weight layout.
    """
    epilogue_tile_m = 128
    gemm1_weights = gemm1_weights.view(torch.uint8).reshape(
        num_local_experts, 2 * intermediate_size, hidden_size // 2
    )
    gemm1_scales = gemm1_scales.view(torch.float8_e4m3fn).reshape(
        num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size
    )
    gemm2_weights = gemm2_weights.view(torch.uint8).reshape(
        num_local_experts, hidden_size, intermediate_size // 2
    )
    gemm2_scales = gemm2_scales.view(torch.float8_e4m3fn).reshape(
        num_local_experts, hidden_size, intermediate_size // sf_vec_size
    )

    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []
    gemm1_bias_shuffled = [] if gemm1_bias is not None else None
    gemm2_bias_shuffled = [] if gemm2_bias is not None else None
    permute_cache = {}
    for expert_idx in range(num_local_experts):
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            gemm1_weights[expert_idx],
            epilogue_tile_m,
            is_gated_act_gemm=True,
        )
        gemm1_weights_shuffled.append(
            gemm1_weights[expert_idx][permute_indices.to(gemm1_weights.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            gemm1_scales[expert_idx].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        gemm1_scales_shuffled.append(
            block_scale_interleave(
                gemm1_scales[expert_idx]
                .view(torch.uint8)[permute_sf_indices.to(gemm1_scales.device)]
                .contiguous()
            )
        )

        if gemm1_bias_shuffled is not None:
            permute_bias_indices = _maybe_get_cached_w3_w1_permute_indices(
                permute_cache,
                gemm1_bias[expert_idx].reshape(-1, 1),
                epilogue_tile_m,
                is_gated_act_gemm=True,
            )
            gemm1_bias_shuffled.append(
                gemm1_bias[expert_idx]
                .reshape(-1, 1)[permute_bias_indices.to(gemm1_bias.device)]
                .contiguous()
            )

        permute_indices = get_w2_permute_indices_with_cache(
            permute_cache, gemm2_weights[expert_idx], epilogue_tile_m
        )
        gemm2_weights_shuffled.append(
            gemm2_weights[expert_idx][permute_indices.to(gemm2_weights.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            permute_cache,
            gemm2_scales[expert_idx].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_shuffled.append(
            block_scale_interleave(
                gemm2_scales[expert_idx]
                .view(torch.uint8)[permute_sf_indices.to(gemm2_scales.device)]
                .contiguous()
            )
        )

        if gemm2_bias_shuffled is not None:
            permute_bias_indices = get_w2_permute_indices_with_cache(
                permute_cache,
                gemm2_bias[expert_idx].reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm2_bias_shuffled.append(
                gemm2_bias[expert_idx]
                .reshape(-1, 1)[permute_bias_indices.to(gemm2_bias.device)]
                .contiguous()
            )

    result = {
        "gemm1_weights": torch.stack(gemm1_weights_shuffled),
        "gemm1_weights_scale": torch.stack(gemm1_scales_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size),
        "gemm2_weights": torch.stack(gemm2_weights_shuffled),
        "gemm2_weights_scale": torch.stack(gemm2_scales_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, hidden_size, intermediate_size // sf_vec_size),
        "gemm1_bias": None,
        "gemm2_bias": None,
    }
    if gemm1_bias_shuffled is not None:
        result["gemm1_bias"] = torch.stack(gemm1_bias_shuffled).reshape(
            num_local_experts, 2 * intermediate_size
        )
    if gemm2_bias_shuffled is not None:
        result["gemm2_bias"] = torch.stack(gemm2_bias_shuffled).reshape(
            num_local_experts, hidden_size
        )
    return result


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
    backends: list[str],
    tuning_buckets: Optional[list[int]] = None,
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
            common_fn_kwargs = dict(
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
            for backend in backends:
                setups.append(
                    BenchmarkSetup(
                        batch_size,
                        backend,
                        partial(_fp8_per_tensor_op(backend), **common_fn_kwargs),
                        input_kwargs,
                    )
                )
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
                common_fn_kwargs = dict(
                    topk_ids=_pack_topk(batch_size, top_k, num_experts, device),
                    routing_method_type=RoutingMethodType.Renormalize.value,
                    **block_scale_kwargs,
                )
            else:
                common_fn_kwargs = dict(
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
            for backend in backends:
                setups.append(
                    BenchmarkSetup(
                        batch_size,
                        backend,
                        partial(_fp8_block_ops(backend, routed), **common_fn_kwargs),
                        input_kwargs,
                    )
                )

    mode_str = "routed" if routed else "non_routed"
    _run_benchmark(
        setups,
        warmups,
        iterations,
        f"quant_mode={quant_mode}  routing={mode_str}  experts={num_experts}"
        f"  hidden={hidden_size}  intermediate={intermediate_size}  top_k={top_k}",
        tuning_buckets=tuning_buckets,
    )


def bench_trtllm_gen_fused_moe_autotuner_fp4(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
    num_tokens_list: list[int],
    num_experts: int,
    local_num_experts: int,
    local_expert_offset: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
    activation_type: int,
    backends: list[str],
    tuning_buckets: Optional[list[int]] = None,
    use_bias: bool = True,
    routed: bool = False,
):
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    tune_max = (
        max(num_tokens_list) if tune_max_num_tokens is None else tune_max_num_tokens
    )
    if local_expert_offset < 0 or local_num_experts <= 0:
        raise ValueError(
            "local_expert_offset must be non-negative and local_num_experts positive"
        )
    if local_expert_offset + local_num_experts > num_experts:
        raise ValueError(
            "local_expert_offset + local_num_experts must be <= num_experts"
        )

    # --- num_tokens-independent setup ---
    w13 = torch.randn(
        local_num_experts, intermediate_size * 2, hidden_size, device=device
    ).to(torch.bfloat16)
    w2 = torch.randn(
        local_num_experts, hidden_size, intermediate_size, device=device
    ).to(
        torch.bfloat16
    )
    bias13 = (
        torch.randn(local_num_experts, intermediate_size * 2, device=device) * 10
        if use_bias
        else None
    )
    bias2 = (
        torch.randn(local_num_experts, hidden_size, device=device) * 10
        if use_bias
        else None
    )

    if quant_mode == "NvFP4xNvFP4":
        w13, w13_scale = fp4_quantize(
            w13,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            local_num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            local_num_experts, hidden_size, -1
        )
        w13_global_scale = w2_global_scale = 1.0 / 448.0 / 6.0
        hidden_states_global_scale = 1.0 / 448.0 / 6.0
        sf_vec_size = 16
    else:
        assert activation_type != ActivationType.Relu2.value, (
            "Relu2 activation is supported for FP4 only with 'NvFP4xNvFP4' quant mode"
        )
        w13, w13_scale = fp4_quantize(
            w13,
            torch.tensor([1.0], device=device),
            sf_vec_size=32,
            sf_use_ue8m0=True,
            is_sf_swizzled_layout=False,
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            local_num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            torch.tensor([1.0], device=device),
            sf_vec_size=32,
            sf_use_ue8m0=True,
            is_sf_swizzled_layout=False,
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            local_num_experts, hidden_size, -1
        )
        w13_global_scale = w2_global_scale = 1.0
        hidden_states_global_scale = 1.0
        sf_vec_size = 32

    output1_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * local_num_experts,
        device=device,
    )
    output1_scale_gate_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * local_num_experts,
        device=device,
    )
    output2_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w2_global_scale] * local_num_experts,
        device=device,
    )

    shuffled = _shuffle_fp4_major_k(
        w13,
        w13_scale,
        w2,
        w2_scale,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=local_num_experts,
        sf_vec_size=sf_vec_size,
        gemm1_bias=bias13,
        gemm2_bias=bias2,
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
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
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
            common_fn_kwargs = dict(
                topk_ids=_pack_topk(batch_size, top_k, num_experts, device),
                **fp4_kwargs,
            )
        else:
            common_fn_kwargs = dict(
                routing_logits=torch.rand(batch_size, num_experts, device=device).to(
                    torch.bfloat16
                ),
                **fp4_kwargs,
            )

        input_kwargs = {
            "hidden_states": hidden_states,
            "hidden_states_scale": hidden_states_scale,
            "gemm1_weights": shuffled["gemm1_weights"],
            "gemm1_weights_scale": shuffled["gemm1_weights_scale"],
            "gemm2_weights": shuffled["gemm2_weights"],
            "gemm2_weights_scale": shuffled["gemm2_weights_scale"],
            "gemm1_bias": shuffled["gemm1_bias"],
            "gemm2_bias": shuffled["gemm2_bias"],
        }
        for backend in backends:
            setups.append(
                BenchmarkSetup(
                    batch_size,
                    backend,
                    partial(_fp4_ops(backend, routed), **common_fn_kwargs),
                    input_kwargs,
                )
            )

    mode_str = "routed" if routed else "non_routed"
    _run_benchmark(
        setups,
        warmups,
        iterations,
        f"quant_mode={quant_mode}  routing={mode_str}  experts={num_experts}"
        f"  local_experts={local_num_experts}  local_offset={local_expert_offset}"
        f"  hidden={hidden_size}  intermediate={intermediate_size}  top_k={top_k}"
        f"  bias={use_bias}",
        tuning_buckets=tuning_buckets,
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
        setups.append(BenchmarkSetup(batch_size, "trtllm", fn, input_kwargs))

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
        default=None,
        help="Number of tokens (one or more)",
    )
    parser.add_argument(
        "--tuning-buckets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit autotune measurement buckets. Defaults to the API bucket "
            "generation when omitted."
        ),
    )
    parser.add_argument(
        "--tune-max-num-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens for tuning (defaults to max of --num-tokens)",
    )
    parser.add_argument(
        "--num-experts", type=int, default=None, help="Number of global experts"
    )
    parser.add_argument(
        "--local-num-experts",
        type=int,
        default=None,
        help="Number of resident local experts",
    )
    parser.add_argument(
        "--local-expert-offset",
        type=int,
        default=0,
        help="Global expert id offset for the first local expert",
    )
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden size")
    parser.add_argument(
        "--intermediate-size", type=int, default=None, help="Intermediate size"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism degree; divides intermediate-size",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k experts per token"
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["both"],
        choices=["trtllm", "prims_ts", "both"],
        help="Backends to benchmark. Use 'both' to compare TRT-LLM Gen and Prims-TS.",
    )
    parser.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include FC1/FC2 bias tensors in FP4 benchmarks.",
    )
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

    if args.num_tokens is None:
        args.num_tokens = [512]
    if args.num_experts is None:
        args.num_experts = 128
    if args.hidden_size is None:
        args.hidden_size = 3072
    if args.intermediate_size is None:
        args.intermediate_size = 3072
    if args.top_k is None:
        args.top_k = 4
    if args.use_bias is None:
        args.use_bias = True

    if args.local_num_experts is None:
        args.local_num_experts = args.num_experts

    backends = _normalize_backends(args.backends, args.quant_mode)
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
            backends,
            tuning_buckets=args.tuning_buckets,
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
            args.local_num_experts,
            args.local_expert_offset,
            args.hidden_size,
            args.intermediate_size,
            args.top_k,
            args.warmups,
            args.iterations,
            args.activation_type,
            backends,
            tuning_buckets=args.tuning_buckets,
            use_bias=args.use_bias,
            routed=args.routed,
        )
