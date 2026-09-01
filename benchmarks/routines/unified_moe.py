"""Apples-to-apples benchmarks for unified CUTLASS and cuTile MoE runners."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import flashinfer
from flashinfer import ActivationType, fp4_quantize
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CutlassBf16Config,
    CutlassNvfp4Config,
    CuTileBf16Config,
    CuTileNvfp4Config,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    ReLU2,
    RoutingConfig,
    SwiGLU,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    enum_type,
    get_device,
    print_perf_metrics,
)
from .moe_utils import (
    add_common_moe_args,
    calculate_moe_kernel_bandwidth,
    calculate_moe_tflops,
)


_BACKEND_CONFIGS = {
    ("bf16", "cutlass"): CutlassBf16Config,
    ("bf16", "cutile"): CuTileBf16Config,
    ("nvfp4", "cutlass"): CutlassNvfp4Config,
    ("nvfp4", "cutile"): CuTileNvfp4Config,
}

_ACTIVATIONS = {
    ActivationType.Swiglu: SwiGLU,
    ActivationType.Relu2: ReLU2,
}


def parse_unified_moe_args(line, parser: argparse.ArgumentParser):
    """Add arguments for the unified MoE comparison routine."""
    add_common_moe_args(parser)
    parser.add_argument("--intermediate_size", type=int, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cutlass", "cutile"),
        default=["cutlass", "cutile"],
        help="Unified MoE backends to benchmark with the same inputs.",
    )
    parser.add_argument(
        "--quant-variant",
        "--quant_variant",
        dest="quant_variant",
        choices=("bf16", "nvfp4"),
        default="bf16",
    )
    parser.add_argument(
        "--activation-type",
        type=enum_type(ActivationType),
        metavar=str([member.name for member in _ACTIVATIONS]),
        default=ActivationType.Swiglu,
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help="Autotune each backend independently before measuring it.",
    )
    args = parser.parse_args(line)
    args.backends = list(dict.fromkeys(args.backends))
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def _canonical_inputs(args, activation, device: torch.device):
    if args.num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if args.hidden_size <= 0 or args.intermediate_size <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive")
    if args.num_experts <= 0 or not 0 < args.top_k <= args.num_experts:
        raise ValueError("require 0 < top_k <= num_experts")

    torch.manual_seed(args.random_seed)
    w1_rows = args.intermediate_size * (2 if activation.is_gated else 1)
    hidden_states = torch.randn(
        args.num_tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w1 = (
        torch.randn(
            args.num_experts,
            w1_rows,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / args.hidden_size**0.5
    )
    w2 = (
        torch.randn(
            args.num_experts,
            args.hidden_size,
            args.intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / args.intermediate_size**0.5
    )
    topk_ids = (
        torch.arange(
            args.num_tokens * args.top_k,
            dtype=torch.int32,
            device=device,
        ).reshape(args.num_tokens, args.top_k)
        % args.num_experts
    )
    topk_weights = torch.rand(
        args.num_tokens, args.top_k, dtype=torch.float32, device=device
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    activations = MoEActivationPack(
        hidden_states_q=hidden_states,
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    return activations, w1, w2


def _quantize_cutile_nvfp4_source(weight: torch.Tensor):
    """Quantize canonical BF16 weights into cuTile's checkpoint input layout."""
    num_experts, rows, cols = weight.shape
    global_scales = torch.ones(num_experts, dtype=torch.float32, device=weight.device)
    packed_experts = []
    scale_experts = []
    for expert in range(num_experts):
        packed, scale = fp4_quantize(
            weight[expert],
            global_scale=global_scales[expert : expert + 1],
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
            enable_pdl=False,
        )
        packed_experts.append(packed)
        scale_experts.append(scale.view(torch.float8_e4m3fn).reshape(rows, cols // 16))
    return (
        torch.stack(packed_experts).contiguous(),
        torch.stack(scale_experts).contiguous(),
        global_scales,
    )


def _prepare_weight_view(
    backend: str,
    quant_variant: str,
    config_type,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation,
    args,
    device: torch.device,
):
    common = {
        "num_local_experts": args.num_experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "activation": activation,
        "device": device,
    }
    if quant_variant == "bf16" or backend == "cutlass":
        return config_type.prepare_weights(w1, w2, **common)

    w1_q, w1_scale, w1_global = _quantize_cutile_nvfp4_source(w1)
    w2_q, w2_scale, w2_global = _quantize_cutile_nvfp4_source(w2)
    return config_type.prepare_weights(
        w1_q,
        w1_scale,
        w1_global,
        w2_q,
        w2_scale,
        w2_global,
        **common,
    )


def _reference_moe(
    activations: MoEActivationPack,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation,
) -> torch.Tensor:
    hidden_states = activations.hidden_states_q
    topk_ids = activations.topk_ids
    topk_weights = activations.topk_weights
    assert topk_ids is not None and topk_weights is not None
    intermediate_size = w2.shape[2]
    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token_ids, slots = torch.where(topk_ids == expert)
        if token_ids.numel() == 0:
            continue
        gemm1 = (hidden_states[token_ids].float() @ w1[expert].float().T).to(
            torch.bfloat16
        )
        if isinstance(activation, SwiGLU):
            up, gate = gemm1.split(intermediate_size, dim=-1)
            intermediate = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        else:
            intermediate = F.relu(gemm1.float()).square().to(torch.bfloat16)
        expert_output = intermediate.float() @ w2[expert].float().T
        result.index_add_(
            0,
            token_ids,
            expert_output * topk_weights[token_ids, slots, None],
        )
    return result.to(torch.bfloat16)


def _config_for_backend(args, activation, backend_config) -> MoEConfig:
    quant_variant = (
        QuantVariant.BF16 if args.quant_variant == "bf16" else QuantVariant.NVFP4
    )
    return MoEConfig(
        routing=RoutingConfig(num_experts=args.num_experts, top_k=args.top_k),
        quant=QuantConfig(variant=quant_variant),
        experts=ExpertConfig(intermediate_size=args.intermediate_size),
        activation=activation,
        backend=BackendOptions((backend_config,)),
        finalize=MoEFinalizeConfig(do_finalize=True, use_fused_finalize=True),
        execution=ExecutionConfig(
            enable_pdl=False,
            tune_max_num_tokens=args.num_tokens,
        ),
    )


def _choose_tactic(args, runner, inputs: list[torch.Tensor]):
    if not args.autotune and args.autotune_cache is None:
        return -1
    with autotune(args.autotune, cache=args.autotune_cache):
        _, tactic = AutoTuner.get().choose_one(
            custom_op=f"moe_{runner.backend_key}",
            runners=[runner],
            tuning_config=runner.tuning_config,
            inputs=inputs,
        )
    return tactic


def _measure_runner(args, runner, inputs: list[torch.Tensor], tactic: Any):
    runner.forward(inputs, tactic=tactic, do_preparation=True)
    torch.cuda.synchronize()
    output = runner.forward(inputs, tactic=tactic).detach().clone()

    def run(*profile_inputs):
        return runner.forward(list(profile_inputs), tactic=tactic)

    times = bench_gpu_time(
        fn=run,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=not args.no_cuda_graph,
        num_iters_within_graph=1,
        cold_l2_cache=True,
        input_args=tuple(inputs),
    )
    return output, float(np.median(times)), float(np.std(times))


def run_unified_moe_test(args):
    """Benchmark each supported unified MoE backend on identical inputs."""
    if args.verbose >= 1:
        print("[INFO] Running unified_moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    if args.generate_repro_command:
        print(f"[INFO] To reproduce this test case, run: {args.repro_command}")
    if args.enable_pdl:
        raise ValueError("unified_moe requires PDL disabled for backend parity")

    device = get_device(args)
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype is not torch.bfloat16:
        raise ValueError("unified_moe currently requires --input_dtype bfloat16")
    try:
        activation = _ACTIVATIONS[args.activation_type]()
    except KeyError:
        supported = ", ".join(member.name for member in _ACTIVATIONS)
        raise ValueError(
            f"unified_moe supports activations [{supported}], got "
            f"{args.activation_type.name}"
        ) from None

    activations, w1, w2 = _canonical_inputs(args, activation, device)
    reference = (
        _reference_moe(activations, w1, w2, activation) if args.refcheck else None
    )
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    assert activations.topk_ids is not None
    active_experts = int(activations.topk_ids.unique().numel())
    results = []

    for backend in args.backends:
        config_type = _BACKEND_CONFIGS[(args.quant_variant, backend)]
        if not config_type.supported(arch):
            print(
                f"[INFO] {backend} does not support {args.quant_variant} "
                f"unified MoE on SM{arch}; skipping."
            )
            continue

        backend_config = config_type()
        config = _config_for_backend(args, activation, backend_config)
        try:
            layer = MoELayer(config, device=device)
            runner = layer.runners[0]
            view = _prepare_weight_view(
                backend,
                args.quant_variant,
                config_type,
                w1,
                w2,
                activation,
                args,
                device,
            )
            weights = MoEWeightPack()
            weights.prepare_for(runner.backend_key, view)
            inputs = runner.pack_inputs(activations, weights)
            # Like mm_fp4, execute the fallback once to reject configurations
            # that pass the coarse architecture check but fail at runtime.
            runner.forward(inputs, tactic=-1, do_preparation=True)
            torch.cuda.synchronize()
        except (NotImplementedError, RuntimeError, TypeError, ValueError) as error:
            print(
                f"[INFO] {backend} does not support this configuration: "
                f"{type(error).__name__}: {error}"
            )
            continue

        tactic = _choose_tactic(args, runner, inputs)
        output, median_time, std_time = _measure_runner(args, runner, inputs, tactic)
        backend_label = f"{backend}_autotune" if args.autotune else backend

        refcheck_passed: bool | str = ""
        if reference is not None:
            rtol, atol = (3e-2, 5e-1) if args.quant_variant == "bf16" else (0.25, 1.0)
            try:
                torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
                refcheck_passed = True
            except AssertionError:
                refcheck_passed = False
                print(f"[ERROR] {runner.backend_key} failed the shared reference check")
                if not args.allow_output_mismatch:
                    raise

        tflops = calculate_moe_tflops(
            args.num_tokens,
            args.hidden_size,
            args.intermediate_size,
            args.num_experts,
            args.top_k,
            median_time,
            is_gated=activation.is_gated,
        )
        weight_format = "nvfp4" if args.quant_variant == "nvfp4" else None
        weight_dtype = torch.uint8 if weight_format else torch.bfloat16
        tb_per_sec = calculate_moe_kernel_bandwidth(
            args.num_tokens,
            args.hidden_size,
            args.intermediate_size,
            args.num_experts,
            args.top_k,
            median_time,
            torch.bfloat16,
            weight_dtype,
            input_format=None,
            weight_format=weight_format,
            routing_logits_dtype=None,
            active_experts=active_experts,
            verbose=args.verbose,
            is_gated=activation.is_gated,
        )
        print_perf_metrics(backend_label, median_time, std_time, tflops, tb_per_sec)

        current = defaultdict(str)
        current.update(
            routine=args.routine,
            median_time=median_time,
            std_time=std_time,
            tflops=tflops,
            tb_per_sec=tb_per_sec,
            backend=backend_label,
            resolved_backend=backend,
            num_tokens=args.num_tokens,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            routing_method="precomputed",
            input_dtype=args.input_dtype,
            weight_dtype="nvfp4" if weight_format else "bfloat16",
            activation_type=activation.type.name,
            quant_variant=args.quant_variant,
            autotune=args.autotune,
            tactic=repr(tuple(tactic) if isinstance(tactic, (tuple, list)) else tactic),
            refcheck_passed=refcheck_passed,
            fp4_mode="nvfp4" if weight_format else "",
            cold_l2_cache=True,
        )
        results.append(current)

    if not results:
        print("[ERROR] No unified MoE backends passed runtime validation.")
    return results
