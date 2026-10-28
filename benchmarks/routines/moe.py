import inspect
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import flashinfer

try:
    from flashinfer import ActivationType
except ImportError:
    # ActivationType was not exported from the top-level package until 0.6.3
    from flashinfer.fused_moe.core import ActivationType

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    alphamoe_nvfp4_aligned_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    cutlass_fused_moe,
    fused_topk_deepseek,
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.tllm_enums import RoutingMethodType, is_gated_activation
from flashinfer import fp4_quantize, mxfp8_quantize
from flashinfer.tllm_enums import SfLayout
from flashinfer.testing.utils import (
    bench_gpu_time,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    enum_type,
    get_device,
    print_perf_metrics,
    filter_backends_by_compute_capability,
    warn_if_pdl_unsupported,
)
from .moe_utils import (
    calculate_fp4_global_scale,
    quantize_fp4,
    quantize_fp4_batched,
    quantize_fp8,
    calculate_moe_tflops,
    calculate_moe_kernel_bandwidth,
    compute_routing,
    generate_moe_weights,
    add_common_moe_args,
    process_fp8_weight_layout,
    create_moe_output_scale_scalars,
    pack_topk_ids,
    FLOAT8_E4M3_MAX,
    FLOAT4_E2M1_MAX,
    SF_VEC_SIZE,
)

# Before 0.6.3, MoE APIs used "gated_act_type" (SwiGlu=0, GeGlu=1) instead of
# "activation_type" (Swiglu=3, Geglu=4). Some prior APIs omit the parameter entirely.
_ACTIVATION_TO_GATED_ACT = {
    ActivationType.Swiglu: 0,
    ActivationType.Geglu: 1,
}


# Flat ActivationType -> typed activation of the library's prepare_* helpers
# (they choose the gated / non-gated weight permutation from it).
_UNIFIED_ACTIVATIONS = {
    ActivationType.Swiglu: SwiGLU,
    ActivationType.SwigluStep: SwiGLUStep,
    ActivationType.Geglu: GeGLU,
    ActivationType.GegluTanh: GeGLUTanh,
    ActivationType.Situ: SiTU,
    ActivationType.Relu2: ReLU2,
    ActivationType.Identity: Identity,
    ActivationType.Gelu: GELU,
    ActivationType.Relu: ReLU,
    ActivationType.Silu: SiLU,
}

# Activations accepted by cute_dsl_fp4_block_scale_moe (CuteDslMoEWrapper).
_CUTE_DSL_FLAT_ACTIVATIONS = (
    ActivationType.Swiglu,
    ActivationType.GegluTanh,
    ActivationType.Relu2,
)

# CLI spelling -> RoutingMethodType value. "hash" (DeepSeek-V4 hash_topk) has
# no RoutingMethodType; it maps to -1 and is accepted by moe_routing only.
_ROUTING_METHOD_NAME_TO_TYPE = {
    "default": int(RoutingMethodType.Default),
    "renormalize": int(RoutingMethodType.Renormalize),
    "deepseek_v3": int(RoutingMethodType.DeepSeekV3),
    "llama4": int(RoutingMethodType.Llama4),
    "renormalize_naive": int(RoutingMethodType.RenormalizeNaive),
    "topk": int(RoutingMethodType.TopK),
    "sigmoid_renorm": int(RoutingMethodType.SigmoidRenorm),
    "minimax2": int(RoutingMethodType.MiniMax2),
    "sigmoid": int(RoutingMethodType.Sigmoid),
    "topk_sigmoid": int(RoutingMethodType.TopKSigmoid),
    "hash": -1,
}


def _activation_kwarg(fn, activation_type: ActivationType) -> dict:
    """Return the correct activation keyword argument for *fn* in the installed version."""
    sig = inspect.signature(fn)
    if "activation_type" in sig.parameters:
        return {"activation_type": activation_type.value}
    if "gated_act_type" in sig.parameters:
        if activation_type not in _ACTIVATION_TO_GATED_ACT:
            raise ValueError(
                f"Activation type {activation_type.name} is not supported by the "
                f"installed flashinfer version (pre-0.6.3 only supports "
                f"{[k.name for k in _ACTIVATION_TO_GATED_ACT]})"
            )
        return {"gated_act_type": _ACTIVATION_TO_GATED_ACT[activation_type]}
    return {}


def run_moe_test(args):
    """
    Run a MOE test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "trtllm_fp4_block_scale_moe":
        return testTrtllmFp4BlockScaleMoe(args)
    elif args.routine == "trtllm_fp8_block_scale_moe":
        return testTrtllmFp8BlockScaleMoe(args)
    elif args.routine == "trtllm_fp8_per_tensor_scale_moe":
        return testTrtllmFp8PerTensorScaleMoe(args)
    elif args.routine == "cutlass_fused_moe":
        return testCutlassFusedMoe(args)
    elif args.routine == "cute_dsl_fp4_block_scale_moe":
        return testCuteDslFp4BlockScaleMoe(args)
    elif args.routine == "cute_dsl_bf16_moe":
        return testCuteDslBf16Moe(args)
    elif args.routine == "b12x_fused_moe":
        return testB12xFusedMoe(args)
    elif args.routine == "alphamoe_nvfp4_aligned_moe":
        return testAlphaMoeNvfp4AlignedMoe(args)
    elif args.routine == "bgmv_moe":
        return testBgmvMoe(args)
    elif args.routine == "unified_nvfp4_moe":
        return testUnifiedNvfp4Moe(args)
    elif args.routine == "trtllm_bf16_moe":
        return testTrtllmBf16Moe(args)
    elif args.routine == "trtllm_mxint4_block_scale_moe":
        return testTrtllmMxint4BlockScaleMoe(args)
    elif args.routine == "moe_routing":
        return testMoeRouting(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_moe_args(line, parser):
    """
    Parse command line arguments for MOE test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    # moe_routing times the router only and has no expert FFN.
    routing_only = "moe_routing" in line
    add_common_moe_args(parser, hidden_size_required=not routing_only)
    # Note: num_tokens/hidden_size is added by add_common_moe_args
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=not routing_only,
        help="Intermediate dimension size (not used for bgmv_moe or moe_routing).",
    )
    # Note: num_experts/top_k is added by add_common_moe_args
    parser.add_argument(
        "--n_group",
        type=int,
        required=False,
        default=None,
        help="Number of expert groups (for DeepSeek routing). Only used with DeepSeekV3 routing method.",
    )
    parser.add_argument(
        "--topk_group",
        type=int,
        required=False,
        default=None,
        help="Number of groups to consider for top-k routing. Only used with DeepSeekV3 routing method.",
    )
    parser.add_argument(
        "--routed_scaling_factor",
        type=float,
        required=False,
        default=2.5,
        help="Scaling factor for routing.",
    )
    parser.add_argument(
        "--block_m",
        type=int,
        required=False,
        default=8,
        help="Aligned routing-plan block size (AlphaMoE only; default: 8).",
    )
    parser.add_argument(
        "--local_expert_offset",
        type=int,
        required=False,
        default=0,
        help="Offset of local experts in global expert space.",
    )
    parser.add_argument(
        "--local_num_experts",
        type=int,
        required=False,
        default=None,
        help="Number of experts handled by this device. Defaults to num_experts.",
    )
    parser.add_argument(
        "--routing_method",
        type=str,
        required=False,
        default="deepseek_v3",
        choices=list(_ROUTING_METHOD_NAME_TO_TYPE),
        help=(
            "Routing method (RoutingMethodType): "
            + " | ".join(_ROUTING_METHOD_NAME_TO_TYPE)
            + ". 'hash' is DeepSeek-V4 hash routing and is only valid for moe_routing."
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help=(
            "Backends to benchmark. trtllm_* routines: trtllm (default) and/or "
            "prims-ts. moe_routing: trtllm, dsv3, alphamoe, hash (default: "
            "every backend that implements --routing_method)."
        ),
    )
    parser.add_argument(
        "--routing_input",
        type=str,
        required=False,
        default="logits",
        choices=["logits", "precomputed"],
        help=(
            "trtllm_* routines: logits runs the in-kernel router on routing logits; "
            "precomputed calls the *_routed_moe API with packed top-k ids computed "
            "by a torch reference of --routing_method."
        ),
    )
    parser.add_argument(
        "--fp8_quant_type",
        type=str,
        required=False,
        default="deepseek",
        choices=["deepseek", "mxfp8"],
        help=(
            "trtllm_fp8_block_scale_moe weight/activation format: deepseek "
            "(128x128 FP8 block scales) or mxfp8 (MXFP8 x MXFP8)."
        ),
    )
    parser.add_argument(
        "--fp8_scale_granularity",
        type=str,
        required=False,
        default="per_tensor",
        choices=["per_tensor", "per_channel"],
        help=(
            "trtllm_fp8_per_tensor_scale_moe weight scale: per_tensor or "
            "per_channel (trtllm_fp8_per_channel_scale_moe)."
        ),
    )
    parser.add_argument(
        "--swiglu_alpha",
        type=float,
        required=False,
        default=None,
        help="SwiGLU alpha (gpt-oss style) for trtllm_fp4_block_scale_moe and cutlass_fused_moe.",
    )
    parser.add_argument(
        "--swiglu_beta",
        type=float,
        required=False,
        default=None,
        help="SwiGLU beta (gpt-oss style) for trtllm_fp4_block_scale_moe and cutlass_fused_moe.",
    )
    parser.add_argument(
        "--swiglu_limit",
        type=float,
        required=False,
        default=None,
        help="SwiGLU clamp limit (gpt-oss style) for trtllm_fp4_block_scale_moe and cutlass_fused_moe.",
    )
    parser.add_argument(
        "--gemm_bias",
        action="store_true",
        default=False,
        help="Add FC1/FC2 expert biases (trtllm_fp4_block_scale_moe only).",
    )
    parser.add_argument(
        "--valid_hidden_size",
        type=int,
        required=False,
        default=None,
        help=(
            "trtllm_* routines: unpadded hidden size when --hidden_size is the padded "
            "weight/activation width (e.g. GPT-OSS: --hidden_size 3072 "
            "--valid_hidden_size 2880)."
        ),
    )
    parser.add_argument(
        "--valid_intermediate_size",
        type=int,
        required=False,
        default=None,
        help=(
            "trtllm_* routines: unpadded intermediate size when --intermediate_size is "
            "padded (e.g. GPT-OSS: --intermediate_size 2944 --valid_intermediate_size 2880)."
        ),
    )
    parser.add_argument(
        "--num_fused_shared_experts",
        type=int,
        required=False,
        default=0,
        help="Fused shared experts appended to the routed experts (trtllm FP4/FP8 block, DeepSeekV3 routing only).",
    )
    parser.add_argument(
        "--alphamoe_routing",
        type=str,
        required=False,
        default="aligned",
        choices=["aligned", "routed"],
        help=(
            "alphamoe_nvfp4_aligned_moe: aligned times the MoE over a pre-built "
            "aligned plan; routed times alphamoe_nvfp4_routed_moe (route + align + MoE)."
        ),
    )
    parser.add_argument(
        "--tile_tokens_dim",
        type=int,
        required=False,
        default=8,
        help="moe_routing trtllm backend: token tile of the downstream grouped GEMM.",
    )
    parser.add_argument(
        "--hash_vocab_size",
        type=int,
        required=False,
        default=129280,
        help="moe_routing hash backend: vocabulary size of the token-to-expert table.",
    )
    parser.add_argument(
        "--use_shuffled_weight",
        action="store_true",
        default=False,
        help="Whether to use shuffled weight layout.",
    )
    parser.add_argument(
        "--weight_layout",
        type=int,
        required=False,
        default=0,
        choices=[0, 1, 2],
        help="Weight layout: 0=MajorK, 1=MajorMn, 2=BlockMajorK.",
    )
    parser.add_argument(
        "--use_routing_bias",
        action="store_true",
        default=False,
        help="Whether to use routing bias.",
    )
    parser.add_argument(
        "--use_routing_scales_on_input",
        action="store_true",
        default=False,
        help="Whether to use routing scales on input (for Llama4 routing).",
    )
    # Note: input_dtype is added by add_common_moe_args
    parser.add_argument(
        "--weight_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the weights (before quantization).",
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
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Enable autotuner warmup for supported routines (trtllm_fp4_block_scale_moe and cutlass_fused_moe)."
        ),
    )
    parser.add_argument(
        "--fp4_mode",
        type=str,
        required=False,
        default="nvfp4",
        choices=["nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"],
        help=(
            "FP4 quantization mode for trtllm_fp4_block_scale_moe: "
            "nvfp4 (NvFP4 weights + NvFP4 hidden states, block_size=16), "
            "mxfp4_mxfp8 (MXFP4 weights + MXFP8 hidden states, block_size=32), "
            "mxfp4_bf16 (MXFP4 weights + BF16 hidden states, block_size=32). "
            "Default: nvfp4"
        ),
    )

    # CuTe DSL / b12x MoE specific
    parser.add_argument(
        "--use_functional_api",
        action="store_true",
        default=False,
        help=(
            "Use the functional MoE API instead of the wrapper class: "
            "cute_dsl_fused_moe vs CuteDslMoEWrapper for "
            "cute_dsl_fp4_block_scale_moe, and b12x_fused_moe vs B12xMoEWrapper for "
            "b12x_fused_moe. Useful for verifying that the wrapper's workspace cache "
            "eliminates per-call allocation overhead."
        ),
    )
    parser.add_argument(
        "--b12x_quant_mode",
        type=str,
        required=False,
        default="nvfp4",
        choices=["nvfp4", "w4a16"],
        help=(
            "Quantization mode for b12x_fused_moe: nvfp4 (W4A4, FP4 weights + "
            "FP4 activations) or w4a16 (FP4 weights + BF16 activations). "
            "Default: nvfp4"
        ),
    )

    # CUTLASS fused MoE specific
    parser.add_argument(
        "--cutlass_variant",
        type=str,
        required=False,
        default="base",
        choices=["base", "fp8", "nvfp4"],
        help="Variant for cutlass_fused_moe benchmark: base (no quant), fp8 (per-tensor), nvfp4 (fp4 blockscale)",
    )
    parser.add_argument(
        "--quantized_input",
        action="store_true",
        default=False,
        help="Quantize input activations (only used for nvfp4).",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        required=False,
        default=1,
        help="Tensor parallel size for cutlass_fused_moe.",
    )
    parser.add_argument(
        "--tp_rank",
        type=int,
        required=False,
        default=0,
        help="Tensor parallel rank for cutlass_fused_moe.",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        required=False,
        default=1,
        help="Expert parallel size for cutlass_fused_moe.",
    )
    parser.add_argument(
        "--ep_rank",
        type=int,
        required=False,
        default=0,
        help="Expert parallel rank for cutlass_fused_moe.",
    )

    # BGMV MoE specific arguments
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        default=32,
        help="LoRA rank for bgmv_moe benchmark.",
    )
    parser.add_argument(
        "--num_loras",
        type=int,
        required=False,
        default=8,
        help="Number of LoRA adapters for bgmv_moe benchmark.",
    )
    parser.add_argument(
        "--num_slices",
        type=int,
        required=False,
        default=1,
        help="Number of weight slices for bgmv_moe benchmark (e.g., 2 for gate+up).",
    )

    args = parser.parse_args(line)

    # Normalize routing method (map string to internal int expected by kernels)
    args.routing_method_type = _ROUTING_METHOD_NAME_TO_TYPE[args.routing_method]
    if args.routing_method == "hash" and args.routine != "moe_routing":
        raise ValueError("--routing_method hash is only valid for moe_routing")

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def create_trtllm_moe_test_data(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    routing_method_type: int,
    use_routing_bias: bool,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    device: torch.device,
    moe_kernel_type: str = "fp8_per_tensor",
    is_gated: bool = True,
    num_weight_experts: Optional[int] = None,
):
    """
    ``num_weight_experts`` is the number of expert weight slices to build: the
    launcher requires ``local_num_experts`` (+ fused shared experts) rows, so an
    EP-local run passes that instead of the global ``num_experts``.
    ``is_gated=False`` (ReLU2) builds a single-projection GEMM1.
    Create test data for TensorRT-LLM fused MoE benchmarking (trtllm_*_moe APIs).

    This helper prepares inputs for the TensorRT-LLM fused MoE kernels exposed via
    flashinfer.fused_moe (e.g., trtllm_fp4_block_scale_moe, trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe). It is NOT used for CUTLASS MoE benchmarks,
    which construct their own inputs specific to the CUTLASS path.

    Returns:
        Tuple of tensors needed for trtllm fused MoE computation
    """
    # Create routing logits - dtype depends on both routing method AND MOE kernel type
    # Different MOE kernels have different routing_logits dtype requirements:

    if moe_kernel_type == "fp8_block_scale":
        # DeepSeekV3 routing uses float32, others use bfloat16
        routing_dtype = torch.float32 if routing_method_type == 2 else torch.bfloat16
        routing_logits = torch.randn(
            (num_tokens, num_experts), device=device, dtype=routing_dtype
        )
    elif moe_kernel_type == "fp8_per_tensor":
        # FP8 per-tensor MOE dtype depends on use_routing_scales_on_input parameter
        # For Llama4: use_routing_scales_on_input=True -> bfloat16
        # For others: use_routing_scales_on_input=False -> float32
        routing_dtype = torch.bfloat16 if routing_method_type == 3 else torch.float32
        routing_logits = torch.randn(
            (num_tokens, num_experts), device=device, dtype=routing_dtype
        )
    elif moe_kernel_type in ("fp4_block_scale", "bf16", "mxint4"):
        # FP4 block scale MOE follows the test pattern: float32 for DeepSeekV3, bfloat16 for others
        routing_dtype = torch.float32 if routing_method_type == 2 else torch.bfloat16
        routing_logits = torch.randn(
            (num_tokens, num_experts), device=device, dtype=routing_dtype
        )
    else:
        raise ValueError(f"Unknown MOE kernel type: {moe_kernel_type}")

    # Create routing bias if needed - always bfloat16
    routing_bias = None
    if use_routing_bias:
        # Use uniform routing bias for less skewed expert distribution
        routing_bias = (
            torch.ones(num_experts, device=device, dtype=torch.bfloat16) * 0.1
        )

    # Create hidden states - always start with bfloat16 for proper quantization
    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
    )

    # Create weights - always start with bfloat16 for proper quantization
    gemm1_weights, gemm2_weights = generate_moe_weights(
        num_weight_experts or num_experts,
        hidden_size,
        intermediate_size,
        device,
        dtype=torch.bfloat16,
        is_gated=is_gated,
    )

    return routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights


def _reference_routing(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    top_k: int,
    routing_method_type: int,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    routed_scaling_factor: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference of every ``RoutingMethodType`` (see flashinfer/tllm_enums.py).

    Returns ``(topk_ids int32, topk_weights float32)``, both ``[num_tokens, top_k]``.
    It feeds ``--routing_input precomputed``, the bandwidth model's active-expert
    count, and the moe_routing cross-check. It is deliberately independent of
    ``fused_topk_deepseek``, whose kernel limits (top_k <= 8, ...) are not the
    fused-MoE routing limits.
    """
    logits = routing_logits.float()
    method = RoutingMethodType(routing_method_type)
    if method in (RoutingMethodType.DeepSeekV3, RoutingMethodType.MiniMax2):
        scores = torch.sigmoid(logits)
        biased = scores
        if routing_bias is not None:
            biased = scores + routing_bias.float()
        if method == RoutingMethodType.DeepSeekV3 and n_group and n_group > 1:
            num_tokens, num_experts = biased.shape
            grouped = biased.view(num_tokens, n_group, num_experts // n_group)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            keep = torch.zeros_like(group_scores, dtype=torch.bool)
            keep.scatter_(1, group_scores.topk(topk_group, dim=-1).indices, True)
            biased = biased.masked_fill(
                ~keep.repeat_interleave(num_experts // n_group, dim=1), float("-inf")
            )
        ids = biased.topk(top_k, dim=-1).indices
        weights = scores.gather(1, ids)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        if method == RoutingMethodType.DeepSeekV3 and routed_scaling_factor:
            weights = weights * routed_scaling_factor
    elif method in (RoutingMethodType.Default, RoutingMethodType.RenormalizeNaive):
        weights, ids = torch.softmax(logits, dim=-1).topk(top_k, dim=-1)
        if method == RoutingMethodType.RenormalizeNaive:
            weights = weights / weights.sum(dim=-1, keepdim=True)
    elif method == RoutingMethodType.Renormalize:
        values, ids = logits.topk(top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
    elif method in (RoutingMethodType.Llama4, RoutingMethodType.TopKSigmoid):
        values, ids = logits.topk(top_k, dim=-1)
        weights = torch.sigmoid(values)
    elif method in (RoutingMethodType.Sigmoid, RoutingMethodType.SigmoidRenorm):
        weights, ids = torch.sigmoid(logits).topk(top_k, dim=-1)
        if method == RoutingMethodType.SigmoidRenorm:
            weights = weights / weights.sum(dim=-1, keepdim=True)
    elif method == RoutingMethodType.TopK:
        weights, ids = logits.topk(top_k, dim=-1)
    else:
        raise ValueError(f"No reference routing for {method!r}")
    return ids.to(torch.int32), weights.float().contiguous()


def _compute_routing_for_method(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    top_k: int,
    routing_method_type: int,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute selected experts based on routing method type.
    Returns only the selected expert indices tensor.

    DeepSeekV3 keeps using ``fused_topk_deepseek`` where that kernel accepts the
    configuration (so existing CSV rows are unchanged) and falls back to the torch
    reference where it does not (e.g. n_group=1 with top_k > 8, which the
    fused MoE accepts but fused_topk_deepseek rejects).
    """
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        if n_group is None or topk_group is None or routed_scaling_factor is None:
            raise ValueError(
                "DeepSeekV3 routing requires n_group, topk_group, and routed_scaling_factor"
            )
        num_tokens = routing_logits.shape[0]
        device = routing_logits.device
        bias = routing_bias
        if bias is None:
            bias = torch.zeros(
                routing_logits.shape[1], device=device, dtype=routing_logits.dtype
            )
        topk_values = torch.empty(num_tokens, top_k, device=device, dtype=torch.float32)
        topk_indices = torch.empty(num_tokens, top_k, device=device, dtype=torch.int32)
        try:
            fused_topk_deepseek(
                scores=routing_logits.float(),
                bias=bias.float(),
                n_group=n_group,
                topk_group=topk_group,
                topk=top_k,
                routed_scaling_factor=routed_scaling_factor,
                topk_values=topk_values,
                topk_indices=topk_indices,
            )
            return topk_indices
        except ValueError:
            pass
    ids, _ = _reference_routing(
        routing_logits,
        routing_bias,
        top_k,
        routing_method_type,
        n_group,
        topk_group,
        routed_scaling_factor,
    )
    return ids


# =============================================================================
# TRT-LLM Gen MoE family: the trtllm_* flat APIs, their *_routed_moe twins
# (--routing_input precomputed) and the signature-compatible Prims-TS
# implementations (--backends prims-ts).
# =============================================================================

# (family, backend, routing_input) -> public API in flashinfer.fused_moe. The
# Prims-TS entry points share the TRT-LLM signatures (backends/prims_ts/*_op.py)
# and are lazy exports, so every API is resolved by name at call time.
_TRTLLM_MOE_APIS = {
    ("fp4", "trtllm", "logits"): "trtllm_fp4_block_scale_moe",
    ("fp4", "trtllm", "precomputed"): "trtllm_fp4_block_scale_routed_moe",
    ("fp4", "prims-ts", "logits"): "prims_ts_fp4_block_scale_moe",
    ("fp4", "prims-ts", "precomputed"): "prims_ts_fp4_block_scale_routed_moe",
    ("fp8_block", "trtllm", "logits"): "trtllm_fp8_block_scale_moe",
    ("fp8_block", "trtllm", "precomputed"): "trtllm_fp8_block_scale_routed_moe",
    ("fp8_block", "prims-ts", "logits"): "prims_ts_fp8_block_scale_moe",
    ("fp8_block", "prims-ts", "precomputed"): "prims_ts_fp8_block_scale_routed_moe",
    ("fp8_per_tensor", "trtllm", "logits"): "trtllm_fp8_per_tensor_scale_moe",
    (
        "fp8_per_tensor",
        "trtllm",
        "precomputed",
    ): "trtllm_fp8_per_tensor_scale_routed_moe",
    ("fp8_per_tensor", "prims-ts", "logits"): "prims_ts_fp8_per_tensor_scale_moe",
    ("fp8_per_channel", "trtllm", "logits"): "trtllm_fp8_per_channel_scale_moe",
    (
        "fp8_per_channel",
        "trtllm",
        "precomputed",
    ): "trtllm_fp8_per_channel_scale_routed_moe",
    ("bf16", "trtllm", "logits"): "trtllm_bf16_moe",
    ("bf16", "trtllm", "precomputed"): "trtllm_bf16_routed_moe",
    ("bf16", "prims-ts", "logits"): "prims_ts_bf16_moe",
    ("bf16", "prims-ts", "precomputed"): "prims_ts_bf16_routed_moe",
    ("mxint4", "trtllm", "logits"): "trtllm_mxint4_block_scale_moe",
    ("mxint4", "trtllm", "precomputed"): "trtllm_mxint4_block_scale_routed_moe",
}


def _select_trtllm_moe_backends(args, family, device):
    """Resolve --backends (default trtllm) against the APIs that exist for this mode."""
    backends = []
    for backend in dict.fromkeys(args.backends or ["trtllm"]):
        if (family, backend, args.routing_input) in _TRTLLM_MOE_APIS:
            backends.append(backend)
        else:
            print(
                f"[WARNING] {backend} has no {args.routing_input}-routing API for "
                f"{args.routine} ({family}). Skipping."
            )
    return filter_backends_by_compute_capability(backends, args.routine, device)


def _trtllm_moe_setup(args, moe_kernel_type, device):
    """Shared geometry, test data and routing for every trtllm_* routine."""
    activation_type = args.activation_type
    is_gated = is_gated_activation(activation_type)
    num_shared = args.num_fused_shared_experts
    if num_shared and args.routing_method_type != RoutingMethodType.DeepSeekV3:
        raise ValueError("--num_fused_shared_experts requires DeepSeekV3 routing")
    if num_shared and args.routing_input == "precomputed":
        raise ValueError(
            "--num_fused_shared_experts is only supported with --routing_input logits"
        )
    local_num_experts = args.local_num_experts or args.num_experts
    # Physical expert rows: the launcher sizes every per-expert tensor as
    # local_num_experts + num_fused_shared_experts.
    weight_rows = local_num_experts + num_shared
    routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights = (
        create_trtllm_moe_test_data(
            args.num_tokens,
            args.hidden_size,
            args.intermediate_size,
            args.num_experts,
            args.routing_method_type,
            args.use_routing_bias,
            dtype_str_to_torch_dtype(args.input_dtype),
            dtype_str_to_torch_dtype(args.weight_dtype),
            device,
            moe_kernel_type=moe_kernel_type,
            is_gated=is_gated,
            num_weight_experts=weight_rows,
        )
    )
    n_group = args.n_group if args.n_group is not None and args.n_group > 0 else None
    topk_group = (
        args.topk_group if args.topk_group is not None and args.topk_group > 0 else None
    )
    routing = dict(
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=args.routed_scaling_factor,
    )
    # Selected experts drive the bandwidth model; the torch reference also feeds
    # --routing_input precomputed.
    selected_experts = _compute_routing_for_method(
        routing_logits,
        routing_bias,
        args.top_k,
        args.routing_method_type,
        **routing,
    )
    routing_ref = None
    if args.routing_input == "precomputed":
        routing_ref = _reference_routing(
            routing_logits,
            routing_bias,
            args.top_k,
            args.routing_method_type,
            **routing,
        )
    return dict(
        activation_type=activation_type,
        is_gated=is_gated,
        local_num_experts=local_num_experts,
        weight_rows=weight_rows,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        selected_experts=selected_experts,
        routing_ref=routing_ref,
        **routing,
    )


def _trtllm_moe_common_kwargs(args, setup, zero_groups=False):
    """Routing/geometry keywords of the logits-routed trtllm_* API."""
    n_group, topk_group = setup["n_group"], setup["topk_group"]
    if zero_groups:
        # FP8 entry points take int64 n_group/topk_group: 0 means "no groups".
        n_group = n_group if n_group is not None else 0
        topk_group = topk_group if topk_group is not None else 0
    kwargs = dict(
        routing_logits=setup["routing_logits"],
        routing_bias=setup["routing_bias"],
        num_experts=args.num_experts,
        top_k=args.top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=args.intermediate_size,
        local_expert_offset=args.local_expert_offset,
        local_num_experts=setup["local_num_experts"],
        routed_scaling_factor=setup["routed_scaling_factor"],
        routing_method_type=args.routing_method_type,
        enable_pdl=args.enable_pdl,
    )
    if args.num_fused_shared_experts:
        kwargs["num_fused_shared_experts"] = args.num_fused_shared_experts
    # Padded GEMM geometry (GPT-OSS): the kernels clamp to the valid extents.
    if args.valid_hidden_size is not None:
        kwargs["valid_hidden_size"] = args.valid_hidden_size
    if args.valid_intermediate_size is not None:
        kwargs["valid_intermediate_size"] = args.valid_intermediate_size
    return kwargs


def _bind_trtllm_moe_api(api_name, kwargs, routing_ref):
    """Bind logits-routed kwargs to ``api_name``, switching to packed top-k ids
    for the *_routed_moe APIs. Raises if a non-default argument is not accepted
    instead of silently dropping it."""
    fn = getattr(flashinfer.fused_moe, api_name)
    kwargs = dict(kwargs)
    if routing_ref is not None:
        kwargs.pop("routing_logits")
        topk_ids, topk_weights = routing_ref
        # Global expert ids: the kernel maps them to the local shard through
        # local_expert_offset (tests/moe/test_trtllm_gen_routed_fused_moe.py).
        kwargs["topk_ids"] = pack_topk_ids(topk_ids, topk_weights, 0)
    params = inspect.signature(fn).parameters
    for name in list(kwargs):
        if name in params:
            continue
        value = kwargs.pop(name)
        # routing_bias is already folded into precomputed ids; Prims-TS takes the
        # linear activation-scale layout without a layout argument.
        benign = (
            value is None
            or value is False
            or name == "routing_bias"
            or (
                name == "hidden_states_scale_layout" and value == SfLayout.layout_linear
            )
        )
        if not benign:
            raise ValueError(f"{api_name} does not accept {name}={value!r}")
    return fn, kwargs


def _run_trtllm_moe_backends(
    args,
    family,
    backends,
    setup,
    kwargs,
    *,
    input_format,
    weight_format,
    csv_fields,
    write_activation_type=True,
):
    """Autotune (optional), time and report ``kwargs`` on every backend.

    Metrics count the work this rank executes: only routed (token, slot) pairs
    whose expert lies in the local shard, plus one pair per token for each fused
    shared expert. TFLOPS use the valid (unpadded) GEMM dims when
    --valid_hidden_size / --valid_intermediate_size are given; bandwidth keeps the
    physical (padded) weight bytes the kernel reads. For EP=1, no shared experts
    and no padding this equals the plain num_tokens * top_k accounting.
    """
    res = []
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)
    cache_path = getattr(args, "autotune_cache", None)
    selected = setup["selected_experts"].long()
    in_shard = (selected >= args.local_expert_offset) & (
        selected < args.local_expert_offset + setup["local_num_experts"]
    )
    num_shared = args.num_fused_shared_experts
    computed_pairs = int(in_shard.sum().item()) + args.num_tokens * num_shared
    active_experts = int(selected[in_shard].unique().numel()) + num_shared
    flop_hidden = args.valid_hidden_size or args.hidden_size
    flop_intermediate = args.valid_intermediate_size or args.intermediate_size
    for backend in backends:
        api_name = _TRTLLM_MOE_APIS[(family, backend, args.routing_input)]
        fn, call = _bind_trtllm_moe_api(api_name, kwargs, setup["routing_ref"])
        tensor_names = [k for k, v in call.items() if isinstance(v, torch.Tensor)]
        static = {k: v for k, v in call.items() if k not in tensor_names}
        input_args = tuple(call[k] for k in tensor_names)

        def run_moe(*tensors, fn=fn, static=static, names=tensor_names):
            return fn(**static, **dict(zip(names, tensors, strict=True)))

        label = backend
        if args.autotune:
            warmup_iters = (
                args.dry_run_iters
                if args.dry_run_iters and args.dry_run_iters > 0
                else 10
            )
            label = f"{backend}_autotune"
            if args.verbose >= 1:
                print(f"[INFO] Autotune warmup for {api_name}: {warmup_iters} iters")
            with autotune(True, cache=cache_path):
                for _ in range(warmup_iters):
                    run_moe(*input_args)
        elif cache_path:
            with autotune(False, cache=cache_path):
                pass

        times = bench_gpu_time(
            fn=run_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
            input_args=input_args,
        )
        median_time = np.median(times)
        std_time = np.std(times)
        # computed_pairs (token, expert) GEMM pairs == num_tokens * top_k for EP=1.
        tflops = calculate_moe_tflops(
            computed_pairs,
            flop_hidden,
            flop_intermediate,
            args.num_experts,
            1,
            median_time,
            is_gated=setup["is_gated"],
        )
        tb_per_sec = calculate_moe_kernel_bandwidth(
            args.num_tokens,
            args.hidden_size,
            args.intermediate_size,
            args.num_experts,
            args.top_k,
            median_time,
            input_dtype,
            weight_dtype,
            input_format=input_format,
            weight_format=weight_format,
            # The *_routed_moe APIs never read routing logits.
            routing_logits_dtype=(
                None
                if args.routing_input == "precomputed"
                else setup["routing_logits"].dtype
            ),
            active_experts=active_experts,
            verbose=args.verbose,
            is_gated=setup["is_gated"],
        )
        print_perf_metrics(label, median_time, std_time, tflops, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = label
            cur_res["num_tokens"] = args.num_tokens
            cur_res["hidden_size"] = args.hidden_size
            cur_res["intermediate_size"] = args.intermediate_size
            cur_res["num_experts"] = args.num_experts
            cur_res["top_k"] = args.top_k
            cur_res["n_group"] = setup["n_group"]
            cur_res["topk_group"] = setup["topk_group"]
            cur_res["routed_scaling_factor"] = setup["routed_scaling_factor"]
            cur_res["local_expert_offset"] = args.local_expert_offset
            cur_res["local_num_experts"] = setup["local_num_experts"]
            cur_res["routing_method"] = args.routing_method
            cur_res["routing_input"] = args.routing_input
            cur_res["num_fused_shared_experts"] = args.num_fused_shared_experts
            cur_res["valid_hidden_size"] = args.valid_hidden_size or ""
            cur_res["valid_intermediate_size"] = args.valid_intermediate_size or ""
            cur_res["use_routing_bias"] = args.use_routing_bias
            cur_res["use_routing_scales_on_input"] = args.use_routing_scales_on_input
            cur_res["input_dtype"] = input_dtype
            cur_res["weight_dtype"] = weight_dtype
            if write_activation_type:
                # trtllm_fp8_block_scale_moe never wrote it; its rows keep the
                # args backfill so existing dashboard keys stay unchanged.
                cur_res["activation_type"] = setup["activation_type"].name
            cur_res["autotune"] = args.autotune
            cur_res.update(csv_fields)
            res.append(cur_res)
    return res


def _gemm1_activation_params(args, rows, device):
    """Per-expert SwiGLU alpha/beta/limit and FC1/FC2 biases (gpt-oss style)."""
    params = {}
    for name, value in (
        ("gemm1_alpha", args.swiglu_alpha),
        ("gemm1_beta", args.swiglu_beta),
        ("gemm1_clamp_limit", args.swiglu_limit),
    ):
        if value is not None:
            params[name] = torch.full(
                (rows,), value, device=device, dtype=torch.float32
            )
    return params


def testTrtllmFp4BlockScaleMoe(args):
    """
    Test trtllm_fp4_block_scale_moe API (TensorRT-LLM fused MoE).

    This test:
    1. Creates quantized FP4 weights and scales
    2. Runs FP4 block scale MOE (TRT-LLM Gen and/or Prims-TS, logits or
       precomputed routing)
    3. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTrtllmFp4BlockScaleMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = _select_trtllm_moe_backends(args, "fp4", device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={args.num_experts}, top_k={args.top_k}"
        )
    setup = _trtllm_moe_setup(args, "fp4_block_scale", device)
    rows = setup["weight_rows"]
    w1_rows = (2 if setup["is_gated"] else 1) * intermediate_size

    # Determine FP4 quantization mode
    fp4_mode = getattr(args, "fp4_mode", "nvfp4")
    is_mxfp4 = fp4_mode in ("mxfp4_mxfp8", "mxfp4_bf16")
    use_ue8m0 = is_mxfp4
    sf_vec_size = SF_VEC_SIZE["mxfp4" if is_mxfp4 else "nvfp4"]

    if args.verbose >= 1:
        print(
            f"[INFO] FP4 mode: {fp4_mode} (use_ue8m0={use_ue8m0}, sf_vec_size={sf_vec_size})"
        )

    # Quantize weights using proper FP4 quantization
    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, _ = quantize_fp4_batched(
        setup["gemm1_weights"], rows, use_ue8m0, True
    )
    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, _ = quantize_fp4_batched(
        setup["gemm2_weights"], rows, use_ue8m0, True
    )

    # Prepare hidden states and scale based on fp4_mode. The trtllm-gen routed
    # GEMM requires the LINEAR activation SF layout ("Tokens need use SF linear
    # layout when being routed"), so both quantized modes quantize non-swizzled.
    # The swizzled layout used to be passed here: it pads rows to 128, so the
    # mxfp4_mxfp8 path raised and the nvfp4 path substituted all-ones scales for
    # num_tokens % 128 != 0 -- and at multiples of 128 both silently read
    # swizzled bytes as linear. Linear is exact at every M.
    hidden_states = setup["hidden_states"]
    if fp4_mode == "mxfp4_bf16":
        hidden_states_fp4 = hidden_states.to(torch.bfloat16)
        hidden_states_scale_linear_fp4 = None
    elif fp4_mode == "mxfp4_mxfp8":
        hs_quant, hs_scale = mxfp8_quantize(hidden_states, False)
        hidden_states_fp4 = hs_quant
        hidden_states_scale_linear_fp4 = hs_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
    else:
        hidden_states_scale_global = calculate_fp4_global_scale(hidden_states)
        hidden_states_fp4_bytes, hidden_states_scale_fp4_bytes, _ = quantize_fp4(
            hidden_states, hidden_states_scale_global, use_ue8m0, False
        )
        hidden_states_fp4 = hidden_states_fp4_bytes.view(torch.uint8).reshape(
            num_tokens, hidden_size // 2
        )
        hidden_states_scale_linear_fp4 = hidden_states_scale_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(num_tokens, hidden_size // sf_vec_size)

    # Prepare weights for kernel (packed uint8 format, scale as float8_e4m3fn)
    # fp4_quantize pads the scale tensor on both dims; the data tensor is unpadded.
    def _sf_outer(dim):
        return (dim + 127) // 128 * 128

    def _sf_inner(dim):
        n = dim // sf_vec_size
        return (n + 3) // 4 * 4

    gemm1_weights_fp4 = gemm1_weights_fp4_bytes.view(torch.uint8).reshape(
        rows, w1_rows, hidden_size // 2
    )
    gemm1_weights_scale = gemm1_scales_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        rows, _sf_outer(w1_rows), _sf_inner(hidden_size)
    )
    gemm2_weights_fp4 = gemm2_weights_fp4_bytes.view(torch.uint8).reshape(
        rows, hidden_size, intermediate_size // 2
    )
    gemm2_weights_scale = gemm2_scales_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        rows, _sf_outer(hidden_size), _sf_inner(intermediate_size)
    )

    # Create scale scalars using shared utility
    output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar = (
        create_moe_output_scale_scalars(rows, device)
    )
    gemm1_bias = gemm2_bias = None
    if args.gemm_bias:
        gemm1_bias = 0.01 * torch.randn(rows, w1_rows, device=device)
        gemm2_bias = 0.01 * torch.randn(rows, hidden_size, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] hidden_states.shape = {hidden_states.shape}")
        print(f"[VVERBOSE] gemm1_weights_fp4.shape = {gemm1_weights_fp4.shape}")
        print(f"[VVERBOSE] gemm2_weights_fp4.shape = {gemm2_weights_fp4.shape}")

    kwargs = _trtllm_moe_common_kwargs(args, setup)
    kwargs.update(
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale_linear_fp4,
        gemm1_weights=gemm1_weights_fp4,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights_fp4,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=gemm2_bias,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        do_finalize=True,
        hidden_states_scale_layout=SfLayout.layout_linear,
        **_activation_kwarg(trtllm_fp4_block_scale_moe, setup["activation_type"]),
    )
    kwargs.update(_gemm1_activation_params(args, rows, device))

    input_format_str = {"nvfp4": "nvfp4", "mxfp4_mxfp8": "mxfp8", "mxfp4_bf16": "bf16"}[
        fp4_mode
    ]
    return _run_trtllm_moe_backends(
        args,
        "fp4",
        backends,
        setup,
        kwargs,
        input_format=input_format_str,
        weight_format="mxfp4" if is_mxfp4 else "nvfp4",
        csv_fields={
            "fp4_mode": fp4_mode,
            "use_shuffled_weight": args.use_shuffled_weight,
            "weight_layout": args.weight_layout,
        },
    )


def testTrtllmFp8BlockScaleMoe(args):
    """
    Test trtllm_fp8_block_scale_moe API (TensorRT-LLM fused MoE).

    --fp8_quant_type deepseek (default): FP8 weights with 128x128 block scales.
    --fp8_quant_type mxfp8: MXFP8 x MXFP8, weights/activations prepared by the
    library's ``prepare_trtllm_fp8_block_{weights,activations}``.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTrtllmFp8BlockScaleMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = _select_trtllm_moe_backends(args, "fp8_block", device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    use_shuffled_weight = args.use_shuffled_weight
    weight_layout = args.weight_layout
    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={args.num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={args.num_experts}, top_k={args.top_k}"
        )
    setup = _trtllm_moe_setup(args, "fp8_block_scale", device)
    rows = setup["weight_rows"]
    w1_rows = (2 if setup["is_gated"] else 1) * intermediate_size

    if args.fp8_quant_type == "mxfp8":
        from flashinfer.fused_moe import QuantConfig, QuantFormat
        from flashinfer.fused_moe.prepare import (
            prepare_trtllm_fp8_block_activations,
            prepare_trtllm_fp8_block_weights,
        )
        from flashinfer.tllm_enums import Fp8QuantizationType, WeightLayout

        quant = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
        view = prepare_trtllm_fp8_block_weights(
            setup["gemm1_weights"],
            setup["gemm2_weights"],
            quant=quant,
            num_local_experts=rows,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=_UNIFIED_ACTIVATIONS[setup["activation_type"]](),
            device=device,
        )
        hidden_states_fp8, hidden_states_scale = prepare_trtllm_fp8_block_activations(
            setup["hidden_states"], quant=quant
        )
        # MXFP8 weights come out shuffled in MajorK (TrtllmFp8BlockRunner).
        use_shuffled_weight = True
        weight_layout = int(WeightLayout.MajorK)
        weights = dict(view)
        quant_kwargs = {"fp8_quantization_type": Fp8QuantizationType.MxFp8}
        weight_format = "mxfp8"
    else:
        gemm1_weights_fp8 = setup["gemm1_weights"].to(torch.float8_e4m3fn)
        gemm2_weights_fp8 = setup["gemm2_weights"].to(torch.float8_e4m3fn)
        # Optionally shuffle weights and convert to BlockMajorK layout to match kernel expectation
        if use_shuffled_weight:
            gemm1_weights_fp8 = torch.stack(
                [
                    process_fp8_weight_layout(
                        w, use_shuffled_weight, weight_layout
                    ).view(torch.uint8)
                    for w in gemm1_weights_fp8
                ]
            ).view(torch.float8_e4m3fn)
            gemm2_weights_fp8 = torch.stack(
                [
                    process_fp8_weight_layout(
                        w, use_shuffled_weight, weight_layout
                    ).view(torch.uint8)
                    for w in gemm2_weights_fp8
                ]
            ).view(torch.float8_e4m3fn)
        # Block scales (values do not affect timing): hidden_states_scale is
        # [hidden_size // 128, num_tokens].
        hidden_states_scale = 2.0 * torch.ones(
            (hidden_size // 128, args.num_tokens), device=device, dtype=torch.float32
        )
        # The FP8 cast of the activations happens once here. It used to sit inside
        # the timed function, so every iteration also timed a BF16->FP8 cast.
        hidden_states_fp8 = setup["hidden_states"].to(torch.float8_e4m3fn)
        weights = dict(
            gemm1_weights=gemm1_weights_fp8,
            gemm1_weights_scale=2.0
            * torch.ones(
                (rows, w1_rows // 128, hidden_size // 128),
                device=device,
                dtype=torch.float32,
            ),
            gemm2_weights=gemm2_weights_fp8,
            gemm2_weights_scale=2.0
            * torch.ones(
                (rows, hidden_size // 128, intermediate_size // 128),
                device=device,
                dtype=torch.float32,
            ),
        )
        quant_kwargs = {}
        weight_format = "fp8"

    kwargs = _trtllm_moe_common_kwargs(args, setup, zero_groups=True)
    kwargs.update(
        hidden_states=hidden_states_fp8,
        hidden_states_scale=hidden_states_scale,
        use_shuffled_weight=use_shuffled_weight,
        weight_layout=weight_layout,
        **weights,
        **quant_kwargs,
    )
    if setup["activation_type"] != ActivationType.Swiglu:
        kwargs.update(
            _activation_kwarg(trtllm_fp8_block_scale_moe, setup["activation_type"])
        )
    return _run_trtllm_moe_backends(
        args,
        "fp8_block",
        backends,
        setup,
        kwargs,
        input_format=weight_format,
        weight_format=weight_format,
        csv_fields={
            "use_shuffled_weight": use_shuffled_weight,
            "weight_layout": weight_layout,
            # Written only for the non-default mode: existing DeepSeek-FP8 rows
            # keep quant_variant="" (rule 1, dashboard key continuity).
            **({"quant_variant": "mxfp8"} if args.fp8_quant_type == "mxfp8" else {}),
        },
        write_activation_type=False,
    )


def testTrtllmFp8PerTensorScaleMoe(args):
    """
    Test trtllm_fp8_per_tensor_scale_moe API (TensorRT-LLM fused MoE).

    --fp8_scale_granularity per_channel benchmarks trtllm_fp8_per_channel_scale_moe
    (per-output-channel weight scales, per-token activation scales).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTrtllmFp8PerTensorScaleMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    per_channel = args.fp8_scale_granularity == "per_channel"
    family = "fp8_per_channel" if per_channel else "fp8_per_tensor"
    backends = _select_trtllm_moe_backends(args, family, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={args.num_tokens}, hidden={args.hidden_size}, "
            f"intermediate={args.intermediate_size}, experts={args.num_experts}, top_k={args.top_k}"
        )
    setup = _trtllm_moe_setup(args, "fp8_per_tensor", device)
    rows = setup["weight_rows"]

    # For FP8 per-tensor scale, create quantized weights and per-tensor scales
    gemm1_weights_fp8 = setup["gemm1_weights"].to(torch.float8_e4m3fn)
    gemm2_weights_fp8 = setup["gemm2_weights"].to(torch.float8_e4m3fn)
    hidden_states_fp8 = setup["hidden_states"].to(torch.float8_e4m3fn)
    output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar = (
        create_moe_output_scale_scalars(rows, device)
    )

    kwargs = _trtllm_moe_common_kwargs(args, setup, zero_groups=not per_channel)
    kwargs.update(
        hidden_states=hidden_states_fp8,
        gemm1_weights=gemm1_weights_fp8,
        gemm2_weights=gemm2_weights_fp8,
        use_routing_scales_on_input=args.use_routing_scales_on_input,
        **_activation_kwarg(trtllm_fp8_per_tensor_scale_moe, setup["activation_type"]),
    )
    if per_channel:
        kwargs.update(
            hidden_states_scale=torch.ones(
                args.num_tokens, 1, device=device, dtype=torch.float32
            ),
            gemm1_per_channel_weight_scale=torch.ones(
                rows, gemm1_weights_fp8.shape[1], device=device, dtype=torch.float32
            ),
            gemm2_per_channel_weight_scale=torch.ones(
                rows, args.hidden_size, device=device, dtype=torch.float32
            ),
            output1_scale_scalar=output1_scales_scalar,
            output1_scale_gate_scalar=output1_scales_gate_scalar,
            output2_scale_scalar=output2_scales_scalar,
        )
    else:
        kwargs.update(
            output1_scales_scalar=output1_scales_scalar,
            output1_scales_gate_scalar=output1_scales_gate_scalar,
            output2_scales_scalar=output2_scales_scalar,
        )
    return _run_trtllm_moe_backends(
        args,
        family,
        backends,
        setup,
        kwargs,
        input_format="fp8",
        weight_format="fp8",
        # Written only for per-channel: existing per-tensor rows keep quant_variant="".
        csv_fields={"quant_variant": "per_channel"} if per_channel else {},
    )


def testTrtllmBf16Moe(args):
    """
    Test trtllm_bf16_moe (TRT-LLM Gen BF16 x BF16 fused MoE, BlockMajorK weights).

    Weights are laid out by the library's ``prepare_trtllm_bf16_weights``; the same
    view feeds ``prims_ts_bf16_moe`` (--backends prims-ts).
    """
    if args.verbose >= 1:
        print("[INFO] Running testTrtllmBf16Moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = _select_trtllm_moe_backends(args, "bf16", device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    from flashinfer.fused_moe.prepare import prepare_trtllm_bf16_weights
    from flashinfer.tllm_enums import WeightLayout

    setup = _trtllm_moe_setup(args, "bf16", device)
    view = prepare_trtllm_bf16_weights(
        setup["gemm1_weights"],
        setup["gemm2_weights"],
        num_local_experts=setup["weight_rows"],
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation=_UNIFIED_ACTIVATIONS[setup["activation_type"]](),
        device=device,
    )
    kwargs = _trtllm_moe_common_kwargs(args, setup)
    kwargs.update(
        hidden_states=setup["hidden_states"],
        use_shuffled_weight=True,
        weight_layout=int(WeightLayout.BlockMajorK),
        **view,
        **_activation_kwarg(trtllm_fp4_block_scale_moe, setup["activation_type"]),
    )
    return _run_trtllm_moe_backends(
        args,
        "bf16",
        backends,
        setup,
        kwargs,
        input_format="bf16",
        weight_format="bf16",
        csv_fields={
            "use_shuffled_weight": True,
            "weight_layout": int(WeightLayout.BlockMajorK),
        },
    )


def testTrtllmMxint4BlockScaleMoe(args):
    """
    Test trtllm_mxint4_block_scale_moe (MxInt4 weights x BF16 activations,
    e.g. Kimi-K2 INT4). Weights are quantized and laid out by the library's
    ``prepare_trtllm_mxint4_weights``. SwiGLU only.
    """
    if args.verbose >= 1:
        print("[INFO] Running testTrtllmMxint4BlockScaleMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = _select_trtllm_moe_backends(args, "mxint4", device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    from flashinfer.fused_moe.prepare import prepare_trtllm_mxint4_weights

    setup = _trtllm_moe_setup(args, "mxint4", device)
    view = prepare_trtllm_mxint4_weights(
        setup["gemm1_weights"],
        setup["gemm2_weights"],
        num_local_experts=setup["weight_rows"],
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation=_UNIFIED_ACTIVATIONS[setup["activation_type"]](),
        device=device,
    )
    kwargs = _trtllm_moe_common_kwargs(args, setup)
    kwargs.update(
        hidden_states=setup["hidden_states"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        **view,
    )
    return _run_trtllm_moe_backends(
        args,
        "mxint4",
        backends,
        setup,
        kwargs,
        input_format="bf16",
        weight_format="mxint4",
        csv_fields={"quant_variant": "mxint4"},
    )


def testCutlassFusedMoe(args):
    """
    Benchmark cutlass_fused_moe (CUTLASS MoE) with variants mirroring tests in tests/moe/test_trtllm_cutlass_fused_moe.py
    Variants:
      - base: no quantization
      - fp8: per-tensor fp8 for weights and activation scale
      - nvfp4: FP4 block-scale weights, optional quantized input
    Supports TP/EP via tp_size/tp_rank and ep_size/ep_rank.
    """
    if args.verbose >= 1:
        print("[INFO] Running testCutlassFusedMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    # Shapes
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    tp_size = getattr(args, "tp_size", 1)
    tp_rank = getattr(args, "tp_rank", 0)
    ep_size = getattr(args, "ep_size", 1)
    ep_rank = getattr(args, "ep_rank", 0)
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []
    backends = ["cutlass"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    # Create base tensors
    activation_type = args.activation_type
    is_gated = is_gated_activation(activation_type)
    w1_rows = (2 if is_gated else 1) * intermediate_size

    torch.manual_seed(args.random_seed)
    x = torch.randn(num_tokens, hidden_size, dtype=input_dtype, device=device)
    w31_weight = (
        torch.randn(
            num_experts,
            w1_rows,
            hidden_size,
            dtype=input_dtype,
            device=device,
        )
        / 10
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=input_dtype,
            device=device,
        )
        / 10
    )

    # Routing
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=input_dtype, device=device
    )
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    if args.verbose >= 2:
        print(f"[VVERBOSE] x.shape = {x.shape}")
        print(f"[VVERBOSE] w31_weight.shape = {w31_weight.shape}")
        print(f"[VVERBOSE] w2_weight.shape = {w2_weight.shape}")

    # Build local weights per EP/TP like tests do
    experts_per_rank = num_experts // max(ep_size, 1)
    expert_start = ep_rank * experts_per_rank
    expert_end = expert_start + experts_per_rank
    w31_ep = w31_weight[expert_start:expert_end, :]
    w2_ep = w2_weight[expert_start:expert_end, :]

    def build_tp_shards(w31_ep_tensor: torch.Tensor, w2_ep_tensor: torch.Tensor):
        if tp_size <= 1:
            return w31_ep_tensor, w2_ep_tensor
        shard = intermediate_size // tp_size
        start = tp_rank * shard
        end = start + shard
        if is_gated:
            # Split w31 into w3 and w1 along intermediate dim
            w3_weight, w1_weight = torch.chunk(w31_ep_tensor, 2, dim=1)
            w3_local = w3_weight[:, start:end, :]
            w1_local = w1_weight[:, start:end, :]
            w31_local = torch.cat([w3_local, w1_local], dim=1)
        else:
            w31_local = w31_ep_tensor[:, start:end, :]
        w2_local = w2_ep_tensor[:, :, start:end]
        return w31_local.contiguous(), w2_local.contiguous()

    w31_local, w2_local = build_tp_shards(w31_ep, w2_ep)

    # Prepare variant-specific inputs (outside of the timed/captured region)
    variant = getattr(args, "cutlass_variant", "base")
    out = torch.empty_like(x)
    # gpt-oss style SwiGLU: per-local-expert float32 alpha/beta/limit.
    cutlass_extra = {
        name: torch.full(
            (w31_local.shape[0],), value, device=device, dtype=torch.float32
        )
        for name, value in (
            ("swiglu_alpha", args.swiglu_alpha),
            ("swiglu_beta", args.swiglu_beta),
            ("swiglu_limit", args.swiglu_limit),
        )
        if value is not None
    }
    if args.gemm_bias:
        raise ValueError("--gemm_bias is not wired for cutlass_fused_moe")

    if variant == "base":

        def run_cutlass(x, selected_experts, routing_weights, w31_local, w2_local, out):
            return cutlass_fused_moe(
                x,
                selected_experts.to(torch.int),
                routing_weights,
                w31_local,
                w2_local,
                input_dtype,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                quant_scales=None,
                output=out,
                enable_pdl=args.enable_pdl,
                **_activation_kwarg(cutlass_fused_moe, activation_type),
                **cutlass_extra,
            )

        input_args_for_bench = (
            x,
            selected_experts,
            routing_weights,
            w31_local,
            w2_local,
            out,
        )

    elif variant == "fp8":
        # Per-tensor FP8 for weights and activation scale
        w31_weight_fp8 = torch.empty_like(w31_local, dtype=torch.float8_e4m3fn)
        w2_weight_fp8 = torch.empty_like(w2_local, dtype=torch.float8_e4m3fn)
        local_num_experts = w31_local.shape[0]
        w31_scales = torch.empty(local_num_experts, 2, dtype=input_dtype, device=device)
        w2_scales = torch.empty(local_num_experts, 1, dtype=input_dtype, device=device)

        # Quantize weights per expert
        for expert_id in range(local_num_experts):
            w31_expert = w31_local[expert_id]
            w2_expert = w2_local[expert_id]
            w31_q, s31 = quantize_fp8(w31_expert)
            w2_q, s2 = quantize_fp8(w2_expert)
            w31_weight_fp8[expert_id].copy_(w31_q)
            w2_weight_fp8[expert_id].copy_(w2_q)
            # Store the same scalar twice to mimic test layout (avoid torch.tensor())
            w31_scales[expert_id, 0] = s31.to(dtype=input_dtype, device=device)
            w31_scales[expert_id, 1] = s31.to(dtype=input_dtype, device=device)
            w2_scales[expert_id, 0] = s2.to(dtype=input_dtype, device=device)

        x_quant, hidden_states_scale = quantize_fp8(x)
        hidden_states_scale_scalar = hidden_states_scale[0].to(device)

        # Note: follow tests quant_scales format
        # [w1_scales * hidden_states_scale, 1.0, 1.0 * w2_scales, hidden_states_scale]
        w1_scales = w31_scales[:, 1]
        one_const = torch.ones((), device=device)
        quant_scales = [
            (w1_scales * hidden_states_scale_scalar).float().squeeze(),
            one_const,
            w2_scales.squeeze().float(),
            hidden_states_scale_scalar,
        ]

        def run_cutlass(
            x_quant,
            selected_experts,
            routing_weights,
            w31_weight_fp8,
            w2_weight_fp8,
            out,
        ):
            return cutlass_fused_moe(
                x_quant,
                selected_experts.to(torch.int),
                routing_weights,
                w31_weight_fp8,
                w2_weight_fp8,
                input_dtype,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                quant_scales=quant_scales,
                output=out,
                enable_pdl=args.enable_pdl,
                **_activation_kwarg(cutlass_fused_moe, activation_type),
                **cutlass_extra,
            )

        input_args_for_bench = (
            x_quant,
            selected_experts,
            routing_weights,
            w31_weight_fp8,
            w2_weight_fp8,
            out,
        )

    elif variant == "nvfp4":
        # NVFP4: FP4 block-scale weights, optional quantized input

        def round_up(x_val, y):
            return (x_val + y - 1) // y * y

        e = w31_local.shape[0]
        n = w2_local.shape[2]  # local intermediate size after TP
        k = hidden_size
        quant_blocksize = 16
        w1_n = w31_local.shape[1]  # 2*n for gated, n for non-gated

        # Weight quantization buffers
        w1_q = torch.empty((e, w1_n, k // 2), device=device, dtype=torch.uint8)
        w2_q = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
        w1_blockscale = torch.empty(
            (e, round_up(w1_n, 128), round_up(k // quant_blocksize, 4)),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        w2_blockscale = torch.empty(
            (e, round_up(k, 128), round_up(n // quant_blocksize, 4)),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        w1_gs = torch.empty((e,), device=device, dtype=torch.float32)
        w2_gs = torch.empty((e,), device=device, dtype=torch.float32)

        # Quantize from local shards
        for expert in range(e):
            w1_src = w31_local[expert]
            # w31 layout is [2n, k]; w2 layout is [k, n]
            w2_src = w2_local[expert].contiguous()  # [hidden_size, n]
            w1_amax = torch.abs(w1_src).max().to(torch.float32)
            w2_amax = torch.abs(w2_src).max().to(torch.float32)
            w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
            w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1_src, w1_gs[expert])
            w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2_src, w2_gs[expert])

        a1_gs = torch.ones((), device=device, dtype=torch.float32)
        a2_gs = torch.ones((), device=device, dtype=torch.float32)

        hidden_states = x
        input_sf = None
        if getattr(args, "quantized_input", False):
            hidden_states, input_sf = fp4_quantize(x, a1_gs)

        quant_scales = [
            a1_gs,
            w1_blockscale.view(torch.int32),
            1.0 / (a1_gs * w1_gs),
            a2_gs,
            w2_blockscale.view(torch.int32),
            1.0 / (a2_gs * w2_gs),
        ]

        def run_cutlass(
            hidden_states, selected_experts, routing_weights, w1_q, w2_q, out
        ):
            return cutlass_fused_moe(
                hidden_states,
                selected_experts.to(torch.int),
                routing_weights,
                w1_q.contiguous().view(torch.long),
                w2_q.contiguous().view(torch.long),
                input_dtype,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                quant_scales=quant_scales,
                input_sf=input_sf,
                output=out,
                enable_pdl=args.enable_pdl,
                **_activation_kwarg(cutlass_fused_moe, activation_type),
                **cutlass_extra,
            )

        input_args_for_bench = (
            hidden_states,
            selected_experts,
            routing_weights,
            w1_q,
            w2_q,
            out,
        )
    else:
        raise ValueError(f"Unknown cutlass_variant: {variant}")

    backend = "cutlass"

    # Optional autotune warmup (supported for CUTLASS fused MoE)
    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        backend = "cutlass_autotune"
        if args.verbose >= 1:
            print(f"[INFO] Autotune warmup for CUTLASS fused MoE: {warmup_iters} iters")
        with autotune(True, cache=cache_path):
            for _ in range(warmup_iters):
                run_cutlass(*input_args_for_bench)
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Measure
    times = bench_gpu_time(
        fn=run_cutlass,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=input_args_for_bench,
    )

    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        is_gated=is_gated,
    )
    tb_per_sec = calculate_moe_kernel_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        input_dtype,
        input_format=variant,
        weight_format=variant,
        routing_logits_dtype=router_logits.dtype,
        active_experts=int(selected_experts.unique().numel()),
        verbose=args.verbose,
        is_gated=is_gated,
    )

    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = tflops
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["num_tokens"] = num_tokens
        cur_res["hidden_size"] = hidden_size
        cur_res["intermediate_size"] = intermediate_size
        cur_res["num_experts"] = num_experts
        cur_res["top_k"] = top_k
        # Routing method/weight layout not applicable; leave defaults
        cur_res["use_shuffled_weight"] = False
        cur_res["weight_layout"] = 0
        cur_res["use_routing_scales_on_input"] = False
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = input_dtype
        # CUTLASS fused MoE specific
        cur_res["cutlass_variant"] = variant
        cur_res["quantized_input"] = args.quantized_input
        cur_res["tp_size"] = tp_size
        cur_res["tp_rank"] = tp_rank
        cur_res["ep_size"] = ep_size
        cur_res["ep_rank"] = ep_rank
        res.append(cur_res)

    return res


def _interleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64, dim: int = -1
) -> torch.Tensor:
    """Interleave linear and gate weights for CuteDSL SwiGLU layout."""
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def _create_nvfp4_moe_test_data(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    device: torch.device,
    backend: str,
    is_gated: bool = True,
):
    """Create NVFP4-quantized test data for CuTe-DSL-family MoE kernels.

    Supports two backends that share the weight-quantization recipe but differ
    in the FC1 weight layout and in whether input activations are pre-quantized:

    - ``backend="cute-dsl"`` (SM100/SM103/SM107 ``CuteDslMoEWrapper``): gated FC1
      weights interleaved via ``_interleave_linear_and_gate`` (non-gated ReLU2 FC1
      is not interleaved), fp4-quantized input activations + scale factors.
    - ``backend="b12x"`` (SM120/SM121 ``B12xMoEWrapper``): either gated (SwiGLU,
      ``[E, 2n, k]``) or non-gated (ReLU2, ``[E, n, k]``) FC1, no interleave,
      bf16 input activations (the kernel fuses quantization internally).

    Routing is computed externally via simple top-k (the wrappers take
    pre-computed ``token_selected_experts`` and ``token_final_scales``).
    """
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    if backend not in ("cute-dsl", "b12x"):
        raise ValueError(f"Unsupported backend: {backend!r}")

    sf_vec_size = 16

    # Input activations: fp4-quantized for cute-dsl, bf16 for b12x
    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 10
    )
    a1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    x_quantized, x_sf = fp4_quantize(
        x_bf16, global_scale=a1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=False
    )
    x_sf = x_sf.unsqueeze(-1)

    # Routing (simple top-k; the wrapper takes pre-computed assignments)
    routing_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights, selected_experts = compute_routing(routing_logits, top_k)
    selected_experts = selected_experts.to(torch.int32)

    # GEMM1 weights
    # Gated (SwiGLU): [E, 2*n, k] — gate + up fused
    # Non-gated (ReLU2, b12x only): [E, n, k] — single FC1 matrix
    w1_rows = (2 if is_gated else 1) * intermediate_size
    w1_bf16 = (
        torch.randn(
            num_local_experts,
            w1_rows,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    # cute-dsl (SM100) expects gated gate/up interleaved; b12x (SM120) does not.
    if backend == "cute-dsl" and is_gated:
        w1_bf16_prepared = _interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
    else:
        w1_bf16_prepared = w1_bf16
    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_flat = w1_bf16_prepared.view(num_local_experts * w1_rows, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat, global_scale=w1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w1_weight = w1_q_flat.view(num_local_experts, w1_rows, hidden_size // 2)
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=w1_rows,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w1_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)

    # GEMM2 weights (down projection)
    w2_bf16 = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat, global_scale=w2_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w2_weight = w2_q_flat.view(num_local_experts, hidden_size, intermediate_size // 2)
    w2_weight_sf = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w2_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)

    fc2_input_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    return {
        "x": x_quantized,
        "x_bf16": x_bf16,
        "x_sf": x_sf,
        "token_selected_experts": selected_experts,
        "token_final_scales": routing_weights,
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale,
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": w2_alpha,
    }


_ALPHAMOE_SF_VEC = 16
_ALPHAMOE_INTERMEDIATE_TILE = 128
_ALPHAMOE_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _make_alphamoe_tight_plan(
    topk_ids: torch.Tensor,
    *,
    block_m: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the tight aligned plan used by the originating Cake benchmark."""

    flat_ids = topk_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
    sentinel = int(flat_ids.numel())
    sorted_positions: list[int] = []
    block_experts: list[int] = []
    for expert in range(num_experts):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().tolist()
        if not positions:
            continue
        padded = ((len(positions) + block_m - 1) // block_m) * block_m
        sorted_positions.extend(positions)
        sorted_positions.extend([sentinel] * (padded - len(positions)))
        block_experts.extend([expert] * (padded // block_m))
    device = topk_ids.device
    return (
        torch.tensor(sorted_positions, dtype=torch.int32, device=device),
        torch.tensor(block_experts, dtype=torch.int32, device=device),
        torch.tensor([len(sorted_positions)], dtype=torch.int32, device=device),
    )


def _random_alphamoe_nvfp4(
    leading_shape: tuple[int, ...],
    columns: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    min_scale_exp: float,
    max_scale_exp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.randint(
        0,
        256,
        (*leading_shape, columns // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scale_exp = torch.rand(
        (*leading_shape, columns // _ALPHAMOE_SF_VEC),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    scales = torch.exp2(min_scale_exp + scale_exp * (max_scale_exp - min_scale_exp)).to(
        torch.float8_e4m3fn
    )
    return packed, scales.contiguous()


def _make_alphamoe_nvfp4_data(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
    device: torch.device,
    seed: int,
) -> dict:
    gemm1_out_size = 2 * intermediate_size
    generator = torch.Generator(device=device).manual_seed(seed)
    x, x_scale = _random_alphamoe_nvfp4(
        (num_tokens,),
        hidden_size,
        generator=generator,
        device=device,
        min_scale_exp=-3.0,
        max_scale_exp=-2.0,
    )
    w1, w1_scale = _random_alphamoe_nvfp4(
        (num_experts, gemm1_out_size),
        hidden_size,
        generator=generator,
        device=device,
        min_scale_exp=-5.0,
        max_scale_exp=-4.0,
    )
    w2, w2_scale = _random_alphamoe_nvfp4(
        (num_experts, hidden_size),
        intermediate_size,
        generator=generator,
        device=device,
        min_scale_exp=-5.0,
        max_scale_exp=-4.0,
    )
    routing_scores = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_ids = torch.topk(routing_scores, top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            (num_tokens, top_k),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _make_alphamoe_tight_plan(
        topk_ids,
        block_m=block_m,
        num_experts=num_experts,
    )
    return {
        "M": num_tokens,
        "N": gemm1_out_size,
        "K": hidden_size,
        "E": num_experts,
        "top_k": top_k,
        "block_m": block_m,
        "scaling_factor": routed_scaling_factor,
        "x": x,
        "x_scale": x_scale,
        "w1": w1,
        "w1_scale": w1_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "output1_scale_gate_scalar": torch.ones(
            num_experts, dtype=torch.float32, device=device
        ),
        "output1_scale_scalar": torch.ones(
            num_experts, dtype=torch.float32, device=device
        ),
        "output2_scale_scalar": torch.ones(
            num_experts, dtype=torch.float32, device=device
        ),
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "out": torch.zeros(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
    }


def _decode_alphamoe_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    columns: int,
) -> torch.Tensor:
    codes = torch.empty(
        (*packed.shape[:-1], columns), dtype=torch.uint8, device=packed.device
    )
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = (packed >> 4) & 0x0F
    lut = torch.tensor(_ALPHAMOE_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    return (
        lut[codes.to(torch.int64)]
        * scale.float().repeat_interleave(_ALPHAMOE_SF_VEC, dim=-1)[..., :columns]
    )


def _quantize_alphamoe_nvfp4_intermediate(values: torch.Tensor) -> torch.Tensor:
    rows, columns = values.shape
    blocks = values.reshape(rows, columns // _ALPHAMOE_SF_VEC, _ALPHAMOE_SF_VEC)
    scale = (blocks.abs().amax(dim=-1) * (1.0 / 6.0)).to(torch.float8_e4m3fn).float()
    normalized = (
        blocks / torch.where(scale == 0, torch.ones_like(scale), scale)[..., None]
    )
    levels = torch.tensor(
        (
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ),
        dtype=torch.float32,
        device=values.device,
    )
    nearest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[nearest] * scale[..., None]).reshape(rows, columns)


def _alphamoe_nvfp4_reference(data: dict) -> torch.Tensor:
    """Independent source-ordered oracle for ``--refcheck``."""

    m, n, k = data["M"], data["N"], data["K"]
    top_k = data["top_k"]
    intermediate = n // 2
    x = _decode_alphamoe_nvfp4(data["x"], data["x_scale"], columns=k)
    pair_expert = torch.full((m * top_k,), -1, dtype=torch.int64, device=x.device)
    extent = int(data["num_tokens_post_padded"].item())
    sorted_ids = data["sorted_token_ids"][:extent].to(torch.int64)
    for block in range(data["expert_ids"].numel()):
        pairs = sorted_ids[block * data["block_m"] : (block + 1) * data["block_m"]]
        valid = pairs < m * top_k
        pair_expert[pairs[valid]] = data["expert_ids"][block].to(torch.int64)

    output = torch.zeros((m, k), dtype=torch.bfloat16, device=x.device)
    flat_weights = data["topk_weights"].reshape(-1).float()
    for expert in range(data["E"]):
        pairs = torch.nonzero(pair_expert == expert, as_tuple=False).flatten()
        if pairs.numel() == 0:
            continue
        tokens = torch.div(pairs, top_k, rounding_mode="floor")
        w1 = _decode_alphamoe_nvfp4(
            data["w1"][expert], data["w1_scale"][expert], columns=k
        )
        gate_up = torch.zeros((tokens.numel(), n), dtype=torch.float32, device=x.device)
        for base in range(0, k, 64):
            gate_up += x[tokens, base : base + 64] @ w1[:, base : base + 64].transpose(
                0, 1
            )
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        activated = _quantize_alphamoe_nvfp4_intermediate(
            torch.nn.functional.silu(gate) * up
        )
        w2 = _decode_alphamoe_nvfp4(
            data["w2"][expert], data["w2_scale"][expert], columns=intermediate
        )
        for base in range(0, intermediate, _ALPHAMOE_INTERMEDIATE_TILE):
            down = torch.zeros(
                (tokens.numel(), k), dtype=torch.float32, device=x.device
            )
            for subbase in range(base, base + _ALPHAMOE_INTERMEDIATE_TILE, 64):
                down += activated[:, subbase : subbase + 64] @ w2[
                    :, subbase : subbase + 64
                ].transpose(0, 1)
            down *= flat_weights[pairs, None] * data["scaling_factor"]
            routed = down.to(torch.bfloat16)
            output[tokens] = (output[tokens].float() + routed.float()).to(
                torch.bfloat16
            )
    return output


def testAlphaMoeNvfp4AlignedMoe(args):
    """Benchmark the frozen SM100/SM103 AlphaMoE NVFP4 megakernel."""

    if args.verbose >= 1:
        print("[INFO] Running testAlphaMoeNvfp4AlignedMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)
    device = get_device(args)
    backends = filter_backends_by_compute_capability(["alphamoe"], args.routine, device)
    if not backends:
        print("[ERROR] No backends to test. Exiting.")
        return []

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    block_m = args.block_m
    if args.input_dtype != "nvfp4" or args.weight_dtype != "nvfp4":
        raise ValueError(
            "alphamoe_nvfp4_aligned_moe requires --input_dtype nvfp4 "
            "and --weight_dtype nvfp4"
        )
    if args.activation_type != ActivationType.Swiglu:
        raise ValueError("alphamoe_nvfp4_aligned_moe supports only Swiglu")
    if args.local_expert_offset != 0 or (
        args.local_num_experts is not None and args.local_num_experts != num_experts
    ):
        raise ValueError("alphamoe_nvfp4_aligned_moe does not shard experts")
    if hidden_size < 256 or hidden_size % 256:
        raise ValueError("hidden_size must be at least 256 and divisible by 256")
    if intermediate_size < 128 or intermediate_size % 128:
        raise ValueError("intermediate_size must be at least 128 and divisible by 128")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if block_m < 8 or block_m % 8:
        raise ValueError("block_m must be at least 8 and divisible by 8")

    data = _make_alphamoe_nvfp4_data(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        block_m=block_m,
        routed_scaling_factor=args.routed_scaling_factor,
        device=device,
        seed=args.random_seed,
    )
    input_args = (
        data["x"],
        data["x_scale"],
        data["w1"],
        data["w1_scale"],
        data["w2"],
        data["w2_scale"],
        data["output1_scale_gate_scalar"],
        data["output1_scale_scalar"],
        data["output2_scale_scalar"],
        data["sorted_token_ids"],
        data["expert_ids"],
        data["num_tokens_post_padded"],
        data["topk_weights"],
        data["out"],
    )

    def run_alphamoe(*tensors):
        (*kernel_inputs, out) = tensors
        result = alphamoe_nvfp4_aligned_moe(
            *kernel_inputs,
            out,
            top_k,
            block_m,
            args.routed_scaling_factor,
        )
        if result is not None:
            raise RuntimeError("AlphaMoE preallocated-output API must return None")
        return out

    routed = args.alphamoe_routing == "routed"
    if routed:
        # alphamoe_nvfp4_routed_moe builds the aligned plan itself (route
        # alignment + MoE + weighted accumulation in one call). The plan tensors
        # of ``data`` become its caller-owned workspaces, sized like
        # tests/moe/test_alphamoe_nvfp4_sm100.py.
        from flashinfer.fused_moe import alphamoe_nvfp4_routed_moe

        pairs = num_tokens * top_k
        raw_capacity = (
            pairs * block_m
            if pairs < num_experts + 1
            else pairs + (num_experts + 1) * (block_m - 1)
        )
        expert_capacity = (raw_capacity + block_m - 1) // block_m
        input_args = input_args[:9] + (
            torch.empty((expert_capacity * block_m,), dtype=torch.int32, device=device),
            torch.empty((expert_capacity,), dtype=torch.int32, device=device),
            torch.empty((1,), dtype=torch.int32, device=device),
            data["topk_weights"],
            data["topk_ids"],
            torch.empty((num_experts + 2,), dtype=torch.int32, device=device),
            data["out"],
        )

        def run_alphamoe(*tensors):  # noqa: F811
            (*kernel_inputs, topk_ids, cumsum_buffer, out) = tensors
            return alphamoe_nvfp4_routed_moe(
                *kernel_inputs,
                out,
                topk_ids,
                cumsum_buffer,
                top_k,
                block_m,
                args.routed_scaling_factor,
                accumulate=False,
            )

    if args.refcheck:
        expected = _alphamoe_nvfp4_reference(data)
        data["out"].zero_()
        actual = run_alphamoe(*input_args).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.1)

    # The operation is an accumulator. Initialize once, outside CUPTI's
    # per-iteration activity window, so the reported latency contains only the
    # frozen megakernel rather than an additional memset launch.
    data["out"].zero_()
    times = bench_gpu_time(
        fn=run_alphamoe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=not args.no_cuda_graph,
        cold_l2_cache=True,
        input_args=input_args,
    )
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        is_gated=True,
    )
    active_experts = int(data["topk_ids"].unique().numel())
    tb_per_sec = calculate_moe_kernel_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        torch.bfloat16,
        torch.bfloat16,
        input_format="nvfp4",
        weight_format="nvfp4",
        routing_logits_dtype=None,
        active_experts=active_experts,
        verbose=args.verbose,
        is_gated=True,
    )
    print_perf_metrics("alphamoe", median_time, std_time, tflops, tb_per_sec)

    result_rows = []
    if args.output_path is not None:
        row = defaultdict(str)
        row.update(
            routine=args.routine,
            median_time=median_time,
            std_time=std_time,
            tflops=tflops,
            tb_per_sec=tb_per_sec,
            backend="alphamoe",
            resolved_backend="alphamoe",
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            block_m=block_m,
            routed_scaling_factor=args.routed_scaling_factor,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routing_method="routed" if routed else "prealigned",
            input_dtype="nvfp4",
            weight_dtype="nvfp4",
            activation_type=ActivationType.Swiglu.name,
            fp4_mode="nvfp4",
            cold_l2_cache=True,
        )
        result_rows.append(row)
    return result_rows


def testCuteDslFp4BlockScaleMoe(args):
    """
    Test cute_dsl_fp4_block_scale_moe (CuTe DSL NVFP4 MoE on SM100/SM103).

    This test:
    1. Creates NVFP4-quantized weights and fp4-quantized inputs for CuTe DSL kernels
    2. Runs MoE via CuteDslMoEWrapper (or cute_dsl_fused_moe when
       ``--use_functional_api`` is set). SwiGLU only.
    3. Measures performance metrics (TFLOPS, TB/sec)

    Note: SM120/SM121 has moved to the dedicated ``b12x_fused_moe`` routine
    (see ``testB12xFusedMoe``).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testCuteDslFp4BlockScaleMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    from flashinfer import CuteDslMoEWrapper

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    local_expert_offset = args.local_expert_offset
    local_num_experts = args.local_num_experts or num_experts
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = ["cute-dsl"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}"
        )

    # CuteDslMoEWrapper activations: gated SwiGLU / GeGLU-tanh, non-gated ReLU2
    # (cute_dsl/fused_moe.py cute_dsl_fused_moe docstring). SiTU needs per-expert
    # situ_beta tensors and is benchmarked through unified_moe instead.
    activation_type = args.activation_type
    if activation_type not in _CUTE_DSL_FLAT_ACTIVATIONS:
        raise ValueError(
            f"cute_dsl_fp4_block_scale_moe supports "
            f"{[a.name for a in _CUTE_DSL_FLAT_ACTIVATIONS]}, got {activation_type.name}"
        )
    is_gated = is_gated_activation(activation_type)

    # Create CuteDSL-specific NVFP4 test data (fp4 input)
    tensors = _create_nvfp4_moe_test_data(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=local_num_experts,
        top_k=top_k,
        device=device,
        backend="cute-dsl",
        is_gated=is_gated,
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] x.shape = {tensors['x'].shape}")
        print(f"[VVERBOSE] w1_weight.shape = {tensors['w1_weight'].shape}")
        print(f"[VVERBOSE] w2_weight.shape = {tensors['w2_weight'].shape}")

    use_functional = getattr(args, "use_functional_api", False)

    if use_functional:
        from functools import partial
        from flashinfer import cute_dsl_fused_moe

        # Pre-allocate output buffer to avoid per-call allocation
        moe_output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )

        if args.verbose >= 1:
            print("[INFO] Using CuTe DSL functional API (cute_dsl_fused_moe)")
        runner = partial(
            cute_dsl_fused_moe,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
            moe_output=moe_output,
            enable_pdl=args.enable_pdl,
            activation_type=activation_type.value,
        )

        # Warmup call to populate workspace cache before timed region
        runner(
            x=tensors["x"],
            x_sf=tensors["x_sf"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
        )
    else:
        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=is_cuda_graph_compatible,
            max_num_tokens=num_tokens,
            num_local_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
            enable_pdl=args.enable_pdl,
            activation_type=activation_type.value,
        )
        runner = moe.run

    def run_cute_dsl_moe(
        x,
        x_sf,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        fc2_input_scale,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
    ):
        return runner(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
        )

    input_args = (
        tensors["x"],
        tensors["x_sf"],
        tensors["token_selected_experts"],
        tensors["token_final_scales"],
        tensors["w1_weight"],
        tensors["w1_weight_sf"],
        tensors["w1_alpha"],
        tensors["fc2_input_scale"],
        tensors["w2_weight"],
        tensors["w2_weight_sf"],
        tensors["w2_alpha"],
    )

    # Snapshot active expert count before any kernel execution, since
    # autotune tactic exploration may corrupt input tensors.
    num_active_experts = int(tensors["token_selected_experts"].unique().numel())

    backend = "cute-dsl"

    # Optional autotune warmup.
    # Clone input_args so autotune tactic exploration doesn't corrupt the
    # original tensors used by the subsequent benchmark.
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        backend = "cute-dsl_autotune"
        if args.verbose >= 1:
            print(f"[INFO] Autotune warmup for CuteDSL NVFP4 MoE: {warmup_iters} iters")
        autotune_args = tuple(
            t.clone() if isinstance(t, torch.Tensor) else t for t in input_args
        )
        with autotune(True):
            for _ in range(warmup_iters):
                run_cute_dsl_moe(*autotune_args)
        del autotune_args

    # Benchmark timing
    times = bench_gpu_time(
        fn=run_cute_dsl_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=input_args,
    )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        is_gated=is_gated,
    )
    tb_per_sec = calculate_moe_kernel_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        weight_dtype,
        input_format="nvfp4",
        weight_format="nvfp4",
        routing_logits_dtype=None,
        active_experts=num_active_experts,
        verbose=args.verbose,
        is_gated=is_gated,
    )

    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = tflops
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["num_tokens"] = num_tokens
        cur_res["hidden_size"] = hidden_size
        cur_res["intermediate_size"] = intermediate_size
        cur_res["num_experts"] = num_experts
        cur_res["top_k"] = top_k
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        cur_res["fp4_mode"] = "nvfp4"
        cur_res["activation_type"] = activation_type.name
        res.append(cur_res)

    return res


def testCuteDslBf16Moe(args):
    """
    Test the cute_dsl_bf16_moe routine (the SM90 CuTe DSL BF16/FP16 MoE,
    ``cute_dsl_fused_moe_bf16``).

    This test:
    1. Creates bf16/fp16 weights (gated activations: [gate; up] w13 repacked
       to the kernel's 32-column up/gate interleave; ReLU2: a single [E, I, H]
       projection) and pre-routed top-k ids/scales
    2. Runs MoE via CuteDslBf16MoEWrapper (or cute_dsl_fused_moe_bf16 when
       ``--use_functional_api`` is set). ``--activation-type`` selects Swiglu
       (default), GegluTanh or Relu2. ``--autotune`` runs the AutoTuner pass,
       compiling each candidate specialization as it is profiled.
    3. Measures performance metrics (TFLOPS, TB/sec)

    Pass per-rank ``--hidden_size`` / ``--intermediate_size`` for TP shapes
    (this routine has no tp_size/ep_size flags; EP uses
    ``--local_num_experts`` / ``--local_expert_offset``).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testCuteDslBf16Moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    from flashinfer import CuteDslBf16MoEWrapper
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    local_expert_offset = args.local_expert_offset
    local_num_experts = args.local_num_experts or num_experts
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = ["cute-dsl"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    activation_type = args.activation_type
    if activation_type not in (
        ActivationType.Swiglu,
        ActivationType.GegluTanh,
        ActivationType.Relu2,
    ):
        raise ValueError(
            "cute_dsl_fused_moe_bf16 supports Swiglu, GegluTanh and Relu2 "
            f"activations, got {activation_type.name}."
        )
    is_gated = is_gated_activation(activation_type)

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts} "
            f"(local {local_num_experts} @ {local_expert_offset}), top_k={top_k}"
        )

    torch.manual_seed(args.random_seed)
    x = torch.randn(num_tokens, hidden_size, dtype=input_dtype, device=device) / 10
    # Gated: [gate; up]-concatenated w13 (vLLM layout), repacked to the kernel's
    # 32-column up/gate interleave. Non-gated (Relu2): one [E, I, H] projection.
    w1_model = (
        torch.randn(
            local_num_experts,
            (2 if is_gated else 1) * intermediate_size,
            hidden_size,
            dtype=input_dtype,
            device=device,
        )
        / 10
    )
    w1 = interleave_up_gate_sm90(w1_model) if is_gated else w1_model
    w2 = (
        torch.randn(
            local_num_experts,
            hidden_size,
            intermediate_size,
            dtype=input_dtype,
            device=device,
        )
        / 10
    )

    router_logits = torch.randn(
        num_tokens, num_experts, dtype=input_dtype, device=device
    )
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    token_selected_experts = selected_experts.to(torch.int32)
    token_final_scales = routing_weights.to(torch.float32)

    if args.verbose >= 2:
        print(f"[VVERBOSE] x.shape = {x.shape}")
        print(
            f"[VVERBOSE] w1.shape = {w1.shape} "
            f"({'interleaved' if is_gated else 'non-gated'})"
        )
        print(f"[VVERBOSE] w2.shape = {w2.shape}")

    moe_output = torch.empty(num_tokens, hidden_size, dtype=input_dtype, device=device)

    use_functional = getattr(args, "use_functional_api", False)
    if use_functional:
        from functools import partial
        from flashinfer import cute_dsl_fused_moe_bf16

        if args.verbose >= 1:
            print("[INFO] Using CuTe DSL functional API (cute_dsl_fused_moe_bf16)")
        runner = partial(
            cute_dsl_fused_moe_bf16,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
            moe_output=moe_output,
            enable_pdl=args.enable_pdl,
            **_activation_kwarg(cute_dsl_fused_moe_bf16, activation_type),
        )
    else:
        moe = CuteDslBf16MoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
            enable_pdl=args.enable_pdl,
            **_activation_kwarg(CuteDslBf16MoEWrapper.__init__, activation_type),
        )

        def runner(x, ids, scales, w1, w2):
            return moe.run(x, ids, scales, w1, w2, moe_output=moe_output)

    def run_cute_dsl_bf16_moe(x, token_selected_experts, token_final_scales, w1, w2):
        return runner(x, token_selected_experts, token_final_scales, w1, w2)

    input_args = (x, token_selected_experts, token_final_scales, w1, w2)

    # Snapshot active expert count before any kernel execution, since
    # autotune tactic exploration may corrupt input tensors.
    num_active_experts = int(token_selected_experts.unique().numel())

    backend = "cute-dsl"

    if args.refcheck:
        out = run_cute_dsl_bf16_moe(*input_args).float()
        w1_ref = w1_model.float()
        ref = torch.zeros(num_tokens, hidden_size, device=device)
        for slot in range(top_k):
            e_global = token_selected_experts[:, slot].long()
            scale = token_final_scales[:, slot].unsqueeze(1)
            local = e_global - local_expert_offset
            mask = (local >= 0) & (local < local_num_experts)
            e = local.clamp(0, local_num_experts - 1)
            # Model layout: gated w1 rows are [gate; up], non-gated rows are h.
            h = torch.einsum("th,tih->ti", x.float(), w1_ref[e])
            if activation_type == ActivationType.Swiglu:
                g, u = h[:, :intermediate_size], h[:, intermediate_size:]
                act = torch.nn.functional.silu(g) * u
            elif activation_type == ActivationType.GegluTanh:
                g, u = h[:, :intermediate_size], h[:, intermediate_size:]
                act = torch.nn.functional.gelu(g, approximate="tanh") * u
            elif activation_type == ActivationType.Relu2:
                act = torch.relu(h) ** 2
            else:
                raise ValueError(f"unsupported activation_type {activation_type}")
            contrib = torch.einsum("ti,thi->th", act, w2.float()[e])
            ref += torch.where(mask.unsqueeze(1), scale * contrib, 0.0)
        max_err = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, atol=0.3, rtol=0.05)
        status = "PASS" if ok else "FAIL"
        print(f"[INFO] Refcheck {status} (max abs err {max_err:.3e})")
        if not ok and not args.allow_output_mismatch:
            raise AssertionError(f"refcheck failed: max abs err {max_err:.3e}")

    # Optional autotune warmup.
    # Clone input_args so autotune tactic exploration doesn't corrupt the
    # original tensors used by the subsequent benchmark.
    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        backend = "cute-dsl_autotune"
        if args.verbose >= 1:
            print("[INFO] Autotune warmup for CuteDSL SM90 MoE")
        autotune_args = tuple(
            t.clone() if isinstance(t, torch.Tensor) else t for t in input_args
        )
        with autotune(True, cache=cache_path):
            run_cute_dsl_bf16_moe(*autotune_args)
        del autotune_args
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Benchmark timing
    times = bench_gpu_time(
        fn=run_cute_dsl_bf16_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=input_args,
    )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        is_gated=is_gated,
    )
    tb_per_sec = calculate_moe_kernel_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        input_dtype,
        input_format="base",
        weight_format="base",
        routing_logits_dtype=None,
        active_experts=num_active_experts,
        verbose=args.verbose,
        is_gated=is_gated,
    )

    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = tflops
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["num_tokens"] = num_tokens
        cur_res["hidden_size"] = hidden_size
        cur_res["intermediate_size"] = intermediate_size
        cur_res["num_experts"] = num_experts
        cur_res["top_k"] = top_k
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = input_dtype
        cur_res["activation_type"] = activation_type.name
        res.append(cur_res)

    return res


def testB12xFusedMoe(args):
    """
    Test b12x_fused_moe (SM120/SM121 CuTe DSL FP4-weight MoE).

    The b12x MoE takes **bf16** hidden states and NVFP4-quantized weights.
    ``--b12x_quant_mode`` selects the activation precision: ``nvfp4``
    (W4A4, the kernel fuses activation quantization internally) or
    ``w4a16`` (BF16 activations). Supports both SwiGLU (gated) and ReLU2
    (non-gated) activations.

    This test:
    1. Creates NVFP4-quantized weights and bf16 inputs for b12x kernels
    2. Runs MoE via B12xMoEWrapper (or b12x_fused_moe when
       ``--use_functional_api`` is set)
    3. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testB12xFusedMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    from flashinfer import B12xMoEWrapper

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    local_num_experts = args.local_num_experts or num_experts
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    backends = ["b12x"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}"
        )

    # b12x supports SwiGLU / GeGLU (gated) and ReLU2 (non-gated)
    activation_type = args.activation_type
    _ACT_STR = {
        ActivationType.Swiglu: "silu",
        ActivationType.Geglu: "gelu_tanh",
        ActivationType.GegluTanh: "gelu_tanh",
        ActivationType.Relu2: "relu2",
    }
    if activation_type not in _ACT_STR:
        raise ValueError(
            f"b12x_fused_moe only supports Swiglu, Geglu, GegluTanh, and Relu2 "
            f"activations, got {activation_type.name}"
        )
    activation_str = _ACT_STR[activation_type]
    is_gated = activation_type.is_gated

    # Create b12x-specific NVFP4 test data (weights quantized, input stays bf16)
    tensors = _create_nvfp4_moe_test_data(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=local_num_experts,
        top_k=top_k,
        device=device,
        backend="b12x",
        is_gated=is_gated,
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] x_bf16.shape = {tensors['x_bf16'].shape}")
        print(f"[VVERBOSE] w1_weight.shape = {tensors['w1_weight'].shape}")
        print(f"[VVERBOSE] w2_weight.shape = {tensors['w2_weight'].shape}")

    use_functional = getattr(args, "use_functional_api", False)
    quant_mode = getattr(args, "b12x_quant_mode", "nvfp4")
    x_input = tensors["x_bf16"]

    if use_functional:
        from functools import partial
        from flashinfer import b12x_fused_moe

        # Pre-allocate output buffer to avoid per-call allocation
        moe_output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )

        if args.verbose >= 1:
            print("[INFO] Using b12x functional API (b12x_fused_moe)")
        runner = partial(
            b12x_fused_moe,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=local_num_experts,
            output=moe_output,
            activation=activation_str,
            quant_mode=quant_mode,
        )

        # Warmup call to populate workspace cache before timed region
        runner(
            x=x_input,
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )
    else:
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=is_cuda_graph_compatible,
            max_num_tokens=num_tokens,
            num_local_experts=local_num_experts,
            activation=activation_str,
            quant_mode=quant_mode,
        )
        runner = moe.run

    def run_b12x_moe(
        x,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        fc2_input_scale,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        token_selected_experts,
        token_final_scales,
    ):
        return runner(
            x=x,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
        )

    input_args = (
        x_input,
        tensors["w1_weight"],
        tensors["w1_weight_sf"],
        tensors["w1_alpha"],
        tensors["fc2_input_scale"],
        tensors["w2_weight"],
        tensors["w2_weight_sf"],
        tensors["w2_alpha"],
        tensors["token_selected_experts"],
        tensors["token_final_scales"],
    )

    # Snapshot active expert count before any kernel execution, since
    # autotune tactic exploration may corrupt input tensors.
    num_active_experts = int(tensors["token_selected_experts"].unique().numel())

    backend = "b12x"

    # Optional autotune warmup.
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        backend = "b12x_autotune"
        if args.verbose >= 1:
            print(f"[INFO] Autotune warmup for b12x NVFP4 MoE: {warmup_iters} iters")
        autotune_args = tuple(
            t.clone() if isinstance(t, torch.Tensor) else t for t in input_args
        )
        with autotune(True):
            for _ in range(warmup_iters):
                run_b12x_moe(*autotune_args)
        del autotune_args

    # Cold-L2 rotation clones the weights inside CUDA-graph capture, but
    # W4A16 packs weights on first use and cannot pack during capture.
    # Keep L2 warm for that combination.
    cold_l2_cache = not (quant_mode == "w4a16" and is_cuda_graph_compatible)
    if not cold_l2_cache and args.verbose >= 1:
        print(
            "[INFO] w4a16 + CUDA graph: cold-L2 buffer rotation disabled "
            "(W4A16 packed-weight cache cannot be populated during capture)"
        )

    # Benchmark timing
    times = bench_gpu_time(
        fn=run_b12x_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=cold_l2_cache,
        input_args=input_args,
    )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        is_gated=is_gated,
    )
    # Input format is bf16 for b12x (kernel fuses quantization), weights are nvfp4.
    tb_per_sec = calculate_moe_kernel_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        weight_dtype,
        input_format="bf16",
        weight_format="nvfp4",
        routing_logits_dtype=None,
        active_experts=num_active_experts,
        verbose=args.verbose,
        is_gated=is_gated,
    )

    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = tflops
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["backend"] = backend
        cur_res["num_tokens"] = num_tokens
        cur_res["hidden_size"] = hidden_size
        cur_res["intermediate_size"] = intermediate_size
        cur_res["num_experts"] = num_experts
        cur_res["top_k"] = top_k
        cur_res["local_num_experts"] = local_num_experts
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        cur_res["fp4_mode"] = quant_mode
        cur_res["cold_l2_cache"] = cold_l2_cache
        cur_res["activation_type"] = activation_type.name
        res.append(cur_res)

    return res


# =============================================================================
# MoE routing kernels (router only, no expert FFN)
# =============================================================================

# Routing methods each moe_routing backend implements. trtllm runs the same
# routing kernels the trtllm-gen fused MoE launches before its GEMMs, plus the
# public API's torch topk_ids reconstruction (see testMoeRouting).
_MOE_ROUTING_BACKEND_METHODS = {
    "trtllm": tuple(m for m in _ROUTING_METHOD_NAME_TO_TYPE if m != "hash"),
    "dsv3": ("deepseek_v3",),
    "alphamoe": ("renormalize",),
    "hash": ("hash",),
}


def testMoeRouting(args):
    """
    Benchmark standalone MoE routing kernels on the same router logits.

    Backends (picked by --backends, filtered by --routing_method):
      - trtllm:   flashinfer.fused_moe.trtllm_gen_routing (every RoutingMethodType).
                  The timed call is the public API: the 2 routing kernels the fused MoE
                  runs plus ~10 torch kernels that rebuild topk_ids from the permutation
                  (trtllm_gen_routing.py:340-351), which the fused MoE never runs. On B200
                  at 1 token that epilogue is ~70% of the time
                  (evidence/.../moe/logs/profile_trtllm_gen_routing_b200.txt), so trtllm
                  rows are not comparable to dsv3 kernel time.
      - dsv3:     fused_topk_deepseek (DeepSeekV3 grouped routing; the cake
                  backend of this API is deliberately not benchmarked)
      - alphamoe: alphamoe_fused_router (top-k + softmax + aligned route plan)
      - hash:     hash_topk (DeepSeek-V4 token-id hash routing)
    --refcheck compares each backend's selected expert set with a torch reference.
    """
    if args.verbose >= 1:
        print("[INFO] Running testMoeRouting")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    method = args.routing_method
    requested = args.backends or list(_MOE_ROUTING_BACKEND_METHODS)
    backends = []
    for backend in dict.fromkeys(requested):
        if backend not in _MOE_ROUTING_BACKEND_METHODS:
            raise ValueError(f"Unknown moe_routing backend {backend!r}")
        if method in _MOE_ROUTING_BACKEND_METHODS[backend]:
            backends.append(backend)
        elif args.backends:
            print(f"[WARNING] {backend} does not implement {method} routing. Skipping.")
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    num_tokens, num_experts, top_k = args.num_tokens, args.num_experts, args.top_k
    torch.manual_seed(args.random_seed)
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    bias = None
    if args.use_routing_bias:
        bias = 0.1 * torch.randn(num_experts, device=device, dtype=torch.float32)
    n_group = args.n_group if args.n_group else 1
    topk_group = args.topk_group if args.topk_group else 1
    rsf = args.routed_scaling_factor
    local_num_experts = args.local_num_experts or num_experts

    if method == "hash":
        input_ids = torch.randint(
            0, args.hash_vocab_size, (num_tokens,), device=device, dtype=torch.int64
        )
        tid2eid = torch.stack(
            [
                torch.randperm(num_experts, device=device)[:top_k]
                for _ in range(min(args.hash_vocab_size, 1024))
            ]
        ).to(torch.int32)
        # Tile the 1024 sampled rows to the full vocabulary (table contents do
        # not change the kernel's memory traffic).
        tid2eid = tid2eid.repeat(
            (args.hash_vocab_size + tid2eid.shape[0] - 1) // tid2eid.shape[0], 1
        )[: args.hash_vocab_size].contiguous()
        ref_ids = tid2eid[input_ids].long()
    else:
        ref_ids, _ = _reference_routing(
            logits,
            bias,
            top_k,
            args.routing_method_type,
            n_group if method == "deepseek_v3" else None,
            topk_group if method == "deepseek_v3" else None,
            rsf,
        )

    def make_runner(backend):
        if backend == "trtllm":
            from flashinfer.fused_moe import trtllm_gen_routing

            def run(logits, bias):
                return trtllm_gen_routing(
                    logits,
                    bias,
                    RoutingMethodType(args.routing_method_type),
                    top_k,
                    n_group=n_group if method == "deepseek_v3" else 0,
                    topk_group=topk_group if method == "deepseek_v3" else 0,
                    local_expert_offset=args.local_expert_offset,
                    local_num_experts=local_num_experts,
                    routed_scaling_factor=rsf,
                    tile_tokens_dim=args.tile_tokens_dim,
                    enable_pdl=args.enable_pdl,
                )

            return run, (lambda out: out.topk_ids), (logits, bias)
        if backend == "dsv3":
            topk_values = torch.empty(num_tokens, top_k, device=device)
            topk_indices = torch.empty(
                num_tokens, top_k, device=device, dtype=torch.int32
            )
            dsv3_bias = bias if bias is not None else torch.zeros_like(logits[0])

            def run(logits, bias, topk_values, topk_indices):
                fused_topk_deepseek(
                    logits,
                    bias,
                    n_group,
                    topk_group,
                    top_k,
                    rsf,
                    topk_values,
                    topk_indices,
                    backend="default",
                )
                return topk_indices

            return (
                run,
                (lambda out: out),
                (logits, dsv3_bias, topk_values, topk_indices),
            )
        if backend == "alphamoe":
            from flashinfer.fused_moe import (
                allocate_alphamoe_route_plan,
                alphamoe_fused_router,
            )

            plan = allocate_alphamoe_route_plan(
                logits, top_k=top_k, block_m=args.block_m
            )

            def run(logits):
                return alphamoe_fused_router(
                    logits, top_k=top_k, block_m=args.block_m, plan=plan
                )

            return run, (lambda out: out.topk_ids), (logits,)
        # hash
        from flashinfer.fused_moe import hash_topk

        def run(logits, input_ids, tid2eid):
            return hash_topk(
                logits,
                input_ids,
                tid2eid,
                routed_scaling_factor=rsf,
                launch_with_pdl=bool(args.enable_pdl),
            )

        return run, (lambda out: out[1]), (logits, input_ids, tid2eid)

    res = []
    for backend in backends:
        run, ids_of, input_args = make_runner(backend)
        refcheck_passed = ""
        if args.refcheck:
            got = ids_of(run(*input_args)).long()
            # Compare expert sets per token (slot order is backend-specific);
            # near-ties in random logits may legitimately flip a few experts.
            match = (
                (got.sort(dim=-1).values[:, :top_k] == ref_ids.sort(dim=-1).values)
                .all(dim=-1)
                .float()
                .mean()
                .item()
            )
            refcheck_passed = match >= 0.99
            print(
                f"[REFCHECK] {backend}: {match * 100:.2f}% of tokens select the "
                "reference expert set"
            )
            if not refcheck_passed and not args.allow_output_mismatch:
                raise AssertionError(f"{backend} routing does not match the reference")

        times = bench_gpu_time(
            fn=run,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
            input_args=input_args,
        )
        median_time = np.median(times)
        std_time = np.std(times)
        # Logits (+ token ids / hash table rows) in, top-k ids + weights out.
        moved_bytes = num_tokens * num_experts * 4 + num_tokens * top_k * 8
        if backend == "hash":
            moved_bytes += num_tokens * (8 + top_k * 4)
        tb_per_sec = moved_bytes / (median_time * 1e-3) / 1e12
        print_perf_metrics(backend, median_time, std_time, 0.0, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = 0.0
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend
            cur_res["num_tokens"] = num_tokens
            cur_res["num_experts"] = num_experts
            cur_res["top_k"] = top_k
            cur_res["n_group"] = n_group if method == "deepseek_v3" else ""
            cur_res["topk_group"] = topk_group if method == "deepseek_v3" else ""
            cur_res["routed_scaling_factor"] = rsf
            cur_res["local_expert_offset"] = args.local_expert_offset
            cur_res["local_num_experts"] = local_num_experts
            cur_res["routing_method"] = method
            cur_res["use_routing_bias"] = args.use_routing_bias
            cur_res["block_m"] = args.block_m if backend == "alphamoe" else ""
            cur_res["input_dtype"] = torch.float32
            cur_res["refcheck_passed"] = refcheck_passed
            res.append(cur_res)
    return res


# =============================================================================
# Unified NVFP4 MoE — MoELayer-based cross-backend autotune
# =============================================================================


def testUnifiedNvfp4Moe(args):
    """MoELayer cross-backend autotune for NVFP4.

    Exercises the unified MoE API: per-shape, builds MoELayer over
    {CuteDslConfig, TrtllmFp4Config}, registers both backend-native weight
    views on a single MoEWeightPack, dispatches once, reports the winner
    and both backends' best-tactic latencies.

    Emits one result row per backend candidate so CSV output carries the
    cross-backend comparison directly.
    """
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import (
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        SwiGLU,
        RoutingConfig,
        TrtllmFp4Config,
    )
    from flashinfer.fused_moe.api import BackendOptions

    if args.verbose >= 1:
        print("[INFO] Running testUnifiedNvfp4Moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    local_expert_offset = args.local_expert_offset
    local_num_experts = args.local_num_experts or num_experts
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)

    res = []
    backends = ["unified"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}, "
            f"local_experts={local_num_experts}"
        )

    # ---- Build shared bf16 reference weights ------------------------------
    torch.manual_seed(0)
    w1_bf16 = (
        torch.randn(
            local_num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w2_bf16 = (
        torch.randn(
            local_num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )

    # ---- Backend-native weight views — first-class prepare helpers (CR2) --
    # Both views are built from the SAME canonical bf16 weights, so the two
    # backends are directly comparable (and a shared reference is meaningful).
    cute_dsl_view = CuteDslConfig.prepare_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    trtllm_view = TrtllmFp4Config.prepare_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    # ---- Activation pack --------------------------------------------------
    # The activation (NVFP4-quantized hidden states + pre-routed indices) still
    # comes from the canonical test data creator; only weight prep is
    # first-class today (activation-prep promotion tracked under CR2).
    import os
    import sys

    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from tests.moe.test_cute_dsl_fused_moe import (
        check_accuracy,
        compute_reference_moe_fp4,
        create_moe_tensors,
    )

    # Wide-EP (MVP): model a single rank as a complete MoE over its
    # local_num_experts experts — route the activation WITHIN the local experts
    # (selected ids in [0, local_num_experts)) so every token is computed
    # locally. This is the "local-only" proxy: it avoids the unfaithful
    # global-routing-but-local-weights setup (most tokens skipped) and keeps the
    # derived FLOP/bandwidth metrics correct (all tokens computed, active
    # experts <= local). For EP=1 (local == global) this is unchanged.
    routing_num_experts = local_num_experts
    routing_top_k = min(top_k, local_num_experts)
    cute_dsl_data = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=routing_num_experts,
        num_local_experts=local_num_experts,
        top_k=routing_top_k,
        device=device,
    )
    # cute_dsl_data["x_sf"] is already unsqueezed to [M, H//16, 1]; strip that
    # for the Pack (runner re-applies unsqueeze in pack_inputs).
    x_sf = cute_dsl_data["x_sf"].squeeze(-1)
    # The local-only proxy generates ids in [0, local_num_experts); the runners
    # expect GLOBAL ids (the kernel maps them to the local shard via the
    # separately passed local_expert_offset), so lift the pack ids to global.
    # The reference oracle below keeps the LOCAL ids: it indexes the local
    # weight slices directly.
    local_topk_ids = cute_dsl_data["token_selected_experts"]
    act_pack = MoEActivationPack(
        hidden_states_q=cute_dsl_data["x"],
        hidden_states_scale=x_sf,
        topk_ids=local_topk_ids + local_expert_offset,
        topk_weights=cute_dsl_data["token_final_scales"],
    )

    num_active_experts = int(local_topk_ids.unique().numel())

    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("cute_dsl", cute_dsl_view)
    weight_pack.prepare_for("trtllm_fp4_routed", trtllm_view)

    # ---- MoELayer config --------------------------------------------------
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=num_experts,
            # top_k must match the pack's column count: the tensors above were
            # generated with routing_top_k (clamped to local_num_experts for
            # the local-only EP proxy), not the raw CLI top_k.
            top_k=routing_top_k,
            n_group=args.n_group,
            topk_group=args.topk_group,
            routed_scaling_factor=args.routed_scaling_factor,
        ),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
        execution=ExecutionConfig(tune_max_num_tokens=max(num_tokens, 8192)),
    )

    # ---- Dispatch: trigger cross-backend selection ------------------------
    with autotune(True):
        layer = MoELayer(config, device=device)
        _ = layer(act_pack, weight_pack)

    winner_key = layer.winner_backend or "?"
    if args.verbose >= 1:
        print(
            f"[INFO] MoELayer winner: {winner_key} "
            f"(candidates: {[r.backend_key for r in layer.runners]})"
        )

    # ---- Optional shared-reference accuracy check (CR10/CR11) -------------
    # Both backend views derive from the same bf16 weights, so a single bf16
    # reference is valid for every candidate.  Catches shared-mode errors that
    # cross-backend agreement would miss.
    ref_output = None
    if args.refcheck:
        ref_output = compute_reference_moe_fp4(
            hidden_states=cute_dsl_data["x_bf16"].float().to(device),
            gemm1_weights=w1_bf16.float().to(device),
            gemm2_weights=w2_bf16.float().to(device),
            token_selected_experts=local_topk_ids,
            token_final_scales=act_pack.topk_weights,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=routing_top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=cute_dsl_data["fc2_input_scale"],
        )

    # ---- Per-backend best-tactic timing (one result row per candidate) ----
    tuner = AutoTuner.get()
    for runner in layer.runners:
        inputs = runner.pack_inputs(act_pack, weight_pack)
        launch_kwargs = runner.launch_kwargs_for(inputs)
        with autotune(True):
            _, tactic = tuner.choose_one(
                custom_op=f"moe_{runner.backend_key}",
                runners=[runner],
                tuning_config=runner.tuning_config_for(inputs),
                inputs=inputs,
                **launch_kwargs,
            )

        def _call(r=runner, i=inputs, t=tactic, kw=launch_kwargs):
            return r.forward(i, tactic=t, **kw)

        times = bench_gpu_time(
            fn=_call,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            cold_l2_cache=True,
        )
        median_time = float(np.median(times))
        std_time = float(np.std(times))
        # Use the actual routing geometry (local experts / clamped top_k), not
        # the global config, so the wide-EP local-only proxy reports honest
        # FLOP/byte counts. For EP=1 these equal num_experts/top_k.
        tflops = calculate_moe_tflops(
            num_tokens,
            hidden_size,
            intermediate_size,
            routing_num_experts,
            routing_top_k,
            median_time,
        )
        tb_per_sec = calculate_moe_kernel_bandwidth(
            num_tokens,
            hidden_size,
            intermediate_size,
            routing_num_experts,
            routing_top_k,
            median_time,
            input_dtype,
            weight_dtype,
            input_format="nvfp4",
            weight_format="nvfp4",
            routing_logits_dtype=None,
            active_experts=num_active_experts,
            verbose=args.verbose,
        )

        # Tag: "unified/<backend>" + star the winner
        is_winner = runner.backend_key == winner_key
        backend_label = f"unified/{runner.backend_key}" + ("*" if is_winner else "")
        print_perf_metrics(backend_label, median_time, std_time, tflops, tb_per_sec)

        # Shared-reference accuracy check for this candidate (CR10/CR11).
        refcheck_passed = None
        if ref_output is not None:
            out = runner.forward(inputs, tactic=tactic, **launch_kwargs)
            refcheck_passed, pct, atol = check_accuracy(out, ref_output)
            status = "PASS" if refcheck_passed else "FAIL"
            print(
                f"[REFCHECK] {backend_label}: {status} "
                f"({pct * 100:.2f}% within atol={atol:.4f} vs bf16 reference)"
            )
            if not refcheck_passed and not args.allow_output_mismatch:
                print(
                    f"[ERROR] {backend_label} failed reference check. "
                    f"Re-run with --allow_output_mismatch to continue past mismatches."
                )

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend_label
            cur_res["refcheck_passed"] = refcheck_passed
            cur_res["num_tokens"] = num_tokens
            cur_res["hidden_size"] = hidden_size
            cur_res["intermediate_size"] = intermediate_size
            cur_res["num_experts"] = num_experts
            cur_res["top_k"] = top_k
            cur_res["local_expert_offset"] = local_expert_offset
            cur_res["local_num_experts"] = local_num_experts
            cur_res["input_dtype"] = input_dtype
            cur_res["weight_dtype"] = weight_dtype
            cur_res["fp4_mode"] = "nvfp4"
            res.append(cur_res)

    return res


def testBgmvMoe(args):
    """
    Benchmark BGMV MoE kernels for multi-LoRA inference.

    Measures the performance of the fused shrink + expand CUDA kernels
    that apply LoRA adapters in MoE models. Compares against FlashInfer's
    grouped_mm_bf16 as a baseline.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBgmvMoe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    num_experts = args.num_experts
    top_k = args.top_k
    rank = args.rank
    num_loras = args.num_loras
    num_slices = getattr(args, "num_slices", 1)
    res = []

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"rank={rank}, experts={num_experts}, top_k={top_k}, "
            f"num_loras={num_loras}, num_slices={num_slices}"
        )

    # Import BGMV MoE kernels
    from flashinfer.fused_moe.bgmv_moe import (
        bgmv_moe_shrink,
        bgmv_moe_expand,
        fill_w_ptr,
    )

    # Generate test data
    num_pairs = num_tokens * top_k
    feat_out = hidden_size

    x = torch.randn(num_tokens, hidden_size, dtype=input_dtype, device=device) * 0.1
    lora_a_weights = [
        torch.randn(
            num_loras, num_experts, rank, hidden_size, dtype=input_dtype, device=device
        )
        * 0.01
        for _ in range(num_slices)
    ]
    lora_b_weights = [
        torch.randn(
            num_loras, num_experts, feat_out, rank, dtype=input_dtype, device=device
        )
        * 0.01
        for _ in range(num_slices)
    ]

    sorted_token_ids = torch.arange(
        num_tokens, device=device, dtype=torch.int64
    ).repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), device=device, dtype=torch.int64
    )
    topk_weights = (
        torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
        .view(-1)
        .to(torch.float32)
    )
    lora_indices = torch.randint(
        0, num_loras, (num_tokens,), device=device, dtype=torch.int64
    )

    # Pre-allocate buffers for BGMV MoE
    w_ptr_a = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_a = 0
    for s in range(num_slices):
        lora_stride_a = fill_w_ptr(w_ptr_a, lora_a_weights[s], num_experts, s)

    w_ptr_b = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_b = 0
    for s in range(num_slices):
        lora_stride_b = fill_w_ptr(w_ptr_b, lora_b_weights[s], num_experts, s)

    shrink_out = torch.zeros(
        num_slices, num_pairs, rank, dtype=input_dtype, device=device
    )
    slice_start_loc = torch.zeros(num_slices, dtype=torch.int64, device=device)
    for s in range(num_slices):
        slice_start_loc[s] = s * feat_out
    output_slices = [feat_out] * num_slices

    y_accum = torch.zeros(
        num_tokens, feat_out * num_slices, dtype=torch.float32, device=device
    )

    # Define benchmark function (shrink + expand)
    def run_bgmv_moe(
        shrink_out,
        x,
        w_ptr_a,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        lora_stride_a,
        y_accum,
        w_ptr_b,
        topk_weights,
        slice_start_loc,
        output_slices,
        lora_stride_b,
    ):
        shrink_out.zero_()
        y_accum.zero_()
        bgmv_moe_shrink(
            shrink_out,
            x,
            w_ptr_a,
            sorted_token_ids,
            expert_ids,
            lora_indices,
            lora_stride_a,
        )
        bgmv_moe_expand(
            y_accum,
            shrink_out,
            w_ptr_b,
            sorted_token_ids,
            expert_ids,
            topk_weights,
            lora_indices,
            slice_start_loc,
            output_slices,
            lora_stride_b,
        )

    is_cuda_graph_compatible = not args.no_cuda_graph

    # Benchmark BGMV MoE
    times = bench_gpu_time(
        fn=run_bgmv_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=(
            shrink_out,
            x,
            w_ptr_a,
            sorted_token_ids,
            expert_ids,
            lora_indices,
            lora_stride_a,
            y_accum,
            w_ptr_b,
            topk_weights,
            slice_start_loc,
            output_slices,
            lora_stride_b,
        ),
    )

    bgmv_median = np.median(times)
    bgmv_std = np.std(times)

    # Benchmark grouped_mm_bf16 as baseline comparison (single-slice only)
    gg_kernel_median = float("nan")
    gg_full_median = float("nan")
    if num_slices != 1:
        print(
            "[INFO] grouped_mm_bf16 baseline skipped: only supported for num_slices=1"
        )
    else:
        try:
            from flashinfer.grouped_mm import grouped_mm_bf16

            # For grouped GEMM, sort tokens by (lora_id, expert_id) and build m_indptr
            num_groups = num_loras * num_experts

            # Assign each pair a group_id = lora_id * num_experts + expert_id
            lora_ids_expanded = lora_indices[sorted_token_ids]
            group_ids = lora_ids_expanded * num_experts + expert_ids
            group_ids[lora_ids_expanded < 0] = num_groups

            # Sort by group_id
            sorted_indices = torch.argsort(group_ids)
            sorted_group_ids = group_ids[sorted_indices]

            valid_mask = sorted_group_ids < num_groups
            num_valid = valid_mask.sum().item()

            sorted_token_indices = sorted_token_ids[sorted_indices[:num_valid]]
            g_input = x[sorted_token_indices]

            # Build m_indptr
            counts = torch.zeros(num_groups + 1, dtype=torch.int32, device=device)
            valid_groups = sorted_group_ids[:num_valid]
            for g in range(num_groups):
                counts[g + 1] = counts[g] + (valid_groups == g).sum().to(torch.int32)
            g_m_indptr = counts

            # Reshape LoRA weights for grouped GEMM
            g_lora_a = lora_a_weights[0].view(
                num_loras * num_experts, rank, hidden_size
            )
            g_lora_b = lora_b_weights[0].view(num_loras * num_experts, feat_out, rank)

            g_shrink_out = torch.zeros(
                num_valid, rank, dtype=input_dtype, device=device
            )
            g_expand_out = torch.zeros(
                num_valid, feat_out, dtype=input_dtype, device=device
            )

            # Warmup
            grouped_mm_bf16(g_input, g_lora_a, g_m_indptr, out=g_shrink_out)
            grouped_mm_bf16(g_shrink_out, g_lora_b, g_m_indptr, out=g_expand_out)
            torch.cuda.synchronize()

            # Benchmark: kernel only (pre-sorted, no sort overhead)
            def run_gg_kernel(
                g_input, g_lora_a, g_m_indptr, g_shrink_out, g_lora_b, g_expand_out
            ):
                g_shrink_out.zero_()
                g_expand_out.zero_()
                grouped_mm_bf16(g_input, g_lora_a, g_m_indptr, out=g_shrink_out)
                grouped_mm_bf16(g_shrink_out, g_lora_b, g_m_indptr, out=g_expand_out)

            gg_kernel_times = bench_gpu_time(
                fn=run_gg_kernel,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                sleep_after_run=False,
                enable_cupti=args.use_cupti,
                use_cuda_graph=is_cuda_graph_compatible,
                cold_l2_cache=True,
                input_args=(
                    g_input,
                    g_lora_a,
                    g_m_indptr,
                    g_shrink_out,
                    g_lora_b,
                    g_expand_out,
                ),
            )
            gg_kernel_median = np.median(gg_kernel_times)

            # Benchmark: sort + kernel (full end-to-end)
            def run_gg_full(
                x,
                sorted_token_ids,
                lora_indices,
                expert_ids,
                num_experts,
                num_groups,
                g_lora_a,
                g_lora_b,
                g_m_indptr,
                g_shrink_out,
                g_expand_out,
            ):
                _lora_ids = lora_indices[sorted_token_ids]
                _group_ids = _lora_ids * num_experts + expert_ids
                _group_ids[_lora_ids < 0] = num_groups
                _sorted_indices = torch.argsort(_group_ids)
                _sorted_token_indices = sorted_token_ids[_sorted_indices[:num_valid]]
                _g_input = x[_sorted_token_indices]
                g_shrink_out.zero_()
                g_expand_out.zero_()
                grouped_mm_bf16(_g_input, g_lora_a, g_m_indptr, out=g_shrink_out)
                grouped_mm_bf16(g_shrink_out, g_lora_b, g_m_indptr, out=g_expand_out)

            gg_full_times = bench_gpu_time(
                fn=run_gg_full,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                sleep_after_run=False,
                enable_cupti=args.use_cupti,
                use_cuda_graph=False,  # sort is not cuda-graph compatible
                cold_l2_cache=True,
                input_args=(
                    x,
                    sorted_token_ids,
                    lora_indices,
                    expert_ids,
                    num_experts,
                    num_groups,
                    g_lora_a,
                    g_lora_b,
                    g_m_indptr,
                    g_shrink_out,
                    g_expand_out,
                ),
            )
            gg_full_median = np.median(gg_full_times)

        except (ImportError, RuntimeError) as e:
            print(f"[INFO] grouped_mm_bf16 baseline skipped: {e}")

    # Print comparison results
    def _fmt(v):
        return f"{v:.3f}" if v == v else "N/A"

    def _speedup(baseline, kernel):
        if baseline != baseline or kernel != kernel or kernel == 0:
            return "N/A"
        return f"{baseline / kernel:.2f}x"

    print(f"\n{'=' * 70}")
    print(
        f"  BGMV MoE Benchmark: tokens={num_tokens}, hidden={hidden_size}, "
        f"rank={rank}, experts={num_experts}, top_k={top_k}"
    )
    print(f"{'=' * 70}")
    print(f"  {'Method':<25} {'Median (ms)':>12} {'Speedup vs BGMV MoE':>20}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 20}")
    print(f"  {'BGMV MoE (this PR)':<25} {_fmt(bgmv_median):>12} {'—':>20}")
    print(
        f"  {'grouped_mm (kernel)':<25} {_fmt(gg_kernel_median):>12} {_speedup(gg_kernel_median, bgmv_median):>20}"
    )
    print(
        f"  {'grouped_mm (sort+kern)':<25} {_fmt(gg_full_median):>12} {_speedup(gg_full_median, bgmv_median):>20}"
    )
    print(f"{'=' * 70}\n")

    # Also print in standard format
    flops = num_slices * num_pairs * (hidden_size * rank + rank * feat_out) * 2
    tflops = flops / (bgmv_median * 1e-3) / 1e12

    backend = "bgmv_moe"
    print_perf_metrics(backend, bgmv_median, bgmv_std, tflops, 0.0)

    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = args.routine
        cur_res["median_time"] = bgmv_median
        cur_res["std_time"] = bgmv_std
        cur_res["tflops"] = tflops
        cur_res["backend"] = backend
        cur_res["num_tokens"] = num_tokens
        cur_res["hidden_size"] = hidden_size
        cur_res["num_experts"] = num_experts
        cur_res["top_k"] = top_k
        cur_res["rank"] = rank
        cur_res["num_loras"] = num_loras
        cur_res["num_slices"] = num_slices
        cur_res["input_dtype"] = str(input_dtype)
        cur_res["gg_kernel_median"] = gg_kernel_median
        cur_res["gg_full_median"] = gg_full_median
        res.append(cur_res)

    return res
