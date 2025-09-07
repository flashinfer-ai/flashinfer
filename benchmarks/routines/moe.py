from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    WeightLayout,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    cutlass_fused_moe,
    convert_to_block_layout,
)
from flashinfer import fp4_quantize, shuffle_matrix_a
from flashinfer.testing.utils import (
    bench_gpu_time,
    bench_gpu_time_with_cudagraph,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
)


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
    parser.add_argument(
        "--num_tokens", type=int, required=True, help="Number of input tokens."
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Hidden dimension size."
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=True,
        help="Intermediate dimension size.",
    )
    parser.add_argument(
        "--num_experts", type=int, required=True, help="Total number of experts."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Number of experts to route to per token.",
    )
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
        "--tile_tokens_dim",
        type=int,
        required=False,
        default=8,
        help="Tile dimension for tokens.",
    )
    parser.add_argument(
        "--routing_method",
        type=str,
        required=False,
        default="deepseek_v3",
        choices=[
            "renormalize",
            "deepseek_v3",
            "llama4",
            "renormalize_naive",
            "topk",
        ],
        help=(
            "Routing method: renormalize | deepseek_v3 | llama4 | renormalize_naive | topk."
        ),
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
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the input hidden states.",
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the weights (before quantization).",
    )
    parser.add_argument(
        "--gated_act",
        type=str,
        required=False,
        default="swiglu",
        choices=["swiglu", "geglu"],
        help="Type of gated activation function: swiglu | geglu.",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Enable autotuner warmup for supported routines (trtllm_fp4_block_scale_moe and cutlass_fused_moe)."
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

    args = parser.parse_args(line)

    # Normalize routing method (map string to internal int expected by kernels)
    routing_method_name_to_type = {
        "renormalize": 1,
        "deepseek_v3": 2,
        "llama4": 3,
        "renormalize_naive": 4,
        "topk": 5,
    }
    args.routing_method_type = routing_method_name_to_type[args.routing_method]

    # Normalize gated act type (map string to internal int expected by kernels)
    gated_act_name_to_type = {
        "swiglu": 0,
        "geglu": 1,
    }
    args.gated_act_type = gated_act_name_to_type[args.gated_act]

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
):
    """
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
        # FP8 block scale MOE always expects float32 routing logits (line 333 in kernel_launcher.cu)
        routing_logits = torch.randn(
            (num_tokens, num_experts), device=device, dtype=torch.float32
        )
    elif moe_kernel_type == "fp8_per_tensor":
        # FP8 per-tensor MOE dtype depends on use_routing_scales_on_input parameter
        # For Llama4: use_routing_scales_on_input=True -> bfloat16
        # For others: use_routing_scales_on_input=False -> float32
        if routing_method_type == 3:  # Llama4 uses routing scales on input
            routing_logits = torch.randn(
                (num_tokens, num_experts), device=device, dtype=torch.bfloat16
            )
        else:
            routing_logits = torch.randn(
                (num_tokens, num_experts), device=device, dtype=torch.float32
            )
    elif moe_kernel_type == "fp4_block_scale":
        # FP4 block scale MOE follows the test pattern: float32 for DeepSeekV3, bfloat16 for others
        if routing_method_type == 2:  # DeepSeekV3 - uses float32
            routing_logits = torch.randn(
                (num_tokens, num_experts), device=device, dtype=torch.float32
            )
        else:  # All other routing methods (Renormalize, RenormalizeNaive, Llama4) - use bfloat16
            routing_logits = torch.randn(
                (num_tokens, num_experts), device=device, dtype=torch.bfloat16
            )
    else:
        raise ValueError(f"Unknown MOE kernel type: {moe_kernel_type}")

    # Create routing bias if needed - always bfloat16
    routing_bias = None
    if use_routing_bias:
        routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)

    # Create hidden states - always start with bfloat16 for proper quantization
    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
    )

    # Create weights - always start with bfloat16 for proper quantization
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )

    return routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights


def calculate_fp4_global_scale_factor(tensor):
    """Calculate global scale factor for FP4 quantization."""
    # Calculate as a tensor on the same device
    # Using the same formula as in test files: FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    tensor_amax = tensor.abs().max().to(torch.float32)
    # FLOAT8_E4M3_MAX = 448, FLOAT4_E2M1_MAX = 6
    global_scale = (448.0 * 6.0) / tensor_amax
    return global_scale


def quant_fp4_simple(a, a_global_sf, use_ue8m0=False, is_sf_swizzled_layout=True):
    """
    Simplified FP4 quantization for benchmarking.
    In production, use the actual fp4_quantize function.
    """
    sf_vec_size = 16

    # Use the actual fp4_quantize function from flashinfer
    a_fp4, a_sf = fp4_quantize(
        a, a_global_sf, sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches_simple(
    a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True
):
    """Simplified FP4 batch quantization for benchmarking."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        # Calculate global scale factor (returns tensor)
        a_global_sf = calculate_fp4_global_scale_factor(a[i])
        a_fp4, a_sf, _ = quant_fp4_simple(
            a[i], a_global_sf, use_ue8m0, is_sf_swizzled_layout
        )
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def calculate_moe_tflops(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    time_ms: float,
) -> float:
    """
    Calculate TFLOPS for MOE operation.

    MOE computation involves:
    1. First GEMM: [num_tokens, hidden_size] x [num_experts, hidden_size, 2*intermediate_size]
    2. Activation function (SwiGLU gate)
    3. Second GEMM: [num_tokens, intermediate_size] x [num_experts, intermediate_size, hidden_size]

    For each token, we only compute for top_k experts.

    """
    # FLOPS per token per expert (base calculation)
    flops_per_token_per_expert = (
        2 * hidden_size * 2 * intermediate_size  # First GEMM
        + 2 * intermediate_size * hidden_size  # Second GEMM
    )

    total_flops = num_tokens * top_k * flops_per_token_per_expert
    tflops = total_flops / (time_ms * 1e-3) / 1e12  # Convert to TFLOPS
    return tflops


def calculate_moe_bandwidth(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    time_ms: float,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    input_format: Optional[str] = None,
    weight_format: Optional[str] = None,
    routing_logits_dtype: Optional[torch.dtype] = torch.float32,
    active_experts: Optional[int] = None,
) -> float:
    """
    Calculate memory bandwidth for MOE operation in TB/sec.

    Args:
        input_format: Override for input representation ("fp8" or "fp4"); None uses dtype.itemsize
        weight_format: Override for weight representation ("fp8" or "fp4"); None uses dtype.itemsize
        routing_logits_dtype: Dtype for routing logits memory accounting (default float32)
    """

    # Get effective byte sizes
    def get_effective_bytes(dtype: torch.dtype, fmt: Optional[str]) -> float:
        if fmt == "fp4":
            return 0.5
        if fmt == "fp8":
            return 1.0
        return dtype.itemsize

    input_bytes_per_element = get_effective_bytes(input_dtype, input_format)
    weight_bytes_per_element = get_effective_bytes(weight_dtype, weight_format)

    # Input memory: hidden states + routing logits
    # Note: routing logits dtype depends on kernel; pass in when known, default float32; None means excluded
    routing_logits_bytes = (
        0 if routing_logits_dtype is None else routing_logits_dtype.itemsize
    )
    input_bytes = (
        # Count hidden states once; kernels typically reuse inputs for multiple experts
        num_tokens * hidden_size * input_bytes_per_element
        + num_tokens * num_experts * routing_logits_bytes
    )

    # Weight memory (reuse weights across tokens by grouping tokens per expert)
    # Assume each active expert's weights are read once per run.
    weight_bytes_per_expert = (
        2 * intermediate_size * hidden_size * weight_bytes_per_element  # gemm1
        + hidden_size * intermediate_size * weight_bytes_per_element  # gemm2
    )
    if active_experts is not None:
        num_active_experts = active_experts
    else:
        num_active_experts = min(num_experts, top_k * num_tokens)
    weight_bytes = num_active_experts * weight_bytes_per_expert

    # Output memory (typically full precision)
    output_bytes = num_tokens * hidden_size * input_dtype.itemsize

    total_bytes = input_bytes + weight_bytes + output_bytes
    tb_per_sec = total_bytes / (time_ms * 1e-3) / 1e12  # Convert to TB/sec
    return tb_per_sec


def _compute_routing(router_logits: torch.Tensor, top_k: int):
    routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def _dynamic_per_tensor_fp8_quant(x: torch.Tensor):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    x_max = x.abs().max().float().clamp(min=1e-6)
    scale = x_max / fp8_max
    inv_scale = 1.0 / scale
    out = (x.float() * inv_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return out, scale.view((1,))


def testTrtllmFp4BlockScaleMoe(args):
    """
    Test trtllm_fp4_block_scale_moe API (TensorRT-LLM fused MoE).

    This test:
    1. Creates quantized FP4 weights and scales
    2. Runs FP4 block scale MOE
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

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)

    # Parse configuration
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    n_group = (
        args.n_group
        if hasattr(args, "n_group") and args.n_group is not None and args.n_group > 0
        else None
    )
    topk_group = (
        args.topk_group
        if hasattr(args, "topk_group")
        and args.topk_group is not None
        and args.topk_group > 0
        else None
    )
    routed_scaling_factor = (
        args.routed_scaling_factor
        if hasattr(args, "routed_scaling_factor")
        and args.routed_scaling_factor is not None
        else None
    )
    local_expert_offset = args.local_expert_offset
    local_num_experts = args.local_num_experts or num_experts
    tile_tokens_dim = args.tile_tokens_dim
    routing_method_type = args.routing_method_type
    use_shuffled_weight = args.use_shuffled_weight
    weight_layout = args.weight_layout
    is_cuda_graph_compatible = not args.no_cuda_graph
    gated_act_type = args.gated_act_type

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}"
        )

    # Create test data
    routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights = (
        create_trtllm_moe_test_data(
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            routing_method_type,
            args.use_routing_bias,
            input_dtype,
            weight_dtype,
            device,
            moe_kernel_type="fp4_block_scale",
        )
    )

    # For FP4, we need to properly quantize weights and create scales
    use_ue8m0 = False

    # Calculate global scale factor for hidden states
    hidden_states_scale_global = calculate_fp4_global_scale_factor(hidden_states)

    # Quantize weights using proper FP4 quantization
    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
        quant_fp4_batches_simple(gemm1_weights, num_experts, use_ue8m0, True)
    )
    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
        quant_fp4_batches_simple(gemm2_weights, num_experts, use_ue8m0, True)
    )

    # Quantize hidden states
    hidden_states_fp4_bytes, hidden_states_scale_fp4_bytes, _ = quant_fp4_simple(
        hidden_states, hidden_states_scale_global, use_ue8m0, True
    )

    # Reshape hidden states for the kernel (pack 2 FP4 values into 1 byte)
    # Keep as uint8 format for FP4 packed data
    hidden_states_fp4 = hidden_states_fp4_bytes.view(torch.uint8).reshape(
        hidden_states.shape[0], hidden_states.shape[1] // 2
    )
    # Hidden-states scale for FP4 must be 2D: [num_tokens, hidden_size // 16]
    hidden_states_scale_linear_fp4 = hidden_states_scale_fp4_bytes.view(
        torch.float8_e4m3fn
    )
    # Ensure expected shape (16 elements per hidden value for NvFP4)
    expected_scale_elems = (num_tokens * hidden_size) // 16
    if hidden_states_scale_linear_fp4.numel() != expected_scale_elems:
        if args.verbose >= 1:
            print(
                f"[INFO] Adjusting FP4 hidden_states_scale from {hidden_states_scale_linear_fp4.numel()} to {expected_scale_elems} elements"
            )
        hidden_states_scale_linear_fp4 = torch.ones(
            expected_scale_elems, device=device, dtype=torch.float8_e4m3fn
        )
    hidden_states_scale_linear_fp4 = hidden_states_scale_linear_fp4.reshape(
        num_tokens, hidden_size // 16
    )

    # Prepare weights for kernel
    # For FP4 weights, keep them as uint8 (packed format) - don't convert to float8_e4m3fn
    gemm1_weights_fp4 = gemm1_weights_fp4_bytes.view(torch.uint8).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    # Scale factors should be viewed as float8_e4m3fn
    gemm1_weights_scale = gemm1_scales_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )

    gemm2_weights_fp4 = gemm2_weights_fp4_bytes.view(torch.uint8).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    gemm2_weights_scale = gemm2_scales_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )

    # Optional parameters for FP4 (using None for simplicity in benchmarking)
    gemm1_bias = None
    gemm1_alpha = None
    gemm1_beta = None
    gemm1_clamp_limit = None
    gemm2_bias = None

    # Create scale scalars (simplified - in practice these would be computed)
    output1_scale_scalar = torch.ones(
        local_num_experts, device=device, dtype=torch.float32
    )
    output1_scale_gate_scalar = torch.ones(
        local_num_experts, device=device, dtype=torch.float32
    )
    output2_scale_scalar = torch.ones(
        local_num_experts, device=device, dtype=torch.float32
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] routing_logits.shape = {routing_logits.shape}")
        print(f"[VVERBOSE] hidden_states.shape = {hidden_states.shape}")
        print(f"[VVERBOSE] gemm1_weights_fp4.shape = {gemm1_weights_fp4.shape}")
        print(f"[VVERBOSE] gemm2_weights_fp4.shape = {gemm2_weights_fp4.shape}")

    def run_fp4_moe():
        return trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states_fp4,
            hidden_states_scale=hidden_states_scale_linear_fp4,
            gemm1_weights=gemm1_weights_fp4,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=gemm1_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights_fp4,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=output2_scale_scalar,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            tile_tokens_dim=tile_tokens_dim,
            routing_method_type=routing_method_type,
            gated_act_type=gated_act_type,
            do_finalize=True,
        )

    backend = "trtllm"

    # Optional autotune warmup (supported for FP4 TRTLlm fused MoE)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        backend = "trtllm_autotune"
        if args.verbose >= 1:
            print(
                f"[INFO] Autotune warmup for FP4 block scale MoE: {warmup_iters} iters"
            )
        with autotune(True):
            for _ in range(warmup_iters):
                run_fp4_moe()

    # Benchmark timing
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=run_fp4_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            num_iters_within_graph=20,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=run_fp4_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
    )
    tb_per_sec = calculate_moe_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        weight_dtype,
        input_format="fp4",
        weight_format="fp4",
        routing_logits_dtype=routing_logits.dtype,
    )

    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    res = []
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
        cur_res["n_group"] = n_group
        cur_res["topk_group"] = topk_group
        cur_res["routed_scaling_factor"] = routed_scaling_factor
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["tile_tokens_dim"] = tile_tokens_dim
        cur_res["routing_method"] = args.routing_method
        cur_res["use_shuffled_weight"] = use_shuffled_weight
        cur_res["weight_layout"] = weight_layout
        cur_res["use_routing_bias"] = args.use_routing_bias
        cur_res["use_routing_scales_on_input"] = args.use_routing_scales_on_input
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        cur_res["gated_act"] = args.gated_act
        res.append(cur_res)

    return res


def testCutlassFusedMoe(args):
    """
    Benchmark cutlass_fused_moe (CUTLASS MoE) with variants mirroring tests in tests/test_trtllm_cutlass_fused_moe.py
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

    # Create base tensors
    torch.manual_seed(args.random_seed)
    x = torch.randn(num_tokens, hidden_size, dtype=input_dtype, device=device)
    w31_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
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
    routing_weights, selected_experts = _compute_routing(router_logits, top_k)

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
        # Split w31 into w3 and w1 along intermediate dim
        w3_weight, w1_weight = torch.chunk(w31_ep_tensor, 2, dim=1)
        shard = intermediate_size // tp_size
        start = tp_rank * shard
        end = start + shard
        w3_local = w3_weight[:, start:end, :]
        w1_local = w1_weight[:, start:end, :]
        w31_local = torch.cat([w3_local, w1_local], dim=1)
        w2_local = w2_ep_tensor[:, :, start:end]
        return w31_local.contiguous(), w2_local.contiguous()

    w31_local, w2_local = build_tp_shards(w31_ep, w2_ep)

    # Prepare variant-specific inputs (outside of the timed/captured region)
    variant = getattr(args, "cutlass_variant", "base")
    out = torch.empty_like(x)

    if variant == "base":

        def run_cutlass():
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
            w31_q, s31 = _dynamic_per_tensor_fp8_quant(w31_expert)
            w2_q, s2 = _dynamic_per_tensor_fp8_quant(w2_expert)
            w31_weight_fp8[expert_id].copy_(w31_q)
            w2_weight_fp8[expert_id].copy_(w2_q)
            # Store the same scalar twice to mimic test layout (avoid torch.tensor())
            w31_scales[expert_id, 0] = s31.to(dtype=input_dtype, device=device)
            w31_scales[expert_id, 1] = s31.to(dtype=input_dtype, device=device)
            w2_scales[expert_id, 0] = s2.to(dtype=input_dtype, device=device)

        x_quant, hidden_states_scale = _dynamic_per_tensor_fp8_quant(x)
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

        def run_cutlass():
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
            )

    elif variant == "nvfp4":
        # NVFP4: FP4 block-scale weights, optional quantized input
        FLOAT4_E2M1_MAX = 6.0
        FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

        def round_up(x_val, y):
            return (x_val + y - 1) // y * y

        e = w31_local.shape[0]
        n = w2_local.shape[2]  # local intermediate size after TP
        k = hidden_size
        quant_blocksize = 16

        # Weight quantization buffers
        w1_q = torch.empty((e, 2 * n, k // 2), device=device, dtype=torch.uint8)
        w2_q = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
        w1_blockscale = torch.empty(
            (e, round_up(2 * n, 128), round_up(k // quant_blocksize, 4)),
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

        def run_cutlass():
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
            )
    else:
        raise ValueError(f"Unknown cutlass_variant: {variant}")

    backend = "cutlass"

    # Optional autotune warmup (supported for CUTLASS fused MoE)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        backend = "cutlass_autotune"
        if args.verbose >= 1:
            print(f"[INFO] Autotune warmup for CUTLASS fused MoE: {warmup_iters} iters")
        with autotune(True):
            for _ in range(warmup_iters):
                run_cutlass()

    # Measure
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=run_cutlass,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            num_iters_within_graph=20,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=run_cutlass,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
    )
    tb_per_sec = calculate_moe_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        input_dtype,
        input_format=(
            "fp8"
            if variant == "fp8"
            else (
                "fp4"
                if (variant == "nvfp4" and getattr(args, "quantized_input", False))
                else None
            )
        ),
        weight_format=(
            "fp8" if variant == "fp8" else ("fp4" if variant == "nvfp4" else None)
        ),
        routing_logits_dtype=router_logits.dtype,
        active_experts=int(selected_experts.unique().numel()),
    )

    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    res = []
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


def testTrtllmFp8BlockScaleMoe(args):
    """
    Test trtllm_fp8_block_scale_moe API (TensorRT-LLM fused MoE).

    This test:
    1. Creates quantized FP8 weights and block scales
    2. Runs FP8 block scale MOE
    3. Measures performance metrics (TFLOPS, TB/sec)

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

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)

    # Parse configuration
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    n_group = (
        args.n_group
        if hasattr(args, "n_group") and args.n_group is not None and args.n_group > 0
        else None
    )
    topk_group = (
        args.topk_group
        if hasattr(args, "topk_group")
        and args.topk_group is not None
        and args.topk_group > 0
        else None
    )
    routed_scaling_factor = (
        args.routed_scaling_factor
        if hasattr(args, "routed_scaling_factor")
        and args.routed_scaling_factor is not None
        else None
    )
    local_expert_offset = args.local_expert_offset
    local_num_experts = args.local_num_experts or num_experts
    tile_tokens_dim = args.tile_tokens_dim
    routing_method_type = args.routing_method_type
    use_shuffled_weight = args.use_shuffled_weight
    weight_layout = args.weight_layout
    is_cuda_graph_compatible = not args.no_cuda_graph

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}"
        )

    # Create test data
    routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights = (
        create_trtllm_moe_test_data(
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            routing_method_type,
            args.use_routing_bias,
            input_dtype,
            weight_dtype,
            device,
            moe_kernel_type="fp8_block_scale",
        )
    )

    # For FP8 block scale, create quantized weights and block scales
    # Quantize to FP8
    gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
    gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)

    # Optionally shuffle weights and convert to BlockMajorK layout to match kernel expectation
    if use_shuffled_weight:
        # This tile size follows test implementations
        epilogue_tile_m = 64

        gemm1_weights_fp8_shuffled = []
        gemm2_weights_fp8_shuffled = []
        for i in range(num_experts):
            tmp_w1 = shuffle_matrix_a(
                gemm1_weights_fp8[i].view(torch.uint8), epilogue_tile_m
            )
            tmp_w2 = shuffle_matrix_a(
                gemm2_weights_fp8[i].view(torch.uint8), epilogue_tile_m
            )
            if weight_layout == WeightLayout.BlockMajorK:
                block_k = 128
                tmp_w1 = convert_to_block_layout(tmp_w1, block_k)
                tmp_w2 = convert_to_block_layout(tmp_w2, block_k)
            gemm1_weights_fp8_shuffled.append(tmp_w1)
            gemm2_weights_fp8_shuffled.append(tmp_w2)

        kernel_gemm1_weights = torch.stack(gemm1_weights_fp8_shuffled).view(
            torch.float8_e4m3fn
        )
        kernel_gemm2_weights = torch.stack(gemm2_weights_fp8_shuffled).view(
            torch.float8_e4m3fn
        )
    else:
        kernel_gemm1_weights = gemm1_weights_fp8
        kernel_gemm2_weights = gemm2_weights_fp8

    # Create block scale tensors for hidden states and weights (use float32 for scales)
    # TensorRT-LLM FP8 block-scale expects hidden_states_scale shape [hidden_size // 128, num_tokens]
    hidden_states_scale = 2.0 * torch.ones(
        (hidden_size // 128, num_tokens), device=device, dtype=torch.float32
    )
    gemm1_weights_scale = 2.0 * torch.ones(
        (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights_scale = 2.0 * torch.ones(
        (num_experts, hidden_size // 128, intermediate_size // 128),
        device=device,
        dtype=torch.float32,
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] routing_logits.shape = {routing_logits.shape}")
        print(f"[VVERBOSE] hidden_states.shape = {hidden_states.shape}")
        print(f"[VVERBOSE] gemm1_weights_fp8.shape = {gemm1_weights_fp8.shape}")
        print(f"[VVERBOSE] gemm2_weights_fp8.shape = {gemm2_weights_fp8.shape}")

    # Match test heuristic for tile_tokens_dim when using BlockMajorK
    if use_shuffled_weight and weight_layout == WeightLayout.BlockMajorK:

        def _next_pow2(x: int) -> int:
            x = max(1, x)
            x -= 1
            x |= x >> 1
            x |= x >> 2
            x |= x >> 4
            x |= x >> 8
            x |= x >> 16
            return x + 1

        tokens_per_expert = max(1, (num_tokens * top_k) // max(local_num_experts, 1))
        suggested_tile = min(max(_next_pow2(tokens_per_expert), 8), 64)
        if suggested_tile != tile_tokens_dim and args.verbose >= 1:
            print(
                f"[INFO] Overriding tile_tokens_dim {tile_tokens_dim} -> {suggested_tile} for BlockMajorK"
            )
        tile_tokens_dim = suggested_tile

    def run_fp8_block_moe():
        # Quantize hidden states to FP8 for block scale MOE
        hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)
        # Note: FP8 block scale MOE expects int64_t for n_group/topk_group, not Optional[int64_t]
        # So we convert None to 0 to indicate "no groups" mode
        return trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states_fp8,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=kernel_gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm2_weights=kernel_gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group if n_group is not None else 0,
            topk_group=topk_group if topk_group is not None else 0,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            tile_tokens_dim=tile_tokens_dim,
            routing_method_type=routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            enable_pdl=True,
        )

    # Benchmark timing
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=run_fp8_block_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            num_iters_within_graph=20,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=run_fp8_block_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
    )
    tb_per_sec = calculate_moe_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        weight_dtype,
        input_format="fp8",
        weight_format="fp8",
        routing_logits_dtype=routing_logits.dtype,
    )

    backend = "trtllm"
    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    res = []
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
        cur_res["n_group"] = n_group
        cur_res["topk_group"] = topk_group
        cur_res["routed_scaling_factor"] = routed_scaling_factor
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["tile_tokens_dim"] = tile_tokens_dim
        cur_res["routing_method"] = args.routing_method
        cur_res["use_shuffled_weight"] = use_shuffled_weight
        cur_res["weight_layout"] = weight_layout
        cur_res["use_routing_bias"] = args.use_routing_bias
        cur_res["use_routing_scales_on_input"] = args.use_routing_scales_on_input
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        res.append(cur_res)

    return res


def testTrtllmFp8PerTensorScaleMoe(args):
    """
    Test trtllm_fp8_per_tensor_scale_moe API (TensorRT-LLM fused MoE).

    This test:
    1. Creates quantized FP8 weights and per-tensor scales
    2. Runs FP8 per-tensor scale MOE
    3. Measures performance metrics (TFLOPS, TB/sec)

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

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    weight_dtype = dtype_str_to_torch_dtype(args.weight_dtype)

    # Parse configuration
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    top_k = args.top_k
    n_group = (
        args.n_group
        if hasattr(args, "n_group") and args.n_group is not None and args.n_group > 0
        else None
    )
    topk_group = (
        args.topk_group
        if hasattr(args, "topk_group")
        and args.topk_group is not None
        and args.topk_group > 0
        else None
    )
    routed_scaling_factor = (
        args.routed_scaling_factor
        if hasattr(args, "routed_scaling_factor")
        and args.routed_scaling_factor is not None
        else None
    )
    local_expert_offset = args.local_expert_offset
    local_num_experts = args.local_num_experts or num_experts
    tile_tokens_dim = args.tile_tokens_dim
    routing_method_type = args.routing_method_type
    use_routing_scales_on_input = args.use_routing_scales_on_input
    is_cuda_graph_compatible = not args.no_cuda_graph

    if args.verbose >= 1:
        print(
            f"[INFO] Configuration: tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}"
        )

    # Create test data
    routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights = (
        create_trtllm_moe_test_data(
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            routing_method_type,
            args.use_routing_bias,
            input_dtype,
            weight_dtype,
            device,
            moe_kernel_type="fp8_per_tensor",
        )
    )

    # For FP8 per-tensor scale, create quantized weights and per-tensor scales
    # Quantize to FP8
    gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
    gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)

    # Quantize hidden states to FP8 for per-tensor scale
    hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)

    # Create per-tensor scale scalars
    output1_scales_scalar = torch.ones(
        local_num_experts, device=device, dtype=torch.float32
    )
    output1_scales_gate_scalar = torch.ones(
        local_num_experts, device=device, dtype=torch.float32
    )
    output2_scales_scalar = torch.ones(
        local_num_experts, device=device, dtype=torch.float32
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] routing_logits.shape = {routing_logits.shape}")
        print(f"[VVERBOSE] hidden_states.shape = {hidden_states.shape}")
        print(f"[VVERBOSE] gemm1_weights_fp8.shape = {gemm1_weights_fp8.shape}")
        print(f"[VVERBOSE] gemm2_weights_fp8.shape = {gemm2_weights_fp8.shape}")

    def run_fp8_per_tensor_moe():
        # Note: FP8 per-tensor MOE expects int64_t for n_group/topk_group, not Optional[int64_t]
        # So we convert None to 0 to indicate "no groups" mode
        return trtllm_fp8_per_tensor_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states_fp8,
            gemm1_weights=gemm1_weights_fp8,
            output1_scales_scalar=output1_scales_scalar,
            output1_scales_gate_scalar=output1_scales_gate_scalar,
            gemm2_weights=gemm2_weights_fp8,
            output2_scales_scalar=output2_scales_scalar,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group if n_group is not None else 0,
            topk_group=topk_group if topk_group is not None else 0,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=use_routing_scales_on_input,
            tile_tokens_dim=tile_tokens_dim,
            routing_method_type=routing_method_type,
        )

    # Benchmark timing
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=run_fp8_per_tensor_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            num_iters_within_graph=20,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=run_fp8_per_tensor_moe,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
    )
    tb_per_sec = calculate_moe_bandwidth(
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        median_time,
        input_dtype,
        weight_dtype,
        input_format="fp8",
        weight_format="fp8",
        routing_logits_dtype=routing_logits.dtype,
    )

    backend = "trtllm"
    print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

    res = []
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
        cur_res["n_group"] = n_group
        cur_res["topk_group"] = topk_group
        cur_res["routed_scaling_factor"] = routed_scaling_factor
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["tile_tokens_dim"] = tile_tokens_dim
        cur_res["routing_method"] = args.routing_method
        cur_res["use_routing_bias"] = args.use_routing_bias
        cur_res["use_routing_scales_on_input"] = use_routing_scales_on_input
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        res.append(cur_res)

    return res
