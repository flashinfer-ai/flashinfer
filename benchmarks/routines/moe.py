from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import flashinfer
from flashinfer import ActivationType
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    cutlass_fused_moe,
    fused_topk_deepseek,
)
from flashinfer.fused_moe.core import RoutingMethodType
from flashinfer import fp4_quantize
from flashinfer.testing.utils import (
    bench_gpu_time,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    enum_type,
    get_device,
    print_perf_metrics,
    filter_backends_by_compute_capability,
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
    FLOAT8_E4M3_MAX,
    FLOAT4_E2M1_MAX,
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
    add_common_moe_args(parser)
    # Note: num_tokens/hidden_size is added by add_common_moe_args
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=True,
        help="Intermediate dimension size.",
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
    elif moe_kernel_type == "fp4_block_scale":
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
        num_experts, hidden_size, intermediate_size, device, dtype=torch.bfloat16
    )

    return routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights


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

    Args:
        routing_logits: [num_tokens, num_experts] routing scores
        routing_bias: Optional [num_experts] routing bias
        top_k: Number of experts to select per token
        routing_method_type: Type of routing method (see RoutingMethodType enum)
        n_group: Number of expert groups (for DeepSeekV3)
        topk_group: Number of top groups (for DeepSeekV3)
        routed_scaling_factor: Scaling factor (for DeepSeekV3)

    Returns:
        selected_experts: [num_tokens, top_k] tensor of selected expert indices
    """
    num_tokens = routing_logits.shape[0]
    device = routing_logits.device

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        # Use fused_topk_deepseek for accurate DeepSeekV3 routing
        if n_group is None or topk_group is None or routed_scaling_factor is None:
            raise ValueError(
                "DeepSeekV3 routing requires n_group, topk_group, and routed_scaling_factor"
            )
        if routing_bias is None:
            routing_bias = torch.zeros(
                routing_logits.shape[1], device=device, dtype=routing_logits.dtype
            )

        # Allocate output tensors
        topk_values = torch.empty(num_tokens, top_k, device=device, dtype=torch.float32)
        topk_indices = torch.empty(num_tokens, top_k, device=device, dtype=torch.int32)

        fused_topk_deepseek(
            scores=routing_logits.float(),
            bias=routing_bias.float(),
            n_group=n_group,
            topk_group=topk_group,
            topk=top_k,
            routed_scaling_factor=routed_scaling_factor,
            topk_values=topk_values,
            topk_indices=topk_indices,
        )
        return topk_indices
    else:
        # For other routing methods, use simple top-k as approximation
        # This is accurate for Default, Renormalize, RenormalizeNaive, TopK
        # and approximate for Llama4
        _, selected_experts = compute_routing(routing_logits.float(), top_k)
        return selected_experts


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
    routing_method_type = args.routing_method_type
    use_shuffled_weight = args.use_shuffled_weight
    weight_layout = args.weight_layout
    is_cuda_graph_compatible = not args.no_cuda_graph
    activation_type = args.activation_type
    res = []

    backends = ["trtllm"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

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

    # Compute selected experts for accurate bandwidth calculation
    # Use the actual routing method to get correct expert assignments
    selected_experts = _compute_routing_for_method(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        top_k=top_k,
        routing_method_type=routing_method_type,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
    )

    # For FP4, we need to properly quantize weights and create scales
    use_ue8m0 = False

    # Calculate global scale factor for hidden states
    hidden_states_scale_global = calculate_fp4_global_scale(hidden_states)

    # Quantize weights using proper FP4 quantization
    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
        quantize_fp4_batched(gemm1_weights, num_experts, use_ue8m0, True)
    )
    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
        quantize_fp4_batched(gemm2_weights, num_experts, use_ue8m0, True)
    )

    # Quantize hidden states
    hidden_states_fp4_bytes, hidden_states_scale_fp4_bytes, _ = quantize_fp4(
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

    # Create scale scalars using shared utility
    output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar = (
        create_moe_output_scale_scalars(local_num_experts, device)
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] routing_logits.shape = {routing_logits.shape}")
        print(f"[VVERBOSE] hidden_states.shape = {hidden_states.shape}")
        print(f"[VVERBOSE] gemm1_weights_fp4.shape = {gemm1_weights_fp4.shape}")
        print(f"[VVERBOSE] gemm2_weights_fp4.shape = {gemm2_weights_fp4.shape}")

    def run_fp4_moe(
        routing_logits,
        routing_bias,
        hidden_states_fp4,
        hidden_states_scale_linear_fp4,
        gemm1_weights_fp4,
        gemm1_weights_scale,
        gemm2_weights_fp4,
        gemm2_weights_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
    ):
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
            routing_method_type=routing_method_type,
            activation_type=activation_type.value,
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
                run_fp4_moe(
                    routing_logits,
                    routing_bias,
                    hidden_states_fp4,
                    hidden_states_scale_linear_fp4,
                    gemm1_weights_fp4,
                    gemm1_weights_scale,
                    gemm2_weights_fp4,
                    gemm2_weights_scale,
                    output1_scale_scalar,
                    output1_scale_gate_scalar,
                    output2_scale_scalar,
                )

    # Benchmark timing
    times = bench_gpu_time(
        fn=run_fp4_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=(
            routing_logits,
            routing_bias,
            hidden_states_fp4,
            hidden_states_scale_linear_fp4,
            gemm1_weights_fp4,
            gemm1_weights_scale,
            gemm2_weights_fp4,
            gemm2_weights_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
        ),
    )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
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
        routing_logits_dtype=routing_logits.dtype,
        active_experts=int(selected_experts.unique().numel()),
        verbose=args.verbose,
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
        cur_res["n_group"] = n_group
        cur_res["topk_group"] = topk_group
        cur_res["routed_scaling_factor"] = routed_scaling_factor
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["routing_method"] = args.routing_method
        cur_res["use_shuffled_weight"] = use_shuffled_weight
        cur_res["weight_layout"] = weight_layout
        cur_res["use_routing_bias"] = args.use_routing_bias
        cur_res["use_routing_scales_on_input"] = args.use_routing_scales_on_input
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        cur_res["activation_type"] = args.activation_type.name
        res.append(cur_res)

    return res


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
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        backend = "cutlass_autotune"
        if args.verbose >= 1:
            print(f"[INFO] Autotune warmup for CUTLASS fused MoE: {warmup_iters} iters")
        with autotune(True):
            for _ in range(warmup_iters):
                run_cutlass(*input_args_for_bench)

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
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
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
    routing_method_type = args.routing_method_type
    use_shuffled_weight = args.use_shuffled_weight
    weight_layout = args.weight_layout
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []
    backends = ["trtllm"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

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

    # Compute selected experts for accurate bandwidth calculation
    # Use the actual routing method to get correct expert assignments
    selected_experts = _compute_routing_for_method(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        top_k=top_k,
        routing_method_type=routing_method_type,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
    )

    # For FP8 block scale, create quantized weights and block scales
    # Quantize to FP8
    gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
    gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)

    # Optionally shuffle weights and convert to BlockMajorK layout to match kernel expectation
    if use_shuffled_weight:
        gemm1_weights_fp8_shuffled = []
        gemm2_weights_fp8_shuffled = []
        for i in range(num_experts):
            tmp_w1 = process_fp8_weight_layout(
                gemm1_weights_fp8[i], use_shuffled_weight, weight_layout
            )
            tmp_w2 = process_fp8_weight_layout(
                gemm2_weights_fp8[i], use_shuffled_weight, weight_layout
            )
            gemm1_weights_fp8_shuffled.append(tmp_w1.view(torch.uint8))
            gemm2_weights_fp8_shuffled.append(tmp_w2.view(torch.uint8))

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

    def run_fp8_block_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        kernel_gemm1_weights,
        gemm1_weights_scale,
        kernel_gemm2_weights,
        gemm2_weights_scale,
    ):
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
            routing_method_type=routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            enable_pdl=True,
        )

    # Benchmark timing
    times = bench_gpu_time(
        fn=run_fp8_block_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            kernel_gemm1_weights,
            gemm1_weights_scale,
            kernel_gemm2_weights,
            gemm2_weights_scale,
        ),
    )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
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
        input_format="fp8",
        weight_format="fp8",
        routing_logits_dtype=routing_logits.dtype,
        active_experts=int(selected_experts.unique().numel()),
        verbose=args.verbose,
    )

    backend = "trtllm"
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
        cur_res["n_group"] = n_group
        cur_res["topk_group"] = topk_group
        cur_res["routed_scaling_factor"] = routed_scaling_factor
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
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
    routing_method_type = args.routing_method_type
    use_routing_scales_on_input = args.use_routing_scales_on_input
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []
    backends = ["trtllm"]
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

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

    # Compute selected experts for accurate bandwidth calculation
    # Use the actual routing method to get correct expert assignments
    selected_experts = _compute_routing_for_method(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        top_k=top_k,
        routing_method_type=routing_method_type,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
    )

    # For FP8 per-tensor scale, create quantized weights and per-tensor scales
    # Quantize to FP8
    gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
    gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)

    # Quantize hidden states to FP8 for per-tensor scale
    hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)

    # Create per-tensor scale scalars using shared utility
    output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar = (
        create_moe_output_scale_scalars(local_num_experts, device)
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] routing_logits.shape = {routing_logits.shape}")
        print(f"[VVERBOSE] hidden_states.shape = {hidden_states.shape}")
        print(f"[VVERBOSE] gemm1_weights_fp8.shape = {gemm1_weights_fp8.shape}")
        print(f"[VVERBOSE] gemm2_weights_fp8.shape = {gemm2_weights_fp8.shape}")

    def run_fp8_per_tensor_moe(
        routing_logits,
        routing_bias,
        hidden_states_fp8,
        gemm1_weights_fp8,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights_fp8,
        output2_scales_scalar,
        activation_type,
    ):
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
            routing_method_type=routing_method_type,
            activation_type=activation_type.value,
        )

    # Benchmark timing
    times = bench_gpu_time(
        fn=run_fp8_per_tensor_moe,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        sleep_after_run=False,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        cold_l2_cache=True,
        input_args=(
            routing_logits,
            routing_bias,
            hidden_states_fp8,
            gemm1_weights_fp8,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights_fp8,
            output2_scales_scalar,
            args.activation_type,
        ),
    )

    # Compute performance metrics
    median_time = np.median(times)
    std_time = np.std(times)
    tflops = calculate_moe_tflops(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, median_time
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
        input_format="fp8",
        weight_format="fp8",
        routing_logits_dtype=routing_logits.dtype,
        active_experts=int(selected_experts.unique().numel()),
        verbose=args.verbose,
    )

    backend = "trtllm"
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
        cur_res["n_group"] = n_group
        cur_res["topk_group"] = topk_group
        cur_res["routed_scaling_factor"] = routed_scaling_factor
        cur_res["local_expert_offset"] = local_expert_offset
        cur_res["local_num_experts"] = local_num_experts
        cur_res["routing_method"] = args.routing_method
        cur_res["use_routing_bias"] = args.use_routing_bias
        cur_res["use_routing_scales_on_input"] = use_routing_scales_on_input
        cur_res["input_dtype"] = input_dtype
        cur_res["weight_dtype"] = weight_dtype
        cur_res["activation_type"] = args.activation_type.name
        res.append(cur_res)

    return res
