import argparse
from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer import next_positive_power_of_2, reorder_rows_for_gated_act_gemm
from flashinfer.fused_moe import (
    RoutingMethodType,
    WeightLayout,
    convert_to_block_layout,
    shuffle_matrix_a,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
)
from flashinfer.testing.utils import (
    bench_gpu_time,
    bench_gpu_time_with_cudagraph,
    set_seed,
)


def run_moe_test(args):
    """
    Run a TensorRT-LLM MoE test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: List of dictionaries containing performance results
    """
    if args.routine == "trtllm_fp4_block_scale_moe":
        return test_trtllm_fp4_block_scale_moe(args)
    elif args.routine == "trtllm_fp8_block_scale_moe":
        return test_trtllm_fp8_block_scale_moe(args)
    elif args.routine == "trtllm_fp8_per_tensor_scale_moe":
        return test_trtllm_fp8_per_tensor_scale_moe(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_moe_args(line, parser):
    """
    Parse command line arguments for MoE test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    # MoE-specific arguments
    parser.add_argument(
        "--num_tokens", type=int, required=True, help="Number of tokens in the input"
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Hidden size of the model"
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=True,
        help="Intermediate size of the MoE layer",
    )
    parser.add_argument(
        "--num_experts", type=int, required=True, help="Total number of experts"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Number of experts to route to per token",
    )
    parser.add_argument(
        "--n_groups",
        type=int,
        required=False,
        default=1,
        help="Number of expert groups (used for DeepSeek-style routing)",
    )
    parser.add_argument(
        "--top_k_groups",
        type=int,
        required=False,
        default=1,
        help="Number of groups to consider for top-k routing",
    )
    parser.add_argument(
        "--routed_scaling_factor",
        type=float,
        required=False,
        default=1.0,
        help="Scaling factor for routing",
    )
    parser.add_argument(
        "--routing_method_type",
        type=str,
        required=False,
        default="Default",
        choices=["Default", "Renormalize", "DeepSeekV3", "Llama4", "RenormalizeNaive"],
        help="Type of routing method to use",
    )
    parser.add_argument(
        "--use_shuffled_weight",
        action="store_true",
        default=False,
        help="Use shuffled weight layout for better performance (FP8 block scale only)",
    )
    parser.add_argument(
        "--weight_layout",
        type=str,
        required=False,
        default="MajorK",
        choices=["MajorK", "MajorMn", "BlockMajorK"],
        help="Weight layout for FP8 block scale MoE",
    )
    parser.add_argument(
        "--local_expert_offset",
        type=int,
        required=False,
        default=0,
        help="Offset of local experts in global expert space",
    )
    parser.add_argument(
        "--local_num_experts",
        type=int,
        required=False,
        default=None,
        help="Number of experts handled by this device (defaults to num_experts)",
    )

    args = parser.parse_args(line)

    # Set defaults and validation
    if args.local_num_experts is None:
        args.local_num_experts = args.num_experts

    # Validate configuration
    if args.top_k > args.num_experts:
        raise ValueError(
            f"top_k ({args.top_k}) cannot be greater than num_experts ({args.num_experts})"
        )

    if args.top_k > 8:
        raise ValueError(f"top_k ({args.top_k}) cannot be greater than 8")

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def calculate_tile_tokens_dim(num_tokens: int, num_experts: int, top_k: int) -> int:
    """Calculate optimal tile tokens dimension for the MoE kernel."""
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = num_tokens * top_k // num_experts

    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim


def create_test_data(args):
    """Create test data for MoE benchmarking."""
    device = "cuda"
    dtype = torch.bfloat16

    # Convert routing method string to enum
    routing_method_map = {
        "Default": RoutingMethodType.Default,
        "Renormalize": RoutingMethodType.Renormalize,
        "DeepSeekV3": RoutingMethodType.DeepSeekV3,
        "Llama4": RoutingMethodType.Llama4,
        "RenormalizeNaive": RoutingMethodType.RenormalizeNaive,
    }
    routing_method_type = routing_method_map[args.routing_method_type]

    # Convert weight layout string to enum
    weight_layout_map = {
        "MajorK": WeightLayout.MajorK,
        "MajorMn": WeightLayout.MajorMn,
        "BlockMajorK": WeightLayout.BlockMajorK,
    }
    weight_layout = weight_layout_map[args.weight_layout]

    # Calculate tile tokens dimension
    tile_tokens_dim = calculate_tile_tokens_dim(
        args.num_tokens, args.num_experts, args.top_k
    )

    # Create routing logits and bias
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn(
            (args.num_tokens, args.num_experts), device=device, dtype=torch.float
        )
    else:
        expert_logits = torch.randn(
            (args.num_tokens, args.num_experts), device=device, dtype=dtype
        )

    routing_bias = (
        torch.randn(args.num_experts, device=device, dtype=dtype)
        if routing_method_type
        in [RoutingMethodType.DeepSeekV3, RoutingMethodType.Llama4]
        else None
    )

    # Create hidden states
    hidden_states = torch.randn(
        (args.num_tokens, args.hidden_size), device=device, dtype=dtype
    )

    return {
        "expert_logits": expert_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "routing_method_type": routing_method_type,
        "weight_layout": weight_layout,
        "tile_tokens_dim": tile_tokens_dim,
    }


def test_trtllm_fp4_block_scale_moe(args):
    """
    Test trtllm_fp4_block_scale_moe API.

    This test:
    1. Creates quantized FP4 weights and scales
    2. Runs FP4 MoE with different configurations
    3. Measures performance metrics

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print(f"[INFO] Running test_trtllm_fp4_block_scale_moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")

    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")

    # Create test data
    test_data = create_test_data(args)

    # Create FP4 quantized weights and scales
    # Hidden states: FP4 quantized to [num_tokens, hidden_size//2]
    hidden_states_fp4 = (
        test_data["hidden_states"]
        .view(torch.uint8)
        .reshape(args.num_tokens, args.hidden_size // 2)
    )
    hidden_states_scale = torch.randn(
        args.hidden_size // 16,
        args.num_tokens,
        device=device,
        dtype=torch.float,
    ).to(torch.float8_e4m3fn)

    # GEMM1 weights: [num_experts, 2*intermediate_size, hidden_size//2] (FP4 packed)
    gemm1_weights = torch.randint(
        0,
        255,
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    gemm1_weights_scale = torch.randn(
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size // 16),
        device=device,
        dtype=torch.float,
    ).to(torch.float8_e4m3fn)

    # GEMM2 weights: [num_experts, hidden_size, intermediate_size//2] (FP4 packed)
    gemm2_weights = torch.randint(
        0,
        255,
        (args.num_experts, args.hidden_size, args.intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    gemm2_weights_scale = torch.randn(
        (args.num_experts, args.hidden_size, args.intermediate_size // 16),
        device=device,
        dtype=torch.float,
    ).to(torch.float8_e4m3fn)

    # Output scaling factors
    output1_scale_scalar = torch.randn(
        args.local_num_experts, device=device, dtype=torch.float
    )
    output1_scale_gate_scalar = torch.randn(
        args.local_num_experts, device=device, dtype=torch.float
    )
    output2_scale_scalar = torch.randn(
        args.local_num_experts, device=device, dtype=torch.float
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] hidden_states_fp4.shape = {hidden_states_fp4.shape}")
        print(f"[VVERBOSE] gemm1_weights.shape = {gemm1_weights.shape}")
        print(f"[VVERBOSE] gemm2_weights.shape = {gemm2_weights.shape}")
        print(f"[VVERBOSE] routing_method_type = {test_data['routing_method_type']}")
        print(f"[VVERBOSE] tile_tokens_dim = {test_data['tile_tokens_dim']}")

    # Warmup run
    try:
        output = trtllm_fp4_block_scale_moe(
            test_data["expert_logits"],
            test_data["routing_bias"],
            hidden_states_fp4,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            args.num_experts,
            args.top_k,
            args.n_groups,
            args.top_k_groups,
            args.intermediate_size,
            args.local_expert_offset,
            args.local_num_experts,
            args.routed_scaling_factor,
            test_data["tile_tokens_dim"],
            test_data["routing_method_type"],
            do_finalize=True,
        )

        if args.verbose >= 2:
            print(f"[VVERBOSE] output.shape = {output[0].shape}")

    except Exception as e:
        print(f"[ERROR] Failed to run FP4 MoE: {e}")
        return []

    # Benchmark function
    def benchmark_fn():
        return trtllm_fp4_block_scale_moe(
            test_data["expert_logits"],
            test_data["routing_bias"],
            hidden_states_fp4,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            args.num_experts,
            args.top_k,
            args.n_groups,
            args.top_k_groups,
            args.intermediate_size,
            args.local_expert_offset,
            args.local_num_experts,
            args.routed_scaling_factor,
            test_data["tile_tokens_dim"],
            test_data["routing_method_type"],
            do_finalize=True,
        )

    # Run benchmark
    is_cuda_graph_compatible = not args.no_cuda_graph
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=benchmark_fn,
            dry_runs=args.dry_run_iters,
            num_iters_within_graph=20,
            num_iters=args.num_iters,
            nvtx_range_name="trtllm_fp4_moe",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=benchmark_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name="trtllm_fp4_moe",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    # Compute metrics
    res = []
    if len(times) > 0:
        median_time = np.median(times)
        std_time = np.std(times)

        # Calculate FLOPs (approximate)
        # For MoE: FLOPs ≈ num_tokens * top_k * (2 * hidden_size * intermediate_size * 2)
        # Factor of 2 for GEMM1 (2*intermediate_size), factor of 2 for both GEMM1 and GEMM2
        flops = (
            args.num_tokens
            * args.top_k
            * (4 * args.hidden_size * args.intermediate_size)
        )
        tflops = (flops / (median_time * 1e-3)) / 1e12

        # Calculate memory throughput (approximate)
        # Input: num_tokens * hidden_size * 2 bytes (bfloat16)
        # Weights: num_experts * (2*intermediate_size*hidden_size + hidden_size*intermediate_size) * 0.5 bytes (FP4)
        # Output: num_tokens * hidden_size * 2 bytes
        input_bytes = args.num_tokens * args.hidden_size * 2  # bfloat16
        weight_bytes = (
            args.num_experts * (3 * args.intermediate_size * args.hidden_size) * 0.5
        )  # FP4
        output_bytes = args.num_tokens * args.hidden_size * 2  # bfloat16
        total_bytes = input_bytes + weight_bytes + output_bytes
        tb_per_sec = (total_bytes / (median_time * 1e-3)) / 1e12

        print(
            f"[PERF] FP4_MoE :: median time {median_time:.3f} ms; std {std_time:.3f} ms; "
            f"achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
        )

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = "trtllm_fp4_block_scale_moe"
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["num_tokens"] = args.num_tokens
            cur_res["hidden_size"] = args.hidden_size
            cur_res["intermediate_size"] = args.intermediate_size
            cur_res["num_experts"] = args.num_experts
            cur_res["top_k"] = args.top_k
            cur_res["n_groups"] = args.n_groups
            cur_res["top_k_groups"] = args.top_k_groups
            cur_res["routing_method_type"] = args.routing_method_type
            cur_res["routed_scaling_factor"] = args.routed_scaling_factor
            cur_res["tile_tokens_dim"] = test_data["tile_tokens_dim"]
            cur_res["use_shuffled_weight"] = False  # FP4 always uses shuffled
            cur_res["weight_layout"] = "MajorK"  # FP4 always uses MajorK
            res.append(cur_res)

    return res


def test_trtllm_fp8_block_scale_moe(args):
    """
    Test trtllm_fp8_block_scale_moe API.

    This test:
    1. Creates quantized FP8 weights and block scales
    2. Runs FP8 block scale MoE with different configurations
    3. Measures performance metrics

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print(f"[INFO] Running test_trtllm_fp8_block_scale_moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")

    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")

    # Create test data
    test_data = create_test_data(args)

    # Create FP8 quantized weights and block scales
    hidden_states_fp8 = test_data["hidden_states"].to(torch.float8_e4m3fn)
    # Use deterministic scales for testing consistency (matching reference implementation)
    hidden_states_scale = 2.0 * torch.ones(
        (args.hidden_size // 128, args.num_tokens), device=device, dtype=torch.float
    )

    # Create base weights in float format first
    gemm1_weights_base = torch.randn(
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size),
        device=device,
        dtype=torch.float,
    )
    gemm2_weights_base = torch.randn(
        (args.num_experts, args.hidden_size, args.intermediate_size),
        device=device,
        dtype=torch.float,
    )

    # Convert to FP8 and generate scales (matching reference implementation)
    gemm1_weights = gemm1_weights_base.to(torch.float8_e4m3fn)
    gemm1_weights_scale = 2 * torch.rand(
        (args.num_experts, 2 * args.intermediate_size // 128, args.hidden_size // 128),
        device=device,
    ).to(torch.float)

    gemm2_weights = gemm2_weights_base.to(torch.float8_e4m3fn)
    gemm2_weights_scale = 2 * torch.rand(
        (args.num_experts, args.hidden_size // 128, args.intermediate_size // 128),
        device=device,
    ).to(torch.float)

    # Apply weight processing (shuffling and layout conversion) if requested
    if args.use_shuffled_weight:
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 64  # For FP8 block scale

        gemm1_weights_shuffled = []
        gemm2_weights_shuffled = []
        for i in range(args.num_experts):
            tmp_weights1 = shuffle_matrix_a(
                gemm1_weights[i].view(torch.uint8), epilogue_tile_m
            )
            tmp_weights2 = shuffle_matrix_a(
                gemm2_weights[i].view(torch.uint8), epilogue_tile_m
            )

            if test_data["weight_layout"] == WeightLayout.BlockMajorK:
                block_k = 128
                tmp_weights1 = convert_to_block_layout(tmp_weights1, block_k)
                tmp_weights2 = convert_to_block_layout(tmp_weights2, block_k)

            gemm1_weights_shuffled.append(tmp_weights1)
            gemm2_weights_shuffled.append(tmp_weights2)

        gemm1_weights = torch.stack(gemm1_weights_shuffled).view(torch.float8_e4m3fn)
        gemm2_weights = torch.stack(gemm2_weights_shuffled).view(torch.float8_e4m3fn)

    if args.verbose >= 2:
        print(f"[VVERBOSE] hidden_states_fp8.shape = {hidden_states_fp8.shape}")
        print(f"[VVERBOSE] gemm1_weights.shape = {gemm1_weights.shape}")
        print(f"[VVERBOSE] gemm2_weights.shape = {gemm2_weights.shape}")
        print(f"[VVERBOSE] routing_method_type = {test_data['routing_method_type']}")
        print(f"[VVERBOSE] use_shuffled_weight = {args.use_shuffled_weight}")
        print(f"[VVERBOSE] weight_layout = {test_data['weight_layout']}")
        print(f"[VVERBOSE] tile_tokens_dim = {test_data['tile_tokens_dim']}")

    # Warmup run
    try:
        output = trtllm_fp8_block_scale_moe(
            test_data["expert_logits"],
            test_data["routing_bias"],
            hidden_states_fp8,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            args.num_experts,
            args.top_k,
            args.n_groups,
            args.top_k_groups,
            args.intermediate_size,
            args.local_expert_offset,
            args.local_num_experts,
            args.routed_scaling_factor,
            test_data["tile_tokens_dim"],
            test_data["routing_method_type"],
            use_shuffled_weight=args.use_shuffled_weight,
            weight_layout=int(test_data["weight_layout"]),
        )

        if args.verbose >= 2:
            print(f"[VVERBOSE] output.shape = {output.shape}")

    except Exception as e:
        print(f"[ERROR] Failed to run FP8 block scale MoE: {e}")
        return []

    # Benchmark function
    def benchmark_fn():
        return trtllm_fp8_block_scale_moe(
            test_data["expert_logits"],
            test_data["routing_bias"],
            hidden_states_fp8,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            args.num_experts,
            args.top_k,
            args.n_groups,
            args.top_k_groups,
            args.intermediate_size,
            args.local_expert_offset,
            args.local_num_experts,
            args.routed_scaling_factor,
            test_data["tile_tokens_dim"],
            test_data["routing_method_type"],
            use_shuffled_weight=args.use_shuffled_weight,
            weight_layout=int(test_data["weight_layout"]),
        )

    # Run benchmark
    is_cuda_graph_compatible = not args.no_cuda_graph
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=benchmark_fn,
            dry_runs=args.dry_run_iters,
            num_iters_within_graph=20,
            num_iters=args.num_iters,
            nvtx_range_name="trtllm_fp8_block_moe",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=benchmark_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name="trtllm_fp8_block_moe",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    # Compute metrics
    res = []
    if len(times) > 0:
        median_time = np.median(times)
        std_time = np.std(times)

        # Calculate FLOPs (approximate)
        flops = (
            args.num_tokens
            * args.top_k
            * (4 * args.hidden_size * args.intermediate_size)
        )
        tflops = (flops / (median_time * 1e-3)) / 1e12

        # Calculate memory throughput (approximate)
        input_bytes = args.num_tokens * args.hidden_size * 1  # FP8
        weight_bytes = (
            args.num_experts * (3 * args.intermediate_size * args.hidden_size) * 1
        )  # FP8
        output_bytes = args.num_tokens * args.hidden_size * 2  # bfloat16
        total_bytes = input_bytes + weight_bytes + output_bytes
        tb_per_sec = (total_bytes / (median_time * 1e-3)) / 1e12

        print(
            f"[PERF] FP8_Block:: median time {median_time:.3f} ms; std {std_time:.3f} ms; "
            f"achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
        )

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = "trtllm_fp8_block_scale_moe"
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["num_tokens"] = args.num_tokens
            cur_res["hidden_size"] = args.hidden_size
            cur_res["intermediate_size"] = args.intermediate_size
            cur_res["num_experts"] = args.num_experts
            cur_res["top_k"] = args.top_k
            cur_res["n_groups"] = args.n_groups
            cur_res["top_k_groups"] = args.top_k_groups
            cur_res["routing_method_type"] = args.routing_method_type
            cur_res["routed_scaling_factor"] = args.routed_scaling_factor
            cur_res["tile_tokens_dim"] = test_data["tile_tokens_dim"]
            cur_res["use_shuffled_weight"] = args.use_shuffled_weight
            cur_res["weight_layout"] = args.weight_layout
            res.append(cur_res)

    return res


def test_trtllm_fp8_per_tensor_scale_moe(args):
    """
    Test trtllm_fp8_per_tensor_scale_moe API.

    This test:
    1. Creates quantized FP8 weights with per-tensor scales
    2. Runs FP8 per-tensor scale MoE with different configurations
    3. Measures performance metrics

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print(f"[INFO] Running test_trtllm_fp8_per_tensor_scale_moe")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")

    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")

    # Create test data
    test_data = create_test_data(args)

    # Create FP8 quantized weights with per-tensor scales
    hidden_states_fp8 = test_data["hidden_states"].to(torch.float8_e4m3fn)

    # Create base weights in float format first
    gemm1_weights_base = torch.randn(
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size),
        device=device,
        dtype=torch.float,
    )
    gemm2_weights_base = torch.randn(
        (args.num_experts, args.hidden_size, args.intermediate_size),
        device=device,
        dtype=torch.float,
    )

    # Convert to FP8 (matching reference implementation)
    gemm1_weights = gemm1_weights_base.to(torch.float8_e4m3fn)
    gemm2_weights = gemm2_weights_base.to(torch.float8_e4m3fn)

    # Apply weight processing for per-tensor scale (matching reference implementation)
    # FIXME: this depends on the kernel internals
    epilogue_tile_m = 128  # For FP8 per-tensor scale

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(args.num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i].clone())
        )

    # Stack weights for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        args.num_experts, 2 * args.intermediate_size, args.hidden_size
    )

    # Shuffle weights for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(args.num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights = torch.stack(gemm1_weights_fp8_shuffled).view(torch.float8_e4m3fn)
    gemm2_weights = torch.stack(gemm2_weights_fp8_shuffled).view(torch.float8_e4m3fn)

    # Per-tensor scaling factors
    output1_scales_scalar = torch.randn(
        args.local_num_experts, device=device, dtype=torch.float
    )
    output1_scales_gate_scalar = torch.randn(
        args.local_num_experts, device=device, dtype=torch.float
    )
    output2_scales_scalar = torch.randn(
        args.local_num_experts, device=device, dtype=torch.float
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] hidden_states_fp8.shape = {hidden_states_fp8.shape}")
        print(f"[VVERBOSE] gemm1_weights.shape = {gemm1_weights.shape}")
        print(f"[VVERBOSE] gemm2_weights.shape = {gemm2_weights.shape}")
        print(f"[VVERBOSE] routing_method_type = {test_data['routing_method_type']}")
        print(f"[VVERBOSE] tile_tokens_dim = {test_data['tile_tokens_dim']}")

    # Determine use_routing_scales_on_input
    use_routing_scales_on_input = (
        test_data["routing_method_type"] == RoutingMethodType.Llama4
    )

    # Warmup run
    try:
        output = trtllm_fp8_per_tensor_scale_moe(
            test_data["expert_logits"],
            test_data["routing_bias"],
            hidden_states_fp8,
            gemm1_weights,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights,
            output2_scales_scalar,
            args.num_experts,
            args.top_k,
            args.n_groups,
            args.top_k_groups,
            args.intermediate_size,
            args.local_expert_offset,
            args.local_num_experts,
            args.routed_scaling_factor,
            use_routing_scales_on_input,
            test_data["tile_tokens_dim"],
            test_data["routing_method_type"],
        )

        if args.verbose >= 2:
            print(f"[VVERBOSE] output.shape = {output.shape}")

    except Exception as e:
        print(f"[ERROR] Failed to run FP8 per-tensor scale MoE: {e}")
        return []

    # Benchmark function
    def benchmark_fn():
        return trtllm_fp8_per_tensor_scale_moe(
            test_data["expert_logits"],
            test_data["routing_bias"],
            hidden_states_fp8,
            gemm1_weights,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights,
            output2_scales_scalar,
            args.num_experts,
            args.top_k,
            args.n_groups,
            args.top_k_groups,
            args.intermediate_size,
            args.local_expert_offset,
            args.local_num_experts,
            args.routed_scaling_factor,
            use_routing_scales_on_input,
            test_data["tile_tokens_dim"],
            test_data["routing_method_type"],
        )

    # Run benchmark
    is_cuda_graph_compatible = not args.no_cuda_graph
    if is_cuda_graph_compatible:
        times = bench_gpu_time_with_cudagraph(
            fn=benchmark_fn,
            dry_runs=args.dry_run_iters,
            num_iters_within_graph=20,
            num_iters=args.num_iters,
            nvtx_range_name="trtllm_fp8_tensor_moe",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )
    else:
        times = bench_gpu_time(
            fn=benchmark_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name="trtllm_fp8_tensor_moe",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=False,
        )

    # Compute metrics
    res = []
    if len(times) > 0:
        median_time = np.median(times)
        std_time = np.std(times)

        # Calculate FLOPs (approximate)
        flops = (
            args.num_tokens
            * args.top_k
            * (4 * args.hidden_size * args.intermediate_size)
        )
        tflops = (flops / (median_time * 1e-3)) / 1e12

        # Calculate memory throughput (approximate)
        input_bytes = args.num_tokens * args.hidden_size * 1  # FP8
        weight_bytes = (
            args.num_experts * (3 * args.intermediate_size * args.hidden_size) * 1
        )  # FP8
        output_bytes = args.num_tokens * args.hidden_size * 2  # bfloat16
        total_bytes = input_bytes + weight_bytes + output_bytes
        tb_per_sec = (total_bytes / (median_time * 1e-3)) / 1e12

        print(
            f"[PERF] FP8_Tensor:: median time {median_time:.3f} ms; std {std_time:.3f} ms; "
            f"achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
        )

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = "trtllm_fp8_per_tensor_scale_moe"
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["num_tokens"] = args.num_tokens
            cur_res["hidden_size"] = args.hidden_size
            cur_res["intermediate_size"] = args.intermediate_size
            cur_res["num_experts"] = args.num_experts
            cur_res["top_k"] = args.top_k
            cur_res["n_groups"] = args.n_groups
            cur_res["top_k_groups"] = args.top_k_groups
            cur_res["routing_method_type"] = args.routing_method_type
            cur_res["routed_scaling_factor"] = args.routed_scaling_factor
            cur_res["tile_tokens_dim"] = test_data["tile_tokens_dim"]
            cur_res["use_shuffled_weight"] = True  # FP8 per-tensor always uses shuffled
            cur_res["weight_layout"] = "MajorK"  # FP8 per-tensor always uses MajorK
            res.append(cur_res)

    return res
