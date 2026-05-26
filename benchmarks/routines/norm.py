"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    is_close_stats,
    filter_backends_by_compute_capability,
)


def run_norm_test(args):
    """
    Run a norm test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "rmsnorm":
        return testRmsnorm(args)
    elif args.routine == "rmsnorm_quant":
        return testRmsnormQuant(args)
    elif args.routine == "fused_add_rmsnorm_quant":
        return testFusedAddRmsnormQuant(args)
    elif args.routine == "rmsnorm_fp4quant":
        return testRmsnormFp4quant(args)
    elif args.routine == "add_rmsnorm_fp4quant":
        return testAddRmsnormFp4quant(args)
    elif args.routine == "fused_rmsnorm_silu":
        return testFusedRmsnormSilu(args)
    elif args.routine == "fused_dit_layernorm":
        return testFusedDitLayernorm(args)
    elif args.routine == "fused_qk_rmsnorm_rope":
        return testFusedQkRmsnormRope(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_norm_args(line, parser):
    """
    Parse command line arguments for norm test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=True,
        help="Hidden dimension size.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=False,
        default=None,
        help="Number of heads (for 3D input shape). If not specified, uses 2D shape.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the input tensor.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        required=False,
        default=1e-6,
        help="Epsilon for numerical stability.",
    )
    parser.add_argument(
        "--enable_pdl",
        action="store_true",
        default=False,
        help="Enable programmatic dependent launch.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        default=1.0,
        help="Scale factor for quantization (used by rmsnorm_quant and fused_add_rmsnorm_quant).",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        choices=["fp8_e4m3", "fp8_e5m2", "nvfp4", "mxfp4"],
        help="Output dtype for quantized operations. fp8_e4m3/fp8_e5m2 for FP8 quant; nvfp4/mxfp4 for FP4 quant.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda", "cute-dsl"],
        help="Backend to test. Default: cuda. Use cute-dsl for FP4 quantization.",
    )
    # FP4 quantization specific arguments (for rmsnorm_fp4quant, add_rmsnorm_fp4quant)
    parser.add_argument(
        "--use_global_scale",
        action="store_true",
        default=False,
        help="Use global scale factor (NVFP4 format). Default: False",
    )
    parser.add_argument(
        "--dit_mode",
        type=str,
        required=False,
        default="gate_residual_gamma_beta",
        choices=[
            "gate_residual_gamma_beta",
            "gate_residual_scale_shift",
            "residual_scale_shift",
        ],
        help="DIT LayerNorm mode (for fused_dit_layernorm routine only).",
    )
    parser.add_argument(
        "--is_sf_swizzled_layout",
        action="store_true",
        default=False,
        help="Use swizzled scale factor layout for tensor core GEMM. Default: False",
    )
    parser.add_argument(
        "--output_both_sf_layouts",
        action="store_true",
        default=False,
        help="Output both swizzled and unswizzled scale factors. When enabled, "
        "overrides --is_sf_swizzled_layout and returns both layouts. Default: False",
    )

    # fused_qk_rmsnorm_rope specific arguments
    parser.add_argument(
        "--ppf",
        type=int,
        required=False,
        default=5,
        help="Number of patches in frame dimension (for fused_qk_rmsnorm_rope).",
    )
    parser.add_argument(
        "--pph",
        type=int,
        required=False,
        default=12,
        help="Number of patches in height dimension (for fused_qk_rmsnorm_rope).",
    )
    parser.add_argument(
        "--ppw",
        type=int,
        required=False,
        default=32,
        help="Number of patches in width dimension (for fused_qk_rmsnorm_rope).",
    )

    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testRmsnorm(args):
    """
    Test rmsnorm API.

    This test:
    1. Generates random input tensors
    2. Runs rmsnorm
    3. Runs reference check
    4. Measures performance metrics (memory bandwidth)

    Note: RMSNorm is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testRmsnorm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    eps = args.eps
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )
    ## Done parsing input arguments

    ## Prepare input tensors
    if num_heads is not None:
        input_shape = (batch_size, num_heads, hidden_size)
    else:
        input_shape = (batch_size, hidden_size)

    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {weight.shape = }")

    def run_backend(backend, input_tensor, weight):
        if backend == "cuda":
            return flashinfer.rmsnorm(
                input_tensor, weight, eps=eps, enable_pdl=enable_pdl
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference: PyTorch implementation of RMSNorm
    has_reference_output = False
    if run_refcheck:
        rms = torch.sqrt(
            torch.mean(input_tensor.float() ** 2, dim=-1, keepdim=True) + eps
        )
        reference_output = (input_tensor.float() / rms * weight.float()).to(input_dtype)
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_tensor, weight
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor, weight),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(
                    reference_output, tested_outputs[i], rtol=1e-2, atol=1e-2
                )
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for RMSNorm
            # Read: input tensor + weight tensor
            # Write: output tensor (same shape as input)
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements * input_dtype.itemsize  # output write
            )
            # RMSNorm is memory-bound, so TFLOPS is not the primary metric
            # But we compute approximate FLOPS for completeness:
            # Per element: square, sum reduction, sqrt, divide, multiply
            problem_flops = num_elements * 5  # rough estimate
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["num_heads"] = num_heads if num_heads else ""
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testRmsnormQuant(args):
    """
    Test rmsnorm_quant API.

    This test:
    1. Generates random input tensors
    2. Runs rmsnorm_quant with quantized output
    3. Runs reference check
    4. Measures performance metrics (memory bandwidth)

    Note: RMSNorm is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testRmsnormQuant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    scale = args.scale
    eps = args.eps
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported out dtype: {args.out_dtype}. Supported dtypes are fp8_e4m3, fp8_e5m2."
        )
    ## Done parsing input arguments

    ## Prepare input tensors (2D only for rmsnorm_quant)
    input_shape = (batch_size, hidden_size)

    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)
    out_tensor = torch.empty(input_shape, dtype=out_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {weight.shape = }")
        print(f"[VVERBOSE] {out_tensor.dtype = }")
        print(f"[VVERBOSE] {scale = }")

    def run_backend(backend, out_tensor, input_tensor, weight):
        if backend == "cuda":
            flashinfer.norm.rmsnorm_quant(
                out_tensor,
                input_tensor,
                weight,
                scale=scale,
                eps=eps,
                enable_pdl=enable_pdl,
            )
            return out_tensor
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference: PyTorch implementation of RMSNorm + quantization
    has_reference_output = False
    if run_refcheck:
        rms = torch.sqrt(
            torch.mean(input_tensor.float() ** 2, dim=-1, keepdim=True) + eps
        )
        rmsnorm_output = input_tensor.float() / rms * weight.float()
        # Quantize to output dtype
        reference_output = (
            (rmsnorm_output * scale)
            .clamp(torch.finfo(out_dtype).min, torch.finfo(out_dtype).max)
            .to(out_dtype)
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        # Create fresh output tensor for each run
        cur_out = torch.empty(input_shape, dtype=out_dtype, device=device)
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend(cur_backend, cur_out, input_tensor, weight).detach().clone()
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, out_tensor, input_tensor, weight),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                # Compare in float for FP8 outputs
                ref_float = reference_output.float()
                out_float = tested_outputs[i].float()
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(ref_float, out_float, rtol=1e-1, atol=1e-1)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for RMSNorm + Quant
            # Read: input tensor + weight tensor
            # Write: output tensor (quantized, smaller dtype)
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements * out_dtype.itemsize  # output write (quantized)
            )
            problem_flops = num_elements * 5  # rough estimate
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = str(out_dtype)
                cur_res["scale"] = scale
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testFusedAddRmsnormQuant(args):
    """
    Test fused_add_rmsnorm_quant API.

    This test:
    1. Generates random input and residual tensors
    2. Runs fused_add_rmsnorm_quant (residual += input, then RMSNorm with quantized output)
    3. Runs reference check
    4. Measures performance metrics (memory bandwidth)

    Note: This operation is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedAddRmsnormQuant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    scale = args.scale
    eps = args.eps
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported out dtype: {args.out_dtype}. Supported dtypes are fp8_e4m3, fp8_e5m2."
        )
    ## Done parsing input arguments

    ## Prepare input tensors (2D only for fused_add_rmsnorm_quant)
    input_shape = (batch_size, hidden_size)

    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    residual_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)
    out_tensor = torch.empty(input_shape, dtype=out_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {residual_tensor.shape = }")
        print(f"[VVERBOSE] {weight.shape = }")
        print(f"[VVERBOSE] {out_tensor.dtype = }")
        print(f"[VVERBOSE] {scale = }")

    def run_backend(backend, out_tensor, input_tensor, residual_tensor, weight):
        if backend == "cuda":
            flashinfer.norm.fused_add_rmsnorm_quant(
                out_tensor,
                input_tensor,
                residual_tensor,
                weight,
                scale=scale,
                eps=eps,
                enable_pdl=enable_pdl,
            )
            return out_tensor
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference: PyTorch implementation of fused add + RMSNorm + quantization
    has_reference_output = False
    if run_refcheck:
        # Clone residual for reference computation since it gets modified
        ref_residual = residual_tensor.clone()
        # Step 1: residual += input
        ref_residual = ref_residual + input_tensor
        # Step 2: RMSNorm on residual
        rms = torch.sqrt(
            torch.mean(ref_residual.float() ** 2, dim=-1, keepdim=True) + eps
        )
        rmsnorm_output = ref_residual.float() / rms * weight.float()
        # Quantize to output dtype
        reference_output = (
            (rmsnorm_output * scale)
            .clamp(torch.finfo(out_dtype).min, torch.finfo(out_dtype).max)
            .to(out_dtype)
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        # Create fresh tensors for each run (residual is mutated)
        cur_out = torch.empty(input_shape, dtype=out_dtype, device=device)
        cur_residual = residual_tensor.clone()
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend(cur_backend, cur_out, input_tensor, cur_residual, weight)
                .detach()
                .clone()
            )
        # For timing, use fresh residual each iteration
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(
                cur_backend,
                out_tensor,
                input_tensor,
                residual_tensor.clone(),
                weight,
            ),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                # Compare in float for FP8 outputs
                ref_float = reference_output.float()
                out_float = tested_outputs[i].float()
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(ref_float, out_float, rtol=1e-1, atol=1e-1)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for Fused Add + RMSNorm + Quant
            # Read: input tensor + residual tensor + weight tensor
            # Write: residual tensor (updated) + output tensor (quantized)
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + num_elements * input_dtype.itemsize  # residual read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements * input_dtype.itemsize  # residual write
                + num_elements * out_dtype.itemsize  # output write (quantized)
            )
            problem_flops = num_elements * 6  # rough estimate (add + rmsnorm ops)
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = str(out_dtype)
                cur_res["scale"] = scale
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testRmsnormFp4quant(args):
    """
    Test rmsnorm_fp4quant API from flashinfer.cute_dsl.

    This test:
    1. Generates random input tensors
    2. Runs rmsnorm_fp4quant with FP4 quantized output
    3. Runs reference check
    4. Measures performance metrics (memory bandwidth)

    Note: This is a CuTe-DSL kernel requiring SM10.0+ (Blackwell).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testRmsnormFp4quant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original

    # Default backend to cute-dsl for FP4 quantization routines
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    eps = args.eps
    out_dtype = args.out_dtype
    use_global_scale = args.use_global_scale
    is_sf_swizzled_layout = args.is_sf_swizzled_layout
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    # Default to nvfp4 if FP8 dtype is specified (for backwards compatibility)
    if out_dtype in ["fp8_e4m3", "fp8_e5m2"]:
        out_dtype = "nvfp4"

    # Derive block_size from out_dtype
    # nvfp4: block_size=16, e4m3 scale factors
    # mxfp4: block_size=32, ue8m0 scale factors
    if out_dtype == "nvfp4":
        block_size = 16
    elif out_dtype == "mxfp4":
        block_size = 32
    else:
        raise ValueError(
            f"Unsupported out_dtype for FP4 quant: {out_dtype}. Supported: nvfp4, mxfp4."
        )

    # Validate alignment: hidden_size must be divisible by block_size
    if hidden_size % block_size != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by block_size ({block_size}) "
            f"for {out_dtype} quantization."
        )

    enable_pdl = args.enable_pdl

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )
    ## Done parsing input arguments

    ## Prepare input tensors
    if num_heads is not None:
        input_shape = (batch_size, num_heads, hidden_size)
    else:
        input_shape = (batch_size, hidden_size)

    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    # Prepare global_scale if using NVFP4 format
    # Note: API expects a 1D tensor of shape [1], not a 0D scalar
    global_scale = None
    if use_global_scale:
        global_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {weight.shape = }")
        print(f"[VVERBOSE] {out_dtype = }")
        print(f"[VVERBOSE] {block_size = }")
        print(f"[VVERBOSE] {use_global_scale = }")
        print(f"[VVERBOSE] {is_sf_swizzled_layout = }")
        print(f"[VVERBOSE] {enable_pdl = }")

    # Warn user that refcheck is not supported for FP4 quantization fusion
    if run_refcheck:
        print("[WARNING] --refcheck is not supported for rmsnorm_fp4quant.")

    # Warn user that output_both_sf_layouts is not supported for rmsnorm_fp4quant
    if args.output_both_sf_layouts:
        print(
            "[WARNING] --output_both_sf_layouts is not supported for rmsnorm_fp4quant. "
            "Use add_rmsnorm_fp4quant instead. Flag will be ignored."
        )

    def run_backend(backend, input_tensor, weight):
        if backend == "cute-dsl":
            return flashinfer.rmsnorm_fp4quant(
                input_tensor,
                weight,
                eps=eps,
                block_size=block_size,
                global_scale=global_scale,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
                enable_pdl=enable_pdl,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor, weight),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for RMSNorm + FP4 Quant
            # Read: input tensor + weight tensor
            # Write: FP4 output (2 elements per byte) + scale factors
            num_elements = np.prod(input_shape)
            num_scale_elements = num_elements // block_size
            # FP4: 2 elements per byte (4 bits each), packed in float4_e2m1fn_x2
            fp4_output_bytes = num_elements // 2
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + hidden_size * input_dtype.itemsize  # weight read
                + fp4_output_bytes  # FP4 output write
                + num_scale_elements  # scale factors write (1 byte each)
            )
            problem_flops = num_elements * 5  # rough estimate
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["num_heads"] = num_heads if num_heads else ""
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = out_dtype
                cur_res["eps"] = eps
                cur_res["use_global_scale"] = use_global_scale
                cur_res["is_sf_swizzled_layout"] = is_sf_swizzled_layout
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testAddRmsnormFp4quant(args):
    """
    Test add_rmsnorm_fp4quant API from flashinfer.cute_dsl.

    This test:
    1. Generates random input and residual tensors
    2. Runs add_rmsnorm_fp4quant:
       - residual is updated in-place: residual = input + residual
       - RMSNorm is computed on the updated residual
       - Output is FP4 quantized
    3. Runs reference check
    4. Measures performance metrics (memory bandwidth)

    Note: This is a CuTe-DSL kernel requiring SM10.0+ (Blackwell).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testAddRmsnormFp4quant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original

    # Default backend to cute-dsl for FP4 quantization routines
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    eps = args.eps
    out_dtype = args.out_dtype
    use_global_scale = args.use_global_scale
    is_sf_swizzled_layout = args.is_sf_swizzled_layout
    output_both_sf_layouts = args.output_both_sf_layouts
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    # Default to nvfp4 if FP8 dtype is specified (for backwards compatibility)
    if out_dtype in ["fp8_e4m3", "fp8_e5m2"]:
        out_dtype = "nvfp4"

    # Derive block_size from out_dtype
    # nvfp4: block_size=16, e4m3 scale factors
    # mxfp4: block_size=32, ue8m0 scale factors
    if out_dtype == "nvfp4":
        block_size = 16
    elif out_dtype == "mxfp4":
        block_size = 32
    else:
        raise ValueError(
            f"Unsupported out_dtype for FP4 quant: {out_dtype}. Supported: nvfp4, mxfp4."
        )

    # Validate alignment: hidden_size must be divisible by block_size
    if hidden_size % block_size != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by block_size ({block_size}) "
            f"for {out_dtype} quantization."
        )

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )
    ## Done parsing input arguments

    ## Prepare input tensors
    if num_heads is not None:
        input_shape = (batch_size, num_heads, hidden_size)
    else:
        input_shape = (batch_size, hidden_size)

    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    residual_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    # Prepare global_scale if using NVFP4 format
    # Note: API expects a 1D tensor of shape [1], not a 0D scalar
    global_scale = None
    if use_global_scale:
        global_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {residual_tensor.shape = }")
        print(f"[VVERBOSE] {weight.shape = }")
        print(f"[VVERBOSE] {out_dtype = }")
        print(f"[VVERBOSE] {block_size = }")
        print(f"[VVERBOSE] {use_global_scale = }")
        print(f"[VVERBOSE] {is_sf_swizzled_layout = }")
        print(f"[VVERBOSE] {output_both_sf_layouts = }")

    # Warn user that refcheck is not supported for FP4 quantization fusion
    if run_refcheck:
        print("[WARNING] --refcheck is not supported for add_rmsnorm_fp4quant. ")

    def run_backend(backend, input_tensor, residual_tensor, weight):
        if backend == "cute-dsl":
            return flashinfer.add_rmsnorm_fp4quant(
                input_tensor,
                residual_tensor,
                weight,
                eps=eps,
                block_size=block_size,
                global_scale=global_scale,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
                output_both_sf_layouts=output_both_sf_layouts,
                enable_pdl=enable_pdl,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor, residual_tensor.clone(), weight),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for Add + RMSNorm + FP4 Quant
            # Read: input tensor + residual tensor + weight tensor
            # Write: residual tensor (in-place update with input + residual) + FP4 output + scale factors
            num_elements = np.prod(input_shape)
            num_scale_elements = num_elements // block_size
            # FP4: 2 elements per byte (4 bits each)
            fp4_output_bytes = num_elements // 2
            # Scale factors: 1 byte each. When output_both_sf_layouts=True, write 2x scale factors
            sf_write_multiplier = 2 if output_both_sf_layouts else 1
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + num_elements * input_dtype.itemsize  # residual read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements
                * input_dtype.itemsize  # residual write (in-place: input + residual)
                + fp4_output_bytes  # FP4 output write
                + num_scale_elements
                * sf_write_multiplier  # scale factors write (1 byte each)
            )
            problem_flops = num_elements * 6  # rough estimate (add + rmsnorm ops)
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["num_heads"] = num_heads if num_heads else ""
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = out_dtype
                cur_res["eps"] = eps
                cur_res["use_global_scale"] = use_global_scale
                cur_res["is_sf_swizzled_layout"] = is_sf_swizzled_layout
                cur_res["output_both_sf_layouts"] = output_both_sf_layouts
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testFusedRmsnormSilu(args):
    """
    Test fused_rmsnorm_silu API (RMSNorm + SiLU activation).

    This test:
    1. Generates random input tensors
    2. Runs fused_rmsnorm_silu with bf16 output
    3. Optionally runs reference check
    4. Measures performance metrics (memory bandwidth)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedRmsnormSilu")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    eps = args.eps
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"fused_rmsnorm_silu requires bfloat16 input, got {args.input_dtype}"
        )

    input_shape = (batch_size, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device=device) * 1.5 + 0.5
    out = torch.empty(input_shape, dtype=torch.bfloat16, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {weight.shape = }")

    def run_fn(input_tensor, weight, out):
        return flashinfer.fused_rmsnorm_silu(input_tensor, weight, eps=eps, out=out)

    has_reference_output = False
    if run_refcheck:
        rms = torch.sqrt(
            torch.mean(input_tensor.float() ** 2, dim=-1, keepdim=True) + eps
        )
        x_norm = input_tensor.float() / rms * weight.float()
        reference_output = torch.nn.functional.silu(x_norm).to(torch.bfloat16)
        has_reference_output = True

    if run_refcheck:
        test_out = run_fn(input_tensor, weight, out)
        if has_reference_output:
            (
                num_different_elements,
                num_elements,
                num_different_elements_percentage,
            ) = is_close_stats(reference_output, test_out, rtol=2e-2, atol=2e-2)
            if num_different_elements > 0:
                print(
                    f"[ERROR] Output tensor mismatch: "
                    f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Output mismatch with {num_different_elements} elements"
                    )

    times = bench_gpu_time(
        fn=run_fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        input_args=(input_tensor, weight, out),
    )

    if len(times) > 0:
        median_time = np.median(times)
        std_time = np.std(times)

        num_elements = np.prod(input_shape)
        problem_bytes = (
            num_elements * input_dtype.itemsize  # input read
            + hidden_size * input_dtype.itemsize  # weight read
            + num_elements * input_dtype.itemsize  # output write
        )
        problem_flops = num_elements * 7  # rmsnorm (5) + silu (2: exp + div)
        tflops = problem_flops / (10**9 * median_time)
        tb_per_sec = problem_bytes / (10**9 * median_time)

        print_perf_metrics("cuda", median_time, std_time, tflops, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["input_dtype"] = str(input_dtype)
            cur_res["eps"] = eps
            cur_res["backend"] = "cuda"
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testFusedDitLayernorm(args):
    """
    Test fused DIT LayerNorm API (3 modes).

    Benchmarks the fused kernel against eager PyTorch for the specified mode.
    Hidden dim is fixed at 3072 (WAN 2.2 5B target).
    """
    from flashinfer.diffusion_ops import (
        fused_dit_gate_residual_layernorm_gamma_beta,
        fused_dit_gate_residual_layernorm_scale_shift,
        fused_dit_residual_layernorm_scale_shift,
    )

    device = get_device(args)
    # For DIT LayerNorm, hidden_size is always 3072 (WAN 2.2 5B).
    # We reuse --hidden_size as num_rows (sequence length) for CLI compatibility.
    num_rows = args.hidden_size
    batch_size = args.batch_size
    hidden_dim = 3072
    eps = args.eps
    mode = args.dit_mode

    torch.manual_seed(42)
    input_t = torch.randn(
        batch_size, num_rows, hidden_dim, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_t)
    gamma = torch.randn(hidden_dim, dtype=torch.float32, device=device)
    beta = torch.randn(hidden_dim, dtype=torch.float32, device=device)

    # Create WAN-style strided gate/scale/shift tensors
    scale_shift_table = torch.randn(
        1, 6, hidden_dim, dtype=torch.float32, device=device
    )
    temb = torch.randn(
        batch_size, num_rows, 6, hidden_dim, dtype=torch.bfloat16, device=device
    )
    temb_chunks = temb.chunk(6, dim=2)
    table_chunks = scale_shift_table.chunk(6, dim=1)

    gate = temb_chunks[2].squeeze(2)
    gate_bias = table_chunks[2].squeeze(1)
    scale = temb_chunks[1].squeeze(2)
    scale_bias = table_chunks[1].squeeze(1)
    shift = temb_chunks[0].squeeze(2)
    shift_bias = table_chunks[0].squeeze(1)
    c_gate = temb_chunks[5].squeeze(2)
    c_gate_bias = table_chunks[5].squeeze(1)
    c_scale = temb_chunks[4].squeeze(2)
    c_scale_bias = table_chunks[4].squeeze(1)
    c_shift = temb_chunks[3].squeeze(2)
    c_shift_bias = table_chunks[3].squeeze(1)

    if mode == "gate_residual_gamma_beta":

        def fused_fn():
            return fused_dit_gate_residual_layernorm_gamma_beta(
                input_t,
                residual,
                gate,
                gamma,
                beta,
                gate_bias=gate_bias,
                epsilon=eps,
            )

        def eager_fn():
            r = residual.float() + input_t.float() * (gate.float() + gate_bias.float())
            n = torch.layer_norm(r, [hidden_dim], weight=gamma, bias=beta, eps=eps)
            return r.to(torch.bfloat16), n.to(torch.bfloat16)

    elif mode == "gate_residual_scale_shift":

        def fused_fn():
            return fused_dit_gate_residual_layernorm_scale_shift(
                input_t,
                residual,
                c_gate,
                scale,
                shift,
                gate_bias=c_gate_bias,
                scale_bias=scale_bias,
                shift_bias=shift_bias,
                epsilon=eps,
            )

        def eager_fn():
            r = residual.float() + input_t.float() * (
                c_gate.float() + c_gate_bias.float()
            )
            n = torch.layer_norm(r, [hidden_dim], eps=eps)
            n = n * (1 + scale.float() + scale_bias.float()) + (
                shift.float() + shift_bias.float()
            )
            return r.to(torch.bfloat16), n.to(torch.bfloat16)

    elif mode == "residual_scale_shift":

        def fused_fn():
            return fused_dit_residual_layernorm_scale_shift(
                input_t,
                c_scale,
                c_shift,
                residual=residual,
                scale_bias=c_scale_bias,
                shift_bias=c_shift_bias,
                epsilon=eps,
            )

        def eager_fn():
            r = residual.float() + input_t.float()
            n = torch.layer_norm(r, [hidden_dim], eps=eps)
            n = n * (1 + c_scale.float() + c_scale_bias.float()) + (
                c_shift.float() + c_shift_bias.float()
            )
            return r.to(torch.bfloat16), n.to(torch.bfloat16)

    else:
        raise ValueError(f"Unknown DIT mode: {mode}")

    # Warmup + benchmark
    fused_times = bench_gpu_time(
        fused_fn, enable_cupti=True, dry_run_iters=10, repeat_iters=100
    )
    eager_times = bench_gpu_time(
        eager_fn, enable_cupti=True, dry_run_iters=10, repeat_iters=100
    )

    fused_ms = float(np.median(fused_times))
    eager_ms = float(np.median(eager_times))

    # Reference check
    if args.refcheck:
        r_fused, n_fused = fused_fn()
        r_eager, n_eager = eager_fn()
        torch.testing.assert_close(
            r_fused.float(), r_eager.float(), rtol=1.6e-2, atol=1e-5
        )
        torch.testing.assert_close(
            n_fused.float(), n_eager.float(), rtol=1.6e-2, atol=1e-5
        )

    total_bytes = batch_size * num_rows * hidden_dim * 2 * 4
    tb_per_sec = (total_bytes / 1e12) / (fused_ms / 1e3) if fused_ms > 0 else 0

    res = []
    cur_res = defaultdict(str)
    cur_res["routine"] = args.routine
    cur_res["median_ms"] = fused_ms
    cur_res["std_ms"] = float(np.std(fused_times))
    cur_res["batch_size"] = batch_size
    cur_res["hidden_size"] = hidden_dim
    cur_res["input_dtype"] = str(torch.bfloat16)
    cur_res["dit_mode"] = mode
    cur_res["eps"] = eps
    cur_res["backend"] = "cuda"
    cur_res["case_tag"] = args.case_tag
    print_perf_metrics("fused", fused_ms, float(np.std(fused_times)), 0.0, tb_per_sec)

    if args.verbose >= 1:
        speedup = eager_ms / fused_ms if fused_ms > 0 else 0
        print(
            f"  Eager: {eager_ms:.4f} ms | Fused: {fused_ms:.4f} ms | Speedup: {speedup:.2f}x"
        )

    res.append(cur_res)


def testFusedQkRmsnormRope(args):
    """
    Test fused QK RMSNorm + 3D RoPE + V copy API.

    Benchmarks the fused kernel for video generation DIT self-attention
    (e.g. WAN 2.1/2.2). Compares against eager PyTorch (separate RMSNorm + RoPE).

    Args:
        args: Parsed command line arguments

    Returns:
        list: List of dicts containing performance results
    """
    from flashinfer.diffusion_ops import fused_qk_rmsnorm_rope

    if args.verbose >= 1:
        print("[INFO] Running testFusedQkRmsnormRope")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)

    cc = torch.cuda.get_device_capability(device)
    cc_int = cc[0] * 10 + cc[1]
    if not fused_qk_rmsnorm_rope.is_compute_capability_supported(cc_int):
        print(f"[SKIP] fused_qk_rmsnorm_rope not supported on SM{cc_int}")
        return []

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    eps = args.eps
    ppf = args.ppf
    pph = args.pph
    ppw = args.ppw
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    if num_heads is None:
        raise ValueError("--num_heads is required for fused_qk_rmsnorm_rope")

    head_dim = hidden_size // num_heads
    seq_len = ppf * pph * ppw

    h_dim = w_dim = 2 * (head_dim // 6)
    t_dim = head_dim - h_dim - w_dim

    input_dtype = torch.bfloat16

    torch.manual_seed(42)
    qkv = torch.randn(
        batch_size, seq_len, 3 * hidden_size, dtype=input_dtype, device=device
    )
    q_weight = torch.randn(hidden_size, dtype=input_dtype, device=device)
    k_weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] qkv.shape = {qkv.shape}")
        print(f"[VVERBOSE] seq_len = {seq_len} (ppf={ppf}, pph={pph}, ppw={ppw})")
        print(f"[VVERBOSE] head_dim = {head_dim}, t/h/w = {t_dim}/{h_dim}/{w_dim}")

    kwargs = dict(
        ppf=ppf,
        pph=pph,
        ppw=ppw,
        num_frame_channels=t_dim,
        num_height_channels=h_dim,
        num_width_channels=w_dim,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        num_heads_v=num_heads,
        head_dim=head_dim,
        eps=eps,
        base=10000.0,
        interleave=True,
        is_qk_norm=True,
    )

    def run_fused():
        return fused_qk_rmsnorm_rope(qkv, q_weight, k_weight, **kwargs)

    # Reference check
    if run_refcheck:
        import torch.nn as nn

        query = qkv[..., :hidden_size]
        key = qkv[..., hidden_size : 2 * hidden_size]

        norm_q = nn.RMSNorm(hidden_size, eps=eps).to(device).to(input_dtype)
        norm_k = nn.RMSNorm(hidden_size, eps=eps).to(device).to(input_dtype)
        with torch.no_grad():
            norm_q.weight.copy_(q_weight)
            norm_k.weight.copy_(k_weight)

        q_ref = norm_q(query).unflatten(2, (num_heads, head_dim))
        k_ref = norm_k(key).unflatten(2, (num_heads, head_dim))

        q_fused, k_fused, _ = run_fused()

        # Compare after norm, before RoPE (RoPE adds position-dependent rotation)
        # For a rough check, compare magnitudes
        q_diff = (q_fused.flatten(2).float() - q_ref.flatten(2).float()).abs().max()
        k_diff = (k_fused.flatten(2).float() - k_ref.flatten(2).float()).abs().max()
        if args.verbose >= 1:
            print(
                f"[INFO] Refcheck: Q max diff = {q_diff:.4f}, K max diff = {k_diff:.4f}"
            )
            print("[INFO] (Note: diff includes RoPE rotation, so nonzero is expected)")

    backend_times = bench_gpu_time(
        fn=run_fused,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
    )

    if len(backend_times) > 0:
        median_time = np.median(backend_times)
        std_time = np.std(backend_times)

        # Memory bandwidth: read QKV + Q/K weights, write Q + K + V
        num_tokens = batch_size * seq_len
        problem_bytes = (
            num_tokens * 3 * hidden_size * input_dtype.itemsize  # QKV read
            + 2 * hidden_size * input_dtype.itemsize  # Q/K weight read
            + num_tokens * 3 * hidden_size * input_dtype.itemsize  # Q+K+V write
        )
        problem_flops = num_tokens * hidden_size * 10  # rough estimate
        tflops = problem_flops / (10**9 * median_time)
        tb_per_sec = problem_bytes / (10**9 * median_time)

        print_perf_metrics("cuda", median_time, std_time, tflops, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["num_heads"] = num_heads
            cur_res["input_dtype"] = str(input_dtype)
            cur_res["eps"] = eps
            cur_res["backend"] = "cuda"
            cur_res["ppf"] = ppf
            cur_res["pph"] = pph
            cur_res["ppw"] = ppw
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res
