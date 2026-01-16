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


def run_quantization_test(args):
    """
    Run a quantization test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "mxfp8_quantize":
        return testMxfp8Quantize(args)
    elif args.routine == "mxfp4_quantize":
        return testMxfp4Quantize(args)
    elif args.routine == "nvfp4_quantize":
        return testNvfp4Quantize(args)
    elif args.routine == "nvfp4_batched_quantize":
        return testNvfp4BatchedQuantize(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_quantization_args(line, parser):
    """
    Parse command line arguments for quantization test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Number of rows in input tensor.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of columns in input tensor (must be divisible by 32).",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type of the input tensor.",
    )
    parser.add_argument(
        "--is_sf_swizzled_layout",
        dest="is_sf_swizzled_layout",
        action="store_true",
        help="Use swizzled layout for scale factors. (default)",
    )
    parser.add_argument(
        "--no_sf_swizzled_layout",
        dest="is_sf_swizzled_layout",
        action="store_false",
        help="Disable swizzled layout for scale factors.",
    )
    parser.set_defaults(is_sf_swizzled_layout=True)
    parser.add_argument(
        "--alignment",
        type=int,
        required=False,
        default=32,
        help="sfVecSize for quantization. Default: 32",
    )
    parser.add_argument(
        "--enable_pdl",
        action="store_true",
        default=False,
        help="Enable programmatic dependent launch.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda"],
        help="Backend to test. Default: cuda",
    )
    # FP4 quantization specific arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=None,
        help="Batch size for batched quantization (nvfp4_batched_quantize).",
    )
    parser.add_argument(
        "--global_scale",
        type=float,
        required=False,
        default=1.0,
        help="Global scale factor for NVFP4 quantization. Default: 1.0",
    )
    parser.add_argument(
        "--sf_layout",
        type=str,
        required=False,
        default="128x4",
        choices=["128x4", "8x4", "linear"],
        help="Scale factor layout for NVFP4 quantization. Default: 128x4",
    )
    parser.add_argument(
        "--do_shuffle",
        action="store_true",
        default=False,
        help="Shuffle scale factors for TRTLLM backend (nvfp4_quantize only).",
    )
    parser.add_argument(
        "--sf_vec_size",
        type=int,
        required=False,
        default=16,
        help="Scale factor vector size for NVFP4 quantization. Default: 16",
    )

    args = parser.parse_args(line)

    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testMxfp8Quantize(args):
    """
    Test mxfp8_quantize API.

    This test:
    1. Generates random input tensors
    2. Runs mxfp8_quantize
    3. Runs reference check (via dequantize round-trip)
    4. Measures performance metrics (memory bandwidth)

    Note: Quantization is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMxfp8Quantize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    m = args.m
    k = args.k
    is_sf_swizzled_layout = args.is_sf_swizzled_layout
    alignment = args.alignment
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    # Validate k is divisible by alignment (sf_vec_size)
    sf_vec_size = alignment
    if k % sf_vec_size != 0:
        raise ValueError(f"k ({k}) must be divisible by {sf_vec_size} (sf_vec_size)")

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
    input_shape = (m, k)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {is_sf_swizzled_layout = }")
        print(f"[VVERBOSE] {alignment = }")
        print(f"[VVERBOSE] {enable_pdl = }")

    def run_backend(backend, input_tensor):
        if backend == "cuda":
            return flashinfer.mxfp8_quantize(
                input_tensor,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
                alignment=alignment,
                enable_pdl=enable_pdl,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference check via dequantize round-trip
    has_reference_output = False
    if run_refcheck:
        # For mxfp8, we verify by dequantizing and comparing
        # This tests the quantize->dequantize round-trip
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            x_q, sf = run_backend(cur_backend, input_tensor)
            outputs[cur_backend] = (x_q.detach().clone(), sf.detach().clone())
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                x_q, sf = outputs[tested_backends[i]]
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {tested_backends[i]}: "
                        f"x_q.shape = {x_q.shape}, x_q.dtype = {x_q.dtype}, "
                        f"sf.shape = {sf.shape}, sf.dtype = {sf.dtype}"
                    )
                # Dequantize and compare with original
                # Note: mxfp8_dequantize_host is a HOST function, so tensors must be on CPU
                # and expects uint8 dtype
                try:
                    x_q_cpu = x_q.cpu().view(torch.uint8)
                    sf_cpu = sf.cpu().view(torch.uint8).reshape(-1)
                    dequantized = flashinfer.mxfp8_dequantize_host(
                        x_q_cpu, sf_cpu, is_sf_swizzled_layout=is_sf_swizzled_layout
                    )
                    # Move back to GPU for comparison
                    dequantized = dequantized.to(input_tensor.device)
                    # Compare with original input (allowing for quantization error)
                    (
                        num_different_elements,
                        num_elements,
                        num_different_elements_percentage,
                    ) = is_close_stats(
                        input_tensor.float(), dequantized, rtol=0.5, atol=0.5
                    )
                    if args.verbose >= 2:
                        print(
                            f"[VVERBOSE] Round-trip error: {num_different_elements}/{num_elements} "
                            f"({num_different_elements_percentage:.2f}%) elements differ"
                        )
                    # Enforce refcheck: fail or warn on mismatches
                    if num_different_elements > 0:
                        mismatch_msg = (
                            f"[mxfp8_quantize] Round-trip mismatch: "
                            f"{num_different_elements}/{num_elements} "
                            f"({num_different_elements_percentage:.2f}%) elements differ"
                        )
                        if args.allow_output_mismatch:
                            print(f"[WARNING] {mismatch_msg}")
                        else:
                            raise AssertionError(mismatch_msg)
                except Exception as e:
                    if args.verbose >= 1:
                        print(
                            f"[WARNING] [mxfp8_quantize] Dequantize check failed: {e}"
                        )
                    if not args.allow_output_mismatch:
                        raise

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for mxfp8_quantize
            # Read: input tensor
            # Write: quantized tensor (fp8) + scale factors
            num_elements = m * k
            num_scale_factors = num_elements // sf_vec_size
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + num_elements * 1  # quantized output write (fp8 = 1 byte)
                + num_scale_factors * 1  # scale factors write (1 byte each)
            )
            # Quantization is memory-bound, TFLOPS not primary metric
            problem_flops = num_elements * 3  # rough estimate (scale, clamp, convert)
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
                cur_res["m"] = m
                cur_res["k"] = k
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["is_sf_swizzled_layout"] = is_sf_swizzled_layout
                cur_res["alignment"] = alignment
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMxfp4Quantize(args):
    """
    Test mxfp4_quantize API.

    This test:
    1. Generates random input tensors
    2. Runs mxfp4_quantize
    3. Runs reference check (via dequantize round-trip)
    4. Measures performance metrics (memory bandwidth)

    Note: Quantization is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMxfp4Quantize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    m = args.m
    k = args.k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    # mxfp4 uses sf_vec_size=32 (hardcoded in the API)
    sf_vec_size = 32
    if k % sf_vec_size != 0:
        raise ValueError(
            f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size}) for mxfp4_quantize"
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
    input_shape = (m, k)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")

    def run_backend(backend, input_tensor):
        if backend == "cuda":
            return flashinfer.mxfp4_quantize(input_tensor)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference check via dequantize round-trip
    has_reference_output = False
    if run_refcheck:
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            x_q, sf = run_backend(cur_backend, input_tensor)
            outputs[cur_backend] = (x_q.detach().clone(), sf.detach().clone())
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                x_q, sf = outputs[tested_backends[i]]
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {tested_backends[i]}: "
                        f"x_q.shape = {x_q.shape}, x_q.dtype = {x_q.dtype}, "
                        f"sf.shape = {sf.shape}, sf.dtype = {sf.dtype}"
                    )
                # Dequantize and compare with original
                try:
                    dequantized = flashinfer.mxfp4_dequantize(x_q, sf)
                    # Move back to GPU for comparison
                    dequantized = dequantized.to(input_tensor.device)
                    # Compare with original input (allowing for quantization error)
                    (
                        num_different_elements,
                        num_elements,
                        num_different_elements_percentage,
                    ) = is_close_stats(
                        input_tensor.float(), dequantized, rtol=0.5, atol=0.5
                    )
                    if args.verbose >= 2:
                        print(
                            f"[VVERBOSE] Round-trip error: {num_different_elements}/{num_elements} "
                            f"({num_different_elements_percentage:.2f}%) elements differ"
                        )
                    # Enforce refcheck: fail or warn on mismatches
                    if num_different_elements > 0:
                        mismatch_msg = (
                            f"[mxfp4_quantize] Round-trip mismatch: "
                            f"{num_different_elements}/{num_elements} "
                            f"({num_different_elements_percentage:.2f}%) elements differ"
                        )
                        if args.allow_output_mismatch:
                            print(f"[WARNING] {mismatch_msg}")
                        else:
                            raise AssertionError(mismatch_msg)
                except Exception as e:
                    if args.verbose >= 1:
                        print(
                            f"[WARNING] [mxfp4_quantize] Dequantize check failed: {e}"
                        )
                    if not args.allow_output_mismatch:
                        raise

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for mxfp4_quantize
            # Read: input tensor
            # Write: quantized tensor (fp4 = 0.5 bytes per element) + scale factors
            num_elements = m * k
            sf_vec_size = 32  # mxfp4 uses sf_vec_size=32
            num_scale_factors = num_elements // sf_vec_size
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + num_elements // 2  # quantized output write (fp4 = 0.5 byte)
                + num_scale_factors * 1  # scale factors write (1 byte each, ue8m0)
            )
            problem_flops = num_elements * 3  # rough estimate
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["k"] = k
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testNvfp4Quantize(args):
    """
    Test nvfp4_quantize API.

    This test:
    1. Generates random input tensors
    2. Runs nvfp4_quantize with specified layout
    3. Verifies output shapes
    4. Measures performance metrics (memory bandwidth)

    Note: Quantization is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    from flashinfer.fp4_quantization import SfLayout

    if args.verbose >= 1:
        print("[INFO] Running testNvfp4Quantize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    m = args.m
    k = args.k
    global_scale = args.global_scale
    sf_layout_str = args.sf_layout
    do_shuffle = args.do_shuffle
    sf_vec_size = args.sf_vec_size
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    # do_shuffle involves CPU index generation which is not CUDA graph compatible
    if do_shuffle and is_cuda_graph_compatible:
        print(
            "[WARNING] do_shuffle=True is not CUDA graph compatible. Disabling CUDA graph."
        )
        is_cuda_graph_compatible = False

    # Convert sf_layout string to enum
    sf_layout_map = {
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
        "linear": SfLayout.layout_linear,
    }
    sf_layout = sf_layout_map[sf_layout_str]

    # Validate k is divisible by sf_vec_size
    if k % sf_vec_size != 0:
        raise ValueError(
            f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size}) for nvfp4_quantize"
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
    input_shape = (m, k)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    global_sf_tensor = torch.tensor([global_scale], dtype=torch.float32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {global_scale = }")
        print(f"[VVERBOSE] {sf_layout_str = }")
        print(f"[VVERBOSE] {do_shuffle = }")
        print(f"[VVERBOSE] {sf_vec_size = }")
        print(f"[VVERBOSE] {enable_pdl = }")

    def run_backend(backend, input_tensor, global_sf_tensor):
        if backend == "cuda":
            return flashinfer.nvfp4_quantize(
                input_tensor,
                global_sf_tensor,
                sfLayout=sf_layout,
                do_shuffle=do_shuffle,
                sf_vec_size=sf_vec_size,
                enable_pdl=enable_pdl,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            x_q, sf = run_backend(cur_backend, input_tensor, global_sf_tensor)
            outputs[cur_backend] = (x_q.detach().clone(), sf.detach().clone())
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor, global_sf_tensor),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0:
        if run_refcheck:
            for i in range(len(tested_backends)):
                x_q, sf = outputs[tested_backends[i]]
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {tested_backends[i]}: "
                        f"x_q.shape = {x_q.shape}, x_q.dtype = {x_q.dtype}, "
                        f"sf.shape = {sf.shape}, sf.dtype = {sf.dtype}"
                    )
                # Verify output shape (M, K/2) for FP4
                expected_shape = (m, k // 2)
                if x_q.shape != expected_shape:
                    print(
                        f"[WARNING] Unexpected output shape: {x_q.shape}, expected {expected_shape}"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for nvfp4_quantize
            # Read: input tensor + global_sf
            # Write: quantized tensor (fp4 = 0.5 bytes per element) + scale factors
            num_elements = m * k
            num_scale_factors = num_elements // sf_vec_size
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + 4  # global_sf read (float32)
                + num_elements // 2  # quantized output write (fp4 = 0.5 byte)
                + num_scale_factors * 1  # scale factors write (1 byte each, e4m3)
            )
            problem_flops = num_elements * 3  # rough estimate
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["k"] = k
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["global_scale"] = global_scale
                cur_res["sf_layout"] = sf_layout_str
                cur_res["do_shuffle"] = do_shuffle
                cur_res["sf_vec_size"] = sf_vec_size
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testNvfp4BatchedQuantize(args):
    """
    Test nvfp4_batched_quantize API.

    This test:
    1. Generates random batched input tensors
    2. Runs nvfp4_batched_quantize
    3. Verifies output shapes
    4. Measures performance metrics (memory bandwidth)

    Note: Quantization is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testNvfp4BatchedQuantize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original
    batch_size = args.batch_size
    m = args.m
    k = args.k
    global_scale = args.global_scale
    sf_vec_size = args.sf_vec_size
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    if batch_size is None:
        raise ValueError("--batch_size is required for nvfp4_batched_quantize")

    # Validate k is divisible by sf_vec_size
    if k % sf_vec_size != 0:
        raise ValueError(
            f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size}) for nvfp4_batched_quantize"
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
    input_shape = (batch_size, m, k)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    global_sf_tensor = torch.tensor([global_scale], dtype=torch.float32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {global_scale = }")
        print(f"[VVERBOSE] {sf_vec_size = }")

    def run_backend(backend, input_tensor, global_sf_tensor):
        if backend == "cuda":
            return flashinfer.nvfp4_batched_quantize(
                input_tensor,
                global_sf_tensor,
                sf_vec_size=sf_vec_size,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            x_q, sf = run_backend(cur_backend, input_tensor, global_sf_tensor)
            outputs[cur_backend] = (x_q.detach().clone(), sf.detach().clone())
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor, global_sf_tensor),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0:
        if run_refcheck:
            for i in range(len(tested_backends)):
                x_q, sf = outputs[tested_backends[i]]
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {tested_backends[i]}: "
                        f"x_q.shape = {x_q.shape}, x_q.dtype = {x_q.dtype}, "
                        f"sf.shape = {sf.shape}, sf.dtype = {sf.dtype}"
                    )
                # Verify output shape (B, M, K/2) for FP4
                expected_shape = (batch_size, m, k // 2)
                if x_q.shape != expected_shape:
                    print(
                        f"[WARNING] Unexpected output shape: {x_q.shape}, expected {expected_shape}"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for nvfp4_batched_quantize
            # Read: input tensor + global_sf
            # Write: quantized tensor (fp4 = 0.5 bytes per element) + scale factors
            num_elements = batch_size * m * k
            num_scale_factors = num_elements // sf_vec_size
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + 4  # global_sf read (float32)
                + num_elements // 2  # quantized output write (fp4 = 0.5 byte)
                + num_scale_factors * 1  # scale factors write (1 byte each)
            )
            problem_flops = num_elements * 3  # rough estimate
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)

            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["batch_size"] = batch_size
                cur_res["m"] = m
                cur_res["k"] = k
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["global_scale"] = global_scale
                cur_res["sf_vec_size"] = sf_vec_size
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
