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
import os

import numpy as np
import torch

import flashinfer
from flashinfer.quantization.fp4_quantization import NVFP4_QUANT_ENV_VARS
from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
    current_nvfp4_4over6_config,
    env_flag_enabled as _env_flag_enabled,
    make_nvfp4_global_scale,
    nvfp4_e4m3_max,
)
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    blank_unused_mode_columns,
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    is_close_stats,
    filter_backends_by_compute_capability,
    fp8_code_distance,
    warn_if_pdl_unsupported,
)


def _assert_nvfp4_refcheck_tensor_equal(
    name: str,
    backend: str,
    ref_backend: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    min_match_pct: float = 100.0,
):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"[nvfp4_quantize] {name} shape mismatch: "
            f"{backend}={actual.shape}, {ref_backend}={expected.shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"[nvfp4_quantize] {name} dtype mismatch: "
            f"{backend}={actual.dtype}, {ref_backend}={expected.dtype}"
        )
    if not torch.equal(actual, expected):
        mismatch_pct = (actual != expected).float().mean().item() * 100
        if 100.0 - mismatch_pct >= min_match_pct and min_match_pct < 100.0:
            return
        raise AssertionError(
            f"[nvfp4_quantize] {name} value mismatch: "
            f"{backend} vs {ref_backend}, mismatch={mismatch_pct:.4f}% "
            f"(required match {min_match_pct:.1f}%)"
        )


def _nvfp4_4over6_from_args(args):
    """Map --nvfp4_4over6 to (nvfp4_4over6= kwargs, resolved recipe).

    ``env`` leaves the parameter unset (the API then reads the legacy
    FLASHINFER_NVFP4_4OVER6* variables, the historical harness behavior).
    """
    if args.nvfp4_4over6 == "env":
        return {}, current_nvfp4_4over6_config()
    if args.nvfp4_4over6 == "off":
        return {"nvfp4_4over6": None}, None
    config = NVFP44Over6Config(
        e4m3_max=args.nvfp4_4over6_e4m3_max, err_mode=args.nvfp4_4over6.upper()
    )
    return {"nvfp4_4over6": config}, config


def _check_mismatch(args, routine, what, num_bad, num_total):
    """Fail (or warn with --allow_output_mismatch) on a refcheck mismatch."""
    if num_bad == 0:
        return
    msg = (
        f"[{routine}] {what}: {num_bad}/{num_total} "
        f"({100.0 * num_bad / max(num_total, 1):.4f}%) elements differ"
    )
    if args.allow_output_mismatch:
        print(f"[WARNING] {msg}")
    else:
        raise AssertionError(msg)


def _bench_backends(args, backends, run_backend, input_args, use_cuda_graph):
    """Time every backend with the harness-wide bench_gpu_time settings."""
    return {
        backend: bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=use_cuda_graph,
            input_args=(backend, *input_args),
        )
        for backend in backends
    }


def _report_results(args, backend_times, problem_bytes, problem_flops, columns):
    """Print [PERF] lines and build CSV rows (columns = routine-specific fields)."""
    res = []
    for backend, times in backend_times.items():
        if len(times) == 0:
            continue
        median_time = np.median(times)
        std_time = np.std(times)
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
            cur_res.update(columns)
            cur_res["backend"] = backend
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


# Masked grouped ([B, M, K] + per-group valid-row mask) routines
_GROUPED_ROUTINES = (
    "mxfp8_grouped_quantize",
    "scaled_fp4_grouped_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "silu_and_mul_mxfp8_experts_quantize",
)


def run_quantization_test(args):
    """Run one quantization testlist line; see _dispatch_quantization_test for the routine table."""
    res = _dispatch_quantization_test(args)
    # Mode flags added in 2026-10 stay blank on rows that do not use them, so
    # pre-existing lines keep their CSV columns (and dashboard keys).
    return blank_unused_mode_columns(
        args,
        res,
        defaults={
            "per_token_activation": {"False"},
            "nvfp4_4over6": {"env"},
            "use_4over6": {"False"},
            "disable_quant_fast_math": {"False"},
            "nvfp4_4over6_e4m3_max": {"448"},
        },
        owners={
            "masked_m": _GROUPED_ROUTINES,
            "quant_dtype": ("per_token_group_quant_8bit",),
            "group_size": ("per_token_group_quant_8bit",),
            "column_major_scales": ("per_token_group_quant_8bit",),
            "scale_tma_aligned": ("per_token_group_quant_8bit",),
            "scale_ue8m0": ("per_token_group_quant_8bit",),
            "num_kv_heads": ("nvfp4_kv_dequantize_paged",),
            "page_size": ("nvfp4_kv_dequantize_paged",),
            "kv_layout": ("nvfp4_kv_dequantize_paged",),
            "bitorder": ("packbits", "segment_packbits"),
        },
    )


def _dispatch_quantization_test(args):
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
    elif args.routine in _GROUPED_ROUTINES:
        return testGroupedQuantize(args)
    elif args.routine == "per_token_group_quant_8bit":
        return testPerTokenGroupQuant8bit(args)
    elif args.routine == "silu_and_mul_nvfp4_quantize":
        return testSiluAndMulNvfp4Quantize(args)
    elif args.routine in ("nvfp4_kv_quantize", "nvfp4_kv_dequantize"):
        return testNvfp4KvQuantize(args)
    elif args.routine == "nvfp4_kv_dequantize_paged":
        return testNvfp4KvDequantizePaged(args)
    elif args.routine in ("packbits", "segment_packbits"):
        return testPackbits(args)
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
        help="Number of rows in input tensor. nvfp4_kv_dequantize_paged: tokens "
        "per request. packbits/segment_packbits: number of segments.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of columns in input tensor (must be divisible by 32). "
        "silu_and_mul_*: output width (the input is [M, 2K]). "
        "nvfp4_kv_dequantize_paged: head_dim. packbits/segment_packbits: bits "
        "per segment.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "fp8_e4m3"],
        help="Data type of the input tensor. float32 is supported by "
        "mxfp8_quantize with the cute-dsl backend only; fp8_e4m3 (FP8->FP4 "
        "path) by nvfp4_quantize only.",
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
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda", "cute-dsl", "cutile"],
        help="Backend to test. Default: cuda. cutile: mxfp8_grouped_quantize and "
        "per_token_group_quant_8bit.",
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
        help="Scale factor layout for FP4 quantization. Default: 128x4",
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
    parser.add_argument(
        "--per_token_activation",
        action="store_true",
        default=False,
        help="Benchmark NVFP4 per-token activation quantization.",
    )
    parser.add_argument(
        "--nvfp4_4over6",
        type=str,
        required=False,
        default="env",
        choices=["env", "off", "mae", "mse"],
        help="NVFP4 4over6 recipe passed as nvfp4_4over6=. env (default) leaves "
        "the parameter unset so the FLASHINFER_NVFP4_4OVER6* variables apply; "
        "off passes None; mae/mse pass NVFP44Over6Config(err_mode=...).",
    )
    parser.add_argument(
        "--nvfp4_4over6_e4m3_max",
        type=int,
        required=False,
        default=448,
        choices=[448, 256],
        help="E4M3 block-scale ceiling of the mae/mse 4over6 recipe. Default: 448",
    )
    # Grouped ([B, M, K] + mask) routines
    parser.add_argument(
        "--masked_m",
        type=int,
        required=False,
        default=None,
        help="Valid rows per group for the masked grouped routines. Default: --m.",
    )
    # per_token_group_quant_8bit
    parser.add_argument(
        "--group_size",
        type=int,
        required=False,
        default=128,
        help="Elements per quantization group (per_token_group_quant_8bit). Default: 128",
    )
    parser.add_argument(
        "--quant_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        choices=["fp8_e4m3", "fp8_e5m2", "int8"],
        help="Output dtype of per_token_group_quant_8bit. Default: fp8_e4m3",
    )
    parser.add_argument(
        "--column_major_scales",
        action="store_true",
        default=False,
        help="per_token_group_quant_8bit: column-major scale output.",
    )
    parser.add_argument(
        "--scale_tma_aligned",
        action="store_true",
        default=False,
        help="per_token_group_quant_8bit: 16B-aligned (TMA) scale rows. "
        "Requires --column_major_scales.",
    )
    parser.add_argument(
        "--scale_ue8m0",
        action="store_true",
        default=False,
        help="per_token_group_quant_8bit: power-of-two (UE8M0) scales. "
        "Requires --column_major_scales.",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        required=False,
        default=8,
        help="KV heads for nvfp4_kv_dequantize_paged. Default: 8",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        required=False,
        default=16,
        help="Page size for nvfp4_kv_dequantize_paged. Default: 16",
    )
    parser.add_argument(
        "--kv_layout",
        type=str,
        required=False,
        default="NHD",
        choices=["NHD", "HND"],
        help="Paged cache layout for nvfp4_kv_dequantize_paged. Default: NHD",
    )
    # packbits / segment_packbits
    parser.add_argument(
        "--bitorder",
        type=str,
        required=False,
        default="big",
        choices=["big", "little"],
        help="Bit order for packbits/segment_packbits. Default: big",
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

    from flashinfer import SfLayout

    # --no_sf_swizzled_layout selects the linear layout; otherwise --sf_layout
    # (default 128x4) picks the swizzled variant.
    sf_layout_str = args.sf_layout if is_sf_swizzled_layout else "linear"
    sf_layout = {
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
        "linear": SfLayout.layout_linear,
    }[sf_layout_str]

    # MXFP8 always uses 32-element scale blocks (the API asserts K % 32);
    # --alignment only pads the quantized K up to a multiple of itself.
    sf_vec_size = 32
    if k % sf_vec_size != 0:
        raise ValueError(f"k ({k}) must be divisible by {sf_vec_size} (sf_vec_size)")
    padded_k = (k + alignment - 1) // alignment * alignment

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. "
            "Supported dtypes are bfloat16, float16, float32."
        )
    if input_dtype == torch.float32 and "cuda" in backends:
        print("[INFO] cuda backend does not support float32 input for mxfp8_quantize")
        backends = [b for b in backends if b != "cuda"]
        if len(backends) == 0:
            print("[ERROR] No backends to test. Exiting.")
            return res
    ## Done parsing input arguments

    ## Prepare input tensors
    input_shape = (m, k)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {sf_layout_str = }")
        print(f"[VVERBOSE] {alignment = }")
        print(f"[VVERBOSE] {enable_pdl = }")

    def run_backend(backend, input_tensor):
        return flashinfer.mxfp8_quantize(
            input_tensor,
            alignment=alignment,
            enable_pdl=enable_pdl,
            backend=backend,
            sf_swizzle_layout=sf_layout,
        )

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
    if run_refcheck and sf_layout == SfLayout.layout_8x4:
        # mxfp8_dequantize_host implements only the 128x4 and linear layouts
        # (fp8Quantize.h "Other layout not implemented yet"): compare backends.
        has_reference_output = False
        ref_x_q, ref_sf = outputs[tested_backends[0]]
        for cur_backend in tested_backends[1:]:
            cur_x_q, cur_sf = outputs[cur_backend]
            dist = fp8_code_distance(cur_x_q, ref_x_q)
            _check_mismatch(
                args,
                args.routine,
                f"8x4 x_q {cur_backend} vs {tested_backends[0]} (> 1 fp8 step)",
                int((dist > 1).sum().item()),
                dist.numel(),
            )
            _check_mismatch(
                args,
                args.routine,
                f"8x4 sf {cur_backend} vs {tested_backends[0]}",
                int(
                    (cur_sf.view(torch.uint8) != ref_sf.view(torch.uint8)).sum().item()
                ),
                cur_sf.numel(),
            )
        if len(tested_backends) < 2:
            print(
                "[INFO] mxfp8_quantize 8x4 refcheck needs two backends "
                "(no host dequantization for 8x4); skipped."
            )
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
                        x_q_cpu, sf_cpu, sf_swizzle_layout=sf_layout
                    )
                    # Move back to GPU for comparison; drop the --alignment padding
                    dequantized = dequantized[:, :k].to(input_tensor.device)
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
            # Write: quantized tensor (fp8, padded to --alignment) + scale factors
            num_elements = m * k
            num_scale_factors = m * padded_k // sf_vec_size
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + m * padded_k * 1  # quantized output write (fp8 = 1 byte)
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
                cur_res["sf_layout"] = sf_layout_str
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
    sf_layout_str = args.sf_layout
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    from flashinfer import SfLayout

    sf_layout_map = {
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
        "linear": SfLayout.layout_linear,
    }
    sf_layout = sf_layout_map[sf_layout_str]

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
        print(f"[VVERBOSE] {sf_layout_str = }")
        print(f"[VVERBOSE] {enable_pdl = }")

    def run_backend(backend, input_tensor):
        return flashinfer.mxfp4_quantize(
            input_tensor,
            backend=backend,
            enable_pdl=enable_pdl,
            sfLayout=sf_layout,
        )

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
                # Dequantize and compare with original.
                try:
                    dequantized = flashinfer.mxfp4_dequantize(
                        x_q,
                        sf,
                        sfLayout=sf_layout,
                    ).to(input_tensor.device)
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
                cur_res["sf_layout"] = sf_layout_str
                cur_res["enable_pdl"] = enable_pdl
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
    from flashinfer import SfLayout

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
    per_token_activation = args.per_token_activation
    nvfp4_4over6_kwargs, nvfp4_4over6_config = _nvfp4_4over6_from_args(args)
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
    if per_token_activation and sf_vec_size != 16:
        raise ValueError("Per-token NVFP4 quantization only supports sf_vec_size=16")

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are "
            "bfloat16, float16, fp8_e4m3."
        )
    ## Done parsing input arguments

    ## Prepare input tensors
    input_shape = (m, k)
    if input_dtype == torch.float8_e4m3fn:
        # randn has no fp8 kernel; draw in fp32 and cast (FP8->FP4 path)
        input_tensor = torch.randn(input_shape, device=device).to(input_dtype)
    else:
        input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    global_sf_tensor = make_nvfp4_global_scale(
        input_tensor,
        per_token_activation=per_token_activation,
        global_scale=global_scale,
        nvfp4_4over6_config=nvfp4_4over6_config,
    )
    # The per-token cuda path reads a tensor scale with .item() and the cute-dsl
    # path uploads a host float with torch.tensor(); either breaks CUDA graph
    # capture, so give cuda a host float and the others the device tensor.
    global_sf_host = float(global_sf_tensor.item()) if per_token_activation else None

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {global_scale = }")
        print(f"[VVERBOSE] {sf_layout_str = }")
        print(f"[VVERBOSE] {do_shuffle = }")
        print(f"[VVERBOSE] {sf_vec_size = }")
        print(f"[VVERBOSE] {enable_pdl = }")
        print(f"[VVERBOSE] {per_token_activation = }")
        for name in NVFP4_QUANT_ENV_VARS:
            print(f"[VVERBOSE] {name} = {os.environ.get(name, '<unset>')}")
        print(f"[VVERBOSE] nvfp4_e4m3_max = {nvfp4_e4m3_max(nvfp4_4over6_config):.0f}")

    def run_backend(backend, input_tensor, global_sf_tensor):
        if per_token_activation and backend == "cuda":
            global_sf_tensor = global_sf_host
        return flashinfer.nvfp4_quantize(
            input_tensor,
            global_sf_tensor,
            sfLayout=sf_layout,
            do_shuffle=do_shuffle,
            sf_vec_size=sf_vec_size,
            enable_pdl=enable_pdl,
            backend=backend,
            per_token_activation=per_token_activation,
            **nvfp4_4over6_kwargs,
        )

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            result = run_backend(cur_backend, input_tensor, global_sf_tensor)
            if per_token_activation:
                x_q, sf, per_token_scale = result
                outputs[cur_backend] = (
                    x_q.detach().clone(),
                    sf.detach().clone(),
                    per_token_scale.detach().clone(),
                )
            else:
                x_q, sf = result
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
            ref_backend = tested_backends[0]
            ref_output = outputs[ref_backend]
            for i in range(len(tested_backends)):
                cur_backend = tested_backends[i]
                if per_token_activation:
                    x_q, sf, per_token_scale = outputs[cur_backend]
                else:
                    x_q, sf = outputs[cur_backend]
                    per_token_scale = None
                if args.verbose >= 2:
                    print(
                        f"[VVERBOSE] Backend {cur_backend}: "
                        f"x_q.shape = {x_q.shape}, x_q.dtype = {x_q.dtype}, "
                        f"sf.shape = {sf.shape}, sf.dtype = {sf.dtype}"
                    )
                    if per_token_scale is not None:
                        print(
                            f"[VVERBOSE] Backend {cur_backend}: "
                            f"per_token_scale.shape = {per_token_scale.shape}, "
                            f"per_token_scale.dtype = {per_token_scale.dtype}"
                        )
                # Verify output shape (M, K/2) for FP4
                expected_shape = (m, k // 2)
                if x_q.shape != expected_shape:
                    print(
                        f"[WARNING] Unexpected output shape: {x_q.shape}, expected {expected_shape}"
                    )
                if cur_backend != ref_backend:
                    # Bitwise by default. FP8 input: cuda and cute-dsl agree on
                    # >95% of bytes (tests/utils/test_fp4_quantize.py
                    # test_nvfp4_quantize_fp8_backend_parity).
                    if input_dtype == torch.float8_e4m3fn:
                        min_match_pct = 95.0
                    else:
                        min_match_pct = 100.0
                    ref_x_q, ref_sf = ref_output[:2]
                    _assert_nvfp4_refcheck_tensor_equal(
                        "x_q", cur_backend, ref_backend, x_q, ref_x_q, min_match_pct
                    )
                    _assert_nvfp4_refcheck_tensor_equal(
                        "sf", cur_backend, ref_backend, sf, ref_sf, min_match_pct
                    )
                    if per_token_activation:
                        if min_match_pct < 100.0:
                            torch.testing.assert_close(
                                per_token_scale, ref_output[2], rtol=1e-2, atol=1e-2
                            )
                        else:
                            _assert_nvfp4_refcheck_tensor_equal(
                                "per_token_scale",
                                cur_backend,
                                ref_backend,
                                per_token_scale,
                                ref_output[2],
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
            if per_token_activation:
                problem_bytes += m * 4  # per-token fp32 scale writes
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
                cur_res["per_token_activation"] = per_token_activation
                cur_res["nvfp4_4over6"] = args.nvfp4_4over6
                cur_res["use_4over6"] = nvfp4_4over6_config is not None
                cur_res["disable_quant_fast_math"] = _env_flag_enabled(
                    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"
                )
                if nvfp4_4over6_config is not None:
                    cur_res["nvfp4_4over6_err_mode"] = nvfp4_4over6_config.err_mode_name
                    cur_res["nvfp4_4over6_err_use_fast_math"] = (
                        nvfp4_4over6_config.err_use_fast_math
                    )
                    cur_res["nvfp4_4over6_e4m3_max"] = nvfp4_4over6_config.e4m3_max
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
    warn_if_pdl_unsupported(args, args.routine)
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


def testGroupedQuantize(args):
    """
    Test the masked grouped quantizers that feed masked grouped GEMMs.

    Routines (input [B, M, K] plus an int32 mask of valid rows per group):
    - mxfp8_grouped_quantize: MXFP8, backends cuda / cutile
    - silu_and_mul_mxfp8_experts_quantize: SiLU-and-mul + MXFP8, backend cuda;
      the input is [B, M, 2K] and --k is the output width
    - scaled_fp4_grouped_quantize: NVFP4, backend cuda
    - silu_and_mul_scaled_nvfp4_experts_quantize: SiLU-and-mul + NVFP4, backend
      cuda; the input is [B, M, 2K] and --k is the output width

    Reference check: mxfp8_grouped_quantize compares each backend's valid rows
    against ungrouped mxfp8_quantize and against the other backend (fp8 code
    space, <= 1 step). The NVFP4 routines compare valid rows bitwise
    against per-group fp4_quantize (after silu_and_mul for the fused routine).
    The NVFP4 global scale is per group ([B], all --global_scale).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print(f"[INFO] Running testGroupedQuantize ({args.routine})")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    routine = args.routine
    backends = args.backends[:]
    batch_size = args.batch_size
    m = args.m
    k = args.k
    masked_m = m if args.masked_m is None else args.masked_m
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    if batch_size is None:
        raise ValueError(f"--batch_size (number of groups) is required for {routine}")
    if not 0 <= masked_m <= m:
        raise ValueError(f"--masked_m ({masked_m}) must be in [0, --m ({m})]")
    is_mxfp8 = routine in (
        "mxfp8_grouped_quantize",
        "silu_and_mul_mxfp8_experts_quantize",
    )
    is_act = routine in (
        "silu_and_mul_scaled_nvfp4_experts_quantize",
        "silu_and_mul_mxfp8_experts_quantize",
    )
    sf_vec_size = 32 if is_mxfp8 else 16
    if k % sf_vec_size != 0:
        raise ValueError(f"k ({k}) must be divisible by {sf_vec_size} for {routine}")

    backends = filter_backends_by_compute_capability(backends, routine, device)
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
    in_k = 2 * k if is_act else k
    a = torch.randn(batch_size, m, in_k, dtype=input_dtype, device=device)
    mask = torch.full((batch_size,), masked_m, dtype=torch.int32, device=device)
    if is_mxfp8:
        nvfp4_4over6_kwargs, nvfp4_4over6_config = {}, None
        global_sf = None
    else:
        nvfp4_4over6_kwargs, nvfp4_4over6_config = _nvfp4_4over6_from_args(args)
        # One global scale per group: the kernel checks
        # input_global_scale.shape[0] == mask.shape[0] (fp4Quantize.cpp), although
        # the docstring says [1].
        global_sf = torch.full(
            (batch_size,), args.global_scale, dtype=torch.float32, device=device
        )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a.shape = }")
        print(f"[VVERBOSE] {a.dtype = }")
        print(f"[VVERBOSE] {masked_m = }")

    def run_backend(backend, a, mask, global_sf):
        if routine == "mxfp8_grouped_quantize":
            return flashinfer.mxfp8_grouped_quantize(a, mask, backend=backend)
        if backend != "cuda":
            raise ValueError(f"Unsupported backend: {backend}")
        if routine == "silu_and_mul_mxfp8_experts_quantize":
            return flashinfer.silu_and_mul_mxfp8_experts_quantize(
                a, mask, enable_pdl=args.enable_pdl
            )
        if backend != "cuda":
            raise ValueError(f"Unsupported backend: {backend}")
        if routine == "scaled_fp4_grouped_quantize":
            return flashinfer.scaled_fp4_grouped_quantize(
                a, mask, global_sf, **nvfp4_4over6_kwargs
            )
        return flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(
            a, mask, global_sf, **nvfp4_4over6_kwargs
        )

    outputs = {}
    if run_refcheck:
        for backend in backends:
            x_q, _ = run_backend(backend, a, mask, global_sf)
            # Logical [M, K', B] -> physical [B, M, K'], valid rows / columns only
            outputs[backend] = (
                x_q.permute(2, 0, 1)[:, :masked_m, : (k if is_mxfp8 else k // 2)]
                .detach()
                .clone()
            )

    backend_times = _bench_backends(
        args, backends, run_backend, (a, mask, global_sf), is_cuda_graph_compatible
    )

    if run_refcheck and len(outputs) > 0:
        if is_mxfp8:
            # Oracle: ungrouped mxfp8_quantize (cuda) of every group's rows, valid
            # rows only, within one fp8 step; plus a cross-backend compare.
            # (silu_and_mul first for the fused routine, rounded to the input
            # dtype as the kernel does)
            src = flashinfer.silu_and_mul(a.reshape(-1, in_k)) if is_act else a
            ref_q, _ = flashinfer.mxfp8_quantize(
                src.reshape(-1, k), is_sf_swizzled_layout=False
            )
            ref_q = ref_q.view(batch_size, m, -1)[:, :masked_m, :k]
            tested = list(outputs.keys())
            for backend in tested:
                dist = fp8_code_distance(outputs[backend], ref_q)
                _check_mismatch(
                    args,
                    routine,
                    f"{backend} vs per-group mxfp8_quantize (> 1 fp8 step)",
                    int((dist > 1).sum().item()),
                    dist.numel(),
                )
            for backend in tested[1:]:
                dist = fp8_code_distance(outputs[backend], outputs[tested[0]])
                _check_mismatch(
                    args,
                    routine,
                    f"{backend} vs {tested[0]} (> 1 fp8 step)",
                    int((dist > 1).sum().item()),
                    dist.numel(),
                )
        else:
            # Per-group fp4_quantize (silu_and_mul first for the fused routine),
            # exact on the valid rows, as tests/utils/test_fp4_quantize.py does.
            ref_name = ("silu_and_mul + " if is_act else "") + "per-group fp4_quantize"
            ref_rows = []
            for g in range(batch_size):
                src = flashinfer.silu_and_mul(a[g]) if is_act else a[g]
                ref_q, _ = flashinfer.fp4_quantize(
                    src, global_sf[g : g + 1], 16, False, True, **nvfp4_4over6_kwargs
                )
                ref_rows.append(ref_q[:masked_m].view(torch.uint8))
            ref_bytes = torch.stack(ref_rows)
            for backend, x_q in outputs.items():
                cur_bytes = x_q.contiguous().view(torch.uint8)
                _check_mismatch(
                    args,
                    routine,
                    f"{backend} vs {ref_name}",
                    int((cur_bytes != ref_bytes).sum().item()),
                    cur_bytes.numel(),
                )

    valid_elems = batch_size * masked_m * k
    out_bytes_per_elem = 1 if is_mxfp8 else 0.5
    problem_bytes = (
        batch_size * masked_m * in_k * input_dtype.itemsize  # input read
        + valid_elems * out_bytes_per_elem  # quantized output write
        + valid_elems // sf_vec_size  # 1-byte scale factors write
    )
    # rough estimate, same constants as the ungrouped routines (fused SwiGLU: 5)
    problem_flops = valid_elems * (5 if is_act else 3)
    columns = {
        "batch_size": batch_size,
        "m": m,
        "k": k,
        "masked_m": masked_m,
        "input_dtype": str(input_dtype),
        "sf_vec_size": sf_vec_size,
    }
    if not is_mxfp8:
        columns["global_scale"] = args.global_scale
        columns["nvfp4_4over6"] = args.nvfp4_4over6
        columns["use_4over6"] = nvfp4_4over6_config is not None
    return _report_results(args, backend_times, problem_bytes, problem_flops, columns)


def testPerTokenGroupQuant8bit(args):
    """
    Test per_token_group_quant_8bit API (cuTile backend).

    Quantizes [M, K] along K in groups of --group_size to fp8_e4m3 / fp8_e5m2 /
    int8 with one fp32 scale per group (the DeepGEMM-style block-scaled FP8
    activation quantization).

    Reference check: recompute amax-based scales and quantized values in torch;
    scales must match (rtol 1e-5), quantized values within one fp8/int8 step.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    from flashinfer.quantization import per_token_group_quant_8bit

    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testPerTokenGroupQuant8bit")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    m = args.m
    k = args.k
    group_size = args.group_size
    column_major_scales = args.column_major_scales
    scale_tma_aligned = args.scale_tma_aligned
    scale_ue8m0 = args.scale_ue8m0
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    eps = 1e-10
    res = []

    if k % group_size != 0:
        raise ValueError(f"k ({k}) must be divisible by --group_size ({group_size})")
    if (scale_tma_aligned or scale_ue8m0) and not column_major_scales:
        raise ValueError(
            "--scale_tma_aligned and --scale_ue8m0 require --column_major_scales"
        )

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. "
            "Supported dtypes are bfloat16, float16, float32."
        )
    dst_dtype = (
        torch.int8
        if args.quant_dtype == "int8"
        else dtype_str_to_torch_dtype(args.quant_dtype)
    )
    ## Done parsing input arguments

    x = torch.randn(m, k, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {x.shape = }")
        print(f"[VVERBOSE] {group_size = }, {dst_dtype = }")
        print(f"[VVERBOSE] {column_major_scales = }, {scale_tma_aligned = }")
        print(f"[VVERBOSE] {scale_ue8m0 = }")

    def run_backend(backend, x):
        return per_token_group_quant_8bit(
            x,
            group_size,
            eps=eps,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
            backend=backend,
        )

    outputs = {}
    if run_refcheck:
        for backend in backends:
            x_q, x_s = run_backend(backend, x)
            outputs[backend] = (x_q.detach().clone(), x_s.float().contiguous())

    backend_times = _bench_backends(
        args, backends, run_backend, (x,), is_cuda_graph_compatible
    )

    if run_refcheck and len(outputs) > 0:
        if dst_dtype == torch.int8:
            q_min, q_max = (
                float(torch.iinfo(dst_dtype).min),
                float(torch.iinfo(dst_dtype).max),
            )
        else:
            q_min, q_max = torch.finfo(dst_dtype).min, torch.finfo(dst_dtype).max
        groups = x.float().view(m, k // group_size, group_size)
        s_ref = groups.abs().amax(dim=-1).clamp(min=eps) / q_max
        if scale_ue8m0:
            s_ref = torch.exp2(torch.ceil(torch.log2(s_ref)))
        for backend, (x_q, x_s) in outputs.items():
            x_s = x_s.view(m, k // group_size)
            num_bad_scale = int(
                (~torch.isclose(x_s, s_ref, rtol=1e-5, atol=0)).sum().item()
            )
            _check_mismatch(
                args, args.routine, f"{backend} scales", num_bad_scale, x_s.numel()
            )
            # Quantize with the kernel's own scales so only rounding is compared
            q_ref = (groups / x_s.unsqueeze(-1)).clamp(q_min, q_max).view(m, k)
            if dst_dtype == torch.int8:
                q_ref = torch.floor(q_ref + 0.5)
            dist = fp8_code_distance(x_q, q_ref.to(dst_dtype))
            _check_mismatch(
                args,
                args.routine,
                f"{backend} values (> 1 step)",
                int((dist > 1).sum().item()),
                dist.numel(),
            )

    num_elements = m * k
    problem_bytes = (
        num_elements * input_dtype.itemsize  # input read
        + num_elements * 1  # 8-bit output write
        + (num_elements // group_size) * 4  # fp32 scale write
    )
    columns = {
        "m": m,
        "k": k,
        "input_dtype": str(input_dtype),
        "quant_dtype": args.quant_dtype,
        "group_size": group_size,
        "column_major_scales": column_major_scales,
        "scale_tma_aligned": scale_tma_aligned,
        "scale_ue8m0": scale_ue8m0,
    }
    return _report_results(
        args, backend_times, problem_bytes, num_elements * 3, columns
    )


def testSiluAndMulNvfp4Quantize(args):
    """
    Test silu_and_mul_nvfp4_quantize API (fused SwiGLU + NVFP4, CuTe-DSL).

    The input is [M, 2K] (gate then up); --k is the output width.

    Reference check: unfused silu_and_mul + fp4_quantize(backend="cute-dsl");
    >= 99% of quantized bytes must match (benchmarks/bench_silu_and_mul_nvfp4_quantize.py).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testSiluAndMulNvfp4Quantize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    m = args.m
    k = args.k
    sf_layout_str = args.sf_layout
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    nvfp4_4over6_kwargs, nvfp4_4over6_config = _nvfp4_4over6_from_args(args)
    res = []

    if k % 16 != 0:
        raise ValueError(f"k ({k}) must be divisible by 16 for {args.routine}")

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

    x = torch.randn(m, 2 * k, dtype=input_dtype, device=device)
    global_sf = torch.tensor([args.global_scale], dtype=torch.float32, device=device)
    is_swizzled = sf_layout_str != "linear"
    is_8x4 = sf_layout_str == "8x4"

    if args.verbose >= 2:
        print(f"[VVERBOSE] {x.shape = }")
        print(f"[VVERBOSE] {sf_layout_str = }")

    def run_backend(backend, x, global_sf):
        if backend != "cute-dsl":
            raise ValueError(f"Unsupported backend: {backend}")
        return flashinfer.silu_and_mul_nvfp4_quantize(
            x,
            global_sf,
            sf_vec_size=16,
            is_sf_swizzled_layout=is_swizzled,
            is_sf_8x4_layout=is_8x4,
            enable_pdl=enable_pdl,
            **nvfp4_4over6_kwargs,
        )

    outputs = {}
    if run_refcheck:
        for backend in backends:
            x_q, _ = run_backend(backend, x, global_sf)
            outputs[backend] = x_q.detach().clone()

    backend_times = _bench_backends(
        args, backends, run_backend, (x, global_sf), is_cuda_graph_compatible
    )

    if run_refcheck and len(outputs) > 0:
        ref_x_q, _ = flashinfer.fp4_quantize(
            flashinfer.silu_and_mul(x),
            global_sf,
            sf_vec_size=16,
            is_sf_swizzled_layout=is_swizzled,
            is_sf_8x4_layout=is_8x4,
            backend="cute-dsl",
            **nvfp4_4over6_kwargs,
        )
        ref_bytes = ref_x_q.view(torch.uint8)
        for backend, x_q in outputs.items():
            num_bad = int((x_q.view(torch.uint8) != ref_bytes).sum().item())
            if num_bad <= 0.01 * ref_bytes.numel():
                num_bad = 0
            _check_mismatch(
                args,
                args.routine,
                f"{backend} vs silu_and_mul + fp4_quantize (< 99% bytes match)",
                num_bad,
                ref_bytes.numel(),
            )

    num_elements = m * k
    problem_bytes = (
        2 * num_elements * input_dtype.itemsize  # gate + up read
        + num_elements // 2  # fp4 output write
        + num_elements // 16  # e4m3 scale write
    )
    columns = {
        "m": m,
        "k": k,
        "input_dtype": str(input_dtype),
        "global_scale": args.global_scale,
        "sf_layout": sf_layout_str,
        "sf_vec_size": 16,
        "enable_pdl": enable_pdl,
        "nvfp4_4over6": args.nvfp4_4over6,
        "use_4over6": nvfp4_4over6_config is not None,
    }
    return _report_results(
        args, backend_times, problem_bytes, num_elements * 5, columns
    )


def testNvfp4KvQuantize(args):
    """
    Test nvfp4_kv_quantize / nvfp4_kv_dequantize APIs (NVFP4 KV-cache format,
    linear block scales, [M, K] with K = head_dim).

    nvfp4_kv_quantize times quantization of a bf16/fp16 [M, K] tensor;
    nvfp4_kv_dequantize times dequantization of its output back to the input dtype.

    Reference check: quantize -> dequantize round trip, mean relative error < 0.5
    (tests/utils/test_fp4_kv_quantization.py test_nvfp4_kv_roundtrip).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print(f"[INFO] Running testNvfp4KvQuantize ({args.routine})")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    is_quant = args.routine == "nvfp4_kv_quantize"
    backends = args.backends[:]
    m = args.m
    k = args.k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    if k % 16 != 0:
        raise ValueError(f"k ({k}) must be divisible by 16 for {args.routine}")

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

    x = torch.randn(m, k, dtype=input_dtype, device=device)
    global_sf = torch.tensor([args.global_scale], dtype=torch.float32, device=device)
    fp4_data, block_scales = flashinfer.nvfp4_kv_quantize(x, global_sf)

    def run_backend(backend, x, fp4_data, block_scales, global_sf):
        if backend != "cuda":
            raise ValueError(f"Unsupported backend: {backend}")
        if is_quant:
            return flashinfer.nvfp4_kv_quantize(x, global_sf)
        return flashinfer.nvfp4_kv_dequantize(
            fp4_data, block_scales, global_sf, output_dtype=input_dtype
        )

    backend_times = _bench_backends(
        args,
        backends,
        run_backend,
        (x, fp4_data, block_scales, global_sf),
        is_cuda_graph_compatible,
    )

    if run_refcheck:
        for backend in backends:
            if is_quant:
                q, sf = run_backend(backend, x, fp4_data, block_scales, global_sf)
                recon = flashinfer.nvfp4_kv_dequantize(
                    q, sf, global_sf, output_dtype=input_dtype
                )
            else:
                recon = run_backend(backend, x, fp4_data, block_scales, global_sf)
            x_f, recon_f = x.float(), recon.float()
            nonzero = x_f.abs() > 1e-6
            rel_err = (
                (x_f[nonzero] - recon_f[nonzero]).abs() / x_f[nonzero].abs()
            ).mean()
            if args.verbose >= 2:
                print(f"[VVERBOSE] {backend}: round-trip mean rel err {rel_err:.4f}")
            _check_mismatch(
                args,
                args.routine,
                f"{backend} round-trip mean relative error {rel_err:.4f} >= 0.5",
                int(rel_err >= 0.5),
                1,
            )

    num_elements = m * k
    problem_bytes = (
        num_elements * input_dtype.itemsize  # bf16/fp16 side
        + num_elements // 2  # packed fp4 side
        + num_elements // 16  # e4m3 block scales
    )
    columns = {
        "m": m,
        "k": k,
        "input_dtype": str(input_dtype),
        "global_scale": args.global_scale,
    }
    return _report_results(
        args, backend_times, problem_bytes, num_elements * 2, columns
    )


def testNvfp4KvDequantizePaged(args):
    """
    Test nvfp4_kv_dequantize_paged API: gather an NVFP4 paged K/V cache through
    a page table and write contiguous [B, S, H, D] K/V outputs.

    Mapping: --batch_size requests, --m tokens per request, --k head_dim,
    --num_kv_heads, --page_size, --kv_layout. Pages are randomly permuted.

    Reference check: gather the same pages in torch and dequantize them with
    nvfp4_kv_dequantize; outputs must match (atol/rtol 1e-3, as the library test).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testNvfp4KvDequantizePaged")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]
    batch_size = args.batch_size
    seq_len = args.m
    head_dim = args.k
    num_kv_heads = args.num_kv_heads
    page_size = args.page_size
    kv_layout = args.kv_layout
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    if batch_size is None:
        raise ValueError("--batch_size is required for nvfp4_kv_dequantize_paged")
    if head_dim % 16 != 0:
        raise ValueError(f"head_dim (--k {head_dim}) must be divisible by 16")

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

    pages_per_request = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * pages_per_request
    if kv_layout == "NHD":
        page_shape = (num_pages, page_size, num_kv_heads)
    else:
        page_shape = (num_pages, num_kv_heads, page_size)

    def random_cache():
        data = torch.randint(
            0, 256, (*page_shape, head_dim // 2), dtype=torch.uint8, device=device
        )
        scales = torch.randint(
            1, 120, (*page_shape, head_dim // 16), dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn)
        return data, scales

    k_cache, k_sf = random_cache()
    v_cache, v_sf = random_cache()
    block_tables = (
        torch.randperm(num_pages, device=device)
        .to(torch.int32)
        .view(batch_size, pages_per_request)
    )
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    k_scale = torch.tensor([args.global_scale], dtype=torch.float32, device=device)
    v_scale = torch.tensor([args.global_scale], dtype=torch.float32, device=device)
    out_shape = (batch_size, seq_len, num_kv_heads, head_dim)
    output_k = torch.empty(out_shape, dtype=input_dtype, device=device)
    output_v = torch.empty(out_shape, dtype=input_dtype, device=device)

    def run_backend(backend, output_k, output_v):
        if backend != "cuda":
            raise ValueError(f"Unsupported backend: {backend}")
        return flashinfer.nvfp4_kv_dequantize_paged(
            (k_cache, v_cache),
            (k_sf, v_sf),
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
            output_k,
            output_v,
            kv_layout=kv_layout,
        )

    backend_times = _bench_backends(
        args, backends, run_backend, (output_k, output_v), is_cuda_graph_compatible
    )

    if run_refcheck:

        def gathered_reference(cache, sf, scale):
            # [B, pages, page_size, H, ...] in token order, then drop the tail
            def to_tokens(t):
                t = t[block_tables.long()]
                if kv_layout == "HND":
                    t = t.transpose(2, 3)
                return t.reshape(batch_size, -1, num_kv_heads, t.shape[-1])[:, :seq_len]

            data = to_tokens(cache).reshape(-1, head_dim // 2)
            scales = to_tokens(sf.view(torch.uint8)).reshape(-1, head_dim // 16)
            return flashinfer.nvfp4_kv_dequantize(
                data.contiguous(), scales.contiguous(), scale, output_dtype=input_dtype
            ).view(out_shape)

        ref_k = gathered_reference(k_cache, k_sf, k_scale)
        ref_v = gathered_reference(v_cache, v_sf, v_scale)
        for backend in backends:
            run_backend(backend, output_k, output_v)
            for name, out, ref in (("K", output_k, ref_k), ("V", output_v, ref_v)):
                num_bad, num_total, _ = is_close_stats(
                    ref.float(), out.float(), rtol=1e-3, atol=1e-3
                )
                _check_mismatch(
                    args,
                    args.routine,
                    f"{backend} {name} vs gathered dequant",
                    num_bad,
                    num_total,
                )

    num_tokens = batch_size * seq_len * num_kv_heads
    problem_bytes = (
        2
        * num_tokens
        * (
            head_dim // 2  # packed fp4 read
            + head_dim // 16  # e4m3 scale read
            + head_dim * input_dtype.itemsize  # dequantized write
        )
    )
    columns = {
        "batch_size": batch_size,
        "m": seq_len,
        "k": head_dim,
        "num_kv_heads": num_kv_heads,
        "page_size": page_size,
        "kv_layout": kv_layout,
        "input_dtype": str(input_dtype),
        "global_scale": args.global_scale,
    }
    return _report_results(
        args, backend_times, problem_bytes, 2 * num_tokens * head_dim, columns
    )


def testPackbits(args):
    """
    Test packbits / segment_packbits APIs (bool -> uint8 bit packing, used for
    custom attention masks).

    packbits packs --m * --k bits; segment_packbits packs --m segments of --k
    bits each. segment_packbits sizes its output on the host (a device sync), so
    it is timed without CUDA graphs.

    Reference check: numpy.packbits per segment, exact.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print(f"[INFO] Running testPackbits ({args.routine})")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    is_segment = args.routine == "segment_packbits"
    backends = args.backends[:]
    num_segments = args.m
    seg_len = args.k
    bitorder = args.bitorder
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    if is_segment and is_cuda_graph_compatible:
        print(
            "[INFO] segment_packbits synchronizes with the host to size its "
            "output; disabling CUDA graph."
        )
        is_cuda_graph_compatible = False

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res
    ## Done parsing input arguments

    x = torch.randint(0, 2, (num_segments * seg_len,), device=device).bool()
    indptr = torch.arange(
        0, (num_segments + 1) * seg_len, seg_len, dtype=torch.int32, device=device
    )

    def run_backend(backend, x, indptr):
        if backend != "cuda":
            raise ValueError(f"Unsupported backend: {backend}")
        if is_segment:
            return flashinfer.segment_packbits(x, indptr, bitorder=bitorder)
        return flashinfer.packbits(x, bitorder=bitorder)

    backend_times = _bench_backends(
        args, backends, run_backend, (x, indptr), is_cuda_graph_compatible
    )

    if run_refcheck:
        x_np = x.cpu().numpy()
        if is_segment:
            ref = np.concatenate(
                [
                    np.packbits(seg, bitorder=bitorder)
                    for seg in x_np.reshape(num_segments, seg_len)
                ]
            )
        else:
            ref = np.packbits(x_np, bitorder=bitorder)
        for backend in backends:
            out = run_backend(backend, x, indptr)
            out = (out[0] if is_segment else out).cpu().numpy()
            num_bad = int((out != ref).sum()) if out.shape == ref.shape else ref.size
            _check_mismatch(
                args, args.routine, f"{backend} vs numpy.packbits", num_bad, ref.size
            )

    num_bits = num_segments * seg_len
    packed_bytes = (
        num_segments * ((seg_len + 7) // 8) if is_segment else (num_bits + 7) // 8
    )
    problem_bytes = num_bits + packed_bytes  # 1-byte bool read + packed write
    columns = {"m": num_segments, "k": seg_len, "bitorder": bitorder}
    return _report_results(args, backend_times, problem_bytes, num_bits, columns)
