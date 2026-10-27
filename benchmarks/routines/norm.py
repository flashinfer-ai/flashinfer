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
    blank_unused_mode_columns,
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    is_close_stats,
    filter_backends_by_compute_capability,
    warn_if_pdl_unsupported,
)


def _resolved_norm_backend(device):
    """Return the kernel path the dual-path norm APIs actually take on ``device``.

    rmsnorm / fused_add_rmsnorm / gemma_* / rmsnorm_quant / fused_add_rmsnorm_quant /
    layernorm have no ``backend=`` argument: they run CuTe-DSL kernels unless
    FLASHINFER_USE_CUDA_NORM=1, nvidia-cutlass-dsl is missing, or the installed DSL
    cannot target this arch, in which case they fall back to the CUDA JIT kernels.
    """
    use_cuda_norm = getattr(flashinfer.norm, "_use_cuda_norm", None)
    if use_cuda_norm is not None and use_cuda_norm(device):
        return "cuda"
    return "cute-dsl"


def _out_dtype_or_bf16(args):
    """--out_dtype for routines whose API defaults to bf16 output (see parse_norm_args)."""
    return args.out_dtype if args.out_dtype_given else "bfloat16"


def _csv_out_dtype(out_dtype):
    """CSV spelling of an out_dtype, matching the sibling quant routines: fp8 as the torch
    dtype string (as rmsnorm_quant writes), nvfp4/mxfp8 as their flag names (as
    rmsnorm_fp4quant writes)."""
    if out_dtype.startswith("fp8") or out_dtype == "bfloat16":
        return str(dtype_str_to_torch_dtype(out_dtype))
    return out_dtype


def _fp8_code_stats(ref, out):
    """Compare two fp8 tensors in code space. Returns (n_beyond_one_ulp, n_elements, pct, n_ties)."""

    def order(t):
        c = t.view(torch.uint8).to(torch.int16)
        return torch.where(c & 0x80 != 0, -(c & 0x7F), c & 0x7F)

    n = ref.numel()
    nan = torch.isnan(ref.float()) | torch.isnan(out.float())
    dist = (order(out) - order(ref)).abs()
    dist = torch.where(nan, torch.zeros_like(dist), dist)
    n_nan = int((torch.isnan(out.float()) & ~torch.isnan(ref.float())).sum())
    n_bad = int((dist > 1).sum()) + n_nan
    n_ties = int((dist == 1).sum())
    return n_bad, n, n_bad / n * 100.0, n_ties


def run_norm_test(args):
    """Run one norm testlist line; see _dispatch_norm_test for the routine table."""
    res = _dispatch_norm_test(args)
    # Mode flags added in 2026-10 stay blank on rows that do not use them, so
    # pre-existing lines keep their CSV columns (and dashboard keys).
    return blank_unused_mode_columns(
        args,
        res,
        defaults={},
        owners={
            "norm_variant": (
                "gated_residual_rmsnorm",
                "gated_residual_next_rmsnorm",
                "rmsnorm_sigmoid_gate",
            ),
        },
    )


def _dispatch_norm_test(args):
    """
    Run a norm test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "rmsnorm":
        return testRmsnorm(args)
    elif args.routine == "fused_add_rmsnorm":
        return testFusedAddRmsnorm(args)
    elif args.routine == "gemma_rmsnorm":
        return testGemmaRmsnorm(args)
    elif args.routine == "gemma_fused_add_rmsnorm":
        return testGemmaFusedAddRmsnorm(args)
    elif args.routine == "rmsnorm_quant":
        return testRmsnormQuant(args)
    elif args.routine == "fused_add_rmsnorm_quant":
        return testFusedAddRmsnormQuant(args)
    elif args.routine == "layernorm_quant":
        return testLayernormQuant(args)
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
    elif args.routine == "fused_add_rmsnorm_fp8_block_quant":
        return testFusedAddRmsnormFp8BlockQuant(args)
    elif args.routine == "layernorm":
        return testLayernorm(args)
    elif args.routine == "act_and_mul":
        return testActAndMul(args)
    elif args.routine == "mhc":
        return testMhc(args)
    elif args.routine in _GATED_NORM_VARIANTS:
        return testGatedResidualNorm(args)
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
        default=None,
        choices=["fp8_e4m3", "fp8_e5m2", "nvfp4", "mxfp4", "mxfp8", "bfloat16"],
        help="Output dtype for quantized operations. fp8_e4m3/fp8_e5m2 for FP8 quant; nvfp4/mxfp4 for FP4 quant. "
        "Default: fp8_e4m3. "
        "fused_rmsnorm_silu (bfloat16/fp8_e4m3/nvfp4), fused_dit_layernorm (bfloat16/nvfp4/mxfp8) and "
        "fused_qk_rmsnorm_rope (bfloat16/fp8_e4m3) default to bfloat16 output when this flag is omitted.",
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

    # act_and_mul specific arguments
    parser.add_argument(
        "--act_type",
        type=str,
        required=False,
        default="silu",
        choices=["silu", "gelu", "gelu_tanh"],
        help="Gated activation for act_and_mul: silu_and_mul, gelu_and_mul or gelu_tanh_and_mul. "
        "--hidden_size is the output width; the input is (batch_size, 2 * hidden_size).",
    )

    # mhc specific arguments
    parser.add_argument(
        "--mhc_mode",
        type=str,
        required=False,
        default="post",
        choices=["post", "pre_big_fuse", "pre_big_fuse_with_prenorm"],
        help="mHC API for the mhc routine: mhc_post, mhc_pre_big_fuse or mhc_pre_big_fuse_with_prenorm.",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        required=False,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Split factor of dot_mix/sqrsum for mhc --mhc_mode pre_big_fuse. Default: 1",
    )

    # gated residual norms (gated_residual_rmsnorm, gated_residual_next_rmsnorm,
    # rmsnorm_sigmoid_gate)
    parser.add_argument(
        "--norm_variant",
        type=str,
        required=False,
        default="",
        choices=[
            "",
            "weighted",
            "unweighted",
            "tensor_alpha",
            "scalar_alpha",
            "post_weight",
            "default",
        ],
        help="Gated residual norm variant. gated_residual_rmsnorm: weighted (default) / "
        "unweighted; gated_residual_next_rmsnorm: tensor_alpha (default) / scalar_alpha / "
        "post_weight; rmsnorm_sigmoid_gate: default.",
    )

    args = parser.parse_args(line)
    # Routines whose API emits bf16 by default (fused_rmsnorm_silu, fused_dit_layernorm,
    # fused_qk_rmsnorm_rope) only switch to a quantized output when --out_dtype is given
    # explicitly; the shared fp8_e4m3 default keeps its meaning for every other routine.
    # (default=None is the "not given" sentinel; it is resolved to fp8_e4m3 right here.)
    args.out_dtype_given = args.out_dtype is not None
    if not args.out_dtype_given:
        args.out_dtype = "fp8_e4m3"
    # Blank the routine-specific flags of other routines so their CSV columns stay empty.
    if args.routine != "act_and_mul":
        args.act_type = ""
    if args.routine != "mhc":
        args.mhc_mode = ""
        args.num_splits = ""
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

    # Default backend to cute-dsl for rmsnorm variants (CuTe-DSL kernels by default)
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

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
        if backend == "cute-dsl":
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
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testFusedAddRmsnorm(args):
    """
    Test fused_add_rmsnorm API.

    This test benchmarks residual += input followed by RMSNorm, with the
    normalized output written back to input and residual updated in-place.
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedAddRmsnorm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = args.backends[:]

    # Default backend to cute-dsl for rmsnorm variants (CuTe-DSL kernels by default)
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
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

    input_shape = (batch_size, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    residual_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {residual_tensor.shape = }")
        print(f"[VVERBOSE] {weight.shape = }")

    def run_backend(backend, input_tensor, residual_tensor, weight):
        if backend == "cute-dsl":
            flashinfer.fused_add_rmsnorm(
                input_tensor,
                residual_tensor,
                weight,
                eps=eps,
                enable_pdl=enable_pdl,
            )
            return input_tensor, residual_tensor
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_residual = residual_tensor.float() + input_tensor.float()
        rms = torch.sqrt(torch.mean(reference_residual**2, dim=-1, keepdim=True) + eps)
        reference_output = (reference_residual / rms * weight.float()).to(input_dtype)
        reference_residual = reference_residual.to(input_dtype)
        has_reference_output = True

    backend_times = {backend: [] for backend in backends}
    outputs = {}
    residual_outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            cur_input = input_tensor.clone()
            cur_residual = residual_tensor.clone()
            out, residual_out = run_backend(
                cur_backend, cur_input, cur_residual, weight
            )
            outputs[cur_backend] = out.detach().clone()
            residual_outputs[cur_backend] = residual_out.detach().clone()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(
                cur_backend,
                input_tensor.clone(),
                residual_tensor.clone(),
                weight,
            ),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0 and run_refcheck and has_reference_output:
        for backend in tested_backends:
            for name, ref, out in (
                ("output", reference_output, outputs[backend]),
                ("residual", reference_residual, residual_outputs[backend]),
            ):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(ref, out, rtol=1e-2, atol=1e-2)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] {name} tensor mismatch from backend {backend}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {backend} {name} mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth: read input + residual + weight; write residual + normalized input
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + num_elements * input_dtype.itemsize  # residual read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements * input_dtype.itemsize  # residual write
                + num_elements * input_dtype.itemsize  # normalized input write
            )
            problem_flops = num_elements * 6
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
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testGemmaRmsnorm(args):
    """
    Test gemma_rmsnorm API.

    Gemma RMSNorm uses (weight + 1) instead of weight.
    """
    if args.verbose >= 1:
        print("[INFO] Running testGemmaRmsnorm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = args.backends[:]

    # Default backend to cute-dsl for rmsnorm variants (CuTe-DSL kernels by default)
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

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
        if backend == "cute-dsl":
            return flashinfer.gemma_rmsnorm(
                input_tensor, weight, eps=eps, enable_pdl=enable_pdl
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        rms = torch.sqrt(
            torch.mean(input_tensor.float() ** 2, dim=-1, keepdim=True) + eps
        )
        reference_output = (input_tensor.float() / rms * (weight.float() + 1.0)).to(
            input_dtype
        )
        has_reference_output = True

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

            # Memory bandwidth: read input + weight; write output
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements * input_dtype.itemsize  # output write
            )
            problem_flops = num_elements * 5
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
                cur_res["num_heads"] = num_heads if num_heads else ""
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testGemmaFusedAddRmsnorm(args):
    """
    Test gemma_fused_add_rmsnorm API.

    This is fused_add_rmsnorm with Gemma-style (weight + 1) scaling.
    """
    if args.verbose >= 1:
        print("[INFO] Running testGemmaFusedAddRmsnorm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = args.backends[:]

    # Default backend to cute-dsl for rmsnorm variants (CuTe-DSL kernels by default)
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
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

    input_shape = (batch_size, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    residual_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {residual_tensor.shape = }")
        print(f"[VVERBOSE] {weight.shape = }")

    def run_backend(backend, input_tensor, residual_tensor, weight):
        if backend == "cute-dsl":
            flashinfer.gemma_fused_add_rmsnorm(
                input_tensor,
                residual_tensor,
                weight,
                eps=eps,
                enable_pdl=enable_pdl,
            )
            return input_tensor, residual_tensor
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_residual = residual_tensor.float() + input_tensor.float()
        rms = torch.sqrt(torch.mean(reference_residual**2, dim=-1, keepdim=True) + eps)
        reference_output = (reference_residual / rms * (weight.float() + 1.0)).to(
            input_dtype
        )
        reference_residual = reference_residual.to(input_dtype)
        has_reference_output = True

    backend_times = {backend: [] for backend in backends}
    outputs = {}
    residual_outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            cur_input = input_tensor.clone()
            cur_residual = residual_tensor.clone()
            out, residual_out = run_backend(
                cur_backend, cur_input, cur_residual, weight
            )
            outputs[cur_backend] = out.detach().clone()
            residual_outputs[cur_backend] = residual_out.detach().clone()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(
                cur_backend,
                input_tensor.clone(),
                residual_tensor.clone(),
                weight,
            ),
        )

    tested_backends = list(outputs.keys())
    if len(tested_backends) > 0 and run_refcheck and has_reference_output:
        for backend in tested_backends:
            for name, ref, out in (
                ("output", reference_output, outputs[backend]),
                ("residual", reference_residual, residual_outputs[backend]),
            ):
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                ) = is_close_stats(ref, out, rtol=1e-2, atol=1e-2)
                if num_different_elements > 0:
                    print(
                        f"[ERROR] {name} tensor mismatch from backend {backend}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {backend} {name} mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth: read input + residual + weight; write residual + normalized input
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + num_elements * input_dtype.itemsize  # residual read
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements * input_dtype.itemsize  # residual write
                + num_elements * input_dtype.itemsize  # normalized input write
            )
            problem_flops = num_elements * 6
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
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
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

    # Default backend to cute-dsl for rmsnorm variants (CuTe-DSL kernels by default)
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    # Pre-allocate scale on device once. Passing a Python float forces flashinfer's
    # _normalize_scale_tensor to create a tensor on each call, which triggers a
    # host->device copy that is illegal under CUDA graph capture (and emits a
    # FutureWarning about the deprecated float form).
    scale = torch.tensor([args.scale], dtype=torch.float32, device=device)
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
        if backend == "cute-dsl":
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
            (rmsnorm_output / scale)
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
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                    num_one_ulp_ties,
                ) = _fp8_code_stats(reference_output, tested_outputs[i])
                if num_one_ulp_ties > 0 and args.verbose >= 1:
                    print(
                        f"[INFO] Backend {tested_backends[i]}: {num_one_ulp_ties}/{num_elements} "
                        f"({num_one_ulp_ties / num_elements * 100:.2f}%) elements land on an adjacent fp8 code"
                    )
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ by more than one fp8 code"
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
                cur_res["scale"] = scale.item()
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testLayernormQuant(args):
    """
    Test layernorm_quant API.

    This test:
    1. Generates random input tensors
    2. Runs layernorm_quant with quantized output
    3. Runs reference check
    4. Measures performance metrics (memory bandwidth)

    Note: LayerNorm is memory-bandwidth bound, so TB/sec is the primary metric.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testLayernormQuant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends[:]  # Make a copy to avoid modifying the original

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    # Pre-allocate scale on device once. Passing a Python float forces flashinfer's
    # _normalize_scale_tensor to create a tensor on each call, which triggers a
    # host->device copy that is illegal under CUDA graph capture (and emits a
    # FutureWarning about the deprecated float form).
    scale = torch.tensor([args.scale], dtype=torch.float32, device=device)
    eps = args.eps
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. layernorm_quant only supports bfloat16."
        )

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported out dtype: {args.out_dtype}. Supported dtypes are fp8_e4m3, fp8_e5m2."
        )
    ## Done parsing input arguments

    ## Prepare input tensors (2D only for layernorm_quant)
    input_shape = (batch_size, hidden_size)

    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    gamma = torch.randn(hidden_size, dtype=torch.float32, device=device)
    beta = torch.randn(hidden_size, dtype=torch.float32, device=device)
    out_tensor = torch.empty(input_shape, dtype=out_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {gamma.shape = }")
        print(f"[VVERBOSE] {out_tensor.dtype = }")
        print(f"[VVERBOSE] {scale = }")

    def run_backend(backend, out_tensor, input_tensor, gamma, beta):
        if backend == "cuda":
            # Call the CUDA JIT module directly: the public API dispatches to
            # the CuTe-DSL kernel by default, so both arms can be timed in
            # one process.
            flashinfer.norm.get_norm_module().layernorm_quant(
                out_tensor,
                input_tensor,
                gamma,
                beta,
                scale,
                eps,
            )
            return out_tensor
        elif backend == "cute-dsl":
            flashinfer.norm.layernorm_quant(
                out_tensor,
                input_tensor,
                gamma,
                beta,
                scale=scale,
                eps=eps,
            )
            return out_tensor
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Reference: PyTorch implementation of LayerNorm + quantization
    has_reference_output = False
    if run_refcheck:
        layernorm_output = torch.nn.functional.layer_norm(
            input_tensor.float(), (hidden_size,), weight=gamma, bias=beta, eps=eps
        )
        # The CUDA kernel rounds the normalized value to bf16 before scaling
        # and again before casting to fp8; the CuTe-DSL kernel stays in fp32.
        # Mirror each so a backend's own rounding does not inflate the
        # adjacent-fp8-code tie count.
        cuda_ref = (
            (layernorm_output.to(input_dtype).float() / scale)
            .to(input_dtype)
            .float()
            .clamp(torch.finfo(out_dtype).min, torch.finfo(out_dtype).max)
            .to(out_dtype)
        )
        cute_ref = (
            (layernorm_output / scale)
            .clamp(torch.finfo(out_dtype).min, torch.finfo(out_dtype).max)
            .to(out_dtype)
        )
        reference_outputs = {"cuda": cuda_ref, "cute-dsl": cute_ref}
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        # Create fresh output tensor for each run
        cur_out = torch.empty(input_shape, dtype=out_dtype, device=device)
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend(cur_backend, cur_out, input_tensor, gamma, beta)
                .detach()
                .clone()
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, out_tensor, input_tensor, gamma, beta),
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
                    num_one_ulp_ties,
                ) = _fp8_code_stats(
                    reference_outputs[tested_backends[i]], tested_outputs[i]
                )
                if num_one_ulp_ties > 0 and args.verbose >= 1:
                    print(
                        f"[INFO] Backend {tested_backends[i]}: {num_one_ulp_ties}/{num_elements} "
                        f"({num_one_ulp_ties / num_elements * 100:.2f}%) elements land on an adjacent fp8 code"
                    )
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ by more than one fp8 code"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Memory bandwidth calculation for LayerNorm + Quant
            # Read: input tensor + gamma/beta tensors
            # Write: output tensor (quantized, smaller dtype)
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize  # input read
                + 2 * hidden_size * torch.float32.itemsize  # gamma and beta read
                + num_elements * out_dtype.itemsize  # output write (quantized)
            )
            problem_flops = num_elements * 7  # rough estimate
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
                cur_res["scale"] = scale.item()
                cur_res["eps"] = eps
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

    # Default backend to cute-dsl for rmsnorm variants (CuTe-DSL kernels by default)
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    # Pre-allocate scale on device once. Passing a Python float forces flashinfer's
    # _normalize_scale_tensor to create a tensor on each call, which triggers a
    # host->device copy that is illegal under CUDA graph capture (and emits a
    # FutureWarning about the deprecated float form).
    scale = torch.tensor([args.scale], dtype=torch.float32, device=device)
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
        if backend == "cute-dsl":
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
        # Compute the add in fp32 to match the CuTe kernel and unit-test reference.
        ref_residual = input_tensor.float() + residual_tensor.float()
        # Step 2: RMSNorm on residual
        rms = torch.sqrt(torch.mean(ref_residual**2, dim=-1, keepdim=True) + eps)
        rmsnorm_output = ref_residual / rms * weight.float()
        # Quantize to output dtype
        reference_output = (
            (rmsnorm_output / scale)
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
                (
                    num_different_elements,
                    num_elements,
                    num_different_elements_percentage,
                    num_one_ulp_ties,
                ) = _fp8_code_stats(reference_output, tested_outputs[i])
                if num_one_ulp_ties > 0 and args.verbose >= 1:
                    print(
                        f"[INFO] Backend {tested_backends[i]}: {num_one_ulp_ties}/{num_elements} "
                        f"({num_one_ulp_ties / num_elements * 100:.2f}%) elements land on an adjacent fp8 code"
                    )
                if num_different_elements > 0:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}: "
                        f"{num_different_elements}/{num_elements} ({num_different_elements_percentage:.2f}%) elements differ by more than one fp8 code"
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
                cur_res["scale"] = scale.item()
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
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
    2. Runs fused_rmsnorm_silu; bf16 output unless --out_dtype selects fp8_e4m3 (SM89+)
       or nvfp4 (SM100+, packed FP4 values plus one E4M3 scale per 16 elements)
    3. Optionally runs reference check (bf16 and fp8 outputs)
    4. Measures performance metrics (memory bandwidth)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedRmsnormSilu")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)

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

    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"fused_rmsnorm_silu requires bfloat16 input, got {args.input_dtype}"
        )
    out_dtype = _out_dtype_or_bf16(args)
    if out_dtype not in ("bfloat16", "fp8_e4m3", "nvfp4"):
        raise ValueError(
            f"fused_rmsnorm_silu supports --out_dtype bfloat16, fp8_e4m3 or nvfp4, got {out_dtype}"
        )

    input_shape = (batch_size, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device=device) * 1.5 + 0.5
    block_scale = None
    if out_dtype == "nvfp4":
        out = torch.empty(
            batch_size, hidden_size // 2, dtype=torch.float4_e2m1fn_x2, device=device
        )
        block_scale = torch.empty(
            batch_size, hidden_size // 16, dtype=torch.float8_e4m3fn, device=device
        )
    else:
        out = torch.empty(
            input_shape, dtype=dtype_str_to_torch_dtype(out_dtype), device=device
        )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {weight.shape = }")
        print(f"[VVERBOSE] {out_dtype = }")

    def run_fn(input_tensor, weight, out, block_scale):
        return flashinfer.fused_rmsnorm_silu(
            input_tensor, weight, eps=eps, out=out, block_scale=block_scale
        )

    if run_refcheck and out_dtype == "nvfp4":
        print("[WARNING] --refcheck is not supported for nvfp4 output.")
    elif run_refcheck:
        rms = torch.sqrt(
            torch.mean(input_tensor.float() ** 2, dim=-1, keepdim=True) + eps
        )
        x_norm = input_tensor.float() / rms * weight.float()
        reference_f32 = torch.nn.functional.silu(x_norm)
        test_out = run_fn(input_tensor, weight, out, block_scale)
        if out_dtype == "bfloat16":
            reference_output = reference_f32.to(torch.bfloat16)
            tol = 2e-2
        else:
            # Tolerance from tests/norm/test_fused_rmsnorm_silu.py (test_lut_fp8).
            reference_output = (
                reference_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
            )
            test_out = test_out.float()
            tol = 0.125
        (
            num_different_elements,
            num_elements,
            num_different_elements_percentage,
        ) = is_close_stats(reference_output, test_out, rtol=tol, atol=tol)
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
        input_args=(input_tensor, weight, out, block_scale),
    )

    if len(times) > 0:
        median_time = np.median(times)
        std_time = np.std(times)

        num_elements = np.prod(input_shape)
        if out_dtype == "nvfp4":
            output_bytes = num_elements // 2 + num_elements // 16  # FP4 + E4M3 scales
        else:
            output_bytes = num_elements * out.dtype.itemsize
        problem_bytes = (
            num_elements * input_dtype.itemsize  # input read
            + hidden_size * input_dtype.itemsize  # weight read
            + output_bytes  # output write
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
            if args.out_dtype_given:
                cur_res["out_dtype"] = _csv_out_dtype(out_dtype)
            cur_res["eps"] = eps
            cur_res["backend"] = "cuda"
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testFusedDitLayernorm(args):
    """
    Test fused DIT LayerNorm API (3 modes).

    Benchmarks the fused kernel for the specified mode. Output is bf16 unless
    --out_dtype selects nvfp4 or mxfp8 (SM100+). Hidden dim is fixed at 3072
    (WAN 2.2 5B target). At -v and above, bf16 output is also compared against
    eager PyTorch.
    """
    from flashinfer.diffusion_ops import (
        fused_dit_gate_residual_layernorm_gamma_beta,
        fused_dit_gate_residual_layernorm_scale_shift,
        fused_dit_residual_layernorm_scale_shift,
    )

    if args.verbose >= 1:
        print("[INFO] Running testFusedDitLayernorm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    # For DIT LayerNorm, hidden_size is always 3072 (WAN 2.2 5B).
    # We reuse --hidden_size as num_rows (sequence length) for CLI compatibility.
    # (Not written to the num_rows column: existing rows would change their CSV key.)
    num_rows = args.hidden_size
    batch_size = args.batch_size
    hidden_dim = 3072
    eps = args.eps
    mode = args.dit_mode
    is_cuda_graph_compatible = not args.no_cuda_graph

    out_dtype = _out_dtype_or_bf16(args)
    if out_dtype not in ("bfloat16", "nvfp4", "mxfp8"):
        raise ValueError(
            f"fused_dit_layernorm supports --out_dtype bfloat16, nvfp4 or mxfp8, got {out_dtype}"
        )
    quant_kwargs = dict(
        use_nvfp4=out_dtype == "nvfp4",
        use_mxfp8=out_dtype == "mxfp8",
        global_scaling_factor=(
            torch.ones(1, dtype=torch.float32, device=device)
            if out_dtype == "nvfp4"
            else None
        ),
    )

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

        def fused_fn(input_t, residual):
            return fused_dit_gate_residual_layernorm_gamma_beta(
                input_t,
                residual,
                gate,
                gamma,
                beta,
                gate_bias=gate_bias,
                epsilon=eps,
                **quant_kwargs,
            )

        def eager_fn(input_t, residual):
            r = residual.float() + input_t.float() * (gate.float() + gate_bias.float())
            n = torch.layer_norm(r, [hidden_dim], weight=gamma, bias=beta, eps=eps)
            return r.to(torch.bfloat16), n.to(torch.bfloat16)

    elif mode == "gate_residual_scale_shift":

        def fused_fn(input_t, residual):
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
                **quant_kwargs,
            )

        def eager_fn(input_t, residual):
            r = residual.float() + input_t.float() * (
                c_gate.float() + c_gate_bias.float()
            )
            n = torch.layer_norm(r, [hidden_dim], eps=eps)
            n = n * (1 + scale.float() + scale_bias.float()) + (
                shift.float() + shift_bias.float()
            )
            return r.to(torch.bfloat16), n.to(torch.bfloat16)

    elif mode == "residual_scale_shift":

        def fused_fn(input_t, residual):
            return fused_dit_residual_layernorm_scale_shift(
                input_t,
                c_scale,
                c_shift,
                residual=residual,
                scale_bias=c_scale_bias,
                shift_bias=c_shift_bias,
                epsilon=eps,
                **quant_kwargs,
            )

        def eager_fn(input_t, residual):
            r = residual.float() + input_t.float()
            n = torch.layer_norm(r, [hidden_dim], eps=eps)
            n = n * (1 + c_scale.float() + c_scale_bias.float()) + (
                c_shift.float() + c_shift_bias.float()
            )
            return r.to(torch.bfloat16), n.to(torch.bfloat16)

    else:
        raise ValueError(f"Unknown DIT mode: {mode}")

    fused_times = bench_gpu_time(
        fused_fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        input_args=(input_t, residual),
    )
    fused_ms = float(np.median(fused_times))

    # Reference check: residual_out always, norm_out for bf16 output only.
    if args.refcheck:
        r_fused, n_fused = fused_fn(input_t, residual)
        r_eager, n_eager = eager_fn(input_t, residual)
        checks = [("residual_out", r_fused, r_eager)]
        if out_dtype == "bfloat16":
            checks.append(("norm_out", n_fused, n_eager))
        else:
            print(f"[WARNING] --refcheck skips norm_out for {out_dtype} output.")
        for name, fused, eager in checks:
            try:
                torch.testing.assert_close(
                    fused.float(), eager.float(), rtol=1.6e-2, atol=1e-5
                )
            except AssertionError as e:
                print(f"[ERROR] {name} mismatch: {e}")
                if not args.allow_output_mismatch:
                    raise

    num_elements = batch_size * num_rows * hidden_dim
    if out_dtype == "nvfp4":
        norm_bytes = num_elements // 2 + num_elements // 16  # FP4 + E4M3 scales
    elif out_dtype == "mxfp8":
        norm_bytes = num_elements + num_elements // 32  # FP8 + UE8M0 scales
    else:
        norm_bytes = num_elements * 2
    # input + residual read, residual_out + norm_out write
    total_bytes = num_elements * 2 * 3 + norm_bytes
    tb_per_sec = (total_bytes / 1e12) / (fused_ms / 1e3) if fused_ms > 0 else 0

    res = []
    cur_res = defaultdict(str)
    cur_res["routine"] = args.routine
    cur_res["median_time"] = fused_ms
    cur_res["std_time"] = float(np.std(fused_times))
    cur_res["tflops"] = 0.0
    cur_res["tb_per_sec"] = tb_per_sec
    cur_res["batch_size"] = batch_size
    cur_res["hidden_size"] = hidden_dim
    cur_res["input_dtype"] = str(torch.bfloat16)
    if args.out_dtype_given:
        cur_res["out_dtype"] = _csv_out_dtype(out_dtype)
    cur_res["dit_mode"] = mode
    cur_res["eps"] = eps
    cur_res["backend"] = "cuda"
    cur_res["case_tag"] = args.case_tag
    print_perf_metrics("cuda", fused_ms, float(np.std(fused_times)), 0.0, tb_per_sec)

    if args.verbose >= 1 and out_dtype == "bfloat16":
        eager_times = bench_gpu_time(
            eager_fn,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(input_t, residual),
        )
        eager_ms = float(np.median(eager_times))
        speedup = eager_ms / fused_ms if fused_ms > 0 else 0
        print(
            f"  Eager: {eager_ms:.4f} ms | Fused: {fused_ms:.4f} ms | Speedup: {speedup:.2f}x"
        )

    res.append(cur_res)
    return res


def _qk_rmsnorm_rope_reference(
    qkv, q_weight, k_weight, num_heads, head_dim, rope_dims, ppf, pph, ppw, eps, base
):
    """fp32 reference of fused_qk_rmsnorm_rope with interleaved 3D RoPE.

    RMSNorm runs across all heads of Q (and K); head_dim is split into
    frame/height/width channel groups, each rotated by its own position with
    inv_freq = base^(-2i/dim) (as tests/norm/test_fused_qk_rmsnorm_rope.py).
    """
    hidden_size = num_heads * head_dim
    batch_size, seq_len = qkv.shape[:2]
    grids = torch.meshgrid(
        torch.arange(ppf, device=qkv.device),
        torch.arange(pph, device=qkv.device),
        torch.arange(ppw, device=qkv.device),
        indexing="ij",
    )
    angles = []
    for pos, dim in zip(grids, rope_dims, strict=True):
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, device=qkv.device, dtype=torch.float64) / dim)
        )
        angles.append(pos.reshape(-1, 1).double() * inv_freq)
    angle = torch.cat(angles, dim=-1).view(1, seq_len, 1, head_dim // 2)
    cos, sin = angle.cos().float(), angle.sin().float()

    def norm_rope(x, weight):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    q_ref = norm_rope(qkv[..., :hidden_size], q_weight)
    k_ref = norm_rope(qkv[..., hidden_size : 2 * hidden_size], k_weight)
    v_ref = qkv[..., 2 * hidden_size :].float().view(q_ref.shape)
    return q_ref, k_ref, v_ref


def testFusedQkRmsnormRope(args):
    """
    Test fused QK RMSNorm + 3D RoPE + V copy API.

    Benchmarks the fused kernel for video generation DIT self-attention
    (e.g. WAN 2.1/2.2). Output is bf16 unless --out_dtype fp8_e4m3 selects FP8
    Q/K/V output (output_fp8=True, unit quant scales).

    Args:
        args: Parsed command line arguments

    Returns:
        list: List of dicts containing performance results
    """
    from flashinfer.diffusion_ops import fused_qk_rmsnorm_rope

    if args.verbose >= 1:
        print("[INFO] Running testFusedQkRmsnormRope")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)

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
    out_dtype = _out_dtype_or_bf16(args)
    if out_dtype not in ("bfloat16", "fp8_e4m3"):
        raise ValueError(
            f"fused_qk_rmsnorm_rope supports --out_dtype bfloat16 or fp8_e4m3, got {out_dtype}"
        )
    output_fp8 = out_dtype == "fp8_e4m3"

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
        print(f"[VVERBOSE] {out_dtype = }")

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
        output_fp8=output_fp8,
    )

    def run_fused(qkv=qkv):
        return fused_qk_rmsnorm_rope(qkv, q_weight, k_weight, **kwargs)

    if run_refcheck:
        refs = _qk_rmsnorm_rope_reference(
            qkv,
            q_weight,
            k_weight,
            num_heads,
            head_dim,
            (t_dim, h_dim, w_dim),
            ppf,
            pph,
            ppw,
            eps,
            kwargs["base"],
        )
        for name, ref, out in zip(("q", "k", "v"), refs, run_fused(), strict=True):
            if output_fp8:
                num_bad, num_elements, pct, _ = _fp8_code_stats(
                    ref.to(torch.float8_e4m3fn), out
                )
                detail = "differ by more than one fp8 code"
            else:
                num_bad, num_elements, pct = is_close_stats(
                    ref.to(input_dtype), out, rtol=2e-2, atol=2e-2
                )
                detail = "differ"
            if num_bad > 0:
                print(
                    f"[ERROR] {name}_out mismatch: {num_bad}/{num_elements} ({pct:.2f}%) elements {detail}"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] {name}_out mismatch with {num_bad} elements"
                    )

    backend_times = bench_gpu_time(
        fn=run_fused,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        input_args=(qkv,),
    )

    if len(backend_times) > 0:
        median_time = np.median(backend_times)
        std_time = np.std(backend_times)

        # Memory bandwidth: read QKV + Q/K weights, write Q + K + V
        num_tokens = batch_size * seq_len
        out_itemsize = 1 if output_fp8 else input_dtype.itemsize
        problem_bytes = (
            num_tokens * 3 * hidden_size * input_dtype.itemsize  # QKV read
            + 2 * hidden_size * input_dtype.itemsize  # Q/K weight read
            + num_tokens * 3 * hidden_size * out_itemsize  # Q+K+V write
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
            if args.out_dtype_given:
                cur_res["out_dtype"] = _csv_out_dtype(out_dtype)
            cur_res["eps"] = eps
            cur_res["backend"] = "cuda"
            cur_res["ppf"] = ppf
            cur_res["pph"] = pph
            cur_res["ppw"] = ppw
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testFusedAddRmsnormFp8BlockQuant(args):
    """
    Test fused_add_rmsnorm_fp8_block_quant API.

    One kernel computes residual += input, normed = RMSNorm(residual) * weight
    (written as bf16/fp16 normed_out), and a 1x128 fp8 e4m3 block quantization of
    normed with fp32 block scales in the column-major layout deep_gemm consumes.
    hidden_size must be <= 16384 and a multiple of 256 (512 above 8192).
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedAddRmsnormFp8BlockQuant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    eps = args.eps
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )

    input_shape = (batch_size, hidden_size)
    num_blocks = hidden_size // 128
    padded_m = (batch_size + 3) // 4 * 4
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    residual_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    weight = torch.randn(hidden_size, dtype=input_dtype, device=device)

    def make_outputs():
        return (
            torch.empty(input_shape, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(num_blocks, padded_m, dtype=torch.float32, device=device),
            torch.empty(input_shape, dtype=input_dtype, device=device),
        )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] block_scale.shape = {(num_blocks, padded_m)}")

    def run_backend(
        backend, out, block_scale, normed_out, input_tensor, residual_tensor
    ):
        if backend == "cuda":
            flashinfer.fused_add_rmsnorm_fp8_block_quant(
                out,
                block_scale,
                normed_out,
                input_tensor,
                residual_tensor,
                weight,
                eps=eps,
                enable_pdl=enable_pdl,
            )
            return out, block_scale, normed_out, residual_tensor
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        if run_refcheck:
            residual_in = residual_tensor.clone()
            out, block_scale, normed_out, residual_out = run_backend(
                cur_backend, *make_outputs(), input_tensor, residual_in
            )
            x = residual_tensor.float() + input_tensor.float()
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
            reference_normed = (x / rms * weight.float()).to(input_dtype)
            # The kernel quantizes its own dtype-rounded normed_out, so the fp8 and
            # scale references are built from normed_out (scale = max(amax, 1e-4) / 448).
            blocks = normed_out.float().view(batch_size, num_blocks, 128)
            amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
            reference_out = (blocks * (448.0 / amax)).to(torch.float8_e4m3fn)
            reference_scale = (amax / 448.0).squeeze(-1)
            checks = [
                ("residual", x.to(input_dtype), residual_out, 1e-2),
                ("normed_out", reference_normed, normed_out, 1e-2),
                ("block_scale", reference_scale, block_scale.t()[:batch_size], 1e-3),
            ]
            for name, ref, got, tol in checks:
                num_bad, num_elements, pct = is_close_stats(
                    ref, got, rtol=tol, atol=tol
                )
                if num_bad > 0:
                    print(
                        f"[ERROR] {name} mismatch from backend {cur_backend}: "
                        f"{num_bad}/{num_elements} ({pct:.2f}%) elements differ"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {cur_backend} {name} mismatch with {num_bad} elements"
                        )
            num_bad, num_elements, pct, num_ties = _fp8_code_stats(
                reference_out.view(input_shape), out
            )
            if num_ties > 0 and args.verbose >= 1:
                print(
                    f"[INFO] Backend {cur_backend}: {num_ties}/{num_elements} "
                    f"({num_ties / num_elements * 100:.2f}%) elements land on an adjacent fp8 code"
                )
            if num_bad > 0:
                print(
                    f"[ERROR] fp8 output mismatch from backend {cur_backend}: "
                    f"{num_bad}/{num_elements} ({pct:.2f}%) elements differ by more than one fp8 code"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {cur_backend} fp8 output mismatch with {num_bad} elements"
                    )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(
                cur_backend,
                *make_outputs(),
                input_tensor,
                residual_tensor.clone(),
            ),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            # Read input + residual + weight; write residual + normed_out + fp8 out + scales
            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements
                * input_dtype.itemsize
                * 4  # in/residual rd, residual/normed wr
                + hidden_size * input_dtype.itemsize  # weight read
                + num_elements  # fp8 output write
                + num_blocks * padded_m * 4  # block scale write
            )
            problem_flops = num_elements * 8  # add + rmsnorm + amax + quant
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
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = str(torch.float8_e4m3fn)
                cur_res["eps"] = eps
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testLayernorm(args):
    """
    Test layernorm API (bf16 input and output, fp32 gamma/beta).

    Like rmsnorm it has no backend= argument: CuTe-DSL kernels by default, CUDA
    JIT fallback (recorded in the resolved_backend CSV column).
    """
    if args.verbose >= 1:
        print("[INFO] Running testLayernorm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = args.backends[:]
    # Default backend to cute-dsl, as for the rmsnorm variants
    if backends == ["cuda"]:
        backends = ["cute-dsl"]

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    eps = args.eps
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. layernorm only supports bfloat16."
        )

    input_shape = (batch_size, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)
    gamma = torch.randn(hidden_size, dtype=torch.float32, device=device)
    beta = torch.randn(hidden_size, dtype=torch.float32, device=device)

    def run_backend(backend, input_tensor):
        if backend == "cute-dsl":
            return flashinfer.layernorm(input_tensor, gamma, beta, eps=eps)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        if run_refcheck:
            reference_output = torch.layer_norm(
                input_tensor.float(), [hidden_size], gamma, beta, eps
            ).to(input_dtype)
            out = run_backend(cur_backend, input_tensor)
            num_bad, num_elements, pct = is_close_stats(
                reference_output, out, rtol=1e-2, atol=1e-2
            )
            if num_bad > 0:
                print(
                    f"[ERROR] Output tensor mismatch from backend {cur_backend}: "
                    f"{num_bad}/{num_elements} ({pct:.2f}%) elements differ"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {cur_backend} output mismatch with {num_bad} elements"
                    )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            num_elements = np.prod(input_shape)
            problem_bytes = (
                num_elements * input_dtype.itemsize * 2  # input read + output write
                + 2 * hidden_size * 4  # fp32 gamma + beta read
            )
            problem_flops = num_elements * 7  # mean, var, normalize, affine
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
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = str(input_dtype)
                cur_res["eps"] = eps
                cur_res["backend"] = backend
                cur_res["resolved_backend"] = _resolved_norm_backend(device)
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testActAndMul(args):
    """
    Test the gated activation APIs silu_and_mul / gelu_and_mul / gelu_tanh_and_mul.

    act(input[:, :hidden_size]) * input[:, hidden_size:], input shape
    (batch_size, 2 * hidden_size), output shape (batch_size, hidden_size).
    """
    if args.verbose >= 1:
        print("[INFO] Running testActAndMul")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    act_type = args.act_type
    enable_pdl = args.enable_pdl
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported input dtype: {args.input_dtype}. Supported dtypes are bfloat16, float16."
        )

    act_fn, reference_act = {
        "silu": (flashinfer.silu_and_mul, torch.nn.functional.silu),
        "gelu": (flashinfer.gelu_and_mul, torch.nn.functional.gelu),
        "gelu_tanh": (
            flashinfer.gelu_tanh_and_mul,
            lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        ),
    }[act_type]

    input_tensor = torch.randn(
        batch_size, 2 * hidden_size, dtype=input_dtype, device=device
    )
    out = torch.empty(batch_size, hidden_size, dtype=input_dtype, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {act_type = }")

    def run_backend(backend, input_tensor, out):
        if backend == "cuda":
            return act_fn(input_tensor, out=out, enable_pdl=enable_pdl)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    backend_times = {backend: [] for backend in backends}
    for cur_backend in backends:
        if run_refcheck:
            x = input_tensor.float()
            reference_output = (
                reference_act(x[:, :hidden_size]) * x[:, hidden_size:]
            ).to(input_dtype)
            test_out = run_backend(cur_backend, input_tensor, out.clone())
            num_bad, num_elements, pct = is_close_stats(
                reference_output, test_out, rtol=1e-2, atol=1e-2
            )
            if num_bad > 0:
                print(
                    f"[ERROR] Output tensor mismatch from backend {cur_backend}: "
                    f"{num_bad}/{num_elements} ({pct:.2f}%) elements differ"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {cur_backend} output mismatch with {num_bad} elements"
                    )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(cur_backend, input_tensor, out),
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            num_out = batch_size * hidden_size
            problem_bytes = num_out * input_dtype.itemsize * 3  # 2x read + 1x write
            problem_flops = num_out * 5  # activation (~4) + multiply
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
                cur_res["input_dtype"] = str(input_dtype)
                cur_res["out_dtype"] = str(input_dtype)
                cur_res["act_type"] = act_type
                cur_res["enable_pdl"] = enable_pdl
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMhc(args):
    """
    Test the mHC (manifold-constrained hyper-connection, HC=4) APIs.

    --mhc_mode post: flashinfer.mhc.mhc_post (residual mixing).
    --mhc_mode pre_big_fuse: flashinfer.mhc.mhc_pre_big_fuse (Sinkhorn + RMSNorm
    from an external projection and sqrsum, --num_splits split-K partials).
    --mhc_mode pre_big_fuse_with_prenorm: computes sqrsum from the residual itself.
    --batch_size is the token count, --hidden_size the hidden size H.
    Inputs follow benchmarks/bench_mhc_post.py and bench_mhc_pre_big_fuse.py.
    """
    if args.verbose >= 1:
        print("[INFO] Running testMhc")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    tokens = args.batch_size
    hidden_size = args.hidden_size
    mode = args.mhc_mode
    num_splits = args.num_splits
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res
    if num_splits != 1 and mode != "pre_big_fuse":
        raise ValueError("--num_splits applies to --mhc_mode pre_big_fuse only")

    hc = 4
    mix = hc * (2 + hc)
    k = hc * hidden_size
    residual = (
        torch.randn(tokens, hc, hidden_size, dtype=torch.float32, device=device) * 0.01
    ).bfloat16()
    mhc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    mhc_base = torch.randn(mix, dtype=torch.float32, device=device) * 0.1
    split_shape = (tokens,) if num_splits == 1 else (num_splits, tokens)
    dot_mix = torch.randn(*split_shape, mix, dtype=torch.float32, device=device) * 0.01

    if mode == "post":
        x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        post_layer_mix = torch.randn(tokens, hc, 1, dtype=torch.float32, device=device)
        comb_res_mix = torch.randn(tokens, hc, hc, dtype=torch.float32, device=device)

        def run_fn(residual):
            return flashinfer.mhc.mhc_post(x, residual, post_layer_mix, comb_res_mix)

        # x + residual read, out write (all bf16), plus fp32 mixing coefficients
        problem_bytes = (
            tokens * hidden_size * 2 * (1 + 2 * hc) + tokens * (hc + hc * hc) * 4
        )
    elif mode == "pre_big_fuse":
        sqrsum = torch.rand(*split_shape, dtype=torch.float32, device=device) * (
            float(k) / num_splits
        )

        def run_fn(residual):
            return flashinfer.mhc.mhc_pre_big_fuse(
                dot_mix,
                sqrsum,
                residual,
                mhc_scale,
                mhc_base,
                k,
                num_splits=num_splits,
            )

        problem_bytes = num_splits * tokens * (mix + 1) * 4
    elif mode == "pre_big_fuse_with_prenorm":

        def run_fn(residual):
            return flashinfer.mhc.mhc_pre_big_fuse_with_prenorm(
                dot_mix, residual, mhc_scale, mhc_base
            )

        problem_bytes = tokens * mix * 4
    else:
        raise ValueError(f"Unknown mHC mode: {mode}")
    if mode != "post":
        # residual read + layer_input write (bf16), post_mix + comb_mix write (fp32)
        problem_bytes += (
            tokens * hidden_size * 2 * (hc + 1) + tokens * (hc + hc * hc) * 4
        )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {residual.shape = }")
        print(f"[VVERBOSE] {mode = } {num_splits = }")

    if run_refcheck and mode != "post":
        print(f"[WARNING] --refcheck is not supported for --mhc_mode {mode}.")
    elif run_refcheck:
        reference_output = (
            x.float().unsqueeze(1) * post_layer_mix
            + torch.einsum("toh,ton->tnh", residual.float(), comb_res_mix)
        ).bfloat16()
        num_bad, num_elements, pct = is_close_stats(
            reference_output, run_fn(residual), rtol=1e-2, atol=1e-2
        )
        if num_bad > 0:
            print(
                f"[ERROR] Output tensor mismatch: {num_bad}/{num_elements} ({pct:.2f}%) elements differ"
            )
            if not args.allow_output_mismatch:
                raise AssertionError(f"[ERROR] Output mismatch with {num_bad} elements")

    times = bench_gpu_time(
        fn=run_fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=args.use_cupti,
        use_cuda_graph=is_cuda_graph_compatible,
        input_args=(residual,),
    )

    if len(times) > 0:
        median_time = np.median(times)
        std_time = np.std(times)
        tb_per_sec = problem_bytes / (10**9 * median_time)

        print_perf_metrics("cuda", median_time, std_time, 0.0, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = 0.0
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["input_dtype"] = str(torch.bfloat16)
            cur_res["out_dtype"] = str(torch.bfloat16)
            cur_res["mhc_mode"] = mode
            cur_res["num_splits"] = num_splits
            cur_res["backend"] = "cuda"
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


_GATED_NORM_VARIANTS = {
    # routine -> (allowed --norm_variant values, default)
    "gated_residual_rmsnorm": (("weighted", "unweighted"), "weighted"),
    "gated_residual_next_rmsnorm": (
        ("tensor_alpha", "scalar_alpha", "post_weight"),
        "tensor_alpha",
    ),
    "rmsnorm_sigmoid_gate": (("default",), "default"),
}


def _rms_normalize(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def testGatedResidualNorm(args):
    """
    Test the gated residual RMSNorm family (CuTe-DSL kernels, hidden size 8192 /
    head dim 64; SM100 / SM103 / SM107 only).

    - gated_residual_rmsnorm: out = alpha * residual + beta * RMSNorm(branch [* w]).
      --norm_variant weighted (default) / unweighted (weight=None).
    - gated_residual_next_rmsnorm: the same blend, then RMSNorm of the new stream
      times next_weight (bf16 FFN input), in one launch. --norm_variant
      tensor_alpha (default) / scalar_alpha / post_weight.
    - rmsnorm_sigmoid_gate: in-place per-head RMSNorm(x) * sigmoid(gate) at head
      dim 64; --num_heads heads per token (--hidden_size must be 64).

    --batch_size is the row (token) count. branch is bf16, the residual stream fp32.
    The library also has a Cake route (backend="auto" prefers it); it is out of
    scope here, so the routine always calls backend="cute".
    """
    if args.verbose >= 1:
        print(f"[INFO] Running testGatedResidualNorm ({args.routine})")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    routine = args.routine
    backends = args.backends[:]
    if backends == ["cuda"]:
        backends = ["cute-dsl"]
    rows = args.batch_size
    hidden = args.hidden_size
    eps = args.eps
    enable_pdl = bool(args.enable_pdl)
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    allowed, default_variant = _GATED_NORM_VARIANTS[routine]
    variant = args.norm_variant or default_variant
    if variant not in allowed:
        raise ValueError(
            f"--norm_variant {variant} is not one of {allowed} for {routine}"
        )

    backends = filter_backends_by_compute_capability(backends, routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res
    if args.input_dtype != "bfloat16":
        raise ValueError(f"{routine} supports bfloat16 input only")

    bf16, f32 = torch.bfloat16, torch.float32
    if routine == "rmsnorm_sigmoid_gate":
        if hidden != 64:
            raise ValueError(
                "rmsnorm_sigmoid_gate: --hidden_size must be 64 (head dim)"
            )
        heads = args.num_heads or 1
        shape = (rows, heads, hidden) if args.num_heads else (rows, hidden)
        x0 = torch.randn(shape, dtype=bf16, device=device)
        gate = torch.randn(shape, dtype=bf16, device=device)
        x = x0.clone()

        def run_backend(backend, x):
            flashinfer.rmsnorm_sigmoid_gate(
                x, gate, eps, enable_pdl=enable_pdl, backend="cute"
            )
            return x

        def reference():
            xf = x0.float()
            return (_rms_normalize(xf, eps) * torch.sigmoid(gate.float())).to(bf16)

        input_args = (x,)
        n = x0.numel()
        problem_bytes = n * 2 * 3  # x read + gate read + x write (bf16)
        problem_flops = n * 8
        out_dtype = bf16
    else:
        if hidden != 8192:
            raise ValueError(f"{routine}: the kernel supports --hidden_size 8192 only")
        branch = torch.randn(rows, hidden, dtype=bf16, device=device)
        residual = torch.randn(rows, hidden, dtype=f32, device=device)
        alpha = torch.rand(hidden, dtype=f32, device=device) + 0.5
        beta = torch.rand(hidden, dtype=f32, device=device) + 0.5
        weight = torch.rand(hidden, dtype=bf16, device=device) + 0.5
        n = rows * hidden
        if routine == "gated_residual_rmsnorm":
            w = weight if variant == "weighted" else None
            out = torch.empty_like(residual)

            def run_backend(backend, branch):
                return flashinfer.gated_residual_rmsnorm(
                    branch,
                    residual,
                    w,
                    alpha,
                    beta,
                    eps,
                    out=out,
                    enable_pdl=enable_pdl,
                    backend="cute",
                )

            def reference():
                b = branch.float()
                if w is not None:
                    b = b * w.float()
                return alpha * residual + beta * _rms_normalize(b, eps)

            # branch bf16 + residual fp32 read, fp32 out write, gates
            problem_bytes = n * (2 + 4 + 4) + hidden * (
                4 * 2 + (2 if w is not None else 0)
            )
            problem_flops = n * 8
            out_dtype = f32
        else:
            alpha_arg = 0.75 if variant == "scalar_alpha" else alpha
            post_w = weight if variant == "post_weight" else None
            next_w = torch.rand(hidden, dtype=bf16, device=device) + 0.5
            res_out = torch.empty_like(residual)
            ffn_in = torch.empty_like(branch)

            def run_backend(backend, branch):
                return flashinfer.gated_residual_next_rmsnorm(
                    branch,
                    residual,
                    next_w,
                    alpha_arg,
                    beta,
                    eps,
                    eps,
                    residual_out=res_out,
                    ffn_input=ffn_in,
                    post_weight=post_w,
                    enable_pdl=enable_pdl,
                    backend="cute",
                )

            def reference():
                b = branch.float()
                if post_w is not None:
                    b = b * post_w.float()
                stream = alpha_arg * residual + beta * _rms_normalize(b, eps)
                return stream, (_rms_normalize(stream, eps) * next_w.float()).to(bf16)

            # branch bf16 + residual fp32 read; fp32 stream + bf16 ffn input write
            problem_bytes = n * (2 + 4 + 4 + 2) + hidden * 4 * 3
            problem_flops = n * 14
            out_dtype = bf16
        input_args = (branch,)

    if run_refcheck:
        got = run_backend(backends[0], *input_args)
        ref = reference()
        if routine == "rmsnorm_sigmoid_gate":
            pairs = [("x", got, ref)]
        elif routine == "gated_residual_rmsnorm":
            pairs = [("out", got, ref)]
        else:
            pairs = [("residual_out", got[0], ref[0]), ("ffn_input", got[1], ref[1])]
        for name, g, r in pairs:
            num_bad, num_elements, pct = is_close_stats(r, g, rtol=2e-2, atol=2e-2)
            if num_bad > 0:
                print(
                    f"[ERROR] {routine} {name} mismatch vs torch reference: "
                    f"{num_bad}/{num_elements} ({pct:.2f}%) elements differ"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(f"[ERROR] {routine} {name} mismatch")
        if routine == "rmsnorm_sigmoid_gate":
            x.copy_(x0)

    for backend in backends:
        times = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            input_args=(backend, *input_args),
        )
        if len(times) == 0:
            continue
        median_time = np.median(times)
        std_time = np.std(times)
        tflops = problem_flops / (10**9 * median_time)
        tb_per_sec = problem_bytes / (10**9 * median_time)
        print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["input_dtype"] = str(bf16)
            cur_res["out_dtype"] = str(out_dtype)
            cur_res["eps"] = eps
            cur_res["norm_variant"] = variant
            cur_res["enable_pdl"] = enable_pdl
            cur_res["backend"] = backend
            cur_res["resolved_backend"] = "cute"
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res
